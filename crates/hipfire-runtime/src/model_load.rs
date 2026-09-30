//! Arch-generic whole-model weight orchestration (Tier-2). Sequences
//! embed → final-norm → output → per-device layer loop over a `WeightSource`,
//! whose impls own the format-specific reads (HFQ vs ParoQuant) and bake their
//! own config. Per-arch crates wrap the returned `LoadedWeights<L>` into their
//! own weights struct. Complements `weight_backend::WeightBackend` (Tier-3,
//! per-tensor dequant), which `WeightSource::read_layer` calls internally.

use crate::device_mesh::{DeviceMesh, DimKind};
use crate::llama::{EmbeddingFormat, WeightTensor};
use crate::multi_gpu::Gpus;
use hip_bridge::HipResult;
use rdna_compute::{Gpu, GpuTensor};

/// Where one layer's weights physically live. Resolved once per load and fixed
/// for the model's lifetime.
///
/// `Device` is VRAM. `HostMapped` is `hipHostMalloc(hipHostMallocMapped)` system
/// RAM that the kernels dereference through a device-visible alias — the same
/// bytes, read over PCIe. Every upload seam takes this so a spilled layer can
/// never silently land in the VRAM the spill exists to free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    Device,
    HostMapped,
}

/// What one load actually allocated, split by where it landed.
///
/// Measured from the tensors the source produced, so the loader's own accounting
/// is what admission charges. This is the honest replacement for deriving a
/// footprint from the file size or from tensor-name parsing — an offloaded layer's
/// bytes are host-pinned, and charging them to VRAM would deny the context the
/// spill exists to afford.
///
/// Scope, stated because the numbers are used for admission: these are *tensor*
/// bytes. They exclude the allocator's own granularity and padding, which on the
/// 9B fixture measured ~300 MB (~6%) above `device_bytes`, and they exclude the
/// 1 MiB host-tail pad each host-mapped tensor carries. A caller that needs the
/// real VRAM commitment asks the device (`hipMemGetInfo`), which is what the serve
/// preflight does.
///
/// The paged-experts route is the one deliberate exclusion: its expert buffers
/// belong to the `WeightPager`, which accounts for them itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadStats {
    /// Weight bytes owned in device memory (VRAM) for the model's lifetime.
    pub device_bytes: u64,
    /// Weight bytes owned as pinned host memory (`hipHostMalloc`) for the model's
    /// lifetime. Unreclaimable while the model is loaded, which is why the host
    /// admission tier charges it.
    pub host_pinned_bytes: u64,
}

impl LoadStats {
    /// Accumulate another `(device, host)` pair.
    pub fn add(&mut self, (device, host): (u64, u64)) {
        self.device_bytes += device;
        self.host_pinned_bytes += host;
    }
}

/// Sum `tensors` into `(device, pinned-host)` bytes, splitting on each tensor's own
/// allocation tag. Used by `WeightSource::layer_bytes` impls, which are then a list
/// of the layer's tensors rather than per-tensor arithmetic.
pub fn split_tensor_bytes<'a, I>(tensors: I) -> (u64, u64)
where
    I: IntoIterator<Item = &'a GpuTensor>,
{
    let mut device = 0u64;
    let mut host = 0u64;
    for tensor in tensors {
        let bytes = tensor.byte_size() as u64;
        if tensor.buf.is_host_mapped() {
            host += bytes;
        } else {
            device += bytes;
        }
    }
    (device, host)
}

/// Accumulate one `(device, host)` pair into another.
pub fn add_bytes((device, host): (u64, u64), other: (u64, u64)) -> (u64, u64) {
    (device + other.0, host + other.1)
}

/// Where each piece of the model lands across a device slice. `single` = the
/// n==1 degenerate case (everything on device 0). Moved verbatim from
/// `hipfire-arch-qwen35::qwen35::Layout` — arch-agnostic (depends only on `Gpus`).
pub struct Layout {
    output_device: usize,
    layer_to_device: Vec<usize>,
    /// Resolved placement per layer index. Always the same length as
    /// `layer_to_device`; `validate` enforces it. Fixed for the load.
    layer_residency: Vec<Residency>,
}
impl Layout {
    pub fn single(n_layers: usize) -> Self {
        Self {
            output_device: 0,
            layer_to_device: vec![0; n_layers],
            layer_residency: vec![Residency::Device; n_layers],
        }
    }
    pub fn from_gpus(g: &Gpus, n_layers: usize) -> Self {
        Self {
            output_device: g.output_device,
            layer_to_device: (0..n_layers).map(|i| g.device_for_layer(i)).collect(),
            layer_residency: vec![Residency::Device; n_layers],
        }
    }

    /// Build the canonical stage/rank-0 view from an admitted mesh. The
    /// manifest planner owns the full stage grid; this legacy loader view
    /// selects rank zero for each layer so existing orchestrators continue to
    /// have one deterministic device index until their typed mesh path lands.
    ///
    /// Coordinates derive from the mesh itself on an admitted mesh, so the
    /// fallible coordinate lookups cannot fail here.
    pub fn from_mesh(mesh: &DeviceMesh, n_layers: usize) -> Self {
        let mut output_coord = mesh
            .coord_of(0)
            .expect("device-mesh coordinate 0 exists on an admitted mesh");
        if let Some(index) = mesh.axes().iter().position(|axis| axis.kind == DimKind::Pp) {
            output_coord[index] = mesh.size_of(DimKind::Pp).saturating_sub(1);
        }
        let layer_to_device = (0..n_layers)
            .map(|layer| {
                let mut coord = mesh
                    .coord_of(0)
                    .expect("device-mesh coordinate 0 exists on an admitted mesh");
                if let Some(index) = mesh.axes().iter().position(|axis| axis.kind == DimKind::Pp) {
                    coord[index] = mesh.stage_for_layer(layer, n_layers);
                }
                mesh.device_of(&coord)
                    .expect("stage coordinate is in bounds on an admitted mesh")
            })
            .collect();
        Self {
            output_device: mesh
                .device_of(&output_coord)
                .expect("output stage coordinate is in bounds on an admitted mesh"),
            layer_to_device,
            layer_residency: vec![Residency::Device; n_layers],
        }
    }

    /// Validate the pure layout before any source preparation or GPU upload.
    pub fn validate(&self, n_devices: usize, n_layers: usize) -> Result<(), String> {
        if self.output_device >= n_devices {
            return Err(format!(
                "layout output device {} outside device count {}",
                self.output_device, n_devices
            ));
        }
        if self.layer_to_device.len() != n_layers {
            return Err(format!(
                "layout has {} layer assignments, expected {n_layers}",
                self.layer_to_device.len()
            ));
        }
        if self.layer_residency.len() != n_layers {
            return Err(format!(
                "layout has {} layer placements, expected {n_layers}",
                self.layer_residency.len()
            ));
        }
        if let Some((layer, &device)) = self
            .layer_to_device
            .iter()
            .enumerate()
            .find(|(_, &device)| device >= n_devices)
        {
            return Err(format!(
                "layout layer {layer} device {device} outside device count {n_devices}"
            ));
        }
        Ok(())
    }
    pub fn device_for_layer(&self, i: usize) -> usize {
        self.layer_to_device[i]
    }
    pub fn output_device(&self) -> usize {
        self.output_device
    }

    /// Pure split arithmetic for the resident-tail budget: layers `[0 .. spilled)`
    /// spill, `[spilled .. n_layers)` stay on the GPU. `requested_resident` counts
    /// layers left ON the GPU, so an unset budget (`None`) spills nothing and an
    /// over-large request saturates to nothing spilled.
    ///
    /// This is the whole placement policy: the engine never measures the device
    /// or probes a layer's size. Whatever the caller resolves here is what every
    /// arch loads, which is why the same `memory.gpu_layer_budget` means the same
    /// thing for llama, qwen2 and qwen35.
    pub fn spill_count(n_layers: usize, requested_resident: Option<usize>) -> usize {
        requested_resident.map_or(0, |resident| n_layers.saturating_sub(resident))
    }

    /// Resolved placement for one layer. Fixed for the load: nothing re-decides
    /// placement per request.
    pub fn residency_for_layer(&self, i: usize) -> Residency {
        self.layer_residency[i]
    }

    /// Whether this load spills anything at all — the gate for the fail-closed
    /// capability checks (a source or device that cannot honour a spill must
    /// refuse the load before it allocates, not half-apply it).
    pub fn spills_anything(&self) -> bool {
        self.layer_residency
            .iter()
            .any(|r| *r == Residency::HostMapped)
    }

    /// Resolve this layout's placement from the configured budget, mutating it in
    /// place, and return the number of spilled layers plus the report line to
    /// print (`None` for "print nothing": the unset default).
    ///
    /// Infallible by construction — the budget is pure arithmetic on the layer
    /// count — and silent only when nothing was configured, so a stock-vs-branch
    /// log diff stays readable.
    pub fn resolve_residency(
        &mut self,
        requested_resident: Option<usize>,
    ) -> (usize, Option<String>) {
        let n_layers = self.layer_residency.len();
        let spilled = Self::spill_count(n_layers, requested_resident);
        for residency in self.layer_residency.iter_mut().take(spilled) {
            *residency = Residency::HostMapped;
        }
        let resident = n_layers - spilled;
        let report = match requested_resident {
            None => None,
            // Nothing spilled while something *was* configured: either the request
            // exceeds the layer count (a configured no-op worth saying) or it asked
            // for exactly every layer (worth saying only because it was explicit).
            Some(requested) if spilled == 0 => Some(if requested > n_layers {
                format!(
                    "  partial offload: gpu_layer_budget={requested} is more than this model's \
                     {n_layers} layers; keeping every layer on the GPU"
                )
            } else {
                format!("  partial offload: 0 offloaded ({n_layers} resident), i_gpu_start=0")
            }),
            Some(_) => Some(format!(
                "  partial offload: {resident} resident / {spilled} offloaded, i_gpu_start={spilled}"
            )),
        };
        (spilled, report)
    }
}

/// Neutral result of the orchestrator. Each arch assembles its own weights
/// struct from this (qwen35 adds `pager`).
pub struct LoadedWeights<L> {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensor,
    pub output: WeightTensor,
    pub layers: Vec<L>,
    /// True iff the tied lm_head aliases the embedding buffer on this
    /// single-device route; false means a separate output allocation exists.
    pub lm_head_aliases_embd: bool,
    /// What this load actually allocated, by destination. See [`LoadStats`].
    pub stats: LoadStats,
}

/// Whole-model weight source — the one place HFQ vs PaRo differs. Config is held
/// by the impl (not passed per-call) so the orchestrator stays config-agnostic.
/// `read_layer` reuses Tier-3 `load_layer<B: WeightBackend>` internally.
pub trait WeightSource {
    type Layer;
    fn n_layers(&self) -> usize;
    /// Pre-load hook. HFQ drops the mmap when n==1; PaRo rejects n>1; llama no-op.
    fn prepare(&mut self, n_devices: usize) -> HipResult<()>;
    fn read_embed(&mut self, gpu: &mut Gpu) -> HipResult<(GpuTensor, EmbeddingFormat)>;
    fn read_final_norm(&mut self, gpu: &mut Gpu) -> HipResult<GpuTensor>;
    /// `can_alias` is true iff embed and output share a device (n==1); the impl
    /// decides whether to use it (qwen35 aliases; llama ignores it and reuploads).
    fn read_output(
        &mut self,
        gpu: &mut Gpu,
        embd: &GpuTensor,
        embd_fmt: EmbeddingFormat,
        can_alias: bool,
    ) -> HipResult<(WeightTensor, bool)>;
    fn read_layer(
        &mut self,
        gpu: &mut Gpu,
        layer_idx: usize,
        residency: Residency,
    ) -> HipResult<Self::Layer>;
    /// Why this source cannot spill, or `None` if it can.
    ///
    /// The loader fails the load when a budget is set and this returns `Some`, so
    /// a source that cannot honour a spill refuses it up front instead of
    /// half-applying it (host-locating the layers it can see while a
    /// source-internal allocation stays in VRAM).
    fn spill_refusal(&self) -> Option<String>;
    /// `(device bytes, pinned-host bytes)` owned by one loaded layer, for the
    /// loader's own accounting ([`LoadStats`]). Split on the tensor's own
    /// allocation tag, so a spilled layer reports its bytes as host-pinned and a
    /// resident one as device.
    fn layer_bytes(&self, layer: &Self::Layer) -> (u64, u64);
    /// Release one successfully loaded layer during whole-model rollback.
    ///
    /// Layer ownership is architecture-specific (Qwen3.5 carries MoE
    /// pointer tables and shared Paro sidecars), so the source supplies the
    /// exact teardown instead of relying on a generic `Drop`.
    fn free_layer(&mut self, gpu: &mut Gpu, layer: Self::Layer);
}

/// Deterministic failure points for the staged load transaction.
///
/// Typed test seam, never environment-controlled: the production route always
/// runs with `fault = None`. Mirrors `DeepseekV4DsparkFault` for the DSpark
/// sidecar loader — each variant fires AFTER the named owner is published to
/// staging, so the existing reverse-order rollback must reclaim it.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagedLoadFault {
    /// Fail after the embedding is published (rollback frees the embedding).
    AfterEmbed,
    /// Fail after the final norm is published (frees norm + embedding).
    AfterFinalNorm,
    /// Fail after the output/lm_head is published (frees output + norm + embedding).
    AfterOutput,
    /// Fail after layer `usize` is published (frees that layer and everything staged before it).
    AfterLayer(usize),
}

/// Resource-neutral operations used by the staged load transaction.
///
/// Keeping the transaction separate from HIP resource types gives the CPU
/// contract tests a deterministic allocator/source seam. The production
/// adapter below is the only implementation that knows how to free a
/// `GpuTensor` or `WeightTensor`; the ordering and ownership rules are shared
/// by both paths.
trait StagedLoadOps {
    type Layer;
    type Embedding;
    type Norm;
    type Output;
    type Error;

    fn prepare(&mut self, n_devices: usize) -> Result<(), Self::Error>;
    fn n_layers(&self) -> usize;
    fn read_embed(&mut self) -> Result<(Self::Embedding, EmbeddingFormat), Self::Error>;
    fn read_final_norm(&mut self) -> Result<Self::Norm, Self::Error>;
    fn read_output(
        &mut self,
        embedding: &Self::Embedding,
        format: EmbeddingFormat,
        can_alias: bool,
    ) -> Result<(Self::Output, bool), Self::Error>;
    fn read_layer(&mut self, layer_idx: usize) -> Result<Self::Layer, Self::Error>;
    fn free_layer(&mut self, layer_idx: usize, layer: Self::Layer);
    fn free_output(&mut self, output: Self::Output, aliases_embedding: bool);
    fn free_final_norm(&mut self, norm: Self::Norm);
    fn free_embed(&mut self, embedding: Self::Embedding);
    /// Build the injected-fault error for [`StagedLoadFault`]. Called only by
    /// the with-fault seam when it runs with `Some`; the production path
    /// (`None`) never invokes it, so its wording is test-only.
    fn injected_error(fault: StagedLoadFault) -> Self::Error;
}

struct StagedWeights<E, N, O, L> {
    token_embd: E,
    embd_format: EmbeddingFormat,
    output_norm: N,
    output: O,
    layers: Vec<L>,
    lm_head_aliases_embd: bool,
}

/// Execute the common embed → norm → output → layer transaction.
///
/// Every successful publication is retained until the transaction commits.
/// Any error drains completed layers in reverse publication order, then output,
/// final norm, and embedding. This is deliberately generic so a CPU test source
/// can observe the exact order without initializing HIP.
fn run_staged_load<O: StagedLoadOps>(
    ops: &mut O,
    n_devices: usize,
    can_alias: bool,
) -> Result<StagedWeights<O::Embedding, O::Norm, O::Output, O::Layer>, O::Error> {
    run_staged_load_with_fault(ops, n_devices, can_alias, None)
}

/// [`run_staged_load`] with a deterministic post-publication fault.
///
/// Each `Some` fault fires AFTER the named owner is admitted to staging, so a
/// failing run exercises the exact reverse-order rollback the production path
/// relies on. Crates drive this through [`load_weights_with_fault`];
/// `StagedLoadOps` is crate-private, so this runner stays private too.
fn run_staged_load_with_fault<O: StagedLoadOps>(
    ops: &mut O,
    n_devices: usize,
    can_alias: bool,
    fault: Option<StagedLoadFault>,
) -> Result<StagedWeights<O::Embedding, O::Norm, O::Output, O::Layer>, O::Error> {
    ops.prepare(n_devices)?;
    let mut staged_embedding = None;
    let mut staged_norm = None;
    let mut staged_output = None;
    let mut staged_layers = Vec::with_capacity(ops.n_layers());

    let result = (|| {
        let (embedding, format) = ops.read_embed()?;
        staged_embedding = Some((embedding, format));
        if fault == Some(StagedLoadFault::AfterEmbed) {
            return Err(O::injected_error(StagedLoadFault::AfterEmbed));
        }

        let norm = ops.read_final_norm()?;
        staged_norm = Some(norm);
        if fault == Some(StagedLoadFault::AfterFinalNorm) {
            return Err(O::injected_error(StagedLoadFault::AfterFinalNorm));
        }

        let (output, aliases_embedding) = ops.read_output(
            &staged_embedding
                .as_ref()
                .expect("embedding staged before output")
                .0,
            staged_embedding
                .as_ref()
                .expect("embedding staged before output")
                .1,
            can_alias,
        )?;
        staged_output = Some((output, aliases_embedding));
        if fault == Some(StagedLoadFault::AfterOutput) {
            return Err(O::injected_error(StagedLoadFault::AfterOutput));
        }

        for layer_idx in 0..ops.n_layers() {
            staged_layers.push(ops.read_layer(layer_idx)?);
            if fault == Some(StagedLoadFault::AfterLayer(layer_idx)) {
                return Err(O::injected_error(StagedLoadFault::AfterLayer(layer_idx)));
            }
        }

        let (token_embd, embd_format) = staged_embedding.take().expect("embedding staged");
        let output_norm = staged_norm.take().expect("output norm staged");
        let (output, lm_head_aliases_embd) = staged_output.take().expect("output staged");
        Ok(StagedWeights {
            token_embd,
            embd_format,
            output_norm,
            output,
            layers: std::mem::take(&mut staged_layers),
            lm_head_aliases_embd,
        })
    })();

    if result.is_err() {
        for (layer_idx, layer) in staged_layers.drain(..).enumerate().rev() {
            ops.free_layer(layer_idx, layer);
        }
        if let Some((output, aliases_embedding)) = staged_output.take() {
            ops.free_output(output, aliases_embedding);
        }
        if let Some(norm) = staged_norm.take() {
            ops.free_final_norm(norm);
        }
        if let Some((embedding, _)) = staged_embedding.take() {
            ops.free_embed(embedding);
        }
    }
    result
}

struct GpuStagedLoadOps<'a, S> {
    source: &'a mut S,
    devices: &'a mut [Gpu],
    layout: &'a Layout,
}

impl<S: WeightSource> StagedLoadOps for GpuStagedLoadOps<'_, S> {
    type Layer = S::Layer;
    type Embedding = GpuTensor;
    type Norm = GpuTensor;
    type Output = WeightTensor;
    type Error = hip_bridge::HipError;

    fn prepare(&mut self, n_devices: usize) -> Result<(), Self::Error> {
        self.source.prepare(n_devices)
    }

    fn n_layers(&self) -> usize {
        self.source.n_layers()
    }

    fn read_embed(&mut self) -> Result<(Self::Embedding, EmbeddingFormat), Self::Error> {
        self.source.read_embed(&mut self.devices[0])
    }

    fn read_final_norm(&mut self) -> Result<Self::Norm, Self::Error> {
        let device = self.layout.output_device();
        self.source.read_final_norm(&mut self.devices[device])
    }

    fn read_output(
        &mut self,
        embedding: &Self::Embedding,
        format: EmbeddingFormat,
        can_alias: bool,
    ) -> Result<(Self::Output, bool), Self::Error> {
        let device = self.layout.output_device();
        self.source
            .read_output(&mut self.devices[device], embedding, format, can_alias)
    }

    fn read_layer(&mut self, layer_idx: usize) -> Result<Self::Layer, Self::Error> {
        let device = self.layout.device_for_layer(layer_idx);
        let residency = self.layout.residency_for_layer(layer_idx);
        self.source
            .read_layer(&mut self.devices[device], layer_idx, residency)
    }

    fn free_layer(&mut self, layer_idx: usize, layer: Self::Layer) {
        let device = self.layout.device_for_layer(layer_idx);
        self.source.free_layer(&mut self.devices[device], layer);
    }

    fn free_output(&mut self, output: Self::Output, aliases_embedding: bool) {
        let device = self.layout.output_device();
        if aliases_embedding {
            output.free_metadata_only(&mut self.devices[device]);
        } else {
            output.free_all(&mut self.devices[device]);
        }
    }

    fn free_final_norm(&mut self, norm: Self::Norm) {
        let device = self.layout.output_device();
        let _ = self.devices[device].free_tensor(norm);
    }

    fn free_embed(&mut self, embedding: Self::Embedding) {
        let _ = self.devices[0].free_tensor(embedding);
    }

    fn injected_error(fault: StagedLoadFault) -> Self::Error {
        hip_bridge::HipError::new(
            0,
            &format!("model_load: injected staged-load failure at {fault:?}"),
        )
    }
}

/// Whether a CPU-executed spill conflicts with a retained-replay backend. Pure, so
/// the conjunction is testable without a GPU or a process snapshot.
pub fn cpu_exec_offends_replay(
    spill_requested: bool,
    cpu_exec: bool,
    replay_enabled: bool,
) -> bool {
    spill_requested && cpu_exec && replay_enabled
}

/// Whether a load must be refused, and why, before a byte is allocated.
///
/// Two fail-closed rules, both about a spill that cannot be honoured:
///
/// * a source that cannot spill at all (`WeightSource::spill_refusal`), and
/// * a unified-memory device, where host-mapped and device memory come from one
///   pool so spilling frees no VRAM.
///
/// Plus the one combination that is not a placement problem but a route problem:
/// `memory.offload_exec=cpu` with a retained-replay backend. Redline's tape
/// records GPU launches and replays them, so a CPU-executed step would run while
/// the tape is built and then be *absent* from every replay — the replayed route
/// would compute from activations that step should have refreshed, and nothing in
/// the tape can express that. Refusing the load beats running a stale route.
///
/// Returns `Ok(())` when the load may proceed.
pub fn offload_load_refusal(
    spills_anything: bool,
    source_refusal: Option<String>,
    uma_device: Option<usize>,
    cpu_exec: bool,
    replay_enabled: bool,
) -> HipResult<()> {
    if !spills_anything {
        return Ok(());
    }
    if let Some(reason) = source_refusal {
        return Err(hip_bridge::HipError::new(
            0,
            &format!("memory.gpu_layer_budget is set but {reason}"),
        ));
    }
    if let Some(device) = uma_device {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "memory.gpu_layer_budget is set on a unified-memory device (device {device}): \
                 host-mapped and device memory come from the same pool, so spilling frees no \
                 VRAM. Unset the key"
            ),
        ));
    }
    if cpu_exec_offends_replay(true, cpu_exec, replay_enabled) {
        return Err(hip_bridge::HipError::new(
            0,
            "memory.offload_exec=cpu conflicts with the retained-replay (Redline) backend: the \
             replay tape records GPU launches and does not execute the CPU-executed steps, so a \
             replayed route would compute from stale activations. Set memory.offload_exec=pcie \
             (HIPFIRE_OFFLOAD_EXEC=pcie) or replay.backend=hip",
        ));
    }
    Ok(())
}

/// Drive a `WeightSource` across a device slice. Single shared copy of the
/// embed → norm → output → per-device layer loop.
///
/// `layout` is taken mutably because this is where placement is resolved: the
/// configured `memory.gpu_layer_budget` becomes a per-layer `Residency` here and
/// is then carried to every arch's source unchanged.
pub fn load_weights<S: WeightSource>(
    source: &mut S,
    devices: &mut [Gpu],
    layout: &mut Layout,
) -> HipResult<LoadedWeights<S::Layer>> {
    load_weights_inner(source, devices, layout, None)
}

/// Deterministic fault-injection seam over the production staged transaction.
///
/// The production route always calls [`load_weights`] (fault `None`), so a
/// fault is unreachable without this seam. Mirrors
/// `DeepseekV4::load_dspark_with_fault`: typed, never environment-controlled;
/// fixture tests fail a real load after one publication boundary and prove
/// rollback reclaims every staged owner.
#[doc(hidden)]
pub fn load_weights_with_fault<S: WeightSource>(
    source: &mut S,
    devices: &mut [Gpu],
    layout: &mut Layout,
    fault: StagedLoadFault,
) -> HipResult<LoadedWeights<S::Layer>> {
    load_weights_inner(source, devices, layout, Some(fault))
}

fn load_weights_inner<S: WeightSource>(
    source: &mut S,
    devices: &mut [Gpu],
    layout: &mut Layout,
    fault: Option<StagedLoadFault>,
) -> HipResult<LoadedWeights<S::Layer>> {
    let n_devices = devices.len();
    if n_devices == 0 {
        return Err(hip_bridge::HipError::new(
            0,
            "load_weights: at least one device is required",
        ));
    }
    layout
        .validate(n_devices, source.n_layers())
        .map_err(|reason| hip_bridge::HipError::new(0, &reason))?;
    // The one placement decision in the engine. Every arch reads it back through
    // `residency_for_layer`, so `memory.gpu_layer_budget` means the same thing
    // everywhere instead of being re-derived per loader.
    let (_spilled, report) = layout.resolve_residency(hipfire_config::memory::gpu_layer_budget());
    if let Some(line) = report {
        eprintln!("{line}");
    }
    // Fail closed before a single byte is allocated. A spill this source cannot
    // honour is a refused load, never a half-applied placement — and the check
    // runs before `prepare`/`read_embed` so the user sees the budget message
    // rather than whatever the source would have objected to first.
    offload_load_refusal(
        layout.spills_anything(),
        source.spill_refusal(),
        devices.iter().position(|device| device.is_uma()),
        hipfire_config::memory::offload_exec() == hipfire_config::memory::OffloadExec::Cpu,
        devices.iter().any(|device| device.replay.is_enabled()),
    )?;
    // Record the execution target for the whole device: the capture gate and the
    // dispatch CPU seam both read this instead of re-deriving it per step, so a
    // model whose steps include a CPU-executed weight cannot be captured or
    // replayed by accident. `false` on every load that spills nothing.
    let cpu_exec_weights = layout.spills_anything()
        && hipfire_config::memory::offload_exec() == hipfire_config::memory::OffloadExec::Cpu;
    for device in devices.iter_mut() {
        device.set_cpu_exec_weights(cpu_exec_weights);
    }
    let mut ops = GpuStagedLoadOps {
        source,
        devices,
        layout,
    };
    let staged = match fault {
        None => run_staged_load(&mut ops, n_devices, n_devices == 1)?,
        Some(fault) => {
            run_staged_load_with_fault(&mut ops, n_devices, n_devices == 1, Some(fault))?
        }
    };
    // Measured, not estimated: the bytes the source actually produced, split by
    // where each tensor landed. A tied output head accounts only its metadata,
    // because its buffer is the embedding's (the same split `free_output` uses).
    let mut stats = LoadStats::default();
    stats.add(split_tensor_bytes([
        &staged.token_embd,
        &staged.output_norm,
    ]));
    stats.add(if staged.lm_head_aliases_embd {
        staged.output.owned_metadata_bytes()
    } else {
        staged.output.owned_bytes()
    });
    for layer in &staged.layers {
        stats.add(source.layer_bytes(layer));
    }
    if hipfire_config::developer_var("HIPFIRE_OFFLOAD_DEBUG").is_ok() {
        eprintln!(
            "[offload-debug] load stats: device={} B, host_pinned={} B ({} layers, {_spilled} spilled)",
            stats.device_bytes,
            stats.host_pinned_bytes,
            layout.layer_residency.len(),
        );
    }
    Ok(LoadedWeights {
        token_embd: staged.token_embd,
        embd_format: staged.embd_format,
        output_norm: staged.output_norm,
        output: staged.output,
        layers: staged.layers,
        lm_head_aliases_embd: staged.lm_head_aliases_embd,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn single_layout_all_on_device_0() {
        let l = Layout::single(5);
        assert_eq!(l.output_device(), 0);
        for i in 0..5 {
            assert_eq!(l.device_for_layer(i), 0);
        }
    }

    #[test]
    fn mesh_layout_selects_stage_rank_zero_without_io() {
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2), (DimKind::Tp, 2)])
            .expect("small test mesh construction cannot overflow");
        let layout = Layout::from_mesh(&mesh, 4);
        assert_eq!(layout.output_device(), 2);
        assert_eq!(
            (0..4)
                .map(|layer| layout.device_for_layer(layer))
                .collect::<Vec<_>>(),
            vec![0, 0, 2, 2]
        );
        assert!(layout.validate(mesh.n_devices(), 4).is_ok());
    }

    #[test]
    fn invalid_layout_is_rejected_before_source_work() {
        let layout = Layout::single(2);
        assert!(layout.validate(0, 2).is_err());
        assert!(layout.validate(1, 3).is_err());
    }

    /// The whole placement policy, in one test: the split arithmetic, the
    /// resolved per-layer `Residency`, and the report line each case produces.
    /// `resident` counts layers LEFT ON the GPU, so `Some(3)` of 64 spills 61 —
    /// the contract the `memory.gpu_layer_budget` help text states.
    #[test]
    fn spill_count_is_the_whole_policy() {
        // Unset: nothing spilled, and silent — the only case allowed to print
        // nothing, because its absence is what keeps stock-vs-branch log diffs
        // readable.
        assert_eq!(Layout::spill_count(64, None), 0);
        let mut layout = Layout::single(64);
        let (spilled, report) = layout.resolve_residency(None);
        assert_eq!((spilled, report), (0, None));
        assert!(!layout.spills_anything());

        // Pinned resident count: exactly one boundary, everything before it spills.
        assert_eq!(Layout::spill_count(64, Some(3)), 61);
        let mut layout = Layout::single(64);
        let (spilled, report) = layout.resolve_residency(Some(3));
        assert_eq!(spilled, 61);
        assert_eq!(
            report.as_deref(),
            Some("  partial offload: 3 resident / 61 offloaded, i_gpu_start=61")
        );
        assert!(layout.spills_anything());
        for layer in 0..64 {
            let expected = if layer < 61 {
                Residency::HostMapped
            } else {
                Residency::Device
            };
            assert_eq!(layout.residency_for_layer(layer), expected, "layer {layer}");
        }

        // 0 resident spills everything; n resident spills nothing.
        assert_eq!(Layout::spill_count(64, Some(0)), 64);
        assert_eq!(Layout::spill_count(64, Some(64)), 0);

        // A request above the layer count saturates to fully resident and says so
        // rather than letting a configured budget do nothing in silence.
        let mut layout = Layout::single(64);
        let (spilled, report) = layout.resolve_residency(Some(200));
        assert_eq!(spilled, 0);
        assert_eq!(
            report.as_deref(),
            Some(
                "  partial offload: gpu_layer_budget=200 is more than this model's 64 layers; \
                 keeping every layer on the GPU"
            )
        );

        // Exactly every layer is also a no-op, but it was asked for explicitly, so
        // it is reported.
        let mut layout = Layout::single(64);
        let (_, report) = layout.resolve_residency(Some(64));
        assert_eq!(
            report.as_deref(),
            Some("  partial offload: 0 offloaded (64 resident), i_gpu_start=0")
        );

        // Degenerate models must not underflow.
        assert_eq!(Layout::spill_count(0, Some(0)), 0);
        assert_eq!(Layout::spill_count(1, Some(999)), 0);
    }

    /// `validate` must reject a placement vector that does not match the layer
    /// count: `residency_for_layer` indexes it directly, so a short vector would
    /// panic mid-load instead of failing the load.
    #[test]
    fn layout_rejects_a_mismatched_placement_vector() {
        let mut layout = Layout::single(2);
        layout.layer_residency.clear();
        let err = layout.validate(1, 2).unwrap_err();
        assert!(err.contains("placements"), "{err}");
    }

    /// The refusal conjunction is exact: each term alone must not refuse a load,
    /// or a plain `memory.offload_exec=pcie` run (whose `spills_anything` is true)
    /// would stop loading.
    #[test]
    fn offload_refusal_is_exactly_the_conjunction() {
        // No spill: nothing can be refused, whatever else is set.
        assert!(offload_load_refusal(false, None, None, true, true).is_ok());
        assert!(
            offload_load_refusal(
                false,
                Some("a source that cannot spill".into()),
                None,
                true,
                true
            )
            .is_ok(),
            "a refusal reason with no spill requested is not a refusal"
        );

        // A spill with a source that cannot honour it.
        let err = offload_load_refusal(
            true,
            Some("this source is PaRoQuant".into()),
            None,
            false,
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("this source is PaRoQuant"), "{err}");

        // A spill on unified memory.
        let err = offload_load_refusal(true, None, Some(0), false, false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unified-memory device"), "{err}");

        // A spill executed on the CPU with a retained-replay backend.
        assert!(cpu_exec_offends_replay(true, true, true));
        assert!(
            !cpu_exec_offends_replay(false, true, true),
            "nothing spilled"
        );
        assert!(!cpu_exec_offends_replay(true, false, true), "pcie is fine");
        assert!(!cpu_exec_offends_replay(true, true, false), "no replay");
        let err = offload_load_refusal(true, None, None, true, true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("retained-replay"), "{err}");

        // The clean cases: a spill on a dense source, CPU-executed, with no replay.
        assert!(offload_load_refusal(true, None, None, true, false).is_ok());
        assert!(offload_load_refusal(true, None, None, false, true).is_ok());
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum FailAt {
        Prepare,
        Embed,
        FinalNorm,
        Output,
        Layer(usize),
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Allocation {
        id: usize,
        kind: &'static str,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct OutputAllocation {
        primary: Option<Allocation>,
        metadata: Allocation,
    }

    #[derive(Debug, Default)]
    struct TestAllocator {
        next_id: usize,
        allocations: usize,
        reuses: usize,
        live: BTreeSet<usize>,
        free_ids: BTreeSet<usize>,
        frees: usize,
    }

    impl TestAllocator {
        fn alloc(&mut self, kind: &'static str) -> Allocation {
            let id = if let Some(id) = self.free_ids.pop_first() {
                self.reuses += 1;
                id
            } else {
                let id = self.next_id;
                self.next_id += 1;
                id
            };
            self.allocations += 1;
            assert!(
                self.live.insert(id),
                "test allocator id reused while live: {id}"
            );
            Allocation { id, kind }
        }

        fn free(&mut self, allocation: Allocation) {
            assert!(
                self.live.remove(&allocation.id),
                "double free of {}#{}",
                allocation.kind,
                allocation.id
            );
            assert!(
                self.free_ids.insert(allocation.id),
                "allocator released {}#{} twice",
                allocation.kind,
                allocation.id
            );
            self.frees += 1;
        }
    }

    /// CPU-only WeightSource seam: every resource is a tracked token, so a
    /// failure test can prove ownership transfer and exact cleanup without
    /// constructing `Gpu` or relying on a global `Drop` implementation.
    struct TestWeightSource {
        allocator: TestAllocator,
        n_layers: usize,
        fail_at: Option<FailAt>,
        alias_output: bool,
    }

    impl TestWeightSource {
        fn new(n_layers: usize, fail_at: Option<FailAt>) -> Self {
            Self {
                allocator: TestAllocator::default(),
                n_layers,
                fail_at,
                alias_output: false,
            }
        }

        fn fail(&self, at: FailAt) -> Result<(), String> {
            if self.fail_at == Some(at) {
                Err(format!("injected {at:?} failure"))
            } else {
                Ok(())
            }
        }

        fn assert_clean(&self) {
            assert!(
                self.allocator.live.is_empty(),
                "staged resources leaked: {:?}",
                self.allocator.live
            );
            assert_eq!(
                self.allocator.frees, self.allocator.allocations,
                "every allocation must be released exactly once"
            );
        }
    }

    impl StagedLoadOps for TestWeightSource {
        type Layer = Allocation;
        type Embedding = Allocation;
        type Norm = Allocation;
        type Output = OutputAllocation;
        type Error = String;

        fn prepare(&mut self, _n_devices: usize) -> Result<(), Self::Error> {
            self.fail(FailAt::Prepare)
        }

        fn n_layers(&self) -> usize {
            self.n_layers
        }

        fn read_embed(&mut self) -> Result<(Self::Embedding, EmbeddingFormat), Self::Error> {
            self.fail(FailAt::Embed)?;
            Ok((self.allocator.alloc("embedding"), EmbeddingFormat::F32))
        }

        fn read_final_norm(&mut self) -> Result<Self::Norm, Self::Error> {
            self.fail(FailAt::FinalNorm)?;
            Ok(self.allocator.alloc("final-norm"))
        }

        fn read_output(
            &mut self,
            _embedding: &Self::Embedding,
            _format: EmbeddingFormat,
            can_alias: bool,
        ) -> Result<(Self::Output, bool), Self::Error> {
            self.fail(FailAt::Output)?;
            let aliases_embedding = can_alias && self.alias_output;
            let primary = (!aliases_embedding).then(|| self.allocator.alloc("output"));
            let metadata = self.allocator.alloc("output-metadata");
            Ok((OutputAllocation { primary, metadata }, aliases_embedding))
        }

        fn read_layer(&mut self, layer_idx: usize) -> Result<Self::Layer, Self::Error> {
            self.fail(FailAt::Layer(layer_idx))?;
            Ok(self.allocator.alloc("layer"))
        }

        fn free_layer(&mut self, _layer_idx: usize, layer: Self::Layer) {
            self.allocator.free(layer);
        }

        fn free_output(&mut self, mut output: Self::Output, aliases_embedding: bool) {
            self.allocator.free(output.metadata);
            if !aliases_embedding {
                self.allocator
                    .free(output.primary.take().expect("owned output primary"));
            } else {
                assert!(
                    output.primary.is_none(),
                    "alias output must not own a second primary buffer"
                );
            }
        }

        fn free_final_norm(&mut self, norm: Self::Norm) {
            self.allocator.free(norm);
        }

        fn free_embed(&mut self, embedding: Self::Embedding) {
            self.allocator.free(embedding);
        }

        fn injected_error(fault: StagedLoadFault) -> Self::Error {
            format!("model_load: injected staged-load failure at {fault:?}")
        }
    }

    #[test]
    fn cpu_staged_load_sweep_rolls_back_at_every_boundary() {
        let failures = [
            FailAt::Prepare,
            FailAt::Embed,
            FailAt::FinalNorm,
            FailAt::Output,
            FailAt::Layer(0),
            FailAt::Layer(2),
        ];

        for failure in failures {
            let mut source = TestWeightSource::new(4, Some(failure));
            assert!(
                run_staged_load(&mut source, 2, false).is_err(),
                "{failure:?} must fail"
            );
            source.assert_clean();
        }
    }

    #[test]
    fn cpu_staged_load_with_fault_rolls_back_after_publication() {
        // The pre-publication sweep above fails each read before staging; the
        // seam instead fails AFTER the named owner is admitted, so rollback
        // must reclaim already-published owners (embed alone through the full
        // four-layer sweep).
        let faults = [
            StagedLoadFault::AfterEmbed,
            StagedLoadFault::AfterFinalNorm,
            StagedLoadFault::AfterOutput,
            StagedLoadFault::AfterLayer(0),
            StagedLoadFault::AfterLayer(3),
        ];
        for fault in faults {
            let mut source = TestWeightSource::new(4, None);
            let err = match run_staged_load_with_fault(&mut source, 2, false, Some(fault)) {
                Ok(_) => panic!("seam fault {fault:?} must fire after publication"),
                Err(err) => err,
            };
            assert!(
                err.contains("injected staged-load failure"),
                "{fault:?} error bypassed the fault seam: {err}"
            );
            source.assert_clean();
        }
    }

    #[test]
    fn cpu_production_path_has_no_fault_parameter() {
        // `run_staged_load` takes no fault argument (compiler-enforced by its
        // signature): the same source that fails under every seam fault loads
        // cleanly through the production entry, so the hook is unreachable
        // without the test seam. Each fault is followed by an immediate
        // production retry to prove the released owners are reusable.
        let faults = [
            StagedLoadFault::AfterEmbed,
            StagedLoadFault::AfterFinalNorm,
            StagedLoadFault::AfterOutput,
            StagedLoadFault::AfterLayer(0),
            StagedLoadFault::AfterLayer(3),
        ];
        for fault in faults {
            let mut source = TestWeightSource::new(4, None);
            assert!(
                run_staged_load_with_fault(&mut source, 2, false, Some(fault)).is_err(),
                "seam fault {fault:?} must fire"
            );
            source.assert_clean();
            let loaded =
                run_staged_load(&mut source, 2, false).expect("production retry after seam fault");
            // Non-aliased output owns primary + metadata: 1 embed + 1 norm +
            // 2 output + 4 layers live after a clean production load.
            assert_eq!(source.allocator.live.len(), 8);
            let StagedWeights {
                token_embd,
                output_norm,
                output,
                layers,
                lm_head_aliases_embd,
                ..
            } = loaded;
            for (layer_idx, layer) in layers.into_iter().enumerate().rev() {
                source.free_layer(layer_idx, layer);
            }
            source.free_output(output, lm_head_aliases_embd);
            source.free_final_norm(output_norm);
            source.free_embed(token_embd);
            source.assert_clean();
        }
    }

    #[test]
    fn cpu_failed_load_retry_reuses_released_allocations() {
        let mut source = TestWeightSource::new(3, Some(FailAt::Layer(2)));
        assert!(run_staged_load(&mut source, 1, true).is_err());
        source.assert_clean();
        assert_eq!(source.allocator.allocations, 6);
        assert_eq!(source.allocator.reuses, 0);
        assert_eq!(source.allocator.next_id, 6);

        source.fail_at = None;
        let loaded = run_staged_load(&mut source, 1, true).expect("immediate retry");
        assert_eq!(source.allocator.live.len(), 7);
        assert_eq!(source.allocator.allocations, 13);
        assert_eq!(source.allocator.reuses, 6);
        assert_eq!(source.allocator.next_id, 7);
        let StagedWeights {
            token_embd,
            output_norm,
            output,
            layers,
            lm_head_aliases_embd,
            ..
        } = loaded;
        for (layer_idx, layer) in layers.into_iter().enumerate().rev() {
            source.free_layer(layer_idx, layer);
        }
        source.free_output(output, lm_head_aliases_embd);
        source.free_final_norm(output_norm);
        source.free_embed(token_embd);
        source.assert_clean();

        let loaded = run_staged_load(&mut source, 1, true).expect("reload after retry");
        assert_eq!(source.allocator.live.len(), 7);
        assert_eq!(source.allocator.allocations, 20);
        assert_eq!(source.allocator.reuses, 13);
        assert_eq!(source.allocator.next_id, 7);
        let StagedWeights {
            token_embd,
            output_norm,
            output,
            layers,
            lm_head_aliases_embd,
            ..
        } = loaded;
        for (layer_idx, layer) in layers.into_iter().enumerate().rev() {
            source.free_layer(layer_idx, layer);
        }
        source.free_output(output, lm_head_aliases_embd);
        source.free_final_norm(output_norm);
        source.free_embed(token_embd);
        source.assert_clean();
    }

    #[test]
    fn cpu_staged_load_alias_has_one_embedding_owner() {
        let mut source = TestWeightSource::new(2, None);
        source.alias_output = true;
        let loaded = run_staged_load(&mut source, 1, true).expect("staged load");
        assert!(loaded.lm_head_aliases_embd);
        assert_eq!(source.allocator.live.len(), 5);
        assert_eq!(source.allocator.allocations, 5);

        let StagedWeights {
            token_embd,
            output_norm,
            output,
            layers,
            lm_head_aliases_embd,
            ..
        } = loaded;
        for (layer_idx, layer) in layers.into_iter().enumerate().rev() {
            source.free_layer(layer_idx, layer);
        }
        source.free_output(output, lm_head_aliases_embd);
        source.free_final_norm(output_norm);
        source.free_embed(token_embd);
        source.assert_clean();
        assert_eq!(source.allocator.frees, source.allocator.allocations);
    }
}
