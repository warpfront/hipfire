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

/// Where each piece of the model lands across a device slice. `single` = the
/// n==1 degenerate case (everything on device 0). Moved verbatim from
/// `hipfire-arch-qwen35::qwen35::Layout` — arch-agnostic (depends only on `Gpus`).
pub struct Layout {
    output_device: usize,
    layer_to_device: Vec<usize>,
}
impl Layout {
    pub fn single(n_layers: usize) -> Self {
        Self {
            output_device: 0,
            layer_to_device: vec![0; n_layers],
        }
    }
    pub fn from_gpus(g: &Gpus, n_layers: usize) -> Self {
        Self {
            output_device: g.output_device,
            layer_to_device: (0..n_layers).map(|i| g.device_for_layer(i)).collect(),
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
    fn read_layer(&mut self, gpu: &mut Gpu, layer_idx: usize) -> HipResult<Self::Layer>;
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
        self.source.read_layer(&mut self.devices[device], layer_idx)
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

/// Drive a `WeightSource` across a device slice. Single shared copy of the
/// embed → norm → output → per-device layer loop.
pub fn load_weights<S: WeightSource>(
    source: &mut S,
    devices: &mut [Gpu],
    layout: &Layout,
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
    layout: &Layout,
    fault: StagedLoadFault,
) -> HipResult<LoadedWeights<S::Layer>> {
    load_weights_inner(source, devices, layout, Some(fault))
}

fn load_weights_inner<S: WeightSource>(
    source: &mut S,
    devices: &mut [Gpu],
    layout: &Layout,
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
    let mut ops = GpuStagedLoadOps {
        source,
        devices,
        layout,
    };
    let staged = match fault {
        None => run_staged_load(&mut ops, n_devices, n_devices == 1)?,
        Some(fault) => run_staged_load_with_fault(&mut ops, n_devices, n_devices == 1, Some(fault))?,
    };
    Ok(LoadedWeights {
        token_embd: staged.token_embd,
        embd_format: staged.embd_format,
        output_norm: staged.output_norm,
        output: staged.output,
        layers: staged.layers,
        lm_head_aliases_embd: staged.lm_head_aliases_embd,
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
            let loaded = run_staged_load(&mut source, 2, false)
                .expect("production retry after seam fault");
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
