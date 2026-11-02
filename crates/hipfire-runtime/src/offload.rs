// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Placement arithmetic, admission and reporting for partial GPU offload.
//!
//! One home for "where do this model's weights live", shared by every arch that
//! adopts the loader seam. An arch supplies only its own byte census
//! ([`LayerBytes`], built from its own tensor index and its own tensor-name
//! classifier) plus the measured capacity; the tier order, the admission checks
//! and the operator-facing line all live here, so a new arch adopts a placement
//! policy instead of copying one.
//!
//! Layering: the config *vocabulary* ([`OffloadBudget`]) lives in
//! `hipfire_config::memory`, which has no intra-workspace dependencies. What
//! moves here is the *arithmetic* config used to own (`largest_fitting_tail`),
//! because placement arithmetic is not configuration vocabulary.
//!
//! Nothing here is a fixed host budget: `MemAvailable`, live GTT and TTM's page
//! pool are read at load time, and the intended failure mode is a load-time
//! refusal that prints numbers rather than an OOM later.

use crate::weight_backend::MemoryTarget;
pub use crate::weight_manifest::WeightResidency;
use hipfire_config::memory::OffloadBudget;

const MIB: u64 = 1 << 20;
const GIB_F: f64 = (1u64 << 30) as f64;

/// One layer's resolved residency: where its own weights live and where its
/// routed experts live.
///
/// The pair the loader hands a `WeightSource::read_layer`, so an arch reads one
/// value instead of two lookups and cannot mix the axes up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerResidency {
    pub weights: WeightResidency,
    pub experts: ExpertResidency,
}

impl LayerResidency {
    /// Both axes on the device (the zero-diff default).
    pub const DEVICE: Self = Self {
        weights: WeightResidency::Resident,
        experts: ExpertResidency::Device,
    };

    /// The layer's own weights are host-placed (dense partial offload).
    pub fn weights_host(&self) -> bool {
        self.weights == WeightResidency::HostMapped
    }

    /// The layer's routed experts are host-placed.
    pub fn experts_host(&self) -> bool {
        self.experts == ExpertResidency::HostMapped
    }
}

/// Where one layer's routed-expert weights live.
///
/// A second axis beside [`WeightResidency`]: a layer can be device-resident while
/// its routed experts are in host RAM (exactly what `memory.moe_expert_budget`
/// buys). Kept as a distinct type so the two axes cannot be swapped by accident.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertResidency {
    Device,
    HostMapped,
}

/// Which end of the layer range keeps its bytes on the device.
///
/// `memory.gpu_layer_budget = N` is the dense reading — the *last* N layers stay
/// resident — while Flash-Next's `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS = N` is the
/// first N. One orientation parameter expresses both, so neither arch forks the
/// search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    /// The last N layers stay on the device; the prefix spills.
    SuffixResident,
    /// The first N layers stay on the device; the suffix spills.
    PrefixResident,
}

impl Orientation {
    /// True when layer `layer` falls outside the `resident` end, given
    /// `host_layers` layers are host-placed from the other end.
    fn is_host(self, layer: usize, n_layers: usize, host_layers: usize) -> bool {
        match self {
            Orientation::SuffixResident => layer < host_layers,
            Orientation::PrefixResident => layer >= n_layers - host_layers.min(n_layers),
        }
    }
}

/// One fully resolved placement: per-layer residency for a layer's *own* weights
/// and for its *routed experts*, as two explicit vectors.
///
/// Both are explicit per layer rather than a count plus an end, so no consumer
/// has to re-derive the orientation — the search owns it and nothing else does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    pub layer_residency: Vec<WeightResidency>,
    pub expert_residency: Vec<ExpertResidency>,
}

impl Placement {
    /// Every layer and every expert on the device (the zero-diff default).
    pub fn all_device(n_layers: usize) -> Self {
        Self {
            layer_residency: vec![WeightResidency::Resident; n_layers],
            expert_residency: vec![ExpertResidency::Device; n_layers],
        }
    }

    /// Dense partial offload: the first `i_gpu_start` layers' own weights are
    /// host-placed and every routed expert stays on the device.
    ///
    /// The shape the dense loader used before the placement search landed, kept
    /// as one constructor so the dense route is expressed through the same value
    /// (and so it can be asserted against the search's output).
    pub fn dense_prefix(n_layers: usize, i_gpu_start: usize) -> Self {
        let host = i_gpu_start.min(n_layers);
        Self {
            layer_residency: (0..n_layers)
                .map(|layer| {
                    if layer < host {
                        WeightResidency::HostMapped
                    } else {
                        WeightResidency::Resident
                    }
                })
                .collect(),
            expert_residency: vec![ExpertResidency::Device; n_layers],
        }
    }

    pub fn layer(&self, layer: usize) -> WeightResidency {
        self.layer_residency
            .get(layer)
            .copied()
            .unwrap_or(WeightResidency::Resident)
    }

    pub fn experts(&self, layer: usize) -> ExpertResidency {
        self.expert_residency
            .get(layer)
            .copied()
            .unwrap_or(ExpertResidency::Device)
    }

    pub fn n_layers(&self) -> usize {
        self.layer_residency.len()
    }

    /// Layers whose own weights are host-placed.
    pub fn host_layers(&self) -> usize {
        self.layer_residency
            .iter()
            .filter(|r| **r == WeightResidency::HostMapped)
            .count()
    }

    /// Layers whose routed experts are host-placed.
    pub fn host_expert_layers(&self) -> usize {
        self.expert_residency
            .iter()
            .filter(|r| **r == ExpertResidency::HostMapped)
            .count()
    }

    /// True when nothing at all is host-placed.
    pub fn is_fully_resident(&self) -> bool {
        self.host_layers() == 0 && self.host_expert_layers() == 0
    }
}

/// Translate a placement residency into the upload destination the loader uses.
pub fn memory_target(residency: WeightResidency) -> MemoryTarget {
    match residency {
        WeightResidency::HostMapped => MemoryTarget::HostMapped,
        // `ExternalRows` never reaches the loader's upload path (it is census-only
        // in the manifest), and `Resident` is VRAM.
        WeightResidency::Resident | WeightResidency::ExternalRows { .. } => MemoryTarget::Device,
    }
}

/// The routed-expert axis of the same translation.
pub fn expert_memory_target(residency: ExpertResidency) -> MemoryTarget {
    match residency {
        ExpertResidency::HostMapped => MemoryTarget::HostMapped,
        ExpertResidency::Device => MemoryTarget::Device,
    }
}

/// Per-layer byte census, built by the arch from its own tensor index.
///
/// `non_expert[layer]` is everything a layer owns except its routed experts
/// (attention, DeltaNet, router, shared expert, norms); `expert[layer]` is its
/// routed-expert payload; `always_resident` is the embedding, the language head
/// and any other global tensor that no budget moves.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LayerBytes {
    pub non_expert: Vec<u64>,
    pub expert: Vec<u64>,
    pub always_resident: u64,
}

impl LayerBytes {
    pub fn n_layers(&self) -> usize {
        self.non_expert.len()
    }

    pub fn expert_total(&self) -> u64 {
        self.expert.iter().sum()
    }

    pub fn weight_total(&self) -> u64 {
        self.always_resident + self.non_expert.iter().sum::<u64>() + self.expert_total()
    }

    /// Device bytes a placement leaves on the card.
    pub fn device_bytes(&self, placement: &Placement) -> u64 {
        let mut bytes = self.always_resident;
        for layer in 0..self.n_layers() {
            if placement.layer(layer) != WeightResidency::HostMapped {
                bytes += self.non_expert[layer];
            }
            if placement.experts(layer) != ExpertResidency::HostMapped {
                bytes += self.expert[layer];
            }
        }
        bytes
    }

    /// Host-pinned bytes a placement takes (routed experts only — a host-placed
    /// layer's own weights are the arch loader's business, and its expert bytes
    /// are already counted here).
    pub fn host_expert_bytes(&self, placement: &Placement) -> u64 {
        (0..self.n_layers())
            .filter(|layer| placement.experts(*layer) == ExpertResidency::HostMapped)
            .map(|layer| self.expert[layer])
            .sum()
    }

    /// Every pinned host byte a placement takes: a host-placed layer's own
    /// weights *and* its experts (the layer knob dominates, so its experts are
    /// counted once), plus the routed experts of layers whose experts moved
    /// without the layer moving. This is what `hipHostMalloc` will hold, so it is
    /// what the host tier must be charged for.
    pub fn host_bytes(&self, placement: &Placement) -> u64 {
        let mut bytes = 0u64;
        for layer in 0..self.n_layers() {
            if placement.layer(layer) == WeightResidency::HostMapped {
                bytes += self.non_expert[layer] + self.expert[layer];
            } else if placement.experts(layer) == ExpertResidency::HostMapped {
                bytes += self.expert[layer];
            }
        }
        bytes
    }

    /// The largest number of routed-expert layers that can stay on the device,
    /// counted from `orientation`'s resident end, given this much device room.
    /// Returns `None` when even spilling every expert does not free enough.
    pub fn largest_resident_expert_prefix(
        &self,
        orientation: Orientation,
        device_room: u64,
    ) -> Option<usize> {
        let n = self.n_layers();
        // Resident end first: try keeping every expert resident, then give one
        // expert layer at a time back to host RAM until it fits.
        for resident in (0..=n).rev() {
            let host_layers = n - resident;
            let mut bytes = self.always_resident + self.non_expert.iter().sum::<u64>();
            for layer in 0..n {
                if !orientation.is_host(layer, n, host_layers) {
                    bytes += self.expert[layer];
                }
            }
            if bytes <= device_room {
                return Some(resident);
            }
        }
        None
    }
}

/// Device room available to weights, after the KV reservation and the prefill
/// floor have already been subtracted by the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capacity {
    pub weight_bytes: u64,
}

/// Device room a placement may spend, from the measured free VRAM and every
/// claim that has a prior call on it: the KV reservation, the prefill floor, any
/// device-resident draft's bytes, and a fixed slack for the allocations nobody
/// sized (screen caches, expanded tensors).
///
/// One function so the two loaders (slot and sequential) cannot drift: a spill
/// that fits one must fit the other.
pub fn weight_capacity(
    free_vram: u64,
    kv_bytes: u64,
    prefill_bytes: u64,
    draft_bytes: u64,
    slack_bytes: u64,
) -> Capacity {
    let reserved = kv_bytes
        .saturating_add(prefill_bytes)
        .saturating_add(draft_bytes)
        .saturating_add(slack_bytes);
    Capacity {
        weight_bytes: free_vram.saturating_sub(reserved),
    }
}

/// The KV footprint a placement must leave room for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvReserve {
    pub bytes: u64,
    pub rows: usize,
    pub stride_bytes: u64,
}

/// Rows × stride with checked multiply. `None` on overflow (an absurd context or
/// stride is a refusal, never a silent wrap).
pub fn kv_reserve(stride_bytes_per_token: u64, rows: usize) -> Option<KvReserve> {
    let bytes = stride_bytes_per_token.checked_mul(u64::try_from(rows).ok()?)?;
    Some(KvReserve {
        bytes,
        rows,
        stride_bytes: stride_bytes_per_token,
    })
}

/// The result of a placement search: the placement and the line that explains it.
pub type PlanResult = Result<(Placement, String), String>;

/// Resolve a placement from the two budgets and the measured device capacity.
///
/// Tier order is llama.cpp's default stance — *experts cold, attention resident*:
/// the routed-expert tier grows first and whole layers only when the expert tier
/// alone cannot free enough. `Auto` on both stops at the first fit, so the spill
/// is the minimum that loads; a model that already fits is untouched and gets
/// `all_device`.
///
/// `Err` is a load refusal that names the bytes and the knob that could have
/// freed them.
pub fn plan(
    layers: &LayerBytes,
    budgets: (OffloadBudget, OffloadBudget),
    capacity: &Capacity,
    orientation: Orientation,
) -> PlanResult {
    let n = layers.n_layers();
    let (layer_budget, expert_budget) = budgets;

    // Explicit targets: how many layers' worth of bytes leave the device.
    let explicit_host = |budget: OffloadBudget| match budget {
        OffloadBudget::Full => Some(0usize),
        OffloadBudget::Layers(resident) => Some(n.saturating_sub(resident.min(n))),
        OffloadBudget::Auto => None,
    };
    let layer_pin = explicit_host(layer_budget);
    let expert_pin = explicit_host(expert_budget);

    // A host-placed layer takes its routed experts with it, so an explicit
    // per-layer expert count is a promise about which layers may host at all:
    // the layer tier can never exceed it.
    let layer_cap = match expert_budget {
        OffloadBudget::Layers(_) => expert_pin.unwrap_or(n),
        OffloadBudget::Full | OffloadBudget::Auto => n,
    };

    let placement = |layer_host: usize, expert_host: usize| -> Placement {
        // A host-placed layer covers its experts: the layer knob dominates.
        let expert_host = expert_host.max(layer_host);
        Placement {
            layer_residency: (0..n)
                .map(|layer| {
                    if orientation.is_host(layer, n, layer_host) {
                        WeightResidency::HostMapped
                    } else {
                        WeightResidency::Resident
                    }
                })
                .collect(),
            expert_residency: (0..n)
                .map(|layer| {
                    if orientation.is_host(layer, n, expert_host) {
                        ExpertResidency::HostMapped
                    } else {
                        ExpertResidency::Device
                    }
                })
                .collect(),
        }
    };

    let fits = |layer_host: usize, expert_host: usize| -> bool {
        layers.device_bytes(&placement(layer_host, expert_host)) <= capacity.weight_bytes
    };

    let fits_line = |placement: &Placement| -> String {
        if placement.is_fully_resident() {
            "partial offload: model fits - every layer resident".to_string()
        } else if layers.expert_total() == 0 {
            // A dense model has no routed-expert tier; saying "routed experts host
            // on N layers" would be technically true (0 bytes) and actively
            // misleading.
            format!(
                "partial offload: {} of {n} layers host-placed, device weights {} MiB of {} MiB \
                 available",
                placement.host_layers(),
                layers.device_bytes(placement) / MIB,
                capacity.weight_bytes / MIB,
            )
        } else {
            format!(
                "partial offload: {} of {n} layers host-placed, routed experts host on {} layers, \
                 device weights {} MiB of {} MiB available",
                placement.host_layers(),
                placement.host_expert_layers(),
                layers.device_bytes(placement) / MIB,
                capacity.weight_bytes / MIB,
            )
        }
    };

    let base_layer = layer_pin.unwrap_or(0);
    let base_expert = expert_pin.unwrap_or(0).max(base_layer);
    let layer_open = layer_pin.is_none();
    let expert_open = expert_pin.is_none();

    // The candidate implied by the knobs themselves: zero spill when nothing is
    // pinned, and the pinned placement otherwise. Checked first so a model that
    // fits is untouched and an explicit placement is never silently widened.
    if fits(base_layer, base_expert) {
        let placement = placement(base_layer, base_expert);
        let line = fits_line(&placement);
        return Ok((placement, line));
    }

    // Both tiers pinned: nothing may move beyond them.
    if !layer_open && !expert_open {
        let placement = placement(base_layer, base_expert);
        return Err(refusal(
            layers,
            &placement,
            capacity,
            &knob_hint(false, false),
        ));
    }

    // Tier 1 — routed experts, the minimum that frees enough.
    let mut best_expert = base_expert;
    if expert_open {
        for expert_host in (base_expert + 1)..=n {
            if fits(base_layer, expert_host) {
                let placement = placement(base_layer, expert_host);
                let line = fits_line(&placement);
                return Ok((placement, line));
            }
        }
        // The expert tier is exhausted; the layer tier starts from every expert
        // already host, so it spills whole layers only as far as it must.
        best_expert = n;
    }

    // Tier 2 — whole layers, bounded by what an explicit expert count promised.
    if layer_open {
        for layer_host in (base_layer + 1)..=layer_cap {
            if fits(layer_host, best_expert.max(layer_host)) {
                let placement = placement(layer_host, best_expert.max(layer_host));
                let line = fits_line(&placement);
                return Ok((placement, line));
            }
        }
    }

    // Nothing fits: report the widest placement the knobs allow, so the numbers
    // say what even a maximal spill leaves over capacity.
    let widest = placement(layer_cap.max(base_layer), n);
    Err(refusal(
        layers,
        &widest,
        capacity,
        &knob_hint(!layer_open, !expert_open),
    ))
}

/// Which budget the operator would have to change, given what they pinned. The
/// direction is deliberately "adjust", not "raise/lower": `Full` and `Layers(n)`
/// need opposite moves, and guessing wrong would send an operator the wrong way.
fn knob_hint(layer_pinned: bool, expert_pinned: bool) -> String {
    match (layer_pinned, expert_pinned) {
        (true, false) => "adjust memory.gpu_layer_budget so more whole layers spill, or lower \
                          memory.max_seq"
            .to_string(),
        (false, true) => "adjust memory.moe_expert_budget so more routed experts spill, or lower \
                          memory.max_seq"
            .to_string(),
        (true, true) => "lower memory.max_seq, or use a smaller quantization".to_string(),
        (false, false) => "lower memory.max_seq, or use a smaller quantization".to_string(),
    }
}

fn refusal(layers: &LayerBytes, placement: &Placement, capacity: &Capacity, hint: &str) -> String {
    let device = layers.device_bytes(placement);
    format!(
        "load refused: this placement needs {} MiB of device weights but only {} MiB is \
         available after the KV reservation and the prefill floor ({} MiB over); {hint}",
        device / MIB,
        capacity.weight_bytes / MIB,
        device.saturating_sub(capacity.weight_bytes) / MIB,
    )
}

/// The operator-facing residency line: the per-layer expert figure they need to
/// pick a count, plus the admission numbers.
pub fn report(placement: &Placement, layers: &LayerBytes, host_bytes: u64) -> String {
    let n = placement.n_layers();
    if layers.expert_total() == 0 {
        return format!(
            "partial offload: {} of {n} layers host-placed; pinned host {} MiB; device weights \
             {} MiB",
            placement.host_layers(),
            host_bytes / MIB,
            layers.device_bytes(placement) / MIB,
        );
    }
    let per_layer = if n == 0 {
        0
    } else {
        layers.expert_total() / n as u64
    };
    // The total must be the *spilled* subset, not the model's whole expert
    // stack: "pinned host" beside it is the spilled bytes, and reporting the
    // whole model there reads as an operator error.
    let spilled_experts = layers.host_expert_bytes(placement);
    format!(
        "moe offload: {} of {n} layers host-placed, routed experts host on {} layers \
         ({} MiB of routed experts per layer, {} MiB total); pinned host {} MiB; \
         device weights {} MiB",
        placement.host_layers(),
        placement.host_expert_layers(),
        per_layer / MIB,
        spilled_experts / MIB,
        host_bytes / MIB,
        layers.device_bytes(placement) / MIB,
    )
}

/// Every measured claim on device memory a placement search must respect,
/// gathered by the arch loader before the first allocation.
///
/// One struct so the sequential carrier and the slots engine cannot drift:
/// both fill the same fields from their own loaders and hand them to
/// [`resolve_placement`]. `kv_rows`/`kv_stride` and `prefill_floor_absent`
/// are display-only — they let the single operator line below name the KV
/// shape and the prefill-floor reason without re-deriving them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapacityInputs {
    /// Measured free VRAM, in bytes.
    pub free_vram: u64,
    /// KV footprint that must stay resident, in bytes.
    pub kv_bytes: u64,
    /// Display-only KV shape: `kv reserved: {kv_rows} tokens x {kv_stride} B`.
    pub kv_rows: usize,
    /// Display-only KV stride, in bytes per token.
    pub kv_stride: u64,
    /// Prefill-scratch floor, in bytes (0 when the arch has no sizing model).
    pub prefill_floor: u64,
    /// True when `prefill_floor` is 0 for lack of a sizing model (MoE), not
    /// because the floor measured zero — the operator line says so.
    pub prefill_floor_absent: bool,
    /// Device-resident draft bytes (MTP head, DFlash draft) that ride beside
    /// the trunk and are not part of its spill.
    pub draft_bytes: u64,
    /// Fixed slack for the allocations nobody sized.
    pub slack: u64,
}

/// The whole placement assembly in one place: capacity, search, operator
/// lines, host admission.
///
/// Computes the [`Capacity`] via [`weight_capacity`], runs [`plan`], prints
/// the plan line plus the single operator capacity line ([`capacity_line`])
/// and — only for a spilling placement — the residency [`report`], then
/// admits the pinned host bytes via [`admit_host_placement`]. The tier order
/// in `plan` and the `orientation` parameter are untouched; this owns only
/// the order the steps run in, so the two loaders cannot drift apart again.
pub fn resolve_placement(
    layer_bytes: &LayerBytes,
    inputs: &CapacityInputs,
    budgets: (OffloadBudget, OffloadBudget),
    orientation: Orientation,
) -> Result<Placement, String> {
    let capacity = weight_capacity(
        inputs.free_vram,
        inputs.kv_bytes,
        inputs.prefill_floor,
        inputs.draft_bytes,
        inputs.slack,
    );
    let (placement, line) = plan(layer_bytes, budgets, &capacity, orientation)?;
    eprintln!("  {line}");
    eprintln!("  {}", capacity_line(inputs, &capacity));
    let host_bytes = layer_bytes.host_bytes(&placement);
    if !placement.is_fully_resident() {
        eprintln!("  {}", report(&placement, layer_bytes, host_bytes));
    }
    admit_host_placement(host_bytes)?;
    Ok(placement)
}

/// The single operator capacity line both loaders print.
///
/// The sequential form (the richer one): KV as rows × stride, the prefill
/// floor with its reason when the arch has no sizing model for it, then the
/// draft reserve, the slack, and the resulting weight budget against free
/// VRAM. No caller formats its own copy.
pub fn capacity_line(inputs: &CapacityInputs, capacity: &Capacity) -> String {
    let pbs = if inputs.prefill_floor_absent {
        "0 MiB (no MoE prefill sizing)".to_string()
    } else {
        format!("{} MiB", inputs.prefill_floor / MIB)
    };
    format!(
        "kv reserved: {} tokens x {} B = {} MiB; pbs floor {}; draft {} MiB; slack {} MiB; \
         weight budget {} MiB of {} MiB free",
        inputs.kv_rows,
        inputs.kv_stride,
        inputs.kv_bytes / MIB,
        pbs,
        inputs.draft_bytes / MIB,
        inputs.slack / MIB,
        capacity.weight_bytes / MIB,
        inputs.free_vram / MIB,
    )
}

/// Largest contiguous resident tail fitting `effective_capacity_bytes`, kept here
/// (moved out of `hipfire_config`) because it is placement arithmetic, not config
/// vocabulary. `Layers`-pinned placements use [`plan`]; this is the scan `auto`
/// uses for the whole-layer tier in a single-pass form.
///
/// Returns the smallest `i_gpu_start` that fits — spilling only the prefix that
/// must — or `None` when even offloading every layer overshoots.
pub fn largest_fitting_tail(
    per_layer_weight_bytes: &[u64],
    kv_bytes: u64,
    always_resident_bytes: u64,
    effective_capacity_bytes: u64,
) -> Option<usize> {
    let mut resident =
        always_resident_bytes + per_layer_weight_bytes.iter().sum::<u64>() + kv_bytes;
    if resident <= effective_capacity_bytes {
        return Some(0);
    }
    for i_gpu_start in 1..=per_layer_weight_bytes.len() {
        resident = resident.saturating_sub(per_layer_weight_bytes[i_gpu_start - 1]);
        if resident <= effective_capacity_bytes {
            return Some(i_gpu_start);
        }
    }
    None
}

// ── Host admission ──────────────────────────────────────────────────────────

/// Host RAM that must remain available after the pinned weights are placed.
/// Pinned pages cannot be reclaimed, so over-committing them starves the host.
pub const HOST_RAM_HEADROOM_BYTES: u64 = 4 << 30;

/// Slack [`rdna_compute::Gpu::upload_raw_host_mapped`] adds to every host-mapped
/// tensor.
pub const HOST_MAPPED_PAD_BYTES: u64 = 1 << 20;

/// TTM's page limit, in pages, which caps every GTT allocation on the host.
const TTM_PAGES_LIMIT: &str = "/sys/module/ttm/parameters/pages_limit";

/// TTM's page pool cap, in pages. Freed GTT pages past it go back to the kernel
/// at once.
const TTM_PAGE_POOL_SIZE: &str = "/sys/module/ttm/parameters/page_pool_size";

/// TTM's page size on x86_64, the only host ROCm supports for discrete GPUs.
const TTM_PAGE_BYTES: u64 = 4096;

/// Host memory outside every `/proc/meminfo` counter that is not TTM's pool:
/// other drivers' pages, DMA buffers, firmware. [`ttm_pool_estimate`] never
/// counts this much as pool. Measured on the 5-card gfx1201 host with the pool
/// empty and 46.2 GiB of live GTT: 2.76 GiB.
pub const UNTRACKED_KERNEL_BYTES: u64 = 4 << 30;

/// `/proc/meminfo` fields, in bytes, that together account for every allocated
/// page except driver pages (TTM's pool and live GTT among them). Subset fields
/// (`Shmem`, `Mlocked`, `AnonHugePages`, ...) are left out.
const MEMINFO_TRACKED: &[&str] = &[
    "MemFree",
    "Buffers",
    "Cached",
    "SwapCached",
    "AnonPages",
    "Slab",
    "KernelStack",
    "ShadowCallStack",
    "PageTables",
    "SecPageTables",
    "VmallocUsed",
    "Percpu",
    "Hugetlb",
    "Zswap",
    "Unaccepted",
    "Balloon",
];

fn host_mapped_is_gtt() -> bool {
    std::env::var("HSA_USERPTR_FOR_PAGED_MEM").is_ok_and(|value| value.trim() == "0")
}

fn read_u64(path: &std::path::Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// `MemAvailable - CMA` in bytes, read now
/// ([`rdna_compute::uma_memory::mem_available_bytes`]). `None` when unreadable
/// (admission then refuses rather than assuming room).
pub fn mem_available_bytes() -> Option<u64> {
    rdna_compute::uma_memory::mem_available_bytes()
}

/// `mem_info_gtt_used` summed over every amdgpu device. Host allocations may
/// count against another device than the one the process runs on (on a 5-card
/// gfx1201 host, a card-2 process's host-mapped experts show in card 0's
/// `mem_info_gtt_used`).
fn amdgpu_gtt_used_bytes() -> Option<u64> {
    let mut used_bytes = 0u64;
    for card in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
        let name = card.file_name();
        let is_card = name
            .to_str()
            .and_then(|name| name.strip_prefix("card"))
            .is_some_and(|index| !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()));
        if is_card {
            used_bytes += read_u64(&card.path().join("device/mem_info_gtt_used")).unwrap_or(0);
        }
    }
    Some(used_bytes)
}

/// Estimate of the freed GTT pages TTM keeps in its page pool, from
/// `/proc/meminfo` text, the GTT amdgpu devices hold, and TTM's `page_pool_size`
/// in pages.
///
/// The pool is invisible to an unprivileged process: its pages are in no
/// `/proc/meminfo` counter, so `MemAvailable` excludes them, and
/// `mem_info_gtt_used` drops them when their buffer is freed (only root's
/// `/sys/kernel/debug/ttm/page_pool` reads it). It is `MemTotal` minus every
/// tracked counter, minus the live GTT and [`UNTRACKED_KERNEL_BYTES`], capped at
/// `page_pool_size`. `None` when a field is missing.
pub fn ttm_pool_estimate_from(meminfo: &str, gtt_used: u64, pool_size_pages: u64) -> Option<u64> {
    let field = |key: &str| -> Option<u64> {
        meminfo.lines().find_map(|line| {
            let rest = line.strip_prefix(key)?.strip_prefix(':')?;
            Some(rest.split_whitespace().next()?.parse::<u64>().ok()? * 1024)
        })
    };
    let total = field("MemTotal")?;
    field("MemFree")?;
    let tracked: u64 = MEMINFO_TRACKED
        .iter()
        .filter_map(|key| field(key))
        .sum::<u64>()
        + field("KReclaimable")?.saturating_sub(field("SReclaimable")?);
    let untracked = total
        .saturating_sub(tracked)
        .saturating_sub(gtt_used)
        .saturating_sub(UNTRACKED_KERNEL_BYTES);
    Some(untracked.min(pool_size_pages.saturating_mul(TTM_PAGE_BYTES)))
}

/// Host RAM held in TTM's page pool that host-mapped weights can take. amdgpu
/// parks the write-combined and uncached pages of freed GTT buffers there, up to
/// `page_pool_size` (half of RAM by default), so a process that exits leaves its
/// host-mapped weights' pages in it. A new GTT allocation takes pages from the
/// pool first, and TTM's shrinker frees it under memory pressure. 0 unless
/// host-mapped memory is GTT-backed (`HSA_USERPTR_FOR_PAGED_MEM=0`), or when
/// sysfs or `/proc/meminfo` is unreadable.
pub fn ttm_pool_estimate() -> u64 {
    if !host_mapped_is_gtt() {
        return 0;
    }
    let estimate = || -> Option<u64> {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let pool_size_pages = read_u64(TTM_PAGE_POOL_SIZE.as_ref())?;
        ttm_pool_estimate_from(&meminfo, amdgpu_gtt_used_bytes()?, pool_size_pages)
    };
    estimate().unwrap_or(0)
}

/// GTT room on the host: TTM's cap and what amdgpu devices already hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GttBudget {
    pub limit_bytes: u64,
    pub used_bytes: u64,
}

/// The GTT budget when host-mapped memory is GTT-backed, i.e. when
/// `HSA_USERPTR_FOR_PAGED_MEM` is `0` (hip-bridge's default). It is `None` under
/// userptr, or when sysfs does not expose TTM's limit. `used_bytes` sums
/// `mem_info_gtt_used` over every amdgpu device.
pub fn gtt_budget() -> Option<GttBudget> {
    if !host_mapped_is_gtt() {
        return None;
    }
    let limit_bytes = read_u64(TTM_PAGES_LIMIT.as_ref())?.checked_mul(TTM_PAGE_BYTES)?;
    Some(GttBudget {
        limit_bytes,
        used_bytes: amdgpu_gtt_used_bytes()?,
    })
}

/// Refuse before any allocation when the pinned weights would not leave
/// [`HOST_RAM_HEADROOM_BYTES`] of the host RAM they can take: `mem_available`
/// (`MemAvailable - CMA`, [`mem_available_bytes`]; CMA pageblocks serve
/// only movable allocations) plus `ttm_pool_bytes`, the estimate of freed GTT
/// pages parked in TTM's page pool ([`ttm_pool_estimate`]). Pinned pages
/// cannot be reclaimed, so over-committing them starves the rest of the host.
pub fn check_host_ram(
    host_bytes: u64,
    mem_available: Option<u64>,
    ttm_pool_bytes: u64,
) -> Result<(), String> {
    if host_bytes == 0 {
        return Ok(());
    }
    let Some(available) = mem_available else {
        return Err(format!(
            "host-placed weights need {:.1} GiB of pinned host RAM, but MemAvailable is unreadable",
            host_bytes as f64 / GIB_F
        ));
    };
    let needed = host_bytes + HOST_RAM_HEADROOM_BYTES;
    if available.saturating_add(ttm_pool_bytes) < needed {
        return Err(format!(
            "host-placed weights need {:.1} GiB of pinned host RAM plus {:.0} GiB headroom, but \
             MemAvailable - CMA is {:.1} GiB (plus {:.1} GiB estimated in TTM's page pool); free host \
             memory or keep more weights in VRAM (memory.moe_expert_budget / \
             memory.gpu_layer_budget)",
            host_bytes as f64 / GIB_F,
            HOST_RAM_HEADROOM_BYTES as f64 / GIB_F,
            available as f64 / GIB_F,
            ttm_pool_bytes as f64 / GIB_F
        ));
    }
    Ok(())
}

/// Refuse before any allocation when GTT-backed host-mapped weights would not fit
/// under TTM's `pages_limit` beside what amdgpu devices already hold. Past it
/// `hipHostMalloc` fails, after the load has uploaded the VRAM weights. `None`
/// (userptr, or no TTM limit in sysfs) skips the check.
pub fn check_gtt_cap(host_bytes: u64, budget: Option<GttBudget>) -> Result<(), String> {
    let Some(GttBudget {
        limit_bytes,
        used_bytes,
    }) = budget
    else {
        return Ok(());
    };
    let free = limit_bytes.saturating_sub(used_bytes);
    if host_bytes > free {
        return Err(format!(
            "host-placed weights need {:.1} GiB of GTT-backed host RAM, but TTM's GTT cap \
             ({TTM_PAGES_LIMIT} = {} pages, {:.1} GiB) has {:.1} GiB left beside the {:.1} GiB \
             amdgpu devices already hold; keep more weights in VRAM (memory.moe_expert_budget / \
             memory.gpu_layer_budget), raise ttm.pages_limit, or set HSA_USERPTR_FOR_PAGED_MEM=1 \
             for pageable userptr host memory, which host-memory pressure can stall",
            host_bytes as f64 / GIB_F,
            limit_bytes / TTM_PAGE_BYTES,
            limit_bytes as f64 / GIB_F,
            free as f64 / GIB_F,
            used_bytes as f64 / GIB_F
        ));
    }
    Ok(())
}

/// Both host-tier admissions, read now: `MemAvailable - CMA` + TTM's page pool, then
/// GTT's cap. This is the one call a loader makes before allocating anything
/// host-mapped.
pub fn admit_host_placement(host_bytes: u64) -> Result<(), String> {
    check_host_ram(host_bytes, mem_available_bytes(), ttm_pool_estimate())?;
    check_gtt_cap(host_bytes, gtt_budget())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(non_expert: &[u64], expert: &[u64], always: u64) -> LayerBytes {
        LayerBytes {
            non_expert: non_expert.to_vec(),
            expert: expert.to_vec(),
            always_resident: always,
        }
    }

    #[test]
    fn a_model_that_fits_is_untouched() {
        let layers = bytes(&[100, 100, 100], &[400, 400, 400], 50);
        let (placement, line) = plan(
            &layers,
            (OffloadBudget::Auto, OffloadBudget::Auto),
            &Capacity { weight_bytes: 4096 },
            Orientation::SuffixResident,
        )
        .unwrap();
        assert!(placement.is_fully_resident(), "{placement:?}");
        assert_eq!(
            placement.layer_residency,
            vec![WeightResidency::Resident; 3]
        );
        assert!(line.contains("model fits"), "{line}");
    }

    #[test]
    fn auto_spills_experts_before_whole_layers() {
        // 3 layers x (100 non-expert + 400 experts) + 50 = 1550 total.
        let layers = bytes(&[100, 100, 100], &[400, 400, 400], 50);
        // Room for the non-experts + two expert layers: 50 + 300 + 800 = 1150.
        let (placement, _) = plan(
            &layers,
            (OffloadBudget::Auto, OffloadBudget::Auto),
            &Capacity { weight_bytes: 1150 },
            Orientation::SuffixResident,
        )
        .unwrap();
        // Experts first: no whole layer spilled, one expert layer host.
        assert_eq!(placement.host_layers(), 0);
        assert_eq!(
            placement.expert_residency,
            vec![
                ExpertResidency::HostMapped,
                ExpertResidency::Device,
                ExpertResidency::Device
            ]
        );
        assert_eq!(layers.device_bytes(&placement), 1150);
    }

    #[test]
    fn auto_reaches_whole_layers_when_experts_cannot_free_enough() {
        let layers = bytes(&[100, 100, 100], &[400, 400, 400], 50);
        // Every expert host (50 + 300 + 0 = 350) still overshoots 200, so the
        // whole-layer tier must run.
        let (placement, _) = plan(
            &layers,
            (OffloadBudget::Auto, OffloadBudget::Auto),
            &Capacity { weight_bytes: 200 },
            Orientation::SuffixResident,
        )
        .unwrap();
        assert_eq!(placement.host_layers(), 2);
        assert_eq!(layers.device_bytes(&placement), 150);
        // A host layer covers its experts.
        assert!(placement.experts(0) == ExpertResidency::HostMapped);
        assert!(placement.experts(1) == ExpertResidency::HostMapped);
    }

    #[test]
    fn layer_knob_pins_and_experts_follow_it() {
        let layers = bytes(&[100, 100, 100, 100], &[400, 400, 400, 400], 50);
        // "the last 2 layers resident" -> first 2 host.
        let (placement, _) = plan(
            &layers,
            (OffloadBudget::Layers(2), OffloadBudget::Full),
            &Capacity {
                weight_bytes: u64::MAX,
            },
            Orientation::SuffixResident,
        )
        .unwrap();
        assert_eq!(placement.host_layers(), 2);
        assert_eq!(
            placement.expert_residency,
            vec![
                ExpertResidency::HostMapped,
                ExpertResidency::HostMapped,
                ExpertResidency::Device,
                ExpertResidency::Device
            ]
        );
        // Host layer bytes left the device even though the budget was ignored:
        // plan() refuses rather than silently placing resident.
        assert!(plan(
            &layers,
            (OffloadBudget::Layers(2), OffloadBudget::Full),
            &Capacity { weight_bytes: 10 },
            Orientation::SuffixResident,
        )
        .is_err());
    }

    #[test]
    fn prefix_orientation_expresses_the_qwen4_policy() {
        let layers = bytes(&[100, 100, 100, 100], &[400, 400, 400, 400], 50);
        // "the first 2 layers keep their experts resident".
        let (placement, _) = plan(
            &layers,
            (OffloadBudget::Full, OffloadBudget::Layers(2)),
            &Capacity {
                weight_bytes: u64::MAX,
            },
            Orientation::PrefixResident,
        )
        .unwrap();
        assert_eq!(
            placement.expert_residency,
            vec![
                ExpertResidency::Device,
                ExpertResidency::Device,
                ExpertResidency::HostMapped,
                ExpertResidency::HostMapped
            ]
        );
        assert_eq!(placement.host_layers(), 0);
    }

    #[test]
    fn auto_refuses_when_even_a_full_spill_overshoots() {
        let layers = bytes(&[100, 100], &[400, 400], 50);
        let error = plan(
            &layers,
            (OffloadBudget::Auto, OffloadBudget::Auto),
            &Capacity { weight_bytes: 10 },
            Orientation::SuffixResident,
        )
        .unwrap_err();
        assert!(error.contains("load refused"), "{error}");
        assert!(error.contains("memory.max_seq"), "{error}");
    }

    #[test]
    fn explicit_expert_budget_alone_cannot_be_exceeded_silently() {
        let layers = bytes(&[100, 100, 100], &[400, 400, 400], 50);
        // Every expert pinned on the device and no room: refuse, naming the knob
        // that would have to move. Whole layers may not be spilled either, since
        // a host layer would take its pinned experts with it.
        let error = plan(
            &layers,
            (OffloadBudget::Auto, OffloadBudget::Layers(3)),
            &Capacity { weight_bytes: 500 },
            Orientation::SuffixResident,
        )
        .unwrap_err();
        assert!(error.contains("memory.moe_expert_budget"), "{error}");
    }

    #[test]
    fn a_pinned_layer_budget_names_the_layer_knob_when_experts_cannot_free_enough() {
        let layers = bytes(&[400, 400, 400], &[100, 100, 100], 50);
        // Experts are already all host and layers are pinned resident: the layer
        // knob is the one that has to move.
        let error = plan(
            &layers,
            (OffloadBudget::Layers(3), OffloadBudget::Auto),
            &Capacity { weight_bytes: 500 },
            Orientation::SuffixResident,
        )
        .unwrap_err();
        assert!(error.contains("memory.gpu_layer_budget"), "{error}");
    }

    #[test]
    fn largest_fitting_tail_spills_the_prefix_that_must() {
        let per_layer = [100u64, 100, 100];
        assert_eq!(largest_fitting_tail(&per_layer, 0, 50, 400), Some(0));
        assert_eq!(largest_fitting_tail(&per_layer, 0, 50, 350), Some(0));
        assert_eq!(largest_fitting_tail(&per_layer, 50, 50, 300), Some(1));
        assert_eq!(largest_fitting_tail(&per_layer, 0, 50, 50), Some(3));
        assert_eq!(largest_fitting_tail(&per_layer, 0, 50, 49), None);
    }

    #[test]
    fn kv_reserve_is_checked() {
        assert_eq!(
            kv_reserve(272, 4),
            Some(KvReserve {
                bytes: 1088,
                rows: 4,
                stride_bytes: 272
            })
        );
        assert_eq!(kv_reserve(u64::MAX, 4), None);
    }

    #[test]
    fn host_ram_check_refuses_below_headroom() {
        let host = 60u64 << 30;
        assert!(check_host_ram(host, Some(host + HOST_RAM_HEADROOM_BYTES), 0).is_ok());
        let error = check_host_ram(host, Some(host + HOST_RAM_HEADROOM_BYTES - 1), 0).unwrap_err();
        assert!(error.contains("MemAvailable"), "{error}");
        assert!(check_host_ram(host, None, 0).is_err());
        assert!(check_host_ram(host, None, u64::MAX).is_err());
        assert!(check_host_ram(0, None, 0).is_ok());
        // The pool counts toward the room, byte for byte.
        assert!(check_host_ram(host, Some(host), HOST_RAM_HEADROOM_BYTES).is_ok());
        assert!(check_host_ram(host, Some(host), HOST_RAM_HEADROOM_BYTES - 1).is_err());
    }

    /// `/proc/meminfo` text from `(field, MiB)` pairs.
    fn meminfo(fields: &[(&str, u64)]) -> String {
        fields
            .iter()
            .map(|(key, mib)| format!("{key}:{:>16} kB\n", mib << 10))
            .collect()
    }

    /// The measured 5-card gfx1201 host (MemTotal 128865156 kB, pool cap
    /// 16108144 pages), 5 x 16 MiB of idle GTT, and `pool_mib` of pages outside
    /// every counter on top of `baseline_mib`.
    fn gfx1201_host(pool_mib: u64, baseline_mib: u64) -> String {
        let total = 128_865_156 >> 10;
        let (anon, slab, sreclaim) = (14 << 10, 3 << 10, 2 << 10);
        let cached = total - pool_mib - baseline_mib - 80 - anon - slab - 1024;
        meminfo(&[
            ("MemTotal", total),
            ("MemFree", 1024),
            ("MemAvailable", 45_800),
            ("Cached", cached),
            ("SwapCached", 0),
            ("AnonPages", anon),
            ("Shmem", 2048),
            ("KReclaimable", sreclaim),
            ("Slab", slab),
            ("SReclaimable", sreclaim),
        ])
    }

    #[test]
    fn ttm_pool_estimate_credits_only_untracked_pages_past_the_baseline() {
        const POOL_PAGES: u64 = 16_108_144;
        let gtt = 5 * (16 << 20);
        let mib = |bytes: u64| bytes >> 20;
        // The measured baseline (2.76 GiB, pool empty) is never credited.
        assert_eq!(
            ttm_pool_estimate_from(&gfx1201_host(0, 2826), gtt, POOL_PAGES),
            Some(0)
        );
        // The reported leftover: 15,292,712 pool pages (59,737 MiB).
        let pool = ttm_pool_estimate_from(&gfx1201_host(59_737, 2826), gtt, POOL_PAGES).unwrap();
        assert_eq!(mib(pool), 59_737 + 2826 - (UNTRACKED_KERNEL_BYTES >> 20));
        // Capped at page_pool_size.
        let capped = ttm_pool_estimate_from(&gfx1201_host(59_737, 2826), gtt, 1 << 20).unwrap();
        assert_eq!(capped, (1 << 20) * TTM_PAGE_BYTES);
        // Live GTT (another host-mapped load) is not pool.
        let live = ttm_pool_estimate_from(&gfx1201_host(0, 2826), 46 << 30, POOL_PAGES);
        let held = ttm_pool_estimate_from(&gfx1201_host(46 << 10, 2826), 46 << 30, POOL_PAGES);
        assert_eq!((live, held), (Some(0), Some(0)));
    }

    #[test]
    fn leftover_pool_admits_the_reload_and_a_real_shortage_still_refuses() {
        // N=12 host-maps 46.1 GiB; the reported refusal read MemAvailable 45.8 GiB.
        let host = 47_206u64 << 20;
        let available = Some(46_899u64 << 20);
        let gtt = 5 * (16 << 20);
        assert!(check_host_ram(host, available, 0).is_err());
        let leftover = ttm_pool_estimate_from(&gfx1201_host(59_737, 2826), gtt, 16_108_144);
        assert!(check_host_ram(host, available, leftover.unwrap()).is_ok());
        // The same MemAvailable with nothing parked in the pool.
        let empty = ttm_pool_estimate_from(&gfx1201_host(0, 2826), gtt, 16_108_144);
        let error = check_host_ram(host, available, empty.unwrap()).unwrap_err();
        assert!(
            error.contains("MemAvailable - CMA is 45.8 GiB (plus 0.0 GiB"),
            "{error}"
        );
        // A pool smaller than the 4.3 GiB gap still refuses.
        let small = ttm_pool_estimate_from(&gfx1201_host(3 << 10, 2826), gtt, 16_108_144);
        assert!(check_host_ram(host, available, small.unwrap()).is_err());
    }

    #[test]
    fn gtt_cap_counts_what_devices_already_hold() {
        // This host: pages_limit 16108144 (61.4 GiB); N=12 host-maps 46.1 GiB.
        let limit_bytes = 16_108_144 * TTM_PAGE_BYTES;
        let host = 46u64 << 30;
        let fresh = GttBudget {
            limit_bytes,
            used_bytes: 16 << 20,
        };
        assert!(check_gtt_cap(host, Some(fresh)).is_ok());
        assert!(check_gtt_cap(limit_bytes - fresh.used_bytes, Some(fresh)).is_ok());
        let error = check_gtt_cap(limit_bytes - fresh.used_bytes + 1, Some(fresh)).unwrap_err();
        assert!(
            error.contains("pages_limit") && error.contains("16108144 pages"),
            "{error}"
        );
        // A second load beside one already holding 46 GiB of GTT is refused.
        let beside = GttBudget {
            limit_bytes,
            used_bytes: 46 << 30,
        };
        assert!(check_gtt_cap(host, Some(beside)).is_err());
        // Userptr host memory, or no TTM limit in sysfs, is not GTT-capped.
        assert!(check_gtt_cap(u64::MAX, None).is_ok());
    }

    #[test]
    fn report_names_the_per_layer_expert_figure() {
        let layers = bytes(&[100, 100, 100, 100], &[400, 400, 400, 400], 50);
        let (placement, _) = plan(
            &layers,
            (OffloadBudget::Full, OffloadBudget::Layers(2)),
            &Capacity {
                weight_bytes: u64::MAX,
            },
            Orientation::SuffixResident,
        )
        .unwrap();
        let line = report(&placement, &layers, 800 << 20);
        assert!(line.contains("routed experts per layer"), "{line}");
        assert!(line.contains("pinned host 800 MiB"), "{line}");
    }

    #[test]
    fn resolve_placement_runs_the_assembly_unchanged() {
        let layers = bytes(&[100, 100, 100], &[400, 400, 400], 50);
        let inputs = CapacityInputs {
            free_vram: 4096,
            kv_bytes: 200,
            kv_rows: 4,
            kv_stride: 50,
            prefill_floor: 0,
            prefill_floor_absent: true,
            draft_bytes: 0,
            slack: 128,
        };
        let budgets = (OffloadBudget::Auto, OffloadBudget::Auto);
        let expected_capacity = weight_capacity(
            inputs.free_vram,
            inputs.kv_bytes,
            inputs.prefill_floor,
            inputs.draft_bytes,
            inputs.slack,
        );
        let (expected_placement, _) = plan(
            &layers,
            budgets,
            &expected_capacity,
            Orientation::SuffixResident,
        )
        .unwrap();
        let expected_host = layers.host_bytes(&expected_placement);
        let placement =
            resolve_placement(&layers, &inputs, budgets, Orientation::SuffixResident).unwrap();
        assert_eq!(placement, expected_placement);
        assert_eq!(
            layers.device_bytes(&placement),
            layers.device_bytes(&expected_placement)
        );
        assert_eq!(layers.host_bytes(&placement), expected_host);
        // The shared capacity line names every claim (KV shape, prefill
        // reason, draft, slack, budget) in the one adopted format.
        let line = capacity_line(&inputs, &expected_capacity);
        assert!(line.contains("kv reserved: 4 tokens x 50 B"), "{line}");
        assert!(
            line.contains("pbs floor 0 MiB (no MoE prefill sizing)"),
            "{line}"
        );
        assert!(line.contains("weight budget"), "{line}");
        // A measured prefill floor prints its value, not the MoE reason.
        let sized = CapacityInputs {
            prefill_floor: 1 << 20,
            prefill_floor_absent: false,
            ..inputs
        };
        let sized_line = capacity_line(
            &sized,
            &weight_capacity(
                sized.free_vram,
                sized.kv_bytes,
                sized.prefill_floor,
                sized.draft_bytes,
                sized.slack,
            ),
        );
        assert!(sized_line.contains("pbs floor 1 MiB"), "{sized_line}");
        assert!(
            !sized_line.contains("no MoE prefill sizing"),
            "{sized_line}"
        );
    }

    #[test]
    fn spilling_inputs_resolve_identically_whichever_loader_built_them() {
        // Same census + same charged totals + same budgets ⇒ identical
        // Capacity/Placement no matter which loader built the inputs: this is
        // the invariant that keeps the sequential and slots routes from
        // drifting apart again.
        let layers = bytes(&[100, 100, 100], &[400, 400, 400], 50);
        let budgets = (OffloadBudget::Auto, OffloadBudget::Auto);
        let kv = kv_reserve(10, 10).unwrap();
        let sequential = CapacityInputs {
            free_vram: 1500,
            kv_bytes: kv.bytes,
            kv_rows: kv.rows,
            kv_stride: kv.stride_bytes,
            prefill_floor: 0,
            prefill_floor_absent: true,
            draft_bytes: 250,
            slack: 0,
        };
        // The slots route factors the same arena product through kv_reserve;
        // only the (rows, stride) factorisation differs, never the total.
        let through_slots = kv_reserve(kv.bytes, 1).unwrap();
        assert_eq!(through_slots.bytes, kv.bytes);
        let slots = CapacityInputs {
            kv_rows: through_slots.rows,
            kv_stride: through_slots.stride_bytes,
            ..sequential
        };
        let cap_seq = weight_capacity(
            sequential.free_vram,
            sequential.kv_bytes,
            sequential.prefill_floor,
            sequential.draft_bytes,
            sequential.slack,
        );
        let cap_slots = weight_capacity(
            slots.free_vram,
            slots.kv_bytes,
            slots.prefill_floor,
            slots.draft_bytes,
            slots.slack,
        );
        assert_eq!(cap_seq, cap_slots);
        let (p_seq, _) = plan(&layers, budgets, &cap_seq, Orientation::SuffixResident).unwrap();
        let (p_slots, _) = plan(&layers, budgets, &cap_slots, Orientation::SuffixResident).unwrap();
        assert_eq!(p_seq, p_slots);
        assert!(!p_seq.is_fully_resident());
    }

    #[test]
    fn memory_target_maps_residency() {
        assert_eq!(
            memory_target(WeightResidency::Resident),
            MemoryTarget::Device
        );
        assert_eq!(
            memory_target(WeightResidency::HostMapped),
            MemoryTarget::HostMapped
        );
    }
}
