// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host-side helpers for the Qwen4 MoE CPU expert splice.
//!
//! A host-mapped routed-expert layer keeps its experts in two aggregate
//! allocations (gate/up and down), `experts` equal-stride extents each. The CPU
//! splice multiplies straight out of the pinned host bytes, so it needs, per
//! expert, one borrowed `&[u8]` into each aggregate. [`CpuExpertOwners`] caches
//! the host address of the two OWNING allocations once (a registry lookup per
//! aggregate, not per token) and hands out checked sub-slices.
//!
//! [`CpuSpliceAdmission`] is the pure arming decision: the splice runs only for
//! host-mapped, supported layers with no EP / slab / tier / AWQ route that
//! owns the weights differently.
//!
//! [`CpuSplitAdmission`] is the same decision for the D2 split, over every
//! host-span owner kind ([`SplitOwnerKind`]: hot-set tier, whole-layer host
//! aggregate, host-mapped expert-slab arena). Both are architecture-generic:
//! neither has an arch input. D2 arms on any tier plan the loader admitted
//! (default sibling sidecar on gfx1201 only, an explicit
//! `HIPFIRE_QWEN4_EXPERT_HOTSET` path on any arch; see
//! `expert_hotset::hotset_arch_admits`) and on any host span whose experts all
//! have a CPU decoder, and refuses, with one log line per reason, every route
//! that has no masked ids or cold-slot installation.

use super::*;

/// Largest verify window (rows) that counts as "narrow" for the CPU splice.
pub(super) const NARROW_VERIFY_MAX_ROWS: usize =
    hipfire_dispatch::families::moe::NARROW_VERIFY_MAX_ROWS;

/// Whether the CPU expert splice may arm for one layer. Every field is an
/// input; [`Self::armed`] is the single decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct CpuSpliceAdmission {
    /// `memory.offload_exec=cpu` selected the CPU splice.
    pub enabled: bool,
    /// The layer's routed-expert aggregates are host-mapped (not VRAM-resident).
    pub host_mapped: bool,
    /// The layer is routed through expert parallelism.
    pub ep: bool,
    /// The layer is a sealed expert-slab arena.
    pub slab: bool,
    /// The layer is governed by a per-expert tier plan.
    pub tier: bool,
    /// The layer carries AWQ scales (a bare CPU gemv would ignore them).
    pub awq: bool,
    /// Both aggregate dtypes have a CPU executor.
    pub supported: bool,
}

impl CpuSpliceAdmission {
    /// Armed only when enabled, host-mapped and supported, with no exclusion.
    pub(super) fn armed(&self) -> bool {
        self.enabled
            && self.host_mapped
            && self.supported
            && !self.ep
            && !self.slab
            && !self.tier
            && !self.awq
    }
}

/// `[expert_idx * stride, (expert_idx + 1) * stride)` inside an aggregate of
/// `experts` extents, `None` when the index is out of range or the arithmetic
/// overflows. The only place the per-expert offset is computed.
fn expert_range(
    expert_idx: usize,
    experts: usize,
    stride: usize,
) -> Option<std::ops::Range<usize>> {
    if expert_idx >= experts {
        return None;
    }
    let start = expert_idx.checked_mul(stride)?;
    let end = start.checked_add(stride)?;
    Some(start..end)
}

/// Cached host views of one layer's two aggregate expert allocations.
///
/// The private runtime borrows these addresses under the bundle's ownership:
/// aggregate weights outlive the forward and are not replaced after attachment.
/// As with `slab_stage`'s host sources, only `Gpu::host_bytes` of the owning
/// tensors supplies host addresses; borrowed device views are never re-tagged.
pub(super) struct CpuExpertOwners {
    gate_up_ptr: usize,
    gate_up_len: usize,
    down_ptr: usize,
    down_len: usize,
    experts: usize,
    gate_up_stride: usize,
    down_stride: usize,
}

impl CpuExpertOwners {
    /// Resolve the OWNING host-mapped tensors' pinned host bytes and check that
    /// `experts` extents of the given strides fit inside them. `gate_up` and
    /// `down` must be the aggregate owners, never borrowed sub-views: a view's
    /// `host_bytes` lookup fails (its pointer is not a registry key), which is
    /// reported as `None` rather than retagged.
    pub(super) fn new(
        gpu: &Gpu,
        gate_up: &GpuTensor,
        down: &GpuTensor,
        experts: usize,
        gate_up_stride: usize,
        down_stride: usize,
    ) -> Option<Self> {
        let gate_up_bytes = gpu.host_bytes(gate_up)?;
        let down_bytes = gpu.host_bytes(down)?;
        Self::from_slices(
            gate_up_bytes,
            down_bytes,
            experts,
            gate_up_stride,
            down_stride,
        )
    }

    /// Validate extents and cache the slices' addresses. Zero experts or a zero
    /// stride is refused (an empty expert is not a CPU-multipliable payload).
    fn from_slices(
        gate_up: &[u8],
        down: &[u8],
        experts: usize,
        gate_up_stride: usize,
        down_stride: usize,
    ) -> Option<Self> {
        if experts == 0 || gate_up_stride == 0 || down_stride == 0 {
            return None;
        }
        let gate_up_total = experts.checked_mul(gate_up_stride)?;
        let down_total = experts.checked_mul(down_stride)?;
        if gate_up_total > gate_up.len() || down_total > down.len() {
            return None;
        }
        Some(Self {
            gate_up_ptr: gate_up.as_ptr() as usize,
            gate_up_len: gate_up.len(),
            down_ptr: down.as_ptr() as usize,
            down_len: down.len(),
            experts,
            gate_up_stride,
            down_stride,
        })
    }

    /// Checked byte ranges of `expert_idx` in the gate/up and down aggregates.
    fn ranges(
        &self,
        expert_idx: usize,
    ) -> Option<(std::ops::Range<usize>, std::ops::Range<usize>)> {
        let gate_up = expert_range(expert_idx, self.experts, self.gate_up_stride)?;
        let down = expert_range(expert_idx, self.experts, self.down_stride)?;
        if gate_up.end > self.gate_up_len || down.end > self.down_len {
            return None;
        }
        Some((gate_up, down))
    }

    /// Borrowed CPU payload for one expert, `None` for an out-of-range expert or
    /// a dtype without a CPU executor.
    pub(super) fn payload(
        &self,
        expert_idx: usize,
        gate_up_dtype: DType,
        down_dtype: DType,
    ) -> Option<hipfire_dispatch::cpu_exec::CpuMoeExpert<'_>> {
        let (gate_up_range, down_range) = self.ranges(expert_idx)?;
        let gate_up_quant = hipfire_dispatch::cpu_exec::cpu_quant_for(gate_up_dtype)?;
        let down_quant = hipfire_dispatch::cpu_exec::cpu_quant_for(down_dtype)?;
        // SAFETY: `gate_up_ptr`/`down_ptr` are the addresses of the OWNING
        // host-mapped allocations' pinned bytes (`Gpu::host_bytes`), of
        // `*_len` bytes, and `ranges` has checked both extents lie inside
        // them. The allocations live in the bundle's weights, which outlive
        // the private runtime holding this cache (the runtime is torn down
        // first, then the weights free their host allocations), and no
        // aggregate owner is replaced after the forward attaches — the same
        // cached-address pattern as the host arena sources in `slab_stage.rs`.
        // The returned slices are shared reads only and borrow `self`, so they
        // cannot outlive this cache.
        let (gate_up, down) = unsafe {
            (
                std::slice::from_raw_parts(
                    (self.gate_up_ptr + gate_up_range.start) as *const u8,
                    gate_up_range.len(),
                ),
                std::slice::from_raw_parts(
                    (self.down_ptr + down_range.start) as *const u8,
                    down_range.len(),
                ),
            )
        };
        Some(hipfire_dispatch::cpu_exec::CpuMoeExpert {
            gate_up_quant,
            gate_up,
            down_quant,
            down,
        })
    }
}

/// Whether the D2 split may arm for one layer. Distinct from
/// [`CpuSpliceAdmission`]: the whole-layer D1 splice never arms on a tier
/// layer, and this mode never sets the layer's `cpu_deferred_experts` flag.
/// One decision, several span-owner kinds ([`SplitOwnerKind`]): a hot-set tier
/// layer, an un-tiered host-mapped aggregate layer, or a host-mapped
/// expert-slab arena. Every field is an input;
/// [`Self::refusal`] is the single decision and [`Self::armed`] its yes/no.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct CpuSplitAdmission {
    /// The D2 split was requested for this layer: `memory.offload_exec=cpu`
    /// (tier layers only), or the per-load measured default
    /// ([`crate::d2_default`]) with the setting unset.
    pub enabled: bool,
    /// Which owner kind the layer would read its cold experts from; `None`
    /// when it has neither a tier plan, a host-mapped aggregate the D1 splice
    /// has not already taken, nor a host-mapped slab arena.
    pub kind: Option<SplitOwnerKind>,
    /// The tier plan came from a hot-set sidecar: the loader bound that
    /// sidecar's SHA-256 to the lease before attach. A custom tier plan
    /// without a bound sidecar never arms. Consulted for tier layers only.
    pub bound_hotset: bool,
    /// How many of the layer's experts are host-mapped (cold). Zero leaves
    /// nothing for the CPU to do.
    pub host_experts: usize,
    /// Flash-Next profile.
    pub flash_next: bool,
    /// Every host-placed expert's gate/up and down formats have a CPU decoder
    /// (`hipfire_dispatch::cpu_exec::cpu_quant_for`).
    pub cpu_executed: bool,
    /// The layer is routed through expert parallelism.
    pub ep: bool,
    /// The layer carries AWQ scales (a bare CPU gemv would ignore them).
    pub awq: bool,
    /// Stock FP8 blocks.
    pub fp8: bool,
    /// Long-prefill expert staging would take a window of at most
    /// [`hipfire_dispatch::cpu_split::CPU_SPLIT_MAX_ROWS`] rows
    /// (`HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS` forced that low): a staged
    /// table is a different pointer set than the masked route's, so the split
    /// would be refused per call instead of served.
    pub narrow_stage: bool,
}

/// Why the D2 split does not arm on a layer. The first three are the normal
/// shape of a layer outside the mode (never logged); every other reason is a
/// route the split has no masked-route or cold-slot installation for, logged
/// once per process by [`log_split_refusal_once`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SplitRefusal {
    NotRequested,
    NoSpanOwner,
    NoColdExperts,
    UnboundHotset,
    NotFlashNext,
    NotCpuExecuted,
    ExpertParallel,
    Awq,
    Fp8,
    NarrowStage,
}

impl SplitRefusal {
    /// A layer outside the mode, not a refused route.
    pub(super) fn is_silent(self) -> bool {
        matches!(
            self,
            Self::NotRequested | Self::NoSpanOwner | Self::NoColdExperts
        )
    }

    fn reason(self) -> &'static str {
        match self {
            Self::NotRequested => "the CPU split is not requested",
            Self::NoSpanOwner => {
                "the layer has neither a hot-set tier plan, a host-mapped aggregate nor a \
                 host-mapped expert-slab arena"
            }
            Self::NoColdExperts => "the layer has no host-mapped expert",
            Self::UnboundHotset => "the tier plan is not backed by a bound hot-set sidecar",
            Self::NotFlashNext => "the layer is not a Flash-Next profile layer",
            Self::NotCpuExecuted => "a host-placed expert's format has no CPU decoder",
            Self::ExpertParallel => "expert-parallel layers have no masked route",
            Self::Awq => "AWQ layers carry scales the CPU arithmetic would ignore",
            Self::Fp8 => "stock-FP8 layers have no masked route or cold-slot installation",
            Self::NarrowStage => {
                "HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS makes long-prefill staging take windows of \
                 at most 4 rows, and a staged expert table has no masked route"
            }
        }
    }
}

impl CpuSplitAdmission {
    /// The first reason the split does not arm, `None` when it arms.
    pub(super) fn refusal(&self) -> Option<SplitRefusal> {
        let checks = [
            (!self.enabled, SplitRefusal::NotRequested),
            (self.kind.is_none(), SplitRefusal::NoSpanOwner),
            (self.host_experts == 0, SplitRefusal::NoColdExperts),
            (
                self.kind == Some(SplitOwnerKind::Tier) && !self.bound_hotset,
                SplitRefusal::UnboundHotset,
            ),
            (!self.flash_next, SplitRefusal::NotFlashNext),
            (!self.cpu_executed, SplitRefusal::NotCpuExecuted),
            (self.ep, SplitRefusal::ExpertParallel),
            (self.awq, SplitRefusal::Awq),
            (self.fp8, SplitRefusal::Fp8),
            (self.narrow_stage, SplitRefusal::NarrowStage),
        ];
        checks
            .into_iter()
            .find_map(|(refused, reason)| refused.then_some(reason))
    }

    pub(super) fn armed(&self) -> bool {
        self.refusal().is_none()
    }
}

/// Log a refused (non-silent) split route exactly once per process and reason,
/// so a 48-layer hot set reports one line, not 48. Silent reasons never log.
pub(super) fn log_split_refusal_once(refusal: SplitRefusal, arch: &str) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEEN: AtomicU32 = AtomicU32::new(0);
    if refusal.is_silent() {
        return;
    }
    let bit = 1u32 << (refusal as u32);
    if SEEN.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
        eprintln!("{}", split_refusal_line(refusal, arch));
    }
}

/// The one log line of a refused split route.
fn split_refusal_line(refusal: SplitRefusal, arch: &str) -> String {
    format!(
        "  [moe-cpu-split] qwen4: not armed on {arch}: {}; the affected hot-set layers keep the \
         PCIe mapped-expert read",
        refusal.reason()
    )
}

/// Validated byte ranges of one cold expert inside the host owner.
type HostExpertRanges = (std::ops::Range<usize>, std::ops::Range<usize>);

/// Resolve `(offset, len)` relative to the host owner into a checked range:
/// `len` must be exactly the expected per-expert extent and the range must lie
/// inside the owner.
fn host_span(
    offset: u64,
    len: u64,
    expected: usize,
    host_len: usize,
) -> Option<std::ops::Range<usize>> {
    if expected == 0 || usize::try_from(len).ok()? != expected {
        return None;
    }
    let start = usize::try_from(offset).ok()?;
    let end = start.checked_add(expected)?;
    (end <= host_len).then_some(start..end)
}

/// Per-expert cold ranges of one layer: `Some` for every host-mapped expert,
/// `None` for a device-resident one (the CPU never reads those). Pure: the
/// first unresolvable host expert is reported by index.
fn host_range_table(
    entries: impl Iterator<Item = (bool, u64, u64, u64, u64)>,
    gate_up_bytes: usize,
    down_bytes: usize,
    host_len: usize,
) -> Result<Vec<Option<HostExpertRanges>>, String> {
    entries
        .enumerate()
        .map(|(expert, (host, gu_off, gu_len, dn_off, dn_len))| {
            if !host {
                return Ok(None);
            }
            let gate_up = host_span(gu_off, gu_len, gate_up_bytes, host_len)
                .ok_or_else(|| format!("expert {expert} gate/up span is outside the host owner"))?;
            let down = host_span(dn_off, dn_len, down_bytes, host_len)
                .ok_or_else(|| format!("expert {expert} down span is outside the host owner"))?;
            if gate_up.start < down.end && down.start < gate_up.end {
                return Err(format!("expert {expert} gate/up and down spans overlap"));
            }
            Ok(Some((gate_up, down)))
        })
        .collect()
}

/// The CPU arm's ownership mask over one tier layer: `1` for a cold expert the
/// GPU arm does not own, `0` otherwise. An empty `pcie` (no share) and an
/// all-`false` one both give `1` for every cold expert, the mask before the
/// share existed.
pub(super) fn cpu_owned_mask(cold: &[bool], pcie: &[bool]) -> Vec<u8> {
    cold.iter()
        .enumerate()
        .map(|(expert, &is_cold)| u8::from(is_cold && !pcie.get(expert).copied().unwrap_or(false)))
        .collect()
}

/// Which cold experts of one tier layer the GPU arm reads over PCIe instead of
/// the CPU arm computing them: a deterministic partition by routing mass.
///
/// `cold[e]` marks the host-mapped experts and `mass[e]` is every expert's
/// routing count (the hot-set sidecar's `counts`); `ppm` is the fraction of the
/// cold mass the GPU arm takes, in parts per million. Cold experts are
/// visited in cold rank order (mass descending, expert id ascending), each
/// weighted `mass + 1` (an expert the calibration never routed still counts, so
/// the rule balances per-call counts rather than degenerating on zero mass).
/// An expert goes to the GPU arm exactly when doing so leaves the running GPU
/// mass nearer the running target `ppm * (mass visited so far)` than leaving
/// it on the CPU arm, i.e. when `assigned + w / 2 < target` (strict). That
/// error-diffusion rule spreads the GPU-owned experts evenly across the cold
/// rank order (a `0.5` share alternates CPU, GPU, CPU, GPU, ...; no contiguous
/// block of the hottest or the coldest), keeps the GPU-owned mass within half
/// the heaviest expert of `ppm * (cold mass)`, and is exact integer arithmetic
/// (`u128`), so the same inputs give the same partition on every host.
/// `ppm == 0` owns nothing on the GPU arm (every cold expert stays CPU-owned:
/// today's mask, byte for byte) and `ppm >= 1_000_000` owns every cold expert.
/// A non-cold expert is never GPU-owned. `mass` shorter than `cold` is treated
/// as zero-mass past its end.
pub(super) fn pcie_partition(cold: &[bool], mass: &[u64], ppm: u32) -> Vec<bool> {
    const ONE: u128 = hipfire_config::memory::D2_PCIE_SHARE_PPM_ONE as u128;
    let ppm = u128::from(ppm).min(ONE);
    let weight = |expert: usize| u128::from(mass.get(expert).copied().unwrap_or(0)) + 1;
    let mut order: Vec<usize> = (0..cold.len()).filter(|&e| cold[e]).collect();
    order.sort_by(|&a, &b| weight(b).cmp(&weight(a)).then(a.cmp(&b)));
    let mut pcie = vec![false; cold.len()];
    let (mut visited, mut assigned) = (0u128, 0u128);
    for expert in order {
        let w = weight(expert);
        visited += w;
        // assigned + w/2 < ppm * visited / ONE, scaled by 2 * ONE.
        if 2 * assigned * ONE + w * ONE < 2 * ppm * visited {
            pcie[expert] = true;
            assigned += w;
        }
    }
    pcie
}

/// The fraction (`0..=1`) of a layer's routing mass its cold experts carry.
/// With no routing mass at all (`mass` empty or all zero) the cold fraction of
/// the expert count.
pub(super) fn cold_mass_fraction(cold: &[bool], mass: &[u64]) -> f64 {
    let (mut cold_sum, mut all_sum) = (0u128, 0u128);
    for (expert, &is_cold) in cold.iter().enumerate() {
        let m = u128::from(mass.get(expert).copied().unwrap_or(0));
        all_sum += m;
        if is_cold {
            cold_sum += m;
        }
    }
    if all_sum == 0 {
        let cold_count = cold.iter().filter(|&&c| c).count();
        return if cold.is_empty() {
            0.0
        } else {
            cold_count as f64 / cold.len() as f64
        };
    }
    cold_sum as f64 / all_sum as f64
}

/// CPU views of one tier layer's COLD (host-mapped) experts.
///
/// Addresses come from the lease's `host_cpu_range` (the pinned host owner's
/// CPU address) plus each tier entry's owner-relative spans, never from the
/// mapped device alias a device pointer table carries. The lease is retained
/// here, so the owner outlives every returned slice.
///
/// Every cold expert keeps its CPU span (the parity lab's all-cold endpoint
/// reads any of them); [`Self::placement`] names the subset the CPU arm owns,
/// which is every cold expert unless [`Self::with_pcie_share`] gave a share of
/// them to the GPU arm.
pub(super) struct CpuTierOwners {
    /// Keeps the immutable CPU owner mapping alive.
    _lease: crate::expert_tier_owners::ExpertTierLease,
    layer: usize,
    host_base: usize,
    ranges: Vec<Option<HostExpertRanges>>,
    /// Cold experts the GPU arm reads over PCIe; empty when the CPU arm owns
    /// every cold expert.
    pcie: Vec<bool>,
    /// The PCIe share this layer was partitioned with (parts per million).
    share_ppm: u32,
    /// How the layer's share was sized (`auto`'s predicted times).
    share_plan: Option<crate::d2_default::LayerSharePlan>,
}

impl CpuTierOwners {
    /// Resolve every host-placed expert of trunk layer `layer`. `Err` names the
    /// first span that cannot be resolved; the caller keeps that layer on PCIe.
    pub(super) fn new(
        lease: &crate::expert_tier_owners::ExpertTierLease,
        layer: usize,
        experts: usize,
        gate_up_bytes: usize,
        down_bytes: usize,
    ) -> Result<Self, String> {
        use crate::expert_tier_plan::Placement;
        let view = lease
            .layer(layer)
            .ok_or_else(|| format!("layer {layer} is not a tier layer"))?;
        let (host_base, host_len) = lease
            .host_cpu_range()
            .ok_or_else(|| "the tier plan has no pinned host owner".to_string())?;
        let entries = (0..experts)
            .map(|eid| {
                view.entry(eid).map(|entry| {
                    (
                        entry.tier == Placement::HostMapped,
                        entry.gate_up_offset,
                        entry.gate_up_len,
                        entry.down_offset,
                        entry.down_len,
                    )
                })
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| format!("layer {layer} has fewer than {experts} tier entries"))?;
        let ranges = host_range_table(entries.into_iter(), gate_up_bytes, down_bytes, host_len)?;
        Ok(Self {
            _lease: lease.clone(),
            layer,
            host_base,
            ranges,
            pcie: Vec::new(),
            share_ppm: 0,
            share_plan: None,
        })
    }

    /// Give a share of this layer's cold experts to the GPU arm: the plan's
    /// share for this layer ([`crate::d2_default::PcieShare::plan_layer`]: fixed,
    /// or priced from the layer's own routing counts), partitioned by
    /// [`pcie_partition`] over the lease's sidecar counts (uniform mass when
    /// none are bound). A share of `0` leaves the owner exactly as
    /// [`Self::new`] built it.
    pub(super) fn with_pcie_share(mut self, share: crate::d2_default::PcieShare) -> Self {
        let uniform;
        let counts = match self._lease.hotset_layer_counts(self.layer) {
            Some(counts) => counts,
            None => {
                uniform = vec![1u64; self.ranges.len()];
                &uniform
            }
        };
        let cold: Vec<bool> = self.ranges.iter().map(Option::is_some).collect();
        let plan = share.plan_layer(&cold, counts, &|ppm| pcie_partition(&cold, counts, ppm));
        self.share_plan = Some(plan);
        if plan.ppm == 0 {
            return self;
        }
        self.pcie = pcie_partition(&cold, counts, plan.ppm);
        self.share_ppm = plan.ppm;
        self
    }

    /// The share plan this layer was sized with, `None` before
    /// [`Self::with_pcie_share`].
    pub(super) fn share_plan(&self) -> Option<crate::d2_default::LayerSharePlan> {
        self.share_plan
    }

    /// Borrowed CPU payload of one cold expert; `None` for a device-resident
    /// or out-of-range expert, or a dtype without a CPU executor.
    pub(super) fn payload(
        &self,
        expert_idx: usize,
        gate_up_dtype: DType,
        down_dtype: DType,
    ) -> Option<hipfire_dispatch::cpu_exec::CpuMoeExpert<'_>> {
        let (gate_up_range, down_range) = self.ranges.get(expert_idx)?.as_ref()?;
        let gate_up_quant = hipfire_dispatch::cpu_exec::cpu_quant_for(gate_up_dtype)?;
        let down_quant = hipfire_dispatch::cpu_exec::cpu_quant_for(down_dtype)?;
        // SAFETY: `host_base` is the CPU address of the pinned host owner
        // (`ExpertTierLease::host_cpu_range`), whose lease this value retains
        // for as long as the slices can exist; `host_range_table` proved both
        // ranges lie inside the owner's byte length, and the owner is never
        // written after materialization. The slices are shared reads that
        // borrow `self`.
        let (gate_up, down) = unsafe {
            (
                std::slice::from_raw_parts(
                    (self.host_base + gate_up_range.start) as *const u8,
                    gate_up_range.len(),
                ),
                std::slice::from_raw_parts(
                    (self.host_base + down_range.start) as *const u8,
                    down_range.len(),
                ),
            )
        };
        Some(hipfire_dispatch::cpu_exec::CpuMoeExpert {
            gate_up_quant,
            gate_up,
            down_quant,
            down,
        })
    }

    /// Number of cold experts this layer resolved.
    pub(super) fn cold_experts(&self) -> usize {
        self.ranges.iter().flatten().count()
    }

    /// The split's per-expert ownership mask: `1` for every host-mapped (cold)
    /// expert the CPU arm computes, `0` for a device-resident one and for a
    /// cold expert the GPU arm reads over PCIe ([`Self::with_pcie_share`]).
    /// With no share this is every cold expert, as before the share existed.
    pub(super) fn placement(&self) -> Vec<u8> {
        let cold: Vec<bool> = self.ranges.iter().map(Option::is_some).collect();
        cpu_owned_mask(&cold, &self.pcie)
    }

    /// `1` for every host-mapped (cold) expert of this layer, whichever arm serves it
    /// (CPU-owned and PCIe-owned alike): the tier counters' host mask. Unlike
    /// [`Self::placement`] it ignores the PCIe share.
    pub(super) fn host_mask(&self) -> Vec<u8> {
        cpu_owned_mask(
            &self.ranges.iter().map(Option::is_some).collect::<Vec<_>>(),
            &[],
        )
    }

    /// `1` for each cold expert the GPU arm reads over PCIe; `None` when the
    /// CPU arm owns every cold expert.
    pub(super) fn pcie_owned(&self) -> Option<Vec<u8>> {
        (!self.pcie.is_empty()).then(|| self.pcie.iter().map(|&p| u8::from(p)).collect())
    }

    /// The PCIe share this layer was partitioned with, parts per million.
    pub(super) fn share_ppm(&self) -> u32 {
        self.share_ppm
    }

    /// `(CPU-owned, PCIe-owned)` cold experts of this layer.
    pub(super) fn owned_counts(&self) -> (usize, usize) {
        let pcie = self.pcie.iter().filter(|&&p| p).count();
        (self.cold_experts() - pcie, pcie)
    }

    /// Bytes of one host-mapped expert (gate/up + down; every cold expert has the
    /// same extents), `None` when the layer has no cold expert.
    fn cold_expert_bytes(&self) -> Option<usize> {
        self.ranges
            .iter()
            .flatten()
            .next()
            .map(|(gate_up, down)| gate_up.len() + down.len())
    }
}

/// The kind of host-span owner a D2 split layer reads its cold experts from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SplitOwnerKind {
    /// A hot-set tier layer: the cold experts are the plan's host-mapped
    /// entries inside the pinned tier owner.
    Tier,
    /// An un-tiered aggregate layer whose two expert allocations are
    /// host-mapped: every expert is cold.
    WholeLayer,
    /// An expert-slab layer whose arena is host-mapped: every present expert
    /// is cold, each in its own codec.
    Slab,
}

impl SplitOwnerKind {
    /// Stable spelling for logs and lab receipts.
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Tier => "tier",
            Self::WholeLayer => "whole-layer",
            Self::Slab => "slab",
        }
    }
}

/// The cold-expert byte source of one D2 split layer: borrowed views over
/// memory the loader already owns, never a copy.
pub(super) enum HostSpanOwner {
    Tier(CpuTierOwners),
    WholeLayer(CpuExpertOwners),
    Slab(super::cpu_slab_owner::CpuSlabOwners),
}

impl HostSpanOwner {
    pub(super) fn kind(&self) -> SplitOwnerKind {
        match self {
            Self::Tier(_) => SplitOwnerKind::Tier,
            Self::WholeLayer(_) => SplitOwnerKind::WholeLayer,
            Self::Slab(_) => SplitOwnerKind::Slab,
        }
    }

    /// Borrowed CPU payload of one cold expert (see the owner's `payload`).
    pub(super) fn payload(
        &self,
        expert_idx: usize,
        gate_up_dtype: DType,
        down_dtype: DType,
    ) -> Option<hipfire_dispatch::cpu_exec::CpuMoeExpert<'_>> {
        match self {
            Self::Tier(owners) => owners.payload(expert_idx, gate_up_dtype, down_dtype),
            Self::WholeLayer(owners) => owners.payload(expert_idx, gate_up_dtype, down_dtype),
            Self::Slab(owners) => owners.payload(expert_idx, gate_up_dtype, down_dtype),
        }
    }

    /// The split's per-expert ownership mask: `1` = host (CPU arm), `0` = GPU.
    pub(super) fn placement(&self) -> Vec<u8> {
        match self {
            Self::Tier(owners) => owners.placement(),
            Self::WholeLayer(owners) => vec![1; owners.experts],
            Self::Slab(owners) => owners.placement(),
        }
    }

    /// Every expert that lives in host memory, whichever arm serves it (`1`), against the
    /// device-resident ones (`0`). Equal to [`Self::placement`] while the CPU arm serves
    /// every host-resident expert; the tier counters read this one so a CPU share of
    /// the cold experts still counts the rest as PCIe-served.
    pub(super) fn host_mask(&self) -> Vec<u8> {
        match self {
            Self::Tier(owners) => owners.host_mask(),
            other => other.placement(),
        }
    }

    /// Bytes of one host-resident expert when uniform across the layer (a codec-mixed
    /// slab arena is not).
    pub(super) fn cold_expert_bytes(&self) -> Option<usize> {
        match self {
            Self::Tier(owners) => owners.cold_expert_bytes(),
            Self::WholeLayer(owners) => Some(owners.gate_up_stride + owners.down_stride),
            Self::Slab(_) => None,
        }
    }

    /// Experts the CPU arm serves on this layer.
    pub(super) fn cold_experts(&self) -> usize {
        match self {
            Self::Tier(owners) => owners.cold_experts(),
            Self::WholeLayer(owners) => owners.experts,
            Self::Slab(owners) => owners.cold_experts(),
        }
    }

    /// Apply the load's PCIe share to a hot-set tier owner (the only kind
    /// that has device-resident experts to balance against); every other owner
    /// kind is returned unchanged.
    pub(super) fn with_pcie_share(self, share: crate::d2_default::PcieShare) -> Self {
        match self {
            Self::Tier(owners) => Self::Tier(owners.with_pcie_share(share)),
            other => other,
        }
    }

    /// Cold experts the GPU arm reads over PCIe (`1` per expert id); `None`
    /// when the CPU arm owns every cold expert.
    pub(super) fn pcie_owned(&self) -> Option<Vec<u8>> {
        match self {
            Self::Tier(owners) => owners.pcie_owned(),
            Self::WholeLayer(_) | Self::Slab(_) => None,
        }
    }

    /// `(CPU-owned, PCIe-owned)` cold experts and the layer's share in parts
    /// per million. Un-tiered owner kinds are all CPU-owned.
    pub(super) fn share_counts(&self) -> (usize, usize, u32) {
        match self {
            Self::Tier(owners) => {
                let (cpu, pcie) = owners.owned_counts();
                (cpu, pcie, owners.share_ppm())
            }
            other => (other.cold_experts(), 0, 0),
        }
    }

    /// How a tier layer's share was sized; `None` for other owner kinds.
    pub(super) fn share_plan(&self) -> Option<crate::d2_default::LayerSharePlan> {
        match self {
            Self::Tier(owners) => owners.share_plan(),
            _ => None,
        }
    }
}

/// What names one host layer's cold experts to [`ColdLayerSource`]: the layer's
/// aggregate expert tensors, or its slab arena.
#[derive(Clone, Copy)]
pub(crate) enum ColdLayerRefs<'w> {
    /// Aggregate gate/up and down expert references (a hot-set tier layer reads
    /// only the layer index; an un-tiered layer reads both tensors).
    Aggregate {
        gate_up: &'w TensorRef,
        down: &'w TensorRef,
    },
    /// The expert-slab layer's one owning arena.
    Arena(&'w TensorRef),
}

/// How a source's experts are formatted.
enum ColdCodecs {
    /// Every expert holds one `(gate_up, down)` codec pair read in place: the
    /// GPU reads exactly the CPU payload bytes.
    Uniform(DType, DType),
    /// Each slab expert carries its own codec pair and its own GPU read bytes
    /// ([`cpu_slab_owner::gpu_read_bytes`], one entry per expert, `None` when
    /// absent).
    Slab { gpu_read_bytes: Vec<Option<u64>> },
}

/// The cold-expert byte source of one host layer, resolved from the loaded
/// weights exactly as the layer build resolves it, for the D2 selector's
/// real-arm probe ([`crate::d2_default`]). It runs before the layer runtimes
/// exist, so it builds the same borrowed views ([`CpuTierOwners`] for a hot-set
/// tier layer, [`CpuExpertOwners`] for an un-tiered host-mapped aggregate,
/// [`cpu_slab_owner::CpuSlabOwners`] for a host-mapped expert-slab arena) and
/// copies nothing.
pub(crate) struct ColdLayerSource {
    owner: HostSpanOwner,
    codecs: ColdCodecs,
    experts: usize,
}

impl ColdLayerSource {
    /// Resolve trunk layer `layer`'s cold experts. An aggregate layer's codecs
    /// are the fixed QT44 gate/up + QT53 down pair for a tier layer (as in the
    /// layer build) and the resident tensors' own for an un-tiered one; a slab
    /// layer's are per expert, from the arena plan. `Err` names what could not
    /// be resolved.
    pub(crate) fn new(
        gpu: &Gpu,
        weights: &Qwen4Weights,
        config: &Qwen4Config,
        layer: usize,
        refs: ColdLayerRefs<'_>,
    ) -> Result<Self, String> {
        let experts = config.num_experts;
        let (gate_up, down) = match refs {
            ColdLayerRefs::Arena(arena_ref) => {
                return Self::slab(gpu, weights, config, layer, arena_ref)
            }
            ColdLayerRefs::Aggregate { gate_up, down } => (gate_up, down),
        };
        let extents = |gate_up_dtype: DType, down_dtype: DType| {
            let gate_up = crate::projection::checked_bytes(
                2 * config.moe_intermediate_size,
                row_stride(gate_up_dtype, config.hidden_size),
            );
            let down = crate::projection::checked_bytes(
                config.hidden_size,
                row_stride(down_dtype, config.moe_intermediate_size),
            );
            gate_up
                .zip(down)
                .ok_or_else(|| format!("layer {layer}: expert byte extent overflows"))
        };
        let (gate_up_dtype, down_dtype, owner) = match weights.expert_tier_layer(layer) {
            Some(view) => {
                let (gate_up_dtype, down_dtype) = (DType::MQ4G256V2, DType::MQ4G128V2);
                let (gate_up_bytes, down_bytes) = extents(gate_up_dtype, down_dtype)?;
                let owners = CpuTierOwners::new(
                    &view.lease(),
                    view.layer(),
                    experts,
                    gate_up_bytes,
                    down_bytes,
                )?;
                (gate_up_dtype, down_dtype, HostSpanOwner::Tier(owners))
            }
            None => {
                let gate_up = weights
                    .resident(gate_up)
                    .map_err(|error| format!("layer {layer}: {error}"))?;
                let down = weights
                    .resident(down)
                    .map_err(|error| format!("layer {layer}: {error}"))?;
                let (gate_up_bytes, down_bytes) = extents(gate_up.dtype, down.dtype)?;
                let owners =
                    CpuExpertOwners::new(gpu, gate_up, down, experts, gate_up_bytes, down_bytes)
                        .ok_or_else(|| {
                            format!(
                        "layer {layer}: the aggregate expert allocations have no resolvable \
                         pinned host bytes"
                    )
                        })?;
                (gate_up.dtype, down.dtype, HostSpanOwner::WholeLayer(owners))
            }
        };
        Ok(Self {
            owner,
            codecs: ColdCodecs::Uniform(gate_up_dtype, down_dtype),
            experts,
        })
    }

    /// A host-mapped expert-slab layer: the owning arena's pinned bytes and
    /// each expert's own table entry, from the same plan the layer build reads
    /// ([`cpu_slab_owner::plan_entries`]).
    fn slab(
        gpu: &Gpu,
        weights: &Qwen4Weights,
        config: &Qwen4Config,
        layer: usize,
        arena_ref: &TensorRef,
    ) -> Result<Self, String> {
        let plan = match &weights.manifest.identity.recipe {
            crate::weights::Qwen4TierRecipe::FlashNextSlab { arenas, .. } => arenas
                .layer(layer)
                .ok_or_else(|| format!("layer {layer}: the expert arena plan has no such layer"))?,
            _ => {
                return Err(format!(
                    "layer {layer}: an expert arena under a non-slab recipe"
                ))
            }
        };
        let (hidden, mi) = (config.hidden_size, config.moe_intermediate_size);
        // The plan's per-format extents assume the frozen slab expert geometry,
        // the same guard the layer build applies.
        if config.num_experts != plan.experts.len() || [2 * mi, hidden, mi] != [1280, 2560, 640] {
            return Err(format!(
                "layer {layer}: the expert slab geometry is 512 experts of [1280,2560] / [2560,640]"
            ));
        }
        let arena = weights
            .resident(arena_ref)
            .map_err(|error| format!("layer {layer}: {error}"))?;
        if arena.buf.size() < plan.bytes {
            return Err(format!(
                "layer {layer}: the arena holds {} bytes, its plan needs {}",
                arena.buf.size(),
                plan.bytes
            ));
        }
        let entries = cpu_slab_owner::plan_entries(plan, config.kept_expert_ids(layer, false))
            .map_err(|error| format!("layer {layer}: {error}"))?;
        let owners =
            cpu_slab_owner::CpuSlabOwners::new(gpu, arena, plan.bytes, entries.into_iter())
                .map_err(|error| format!("layer {layer}: {error}"))?;
        Ok(Self {
            owner: HostSpanOwner::Slab(owners),
            codecs: ColdCodecs::Slab {
                gpu_read_bytes: cpu_slab_owner::gpu_read_bytes(plan),
            },
            experts: config.num_experts,
        })
    }

    /// The distinct resident `(gate/up, down)` codec pairs of the layer's cold
    /// experts, sorted: one pair for an aggregate layer, every pair a present
    /// slab expert holds for a slab layer.
    pub(crate) fn codecs(&self) -> Vec<(DType, DType)> {
        let mut pairs = match (&self.codecs, &self.owner) {
            (ColdCodecs::Uniform(gate_up, down), _) => vec![(*gate_up, *down)],
            (ColdCodecs::Slab { .. }, HostSpanOwner::Slab(owners)) => (0..self.experts)
                .filter_map(|expert| owners.expert_formats(expert))
                .map(|[(gate_up, _), (down, _)]| (gate_up, down))
                .collect(),
            (ColdCodecs::Slab { .. }, _) => Vec::new(),
        };
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    /// Experts the layer addresses (cold or not).
    pub(crate) fn experts(&self) -> usize {
        self.experts
    }

    /// Cold expert `expert` as a probe slot: its borrowed CPU payload in its
    /// own codecs and the bytes the GPU reads for it over the link. An
    /// aggregate layer's experts are one codec read in place (the GPU reads the
    /// CPU payload bytes); a slab slot's GPU read is its own plan format's
    /// gate/up plus down bytes, resolved independently of its CPU codec and
    /// span lengths. `None` for a device-resident, absent or out-of-range expert or a
    /// codec with no CPU executor.
    pub(crate) fn cold_probe(&self, expert: usize) -> Option<crate::d2_default::ProbeExpert<'_>> {
        match (&self.codecs, &self.owner) {
            (ColdCodecs::Uniform(gate_up, down), owner) => owner
                .payload(expert, *gate_up, *down)
                .map(crate::d2_default::ProbeExpert::same_bytes),
            (ColdCodecs::Slab { gpu_read_bytes }, HostSpanOwner::Slab(owners)) => {
                owners.probe_expert(expert, (*gpu_read_bytes.get(expert)?)?)
            }
            (ColdCodecs::Slab { .. }, _) => None,
        }
    }
}

/// Hex of a layer bitmap: bit `i` is layer label `i`, rendered as one big
/// number (most significant byte first), `"0"` when empty. Pure.
pub(super) fn layer_bitmap_hex(labels: &[usize]) -> String {
    let Some(&max) = labels.iter().max() else {
        return "0".to_string();
    };
    let mut bytes = vec![0u8; max / 8 + 1];
    for &label in labels {
        bytes[label / 8] |= 1 << (label % 8);
    }
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes.iter().rev() {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// The frozen numerical identity of the armed host arithmetic (see
/// `Qwen4GpuForward::host_exec_identity`): empty when nothing is armed,
/// `cpu-all-v1/<bitmap>` for D1 layers, then one part per D2 owner kind in the
/// caller's order: `cpu-tier-split-v1/<64-hex tier logical digest>/<bitmap>/
/// hotset-sha256=<hex | unbound>` for hot-set tier layers,
/// `cpu-whole-split-v1/<bitmap>` for un-tiered whole-layer host aggregates and
/// `cpu-slab-split-v1/<bitmap>` for host-mapped expert-slab arenas (no tier
/// digest or sidecar: every present host expert is cold and its codec is the
/// artifact's, so there is no plan to freeze). Parts are joined with `+`.
/// `tier` is the tier-split layers' digest and bound sidecar SHA; a tier layer
/// set without it is not an identity and yields `unbound` digest zeros rather
/// than silently collapsing into another mode. `pcie_share` is the
/// [`pcie_share_digest`] of a load whose tier layers gave some cold experts to
/// the GPU arm: it appends `/pcie-share-v1=<hex>` to the tier part (those
/// experts run the GPU arithmetic instead of the CPU arm's, a different
/// numerical domain per partition). `None` leaves the string exactly as it was
/// before the share existed.
pub(super) fn host_exec_identity(
    cpu_all: &[usize],
    splits: &[(SplitOwnerKind, &[usize])],
    tier: Option<(&[u8; 32], Option<&str>)>,
    pcie_share: Option<&str>,
) -> String {
    let mut parts = Vec::new();
    if !cpu_all.is_empty() {
        parts.push(format!("cpu-all-v1/{}", layer_bitmap_hex(cpu_all)));
    }
    for &(kind, layers) in splits {
        if layers.is_empty() {
            continue;
        }
        match kind {
            SplitOwnerKind::Tier => {
                let zero = [0u8; 32];
                let (digest, sidecar) = tier.unwrap_or((&zero, None));
                let mut digest_hex = String::with_capacity(64);
                for byte in digest {
                    use std::fmt::Write as _;
                    let _ = write!(digest_hex, "{byte:02x}");
                }
                parts.push(format!(
                    "cpu-tier-split-v1/{digest_hex}/{}/hotset-sha256={}{}",
                    layer_bitmap_hex(layers),
                    sidecar.unwrap_or("unbound"),
                    pcie_share
                        .map(|share| format!("/pcie-share-v1={share}"))
                        .unwrap_or_default(),
                ));
            }
            SplitOwnerKind::WholeLayer => {
                parts.push(format!("cpu-whole-split-v1/{}", layer_bitmap_hex(layers)));
            }
            SplitOwnerKind::Slab => {
                parts.push(format!("cpu-slab-split-v1/{}", layer_bitmap_hex(layers)));
            }
        }
    }
    parts.join("+")
}

/// The identity token of a PCIe-share partition: SHA-256 over every armed tier
/// layer's `(label, GPU-owned cold-expert mask)` in layer order, 32 hex
/// digits. `layers` holds only the layers that gave experts to the GPU arm; no
/// such layer is `None`, which keeps the identity string exactly as it was
/// without a share. The digest names the partition itself, not the knob that
/// produced it, so a measured (`auto`) share never aliases another partition.
pub(super) fn pcie_share_digest(layers: &[(usize, &[u8])]) -> Option<String> {
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;
    if layers.is_empty() {
        return None;
    }
    let mut hash = Sha256::new();
    hash.update(b"hipfire.d2.pcie_share.v1");
    for &(label, mask) in layers {
        hash.update((label as u64).to_le_bytes());
        hash.update((mask.len() as u64).to_le_bytes());
        hash.update(mask);
    }
    let digest = hash.finalize();
    let mut hex = String::with_capacity(32);
    for byte in &digest[..16] {
        let _ = write!(hex, "{byte:02x}");
    }
    Some(hex)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real per-expert extents (QT44 / QT53 strides), 512 experts.
    const STRIDE_A: usize = 1_740_800;
    const STRIDE_B: usize = 870_400;
    const EXPERTS: usize = 512;

    fn admission(
        enabled: bool,
        host_mapped: bool,
        ep: bool,
        slab: bool,
        tier: bool,
        awq: bool,
        supported: bool,
    ) -> CpuSpliceAdmission {
        CpuSpliceAdmission {
            enabled,
            host_mapped,
            ep,
            slab,
            tier,
            awq,
            supported,
        }
    }

    #[test]
    fn narrow_verify_window_is_four_rows() {
        assert_eq!(NARROW_VERIFY_MAX_ROWS, 4);
    }

    #[test]
    fn arming_decision_table() {
        // (name, admission, armed)
        let table = [
            (
                "enabled host supported",
                admission(true, true, false, false, false, false, true),
                true,
            ),
            (
                "disabled",
                admission(false, true, false, false, false, false, true),
                false,
            ),
            (
                "ep",
                admission(true, true, true, false, false, false, true),
                false,
            ),
            (
                "slab",
                admission(true, true, false, true, false, false, true),
                false,
            ),
            (
                "tier",
                admission(true, true, false, false, true, false, true),
                false,
            ),
            (
                "awq",
                admission(true, true, false, false, false, true, true),
                false,
            ),
            (
                "unsupported",
                admission(true, true, false, false, false, false, false),
                false,
            ),
            (
                "resident",
                admission(true, false, false, false, false, false, true),
                false,
            ),
            (
                "all exclusions",
                admission(true, true, true, true, true, true, true),
                false,
            ),
            ("nothing set", CpuSpliceAdmission::default(), false),
        ];
        for (name, a, armed) in table {
            assert_eq!(a.armed(), armed, "{name}");
        }
    }

    #[test]
    fn arming_is_exhaustive_over_all_inputs() {
        for bits in 0u32..128 {
            let b = |i: u32| bits & (1 << i) != 0;
            let a = admission(b(0), b(1), b(2), b(3), b(4), b(5), b(6));
            // Exactly one assignment arms: enabled | host_mapped | supported,
            // no exclusion bit.
            assert_eq!(a.armed(), bits == 0b100_0011, "bits={bits:#09b}");
        }
    }

    #[test]
    fn default_pcie_disabled_arms_no_layer() {
        // Default (PCIe) mode: the splice is not selected, so none of the 48
        // layers arms, however host-mapped and supported they are.
        const LAYERS: usize = 48;
        let disabled = admission(false, true, false, false, false, false, true);
        assert_eq!((0..LAYERS).filter(|_| disabled.armed()).count(), 0);
        // Control: the same layers arm once enabled.
        let enabled = admission(true, true, false, false, false, false, true);
        assert_eq!((0..LAYERS).filter(|_| enabled.armed()).count(), LAYERS);
    }

    #[test]
    fn expert_range_ids_0_1_511_at_real_strides() {
        for stride in [STRIDE_A, STRIDE_B] {
            for idx in [0usize, 1, 511] {
                let r = expert_range(idx, EXPERTS, stride)
                    .unwrap_or_else(|| panic!("stride {stride} idx {idx}"));
                assert_eq!(r.start, idx * stride, "stride {stride} idx {idx}");
                assert_eq!(r.end, (idx + 1) * stride, "stride {stride} idx {idx}");
                assert_eq!(r.len(), stride, "stride {stride} idx {idx}");
                assert!(r.end <= EXPERTS * stride, "stride {stride} idx {idx}");
            }
            // Last expert ends exactly at the aggregate end.
            assert_eq!(
                expert_range(EXPERTS - 1, EXPERTS, stride).unwrap().end,
                EXPERTS * stride
            );
            assert!(expert_range(EXPERTS, EXPERTS, stride).is_none());
        }
        assert_eq!(
            expert_range(511, EXPERTS, STRIDE_A).unwrap().start,
            889_548_800
        );
        assert_eq!(
            expert_range(511, EXPERTS, STRIDE_B).unwrap().start,
            444_774_400
        );
        assert_eq!(EXPERTS * STRIDE_A, 891_289_600);
        assert_eq!(EXPERTS * STRIDE_B, 445_644_800);
    }

    #[test]
    fn expert_range_rejects_out_of_range_and_overflow() {
        assert!(expert_range(512, EXPERTS, STRIDE_A).is_none());
        assert!(expert_range(usize::MAX, EXPERTS, STRIDE_A).is_none());
        assert!(expert_range(0, 0, STRIDE_A).is_none());
        // idx * stride overflows.
        assert!(expert_range(3, usize::MAX, usize::MAX / 2).is_none());
        // start fits but start + stride overflows.
        assert!(expert_range(1, 2, usize::MAX - 1).is_none());
    }

    // Small fake layout: 512 experts, stride 64 (gate/up) and 32 (down), each
    // expert filled with a byte derived from its id so the resolved subspan
    // identity is observable. No large allocation.
    const FAKE_GU: usize = 64;
    const FAKE_DN: usize = 32;

    fn fake_layout() -> (Vec<u8>, Vec<u8>) {
        let mut gate_up = Vec::with_capacity(EXPERTS * FAKE_GU);
        let mut down = Vec::with_capacity(EXPERTS * FAKE_DN);
        for e in 0..EXPERTS {
            gate_up.extend(std::iter::repeat((e % 251) as u8 + 1).take(FAKE_GU));
            down.extend(std::iter::repeat(0x80 | (e % 127) as u8).take(FAKE_DN));
        }
        (gate_up, down)
    }

    #[test]
    fn payload_resolves_exact_subspans_for_ids_0_1_511() {
        let (gate_up, down) = fake_layout();
        let owners =
            CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, FAKE_GU, FAKE_DN).unwrap();
        for idx in [0usize, 1, 511] {
            let p = owners
                .payload(idx, DType::MQ4G256, DType::MQ4G256)
                .unwrap_or_else(|| panic!("idx {idx}"));
            assert_eq!(p.gate_up.len(), FAKE_GU, "idx {idx}");
            assert_eq!(p.down.len(), FAKE_DN, "idx {idx}");
            assert_eq!(
                p.gate_up.as_ptr() as usize,
                gate_up.as_ptr() as usize + idx * FAKE_GU,
                "idx {idx}"
            );
            assert_eq!(
                p.down.as_ptr() as usize,
                down.as_ptr() as usize + idx * FAKE_DN,
                "idx {idx}"
            );
            assert_eq!(
                p.gate_up,
                &gate_up[idx * FAKE_GU..(idx + 1) * FAKE_GU],
                "idx {idx}"
            );
            assert_eq!(
                p.down,
                &down[idx * FAKE_DN..(idx + 1) * FAKE_DN],
                "idx {idx}"
            );
        }
        // Distinct experts resolve to distinct, adjacent, non-overlapping bytes.
        let p0 = owners.payload(0, DType::MQ4G256, DType::MQ4G256).unwrap();
        let p1 = owners.payload(1, DType::MQ4G256, DType::MQ4G256).unwrap();
        assert_ne!(p0.gate_up[0], p1.gate_up[0]);
        assert_eq!(
            p1.gate_up.as_ptr() as usize - p0.gate_up.as_ptr() as usize,
            FAKE_GU
        );
    }

    #[test]
    fn payload_accepts_aggregate_longer_than_experts_times_stride() {
        // A mapped allocation may carry slack past the last expert.
        let (mut gate_up, mut down) = fake_layout();
        gate_up.extend([0xEE; 17]);
        down.extend([0xEE; 5]);
        let owners =
            CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, FAKE_GU, FAKE_DN).unwrap();
        let p = owners.payload(511, DType::MQ4G256, DType::MQ4G256).unwrap();
        assert_eq!(p.gate_up.len(), FAKE_GU);
        assert_eq!(p.down.len(), FAKE_DN);
        assert!(owners
            .payload(512, DType::MQ4G256, DType::MQ4G256)
            .is_none());
    }

    #[test]
    fn payload_rejects_out_of_range_expert() {
        let (gate_up, down) = fake_layout();
        let owners =
            CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, FAKE_GU, FAKE_DN).unwrap();
        for idx in [EXPERTS, EXPERTS + 1, usize::MAX] {
            assert!(
                owners
                    .payload(idx, DType::MQ4G256, DType::MQ4G256)
                    .is_none(),
                "idx {idx}"
            );
        }
    }

    #[test]
    fn payload_rejects_untyped_dtype() {
        let (gate_up, down) = fake_layout();
        let owners =
            CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, FAKE_GU, FAKE_DN).unwrap();
        // `Raw` is an untyped byte blob, never a CPU-executed weight dtype.
        assert!(owners.payload(0, DType::Raw, DType::MQ4G256).is_none());
        assert!(owners.payload(0, DType::MQ4G256, DType::Raw).is_none());
        assert!(owners.payload(0, DType::Raw, DType::Raw).is_none());
    }

    #[test]
    fn from_slices_rejects_short_owner() {
        let (gate_up, down) = fake_layout();
        // One byte short in either aggregate.
        assert!(CpuExpertOwners::from_slices(
            &gate_up[..gate_up.len() - 1],
            &down,
            EXPERTS,
            FAKE_GU,
            FAKE_DN
        )
        .is_none());
        assert!(CpuExpertOwners::from_slices(
            &gate_up,
            &down[..down.len() - 1],
            EXPERTS,
            FAKE_GU,
            FAKE_DN
        )
        .is_none());
        // Empty owners.
        assert!(CpuExpertOwners::from_slices(&[], &[], EXPERTS, FAKE_GU, FAKE_DN).is_none());
        // Strides larger than the owner can hold.
        assert!(
            CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, FAKE_GU + 1, FAKE_DN).is_none()
        );
        assert!(
            CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, FAKE_GU, FAKE_DN + 1).is_none()
        );
        // Exactly sized owners are accepted.
        assert!(CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, FAKE_GU, FAKE_DN).is_some());
    }

    #[test]
    fn from_slices_rejects_overflow_and_zero_geometry() {
        let (gate_up, down) = fake_layout();
        assert!(
            CpuExpertOwners::from_slices(&gate_up, &down, usize::MAX, FAKE_GU, FAKE_DN).is_none()
        );
        assert!(
            CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, usize::MAX, FAKE_DN).is_none()
        );
        assert!(CpuExpertOwners::from_slices(
            &gate_up,
            &down,
            EXPERTS,
            FAKE_GU,
            usize::MAX / 2 + 1
        )
        .is_none());
        assert!(CpuExpertOwners::from_slices(&gate_up, &down, 0, FAKE_GU, FAKE_DN).is_none());
        assert!(CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, 0, FAKE_DN).is_none());
        assert!(CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, FAKE_GU, 0).is_none());
    }

    fn split_admission() -> CpuSplitAdmission {
        CpuSplitAdmission {
            enabled: true,
            kind: Some(SplitOwnerKind::Tier),
            bound_hotset: true,
            host_experts: 100,
            flash_next: true,
            cpu_executed: true,
            ep: false,
            awq: false,
            fp8: false,
            narrow_stage: false,
        }
    }

    /// An un-tiered host-mapped aggregate layer: every one of its experts is
    /// host-placed, and there is no hot-set sidecar to bind.
    fn whole_layer_admission() -> CpuSplitAdmission {
        CpuSplitAdmission {
            kind: Some(SplitOwnerKind::WholeLayer),
            bound_hotset: false,
            host_experts: 512,
            ..split_admission()
        }
    }

    /// The layer admission of an `offload_exec=cpu` load on `arch` whose hot
    /// set came from the default sibling sidecar (`explicit_path = false`) or
    /// an explicit `HIPFIRE_QWEN4_EXPERT_HOTSET` path. The loader builds a tier
    /// plan (and binds its sidecar) only where `hotset_arch_admits` held; the
    /// layer has no arch input of its own.
    fn split_on(arch: &str, explicit_path: bool) -> CpuSplitAdmission {
        let loaded = crate::expert_hotset::hotset_arch_admits(arch, explicit_path);
        CpuSplitAdmission {
            kind: loaded.then_some(SplitOwnerKind::Tier),
            bound_hotset: loaded,
            ..split_admission()
        }
    }

    #[test]
    fn split_arming_truth_table_any_arch() {
        // (arch, explicit hot-set path, armed)
        for (arch, explicit_path, armed) in [
            // The default sibling sidecar builds a plan on gfx1201 and (the gfx1100 tier
            // package, default on) exact gfx1100; every other arch is byte-unchanged.
            ("gfx1201", false, true),
            ("gfx1100", false, true),
            ("gfx1101", false, false),
            ("gfx1151", false, false),
            ("gfx1200", false, false),
            ("gfx942", false, false),
            // An explicit path arms on any arch whose plan the loader admitted.
            ("gfx1201", true, true),
            ("gfx1100", true, true),
            ("gfx1151", true, true),
            ("gfx1200", true, true),
            ("gfx942", true, true),
        ] {
            assert_eq!(
                split_on(arch, explicit_path).armed(),
                armed,
                "{arch} explicit={explicit_path}"
            );
        }
        // The not-armed rows are "no span owner", never an arch refusal: an
        // un-tiered host-mapped aggregate layer arms on every arch whatever
        // the hot-set rule says.
        assert_eq!(
            split_on("gfx1151", false).refusal(),
            Some(SplitRefusal::NoSpanOwner)
        );
        for arch in ["gfx1100", "gfx1151", "gfx1201", "gfx942"] {
            let untiered = CpuSplitAdmission {
                kind: Some(SplitOwnerKind::WholeLayer),
                ..split_on(arch, false)
            };
            assert!(untiered.armed(), "{arch}");
        }
        // Without the request nothing arms on any arch.
        for arch in ["gfx1100", "gfx1201", "gfx1151"] {
            let mut a = split_on(arch, true);
            a.enabled = false;
            assert_eq!(a.refusal(), Some(SplitRefusal::NotRequested), "{arch}");
        }
    }

    #[test]
    fn split_admission_arms_only_the_bound_uniform_tier_shape() {
        assert!(split_admission().armed());
        assert_eq!(split_admission().refusal(), None);
        let refusals: [(&str, SplitRefusal, fn(&mut CpuSplitAdmission)); 10] = [
            ("disabled", SplitRefusal::NotRequested, |a| {
                a.enabled = false
            }),
            ("no span owner", SplitRefusal::NoSpanOwner, |a| {
                a.kind = None
            }),
            (
                "custom tier, no bound sidecar",
                SplitRefusal::UnboundHotset,
                |a| a.bound_hotset = false,
            ),
            ("no cold experts", SplitRefusal::NoColdExperts, |a| {
                a.host_experts = 0
            }),
            ("not flash-next", SplitRefusal::NotFlashNext, |a| {
                a.flash_next = false
            }),
            ("mixed codec", SplitRefusal::NotCpuExecuted, |a| {
                a.cpu_executed = false
            }),
            ("ep", SplitRefusal::ExpertParallel, |a| a.ep = true),
            ("awq", SplitRefusal::Awq, |a| a.awq = true),
            ("fp8", SplitRefusal::Fp8, |a| a.fp8 = true),
            ("staged narrow window", SplitRefusal::NarrowStage, |a| {
                a.narrow_stage = true
            }),
        ];
        for (name, reason, mutate) in refusals {
            let mut a = split_admission();
            mutate(&mut a);
            assert!(!a.armed(), "{name}");
            assert_eq!(a.refusal(), Some(reason), "{name}");
        }
        assert!(!CpuSplitAdmission::default().armed());
    }

    #[test]
    fn whole_layer_split_arms_without_a_sidecar_and_refuses_every_unsupported_route() {
        let whole = whole_layer_admission();
        assert!(whole.armed());
        assert_eq!(whole.kind, Some(SplitOwnerKind::WholeLayer));
        // Only a tier layer consults the sidecar binding.
        assert!(!whole.bound_hotset);
        let refusals: [(&str, SplitRefusal, fn(&mut CpuSplitAdmission)); 8] = [
            ("disabled", SplitRefusal::NotRequested, |a| {
                a.enabled = false
            }),
            (
                "device-resident aggregate",
                SplitRefusal::NoSpanOwner,
                |a| a.kind = None,
            ),
            ("no host experts", SplitRefusal::NoColdExperts, |a| {
                a.host_experts = 0
            }),
            ("not flash-next", SplitRefusal::NotFlashNext, |a| {
                a.flash_next = false
            }),
            ("non QT44/QT53 experts", SplitRefusal::NotCpuExecuted, |a| {
                a.cpu_executed = false
            }),
            ("ep", SplitRefusal::ExpertParallel, |a| a.ep = true),
            ("awq", SplitRefusal::Awq, |a| a.awq = true),
            ("staged narrow window", SplitRefusal::NarrowStage, |a| {
                a.narrow_stage = true
            }),
        ];
        for (name, reason, mutate) in refusals {
            let mut a = whole_layer_admission();
            mutate(&mut a);
            assert!(!a.armed(), "{name}");
            assert_eq!(a.refusal(), Some(reason), "{name}");
        }
    }

    #[test]
    fn split_refusals_log_one_line_only_for_unsupported_routes() {
        // A layer outside the mode, or with nothing cold, is not a refused
        // route: no line.
        for reason in [
            SplitRefusal::NotRequested,
            SplitRefusal::NoSpanOwner,
            SplitRefusal::NoColdExperts,
        ] {
            assert!(reason.is_silent(), "{reason:?}");
        }
        for reason in [
            SplitRefusal::UnboundHotset,
            SplitRefusal::NotFlashNext,
            SplitRefusal::NotCpuExecuted,
            SplitRefusal::ExpertParallel,
            SplitRefusal::Awq,
            SplitRefusal::Fp8,
            SplitRefusal::NarrowStage,
        ] {
            assert!(!reason.is_silent(), "{reason:?}");
            let line = split_refusal_line(reason, "gfx1201");
            assert!(!line.contains('\n'), "{line}");
            assert!(
                line.starts_with("  [moe-cpu-split] qwen4: not armed on gfx1201: "),
                "{line}"
            );
            assert!(line.ends_with("keep the PCIe mapped-expert read"), "{line}");
        }
        // One bit per reason: the once-per-process latch cannot alias.
        let mut bits = 0u32;
        for reason in [
            SplitRefusal::NotRequested,
            SplitRefusal::NoSpanOwner,
            SplitRefusal::NoColdExperts,
            SplitRefusal::UnboundHotset,
            SplitRefusal::NotFlashNext,
            SplitRefusal::NotCpuExecuted,
            SplitRefusal::ExpertParallel,
            SplitRefusal::Awq,
            SplitRefusal::Fp8,
            SplitRefusal::NarrowStage,
        ] {
            let bit = 1u32 << (reason as u32);
            assert_eq!(bits & bit, 0, "{reason:?}");
            bits |= bit;
        }
    }

    #[test]
    fn split_never_arms_the_whole_layer_splice() {
        // The D1 admission refuses every tier layer, so one layer never carries
        // both numerical modes.
        let d1 = admission(true, true, false, false, true, false, true);
        assert!(!d1.armed());
        assert!(split_admission().armed());
    }

    #[test]
    fn host_span_requires_exact_extent_inside_owner() {
        assert_eq!(host_span(10, 20, 20, 100), Some(10..30));
        assert_eq!(host_span(80, 20, 20, 100), Some(80..100));
        assert_eq!(host_span(81, 20, 20, 100), None, "past the owner");
        assert_eq!(host_span(0, 19, 20, 100), None, "wrong entry length");
        assert_eq!(host_span(0, 0, 0, 100), None, "empty expert");
        assert_eq!(host_span(u64::MAX, 20, 20, 100), None, "overflow");
    }

    #[test]
    fn host_range_table_resolves_only_cold_and_names_bad_span() {
        // (host, gu_off, gu_len, dn_off, dn_len)
        let entries = vec![
            (false, 0, 0, 0, 0), // device expert: never read, no validation
            (true, 0, 8, 8, 4),
            (true, 12, 8, 20, 4),
        ];
        let table = host_range_table(entries.into_iter(), 8, 4, 24).unwrap();
        assert_eq!(table.len(), 3);
        assert!(table[0].is_none());
        assert_eq!(table[1].clone().unwrap(), (0..8, 8..12));
        assert_eq!(table[2].clone().unwrap(), (12..20, 20..24));

        let outside = vec![(true, 0, 8, 8, 4), (true, 20, 8, 28, 4)];
        let error = host_range_table(outside.into_iter(), 8, 4, 24).unwrap_err();
        assert!(error.contains("expert 1 gate/up"), "{error}");

        let overlap = vec![(true, 0, 8, 4, 4)];
        let error = host_range_table(overlap.into_iter(), 8, 4, 24).unwrap_err();
        assert!(error.contains("overlap"), "{error}");
    }

    #[test]
    fn layer_bitmap_is_hex_by_layer_label() {
        assert_eq!(layer_bitmap_hex(&[]), "0");
        assert_eq!(layer_bitmap_hex(&[0]), "01");
        assert_eq!(layer_bitmap_hex(&[0, 3, 9]), "0209");
        let all: Vec<usize> = (0..48).collect();
        assert_eq!(layer_bitmap_hex(&all), "ffffffffffff");
        // Order never matters.
        assert_eq!(layer_bitmap_hex(&[9, 0, 3]), layer_bitmap_hex(&[0, 3, 9]));
    }

    #[test]
    fn host_exec_identity_is_empty_unless_cpu_arithmetic_armed() {
        use SplitOwnerKind::{Tier, WholeLayer};
        assert_eq!(host_exec_identity(&[], &[], None, None), "");
        let all: Vec<usize> = (0..48).collect();
        assert_eq!(
            host_exec_identity(&all, &[], None, None),
            "cpu-all-v1/ffffffffffff"
        );
        let digest = [0xabu8; 32];
        let split = host_exec_identity(
            &[],
            &[(Tier, &all)],
            Some((&digest, Some(&"c".repeat(64)))),
            None,
        );
        assert_eq!(
            split,
            format!(
                "cpu-tier-split-v1/{}/ffffffffffff/hotset-sha256={}",
                "ab".repeat(32),
                "c".repeat(64)
            )
        );
        let unbound = host_exec_identity(&[], &[(Tier, &all)], Some((&digest, None)), None);
        assert!(unbound.ends_with("/hotset-sha256=unbound"), "{unbound}");
        // Distinct modes, plans and armed layer sets never collide.
        let fewer: Vec<usize> = (0..47).collect();
        let other = host_exec_identity(
            &[],
            &[(Tier, &fewer)],
            Some((&digest, Some(&"c".repeat(64)))),
            None,
        );
        assert_ne!(split, other);
        let other_digest = [0xacu8; 32];
        assert_ne!(
            split,
            host_exec_identity(
                &[],
                &[(Tier, &all)],
                Some((&other_digest, Some(&"c".repeat(64)))),
                None,
            )
        );
        assert_ne!(split, host_exec_identity(&all, &[], None, None));
        assert!(
            host_exec_identity(&all, &[(Tier, &all)], Some((&digest, None)), None).contains('+')
        );
        // The un-tiered whole-layer split has its own identity: no tier digest
        // or sidecar (there is no plan to freeze: every host expert is cold),
        // never aliasing the tier or D1 strings.
        let whole = host_exec_identity(&[], &[(WholeLayer, &all)], None, None);
        assert_eq!(whole, "cpu-whole-split-v1/ffffffffffff");
        assert_ne!(whole, split);
        assert_ne!(whole, host_exec_identity(&all, &[], None, None));
        assert_ne!(
            whole,
            host_exec_identity(&[], &[(WholeLayer, &fewer)], None, None)
        );
        // Both split kinds on one load join in the fixed caller order.
        let tier_labels = [3usize];
        let both = host_exec_identity(
            &[],
            &[(Tier, &tier_labels), (WholeLayer, &fewer)],
            Some((&digest, Some(&"c".repeat(64)))),
            None,
        );
        assert!(both.starts_with("cpu-tier-split-v1/"), "{both}");
        assert!(both.contains("+cpu-whole-split-v1/"), "{both}");
        // The slab kind has its own frozen spelling, distinct from both.
        let slab = host_exec_identity(&[], &[(SplitOwnerKind::Slab, &fewer)], None, None);
        assert!(slab.starts_with("cpu-slab-split-v1/"), "{slab}");
        assert_ne!(
            slab,
            host_exec_identity(&[], &[(WholeLayer, &fewer)], None, None)
        );
        let all_kinds = host_exec_identity(
            &[],
            &[
                (Tier, &tier_labels),
                (WholeLayer, &fewer),
                (SplitOwnerKind::Slab, &fewer),
            ],
            Some((&digest, Some(&"c".repeat(64)))),
            None,
        );
        assert!(all_kinds.ends_with(&format!("+{slab}")), "{all_kinds}");
    }

    /// A 512-expert layer: experts `0..hot` device-resident, the rest cold,
    /// with a strictly decreasing routing count by expert id (so id order is
    /// the cold rank order).
    fn share_layer(hot: usize) -> (Vec<bool>, Vec<u64>) {
        let cold = (0..EXPERTS).map(|e| e >= hot).collect();
        let mass = (0..EXPERTS).map(|e| 100_000 - 150 * e as u64).collect();
        (cold, mass)
    }

    fn mask_bytes(mask: &[bool]) -> Vec<u8> {
        mask.iter().map(|&b| u8::from(b)).collect()
    }

    #[test]
    fn share_identity_names_the_partition_and_is_unchanged_without_one() {
        use SplitOwnerKind::Tier;
        let all: Vec<usize> = (0..48).collect();
        let digest = [0xabu8; 32];
        let side = Some((&digest, Some(&"c".repeat(64)[..])));
        let plain = host_exec_identity(&[], &[(Tier, &all)], side, None);
        // No share: the string is byte-for-byte the pre-share identity.
        assert!(!plain.contains("pcie-share"), "{plain}");
        assert_eq!(pcie_share_digest(&[]), None);
        let (cold, mass) = share_layer(100);
        let half = mask_bytes(&pcie_partition(&cold, &mass, 500_000));
        let third = mask_bytes(&pcie_partition(&cold, &mass, 333_333));
        let token = |layers: &[(usize, &[u8])]| pcie_share_digest(layers).unwrap();
        let (a, b) = (token(&[(3, &half)]), token(&[(3, &third)]));
        assert_eq!(a.len(), 32);
        assert_ne!(a, b, "different partitions never share a token");
        assert_ne!(a, token(&[(4, &half)]), "the layer label is part of it");
        assert_ne!(a, token(&[(3, &half), (4, &half)]));
        assert_eq!(a, token(&[(3, &half)]), "deterministic");
        let shared = host_exec_identity(&[], &[(Tier, &all)], side, Some(&a));
        assert_eq!(shared, format!("{plain}/pcie-share-v1={a}"));
        assert_ne!(
            shared,
            host_exec_identity(&[], &[(Tier, &all)], side, Some(&b))
        );
    }

    #[test]
    fn share_zero_keeps_exactly_todays_placement_vector() {
        for hot in [0usize, 1, 100, 300, 511, 512] {
            let (cold, mass) = share_layer(hot);
            let none = pcie_partition(&cold, &mass, 0);
            assert!(none.iter().all(|&owned| !owned), "hot={hot}");
            // Today's mask is `1` for every cold expert; no GPU share is the
            // empty list the owner starts with, and an all-false partition
            // yields the same vector.
            let today = mask_bytes(&cold);
            assert_eq!(cpu_owned_mask(&cold, &[]), today, "hot={hot}");
            assert_eq!(cpu_owned_mask(&cold, &none), today, "hot={hot}");
        }
    }

    #[test]
    fn share_one_gives_every_cold_expert_to_the_gpu_arm_and_the_cpu_arm_none() {
        for hot in [0usize, 100, 511] {
            let (cold, mass) = share_layer(hot);
            let all = pcie_partition(&cold, &mass, hipfire_config::memory::D2_PCIE_SHARE_PPM_ONE);
            assert_eq!(all, cold, "hot={hot}");
            // Nothing left for the CPU arm: such a layer is not armed.
            assert!(cpu_owned_mask(&cold, &all).iter().all(|&v| v == 0));
            // A share above one is the same as one.
            assert_eq!(pcie_partition(&cold, &mass, u32::MAX), all);
        }
        // No cold expert at all: nothing to own, at any share.
        let (cold, mass) = share_layer(512);
        assert!(pcie_partition(&cold, &mass, 500_000).iter().all(|&v| !v));
    }

    #[test]
    fn share_partition_is_deterministic_and_never_touches_device_experts() {
        let (cold, mass) = share_layer(180);
        for ppm in [1u32, 50_000, 250_000, 500_000, 750_000, 999_999] {
            let first = pcie_partition(&cold, &mass, ppm);
            assert_eq!(first, pcie_partition(&cold, &mass, ppm), "ppm={ppm}");
            for expert in 0..180 {
                assert!(!first[expert], "ppm={ppm}: device expert {expert} owned");
            }
            // CPU-owned and GPU-owned cold experts partition the cold set.
            let cpu = cpu_owned_mask(&cold, &first);
            for expert in 0..EXPERTS {
                let owners = u8::from(first[expert]) + cpu[expert];
                assert_eq!(owners, u8::from(cold[expert]), "ppm={ppm} expert {expert}");
            }
        }
        // Order of equal masses is the expert id, so ties are stable.
        let flat = vec![7u64; EXPERTS];
        let all_cold = vec![true; EXPERTS];
        let first = pcie_partition(&all_cold, &flat, 400_000);
        assert_eq!(first, pcie_partition(&all_cold, &flat, 400_000));
    }

    #[test]
    fn share_partition_balances_mass_and_interleaves_the_rank_order() {
        let (cold, mass) = share_layer(100);
        let cold_mass: u128 = (100..EXPERTS).map(|e| u128::from(mass[e]) + 1).sum();
        let heaviest = u128::from(mass[100]) + 1;
        for ppm in [100_000u32, 250_000, 500_000, 800_000] {
            let pcie = pcie_partition(&cold, &mass, ppm);
            let owned: u128 = (0..EXPERTS)
                .filter(|&e| pcie[e])
                .map(|e| u128::from(mass[e]) + 1)
                .sum();
            let target = u128::from(ppm) * cold_mass / 1_000_000;
            let error = owned.abs_diff(target);
            assert!(
                error <= heaviest,
                "ppm={ppm}: owned {owned} vs target {target}"
            );
            // Spread, not blocked: the GPU-owned experts are found in every
            // quarter of the cold rank order.
            let quarter = (EXPERTS - 100) / 4;
            for q in 0..4 {
                let lo = 100 + q * quarter;
                assert!(
                    pcie[lo..lo + quarter].iter().any(|&p| p),
                    "ppm={ppm}: quarter {q} has no GPU-owned expert"
                );
            }
        }
        // One half alternates CPU, GPU, CPU, GPU, ... over equal masses.
        let flat = vec![0u64; 8];
        let pcie = pcie_partition(&[true; 8], &flat, 500_000);
        assert_eq!(
            pcie,
            [false, true, false, true, false, true, false, true],
            "equal masses at one half alternate"
        );
    }

    #[test]
    fn share_partition_grows_monotonically_with_the_share() {
        // Equal masses: the GPU-owned count is a rounding of `share * cold`,
        // so it never shrinks as the share grows (with unequal masses the
        // owned MASS tracks the target within the heaviest expert instead).
        let (cold, _) = share_layer(100);
        let mass = vec![5u64; EXPERTS];
        let mut last = 0usize;
        for ppm in (0..=1_000_000u32).step_by(50_000) {
            let owned = pcie_partition(&cold, &mass, ppm)
                .iter()
                .filter(|&&p| p)
                .count();
            assert!(owned >= last, "ppm={ppm}: {owned} < {last}");
            last = owned;
        }
        assert_eq!(last, EXPERTS - 100);
    }

    #[test]
    fn tier_counters_count_pcie_owned_cold_experts_as_cold_not_hot() {
        // The tier counters' host mask is every host-mapped expert whichever arm serves it;
        // the executor's placement excludes the GPU-owned ones. Counting with the placement
        // as the host mask would call PCIe-owned experts device-resident.
        let (cold, mass) = share_layer(300);
        let pcie = pcie_partition(&cold, &mass, 250_000);
        let placement = cpu_owned_mask(&cold, &pcie);
        let host = cpu_owned_mask(&cold, &[]);
        assert_eq!(host, mask_bytes(&cold), "the host mask ignores the share");
        let pcie_ids: Vec<i32> = (0..EXPERTS)
            .filter(|&e| pcie[e])
            .map(|e| e as i32)
            .collect();
        let cpu_ids: Vec<i32> = (0..EXPERTS)
            .filter(|&e| cold[e] && !pcie[e])
            .take(7)
            .map(|e| e as i32)
            .collect();
        let hot_ids = [0i32, 5, 299];
        assert!(pcie_ids.len() > 10);
        let ids: Vec<i32> = pcie_ids
            .iter()
            .chain(&cpu_ids)
            .chain(&hot_ids)
            .copied()
            .collect();
        let mut seen = vec![0u64; EXPERTS.div_ceil(64)];
        let right = hipfire_dispatch::cpu_split::tally_call(&ids, &placement, &host, &mut seen);
        assert_eq!(
            (right.hot, right.cold_cpu, right.cold_pcie),
            (3, 7, pcie_ids.len() as u32)
        );
        // The old (placement-as-host) mask misfiles every PCIe-owned expert as hot.
        let wrong =
            hipfire_dispatch::cpu_split::tally_call(&ids, &placement, &placement, &mut seen);
        assert_eq!(wrong.cold_pcie, 0);
        assert_eq!(wrong.hot, 3 + pcie_ids.len() as u32);
        // One PCIe-owned expert is one 2.6112 MB QT44+QT53 read.
        assert_eq!(1_740_800u64 + 870_400, 2_611_200);
    }

    #[test]
    fn cold_mass_fraction_is_the_mass_share_not_the_count_share() {
        let cold = [true, false, true, false];
        assert_eq!(cold_mass_fraction(&cold, &[1, 1, 1, 1]), 0.5);
        assert_eq!(cold_mass_fraction(&cold, &[3, 0, 1, 0]), 1.0);
        assert_eq!(cold_mass_fraction(&cold, &[1, 3, 0, 4]), 0.125);
        // No routing mass: the count share.
        assert_eq!(cold_mass_fraction(&cold, &[0, 0, 0, 0]), 0.5);
        assert_eq!(cold_mass_fraction(&cold, &[]), 0.5);
        assert_eq!(cold_mass_fraction(&[], &[]), 0.0);
    }

    /// The whole-layer owner resolves the same bytes D1's owners do, and
    /// places every expert on the host.
    #[test]
    fn whole_layer_owner_serves_d1_spans_and_places_every_expert_cold() {
        let (gate_up, down) = fake_layout();
        let d1 = CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, FAKE_GU, FAKE_DN).unwrap();
        let owner = HostSpanOwner::WholeLayer(
            CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, FAKE_GU, FAKE_DN).unwrap(),
        );
        assert_eq!(owner.kind(), SplitOwnerKind::WholeLayer);
        assert_eq!(owner.cold_experts(), EXPERTS);
        assert_eq!(owner.placement(), vec![1u8; EXPERTS]);
        for idx in [0usize, 1, 255, 511] {
            let spans = owner.payload(idx, DType::MQ4G256, DType::MQ4G256).unwrap();
            let reference = d1.payload(idx, DType::MQ4G256, DType::MQ4G256).unwrap();
            assert_eq!(
                spans.gate_up.as_ptr(),
                reference.gate_up.as_ptr(),
                "idx {idx}"
            );
            assert_eq!(spans.down.as_ptr(), reference.down.as_ptr(), "idx {idx}");
            assert_eq!(spans.gate_up.len(), FAKE_GU);
            assert_eq!(spans.down.len(), FAKE_DN);
        }
        assert!(owner
            .payload(EXPERTS, DType::MQ4G256, DType::MQ4G256)
            .is_none());
        // `F32` is a dtype the CPU executor supports (`cpu_quant_for`), so the
        // generic owner serves it as the same borrowed spans, tagged F32.
        // Format/span-size safety is the executor's per-expert length check;
        // the owner only guarantees in-range, in-bounds spans.
        let f32_payload = owner.payload(0, DType::F32, DType::MQ4G256).unwrap();
        assert_eq!(
            Some(f32_payload.gate_up_quant),
            hipfire_dispatch::cpu_exec::cpu_quant_for(DType::F32)
        );
        assert_eq!(f32_payload.gate_up.len(), FAKE_GU);
        // A dtype without a CPU executor has no payload, never a guess.
        assert!(owner.payload(0, DType::Raw, DType::MQ4G256).is_none());
        assert!(owner.payload(0, DType::MQ4G256, DType::Raw).is_none());
    }
}
