// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//! MoE kernel family: dispatching expert GEMM operations.
//!
//! Supports 3 variants:
//! - **IndexedGateUp**: gate+up projection for a single expert (indexed by token)
//! - **IndexedDown**: down projection for a single expert (indexed by token)
//! - **GroupedGemm**: batched grouped-expert GEMM (all experts in one launch)
//!
//! # Current status
//!
//! `run()` is the centralized single-token MoE decode entry — it delegates to
//! [`crate::pipeline::run_moe_decode`] (the GPU top-K fast path plus the generic
//! CPU-top-K fallback). The family owns resolution (`MoeDtypes` → `MoeResolution`);
//! the model passes only the dtype snapshot + k. One `DispatchCtx` is threaded
//! end-to-end from the call site through every inner GEMV. Scratch stays model-owned.
//! Grouped-GEMM prefill is a future arm (gated on `ShapeInfo.batch_size`).

use rdna_compute::DType;
use rdna_compute::{Gpu, GpuTensor};

use crate::context::DispatchCtx;
use crate::families::gemv::{GivensRef, WeightRef};
use crate::pipeline::sealed_moe::PrefillRouteMode;
use crate::tables::KernelRegistry;
use crate::traits::KernelFamily;
use crate::types::*;

/// Grouped-MoE WMMA tile row count used by capacity bounds and scatter.
pub const MOE_GROUPED_BLOCK_M: usize = 16;

/// The complete semantic recipe selected by an architecture binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeRecipe {
    SoftmaxGatedShared,
    SigmoidRoutedNoShared,
}

/// Source of the activation consumed by a MoE recipe.
pub enum MoeNormalization<'a> {
    Provided,
    RmsNorm {
        weight: &'a GpuTensor,
        plain_out: &'a GpuTensor,
        eps: f32,
    },
}

/// Dtypes for the optional replicated/shared expert.
#[derive(Clone, Copy, Debug)]
pub struct MoeSharedDtypes {
    pub selector: DType,
    pub gate: DType,
    pub up: DType,
    pub down: DType,
}

/// Borrowed weights for the optional replicated/shared expert.
pub struct MoeSharedWeights<'a> {
    pub selector: WeightRef<'a>,
    pub gate: WeightRef<'a>,
    pub up: WeightRef<'a>,
    pub down: WeightRef<'a>,
}

/// Decode scratch and weights for the optional replicated/shared expert.
pub struct MoeSharedDecode<'a> {
    pub weights: MoeSharedWeights<'a>,
    pub intermediate: usize,
    pub scalar: &'a GpuTensor,
    pub gate_out: &'a GpuTensor,
    pub up_out: &'a GpuTensor,
}

/// Prefill scratch and weights for the optional replicated/shared expert.
pub struct MoeSharedPrefill<'a> {
    pub weights: MoeSharedWeights<'a>,
    pub intermediate: usize,
    pub scalar: &'a GpuTensor,
    pub gate_out: &'a GpuTensor,
    pub up_out: &'a GpuTensor,
    pub rotated: &'a GpuTensor,
}

/// Router projection policy for Q8 prefill inputs.
pub enum MoeQ8RouterPolicy<'a> {
    DispatcherEntry,
    FreshFp16Wmma {
        enabled: bool,
        scratch: &'a GpuTensor,
    },
}

/// Complete prefill prologue declaration.
pub struct MoePrefillPrelude<'a> {
    pub normalization: MoeNormalization<'a>,
    pub router: WeightRef<'a>,
    pub router_logits: &'a GpuTensor,
    pub router_scores: &'a GpuTensor,
    pub norm_topk_prob: bool,
    pub route: PrefillRouteMode<'a>,
    pub shared: Option<MoeSharedPrefill<'a>>,
    pub q8_router_policy: MoeQ8RouterPolicy<'a>,
}

/// Whether the Qwen softmax/shared gate quartet has the exact V1 MQ4 layout
/// consumed by the fused gate-side kernel.
pub fn softmax_shared_gate_uniform_mq4(
    router: DType,
    selector: DType,
    gate: DType,
    up: DType,
) -> bool {
    [router, selector, gate, up]
        .into_iter()
        .all(|dtype| dtype == DType::MQ4G256)
}

/// Whether normalization may write the Qwen softmax/shared activation directly
/// in the prerotated basis consumed by the fused gate-side operation.
///
/// V1 is dtype-selected on every architecture, matching the established
/// `fused_qkvza_hfq4g256` route. V2 is restricted to its exact Ornith quartet,
/// no AWQ companions, and the two architectures with the matching fused kernel.
pub fn softmax_shared_gate_prerotated(
    ctx: &DispatchCtx,
    router: &WeightRef<'_>,
    shared: &MoeSharedWeights<'_>,
) -> bool {
    if softmax_shared_gate_uniform_mq4(
        router.dtype,
        shared.selector.dtype,
        shared.gate.dtype,
        shared.up.dtype,
    ) {
        return true;
    }
    if !(ctx.arch.is_gfx1100() || ctx.arch.is_gfx1201()) {
        return false;
    }
    softmax_shared_gate_mq4v2_layouts([
        (router.dtype, router.m, router.k, router.awq_scale.is_some()),
        (
            shared.selector.dtype,
            shared.selector.m,
            shared.selector.k,
            shared.selector.awq_scale.is_some(),
        ),
        (
            shared.gate.dtype,
            shared.gate.m,
            shared.gate.k,
            shared.gate.awq_scale.is_some(),
        ),
        (
            shared.up.dtype,
            shared.up.m,
            shared.up.k,
            shared.up.awq_scale.is_some(),
        ),
    ])
}

fn softmax_shared_gate_mq4v2_layouts(layouts: [(DType, usize, usize, bool); 4]) -> bool {
    let expected = [(256, 2_048), (1, 2_048), (512, 2_048), (512, 2_048)];
    layouts
        .into_iter()
        .zip(expected)
        .all(|((dtype, m, k, has_awq), (want_m, want_k))| {
            dtype == DType::MQ4G256V2 && m == want_m && k == want_k && !has_awq
        })
}
// ── MoE eligibility lattice ────────────────────────────

/// Routed-expert tiers the mixed-tier graded decode path can execute: the
/// tiers for which per-tier indexed gate_up/down GEMV kernels exist (served
/// on-device via `run_moe_decode`'s `expert_dtype_tags` branch). A per-expert
/// tier table containing any other DType
/// cannot be served by the mixed path and is rejected up front with a clear
/// error rather than failing deep in the per-bucket dispatch.
///
/// Includes V1 affine MQ4/MQ6, Paro, and the dual-half V2 layouts (qt44/qt47)
/// consumed by mixed kernel branch tags 7..18. V1/V2 must stay distinct —
/// collapsing either pair silently corrupts scale/zero decode.
pub const MIXED_SUPPORTED_TIERS: [DType; 5] = [
    DType::MQ4G256,
    DType::MQ6G256,
    DType::ParoQ4G128,
    DType::MQ4G256V2,
    DType::MQ6G256V2,
];

/// Per-layer dtype snapshot the MoE eligibility lattice reads. Built by the
/// model from its weight structs; kept dtype-only so this stays GPU-free and
/// the dispatch crate needs no dependency on any arch crate.
///
/// `experts_all_gate_up_mq4` mirrors the `ffn.experts.iter().all(..)` clause
/// the original `gate_side_mq4` check used (qwen35.rs:4598-4605). The routed
pub struct MoeDtypes<'a> {
    pub router: DType,
    /// Optional shared-expert projection dtypes. `None` is a first-class
    /// no-shared recipe, independent of `has_paro_shared`.
    pub shared: Option<MoeSharedDtypes>,
    pub experts_all_gate_up_mq4: bool,
    pub routed_gate_up: DType, // ffn.experts[0].gate_up
    pub routed_down: DType,    // ffn.experts[0].down
    /// Per-expert mixed routed dtype: experts in one layer carry DIFFERENT
    /// gate_up and/or down dtypes (N-tier graded: MQ6 hot / MQ4 mid / MQ2L
    /// or MQ3L or E8-family cold), so `routed_gate_up` / `routed_down`
    /// (= experts[0]) are NOT representative.
    pub routed_has_mixed_experts: bool,
    pub has_paro_shared: bool, // ffn.paro_shared.is_some()
    /// Per-expert gate_up tiers for intra-layer mixed-tier dispatch.
    pub per_expert_gate_up: Option<&'a [DType]>,
    /// Per-expert down tiers (parallel to `per_expert_gate_up`).
    pub per_expert_down: Option<&'a [DType]>,
}

impl MoeDtypes<'_> {
    pub fn has_mq6_projection(&self) -> bool {
        let shared_has_mq6 = self.shared.is_some_and(|shared| {
            [shared.selector, shared.gate, shared.up, shared.down]
                .iter()
                .any(|dt| matches!(*dt, DType::MQ6G256 | DType::MQ6G256V2))
        });
        shared_has_mq6
            || [self.routed_gate_up, self.routed_down]
                .iter()
                .any(|dt| matches!(*dt, DType::MQ6G256 | DType::MQ6G256V2))
    }
}

/// Resolved fused-vs-fallback eligibility for one MoE decode layer. This IS the
/// routing-config logic, relocated from `moe_ffn_decode_impl` into one typed,
/// testable place (review finding #1). Pure function of `MoeDtypes` + k.
#[derive(Clone, Copy, Debug)]
pub struct MoeResolution {
    pub gate_side_mq4: bool,
    /// Router + shared expert gate/up are an exact-uniform MQ4G256 V1
    /// quartet. The fused gate path is independent of routed-expert dtype.
    /// MQ4G256V2 is admitted separately via `gate_fusable_mq4v2` (exact V2
    /// quartet → `fused_qkvza_hfq4g256_mq4v2`); mixed V1/V2 stays non-fusable.
    pub gate_fusable: bool,
    /// Router + shared scalar gate + shared expert gate/up are an
    /// exact-uniform MQ4G256V2 quartet. Independent of routed-expert dtype.
    /// Mixed V1/V2 gate-side dtypes never set this (or `gate_fusable`).
    pub gate_fusable_mq4v2: bool,
    pub routed_indexable_mq4: bool,
    pub routed_indexable_mq4v2: bool,
    pub routed_indexable_mq5: bool,
    pub routed_indexable_mq6: bool,
    /// Uniform all-MQ6G256V2 (qt47) routed experts. Gated on BOTH gate_up and
    /// down being V2 — qt47 dual-f16-grid header is incompatible with V1
    /// MQ6G256's f32 header; a split pairing must never claim either arm.
    pub routed_indexable_mq6v2: bool,
    /// Mixed routed experts: gate_up MQ4, down MQ6 (the "mq6-down" lever —
    /// promote only the sensitive residual-write projection to 6-bit while
    /// gate_up stays 4-bit). Indexable on the decode GPU-top-K path: gate_up
    /// uses the MQ4 indexed GEMV, down uses the MQ6 indexed GEMV, silu+rotate
    /// (optionally AWQ) is weight-agnostic. Decode-only (prefill Path-0 on
    /// gfx9* has no MQ6 down arm; eval scores per-token = decode).
    pub routed_indexable_mixed_gu4_dn6: bool,
    pub routed_indexable_paro: bool,
    /// Uniform all-MQ2-Lloyd routed experts (gate_up == down == MQ2G256Lloyd).
    /// Reuses the ds4/minimax indexed Lloyd MoE GEMVs on the decode GPU-top-K
    /// path: gate_up uses the MQ2-Lloyd indexed GEMV, down uses the MQ2-Lloyd
    /// atomic-residual GEMV (self-combining -> no separate down combine).
    pub routed_indexable_mq2lloyd: bool,
    /// Uniform all-MQ3-Lloyd routed experts (gate_up == down == MQ3G256Lloyd).
    /// Same indexed-Lloyd decode path as mq2lloyd, MQ3 launchers.
    pub routed_indexable_mq3lloyd: bool,
    /// Routed experts whose gate_up and down are each drawn, INDEPENDENTLY, from
    /// the codebook family `{MQ2G256Lloyd, MQ3G256Lloyd, MQ2G256GL, MQ3G256GL}` —
    /// the per-projection allocation (e.g. gate_up 2-bit, down 3-bit) that puts
    /// the cheap bits on the larger projection and the accurate ones on the
    /// residual write. Indexable because `run_moe_decode` already picks the
    /// gate_up and down GEMVs from their own dtypes rather than a coupled flag,
    /// and because ALL FOUR down kernels self-combine via atomicAdd — so
    /// `routed_down_self_combines` (keyed on `routed_down` alone) stays correct
    /// and the shared down-combine is skipped exactly once. silu+rotate is
    /// weight-agnostic (it reads activations only). Subsumes the two uniform
    /// Lloyd arms above and the uniform GL cases.
    ///
    /// Lloyd and GL are freely mixable across the two projections: both are
    /// FWHT-G256 formats consuming the same rotated activation, and each GEMV is
    /// selected from its own projection's dtype. The ONLY thing that differs is
    /// where the codebook comes from (per-group fp16 header vs scalar kernel
    /// args), which is entirely inside the launcher.
    ///
    /// Decode-only: batched prefill rejects MoE MQ3-Lloyd outright (see
    /// `moe_ffn_has_mq3_experts_uniform` in hipfire-arch-qwen35), which already
    /// blocks the pre-existing uniform MQ3-Lloyd path too; the GL dtypes are
    /// likewise not admitted by `moe_ffn_batched_admissible_for_dtypes`, so a
    /// GL model prefills through the per-token path.
    pub routed_indexable_mixed_lloyd: bool,
    /// Per-expert N-tier graded routed experts (MQ6 hot / MQ4 mid / MQ2L or
    /// MQ3L cold, applied to BOTH gate_up and down). Indexable on the decode
    /// GPU-top-K path via the merged dtype-tag-branched gate_up AND down
    /// kernels. The merged down writes the EXPANDED buffer for all dtypes →
    /// the single shared `moe_down_combine_k8_batched` runs (NOT Lloyd atomic
    /// self-combine). silu+rotate is weight-agnostic (unchanged).
    pub routed_indexable_mixed_per_expert: bool,
    /// Uniform UNROTATED Lloyd routed experts (MQ2G256LloydU, qt=51) on BOTH
    /// gate_up and down. Binds the same indexed MQ2-Lloyd kernels as qt19 but
    /// consumes x in the natural basis (`needs_x_rot_local == false`).
    pub routed_indexable_mq2lloyd_u: bool,
    pub use_gpu_topk: bool,
    pub needs_x_rot_local: bool,
    /// True when a per-expert tier table is `Some` AND contains >1 distinct
    /// DType — the layer's routed experts span multiple quant tiers and need
    /// the bucketed dispatch path (Task 3). `None` tables or all-equal `Some`
    /// tables leave this `false` ⇒ unchanged uniform fast path.
    pub mixed: bool,
}

impl MoeResolution {
    /// Arch-agnostic entry. The E8 indexed/grouped kernels exist on the RDNA3
    /// wave32-WMMA family (gfx11; `arch_has_e8_wmma`); passing `false` here routes
    /// E8 to the CPU-top-K fallback — preserving every existing caller + test.
    pub fn resolve(d: &MoeDtypes<'_>, k: usize) -> Self {
        Self::resolve_arch(d, k, false)
    }

    pub fn resolve_arch(d: &MoeDtypes<'_>, k: usize, arch_has_e8_wmma: bool) -> Self {
        use DType::*;
        // The fused four-weight gate kernel is admitted only for the exact
        // MQ4G256 V1 quartet. The exact MQ4G256V2 quartet is admitted on a
        // separate predicate (`gate_fusable_mq4v2`) that routes to the V2
        // scalar fused launcher. Mixed V1/V2 gate-side stays on the generic
        // four-GEMV path. Independent of routed-expert dtype: all rotated MQ
        // families consume the same FwhtG256 activation.
        let shared = d.shared;
        // A no-shared recipe cannot enter any shared/gate-quartet fusion.
        // `has_paro_shared` remains independent metadata for routed Paro.
        let gate_fusable = shared.is_some_and(|shared| {
            d.router == MQ4G256
                && shared.selector == MQ4G256
                && shared.gate == MQ4G256
                && shared.up == MQ4G256
        });
        let gate_fusable_mq4v2 = shared.is_some_and(|shared| {
            d.router == MQ4G256V2
                && shared.selector == MQ4G256V2
                && shared.gate == MQ4G256V2
                && shared.up == MQ4G256V2
        });
        // gate_side_mq4 keeps the stricter all-MQ4 meaning (incl. routed experts)
        // for the rotate/AWQ branch + callers that assume a uniform-MQ4 FFN.
        let gate_side_mq4 = gate_fusable && d.experts_all_gate_up_mq4;

        let routed_gate_up_mq4 = d.routed_gate_up == MQ4G256;
        // qt44/qt45 are FWHT-G256 formats exactly like qt13 — their kernels read
        // the ROTATED activations. Omitting them from `needs_x_rot_local` below
        // feeds unrotated x into rotated weights, which is silent: the model
        // still emits fluent text. Measured on Ornith 1.5 35B-A3B, prefill KLD
        // was 0.993 against 0.044 on the per-token path for the SAME artifact.
        let routed_gate_up_mq4v2 = d.routed_gate_up == MQ4G256V2;
        let routed_gate_up_mq4c = d.routed_gate_up == MQ4CG256;
        let routed_gate_up_mq5 = d.routed_gate_up == MQ5G256;
        let routed_gate_up_mq6 = d.routed_gate_up == MQ6G256;
        // qt47. FWHT-G256 dual-half header like qt44 — kernels read rotated x.
        let routed_gate_up_mq6v2 = d.routed_gate_up == MQ6G256V2;
        let routed_gate_up_paro = d.routed_gate_up == ParoQ4G128 && d.has_paro_shared;
        let routed_gate_up_mq2lloyd = d.routed_gate_up == MQ2G256Lloyd;
        let routed_gate_up_mq3lloyd = d.routed_gate_up == MQ3G256Lloyd;
        // UNROTATED Lloyd sibling (qt=51). Same kernels, same byte layout —
        // the ONLY difference is that it must not rotate x.
        let routed_gate_up_mq2lloyd_u = d.routed_gate_up == MQ2G256LloydU;

        let routed_indexable_mq4 = (d.routed_down == MQ4G256) && routed_gate_up_mq4;
        // qt44. Gated on BOTH sides being MQ4G256V2, like every other uniform
        // pairing: the indexed GEMVs decode qt44's dual-f16-grid header, and
        // handing one a qt13 f32-header row reads the scale/zero as the two
        // halves of a float — silently wrong output, not a fault.
        let routed_indexable_mq4v2 = (d.routed_down == MQ4G256V2) && routed_gate_up_mq4v2;
        let routed_indexable_mq5 = (d.routed_down == MQ5G256) && routed_gate_up_mq5;
        let routed_indexable_mq6 = (d.routed_down == MQ6G256) && routed_gate_up_mq6;
        // qt47. BOTH sides MQ6G256V2 — same dual-half hazard as mq4v2: V1 MQ6
        // stores f32 scale/zero while V2 stores two f16 pairs; same 200 B
        // stride so a mis-pair is silent fluent garbage, not a fault.
        let routed_indexable_mq6v2 = (d.routed_down == MQ6G256V2) && routed_gate_up_mq6v2;
        let routed_indexable_mixed_gu4_dn6 = routed_gate_up_mq4 && (d.routed_down == MQ6G256);
        let routed_indexable_mq2lloyd = (d.routed_down == MQ2G256Lloyd) && routed_gate_up_mq2lloyd;
        // Both sides must be the UNROTATED dtype. A rotated/unrotated mix has
        // no coherent single rotation decision for the layer, so it must fall
        // out of the indexed path entirely rather than pick one and silently
        // corrupt the other projection.
        let routed_indexable_mq2lloyd_u =
            (d.routed_down == MQ2G256LloydU) && routed_gate_up_mq2lloyd_u;
        let routed_indexable_mq3lloyd = (d.routed_down == MQ3G256Lloyd) && routed_gate_up_mq3lloyd;
        // gate_up on one of the codebook (Lloyd / GL) formats — needed both for
        // the per-projection mix below and for `needs_x_rot_local` (all four are
        // FwhtG256 and consume the pre-rotated activation).
        let routed_gate_up_gl = matches!(d.routed_gate_up, MQ2G256GL | MQ3G256GL);
        // Per-projection codebook mix (e.g. gate_up MQ2-GL + down MQ3-GL, the
        // 2-bit-gate/3-bit-down allocation; or any Lloyd×GL cross). Subsumes the
        // two uniform Lloyd arms above and the uniform GL cases; the OR below
        // makes the overlap harmless.
        //
        // SAFETY INVARIANT: every dtype admitted here MUST have (a) an indexed
        // gate_up GEMV arm in `run_moe_decode`, (b) an ATOMIC SELF-COMBINING
        // down GEMV arm there, and (c) membership in the
        // `routed_down_self_combines` set in pipeline/mod.rs. Admitting a dtype
        // that misses (c) double-counts every MoE layer, silently.
        const CODEBOOK_INDEXABLE: [DType; 4] = [MQ2G256Lloyd, MQ3G256Lloyd, MQ2G256GL, MQ3G256GL];
        let routed_indexable_mixed_lloyd = CODEBOOK_INDEXABLE.contains(&d.routed_gate_up)
            && CODEBOOK_INDEXABLE.contains(&d.routed_down);
        let routed_indexable_paro =
            (d.routed_down == ParoQ4G128 && d.has_paro_shared) && routed_gate_up_paro;
        // Per-expert mixed: the model already verified the experts carry
        // different down dtypes and built the tag table (single source of
        // truth). gate_up stays uniform MQ4, so it pairs with the MQ4 indexed
        // gate_up GEMV; the merged dtype-tag kernel serves the down step.
        let routed_indexable_mixed_per_expert = d.routed_has_mixed_experts;
        // mfp4/mfp3/mfp2-E8 grouped experts (RDNA3 wave32-WMMA): uniform E8-family
        // gate_up + down → the gemv_mfp4g32_e8_moe_{gate_up,down}_k8_indexed kernels
        // (for uniform E8 models). FWHT-rotated (FwhtG256), same as MQ4, so the
        // shared silu+mul+rotate plumbing applies. Graded mixed-E8 uses the tag-table
        // path (routed_indexable_mixed_per_expert) rather than this uniform arm.
        let routed_gate_up_e8 = matches!(d.routed_gate_up, MFP4G32E8 | MFP3G32E8 | MFP2G32E8);
        let routed_indexable_e8 = arch_has_e8_wmma
            && routed_gate_up_e8
            && matches!(d.routed_down, MFP4G32E8 | MFP3G32E8 | MFP2G32E8);

        let routed_dtype_indexable = routed_indexable_mq4
            || routed_indexable_mq4v2
            || routed_indexable_mq5
            || routed_indexable_mq6
            || routed_indexable_mq6v2
            || routed_indexable_mixed_gu4_dn6
            || routed_indexable_mixed_per_expert
            || routed_indexable_mq2lloyd
            || routed_indexable_mq2lloyd_u
            || routed_indexable_mq3lloyd
            || routed_indexable_mixed_lloyd
            || routed_indexable_paro
            || routed_indexable_e8;

        let use_gpu_topk = k == 8 && routed_dtype_indexable;
        let needs_x_rot_local = gate_side_mq4
            || gate_fusable_mq4v2
            || routed_indexable_mixed_per_expert
            || routed_gate_up_mq4
            || routed_gate_up_mq4v2
            || routed_gate_up_mq4c
            || routed_gate_up_mq5
            || routed_gate_up_mq6
            || routed_gate_up_mq6v2
            || routed_gate_up_mq2lloyd
            || routed_gate_up_mq3lloyd
            // MQ2/MQ3-G256-GL are FWHT-G256 formats: their gate_up kernel reads
            // `x_rot`, so the local rotation MUST be produced. Missing this is a
            // silent garbage-output failure (unrotated x into a rotated weight).
            || routed_gate_up_gl
            || routed_gate_up_paro
            || routed_indexable_e8;
        // NOTE: `routed_gate_up_mq2lloyd_u` is DELIBERATELY ABSENT from the
        // chain above. MQ2G256LloydU is the unrotated sibling: its weights are
        // encoded in the natural basis, so producing x_rot and handing it to
        // the kernel would be the exact "unrotated x into a rotated weight"
        // failure the comment above warns about, only mirrored — and equally
        // silent. It is also deliberately NOT in `CODEBOOK_INDEXABLE`, because
        // membership there would let a rotated/unrotated cross-pair resolve via
        // `routed_indexable_mixed_lloyd` with no coherent rotation decision.
        // See docs/design/2026-08-22-maple-preview-20b-a1b.md.

        // A per-expert tier table is "mixed" only when it is Some AND spans more
        // than one distinct DType. A Some table that is all-equal collapses to
        // the uniform fast path (mixed = false), so existing arches — which pass
        // None for both tables — are always uniform and byte-identical to today.
        let table_varies = |t: &Option<&[DType]>| {
            t.as_ref()
                .and_then(|v| v.split_first())
                .map(|(first, rest)| rest.iter().any(|dt| dt != first))
                .unwrap_or(false)
        };
        let mixed = table_varies(&d.per_expert_gate_up) || table_varies(&d.per_expert_down);

        Self {
            gate_side_mq4,
            gate_fusable,
            gate_fusable_mq4v2,
            routed_indexable_mq4,
            routed_indexable_mq4v2,
            routed_indexable_mq5,
            routed_indexable_mq6,
            routed_indexable_mq6v2,
            routed_indexable_mixed_gu4_dn6,
            routed_indexable_mq2lloyd,
            routed_indexable_mq2lloyd_u,
            routed_indexable_mq3lloyd,
            routed_indexable_mixed_lloyd,
            routed_indexable_mixed_per_expert,
            routed_indexable_paro,
            use_gpu_topk,
            needs_x_rot_local,
            mixed,
        }
    }

    pub fn routed_indexable(&self) -> bool {
        self.routed_indexable_mq4
            || self.routed_indexable_mq4v2
            || self.routed_indexable_mq5
            || self.routed_indexable_mq6
            || self.routed_indexable_mq6v2
            || self.routed_indexable_mixed_gu4_dn6
            || self.routed_indexable_mixed_per_expert
            || self.routed_indexable_mq2lloyd
            || self.routed_indexable_mq3lloyd
            || self.routed_indexable_mixed_lloyd
            || self.routed_indexable_paro
    }
}

/// Sealed decode mode for one indexed-MoE call. This is the model-declared
/// intent; the sealer (`pipeline::sealed_moe`) cross-checks it against the
/// bound expert table, the execution contract, and the process environment
/// before any launch, and invalid combinations fail there — never silently.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MoeEpMode {
    /// Single-GPU (or Single-owner) decode, including the existing
    /// next-layer deferred-combine experiment. Requires `routed_out=None`.
    None,
    /// Root-routed expert-parallel decode: per-rank zeroed partials
    /// (`routed_out=Some`, `defer_routed_combine=false`).
    /// The root routes on-GPU (SoftmaxTopK), seals the route-producer
    /// proof, and folds owned experts plus the shared expert once into
    /// its partial (`skip_shared=false`); non-root ranks skip the shared
    /// expert (`skip_shared=true`), adopt the broadcast root route, and
    /// fold owned experts (zero dummies read 0) into their partials.
    /// The driver then all-reduce-sums the partials and adds the result
    /// into each rank's residual once. Requires a plan-bound compact
    /// table carrying the root-routed execution contract.
    RootRoutedPartial,
}

/// True when the routed-down GEMV for `routed_down` materializes per-slot
/// rows into `down_expanded` (so a later fold can combine them).
///
/// The whole codebook family (`MQ2/MQ3-Lloyd` and their global-codebook
/// siblings `MQ2/MQ3-GL`) self-combines via atomic adds straight into the
/// residual target and writes NO expanded buffer — folding `down_expanded`
/// afterwards would double-count (or fold stale rows). Per-expert mixed
/// mode writes the expanded buffer for every tier, so it always returns
/// true here.
///
/// This is the single home of the predicate `run_moe_decode` inlines as
/// `routed_down_self_combines` (negated, ORed with the gfx1100 down-last
/// experiment) and the sealed canonical preflight requires. Keep them in
/// lockstep: miss a dtype here and a route silently zeroes out or
/// double-counts.
pub fn moe_down_writes_expanded(routed_down: DType, has_dtype_tags: bool) -> bool {
    has_dtype_tags
        || !matches!(
            routed_down,
            DType::MQ2G256Lloyd
                | DType::MQ2G256LloydU
                | DType::MQ3G256Lloyd
                | DType::MQ2G256GL
                | DType::MQ3G256GL
        )
}

// ── Dispatch parameters ────────────────────────────────

/// Host-resident routed weights used only by the generic CPU-top-K fallback.
///
/// Implementations return borrowed dispatch views on demand so model hot paths
/// do not rebuild an expert-reference vector for every token.
pub trait RoutedExpertWeights {
    fn len(&self) -> usize;

    fn get(&self, expert_idx: usize) -> Option<(WeightRef<'_>, WeightRef<'_>)>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Everything the MoE decode executor arm reads, marshaled by the model from
/// its weight/config/scratch structs. Resolution is owned by the family
/// (the model passes only the dtype snapshot + k); the executor computes
/// [`MoeResolution`] from [`MoeDtypes`] on entry.
pub struct MoeParams<'a> {
    pub dtypes: MoeDtypes<'a>,
    pub recipe: MoeRecipe,
    pub normalization: MoeNormalization<'a>,
    /// Token-batch width. Decode = 1. >1 must route to grouped prefill (Step 8).
    pub batch_size: usize,
    // dims / config scalars
    pub hidden: usize,
    pub mi: usize,
    pub k: usize,
    pub n_exp: usize,
    pub norm_topk_prob: bool,
    pub x_rot_prerotated: bool,
    /// Single-GPU lowered-decode experiment: leave the atomic-free routed
    /// output expanded so the architecture layer can combine it into the
    /// residual while producing the next layer's normalized activation.
    pub defer_routed_combine: bool,
    /// Sealed decode mode (see [`MoeEpMode`]).
    pub ep_mode: MoeEpMode,
    /// Safetensors layer index used by native GPTQ-on-E8 Hessian capture.
    pub layer_idx: u16,
    // activations / residual
    pub x_norm: &'a GpuTensor,
    pub x_residual: &'a GpuTensor,
    pub routed_out: Option<&'a GpuTensor>,
    /// EP: skip the shared-expert down projection on non-root ranks.
    pub skip_shared: bool,
    // gate-side weights
    pub router: WeightRef<'a>,
    pub shared: Option<MoeSharedDecode<'a>>,
    // routed expert pointer tables + dims
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    /// Route A MoE-AWQ: per-routed-expert down `awq_scale` pointer table
    /// (`[2·n_exp]` f32 = n_exp `u64` ptrs → each expert's `[routed_down_k]`
    /// f32 scale). `Some` only when the `.hfq` carries per-expert
    /// `down_proj.awq_scale` sidecars; the executor then runs the AWQ-aware
    /// indexed silu+rotate (`x/s` before the FWHT). `None` (default) = the
    /// plain silu+rotate, byte-identical to pre-AWQ.
    pub expert_down_awq_ptrs: Option<&'a GpuTensor>,
    /// Per-expert mixed-precision decode: `[n_exp]` u8 (DType::Raw, 1 B/exp)
    /// dtype-tag table, `Some` iff any expert's gate_up or down dtype differs
    /// from experts[0] (N-tier graded files). The merged dtype-tag-branched
    /// gate_up AND down kernels read `dtype_tags[expert_id]` per block:
    ///   0=MQ6G256 (200 B/grp), 1=MQ2G256Lloyd (72 B/grp),
    ///   2=MQ4G256 (136 B/grp), 3=MQ3G256Lloyd (112 B/grp).
    /// `None` (default) ⇒ uniform path, byte-identical to pre-mixed.
    pub expert_dtype_tags: Option<&'a GpuTensor>,
    pub routed_gate_up_k: usize,
    pub routed_down_m: usize,
    pub routed_down_k: usize,
    /// Allocation-free access to resident per-expert weights for the generic
    /// CPU-top-K fallback (`!use_gpu_topk`). Empty under paged residency, where
    /// the indexed GPU-top-K path is the only supported mode.
    pub routed_experts: &'a dyn RoutedExpertWeights,
    // paro sidecars
    pub routed_gate_up_paro: Option<GivensRef<'a>>,
    pub routed_down_paro: Option<GivensRef<'a>>,
    // scratch buffers
    pub router_logits: &'a GpuTensor,
    pub x_rot_local: &'a GpuTensor,
    /// Fused [gate||up] scratch for a single routed expert in the host path.
    pub gate_up_buf: &'a GpuTensor,
    pub ffn_hidden: &'a GpuTensor,
    pub ffn_out: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub down_expanded: &'a GpuTensor,
}

// ── DeepSeek-V4 bias-aware decode parameters ───────────

/// Exact-device MQ2-Lloyd operations used by the DeepSeek4 bias-aware decode
/// executor.
///
/// The dispatch crate deliberately has no architecture detection here. A
/// model crate may provide this capability only after its loader has admitted
/// a model-owned backend. The implementation must still fail closed when the
/// supplied [`Gpu`] is not the device proven by that backend.
pub trait MoeBiasAwareMq2Backend {
    #[allow(clippy::too_many_arguments)]
    fn gate_up(
        &self,
        gpu: &mut Gpu,
        expert_ptrs: &GpuTensor,
        nonowned_gate_up_dummy: Option<&GpuTensor>,
        topk_indices: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> Result<(), String>;

    fn rotate_x_batched(
        &self,
        gpu: &mut Gpu,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> Result<(), String>;

    #[allow(clippy::too_many_arguments)]
    fn down_expanded(
        &self,
        gpu: &mut Gpu,
        expert_ptrs: &GpuTensor,
        ownership_ptrs: &GpuTensor,
        nonowned_gate_up_dummy: Option<&GpuTensor>,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> Result<(), String>;

    #[allow(clippy::too_many_arguments)]
    fn down_residual_scaled(
        &self,
        gpu: &mut Gpu,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,
        residual: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> Result<(), String>;
}

/// Parameters for the deepseek4 bias-aware MoE decode arm (k=6, MQ2-Lloyd routed
/// experts). Kept distinct from [`MoeParams`] because the ds4 sub-graph has no
/// fused gate-side and no shared-expert block: the shared expert is a separate
/// model-owned step (`ffn_stub`) that runs first and seeds `ffn_out`, and this
/// arm's routed-down kernel atomic-accumulates into that same buffer.
///
/// `scores` is the post-`sqrt_softplus(gate·x)` router output — the model owns
/// the router GEMV + activation. Selection adds `gate_bias` while the routing
/// weights use the *unbiased* `scores`; the bias-aware kernel handles that
/// two-score semantic and folds in `route_scale`, all in one launch. The model
/// pre-rotates the activation, so `x_rot` is consumed as-is (no re-rotation).
pub struct MoeBiasAwareParams<'a> {
    // dims / config scalars
    pub hidden: usize,
    pub mi: usize,
    pub k_top: usize,
    pub n_exp: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    /// Model-local dispatch policy. The DS4 loader derives this from the
    /// verified MQ2R backend; generic GPU state must not influence it.
    pub uses_atomic_moe_down: bool,
    /// Optional exact-device MQ2 backend selected by the loaded DeepSeek4
    /// model. `None` retains the portable dispatcher for every other model and
    /// architecture.
    pub native_mq2_backend: Option<&'a dyn MoeBiasAwareMq2Backend>,
    /// EP-shard-only zero weight buffer. Exact-device backends may compare
    /// selected gate/up pointers against it to skip non-owned expert work
    /// while retaining the fixed graph shape. `None` on unsharded models.
    pub nonowned_gate_up_dummy: Option<&'a GpuTensor>,
    /// Token-batch width. Decode = 1. A value > 1 must route to the grouped
    /// prefill executor (Step 8), never this decode arm — guarded in the executor.
    pub batch_size: usize,
    // activations / residual
    /// FWHT-rotated activation (model pre-rotates; this arm does not re-rotate).
    pub x_rot: &'a GpuTensor,
    /// Residual stream the routed-down kernel atomic-accumulates into. The
    /// model's shared-expert step must have run first to seed this buffer.
    pub ffn_out: &'a GpuTensor,
    // router
    pub scores: &'a GpuTensor, // post-sqrt_softplus gate·x (weights use these)
    pub gate_bias: &'a GpuTensor, // per-expert routing bias (selection only)
    // routed expert pointer tables
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    // scratch buffers (model-owned)
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    /// `[k_top × hidden]` per-expert down outputs for the deterministic combine.
    pub down_expanded: &'a GpuTensor,
}

impl<'a> MoeBiasAwareParams<'a> {
    /// Borrow the routed-expert portion after route selection has already
    /// populated `topk_indices` and `topk_weights`. Heterogeneous DS4 uses
    /// this boundary to select routes on the dense owner and execute only the
    /// selected experts on the routed owner.
    pub fn selected(&self) -> MoeSelectedParams<'_> {
        MoeSelectedParams {
            hidden: self.hidden,
            mi: self.mi,
            k_top: self.k_top,
            swiglu_limit: self.swiglu_limit,
            uses_atomic_moe_down: self.uses_atomic_moe_down,
            native_mq2_backend: self.native_mq2_backend,
            nonowned_gate_up_dummy: self.nonowned_gate_up_dummy,
            batch_size: self.batch_size,
            x_rot: self.x_rot,
            ffn_out: self.ffn_out,
            expert_gate_up_ptrs: self.expert_gate_up_ptrs,
            expert_down_ptrs: self.expert_down_ptrs,
            topk_indices: self.topk_indices,
            topk_weights: self.topk_weights,
            gate_batch: self.gate_batch,
            up_batch: self.up_batch,
            rot_batch: self.rot_batch,
            down_expanded: self.down_expanded,
        }
    }
}

/// Selected routed-expert decode subgraph. Route selection is intentionally
/// absent: callers must provide the exact normalized IDs and weights produced
/// by the model-owned router. This is useful for split ownership where the
/// router and expert weights cannot reside on the same device.
pub struct MoeSelectedParams<'a> {
    pub hidden: usize,
    pub mi: usize,
    pub k_top: usize,
    pub swiglu_limit: f32,
    pub uses_atomic_moe_down: bool,
    pub native_mq2_backend: Option<&'a dyn MoeBiasAwareMq2Backend>,
    pub nonowned_gate_up_dummy: Option<&'a GpuTensor>,
    pub batch_size: usize,
    pub x_rot: &'a GpuTensor,
    pub ffn_out: &'a GpuTensor,
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    pub down_expanded: &'a GpuTensor,
}

// ── DeepSeek-V4 batched/prefill MoE parameters ─────────

/// Router-selection mode for the batched/prefill MoE path. DeepSeek-V4 uses
/// static hash routing for the first `num_hash_layers` layers and bias-aware
/// top-k for the rest; the executor branches on this.
pub enum MoePrefillRouting<'a> {
    /// Bias-aware batched top-k (select on `scores + gate_bias`, weight on the
    /// unbiased `scores`, normalize, `*route_scale`).
    BiasAware { gate_bias: &'a GpuTensor },
    /// Static `tid2eid` hash routing (layers `0..num_hash_layers`). `tokens` is
    /// the device-side `[B]` i32 token-id buffer.
    Hash {
        tid2eid: &'a GpuTensor,
        tokens: &'a GpuTensor,
    },
}

/// Parameters for the deepseek4 batched/prefill MoE (k=6, MQ2-Lloyd). The
/// model owns RMSNorm, the shared expert, the router GEMV + `sqrt_softplus`
/// (producing `scores`); this arm runs routing → routed experts → combine,
/// accumulating into `ffn_out` (the shared expert already seeded it).
///
/// Picks the grouped-GEMM path when `batch_size >= HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE`
/// (default 128), else the scalar K4 indexed path — mirroring `ffn_batched`.
pub struct MoeBiasAwarePrefillParams<'a> {
    // dims / config scalars
    pub hidden: usize,
    pub mi: usize,
    pub n_exp: usize,
    pub k_top: usize,
    pub batch_size: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    /// Model-local dispatch policy. The DS4 loader derives this from the
    /// verified MQ2R backend; generic GPU state must not influence it.
    pub uses_atomic_moe_down: bool,
    pub layer_idx: usize, // for the optional HIPFIRE_DEEPSEEK4_DUMP_TOPK header
    // routing
    pub routing: MoePrefillRouting<'a>,
    pub scores: &'a GpuTensor, // post-sqrt_softplus moe_scores_batch [B, n_exp]
    pub topk_indices: &'a GpuTensor, // [B, k_top] (routing out, expert in)
    pub topk_weights: &'a GpuTensor, // [B, k_top]
    // routed expert pointer tables
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    // activation / residual
    pub x_rot: &'a GpuTensor,   // ffn_x_rot_batch [B, hidden]
    pub ffn_out: &'a GpuTensor, // ffn_out_batch [B, hidden] (accumulate target)
    // grouped-path scratch
    pub expert_token_counts: &'a GpuTensor,
    pub expert_offsets: &'a GpuTensor,
    pub sorted_slot_index: &'a GpuTensor,
    pub expert_tile_ids: &'a GpuTensor,
    pub inverse_perm: &'a GpuTensor,
    pub y_gate_up_grouped: &'a GpuTensor,
    pub y_down_grouped: &'a GpuTensor,
    // shared scratch (grouped + scalar)
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    // scalar-path scratch (expanded deterministic down)
    pub down_expert_outputs: &'a GpuTensor,
}

// ── Qwen3.5 softmax-top-k MoE prefill parameters (Ship 4.2) ──

/// Parameters for the qwen35 batched/prefill MoE routed-expert block.
///
/// Distinct from [`MoeBiasAwarePrefillParams`] — qwen35 uses softmax top-k
/// routing (k=8) with MQ4/MQ6/Paro routed experts, a fused gate-side, and a
/// shared expert that seeds `x_batch` before this arm runs.
///
/// The model owns RMSNorm, the router GEMV + softmax top-k (producing
/// `topk_indices` / `topk_weights`), and the shared expert (which already
/// accumulated into `x_batch`). This arm runs scatter → gate_up → unscatter →
/// SwiGLU+rotate → down → combine, accumulating into `x_batch`.
///
/// All tensor refs are `&'a GpuTensor` (shared, not `&mut` — GpuTensor is Copy).
/// Scratch tensors are model-owned; the family holds only references.
pub struct MoePrefillParams<'a> {
    // dtype snapshot
    pub dtypes: MoeDtypes<'a>,
    pub recipe: MoeRecipe,
    /// Complete normalization/router/shared prelude. The routed tail below
    /// remains unchanged and consumes the declared route buffers.
    pub prelude: MoePrefillPrelude<'a>,
    // dims
    pub batch_size: usize,
    pub mi: usize,
    pub down_m: usize,
    pub down_k: usize,
    pub gate_up_k: usize,
    pub k_top: usize,
    pub n_exp: usize,
    /// m_total upper bound pre-computed by the model via
    /// `moe_grouped_m_total_bound(total_slots, n_exp)`. Used by Path 2
    /// scatter + grouped GEMM for grid sizing.
    pub m_total_max: usize,
    /// Model-level safety fence for promoted/mixed MQ6 checkpoints. When true,
    /// MQ4 grouped prefill calls use FP16 WMMA even for layers whose local
    /// routed dtype snapshot is pure MQ4. This keeps pure MQ4 models on the
    /// existing i8 default while avoiding mixed-checkpoint corruption.
    pub force_mq4_grouped_fp16: bool,
    // routing inputs (model-produced)
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    // destination = x_batch (residual; combine accumulates here)
    pub x_batch: &'a GpuTensor,
    // activation buffers
    pub x_norm_batch: &'a GpuTensor,
    pub x_rot_batch: &'a GpuTensor,
    // routed gate_up/down pointer tables
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    /// Exact live expert owners bound to the sealed load-time identity.
    pub routed_experts: &'a dyn RoutedExpertWeights,
    /// Route A MoE-AWQ: per-routed-expert down `awq_scale` pointer table (see
    /// [`MoeParams::expert_down_awq_ptrs`]). When `Some`, the prefill silu+rotate
    /// uses the indexed AWQ kernel (per-slot scale via `topk_indices`),
    /// superseding the single-scale `down_awq_scale` stub below for routed
    /// experts. `None` (default) = plain silu+rotate.
    pub expert_down_awq_ptrs: Option<&'a GpuTensor>,
    /// Per-expert mixed-precision prefill: `[n_exp]` u8 dtype-tag table,
    /// `Some` iff the routed experts carry mixed dtypes (graded T3-3L). Drives
    /// the merged grouped-WMMA prefill kernel. `None` ⇒ uniform path, byte-identical.
    pub expert_dtype_tags: Option<&'a GpuTensor>,
    // intermediate buffers
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    // Path 1 expanded-down scratch
    pub down_expanded: &'a GpuTensor,
    // Path 2 scatter scratch (model-owned)
    pub expert_token_counts: &'a GpuTensor,
    pub expert_offsets: &'a GpuTensor,
    pub sorted_slot_index: &'a GpuTensor,
    pub expert_tile_ids: &'a GpuTensor,
    pub inverse_perm: &'a GpuTensor,
    pub y_gate_up_grouped: &'a GpuTensor,
    pub y_down_grouped: &'a GpuTensor,
    // paro sidecars (per-layer shared Givens rotation tables)
    pub paro_gate_up: Option<GivensRef<'a>>,
    pub paro_down: Option<GivensRef<'a>>,
    /// AWQ scale for the routed down weight (experts[0].down.awq_scale).
    /// Used by the AWQ-aware silu+rotate step. `None` when the routed
    /// experts are non-AWQ (the common case for A3B).
    pub down_awq_scale: Option<&'a GpuTensor>,
    /// EP (Ship 6 substrate-EP prefill): when `Some`, the **routed** combine
    /// accumulates into this **zeroed** `[batch × dim]` partial instead of
    /// `x_batch`; the EP prefill driver then all-reduce-sums the partial across
    /// ranks and adds it into each rank's `x_batch`. The **shared** expert stays
    /// in `x_batch` (replicated per rank — added once to each rank's own copy,
    /// no all-reduce). `None` (the default) accumulates routed into `x_batch`,
    /// byte-identical to pre-EP behavior.
    pub routed_out: Option<&'a GpuTensor>,
}
/// Dtype-only admission snapshot for grouped MoE prefill.
///
/// This record intentionally has no architecture dependency. Model loaders
/// assemble it from their owned weights and pass it to the shared policy.
#[derive(Debug, Clone, Copy)]
pub struct MoePrefillDtypes {
    pub router: DType,
    pub shared_expert_scalar_gate: DType,
    pub shared_expert_gate: DType,
    pub shared_expert_up: DType,
    pub shared_expert_down: DType,
    pub expert_gate_up: DType,
    pub expert_down: DType,
    pub expert_gate_up_uniform: bool,
    pub expert_down_uniform: bool,
    pub routed_mixed_merged: bool,
}

impl MoePrefillDtypes {
    /// Construct a uniform prefill dtype snapshot.
    pub fn uniform(dtype: DType) -> Self {
        Self {
            router: dtype,
            shared_expert_scalar_gate: dtype,
            shared_expert_gate: dtype,
            shared_expert_up: dtype,
            shared_expert_down: dtype,
            expert_gate_up: dtype,
            expert_down: dtype,
            expert_gate_up_uniform: true,
            expert_down_uniform: true,
            routed_mixed_merged: false,
        }
    }
}

#[inline]
pub fn paro_batched_admit_enabled_from_env(value: Option<&str>) -> bool {
    value == Some("1")
}

#[inline]
pub fn moe_prefill_topk_shape_supported(k_top: usize, num_experts: usize) -> bool {
    k_top == 8 && num_experts <= 1024
}

/// Quantized routed-expert formats with an implemented Cohere sigmoid
/// grouped-prefill path.
#[inline]
pub fn sigmoid_routed_prefill_supported(gate_up: DType) -> bool {
    matches!(
        gate_up,
        DType::MQ4G256 | DType::HFQ4G256 | DType::MQ6G256 | DType::HFQ6G256
    )
}

#[inline]
pub fn routed_codebook_grouped_supported(dt: DType) -> bool {
    matches!(
        dt,
        DType::MQ4G256 | DType::MQ2G256Lloyd | DType::MQ3G256Lloyd
    )
}

#[inline]
pub fn routed_uniform_mqv2_grouped_supported(dt: DType) -> bool {
    matches!(dt, DType::MQ4G256V2 | DType::MQ6G256V2)
}

#[inline]
pub fn routed_codebook_pair_batched_supported(gate_up: DType, down: DType) -> bool {
    routed_codebook_grouped_supported(gate_up)
        && routed_codebook_grouped_supported(down)
        && (matches!(gate_up, DType::MQ2G256Lloyd | DType::MQ3G256Lloyd)
            || matches!(down, DType::MQ2G256Lloyd | DType::MQ3G256Lloyd))
}

pub fn codebook_batched_admit_enabled_from_env(
    value: Option<&str>,
    grouped_gemm_value: Option<&str>,
    _arch: &str,
) -> bool {
    let grouped_gemm_on = !matches!(grouped_gemm_value, Some("0") | Some("off"));
    match value {
        Some("0") | Some("off") | Some("false") => false,
        Some("1") | Some("on") | Some("true") => grouped_gemm_on,
        _ => false,
    }
}

pub fn codebook_batched_admit_enabled(arch: &str) -> bool {
    codebook_batched_admit_enabled_from_env(
        hipfire_config::developer_var("HIPFIRE_MOE_CODEBOOK_BATCHED")
            .ok()
            .as_deref(),
        hipfire_config::developer_var("HIPFIRE_MOE_GROUPED_GEMM")
            .ok()
            .as_deref(),
        arch,
    )
}

/// Whether MQ6 grouped prefill is admitted by the process policy.
pub fn mq6_batched_admit_enabled_from_env(value: Option<&str>, arch: &str) -> bool {
    match value {
        Some("0") | Some("off") | Some("false") => false,
        Some("1") | Some("on") | Some("true") => true,
        _ => arch.starts_with("gfx11") || arch.starts_with("gfx12"),
    }
}

pub fn gated_moe_prefill_admissible_for_dtypes(
    dtypes: &MoePrefillDtypes,
    admit_mq6: bool,
    admit_paro: bool,
    admit_e8: bool,
    admit_codebook: bool,
) -> bool {
    let router_ok = matches!(
        dtypes.router,
        DType::MQ4G256 | DType::MQ4G256V2 | DType::Q8_0 | DType::F32
    );
    let shared_gate_ok = matches!(
        dtypes.shared_expert_scalar_gate,
        DType::MQ4G256 | DType::MQ4G256V2 | DType::Q8_0 | DType::F32
    );
    let routed_ok =
        dtypes.routed_mixed_merged || (dtypes.expert_gate_up_uniform && dtypes.expert_down_uniform);
    if !(router_ok && shared_gate_ok && routed_ok) {
        return false;
    }

    let is_e8_family = |dt| matches!(dt, DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8);
    if dtypes.routed_mixed_merged {
        let shared_gu_ok = (dtypes.shared_expert_gate == dtypes.shared_expert_up
            && matches!(dtypes.shared_expert_gate, DType::MQ4G256 | DType::MQ4G256V2))
            || (admit_mq6
                && matches!(
                    dtypes.shared_expert_gate,
                    DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2
                )
                && dtypes.shared_expert_up == dtypes.shared_expert_gate);
        let shared_dn_ok = matches!(dtypes.shared_expert_down, DType::MQ4G256 | DType::MQ4G256V2)
            || (admit_mq6
                && matches!(dtypes.shared_expert_down, DType::MQ6G256 | DType::MQ6G256V2));
        return shared_gu_ok && shared_dn_ok;
    }

    if admit_e8
        && dtypes.shared_expert_gate == DType::Q8_0
        && dtypes.shared_expert_up == DType::Q8_0
        && dtypes.shared_expert_down == DType::Q8_0
        && is_e8_family(dtypes.expert_gate_up)
        && is_e8_family(dtypes.expert_down)
    {
        return true;
    }

    if admit_e8
        && is_e8_family(dtypes.expert_gate_up)
        && is_e8_family(dtypes.expert_down)
        && dtypes.shared_expert_gate == dtypes.shared_expert_up
        && matches!(
            dtypes.shared_expert_gate,
            DType::Q8_0 | DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8
        )
        && matches!(
            dtypes.shared_expert_down,
            DType::Q8_0 | DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8
        )
    {
        return true;
    }

    if admit_codebook
        && routed_codebook_pair_batched_supported(dtypes.expert_gate_up, dtypes.expert_down)
    {
        let shared_gu_ok = (dtypes.shared_expert_gate == dtypes.shared_expert_up
            && matches!(dtypes.shared_expert_gate, DType::MQ4G256 | DType::MQ4G256V2))
            || (admit_mq6
                && matches!(
                    dtypes.shared_expert_gate,
                    DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2
                )
                && dtypes.shared_expert_up == dtypes.shared_expert_gate);
        let shared_dn_ok = matches!(dtypes.shared_expert_down, DType::MQ4G256 | DType::MQ4G256V2)
            || (admit_mq6
                && matches!(dtypes.shared_expert_down, DType::MQ6G256 | DType::MQ6G256V2));
        if shared_gu_ok && shared_dn_ok {
            return true;
        }
    }

    if admit_paro
        && dtypes.shared_expert_gate == DType::ParoQ4G128
        && dtypes.shared_expert_up == DType::ParoQ4G128
        && dtypes.shared_expert_down == DType::ParoQ4G128
        && dtypes.expert_gate_up == DType::ParoQ4G128
        && dtypes.expert_down == DType::ParoQ4G128
    {
        return true;
    }

    if admit_mq6 {
        let shared_gu_dt = dtypes.shared_expert_gate;
        let shared_gu_ok = matches!(
            shared_gu_dt,
            DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2
        ) && dtypes.shared_expert_up == shared_gu_dt;
        let shared_dn_ok = matches!(
            dtypes.shared_expert_down,
            DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2
        );
        let experts_ok = matches!(
            dtypes.expert_gate_up,
            DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2
        ) && matches!(
            dtypes.expert_down,
            DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2
        );
        shared_gu_ok && shared_dn_ok && experts_ok
    } else {
        dtypes.shared_expert_gate == dtypes.shared_expert_up
            && matches!(dtypes.shared_expert_gate, DType::MQ4G256 | DType::MQ4G256V2)
            && matches!(dtypes.shared_expert_down, DType::MQ4G256 | DType::MQ4G256V2)
            && matches!(dtypes.expert_gate_up, DType::MQ4G256 | DType::MQ4G256V2)
            && matches!(dtypes.expert_down, DType::MQ4G256 | DType::MQ4G256V2)
    }
}

/// Shared admission policy for Qwen-style softmax gated/shared prefill.
pub fn gated_moe_prefill_admissible(
    dtypes: &MoePrefillDtypes,
    admit_mq6: bool,
    arch: &str,
) -> bool {
    let admit_e8 = matches!(
        arch,
        "gfx1100"
            | "gfx1101"
            | "gfx1102"
            | "gfx1150"
            | "gfx1151"
            | "gfx1152"
            | "gfx1200"
            | "gfx1201"
    ) && hipfire_config::developer_var("HIPFIRE_E8_GFX12")
        .ok()
        .as_deref()
        == Some("1");
    let admit_paro = paro_batched_admit_enabled_from_env(
        hipfire_config::developer_var("HIPFIRE_PARO_BATCHED")
            .ok()
            .as_deref(),
    );
    let admit_codebook = codebook_batched_admit_enabled(arch);
    gated_moe_prefill_admissible_for_dtypes(dtypes, admit_mq6, admit_paro, admit_e8, admit_codebook)
}

/// Resolved dispatch plan for the qwen35 batched MoE prefill routed block.
///
/// Distinct from [`MoeResolution`] (decode) — prefill adds the Path 0/1/2
/// grouped-vs-scalar down selection and the Paro i8/k8 levers.
/// Pure function of [`MoeDtypes`] + arch + [`FeatureFlags`].
pub struct MoePrefillResolution {
    /// Gate_up + down via grouped-GEMM scatter pipeline (Path 2).
    /// Requires WMMA-capable arch (gfx11/gfx12) + `moe_grouped_gemm` flag.
    pub use_path2: bool,
    /// Down uses atomic-accumulate GEMV (Path 0) instead of atomic-free
    /// expanded+combine (Path 1). gfx9* wave64 archs (gfx906/gfx908/gfx94x).
    pub down_path0: bool,
    /// gfx1151 Paro i8 MMQ grouped GEMM (Path 2 only).
    pub use_paro_i8: bool,
    /// gfx1151 Paro i8 MMQ k8 grouped GEMM (Path 2 only).
    pub use_paro_i8_k8: bool,
    /// Routed experts use ParoQ4G128 (determines SwiGLU+rotate kernel selection).
    pub paro_mode: bool,
    /// gfx1151's HFQ4 grouped-i8 path is correct for pure MQ4, but corrupts
    /// MQ6-promoted A3B MTP prefill when the same MoE layer mixes MQ4 and MQ6
    /// projections. Default mixed layers back to FP16 WMMA; explicit
    /// HIPFIRE_MOE_GROUPED_I8=1 still opts into the research path.
    pub force_mq4_grouped_fp16: bool,
}

impl MoePrefillResolution {
    /// Resolve the prefill dispatch plan from dtypes, arch, and flags.
    ///
    /// Reads MoE prefill env levers from `flags` (parsed once at `Gpu::init`),
    /// not `std::env` — mid-prefill env mutation is not honored.
    pub fn resolve(
        d: &MoeDtypes<'_>,
        arch: &rdna_compute::arch_caps::ArchCaps,
        flags: &rdna_compute::feature_flags::FeatureFlags,
    ) -> Self {
        let paro_mode = d.routed_gate_up == DType::ParoQ4G128 && d.has_paro_shared;
        let use_path2 = flags.moe_grouped_gemm && arch.has_wmma();
        // MQ6 / MQ6V2 grouped-WMMA: gfx11 `_k2` kernel now exists (alongside the
        // gfx12 `_gfx12` / `mq6g256v2` sisters). Only suppress Path 2 on archs
        // that have NEITHER (gfx9*, gfx1010/1030, CDNA) — i.e. no wmma_w32 and
        // not gfx12. gfx1100/1101/1102/1103/1150/1151/1152 all have wmma_w32.
        // (Master's narrower gfx1151-only MQ6 admit (dfed8cc6) is subsumed by
        // this wider gfx11 widen (8d555fc6); master's mixed-checkpoint safety
        // is preserved separately via `force_mq4_grouped_fp16` below.)
        // qt47 (MQ6G256V2) shares the same gfx11/gfx12 grouped availability —
        // never collapse it onto the V1 MQ6G256 path (dual-half vs f32 header).
        let mq6_on_non_wmma = matches!(d.routed_gate_up, DType::MQ6G256 | DType::MQ6G256V2)
            && !arch.has_wmma_w32()
            && !(arch.is_gfx1200() || arch.is_gfx1201());
        let use_path2 = use_path2 && !mq6_on_non_wmma;
        // MQ5 grouped-WMMA (`gemm_hfq5g256_moe_grouped_wmma`) is gfx12-only
        // (same as MQ6) — fall back to Path 1 (indexed batched GEMV) on
        // gfx11/gfx9 to avoid the gfx12-only kernel panic.
        let mq5_on_non_gfx12 =
            d.routed_gate_up == DType::MQ5G256 && !(arch.is_gfx1200() || arch.is_gfx1201());
        let use_path2 = use_path2 && !mq5_on_non_gfx12;
        // Mixed per-expert: the merged grouped kernel covers all four dtype
        // tags on any WMMA arch (gfx11 _k2 or gfx12 .gfx12). The routed
        // representative dtype may be MQ6/MQ5 and trip the suppression above,
        // so re-admit Path 2 when the file is graded-mixed (tag table present).
        let use_path2 =
            use_path2 || (d.routed_has_mixed_experts && flags.moe_grouped_gemm && arch.has_wmma());
        // mfp4-E8 routed experts: use Path 2 (grouped-WMMA) on gfx1151 and gfx12
        // (RDNA4). Both have a native E8 grouped-WMMA GEMM kernel:
        //   gfx1151 → gemm_mfp4g32_e8_moe_grouped_wmma (gfx1151.hip)
        //   gfx12   → gemm_mfp4g32_e8_moe_grouped_wmma_gfx12 (gfx12.hip)
        // Other archs (gfx1100 dGPU, gfx9*/CDNA) have no grouped E8 sister → Path 1.
        // mfp4-E8 grouped-WMMA prefill on ALL WMMA arches (RDNA3 gfx11 + RDNA4
        // gfx12). The gfx1151 kernel uses the RDNA3 wave32-WMMA builtin and runs
        // correctly on gfx1100/1101/1102; gfx12 uses its .gfx12 sister. The prior
        // "gfx1151-only / gfx1100 wash" call rested on pp512 97.5-vs-97.6 — which is
        // DECODE tok/s, not prefill throughput (a prefill change can't move decode
        // tok/s). Real prefill throughput is what bench_sweep measures, so route
        // gfx1100 through Path 2 and re-measure. Only ever active under the
        // HIPFIRE_E8_GFX12 batched-prefill gate.
        let e8_no_grouped = matches!(
            d.routed_gate_up,
            DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8
        ) && !(arch.is_rdna3() || arch.is_rdna4());
        let use_path2 = use_path2 && !e8_no_grouped;
        // Path 0: gfx9* wave64 archs (gfx906/gfx908/gfx94x) — cheap HBM
        // atomics make the atomic GEMV pattern competitive vs expanded scratch.
        let down_path0 = arch.is_gcn5() || arch.is_cdna1() || arch.is_cdna3();
        let is_gfx1151 = arch.is_gfx1151();
        let use_paro_i8 = paro_mode && use_path2 && is_gfx1151 && flags.moe_paro_i8.unwrap_or(true);
        let use_paro_i8_k8 = use_paro_i8 && flags.moe_paro_i8_k8.unwrap_or(true);
        let force_mq4_grouped_fp16 =
            use_path2 && is_gfx1151 && d.has_mq6_projection() && flags.moe_grouped_i8.is_none();
        Self {
            use_path2,
            down_path0,
            use_paro_i8,
            use_paro_i8_k8,
            paro_mode,
            force_mq4_grouped_fp16,
        }
    }
}

use crate::tables::moe_table;

// ── Family ─────────────────────────────────────────────

pub struct MoeFamily {
    registry: KernelRegistry,
}

impl MoeFamily {
    pub fn new() -> Self {
        let mut registry = KernelRegistry::new();
        moe_table::populate(&mut registry);
        registry
            .validate()
            .expect("moe kernel table has empty entries");
        Self { registry }
    }

    pub fn registry(&self) -> &KernelRegistry {
        &self.registry
    }

    /// Resolve the best kernel key for the given MoE variant.
    ///
    /// Applies arch gating through `KernelRegistry::resolve`.
    pub fn resolve(
        &self,
        variant: MoeVariant,
        ctx: &DispatchCtx,
        shape: Option<&ShapeInfo>,
    ) -> Result<&KernelVariant, DispatchError> {
        let key = match variant {
            MoeVariant::IndexedGateUp => KernelKey::MoeIndexedGateUpLloyd,
            MoeVariant::IndexedDown => KernelKey::MoeIndexedDownLloyd,
            MoeVariant::GroupedGemm => KernelKey::MoeGroupedGemm,
        };
        self.registry.resolve(key, ctx, shape)
    }

    /// Run a single-token deepseek4 bias-aware MoE decode step (k=6, MQ2-Lloyd
    /// routed experts). Delegates to [`crate::pipeline::run_moe_decode_bias_aware`].
    ///
    /// The model owns the router GEMV + `sqrt_softplus` (producing
    /// `params.scores`) and the shared expert (`ffn_stub`, which seeds
    /// `params.ffn_out`); this entry runs only the bias-aware top-k + routed
    /// MQ2-Lloyd expert sub-graph.
    ///
    /// Takes no `DispatchCtx`: the bias-aware path dispatches fixed MQ2-Lloyd
    /// kernels with no arch-gated sub-dispatch, so building a `DispatchCtx`
    /// per layer per token (an uncached generic policy parse) would
    /// be pure waste on the decode hot path.
    pub fn run_bias_aware(
        &self,
        gpu: &mut rdna_compute::Gpu,
        params: &MoeBiasAwareParams,
    ) -> Result<(), DispatchError> {
        crate::pipeline::run_moe_decode_bias_aware(gpu, params)
    }

    /// Run only the selected-expert portion of the single-token DeepSeek4
    /// MQ2-Lloyd subgraph. The caller owns route selection and must already
    /// have populated `topk_indices` and `topk_weights`.
    pub fn run_selected(
        &self,
        gpu: &mut rdna_compute::Gpu,
        params: &MoeSelectedParams,
    ) -> Result<(), DispatchError> {
        crate::pipeline::run_moe_decode_selected(gpu, params)
    }

    /// Run a batched/prefill deepseek4 MoE step (k=6, MQ2-Lloyd): routing
    /// (bias-aware or hash) → routed experts (grouped GEMM when
    /// `batch_size >= gate`, else scalar K4 indexed) → combine, accumulating
    /// into `params.ffn_out`. Delegates to
    /// [`crate::pipeline::run_moe_prefill_bias_aware`]. The model owns RMSNorm,
    /// the shared expert, and the router GEMV + `sqrt_softplus`.
    pub fn run_bias_aware_prefill(
        &self,
        gpu: &mut rdna_compute::Gpu,
        params: &MoeBiasAwarePrefillParams,
    ) -> Result<(), DispatchError> {
        crate::pipeline::run_moe_prefill_bias_aware(gpu, params)
    }
}

impl KernelFamily for MoeFamily {
    fn name(&self) -> &'static str {
        "moe"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_mq4() -> MoeDtypes<'static> {
        MoeDtypes {
            router: DType::MQ4G256,
            shared: Some(MoeSharedDtypes {
                selector: DType::MQ4G256,
                gate: DType::MQ4G256,
                up: DType::MQ4G256,
                down: DType::MQ4G256,
            }),
            experts_all_gate_up_mq4: true,
            routed_gate_up: DType::MQ4G256,
            routed_down: DType::MQ4G256,
            routed_has_mixed_experts: false,
            has_paro_shared: false,
            per_expert_gate_up: None,
            per_expert_down: None,
        }
    }

    #[test]
    fn qwen_gate_prerotation_requires_exact_v1_or_ornith_v2_quartet() {
        assert!(softmax_shared_gate_uniform_mq4(
            DType::MQ4G256,
            DType::MQ4G256,
            DType::MQ4G256,
            DType::MQ4G256,
        ));
        assert!(!softmax_shared_gate_uniform_mq4(
            DType::MQ4G256V2,
            DType::MQ4G256V2,
            DType::MQ4G256V2,
            DType::MQ4G256V2,
        ));
        let exact = [
            (DType::MQ4G256V2, 256, 2_048, false),
            (DType::MQ4G256V2, 1, 2_048, false),
            (DType::MQ4G256V2, 512, 2_048, false),
            (DType::MQ4G256V2, 512, 2_048, false),
        ];
        assert!(softmax_shared_gate_mq4v2_layouts(exact));
        for slot in 0..4 {
            let mut mixed = exact;
            mixed[slot].0 = DType::MQ4G256;
            assert!(!softmax_shared_gate_mq4v2_layouts(mixed));
            let mut awq = exact;
            awq[slot].3 = true;
            assert!(!softmax_shared_gate_mq4v2_layouts(awq));
        }
        let mut wrong_shape = exact;
        wrong_shape[2].1 = 511;
        assert!(!softmax_shared_gate_mq4v2_layouts(wrong_shape));
    }

    #[test]
    fn resolve_none_per_expert_is_not_mixed() {
        let d = uniform_mq4();
        let r = MoeResolution::resolve(&d, 8);
        assert!(!r.mixed);
    }

    #[test]
    fn resolve_some_per_expert_with_varied_tiers_is_mixed() {
        let mut d = uniform_mq4();
        d.per_expert_gate_up = Some(&[DType::MQ4G256, DType::MQ6G256]); // varies
        d.per_expert_down = Some(&[DType::MQ4G256, DType::MQ6G256]);
        let r = MoeResolution::resolve(&d, 8);
        assert!(r.mixed);
    }

    #[test]
    fn resolve_empty_per_expert_table_is_not_mixed_and_does_not_panic() {
        // A degenerate empty table must not index v[0]; it collapses to uniform.
        let mut d = uniform_mq4();
        d.per_expert_gate_up = Some(&[]);
        d.per_expert_down = Some(&[]);
        let r = MoeResolution::resolve(&d, 8);
        assert!(!r.mixed);
    }

    #[test]
    fn resolve_some_per_expert_all_same_is_not_mixed() {
        // a per-expert table that is uniform should NOT trigger the mixed path
        let mut d = uniform_mq4();
        d.per_expert_gate_up = Some(&[DType::MQ4G256, DType::MQ4G256]);
        d.per_expert_down = Some(&[DType::MQ4G256, DType::MQ4G256]);
        let r = MoeResolution::resolve(&d, 8);
        assert!(
            !r.mixed,
            "a uniform per-expert table must take the fast uniform path"
        );
    }

    #[test]
    fn resolve_mq6v2_uniform_is_indexable() {
        // qt47 uniform: both projections MQ6G256V2 => indexable, GPU top-K on.
        let mut d = uniform_mq4();
        d.routed_gate_up = DType::MQ6G256V2;
        d.routed_down = DType::MQ6G256V2;
        d.experts_all_gate_up_mq4 = false;
        let r = MoeResolution::resolve(&d, 8);
        assert!(r.routed_indexable_mq6v2);
        assert!(!r.routed_indexable_mq6, "must not claim the V1 MQ6 arm");
        assert!(!r.routed_indexable_mq4);
        assert!(!r.routed_indexable_mq4v2);
        assert!(r.use_gpu_topk);
        assert!(r.needs_x_rot_local, "qt47 kernels read ROTATED activations");
        assert!(r.routed_indexable());
    }

    #[test]
    fn resolve_mq6v2_mixed_with_v1_is_not_indexable() {
        // Same dual-half hazard as mq4v2/qt13: V1 and V2 share the 200 B group
        // stride and 6-bit packing, differing ONLY in the 8-byte header. A
        // split pairing must NOT be indexable on either arm.
        for (gu, dn) in [
            (DType::MQ6G256V2, DType::MQ6G256),
            (DType::MQ6G256, DType::MQ6G256V2),
            (DType::MQ6G256V2, DType::MQ4G256),
            (DType::MQ4G256V2, DType::MQ6G256V2),
        ] {
            let mut d = uniform_mq4();
            d.routed_gate_up = gu;
            d.routed_down = dn;
            d.experts_all_gate_up_mq4 = false;
            let r = MoeResolution::resolve(&d, 8);
            assert!(!r.routed_indexable_mq6v2, "{gu:?}/{dn:?}");
            assert!(!r.routed_indexable_mq6, "{gu:?}/{dn:?}");
            assert!(
                !r.use_gpu_topk,
                "{gu:?}/{dn:?} must fall back, not guess a layout"
            );
        }
    }

    #[test]
    fn resolve_mq6_v1_still_indexable_without_mq6v2() {
        // Preserve V1: uniform MQ6G256 must keep the V1 arm and never claim V2.
        let mut d = uniform_mq4();
        d.routed_gate_up = DType::MQ6G256;
        d.routed_down = DType::MQ6G256;
        d.experts_all_gate_up_mq4 = false;
        let r = MoeResolution::resolve(&d, 8);
        assert!(r.routed_indexable_mq6);
        assert!(!r.routed_indexable_mq6v2);
        assert!(r.use_gpu_topk);
        assert!(r.needs_x_rot_local);
    }

    #[test]
    fn has_mq6_projection_recognizes_v1_and_v2() {
        let mut d = uniform_mq4();
        assert!(!d.has_mq6_projection());

        d.routed_down = DType::MQ6G256;
        assert!(
            d.has_mq6_projection(),
            "V1 MQ6 must trip has_mq6_projection"
        );

        d.routed_down = DType::MQ6G256V2;
        assert!(
            d.has_mq6_projection(),
            "V2 MQ6 must trip has_mq6_projection"
        );

        d.routed_down = DType::MQ4G256;
        d.shared = Some(MoeSharedDtypes {
            selector: DType::MQ4G256,
            gate: DType::MQ6G256V2,
            up: DType::MQ4G256,
            down: DType::MQ4G256,
        });
        assert!(
            d.has_mq6_projection(),
            "shared-expert MQ6V2 must trip has_mq6_projection"
        );

        d.shared = Some(MoeSharedDtypes {
            selector: DType::MQ4G256,
            gate: DType::MQ4G256V2,
            up: DType::MQ4G256,
            down: DType::MQ4G256,
        });
        assert!(
            !d.has_mq6_projection(),
            "MQ4V2 must not be treated as an MQ6 projection"
        );
    }

    #[test]
    fn mixed_supported_tiers_include_v1_and_v2_affine() {
        // Kernel branch tags 7..18 consume MQ4V2/MQ6V2; admission must list
        // them alongside the preserved V1 tiers. Exact membership — no MQ2/3/5V2.
        assert!(MIXED_SUPPORTED_TIERS.contains(&DType::MQ4G256));
        assert!(MIXED_SUPPORTED_TIERS.contains(&DType::MQ6G256));
        assert!(MIXED_SUPPORTED_TIERS.contains(&DType::ParoQ4G128));
        assert!(MIXED_SUPPORTED_TIERS.contains(&DType::MQ4G256V2));
        assert!(MIXED_SUPPORTED_TIERS.contains(&DType::MQ6G256V2));
        assert_eq!(MIXED_SUPPORTED_TIERS.len(), 5);
        assert!(!MIXED_SUPPORTED_TIERS.contains(&DType::MQ5G256V2));
        assert!(!MIXED_SUPPORTED_TIERS.contains(&DType::MQ2G256V2));
        assert!(!MIXED_SUPPORTED_TIERS.contains(&DType::MQ3G256V2));
    }

    #[test]
    fn resolve_mq6v2_k_ne_8_disables_gpu_topk() {
        let mut d = uniform_mq4();
        d.routed_gate_up = DType::MQ6G256V2;
        d.routed_down = DType::MQ6G256V2;
        d.experts_all_gate_up_mq4 = false;
        let r = MoeResolution::resolve(&d, 6);
        assert!(r.routed_indexable_mq6v2);
        assert!(!r.use_gpu_topk);
    }
}
