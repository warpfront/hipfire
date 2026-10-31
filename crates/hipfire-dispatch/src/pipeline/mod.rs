// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
use crate::context::DispatchCtx;
use crate::families::fused_qkv::{fused_gate_up_key_for, FusedQkvFamily, FusedQkvParams};
use crate::families::gemm::{GemmFamily, GemmParams};
use crate::families::gemv::{GemvFamily, WeightRef};
use crate::families::moe::{MoeMq2Lloyd, MoeRouteCapability, MOE_GROUPED_BLOCK_M};
use crate::tables::KernelRegistry;
use crate::types::*;
#[allow(unused_imports)]
use hip_bridge;
use rdna_compute::tensor_ops::{bf16_scaled_add, Bf16ScaledAdd};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::cell::OnceCell;
use std::sync::{LazyLock, OnceLock};

#[cfg(feature = "moe-oracle")]
pub mod router_observer;
pub mod sealed_moe;
pub use sealed_moe::{
    checked_grouped_m_total_bound, checked_tp_local_shapes, seal_decode, seal_ep_routed_contrib,
    seal_prefill, seal_prefill_ep, ActivationIdentity, ActivationInput, BoundMoeExperts,
    ExpertBindingCache, ExpertMetadata, ExpertResource, ExpertResources, ExpertTable,
    MoeContribution, MoePrefillRouteProducerProof, MoeProtocol, MoeRouteProducerProof,
    MoeRouterInput, MoeSharedContribution, PrefillRouteMode, ResourceAlias, SealedMoeCall,
};
pub(crate) mod moe_program;
pub(crate) mod qt44_qt53_prefill;
pub mod slab_moe_decode;
pub mod slab_route;
pub mod slowlink_prefill;
pub mod sp_layer;
pub use moe_program::{
    execute_gated_ffn, execute_grouped_moe, CastPolicy, CombineOrder, FfnActivation, GatedFfnOp,
    GroupedMoeActTier, GroupedMoeGroupedScratch, GroupedMoeIndexedScratch, GroupedMoeOp,
    GroupedMoePolicy, GroupedMoeProtocol, GroupedMoeRouted, GroupedMoeRouter, GroupedMoeScratch,
    GroupedMoeScratchPlan, GroupedMoeShared, GroupedMoeSharedScratch, RouterKind,
    SharedExpertPolicy,
};
pub use moe_program::{MoeStage, SealedMoeOp};
pub(crate) mod draft_head;
pub mod moe_dump;
pub(crate) mod norm_fold;
pub use draft_head::{DraftHead, DraftHeadLayout, DraftHeadPolicy, DraftHeadRequestState};
pub mod batched;
#[cfg(feature = "deltanet")]
pub mod batched_attention;
#[cfg(feature = "deltanet")]
pub mod batched_deltanet;
#[cfg(feature = "deltanet")]
pub mod batched_moe;
pub mod gated_mlp;
#[cfg(feature = "deltanet")]
pub mod hybrid;
pub mod latent_attention;
pub(crate) mod layer_ops;
pub mod mla;
pub mod projection_split;
pub mod sandwich;
pub mod step_graph;
pub(crate) mod steps;
pub mod stream_mix;
pub use layer_ops::{
    execute_argmax, execute_broadcast_add, execute_clear, execute_embedding, execute_final_hyper,
    execute_gated_delta_net, execute_gated_delta_net_arena, execute_gated_delta_net_core_rows,
    execute_gated_delta_net_uncovered, execute_grouped_depthwise, execute_grouped_depthwise_rows,
    execute_hyper_norm, execute_hyper_read, execute_hyper_write, execute_indexed_attention,
    execute_indexed_attention_arena, execute_indexed_attention_core_rows,
    execute_indexed_attention_uncovered, execute_lm_head, execute_project, gdn_qkv_projection_bf16,
    project_weight, project_weights, qsa_projection_hook_installed, reset_qsa_projection_slot,
    set_qsa_projection_hook, validate_lm_head, BroadcastAddOp, ClearOp, CoreWindow, EmbeddingOp,
    GatedDeltaNetOp, GatedShortConvOp, GdnRowCapture, GdnTrunkArenas, GroupedDepthwiseOp,
    HyperNormOp, HyperReadOp, HyperWriteOp, IndexedAttentionMode, IndexedAttentionOp,
    IndexedAttentionState, LaneAttentionOp, MixerPhase, ProjectLadder, ProjectOp,
    QsaProjectionHook, QsaTrunkArenas,
};
pub use step_graph::{
    op_kind_of, Alias, OpDecl, OpId, OpKind, ResourceClass, ResourceDecl, ResourceId, SplitRule,
    StateClass, StepGraph, StepGraphError,
};
pub use steps::{
    execute_steps, execute_validated_steps, execute_validated_steps_window,
    execute_validated_steps_with_moe_hooks, validate_steps, FusedPattern, GemvInput,
    HcFusionCursor, MoeStepHooks, Step,
};

/// The rows an operation that serves both decode and batched rows runs.
/// The route alone picks decode or batched kernels; `rows` is only the
/// extent, so a one-row prefill chunk keeps its batched arithmetic.
/// Operations whose batched chain needs route-only operands carry a per-op
/// `XRoute::{Decode { .. }, Batched { .. }}` instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowsRoute {
    /// One decode row.
    Decode,
    /// A prefill chunk.
    Prefill,
    /// A speculative-verify block.
    Verify,
}

impl RowsRoute {
    /// Whether the rows take the batched kernels.
    pub fn batched(self) -> bool {
        self != Self::Decode
    }
}

// #397 Ship 6 — forward-as-pipeline C-design lowered super-op substrate (types
// only at this step; not on any live path until wired behind HIPFIRE_FORWARD_LOWERED).
pub mod superop;

/// Generic-pipeline refusal for the typed-only formats: MQ4G128V2 (qt=53) and
/// the planar MQ3-I4 / MQ2-I4 dtypes (qt=131..=134).
fn reject_mq4g128v2(dtype: DType, family: &'static str) -> Result<(), DispatchError> {
    if dtype == DType::MQ4G128V2 {
        return Err(DispatchError::UnsupportedVariant {
            family,
            variant: "mq4g128v2_specialized_route_only",
            arch: "",
            quant: "MQ4G128V2",
        });
    }
    if dtype.is_mq_i4() {
        return Err(reject_mq_i4(dtype, family));
    }
    Ok(())
}

/// qt=55 FP8E4M3B128 has no generic kernel: only the typed gfx942 FP8 route
/// may read it. Guard for per-dtype generic MoE entrypoints.
fn reject_fp8_b128(dtype: DType) -> Result<(), DispatchError> {
    if dtype == DType::FP8E4M3B128 {
        return Err(crate::families::moe::fp8_b128_generic_refusal());
    }
    Ok(())
}

/// Refuse any layer carrying a qt=55 operand (router, shared, routed, or a
/// per-expert tier). Every generic route — decode gpu-top-k/CPU fallback and
/// prefill Path 0/1/2 — MUST pass through this (or the equivalent
/// `check_generic_route` on a resolution) before launching.
pub(crate) fn reject_fp8_moe(
    dtypes: &crate::families::moe::MoeDtypes<'_>,
) -> Result<(), DispatchError> {
    if dtypes.has_fp8_b128_projection() {
        return Err(crate::families::moe::fp8_b128_generic_refusal());
    }
    Ok(())
}

/// Guard for stages that only touch the router / shared projections (and so
/// are legitimately shared with the typed FP8 route): under the FP8 route the
/// router and shared operands must be non-qt55; for every other route any
/// qt=55 operand is refused.
fn reject_fp8_moe_unless_routed(
    dtypes: &crate::families::moe::MoeDtypes<'_>,
    route: Option<MoeRouteCapability>,
) -> Result<(), DispatchError> {
    if !route.is_some_and(MoeRouteCapability::is_fp8_grouped) {
        return reject_fp8_moe(dtypes);
    }
    let shared_fp8 = dtypes.shared.is_some_and(|shared| {
        [shared.selector, shared.gate, shared.up, shared.down].contains(&DType::FP8E4M3B128)
    });
    if shared_fp8 || dtypes.router == DType::FP8E4M3B128 {
        return Err(crate::families::moe::fp8_b128_generic_refusal());
    }
    Ok(())
}

fn reject_mq4g128v2_moe(dtypes: &crate::families::moe::MoeDtypes<'_>) -> Result<(), DispatchError> {
    reject_fp8_moe(dtypes)?;
    // qwen35 qt53 routed-down typed fallback: the one narrow exception to this
    // generic refusal (uniform qt44 gate/up + uniform qt53 routed down, qt53
    // nowhere else). Reached only for MoE without a route policy: with one, a
    // missing route is refused before this guard runs. Tags / AWQ / Paro /
    // CPU-deferred operands the snapshot cannot see are refused by
    // [`qt53_fallback_operand_refusal`] in every stage that launches.
    if dtypes.qt53_routed_down_fallback() {
        return Ok(());
    }
    let tagged = dtypes
        .per_expert_gate_up
        .is_some_and(|values| values.contains(&DType::MQ4G128V2))
        || dtypes
            .per_expert_down
            .is_some_and(|values| values.contains(&DType::MQ4G128V2));
    let shared_tagged = dtypes.shared.is_some_and(|shared| {
        [shared.selector, shared.gate, shared.up, shared.down].contains(&DType::MQ4G128V2)
    });
    if tagged
        || shared_tagged
        || [dtypes.router, dtypes.routed_gate_up, dtypes.routed_down].contains(&DType::MQ4G128V2)
    {
        return reject_mq4g128v2(DType::MQ4G128V2, "moe");
    }
    let first_mq_i4 = dtypes
        .per_expert_gate_up
        .into_iter()
        .chain(dtypes.per_expert_down)
        .flatten()
        .copied()
        .chain(
            dtypes
                .shared
                .into_iter()
                .flat_map(|shared| [shared.selector, shared.gate, shared.up, shared.down]),
        )
        .chain([dtypes.router, dtypes.routed_gate_up, dtypes.routed_down])
        .find(|dtype| dtype.is_mq_i4());
    if let Some(dtype) = first_mq_i4 {
        return reject_mq4g128v2(dtype, "moe");
    }
    Ok(())
}

/// Basis of the activation a routed expert down projection reads.
///
/// The routed activation (`silu(gate) * up`) must be rotated into the basis the
/// down weights were packed in; a mismatch is silent fluent corruption, so every
/// caller (decode, prefill, MTP head) asks this one helper instead of matching
/// dtypes itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutedDownRotation {
    /// FWHT over 256-element groups (every MQ4/MQ6 G256 expert down and the
    /// codebook / E8 families riding the same fused silu+rotate).
    Fwht256,
    /// FWHT over 128-element groups (qt53 `MQ4G128V2` expert down): F32
    /// `silu_mul_f32` then `rotate_x_mq_128_v2` over the live rows; never the
    /// fused FWHT256 silu+rotate.
    Fwht128,
    /// Natural basis, no rotation (`MQ2G256LloydU`).
    Natural,
    /// Givens (PARO sidecar) rotation, not an FWHT (`ParoQ4G128`).
    Givens,
}

/// The [`RoutedDownRotation`] of a routed expert down dtype.
pub fn routed_down_rotation(dtype: DType) -> RoutedDownRotation {
    match dtype {
        DType::MQ4G128V2 => RoutedDownRotation::Fwht128,
        DType::ParoQ4G128 => RoutedDownRotation::Givens,
        dtype if gate_down_skips_rotation(dtype) => RoutedDownRotation::Natural,
        _ => RoutedDownRotation::Fwht256,
    }
}

/// Operands the dtype snapshot cannot see, for the qwen35 qt53 routed-down
/// typed fallback. Pure: `Ok` unless the layer is on the fallback (no route,
/// routed down qt53) and carries something the F32 FWHT128 recipe cannot serve.
///
/// * qt53 outside the exact fallback dtype shape (defence behind the generic
///   guard) is refused;
/// * per-expert dtype tags (no tag exists for the qt44/qt53 pair), per-expert
///   AWQ scales and Paro sidecars are refused;
/// * CPU-deferred routed experts are refused (the CPU splice has no qt53 F32
///   FWHT128 recipe for the non-route F32 path).
///
/// A layer on a specialized route (`route.is_some()`, Flash-Next) is never
/// consulted.
pub(crate) fn qt53_fallback_operand_refusal(
    dtypes: &crate::families::moe::MoeDtypes<'_>,
    route: Option<MoeRouteCapability>,
    has_dtype_tags: bool,
    has_expert_awq: bool,
    has_paro_sidecar: bool,
    cpu_deferred_experts: bool,
) -> Result<(), DispatchError> {
    use crate::families::moe::qt53_fallback_unsupported as unsupported;
    if route.is_some() || dtypes.routed_down != DType::MQ4G128V2 {
        return Ok(());
    }
    if !dtypes.qt53_routed_down_fallback() {
        return Err(unsupported("qt53-fallback-operands-unsupported"));
    }
    if has_dtype_tags {
        return Err(unsupported("qt53-fallback-dtype-tags-unsupported"));
    }
    if has_expert_awq {
        return Err(unsupported("qt53-fallback-awq-unsupported"));
    }
    if has_paro_sidecar {
        return Err(unsupported("qt53-fallback-paro-unsupported"));
    }
    if cpu_deferred_experts {
        return Err(unsupported("qt53-fallback-cpu-deferred-unsupported"));
    }
    Ok(())
}

/// Decode-stage operand guard (see [`qt53_fallback_operand_refusal`]).
fn check_qt53_decode_fallback(
    p: &crate::families::moe::MoeParams<'_>,
    route: Option<MoeRouteCapability>,
) -> Result<(), DispatchError> {
    // Every other layer (all other dtypes, every specialized route) pays nothing.
    if route.is_some() || p.dtypes.routed_down != DType::MQ4G128V2 {
        return Ok(());
    }
    let expert_awq = p.expert_down_awq_ptrs.is_some()
        || p.routed_experts
            .get(0)
            .is_some_and(|(gate_up, down)| gate_up.awq_scale.is_some() || down.awq_scale.is_some());
    qt53_fallback_operand_refusal(
        &p.dtypes,
        route,
        p.expert_dtype_tags.is_some(),
        expert_awq,
        p.routed_gate_up_paro.is_some() || p.routed_down_paro.is_some(),
        p.cpu_deferred_experts,
    )
}

/// Generic-prefill-stage operand guard (the specialized route lowers through
/// `qt44_qt53_prefill`, never these stages, so no route is ever present here).
fn check_qt53_prefill_fallback(
    p: &crate::families::moe::MoePrefillParams<'_>,
) -> Result<(), DispatchError> {
    qt53_fallback_operand_refusal(
        &p.dtypes,
        None,
        p.expert_dtype_tags.is_some(),
        p.expert_down_awq_ptrs.is_some() || p.down_awq_scale.is_some(),
        p.paro_gate_up.is_some() || p.paro_down.is_some(),
        p.cpu_deferred_experts,
    )
}
/// Select only a route whose architecture-declared kernel contract matches
/// the bound geometry, formats, and sidecars. A mismatched QT53 route is
/// rejected by the sealer rather than delegated to generic kernels.
#[allow(clippy::too_many_arguments)]
fn select_grouped_route(
    dtypes: &crate::families::moe::MoeDtypes<'_>,
    n_exp: usize,
    k_top: usize,
    hidden: usize,
    intermediate: usize,
    shared_intermediate: usize,
    gate_up_k: usize,
    down_m: usize,
    down_k: usize,
    dtype_tags: Option<&GpuTensor>,
    has_sidecars: bool,
    route_policy: Option<crate::families::moe::MoeRoutePolicy>,
    arch: &str,
) -> Option<crate::families::moe::MoeRouteCapability> {
    select_grouped_route_by(
        dtypes,
        n_exp,
        k_top,
        hidden,
        intermediate,
        shared_intermediate,
        gate_up_k,
        down_m,
        down_k,
        dtype_tags,
        has_sidecars,
        route_policy,
        &|codec| rdna_compute::qwen4_slab_codec_admitted(arch, codec),
    )
}

/// [`select_grouped_route`] with the sealed slab route's per-codec admission given as a
/// predicate (the architecture's `rdna_compute::slab_admission::qwen4_slab_codec_admitted_with`),
/// so the pure decision is testable for both states of the gfx1151 E8P receipts switch.
#[allow(clippy::too_many_arguments)]
fn select_grouped_route_by(
    dtypes: &crate::families::moe::MoeDtypes<'_>,
    n_exp: usize,
    k_top: usize,
    hidden: usize,
    intermediate: usize,
    shared_intermediate: usize,
    gate_up_k: usize,
    down_m: usize,
    down_k: usize,
    dtype_tags: Option<&GpuTensor>,
    has_sidecars: bool,
    route_policy: Option<crate::families::moe::MoeRoutePolicy>,
    admits: &dyn Fn(u8) -> bool,
) -> Option<crate::families::moe::MoeRouteCapability> {
    let policy = route_policy?;
    let shared = dtypes.shared?;
    let formats = policy.formats;
    let uniform = |table: Option<&[DType]>, expected: DType| {
        table.map_or(true, |values| values.iter().all(|dtype| *dtype == expected))
    };
    match policy.capability {
        MoeRouteCapability::Qt44Qt53Grouped | MoeRouteCapability::Qt44Qt53GroupedSymmetric
            if policy.capability.admitted_on()
                && crate::families::moe::grouped_route_geometry_supported(
                    &policy,
                    n_exp,
                    k_top,
                    hidden,
                    intermediate,
                    gate_up_k,
                    down_m,
                    down_k,
                    dtypes.routed_gate_up,
                    dtypes.routed_down,
                )
                && shared_intermediate == policy.geometry.shared_intermediate
                && !dtypes.routed_has_mixed_experts
                && !dtypes.has_paro_shared
                && !has_sidecars
                && dtype_tags.is_none()
                && formats.router == DType::BF16
                && formats.shared_selector == DType::BF16
                // The shared gate/up read the natural activation as one
                // format; down may also be the row-local QT53.
                && matches!(formats.shared_gate, DType::BF16 | DType::Q8_0)
                && formats.shared_up == formats.shared_gate
                && matches!(formats.shared_down, DType::BF16 | DType::Q8_0 | DType::MQ4G128V2)
                && [
                    dtypes.router,
                    shared.selector,
                    shared.gate,
                    shared.up,
                    shared.down,
                ] == [
                    formats.router,
                    formats.shared_selector,
                    formats.shared_gate,
                    formats.shared_up,
                    formats.shared_down,
                ]
                && uniform(dtypes.per_expert_gate_up, formats.routed_gate_up)
                && uniform(dtypes.per_expert_down, formats.routed_down) =>
        {
            Some(policy.capability)
        }
        // The sealed expert-slab route: per-expert codecs, the codec table
        // standing in for the legacy dtype-tag table.
        MoeRouteCapability::SlabCodecGrouped
            if policy.capability.admitted_on()
                && crate::families::moe::slab_route_geometry_supported(
                    Some(admits),
                    &policy,
                    n_exp,
                    k_top,
                    hidden,
                    intermediate,
                    gate_up_k,
                    down_m,
                    down_k,
                    dtypes,
                    dtype_tags.is_some(),
                )
                && shared_intermediate == policy.geometry.shared_intermediate
                && !dtypes.has_paro_shared
                && !has_sidecars
                && formats.router == DType::BF16
                && formats.shared_selector == DType::BF16
                && matches!(formats.shared_gate, DType::BF16 | DType::Q8_0)
                && formats.shared_up == formats.shared_gate
                && matches!(
                    formats.shared_down,
                    DType::BF16 | DType::Q8_0 | DType::MQ4G128V2
                )
                && [
                    dtypes.router,
                    shared.selector,
                    shared.gate,
                    shared.up,
                    shared.down,
                ] == [
                    formats.router,
                    formats.shared_selector,
                    formats.shared_gate,
                    formats.shared_up,
                    formats.shared_down,
                ] =>
        {
            Some(policy.capability)
        }
        _ => None,
    }
}

/// Pure selection of the gfx942 qt=55 FP8E4M3B128 grouped route; runs before
/// [`select_grouped_route`] / generic selection.
///
/// * `Ok(None)` — the policy does not declare the FP8 route and no operand is
///   qt=55: continue with the existing selection.
/// * `Ok(Some(Fp8B128Grouped))` — gfx942, the policy declares the FP8 route,
///   and the live operands match the canonical contract exactly (512/10/2560/
///   640/640, routed gate_up + down qt=55, BF16 router/shared, uniform
///   experts, no tags or sidecars).
/// * `Err` — typed refusal: a qt=55 operand without an FP8 declaration (absent
///   policy included), or an FP8 declaration the device or operands do not
///   satisfy. qt=55 never falls through to generic selection.
#[allow(clippy::too_many_arguments)]
pub(crate) fn select_fp8_grouped_route(
    ctx: &DispatchCtx,
    dtypes: &crate::families::moe::MoeDtypes<'_>,
    route_policy: Option<crate::families::moe::MoeRoutePolicy>,
    n_exp: usize,
    k_top: usize,
    hidden: usize,
    intermediate: usize,
    shared_intermediate: usize,
    gate_up_k: usize,
    down_m: usize,
    down_k: usize,
    dtype_tags: Option<&GpuTensor>,
    has_sidecars: bool,
) -> Result<Option<MoeRouteCapability>, DispatchError> {
    let Some(policy) = route_policy.filter(|policy| policy.capability.is_fp8_grouped()) else {
        return reject_fp8_moe(dtypes).map(|()| None);
    };
    let shared = dtypes.shared.ok_or_else(fp8_route_unsupported)?;
    let uniform_fp8 = |table: Option<&[DType]>| {
        table.map_or(true, |values| {
            values.iter().all(|dtype| *dtype == DType::FP8E4M3B128)
        })
    };
    if policy.capability.admitted_on_arch(&ctx.arch)
        && crate::families::moe::fp8_grouped_route_geometry_supported(
            &policy,
            n_exp,
            k_top,
            hidden,
            intermediate,
            gate_up_k,
            down_m,
            down_k,
            dtypes.routed_gate_up,
            dtypes.routed_down,
        )
        && shared_intermediate == policy.geometry.shared_intermediate
        && !dtypes.routed_has_mixed_experts
        && !dtypes.has_paro_shared
        && !has_sidecars
        && dtype_tags.is_none()
        && [
            dtypes.router,
            shared.selector,
            shared.gate,
            shared.up,
            shared.down,
        ]
        .iter()
        .all(|dtype| *dtype == DType::BF16)
        && uniform_fp8(dtypes.per_expert_gate_up)
        && uniform_fp8(dtypes.per_expert_down)
    {
        Ok(Some(policy.capability))
    } else {
        Err(fp8_route_unsupported())
    }
}

fn fp8_route_unsupported() -> DispatchError {
    DispatchError::UnsupportedVariant {
        family: "moe",
        variant: "fp8b128-route-unsupported",
        arch: "",
        quant: "FP8E4M3B128",
    }
}

#[cfg(test)]
mod qt53_fallback_tests {
    use super::*;
    use crate::families::moe::{
        MoeDtypes, MoeResolution, MoeRouteCapability, MoeRouteFormats, MoeRouteGeometry,
        MoeRoutePolicy, MoeSharedDtypes, MOE_GROUPED_BLOCK_M,
    };

    /// A3B-style dtype snapshot: BF16 router/shared projections, qt44 routed
    /// gate/up, qt53 routed down, no route policy.
    fn fallback_dtypes() -> MoeDtypes<'static> {
        MoeDtypes {
            router: DType::BF16,
            shared: Some(MoeSharedDtypes {
                selector: DType::BF16,
                gate: DType::BF16,
                up: DType::BF16,
                down: DType::BF16,
            }),
            experts_all_gate_up_mq4: false,
            routed_gate_up: DType::MQ4G256V2,
            routed_down: DType::MQ4G128V2,
            routed_has_mixed_experts: false,
            has_paro_shared: false,
            per_expert_gate_up: None,
            per_expert_down: None,
        }
    }

    fn flash_next_policy(shared_down: DType) -> MoeRoutePolicy {
        MoeRoutePolicy {
            expert_keep_mask: 0,
            capability: MoeRouteCapability::Qt44Qt53Grouped,
            geometry: MoeRouteGeometry {
                experts: 512,
                top_k: 10,
                hidden: 2560,
                routed_intermediate: 640,
                shared_intermediate: 640,
            },
            formats: MoeRouteFormats {
                router: DType::BF16,
                shared_selector: DType::BF16,
                shared_gate: DType::BF16,
                shared_up: DType::BF16,
                shared_down,
                routed_gate_up: DType::MQ4G256V2,
                routed_down: DType::MQ4G128V2,
            },
            slab_cold_codec: None,
        }
    }

    fn select(
        dtypes: &MoeDtypes<'_>,
        policy: Option<MoeRoutePolicy>,
        shape: [usize; 7],
    ) -> Option<MoeRouteCapability> {
        let [n_exp, k_top, hidden, mi, gate_up_k, down_m, down_k] = shape;
        select_grouped_route(
            dtypes, n_exp, k_top, hidden, mi, 640, gate_up_k, down_m, down_k, None, false, policy,
            "gfx1201",
        )
    }

    fn is_qt53_refusal(result: Result<(), DispatchError>) -> bool {
        matches!(
            result,
            Err(DispatchError::UnsupportedVariant {
                family: "moe",
                quant: "MQ4G128V2",
                ..
            })
        )
    }

    #[test]
    fn routed_down_rotation_basis_table() {
        use RoutedDownRotation::*;
        for (dtype, want) in [
            (DType::MQ4G128V2, Fwht128),
            (DType::MQ4G256, Fwht256),
            (DType::MQ4G256V2, Fwht256),
            (DType::MQ5G256, Fwht256),
            (DType::MQ6G256, Fwht256),
            (DType::MQ6G256V2, Fwht256),
            (DType::MQ2G256Lloyd, Fwht256),
            (DType::MFP4G32E8, Fwht256),
            (DType::MQ2G256LloydU, Natural),
            (DType::ParoQ4G128, Givens),
        ] {
            assert_eq!(routed_down_rotation(dtype), want, "{dtype:?}");
        }
    }

    /// Kind predicates accept qt53 on the down projections only. Gate/up kinds,
    /// the shared dense down and the plain grouped kind keep refusing it.
    #[test]
    fn qt53_kind_predicates_accept_down_only() {
        assert_eq!(
            decode_expanded_down_kind(DType::MQ4G128V2),
            Some("qt53_f32")
        );
        assert_eq!(prefill_path1_down_kind(DType::MQ4G128V2), Some("qt53_f32"));
        assert_eq!(
            prefill_path1_down_kind_tag_aware(DType::MQ4G128V2, false),
            Some("qt53_f32")
        );
        assert_eq!(grouped_gemm_down_kind(DType::MQ4G128V2), Some("qt53_f32"));
        // Never a gate_up kind, the shared dense down, or the plain grouped kind.
        assert_eq!(prefill_path1_gate_up_kind(DType::MQ4G128V2), None);
        assert_eq!(grouped_gemm_kind(DType::MQ4G128V2), None);
        assert_eq!(shared_dense_down_kind(DType::MQ4G128V2), None);
        // Existing down kinds are unchanged.
        for dtype in [DType::MQ4G256, DType::MQ4G256V2, DType::MQ6G256V2] {
            assert_eq!(grouped_gemm_down_kind(dtype), grouped_gemm_kind(dtype));
        }
        assert_eq!(decode_expanded_down_kind(DType::MQ4G256V2), Some("mq4v2"));
    }

    #[test]
    fn grouped_gemm_role_admits_qt53_down_only() {
        use GroupedGemmRole::{Down, GateUp};
        assert!(grouped_gemm_qt53_down_admitted(Down, false, 1, 256).is_ok());
        // GU qt53, tagged layers, a non-unit x_row_div and a missing n_exp are
        // refused before any launch.
        assert!(is_qt53_refusal(grouped_gemm_qt53_down_admitted(
            GateUp, false, 1, 256
        )));
        assert!(is_qt53_refusal(grouped_gemm_qt53_down_admitted(
            GateUp, false, 8, 256
        )));
        assert!(is_qt53_refusal(grouped_gemm_qt53_down_admitted(
            Down, true, 1, 256
        )));
        assert!(is_qt53_refusal(grouped_gemm_qt53_down_admitted(
            Down, false, 8, 256
        )));
        assert!(is_qt53_refusal(grouped_gemm_qt53_down_admitted(
            Down, false, 1, 0
        )));
    }

    /// Grouped k8 scatter ABI the qt53 `Multirow` kernel consumes:
    /// `moe_scatter_fused_k8.hip` pads each expert to `block_m` rows with -1 flat
    /// slots and writes one expert id per `block_m` tile, and
    /// `gemm_mq4g128v2_moe_grouped_top10_multirow.hip` reads 16-slot tiles
    /// (`slot_start = tile * 16`, `sorted_slot_index[..] < 0` skipped).
    #[test]
    fn grouped_scatter_block_m_matches_qt53_multirow_tile() {
        assert_eq!(MOE_GROUPED_BLOCK_M, 16);
    }

    /// The narrow exception to the generic guard is exactly the fallback shape.
    #[test]
    fn generic_guard_admits_only_the_uniform_qt44_qt53_down_shape() {
        assert!(reject_mq4g128v2_moe(&fallback_dtypes()).is_ok());
        let base = fallback_dtypes();
        let shared = |edit: fn(&mut MoeSharedDtypes)| {
            let mut d = fallback_dtypes();
            let mut s = d.shared.unwrap();
            edit(&mut s);
            d.shared = Some(s);
            d
        };
        static MIXED_DOWN: [DType; 2] = [DType::MQ4G128V2, DType::MQ4G256V2];
        let refused = [
            // qt53 in any non-routed-down role, or a non-qt44 gate/up partner.
            MoeDtypes {
                router: DType::MQ4G128V2,
                ..base
            },
            shared(|s| s.selector = DType::MQ4G128V2),
            shared(|s| s.gate = DType::MQ4G128V2),
            shared(|s| s.up = DType::MQ4G128V2),
            shared(|s| s.down = DType::MQ4G128V2),
            MoeDtypes {
                routed_gate_up: DType::MQ4G128V2,
                routed_down: DType::MQ4G256V2,
                ..base
            },
            MoeDtypes {
                routed_gate_up: DType::MQ4G128V2,
                ..base
            },
            MoeDtypes {
                routed_gate_up: DType::MQ4G256,
                ..base
            },
            // mixed / tagged tiers, Paro.
            MoeDtypes {
                routed_has_mixed_experts: true,
                ..base
            },
            MoeDtypes {
                per_expert_down: Some(&MIXED_DOWN),
                ..base
            },
            MoeDtypes {
                has_paro_shared: true,
                ..base
            },
        ];
        for (index, dtypes) in refused.iter().enumerate() {
            assert!(
                is_qt53_refusal(reject_mq4g128v2_moe(dtypes)),
                "case {index}"
            );
        }
    }

    /// Operands the dtype snapshot cannot see: tags, AWQ, Paro, CPU-deferred.
    #[test]
    fn qt53_operand_guard_table() {
        let dtypes = fallback_dtypes();
        let guard = |tags, awq, paro, deferred| {
            qt53_fallback_operand_refusal(&dtypes, None, tags, awq, paro, deferred)
        };
        assert!(guard(false, false, false, false).is_ok());
        for (tags, awq, paro, deferred) in [
            (true, false, false, false),
            (false, true, false, false),
            (false, false, true, false),
            (false, false, false, true),
        ] {
            assert!(
                is_qt53_refusal(guard(tags, awq, paro, deferred)),
                "{tags} {awq} {paro} {deferred}"
            );
        }
        // qt53 outside the fallback shape is refused even without sidecars.
        let crossed = MoeDtypes {
            routed_gate_up: DType::MQ4G256,
            ..dtypes
        };
        assert!(is_qt53_refusal(qt53_fallback_operand_refusal(
            &crossed, None, false, false, false, false
        )));
        // A specialized route (Flash-Next) and every other down dtype are never consulted.
        let route = Some(MoeRouteCapability::Qt44Qt53Grouped);
        assert!(qt53_fallback_operand_refusal(&dtypes, route, true, true, true, true).is_ok());
        let other = MoeDtypes {
            routed_down: DType::MQ4G256V2,
            ..dtypes
        };
        assert!(qt53_fallback_operand_refusal(&other, None, true, true, true, true).is_ok());
    }

    /// Flash-Next (`route_policy = Some(Qt44Qt53Grouped*)`) selection and its
    /// qt53 GU / shared / router refusals are unchanged; a MoE without a route
    /// policy never selects the typed route (it lowers through the fallback).
    #[test]
    fn flash_next_route_policy_selection_is_unchanged() {
        let fn_shape = [512, 10, 2560, 640, 2560, 2560, 640];
        let deltanet = cfg!(feature = "deltanet");
        let canonical = fallback_dtypes();
        assert_eq!(
            select(&canonical, Some(flash_next_policy(DType::BF16)), fn_shape),
            deltanet.then_some(MoeRouteCapability::Qt44Qt53Grouped)
        );
        // Shared down qt53 stays admitted ONLY by the typed route (declared) and
        // refused by the generic guard (shape predicate is false for it).
        let mut shared_qt53 = fallback_dtypes();
        shared_qt53.shared.as_mut().unwrap().down = DType::MQ4G128V2;
        assert_eq!(
            select(
                &shared_qt53,
                Some(flash_next_policy(DType::MQ4G128V2)),
                fn_shape
            ),
            deltanet.then_some(MoeRouteCapability::Qt44Qt53Grouped)
        );
        assert!(is_qt53_refusal(reject_mq4g128v2_moe(&shared_qt53)));
        // GU / router / shared selector / gate / up qt53 never enter the route.
        for edit in [
            (|d: &mut MoeDtypes<'static>| d.router = DType::MQ4G128V2)
                as fn(&mut MoeDtypes<'static>),
            |d| d.shared.as_mut().unwrap().selector = DType::MQ4G128V2,
            |d| d.shared.as_mut().unwrap().gate = DType::MQ4G128V2,
            |d| d.shared.as_mut().unwrap().up = DType::MQ4G128V2,
            |d| d.routed_gate_up = DType::MQ4G128V2,
        ] {
            let mut dtypes = fallback_dtypes();
            edit(&mut dtypes);
            assert_eq!(
                select(&dtypes, Some(flash_next_policy(DType::BF16)), fn_shape),
                None
            );
            assert!(is_qt53_refusal(reject_mq4g128v2_moe(&dtypes)));
        }
        // No route policy: never the typed route, whatever the shape (A3B included).
        assert_eq!(select(&canonical, None, fn_shape), None);
        assert_eq!(
            select(&canonical, None, [256, 8, 2048, 512, 2048, 2048, 512]),
            None
        );
        // A declared Flash-Next policy never admits the A3B shape (256/8/2048/512):
        // the rdna-compute `m == 2560 && k == 640` / WMMA specializations stay
        // Flash-Next-only and A3B 2048/512 takes the Multirow kernel.
        assert_eq!(
            select(
                &canonical,
                Some(flash_next_policy(DType::BF16)),
                [256, 8, 2048, 512, 2048, 2048, 512]
            ),
            None
        );
        // Flash-Next decode resolution at top-10 is not a GPU top-8 route; the
        // typed route sets it itself and the qt53 refusal state is exempt for it.
        let decode = MoeResolution::resolve_arch(&canonical, 10, false);
        assert!(!decode.use_gpu_topk);
        assert!(decode
            .check_generic_route(Some(MoeRouteCapability::Qt44Qt53Grouped))
            .is_ok());
    }

    /// Inventory of every qt53 routed-down mode and its pure outcome: a typed
    /// route / kernel kind, or a typed refusal naming the variant.
    #[test]
    fn qt53_fallback_mode_inventory() {
        let dtypes = fallback_dtypes();
        let ctx = |arch: &str| crate::context::DispatchCtx::for_test(arch);
        let variant = |result: Result<(), DispatchError>| match result {
            Ok(()) => "ok",
            Err(DispatchError::UnsupportedVariant {
                variant,
                quant: "MQ4G128V2",
                ..
            }) => variant,
            Err(other) => panic!("unexpected error {other:?}"),
        };
        // decode: GPU top-K indexed at k == 8 only.
        assert_eq!(
            variant(MoeResolution::resolve_arch(&dtypes, 8, false).check_generic_route(None)),
            "ok"
        );
        assert_eq!(
            variant(MoeResolution::resolve_arch(&dtypes, 4, false).check_generic_route(None)),
            "qt53-routed-down-fallback-needs-gpu-topk-k8"
        );
        // prefill: Path 0 refused, Path 1 (no WMMA) and Path 2 (WMMA) routed.
        for (arch, want) in [
            ("gfx942", "qt53-routed-down-fallback-path0-unsupported"),
            ("gfx906", "qt53-routed-down-fallback-path0-unsupported"),
            ("gfx1030", "ok"),
            ("gfx1100", "ok"),
            ("gfx1151", "ok"),
            ("gfx1201", "ok"),
        ] {
            let c = ctx(arch);
            let res =
                crate::families::moe::MoePrefillResolution::resolve(&dtypes, &c.arch, &c.flags);
            assert_eq!(variant(res.check_generic_route(None)), want, "{arch}");
        }
        // operands.
        for (tags, awq, paro, deferred, want) in [
            (
                true,
                false,
                false,
                false,
                "qt53-fallback-dtype-tags-unsupported",
            ),
            (false, true, false, false, "qt53-fallback-awq-unsupported"),
            (false, false, true, false, "qt53-fallback-paro-unsupported"),
            (
                false,
                false,
                false,
                true,
                "qt53-fallback-cpu-deferred-unsupported",
            ),
        ] {
            assert_eq!(
                variant(qt53_fallback_operand_refusal(
                    &dtypes, None, tags, awq, paro, deferred
                )),
                want
            );
        }
        // grouped GEMM roles.
        assert_eq!(
            variant(grouped_gemm_qt53_down_admitted(
                GroupedGemmRole::Down,
                false,
                1,
                256
            )),
            "ok"
        );
        assert_eq!(
            variant(grouped_gemm_qt53_down_admitted(
                GroupedGemmRole::GateUp,
                false,
                1,
                256
            )),
            "qt53-grouped-down-only"
        );
        // generic per-expert GEMV/GEMM entry points keep refusing qt53.
        assert_eq!(
            variant(reject_mq4g128v2(DType::MQ4G128V2, "moe")),
            "mq4g128v2_specialized_route_only"
        );
        // The CPU top-K / per-expert fallback has no qt53 GEMV (refused above for
        // k != 8); the only qt53 decode kind is the indexed F32 expanded down.
        assert_eq!(
            decode_expanded_down_kind(DType::MQ4G128V2),
            Some("qt53_f32")
        );
    }
}
#[cfg(test)]
mod qwen4_route_tests {
    use super::*;
    use crate::families::moe::{
        MoeDtypes, MoeRouteCapability, MoeRouteFormats, MoeRouteGeometry, MoeRoutePolicy,
        MoeSharedDtypes,
    };
    fn canonical_dtypes() -> MoeDtypes<'static> {
        MoeDtypes {
            router: DType::BF16,
            shared: Some(MoeSharedDtypes {
                selector: DType::BF16,
                gate: DType::BF16,
                up: DType::BF16,
                down: DType::MQ4G128V2,
            }),
            experts_all_gate_up_mq4: true,
            routed_gate_up: DType::MQ4G256V2,
            routed_down: DType::MQ4G128V2,
            routed_has_mixed_experts: false,
            has_paro_shared: false,
            per_expert_gate_up: None,
            per_expert_down: None,
        }
    }

    fn policy(shared_down: DType) -> MoeRoutePolicy {
        MoeRoutePolicy {
            expert_keep_mask: 0,
            capability: MoeRouteCapability::Qt44Qt53Grouped,
            geometry: MoeRouteGeometry {
                experts: 512,
                top_k: 10,
                hidden: 2560,
                routed_intermediate: 640,
                shared_intermediate: 640,
            },
            formats: MoeRouteFormats {
                router: DType::BF16,
                shared_selector: DType::BF16,
                shared_gate: DType::BF16,
                shared_up: DType::BF16,
                shared_down,
                routed_gate_up: DType::MQ4G256V2,
                routed_down: DType::MQ4G128V2,
            },
            slab_cold_codec: None,
        }
    }

    fn route_for_shape(
        dtypes: &MoeDtypes<'_>,
        n_exp: usize,
        k_top: usize,
        hidden: usize,
        intermediate: usize,
        gate_up_k: usize,
        down_m: usize,
        down_k: usize,
    ) -> bool {
        select_grouped_route(
            dtypes,
            n_exp,
            k_top,
            hidden,
            intermediate,
            640,
            gate_up_k,
            down_m,
            down_k,
            None,
            false,
            Some(policy(
                dtypes.shared.map_or(DType::MQ4G128V2, |shared| shared.down),
            )),
            "gfx1201",
        )
        .is_some()
    }

    fn is_canonical_qwen4(dtypes: &MoeDtypes<'_>) -> bool {
        route_for_shape(dtypes, 512, 10, 2560, 640, 2560, 2560, 640)
    }

    #[test]
    fn qwen4_route_rejects_noncanonical_geometry_and_wire_dtype() {
        let dtypes = canonical_dtypes();
        assert!(!route_for_shape(
            &dtypes, 513, 10, 2560, 640, 2560, 2560, 640
        ));
        assert!(!route_for_shape(
            &dtypes, 512, 7, 2560, 640, 2560, 2560, 640
        ));
        assert!(!route_for_shape(&dtypes, 512, 10, 2560, 640, 2560, 1, 640));

        let mut dtypes = canonical_dtypes();
        dtypes.routed_gate_up = DType::BF16;
        assert!(!is_canonical_qwen4(&dtypes));

        let mut dtypes = canonical_dtypes();
        dtypes.router = DType::MQ4G256V2;
        assert!(!is_canonical_qwen4(&dtypes));

        let mut dtypes = canonical_dtypes();
        dtypes.shared.as_mut().unwrap().gate = DType::MQ4G256V2;
        assert!(!is_canonical_qwen4(&dtypes));
    }

    #[test]
    fn qwen4_route_rejects_q53_on_generic_moe() {
        let dtypes = canonical_dtypes();
        // The generic MoE gate remains a hard rejection; only the canonical
        // typed route may consume this shared-down q53 operand.
        assert!(matches!(
            reject_mq4g128v2_moe(&dtypes),
            Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: _,
                quant: "MQ4G128V2",
                ..
            })
        ));
    }

    #[test]
    fn qwen4_route_accepts_canonical_shared_down_bf16() {
        let mut dtypes = canonical_dtypes();
        dtypes.shared.as_mut().unwrap().down = DType::BF16;
        assert_eq!(is_canonical_qwen4(&dtypes), cfg!(feature = "deltanet"));
    }

    #[test]
    fn qwen4_route_admits_q8_shared_expert_only_as_a_matched_pair() {
        let route = |gate: DType, up: DType, down: DType| {
            let mut dtypes = canonical_dtypes();
            let shared = dtypes.shared.as_mut().unwrap();
            (shared.gate, shared.up, shared.down) = (gate, up, down);
            let mut declared = policy(down);
            (declared.formats.shared_gate, declared.formats.shared_up) = (gate, up);
            select_grouped_route(
                &dtypes,
                512,
                10,
                2560,
                640,
                640,
                2560,
                2560,
                640,
                None,
                false,
                Some(declared),
                "gfx1201",
            )
            .is_some()
        };
        let deltanet = cfg!(feature = "deltanet");
        assert_eq!(route(DType::Q8_0, DType::Q8_0, DType::Q8_0), deltanet);
        assert_eq!(route(DType::Q8_0, DType::Q8_0, DType::BF16), deltanet);
        // Gate and up share one activation read: a mixed pair is refused, as
        // are formats the route has no projection for.
        assert!(!route(DType::Q8_0, DType::BF16, DType::Q8_0));
        assert!(!route(DType::MQ6G256V2, DType::MQ6G256V2, DType::Q8_0));
        assert!(!route(DType::Q8_0, DType::Q8_0, DType::MQ6G256V2));
    }

    #[test]
    fn route_accepts_canonical_contract_and_rejects_sidecars_or_shape() {
        let dtypes = canonical_dtypes();
        // Admission is device-independent: gfx1100, gfx1201, and gfx1151
        // select the same route when the feature and operands match.
        assert_eq!(
            select_grouped_route(
                &dtypes,
                512,
                10,
                2560,
                640,
                640,
                2560,
                2560,
                640,
                None,
                false,
                Some(policy(DType::MQ4G128V2)),
                "gfx1201",
            ),
            cfg!(feature = "deltanet").then_some(MoeRouteCapability::Qt44Qt53Grouped),
        );
        assert_eq!(
            select_grouped_route(
                &dtypes,
                512,
                10,
                2560,
                640,
                640,
                2560,
                2560,
                640,
                None,
                true,
                Some(policy(DType::MQ4G128V2)),
                "gfx1201",
            ),
            None,
        );
        assert_eq!(
            select_grouped_route(
                &dtypes,
                512,
                10,
                2560,
                640,
                1024,
                2560,
                2560,
                640,
                None,
                false,
                Some(policy(DType::MQ4G128V2)),
                "gfx1201",
            ),
            None,
        );
    }

    #[test]
    fn route_rejects_false_or_unsupported_declaration() {
        let dtypes = canonical_dtypes();
        let accepts = |declared: MoeRoutePolicy, experts| {
            select_grouped_route(
                &dtypes,
                experts,
                10,
                2560,
                640,
                640,
                2560,
                2560,
                640,
                None,
                false,
                Some(declared),
                "gfx1201",
            )
            .is_some()
        };
        let declared = policy(DType::MQ4G128V2);

        let mut unsupported_geometry = declared;
        unsupported_geometry.geometry.experts = 513;
        assert!(!accepts(unsupported_geometry, 512)); // Declaration disagrees with bound operands.
        assert!(!accepts(unsupported_geometry, 513)); // Agreement cannot widen a fixed kernel.

        let mut wrong_format = declared;
        wrong_format.formats.shared_down = DType::BF16;
        assert!(!accepts(wrong_format, 512)); // Both formats are legal, but the binding disagrees.
        let mut wrong_format = declared;
        wrong_format.formats.router = DType::MQ4G256V2;
        assert!(!accepts(wrong_format, 512));
    }

    #[test]
    fn qwen4_route_rejects_missing_shared_operands() {
        let mut dtypes = canonical_dtypes();
        dtypes.shared = None;
        assert!(!is_canonical_qwen4(&dtypes));
    }

    #[test]
    fn qwen4_route_rejects_q53_outside_shared_down_or_canonical_pair() {
        let mut assert_rejected = |dtypes: &MoeDtypes<'_>| {
            assert!(
                !is_canonical_qwen4(dtypes),
                "illegal q53 placement entered the typed Qwen4 route"
            );
            assert!(matches!(
                reject_mq4g128v2_moe(dtypes),
                Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: _,
                    quant: "MQ4G128V2",
                    ..
                })
            ));
        };

        let mut dtypes = canonical_dtypes();
        dtypes.router = DType::MQ4G128V2;
        assert_rejected(&dtypes);

        let mut dtypes = canonical_dtypes();
        dtypes.shared.as_mut().unwrap().selector = DType::MQ4G128V2;
        assert_rejected(&dtypes);

        let mut dtypes = canonical_dtypes();
        dtypes.shared.as_mut().unwrap().gate = DType::MQ4G128V2;
        assert_rejected(&dtypes);

        let mut dtypes = canonical_dtypes();
        dtypes.shared.as_mut().unwrap().up = DType::MQ4G128V2;
        assert_rejected(&dtypes);

        let mut dtypes = canonical_dtypes();
        dtypes.routed_gate_up = DType::MQ4G128V2;
        dtypes.routed_down = DType::MQ4G256V2;
        assert_rejected(&dtypes);

        let mut dtypes = canonical_dtypes();
        dtypes.routed_gate_up = DType::MQ4G256;
        assert_rejected(&dtypes);
    }

    /// The sealed slab route admits gfx1201 MQ4 / W3 / E8P, gfx1151 MQ4 / W3 / E8P (E8P via
    /// `SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED`) and gfx1100 MQ4 / E8P; every other arch, a
    /// crossed pair and W2 (everywhere) refuse, and a declared cold codec must be the one the
    /// dtypes name. Both states of the gfx1151 E8P switch are pure-decided.
    #[test]
    fn slab_route_follows_the_arch_codec_matrix() {
        use rdna_compute::slab_admission::{
            qwen4_slab_codec_admitted_with, SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED as RECEIPTS,
        };
        let pairs = [
            (DType::MQ4G256V2, DType::MQ4G128V2),
            (DType::MQ3I4G128R256, DType::MQ3I4G128R128),
            (DType::E8PI4G128R256, DType::E8PI4G128R128),
            (DType::E8PI4G128R256, DType::MQ3I4G128R128),
            (DType::MQ2I4G128R256, DType::MQ2I4G128R128),
            (DType::MQ3I4G128R256, DType::MQ4G128V2),
        ];
        let declared_cold = [None, Some(1), Some(3), None, None, None];
        // `landed`: `None` is the shipped switch; `Some` is the pure decision for that state.
        let route = |arch: &str,
                     landed: Option<bool>,
                     (gate_up, down): (DType, DType),
                     cold: Option<u8>| {
            let mut dtypes = canonical_dtypes();
            (dtypes.routed_gate_up, dtypes.routed_down) = (gate_up, down);
            let mut declared = policy(DType::MQ4G128V2);
            declared.capability = MoeRouteCapability::SlabCodecGrouped;
            declared.slab_cold_codec = cold;
            match landed {
                None => select_grouped_route(
                    &dtypes,
                    512,
                    10,
                    2560,
                    640,
                    640,
                    2560,
                    2560,
                    640,
                    None,
                    false,
                    Some(declared),
                    arch,
                ),
                Some(landed) => select_grouped_route_by(
                    &dtypes,
                    512,
                    10,
                    2560,
                    640,
                    640,
                    2560,
                    2560,
                    640,
                    None,
                    false,
                    Some(declared),
                    &|codec| qwen4_slab_codec_admitted_with(arch, codec, landed),
                ),
            }
            .is_some()
        };
        let deltanet = cfg!(feature = "deltanet");
        // The shipped switch: gfx1151 E8P follows exactly it.
        assert_eq!(
            route("gfx1151", None, pairs[2], Some(3)),
            deltanet && RECEIPTS
        );
        assert_eq!(route("gfx1151", None, pairs[1], Some(1)), deltanet);
        for landed in [false, true] {
            for (arch, admitted) in [
                ("gfx1201", [true, true, true, false, false, false]),
                ("gfx1100", [true, false, true, false, false, false]),
                ("gfx1151", [true, true, landed, false, false, false]),
                ("gfx1200", [false; 6]),
                ("gfx942", [false; 6]),
            ] {
                for ((pair, cold), admitted) in pairs.into_iter().zip(declared_cold).zip(admitted) {
                    assert_eq!(
                        route(arch, Some(landed), pair, cold),
                        admitted && deltanet,
                        "{arch} {pair:?} landed={landed}"
                    );
                }
            }
            // A declaration that disagrees with the dtypes refuses.
            for arch in ["gfx1201", "gfx1100"] {
                assert!(!route(arch, Some(landed), pairs[2], Some(1)), "{arch}");
                assert!(!route(arch, Some(landed), pairs[2], None), "{arch}");
                assert!(!route(arch, Some(landed), pairs[0], Some(3)), "{arch}");
            }
            assert!(!route("gfx1201", Some(landed), pairs[1], Some(3)));
            assert!(!route("gfx1201", Some(landed), pairs[1], None));
            assert!(!route("gfx1201", Some(landed), pairs[0], Some(1)));
            assert!(!route("gfx1151", Some(landed), pairs[1], Some(3)));
            assert!(!route("gfx1151", Some(landed), pairs[1], None));
            assert!(!route("gfx1151", Some(landed), pairs[0], Some(1)));
            assert!(!route("gfx1151", Some(landed), pairs[2], Some(1)));
            assert!(!route("gfx1151", Some(landed), pairs[2], None));
            assert!(!route("gfx1151", Some(landed), pairs[0], Some(3)));
            // W2 is refused everywhere, whatever it is declared as.
            for arch in ["gfx1201", "gfx1100", "gfx1151"] {
                for cold in [None, Some(0), Some(1), Some(2), Some(3)] {
                    assert!(
                        !route(arch, Some(landed), pairs[4], cold),
                        "{arch} {cold:?}"
                    );
                }
            }
            // Cold codec = bucket 0, MQ4 = bucket 1.
            use crate::families::moe::{slab_bucket_of_pair, slab_bucket_of_pair_by};
            let bucket = |arch: &str, pair: (DType, DType)| {
                slab_bucket_of_pair_by(
                    &|codec| qwen4_slab_codec_admitted_with(arch, codec, landed),
                    pair.0,
                    pair.1,
                )
            };
            for arch in ["gfx1201", "gfx1100"] {
                assert_eq!(bucket(arch, pairs[0]), Some(1));
                assert_eq!(bucket(arch, pairs[2]), Some(0));
                assert_eq!(bucket(arch, pairs[4]), None);
            }
            assert_eq!(bucket("gfx1201", pairs[1]), Some(0));
            assert_eq!(bucket("gfx1100", pairs[1]), None);
            // gfx1151: MQ4 = bucket 1, W3 = bucket 0, E8P = bucket 0 only once its receipts landed;
            // W2 has none.
            assert_eq!(bucket("gfx1151", pairs[0]), Some(1));
            assert_eq!(bucket("gfx1151", pairs[1]), Some(0));
            assert_eq!(bucket("gfx1151", pairs[2]), landed.then_some(0));
            assert_eq!(bucket("gfx1151", pairs[4]), None);
            // The shipped entry reads the one switch.
            assert_eq!(
                slab_bucket_of_pair("gfx1151", pairs[2].0, pairs[2].1),
                RECEIPTS.then_some(0)
            );
        }
    }

    /// A mixed slab layer (per-expert pairs plus a codec table) is MQ4 plus
    /// exactly one admitted cold codec the policy declares: gfx1100 serves an
    /// E8P layer (MQ4 first or cold first) and refuses W3, gfx1151 serves a W3 layer and, until
    /// `SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED`, refuses E8P (afterwards it serves an E8P layer
    /// like gfx1100), and every arch refuses W3 + E8P in one
    /// layer, W2 anywhere, and a declaration that disagrees with the table.
    #[test]
    fn slab_route_mixed_layers_follow_the_arch_codec_matrix() {
        use rdna_compute::slab_admission::SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED as RECEIPTS;
        const MQ4: (DType, DType) = (DType::MQ4G256V2, DType::MQ4G128V2);
        const W3: (DType, DType) = (DType::MQ3I4G128R256, DType::MQ3I4G128R128);
        const E8P: (DType, DType) = (DType::E8PI4G128R256, DType::E8PI4G128R128);
        const W2: (DType, DType) = (DType::MQ2I4G128R256, DType::MQ2I4G128R128);
        let tags = GpuTensor::null_for_test();
        let route_with = |arch: &str,
                          landed: Option<bool>,
                          first: (DType, DType),
                          tail: &[(DType, DType)],
                          cold: Option<u8>| {
            let mut pairs = vec![first; 512];
            pairs[512 - tail.len()..].copy_from_slice(tail);
            let gate_up: Vec<DType> = pairs.iter().map(|pair| pair.0).collect();
            let down: Vec<DType> = pairs.iter().map(|pair| pair.1).collect();
            let mut dtypes = canonical_dtypes();
            dtypes.routed_has_mixed_experts = true;
            (dtypes.routed_gate_up, dtypes.routed_down) = first;
            dtypes.per_expert_gate_up = Some(&gate_up);
            dtypes.per_expert_down = Some(&down);
            let mut declared = policy(DType::MQ4G128V2);
            declared.capability = MoeRouteCapability::SlabCodecGrouped;
            declared.slab_cold_codec = cold;
            match landed {
                // The shipped switch.
                None => select_grouped_route(
                    &dtypes,
                    512,
                    10,
                    2560,
                    640,
                    640,
                    2560,
                    2560,
                    640,
                    Some(&tags),
                    false,
                    Some(declared),
                    arch,
                ),
                Some(landed) => select_grouped_route_by(
                    &dtypes,
                    512,
                    10,
                    2560,
                    640,
                    640,
                    2560,
                    2560,
                    640,
                    Some(&tags),
                    false,
                    Some(declared),
                    &|codec| {
                        rdna_compute::slab_admission::qwen4_slab_codec_admitted_with(
                            arch, codec, landed,
                        )
                    },
                ),
            }
            .is_some()
        };
        let deltanet = cfg!(feature = "deltanet");
        // The shipped switch decides the shipped route; the pure form decides both states.
        assert_eq!(
            route_with("gfx1151", None, MQ4, &[E8P], Some(3)),
            deltanet && RECEIPTS
        );
        assert_eq!(
            route_with("gfx1151", None, E8P, &[MQ4], Some(3)),
            deltanet && RECEIPTS
        );
        assert_eq!(route_with("gfx1151", None, MQ4, &[W3], Some(1)), deltanet);
        assert_eq!(route_with("gfx1100", None, MQ4, &[E8P], Some(3)), deltanet);
        for landed in [false, true] {
            let route = |arch: &str, first, tail: &[(DType, DType)], cold| {
                route_with(arch, Some(landed), first, tail, cold)
            };
            // MQ4-first E8P layer and an E8P-first one.
            assert_eq!(route("gfx1100", MQ4, &[E8P], Some(3)), deltanet);
            assert_eq!(route("gfx1100", E8P, &[MQ4], Some(3)), deltanet);
            assert_eq!(route("gfx1201", MQ4, &[E8P], Some(3)), deltanet);
            assert_eq!(route("gfx1201", MQ4, &[W3], Some(1)), deltanet);
            // W3 is not served on gfx1100.
            assert!(!route("gfx1100", MQ4, &[W3], Some(1)));
            // W3 + E8P in one layer, W2 anywhere, a mismatched declaration.
            for arch in ["gfx1201", "gfx1100"] {
                for cold in [None, Some(1), Some(3)] {
                    assert!(!route(arch, MQ4, &[W3, E8P], cold), "{arch} {cold:?}");
                    assert!(!route(arch, MQ4, &[E8P, W2], cold), "{arch} {cold:?}");
                    assert!(!route(arch, W2, &[MQ4, E8P], cold), "{arch} {cold:?}");
                }
                assert!(!route(arch, MQ4, &[E8P], Some(1)), "{arch}");
                assert!(!route(arch, MQ4, &[E8P], None), "{arch}");
            }
            // gfx1151 serves W3 next to MQ4 (either order) whatever the E8P receipts say, and
            // refuses a W3 + E8P layer (the two cold codecs, or the unserved E8P) and W2.
            assert_eq!(route("gfx1151", MQ4, &[W3], Some(1)), deltanet);
            assert_eq!(route("gfx1151", W3, &[MQ4], Some(1)), deltanet);
            for cold in [None, Some(1), Some(3)] {
                assert!(
                    !route("gfx1151", MQ4, &[W3, E8P], cold),
                    "{landed} {cold:?}"
                );
                assert!(!route("gfx1151", MQ4, &[W3, W2], cold), "{landed} {cold:?}");
                assert!(
                    !route("gfx1151", MQ4, &[E8P, W2], cold),
                    "{landed} {cold:?}"
                );
            }
            assert!(!route("gfx1151", MQ4, &[W3], Some(3)));
            assert!(!route("gfx1151", MQ4, &[W3], None));
            // gfx1151 E8P: refused until the receipts land, then served exactly like gfx1100.
            assert_eq!(
                route("gfx1151", MQ4, &[E8P], Some(3)),
                deltanet && landed,
                "{landed}"
            );
            assert_eq!(
                route("gfx1151", E8P, &[MQ4], Some(3)),
                deltanet && landed,
                "{landed}"
            );
            assert!(!route("gfx1151", MQ4, &[E8P], Some(1)), "{landed}");
            assert!(!route("gfx1151", MQ4, &[E8P], None), "{landed}");
            for arch in ["gfx1200", "gfx942", "gfx1030"] {
                assert!(!route(arch, MQ4, &[E8P], Some(3)), "{arch}");
            }
        }
    }
}

#[cfg(test)]
mod fp8_route_tests {
    use super::*;
    use crate::families::moe::{
        MoeDtypes, MoeResolution, MoeRouteCapability, MoeRouteFormats, MoeRouteGeometry,
        MoeRoutePolicy, MoeSharedDtypes,
    };

    fn fp8_dtypes() -> MoeDtypes<'static> {
        MoeDtypes {
            router: DType::BF16,
            shared: Some(MoeSharedDtypes {
                selector: DType::BF16,
                gate: DType::BF16,
                up: DType::BF16,
                down: DType::BF16,
            }),
            experts_all_gate_up_mq4: false,
            routed_gate_up: DType::FP8E4M3B128,
            routed_down: DType::FP8E4M3B128,
            routed_has_mixed_experts: false,
            has_paro_shared: false,
            per_expert_gate_up: None,
            per_expert_down: None,
        }
    }

    fn fp8_policy() -> MoeRoutePolicy {
        MoeRoutePolicy {
            expert_keep_mask: 0,
            capability: MoeRouteCapability::Fp8B128Grouped,
            geometry: MoeRouteGeometry {
                experts: 512,
                top_k: 10,
                hidden: 2560,
                routed_intermediate: 640,
                shared_intermediate: 640,
            },
            formats: MoeRouteFormats {
                router: DType::BF16,
                shared_selector: DType::BF16,
                shared_gate: DType::BF16,
                shared_up: DType::BF16,
                shared_down: DType::BF16,
                routed_gate_up: DType::FP8E4M3B128,
                routed_down: DType::FP8E4M3B128,
            },
            slab_cold_codec: None,
        }
    }

    fn qt44_policy() -> MoeRoutePolicy {
        let mut policy = fp8_policy();
        policy.capability = MoeRouteCapability::Qt44Qt53Grouped;
        policy.formats.routed_gate_up = DType::MQ4G256V2;
        policy.formats.routed_down = DType::MQ4G128V2;
        policy
    }

    fn select(
        arch: &str,
        dtypes: &MoeDtypes<'_>,
        policy: Option<MoeRoutePolicy>,
        shape: [usize; 7],
        has_sidecars: bool,
    ) -> Result<Option<MoeRouteCapability>, DispatchError> {
        let [n_exp, k_top, hidden, mi, gate_up_k, down_m, down_k] = shape;
        select_fp8_grouped_route(
            &DispatchCtx::for_test(arch),
            dtypes,
            policy,
            n_exp,
            k_top,
            hidden,
            mi,
            640,
            gate_up_k,
            down_m,
            down_k,
            None,
            has_sidecars,
        )
    }

    const CANONICAL: [usize; 7] = [512, 10, 2560, 640, 2560, 2560, 640];

    fn is_fp8_refusal(result: Result<Option<MoeRouteCapability>, DispatchError>) -> bool {
        matches!(
            result,
            Err(DispatchError::UnsupportedVariant {
                family: "moe",
                quant: "FP8E4M3B128",
                ..
            })
        )
    }

    #[test]
    fn gfx942_qt55_canonical_selects_fp8_route() {
        let selected = select(
            "gfx942",
            &fp8_dtypes(),
            Some(fp8_policy()),
            CANONICAL,
            false,
        );
        assert!(matches!(
            selected,
            Ok(Some(MoeRouteCapability::Fp8B128Grouped))
        ));
    }

    #[test]
    fn non_gfx942_qt55_refuses() {
        for arch in ["gfx1151", "gfx1201", "gfx1100", "gfx1030", "gfx908"] {
            assert!(
                is_fp8_refusal(select(
                    arch,
                    &fp8_dtypes(),
                    Some(fp8_policy()),
                    CANONICAL,
                    false
                )),
                "{arch} must refuse qt55",
            );
        }
    }

    #[test]
    fn absent_or_foreign_policy_refuses_qt55_generically() {
        for policy in [None, Some(qt44_policy())] {
            assert!(is_fp8_refusal(select(
                "gfx942",
                &fp8_dtypes(),
                policy,
                CANONICAL,
                false
            )));
        }
        // Generic guards refuse the same operands.
        assert!(is_fp8_refusal(reject_fp8_moe(&fp8_dtypes()).map(|()| None)));
        assert!(is_fp8_refusal(
            reject_mq4g128v2_moe(&fp8_dtypes()).map(|()| None)
        ));
        assert!(is_fp8_refusal(
            reject_fp8_moe_unless_routed(&fp8_dtypes(), None).map(|()| None)
        ));
        assert!(is_fp8_refusal(
            reject_fp8_moe_unless_routed(&fp8_dtypes(), Some(MoeRouteCapability::Qt44Qt53Grouped))
                .map(|()| None)
        ));
        assert!(is_fp8_refusal(
            reject_fp8_b128(DType::FP8E4M3B128).map(|()| None)
        ));
        reject_fp8_b128(DType::MQ4G256).unwrap();
        // Generic resolution carries the refusal state, never a route.
        let ctx = DispatchCtx::for_test("gfx942");
        let resolution = MoeResolution::resolve_arch(&fp8_dtypes(), 10, ctx.arch.has_wmma());
        assert!(resolution.fp8_b128_refused && !resolution.routed_indexable());
        assert!(resolution.check_generic_route(None).is_err());
    }

    #[test]
    fn typed_fp8_route_may_share_only_bf16_router_and_shared_stages() {
        let fp8 = Some(MoeRouteCapability::Fp8B128Grouped);
        reject_fp8_moe_unless_routed(&fp8_dtypes(), fp8).unwrap();
        let mut shared_fp8 = fp8_dtypes();
        shared_fp8.shared.as_mut().unwrap().down = DType::FP8E4M3B128;
        assert!(reject_fp8_moe_unless_routed(&shared_fp8, fp8).is_err());
        let mut router_fp8 = fp8_dtypes();
        router_fp8.router = DType::FP8E4M3B128;
        assert!(reject_fp8_moe_unless_routed(&router_fp8, fp8).is_err());
    }

    #[test]
    fn non_fp8_layers_are_untouched_by_the_fp8_selector() {
        let mut qt44 = fp8_dtypes();
        qt44.routed_gate_up = DType::MQ4G256V2;
        qt44.routed_down = DType::MQ4G128V2;
        for arch in ["gfx942", "gfx1151"] {
            for policy in [None, Some(qt44_policy())] {
                assert!(matches!(
                    select(arch, &qt44, policy, CANONICAL, false),
                    Ok(None)
                ));
            }
        }
        reject_fp8_moe(&qt44).unwrap();
    }

    #[test]
    fn fp8_route_rejects_noncanonical_geometry_formats_and_operands() {
        let refuses = |dtypes: &MoeDtypes<'_>, policy: MoeRoutePolicy, shape, sidecars| {
            is_fp8_refusal(select("gfx942", dtypes, Some(policy), shape, sidecars))
        };
        assert!(refuses(
            &fp8_dtypes(),
            fp8_policy(),
            [513, 10, 2560, 640, 2560, 2560, 640],
            false
        ));
        assert!(refuses(
            &fp8_dtypes(),
            fp8_policy(),
            [512, 8, 2560, 640, 2560, 2560, 640],
            false
        ));
        assert!(refuses(
            &fp8_dtypes(),
            fp8_policy(),
            [512, 10, 2560, 640, 2560, 1, 640],
            false
        ));
        assert!(refuses(&fp8_dtypes(), fp8_policy(), CANONICAL, true));

        let mut widened = fp8_policy();
        widened.geometry.experts = 513;
        assert!(refuses(
            &fp8_dtypes(),
            widened,
            [513, 10, 2560, 640, 2560, 2560, 640],
            false
        ));

        let mut q8_shared = fp8_policy();
        q8_shared.formats.shared_gate = DType::Q8_0;
        assert!(refuses(&fp8_dtypes(), q8_shared, CANONICAL, false));

        let mut qt44_routed = fp8_policy();
        qt44_routed.formats.routed_gate_up = DType::MQ4G256V2;
        assert!(refuses(&fp8_dtypes(), qt44_routed, CANONICAL, false));

        let mut not_fp8_operands = fp8_dtypes();
        not_fp8_operands.routed_down = DType::MQ4G128V2;
        assert!(refuses(&not_fp8_operands, fp8_policy(), CANONICAL, false));

        let mut quantized_router = fp8_dtypes();
        quantized_router.router = DType::Q8_0;
        assert!(refuses(&quantized_router, fp8_policy(), CANONICAL, false));

        let mut no_shared = fp8_dtypes();
        no_shared.shared = None;
        assert!(refuses(&no_shared, fp8_policy(), CANONICAL, false));

        let mut mixed = fp8_dtypes();
        mixed.routed_has_mixed_experts = true;
        assert!(refuses(&mixed, fp8_policy(), CANONICAL, false));

        static MIXED_TIERS: [DType; 2] = [DType::FP8E4M3B128, DType::MQ4G256V2];
        let mut tiered = fp8_dtypes();
        tiered.per_expert_gate_up = Some(&MIXED_TIERS);
        assert!(refuses(&tiered, fp8_policy(), CANONICAL, false));
    }
}

pub struct Pipeline {
    pub ops: &'static [PipelineOp],
}

impl Pipeline {
    pub fn new(ops: &'static [PipelineOp]) -> Self {
        Self { ops }
    }

    pub fn can_satisfy(&self, requested: &[PipelineOp]) -> bool {
        if self.ops.len() > requested.len() {
            return false;
        }
        self.ops.iter().zip(requested.iter()).all(|(a, b)| a == b)
    }
}

pub struct LinearParams<'a> {
    pub x: &'a GpuTensor,
    pub y: &'a GpuTensor,
    pub buf: &'a GpuTensor,
    pub m: usize,
    pub k: usize,
}

pub enum PipelineParams<'a> {
    Linear(LinearParams<'a>),
}

pub fn execute_pipeline(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    steps: &[PipelineOp],
    params: &PipelineParams,
    dtype: rdna_compute::DType,
    registry: &KernelRegistry,
) -> Result<(), DispatchError> {
    reject_mq4g128v2(dtype, "pipeline")?;
    if let Some(key) = find_fused(registry, ctx, dtype, steps) {
        return dispatch_fused(ctx, gpu, key, params);
    }
    let PipelineParams::Linear(params) = params;
    for &step in steps {
        match step {
            PipelineOp::RotateFwht => {
                use crate::families::rotation::{RotationFamily, RotationParams};
                let rot = RotationFamily::new();
                gpu.ensure_mq_signs()
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                let x_rot = unsafe {
                    GpuTensor {
                        buf: gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias(),
                        shape: vec![params.k],
                        dtype: rdna_compute::DType::F32,
                    }
                };
                rot.run(
                    ctx,
                    gpu,
                    RotationParams {
                        x: params.x,
                        x_up: None,
                        w_norm: None,
                        x_plain: &x_rot,
                        x_rot: &x_rot,
                        awq_scale: None,
                        k: params.k,
                        eps: 1e-6,
                        batch_size: 1,
                        variant: RotationVariant::Plain,
                        givens_pairs: None,
                        givens_theta: None,
                        givens_scales: None,
                        givens_krot: None,
                        givens_group_size: None,
                    },
                )
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            }
            PipelineOp::Gemv => {
                static GEMV_PIPELINE: OnceLock<GemvFamily> = OnceLock::new();
                let gemv = GEMV_PIPELINE.get_or_init(GemvFamily::new);
                let w = WeightRef {
                    buf: params.buf,
                    dtype,
                    m: params.m,
                    k: params.k,
                    row_stride: params.k,
                    rotation: None,
                    awq_scale: None,
                    lloyd_lut_e4m3: None,
                    lloyd_lut_f16: None,
                    lloyd_lut_c16: None,
                };
                gemv.run_auto(ctx, gpu, &w, params.x, params.y)?;
            }
            _ => {
                return Err(DispatchError::UnsupportedVariant {
                    family: "pipeline",
                    variant: "step",
                    arch: "",
                    quant: "",
                });
            }
        }
    }
    Ok(())
}

fn find_fused(
    registry: &KernelRegistry,
    ctx: &DispatchCtx,
    dtype: rdna_compute::DType,
    requested: &[PipelineOp],
) -> Option<KernelKey> {
    use rdna_compute::DType;
    if dtype == DType::MFP4G32
        && requested.len() == 2
        && requested[0] == PipelineOp::RotateFwht
        && requested[1] == PipelineOp::Gemv
    {
        let key = KernelKey::GemvMfp4G32Fused;
        if registry.resolve(key, ctx, None).is_ok() {
            return Some(key);
        }
    }
    None
}

/// Slice a subrange of a flat F32 GpuTensor by element offset + length.
/// The view aliases device memory owned by the source tensor: do NOT free
/// it, and do NOT let it outlive the source. Out-of-range slices panic via
/// the [`hip_bridge::DeviceBuffer::byte_view`] range check.
fn slice_moe_f32_view(src: &GpuTensor, offset_elems: usize, len_elems: usize) -> GpuTensor {
    let byte_offset = offset_elems
        .checked_mul(4)
        .expect("slice_moe_f32_view offset overflow");
    let byte_len = len_elems
        .checked_mul(4)
        .expect("slice_moe_f32_view length overflow");
    GpuTensor {
        buf: src.buf.byte_view(byte_offset, byte_len),
        shape: vec![len_elems],
        dtype: DType::F32,
    }
}

/// Debug localization hook controlled by `HIPFIRE_DUMP_HIDDEN`.
///
/// When enabled, append the selected absolute-position row as a `u32` layer
/// index followed by little-endian F32 values to `{prefix}.{tag}`.  The
/// synchronous readback is refused while graph capture is active or while the
/// retained body is being recorded or routed — a host sync and a D2H inside
/// either window would break the tape — so the check lives here rather than in
/// every caller. This helper preserves the existing opt-in diagnostics
/// semantics and is otherwise allocation-free on the disabled path.
#[allow(clippy::too_many_arguments)]
pub fn dump_hidden_localize(
    gpu: &Gpu,
    x: &GpuTensor,
    n_rows: usize,
    abs_pos_of_row0: usize,
    dim: usize,
    layer_idx: usize,
    tag: &str,
) {
    let prefix = match hipfire_config::developer_var("HIPFIRE_DUMP_HIDDEN") {
        Ok(prefix) => prefix,
        Err(_) => return,
    };
    // A readback cannot happen inside a window whose contents are being recorded
    // or replayed, nor while a hipGraph is capturing.
    if gpu.replay.retained_body_active() || gpu.graphs.capture_mode {
        return;
    }
    let target = hipfire_config::developer_var("HIPFIRE_DUMP_HIDDEN_POS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0usize);
    if target < abs_pos_of_row0 {
        return;
    }
    let row = target - abs_pos_of_row0;
    if row >= n_rows || gpu.hip.device_synchronize().is_err() {
        return;
    }
    let all = match gpu.download_f32(x) {
        Ok(values) => values,
        Err(_) => return,
    };
    let offset = match row.checked_mul(dim) {
        Some(offset) if offset.checked_add(dim).is_some_and(|end| end <= all.len()) => offset,
        _ => return,
    };
    use std::io::Write;
    let path = format!("{prefix}.{tag}");
    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    else {
        return;
    };
    let _ = file.write_all(&(layer_idx as u32).to_le_bytes());
    for value in &all[offset..offset + dim] {
        let _ = file.write_all(&value.to_le_bytes());
    }
}

/// GPU-free unit for the runtime decode batch-size guard (CB5).
/// Extracted so the guard is testable without a GPU or `MoeParams`.
pub fn check_moe_decode_batch_size(batch_size: usize) -> Result<(), DispatchError> {
    if batch_size != 1 {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "decode-requires-batch-1",
            arch: "",
            quant: "",
        });
    }
    Ok(())
}

/// GPU-free pre-guard for MoE decode (#397 Ship 4c). Rejects the two
/// truly-unsupported cases up front — *before* any GPU work — so the caller
/// gets a clean [`DispatchError`] instead of a deep panic in the CPU-top-K
/// fallback (`select_nth_unstable_by(k-1)` panics when `k == 0 || k > n_exp`)
/// or in a kernel launch with no expert to run.
///
/// IMPORTANT: `k != 8` is NOT itself an error. The CPU-top-K fallback
/// (`decode_route_cpu_stage` + `decode_cpu_experts_stage`) legitimately handle any
/// (k=4 for MQ4, k=2 for an F32 router, etc.). This guard must only reject:
///
/// - **(a)** `k` outside `[1, n_exp]` — invalid for top-K selection on either
///   the GPU-top-K fast path or the CPU fallback.
/// - **(b)** a routed dtype that neither path supports: the dtype is not on the
///   GPU-top-K fast path (`!use_gpu_topk`) *and* there are no resident per-expert
///   weights for the CPU fallback to iterate. (When the routed dtype is the only
///   issue but experts are resident, the fallback runs it and its inner
///   `gemv.run_auto` surfaces any genuinely-unsupported dtype as its own clean
///   `DispatchError` — so we must NOT reject that case here.)
///
/// `routed_experts_resident` mirrors `!MoeParams::routed_experts.is_empty()`
/// (false under paged residency, where only the GPU-top-K path is available).
pub fn check_moe_decode_supported(
    use_gpu_topk: bool,
    k: usize,
    n_exp: usize,
    routed_experts_resident: bool,
) -> Result<(), DispatchError> {
    // (a) k-range — required by BOTH the GPU-top-K path and the CPU fallback's
    // `select_nth_unstable_by(k-1)`. Universal precondition, not a k==8 check.
    if k == 0 || k > n_exp {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "decode-k-out-of-range",
            arch: "",
            quant: "",
        });
    }
    // (b) routed dtype on neither path: not GPU-top-K-indexable AND no resident
    // experts to drive the CPU fallback. A non-fast-path dtype WITH resident
    // experts is a valid fallback case (do not reject it here).
    if !use_gpu_topk && !routed_experts_resident {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "decode-routed-dtype-unsupported-no-fallback",
            arch: "",
            quant: "",
        });
    }
    Ok(())
}

/// True when the gate→down step must NOT fuse the FWHT rotation, because the
/// routed `down` weights were packed in the NATURAL basis.
///
/// Every rotated dtype folds the rotation into the silu+mul step so the down
/// GEMV receives an FWHT-basis activation. `MQ2G256LloydU` is the unrotated
/// sibling of MQ2-Lloyd: rotating here would feed a rotated activation to
/// unrotated weights, which produces plausible-looking garbage rather than an
/// error. Extracted as a predicate so the invariant "unrotated dtype ⇔ no
/// rotation anywhere in its path" can be pinned by test — see
/// `unrotated_dtype_skips_both_rotations`.
pub(crate) fn gate_down_skips_rotation(routed_down: DType) -> bool {
    matches!(routed_down, DType::MQ2G256LloydU)
}

/// Frozen mixed-expert in-memory dtype tags (process-local loader metadata,
/// not HFQ wire ABI). Tags 0..6 retain V1 meaning; 7..18 are the MQV2 pairs.
/// Unknown pairs return `None` — callers must Err loudly, never collapse V2→V1.
pub(crate) fn mixed_expert_dtype_tag(gate: DType, down: DType) -> Option<u8> {
    // GL in either position is always rejected — the tag-branched decoder has
    // no GL branch and would silently mis-decode as MQ4.
    if matches!(gate, DType::MQ2G256GL | DType::MQ3G256GL)
        || matches!(down, DType::MQ2G256GL | DType::MQ3G256GL)
    {
        return None;
    }
    match (gate, down) {
        // Tags 0..6 — V1 pair identities (unchanged).
        (DType::MQ4G256, DType::MQ6G256) => Some(0),
        (DType::MQ4G256, DType::MQ2G256Lloyd) => Some(1),
        (DType::MQ4G256, DType::MQ4G256) => Some(2),
        (DType::MQ4G256, DType::MQ3G256Lloyd) => Some(3),
        (DType::MQ4G256, DType::MFP4G32E8) => Some(4),
        (DType::MQ4G256, DType::MFP3G32E8) => Some(5),
        (DType::MQ4G256, DType::MFP2G32E8) => Some(6),
        // Matching non-MQ4 V1 pairs reuse the same tag numbers.
        (DType::MQ6G256, DType::MQ6G256) => Some(0),
        (DType::MQ2G256Lloyd, DType::MQ2G256Lloyd) => Some(1),
        (DType::MQ3G256Lloyd, DType::MQ3G256Lloyd) => Some(3),
        (DType::MFP4G32E8, DType::MFP4G32E8) => Some(4),
        (DType::MFP3G32E8, DType::MFP3G32E8) => Some(5),
        (DType::MFP2G32E8, DType::MFP2G32E8) => Some(6),
        // Tags 7..18 — frozen MQV2 mixed identities (never collapse to 0..6).
        (DType::MQ4G256V2, DType::MQ4G256V2) => Some(7),
        (DType::MQ6G256V2, DType::MQ6G256V2) => Some(8),
        (DType::MQ4G256V2, DType::MQ6G256) => Some(9),
        (DType::MQ4G256V2, DType::MQ2G256Lloyd) => Some(10),
        (DType::MQ4G256V2, DType::MQ4G256) => Some(11),
        (DType::MQ4G256, DType::MQ4G256V2) => Some(12),
        (DType::MQ4G256V2, DType::MQ3G256Lloyd) => Some(13),
        (DType::MQ4G256V2, DType::MFP4G32E8) => Some(14),
        (DType::MQ4G256V2, DType::MFP3G32E8) => Some(15),
        (DType::MQ4G256V2, DType::MFP2G32E8) => Some(16),
        (DType::MQ4G256V2, DType::MQ6G256V2) => Some(17),
        (DType::MQ4G256, DType::MQ6G256V2) => Some(18),
        _ => None,
    }
}

/// Which ninepath D3 family a uniform (gate, down) pair selects, if any.
/// Only HFQ4/MQ4V1 (MQ4G256) may call `gemv_hfq4g256_moe_ninepath_d3`;
/// V2 pairs (MQ4G256V2, MQ6G256V2) must use exact native indexed gate + V2 D4.
/// Split V1/V2 pairings never share a family — wrong header is silent garbage.
pub(crate) fn ninepath_d3_family(gate: DType, down: DType) -> Option<&'static str> {
    match (gate, down) {
        (DType::MQ4G256, DType::MQ4G256) => Some("hfq4"),
        _ => None,
    }
}
/// Bind the current G1 window seal (`#[cfg(lab)]` only) from the stage's
/// existing immutable-identity proof. Fails closed when the routed set
/// offers no immutability proof or no experts resolve: a merely
/// same-pointer-but-mutable cache bank must never pass as the ORIGINAL
/// aggregate owner. Non-lab builds pay nothing (no hook, no seal).
#[cfg(feature = "test-utils")]
pub(crate) fn bind_g1_window(
    gpu: &mut Gpu,
    experts: &dyn crate::families::moe::RoutedExpertWeights,
    n_exp: usize,
    probabilities: &GpuTensor,
    bf16_round_trip: bool,
) -> Result<(), DispatchError> {
    if !gpu.lab_g1_observer_active() {
        return Ok(());
    }
    let view = slice_moe_f32_view(probabilities, 0, 10);
    let values = gpu
        .download_f32(&view)
        .map_err(|e| DispatchError::Hip(e.to_string()))?;
    let router_probability_words = std::array::from_fn(|i| values[i].to_bits());
    let Some(identity) = experts.immutable_identity() else {
        return Err(DispatchError::Hip(
            "g1 window: routed expert set has no immutable identity".into(),
        ));
    };
    if experts.len() != n_exp || experts.is_empty() {
        return Err(DispatchError::Hip(
            "g1 window: routed expert count mismatch".into(),
        ));
    }
    gpu.set_lab_g1_window(Some(rdna_compute::LabG1WindowSeal {
        expert_set_identity: identity,
        n_exp,
        experts_host_mapped: experts.host_mapped(),
        router_probability_words,
        bf16_round_trip,
    }));
    Ok(())
}

/// True when per-expert gate_up dtypes contain >1 distinct DType (exact equality,
/// not family). Used to decide mixed gate precedence: whenever gate_up exact
/// dtype varies, mixed gate kernel must run before representative MQ4V2/MQ6V2/V1 arms.
/// `None` (uniform table) or all-equal `Some` ⇒ false (uniform shortcut allowed).
pub(crate) fn gate_up_varies(per_expert_gate_up: Option<&[DType]>) -> bool {
    match per_expert_gate_up {
        Some(slice) => slice
            .split_first()
            .map_or(false, |(first, rest)| rest.iter().any(|dt| dt != first)),
        None => false,
    }
}

/// Whether decode should use the mixed gate_up kernel. Precedence gate:
/// whenever `has_tags && gate_up_varies`, mixed gate must run before
/// representative V1/V2 arms; uniform shortcut only when !gate_up_varies.
pub(crate) fn decode_gate_uses_mixed(has_tags: bool, gate_up_varies: bool) -> bool {
    has_tags && gate_up_varies
}

/// Prefill Path-1 batched gate_up kind with tag awareness.
/// When `has_tags && gate_up_varies`, the tagged layer must use the mixed
/// batched gate launcher (`gemv_mixed_moe_gate_up_k8_indexed_batched`);
/// uniform shortcut only when `!gate_up_varies` (exact DType equality).
/// Returns `Some("mixed")` for the mixed case, else delegates to
/// `prefill_path1_gate_up_kind`.
pub(crate) fn prefill_path1_gate_up_kind_tag_aware(
    gate_up: DType,
    has_tags: bool,
    gate_up_varies: bool,
) -> Option<&'static str> {
    if decode_gate_uses_mixed(has_tags, gate_up_varies) {
        Some("mixed")
    } else {
        prefill_path1_gate_up_kind(gate_up)
    }
}

/// Prefill Path-1 expanded-down kind with tag awareness.
/// Mixed Path1 must use the mixed batched down launcher
/// (`gemv_mixed_moe_down_k8_indexed_batched_expanded`) when tags exist;
/// do not representative-dispatch tagged layers. Returns `Some("mixed")`
/// when `has_tags`, else delegates to `prefill_path1_down_kind`.
pub(crate) fn prefill_path1_down_kind_tag_aware(
    down: DType,
    has_tags: bool,
) -> Option<&'static str> {
    if has_tags {
        Some("mixed")
    } else {
        prefill_path1_down_kind(down)
    }
}

/// Which ninepath D4 family a uniform (gate, down) pair selects, if any.
/// Split V1/V2 pairings never share a family — wrong header is silent garbage.
pub(crate) fn ninepath_d4_family(gate: DType, down: DType) -> Option<&'static str> {
    match (gate, down) {
        (DType::MQ4G256, DType::MQ4G256) => Some("hfq4"),
        (DType::MQ2G256Lloyd, DType::MQ3G256Lloyd) => Some("mq3l"),
        (DType::MQ4G256V2, DType::MQ4G256V2) => Some("mq4v2"),
        (DType::MQ6G256V2, DType::MQ6G256V2) => Some("mq6v2"),
        _ => None,
    }
}

/// Decode non-ninepath expanded-down kernel kind for a uniform routed_down.
/// V2 never aliases V1 HFQ4/HFQ6 — the final HFQ4 fallthrough is V1-only.
pub(crate) fn decode_expanded_down_kind(down: DType) -> Option<&'static str> {
    match down {
        DType::MQ4G256 => Some("hfq4"),
        DType::MQ4G256V2 => Some("mq4v2"),
        DType::MQ4G128V2 => Some("qt53_f32"),
        DType::MQ5G256 => Some("hfq5"),
        DType::MQ6G256 => Some("hfq6"),
        DType::MQ6G256V2 => Some("mq6v2"),
        DType::MFP4G32E8 => Some("mfp4e8"),
        DType::ParoQ4G128 => Some("paro"),
        DType::MQ2G256Lloyd | DType::MQ2G256LloydU => Some("mq2lloyd_atomic"),
        DType::MQ3G256Lloyd => Some("mq3lloyd_atomic"),
        DType::MQ2G256GL => Some("mq2gl_atomic"),
        DType::MQ3G256GL => Some("mq3gl_atomic"),
        _ => None,
    }
}

/// Prefill Path-1 batched gate_up kernel kind. V2 has native batched launchers.
pub(crate) fn prefill_path1_gate_up_kind(gate_up: DType) -> Option<&'static str> {
    match gate_up {
        DType::MQ4G256 => Some("hfq4"),
        DType::MQ4G256V2 => Some("mq4v2"),
        DType::MQ5G256 => Some("hfq5"),
        DType::MQ6G256 => Some("hfq6"),
        DType::MQ6G256V2 => Some("mq6v2"),
        DType::MFP4G32E8 => Some("mfp4e8"),
        DType::ParoQ4G128 => Some("paro"),
        _ => None,
    }
}

/// Prefill Path-1 expanded-down kernel kind.
pub(crate) fn prefill_path1_down_kind(down: DType) -> Option<&'static str> {
    match down {
        DType::MQ4G256 => Some("hfq4"),
        DType::MQ4G256V2 => Some("mq4v2"),
        DType::MQ4G128V2 => Some("qt53_f32"),
        DType::MQ5G256 => Some("hfq5"),
        DType::MQ6G256 => Some("hfq6"),
        DType::MQ6G256V2 => Some("mq6v2"),
        DType::MFP4G32E8 => Some("mfp4e8"),
        DType::ParoQ4G128 => Some("paro"),
        _ => None,
    }
}

/// Uniform grouped-GEMM kernel kind admitted on gfx11/gfx12 Path 2.
pub(crate) fn grouped_gemm_kind(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::MQ4G256 => Some("hfq4"),
        DType::MQ4G256V2 => Some("mq4v2"),
        DType::MQ6G256 => Some("hfq6"),
        DType::MQ6G256V2 => Some("mq6v2"),
        DType::MFP4G32E8 => Some("mfp4e8"),
        DType::ParoQ4G128 => Some("paro"),
        DType::MQ2G256Lloyd | DType::MQ2G256LloydU => Some("mq2lloyd"),
        DType::MQ3G256Lloyd => Some("mq3lloyd"),
        _ => None,
    }
}

/// Grouped-GEMM kernel kind for a routed *down* projection: the gate/up kinds
/// of [`grouped_gemm_kind`] plus the qwen35 qt53 F32 grouped down. qt53 has no
/// gate/up kind: [`grouped_gemm_kind`] keeps refusing it.
pub(crate) fn grouped_gemm_down_kind(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::MQ4G128V2 => Some("qt53_f32"),
        _ => grouped_gemm_kind(dtype),
    }
}

/// Which projection a grouped GEMM serves. Only `Down` may carry qt53.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GroupedGemmRole {
    GateUp,
    Down,
}

/// Pure admission of a qt53 grouped GEMM: the routed DOWN projection only
/// (`x_row_div == 1`, one X row per top-k slot, no dtype tags, a live
/// `n_exp`); gate/up qt53 does not exist and is refused before any launch.
pub(crate) fn grouped_gemm_qt53_down_admitted(
    role: GroupedGemmRole,
    has_dtype_tags: bool,
    x_row_div: usize,
    n_exp: usize,
) -> Result<(), DispatchError> {
    if role == GroupedGemmRole::Down && !has_dtype_tags && x_row_div == 1 && n_exp > 0 {
        Ok(())
    } else {
        Err(crate::families::moe::qt53_fallback_unsupported(
            "qt53-grouped-down-only",
        ))
    }
}

/// Shared-expert dense down route. V2 must use exact dense V2 GEMV — never the
/// V1 HFQ4 residual_sigmoid_scaled launcher.
pub(crate) fn shared_dense_down_kind(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::MQ4G256 => Some("hfq4_sigmoid_scaled"),
        DType::MQ4G256V2 => Some("mq4v2_prerotated"),
        DType::MQ6G256V2 => Some("mq6v2_prerotated"),
        DType::MQ4G128V2 => None,
        _ => Some("run_auto"),
    }
}

/// Dispatch a uniform single-token routed-expert gate/up projection by its exact
/// wire dtype. Architecture code supplies tensors and dimensions; format
/// selection remains in the dispatch layer so V1/V2 headers cannot be confused.
#[allow(clippy::too_many_arguments)]
pub fn run_uniform_moe_gate_up(
    gpu: &mut Gpu,
    dtype: DType,
    expert_ptrs: &GpuTensor,
    topk_indices: &GpuTensor,
    x_rot: &GpuTensor,
    gate: &GpuTensor,
    up: &GpuTensor,
    m: usize,
    k: usize,
    k_top: usize,
) -> Result<(), DispatchError> {
    reject_fp8_b128(dtype)?;
    reject_mq4g128v2(dtype, "moe")?;
    let hip = |result: hip_bridge::HipResult<()>| {
        result.map_err(|error| DispatchError::Hip(error.to_string()))
    };
    match dtype {
        DType::MQ4G256 | DType::HFQ4G256 => hip(gpu.gemv_hfq4g256_moe_gate_up_k8_indexed(
            expert_ptrs,
            topk_indices,
            x_rot,
            gate,
            up,
            m,
            k,
            k_top,
        )),
        DType::MQ6G256 | DType::HFQ6G256 => hip(gpu.gemv_hfq6g256_moe_gate_up_k8_indexed(
            expert_ptrs,
            topk_indices,
            x_rot,
            gate,
            up,
            m,
            k,
            k_top,
        )),
        DType::MQ4G256V2 => {
            if k_top != 8 {
                return Err(DispatchError::Hip(format!(
                    "MQ4G256V2 indexed gate/up requires top_k=8, got {k_top}"
                )));
            }
            hip(gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed(
                expert_ptrs,
                topk_indices,
                x_rot,
                gate,
                up,
                m,
                k,
            ))
        }
        DType::MQ6G256V2 => {
            if k_top != 8 {
                return Err(DispatchError::Hip(format!(
                    "MQ6G256V2 indexed gate/up requires top_k=8, got {k_top}"
                )));
            }
            hip(gpu.gemv_mq6g256v2_moe_gate_up_k8_indexed(
                expert_ptrs,
                topk_indices,
                x_rot,
                gate,
                up,
                m,
                k,
            ))
        }
        // qt=52 has NO indexed-LUT MoE kernel: fail closed naming the family.
        // Lloyd routed experts take the generic per-expert fallback (per-expert
        // LUT GEMVs through GemvFamily), never this indexed path.
        DType::MQ4G256V2Lloyd => Err(DispatchError::Hip(format!(
            "moe indexed gate/up: MQ4G256V2Lloyd has no indexed LUT kernel; refusing uniform decode"
        ))),
        other => Err(DispatchError::Hip(format!(
            "uniform indexed gate/up unsupported dtype {other:?}"
        ))),
    }
}

/// Dispatch a uniform routed-expert expanded-down projection by its exact wire
/// dtype. The expanded result remains uncombined for the caller's weighted sum.
#[allow(clippy::too_many_arguments)]
pub fn run_uniform_moe_down_expanded(
    gpu: &mut Gpu,
    dtype: DType,
    expert_ptrs: &GpuTensor,
    topk_indices: &GpuTensor,
    x_rot: &GpuTensor,
    out: &GpuTensor,
    m: usize,
    k: usize,
    k_top: usize,
    batch_size: usize,
) -> Result<(), DispatchError> {
    reject_fp8_b128(dtype)?;
    // qt53 (the qwen35 routed-down typed fallback) is dispatched below; every
    // other refusal stays.
    if dtype != DType::MQ4G128V2 {
        reject_mq4g128v2(dtype, "moe")?;
    }
    let hip = |result: hip_bridge::HipResult<()>| {
        result.map_err(|error| DispatchError::Hip(error.to_string()))
    };
    match dtype {
        DType::MQ4G256 => hip(gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
            expert_ptrs,
            topk_indices,
            x_rot,
            out,
            m,
            k,
            k_top,
            batch_size,
        )),
        DType::MQ6G256 => hip(gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
            expert_ptrs,
            topk_indices,
            x_rot,
            out,
            m,
            k,
            k_top,
            batch_size,
        )),
        DType::MQ4G256V2 => hip(gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
            expert_ptrs,
            topk_indices,
            x_rot,
            out,
            m,
            k,
            k_top,
            batch_size,
        )),
        DType::MQ6G256V2 => hip(gpu.gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded(
            expert_ptrs,
            topk_indices,
            x_rot,
            out,
            m,
            k,
            k_top,
            batch_size,
        )),
        // qwen35 qt53 routed-down typed fallback (MTP head): the caller has
        // rotated `x_rot` into FWHT128 (see `routed_down_rotation`). The pointer
        // table is `[2 * n_exp]` F32 (`n_exp` u64 pointers), so `size / 8` is an
        // upper bound of the expert count for the kernel's expert-id guard.
        DType::MQ4G128V2 => hip(gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
            expert_ptrs,
            topk_indices,
            x_rot,
            out,
            m,
            k,
            batch_size,
            expert_ptrs.buf.size() / 8,
            k_top,
        )),
        // qt=52 has NO indexed-LUT MoE kernel: fail closed naming the family.
        DType::MQ4G256V2Lloyd => Err(DispatchError::Hip(format!(
            "moe indexed down: MQ4G256V2Lloyd has no indexed LUT kernel; refusing uniform decode"
        ))),
        other => Err(DispatchError::Hip(format!(
            "uniform expanded down unsupported dtype {other:?}"
        ))),
    }
}

/// Build the permute-to-contiguous mapping for the mixed-tier path (pure; CPU-
/// unit-tested). Given the per-tier `buckets` (first-seen order) and the top-k
/// width `k`, returns `(perm, ranges)` where:
///   - `perm[new_rank] = old_rank` — concatenating each bucket's `ranks` makes
///     every tier a contiguous block in the new order.
///   - `ranges[b] = (lo, n)` — bucket `b`'s contiguous `[lo, lo+n)` slice.
///
/// EQUIVALENCE INVARIANT (CPU-checkable here): for an all-ONE-tier table there
/// is exactly one bucket whose `ranks` are already `0..k` in order, so `perm`
/// is the IDENTITY and `ranges == [(0, k)]`. That is what makes the mixed path
/// emit the same kernel calls as the uniform path for a uniform table.
fn build_contiguous_permutation(
    buckets: &[crate::families::moe_buckets::TierBucket],
    k: usize,
) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut perm: Vec<usize> = Vec::with_capacity(k);
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(buckets.len());
    for b in buckets {
        let lo = perm.len();
        perm.extend_from_slice(&b.ranks);
        ranges.push((lo, b.ranks.len()));
    }
    debug_assert_eq!(perm.len(), k, "permutation must cover all k ranks");
    (perm, ranges)
}

/// Static name for a DType (for UnsupportedVariant.quant in the mixed path).
/// Covers the tiers a routed expert can realistically carry so an
/// unsupported-tier error names the actual offending tier (e.g. "Q8_0")
/// instead of a useless "other".
fn dtype_name(d: DType) -> &'static str {
    match d {
        DType::MQ4G256 => "MQ4G256",
        DType::MQ4G256V2 => "MQ4G256V2",
        DType::MQ6G256 => "MQ6G256",
        DType::MQ6G256V2 => "MQ6G256V2",
        DType::ParoQ4G128 => "ParoQ4G128",
        DType::Q8_0 => "Q8_0",
        DType::MQ3G256 => "MQ3G256",
        DType::MQ2G256 => "MQ2G256",
        DType::MQ2G256Lloyd => "MQ2G256Lloyd",
        DType::MQ2G256LloydU => "MQ2G256LloydU",
        DType::MQ3G256Lloyd => "MQ3G256Lloyd",
        DType::MQ2G256GL => "MQ2G256GL",
        DType::MQ3G256GL => "MQ3G256GL",
        _ => "other",
    }
}

/// Execute the routed-expert decode subgraph after model-owned route
/// selection. Keeping this boundary in the shared family guarantees that the
/// ordinary and heterogeneous DS4 paths use the same gate/up, activation,
/// rotation, down, and combine sequence.
pub fn run_moe_decode_selected(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeSelectedParams,
) -> Result<(), DispatchError> {
    if p.batch_size != 1 {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "selected-decode-requires-batch-1",
            arch: "",
            quant: "",
        });
    }

    let lloyd = MoeMq2Lloyd {
        swiglu_limit: p.swiglu_limit,
        uses_atomic_down: p.uses_atomic_moe_down,
        native: p.native_mq2_backend,
        nonowned_gate_up_dummy: p.nonowned_gate_up_dummy,
    };
    // 2. Indexed MQ2-Lloyd gate_up: all k_top experts in one launch
    //    (M = 2*mi; the kernel splits rows r<mi to gate, r>=mi to up).
    mq2_lloyd_gate_up(
        gpu,
        Some(&lloyd),
        p.expert_gate_up_ptrs,
        p.topk_indices,
        p.x_rot,
        p.gate_batch,
        p.up_batch,
        2 * p.mi,
        p.hidden,
        p.k_top,
    )?;
    // 3. Batched silu*mul*clamp (in place into gate_batch), then batched FWHT rotate.
    mq2_lloyd_silu_rotate(
        gpu,
        &lloyd,
        p.gate_batch,
        p.up_batch,
        p.rot_batch,
        p.mi,
        p.k_top,
    )?;
    // 4. Indexed MQ2-Lloyd down. Deterministic (default): expanded per-expert
    //    write + fixed-order combine into ffn_out, bit-reproducible for greedy and
    //    spec-decode. MOE_DETERMINISTIC=0 keeps the atomic residual (bench only).
    if lloyd.deterministic_down() {
        mq2_lloyd_down_expanded(
            gpu,
            &lloyd,
            p.expert_down_ptrs,
            p.expert_gate_up_ptrs,
            p.topk_indices,
            p.rot_batch,
            p.down_expanded,
            p.hidden,
            p.mi,
            p.k_top,
        )?;
        hip!(gpu.moe_down_combine_k8_batched(
            p.down_expanded,
            p.topk_weights,
            p.ffn_out,
            p.hidden,
            p.k_top,
            1,
        ))?;
    } else {
        mq2_lloyd_down_residual(
            gpu,
            p.expert_down_ptrs,
            p.topk_indices,
            p.topk_weights,
            p.rot_batch,
            p.ffn_out,
            p.hidden,
            p.mi,
            p.k_top,
            p.uses_atomic_moe_down,
        )?;
    }
    Ok(())
}

/// Indexed MQ2-Lloyd gate_up over the k_top experts (M = 2*mi: rows r<mi feed
/// `y_gate`, r>=mi feed `y_up`). A native backend, when present, takes it.
#[allow(clippy::too_many_arguments)]
fn mq2_lloyd_gate_up(
    gpu: &mut Gpu,
    lloyd: Option<&MoeMq2Lloyd<'_>>,
    expert_gate_up_ptrs: &GpuTensor,
    topk_indices: &GpuTensor,
    x_rot: &GpuTensor,
    y_gate: &GpuTensor,
    y_up: &GpuTensor,
    m: usize,
    k: usize,
    k_top: usize,
) -> Result<(), DispatchError> {
    if let Some(native) = lloyd.and_then(|l| l.native) {
        let dummy = lloyd.and_then(|l| l.nonowned_gate_up_dummy);
        hip!(native.gate_up(
            gpu,
            expert_gate_up_ptrs,
            dummy,
            topk_indices,
            x_rot,
            y_gate,
            y_up,
            m,
            k,
            k_top,
        ))
    } else {
        hip!(gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
            expert_gate_up_ptrs,
            topk_indices,
            x_rot,
            y_gate,
            y_up,
            m,
            k,
            k_top,
        ))
    }
}

/// Batched SwiGLU silu*mul*clamp (in place into `gate`), then the batched FWHT
/// rotate of each expert's activation into `rot`.
fn mq2_lloyd_silu_rotate(
    gpu: &mut Gpu,
    lloyd: &MoeMq2Lloyd<'_>,
    gate: &GpuTensor,
    up: &GpuTensor,
    rot: &GpuTensor,
    mi: usize,
    k_top: usize,
) -> Result<(), DispatchError> {
    hip!(gpu.deepseek4_silu_mul_clamp_f32_batched(gate, up, gate, mi, k_top, lloyd.swiglu_limit,))?;
    if let Some(native) = lloyd.native {
        hip!(native.rotate_x_batched(gpu, gate, rot, mi, k_top))
    } else {
        hip!(gpu.rotate_x_mq_batched(gate, rot, mi, k_top))
    }
}

/// Indexed MQ2-Lloyd down into the per-expert expanded buffer (the deterministic
/// path; the caller combines). A native backend, when present, takes it.
#[allow(clippy::too_many_arguments)]
fn mq2_lloyd_down_expanded(
    gpu: &mut Gpu,
    lloyd: &MoeMq2Lloyd<'_>,
    expert_down_ptrs: &GpuTensor,
    expert_gate_up_ptrs: &GpuTensor,
    topk_indices: &GpuTensor,
    rot: &GpuTensor,
    expanded: &GpuTensor,
    m: usize,
    k: usize,
    k_top: usize,
) -> Result<(), DispatchError> {
    if let Some(native) = lloyd.native {
        hip!(native.down_expanded(
            gpu,
            expert_down_ptrs,
            expert_gate_up_ptrs,
            lloyd.nonowned_gate_up_dummy,
            topk_indices,
            rot,
            expanded,
            m,
            k,
            k_top,
            1,
        ))
    } else {
        hip!(gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
            expert_down_ptrs,
            topk_indices,
            rot,
            expanded,
            m,
            k,
            k_top,
            1,
        ))
    }
}

/// Indexed MQ2-Lloyd down accumulated into `residual` (the atomic or
/// non-deterministic path). Portable only, as in beta's selected decode.
#[allow(clippy::too_many_arguments)]
fn mq2_lloyd_down_residual(
    gpu: &mut Gpu,
    expert_down_ptrs: &GpuTensor,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
    rot: &GpuTensor,
    residual: &GpuTensor,
    m: usize,
    k: usize,
    k_top: usize,
    atomic: bool,
) -> Result<(), DispatchError> {
    hip!(
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
            expert_down_ptrs,
            topk_indices,
            topk_weights,
            rot,
            residual,
            m,
            k,
            k_top,
            atomic,
        )
    )
}

/// MQ2-Lloyd grouped-GEMM kernel variant (deepseek4 research levers; default
/// `Lloyd4w` on gfx11+, `Base` otherwise). Selected once per gate_up/down call.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GroupedLloydVariant {
    /// Native CDNA3 wave64 MFMA path. Selected only for gfx942 so the
    /// wave32 RDNA WMMA variants below retain their existing routes.
    MfmaGfx942,
    /// i8 WMMA MMQ path (gfx1151): decodes the 2-bit Lloyd index via an int8
    /// codebook LUT and runs i8 WMMA at ~2x the FP16 rate. Top priority when
    /// enabled — ~1.7x the FP16 grouped GEMM on the DeepSeek-V4 prefill shape.
    I8,
    N32,
    Cnd,
    EightW,
    Nosync,
    Mmqload,
    Lloyd4w,
    Base,
}

/// Mirror of `ffn_batched`'s grouped-GEMM if/else-if ladder (priority order:
/// n32 > cnd > 8w > nosync > mmqload > 4w > base). `n32`/`cnd`/`eightw` apply
/// only on the 4w path; `use_nosync` ⊂ `use_mmqload` ⊂ `use_lloyd_4w`.
fn select_grouped_lloyd_variant(
    mfma_gfx942: bool,
    use_lloyd_4w: bool,
    i8: bool,
    n32: bool,
    cnd: bool,
    eightw: bool,
    use_mmqload: bool,
    use_nosync: bool,
) -> GroupedLloydVariant {
    if mfma_gfx942 {
        GroupedLloydVariant::MfmaGfx942
    } else if i8 {
        GroupedLloydVariant::I8
    } else if use_lloyd_4w && n32 {
        GroupedLloydVariant::N32
    } else if use_lloyd_4w && cnd {
        GroupedLloydVariant::Cnd
    } else if use_lloyd_4w && eightw {
        GroupedLloydVariant::EightW
    } else if use_nosync {
        GroupedLloydVariant::Nosync
    } else if use_mmqload {
        GroupedLloydVariant::Mmqload
    } else if use_lloyd_4w {
        GroupedLloydVariant::Lloyd4w
    } else {
        GroupedLloydVariant::Base
    }
}

fn use_gfx1151_i8_moe(arch: &str) -> bool {
    arch == "gfx1151"
}

fn use_gfx1151_i8_moe_perm() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MQ2_PERM")
            .ok()
            .as_deref()
            != Some("0")
    })
}

/// Dispatch one MQ2-Lloyd grouped GEMM. All seven variants share the signature
/// `(ptrs, tile_ids, slot_index, x, y, m, k, x_row_div, m_total_max, rows)`, so
/// this is called identically for gate_up (m=2*im, k=hidden, x_row_div=k_top,
/// rows=B) and down (m=hidden, k=im, x_row_div=1, rows=B*k_top).
#[allow(clippy::too_many_arguments)]
fn dispatch_grouped_lloyd(
    gpu: &mut Gpu,
    variant: GroupedLloydVariant,
    ptrs: &GpuTensor,
    tile_ids: &GpuTensor,
    slot_index: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    x_row_div: usize,
    m_total_max: usize,
    rows: usize,
) -> Result<(), DispatchError> {
    use GroupedLloydVariant as V;
    let r = match variant {
        V::MfmaGfx942 => gpu.gemm_mq2g256_lloyd_moe_grouped_mfma_gfx942(
            ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total_max,
            rows,
        ),
        V::I8 if use_gfx1151_i8_moe_perm() => gpu.gemm_mq2g256_lloyd_moe_grouped_mmq_perm_gfx1151(
            ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total_max,
            rows,
        ),
        V::I8 => gpu.gemm_mq2g256_lloyd_moe_grouped_mmq_gfx1151(
            ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total_max,
            rows,
        ),
        V::N32 => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32(
            ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total_max,
            rows,
        ),
        V::Cnd => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_cnd(
            ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total_max,
            rows,
        ),
        V::EightW => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_8w_k2(
            ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total_max,
            rows,
        ),
        V::Nosync => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload_nosync(
            ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total_max,
            rows,
        ),
        V::Mmqload => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload(
            ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total_max,
            rows,
        ),
        V::Lloyd4w => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2(
            ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total_max,
            rows,
        ),
        V::Base => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_k2(
            ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total_max,
            rows,
        ),
    };
    r.map_err(|e| DispatchError::Hip(e.to_string()))
}

/// DeepSeek-V4 batched/prefill MoE executor. Transcribes the routed block of
/// `hipfire-arch-deepseek4::forward::ffn_batched`: routing (hash or bias-aware)
/// → routed experts (grouped GEMM when `batch_size >= gate`, else scalar K4
/// indexed) → combine into `p.ffn_out` (the shared expert already seeded it).
/// Router GEMV + `sqrt_softplus` and the shared expert stay model-owned.
pub fn run_moe_prefill_bias_aware(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeBiasAwarePrefillParams,
) -> Result<(), DispatchError> {
    use crate::families::moe::MoePrefillRouting;
    let (hidden, im, n_exp, k_top, batch_size) = (p.hidden, p.mi, p.n_exp, p.k_top, p.batch_size);

    // ── Routing → topk_indices / topk_weights ────────────────────────────────
    match &p.routing {
        MoePrefillRouting::Hash { tid2eid, tokens } => {
            hip!(gpu.hash_router_normalize_f32_batched(
                tid2eid,
                p.scores,
                tokens,
                p.topk_indices,
                p.topk_weights,
                n_exp as i32,
                k_top as i32,
                p.route_scale,
                batch_size as i32,
            ))?;
        }
        MoePrefillRouting::BiasAware { gate_bias } => {
            hip!(gpu.deepseek4_moe_topk_bias_aware_batched_f32(
                p.scores,
                gate_bias,
                p.topk_indices,
                p.topk_weights,
                n_exp as i32,
                k_top as i32,
                p.route_scale,
                batch_size as i32,
            ))?;
        }
    }

    // DIAG: dump per-layer topk indices ([B, k_top] i32) — off by default.
    if let Ok(path) = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_DUMP_TOPK") {
        use std::io::Write;
        let raw = hip!(gpu.download_f32(p.topk_indices))?;
        let n = batch_size * k_top;
        let mut indices: Vec<i32> = Vec::with_capacity(n);
        for i in 0..n {
            indices.push(raw[i].to_bits() as i32);
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| DispatchError::Hip(format!("dump_topk open {path}: {e:?}")))?;
        let header = [p.layer_idx as i32, batch_size as i32, k_top as i32];
        let header_bytes = unsafe { std::slice::from_raw_parts(header.as_ptr() as *const u8, 12) };
        f.write_all(header_bytes)
            .map_err(|e| DispatchError::Hip(format!("dump_topk header: {e:?}")))?;
        let data_bytes =
            unsafe { std::slice::from_raw_parts(indices.as_ptr() as *const u8, indices.len() * 4) };
        f.write_all(data_bytes)
            .map_err(|e| DispatchError::Hip(format!("dump_topk data: {e:?}")))?;
    }

    // ── Grouped vs scalar gate ────────────────────────────────────────────────
    let gate_threshold: usize = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let use_grouped = batch_size >= gate_threshold
        && hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_GROUPED").as_deref() != Ok("0");

    // Shared research levers (read once; default 4w on gfx11+).
    let lloyd_4w_base =
        match hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W").as_deref() {
            Ok("0") => Some(false),
            Ok("1") => Some(true),
            _ => None,
        };
    let arch_4w = gpu.arch.starts_with("gfx11") || gpu.arch.starts_with("gfx12");
    let n32 = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_N32").as_deref() == Ok("1");
    let cnd = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_CND").as_deref() == Ok("1");
    let eightw = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_8W").as_deref() == Ok("1");
    let mmqload_env =
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_MMQLOAD").as_deref() == Ok("1");
    let nosync_env =
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_NOSYNC").as_deref() == Ok("1");
    // i8 MMQ path (gfx1151 only): 2-bit Lloyd → int8 codebook LUT + i8 WMMA.
    let i8_moe = use_gfx1151_i8_moe(&gpu.arch);

    if use_grouped {
        const BLOCK_M: usize = 16;
        let m_total_max = batch_size * k_top + n_exp * BLOCK_M;

        // Scatter: histogram + offsets + permute (single launch).
        hip!(gpu.moe_scatter_fused_k8(
            p.topk_indices,
            p.expert_token_counts,
            p.expert_offsets,
            p.sorted_slot_index,
            p.expert_tile_ids,
            p.inverse_perm,
            batch_size * k_top,
            n_exp,
            m_total_max,
            BLOCK_M,
        ))?;

        // Grouped gate_up GEMM (M=2*im, K=hidden, x_row_div=k_top, rows=B).
        let use_lloyd_4w_gu =
            lloyd_4w_base.unwrap_or(arch_4w) && (2 * im) % 64 == 0 && hidden % 256 == 0;
        let use_mmqload_gu = use_lloyd_4w_gu && mmqload_env;
        let use_nosync_gu = use_mmqload_gu && nosync_env;
        // i8 path requires (2*im)%16==0 && hidden%256==0 (looser than 4w's %64).
        let use_i8_gu = i8_moe && (2 * im) % 16 == 0 && hidden % 256 == 0;
        let v_gu = select_grouped_lloyd_variant(
            false,
            use_lloyd_4w_gu,
            use_i8_gu,
            n32,
            cnd,
            eightw,
            use_mmqload_gu,
            use_nosync_gu,
        );
        dispatch_grouped_lloyd(
            gpu,
            v_gu,
            p.expert_gate_up_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            p.x_rot,
            p.y_gate_up_grouped,
            2 * im,
            hidden,
            k_top,
            m_total_max,
            batch_size,
        )?;

        // Unscatter + SwiGLU·clamp.
        // Exact gfx1151 defaults to the raw-bit-certified fused consumer: it
        // removes the full up_batch write/read and one launch per routed MoE
        // layer. Other architectures retain the established two-kernel path.
        // The developer override remains an explicit rollback/screening aid;
        // shipping gfx1151 behavior does not depend on an environment flag.
        let use_fused_unscatter_silu = match hipfire_config::developer_var(
            "HIPFIRE_DEEPSEEK4_FUSED_UNSCATTER_SILU",
        )
        .as_deref()
        {
            Ok("0") => false,
            Ok(_) => true,
            Err(_) => gpu.arch.eq_ignore_ascii_case("gfx1151"),
        };
        if use_fused_unscatter_silu {
            hip!(gpu.moe_unscatter_silu_clamp_k8(
                p.y_gate_up_grouped,
                p.sorted_slot_index,
                p.gate_batch,
                im,
                k_top,
                m_total_max,
                p.swiglu_limit,
            ))?;
        } else {
            hip!(gpu.moe_gate_up_unscatter_k8(
                p.y_gate_up_grouped,
                p.sorted_slot_index,
                p.gate_batch,
                p.up_batch,
                im,
                k_top,
                m_total_max,
            ))?;
            hip!(gpu.deepseek4_silu_mul_clamp_f32_batched(
                p.gate_batch,
                p.up_batch,
                p.gate_batch,
                im,
                batch_size * k_top,
                p.swiglu_limit,
            ))?;
        }

        // FWHT rotate.
        hip!(gpu.rotate_x_mq_batched(p.gate_batch, p.rot_batch, im, batch_size * k_top))?;

        // Grouped down GEMM (M=hidden, K=im, x_row_div=1, rows=B*k_top).
        let use_lloyd_4w_dn = lloyd_4w_base.unwrap_or(arch_4w) && hidden % 64 == 0 && im % 256 == 0;
        let use_mmqload_dn = use_lloyd_4w_dn && mmqload_env;
        let use_nosync_dn = use_mmqload_dn && nosync_env;
        let use_i8_dn = i8_moe && hidden % 16 == 0 && im % 256 == 0;
        let v_dn = select_grouped_lloyd_variant(
            false,
            use_lloyd_4w_dn,
            use_i8_dn,
            n32,
            cnd,
            eightw,
            use_mmqload_dn,
            use_nosync_dn,
        );
        dispatch_grouped_lloyd(
            gpu,
            v_dn,
            p.expert_down_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            p.rot_batch,
            p.y_down_grouped,
            hidden,
            im,
            1,
            m_total_max,
            batch_size * k_top,
        )?;

        // Down-combine: weighted Σ over k_top slots, per (token, m), into ffn_out.
        hip!(gpu.moe_down_combine_grouped_k8(
            p.y_down_grouped,
            p.inverse_perm,
            p.topk_weights,
            p.ffn_out,
            hidden,
            k_top,
            batch_size,
        ))?;
    } else {
        // ── Scalar K4 path (batch_size < gate, or grouped opt-out) ──
        let use_gate_up_k4096_lds = gpu.arch.eq_ignore_ascii_case("gfx1151")
            && 2 * im == 4096
            && hidden == 4096
            && hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_GATE_UP_BATCHED_K4096_LDS")
                .ok()
                .as_deref()
                != Some("0");
        if use_gate_up_k4096_lds {
            hip!(
                gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4096_lds(
                    p.expert_gate_up_ptrs,
                    p.topk_indices,
                    p.x_rot,
                    p.gate_batch,
                    p.up_batch,
                    2 * im,
                    hidden,
                    k_top,
                    batch_size,
                )
            )?;
        } else {
            hip!(
                gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4(
                    p.expert_gate_up_ptrs,
                    p.topk_indices,
                    p.x_rot,
                    p.gate_batch,
                    p.up_batch,
                    2 * im,
                    hidden,
                    k_top,
                    batch_size,
                )
            )?;
        }
        hip!(gpu.deepseek4_silu_mul_clamp_f32_batched(
            p.gate_batch,
            p.up_batch,
            p.gate_batch,
            im,
            batch_size * k_top,
            p.swiglu_limit,
        ))?;
        hip!(gpu.rotate_x_mq_batched(p.gate_batch, p.rot_batch, im, batch_size * k_top))?;

        // Down: deterministic expanded+combine (default; bit-reproducible for
        // spec-decode) vs non-deterministic atomic-accumulate.
        let deterministic = !p.uses_atomic_moe_down
            && hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC").as_deref()
                != Ok("0");
        if deterministic {
            hip!(gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
                p.expert_down_ptrs,
                p.topk_indices,
                p.rot_batch,
                p.down_expert_outputs,
                hidden,
                im,
                k_top,
                batch_size,
            ))?;
            hip!(gpu.moe_down_combine_k8_batched(
                p.down_expert_outputs,
                p.topk_weights,
                p.ffn_out,
                hidden,
                k_top,
                batch_size,
            ))?;
        } else {
            // DeepSeek4's routed down projection has fixed K=2048. The ordinary
            // AR route already uses the K8-all body; use its position-batched
            // twin for gfx1151 verify as well. Keep an explicit rollback switch
            // while this route is certified for retained replay.
            let use_batched_k8all = gpu.arch.eq_ignore_ascii_case("gfx1151")
                && im == 2048
                && hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_DOWN_BATCHED_K8ALL")
                    .as_deref()
                    != Ok("0");
            if use_batched_k8all {
                hip!(gpu
                    .deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched_k8all(
                        p.expert_down_ptrs,
                        p.topk_indices,
                        p.topk_weights,
                        p.rot_batch,
                        p.ffn_out,
                        hidden,
                        im,
                        k_top,
                        batch_size,
                    ))?;
            } else {
                hip!(gpu
                    .deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched_k4(
                        p.expert_down_ptrs,
                        p.topk_indices,
                        p.topk_weights,
                        p.rot_batch,
                        p.ffn_out,
                        hidden,
                        im,
                        k_top,
                        batch_size,
                    ))?;
            }
        }
    }

    Ok(())
}

// ── Qwen3.5 batched MoE prefill (Ship 4.2) ──────────────────────────

/// Dispatch one grouped-GEMM for the given routed expert dtype, as a gate/up
/// (or other non-qt53) projection. qt53 is refused here: it exists only as a
/// routed down, via [`dispatch_grouped_gemm_role`].
#[allow(clippy::too_many_arguments)]
fn dispatch_grouped_gemm(
    gpu: &mut Gpu,
    dtype: DType,
    expert_dtype_tags: Option<&GpuTensor>,
    ptrs: &GpuTensor,
    tile_ids: &GpuTensor,
    sorted_slot_index: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    x_row_div: usize,
    m_total: usize,
    rows: usize,
    force_mq4_fp16: bool,
    paro_i8: bool,
    paro_i8_k8: bool,
    coalesced: bool,
) -> Result<(), DispatchError> {
    dispatch_grouped_gemm_role(
        gpu,
        GroupedGemmRole::GateUp,
        0,
        dtype,
        expert_dtype_tags,
        ptrs,
        tile_ids,
        sorted_slot_index,
        x,
        y,
        m,
        k,
        x_row_div,
        m_total,
        rows,
        force_mq4_fp16,
        paro_i8,
        paro_i8_k8,
        coalesced,
    )
}

/// [`dispatch_grouped_gemm`] with the projection role and the expert count
/// (read only by the qt53 down kernel). qt53 is admitted for `Down` only.
#[allow(clippy::too_many_arguments)]
fn dispatch_grouped_gemm_role(
    gpu: &mut Gpu,
    role: GroupedGemmRole,
    n_exp: usize,
    dtype: DType,
    expert_dtype_tags: Option<&GpuTensor>,
    ptrs: &GpuTensor,
    tile_ids: &GpuTensor,
    sorted_slot_index: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    x_row_div: usize,
    m_total: usize,
    rows: usize,
    force_mq4_fp16: bool,
    paro_i8: bool,
    paro_i8_k8: bool,
    coalesced: bool,
) -> Result<(), DispatchError> {
    reject_fp8_b128(dtype)?;
    if dtype == DType::MQ4G128V2 {
        grouped_gemm_qt53_down_admitted(role, expert_dtype_tags.is_some(), x_row_div, n_exp)?;
        // Grouped k8 scatter ABI (kernels/src/moe_scatter_fused_k8.hip): every
        // expert is padded to `MOE_GROUPED_BLOCK_M` (16) rows, padding slots hold
        // flat slot -1 (:44-59 init, :75 padding) and `expert_tile_ids[tile]`
        // names the tile's expert (:111-112). The Multirow qt53 kernel
        // (gemm_mq4g128v2_moe_grouped_top10_multirow.hip) consumes exactly that:
        // :39 `slot_start = tile * 16`, :42 `expert_tile_ids[tile]`, :55-59
        // grouped slot -> `sorted_slot_index` flat (skipped when -1) ->
        // `x_row = flat / x_row_div`, F32 X/Y (:27-28), unweighted F32 grouped
        // rows (:116) that the existing `moe_down_combine_grouped_k8` folds.
        // A3B m=2048/k=512 is neither the 2560/640 WMMA / twin / O8R16
        // specialization (rdna-compute moe.rs `gemm_mq4g128v2_moe_grouped_top10`),
        // so it runs Multirow.
        return hip!(gpu.gemm_mq4g128v2_moe_grouped_top10(
            ptrs,
            tile_ids,
            sorted_slot_index,
            x,
            y,
            m,
            k,
            1,
            m_total,
            rows,
            n_exp,
        ));
    }
    reject_mq4g128v2(dtype, "moe")?;
    // Mixed per-expert: the merged grouped kernel carries the per-expert stride
    // via the dtype_tags table; takes priority over the uniform dtype dispatch.
    if let Some(tags) = expert_dtype_tags {
        if coalesced {
            return hip!(gpu.gemm_mixed_moe_grouped_wmma_coalesced(
                ptrs,
                tags,
                tile_ids,
                sorted_slot_index,
                x,
                y,
                m,
                k,
                x_row_div,
                m_total,
                rows,
            ));
        }
        return hip!(gpu.gemm_mixed_moe_grouped_wmma(
            ptrs,
            tags,
            tile_ids,
            sorted_slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total,
            rows,
        ));
    }
    match dtype {
        DType::MQ4G256 => {
            if force_mq4_fp16 {
                hip!(gpu.gemm_hfq4g256_moe_grouped_wmma_k2_fp16(
                    ptrs,
                    tile_ids,
                    sorted_slot_index,
                    x,
                    y,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    rows,
                ))
            } else {
                hip!(gpu.gemm_hfq4g256_moe_grouped_wmma_k2(
                    ptrs,
                    tile_ids,
                    sorted_slot_index,
                    x,
                    y,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    rows,
                ))
            }
        }
        // DType::MQ5G256: grouped-WMMA path is gfx12-only and the kernel
        // (`gemm_hfq5g256_moe_grouped_wmma`) is not yet wired in rdna-compute.
        // MQ5 falls through to `_other => UnsupportedVariant`; on gfx942 the
        // `mq5_on_non_gfx12` guard forces Path 1 so this is never reached.
        DType::MQ6G256 => hip!(gpu.gemm_hfq6g256_moe_grouped_wmma(
            ptrs,
            tile_ids,
            sorted_slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total,
            rows,
        )),
        // mfp4-E8 grouped-WMMA (gfx1151 + gfx12/RDNA4; MoePrefillResolution admits
        // Path 2 for E8 on gfx1151 and gfx1200/gfx1201). The launcher selects the
        // correct WMMA intrinsic variant (gfx1151 vs _gfx12) internally.
        // Amortizes expert-weight reads vs the indexed GEMV — the memory-bound
        // prefill / batched-verify lever.
        DType::MFP4G32E8 => hip!(gpu.gemm_mfp4g32_e8_moe_grouped_wmma(
            ptrs,
            tile_ids,
            sorted_slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total,
            rows,
        )),
        DType::ParoQ4G128 => {
            if paro_i8_k8 {
                hip!(gpu.gemm_paro_q4g128_moe_grouped_mmq_k8_gfx1151(
                    ptrs,
                    tile_ids,
                    sorted_slot_index,
                    x,
                    y,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    rows,
                ))
            } else if paro_i8 {
                hip!(gpu.gemm_paro_q4g128_moe_grouped_mmq_gfx1151(
                    ptrs,
                    tile_ids,
                    sorted_slot_index,
                    x,
                    y,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    rows,
                ))
            } else {
                hip!(gpu.gemm_paro_q4g128_moe_grouped_wmma_k2(
                    ptrs,
                    tile_ids,
                    sorted_slot_index,
                    x,
                    y,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    rows,
                ))
            }
        }
        // Uniform-per-projection codebook routed experts (the antirez asymmetric
        // recipe: gate_up = MQ2-Lloyd 72 B/group, down = MQ3-Lloyd 112 B/group).
        // Both entries are arch-selecting (gfx11 `_k2` / gfx12 `_gfx12`) — do NOT
        // swap either for the bare `_k2` launcher, which fails the JIT on RDNA4.
        DType::MQ2G256Lloyd | DType::MQ2G256LloydU => {
            hip!(gpu.gemm_mq2g256_lloyd_moe_grouped_wmma(
                ptrs,
                tile_ids,
                sorted_slot_index,
                x,
                y,
                m,
                k,
                x_row_div,
                m_total,
                rows,
            ))
        }
        DType::MQ3G256Lloyd => hip!(gpu.gemm_mq3g256_lloyd_moe_grouped_wmma(
            ptrs,
            tile_ids,
            sorted_slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total,
            rows,
        )),
        // qt44. Routed experts are ~99% of an A3B MoE's tensors, so without
        // this arm a qt44 MoE model cannot prefill at all — it falls into the
        // `_other` error below. Arch-selecting (gfx11 `_k2` / gfx12 `_gfx12`)
        // like the MQ2/MQ3-Lloyd sisters — do NOT swap it for the bare `_k2`
        // launcher, which fails the JIT on RDNA4.
        DType::MQ4G256V2 => hip!(gpu.gemm_mq4g256v2_moe_grouped_wmma_k2(
            ptrs,
            tile_ids,
            sorted_slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total,
            rows,
        )),
        // qt47. Same as qt44 — without this arm a pure-MQ6V2 MoE cannot Path-2
        // prefill. Arch-selecting launcher (gfx11 `_k2` / gfx12 `_gfx12`).
        DType::MQ6G256V2 => hip!(gpu.gemm_mq6g256v2_moe_grouped_wmma_k2(
            ptrs,
            tile_ids,
            sorted_slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total,
            rows,
        )),
        _other => Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "prefill-grouped-gemm-dtype",
            arch: "",
            quant: "other",
        }),
    }
}

pub fn dispatch_fused(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    key: KernelKey,
    params: &PipelineParams,
) -> Result<(), DispatchError> {
    let PipelineParams::Linear(params) = params;
    match key {
        KernelKey::GemvMfp4G32Fused => {
            gpu.ensure_mq_signs()
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            let x_rot = unsafe {
                GpuTensor {
                    buf: gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias(),
                    shape: vec![params.k],
                    dtype: rdna_compute::DType::F32,
                }
            };
            hip!(gpu.gemv_mfp4g32_with_rotate(
                params.buf, params.x, params.y, &x_rot, params.m, params.k,
            ))
        }
        _ => Err(DispatchError::UnsupportedVariant {
            family: "pipeline_fused",
            variant: "unknown",
            arch: "",
            quant: "",
        }),
    }
}

#[cfg(test)]
mod mixed_dispatch_tests {
    use super::{
        build_contiguous_permutation, decode_expanded_down_kind, decode_gate_uses_mixed,
        gate_down_skips_rotation, gate_up_varies, grouped_gemm_kind, mixed_expert_dtype_tag,
        ninepath_d3_family, ninepath_d4_family, prefill_path1_down_kind,
        prefill_path1_down_kind_tag_aware, prefill_path1_gate_up_kind,
        prefill_path1_gate_up_kind_tag_aware, shared_dense_down_kind, use_gfx1151_i8_moe,
    };
    use crate::families::moe_buckets::bucket_topk_by_tier;
    use rdna_compute::DType::*;

    #[test]
    fn deepseek4_i8_moe_is_exact_gfx1151_only() {
        assert!(use_gfx1151_i8_moe("gfx1151"));
        for arch in ["gfx1100", "gfx1150", "gfx1152", "gfx1200", "gfx1201"] {
            assert!(!use_gfx1151_i8_moe(arch), "unexpected i8 route on {arch}");
        }
    }

    /// EQUIVALENCE INVARIANT (host half): an all-ONE-tier table yields the
    /// IDENTITY permutation and a single full-width range. This is the
    /// host-side proof that the mixed path emits the same per-rank kernel
    /// addressing as the uniform path for a uniform table — the device half
    /// (bit-identical `down_expanded`) is the GPU-deferred gate below.
    #[test]
    fn all_one_tier_is_identity_permutation() {
        let topk = [3u32, 7, 1, 5, 0, 2, 6, 4];
        let tier_of = vec![MQ4G256; 8];
        let buckets = bucket_topk_by_tier(&topk, &tier_of).unwrap();
        assert_eq!(buckets.len(), 1, "uniform table ⇒ one bucket");
        let (perm, ranges) = build_contiguous_permutation(&buckets, 8);
        assert_eq!(perm, (0..8).collect::<Vec<_>>(), "identity perm");
        assert_eq!(ranges, vec![(0, 8)], "single full-width range");
    }

    /// A mixed table groups each tier into a contiguous range; `perm` is a
    /// bijection over 0..k and `ranges` tile [0, k) with no gaps/overlap.
    #[test]
    fn mixed_table_is_contiguous_partition() {
        // experts: even→MQ4, odd→MQ6. top-k interleaves tiers.
        let tier_of = vec![MQ4G256, MQ6G256, MQ4G256, MQ6G256, MQ4G256, MQ6G256];
        let topk = [1u32, 0, 3, 2, 5, 4]; // ranks 0..5; tiers MQ6,MQ4,MQ6,MQ4,MQ6,MQ4
        let buckets = bucket_topk_by_tier(&topk, &tier_of).unwrap();
        assert_eq!(buckets.len(), 2);
        let (perm, ranges) = build_contiguous_permutation(&buckets, 6);

        // perm is a bijection over 0..6.
        let mut seen = perm.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..6).collect::<Vec<_>>());

        // ranges tile [0,6) contiguously, summing to k.
        let total: usize = ranges.iter().map(|&(_, n)| n).sum();
        assert_eq!(total, 6);
        let mut cursor = 0;
        for &(lo, n) in &ranges {
            assert_eq!(lo, cursor, "ranges must be gap-free & contiguous");
            cursor += n;
        }
        assert_eq!(cursor, 6);

        // Within each range, perm lists exactly that bucket's original ranks.
        for (bi, b) in buckets.iter().enumerate() {
            let (lo, n) = ranges[bi];
            assert_eq!(&perm[lo..lo + n], b.ranks.as_slice());
        }
    }

    // ── MQV2 MoE wiring predicates (decode / prefill / shared / mixed tags) ──

    #[test]
    fn ninepath_d4_separates_v1_and_v2_families() {
        assert_eq!(ninepath_d4_family(MQ4G256, MQ4G256), Some("hfq4"));
        assert_eq!(ninepath_d4_family(MQ4G256V2, MQ4G256V2), Some("mq4v2"));
        assert_eq!(ninepath_d4_family(MQ6G256V2, MQ6G256V2), Some("mq6v2"));
        // Split V1/V2 pairs never ride a ninepath family.
        assert_eq!(ninepath_d4_family(MQ4G256V2, MQ4G256), None);
        assert_eq!(ninepath_d4_family(MQ4G256, MQ4G256V2), None);
        assert_eq!(ninepath_d4_family(MQ6G256V2, MQ6G256), None);
        assert_eq!(ninepath_d4_family(MQ4G256V2, MQ6G256V2), None);
    }

    #[test]
    fn decode_expanded_down_never_aliases_v2_to_v1() {
        assert_eq!(decode_expanded_down_kind(MQ4G256), Some("hfq4"));
        assert_eq!(decode_expanded_down_kind(MQ4G256V2), Some("mq4v2"));
        assert_eq!(decode_expanded_down_kind(MQ6G256), Some("hfq6"));
        assert_eq!(decode_expanded_down_kind(MQ6G256V2), Some("mq6v2"));
        // V2 kinds are distinct string identities from V1.
        assert_ne!(
            decode_expanded_down_kind(MQ4G256V2),
            decode_expanded_down_kind(MQ4G256)
        );
        assert_ne!(
            decode_expanded_down_kind(MQ6G256V2),
            decode_expanded_down_kind(MQ6G256)
        );
    }

    #[test]
    fn path1_batched_admits_mq4v2_and_mq6v2() {
        assert_eq!(prefill_path1_gate_up_kind(MQ4G256V2), Some("mq4v2"));
        assert_eq!(prefill_path1_gate_up_kind(MQ6G256V2), Some("mq6v2"));
        assert_eq!(prefill_path1_down_kind(MQ4G256V2), Some("mq4v2"));
        assert_eq!(prefill_path1_down_kind(MQ6G256V2), Some("mq6v2"));
        // V1 remains on its own kinds.
        assert_eq!(prefill_path1_gate_up_kind(MQ4G256), Some("hfq4"));
        assert_eq!(prefill_path1_gate_up_kind(MQ6G256), Some("hfq6"));
    }

    #[test]
    fn grouped_gemm_admits_mq6v2() {
        assert_eq!(grouped_gemm_kind(MQ4G256V2), Some("mq4v2"));
        assert_eq!(grouped_gemm_kind(MQ6G256V2), Some("mq6v2"));
        assert_eq!(grouped_gemm_kind(MQ4G256), Some("hfq4"));
        assert_eq!(grouped_gemm_kind(MQ6G256), Some("hfq6"));
        assert_ne!(grouped_gemm_kind(MQ6G256V2), grouped_gemm_kind(MQ6G256));
    }

    #[test]
    fn shared_dense_v2_never_uses_v1_sigmoid_scaled() {
        assert_eq!(shared_dense_down_kind(MQ4G256), Some("hfq4_sigmoid_scaled"));
        assert_eq!(shared_dense_down_kind(MQ4G256V2), Some("mq4v2_prerotated"));
        assert_eq!(shared_dense_down_kind(MQ6G256V2), Some("mq6v2_prerotated"));
        assert_ne!(
            shared_dense_down_kind(MQ4G256V2),
            shared_dense_down_kind(MQ4G256)
        );
    }

    #[test]
    fn frozen_mixed_tags_7_to_18_are_v2_identities() {
        // Uniform V2 pairs.
        assert_eq!(mixed_expert_dtype_tag(MQ4G256V2, MQ4G256V2), Some(7));
        assert_eq!(mixed_expert_dtype_tag(MQ6G256V2, MQ6G256V2), Some(8));
        // Split V2/V1 and V1/V2 pairs occupy 9..18 and NEVER collapse to 0..6.
        assert_eq!(mixed_expert_dtype_tag(MQ4G256V2, MQ6G256), Some(9));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256V2, MQ2G256Lloyd), Some(10));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256V2, MQ4G256), Some(11));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256, MQ4G256V2), Some(12));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256V2, MQ3G256Lloyd), Some(13));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256V2, MFP4G32E8), Some(14));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256V2, MFP3G32E8), Some(15));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256V2, MFP2G32E8), Some(16));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256V2, MQ6G256V2), Some(17));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256, MQ6G256V2), Some(18));
        // V1 pairs keep 0..6.
        assert_eq!(mixed_expert_dtype_tag(MQ4G256, MQ4G256), Some(2));
        assert_eq!(mixed_expert_dtype_tag(MQ4G256, MQ6G256), Some(0));
        // Unknown / GL pairs refuse loudly (None).
        assert_eq!(mixed_expert_dtype_tag(MQ6G256V2, MQ4G256V2), None);
        assert_eq!(mixed_expert_dtype_tag(MQ2G256GL, MQ4G256), None);
        assert_eq!(mixed_expert_dtype_tag(MQ4G256, MQ2G256GL), None);
        // No V2 pair may share a tag with a V1 pair.
        for tag in 7u8..=18 {
            let v1_tags: [u8; 7] = [0, 1, 2, 3, 4, 5, 6];
            assert!(!v1_tags.contains(&tag));
        }
    }

    #[test]
    fn unrotated_dtype_still_skips_rotation() {
        assert!(gate_down_skips_rotation(MQ2G256LloydU));
        assert!(!gate_down_skips_rotation(MQ4G256V2));
        assert!(!gate_down_skips_rotation(MQ6G256V2));
    }

    #[test]
    fn ninepath_d3_is_hfq4_only_and_v2_uses_native_d4() {
        // Only HFQ4/MQ4V1 (MQ4G256) may call the V1 ninepath D3 kernel.
        assert_eq!(ninepath_d3_family(MQ4G256, MQ4G256), Some("hfq4"));
        // V2 pairs must NOT use D3 — they use native indexed gate + V2 D4.
        assert_eq!(ninepath_d3_family(MQ4G256V2, MQ4G256V2), None);
        assert_eq!(ninepath_d3_family(MQ6G256V2, MQ6G256V2), None);
        // Split V1/V2 pairs never share D3.
        assert_eq!(ninepath_d3_family(MQ4G256V2, MQ4G256), None);
        assert_eq!(ninepath_d3_family(MQ4G256, MQ4G256V2), None);
        assert_eq!(ninepath_d3_family(MQ4G256V2, MQ6G256V2), None);
        // D4 still separates V2 families correctly.
        assert_eq!(ninepath_d4_family(MQ4G256V2, MQ4G256V2), Some("mq4v2"));
        assert_eq!(ninepath_d4_family(MQ6G256V2, MQ6G256V2), Some("mq6v2"));
    }

    #[test]
    fn gate_up_varies_is_exact_dtype_equality() {
        // None => uniform (no tags).
        assert!(!gate_up_varies(None));
        // All-equal Some => uniform.
        assert!(!gate_up_varies(Some(&[MQ4G256, MQ4G256, MQ4G256])));
        assert!(!gate_up_varies(Some(&[MQ4G256V2, MQ4G256V2])));
        // Exact DType variation => true (V1 vs V2 is a variation).
        assert!(gate_up_varies(Some(&[MQ4G256, MQ4G256V2])));
        assert!(gate_up_varies(Some(&[MQ4G256V2, MQ6G256V2])));
        assert!(gate_up_varies(Some(&[MQ4G256, MQ6G256])));
        // Single element => uniform.
        assert!(!gate_up_varies(Some(&[MQ6G256V2])));
    }

    #[test]
    fn mixed_gate_precedence_before_representative_v1_v2_arms() {
        // Whenever gate_up exact dtype varies, mixed gate must be chosen
        // before representative MQ4V2/MQ6V2/V1 arms. Uniform shortcut only
        // when exact equality.
        // Gate varies + tags => mixed.
        assert!(decode_gate_uses_mixed(true, true));
        // Gate uniform + tags => uniform (down-only-graded optimization).
        assert!(!decode_gate_uses_mixed(true, false));
        // No tags => never mixed, even if varies flag somehow true.
        assert!(!decode_gate_uses_mixed(false, true));
        assert!(!decode_gate_uses_mixed(false, false));
        // Prefill Path1 tag-aware gate mirrors the same precedence.
        assert_eq!(
            prefill_path1_gate_up_kind_tag_aware(MQ4G256V2, true, true),
            Some("mixed")
        );
        assert_eq!(
            prefill_path1_gate_up_kind_tag_aware(MQ4G256, true, true),
            Some("mixed")
        );
        // Uniform gate (exact equality) uses representative kind, not mixed.
        assert_eq!(
            prefill_path1_gate_up_kind_tag_aware(MQ4G256, true, false),
            Some("hfq4")
        );
        assert_eq!(
            prefill_path1_gate_up_kind_tag_aware(MQ4G256V2, true, false),
            Some("mq4v2")
        );
        assert_eq!(
            prefill_path1_gate_up_kind_tag_aware(MQ6G256V2, true, false),
            Some("mq6v2")
        );
        // No tags => uniform regardless of varies flag.
        assert_eq!(
            prefill_path1_gate_up_kind_tag_aware(MQ4G256, false, false),
            Some("hfq4")
        );
        assert_eq!(
            prefill_path1_gate_up_kind_tag_aware(MQ4G256V2, false, false),
            Some("mq4v2")
        );
    }

    #[test]
    fn mixed_down_path1_tag_aware_uses_mixed_when_tagged() {
        // Mixed Path1 down: when tags exist, use mixed batched down launcher;
        // do not representative-dispatch tagged layers.
        assert_eq!(
            prefill_path1_down_kind_tag_aware(MQ4G256, true),
            Some("mixed")
        );
        assert_eq!(
            prefill_path1_down_kind_tag_aware(MQ4G256V2, true),
            Some("mixed")
        );
        assert_eq!(
            prefill_path1_down_kind_tag_aware(MQ6G256V2, true),
            Some("mixed")
        );
        // No tags => uniform native kind or loud error (None) is delegated.
        assert_eq!(
            prefill_path1_down_kind_tag_aware(MQ4G256, false),
            Some("hfq4")
        );
        assert_eq!(
            prefill_path1_down_kind_tag_aware(MQ4G256V2, false),
            Some("mq4v2")
        );
        assert_eq!(
            prefill_path1_down_kind_tag_aware(MQ6G256V2, false),
            Some("mq6v2")
        );
        // Unknown dtype without tags should be None (loud reject).
        assert_eq!(prefill_path1_down_kind_tag_aware(MQ2G256Lloyd, false), None);
        // Tagged layer with unsupported pair still returns mixed (the mixed
        // kernel itself will reject loudly per tag table) — never collapses
        // to representative V1.
        assert_eq!(
            prefill_path1_down_kind_tag_aware(MQ2G256Lloyd, true),
            Some("mixed")
        );
    }

    #[test]
    fn grouped_path2_preserves_tags_on_v1_v2_variation() {
        // Grouped Path2 must preserve tags when gate_up exact dtype varies
        // (V1/V2 graded). Uniform shortcut only with exact DType equality.
        // Simulate the dispatch decision: tags preserved iff gate_up_varies.
        let uniform_mq4 = Some(vec![MQ4G256, MQ4G256, MQ4G256]);
        let uniform_v2 = Some(vec![MQ4G256V2, MQ4G256V2]);
        let mixed_v1_v2 = Some(vec![MQ4G256, MQ4G256V2]);
        let mixed_mq4_mq6 = Some(vec![MQ4G256V2, MQ6G256V2]);
        assert!(!gate_up_varies(uniform_mq4.as_deref()));
        assert!(!gate_up_varies(uniform_v2.as_deref()));
        assert!(gate_up_varies(mixed_v1_v2.as_deref()));
        assert!(gate_up_varies(mixed_mq4_mq6.as_deref()));
        // Tags preserved only when varies.
        let has_tags = true;
        assert!(!decode_gate_uses_mixed(
            has_tags,
            gate_up_varies(uniform_mq4.as_deref())
        ));
        assert!(decode_gate_uses_mixed(
            has_tags,
            gate_up_varies(mixed_v1_v2.as_deref())
        ));
        assert!(decode_gate_uses_mixed(
            has_tags,
            gate_up_varies(mixed_mq4_mq6.as_deref())
        ));
    }

    // ── [GPU — DEFERRED] bucketing-equivalence numeric gate ─────────────────
    //
    // NOTE: deferred — GPU under embargo. Cannot run; left as an executable
    // stub so a future GPU session has the exact contract.
    //
    // WHY MULTI-BUCKET (not uniform/identity): the uniform table produces the
    // IDENTITY permutation — a single bucket with lo=0, n=k — so every per-rank
    // sub-view is the full buffer and grid.y = k. That case CANNOT expose the
    // class of bug this gate guards against (gate_up grid.y must equal the
    // bucket's rank count `n`, not a hardwired 8; OOB only manifests for a
    // bucket with lo>0 and/or n<8). So the gate MUST use a real ≥2-tier table.
    //
    // WHAT IT MUST VERIFY: build a real qwen35/lfm2moe MoE decode layer with a
    // GENUINELY MIXED ≥2-tier per-expert table — e.g. n_exp=8 experts split as
    //   5 × MQ4G256  +  3 × MQ6G256
    // with top-k routing (k=8) chosen so that BOTH tiers are selected. After
    // `bucket_topk_by_tier` + `build_contiguous_permutation` this yields at
    // least two contiguous buckets, e.g. ranges [(0, n0), (n0, n1)] with
    //   - a non-first bucket whose base offset lo = n0 > 0, and
    //   - at least one bucket with n < 8,
    // which is EXACTLY the OOB-trigger geometry (gate_up over a `lo`-based
    // sub-view, grid.y = n). Run it TWICE on identical inputs:
    //   (1) MIXED path: per_expert_gate_up/down = Some(<the mixed table>)
    //                   (mixed = true; real bucketing exercised).
    //   (2) REFERENCE  : a per-rank reference that runs each selected expert's
    //                    gate_up→silu·mul·rotate→down→combine for its OWN tier
    //                    in natural (unpermuted) rank order — i.e. the
    //                    mathematically-correct mixed result with no bucketing.
    // ASSERT the `down_expanded` slots (compared rank-for-rank under the
    // permutation) and the final residual `out_target` match within tight fp
    // tolerance (bit-identical if the reference replays the same kernels). This
    // proves the permute-to-contiguous + per-tier-sub-view decomposition is
    // exact for a TRUE multi-bucket layout. Run on BOTH dispatch sites once the
    // ds4 bias-aware/hash sites gain the same bucket loop (two-dispatch-site
    // gotcha).
    #[test]
    #[ignore = "GPU-deferred: requires device; see NOTE — multi-bucket (5×MQ4 + 3×MQ6) bucketing-equivalence numeric gate"]
    fn mixed_dispatch_bucketing_equivalence() {
        // Intentionally empty under the GPU embargo. A GPU session must build
        // the MIXED ≥2-tier MoeParams described in the NOTE above (5×MQ4G256 +
        // 3×MQ6G256, routing that selects both tiers so ≥1 bucket has lo>0 and
        // ≥1 has n<8), call run_moe_decode, download down_expanded + out_target,
        // and assert equality against the per-rank mixed reference.
    }
}
fn prefill_shared_gemm_key(dtype: DType) -> Option<KernelKey> {
    Some(match dtype {
        DType::Q8_0 => KernelKey::GemmQ8_0BatchedChunked,
        DType::F32 => KernelKey::GemmF32Batched,
        DType::MQ4G256 => KernelKey::GemmHfq4G256,
        DType::MQ4G256V2 => KernelKey::GemmMq4G256V2,
        DType::MQ6G256 => return None,
        DType::MQ6G256V2 => KernelKey::GemmMq6G256V2,
        DType::HFQ4G128 | DType::ParoQ4G128 => KernelKey::GemmHfq4G128,
        _ => return None,
    })
}

/// `HIPFIRE_MOE_SHARED_PREPARED_IU4` (`developer.moe_shared_prepared_iu4`, a
/// typed boolean): the explicit value, read once. `true`/`1` arms the shared
/// selector and gate/up reading one IU4-quantized activation, prepared once,
/// wherever the GPU capability admits it; `false`/`0` is the kill switch.
/// Unset follows `prefill_route_capabilities` qualification of the device
/// architecture.
static SHARED_PREPARED_IU4: LazyLock<Option<bool>> =
    LazyLock::new(|| hipfire_config::developer_typed_bool("HIPFIRE_MOE_SHARED_PREPARED_IU4"));

/// Pure admission of the shared-expert quantize-once IU4 route: flag on, the
/// GPU capability (arch, IU4 prefill, fp8/a8 precedence), no replay recording,
/// no graph capture, all three projections MQ4G256V2 with one nonzero K
/// divisible by 256, and at least 64 rows.
fn shared_prepared_iu4_admitted(
    flag: bool,
    gpu_capable: bool,
    recording: bool,
    capture: bool,
    dtypes: [DType; 3],
    ks: [usize; 3],
    rows: usize,
) -> bool {
    flag && gpu_capable
        && !recording
        && !capture
        && dtypes.iter().all(|d| *d == DType::MQ4G256V2)
        && ks[0] != 0
        && ks[0] % 256 == 0
        && ks.iter().all(|k| *k == ks[0])
        && rows >= 64
}

/// Whether the shared selector/gate/up of `p` take the quantize-once IU4 route
/// on `gpu`: the flag, the GPU capability (replay recording and graph capture
/// included), all three projections MQ4G256V2 with one nonzero K divisible by
/// 256, and at least 64 rows. Pure over the current device state; the router
/// stage uses it (plus its own dtype/K/rotation check) to decide whether to
/// quantize `p.x_rot_batch` once for the router and carry the token here.
pub(super) fn prefill_shared_prepared_iu4_admitted(
    gpu: &Gpu,
    p: &crate::families::moe::MoePrefillParams<'_>,
) -> bool {
    use rdna_compute::prefill_route_capabilities::{enabled, PrefillRoute};
    let flag = enabled(
        *SHARED_PREPARED_IU4,
        &gpu.arch,
        PrefillRoute::MoeSharedPreparedIu4,
    );
    if !flag {
        return false;
    }
    let Some(shared) = p.prelude.shared.as_ref() else {
        return false;
    };
    let w = &shared.weights;
    let dtypes = [w.selector.dtype, w.gate.dtype, w.up.dtype];
    let ks = [w.selector.k, w.gate.k, w.up.k];
    shared_prepared_iu4_admitted(
        flag,
        gpu.mq4g256v2_shared_iu4_supported(ks[0], p.batch_size),
        gpu.replay.is_recording(),
        gpu.graphs.capture_mode,
        dtypes,
        ks,
        p.batch_size,
    )
}

/// `prepared` is the IU4 token the router stage quantized from
/// `p.x_rot_batch` (`Some` only when it admitted through
/// [`prefill_shared_prepared_iu4_admitted`]); it is consumed here without a
/// second quantize. `Some` with the route not admitted, or declined by the GPU,
/// fails closed instead of running the incumbent. The calibration taps run
/// inside the grouped GPU helper in incumbent order.
pub(super) fn prefill_shared_gate_up_stage(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    p: &crate::families::moe::MoePrefillParams<'_>,
    prepared: Option<&rdna_compute::scratch::Int4MmqPrepared>,
) -> Result<(), DispatchError> {
    reject_fp8_moe_unless_routed(&p.dtypes, p.route_policy.map(|policy| policy.capability))?;
    let shared =
        p.prelude.shared.as_ref().ok_or_else(|| {
            DispatchError::Hip("shared gate/up stage requires shared weights".into())
        })?;
    let n = p.batch_size;
    if prefill_shared_prepared_iu4_admitted(gpu, p) {
        let (selector, gate, up) = (
            &shared.weights.selector,
            &shared.weights.gate,
            &shared.weights.up,
        );
        let k = selector.k;
        let launched = gpu
            .try_gemm_mq4g256v2_shared_iu4(
                [
                    (selector.buf, shared.scalar, selector.m),
                    (gate.buf, shared.gate_out, gate.m),
                    (up.buf, shared.up_out, up.m),
                ],
                p.x_rot_batch,
                k,
                n,
                prepared,
            )
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
        if !launched {
            // Admission already required the helper's own capability, so this
            // is unreachable; fail closed rather than re-tap and re-run the
            // incumbent.
            return Err(DispatchError::Hip(
                "shared gate/up prepared IU4 admitted but helper declined".into(),
            ));
        }
        return Ok(());
    }
    if prepared.is_some() {
        return Err(DispatchError::Hip(
            "shared gate/up received a prepared IU4 token but the route is not admitted".into(),
        ));
    }
    let selector_x = match shared.weights.selector.dtype {
        DType::Q8_0 | DType::F32 => p.x_norm_batch,
        _ => p.x_rot_batch,
    };
    let gemm = GemmFamily::new();
    let selector_key = prefill_shared_gemm_key(shared.weights.selector.dtype).ok_or(
        DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "prefill-shared-selector-dtype",
            arch: "",
            quant: "",
        },
    )?;
    gemm.run_key(
        selector_key,
        ctx,
        gpu,
        &GemmParams {
            w: &shared.weights.selector,
            x: selector_x,
            y: shared.scalar,
            batch_size: n,
        },
    )?;

    let gate = &shared.weights.gate;
    let up = &shared.weights.up;
    let x = match gate.dtype {
        DType::Q8_0 | DType::F32 => p.x_norm_batch,
        _ => p.x_rot_batch,
    };
    match gate.dtype {
        DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2 => {
            let fused = FusedQkvFamily::new();
            let key = fused_gate_up_key_for(gate.dtype);
            let weights = [gate.buf, up.buf];
            let outputs = [shared.gate_out, shared.up_out];
            let m = [gate.m, up.m];
            fused.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &weights,
                    x,
                    outputs: &outputs,
                    m: &m,
                    k: gate.k,
                    rot_scratch: &[],
                    batch_size: Some(n),
                },
            )
        }
        DType::ParoQ4G128 => {
            let gate_rot = gate.rotation.as_ref().ok_or_else(|| {
                DispatchError::Hip("shared Paro gate has no rotation metadata".into())
            })?;
            let up_rot = up.rotation.as_ref().ok_or_else(|| {
                DispatchError::Hip("shared Paro up has no rotation metadata".into())
            })?;
            gpu.givens_rotate_to(
                p.x_norm_batch,
                p.x_rot_batch,
                gate_rot.pairs,
                gate_rot.theta,
                gate_rot.scales,
                n,
                gate.k,
                gate_rot.krot,
                128,
            )
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
            let gate_key = prefill_shared_gemm_key(gate.dtype)
                .ok_or_else(|| DispatchError::Hip("shared Paro gate has no GEMM key".into()))?;
            gemm.run_key(
                gate_key,
                ctx,
                gpu,
                &GemmParams {
                    w: gate,
                    x: p.x_rot_batch,
                    y: shared.gate_out,
                    batch_size: n,
                },
            )?;
            gpu.givens_rotate_to(
                p.x_norm_batch,
                p.x_rot_batch,
                up_rot.pairs,
                up_rot.theta,
                up_rot.scales,
                n,
                up.k,
                up_rot.krot,
                128,
            )
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
            gemm.run_key(
                prefill_shared_gemm_key(up.dtype)
                    .ok_or_else(|| DispatchError::Hip("shared Paro up has no GEMM key".into()))?,
                ctx,
                gpu,
                &GemmParams {
                    w: up,
                    x: p.x_rot_batch,
                    y: shared.up_out,
                    batch_size: n,
                },
            )
        }
        DType::Q8_0 | DType::F32 => {
            for (weight, out) in [(gate, shared.gate_out), (up, shared.up_out)] {
                gemm.run_key(
                    prefill_shared_gemm_key(weight.dtype).ok_or_else(|| {
                        DispatchError::Hip("shared dense gate/up has no GEMM key".into())
                    })?,
                    ctx,
                    gpu,
                    &GemmParams {
                        w: weight,
                        x,
                        y: out,
                        batch_size: n,
                    },
                )?;
            }
            Ok(())
        }
        DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8 => {
            let gate_f16 = gpu
                .alloc_tensor(&[gate.m * gate.k], DType::F16)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            gpu.dequantize_mfp4g32_e8_to_f16(&gate.buf.buf, &gate_f16.buf, gate.m, gate.k)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            let up_f16 = gpu
                .alloc_tensor(&[up.m * up.k], DType::F16)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            gpu.dequantize_mfp4g32_e8_to_f16(&up.buf.buf, &up_f16.buf, up.m, up.k)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            let gate_ref = WeightRef {
                buf: &gate_f16,
                dtype: DType::F16,
                m: gate.m,
                k: gate.k,
                row_stride: gate.k,
                rotation: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            };
            let up_ref = WeightRef {
                buf: &up_f16,
                dtype: DType::F16,
                m: up.m,
                k: up.k,
                row_stride: up.k,
                rotation: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            };
            let key = KernelKey::GemmF16WmmaMb8;
            let result = gemm
                .run_key(
                    key,
                    ctx,
                    gpu,
                    &GemmParams {
                        w: &gate_ref,
                        x: p.x_rot_batch,
                        y: shared.gate_out,
                        batch_size: n,
                    },
                )
                .and_then(|_| {
                    gemm.run_key(
                        key,
                        ctx,
                        gpu,
                        &GemmParams {
                            w: &up_ref,
                            x: p.x_rot_batch,
                            y: shared.up_out,
                            batch_size: n,
                        },
                    )
                });
            gpu.free_tensor(gate_f16)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            gpu.free_tensor(up_f16)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            result
        }
        _ => Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "prefill-shared-gate-up-dtype",
            arch: "",
            quant: "",
        }),
    }
}

#[cfg(test)]
mod shared_prepared_iu4_tests {
    use super::*;

    const MQ: DType = DType::MQ4G256V2;

    fn admitted(rows: usize, ks: [usize; 3], dtypes: [DType; 3]) -> bool {
        shared_prepared_iu4_admitted(true, true, false, false, dtypes, ks, rows)
    }

    #[test]
    fn admits_the_exact_incumbent_shape() {
        for rows in [64, 65, 512, 4096] {
            assert!(admitted(rows, [2048; 3], [MQ; 3]), "rows {rows}");
        }
        assert!(admitted(64, [256; 3], [MQ; 3]));
    }

    #[test]
    fn flag_capability_recording_and_capture_truth_table() {
        for bits in 0..16u8 {
            let (flag, cap, rec, graph) =
                (bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0);
            for rows in [63, 64, 512, 4096] {
                let want = flag && cap && !rec && !graph && rows >= 64;
                assert_eq!(
                    shared_prepared_iu4_admitted(flag, cap, rec, graph, [MQ; 3], [2048; 3], rows),
                    want,
                    "flag {flag} cap {cap} rec {rec} graph {graph} rows {rows}"
                );
            }
        }
    }

    #[test]
    fn batch_threshold_is_sixty_four_rows() {
        assert!(!admitted(0, [2048; 3], [MQ; 3]));
        assert!(!admitted(1, [2048; 3], [MQ; 3]));
        assert!(!admitted(63, [2048; 3], [MQ; 3]));
        assert!(admitted(64, [2048; 3], [MQ; 3]));
        assert!(admitted(512, [2048; 3], [MQ; 3]));
        assert!(admitted(4096, [2048; 3], [MQ; 3]));
    }

    #[test]
    fn any_non_mq4g256v2_dtype_in_any_slot_rejects() {
        for other in [DType::BF16, DType::F32, DType::Q8_0, DType::F16] {
            for slot in 0..3 {
                let mut dtypes = [MQ; 3];
                dtypes[slot] = other;
                assert!(
                    !admitted(512, [2048; 3], dtypes),
                    "{other:?} in slot {slot}"
                );
            }
            assert!(!admitted(512, [2048; 3], [other; 3]), "{other:?} x3");
        }
    }

    #[test]
    fn k_mismatch_non_multiple_of_256_and_zero_reject() {
        for slot in 0..3 {
            let mut ks = [2048; 3];
            ks[slot] = 2304;
            assert!(!admitted(512, ks, [MQ; 3]), "mismatch in slot {slot}");
            ks[slot] = 0;
            assert!(!admitted(512, ks, [MQ; 3]), "zero in slot {slot}");
        }
        for k in [0, 1, 255, 257, 2047, 2049] {
            assert!(!admitted(512, [k; 3], [MQ; 3]), "k {k}");
        }
        assert!(admitted(512, [256; 3], [MQ; 3]));
        assert!(admitted(512, [512; 3], [MQ; 3]));
    }
}

/// `HIPFIRE_PREFILL_IU4_PRODUCER_FUSED` (`developer.prefill_iu4_producer_fused`,
/// a typed boolean): the explicit value, read once. The capability table has
/// raw exact-gfx1201 inventory and a qualified row there, so unset is on at
/// exact gfx1201 and off elsewhere, `1` arms where the admission below holds,
/// `0` is the kill switch.
static PREFILL_IU4_PRODUCER_FUSED: LazyLock<Option<bool>> =
    LazyLock::new(|| hipfire_config::developer_typed_bool("HIPFIRE_PREFILL_IU4_PRODUCER_FUSED"));

/// `HIPFIRE_PREFILL_IU4_PRODUCER_FUSED` resolved through the capability table
/// for `gpu`'s architecture: an explicit `1` needs the raw gfx1201 inventory,
/// `0` is off, unset follows the exact-gfx1201 qualification.
pub(super) fn prefill_iu4_producer_fused_enabled(gpu: &Gpu) -> bool {
    use rdna_compute::prefill_route_capabilities::{enabled, raw_inventory, PrefillRoute};
    let route = PrefillRoute::PrefillIu4ProducerFused;
    // An explicit `1` wins over the qualified default on any architecture, so
    // the raw inventory (exact gfx1201 launchers) is a separate conjunct.
    enabled(*PREFILL_IU4_PRODUCER_FUSED, &gpu.arch, route) && raw_inventory(&gpu.arch, route)
}

/// The only shared-down K the fused producer serves (the A3B shared expert's
/// intermediate width).
const SHARED_DOWN_PRODUCER_FUSED_K: usize = 512;

/// Pure admission of the fused shared-down SwiGLU/FWHT/`block_i4_128` producer:
/// the flag resolved through the capability table, the GPU capability of the
/// generic MQ4G256V2 GEMM's IU4 ADD consumer (`mq4g256v2_shared_iu4_supported`:
/// exact gfx1201 WMMA, IU4 prefill, no A8/FP8 precedence, eager, K % 256, rows >= 64),
/// an MQ4G256V2 down whose K is the shared intermediate width and equals
/// [`SHARED_DOWN_PRODUCER_FUSED_K`], at least 64 rows, and a recipe whose
/// `SharedDown` lowers immediately after `SharedActivation`
/// (`!shared_after_combine`), so no routed stage can run between the producer
/// and the consumer and the sidecar scratch slot cannot be re-reserved under
/// the token.
fn shared_down_producer_fused_admitted(
    flag: bool,
    gpu_capable: bool,
    down_dtype: DType,
    down_k: usize,
    intermediate: usize,
    rows: usize,
    shared_after_combine: bool,
) -> bool {
    flag && gpu_capable
        && !shared_after_combine
        && down_dtype == DType::MQ4G256V2
        && down_k == SHARED_DOWN_PRODUCER_FUSED_K
        && intermediate == down_k
        && rows >= 64
}

/// Whether the shared down of `p` takes the fused producer on `gpu`. The shared
/// activation stage and the shared down stage both evaluate it; the token the
/// activation stage returns is the only thing the down stage trusts.
fn prefill_shared_down_producer_fused_admitted(
    gpu: &Gpu,
    p: &crate::families::moe::MoePrefillParams<'_>,
) -> bool {
    let flag = prefill_iu4_producer_fused_enabled(gpu);
    if !flag {
        return false;
    }
    let Some(shared) = p.prelude.shared.as_ref() else {
        return false;
    };
    let down = &shared.weights.down;
    shared_down_producer_fused_admitted(
        flag,
        gpu.mq4g256v2_shared_iu4_supported(down.k, p.batch_size),
        down.dtype,
        down.k,
        shared.intermediate,
        p.batch_size,
        p.recipe.shared_after_combine(),
    )
}

/// `HIPFIRE_MOE_SHARED_DOWN_SIGMOID_EPI` (`developer.moe_shared_down_sigmoid_epi`, a
/// typed boolean): the explicit value, read once. Unset follows the route's exact-gfx1201
/// qualification, `0` is off and `1` opts in (where the raw exact-gfx1201 inventory and the
/// admission below hold).
static SHARED_DOWN_SIGMOID_EPI: LazyLock<Option<bool>> =
    LazyLock::new(|| hipfire_config::developer_typed_bool("HIPFIRE_MOE_SHARED_DOWN_SIGMOID_EPI"));

/// Pure admission of the shared-down sigmoid residual epilogue: the flag resolved
/// through the capability table (with raw inventory), the GPU capability of the V3
/// IU4 ADD consumer (`mq4g256v2_shared_down_sigmoid_epi_supported`, which carries
/// the recording/capture, rows, K and symmetric-fold conjuncts), an MQ4G256V2 down
/// whose output rows are the residual row width, and no consumer of the shared
/// intermediate or per-stage output armed (the fused launch writes neither the
/// shared down temp nor a separate pre-add residual).
fn shared_down_sigmoid_epi_admitted(
    flag: bool,
    gpu_capable: bool,
    down_dtype: DType,
    down_m: usize,
    residual_dim: usize,
    collector_active: bool,
) -> bool {
    flag && gpu_capable
        && down_dtype == DType::MQ4G256V2
        && down_m != 0
        && down_m == residual_dim
        && !collector_active
}

/// Whether a calibration / Hessian collector, the MoE stage dump or the hidden-state
/// dump is armed: the same guards the decode head fusion honours.
fn shared_down_collectors_active(gpu: &Gpu) -> bool {
    gpu.active_capture.is_some()
        || gpu.hessian_capture.is_some()
        || moe_dump::enabled()
        || hipfire_config::developer_var("HIPFIRE_DUMP_HIDDEN").is_ok()
}

fn prefill_shared_down_sigmoid_epi_admitted(
    gpu: &Gpu,
    p: &crate::families::moe::MoePrefillParams<'_>,
) -> bool {
    use rdna_compute::prefill_route_capabilities::{enabled, raw_inventory, PrefillRoute};
    let route = PrefillRoute::MoeSharedDownSigmoidEpi;
    let flag =
        enabled(*SHARED_DOWN_SIGMOID_EPI, &gpu.arch, route) && raw_inventory(&gpu.arch, route);
    if !flag {
        return false;
    }
    let Some(shared) = p.prelude.shared.as_ref() else {
        return false;
    };
    let down = &shared.weights.down;
    shared_down_sigmoid_epi_admitted(
        flag,
        gpu.mq4g256v2_shared_down_sigmoid_epi_supported(down.m, down.k, p.batch_size),
        down.dtype,
        down.m,
        p.down_m,
        shared_down_collectors_active(gpu),
    )
}

#[cfg(test)]
mod shared_down_producer_fused_tests {
    use super::*;

    fn admitted(
        flag: bool,
        cap: bool,
        dtype: DType,
        k: usize,
        intermediate: usize,
        rows: usize,
        after_combine: bool,
    ) -> bool {
        shared_down_producer_fused_admitted(flag, cap, dtype, k, intermediate, rows, after_combine)
    }

    #[test]
    fn admits_only_the_a3b_mq4v2_shared_down_shape() {
        let mq = DType::MQ4G256V2;
        assert!(admitted(true, true, mq, 512, 512, 64, false));
        assert!(admitted(true, true, mq, 512, 512, 4096, false));
        for (flag, cap) in [(false, true), (true, false), (false, false)] {
            assert!(!admitted(flag, cap, mq, 512, 512, 512, false));
        }
        // Rows below the IU4 consumer's floor.
        assert!(!admitted(true, true, mq, 512, 512, 63, false));
        assert!(!admitted(true, true, mq, 512, 512, 0, false));
        // K is the A3B shared width only.
        for k in [0, 256, 768, 1024, 2048, 17408] {
            assert!(!admitted(true, true, mq, k, k, 512, false), "k {k}");
        }
        // The down K must be the shared intermediate width.
        assert!(!admitted(true, true, mq, 512, 1024, 512, false));
        assert!(!admitted(true, true, mq, 512, 0, 512, false));
    }

    #[test]
    fn rejects_every_other_down_dtype() {
        for dtype in [
            DType::MQ4G256,
            DType::MQ6G256,
            DType::MQ6G256V2,
            DType::ParoQ4G128,
            DType::Q8_0,
            DType::F32,
            DType::BF16,
            DType::MQ4G128V2,
        ] {
            assert!(
                !admitted(true, true, dtype, 512, 512, 512, false),
                "{dtype:?}"
            );
        }
    }

    #[test]
    fn shared_after_combine_keeps_the_incumbent() {
        // `SharedDown` is lowered after the routed stages: the sidecar slot
        // is not provably untouched between producer and consumer.
        assert!(!admitted(true, true, DType::MQ4G256V2, 512, 512, 512, true));
    }
}

#[cfg(test)]
mod shared_down_sigmoid_epi_tests {
    use super::*;

    fn admitted(
        flag: bool,
        cap: bool,
        dtype: DType,
        down_m: usize,
        dim: usize,
        collector: bool,
    ) -> bool {
        shared_down_sigmoid_epi_admitted(flag, cap, dtype, down_m, dim, collector)
    }

    #[test]
    fn admits_only_the_flagged_capable_mq4v2_uncollected_shape() {
        assert!(admitted(true, true, DType::MQ4G256V2, 2048, 2048, false));
        for (flag, cap, collector) in [
            (false, true, false),
            (true, false, false),
            (true, true, true),
            (false, false, true),
        ] {
            assert!(
                !admitted(flag, cap, DType::MQ4G256V2, 2048, 2048, collector),
                "flag {flag} cap {cap} collector {collector}"
            );
        }
    }

    #[test]
    fn rejects_every_other_down_dtype() {
        for dtype in [
            DType::MQ6G256V2,
            DType::MQ4G256,
            DType::MQ6G256,
            DType::Q8_0,
            DType::F32,
            DType::ParoQ4G128,
            DType::MFP4G32E8,
        ] {
            assert!(!admitted(true, true, dtype, 2048, 2048, false), "{dtype:?}");
        }
    }

    #[test]
    fn rejects_a_down_that_is_not_the_residual_row_width() {
        assert!(!admitted(true, true, DType::MQ4G256V2, 0, 0, false));
        assert!(!admitted(true, true, DType::MQ4G256V2, 2048, 4096, false));
    }
}

/// Shared expert activation. Returns `Some(token)` only when the fused exact
/// producer ran: it wrote both the F32 `shared.rotated` (every other reader of
/// it is preserved) and the IU4 sidecar the shared down consumes. Every other
/// arm returns `None` and the shared down quantizes from `shared.rotated`.
fn prefill_shared_activation_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoePrefillParams<'_>,
) -> Result<Option<rdna_compute::scratch::Int4MmqPrepared>, DispatchError> {
    reject_fp8_moe_unless_routed(&p.dtypes, p.route_policy.map(|policy| policy.capability))?;
    let shared =
        p.prelude.shared.as_ref().ok_or_else(|| {
            DispatchError::Hip("shared activation requires shared weights".into())
        })?;
    let down = &shared.weights.down;
    if prefill_shared_down_producer_fused_admitted(gpu, p) {
        let (k, n) = (shared.intermediate, p.batch_size);
        let reservation = gpu
            .reserve_int4_mmq(k, n)
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
        let token = gpu
            .fused_silu_mul_rotate_mq_i4_gfx12_exact_batched(
                shared.gate_out,
                shared.up_out,
                down.awq_scale,
                Some(shared.rotated),
                reservation,
                k,
                n,
            )
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
        return Ok(Some(token));
    }
    let launched: Result<(), DispatchError> = if down.dtype == DType::ParoQ4G128 {
        let rot = down.rotation.as_ref().ok_or_else(|| {
            DispatchError::Hip("shared Paro down has no rotation metadata".into())
        })?;
        gpu.fused_silu_mul_givens_rotate_f32(
            shared.gate_out,
            shared.up_out,
            shared.rotated,
            rot.pairs,
            rot.theta,
            rot.scales,
            p.batch_size,
            shared.intermediate,
            rot.krot,
        )
        .map_err(|e| DispatchError::Hip(e.to_string()))
    } else if matches!(down.dtype, DType::Q8_0 | DType::F32) {
        gpu.silu_mul_f32(shared.gate_out, shared.up_out, shared.rotated)
            .map_err(|e| DispatchError::Hip(e.to_string()))
    } else if let Some(awq) = down.awq_scale {
        gpu.fused_silu_mul_rotate_mq_awq_batched(
            shared.gate_out,
            shared.up_out,
            awq,
            shared.rotated,
            shared.intermediate,
            p.batch_size,
        )
        .map_err(|e| DispatchError::Hip(e.to_string()))
    } else {
        gpu.fused_silu_mul_rotate_mq_batched(
            shared.gate_out,
            shared.up_out,
            shared.rotated,
            shared.intermediate,
            p.batch_size,
        )
        .map_err(|e| DispatchError::Hip(e.to_string()))
    };
    launched.map(|()| None)
}

/// `prepared` is the IU4 sidecar the fused shared activation wrote (`Some` only
/// when that stage admitted the fused producer). It is consumed with the
/// incumbent GEMM's exact sequence (calibration tap on `shared.rotated`, fp16-X
/// cache invalidate, `memset(out)`, IU4 ADD consumer) and never re-quantized.
/// The stages are adjacent, so a token that is not admitted here or whose
/// scratch generation was superseded is a bug and an `Err` before the tap or
/// any launch, never a fallback.
fn prefill_shared_down_stage(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    p: &crate::families::moe::MoePrefillParams<'_>,
    prepared: Option<&rdna_compute::scratch::Int4MmqPrepared>,
) -> Result<(), DispatchError> {
    reject_fp8_moe_unless_routed(&p.dtypes, p.route_policy.map(|policy| policy.capability))?;
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("shared down requires shared weights".into()))?;
    let down = &shared.weights.down;
    let dim = p.down_m;
    let down_tmp = || GpuTensor {
        buf: unsafe { p.down_expanded.buf.alias() },
        shape: vec![p.batch_size * dim],
        dtype: DType::F32,
    };
    match down.dtype {
        DType::MQ4G256 => gpu
            .gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched(
                &down.buf,
                shared.rotated,
                p.x_batch,
                shared.scalar,
                down.m,
                down.k,
                p.batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        DType::MQ6G256 => gpu
            .gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched(
                &down.buf,
                shared.rotated,
                p.x_batch,
                shared.scalar,
                down.m,
                down.k,
                p.batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        DType::ParoQ4G128 => gpu
            .gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched(
                &down.buf,
                shared.rotated,
                p.x_batch,
                shared.scalar,
                down.m,
                down.k,
                p.batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string())),
        DType::MQ4G256V2 | DType::MQ6G256V2 | DType::Q8_0 | DType::F32 => {
            // Fused epilogue (opt-in): the V3 IU4 ADD entry applies
            // `x_batch += sigmoid(scalar) * down` directly, so neither the temp
            // GEMM nor the separate gated add below runs. A declined or
            // unsupported call returns false with no tap and no launch and the
            // incumbent sequence below (including its prepared-token checks) runs.
            if down.dtype == DType::MQ4G256V2
                && (prepared.is_none() || prefill_shared_down_producer_fused_admitted(gpu, p))
                && prefill_shared_down_sigmoid_epi_admitted(gpu, p)
                && gpu
                    .try_gemm_mq4g256v2_shared_down_sigmoid_epi(
                        down.buf,
                        shared.rotated,
                        prepared,
                        p.x_batch,
                        shared.scalar,
                        down.m,
                        down.k,
                        p.batch_size,
                    )
                    .map_err(|e| DispatchError::Hip(e.to_string()))?
            {
                return Ok(());
            }
            let out = down_tmp();
            if let Some(token) = prepared {
                if !prefill_shared_down_producer_fused_admitted(gpu, p) {
                    return Err(DispatchError::Hip(
                        "shared down received a prepared IU4 token but the fused route is not admitted"
                            .into(),
                    ));
                }
                gpu.int4_mmq_prepared_ptr(token, down.k, p.batch_size)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                // The incumbent GEMM's sequence with the quantize already done by
                // the producer: the tap the generic family takes (on the F32
                // `shared.rotated` the producer also wrote), then the helper's receipt,
                // fp16-X invalidate, `memset(out)` and IU4 ADD consumer.
                gpu.maybe_capture_activation(down.buf, shared.rotated, p.batch_size, down.k);
                gpu.gemm_mq4g256v2_prepared_iu4(
                    down.buf,
                    token,
                    &out,
                    down.m,
                    down.k,
                    p.batch_size,
                )
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            } else {
                // This aliases reused routed scratch, not a residual. Plain GEMM
                // must overwrite it before the gated add into x_batch.
                let key = prefill_shared_gemm_key(down.dtype)
                    .ok_or_else(|| DispatchError::Hip("shared down has no GEMM key".into()))?;
                let gemm = GemmFamily::new();
                gemm.run_key(
                    key,
                    ctx,
                    gpu,
                    &GemmParams {
                        w: down,
                        x: shared.rotated,
                        y: &out,
                        batch_size: p.batch_size,
                    },
                )?;
            }
            gpu.sigmoid_scaled_residual_add_batched_f32(
                p.x_batch,
                &out,
                shared.scalar,
                p.batch_size,
                dim,
            )
            .map_err(|e| DispatchError::Hip(e.to_string()))
        }
        DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8 => {
            let down_f16 = gpu
                .alloc_tensor(&[down.m * down.k], DType::F16)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            gpu.dequantize_mfp4g32_e8_to_f16(&down.buf.buf, &down_f16.buf, down.m, down.k)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            let out = down_tmp();
            let down_ref = WeightRef {
                buf: &down_f16,
                dtype: DType::F16,
                m: down.m,
                k: down.k,
                row_stride: down.k,
                rotation: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            };
            let gemm = GemmFamily::new();
            let result = gemm
                .run_key(
                    KernelKey::GemmF16WmmaMb8,
                    ctx,
                    gpu,
                    &GemmParams {
                        w: &down_ref,
                        x: shared.rotated,
                        y: &out,
                        batch_size: p.batch_size,
                    },
                )
                .and_then(|_| {
                    gpu.sigmoid_scaled_residual_add_batched_f32(
                        p.x_batch,
                        &out,
                        shared.scalar,
                        p.batch_size,
                        dim,
                    )
                    .map_err(|e| DispatchError::Hip(e.to_string()))
                });
            gpu.free_tensor(down_f16)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            result
        }
        _ => Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "prefill-shared-down-dtype",
            arch: "",
            quant: "",
        }),
    }
}

fn decode_gate_side_stage(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeParams<'_>,
    res: crate::families::moe::MoeResolution,
    x_rot_local: Option<&GpuTensor>,
    shared_gate: &GpuTensor,
    shared_up: &GpuTensor,
    route: Option<MoeRouteCapability>,
) -> Result<(), DispatchError> {
    reject_fp8_moe_unless_routed(&p.dtypes, route)?;
    check_qt53_decode_fallback(p, route)?;
    let shared = p.shared.as_ref().ok_or_else(|| {
        DispatchError::Hip("shared gate-side stage requires shared weights".into())
    })?;
    let shared_expert_gate = &shared.weights.selector;
    let shared_gate_w = &shared.weights.gate;
    let shared_up_w = &shared.weights.up;
    let scalar_buf = shared.scalar;
    // ── Gate-side GEMV ───────────────────────────────────────────────────────
    // Views are borrowed from the executor so this stage does not allocate
    // duplicate tensor descriptors.
    // Pre-routed experts-only (canonical non-root): the rank's top-k buffers
    // already hold the root-authoritative IDs, so in-call routing (gate-side
    // GEMV, CPU fallback, DIAG dump, top-k) is skipped and only rotation plus
    // the indexed experts run below. Single/legacy calls always route here
    // (byte-identical: this flag is false for every non-pre-routed seal).
    let skip_routing = false;
    if !skip_routing {
        if res.gate_fusable {
            let xr = x_rot_local.expect("gate_fusable implies x_rot_local (needs_x_rot_local)");
            // Exact-uniform V1 MQ4G256 gate quartet → one fused launch.
            hip!(gpu.fused_qkvza_hfq4g256(
                &p.router.buf,
                &shared_expert_gate.buf,
                &shared_gate_w.buf,
                &shared_up_w.buf,
                xr,
                p.router_logits,
                scalar_buf,
                &shared_gate,
                &shared_up,
                p.router.m,
                shared_expert_gate.m,
                shared_gate_w.m,
                shared_up_w.m,
                p.router.k,
            ))?;
        } else if res.gate_fusable_mq4v2
            && (ctx.arch.is_gfx1100() || ctx.arch.is_gfx1201())
            && p.batch_size == 1
            && p.router.awq_scale.is_none()
            && shared_expert_gate.awq_scale.is_none()
            && shared_gate_w.awq_scale.is_none()
            && shared_up_w.awq_scale.is_none()
            && p.router.k == 2048
            && shared_expert_gate.k == 2048
            && shared_gate_w.k == 2048
            && shared_up_w.k == 2048
            && p.router.m == 256
            && shared_expert_gate.m == 1
            && shared_gate_w.m == 512
            && shared_up_w.m == 512
        {
            let xr =
                x_rot_local.expect("gate_fusable_mq4v2 implies x_rot_local (needs_x_rot_local)");
            // Exact official Ornith qt44 gate quartet on gfx1100/gfx1201 only →
            // one V2 fused launch. Any miss (arch/AWQ/shape/batch) falls through
            // to the generic four-GEMV branch below — never errors.
            hip!(gpu.fused_qkvza_hfq4g256_mq4v2(
                &p.router.buf,
                &shared_expert_gate.buf,
                &shared_gate_w.buf,
                &shared_up_w.buf,
                xr,
                p.router_logits,
                scalar_buf,
                &shared_gate,
                &shared_up,
                p.router.m,
                shared_expert_gate.m,
                shared_gate_w.m,
                shared_up_w.m,
                p.router.k,
            ))?;
        } else if route.is_some_and(MoeRouteCapability::is_qt44_qt53_grouped)
            && [&p.router, shared_expert_gate, shared_gate_w, shared_up_w]
                .iter()
                .all(|w| w.awq_scale.is_none() && w.k == p.router.k)
            && [p.router.dtype, shared_expert_gate.dtype] == [DType::BF16; 2]
            && matches!(shared_gate_w.dtype, DType::BF16 | DType::Q8_0)
            && shared_up_w.dtype == shared_gate_w.dtype
        {
            // Router, selector and shared gate/up are projections of the same
            // natural activation, each row computed as the single-matrix GEMV
            // would: the BF16 ones share one launch, a Q8_0 gate/up pair
            // (recipe r2) shares another.
            let bf16_shared = shared_gate_w.dtype == DType::BF16;
            // Exact gfx1151 decode (flag `qwen4_decode_head_fuse`): with a Q8_0
            // gate/up pair at K = 2560 the two launches become one, every
            // output row the incumbent's bitwise. Never under a recorder or a
            // graph capture.
            if !bf16_shared
                && gpu.flags.qwen4_decode_head_fuse_enabled()
                && p.batch_size == 1
                && p.router.k == 2560
                && shared_gate_w.m == shared_up_w.m
                && !gpu.replay.is_recording()
                && !gpu.graphs.capture_mode
                && gpu.active_capture.is_none()
                && gpu.hessian_capture.is_none()
                && !moe_dump::enabled()
                && hipfire_config::developer_var("HIPFIRE_DUMP_HIDDEN").is_err()
                && p.ep_mode == crate::families::moe::MoeEpMode::None
                && p.ep_row_gather.is_none()
            {
                hip!(gpu.gemv_bf16_xf32_x2_q8_0_k2560_staged_pair(
                    [
                        (p.router.buf, p.router_logits, p.router.m),
                        (shared_expert_gate.buf, scalar_buf, shared_expert_gate.m),
                    ],
                    [
                        (shared_gate_w.buf, shared_gate),
                        (shared_up_w.buf, shared_up)
                    ],
                    shared_gate_w.m,
                    p.x_norm,
                    p.router.k,
                ))?;
            } else {
                let rows = |m: usize| if bf16_shared { m } else { 0 };
                hip!(gpu.gemv_bf16_xf32_x4(
                    [
                        (p.router.buf, p.router_logits, p.router.m),
                        (shared_expert_gate.buf, scalar_buf, shared_expert_gate.m),
                        (shared_gate_w.buf, shared_gate, rows(shared_gate_w.m)),
                        (shared_up_w.buf, shared_up, rows(shared_up_w.m)),
                    ],
                    p.x_norm,
                    p.router.k,
                ))?;
                if !bf16_shared {
                    hip!(gpu.gemv_q8_0_pair(
                        shared_gate_w.buf,
                        shared_up_w.buf,
                        p.x_norm,
                        shared_gate,
                        shared_up,
                        shared_gate_w.m,
                        shared_gate_w.k,
                    ))?;
                }
            }
        } else {
            static GEMV_GATE: OnceLock<GemvFamily> = OnceLock::new();
            let gemv = GEMV_GATE.get_or_init(GemvFamily::new);
            // Reuse the single normalized FWHT activation for structural MQ
            // router/scalar weights when the fused quartet is not admitted
            // (mixed dtype, non-MQ4 gate side, AWQ, etc.).
            let router_prerot = x_rot_local.is_some()
                && p.router.awq_scale.is_none()
                && shared_expert_gate.awq_scale.is_none()
                && matches!(
                    crate::types::dtype_post_rotation_variant(p.router.dtype),
                    crate::types::GemvVariant::Prerotated
                )
                && matches!(
                    crate::types::dtype_post_rotation_variant(shared_expert_gate.dtype),
                    crate::types::GemvVariant::Prerotated
                )
                && crate::types::KernelKey::dtype_arch_predicate(p.router.dtype).eval_arch(ctx)
                && crate::types::KernelKey::dtype_arch_predicate(shared_expert_gate.dtype)
                    .eval_arch(ctx);
            if router_prerot {
                let xr = x_rot_local.expect("router_prerot implies x_rot_local");
                gemv.run(
                    ctx,
                    gpu,
                    &crate::families::gemv::GemvParams {
                        w: &p.router,
                        x: xr,
                        y: p.router_logits,
                        variant: crate::types::GemvVariant::Prerotated,
                        residual: None,
                        gate: None,
                        up: None,
                    },
                )
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
                gemv.run(
                    ctx,
                    gpu,
                    &crate::families::gemv::GemvParams {
                        w: shared_expert_gate,
                        x: xr,
                        y: scalar_buf,
                        variant: crate::types::GemvVariant::Prerotated,
                        residual: None,
                        gate: None,
                        up: None,
                    },
                )
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            } else {
                gemv.run_auto(ctx, gpu, &p.router, p.x_norm, p.router_logits)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                gemv.run_auto(ctx, gpu, shared_expert_gate, p.x_norm, scalar_buf)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
            }
            // Shared-expert gate/up: on a graded file the all-MQ4 fused gate path
            // (fused_qkvza_hfq4g256 on the single rotated `xr`) doesn't apply because
            // the router is Q8. But the dense shared gate/up are still MQ-family, and
            // `x_rot_local` has ALREADY been FWHT-rotated once for the routed experts.
            // `run_auto` here would re-rotate x_norm per call (+2 mq_rotate_x/layer);
            // instead reuse the existing rotation via the Prerotated path. Numerically
            // identical (same rotated activation). Q8/HFQ shared weights (no rotation)
            // or AWQ-scaled shared weights fall through to run_auto unchanged.
            let shared_prerot = x_rot_local.is_some()
                && shared_gate_w.awq_scale.is_none()
                && shared_up_w.awq_scale.is_none()
                && matches!(
                    crate::types::dtype_post_rotation_variant(shared_gate_w.dtype),
                    crate::types::GemvVariant::Prerotated
                )
                && matches!(
                    crate::types::dtype_post_rotation_variant(shared_up_w.dtype),
                    crate::types::GemvVariant::Prerotated
                )
                && crate::types::KernelKey::dtype_arch_predicate(shared_gate_w.dtype)
                    .eval_arch(ctx)
                && crate::types::KernelKey::dtype_arch_predicate(shared_up_w.dtype).eval_arch(ctx);
            if shared_prerot {
                let xr = x_rot_local.expect("shared_prerot implies x_rot_local");
                gemv.run(
                    ctx,
                    gpu,
                    &crate::families::gemv::GemvParams {
                        w: shared_gate_w,
                        x: xr,
                        y: &shared_gate,
                        variant: crate::types::GemvVariant::Prerotated,
                        residual: None,
                        gate: None,
                        up: None,
                    },
                )
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
                gemv.run(
                    ctx,
                    gpu,
                    &crate::families::gemv::GemvParams {
                        w: shared_up_w,
                        x: xr,
                        y: &shared_up,
                        variant: crate::types::GemvVariant::Prerotated,
                        residual: None,
                        gate: None,
                        up: None,
                    },
                )
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            } else {
                gemv.run_auto(ctx, gpu, shared_gate_w, p.x_norm, &shared_gate)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                gemv.run_auto(ctx, gpu, shared_up_w, p.x_norm, &shared_up)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
            }
        }
    } // end `if !skip_routing` (gate-side GEMV)
      // BF16-source recipes round F32 projection scratch at their storage
      // boundary before the router and shared nonlinearities consume it. The
      // grouped route's consumers round what they read themselves (the top-10
      // router its logits, `shared_expert_activation_bf16_f32` the selector
      // and the shared gate/up), so no pass is owed here.
    if p.recipe.bf16_round_trip() && !route.is_some_and(MoeRouteCapability::is_qt44_qt53_grouped) {
        // The decode step program is single-row (`build_moe_decode` binds
        // `batch_size: 1`), but every scratch buffer is sized for the prefill
        // chunk cap (512 rows). Rounding the whole buffer would carry 512x the
        // traffic of the one live row, so each boundary is bound to the extent
        // its consumer actually reads: one router row (n_exp), the selector's
        // single scalar, and the already-live-sized shared gate/up slices.
        hip!(gpu.bf16_round_trip_f32(&slice_moe_f32_view(p.router_logits, 0, p.n_exp)))?;
        hip!(gpu.bf16_round_trip_f32(&slice_moe_f32_view(scalar_buf, 0, 1)))?;
        hip!(gpu.bf16_round_trip_f32(shared_gate))?;
        hip!(gpu.bf16_round_trip_f32(shared_up))?;
    }
    Ok(())
}

fn decode_route_gpu_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeParams<'_>,
    shared_gate: &GpuTensor,
    shared_up: &GpuTensor,
    router_shared_fuse: bool,
    exact_wave64_router: bool,
    wave64_router: bool,
    route: Option<MoeRouteCapability>,
) -> Result<(), DispatchError> {
    reject_fp8_moe_unless_routed(&p.dtypes, route)?;
    check_qt53_decode_fallback(p, route)?;
    let smi = p
        .shared
        .as_ref()
        .map(|shared| shared.intermediate)
        .unwrap_or(0);
    let skip_routing = false;
    // DIAG: dump router logits before softmax (mirrors qwen35 HIPFIRE_DUMP_HIDDEN)
    // Skipped for experts-only: no router ran, so the buffer holds
    // root-distributed IDs' stale logits, not this call's routing.
    if !skip_routing {
        if let Ok(dump_path) = hipfire_config::developer_var("HIPFIRE_DUMP_HIDDEN") {
            if gpu.hip.device_synchronize().is_ok() {
                if let Ok(all) = gpu.download_f32(p.router_logits) {
                    use std::io::Write;
                    let path = format!("{dump_path}.router_raw_p");
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&path)
                    {
                        let _ = f.write_all(&(0u32).to_le_bytes());
                        for v in &all[..all.len().min(p.n_exp * 4 / 4)] {
                            let _ = f.write_all(&v.to_le_bytes());
                        }
                    }
                }
            }
        }
    } // end `if !skip_routing` (DIAG dump)
      // Router variant flags were resolved during typed-program lowering.
      // Pre-routed experts-only skips top-k production: the buffers already hold
      // the root-authoritative IDs (a per-rank re-ranking is exactly the
      // divergence this removes).
    if !skip_routing {
        let keep_mask = p.route_policy.map_or(0, |policy| policy.expert_keep_mask);
        if keep_mask != 0 {
            // The masked kernel reads unrounded logits whenever the in-place
            // round in the gate stage was skipped (a grouped route), so the
            // observer copy owes the same round: the exact complement of it.
            #[cfg(feature = "moe-oracle")]
            router_observer::record(
                "decode-top10-masked",
                router_observer::RouterScoresAfterRoute::Logits,
                p.recipe.bf16_round_trip()
                    && route.is_some_and(MoeRouteCapability::is_qt44_qt53_grouped),
            );
            hip!(gpu.moe_router_softmax_top10_masked_f32(
                p.router_logits,
                p.topk_indices,
                p.topk_weights,
                1,
                p.norm_topk_prob,
                keep_mask,
                p.recipe.bf16_round_trip(),
                // Decode is one row; same fault ABI as the dense router below.
                p.layer_idx as usize,
                0,
            ))?;
        } else if route.is_some_and(MoeRouteCapability::is_qt44_qt53_grouped) {
            #[cfg(feature = "moe-oracle")]
            router_observer::record(
                "decode-top10-grouped",
                router_observer::RouterScoresAfterRoute::Logits,
                p.recipe.bf16_round_trip(),
            );
            hip!(gpu.moe_router_softmax_top10_f32(
                p.router_logits,
                p.topk_indices,
                p.topk_weights,
                1,
                p.norm_topk_prob,
                p.recipe.bf16_round_trip(),
                // Decode is one row; a captured or retained body bakes the launch's
                // row base in, so a fault reports row 0 (the layer names the site).
                p.layer_idx as usize,
                0,
            ))?;
        } else if router_shared_fuse {
            #[cfg(feature = "moe-oracle")]
            router_observer::record(
                "decode-fused-wave64-exact-shared-silu-rotate",
                router_observer::RouterScoresAfterRoute::Logits,
                false,
            );
            let shared_x_rot = unsafe {
                GpuTensor {
                    buf: gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias(),
                    shape: vec![gpu.scratch.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                    dtype: DType::F32,
                }
            };
            hip!(
                gpu.moe_router_softmax_topk_k8_wave64_exact_shared_silu_mq_rotate(
                    p.router_logits,
                    p.topk_indices,
                    p.topk_weights,
                    p.n_exp,
                    p.norm_topk_prob,
                    &shared_gate,
                    &shared_up,
                    &shared_x_rot,
                    smi,
                )
            )?;
        } else if exact_wave64_router {
            #[cfg(feature = "moe-oracle")]
            router_observer::record(
                "decode-fused-wave64-exact",
                router_observer::RouterScoresAfterRoute::Logits,
                false,
            );
            hip!(gpu.moe_router_softmax_topk_k8_wave64_exact(
                p.router_logits,
                p.topk_indices,
                p.topk_weights,
                p.n_exp,
                p.norm_topk_prob
            ))?;
        } else if wave64_router {
            #[cfg(feature = "moe-oracle")]
            router_observer::record(
                "decode-fused-wave64",
                router_observer::RouterScoresAfterRoute::Logits,
                false,
            );
            hip!(gpu.moe_router_softmax_topk_k8_wave64(
                p.router_logits,
                p.topk_indices,
                p.topk_weights,
                p.n_exp,
                p.norm_topk_prob
            ))?;
        } else {
            #[cfg(feature = "moe-oracle")]
            router_observer::record(
                "decode-softmax-topk",
                router_observer::RouterScoresAfterRoute::Probabilities,
                false,
            );
            hip!(gpu.softmax_f32(p.router_logits))?;
            hip!(gpu.moe_topk_renorm_k8(
                p.router_logits,
                p.topk_indices,
                p.topk_weights,
                p.n_exp,
                p.norm_topk_prob
            ))?;
        }
        // The grouped top-10 router stores its weights BF16-rounded itself.
        if p.recipe.bf16_round_trip()
            && !route.is_some_and(MoeRouteCapability::is_qt44_qt53_grouped)
        {
            // Only the selected slots are live; scratch may be prefill-sized.
            hip!(gpu.bf16_round_trip_f32(&slice_moe_f32_view(p.topk_weights, 0, p.k)))?;
        }
    } // end `if !skip_routing` (top-k)
    Ok(())
}

fn decode_shared_down_stage(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeParams<'_>,
    shared_gate: &GpuTensor,
    shared_up: &GpuTensor,
    target: &GpuTensor,
    router_shared_fuse: bool,
    route: Option<MoeRouteCapability>,
) -> Result<(), DispatchError> {
    reject_fp8_moe_unless_routed(&p.dtypes, route)?;
    let shared = p
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("shared down stage requires shared weights".into()))?;
    let shared_down_w = &shared.weights.down;
    let smi = shared.intermediate;
    let scalar_buf = shared.scalar;
    let out_target = target;
    // ── Shared expert down ───────────────────────────────────────────────────
    // EP: on rank>0 `skip_shared` is set so the replicated shared expert is
    // summed exactly once (computed on rank 0 only). Router + shared gate/up
    // still ran above (fused with the router GEMV) — only the down/accumulate
    // is skipped here. Accumulates into `out_target` (= the EP partial when
    // `routed_out` is set, else `x_residual`).
    if route.is_some_and(MoeRouteCapability::is_qt44_qt53_grouped) && !p.skip_shared {
        #[cfg(feature = "deltanet")]
        {
            // Single-row decode consumes only selector scalar[0] for the
            // shared add. BF16 recipes round it on the way into the sigmoid;
            // `bf16_scaled_add` rounds the sigmoid it reads, so no round trip
            // follows. The shared gate/up are rounded as silu_mul reads them.
            let scalar_live = slice_moe_f32_view(scalar_buf, 0, 1);
            let shared_hid = slice_moe_f32_view(p.ffn_hidden, 0, smi);
            if qt44_shared_activation_in_routed(p, route) {
                // The routed activation launch already ran it.
            } else if p.recipe.bf16_round_trip() {
                hip!(gpu.shared_expert_activation_bf16_f32(
                    shared_gate,
                    shared_up,
                    &shared_hid,
                    &scalar_live,
                    1,
                ))?;
            } else {
                hip!(gpu.sigmoid_f32(&scalar_live))?;
                hip!(gpu.silu_mul_f32(shared_gate, shared_up, &shared_hid))?;
            }
            static GEMV_QT44_QT53_SHARED_DOWN: LazyLock<GemvFamily> =
                LazyLock::new(GemvFamily::new);
            if p.recipe.bf16_round_trip()
                && shared_down_w.dtype == DType::BF16
                && shared_down_w.awq_scale.is_none()
            {
                // The BF16 shared down GEMV folds its rows into the residual
                // with the source BF16 scaled add in its epilogue.
                hip!(gpu.gemv_bf16_xf32_bf16_scaled_add(
                    shared_down_w.buf,
                    &shared_hid,
                    out_target,
                    scalar_buf,
                    shared_down_w.m,
                    shared_down_w.k,
                ))?;
                return Ok(());
            }
            // Q8_0 K=640 shared down with the same BF16 fold in its epilogue
            // (exact gfx1151, eager; bitwise the GEMV + scaled-add launches).
            if p.recipe.bf16_round_trip()
                && shared_down_w.dtype == DType::Q8_0
                && shared_down_w.awq_scale.is_none()
                && hip!(gpu.gemv_q8_0_k640_bf16_scaled_add(
                    shared_down_w.buf,
                    &shared_hid,
                    p.ffn_out,
                    out_target,
                    scalar_buf,
                    shared_down_w.m,
                    shared_down_w.k,
                ))?
            {
                return Ok(());
            }
            if shared_down_w.dtype == DType::MQ4G128V2 {
                let shared_down_rot = slice_moe_f32_view(p.rot_batch, 0, shared_down_w.k);
                hip!(gpu.rotate_x_mq_128_v2(&shared_hid, &shared_down_rot, shared_down_w.k, 1,))?;
                hip!(gpu.gemv_mq4g128v2(
                    &shared_down_w.buf,
                    &shared_down_rot,
                    p.ffn_out,
                    shared_down_w.m,
                    shared_down_w.k,
                ))?;
            } else {
                GEMV_QT44_QT53_SHARED_DOWN
                    .run_auto(ctx, gpu, shared_down_w, &shared_hid, p.ffn_out)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
            }
            // One live row of the shared-expert output (the buffer is the
            // prefill chunk cap wide). `bf16_scaled_add` rounds the residual,
            // the value and the gate it reads and stores a BF16-exact sum, so
            // no round trip brackets it.
            if p.recipe.bf16_round_trip() {
                hip!(bf16_scaled_add(
                    gpu,
                    &Bf16ScaledAdd {
                        residual: out_target,
                        value: p.ffn_out,
                        scalar: scalar_buf,
                        elements: p.hidden,
                    },
                ))?;
            } else {
                hip!(gpu.scaled_add_inplace_gpu_scalar_f32(out_target, p.ffn_out, scalar_buf))?;
            }
            return Ok(());
        }
        #[cfg(not(feature = "deltanet"))]
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "qt44-qt53-shared-down-requires-deltanet",
            arch: "",
            quant: "",
        });
    }
    if !p.skip_shared {
        if shared_down_w.dtype == DType::MQ4G256 {
            hip!(gpu.ensure_mq_signs())?;
            let x_rot_alias = unsafe {
                GpuTensor {
                    buf: gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias(),
                    shape: vec![gpu.scratch.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                    dtype: DType::F32,
                }
            };
            if let Some(awq) = shared_down_w.awq_scale {
                hip!(gpu.fused_silu_mul_rotate_mq_awq(
                    &shared_gate,
                    &shared_up,
                    awq,
                    &x_rot_alias,
                    smi
                ))?;
            } else if !router_shared_fuse {
                hip!(gpu.fused_silu_mul_rotate_mq(&shared_gate, &shared_up, &x_rot_alias, smi))?;
            }
            hip!(gpu.gemv_hfq4g256_residual_sigmoid_scaled_gpu(
                &shared_down_w.buf,
                &x_rot_alias,
                out_target,
                scalar_buf,
                shared_down_w.m,
                shared_down_w.k,
            ))?;
        // qt=52 rides this exact-V2 arm (NOT the V1 sigmoid_scaled launcher):
        // silu+FWHT, then the GemvFamily prerotated LUT kernel + scalar add.
        // The k512 fused launcher below stays exact-qt44-only.
        } else if matches!(
            shared_down_w.dtype,
            DType::MQ4G256V2 | DType::MQ6G256V2 | DType::MQ4G256V2Lloyd
        ) {
            // Exact dense V2 shared-down. qt44/qt47 dual-half headers MUST NOT
            // ride the V1 HFQ4 residual_sigmoid kernel (silent fluent corruption).
            // Sequence mirrors MQ4: silu+FWHT → prerotated dense V2 GEMV →
            // sigmoid(c)·add. Dense routing goes through GemvFamily so the
            // container dtype selects gemv_mq{4,6}g256v2 / residual sisters.
            hip!(gpu.ensure_mq_signs())?;
            let x_rot_alias = unsafe {
                GpuTensor {
                    buf: gpu.scratch.mq_x_rot.as_ref().unwrap().buf.alias(),
                    shape: vec![gpu.scratch.mq_x_rot.as_ref().unwrap().buf.size() / 4],
                    dtype: DType::F32,
                }
            };
            if let Some(awq) = shared_down_w.awq_scale {
                hip!(gpu.fused_silu_mul_rotate_mq_awq(
                    &shared_gate,
                    &shared_up,
                    awq,
                    &x_rot_alias,
                    smi
                ))?;
            } else if !router_shared_fuse {
                hip!(gpu.fused_silu_mul_rotate_mq(&shared_gate, &shared_up, &x_rot_alias, smi))?;
            }
            // Default-on one-launch fuse for the exact Ornith qt44 shared-down
            // shape. The scalar has no later read in this executor. `ffn_out`
            // is dead here on the indexed path; on the generic fallback its
            // next access is an overwrite by routed-expert down. `0` or an
            // invalid override keeps sigmoid + plain V2 GEMV + scaled_add.
            static MQ4V2_SHARED_DOWN_FUSED: LazyLock<bool> = LazyLock::new(|| {
                hipfire_config::developer_var("HIPFIRE_MQ4V2_SHARED_DOWN_FUSED")
                    .map(|value| value == "1")
                    .unwrap_or(true)
            });
            let use_mq4v2_shared_down_fused = *MQ4V2_SHARED_DOWN_FUSED
                && shared_down_w.dtype == DType::MQ4G256V2
                && shared_down_w.awq_scale.is_none()
                && (ctx.arch.is_gfx1100() || ctx.arch.is_gfx1201())
                && shared_down_w.m == 2_048
                && shared_down_w.k == 512;
            if use_mq4v2_shared_down_fused {
                hip!(gpu.gemv_mq4g256v2_residual_sigmoid_scaled_k512(
                    &shared_down_w.buf,
                    &x_rot_alias,
                    out_target,
                    scalar_buf,
                    shared_down_w.m,
                    shared_down_w.k,
                ))?;
            } else {
                #[cfg(feature = "deltanet")]
                {
                    hip!(gpu.sigmoid_f32(scalar_buf))?;
                    static GEMV_SHARED_V2: OnceLock<GemvFamily> = OnceLock::new();
                    let gemv = GEMV_SHARED_V2.get_or_init(GemvFamily::new);
                    gemv.run(
                        ctx,
                        gpu,
                        &crate::families::gemv::GemvParams {
                            w: shared_down_w,
                            x: &x_rot_alias,
                            y: p.ffn_out,
                            variant: crate::types::GemvVariant::Prerotated,
                            residual: None,
                            gate: None,
                            up: None,
                        },
                    )?;
                    hip!(gpu.scaled_add_inplace_gpu_scalar_f32(out_target, p.ffn_out, scalar_buf))?;
                }
                #[cfg(not(feature = "deltanet"))]
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "shared-down-v2-requires-deltanet",
                    arch: "",
                    quant: dtype_name(shared_down_w.dtype),
                });
            }
        } else {
            // Non-MQ4 / non-V2 shared expert down. Requires deltanet feature for
            // sigmoid_f32. Returns UnsupportedVariant for builds without the
            // feature to keep hipfire-dispatch compilable without deltanet.
            #[cfg(feature = "deltanet")]
            {
                hip!(gpu.sigmoid_f32(scalar_buf))?;
                let shared_hid = slice_moe_f32_view(p.ffn_hidden, 0, smi);
                hip!(gpu.silu_mul_f32(&shared_gate, &shared_up, &shared_hid))?;
                static GEMV_DOWN: OnceLock<GemvFamily> = OnceLock::new();
                let gemv = GEMV_DOWN.get_or_init(GemvFamily::new);
                gemv.run_auto(ctx, gpu, shared_down_w, &shared_hid, p.ffn_out)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                hip!(gpu.scaled_add_inplace_gpu_scalar_f32(out_target, p.ffn_out, scalar_buf))?;
            }
            #[cfg(not(feature = "deltanet"))]
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "shared-down-non-mq4-requires-deltanet",
                arch: "",
                quant: "",
            });
        }
    }
    Ok(())
}

fn decode_gate_up_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeParams<'_>,
    res: crate::families::moe::MoeResolution,
    activation_input: &GpuTensor,
    ninepath_d3: bool,
    route: Option<MoeRouteCapability>,
) -> Result<(), DispatchError> {
    reject_fp8_moe(&p.dtypes)?;
    check_qt53_decode_fallback(p, route)?;
    let xr = activation_input;
    let gate_up_k = p.routed_gate_up_k;
    // ── Routed-expert dispatch via device-indexed merged kernels ──────────
    //
    // Mixed-tier graded quants (mq4p/mq3p/mq4r/mq4rug) carry a per-expert
    // `expert_dtype_tags` table.  The merged kernels
    // (`gemv_mixed_moe_gate_up_k8_indexed_batched` and
    // `gemv_mixed_moe_down_k8_indexed_batched_expanded`) read that table
    // on-device — no D2H, fully hipGraph-capturable — and branch the dequant
    // per-block.  Uniform quants simply have `expert_dtype_tags = None` and
    // fall through to the single-dtype arms below, which is byte-identical to
    // the old pre-SP2 behaviour.
    // Select gate_up + down GEMVs by their INDIVIDUAL dtypes, not a coupled
    // Precedence: mixed gate when gate_up exact dtype varies must run
    // before representative MQ4V2/MQ6V2/V1 arms. Uniform shortcut only
    // when gate_up exact DType equality (per_expert_gate_up uniform).
    let gate_up_varies = gate_up_varies(p.dtypes.per_expert_gate_up.as_deref());
    if route.is_some_and(MoeRouteCapability::is_qt44_qt53_grouped) {
        let routed_slots = 10usize
            .checked_mul(p.mi)
            .ok_or_else(|| DispatchError::Hip("QT44/QT53 routed slot width overflow".into()))?;
        let gate_batch = slice_moe_f32_view(p.gate_batch, 0, routed_slots);
        let up_batch = slice_moe_f32_view(p.up_batch, 0, routed_slots);
        if route.is_some_and(MoeRouteCapability::is_slab) {
            // Sealed expert-slab C1 decode: the K1 lane-union GEMV over codec
            // buckets (per-expert codecs, pointer-addressed).
            let layer = slab_route::layer(
                &gpu.arch,
                p.route_policy.as_ref(),
                &p.dtypes,
                p.expert_dtype_tags,
            )?;
            slab_route::k1_gate_up(
                gpu,
                layer,
                p.expert_gate_up_ptrs,
                p.topk_indices,
                xr,
                &gate_batch,
                &up_batch,
                1,
            )?;
            return Ok(());
        }
        #[cfg(feature = "test-utils")]
        bind_g1_window(
            gpu,
            p.routed_experts,
            p.n_exp,
            p.topk_weights,
            p.recipe.bf16_round_trip(),
        )?;
        hip!(gpu.gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(
            p.expert_gate_up_ptrs,
            p.topk_indices,
            xr,
            &gate_batch,
            &up_batch,
            2 * p.mi,
            p.hidden,
            1,
        ))?;
        // BF16 recipes round gate and up where the activation reads them
        // (`silu_mul_bf16_rt_f32`), their only reader.
        return Ok(());
    }
    if ninepath_d3 {
        // Only HFQ4/MQ4V1 may call the V1 ninepath D3 kernel. V2 uses
        // exact native indexed gate (below) + V2 D4.
        debug_assert_eq!(
            ninepath_d3_family(p.dtypes.routed_gate_up, p.dtypes.routed_down),
            Some("hfq4"),
            "ninepath D3 is HFQ4-only"
        );
        hip!(gpu.gemv_hfq4g256_moe_ninepath_d3(
            p.expert_gate_up_ptrs,
            p.topk_indices,
            xr,
            p.gate_batch,
            p.up_batch,
            p.mi,
            gate_up_k,
        ))?;
    } else if decode_gate_uses_mixed(p.expert_dtype_tags.is_some(), gate_up_varies) {
        // Per-expert mixed gate_up (V1/V2 graded, N-tier etc). Must
        // precede the representative MQ4V2/MQ6V2/V1 arms — otherwise a
        // layer with mixed V1+V2 gate dtypes would silently mis-decode
        // the minority tier via the representative's header.
        let tags = p.expert_dtype_tags.expect("mixed gate requires tags");
        hip!(gpu.gemv_mixed_moe_gate_up_k8_indexed_batched(
            p.expert_gate_up_ptrs,
            tags,
            p.topk_indices,
            xr,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            gate_up_k,
            p.k,
            1,
        ))?;
    } else if res.routed_indexable_mq4v2 || p.dtypes.routed_gate_up == DType::MQ4G256V2 {
        // qt44. MUST precede the trailing `else`, which dispatches the qt13
        // kernel: qt13 reads bytes [0..8) as one f32 scale + one f32 zero,
        // where qt44 stores two f16 scale/zero pairs. Same 136 B stride, so
        // the misread is silent — fluent text, wrong numbers. Match on the
        // gate_up dtype (not only the coupled flag) so a split pair still
        // selects the V2 gate decoder.
        hip!(gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed(
            p.expert_gate_up_ptrs,
            p.topk_indices,
            xr,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            gate_up_k,
        ))?;
    } else if res.routed_indexable_mq6v2 || p.dtypes.routed_gate_up == DType::MQ6G256V2 {
        // qt47. MUST precede HFQ4/HFQ6 arms: dual-half f16 header is
        // incompatible with V1 MQ6 f32 scale/zero; same 200 B stride so a
        // misread is silent fluent corruption.
        hip!(gpu.gemv_mq6g256v2_moe_gate_up_k8_indexed(
            p.expert_gate_up_ptrs,
            p.topk_indices,
            xr,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            gate_up_k,
        ))?;
    } else if res.routed_indexable_paro {
        hip!(gpu.gemv_paro_q4g128_moe_gate_up_k8_indexed(
            p.expert_gate_up_ptrs,
            p.topk_indices,
            xr,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            gate_up_k,
            p.k,
        ))?;
    } else if matches!(
        p.dtypes.routed_gate_up,
        DType::MQ2G256Lloyd | DType::MQ2G256LloydU
    ) {
        // Uniform MQ2-Lloyd routed experts: ds4/minimax indexed Lloyd gate_up
        // GEMV. y_gate/y_up are separate buffers; m = 2*p.mi (kernel splits at
        // M/2 internally); trailing k_top = p.k. X is the FWHT-rotated xr.
        mq2_lloyd_gate_up(
            gpu,
            p.mq2_lloyd.as_ref(),
            p.expert_gate_up_ptrs,
            p.topk_indices,
            xr,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            gate_up_k,
            p.k,
        )?;
    } else if p.dtypes.routed_gate_up == DType::MQ3G256Lloyd {
        // Uniform MQ3-Lloyd routed experts: same indexed-Lloyd gate_up path,
        // MQ3 launcher.
        hip!(gpu.deepseek4_gemv_mq3g256_lloyd_moe_gate_up_indexed(
            p.expert_gate_up_ptrs,
            p.topk_indices,
            xr,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            gate_up_k,
            p.k,
        ))?;
    } else if p.dtypes.routed_gate_up == DType::MQ2G256GL {
        // Uniform MQ2-GL routed gate_up: 2-bit indices against the
        // TENSOR-GLOBAL codebook (GL_CB2, passed as scalar kernel args) plus
        // a per-block fp16 scale, SoA. Same call shape as the MQ2-Lloyd arm
        // above — y_gate/y_up separate, m = 2*p.mi (kernel splits at M/2),
        // X is the FWHT-rotated xr.
        hip!(gpu.gemv_mq2g256gl_moe_gate_up_indexed(
            p.expert_gate_up_ptrs,
            p.topk_indices,
            xr,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            gate_up_k,
            p.k,
        ))?;
    } else if p.dtypes.routed_gate_up == DType::MQ3G256GL {
        // Uniform MQ3-GL routed gate_up: same path, 8-entry global codebook
        // (GL_CB3) and 96 B of indices per group.
        hip!(gpu.gemv_mq3g256gl_moe_gate_up_indexed(
            p.expert_gate_up_ptrs,
            p.topk_indices,
            xr,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            gate_up_k,
            p.k,
        ))?;
    } else if p.dtypes.routed_gate_up == DType::MQ5G256 {
        hip!(gpu.gemv_hfq5g256_moe_gate_up_k8_indexed(
            p.expert_gate_up_ptrs,
            p.topk_indices,
            xr,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            gate_up_k,
        ))?;
    } else if p.dtypes.routed_gate_up == DType::MQ6G256 && p.k < 8 {
        // Below top-8 (the biased sigmoid route) take the batched launcher,
        // as the HFQ4 arm below does.
        hip!(gpu.gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
            p.expert_gate_up_ptrs,
            p.topk_indices,
            xr,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            gate_up_k,
            p.k,
            1,
        ))?;
    } else if p.dtypes.routed_gate_up == DType::MQ6G256 {
        hip!(gpu.gemv_hfq6g256_moe_gate_up_k8_indexed(
            p.expert_gate_up_ptrs,
            p.topk_indices,
            xr,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            gate_up_k,
            p.k,
        ))?;
    } else if p.dtypes.routed_gate_up == DType::MFP4G32E8 {
        // mfp4-E8 grouped experts (gfx1151-only; gated in MoeResolution).
        hip!(gpu.gemv_mfp4g32_e8_moe_gate_up_k8_indexed(
            p.expert_gate_up_ptrs,
            p.topk_indices,
            xr,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            gate_up_k,
        ))?;
    } else if p.k < 8 {
        // Below top-8 (the biased sigmoid route) the batched launcher is the
        // faster kernel: bitwise identical, +1.2% LFM2.5-8B-A1B decode (k=4,
        // gfx1151).
        hip!(gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
            p.expert_gate_up_ptrs,
            p.topk_indices,
            xr,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            gate_up_k,
            p.k,
            1,
        ))?;
    } else {
        hip!(gpu.gemv_hfq4g256_moe_gate_up_k8_indexed(
            p.expert_gate_up_ptrs,
            p.topk_indices,
            xr,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            gate_up_k,
            p.k,
        ))?;
    }
    if p.recipe.bf16_round_trip() {
        let active = p
            .k
            .checked_mul(p.mi)
            .ok_or_else(|| DispatchError::Hip("MoE gate/up round-trip extent overflows".into()))?;
        hip!(gpu.bf16_round_trip_f32(&slice_moe_f32_view(p.gate_batch, 0, active)))?;
        hip!(gpu.bf16_round_trip_f32(&slice_moe_f32_view(p.up_batch, 0, active)))?;
    }
    Ok(())
}

/// Whether the grouped BF16 decode runs the shared expert's activation (and
/// selector sigmoid) inside the routed activation launch; its inputs are the
/// router GEMV's shared gate/up, so it need not wait for the shared stage.
fn qt44_shared_activation_in_routed(
    p: &crate::families::moe::MoeParams<'_>,
    route: Option<MoeRouteCapability>,
) -> bool {
    route.is_some_and(MoeRouteCapability::is_qt44_qt53_grouped)
        && p.recipe.bf16_round_trip()
        && !p.skip_shared
        // The fused launch is the routed Activation stage, which a CPU-deferred
        // call does not lower: its shared expert activation runs in its own stage.
        && !p.cpu_deferred_experts
        && p.ep_mode == crate::families::moe::MoeEpMode::None
        && p.shared
            .as_ref()
            .is_some_and(|shared| shared.intermediate <= p.mi.div_ceil(128) * 128)
}

fn decode_activation_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeParams<'_>,
    res: crate::families::moe::MoeResolution,
    route: Option<MoeRouteCapability>,
    shared_views: Option<(&GpuTensor, &GpuTensor)>,
) -> Result<(), DispatchError> {
    reject_fp8_moe(&p.dtypes)?;
    check_qt53_decode_fallback(p, route)?;
    if route.is_some_and(MoeRouteCapability::is_qt44_qt53_grouped) {
        let routed_slots =
            p.k.checked_mul(p.mi)
                .ok_or_else(|| DispatchError::Hip("MoE routed slot width overflows".into()))?;
        let gate_batch = slice_moe_f32_view(p.gate_batch, 0, routed_slots);
        let up_batch = slice_moe_f32_view(p.up_batch, 0, routed_slots);
        let rot_batch = slice_moe_f32_view(p.rot_batch, 0, routed_slots);
        if p.recipe.bf16_round_trip() {
            // Activation and the down projection's 128-wide rotation, one pass
            // (with the shared expert's activation when it rides along).
            let shared = match (
                qt44_shared_activation_in_routed(p, route),
                shared_views,
                &p.shared,
            ) {
                (true, Some((gate, up)), Some(shared)) => Some((
                    gate,
                    up,
                    slice_moe_f32_view(p.ffn_hidden, 0, shared.intermediate),
                    slice_moe_f32_view(shared.scalar, 0, 1),
                )),
                (true, ..) => {
                    return Err(DispatchError::Hip(
                        "sealed moe: fused shared activation has no shared views".into(),
                    ))
                }
                _ => None,
            };
            hip!(gpu.silu_mul_bf16_rt_rotate_x_mq_128_v2_shared(
                &gate_batch,
                &up_batch,
                &rot_batch,
                p.mi,
                p.k,
                shared.as_ref().map(|(g, u, out, sel)| (*g, *u, out, sel)),
            ))?;
        } else {
            hip!(gpu.silu_mul_f32(&gate_batch, &up_batch, &rot_batch))?;
            hip!(gpu.rotate_x_mq_128_v2(&rot_batch, &rot_batch, p.mi, p.k))?;
        }
        return Ok(());
    }
    // Gate→down: fused silu+mul+rotate
    if p.dtypes.routed_down == DType::MQ4G128V2 {
        // qwen35 qt53 routed-down typed fallback: F32 silu_mul then an in-place
        // FWHT128 rotation of the live [k, mi] rows (`RoutedDownRotation::Fwht128`;
        // the Flash-Next non-BF16 recipe), never the fused FWHT256 silu+rotate.
        let live = p
            .k
            .checked_mul(p.mi)
            .ok_or_else(|| DispatchError::Hip("qt53 routed activation extent overflows".into()))?;
        let gate_live = slice_moe_f32_view(p.gate_batch, 0, live);
        let up_live = slice_moe_f32_view(p.up_batch, 0, live);
        let rot_live = slice_moe_f32_view(p.rot_batch, 0, live);
        hip!(gpu.silu_mul_f32(&gate_live, &up_live, &rot_live))?;
        hip!(gpu.rotate_x_mq_128_v2(&rot_live, &rot_live, p.mi, p.k))?;
    } else if res.routed_indexable_paro {
        let paro_down = p
            .routed_down_paro
            .as_ref()
            .expect("routed_indexable_paro implies down paro sidecar");
        hip!(gpu.fused_silu_mul_givens_rotate_f32(
            p.gate_batch,
            p.up_batch,
            p.rot_batch,
            &paro_down.pairs,
            &paro_down.theta,
            &paro_down.scales,
            p.k,
            p.mi,
            paro_down.krot,
        ))?;
    } else if let Some(awq_ptrs) = p.expert_down_awq_ptrs {
        // Route A MoE-AWQ: per-routed-expert down.awq_scale selected by
        // topk_indices[krank]. Divides silu(g)*u by the expert's scale before
        // the FWHT (AWQ math (W·s)·(x/s)=W·x). Only reached on .hfq files
        // carrying per-expert down sidecars — byte-identical otherwise.
        hip!(gpu.fused_silu_mul_rotate_mq_awq_indexed_batched(
            p.gate_batch,
            p.up_batch,
            awq_ptrs,
            p.topk_indices,
            p.rot_batch,
            p.mi,
            p.k,
        ))?;
    } else if gate_down_skips_rotation(p.dtypes.routed_down) {
        // UNROTATED down weights: silu+mul with NO FWHT.
        //
        // Every other arm here fuses the rotation in because its down
        // weights were packed in the FWHT basis. MQ2G256LloydU is the
        // unrotated sibling — rotating the intermediate before an
        // unrotated down GEMV is exactly the silent-garbage failure this
        // dtype exists to avoid, and it would not crash or even look
        // wrong until the model generated text.
        //
        // silu_mul is elementwise, so one launch over the whole
        // [k_top * mi] buffer covers every selected expert.
        hip!(gpu.silu_mul_f32(p.gate_batch, p.up_batch, p.rot_batch))?;
    } else if let (_, Some(awq)) = no_shared_routed_awq(p) {
        hip!(gpu.fused_silu_mul_rotate_mq_awq_batched(
            p.gate_batch,
            p.up_batch,
            awq,
            p.rot_batch,
            p.mi,
            p.k
        ))?;
    } else if let Some(lloyd) = p.mq2_lloyd.as_ref() {
        // Declared MQ2-Lloyd: SwiGLU clamp, then rotate (beta's selected-decode order).
        mq2_lloyd_silu_rotate(gpu, lloyd, p.gate_batch, p.up_batch, p.rot_batch, p.mi, p.k)?;
    } else {
        // MQ4/MQ6, no AWQ on expert down weights (the common case for A3B).
        hip!(gpu.fused_silu_mul_rotate_mq_batched(
            p.gate_batch,
            p.up_batch,
            p.rot_batch,
            p.mi,
            p.k
        ))?;
    }
    if p.recipe.bf16_round_trip() {
        let active = p.k.checked_mul(p.mi).ok_or_else(|| {
            DispatchError::Hip("MoE activation round-trip extent overflows".into())
        })?;
        hip!(gpu.bf16_round_trip_f32(&slice_moe_f32_view(p.rot_batch, 0, active)))?;
    }
    Ok(())
}

fn decode_down_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeParams<'_>,
    res: crate::families::moe::MoeResolution,
    target: &GpuTensor,
    ninepath_d4: bool,
    ninepath_mq3l: bool,
    ninepath_mq4v2: bool,
    ninepath_mq4g128v2: bool,
    ninepath_mq6v2: bool,
    down_last_combine: bool,
    route: Option<MoeRouteCapability>,
) -> Result<(), DispatchError> {
    reject_fp8_moe(&p.dtypes)?;
    check_qt53_decode_fallback(p, route)?;
    let out_target = target;
    let down_m = p.routed_down_m;
    let down_k = p.routed_down_k;
    if route.is_some_and(MoeRouteCapability::is_qt44_qt53_grouped) {
        let routed_slots = 10usize
            .checked_mul(p.mi)
            .ok_or_else(|| DispatchError::Hip("QT44/QT53 routed slot width overflow".into()))?;
        let rot_batch = slice_moe_f32_view(p.rot_batch, 0, routed_slots);
        let down_expanded = slice_moe_f32_view(p.down_expanded, 0, 10 * p.hidden);
        if route.is_some_and(MoeRouteCapability::is_slab) {
            // One lane of an MQ4/W3 layer runs K1-C1 straight from `topk`; otherwise over the union
            // this call's gate/up planned (same stream, no other slab launch between the two).
            let layer = slab_route::layer(
                &gpu.arch,
                p.route_policy.as_ref(),
                &p.dtypes,
                p.expert_dtype_tags,
            )?;
            slab_route::k1_down(
                gpu,
                layer,
                p.topk_indices,
                p.expert_down_ptrs,
                &rot_batch,
                &down_expanded,
                1,
            )?;
            // The top-10 combine rounds every expert output it reads to BF16; a
            // deferred combine may read the buffer elsewhere, so it keeps the pass.
            if p.recipe.bf16_round_trip() && p.defer_routed_combine {
                hip!(gpu.bf16_round_trip_f32(&down_expanded))?;
            }
            return Ok(());
        }
        #[cfg(feature = "test-utils")]
        bind_g1_window(
            gpu,
            p.routed_experts,
            p.n_exp,
            p.topk_weights,
            p.recipe.bf16_round_trip(),
        )?;
        hip!(gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
            p.expert_down_ptrs,
            p.topk_indices,
            &rot_batch,
            &down_expanded,
            down_m,
            down_k,
            1,
            p.n_exp,
            10,
        ))?;
        // The top-10 combine rounds every expert output it reads to BF16; a
        // deferred combine may read the buffer elsewhere, so it keeps the pass.
        if p.recipe.bf16_round_trip() && p.defer_routed_combine {
            hip!(gpu.bf16_round_trip_f32(&down_expanded))?;
        }
        return Ok(());
    }
    // Expanded write — down GEMV by the DOWN dtype (mixed mq6-down lands here).
    // FIXME(Step 8): replace hardcoded 1 with p.batch_size when grouped prefill lands
    if ninepath_d4 && ninepath_mq4g128v2 {
        // qwen35 qt53 nine-path: the Activation stage launched nothing; this one
        // kernel forms silu(gate)*up, the FWHT128 rotation, the eight routed
        // down GEMVs and the ascending-slot weighted fold into `out_target`
        // (single owner per row), so the shared combine below is skipped like
        // every other ninepath arm. Byte-exact with the sequence it replaces.
        hip!(gpu.gemv_mq4g128v2_moe_ninepath_d4(
            p.expert_down_ptrs,
            p.topk_indices,
            p.topk_weights,
            p.gate_batch,
            p.up_batch,
            out_target,
            down_m,
            down_k,
            p.n_exp,
        ))?;
    } else if p.dtypes.routed_down == DType::MQ4G128V2 {
        // qwen35 qt53 routed-down typed fallback: unweighted F32 expanded rows
        // `[k, hidden]` for the live `p.k` slots (the launcher's runtime k_top);
        // the existing F32 `moe_down_combine_k8_batched` applies route weights.
        hip!(gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
            p.expert_down_ptrs,
            p.topk_indices,
            p.rot_batch,
            p.down_expanded,
            down_m,
            down_k,
            1,
            p.n_exp,
            p.k,
        ))?;
    } else if ninepath_d4 && ninepath_mq3l {
        // MQ3-Lloyd codebook port. Folds the 8 partials in LDS in ascending
        // krank order (single owner per row, no atomics), so the shared
        // combine below is skipped exactly as it is for the HFQ4 arm.
        hip!(gpu.gemv_mq3g256_lloyd_moe_ninepath_d4(
            p.expert_down_ptrs,
            p.topk_indices,
            p.topk_weights,
            p.rot_batch,
            out_target,
            down_m,
            down_k,
        ))?;
    } else if ninepath_d4 && ninepath_mq4v2 {
        // MUST precede the bare `ninepath_d4` arm below, which calls the
        // qt13 kernel. qt13 and qt44 share a 136 B stride and nibble
        // packing and differ only in the header, so that fallthrough would
        // misread qt44 silently.
        hip!(gpu.gemv_mq4g256v2_moe_ninepath_d4(
            p.expert_down_ptrs,
            p.topk_indices,
            p.topk_weights,
            p.rot_batch,
            out_target,
            down_m,
            down_k,
        ))?;
    } else if ninepath_d4 && ninepath_mq6v2 {
        // qt47. Same silent-header hazard vs HFQ4 ninepath: 200 B stride
        // still "fits" a wrong decoder. Pin the V2 dual-half path first.
        hip!(gpu.gemv_mq6g256v2_moe_ninepath_d4(
            p.expert_down_ptrs,
            p.topk_indices,
            p.topk_weights,
            p.rot_batch,
            out_target,
            down_m,
            down_k,
        ))?;
    } else if ninepath_d4 {
        hip!(gpu.gemv_hfq4g256_moe_ninepath_d4(
            p.expert_down_ptrs,
            p.topk_indices,
            p.topk_weights,
            p.rot_batch,
            out_target,
            down_m,
            down_k,
        ))?;
    } else if let Some(tags) = p.expert_dtype_tags {
        // Per-expert mixed down (graded MQ6 hot / MQ2-Lloyd cold). One
        // merged kernel; block-per-(row,krank,token) reads tags[expert_id]
        // (block-uniform → no warp divergence) and branches the dequant.
        // Writes the EXPANDED buffer for BOTH dtypes → the single shared
        // moe_down_combine_k8_batched runs below (self-combine forced off).
        hip!(gpu.gemv_mixed_moe_down_k8_indexed_batched_expanded(
            p.expert_down_ptrs,
            tags,
            p.topk_indices,
            p.rot_batch,
            p.down_expanded,
            down_m,
            down_k,
            p.k,
            1,
        ))?;
    } else if res.routed_indexable_paro {
        hip!(gpu.gemv_paro_q4g128_moe_down_k8_indexed_batched(
            p.expert_down_ptrs,
            p.topk_indices,
            p.rot_batch,
            p.down_expanded,
            down_m,
            down_k,
            p.k,
            1,
        ))?;
    } else if let Some(lloyd) = p.mq2_lloyd.as_ref() {
        // Declared MQ2-Lloyd: expanded down + combine when deterministic (default),
        // else the atomic self-combining residual (beta's selected-decode policy).
        if lloyd.deterministic_down() {
            mq2_lloyd_down_expanded(
                gpu,
                lloyd,
                p.expert_down_ptrs,
                p.expert_gate_up_ptrs,
                p.topk_indices,
                p.rot_batch,
                p.down_expanded,
                down_m,
                down_k,
                p.k,
            )?;
        } else {
            mq2_lloyd_down_residual(
                gpu,
                p.expert_down_ptrs,
                p.topk_indices,
                p.topk_weights,
                p.rot_batch,
                out_target,
                down_m,
                down_k,
                p.k,
                lloyd.uses_atomic_down,
            )?;
        }
    } else if matches!(
        p.dtypes.routed_down,
        DType::MQ2G256Lloyd | DType::MQ2G256LloydU
    ) {
        // MQ2-Lloyd down: atomic, weighted, SELF-COMBINING residual GEMV.
        // silu-output rotate (rot_batch) -> down -> * topk_weight[krank] ->
        // atomicAdd into out_target, all in one launch. NO separate combine
        // (skipped below). out_target = routed_out (EP zeroed partial) or
        // x_residual; the atomic accumulate is EP-correct unchanged.
        hip!(
            gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
                p.expert_down_ptrs,
                p.topk_indices,
                p.topk_weights,
                p.rot_batch,
                out_target,
                down_m,
                down_k,
                p.k,
                false,
            )
        )?;
    } else if p.dtypes.routed_down == DType::MQ3G256Lloyd {
        // MQ3-Lloyd down: same atomic self-combining residual GEMV, MQ3 launcher.
        hip!(
            gpu.deepseek4_gemv_mq3g256_lloyd_moe_down_residual_scaled_indexed(
                p.expert_down_ptrs,
                p.topk_indices,
                p.topk_weights,
                p.rot_batch,
                out_target,
                down_m,
                down_k,
                p.k,
            )
        )?;
    } else if p.dtypes.routed_down == DType::MQ2G256GL {
        // MQ2-GL down: atomic, weighted, SELF-COMBINING residual GEMV —
        // same epilogue contract as the MQ2/MQ3-Lloyd down kernels (one
        // launch does down -> * topk_weight[krank] -> atomicAdd into
        // out_target). NO separate combine; `routed_down_self_combines`
        // below MUST include this dtype or every MoE layer double-counts.
        hip!(gpu.gemv_mq2g256gl_moe_down_residual_scaled_indexed(
            p.expert_down_ptrs,
            p.topk_indices,
            p.topk_weights,
            p.rot_batch,
            out_target,
            down_m,
            down_k,
            p.k,
        ))?;
    } else if p.dtypes.routed_down == DType::MQ3G256GL {
        // MQ3-GL down: same atomic self-combining residual GEMV, 8-entry
        // global codebook.
        hip!(gpu.gemv_mq3g256gl_moe_down_residual_scaled_indexed(
            p.expert_down_ptrs,
            p.topk_indices,
            p.topk_weights,
            p.rot_batch,
            out_target,
            down_m,
            down_k,
            p.k,
        ))?;
    } else if p.dtypes.routed_down == DType::MQ5G256 {
        hip!(gpu.gemv_hfq5g256_moe_down_k8_indexed_batched_expanded(
            p.expert_down_ptrs,
            p.topk_indices,
            p.rot_batch,
            p.down_expanded,
            down_m,
            down_k,
            p.k,
            1,
        ))?;
    } else if p.dtypes.routed_down == DType::MQ4G256V2 {
        // qt44 non-ninepath expanded down. MUST precede the final HFQ4
        // else: same 136 B stride, dual-half header — silent misread.
        hip!(gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
            p.expert_down_ptrs,
            p.topk_indices,
            p.rot_batch,
            p.down_expanded,
            down_m,
            down_k,
            p.k,
            1,
        ))?;
    } else if p.dtypes.routed_down == DType::MQ6G256 {
        hip!(gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
            p.expert_down_ptrs,
            p.topk_indices,
            p.rot_batch,
            p.down_expanded,
            down_m,
            down_k,
            p.k,
            1,
        ))?;
    } else if p.dtypes.routed_down == DType::MQ6G256V2 {
        // qt47 non-ninepath expanded down. Same silent-header hazard vs
        // V1 MQ6 / HFQ4 fallthrough.
        hip!(gpu.gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded(
            p.expert_down_ptrs,
            p.topk_indices,
            p.rot_batch,
            p.down_expanded,
            down_m,
            down_k,
            p.k,
            1,
        ))?;
    } else if p.dtypes.routed_down == DType::MFP4G32E8 {
        // mfp4-E8 grouped expert down (atomic-free expanded; combine below).
        hip!(gpu.gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded(
            p.expert_down_ptrs,
            p.topk_indices,
            p.rot_batch,
            p.down_expanded,
            down_m,
            down_k,
            p.k,
            1,
        ))?;
    } else if down_last_combine {
        hip!(gpu.gemv_hfq4g256_moe_down_k8_indexed_last_combine(
            p.expert_down_ptrs,
            p.topk_indices,
            p.rot_batch,
            p.down_expanded,
            p.topk_weights,
            out_target,
            down_m,
            down_k,
            p.k,
            1,
        ))?;
    } else {
        hip!(gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
            p.expert_down_ptrs,
            p.topk_indices,
            p.rot_batch,
            p.down_expanded,
            down_m,
            down_k,
            p.k,
            1,
        ))?;
    }
    if p.recipe.bf16_round_trip() && p.down_writes_expanded() && !ninepath_d4 && !down_last_combine
    {
        let active =
            p.k.checked_mul(p.hidden)
                .ok_or_else(|| DispatchError::Hip("MoE down round-trip extent overflows".into()))?;
        hip!(gpu.bf16_round_trip_f32(&slice_moe_f32_view(p.down_expanded, 0, active)))?;
    }
    Ok(())
}

fn decode_input_basis_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeParams<'_>,
    res: crate::families::moe::MoeResolution,
) -> Result<(), DispatchError> {
    if !res.needs_x_rot_local {
        return Ok(());
    }
    if !res.routed_indexable_paro {
        hip!(gpu.ensure_mq_signs())?;
    }
    if !p.x_rot_prerotated {
        if res.routed_indexable_paro {
            let paro = p
                .routed_gate_up_paro
                .as_ref()
                .expect("routed_indexable_paro implies gate_up paro sidecar");
            hip!(gpu.givens_rotate_to(
                p.x_norm,
                p.x_rot_local,
                &paro.pairs,
                &paro.theta,
                &paro.scales,
                1,
                p.hidden,
                paro.krot,
                128,
            ))?;
        } else if res.gate_side_mq4 {
            if let Some(awq) = p.router.awq_scale {
                hip!(gpu.rotate_x_mq_awq(p.x_norm, awq, p.x_rot_local, p.hidden))?;
            } else {
                hip!(gpu.rotate_x_mq(p.x_norm, p.x_rot_local, p.hidden))?;
            }
        } else if let (Some(awq), _) = no_shared_routed_awq(p) {
            hip!(gpu.rotate_x_mq_awq(p.x_norm, awq, p.x_rot_local, p.hidden))?;
        } else {
            hip!(gpu.rotate_x_mq(p.x_norm, p.x_rot_local, p.hidden))?;
        }
    }
    Ok(())
}

/// `(gate_up, down)` per-layer AWQ activation scales of the routed experts
/// (`(W·s)·(x/s) = W·x`). Only a no-shared recipe owns its input rotation.
fn no_shared_routed_awq<'a>(
    p: &crate::families::moe::MoeParams<'a>,
) -> (Option<&'a GpuTensor>, Option<&'a GpuTensor>) {
    match p.routed_experts.get(0) {
        Some((gate_up, down)) if p.shared.is_none() => (gate_up.awq_scale, down.awq_scale),
        _ => (None, None),
    }
}

fn decode_route_cpu_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeParams<'_>,
    runtime: &OnceCell<crate::pipeline::moe_program::HostRoute>,
    router: crate::pipeline::sealed_moe::MoeRouterInput,
) -> Result<(), DispatchError> {
    reject_fp8_moe(&p.dtypes)?;
    let k = p.k;
    let n_exp = p.n_exp;
    match router {
        crate::pipeline::sealed_moe::MoeRouterInput::SigmoidTopK => {
            #[cfg(feature = "deltanet")]
            hip!(gpu.sigmoid_f32(p.router_logits))?;
            #[cfg(not(feature = "deltanet"))]
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "sigmoid-router-requires-deltanet",
                arch: "",
                quant: "",
            });
        }
        _ => hip!(gpu.softmax_f32(p.router_logits))?,
    }
    #[cfg(feature = "moe-oracle")]
    {
        let (route, scores) = if matches!(
            router,
            crate::pipeline::sealed_moe::MoeRouterInput::SigmoidTopK
        ) {
            (
                "decode-host-sigmoid-topk",
                router_observer::RouterScoresAfterRoute::Sigmoid,
            )
        } else {
            (
                "decode-host-softmax-topk",
                router_observer::RouterScoresAfterRoute::Probabilities,
            )
        };
        router_observer::record(route, scores, false);
    }
    let (topk_indices, topk_weights): (Vec<usize>, Vec<f32>) = if k == 8 {
        hip!(gpu.moe_topk_renorm_k8(
            p.router_logits,
            p.topk_indices,
            p.topk_weights,
            n_exp,
            p.norm_topk_prob
        ))?;
        let idx_f32 = hip!(gpu.download_f32(p.topk_indices))?;
        let wts = hip!(gpu.download_f32(p.topk_weights))?;
        let idx_usize: Vec<usize> = idx_f32
            .iter()
            .map(|&f| i32::from_ne_bytes(f.to_ne_bytes()) as usize)
            .collect();
        (idx_usize, wts)
    } else {
        let probs = hip!(gpu.download_f32(p.router_logits))?;
        let mut indices: Vec<usize> = (0..n_exp).collect();
        indices.select_nth_unstable_by(k - 1, |&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut sel: Vec<usize> = indices.into_iter().take(k).collect();
        sel.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut wts: Vec<f32> = sel.iter().map(|&i| probs[i]).collect();
        if p.norm_topk_prob {
            let sum: f32 = wts.iter().sum();
            if sum > 0.0 {
                for w in wts.iter_mut() {
                    *w /= sum;
                }
            }
        }
        (sel, wts)
    };
    runtime
        .set(crate::pipeline::moe_program::HostRoute {
            indices: topk_indices,
            weights: topk_weights,
        })
        .map_err(|_| DispatchError::Hip("sealed moe: CPU route was already published".into()))?;
    Ok(())
}

fn run_host_weight_gemv(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    gemv: &GemvFamily,
    weight: &WeightRef<'_>,
    x: &GpuTensor,
    y: &GpuTensor,
) -> Result<(), DispatchError> {
    gpu.maybe_capture_activation(weight.buf, x, 1, weight.k);
    gemv.run_auto(ctx, gpu, weight, x, y)
}

fn run_host_weight_gemv_residual(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    gemv: &GemvFamily,
    weight: &WeightRef<'_>,
    x: &GpuTensor,
    residual: &GpuTensor,
) -> Result<(), DispatchError> {
    // Match weight_gemv_residual's observable host-fallback order: residual
    // tap, temporary allocation, ordinary GEMV (and its tap), add, then free.
    gpu.maybe_capture_activation(weight.buf, x, 1, weight.k);
    let tmp = gpu
        .alloc_tensor(&[weight.m], DType::F32)
        .map_err(|e| DispatchError::Hip(e.to_string()))?;
    run_host_weight_gemv(ctx, gpu, gemv, weight, x, &tmp)?;
    gpu.add_inplace_f32(residual, &tmp)
        .map_err(|e| DispatchError::Hip(e.to_string()))?;
    gpu.free_tensor(tmp)
        .map_err(|e| DispatchError::Hip(e.to_string()))
}

fn decode_cpu_experts_stage(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeParams<'_>,
    topk_indices: &[usize],
    topk_weights: &[f32],
) -> Result<(), DispatchError> {
    reject_fp8_moe(&p.dtypes)?;
    let mi = p.mi;
    static GEMV_FB: OnceLock<GemvFamily> = OnceLock::new();
    let gemv = GEMV_FB.get_or_init(GemvFamily::new);

    let hess_x_norm: Option<Vec<f32>> = if gpu.hessian_capture.is_some()
        && matches!(
            p.recipe,
            crate::families::moe::MoeRecipe::SoftmaxGatedShared { .. }
        ) {
        Some(hip!(gpu.download_f32(p.x_norm))?)
    } else {
        None
    };
    let mut hess_down_keys: Vec<(String, Vec<f32>)> = if hess_x_norm.is_some() {
        Vec::with_capacity(topk_indices.len())
    } else {
        Vec::new()
    };
    let mut hess_gate_keys: Vec<String> = if hess_x_norm.is_some() {
        Vec::with_capacity(topk_indices.len())
    } else {
        Vec::new()
    };

    for (&expert_idx, &weight) in topk_indices.iter().zip(topk_weights.iter()) {
        let (gate_up_w, down_w) =
            p.routed_experts
                .get(expert_idx)
                .ok_or(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "cpu-topk-expert-not-resident",
                    arch: "",
                    quant: "",
                })?;
        run_host_weight_gemv(ctx, gpu, gemv, &gate_up_w, p.x_norm, p.gate_up_buf)?;
        let gate_view = slice_moe_f32_view(p.gate_up_buf, 0, mi);
        let up_view = slice_moe_f32_view(p.gate_up_buf, mi, mi);
        let hid_view = slice_moe_f32_view(p.ffn_hidden, 0, mi);
        hip!(gpu.silu_mul_f32(&gate_view, &up_view, &hid_view))?;
        if hess_x_norm.is_some() {
            let hid_host = hip!(gpu.download_f32(&hid_view))?;
            let l = p.layer_idx;
            let e = expert_idx;
            hess_gate_keys.push(format!(
                "model.language_model.layers.{l}.mlp.experts.{e}.gate_up_proj.weight"
            ));
            hess_down_keys.push((
                format!("model.language_model.layers.{l}.mlp.experts.{e}.down_proj.weight"),
                hid_host,
            ));
        }
        if p.recipe == crate::families::moe::MoeRecipe::SigmoidRoutedNoShared {
            // Cohere folds the route weight into the activation before down;
            // preserve this order for its F32/F16/BF16/Q8 host fallback.
            #[cfg(feature = "deltanet")]
            hip!(gpu.scale_f32(&hid_view, weight))?;
            #[cfg(not(feature = "deltanet"))]
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "sigmoid-host-experts-require-deltanet",
                arch: "",
                quant: "",
            });
            run_host_weight_gemv_residual(ctx, gpu, gemv, &down_w, &hid_view, p.x_residual)?;
        } else {
            gemv.run_auto(ctx, gpu, &down_w, &hid_view, p.ffn_out)?;
            hip!(gpu.scaled_add_inplace_cpu_scalar_f32(p.x_residual, p.ffn_out, weight))?;
        }
    }
    if let Some(xn) = &hess_x_norm {
        let mut items: Vec<(String, &[f32], usize)> =
            Vec::with_capacity(hess_gate_keys.len() + hess_down_keys.len());
        for gk in &hess_gate_keys {
            items.push((gk.clone(), xn.as_slice(), p.hidden));
        }
        for (dk, hid) in &hess_down_keys {
            items.push((dk.clone(), hid.as_slice(), mi));
        }
        if let Some(cap) = gpu.hessian_capture.as_mut() {
            cap.accumulate_token(&items);
        }
    }
    Ok(())
}

fn decode_mutation_fence_stage() -> Result<(), DispatchError> {
    #[cfg(feature = "serve-fault-inject")]
    if crate::pipeline::sealed_moe::take_fault_after_expert_mutation() {
        return Err(DispatchError::Hip(
            "injected fault after sealed MoE expert mutation before combine publication".into(),
        ));
    }
    Ok(())
}

fn decode_combine_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoeParams<'_>,
    target: &GpuTensor,
    route: Option<MoeRouteCapability>,
) -> Result<(), DispatchError> {
    reject_fp8_moe_unless_routed(&p.dtypes, route)?;
    check_qt53_decode_fallback(p, route)?;
    let grouped = route.is_some_and(MoeRouteCapability::is_qt44_qt53_grouped);
    let dump = moe_dump::fold_begin(gpu, p.layer_idx as usize, p.hidden, p.x_residual, target)?;
    if grouped {
        let down_expanded = slice_moe_f32_view(p.down_expanded, 0, 10 * p.hidden);
        hip!(gpu.moe_down_combine_top10_batched(
            &down_expanded,
            p.topk_indices,
            p.topk_weights,
            target,
            p.hidden,
            1,
            p.n_exp,
        ))?;
    } else {
        hip!(gpu.moe_down_combine_k8_batched(
            p.down_expanded,
            p.topk_weights,
            target,
            p.routed_down_m,
            p.k,
            1,
        ))?;
    }
    // The grouped route's shared stage then adds into this same target with
    // `bf16_scaled_add`, which rounds the residual it reads: the round trip is
    // owed only when nothing follows (mirrors the grouped prefill combine).
    let shared_add_follows = route.is_some_and(MoeRouteCapability::is_qt44_qt53_grouped)
        && p.recipe.shared_after_combine()
        && !p.skip_shared
        && p.ep_mode == crate::families::moe::MoeEpMode::None;
    if p.recipe.bf16_round_trip() && !shared_add_follows {
        // One live row: the target may be allocated to the prefill chunk cap.
        hip!(gpu.bf16_round_trip_f32(&slice_moe_f32_view(target, 0, p.hidden)))?;
    }
    if dump.is_some() {
        // Dump-only: the rows the kernel above actually read (stride = its `m`).
        let (slots, stride) = if grouped {
            (10, p.hidden)
        } else {
            (p.k, p.routed_down_m)
        };
        moe_dump::fold_end(
            gpu,
            dump,
            target,
            p.routed_out.is_some(),
            &moe_dump::FoldInfo {
                path: if grouped {
                    "expanded_top10_combine"
                } else {
                    "expanded_k8_combine"
                },
                ep_mode: &format!("{:?}", p.ep_mode),
                hidden: p.hidden,
                slots,
                expanded: Some((p.down_expanded, stride)),
                indices: p.topk_indices,
                weights: p.topk_weights,
                expanded_absent_reason: "",
            },
        )?;
    }
    Ok(())
}

fn prefill_scatter_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoePrefillParams<'_>,
    path2_m_total: usize,
) -> Result<(), DispatchError> {
    reject_fp8_moe(&p.dtypes)?;
    check_qt53_prefill_fallback(p)?;
    let total_slots = p
        .batch_size
        .checked_mul(p.k_top)
        .ok_or_else(|| DispatchError::Hip("sealed moe: prefill slot count overflows".into()))?;
    hip!(gpu.moe_scatter_fused_k8(
        p.topk_indices,
        p.expert_token_counts,
        p.expert_offsets,
        p.sorted_slot_index,
        p.expert_tile_ids,
        p.inverse_perm,
        total_slots,
        p.n_exp,
        path2_m_total,
        MOE_GROUPED_BLOCK_M,
    ))?;
    Ok(())
}

fn prefill_input_basis_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoePrefillParams<'_>,
) -> Result<(), DispatchError> {
    let paro = p
        .paro_gate_up
        .as_ref()
        .expect("paro_mode implies paro_gate_up sidecar");
    hip!(gpu.givens_rotate_to(
        p.x_norm_batch,
        p.x_rot_batch,
        paro.pairs,
        paro.theta,
        paro.scales,
        p.batch_size,
        p.gate_up_k,
        paro.krot,
        128,
    ))?;
    Ok(())
}

fn prefill_gate_up_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoePrefillParams<'_>,
    res: &crate::families::moe::MoePrefillResolution,
    path2_m_total: usize,
    force_mq4_grouped_fp16: bool,
    coalesced_verify: bool,
) -> Result<(), DispatchError> {
    reject_fp8_moe(&p.dtypes)?;
    check_qt53_prefill_fallback(p)?;
    let (n, mi, k_top) = (p.batch_size, p.mi, p.k_top);
    let gate_up_k = p.gate_up_k;
    if res.use_path2 {
        // x_rot_batch was prepared by the explicit input-basis stage.
        // Down-only-graded redline: the tag table describes the DOWN dtypes, so
        // for a UNIFORM gate_up (exact DType equality) it must NOT be passed
        // here (the mixed grouped kernel would read MQ4 gate_up bytes with
        // the down's MQ6/MQ3L tags → garbage). Pass None → the uniform
        // grouped kernel. When gate_up exact dtype VARIES (V1/V2 graded,
        // N-tier), preserve tags — mixed gate before representative.
        // Mirrors the decode gate_up fix and satisfies issue 1's
        // "Grouped Path2 preserves tags on V1/V2 variation".
        let gate_up_varies = gate_up_varies(p.dtypes.per_expert_gate_up.as_deref());
        let gate_up_tags = if gate_up_varies {
            p.expert_dtype_tags
        } else {
            None
        };
        dispatch_grouped_gemm(
            gpu,
            p.dtypes.routed_gate_up,
            gate_up_tags,
            p.expert_gate_up_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            p.x_rot_batch,
            p.y_gate_up_grouped,
            2 * mi,
            gate_up_k,
            k_top,
            path2_m_total,
            n,
            force_mq4_grouped_fp16,
            res.use_paro_i8,
            res.use_paro_i8_k8,
            coalesced_verify
                && crate::families::moe::routed_host_mapped_projections(p.routed_experts).0,
        )?;
        // The explicit unscatter stage follows this grouped GEMM.
    } else {
        // Path 1 fallback: per-token indexed GEMV, batched over N tokens.
        // Mixed Path1 (issue 2): when tags exist, use mixed batched gate
        // launcher; do not representative-dispatch tagged layers. Gate uses
        // mixed only when gate_up exact dtype varies (V1/V2 graded); uniform
        // gate shortcut requires exact DType equality. Down always uses mixed
        // when tags exist (handled in the down block below).
        if res.paro_mode {
            hip!(gpu.gemv_paro_q4g128_moe_gate_up_k8_indexed_batched(
                p.expert_gate_up_ptrs,
                p.topk_indices,
                p.x_rot_batch,
                p.gate_batch,
                p.up_batch,
                2 * mi,
                gate_up_k,
                k_top,
                n,
            ))?;
        } else if let Some(tags) = p.expert_dtype_tags {
            let gate_up_varies = gate_up_varies(p.dtypes.per_expert_gate_up.as_deref());
            if decode_gate_uses_mixed(true, gate_up_varies) {
                // Gate varies → mixed gate kernel before representative arms.
                hip!(gpu.gemv_mixed_moe_gate_up_k8_indexed_batched(
                    p.expert_gate_up_ptrs,
                    tags,
                    p.topk_indices,
                    p.x_rot_batch,
                    p.gate_batch,
                    p.up_batch,
                    2 * p.mi,
                    gate_up_k,
                    k_top,
                    n,
                ))?;
            } else {
                // Gate uniform (exact equality) → representative uniform gate.
                let gate_up_result = match p.dtypes.routed_gate_up {
                    DType::MQ4G256 => hip!(gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
                        p.expert_gate_up_ptrs,
                        p.topk_indices,
                        p.x_rot_batch,
                        p.gate_batch,
                        p.up_batch,
                        2 * mi,
                        gate_up_k,
                        k_top,
                        n,
                    )),
                    DType::MQ4G256V2 => hip!(gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed_batched(
                        p.expert_gate_up_ptrs,
                        p.topk_indices,
                        p.x_rot_batch,
                        p.gate_batch,
                        p.up_batch,
                        2 * mi,
                        gate_up_k,
                        k_top,
                        n,
                    )),
                    DType::MQ5G256 => hip!(gpu.gemv_hfq5g256_moe_gate_up_k8_indexed_batched(
                        p.expert_gate_up_ptrs,
                        p.topk_indices,
                        p.x_rot_batch,
                        p.gate_batch,
                        p.up_batch,
                        2 * mi,
                        gate_up_k,
                        k_top,
                        n,
                    )),
                    DType::MQ6G256 => hip!(gpu.gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
                        p.expert_gate_up_ptrs,
                        p.topk_indices,
                        p.x_rot_batch,
                        p.gate_batch,
                        p.up_batch,
                        2 * mi,
                        gate_up_k,
                        k_top,
                        n,
                    )),
                    DType::MQ6G256V2 => hip!(gpu.gemv_mq6g256v2_moe_gate_up_k8_indexed_batched(
                        p.expert_gate_up_ptrs,
                        p.topk_indices,
                        p.x_rot_batch,
                        p.gate_batch,
                        p.up_batch,
                        2 * mi,
                        gate_up_k,
                        k_top,
                        n,
                    )),
                    DType::MFP4G32E8 => hip!(gpu.gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched(
                        p.expert_gate_up_ptrs,
                        p.topk_indices,
                        p.x_rot_batch,
                        p.gate_batch,
                        p.up_batch,
                        2 * mi,
                        gate_up_k,
                        k_top,
                        n,
                    )),
                    _other => {
                        return Err(DispatchError::UnsupportedVariant {
                            family: "moe",
                            variant: "prefill-gate-up-path1-dtype",
                            arch: "",
                            quant: "other",
                        });
                    }
                };
                gate_up_result?;
            }
        } else {
            // MQ4/MQ6 indexed batched GEMV (x_rot_batch is already FWHT-rotated
            // by the model). Uniform path, no tags.
            let gate_up_result = match p.dtypes.routed_gate_up {
                DType::MQ4G256 => hip!(gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
                    p.expert_gate_up_ptrs,
                    p.topk_indices,
                    p.x_rot_batch,
                    p.gate_batch,
                    p.up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )),
                DType::MQ4G256V2 => hip!(gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed_batched(
                    p.expert_gate_up_ptrs,
                    p.topk_indices,
                    p.x_rot_batch,
                    p.gate_batch,
                    p.up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )),
                DType::MQ5G256 => hip!(gpu.gemv_hfq5g256_moe_gate_up_k8_indexed_batched(
                    p.expert_gate_up_ptrs,
                    p.topk_indices,
                    p.x_rot_batch,
                    p.gate_batch,
                    p.up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )),
                DType::MQ6G256 => hip!(gpu.gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
                    p.expert_gate_up_ptrs,
                    p.topk_indices,
                    p.x_rot_batch,
                    p.gate_batch,
                    p.up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )),
                DType::MQ6G256V2 => hip!(gpu.gemv_mq6g256v2_moe_gate_up_k8_indexed_batched(
                    p.expert_gate_up_ptrs,
                    p.topk_indices,
                    p.x_rot_batch,
                    p.gate_batch,
                    p.up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )),
                DType::MFP4G32E8 => hip!(gpu.gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched(
                    p.expert_gate_up_ptrs,
                    p.topk_indices,
                    p.x_rot_batch,
                    p.gate_batch,
                    p.up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )),
                _other => {
                    return Err(DispatchError::UnsupportedVariant {
                        family: "moe",
                        variant: "prefill-gate-up-path1-dtype",
                        arch: "",
                        quant: "other",
                    });
                }
            };
            gate_up_result?;
        }
    }
    Ok(())
}

fn prefill_gate_up_unscatter_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoePrefillParams<'_>,
    path2_m_total: usize,
) -> Result<(), DispatchError> {
    reject_fp8_moe(&p.dtypes)?;
    hip!(gpu.moe_gate_up_unscatter_k8(
        p.y_gate_up_grouped,
        p.sorted_slot_index,
        p.gate_batch,
        p.up_batch,
        p.mi,
        p.k_top,
        path2_m_total,
    ))?;
    Ok(())
}

fn prefill_activation_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoePrefillParams<'_>,
    res: &crate::families::moe::MoePrefillResolution,
    total_slots: usize,
) -> Result<(), DispatchError> {
    reject_fp8_moe(&p.dtypes)?;
    check_qt53_prefill_fallback(p)?;
    let mi = p.mi;
    // ── SwiGLU + rotate over [N*K_TOP × mi] ────────────────────────────
    if p.dtypes.routed_down == DType::MQ4G128V2 {
        // qwen35 qt53 routed-down typed fallback: F32 silu_mul then an in-place
        // FWHT128 rotation of the live [total_slots, mi] rows
        // (`RoutedDownRotation::Fwht128`), never the fused FWHT256 silu+rotate.
        let live = total_slots
            .checked_mul(mi)
            .ok_or_else(|| DispatchError::Hip("qt53 routed activation extent overflows".into()))?;
        let gate_live = slice_moe_f32_view(p.gate_batch, 0, live);
        let up_live = slice_moe_f32_view(p.up_batch, 0, live);
        let rot_live = slice_moe_f32_view(p.rot_batch, 0, live);
        hip!(gpu.silu_mul_f32(&gate_live, &up_live, &rot_live))?;
        hip!(gpu.rotate_x_mq_128_v2(&rot_live, &rot_live, mi, total_slots))?;
    } else if res.paro_mode {
        let paro = p
            .paro_down
            .as_ref()
            .expect("paro_mode implies paro_down sidecar");
        hip!(gpu.fused_silu_mul_givens_rotate_f32(
            p.gate_batch,
            p.up_batch,
            p.rot_batch,
            paro.pairs,
            paro.theta,
            paro.scales,
            total_slots,
            mi,
            paro.krot,
        ))?;
    } else if p.expert_dtype_tags.is_some() {
        // Graded/mixed routed experts: the silu+rotate is weight-agnostic (the
        // per-expert down dtype only affects the down GEMM that READS rot_batch;
        // graded files carry no expert AWQ). This mirrors run_moe_decode, which
        // calls this unconditionally. Without this, the routed_down dtype match
        // below rejects the cold-tier Lloyd dtype (experts[0].down) as `_other`
        // and the prefill forward panics.
        hip!(gpu.fused_silu_mul_rotate_mq_batched(
            p.gate_batch,
            p.up_batch,
            p.rot_batch,
            mi,
            total_slots,
        ))?;
    } else {
        // MQ4/MQ6: the silu+rotate kernel is weight-agnostic (reads only
        // activations, not weight data). AWQ-aware variant when down has AWQ.
        match p.dtypes.routed_down {
            // MFP4G32E8 reuses the weight-agnostic silu+FWHT-rotate (E8 down expects
            // FWHT(silu(g)*u), same as MQ4 — see the decode E8 path).
            //
            // MQ2/MQ3-Lloyd likewise: the decode path feeds those down GEMVs
            // `rot_batch` = FWHT(silu(g)·u) built by the same weight-agnostic
            // kernel family, so the batched twin is the correct analogue. This
            // was a pure MATCH gap — the kernel reads activations only, never
            // weight bytes. The GL dtypes are deliberately absent: they are not
            // batched-prefill admissible (no grouped GEMM, no batched GEMV), so
            // reaching here with a GL down is a bug and must stay a loud error.
            // qt44 belongs here for the same reason as every other MQ dtype:
            // the silu/rotate kernels below consume f32 gate_batch/up_batch
            // activations and never touch quantized weights, so this match is
            // selecting a branch, not a decode path.
            DType::MQ4G256
            | DType::MQ4G256V2
            | DType::MQ5G256
            | DType::MQ6G256
            | DType::MQ6G256V2
            | DType::MQ2G256Lloyd
            | DType::MQ3G256Lloyd
            | DType::MFP4G32E8
            | DType::MFP3G32E8
            | DType::MFP2G32E8 => {
                if let Some(awq_ptrs) = p.expert_down_awq_ptrs {
                    // Route A MoE-AWQ (per-routed-expert, indexed by topk slot).
                    // total_slots rows = N·k_top; each slot's expert is
                    // topk_indices[slot] — the same slot→expert mapping the
                    // indexed down GEMV below uses. Supersedes the single-scale
                    // `down_awq_scale` (Ship 4.2 stub) which incorrectly applied
                    // experts[0]'s scale to every routed slot.
                    //
                    // NOTE: correct for the indexed batched gate_up (Path 0/1,
                    // gfx9*/non-grouped) where rot_batch[slot] aligns with
                    // topk_indices[slot]. Path 2 grouped-WMMA (gfx11/gfx12)
                    // reorders via sorted_slot_index — AWQ+Path2 ordering is
                    // unverified; the only current MoE-AWQ target is A3B on
                    // gfx942 (Path 0). See docs/moe-awq/MOE_AWQ_EXPERTS.md.
                    hip!(gpu.fused_silu_mul_rotate_mq_awq_indexed_batched(
                        p.gate_batch,
                        p.up_batch,
                        awq_ptrs,
                        p.topk_indices,
                        p.rot_batch,
                        mi,
                        total_slots,
                    ))?;
                } else if let Some(awq) = p.down_awq_scale {
                    hip!(gpu.fused_silu_mul_rotate_mq_awq_batched(
                        p.gate_batch,
                        p.up_batch,
                        awq,
                        p.rot_batch,
                        mi,
                        total_slots,
                    ))?;
                } else {
                    hip!(gpu.fused_silu_mul_rotate_mq_batched(
                        p.gate_batch,
                        p.up_batch,
                        p.rot_batch,
                        mi,
                        total_slots,
                    ))?;
                }
            }
            _other => {
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "prefill-silu-rotate-dtype",
                    arch: "",
                    quant: "other",
                });
            }
        }
    }
    Ok(())
}

fn prefill_down_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoePrefillParams<'_>,
    res: &crate::families::moe::MoePrefillResolution,
    path2_m_total: usize,
    total_slots: usize,
    force_mq4_grouped_fp16: bool,
    coalesced_verify: bool,
) -> Result<(), DispatchError> {
    reject_fp8_moe(&p.dtypes)?;
    check_qt53_prefill_fallback(p)?;
    let (n, k_top) = (p.batch_size, p.k_top);
    let (down_m, down_k) = (p.down_m, p.down_k);
    let out_target = p.routed_out.unwrap_or(p.x_batch);
    // ── Down projection ───────────────────────────────────────────────
    if res.use_path2 {
        // Path 2: grouped-WMMA-GEMM + non-atomic combine via inverse_perm.
        dispatch_grouped_gemm_role(
            gpu,
            GroupedGemmRole::Down,
            p.n_exp,
            p.dtypes.routed_down,
            p.expert_dtype_tags,
            p.expert_down_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            p.rot_batch,
            p.y_down_grouped,
            down_m,
            down_k,
            1, /* x_row_div */
            path2_m_total,
            total_slots,
            force_mq4_grouped_fp16,
            res.use_paro_i8,
            res.use_paro_i8_k8,
            coalesced_verify
                && crate::families::moe::routed_host_mapped_projections(p.routed_experts).1,
        )?;
        // Publication is deferred to the explicit combine stage.
    } else if res.down_path0 {
        // Path 0: gfx9* wave64 — residual-scaled atomic GEMV (MQ4 only;
        // MQ6/Paro never reach here — their admit predicates require WMMA).
        let down_result = match p.dtypes.routed_down {
            DType::MQ4G256 => hip!(
                gpu.gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.topk_weights,
                    p.rot_batch,
                    out_target,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )
            ),
            DType::MQ4G128V2 => {
                // Resolution refuses this before any launch
                // (`MoePrefillResolution::qt53_refused`); kept as a typed arm.
                return Err(crate::families::moe::qt53_fallback_unsupported(
                    "qt53-routed-down-fallback-path0-unsupported",
                ));
            }
            _other => {
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "prefill-down-path0-dtype",
                    arch: "",
                    quant: "other",
                });
            }
        };
        down_result?;
        // The mutation fence is a distinct stage after all down writes.
    } else {
        // Path 1: atomic-free expanded GEMV write + combine.
        // Mixed Path1 (issue 2): when `expert_dtype_tags` exists, use the
        // mixed batched down launcher (`gemv_mixed_moe_down_k8_indexed_batched_expanded`);
        // do not representative-dispatch tagged layers. Otherwise dispatch
        // the uniform dtype's native batched expanded kernel or reject loudly.
        if let Some(tags) = p.expert_dtype_tags {
            hip!(gpu.gemv_mixed_moe_down_k8_indexed_batched_expanded(
                p.expert_down_ptrs,
                tags,
                p.topk_indices,
                p.rot_batch,
                p.down_expanded,
                down_m,
                down_k,
                k_top,
                n,
            ))?;
        } else {
            let down_result = match p.dtypes.routed_down {
                DType::MQ4G256 => hip!(gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.rot_batch,
                    p.down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )),
                DType::MQ4G256V2 => hip!(gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.rot_batch,
                    p.down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )),
                DType::MQ4G128V2 => hip!(gpu
                    .gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
                        p.expert_down_ptrs,
                        p.topk_indices,
                        p.rot_batch,
                        p.down_expanded,
                        down_m,
                        down_k,
                        n,
                        p.n_exp,
                        k_top,
                    )),
                DType::MQ5G256 => hip!(gpu.gemv_hfq5g256_moe_down_k8_indexed_batched_expanded(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.rot_batch,
                    p.down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )),
                DType::MQ6G256 => hip!(gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.rot_batch,
                    p.down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )),
                DType::MQ6G256V2 => hip!(gpu.gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.rot_batch,
                    p.down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )),
                DType::MFP4G32E8 => hip!(gpu.gemv_mfp4g32_e8_moe_down_k8_indexed_batched_expanded(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.rot_batch,
                    p.down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )),
                DType::ParoQ4G128 => hip!(gpu.gemv_paro_q4g128_moe_down_k8_indexed_batched(
                    p.expert_down_ptrs,
                    p.topk_indices,
                    p.rot_batch,
                    p.down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )),
                _other => {
                    return Err(DispatchError::UnsupportedVariant {
                        family: "moe",
                        variant: "prefill-down-path1-dtype",
                        arch: "",
                        quant: "other",
                    });
                }
            };
            down_result?;
        }
        // Publication is deferred to the explicit combine stage.
    }
    Ok(())
}

fn prefill_down_unscatter_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoePrefillParams<'_>,
    total_slots: usize,
) -> Result<(), DispatchError> {
    reject_fp8_moe(&p.dtypes)?;
    // Grouped GEMM's atomic bucket order is rank-local. Restore canonical
    // flat slot order before this buffer participates in an EP gather.
    hip!(gpu.moe_down_unscatter_k8(
        p.y_down_grouped,
        p.inverse_perm,
        p.down_expanded,
        p.down_m,
        total_slots,
    ))?;
    Ok(())
}

fn prefill_combine_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoePrefillParams<'_>,
    res: &crate::families::moe::MoePrefillResolution,
    target: &GpuTensor,
    canonical_slot_order: bool,
) -> Result<(), DispatchError> {
    res.check_generic_route(None)?;
    if canonical_slot_order {
        if res.down_path0 {
            return Err(DispatchError::Hip(
                "canonical EP combine requires expanded expert outputs".into(),
            ));
        }
        hip!(gpu.moe_down_combine_k8_batched(
            p.down_expanded,
            p.topk_weights,
            target,
            p.down_m,
            p.k_top,
            p.batch_size,
        ))?;
    } else if res.use_path2 {
        hip!(gpu.moe_down_combine_grouped_k8(
            p.y_down_grouped,
            p.inverse_perm,
            p.topk_weights,
            target,
            p.down_m,
            p.k_top,
            p.batch_size,
        ))?;
    } else if !res.down_path0 {
        hip!(gpu.moe_down_combine_k8_batched(
            p.down_expanded,
            p.topk_weights,
            target,
            p.down_m,
            p.k_top,
            p.batch_size,
        ))?;
    }
    Ok(())
}

// ───────────────────── opt-in expert IU4 prefill stages ─────────────────────
//
// Launchers of the verdict `moe_program::decide_expert_iu4` admitted for a call
// (`HIPFIRE_MOE_SYM_IU4`, default ON by capability; `=0` kills it): the scatter and every router/shared stage
// are the existing ones; the stages below replace gate/up, the SwiGLU/rotation,
// down and combine of the qt53 F32 fallback. They run only from an
// `ExpertIu4Decision::Admitted` plan, re-check that the operands still describe
// the proven call, and never fall back: a mismatch is an error, not a reroute.

/// BF16 `[rows, width]` view over the front of `src`'s storage (the grouped
/// scratch the F32 fallback uses as F32 rows). Aliases `src`; never freed.
fn expert_iu4_bf16_view(
    src: &GpuTensor,
    rows: usize,
    width: usize,
    what: &str,
) -> Result<GpuTensor, DispatchError> {
    let bytes = moe_program::bf16_grouped_view_bytes(rows, width).ok_or_else(|| {
        DispatchError::Hip(format!("expert IU4 {what}: BF16 view extent overflows"))
    })?;
    if src.buf.size() < bytes {
        return Err(DispatchError::Hip(format!(
            "expert IU4 {what}: buffer holds {} bytes, the BF16 [{rows}, {width}] view needs {bytes}",
            src.buf.size()
        )));
    }
    Ok(GpuTensor {
        buf: src.buf.byte_view(0, bytes),
        shape: vec![rows, width],
        dtype: DType::BF16,
    })
}

fn expert_iu4_require_plan(
    p: &crate::families::moe::MoePrefillParams<'_>,
    plan: &moe_program::ExpertIu4Plan,
) -> Result<(), DispatchError> {
    reject_fp8_moe(&p.dtypes)?;
    check_qt53_prefill_fallback(p)?;
    if !plan.matches(p) {
        return Err(DispatchError::Hip(
            "sealed moe: expert IU4 stage operands differ from the admitted plan".into(),
        ));
    }
    Ok(())
}

/// F32 `x_norm_batch` (not `x_rot_batch`) -> FWHT256/C2 A4 -> symmetric gate/up
/// with the SwiGLU epilogue into BF16 `y_gate_up_grouped [grouped_rows, I]`.
fn prefill_expert_iu4_gate_up_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoePrefillParams<'_>,
    plan: &moe_program::ExpertIu4Plan,
) -> Result<(), DispatchError> {
    expert_iu4_require_plan(p, plan)?;
    let y = expert_iu4_bf16_view(
        p.y_gate_up_grouped,
        plan.grouped_rows,
        plan.intermediate,
        "gate/up rows",
    )?;
    let xq = hip!(gpu.qwen4_moe_rotate256_i4(p.x_norm_batch, plan.hidden, plan.rows))?;
    hip!(gpu.gemm_qwen4_moe_gate_up_silu_iu4_sym(
        p.expert_gate_up_ptrs,
        p.expert_tile_ids,
        p.sorted_slot_index,
        &xq,
        &y,
        2 * plan.intermediate,
        plan.hidden,
        plan.k_top,
        plan.grouped_rows,
        plan.rows,
        plan.host_mapped,
    ))
}

/// BF16 grouped SwiGLU rows -> FWHT128/C2 A4 (compact flat-slot record) ->
/// symmetric down into BF16 `y_down_grouped [grouped_rows, H]`. The PM producer writes the grouped-position plane the
/// byte-identical `..._gp` down entry reads (`Gpu::qwen4_moe_rotate128_i4_gp`).
fn prefill_expert_iu4_down_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoePrefillParams<'_>,
    plan: &moe_program::ExpertIu4Plan,
) -> Result<(), DispatchError> {
    expert_iu4_require_plan(p, plan)?;
    let h = expert_iu4_bf16_view(
        p.y_gate_up_grouped,
        plan.grouped_rows,
        plan.intermediate,
        "gate/up rows",
    )?;
    let y = expert_iu4_bf16_view(
        p.y_down_grouped,
        plan.grouped_rows,
        plan.hidden,
        "down rows",
    )?;
    let xq = hip!(gpu.qwen4_moe_rotate128_i4_gp(
        &h,
        p.sorted_slot_index,
        plan.intermediate,
        plan.grouped_rows,
        plan.total_slots,
    ))?;
    hip!(gpu.gemm_qwen4_moe_down_iu4_sym(
        p.expert_down_ptrs,
        p.expert_tile_ids,
        p.sorted_slot_index,
        &xq,
        &y,
        plan.hidden,
        plan.intermediate,
        1,
        plan.grouped_rows,
        plan.total_slots,
        plan.host_mapped,
    ))
}

/// BF16 grouped down rows widened to F32 and combined in ascending top-k slot
/// order (`fma(w, v, acc)`), then `+=` into the target the Path-2 combine
/// (`moe_down_combine_grouped_k8`) would use. Runs the v2 twin only when
/// `moe_program::expert_iu4_combine_v2_selected` admits it (`HIPFIRE_MOE_COMBINE_V2`
/// resolved on: unset on at exact gfx1201, `0` kills; v2 bundle present, supported shape); otherwise the v1 twin. Its position relative to the shared
/// expert and the mutation fence is the lowering's, unchanged.
fn prefill_expert_iu4_combine_stage(
    gpu: &mut Gpu,
    p: &crate::families::moe::MoePrefillParams<'_>,
    plan: &moe_program::ExpertIu4Plan,
) -> Result<(), DispatchError> {
    expert_iu4_require_plan(p, plan)?;
    let y = expert_iu4_bf16_view(
        p.y_down_grouped,
        plan.grouped_rows,
        plan.hidden,
        "down rows",
    )?;
    let target = p.routed_out.unwrap_or(p.x_batch);
    if moe_program::expert_iu4_combine_v2_selected(gpu, plan) {
        return hip!(gpu.moe_down_combine_grouped_bf16in_v2_pm(
            &y,
            p.inverse_perm,
            p.topk_weights,
            target,
            plan.hidden,
            plan.k_top,
            plan.rows,
            plan.grouped_rows,
        ));
    }
    hip!(gpu.moe_down_combine_grouped_bf16in_pm(
        &y,
        p.inverse_perm,
        p.topk_weights,
        target,
        plan.hidden,
        plan.k_top,
        plan.rows,
        plan.grouped_rows,
    ))
}

fn prefill_mutation_fence_stage() -> Result<(), DispatchError> {
    #[cfg(feature = "serve-fault-inject")]
    if crate::pipeline::sealed_moe::take_fault_after_expert_mutation() {
        return Err(DispatchError::Hip(
            "injected fault after sealed MoE expert mutation before combine publication".into(),
        ));
    }
    Ok(())
}
