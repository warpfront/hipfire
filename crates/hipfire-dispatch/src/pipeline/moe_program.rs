// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Private executable programs for sealed MoE calls.
//!
//! [`SealedMoeCall`](super::sealed_moe::SealedMoeCall) remains the complete-call
//! authority. This module only lowers an already sealed call into borrowed,
//! fixed-capacity [`Step`](super::steps::Step) values. The opaque operation
//! handles below are intentionally impossible to construct outside this
//! lowerer: a caller can submit a complete sealed call or nothing.

use super::steps::Step;
use crate::context::DispatchCtx;
use crate::cpu_exec;
use crate::cpu_split::CpuSplitExecutor;
use crate::families::moe::{
    CpuWideVerifyScope, MoeNormalization, MoeParams, MoePrefillParams, MoePrefillResolution,
    MoeQ8RouterPolicy, MoeRecipe, MoeResolution, MoeRouteCapability, MOE_BIASED_SIGMOID_MAX_K,
    NARROW_VERIFY_MAX_ROWS,
};
use crate::pipeline::sealed_moe::{MoeProtocol, MoeRouterInput, SealedMoeCall};
use crate::types::DispatchError;
use rdna_compute::{Gpu, GpuTensor};
use smallvec::SmallVec;
use std::cell::{Cell, OnceCell, RefCell};
use std::sync::OnceLock;

const INLINE_STEP_CAPACITY: usize = 32;

/// One operation in a sealed MoE program. The lowerer alone chooses and orders
/// stages; external callers cannot create the matching sealed operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeStage {
    Normalize,
    InputBasis,
    GateSide,
    RouterProjection,
    Route,
    /// Balanced EP ownership ([`MoeRowGather::assigns_owners`]): rewrites the
    /// layer's live expert tables and slot owners from the routed ids; lowered
    /// only immediately after the stage that wrote them, before any routed
    /// table reader, and only when the installed gather asks for it.
    OwnerAssign,
    SharedGateUp,
    SharedActivation,
    SharedDown,
    Scatter,
    GateUp,
    Unscatter,
    Activation,
    Down,
    MutationFence,
    /// Expert-parallel copy-only exchange of peer-owned expanded rows
    /// ([`MoeParams::ep_row_gather`]); lowered only between `MutationFence`
    /// and `Combine` of an indexed decode that supplies a gather.
    RowGather,
    /// Owner-routed EP chunk exchange ([`MoePrefillParams::ep_tile_exchange`]): the
    /// masked copy of the tile ids (non-owned experts' tiles set to the -1 sentinel
    /// the grouped symmetric GEMMs exit on); lowered only right after the
    /// scatter of a grouped symmetric prefill that supplies an exchange.
    TileMask,
    /// Expert-parallel copy-only exchange of peer-owned grouped down rows
    /// ([`MoePrefillParams::ep_tile_exchange`]); lowered only between
    /// `MutationFence` and `Combine` of that same prefill.
    TileExchange,
    Combine,
    HostExperts,
    CpuSplice,
    /// Eager CPU-cold/GPU-hot split ([`MoeDecodeSelection::cpu_split`],
    /// [`MoePrefillSelection::cpu_split`]): the one input-readiness wait.
    /// Downloads the ORIGINAL route and rotated activation, validates every
    /// cold span and uploads the executor's separate masked ids. Lowered only
    /// after the router and the rotated input exist and strictly before the
    /// routed GPU tail, which is the only reader of the masked ids.
    PrepareSplit,
    /// Runs the cold slots' unweighted CPU arithmetic while the masked GPU
    /// tail executes, then queues their F32 slot rows behind the GPU down on
    /// the same stream. Lowered only after the routed `Down` and before the
    /// canonical `Combine`; it never completes or synchronizes the stream.
    JoinSplit,
}

fn router_shared_fuse_allowed(
    skip_routing: bool,
    use_gpu_topk: bool,
    exact_wave64_router: bool,
    batch_size: usize,
    skip_shared: bool,
    smi: usize,
    shared_down_dtype: rdna_compute::DType,
    shared_down_has_awq: bool,
    env_enabled: bool,
) -> bool {
    !skip_routing
        && use_gpu_topk
        && exact_wave64_router
        && batch_size == 1
        && !skip_shared
        && smi == 512
        && shared_down_dtype == rdna_compute::DType::MQ4G256
        && !shared_down_has_awq
        && env_enabled
}

/// Admission of the A3B decode norm fold (`HIPFIRE_A3B_NORM_FOLD`), static at
/// seal: the plain norm + FWHT before the exact MQ4G256V2 hidden-2048 router /
/// selector / shared gate / shared up quartet runs inside that quartet's launch
/// ([`Gpu::fused_rmsnorm_proj4_mq4g256v2_fold`]). Exact gfx1201, rows == 1;
/// anything else keeps the separate norm launch.
#[allow(clippy::too_many_arguments)]
fn a3b_norm_fold_admitted(
    is_gfx1201: bool,
    params: &MoeParams<'_>,
    resolution: &MoeResolution,
    route: Option<MoeRouteCapability>,
    router_shared_fuse: bool,
    skip_routing: bool,
    cpu_split: bool,
) -> bool {
    use rdna_compute::DType::MQ4G256V2;
    if !is_gfx1201 || params.batch_size != 1 || skip_routing || cpu_split {
        return false;
    }
    let Some(shared) = params.shared.as_ref() else {
        return false;
    };
    let w = |w: &crate::families::gemv::WeightRef<'_>, m: usize| {
        w.dtype == MQ4G256V2
            && w.awq_scale.is_none()
            && w.m == m
            && w.k == rdna_compute::norm_fold::NORM_FOLD_HIDDEN
    };
    matches!(params.recipe, MoeRecipe::SoftmaxGatedShared { .. })
        && matches!(params.normalization, MoeNormalization::RmsNorm { .. })
        && params.x_rot_prerotated
        && params.hidden == rdna_compute::norm_fold::NORM_FOLD_HIDDEN
        && !params.skip_shared
        && !params.cpu_deferred_experts
        && params.ep_mode == crate::families::moe::MoeEpMode::None
        && params.ep_row_gather.is_none()
        && route.is_none()
        && !router_shared_fuse
        && resolution.needs_x_rot_local
        && resolution.gate_fusable_mq4v2
        && !resolution.gate_fusable
        && w(&params.router, 256)
        && w(&shared.weights.selector, 1)
        && w(&shared.weights.gate, 512)
        && w(&shared.weights.up, 512)
        && super::norm_fold::env_enabled()
}

/// Opt-in gfx1100 decode down+last-combine admission. The fused kernel is a
/// K=512 (two-group) body, so any other expert-down K (`mi`) stays on the
/// expanded down + separate combine path. `env_enabled` is evaluated last and
/// only when every structural condition holds.
#[allow(clippy::too_many_arguments)]
fn down_last_combine_allowed(
    ep_none: bool,
    is_gfx1100: bool,
    batch_size: usize,
    topk: usize,
    has_expert_dtype_tags: bool,
    routed_down: rdna_compute::DType,
    down_k: usize,
    env_enabled: impl FnOnce() -> bool,
) -> bool {
    ep_none
        && is_gfx1100
        && batch_size == 1
        && topk == 8
        && !has_expert_dtype_tags
        && routed_down == rdna_compute::DType::MQ4G256
        && down_k == 512
        && env_enabled()
}

/// Static decode choices resolved while sealing, before the call is published.
/// This record contains no borrowed call data and no executable instruction
/// sequence.
#[derive(Clone, Copy, Debug)]
pub(super) struct MoeDecodeSelection {
    pub(super) resolution: MoeResolution,
    pub(super) router_shared_fuse: bool,
    pub(super) exact_wave64_router: bool,
    pub(super) wave64_router: bool,
    pub(super) down_last_combine: bool,
    pub(super) route: Option<crate::families::moe::MoeRouteCapability>,
    pub(super) ninepath_d3: bool,
    pub(super) ninepath_d4: bool,
    pub(super) ninepath_mq3l: bool,
    pub(super) ninepath_mq4v2: bool,
    /// qt53 (`MQ4G128V2`) routed down on the typed fallback: the nine-path
    /// kernel also owns the routed activation (silu*up + FWHT128), so the
    /// Activation stage launches nothing. Implies `ninepath_d4`.
    pub(super) ninepath_mq4g128v2: bool,
    pub(super) ninepath_mq6v2: bool,
    /// The routed expert FFN runs in the arch's CPU splice after this program:
    /// no routed GPU stage may be lowered (see
    /// [`MoeParams::cpu_deferred_experts`](crate::families::moe::MoeParams::cpu_deferred_experts)).
    pub(super) cpu_deferred_experts: bool,
    /// Distinct eager CPU-cold/GPU-hot split mode (see
    /// [`RoutedExpertWeights::cpu_split`](crate::families::moe::RoutedExpertWeights::cpu_split)).
    /// Mutually exclusive with `cpu_deferred_experts`: the routed GPU tail is
    /// lowered over the executor's masked ids and the canonical combine reads
    /// the original route.
    pub(super) cpu_split: bool,
    /// The split runs the CP-wait order: the canonical combine is queued behind
    /// a command-processor wait before the CPU arithmetic, and `JoinSplit`
    /// (last) releases it. Implies `cpu_split`.
    pub(super) cpwait: bool,
    /// A3B decode norm fold admitted for this layer (exact gfx1201, MQ4G256V2
    /// hidden-2048 router + shared quartet after a plain norm, rows == 1, kill
    /// switch `HIPFIRE_A3B_NORM_FOLD` not `0`); see [`a3b_norm_fold_admitted`].
    pub(super) norm_fold: bool,
}

/// Static grouped-prefill choices resolved while sealing. It deliberately
/// contains only scalar choices and the family-owned resolution record.
pub(super) struct MoePrefillSelection {
    pub(super) resolution: MoePrefillResolution,
    pub(super) path2_m_total: usize,
    pub(super) force_mq4_grouped_fp16: bool,
    pub(super) coalesced_verify: bool,
    /// Exact architecture-selected route, or the generic grouped path.
    pub(super) route: Option<crate::families::moe::MoeRouteCapability>,
    /// The routed expert FFN runs in the arch's CPU splice after this program:
    /// no routed GPU stage may be lowered (see
    /// [`MoePrefillParams::cpu_deferred_experts`](crate::families::moe::MoePrefillParams::cpu_deferred_experts)).
    pub(super) cpu_deferred_experts: bool,
    /// Distinct eager CPU-cold/GPU-hot split mode for a window of at most
    /// [`NARROW_VERIFY_MAX_ROWS`] rows (see [`MoeDecodeSelection::cpu_split`]).
    /// Mutually exclusive with `cpu_deferred_experts`.
    pub(super) cpu_split: bool,
    /// The split runs the CP-wait order: the canonical combine is queued behind
    /// a command-processor wait before the CPU arithmetic, and `JoinSplit`
    /// (last) releases it. Implies `cpu_split`.
    pub(super) cpwait: bool,
}

impl MoePrefillSelection {
    /// FP8 owns its grouped arm; it never enables the generic WMMA Path 2.
    pub(super) fn uses_grouped_rows(&self) -> bool {
        self.route.is_some_and(MoeRouteCapability::is_fp8_grouped) || self.resolution.use_path2
    }
}

/// The call's static kernel selection. Neither variant stores a step list or
/// a borrow of its enclosing call.
pub(super) enum MoeKernelSelection {
    Decode(MoeDecodeSelection),
    Prefill(MoePrefillSelection),
}

/// Host route produced by the CPU top-k fallback. The vectors are allocated
/// only when that route is selected, and remain execution-local.
pub(super) struct HostRoute {
    pub(super) indices: Vec<usize>,
    pub(super) weights: Vec<f32>,
}

/// Lifetime-free route metadata retained only for this execution. Tensor
/// identities are host reference identities, not device addresses.
#[derive(Clone, Copy)]
struct RouteStamp {
    invocation: u64,
    protocol: MoeProtocol,
    router: MoeRouterInput,
    n_experts: usize,
    k_top: usize,
    indices_tensor: usize,
    weights_tensor: usize,
    adopted_from: Option<u64>,
}

/// Pure part of the router-to-shared IU4 token carry: the sealed prefill router
/// is the MQ4G256V2 projection of a `SoftmaxGatedShared` recipe, the route is
/// produced here (not adopted), and its K equals the shared selector's K, so
/// the router's quantized `x_rot_batch` is exactly the one the shared group
/// reads. The remaining truths (flag, all-three-MQ4 shared, rows, arch,
/// capture/replay) are owned by `prefill_shared_prepared_iu4_admitted`.
fn router_carries_shared_iu4(
    recipe: MoeRecipe,
    router_dtype: rdna_compute::DType,
    router_k: usize,
    shared_selector_k: Option<usize>,
    adopted: bool,
) -> bool {
    matches!(recipe, MoeRecipe::SoftmaxGatedShared { .. })
        && !adopted
        && router_dtype == rdna_compute::DType::MQ4G256V2
        && shared_selector_k == Some(router_k)
}

/// Pure part of the exact fused input-basis producer
/// (`HIPFIRE_PREFILL_IU4_PRODUCER_FUSED`): the shared gate is MQ4G256V2 over the
/// router's K, K a nonzero multiple of 256 (the IU4 consumers' floor), and the
/// router/shared IU4 carry already holds (so the next two stages are the router
/// and the shared gate/up, which read the same `x_rot_batch`, and nothing else
/// can reserve or quantize into the sidecar slot before they consume it).
fn input_basis_producer_fused_admitted(
    flag: bool,
    carries: bool,
    gate_dtype: rdna_compute::DType,
    gate_k: usize,
    router_k: usize,
) -> bool {
    flag && carries
        && gate_dtype == rdna_compute::DType::MQ4G256V2
        && gate_k != 0
        && gate_k % 256 == 0
        && gate_k == router_k
}

/// Invocation-local mutable state. It borrows the sealed call but never gets
/// stored by that call, so borrowed Steps cannot form a self-referential owner.
pub(super) struct MoeStepState<'a> {
    call: &'a SealedMoeCall<'a>,
    host_route: OnceCell<HostRoute>,
    route_stamp: Cell<Option<RouteStamp>>,
    /// The IU4 activation token the router stage quantized for the shared
    /// group: written only by the router stage, taken (consumed, never
    /// re-quantized) by the shared gate/up stage of the same invocation.
    shared_iu4: RefCell<Option<rdna_compute::scratch::Int4MmqPrepared>>,
    /// The IU4 sidecar the fused shared activation wrote for the shared down
    /// (`HIPFIRE_PREFILL_IU4_PRODUCER_FUSED`): written only by the generic
    /// shared activation stage, taken by the shared down stage that the lowering
    /// places immediately after it (the fused route is not admitted for
    /// `shared_after_combine`, whose routed stages would sit in between).
    shared_down_iu4: RefCell<Option<rdna_compute::scratch::Int4MmqPrepared>>,
    shared_gate: Option<GpuTensor>,
    shared_up: Option<GpuTensor>,
    /// H8a: the routed target's zero fill was absorbed; the qt44 combine
    /// writes it from +0.0 and clears this.
    initial_zero: Cell<bool>,
    /// H4 level 3: the paired HC write the shared down may carry (taken by the
    /// stage), and whether it did.
    hc_write: RefCell<Option<super::qt44_qt53_prefill::HcSharedDown>>,
    hc_folded: Cell<bool>,
    /// `HIPFIRE_QWEN4_HC_ROW_FOLD`: the shared down ran the whole tail as the
    /// HC row fold, including the next HC read's norm.
    hc_row_folded: Cell<bool>,
    /// Where this invocation is in the split transaction (never read when the
    /// call is not in the split mode).
    split_phase: Cell<SplitPhase>,
    /// The post-combine parity reference re-runs the routed GPU tail over the
    /// ORIGINAL ids; while set, the masked view is not applied.
    split_reference: Cell<bool>,
    /// The call's one expert IU4 verdict ([`MoeStepState::expert_iu4_decision`]),
    /// decided at the first routed stage and never re-decided.
    expert_iu4: OnceCell<ExpertIu4Decision>,
    /// L4: the grouped prefill normalize stage already ran the RMSNorm, the F32 normalized row, the
    /// rotated F32 basis and the shared IU4 token as one exact C8 PM producer
    /// ([`SealedMoeOp::fold_prefill_norm_into_input_basis`]); the input-basis stage has nothing
    /// left to produce.
    norm_basis_folded: Cell<bool>,
    /// `HIPFIRE_QWEN4_MOE_FRONT_FUSE`: the plain (`execute`) program may run the fused MoE
    /// front producer; every other lowering (the S1 layer-major prepare, the slowlink
    /// tiles, ...) leaves this false and runs the unfused producers.
    front_allowed: Cell<bool>,
    /// What the fused front producer left for the shared gate/up and routed gate/up stages
    /// ([`super::qt44_qt53_prefill::input_basis_front`]): written by the input-basis stage,
    /// borrowed by the shared gate/up, taken by the routed gate/up of the same invocation.
    front: RefCell<Option<super::qt44_qt53_prefill::MoeFront>>,
    /// The normalize stage skipped the plain norm because the gate-side launch
    /// recomputes it ([`a3b_norm_fold_admitted`]).
    norm_fold_active: Cell<bool>,
}

/// A call abandoned between `Combine` (which queued the CP wait) and `JoinSplit`
/// (which releases it) leaves the stream gated: any later synchronize hangs.
/// Every exit of the program, including an error return or a panic, therefore
/// releases the wait with the abort value before anything can drain the device.
/// The combine consumed garbage, so the executor is poisoned and the request
/// must be discarded by whoever handles the error.
impl Drop for MoeStepState<'_> {
    fn drop(&mut self) {
        if self.split_phase.get() == SplitPhase::Armed {
            if let Some(executor) = self.split_executor() {
                executor.abort_armed();
            }
        }
    }
}

impl<'a> MoeStepState<'a> {
    fn new(call: &'a SealedMoeCall<'a>) -> Self {
        let (shared_gate, shared_up) = match call.decode_params() {
            Some(params) => match params.shared.as_ref() {
                Some(shared) if shared.intermediate != 0 => (
                    Some(super::slice_moe_f32_view(
                        shared.gate_out,
                        0,
                        shared.intermediate,
                    )),
                    Some(super::slice_moe_f32_view(
                        shared.up_out,
                        0,
                        shared.intermediate,
                    )),
                ),
                _ => (None, None),
            },
            None => (None, None),
        };
        let route_stamp = call
            .attached_route_stamp()
            .or_else(|| call.prebound_route_stamp())
            .map(
                |(
                    invocation,
                    protocol,
                    router,
                    n_experts,
                    k_top,
                    indices_tensor,
                    weights_tensor,
                    adopted_from,
                )| RouteStamp {
                    invocation,
                    protocol,
                    router,
                    n_experts,
                    k_top,
                    indices_tensor,
                    weights_tensor,
                    adopted_from,
                },
            );
        Self {
            call,
            host_route: OnceCell::new(),
            route_stamp: Cell::new(route_stamp),
            shared_iu4: RefCell::new(None),
            shared_down_iu4: RefCell::new(None),
            shared_gate,
            shared_up,
            initial_zero: Cell::new(false),
            hc_write: RefCell::new(None),
            hc_folded: Cell::new(false),
            hc_row_folded: Cell::new(false),
            split_phase: Cell::new(SplitPhase::Idle),
            split_reference: Cell::new(false),
            expert_iu4: OnceCell::new(),
            norm_basis_folded: Cell::new(false),
            front_allowed: Cell::new(false),
            front: RefCell::new(None),
            norm_fold_active: Cell::new(false),
        }
    }

    fn dispatch_ctx(&self) -> &DispatchCtx {
        self.call.dispatch_ctx()
    }

    fn decode_parts(&self) -> Result<(&MoeParams<'_>, &MoeDecodeSelection), DispatchError> {
        let params = self.call.decode_params().ok_or_else(|| {
            DispatchError::Hip("sealed moe: operation requires decode operands".into())
        })?;
        let selection = match self.call.kernel_selection() {
            MoeKernelSelection::Decode(selection) => selection,
            MoeKernelSelection::Prefill(_) => {
                return Err(DispatchError::Hip(
                    "sealed moe: decode operation received prefill selection".into(),
                ));
            }
        };
        Ok((params, selection))
    }

    fn prefill_parts(
        &self,
    ) -> Result<(&MoePrefillParams<'_>, &MoePrefillSelection), DispatchError> {
        let params = self.call.prefill_params().ok_or_else(|| {
            DispatchError::Hip("sealed moe: operation requires prefill operands".into())
        })?;
        let selection = match self.call.kernel_selection() {
            MoeKernelSelection::Prefill(selection) => selection,
            MoeKernelSelection::Decode(_) => {
                return Err(DispatchError::Hip(
                    "sealed moe: prefill operation received decode selection".into(),
                ));
            }
        };
        Ok((params, selection))
    }

    /// Whether the qt44 combine of this call can take over `clear`'s zero
    /// fill (`HIPFIRE_QWEN4_MOE_COMBINE_ZINIT`, exact gfx1151 / gfx1201):
    /// the combine writes every element `clear` zeroes, into that very
    /// tensor, nothing else writes it before the combine, and nothing is
    /// recorded or captured (a retained tape keeps the clear's shape).
    fn absorbs_clear(&self, gpu: &Gpu, clear: &super::layer_ops::ClearOp<'_>) -> bool {
        if !gpu.flags.qwen4_moe_combine_zinit_enabled()
            || gpu.replay.is_recording()
            || gpu.replay.retained_body_active()
            || gpu.graphs.capture_mode
        {
            return false;
        }
        let Ok((params, selection)) = self.prefill_parts() else {
            return false;
        };
        let target = params.routed_out.unwrap_or(params.x_batch);
        selection.route.is_some_and(MoeRouteCapability::is_sealed_grouped)
            // A CPU-deferred call omits the combine that would consume the
            // absorbed fill, so it must never absorb one.
            && !selection.cpu_deferred_experts
            && !selection.cpu_split
            && matches!(params.prelude.normalization, MoeNormalization::Provided)
            && super::qt44_qt53_prefill::combine_initial_zero_applies(
                gpu,
                params,
                selection.uses_grouped_rows(),
            )
            && clear.tensor.buf.as_ptr() == target.buf.as_ptr()
            && Some(clear.elements) == params.batch_size.checked_mul(params.down_m)
    }

    fn shared_views(&self) -> Result<(&GpuTensor, &GpuTensor), DispatchError> {
        match (&self.shared_gate, &self.shared_up) {
            (Some(gate), Some(up)) => Ok((gate, up)),
            _ => Err(DispatchError::Hip(
                "sealed moe: shared operation has no decode shared-expert views".into(),
            )),
        }
    }

    fn install_route_stamp(
        &self,
        n_experts: usize,
        k_top: usize,
        adopted_from: Option<u64>,
    ) -> Result<(), DispatchError> {
        if self.route_stamp.get().is_some() {
            return Err(DispatchError::Hip(
                "sealed moe: route producer ran twice in one invocation".into(),
            ));
        }
        let (indices_tensor, weights_tensor) = match self.call.protocol() {
            MoeProtocol::IndexedDecode => {
                let params = self.decode_parts()?.0;
                (
                    std::ptr::addr_of!(*params.topk_indices) as usize,
                    std::ptr::addr_of!(*params.topk_weights) as usize,
                )
            }
            MoeProtocol::GroupedPrefill => {
                let params = self.prefill_parts()?.0;
                (
                    std::ptr::addr_of!(*params.topk_indices) as usize,
                    std::ptr::addr_of!(*params.topk_weights) as usize,
                )
            }
        };
        self.route_stamp.set(Some(RouteStamp {
            invocation: self.call.invocation(),
            protocol: self.call.protocol(),
            router: self.call.router_input(),
            n_experts,
            k_top,
            indices_tensor,
            weights_tensor,
            adopted_from,
        }));
        Ok(())
    }

    fn require_route_ready(&self) -> Result<(), DispatchError> {
        let stamp = self.route_stamp.get().ok_or_else(|| {
            DispatchError::Hip("sealed moe: route consumer ran before route production".into())
        })?;
        let (n_experts, k_top, indices_tensor, weights_tensor) = match self.call.protocol() {
            MoeProtocol::IndexedDecode => {
                let params = self.decode_parts()?.0;
                (
                    params.n_exp,
                    params.k,
                    std::ptr::addr_of!(*params.topk_indices) as usize,
                    std::ptr::addr_of!(*params.topk_weights) as usize,
                )
            }
            MoeProtocol::GroupedPrefill => {
                let params = self.prefill_parts()?.0;
                (
                    params.n_exp,
                    params.k_top,
                    std::ptr::addr_of!(*params.topk_indices) as usize,
                    std::ptr::addr_of!(*params.topk_weights) as usize,
                )
            }
        };
        if stamp.invocation != self.call.invocation()
            || stamp.protocol != self.call.protocol()
            || stamp.router != self.call.router_input()
            || stamp.n_experts != n_experts
            || stamp.k_top != k_top
            || stamp.indices_tensor != indices_tensor
            || stamp.weights_tensor != weights_tensor
        {
            return Err(DispatchError::Hip(
                "sealed moe: route readiness stamp does not match this invocation".into(),
            ));
        }
        Ok(())
    }

    fn require_split_phase(&self, want: SplitPhase, stage: &str) -> Result<(), DispatchError> {
        let have = self.split_phase.get();
        if have != want {
            return Err(split_error(format!(
                "{stage} ran out of order (phase {have:?}, expected {want:?})"
            )));
        }
        Ok(())
    }

    /// Whether this call runs the CP-wait order (combine queued before the CPU
    /// arithmetic, join last). Never true outside the split mode.
    fn split_cpwait(&self) -> bool {
        if let Ok((_, selection)) = self.decode_parts() {
            return selection.cpu_split && selection.cpwait;
        }
        if let Ok((_, selection)) = self.prefill_parts() {
            return selection.cpu_split && selection.cpwait;
        }
        false
    }

    /// The split executor this call reaches, if any.
    fn split_executor(&self) -> Option<&CpuSplitExecutor> {
        if let Ok((params, selection)) = self.decode_parts() {
            return selection
                .cpu_split
                .then(|| decode_split_executor(params).ok())
                .flatten();
        }
        if let Ok((params, selection)) = self.prefill_parts() {
            return selection
                .cpu_split
                .then(|| prefill_split_executor(params).ok())
                .flatten();
        }
        None
    }

    /// The decode params the routed GPU tail reads. `None` outside the split
    /// mode, so the caller keeps the original call.
    ///
    /// Normal split run (`Prepared` phase): the original call with only
    /// `topk_indices` replaced by the executor's masked ids.
    ///
    /// Post-combine reference rerun: the original call (original ids) with
    /// `skip_shared = true`. The rerun replays gate/up, activation and down,
    /// and the fused routed activation sigmoids the shared selector scalar IN
    /// PLACE; the original shared stage already consumed that scalar, so a
    /// second pass would apply `sigmoid(sigmoid(x))` before the shared down.
    /// `skip_shared` keeps the shared activation out of the routed launch.
    /// Every other consumer (combine, shared, router) never takes this view.
    fn masked_decode<'p>(
        &self,
        params: &MoeParams<'p>,
        selection: &MoeDecodeSelection,
    ) -> Result<Option<MoeParams<'p>>, DispatchError> {
        if !selection.cpu_split {
            return Ok(None);
        }
        if self.split_reference.get() {
            let mut view = *params;
            view.skip_shared = true;
            return Ok(Some(view));
        }
        self.require_split_phase(SplitPhase::Prepared, "the routed GPU tail")?;
        let executor = decode_split_executor(params)?;
        Ok(Some(params.with_topk_indices(executor.masked_indices())))
    }

    /// Prefill counterpart of [`Self::masked_decode`] (the grouped scatter
    /// reads it too). The reference rerun view clears `prelude.shared`: the
    /// fused unscatter activation would otherwise sigmoid the shared selector
    /// scalar in place a second time.
    fn masked_prefill<'p>(
        &self,
        params: &MoePrefillParams<'p>,
        selection: &MoePrefillSelection,
    ) -> Result<Option<MoePrefillParams<'p>>, DispatchError> {
        if !selection.cpu_split {
            return Ok(None);
        }
        if self.split_reference.get() {
            let mut view = *params;
            view.prelude.shared = None;
            return Ok(Some(view));
        }
        self.require_split_phase(SplitPhase::Prepared, "the routed GPU tail")?;
        let executor = prefill_split_executor(params)?;
        Ok(Some(params.with_topk_indices(executor.masked_indices())))
    }
}

/// Opaque operand for one granular sealed MoE operation. The private state
/// field is the API-privacy boundary: external code cannot construct one,
/// clone one, or extract its program/state.
///
/// ```compile_fail
/// use hipfire_dispatch::pipeline::SealedMoeOp;
///
/// fn forge<'a>() -> SealedMoeOp<'a> {
///     SealedMoeOp { state: todo!() }
/// }
/// ```
pub struct SealedMoeOp<'a> {
    state: &'a MoeStepState<'a>,
}

impl<'a> SealedMoeOp<'a> {
    fn new(state: &'a MoeStepState<'a>) -> Self {
        Self { state }
    }

    pub(super) fn execute_stage(
        &self,
        gpu: &mut Gpu,
        stage: MoeStage,
    ) -> Result<(), DispatchError> {
        match stage {
            MoeStage::Normalize => self.normalize(gpu),
            MoeStage::InputBasis => self.input_basis(gpu),
            MoeStage::GateSide => self.gate_side(gpu),
            MoeStage::RouterProjection => self.router_projection(gpu),
            MoeStage::Route => self.route(gpu),
            MoeStage::OwnerAssign => self.owner_assign(gpu),
            MoeStage::SharedGateUp => self.shared_gate_up(gpu),
            MoeStage::SharedActivation => self.shared_activation(gpu),
            MoeStage::SharedDown => self.shared_down(gpu),
            MoeStage::Scatter => self.scatter(gpu),
            MoeStage::GateUp => self.gate_up(gpu),
            MoeStage::Unscatter => self.unscatter(gpu),
            MoeStage::Activation => self.activation(gpu),
            MoeStage::Down => self.down(gpu),
            MoeStage::MutationFence => self.mutation_fence(gpu),
            MoeStage::RowGather => self.row_gather(gpu),
            MoeStage::TileMask => self.tile_mask(gpu),
            MoeStage::TileExchange => self.tile_exchange(gpu),
            MoeStage::Combine => self.combine(gpu),
            MoeStage::HostExperts => self.host_experts(gpu),
            MoeStage::CpuSplice => self.cpu_splice(gpu),
            MoeStage::PrepareSplit => self.prepare_split(gpu),
            MoeStage::JoinSplit => self.join_split(gpu),
        }
    }

    pub(super) fn normalize(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => {
                let (params, selection) = self.state.decode_parts()?;
                if let MoeNormalization::RmsNorm {
                    weight,
                    plain_out,
                    eps,
                } = &params.normalization
                {
                    if params.x_rot_prerotated {
                        if selection.norm_fold && super::norm_fold::launchable(gpu) {
                            // The gate-side launch recomputes this norm itself.
                            self.state.norm_fold_active.set(true);
                            return Ok(());
                        }
                        gpu.fused_rmsnorm_rotate_mq(
                            params.x_residual,
                            weight,
                            params.x_rot_local,
                            params.hidden,
                            *eps,
                        )
                    } else {
                        gpu.rmsnorm_f32(params.x_residual, weight, plain_out, *eps)
                    }
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                }
            }
            MoeProtocol::GroupedPrefill => {
                let (params, selection) = self.state.prefill_parts()?;
                if let MoeNormalization::RmsNorm {
                    weight,
                    plain_out,
                    eps,
                } = &params.prelude.normalization
                {
                    if !self.fold_prefill_norm_into_input_basis(
                        gpu, params, selection, weight, plain_out, *eps,
                    )? {
                        gpu.rmsnorm_batched(
                            params.x_batch,
                            weight,
                            plain_out,
                            params.batch_size,
                            params.gate_up_k,
                            *eps,
                        )
                        .map_err(|e| DispatchError::Hip(e.to_string()))?;
                    }
                }
            }
        }
        Ok(())
    }

    /// L4: the grouped prefill's standalone `rmsnorm_f32` folded into the exact C8 PM input-basis
    /// producer that consumes its output (`HIPFIRE_PREFILL_IU4_C8_PM` nested under the F3 producer
    /// fusion). Taken only where [`Self::input_basis`] would run the producer-fused arm for this call
    /// (same selection, carry and shared-gate admission) and the PM rotate itself is admitted for the
    /// operands: one launch writes the normalized F32 row into `x_norm_batch` (the router, shared
    /// selector and routed-expert producers read it), the rotated F32 basis into `x_rot_batch`, and
    /// the sidecar token the shared gate/up consumes, with the bytes of `rmsnorm_batched` followed by
    /// the exact rotate. Returns whether the fold ran; `false` launched nothing and leaves the
    /// unfused chain (the standalone norm, then [`Self::input_basis`]) to run unchanged.
    fn fold_prefill_norm_into_input_basis(
        &self,
        gpu: &mut Gpu,
        params: &MoePrefillParams<'_>,
        selection: &MoePrefillSelection,
        norm_weight: &GpuTensor,
        plain_out: &GpuTensor,
        eps: f32,
    ) -> Result<bool, DispatchError> {
        if selection.cpu_split
            || selection
                .route
                .is_some_and(MoeRouteCapability::is_sealed_grouped)
            || selection.resolution.paro_mode
        {
            return Ok(false);
        }
        let Some(shared) = params.prelude.shared.as_ref() else {
            return Ok(false);
        };
        let weight = &shared.weights.gate;
        if !input_basis_producer_fused_admitted(
            super::prefill_iu4_producer_fused_enabled(gpu),
            self.prefill_router_carries_shared_iu4(gpu, params),
            weight.dtype,
            weight.k,
            params.prelude.router.k,
        ) || self.state.shared_iu4.borrow().is_some()
        {
            return Ok(false);
        }
        let (k, n) = (weight.k, params.batch_size);
        if k != params.gate_up_k || plain_out.buf.as_ptr() != params.x_norm_batch.buf.as_ptr() {
            return Ok(false);
        }
        if !gpu.iu4_c8_pm_norm_out_admitted(
            params.x_batch,
            norm_weight,
            weight.awq_scale,
            params.x_norm_batch,
            params.x_rot_batch,
            k,
            n,
        ) {
            return Ok(false);
        }
        let reservation = gpu
            .reserve_int4_mmq(k, n)
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
        let token = gpu
            .fused_rmsnorm_out_mq_rotate_i4_c8_pm(
                params.x_batch,
                norm_weight,
                weight.awq_scale,
                params.x_norm_batch,
                Some(params.x_rot_batch),
                reservation,
                k,
                eps,
                n,
            )
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
        *self.state.shared_iu4.borrow_mut() = Some(token);
        self.state.norm_basis_folded.set(true);
        Ok(true)
    }

    pub(super) fn input_basis(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if let Ok((params, selection)) = self.state.decode_parts() {
            return super::decode_input_basis_stage(gpu, params, selection.resolution);
        }
        let (params, selection) = self.state.prefill_parts()?;
        if self.state.norm_basis_folded.get() {
            // The normalize stage produced the F32 basis and the token (L4 fold).
            return Ok(());
        }
        if selection.cpu_split {
            // The CPU computes from the F32 rotated basis. The grouped stage may
            // skip it when the GPU gate/up rotates to F16 itself, so the split
            // always writes it (the same rotation the stage would have run).
            return gpu
                .rotate_x_mq_batched(
                    params.x_norm_batch,
                    params.x_rot_batch,
                    params.gate_up_k,
                    params.batch_size,
                )
                .map_err(|e| DispatchError::Hip(e.to_string()));
        }
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            if self.state.front_allowed.get()
                && !selection.cpu_deferred_experts
                && super::qt44_qt53_prefill::front_applies(
                    gpu,
                    params,
                    selection.uses_grouped_rows(),
                )
            {
                // One read of `mixed`: the routed IU4 sidecar and the F16 X of the
                // router / shared projections, bytes of the unfused producers; the dead
                // F32 rotation is not run.
                let front = super::qt44_qt53_prefill::input_basis_front(gpu, params)?;
                *self.state.front.borrow_mut() = Some(front);
                return Ok(());
            }
            return super::qt44_qt53_prefill::input_basis(
                gpu,
                params,
                selection.uses_grouped_rows(),
            );
        }
        let input_weight = params
            .prelude
            .shared
            .as_ref()
            .map(|shared| &shared.weights.gate);
        if !selection.resolution.paro_mode {
            if let Some(weight) = input_weight {
                if input_basis_producer_fused_admitted(
                    super::prefill_iu4_producer_fused_enabled(gpu),
                    self.prefill_router_carries_shared_iu4(gpu, params),
                    weight.dtype,
                    weight.k,
                    params.prelude.router.k,
                ) {
                    if self.state.shared_iu4.borrow().is_some() {
                        return Err(DispatchError::Hip(
                            "sealed moe: shared IU4 token already live before the input-basis producer"
                                .into(),
                        ));
                    }
                    // Exact8 rotate + `block_i4_128` producer on the same input and AWQ
                    // as the incumbent rotate; it also writes the F32 `x_rot_batch`
                    // every other reader (and the router tap) sees.
                    let (k, n) = (weight.k, params.batch_size);
                    let reservation = gpu
                        .reserve_int4_mmq(k, n)
                        .map_err(|e| DispatchError::Hip(e.to_string()))?;
                    let token = gpu
                        .rotate_x_mq_i4_gfx12_exact_batched(
                            params.x_norm_batch,
                            weight.awq_scale,
                            Some(params.x_rot_batch),
                            reservation,
                            k,
                            n,
                        )
                        .map_err(|e| DispatchError::Hip(e.to_string()))?;
                    *self.state.shared_iu4.borrow_mut() = Some(token);
                    return Ok(());
                }
            }
        }
        if selection.resolution.paro_mode {
            super::prefill_input_basis_stage(gpu, params)?;
        } else if let Some(weight) = input_weight {
            if let Some(awq) = weight.awq_scale {
                gpu.rotate_x_mq_awq_batched(
                    params.x_norm_batch,
                    awq,
                    params.x_rot_batch,
                    weight.k,
                    params.batch_size,
                )
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            } else {
                gpu.rotate_x_mq_batched(
                    params.x_norm_batch,
                    params.x_rot_batch,
                    weight.k,
                    params.batch_size,
                )
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            }
        } else if let Some((gate_up, _)) = params.routed_experts.get(0) {
            // No-shared recipes still need the routed expert's input basis.
            gpu.rotate_x_mq_batched(
                params.x_norm_batch,
                params.x_rot_batch,
                gate_up.k,
                params.batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
        }
        Ok(())
    }

    pub(super) fn gate_side(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let (params, selection) = self.state.decode_parts()?;
        if params.recipe == MoeRecipe::SigmoidRoutedNoShared {
            return self.router_projection(gpu);
        }
        let (shared_gate, shared_up) = self.state.shared_views()?;
        if self.state.norm_fold_active.get() {
            return self.a3b_norm_fold_gate_side(gpu, params, selection, shared_gate, shared_up);
        }
        super::decode_gate_side_stage(
            self.state.dispatch_ctx(),
            gpu,
            params,
            selection.resolution,
            selection
                .resolution
                .needs_x_rot_local
                .then_some(params.x_rot_local),
            shared_gate,
            shared_up,
            selection.route,
        )
    }

    /// Gate-side stage of a folded call: the norm + rotate and the router /
    /// selector / shared gate / shared up rows of the incumbent
    /// `fused_qkvza_mq4g256v2` launch as one dispatch over the same buffers.
    fn a3b_norm_fold_gate_side(
        &self,
        gpu: &mut Gpu,
        params: &MoeParams<'_>,
        selection: &MoeDecodeSelection,
        shared_gate: &GpuTensor,
        shared_up: &GpuTensor,
    ) -> Result<(), DispatchError> {
        super::reject_fp8_moe_unless_routed(&params.dtypes, selection.route)?;
        super::check_qt53_decode_fallback(params, selection.route)?;
        let MoeNormalization::RmsNorm { weight, eps, .. } = &params.normalization else {
            return Err(DispatchError::Hip(
                "a3b norm fold requires a plain RMSNorm normalization".into(),
            ));
        };
        let shared = params
            .shared
            .as_ref()
            .ok_or_else(|| DispatchError::Hip("a3b norm fold requires the shared expert".into()))?;
        let args = rdna_compute::norm_fold::NormFoldArgs {
            x: params.x_residual,
            norm_weight: weight,
            x_rot: params.x_rot_local,
            weights: [
                params.router.buf,
                shared.weights.selector.buf,
                shared.weights.gate.buf,
                shared.weights.up.buf,
            ],
            outs: [params.router_logits, shared.scalar, shared_gate, shared_up],
            m: [
                params.router.m,
                shared.weights.selector.m,
                shared.weights.gate.m,
                shared.weights.up.m,
            ],
            eps: *eps,
        };
        let grid = rdna_compute::norm_fold::norm_fold_grid(
            args.total_rows(),
            super::norm_fold::MOE_GATE_ROWS_PER_WAVE,
        );
        gpu.fused_rmsnorm_proj4_mq4g256v2_fold(&args, grid)
            .map_err(|e| DispatchError::Hip(e.to_string()))
    }

    pub(super) fn router_projection(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => {
                let (params, _) = self.state.decode_parts()?;
                if matches!(params.recipe, MoeRecipe::SoftmaxGatedShared { .. }) {
                    return self.gate_side(gpu);
                }
                static GEMV_ROUTER: OnceLock<crate::families::gemv::GemvFamily> = OnceLock::new();
                GEMV_ROUTER
                    .get_or_init(crate::families::gemv::GemvFamily::new)
                    .run_auto(
                        self.state.dispatch_ctx(),
                        gpu,
                        &params.router,
                        params.x_norm,
                        params.router_logits,
                    )
                    .map_err(|e| DispatchError::Hip(e.to_string()))
            }
            MoeProtocol::GroupedPrefill => {
                let (params, selection) = self.state.prefill_parts()?;
                if selection
                    .route
                    .is_some_and(MoeRouteCapability::is_sealed_grouped)
                {
                    return super::qt44_qt53_prefill::router_projection(gpu, params);
                }
                let carry = self.prefill_router_carries_shared_iu4(gpu, params);
                let prelude = &params.prelude;
                let x = match prelude.router.dtype {
                    rdna_compute::DType::Q8_0 | rdna_compute::DType::F32 => params.x_norm_batch,
                    _ => params.x_rot_batch,
                };
                let result = match prelude.q8_router_policy {
                    _ if carry => self.router_projection_carrying_shared_iu4(gpu, params),
                    MoeQ8RouterPolicy::FreshFp16Wmma { enabled, scratch }
                        if prelude.router.dtype == rdna_compute::DType::Q8_0 =>
                    {
                        crate::families::gemm::run_q8_projection_fresh(
                            gpu,
                            &prelude.router.buf,
                            x,
                            prelude.router_logits,
                            prelude.router.m,
                            prelude.router.k,
                            params.batch_size,
                            scratch,
                            enabled,
                        )
                        .map_err(|e| DispatchError::Hip(e.to_string()))
                    }
                    _ => {
                        static GEMM_ROUTER: OnceLock<crate::families::gemm::GemmFamily> =
                            OnceLock::new();
                        let key = match prelude.router.dtype {
                            rdna_compute::DType::Q8_0 => {
                                crate::types::KernelKey::GemmQ8_0BatchedChunked
                            }
                            rdna_compute::DType::MQ4G256 => crate::types::KernelKey::GemmHfq4G256,
                            rdna_compute::DType::MQ4G256V2 => {
                                crate::types::KernelKey::GemmMq4G256V2
                            }
                            rdna_compute::DType::F32 => crate::types::KernelKey::GemmF32Batched,
                            _ => {
                                return Err(DispatchError::UnsupportedVariant {
                                    family: "moe",
                                    variant: "prefill-router-dtype",
                                    arch: "",
                                    quant: "",
                                });
                            }
                        };
                        GEMM_ROUTER
                            .get_or_init(crate::families::gemm::GemmFamily::new)
                            .run_key(
                                key,
                                self.state.dispatch_ctx(),
                                gpu,
                                &crate::families::gemm::GemmParams {
                                    w: &prelude.router,
                                    x,
                                    y: prelude.router_logits,
                                    batch_size: params.batch_size,
                                },
                            )
                    }
                };
                result?;
                if matches!(params.recipe, MoeRecipe::SoftmaxGatedShared { .. }) {
                    super::dump_hidden_localize(
                        gpu,
                        prelude.router_logits,
                        params.batch_size,
                        0,
                        params.n_exp,
                        0,
                        "router_b",
                    );
                }
                Ok(())
            }
        }
    }

    /// Whether the generic prefill router stage hands its IU4 activation token to
    /// the shared gate/up (and so the input-basis stage may produce it): the pure
    /// carry predicate over the sealed call plus the GPU admission. Pure over the
    /// call and device state, so the input-basis and router stages agree.
    fn prefill_router_carries_shared_iu4(&self, gpu: &Gpu, params: &MoePrefillParams<'_>) -> bool {
        let adopted = matches!(
            params.prelude.route,
            super::sealed_moe::PrefillRouteMode::AdoptRoot { .. }
        ) || self
            .state
            .route_stamp
            .get()
            .is_some_and(|stamp| stamp.adopted_from.is_some());
        router_carries_shared_iu4(
            params.recipe,
            params.prelude.router.dtype,
            params.prelude.router.k,
            params
                .prelude
                .shared
                .as_ref()
                .map(|shared| shared.weights.selector.k),
            adopted,
        ) && super::prefill_shared_prepared_iu4_admitted(gpu, params)
    }

    /// Router projection that quantizes `x_rot_batch` once and keeps the typed
    /// token for the shared gate/up stage. Taps the calibration capture exactly
    /// where the incumbent `GemmFamily` MQ4G256V2 path taps it; the GPU entry
    /// itself never taps. Admission was already established by the caller, so a
    /// declined launch is an error (never a silent re-tap and re-run).
    fn router_projection_carrying_shared_iu4(
        &self,
        gpu: &mut Gpu,
        params: &MoePrefillParams<'_>,
    ) -> Result<(), DispatchError> {
        let router = &params.prelude.router;
        let (k, n) = (router.k, params.batch_size);
        // A token already in the slot is the exact fused input-basis producer's
        // (this stage is the only other writer, and runs once per invocation).
        // It is consumed here with no quantize. The producer stage runs
        // immediately before, so a superseded token is a bug and an `Err`
        // before the tap or any launch, never a re-quantize.
        let pending = self.state.shared_iu4.borrow_mut().take();
        if let Some(token) = pending {
            gpu.int4_mmq_prepared_ptr(&token, k, n)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            gpu.maybe_capture_activation(router.buf, params.x_rot_batch, n, k);
            gpu.gemm_mq4g256v2_prepared_iu4(
                router.buf,
                &token,
                params.prelude.router_logits,
                router.m,
                k,
                n,
            )
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
            *self.state.shared_iu4.borrow_mut() = Some(token);
            return Ok(());
        }
        gpu.maybe_capture_activation(router.buf, params.x_rot_batch, params.batch_size, router.k);
        let token = gpu
            .try_gemm_mq4g256v2_shared_router_iu4(
                router.buf,
                params.x_rot_batch,
                params.prelude.router_logits,
                router.m,
                router.k,
                params.batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string()))?
            .ok_or_else(|| {
                DispatchError::Hip(
                    "sealed moe: router shared IU4 admitted but the GPU declined".into(),
                )
            })?;
        *self.state.shared_iu4.borrow_mut() = Some(token);
        Ok(())
    }

    pub(super) fn route(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        match self.state.call.protocol() {
            MoeProtocol::GroupedPrefill => {
                if let Some(stamp) = self.state.route_stamp.get() {
                    if stamp.adopted_from.is_some() {
                        return Ok(());
                    }
                    return Err(DispatchError::Hip(
                        "sealed moe: route producer ran twice in one invocation".into(),
                    ));
                }
                let (params, _) = self.state.prefill_parts()?;
                let scores = params.prelude.router_scores;
                let receipt = super::sealed_moe::produce_prefill_route(
                    self.state.call,
                    gpu,
                    scores,
                    params.prelude.norm_topk_prob,
                )?;
                self.state.call.validate_route_receipt(&receipt)?;
                if let super::sealed_moe::PrefillRouteMode::ProduceRoot { slot } =
                    params.prelude.route
                {
                    let proof = self
                        .state
                        .call
                        .prefill_route_producer_proof_for_receipt(&receipt)?;
                    slot.set(Some(proof));
                }
                self.state.install_route_stamp(
                    params.n_exp,
                    params.k_top,
                    receipt.adopted_from(),
                )?;
                Ok(())
            }
            MoeProtocol::IndexedDecode => {
                let (params, selection) = self.state.decode_parts()?;
                if matches!(
                    self.state.call.router_input(),
                    MoeRouterInput::PrecomputedSoftmaxTopK | MoeRouterInput::PrecomputedSigmoidTopK
                ) {
                    if self.state.route_stamp.get().is_none() {
                        return Err(DispatchError::Hip(
                            "sealed moe: adopted route is not ready before route consumers".into(),
                        ));
                    }
                    return Ok(());
                }
                if selection.resolution.use_gpu_topk {
                    if self.state.call.router_input() == MoeRouterInput::SigmoidTopK {
                        #[cfg(feature = "deltanet")]
                        {
                            // Precomputed scores are already activated.
                            if !params.router_scores_precomputed() {
                                gpu.sigmoid_f32(params.router_logits)
                                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                            }
                            match params.selection_bias {
                                Some(bias) => gpu.deepseek4_moe_topk_bias_aware_f32(
                                    params.router_logits,
                                    bias.bias,
                                    params.topk_indices,
                                    params.topk_weights,
                                    params.n_exp as i32,
                                    params.k as i32,
                                    bias.route_scale,
                                ),
                                None => gpu.moe_topk_renorm_k8(
                                    params.router_logits,
                                    params.topk_indices,
                                    params.topk_weights,
                                    params.n_exp,
                                    params.norm_topk_prob,
                                ),
                            }
                            .map_err(|e| DispatchError::Hip(e.to_string()))?;
                        }
                        #[cfg(not(feature = "deltanet"))]
                        return Err(DispatchError::UnsupportedVariant {
                            family: "moe",
                            variant: "sigmoid-router-requires-deltanet",
                            arch: "",
                            quant: "",
                        });
                    } else {
                        let (shared_gate, shared_up) = self.state.shared_views()?;
                        super::decode_route_gpu_stage(
                            gpu,
                            params,
                            shared_gate,
                            shared_up,
                            selection.router_shared_fuse,
                            selection.exact_wave64_router,
                            selection.wave64_router,
                            selection.route,
                        )?;
                    }
                } else {
                    super::decode_route_cpu_stage(
                        gpu,
                        params,
                        &self.state.host_route,
                        self.state.call.router_input(),
                    )?;
                }
                self.state.install_route_stamp(params.n_exp, params.k, None)
            }
        }
    }

    pub(super) fn shared_gate_up(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if let Ok((params, selection)) = self.state.decode_parts() {
            let (shared_gate, shared_up) = self.state.shared_views()?;
            return super::decode_gate_side_stage(
                self.state.dispatch_ctx(),
                gpu,
                params,
                selection.resolution,
                selection
                    .resolution
                    .needs_x_rot_local
                    .then_some(params.x_rot_local),
                shared_gate,
                shared_up,
                selection.route,
            );
        }
        let (params, selection) = self.state.prefill_parts()?;
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            let front = self.state.front.borrow();
            return super::qt44_qt53_prefill::shared_gate_up_front(
                gpu,
                params,
                selection.uses_grouped_rows(),
                front.as_ref(),
            );
        }
        // The router stage's IU4 token (if any) is consumed by exactly this
        // stage; it never outlives the shared gate/up of its invocation.
        let prepared = self.state.shared_iu4.borrow_mut().take();
        super::prefill_shared_gate_up_stage(
            self.state.dispatch_ctx(),
            gpu,
            params,
            prepared.as_ref(),
        )
    }

    pub(super) fn shared_activation(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if let Ok((params, selection)) = self.state.decode_parts() {
            let (shared_gate, shared_up) = self.state.shared_views()?;
            let target = if params.ep_mode == crate::families::moe::MoeEpMode::RootRoutedPartial {
                params.x_residual
            } else {
                params.routed_out.unwrap_or(params.x_residual)
            };
            return super::decode_shared_down_stage(
                self.state.dispatch_ctx(),
                gpu,
                params,
                shared_gate,
                shared_up,
                target,
                selection.router_shared_fuse,
                selection.route,
            );
        }
        let (params, selection) = self.state.prefill_parts()?;
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            return super::qt44_qt53_prefill::shared_activation(
                gpu,
                params,
                selection.uses_grouped_rows(),
            );
        }
        if self.state.shared_down_iu4.borrow().is_some() {
            return Err(DispatchError::Hip(
                "sealed moe: shared down IU4 token already live before the shared activation"
                    .into(),
            ));
        }
        let token = super::prefill_shared_activation_stage(gpu, params)?;
        *self.state.shared_down_iu4.borrow_mut() = token;
        Ok(())
    }

    pub(super) fn shared_down(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if let Ok((params, selection)) = self.state.decode_parts() {
            let (shared_gate, shared_up) = self.state.shared_views()?;
            let target = if params.ep_mode == crate::families::moe::MoeEpMode::RootRoutedPartial {
                params.x_residual
            } else {
                params.routed_out.unwrap_or(params.x_residual)
            };
            let dump = if params.skip_shared {
                None
            } else {
                super::moe_dump::shared_begin(
                    gpu,
                    params.layer_idx as usize,
                    target,
                    params.hidden,
                )?
            };
            super::decode_shared_down_stage(
                self.state.dispatch_ctx(),
                gpu,
                params,
                shared_gate,
                shared_up,
                target,
                selection.router_shared_fuse,
                selection.route,
            )?;
            return super::moe_dump::shared_end(
                gpu,
                dump,
                target,
                params.hidden,
                !std::ptr::eq(target, params.x_residual),
            );
        }
        let (params, selection) = self.state.prefill_parts()?;
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            // H4 (`HIPFIRE_QWEN4_HC_FUSE` >= 3): the paired HC write rides the
            // shared down's GEMM epilogue when this call admits it.
            if let Some(hc) = self.state.hc_write.take() {
                let row_fold = hc.row_fold.is_some();
                if super::qt44_qt53_prefill::shared_down_hc(
                    gpu,
                    params,
                    &hc,
                    selection.uses_grouped_rows(),
                )? {
                    self.state.hc_folded.set(true);
                    self.state.hc_row_folded.set(row_fold);
                    return Ok(());
                }
            }
            return super::qt44_qt53_prefill::shared_down(
                gpu,
                params,
                selection.uses_grouped_rows(),
            );
        }
        let prepared = self.state.shared_down_iu4.borrow_mut().take();
        super::prefill_shared_down_stage(self.state.dispatch_ctx(), gpu, params, prepared.as_ref())
    }

    pub(super) fn scatter(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let (params, selection) = self.state.prefill_parts()?;
        self.state.require_route_ready()?;
        let view = self.state.masked_prefill(params, selection)?;
        let params = view.as_ref().unwrap_or(params);
        if !selection.uses_grouped_rows() {
            return Err(DispatchError::Hip(
                "sealed moe: scatter is only valid for grouped prefill path 2".into(),
            ));
        }
        // The expert IU4 verdict (default off: unbound, returns here without a
        // read) is decided once, before the first routed launch; every later stage
        // reads the same cached verdict.
        self.state.expert_iu4_decision(gpu, params, selection);
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            return super::qt44_qt53_prefill::scatter(gpu, params, selection.path2_m_total);
        }
        super::prefill_scatter_stage(gpu, params, selection.path2_m_total)
    }

    pub(super) fn gate_up(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        self.state.require_route_ready()?;
        if let Ok((params, selection)) = self.state.decode_parts() {
            let view = self.state.masked_decode(params, selection)?;
            let params = view.as_ref().unwrap_or(params);
            return super::decode_gate_up_stage(
                gpu,
                params,
                selection.resolution,
                selection
                    .resolution
                    .needs_x_rot_local
                    .then_some(params.x_rot_local)
                    .unwrap_or(params.x_norm),
                selection.ninepath_d3,
                selection.route,
            );
        }
        let (params, selection) = self.state.prefill_parts()?;
        let view = self.state.masked_prefill(params, selection)?;
        let params = view.as_ref().unwrap_or(params);
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            let front = self.state.front.borrow_mut().take();
            return super::qt44_qt53_prefill::gate_up_front(
                gpu,
                params,
                selection.uses_grouped_rows(),
                selection.path2_m_total,
                front,
            );
        }
        if let ExpertIu4Decision::Admitted(plan) =
            self.state.expert_iu4_decision(gpu, params, selection)
        {
            return super::prefill_expert_iu4_gate_up_stage(gpu, params, &plan);
        }
        super::prefill_gate_up_stage(
            gpu,
            params,
            &selection.resolution,
            selection.path2_m_total,
            selection.force_mq4_grouped_fp16,
            selection.coalesced_verify,
        )
    }

    pub(super) fn unscatter(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let (params, selection) = self.state.prefill_parts()?;
        let view = self.state.masked_prefill(params, selection)?;
        let params = view.as_ref().unwrap_or(params);
        if !selection.uses_grouped_rows() {
            return Err(DispatchError::Hip(
                "sealed moe: unscatter is only valid for grouped prefill path 2".into(),
            ));
        }
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            return super::qt44_qt53_prefill::unscatter(gpu, params, selection.path2_m_total);
        }
        if matches!(
            self.state.expert_iu4_decision(gpu, params, selection),
            ExpertIu4Decision::Admitted(_)
        ) {
            // The gate/up epilogue already wrote the SwiGLU rows grouped.
            return Ok(());
        }
        super::prefill_gate_up_unscatter_stage(gpu, params, selection.path2_m_total)
    }

    pub(super) fn activation(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if let Ok((params, selection)) = self.state.decode_parts() {
            if selection.ninepath_mq4g128v2 {
                // The fused down kernel forms silu(gate)*up and the qt53
                // FWHT128 rotation itself; nothing is owed here.
                return Ok(());
            }
            let view = self.state.masked_decode(params, selection)?;
            let params = view.as_ref().unwrap_or(params);
            return super::decode_activation_stage(
                gpu,
                params,
                selection.resolution,
                selection.route,
                self.state.shared_views().ok(),
            );
        }
        let (params, selection) = self.state.prefill_parts()?;
        let view = self.state.masked_prefill(params, selection)?;
        let params = view.as_ref().unwrap_or(params);
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            return super::qt44_qt53_prefill::activation(
                gpu,
                params,
                selection.uses_grouped_rows(),
            );
        }
        if matches!(
            self.state.expert_iu4_decision(gpu, params, selection),
            ExpertIu4Decision::Admitted(_)
        ) {
            // The SwiGLU is the gate/up epilogue and the rotation is the down A4
            // producer: nothing is owed between them.
            return Ok(());
        }
        let total_slots = params
            .batch_size
            .checked_mul(params.k_top)
            .ok_or_else(|| DispatchError::Hip("sealed moe: prefill slot count overflows".into()))?;
        super::prefill_activation_stage(gpu, params, &selection.resolution, total_slots)
    }

    pub(super) fn down(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if let Ok((params, selection)) = self.state.decode_parts() {
            let view = self.state.masked_decode(params, selection)?;
            let params = view.as_ref().unwrap_or(params);
            let target = params.routed_out.unwrap_or(params.x_residual);
            // The down kernel itself folds into `target` (no expanded rows, no
            // later combine stage) on these routes; observe that boundary here.
            let fused_fold = if params.ep_mode == crate::families::moe::MoeEpMode::RootRoutedPartial
                || selection.route.is_some()
            {
                None
            } else if selection.ninepath_d4 {
                Some("ninepath_d4")
            } else if selection.down_last_combine {
                Some("down_last_combine")
            } else if !params.down_writes_expanded() {
                Some("self_combining_down")
            } else {
                None
            };
            let dump = match fused_fold {
                Some(_) => super::moe_dump::fold_begin(
                    gpu,
                    params.layer_idx as usize,
                    params.hidden,
                    params.x_residual,
                    target,
                )?,
                None => None,
            };
            super::decode_down_stage(
                gpu,
                params,
                selection.resolution,
                target,
                selection.ninepath_d4,
                selection.ninepath_mq3l,
                selection.ninepath_mq4v2,
                selection.ninepath_mq4g128v2,
                selection.ninepath_mq6v2,
                selection.down_last_combine,
                selection.route,
            )?;
            if let Some(path) = fused_fold {
                super::moe_dump::fold_end(
                    gpu,
                    dump,
                    target,
                    params.routed_out.is_some(),
                    &super::moe_dump::FoldInfo {
                        path,
                        ep_mode: &format!("{:?}", params.ep_mode),
                        hidden: params.hidden,
                        slots: params.k,
                        expanded: None,
                        indices: params.topk_indices,
                        weights: params.topk_weights,
                        expanded_absent_reason:
                            "down kernel fused the weighted fold; no expanded rows were written",
                    },
                )?;
            }
            return Ok(());
        }
        let (params, selection) = self.state.prefill_parts()?;
        let view = self.state.masked_prefill(params, selection)?;
        let params = view.as_ref().unwrap_or(params);
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            return super::qt44_qt53_prefill::down(
                gpu,
                params,
                selection.uses_grouped_rows(),
                selection.path2_m_total,
            );
        }
        if let ExpertIu4Decision::Admitted(plan) =
            self.state.expert_iu4_decision(gpu, params, selection)
        {
            return super::prefill_expert_iu4_down_stage(gpu, params, &plan);
        }
        let total_slots = params
            .batch_size
            .checked_mul(params.k_top)
            .ok_or_else(|| DispatchError::Hip("sealed moe: prefill slot count overflows".into()))?;
        super::prefill_down_stage(
            gpu,
            params,
            &selection.resolution,
            selection.path2_m_total,
            total_slots,
            selection.force_mq4_grouped_fp16,
            selection.coalesced_verify,
        )?;
        let compact_ep = matches!(
            &params.prelude.route,
            super::sealed_moe::PrefillRouteMode::ProduceRoot { .. }
                | super::sealed_moe::PrefillRouteMode::AdoptRoot { .. }
        );
        if compact_ep && selection.uses_grouped_rows() {
            super::prefill_down_unscatter_stage(gpu, params, total_slots)?;
        }
        Ok(())
    }

    pub(super) fn combine(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if let Ok((params, selection)) = self.state.decode_parts() {
            if selection.cpu_split {
                return self.split_combine_decode(gpu, params, selection);
            }
            return super::decode_combine_stage(
                gpu,
                params,
                params.routed_out.unwrap_or(params.x_residual),
                selection.route,
            );
        }
        let (params, selection) = self.state.prefill_parts()?;
        if selection.cpu_split {
            return self.split_combine_prefill(gpu, params, selection);
        }
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            // The row fold combines inside its own kernel; only the rank order
            // is owed here, and the absorbed target fill is not (the shared
            // down's rows overwrite the target).
            if self
                .state
                .hc_write
                .borrow()
                .as_ref()
                .is_some_and(|hc| hc.row_fold.is_some())
            {
                self.state.initial_zero.set(false);
                return super::qt44_qt53_prefill::combine_order_only(
                    gpu,
                    params,
                    selection.path2_m_total,
                );
            }
            return super::qt44_qt53_prefill::combine(
                gpu,
                params,
                selection.uses_grouped_rows(),
                selection.path2_m_total,
                self.state.initial_zero.take(),
            );
        }
        if let ExpertIu4Decision::Admitted(plan) =
            self.state.expert_iu4_decision(gpu, params, selection)
        {
            return super::prefill_expert_iu4_combine_stage(gpu, params, &plan);
        }
        let canonical_slot_order = matches!(
            &params.prelude.route,
            super::sealed_moe::PrefillRouteMode::ProduceRoot { .. }
                | super::sealed_moe::PrefillRouteMode::AdoptRoot { .. }
        );
        super::prefill_combine_stage(
            gpu,
            params,
            &selection.resolution,
            params.routed_out.unwrap_or(params.x_batch),
            canonical_slot_order,
        )
    }

    pub(super) fn host_experts(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let (params, _) = self.state.decode_parts()?;
        let route = self.state.host_route.get().ok_or_else(|| {
            DispatchError::Hip("sealed moe: host experts consumed before host route".into())
        })?;
        super::decode_cpu_experts_stage(
            self.state.dispatch_ctx(),
            gpu,
            params,
            &route.indices,
            &route.weights,
        )
    }

    /// The CPU routed-expert splice, executed at the point the routed GPU tail
    /// would have run. The arch declares it (`cpu_deferred_experts`) and supplies
    /// the expert payload through `RoutedExpertWeights::cpu_splice_expert`; the
    /// seam owns the invocation order, the capture/replay refusal inside the
    /// executor and the coverage counters.
    pub(super) fn cpu_splice(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => {
                let (params, _) = self.state.decode_parts()?;
                let gpu_ref: &Gpu = &*gpu;
                cpu_exec::moe_cpu_experts(
                    gpu_ref,
                    params.hidden,
                    params.mi,
                    params.batch_size,
                    params.k,
                    params.x_rot_local,
                    params.topk_indices,
                    params.topk_weights,
                    params.x_residual,
                    params.expert_down_awq_ptrs.is_some(),
                    params.recipe.bf16_round_trip(),
                    |id| {
                        params
                            .routed_experts
                            .cpu_splice_expert(gpu_ref, id)
                            .ok_or_else(|| {
                                DispatchError::Hip(format!(
                                    "moe cpu expert splice: expert {id} has no CPU payload"
                                ))
                            })
                    },
                )
            }
            MoeProtocol::GroupedPrefill => {
                let (params, _) = self.state.prefill_parts()?;
                let gpu_ref: &Gpu = &*gpu;
                // `dim` is the model hidden size, which is exactly the routed down
                // projection's row count: `MoePrefillParams` has no `hidden` field
                // and `down_m` is built as `experts[0].down.m` (its `down_k` is
                // `mi`), matching `moe_cpu_experts`' `(dim, mi)` pair. Do not
                // "correct" this to a hidden field that does not exist.
                // The residual destination is the arch's unconditional
                // `&pbs.x_batch`: a CPU-deferred window never carries `routed_out`
                // (the sealed program refuses that combination), so the
                // `unwrap_or` below is defensive only.
                cpu_exec::moe_cpu_experts(
                    gpu_ref,
                    params.down_m,
                    params.mi,
                    params.batch_size,
                    params.k_top,
                    params.x_rot_batch,
                    params.topk_indices,
                    params.topk_weights,
                    params.routed_out.unwrap_or(params.x_batch),
                    params.expert_down_awq_ptrs.is_some(),
                    params.recipe.bf16_round_trip(),
                    |id| {
                        params
                            .routed_experts
                            .cpu_splice_expert(gpu_ref, id)
                            .ok_or_else(|| {
                                DispatchError::Hip(format!(
                                    "moe cpu expert splice: expert {id} has no CPU payload"
                                ))
                            })
                    },
                )
            }
        }
    }

    /// `PrepareSplit`: the one input-readiness wait of the eager CPU split.
    /// Every refusal (capture, retained/shadow bodies, PM4/AQL, route width,
    /// non-expanded down, shape) is raised before the executor touches any
    /// state; the executor then downloads the ORIGINAL ids/weights and the
    /// rotated activation, completes them, validates every cold span and
    /// uploads the separate masked ids. No routed GPU launch precedes it.
    pub(super) fn prepare_split(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        self.state.require_route_ready()?;
        self.state
            .require_split_phase(SplitPhase::Idle, "PrepareSplit")?;
        match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => {
                let (params, selection) = self.state.decode_parts()?;
                if !selection.cpu_split {
                    return Err(split_error(
                        "PrepareSplit lowered without the split selection",
                    ));
                }
                validate_split_decode(gpu, params, selection)?;
                let executor = decode_split_executor(params)?;
                executor.prepare(
                    &*gpu,
                    params.batch_size,
                    params.x_rot_local,
                    params.topk_indices,
                    params.topk_weights,
                    params.routed_experts,
                )?;
            }
            MoeProtocol::GroupedPrefill => {
                let (params, selection) = self.state.prefill_parts()?;
                if !selection.cpu_split {
                    return Err(split_error(
                        "PrepareSplit lowered without the split selection",
                    ));
                }
                validate_split_prefill_exec(
                    gpu,
                    params,
                    selection,
                    self.state.dispatch_ctx().cpu_wide_verify(),
                )?;
                let executor = prefill_split_executor(params)?;
                executor.prepare(
                    &*gpu,
                    params.batch_size,
                    params.x_rot_batch,
                    params.topk_indices,
                    params.topk_weights,
                    params.routed_experts,
                )?;
            }
        }
        self.state.split_phase.set(SplitPhase::Prepared);
        Ok(())
    }

    /// `JoinSplit`: the cold CPU arithmetic runs on the orchestration thread
    /// while the already queued masked GPU tail executes; the cold F32 slot
    /// rows are then queued on the same stream behind the GPU down (whose
    /// masked slots store zero). Never completes or synchronizes the stream.
    pub(super) fn join_split(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let want = if self.state.split_cpwait() {
            SplitPhase::Armed
        } else {
            SplitPhase::Prepared
        };
        self.state.require_split_phase(want, "JoinSplit")?;
        match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => {
                let (params, selection) = self.state.decode_parts()?;
                if !selection.cpu_split {
                    return Err(split_error("JoinSplit lowered without the split selection"));
                }
                decode_split_executor(params)?.join(
                    &*gpu,
                    params.batch_size,
                    params.down_expanded,
                    params.routed_experts,
                )?;
            }
            MoeProtocol::GroupedPrefill => {
                let (params, selection) = self.state.prefill_parts()?;
                if !selection.cpu_split {
                    return Err(split_error("JoinSplit lowered without the split selection"));
                }
                prefill_split_executor(params)?.join(
                    &*gpu,
                    params.batch_size,
                    params.down_expanded,
                    params.routed_experts,
                )?;
            }
        }
        self.state.split_phase.set(SplitPhase::Joined);
        Ok(())
    }

    /// Whether this sealed call runs the CP-wait order, checked against the
    /// executor the call actually reaches: a selection sealed for one join mode
    /// must never meet an executor of the other.
    fn split_join_mode(
        &self,
        executor: &CpuSplitExecutor,
        sealed: bool,
    ) -> Result<bool, DispatchError> {
        if sealed != executor.cpwait_active() {
            return Err(split_error(
                "the sealed join mode differs from the executor's join mode",
            ));
        }
        Ok(sealed)
    }

    /// CP-wait `Combine`: queue the wait behind the routed tail and hand the
    /// canonical combine the mapped view. From here until `JoinSplit` the
    /// stream is gated: no stage may synchronize it, and a call abandoned in
    /// the `Armed` phase aborts the wait (`MoeStepState`'s `Drop`).
    fn arm_cpwait_combine<'e>(
        &self,
        gpu: &Gpu,
        executor: &'e CpuSplitExecutor,
    ) -> Result<&'e GpuTensor, DispatchError> {
        let view = executor.cpwait_arm(gpu)?;
        self.state.split_phase.set(SplitPhase::Armed);
        Ok(view)
    }

    /// The canonical combine of a split call: the unchanged BF16 combine over
    /// the original ids/weights and the joined slot rows, then (only while the
    /// lab parity record asks for it) the unmasked GPU tail is re-run into the
    /// now-dead `down_expanded`. The reference never re-combines.
    ///
    /// The combine reads [`CpuSplitExecutor::combine_outputs`]: the joined
    /// `down_expanded` (copy return) or the mapped cold-result view (mapped
    /// return, all-cold layers). Nothing else about the combine changes; the
    /// lab reference re-run still targets the `down_expanded` scratch.
    fn split_combine_decode(
        &self,
        gpu: &mut Gpu,
        params: &MoeParams<'_>,
        selection: &MoeDecodeSelection,
    ) -> Result<(), DispatchError> {
        let executor = decode_split_executor(params)?;
        let cpwait = self.split_join_mode(executor, selection.cpwait)?;
        self.state.require_split_phase(
            if cpwait {
                SplitPhase::Prepared
            } else {
                SplitPhase::Joined
            },
            "the canonical combine",
        )?;
        let target = params.routed_out.unwrap_or(params.x_residual);
        let rows = params.batch_size;
        let combine_input = if cpwait {
            self.arm_cpwait_combine(gpu, executor)?
        } else {
            executor.combine_outputs(params.down_expanded)?
        };
        let combine_params = MoeParams {
            down_expanded: combine_input,
            ..*params
        };
        executor.parity_before_combine(&*gpu, rows, combine_input, target)?;
        super::decode_combine_stage(gpu, &combine_params, target, selection.route)?;
        if !cpwait {
            executor.combine_enqueued(&*gpu)?;
        }
        executor.parity_after_combine(&*gpu, rows, target)?;
        if executor.wants_reference() {
            self.split_reference_tail(gpu)?;
            executor.parity_reference(&*gpu, rows, params.down_expanded)?;
        }
        Ok(())
    }

    /// Prefill counterpart of [`Self::split_combine_decode`].
    fn split_combine_prefill(
        &self,
        gpu: &mut Gpu,
        params: &MoePrefillParams<'_>,
        selection: &MoePrefillSelection,
    ) -> Result<(), DispatchError> {
        let executor = prefill_split_executor(params)?;
        let cpwait = self.split_join_mode(executor, selection.cpwait)?;
        self.state.require_split_phase(
            if cpwait {
                SplitPhase::Prepared
            } else {
                SplitPhase::Joined
            },
            "the canonical combine",
        )?;
        // The absorbed zero fill belongs to the BF16-row grouped combine; the
        // split folds F32 slot rows and always accumulates into a filled target.
        if self.state.initial_zero.take() {
            return Err(split_error(
                "the canonical expanded combine cannot consume an absorbed target fill",
            ));
        }
        let target = params.routed_out.unwrap_or(params.x_batch);
        let rows = params.batch_size;
        let combine_input = if cpwait {
            self.arm_cpwait_combine(gpu, executor)?
        } else {
            executor.combine_outputs(params.down_expanded)?
        };
        let combine_params = MoePrefillParams {
            down_expanded: combine_input,
            ..*params
        };
        executor.parity_before_combine(&*gpu, rows, combine_input, target)?;
        super::qt44_qt53_prefill::combine(
            gpu,
            &combine_params,
            selection.uses_grouped_rows(),
            selection.path2_m_total,
            false,
        )?;
        if !cpwait {
            executor.combine_enqueued(&*gpu)?;
        }
        executor.parity_after_combine(&*gpu, rows, target)?;
        if executor.wants_reference() {
            self.split_reference_tail(gpu)?;
            executor.parity_reference(&*gpu, rows, params.down_expanded)?;
        }
        Ok(())
    }

    /// Re-run the SAME routed gate/up, activation and down stages over the
    /// ORIGINAL (unmasked) ids into `down_expanded`. Runs strictly after the
    /// canonical combine consumed the D2 rows and strictly before the shared
    /// down, so it writes only scratch that is dead at that point: routed
    /// `gate_batch`/`up_batch`/`rot_batch`, the grouped scatter/`y_*_grouped`
    /// scratch and `down_expanded`. The shared expert's hidden (`shared.rotated`
    /// and the gate/up outputs) must stay untouched: the fused routed
    /// activation / unscatter launches also activate the shared expert and
    /// sigmoid its selector scalar IN PLACE, which is NOT idempotent (a second
    /// pass yields `sigmoid(sigmoid(x))` before the shared down). The rerun
    /// therefore takes the reference view of [`Self::masked_decode`] /
    /// [`Self::masked_prefill`], which disables the shared side of those
    /// stages. No combine.
    fn split_reference_tail(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        self.state.split_reference.set(true);
        let result = self.split_reference_stages(gpu);
        self.state.split_reference.set(false);
        result
    }

    fn split_reference_stages(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let grouped = match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => false,
            MoeProtocol::GroupedPrefill => self.state.prefill_parts()?.1.uses_grouped_rows(),
        };
        if grouped {
            self.scatter(gpu)?;
        }
        self.gate_up(gpu)?;
        if grouped {
            self.unscatter(gpu)?;
        }
        self.activation(gpu)?;
        self.down(gpu)
    }

    pub(super) fn mutation_fence(&self, _gpu: &mut Gpu) -> Result<(), DispatchError> {
        match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => super::decode_mutation_fence_stage(),
            MoeProtocol::GroupedPrefill => super::prefill_mutation_fence_stage(),
        }
    }

    pub(super) fn owner_assign(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let missing =
            || DispatchError::Hip("sealed moe: owner assignment lowered without a gather".into());
        let overflow =
            || DispatchError::Hip("sealed moe: owner assignment extent overflows".into());
        // The routed ids must exist before ownership is derived from them.
        self.state.require_route_ready()?;
        let (gather, layer_idx, slots, topk_indices) = match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => {
                let (params, _) = self.state.decode_parts()?;
                (
                    params.ep_row_gather.ok_or_else(missing)?,
                    params.layer_idx,
                    params.k,
                    params.topk_indices,
                )
            }
            MoeProtocol::GroupedPrefill => {
                let (params, _) = self.state.prefill_parts()?;
                (
                    params.ep_row_gather.ok_or_else(missing)?,
                    params.ep_layer_idx,
                    params
                        .batch_size
                        .checked_mul(params.k_top)
                        .ok_or_else(overflow)?,
                    params.topk_indices,
                )
            }
        };
        // Like the exchange it serves, the assignment is eager-only.
        if gpu.graphs.capture_mode
            || gpu.graphs.verify.capturing.is_some()
            || gpu.graphs.replay.capturing.is_some()
            || gpu.replay.is_recording()
        {
            return Err(DispatchError::Hip(
                "sealed moe: expert-parallel owner assignment refused while a graph capture or replay record is armed".into(),
            ));
        }
        gather
            .assign_owners(gpu, layer_idx, slots, topk_indices)
            .map_err(|error| {
                DispatchError::Hip(format!("sealed moe: EP owner assignment: {error}"))
            })
    }

    pub(super) fn row_gather(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let missing =
            || DispatchError::Hip("sealed moe: row gather lowered without a gather".into());
        let overflow = || DispatchError::Hip("sealed moe: row gather extent overflows".into());
        // The gather, the layer, the slot count and the producer's buffers, per
        // protocol: one decode row's `k` slots, or a few-row prefill's
        // `batch_size * k_top` slots in slot order.
        let (gather, layer_idx, slots, width, down_expanded, topk_indices) = match self
            .state
            .call
            .protocol()
        {
            MoeProtocol::IndexedDecode => {
                let (params, _) = self.state.decode_parts()?;
                let gather = params.ep_row_gather.ok_or_else(missing)?;
                (
                    gather,
                    params.layer_idx,
                    params.k,
                    params.hidden,
                    params.down_expanded,
                    params.topk_indices,
                )
            }
            MoeProtocol::GroupedPrefill => {
                let (params, selection) = self.state.prefill_parts()?;
                let gather = params.ep_row_gather.ok_or_else(missing)?;
                if !selection
                    .route
                    .is_some_and(MoeRouteCapability::is_sealed_grouped)
                    || !super::qt44_qt53_prefill::writes_slot_ordered_rows(
                        gpu,
                        params,
                        selection.uses_grouped_rows(),
                    )
                {
                    return Err(DispatchError::Hip(
                            "sealed moe: an EP row gather needs the grouped QT44/QT53 down to have written slot-ordered expanded rows".into(),
                        ));
                }
                // The unfused shared down runs after the combine and would
                // overwrite rows a peer may still be reading.
                let shared_out = params.shared_down_out.ok_or_else(|| {
                        DispatchError::Hip(
                            "sealed moe: a prefill EP row gather requires a separate shared-down output".into(),
                        )
                    })?;
                if shared_out.buf.as_ptr() == params.down_expanded.buf.as_ptr() {
                    return Err(DispatchError::Hip(
                        "sealed moe: the shared-down output aliases the gathered expanded rows"
                            .into(),
                    ));
                }
                (
                    gather,
                    params.ep_layer_idx,
                    params
                        .batch_size
                        .checked_mul(params.k_top)
                        .ok_or_else(overflow)?,
                    params.down_m,
                    params.down_expanded,
                    params.topk_indices,
                )
            }
        };
        // The exchange waits on peer devices and a host barrier; neither can
        // be recorded into a graph or retained body. EP is eager-only.
        if gpu.graphs.capture_mode
            || gpu.graphs.verify.capturing.is_some()
            || gpu.graphs.replay.capturing.is_some()
            || gpu.replay.is_recording()
        {
            return Err(DispatchError::Hip(
                "sealed moe: expert-parallel row gather refused while a graph capture or replay record is armed".into(),
            ));
        }
        let rows = slots.checked_mul(width).ok_or_else(overflow)?;
        let down_expanded = super::slice_moe_f32_view(down_expanded, 0, rows);
        gather
            .gather_rows(gpu, layer_idx, slots, &down_expanded, topk_indices)
            .map_err(|error| DispatchError::Hip(format!("sealed moe: EP row gather: {error}")))
    }

    pub(super) fn tile_mask(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let (params, selection) = self.state.prefill_parts()?;
        let tx = params.ep_tile_exchange.ok_or_else(|| {
            DispatchError::Hip("sealed moe: tile mask lowered without a tile exchange".into())
        })?;
        self.state.require_route_ready()?;
        super::qt44_qt53_prefill::tile_mask(gpu, params, tx, selection.path2_m_total)
    }

    pub(super) fn tile_exchange(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let (params, selection) = self.state.prefill_parts()?;
        let tx = params.ep_tile_exchange.ok_or_else(|| {
            DispatchError::Hip("sealed moe: tile exchange lowered without a tile exchange".into())
        })?;
        super::qt44_qt53_prefill::tile_exchange(gpu, params, tx, selection.path2_m_total)
    }
}

/// Append one borrowed instruction without allowing SmallVec to spill.
fn append_step<'a>(
    steps: &mut SmallVec<[Step<'a>; INLINE_STEP_CAPACITY]>,
    step: Step<'a>,
) -> Result<(), DispatchError> {
    if steps.len() >= INLINE_STEP_CAPACITY {
        return Err(DispatchError::Hip(
            "sealed moe: step program exceeds inline capacity".into(),
        ));
    }
    steps.push(step);
    Ok(())
}

fn lower_decode<'a>(
    state: &'a MoeStepState<'a>,
) -> Result<SmallVec<[Step<'a>; INLINE_STEP_CAPACITY]>, DispatchError> {
    let (params, selection) = state.decode_parts()?;
    if selection.cpu_split {
        if let Some(reason) = cpu_split_decode_conflict(
            params,
            selection.route,
            selection.cpu_deferred_experts,
            // The executor-dependent shape check ran at seal and runs again in
            // `validate_split_decode`; lowering needs no device object.
            None,
        ) {
            return Err(split_error(reason));
        }
    }
    let mut steps = SmallVec::new();
    let op = |state: &'a MoeStepState<'a>| SealedMoeOp::new(state);
    append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Normalize))?;
    let adopted = matches!(
        state.call.router_input(),
        MoeRouterInput::PrecomputedSoftmaxTopK | MoeRouterInput::PrecomputedSigmoidTopK
    );
    // Balanced EP ownership: the stage goes right after the stage that writes
    // the routed ids and before every routed-table reader. A selection with no
    // such stage in this program (an adopted route) cannot be balanced.
    let owner_assign = params
        .ep_row_gather
        .is_some_and(|gather| gather.assigns_owners(params.layer_idx));
    match params.recipe {
        MoeRecipe::SoftmaxGatedShared { .. } => {
            if selection.resolution.needs_x_rot_local {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::InputBasis))?;
            }
            if !adopted {
                // Qwen's gate-side operation preserves the existing fused
                // router/selector/gate/up launch when its static quartet is
                // admitted; the route operation then owns top-k production.
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::GateSide))?;
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Route))?;
                if owner_assign {
                    append_step(&mut steps, Step::MoeStage(op(state), MoeStage::OwnerAssign))?;
                }
            }
            if !params.skip_shared && !params.recipe.shared_after_combine() {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::SharedDown))?;
            }
        }
        MoeRecipe::SigmoidRoutedNoShared => {
            if !adopted {
                // Precomputed scores arrive activated: no router projection.
                if !params.router_scores_precomputed() {
                    append_step(
                        &mut steps,
                        Step::MoeStage(op(state), MoeStage::RouterProjection),
                    )?;
                }
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Route))?;
                if owner_assign {
                    append_step(&mut steps, Step::MoeStage(op(state), MoeStage::OwnerAssign))?;
                }
            }
            // Cohere's sigmoid route consumes the normalized natural input
            // first and transforms it only after top-eight selection.
            if selection.resolution.needs_x_rot_local {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::InputBasis))?;
            }
        }
    }
    if owner_assign && (adopted || !selection.resolution.use_gpu_topk) {
        return Err(DispatchError::Hip(
            "sealed moe: balanced EP ownership needs this program's GPU route stage to write the routed ids".into(),
        ));
    }
    if selection.resolution.use_gpu_topk {
        // A CPU-deferred call keeps the GPU router/rotation/shared stages, but
        // must not lower any routed GPU stage: the arch's CPU splice supplies
        // the routed contribution into the residual after this program, and the
        // expanded routed scratch is not refreshed here, so combining it would
        // fold stale rows. The mutation fence still marks the routed-stage
        // boundary the splice completes.
        if selection.cpu_split {
            // The sole input-readiness wait: after the router wrote the
            // original route and the rotated input exists, strictly before the
            // first routed GPU launch.
            append_step(
                &mut steps,
                Step::MoeStage(op(state), MoeStage::PrepareSplit),
            )?;
        }
        if !selection.cpu_deferred_experts {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::GateUp))?;
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Activation))?;
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Down))?;
        }
        append_step(
            &mut steps,
            Step::MoeStage(op(state), MoeStage::MutationFence),
        )?;
        if selection.cpu_split && !selection.cpwait {
            // The cold slot rows are queued behind the GPU down (same stream)
            // and before the canonical combine of the original route.
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::JoinSplit))?;
        }
        let down_self_combines = selection.down_last_combine || !params.down_writes_expanded();
        let combine_after_down = !selection.cpu_deferred_experts
            && params.ep_mode != crate::families::moe::MoeEpMode::RootRoutedPartial
            && !selection.ninepath_d4
            && (selection
                .route
                .is_some_and(MoeRouteCapability::is_sealed_grouped)
                || !down_self_combines)
            && !params.defer_routed_combine;
        if selection.cpu_split && !combine_after_down {
            return Err(split_error(
                "requires the canonical combine over expanded slot rows (no fused, ninepath or self-combining down)",
            ));
        }
        if params.ep_row_gather.is_some() {
            // The gather fills exactly the ten expanded rows the grouped
            // indexed down wrote; every other decode lowering either folds
            // inside its down kernel or follows a different EP contract.
            let grouped = selection
                .route
                .is_some_and(MoeRouteCapability::is_sealed_grouped);
            if !grouped
                || !combine_after_down
                || params.batch_size != 1
                || params.ep_mode != crate::families::moe::MoeEpMode::None
            {
                return Err(DispatchError::Hip(
                    "sealed moe: an EP row gather requires the one-row grouped QT44/QT53 decode with its canonical combine".into(),
                ));
            }
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::RowGather))?;
        }
        // A CPU-deferred call lowers no routed `Combine`: the splice takes its
        // place. A `shared_after_combine` recipe (Qwen4) folds the routed
        // contribution first and the shared expert second, so the splice runs
        // before `SharedDown`; the other recipes keep the splice last, exactly
        // where the arch ran it before the stage existed. The splice and
        // `SharedDown` accumulate into the same residual, so each arrangement
        // keeps its recipe's fp accumulation sequence.
        let splice_first = selection.cpu_deferred_experts && params.recipe.shared_after_combine();
        if splice_first {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::CpuSplice))?;
        }
        if combine_after_down {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Combine))?;
        }
        if params.recipe.shared_after_combine() && !params.skip_shared {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::SharedDown))?;
        }
        if selection.cpu_deferred_experts && !splice_first {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::CpuSplice))?;
        }
        if selection.cpu_split && selection.cpwait {
            // CP-wait order: the combine (and the shared down) are queued
            // behind the wait first; the CPU arithmetic runs last and releases
            // the wait. See `CpuSplitExecutor::cpwait_arm`.
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::JoinSplit))?;
        }
    } else {
        if params.ep_row_gather.is_some() {
            return Err(DispatchError::Hip(
                "sealed moe: an EP row gather requires GPU top-k routing".into(),
            ));
        }
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::HostExperts))?;
        if params.recipe.shared_after_combine() && !params.skip_shared {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::SharedDown))?;
        }
    }
    Ok(steps)
}

fn lower_prefill<'a>(
    state: &'a MoeStepState<'a>,
) -> Result<SmallVec<[Step<'a>; INLINE_STEP_CAPACITY]>, DispatchError> {
    let (params, selection) = state.prefill_parts()?;
    if selection.cpu_split {
        // Checked before any other lowering decision: the mode must never
        // fall into the generic or deferred arms by accident.
        if let Some(reason) = cpu_split_prefill_conflict(
            params,
            selection.route,
            selection.cpu_deferred_experts,
            // The executor-dependent shape check ran at seal and runs again in
            // `validate_split_prefill`; lowering needs no device object.
            None,
        ) {
            return Err(split_error(reason));
        }
    }
    let mut steps = SmallVec::new();
    let op = |state: &'a MoeStepState<'a>| SealedMoeOp::new(state);
    append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Normalize))?;
    if params.ep_row_gather.is_some()
        && (!selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
            || params.routed_out.is_some())
    {
        // The gather fills the expanded rows the grouped QT44/QT53 down wrote,
        // for the canonical combine into the residual; every other prefill
        // lowering has a different row layout or EP contract.
        return Err(DispatchError::Hip(
            "sealed moe: a prefill EP row gather requires the grouped QT44/QT53 route combining into the residual".into(),
        ));
    }
    if params.ep_tile_exchange.is_some()
        && (!selection
            .route
            .is_some_and(MoeRouteCapability::is_qt44_qt53_grouped)
            || !selection.uses_grouped_rows()
            || params.routed_out.is_some()
            || params.ep_row_gather.is_some())
    {
        // The exchange moves the BF16 grouped rows of the symmetric IU4 down or of the F16 WMMA
        // grouped down (the gfx1100 route) for the canonical combine into the residual; the
        // stages re-check the exact arm against the device (`qt44_qt53_prefill::tile_mask`).
        // Every other prefill lowering has a different row layout, a different down kernel or
        // EP contract.
        return Err(DispatchError::Hip(
            "sealed moe: a prefill EP tile exchange requires the grouped QT44/QT53 route (symmetric IU4 or F16 WMMA) combining into the residual".into(),
        ));
    }
    if selection
        .route
        .is_some_and(MoeRouteCapability::is_sealed_grouped)
    {
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::InputBasis))?;
        append_step(
            &mut steps,
            Step::MoeStage(op(state), MoeStage::RouterProjection),
        )?;
        append_step(
            &mut steps,
            Step::MoeStage(op(state), MoeStage::SharedGateUp),
        )?;
        append_step(
            &mut steps,
            Step::MoeStage(op(state), MoeStage::SharedActivation),
        )?;
        if !params.recipe.shared_after_combine() {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::SharedDown))?;
        }
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Route))?;
        // Balanced EP ownership: right after the route wrote the ids, before
        // the scatter and every routed-table reader.
        if params
            .ep_row_gather
            .is_some_and(|gather| gather.assigns_owners(params.ep_layer_idx))
        {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::OwnerAssign))?;
        }
        if selection.cpu_split {
            // Distinct eager CPU-cold/GPU-hot split. The original route feeds
            // the unchanged canonical combine and the shared expert; only the
            // routed GPU tail between `PrepareSplit` and `JoinSplit` reads the
            // executor's masked ids. `PrepareSplit` is the sole input-readiness
            // wait and precedes the first routed GPU launch; `JoinSplit` runs
            // the cold CPU arithmetic while that tail executes and queues the
            // cold F32 slot rows behind the GPU down, before the combine.
            append_step(
                &mut steps,
                Step::MoeStage(op(state), MoeStage::PrepareSplit),
            )?;
            if selection.uses_grouped_rows() {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Scatter))?;
            }
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::GateUp))?;
            if selection.uses_grouped_rows() {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Unscatter))?;
            }
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Activation))?;
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Down))?;
            append_step(
                &mut steps,
                Step::MoeStage(op(state), MoeStage::MutationFence),
            )?;
            if !selection.cpwait {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::JoinSplit))?;
            }
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Combine))?;
            if params.recipe.shared_after_combine() {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::SharedDown))?;
            }
            if selection.cpwait {
                // CP-wait order: the CPU arithmetic runs last and releases the
                // wait the combine is queued behind.
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::JoinSplit))?;
            }
            return Ok(steps);
        }
        if selection.cpu_deferred_experts {
            // The arch's CPU splice owns the routed FFN for this narrow window;
            // lowering the grouped routed tail would read/write scratch the
            // splice does not consume and fold stale expanded rows. The fence
            // still marks the routed-stage boundary the splice completes.
            append_step(
                &mut steps,
                Step::MoeStage(op(state), MoeStage::MutationFence),
            )?;
            // The splice stands where the routed `Combine` would be: before a
            // `shared_after_combine` recipe's `SharedDown` (routed first, shared
            // second, like the GPU tail), and last for the other recipes whose
            // shared expert already ran above. Both accumulate into one
            // residual, so this keeps each recipe's fp accumulation sequence.
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::CpuSplice))?;
            if params.recipe.shared_after_combine() {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::SharedDown))?;
            }
            return Ok(steps);
        }
        if selection.uses_grouped_rows() {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Scatter))?;
        }
        if params.ep_tile_exchange.is_some() {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::TileMask))?;
        }
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::GateUp))?;
        if selection.uses_grouped_rows() {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Unscatter))?;
        }
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Activation))?;
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Down))?;
        append_step(
            &mut steps,
            Step::MoeStage(op(state), MoeStage::MutationFence),
        )?;
        if params.ep_row_gather.is_some() {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::RowGather))?;
        }
        if params.ep_tile_exchange.is_some() {
            append_step(
                &mut steps,
                Step::MoeStage(op(state), MoeStage::TileExchange),
            )?;
        }
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Combine))?;
        if params.recipe.shared_after_combine() {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::SharedDown))?;
        }
        return Ok(steps);
    }
    let adopted = matches!(
        params.prelude.route,
        super::sealed_moe::PrefillRouteMode::AdoptRoot { .. }
    );
    match params.recipe {
        MoeRecipe::SoftmaxGatedShared { .. } => {
            // Paro: the routed basis is the routed experts' own Givens rotation
            // in `x_rot_batch`, which a Paro shared gate/up reuses as its
            // rotation scratch. Produce it after the shared stages, right
            // before the routed tail reads it.
            let paro = selection.resolution.paro_mode;
            if !paro {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::InputBasis))?;
            }
            if !adopted {
                append_step(
                    &mut steps,
                    Step::MoeStage(op(state), MoeStage::RouterProjection),
                )?;
            }
            append_step(
                &mut steps,
                Step::MoeStage(op(state), MoeStage::SharedGateUp),
            )?;
            append_step(
                &mut steps,
                Step::MoeStage(op(state), MoeStage::SharedActivation),
            )?;
            if !params.recipe.shared_after_combine() {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::SharedDown))?;
            }
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Route))?;
            if paro {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::InputBasis))?;
            }
        }
        MoeRecipe::SigmoidRoutedNoShared => {
            if !adopted {
                append_step(
                    &mut steps,
                    Step::MoeStage(op(state), MoeStage::RouterProjection),
                )?;
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Route))?;
            }
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::InputBasis))?;
        }
    }
    if !selection.cpu_deferred_experts {
        if selection.uses_grouped_rows() {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Scatter))?;
        }
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::GateUp))?;
        if selection.uses_grouped_rows() {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Unscatter))?;
        }
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Activation))?;
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Down))?;
    }
    append_step(
        &mut steps,
        Step::MoeStage(op(state), MoeStage::MutationFence),
    )?;
    let compact_ep = matches!(
        params.prelude.route,
        super::sealed_moe::PrefillRouteMode::ProduceRoot { .. }
            | super::sealed_moe::PrefillRouteMode::AdoptRoot { .. }
    );
    // A CPU-deferred window must not combine: the expanded routed scratch was
    // not refreshed by this program, so folding it would publish stale rows.
    // The arch's CPU splice accumulates the routed contribution into `x_batch`.
    if !selection.cpu_deferred_experts
        && !compact_ep
        && (selection.uses_grouped_rows() || !selection.resolution.down_path0)
    {
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Combine))?;
    }
    let splice_first = selection.cpu_deferred_experts && params.recipe.shared_after_combine();
    if splice_first {
        // Where the routed `Combine` would be: a `shared_after_combine` recipe
        // folds the routed contribution before the shared expert.
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::CpuSplice))?;
    }
    if params.recipe.shared_after_combine() {
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::SharedDown))?;
    }
    if selection.cpu_deferred_experts && !splice_first {
        // The shared expert ran before the routed tail for this recipe, so the
        // splice is the last stage, as the arch's own splice was.
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::CpuSplice))?;
    }
    Ok(steps)
}

/// Lower and execute a sealed call using the shared Step interpreter.
pub(super) fn execute(gpu: &mut Gpu, call: &SealedMoeCall<'_>) -> Result<(), DispatchError> {
    // Reborrow the call at the execution lifetime. The state and every Step
    // borrow only this invocation and are dropped before the call returns.
    let state = MoeStepState::new(call);
    state.front_allowed.set(true);
    let steps = match call.protocol() {
        MoeProtocol::IndexedDecode => lower_decode(&state)?,
        MoeProtocol::GroupedPrefill => lower_prefill(&state)?,
    };
    super::steps::execute_validated_steps(gpu, call.dispatch_ctx(), &steps)
}

/// The stages of the S1 layer-major PREPARE half of a sealed grouped prefill, in
/// lowering order: everything [`lower_prefill`] runs before its routed tail
/// (`Scatter` .. `Combine`) for a `shared_after_combine` recipe.  The routed tail
/// runs later, window-wide, from `slowlink_prefill`; the shared down runs after
/// it from `slowlink_prefill::finish_shared`.
pub(super) const PREPARE_STAGES: [MoeStage; 6] = [
    MoeStage::Normalize,
    MoeStage::InputBasis,
    MoeStage::RouterProjection,
    MoeStage::SharedGateUp,
    MoeStage::SharedActivation,
    MoeStage::Route,
];

/// Run [`PREPARE_STAGES`] of a sealed grouped prefill call (one tile of the S1
/// window) through the shared Step interpreter.  Refuses, before any launch,
/// every call whose lowering is not exactly the plain grouped tail followed by a
/// shared down after the combine.
pub(super) fn execute_prepare(
    gpu: &mut Gpu,
    call: &SealedMoeCall<'_>,
) -> Result<(), DispatchError> {
    let state = MoeStepState::new(call);
    if call.protocol() == MoeProtocol::IndexedDecode {
        return execute_prepare_decode(gpu, call, &state);
    }
    let (params, selection) = state.prefill_parts()?;
    if !selection
        .route
        .is_some_and(MoeRouteCapability::is_sealed_grouped)
        || selection.cpu_split
        || selection.cpu_deferred_experts
        || !params.recipe.shared_after_combine()
        || params.ep_row_gather.is_some()
        || params.ep_tile_exchange.is_some()
        || params.routed_out.is_some()
    {
        return Err(DispatchError::Hip(
            "slowlink prepare: the call is not a plain sealed grouped prefill with the shared down after the combine".into(),
        ));
    }
    let mut steps: SmallVec<[Step<'_>; INLINE_STEP_CAPACITY]> = SmallVec::new();
    for stage in PREPARE_STAGES {
        append_step(&mut steps, Step::MoeStage(SealedMoeOp::new(&state), stage))?;
    }
    super::steps::execute_validated_steps(gpu, call.dispatch_ctx(), &steps)
}

/// The decode (one-row, `route_rows == 1`) twin of [`execute_prepare`]: the incumbent's
/// own decode prelude up to and including the route (normalize, the optional input basis,
/// the fused router/shared gate-side stage, the GPU top-k), and nothing of its routed tail
/// or shared down.  The routed tail of a one-row tile is the K1 cohort of `slowlink_prefill`.
fn execute_prepare_decode<'a>(
    gpu: &mut Gpu,
    call: &'a SealedMoeCall<'a>,
    state: &'a MoeStepState<'a>,
) -> Result<(), DispatchError> {
    let (params, selection) = state.decode_parts()?;
    let adopted = matches!(
        call.router_input(),
        MoeRouterInput::PrecomputedSoftmaxTopK | MoeRouterInput::PrecomputedSigmoidTopK
    );
    if !selection
        .route
        .is_some_and(MoeRouteCapability::is_sealed_grouped)
        || !matches!(params.recipe, MoeRecipe::SoftmaxGatedShared { .. })
        || !params.recipe.shared_after_combine()
        || !selection.resolution.use_gpu_topk
        || adopted
        || selection.cpu_split
        || selection.cpu_deferred_experts
        || selection.down_last_combine
        || selection.ninepath_d3
        || selection.ninepath_d4
        || params.ep_row_gather.is_some()
        || params.ep_mode != crate::families::moe::MoeEpMode::None
        || params.routed_out.is_some()
        || params.defer_routed_combine
        || params.skip_shared
        || params.shared.is_none()
        || params.batch_size != 1
        || params.k != 10
    {
        return Err(DispatchError::Hip(
            "slowlink prepare: the one-row call is not a plain sealed grouped decode with the shared down after the combine".into(),
        ));
    }
    let op = |state: &'a MoeStepState<'a>| SealedMoeOp::new(state);
    let mut steps: SmallVec<[Step<'a>; INLINE_STEP_CAPACITY]> = SmallVec::new();
    append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Normalize))?;
    if selection.resolution.needs_x_rot_local {
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::InputBasis))?;
    }
    append_step(&mut steps, Step::MoeStage(op(state), MoeStage::GateSide))?;
    append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Route))?;
    super::steps::execute_validated_steps(gpu, call.dispatch_ctx(), &steps)
}

/// The MoE stages a call lowers to, in order (test probe: nothing launches).
#[cfg(test)]
pub(super) fn lowered_stages(call: &SealedMoeCall<'_>) -> Result<Vec<MoeStage>, DispatchError> {
    let state = MoeStepState::new(call);
    let steps = match call.protocol() {
        MoeProtocol::IndexedDecode => lower_decode(&state)?,
        MoeProtocol::GroupedPrefill => lower_prefill(&state)?,
    };
    Ok(steps
        .iter()
        .filter_map(|step| match step {
            Step::MoeStage(_, stage) => Some(*stage),
            _ => None,
        })
        .collect())
}

/// The split transaction's phase discipline for a call sealed in the split
/// mode (test probe: nothing launches and no executor is needed). The routed
/// GPU tail may not take the masked view before `PrepareSplit`, `PrepareSplit`
/// and `JoinSplit` each run exactly once and in order, the combine needs the
/// joined phase, and the post-combine reference keeps the original routed
/// operands with every shared-expert side effect disabled.
#[cfg(test)]
pub(super) fn split_phase_probe(call: &SealedMoeCall<'_>) -> Result<(), String> {
    let state = MoeStepState::new(call);
    if let Ok((params, selection)) = state.decode_parts() {
        if state.masked_decode(params, selection).is_ok() {
            return Err("decode tail took the masked view before PrepareSplit".into());
        }
    }
    if let Ok((params, selection)) = state.prefill_parts() {
        if state.masked_prefill(params, selection).is_ok() {
            return Err("prefill tail took the masked view before PrepareSplit".into());
        }
    }
    for want in [SplitPhase::Prepared, SplitPhase::Joined] {
        if state.require_split_phase(want, "probe").is_ok() {
            return Err(format!("{want:?} accepted from Idle"));
        }
    }
    state
        .require_split_phase(SplitPhase::Idle, "PrepareSplit")
        .map_err(|e| format!("{e:?}"))?;
    state.split_phase.set(SplitPhase::Prepared);
    if state
        .require_split_phase(SplitPhase::Idle, "PrepareSplit")
        .is_ok()
    {
        return Err("PrepareSplit ran twice".into());
    }
    if state
        .require_split_phase(SplitPhase::Joined, "Combine")
        .is_ok()
    {
        return Err("the canonical combine ran before JoinSplit".into());
    }
    state
        .require_split_phase(SplitPhase::Prepared, "JoinSplit")
        .map_err(|e| format!("{e:?}"))?;
    state.split_phase.set(SplitPhase::Joined);
    if state
        .require_split_phase(SplitPhase::Prepared, "JoinSplit")
        .is_ok()
    {
        return Err("JoinSplit ran twice".into());
    }
    state
        .require_split_phase(SplitPhase::Joined, "Combine")
        .map_err(|e| format!("{e:?}"))?;
    state.split_reference.set(true);
    // The reference rerun replays gate/up/unscatter/activation/down with the
    // ORIGINAL route, so its view must (a) exist, (b) keep every routed operand
    // of the original call, and (c) have the shared-expert side effects of
    // those stages disabled: the fused activation / unscatter launches
    // sigmoid the shared selector scalar IN PLACE, and the original call's
    // shared stage has already consumed (and will not recompute) it.
    if let Ok((params, selection)) = state.decode_parts() {
        if !super::qt44_shared_activation_in_routed(params, selection.route) {
            return Err("decode fixture does not exercise the fused shared activation".into());
        }
        let Ok(Some(view)) = state.masked_decode(params, selection) else {
            return Err("decode reference rerun has no shared-disabled view".into());
        };
        let same = [
            (view.topk_indices, params.topk_indices),
            (view.topk_weights, params.topk_weights),
            (view.expert_gate_up_ptrs, params.expert_gate_up_ptrs),
            (view.expert_down_ptrs, params.expert_down_ptrs),
            (view.gate_batch, params.gate_batch),
            (view.up_batch, params.up_batch),
            (view.rot_batch, params.rot_batch),
            (view.down_expanded, params.down_expanded),
        ];
        if !same.iter().all(|(a, b)| std::ptr::eq(*a, *b)) {
            return Err("decode reference view does not keep the original routed operands".into());
        }
        if !view.skip_shared {
            return Err("decode reference view still runs the shared expert side effects".into());
        }
        if super::qt44_shared_activation_in_routed(&view, selection.route) {
            return Err("decode reference view still fuses the shared activation".into());
        }
        if params.skip_shared {
            return Err("decode reference view mutated the original call".into());
        }
    }
    if let Ok((params, selection)) = state.prefill_parts() {
        if params.prelude.shared.is_none() {
            return Err("prefill fixture does not carry a shared expert".into());
        }
        let Ok(Some(view)) = state.masked_prefill(params, selection) else {
            return Err("prefill reference rerun has no shared-disabled view".into());
        };
        let same = [
            (view.topk_indices, params.topk_indices),
            (view.topk_weights, params.topk_weights),
            (view.expert_gate_up_ptrs, params.expert_gate_up_ptrs),
            (view.expert_down_ptrs, params.expert_down_ptrs),
            (view.gate_batch, params.gate_batch),
            (view.up_batch, params.up_batch),
            (view.rot_batch, params.rot_batch),
            (view.down_expanded, params.down_expanded),
        ];
        if !same.iter().all(|(a, b)| std::ptr::eq(*a, *b)) {
            return Err("prefill reference view does not keep the original routed operands".into());
        }
        if view.prelude.shared.is_some() {
            return Err("prefill reference view still runs the shared expert side effects".into());
        }
        if params.prelude.shared.is_none() {
            return Err("prefill reference view mutated the original call".into());
        }
    }
    Ok(())
}

/// [`execute`] for a call preceded by `Step::Clear(clear)` and, optionally,
/// followed by the HC write `hc` whose gates are ready.
///
/// H8a (`HIPFIRE_QWEN4_MOE_COMBINE_ZINIT`): when the qt44 combine can start
/// from +0.0 into exactly the cleared target, the zero fill is not launched;
/// otherwise it runs here, first, exactly as the separate step would have.
/// The decision is made once from the same predicate the combine stage
/// re-checks, and the call fails loudly if an absorbed fill was not consumed
/// by the combine (the target would otherwise be left uncleared).
///
/// H4 level 3 (`HIPFIRE_QWEN4_HC_FUSE` >= 3): `hc` is offered to the shared
/// down stage, which folds it into its GEMM epilogue when it fits.  Returns
/// whether it did; the caller then skips the write.  When the stage declines
/// (or never runs), nothing was launched for it and the write runs unfused.
///
/// `HIPFIRE_QWEN4_HC_ROW_FOLD`: with `next` (the HC read right after `hc` and
/// that read's paired write, both already admitted by the step interpreter),
/// the shared down runs the whole tail as one row kernel when
/// [`super::qt44_qt53_prefill::row_fold_applies`]; the result then reports
/// `row_folded` and the read's norm + its write's gates are already done.
pub(super) fn execute_after_clear(
    gpu: &mut Gpu,
    call: &SealedMoeCall<'_>,
    clear: &super::layer_ops::ClearOp<'_>,
    hc: Option<&super::layer_ops::HyperWriteOp<'_>>,
    next: Option<(
        &super::layer_ops::HyperReadOp<'_>,
        &super::layer_ops::HyperWriteOp<'_>,
    )>,
) -> Result<AfterClear, DispatchError> {
    let state = MoeStepState::new(call);
    state.front_allowed.set(true);
    let absorbed = state.absorbs_clear(gpu, clear);
    if !absorbed {
        super::layer_ops::execute_clear(gpu, clear)?;
    }
    state.initial_zero.set(absorbed);
    if let Some(write) = hc {
        let mut offered = super::qt44_qt53_prefill::HcSharedDown::offer(write);
        if let (Some(offered), Some((read, next_write))) = (offered.as_mut(), next) {
            if let Ok((params, selection)) = state.prefill_parts() {
                if selection
                    .route
                    .is_some_and(MoeRouteCapability::is_sealed_grouped)
                    && !selection.cpu_split
                    && super::qt44_qt53_prefill::row_fold_applies(
                        gpu,
                        params,
                        selection.uses_grouped_rows(),
                        offered,
                    )
                {
                    offered.row_fold = Some(super::qt44_qt53_prefill::HcRowFoldNext::offer(
                        read, next_write,
                    ));
                }
            }
        }
        *state.hc_write.borrow_mut() = offered;
    }
    let steps = match call.protocol() {
        MoeProtocol::IndexedDecode => lower_decode(&state)?,
        MoeProtocol::GroupedPrefill => lower_prefill(&state)?,
    };
    super::steps::execute_validated_steps(gpu, call.dispatch_ctx(), &steps)?;
    if state.initial_zero.get() {
        return Err(DispatchError::Hip(
            "sealed moe: the absorbed target clear was not consumed by the combine".into(),
        ));
    }
    Ok(AfterClear {
        folded: state.hc_folded.get(),
        row_folded: state.hc_row_folded.get(),
    })
}

/// What [`execute_after_clear`] did beyond the call itself.
pub(super) struct AfterClear {
    /// The shared down carried the call's HC write.
    pub folded: bool,
    /// ... and the next HC read's norm + its write's gates (the HC row fold).
    pub row_folded: bool,
}

/// Execute only the canonical slot-order combine for a validated EP call.
pub(super) fn execute_ep_slot_combine(
    gpu: &mut Gpu,
    call: &SealedMoeCall<'_>,
) -> Result<(), DispatchError> {
    let state = MoeStepState::new(call);
    SealedMoeOp::new(&state).combine(gpu)
}

/// The specialized wire path does not apply per-projection transformations.
fn projection_has_sidecars(
    router: &crate::families::gemv::WeightRef<'_>,
    shared: &crate::families::moe::MoeSharedWeights<'_>,
) -> bool {
    [
        router,
        &shared.selector,
        &shared.gate,
        &shared.up,
        &shared.down,
    ]
    .into_iter()
    .any(|weight| weight.rotation.is_some() || weight.awq_scale.is_some())
}

/// Resolve all decode choices without a GPU. Dynamic GPU capture state remains
/// in `SealedMoeCall::validate_for_gpu` so capture refusal happens before the
/// first launch rather than during sealing.
pub(super) fn select_decode(
    ctx: &DispatchCtx,
    params: &MoeParams<'_>,
    router: MoeRouterInput,
) -> Result<MoeDecodeSelection, DispatchError> {
    super::check_moe_decode_batch_size(params.batch_size)?;
    // Qt55 decode uses the grouped protocol even for one row. An indexed
    // descriptor cannot supply its stable permutation and must fail closed.
    super::reject_fp8_moe(&params.dtypes)?;
    if params.recipe == MoeRecipe::SigmoidRoutedNoShared
        && (!super::sealed_moe::sigmoid_decode_width_ok(params)
            || params.ep_mode != crate::families::moe::MoeEpMode::None
            || params.defer_routed_combine)
    {
        return Err(DispatchError::Hip(
            "sigmoid MoE decode requires k=8 (k<=8 with a selection bias), no EP, and no deferred combine"
                .into(),
        ));
    }
    let mut resolution = MoeResolution::resolve_arch(&params.dtypes, params.k, ctx.arch.has_wmma());
    let route = (matches!(params.recipe, MoeRecipe::SoftmaxGatedShared { .. })
        && params.batch_size == 1
        && params.ep_mode == crate::families::moe::MoeEpMode::None
        && params.routed_gate_up_paro.is_none()
        && params.routed_down_paro.is_none()
        && !params.defer_routed_combine)
        .then(|| {
            super::select_grouped_route(
                &params.dtypes,
                params.n_exp,
                params.k,
                params.hidden,
                params.mi,
                params
                    .shared
                    .as_ref()
                    .map_or(0, |shared| shared.intermediate),
                params.routed_gate_up_k,
                params.routed_down_m,
                params.routed_down_k,
                params.expert_dtype_tags,
                params.expert_down_awq_ptrs.is_some()
                    || params.shared.as_ref().is_some_and(|shared| {
                        projection_has_sidecars(&params.router, &shared.weights)
                    }),
                params.route_policy,
                ctx.arch.arch(),
            )
        })
        .flatten();
    if params.route_policy.is_some() && route.is_none() {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "specialized-decode-route-unsupported",
            arch: "",
            quant: "QT44/QT53",
        });
    }
    if route.is_none() {
        super::reject_mq4g128v2_moe(&params.dtypes)?;
        // qwen35 qt53 routed-down typed fallback (no route policy only): refuse
        // tags / AWQ / Paro / CPU-deferred operands before anything is lowered.
        super::check_qt53_decode_fallback(params, route)?;
    } else {
        resolution.use_gpu_topk = true;
        resolution.needs_x_rot_local = true;
    }
    if params.mq2_lloyd.is_some() {
        // A declared MQ2-Lloyd route runs only the uniform MQ2G256Lloyd kernels.
        use rdna_compute::DType::MQ2G256Lloyd;
        let uniform = matches!(
            (params.dtypes.routed_gate_up, params.dtypes.routed_down),
            (MQ2G256Lloyd, MQ2G256Lloyd)
        ) && !params.dtypes.routed_has_mixed_experts
            && params.expert_dtype_tags.is_none()
            && params.expert_down_awq_ptrs.is_none()
            && params.routed_gate_up_paro.is_none()
            && params.routed_down_paro.is_none();
        if !uniform {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "mq2-lloyd-declaration-needs-uniform-mq2g256-lloyd-experts",
                arch: "",
                quant: "",
            });
        }
    }
    if params.selection_bias.is_some() {
        // The bias-aware selector writes any width up to eight. At top-8 every
        // generic indexed route applies; below it, only the declared MQ2-Lloyd
        // route (uniform, checked above) and the uniform FWHT-basis MQ4G256 and
        // MQ6G256 gate/up, activation, down and combine read that width at run time.
        use rdna_compute::DType::{MQ4G256, MQ6G256};
        let runtime_width = params.mq2_lloyd.is_some()
            || matches!(
                (params.dtypes.routed_gate_up, params.dtypes.routed_down),
                (MQ4G256, MQ4G256) | (MQ6G256, MQ6G256)
            ) && !params.dtypes.routed_has_mixed_experts
                && params.expert_dtype_tags.is_none()
                && params.expert_down_awq_ptrs.is_none()
                && params.routed_gate_up_paro.is_none();
        if params.k < MOE_BIASED_SIGMOID_MAX_K && !runtime_width {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "biased-sigmoid-router-needs-runtime-width-experts",
                arch: "",
                quant: "",
            });
        }
        resolution.use_gpu_topk = true;
    }
    super::check_moe_decode_supported(
        resolution.use_gpu_topk,
        params.k,
        params.n_exp,
        !params.routed_experts.is_empty(),
    )?;
    if params.cpu_deferred_experts {
        // Width gate (D1, whole-window, no chunking): the legacy `<= 4`-row
        // windows always qualify; a wider window qualifies only inside an
        // explicit lab-wide scope with actual prewarmed host scratch.
        // Unqualified wide windows refuse here — never a silent GPU fallback.
        {
            // Decode protocol is one row: the ctx flag is the eager-verify
            // scope carrier (a plain decode keeps it false and the legacy gate).
            let scope = CpuWideVerifyScope::from_lab_presence(
                ctx.cpu_wide_verify() && cpu_lab_override_present_for_arch(ctx.arch.arch()),
                rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows(ctx.arch.arch()),
            );
            // Decode protocol is one row; route rows are trivially whole.
            if !crate::families::moe::cpu_d1_wide_window_qualified(
                true,
                params.batch_size,
                params.batch_size,
                scope,
                crate::families::moe::cpu_d1_prewarmed_scratch_rows(
                    params.hidden,
                    params.mi,
                    params.k,
                    ctx.device_id(),
                ),
            ) {
                return Err(DispatchError::Hip(
                    "sealed moe: CPU-deferred routed experts: the window exceeds the qualified CPU window for this scope (wider windows keep the GPU route)".into(),
                ));
            }
        }
        // The arch's CPU splice consumes the GPU-produced route and the rotated
        // input, then accumulates the routed FFN into `x_residual` after this
        // program. It therefore needs the GPU top-K path (never the per-expert
        // host GEMV fallback), a rotated activation, a single-device non-EP
        // call, and a recipe whose routed combine owes no bf16 round trip.
        if !resolution.use_gpu_topk || !resolution.needs_x_rot_local {
            return Err(DispatchError::Hip(
                "CPU-deferred routed experts require the GPU top-K path with a rotated local activation"
                    .into(),
            ));
        }
        if params.ep_mode != crate::families::moe::MoeEpMode::None
            || params.routed_out.is_some()
            || params.defer_routed_combine
        {
            return Err(DispatchError::Hip(
                "CPU-deferred routed experts require a single-device call with no EP partial and no deferred routed combine"
                    .into(),
            ));
        }
        // A BF16 recipe is admitted only on the grouped QT44/QT53 route: the
        // router and shared stages this program still lowers carry its BF16
        // round trips themselves, and the omitted routed stages' round trips
        // (gate/up, activation, weighted combine) are the CPU splice's, which
        // receives the recipe flag. The generic route also rounds the rotated
        // input, which the CPU does not emulate.
        if params.recipe.bf16_round_trip()
            && !route.is_some_and(MoeRouteCapability::is_qt44_qt53_grouped)
        {
            return Err(DispatchError::Hip(
                "CPU-deferred routed experts emulate a recipe's bf16 round trips only on the grouped QT44/QT53 route"
                    .into(),
            ));
        }
    }
    // The distinct CPU-cold/GPU-hot split of the one-row decode. Every
    // admission refusal is raised here, before the call is published.
    let cpu_split =
        match crate::families::moe::cpu_split_executor(params.routed_experts, params.batch_size) {
            Some(executor) => {
                if let Some(reason) = cpu_split_decode_conflict(
                    params,
                    route,
                    params.cpu_deferred_experts,
                    Some(executor),
                ) {
                    return Err(DispatchError::Hip(format!(
                        "sealed moe: CPU tier split: {reason}"
                    )));
                }
                if !resolution.use_gpu_topk || !resolution.needs_x_rot_local {
                    return Err(split_error(
                        "requires the GPU top-K path with a rotated local activation",
                    ));
                }
                true
            }
            None => false,
        };
    if params.ep_mode == crate::families::moe::MoeEpMode::RootRoutedPartial {
        if !resolution.use_gpu_topk {
            return Err(DispatchError::Hip(
                "root-routed EP decode requires the GPU top-K path; the CPU-top-K fallback is rejected"
                    .into(),
            ));
        }
        if params.defer_routed_combine {
            return Err(DispatchError::Hip(
                "root-routed EP decode requires defer_routed_combine=false (weighted combine into the partial)"
                    .into(),
            ));
        }
        if params.routed_out.is_none() {
            return Err(DispatchError::Hip(
                "root-routed EP decode requires routed_out=Some (the zeroed partial)".into(),
            ));
        }
        if !params.down_writes_expanded() {
            return Err(DispatchError::Hip(
                "root-routed EP decode requires an expanded-output down projection".into(),
            ));
        }
    }
    let skip_routing = matches!(
        router,
        MoeRouterInput::PrecomputedSoftmaxTopK | MoeRouterInput::PrecomputedSigmoidTopK
    );
    if !resolution.use_gpu_topk {
        if skip_routing {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "cpu-topk-fallback-needs-gpu-route-for-precomputed-input",
                arch: "",
                quant: "",
            });
        }
        if params.routed_out.is_some() {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "ep-routed-out-unsupported-in-cpu-topk-fallback",
                arch: "",
                quant: "",
            });
        }
        if params.routed_experts.is_empty() {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "cpu-topk-fallback-needs-resident-experts",
                arch: "",
                quant: "",
            });
        }
    }
    if resolution.needs_x_rot_local
        && !params.x_rot_prerotated
        && resolution.routed_indexable_paro
        && params.routed_gate_up_paro.is_none()
    {
        return Err(DispatchError::Hip(
            "sealed moe: Paro decode program has no gate-up sidecar".into(),
        ));
    }

    let gfx1100_router_mode = hipfire_config::developer_var("HIPFIRE_GFX1100_ROUTER_W64").ok();
    let exact_wave64_router = params.n_exp == 256
        && ((ctx.arch.is_gfx1100()
            && !matches!(gfx1100_router_mode.as_deref(), Some("0" | "approx")))
            || ctx.arch.is_gfx1151());
    let wave64_router = (ctx.arch.is_gfx1201()
        && hipfire_config::developer_var("HIPFIRE_GFX1201_ROUTER_W64").as_deref() != Ok("0"))
        || (ctx.arch.is_gfx1100()
            && params.n_exp == 256
            && gfx1100_router_mode.as_deref() == Some("approx"));
    let (shared_smi, shared_down_dtype, shared_down_has_awq) = params
        .shared
        .as_ref()
        .map(|shared| {
            (
                shared.intermediate,
                shared.weights.down.dtype,
                shared.weights.down.awq_scale.is_some(),
            )
        })
        .unwrap_or((0, rdna_compute::DType::F32, false));
    static ROUTER_SHARED_FUSE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_MOE_ROUTER_SHARED_FUSE").as_deref() == Ok("1")
    });
    let router_shared_fuse = router_shared_fuse_allowed(
        skip_routing,
        resolution.use_gpu_topk,
        exact_wave64_router,
        params.batch_size,
        params.skip_shared,
        shared_smi,
        shared_down_dtype,
        shared_down_has_awq,
        *ROUTER_SHARED_FUSE,
    );

    static DOWN_LAST_COMBINE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let down_last_combine = down_last_combine_allowed(
        params.ep_mode == crate::families::moe::MoeEpMode::None,
        ctx.arch.is_gfx1100(),
        params.batch_size,
        params.k,
        params.expert_dtype_tags.is_some(),
        params.dtypes.routed_down,
        params.mi,
        || {
            *DOWN_LAST_COMBINE.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_MOE_DOWN_LAST_COMBINE").as_deref() == Ok("1")
            })
        },
    );

    static MOE_NINEPATH: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_MOE_NINEPATH").unwrap_or_default()
    });
    let ninepath_mode = MOE_NINEPATH.as_str();
    let ninepath_shape_ok = params.ep_mode == crate::families::moe::MoeEpMode::None
        && params.k == 8
        && params.batch_size == 1
        && params.hidden <= 2048
        && params.mi == 512
        && params.expert_dtype_tags.is_none()
        && params.expert_down_awq_ptrs.is_none()
        && !params.defer_routed_combine;
    let ninepath_hfq4 = ninepath_shape_ok
        && params.dtypes.routed_gate_up == rdna_compute::DType::MQ4G256
        && params.dtypes.routed_down == rdna_compute::DType::MQ4G256;
    let ninepath_mq3l = ninepath_shape_ok
        && params.dtypes.routed_gate_up == rdna_compute::DType::MQ2G256Lloyd
        && params.dtypes.routed_down == rdna_compute::DType::MQ3G256Lloyd;
    let ninepath_mq4v2 = ninepath_shape_ok
        && params.dtypes.routed_gate_up == rdna_compute::DType::MQ4G256V2
        && params.dtypes.routed_down == rdna_compute::DType::MQ4G256V2;
    let ninepath_mq6v2 = ninepath_shape_ok
        && params.dtypes.routed_gate_up == rdna_compute::DType::MQ6G256V2
        && params.dtypes.routed_down == rdna_compute::DType::MQ6G256V2;
    // qwen35 qt53 routed-down typed fallback (A3B mq4rs): MQ4G256V2 gate/up and
    // MQ4G128V2 down, F32 recipe, no specialized route. wave32 gfx1201 only
    // (the kernel is exact-shaped: down_k 512, down_m % 16 == 0).
    let ninepath_mq4g128v2 = ninepath_shape_ok
        && route.is_none()
        && !params.recipe.bf16_round_trip()
        && ctx.arch.is_gfx1201()
        && params.routed_down_k == 512
        && params.routed_down_m % 16 == 0
        && params.routed_gate_up_paro.is_none()
        && params.routed_down_paro.is_none()
        && params.dtypes.routed_gate_up == rdna_compute::DType::MQ4G256V2
        && params.dtypes.routed_down == rdna_compute::DType::MQ4G128V2
        && params.dtypes.qt53_routed_down_fallback();
    let ninepath_eligible =
        ninepath_hfq4 || ninepath_mq3l || ninepath_mq4v2 || ninepath_mq4g128v2 || ninepath_mq6v2;
    resolution.check_generic_route(route)?;
    Ok(MoeDecodeSelection {
        resolution,
        router_shared_fuse: router_shared_fuse && !cpu_split,
        exact_wave64_router,
        wave64_router,
        down_last_combine: down_last_combine && !cpu_split,
        route,
        // The split's routed down always writes the expanded F32 slot rows the
        // canonical combine folds: no fused or self-combining down arm exists.
        ninepath_d3: !cpu_split && ninepath_hfq4 && matches!(ninepath_mode, "1" | "d3" | "on"),
        ninepath_d4: !cpu_split
            && ninepath_eligible
            && !matches!(ninepath_mode, "0" | "off" | "d3"),
        ninepath_mq3l: ninepath_mq3l && !cpu_split,
        ninepath_mq4v2: ninepath_mq4v2 && !cpu_split,
        ninepath_mq4g128v2: ninepath_mq4g128v2
            && !cpu_split
            && !matches!(ninepath_mode, "0" | "off" | "d3"),
        ninepath_mq6v2: ninepath_mq6v2 && !cpu_split,
        cpu_deferred_experts: params.cpu_deferred_experts,
        cpu_split,
        cpwait: cpu_split
            && crate::families::moe::cpu_split_executor(params.routed_experts, params.batch_size)
                .is_some_and(CpuSplitExecutor::cpwait_active),
        norm_fold: a3b_norm_fold_admitted(
            ctx.arch.is_gfx1201(),
            params,
            &resolution,
            route,
            router_shared_fuse && !cpu_split,
            skip_routing,
            cpu_split,
        ),
    })
}

/// Snapshot of device states that make a host-synchronous CPU split unsound.
/// Its host work cannot be captured, recorded, replayed or shadow-validated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct CpuSplitRunGate {
    pub(super) graph_capture: bool,
    pub(super) verify_capture: bool,
    pub(super) replay_capture: bool,
    pub(super) stream_capture: bool,
    pub(super) recording: bool,
    pub(super) aql: bool,
    pub(super) pm4: bool,
    pub(super) shadow: bool,
}

impl CpuSplitRunGate {
    pub(super) fn observe(gpu: &Gpu) -> Result<Self, DispatchError> {
        let stream_capture = match gpu.active_stream.as_ref() {
            Some(stream) => gpu
                .hip
                .stream_is_capturing(stream)
                .map_err(|e| split_error(format!("stream capture query: {e}")))?,
            None => false,
        };
        Ok(Self {
            graph_capture: gpu.graphs.capture_mode,
            verify_capture: gpu.graphs.verify.capturing.is_some(),
            replay_capture: gpu.graphs.replay.capturing.is_some(),
            stream_capture,
            recording: gpu.replay.is_recording(),
            aql: gpu.replay.should_route_aql(),
            pm4: gpu.replay.should_route_pm4(),
            shadow: gpu.replay.shadow_body_route().is_some()
                || gpu.replay.state() == rdna_compute::replay::ReplayState::ShadowValidated,
        })
    }

    /// The first refused state, named for the operator.
    pub(super) fn refusal(self) -> Option<&'static str> {
        if self.graph_capture || self.verify_capture || self.replay_capture || self.stream_capture {
            Some("graph capture")
        } else if self.recording {
            Some("replay recording")
        } else if self.pm4 {
            Some("a retained PM4 body")
        } else if self.aql {
            Some("a retained AQL body")
        } else if self.shadow {
            Some("a shadow-validated retained body")
        } else {
            None
        }
    }

    pub(super) fn require_eager(self) -> Result<(), DispatchError> {
        match self.refusal() {
            None => Ok(()),
            Some(state) => Err(split_error(format!(
                "is a host-synchronous step and is refused under {state}; run eager (replay.backend=hip, memory.offload_exec=pcie)"
            ))),
        }
    }
}

/// Where one invocation is in the eager CPU split transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SplitPhase {
    /// Nothing has run; the routed GPU tail must not.
    Idle,
    /// `PrepareSplit` completed: the masked ids are queued on the stream.
    Prepared,
    /// CP-wait join only: `Combine` queued the wait and then the canonical
    /// combine (reading the mapped view) behind it. `JoinSplit` must still run
    /// and release the wait; an owner that abandons the call in this phase
    /// aborts the wait (see [`MoeStepState`]'s `Drop`).
    Armed,
    /// `JoinSplit` completed: the cold slot rows are queued behind the GPU
    /// down, so the canonical combine may run.
    Joined,
}

/// The executor constructor's fixed route width (it refuses any other).
const SPLIT_TOP_K: usize = 10;
const _: () = assert!(crate::cpu_split::CPU_SPLIT_MAX_ROWS == NARROW_VERIFY_MAX_ROWS);

fn split_error(message: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(format!("sealed moe: CPU tier split: {message}"))
}

/// The executor sealed into a decode call (the same predicate that sealed it).
fn decode_split_executor<'p>(
    params: &MoeParams<'p>,
) -> Result<&'p CpuSplitExecutor, DispatchError> {
    let experts: &'p dyn crate::families::moe::RoutedExpertWeights = params.routed_experts;
    crate::families::moe::cpu_split_executor(experts, params.batch_size)
        .ok_or_else(|| split_error("the executor sealed into this call is no longer attached"))
}

/// The executor sealed into a prefill call (the same predicate that sealed it).
/// Wide-qualified calls (sealed with an explicit lab-wide scope) resolve
/// through the storage gate; default-route calls keep the 4-row gate. Both
/// read the executor's allocated capacity, never a constant.
fn prefill_split_executor<'p>(
    params: &MoePrefillParams<'p>,
) -> Result<&'p CpuSplitExecutor, DispatchError> {
    let experts: &'p dyn crate::families::moe::RoutedExpertWeights = params.routed_experts;
    // Default-4 gate first (back-compatible with executors that predate the
    // storage widening): `cpu_split_executor` already checks allocated
    // capacity. The wide storage gate below admits qualified 5..64-row windows.
    if let Some(executor) = crate::families::moe::cpu_split_executor(experts, params.batch_size) {
        return Ok(executor);
    }
    crate::families::moe::cpu_split_executor_wide(experts, params.batch_size)
        .ok_or_else(|| split_error("the executor sealed into this call is no longer attached"))
}

fn split_shape_conflict(
    executor: &CpuSplitExecutor,
    n_exp: usize,
    k_top: usize,
    gate_up_k: usize,
    down_m: usize,
    down_k: usize,
    mi: usize,
) -> Option<&'static str> {
    if k_top != SPLIT_TOP_K
        || n_exp != executor.placement().len()
        || gate_up_k != down_m
        || down_k != mi
        || executor.dim() != down_m
        || executor.mi() != mi
    {
        return Some(
            "the executor plan does not match this layer's expert count, top-k or hidden/intermediate geometry",
        );
    }
    None
}

/// Why a one-row decode call cannot run in the CPU split mode, or `None`.
/// One predicate for sealing, lowering and execution.
///
/// The mode keeps the router, the shared expert and the canonical combine on
/// the original grouped QT44/QT53 or expert-slab route unchanged: only the
/// routed GPU tail reads the executor's masked ids (the slab K1 planner and
/// kernels skip an id outside `[0, n_experts)` with no load and no store), the
/// cold slots' unweighted F32 rows are installed into `down_expanded`, and the
/// unchanged BF16 combine folds the original route once. Anything that changes
/// that bracket refuses.
pub(super) fn cpu_split_decode_conflict(
    params: &MoeParams<'_>,
    route: Option<MoeRouteCapability>,
    cpu_deferred_experts: bool,
    executor: Option<&CpuSplitExecutor>,
) -> Option<&'static str> {
    if params.batch_size != 1 {
        return Some("indexed decode is one row; windows of two to four rows are the grouped prefill program");
    }
    if cpu_deferred_experts || params.cpu_deferred_experts {
        return Some("CPU tier split and CPU-deferred routed experts are mutually exclusive");
    }
    if !route.is_some_and(MoeRouteCapability::is_qt44_qt53_grouped) {
        return Some("requires the original grouped QT44/QT53 or expert-slab route");
    }
    if !matches!(
        params.recipe,
        MoeRecipe::SoftmaxGatedShared {
            bf16_round_trip: true,
            shared_after_combine: true,
        }
    ) {
        return Some("requires the bf16 shared_after_combine softmax recipe");
    }
    if params.ep_mode != crate::families::moe::MoeEpMode::None
        || params.routed_out.is_some()
        || params.defer_routed_combine
        || params.ep_row_gather.is_some()
    {
        return Some("is a single-device call with the canonical routed combine (no EP, partial, deferred or gathered combine)");
    }
    if params.expert_down_awq_ptrs.is_some()
        || params.routed_gate_up_paro.is_some()
        || params.routed_down_paro.is_some()
    {
        return Some("does not admit AWQ or Paro sidecars");
    }
    if !params.down_writes_expanded() {
        return Some("requires an expanded-output (non self-combining) down projection");
    }
    executor.and_then(|executor| {
        split_shape_conflict(
            executor,
            params.n_exp,
            params.k,
            params.routed_gate_up_k,
            params.routed_down_m,
            params.routed_down_k,
            params.mi,
        )
    })
}

/// Existing-oracle-override presence on an exact oracle arch: the canonical
/// [`rdna_compute::prefill_invariant::qwen4_wide_verify_override_present`]
/// signal (runtime override — including the `Some(8)` baseline — or the parsed
/// developer var). Pure presence signal, never a capacity value.
fn cpu_lab_override_present_for_arch(arch: &str) -> bool {
    rdna_compute::prefill_invariant::qwen4_wide_verify_override_present(arch)
}

/// Executor lookup for the sealed join mode: the default-4 gate first, then
/// the lab-wide storage gate. Both read allocated capacity, never a constant.
fn cpu_split_executor_for_join(
    experts: &dyn crate::families::moe::RoutedExpertWeights,
    rows: usize,
    scope: CpuWideVerifyScope,
) -> Option<&CpuSplitExecutor> {
    if let Some(executor) = crate::families::moe::cpu_split_executor(experts, rows) {
        return Some(executor);
    }
    let capacity = experts
        .cpu_split()
        .map(|executor| executor.capacity_rows())
        .unwrap_or(0);
    if !crate::families::moe::cpu_wide_window_qualified(true, rows, rows, scope, capacity) {
        return None;
    }
    crate::families::moe::cpu_split_executor_wide(experts, rows)
}

/// Pure width truth of [`cpu_split_prefill_conflict`] for tests: the
/// structural refusal for a `rows`-row window given an allocated `capacity`,
/// without params or an executor. Wiring only, not numeric proof.
#[cfg(test)]
pub(super) fn cpu_split_prefill_conflict_width(
    rows: usize,
    capacity: usize,
) -> Option<&'static str> {
    let structural =
        NARROW_VERIFY_MAX_ROWS.max(capacity.min(crate::families::moe::CPU_SPLIT_STORAGE_MAX_ROWS));
    if rows == 0 || rows > structural {
        if rows > NARROW_VERIFY_MAX_ROWS {
            return Some(
                "the window exceeds the executor's allocated CPU window (wider windows are ordinary wide prefill)",
            );
        }
        return Some(
            "the window exceeds the four-row cap (wider windows are ordinary wide prefill)",
        );
    }
    None
}

/// Prefill counterpart of [`cpu_split_decode_conflict`]: the default route
/// admits a window of one to [`NARROW_VERIFY_MAX_ROWS`] rows; an explicit
/// lab-wide scope admits a whole logical window of one to
/// [`crate::families::moe::CPU_SPLIT_STORAGE_MAX_ROWS`] rows when the
/// executor's allocated capacity covers it. Unqualified wider windows never
/// reach it (they stay ordinary wide prefill).
pub(super) fn cpu_split_prefill_conflict(
    params: &MoePrefillParams<'_>,
    route: Option<MoeRouteCapability>,
    cpu_deferred_experts: bool,
    executor: Option<&CpuSplitExecutor>,
) -> Option<&'static str> {
    // Structural width check only (storage, not scope): the call fits when it
    // is inside the default 4-row window or inside the executor's allocated
    // capacity (up to 64). The lab-wide SCOPE check (default vs explicit
    // verify width) lives at seal (`select_prefill`, which measures the frozen
    // row-exact rows from the arch) and at execution (`validate_split_prefill`,
    // which measures them from the GPU): this predicate stays a pure function
    // of the call for the lowering seam, which has no device or arch handle.
    // Ordinary prefill keeps its refusal: without an admitted executor the
    // seal never reaches this predicate with a wide window.
    let capacity = executor
        .map(|executor| executor.capacity_rows())
        .unwrap_or(0);
    let structural =
        NARROW_VERIFY_MAX_ROWS.max(capacity.min(crate::families::moe::CPU_SPLIT_STORAGE_MAX_ROWS));
    if params.batch_size == 0 || params.batch_size > structural {
        if params.batch_size > NARROW_VERIFY_MAX_ROWS {
            return Some(
                "the window exceeds the executor's allocated CPU window (wider windows are ordinary wide prefill)",
            );
        }
        return Some(
            "the window exceeds the four-row cap (wider windows are ordinary wide prefill)",
        );
    }
    if cpu_deferred_experts || params.cpu_deferred_experts {
        return Some("CPU tier split and CPU-deferred routed experts are mutually exclusive");
    }
    if !route.is_some_and(MoeRouteCapability::is_qt44_qt53_grouped) {
        return Some("requires the original grouped QT44/QT53 or expert-slab route");
    }
    if !matches!(
        params.recipe,
        MoeRecipe::SoftmaxGatedShared {
            bf16_round_trip: true,
            shared_after_combine: true,
        }
    ) {
        return Some("requires the bf16 shared_after_combine softmax recipe");
    }
    if !matches!(params.prelude.normalization, MoeNormalization::Provided) {
        return Some("requires a provided natural activation (no in-program norm)");
    }
    if params.ep_row_gather.is_some()
        || params.ep_tile_exchange.is_some()
        || params.routed_out.is_some()
        || !matches!(
            params.prelude.route,
            super::sealed_moe::PrefillRouteMode::Replicated
        )
    {
        return Some(
            "is a replicated single-device mode (no EP gather/exchange/partial/root route)",
        );
    }
    if params.expert_stage_ptrs.is_some()
        || params.expert_down_awq_ptrs.is_some()
        || params.paro_gate_up.is_some()
        || params.paro_down.is_some()
    {
        return Some("does not admit a staged expert table or AWQ/Paro sidecars");
    }
    if !crate::families::moe::moe_down_writes_expanded(
        params.dtypes.routed_down,
        params.expert_dtype_tags.is_some(),
    ) {
        return Some("requires an expanded-output (non self-combining) down projection");
    }
    executor.and_then(|executor| {
        split_shape_conflict(
            executor,
            params.n_exp,
            params.k_top,
            params.gate_up_k,
            params.down_m,
            params.down_k,
            params.mi,
        )
    })
}

/// Eager-only gate of the split: it blocks on the host between the router and
/// the routed tail, so a graph capture (ordinary, specialized verify, replay
/// or raw stream), a replay recording, a retained body, retained PM4/AQL
/// transports and a shadow-validated body all refuse, before any state of this
/// call is mutated.
fn split_run_gate(gpu: &Gpu) -> Result<(), DispatchError> {
    CpuSplitRunGate::observe(gpu)?.require_eager()?;
    if gpu.replay.retained_body_active() {
        return Err(split_error(
            "is a host-synchronous step and is refused under a retained replay body; run eager (replay.backend=hip, memory.offload_exec=pcie)",
        ));
    }
    Ok(())
}

/// The expanded slot scratch must hold `rows * k * dim` F32 values.
fn split_expanded_extent(
    expanded: &GpuTensor,
    rows: usize,
    k_top: usize,
    dim: usize,
) -> Result<(), DispatchError> {
    let need = rows
        .checked_mul(k_top)
        .and_then(|slots| slots.checked_mul(dim))
        .and_then(|values| values.checked_mul(4))
        .ok_or_else(|| split_error("expanded slot extent overflows"))?;
    if expanded.buf.size() < need {
        return Err(split_error(format!(
            "down_expanded holds {} bytes, the slot rows need {need}",
            expanded.buf.size()
        )));
    }
    Ok(())
}

/// Everything the split needs from the device, checked before the first
/// launch (and again by `PrepareSplit`): eager state, the route width the
/// router actually produces, and that the routed tail leaves F32 slot rows.
pub(super) fn validate_split_decode(
    gpu: &Gpu,
    params: &MoeParams<'_>,
    selection: &MoeDecodeSelection,
) -> Result<(), DispatchError> {
    split_run_gate(gpu)?;
    let executor = decode_split_executor(params)?;
    if let Some(reason) = cpu_split_decode_conflict(
        params,
        selection.route,
        selection.cpu_deferred_experts,
        Some(executor),
    ) {
        return Err(split_error(reason));
    }
    // The ordinary decode route produces exactly `rows` rows; a canonicalised
    // (prefill-invariant) route width is a different route contract.
    if gpu.route_rows(params.batch_size) != params.batch_size {
        return Err(split_error(
            "requires route rows equal to the window rows (prefill-invariant scope is refused)",
        ));
    }
    split_expanded_extent(
        params.down_expanded,
        params.batch_size,
        params.k,
        params.routed_down_m,
    )
}

/// Prefill counterpart of [`validate_split_decode`]. The routed down must be
/// the decode-kernel F32 expanded rows in slot order `[row * 10 + rank, dim]`
/// ([`super::qt44_qt53_prefill::writes_slot_ordered_rows`]), never grouped BF16
/// rows or a row-fold target; this device/width cannot satisfy that on a
/// symmetric-IU4 or WMMA grouped down, and then the call refuses rather than
/// silently changing arithmetic.
pub(super) fn validate_split_prefill(
    gpu: &Gpu,
    params: &MoePrefillParams<'_>,
    selection: &MoePrefillSelection,
) -> Result<(), DispatchError> {
    // Execution recheck reads the sealed call's per-call ctx flag via the
    // `cpu_wide_verify_exec` out-param pattern below: the sealed_moe caller
    // passes `self.dispatch_ctx.cpu_wide_verify` through. Until that call
    // site threads it, default-false (fail-closed for wide).
    return validate_split_prefill_inner(gpu, params, selection, false);
}

pub(super) fn validate_split_prefill_exec(
    gpu: &Gpu,
    params: &MoePrefillParams<'_>,
    selection: &MoePrefillSelection,
    cpu_wide_verify: bool,
) -> Result<(), DispatchError> {
    validate_split_prefill_inner(gpu, params, selection, cpu_wide_verify)
}

fn validate_split_prefill_inner(
    gpu: &Gpu,
    params: &MoePrefillParams<'_>,
    selection: &MoePrefillSelection,
    cpu_wide_verify: bool,
) -> Result<(), DispatchError> {
    split_run_gate(gpu)?;
    let executor = prefill_split_executor(params)?;
    if let Some(reason) = cpu_split_prefill_conflict(
        params,
        selection.route,
        selection.cpu_deferred_experts,
        Some(executor),
    ) {
        return Err(split_error(reason));
    }
    // Execution-time lab-wide scope check: a wide (> 4-row) window runs only
    // inside the sealed call's per-call eager-verify scope (`cpu_wide_verify`
    // ctx flag) with the canonical override present and allocated capacity.
    // A whole 64-row ordinary prefill refuses here even if a wide executor is
    // attached — `rows == route_rows` alone never implies verify. No hidden
    // fallback, no clamp.
    if params.batch_size > NARROW_VERIFY_MAX_ROWS {
        let scope = CpuWideVerifyScope::from_lab_presence(
            cpu_wide_verify && cpu_lab_override_present_for_arch(&gpu.arch),
            rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows(&gpu.arch),
        );
        if !crate::families::moe::cpu_wide_window_qualified(
            true,
            params.batch_size,
            params.batch_size,
            scope,
            executor.capacity_rows(),
        ) {
            return Err(split_error(
                "the window exceeds the qualified CPU window for this scope (wider windows are ordinary wide prefill)",
            ));
        }
    }
    if gpu.route_rows(params.batch_size) != params.batch_size {
        return Err(split_error(
            "requires route rows equal to the window rows (prefill-invariant scope is refused)",
        ));
    }
    if !super::qt44_qt53_prefill::writes_slot_ordered_rows(
        gpu,
        params,
        selection.uses_grouped_rows(),
    ) {
        return Err(split_error(
            "this device and width take a grouped BF16 down, not the F32 slot-ordered expanded rows the canonical combine and the cold-slot install require",
        ));
    }
    split_expanded_extent(
        params.down_expanded,
        params.batch_size,
        params.k_top,
        params.down_m,
    )
}

/// A grouped FP8 launcher reads one immutable contiguous block table, not the
/// legacy device pointer indirection. Prove that both tables name those same
/// live blocks before constructing the launcher's borrowed base view.
fn validate_fp8_expert_blocks(params: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    let (first_gate, first_down) = params
        .routed_experts
        .get(0)
        .ok_or_else(|| DispatchError::Hip("sealed fp8 moe: empty expert block table".into()))?;
    let gate_stride = rdna_compute::fp8_e4m3::b128_block_stride(2 * params.mi, params.gate_up_k)
        .ok_or_else(|| DispatchError::Hip("sealed fp8 moe: gate stride overflows".into()))?;
    let down_stride = rdna_compute::fp8_e4m3::b128_block_stride(params.down_m, params.down_k)
        .ok_or_else(|| DispatchError::Hip("sealed fp8 moe: down stride overflows".into()))?;
    let gate_base = first_gate.buf.buf.as_ptr() as usize;
    let down_base = first_down.buf.buf.as_ptr() as usize;
    for expert in 0..params.n_exp {
        let (gate, down) = params.routed_experts.get(expert).ok_or_else(|| {
            DispatchError::Hip("sealed fp8 moe: incomplete expert block table".into())
        })?;
        let address = |base: usize, stride: usize| {
            expert
                .checked_mul(stride)
                .and_then(|offset| base.checked_add(offset))
        };
        if Some(gate.buf.buf.as_ptr() as usize) != address(gate_base, gate_stride)
            || Some(down.buf.buf.as_ptr() as usize) != address(down_base, down_stride)
            || gate_base % 256 != 0
            || down_base % 256 != 0
            || gate.m != 2 * params.mi
            || gate.k != params.gate_up_k
            || down.m != params.down_m
            || down.k != params.down_k
            || gate.dtype != rdna_compute::DType::FP8E4M3B128
            || down.dtype != rdna_compute::DType::FP8E4M3B128
            || gate.buf.buf.size() < gate_stride
            || down.buf.buf.size() < down_stride
            || gate.rotation.is_some()
            || down.rotation.is_some()
            || gate.awq_scale.is_some()
            || down.awq_scale.is_some()
        {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "fp8-immutable-contiguous-expert-blocks",
                arch: "gfx942",
                quant: "FP8E4M3B128",
            });
        }
    }
    Ok(())
}

/// Resolve all grouped-prefill choices without a GPU or an instruction list.
///
/// The prefill [`MoePrefillResolution`] reads owner-derived residency via
/// `routed_host_mapped_projections` (tier identity first, then
/// `routed_experts.host_mapped_projections()`). A narrow MTP verify window over
/// host-mapped experts takes the indexed Path 1, except a host-mapped uniform
/// MQ4 gate_up+down on gfx1201, which is re-admitted to Path 2 (grouped-WMMA).
/// Residency comes from the packed weight owners or an explicit tier identity
/// (its layer-wide `any_host`, never expert 0) — never inferred from
/// borrowed tensor kinds — and resident experts keep the legacy route.
pub(super) fn select_prefill(
    ctx: &DispatchCtx,
    params: &MoePrefillParams<'_>,
) -> Result<MoePrefillSelection, DispatchError> {
    let fp8_route = super::select_fp8_grouped_route(
        ctx,
        &params.dtypes,
        params.route_policy,
        params.n_exp,
        params.k_top,
        params.gate_up_k,
        params.mi,
        params.prelude.shared.as_ref().map_or(0, |s| s.intermediate),
        params.gate_up_k,
        params.down_m,
        params.down_k,
        params.expert_dtype_tags,
        params.expert_down_awq_ptrs.is_some()
            || params.down_awq_scale.is_some()
            || params.paro_gate_up.is_some()
            || params.paro_down.is_some()
            || params
                .prelude
                .shared
                .as_ref()
                .is_some_and(|s| projection_has_sidecars(&params.prelude.router, &s.weights)),
    )?;
    if let Some(route) = fp8_route {
        if params.recipe
            != (MoeRecipe::SoftmaxGatedShared {
                bf16_round_trip: true,
                shared_after_combine: true,
            })
            || params.cpu_deferred_experts
            || crate::families::moe::cpu_split_executor(params.routed_experts, params.batch_size)
                .is_some()
            || params.routed_out.is_some()
            || params.expert_stage_ptrs.is_some()
            || params.ep_row_gather.is_some()
            || params.ep_tile_exchange.is_some()
            || !matches!(
                params.prelude.route,
                super::sealed_moe::PrefillRouteMode::Replicated
            )
            || crate::families::moe::routed_host_mapped(params.routed_experts)
        {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "fp8-sealed-replicated-bf16-only",
                arch: "gfx942",
                quant: "FP8E4M3B128",
            });
        }
        validate_fp8_expert_blocks(params)?;
        return Ok(MoePrefillSelection {
            resolution: MoePrefillResolution::sealed_fp8(),
            path2_m_total: params.m_total_max,
            force_mq4_grouped_fp16: false,
            coalesced_verify: false,
            route: Some(route),
            cpu_deferred_experts: false,
            cpu_split: false,
            cpwait: false,
        });
    }
    let route = (matches!(params.recipe, MoeRecipe::SoftmaxGatedShared { .. })
        && params.routed_out.is_none()
        && params.paro_gate_up.is_none()
        && params.paro_down.is_none()
        && matches!(
            params.prelude.route,
            super::sealed_moe::PrefillRouteMode::Replicated
        ))
    .then(|| {
        super::select_grouped_route(
            &params.dtypes,
            params.n_exp,
            params.k_top,
            params.gate_up_k,
            params.mi,
            params
                .prelude
                .shared
                .as_ref()
                .map_or(0, |shared| shared.intermediate),
            params.gate_up_k,
            params.down_m,
            params.down_k,
            params.expert_dtype_tags,
            params.expert_down_awq_ptrs.is_some()
                || params.down_awq_scale.is_some()
                || params.prelude.shared.as_ref().is_some_and(|shared| {
                    projection_has_sidecars(&params.prelude.router, &shared.weights)
                }),
            params.route_policy,
            ctx.arch.arch(),
        )
    })
    .flatten();
    if params.route_policy.is_some() && route.is_none() {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "specialized-prefill-route-unsupported",
            arch: "",
            quant: "QT44/QT53",
        });
    }
    if route.is_none() {
        super::reject_mq4g128v2_moe(&params.dtypes)?;
        // qwen35 qt53 routed-down typed fallback (no route policy only): refuse
        // tags / AWQ / Paro / CPU-deferred operands before anything is lowered.
        super::check_qt53_prefill_fallback(params)?;
    }
    let (host_gate_up, host_down) =
        crate::families::moe::routed_host_mapped_projections(params.routed_experts);
    let resolution = MoePrefillResolution::resolve_with_batch_ctx(
        &params.dtypes,
        &ctx.arch,
        &ctx.flags,
        ctx.batch_size,
        ctx.workload == crate::context::DispatchWorkload::SpeculativeVerify,
        host_gate_up || host_down,
    );
    resolution.check_generic_route(route)?;
    if route.is_some_and(MoeRouteCapability::is_slab) {
        // The slab route is a replicated, single-device lowering (resident or staged experts)
        // lowering whose K2 arm is the grouped protocol: nothing else may
        // ride on it, and a device without grouped GEMM refuses.
        if !resolution.use_path2
            || params.cpu_deferred_experts
            || params.routed_out.is_some()
            || params.ep_row_gather.is_some()
            || params.ep_tile_exchange.is_some()
            || !matches!(
                params.prelude.route,
                super::sealed_moe::PrefillRouteMode::Replicated
            )
        {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "slab-sealed-replicated-grouped-only",
                arch: if ctx.arch.is_gfx1100() {
                    "gfx1100"
                } else if ctx.arch.is_gfx1151() {
                    "gfx1151"
                } else {
                    "gfx1201"
                },
                quant: "MQ4/W3/E8P-I4 slab",
            });
        }
    }
    // The distinct CPU-cold/GPU-hot split. Default route: a window of at most
    // four rows over an attached executor with allocated capacity; wider
    // windows are ordinary wide prefill. Inside an explicit lab-wide
    // eager-verify scope (`cpu_wide_verify_rows > 8`, the frozen oracle
    // override) a whole logical 1..64-row window qualifies when the executor's
    // allocated capacity covers it — one forward, full-window state/capture,
    // never chunked. Every admission refusal is raised here, before the call
    // is published; a refused wide window falls back to ordinary wide prefill
    // only when no executor was bound (the `None` arm), never silently after
    // an executor was admitted.
    // This call's lab-wide scope (G2/G4 presence gate): the existing oracle
    // override must be PRESENT (env var on an exact oracle arch — never the
    // `few_row_limit` value, which is a GPU projection-route limit, not CPU
    // storage) and this must be an eager verify call (verify workload, known
    // whole-window route rows). `verify_rows` below is informational only
    // (GPU arm width), never a clamp: allocated executor capacity binds. Route rows come
    // from the dispatch context's batch width (0 = unknown = default route):
    // the forward loop threads the canonical `route_rows` there, so a short
    // tail of a long prefill keeps `rows != route_rows` and the GPU route.
    // Eager scope here is NOT `ctx.workload` (the FN forward seals with a
    // `Standard` context; only qwen35 sets `SpeculativeVerify`): it is the
    // forward threading real canonical route rows (`ctx.batch_size ==
    // params.batch_size`, non-decode width) with the override present. A short
    // tail of a long prefill keeps `rows != route_rows` and the GPU route.
    let cpu_wide_scope = CpuWideVerifyScope::from_lab_presence(
        ctx.cpu_wide_verify()
            && cpu_lab_override_present_for_arch(ctx.arch.arch())
            && ctx.batch_size == params.batch_size
            && params.batch_size >= 2,
        rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows(ctx.arch.arch()),
    );
    let cpu_split = match crate::families::moe::cpu_split_executor(
        params.routed_experts,
        params.batch_size,
    ) {
        Some(executor) => {
            if let Some(reason) = cpu_split_prefill_conflict(
                params,
                route,
                params.cpu_deferred_experts,
                Some(executor),
            ) {
                return Err(DispatchError::Hip(format!(
                    "sealed moe: CPU tier split: {reason}"
                )));
            }
            true
        }
        None => {
            // Storage-plus-scope gate for the actual CPU-owned expert window:
            // armed whole-routing-window calls only (tails keep the GPU route),
            // qualified width only, allocated capacity only. Ordinary prefill
            // and lanes observe the default scope (`<= 8`) and stay `None`
            // here, exactly as before.
            let capacity = params
                .routed_experts
                .cpu_split()
                .map(|executor| executor.capacity_rows())
                .unwrap_or(0);
            // `ctx.batch_size == 0` (unknown: lanes, S1, tests, non-FN seals)
            // never qualifies wide: only the FN forward loop threads real
            // route rows. `route_rows` below falls back to `params.batch_size`
            // only to keep the whole-window predicate total; an unknown route
            // with a default scope still refuses via the scope gate.
            let route_rows = if ctx.batch_size == 0 {
                usize::MAX
            } else {
                ctx.batch_size
            };
            if !crate::families::moe::cpu_wide_window_qualified(
                true,
                params.batch_size,
                route_rows,
                cpu_wide_scope,
                capacity,
            ) {
                false
            } else {
                match crate::families::moe::cpu_split_executor_wide(
                    params.routed_experts,
                    params.batch_size,
                ) {
                    Some(executor) => {
                        if let Some(reason) = cpu_split_prefill_conflict(
                            params,
                            route,
                            params.cpu_deferred_experts,
                            Some(executor),
                        ) {
                            return Err(DispatchError::Hip(format!(
                                "sealed moe: CPU tier split: {reason}"
                            )));
                        }
                        if ctx.batch_size != params.batch_size {
                            return Err(DispatchError::Hip(
                                "sealed moe: CPU tier split: requires route rows equal to the window rows (prefill-invariant scope is refused)".into(),
                            ));
                        }
                        true
                    }
                    None => false,
                }
            }
        }
    };
    let _total_slots = params
        .batch_size
        .checked_mul(params.k_top)
        .ok_or_else(|| DispatchError::Hip("sealed moe: prefill slot count overflows".into()))?;
    let path2_m_total = if resolution.use_path2 {
        params.m_total_max
    } else {
        0
    };
    let force_mq4_grouped_fp16 = resolution.force_mq4_grouped_fp16 || params.force_mq4_grouped_fp16;
    // Lower bound of 2 is load-bearing: a 1-row window is decode, not a verify
    // window, so it must not take the coalesced/Path-2 re-admit.
    let coalesced_verify = ctx.arch.is_gfx1201()
        && ctx.workload == crate::context::DispatchWorkload::SpeculativeVerify
        && (2..=NARROW_VERIFY_MAX_ROWS).contains(&ctx.batch_size)
        && (2..=NARROW_VERIFY_MAX_ROWS).contains(&params.batch_size)
        && route.is_none()
        && resolution.use_path2;
    let compact_ep = matches!(
        params.prelude.route,
        super::sealed_moe::PrefillRouteMode::ProduceRoot { .. }
            | super::sealed_moe::PrefillRouteMode::AdoptRoot { .. }
    );
    if compact_ep && resolution.down_path0 {
        return Err(DispatchError::Hip(
            "root-routed EP prefill requires an expanded-output down projection".into(),
        ));
    }
    if params.cpu_deferred_experts {
        // Width gate (D1 prefill twin of the decode gate above): `ctx.batch_size`
        // carries the canonical route rows (0 = unknown = default route, which
        // refuses wide). Whole logical 1..64-row windows qualify only with an
        // explicit lab-wide scope plus prewarmed scratch; tails keep the GPU
        // route via the seal, and unknown-route wide windows refuse here.
        {
            let scope = CpuWideVerifyScope::from_lab_presence(
                ctx.cpu_wide_verify()
                    && cpu_lab_override_present_for_arch(ctx.arch.arch())
                    && ctx.batch_size == params.batch_size
                    && params.batch_size >= 2,
                rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows(ctx.arch.arch()),
            );
            let route_rows = if ctx.batch_size == 0 {
                usize::MAX
            } else {
                ctx.batch_size
            };
            if !crate::families::moe::cpu_d1_wide_window_qualified(
                true,
                params.batch_size,
                route_rows,
                scope,
                crate::families::moe::cpu_d1_prewarmed_scratch_rows(
                    params.down_m,
                    params.mi,
                    params.k_top,
                    ctx.device_id(),
                ),
            ) {
                return Err(DispatchError::Hip(
                    "sealed moe: CPU-deferred routed experts: the window exceeds the qualified CPU window for this scope (wider windows keep the GPU route)".into(),
                ));
            }
        }
        // A CPU-deferred narrow window keeps the GPU router/shared stages and
        // the arch's CPU splice; it must not be an EP partial (the routed
        // contribution would be folded twice) and must not rely on the grouped
        // routed tail it is about to omit.
        if compact_ep || params.routed_out.is_some() {
            return Err(DispatchError::Hip(
                "CPU-deferred routed experts require a replicated (non-EP) prefill".into(),
            ));
        }
        // A BF16 recipe is admitted only on the grouped QT44/QT53 route: the
        // router and shared stages this program still lowers carry its BF16
        // round trips themselves, and the omitted routed stages' round trips
        // (gate/up, activation, weighted combine) are the CPU splice's, which
        // receives the recipe flag. The generic route also rounds the rotated
        // input, which the CPU does not emulate.
        if params.recipe.bf16_round_trip()
            && !route.is_some_and(MoeRouteCapability::is_qt44_qt53_grouped)
        {
            return Err(DispatchError::Hip(
                "CPU-deferred routed experts emulate a recipe's bf16 round trips only on the grouped QT44/QT53 route"
                    .into(),
            ));
        }
    }
    Ok(MoePrefillSelection {
        resolution,
        path2_m_total,
        force_mq4_grouped_fp16,
        coalesced_verify,
        route,
        cpu_deferred_experts: params.cpu_deferred_experts,
        cpu_split,
        cpwait: cpu_split
            && cpu_split_executor_for_join(
                params.routed_experts,
                params.batch_size,
                cpu_wide_scope,
            )
            .is_some_and(CpuSplitExecutor::cpwait_active),
    })
}

// ───────────────────── opt-in expert IU4 prefill route ─────────────────────
//
// `HIPFIRE_MOE_SYM_IU4` (default ON by capability; `=0` kills it). Only a prefill
// whose arch bound the loader capability (`MoePrefillParams::expert_iu4`) ever
// consults any of this; an unbound call (explicit kill or a non-artifact model)
// returns [`ExpertIu4Decision::Unbound`] before reading a flag, the device or an
// operand, so it adds no launch, query or allocation.
//
// The decision is made ONCE per invocation (first routed stage, the scatter) from
// [`ExpertIu4Facts`], cached in [`MoeStepState`] and read by every later stage, so
// the scatter, gate/up, down and combine can never disagree and no stage falls
// back after an expert IU4 launch. Every capacity, scratch and shape check is part
// of that one pure decision and precedes the first IU4 launch; hipGraph capture and
// host-mapped experts are admitted, only Redline recording is excluded.

use crate::families::moe_expert_iu4::{
    expert_iu4_admission, ExpertIu4Inputs, ExpertIu4Reject, ExpertIu4Spec, EXPERT_IU4_MIN_ROWS,
};

/// Grouped-row tile of the symmetric GEMM launchers (`grouped_rows % 16 == 0`).
const EXPERT_IU4_TILE_ROWS: usize = 16;
/// `x_row_div * x_src_rows` bound of the PM symmetric GEMMs
/// (`QWEN4_MOE_SYM_PM_MAX_SLOTS`, private to rdna-compute).
const EXPERT_IU4_MAX_SLOTS: usize = 1 << 22;
/// HIP grid-y extent the PM combine's `ceil(dim / 256)` must fit.
const EXPERT_IU4_GRID_Y_MAX: usize = 65_535;
/// Threads per workgroup of the PM combine (one output column each).
const EXPERT_IU4_COMBINE_BLOCK: usize = 256;

/// Bytes of a BF16 `[rows, width]` view, `None` on overflow.
pub(super) fn bf16_grouped_view_bytes(rows: usize, width: usize) -> Option<usize> {
    rows.checked_mul(width)?.checked_mul(2)
}

/// Why an IU4-capable call stays on the qt53 F32 fallback although it was not
/// refused by [`expert_iu4_admission`]: a call shape this route does not own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExpertIu4Exclusion {
    /// A specialized route (route policy / sealed grouped route) owns the call.
    SpecializedRoute,
    /// CPU-deferred routed experts.
    CpuDeferred,
    /// CPU-cold/GPU-hot split.
    CpuSplit,
    /// EP partial accumulator (`routed_out`).
    RoutedOut,
    /// EP row gather.
    EpRowGather,
    /// EP tile exchange.
    EpTileExchange,
    /// Root-routed EP prefill (canonical slot-order combine).
    CanonicalSlotOrder,
    /// Staged (paged / host-staged) expert pointer tables.
    StagedExpertTables,
    /// Paro rotation sidecars.
    Paro,
    /// The loader spec disagrees with this call's geometry or dtypes.
    SpecMismatch,
    /// The normalized activation is not F32.
    ActivationDtype,
    /// A buffer, scratch, extent or kernel domain limit (named) is below what the
    /// route needs.
    Capacity(&'static str),
    /// A Redline replay recording is capturing this call (not a hipGraph capture,
    /// which is admitted). Plain-AR decode is the only recorded forward, so a
    /// prefill-sized call never meets it in the shipped lifecycle.
    RedlineRecording,
}

impl ExpertIu4Exclusion {
    pub(super) const fn reason(&self) -> &'static str {
        match self {
            Self::SpecializedRoute => "a specialized route owns this call",
            Self::CpuDeferred => "CPU-deferred routed experts",
            Self::CpuSplit => "CPU-cold/GPU-hot split",
            Self::RoutedOut => "EP partial accumulator (routed_out)",
            Self::EpRowGather => "EP row gather",
            Self::EpTileExchange => "EP tile exchange",
            Self::CanonicalSlotOrder => "root-routed EP canonical slot-order combine",
            Self::StagedExpertTables => "staged expert pointer tables",
            Self::Paro => "Paro rotation sidecars",
            Self::SpecMismatch => "loader spec disagrees with the call geometry or dtypes",
            Self::ActivationDtype => "normalized activation is not F32",
            Self::Capacity(_) => "a buffer, scratch or kernel limit is below the route's need",
            Self::RedlineRecording => "a Redline recording is active",
        }
    }

    /// Stable log class (distinct from every [`ExpertIu4Reject`] class).
    const fn class(&self) -> usize {
        match self {
            Self::SpecializedRoute => 19,
            Self::CpuDeferred => 21,
            Self::CpuSplit => 22,
            Self::RoutedOut => 23,
            Self::EpRowGather => 24,
            Self::EpTileExchange => 25,
            Self::CanonicalSlotOrder => 26,
            Self::StagedExpertTables => 27,
            Self::Paro => 28,
            Self::SpecMismatch => 29,
            Self::ActivationDtype => 30,
            Self::Capacity(_) => 31,
            Self::RedlineRecording => 32,
        }
    }
}

/// Why the IU4 route is not taken: the pure admission or a call-shape exclusion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExpertIu4Fallback {
    Admission(ExpertIu4Reject),
    Exclusion(ExpertIu4Exclusion),
}

impl ExpertIu4Fallback {
    pub(super) const fn reason(&self) -> &'static str {
        match self {
            Self::Admission(reject) => reject.reason(),
            Self::Exclusion(exclusion) => exclusion.reason(),
        }
    }
}

/// Geometry proven by the decision; every IU4 stage launches exactly these extents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExpertIu4Plan {
    /// Physical rows (`batch_size`); `route_rows` only gated the floor.
    pub(super) rows: usize,
    /// Grouped (padded) rows: the scatter's `path2_m_total`.
    pub(super) grouped_rows: usize,
    /// `rows * k_top`.
    pub(super) total_slots: usize,
    /// Model width H (gate/up `k`, down `m`).
    pub(super) hidden: usize,
    /// Routed width I (gate/up output is `2 * I`, down `k`).
    pub(super) intermediate: usize,
    pub(super) k_top: usize,
    /// Routed experts are host-mapped: both PM GEMM launchers take the NT8 entries.
    pub(super) host_mapped: bool,
}

impl ExpertIu4Plan {
    /// The stage's operands still describe the call this plan was proven for.
    pub(super) fn matches(&self, p: &MoePrefillParams<'_>) -> bool {
        p.batch_size == self.rows
            && p.m_total_max == self.grouped_rows
            && p.gate_up_k == self.hidden
            && p.down_m == self.hidden
            && p.mi == self.intermediate
            && p.down_k == self.intermediate
            && p.k_top == self.k_top
            && p.routed_out.is_none()
            && expert_iu4_host_mapped(p) == self.host_mapped
    }
}

/// The single per-invocation verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExpertIu4Decision {
    /// No capability bound by the arch (default): this route does not exist.
    Unbound,
    /// The existing qt53 F32 fallback runs, unchanged.
    Fallback(ExpertIu4Fallback),
    /// The IU4 sequence runs, from the scatter to the combine.
    Admitted(ExpertIu4Plan),
}

/// Call-shape exclusions (each `true` refuses the route).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ExpertIu4CallShape {
    pub(super) specialized_route: bool,
    pub(super) cpu_deferred: bool,
    pub(super) cpu_split: bool,
    pub(super) routed_out: bool,
    pub(super) ep_row_gather: bool,
    pub(super) ep_tile_exchange: bool,
    pub(super) canonical_slot_order: bool,
    pub(super) staged_expert_tables: bool,
    pub(super) paro: bool,
    /// A Redline recording is capturing this call (not a hipGraph capture).
    pub(super) redline_recording: bool,
}

/// Extents of the call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExpertIu4Dims {
    pub(super) rows: usize,
    pub(super) k_top: usize,
    pub(super) n_exp: usize,
    pub(super) hidden: usize,
    pub(super) intermediate: usize,
    pub(super) down_m: usize,
    pub(super) down_k: usize,
    pub(super) grouped_rows: usize,
}

/// Device bytes each operand holds (`DeviceBuffer::size`, the span every launcher checks).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ExpertIu4Held {
    pub(super) x_norm: usize,
    pub(super) x_batch: usize,
    pub(super) sorted_slot_index: usize,
    pub(super) inverse_perm: usize,
    pub(super) topk_weights: usize,
    pub(super) y_gate_up_grouped: usize,
    pub(super) y_down_grouped: usize,
}

/// Usable bytes of the two reserved IU4 producer sidecars (`None` = not allocated):
/// the smaller of the tracked reservation and the live buffer size. The producers
/// grow these in place when short, which must never happen inside the sequence
/// (it would invalidate captures and can allocate mid-prefill); the decision
/// therefore admits only rows the load-time reservation already covers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ExpertIu4Scratch {
    /// `int4_mmq_x_scratch`: the gate/up A4 record `[hidden / 128, rows]`.
    pub(super) gate_x: Option<usize>,
    /// `qwen4_moe_down_i4_scratch`: the down A4 record `[intermediate / 128, slots]`.
    pub(super) down_i4: Option<usize>,
}

/// Pure scratch preflight: both sidecars exist and hold the producers' exact
/// extents (`int4_mmq_reservation_bytes`, `qwen4_moe_down_i4_bytes`). The stable
/// grouping scratch is not read: the route reuses the fused k8 scatter.
pub(super) fn expert_iu4_scratch_ready(
    dims: &ExpertIu4Dims,
    scratch: &ExpertIu4Scratch,
) -> Result<(), &'static str> {
    let total_slots = dims.rows.checked_mul(dims.k_top).ok_or("rows")?;
    let gate = rdna_compute::scratch::int4_mmq_reservation_bytes(dims.hidden, dims.rows)
        .ok_or("int4_mmq_x_scratch")?;
    let down = rdna_compute::scratch::qwen4_moe_down_i4_bytes(dims.intermediate, total_slots)
        .ok_or("qwen4_moe_down_i4_scratch")?;
    match scratch.gate_x {
        Some(have) if have >= gate => {}
        _ => return Err("int4_mmq_x_scratch"),
    }
    match scratch.down_i4 {
        Some(have) if have >= down => {}
        _ => return Err("qwen4_moe_down_i4_scratch"),
    }
    Ok(())
}

/// Everything the verdict reads; built from the params and the GPU, evaluated purely.
#[derive(Clone, Copy, Debug)]
pub(super) struct ExpertIu4Facts {
    pub(super) capability: Option<(ExpertIu4Spec, bool)>,
    /// `Gpu::moe_sym_iu4_requested()`.
    pub(super) requested: bool,
    /// `Gpu::route_rows(batch_size)`: the measure the row floor ([`EXPERT_IU4_MIN_ROWS`]) reads.
    pub(super) route_rows: usize,
    pub(super) min_rows: usize,
    pub(super) use_path2: bool,
    pub(super) has_dtype_tags: bool,
    pub(super) has_down_awq: bool,
    /// A hipGraph capture is active (`graphs.capture_mode`, verify or replay
    /// capture). Informational: admitted, never a refusal.
    pub(super) capturing_graph: bool,
    /// Layer-wide host-mapped bit. Informational to admission (host-mapped experts
    /// are admitted); the plan carries it to both PM GEMM launchers (NT8 entries).
    pub(super) host_mapped: bool,
    pub(super) shape: ExpertIu4CallShape,
    pub(super) routed_gate_up: rdna_compute::DType,
    pub(super) routed_down: rdna_compute::DType,
    pub(super) x_norm_f32: bool,
    pub(super) dims: ExpertIu4Dims,
    pub(super) held: ExpertIu4Held,
    pub(super) scratch: ExpertIu4Scratch,
}

/// Pure capacity / kernel-domain check of the IU4 sequence; `Err` names the
/// first operand or limit that fails. Mirrors every launcher precondition
/// (`require_span`s, 16-row tiles, slot bound, 32-bit PM byte extents, combine
/// grid) so none of them can first fail after an IU4 launch.
pub(super) fn expert_iu4_capacity(
    dims: &ExpertIu4Dims,
    held: &ExpertIu4Held,
) -> Result<(), &'static str> {
    let total_slots = dims
        .rows
        .checked_mul(dims.k_top)
        .filter(|slots| *slots > 0)
        .ok_or("rows")?;
    if i32::try_from(dims.rows).is_err() || i32::try_from(dims.grouped_rows).is_err() {
        return Err("rows");
    }
    if total_slots > EXPERT_IU4_MAX_SLOTS {
        return Err("slots");
    }
    if dims.grouped_rows < total_slots || !dims.grouped_rows.is_multiple_of(EXPERT_IU4_TILE_ROWS) {
        return Err("grouped_rows");
    }
    if dims.hidden.div_ceil(EXPERT_IU4_COMBINE_BLOCK) > EXPERT_IU4_GRID_Y_MAX {
        return Err("hidden");
    }
    let fits = |bytes: Option<usize>, have: usize, name: &'static str| match bytes {
        Some(bytes) if bytes <= u32::MAX as usize && have >= bytes => Ok(()),
        _ => Err(name),
    };
    let f32_bytes = |a: usize, b: usize| a.checked_mul(b)?.checked_mul(4);
    fits(
        f32_bytes(dims.rows, dims.hidden),
        held.x_norm,
        "x_norm_batch",
    )?;
    fits(f32_bytes(dims.rows, dims.hidden), held.x_batch, "x_batch")?;
    fits(
        dims.grouped_rows.checked_mul(4),
        held.sorted_slot_index,
        "sorted_slot_index",
    )?;
    fits(
        total_slots.checked_mul(4),
        held.inverse_perm,
        "inverse_perm",
    )?;
    fits(
        total_slots.checked_mul(4),
        held.topk_weights,
        "topk_weights",
    )?;
    fits(
        bf16_grouped_view_bytes(dims.grouped_rows, dims.intermediate),
        held.y_gate_up_grouped,
        "y_gate_up_grouped",
    )?;
    fits(
        bf16_grouped_view_bytes(dims.grouped_rows, dims.hidden),
        held.y_down_grouped,
        "y_down_grouped",
    )?;
    Ok(())
}

/// The pure verdict. Order: unbound; [`expert_iu4_admission`] (requested, spec,
/// headers verified, Path 2, row floor [`EXPERT_IU4_MIN_ROWS`], tags, AWQ; hipGraph
/// capture and host-mapped experts are admitted); the call-shape exclusions (incl.
/// Redline recording); spec/geometry agreement; activation dtype; operand
/// capacity; reserved scratch. An exclusion is reported only after admission
/// accepted, so a Path-2 EP call is never labelled `NotPath2`.
pub(super) fn decide_expert_iu4(f: &ExpertIu4Facts) -> ExpertIu4Decision {
    use ExpertIu4Exclusion as X;
    let Some((spec, headers_verified)) = f.capability else {
        return ExpertIu4Decision::Unbound;
    };
    let inputs = ExpertIu4Inputs {
        requested: f.requested,
        symmetric_headers_verified: headers_verified,
        use_path2: f.use_path2,
        rows: f.route_rows,
        min_rows: f.min_rows,
        has_dtype_tags: f.has_dtype_tags,
        has_down_awq: f.has_down_awq,
        capturing_graph: f.capturing_graph,
        host_mapped: f.host_mapped,
    };
    if let Err(reject) = expert_iu4_admission(&spec, &inputs) {
        return ExpertIu4Decision::Fallback(ExpertIu4Fallback::Admission(reject));
    }
    let exclude = |x| ExpertIu4Decision::Fallback(ExpertIu4Fallback::Exclusion(x));
    let s = &f.shape;
    if let Some((_, x)) = [
        (s.specialized_route, X::SpecializedRoute),
        (s.cpu_deferred, X::CpuDeferred),
        (s.cpu_split, X::CpuSplit),
        (s.routed_out, X::RoutedOut),
        (s.ep_row_gather, X::EpRowGather),
        (s.ep_tile_exchange, X::EpTileExchange),
        (s.canonical_slot_order, X::CanonicalSlotOrder),
        (s.staged_expert_tables, X::StagedExpertTables),
        (s.paro, X::Paro),
        (s.redline_recording, X::RedlineRecording),
    ]
    .into_iter()
    .find(|(on, _)| *on)
    {
        return exclude(x);
    }
    let d = &f.dims;
    let agrees = spec.experts as usize == d.n_exp
        && spec.top_k as usize == d.k_top
        && spec.hidden as usize == d.hidden
        && spec.intermediate as usize == d.intermediate
        && d.down_m == d.hidden
        && d.down_k == d.intermediate
        && f.routed_gate_up == spec.gate_up_codec
        && f.routed_down == spec.down_codec;
    if !agrees {
        return exclude(X::SpecMismatch);
    }
    if !f.x_norm_f32 {
        return exclude(X::ActivationDtype);
    }
    if let Err(name) = expert_iu4_capacity(d, &f.held) {
        return exclude(X::Capacity(name));
    }
    if let Err(name) = expert_iu4_scratch_ready(d, &f.scratch) {
        return exclude(X::Capacity(name));
    }
    ExpertIu4Decision::Admitted(ExpertIu4Plan {
        rows: d.rows,
        grouped_rows: d.grouped_rows,
        total_slots: d.rows * d.k_top,
        hidden: d.hidden,
        intermediate: d.intermediate,
        k_top: d.k_top,
        host_mapped: f.host_mapped,
    })
}

/// Whether a hipGraph capture is active (`graphs.capture_mode`, verify capture or
/// replay capture). Informational: the route is capture-safe (one cached plan, a
/// preflight that never grows scratch, fixed-pointer launches) and admitted.
fn expert_iu4_graph_capturing(gpu: &Gpu) -> bool {
    gpu.graphs.capture_mode
        || gpu.graphs.verify.capturing.is_some()
        || gpu.graphs.replay.capturing.is_some()
}

/// A Redline recording would capture this call (not a hipGraph capture): the
/// route has no retained-tape proof, so the call is excluded by name.
///
/// Deliberately `is_recording()` alone, not `retained_body_active()`: a prepared
/// PM4/AQL route replaces the whole plain-AR forward before any layer program
/// runs (qwen35 `forward_scratch` returns from its replay branch), so inside a MoE
/// program `should_route_pm4()` / `should_route_aql()` can only be the plain-AR
/// eligibility left over from the previous decode forward. Batched prefill does
/// not reset it, and testing it here sent every prefill after the first Redline
/// decode of a `.mq4r` load onto the slow qt53 F32 fallback.
fn expert_iu4_redline_active(gpu: &Gpu) -> bool {
    gpu.replay.is_recording()
}

/// Layer-wide host-mapped bit of the routed experts (any projection or the layer);
/// the one bit both PM GEMM launchers receive.
fn expert_iu4_host_mapped(p: &MoePrefillParams<'_>) -> bool {
    let (host_gate_up, host_down) =
        crate::families::moe::routed_host_mapped_projections(p.routed_experts);
    host_gate_up || host_down || crate::families::moe::routed_host_mapped(p.routed_experts)
}

/// Gather the verdict's facts. Only called with a bound capability.
fn expert_iu4_facts(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    selection: &MoePrefillSelection,
) -> ExpertIu4Facts {
    let held = |t: &GpuTensor| t.buf.size();
    let usable = |buf: &Option<hip_bridge::DeviceBuffer>, tracked: usize| {
        buf.as_ref().map(|b| b.size().min(tracked))
    };
    ExpertIu4Facts {
        capability: p.expert_iu4,
        requested: gpu.moe_sym_iu4_requested(),
        route_rows: gpu.route_rows(p.batch_size),
        min_rows: EXPERT_IU4_MIN_ROWS,
        use_path2: selection.resolution.use_path2,
        has_dtype_tags: p.expert_dtype_tags.is_some() || p.dtypes.routed_has_mixed_experts,
        has_down_awq: p.expert_down_awq_ptrs.is_some() || p.down_awq_scale.is_some(),
        capturing_graph: expert_iu4_graph_capturing(gpu),
        host_mapped: expert_iu4_host_mapped(p),
        shape: ExpertIu4CallShape {
            specialized_route: p.route_policy.is_some() || selection.route.is_some(),
            cpu_deferred: selection.cpu_deferred_experts || p.cpu_deferred_experts,
            cpu_split: selection.cpu_split,
            routed_out: p.routed_out.is_some(),
            ep_row_gather: p.ep_row_gather.is_some(),
            ep_tile_exchange: p.ep_tile_exchange.is_some(),
            canonical_slot_order: !matches!(
                p.prelude.route,
                super::sealed_moe::PrefillRouteMode::Replicated
            ),
            staged_expert_tables: p.expert_stage_ptrs.is_some(),
            paro: p.paro_gate_up.is_some() || p.paro_down.is_some(),
            redline_recording: expert_iu4_redline_active(gpu),
        },
        routed_gate_up: p.dtypes.routed_gate_up,
        routed_down: p.dtypes.routed_down,
        x_norm_f32: p.x_norm_batch.dtype == rdna_compute::DType::F32,
        dims: ExpertIu4Dims {
            rows: p.batch_size,
            k_top: p.k_top,
            n_exp: p.n_exp,
            hidden: p.gate_up_k,
            intermediate: p.mi,
            down_m: p.down_m,
            down_k: p.down_k,
            grouped_rows: selection.path2_m_total,
        },
        held: ExpertIu4Held {
            x_norm: held(p.x_norm_batch),
            x_batch: held(p.x_batch),
            sorted_slot_index: held(p.sorted_slot_index),
            inverse_perm: held(p.inverse_perm),
            topk_weights: held(p.topk_weights),
            y_gate_up_grouped: held(p.y_gate_up_grouped),
            y_down_grouped: held(p.y_down_grouped),
        },
        scratch: ExpertIu4Scratch {
            gate_x: usable(
                &gpu.scratch.int4_mmq_x_scratch,
                gpu.scratch.int4_mmq_x_scratch_bytes,
            ),
            down_i4: usable(
                &gpu.scratch.qwen4_moe_down_i4_scratch,
                gpu.scratch.qwen4_moe_down_i4_scratch_bytes,
            ),
        },
    }
}

/// Log classes (one rate-limit counter each): admitted 0, each
/// [`ExpertIu4Reject`] `1 + index`, each exclusion 19..=32, admitted host-mapped 33, admitted
/// resident with the NT8 gate/up entry and the NT4 down entry (`HIPFIRE_MOE_IU4_WIDE_TILES`) 34.
const EXPERT_IU4_LOG_CLASSES: usize = 64;
const EXPERT_IU4_CLASS_ADMITTED_HOST_MAPPED: usize = 33;
const EXPERT_IU4_CLASS_ADMITTED_RESIDENT_NT8_GATE_UP: usize = 34;
/// A WARNING repeats on its class's first occurrence and then every this many.
const EXPERT_IU4_WARN_EVERY: u64 = 1000;

/// Log class of a verdict (`None` for the unbound default, which logs nothing).
fn expert_iu4_log_class(
    decision: &ExpertIu4Decision,
    symbols: Option<[&'static str; 2]>,
) -> Option<usize> {
    match decision {
        ExpertIu4Decision::Unbound => None,
        ExpertIu4Decision::Admitted(plan) if plan.host_mapped => {
            Some(EXPERT_IU4_CLASS_ADMITTED_HOST_MAPPED)
        }
        ExpertIu4Decision::Admitted(_)
            if symbols.is_some_and(|[gate_up, _]| gate_up.ends_with("_nt8")) =>
        {
            Some(EXPERT_IU4_CLASS_ADMITTED_RESIDENT_NT8_GATE_UP)
        }
        ExpertIu4Decision::Admitted(_) => Some(0),
        ExpertIu4Decision::Fallback(ExpertIu4Fallback::Admission(reject)) => ExpertIu4Reject::ALL
            .iter()
            .position(|r| r == reject)
            .map(|index| 1 + index),
        ExpertIu4Decision::Fallback(ExpertIu4Fallback::Exclusion(x)) => Some(x.class()),
    }
}

/// Severity of a verdict's log line. Every bound fallback except the intended
/// [`ExpertIu4Reject::NotPath2`] one (the narrow host-verify Path 1 indexed GEMV, or
/// a user-disabled grouped GEMM) means the ~3.7x slower qt53 F32 route ran, so it is
/// a WARNING; admissions and NotPath2 are INFO.
fn expert_iu4_log_level(decision: &ExpertIu4Decision) -> Option<ExpertIu4LogLevel> {
    match decision {
        ExpertIu4Decision::Unbound => None,
        ExpertIu4Decision::Admitted(_) => Some(ExpertIu4LogLevel::Info),
        ExpertIu4Decision::Fallback(ExpertIu4Fallback::Admission(ExpertIu4Reject::NotPath2)) => {
            Some(ExpertIu4LogLevel::Info)
        }
        ExpertIu4Decision::Fallback(_) => Some(ExpertIu4LogLevel::Warning),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExpertIu4LogLevel {
    Info,
    Warning,
}

/// Rate limit: `occurrence` is the number of earlier verdicts of the same class.
/// INFO logs once; WARNING logs the first and every [`EXPERT_IU4_WARN_EVERY`]th.
fn expert_iu4_should_log(level: ExpertIu4LogLevel, occurrence: u64) -> bool {
    match level {
        ExpertIu4LogLevel::Info => occurrence == 0,
        ExpertIu4LogLevel::Warning => occurrence.is_multiple_of(EXPERT_IU4_WARN_EVERY),
    }
}

/// One stderr line describing the verdict (same `  <what> (<flag>): ...` shape as
/// the loader's IU4 line, `WARNING:`-prefixed for [`ExpertIu4LogLevel::Warning`]);
/// `occurrence` is the 1-based count of this class's verdicts so far. `None` for the
/// unbound default.
pub(super) fn expert_iu4_log_line(
    arch: &str,
    f: &ExpertIu4Facts,
    decision: &ExpertIu4Decision,
    occurrence: u64,
    symbols: Option<[&'static str; 2]>,
) -> Option<String> {
    let (spec, _) = f.capability?;
    let common = format!(
        "arch={arch} rows={} route_rows={} gate_up={:?} down={:?} numerics={:?}",
        f.dims.rows, f.route_rows, spec.gate_up_codec, spec.down_codec, spec.numerics
    );
    let head = "  moe expert IU4 prefill (HIPFIRE_MOE_SYM_IU4):";
    match decision {
        ExpertIu4Decision::Unbound => None,
        ExpertIu4Decision::Admitted(plan) => {
            let [gate_up, down] = symbols.unwrap_or(["unreported", "unreported"]);
            // The raw inventory is the v2 bundle's own arch table (a test anchors them), so
            // this names the combine `prefill_expert_iu4_combine_stage` selects on `arch`.
            let combine = if expert_iu4_combine_v2_admits(
                *MOE_COMBINE_V2,
                arch,
                rdna_compute::prefill_route_capabilities::raw_inventory(
                    arch,
                    rdna_compute::prefill_route_capabilities::PrefillRoute::MoeCombineV2,
                ),
                plan,
            ) {
                "moe_slot_ops_v2_pm_gfx1201[moe_down_combine_grouped_bf16in_v2_pm_gfx1201]"
            } else {
                "moe_slot_ops_pm_gfx1201[moe_down_combine_grouped_bf16in_pm_gfx1201]"
            };
            Some(format!(
                "{head} requested={} admitted host_mapped={} graph_capture={} {common} \
                 modules=qwen4_moe_iu4_sym_pm_gfx1201\
                 [{gate_up},{down}]\
                 +{combine}",
                f.requested, plan.host_mapped, f.capturing_graph
            ))
        }
        ExpertIu4Decision::Fallback(fallback) => {
            let detail = match fallback {
                ExpertIu4Fallback::Exclusion(ExpertIu4Exclusion::Capacity(name)) => {
                    format!("{} ({name})", fallback.reason())
                }
                _ => fallback.reason().to_string(),
            };
            Some(match expert_iu4_log_level(decision) {
                Some(ExpertIu4LogLevel::Warning) => format!(
                    "WARNING:{head} requested={} fallback to the qt53 F32 route \
                     (~3.7x slower than the IU4 route at prefill sizes): {detail} \
                     occurrence={occurrence} {common}",
                    f.requested
                ),
                _ => format!(
                    "{head} requested={} fallback to the qt53 F32 route: {detail} {common}",
                    f.requested
                ),
            })
        }
    }
}

/// Log the verdict: INFO once per process per class, WARNING on each class's first
/// occurrence and then every [`EXPERT_IU4_WARN_EVERY`]th.
fn log_expert_iu4(gpu: &Gpu, f: &ExpertIu4Facts, decision: &ExpertIu4Decision) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEEN: [AtomicU64; EXPERT_IU4_LOG_CLASSES] =
        [const { AtomicU64::new(0) }; EXPERT_IU4_LOG_CLASSES];
    let symbols = expert_iu4_launched_symbols(gpu, f, decision);
    let (Some(class), Some(level)) = (
        expert_iu4_log_class(decision, symbols),
        expert_iu4_log_level(decision),
    ) else {
        return;
    };
    let occurrence = SEEN[class].fetch_add(1, Ordering::Relaxed);
    if expert_iu4_should_log(level, occurrence) {
        if let Some(line) =
            expert_iu4_log_line(gpu.arch.as_str(), f, decision, occurrence + 1, symbols)
        {
            eprintln!("{line}");
        }
    }
}

/// The `[gate/up, down]` PM entries the two launchers run for an admitted verdict: the same pure
/// selector they call (`Gpu::qwen4_moe_sym_gemm_symbols`) over the plan's real slots
/// (`rows * k_top`, not padded grouped rows) and the expert-table length `n_exp`, so the log names
/// what launches. `None` for every other verdict.
fn expert_iu4_launched_symbols(
    gpu: &Gpu,
    f: &ExpertIu4Facts,
    decision: &ExpertIu4Decision,
) -> Option<[&'static str; 2]> {
    match decision {
        ExpertIu4Decision::Admitted(plan) => {
            gpu.qwen4_moe_sym_gemm_symbols_wide(plan.host_mapped, plan.total_slots, f.dims.n_exp)
        }
        _ => None,
    }
}

impl MoeStepState<'_> {
    /// The call's one IU4 verdict, decided on first use and cached. Unbound (the
    /// default: the arch bound no capability) returns before any flag, device or
    /// operand read.
    fn expert_iu4_decision(
        &self,
        gpu: &Gpu,
        params: &MoePrefillParams<'_>,
        selection: &MoePrefillSelection,
    ) -> ExpertIu4Decision {
        if params.expert_iu4.is_none() {
            return ExpertIu4Decision::Unbound;
        }
        *self.expert_iu4.get_or_init(|| {
            let facts = expert_iu4_facts(gpu, params, selection);
            let decision = decide_expert_iu4(&facts);
            log_expert_iu4(gpu, &facts, &decision);
            decision
        })
    }
}

/// `HIPFIRE_MOE_COMBINE_V2` (`developer.moe_combine_v2`, a typed boolean): the
/// explicit value, read once. `true`/`1` opts the IU4 grouped-expert combine
/// into the v2 BF16-input twin wherever its bundle and shape admit it;
/// `false`/`0` kills it. Unset follows `prefill_route_capabilities`
/// qualification of `PrefillRoute::MoeCombineV2`, which has an exact-gfx1201
/// `QUALIFIED` row: on at gfx1201, off elsewhere.
static MOE_COMBINE_V2: std::sync::LazyLock<Option<bool>> =
    std::sync::LazyLock::new(|| hipfire_config::developer_typed_bool("HIPFIRE_MOE_COMBINE_V2"));

/// Pure selection of the v2 combine twin over the v1 twin for an admitted IU4
/// plan: the route flag resolves on `arch` (explicit `1`/`0` wins, unset follows
/// qualification: on at exact gfx1201), the v2 bundle exists on the device (`available`), and
/// the plan's launch shape is one the v2 kernel accepts (`dim` a positive
/// multiple of 2048, `k_top` in `1..=8`, extent limits). Anything else keeps v1.
pub(super) fn expert_iu4_combine_v2_admits(
    flag: Option<bool>,
    arch: &str,
    available: bool,
    plan: &ExpertIu4Plan,
) -> bool {
    use rdna_compute::prefill_route_capabilities::{enabled, PrefillRoute};
    enabled(flag, arch, PrefillRoute::MoeCombineV2)
        && available
        && rdna_compute::moe_slot_ops::moe_combine_v2_shape_supported(
            plan.hidden,
            plan.k_top,
            plan.rows,
            plan.grouped_rows,
        )
}

/// Whether the IU4 combine of `plan` runs the v2 twin on `gpu`.
pub(super) fn expert_iu4_combine_v2_selected(gpu: &Gpu, plan: &ExpertIu4Plan) -> bool {
    expert_iu4_combine_v2_admits(
        *MOE_COMBINE_V2,
        &gpu.arch,
        gpu.moe_combine_v2_available(),
        plan,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_topk_disables_router_shared_fuse() {
        assert!(!router_shared_fuse_allowed(
            false,
            false,
            true,
            1,
            false,
            512,
            rdna_compute::DType::MQ4G256,
            false,
            true,
        ));
    }

    #[test]
    fn router_shared_iu4_carry_admits_only_compatible_router() {
        use rdna_compute::DType;
        let gated = MoeRecipe::SoftmaxGatedShared {
            bf16_round_trip: true,
            shared_after_combine: true,
        };
        assert!(router_carries_shared_iu4(
            gated,
            DType::MQ4G256V2,
            2048,
            Some(2048),
            false
        ));
        assert!(!router_carries_shared_iu4(
            gated,
            DType::MQ4G256V2,
            2048,
            Some(2048),
            true
        ));
        assert!(!router_carries_shared_iu4(
            gated,
            DType::MQ4G256V2,
            2048,
            Some(1024),
            false
        ));
        assert!(!router_carries_shared_iu4(
            gated,
            DType::MQ4G256V2,
            2048,
            None,
            false
        ));
        for dtype in [DType::MQ4G256, DType::Q8_0, DType::F32, DType::MQ6G256V2] {
            assert!(!router_carries_shared_iu4(
                gated,
                dtype,
                2048,
                Some(2048),
                false
            ));
        }
        assert!(!router_carries_shared_iu4(
            MoeRecipe::SigmoidRoutedNoShared,
            DType::MQ4G256V2,
            2048,
            Some(2048),
            false
        ));
    }

    #[test]
    fn input_basis_producer_fused_admits_only_a_carried_mq4v2_gate() {
        use rdna_compute::DType;
        let mq = DType::MQ4G256V2;
        assert!(input_basis_producer_fused_admitted(
            true, true, mq, 2048, 2048
        ));
        assert!(input_basis_producer_fused_admitted(
            true, true, mq, 512, 512
        ));
        // Flag off, or the router/shared carry does not hold (adopted route,
        // non-MQ4V2 router, selector K mismatch, flag, capability, rows).
        assert!(!input_basis_producer_fused_admitted(
            false, true, mq, 2048, 2048
        ));
        assert!(!input_basis_producer_fused_admitted(
            true, false, mq, 2048, 2048
        ));
        // The gate must be MQ4G256V2 over the router's K, a nonzero multiple of 256.
        for dtype in [DType::MQ4G256, DType::MQ6G256V2, DType::Q8_0, DType::F32] {
            assert!(!input_basis_producer_fused_admitted(
                true, true, dtype, 2048, 2048
            ));
        }
        for k in [0usize, 1, 255, 257, 2049] {
            assert!(!input_basis_producer_fused_admitted(true, true, mq, k, k));
        }
        assert!(!input_basis_producer_fused_admitted(
            true, true, mq, 2048, 1024
        ));
    }

    fn last_combine(
        ep_none: bool,
        is_gfx1100: bool,
        batch_size: usize,
        topk: usize,
        has_tags: bool,
        down: rdna_compute::DType,
        down_k: usize,
    ) -> bool {
        down_last_combine_allowed(
            ep_none,
            is_gfx1100,
            batch_size,
            topk,
            has_tags,
            down,
            down_k,
            || true,
        )
    }

    #[test]
    fn down_last_combine_admits_only_k512() {
        let mq4 = rdna_compute::DType::MQ4G256;
        assert!(last_combine(true, true, 1, 8, false, mq4, 512));
        for k in [1792, 1536, 256, 0, 1024] {
            assert!(!last_combine(true, true, 1, 8, false, mq4, k), "K={k}");
        }
    }

    #[test]
    fn down_last_combine_keeps_other_predicate_restrictions() {
        let mq4 = rdna_compute::DType::MQ4G256;
        assert!(!last_combine(false, true, 1, 8, false, mq4, 512));
        assert!(!last_combine(true, false, 1, 8, false, mq4, 512));
        assert!(!last_combine(true, true, 2, 8, false, mq4, 512));
        assert!(!last_combine(true, true, 1, 4, false, mq4, 512));
        assert!(!last_combine(true, true, 1, 8, true, mq4, 512));
        assert!(!last_combine(
            true,
            true,
            1,
            8,
            false,
            rdna_compute::DType::F32,
            512
        ));
    }

    #[test]
    fn down_last_combine_env_is_last_and_short_circuited() {
        let mq4 = rdna_compute::DType::MQ4G256;
        let calls = std::cell::Cell::new(0);
        let probe = |ep_none: bool, down_k: usize, env: bool| {
            down_last_combine_allowed(ep_none, true, 1, 8, false, mq4, down_k, || {
                calls.set(calls.get() + 1);
                env
            })
        };
        assert!(!probe(true, 1792, true));
        assert!(!probe(false, 512, true));
        assert_eq!(calls.get(), 0);
        assert!(!probe(true, 512, false));
        assert_eq!(calls.get(), 1);
        assert!(probe(true, 512, true));
        assert_eq!(calls.get(), 2);
    }

    // ── opt-in expert IU4 prefill route: the pure verdict ────────────────────

    use crate::families::moe_expert_iu4::{ExpertNumerics, EXPERT_IU4_MIN_ROWS};
    use rdna_compute::DType;

    const A3B_ROWS: usize = 8192;
    const A3B_TOP_K: usize = 8;
    const A3B_GROUPED: usize = A3B_ROWS * A3B_TOP_K + 256 * 15;

    fn a3b_spec() -> ExpertIu4Spec {
        ExpertIu4Spec {
            experts: 256,
            top_k: 8,
            hidden: 2048,
            intermediate: 512,
            gate_up_codec: DType::MQ4G256V2,
            down_codec: DType::MQ4G128V2,
            numerics: ExpertNumerics::SymmetricBf16SlotsF32OrderedCombine,
        }
    }

    fn a3b_dims() -> ExpertIu4Dims {
        ExpertIu4Dims {
            rows: A3B_ROWS,
            k_top: A3B_TOP_K,
            n_exp: 256,
            hidden: 2048,
            intermediate: 512,
            down_m: 2048,
            down_k: 512,
            grouped_rows: A3B_GROUPED,
        }
    }

    fn a3b_held() -> ExpertIu4Held {
        ExpertIu4Held {
            x_norm: A3B_ROWS * 2048 * 4,
            x_batch: A3B_ROWS * 2048 * 4,
            sorted_slot_index: A3B_GROUPED * 4,
            inverse_perm: A3B_ROWS * A3B_TOP_K * 4,
            topk_weights: A3B_ROWS * A3B_TOP_K * 4,
            y_gate_up_grouped: A3B_GROUPED * 512 * 2,
            y_down_grouped: A3B_GROUPED * 2048 * 2,
        }
    }

    fn a3b_scratch() -> ExpertIu4Scratch {
        ExpertIu4Scratch {
            gate_x: rdna_compute::scratch::int4_mmq_reservation_bytes(2048, A3B_ROWS),
            down_i4: rdna_compute::scratch::qwen4_moe_down_i4_bytes(512, A3B_ROWS * A3B_TOP_K),
        }
    }

    /// An admitted A3B prefill chunk on a device that requested the route.
    fn admitted_facts() -> ExpertIu4Facts {
        ExpertIu4Facts {
            capability: Some((a3b_spec(), true)),
            requested: true,
            route_rows: A3B_ROWS,
            min_rows: EXPERT_IU4_MIN_ROWS,
            use_path2: true,
            has_dtype_tags: false,
            has_down_awq: false,
            capturing_graph: false,
            host_mapped: false,
            shape: ExpertIu4CallShape::default(),
            routed_gate_up: DType::MQ4G256V2,
            routed_down: DType::MQ4G128V2,
            x_norm_f32: true,
            dims: a3b_dims(),
            held: a3b_held(),
            scratch: a3b_scratch(),
        }
    }

    fn fallback(facts: ExpertIu4Facts) -> ExpertIu4Fallback {
        match decide_expert_iu4(&facts) {
            ExpertIu4Decision::Fallback(fallback) => fallback,
            other => panic!("expected a fallback, got {other:?}"),
        }
    }

    #[test]
    fn expert_iu4_admits_the_a3b_prefill_chunk_with_exact_plan() {
        assert_eq!(
            decide_expert_iu4(&admitted_facts()),
            ExpertIu4Decision::Admitted(ExpertIu4Plan {
                rows: A3B_ROWS,
                grouped_rows: A3B_GROUPED,
                total_slots: A3B_ROWS * A3B_TOP_K,
                hidden: 2048,
                intermediate: 512,
                k_top: 8,
                host_mapped: false,
            })
        );
        // The row floor is inclusive, on the route-row measure, and is a single row.
        let at_floor = ExpertIu4Facts {
            route_rows: EXPERT_IU4_MIN_ROWS,
            ..admitted_facts()
        };
        assert!(matches!(
            decide_expert_iu4(&at_floor),
            ExpertIu4Decision::Admitted(_)
        ));
    }

    /// Facts of an A3B-geometry call carrying `rows` physical rows (the grouped
    /// extent is the worst-case tile-padded one the scatter publishes).
    fn a3b_rows_facts(rows: usize) -> ExpertIu4Facts {
        let grouped = (rows * A3B_TOP_K + 256 * 15).next_multiple_of(EXPERT_IU4_TILE_ROWS);
        ExpertIu4Facts {
            route_rows: rows,
            dims: ExpertIu4Dims {
                rows,
                grouped_rows: grouped,
                ..a3b_dims()
            },
            held: ExpertIu4Held {
                x_norm: rows * 2048 * 4,
                x_batch: rows * 2048 * 4,
                sorted_slot_index: grouped * 4,
                inverse_perm: rows * A3B_TOP_K * 4,
                topk_weights: rows * A3B_TOP_K * 4,
                y_gate_up_grouped: grouped * 512 * 2,
                y_down_grouped: grouped * 2048 * 2,
            },
            scratch: ExpertIu4Scratch {
                gate_x: rdna_compute::scratch::int4_mmq_reservation_bytes(2048, rows),
                down_i4: rdna_compute::scratch::qwen4_moe_down_i4_bytes(512, rows * A3B_TOP_K),
            },
            ..admitted_facts()
        }
    }

    #[test]
    fn expert_iu4_floor_is_one_row_for_every_prefill_size() {
        assert_eq!(EXPERT_IU4_MIN_ROWS, 1);
        for rows in [1usize, 2, 4, 7, 16, 17, 63, 100, 511, 512, 4096, A3B_ROWS] {
            let facts = a3b_rows_facts(rows);
            match decide_expert_iu4(&facts) {
                ExpertIu4Decision::Admitted(plan) => {
                    assert_eq!(plan.rows, rows);
                    assert_eq!(plan.total_slots, rows * A3B_TOP_K);
                    assert_eq!(plan.grouped_rows, facts.dims.grouped_rows);
                    assert!(!plan.host_mapped);
                }
                other => panic!("{rows} rows: {other:?}"),
            }
        }
        // Zero routing rows is still below the floor.
        assert_eq!(
            fallback(ExpertIu4Facts {
                route_rows: 0,
                ..admitted_facts()
            }),
            ExpertIu4Fallback::Admission(ExpertIu4Reject::BelowMinRows)
        );
    }

    #[test]
    fn expert_iu4_admits_graph_capture_and_host_mapped_experts() {
        let device = match decide_expert_iu4(&admitted_facts()) {
            ExpertIu4Decision::Admitted(plan) => plan,
            other => panic!("{other:?}"),
        };
        assert!(!device.host_mapped);
        // hipGraph capture (capture_mode, verify capture, replay capture) is admitted:
        // the same cached plan, no extra requirement.
        let capturing = ExpertIu4Facts {
            capturing_graph: true,
            ..admitted_facts()
        };
        assert_eq!(
            decide_expert_iu4(&capturing),
            ExpertIu4Decision::Admitted(device)
        );
        // Host-mapped experts are admitted and the plan carries the bit to the PM
        // launchers (NT8 entries) with the same geometry.
        let host = ExpertIu4Facts {
            host_mapped: true,
            ..admitted_facts()
        };
        assert_eq!(
            decide_expert_iu4(&host),
            ExpertIu4Decision::Admitted(ExpertIu4Plan {
                host_mapped: true,
                ..device
            })
        );
        let both = ExpertIu4Facts {
            capturing_graph: true,
            host_mapped: true,
            ..admitted_facts()
        };
        assert_eq!(
            decide_expert_iu4(&both),
            ExpertIu4Decision::Admitted(ExpertIu4Plan {
                host_mapped: true,
                ..device
            })
        );
        // The capacity / scratch preflight is the same bytes under capture and host:
        // a short reservation still refuses (never grows mid-capture).
        let short = ExpertIu4Facts {
            scratch: ExpertIu4Scratch {
                gate_x: None,
                ..a3b_scratch()
            },
            ..both
        };
        assert_eq!(
            fallback(short),
            ExpertIu4Fallback::Exclusion(ExpertIu4Exclusion::Capacity("int4_mmq_x_scratch"))
        );
    }

    #[test]
    fn expert_iu4_unbound_is_byte_neutral_whatever_else_is_true() {
        // No capability (the default and every non-Qwen35 arch): nothing else is read.
        let hostile = ExpertIu4Facts {
            capability: None,
            requested: false,
            use_path2: false,
            capturing_graph: true,
            host_mapped: true,
            has_dtype_tags: true,
            route_rows: 1,
            shape: ExpertIu4CallShape {
                routed_out: true,
                ep_tile_exchange: true,
                ..ExpertIu4CallShape::default()
            },
            ..admitted_facts()
        };
        assert_eq!(decide_expert_iu4(&hostile), ExpertIu4Decision::Unbound);
        assert_eq!(
            expert_iu4_log_class(&ExpertIu4Decision::Unbound, None),
            None
        );
        assert_eq!(
            expert_iu4_log_line("gfx1201", &hostile, &ExpertIu4Decision::Unbound, 1, None),
            None
        );
        assert_eq!(expert_iu4_log_level(&ExpertIu4Decision::Unbound), None);
    }

    #[test]
    fn expert_iu4_fallback_table() {
        use ExpertIu4Exclusion as X;
        use ExpertIu4Fallback::{Admission, Exclusion};
        let base = admitted_facts();
        let shape = |f: fn(&mut ExpertIu4CallShape)| {
            let mut s = ExpertIu4CallShape::default();
            f(&mut s);
            s
        };
        let cases: Vec<(&str, ExpertIu4Facts, ExpertIu4Fallback)> = vec![
            (
                "flag off / not requested",
                ExpertIu4Facts {
                    requested: false,
                    ..base
                },
                Admission(ExpertIu4Reject::NotRequested),
            ),
            (
                "headers not verified on every layer",
                ExpertIu4Facts {
                    capability: Some((a3b_spec(), false)),
                    ..base
                },
                Admission(ExpertIu4Reject::HeadersNotVerified),
            ),
            (
                "path 1",
                ExpertIu4Facts {
                    use_path2: false,
                    ..base
                },
                Admission(ExpertIu4Reject::NotPath2),
            ),
            (
                "below the row floor",
                ExpertIu4Facts {
                    route_rows: EXPERT_IU4_MIN_ROWS - 1,
                    ..base
                },
                Admission(ExpertIu4Reject::BelowMinRows),
            ),
            (
                "dtype tags",
                ExpertIu4Facts {
                    has_dtype_tags: true,
                    ..base
                },
                Admission(ExpertIu4Reject::MixedTags),
            ),
            (
                "down AWQ",
                ExpertIu4Facts {
                    has_down_awq: true,
                    ..base
                },
                Admission(ExpertIu4Reject::DownAwq),
            ),
            (
                "Redline recording (the named pipeline exclusion)",
                ExpertIu4Facts {
                    shape: shape(|s| s.redline_recording = true),
                    ..base
                },
                Exclusion(X::RedlineRecording),
            ),
            (
                "wrong gate/up codec in the spec",
                ExpertIu4Facts {
                    capability: Some((
                        ExpertIu4Spec {
                            gate_up_codec: DType::MQ4G256,
                            ..a3b_spec()
                        },
                        true,
                    )),
                    ..base
                },
                Admission(ExpertIu4Reject::GateUpCodec),
            ),
            (
                "specialized route",
                ExpertIu4Facts {
                    shape: shape(|s| s.specialized_route = true),
                    ..base
                },
                Exclusion(X::SpecializedRoute),
            ),
            (
                "CPU-deferred",
                ExpertIu4Facts {
                    shape: shape(|s| s.cpu_deferred = true),
                    ..base
                },
                Exclusion(X::CpuDeferred),
            ),
            (
                "CPU split",
                ExpertIu4Facts {
                    shape: shape(|s| s.cpu_split = true),
                    ..base
                },
                Exclusion(X::CpuSplit),
            ),
            (
                "EP routed_out (Path 2 stays true: never NotPath2)",
                ExpertIu4Facts {
                    shape: shape(|s| s.routed_out = true),
                    ..base
                },
                Exclusion(X::RoutedOut),
            ),
            (
                "EP row gather",
                ExpertIu4Facts {
                    shape: shape(|s| s.ep_row_gather = true),
                    ..base
                },
                Exclusion(X::EpRowGather),
            ),
            (
                "EP tile exchange",
                ExpertIu4Facts {
                    shape: shape(|s| s.ep_tile_exchange = true),
                    ..base
                },
                Exclusion(X::EpTileExchange),
            ),
            (
                "canonical slot-order (root-routed EP)",
                ExpertIu4Facts {
                    shape: shape(|s| s.canonical_slot_order = true),
                    ..base
                },
                Exclusion(X::CanonicalSlotOrder),
            ),
            (
                "staged expert tables",
                ExpertIu4Facts {
                    shape: shape(|s| s.staged_expert_tables = true),
                    ..base
                },
                Exclusion(X::StagedExpertTables),
            ),
            (
                "paro",
                ExpertIu4Facts {
                    shape: shape(|s| s.paro = true),
                    ..base
                },
                Exclusion(X::Paro),
            ),
            (
                "spec geometry disagrees",
                ExpertIu4Facts {
                    dims: ExpertIu4Dims {
                        intermediate: 768,
                        down_k: 768,
                        ..a3b_dims()
                    },
                    ..base
                },
                Exclusion(X::SpecMismatch),
            ),
            (
                "routed down dtype disagrees",
                ExpertIu4Facts {
                    routed_down: DType::MQ4G256V2,
                    ..base
                },
                Exclusion(X::SpecMismatch),
            ),
            (
                "activation is not F32",
                ExpertIu4Facts {
                    x_norm_f32: false,
                    ..base
                },
                Exclusion(X::ActivationDtype),
            ),
            (
                "BF16 gate/up view does not fit",
                ExpertIu4Facts {
                    held: ExpertIu4Held {
                        y_gate_up_grouped: A3B_GROUPED * 512 * 2 - 1,
                        ..a3b_held()
                    },
                    ..base
                },
                Exclusion(X::Capacity("y_gate_up_grouped")),
            ),
            (
                "gate A4 scratch undersized (larger chunk than the load reservation)",
                ExpertIu4Facts {
                    scratch: ExpertIu4Scratch {
                        gate_x: rdna_compute::scratch::int4_mmq_reservation_bytes(
                            2048,
                            A3B_ROWS / 2,
                        ),
                        ..a3b_scratch()
                    },
                    ..base
                },
                Exclusion(X::Capacity("int4_mmq_x_scratch")),
            ),
            (
                "down A4 scratch never reserved",
                ExpertIu4Facts {
                    scratch: ExpertIu4Scratch {
                        down_i4: None,
                        ..a3b_scratch()
                    },
                    ..base
                },
                Exclusion(X::Capacity("qwen4_moe_down_i4_scratch")),
            ),
        ];
        for (label, facts, want) in cases {
            assert_eq!(fallback(facts), want, "{label}");
        }
    }

    #[test]
    fn expert_iu4_admission_precedes_exclusions() {
        // A call that is both below the floor and EP reports the admission reason.
        let facts = ExpertIu4Facts {
            route_rows: EXPERT_IU4_MIN_ROWS - 1,
            shape: ExpertIu4CallShape {
                routed_out: true,
                ..ExpertIu4CallShape::default()
            },
            ..admitted_facts()
        };
        assert_eq!(
            fallback(facts),
            ExpertIu4Fallback::Admission(ExpertIu4Reject::BelowMinRows)
        );
    }

    #[test]
    fn expert_iu4_plan_is_the_physical_call_not_the_route_rows() {
        // Under the batch-invariant profile `route_rows` is the canonical chunk, but the
        // launches carry the physical rows: the floor reads route_rows, the plan rows.
        let physical = 1024usize;
        let grouped = physical * A3B_TOP_K + 256 * 15;
        let facts = ExpertIu4Facts {
            route_rows: A3B_ROWS,
            dims: ExpertIu4Dims {
                rows: physical,
                grouped_rows: grouped,
                ..a3b_dims()
            },
            held: ExpertIu4Held {
                x_norm: physical * 2048 * 4,
                x_batch: physical * 2048 * 4,
                sorted_slot_index: grouped * 4,
                inverse_perm: physical * A3B_TOP_K * 4,
                topk_weights: physical * A3B_TOP_K * 4,
                y_gate_up_grouped: grouped * 512 * 2,
                y_down_grouped: grouped * 2048 * 2,
            },
            scratch: ExpertIu4Scratch {
                gate_x: rdna_compute::scratch::int4_mmq_reservation_bytes(2048, physical),
                down_i4: rdna_compute::scratch::qwen4_moe_down_i4_bytes(512, physical * A3B_TOP_K),
            },
            ..admitted_facts()
        };
        match decide_expert_iu4(&facts) {
            ExpertIu4Decision::Admitted(plan) => {
                assert_eq!(
                    (plan.rows, plan.total_slots, plan.grouped_rows),
                    (physical, physical * 8, grouped)
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn expert_iu4_bf16_view_bytes_are_exact_and_checked() {
        assert_eq!(
            bf16_grouped_view_bytes(A3B_GROUPED, 512),
            Some(A3B_GROUPED * 512 * 2)
        );
        assert_eq!(
            bf16_grouped_view_bytes(A3B_GROUPED, 2048),
            Some(A3B_GROUPED * 2048 * 2)
        );
        assert_eq!(bf16_grouped_view_bytes(0, 512), Some(0));
        assert_eq!(bf16_grouped_view_bytes(usize::MAX, 2), None);
        assert_eq!(bf16_grouped_view_bytes(usize::MAX / 2 + 1, 1), None);
    }

    #[test]
    fn expert_iu4_capacity_table() {
        let d = a3b_dims();
        let h = a3b_held();
        assert_eq!(expert_iu4_capacity(&d, &h), Ok(()));
        let short = |f: fn(&mut ExpertIu4Held), name: &str| {
            let mut held = h;
            f(&mut held);
            assert_eq!(
                expert_iu4_capacity(&d, &held),
                Err(name_static(name)),
                "{name}"
            );
        };
        short(|h| h.x_norm -= 1, "x_norm_batch");
        short(|h| h.x_batch -= 1, "x_batch");
        short(|h| h.sorted_slot_index -= 1, "sorted_slot_index");
        short(|h| h.inverse_perm -= 1, "inverse_perm");
        short(|h| h.topk_weights -= 1, "topk_weights");
        short(|h| h.y_gate_up_grouped -= 1, "y_gate_up_grouped");
        short(|h| h.y_down_grouped -= 1, "y_down_grouped");
        // Domain limits.
        let with = |d: ExpertIu4Dims| expert_iu4_capacity(&d, &h);
        assert_eq!(with(ExpertIu4Dims { rows: 0, ..d }), Err("rows"));
        assert_eq!(
            with(ExpertIu4Dims {
                grouped_rows: d.grouped_rows + 1,
                ..d
            }),
            Err("grouped_rows")
        );
        assert_eq!(
            with(ExpertIu4Dims {
                grouped_rows: A3B_ROWS * 8 - 16,
                ..d
            }),
            Err("grouped_rows")
        );
        assert_eq!(
            with(ExpertIu4Dims {
                rows: (1 << 22) / 8 + 1,
                ..d
            }),
            Err("slots")
        );
        assert_eq!(
            with(ExpertIu4Dims {
                hidden: 256 * 65_536,
                ..d
            }),
            Err("hidden")
        );
        // The PM down kernel's 32-bit Y byte extent.
        let wide = ExpertIu4Dims {
            grouped_rows: 1 << 21,
            ..d
        };
        let held = ExpertIu4Held {
            sorted_slot_index: usize::MAX / 8,
            y_gate_up_grouped: usize::MAX / 8,
            y_down_grouped: usize::MAX / 8,
            ..h
        };
        assert_eq!(expert_iu4_capacity(&wide, &held), Err("y_down_grouped"));
    }

    /// Interns the table's names as the `&'static str` the function returns.
    fn name_static(name: &str) -> &'static str {
        [
            "x_norm_batch",
            "x_batch",
            "sorted_slot_index",
            "inverse_perm",
            "topk_weights",
            "y_gate_up_grouped",
            "y_down_grouped",
        ]
        .into_iter()
        .find(|candidate| *candidate == name)
        .expect("known operand")
    }

    #[test]
    fn expert_iu4_scratch_preflight_table() {
        let d = a3b_dims();
        let ok = a3b_scratch();
        assert_eq!(expert_iu4_scratch_ready(&d, &ok), Ok(()));
        // Exact extents of the producers' allocators: one byte short fails.
        let gate = ok.gate_x.unwrap();
        let down = ok.down_i4.unwrap();
        assert_eq!(gate, 2048 / 128 * A3B_ROWS * 72);
        assert_eq!(down, 512 / 128 * A3B_ROWS * A3B_TOP_K * 72);
        assert_eq!(
            expert_iu4_scratch_ready(
                &d,
                &ExpertIu4Scratch {
                    gate_x: Some(gate - 1),
                    ..ok
                }
            ),
            Err("int4_mmq_x_scratch")
        );
        assert_eq!(
            expert_iu4_scratch_ready(
                &d,
                &ExpertIu4Scratch {
                    down_i4: Some(down - 1),
                    ..ok
                }
            ),
            Err("qwen4_moe_down_i4_scratch")
        );
        assert_eq!(
            expert_iu4_scratch_ready(&d, &ExpertIu4Scratch { gate_x: None, ..ok }),
            Err("int4_mmq_x_scratch")
        );
        assert_eq!(
            expert_iu4_scratch_ready(
                &d,
                &ExpertIu4Scratch {
                    down_i4: None,
                    ..ok
                }
            ),
            Err("qwen4_moe_down_i4_scratch")
        );
        // Larger reservations are fine; a rejected shape is never admitted.
        assert_eq!(
            expert_iu4_scratch_ready(
                &d,
                &ExpertIu4Scratch {
                    gate_x: Some(gate * 2),
                    down_i4: Some(down * 2)
                }
            ),
            Ok(())
        );
        let odd = ExpertIu4Dims {
            hidden: 2048 + 128,
            ..d
        };
        assert_eq!(
            expert_iu4_scratch_ready(&odd, &ok),
            Err("int4_mmq_x_scratch")
        );
    }

    #[test]
    fn expert_iu4_log_classes_are_distinct_and_lines_carry_the_decision() {
        use ExpertIu4Exclusion as X;
        const PM_NT4: [&str; 2] = [
            "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt4",
            "qwen4_moe_down_iu4_sym_pm_gfx1201_nt4",
        ];
        // Resident widened: gate/up NT8, down NT4 (the down is never widened resident).
        const PM_MIXED: [&str; 2] = [
            "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8",
            "qwen4_moe_down_iu4_sym_pm_gfx1201_nt4",
        ];
        const PM_NT8: [&str; 2] = [
            "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8",
            "qwen4_moe_down_iu4_sym_pm_gfx1201_nt8",
        ];
        let mut classes = std::collections::BTreeSet::new();
        let mut push_with = |decision: ExpertIu4Decision, symbols: Option<[&'static str; 2]>| {
            let class = expert_iu4_log_class(&decision, symbols).expect("loggable");
            assert!(class < EXPERT_IU4_LOG_CLASSES);
            assert!(classes.insert(class), "{decision:?} reuses class {class}");
        };
        let mut push = |decision: ExpertIu4Decision| push_with(decision, None);
        let plan = ExpertIu4Plan {
            rows: 1,
            grouped_rows: 16,
            total_slots: 8,
            hidden: 2048,
            intermediate: 512,
            k_top: 8,
            host_mapped: false,
        };
        push(ExpertIu4Decision::Admitted(plan));
        push(ExpertIu4Decision::Admitted(ExpertIu4Plan {
            host_mapped: true,
            ..plan
        }));
        for reject in ExpertIu4Reject::ALL {
            push(ExpertIu4Decision::Fallback(ExpertIu4Fallback::Admission(
                reject,
            )));
        }
        for x in [
            X::SpecializedRoute,
            X::CpuDeferred,
            X::CpuSplit,
            X::RoutedOut,
            X::EpRowGather,
            X::EpTileExchange,
            X::CanonicalSlotOrder,
            X::StagedExpertTables,
            X::Paro,
            X::SpecMismatch,
            X::ActivationDtype,
            X::Capacity("x"),
            X::RedlineRecording,
        ] {
            push(ExpertIu4Decision::Fallback(ExpertIu4Fallback::Exclusion(x)));
        }
        // A resident admission that launches the NT8 gate/up entry is its own class.
        push_with(ExpertIu4Decision::Admitted(plan), Some(PM_MIXED));

        let facts = admitted_facts();
        let admitted = decide_expert_iu4(&facts);
        let line = expert_iu4_log_line("gfx1201", &facts, &admitted, 1, Some(PM_NT4)).unwrap();
        for needle in [
            "requested=true",
            "admitted",
            "host_mapped=false",
            "graph_capture=false",
            "arch=gfx1201",
            "MQ4G256V2",
            "MQ4G128V2",
            "SymmetricBf16SlotsF32OrderedCombine",
            "qwen4_moe_iu4_sym_pm_gfx1201",
            "_nt4",
            // Unset HIPFIRE_MOE_COMBINE_V2 is qualified on at exact gfx1201, so the admitted
            // line names the v2 combine twin.
            "moe_slot_ops_v2_pm_gfx1201",
            "moe_down_combine_grouped_bf16in_v2_pm_gfx1201",
            "rows=8192",
        ] {
            assert!(line.contains(needle), "{needle}: {line}");
        }
        // The v1 combine is named only when the flag is explicitly 0 (not reachable from this
        // process-wide static; `expert_iu4_combine_v2_admits` tests cover the explicit-0 gate).
        assert!(!line.contains("moe_slot_ops_pm_gfx1201"), "{line}");
        assert!(
            !line.contains("moe_down_combine_grouped_bf16in_pm_gfx1201"),
            "{line}"
        );
        assert!(!line.starts_with("WARNING"), "{line}");

        // A host-mapped admission names the NT8 entries and its own log class.
        let host = ExpertIu4Facts {
            host_mapped: true,
            capturing_graph: true,
            ..facts
        };
        let admitted_host = decide_expert_iu4(&host);
        let line = expert_iu4_log_line("gfx1201", &host, &admitted_host, 1, Some(PM_NT8)).unwrap();
        for needle in ["host_mapped=true", "graph_capture=true", "_nt8"] {
            assert!(line.contains(needle), "{needle}: {line}");
        }
        assert!(!line.contains("_nt4"), "{line}");
        assert_ne!(
            expert_iu4_log_class(&admitted, Some(PM_NT4)),
            expert_iu4_log_class(&admitted_host, Some(PM_NT8))
        );

        // A resident admission the launcher widens (flag + shape) names the entries it launches:
        // gate/up NT8 and down NT4. It logs in its own class, distinct from both the NT4 resident
        // and host-mapped ones.
        let line = expert_iu4_log_line("gfx1201", &facts, &admitted, 1, Some(PM_MIXED)).unwrap();
        assert!(line.contains("host_mapped=false"), "{line}");
        assert!(
            line.contains("[qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8,qwen4_moe_down_iu4_sym_pm_gfx1201_nt4]"),
            "{line}"
        );
        assert!(
            !line.contains("qwen4_moe_down_iu4_sym_pm_gfx1201_nt8"),
            "{line}"
        );
        assert!(
            !line.contains("gate_up_silu_iu4_sym_pm_gfx1201_nt4"),
            "{line}"
        );
        let classes_of = [
            expert_iu4_log_class(&admitted, Some(PM_NT4)),
            expert_iu4_log_class(&admitted, Some(PM_MIXED)),
            expert_iu4_log_class(&admitted_host, Some(PM_NT8)),
        ];
        assert_eq!(classes_of, [Some(0), Some(34), Some(33)]);

        let refused = ExpertIu4Facts {
            requested: false,
            ..facts
        };
        let line = expert_iu4_log_line("gfx1100", &refused, &decide_expert_iu4(&refused), 1, None)
            .unwrap();
        assert!(line.contains("requested=false"), "{line}");
        assert!(
            line.contains(ExpertIu4Reject::NotRequested.reason()),
            "{line}"
        );
        assert!(line.contains("arch=gfx1100"), "{line}");

        let tight = ExpertIu4Facts {
            scratch: ExpertIu4Scratch {
                gate_x: None,
                ..a3b_scratch()
            },
            ..facts
        };
        let line =
            expert_iu4_log_line("gfx1201", &tight, &decide_expert_iu4(&tight), 1, None).unwrap();
        assert!(line.contains("int4_mmq_x_scratch"), "{line}");
    }

    #[test]
    fn expert_iu4_log_levels_warn_on_every_slow_fallback_and_info_otherwise() {
        use ExpertIu4Exclusion as X;
        use ExpertIu4Fallback::{Admission, Exclusion};
        use ExpertIu4LogLevel::{Info, Warning};
        let level = |decision| expert_iu4_log_level(&decision);
        assert_eq!(level(ExpertIu4Decision::Unbound), None);
        let plan = ExpertIu4Plan {
            rows: 1,
            grouped_rows: 16,
            total_slots: 8,
            hidden: 2048,
            intermediate: 512,
            k_top: 8,
            host_mapped: false,
        };
        assert_eq!(level(ExpertIu4Decision::Admitted(plan)), Some(Info));
        assert_eq!(
            level(ExpertIu4Decision::Admitted(ExpertIu4Plan {
                host_mapped: true,
                ..plan
            })),
            Some(Info)
        );
        // The narrow host-verify Path 1 (or a user-disabled grouped GEMM) is the one
        // intended fallback: info, once.
        assert_eq!(
            level(ExpertIu4Decision::Fallback(Admission(
                ExpertIu4Reject::NotPath2
            ))),
            Some(Info)
        );
        for reject in ExpertIu4Reject::ALL {
            if reject != ExpertIu4Reject::NotPath2 {
                assert_eq!(
                    level(ExpertIu4Decision::Fallback(Admission(reject))),
                    Some(Warning),
                    "{reject:?}"
                );
            }
        }
        for x in [
            X::SpecializedRoute,
            X::CpuDeferred,
            X::CpuSplit,
            X::RoutedOut,
            X::EpRowGather,
            X::EpTileExchange,
            X::CanonicalSlotOrder,
            X::StagedExpertTables,
            X::Paro,
            X::SpecMismatch,
            X::ActivationDtype,
            X::Capacity("y_down_grouped"),
            X::RedlineRecording,
        ] {
            assert_eq!(
                level(ExpertIu4Decision::Fallback(Exclusion(x))),
                Some(Warning),
                "{x:?}"
            );
        }
    }

    #[test]
    fn expert_iu4_log_rate_limit_is_first_then_every_thousand() {
        use ExpertIu4LogLevel::{Info, Warning};
        // Info: once per class.
        assert!(expert_iu4_should_log(Info, 0));
        for occurrence in [1, 2, 999, 1000, 1001, 2000] {
            assert!(!expert_iu4_should_log(Info, occurrence), "{occurrence}");
        }
        // Warning: the first occurrence and then every 1000th.
        assert_eq!(EXPERT_IU4_WARN_EVERY, 1000);
        let logged: Vec<u64> = (0..=3001)
            .filter(|n| expert_iu4_should_log(Warning, *n))
            .collect();
        assert_eq!(logged, vec![0, 1000, 2000, 3000]);
    }

    #[test]
    fn expert_iu4_warning_lines_name_the_reason_the_slowdown_and_the_count() {
        let recording = ExpertIu4Facts {
            shape: ExpertIu4CallShape {
                redline_recording: true,
                ..ExpertIu4CallShape::default()
            },
            ..admitted_facts()
        };
        let decision = decide_expert_iu4(&recording);
        let line = expert_iu4_log_line("gfx1201", &recording, &decision, 1001, None).unwrap();
        for needle in [
            "WARNING",
            ExpertIu4Exclusion::RedlineRecording.reason(),
            "qt53 F32",
            "~3.7x slower",
            "occurrence=1001",
            "arch=gfx1201",
        ] {
            assert!(line.contains(needle), "{needle}: {line}");
        }

        for (label, facts, name) in [
            (
                "capacity",
                ExpertIu4Facts {
                    scratch: ExpertIu4Scratch {
                        down_i4: None,
                        ..a3b_scratch()
                    },
                    ..admitted_facts()
                },
                "qwen4_moe_down_i4_scratch",
            ),
            (
                "spec mismatch",
                ExpertIu4Facts {
                    routed_down: DType::MQ4G256V2,
                    ..admitted_facts()
                },
                ExpertIu4Exclusion::SpecMismatch.reason(),
            ),
            (
                "activation dtype",
                ExpertIu4Facts {
                    x_norm_f32: false,
                    ..admitted_facts()
                },
                ExpertIu4Exclusion::ActivationDtype.reason(),
            ),
        ] {
            let decision = decide_expert_iu4(&facts);
            let line = expert_iu4_log_line("gfx1201", &facts, &decision, 1, None).unwrap();
            assert!(line.starts_with("WARNING"), "{label}: {line}");
            assert!(line.contains(name), "{label}: {line}");
        }

        // NotPath2 is the info-level fallback: no WARNING prefix.
        let path1 = ExpertIu4Facts {
            use_path2: false,
            ..admitted_facts()
        };
        let line =
            expert_iu4_log_line("gfx1201", &path1, &decide_expert_iu4(&path1), 1, None).unwrap();
        assert!(!line.starts_with("WARNING"), "{line}");
        assert!(line.contains(ExpertIu4Reject::NotPath2.reason()), "{line}");
    }

    #[test]
    fn expert_iu4_combine_v2_selection_truth_table() {
        use rdna_compute::prefill_route_capabilities::{enabled, PrefillRoute};
        let a3b = ExpertIu4Plan {
            rows: 8192,
            grouped_rows: 8192 * 8 + 256 * 15,
            total_slots: 8192 * 8,
            hidden: 2048,
            intermediate: 512,
            k_top: 8,
            host_mapped: false,
        };
        let shapes = [
            (a3b, true),
            (
                ExpertIu4Plan {
                    hidden: 4096,
                    ..a3b
                },
                true,
            ),
            (ExpertIu4Plan { k_top: 1, ..a3b }, true),
            (
                ExpertIu4Plan {
                    hidden: 2560,
                    ..a3b
                },
                false,
            ),
            (
                ExpertIu4Plan {
                    hidden: 1024,
                    ..a3b
                },
                false,
            ),
            (ExpertIu4Plan { hidden: 0, ..a3b }, false),
            (ExpertIu4Plan { k_top: 0, ..a3b }, false),
            (ExpertIu4Plan { k_top: 9, ..a3b }, false),
            (ExpertIu4Plan { rows: 0, ..a3b }, false),
        ];
        for arch in ["gfx1201", "gfx1151", "gfx1100", "gfx942", "unknown"] {
            for flag in [None, Some(false), Some(true)] {
                for available in [false, true] {
                    for (plan, shape_ok) in shapes {
                        let want = flag != Some(false)
                            && (flag == Some(true) || arch == "gfx1201")
                            && available
                            && shape_ok;
                        assert_eq!(
                            expert_iu4_combine_v2_admits(flag, arch, available, &plan),
                            want,
                            "flag {flag:?} arch {arch} available {available} plan {plan:?}"
                        );
                    }
                }
            }
            // Unset follows qualification: on at exact gfx1201 only.
            assert_eq!(
                enabled(None, arch, PrefillRoute::MoeCombineV2),
                arch == "gfx1201"
            );
        }
    }

    #[test]
    fn expert_iu4_combine_v2_requires_the_flag_then_the_bundle_then_the_shape() {
        let plan = ExpertIu4Plan {
            rows: 4096,
            grouped_rows: 4096 * 8,
            total_slots: 4096 * 8,
            hidden: 2048,
            intermediate: 512,
            k_top: 8,
            host_mapped: true,
        };
        // gfx1201 with the bundle and a supported shape: unset is on by qualification, an
        // explicit 0 kills it, an explicit 1 opts in.
        assert!(expert_iu4_combine_v2_admits(None, "gfx1201", true, &plan));
        assert!(!expert_iu4_combine_v2_admits(None, "gfx1201", false, &plan));
        assert!(!expert_iu4_combine_v2_admits(None, "gfx1151", true, &plan));
        assert!(!expert_iu4_combine_v2_admits(
            Some(false),
            "gfx1201",
            true,
            &plan
        ));
        assert!(expert_iu4_combine_v2_admits(
            Some(true),
            "gfx1201",
            true,
            &plan
        ));
        // The bundle missing (any other arch) keeps v1 even when forced on.
        assert!(!expert_iu4_combine_v2_admits(
            Some(true),
            "gfx1151",
            false,
            &plan
        ));
        // The pure shape predicate is the launcher's own.
        assert!(rdna_compute::moe_slot_ops::moe_combine_v2_shape_supported(
            plan.hidden,
            plan.k_top,
            plan.rows,
            plan.grouped_rows
        ));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Generic GroupedMoeOp
//
// A sealed, borrowed description of one routed-expert FFN layer (router kind,
// activation, shared-expert policy, cast policy, combine order, logical versus
// physical widths) executed over the tuned qt44 cores: the indexed GEMVs for
// decode rows and the grouped WMMA core plus stable scatter for prefill rows.
// The program carries no model identity: admission is geometry/format/kernel
// limits only (`GroupedMoeGeometry::validate`, named refusals), and the
// existing recipe-driven sealed calls above keep their arithmetic untouched.
// ════════════════════════════════════════════════════════════════════════════

use crate::families::gemm::{GemmFamily, GemmParams};
use crate::families::gemv::WeightRef;
use crate::families::moe::{grouped_moe_m_total_bound, GroupedMoeGeometry, MOE_GROUPED_BLOCK_M};
pub use rdna_compute::ffn_activation::{CombineOrder, FfnActivation, RouterKind};
use rdna_compute::ffn_activation::{
    FfnActivationGrouped, FfnActivationRows, FfnAwq, MoeCombine, MoeRouterTopK,
};
use rdna_compute::DType;

/// Rounding boundaries of a grouped MoE / gated FFN program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastPolicy {
    /// Every boundary stays FP32 (the separately labelled FP32 oracle mode).
    Exact32,
    /// Model-dtype (BF16, round-to-nearest-even) boundaries: projection
    /// outputs before the activation, the activation result, each expert's
    /// down output, the routed sum (cast once) and the final sum.
    Bf16Boundaries,
}

impl CastPolicy {
    #[inline]
    fn bf16(self) -> bool {
        self == Self::Bf16Boundaries
    }
}

/// How the shared expert joins the routed sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharedExpertPolicy {
    /// No shared expert.
    Absent,
    /// Ungated shared expert (no scalar selector) added AFTER the routed sum
    /// has been cast; the shared output is cast on its own first.
    UngatedAfterRoutedCast,
}

/// Every load-time policy of a grouped MoE layer. Shared kernels receive these
/// as launch arguments; nothing in a kernel inspects a model name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupedMoePolicy {
    pub router: RouterKind,
    pub activation: FfnActivation,
    pub shared: SharedExpertPolicy,
    pub cast: CastPolicy,
    pub combine: CombineOrder,
}

/// Which routed-expert core runs the layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupedMoeProtocol {
    /// `rows <= GROUPED_MOE_INDEXED_MAX_ROWS` takes the indexed GEMVs, wider
    /// batches the grouped WMMA core.
    Auto,
    /// Per-(row, rank) indexed qt44 gate/up and down GEMVs (decode core).
    Indexed,
    /// Stable scatter + grouped WMMA gate/up and down (prefill core).
    Grouped,
}

/// Row count up to which [`GroupedMoeProtocol::Auto`] runs the indexed core.
pub const GROUPED_MOE_INDEXED_MAX_ROWS: usize = 1;

impl GroupedMoeProtocol {
    /// The concrete core for a call of `rows` rows.
    pub fn resolve(self, rows: usize) -> Self {
        match self {
            Self::Auto if rows <= GROUPED_MOE_INDEXED_MAX_ROWS => Self::Indexed,
            Self::Auto => Self::Grouped,
            other => other,
        }
    }
}

/// Element counts (F32 or i32 words) of every scratch tensor one call needs.
/// Zero means the tensor is not used by that core. Allocate with
/// [`GroupedMoeScratchPlan::for_rows`] (union of both cores) to run any
/// protocol at up to `rows` rows without reallocating.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupedMoeScratchPlan {
    pub router_logits: usize,
    pub topk_indices: usize,
    pub topk_weights: usize,
    pub x_rot: usize,
    pub act: usize,
    pub rot: usize,
    pub down_rows: usize,
    pub indexed_gate: usize,
    pub indexed_up: usize,
    pub gate_up_rows: usize,
    pub m_total: usize,
    pub expert_token_counts: usize,
    pub expert_offsets: usize,
    pub sorted_slot_index: usize,
    pub expert_tile_ids: usize,
    pub inverse_perm: usize,
    pub shared_gate: usize,
    pub shared_up: usize,
    pub shared_act: usize,
    pub shared_rot: usize,
    pub shared_out: usize,
    /// Single-expert grouping tables of the shared expert projections.
    pub shared_tables: usize,
}

fn plan_overflow() -> DispatchError {
    DispatchError::Hip("grouped moe: scratch extent overflows usize".into())
}

fn mul(a: usize, b: usize) -> Result<usize, DispatchError> {
    a.checked_mul(b).ok_or_else(plan_overflow)
}

impl GroupedMoeScratchPlan {
    /// Extents of one core (`protocol` is resolved against `rows`).
    pub fn for_protocol(
        g: &GroupedMoeGeometry,
        rows: usize,
        protocol: GroupedMoeProtocol,
    ) -> Result<Self, DispatchError> {
        g.validate()?;
        let slots = mul(rows, g.top_k)?;
        let mut p = Self {
            router_logits: mul(rows, g.n_experts)?,
            topk_indices: slots,
            topk_weights: slots,
            x_rot: mul(rows, g.hidden)?,
            act: mul(slots, g.down_k)?,
            rot: mul(slots, g.down_k)?,
            shared_gate: mul(rows, g.shared_mi)?,
            shared_up: mul(rows, g.shared_mi)?,
            shared_act: mul(rows, g.shared_down_k)?,
            shared_rot: mul(rows, g.shared_down_k)?,
            shared_tables: if g.shared_mi == 0 {
                0
            } else {
                rdna_compute::ffn_activation::single_expert_table_words(rows)
            },
            shared_out: if g.shared_mi == 0 {
                0
            } else {
                mul(rows, g.hidden)?
            },
            ..Self::default()
        };
        match protocol.resolve(rows) {
            GroupedMoeProtocol::Indexed => {
                p.down_rows = mul(slots, g.hidden)?;
                p.indexed_gate = mul(slots, g.logical_mi)?;
                p.indexed_up = p.indexed_gate;
            }
            GroupedMoeProtocol::Grouped => {
                let m_total = grouped_moe_m_total_bound(slots, g.n_experts);
                p.m_total = m_total;
                p.down_rows = mul(m_total, g.hidden)?;
                p.gate_up_rows = mul(m_total, 2 * g.logical_mi)?;
                p.expert_token_counts = g.n_experts;
                p.expert_offsets = g.n_experts + 1;
                p.sorted_slot_index = m_total;
                p.expert_tile_ids = m_total / MOE_GROUPED_BLOCK_M;
                p.inverse_perm = slots;
            }
            GroupedMoeProtocol::Auto => unreachable!("resolve never returns Auto"),
        }
        Ok(p)
    }

    /// Element-wise maximum.
    pub fn max_with(self, o: Self) -> Self {
        Self {
            router_logits: self.router_logits.max(o.router_logits),
            topk_indices: self.topk_indices.max(o.topk_indices),
            topk_weights: self.topk_weights.max(o.topk_weights),
            x_rot: self.x_rot.max(o.x_rot),
            act: self.act.max(o.act),
            rot: self.rot.max(o.rot),
            down_rows: self.down_rows.max(o.down_rows),
            indexed_gate: self.indexed_gate.max(o.indexed_gate),
            indexed_up: self.indexed_up.max(o.indexed_up),
            gate_up_rows: self.gate_up_rows.max(o.gate_up_rows),
            m_total: self.m_total.max(o.m_total),
            expert_token_counts: self.expert_token_counts.max(o.expert_token_counts),
            expert_offsets: self.expert_offsets.max(o.expert_offsets),
            sorted_slot_index: self.sorted_slot_index.max(o.sorted_slot_index),
            expert_tile_ids: self.expert_tile_ids.max(o.expert_tile_ids),
            inverse_perm: self.inverse_perm.max(o.inverse_perm),
            shared_gate: self.shared_gate.max(o.shared_gate),
            shared_up: self.shared_up.max(o.shared_up),
            shared_act: self.shared_act.max(o.shared_act),
            shared_rot: self.shared_rot.max(o.shared_rot),
            shared_out: self.shared_out.max(o.shared_out),
            shared_tables: self.shared_tables.max(o.shared_tables),
        }
    }

    /// Extents covering both cores at `rows` rows (allocate once for the
    /// widest call: the indexed core at `rows` and the grouped core at `rows`).
    pub fn for_rows(g: &GroupedMoeGeometry, rows: usize) -> Result<Self, DispatchError> {
        Ok(
            Self::for_protocol(g, rows, GroupedMoeProtocol::Indexed)?.max_with(Self::for_protocol(
                g,
                rows,
                GroupedMoeProtocol::Grouped,
            )?),
        )
    }
}

/// Router linear weights and selection bias.
#[derive(Clone, Copy)]
pub struct GroupedMoeRouter<'a> {
    /// F32 `[n_experts, hidden]` (the router projection runs in FP32).
    pub weight: &'a GpuTensor,
    /// F32 `[n_experts]` selection-only correction bias; `None` is zero. Must
    /// be `None` for [`RouterKind::SoftmaxRenorm`].
    pub correction_bias: Option<&'a GpuTensor>,
}

/// Routed expert tables.
#[derive(Clone, Copy)]
pub struct GroupedMoeRouted<'a> {
    /// Wire format of every routed projection. Only `MQ4G256V2` (qt44) has the
    /// indexed and grouped cores wired here; anything else is a named refusal.
    pub dtype: DType,
    /// `[n_experts]` u64 device pointers to fused `[2*logical_mi, hidden]`
    /// gate-then-up tensors.
    pub gate_up_ptrs: &'a GpuTensor,
    /// `[n_experts]` u64 device pointers to `[hidden, down_k]` down tensors.
    pub down_ptrs: &'a GpuTensor,
    /// F32 `[hidden]` AWQ divide of the gate/up input basis (before the FWHT);
    /// shared with the shared expert's gate/up when present.
    pub input_awq: Option<&'a GpuTensor>,
    /// `[n_experts]` u64 pointers to per-expert F32 `[down_k]` AWQ divides of
    /// the down input (applied after the pad, before the FWHT).
    pub down_awq_ptrs: Option<&'a GpuTensor>,
    /// Every routed gate/up and down group header obeys `zp == -8 * sc` (checked at load); admits
    /// the grouped W4A4 core ([`grouped_moe_iu4_admitted`]).
    pub iu4_symmetric_verified: bool,
    /// Activation tier the caller selects for the grouped routed-expert GEMMs
    /// ([`grouped_moe_site_tiers`]); `F16` keeps the F16 WMMA core (kill switch). Only honoured when
    /// the capability admission holds, and overridable per site by `HIPFIRE_MOE_GROUPED_TIERS`.
    pub act_tier: GroupedMoeActTier,
}

/// Gate/up/down projections of one gated FFN (shared expert or dense layer).
#[derive(Clone, Copy)]
pub struct GroupedMoeShared<'a> {
    pub gate: WeightRef<'a>,
    pub up: WeightRef<'a>,
    pub down: WeightRef<'a>,
}

#[derive(Clone, Copy)]
pub struct GroupedMoeIndexedScratch<'a> {
    /// F32 `[rows * top_k * logical_mi]` each.
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
}

#[derive(Clone, Copy)]
pub struct GroupedMoeGroupedScratch<'a> {
    /// F32 `[m_total * 2 * logical_mi]`.
    pub gate_up_rows: &'a GpuTensor,
    /// i32 tables, see [`GroupedMoeScratchPlan`].
    pub expert_token_counts: &'a GpuTensor,
    pub expert_offsets: &'a GpuTensor,
    pub sorted_slot_index: &'a GpuTensor,
    pub expert_tile_ids: &'a GpuTensor,
    pub inverse_perm: &'a GpuTensor,
}

#[derive(Clone, Copy)]
pub struct GroupedMoeSharedScratch<'a> {
    pub gate_out: &'a GpuTensor,
    pub up_out: &'a GpuTensor,
    pub act: &'a GpuTensor,
    pub rot: &'a GpuTensor,
    /// Cast shared expert output `[rows, hidden]` (diagnostic: the shared-only
    /// result of the layer).
    pub out: &'a GpuTensor,
    /// F32 words, at least `single_expert_table_words(rows)`: grouping tables
    /// that run the qt44 gate/up/down weights as one-entry expert tables.
    pub tables: &'a GpuTensor,
}

#[derive(Clone, Copy)]
pub struct GroupedMoeScratch<'a> {
    pub router_logits: &'a GpuTensor,
    /// i32 `[rows, top_k]` (selected experts) and F32 weights; both are
    /// device-resident outputs of the router, never downloaded.
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub x_rot: &'a GpuTensor,
    pub act: &'a GpuTensor,
    pub rot: &'a GpuTensor,
    pub down_rows: &'a GpuTensor,
    pub indexed: Option<GroupedMoeIndexedScratch<'a>>,
    pub grouped: Option<GroupedMoeGroupedScratch<'a>>,
    pub shared: Option<GroupedMoeSharedScratch<'a>>,
}

/// Complete borrowed description of one grouped MoE layer call.
#[derive(Clone, Copy)]
pub struct GroupedMoeOp<'a> {
    pub ctx: &'a DispatchCtx,
    pub policy: GroupedMoePolicy,
    pub geometry: GroupedMoeGeometry,
    pub protocol: GroupedMoeProtocol,
    pub rows: usize,
    /// F32 `[rows, hidden]` normalized layer input (BF16-valued under
    /// [`CastPolicy::Bf16Boundaries`]).
    pub x: &'a GpuTensor,
    pub router: GroupedMoeRouter<'a>,
    pub routed: GroupedMoeRouted<'a>,
    pub shared: Option<GroupedMoeShared<'a>>,
    pub scratch: GroupedMoeScratch<'a>,
    /// F32 `[rows, hidden]` layer output (routed sum + shared).
    pub out: &'a GpuTensor,
    /// Optional F32 `[rows, hidden]` diagnostic: the routed sum after its cast.
    pub routed_out: Option<&'a GpuTensor>,
}

fn grouped_refusal(variant: &'static str, quant: &'static str) -> DispatchError {
    DispatchError::UnsupportedVariant {
        family: "moe",
        variant,
        arch: "",
        quant,
    }
}

fn scratch_need(t: &GpuTensor, elems: usize, what: &str) -> Result<(), DispatchError> {
    let need = elems.checked_mul(4).ok_or_else(plan_overflow)?;
    if t.buf.size() < need {
        return Err(DispatchError::Hip(format!(
            "grouped moe: scratch `{what}` holds {} bytes, needs {need}",
            t.buf.size()
        )));
    }
    Ok(())
}

fn f32_like(t: &GpuTensor, what: &str) -> Result<(), DispatchError> {
    if matches!(t.dtype, DType::F32 | DType::Raw) {
        Ok(())
    } else {
        Err(DispatchError::Hip(format!(
            "grouped moe: `{what}` must be an F32 tensor, got {:?}",
            t.dtype
        )))
    }
}

/// The projection formats the shared expert / dense FFN accepts and whether the
/// format consumes the FWHT-rotated (AWQ-divided) basis of its input.
fn projection_kind(
    dtype: DType,
    variant: &'static str,
) -> Result<(crate::types::KernelKey, bool), DispatchError> {
    use crate::types::KernelKey;
    match dtype {
        DType::MQ4G256V2 => Ok((KernelKey::GemmMq4G256V2, true)),
        DType::Q8_0 => Ok((KernelKey::GemmQ8_0BatchedChunked, false)),
        // F16-resident exact image of a BF16 source weight (prefill); decode
        // reads a Q8_0 copy of the same tensor.
        DType::F16 => Ok((KernelKey::GemmF16WmmaMb8, false)),
        DType::F32 => Ok((KernelKey::GemmF32Batched, false)),
        _ => Err(grouped_refusal(variant, "projection-dtype")),
    }
}

fn check_projection(
    w: &WeightRef<'_>,
    m: usize,
    k: usize,
    variant: &'static str,
) -> Result<(), DispatchError> {
    let (_, rotated) = projection_kind(w.dtype, variant)?;
    if w.m != m || w.k != k {
        return Err(DispatchError::Hip(format!(
            "grouped moe: projection geometry [{}x{}] != expected [{m}x{k}] ({variant})",
            w.m, w.k
        )));
    }
    if rotated && k % 256 != 0 {
        return Err(grouped_refusal(variant, "qt44-k-not-256-multiple"));
    }
    if !rotated && w.awq_scale.is_some() {
        return Err(grouped_refusal(variant, "awq-on-unrotated-projection"));
    }
    Ok(())
}

/// Both projections share one rotated input basis: their AWQ sidecars must be
/// both absent or both present (value identity is the loader's verified
/// contract; the divide actually applied is `input_awq`).
fn awq_presence_matches(a: Option<&GpuTensor>, b: Option<&GpuTensor>) -> bool {
    a.is_some() == b.is_some()
}

impl<'a> GroupedMoeOp<'a> {
    /// Admission and operand validation; no launch happens before it passes.
    /// Every refusal is named; nothing falls back to a token loop.
    pub fn validate(&self) -> Result<GroupedMoeScratchPlan, DispatchError> {
        let g = &self.geometry;
        g.validate()?;
        let protocol = self.protocol.resolve(self.rows);
        let plan = GroupedMoeScratchPlan::for_protocol(g, self.rows, protocol)?;
        if self.routed.dtype != DType::MQ4G256V2 {
            return Err(grouped_refusal("grouped-op-routed-dtype", "non-qt44"));
        }
        if protocol == GroupedMoeProtocol::Grouped
            && !(self.ctx.arch.has_wmma_w32() || self.ctx.arch.is_rdna4())
        {
            return Err(grouped_refusal("grouped-op-grouped-core-needs-wmma", ""));
        }
        match (self.policy.router, self.router.correction_bias) {
            (RouterKind::SoftmaxRenorm, Some(_)) => {
                return Err(grouped_refusal("grouped-op-softmax-router-with-bias", ""));
            }
            _ => {}
        }
        if self.policy.shared == SharedExpertPolicy::Absent {
            if self.shared.is_some() || g.shared_mi != 0 || self.scratch.shared.is_some() {
                return Err(grouped_refusal("grouped-op-shared-policy-mismatch", ""));
            }
        } else if self.shared.is_none() || g.shared_mi == 0 || self.scratch.shared.is_none() {
            return Err(grouped_refusal("grouped-op-shared-policy-mismatch", ""));
        }
        f32_like(self.x, "x")?;
        f32_like(self.out, "out")?;
        f32_like(self.router.weight, "router.weight")?;
        scratch_need(self.x, plan.x_rot, "x")?;
        scratch_need(self.out, plan.x_rot, "out")?;
        scratch_need(
            self.router.weight,
            mul(g.n_experts, g.hidden)?,
            "router.weight",
        )?;
        if let Some(b) = self.router.correction_bias {
            f32_like(b, "router.correction_bias")?;
            scratch_need(b, g.n_experts, "router.correction_bias")?;
        }
        if let Some(r) = self.routed_out {
            scratch_need(r, plan.x_rot, "routed_out")?;
        }
        scratch_need(
            self.routed.gate_up_ptrs,
            2 * g.n_experts,
            "routed.gate_up_ptrs",
        )?;
        scratch_need(self.routed.down_ptrs, 2 * g.n_experts, "routed.down_ptrs")?;
        if let Some(t) = self.routed.down_awq_ptrs {
            scratch_need(t, 2 * g.n_experts, "routed.down_awq_ptrs")?;
        }
        let s = &self.scratch;
        scratch_need(s.router_logits, plan.router_logits, "router_logits")?;
        scratch_need(s.topk_indices, plan.topk_indices, "topk_indices")?;
        scratch_need(s.topk_weights, plan.topk_weights, "topk_weights")?;
        scratch_need(s.x_rot, plan.x_rot, "x_rot")?;
        scratch_need(s.act, plan.act, "act")?;
        scratch_need(s.rot, plan.rot, "rot")?;
        scratch_need(s.down_rows, plan.down_rows, "down_rows")?;
        match protocol {
            GroupedMoeProtocol::Indexed => {
                let i = s
                    .indexed
                    .as_ref()
                    .ok_or_else(|| grouped_refusal("grouped-op-indexed-scratch-missing", ""))?;
                scratch_need(i.gate, plan.indexed_gate, "indexed.gate")?;
                scratch_need(i.up, plan.indexed_up, "indexed.up")?;
            }
            GroupedMoeProtocol::Grouped => {
                let t = s
                    .grouped
                    .as_ref()
                    .ok_or_else(|| grouped_refusal("grouped-op-grouped-scratch-missing", ""))?;
                scratch_need(t.gate_up_rows, plan.gate_up_rows, "grouped.gate_up_rows")?;
                scratch_need(
                    t.expert_token_counts,
                    plan.expert_token_counts,
                    "grouped.expert_token_counts",
                )?;
                scratch_need(
                    t.expert_offsets,
                    plan.expert_offsets,
                    "grouped.expert_offsets",
                )?;
                scratch_need(
                    t.sorted_slot_index,
                    plan.sorted_slot_index,
                    "grouped.sorted_slot_index",
                )?;
                scratch_need(
                    t.expert_tile_ids,
                    plan.expert_tile_ids,
                    "grouped.expert_tile_ids",
                )?;
                scratch_need(t.inverse_perm, plan.inverse_perm, "grouped.inverse_perm")?;
            }
            GroupedMoeProtocol::Auto => unreachable!("resolve never returns Auto"),
        }
        if let (Some(shared), Some(ss)) = (self.shared.as_ref(), s.shared.as_ref()) {
            check_projection(
                &shared.gate,
                g.shared_mi,
                g.hidden,
                "grouped-op-shared-gate",
            )?;
            check_projection(&shared.up, g.shared_mi, g.hidden, "grouped-op-shared-up")?;
            check_projection(
                &shared.down,
                g.hidden,
                g.shared_down_k,
                "grouped-op-shared-down",
            )?;
            // One rotated input basis feeds the routed gate/up and every
            // rotated shared gate/up: their AWQ presence must agree.
            let rotated_gate = projection_kind(shared.gate.dtype, "grouped-op-shared-gate")?.1;
            let rotated_up = projection_kind(shared.up.dtype, "grouped-op-shared-up")?.1;
            if (rotated_gate && !awq_presence_matches(shared.gate.awq_scale, self.routed.input_awq))
                || (rotated_up && !awq_presence_matches(shared.up.awq_scale, self.routed.input_awq))
            {
                return Err(grouped_refusal("grouped-op-input-awq-mismatch", ""));
            }
            scratch_need(ss.gate_out, plan.shared_gate, "shared.gate_out")?;
            scratch_need(ss.up_out, plan.shared_up, "shared.up_out")?;
            scratch_need(ss.act, plan.shared_act, "shared.act")?;
            scratch_need(ss.rot, plan.shared_rot, "shared.rot")?;
            scratch_need(ss.out, plan.shared_out, "shared.out")?;
            scratch_need(ss.tables, plan.shared_tables, "shared.tables")?;
        }
        Ok(plan)
    }
}

/// Rotated (FWHT-256, optional AWQ-divided) basis of an F32 activation block.
fn rotate_input_basis(
    gpu: &mut Gpu,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    awq: Option<&GpuTensor>,
    k: usize,
    rows: usize,
) -> Result<(), DispatchError> {
    match awq {
        Some(awq) => hip!(gpu.rotate_x_mq_awq_batched(x, awq, x_rot, k, rows)),
        None => hip!(gpu.rotate_x_mq_batched(x, x_rot, k, rows)),
    }
}

/// Gated FFN from the already prepared input bases: `down(act(gate(x), up(x)))`.
/// `x` is the plain F32 input (consumed by unrotated projection formats) and
/// `x_rot` its rotated (AWQ-divided) basis, consumed by qt44 projections. The
/// activation result is padded from `mi` to the down projection's physical K
/// with exact zeros, AWQ-divided by the down weight's own sidecar when it
/// carries one, rotated for qt44 down weights, then projected; the down output
/// is cast per policy.
#[allow(clippy::too_many_arguments)]
fn gated_ffn_from_basis(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    activation: FfnActivation,
    cast: CastPolicy,
    rows: usize,
    hidden: usize,
    mi: usize,
    down_k: usize,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    w: &GroupedMoeShared<'_>,
    s: &GroupedMoeSharedScratch<'_>,
    out: &GpuTensor,
) -> Result<(), DispatchError> {
    static GEMM: std::sync::LazyLock<GemmFamily> = std::sync::LazyLock::new(GemmFamily::new);
    let gemm = &*GEMM;
    let (down_key, down_rotated) = projection_kind(w.down.dtype, "gated-ffn-down")?;
    // One-entry grouping tables drive qt44 weights through the grouped F16 WMMA
    // core (BF16 policy, rows >= 2) or the F32 indexed GEMV (exact policy), so a
    // shared/dense projection never takes the quantized-activation prefill GEMMs
    // (A8 / IU4 / FP8) that the dense GEMM entry selects at wide batches.
    let any_qt44 = [&w.gate, &w.up, &w.down]
        .iter()
        .any(|p| p.dtype == DType::MQ4G256V2);
    let views = if any_qt44 && !(cast.bf16() && rows == 1) {
        hip!(gpu.ffn_single_expert_tables(s.tables, rows, [w.gate.buf, w.up.buf, w.down.buf]))?;
        Some(Gpu::single_expert_views(s.tables, rows))
    } else {
        None
    };
    let project = |gpu: &mut Gpu,
                   weight: &WeightRef<'_>,
                   which: usize,
                   xin: &GpuTensor,
                   y: &GpuTensor,
                   key: crate::types::KernelKey,
                   m: usize,
                   k: usize|
     -> Result<(), DispatchError> {
        if let (Some(v), DType::MQ4G256V2) = (views.as_ref(), weight.dtype) {
            let ptr = [&v.gate_ptr, &v.up_ptr, &v.down_ptr][which];
            if !cast.bf16() {
                // FP32 activations and accumulation: the indexed GEMV with
                // one rank per row (`top_k = 1`, expert 0).
                return hip!(gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
                    ptr, &v.index, xin, y, m, k, 1, rows,
                ));
            }
            if ctx.arch.has_wmma_w32() || ctx.arch.is_rdna4() {
                return hip!(gpu.gemm_mq4g256v2_moe_grouped_wmma_k2(
                    ptr, &v.tiles, &v.sorted, xin, y, m, k, 1, rows, rows,
                ));
            }
        }
        // A single row (decode) over a Q8_0 weight: the Q8 GEMV, not a
        // one-column WMMA tile.
        if rows == 1 && weight.dtype == DType::Q8_0 {
            return hip!(gpu.gemv_q8_0(weight.buf, xin, y, m, k));
        }
        gemm.run_key(
            key,
            ctx,
            gpu,
            &GemmParams {
                w: weight,
                x: xin,
                y,
                batch_size: rows,
            },
        )
    };
    for (which, (weight, y)) in [(&w.gate, s.gate_out), (&w.up, s.up_out)]
        .into_iter()
        .enumerate()
    {
        let (key, rotated) = projection_kind(weight.dtype, "gated-ffn-gate-up")?;
        project(
            gpu,
            weight,
            which,
            if rotated { x_rot } else { x },
            y,
            key,
            mi,
            hidden,
        )?;
    }
    let awq = match w.down.awq_scale {
        Some(t) if down_rotated => FfnAwq::Vector(t),
        _ => FfnAwq::None,
    };
    hip!(gpu.ffn_activation_rows_f32(&FfnActivationRows {
        activation,
        gate: s.gate_out,
        gate_offset: 0,
        gate_ld: mi,
        up: s.up_out,
        up_offset: 0,
        up_ld: mi,
        out: s.act,
        rows,
        logical_n: mi,
        phys_n: down_k,
        round_inputs: cast.bf16(),
        round_output: cast.bf16(),
        awq,
    }))?;
    let down_x = if down_rotated {
        hip!(gpu.rotate_x_mq_batched(s.act, s.rot, down_k, rows))?;
        s.rot
    } else {
        s.act
    };
    project(gpu, &w.down, 2, down_x, out, down_key, hidden, down_k)?;
    if cast.bf16() {
        hip!(gpu.ffn_round_bf16_f32(out, rows * hidden))?;
    }
    Ok(())
}

/// Dense gated FFN (the first-layer dense MLP, or any non-routed gated FFN) as
/// a standalone program: one slot, no router.
#[derive(Clone, Copy)]
pub struct GatedFfnOp<'a> {
    pub ctx: &'a DispatchCtx,
    pub activation: FfnActivation,
    pub cast: CastPolicy,
    pub rows: usize,
    pub hidden: usize,
    /// Logical intermediate width of gate/up (`weights.gate.m`).
    pub mi: usize,
    /// Physical K of the down projection (`>= mi`, `% 256 == 0`).
    pub down_k: usize,
    /// F32 `[rows, hidden]` normalized input.
    pub x: &'a GpuTensor,
    /// AWQ divide of the gate/up input basis (before the FWHT).
    pub input_awq: Option<&'a GpuTensor>,
    pub weights: GroupedMoeShared<'a>,
    /// F32 `[rows, hidden]` rotated basis scratch.
    pub x_rot: &'a GpuTensor,
    pub scratch: GroupedMoeSharedScratch<'a>,
    /// F32 `[rows, hidden]` result (aliasing `scratch.out` is allowed).
    pub out: &'a GpuTensor,
}

/// Execute a [`GatedFfnOp`].
pub fn execute_gated_ffn(gpu: &mut Gpu, op: &GatedFfnOp<'_>) -> Result<(), DispatchError> {
    if op.rows == 0 {
        return Ok(());
    }
    if op.hidden == 0 || op.mi == 0 || op.down_k < op.mi {
        return Err(grouped_refusal("gated-ffn-geometry", ""));
    }
    let r = op.rows;
    check_projection(&op.weights.gate, op.mi, op.hidden, "gated-ffn-gate")?;
    check_projection(&op.weights.up, op.mi, op.hidden, "gated-ffn-up")?;
    check_projection(&op.weights.down, op.hidden, op.down_k, "gated-ffn-down")?;
    let rotated_gate = projection_kind(op.weights.gate.dtype, "gated-ffn-gate")?.1;
    let rotated_up = projection_kind(op.weights.up.dtype, "gated-ffn-up")?.1;
    if (rotated_gate && !awq_presence_matches(op.weights.gate.awq_scale, op.input_awq))
        || (rotated_up && !awq_presence_matches(op.weights.up.awq_scale, op.input_awq))
    {
        return Err(grouped_refusal("gated-ffn-input-awq-mismatch", ""));
    }
    f32_like(op.x, "x")?;
    scratch_need(op.x, mul(r, op.hidden)?, "x")?;
    scratch_need(op.x_rot, mul(r, op.hidden)?, "x_rot")?;
    scratch_need(op.out, mul(r, op.hidden)?, "out")?;
    scratch_need(op.scratch.gate_out, mul(r, op.mi)?, "gate_out")?;
    scratch_need(op.scratch.up_out, mul(r, op.mi)?, "up_out")?;
    scratch_need(op.scratch.act, mul(r, op.down_k)?, "act")?;
    scratch_need(op.scratch.rot, mul(r, op.down_k)?, "rot")?;
    scratch_need(
        op.scratch.tables,
        rdna_compute::ffn_activation::single_expert_table_words(r),
        "tables",
    )?;
    if rotated_gate || rotated_up {
        if op.hidden % 256 != 0 {
            return Err(grouped_refusal("gated-ffn-geometry", ""));
        }
        rotate_input_basis(gpu, op.x, op.x_rot, op.input_awq, op.hidden, r)?;
    }
    gated_ffn_from_basis(
        gpu,
        op.ctx,
        op.activation,
        op.cast,
        r,
        op.hidden,
        op.mi,
        op.down_k,
        op.x,
        op.x_rot,
        &op.weights,
        &op.scratch,
        op.out,
    )
}

/// Activation precision of one grouped routed-expert GEMM site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupedMoeActTier {
    /// F16 WMMA (`gemm_mq4g256v2_moe_grouped_wmma_k2`), the default route.
    F16,
    /// Symmetric W4A4 (`wmma_i32_16x16x32_iu4`, `block_i4_128` activations).
    A4,
    /// Symmetric W4A8 (`wmma_i32_16x16x16_iu8`, `block_i8_128` activations).
    A8,
}

impl GroupedMoeActTier {
    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f16" => Some(Self::F16),
            "a4" => Some(Self::A4),
            "a8" => Some(Self::A8),
            _ => None,
        }
    }
}

/// `HIPFIRE_MOE_GROUPED_TIERS=<gate_up>,<down>` (developer; each `f16|a4|a8`): per-site tier override
/// of the grouped routed-expert core, applied on top of the capability admission (symmetric headers,
/// qt44, shape, gfx1201, rows). Unset: both sites take the device tier ([`grouped_moe_device_tier`]).
fn parse_site_tiers(s: &str) -> Option<(GroupedMoeActTier, GroupedMoeActTier)> {
    let (g, d) = s.split_once(',')?;
    Some((GroupedMoeActTier::parse(g)?, GroupedMoeActTier::parse(d)?))
}

/// Capability admission of the grouped symmetric quantized-activation routed-expert core of
/// [`execute_grouped_moe`], resolved per site `(gate_up, down)`: the caller bound loader-verified
/// symmetric headers (`zp == -8 * sc` on every gate/up and down group,
/// [`GroupedMoeRouted::iu4_symmetric_verified`]), the device is gfx1201, the experts are qt44
/// (`MQ4G256V2`) and the shape fits the kernels ([`grouped_moe_iu4_shape_admits`]). The tier of each
/// site is the caller's [`GroupedMoeRouted::act_tier`] unless `HIPFIRE_MOE_GROUPED_TIERS` overrides
/// it. AWQ (input vector, per-expert down table), the activation (SiLU/SiTU), the pad of the down K
/// and the cast policy are handled by the stages around the two GEMMs and place no condition here. The
/// route allocates only through the activation scratch, so it is refused under graph capture and
/// Redline recording. Anything not admitted is `(F16, F16)`.
pub fn grouped_moe_site_tiers(
    gpu: &Gpu,
    op: &GroupedMoeOp<'_>,
) -> (GroupedMoeActTier, GroupedMoeActTier) {
    use GroupedMoeActTier::F16;
    static OVERRIDE: std::sync::LazyLock<Option<(GroupedMoeActTier, GroupedMoeActTier)>> =
        std::sync::LazyLock::new(|| {
            let v = hipfire_config::developer_var("HIPFIRE_MOE_GROUPED_TIERS").ok()?;
            let parsed = parse_site_tiers(&v);
            if parsed.is_none() {
                eprintln!("HIPFIRE_MOE_GROUPED_TIERS={v:?}: expected `<gate_up>,<down>` with each of f16|a4|a8; ignored");
            }
            parsed
        });
    let admitted = gpu.arch == "gfx1201"
        && op.routed.iu4_symmetric_verified
        && op.routed.dtype == DType::MQ4G256V2
        && grouped_moe_iu4_shape_admits(&op.geometry, op.rows)
        && !gpu.graphs.capture_mode
        && !gpu.replay.is_recording();
    if !admitted {
        return (F16, F16);
    }
    let tier = op.routed.act_tier;
    OVERRIDE.unwrap_or((tier, tier))
}

/// Shape half of [`grouped_moe_iu4_admitted`] (pure): at least [`GROUPED_MOE_IU4_MIN_ROWS`] rows,
/// both GEMM K extents (`hidden`, the padded `down_k`) whole qt44 groups of 256, and both M extents
/// (`2 * logical_mi` gate/up rows, `hidden` down rows) multiples of the kernel's 64-row CTA tile.
pub fn grouped_moe_iu4_shape_admits(g: &GroupedMoeGeometry, rows: usize) -> bool {
    rows >= GROUPED_MOE_IU4_MIN_ROWS
        && g.hidden % 256 == 0
        && g.down_k % 256 == 0
        && (2 * g.logical_mi) % 64 == 0
        && g.hidden % 64 == 0
}

/// Row floor of the grouped IU4 core: below it the weight stream of the expert set dominates
/// both routes and the F16 core stays.
pub const GROUPED_MOE_IU4_MIN_ROWS: usize = 64;

/// Execute one grouped MoE layer: FP32 router linear and device-resident
/// top-k, shared rotated input basis, routed experts on the indexed or the
/// grouped core, the optional ungated shared expert, and the policy combine.
/// Never downloads routing and allocates nothing: every launch is graph-capture
/// safe (`launch_maybe_blob`).
pub fn execute_grouped_moe(gpu: &mut Gpu, op: &GroupedMoeOp<'_>) -> Result<(), DispatchError> {
    if op.rows == 0 {
        return Ok(());
    }
    let _plan = op.validate()?;
    let g = op.geometry;
    let s = &op.scratch;
    let rows = op.rows;
    let k = g.top_k;
    let slots = rows * k;
    let bf16 = op.policy.cast.bf16();
    let protocol = op.protocol.resolve(rows);
    let (tier_gu, tier_dn) = if protocol == GroupedMoeProtocol::Grouped {
        grouped_moe_site_tiers(gpu, op)
    } else {
        (GroupedMoeActTier::F16, GroupedMoeActTier::F16)
    };

    // Router: FP32 linear of the (BF16-valued) F32 input, then device top-k.
    hip!(crate::pipeline::stream_mix::f32_rows_gemm(
        gpu,
        op.router.weight,
        op.x,
        s.router_logits,
        g.n_experts,
        g.hidden,
        rows,
    ))?;
    hip!(gpu.ffn_moe_router_topk_f32(&MoeRouterTopK {
        kind: op.policy.router,
        logits: s.router_logits,
        correction_bias: op.router.correction_bias,
        topk_indices: s.topk_indices,
        topk_weights: s.topk_weights,
        rows,
        n_experts: g.n_experts,
        top_k: k,
    }))?;

    // One rotated input basis for the routed gate/up and the shared gate/up.
    rotate_input_basis(gpu, op.x, s.x_rot, op.routed.input_awq, g.hidden, rows)?;

    let down_awq_table = op.routed.down_awq_ptrs;
    let row_of_slot = match protocol {
        GroupedMoeProtocol::Indexed => {
            let i = s.indexed.as_ref().expect("validated");
            hip!(gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed_batched(
                op.routed.gate_up_ptrs,
                s.topk_indices,
                s.x_rot,
                i.gate,
                i.up,
                2 * g.logical_mi,
                g.hidden,
                k,
                rows,
            ))?;
            let awq = match down_awq_table {
                Some(table) => FfnAwq::PerExpert {
                    table,
                    row_expert: s.topk_indices,
                    n_experts: g.n_experts,
                },
                None => FfnAwq::None,
            };
            hip!(gpu.ffn_activation_rows_f32(&FfnActivationRows {
                activation: op.policy.activation,
                gate: i.gate,
                gate_offset: 0,
                gate_ld: g.logical_mi,
                up: i.up,
                up_offset: 0,
                up_ld: g.logical_mi,
                out: s.act,
                rows: slots,
                logical_n: g.logical_mi,
                phys_n: g.down_k,
                round_inputs: bf16,
                round_output: bf16,
                awq,
            }))?;
            hip!(gpu.rotate_x_mq_batched(s.act, s.rot, g.down_k, slots))?;
            hip!(gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
                op.routed.down_ptrs,
                s.topk_indices,
                s.rot,
                s.down_rows,
                g.hidden,
                g.down_k,
                k,
                rows,
            ))?;
            None
        }
        GroupedMoeProtocol::Grouped => {
            let t = s.grouped.as_ref().expect("validated");
            let m_total = grouped_moe_m_total_bound(slots, g.n_experts);
            hip!(gpu.moe_scatter_fused_k8(
                s.topk_indices,
                t.expert_token_counts,
                t.expert_offsets,
                t.sorted_slot_index,
                t.expert_tile_ids,
                t.inverse_perm,
                slots,
                g.n_experts,
                m_total,
                MOE_GROUPED_BLOCK_M,
            ))?;
            if tier_gu == GroupedMoeActTier::A8 {
                let (nt, rf) =
                    rdna_compute::moe_grouped_qt44_iu4::select_entry_a8(m_total, 2 * g.logical_mi);
                let xq = hip!(gpu.ensure_int8_mmq_x(s.x_rot, rows, g.hidden))?;
                hip!(gpu.gemm_mq4g256v2_moe_grouped_a8_sym(
                    op.routed.gate_up_ptrs,
                    t.expert_tile_ids,
                    t.expert_offsets,
                    t.sorted_slot_index,
                    xq,
                    t.gate_up_rows,
                    2 * g.logical_mi,
                    g.hidden,
                    k,
                    m_total,
                    rows,
                    nt,
                    rf,
                ))?;
            } else if tier_gu == GroupedMoeActTier::A4 {
                let (nt, rf) =
                    rdna_compute::moe_grouped_qt44_iu4::select_entry(m_total, 2 * g.logical_mi);
                let xq = hip!(gpu.ensure_int4_mmq_x(s.x_rot, rows, g.hidden))?;
                hip!(gpu.gemm_mq4g256v2_moe_grouped_iu4_sym(
                    op.routed.gate_up_ptrs,
                    t.expert_tile_ids,
                    t.expert_offsets,
                    t.sorted_slot_index,
                    xq,
                    t.gate_up_rows,
                    2 * g.logical_mi,
                    g.hidden,
                    k,
                    m_total,
                    rows,
                    nt,
                    rf,
                ))?;
            } else {
                hip!(gpu.gemm_mq4g256v2_moe_grouped_wmma_k2(
                    op.routed.gate_up_ptrs,
                    t.expert_tile_ids,
                    t.sorted_slot_index,
                    s.x_rot,
                    t.gate_up_rows,
                    2 * g.logical_mi,
                    g.hidden,
                    k,
                    m_total,
                    rows,
                ))?;
            }
            hip!(gpu.ffn_activation_grouped_f32(&FfnActivationGrouped {
                activation: op.policy.activation,
                y_grouped: t.gate_up_rows,
                sorted_slot_index: t.sorted_slot_index,
                expert_tile_ids: t.expert_tile_ids,
                out: s.act,
                m_total,
                block_m: MOE_GROUPED_BLOCK_M,
                logical_n: g.logical_mi,
                phys_n: g.down_k,
                round_inputs: bf16,
                round_output: bf16,
                awq_table: down_awq_table,
                n_experts: g.n_experts,
            }))?;
            hip!(gpu.rotate_x_mq_batched(s.act, s.rot, g.down_k, slots))?;
            if tier_dn == GroupedMoeActTier::A8 {
                let (nt, rf) =
                    rdna_compute::moe_grouped_qt44_iu4::select_entry_a8(m_total, g.hidden);
                let xq = hip!(gpu.ensure_int8_mmq_x(s.rot, slots, g.down_k))?;
                hip!(gpu.gemm_mq4g256v2_moe_grouped_a8_sym(
                    op.routed.down_ptrs,
                    t.expert_tile_ids,
                    t.expert_offsets,
                    t.sorted_slot_index,
                    xq,
                    s.down_rows,
                    g.hidden,
                    g.down_k,
                    1,
                    m_total,
                    slots,
                    nt,
                    rf,
                ))?;
            } else if tier_dn == GroupedMoeActTier::A4 {
                let (nt, rf) = rdna_compute::moe_grouped_qt44_iu4::select_entry(m_total, g.hidden);
                let xq = hip!(gpu.ensure_int4_mmq_x(s.rot, slots, g.down_k))?;
                hip!(gpu.gemm_mq4g256v2_moe_grouped_iu4_sym(
                    op.routed.down_ptrs,
                    t.expert_tile_ids,
                    t.expert_offsets,
                    t.sorted_slot_index,
                    xq,
                    s.down_rows,
                    g.hidden,
                    g.down_k,
                    1,
                    m_total,
                    slots,
                    nt,
                    rf,
                ))?;
            } else {
                hip!(gpu.gemm_mq4g256v2_moe_grouped_wmma_k2(
                    op.routed.down_ptrs,
                    t.expert_tile_ids,
                    t.sorted_slot_index,
                    s.rot,
                    s.down_rows,
                    g.hidden,
                    g.down_k,
                    1,
                    m_total,
                    slots,
                ))?;
            }
            Some(t.inverse_perm)
        }
        GroupedMoeProtocol::Auto => unreachable!("resolve never returns Auto"),
    };

    // Ungated shared expert from the same rotated basis; cast on its own, then
    // added after the routed sum is cast.
    let shared_sum = match (op.shared.as_ref(), s.shared.as_ref()) {
        (Some(w), Some(ss)) => {
            gated_ffn_from_basis(
                gpu,
                op.ctx,
                op.policy.activation,
                op.policy.cast,
                rows,
                g.hidden,
                g.shared_mi,
                g.shared_down_k,
                op.x,
                s.x_rot,
                w,
                ss,
                ss.out,
            )?;
            Some(ss.out)
        }
        _ => None,
    };

    hip!(gpu.ffn_moe_combine_f32(&MoeCombine {
        order: op.policy.combine,
        round_expert_outputs: bf16,
        round_routed_sum: bf16,
        round_final: bf16,
        down_rows: s.down_rows,
        row_of_slot,
        topk_indices: s.topk_indices,
        topk_weights: s.topk_weights,
        shared: shared_sum,
        routed_out: op.routed_out,
        out: op.out,
        rows,
        hidden: g.hidden,
        top_k: k,
        n_experts: g.n_experts,
    }))
}

#[cfg(test)]
mod grouped_moe_tests {
    use super::*;

    fn raon_geometry() -> GroupedMoeGeometry {
        GroupedMoeGeometry {
            n_experts: 128,
            top_k: 6,
            hidden: 2560,
            logical_mi: 1088,
            down_k: 1280,
            shared_mi: 1088,
            shared_down_k: 1280,
        }
    }

    #[test]
    fn iu4_shape_admits_the_raon_geometry_and_refuses_off_domain_shapes() {
        let g = raon_geometry();
        assert!(grouped_moe_iu4_shape_admits(&g, 4096));
        assert!(grouped_moe_iu4_shape_admits(&g, GROUPED_MOE_IU4_MIN_ROWS));
        assert!(!grouped_moe_iu4_shape_admits(
            &g,
            GROUPED_MOE_IU4_MIN_ROWS - 1
        ));
        // down K must be whole qt44 groups (the logical 1088 is padded to 1280).
        assert!(!grouped_moe_iu4_shape_admits(
            &GroupedMoeGeometry { down_k: 1088, ..g },
            4096
        ));
        assert!(!grouped_moe_iu4_shape_admits(
            &GroupedMoeGeometry { hidden: 2688, ..g },
            4096
        ));
        assert!(!grouped_moe_iu4_shape_admits(
            &GroupedMoeGeometry {
                logical_mi: 1040,
                ..g
            },
            4096
        ));
    }

    #[test]
    fn site_tiers_parse() {
        use GroupedMoeActTier::{A4, A8, F16};
        assert_eq!(parse_site_tiers("a4,f16"), Some((A4, F16)));
        assert_eq!(parse_site_tiers(" F16 , A8 "), Some((F16, A8)));
        assert_eq!(parse_site_tiers("a4"), None);
        assert_eq!(parse_site_tiers("a4,a5"), None);
    }

    #[test]
    fn protocol_auto_resolves_by_rows() {
        assert_eq!(
            GroupedMoeProtocol::Auto.resolve(1),
            GroupedMoeProtocol::Indexed
        );
        assert_eq!(
            GroupedMoeProtocol::Auto.resolve(2),
            GroupedMoeProtocol::Grouped
        );
        assert_eq!(
            GroupedMoeProtocol::Grouped.resolve(1),
            GroupedMoeProtocol::Grouped
        );
        assert_eq!(
            GroupedMoeProtocol::Indexed.resolve(512),
            GroupedMoeProtocol::Indexed
        );
    }

    #[test]
    fn scratch_plan_matches_grouped_row_bound() {
        let g = raon_geometry();
        let p = GroupedMoeScratchPlan::for_protocol(&g, 512, GroupedMoeProtocol::Auto).unwrap();
        // 3072 slots + 128 touched experts * 15 pad rows.
        assert_eq!(p.m_total, 4992);
        assert_eq!(p.m_total % MOE_GROUPED_BLOCK_M, 0);
        assert_eq!(p.gate_up_rows, 4992 * 2176);
        assert_eq!(p.down_rows, 4992 * 2560);
        assert_eq!(p.act, 3072 * 1280);
        assert_eq!(p.expert_tile_ids, 4992 / 16);
        assert_eq!(p.inverse_perm, 3072);
        assert_eq!(p.indexed_gate, 0);
        assert_eq!(p.shared_act, 512 * 1280);
        assert_eq!(p.shared_out, 512 * 2560);
        // 3 u64 pointers + index/sorted (512 each) + 32 tile ids.
        assert_eq!(p.shared_tables, 6 + 2 * 512 + 32);
    }

    #[test]
    fn scratch_plan_decode_uses_expanded_rows() {
        let g = raon_geometry();
        let p = GroupedMoeScratchPlan::for_protocol(&g, 1, GroupedMoeProtocol::Auto).unwrap();
        assert_eq!(p.down_rows, 6 * 2560);
        assert_eq!(p.indexed_gate, 6 * 1088);
        assert_eq!(p.indexed_up, 6 * 1088);
        assert_eq!(p.m_total, 0);
        let u = GroupedMoeScratchPlan::for_rows(&g, 1).unwrap();
        assert_eq!(u.indexed_gate, 6 * 1088);
        // One grouped row tile per touched expert at N = 1: 6 slots + 6*15.
        assert_eq!(u.m_total, 96);
    }

    #[test]
    fn geometry_admits_raon_and_other_shapes_and_refuses_with_names() {
        assert!(raon_geometry().validate().is_ok());
        let small = GroupedMoeGeometry {
            n_experts: 16,
            top_k: 3,
            hidden: 256,
            logical_mi: 192,
            down_k: 256,
            shared_mi: 0,
            shared_down_k: 0,
        };
        assert!(small.validate().is_ok());
        for k in 1..=8 {
            let mut g = raon_geometry();
            g.top_k = k;
            assert!(g.validate().is_ok(), "top_k {k}");
        }
        let refuse = |g: GroupedMoeGeometry| match g.validate() {
            Err(DispatchError::UnsupportedVariant { variant, .. }) => variant,
            other => panic!("expected refusal, got {other:?}"),
        };
        let mut g = raon_geometry();
        g.top_k = 9;
        assert_eq!(refuse(g), "grouped-op-top-k-outside-1..=8-or-experts");
        g = raon_geometry();
        g.top_k = 0;
        assert_eq!(refuse(g), "grouped-op-top-k-outside-1..=8-or-experts");
        g = raon_geometry();
        g.n_experts = 1025;
        assert_eq!(refuse(g), "grouped-op-top-k-outside-1..=8-or-experts");
        g = raon_geometry();
        g.down_k = 1088;
        assert_eq!(refuse(g), "grouped-op-down-k-geometry");
        g = raon_geometry();
        g.hidden = 2500;
        assert_eq!(refuse(g), "grouped-op-hidden-not-256-multiple");
        g = raon_geometry();
        g.shared_down_k = 1000;
        assert_eq!(refuse(g), "grouped-op-shared-geometry");
    }
}
