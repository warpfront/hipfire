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
use crate::families::moe::{
    MoeNormalization, MoeParams, MoePrefillParams, MoePrefillResolution, MoeQ8RouterPolicy,
    MoeRecipe, MoeResolution,
};
use crate::pipeline::sealed_moe::{MoeProtocol, MoeRouterInput, SealedMoeCall};
use crate::types::DispatchError;
use rdna_compute::{Gpu, GpuTensor};
use smallvec::SmallVec;
use std::cell::{Cell, OnceCell};
use std::sync::OnceLock;

const INLINE_STEP_CAPACITY: usize = 32;

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
    pub(super) ninepath_d3: bool,
    pub(super) ninepath_d4: bool,
    pub(super) ninepath_mq3l: bool,
    pub(super) ninepath_mq4v2: bool,
    pub(super) ninepath_mq6v2: bool,
}

/// Static grouped-prefill choices resolved while sealing. It deliberately
/// contains only scalar choices and the family-owned resolution record.
pub(super) struct MoePrefillSelection {
    pub(super) resolution: MoePrefillResolution,
    pub(super) path2_m_total: usize,
    pub(super) force_mq4_grouped_fp16: bool,
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
    scores: usize,
    normalized: bool,
    adopted_from: Option<u64>,
}

/// Invocation-local mutable state. It borrows the sealed call but never gets
/// stored by that call, so borrowed Steps cannot form a self-referential owner.
pub(super) struct MoeStepState<'a> {
    call: &'a SealedMoeCall<'a>,
    host_route: OnceCell<HostRoute>,
    route_stamp: Cell<Option<RouteStamp>>,
    shared_gate: Option<GpuTensor>,
    shared_up: Option<GpuTensor>,
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
                    scores,
                    normalized,
                    adopted_from,
                )| RouteStamp {
                    invocation,
                    protocol,
                    router,
                    n_experts,
                    k_top,
                    indices_tensor,
                    weights_tensor,
                    scores,
                    normalized,
                    adopted_from,
                },
            );
        Self {
            call,
            host_route: OnceCell::new(),
            route_stamp: Cell::new(route_stamp),
            shared_gate,
            shared_up,
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
                ))
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
                ))
            }
        };
        Ok((params, selection))
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
        scores: usize,
        normalized: bool,
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
            scores,
            normalized,
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
        // Producer score identity and normalization/provenance are diagnostic;
        // invocation and route-buffer identity remain the execution authority.
        let _ = (stamp.scores, stamp.normalized, stamp.adopted_from);
        Ok(())
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

    pub(super) fn normalize(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => {
                let (params, _) = self.state.decode_parts()?;
                if let MoeNormalization::RmsNorm {
                    weight,
                    plain_out,
                    eps,
                } = &params.normalization
                {
                    if params.x_rot_prerotated {
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
                let (params, _) = self.state.prefill_parts()?;
                if let MoeNormalization::RmsNorm {
                    weight,
                    plain_out,
                    eps,
                } = &params.prelude.normalization
                {
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
        Ok(())
    }

    pub(super) fn input_basis(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if let Ok((params, selection)) = self.state.decode_parts() {
            return super::decode_input_basis_stage(gpu, params, selection.resolution);
        }
        let (params, selection) = self.state.prefill_parts()?;
        let input_weight = params
            .prelude
            .shared
            .as_ref()
            .map(|shared| &shared.weights.gate);
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
        )
    }

    pub(super) fn router_projection(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => {
                let (params, _) = self.state.decode_parts()?;
                if params.recipe == MoeRecipe::SoftmaxGatedShared {
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
                let (params, _) = self.state.prefill_parts()?;
                let prelude = &params.prelude;
                let x = match prelude.router.dtype {
                    rdna_compute::DType::Q8_0 | rdna_compute::DType::F32 => params.x_norm_batch,
                    _ => params.x_rot_batch,
                };
                let result = match prelude.q8_router_policy {
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
                                })
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
                if params.recipe == MoeRecipe::SoftmaxGatedShared {
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
                    scores.buf.as_ptr() as usize,
                    params.prelude.norm_topk_prob,
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
                            gpu.sigmoid_f32(params.router_logits)
                                .map_err(|e| DispatchError::Hip(e.to_string()))?;
                            gpu.moe_topk_renorm_k8(
                                params.router_logits,
                                params.topk_indices,
                                params.topk_weights,
                                params.n_exp,
                                params.norm_topk_prob,
                            )
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
                self.state.install_route_stamp(
                    params.n_exp,
                    params.k,
                    std::ptr::addr_of!(*params.router_logits) as usize,
                    params.norm_topk_prob,
                    None,
                )
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
            );
        }
        let (params, _) = self.state.prefill_parts()?;
        super::prefill_shared_gate_up_stage(self.state.dispatch_ctx(), gpu, params)
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
            );
        }
        let (params, _) = self.state.prefill_parts()?;
        super::prefill_shared_activation_stage(gpu, params)
    }

    pub(super) fn shared_down(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
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
            );
        }
        let (params, _) = self.state.prefill_parts()?;
        super::prefill_shared_down_stage(self.state.dispatch_ctx(), gpu, params)
    }

    pub(super) fn scatter(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let (params, selection) = self.state.prefill_parts()?;
        self.state.require_route_ready()?;
        if !selection.resolution.use_path2 {
            return Err(DispatchError::Hip(
                "sealed moe: scatter is only valid for grouped prefill path 2".into(),
            ));
        }
        super::prefill_scatter_stage(gpu, params, selection.path2_m_total)
    }

    pub(super) fn gate_up(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        self.state.require_route_ready()?;
        if let Ok((params, selection)) = self.state.decode_parts() {
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
            );
        }
        let (params, selection) = self.state.prefill_parts()?;
        let total_slots = params
            .batch_size
            .checked_mul(params.k_top)
            .ok_or_else(|| DispatchError::Hip("sealed moe: prefill slot count overflows".into()))?;
        super::prefill_gate_up_stage(
            gpu,
            params,
            &selection.resolution,
            selection.path2_m_total,
            selection.force_mq4_grouped_fp16,
        )
        .and_then(|_| {
            let _ = total_slots;
            Ok(())
        })
    }

    pub(super) fn unscatter(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let (params, selection) = self.state.prefill_parts()?;
        if !selection.resolution.use_path2 {
            return Err(DispatchError::Hip(
                "sealed moe: unscatter is only valid for grouped prefill path 2".into(),
            ));
        }
        super::prefill_gate_up_unscatter_stage(gpu, params, selection.path2_m_total)
    }

    pub(super) fn activation(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if let Ok((params, selection)) = self.state.decode_parts() {
            return super::decode_activation_stage(gpu, params, selection.resolution);
        }
        let (params, selection) = self.state.prefill_parts()?;
        let total_slots = params
            .batch_size
            .checked_mul(params.k_top)
            .ok_or_else(|| DispatchError::Hip("sealed moe: prefill slot count overflows".into()))?;
        super::prefill_activation_stage(gpu, params, &selection.resolution, total_slots)
    }

    pub(super) fn down(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if let Ok((params, selection)) = self.state.decode_parts() {
            let target = params.routed_out.unwrap_or(params.x_residual);
            return super::decode_down_stage(
                gpu,
                params,
                selection.resolution,
                target,
                selection.ninepath_d4,
                selection.ninepath_mq3l,
                selection.ninepath_mq4v2,
                selection.ninepath_mq6v2,
                selection.down_last_combine,
            );
        }
        let (params, selection) = self.state.prefill_parts()?;
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
        )?;
        let compact_ep = matches!(
            &params.prelude.route,
            super::sealed_moe::PrefillRouteMode::ProduceRoot { .. }
                | super::sealed_moe::PrefillRouteMode::AdoptRoot { .. }
        );
        if compact_ep && selection.resolution.use_path2 {
            super::prefill_down_unscatter_stage(gpu, params, total_slots)?;
        }
        Ok(())
    }

    pub(super) fn combine(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if let Ok((params, _selection)) = self.state.decode_parts() {
            return super::decode_combine_stage(
                gpu,
                params,
                params.routed_out.unwrap_or(params.x_residual),
            );
        }
        let (params, selection) = self.state.prefill_parts()?;
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

    pub(super) fn mutation_fence(&self, _gpu: &mut Gpu) -> Result<(), DispatchError> {
        match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => super::decode_mutation_fence_stage(),
            MoeProtocol::GroupedPrefill => super::prefill_mutation_fence_stage(),
        }
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
    let mut steps = SmallVec::new();
    let op = |state: &'a MoeStepState<'a>| SealedMoeOp::new(state);
    append_step(&mut steps, Step::MoeNormalize(op(state)))?;
    let adopted = matches!(
        state.call.router_input(),
        MoeRouterInput::PrecomputedSoftmaxTopK | MoeRouterInput::PrecomputedSigmoidTopK
    );
    match params.recipe {
        MoeRecipe::SoftmaxGatedShared => {
            if selection.resolution.needs_x_rot_local {
                append_step(&mut steps, Step::MoeInputBasis(op(state)))?;
            }
            if !adopted {
                // Qwen's gate-side operation preserves the existing fused
                // router/selector/gate/up launch when its static quartet is
                // admitted; the route operation then owns top-k production.
                append_step(&mut steps, Step::MoeGateSide(op(state)))?;
                append_step(&mut steps, Step::MoeRoute(op(state)))?;
            }
            if !params.skip_shared {
                append_step(&mut steps, Step::MoeSharedDown(op(state)))?;
            }
        }
        MoeRecipe::SigmoidRoutedNoShared => {
            if !adopted {
                append_step(&mut steps, Step::MoeRouterProjection(op(state)))?;
                append_step(&mut steps, Step::MoeRoute(op(state)))?;
            }
            // Cohere's sigmoid route consumes the normalized natural input
            // first and transforms it only after top-eight selection.
            if selection.resolution.needs_x_rot_local {
                append_step(&mut steps, Step::MoeInputBasis(op(state)))?;
            }
        }
    }
    if selection.resolution.use_gpu_topk {
        append_step(&mut steps, Step::MoeGateUp(op(state)))?;
        append_step(&mut steps, Step::MoeActivation(op(state)))?;
        append_step(&mut steps, Step::MoeDown(op(state)))?;
        append_step(&mut steps, Step::MoeMutationFence(op(state)))?;
        let down_self_combines = selection.down_last_combine
            || !crate::families::moe::moe_down_writes_expanded(
                params.dtypes.routed_down,
                params.expert_dtype_tags.is_some(),
            );
        if params.ep_mode != crate::families::moe::MoeEpMode::RootRoutedPartial
            && !selection.ninepath_d4
            && !down_self_combines
            && !params.defer_routed_combine
        {
            append_step(&mut steps, Step::MoeCombine(op(state)))?;
        }
    } else {
        append_step(&mut steps, Step::MoeHostExperts(op(state)))?;
    }
    Ok(steps)
}

fn lower_prefill<'a>(
    state: &'a MoeStepState<'a>,
) -> Result<SmallVec<[Step<'a>; INLINE_STEP_CAPACITY]>, DispatchError> {
    let (params, selection) = state.prefill_parts()?;
    let mut steps = SmallVec::new();
    let op = |state: &'a MoeStepState<'a>| SealedMoeOp::new(state);
    append_step(&mut steps, Step::MoeNormalize(op(state)))?;
    let adopted = matches!(
        params.prelude.route,
        super::sealed_moe::PrefillRouteMode::AdoptRoot { .. }
    );
    match params.recipe {
        MoeRecipe::SoftmaxGatedShared => {
            append_step(&mut steps, Step::MoeInputBasis(op(state)))?;
            if !adopted {
                append_step(&mut steps, Step::MoeRouterProjection(op(state)))?;
            }
            append_step(&mut steps, Step::MoeSharedGateUp(op(state)))?;
            append_step(&mut steps, Step::MoeSharedActivation(op(state)))?;
            append_step(&mut steps, Step::MoeSharedDown(op(state)))?;
            append_step(&mut steps, Step::MoeRoute(op(state)))?;
        }
        MoeRecipe::SigmoidRoutedNoShared => {
            if !adopted {
                append_step(&mut steps, Step::MoeRouterProjection(op(state)))?;
                append_step(&mut steps, Step::MoeRoute(op(state)))?;
            }
            append_step(&mut steps, Step::MoeInputBasis(op(state)))?;
        }
    }
    if selection.resolution.use_path2 {
        append_step(&mut steps, Step::MoeScatter(op(state)))?;
    }
    append_step(&mut steps, Step::MoeGateUp(op(state)))?;
    if selection.resolution.use_path2 {
        append_step(&mut steps, Step::MoeUnscatter(op(state)))?;
    }
    append_step(&mut steps, Step::MoeActivation(op(state)))?;
    append_step(&mut steps, Step::MoeDown(op(state)))?;
    append_step(&mut steps, Step::MoeMutationFence(op(state)))?;
    let compact_ep = matches!(
        params.prelude.route,
        super::sealed_moe::PrefillRouteMode::ProduceRoot { .. }
            | super::sealed_moe::PrefillRouteMode::AdoptRoot { .. }
    );
    if !compact_ep && (selection.resolution.use_path2 || !selection.resolution.down_path0) {
        append_step(&mut steps, Step::MoeCombine(op(state)))?;
    }
    Ok(steps)
}

/// Lower and execute a sealed call using the shared Step interpreter.
pub(super) fn execute(gpu: &mut Gpu, call: &SealedMoeCall<'_>) -> Result<(), DispatchError> {
    // Reborrow the call at the execution lifetime. The state and every Step
    // borrow only this invocation and are dropped before the call returns.
    let state = MoeStepState::new(call);
    let steps = match call.protocol() {
        MoeProtocol::IndexedDecode => lower_decode(&state)?,
        MoeProtocol::GroupedPrefill => lower_prefill(&state)?,
    };
    super::steps::execute_validated_steps(gpu, call.dispatch_ctx(), &steps)
}

/// Execute only the canonical slot-order combine for a validated EP call.
pub(super) fn execute_ep_slot_combine(
    gpu: &mut Gpu,
    call: &SealedMoeCall<'_>,
) -> Result<(), DispatchError> {
    let state = MoeStepState::new(call);
    SealedMoeOp::new(&state).combine(gpu)
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
    if params.recipe == MoeRecipe::SigmoidRoutedNoShared
        && (params.k != 8
            || params.ep_mode != crate::families::moe::MoeEpMode::None
            || params.defer_routed_combine)
    {
        return Err(DispatchError::Hip(
            "sigmoid MoE decode requires k=8, no EP, and no deferred combine".into(),
        ));
    }
    let resolution = MoeResolution::resolve_arch(&params.dtypes, params.k, ctx.arch.has_wmma());
    super::check_moe_decode_supported(
        resolution.use_gpu_topk,
        params.k,
        params.n_exp,
        !params.routed_experts.is_empty(),
    )?;
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
        if !crate::families::moe::moe_down_writes_expanded(
            params.dtypes.routed_down,
            params.expert_dtype_tags.is_some(),
        ) {
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
    let down_last_combine = params.ep_mode == crate::families::moe::MoeEpMode::None
        && ctx.arch.is_gfx1100()
        && params.batch_size == 1
        && params.k == 8
        && params.expert_dtype_tags.is_none()
        && params.dtypes.routed_down == rdna_compute::DType::MQ4G256
        && *DOWN_LAST_COMBINE.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_MOE_DOWN_LAST_COMBINE").as_deref() == Ok("1")
        });

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
    let ninepath_eligible = ninepath_hfq4 || ninepath_mq3l || ninepath_mq4v2 || ninepath_mq6v2;
    Ok(MoeDecodeSelection {
        resolution,
        router_shared_fuse,
        exact_wave64_router,
        wave64_router,
        down_last_combine,
        ninepath_d3: ninepath_hfq4 && matches!(ninepath_mode, "1" | "d3" | "on"),
        ninepath_d4: ninepath_eligible && !matches!(ninepath_mode, "0" | "off" | "d3"),
        ninepath_mq3l,
        ninepath_mq4v2,
        ninepath_mq6v2,
    })
}

/// Resolve all grouped-prefill choices without a GPU or an instruction list.
pub(super) fn select_prefill(
    ctx: &DispatchCtx,
    params: &MoePrefillParams<'_>,
) -> Result<MoePrefillSelection, DispatchError> {
    let resolution = MoePrefillResolution::resolve(&params.dtypes, &ctx.arch, &ctx.flags);
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
    Ok(MoePrefillSelection {
        resolution,
        path2_m_total,
        force_mq4_grouped_fp16,
    })
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
}
