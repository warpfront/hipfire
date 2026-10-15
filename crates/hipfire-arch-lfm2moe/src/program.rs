// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Declarative LFM2 decoder-layer program.
//!
//! Each layer is a pre-norm mixer (LIV gated short convolution or GQA
//! attention with per-head QK-norm) followed by a pre-norm FFN (dense SwiGLU
//! or the sealed sigmoid/bias-routed MoE), then the final norm and the tied
//! LM head. This module only binds resident weights, KV/convolution state and
//! scratch to typed `hipfire_dispatch` steps; route selection, fusion and the
//! executor bodies live in dispatch.

use crate::config::Lfm2MoeConfig;
use crate::lfm2moe::{Ffn, Lfm2MoeLayerWeights, Lfm2MoeState, Lfm2MoeWeights, Mixer};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::families::moe::{
    MoeDtypes, MoeEpMode, MoeNormalization, MoeParams, MoeRecipe, MoeSelectionBias,
};
use hipfire_dispatch::pipeline::sandwich::{Rope, RopeKind};
use hipfire_dispatch::pipeline::{
    seal_decode, BoundMoeExperts, GatedShortConvOp, GemvInput, Step,
};
use hipfire_dispatch::types::RotationPlan;
use hipfire_runtime::llama::KvCacheExt;
use rdna_compute::{DType, GpuTensor};

/// Resident weight refs of one layer, borrowed by its steps.
pub(crate) struct LayerRefs<'a> {
    layer: &'a Lfm2MoeLayerWeights,
    mixer: [Option<WeightRef<'a>>; 4],
    ffn: [Option<WeightRef<'a>>; 3],
}

impl<'a> LayerRefs<'a> {
    pub fn new(layer: &'a Lfm2MoeLayerWeights) -> Self {
        let mixer = match &layer.mixer {
            Mixer::Conv(c) => [
                Some(c.in_proj.dispatch_ref()),
                Some(c.out_proj.dispatch_ref()),
                None,
                None,
            ],
            Mixer::Attention(a) => [
                Some(a.wq.dispatch_ref()),
                Some(a.wk.dispatch_ref()),
                Some(a.wv.dispatch_ref()),
                Some(a.wo.dispatch_ref()),
            ],
        };
        let ffn = match &layer.ffn {
            Ffn::Dense(d) => [
                Some(d.w1.dispatch_ref()),
                Some(d.w3.dispatch_ref()),
                Some(d.w2.dispatch_ref()),
            ],
            Ffn::Moe(_) => [None, None, None],
        };
        Self { layer, mixer, ffn }
    }

    /// Mixer block: `h += mixer(operator_norm(h))`.
    pub fn mixer_steps(
        &'a self,
        cfg: &Lfm2MoeConfig,
        state: &'a Lfm2MoeState,
        position: usize,
    ) -> Result<Vec<Step<'a>>, String> {
        let r = |i: usize| self.mixer[i].as_ref().expect("bound mixer weight");
        let eps = cfg.rms_norm_eps;
        Ok(match &self.layer.mixer {
            Mixer::Conv(c) => {
                let (norm, input) = norm_into(state, &self.layer.operator_norm, eps);
                vec![
                    norm,
                    Step::Gemv {
                        w: r(0),
                        input,
                        out: &state.conv_bcx,
                    },
                    Step::GatedShortConv(GatedShortConvOp {
                        bcx: &state.conv_bcx,
                        state: &state.conv_states[c.conv_state_idx],
                        weight: &c.conv_weight,
                        output: &state.conv_y,
                        rows: 1,
                        channels: cfg.hidden_size,
                        kernel_size: cfg.conv_kernel_size,
                    }),
                    Step::GemvResidual {
                        w: r(1),
                        input: GemvInput::Raw(&state.conv_y),
                        residual: &state.h,
                        out: &state.h,
                    },
                ]
            }
            Mixer::Attention(a) => {
                let (n_heads, n_kv, head_dim) = (
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                );
                let (norm, input) = norm_into(state, &self.layer.operator_norm, eps);
                let plan = KvTierPlan::derive(KvTierInputs {
                    pos: position,
                    ..state.kv.tier_inputs()
                })
                .map_err(|e| format!("kv tier: {e}"))?;
                let kv = a.kv_idx;
                vec![
                    norm,
                    Step::Gemv {
                        w: r(0),
                        input,
                        out: &state.fa_q,
                    },
                    Step::Gemv {
                        w: r(1),
                        input,
                        out: &state.fa_k,
                    },
                    Step::Gemv {
                        w: r(2),
                        input,
                        out: &state.fa_v,
                    },
                    Step::QkNorm {
                        x: &state.fa_q,
                        weight: &a.q_norm,
                        n_groups: n_heads,
                        head_dim,
                        eps,
                    },
                    Step::QkNorm {
                        x: &state.fa_k,
                        weight: &a.k_norm,
                        n_groups: n_kv,
                        head_dim,
                        eps,
                    },
                    Step::Rope {
                        q: &state.fa_q,
                        k: &state.fa_k,
                        pos_buf: &state.pos_buf,
                        n_heads,
                        n_kv_heads: n_kv,
                        head_dim,
                        rope: Rope {
                            kind: RopeKind::RotateHalf,
                            theta: cfg.rope_theta,
                        },
                    },
                    Step::Attend {
                        plan,
                        io: AttnParams {
                            q: &state.fa_q,
                            k: &state.fa_k,
                            v: &state.fa_v,
                            k_cache: &state.kv.k_gpu[kv],
                            v_cache: &state.kv.v_gpu[kv],
                            k_scales: None,
                            v_scales: None,
                            pos_buf: &state.pos_buf,
                            pos: position,
                            positions: None,
                            n_heads,
                            n_kv_heads: n_kv,
                            head_dim,
                            physical_cap: state.kv.physical_cap,
                            batch_size: 1,
                            max_ctx_len: 0,
                            flash_partials: None,
                            givens_cos: None,
                            givens_sin: None,
                            tree_bias: None,
                            block_start: 0,
                            block_cols: 0,
                            output_gate: None,
                            output_awq_scale: None,
                            output: &state.fa_attn_out,
                        },
                    },
                    Step::GemvResidual {
                        w: r(3),
                        input: GemvInput::Raw(&state.fa_attn_out),
                        residual: &state.h,
                        out: &state.h,
                    },
                ]
            }
        })
    }

    /// FFN block. Dense: `h += w2(silu(w1·x) ⊙ w3·x)`, `x = ffn_norm(h)`.
    /// MoE: one sealed call (norm, sigmoid router with the expert selection
    /// bias, top-k routed experts, combine into `h`).
    pub fn ffn_steps(
        &'a self,
        cfg: &Lfm2MoeConfig,
        ctx: &'a DispatchCtx,
        state: &'a Lfm2MoeState,
        layer_idx: usize,
    ) -> Result<Vec<Step<'a>>, String> {
        let m = match (&self.layer.ffn, &self.ffn) {
            (Ffn::Dense(_), [Some(w1), Some(w3), Some(w2)]) => {
                let (norm, input) = norm_into(state, &self.layer.ffn_norm, cfg.rms_norm_eps);
                return Ok(vec![
                    norm,
                    Step::Gemv {
                        w: w1,
                        input,
                        out: &state.dense_gate,
                    },
                    Step::Gemv {
                        w: w3,
                        input,
                        out: &state.dense_up,
                    },
                    Step::SiluMul {
                        gate: &state.dense_gate,
                        up: &state.dense_up,
                        out: &state.dense_act,
                    },
                    Step::GemvResidual {
                        w: w2,
                        input: GemvInput::Raw(&state.dense_act),
                        residual: &state.h,
                        out: &state.h,
                    },
                ]);
            }
            (Ffn::Moe(m), _) => m,
            (Ffn::Dense(_), _) => unreachable!("dense FFN refs are bound in LayerRefs::new"),
        };
        let (gate_up, down) = (&m.experts[0].gate_up, &m.experts[0].down);
        let params = MoeParams {
            dtypes: MoeDtypes {
                router: m.router.gpu_dtype,
                shared: None,
                experts_all_gate_up_mq4: gate_up.gpu_dtype == DType::MQ4G256,
                routed_gate_up: gate_up.gpu_dtype,
                routed_down: down.gpu_dtype,
                // `MoeFfn::seal` admits only uniform-format layers.
                routed_has_mixed_experts: false,
                has_paro_shared: false,
                per_expert_gate_up: None,
                per_expert_down: None,
            },
            recipe: MoeRecipe::SigmoidRoutedNoShared,
            route_policy: None,
            normalization: MoeNormalization::RmsNorm {
                weight: &self.layer.ffn_norm,
                plain_out: &state.ffn_tmp,
                eps: cfg.rms_norm_eps,
            },
            batch_size: 1,
            hidden: cfg.hidden_size,
            mi: cfg.moe_intermediate_size,
            k: cfg.num_experts_per_tok,
            n_exp: cfg.num_experts,
            norm_topk_prob: cfg.norm_topk_prob,
            selection_bias: Some(MoeSelectionBias {
                bias: &m.expert_bias,
                route_scale: cfg.routed_scaling_factor,
            }),
            x_rot_prerotated: false,
            defer_routed_combine: false,
            cpu_deferred_experts: false,
            ep_mode: MoeEpMode::None,
            layer_idx: layer_idx as u16,
            x_norm: &state.ffn_tmp,
            x_residual: &state.h,
            routed_out: None,
            skip_shared: false,
            router: m.router.dispatch_ref(),
            shared: None,
            expert_gate_up_ptrs: &m.expert_gate_up_ptrs,
            expert_down_ptrs: &m.expert_down_ptrs,
            expert_ptrs_host: None,
            expert_down_awq_ptrs: None,
            expert_dtype_tags: None,
            routed_gate_up_k: gate_up.k,
            routed_down_m: down.m,
            routed_down_k: down.k,
            routed_experts: m,
            routed_gate_up_paro: None,
            routed_down_paro: None,
            router_logits: &state.router_logits,
            x_rot_local: &state.ffn_x_rot,
            // Host-routed fallback scratch: the biased router always routes on
            // the GPU, so these alias the GPU-route buffers and stay unread.
            gate_up_buf: &state.gate_batch,
            ffn_hidden: &state.up_batch,
            ffn_out: &state.down_expanded,
            gate_batch: &state.gate_batch,
            up_batch: &state.up_batch,
            rot_batch: &state.rot_batch,
            topk_indices: &state.topk_indices,
            topk_weights: &state.topk_weights,
            down_expanded: &state.down_expanded,
            ep_row_gather: None,
        };
        let bound = BoundMoeExperts::from_cache(&m.table, &m.cache)
            .map_err(|e| format!("bind experts: {e:?}"))?;
        let call = seal_decode(bound, ctx, params).map_err(|e| format!("seal MoE: {e:?}"))?;
        Ok(vec![Step::Moe(call)])
    }
}

/// Final norm + tied LM head into `state.logits`.
pub(crate) fn head_steps<'a>(
    cfg: &Lfm2MoeConfig,
    weights: &'a Lfm2MoeWeights,
    lm_head: &'a WeightRef<'a>,
    state: &'a Lfm2MoeState,
) -> Vec<Step<'a>> {
    let (norm, input) = norm_into(state, &weights.embedding_norm, cfg.rms_norm_eps);
    vec![
        norm,
        Step::Gemv {
            w: lm_head,
            input,
            out: &state.logits,
        },
    ]
}

/// RMS norm of `state.h` into `state.tmp`, read raw by every projection.
/// Each projection rotates its own input for its format, which keeps the
/// per-projection kernels and reduction order of the pre-port route
/// (prerotated inputs would admit the cross-projection fusion table).
fn norm_into<'a>(
    state: &'a Lfm2MoeState,
    weight: &'a GpuTensor,
    eps: f32,
) -> (Step<'a>, GemvInput<'a>) {
    (
        Step::RmsnormAutomatic {
            x: &state.h,
            norm_weight: weight,
            x_plain: &state.tmp,
            out: &state.tmp,
            awq_scale: None,
            k: state.h.numel(),
            eps,
            rotation: RotationPlan::None,
            rows: 1,
        },
        GemvInput::Raw(&state.tmp),
    )
}
