// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Declarative MiniMax-M2 decoder-layer program.
//!
//! Every layer is `h += o(attend(prefix_rope(qk_norm(qkv(attn_norm(h))))))`
//! then `h += moe(ffn_norm(h))`: GQA attention over the Q8 KV cache with a
//! flat (whole-projection) QK-norm and rotate-half RoPE on the first
//! `rotary_dim` channels of each head, followed by the sealed sigmoid MoE
//! with the expert selection bias (`e_score_correction_bias`), no shared
//! expert and no routed scaling. This module only binds resident weights, KV
//! state and scratch to typed `hipfire_dispatch` steps; route selection,
//! fusion and the executor bodies live in dispatch.

use crate::minimax::{MiniMaxConfig, MiniMaxLayerWeights, MiniMaxState, MiniMaxWeights};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::families::moe::{
    MoeDtypes, MoeEpMode, MoeNormalization, MoeParams, MoeRecipe, MoeScores, MoeSelectionBias,
};
use hipfire_dispatch::pipeline::sandwich::{Rope, RopeKind};
use hipfire_dispatch::pipeline::{seal_decode, BoundMoeExperts, GemvInput, Step};
use hipfire_dispatch::types::{dtype_rotation_plan, RotationPlan};
use hipfire_runtime::llama::KvCacheExt;
use rdna_compute::DType;

/// Resident weight refs of one layer, borrowed by its steps.
pub(crate) struct LayerRefs<'a> {
    layer: &'a MiniMaxLayerWeights,
    /// q, k, v, o.
    attn: [WeightRef<'a>; 4],
}

impl<'a> LayerRefs<'a> {
    pub fn new(layer: &'a MiniMaxLayerWeights) -> Self {
        let attn = [&layer.wq, &layer.wk, &layer.wv, &layer.wo].map(|w| w.dispatch_ref());
        Self { layer, attn }
    }

    /// Attention block: `h += o(attend(rope(qk_norm(qkv(attn_norm(h))))))`.
    /// Layer `kv_layer` of the cache holds positions `0..=position`.
    pub fn attention_steps(
        &'a self,
        cfg: &MiniMaxConfig,
        state: &'a MiniMaxState,
        kv_layer: usize,
        position: usize,
    ) -> Result<Vec<Step<'a>>, String> {
        let [wq, wk, wv, wo] = &self.attn;
        let (n_heads, n_kv, head_dim) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let eps = cfg.rms_norm_eps;
        // The prerotated input admits the fused Q8 QKV kernel.
        let input = GemvInput::Prerotated(&state.x_rot);
        let mut steps = vec![
            Step::RmsnormAutomatic {
                x: &state.h,
                norm_weight: &self.layer.attn_norm,
                x_plain: &state.tmp,
                out: &state.x_rot,
                awq_scale: self.layer.wq.awq_scale.as_ref(),
                k: wq.k,
                eps,
                rotation: dtype_rotation_plan(wq.dtype),
                rows: 1,
            },
            Step::Gemv {
                w: wq,
                input,
                out: &state.fa_q,
            },
            Step::Gemv {
                w: wk,
                input,
                out: &state.fa_k,
            },
            Step::Gemv {
                w: wv,
                input,
                out: &state.fa_v,
            },
        ];
        if cfg.use_qk_norm {
            // One RMS norm over the whole Q (and K) projection, before the
            // head split: row-complete, so a row norm, not the head-local QkNorm.
            for (x, w, k) in [
                (&state.fa_q, &self.layer.q_norm, cfg.q_dim()),
                (&state.fa_k, &self.layer.k_norm, cfg.kv_dim()),
            ] {
                steps.push(Step::RmsnormAutomatic {
                    x,
                    norm_weight: w,
                    x_plain: x,
                    out: x,
                    awq_scale: None,
                    k,
                    eps,
                    rotation: RotationPlan::None,
                    rows: 1,
                });
            }
        }
        let plan = KvTierPlan::derive(KvTierInputs {
            pos: position,
            ..state.kv.tier_inputs()
        })
        .map_err(|e| format!("kv tier: {e}"))?;
        steps.extend([
            Step::Rope {
                q: &state.fa_q,
                k: &state.fa_k,
                pos_buf: &state.pos_buf,
                positions: None,
                n_heads,
                n_kv_heads: n_kv,
                head_dim,
                rope: Rope {
                    kind: RopeKind::PrefixHalfSplit {
                        rot_dim: cfg.rotary_dim,
                    },
                    theta: cfg.rope_theta,
                },
            },
            Step::Attend {
                plan,
                io: AttnParams {
                    q: &state.fa_q,
                    k: &state.fa_k,
                    v: &state.fa_v,
                    k_cache: &state.kv.k_gpu[kv_layer],
                    v_cache: &state.kv.v_gpu[kv_layer],
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
                w: wo,
                input: GemvInput::Raw(&state.fa_attn_out),
                residual: &state.h,
                out: &state.h,
            },
        ]);
        Ok(steps)
    }

    /// MoE block: one sealed call (ffn norm, sigmoid router with the
    /// selection bias, top-k routed experts, combine into `h`).
    pub fn moe_step(
        &'a self,
        cfg: &MiniMaxConfig,
        ctx: &'a DispatchCtx,
        state: &'a MiniMaxState,
        layer_idx: usize,
        routed: &'a dyn hipfire_dispatch::families::moe::RoutedExpertWeights,
        bound: BoundMoeExperts<'a>,
    ) -> Result<Step<'a>, String> {
        let layer = self.layer;
        let (gate_up, down) = (&layer.experts[0].gate_up, &layer.experts[0].down);
        let params = MoeParams {
            dtypes: MoeDtypes {
                router: layer.router.gpu_dtype,
                shared: None,
                experts_all_gate_up_mq4: gate_up.gpu_dtype == DType::MQ4G256,
                routed_gate_up: gate_up.gpu_dtype,
                routed_down: down.gpu_dtype,
                // One packed format per projection per layer.
                routed_has_mixed_experts: false,
                has_paro_shared: false,
                per_expert_gate_up: None,
                per_expert_down: None,
            },
            mq2_lloyd: None,
            recipe: MoeRecipe::SigmoidRoutedNoShared,
            route_policy: None,
            normalization: MoeNormalization::RmsNorm {
                weight: &layer.ffn_norm,
                plain_out: &state.ffn_tmp,
                eps: cfg.rms_norm_eps,
            },
            batch_size: 1,
            hidden: cfg.hidden_size,
            mi: cfg.intermediate_size,
            k: cfg.num_experts_per_tok,
            n_exp: cfg.num_local_experts,
            // The bias-aware selector always normalizes over the selection.
            norm_topk_prob: true,
            selection_bias: Some(MoeSelectionBias {
                bias: &layer.routing_bias,
                route_scale: 1.0,
                scores: MoeScores::Sigmoid,
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
            router: layer.router.dispatch_ref(),
            shared: None,
            expert_gate_up_ptrs: &layer.expert_gate_up_ptrs,
            expert_down_ptrs: &layer.expert_down_ptrs,
            expert_ptrs_host: None,
            expert_down_awq_ptrs: None,
            expert_dtype_tags: None,
            routed_gate_up_k: gate_up.k,
            routed_down_m: down.m,
            routed_down_k: down.k,
            routed_experts: routed,
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
        let call = seal_decode(bound, ctx, params).map_err(|e| format!("seal MoE: {e:?}"))?;
        Ok(Step::Moe(call))
    }
}

/// Final norm + LM head into `state.logits`.
pub(crate) fn head_steps<'a>(
    cfg: &MiniMaxConfig,
    weights: &'a MiniMaxWeights,
    lm_head: &'a WeightRef<'a>,
    state: &'a MiniMaxState,
) -> [Step<'a>; 2] {
    [
        Step::RmsnormAutomatic {
            x: &state.h,
            norm_weight: &weights.final_norm,
            x_plain: &state.final_norm_buf,
            out: &state.final_norm_buf,
            awq_scale: None,
            k: cfg.hidden_size,
            eps: cfg.rms_norm_eps,
            // Plain norm: the Raw LM-head GEMV rotates its own input.
            rotation: RotationPlan::None,
            rows: 1,
        },
        Step::Gemv {
            w: lm_head,
            input: GemvInput::Raw(&state.final_norm_buf),
            out: &state.logits,
        },
    ]
}
