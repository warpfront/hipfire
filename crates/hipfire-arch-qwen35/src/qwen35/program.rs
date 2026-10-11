// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Declarative Qwen3.5 decoder-layer program.
//!
//! Each layer is one mixer step (gated DeltaNet or gated full attention)
//! followed by one FFN step (dense SwiGLU or sealed MoE). This module only
//! binds resident weights, recurrent state, KV storage and scratch to the
//! shared `hipfire_dispatch::pipeline::hybrid` operations; route selection and
//! executor bodies live in dispatch.

use super::config::Qwen35Config;
use super::forward::Qwen35Scratch;
use super::weights::{DeltaNetState, LayerWeights, StateQuant};
use hipfire_dispatch::pipeline::hybrid::{
    AttentionKv, AttentionTap, DeltaNetMixerOp, GatedAttentionOp, GdnState, HybridDims, MropeRope,
    SwigluFfnOp,
};
use hipfire_dispatch::pipeline::Step;
use hipfire_runtime::llama::{self, KvCacheExt};

pub(crate) fn hybrid_dims(config: &Qwen35Config) -> HybridDims {
    HybridDims {
        dim: config.dim,
        n_layers: config.n_layers,
        n_heads: config.n_heads,
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        n_rot: (config.head_dim as f32 * config.partial_rotary_factor) as usize,
        rope_theta: config.rope_theta,
        norm_eps: config.norm_eps,
        linear_key_heads: config.linear_num_key_heads,
        linear_value_heads: config.linear_num_value_heads,
        linear_key_dim: config.linear_key_head_dim,
        linear_value_dim: config.linear_value_head_dim,
        conv_kernel: config.conv_kernel_dim,
        num_experts: config.num_experts,
    }
}

pub(crate) fn gdn_state(state: &DeltaNetState, delta_layer: usize) -> GdnState<'_> {
    match state.quant {
        StateQuant::FP32 => GdnState::F32 {
            state: &state.s_matrices[delta_layer],
        },
        StateQuant::Q8 => GdnState::Q8 {
            state: &state.s_matrices[delta_layer],
            scales: &state.s_scales[delta_layer],
            error_feedback: state.ef_residual(delta_layer),
        },
        StateQuant::Q4 => GdnState::Q4 {
            state: &state.s_matrices[delta_layer],
            scales: &state.s_scales[delta_layer],
        },
    }
}

/// Per-token decode binding shared by every layer of one forward.
pub(crate) struct DecodeBinding<'a> {
    pub dims: HybridDims,
    pub s: &'a Qwen35Scratch,
    pub kv_cache: &'a llama::KvCache,
    pub dn_state: &'a DeltaNetState,
    pub position: usize,
    /// Vision-language step: (t, h, w) RoPE sections read from `s.pos_buf3`.
    pub mrope_section: Option<[usize; 3]>,
}

impl<'a> DecodeBinding<'a> {
    /// The mixer step of `layer`. `delta_layer` indexes DeltaNet state;
    /// `prerotated_input` marks that the previous layer's MoE combine already
    /// produced this layer's normalized, rotated input in `x_rot`.
    pub fn mixer(
        &self,
        layer: &'a LayerWeights,
        layer_idx: usize,
        delta_layer: usize,
        prerotated_input: bool,
        tap: Option<AttentionTap<'a>>,
    ) -> Step<'a> {
        let s = self.s;
        match layer {
            LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_) => {
                let (attn_norm, wqkv, wz, w_beta, w_alpha, dt_bias, a_log, conv, norm, wo) =
                    match layer {
                        LayerWeights::DeltaNet(l) => (
                            &l.attn_norm,
                            &l.wqkv,
                            &l.wz,
                            &l.w_beta,
                            &l.w_alpha,
                            &l.dt_bias,
                            &l.a_log,
                            &l.conv_weight,
                            &l.norm_weight,
                            &l.wo,
                        ),
                        LayerWeights::DeltaNetMoe(l) => (
                            &l.attn_norm,
                            &l.wqkv,
                            &l.wz,
                            &l.w_beta,
                            &l.w_alpha,
                            &l.dt_bias,
                            &l.a_log,
                            &l.conv_weight,
                            &l.norm_weight,
                            &l.wo,
                        ),
                        _ => unreachable!(),
                    };
                Step::DeltaNetMixer(DeltaNetMixerOp {
                    dims: self.dims,
                    rows: 1,
                    x: &s.x,
                    prerotated_input,
                    attn_norm,
                    wqkv: wqkv.dispatch_ref(),
                    wz: wz.dispatch_ref(),
                    w_beta: w_beta.dispatch_ref(),
                    w_alpha: w_alpha.dispatch_ref(),
                    dt_bias,
                    a_log,
                    conv_weight: conv,
                    norm_weight: norm,
                    wo: wo.dispatch_ref(),
                    conv_state: &self.dn_state.conv_states[delta_layer],
                    state: gdn_state(self.dn_state, delta_layer),
                    plain: &s.tmp,
                    x_rot: &s.x_rot,
                    qkv: &s.dn_qkv,
                    z: &s.dn_z,
                    beta: &s.dn_beta,
                    alpha: &s.dn_alpha,
                    q_raw: &s.dn_q_raw,
                    k_raw: &s.dn_k_raw,
                    v: &s.dn_v,
                    q: &s.dn_q,
                    k: &s.dn_k,
                    recurrent_out: &s.dn_attn_out,
                    normed: &s.dn_normed,
                })
            }
            LayerWeights::FullAttn(_) | LayerWeights::FullAttnMoe(_) => {
                let (attn_norm, wq, wk, wv, q_norm, k_norm, wo) = match layer {
                    LayerWeights::FullAttn(l) => (
                        &l.attn_norm,
                        &l.wq,
                        &l.wk,
                        &l.wv,
                        &l.q_norm,
                        &l.k_norm,
                        &l.wo,
                    ),
                    LayerWeights::FullAttnMoe(l) => (
                        &l.attn_norm,
                        &l.wq,
                        &l.wk,
                        &l.wv,
                        &l.q_norm,
                        &l.k_norm,
                        &l.wo,
                    ),
                    _ => unreachable!(),
                };
                let kv = self.kv_cache;
                Step::GatedAttention(GatedAttentionOp {
                    dims: self.dims,
                    rows: 1,
                    position: self.position,
                    x: &s.x,
                    prerotated_input,
                    attn_norm,
                    wq: wq.dispatch_ref(),
                    wk: wk.dispatch_ref(),
                    wv: wv.dispatch_ref(),
                    q_norm,
                    k_norm,
                    wo: wo.dispatch_ref(),
                    kv: AttentionKv {
                        tier: hipfire_dispatch::families::kv_tier::KvTierInputs {
                            flash_mode: s.flash_mode as usize,
                            ..kv.tier_inputs()
                        },
                        k_cache: &kv.k_gpu[layer_idx],
                        v_cache: &kv.v_gpu[layer_idx],
                        physical_cap: kv.physical_cap,
                        givens_cos: kv.givens_cos.as_ref(),
                        givens_sin: kv.givens_sin.as_ref(),
                        compact_offset: kv.compact_offset,
                    },
                    pos_buf: &s.pos_buf,
                    plain: &s.tmp,
                    x_rot: &s.x_rot,
                    q_gate: &s.fa_q_full,
                    q: &s.fa_q,
                    gate: &s.fa_gate,
                    k: &s.fa_k,
                    v: &s.fa_v,
                    flash_partials: &s.flash_partials,
                    attn_out: &s.fa_attn_out,
                    tap,
                    mrope: self.mrope_section.map(|section| MropeRope {
                        positions: &s.pos_buf3,
                        section,
                    }),
                })
            }
        }
    }

    /// The dense FFN step of `layer`; `None` for MoE layers, whose sealed call
    /// the caller builds (it borrows a dispatch context and MoE scratch).
    pub fn dense_ffn(&self, layer: &'a LayerWeights) -> Option<Step<'a>> {
        let s = self.s;
        let (norm, gate, up, down) = match layer {
            LayerWeights::DeltaNet(l) => (&l.ffn_norm, &l.w_gate, &l.w_up, &l.w_down),
            LayerWeights::FullAttn(l) => (&l.ffn_norm, &l.w_gate, &l.w_up, &l.w_down),
            LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_) => return None,
        };
        Some(Step::SwigluFfn(SwigluFfnOp {
            rows: 1,
            eps: self.dims.norm_eps,
            x: &s.x,
            norm,
            w_gate: gate.dispatch_ref(),
            w_up: up.dispatch_ref(),
            w_down: down.dispatch_ref(),
            plain: &s.tmp,
            x_rot: &s.x_rot,
            gate: &s.gate_ffn,
            up: &s.up,
            hidden: &s.ffn_hidden,
            batch: None,
        }))
    }
}
