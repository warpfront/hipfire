// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Declarative Muse Glimmer decoder-layer program, for one decode row or the
//! rows of a prefill chunk or verify block.
//!
//! A layer is two sandwich sublayers over the residual stream `x`:
//!
//! ```text
//! attention: n  = rmsnorm(x, input_layernorm, eps)
//!            q, k, v, g = Wq·n, Wk·n, Wv·n, Wg·n
//!            q, k = rmsnorm per head (no weight); q *= qk_scale_factor
//!            rope(q, k) on sliding layers only (layer_rope_theta != 0)
//!            a  = attend(q, KV; window on sliding layers) ⊙ sigmoid(g)
//!            x' = x + rmsnorm(Wo·a, post_attention_layernorm, post_eps)
//! FFN:       n  = rmsnorm(x', pre_feedforward_layernorm, eps)
//!            x" = x' + rmsnorm(Wd·(silu(Wgate·n) ⊙ Wup·n), post_feedforward_layernorm, post_eps)
//! head:      logits = softcap(output_multiplier · lm_head·rmsnorm(x, final_norm, eps))
//! ```
//!
//! This module binds resident weights, KV and scratch to the shared
//! `pipeline::sandwich` operations (`SandwichAttention`, `SandwichMlp`);
//! kernels, route choice and fusion live in dispatch. One program serves both
//! shapes: [`LayerIo::route`] [`RowsRoute::Decode`] is the decode row, a
//! batched route its `rows` rows. The attention sublayer writes `x'` into
//! `io.residual` and the FFN sublayer writes `x"` back into `io.x`, so neither
//! copies the residual stream.

use crate::config::{GlimmerConfig, GlimmerLayerType};
use crate::glimmer::{GlimmerLayerWeights, GlimmerState, GlimmerWeights};
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::sandwich::{
    Activation, AttnGate, KvProjection, KvRoute, Rope, RopeKind, SandwichAttentionOp, SandwichKv,
    SandwichMlpOp, SandwichRoute, SandwichStream, ScaleOp, SoftcapOp, StepsRoute,
    WindowedQ8RowsFlash, WindowedQ8Tiled,
};
use hipfire_dispatch::pipeline::{GemvInput, RowsRoute, Step};
use hipfire_dispatch::types::RotationPlan;
use rdna_compute::GpuTensor;

/// Activations and scratch one layer program reads and writes, with the
/// position staging of its rows.
pub(crate) struct LayerIo<'a> {
    /// Residual stream; the layer's input and output.
    pub x: &'a GpuTensor,
    /// Residual stream after the attention sublayer.
    pub residual: &'a GpuTensor,
    /// Pre-norm and sublayer-output scratch.
    pub normed: &'a GpuTensor,
    /// FWHT scratch, `[rows, widest projection K]`.
    pub rotation: &'a GpuTensor,
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub attn_gate: &'a GpuTensor,
    pub attn_out: &'a GpuTensor,
    pub gate_ffn: &'a GpuTensor,
    pub up_ffn: &'a GpuTensor,
    pub ffn_hidden: &'a GpuTensor,
    pub ffn_out: &'a GpuTensor,
    /// Decode: the device position scalar.
    pub pos_buf: &'a hip_bridge::DeviceBuffer,
    pub rows: usize,
    /// The decode row at `position`, or `rows` batched rows.
    pub route: RowsRoute,
    /// Batched rows: `rows` i32 absolute positions.
    pub positions: Option<&'a GpuTensor>,
    pub position: usize,
    /// Batched rows: BF16 staging of BF16 projections.
    pub bf16: Option<&'a GpuTensor>,
    pub tiled: Option<WindowedQ8Tiled<'a>>,
    pub rows_flash: Option<WindowedQ8RowsFlash<'a>>,
}

impl<'a> LayerIo<'a> {
    /// The decode row of `state` at the position staged in `state.pos_buf`.
    pub fn decode(state: &'a GlimmerState, tiled: Option<WindowedQ8Tiled<'a>>) -> Self {
        Self {
            x: &state.x,
            residual: &state.residual,
            normed: &state.tmp,
            rotation: &state.x_rot,
            q: &state.q,
            k: &state.k,
            v: &state.v,
            attn_gate: &state.attn_gate,
            attn_out: &state.attn_out,
            gate_ffn: &state.gate_ffn,
            up_ffn: &state.up_ffn,
            ffn_hidden: &state.ffn_hidden,
            ffn_out: &state.ffn_out,
            pos_buf: &state.pos_buf,
            rows: 1,
            route: RowsRoute::Decode,
            positions: None,
            position: state.pos_host[0] as usize,
            bf16: None,
            tiled,
            rows_flash: None,
        }
    }

    /// One sublayer's stream: reads `x`, writes `out`.
    fn stream(
        &self,
        cfg: &GlimmerConfig,
        x: &'a GpuTensor,
        out: &'a GpuTensor,
    ) -> SandwichStream<'a> {
        SandwichStream {
            rows: self.rows,
            hidden: cfg.dim,
            eps: cfg.rms_norm_eps,
            post_eps: cfg.post_norm_eps,
            x,
            residual: out,
            normed: self.normed,
            out: Some(out),
            route: self.route,
            kernels: SandwichRoute::Steps(StepsRoute {
                shared_rotation: crate::forward::shared_rot_enabled(),
                fused_silu_rotate: crate::forward::fused_silu_rotate_enabled(),
            }),
            bf16: self.bf16,
        }
    }

    /// Final norm of the rows of `x`: into `normed` for the decode row, in
    /// place for batched rows.
    pub fn final_norm(&self, cfg: &GlimmerConfig, weights: &'a GlimmerWeights) -> Step<'a> {
        let out = if self.route.batched() {
            self.x
        } else {
            self.normed
        };
        Step::RmsnormAutomatic {
            x: self.x,
            norm_weight: &weights.final_norm,
            x_plain: out,
            out,
            awq_scale: None,
            k: cfg.dim,
            eps: cfg.rms_norm_eps,
            rotation: RotationPlan::None,
            rows: self.rows,
        }
    }
}

/// Both sublayers of decoder layer `layer` over `io`.
pub(crate) fn layer_steps<'a>(
    cfg: &GlimmerConfig,
    layer: usize,
    lw: &'a GlimmerLayerWeights,
    state: &'a GlimmerState,
    io: &LayerIo<'a>,
) -> [Step<'a>; 2] {
    let kv = match cfg.layer_types[layer] {
        GlimmerLayerType::Sliding => &state.kv_sliding,
        GlimmerLayerType::Full => &state.kv_full,
    };
    let slot = state.kv_slot_for_layer[layer];
    let rope = Rope {
        kind: if cfg.has_rope(layer) {
            RopeKind::RotateHalf
        } else {
            RopeKind::NoPe
        },
        theta: cfg.rope_theta_for(layer),
    };
    [
        Step::SandwichAttention(SandwichAttentionOp {
            stream: io.stream(cfg, io.x, io.residual),
            position: io.position,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            input_norm: &lw.input_layernorm,
            wq: lw.q_proj.dispatch_ref(),
            q_norm: &state.qk_norm_ones,
            kv_proj: Some(KvProjection {
                wk: lw.k_proj.dispatch_ref(),
                wv: Some(lw.v_proj.dispatch_ref()),
                k_norm: &state.qk_norm_ones,
                v_norm: None,
            }),
            q_scale: cfg.qk_scale_factor,
            rope,
            kv: SandwichKv {
                k_cache: &kv.k_gpu[slot],
                v_cache: &kv.v_gpu[slot],
                physical_cap: kv.physical_cap,
                window: cfg.window_for(layer),
                route: KvRoute::WindowedQ8 {
                    tiled: io.tiled,
                    rows_flash: io.rows_flash,
                    wmma_flash: crate::forward::wmma_flash_enabled(),
                },
            },
            pos_buf: io.pos_buf,
            positions: io.positions,
            gate: Some(AttnGate {
                w: lw.attn_gate_proj.dispatch_ref(),
                out: io.attn_gate,
            }),
            wo: lw.o_proj.dispatch_ref(),
            post_norm: Some(&lw.post_attention_layernorm),
            x_rot: io.rotation,
            q: io.q,
            k: io.k,
            v: io.v,
            attn_out: io.attn_out,
        }),
        Step::SandwichMlp(SandwichMlpOp {
            stream: io.stream(cfg, io.residual, io.x),
            pre_norm: &lw.pre_feedforward_layernorm,
            w_gate: lw.gate_proj.dispatch_ref(),
            w_up: lw.up_proj.dispatch_ref(),
            w_down: lw.down_proj.dispatch_ref(),
            activation: Activation::Silu,
            hidden_dim: cfg.hidden_dim,
            post_norm: Some(&lw.post_feedforward_layernorm),
            x_rot: io.rotation,
            gate: io.gate_ffn,
            up: io.up_ffn,
            act: io.ffn_hidden,
            out: io.ffn_out,
        }),
    ]
}

/// The untied LM head GEMV of one normed row into `logits`, then
/// [`logit_steps`].
pub(crate) fn lm_head_steps<'a>(
    cfg: &GlimmerConfig,
    lm_head: &'a WeightRef<'a>,
    normed: &'a GpuTensor,
    logits: &'a GpuTensor,
) -> Vec<Step<'a>> {
    let mut steps = vec![Step::Gemv {
        w: lm_head,
        input: GemvInput::Raw(normed),
        out: logits,
    }];
    steps.extend(logit_steps(cfg, logits, cfg.vocab_size));
    steps
}

/// The output multiplier, then the logit soft cap, over `n` logits.
pub(crate) fn logit_steps<'a>(
    cfg: &GlimmerConfig,
    logits: &'a GpuTensor,
    n: usize,
) -> Vec<Step<'a>> {
    let mut steps = Vec::new();
    if cfg.output_multiplier != 1.0 {
        steps.push(Step::Scale(ScaleOp {
            x: logits,
            factor: cfg.output_multiplier,
        }));
    }
    if cfg.final_logit_softcapping > 0.0 {
        steps.push(Step::Softcap(SoftcapOp {
            logits,
            n,
            cap: cfg.final_logit_softcapping,
        }));
    }
    steps
}
