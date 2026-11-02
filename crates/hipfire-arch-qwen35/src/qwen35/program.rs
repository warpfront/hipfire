// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Declarative Qwen3.5 decoder-layer program.
//!
//! Each layer is one mixer step (gated DeltaNet or gated full attention)
//! followed by one FFN step (dense `GatedMlp` on the `Product` ladder, or
//! sealed MoE). This module only
//! binds resident weights, recurrent state, KV storage and scratch to the
//! shared `hipfire_dispatch::pipeline::hybrid` operations; route selection and
//! executor bodies live in dispatch.

use super::batch::PrefillBatchScratch;
use super::config::Qwen35Config;
use super::forward::Qwen35Scratch;
use super::weights::{
    DeltaNetLayerWeights, DeltaNetMoeLayerWeights, DeltaNetState, FullAttnLayerWeights,
    FullAttnMoeLayerWeights, LayerWeights, StateQuant,
};
use crate::speculative::GdnTape;
use hip_bridge::HipResult;
use hipfire_dispatch::families::kv_tier::KvTierInputs;
use hipfire_dispatch::pipeline::batched::{BatchScratch, DenseBatchMath, DflashFusionCtx};
use hipfire_dispatch::pipeline::batched_attention::{
    AttentionLayerView, BatchedAttentionTap, FlashScratch, KvView,
};
use hipfire_dispatch::pipeline::batched_deltanet::{
    DeltaNetLayerView, DeltaNetStateView, GdnTapeView,
};
use hipfire_dispatch::pipeline::batched_moe::{AttentionMoeLayerView, DeltaNetMoeLayerView};
use hipfire_dispatch::pipeline::gated_mlp::{
    execute_product_rows, Activation, GatedMlpOp, GatedMlpScratch, MlpLadder, MlpOutput,
    MlpStorage, ProductRows, RmsNorm,
};
use hipfire_dispatch::pipeline::hybrid::{
    AttentionKv, AttentionRow, AttentionTap, DeltaNetMixerOp, GatedAttentionOp, GdnState,
    HybridDims, MixerOutput, MropeRope,
};
use hipfire_dispatch::pipeline::Step;
use hipfire_runtime::llama::WeightTensor;
use hipfire_runtime::llama::{self, KvCacheExt};
use rdna_compute::{Gpu, GpuTensor};

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

fn gdn_state(state: &DeltaNetState, delta_layer: usize) -> GdnState<'_> {
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

/// KV storage of attention layer `layer_idx`.
pub(crate) fn attention_kv<'a>(
    kv: &'a llama::KvCache,
    s: &Qwen35Scratch,
    layer_idx: usize,
) -> AttentionKv<'a> {
    AttentionKv {
        tier: KvTierInputs {
            flash_mode: s.flash_mode as usize,
            ..kv.tier_inputs()
        },
        k_cache: &kv.k_gpu[layer_idx],
        v_cache: &kv.v_gpu[layer_idx],
        physical_cap: kv.physical_cap,
        givens_cos: kv.givens_cos.as_ref(),
        givens_sin: kv.givens_sin.as_ref(),
        compact_offset: kv.compact_offset,
    }
}

/// The scratch decode row of one attention layer at `position`.
pub(crate) fn attention_row(s: &Qwen35Scratch, position: usize) -> AttentionRow<'_> {
    AttentionRow {
        position,
        pos_buf: &s.pos_buf,
        q: &s.fa_q,
        k: &s.fa_k,
        v: &s.fa_v,
        gate: &s.fa_gate,
        flash_partials: &s.flash_partials,
        out: &s.fa_attn_out,
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
    /// produced this layer's normalized, rotated input in `x_rot`; `output`
    /// is `Residual` on one device, `Write` of a tensor-parallel partial.
    pub fn mixer(
        &self,
        layer: &'a LayerWeights,
        layer_idx: usize,
        delta_layer: usize,
        prerotated_input: bool,
        tap: Option<AttentionTap<'a>>,
        output: MixerOutput<'a>,
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
                    output,
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
                Step::GatedAttention(GatedAttentionOp {
                    dims: self.dims,
                    x: &s.x,
                    prerotated_input,
                    attn_norm,
                    wq: wq.dispatch_ref(),
                    wk: wk.dispatch_ref(),
                    wv: wv.dispatch_ref(),
                    q_norm,
                    k_norm,
                    wo: wo.dispatch_ref(),
                    kv: attention_kv(self.kv_cache, s, layer_idx),
                    row: attention_row(s, self.position),
                    plain: &s.tmp,
                    x_rot: &s.x_rot,
                    q_gate: &s.fa_q_full,
                    tap,
                    output,
                    mrope: self.mrope_section.map(|section| MropeRope {
                        positions: &s.pos_buf3,
                        section,
                    }),
                })
            }
        }
    }
}

/// The dense FFN op of `layer` on the decode scratch `s`, writing `output`
/// (`Residual` on one device, `Write` of a tensor-parallel partial); `None`
/// for MoE layers, whose sealed call the caller builds (it borrows a dispatch
/// context and MoE scratch).
pub(crate) fn dense_ffn<'a>(
    dims: &HybridDims,
    s: &'a Qwen35Scratch,
    layer: &'a LayerWeights,
    output: MlpOutput<'a>,
) -> Option<GatedMlpOp<'a>> {
    let (norm, gate, up, down) = match layer {
        LayerWeights::DeltaNet(l) => (&l.ffn_norm, &l.w_gate, &l.w_up, &l.w_down),
        LayerWeights::FullAttn(l) => (&l.ffn_norm, &l.w_gate, &l.w_up, &l.w_down),
        LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_) => return None,
    };
    let gate = gate.dispatch_ref();
    Some(GatedMlpOp {
        rows: 1,
        hidden: dims.dim,
        intermediate: gate.m,
        x: &s.x,
        pre_norm: Some(RmsNorm {
            weight: norm,
            eps: dims.norm_eps,
            normed: &s.tmp,
        }),
        gate,
        up: up.dispatch_ref(),
        down: down.dispatch_ref(),
        activation: Activation::Silu,
        storage: MlpStorage::F32,
        output,
        ladder: MlpLadder::Product,
        scratch: GatedMlpScratch {
            gate: &s.gate_ffn,
            up: &s.up,
            act: &s.ffn_hidden,
            x_rot: &s.x_rot,
        },
    })
}

// ── Batched (prefill / verify) bindings ─────────────────────────────────────

impl PrefillBatchScratch {
    /// The batched executors' view of these scratch rows.
    pub(crate) fn view(&self) -> BatchScratch<'_> {
        BatchScratch {
            lean: self.lean,
            x_batch: &self.x_batch,
            x_norm_batch: &self.x_norm_batch,
            x_rot_batch: &self.x_rot_batch,
            x_rot_f16_batch: &self.x_rot_f16_batch,
            gate_ffn_batch: &self.gate_ffn_batch,
            up_batch: &self.up_batch,
            ffn_hidden_batch: &self.ffn_hidden_batch,
            ffn_hidden_f16_batch: &self.ffn_hidden_f16_batch,
            dn_qkv_batch: &self.dn_qkv_batch,
            dn_z_batch: &self.dn_z_batch,
            dn_z_fold_batch: &self.dn_z_fold_batch,
            dn_alpha_batch: &self.dn_alpha_batch,
            dn_beta_batch: &self.dn_beta_batch,
            dn_q_raw_batch: &self.dn_q_raw_batch,
            dn_k_raw_batch: &self.dn_k_raw_batch,
            dn_v_batch: &self.dn_v_batch,
            dn_q_batch: &self.dn_q_batch,
            dn_k_batch: &self.dn_k_batch,
            dn_attn_out_batch: &self.dn_attn_out_batch,
            dn_normed_batch: &self.dn_normed_batch,
            dn_normed_rot_batch: &self.dn_normed_rot_batch,
            dn_normed_rot_f16_batch: &self.dn_normed_rot_f16_batch,
            dn_s_tape_q8: self.dn_s_tape_q8.as_ref(),
            dn_s_tape_scales: self.dn_s_tape_scales.as_ref(),
            dn_s_tape_f32: self.dn_s_tape_f32.as_ref(),
            fa_q_full_batch: &self.fa_q_full_batch,
            fa_q_batch: &self.fa_q_batch,
            fa_gate_batch: &self.fa_gate_batch,
            fa_k_batch: &self.fa_k_batch,
            fa_v_batch: &self.fa_v_batch,
            fa_attn_out_batch: &self.fa_attn_out_batch,
            fa_attn_out_rot_batch: &self.fa_attn_out_rot_batch,
            fa_attn_out_rot_f16_batch: &self.fa_attn_out_rot_f16_batch,
            positions: &self.positions,
            rope_positions: &self.rope_positions,
            mrope: None,
        }
    }
}

impl DeltaNetLayerWeights {
    pub(crate) fn mixer_view(&self) -> DeltaNetLayerView<'_> {
        DeltaNetLayerView {
            attn_norm: &self.attn_norm,
            wqkv: self.wqkv.dispatch_ref(),
            wz: self.wz.dispatch_ref(),
            w_alpha: self.w_alpha.dispatch_ref(),
            w_beta: self.w_beta.dispatch_ref(),
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            conv_weight: &self.conv_weight,
            norm_weight: &self.norm_weight,
            wo: self.wo.dispatch_ref(),
            w_gate: self.w_gate.dispatch_ref(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ffn_rows(
        &self,
        gpu: &mut Gpu,
        eps: f32,
        batch: &BatchScratch<'_>,
        n: usize,
        dim: usize,
        q8_wmma_arch: bool,
        arch_has_wmma: bool,
        output: MlpOutput<'_>,
        fusion: DflashFusionCtx,
        math: DenseBatchMath,
    ) -> HipResult<()> {
        let rows = ProductRows {
            batch: *batch,
            q8_wmma_arch,
            arch_has_wmma,
            fusion,
            math,
            kind: "LA",
        };
        let ffn = [&self.w_gate, &self.w_up, &self.w_down];
        dense_ffn_rows(gpu, &self.ffn_norm, ffn, eps, &rows, n, dim, output)
    }
}

impl FullAttnLayerWeights {
    pub(crate) fn mixer_view(&self) -> AttentionLayerView<'_> {
        AttentionLayerView {
            attn_norm: &self.attn_norm,
            wq: self.wq.dispatch_ref(),
            wk: self.wk.dispatch_ref(),
            wv: self.wv.dispatch_ref(),
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            wo: self.wo.dispatch_ref(),
            w_gate: self.w_gate.dispatch_ref(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ffn_rows(
        &self,
        gpu: &mut Gpu,
        eps: f32,
        batch: &BatchScratch<'_>,
        n: usize,
        dim: usize,
        q8_wmma_arch: bool,
        arch_has_wmma: bool,
        output: MlpOutput<'_>,
        fusion: DflashFusionCtx,
        math: DenseBatchMath,
    ) -> HipResult<()> {
        let rows = ProductRows {
            batch: *batch,
            q8_wmma_arch,
            arch_has_wmma,
            fusion,
            math,
            kind: "FA",
        };
        let ffn = [&self.w_gate, &self.w_up, &self.w_down];
        dense_ffn_rows(gpu, &self.ffn_norm, ffn, eps, &rows, n, dim, output)
    }
}

/// The dense FFN of a layer over `n` rows of `rows.batch` (prefill, verify,
/// TP prefill partial) as one `GatedMlp` `Product` op.
#[allow(clippy::too_many_arguments)]
fn dense_ffn_rows(
    gpu: &mut Gpu,
    norm: &GpuTensor,
    [gate, up, down]: [&WeightTensor; 3],
    eps: f32,
    rows: &ProductRows<'_>,
    n: usize,
    dim: usize,
    output: MlpOutput<'_>,
) -> HipResult<()> {
    let b = &rows.batch;
    let gate = gate.dispatch_ref();
    let op = GatedMlpOp {
        rows: n,
        hidden: dim,
        intermediate: gate.m,
        x: b.x_batch,
        pre_norm: Some(RmsNorm {
            weight: norm,
            eps,
            normed: b.x_rot_batch,
        }),
        gate,
        up: up.dispatch_ref(),
        down: down.dispatch_ref(),
        activation: Activation::Silu,
        storage: MlpStorage::F32,
        output,
        ladder: MlpLadder::Product,
        scratch: GatedMlpScratch {
            gate: b.gate_ffn_batch,
            up: b.up_batch,
            act: b.ffn_hidden_batch,
            x_rot: b.x_rot_batch,
        },
    };
    execute_product_rows(gpu, &op, rows)
}

impl DeltaNetMoeLayerWeights {
    pub(crate) fn mixer_view(&self) -> DeltaNetMoeLayerView<'_> {
        DeltaNetMoeLayerView {
            attn_norm: &self.attn_norm,
            wqkv: self.wqkv.dispatch_ref(),
            wz: self.wz.dispatch_ref(),
            w_alpha: self.w_alpha.dispatch_ref(),
            w_beta: self.w_beta.dispatch_ref(),
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            conv_weight: &self.conv_weight,
            norm_weight: &self.norm_weight,
            wo: self.wo.dispatch_ref(),
        }
    }
}

impl FullAttnMoeLayerWeights {
    pub(crate) fn mixer_view(&self) -> AttentionMoeLayerView<'_> {
        AttentionMoeLayerView {
            attn_norm: &self.attn_norm,
            wq: self.wq.dispatch_ref(),
            wk: self.wk.dispatch_ref(),
            wv: self.wv.dispatch_ref(),
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            wo: self.wo.dispatch_ref(),
        }
    }
}
/// The batched executors' view of the KV cache.
pub(crate) fn kv_view(kv: &llama::KvCache) -> KvView<'_> {
    KvView {
        k_gpu: &kv.k_gpu,
        v_gpu: &kv.v_gpu,
        givens_cos: &kv.givens_cos,
        givens_sin: &kv.givens_sin,
        compact_offset: kv.compact_offset,
        physical_cap: kv.physical_cap,
        quant_q8: kv.quant_q8,
        quant_fp8: kv.quant_fp8,
        v_q8: matches!(kv.v_mode, llama::VMode::Q8),
        tier: kv.tier_inputs(),
        vmm_backend: kv.uses_vmm_backend(),
    }
}

pub(crate) fn flash_scratch(s: &Qwen35Scratch) -> FlashScratch<'_> {
    FlashScratch {
        flash_partials: &s.flash_partials,
        pos_buf: &s.pos_buf,
        flash_mode: s.flash_mode,
    }
}

pub(crate) fn deltanet_state_view(state: &DeltaNetState) -> DeltaNetStateView<'_> {
    DeltaNetStateView {
        s_matrices: &state.s_matrices,
        s_scales: &state.s_scales,
        conv_states: &state.conv_states,
        s_ef_residual: &state.s_ef_residual,
        quant: state.quant,
    }
}

pub(crate) fn gdn_tape_view(tape: &GdnTape) -> GdnTapeView<'_> {
    GdnTapeView {
        max_n: tape.max_n,
        qkv_dim: tape.qkv_dim,
        qkv_bufs: &tape.qkv_bufs,
        alpha_bufs: &tape.alpha_bufs,
        beta_bufs: &tape.beta_bufs,
    }
}

/// The batched attention tap of one forward, bound to `config`. `get()`
/// is the tap exactly when `triattn::tap_enabled()`, so no caller can drop the
/// calibration or admit a fused FA prep while the tap is active.
pub(crate) struct BatchedTap<F>(F);

pub(crate) fn batched_attention_tap(
    config: &Qwen35Config,
) -> BatchedTap<impl Fn(&mut Gpu, usize, &GpuTensor, &GpuTensor, usize) -> HipResult<()> + '_> {
    BatchedTap(
        move |gpu: &mut Gpu, layer: usize, q: &GpuTensor, k: &GpuTensor, rows: usize| {
            record_batched_attention_tap(gpu, config, layer, q, k, rows)
        },
    )
}

impl<F: Fn(&mut Gpu, usize, &GpuTensor, &GpuTensor, usize) -> HipResult<()>> BatchedTap<F> {
    pub(crate) fn get(&self) -> Option<BatchedAttentionTap<'_>> {
        hipfire_runtime::triattn::tap_enabled().then_some(&self.0 as BatchedAttentionTap<'_>)
    }
}

/// TriAttention pre-RoPE tap of the batched attention prep: the GPU reduce
/// when `install_tap_gpu()` was used, else a host readback of Q (and K).
pub(crate) fn record_batched_attention_tap(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    layer_idx: usize,
    q: &GpuTensor,
    k: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    let gpu_handled = hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
        gpu,
        layer_idx,
        &q.buf,
        n,
        config.n_heads,
        config.head_dim,
    )?;
    if !gpu_handled {
        let n_q = config.n_heads * config.head_dim;
        let q_cpu = gpu.download_f32(q)?;
        if hipfire_runtime::triattn::tap_needs_k() {
            let n_k = config.n_kv_heads * config.head_dim;
            let k_cpu = gpu.download_f32(k)?;
            for b in 0..n {
                hipfire_runtime::triattn::record_prerope_qk(
                    layer_idx,
                    &q_cpu[b * n_q..(b + 1) * n_q],
                    Some(&k_cpu[b * n_k..(b + 1) * n_k]),
                );
            }
        } else {
            for b in 0..n {
                hipfire_runtime::triattn::record_prerope_q(
                    layer_idx,
                    &q_cpu[b * n_q..(b + 1) * n_q],
                );
            }
        }
    }
    Ok(())
}
