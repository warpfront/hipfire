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
use hip_bridge::{HipError, HipResult};
use hipfire_dispatch::families::kv_tier::KvTierInputs;
use hipfire_dispatch::pipeline::batched::{BatchScratch, DenseBatchMath, DflashFusionCtx};
use hipfire_dispatch::pipeline::batched_attention::{
    AttentionLayerView, BatchedAttentionTap, FlashScratch, KvView,
};
use hipfire_dispatch::pipeline::batched_deltanet::{
    DeltaNetLayerView, DeltaNetStateView, GdnCarryView, GdnStreamView, GdnTapeView,
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
use rdna_compute::gdn_stream::{
    GdnCarryView as ScanCarryView, GdnLiveState, GdnQ8Planes, GdnScanAlgorithm, GdnStreamConsts,
    GdnStreamMeta, GdnStreamPhase, GdnTensorFormat,
};
use rdna_compute::norm::GdnScanOut;
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
    let stream = &state.stream;
    DeltaNetStateView {
        s_matrices: &state.s_matrices,
        s_scales: &state.s_scales,
        conv_states: &state.conv_states,
        s_ef_residual: &state.s_ef_residual,
        quant: state.quant,
        stream: GdnStreamView {
            recipe_id: stream.recipe_id,
            generation: stream.generation,
            lanes: stream.lanes,
            meta: &stream.meta,
            carry: stream.carry.as_ref().map(|c| GdnCarryView {
                geometry: c.geometry,
                tile_state: &c.tile_state,
                tile_conv: &c.tile_conv,
                raw_qkv: &c.raw_qkv,
                raw_ab: &c.raw_ab,
            }),
            plans: &stream.plans,
            workspaces: stream.binding.as_ref().map_or(&[], |b| &b.workspaces),
        },
    }
}

/// The single genuine prefill→decode/verify fence (CHUNK-CANONICAL §3.5):
/// transition every DeltaNet layer's stream to `Sealed` at the actual prompt
/// end `position`, so decode/verify may read the Q8/EF state.
///
/// Call exactly once where an ordinary-prefill phase ends and decode/verify
/// begins; never per outer chunk, checkpoint, snapshot, VRAM downgrade, lane
/// swap, cancellation check or restore.
///
/// Streams already `Sealed` (owned by decode/verify, which does not maintain
/// the cursors) are stamped at `position` with no pending rows. An `Open`
/// layer finishes its phase through its plan's seal executor
/// (`Gpu::gdn_stream_seal` / `gdn_stream_seal_sequential`): the incomplete tile's
/// effective end is recomputed from the retained tile-entry state and raw rows
/// and quantized exactly once at `position`, the live conv ring (already at
/// `position`) untouched. Every Open cursor must equal `position` (checked before
/// any mutation); the sealed cursors of all layers are published only after
/// every layer succeeded.
pub(crate) fn seal_deltanet_streams(
    gpu: &mut Gpu,
    state: &mut DeltaNetState,
    position: usize,
) -> HipResult<()> {
    if state.stream.is_open() {
        let sealed = seal_open_layers(gpu, state, position as u64)?;
        let generation = state.stream.generation;
        for (slot, mut next) in state.stream.meta.iter_mut().zip(sealed) {
            next.owner_generation = generation;
            *slot = next;
        }
    }
    state.stream.seal_sealed_at(position as u64)
}

fn seal_open_layers(
    gpu: &mut Gpu,
    state: &DeltaNetState,
    position: u64,
) -> HipResult<Vec<GdnStreamMeta>> {
    let stream = &state.stream;
    let fail = |m: String| HipError::new(0, &format!("GDN stream seal: {m}"));
    if stream.lanes != 1 {
        return Err(fail("a lane-batched owner cannot be Open".into()));
    }
    for (l, m) in stream.meta.iter().enumerate() {
        if m.phase == GdnStreamPhase::Open && m.position != position {
            return Err(fail(format!(
                "layer {l} is Open at {} but the prompt ends at {position}",
                m.position
            )));
        }
    }
    let (binding, carry) = match (stream.binding.as_ref(), stream.carry.as_ref()) {
        (Some(b), Some(c)) if stream.plans.len() == stream.meta.len() => (b, c),
        _ => {
            return Err(fail(
                "Open stream without bound plans, constants or carry".into(),
            ))
        }
    };
    let mut out = Vec::with_capacity(stream.meta.len());
    for (l, (plan, meta)) in stream.plans.iter().zip(&stream.meta).enumerate() {
        if meta.phase != GdnStreamPhase::Open {
            out.push(*meta);
            continue;
        }
        let planes = GdnQ8Planes {
            codes: &state.s_matrices[l],
            scales: &state.s_scales[l],
            ef: state.s_ef_residual.get(l),
        };
        let next = match plan.algorithm {
            GdnScanAlgorithm::Scan64 => {
                let fmt = match plan.scan_output_format {
                    GdnTensorFormat::Bf16 => GdnScanOut::Bf16,
                    _ => GdnScanOut::F32,
                };
                let ws = binding.workspace(fmt).ok_or_else(|| {
                    fail("no edge workspace for the layer's output format".into())
                })?;
                let (hk, hv) = (ws.hk, ws.hv);
                let micro = carry.geometry.micro_rows;
                let alpha = carry.raw_ab[l].sub_offset(0, micro * hv);
                let beta = carry.raw_ab[l].sub_offset(micro * hv, micro * hv);
                let view = ScanCarryView {
                    tile_entry_f32: &carry.tile_state[l],
                    tile_entry_conv: &carry.tile_conv[l],
                    pending_qkv: &carry.raw_qkv[l],
                    pending_alpha: &alpha,
                    pending_beta: &beta,
                };
                let consts = GdnStreamConsts {
                    conv_weight: &binding.conv_weight[l],
                    dt_bias: &binding.dt_bias[l],
                    a_log: &binding.a_log[l],
                    q_scale: binding.q_scale,
                    eps: binding.eps,
                    n_key_heads: hk,
                    n_value_heads: hv,
                };
                let live = GdnLiveState {
                    planes,
                    conv: &state.conv_states[l],
                };
                gpu.gdn_stream_seal(plan, meta, &consts, &live, &view, ws)?
            }
            GdnScanAlgorithm::Sequential => gpu.gdn_stream_seal_sequential(
                plan,
                meta,
                &planes,
                &carry.tile_state[l],
                carry.geometry.hv,
            )?,
            GdnScanAlgorithm::Wy16 => return Err(fail("Wy16 plans belong to Qwen4".into())),
        };
        out.push(next);
    }
    Ok(out)
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
