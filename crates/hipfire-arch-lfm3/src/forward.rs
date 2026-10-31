// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3 forward pass (free functions — hot-path static dispatch).
//!
//! Per-layer pipeline (LFM3 with sandwich norm):
//!   tmp = operator_norm(h)
//!   if conv:    proj_out = out_proj( C_gate ⊙ depthwise_causal_conv( B_gate ⊙ x ) )  [in_proj→conv→out_proj]
//!   if attn:    proj_out = o_proj( sigmoid_gate(attn( qk_norm(q/k) + per-type RoPE, v )) )
//!               full_attn:    partial RoPE (n_rot = 0.5·head_dim, theta 1e6), window = 0
//!               sliding_attn: full RoPE (n_rot = head_dim, theta 1e4), window = 256
//!   if sandwich: proj_out = post_operator_norm(proj_out);  h += proj_out
//!   else:        h += proj_out
//!   ffn_tmp = ffn_norm(h)
//!   proj_out = w2( silu(w1·ffn_tmp) ⊙ (w3·ffn_tmp) )           [SwiGLU]
//!   if sandwich: proj_out = post_ffn_norm(proj_out);  h += proj_out
//!   else:        h += proj_out
//! then logits = lm_head( embedding_norm(h) )   (lm_head tied to embed_tokens).
//!
//! Sandwich norm prevents fusing the residual add into the out_proj/o_proj/w2
//! GEMV (the norm must run first), so those use plain `weight_gemv` + `add_f32`
//! instead of `weight_gemv_residual`. The sigmoid output gate uses the fused
//! `sigmoid_head_gate_mul_f32` (per-head broadcast). Sliding-window attention
//! routes through `AttnFlashQ8_0Windowed` via `KvTierPlan{q8_windowed,window}`,
//! mirroring `hipfire-arch-cohere2moe`.

use crate::config::{Lfm3Config, MixerKind};
use crate::lfm3::{AttnWeights, ConvWeights, Lfm3State, Lfm3Weights, Mixer};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::pipeline::{execute_steps, Step};
use hipfire_runtime::llama::weight_gemv;
use rdna_compute::Gpu;

/// Decode one token; returns the full logits vector.
pub fn decode_step(
    cfg: &Lfm3Config,
    weights: &Lfm3Weights,
    state: &mut Lfm3State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    decode_step_inner(cfg, weights, state, gpu, token_id, position, None)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("lfm3: download logits: {e:?}"))
}

/// Decode one token, appending each layer's post-residual hidden state to
/// `capture[layer]` — used by the oracle dumper.
pub fn decode_step_capture(
    cfg: &Lfm3Config,
    weights: &Lfm3Weights,
    state: &mut Lfm3State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture: &mut [Vec<f32>],
) -> Result<(), String> {
    decode_step_inner(cfg, weights, state, gpu, token_id, position, Some(capture))
}

fn decode_step_inner(
    cfg: &Lfm3Config,
    weights: &Lfm3Weights,
    state: &mut Lfm3State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    mut capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let eps = cfg.rms_norm_eps;
    let sandwich = cfg.use_sandwich_norm;
    let seq_len = position as usize + 1;

    // Device position scalar (i32) for rope / kv-write / attention.
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("lfm3: htod pos: {e:?}"))?;

    // Embedding lookup → residual stream h (Q8 table).
    gpu.embedding_lookup_q8(&weights.embed, &state.h, token_id, hidden)
        .map_err(|e| format!("lfm3: embed lookup: {e:?}"))?;

    for (l, layer) in weights.layers.iter().enumerate() {
        // ── Mixer block (pre-norm) ──────────────────────────────────────────
        gpu.rmsnorm_f32(&state.h, &layer.operator_norm, &state.tmp, eps)
            .map_err(|e| format!("lfm3 L{l}: operator rmsnorm: {e:?}"))?;

        match &layer.mixer {
            Mixer::Conv(c) => {
                conv_mixer(gpu, cfg, c, state, l)?;
            }
            Mixer::Attention(a) => {
                attn_mixer(gpu, cfg, a, state, l, seq_len, n_heads, n_kv, head_dim, eps)?;
            }
        }

        // ── Sandwich norm + residual (mixer) ───────────────────────────────
        apply_sandwich_residual(
            gpu,
            sandwich,
            &layer.post_operator_norm,
            &state.proj_out,
            &state.sandwich_buf,
            &state.h,
            eps,
            l,
            "operator",
        )?;

        if let Some(cap) = capture.as_deref_mut() {
            let h = gpu
                .download_f32(&state.h)
                .map_err(|e| format!("lfm3 L{l}: postmixer capture: {e:?}"))?;
            cap[l].extend_from_slice(&h);
        }

        // ── FFN block (pre-norm): dense SwiGLU ─────────────────────────────
        gpu.rmsnorm_f32(&state.h, &layer.ffn_norm, &state.ffn_tmp, eps)
            .map_err(|e| format!("lfm3 L{l}: ffn rmsnorm: {e:?}"))?;
        let d = &layer.ffn;
        weight_gemv(gpu, &d.w1, &state.ffn_tmp, &state.dense_gate)
            .map_err(|e| format!("lfm3 L{l}: dense w1: {e}"))?;
        weight_gemv(gpu, &d.w3, &state.ffn_tmp, &state.dense_up)
            .map_err(|e| format!("lfm3 L{l}: dense w3: {e}"))?;
        gpu.silu_mul_f32(&state.dense_gate, &state.dense_up, &state.dense_act)
            .map_err(|e| format!("lfm3 L{l}: dense silu_mul: {e:?}"))?;
        weight_gemv(gpu, &d.w2, &state.dense_act, &state.proj_out)
            .map_err(|e| format!("lfm3 L{l}: dense w2: {e}"))?;

        // ── Sandwich norm + residual (FFN) ─────────────────────────────────
        apply_sandwich_residual(
            gpu,
            sandwich,
            &layer.post_ffn_norm,
            &state.proj_out,
            &state.sandwich_buf,
            &state.h,
            eps,
            l,
            "ffn",
        )?;

        if let Some(cap) = capture.as_deref_mut() {
            let h = gpu
                .download_f32(&state.h)
                .map_err(|e| format!("lfm3 L{l}: capture download: {e:?}"))?;
            cap[l].extend_from_slice(&h);
        }
    }
    state.n_tokens = seq_len;

    // Final RMSNorm + lm_head (tied to embed_tokens).
    gpu.rmsnorm_f32(&state.h, &weights.embedding_norm, &state.final_norm_buf, eps)
        .map_err(|e| format!("lfm3: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("lfm3: lm_head: {e}"))?;
    Ok(())
}

/// Conv mixer block: in_proj → double-gated depthwise causal conv (K=7) →
/// out_proj into `state.proj_out` (residual handled by the caller, since
/// sandwich norm may intervene).
fn conv_mixer(
    gpu: &mut Gpu,
    cfg: &Lfm3Config,
    c: &ConvWeights,
    state: &Lfm3State,
    l: usize,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    // in_proj → [3*hidden] (B | C_gate | x).
    weight_gemv(gpu, &c.in_proj, &state.tmp, &state.conv_bcx)
        .map_err(|e| format!("lfm3 L{l}: conv in_proj: {e}"))?;
    // double-gated depthwise causal short-conv (advances conv state).
    gpu.conv1d_gated_decode_f32(
        &state.conv_bcx,
        &state.conv_states[c.conv_state_idx],
        &c.conv_weight,
        &state.conv_y,
        1,
        hidden,
        cfg.conv_kernel_size,
    )
    .map_err(|e| format!("lfm3 L{l}: conv gated decode: {e:?}"))?;
    // out_proj → proj_out (NOT residual; sandwich norm + add handled by caller).
    weight_gemv(gpu, &c.out_proj, &state.conv_y, &state.proj_out)
        .map_err(|e| format!("lfm3 L{l}: conv out_proj: {e}"))
}

/// Attention mixer block: q/k/v proj + per-head QK-norm + per-type RoPE +
/// KV write + (windowed) attention + sigmoid output gate + o_proj into
/// `state.proj_out`.
fn attn_mixer(
    gpu: &mut Gpu,
    cfg: &Lfm3Config,
    a: &AttnWeights,
    state: &Lfm3State,
    l: usize,
    seq_len: usize,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), String> {
    let kind = cfg.mixer(l);
    weight_gemv(gpu, &a.wq, &state.tmp, &state.fa_q)
        .map_err(|e| format!("lfm3 L{l}: q_proj: {e}"))?;
    weight_gemv(gpu, &a.wk, &state.tmp, &state.fa_k)
        .map_err(|e| format!("lfm3 L{l}: k_proj: {e}"))?;
    weight_gemv(gpu, &a.wv, &state.tmp, &state.fa_v)
        .map_err(|e| format!("lfm3 L{l}: v_proj: {e}"))?;

    // Per-HEAD QK-norm: RMSNorm over each head's head_dim slice.
    gpu.rmsnorm_batched(&state.fa_q, &a.q_norm, &state.fa_q, n_heads, head_dim, eps)
        .map_err(|e| format!("lfm3 L{l}: q_norm: {e:?}"))?;
    gpu.rmsnorm_batched(&state.fa_k, &a.k_norm, &state.fa_k, n_kv, head_dim, eps)
        .map_err(|e| format!("lfm3 L{l}: k_norm: {e:?}"))?;

    // Per-type RoPE: full_attention → partial 0.5, theta 1e6;
    // sliding_attention → full rotary, theta 1e4. `rope_partial_interleaved_f32`
    // dispatches to the rotate_half (halfsplit) kernel by default — the
    // mathematically-correct match for HF-converted weights (see norm.rs).
    let rope = cfg.rope_for(l);
    let n_rot = rope.n_rot(head_dim);
    gpu.rope_partial_interleaved_f32(
        &state.fa_q,
        &state.fa_k,
        &state.pos_buf,
        n_heads,
        n_kv,
        head_dim,
        n_rot,
        rope.theta,
    )
    .map_err(|e| format!("lfm3 L{l}: rope: {e:?}"))?;

    // KV cache write (Q8).
    let kv_idx = a.kv_idx;
    gpu.kv_cache_write_q8_0(&state.kv.k_gpu[kv_idx], &state.fa_k, &state.pos_buf, n_kv, head_dim)
        .map_err(|e| format!("lfm3 L{l}: kv write k: {e:?}"))?;
    gpu.kv_cache_write_q8_0(&state.kv.v_gpu[kv_idx], &state.fa_v, &state.pos_buf, n_kv, head_dim)
        .map_err(|e| format!("lfm3 L{l}: kv write v: {e:?}"))?;

    // Attention: sliding layers clip to the last `sliding_window` keys
    // (window>0); full layers are full causal (window=0). Routes through
    // AttnFlashQ8_0Windowed unconditionally (q8_windowed), mirroring cohere2moe
    // — the non-windowed AttnQ8_0Kv heuristic would drop the window at
    // ctx>window.
    let window = if kind == MixerKind::SlidingAttention {
        cfg.sliding_window as i32
    } else {
        0
    };
    let ctx = DispatchCtx::new(gpu);
    let plan = KvTierPlan::derive(KvTierInputs {
        pos: seq_len - 1,
        q8_windowed: true,
        window,
        ..state.kv.tier_inputs()
    })
    .map_err(|e| format!("lfm3 L{l}: kv tier: {e}"))?;
    let io = AttnParams {
        q: &state.fa_q,
        k: &state.fa_k,
        v: &state.fa_v,
        k_cache: &state.kv.k_gpu[kv_idx],
        v_cache: &state.kv.v_gpu[kv_idx],
        k_scales: None,
        v_scales: None,
        pos_buf: &state.pos_buf,
        pos: seq_len - 1,
        positions: None,
        n_heads,
        n_kv_heads: n_kv,
        head_dim,
        physical_cap: state.kv.physical_cap,
        batch_size: 1,
        max_ctx_len: 0,
        flash_partials: Some(&state.flash_partials),
        givens_cos: None,
        givens_sin: None,
        tree_bias: None,
        block_start: 0,
        block_cols: 0,
        // LFM3's sigmoid output gate is applied separately AFTER attention
        // (it is a plain per-head broadcast, not the Qwen fused gate+MQ-rotation
        // that `output_gate` triggers). Leave None here.
        output_gate: None,
        output: &state.fa_attn_out,
    };
    execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
        .map_err(|e| format!("lfm3 L{l}: attention: {e:?}"))?;

    // Sigmoid output gate: gate_proj(normed hidden) → sigmoid → per-head
    // broadcast multiply on attn_out. gate_proj reads `state.tmp` (the
    // operator-normed hidden, same input q/k/v took).
    weight_gemv(gpu, &a.gate_proj, &state.tmp, &state.fa_gate)
        .map_err(|e| format!("lfm3 L{l}: gate_proj: {e}"))?;
    gpu.sigmoid_head_gate_mul_f32(&state.fa_attn_out, &state.fa_gate, head_dim)
        .map_err(|e| format!("lfm3 L{l}: sigmoid head gate: {e:?}"))?;

    // o_proj → proj_out (NOT residual; sandwich norm + add handled by caller).
    weight_gemv(gpu, &a.wo, &state.fa_attn_out, &state.proj_out)
        .map_err(|e| format!("lfm3 L{l}: out_proj: {e}"))
}

/// Apply sandwich norm (if configured) to the projection output, then add it
/// to the residual stream. When sandwich norm is off, this is a plain residual
/// add (the projection output is added directly). Uses `add_f32` rather than
/// the fused `weight_gemv_residual` because the norm must run between the
/// GEMV and the add.
fn apply_sandwich_residual(
    gpu: &mut Gpu,
    sandwich: bool,
    post_norm: &Option<rdna_compute::GpuTensor>,
    proj_out: &rdna_compute::GpuTensor,
    sandwich_buf: &rdna_compute::GpuTensor,
    h: &rdna_compute::GpuTensor,
    eps: f32,
    l: usize,
    label: &str,
) -> Result<(), String> {
    if sandwich {
        let norm_w = post_norm
            .as_ref()
            .ok_or_else(|| format!("lfm3 L{l}: sandwich norm on but post_{label}_norm missing"))?;
        gpu.rmsnorm_f32(proj_out, norm_w, sandwich_buf, eps)
            .map_err(|e| format!("lfm3 L{l}: post_{label} rmsnorm: {e:?}"))?;
        gpu.add_f32(sandwich_buf, h, h)
            .map_err(|e| format!("lfm3 L{l}: post_{label} residual add: {e:?}"))
    } else {
        gpu.add_f32(proj_out, h, h)
            .map_err(|e| format!("lfm3 L{l}: {label} residual add: {e:?}"))
    }
}
