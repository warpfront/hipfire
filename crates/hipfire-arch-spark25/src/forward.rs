// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Spark-X2.5 forward pass (free functions — hot-path static dispatch).
//!
//! Per-token pipeline per modeling_spark.py:
//!   x = embed(token)
//!   for each layer:
//!     residual = x
//!     n1 = rmsnorm(x, input_layernorm, 1e-6) -> tmp
//!     qkv = q_k_v_proj(n1)   // single GEMV then split Q|K|V
//!     gate_raw = g_proj(n1)  // [n_heads]
//!     RoPE: sliding full half-split; full partial half-split
//!     KV write + attention_q8_0_kv_swa(window sliding / 0 full)
//!     attn_out *= sigmoid(gate) headwise BEFORE out_proj
//!     attn_out = out_proj(attn_out); x = residual + attn_out
//!     residual = x
//!     n2 = rmsnorm(x, post_attention_layernorm)
//!     gate/up GEMV; ffn = gelu_erf(gate)*up; down; residual
//!   final rmsnorm + tied lm_head GEMV
//!
//! No sandwich post-norms, no QK-norm, no layer_scalar, no softcap.

use crate::config::Spark25Config;
use crate::spark25::{Spark25State, Spark25Weights};
use hipfire_runtime::llama::{embedding_lookup_dispatch, weight_gemv};
use rdna_compute::{Gpu, GpuTensor};

/// Decode one token; returns full logits `Vec<f32>`.
pub fn decode_step(
    cfg: &Spark25Config,
    weights: &Spark25Weights,
    state: &mut Spark25State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    if position as usize >= state.max_seq {
        return Err(format!(
            "spark25: position {position} exceeds max_seq {}",
            state.max_seq
        ));
    }
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("spark25: htod pos: {e:?}"))?;
    embed_lookup(gpu, weights, cfg.dim, token_id, &state.x)?;
    decode_step_body(cfg, weights, state, gpu, position)?;
    state.n_tokens = state.n_tokens.max(position as usize + 1);
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("spark25: download logits: {e:?}"))
}

/// Prefill: loop `decode_step` over tokens from current n_tokens; returns last-position logits.
pub fn prefill(
    cfg: &Spark25Config,
    weights: &Spark25Weights,
    state: &mut Spark25State,
    gpu: &mut Gpu,
    tokens: &[u32],
) -> Result<Vec<f32>, String> {
    let mut last = Vec::new();
    let start = state.n_tokens as u32;
    for (i, &tok) in tokens.iter().enumerate() {
        last = decode_step(cfg, weights, state, gpu, tok, start + i as u32)?;
    }
    Ok(last)
}

/// Prefill with explicit start position.
pub fn prefill_from(
    cfg: &Spark25Config,
    weights: &Spark25Weights,
    state: &mut Spark25State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
) -> Result<Vec<f32>, String> {
    let mut last = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        last = decode_step(cfg, weights, state, gpu, tok, start_pos + i as u32)?;
    }
    Ok(last)
}

/// Decode one token with per-layer hidden capture for oracle parity.
/// `capture` must be [n_layers][hidden] Vecs; each layer's post-residual
/// hidden (pre-final-norm for last layer) is downloaded after that layer.
pub fn decode_step_capture(
    cfg: &Spark25Config,
    weights: &Spark25Weights,
    state: &mut Spark25State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture: &mut [Vec<f32>],
) -> Result<Vec<f32>, String> {
    if position as usize >= state.max_seq {
        return Err(format!(
            "spark25: position {position} exceeds max_seq {}",
            state.max_seq
        ));
    }
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("spark25: htod pos: {e:?}"))?;
    embed_lookup(gpu, weights, cfg.dim, token_id, &state.x)?;
    decode_step_body_capture(cfg, weights, state, gpu, position, Some(capture))?;
    state.n_tokens = state.n_tokens.max(position as usize + 1);
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("spark25: download logits: {e:?}"))
}

/// Body after embed + pos_buf write with optional per-layer capture.
pub fn decode_step_body_capture(
    cfg: &Spark25Config,
    weights: &Spark25Weights,
    state: &mut Spark25State,
    gpu: &mut Gpu,
    position: u32,
    mut capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    if capture.is_none() {
        return decode_step_body(cfg, weights, state, gpu, position);
    }
    let dim = cfg.dim;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let n_heads = cfg.n_heads;
    let n_kv = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let rms_eps = cfg.rms_norm_eps;
    let dim_bytes = dim * 4;
    let q_bytes = q_dim * 4;
    let kv_bytes = kv_dim * 4;
    for layer_idx in 0..cfg.n_layers {
        let lw = &weights.layers[layer_idx];
        let kv_slot = state.kv_slot_for_layer[layer_idx];
        let is_sliding = cfg.is_sliding(layer_idx);
        let window = cfg.window_for(layer_idx);
        let theta = cfg.rope_theta_for(layer_idx);
        let n_rot_pairs = cfg.n_rot_pairs_for(layer_idx);
        gpu.memcpy_dtod_auto(&state.residual.buf, &state.x.buf, dim_bytes)
            .map_err(|e| format!("spark25 L{layer_idx}: save attn residual: {e:?}"))?;
        gpu.rmsnorm_f32(&state.x, &lw.input_layernorm, &state.tmp, rms_eps)
            .map_err(|e| format!("spark25 L{layer_idx}: input rmsnorm: {e:?}"))?;
        weight_gemv(gpu, &lw.q_k_v_proj, &state.tmp, &state.qkv_buf)
            .map_err(|e| format!("spark25 L{layer_idx}: q_k_v_proj: {e}"))?;
        gpu.memcpy_dtod_at_auto(&state.q.buf, 0, &state.qkv_buf.buf, 0, q_bytes)
            .map_err(|e| format!("spark25 L{layer_idx}: split q: {e:?}"))?;
        gpu.memcpy_dtod_at_auto(&state.k.buf, 0, &state.qkv_buf.buf, q_bytes, kv_bytes)
            .map_err(|e| format!("spark25 L{layer_idx}: split k: {e:?}"))?;
        gpu.memcpy_dtod_at_auto(
            &state.v.buf,
            0,
            &state.qkv_buf.buf,
            q_bytes + kv_bytes,
            kv_bytes,
        )
        .map_err(|e| format!("spark25 L{layer_idx}: split v: {e:?}"))?;
        if cfg.headwise_attn_output_gate {
            weight_gemv(gpu, &lw.g_proj, &state.tmp, &state.attn_gate)
                .map_err(|e| format!("spark25 L{layer_idx}: g_proj: {e}"))?;
        }
        if n_rot_pairs == 0 {
        } else if n_rot_pairs * 2 >= head_dim {
            gpu.rope_f32(
                &state.q,
                &state.k,
                &state.pos_buf,
                n_heads,
                n_kv,
                head_dim,
                theta,
            )
            .map_err(|e| format!("spark25 L{layer_idx}: rope_f32: {e:?}"))?;
        } else {
            gpu.rope_partial_halved_f32(
                &state.q,
                &state.k,
                &state.pos_buf,
                n_heads,
                n_kv,
                head_dim,
                n_rot_pairs,
                theta,
            )
            .map_err(|e| format!("spark25 L{layer_idx}: rope_partial: {e:?}"))?;
        }
        {
            let kv = if is_sliding {
                &mut state.kv_sliding
            } else {
                &mut state.kv_full
            };
            kv.ensure_mapped_capacity(gpu, position as usize + 1)
                .map_err(|e| format!("spark25 L{layer_idx}: kv map: {e:?}"))?;
            gpu.kv_cache_write_q8_0(&kv.k_gpu[kv_slot], &state.k, &state.pos_buf, n_kv, head_dim)
                .map_err(|e| format!("spark25 L{layer_idx}: kv write k: {e:?}"))?;
            gpu.kv_cache_write_q8_0(&kv.v_gpu[kv_slot], &state.v, &state.pos_buf, n_kv, head_dim)
                .map_err(|e| format!("spark25 L{layer_idx}: kv write v: {e:?}"))?;
            // Kernel uses absolute indices (`scores[t]`, workspace/q_shared offsets by
            // `seq_len = pos+1`), so the LDS hint must be the actual length — never the
            // SWA window cap. The window only bounds the `[t_lo, seq_len)` loop range.
            let seq_hint = position as usize + 1;
            gpu.attention_q8_0_kv_swa(
                &state.q,
                &kv.k_gpu[kv_slot],
                &kv.v_gpu[kv_slot],
                &state.attn_out,
                &state.pos_buf,
                seq_hint,
                n_heads,
                n_kv,
                head_dim,
                kv.physical_cap,
                window,
            )
            .map_err(|e| format!("spark25 L{layer_idx}: attn swa: {e:?}"))?;
        }
        if cfg.headwise_attn_output_gate {
            gpu.sigmoid_mul_broadcast_f32(&state.attn_out, &state.attn_gate, head_dim)
                .map_err(|e| format!("spark25 L{layer_idx}: sigmoid gate: {e:?}"))?;
        }
        weight_gemv(gpu, &lw.out_proj, &state.attn_out, &state.tmp)
            .map_err(|e| format!("spark25 L{layer_idx}: out_proj: {e}"))?;
        gpu.memcpy_dtod_auto(&state.x.buf, &state.residual.buf, dim_bytes)
            .map_err(|e| format!("spark25 L{layer_idx}: restore residual: {e:?}"))?;
        gpu.add_inplace_f32(&state.x, &state.tmp)
            .map_err(|e| format!("spark25 L{layer_idx}: attn residual add: {e:?}"))?;
        gpu.memcpy_dtod_auto(&state.residual.buf, &state.x.buf, dim_bytes)
            .map_err(|e| format!("spark25 L{layer_idx}: save ffn residual: {e:?}"))?;
        gpu.rmsnorm_f32(&state.x, &lw.post_attention_layernorm, &state.tmp, rms_eps)
            .map_err(|e| format!("spark25 L{layer_idx}: post_attn rmsnorm: {e:?}"))?;
        weight_gemv(gpu, &lw.gate_proj, &state.tmp, &state.gate_ffn)
            .map_err(|e| format!("spark25 L{layer_idx}: gate_proj: {e}"))?;
        weight_gemv(gpu, &lw.up_proj, &state.tmp, &state.up_ffn)
            .map_err(|e| format!("spark25 L{layer_idx}: up_proj: {e}"))?;
        gpu.gelu_erf_mul_f32(&state.gate_ffn, &state.up_ffn, &state.ffn_hidden)
            .map_err(|e| format!("spark25 L{layer_idx}: gelu_erf_mul: {e:?}"))?;
        weight_gemv(gpu, &lw.down_proj, &state.ffn_hidden, &state.tmp)
            .map_err(|e| format!("spark25 L{layer_idx}: down_proj: {e}"))?;
        gpu.memcpy_dtod_auto(&state.x.buf, &state.residual.buf, dim_bytes)
            .map_err(|e| format!("spark25 L{layer_idx}: restore ffn residual: {e:?}"))?;
        gpu.add_inplace_f32(&state.x, &state.tmp)
            .map_err(|e| format!("spark25 L{layer_idx}: ffn residual add: {e:?}"))?;
        if let Some(ref mut cap) = capture {
            let host = gpu
                .download_f32(&state.x)
                .map_err(|e| format!("spark25 L{layer_idx}: capture download: {e:?}"))?;
            cap[layer_idx].extend_from_slice(&host);
        }
    }
    gpu.rmsnorm_f32(&state.x, &weights.final_norm, &state.tmp, rms_eps)
        .map_err(|e| format!("spark25: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.tmp, &state.logits)
        .map_err(|e| format!("spark25: lm_head: {e}"))?;
    Ok(())
}

/// Body after embed + pos_buf write. Leaves logits in `state.logits`.
pub fn decode_step_body(
    cfg: &Spark25Config,
    weights: &Spark25Weights,
    state: &mut Spark25State,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    let dim = cfg.dim;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let n_heads = cfg.n_heads;
    let n_kv = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let rms_eps = cfg.rms_norm_eps;
    let dim_bytes = dim * 4;
    let q_bytes = q_dim * 4;
    let kv_bytes = kv_dim * 4;

    for layer_idx in 0..cfg.n_layers {
        let lw = &weights.layers[layer_idx];
        let kv_slot = state.kv_slot_for_layer[layer_idx];
        let is_sliding = cfg.is_sliding(layer_idx);
        let window = cfg.window_for(layer_idx);
        let theta = cfg.rope_theta_for(layer_idx);
        let n_rot_pairs = cfg.n_rot_pairs_for(layer_idx);

        // residual = x
        gpu.memcpy_dtod_auto(&state.residual.buf, &state.x.buf, dim_bytes)
            .map_err(|e| format!("spark25 L{layer_idx}: save attn residual: {e:?}"))?;

        // n1 = rmsnorm(x)
        gpu.rmsnorm_f32(&state.x, &lw.input_layernorm, &state.tmp, rms_eps)
            .map_err(|e| format!("spark25 L{layer_idx}: input rmsnorm: {e:?}"))?;

        // Fused QKV GEMV → qkv_buf, then split into q/k/v.
        weight_gemv(gpu, &lw.q_k_v_proj, &state.tmp, &state.qkv_buf)
            .map_err(|e| format!("spark25 L{layer_idx}: q_k_v_proj: {e}"))?;
        // Row-major Q|K|V: q[:q_dim], k[q_dim:q_dim+kv], v[q_dim+kv:].
        gpu.memcpy_dtod_at_auto(&state.q.buf, 0, &state.qkv_buf.buf, 0, q_bytes)
            .map_err(|e| format!("spark25 L{layer_idx}: split q: {e:?}"))?;
        gpu.memcpy_dtod_at_auto(&state.k.buf, 0, &state.qkv_buf.buf, q_bytes, kv_bytes)
            .map_err(|e| format!("spark25 L{layer_idx}: split k: {e:?}"))?;
        gpu.memcpy_dtod_at_auto(
            &state.v.buf,
            0,
            &state.qkv_buf.buf,
            q_bytes + kv_bytes,
            kv_bytes,
        )
        .map_err(|e| format!("spark25 L{layer_idx}: split v: {e:?}"))?;

        // Headwise gate: g_proj(n1) → [n_heads]
        if cfg.headwise_attn_output_gate {
            weight_gemv(gpu, &lw.g_proj, &state.tmp, &state.attn_gate)
                .map_err(|e| format!("spark25 L{layer_idx}: g_proj: {e}"))?;
        }

        // RoPE: sliding full half-split; full partial half-split.
        // No QK-norm on Spark.
        if n_rot_pairs == 0 {
            // degenerate — skip
        } else if n_rot_pairs * 2 >= head_dim {
            gpu.rope_f32(
                &state.q,
                &state.k,
                &state.pos_buf,
                n_heads,
                n_kv,
                head_dim,
                theta,
            )
            .map_err(|e| format!("spark25 L{layer_idx}: rope_f32: {e:?}"))?;
        } else {
            gpu.rope_partial_halved_f32(
                &state.q,
                &state.k,
                &state.pos_buf,
                n_heads,
                n_kv,
                head_dim,
                n_rot_pairs,
                theta,
            )
            .map_err(|e| format!("spark25 L{layer_idx}: rope_partial: {e:?}"))?;
        }

        // KV write + SWA/full attention.
        {
            let kv = if is_sliding {
                &mut state.kv_sliding
            } else {
                &mut state.kv_full
            };
            kv.ensure_mapped_capacity(gpu, position as usize + 1)
                .map_err(|e| format!("spark25 L{layer_idx}: kv map: {e:?}"))?;
            if kv_slot >= kv.k_gpu.len() {
                return Err(format!(
                    "spark25 L{layer_idx}: kv_slot {kv_slot} out of range (len={})",
                    kv.k_gpu.len()
                ));
            }
            gpu.kv_cache_write_q8_0(&kv.k_gpu[kv_slot], &state.k, &state.pos_buf, n_kv, head_dim)
                .map_err(|e| format!("spark25 L{layer_idx}: kv write k: {e:?}"))?;
            gpu.kv_cache_write_q8_0(&kv.v_gpu[kv_slot], &state.v, &state.pos_buf, n_kv, head_dim)
                .map_err(|e| format!("spark25 L{layer_idx}: kv write v: {e:?}"))?;

            // Same absolute-index contract as above: hint = actual pos+1 for all layers.
            let seq_hint = position as usize + 1;
            gpu.attention_q8_0_kv_swa(
                &state.q,
                &kv.k_gpu[kv_slot],
                &kv.v_gpu[kv_slot],
                &state.attn_out,
                &state.pos_buf,
                seq_hint,
                n_heads,
                n_kv,
                head_dim,
                kv.physical_cap,
                window,
            )
            .map_err(|e| format!("spark25 L{layer_idx}: attn swa: {e:?}"))?;
        }

        // Headwise sigmoid gate BEFORE out_proj: attn_out *= sigmoid(gate)
        if cfg.headwise_attn_output_gate {
            gpu.sigmoid_mul_broadcast_f32(&state.attn_out, &state.attn_gate, head_dim)
                .map_err(|e| format!("spark25 L{layer_idx}: sigmoid gate: {e:?}"))?;
        }

        // out_proj → tmp; x = residual + tmp
        weight_gemv(gpu, &lw.out_proj, &state.attn_out, &state.tmp)
            .map_err(|e| format!("spark25 L{layer_idx}: out_proj: {e}"))?;
        gpu.memcpy_dtod_auto(&state.x.buf, &state.residual.buf, dim_bytes)
            .map_err(|e| format!("spark25 L{layer_idx}: restore residual: {e:?}"))?;
        gpu.add_inplace_f32(&state.x, &state.tmp)
            .map_err(|e| format!("spark25 L{layer_idx}: attn residual add: {e:?}"))?;

        // FFN residual
        gpu.memcpy_dtod_auto(&state.residual.buf, &state.x.buf, dim_bytes)
            .map_err(|e| format!("spark25 L{layer_idx}: save ffn residual: {e:?}"))?;

        // pre-FFN RMSNorm
        gpu.rmsnorm_f32(&state.x, &lw.post_attention_layernorm, &state.tmp, rms_eps)
            .map_err(|e| format!("spark25 L{layer_idx}: post_attn rmsnorm: {e:?}"))?;

        weight_gemv(gpu, &lw.gate_proj, &state.tmp, &state.gate_ffn)
            .map_err(|e| format!("spark25 L{layer_idx}: gate_proj: {e}"))?;
        weight_gemv(gpu, &lw.up_proj, &state.tmp, &state.up_ffn)
            .map_err(|e| format!("spark25 L{layer_idx}: up_proj: {e}"))?;

        // exact-erf GELU(gate) * up  (NOT tanh, NOT silu)
        gpu.gelu_erf_mul_f32(&state.gate_ffn, &state.up_ffn, &state.ffn_hidden)
            .map_err(|e| format!("spark25 L{layer_idx}: gelu_erf_mul: {e:?}"))?;

        weight_gemv(gpu, &lw.down_proj, &state.ffn_hidden, &state.tmp)
            .map_err(|e| format!("spark25 L{layer_idx}: down_proj: {e}"))?;
        gpu.memcpy_dtod_auto(&state.x.buf, &state.residual.buf, dim_bytes)
            .map_err(|e| format!("spark25 L{layer_idx}: restore ffn residual: {e:?}"))?;
        gpu.add_inplace_f32(&state.x, &state.tmp)
            .map_err(|e| format!("spark25 L{layer_idx}: ffn residual add: {e:?}"))?;
    }

    // Final norm + tied lm_head
    gpu.rmsnorm_f32(&state.x, &weights.final_norm, &state.tmp, rms_eps)
        .map_err(|e| format!("spark25: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.tmp, &state.logits)
        .map_err(|e| format!("spark25: lm_head: {e}"))?;
    Ok(())
}

fn embed_lookup(
    gpu: &mut Gpu,
    weights: &Spark25Weights,
    hidden: usize,
    token_id: u32,
    out: &GpuTensor,
) -> Result<(), String> {
    embedding_lookup_dispatch(
        gpu,
        weights.embed_format,
        &weights.embed_tokens,
        out,
        token_id,
        hidden,
    )
    .map_err(|e| format!("spark25: embed lookup: {e:?}"))
}
