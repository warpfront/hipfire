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
//!     RoPE: sliding full half-split (rope_f32, theta 1e4); full contiguous-block
//!     partial half-split (first 64 dims, pairs (i,i+32), denominator 64, theta
//!     5e6) per modeling_spark.py compute_rope_cos_sin/apply_rotary_pos_emb.
//!     MUST NOT use rope_partial_halved_f32 here: that is Gemma-4 proportional
//!     RoPE (pairs (i,i+head_dim/2), denominator head_dim).
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
use hipfire_runtime::llama::{embedding_lookup_dispatch, weight_gemm, weight_gemv};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Default chunk width for gfx1010 batched prefill (~14 MB scratch @ 4B).
pub const SPARK_PREFILL_CHUNK: usize = 64;

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
/// On gfx1010 routes through chunked `weight_gemm` prefill; other arches keep the serial loop.
pub fn prefill(
    cfg: &Spark25Config,
    weights: &Spark25Weights,
    state: &mut Spark25State,
    gpu: &mut Gpu,
    tokens: &[u32],
) -> Result<Vec<f32>, String> {
    if tokens.is_empty() {
        return Ok(Vec::new());
    }
    let start = state.n_tokens as u32;
    if gpu.arch == "gfx1010" {
        return prefill_gfx1010(cfg, weights, state, gpu, tokens, start, None, &|| false);
    }
    let mut last = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        last = decode_step(cfg, weights, state, gpu, tok, start + i as u32)?;
    }
    Ok(last)
}

/// Prefill with explicit start position.
/// On gfx1010 routes through chunked `weight_gemm` prefill; other arches keep the serial loop.
pub fn prefill_from(
    cfg: &Spark25Config,
    weights: &Spark25Weights,
    state: &mut Spark25State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
) -> Result<Vec<f32>, String> {
    if tokens.is_empty() {
        return Ok(Vec::new());
    }
    if gpu.arch == "gfx1010" {
        return prefill_gfx1010(cfg, weights, state, gpu, tokens, start_pos, None, &|| false);
    }
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
    capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    if capture.is_none() {
        return decode_step_body(cfg, weights, state, gpu, position);
    }
    decode_step_layers_and_head::<false>(cfg, weights, state, gpu, position, capture)
}

/// Body after embed + pos_buf write. Leaves logits in `state.logits`.
pub fn decode_step_body(
    cfg: &Spark25Config,
    weights: &Spark25Weights,
    state: &mut Spark25State,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    decode_step_layers_and_head::<false>(cfg, weights, state, gpu, position, None)
}

/// Bounds-check + device `pos_buf` H2D + embedding lookup. Staging only —
/// outside any retained-replay capture region (mirrors LFM).
#[doc(hidden)]
pub fn prepare_retained_decode_inputs(
    cfg: &Spark25Config,
    weights: &Spark25Weights,
    state: &mut Spark25State,
    gpu: &mut Gpu,
    token: u32,
    position: u32,
) -> Result<(), String> {
    if (position as usize) >= state.max_seq {
        return Err(format!(
            "spark25: position {position} exceeds max_seq {}",
            state.max_seq
        ));
    }
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("spark25: htod pos: {e:?}"))?;
    embed_lookup(gpu, weights, cfg.dim, token, &state.x)?;
    Ok(())
}

/// Captureable retained body: layer stack + final norm + lm_head.
///
/// Reads already-staged `state.x` + `state.pos_buf`; writes `state.logits`.
/// Issues only replay-visible `copy_f32_buffer` D2D (no raw memcpy), provisions
/// attention LDS at `state.max_seq` (not `position+1`), and advances no host
/// counters — the caller commits `n_tokens` / `seq_pos` after execution.
#[doc(hidden)]
pub fn run_retained_decode_body(
    cfg: &Spark25Config,
    weights: &Spark25Weights,
    state: &mut Spark25State,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    decode_step_layers_and_head::<true>(cfg, weights, state, gpu, position, None)
}

/// Shared layer stack + head. `RETAINED=false` keeps ordinary raw D2D +
/// `seq_hint = position+1`. `RETAINED=true` uses typed `copy_f32_buffer` +
/// `sub_offset` views for every residual/QKV copy and LDS `seq_hint =
/// state.max_seq` so a tape captured at N replays at N+k without enlarging
/// shared memory. Never H2D / logits download; never advances `n_tokens`
/// (ordinary callers still commit via `decode_step`).
fn decode_step_layers_and_head<const RETAINED: bool>(
    cfg: &Spark25Config,
    weights: &Spark25Weights,
    state: &mut Spark25State,
    gpu: &mut Gpu,
    position: u32,
    mut capture: Option<&mut [Vec<f32>]>,
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
    // Parent LDS override: retained body provisions the validated device cap so
    // a capture at context C safely replays C+k. Ordinary path keeps pos+1.
    let seq_hint = if RETAINED {
        state.max_seq
    } else {
        position as usize + 1
    };

    for layer_idx in 0..cfg.n_layers {
        let lw = &weights.layers[layer_idx];
        let kv_slot = state.kv_slot_for_layer[layer_idx];
        let is_sliding = cfg.is_sliding(layer_idx);
        let window = cfg.window_for(layer_idx);
        let theta = cfg.rope_theta_for(layer_idx);
        let n_rot_pairs = cfg.n_rot_pairs_for(layer_idx);

        // residual = x
        copy_hidden::<RETAINED>(gpu, &state.residual, &state.x, dim, dim_bytes)
            .map_err(|e| format!("spark25 L{layer_idx}: save attn residual: {e}"))?;

        // n1 = rmsnorm(x)
        gpu.rmsnorm_f32(&state.x, &lw.input_layernorm, &state.tmp, rms_eps)
            .map_err(|e| format!("spark25 L{layer_idx}: input rmsnorm: {e:?}"))?;

        // Fused QKV GEMV → qkv_buf, then split into q/k/v.
        weight_gemv(gpu, &lw.q_k_v_proj, &state.tmp, &state.qkv_buf)
            .map_err(|e| format!("spark25 L{layer_idx}: q_k_v_proj: {e}"))?;
        // Row-major Q|K|V: q[:q_dim], k[q_dim:q_dim+kv], v[q_dim+kv:].
        split_qkv::<RETAINED>(gpu, state, q_dim, kv_dim, q_bytes, kv_bytes)
            .map_err(|e| format!("spark25 L{layer_idx}: {e}"))?;

        // Headwise gate: g_proj(n1) → [n_heads]
        if cfg.headwise_attn_output_gate {
            weight_gemv(gpu, &lw.g_proj, &state.tmp, &state.attn_gate)
                .map_err(|e| format!("spark25 L{layer_idx}: g_proj: {e}"))?;
        }

        // RoPE: sliding full half-split (rope_f32); full contiguous-block partial
        // half-split (first n_rot dims, denominator n_rot). No QK-norm on Spark.
        // Device-pos-buffer paths only — pos is refreshed outside capture.
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
            // Contiguous-block partial RoPE: first n_rot dims rotate as pairs
            // (i, i+n_rot/2) with inv_freq denominator n_rot (full: 64 dims,
            // theta 5e6). rope_partial_interleaved_f32 dispatches the halfsplit
            // kernel by default, which is exactly this HF convention.
            let n_rot = n_rot_pairs * 2;
            gpu.rope_partial_interleaved_f32(
                &state.q,
                &state.k,
                &state.pos_buf,
                n_heads,
                n_kv,
                head_dim,
                n_rot,
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
            // Map enough for the LDS/seq envelope this body provisions.
            kv.ensure_mapped_capacity(gpu, seq_hint)
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

            // Ordinary: absolute-index contract hint = actual pos+1.
            // Retained: fixed device-cap LDS envelope (state.max_seq).
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
        copy_hidden::<RETAINED>(gpu, &state.x, &state.residual, dim, dim_bytes)
            .map_err(|e| format!("spark25 L{layer_idx}: restore residual: {e}"))?;
        gpu.add_inplace_f32(&state.x, &state.tmp)
            .map_err(|e| format!("spark25 L{layer_idx}: attn residual add: {e:?}"))?;

        // FFN residual
        copy_hidden::<RETAINED>(gpu, &state.residual, &state.x, dim, dim_bytes)
            .map_err(|e| format!("spark25 L{layer_idx}: save ffn residual: {e}"))?;

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
        copy_hidden::<RETAINED>(gpu, &state.x, &state.residual, dim, dim_bytes)
            .map_err(|e| format!("spark25 L{layer_idx}: restore ffn residual: {e}"))?;
        gpu.add_inplace_f32(&state.x, &state.tmp)
            .map_err(|e| format!("spark25 L{layer_idx}: ffn residual add: {e:?}"))?;

        if let Some(cap) = &mut capture {
            let host = gpu
                .download_f32(&state.x)
                .map_err(|e| format!("spark25 L{layer_idx}: capture download: {e:?}"))?;
            cap[layer_idx].extend_from_slice(&host);
        }
    }

    // Final norm + tied lm_head
    gpu.rmsnorm_f32(&state.x, &weights.final_norm, &state.tmp, rms_eps)
        .map_err(|e| format!("spark25: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.tmp, &state.logits)
        .map_err(|e| format!("spark25: lm_head: {e}"))?;
    Ok(())
}

#[inline]
fn copy_hidden<const RETAINED: bool>(
    gpu: &mut Gpu,
    dst: &GpuTensor,
    src: &GpuTensor,
    n_elems: usize,
    n_bytes: usize,
) -> Result<(), String> {
    if RETAINED {
        gpu.copy_f32_buffer(dst, src, n_elems)
            .map_err(|e| format!("{e:?}"))
    } else {
        gpu.memcpy_dtod_auto(&dst.buf, &src.buf, n_bytes)
            .map_err(|e| format!("{e:?}"))
    }
}

#[inline]
fn split_qkv<const RETAINED: bool>(
    gpu: &mut Gpu,
    state: &Spark25State,
    q_dim: usize,
    kv_dim: usize,
    q_bytes: usize,
    kv_bytes: usize,
) -> Result<(), String> {
    if RETAINED {
        // Non-owning views into fused qkv_buf; typed copy is recorder-visible.
        let q_src = state.qkv_buf.sub_offset(0, q_dim);
        let k_src = state.qkv_buf.sub_offset(q_dim, kv_dim);
        let v_src = state.qkv_buf.sub_offset(q_dim + kv_dim, kv_dim);
        gpu.copy_f32_buffer(&state.q, &q_src, q_dim)
            .map_err(|e| format!("split q: {e:?}"))?;
        gpu.copy_f32_buffer(&state.k, &k_src, kv_dim)
            .map_err(|e| format!("split k: {e:?}"))?;
        gpu.copy_f32_buffer(&state.v, &v_src, kv_dim)
            .map_err(|e| format!("split v: {e:?}"))?;
    } else {
        gpu.memcpy_dtod_at_auto(&state.q.buf, 0, &state.qkv_buf.buf, 0, q_bytes)
            .map_err(|e| format!("split q: {e:?}"))?;
        gpu.memcpy_dtod_at_auto(&state.k.buf, 0, &state.qkv_buf.buf, q_bytes, kv_bytes)
            .map_err(|e| format!("split k: {e:?}"))?;
        gpu.memcpy_dtod_at_auto(
            &state.v.buf,
            0,
            &state.qkv_buf.buf,
            q_bytes + kv_bytes,
            kv_bytes,
        )
        .map_err(|e| format!("split v: {e:?}"))?;
    }
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

// ───────────────────────── gfx1010 chunked prefill ─────────────────────────

/// Reusable F32 scratch for one Spark prefill request (chunk rows × widths).
///
/// Allocate once per call, reuse across chunks and layers, free on every exit
/// path (including partial allocation failure). Fields are private.
pub struct SparkPrefillScratch {
    x_batch: GpuTensor,
    n1: GpuTensor,
    qkv_batch: GpuTensor,
    q_batch: GpuTensor,
    k_batch: GpuTensor,
    v_batch: GpuTensor,
    attn_out_batch: GpuTensor,
    gate_batch: GpuTensor,
    tmp_batch: GpuTensor,
    gate_ffn_batch: GpuTensor,
    up_ffn_batch: GpuTensor,
    ffn_out_batch: GpuTensor,
    /// F32 storage holding i32 position bits (same cosmetic dtype as llama PBS).
    positions: GpuTensor,
    chunk: usize,
}

impl SparkPrefillScratch {
    /// Allocate `[chunk × width]` F32 rows for the published Spark geometry.
    /// Frees any partial allocations on failure.
    pub fn new(gpu: &mut Gpu, cfg: &Spark25Config, chunk: usize) -> Result<Self, String> {
        if chunk == 0 {
            return Err("spark25: SparkPrefillScratch chunk must be > 0".into());
        }
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let qkv_rows = cfg.qkv_rows();
        let n_heads = cfg.n_heads;
        if dim == 0 || hidden == 0 || q_dim == 0 || kv_dim == 0 || n_heads == 0 {
            return Err("spark25: SparkPrefillScratch: invalid config geometry".into());
        }
        spark_prefill_scratch_alloc(gpu, chunk, dim, hidden, q_dim, kv_dim, qkv_rows, n_heads)
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let SparkPrefillScratch {
            x_batch,
            n1,
            qkv_batch,
            q_batch,
            k_batch,
            v_batch,
            attn_out_batch,
            gate_batch,
            tmp_batch,
            gate_ffn_batch,
            up_ffn_batch,
            ffn_out_batch,
            positions,
            chunk: _,
        } = self;
        for t in [
            x_batch,
            n1,
            qkv_batch,
            q_batch,
            k_batch,
            v_batch,
            attn_out_batch,
            gate_batch,
            tmp_batch,
            gate_ffn_batch,
            up_ffn_batch,
            ffn_out_batch,
            positions,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }

    /// Bytes of device F32 scratch at this chunk width (excludes KV/weights).
    pub fn memory_bytes(cfg: &Spark25Config, chunk: usize) -> usize {
        let floats = chunk.saturating_mul(
            cfg.dim
                + cfg.dim
                + cfg.qkv_rows()
                + cfg.q_dim()
                + cfg.kv_dim()
                + cfg.kv_dim()
                + cfg.q_dim()
                + cfg.n_heads
                + cfg.dim
                + cfg.hidden_dim
                + cfg.hidden_dim
                + cfg.hidden_dim
                + 1,
        );
        floats.saturating_mul(4)
    }
}

fn spark_prefill_scratch_alloc(
    gpu: &mut Gpu,
    chunk: usize,
    dim: usize,
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    qkv_rows: usize,
    n_heads: usize,
) -> Result<SparkPrefillScratch, String> {
    let mut owned: Vec<GpuTensor> = Vec::with_capacity(13);
    let specs: [(usize, &str); 13] = [
        (dim, "x_batch"),
        (dim, "n1"),
        (qkv_rows, "qkv_batch"),
        (q_dim, "q_batch"),
        (kv_dim, "k_batch"),
        (kv_dim, "v_batch"),
        (q_dim, "attn_out_batch"),
        (n_heads, "gate_batch"),
        (dim, "tmp_batch"),
        (hidden, "gate_ffn_batch"),
        (hidden, "up_ffn_batch"),
        (hidden, "ffn_out_batch"),
        (1, "positions"),
    ];
    for &(n_per, label) in &specs {
        let n = match chunk.checked_mul(n_per) {
            Some(v) => v,
            None => {
                for t in owned.drain(..) {
                    let _ = gpu.free_tensor(t);
                }
                return Err(format!("spark25: scratch {label} size overflow"));
            }
        };
        match gpu.zeros(&[n], DType::F32) {
            Ok(t) => owned.push(t),
            Err(e) => {
                for t in owned.drain(..) {
                    let _ = gpu.free_tensor(t);
                }
                return Err(format!("spark25: alloc scratch {label}: {e:?}"));
            }
        }
    }

    let mut it = owned.into_iter();
    Ok(SparkPrefillScratch {
        x_batch: it.next().unwrap(),
        n1: it.next().unwrap(),
        qkv_batch: it.next().unwrap(),
        q_batch: it.next().unwrap(),
        k_batch: it.next().unwrap(),
        v_batch: it.next().unwrap(),
        attn_out_batch: it.next().unwrap(),
        gate_batch: it.next().unwrap(),
        tmp_batch: it.next().unwrap(),
        gate_ffn_batch: it.next().unwrap(),
        up_ffn_batch: it.next().unwrap(),
        ffn_out_batch: it.next().unwrap(),
        positions: it.next().unwrap(),
        chunk,
    })
}

/// gfx1010 auto path: own scratch, run cancellable chunked prefill, free scratch.
fn prefill_gfx1010(
    cfg: &Spark25Config,
    weights: &Spark25Weights,
    state: &mut Spark25State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
    capture: Option<&mut [Vec<f32>]>,
    abort: &dyn Fn() -> bool,
) -> Result<Vec<f32>, String> {
    let chunk = SPARK_PREFILL_CHUNK.min(tokens.len()).max(1);
    let scratch = SparkPrefillScratch::new(gpu, cfg, chunk)?;
    let result = match capture {
        Some(cap) => {
            prefill_chunked_capture(cfg, weights, state, gpu, tokens, start_pos, &scratch, cap)
        }
        None => prefill_chunked_cancellable(
            cfg, weights, state, gpu, tokens, start_pos, &scratch, abort,
        ),
    };
    scratch.free_gpu(gpu);
    result
}

/// Chunked prefill with between-chunk (and between-layer) abort checks.
///
/// On abort or any kernel failure: `state.reset(gpu)` then error — no partial
/// prefix or stale logits escape. lm_head runs once on the final input row.
pub fn prefill_chunked_cancellable(
    cfg: &Spark25Config,
    weights: &Spark25Weights,
    state: &mut Spark25State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
    scratch: &SparkPrefillScratch,
    abort: &dyn Fn() -> bool,
) -> Result<Vec<f32>, String> {
    prefill_chunked_impl(
        cfg, weights, state, gpu, tokens, start_pos, scratch, None, abort,
    )
}

/// Chunked prefill that stores the last input row's post-residual hidden per layer
/// (same convention as `decode_step_capture` for the final position).
pub fn prefill_chunked_capture(
    cfg: &Spark25Config,
    weights: &Spark25Weights,
    state: &mut Spark25State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
    scratch: &SparkPrefillScratch,
    capture: &mut [Vec<f32>],
) -> Result<Vec<f32>, String> {
    prefill_chunked_impl(
        cfg,
        weights,
        state,
        gpu,
        tokens,
        start_pos,
        scratch,
        Some(capture),
        &|| false,
    )
}

fn prefill_chunked_impl(
    cfg: &Spark25Config,
    weights: &Spark25Weights,
    state: &mut Spark25State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
    scratch: &SparkPrefillScratch,
    mut capture: Option<&mut [Vec<f32>]>,
    abort: &dyn Fn() -> bool,
) -> Result<Vec<f32>, String> {
    if tokens.is_empty() {
        return Ok(Vec::new());
    }
    let n = tokens.len();
    let start = start_pos as usize;
    let end = start
        .checked_add(n)
        .ok_or_else(|| "spark25: prefill position range overflow".to_string())?;
    if end > state.max_seq {
        return Err(format!(
            "spark25: prefill end position {end} exceeds max_seq {}",
            state.max_seq
        ));
    }
    if scratch.chunk == 0 {
        return Err("spark25: SparkPrefillScratch chunk must be > 0".into());
    }
    // Active batch cannot exceed allocated chunk capacity.
    if scratch.x_batch.numel() < scratch.chunk.saturating_mul(cfg.dim) {
        return Err("spark25: SparkPrefillScratch capacity mismatch".into());
    }
    if let Some(cap) = capture.as_ref() {
        if cap.len() < cfg.n_layers {
            return Err(format!(
                "spark25: capture len {} < n_layers {}",
                cap.len(),
                cfg.n_layers
            ));
        }
    }

    // All bounds checked — from here any failure resets state (fail closed).
    match prefill_chunked_body(
        cfg,
        weights,
        state,
        gpu,
        tokens,
        start_pos,
        scratch,
        capture.as_deref_mut(),
        abort,
    ) {
        Ok(logits) => Ok(logits),
        Err(e) => {
            let _ = state.reset(gpu);
            Err(e)
        }
    }
}

fn prefill_chunked_body(
    cfg: &Spark25Config,
    weights: &Spark25Weights,
    state: &mut Spark25State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
    scratch: &SparkPrefillScratch,
    mut capture: Option<&mut [Vec<f32>]>,
    abort: &dyn Fn() -> bool,
) -> Result<Vec<f32>, String> {
    let dim = cfg.dim;
    let hidden = cfg.hidden_dim;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let qkv_rows = cfg.qkv_rows();
    let n_heads = cfg.n_heads;
    let n_kv = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let rms_eps = cfg.rms_norm_eps;
    let dim_bytes = dim * 4;
    let q_bytes = q_dim * 4;
    let kv_bytes = kv_dim * 4;
    let n = tokens.len();
    let chunk_cap = scratch.chunk;

    let mut offset = 0usize;
    let mut last_c = 0usize;

    while offset < n {
        if abort() {
            return Err("spark25 prefill aborted".into());
        }
        let c = (n - offset).min(chunk_cap);
        last_c = c;
        let pos0 = start_pos as usize + offset;

        // Active-row views — numel matches live rows so flattened ops never
        // touch unwritten capacity tails on a ragged final chunk.
        let x = scratch.x_batch.sub_offset(0, c * dim);
        let n1 = scratch.n1.sub_offset(0, c * dim);
        let qkv = scratch.qkv_batch.sub_offset(0, c * qkv_rows);
        let q = scratch.q_batch.sub_offset(0, c * q_dim);
        let k = scratch.k_batch.sub_offset(0, c * kv_dim);
        let v = scratch.v_batch.sub_offset(0, c * kv_dim);
        let attn_out = scratch.attn_out_batch.sub_offset(0, c * q_dim);
        let gate = scratch.gate_batch.sub_offset(0, c * n_heads);
        let tmp = scratch.tmp_batch.sub_offset(0, c * dim);
        let gate_ffn = scratch.gate_ffn_batch.sub_offset(0, c * hidden);
        let up_ffn = scratch.up_ffn_batch.sub_offset(0, c * hidden);
        let ffn_out = scratch.ffn_out_batch.sub_offset(0, c * hidden);
        let pos_view = scratch.positions.sub_offset(0, c);

        for r in 0..c {
            let row = scratch.x_batch.sub_offset(r * dim, dim);
            embed_lookup(gpu, weights, dim, tokens[offset + r], &row)
                .map_err(|e| format!("spark25 prefill embed[{}]: {e}", pos0 + r))?;
        }

        let pos_host: Vec<i32> = (0..c).map(|r| (pos0 + r) as i32).collect();
        let pos_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pos_host.as_ptr() as *const u8, c * 4) };
        gpu.hip
            .memcpy_htod(&pos_view.buf, pos_bytes)
            .map_err(|e| format!("spark25 prefill htod positions: {e:?}"))?;

        for layer_idx in 0..cfg.n_layers {
            if abort() {
                return Err("spark25 prefill aborted".into());
            }
            let lw = &weights.layers[layer_idx];
            let kv_slot = state.kv_slot_for_layer[layer_idx];
            let is_sliding = cfg.is_sliding(layer_idx);
            let window = cfg.window_for(layer_idx);
            let theta = cfg.rope_theta_for(layer_idx);
            let n_rot_pairs = cfg.n_rot_pairs_for(layer_idx);

            gpu.rmsnorm_batched(&x, &lw.input_layernorm, &n1, c, dim, rms_eps)
                .map_err(|e| format!("spark25 L{layer_idx} chunk@{pos0}: input rmsnorm: {e:?}"))?;

            weight_gemm(gpu, &lw.q_k_v_proj, &n1, &qkv, c)
                .map_err(|e| format!("spark25 L{layer_idx} chunk@{pos0}: q_k_v_proj: {e}"))?;
            split_qkv_batch(gpu, &qkv, &q, &k, &v, c, q_dim, kv_dim, q_bytes, kv_bytes)
                .map_err(|e| format!("spark25 L{layer_idx} chunk@{pos0}: {e}"))?;

            if cfg.headwise_attn_output_gate {
                weight_gemm(gpu, &lw.g_proj, &n1, &gate, c)
                    .map_err(|e| format!("spark25 L{layer_idx} chunk@{pos0}: g_proj: {e}"))?;
            }

            // Incremental per-row: RoPE → KV write → SWA attention (causality).
            let seq_need = pos0 + c;
            {
                let kv = if is_sliding {
                    &mut state.kv_sliding
                } else {
                    &mut state.kv_full
                };
                kv.ensure_mapped_capacity(gpu, seq_need)
                    .map_err(|e| format!("spark25 L{layer_idx} chunk@{pos0}: kv map: {e:?}"))?;
                if kv_slot >= kv.k_gpu.len() {
                    return Err(format!(
                        "spark25 L{layer_idx}: kv_slot {kv_slot} out of range (len={})",
                        kv.k_gpu.len()
                    ));
                }
                for r in 0..c {
                    let pos = (pos0 + r) as u32;
                    gpu.hip
                        .memcpy_htod(&state.pos_buf, &(pos as i32).to_ne_bytes())
                        .map_err(|e| {
                            format!("spark25 L{layer_idx} chunk@{pos0}: htod pos {pos}: {e:?}")
                        })?;

                    let q_row = scratch.q_batch.sub_offset(r * q_dim, q_dim);
                    let k_row = scratch.k_batch.sub_offset(r * kv_dim, kv_dim);
                    let v_row = scratch.v_batch.sub_offset(r * kv_dim, kv_dim);
                    let out_row = scratch.attn_out_batch.sub_offset(r * q_dim, q_dim);

                    if n_rot_pairs == 0 {
                        // degenerate — skip
                    } else if n_rot_pairs * 2 >= head_dim {
                        gpu.rope_f32(
                            &q_row,
                            &k_row,
                            &state.pos_buf,
                            n_heads,
                            n_kv,
                            head_dim,
                            theta,
                        )
                        .map_err(|e| {
                            format!("spark25 L{layer_idx} chunk@{pos0}: rope_f32 r{r}: {e:?}")
                        })?;
                    } else {
                        let n_rot = n_rot_pairs * 2;
                        gpu.rope_partial_interleaved_f32(
                            &q_row,
                            &k_row,
                            &state.pos_buf,
                            n_heads,
                            n_kv,
                            head_dim,
                            n_rot,
                            theta,
                        )
                        .map_err(|e| {
                            format!("spark25 L{layer_idx} chunk@{pos0}: rope_partial r{r}: {e:?}")
                        })?;
                    }

                    gpu.kv_cache_write_q8_0(
                        &kv.k_gpu[kv_slot],
                        &k_row,
                        &state.pos_buf,
                        n_kv,
                        head_dim,
                    )
                    .map_err(|e| {
                        format!("spark25 L{layer_idx} chunk@{pos0}: kv write k r{r}: {e:?}")
                    })?;
                    gpu.kv_cache_write_q8_0(
                        &kv.v_gpu[kv_slot],
                        &v_row,
                        &state.pos_buf,
                        n_kv,
                        head_dim,
                    )
                    .map_err(|e| {
                        format!("spark25 L{layer_idx} chunk@{pos0}: kv write v r{r}: {e:?}")
                    })?;

                    let seq_hint = pos as usize + 1;
                    gpu.attention_q8_0_kv_swa(
                        &q_row,
                        &kv.k_gpu[kv_slot],
                        &kv.v_gpu[kv_slot],
                        &out_row,
                        &state.pos_buf,
                        seq_hint,
                        n_heads,
                        n_kv,
                        head_dim,
                        kv.physical_cap,
                        window,
                    )
                    .map_err(|e| {
                        format!("spark25 L{layer_idx} chunk@{pos0}: attn swa r{r}: {e:?}")
                    })?;
                }
            }

            // Headwise sigmoid: gate[i/head_dim] with n_heads_total = c*n_heads.
            if cfg.headwise_attn_output_gate {
                gpu.sigmoid_mul_broadcast_f32(&attn_out, &gate, head_dim)
                    .map_err(|e| {
                        format!("spark25 L{layer_idx} chunk@{pos0}: sigmoid gate: {e:?}")
                    })?;
            }

            weight_gemm(gpu, &lw.out_proj, &attn_out, &tmp, c)
                .map_err(|e| format!("spark25 L{layer_idx} chunk@{pos0}: out_proj: {e}"))?;
            gpu.add_inplace_f32(&x, &tmp)
                .map_err(|e| format!("spark25 L{layer_idx} chunk@{pos0}: attn residual: {e:?}"))?;

            // FFN: reuse n1 as post-attn rmsnorm destination.
            gpu.rmsnorm_batched(&x, &lw.post_attention_layernorm, &n1, c, dim, rms_eps)
                .map_err(|e| {
                    format!("spark25 L{layer_idx} chunk@{pos0}: post_attn rmsnorm: {e:?}")
                })?;
            weight_gemm(gpu, &lw.gate_proj, &n1, &gate_ffn, c)
                .map_err(|e| format!("spark25 L{layer_idx} chunk@{pos0}: gate_proj: {e}"))?;
            weight_gemm(gpu, &lw.up_proj, &n1, &up_ffn, c)
                .map_err(|e| format!("spark25 L{layer_idx} chunk@{pos0}: up_proj: {e}"))?;
            gpu.gelu_erf_mul_f32(&gate_ffn, &up_ffn, &ffn_out)
                .map_err(|e| format!("spark25 L{layer_idx} chunk@{pos0}: gelu_erf_mul: {e:?}"))?;
            weight_gemm(gpu, &lw.down_proj, &ffn_out, &tmp, c)
                .map_err(|e| format!("spark25 L{layer_idx} chunk@{pos0}: down_proj: {e}"))?;
            gpu.add_inplace_f32(&x, &tmp)
                .map_err(|e| format!("spark25 L{layer_idx} chunk@{pos0}: ffn residual: {e:?}"))?;

            // Capture final-row post-residual only on the last chunk.
            if let Some(cap) = capture.as_mut() {
                if offset + c == n {
                    let last_row = scratch.x_batch.sub_offset((c - 1) * dim, dim);
                    let host = gpu.download_f32(&last_row).map_err(|e| {
                        format!("spark25 L{layer_idx} chunk@{pos0}: capture download: {e:?}")
                    })?;
                    cap[layer_idx].clear();
                    cap[layer_idx].extend_from_slice(&host);
                }
            }
        }

        // Commit n_tokens per completed chunk (bookkeeping only — no lm_head).
        state.n_tokens = state.n_tokens.max(pos0 + c);
        offset += c;
    }

    // Final head once, last input row only.
    debug_assert!(last_c > 0);
    let last_row = scratch.x_batch.sub_offset((last_c - 1) * dim, dim);
    gpu.memcpy_dtod_auto(&state.x.buf, &last_row.buf, dim_bytes)
        .map_err(|e| format!("spark25: copy final hidden: {e:?}"))?;
    let last_pos = start_pos as usize + n - 1;
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(last_pos as i32).to_ne_bytes())
        .map_err(|e| format!("spark25: htod final pos: {e:?}"))?;
    gpu.rmsnorm_f32(&state.x, &weights.final_norm, &state.tmp, rms_eps)
        .map_err(|e| format!("spark25: final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.tmp, &state.logits)
        .map_err(|e| format!("spark25: lm_head: {e}"))?;
    state.n_tokens = state.n_tokens.max(start_pos as usize + n);

    gpu.download_f32(&state.logits)
        .map_err(|e| format!("spark25: download logits: {e:?}"))
}

/// Split fused QKV batch rows: each row is contiguous Q|K|V.
fn split_qkv_batch(
    gpu: &mut Gpu,
    qkv: &GpuTensor,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    batch: usize,
    q_dim: usize,
    kv_dim: usize,
    q_bytes: usize,
    kv_bytes: usize,
) -> Result<(), String> {
    let qkv_row = q_dim + 2 * kv_dim;
    let qkv_row_bytes = qkv_row * 4;
    for r in 0..batch {
        let src_base = r * qkv_row_bytes;
        let q_dst = r * q_bytes;
        let k_dst = r * kv_bytes;
        let v_dst = r * kv_bytes;
        gpu.memcpy_dtod_at_auto(&q.buf, q_dst, &qkv.buf, src_base, q_bytes)
            .map_err(|e| format!("split q r{r}: {e:?}"))?;
        gpu.memcpy_dtod_at_auto(&k.buf, k_dst, &qkv.buf, src_base + q_bytes, kv_bytes)
            .map_err(|e| format!("split k r{r}: {e:?}"))?;
        gpu.memcpy_dtod_at_auto(
            &v.buf,
            v_dst,
            &qkv.buf,
            src_base + q_bytes + kv_bytes,
            kv_bytes,
        )
        .map_err(|e| format!("split v r{r}: {e:?}"))?;
    }
    Ok(())
}
