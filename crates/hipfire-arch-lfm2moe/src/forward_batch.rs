// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Batched decode forward path for dense LFM2.5 models.
//!
//! Provides `prepare_decode_batch_inputs_lfm`, `forward_decode_batch_lfm` and
//! `forward_decode_batch_prepared_lfm`. The layers run the declarative
//! program (`crate::program`) over the `B` active lanes: each step launches
//! one batched kernel with `rows = B` on lane-isolated `[B, C, K-1]` conv
//! state and Q8 KV. No per-lane full-forward loop exists in steady-state
//! decode (prefill may be sequential).

use hip_bridge::{HipError, HipResult};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::execute_steps;
use rdna_compute::{Gpu, GpuTensor};

use crate::batch::Lfm2DecodeBatchState;
use crate::lfm2moe::Lfm2MoeWeights;
use crate::program::{head_steps, Bufs, Lanes, LayerRefs};

// ──────────────────────────────────────────────────────────────────────────
// helpers
// ──────────────────────────────────────────────────────────────────────────

fn check_dense(cfg: &crate::config::Lfm2MoeConfig) -> HipResult<()> {
    if cfg.num_experts != 0 {
        return Err(HipError::new(
            0,
            "batched decode: MoE configs not supported (dense only)",
        ));
    }
    if cfg.hidden_size == 0
        || cfg.vocab_size == 0
        || cfg.num_hidden_layers == 0
        || cfg.head_dim == 0
        || cfg.num_attention_heads == 0
        || cfg.num_key_value_heads == 0
        || cfg.intermediate_size == 0
    {
        return Err(HipError::new(0, "batched decode: zero geometry"));
    }
    if cfg.conv_kernel_size < 2 || cfg.conv_kernel_size > 8 {
        return Err(HipError::new(
            0,
            "batched decode: conv_kernel_size must be in [2,8]",
        ));
    }
    Ok(())
}

fn check_weight_formats(weights: &Lfm2MoeWeights) -> HipResult<()> {
    crate::lfm2moe::batch_weight_formats_supported(weights).map_err(|e| HipError::new(0, &e))
}

/// Authoritative batch format predicate — single source of truth for daemon attestation
/// and all forward paths. Dense projections: Q8_0/HFQ4G256/MQ4G256; embeddings: Q8/HFQ4;
/// lm_head: Q8/HFQ4/MQ4/HFQ6/MQ6/MQ3; MoE not admitted.
pub fn batch_weight_formats_supported(weights: &Lfm2MoeWeights) -> Result<(), String> {
    crate::lfm2moe::batch_weight_formats_supported(weights)
}

// ──────────────────────────────────────────────────────────────────────────
// public API
// ──────────────────────────────────────────────────────────────────────────

/// Populate batched embedding and position buffers. Returns the max position.
pub fn prepare_decode_batch_inputs_lfm(
    gpu: &mut Gpu,
    weights: &Lfm2MoeWeights,
    cfg: &crate::config::Lfm2MoeConfig,
    tokens: &[u32],
    positions: &[usize],
    state: &Lfm2DecodeBatchState,
) -> HipResult<usize> {
    let n = tokens.len();
    if n == 0 {
        return Err(HipError::new(
            0,
            "prepare_decode_batch_inputs_lfm: requires at least one lane",
        ));
    }
    if n > state.max_batch || positions.len() != n {
        return Err(HipError::new(
            0,
            "prepare_decode_batch_inputs_lfm: batch exceeds max_batch or positions length mismatch",
        ));
    }
    if positions.iter().any(|&p| p >= state.lane_capacity) {
        return Err(HipError::new(
            0,
            "prepare_decode_batch_inputs_lfm: position exceeds lane capacity",
        ));
    }
    check_dense(cfg)?;
    check_weight_formats(weights)?;

    // token ids -> device buffer [n] i32 stored as F32 slots (like Qwen)
    let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let token_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, tokens_host.len() * 4)
    };
    // copy only the active prefix; state.tokens is [max_batch] F32
    let tokens_active = state.tokens.sub_offset(0, n);
    gpu.hip.memcpy_htod(&tokens_active.buf, token_bytes)?;

    match weights.embd_format {
        hipfire_runtime::llama::EmbeddingFormat::HFQ4G256 => {
            let x_active = state.x_batch.sub_offset(0, n * cfg.hidden_size);
            gpu.embedding_lookup_hfq4g256_batched(
                &weights.embed,
                &x_active,
                &tokens_active,
                n,
                cfg.hidden_size,
            )?
        }
        hipfire_runtime::llama::EmbeddingFormat::Q8_0 => {
            let x_active = state.x_batch.sub_offset(0, n * cfg.hidden_size);
            gpu.embedding_lookup_q8_batched(
                &weights.embed,
                &x_active,
                &tokens_active,
                n,
                cfg.hidden_size,
            )?
        }
        // F32 table (a BF16 teacher widens its embedding to F32 — see
        // `lfm2moe.rs` embed loaders). There is no batched F32 lookup kernel, so
        // gather row by row into the batch tensor. This is a dtod row copy per
        // token, not a GEMM: at B=2048 x hidden=1024 it is ~8 MB of copies
        // against multi-GFLOP projections, i.e. noise. Without this arm a
        // full-precision teacher cannot run the batched path at all.
        hipfire_runtime::llama::EmbeddingFormat::F32 => {
            for i in 0..n {
                let row = state
                    .x_batch
                    .sub_offset(i * cfg.hidden_size, cfg.hidden_size);
                gpu.embedding_lookup(&weights.embed, &row, tokens[i], cfg.hidden_size)?;
            }
        }
        _ => {
            return Err(HipError::new(
                0,
                "prepare_decode_batch_inputs_lfm: unsupported embedding format",
            ))
        }
    }

    let positions_host: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
    let pos_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            positions_host.as_ptr() as *const u8,
            positions_host.len() * 4,
        )
    };
    let pos_active = state.positions.sub_offset(0, n);
    gpu.hip.memcpy_htod(&pos_active.buf, pos_bytes)?;

    Ok(positions.iter().copied().max().unwrap_or(0))
}

/// Batched decode: one launch per layer (no per-lane full-forward loop).
pub fn forward_decode_batch_lfm(
    gpu: &mut Gpu,
    weights: &Lfm2MoeWeights,
    cfg: &crate::config::Lfm2MoeConfig,
    tokens: &[u32],
    positions: &[usize],
    state: &mut Lfm2DecodeBatchState,
) -> HipResult<()> {
    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }
    let position = prepare_decode_batch_inputs_lfm(gpu, weights, cfg, tokens, positions, state)?;
    forward_decode_batch_prepared_lfm(gpu, weights, cfg, tokens, positions, position, state)
}

/// Execute a batch whose embeddings/positions were already staged by
/// `prepare_decode_batch_inputs_lfm`. Validates that the staged buffers
/// match the declared shape.
pub fn forward_decode_batch_prepared_lfm(
    gpu: &mut Gpu,
    weights: &Lfm2MoeWeights,
    cfg: &crate::config::Lfm2MoeConfig,
    tokens: &[u32],
    positions: &[usize],
    position: usize,
    state: &mut Lfm2DecodeBatchState,
) -> HipResult<()> {
    let n = tokens.len();
    if n == 0
        || n > state.max_batch
        || positions.len() != n
        || positions.iter().any(|&p| p >= state.lane_capacity)
        || position != positions.iter().copied().max().unwrap_or(0)
    {
        return Err(HipError::new(
            0,
            "forward_decode_batch_prepared_lfm: prepared inputs do not match admitted shape",
        ));
    }
    check_dense(cfg)?;
    check_weight_formats(weights)?;

    let hidden = cfg.hidden_size;
    let (q_dim, kv_dim, inter) = (cfg.q_dim(), cfg.kv_dim(), cfg.intermediate_size);
    // Active windows into the batched buffers (dense packing 0..n).
    let rows = |t: &GpuTensor, width: usize| t.sub_offset(0, n * width);
    let (x, tmp, ffn_in) = (
        rows(&state.x_batch, hidden),
        rows(&state.tmp_batch, hidden),
        rows(&state.ffn_tmp_batch, hidden),
    );
    let (head_in, conv_bcx, conv_y) = (
        rows(&state.final_hidden, hidden),
        rows(&state.conv_bcx_batch, 3 * hidden),
        rows(&state.conv_y_batch, hidden),
    );
    let (fa_q, fa_k, fa_v, fa_attn_out) = (
        rows(&state.fa_q_batch, q_dim),
        rows(&state.fa_k_batch, kv_dim),
        rows(&state.fa_v_batch, kv_dim),
        rows(&state.fa_attn_out_batch, q_dim),
    );
    let (dense_gate, dense_up, dense_act) = (
        rows(&state.dense_gate_batch, inter),
        rows(&state.dense_up_batch, inter),
        rows(&state.dense_act_batch, inter),
    );
    let (logits, lm_rotation, positions) = (
        rows(&state.logits, cfg.vocab_size),
        rows(&state.lm_rot, hidden),
        rows(&state.positions, 1),
    );
    let bufs = Bufs {
        h: &x,
        tmp: &tmp,
        ffn_in: &ffn_in,
        head_in: &head_in,
        conv_bcx: &conv_bcx,
        conv_y: &conv_y,
        fa_q: &fa_q,
        fa_k: &fa_k,
        fa_v: &fa_v,
        fa_attn_out: &fa_attn_out,
        dense_gate: &dense_gate,
        dense_up: &dense_up,
        dense_act: &dense_act,
        logits: &logits,
        // The kernel reads the first `n` lanes of each `[max_batch, ...]` ring.
        conv_states: &state.conv_states,
        lanes: Lanes::Batch {
            rows: n,
            positions: &positions,
            kv: &state.kv,
            lane_capacity: state.lane_capacity,
            rotation: &state.proj_rot,
            lm_rotation: &lm_rotation,
        },
    };
    let err = |what: String| HipError::new(0, &format!("lfm batch {what}"));
    let ctx = DispatchCtx::new(gpu);
    for (l, layer) in weights.layers.iter().enumerate() {
        let refs = LayerRefs::new(layer);
        let mixer = refs
            .mixer_steps(cfg, &bufs)
            .map_err(|e| err(format!("L{l} mixer: {e}")))?;
        execute_steps(gpu, &ctx, &mixer).map_err(|e| err(format!("L{l} mixer: {e}")))?;
        let ffn = refs
            .ffn_steps(cfg, &ctx, &bufs, l)
            .map_err(|e| err(format!("L{l} ffn: {e}")))?;
        execute_steps(gpu, &ctx, &ffn).map_err(|e| err(format!("L{l} ffn: {e}")))?;
    }
    let lm_head = weights.lm_head.dispatch_ref();
    execute_steps(gpu, &ctx, &head_steps(cfg, weights, &lm_head, &bufs))
        .map_err(|e| err(format!("head: {e}")))
}
