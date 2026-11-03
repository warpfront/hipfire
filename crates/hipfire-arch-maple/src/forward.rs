// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Maple-Preview forward pass (free functions, hot-path static dispatch).
//!
//! Standard pre-norm block — NOT cohere2moe's parallel block:
//! ```text
//!   h += o_proj(attn(rmsnorm(h, input_layernorm)))
//!   h += moe(rmsnorm(h, post_attention_layernorm))
//!   logits = lm_head(rmsnorm(h, model.norm))
//! ```
//!
//! The layers are the typed step program of [`crate::program`]; this module
//! owns the scratch, stages the positions and embeddings, and runs it.
//!
//! The four places this arch can go silently wrong, and what pins each:
//!
//! 1. **QK-norm runs BEFORE RoPE** (`modeling_maple.py:300-303`). Both are
//!    in-place on the q/k buffers, so the step order IS the semantics.
//! 2. **Partial rotary, half-split.** `rotary_dim` = 64 of 128; pairs are
//!    `(i, i+n_rot/2)` WITHIN the rotated block and the frequency denominator
//!    is `n_rot`, not `head_dim` (`RopeKind::PrefixHalfSplit`). Gemma-4's
//!    *proportional* convention (`RopeKind::PartialHalved`: pairs span the
//!    full head, denominator `head_dim`) is NOT interchangeable.
//! 3. **NoPE on the global layers.** RoPE is applied only where
//!    `layer_types[l] == sliding_attention`.
//! 4. **Clamped SwiGLU, asymmetric:** `silu(clamp(gate, max=7)) *
//!    clamp(up, -7, 7)`, the declared MQ2-Lloyd clamp of the sealed MoE with
//!    the limit as a parameter — DeepSeek passes 10.0, Maple passes 7.0.
//!
//! The experts are the unrotated MQ2-Lloyd (qt51) format, so the sealed MoE
//! feeds them the natural-basis activation: an FWHT-rotated one is silent
//! garbage, not an error.

use crate::batch::moe_grouped_m_total_bound;
use crate::config::MapleConfig;
use crate::maple::{MapleState, MapleWeights};
use crate::program::{head_steps, lm_head_ref, Bufs, LayerRefs, MoeRow, MoeRows, Route};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::execute_steps;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Decode one token; returns the full logits vector.
pub fn decode_step(
    cfg: &MapleConfig,
    weights: &MapleWeights,
    state: &mut MapleState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    // Fail closed on a position past the KV allocation. Without this the KV
    // write runs off the end of the cache and the first symptom is
    // `hipMemcpy D2H: an illegal memory access was encountered` from a
    // downstream kernel — a fault that names the wrong culprit and says
    // nothing about the cause.
    if position as usize >= state.max_seq {
        return Err(format!(
            "maple: position {position} exceeds max_seq {} — allocate the state \
             with a larger max_seq (prompt + generation must fit)",
            state.max_seq
        ));
    }
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("maple: htod pos: {e:?}"))?;
    embed_lookup(gpu, weights, cfg.hidden_size, token_id, &state.h)?;
    let b = Bufs {
        h: &state.h,
        normed: &state.normed,
        fa_q: &state.fa_q,
        fa_k: &state.fa_k,
        fa_v: &state.fa_v,
        attn_out: &state.fa_attn_out,
        proj_out: &state.fa_attn_out,
        pos_buf: &state.pos_buf,
        kv: &state.kv,
        flash_partials: &state.flash_partials,
        route: Route::Decode {
            position: position as usize,
            moe: MoeRow {
                router_logits: &state.router_logits,
                topk_indices: &state.topk_indices,
                topk_weights: &state.topk_weights,
                gate: &state.gate_batch,
                up: &state.up_batch,
                act: &state.act_batch,
                down_expanded: &state.down_expanded,
                // Attention scratch is dead once the MoE branch runs, and the
                // head norm buffer is written only after the last layer.
                unread: [
                    &state.fa_q,
                    &state.fa_attn_out,
                    &state.fa_k,
                    &state.final_norm_buf,
                ],
            },
        },
    };
    run_layers(cfg, weights, gpu, &b)?;
    let ctx = DispatchCtx::new(gpu);
    let lm_head = lm_head_ref(weights);
    let head = head_steps(
        cfg,
        weights,
        &lm_head,
        &state.h,
        &state.final_norm_buf,
        &state.logits,
    );
    execute_steps(gpu, &ctx, &head).map_err(|e| format!("maple: head: {e:?}"))?;
    // Keep `state.n_tokens` truthful on BOTH paths. `forward_batch` maintains
    // it, `reset` zeroes it, and it is the only record of how far the KV cache
    // is populated (`KvCache` itself carries no cursor). A field that only one
    // of two entry points updates is worse than no field at all: after any
    // decode it would silently under-report.
    state.n_tokens = state.n_tokens.max(position as usize + 1);
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("maple: download logits: {e:?}"))
}

/// Seed the residual stream with the embedding row for `token_id`.
///
/// The converter carries `model.word_embeddings` as BF16, but there is no BF16
/// embedding-lookup kernel; `MapleWeights::load` therefore widens it to F32 on
/// the host at load time (see the note there), so this is the F32 path. Q8 is
/// accepted for a future requantized export.
fn embed_lookup(
    gpu: &mut Gpu,
    weights: &MapleWeights,
    hidden: usize,
    token_id: u32,
    out: &GpuTensor,
) -> Result<(), String> {
    match weights.embed_dtype {
        DType::F32 => gpu
            .embedding_lookup(&weights.embed, out, token_id, hidden)
            .map_err(|e| format!("maple: embed lookup f32: {e:?}")),
        DType::Q8_0 => gpu
            .embedding_lookup_q8(&weights.embed, out, token_id, hidden)
            .map_err(|e| format!("maple: embed lookup q8: {e:?}")),
        other => Err(format!(
            "maple: embed dtype {other:?} has no lookup path (expected F32 or Q8)"
        )),
    }
}

/// Run every layer's program over `b`; on decode, append each layer's residual
/// to the `HIPFIRE_MAPLE_DUMP_HIDDEN` dump when armed.
fn run_layers(
    cfg: &MapleConfig,
    weights: &MapleWeights,
    gpu: &mut Gpu,
    b: &Bufs<'_>,
) -> Result<(), String> {
    let dump = dump_hidden_path().filter(|_| matches!(b.route, Route::Decode { .. }));
    for (l, layer) in weights.layers.iter().enumerate() {
        let refs = LayerRefs::new(layer);
        let ctx = DispatchCtx::new(gpu);
        let attn = refs
            .attention_steps(cfg, b, l)
            .map_err(|e| format!("maple L{l}: attention: {e}"))?;
        execute_steps(gpu, &ctx, &attn).map_err(|e| format!("maple L{l}: attention: {e:?}"))?;
        let moe = refs
            .moe_steps(cfg, &ctx, b, l)
            .map_err(|e| format!("maple L{l}: moe: {e}"))?;
        execute_steps(gpu, &ctx, &moe).map_err(|e| format!("maple L{l}: moe: {e:?}"))?;

        // Per-layer residual capture for reference parity. Off unless
        // HIPFIRE_MAPLE_DUMP_HIDDEN is set, so the hot path is unchanged.
        //
        // A cosine cliff at layer n localises the bug to layer n's block —
        // the method that localised the Bonsai double-norm bug to layer 0.
        // The two silent-wrong-answer risks to check first are
        // RoPE-on-a-NoPE-layer and QK-norm ordering.
        if let Some(path) = dump.as_deref() {
            dump_layer_hidden(gpu, b.h, l, cfg.hidden_size, path)?;
        }
    }
    Ok(())
}

/// Destination for the per-layer residual dump, or `None` when disabled.
fn dump_hidden_path() -> Option<String> {
    static P: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    P.get_or_init(|| hipfire_config::developer_var("HIPFIRE_MAPLE_DUMP_HIDDEN").ok())
        .clone()
}

/// Append `[u32 layer][u32 hidden][hidden f32 LE]` for the current residual
/// (the first row of `h`).
///
/// Appends rather than truncates so a whole prefill produces one file in
/// (position, layer) order; the reader keys on the layer index.
fn dump_layer_hidden(
    gpu: &mut Gpu,
    h: &GpuTensor,
    layer: usize,
    hidden: usize,
    path: &str,
) -> Result<(), String> {
    use std::io::Write;
    let h = gpu
        .download_f32(h)
        .map_err(|e| format!("maple: dump hidden L{layer}: {e:?}"))?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("maple: open {path}: {e}"))?;
    let mut buf = Vec::with_capacity(8 + hidden * 4);
    buf.extend_from_slice(&(layer as u32).to_le_bytes());
    buf.extend_from_slice(&(hidden as u32).to_le_bytes());
    for v in h.iter().take(hidden) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf)
        .map_err(|e| format!("maple: write {path}: {e}"))
}

// ───────────────────────── Batched prefill ─────────────────────────

/// Batched prefill requires uniform qt51 experts (the dense and grouped GEMMs
/// both decode that layout) — anything else falls back to per-token.
///
/// The ROUTER is deliberately checked differently: it is the one Maple weight
/// that is NOT qt51 (BF16 in the published checkpoint), and the batched path
/// drives it from the F16 mirror the loader builds. `router_f16` is `None`
/// exactly when that mirror could not be built, so requiring it here is the
/// router's dtype gate.
pub fn forward_batch_supported(weights: &MapleWeights) -> bool {
    weights.layers.iter().all(|l| {
        l.wq.gpu_dtype == DType::MQ2G256LloydU
            && l.wk.gpu_dtype == DType::MQ2G256LloydU
            && l.wv.gpu_dtype == DType::MQ2G256LloydU
            && l.wo.gpu_dtype == DType::MQ2G256LloydU
            // `.first()` rather than `experts[0]`: an expert-less layer is a
            // malformed checkpoint, not a panic site. Report it as
            // "unsupported" and let the caller fall back to the per-token path.
            && l.moe
                .experts
                .first()
                .is_some_and(|e| {
                    e.gate_up.gpu_dtype == DType::MQ2G256LloydU
                        && e.down.gpu_dtype == DType::MQ2G256LloydU
                })
            && l.moe.router_f16.is_some()
    })
}

/// Prefill `tokens` at `[start_pos, start_pos+B)`; returns the LAST token's
/// logits. Errors above `state.max_b` (the row count the batched scratch was
/// allocated for) rather than splitting — chunking is the caller's job
/// (`batch::prefill_chunks`).
///
/// Both halves are batched: one GEMM per projection per layer over all B rows
/// for attention, and one scatter + two grouped expert GEMMs per layer for the
/// MoE. Nothing on this path loops over tokens except the embedding row copies.
pub fn forward_batch(
    cfg: &MapleConfig,
    weights: &MapleWeights,
    state: &mut MapleState,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: usize,
) -> Result<Vec<f32>, String> {
    let b = tokens.len();
    if b == 0 {
        return Err("maple forward_batch: empty token slice".into());
    }
    // Bound against what the scratch was ACTUALLY allocated from, not against
    // the compile-time cap. `MapleState::new_with_max_seq` currently sizes
    // every `b_*` buffer from `MAPLE_PREFILL_MAX_B`, so today the two are
    // equal — but the recorded follow-up is to size them from
    // `MAPLE_PREFILL_CHUNK` (256) instead. Checking the constant would still
    // admit B=512 the moment that lands, and the dense GEMM would then write
    // `dense_m_total(512) x q_dim` into a 256-row `b_q`. Checking `state.max_b`
    // tracks the allocation by construction.
    if b > state.max_b {
        return Err(format!(
            "maple forward_batch: B={b} exceeds this state's batched-prefill \
             scratch capacity ({} rows) — split the prompt with \
             `batch::prefill_chunks`",
            state.max_b
        ));
    }
    if start_pos + b > state.max_seq {
        return Err(format!(
            "maple forward_batch: positions {start_pos}..{} exceed max_seq {}",
            start_pos + b,
            state.max_seq
        ));
    }
    if !forward_batch_supported(weights) {
        // Name BOTH gates. `forward_batch_supported` fails for two distinct
        // reasons — a non-qt51 attention/expert weight, or a missing router F16
        // mirror — and a checkpoint with a quantized router would otherwise be
        // told its experts were the problem.
        return Err(
            "maple forward_batch: unsupported tier (needs uniform qt51 attention and \
                    expert weights, plus the router's F16 mirror)"
                .into(),
        );
    }

    let hidden = cfg.hidden_size;
    let (n_exp, k_top) = (cfg.num_experts, cfg.num_experts_per_tok);

    // Positions [B] as i32-in-f32, for batched RoPE + the masked windowed attend.
    let pos_bytes: Vec<u8> = (0..b)
        .flat_map(|i| ((start_pos + i) as i32).to_ne_bytes())
        .collect();
    gpu.hip
        .memcpy_htod(&state.b_positions.buf, &pos_bytes)
        .map_err(|e| format!("maple: htod positions: {e:?}"))?;

    // ALSO stage the single-i32 `pos_buf` scalar, mirroring `decode_step`.
    // `AttnParams` carries both `positions` (batch_size > 1) and `pos_buf`
    // (batch_size == 1), and every KV-write/attend dispatch in
    // `hipfire-dispatch/src/families/attention.rs` picks between them purely
    // on `plan.batch_size` — which is `b`, the PER-CALL token count, not the
    // logical prefill size. Any chunk with b == 1 (an unbatched `--b 1` run,
    // or a ragged trailing remainder like `prefill_chunks(256, 17)`'s final
    // 1-token chunk) takes the `pos_buf` branch. Leaving it unwritten here
    // meant that branch silently read whatever a PRIOR call — or GPU-zeroed
    // scratch — left behind, corrupting the KV write and attend position for
    // that chunk. Do not delete this thinking it's dead because b_positions
    // "covers" the batched case; it doesn't cover b == 1.
    gpu.hip
        .memcpy_htod(&state.pos_buf, &((start_pos + b - 1) as i32).to_ne_bytes())
        .map_err(|e| format!("maple: htod pos_buf: {e:?}"))?;

    // Re-stamp the dense slot index for THIS B: identity over the live rows and
    // -1 across the BLOCK_M tail, so the grouped kernel skips the pad tile rows
    // instead of computing them from whatever is left in the scratch. The table
    // is allocated for MAPLE_PREFILL_MAX_B and rewritten per chunk (≤ 2 KB, once
    // per chunk — not per layer). `dense_tile_ids` is all-zero for every B, so
    // the load-time upload of that one still stands.
    let slots = crate::batch::dense_slot_index_host(b);
    let slot_bytes: Vec<u8> = slots.iter().flat_map(|x| x.to_ne_bytes()).collect();
    gpu.hip
        .memcpy_htod(&state.b_slot_index.buf, &slot_bytes)
        .map_err(|e| format!("maple: htod slot index: {e:?}"))?;

    // Seed the residual rows with embeddings (one row copy per token; the
    // lookup kernels are per-token and this is a memcpy-scale cost).
    for (i, &tok) in tokens.iter().enumerate() {
        let row = view(&state.b_h, i * hidden, &[hidden]);
        embed_lookup(gpu, weights, hidden, tok, &row)?;
    }

    // Steps read exactly the live rows (the scratch is sized for `max_b`, and
    // the residual add and the softmax take their row count from the shape),
    // except the grouped projection outputs: the GEMM also writes the rows of
    // the 16-row padding tail.
    let rows = |t: &GpuTensor, width: usize| view(t, 0, &[b * width]);
    let padded =
        |t: &GpuTensor, width: usize| view(t, 0, &[crate::batch::dense_m_total(b) * width]);
    let (q_dim, kv_dim, mi) = (cfg.q_dim(), cfg.kv_dim(), cfg.moe_intermediate_size);
    let h = rows(&state.b_h, hidden);
    let normed = rows(&state.b_normed, hidden);
    let (fa_q, fa_k, fa_v) = (
        padded(&state.b_q, q_dim),
        padded(&state.b_k, kv_dim),
        padded(&state.b_v, kv_dim),
    );
    let attn_out = rows(&state.b_attn_out, q_dim);
    let positions = rows(&state.b_positions, 1);
    let router_logits = rows(&state.b_router_logits, n_exp);
    let router_scores = view(&state.b_router_probs, 0, &[b, n_exp]);
    let (topk_indices, topk_weights) = (
        rows(&state.b_topk_indices, k_top),
        rows(&state.b_topk_weights, k_top),
    );
    let act = rows(&state.b_act, k_top * mi);
    let bufs = Bufs {
        h: &h,
        normed: &normed,
        fa_q: &fa_q,
        fa_k: &fa_k,
        fa_v: &fa_v,
        attn_out: &attn_out,
        proj_out: &state.b_proj_out,
        pos_buf: &state.pos_buf,
        kv: &state.kv,
        flash_partials: &state.flash_partials,
        route: Route::Prefill {
            rows: b,
            positions: &positions,
            last_pos: start_pos + b - 1,
            full_causal: force_full_causal(),
            dense_slot_index: &state.b_slot_index,
            dense_tile_ids: &state.b_tile_ids,
            hidden_f16: &state.b_normed_f16,
            attn_f16: &state.b_attn_out_f16,
            moe: MoeRows {
                router_logits: &router_logits,
                router_scores: &router_scores,
                topk_indices: &topk_indices,
                topk_weights: &topk_weights,
                x_f16: &state.b_normed_f16,
                act: &act,
                act_f16: &state.b_act_f16,
                expert_token_counts: &state.b_expert_counts,
                expert_offsets: &state.b_expert_offsets,
                sorted_slot_index: &state.b_sorted_slot,
                expert_tile_ids: &state.b_expert_tiles,
                inverse_perm: &state.b_inverse_perm,
                y_gate_up_grouped: &state.b_y_gate_up,
                y_down_grouped: &state.b_y_down,
                m_total: moe_grouped_m_total_bound(b * k_top, n_exp),
            },
        },
    };
    run_layers(cfg, weights, gpu, &bufs)?;

    // Head: last row only — the caller wants the next-token distribution, and
    // running lm_head over all B rows would cost B × vocab.
    let last = view(&state.b_h, (b - 1) * hidden, &[hidden]);
    let ctx = DispatchCtx::new(gpu);
    let lm_head = lm_head_ref(weights);
    let head = head_steps(
        cfg,
        weights,
        &lm_head,
        &last,
        &state.final_norm_buf,
        &state.logits,
    );
    execute_steps(gpu, &ctx, &head).map_err(|e| format!("maple: head: {e:?}"))?;
    state.n_tokens = state.n_tokens.max(start_pos + b);
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("maple: download logits: {e:?}"))
}

/// Non-owning view of `shape` (product) 4-byte elements at element `offset`
/// inside `t`, keeping its dtype (F32, or the i32-in-F32 index tables).
///
/// The returned tensor MUST NOT be freed — it aliases `t`'s allocation, and
/// `DeviceBuffer::from_raw` has no Drop-time free, so the alias and the owner
/// coexist until the OWNER is freed once.
fn view(t: &GpuTensor, offset: usize, shape: &[usize]) -> GpuTensor {
    let len: usize = shape.iter().product();
    assert!(
        offset + len <= t.numel() && t.dtype == DType::F32,
        "maple view out of range or not F32: {offset}+{len} > {}",
        t.numel()
    );
    unsafe {
        let base = t.buf.as_ptr() as *mut u8;
        GpuTensor {
            buf: hip_bridge::DeviceBuffer::from_raw(base.add(offset * 4) as *mut _, len * 4),
            shape: shape.to_vec(),
            dtype: DType::F32,
        }
    }
}

/// Test-only override: force full causal on the SLIDING layers too, so the
/// sliding window can be PROVEN live by differential rather than assumed.
///
/// Parity against the per-token path alone cannot show it — if both paths
/// dropped the window they would still agree with each other. Forcing it off on
/// only the batched side makes a live window observable as a divergence, and a
/// dead one as silence. At 1200 tokens this drives cosine from 0.996 to -0.725.
///
/// Cached in a `OnceLock` for the same reason `dump_hidden_path` is:
/// `developer_var` scans the config field table and allocates, and this sits on
/// a per-chunk path that is directly measured.
fn force_full_causal() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_MAPLE_FORCE_FULL_CAUSAL").as_deref() == Ok("1")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MapleConfig {
        let json = r#"{"config":{
            "vocab_size": 151936, "hidden_size": 2048, "num_hidden_layers": 8,
            "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 128,
            "moe_intermediate_size": 512, "num_experts": 256, "num_experts_per_tok": 8,
            "partial_rotary_factor": 0.5, "sliding_window": 512, "rope_theta": 10000,
            "layer_types": [
              "sliding_attention","sliding_attention","sliding_attention","full_attention",
              "sliding_attention","sliding_attention","sliding_attention","full_attention"]
        }}"#;
        MapleConfig::from_metadata_json(json).unwrap()
    }

    /// The reference clamp, transcribed from `modeling_maple.py:110`:
    /// `silu(clamp(gate, max=7)) * clamp(up, min=-7, max=7)`.
    fn reference_swiglu(gate: f32, up: f32, limit: f32) -> f32 {
        let g = gate.min(limit); // capped from ABOVE only
        let u = up.clamp(-limit, limit); // clamped BOTH ways
        (g / (1.0 + (-g).exp())) * u
    }

    #[test]
    fn the_clamp_is_asymmetric_gate_above_only_up_both_ways() {
        let l = 7.0f32;
        // A large positive gate is capped...
        assert_eq!(
            reference_swiglu(100.0, 1.0, l),
            reference_swiglu(7.0, 1.0, l)
        );
        // ...but a large NEGATIVE gate is not. If the gate were clamped from
        // below too, these would be equal.
        assert_ne!(
            reference_swiglu(-100.0, 1.0, l),
            reference_swiglu(-7.0, 1.0, l)
        );
        // `up` is clamped on both sides.
        assert_eq!(
            reference_swiglu(1.0, 100.0, l),
            reference_swiglu(1.0, 7.0, l)
        );
        assert_eq!(
            reference_swiglu(1.0, -100.0, l),
            reference_swiglu(1.0, -7.0, l)
        );
    }

    #[test]
    fn the_clamp_limit_is_seven_not_deepseeks_ten() {
        // The kernel is shared with DeepSeek V4, which passes 10.0. Maple's 7.0
        // comes from the config type, and the two differ on real inputs — so a
        // copied 10.0 would be a silent quality change, not a crash.
        let c = cfg();
        assert_eq!(c.swiglu_clamp, 7.0);
        assert_ne!(
            reference_swiglu(9.0, 1.0, 7.0),
            reference_swiglu(9.0, 1.0, 10.0)
        );
    }

    #[test]
    fn rope_is_applied_on_sliding_layers_only() {
        let c = cfg();
        let roped: Vec<usize> = (0..c.num_hidden_layers)
            .filter(|&l| c.applies_rope(l))
            .collect();
        assert_eq!(roped, vec![0, 1, 2, 4, 5, 6], "layers 3 and 7 are NoPE");
    }

    #[test]
    fn rotary_covers_half_the_head_and_pairs_within_that_block() {
        // n_rot = 64 of head_dim 128. The half-split kernel pairs
        // (i, i + n_rot/2) = (i, i+32) — INSIDE the rotated block. Gemma-4's
        // rope_partial_halved pairs (i, i + head_dim/2) = (i, i+64), which
        // would reach into the pass-through half.
        let c = cfg();
        assert_eq!(c.rotary_dim(), 64);
        assert_eq!(c.rotary_dim() / 2, 32);
        assert_ne!(c.rotary_dim() / 2, c.head_dim / 2);
    }

    #[test]
    fn sliding_window_is_512_and_full_layers_are_unwindowed() {
        let c = cfg();
        assert_eq!(c.attention_span(0, 10_000).len(), 512);
        assert_eq!(c.attention_span(3, 10_000).start, 0);
    }
}
