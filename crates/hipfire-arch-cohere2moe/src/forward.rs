// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cohere2-MoE (North-Mini-Code) forward pass (free functions, hot-path static
//! dispatch).
//!
//! The defining structural trait is the **parallel block**: a SINGLE
//! `RMSNorm` (cohere2_moe uses RMSNorm at `rms_norm_eps`, NOT base Cohere2's
//! mean-centered LayerNorm) feeds BOTH the attention and the FFN branch, and
//! both add into the residual —
//!   `h = h + o_proj(attn(RMSNorm(h))) + ffn(RMSNorm(h))`
//! (note: the FFN reads the SAME `RMSNorm(h)` as attention, NOT the
//! post-attention residual). Per layer:
//!   normed = rmsnorm(h, input_layernorm)                       [gamma scale, no mean-center / no β]
//!   q,k,v  = proj(normed); RoPE only if sliding (full=NoPE); attn; h += o_proj
//!   if dense (first_k_dense_replace prefix): h += down(silu(gate(normed))·up(normed))
//!   if moe:  router = sigmoid(gate·normed); top-8 (NO renorm: norm_topk_prob=false);
//!            h += Σ w_e · expert_e(normed)
//! then logits = lm_head(rmsnorm(h, model.norm)) · logit_scale.
//!
//! Routed experts use the shared sealed-MoE executor: its indexed kernels handle
//! MQ4/MQ6 tiers, while its host-expert operation preserves the F16/Q8
//! correctness path (including activation weighting before down projection).
//! Architecture code only binds the owned weights and scratch resources.
//!
//! Attention: a per-layer NoPE/RoPE split (full_attention layers are NoPE;
//! sliding layers use interleaved RoPE), plus a windowed-mask flash path
//! (`attention_flash_q8_0_windowed`) that applies the sliding-window mask so
//! context beyond the 4096 window attends correctly. At ≤4096, sliding == full
//! causal, so the window is a no-op there.
//!
//! The layers are the typed step program of [`crate::program`]; this module
//! owns the scratch and runs it.

use crate::cohere2moe::{Cohere2MoeState, Cohere2MoeWeights, Ffn};
use crate::config::Cohere2MoeConfig;
use crate::program::{head_steps, lm_head_ref, Bufs, LayerRefs, MoeRow, MoeRows, Route};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::moe::MOE_GROUPED_BLOCK_M;
use hipfire_dispatch::pipeline::sealed_moe::checked_grouped_m_total_bound;
use hipfire_dispatch::pipeline::{execute_steps, ProjectLadder};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Prefill Q8_0 projection ladder: `ProjectLadder::GemmQ8Fp16` (on WMMA archs
/// with K%32==0 the matrix-core `gemm_q8_0_wmma` over a fresh F16 activation —
/// no MAX_BATCH cap and far faster than the scalar 1-wave-per-row kernel at
/// large batch; the scalar chunked driver on CDNA or when K%32≠0).
/// Opt out of the WMMA Q8 prefill projections (force the scalar sub-batched
/// chunked driver) via `HIPFIRE_COHERE2MOE_Q8_SCALAR=1` — a debugging /
/// precision fallback. WMMA is the default: bit-identical to scalar on the
/// validation prompt and ~9× faster prefill.
fn q8_wmma_enabled() -> bool {
    static EN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EN.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_COHERE2MOE_Q8_SCALAR").as_deref() != Ok("1")
    })
}

/// Decode one token; returns the full logits vector.
pub fn decode_step(
    cfg: &Cohere2MoeConfig,
    weights: &Cohere2MoeWeights,
    state: &mut Cohere2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    // Device position scalar (i32) for rope / kv-write / attention.
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("cohere2moe: htod pos: {e:?}"))?;
    embed_lookup(gpu, weights, cfg.hidden_size, token_id, &mut state.h)?;
    decode_step_body(cfg, weights, state, gpu, position)?;
    let mut logits = gpu
        .download_f32(&state.logits)
        .map_err(|e| format!("cohere2moe: download logits: {e:?}"))?;
    // logit_scale (1.0 for North-Mini-Code → no-op; applied host-side so the
    // device logits stay the raw lm_head output for any downstream re-use).
    if (cfg.logit_scale - 1.0).abs() > f32::EPSILON {
        for v in &mut logits {
            *v *= cfg.logit_scale;
        }
    }
    Ok(logits)
}

/// Seed the residual stream `out` with the embedding row for `token_id`.
/// Dispatches on the (tied) embed dtype: Q8 → dequant kernel; F32 → raw row
/// copy. (The F16 path is unused by the current tiers — embed/lm_head stay Q8
/// across the whole sweep, an engine constraint of the tied-embedding lookup.)
fn embed_lookup(
    gpu: &mut Gpu,
    weights: &Cohere2MoeWeights,
    hidden: usize,
    token_id: u32,
    out: &rdna_compute::GpuTensor,
) -> Result<(), String> {
    match weights.embed_dtype {
        DType::Q8_0 => gpu
            .embedding_lookup_q8(&weights.embed, out, token_id, hidden)
            .map_err(|e| format!("cohere2moe: embed lookup q8: {e:?}")),
        DType::F32 => gpu
            .embedding_lookup(&weights.embed, out, token_id, hidden)
            .map_err(|e| format!("cohere2moe: embed lookup f32: {e:?}")),
        other => Err(format!(
            "cohere2moe: embed dtype {other:?} has no lookup path (use Q8 or F32 tied embeddings)"
        )),
    }
}

/// Per-layer parallel-block stack + final norm + lm_head. Reads `state.h`
/// (seeded by the embedding lookup) and `state.pos_buf` (already staged).
fn decode_step_body(
    cfg: &Cohere2MoeConfig,
    weights: &Cohere2MoeWeights,
    state: &mut Cohere2MoeState,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    let b = Bufs {
        h: &state.h,
        normed: &state.normed,
        fa_q: &state.fa_q,
        fa_k: &state.fa_k,
        fa_v: &state.fa_v,
        attn_out: &state.fa_attn_out,
        proj_out: &state.fa_attn_out,
        dense_gate: &state.dense_gate,
        dense_up: &state.dense_up,
        dense_act: &state.dense_act,
        pos_buf: &state.pos_buf,
        kv: &state.kv,
        flash_partials: &state.flash_partials,
        route: Route::Decode {
            position: position as usize,
            moe: MoeRow {
                router_logits: &state.router_logits,
                topk_indices: &state.topk_indices,
                topk_weights: &state.topk_weights,
                x_rot: &state.ffn_x_rot,
                gate_up: &state.expert_gate_up,
                act: &state.expert_act,
                down: &state.expert_down,
                gate_batch: &state.gate_batch,
                up_batch: &state.up_batch,
                rot_batch: &state.rot_batch,
                down_expanded: &state.down_expanded,
            },
        },
    };
    run_layers(cfg, weights, gpu, &b, |_, _, _| {})?;
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
    execute_steps(gpu, &ctx, &head).map_err(|e| format!("cohere2moe: head: {e:?}"))?;
    state.n_tokens = position as usize + 1;
    Ok(())
}

/// Run every layer's program over `b`; `probe(gpu, layer, tag)` observes the
/// residual after each branch.
fn run_layers(
    cfg: &Cohere2MoeConfig,
    weights: &Cohere2MoeWeights,
    gpu: &mut Gpu,
    b: &Bufs<'_>,
    mut probe: impl FnMut(&mut Gpu, usize, &str),
) -> Result<(), String> {
    for (l, layer) in weights.layers.iter().enumerate() {
        let refs = LayerRefs::new(layer);
        let ctx = DispatchCtx::new(gpu);
        let attn = refs
            .attention_steps(cfg, b, l)
            .map_err(|e| format!("cohere2moe L{l}: attention: {e}"))?;
        execute_steps(gpu, &ctx, &attn)
            .map_err(|e| format!("cohere2moe L{l}: attention: {e:?}"))?;
        probe(gpu, l, "post_attn");
        let ffn = refs
            .ffn_steps(cfg, &ctx, b, l)
            .map_err(|e| format!("cohere2moe L{l}: ffn: {e}"))?;
        execute_steps(gpu, &ctx, &ffn).map_err(|e| format!("cohere2moe L{l}: ffn: {e:?}"))?;
        probe(gpu, l, "post_ffn");
        if hipfire_config::developer_var_os("HIPFIRE_COHERE_DEBUG").is_some() {
            if let Ok(hv) = gpu.download_f32(b.h) {
                let l2 = hv.iter().map(|v| v * v).sum::<f32>().sqrt();
                let mx = hv.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let nan = hv.iter().filter(|v| v.is_nan()).count();
                eprintln!(
                    "[dbg] L{l} {:?} h.l2={l2:.2} max={mx:.3} nan={nan}",
                    layer.attn_kind
                );
            }
        }
    }
    Ok(())
}

/// True iff `forward_batch` supports this model: Q8 attention/dense/router and
/// the shared indexed grouped-MoE admission predicate. Unsupported expert tiers
/// fall back to per-token `decode_step`; this is an admission choice, not a
/// second architecture-local MoE execution path.
pub fn forward_batch_supported(weights: &Cohere2MoeWeights) -> bool {
    weights.layers.iter().all(|l| {
        l.wq.gpu_dtype == DType::Q8_0
            && match &l.ffn {
                Ffn::Dense(_) => true,
                Ffn::Moe(m) => hipfire_dispatch::families::moe::sigmoid_routed_prefill_supported(
                    m.experts[0].gate_up.gpu_dtype,
                ),
            }
    })
}

/// Batched prefill over `B` (≤64) tokens — the parallel-block forward run ONCE
/// per weight matrix for all B tokens (bandwidth-amortized) instead of B
/// per-token `decode_step`s. Fills the KV cache for positions
/// `[start_pos, start_pos+B)` and returns the LAST token's logits. Supports the
/// MQ4/MQ6 serving tiers (Q8 attention/dense + indexed batched MoE); see
/// `forward_batch_supported` (caller falls back to per-token otherwise).
#[allow(clippy::too_many_arguments)]
/// Env-gated long-context localization probe. `HIPFIRE_C2M_NORMDUMP=<step>`
/// (default 4096 when set non-numeric) logs the LAST-token hidden-state L2 /
/// max-abs / NaN per layer on chunks that cross a `step` boundary, so a
/// long-context collapse can be pinned to the exact layer + op (post-attn vs
/// post-ffn) where the residual blows up or flattens. Off by default.
fn c2m_normdump_step() -> Option<usize> {
    hipfire_config::developer_var("HIPFIRE_C2M_NORMDUMP")
        .ok()
        .map(|s| s.trim().parse().unwrap_or(4096))
}
fn dbg_row_stats(
    gpu: &mut Gpu,
    t: &GpuTensor,
    b: usize,
    hidden: usize,
    l: usize,
    ctx: usize,
    tag: &str,
) {
    if let Ok(v) = gpu.download_f32(t) {
        let off = (b - 1) * hidden;
        if off + hidden > v.len() {
            return;
        }
        let row = &v[off..off + hidden];
        let mut l2 = 0.0f64;
        let mut maxa = 0.0f32;
        let mut nan = false;
        for &x in row {
            if !x.is_finite() {
                nan = true;
            }
            l2 += (x as f64) * (x as f64);
            if x.abs() > maxa {
                maxa = x.abs();
            }
        }
        eprintln!(
            "[normdump ctx={ctx} L{l:02} {tag}] l2={:.4} maxabs={:.5} nan={nan}",
            l2.sqrt(),
            maxa
        );
    }
}

pub fn forward_batch(
    cfg: &Cohere2MoeConfig,
    weights: &Cohere2MoeWeights,
    state: &mut Cohere2MoeState,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: usize,
) -> Result<Vec<f32>, String> {
    let b = tokens.len();
    if b == 0 {
        return Err("cohere2moe forward_batch: empty token slice".to_string());
    }
    // The Q8 projections go through `gemm_q8_0_batched_chunked`, which handles
    // arbitrary batch (sub-batches the MAX_BATCH=64 scalar kernel, and upgrades
    // to WMMA on RDNA4). The remaining batched ops (attention, layernorm, topk,
    // rotate, MoE scatter/grouped) carry no 64-cap. Larger batch is what makes
    // the grouped-WMMA MoE path pay off: tokens-per-expert = b·k_top/n_exp grows
    // (≈4 at b=64 → ≈16 at b=256 with n_exp=128), shrinking the BLOCK_M=16
    // padding waste. 512 is a generous scratch-memory ceiling.
    if b > 512 {
        return Err(format!(
            "cohere2moe forward_batch: B={b} exceeds scratch cap 512"
        ));
    }
    if !forward_batch_supported(weights) {
        return Err(
            "cohere2moe forward_batch: unsupported tier (needs Q8 attn + indexed experts)"
                .to_string(),
        );
    }
    let hidden = cfg.hidden_size;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let moe_inter = cfg.moe_intermediate_size;
    let dense_inter = cfg.dense_intermediate_size;
    let n_exp = cfg.num_experts;
    let k_top = cfg.num_experts_per_tok;
    let max_ctx = start_pos + b;

    let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
        g.alloc_tensor(&[n], DType::F32)
            .map_err(|e| format!("forward_batch alloc {label}: {e:?}"))
    };
    let x = alloc(gpu, b * hidden, "x")?;
    let normed = alloc(gpu, b * hidden, "normed")?;
    let fq = alloc(gpu, b * q_dim, "fq")?;
    let fk = alloc(gpu, b * kv_dim, "fk")?;
    let fv = alloc(gpu, b * kv_dim, "fv")?;
    let attn_out = alloc(gpu, b * q_dim, "attn_out")?;
    let o = alloc(gpu, b * hidden, "o")?;
    let ffn_x_rot = alloc(gpu, b * hidden, "ffn_x_rot")?;
    let router_logits = gpu
        .alloc_tensor(&[b, n_exp], DType::F32)
        .map_err(|e| format!("forward_batch alloc router_logits: {e:?}"))?;
    let topk_idx = alloc(gpu, b * k_top, "topk_idx")?;
    let topk_w = alloc(gpu, b * k_top, "topk_w")?;
    let gate = alloc(gpu, b * k_top * moe_inter, "gate")?;
    let up = alloc(gpu, b * k_top * moe_inter, "up")?;
    let rot = alloc(gpu, b * k_top * moe_inter, "rot")?;
    let down_exp = alloc(gpu, b * k_top * hidden, "down_exp")?;
    let dense_gate = alloc(gpu, b * dense_inter, "dense_gate")?;
    let dense_up = alloc(gpu, b * dense_inter, "dense_up")?;
    let dense_act = alloc(gpu, b * dense_inter, "dense_act")?;
    // Grouped-MoE prefill scratch — scatter-by-expert indices (i32, held in
    // DType::Raw byte tensors at 4 bytes/elem) + grouped-WMMA output buffers.
    // Path 1 (indexed) ignores these; Path 2 (grouped) uses them. Always
    // allocated since MoePrefillParams takes every buffer by reference.
    let total_slots = b
        .checked_mul(k_top)
        .ok_or_else(|| "forward_batch MoE route-slot count overflows".to_string())?;
    let m_total_max = checked_grouped_m_total_bound(total_slots, n_exp, MOE_GROUPED_BLOCK_M)
        .map_err(|e| format!("forward_batch grouped MoE bound: {e:?}"))?;
    let raw = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
        g.alloc_tensor(&[n], DType::Raw)
            .map_err(|e| format!("forward_batch alloc {label}: {e:?}"))
    };
    let expert_token_counts = raw(gpu, n_exp * 4, "etc")?;
    let expert_offsets = raw(gpu, (n_exp + 1) * 4, "eoff")?;
    let sorted_slot_index = raw(gpu, m_total_max * 4, "ssi")?;
    let inverse_perm = raw(gpu, b * k_top * 4, "iperm")?;
    let expert_tile_ids = raw(gpu, (m_total_max / MOE_GROUPED_BLOCK_M) * 4, "etid")?;
    let y_gate_up_grouped = alloc(gpu, m_total_max * 2 * moe_inter, "ygu")?;
    let y_down_grouped = alloc(gpu, m_total_max * hidden, "ydn")?;
    // F16 activation scratch for the WMMA Q8 projections (converted fresh per
    // q8_proj_raw call). Sized for the widest projection input K.
    let kmax = hidden.max(q_dim).max(dense_inter);
    let x_f16 = gpu
        .alloc_tensor(&[b * kmax], DType::F16)
        .map_err(|e| format!("forward_batch alloc x_f16: {e:?}"))?;

    // positions [B] i32 (stored in an f32-sized buffer; kernels read it as i32).
    let pos_bytes: Vec<u8> = (0..b)
        .flat_map(|i| ((start_pos + i) as i32).to_ne_bytes())
        .collect();
    let pos_array = alloc(gpu, b, "pos_array")?;
    gpu.hip
        .memcpy_htod(&pos_array.buf, &pos_bytes)
        .map_err(|e| format!("forward_batch htod pos: {e:?}"))?;

    // Embedding (Q8) per token → x[B, hidden].
    {
        let xs = alloc(gpu, hidden, "xs")?;
        for (i, &tok) in tokens.iter().enumerate() {
            embed_lookup(gpu, weights, hidden, tok, &xs)?;
            gpu.hip
                .memcpy_dtod_at(&x.buf, i * hidden * 4, &xs.buf, 0, hidden * 4)
                .map_err(|e| format!("forward_batch embed copy: {e:?}"))?;
        }
        gpu.free_tensor(xs).ok();
    }

    let normdump = c2m_normdump_step().map_or(false, |step| {
        step > 0 && (start_pos / step != max_ctx / step)
    });
    let ladder = if q8_wmma_enabled() {
        ProjectLadder::GemmQ8Fp16
    } else {
        ProjectLadder::Gemm
    };
    let bufs = Bufs {
        h: &x,
        normed: &normed,
        fa_q: &fq,
        fa_k: &fk,
        fa_v: &fv,
        attn_out: &attn_out,
        proj_out: &o,
        dense_gate: &dense_gate,
        dense_up: &dense_up,
        dense_act: &dense_act,
        pos_buf: &state.pos_buf,
        kv: &state.kv,
        flash_partials: &state.flash_partials,
        route: Route::Prefill {
            rows: b,
            positions: &pos_array,
            max_ctx,
            ladder,
            scratch: &x_f16,
            moe: MoeRows {
                router_logits: &router_logits,
                topk_indices: &topk_idx,
                topk_weights: &topk_w,
                x_rot: &ffn_x_rot,
                gate: &gate,
                up: &up,
                rot: &rot,
                down_expanded: &down_exp,
                expert_token_counts: &expert_token_counts,
                expert_offsets: &expert_offsets,
                sorted_slot_index: &sorted_slot_index,
                expert_tile_ids: &expert_tile_ids,
                inverse_perm: &inverse_perm,
                y_gate_up_grouped: &y_gate_up_grouped,
                y_down_grouped: &y_down_grouped,
                m_total_max,
            },
        },
    };
    run_layers(cfg, weights, gpu, &bufs, |gpu, l, tag| {
        if normdump {
            dbg_row_stats(gpu, &x, b, hidden, l, max_ctx, tag);
        }
    })?;
    state.n_tokens = start_pos + b;

    // Final norm + LM head on the LAST row only (prefill needs the last
    // position's logits to seed decode).
    let x_last = x.sub_offset((b - 1) * hidden, hidden);
    let ctx = DispatchCtx::new(gpu);
    let lm_head = lm_head_ref(weights);
    let head = head_steps(
        cfg,
        weights,
        &lm_head,
        &x_last,
        &state.final_norm_buf,
        &state.logits,
    );
    execute_steps(gpu, &ctx, &head).map_err(|e| format!("forward_batch head: {e:?}"))?;
    let logits = gpu
        .download_f32(&state.logits)
        .map_err(|e| format!("forward_batch download logits: {e:?}"))?;

    for t in [
        x,
        normed,
        fq,
        fk,
        fv,
        attn_out,
        o,
        ffn_x_rot,
        router_logits,
        topk_idx,
        topk_w,
        gate,
        up,
        rot,
        down_exp,
        dense_gate,
        dense_up,
        dense_act,
        pos_array,
        expert_token_counts,
        expert_offsets,
        sorted_slot_index,
        inverse_perm,
        expert_tile_ids,
        y_gate_up_grouped,
        y_down_grouped,
        x_f16,
    ] {
        gpu.free_tensor(t).ok();
    }
    Ok(logits)
}
