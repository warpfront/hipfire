// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Forward pass: gated attention, split-eps sandwich RMSNorm, silu SwiGLU.
//!
//! Per-token pipeline (see `lib.rs`):
//!   x = embed(token)                 // NO sqrt(dim) scale (only Gemma does)
//!   for each layer:
//!     residual = x
//!     n1 = rmsnorm(x, input_layernorm, eps=1e-5) -> tmp
//!     q = q_proj(n1); k = k_proj(n1); v = v_proj(n1); gate = gate_proj(n1)
//!     q = rmsnorm_batched(q, ones, head_dim, 1e-5) // scale-less QK-norm
//!     k = rmsnorm_batched(k, ones, head_dim, 1e-5)
//!     q *= qk_scale_factor (3.87)   // Do NOT pre-scale by sqrt(head_dim)
//!     RoPE only if layer_rope_theta != 0 (copy cohere2moe shape)
//!     kv write + attention_q8_0_kv_swa(window=2048 sliding / 0 full)
//!     attn_out *= sigmoid(gate) via gpu.sigmoid_mul_f32  BEFORE o_proj
//!     tmp = o_proj(attn_out); tmp = rmsnorm(tmp, post_attention_ln, 1e-8)
//!     x = residual + tmp
//!     residual = x
//!     n2 = rmsnorm(x, pre_feedforward_ln, 1e-5) -> tmp
//!     gate_ffn = gate_proj(n2); up = up_proj(n2)
//!     hidden = silu(gate_ffn) * up
//!     ffn_out = down_proj(hidden); ffn_out = rmsnorm(ffn_out, post_ffn_ln, 1e-8)
//!     x = residual + ffn_out
//!   x = rmsnorm(x, final_norm, 1e-5) -> tmp
//!   logits = lm_head(tmp); logits *= output_multiplier; logits = softcap(logits, 20)

use crate::config::GlimmerConfig;
use crate::glimmer::{
    device_capture_audit_enabled, GlimmerState, GlimmerWeights, GLIMMER_MAX_SPEC_BLOCK,
};
use crate::program;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::execute_steps;
use hipfire_dispatch::pipeline::sandwich::{
    gemm_rows, windowed_q8_partials_elems, windowed_q8_rows_flash, windowed_q8_rows_origin, Ladder,
    WindowedQ8RowsFlash, WindowedQ8Tiled, WINDOWED_Q8_TILED_MIN_SEQ,
};
use hipfire_dispatch::pipeline::{GemvInput, RowsRoute};
use hipfire_runtime::llama::EmbeddingFormat;
use rdna_compute::{DType, Gpu, GpuTensor};

// Opt-out for fusing the FFN activation and MagnumQuant G256 rotation.
pub(crate) fn fused_silu_rotate_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_GLIMMER_FUSED_SILU_ROTATE").as_deref() != Ok("0")
}

/// `HIPFIRE_GLIMMER_WMMA_FULL=0` keeps batched attention on the F32 scalar
/// kernels instead of the query-tiled F16 WMMA flash kernels (relative L2
/// ~1e-3; greedy text measured identical on Muse Glimmer).
pub(crate) fn wmma_flash_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_GLIMMER_WMMA_FULL")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(true)
}

// ───────────────────── Shared rotation gate ─────────────────────
// HIPFIRE_GLIMMER_SHARED_ROT default ON (=1 or unset), =0 selects old path.
// Mirrors Gemma4's attn_input_qkv and Qwen35's run_fa_layer_body precedent.
pub(crate) fn shared_rot_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_GLIMMER_SHARED_ROT").as_deref() != Ok("0")
}

fn batched_lm_head_enabled() -> bool {
    !matches!(
        hipfire_config::developer_var("HIPFIRE_GLIMMER_BATCHED_LM_HEAD")
            .ok()
            .as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

/// Chunk size for batched prefill: 192 for prompts >= 512 tokens, else 256.
/// Overridable via `HIPFIRE_GLIMMER_PREFILL_CHUNK` (clamped to 1-512); the
/// rationale for the split is on the function body below.
///
/// VRAM cost (transient per-chunk scratch):
///   dim=6656, q_dim=4096, kv_dim=256, hidden=19968, n_layers=52
///   B=128: ~ 70 MB, B=256: ~140 MB, B=512: ~280 MB
///   Computed as sum of batched tensors:
///   x/residual/nrm/x_rot/o_out/normed ~ B*dim*6 *4 bytes
///   q/attn_gate/attn_out/o_rot ~ B*q_dim*4 *4
///   gate/up/ffn_hidden/down_rot ~ B*hidden*4 *4
///   plus small k/v/pos. Model weights are 15.5 GB on 16 GB card leaving
///   ~500 MB headroom, so B=256 (140 MB) is safe, B=512 (280 MB) is still
///   within budget but closer to limit; B=128 is more conservative.
///   Both defaults (192 and 256) sit well inside that headroom.
pub fn glimmer_prefill_chunk_size(prompt_len: usize) -> usize {
    if let Some(v) = hipfire_config::developer_var("HIPFIRE_GLIMMER_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        return v.clamp(1, 512);
    }
    // 192 for anything long enough to fill at least two Muse tiles, else 256.
    //
    // The Muse residual sibling requires batch % 192 == 0, so only chunks of
    // exactly 192 (or 384) reach it; a ragged tail silently falls back to the
    // shared kernel. 192 also lands the shared batch-tile heuristic on BT12
    // with exact tiling, which 256 cannot reach (256 % 192 != 0 -> BT8).
    //
    // Measured on gfx1201, median of 3 fresh processes, all byte-identical:
    //    270 tok: 752 -> 743 tok/s  (-1.2%)  <- why short prompts keep 256
    //   1080 tok: 725 -> 783 tok/s  (+8.0%)
    //   4241 tok: 539 -> 569 tok/s  (+5.6%)
    // At 270 the split is one 192 tile plus a 78-token tail, and the extra
    // chunk's fixed cost outweighs the single tile's win. From ~512 tokens up
    // there are at least two full tiles and 192 pulls clearly ahead.
    if prompt_len >= 512 {
        192
    } else {
        256
    }
}

// ───────────────────────────── Decode ─────────────────────────────

/// Decode one token; returns the full logits vector.
pub fn decode_step(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    embed_lookup(cfg, weights, state, gpu, token_id)?;
    decode_step_body(cfg, weights, state, gpu, position)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("glimmer: download logits: {e:?}"))
}
/// On-device sampled decode: same pipeline as [`decode_step`] but samples on
/// the GPU and returns `(token, new_rng)` with only an 8-byte D2H.
///
/// This exists so AR decode pays an 8-byte D2H instead of `vocab_size * 4`
/// bytes (~808 KB at 202048) per token. Greedy (`temp <= 1e-6`) routes to
/// `argmax_f32` to stay byte-identical with the previous host-side path.
pub fn decode_step_sampled(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    rng_state: u32,
) -> Result<(u32, u32), String> {
    embed_lookup(cfg, weights, state, gpu, token_id)?;
    decode_step_body(cfg, weights, state, gpu, position)?;
    if temp <= 1e-6 {
        let tok = gpu
            .argmax_f32(&state.logits, cfg.vocab_size)
            .map_err(|e| format!("glimmer: argmax: {e:?}"))?;
        return Ok((tok, rng_state));
    }
    let top_p_eff = if top_p <= 0.0 || top_p > 1.0 {
        1.0
    } else {
        top_p
    };
    gpu.sample_top_p_pf(
        &state.logits,
        &state.sample_out,
        &state.sample_out,
        cfg.vocab_size,
        temp,
        top_p_eff,
        rng_state,
        0,
        1.0,
        0.0,
        0.0,
        top_k,
        None,
    )
    .map_err(|e| format!("glimmer: sample: {e:?}"))
}

pub(crate) fn embed_lookup(
    _cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    token_id: u32,
) -> Result<(), String> {
    let dim = _cfg.dim;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu
            .embedding_lookup_hfq4g256(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("glimmer: embed hfq4g256: {e:?}"))?,
        EmbeddingFormat::HFQ4G128 => gpu
            .embedding_lookup_hfq4g128(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("glimmer: embed hfq4g128: {e:?}"))?,
        EmbeddingFormat::Q8_0 => gpu
            .embedding_lookup_q8(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("glimmer: embed q8: {e:?}"))?,
        EmbeddingFormat::F32 => gpu
            .embedding_lookup(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("glimmer: embed f32: {e:?}"))?,
        EmbeddingFormat::Q4K => return Err("glimmer: Q4K embedding format unsupported".to_string()),
        EmbeddingFormat::Ggml(_) => return Err(
            "GGUF-native embedding tables are served by the dense qwen35 decode/prefill route only"
                .to_string(),
        ),
    }
    // Scale-less RMSNorm over the embedding.
    //
    // HF wraps the table in `MuseGlimmerTextNormedEmbedding`:
    //     forward(ids) = embed_norm(Embedding::forward(ids))
    // with `MuseGlimmerRMSNorm(eps=config.rms_norm_eps, with_scale=False)`.
    // Upstream explicitly does NOT fold this into the embedding matrix because
    // the DFlash path needs to embed without it, so it runs per lookup here.
    //
    // There is no Gemma-style sqrt(dim) embed_scale in Glimmer — this norm is
    // what takes its place, and omitting it leaves the residual stream at the
    // wrong magnitude for every downstream layer.
    gpu.rmsnorm_f32(
        &state.x,
        &state.embed_norm_ones,
        &state.x,
        _cfg.rms_norm_eps,
    )
    .map_err(|e| format!("glimmer: embed_norm: {e:?}"))?;
    Ok(())
}

/// Batched embed + scale-less embed_norm into `x` `[B*dim]`.
///
/// Returns `Ok(true)` when the format has a batched lookup kernel
/// (HFQ4G256 / HFQ4G128 / Q8_0) and the full path ran. Returns `Ok(false)`
/// for F32/Q4K so the caller keeps its exact sequential loop / error.
/// `token_ids` is a device buffer of `B` i32 native-endian token IDs.
fn embed_lookup_batched(
    gpu: &mut Gpu,
    weights: &GlimmerWeights,
    state: &GlimmerState,
    x: &GpuTensor,
    token_ids: &GpuTensor,
    b: usize,
    dim: usize,
    rms_eps: f32,
) -> Result<bool, String> {
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu
            .embedding_lookup_hfq4g256_batched(&weights.embed_tokens, x, token_ids, b, dim)
            .map_err(|e| format!("glimmer: embed hfq4g256 batched: {e:?}"))?,
        EmbeddingFormat::HFQ4G128 => gpu
            .embedding_lookup_hfq4g128_batched(&weights.embed_tokens, x, token_ids, b, dim)
            .map_err(|e| format!("glimmer: embed hfq4g128 batched: {e:?}"))?,
        EmbeddingFormat::Q8_0 => gpu
            .embedding_lookup_q8_batched(&weights.embed_tokens, x, token_ids, b, dim)
            .map_err(|e| format!("glimmer: embed q8 batched: {e:?}"))?,
        EmbeddingFormat::F32 | EmbeddingFormat::Q4K => return Ok(false),
        EmbeddingFormat::Ggml(_) => return Err(
            "GGUF-native embedding tables are served by the dense qwen35 decode/prefill route only"
                .to_string(),
        ),
    }
    gpu.rmsnorm_batched(x, &state.embed_norm_ones, x, b, dim, rms_eps)
        .map_err(|e| format!("glimmer: embed_norm batched: {e:?}"))?;
    Ok(true)
}

fn decode_step_body(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    stage_decode_position(gpu, cfg, state, position)?;
    let ctx = DispatchCtx::new(gpu);
    for layer in 0..cfg.n_layers {
        run_layer(gpu, &ctx, cfg, weights, state, layer)?;
    }
    state.n_tokens = position as usize + 1;
    run_head(gpu, &ctx, cfg, weights, state)
}

/// Stage `position` for the layer programs: the device position scalar, the
/// tiled-attention position copy and partials once the context reaches
/// [`WINDOWED_Q8_TILED_MIN_SEQ`], and the KV pages up to `position`.
pub(crate) fn stage_decode_position(
    gpu: &mut Gpu,
    cfg: &GlimmerConfig,
    state: &mut GlimmerState,
    position: u32,
) -> Result<(), String> {
    state.pos_host[0] = position as i32;
    // Heap-stable source: the Box outlives every copy below.
    let pos_bytes = unsafe { std::slice::from_raw_parts(state.pos_host.as_ptr() as *const u8, 4) };
    gpu.memcpy_htod_auto(&state.pos_buf, pos_bytes)
        .map_err(|e| format!("glimmer: htod pos: {e:?}"))?;
    let seq_len = position as usize + 1;
    if seq_len >= WINDOWED_Q8_TILED_MIN_SEQ {
        if state.prefill_flash_partials.is_none() {
            let need = windowed_q8_partials_elems(cfg.n_heads, cfg.head_dim, state.max_seq);
            let buf = gpu
                .alloc_tensor(&[need], DType::F32)
                .map_err(|e| format!("glimmer: decode flash partials: {e:?}"))?;
            state.prefill_flash_partials = Some(buf);
        }
        if state.decode_pos.is_none() {
            let t = gpu
                .alloc_tensor(&[1], DType::F32)
                .map_err(|e| format!("glimmer: decode pos tensor: {e:?}"))?;
            state.decode_pos = Some(t);
        }
        let t = state.decode_pos.as_ref().unwrap();
        gpu.hip
            .memcpy_htod(&t.buf, pos_bytes)
            .map_err(|e| format!("glimmer: decode pos htod: {e:?}"))?;
    }
    // No-op on the contiguous backend; maps VMM pages up to the new row.
    for kv in [&mut state.kv_sliding, &mut state.kv_full] {
        kv.ensure_mapped_capacity(gpu, seq_len)
            .map_err(|e| format!("glimmer: kv map: {e:?}"))?;
    }
    Ok(())
}

/// The tiled decode attention's staging, once [`stage_decode_position`] has
/// allocated it (from [`WINDOWED_Q8_TILED_MIN_SEQ`] on).
pub(crate) fn decode_tiled(state: &GlimmerState) -> Option<WindowedQ8Tiled<'_>> {
    match (&state.decode_pos, &state.prefill_flash_partials) {
        (Some(positions), Some(partials)) => Some(WindowedQ8Tiled {
            positions,
            partials,
        }),
        _ => None,
    }
}

/// Run decoder layer `layer`'s step program at the staged position.
fn run_layer(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &GlimmerState,
    layer: usize,
) -> Result<(), String> {
    let lw = &weights.layers[layer];
    let io = program::LayerIo::decode(state, decode_tiled(state));
    let steps = program::layer_steps(cfg, layer, lw, state, &io);
    execute_steps(gpu, ctx, &steps).map_err(|e| format!("glimmer L{layer}: {e:?}"))
}

/// Final norm and LM head into `state.logits`.
fn run_head(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &GlimmerState,
) -> Result<(), String> {
    let lm_head = weights.lm_head.dispatch_ref();
    let io = program::LayerIo::decode(state, None);
    let mut steps = vec![io.final_norm(cfg, weights)];
    steps.extend(program::lm_head_steps(
        cfg,
        &lm_head,
        &state.tmp,
        &state.logits,
    ));
    execute_steps(gpu, ctx, &steps).map_err(|e| format!("glimmer head: {e:?}"))
}

// ─── Block-parallel verify (DFlash) ──────────────────────────────────

/// Verify a block of `B` tokens at `position` in parallel-AR fashion,
/// returning per-position argmax. Leaves `state` advanced by `B` (KV written
/// at positions [position .. position+B-1], n_tokens bumped). Caller must
/// handle partial-accept rollback via `rollback_to`.
///
/// This is the B-position analogue of `decode_step`: same embed (WITH
/// embed_norm — this is the AR verify path, not the draft noise path),
/// same layer loop (via `glimmer_layer_decode`), same final norm +
/// lm_head * output_multiplier + softcap. No new kernels — it reuses the
/// per-token `decode_step_body` logic in a loop, but batches the lm_head
/// downloads per position so the caller sees per-position logits.
///
/// For `B==1` this is byte-identical to `decode_step` (modulo the extra
/// per-pos alloc). For `B>1` it is the speculative-verify primitive the
/// drafter's accept rule consumes at `spec.rs:130-183`.
pub fn verify_block(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    block: &[u32],
    position: u32,
    mut logits_out: Option<&mut Vec<f32>>,
) -> Result<Vec<u32>, String> {
    if logits_out.is_some() && !glimmer_batched_logits_available(cfg, weights) {
        return Err(format!(
            "glimmer verify_block: logits_out requires Q8 lm_head with batched path enabled (lm_head dtype {:?}, batched_lm_head_enabled={})",
            weights.lm_head.gpu_dtype,
            batched_lm_head_enabled()
        ));
    }
    let mut logits_sink = logits_out;
    if let Some(sink) = logits_sink.as_mut() {
        sink.clear();
    }
    let mut picks = Vec::with_capacity(block.len());
    for (i, &tok) in block.iter().enumerate() {
        let pos = position + i as u32;
        // embed (WITH norm — verify is AR) → layers → final_norm → lm_head → scale → softcap
        let logits = decode_step(cfg, weights, state, gpu, tok, pos)?;
        if let Some(sink) = logits_sink.as_mut() {
            sink.extend_from_slice(&logits);
        }
        let pick = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();
        picks.push(pick);
    }
    Ok(picks)
}

/// Roll back `state` to `target_pos` after a partial accept. For the
/// pure-attention Glimmer target (no DeltaNet recurrent state), the only
/// host KV cursor to truncate is `n_tokens`; KV entries beyond `target_pos`
/// will be overwritten by the next verify at that position (see
/// `spec.rs:312-319` stateless commit_prefix contract). No GPU memset needed.
///
/// When device hidden capture is enabled, the log must be Idle; if its
/// committed end differs from `target_pos`, it is rewound first (refusing
/// rewinds that would expose already-evicted rows). Only then is
/// `n_tokens` published. Rejects `target_pos > n_tokens`.
pub fn rollback_to(state: &mut GlimmerState, target_pos: usize) -> Result<(), String> {
    if target_pos > state.n_tokens {
        return Err(format!(
            "glimmer rollback_to: target_pos {target_pos} > n_tokens {}",
            state.n_tokens
        ));
    }
    if let Some(log) = state.target_hidden_log_mut() {
        if !log.stage_is_idle() {
            return Err("glimmer rollback_to: device hidden log stage not Idle".into());
        }
        if log.committed_abs_end() != target_pos {
            log.rewind_to(target_pos)?;
        }
    }
    state.n_tokens = target_pos;
    Ok(())
}

/// Raw embedding lookup without `embed_norm`, used for DFlash draft noise.
///
/// The target AR path applies embedding RMSNorm. The diffusion assistant
/// instead consumes the unnormalized target embedding by model contract.
pub fn embed_raw(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    token_id: u32,
) -> Result<Vec<f32>, String> {
    let dim = cfg.dim;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu
            .embedding_lookup_hfq4g256(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("glimmer embed_raw hfq4g256: {e:?}"))?,
        EmbeddingFormat::HFQ4G128 => gpu
            .embedding_lookup_hfq4g128(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("glimmer embed_raw hfq4g128: {e:?}"))?,
        EmbeddingFormat::Q8_0 => gpu
            .embedding_lookup_q8(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("glimmer embed_raw q8: {e:?}"))?,
        EmbeddingFormat::F32 => gpu
            .embedding_lookup(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("glimmer embed_raw f32: {e:?}"))?,
        EmbeddingFormat::Q4K => {
            return Err("glimmer embed_raw: Q4K unsupported".into());
        }
        EmbeddingFormat::Ggml(_) => return Err(
            "GGUF-native embedding tables are served by the dense qwen35 decode/prefill route only"
                .into(),
        ),
    }
    let host = gpu
        .download_f32(&state.x)
        .map_err(|e| format!("glimmer embed_raw download: {e:?}"))?;
    Ok(host[..dim].to_vec())
}

/// Private capture backend for shared host/device forward paths.
///
/// Host keeps the existing sorted-layer download + position-major `hidden_out`
/// append. Device scatters into the installed device hidden log using the
/// log's validated `layer_to_slot` order (never re-sorted).
enum CaptureBackend<'a> {
    Host {
        layers: &'a [usize],
        hidden_out: &'a mut Vec<f32>,
    },
    Device,
}

/// Commit the accepted device-verify prefix into the hidden ring.
///
/// Requires an installed device log in `VerifyReady`. Does not touch
/// `state.n_tokens` (left provisional); caller follows with
/// [`rollback_to`] to the new committed end when the accept is partial.
pub fn commit_device_verify_capture(
    state: &mut GlimmerState,
    gpu: &Gpu,
    keep_rows: usize,
) -> Result<(), String> {
    let log = state.target_hidden_log_mut().ok_or_else(|| {
        "glimmer: commit_device_verify_capture requires device hidden log".to_string()
    })?;
    log.commit_verified_prefix(gpu, keep_rows)
}

/// Drop any open device-capture transaction without touching the committed ring.
pub fn abort_device_capture_stage(state: &mut GlimmerState) {
    if let Some(log) = state.target_hidden_log_mut() {
        log.abort_stage();
    }
}

/// Decode one token and capture residual hidden at `capture_layers` (0-based).
/// Captured hidden is appended to `hidden_out` as `capture_layers.len()*dim` f32
/// in the order of `capture_layers` (which must be sorted ascending like
/// [1,13,25,37,49] from dflash_extract_layer_ids(52,5)). The capture is the
/// post-layer residual `x` (same tensor the target's `target_hidden` concat
/// expects per dflash.rs:target_hidden contract). Uses `gpu.download_f32`
/// per captured layer — slower but correct for the speculative path; prefill
/// should use a batched variant if hot.
///
/// This is the env-off / host-capture backend. Prefer
/// [`decode_step_with_device_capture`] when the device log is installed.
pub fn decode_step_with_capture(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture_layers: &[usize],
    hidden_out: &mut Vec<f32>,
) -> Result<Vec<f32>, String> {
    decode_step_capture_impl(
        cfg,
        weights,
        state,
        gpu,
        token_id,
        position,
        CaptureBackend::Host {
            layers: capture_layers,
            hidden_out,
        },
    )
}

/// Decode one token, scattering configured extract layers into the device
/// hidden log. Requires `state.device_hidden_capture_enabled()`.
pub fn decode_step_with_device_capture(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    decode_step_capture_impl(
        cfg,
        weights,
        state,
        gpu,
        token_id,
        position,
        CaptureBackend::Device,
    )
}

/// On-device sampled decode that also advances the device hidden log.
///
/// Runs [`decode_step_with_device_capture`] first so target KV and the device
/// hidden log both advance in one forward, then samples from resident
/// `state.logits` with the same temp/top_p/top_k/rng logic as
/// [`decode_step_sampled`]. The logits `Vec` from the capture API is discarded:
/// this preserves correctness but still pays one vocab D2H; a later measured
/// refactor can make the path resident-only.
pub fn decode_step_sampled_with_device_capture(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    rng_state: u32,
) -> Result<(u32, u32), String> {
    let _logits = decode_step_with_device_capture(cfg, weights, state, gpu, token_id, position)?;
    if temp <= 1e-6 {
        let tok = gpu
            .argmax_f32(&state.logits, cfg.vocab_size)
            .map_err(|e| format!("glimmer: argmax: {e:?}"))?;
        return Ok((tok, rng_state));
    }
    let top_p_eff = if top_p <= 0.0 || top_p > 1.0 {
        1.0
    } else {
        top_p
    };
    gpu.sample_top_p_pf(
        &state.logits,
        &state.sample_out,
        &state.sample_out,
        cfg.vocab_size,
        temp,
        top_p_eff,
        rng_state,
        0,
        1.0,
        0.0,
        0.0,
        top_k,
        None,
    )
    .map_err(|e| format!("glimmer: sample: {e:?}"))
}

/// On-device sampled decode that also advances host hidden capture history.
///
/// Runs [`decode_step_with_capture`] first so target KV and the host
/// `hidden_out` both advance in one forward, then samples from resident
/// `state.logits` with the same temp/top_p/top_k/rng logic as
/// [`decode_step_sampled`]. The logits `Vec` from the capture API is discarded:
/// host fallback retains one vocab D2H for now; a later measured refactor can
/// make the path resident-only.
pub fn decode_step_sampled_with_capture(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture_layers: &[usize],
    hidden_out: &mut Vec<f32>,
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    rng_state: u32,
) -> Result<(u32, u32), String> {
    let _logits = decode_step_with_capture(
        cfg,
        weights,
        state,
        gpu,
        token_id,
        position,
        capture_layers,
        hidden_out,
    )?;
    if temp <= 1e-6 {
        let tok = gpu
            .argmax_f32(&state.logits, cfg.vocab_size)
            .map_err(|e| format!("glimmer: argmax: {e:?}"))?;
        return Ok((tok, rng_state));
    }
    let top_p_eff = if top_p <= 0.0 || top_p > 1.0 {
        1.0
    } else {
        top_p
    };
    gpu.sample_top_p_pf(
        &state.logits,
        &state.sample_out,
        &state.sample_out,
        cfg.vocab_size,
        temp,
        top_p_eff,
        rng_state,
        0,
        1.0,
        0.0,
        0.0,
        top_k,
        None,
    )
    .map_err(|e| format!("glimmer: sample: {e:?}"))
}

fn decode_step_capture_impl(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    mut capture: CaptureBackend<'_>,
) -> Result<Vec<f32>, String> {
    let restore_pos = position as usize;
    let device = matches!(&capture, CaptureBackend::Device);

    // Host: HashSet membership; append order follows ascending layer_idx walk
    // (matches sorted capture_layers). Device: clone layer_to_slot once so the
    // layer loop never holds a log borrow across glimmer_layer_decode.
    let host_layers: Option<std::collections::HashSet<usize>> = match &capture {
        CaptureBackend::Host { layers, .. } => {
            let mut s = std::collections::HashSet::new();
            for &l in *layers {
                s.insert(l);
            }
            Some(s)
        }
        CaptureBackend::Device => None,
    };
    let device_slots: Option<Vec<Option<usize>>> = if device {
        if state.n_tokens != restore_pos {
            return Err(format!(
                "glimmer device decode: n_tokens {} != position {restore_pos}",
                state.n_tokens
            ));
        }
        let log = state
            .target_hidden_log_mut()
            .ok_or_else(|| "glimmer device decode: device hidden log not enabled".to_string())?;
        if log.committed_abs_end() != restore_pos {
            return Err(format!(
                "glimmer device decode: log end {} != position {restore_pos}",
                log.committed_abs_end()
            ));
        }
        log.begin_decode(restore_pos)?;
        Some(log.layer_to_slot().to_vec())
    } else {
        None
    };

    let body = (|| {
        embed_lookup(cfg, weights, state, gpu, token_id)?;
        stage_decode_position(gpu, cfg, state, position)?;
        let ctx = DispatchCtx::new(gpu);
        for layer_idx in 0..cfg.n_layers {
            run_layer(gpu, &ctx, cfg, weights, state, layer_idx)?;
            if let Some(caps) = &host_layers {
                if caps.contains(&layer_idx) {
                    let host = gpu
                        .download_f32(&state.x)
                        .map_err(|e| format!("glimmer capture download L{layer_idx}: {e:?}"))?;
                    if let CaptureBackend::Host { hidden_out, .. } = &mut capture {
                        hidden_out.extend_from_slice(&host[..cfg.dim]);
                    }
                }
            }
            if let Some(slots) = &device_slots {
                if let Some(ci) = slots.get(layer_idx).copied().flatten() {
                    let x = &state.x;
                    let log = state
                        .target_hidden_log
                        .as_mut()
                        .ok_or_else(|| "glimmer device decode: log missing mid-step".to_string())?;
                    log.scatter_layer_rows(gpu, ci, x, 1)?;
                }
            }
        }
        run_head(gpu, &ctx, cfg, weights, state)?;
        let logits = gpu
            .download_f32(&state.logits)
            .map_err(|e| format!("glimmer capture: download logits: {e:?}"))?;
        if device {
            let log = state
                .target_hidden_log_mut()
                .ok_or_else(|| "glimmer device decode: log missing at commit".to_string())?;
            log.finish_decode_and_commit(gpu)?;
        }
        // Publish n_tokens only after successful final logits (and device commit).
        state.n_tokens = restore_pos + 1;
        Ok(logits)
    })();

    if device && body.is_err() {
        abort_device_capture_stage(state);
        state.n_tokens = restore_pos;
    }
    body
}

/// True iff a `Some(logits_out)` request can be served (Q8/F32 lm_head + batched
/// path enabled). Mirrors the exact condition used inside the function below.
pub fn glimmer_batched_logits_available(cfg: &GlimmerConfig, weights: &GlimmerWeights) -> bool {
    let _ = cfg;
    (weights.lm_head.gpu_dtype == DType::Q8_0
        || weights.lm_head.gpu_dtype == DType::F32
        || weights.lm_head.gpu_dtype == DType::BF16)
        && batched_lm_head_enabled()
}

/// Shared lm_head routine for draft and verify. `hidden_batch` is [B*hidden]
/// already RMSNormed (the lm_head input). Returns argmax picks for rows
/// [0..batch) in order. Q8/F32/BF16 heads (batched path enabled) run one
/// batched GEMM over all rows with one scale/softcap and a GPU batched argmax;
/// every other head, or `HIPFIRE_GLIMMER_BATCHED_LM_HEAD=0`, runs the decode
/// head per row. This is the single source of truth for draft and verify so
/// they cannot diverge on near-ties due to different accumulation order (the
/// Q8 vs per-row flip that halved tau on the Q8-head artifact).
///
/// `logits_out`: when `Some`, receives `batch * cfg.vocab_size` f32 target
/// logits in row-major [pos][vocab] order. Q8/F32/BF16 lm_head only; returns Err if
/// `Some` is passed while the batched path is unavailable.
pub fn glimmer_lm_head_picks(
    gpu: &mut Gpu,
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    hidden_batch: &GpuTensor,
    batch: usize,
    logits_scratch: &GpuTensor,
    logits_batch: &GpuTensor,
    argmax_batch: &GpuTensor,
    mut logits_out: Option<&mut Vec<f32>>,
) -> Result<Vec<u32>, String> {
    let dim = cfg.dim;
    let vocab = cfg.vocab_size;
    if batch == 0 {
        if let Some(sink) = logits_out.as_mut() {
            sink.clear();
        }
        return Ok(Vec::new());
    }
    if batch > GLIMMER_MAX_SPEC_BLOCK {
        return Err(format!(
            "glimmer lm_head_picks: batch {batch} exceeds GLIMMER_MAX_SPEC_BLOCK {GLIMMER_MAX_SPEC_BLOCK}"
        ));
    }
    let batched = glimmer_batched_logits_available(cfg, weights);
    if logits_out.is_some() && !batched {
        return Err(format!(
            "glimmer lm_head_picks: logits_out requires Q8/F32/BF16 lm_head with batched path enabled (lm_head dtype {:?}, batched_lm_head_enabled={})",
            weights.lm_head.gpu_dtype,
            batched_lm_head_enabled()
        ));
    }
    let ctx = DispatchCtx::new(gpu);
    let lm_head = weights.lm_head.dispatch_ref();
    let argmax_view = argmax_batch.sub_offset(0, batch);
    if batched {
        // Persistent logits buffer: a cold hipMalloc of this ~12.9 MB is slow
        // and synchronizing (69 ms vs 7.6 ms for the same GEMM).
        let logits_b = logits_batch.sub_offset(0, batch * vocab);
        // Q8 / F32 / BF16 heads: none rotates, so the rotation scratch is
        // never written.
        gemm_rows(
            gpu,
            &ctx,
            &lm_head,
            GemvInput::Raw(hidden_batch),
            &logits_b,
            hidden_batch,
            None,
            batch,
            Ladder::Route(RowsRoute::Verify),
        )
        .and_then(|()| {
            execute_steps(
                gpu,
                &ctx,
                &program::logit_steps(cfg, &logits_b, batch * vocab),
            )
        })
        .map_err(|e| format!("glimmer lm_head_picks batched: {e:?}"))?;
        if let Some(sink) = logits_out.as_mut() {
            sink.clear();
            sink.resize(batch * vocab, 0.0);
            let bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(sink.as_mut_ptr() as *mut u8, batch * vocab * 4)
            };
            gpu.hip
                .memcpy_dtoh(bytes, &logits_b.buf)
                .map_err(|e| format!("glimmer lm_head_picks logits dtoh: {e:?}"))?;
        }
        gpu.argmax_f32_batched(&logits_b, &argmax_view, vocab, batch)
            .map_err(|e| format!("glimmer lm_head_picks argmax_batched: {e:?}"))?;
    } else {
        // Re-streams lm_head B times. Vocab-width MQ4 batched GEMM stays off:
        // the gfx12 path faults at this output width (same as gemma4). Each
        // row's argmax lands in argmax_batch[i]; one D2H after the loop.
        for i in 0..batch {
            let hidden_row = hidden_batch.sub_offset(i * dim, dim);
            execute_steps(
                gpu,
                &ctx,
                &program::lm_head_steps(cfg, &lm_head, &hidden_row, logits_scratch),
            )
            .map_err(|e| format!("glimmer lm_head_picks row {i}: {e:?}"))?;
            gpu.argmax_f32_batched(logits_scratch, &argmax_batch.sub_offset(i, 1), vocab, 1)
                .map_err(|e| format!("glimmer lm_head_picks argmax row {i}: {e:?}"))?;
        }
    }
    let mut host_idx = vec![0i32; batch];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(host_idx.as_mut_ptr() as *mut u8, batch * 4) };
    gpu.hip
        .memcpy_dtoh(bytes, &argmax_view.buf)
        .map_err(|e| format!("glimmer lm_head_picks argmax dtoh: {e:?}"))?;
    Ok(host_idx.into_iter().map(|v| v as u32).collect())
}

// ─── Batched verify core ─────────────────────────────────────────────

/// Block verify that also captures hidden at `capture_layers` for each
/// position in the block, appending to `hidden_out` (same order as
/// decode_step_with_capture). Used by the DFlash spec loop to keep
/// `bundle.target_hidden_host` in sync with the committed prefix.
///
/// Batched: ONE forward over B positions reading each weight ONCE for all
/// rows (memory-bound decode → ~single AR cost per window instead of B×).
/// For each of q/k/v/attn_gate (sharing one input) and gate/up (sharing
/// another) we issue one batched GEMM over B rows. For MQ4G256 we pre-rotate
/// the batched input once (`rotate_x_mq_batched_for`) and call the
/// prerotated batched kernel, mirroring gemma4/qwen35. Where no batched
/// kernel exists we fall back to per-row `weight_gemv` (explicit, no
/// approximation) — f32 etc. The lm_head over B rows is vocab 202048
/// (~700 MB). Gemma4 uses batched Q8 but keeps per-row for MQ4 because the
/// batched MQ4 path faults at that output width on gfx12; Glimmer's lm_head
/// is MQ4G256 so we keep the same per-row scalar fallback explicitly (caps
/// the win: lm_head re-streams B times).
///
/// Attention respects per-layer sliding/full split and NoPE (theta 0) exactly
/// as the single path: sliding window 2048 vs full 0, rope skipped when
/// layer_rope_theta==0. Batched attention uses `attention_q8_0_kv_batched_masked`
/// with a per-row additive mask [B×seq_len] (0 causal / -inf masked) and
/// block_start=0 block_cols=seq_len giving full per-row control for both
/// layer types, matching gemma4's mask construction.
///
/// Capture contract (host): `hidden_out` is extended with B rows of
/// `capture_layers.len()*dim` f32, position-major in ascending tap order
/// (pos0: [L1,L13,...], pos1: [L1,L13,...], ...), so the caller's
/// `truncate(hidden_before+keep_rows*row_elems)` keeps the committed prefix.
///
/// This is the env-off / host-capture backend. Prefer
/// [`verify_block_with_device_capture`] when the device log is installed.
pub fn verify_block_with_capture(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    block: &[u32],
    position: u32,
    capture_layers: &[usize],
    hidden_out: &mut Vec<f32>,
    mut logits_out: Option<&mut Vec<f32>>,
) -> Result<Vec<u32>, String> {
    verify_block_capture_impl(
        cfg,
        weights,
        state,
        gpu,
        block,
        position,
        CaptureBackend::Host {
            layers: capture_layers,
            hidden_out,
        },
        logits_out,
    )
}

/// Block verify that scatters extract-layer residuals into the device hidden
/// log. Stages all B rows (`finish_verify` → VerifyReady); the caller later
/// commits the accepted prefix via [`commit_device_verify_capture`].
///
/// Sets provisional `state.n_tokens = position + B`. On error aborts the stage
/// and restores `n_tokens` to `position`.
pub fn verify_block_with_device_capture(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    block: &[u32],
    position: u32,
    mut logits_out: Option<&mut Vec<f32>>,
) -> Result<Vec<u32>, String> {
    verify_block_capture_impl(
        cfg,
        weights,
        state,
        gpu,
        block,
        position,
        CaptureBackend::Device,
        logits_out,
    )
}

fn verify_block_capture_impl(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    block: &[u32],
    position: u32,
    mut capture: CaptureBackend<'_>,
    mut logits_out: Option<&mut Vec<f32>>,
) -> Result<Vec<u32>, String> {
    let b = block.len();
    if b == 0 {
        if let Some(sink) = logits_out.as_mut() {
            sink.clear();
        }
        return Ok(Vec::new());
    }
    if b > GLIMMER_MAX_SPEC_BLOCK {
        return Err(format!(
            "glimmer verify_block: B={b} exceeds GLIMMER_MAX_SPEC_BLOCK {GLIMMER_MAX_SPEC_BLOCK}"
        ));
    }
    let restore_pos = position as usize;
    let device = matches!(&capture, CaptureBackend::Device);
    let mut cap = RowsCapture::plan(cfg, state, &capture, b, restore_pos, "verify")?;
    let s = RowsScratch::alloc(gpu, cfg, b)?;
    if device {
        if let Err(e) = state
            .target_hidden_log_mut()
            .ok_or_else(|| "glimmer device verify: log missing at begin".to_string())
            .and_then(|log| log.begin_verify(restore_pos, b))
        {
            s.free(gpu);
            return Err(e);
        }
    }

    // Every `?` below unwinds to the single scratch release after it.
    let body_result: Result<Vec<u32>, String> = (|| {
        rows_layers(
            cfg,
            weights,
            state,
            gpu,
            &s,
            block,
            restore_pos,
            RowsRoute::Verify,
            &mut cap,
        )?;
        if cap.cnt > 0 {
            if let CaptureBackend::Host { hidden_out, .. } = &mut capture {
                hidden_out.extend_from_slice(&cap.buf);
            }
        }
        {
            let ctx = DispatchCtx::new(gpu);
            let io = rows_io(state, &s, b, RowsRoute::Verify, restore_pos);
            execute_steps(gpu, &ctx, &[io.final_norm(cfg, weights)])
                .map_err(|e| format!("glimmer verify final norm: {e:?}"))?;
        }
        // Final norm + lm_head per position → picks (shared with draft).
        let picks = glimmer_lm_head_picks(
            gpu,
            cfg,
            weights,
            &s.x,
            b,
            &state.logits,
            &state.logits_batch,
            &state.argmax_batch,
            logits_out,
        )?;

        // Publish provisional n_tokens only after successful final logits.
        // Device: finish_verify leaves the ring end unchanged (VerifyReady).
        // Audit: bit-compare full staging vs host dual-collect, then stash host
        // for post-commit ring check in commit_device_verify_capture.
        if device {
            let log = state
                .target_hidden_log_mut()
                .ok_or_else(|| "glimmer device verify: log missing at finish".to_string())?;
            if !cap.buf.is_empty() {
                let (start_abs, stored_rows) = match log.stage() {
                    crate::glimmer::HiddenStageState::Writing(s) => (s.start_abs, s.stored_rows),
                    other => {
                        return Err(format!(
                            "glimmer device-capture audit verify: expected Writing, got {other:?}"
                        ));
                    }
                };
                log.audit_compare_staging_to_host(
                    gpu,
                    "verify_full_staging",
                    &cap.buf,
                    start_abs,
                    /*stored_row0=*/ 0,
                    stored_rows,
                )?;
                log.stash_audit_host(std::mem::take(&mut cap.buf));
            }
            log.finish_verify()?;
        }
        state.n_tokens = restore_pos + b;

        Ok(picks)
    })();

    s.free(gpu);
    if device && body_result.is_err() {
        abort_device_capture_stage(state);
        state.n_tokens = restore_pos;
    }
    body_result
}

// ─── Batched rows: prefill chunks and verify blocks ──────────────────────

/// Activation scratch of one batched forward over `rows` rows, allocated per
/// call (prefill chunks and verify blocks are not the per-kernel hot loop).
pub(crate) struct RowsScratch {
    pub(crate) x: GpuTensor,
    residual: GpuTensor,
    pub(crate) normed: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    attn_gate: GpuTensor,
    attn_out: GpuTensor,
    gate_ffn: GpuTensor,
    up_ffn: GpuTensor,
    ffn_hidden: GpuTensor,
    pub(crate) ffn_out: GpuTensor,
    rotation: GpuTensor,
    positions: GpuTensor,
    shifted_positions: GpuTensor,
    token_ids: GpuTensor,
}

impl RowsScratch {
    /// Per-row elements of each tensor, in field order.
    fn widths(cfg: &GlimmerConfig) -> [usize; 16] {
        let (dim, q_dim, kv_dim, hidden) = (cfg.dim, cfg.q_dim(), cfg.kv_dim(), cfg.hidden_dim);
        let widest = dim.max(q_dim).max(hidden);
        [
            dim, dim, dim, q_dim, kv_dim, kv_dim, q_dim, q_dim, hidden, hidden, hidden, dim,
            widest, 1, 1, 1,
        ]
    }

    /// Every tensor or none: a failure part-way frees what it already holds,
    /// so a retried request cannot leak it.
    pub(crate) fn alloc(gpu: &mut Gpu, cfg: &GlimmerConfig, b: usize) -> Result<Self, String> {
        let sizes = Self::widths(cfg);
        let mut taken = Vec::with_capacity(sizes.len());
        for n in sizes {
            match gpu.alloc_tensor(&[b * n], DType::F32) {
                Ok(t) => taken.push(t),
                Err(e) => {
                    for t in taken {
                        gpu.free_tensor(t).ok();
                    }
                    return Err(format!("glimmer rows scratch alloc: {e:?}"));
                }
            }
        }
        let mut t = taken.into_iter();
        let mut next = || t.next().expect("one tensor per size");
        Ok(Self {
            x: next(),
            residual: next(),
            normed: next(),
            q: next(),
            k: next(),
            v: next(),
            attn_gate: next(),
            attn_out: next(),
            gate_ffn: next(),
            up_ffn: next(),
            ffn_hidden: next(),
            ffn_out: next(),
            rotation: next(),
            positions: next(),
            shifted_positions: next(),
            token_ids: next(),
        })
    }

    /// Views of the first `b` rows of every tensor, so kernels that size
    /// themselves from a tensor's length see exactly `b` rows, as with a
    /// scratch allocated for `b` rows. `b` must not exceed the allocation.
    pub(crate) fn rows(&self, cfg: &GlimmerConfig, b: usize) -> Self {
        let [x, residual, normed, q, k, v, attn_gate, attn_out, gate_ffn, up_ffn, ffn_hidden, ffn_out, rotation, positions, shifted_positions, token_ids] =
            Self::widths(cfg).map(|w| b * w);
        Self {
            x: self.x.sub_offset(0, x),
            residual: self.residual.sub_offset(0, residual),
            normed: self.normed.sub_offset(0, normed),
            q: self.q.sub_offset(0, q),
            k: self.k.sub_offset(0, k),
            v: self.v.sub_offset(0, v),
            attn_gate: self.attn_gate.sub_offset(0, attn_gate),
            attn_out: self.attn_out.sub_offset(0, attn_out),
            gate_ffn: self.gate_ffn.sub_offset(0, gate_ffn),
            up_ffn: self.up_ffn.sub_offset(0, up_ffn),
            ffn_hidden: self.ffn_hidden.sub_offset(0, ffn_hidden),
            ffn_out: self.ffn_out.sub_offset(0, ffn_out),
            rotation: self.rotation.sub_offset(0, rotation),
            positions: self.positions.sub_offset(0, positions),
            shifted_positions: self.shifted_positions.sub_offset(0, shifted_positions),
            token_ids: self.token_ids.sub_offset(0, token_ids),
        }
    }

    pub(crate) fn free(self, gpu: &mut Gpu) {
        for t in [
            self.x,
            self.residual,
            self.normed,
            self.q,
            self.k,
            self.v,
            self.attn_gate,
            self.attn_out,
            self.gate_ffn,
            self.up_ffn,
            self.ffn_hidden,
            self.ffn_out,
            self.rotation,
            self.positions,
            self.shifted_positions,
            self.token_ids,
        ] {
            gpu.free_tensor(t).ok();
        }
    }
}

/// Hidden-row capture of one batched forward. Host: the tapped layers'
/// rows, position-major in ascending tap order. Device: the installed log's
/// validated layer→slot map (never re-sorted), plus the host rows when the
/// device-capture audit dual-collects.
struct RowsCapture {
    index: Vec<Option<usize>>,
    buf: Vec<f32>,
    cnt: usize,
    device_slots: Option<Vec<Option<usize>>>,
}

impl RowsCapture {
    fn plan(
        cfg: &GlimmerConfig,
        state: &GlimmerState,
        capture: &CaptureBackend<'_>,
        b: usize,
        restore_pos: usize,
        what: &str,
    ) -> Result<Self, String> {
        let index_of = |layers: &mut dyn Iterator<Item = usize>| {
            let mut index = vec![None; cfg.n_layers];
            for (ci, li) in layers.enumerate() {
                if li < cfg.n_layers {
                    index[li] = Some(ci);
                }
            }
            index
        };
        match capture {
            CaptureBackend::Host { layers, .. } => {
                let mut sorted = layers.to_vec();
                sorted.sort_unstable();
                let cnt = sorted.len();
                Ok(Self {
                    index: index_of(&mut sorted.into_iter()),
                    buf: vec![0.0; b * cnt * cfg.dim],
                    cnt,
                    device_slots: None,
                })
            }
            CaptureBackend::Device => {
                if state.n_tokens != restore_pos {
                    return Err(format!(
                        "glimmer device {what}: n_tokens {} != position {restore_pos}",
                        state.n_tokens
                    ));
                }
                let log = state.target_hidden_log().ok_or_else(|| {
                    format!("glimmer device {what}: device hidden log not enabled")
                })?;
                if log.committed_abs_end() != restore_pos {
                    return Err(format!(
                        "glimmer device {what}: log end {} != position {restore_pos}",
                        log.committed_abs_end()
                    ));
                }
                if !log.stage_is_idle() {
                    return Err(format!("glimmer device {what}: log stage not Idle"));
                }
                let cnt = log.num_extract();
                let (index, buf) = if device_capture_audit_enabled() {
                    let index = index_of(&mut log.capture_layers().iter().copied());
                    (index, vec![0.0; b * cnt * cfg.dim])
                } else {
                    (vec![None; cfg.n_layers], Vec::new())
                };
                Ok(Self {
                    index,
                    buf,
                    cnt,
                    device_slots: Some(log.layer_to_slot().to_vec()),
                })
            }
        }
    }
}

/// The layer-program buffers of a batched forward over `rows` rows at
/// `position`.
pub(crate) fn rows_io<'a>(
    state: &'a GlimmerState,
    s: &'a RowsScratch,
    rows: usize,
    route: RowsRoute,
    position: usize,
) -> program::LayerIo<'a> {
    program::LayerIo {
        x: &s.x,
        residual: &s.residual,
        normed: &s.normed,
        rotation: &s.rotation,
        q: &s.q,
        k: &s.k,
        v: &s.v,
        attn_gate: &s.attn_gate,
        attn_out: &s.attn_out,
        gate_ffn: &s.gate_ffn,
        up_ffn: &s.up_ffn,
        ffn_hidden: &s.ffn_hidden,
        ffn_out: &s.ffn_out,
        pos_buf: &state.pos_buf,
        rows,
        route,
        positions: Some(&s.positions),
        position,
        bf16: state.prefill_bf16_scratch.as_ref(),
        tiled: None,
        rows_flash: state
            .prefill_flash_partials
            .as_ref()
            .map(|partials| WindowedQ8RowsFlash {
                shifted_positions: &s.shifted_positions,
                partials,
            }),
    }
}

fn htod_i32(gpu: &mut Gpu, t: &GpuTensor, values: impl Iterator<Item = i32>) -> Result<(), String> {
    let bytes: Vec<u8> = values.flat_map(i32::to_ne_bytes).collect();
    gpu.hip
        .memcpy_htod(&t.buf, &bytes)
        .map_err(|e| format!("glimmer rows htod: {e:?}"))
}

/// Stage a batched forward of `tokens` at `position`: row positions and token
/// ids; past the window, the shifted positions and the flash partials of the
/// prefill attention; the KV pages the rows write.
pub(crate) fn stage_rows(
    gpu: &mut Gpu,
    cfg: &GlimmerConfig,
    state: &mut GlimmerState,
    s: &RowsScratch,
    tokens: &[u32],
    position: usize,
    route: RowsRoute,
) -> Result<(), String> {
    let b = tokens.len();
    htod_i32(gpu, &s.positions, (0..b).map(|i| (position + i) as i32))?;
    htod_i32(gpu, &s.token_ids, tokens.iter().map(|&t| t as i32))?;
    if route == RowsRoute::Prefill {
        let window = cfg.sliding_window;
        let origin = windowed_q8_rows_origin(position, b, window);
        if origin > 0 {
            htod_i32(
                gpu,
                &s.shifted_positions,
                (0..b).map(|i| (position + i - origin) as i32),
            )?;
        }
        let flash =
            windowed_q8_rows_flash(position, b, window) || windowed_q8_rows_flash(position, b, 0);
        if flash && state.prefill_flash_partials.is_none() {
            let need = windowed_q8_partials_elems(cfg.n_heads, cfg.head_dim, state.max_seq);
            let buf = gpu
                .alloc_tensor(&[need], DType::F32)
                .map_err(|e| format!("glimmer prefill flash partials alloc: {e:?}"))?;
            state.prefill_flash_partials = Some(buf);
        }
    }
    // No-op on the contiguous backend; maps VMM pages up to the last row.
    for kv in [&mut state.kv_sliding, &mut state.kv_full] {
        kv.ensure_mapped_capacity(gpu, position + b)
            .map_err(|e| format!("glimmer rows kv map: {e:?}"))?;
    }
    Ok(())
}

/// Embedding + scale-less embed_norm of the staged token ids into `s.x`.
pub(crate) fn embed_rows(
    gpu: &mut Gpu,
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &GlimmerState,
    s: &RowsScratch,
    tokens: &[u32],
) -> Result<(), String> {
    let (b, dim, eps) = (tokens.len(), cfg.dim, cfg.rms_norm_eps);
    if embed_lookup_batched(gpu, weights, state, &s.x, &s.token_ids, b, dim, eps)? {
        return Ok(());
    }
    // The helper declines only F32 and Q4K; Q4K has no lookup at all.
    if matches!(weights.embd_format, EmbeddingFormat::Q4K) {
        return Err("glimmer rows: Q4K embed unsupported".to_string());
    }
    for (i, &tok) in tokens.iter().enumerate() {
        let row = s.x.sub_offset(i * dim, dim);
        gpu.embedding_lookup(&weights.embed_tokens, &row, tok, dim)
            .map_err(|e| format!("glimmer rows embed f32: {e:?}"))?;
        gpu.rmsnorm_f32(&row, &state.embed_norm_ones, &row, eps)
            .map_err(|e| format!("glimmer rows embed_norm: {e:?}"))?;
    }
    Ok(())
}

/// Stage and embed `tokens` at `position`, then run every decoder layer's
/// program over them, capturing the rows after each tapped layer.
#[allow(clippy::too_many_arguments)]
fn rows_layers(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    s: &RowsScratch,
    tokens: &[u32],
    position: usize,
    route: RowsRoute,
    cap: &mut RowsCapture,
) -> Result<(), String> {
    let (b, dim) = (tokens.len(), cfg.dim);
    stage_rows(gpu, cfg, state, s, tokens, position, route)?;
    embed_rows(gpu, cfg, weights, state, s, tokens)?;
    let ctx = DispatchCtx::new(gpu);
    for layer in 0..cfg.n_layers {
        {
            let lw = &weights.layers[layer];
            let io = rows_io(state, s, b, route, position);
            let steps = program::layer_steps(cfg, layer, lw, state, &io);
            execute_steps(gpu, &ctx, &steps)
                .map_err(|e| format!("glimmer rows L{layer}: {e:?}"))?;
        }
        if let Some(ci) = cap.index[layer] {
            let host = gpu
                .download_f32(&s.x)
                .map_err(|e| format!("glimmer rows capture L{layer}: {e:?}"))?;
            for row in 0..b {
                let dst = (row * cap.cnt + ci) * dim;
                cap.buf[dst..dst + dim].copy_from_slice(&host[row * dim..(row + 1) * dim]);
            }
        }
        if let Some(ci) = cap
            .device_slots
            .as_ref()
            .and_then(|slots| slots.get(layer).copied().flatten())
        {
            let log = state
                .target_hidden_log_mut()
                .ok_or_else(|| "glimmer device capture: log missing mid-step".to_string())?;
            log.scatter_layer_rows(gpu, ci, &s.x, b)?;
        }
    }
    Ok(())
}

// ─── Chunked batched prefill ─────────────────────────────────────────────

/// Batched forward of one prefill chunk (B = chunk.len()): attention is
/// causal within the chunk and attends all prior KV, sliding layers within
/// their window.
///
/// If `need_logits` is true, final norm + single-row lm_head is run on
/// the LAST position only (not over the whole chunk). This avoids
/// re-streaming the 202048-wide lm_head weight B times; we stream it
/// once instead of B times. Say what we did: prefill discards
/// intermediate logits, so we compute vocab only for position `pos+B-1`.
fn prefill_chunk_batched(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    chunk: &[u32],
    position: u32,
    mut capture: CaptureBackend<'_>,
    need_logits: bool,
) -> Result<Option<Vec<f32>>, String> {
    let b = chunk.len();
    // Persistent BF16 staging of BF16 projections, sized to the 512-row chunk cap.
    let bf16_needed = 512 * cfg.hidden_dim;
    if state.prefill_bf16_scratch.is_none() {
        let t = gpu
            .zeros(&[bf16_needed], DType::BF16)
            .map_err(|e| format!("glimmer prefill bf16 scratch alloc: {e:?}"))?;
        state.prefill_bf16_scratch = Some(t);
    } else if state.prefill_bf16_scratch.as_ref().unwrap().numel() < bf16_needed {
        // Should not happen (hidden_dim fixed), but handle if config changes.
        let _ = state
            .prefill_bf16_scratch
            .take()
            .map(|t| gpu.free_tensor(t));
        let t = gpu
            .zeros(&[bf16_needed], DType::BF16)
            .map_err(|e| format!("glimmer prefill bf16 scratch realloc: {e:?}"))?;
        state.prefill_bf16_scratch = Some(t);
    }
    if b == 0 {
        return Ok(None);
    }
    if b > 512 {
        return Err(format!("glimmer prefill: B={b} exceeds chunk cap 512"));
    }
    let restore_pos = position as usize;
    let device = matches!(&capture, CaptureBackend::Device);
    let mut cap = RowsCapture::plan(cfg, state, &capture, b, restore_pos, "prefill")?;
    let s = RowsScratch::alloc(gpu, cfg, b)?;
    if device {
        if let Err(e) = state
            .target_hidden_log_mut()
            .ok_or_else(|| "glimmer device prefill: log missing at begin".to_string())
            .and_then(|log| log.begin_prefill(restore_pos, b))
        {
            s.free(gpu);
            return Err(e);
        }
    }

    // Every `?` below unwinds to the single scratch release after it.
    let body_result: Result<Option<Vec<f32>>, String> = (|| {
        rows_layers(
            cfg,
            weights,
            state,
            gpu,
            &s,
            chunk,
            restore_pos,
            RowsRoute::Prefill,
            &mut cap,
        )?;
        if cap.cnt > 0 {
            if let CaptureBackend::Host { hidden_out, .. } = &mut capture {
                hidden_out.extend_from_slice(&cap.buf);
            }
        }

        // ── Final norm + lm_head: ONLY last row when needed ──
        let last_logits: Option<Vec<f32>> = if need_logits {
            let ctx = DispatchCtx::new(gpu);
            let lm_head = weights.lm_head.dispatch_ref();
            let io = rows_io(state, &s, b, RowsRoute::Prefill, restore_pos);
            let last = s.x.sub_offset((b - 1) * cfg.dim, cfg.dim);
            let mut steps = vec![io.final_norm(cfg, weights)];
            steps.extend(program::lm_head_steps(cfg, &lm_head, &last, &state.logits));
            execute_steps(gpu, &ctx, &steps).map_err(|e| format!("glimmer prefill head: {e:?}"))?;
            Some(
                gpu.download_f32(&state.logits)
                    .map_err(|e| format!("glimmer prefill download logits: {e:?}"))?,
            )
        } else {
            None
        };

        // Commit device capture and publish n_tokens only after successful work
        // (including optional final logits).
        // Audit: compare retained suffix staging before commit, then wrapped
        // committed rows after successful stage→ring publish.
        if device {
            let log = state
                .target_hidden_log_mut()
                .ok_or_else(|| "glimmer device prefill: log missing at commit".to_string())?;
            let audit_host = if !cap.buf.is_empty() {
                let (start_abs, stored_row0, stored_rows, total_rows) = match log.stage() {
                    crate::glimmer::HiddenStageState::Writing(s) => {
                        (s.start_abs, s.stored_row0, s.stored_rows, s.total_rows)
                    }
                    other => {
                        return Err(format!(
                            "glimmer device-capture audit prefill: expected Writing, got {other:?}"
                        ));
                    }
                };
                log.audit_compare_staging_to_host(
                    gpu,
                    "prefill_suffix_staging",
                    &cap.buf,
                    start_abs,
                    stored_row0,
                    stored_rows,
                )?;
                Some((
                    std::mem::take(&mut cap.buf),
                    start_abs,
                    stored_row0,
                    stored_rows,
                    total_rows,
                ))
            } else {
                None
            };
            log.finish_prefill_and_commit(gpu)?;
            if let Some((host, start_abs, stored_row0, stored_rows, total_rows)) = audit_host {
                let new_end = start_abs + total_rows;
                let abs_start = new_end - stored_rows;
                let row_elems = log.row_elems();
                let host_off = stored_row0 * row_elems;
                let host_len = stored_rows * row_elems;
                if let Err(e) = log.audit_compare_committed_rows(
                    gpu,
                    "prefill_post_commit_ring",
                    &host[host_off..host_off + host_len],
                    abs_start,
                    stored_rows,
                ) {
                    // Watermarks already published; fail-closed.
                    log.poison_for_audit();
                    return Err(format!("{e}; session requires reset"));
                }
            }
        }
        state.n_tokens = restore_pos + b;

        Ok(last_logits)
    })();

    s.free(gpu);
    if device && body_result.is_err() {
        abort_device_capture_stage(state);
        state.n_tokens = restore_pos;
    }
    body_result
}

/// Chunked batched prefill entry point. Processes `prompt` in chunks of
/// `HIPFIRE_GLIMMER_PREFILL_CHUNK` (default 256, env overridable, 1-512).
/// Reuses the batched layer machinery from `verify_block_with_capture`
/// (same `proj_gemm_batched` helpers, same fused rotate, same attention)
/// but over larger B. Returns the last prompt token's logits, which seed
/// greedy decode. When `capture_layers` is non-empty, `hidden_out` is
/// extended with `B*len(capture_layers)*dim` floats per chunk in ascending
/// tap-layer order, position-major — identical to the per-token
/// `decode_step_with_capture` contract.
///
/// Intermediate chunks skip the lm_head entirely (only the final position's
/// logits are needed to seed generation). This is the "do NOT run it over
/// the whole chunk" optimization: we run one `weight_gemv` on the last row
/// instead of a batched GEMM over B rows, streaming the 700 MB lm_head once
/// instead of B times.
pub fn prefill_with_capture(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    prompt: &[u32],
    start_pos: u32,
    capture_layers: &[usize],
    hidden_out: &mut Vec<f32>,
) -> Result<Vec<f32>, String> {
    prefill_capture_impl(
        cfg,
        weights,
        state,
        gpu,
        prompt,
        start_pos,
        CaptureBackend::Host {
            layers: capture_layers,
            hidden_out,
        },
    )
}

/// Chunked batched prefill that scatters extract-layer residuals into the
/// device hidden log. Requires `state.device_hidden_capture_enabled()` and
/// `state.n_tokens == start_pos` with matching log committed end.
pub fn prefill_with_device_capture(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    prompt: &[u32],
    start_pos: u32,
) -> Result<Vec<f32>, String> {
    prefill_capture_impl(
        cfg,
        weights,
        state,
        gpu,
        prompt,
        start_pos,
        CaptureBackend::Device,
    )
}

fn prefill_capture_impl(
    cfg: &GlimmerConfig,
    weights: &GlimmerWeights,
    state: &mut GlimmerState,
    gpu: &mut Gpu,
    prompt: &[u32],
    start_pos: u32,
    capture: CaptureBackend<'_>,
) -> Result<Vec<f32>, String> {
    if prompt.is_empty() {
        return Err("glimmer prefill: empty prompt".to_string());
    }
    let chunk_size = glimmer_prefill_chunk_size(prompt.len());
    let mut last_logits: Option<Vec<f32>> = None;
    let mut pos = start_pos;
    let mut offset = 0usize;

    // Split arms so Host's `&mut Vec` can be reborrowed per chunk without
    // trying to move a mutable reference out of an outer enum.
    match capture {
        CaptureBackend::Host { layers, hidden_out } => {
            while offset < prompt.len() {
                let end = (offset + chunk_size).min(prompt.len());
                let chunk = &prompt[offset..end];
                let is_last = end == prompt.len();
                let logits_opt = prefill_chunk_batched(
                    cfg,
                    weights,
                    state,
                    gpu,
                    chunk,
                    pos,
                    CaptureBackend::Host { layers, hidden_out },
                    is_last,
                )?;
                if let Some(l) = logits_opt {
                    last_logits = Some(l);
                }
                pos += chunk.len() as u32;
                offset = end;
            }
        }
        CaptureBackend::Device => {
            while offset < prompt.len() {
                let end = (offset + chunk_size).min(prompt.len());
                let chunk = &prompt[offset..end];
                let is_last = end == prompt.len();
                let logits_opt = prefill_chunk_batched(
                    cfg,
                    weights,
                    state,
                    gpu,
                    chunk,
                    pos,
                    CaptureBackend::Device,
                    is_last,
                )?;
                if let Some(l) = logits_opt {
                    last_logits = Some(l);
                }
                pos += chunk.len() as u32;
                offset = end;
            }
        }
    }

    last_logits.ok_or_else(|| "glimmer prefill: no logits produced".to_string())
}
