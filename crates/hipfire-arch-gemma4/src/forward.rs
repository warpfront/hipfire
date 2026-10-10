// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 dense forward driver: embedding, per-layer-input staging, hipGraph
//! capture and the batched verify entry. Every decoder layer runs as the
//! declarative step program in `program.rs` (`rows == 1` decode, `rows > 1`
//! batched prefill / verify); the layer math is, per token:
//!
//!   x = embed(token) * sqrt(dim)
//!   for each layer (sandwich RMSNorm around BOTH attn and FFN):
//!     residual = x
//!     n1 = input_layernorm(x)
//!     q = q_proj(n1); k = k_proj(n1)
//!       full + attention_k_eq_v: V = copy of k BEFORE k_norm, then weight-less
//!         RMSNorm on V (ones buffer); sliding: V = v_proj(n1)
//!     per-head q_norm / k_norm over head_dim; q *= sqrt(head_dim) (Gemma
//!       scale = 1.0 vs the kernel's 1/sqrt)
//!     RoPE: sliding → rope_f32(theta 10000, full rotate-half);
//!            full   → rope_partial_halved_f32(theta 1e6, n_rot = head_dim*0.25/2)
//!     KV write (Q8); attention_q8_0_kv_swa(window 1024 sliding / 0 full)
//!     attn = o_proj(attn_out); attn = post_attention_layernorm(attn)
//!     x = residual + attn
//!     residual = x
//!     n2 = pre_feedforward_layernorm(x)
//!     ffn = gelu_tanh(gate_proj(n2)) * up_proj(n2); ffn = down_proj(ffn)
//!     ffn = post_feedforward_layernorm(ffn)
//!     x = residual + ffn
//!     x *= layer_scalar
//!   x = norm(x); logits = lm_head(x); logits = logit_softcap(logits, 30)
//!
//! All RMSNorm here is plain `x * w` (baked at load — see `load_norm`).
//!
//! ## Decode perf levers (gemma4-12B MQ4, hiptrx gfx1201)
//!
//! Decode is memory-bandwidth-bound: rocprofv3 attributes ~79% of GPU time to
//! the weight-reading GEMVs (FFN gate_up/down + attention q/k/v/o + lm_head).
//! All three levers below are byte-identical to the eager baseline (validated
//! over multiple prompts) and stack monotonically:
//!
//!   * `HIPFIRE_GEMMA4_GRAPH` (default ON; set =0 to disable) — hipGraph
//!     48-layer body + lm_head (`decode_step_with_graph`). +2.6%.
//!   * fused FFN — fold pre-FFN rmsnorm+FWHT into one launch then gate+up into
//!     one (`fused_rmsnorm_rotate_mq` + `fused_gate_up_hfq4g256`; MQ4G256 bytes
//!     are HFQ4G256-compatible given a pre-rotated input). +1.0–1.2%.
//!   * fused Q8 q+k projections in one launch (`fused_gate_up_q8_0`, shared
//!     rmsnorm input). +1.1%.
//!
//! Full stack: 46.8 → 50.7 tok/s (+8.3%). The 70–75 tok/s target is NOT
//! reachable via fusion/graph alone — it would require reading ~40% fewer
//! weight bytes/token (lower-bit FFN quant or MQ4 attention), and MQ4 attention
//! is known to break coherence. No Q8 fused-QKV *decode* kernel exists
//! (`gemm_qkv_q8_0_wmma` is batched-prefill WMMA only), so q+k is the most that
//! fuses on the Q8 attention path.

use crate::config::Gemma4Config;
use crate::gemma4::{Gemma4State, Gemma4Weights, LayerWeights, GEMMA4_FORWARD_BATCH_MAX};
use crate::program::{
    eager_layer_kv, head, Geometry, LayerRefs, LayerScratch, PleScratch, ProgramBinding, Resident,
    RowActivations, RowBuffers, RowWidths,
};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::execute_steps;
use hipfire_runtime::llama::{weight_gemv, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Decode one token (eager); returns the full logits vector. Used for prefill,
/// the warm pass, and as the `HIPFIRE_GEMMA4_GRAPH=0` fallback.
pub fn decode_step(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    prepare_token_inputs(cfg, weights, state, gpu, token_id)?;
    stage_position(state, gpu, position)?;
    decode_step_body(cfg, weights, state, gpu, position, None)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("gemma4: download logits: {e:?}"))
}

/// Stage `position` into the device scalar every position-reading kernel
/// uses. Inside a hipGraph capture the copy is recorded and re-reads
/// `state.pos_host` on replay.
fn stage_position(state: &mut Gemma4State, gpu: &mut Gpu, position: u32) -> Result<(), String> {
    state.pos_host[0] = position as i32;
    let pos_bytes = unsafe { std::slice::from_raw_parts(state.pos_host.as_ptr() as *const u8, 4) };
    gpu.memcpy_htod_auto(&state.pos_buf.buf, pos_bytes)
        .map_err(|e| format!("gemma4: htod pos: {e:?}"))
}

/// The token-dependent prologue of one decode step: embedding (× √dim) into
/// `state.x` and, on E-series checkpoints, the per-layer inputs.
pub fn prepare_token(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    token_id: u32,
) -> Result<(), String> {
    prepare_token_inputs(cfg, weights, state, gpu, token_id)
}

/// The position-independent body of one decode step (every layer, final
/// norm, LM head, softcap into `state.logits`). It reads the token from
/// `state.x` ([`prepare_token`]) and the position from `state.pos_buf`, which
/// the caller staged; it issues no copies, so a launch recorder sees the
/// whole step.
pub fn decode_body(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    decode_step_body(cfg, weights, state, gpu, position, None)
}

/// Whether replaying a captured decode reproduces direct decode exactly. The
/// Q8_0 expert `down_proj` accumulates with atomics, whose scheduling-order
/// sum is not exact; a launch recorder must see direct launches.
fn graph_exact(weights: &Gemma4Weights, gpu: &Gpu) -> bool {
    !gpu.replay.is_recording()
        && weights
            .layers
            .iter()
            .filter_map(LayerWeights::moe)
            .all(|moe| moe.down_dtype == DType::HFQ4G128)
}

/// Decode one token, appending each layer's post-residual hidden state (pre
/// final-norm) to `capture[layer]` — used by the oracle dumper. Eager only
/// (the per-layer D2H downloads are incompatible with graph capture).
pub fn decode_step_capture(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture: &mut [Vec<f32>],
) -> Result<(), String> {
    prepare_token_inputs(cfg, weights, state, gpu, token_id)?;
    stage_position(state, gpu, position)?;
    decode_step_body(cfg, weights, state, gpu, position, Some(capture))
}

/// Decode one token via hipGraph capture/replay. **Default ON**
/// (`HIPFIRE_GEMMA4_GRAPH=0` to disable; +2.9% vs eager, byte-identical).
/// The 48-layer body + final-norm +
/// lm_head are captured once and replayed per token, recovering the per-token
/// host launch overhead. This is the biggest launch-bound lever on gemma4
/// decode (~720 kernel launches/token).
///
/// Capture-safety invariants (mirrors the proven MiniMax / DeepSeek-V4 path):
///   - token_id is per-token → embedding lookup + √dim scale run OUTSIDE the
///     capture (token_id is baked into the embedding kernarg).
///   - position is per-token → staged via `state.pos_host` (stable `Box`); the
///     captured `memcpy_htod_auto` re-reads it on every replay.
///   - attention launch geometry is sized for `state.max_seq` (constant), not
///     the live seq_len, so the baked grid/shared-mem stays valid as the KV
///     length grows (the kernel reads the true length from `pos_buf[0]+1`).
pub fn decode_step_with_graph(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    use std::sync::OnceLock;
    static GRAPH_ENV: OnceLock<Option<bool>> = OnceLock::new();
    let env_override = *GRAPH_ENV.get_or_init(|| {
        match hipfire_config::developer_var("HIPFIRE_GEMMA4_GRAPH")
            .ok()
            .as_deref()
        {
            Some("1") => Some(true),
            Some("0") => Some(false),
            _ => None,
        }
    });
    // The captured path is the default. Set HIPFIRE_GEMMA4_GRAPH=0 to retain
    // the eager fallback for diagnostics.
    let graph_on = env_override.unwrap_or(true);
    if !graph_on || !graph_exact(weights, gpu) {
        return decode_step(cfg, weights, state, gpu, token_id, position);
    }

    // Warmup: first decode after a fresh load runs eager (JITs kernels + settles
    // DPM) and drops any stale graph so the next call captures fresh for THIS
    // model's weight pointers / device buffers.
    if !state.ar_warmed_up {
        state.ar_warmed_up = true;
        gpu.graphs.graph_exec = None;
        return decode_step(cfg, weights, state, gpu, token_id, position);
    }

    // Capture + replay both need an explicit (non-null) stream.
    if gpu.active_stream.is_none() {
        let s = gpu
            .hip
            .stream_create()
            .map_err(|e| format!("gemma4 graph: stream_create: {e:?}"))?;
        gpu.active_stream = Some(s);
    }

    // Embedding lookup + √dim scale OUTSIDE the captured region — token_id is
    // baked into the embedding kernarg. Runs on the active stream, ordered
    // before the captured body that reads `state.x`.
    prepare_token_inputs(cfg, weights, state, gpu, token_id)?;

    if gpu.graphs.graph_exec.is_none() {
        // ── Capture phase ──────────────────────────────────────────────
        // The position copy is captured too, so the recorded memcpy node
        // re-reads pos_host on each replay.
        gpu.graphs
            .begin_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("gemma4 begin_graph_capture: {e:?}"))?;
        let body = stage_position(state, gpu, position)
            .and_then(|()| decode_step_body(cfg, weights, state, gpu, position, None));
        let end = gpu
            .graphs
            .end_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("gemma4 end_graph_capture: {e:?}"));
        if let Err(e) = body.and(end) {
            gpu.graphs.capture_mode = false;
            gpu.graphs.graph_destroy(&gpu.hip, gpu.device_id);
            return Err(e);
        }
        // Captured kernels were RECORDED, not run — launch once so this token's
        // logits actually get produced.
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("gemma4 graph_launch (capture): {e:?}"))?;
        eprintln!(
            "[gemma4 hipGraph] captured decode forward — {} kernarg blobs retained",
            gpu.graphs.ar_forward_blobs.len()
        );
    } else {
        // ── Replay phase ───────────────────────────────────────────────
        // Host-only update of the stable position source; the captured memcpy
        // re-reads it and propagates to pos_buf (read by rope / kv-write /
        // attention).
        state.pos_host[0] = position as i32;
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("gemma4 graph_launch (replay): {e:?}"))?;
    }
    state.n_tokens = position as usize + 1;

    // Logits download is outside the captured region (sync dtoh completes after
    // the captured kernels, which the device observes on the active stream).
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("gemma4 graph: download logits: {e:?}"))
}

/// Embedding lookup → x, then scale by sqrt(dim). Kept separate from the body
/// so the hipGraph path can run it OUTSIDE the captured region (token_id is
/// baked into the embedding kernarg).
fn embed_lookup(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    token_id: u32,
) -> Result<(), String> {
    use hipfire_runtime::llama::EmbeddingFormat;
    let dim = cfg.dim;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu
            .embedding_lookup_hfq4g256(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("gemma4: embed hfq4g256: {e:?}"))?,
        EmbeddingFormat::HFQ4G128 => gpu
            .embedding_lookup_hfq4g128(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("gemma4: embed hfq4g128: {e:?}"))?,
        EmbeddingFormat::Q8_0 => gpu
            .embedding_lookup_q8(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("gemma4: embed q8: {e:?}"))?,
        EmbeddingFormat::F32 => gpu
            .embedding_lookup(&weights.embed_tokens, &state.x, token_id, dim)
            .map_err(|e| format!("gemma4: embed f32: {e:?}"))?,
        EmbeddingFormat::Q4K => return Err("gemma4: Q4K embedding format unsupported".to_string()),
    }
    gpu.scale_f32(&state.x, cfg.embed_scale)
        .map_err(|e| format!("gemma4: embed scale: {e:?}"))?;
    Ok(())
}

fn embedding_lookup_to(
    gpu: &mut Gpu,
    format: hipfire_runtime::llama::EmbeddingFormat,
    table: &GpuTensor,
    dst: &GpuTensor,
    token_id: u32,
    dim: usize,
    label: &str,
) -> Result<(), String> {
    use hipfire_runtime::llama::EmbeddingFormat;
    match format {
        EmbeddingFormat::HFQ4G256 => gpu
            .embedding_lookup_hfq4g256(table, dst, token_id, dim)
            .map_err(|e| format!("gemma4: {label} hfq4g256: {e:?}")),
        EmbeddingFormat::HFQ4G128 => gpu
            .embedding_lookup_hfq4g128(table, dst, token_id, dim)
            .map_err(|e| format!("gemma4: {label} hfq4g128: {e:?}")),
        EmbeddingFormat::Q8_0 => gpu
            .embedding_lookup_q8(table, dst, token_id, dim)
            .map_err(|e| format!("gemma4: {label} q8: {e:?}")),
        EmbeddingFormat::F32 => gpu
            .embedding_lookup(table, dst, token_id, dim)
            .map_err(|e| format!("gemma4: {label} f32: {e:?}")),
        EmbeddingFormat::Q4K => Err(format!("gemma4: {label} Q4K embedding format unsupported")),
    }
}

fn embedding_lookup_batched_to(
    gpu: &mut Gpu,
    format: hipfire_runtime::llama::EmbeddingFormat,
    table: &GpuTensor,
    dst: &GpuTensor,
    token_ids: &GpuTensor,
    batch: usize,
    dim: usize,
    label: &str,
) -> Result<bool, String> {
    use hipfire_runtime::llama::EmbeddingFormat;
    let result = match format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256_batched(table, dst, token_ids, batch, dim)
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128_batched(table, dst, token_ids, batch, dim)
        }
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8_batched(table, dst, token_ids, batch, dim),
        EmbeddingFormat::F32 | EmbeddingFormat::Q4K => return Ok(false),
    };
    result
        .map(|_| true)
        .map_err(|e| format!("gemma4: {label} batched: {e:?}"))
}

fn has_batched_embedding_lookup(format: hipfire_runtime::llama::EmbeddingFormat) -> bool {
    use hipfire_runtime::llama::EmbeddingFormat;
    matches!(
        format,
        EmbeddingFormat::HFQ4G256 | EmbeddingFormat::HFQ4G128 | EmbeddingFormat::Q8_0
    )
}

fn prepare_token_inputs(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    token_id: u32,
) -> Result<(), String> {
    embed_lookup(cfg, weights, state, gpu, token_id)?;
    prepare_per_layer_inputs(cfg, weights, state, gpu, token_id)
}

fn prepare_per_layer_inputs(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    token_id: u32,
) -> Result<(), String> {
    let Some(ple) = weights.per_layer_input.as_ref() else {
        return Ok(());
    };
    let ple_dim = cfg.hidden_size_per_layer_input;
    if ple_dim == 0 {
        return Ok(());
    }
    if token_id as usize >= cfg.vocab_size_per_layer_input {
        return Err(format!(
            "gemma4: PLE token id {token_id} out of range for vocab_size_per_layer_input {}",
            cfg.vocab_size_per_layer_input
        ));
    }

    let packed_dim = cfg.n_layers * ple_dim;
    let token_inputs = state
        .ple_token_inputs
        .as_ref()
        .ok_or_else(|| "gemma4: missing ple_token_inputs scratch".to_string())?;
    let projection_all = state
        .ple_projection_all
        .as_ref()
        .ok_or_else(|| "gemma4: missing ple_projection_all scratch".to_string())?;

    embedding_lookup_to(
        gpu,
        ple.embd_format,
        &ple.embed_tokens,
        token_inputs,
        token_id,
        packed_dim,
        "ple embed",
    )?;
    gpu.scale_f32(token_inputs, (ple_dim as f32).sqrt())
        .map_err(|e| format!("gemma4: ple embed scale: {e:?}"))?;
    weight_gemv(gpu, &ple.model_projection, &state.x, projection_all)
        .map_err(|e| format!("gemma4: ple model_projection: {e}"))?;
    gpu.scale_f32(projection_all, (cfg.dim as f32).sqrt().recip())
        .map_err(|e| format!("gemma4: ple projection scale: {e:?}"))?;
    gpu.rmsnorm_batched(
        projection_all,
        &ple.projection_norm,
        projection_all,
        cfg.n_layers,
        ple_dim,
        cfg.norm_eps,
    )
    .map_err(|e| format!("gemma4: ple projection norm: {e:?}"))?;
    gpu.add_inplace_f32(projection_all, token_inputs)
        .map_err(|e| format!("gemma4: ple combine: {e:?}"))?;
    gpu.scale_f32(projection_all, 2.0f32.sqrt().recip())
        .map_err(|e| format!("gemma4: ple combine scale: {e:?}"))
}

fn prepare_per_layer_inputs_batched(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    gpu: &mut Gpu,
    tokens: &[u32],
    token_ids: Option<&GpuTensor>,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    token_inputs: &GpuTensor,
    projection_all: &GpuTensor,
) -> Result<(), String> {
    let Some(ple) = weights.per_layer_input.as_ref() else {
        return Ok(());
    };
    let ple_dim = cfg.hidden_size_per_layer_input;
    if ple_dim == 0 {
        return Ok(());
    }
    let packed_dim = cfg.n_layers * ple_dim;
    let b = tokens.len();
    for &token_id in tokens {
        if token_id as usize >= cfg.vocab_size_per_layer_input {
            return Err(format!(
                "gemma4 forward_batch: PLE token id {token_id} out of range for vocab_size_per_layer_input {}",
                cfg.vocab_size_per_layer_input
            ));
        }
    }
    let batched_embedding = if let Some(token_ids) = token_ids {
        embedding_lookup_batched_to(
            gpu,
            ple.embd_format,
            &ple.embed_tokens,
            token_inputs,
            token_ids,
            b,
            packed_dim,
            "batch ple embed",
        )?
    } else {
        false
    };
    if !batched_embedding {
        for (row, &token_id) in tokens.iter().enumerate() {
            let token_row = token_inputs.sub_offset(row * packed_dim, packed_dim);
            embedding_lookup_to(
                gpu,
                ple.embd_format,
                &ple.embed_tokens,
                &token_row,
                token_id,
                packed_dim,
                "batch ple embed",
            )?;
        }
    }

    if gpu.arch == "gfx1100"
        && gpu.flags.gemma4_ple_batched_prefill
        // Repeated WMMA rounding from both PLE probes can change long greedy
        // trajectories. Prefer the much larger branch win when both are set.
        && !gpu.flags.gemma4_ple_branch_batched_prefill
        && b > 1
        && ple.model_projection.gpu_dtype == DType::Q8_0
    {
        let ctx = DispatchCtx::new(gpu);
        hipfire_dispatch::pipeline::sandwich::gemm_rows(
            gpu,
            &ctx,
            &ple.model_projection.dispatch_ref(),
            x,
            projection_all,
            x_rot,
            b,
        )
        .map_err(|e| format!("gemma4 forward_batch ple model_projection: {e}"))?;
    } else {
        for row in 0..b {
            let x_row = x.sub_offset(row * cfg.dim, cfg.dim);
            let projection_row = projection_all.sub_offset(row * packed_dim, packed_dim);
            weight_gemv(gpu, &ple.model_projection, &x_row, &projection_row)
                .map_err(|e| format!("gemma4 forward_batch ple model_projection row {row}: {e}"))?;
        }
    }

    gpu.scale_f32(token_inputs, (ple_dim as f32).sqrt())
        .map_err(|e| format!("gemma4 forward_batch ple embed scale: {e:?}"))?;
    gpu.scale_f32(projection_all, (cfg.dim as f32).sqrt().recip())
        .map_err(|e| format!("gemma4 forward_batch ple projection scale: {e:?}"))?;
    gpu.rmsnorm_batched(
        projection_all,
        &ple.projection_norm,
        projection_all,
        b * cfg.n_layers,
        ple_dim,
        cfg.norm_eps,
    )
    .map_err(|e| format!("gemma4 forward_batch ple projection norm: {e:?}"))?;
    gpu.add_inplace_f32(projection_all, token_inputs)
        .map_err(|e| format!("gemma4 forward_batch ple combine: {e:?}"))?;
    gpu.scale_f32(projection_all, 2.0f32.sqrt().recip())
        .map_err(|e| format!("gemma4 forward_batch ple combine scale: {e:?}"))
}

fn eager_resident(state: &Gemma4State) -> Resident<'_> {
    Resident {
        kv_sliding: &state.kv_sliding,
        kv_full: &state.kv_full,
        pos_buf: &state.pos_buf.buf,
        v_norm_ones: &state.v_norm_ones,
        flash_partials: &state.q8_flash_partials,
    }
}

fn decode_step_body(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    position: u32,
    mut capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    if position as usize >= state.max_seq {
        return Err(format!(
            "gemma4: decode position {position} exceeds allocated KV capacity {}",
            state.max_seq
        ));
    }

    // The caller staged `position` into `state.pos_buf`.

    // Per-layer program: [SandwichAttention, SandwichMlp, PerLayerInput?, Scale?].
    let ctx = DispatchCtx::new(gpu);
    let mut steps = Vec::with_capacity(4);
    for layer_idx in 0..cfg.n_layers {
        steps.clear();
        let binding = ProgramBinding {
            geo: Geometry::eager(cfg),
            resident: eager_resident(state),
            rows: 1,
            position: position as usize,
            positions: None,
            scratch: LayerScratch::eager(state),
        };
        binding.layer(
            layer_idx,
            LayerRefs::eager(&weights.layers[layer_idx]),
            eager_layer_kv(cfg, state, layer_idx)?,
            &mut steps,
        )?;
        execute_steps(gpu, &ctx, &steps).map_err(|e| format!("gemma4 L{layer_idx}: {e}"))?;
        if let Some(cap) = capture.as_deref_mut() {
            let h = gpu
                .download_f32(&state.x)
                .map_err(|e| format!("gemma4 L{layer_idx}: capture download: {e:?}"))?;
            cap[layer_idx].extend_from_slice(&h);
        }
    }
    drop(steps);
    state.n_tokens = position as usize + 1;

    // Final norm → tied LM head → logit softcap.
    let lm_head = weights.lm_head.dispatch_ref();
    let mut head_steps = Vec::with_capacity(3);
    head(
        &Geometry::eager(cfg),
        &weights.final_norm,
        &lm_head,
        &state.x,
        &state.tmp,
        &state.logits,
        &mut head_steps,
    );
    execute_steps(gpu, &ctx, &head_steps).map_err(|e| format!("gemma4 head: {e}"))
}

// ════════════════════════════════════════════════════════════════════════
//  Batched verify forward — `forward_batch`
// ════════════════════════════════════════════════════════════════════════
//
// Verifies B tokens [tokens[0..B]] starting at absolute position `start_pos`
// in ONE pass (weights read once). The spec-decode keystone: returns the LAST
// token's logits, byte/argmax-identical to running B sequential `decode_step`
// calls, and leaves the two KV caches in the same state the sequential path
// would (so attention over history is correct for the next token).
//
// Structure mirrors the eager `decode_step_body` exactly, batched across B:
//   - embed lookup per token → x[B,dim] → ×√dim
//   - per layer (sliding / full), the same sandwich-norm + dual-RoPE +
//     k_eq_v pipeline, batched
//   - attention via `attention_q8_0_kv_batched_masked` with a per-row
//     causal+sliding-window additive mask ([B × seq_len], 0 = visible,
//     -inf = masked); block_start=0, block_cols=seq_len gives full per-row
//     control for BOTH layer types
//   - last row → final norm → tied lm_head → logit_softcap(30) → return logits
//
// ADDITIVE: does not touch `decode_step` / `decode_step_with_graph`. The eager
// path is unchanged.

/// Whether every projection used by batched prefill has a matching kernel
/// and both caches are Q8 (the batched attention writes Q8 rows). Other
/// loads remain on eager prefill.
pub fn supports_batched_prefill(weights: &Gemma4Weights, state: &Gemma4State) -> bool {
    let supports = |weight: &WeightTensor| {
        hipfire_dispatch::pipeline::sandwich::supports_batched_projection(weight.gpu_dtype)
    };
    state.kv_sliding.quant_q8
        && state.kv_full.quant_q8
        && weights.layers.iter().all(|layer| match layer {
            LayerWeights::Sliding(layer) => {
                supports(&layer.q_proj)
                    && supports(&layer.k_proj)
                    && supports(&layer.v_proj)
                    && supports(&layer.o_proj)
                    && supports(&layer.gate_proj)
                    && supports(&layer.up_proj)
                    && supports(&layer.down_proj)
            }
            LayerWeights::Full(layer) => {
                supports(&layer.q_proj)
                    && supports(&layer.k_proj)
                    && layer.v_proj.as_ref().map_or(true, supports)
                    && supports(&layer.o_proj)
                    && supports(&layer.gate_proj)
                    && supports(&layer.up_proj)
                    && supports(&layer.down_proj)
            }
        })
}

/// argmax over a logits row (spec-decode greedy per-position prediction).
fn argmax_f32_row(v: &[f32]) -> u32 {
    let mut bi = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i as u32;
        }
    }
    bi
}

fn checked_batch_seq_len(start_pos: usize, batch: usize, max_seq: usize) -> Result<usize, String> {
    let seq_len = start_pos
        .checked_add(batch)
        .ok_or_else(|| "gemma4 forward_batch: position overflow".to_string())?;
    if seq_len > max_seq {
        return Err(format!(
            "gemma4 forward_batch: positions [{start_pos}, {seq_len}) exceed allocated KV capacity {max_seq}"
        ));
    }
    Ok(seq_len)
}

/// Batched verify forward. See module-level note above. `tokens.len()` = B
/// (1..=`GEMMA4_FORWARD_BATCH_MAX`). Returns
/// the LAST token's logits. Side effect: writes both KV caches for positions
/// [start_pos, start_pos+B) and sets `state.n_tokens = start_pos + B`.
///
/// Thin wrapper over `forward_batch_spec` with both spec out-params off:
/// behaviour is BYTE-IDENTICAL to the original `forward_batch` (the eager /
/// verify path). All existing callers stay on this signature unchanged.
pub fn forward_batch(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: usize,
) -> Result<Vec<f32>, String> {
    forward_batch_spec(cfg, weights, state, gpu, tokens, start_pos, None, None)
}

/// Batched verify forward with optional spec-decode out-params (Part A of the
/// EAGLE wiring). Identical body to `forward_batch`; the two optional outputs
/// are computed ADDITIVELY after the layer loop and do not alter the returned
/// last-token logits, the KV writes, or `state.n_tokens`.
///
/// * `per_token_hidden_out` — when `Some`, receives each of the B positions'
///   POST-`model.norm` hidden ([B × dim], row-major). Row `i` is exactly the
///   hidden that the eager `decode_step` leaves in `state.tmp` after
///   `final_norm` (the lm_head input). Used to seed the drafter for the next
///   spec round at the accepted-bonus row.
/// * `per_pos_argmax_out` — when `Some`, receives the target's greedy argmax at
///   each block position ([B], `argmax_per_pos[i]` = the target's prediction
///   AFTER block position `i`). Computed by running the SAME lm_head + final
///   logit softcap the eager path uses, per row. Required for greedy accept.
///
/// When BOTH are `None` this is byte-identical to `forward_batch` (only the
/// last-row final-norm + lm_head + softcap + download runs, as before).
#[allow(clippy::too_many_arguments)]
pub fn forward_batch_spec(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: usize,
    per_token_hidden_out: Option<&GpuTensor>,
    per_pos_argmax_out: Option<&mut Vec<u32>>,
) -> Result<Vec<f32>, String> {
    forward_rows(
        cfg,
        weights,
        state,
        gpu,
        tokens,
        start_pos,
        per_token_hidden_out,
        per_pos_argmax_out,
        true,
    )
}

/// Prefill `tokens` from `start_pos` through the batched program in chunks of
/// [`GEMMA4_FORWARD_BATCH_MAX`] rows, writing both KV caches and computing no
/// logits. A calibration collector, when armed, sees every projection input.
pub fn forward_prefill_batch(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: usize,
) -> Result<(), String> {
    for (i, chunk) in tokens.chunks(GEMMA4_FORWARD_BATCH_MAX).enumerate() {
        let pos = start_pos + i * GEMMA4_FORWARD_BATCH_MAX;
        // One row runs the decode route, which reads the staged position.
        if let [token] = chunk {
            decode_step(cfg, weights, state, gpu, *token, pos as u32)?;
            continue;
        }
        forward_rows(cfg, weights, state, gpu, chunk, pos, None, None, false)?;
    }
    Ok(())
}

/// The batched program over `tokens` at `[start_pos, start_pos + B)`; with
/// `last_logits` the last row's logits are computed, downloaded and returned
/// (else an empty vector).
#[allow(clippy::too_many_arguments)]
fn forward_rows(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: usize,
    per_token_hidden_out: Option<&GpuTensor>,
    per_pos_argmax_out: Option<&mut Vec<u32>>,
    last_logits: bool,
) -> Result<Vec<f32>, String> {
    let b = tokens.len();
    if b == 0 {
        return Err("gemma4 forward_batch: empty token slice".to_string());
    }
    if b > GEMMA4_FORWARD_BATCH_MAX {
        return Err(format!(
            "gemma4 forward_batch: B={b} exceeds kernel cap {GEMMA4_FORWARD_BATCH_MAX}"
        ));
    }
    let dim = cfg.dim;
    let eps = cfg.norm_eps;
    let ffn_hd = cfg.max_ffn_hidden_dim();
    let max_q = cfg.max_q_dim();
    let max_kv = cfg.max_kv_dim();
    let ple_dim = cfg.hidden_size_per_layer_input;
    let ple_packed = cfg.n_layers * ple_dim;
    // seq_len after this batch = absolute positions [start_pos, start_pos+B).
    checked_batch_seq_len(start_pos, b, state.max_seq)?;
    if !(state.kv_sliding.quant_q8 && state.kv_full.quant_q8) {
        return Err("gemma4 forward_batch: batched attention needs Q8 caches".to_string());
    }

    let mut bufs = RowBuffers::new(gpu);
    let widths = RowWidths {
        dim,
        max_q,
        max_kv,
        ffn: ffn_hd,
    };
    let acts = RowActivations::new(&mut bufs, gpu, b, start_pos, widths)?;
    let (x, x_rot) = (&acts.x, &acts.attn_rot);
    let mut ple_buf =
        |n: usize, label: &str| (ple_dim != 0).then(|| bufs.alloc(n, label)).transpose();
    let ple_token_inputs = ple_buf(b * ple_packed, "ple_token_inputs")?;
    let ple_projection_all = ple_buf(b * ple_packed, "ple_projection_all")?;
    let ple_gate = ple_buf(b * ple_dim, "ple_gate")?;
    let ple_hidden = ple_buf(b * ple_dim, "ple_hidden")?;
    let ple_out = ple_buf(b * dim, "ple_out")?;

    let batched_embedding_requested = supports_gemma4_batched_prefill_arch(&gpu.arch)
        && gpu.flags.gemma4_batched_embedding_prefill
        && b > 1
        && (has_batched_embedding_lookup(weights.embd_format)
            || weights
                .per_layer_input
                .as_ref()
                .is_some_and(|ple| has_batched_embedding_lookup(ple.embd_format)));
    let token_ids = if batched_embedding_requested {
        let token_data: Vec<i32> = tokens.iter().map(|&token| token as i32).collect();
        let token_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(token_data.as_ptr() as *const u8, token_data.len() * 4)
        };
        let token_ids = bufs.alloc(b, "token_ids")?;
        gpu.hip
            .memcpy_htod(&token_ids.buf, token_bytes)
            .map_err(|e| format!("gemma4 forward_batch htod token ids: {e:?}"))?;
        Some(token_ids)
    } else {
        None
    };

    // ── Embedding: per-token lookup into x[B,dim], then ×√dim over all rows. ──
    let batched_embedding = if let Some(token_ids) = token_ids.as_ref() {
        embedding_lookup_batched_to(
            gpu,
            weights.embd_format,
            &weights.embed_tokens,
            x,
            token_ids,
            b,
            dim,
            "batch main embed",
        )?
    } else {
        false
    };
    if !batched_embedding {
        let x_single = bufs.alloc(dim, "x_single")?;
        for (i, &tok) in tokens.iter().enumerate() {
            embed_lookup_row(cfg, weights, gpu, &x_single, tok)?;
            gpu.hip
                .memcpy_dtod_at(&x.buf, i * dim * 4, &x_single.buf, 0, dim * 4)
                .map_err(|e| format!("gemma4 forward_batch embed copy: {e:?}"))?;
        }
    }
    // √dim scale on the whole [B*dim] buffer (uniform — matches eager scale_f32).
    gpu.scale_f32(x, cfg.embed_scale)
        .map_err(|e| format!("gemma4 forward_batch embed scale: {e:?}"))?;
    if ple_dim != 0 {
        prepare_per_layer_inputs_batched(
            cfg,
            weights,
            gpu,
            tokens,
            token_ids.as_ref(),
            x,
            x_rot,
            ple_token_inputs.as_ref().unwrap(),
            ple_projection_all.as_ref().unwrap(),
        )?;
    }

    let ctx = DispatchCtx::new(gpu);
    let ple = match (
        ple_projection_all.as_ref(),
        ple_gate.as_ref(),
        ple_hidden.as_ref(),
        ple_out.as_ref(),
    ) {
        (Some(inputs), Some(gate), Some(act), Some(out)) => Some(PleScratch {
            inputs,
            gate,
            act,
            out,
        }),
        _ => None,
    };
    // Grouped routed-expert scratch for the batched rows (index buffers are
    // i32 in F32-sized slots).
    let routed_batch = match state.moe.as_ref() {
        Some(_) if b > 1 => {
            let (k, ne, mi) = (
                cfg.top_k_experts,
                cfg.num_experts,
                cfg.moe_intermediate_size,
            );
            let m_total = hipfire_dispatch::pipeline::sandwich::RoutedBatch::m_total(b, k, ne);
            let block_m = hipfire_dispatch::families::moe::MOE_GROUPED_BLOCK_M;
            let mut a = |n: usize, label: &str| bufs.alloc(n, label);
            Some([
                a(b * dim, "moe input")?,
                a(b * dim, "moe input_rot")?,
                a(b * ne, "moe router_logits")?,
                a(b * k, "moe topk_indices")?,
                a(b * k, "moe topk_weights")?,
                a(b * k, "moe inverse_perm")?,
                a(ne, "moe counts")?,
                a(ne + 1, "moe offsets")?,
                a(m_total, "moe sorted_slots")?,
                a(m_total / block_m, "moe tile_ids")?,
                a(m_total * (2 * mi).max(dim), "moe grouped rows")?,
                a(b * k * mi, "moe gate")?,
                a(b * k * mi, "moe up")?,
                a(b * dim, "moe out")?,
            ])
        }
        _ => None,
    };
    let routed = state.moe.as_ref().map(|m| {
        let mut r = crate::program::routed_scratch(m);
        r.batch =
            routed_batch
                .as_ref()
                .map(|t| hipfire_dispatch::pipeline::sandwich::RoutedBatch {
                    input: &t[0],
                    input_rot: &t[1],
                    router_logits: &t[2],
                    topk_indices: &t[3],
                    topk_weights: &t[4],
                    inverse_perm: &t[5],
                    counts: &t[6],
                    offsets: &t[7],
                    sorted_slots: &t[8],
                    tile_ids: &t[9],
                    gate_up_grouped: &t[10],
                    gate: &t[11],
                    up: &t[12],
                    out: &t[13],
                });
        r
    });
    let binding = ProgramBinding {
        geo: Geometry::eager(cfg),
        resident: eager_resident(state),
        rows: b,
        position: start_pos,
        positions: Some(&acts.positions),
        scratch: acts.scratch(ple, routed),
    };
    let mut steps = Vec::with_capacity(4 * cfg.n_layers);
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        binding.layer(
            layer_idx,
            LayerRefs::eager(layer),
            eager_layer_kv(cfg, state, layer_idx)?,
            &mut steps,
        )?;
    }
    execute_steps(gpu, &ctx, &steps).map_err(|e| format!("gemma4 forward_batch: {e}"))?;
    drop(steps);
    state.n_tokens = start_pos + b;

    // ── Spec-decode out-params (ADDITIVE; both default-off). ──
    // `x` here holds the [B, dim] post-residual hidden for all B positions.
    //
    // (1) per_token_hidden_out: batched final-norm over all B rows → the
    //     caller's [B, dim] buffer. Row i is exactly what the eager
    //     decode_step leaves in state.tmp (the lm_head input) at that position.
    // (2) per_pos_argmax_out: batched lm_head over all B rows → [B, vocab],
    //     final logit softcap (elementwise over B*vocab), per-row argmax on host.
    //     Uses the SAME batched proj path (`proj_gemm_batched`) the verify
    //     forward uses for its other projections, so the MQ4 lm_head rotation is
    //     handled with explicit scratch (NOT weight_gemv's internal mq_x_rot,
    //     which the batched path never sizes → would fault on MQ4 lm_head).
    //
    // Both use a [B, dim] normed-hidden buffer. When (1) is requested we write
    // straight into it; otherwise we allocate a local one only if (2) needs it.
    let want_hidden = per_token_hidden_out.is_some();
    let want_argmax = per_pos_argmax_out.is_some();
    if want_hidden || want_argmax {
        // Normed hidden destination: the caller's buffer if provided, else a
        // local scratch sized [B, dim].
        let local_hidden = if want_hidden {
            None
        } else {
            Some(bufs.alloc(b * dim, "spec_normed")?)
        };
        let normed_hidden: &GpuTensor = match per_token_hidden_out {
            Some(h) => h,
            None => local_hidden.as_ref().unwrap(),
        };
        // Batched final RMSNorm over all B rows.
        gpu.rmsnorm_batched(x, &weights.final_norm, normed_hidden, b, dim, eps)
            .map_err(|e| format!("gemma4 forward_batch_spec final rmsnorm: {e:?}"))?;

        if let Some(out) = per_pos_argmax_out {
            out.clear();
            out.reserve(b);
            let vocab = cfg.vocab_size;
            if weights.lm_head.gpu_dtype == DType::Q8_0 {
                // FAST PATH (the spec-verify perf lever): a single batched Q8 WMMA
                // lm_head reads the ~1 GB Q8 weight ONCE for all B rows, instead of
                // the per-row weight_gemv that re-streamed the whole weight B times
                // (~90% of the verify, ~4× off roofline). On gfx12 (RDNA4)
                // `gemm_q8_0_batched_chunked` auto-routes to the WMMA Q8 GEMM (now
                // correct after the stale-fp16-cache fix); elsewhere it sub-batches
                // the scalar Q8 GEMM — either way Y[B, vocab] row-major.
                //
                // Softcap is SKIPPED: it's a strictly-monotonic per-element map
                // (tanh-scaled), so argmax(softcap(z)) == argmax(z). The accept
                // decision is an argmax, so this is bit-exact in the decision.
                let logits_b = bufs.alloc(b * vocab, "spec_logits_b")?;
                // Not `_chunked`: verify stays on the scalar kernel rather
                // than the gfx12 WMMA variant, whose drift accumulates in KV.
                let lm_result = gpu.gemm_q8_0_batched(
                    &weights.lm_head.buf,
                    normed_hidden,
                    &logits_b,
                    weights.lm_head.m,
                    weights.lm_head.k,
                    b,
                );
                lm_result
                    .map_err(|e| format!("gemma4 forward_batch_spec batched lm_head: {e:?}"))?;
                // GPU per-row argmax over [B, vocab]; only B indices land on PCIe.
                let idx_buf = bufs.alloc(b, "spec_argmax_idx")?;
                gpu.argmax_f32_batched(&logits_b, &idx_buf, vocab, b)
                    .map_err(|e| format!("gemma4 forward_batch_spec batched argmax: {e:?}"))?;
                let mut idx_i32 = vec![0i32; b];
                let idx_bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(idx_i32.as_mut_ptr() as *mut u8, b * 4)
                };
                gpu.hip
                    .memcpy_dtoh(idx_bytes, &idx_buf.buf)
                    .map_err(|e| format!("gemma4 forward_batch_spec argmax dtoh: {e:?}"))?;
                for v in idx_i32 {
                    out.push(v as u32);
                }
            } else {
                // FALLBACK (non-Q8 lm_head, e.g. MQ4): per-row SCALAR `weight_gemv`
                // (exactly the eager decode_step / the last-row path below). We
                // deliberately do NOT use the batched MQ4 path here: for an MQ4
                // lm_head (m=vocab≈262144) it routes through
                // `gemm_hfq4g256_residual_wmma[_gfx12]`, whose b>1 path faults
                // (illegal access) at this output width on gfx12. `weight_gemv`
                // (per-row, b=1) sidesteps it entirely.
                for i in 0..b {
                    let hidden_row = normed_hidden.sub_offset(i * dim, dim);
                    weight_gemv(gpu, &weights.lm_head, &hidden_row, &state.logits)
                        .map_err(|e| format!("gemma4 forward_batch_spec lm_head row {i}: {e}"))?;
                    if cfg.final_logit_softcapping > 0.0 {
                        gpu.logit_softcap_f32(&state.logits, vocab, cfg.final_logit_softcapping)
                            .map_err(|e| {
                                format!("gemma4 forward_batch_spec softcap row {i}: {e:?}")
                            })?;
                    }
                    let row = gpu.download_f32(&state.logits).map_err(|e| {
                        format!("gemma4 forward_batch_spec download row {i}: {e:?}")
                    })?;
                    out.push(argmax_f32_row(&row));
                }
            }
        }
    }

    if !last_logits {
        return Ok(Vec::new());
    }
    // ── Final RMSNorm + tied lm_head on the LAST row only (verify needs the
    //    last position's logits). ──
    let x_last = bufs.alloc(dim, "x_last")?;
    gpu.hip
        .memcpy_dtod_at(&x_last.buf, 0, &x.buf, (b - 1) * dim * 4, dim * 4)
        .map_err(|e| format!("gemma4 forward_batch last copy: {e:?}"))?;
    gpu.rmsnorm_f32(&x_last, &weights.final_norm, &state.tmp, eps)
        .map_err(|e| format!("gemma4 forward_batch final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.tmp, &state.logits)
        .map_err(|e| format!("gemma4 forward_batch lm_head: {e}"))?;
    if cfg.final_logit_softcapping > 0.0 {
        gpu.logit_softcap_f32(&state.logits, cfg.vocab_size, cfg.final_logit_softcapping)
            .map_err(|e| format!("gemma4 forward_batch logit softcap: {e:?}"))?;
    }
    let logits = gpu
        .download_f32(&state.logits)
        .map_err(|e| format!("gemma4 forward_batch download logits: {e:?}"))?;

    Ok(logits)
}

/// Embedding lookup for one token into a [dim] buffer (no √dim scale — the
/// caller applies it once over the whole batch). Mirrors `embed_lookup`'s
/// format dispatch.
fn embed_lookup_row(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    gpu: &mut Gpu,
    dst: &GpuTensor,
    token_id: u32,
) -> Result<(), String> {
    use hipfire_runtime::llama::EmbeddingFormat;
    let dim = cfg.dim;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu
            .embedding_lookup_hfq4g256(&weights.embed_tokens, dst, token_id, dim)
            .map_err(|e| format!("gemma4 forward_batch embed hfq4g256: {e:?}")),
        EmbeddingFormat::HFQ4G128 => gpu
            .embedding_lookup_hfq4g128(&weights.embed_tokens, dst, token_id, dim)
            .map_err(|e| format!("gemma4 forward_batch embed hfq4g128: {e:?}")),
        EmbeddingFormat::Q8_0 => gpu
            .embedding_lookup_q8(&weights.embed_tokens, dst, token_id, dim)
            .map_err(|e| format!("gemma4 forward_batch embed q8: {e:?}")),
        EmbeddingFormat::F32 => gpu
            .embedding_lookup(&weights.embed_tokens, dst, token_id, dim)
            .map_err(|e| format!("gemma4 forward_batch embed f32: {e:?}")),
        EmbeddingFormat::Q4K => Err("gemma4 forward_batch: Q4K embedding unsupported".to_string()),
    }
}

#[inline]
fn supports_gemma4_batched_prefill_arch(arch: &str) -> bool {
    matches!(arch, "gfx1100" | "gfx1201")
}

#[cfg(test)]
mod tests {
    use super::{checked_batch_seq_len, supports_gemma4_batched_prefill_arch};

    #[test]
    fn batch_window_must_fit_allocated_kv_capacity() {
        assert_eq!(checked_batch_seq_len(60, 4, 64).unwrap(), 64);
        assert!(checked_batch_seq_len(61, 4, 64).is_err());
        assert!(checked_batch_seq_len(usize::MAX, 1, usize::MAX).is_err());
    }

    #[test]
    fn batched_prefill_arch_scope_is_explicit() {
        assert!(supports_gemma4_batched_prefill_arch("gfx1100"));
        assert!(supports_gemma4_batched_prefill_arch("gfx1201"));
        assert!(!supports_gemma4_batched_prefill_arch("gfx1151"));
        assert!(!supports_gemma4_batched_prefill_arch("gfx1200"));
    }
}
