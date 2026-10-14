// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! LFM2.5-MoE forward pass (free functions — hot-path static dispatch).
//!
//! Per-layer pipeline (pre-norm; mixer = conv OR attention, FFN = dense OR MoE):
//!   tmp = operator_norm(h)
//!   if conv:   h += out_proj( C_gate ⊙ depthwise_causal_conv( B_gate ⊙ x ) )   [in_proj→conv→out_proj]
//!   if attn:   h += out_proj( attn( qk_norm(q/k) + full-RoPE, v ) )             [GQA, Q8 KV]
//!   ffn_tmp = ffn_norm(h)
//!   if dense:  h += w2( silu(w1·ffn_tmp) ⊙ (w3·ffn_tmp) )                        [SwiGLU, Q8]
//!   if moe:    h += combine( experts( sigmoid+bias top-4 route(ffn_tmp) ) )      [FWHT MQ4 experts]
//! then logits = lm_head( embedding_norm(h) )   (lm_head tied to embed_tokens).
//!
//! Non-expert linears (attention q/k/v/out, conv in/out, dense w1/w2/w3, router)
//! are Q8 (plain input). Routed experts are FWHT-pre-rotated MQ4G256: the input
//! is rotated (`rotate_x_mq_for`) and the silu output rotated
//! (`fused_silu_mul_rotate_mq_batched_for`) before the indexed-MoE GEMVs —
//! exactly qwen35's / minimax's MoE path, but with k_top = num_experts_per_tok
//! = 4 (the batched GEMV variants take k_top as a runtime arg).

use crate::config::Lfm2MoeConfig;
use crate::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
use crate::program::{head_steps, LayerRefs};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::execute_steps;
use rdna_compute::Gpu;

/// Decode one token; returns the full logits vector.
///
/// Routes to the retained replay path when the Redline controller is enabled
/// (dense-only) — otherwise falls back to the hipGraph path when
/// `HIPFIRE_LFM2_GRAPH=1` (default OFF → exact prior behavior). Retained takes
/// precedence over HipGraph. Dense-only: non-dense models poison/disable the
/// controller and continue via the existing HIP path without falsely claiming
/// replay. Any decode while `retained_state_poisoned` is true rejects until
/// `reset()` clears it.
pub fn decode_step(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    if state.retained_state_poisoned {
        return Err("lfm2moe: retained state poisoned until reset".to_string());
    }
    if gpu.replay.is_enabled() {
        if !cfg.is_dense() {
            gpu.replay.set_forward_eligible(false);
            gpu.replay
                .poison("LFM retained replay requires dense model");
        } else {
            gpu.replay.set_forward_eligible(true);
            return decode_step_with_retained_replay(cfg, weights, state, gpu, token_id, position);
        }
    }
    if graph_enabled() {
        return decode_step_with_graph(cfg, weights, state, gpu, token_id, position);
    }
    decode_step_inner(cfg, weights, state, gpu, token_id, position, None)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("lfm2moe: download logits: {e:?}"))
}

fn decode_step_with_retained_replay(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    if state.retained_state_poisoned {
        return Err("lfm2moe: retained state poisoned until reset".to_string());
    }
    if !cfg.is_dense() {
        gpu.replay.set_forward_eligible(false);
        gpu.replay
            .poison("LFM retained replay requires dense model");
        decode_step_inner(cfg, weights, state, gpu, token_id, position, None)?;
        return gpu
            .download_f32(&state.logits)
            .map_err(|e| format!("lfm2moe: download logits (non-dense fallback): {e:?}"));
    }
    gpu.replay.set_forward_eligible(true);
    if !state.retained_warmed_up {
        decode_step_inner(cfg, weights, state, gpu, token_id, position, None)?;
        state.retained_warmed_up = true;
        return gpu
            .download_f32(&state.logits)
            .map_err(|e| format!("lfm2moe: download logits (retained warmup): {e:?}"));
    }
    prepare_retained_decode_inputs(cfg, weights, state, gpu, token_id, position)?;
    if !gpu.replay.should_route_aql() && !gpu.replay.should_route_pm4() {
        let _ = gpu.replay.begin_auto_capture_if_armed();
        if gpu.replay.state() == rdna_compute::replay::ReplayState::Armed {
            let _ = gpu.replay.begin_capture();
        }
    }
    if gpu.replay.should_route_aql() || gpu.replay.should_route_pm4() {
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("lfm2moe: retained adapter sync: {e:?}"))?;
        let replay_result = if gpu.replay.should_route_aql() {
            unsafe { gpu.replay.replay_linear_aql(position as usize) }.map(|_| ())
        } else {
            unsafe { gpu.replay.replay_pm4(position as usize) }.map(|_| ())
        };
        match replay_result {
            Ok(_) => {
                state.n_tokens = position as usize + 1;
                return gpu
                    .download_f32(&state.logits)
                    .map_err(|e| format!("lfm2moe: download logits (retained replay): {e:?}"));
            }
            Err(reason) => {
                let msg = format!("LFM retained replay failed: {reason}");
                gpu.replay.poison(msg);
                state.retained_state_poisoned = true;
                return Err(reason);
            }
        }
    }
    run_retained_decode_body(cfg, weights, state, gpu, position)?;
    if gpu.replay.should_auto_finalize_capture() {
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("lfm2moe: retained capture sync: {e:?}"))?;
        let capture = gpu
            .replay
            .finish_capture()
            .map_err(|e| format!("lfm2moe: finish capture: {e}"))?;
        let launches = gpu.replay.recorded_launches().len();
        let prepare = if gpu.replay.uses_pm4_transport() {
            gpu.replay
                .prepare_pm4_prefix(gpu.device_id as usize, launches)
                .map(|_| ())
        } else {
            gpu.replay
                .prepare_linear_aql_prefix(gpu.device_id as usize, launches)
                .map(|_| ())
        };
        match prepare {
            Ok(()) => {
                eprintln!(
                    "[LFM2.5-MoE redline] retained route ready: capture={capture:?} identity={:?}",
                    gpu.replay.prepared_route_identity()
                );
            }
            Err(reason) => {
                gpu.replay
                    .poison(format!("LFM Redline prepare after warmup failed: {reason}"));
                eprintln!("[LFM2.5-MoE redline] falling back to HIP: {reason}");
            }
        }
    } else if gpu.replay.state() == rdna_compute::replay::ReplayState::RecordingWarmup {
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("lfm2moe: retained capture sync: {e:?}"))?;
        if let Ok(capture) = gpu.replay.finish_capture() {
            let launches = gpu.replay.recorded_launches().len();
            let prepare = if gpu.replay.uses_pm4_transport() {
                gpu.replay
                    .prepare_pm4_prefix(gpu.device_id as usize, launches)
                    .map(|_| ())
            } else {
                // Manual capture needs at least 2 dispatches for linear AQL.
                if launches >= 2 {
                    gpu.replay
                        .prepare_linear_aql_prefix(gpu.device_id as usize, launches)
                        .map(|_| ())
                } else {
                    Err("no captured launch sequence".to_owned())
                }
            };
            match prepare {
                Ok(()) => eprintln!(
                    "[LFM2.5-MoE redline] retained route ready (manual): capture={capture:?} identity={:?}",
                    gpu.replay.prepared_route_identity()
                ),
                Err(reason) => {
                    gpu.replay
                        .poison(format!("LFM Redline manual prepare failed: {reason}"));
                    eprintln!("[LFM2.5-MoE redline] falling back to HIP: {reason}");
                }
            }
        }
    }
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("lfm2moe: download logits (retained capture): {e:?}"))
}

/// `HIPFIRE_LFM2_GRAPH=1` opt-in switch. Default OFF (unset / "0") →
/// byte-identical to the legacy per-launch decode path. Parsed once.
fn graph_enabled() -> bool {
    use std::sync::OnceLock;
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| {
        matches!(
            hipfire_config::developer_var("HIPFIRE_LFM2_GRAPH")
                .ok()
                .as_deref(),
            Some("1")
        )
    })
}

/// Decode one token, appending each layer's post-residual hidden state
/// (after the full layer, before the final norm) to `capture[layer]` — used by
/// the oracle dumper. Set `HIPFIRE_LFM2_CAPTURE_POSTMIXER` to capture the
/// post-mixer residual (pre-FFN) instead, for conv/attn-vs-FFN localization.
pub fn decode_step_capture(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture: &mut [Vec<f32>],
) -> Result<(), String> {
    decode_step_inner(cfg, weights, state, gpu, token_id, position, Some(capture))
}

#[doc(hidden)]
pub fn prepare_retained_decode_inputs(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<(), String> {
    if (position as usize) >= state.max_seq {
        return Err(format!(
            "lfm2moe: position {} >= max_seq {}",
            position, state.max_seq
        ));
    }
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("lfm2moe: htod pos: {e:?}"))?;
    hipfire_runtime::llama::embedding_lookup_dispatch(
        gpu,
        weights.embd_format,
        &weights.embed,
        &state.h,
        token_id,
        cfg.hidden_size,
    )
    .map_err(|e| format!("lfm2moe: embed lookup: {e:?}"))?;
    Ok(())
}

/// Prefill one position from a raw embedding row instead of a token id —
/// the VL seam: projected visual features take the `<image>` placeholder
/// positions (see `hipfire-generate::vision::generate_lfm2_vl`). Same
/// staging contract as `prepare_retained_decode_inputs` + the ordinary
/// decode body; returns logits like `decode_step`.
///
/// Note the streaming invariant: unlike retained-decode staging, this runs
/// OUTSIDE any captured graph region (a fresh h-memcpy per call), which is
/// fine — VL prefill is eager by construction.
pub fn prefill_embed_step(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    embedding: &[f32],
    position: u32,
) -> Result<Vec<f32>, String> {
    if state.retained_state_poisoned {
        return Err("lfm2moe: retained state poisoned until reset".to_string());
    }
    if (position as usize) >= state.max_seq {
        return Err(format!(
            "lfm2moe: position {} >= max_seq {}",
            position, state.max_seq
        ));
    }
    if embedding.len() != cfg.hidden_size {
        return Err(format!(
            "lfm2moe: prefill_embed_step expects {} floats, got {}",
            cfg.hidden_size,
            embedding.len()
        ));
    }
    gpu.hip
        .memcpy_htod(&state.pos_buf, &(position as i32).to_ne_bytes())
        .map_err(|e| format!("lfm2moe: htod pos: {e:?}"))?;
    let bytes: Vec<u8> = embedding.iter().flat_map(|v| v.to_le_bytes()).collect();
    gpu.hip
        .memcpy_htod(&state.h.buf, &bytes)
        .map_err(|e| format!("lfm2moe: htod visual embedding: {e:?}"))?;
    decode_step_layers_and_head(cfg, weights, state, gpu, position, None)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("lfm2moe: download logits (embed prefill): {e:?}"))
}

#[doc(hidden)]
pub fn run_retained_decode_body(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    decode_step_layers_and_head(cfg, weights, state, gpu, position, None)
}

fn decode_step_inner(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    if capture.is_none() {
        prepare_retained_decode_inputs(cfg, weights, state, gpu, token_id, position)?;
        return run_retained_decode_body(cfg, weights, state, gpu, position);
    }
    // Oracle-capture path: still staged through the same prepare helper for
    // bounds-check / single staging convention, but body must carry capture.
    prepare_retained_decode_inputs(cfg, weights, state, gpu, token_id, position)?;
    decode_step_layers_and_head(cfg, weights, state, gpu, position, capture)
}

/// Per-layer mixer/FFN stack + final norm + lm_head. Reads the residual
/// stream `state.h` (already seeded by the embedding lookup) and the device
/// position scalar `state.pos_buf` (already staged); writes `state.logits`.
///
/// This is the hipGraph-captureable region: it issues only kernel launches
/// that read STABLE device buffers and (on the MoE path) compute their
/// topk/positions on-device, so a single capture replays correctly at every
/// later position once `state.pos_buf` is refreshed. The per-token-varying
/// embedding lookup (token_id is a kernarg) and the `pos_buf` htod are the
/// caller's responsibility OUTSIDE the captured region.
///
/// `capture` (oracle dumper) is incompatible with hipGraph capture — it issues
/// a sync `download_f32` per layer. The graph path always passes `None`.
fn decode_step_layers_and_head(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    position: u32,
    mut capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    let capture_postmixer =
        hipfire_config::developer_var_os("HIPFIRE_LFM2_CAPTURE_POSTMIXER").is_some();
    let ctx = DispatchCtx::new(gpu);
    let pos = position as usize;
    for (l, layer) in weights.layers.iter().enumerate() {
        let refs = LayerRefs::new(layer);
        let mixer = refs
            .mixer_steps(cfg, state, pos)
            .map_err(|e| format!("lfm2moe L{l}: mixer program: {e}"))?;
        execute_steps(gpu, &ctx, &mixer).map_err(|e| format!("lfm2moe L{l}: mixer: {e}"))?;
        if capture_postmixer {
            capture_h(gpu, state, l, capture.as_deref_mut())?;
        }
        let ffn = refs
            .ffn_steps(cfg, &ctx, state, l)
            .map_err(|e| format!("lfm2moe L{l}: ffn program: {e}"))?;
        execute_steps(gpu, &ctx, &ffn).map_err(|e| format!("lfm2moe L{l}: ffn: {e}"))?;
        if !capture_postmixer {
            capture_h(gpu, state, l, capture.as_deref_mut())?;
        }
    }
    let lm_head = weights.lm_head.dispatch_ref();
    execute_steps(gpu, &ctx, &head_steps(cfg, weights, &lm_head, state))
        .map_err(|e| format!("lfm2moe: head: {e}"))?;
    state.n_tokens = pos + 1;
    Ok(())
}

/// Oracle capture: append `state.h` to `capture[l]` (sync download).
fn capture_h(
    gpu: &mut Gpu,
    state: &Lfm2MoeState,
    l: usize,
    capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    if let Some(cap) = capture {
        let h = gpu
            .download_f32(&state.h)
            .map_err(|e| format!("lfm2moe L{l}: capture download: {e:?}"))?;
        cap[l].extend_from_slice(&h);
    }
    Ok(())
}

/// hipGraph-amortized decode_step. Opt-in via `HIPFIRE_LFM2_GRAPH=1`
/// (default OFF → exact `decode_step_inner` behavior). Mirrors the working
/// DeepSeek-V4 integration (`decode_step_with_graph`).
///
/// Three-state machine driven by `state.graph_warmed_up` and `gpu.graph_exec`:
///   1. !warmed_up                 → direct dispatch once (so kernel JIT and
///                                    any lazy hipMalloc happen OUTSIDE the
///                                    captured region), set the flag.
///   2. warmed_up && no graph      → embedding+pos direct, then capture the
///                                    layer loop + head, instantiate, launch
///                                    once for this position's output.
///   3. graph instantiated         → embedding+pos direct, then `graph_launch`
///                                    re-runs the captured ops which re-read
///                                    `state.pos_buf` (refreshed below) and the
///                                    KV / conv-state / topk device buffers.
///
/// Per-token-varying values handled OUTSIDE the captured region:
///   * `token_id` — baked into the embedding kernarg, so the
///     embedding lookup runs DIRECT each token (writes `state.h`); the
///     captured region begins at layer 0's rmsnorm reading `state.h`.
///   * `position` — staged into the STABLE device buffer `state.pos_buf` via a
///     direct `memcpy_htod` before each `graph_launch`; every captured kernel
///     (rope/kv-write/attention) reads `pos_buf` from the device, so replay at
///     a new position is correct without re-capture. The attention kernel uses
///     a fixed tile/block/shared-mem (ATT_Q8_KV_TILE=2048, block 256) independent
///     of `max_seq`/`seq_len` (tiled online-softmax), so one capture replays
///     correctly at every later position.
///
/// `state.n_tokens` is advanced here to match `decode_step_inner` semantics.
pub fn decode_step_with_graph(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    state: &mut Lfm2MoeState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    // ── Warmup phase: direct dispatch, no capture ──────────────────────────
    // Run the legacy path once so inline JIT / lazy scratch alloc happen
    // before any stream capture (capturing a hipMalloc errors).
    if !state.graph_warmed_up {
        state.graph_warmed_up = true;
        decode_step_inner(cfg, weights, state, gpu, token_id, position, None)?;
        return gpu
            .download_f32(&state.logits)
            .map_err(|e| format!("lfm2moe: download logits (graph warmup): {e:?}"));
    }

    // Capture/replay needs an explicit (non-null) stream.
    if gpu.active_stream.is_none() {
        let s = gpu
            .hip
            .stream_create()
            .map_err(|e| format!("lfm2moe: stream_create: {e:?}"))?;
        gpu.active_stream = Some(s);
    }

    // Per-token-varying ops, DIRECT (outside the captured region).
    // Single staging convention: shared helper handles bounds check + pos H2D + embedding.
    prepare_retained_decode_inputs(cfg, weights, state, gpu, token_id, position)?;
    if gpu.graphs.graph_exec.is_none() {
        // ── Capture phase ──────────────────────────────────────────────────
        gpu.graphs
            .begin_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("lfm2moe: begin_graph_capture: {e:?}"))?;
        run_retained_decode_body(cfg, weights, state, gpu, position)?;
        gpu.graphs
            .end_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("lfm2moe: end_graph_capture: {e:?}"))?;
        // Recorded, not executed — launch once so this position's logits are real.
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("lfm2moe: graph_launch (capture-end): {e:?}"))?;
        eprintln!(
            "[LFM2.5-MoE hipGraph] captured forward — {} kernarg blobs retained",
            gpu.graphs.capture_blobs.len()
        );
        // run_retained_decode_body set n_tokens; capture-end launch ran it.
    } else {
        // ── Replay phase ────────────────────────────────────────────────────
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("lfm2moe: graph_launch (replay): {e:?}"))?;
        // Mirror run_retained_decode_body's `state.n_tokens = position + 1`,
        // which the replayed graph does NOT execute (it is host-side state).
        state.n_tokens = position as usize + 1;
    }

    // Logits download outside the captured region (sync D2H on the null stream;
    // completes after the captured kernels finish on the captured stream).
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("lfm2moe: download logits (graph): {e:?}"))
}
