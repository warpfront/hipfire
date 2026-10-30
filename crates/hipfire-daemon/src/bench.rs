// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The daemon's synthetic `bench_prefill` / `bench_decode` probes.
//!
//! Both run the loaded model's own production forward on deterministic token
//! ids: the architecture carrier's hooks on a single card (and Qwen4 expert
//! parallelism), and `serve`'s own entrypoints on every pipeline-, tensor- and
//! expert-parallel load (`hipfire_generate::bench_topology`). Every result
//! carries the loaded `topology` `{pp, tp, ep, cards}`.

use hipfire_engine::emit::emit_uncorrelated_error;
use hipfire_engine::redline::redline_capture_json;
use hipfire_generate::redline::{redline_bench_decode_deepseek4, redline_bench_decode_lfm2moe};
use hipfire_loader::LoadedModel;
use std::io::Write;
use std::time::Instant;

/// `{"type":"bench_prefill","tokens":N}` -> `prefill_result` (or one error).
pub(crate) fn handle_bench_prefill(
    model: &mut Option<LoadedModel>,
    gpu: &mut rdna_compute::Gpu,
    msg: &serde_json::Value,
    stdout: &mut std::io::Stdout,
) {
    // Synthetic prefill benchmark — measures the architecture's
    // production prefill entry on N deterministic tokens from a
    // zeroed state. Used by `hipfire bench` to produce canonical
    // pp128/pp512/pp1024 numbers that don't depend on a prompt
    // tokenizing to a round number. This stays a synthetic workload;
    // only the forward path must match production.
    let m = match model.as_mut() {
        Some(m) => m,
        None => {
            let _ = emit_uncorrelated_error(
                stdout,
                None,
                "no model loaded",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    };
    // Whole-request PLE serving guard: the synthetic prefill pauses the PLE
    // prewarm readers for its whole run (every return path drops it).
    let _ple_serving = m.ple_serving_guard();
    // The loaded model is the authority on its topology: single card,
    // pipeline-parallel, dense tensor-parallel or expert-parallel, with
    // the degrees and device ordinals read from what is resident. Each
    // routes through the entrypoints `serve` runs on it; only serve's own
    // refusals (e.g. tp and pp together) survive.
    let (bench_route, bench_topology) = match m.bench_topology(gpu.device_id) {
        Ok(resolved) => resolved,
        Err(message) => {
            emit_uncorrelated_error(stdout, None, &message, "unsupported", false, false);
            let _ = stdout.flush();
            return;
        }
    };
    // Diffusion trunks have no token prefill; the carrier-level
    // dispatch below would panic on the unimplemented trait
    // default, so refuse cleanly here.
    if hipfire_loader::img_route(m.arch_id).is_diffusion() {
        emit_uncorrelated_error(
            stdout,
            None,
            "bench_prefill unsupported for diffusion checkpoints (arch 40/45)",
            "unsupported",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    let n = msg.get("tokens").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
    let capture = msg
        .get("redline_capture")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let capture_detail = msg
        .get("redline_detail")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    // Redline capture is bound to the daemon's single `Gpu`; on a
    // pipeline/tensor/expert mesh it would observe one rank only.
    if capture && !bench_route.uses_carrier_hooks() {
        emit_uncorrelated_error(
                stdout,
                None,
                "redline prefill capture refused: capture observes the daemon's single device and cannot cover a multi-device (pp/tp/ep) load",
                "unsupported",
                false,
                false,
            );
        let _ = stdout.flush();
        return;
    }
    // Guard physical_cap — reserve 32 slots of headroom so a subsequent
    // generate request against the loaded model still has room. We guard
    // on the *physical* buffer (not the advertised max_seq) because this
    // bench intentionally bypasses eviction to measure raw prefill.
    if n.saturating_add(32) > m.physical_cap {
        emit_uncorrelated_error(
            stdout,
            None,
            &format!(
                "bench_prefill tokens={} exceeds loaded physical_cap={}",
                n, m.physical_cap
            ),
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    // Deterministic synthetic token IDs. Skip 0 (often <pad>) and the
    // low specials by offsetting, and wrap in a 1000-wide window so the
    // embedding lookup cost stays realistic rather than hitting one
    // cache-hot row repeatedly.
    let synthetic: Vec<u32> = (0..n as u32).map(|i| 10 + (i % 1000)).collect();

    // Reset state BEFORE timing so we're measuring cold prefill, not
    // prefill-on-top-of-prior-state. qwen35 recurrent state lives in
    // the bundle (qwen35 arch bundle, accessed via ArchModel);
    // `LoadedModel.dn_state` was removed (was always None) — the live
    // DeltaNet state is in the bundle.
    if bench_route.uses_carrier_hooks() {
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        let _ = hipfire_generate::common::reset_qwen35_recurrent(m, gpu);
        if let Some(st) = m.state.as_mut() {
            let _ = st.as_mut().reset_session_state(gpu);
        }
    } else if let Err(reason) = hipfire_generate::bench_topology::reset_mesh_for_bench(m, gpu) {
        emit_uncorrelated_error(
            stdout,
            None,
            &format!("bench_prefill state reset failed: {reason}"),
            "internal",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    // Flush any residual GPU work so it doesn't bleed into the
    // measured interval, then time forward_prefill_batch + a
    // trailing device_synchronize so we capture actual GPU
    // completion (kernel launches are async by default).
    if capture {
        if let Err(reason) = gpu.replay.begin_capture() {
            emit_uncorrelated_error(
                stdout,
                None,
                &format!("redline prefill capture refused: {reason}"),
                "unsupported",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    }
    if !bench_route.uses_carrier_hooks() {
        if let Err(reason) = hipfire_generate::bench_topology::quiesce_mesh(bench_route, m) {
            emit_uncorrelated_error(
                stdout,
                None,
                &format!("bench_prefill could not quiesce the loaded devices: {reason}"),
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    }
    let _ = gpu.hip.device_synchronize();
    let t0 = Instant::now();
    let mut prefill_err: Option<String> = None;
    let run_ok = if bench_route.uses_carrier_hooks() {
        // Arch-erased bench-prefill dispatch via Carrier (wave2 GenDispatch).
        // Each carrier implements its architecture's synthetic prefill body
        // verbatim; the daemon no longer matches on arch_id. See
        // `hipfire_loader::Carrier::bench_prefill`. A single card and Qwen4
        // expert parallelism (the bundle fans the call out over its ranks)
        // take this path.
        let carrier =
            hipfire_loader::carrier_for(m.arch_id).expect("bench_prefill: unknown arch_id");
        carrier
            .bench_prefill(m, gpu, &synthetic, n, &mut prefill_err)
            .expect("bench_prefill: carrier does not implement bench_prefill for this arch")
    } else {
        // Pipeline-parallel, dense tensor-parallel and expert-parallel loads
        // run the same prefill entrypoints `serve` runs on that topology; the
        // model devices are drained inside, so `t0..elapsed` spans every rank.
        match hipfire_generate::bench_topology::bench_prefill_mesh(bench_route, m, gpu, &synthetic)
        {
            Ok(()) => true,
            Err(reason) => {
                prefill_err = Some(reason);
                false
            }
        }
    };
    let _ = gpu.hip.device_synchronize();
    let elapsed = t0.elapsed().as_secs_f64();
    let capture_summary = if capture {
        gpu.replay.finish_capture().map(Some)
    } else {
        Ok(None)
    };

    // Reset state AFTER measurement — we've written N KV slots and a
    // DeltaNet state that the next real request must not inherit.
    // qwen35 recurrent state lives in the bundle (qwen35 arch bundle,
    // accessed via ArchModel); `LoadedModel.dn_state` was removed
    // (was always None).
    if bench_route.uses_carrier_hooks() {
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        let _ = hipfire_generate::common::reset_qwen35_recurrent(m, gpu);
        if let Some(st) = m.state.as_mut() {
            let _ = st.as_mut().reset_session_state(gpu);
        }
    } else if let Err(reason) = hipfire_generate::bench_topology::reset_mesh_for_bench(m, gpu) {
        emit_uncorrelated_error(
            stdout,
            None,
            &format!("bench_prefill state reset failed: {reason}"),
            "internal",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    let capture_summary = match capture_summary {
        Ok(summary) => summary,
        Err(reason) => {
            emit_uncorrelated_error(
                stdout,
                None,
                &format!("redline prefill capture failed: {reason}"),
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    };

    if run_ok {
        let tok_s = if elapsed > 0.0 {
            n as f64 / elapsed
        } else {
            0.0
        };
        if let Some(summary) = capture_summary {
            let response = serde_json::json!({
                "type": "prefill_result",
                "tokens": n,
                "ms": elapsed * 1000.0,
                "tok_s": tok_s,
                "topology": bench_topology.to_json(),
                "redline_capture": redline_capture_json(
                    gpu,
                    summary,
                    capture_detail,
                ),
            });
            let _ = writeln!(stdout, "{response}");
        } else {
            let _ = writeln!(
                stdout,
                r#"{{"type":"prefill_result","tokens":{},"ms":{:.2},"tok_s":{:.1},"topology":{}}}"#,
                n,
                elapsed * 1000.0,
                tok_s,
                bench_topology.to_json()
            );
        }
    } else {
        emit_uncorrelated_error(
            stdout,
            None,
            &match &prefill_err {
                Some(e) => format!("bench_prefill forward failed: {e}"),
                None => "bench_prefill forward failed".to_string(),
            },
            "validation",
            false,
            false,
        );
    }
    let _ = stdout.flush();
}

/// `{"type":"bench_decode","context_tokens":N,"iterations":N}` ->
/// `decode_result` (or one error).
pub(crate) fn handle_bench_decode(
    model: &mut Option<LoadedModel>,
    gpu: &mut rdna_compute::Gpu,
    msg: &serde_json::Value,
    stdout: &mut std::io::Stdout,
) {
    // Resident single-token decode probe for Redline and regular
    // daemon benchmarking. Prime deterministic Qwen3.5 state
    // outside the measured/captured interval, then time only the
    // requested number of forward_scratch calls.
    let m = match model.as_mut() {
        Some(m) => m,
        None => {
            emit_uncorrelated_error(stdout, None, "no model loaded", "validation", false, false);
            let _ = stdout.flush();
            return;
        }
    };
    // Whole-request PLE serving guard: priming and the timed decode pause
    // the PLE prewarm readers for the whole probe (every return path drops it).
    let _ple_serving = m.ple_serving_guard();
    // The loaded model is the authority on its topology (degrees and card
    // ordinals read from what is resident); `serve`'s own refusals are the
    // only ones that survive.
    let (bench_route, bench_topology) = match m.bench_topology(gpu.device_id) {
        Ok(resolved) => resolved,
        Err(message) => {
            emit_uncorrelated_error(stdout, None, &message, "unsupported", false, false);
            let _ = stdout.flush();
            return;
        }
    };
    // DeepSeek4 and dense LFM2 keep their own single-card Redline probes;
    // an EP DeepSeek4 load is a mesh route and never reaches them.
    let early_route = if bench_route == hipfire_loader::bench_topology::BenchRoute::Single {
        hipfire_loader::bench_decode_route(m.arch_id)
    } else {
        hipfire_loader::BenchDecodeRoute::Unsupported
    };
    match early_route {
        hipfire_loader::BenchDecodeRoute::Deepseek4 => {
            match redline_bench_decode_deepseek4(gpu, m, msg) {
                Ok(mut response) => {
                    response["topology"] = bench_topology.to_json();
                    let _ = writeln!(stdout, "{response}");
                }
                Err(reason) => {
                    let _ = writeln!(
                        stdout,
                        "{}",
                        serde_json::json!({"type": "error", "message": reason})
                    );
                }
            }
            let _ = stdout.flush();
            return;
        }
        hipfire_loader::BenchDecodeRoute::Lfm2Moe => {
            match redline_bench_decode_lfm2moe(gpu, m, msg) {
                Ok(mut response) => {
                    response["topology"] = bench_topology.to_json();
                    let _ = writeln!(stdout, "{response}");
                }
                Err(reason) => {
                    let _ = writeln!(
                        stdout,
                        "{}",
                        serde_json::json!({"type": "error", "message": reason})
                    );
                }
            }
            let _ = stdout.flush();
            return;
        }
        _ => {}
    }
    // arch 5/6 = Qwen3.5, arch 13 = Gemma4, arch 14 = Muse Glimmer, arch 16
    // = Qwen4. All prime with a batched prefill and then step tokens one
    // at a time, so the same bench shape applies; the two branches below
    // differ only in which forward they call. Pipeline, tensor and expert
    // loads route by topology (above), so only the carrier routes (one
    // card, Qwen4 expert parallelism) need an arch from this list.
    if bench_route.uses_carrier_hooks()
        && m.arch_id != 5
        && m.arch_id != 6
        && m.arch_id != 13
        && m.arch_id != 14
        && m.arch_id != 16
        && m.arch_id != 19
    {
        emit_uncorrelated_error(
            stdout,
            None,
            "bench_decode requires a Qwen3.5, Qwen4, Gemma4, Muse Glimmer, or Raon model",
            "unsupported",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    let context = msg
        .get("context_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(128) as usize;
    let iterations = msg.get("iterations").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let capture = msg
        .get("redline_capture")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let product_route = msg
        .get("redline_product_route")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let capture_detail = msg
        .get("redline_detail")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if capture && product_route {
        emit_uncorrelated_error(
            stdout,
            None,
            "redline_capture and redline_product_route are mutually exclusive",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    // Railgun G0: one plain-AR token either observed eagerly or
    // recorded, compared by `scripts/redline_daemon_harness.py --g0`.
    let g0 = match hipfire_engine::redline::G0Arm::from_request(msg) {
        Ok(g0) => g0,
        Err(reason) => {
            emit_uncorrelated_error(stdout, None, &reason, "validation", false, false);
            let _ = stdout.flush();
            return;
        }
    };
    if g0.is_some() && (capture || product_route || iterations != 1) {
        emit_uncorrelated_error(
            stdout,
            None,
            "g0 requires iterations == 1 and excludes redline_capture/redline_product_route",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    // Capture, the product route and G0 observe the daemon's single
    // `Gpu`; on a pipeline/tensor/expert mesh they would see one rank.
    if !bench_route.uses_carrier_hooks() && (capture || product_route || g0.is_some()) {
        emit_uncorrelated_error(
                stdout,
                None,
                "redline_capture, redline_product_route and g0 observe the daemon's single device and cannot cover a multi-device (pp/tp/ep) load",
                "unsupported",
                false,
                false,
            );
        let _ = stdout.flush();
        return;
    }
    if context == 0 || iterations == 0 {
        emit_uncorrelated_error(
            stdout,
            None,
            "bench_decode context_tokens and iterations must be non-zero",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    if context.saturating_add(iterations).saturating_add(32) > m.physical_cap {
        emit_uncorrelated_error(
            stdout,
            None,
            &format!(
                "bench_decode context+iterations exceeds loaded physical_cap={}",
                m.physical_cap
            ),
            "context_length",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    if bench_route.uses_carrier_hooks() {
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        let _ = hipfire_generate::common::reset_qwen35_recurrent(m, gpu);
    } else if let Err(reason) = hipfire_generate::bench_topology::reset_mesh_for_bench(m, gpu) {
        emit_uncorrelated_error(
            stdout,
            None,
            &format!("bench_decode state reset failed: {reason}"),
            "internal",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    // Railgun G0: both arms author the same forward, so both prime
    // from the same GDN requant frame (a host word the forward
    // consumes: kernarg +76 of gated_delta_net_*). Without this the
    // arms differ only by how many forwards ran before them.
    if g0.is_some() {
        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(0);
    }
    let synthetic: Vec<u32> = (0..context as u32).map(|i| 10 + (i % 1000)).collect();
    let prime_error: Option<String> = if !bench_route.uses_carrier_hooks() {
        // Mesh routes prime through the same prefill entrypoint as
        // `bench_prefill` (and `serve`).
        hipfire_generate::bench_topology::bench_prefill_mesh(bench_route, m, gpu, &synthetic).err()
    } else {
        match hipfire_loader::bench_decode_route(m.arch_id) {
            hipfire_loader::BenchDecodeRoute::Qwen35
            | hipfire_loader::BenchDecodeRoute::Qwen4
            | hipfire_loader::BenchDecodeRoute::Gemma4
            | hipfire_loader::BenchDecodeRoute::MuseGlimmer
            | hipfire_loader::BenchDecodeRoute::Raon => hipfire_loader::carrier_for(m.arch_id)
                .and_then(|c| c.bench_decode_prime(m, gpu, &synthetic))
                .unwrap_or_else(|| {
                    Some(format!(
                        "bench_decode_prime: carrier missing or unimplemented for arch_id={}",
                        m.arch_id
                    ))
                }),
            _ => Some(format!(
                "bench_decode unsupported for arch_id={}",
                m.arch_id
            )),
        }
    };
    let _ = gpu.hip.device_synchronize();
    if let Some(error) = prime_error {
        emit_uncorrelated_error(
            stdout,
            None,
            &format!("bench_decode prefill prime failed: {error:?}"),
            "internal",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    m.seq_pos = context;

    if capture {
        if let Err(reason) = gpu.replay.begin_capture() {
            emit_uncorrelated_error(
                stdout,
                None,
                &format!("redline decode capture refused: {reason}"),
                "unsupported",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    }
    let g0_before = match g0.map(|arm| arm.begin(gpu)).transpose() {
        Ok(before) => before,
        Err(reason) => {
            emit_uncorrelated_error(stdout, None, &reason, "unsupported", false, false);
            let _ = stdout.flush();
            return;
        }
    };

    if product_route {
        gpu.replay.begin_replay_observation_window();
    }
    let replay_before = gpu.replay.replay_observation();
    if !bench_route.uses_carrier_hooks() {
        if let Err(reason) = hipfire_generate::bench_topology::quiesce_mesh(bench_route, m) {
            emit_uncorrelated_error(
                stdout,
                None,
                &format!("bench_decode could not quiesce the loaded devices: {reason}"),
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    }
    let _ = gpu.hip.device_synchronize();
    let t0 = Instant::now();
    let mut decode_err: Option<String> = None;
    let run_ok = if !bench_route.uses_carrier_hooks() {
        // Pipeline-parallel, dense tensor-parallel and expert-parallel loads
        // step the same decode forwards `serve` steps on that topology; the
        // model devices are drained inside, so `t0..elapsed` spans every rank.
        match hipfire_generate::bench_topology::bench_decode_run_mesh(
            bench_route,
            m,
            gpu,
            context,
            iterations,
        ) {
            Ok(()) => true,
            Err(reason) => {
                decode_err = Some(reason);
                false
            }
        }
    } else {
        match hipfire_loader::bench_decode_route(m.arch_id) {
            hipfire_loader::BenchDecodeRoute::Qwen35
            | hipfire_loader::BenchDecodeRoute::Qwen4
            | hipfire_loader::BenchDecodeRoute::Gemma4
            | hipfire_loader::BenchDecodeRoute::MuseGlimmer
            | hipfire_loader::BenchDecodeRoute::Raon => hipfire_loader::carrier_for(m.arch_id)
                .and_then(|c| c.bench_decode_run(m, gpu, context, iterations, &mut decode_err))
                .unwrap_or(false),
            _ => {
                decode_err = Some(format!(
                    "bench_decode unsupported for arch_id={}",
                    m.arch_id
                ));
                false
            }
        }
    };
    let _ = gpu.hip.device_synchronize();
    let elapsed = t0.elapsed().as_secs_f64();
    let replay_after = gpu.replay.replay_observation();
    let capture_summary = if capture {
        match gpu.replay.finish_capture() {
            Ok(summary) => Some(summary),
            Err(reason) => {
                emit_uncorrelated_error(
                    stdout,
                    None,
                    &format!("redline decode capture failed: {reason}"),
                    "internal",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return;
            }
        }
    } else {
        None
    };
    let g0_result = match (g0, g0_before) {
        (Some(arm), Some(before)) => match arm.finish(gpu, before) {
            Ok(value) => Some(value),
            Err(reason) => {
                emit_uncorrelated_error(stdout, None, &reason, "internal", false, false);
                let _ = stdout.flush();
                return;
            }
        },
        _ => None,
    };

    if bench_route.uses_carrier_hooks() {
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        let _ = hipfire_generate::common::reset_qwen35_recurrent(m, gpu);
    } else if let Err(reason) = hipfire_generate::bench_topology::reset_mesh_for_bench(m, gpu) {
        emit_uncorrelated_error(
            stdout,
            None,
            &format!("bench_decode state reset failed: {reason}"),
            "internal",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    if run_ok {
        let tok_s = iterations as f64 / elapsed.max(f64::MIN_POSITIVE);
        let mut response = serde_json::json!({
            "type": "decode_result",
            "context_tokens": context,
            "iterations": iterations,
            "ms": elapsed * 1000.0,
            "us_per_token": elapsed * 1_000_000.0 / iterations as f64,
            "tok_s": tok_s,
            "topology": bench_topology.to_json(),
        });
        if let Some(summary) = capture_summary {
            response["redline_capture"] = redline_capture_json(gpu, summary, capture_detail);
        }
        if let Some(value) = g0_result {
            response["g0"] = value;
        }
        if product_route {
            let prepared = gpu.replay.prepared_route_identity().map(|identity| {
                serde_json::json!({
                    "dispatches": identity.dispatch_count,
                    "packets": identity.packet_count,
                    "queue_id": identity.queue_id,
                    "command_dwords": identity.command_dwords,
                    "queues": identity.queue_count,
                    "phases": identity.phase_count,
                })
            });
            let sequence = gpu.replay.capture_summary();
            let replay_delta = replay_after.count.saturating_sub(replay_before.count);
            response["redline_route"] = serde_json::json!({
                "requested_backend": format!("{:?}", gpu.replay.request()).to_ascii_lowercase(),
                "transport": gpu.replay.transport_name(),
                "state": format!("{:?}", gpu.replay.state()).to_ascii_lowercase(),
                "fallback_reason": gpu.replay.fallback_reason(),
                "execution_mode": "plain_ar",
                "prepared": prepared,
                "sequence": {
                    "launches": sequence.launch_count,
                    "unique_kernels": sequence.unique_kernel_count,
                    "hash": format!("{:016x}", sequence.sequence_hash),
                },
                "observed": {
                    "count_before": replay_before.count,
                    "count_after": replay_after.count,
                    "count_delta": replay_delta,
                    "first_position": replay_after.first_position,
                    "last_position": replay_after.last_position,
                },
                "retained_replay_observed": replay_delta > 0,
            });
        }
        let _ = writeln!(stdout, "{response}");
    } else {
        emit_uncorrelated_error(
            stdout,
            None,
            &match &decode_err {
                Some(e) => format!("bench_decode forward failed: {e}"),
                None => "bench_decode forward failed".to_string(),
            },
            "internal",
            false,
            false,
        );
    }
    let _ = stdout.flush();
}
