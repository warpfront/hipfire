//! Real-model gate. Product loading and state checks reuse the contracts exercised
//! by Kaden Schutt's greedy_mtp_identity_hw, qwen4_wide_verify and session_cache_hw.
//! The reference is sequential AR, not a second call to the speculative program.
use super::receipts::{ensure, number, published, real_native_window, turnover, R};
use super::Args;
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::mtp_spec::Qwen4MtpDrafter;
use hipfire_arch_qwen4::wide_verify_oracle::{self as oracle, DevForcedWindow, DevWindowSource};
use hipfire_arch_qwen4::Qwen4KvBackend;
use hipfire_loader::{load_qwen4_rank, Qwen4RankShared};
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::loader_api::{ModelSource, SpecLoadCfg};
use hipfire_runtime::serve_contract::CacheDomain;
use hipfire_runtime::session_cache::{SessionCache, SessionRoute, SessionState, SnapshotParts};
use hipfire_runtime::spec::{MtpDrafter, SpecRequestConfig, SpecTarget};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::time::Instant;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn digest(bundle: &mut Qwen4Bundle, gpu: &Gpu, route: SessionRoute) -> R<Vec<String>> {
    let pos = bundle.state.position;
    let SnapshotParts { meta, layout } = SessionState::snapshot_parts(bundle, route, pos)?;
    gpu.hip.device_synchronize().map_err(err)?;
    let mut out = vec![format!("{:x}", Sha256::digest(&meta))];
    for (buf, offset, bytes) in layout
        .fixed
        .iter()
        .map(|p| (p.buf, p.offset, p.bytes))
        .chain(layout.rows.iter().map(|p| (p.buf, 0, p.rows * p.row_bytes)))
    {
        let mut host = vec![0; bytes];
        gpu.hip
            .memcpy_dtoh_at(&mut host, buf, offset)
            .map_err(err)?;
        out.push(format!("{:x}", Sha256::digest(&host)));
    }
    Ok(out)
}
fn ar_prefill(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    prompt: &[u32],
    logits: &GpuTensor,
) -> R<()> {
    bundle.reset(gpu).map_err(err)?;
    bundle
        .forward_chunk_final(gpu, prompt, logits, None)
        .map_err(err)
}
#[path = "checkpoint.rs"]
mod checkpoint;
use checkpoint::HostCheckpoint;

struct ArReference {
    ids: Vec<u32>,
    states: BTreeMap<usize, Vec<String>>,
    logits_bits: BTreeMap<usize, Vec<u32>>,
    wall_ms: f64,
}
fn ar_reference(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    prompt: &[u32],
    logits: &GpuTensor,
    count: usize,
) -> R<ArReference> {
    ar_prefill(bundle, gpu, prompt, logits)?;
    let initial_logits = gpu.download_f32(logits).map_err(err)?;
    ensure(
        initial_logits.iter().all(|x| x.is_finite()),
        "nonfinite AR prefill logits",
    )?;
    let mut logits_bits = BTreeMap::new();
    let mut states = BTreeMap::new();
    states.insert(0, digest(bundle, gpu, SessionRoute::Ar)?);
    let start = Instant::now();
    let mut ids = Vec::new();
    for i in 1..=count {
        ids.push(
            bundle
                .forward_token_or_argmax(gpu, None, logits)
                .map_err(err)?,
        );
        if i <= 9 || i == count {
            states.insert(i, digest(bundle, gpu, SessionRoute::Ar)?);
            let row = gpu.download_f32(logits).map_err(err)?;
            ensure(
                row.iter().all(|x| x.is_finite()),
                "nonfinite AR reference logits",
            )?;
            if i <= 9 {
                logits_bits.insert(i, row.into_iter().map(f32::to_bits).collect());
            }
        }
    }
    gpu.hip.device_synchronize().map_err(err)?;
    Ok(ArReference {
        ids,
        states,
        logits_bits,
        wall_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}
fn check_state(bundle: &mut Qwen4Bundle, gpu: &Gpu, want: &[String], label: &str) -> R<()> {
    let got = digest(bundle, gpu, SessionRoute::Ar)?;
    ensure(
        got == want,
        format!(
            "{label}: complete target state differs (first part {:?})",
            got.iter().zip(want).position(|(a, b)| a != b)
        ),
    )
}
fn target_transactions(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    prompt: &[u32],
    logits: &GpuTensor,
    ar: &ArReference,
    _k: usize,
) -> R<Value> {
    ar_prefill(bundle, gpu, prompt, logits)?;
    ensure(
        oracle::dev_effective_row_exact_rows(bundle, gpu) >= 8,
        "requested R8 exceeds live qualified row limit",
    )?;
    let mut scratch = SpecTarget::new_spec_scratch(bundle, gpu, 8)?;
    let mut verified_complete = false;
    let result = (|| -> R<Value> {
        let s0 = HostCheckpoint::capture(bundle, gpu)?;
        verified_complete = true;
        check_state(bundle, gpu, &ar.states[&0], "captured independent S0")?;
        let mut cells = Vec::new();
        for rows in [1, 2, 4, 8] {
            verified_complete = false;
            s0.restore(bundle, gpu)?;
            verified_complete = true;
            scratch.clear();
            check_state(bundle, gpu, &ar.states[&0], "host S0 restore")?;
            let block = &ar.ids[..rows];
            verified_complete = false;
            let (picks, values) = oracle::dev_verify_logits(bundle, gpu, block, scratch.as_mut())?;
            verified_complete = true;
            ensure(
                picks == ar.ids[1..=rows],
                format!("verify R{rows} picks differ from AR"),
            )?;
            let vocab = bundle.config.vocab_size;
            ensure(
                values.len() == rows * vocab && values.iter().all(|x| x.is_finite()),
                "verify logits extent/finiteness",
            )?;
            for row in 0..rows {
                ensure(
                    values[row * vocab..(row + 1) * vocab]
                        .iter()
                        .map(|x| x.to_bits())
                        .eq(ar.logits_bits[&(row + 1)].iter().copied()),
                    format!("verify R{rows} row{row}: full vocabulary differs bitwise from AR"),
                )?;
            }
            check_state(bundle, gpu, &ar.states[&rows], "verify full state")?;
            verified_complete = false;
            oracle::dev_rollback_verify(bundle, gpu, scratch.as_mut())?;
            let restored = digest(bundle, gpu, SessionRoute::Ar)?;
            verified_complete = true;
            ensure(
                restored == ar.states[&0],
                "abandon changed S0 before any checkpoint restore",
            )?;
            // Reusing the same scratch proves its verified ticket ended.
            verified_complete = false;
            let _ = oracle::dev_verify_logits(bundle, gpu, block, scratch.as_mut())?;
            verified_complete = false;
            oracle::dev_rollback_verify(bundle, gpu, scratch.as_mut())?;
            let restored = digest(bundle, gpu, SessionRoute::Ar)?;
            verified_complete = true;
            ensure(restored == ar.states[&0], "second abandon changed S0")?;
            for accept in 0..rows {
                for native in [false, true] {
                    verified_complete = false;
                    s0.restore(bundle, gpu)?;
                    verified_complete = true;
                    scratch.clear();
                    check_state(bundle, gpu, &ar.states[&0], "accepted-prefix S0 restore")?;
                    verified_complete = false;
                    let _ = oracle::dev_verify_logits(bundle, gpu, block, scratch.as_mut())?;
                    verified_complete = false;
                    if native {
                        oracle::dev_commit_native_retain(
                            bundle,
                            gpu,
                            block,
                            accept,
                            scratch.as_mut(),
                        )?;
                    } else {
                        SpecTarget::commit_prefix(
                            bundle,
                            gpu,
                            block,
                            accept,
                            prompt.len(),
                            scratch.as_mut(),
                        )?;
                    }
                    let keep = accept + 1;
                    let now = digest(bundle, gpu, SessionRoute::Ar)?;
                    verified_complete = true;
                    ensure(
                        now == ar.states[&keep],
                        "accepted-prefix target state differs from AR",
                    )?;
                    verified_complete = false;
                    bundle
                        .forward_token(gpu, ar.ids[keep], logits, None)
                        .map_err(err)?;
                    let now = digest(bundle, gpu, SessionRoute::Ar)?;
                    verified_complete = true;
                    ensure(
                        now == ar.states[&(keep + 1)],
                        "post-commit AR continuation differs",
                    )?;
                    cells.push(json!({"rows":rows,"accepted_drafts":accept,"consumed_rows":keep,"path":if native{"native_retain"}else{"generic_commit_prefix"},"state_parts":now.len(),"all_state_exact":true,"subsequent_ar_checked":true}));
                }
            }
        }
        Ok(
            json!({"cells":cells,"abandon_and_ticket_reuse_checked":true,"s0_checkpoint":"host-owned public SessionState layout; no nested target ticket","s0_host_bytes":s0.bytes(),"full_logits_bitwise_claim":true,"limits":"Full vocabulary checked against sequential AR only at qualified R1/2/4/8."}),
        )
    })();
    match result {
        Ok(report) => {
            scratch.free(gpu);
            Ok(report)
        }
        Err(primary) => {
            let cleanup = if verified_complete {
                SpecTarget::abandon_verify(bundle, gpu, scratch.as_mut())
            } else {
                Err(
                    "cleanup deferred: last GPU boundary not proven complete; owners retained"
                        .into(),
                )
            };
            // Never Drop/assert an active scratch over the original failure.
            // The caller also retains bundle, logits and GPU on this Err.
            std::mem::forget(scratch);
            match cleanup {
                Ok(()) => Err(primary),
                Err(error) => Err(format!("{primary}; abandon/cleanup: {error}")),
            }
        }
    }
}

fn prefill(
    d: &mut Qwen4MtpDrafter,
    b: &mut Qwen4Bundle,
    g: &mut Gpu,
    p: &[u32],
    reused: usize,
) -> R<u32> {
    d.mtp_prefill(g, b, p, &p[reused..], reused, reused > 0, &|| false)?
        .into_first_token()
}
fn flush(d: &mut Qwen4MtpDrafter, b: &mut Qwen4Bundle, g: &mut Gpu, seed: u32) -> R<()> {
    let position = b.state.position;
    ensure(
        d.mtp_forced_advance(g, b, &[seed], position, &|| false)?,
        "terminal pending-seed flush was not handled",
    )
}
fn snapshot(bundle: &Qwen4Bundle, cache: bool) -> R<Value> {
    if cache {
        bundle.dev_expert_cache_snapshot().map_err(err)
    } else {
        Ok(Value::Null)
    }
}
fn native_walk(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    prompt: &[u32],
    logits: &GpuTensor,
    ar: &ArReference,
    k: usize,
    cache: bool,
) -> R<Value> {
    let mut d = Qwen4MtpDrafter::new(k, prompt.len() + ar.ids.len() + 128, None);
    d.configure_request(SpecRequestConfig::default());
    let mut seed = prefill(&mut d, bundle, gpu, prompt, 0)?;
    ensure(seed == ar.ids[0], "MTP prefill seed differs from AR")?;
    let mut ids = vec![seed];
    let mut frames = Vec::new();
    let mut acceptance = [false; 3];
    let mut repair_candidate = None;
    let cache_before = snapshot(bundle, cache)?;
    let mut native_cache_end = cache_before.clone();
    let mut seen_native_pending = Vec::new();
    let start = Instant::now();
    while ids.len() < ar.ids.len() {
        let remain = ar.ids.len() - ids.len();
        let drafts = k.min(remain.saturating_sub(1));
        let position = bundle.state.position;
        let before = snapshot(bundle, cache)?;
        let measured = oracle::dev_run_forced_window(
            &mut d,
            bundle,
            gpu,
            position,
            seed,
            &ids,
            u32::MAX,
            DevForcedWindow {
                source: if drafts == 0 {
                    DevWindowSource::ArRouted
                } else {
                    DevWindowSource::Mtp
                },
                drafts,
                timing_only: false,
            },
        )?;
        let after = snapshot(bundle, cache)?;
        let v = json!({"source":format!("{:?}",measured.source),"route":format!("{:?}",measured.route),"requested_drafts":measured.requested_drafts,"actual_drafts":measured.actual_drafts,"accepted_prefix":measured.accepted_prefix,"committed":measured.window.committed.len(),"outcome_mismatch":measured.outcome_mismatch,"position_before":position,"position_after":bundle.state.position,"total_us":measured.total_us});
        ensure(
            !measured.window.committed.is_empty(),
            "empty native/AR tail window",
        )?;
        if drafts > 0 {
            real_native_window(&v)?;
            native_cache_end = after.clone();
            if cache {
                if let Some(p) = after["pending"].as_array() {
                    seen_native_pending.extend(p.iter().cloned());
                }
            }
            if measured.accepted_prefix == 0 {
                acceptance[0] = true;
            } else if measured.accepted_prefix == drafts {
                acceptance[2] = true;
            } else {
                acceptance[1] = true;
            }
            if repair_candidate.is_none() && measured.window.committed.len() > 1 {
                repair_candidate = Some(frames.len());
            }
        }
        let end = ids.len() + measured.window.committed.len();
        ensure(
            end <= ar.ids.len(),
            "window exceeded remaining token budget",
        )?;
        ensure(
            measured.window.committed == ar.ids[ids.len()..end],
            "true MTP window token IDs differ from independent sequential AR",
        )?;
        seed = *measured
            .window
            .committed
            .last()
            .ok_or("missing pending seed")?;
        ids.extend(&measured.window.committed);
        frames.push(json!({"measurement":v,"committed_ids":measured.window.committed,"cache_stats_before":before["stats"],"cache_stats_after":after["stats"],"pending_after":after["pending"]}));
    }
    let cache_after = native_cache_end;
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    flush(&mut d, bundle, gpu, seed)?;
    check_state(
        bundle,
        gpu,
        &ar.states[&ar.ids.len()],
        "native final target state",
    )?;
    let native_target_state = digest(bundle, gpu, SessionRoute::Ar)?;
    let native_state = digest(bundle, gpu, SessionRoute::Mtp)?;
    let pending = oracle::dev_head_pending_hidden(&d, gpu)?;
    ensure(
        pending.len() % 4 == 0
            && pending
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().expect("f32 bytes")).is_finite()),
        "nonfinite pending native hidden",
    )?;
    // Teacher-forced head/target reference consumes every output row, with no
    // speculative acceptance. Its different head schedule is diagnostic only.
    let _ = prefill(&mut d, bundle, gpu, prompt, 0)?;
    let position = bundle.state.position;
    ensure(
        d.mtp_forced_advance(gpu, bundle, &ids, position, &|| false)?,
        "teacher-forced reference not handled",
    )?;
    // Cross-head-schedule equality is diagnostic: native generation and
    // teacher-forced append can have different rounding and draft policy
    // full_steps/margin metadata. Match cache-on/off on the SAME native schedule.
    let teacher_state = digest(bundle, gpu, SessionRoute::Mtp)?;
    let teacher_pending = oracle::dev_head_pending_hidden(&d, gpu)?;
    let teacher_different_parts: Vec<_> = (0..native_state.len().max(teacher_state.len()))
        .filter(|i| native_state.get(*i) != teacher_state.get(*i))
        .collect();
    let teacher_pending_exact = teacher_pending == pending;
    let pending_sha256 = format!("{:x}", Sha256::digest(&pending));
    ensure(acceptance.iter().all(|x|*x),format!("missing actual native acceptance cell zero/interior/full: {acceptance:?}; no inferred coverage"))?;
    let changed = if cache {
        published(&cache_before)?;
        published(&cache_after)?;
        ensure(
            number(&cache_after["stats"], "cache_hits")?
                > number(&cache_before["stats"], "cache_hits")?,
            "native interval had no actual cache hit",
        )?;
        ensure(
            number(&cache_after["stats"], "prefill_restores")?
                == number(&cache_before["stats"], "prefill_restores")?,
            "native verify interval triggered bulk donor restoration",
        )?;
        let batch_rows = frames
            .iter()
            .filter(|f| {
                f["measurement"]["source"] == "Mtp" && f["measurement"]["route"] == "Batched"
            })
            .filter_map(|f| f["measurement"]["actual_drafts"].as_u64().map(|n| n + 1))
            .max();
        if let Some(rows) = batch_rows {
            ensure(
                number(&cache_after["stats"], "verify_batches")?
                    > number(&cache_before["stats"], "verify_batches")?,
                "batched MTP did not retire a production verify guard",
            )?;
            ensure(
                number(&cache_after["stats"], "max_guard_rows")? >= rows,
                "production verify guard did not span the actual batch width",
            )?;
        }
        let initial_pending = cache_before["pending"]
            .as_array()
            .ok_or("missing initial pending tickets")?;
        let changed = turnover(&cache_before, &cache_after)?;
        let changed:Vec<_>=changed.into_iter().filter(|t| {
            let ticket=json!({"layer":t["layer"],"slot":t["slot"],"expert":t["to_expert"],"generation":t["generation_after"]});
            !initial_pending.contains(&ticket)&&seen_native_pending.contains(&ticket)
        }).collect();
        ensure(
            !changed.is_empty(),
            "no completed victim ticket was actually enqueued during native-only windows",
        )?;
        changed
    } else {
        Vec::new()
    };
    let candidate = repair_candidate.ok_or("no native window with an interior terminal prefix")?;
    let terminal = terminal_repair(
        bundle,
        gpu,
        prompt,
        logits,
        ar.ids[0],
        &frames[..=candidate],
        k,
    )?;
    Box::new(d).mtp_free(gpu);
    Ok(
        json!({"ids":ids,"windows":frames,"actual_acceptance_cells":{"zero":acceptance[0],"interior":acceptance[1],"full":acceptance[2]},"completed_victim_turnover":changed,"cache_before_native":cache_before,"cache_after_native":cache_after,"native_target_all_state_exact":true,"native_target_state_sha256":native_target_state,"native_mtp_state_sha256":native_state,"native_head_teacher_forced_comparison":"diagnostic_cross_head_schedule","teacher_forced_different_parts":teacher_different_parts,"teacher_forced_state_sha256":teacher_state,"teacher_forced_pending_exact":teacher_pending_exact,"native_pending_hidden_sha256":pending_sha256,"terminal":terminal,"instrumented_wall_ms":wall_ms,"performance_claim":false}),
    )
}
/// Decode the saved executor receipt, never infer native execution from K.
fn replay_request(frame: &Value) -> R<(DevWindowSource, usize, Vec<u32>)> {
    let m = &frame["measurement"];
    let source = match m["source"].as_str() {
        Some("Mtp") => DevWindowSource::Mtp,
        Some("ArRouted") => DevWindowSource::ArRouted,
        _ => return Err("terminal replay requires a saved native/AR executor source".into()),
    };
    let drafts = usize::try_from(number(m, "actual_drafts")?).map_err(err)?;
    ensure(
        (source == DevWindowSource::ArRouted) == (drafts == 0),
        "terminal replay source/width mismatch",
    )?;
    if source == DevWindowSource::Mtp {
        real_native_window(m)?;
    }
    let committed = frame["committed_ids"]
        .as_array()
        .ok_or("terminal replay missing committed IDs")?
        .iter()
        .map(|id| {
            u32::try_from(id.as_u64().ok_or("terminal replay invalid token ID")?).map_err(err)
        })
        .collect::<R<Vec<_>>>()?;
    ensure(
        !committed.is_empty() && committed.len() as u64 == number(m, "committed")?,
        "terminal replay committed extent mismatch",
    )?;
    Ok((source, drafts, committed))
}
fn same_replay_window(expected: &Value, actual: &Value, ids: &[u32]) -> R<()> {
    for key in [
        "source",
        "route",
        "requested_drafts",
        "actual_drafts",
        "accepted_prefix",
        "committed",
        "outcome_mismatch",
        "position_before",
        "position_after",
    ] {
        ensure(
            actual.get(key).is_some() && actual.get(key) == expected["measurement"].get(key),
            format!("terminal native replay changed {key}"),
        )?;
    }
    ensure(
        json!(ids) == expected["committed_ids"],
        "terminal native replay changed committed token IDs",
    )
}
fn terminal_repair(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    prompt: &[u32],
    logits: &GpuTensor,
    first_seed: u32,
    saved: &[Value],
    k: usize,
) -> R<Value> {
    ensure(
        !saved.is_empty(),
        "terminal replay has no saved native window",
    )?;
    let mut emitted = vec![first_seed];
    let first_position = number(&saved[0]["measurement"], "position_before")?;
    ensure(
        first_position == prompt.len() as u64,
        "terminal replay starts after the original prompt",
    )?;
    let mut reference_tokens = Vec::new();
    let mut repaired = false;
    for (index, frame) in saved.iter().enumerate() {
        let (_, _, committed) = replay_request(frame)?;
        if index + 1 == saved.len() {
            ensure(
                committed.len() > 1,
                "terminal candidate has no strict interior output prefix",
            )?;
            reference_tokens.push(committed[0]);
        } else {
            reference_tokens.extend(committed);
        }
    }
    // The original prompt's first output is the input seed of the first
    // native window. Sequential AR consumes it before every committed output
    // except the final pending seed. Use the seed already proved by the
    // independent AR pass, then force the exact saved history, rather than
    // cold-prefilling it.
    ar_prefill(bundle, gpu, prompt, logits)?;
    let values = gpu.download_f32(logits).map_err(err)?;
    ensure(
        values.iter().all(|x| x.is_finite()),
        "nonfinite terminal AR prompt logits",
    )?;
    let mut history = prompt.to_vec();
    history.push(first_seed);
    history.extend_from_slice(&reference_tokens);
    for &token in history[prompt.len()..].iter() {
        bundle
            .forward_token(gpu, token, logits, None)
            .map_err(err)?;
    }
    let ar_target = digest(bundle, gpu, SessionRoute::Ar)?;

    let mut d = Qwen4MtpDrafter::new(k, bundle.config.max_position_embeddings.min(16384), None);
    d.configure_request(SpecRequestConfig {
        stop_sequences: true,
        ..SpecRequestConfig::default()
    });
    let mut seed = prefill(&mut d, bundle, gpu, prompt, 0)?;
    ensure(
        seed == first_seed,
        "terminal original-prompt seed differs from sequential AR",
    )?;
    let mut replay_frames = Vec::new();
    for (index, frame) in saved.iter().enumerate() {
        let (source, drafts, committed) = replay_request(frame)?;
        let position = bundle.state.position;
        let measured = oracle::dev_run_forced_window(
            &mut d,
            bundle,
            gpu,
            position,
            seed,
            &emitted,
            u32::MAX,
            DevForcedWindow {
                source,
                drafts,
                timing_only: false,
            },
        )?;
        let m = json!({"source":format!("{:?}",measured.source),"route":format!("{:?}",measured.route),"requested_drafts":measured.requested_drafts,"actual_drafts":measured.actual_drafts,"accepted_prefix":measured.accepted_prefix,"committed":measured.window.committed.len(),"outcome_mismatch":measured.outcome_mismatch,"position_before":position,"position_after":bundle.state.position,"total_us":measured.total_us});
        same_replay_window(frame, &m, &measured.window.committed)?;
        replay_frames.push(json!({"measurement":m,"committed_ids":measured.window.committed}));
        if index + 1 == saved.len() {
            ensure(
                source == DevWindowSource::Mtp,
                "terminal candidate was not native MTP",
            )?;
            let consumed = &committed[..1];
            ensure(
                d.mtp_repair_terminal_prefix(gpu, bundle, position, seed, consumed)?,
                "native terminal prefix was not repaired",
            )?;
            flush(&mut d, bundle, gpu, consumed[0])?;
            repaired = true;
        } else {
            seed = *committed.last().ok_or("empty terminal replay window")?;
            emitted.extend_from_slice(&committed);
        }
    }
    ensure(repaired, "terminal candidate was not repaired")?;
    ensure(
        bundle.state.position == history.len(),
        "terminal repaired position includes rejected suffix",
    )?;
    let live = digest(bundle, gpu, SessionRoute::Mtp)?;
    let live_target = digest(bundle, gpu, SessionRoute::Ar)?;
    ensure(
        live_target == ar_target,
        "terminal target differs from independent sequential AR consumed history",
    )?;
    let pending = oracle::dev_head_pending_hidden(&d, gpu)?;
    ensure(
        !pending.is_empty()
            && pending.len() % 4 == 0
            && pending
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().expect("f32 bytes")).is_finite()),
        "nonfinite terminal repaired pending hidden",
    )?;
    bundle.session_commit_live(&history);
    let mut next = history.clone();
    next.extend_from_slice(&prompt[..prompt.len().min(16)]);
    let before_plan = digest(bundle, gpu, SessionRoute::Mtp)?;
    ensure(
        bundle.session_plan(&next, SessionRoute::Mtp) == history.len(),
        "terminal history did not continue live",
    )?;
    ensure(
        digest(bundle, gpu, SessionRoute::Mtp)? == before_plan,
        "in-place session plan changed repaired state",
    )?;
    let next_seed = prefill(&mut d, bundle, gpu, &next, history.len())?;
    ensure(
        (next_seed as usize) < bundle.config.vocab_size,
        "next live turn returned invalid seed",
    )?;
    ensure(
        bundle.state.position == next.len()
            && SessionState::live_position(bundle, SessionRoute::Mtp) == Some(next.len()),
        "next live turn target/head positions differ",
    )?;
    Box::new(d).mtp_free(gpu);
    Ok(
        json!({"source":"native_mtp","verified_drafts":number(&saved.last().ok_or("missing terminal frame")?["measurement"],"actual_drafts")?,"consumed_outputs":1,"retained_prefix_repaired":true,"pending_seed_flushed":true,"native_schedule_exact":true,"replay_frames":replay_frames,"target_state_exact":true,"reference_scope":"original prompt then forced sequential AR over identical emitted and consumed tokens; no cold prefill of generated history","repaired_target_state_sha256":live_target,"repaired_target_ar_reference_sha256":ar_target,"mtp_repaired_state_sha256":live,"native_pending_hidden_sha256":format!("{:x}",Sha256::digest(&pending)),"live_next_turn_in_place":true,"live_plan_state_unchanged":true,"live_next_turn_positions_exact":true,"live_next_turn_exact":false,"live_next_turn_seed":next_seed,"wire_terminal_coverage":false,"limits":["Head exactness requires cache-off/on comparison under the same saved native replay schedule.","Next live turn proves in-place state preservation and aligned target/head positions, not an independent seed or output reference.","Canonical serving terminal/session behavior remains a separate qualification."]}),
    )
}
fn prefix_restore(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    prompt: &[u32],
    k: usize,
    max_seq: usize,
) -> R<Value> {
    let chunk = bundle.spec_chunk_rows().ok_or("missing prompt chunk")?;
    ensure(
        2 * chunk + 128 + k + 1 <= max_seq,
        "max_seq too short for two-link MTP prefix restore",
    )?;
    let a: Vec<_> = (0..2 * chunk + 128)
        .map(|i| prompt[i % prompt.len()])
        .collect();
    let mut b = a[..chunk].to_vec();
    b.extend((0..chunk + 128).map(|i| prompt[(i + 19) % prompt.len()]));
    let mut d = Qwen4MtpDrafter::new(k, max_seq, None);
    let mut cold = Vec::new();
    for tokens in [&a, &b] {
        let seed = prefill(&mut d, bundle, gpu, tokens, 0)?;
        cold.push((digest(bundle, gpu, SessionRoute::Mtp)?, seed));
        bundle.session_commit();
    }
    let mut cells = Vec::new();
    for (i, tokens) in [&a, &b].iter().enumerate() {
        let reused = bundle.session_plan(tokens, SessionRoute::Mtp);
        ensure(
            reused >= chunk,
            format!("prefix {i}: no real cached root to restore"),
        )?;
        let seed = prefill(&mut d, bundle, gpu, tokens, reused)?;
        ensure(
            seed == cold[i].1 && digest(bundle, gpu, SessionRoute::Mtp)? == cold[i].0,
            "restored MTP prefix differs from cold head/target",
        )?;
        bundle.session_commit();
        cells.push(json!({"branch":i,"reused":reused,"chunk":chunk,"all_mtp_state_exact":true,"seed_exact":true}));
    }
    let outcome = d.mtp_prefill(gpu, bundle, prompt, prompt, 0, false, &|| true)?;
    ensure(
        matches!(outcome, hipfire_runtime::spec::PrefillOutcome::Aborted),
        "prefill abort was not classified as aborted",
    )?;
    Box::new(d).mtp_free(gpu);
    Ok(json!({"cells":cells,"prefill_abort_observed":true,"disk_restart_coverage":false}))
}
pub(super) fn run(args: &Args) -> R<Value> {
    let path = args.model.as_ref().ok_or("missing model")?;
    let raw = path.to_str().ok_or("non-UTF8 model path")?;
    let mut source = ModelSource::from_path(raw)?;
    let tokenizer = match &source {
        ModelSource::Hfq(h) => Tokenizer::from_hfq_metadata(&h.metadata_json).map_err(err)?,
        _ => return Err("requires HFQ model".into()),
    };
    hipfire_loader::prepare_host_memory_for(path);
    let mut gpu = Gpu::init().map_err(err)?;
    ensure(
        gpu.arch == "gfx1201",
        "native MTP/LRU qualification requires exact gfx1201",
    )?;
    hipfire_loader::admission::resolve_qwen4_ple_rows(&mut source, raw, &gpu.arch)?;
    let domain = match &source {
        ModelSource::Hfq(h) => CacheDomain::for_model(h, &tokenizer, None, "qwen4", gpu.device_id),
        _ => return Err("requires HFQ".into()),
    };
    let backend = Qwen4KvBackend::automatic(&gpu);
    let granularity = match backend {
        Qwen4KvBackend::Vmm => gpu.vmm_recommended_granularity().map_err(err)?,
        Qwen4KvBackend::Legacy => 1,
    };
    let shared = Qwen4RankShared {
        continuous_batch: false,
        path: raw.into(),
        max_seq: args.max_seq,
        sequence: None,
        backend,
        vmm_granularity: granularity,
        kv_mode: hipfire_runtime::config::get().kv_mode.clone(),
        state_quant: Some("q8".into()),
        spec: SpecLoadCfg {
            mtp: Some(true),
            ..SpecLoadCfg::default()
        },
        native_mtp: true,
        max_k: 7,
    };
    let loaded = load_qwen4_rank(&mut gpu, None, &shared, None, None, source)?;
    ensure(
        loaded.native_mtp,
        "product loader declined native MTP: fallback is not qualification",
    )?;
    let mut bundle = loaded.bundle;
    bundle.attach_session_cache(SessionCache::new(domain, 512 << 20));
    let cache = hipfire_arch_qwen4::expert_cache_config::ExpertCacheConfig::from_env()?.is_some();
    let logits = gpu
        .zeros(&[bundle.config.vocab_size], DType::F32)
        .map_err(err)?;
    let result = (|| -> R<Value> {
        ensure(
            bundle.state_format().gdn.name() == "q8",
            "oracle must preserve plain-Q8 product policy",
        )?;
        // Both arms reserve K7, while actual native windows below remain K3
        // (or the explicitly requested K). R8 target proof needs no second load.
        let sizing = SpecTarget::new_spec_scratch(&mut bundle, &mut gpu, 8)?;
        sizing.free(&mut gpu);
        ensure(
            oracle::dev_effective_row_exact_rows(&bundle, &gpu) >= 8,
            "product load did not admit the requested R8 target proof",
        )?;
        let mut cases = Vec::new();
        for file in &args.prompts {
            let bytes = std::fs::read(file).map_err(err)?;
            let text = std::str::from_utf8(&bytes).map_err(err)?;
            let prompt = tokenizer.encode(text);
            ensure(
                !prompt.is_empty() && prompt.len() + args.tokens + args.k + 1 < loaded.max_seq,
                "prompt/context bounds",
            )?;
            let ar = ar_reference(&mut bundle, &mut gpu, &prompt, &logits, args.tokens.max(9))?;
            // Native goes first so target-oracle batches cannot prime its guard witness.
            let native = native_walk(&mut bundle, &mut gpu, &prompt, &logits, &ar, args.k, cache)?;
            let target = target_transactions(&mut bundle, &mut gpu, &prompt, &logits, &ar, args.k)?;
            let prefix = prefix_restore(&mut bundle, &mut gpu, &prompt, args.k, loaded.max_seq)?;
            cases.push(json!({"prompt":file,"prompt_sha256":format!("{:x}",Sha256::digest(&bytes)),"prompt_ids_sha256":format!("{:x}",Sha256::digest(prompt.iter().flat_map(|t|t.to_le_bytes()).collect::<Vec<_>>())),"prompt_tokens":prompt.len(),"ar_ids":ar.ids,"ar_final_state_sha256":ar.states[&ar.ids.len()],"ar_instrumented_wall_ms":ar.wall_ms,"target_transactions":target,"native":native,"prefix_restore":prefix}));
        }
        Ok(
            json!({"native_mtp_attached":true,"cache_enabled":cache,"native_k":args.k,"head_reserve_k":7,"target_rows":[1,2,4,8],"max_seq":loaded.max_seq,"state_format":format!("{:?}",bundle.state_format()),"kv_backend":backend.name(),"replay_backend":format!("{:?}",gpu.replay.request()),"native_pm4_replay_qualified":false,"scope":"greedy native-MTP transactions and exact expert-cache placement","cases":cases,"performance_claim":false,"limits":["Forced native widths prove execution, not default-policy speed.","AR/state digest and per-window snapshot collection instrument wall timings.","No SSE terminal, sampled MTP, ngram copy, native PM4 replay or live Q8Ef qualification.","Generic commit on this product load uses captured-row repair; its no-row-capture fallback is not claimed.","Bundle state entrypoints under an intentionally busy guard are not GPU-qualified by this tool."]}),
        )
    })();
    if let Err(error) = result {
        // A failed speculative tile may leave PM4 quiescence unknown. Retain
        // every dependent owner, including output tensors, until process exit.
        std::mem::forget(logits);
        std::mem::forget(bundle);
        std::mem::forget(gpu);
        return Err(error);
    }
    let report = result?;
    // Shutdown must prove retained/HIP quiescence before output storage is freed.
    if let Err(error) = bundle.free_gpu(&mut gpu) {
        std::mem::forget(logits);
        std::mem::forget(gpu);
        return Err(error.to_string());
    }
    gpu.free_tensor(logits).map_err(err)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame() -> Value {
        json!({"measurement":{"source":"Mtp","route":"Batched","requested_drafts":3,"actual_drafts":3,"accepted_prefix":1,"committed":2,"outcome_mismatch":false,"position_before":100,"position_after":102,"total_us":1.0},"committed_ids":[11,12]})
    }
    #[test]
    fn terminal_replay_requires_saved_executor_schedule_and_exact_token_history() {
        let saved = frame();
        let (source, drafts, ids) = replay_request(&saved).unwrap();
        assert_eq!(source, DevWindowSource::Mtp);
        assert_eq!(drafts, 3);
        assert_eq!(ids, [11, 12]);
        let mut actual = saved["measurement"].clone();
        actual["total_us"] = json!(99.0);
        same_replay_window(&saved, &actual, &ids).unwrap();
        for (key, value) in [
            ("source", json!("ArRouted")),
            ("route", json!("Interleaved")),
            ("requested_drafts", json!(2)),
            ("actual_drafts", json!(2)),
            ("accepted_prefix", json!(0)),
            ("committed", json!(1)),
            ("outcome_mismatch", json!(true)),
            ("position_before", json!(99)),
            ("position_after", json!(103)),
        ] {
            let mut bad = actual.clone();
            bad[key] = value;
            assert!(same_replay_window(&saved, &bad, &ids).is_err(), "{key}");
        }
        assert!(same_replay_window(&saved, &actual, &[11, 99]).is_err());
        let mut bad = saved.clone();
        bad["measurement"]["source"] = json!("Ngram");
        assert!(replay_request(&bad).is_err());
        bad = saved.clone();
        bad["committed_ids"][1] = json!(-1);
        assert!(replay_request(&bad).is_err());
        bad = saved;
        bad["committed_ids"] = json!([11]);
        assert!(replay_request(&bad).is_err());
    }
}
