//! Real-model gate. Product loading and state checks reuse the contracts exercised
//! by Kaden Schutt's greedy_mtp_identity_hw, qwen4_wide_verify and session_cache_hw.
//! The reference is sequential AR, not a second call to the speculative program.
use super::receipts::{
    ensure, number, pending as pending_tickets, published, real_native_window, target_projection,
    ticket, turnover, R,
};
use super::Args;
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::config::LayerType;
use hipfire_arch_qwen4::expert_tier_owners::{ExpertReadView, ExpertTierLease};
use hipfire_arch_qwen4::expert_tier_plan::Placement;
use hipfire_arch_qwen4::mtp_spec::Qwen4MtpDrafter;
use hipfire_arch_qwen4::wide_verify_oracle::{self as oracle, DevForcedWindow, DevWindowSource};
use hipfire_arch_qwen4::Qwen4KvBackend;
use hipfire_loader::{load_qwen4_rank, Qwen4RankShared};
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::loader_api::{ModelSource, SpecLoadCfg};
use hipfire_runtime::serve_contract::CacheDomain;
use hipfire_runtime::session_cache::{SessionCache, SessionRoute, SessionState};
use hipfire_runtime::spec::{MtpDrafter, SpecRequestConfig, SpecTarget};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::mem::ManuallyDrop;
use std::time::Instant;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
struct PartitionSource {
    artifact_bytes: u64,
    artifact_index_sha256: String,
    layers: u32,
    experts_per_layer: u32,
}
#[derive(Clone)]
struct SemanticExpert {
    layer: u32,
    expert: u32,
    original_device: bool,
    gate_up_qtype: u8,
    down_qtype: u8,
    gate_up_bytes: u64,
    down_bytes: u64,
}
fn semantic_partition_digest(source: &PartitionSource, rows: &[SemanticExpert]) -> R<String> {
    ensure(
        source.layers > 0 && source.experts_per_layer > 0,
        "empty partition geometry",
    )?;
    ensure(
        source.artifact_index_sha256.len() == 64,
        "missing actual artifact index identity",
    )?;
    let expected = u64::from(source.layers)
        .checked_mul(u64::from(source.experts_per_layer))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or("partition extent overflow")?;
    ensure(
        rows.len() == expected,
        "missing original expert partition rows",
    )?;
    let mut digest = Sha256::new();
    digest.update(b"hipfire.fn.original-expert-partition.v1");
    digest.update(serde_json::to_vec(source).map_err(err)?);
    for (index, row) in rows.iter().enumerate() {
        let layer = (index / source.experts_per_layer as usize) as u32;
        let expert = (index % source.experts_per_layer as usize) as u32;
        ensure(
            (row.layer, row.expert) == (layer, expert),
            "duplicate, reordered or missing original expert",
        )?;
        ensure(
            row.gate_up_bytes > 0 && row.down_bytes > 0,
            "empty original projection",
        )?;
        digest.update(row.layer.to_le_bytes());
        digest.update(row.expert.to_le_bytes());
        digest.update([
            u8::from(row.original_device),
            row.gate_up_qtype,
            row.down_qtype,
        ]);
        digest.update(row.gate_up_bytes.to_le_bytes());
        digest.update(row.down_bytes.to_le_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn actual_semantic_partition(
    lease: &ExpertTierLease,
    source: &PartitionSource,
) -> R<(String, u64)> {
    let mut rows = Vec::new();
    let mut resident = 0u64;
    for layer in 0..source.layers {
        // Resident is the original frozen pin. Cache donor mirrors appear in
        // BorrowPrefill/canonical cache views, not this original residency map.
        let view = lease
            .layer_view(layer as usize, ExpertReadView::Resident)
            .ok_or("missing original resident layer view")?;
        for expert in 0..source.experts_per_layer {
            let entry = view
                .entry(expert as usize)
                .ok_or("missing original expert")?;
            let original_device = entry.tier == Placement::Device;
            resident += u64::from(original_device);
            let (gate_up_qtype, down_qtype) = entry.format.pair();
            rows.push(SemanticExpert {
                layer,
                expert,
                original_device,
                gate_up_qtype,
                down_qtype,
                gate_up_bytes: entry.gate_up_len,
                down_bytes: entry.down_len,
            });
        }
    }
    Ok((semantic_partition_digest(source, &rows)?, resident))
}

/// These observations synchronise HIP (including cache DMA). Numerical and
/// transaction correctness only; neither async-overlap nor speed evidence.
#[derive(Clone, Debug, serde::Serialize)]
struct StateDigest {
    target_semantic: Vec<(String, String)>,
    target_physical: Vec<(String, String)>,
    head_active: Option<Vec<(String, String)>>,
}
impl StateDigest {
    fn same_semantic(&self, other: &Self) -> bool {
        self.target_semantic == other.target_semantic && self.head_active == other.head_active
    }
    fn len(&self) -> usize {
        self.target_semantic.len() + self.head_active.as_ref().map_or(0, Vec::len)
    }
}
fn digest(bundle: &mut Qwen4Bundle, gpu: &mut Gpu, route: SessionRoute) -> R<StateDigest> {
    let gdn_layers: Vec<_> = bundle
        .config
        .layer_types
        .iter()
        .enumerate()
        .filter_map(|(layer, kind)| (*kind == LayerType::LinearAttention).then_some(layer))
        .collect();
    ensure(
        gdn_layers.len() == 36,
        "fixture does not have the configured 36 GDN layers",
    )?;
    // The formal state entry rejects an armed verify or unretired cache
    // guard. Only its admission is used: capacity-sized session parts are not
    // hashed as numerical evidence.
    let position = bundle.state.position;
    let _ = SessionState::snapshot_parts(bundle, SessionRoute::Ar, position)?;
    let rows = bundle.state.canonical_digest_rows(gpu).map_err(err)?;
    let (target_semantic, target_physical) =
        target_projection(rows, &gdn_layers, bundle.state_format().gdn.has_residual())?;
    let head_active = if route == SessionRoute::Mtp {
        Some(bundle.dev_mtp_active_digest(gpu).map_err(err)?)
    } else {
        None
    };
    Ok(StateDigest {
        target_semantic,
        target_physical,
        head_active,
    })
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
    states: BTreeMap<usize, StateDigest>,
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
fn check_state(bundle: &mut Qwen4Bundle, gpu: &mut Gpu, want: &StateDigest, label: &str) -> R<()> {
    let got = digest(bundle, gpu, SessionRoute::Ar)?;
    ensure(
        got.same_semantic(want),
        format!(
            "{label}: active target state differs (first semantic row {:?})",
            got.target_semantic
                .iter()
                .zip(&want.target_semantic)
                .position(|(a, b)| a != b)
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
                restored.same_semantic(&ar.states[&0]),
                "abandon changed S0 before any checkpoint restore",
            )?;
            // Reusing the same scratch proves its verified ticket ended.
            verified_complete = false;
            let _ = oracle::dev_verify_logits(bundle, gpu, block, scratch.as_mut())?;
            verified_complete = false;
            oracle::dev_rollback_verify(bundle, gpu, scratch.as_mut())?;
            let restored = digest(bundle, gpu, SessionRoute::Ar)?;
            verified_complete = true;
            ensure(
                restored.same_semantic(&ar.states[&0]),
                "second abandon changed S0",
            )?;
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
                        now.same_semantic(&ar.states[&keep]),
                        "accepted-prefix target state differs from AR",
                    )?;
                    verified_complete = false;
                    bundle
                        .forward_token(gpu, ar.ids[keep], logits, None)
                        .map_err(err)?;
                    let now = digest(bundle, gpu, SessionRoute::Ar)?;
                    verified_complete = true;
                    ensure(
                        now.same_semantic(&ar.states[&(keep + 1)]),
                        "post-commit AR continuation differs",
                    )?;
                    cells.push(json!({"rows":rows,"accepted_drafts":accept,"consumed_rows":keep,"path":if native{"native_retain"}else{"generic_commit_prefix"},"active_state_rows":now.len(),"all_live_semantic_state_exact":true,"subsequent_ar_checked":true}));
                }
            }
        }
        Ok(
            json!({"cells":cells,"abandon_and_ticket_reuse_checked":true,"s0_checkpoint":"host-owned public SessionState layout; no nested target ticket","s0_host_bytes":s0.bytes(),"full_logits_bitwise_claim":true,"state_scope":"canonical active target semantic tensors/marks; raw physical GDN ring receipt retained separately","limits":"Full vocabulary checked against sequential AR only at qualified R1/2/4/8."}),
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
    let mut d = ManuallyDrop::new(Qwen4MtpDrafter::new(
        k,
        prompt.len() + ar.ids.len() + 128,
        None,
    ));
    d.configure_request(SpecRequestConfig::default());
    let mut seed = prefill(&mut d, bundle, gpu, prompt, 0)?;
    ensure(seed == ar.ids[0], "MTP prefill seed differs from AR")?;
    let mut ids = vec![seed];
    let mut frames = Vec::new();
    let mut acceptance = [false; 3];
    let mut repair_candidate = None;
    let cache_before = snapshot(bundle, cache)?;
    let mut native_cache_end = cache_before.clone();
    let mut seen_native_pending = std::collections::BTreeSet::new();
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
                // Evidence is actual PendingFills/SlotBank identity, not JSON
                // object equality or a reservation/eviction counter.
                seen_native_pending.extend(pending_tickets(&after)?);
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
    let head_control_census = bundle.dev_mtp_control_census().map_err(err)?;
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
    let teacher_different_parts: Vec<_> = native_state
        .head_active
        .as_ref()
        .ok_or("missing native head digest")?
        .iter()
        .zip(
            teacher_state
                .head_active
                .as_ref()
                .ok_or("missing teacher head digest")?,
        )
        .filter_map(|((key, before), (other_key, after))| {
            (key != other_key || before != after).then_some(key.clone())
        })
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
        let initial_pending = pending_tickets(&cache_before)?;
        let changed = turnover(&cache_before, &cache_after)?;
        let mut native_changes = Vec::new();
        for change in changed {
            let actual = ticket(&json!({"layer":change["layer"],"slot":change["slot"],
                "global_slot":change["global_slot"],"expert":change["to_expert"],
                "generation":change["generation_after"]}))?;
            if !initial_pending.contains(&actual) && seen_native_pending.contains(&actual) {
                native_changes.push(change);
            }
        }
        let changed = native_changes;
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
    Box::new(ManuallyDrop::into_inner(d)).mtp_free(gpu);
    Ok(
        json!({"ids":ids,"windows":frames,"actual_acceptance_cells":{"zero":acceptance[0],"interior":acceptance[1],"full":acceptance[2]},"completed_victim_turnover":changed,"cache_before_native":cache_before,"cache_after_native":cache_after,"native_target_live_semantic_state_exact":true,"native_target_state":native_target_state,"native_mtp_state":native_state,"native_head_teacher_forced_comparison":"diagnostic_cross_head_schedule","teacher_forced_different_parts":teacher_different_parts,"teacher_forced_state":teacher_state,"teacher_forced_pending_exact":teacher_pending_exact,"native_pending_hidden_sha256":pending_sha256,"native_head_control_census":head_control_census,"terminal":terminal,"instrumented_wall_ms":wall_ms,"performance_claim":false}),
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

    let mut d = ManuallyDrop::new(Qwen4MtpDrafter::new(
        k,
        bundle.config.max_position_embeddings.min(16384),
        None,
    ));
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
        live_target.same_semantic(&ar_target),
        "terminal target differs from independent sequential AR consumed history",
    )?;
    let head_control_census = bundle.dev_mtp_control_census().map_err(err)?;
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
        digest(bundle, gpu, SessionRoute::Mtp)?.same_semantic(&before_plan),
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
    Box::new(ManuallyDrop::into_inner(d)).mtp_free(gpu);
    Ok(
        json!({"source":"native_mtp","verified_drafts":number(&saved.last().ok_or("missing terminal frame")?["measurement"],"actual_drafts")?,"consumed_outputs":1,"retained_prefix_repaired":true,"pending_seed_flushed":true,"native_schedule_exact":true,"replay_frames":replay_frames,"target_state_exact":true,"reference_scope":"original prompt then forced sequential AR over identical emitted and consumed tokens; no cold prefill of generated history","repaired_target_state":live_target,"repaired_target_ar_reference_state":ar_target,"mtp_repaired_state":live,"head_control_census":head_control_census,"native_pending_hidden_sha256":format!("{:x}",Sha256::digest(&pending)),"live_next_turn_in_place":true,"live_plan_state_unchanged":true,"live_next_turn_positions_exact":true,"live_next_turn_exact":false,"live_next_turn_seed":next_seed,"wire_terminal_coverage":false,"limits":["Head active-tensor/session-policy exactness requires cache-off/on comparison under the same saved native replay schedule; actual saved/restored boundary markers are included; additional physical owner/snapshot counters are retained separately in control census.","Next live turn proves in-place state preservation and aligned target/head positions, not an independent seed or output reference.","Canonical serving terminal/session behavior remains a separate qualification."]}),
    )
}
/// One actual forced-native continuation after a normal production prefill.
/// It preserves actual executor measurements, not a manufactured accepted block.
fn prefix_native_frame(
    d: &mut Qwen4MtpDrafter,
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    seed: u32,
    k: usize,
) -> R<(Value, StateDigest, String, u32)> {
    let position = bundle.state.position;
    let measured = oracle::dev_run_forced_window(
        d,
        bundle,
        gpu,
        position,
        seed,
        &[seed],
        u32::MAX,
        DevForcedWindow {
            source: DevWindowSource::Mtp,
            drafts: k,
            timing_only: false,
        },
    )?;
    let measurement = json!({"source":format!("{:?}",measured.source),
        "route":format!("{:?}",measured.route),"requested_drafts":measured.requested_drafts,
        "actual_drafts":measured.actual_drafts,"accepted_prefix":measured.accepted_prefix,
        "committed":measured.window.committed.len(),"outcome_mismatch":measured.outcome_mismatch,
        "position_before":position,"position_after":bundle.state.position,
        "total_us":measured.total_us});
    real_native_window(&measurement)?;
    let next_seed = *measured
        .window
        .committed
        .last()
        .ok_or("empty native prefix continuation")?;
    let active = digest(bundle, gpu, SessionRoute::Mtp)?;
    let pending = oracle::dev_head_pending_hidden(d, gpu)?;
    ensure(
        !pending.is_empty()
            && pending.len() % 4 == 0
            && pending
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().expect("f32 bytes")).is_finite()),
        "prefix native pending extent/finiteness",
    )?;
    let pending_sha256 = format!("{:x}", Sha256::digest(&pending));
    Ok((
        json!({"measurement":measurement,"committed_ids":measured.window.committed}),
        active,
        pending_sha256,
        next_seed,
    ))
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
        2 * chunk + 128 + 2 * (k + 1) <= max_seq,
        "max_seq too short for two-link MTP prefix restore and follow-on",
    )?;
    let a: Vec<_> = (0..2 * chunk + 128)
        .map(|i| prompt[i % prompt.len()])
        .collect();
    let mut b = a[..chunk].to_vec();
    b.extend((0..chunk + 128).map(|i| prompt[(i + 19) % prompt.len()]));
    // Any error retains the drafter's possible live ticket instead of running
    // a scratch Drop assertion over the primary failure. run() retains the
    // dependent bundle/GPU on Err as well; success explicitly frees it below.
    let mut d = ManuallyDrop::new(Qwen4MtpDrafter::new(k, max_seq, None));
    d.configure_request(SpecRequestConfig::default());
    let mut cold = Vec::new();
    for tokens in [&a, &b] {
        let seed = prefill(&mut d, bundle, gpu, tokens, 0)?;
        let boundary = digest(bundle, gpu, SessionRoute::Mtp)?;
        // Publish the normal prompt snapshots before decoding extends live
        // state. A later plan must pick an actual restorable cached prefix.
        bundle.session_commit();
        let (frame, active, pending, next_seed) =
            prefix_native_frame(&mut d, bundle, gpu, seed, k)?;
        cold.push((boundary, seed, frame, active, pending));
        // End the retained repair transaction through the normal API before
        // another prefill. The input seed is the last committed pending row.
        flush(&mut d, bundle, gpu, next_seed)?;
    }
    let mut cells = Vec::new();
    for (i, tokens) in [&a, &b].iter().enumerate() {
        let reused = bundle.session_plan(tokens, SessionRoute::Mtp);
        ensure(
            reused >= chunk && reused < tokens.len(),
            format!("prefix {i}: need an actual cached root and nonempty suffix"),
        )?;
        let seed = prefill(&mut d, bundle, gpu, tokens, reused)?;
        ensure(
            seed == cold[i].1 && digest(bundle, gpu, SessionRoute::Mtp)?.same_semantic(&cold[i].0),
            "restored MTP prefix differs from cold head/target",
        )?;
        bundle.session_commit();
        let (frame, active, pending, next_seed) =
            prefix_native_frame(&mut d, bundle, gpu, seed, k)?;
        let ids = frame["committed_ids"]
            .as_array()
            .ok_or("missing restored native IDs")?
            .iter()
            .map(|id| u32::try_from(id.as_u64().ok_or("invalid restored native ID")?).map_err(err))
            .collect::<R<Vec<_>>>()?;
        same_replay_window(&cold[i].2, &frame["measurement"], &ids)?;
        ensure(
            active.same_semantic(&cold[i].3),
            "post-prefix native active target/head/marks differ from cold reference",
        )?;
        ensure(
            pending == cold[i].4,
            "post-prefix native pending hidden differs from cold reference",
        )?;
        cells.push(json!({"branch":i,"reused":reused,"chunk":chunk,
            "suffix_rows":tokens.len()-reused,"active_target_and_head_state_exact":true,
            "seed_exact":true,"prefix_native_continuation_exact":true,
            "cold_native_frame":cold[i].2,"restored_native_frame":frame,
            "cold_native_state":cold[i].3,"restored_native_state":active,
            "cold_pending_hidden_sha256":cold[i].4,"restored_pending_hidden_sha256":pending,
            "physical_comparison":"raw receipts retained; inactive rings and physical live index are not formal session state and are not equated across restore"}));
        flush(&mut d, bundle, gpu, next_seed)?;
    }
    let outcome = d.mtp_prefill(gpu, bundle, prompt, prompt, 0, false, &|| true)?;
    ensure(
        matches!(outcome, hipfire_runtime::spec::PrefillOutcome::Aborted),
        "prefill abort was not classified as aborted",
    )?;
    Box::new(ManuallyDrop::into_inner(d)).mtp_free(gpu);
    Ok(
        json!({"cells":cells,"prefix_native_continuation_exact":true,
        "prefill_abort_observed":true,"disk_restart_coverage":false,
        "manual_joint_checkpoint_zero_fill_resume_qualification":false,
        "scope":"normal production session-prefix restore plus nonempty suffix prefill then an actual native window; same executor source/width/acceptance/IDs and active target/head/pending bytes"}),
    )
}

pub(super) fn run(args: &Args) -> R<Value> {
    let path = args.model.as_ref().ok_or("missing model")?;
    let raw = path.to_str().ok_or("non-UTF8 model path")?;
    let mut source = ModelSource::from_path(raw)?;
    let source_identity = match &source {
        ModelSource::Hfq(h) => {
            let base = hipfire_arch_qwen4::expert_hotset::HotsetBase::from_hfq(h)?;
            PartitionSource {
                artifact_bytes: base.size(),
                artifact_index_sha256: base.index_sha256().into(),
                layers: 48,
                experts_per_layer: 512,
            }
        }
        _ => return Err("requires reviewed HFQ artifact".into()),
    };
    ensure(
        hipfire_arch_qwen4::expert_hotset::HotsetBudget::from_env()?.select(u64::MAX)?
            == args.hot_budget_mib * (1u64 << 20),
        "requires exact matched hot budget",
    )?;
    ensure(
        hipfire_config::memory::ple_format() == hipfire_config::memory::PleFormatRequest::Q4,
        "requires reviewed q4 PLE",
    )?;
    ensure(
        hipfire_config::developer_var("HIPFIRE_OFFLOAD_EXEC").as_deref() == Ok("pcie"),
        "requires explicit PCIe execution",
    )?;
    let tokenizer = match &source {
        ModelSource::Hfq(h) => Tokenizer::from_hfq_metadata(&h.metadata_json).map_err(err)?,
        _ => return Err("requires HFQ model".into()),
    };
    // Cheap failures must precede the 125GB product load. Tokenize exact raw
    // prompt bytes once and preserve this same sequence for every subcase.
    let mut prepared_prompts = Vec::new();
    let mut prompt_ids_seen = std::collections::BTreeSet::new();
    for file in &args.prompts {
        let bytes = std::fs::read(file).map_err(err)?;
        let text = std::str::from_utf8(&bytes).map_err(err)?;
        let prompt = tokenizer.encode(text);
        ensure(
            !prompt.is_empty()
                && prompt
                    .len()
                    .checked_add(args.tokens)
                    .and_then(|n| n.checked_add(args.k + 1))
                    .is_some_and(|n| n < args.max_seq),
            "preload prompt/context bounds",
        )?;
        ensure(
            prompt_ids_seen.insert(prompt.clone()),
            "duplicate tokenized prompt case",
        )?;
        prepared_prompts.push((file.clone(), bytes, prompt));
    }
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
    ensure(backend.name() == "vmm", "requires actual automatic VMM")?;
    ensure(
        hipfire_runtime::config::get().kv_mode == "fp8",
        "requires explicit native FP8 KV",
    )?;
    let granularity = match backend {
        Qwen4KvBackend::Vmm => gpu.vmm_recommended_granularity().map_err(err)?,
        Qwen4KvBackend::Legacy => 1,
    };
    let shared = Qwen4RankShared {
        batch: Default::default(),
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
    let requested_cache =
        hipfire_arch_qwen4::expert_cache_config::ExpertCacheConfig::from_env()?.is_some();
    let actual_cache = bundle.dev_expert_cache_census(&gpu).map_err(err)?;
    let cache = actual_cache.is_some();
    ensure(
        cache == requested_cache,
        "requested and actual installed cache arms differ",
    )?;
    if let Some(census) = actual_cache.as_ref() {
        ensure(
            census["enabled"] == true,
            "cache census did not establish actual enabled manager",
        )?;
        published(&bundle.dev_expert_cache_snapshot().map_err(err)?)?;
    }
    let logits = gpu
        .zeros(&[bundle.config.vocab_size], DType::F32)
        .map_err(err)?;
    let result = (|| -> R<Value> {
        ensure(
            bundle.state_format().gdn.name() == "q8",
            "oracle must preserve plain-Q8 product policy",
        )?;
        let lease = bundle
            .weights
            .expert_tiers()
            .ok_or("missing product expert tier owners")?;
        let physical_plan_sha256: String = lease
            .logical_digest()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let (semantic_partition_sha256, original_resident_experts) =
            actual_semantic_partition(lease, &source_identity)?;
        let hotset_sha256 = lease
            .hotset_sidecar_sha256()
            .ok_or("missing actual sidecar pin")?
            .to_string();
        // Both arms reserve K7, while actual native windows below remain K3
        // (or the explicitly requested K). R8 target proof needs no second load.
        let sizing = SpecTarget::new_spec_scratch(&mut bundle, &mut gpu, 8)?;
        sizing.free(&mut gpu);
        ensure(
            oracle::dev_effective_row_exact_rows(&bundle, &gpu) >= 8,
            "product load did not admit the requested R8 target proof",
        )?;
        let mut cases = Vec::new();
        for (file, bytes, prompt) in &prepared_prompts {
            ensure(
                !prompt.is_empty() && prompt.len() + args.tokens + args.k + 1 < loaded.max_seq,
                "prompt/context bounds",
            )?;
            let ar = ar_reference(&mut bundle, &mut gpu, &prompt, &logits, args.tokens.max(9))?;
            // Native goes first so target-oracle batches cannot prime its guard witness.
            let native = native_walk(&mut bundle, &mut gpu, &prompt, &logits, &ar, args.k, cache)?;
            let target = target_transactions(&mut bundle, &mut gpu, &prompt, &logits, &ar, args.k)?;
            let prefix = prefix_restore(&mut bundle, &mut gpu, &prompt, args.k, loaded.max_seq)?;
            cases.push(json!({"prompt":file,"prompt_sha256":format!("{:x}",Sha256::digest(&bytes)),"prompt_ids_sha256":format!("{:x}",Sha256::digest(prompt.iter().flat_map(|t|t.to_le_bytes()).collect::<Vec<_>>())),"prompt_tokens":prompt.len(),"ar_ids":ar.ids,"ar_final_state":ar.states[&ar.ids.len()],"ar_full_logits_sha256":ar.logits_bits.iter().map(|(step,row)|json!({"consumed":step,"vocab":row.len(),"bytes":row.len()*4,"sha256":format!("{:x}",Sha256::digest(row.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()))})).collect::<Vec<_>>(),"ar_instrumented_wall_ms":ar.wall_ms,"target_transactions":target,"native":native,"prefix_restore":prefix}));
        }
        Ok(
            json!({"native_mtp_attached":true,"cache_enabled":cache,"hot_budget_mib":args.hot_budget_mib,
            "source_identity":source_identity,"semantic_partition_sha256":semantic_partition_sha256,
            "physical_plan_sha256":physical_plan_sha256,"hotset_sha256":hotset_sha256,
            "original_resident_experts":original_resident_experts,"native_k":args.k,"head_reserve_k":7,"target_rows":[1,2,4,8],"max_seq":loaded.max_seq,"state_format":format!("{:?}",bundle.state_format()),"actual_cache_census_at_load":actual_cache,"kv_backend":backend.name(),"replay_backend":format!("{:?}",gpu.replay.request()),"native_pm4_replay_qualified":false,"scope":"greedy native-MTP transactions and exact expert-cache placement","cases":cases,"performance_claim":false,"limits":["Forced native widths prove execution, not default-policy speed.","INTRUSIVE NUMERICAL ONLY: canonical state and head hashing device-synchronise and drain copy DMA; no async-overlap or timing claim.","Cross AR/verify comparison excludes exactly physical GDN live index and capture ring/inputs/recurrence; all raw physical rows are retained. Same native schedule paired arms must compare them separately.","Head proof covers active head tensors, session marks and draft policy; actual boundary_saved/boundary_restored are included; additional physical snapshot/counter/owner flags are recorded by read-only control census and not paired as semantic bytes.","No SSE terminal, sampled MTP, ngram copy, native PM4 replay or live Q8Ef qualification.","Generic commit on this product load uses captured-row repair; its no-row-capture fallback is not claimed.","Bundle state entrypoints under an intentionally busy guard are not GPU-qualified by this tool.","Manual host joint-checkpoint restore followed by zero-fill native resume is UNQUALIFIED; normal session-prefix restore plus nonempty suffix and native continuation is checked."]}),
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
