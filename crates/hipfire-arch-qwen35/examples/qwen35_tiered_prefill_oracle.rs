// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Fresh-process streaming-prefill state oracle, not a serving/perf/quality gate.
//! KV is physically capped at model load; every prompt chunk is forwarded once.
//! Complete encoded blocks become immutable host history, while only a fixed
//! sink+recent window and an unpublished partial block remain device resident.

#[path = "qwen35_tiered_kv_oracle.rs"]
#[allow(dead_code)]
mod reference;

use hip_bridge::{DeviceBuffer, HipRuntime};
use hipfire_arch_qwen35::qwen35::{self, LayerType, PrefillBatchScratch};
use hipfire_arch_qwen35::{free_qwen35_bundle, Qwen35Bundle};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama;
use hipfire_runtime::tiered_kv::streaming::{plan_step, sink_recent};
use hipfire_runtime::tiered_kv::{ArchiveConfig, ArchiveLayout, FixedKvWindow, PinnedKvArchive};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{DType, Gpu, GpuTensor};
use reference::{float_hash, hash, hash_file, logits, raw_buffer, state_hash};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const BLOCK: usize = 128;
const SCHEMA: &str = "hipfire.tiered_prefill.v1";

fn number(args: &BTreeMap<String, String>, key: &str, default: usize) -> Result<usize> {
    Ok(args
        .get(key)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(default))
}

fn ledger(gpu: &Gpu) -> Result<Value> {
    let s = gpu.emu_budget_snapshot().ok_or("lost managed ledger")?;
    Ok(json!({"limits":{"device_bytes":s.limits.device_bytes,
        "gtt_bytes":s.limits.gtt_bytes,"shared_host_bytes":s.limits.shared_host_bytes,
        "uma":s.limits.uma},
        "committed":{"device_bytes":s.committed.device_bytes,"host_bytes":s.committed.host_bytes,
        "gtt_bytes":s.committed.gtt_bytes,"shared_host_bytes":s.committed.shared_host_bytes},
        "pending":{"device_bytes":s.pending.device_bytes,"host_bytes":s.pending.host_bytes,
        "gtt_bytes":s.pending.gtt_bytes,"shared_host_bytes":s.pending.shared_host_bytes}}))
}

/// Replace only FA storage. The cache metadata already has the exact bounded
/// cap; transient coexistence is two bounded slabs, never a full-history slab.
fn install_window(
    gpu: &mut Gpu,
    b: &mut Qwen35Bundle,
    window: &FixedKvWindow,
    fa: &[usize],
) -> Result<()> {
    for (ordinal, &layer) in fa.iter().enumerate() {
        for (side, tensors) in [(0, &mut b.kv_cache.k_gpu), (1, &mut b.kv_cache.v_gpu)] {
            let backing = &window.sides()[2 * ordinal + side];
            let old = std::mem::replace(
                &mut tensors[layer],
                GpuTensor {
                    // SAFETY: the outer owner removes these borrowed aliases before
                    // bundle teardown and drops the window only after all GPU work.
                    buf: unsafe { backing.alias() },
                    shape: vec![backing.size() / 4],
                    dtype: DType::F32,
                },
            );
            gpu.free_tensor(old)?;
        }
    }
    gpu.drain_pool();
    Ok(())
}

fn run(args: &BTreeMap<String, String>) -> Result<Value> {
    let model = args.get("model").ok_or("--model required")?;
    let prompt_file = args.get("prompt-file").ok_or("--prompt-file required")?;
    let mode = args.get("mode").map(String::as_str).unwrap_or("baseline");
    if !["baseline", "identity", "bounded"].contains(&mode) {
        return Err("mode must be baseline, identity or bounded".into());
    }
    let chunk = number(args, "chunk-tokens", 256)?;
    let decode = number(args, "decode-tokens", 1024)?;
    let records = number(args, "prefix-records", 32)?;
    if !(2..=512).contains(&chunk) || !(256..=32768).contains(&decode) || records > 16384 {
        return Err(
            "require chunk-tokens=2..512, decode-tokens=256..32768, prefix-records<=16384".into(),
        );
    }
    if hipfire_config::developer_var("HIPFIRE_PREFILL_MAX_BATCH")
        .ok()
        .as_deref()
        != Some("512")
    {
        return Err("require HIPFIRE_PREFILL_MAX_BATCH=512 for matched ordinary chunks".into());
    }
    let user = fs::read_to_string(prompt_file)?;
    let prefix: String = (0..records).map(|i| format!("Archive record {i:04}: instruments require calibration; measurements must be checked against a reference.\n")).collect();
    let rendered = format!("<|im_start|>user\n{prefix}\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let hfq = HfqFile::open(Path::new(model))?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)?;
    let ids = tokenizer.encode(&rendered);
    drop(hfq);
    if ids.is_empty() {
        return Err("empty rendered prompt".into());
    }
    let blocks = if mode == "bounded" {
        number(args, "window-blocks", 8)?
    } else {
        ids.len().div_ceil(BLOCK).max(2)
    };
    if blocks < 2 || (mode == "bounded" && blocks > 2048) {
        return Err("bounded window-blocks must be 2..2048".into());
    }
    if mode == "bounded" && ids.len() / BLOCK <= blocks {
        return Err(
            "bounded oracle needs more complete input blocks than the selected window".into(),
        );
    }
    let reserve = chunk
        .max(decode)
        .checked_add(BLOCK - 1)
        .ok_or("reserve overflow")?;
    let physical = blocks
        .checked_mul(BLOCK)
        .and_then(|n| n.checked_add(reserve))
        .ok_or("physical cap overflow")?;
    let max_seq = ids
        .len()
        .checked_add(decode)
        .ok_or("logical cap overflow")?
        .max(physical.checked_add(1).ok_or("logical cap overflow")?);
    let model_sha = hash_file(model)?;
    let binary = std::env::current_exe()?;
    let binary_sha = hash_file(binary.to_str().ok_or("non-UTF8 oracle binary path")?)?;
    let mut gpu = Gpu::init()?;
    let limits = gpu
        .emu_budget_snapshot()
        .ok_or("require an existing managed capacity profile")?;
    if limits.limits.device_bytes > (24u64 << 30) - (512 << 20)
        || gpu.flags.graph_forward != Some(false)
        || gpu.replay.is_enabled()
    {
        return Err(
            "require <=23.5GiB managed profile, HIPFIRE_GRAPH=0 and replay disabled".into(),
        );
    }

    let mut owner = Some(reference::load_with_capacity(
        model,
        max_seq,
        Some(physical),
        &mut gpu,
    )?);
    let mut pbs_owner = None;
    let mut packed_owner = None;
    let mut window_owner = None;
    // Any failure ends this one-shot run. Never retry a chunk whose DeltaNet
    // state might already have advanced; the frontier is published last.
    let result = (|| -> Result<Value> {
        let b = owner.as_mut().unwrap();
        if b.kv_cache.uses_vmm_backend()
            || !b.kv_cache.quant_q8
            || b.kv_adaptive.is_some()
            || b.config.num_experts != 0
            || b.dn_state.s_ef_residual.is_empty()
            || b.kv_cache.physical_cap != physical
            || b.kv_cache.max_seq != max_seq
        {
            return Err(
                "require dense bounded legacy Q8 KV and deterministic Q8+EF recurrent state".into(),
            );
        }
        let after_load = ledger(&gpu)?;
        let fa: Vec<_> = b
            .config
            .layer_types
            .iter()
            .enumerate()
            .filter_map(|(i, kind)| (*kind == LayerType::FullAttention).then_some(i))
            .collect();
        let row = b
            .config
            .n_kv_heads
            .checked_mul(b.config.head_dim / 32)
            .and_then(|n| n.checked_mul(34))
            .ok_or("Q8 row sizing overflow")?;
        if row == 0 || row % 4 != 0 {
            return Err("require whole-u32 Q8 encoded rows".into());
        }
        let history = ids
            .len()
            .checked_add(decode)
            .ok_or("history sizing overflow")?
            .div_ceil(BLOCK)
            .max(blocks)
            .checked_mul(BLOCK)
            .ok_or("history sizing overflow")?;
        let layout = ArchiveLayout::new(ArchiveConfig {
            block_tokens: BLOCK,
            row_bytes_per_side: row,
            full_attention_layers: fa.len(),
            max_history_tokens: history,
            window_blocks: blocks,
            generation_tokens: reserve,
            host_limit_bytes: 12 << 30,
            device_limit_bytes: 4 << 30,
        })?;
        let mut archive = PinnedKvArchive::new(layout)?;
        window_owner = Some(FixedKvWindow::new(Arc::new(HipRuntime::load()?), layout)?);
        let window = window_owner.as_mut().unwrap();
        install_window(&mut gpu, b, window, &fa)?;
        archive.stage(window, &[])?;
        pbs_owner = Some(PrefillBatchScratch::new_opt(
            &mut gpu, &b.config, chunk, false,
        )?);
        packed_owner = Some(gpu.hip.malloc(4)?);
        let pbs = pbs_owner.as_ref().unwrap();
        let packed = packed_owner.as_ref().unwrap();
        for layer in &b.weights.layers {
            qwen35::qwen35_layer_batch_admissible(layer, &b.config, &gpu.arch)?;
        }
        let after_install = ledger(&gpu)?;
        let route = json!({"arch":gpu.arch,"bdf":gpu.hip.device_pci_bus_id(gpu.device_id)?,
            "kv_backend":"legacy","kv_mode":"q8/q8","dn_quant":format!("{:?}",b.dn_state.quant),
            "dn_ef":true,"graph_forward":false,"replay_enabled":false,
            "pbs_max_batch":pbs.max_batch,"pbs_lean":pbs.lean,"pbs_gdn_tape":false,
            "prefill_max_batch_override":512,
            "prefill_path":if mode == "baseline" {"ordinary_with_pbs"} else {"positioned_lab"},
            "singleton_path":if mode == "baseline" {"ordinary_ar"} else {"positioned_ar_lab"}});
        let mut absolute = 0usize;
        let mut visible = 0usize;
        let mut scores = Vec::new();
        let mut prefill_trace = Vec::new();
        while absolute < ids.len() {
            let n = chunk.min(ids.len() - absolute);
            let step = plan_step(BLOCK, blocks, chunk, physical, max_seq, absolute, n)?;
            if step.packed_start != visible
                || archive.committed_tokens() != absolute / BLOCK * BLOCK
            {
                return Err("streaming frontier/archive mismatch before forward".into());
            }
            if n == 1 {
                if mode == "baseline" {
                    qwen35::forward_scratch(
                        &mut gpu,
                        &b.weights,
                        &b.config,
                        ids[absolute],
                        absolute,
                        &mut b.kv_cache,
                        &mut b.dn_state,
                        &b.scratch,
                    )?;
                } else {
                    qwen35::forward::forward_scratch_positioned_lab(
                        &mut gpu,
                        &b.weights,
                        &b.config,
                        ids[absolute],
                        absolute,
                        visible,
                        packed,
                        &mut b.kv_cache,
                        &mut b.dn_state,
                        &b.scratch,
                    )?;
                }
            } else if mode == "baseline" {
                if visible != absolute
                    || !qwen35::prefill_batch_pbs_eligible(
                        &b.weights,
                        &b.config,
                        &b.dn_state,
                        n,
                        &gpu.arch,
                        false,
                    )
                    || qwen35::ordinary_prefill_chunk_limit(
                        &gpu,
                        &b.weights,
                        &b.config,
                        &b.dn_state,
                        &b.kv_cache,
                        Some(pbs),
                    )? < n
                {
                    return Err(
                        "ordinary baseline cannot execute the exact requested PBS chunk".into(),
                    );
                }
                qwen35::forward_prefill_batch_with_pbs(
                    &mut gpu,
                    &b.weights,
                    &b.config,
                    &ids[absolute..step.absolute_end],
                    absolute,
                    &mut b.kv_cache,
                    &mut b.dn_state,
                    &b.scratch,
                    None,
                    None,
                    None,
                    None,
                    Some(pbs),
                    None,
                    None,
                )?;
            } else {
                qwen35::prefill::forward_prefill_chunk_positioned_lab(
                    &mut gpu,
                    &b.weights,
                    &b.config,
                    &ids[absolute..step.absolute_end],
                    absolute,
                    visible,
                    &mut b.kv_cache,
                    &mut b.dn_state,
                    &b.scratch,
                    pbs,
                )?;
            }
            scores = logits(&gpu, b)?;
            let forward_state = state_hash(&gpu, b, visible + n)?;
            // Save the partial block before staging can overwrite its old rows.
            let carry: Vec<_> = window
                .sides()
                .iter()
                .map(|s| raw_buffer(&gpu, s, step.carry_rows * row, step.carry_source_row * row))
                .collect::<Result<_>>()?;
            let carry_hash = hash(&carry.concat());
            let mut appended = Vec::new();
            let sides: Vec<&DeviceBuffer> = window.sides().iter().collect();
            for &source_row in &step.archive_source_rows {
                let key = archive.append_from_device(&gpu.hip, &sides, source_row)?;
                appended.push(json!({"absolute_start":key.start_token,"source_row":source_row}));
            }
            let h2d_bytes = archive.stage(window, &step.next_selected_starts)?;
            let verified = archive.verify_window(window)?;
            let selected_rows = step.next_selected_starts.len() * BLOCK;
            for (side, saved) in window.sides().iter().zip(&carry) {
                if !saved.is_empty() {
                    gpu.hip
                        .memcpy_htod_offset(side, selected_rows * row, saved)?;
                }
            }
            gpu.hip.device_synchronize()?;
            let mut carry_verified = 0;
            for (side, saved) in window.sides().iter().zip(&carry) {
                let actual = raw_buffer(&gpu, side, saved.len(), selected_rows * row)?;
                if actual != *saved {
                    return Err("partial-block carry readback mismatch".into());
                }
                carry_verified += actual.len();
            }
            if archive.committed_tokens() != step.absolute_end / BLOCK * BLOCK
                || window
                    .keys()
                    .ok_or("unpublished window")?
                    .iter()
                    .map(|k| k.start_token)
                    .collect::<Vec<_>>()
                    != step.next_selected_starts
            {
                return Err("archive/map publication mismatch".into());
            }
            let next_visible = selected_rows + step.carry_rows;
            let staged_state = state_hash(&gpu, b, next_visible)?;
            // Full-history controls use the just-computed original GPU prefix
            // as an independent reference, not the archive's own source plan.
            if mode != "bounded" && staged_state != forward_state {
                return Err("full-history source changed during archive/stage/carry".into());
            }
            // Publish only after forward, archive, stage and both readbacks.
            absolute = step.absolute_end;
            visible = next_visible;
            prefill_trace.push(json!({"chunk":prefill_trace.len(),"absolute_start":step.absolute_start,
                "absolute_end":absolute,"tokens":n,"packed_start":step.packed_start,
                "logits_sha256":float_hash(&scores),"argmax":llama::argmax(&scores),
                "forward_state":forward_state,"state":staged_state,
                "selected_block_starts":step.next_selected_starts,"selected_rows":selected_rows,
                "visible_rows":visible,"archive_source_rows":step.archive_source_rows,
                "archived_blocks":appended,"archive_committed_rows":archive.committed_tokens(),
                "archive_host_bytes":archive.host_bytes(),"stage_h2d_bytes":h2d_bytes,
                "staged_history_verified_bytes":verified,"carry_source_row":step.carry_source_row,
                "carry_rows":step.carry_rows,"carry_destination_row":selected_rows,
                "carry_sha256":carry_hash,"carry_verified_bytes":carry_verified,"frontier_published":true}));
            if prefill_trace.len().is_multiple_of(16) {
                eprintln!("{mode}: prefilled {absolute}/{} prompt tokens", ids.len());
            }
        }
        let prefill_state = state_hash(&gpu, b, visible)?;
        let prefill_logits_sha = float_hash(&scores);
        let after_prefill = ledger(&gpu)?;
        let selected = sink_recent(BLOCK, blocks, ids.len() / BLOCK * BLOCK)?;
        let mut generated = Vec::new();
        let mut decode_trace = Vec::new();
        let mut finish = "length";
        let mut first_new_k = None;
        // No decode restaging: selected history is immutable and the fixed
        // generation reserve receives each forwarded token exactly once.
        for index in 0..decode {
            let token = llama::argmax(&scores);
            generated.push(token);
            decode_trace
                .push(json!({"step":index,"token":token,"logits_sha256":float_hash(&scores)}));
            if tokenizer.is_terminator(token) {
                finish = "stop";
                break;
            }
            if index + 1 == decode {
                break;
            } // last emitted token is not forwarded
            let absolute_pos = ids.len() + index;
            if mode == "baseline" {
                qwen35::forward_scratch(
                    &mut gpu,
                    &b.weights,
                    &b.config,
                    token,
                    absolute_pos,
                    &mut b.kv_cache,
                    &mut b.dn_state,
                    &b.scratch,
                )?;
            } else {
                qwen35::forward::forward_scratch_positioned_lab(
                    &mut gpu,
                    &b.weights,
                    &b.config,
                    token,
                    absolute_pos,
                    visible,
                    packed,
                    &mut b.kv_cache,
                    &mut b.dn_state,
                    &b.scratch,
                )?;
            }
            scores = logits(&gpu, b)?;
            if index == 0 {
                first_new_k = Some(hash(&raw_buffer(
                    &gpu,
                    &b.kv_cache.k_gpu[fa[0]].buf,
                    row,
                    visible * row,
                )?));
            }
            visible += 1;
            if (index + 1).is_multiple_of(128) {
                eprintln!("{mode}: {} forwarded decode rows", index + 1);
            }
        }
        let final_state = state_hash(&gpu, b, visible)?;
        let after_decode = ledger(&gpu)?;
        let mut report = json!({"schema":SCHEMA,"pass":true,"mode":mode,
            "scope":"lab streaming positional/state oracle; not serving/performance/quality admission",
            "initial_prefill_bounded":mode == "bounded","performance_claim":false,"quality_claim":false,
            "model":model,"model_sha256":model_sha,"oracle_binary_sha256":binary_sha,
            "prompt_file":prompt_file,"user_prompt_sha256":hash(user.as_bytes()),
            "rendered_prompt_sha256":hash(rendered.as_bytes()),
            "prompt_tokens_sha256":hash(&ids.iter().flat_map(|t|t.to_le_bytes()).collect::<Vec<_>>()),
            "prompt_tokens":ids.len(),"prefix_records":records,"chunk_tokens":chunk,
            "window_blocks":blocks,"block_tokens":BLOCK,"generation_reserve_rows":reserve,
            "physical_cap":physical,"initial_physical_cap":physical,"max_seq":max_seq,
            "initial_full_history_allocation":mode != "bounded","decode_restaging":false});
        let diagnostics = json!({
            "row_bytes_per_side":row,"full_attention_layers":fa.len(),
            "fixed_kv_bytes":layout.device_capacity_bytes(),
            "transient_kv_slabs":"at most two bounded slabs during window installation",
            "route":route,"prefill_logits_sha256":prefill_logits_sha,"prefill_state":prefill_state,
            "selected_block_starts":selected,"archive_committed_rows":archive.committed_tokens(),
            "archive_host_bytes":archive.host_bytes(),"requested_decode_tokens":decode,
            "generated_tokens":generated.len(),"finish_reason":finish,
            "first_new_fa_k_sha256":first_new_k,"final_state":final_state,
            "ledger":{"after_bounded_load":after_load,"after_window_install":after_install,
                "after_prefill":after_prefill,"after_decode":after_decode},
            "text":tokenizer.decode(&generated),"prefill_trace":prefill_trace,"trace":decode_trace});
        if let Value::Object(fields) = diagnostics {
            report.as_object_mut().unwrap().extend(fields);
        }
        Ok(report)
    })();

    // Unconditional single-owner teardown, including a failed partial chunk.
    // An error is never converted into a retry of a stateful forward.
    if let Err(e) = gpu.hip.device_synchronize() {
        // A failed final drain cannot prove that any device allocation is no
        // longer in use. Leak the one-shot owners (and runtime/pool lifetime)
        // instead of freeing memory that queued GPU work may still reference.
        std::mem::forget(pbs_owner.take());
        std::mem::forget(packed_owner.take());
        std::mem::forget(owner.take());
        std::mem::forget(window_owner.take());
        std::mem::forget(gpu);
        let original = result
            .as_ref()
            .err()
            .map(|e| format!("; run error: {e}"))
            .unwrap_or_default();
        return Err(format!(
            "final GPU synchronization failed; live GPU owners intentionally leaked: {e}{original}"
        )
        .into());
    }
    let mut cleanup = Vec::new();
    if let Some(pbs) = pbs_owner.take() {
        if let Err(e) = pbs.free_gpu(&mut gpu) {
            cleanup.push(e.to_string());
        }
    }
    if let Some(packed) = packed_owner.take() {
        if let Err(e) = gpu.hip.free(packed) {
            cleanup.push(e.to_string());
        }
    }
    if let Some(mut b) = owner.take() {
        b.kv_cache.k_gpu.retain(|t| !t.buf.is_borrowed());
        b.kv_cache.v_gpu.retain(|t| !t.buf.is_borrowed());
        if let Err(e) = free_qwen35_bundle(b, &mut gpu) {
            cleanup.push(e);
        }
    }
    drop(window_owner.take());
    gpu.drain_pool();
    let mut report = result.map_err(|e| {
        if cleanup.is_empty() {
            e
        } else {
            format!("{e}; cleanup: {}", cleanup.join("; ")).into()
        }
    })?;
    if !cleanup.is_empty() {
        return Err(format!("cleanup: {}", cleanup.join("; ")).into());
    }
    report["ledger"]["after_cleanup"] = ledger(&gpu)?;
    Ok(report)
}

fn main() -> Result<()> {
    let mut args = BTreeMap::new();
    let mut input = std::env::args().skip(1);
    while let Some(key) = input.next() {
        let key = key
            .strip_prefix("--")
            .ok_or("expected --key value")?
            .to_owned();
        if ![
            "model",
            "mode",
            "prompt-file",
            "decode-tokens",
            "prefix-records",
            "window-blocks",
            "chunk-tokens",
            "out",
        ]
        .contains(&key.as_str())
            || args
                .insert(key, input.next().ok_or("missing argument value")?)
                .is_some()
        {
            return Err("unknown or repeated option".into());
        }
    }
    let out = args.get("out").ok_or("--out required")?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out)?;
    let result = run(&args);
    let report = match &result {
        Ok(v) => v.clone(),
        Err(e) => json!({"schema":SCHEMA,"pass":false,"error":e.to_string()}),
    };
    serde_json::to_writer_pretty(&mut file, &report)?;
    writeln!(file)?;
    println!("pass={} output={out}", result.is_ok());
    result.map(|_| ())
}
