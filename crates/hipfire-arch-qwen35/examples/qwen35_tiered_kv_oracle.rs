// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Real-model, single-request lab oracle; NOT a serving or performance gate.
//! Each arm is a fresh process. Baseline and identity must agree on logits,
//! tokens, KV and recurrent state over long decode. Bounded arms share a full
//! ordinary prefill, then retain sink + recent complete blocks on the GPU.
//! The first newly written FA K must equal the ordinary absolute-position K;
//! wrong-rope is a deliberately broken negative control, never a product arm.
//! This slice does NOT bound the initial prefill, restage during decode, select
//! by query, or implement DFlash/vision/checkpoint resume.

use hip_bridge::{DeviceBuffer, HipRuntime};
use hipfire_arch_qwen35::qwen35::{self, LayerType};
use hipfire_arch_qwen35::{free_qwen35_bundle, load_qwen35_bundle, Qwen35Bundle};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::llama::{self, KvCache};
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
use hipfire_runtime::tiered_kv::{ArchiveConfig, ArchiveLayout, FixedKvWindow, PinnedKvArchive};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const BLOCK: usize = 128;

pub(crate) fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn hash_file(path: &str) -> Result<String> {
    let mut source = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0; 8 << 20];
    loop {
        let n = source.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

fn raw(gpu: &Gpu, t: &GpuTensor, bytes: usize, offset: usize) -> Result<Vec<u8>> {
    raw_buffer(gpu, &t.buf, bytes, offset)
}

pub(crate) fn raw_buffer(
    gpu: &Gpu,
    buffer: &DeviceBuffer,
    bytes: usize,
    offset: usize,
) -> Result<Vec<u8>> {
    if offset.checked_add(bytes).is_none_or(|n| n > buffer.size()) {
        return Err(format!(
            "oracle read offset={offset} bytes={bytes} exceeds backing={}",
            buffer.size()
        )
        .into());
    }
    let mut out = vec![0; bytes];
    gpu.hip.memcpy_dtoh_at(&mut out, buffer, offset)?;
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StagedCopySpan {
    side: usize,
    source_offset: usize,
    packed_offset: usize,
    bytes: usize,
}

/// Plan the independent source/packed read pairs without consulting the
/// archive's mapping or bytes. The old ordinary KV remains the reference.
fn staged_copy_spans(
    selected_starts: &[usize],
    tail_start_row: usize,
    tail_rows: usize,
    row_bytes: usize,
    sides: usize,
) -> Result<Vec<StagedCopySpan>> {
    if row_bytes == 0
        || sides == 0
        || selected_starts.is_empty()
        || tail_rows >= BLOCK
        || !tail_start_row.is_multiple_of(BLOCK)
        || selected_starts.windows(2).any(|p| p[0] >= p[1])
        || selected_starts.iter().any(|&start| {
            !start.is_multiple_of(BLOCK)
                || start
                    .checked_add(BLOCK)
                    .is_none_or(|end| end > tail_start_row)
        })
    {
        return Err("invalid staged-history verification geometry".into());
    }
    let mul = |a: usize, b: usize| {
        a.checked_mul(b)
            .ok_or("staged-history byte extent overflow")
    };
    let block_bytes = mul(BLOCK, row_bytes)?;
    let tail_bytes = mul(tail_rows, row_bytes)?;
    let spans_per_side = selected_starts
        .len()
        .checked_add(usize::from(tail_rows > 0))
        .ok_or("staged-history span count overflow")?;
    let mut spans = Vec::with_capacity(mul(spans_per_side, sides)?);
    for side in 0..sides {
        for (slot, &start) in selected_starts.iter().enumerate() {
            spans.push(StagedCopySpan {
                side,
                source_offset: mul(start, row_bytes)?,
                packed_offset: mul(slot, block_bytes)?,
                bytes: block_bytes,
            });
        }
        if tail_rows > 0 {
            spans.push(StagedCopySpan {
                side,
                source_offset: mul(tail_start_row, row_bytes)?,
                packed_offset: mul(selected_starts.len(), block_bytes)?,
                bytes: tail_bytes,
            });
        }
    }
    Ok(spans)
}

fn verify_staged_history(
    spans: &[StagedCopySpan],
    mut read_source: impl FnMut(usize, usize, usize) -> Result<Vec<u8>>,
    mut read_packed: impl FnMut(usize, usize, usize) -> Result<Vec<u8>>,
) -> Result<usize> {
    let mut verified_bytes = 0usize;
    for span in spans {
        let original = read_source(span.side, span.source_offset, span.bytes)?;
        let actual = read_packed(span.side, span.packed_offset, span.bytes)?;
        if original.len() != span.bytes || actual.len() != span.bytes || original != actual {
            return Err(format!(
                "staged-history mismatch: side={} source_offset={} packed_offset={} bytes={}",
                span.side, span.source_offset, span.packed_offset, span.bytes
            )
            .into());
        }
        verified_bytes = verified_bytes
            .checked_add(span.bytes)
            .ok_or("staged-history verified byte count overflow")?;
    }
    Ok(verified_bytes)
}

pub(crate) fn logits(gpu: &Gpu, b: &Qwen35Bundle) -> Result<Vec<f32>> {
    gpu.hip.device_synchronize()?;
    let v = gpu.download_f32(&b.scratch.logits)?;
    if v.len() != b.config.vocab_size || v.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite or incomplete model logits".into());
    }
    Ok(v)
}

pub(crate) fn float_hash(values: &[f32]) -> String {
    let mut h = Sha256::new();
    for v in values {
        h.update(v.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

pub(crate) fn state_hash(gpu: &Gpu, b: &Qwen35Bundle, visible: usize) -> Result<Value> {
    gpu.hip.device_synchronize()?;
    let mut dn = Sha256::new();
    // Legacy Q8 S deliberately has a byte-count shape but a historical F32
    // dtype tag. Its direct allocation (not a pooled tensor) is authoritative.
    for t in &b.dn_state.s_matrices {
        dn.update(raw(gpu, t, t.buf.size(), 0)?);
    }
    for tensors in [
        &b.dn_state.s_scales,
        &b.dn_state.conv_states,
        &b.dn_state.s_ef_residual,
    ] {
        for t in tensors {
            dn.update(raw(gpu, t, t.numel() * t.dtype.size(), 0)?);
        }
    }
    let row = b.config.n_kv_heads * (b.config.head_dim / 32) * 34;
    let mut kv = Sha256::new();
    for (i, kind) in b.config.layer_types.iter().enumerate() {
        if *kind == LayerType::FullAttention {
            for side in [&b.kv_cache.k_gpu[i], &b.kv_cache.v_gpu[i]] {
                kv.update(raw(gpu, side, visible * row, 0)?);
            }
        }
    }
    let hidden = raw(gpu, &b.scratch.x, b.config.dim * 4, 0)?;
    Ok(json!({"dn_sha256":format!("{:x}",dn.finalize()),
        "kv_sha256":format!("{:x}",kv.finalize()),"hidden_sha256":hash(&hidden),
        "visible_rows":visible}))
}

fn load(model: &str, max_seq: usize, gpu: &mut Gpu) -> Result<Qwen35Bundle> {
    load_with_capacity(model, max_seq, None, gpu)
}

pub(crate) fn load_with_capacity(
    model: &str,
    max_seq: usize,
    physical_cap: Option<usize>,
    gpu: &mut Gpu,
) -> Result<Qwen35Bundle> {
    let cask = CaskConfig::default();
    let mut ctx = LoadCtx {
        path: model,
        max_seq,
        sequence: None,
        deepseek4_compute_placement: Default::default(),
        deepseek4_experts_per_token: None,
        draft_path: None,
        vision_path: None,
        mtp_path: None,
        vision_mode: "off".into(),
        kv_mode_override: Some("q8"),
        kv_k_override: Some("q8"),
        kv_v_override: Some("q8"),
        qwen_default_q8: true,
        kv_backend: KvBackend::Legacy,
        kv_adaptive_override: Some("off"),
        state_quant_override: Some("q8"),
        cask: &cask,
        pp: 1,
        pp_layer_counts: None,
        admitted_mesh: None,
        band_placement: None,
        spec: SpecLoadCfg {
            mtp: Some(false),
            dspark: Some(false),
            ..Default::default()
        },
        gpu,
        gemma4_drafter_path: None,
        gemma4_draft_len: 3,
        xdna: None,
        batch_request: Default::default(),
    };
    let source = ModelSource::from_path(model)?;
    let loaded = match physical_cap {
        Some(cap) => hipfire_arch_qwen35::load_qwen35_bundle_bounded_lab(source, &mut ctx, cap),
        None => load_qwen35_bundle(source, &mut ctx),
    };
    Ok(loaded.map_err(|e| format!("load: {e}"))?)
}

struct Installed {
    archive: PinnedKvArchive,
    window: FixedKvWindow,
    selected_starts: Vec<usize>,
    selected_rows: usize,
    prompt_tail: usize,
    staged_history_verified_bytes: usize,
    staged_history_verified_sides: usize,
    row_bytes_per_side: usize,
    full_attention_layers: usize,
}

fn install(
    gpu: &mut Gpu,
    b: &mut Qwen35Bundle,
    prompt: usize,
    decode: usize,
    blocks: usize,
) -> Result<Installed> {
    let mask: Vec<_> = b
        .config
        .layer_types
        .iter()
        .map(|t| *t == LayerType::FullAttention)
        .collect();
    let fa: Vec<_> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &yes)| yes.then_some(i))
        .collect();
    let complete = prompt / BLOCK;
    if blocks < 2 || complete <= blocks {
        return Err("bounded arm needs >=2 selected blocks and a real history gap".into());
    }
    let selected_starts: Vec<_> = std::iter::once(0)
        .chain((complete - (blocks - 1)..complete).map(|i| i * BLOCK))
        .collect();
    let row = b.config.n_kv_heads * (b.config.head_dim / 32) * 34;
    let tail = prompt % BLOCK;
    let generation = tail
        .checked_add(decode)
        .ok_or("generation extent overflow")?;
    let layout = ArchiveLayout::new(ArchiveConfig {
        block_tokens: BLOCK,
        row_bytes_per_side: row,
        full_attention_layers: fa.len(),
        max_history_tokens: (prompt + decode).div_ceil(BLOCK) * BLOCK,
        window_blocks: blocks,
        generation_tokens: generation,
        host_limit_bytes: 12 << 30,
        device_limit_bytes: 4 << 30,
    })?;
    let mut archive = PinnedKvArchive::new(layout)?;
    let old_sides: Vec<&DeviceBuffer> = fa
        .iter()
        .flat_map(|&i| [&b.kv_cache.k_gpu[i].buf, &b.kv_cache.v_gpu[i].buf])
        .collect();
    for start in (0..complete * BLOCK).step_by(BLOCK) {
        archive.append_from_device(&gpu.hip, &old_sides, start)?;
    }
    let mut window = FixedKvWindow::new(Arc::new(HipRuntime::load()?), layout)?;
    archive.stage(&mut window, &selected_starts)?;
    // The partial prompt block is not in the immutable archive. Append it
    // immediately after ACTUALLY selected rows; never expose a padding hole.
    if tail > 0 {
        for (side, src) in old_sides.iter().enumerate() {
            gpu.hip.memcpy_dtod_at(
                &window.sides()[side],
                blocks * BLOCK * row,
                src,
                complete * BLOCK * row,
                tail * row,
            )?;
        }
    }
    gpu.hip.device_synchronize()?;
    // Verify the actual new backing against the independent original ordinary
    // KV, before dropping that reference. A first-new-K RoPE check alone cannot
    // detect a bad history selection/copy: that K is produced before history
    // attention is read. Compare every encoded byte on every FA K/V side,
    // including the partial prompt block outside the immutable archive.
    let spans = staged_copy_spans(
        &selected_starts,
        complete * BLOCK,
        tail,
        row,
        old_sides.len(),
    )?;
    let verified_bytes = verify_staged_history(
        &spans,
        |side, offset, bytes| raw_buffer(gpu, old_sides[side], bytes, offset),
        |side, offset, bytes| raw_buffer(gpu, &window.sides()[side], bytes, offset),
    )?;
    let verified_sides = old_sides.len();
    let physical = blocks * BLOCK + generation;
    // Metadata-only dummy layers: preserve the exact KvCache contract without
    // allocating another full slab. The window owns FA backing; KvCache views
    // borrow it. Every borrowed descriptor is removed before bundle cleanup.
    let mut view = KvCache::new_gpu_q8_capped_filtered(
        gpu,
        &vec![false; mask.len()],
        b.config.n_kv_heads,
        b.config.head_dim,
        b.kv_cache.max_seq,
        physical,
    )?;
    for (ordinal, &layer) in fa.iter().enumerate() {
        for (side, tensors) in [(0, &mut view.k_gpu), (1, &mut view.v_gpu)] {
            let old = std::mem::replace(
                &mut tensors[layer],
                GpuTensor {
                    // SAFETY: Installed keeps window alive through every forward,
                    // and cleanup removes these aliases before releasing its owner.
                    buf: unsafe { window.sides()[2 * ordinal + side].alias() },
                    shape: vec![layout.device_side_bytes() / 4],
                    dtype: DType::F32,
                },
            );
            gpu.free_tensor(old)?;
        }
    }
    let old = std::mem::replace(&mut b.kv_cache, view);
    old.free_gpu(gpu)?;
    gpu.drain_pool();
    Ok(Installed {
        archive,
        window,
        selected_starts,
        selected_rows: blocks * BLOCK,
        prompt_tail: tail,
        staged_history_verified_bytes: verified_bytes,
        staged_history_verified_sides: verified_sides,
        row_bytes_per_side: row,
        full_attention_layers: fa.len(),
    })
}

fn run(args: &BTreeMap<String, String>) -> Result<Value> {
    let model = args.get("model").ok_or("--model required")?;
    let mode = args.get("mode").map(String::as_str).unwrap_or("baseline");
    if !["baseline", "identity", "bounded", "wrong-rope"].contains(&mode) {
        return Err("unknown mode".into());
    }
    let decode: usize = args
        .get("decode-tokens")
        .map(String::as_str)
        .unwrap_or("1024")
        .parse()?;
    if !(256..=32768).contains(&decode) {
        return Err("decode must be 256..32768".into());
    }
    let records: usize = args
        .get("prefix-records")
        .map(String::as_str)
        .unwrap_or("32")
        .parse()?;
    if records > 8192 {
        return Err("prefix-records exceeds bounded lab limit".into());
    }
    let blocks: usize = args
        .get("window-blocks")
        .map(String::as_str)
        .unwrap_or("2")
        .parse()?;
    let prompt_file = args.get("prompt-file").ok_or("--prompt-file required")?;
    let user = fs::read_to_string(prompt_file)?;
    let prefix:String = (0..records).map(|i|format!("Archive record {i:04}: instruments require calibration; measurements must be checked against a reference.\n")).collect();
    let rendered = format!("<|im_start|>user\n{prefix}\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let hfq = HfqFile::open(Path::new(model))?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)?;
    let ids = tokenizer.encode(&rendered);
    drop(hfq);
    let max_seq = ids
        .len()
        .checked_add(decode)
        .ok_or("logical extent overflow")?;
    let model_sha = hash_file(model)?;
    let binary = std::env::current_exe()?;
    let binary_sha = hash_file(binary.to_str().ok_or("non-UTF8 oracle binary path")?)?;
    let mut gpu = Gpu::init()?;
    let limits = gpu
        .emu_budget_snapshot()
        .ok_or("run under an existing capacity profile, e.g. HIPFIRE_EMU=rx7900xt on gfx1100")?;
    if limits.limits.device_bytes > (24u64 << 30) - (512 << 20)
        || gpu.flags.graph_forward != Some(false)
        || gpu.replay.is_enabled()
    {
        return Err(
            "require <=23.5GiB managed profile, HIPFIRE_GRAPH=0 and HIP replay backend".into(),
        );
    }
    let mut b = load(model, max_seq, &mut gpu)?;
    if b.kv_cache.uses_vmm_backend()
        || !b.kv_cache.quant_q8
        || b.kv_adaptive.is_some()
        || b.config.num_experts != 0
        || b.dn_state.s_ef_residual.is_empty()
    {
        return Err("require dense legacy Q8 KV and deterministic Q8+EF recurrent state".into());
    }
    let route = json!({"arch":gpu.arch,"bdf":gpu.hip.device_pci_bus_id(gpu.device_id)?,
        "kv_backend":"legacy","kv_backend_legacy":true,"explicit_backend":"HIPFIRE_KV_BACKEND=legacy",
        "kv_mode":"q8/q8","dn_quant":format!("{:?}",b.dn_state.quant),"dn_ef":true,
        "managed_limit_bytes":limits.limits.device_bytes,"initial_physical_cap":b.kv_cache.physical_cap});
    qwen35::forward_prefill_batch(
        &mut gpu,
        &b.weights,
        &b.config,
        &ids,
        0,
        &mut b.kv_cache,
        &mut b.dn_state,
        &b.scratch,
        None,
        None,
        None,
        None,
    )?;
    let mut scores = logits(&gpu, &b)?;
    let prefill_state = state_hash(&gpu, &b, ids.len())?;
    let prefill_logits_sha = float_hash(&scores);
    let mut tiered = if mode == "bounded" || mode == "wrong-rope" {
        Some(install(&mut gpu, &mut b, ids.len(), decode, blocks)?)
    } else {
        None
    };
    let packed = gpu.hip.malloc(4)?;
    let first_fa = b
        .config
        .layer_types
        .iter()
        .position(|k| *k == LayerType::FullAttention)
        .ok_or("model has no FA layer")?;
    let row = b.config.n_kv_heads * (b.config.head_dim / 32) * 34;
    let mut generated = vec![];
    let mut trace = vec![];
    let mut visible = tiered
        .as_ref()
        .map(|t| t.selected_rows + t.prompt_tail)
        .unwrap_or(ids.len());
    let mut first_k = None;
    let mut finish = "length";
    for step in 0..decode {
        let next = llama::argmax(&scores);
        trace.push(json!({"step":step,"token":next,"logits_sha256":float_hash(&scores)}));
        generated.push(next);
        if tokenizer.is_terminator(next) {
            finish = "stop";
            break;
        }
        if step + 1 == decode {
            break;
        }
        let absolute = ids.len() + step;
        if mode == "baseline" {
            qwen35::forward_scratch(
                &mut gpu,
                &b.weights,
                &b.config,
                next,
                absolute,
                &mut b.kv_cache,
                &mut b.dn_state,
                &b.scratch,
            )?;
        } else {
            let rope = if mode == "wrong-rope" {
                visible
            } else {
                absolute
            };
            qwen35::forward::forward_scratch_positioned_lab(
                &mut gpu,
                &b.weights,
                &b.config,
                next,
                rope,
                visible,
                &packed,
                &mut b.kv_cache,
                &mut b.dn_state,
                &b.scratch,
            )?;
        }
        gpu.hip.device_synchronize()?;
        if step == 0 {
            first_k = Some(hash(&raw(
                &gpu,
                &b.kv_cache.k_gpu[first_fa],
                row,
                visible * row,
            )?));
        }
        visible += 1;
        scores = logits(&gpu, &b)?;
        if (step + 1).is_multiple_of(128) {
            eprintln!("{mode}: {} forwarded decode rows", step + 1);
        }
    }
    let final_state = state_hash(&gpu, &b, visible)?;
    let tiered_receipt = tiered.as_ref().map(|t|json!({"selected_block_starts":t.selected_starts,
        "selected_rows":t.selected_rows,"partial_prompt_rows":t.prompt_tail,
        "host_archive_bytes":t.archive.host_bytes(),"committed_archived_rows":t.archive.committed_tokens(),
        "fixed_kv_bytes":t.window.sides().iter().map(|s|s.size()).sum::<usize>(),
        "physical_cap":b.kv_cache.physical_cap,"initial_prefill_bounded":false,
        "decode_restaging":false,"staged_history_verified":true,
        "staged_history_verified_bytes":t.staged_history_verified_bytes,
        "staged_history_verified_sides":t.staged_history_verified_sides,
        "staged_history_verified_blocks_per_side":t.selected_starts.len(),
        "staged_history_verified_tail_rows_per_side":t.prompt_tail,
        "row_bytes_per_side":t.row_bytes_per_side,"full_attention_layers":t.full_attention_layers}));
    let snapshot = gpu.emu_budget_snapshot().ok_or("lost managed ledger")?;
    let result = json!({"schema":"hipfire.tiered_kv_model.v2","pass":true,"mode":mode,
        "scope":"lab positional/state oracle; not serving/performance/quality admission",
        "model":model,"model_sha256":model_sha,"prompt_file":prompt_file,
        "oracle_binary_sha256":binary_sha,
        "user_prompt_sha256":hash(user.as_bytes()),"rendered_prompt_sha256":hash(rendered.as_bytes()),
        "prompt_tokens_sha256":hash(&ids.iter().flat_map(|i|i.to_le_bytes()).collect::<Vec<_>>()),
        "prompt_tokens":ids.len(),"max_seq":max_seq,"requested_decode_tokens":decode,
        "generated_tokens":generated.len(),"finish_reason":finish,"route":route,
        "prefill_logits_sha256":prefill_logits_sha,"prefill_state":prefill_state,
        "first_new_fa_k_sha256":first_k,"final_state":final_state,"tiered":tiered_receipt,
        "managed_device_bytes_after_decode":snapshot.committed.device_bytes,
        "managed_host_bytes_after_decode":snapshot.committed.host_bytes,
        "text":tokenizer.decode(&generated),"trace":trace});
    gpu.hip.device_synchronize()?;
    gpu.hip.free(packed)?;
    if tiered.is_some() {
        b.kv_cache.k_gpu.retain(|t| !t.buf.is_borrowed());
        b.kv_cache.v_gpu.retain(|t| !t.buf.is_borrowed());
    }
    free_qwen35_bundle(b, &mut gpu).map_err(|e| format!("free bundle: {e}"))?;
    drop(tiered.take());
    gpu.drain_pool();
    Ok(result)
}

fn main() -> Result<()> {
    let mut args = BTreeMap::new();
    let mut raw = std::env::args().skip(1);
    while let Some(key) = raw.next() {
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
            "out",
        ]
        .contains(&key.as_str())
            || args
                .insert(key, raw.next().ok_or("missing argument value")?)
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
        Err(e) => json!({"pass":false,"error":e.to_string()}),
    };
    serde_json::to_writer_pretty(&mut file, &report)?;
    writeln!(file)?;
    println!("pass={} output={out}", result.is_ok());
    result.map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staged_spans_cover_every_side_block_and_partial_tail() {
        let spans = staged_copy_spans(&[0, 3 * BLOCK], 4 * BLOCK, 7, 2, 2).unwrap();
        assert_eq!(spans.len(), 6);
        assert_eq!(
            spans[1],
            StagedCopySpan {
                side: 0,
                source_offset: 6 * BLOCK,
                packed_offset: 2 * BLOCK,
                bytes: 2 * BLOCK,
            }
        );
        assert_eq!(
            spans[5],
            StagedCopySpan {
                side: 1,
                source_offset: 8 * BLOCK,
                packed_offset: 4 * BLOCK,
                bytes: 14,
            }
        );
        assert_eq!(
            spans.iter().map(|span| span.bytes).sum::<usize>(),
            (2 * BLOCK + 7) * 2 * 2
        );
        assert_eq!(staged_copy_spans(&[0], BLOCK, 0, 2, 2).unwrap().len(), 2);
    }

    #[test]
    fn staged_verification_rejects_block_tail_and_side_corruption() {
        let source: Vec<Vec<u8>> = (0..2)
            .map(|side| {
                (0..4 * BLOCK + 7)
                    .map(|row| ((row * 37 + side * 53) % 251) as u8)
                    .collect()
            })
            .collect();
        // Build the expected packed fixture directly, independently of spans.
        let packed: Vec<Vec<u8>> = source
            .iter()
            .map(|s| {
                s[..BLOCK]
                    .iter()
                    .chain(&s[3 * BLOCK..4 * BLOCK + 7])
                    .copied()
                    .collect()
            })
            .collect();
        let spans = staged_copy_spans(&[0, 3 * BLOCK], 4 * BLOCK, 7, 1, 2).unwrap();
        let verify = |target: &[Vec<u8>]| {
            verify_staged_history(
                &spans,
                |side, offset, bytes| Ok(source[side][offset..offset + bytes].to_vec()),
                |side, offset, bytes| Ok(target[side][offset..offset + bytes].to_vec()),
            )
        };
        assert_eq!(verify(&packed).unwrap(), 2 * (2 * BLOCK + 7));
        for at in [0, BLOCK, 2 * BLOCK + 6] {
            let mut corrupt = packed.clone();
            corrupt[1][at] ^= 1;
            assert!(verify(&corrupt).is_err(), "missed byte {at}");
        }
        let mut swapped = packed.clone();
        swapped.swap(0, 1);
        assert!(verify(&swapped).is_err());
    }

    #[test]
    fn staged_spans_refuse_bad_geometry_and_overflow() {
        for (starts, tail_start, tail, row, sides) in [
            (vec![0, 0], BLOCK, 0, 1, 2),
            (vec![1], BLOCK, 0, 1, 2),
            (vec![BLOCK], BLOCK, 0, 1, 2),
            (vec![0], BLOCK, BLOCK, 1, 2),
            (vec![0], BLOCK, 0, usize::MAX, 2),
            (vec![0, BLOCK], 2 * BLOCK, 0, 1, usize::MAX),
        ] {
            assert!(staged_copy_spans(&starts, tail_start, tail, row, sides).is_err());
        }
    }
}
