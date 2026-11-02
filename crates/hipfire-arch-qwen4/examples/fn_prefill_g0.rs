// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next batch-invariant prefill oracle (cross-request prefill waves, G0).
//!
//! fn_prefill_g0 MODEL.mq4 TEXT.txt OUT_DIR --mode concat|chunk|singleton|bench
//!     [--mix 1,7,17,63,232,366,513,1024] [--depths 0,4096,...]
//!     [--orders given,reverse,rotate] [--chunkings 1+*,7+*,100+132,...]
//!     [--policy shared|production|per-lane|<spec>] [--stage-g0] [--controls]
//!     [--all-logits] [--reps N] [--max-seq N] [--keep-raw]
//!
//! The model loads once. Request `i` takes TEXT tokens starting at offset
//! `i * 997` (wrapping): the first `depth_i` tokens (`--depths`, cycled) are
//! the prefix, the next `mix[i]` tokens the suffix. Every request's prefix is
//! prefilled by one-lane waves in tiles of at most 8192 rows (always with the
//! all-per-lane policy, identically in reference and candidate), so the
//! initial states of reference and candidate are byte-identical by
//! construction.
//!
//! The capture, `Diff`, model loader and `prompt_for` below mirror
//! `fn_lane_oracle.rs` so both oracles compare exactly the same state bytes
//! (GDN recurrent/conv, every QSA byte family below its marks, PLE conv and
//! history, hyper feedback, position).
//!
//! Modes:
//! - `concat`: reference per request = its suffix as a ONE-lane wave
//!   (`WaveHead::Final`); candidate = every request's suffix in ONE wave, once
//!   per `--orders` entry. Per request: logits row, top1 and the full state
//!   bytes must be identical. `--policy` defaults to `shared` (every
//!   non-stateful stage once over all rows). `--all-logits` (total rows <=
//!   1024) additionally compares every row's logits (`WaveHead::All`).
//!   `--stage-g0`: each non-stateful stage alone at once-over-all-rows
//!   (reference = all-per-lane one-lane waves), plus all-per-lane (must pass
//!   by construction) and all-shared; prints a stage -> verdict table.
//!   `--controls` (>= 2 requests): negative controls that MUST differ from the
//!   reference: (a) two lanes' suffix tokens swapped, (b) lane 0 runs on lane
//!   1's prefix state, (c) one suffix token duplicated, (d) lane 0 suffix
//!   shifted by one token.
//! - `chunk`: per request the reference is the whole suffix as one one-lane
//!   wave; each `--chunkings` entry (`a+b+c`, last part may be `*` = the
//!   remainder) splits the suffix into consecutive one-lane waves, and, with
//!   more than one request, also packs the split parts together with the
//!   other requests' corresponding parts. Final logits, top1 and state are
//!   compared; `--all-logits` (suffix <= 1024) compares every row of the
//!   single-request split runs.
//! - `singleton`: the product route (`bundle.reset`, `forward_chunk_final`
//!   tiles on the resident state) against a one-lane wave on a wave state.
//! - `bench`: wall time (device-synchronised) of the concatenated wave vs the
//!   sum of one-lane waves vs the sum of product singleton
//!   `forward_chunk_final` calls; prefixes are re-prefilled outside the timed
//!   region on fresh states; one untimed warmup rep, then `--reps N` (3).
//!
//! `OUT_DIR/summary.json` is written by every mode. Exit 0 only if every
//! compared run is identical and every control is detected (bench exits 0
//! unless an error occurs). Nothing is written outside OUT_DIR; the reference
//! state files under `OUT_DIR/ref` are removed unless `--keep-raw`.
//!
//! Model load follows `qwen4_kld`: admission on the GPU architecture
//! (`admit_hfqm_artifact_on`), slab layer arenas served by
//! `expert_arena::recipe_arena_payload`, and `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N`
//! (a fixed count; `auto` is refused) keeping layers 0..N resident with the
//! rest host-mapped and borrowed from the resident source bytes.

use hipfire_arch_qwen4::admit_hfqm_artifact_on;
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::gpu_forward_lanes::{
    LaneStagePolicy, StageName, StageShare, WaveHead, WaveLane,
};
use hipfire_arch_qwen4::state::Qwen4State;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::{ModelSource as _, SourcePayload};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::prefill_invariant::{
    prefill_pm_invariant_enabled, qwen4_gdn_prefill, Qwen4GdnPrefill,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

type R<T> = Result<T, String>;

fn e<E: std::fmt::Display>(x: E) -> String {
    x.to_string()
}

/// Prefill tile rows (the gfx1151 default chunk).
const TILE_ROWS: usize = 8192;
/// Largest total row count `--all-logits` accepts.
const ALL_LOGITS_MAX_ROWS: usize = 1024;

// ── arguments ────────────────────────────────────────────────────────

struct Args {
    model: PathBuf,
    text: PathBuf,
    out: PathBuf,
    mode: String,
    mix: Vec<usize>,
    depths: Vec<usize>,
    orders: Vec<String>,
    chunkings: Vec<String>,
    policy: String,
    stage_g0: bool,
    controls: bool,
    all_logits: bool,
    reps: usize,
    max_seq: Option<usize>,
    keep_raw: bool,
}

fn parse_list(s: &str, what: &str) -> R<Vec<usize>> {
    s.split(',')
        .filter(|v| !v.trim().is_empty())
        .map(|v| {
            v.trim()
                .parse::<usize>()
                .map_err(|x| format!("bad {what} '{v}': {x}"))
        })
        .collect()
}

fn parse_words(s: &str) -> Vec<String> {
    s.split(',')
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .collect()
}

const USAGE: &str =
    "usage: fn_prefill_g0 MODEL.mq4 TEXT.txt OUT_DIR --mode concat|chunk|singleton|bench \
    [--mix 232,366,...] [--depths 0,4096,...] [--orders given,reverse,rotate] \
    [--chunkings 1+231,100+132,1+*,...] [--policy shared|production|per-lane|<spec>] \
    [--stage-g0] [--controls] [--all-logits] [--reps N] [--max-seq N] [--keep-raw]";

fn parse_args() -> R<Args> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut pos = Vec::new();
    let mut a = Args {
        model: PathBuf::new(),
        text: PathBuf::new(),
        out: PathBuf::new(),
        mode: String::new(),
        mix: vec![1, 7, 17, 63, 232, 366, 513, 1024],
        depths: vec![0],
        orders: vec!["given".into()],
        chunkings: vec!["1+*".into(), "7+*".into(), "100+*".into()],
        policy: "shared".into(),
        stage_g0: false,
        controls: false,
        all_logits: false,
        reps: 3,
        max_seq: None,
        keep_raw: false,
    };
    let mut i = 0;
    let value = |i: &mut usize, flag: &str| -> R<String> {
        *i += 1;
        raw.get(*i)
            .cloned()
            .ok_or_else(|| format!("{flag} needs a value\n{USAGE}"))
    };
    while i < raw.len() {
        match raw[i].as_str() {
            "--mode" => a.mode = value(&mut i, "--mode")?,
            "--mix" => a.mix = parse_list(&value(&mut i, "--mix")?, "--mix")?,
            "--depths" => a.depths = parse_list(&value(&mut i, "--depths")?, "--depths")?,
            "--orders" => a.orders = parse_words(&value(&mut i, "--orders")?),
            "--chunkings" => a.chunkings = parse_words(&value(&mut i, "--chunkings")?),
            "--policy" => a.policy = value(&mut i, "--policy")?,
            "--reps" => a.reps = value(&mut i, "--reps")?.parse().map_err(e)?,
            "--max-seq" => a.max_seq = Some(value(&mut i, "--max-seq")?.parse().map_err(e)?),
            "--stage-g0" => a.stage_g0 = true,
            "--controls" => a.controls = true,
            "--all-logits" => a.all_logits = true,
            "--keep-raw" => a.keep_raw = true,
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}\n{USAGE}")),
            other => pos.push(other.to_string()),
        }
        i += 1;
    }
    if pos.len() != 3 {
        return Err(USAGE.to_string());
    }
    a.model = pos[0].clone().into();
    a.text = pos[1].clone().into();
    a.out = pos[2].clone().into();
    if !matches!(a.mode.as_str(), "concat" | "chunk" | "singleton" | "bench") {
        return Err(format!(
            "--mode must be concat|chunk|singleton|bench\n{USAGE}"
        ));
    }
    if a.mix.is_empty() || a.mix.contains(&0) {
        return Err("--mix must be non-empty and positive".into());
    }
    if a.mix.iter().any(|&n| n > TILE_ROWS) {
        return Err(format!("every --mix entry must be <= {TILE_ROWS} rows"));
    }
    if a.depths.is_empty() {
        return Err("--depths must be non-empty".into());
    }
    if a.orders.is_empty() {
        return Err("--orders must be non-empty".into());
    }
    for o in &a.orders {
        if !matches!(o.as_str(), "given" | "reverse" | "rotate") {
            return Err(format!("unknown order '{o}' (given|reverse|rotate)"));
        }
    }
    if a.reps == 0 {
        return Err("--reps must be >= 1".into());
    }
    if (a.stage_g0 || a.controls) && a.mode != "concat" {
        return Err("--stage-g0 and --controls need --mode concat".into());
    }
    if (a.stage_g0 || a.controls) && a.mix.len() < 2 {
        return Err("--stage-g0 and --controls need at least two requests in --mix".into());
    }
    Ok(a)
}

// ── log ──────────────────────────────────────────────────────────────

#[derive(Default)]
struct Log {
    lines: Vec<String>,
}

impl Log {
    fn say(&mut self, line: String) {
        println!("{line}");
        self.lines.push(line);
    }
}

// ── byte comparison (mirrors fn_lane_oracle) ─────────────────────────

#[derive(Default)]
struct Diff {
    items: usize,
    bytes: u64,
    bad: usize,
    diff_bytes: u64,
    first: Option<String>,
    shown: Vec<String>,
}

impl Diff {
    fn note(&mut self, msg: String) {
        self.bad += 1;
        if self.first.is_none() {
            self.first = Some(msg.clone());
        }
        if self.shown.len() < 8 {
            self.shown.push(msg);
        }
    }

    /// Compare `reference` with `got`; `f32_view` adds the float at the first
    /// differing byte to the report.
    fn check(&mut self, label: &str, reference: &[u8], got: &[u8], f32_view: bool) {
        self.items += 1;
        self.bytes += reference.len() as u64;
        if reference.len() != got.len() {
            self.note(format!(
                "{label}: length ref {} got {}",
                reference.len(),
                got.len()
            ));
            return;
        }
        if reference == got {
            return;
        }
        let i = reference
            .iter()
            .zip(got)
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        let n = reference.iter().zip(got).filter(|(a, b)| a != b).count();
        self.diff_bytes += n as u64;
        let mut msg = format!(
            "{label}: first differing byte {i} (ref 0x{:02x} got 0x{:02x}), {n}/{} bytes differ",
            reference[i],
            got[i],
            reference.len()
        );
        if f32_view {
            let j = i / 4 * 4;
            if j + 4 <= reference.len() {
                let r = f32::from_le_bytes(reference[j..j + 4].try_into().unwrap());
                let g = f32::from_le_bytes(got[j..j + 4].try_into().unwrap());
                msg += &format!("; f32[{}] ref {r:e} got {g:e}", j / 4);
            }
        }
        self.note(msg);
    }

    fn ids(&mut self, label: &str, reference: u32, got: u32) {
        self.items += 1;
        self.bytes += 4;
        if reference != got {
            self.diff_bytes += 4;
            self.note(format!("{label}: token id ref {reference} got {got}"));
        }
    }

    /// The first failing item's label (text before the first `": "`).
    fn first_item(&self) -> Option<String> {
        self.first
            .as_ref()
            .map(|m| m.split(": ").next().unwrap_or(m).to_string())
    }
}

// ── device reads / state capture (mirrors fn_lane_oracle) ────────────

fn read_prefix(gpu: &Gpu, t: &GpuTensor, bytes: usize) -> R<Vec<u8>> {
    if bytes == 0 {
        return Ok(Vec::new());
    }
    if bytes > t.buf.size() {
        return Err(format!(
            "read of {bytes} bytes exceeds the {}-byte device buffer",
            t.buf.size()
        ));
    }
    let mut out = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut out, &t.buf).map_err(e)?;
    Ok(out)
}

/// First `rows` rows of an arena holding `capacity` rows.
fn read_rows(gpu: &Gpu, t: &GpuTensor, rows: usize, capacity: usize) -> R<Vec<u8>> {
    if rows == 0 || capacity == 0 {
        return Ok(Vec::new());
    }
    let total = t.byte_size();
    if total % capacity != 0 {
        return Err(format!(
            "arena of {total} bytes does not hold {capacity} equal rows"
        ));
    }
    read_prefix(gpu, t, rows.min(capacity) * (total / capacity))
}

/// `elems` elements of `t` starting at element `offset`.
fn read_view(gpu: &Gpu, t: &GpuTensor, offset: usize, elems: usize) -> R<Vec<u8>> {
    let v = t.sub_offset(offset, elems);
    read_prefix(gpu, &v, v.byte_size())
}

struct Item {
    name: String,
    bytes: Vec<u8>,
}

fn le_u64s(v: &[usize]) -> Vec<u8> {
    v.iter().flat_map(|x| (*x as u64).to_le_bytes()).collect()
}

/// Every byte family of a request state a step may write: GDN recurrent
/// (live slot) and conv per layer; QSA full K/V, raw/pooled index rows below
/// their marks, partial rows, selected indices and the marks per layer; PLE
/// conv and history; hyper feedback; position. `[n]` is the ordinal within
/// the family's layer list. Identical to `fn_lane_oracle::capture`.
fn capture(gpu: &Gpu, st: &Qwen4State) -> R<Vec<Item>> {
    let mut items = Vec::new();
    let mut push = |name: String, bytes: Vec<u8>| items.push(Item { name, bytes });
    for (i, g) in st.gdn.iter().enumerate() {
        push(
            format!("gdn_recurrent[{i}]"),
            read_prefix(gpu, &g.recurrent, g.recurrent.byte_size())?,
        );
        push(
            format!("gdn_conv[{i}]"),
            read_prefix(gpu, &g.conv, g.conv.byte_size())?,
        );
    }
    for (i, q) in st.qsa.iter().enumerate() {
        push(
            format!("qsa_full_keys[{i}]"),
            read_rows(gpu, &q.full_keys, q.full_len, q.full_capacity)?,
        );
        push(
            format!("qsa_full_values[{i}]"),
            read_rows(gpu, &q.full_values, q.full_len, q.full_capacity)?,
        );
        push(
            format!("qsa_raw_index_keys[{i}]"),
            read_rows(gpu, &q.raw_index_keys, q.raw_len, q.raw_capacity)?,
        );
        push(
            format!("qsa_pooled_keys[{i}]"),
            read_rows(gpu, &q.pooled_keys, q.pooled_len, q.pooled_capacity)?,
        );
        push(
            format!("qsa_partial_keys[{i}]"),
            read_rows(gpu, &q.partial_keys, q.partial_len, q.partial_capacity)?,
        );
        push(
            format!("qsa_partial_values[{i}]"),
            read_rows(gpu, &q.partial_values, q.partial_len, q.partial_capacity)?,
        );
        push(
            format!("qsa_selected_indices[{i}]"),
            read_prefix(
                gpu,
                &q.selected_indices,
                (q.selected_len * 4).min(q.selected_indices.byte_size()),
            )?,
        );
        push(
            format!("qsa_marks[{i}]"),
            le_u64s(&[
                q.full_len,
                q.raw_len,
                q.pooled_len,
                q.partial_len,
                q.selected_len,
                q.position,
            ]),
        );
    }
    // The PLE conv state exists only on profiles that declare PLE.
    if let Some(ple_conv) = &st.ple_conv {
        push(
            "ple_conv".into(),
            read_prefix(gpu, ple_conv, ple_conv.byte_size())?,
        );
    }
    let hyper_feedback = st
        .hyper_feedback
        .as_ref()
        .expect("hyper_feedback is input-band state");
    push(
        "hyper_feedback".into(),
        read_prefix(gpu, hyper_feedback, hyper_feedback.byte_size())?,
    );
    let prev = st.ple_history.previous();
    let mut hist = Vec::with_capacity(12);
    for v in [prev[0], prev[1], st.ple_history.eos_token_id()] {
        hist.extend_from_slice(&v.to_le_bytes());
    }
    push("ple_history".into(), hist);
    push("position".into(), le_u64s(&[st.position]));
    Ok(items)
}

// ── reference store ──────────────────────────────────────────────────

struct RefCkpt {
    path: PathBuf,
    /// (name, offset, len) of every item in the `.bin`.
    index: Vec<(String, u64, u64)>,
}

fn write_ckpt(dir: &Path, items: &[Item]) -> R<RefCkpt> {
    fs::create_dir_all(dir).map_err(e)?;
    let path = dir.join("state.bin");
    let mut w = BufWriter::new(File::create(&path).map_err(e)?);
    let mut index = Vec::with_capacity(items.len());
    let mut offset = 0u64;
    for it in items {
        w.write_all(&it.bytes).map_err(e)?;
        index.push((it.name.clone(), offset, it.bytes.len() as u64));
        offset += it.bytes.len() as u64;
    }
    w.flush().map_err(e)?;
    let idx: Vec<Value> = index
        .iter()
        .map(|(n, o, l)| json!({"name": n, "offset": o, "len": l}))
        .collect();
    fs::write(
        dir.join("state.json"),
        serde_json::to_vec_pretty(&idx).map_err(e)?,
    )
    .map_err(e)?;
    Ok(RefCkpt { path, index })
}

fn compare_ckpt(diff: &mut Diff, ctx: &str, reference: &RefCkpt, got: &[Item]) -> R<()> {
    let mut f = File::open(&reference.path).map_err(e)?;
    if reference.index.len() != got.len() {
        diff.note(format!(
            "{ctx}: state item count ref {} got {}",
            reference.index.len(),
            got.len()
        ));
    }
    for (idx, (name, offset, len)) in reference.index.iter().enumerate() {
        let Some(g) = got.get(idx) else { break };
        if &g.name != name {
            diff.note(format!(
                "{ctx}: state item {idx} is '{}' but the reference has '{name}'",
                g.name
            ));
            continue;
        }
        let mut buf = vec![0u8; *len as usize];
        f.seek(SeekFrom::Start(*offset)).map_err(e)?;
        f.read_exact(&mut buf).map_err(e)?;
        diff.check(&format!("{ctx} state {name}"), &buf, &g.bytes, false);
    }
    Ok(())
}

/// Output rows of one lane (or one product call): F32 logits bytes and the
/// matching top1 ids.
struct LaneOut {
    logits: Vec<u8>,
    top1: Vec<i32>,
}

struct RefReq {
    out: LaneOut,
    ckpt: RefCkpt,
}

struct RefSet {
    reqs: Vec<RefReq>,
}

/// One request's logits/top1/state against its reference.
fn compare_one(ctx: &str, rf: &RefReq, out: &LaneOut, items: &[Item]) -> R<Diff> {
    let mut d = Diff::default();
    d.check(&format!("{ctx} logits"), &rf.out.logits, &out.logits, true);
    compare_top1(&mut d, ctx, &rf.out.top1, &out.top1);
    compare_ckpt(&mut d, ctx, &rf.ckpt, items)?;
    Ok(d)
}

fn compare_top1(d: &mut Diff, ctx: &str, reference: &[i32], got: &[i32]) {
    if reference.len() != got.len() {
        d.items += 1;
        d.note(format!(
            "{ctx} top1: length ref {} got {}",
            reference.len(),
            got.len()
        ));
        return;
    }
    for (j, (a, b)) in reference.iter().zip(got).enumerate() {
        d.ids(&format!("{ctx} top1[{j}]"), *a as u32, *b as u32);
    }
}

// ── world ────────────────────────────────────────────────────────────

struct World {
    gpu: Gpu,
    bundle: Qwen4Bundle,
    vocab: usize,
}

fn load(path: &Path, max_seq: usize) -> R<(HfqModelSource, World, Tokenizer)> {
    let mut hfq = HfqFile::open(path).map_err(e)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(e)?;
    let mut gpu = Gpu::init().map_err(e)?;
    // Admission is GPU-arch aware: an expert-slab artifact admits only on a
    // supported arch. Initialising the GPU allocates no model memory.
    let receipt = admit_hfqm_artifact_on(&hfq, Some(&gpu.arch)).map_err(e)?;
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = DeviceMesh::single().map_err(e)?;
    let expected = WeightOrigin::for_single(&mesh, &gpu);
    let source = HfqModelSource::from_hfq(hfq);
    // `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N` keeps trunk layers 0..N resident
    // and host-maps the rest (a slab artifact larger than one card).
    let weights = {
        let mut weights = receipt.manifest.weights.clone();
        match hipfire_arch_qwen4::expert_residency::expert_vram_layers_from_env()? {
            None => {}
            Some(hipfire_arch_qwen4::expert_residency::ExpertVramLayers::Layers(layers)) => {
                if gpu.is_uma() {
                    return Err("host-mapped experts are only for discrete GPUs".into());
                }
                let moved = hipfire_arch_qwen4::expert_residency::place_routed_experts(
                    &mut weights,
                    layers,
                );
                eprintln!(
                    "fn_prefill_g0: routed experts: layers 0..{layers} in VRAM, {moved} tensors host-mapped"
                );
            }
            Some(hipfire_arch_qwen4::expert_residency::ExpertVramLayers::Auto) => {
                return Err("fn_prefill_g0 takes a fixed expert layer count, not auto".into());
            }
        }
        weights
    };
    let transaction = fulfill_manifest_from_payloads(
        &weights,
        &mesh,
        receipt.config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            if let Some(payload) = hipfire_arch_qwen4::expert_arena::recipe_arena_payload(
                &source,
                &receipt.manifest.identity.recipe,
                entry,
            ) {
                return payload;
            }
            // The store rejects a host-mapped entry as a range: borrow it
            // from the resident source bytes, as the product loader does.
            if entry.residency == hipfire_runtime::weight_manifest::WeightResidency::HostMapped {
                let (info, bytes) = source
                    .tensor_data(&entry.name)
                    .ok_or_else(|| format!("missing resident tensor '{}'", entry.name))?;
                return Ok(SourcePayload::Borrowed { info, bytes });
            }
            source
                .tensor_range(&entry.name)
                .map_err(|x| x.to_string())?
                .map(SourcePayload::Range)
                .ok_or_else(|| format!("missing tensor '{}'", entry.name))
        },
    )
    .map_err(e)?;
    let vocab = receipt.config.vocab_size;
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        &std::env::var("HIPFIRE_STATE_QUANT").unwrap_or_default(),
        &gpu,
        &receipt.config,
    )?;
    let backend = hipfire_arch_qwen4::Qwen4KvBackend::automatic(&gpu);
    let mut bundle = Qwen4Bundle::assemble_with_metadata(
        receipt.config,
        &receipt.manifest.identity.recipe,
        transaction,
        &receipt.placements,
        &mut gpu,
        max_seq,
        receipt.ple,
        state_format,
        backend,
    )
    .map_err(e)?;
    bundle
        .attach_forward(
            &mut gpu,
            max_seq,
            hipfire_arch_qwen4::gpu_forward::Qwen4ForwardScratchMode::Full,
        )
        .map_err(e)?;
    Ok((source, World { gpu, bundle, vocab }, tokenizer))
}

/// Request prompt: TEXT tokens from offset `i * 997`, wrapping.
fn prompt_for(tokens: &[u32], i: usize, ctx: usize) -> Vec<u32> {
    (0..ctx)
        .map(|j| tokens[(i * 997 + j) % tokens.len()])
        .collect()
}

struct Req {
    idx: usize,
    depth: usize,
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    /// The suffix window moved one token later in TEXT (control d).
    shifted: Vec<u32>,
}

fn build_reqs(toks: &[u32], mix: &[usize], depths: &[usize]) -> Vec<Req> {
    mix.iter()
        .enumerate()
        .map(|(i, &len)| {
            let depth = depths[i % depths.len()];
            let all = prompt_for(toks, i, depth + len + 1);
            Req {
                idx: i,
                depth,
                prefix: all[..depth].to_vec(),
                suffix: all[depth..depth + len].to_vec(),
                shifted: all[depth + 1..depth + len + 1].to_vec(),
            }
        })
        .collect()
}

// ── scratch outputs and the rig ──────────────────────────────────────

struct Scratch {
    /// F32, one row per lane of a Final wave.
    final_logits: GpuTensor,
    /// Raw, one i32 per output row.
    top1: GpuTensor,
    /// F32 all-rows logits (only with `--all-logits`).
    all: Option<GpuTensor>,
}

/// One lane of a wave: index into the state pool and the rows to run.
struct LaneSpec<'a> {
    state: usize,
    tokens: &'a [u32],
}

struct Rig {
    w: World,
    pool: Vec<Qwen4State>,
    sc: Scratch,
    tile: usize,
    prefix_policy: LaneStagePolicy,
}

fn read_rows_out(
    gpu: &Gpu,
    logits: &GpuTensor,
    top1: &GpuTensor,
    vocab: usize,
    row0: usize,
    rows: usize,
) -> R<LaneOut> {
    let logits = read_view(gpu, logits, row0 * vocab, rows * vocab)?;
    let raw = read_view(gpu, top1, row0 * 4, rows * 4)?;
    let top1 = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(LaneOut { logits, top1 })
}

impl Rig {
    fn ensure_pool(&mut self, n: usize) -> R<()> {
        while self.pool.len() < n {
            let s = self.w.bundle.new_wave_state(&mut self.w.gpu).map_err(e)?;
            self.pool.push(s);
        }
        Ok(())
    }

    /// One wave over `specs` (device-synchronised before and after); returns
    /// the wall seconds of the forward.
    fn wave(&mut self, specs: &[LaneSpec<'_>], all: bool, policy: &LaneStagePolicy) -> R<f64> {
        if specs.is_empty() {
            return Err("wave without lanes".into());
        }
        let total: usize = specs.iter().map(|s| s.tokens.len()).sum();
        if total == 0 || specs.iter().any(|s| s.tokens.is_empty()) {
            return Err("wave lane without rows".into());
        }
        if total > self.tile {
            return Err(format!(
                "wave of {total} rows exceeds the {}-row tile",
                self.tile
            ));
        }
        let Rig { w, pool, sc, .. } = self;
        let mut slots: Vec<Option<&mut Qwen4State>> = pool.iter_mut().map(Some).collect();
        let mut lanes = Vec::with_capacity(specs.len());
        for s in specs {
            let st = slots
                .get_mut(s.state)
                .and_then(|x| x.take())
                .ok_or_else(|| format!("wave state {} missing or used twice", s.state))?;
            lanes.push(WaveLane {
                state: st,
                tokens: s.tokens,
            });
        }
        let (head, logits) = if all {
            (
                WaveHead::All,
                sc.all
                    .as_ref()
                    .ok_or("--all-logits buffer is not allocated")?,
            )
        } else {
            (WaveHead::Final, &sc.final_logits)
        };
        w.gpu.hip.device_synchronize().map_err(e)?;
        let t = Instant::now();
        w.bundle
            .prefill_wave(&mut w.gpu, &mut lanes, head, logits, Some(&sc.top1), policy)
            .map_err(e)?;
        w.gpu.hip.device_synchronize().map_err(e)?;
        Ok(t.elapsed().as_secs_f64())
    }

    /// Per-lane outputs of the wave just run over `specs`.
    fn outputs(&self, specs: &[LaneSpec<'_>], all: bool) -> R<Vec<LaneOut>> {
        let vocab = self.w.vocab;
        let mut outs = Vec::with_capacity(specs.len());
        let mut row = 0usize;
        for (k, s) in specs.iter().enumerate() {
            if all {
                let t = self
                    .sc
                    .all
                    .as_ref()
                    .ok_or("--all-logits buffer is not allocated")?;
                outs.push(read_rows_out(
                    &self.w.gpu,
                    t,
                    &self.sc.top1,
                    vocab,
                    row,
                    s.tokens.len(),
                )?);
                row += s.tokens.len();
            } else {
                outs.push(read_rows_out(
                    &self.w.gpu,
                    &self.sc.final_logits,
                    &self.sc.top1,
                    vocab,
                    k,
                    1,
                )?);
            }
        }
        Ok(outs)
    }

    /// Reset pool state `idx` and prefill `prefix` by one-lane waves of at
    /// most one tile (all-per-lane policy, identical in every run).
    fn prep(&mut self, idx: usize, prefix: &[u32]) -> R<()> {
        self.ensure_pool(idx + 1)?;
        self.pool[idx].reset(&mut self.w.gpu).map_err(e)?;
        let policy = self.prefix_policy;
        let tile = self.tile;
        for t in prefix.chunks(tile) {
            self.wave(
                &[LaneSpec {
                    state: idx,
                    tokens: t,
                }],
                false,
                &policy,
            )?;
        }
        Ok(())
    }

    fn capture_state(&self, idx: usize) -> R<Vec<Item>> {
        capture(&self.w.gpu, &self.pool[idx])
    }

    /// Product singleton on the resident state: reset, prefix by
    /// `forward_chunk_final` tiles, suffix by `forward_chunk_final`
    /// (`forward_chunk` with `all`). Returns the suffix outputs and the
    /// device-synchronised seconds of the suffix call.
    fn product_run(&mut self, req: &Req, all: bool) -> R<(LaneOut, f64)> {
        let Rig { w, sc, tile, .. } = self;
        // The product entries take exactly the logits they write: one row
        // (`forward_chunk_final`) or every suffix row (`forward_chunk`).
        let final_row = sc.final_logits.sub_offset(0, w.vocab);
        w.bundle.reset(&mut w.gpu).map_err(e)?;
        for t in req.prefix.chunks(*tile) {
            w.bundle
                .forward_chunk_final(&mut w.gpu, t, &final_row, None)
                .map_err(e)?;
        }
        w.gpu.hip.device_synchronize().map_err(e)?;
        let t0 = Instant::now();
        if all {
            let buf = sc
                .all
                .as_ref()
                .ok_or("--all-logits buffer is not allocated")?;
            let rows = buf.sub_offset(0, req.suffix.len() * w.vocab);
            w.bundle
                .forward_chunk(&mut w.gpu, &req.suffix, &rows, Some(&sc.top1))
                .map_err(e)?;
        } else {
            w.bundle
                .forward_chunk_final(&mut w.gpu, &req.suffix, &final_row, Some(&sc.top1))
                .map_err(e)?;
        }
        w.gpu.hip.device_synchronize().map_err(e)?;
        let secs = t0.elapsed().as_secs_f64();
        let out = if all {
            let buf = sc
                .all
                .as_ref()
                .ok_or("--all-logits buffer is not allocated")?;
            read_rows_out(&w.gpu, buf, &sc.top1, w.vocab, 0, req.suffix.len())?
        } else {
            read_rows_out(&w.gpu, &sc.final_logits, &sc.top1, w.vocab, 0, 1)?
        };
        Ok((out, secs))
    }
}

// ── policies ─────────────────────────────────────────────────────────

/// Every non-stateful stage once over all rows.
fn all_shared() -> R<LaneStagePolicy> {
    let mut p = LaneStagePolicy::all_per_lane();
    for s in StageName::ALL {
        if !s.stateful() {
            p = p.with(s, StageShare::OnceAllRows)?;
        }
    }
    Ok(p)
}

fn parse_policy(spec: &str) -> R<LaneStagePolicy> {
    match spec.trim() {
        "shared" | "default" => all_shared(),
        other => LaneStagePolicy::parse(other),
    }
}

// ── references ───────────────────────────────────────────────────────

struct RefCache {
    dir: PathBuf,
    keep_raw: bool,
    sets: HashMap<String, RefSet>,
}

impl RefCache {
    /// Reference per request: the whole suffix as a one-lane wave under
    /// `policy`, with the final row (or every row with `all`), top1 and the
    /// full state bytes.
    fn get(
        &mut self,
        rig: &mut Rig,
        reqs: &[Req],
        policy: &LaneStagePolicy,
        all: bool,
        log: &mut Log,
    ) -> R<&RefSet> {
        let key = format!("{}|all={all}", policy.describe());
        if !self.sets.contains_key(&key) {
            let n = self.sets.len();
            let mut set = RefSet {
                reqs: Vec::with_capacity(reqs.len()),
            };
            let t = Instant::now();
            for r in reqs {
                rig.prep(0, &r.prefix)?;
                let specs = [LaneSpec {
                    state: 0,
                    tokens: &r.suffix,
                }];
                rig.wave(&specs, all, policy)?;
                let out = rig.outputs(&specs, all)?.remove(0);
                let items = rig.capture_state(0)?;
                let dir = self
                    .dir
                    .join(format!("set{n}"))
                    .join(format!("req{}", r.idx));
                let ckpt = write_ckpt(&dir, &items)?;
                if self.keep_raw {
                    fs::write(dir.join("logits.bin"), &out.logits).map_err(e)?;
                    let ids: Vec<u8> = out.top1.iter().flat_map(|v| v.to_le_bytes()).collect();
                    fs::write(dir.join("top1.bin"), ids).map_err(e)?;
                }
                set.reqs.push(RefReq { out, ckpt });
            }
            log.say(format!(
                "reference set{n}: policy [{}] all_logits={all}, {} requests ({:.1}s)",
                policy.describe(),
                reqs.len(),
                t.elapsed().as_secs_f64()
            ));
            self.sets.insert(key.clone(), set);
        }
        Ok(&self.sets[&key])
    }
}

// ── run recording ────────────────────────────────────────────────────

struct Ctx {
    log: Log,
    runs: Vec<Value>,
    all_pass: bool,
}

fn diffs_json(diffs: &[(usize, Diff)]) -> Value {
    Value::Array(
        diffs
            .iter()
            .map(|(r, d)| {
                json!({
                    "req": r,
                    "items": d.items,
                    "bytes": d.bytes,
                    "differing_items": d.bad,
                    "differing_bytes": d.diff_bytes,
                    "first": d.first,
                    "shown": d.shown,
                })
            })
            .collect(),
    )
}

fn diffs_pass(diffs: &[(usize, Diff)]) -> bool {
    diffs.iter().all(|(_, d)| d.bad == 0)
}

fn diffs_first(diffs: &[(usize, Diff)]) -> Option<String> {
    diffs
        .iter()
        .find_map(|(r, d)| d.first.as_ref().map(|m| format!("req{r} {m}")))
}

fn diffs_first_item(diffs: &[(usize, Diff)]) -> Option<String> {
    diffs
        .iter()
        .find_map(|(r, d)| d.first_item().map(|m| format!("req{r} {m}")))
}

struct RunOut {
    diffs: Vec<(usize, Diff)>,
    secs: f64,
    rows: usize,
}

impl Ctx {
    /// Print and store one compared run; returns whether it was identical.
    fn record(&mut self, mode: &str, label: &str, mut meta: Value, res: R<RunOut>) -> bool {
        match res {
            Ok(out) => {
                let pass = diffs_pass(&out.diffs);
                let mut line = format!(
                    "{} {mode} {label}: {} rows in {:.3}s",
                    if pass { "PASS" } else { "FAIL" },
                    out.rows,
                    out.secs
                );
                if let Some(first) = diffs_first(&out.diffs) {
                    line += &format!("\n    first diff: {first}");
                }
                self.log.say(line);
                meta["mode"] = json!(mode);
                meta["label"] = json!(label);
                meta["verdict"] = json!(if pass { "PASS" } else { "FAIL" });
                meta["rows"] = json!(out.rows);
                meta["seconds"] = json!(out.secs);
                meta["first_diff"] = json!(diffs_first(&out.diffs));
                meta["requests"] = diffs_json(&out.diffs);
                self.runs.push(meta);
                self.all_pass &= pass;
                pass
            }
            Err(m) => {
                self.log.say(format!("FAIL {mode} {label}: error: {m}"));
                meta["mode"] = json!(mode);
                meta["label"] = json!(label);
                meta["verdict"] = json!("FAIL");
                meta["error"] = json!(m);
                self.runs.push(meta);
                self.all_pass = false;
                false
            }
        }
    }

    /// A negative control: `Some(true)` detected, `Some(false)` undetected,
    /// `None` not applicable / errored.
    fn record_control(&mut self, name: &str, meta: Value, res: R<Option<RunOut>>) {
        let mut meta = meta;
        meta["mode"] = json!("control");
        meta["label"] = json!(name);
        match res {
            Ok(None) => {
                self.log.say(format!(
                    "SKIP control {name}: not applicable to this mix/depths"
                ));
                meta["verdict"] = json!("SKIP");
            }
            Ok(Some(out)) => {
                let detected = !diffs_pass(&out.diffs);
                let mut line = format!(
                    "{} control {name}",
                    if detected { "DETECTED" } else { "UNDETECTED" }
                );
                if let Some(first) = diffs_first(&out.diffs) {
                    line += &format!(": {first}");
                }
                self.log.say(line);
                meta["verdict"] = json!(if detected { "DETECTED" } else { "UNDETECTED" });
                meta["first_diff"] = json!(diffs_first(&out.diffs));
                meta["requests"] = diffs_json(&out.diffs);
                self.all_pass &= detected;
            }
            Err(m) => {
                self.log.say(format!("FAIL control {name}: error: {m}"));
                meta["verdict"] = json!("ERROR");
                meta["error"] = json!(m);
                self.all_pass = false;
            }
        }
        self.runs.push(meta);
    }
}

// ── concat ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Control {
    None,
    /// Two lanes carry each other's suffix tokens.
    SwapTokens,
    /// Lane 0 runs on lane 1's prefix state (and vice versa).
    SwapStates,
    /// One suffix token duplicated.
    DupToken,
    /// Lane 0's suffix is the TEXT window one token later.
    ShiftOne,
}

impl Control {
    fn name(self) -> &'static str {
        match self {
            Control::None => "none",
            Control::SwapTokens => "swap_suffix_tokens",
            Control::SwapStates => "swap_prefix_states",
            Control::DupToken => "duplicated_token",
            Control::ShiftOne => "shifted_by_one_token",
        }
    }
}

fn lane_order(kind: &str, k: usize) -> Vec<usize> {
    match kind {
        "reverse" => (0..k).rev().collect(),
        "rotate" => (0..k).map(|i| (i + 1) % k).collect(),
        _ => (0..k).collect(),
    }
}

/// All requests' suffixes in one wave (lanes in `order`) against `refs`.
/// `None` when the control cannot differ (identical inputs).
#[allow(clippy::too_many_arguments)]
fn concat_run(
    rig: &mut Rig,
    reqs: &[Req],
    order: &[usize],
    policy: &LaneStagePolicy,
    refs: &RefSet,
    all: bool,
    control: Control,
) -> R<Option<RunOut>> {
    let k = reqs.len();
    rig.ensure_pool(k)?;
    let mut tokens: Vec<Vec<u32>> = order.iter().map(|&r| reqs[r].suffix.clone()).collect();
    let mut states: Vec<usize> = order.to_vec();
    match control {
        Control::None => {}
        Control::SwapTokens => {
            if k < 2 || tokens[0] == tokens[1] {
                return Ok(None);
            }
            tokens.swap(0, 1);
        }
        Control::SwapStates => {
            if k < 2 || reqs[order[0]].prefix == reqs[order[1]].prefix {
                return Ok(None);
            }
            states.swap(0, 1);
        }
        Control::DupToken => {
            let m = tokens[0].len() / 2;
            let t = tokens[0][m];
            tokens[0].insert(m, t);
        }
        Control::ShiftOne => {
            if reqs[order[0]].shifted == reqs[order[0]].suffix {
                return Ok(None);
            }
            tokens[0] = reqs[order[0]].shifted.clone();
        }
    }
    let rows: usize = tokens.iter().map(Vec::len).sum();
    for r in 0..k {
        rig.prep(r, &reqs[r].prefix)?;
    }
    let specs: Vec<LaneSpec<'_>> = states
        .iter()
        .zip(&tokens)
        .map(|(&state, t)| LaneSpec { state, tokens: t })
        .collect();
    let secs = rig.wave(&specs, all, policy)?;
    let outs = rig.outputs(&specs, all)?;
    let mut diffs = Vec::with_capacity(k);
    for (lane, out) in outs.iter().enumerate() {
        let r = order[lane];
        let items = rig.capture_state(states[lane])?;
        let d = compare_one(&format!("req{r}"), &refs.reqs[r], out, &items)?;
        diffs.push((r, d));
    }
    Ok(Some(RunOut { diffs, secs, rows }))
}

fn mix_depths_meta(args: &Args, reqs: &[Req]) -> Value {
    json!({
        "mix": reqs.iter().map(|r| r.suffix.len()).collect::<Vec<_>>(),
        "depths": reqs.iter().map(|r| r.depth).collect::<Vec<_>>(),
        "args_depths": args.depths,
    })
}

fn run_concat(
    args: &Args,
    ctx: &mut Ctx,
    rig: &mut Rig,
    cache: &mut RefCache,
    reqs: &[Req],
) -> R<()> {
    let total: usize = reqs.iter().map(|r| r.suffix.len()).sum();
    let slack = if args.controls { 1 } else { 0 };
    if total + slack > rig.tile {
        return Err(format!(
            "concat wave of {total} rows exceeds the {}-row tile",
            rig.tile
        ));
    }
    if args.all_logits && total > ALL_LOGITS_MAX_ROWS {
        return Err(format!(
            "--all-logits needs a total of <= {ALL_LOGITS_MAX_ROWS} rows (have {total})"
        ));
    }
    let policy = parse_policy(&args.policy)?;
    let meta0 = mix_depths_meta(args, reqs);
    let k = reqs.len();

    for order_kind in &args.orders {
        let order = lane_order(order_kind, k);
        for all in [false, true] {
            if all && !args.all_logits {
                continue;
            }
            let label = format!(
                "order={order_kind} lanes={order:?} policy=[{}] all_logits={all}",
                policy.describe()
            );
            let mut meta = meta0.clone();
            meta["order"] = json!(order_kind);
            meta["lane_order"] = json!(order);
            meta["policy"] = json!(policy.describe());
            meta["all_logits"] = json!(all);
            let res = (|| {
                let refs = cache.get(rig, reqs, &policy, all, &mut ctx.log)?;
                concat_run(rig, reqs, &order, &policy, refs, all, Control::None)?
                    .ok_or_else(|| "internal: concat reported not applicable".to_string())
            })();
            ctx.record("concat", &label, meta, res);
        }
    }

    if args.stage_g0 {
        let per_lane = LaneStagePolicy::all_per_lane();
        let mut variants: Vec<(String, LaneStagePolicy)> = vec![("all-per-lane".into(), per_lane)];
        for s in StageName::ALL {
            if !s.stateful() {
                variants.push((
                    s.key().to_string(),
                    per_lane.with(s, StageShare::OnceAllRows)?,
                ));
            }
        }
        variants.push(("all-shared".into(), all_shared()?));
        let mut table: Vec<(String, String)> = Vec::new();
        for (name, pol) in &variants {
            let mut stage_pass = true;
            let mut stage_first: Option<String> = None;
            for order_kind in &args.orders {
                let order = lane_order(order_kind, k);
                let label = format!(
                    "stage={name} order={order_kind} policy=[{}]",
                    pol.describe()
                );
                let mut meta = meta0.clone();
                meta["stage"] = json!(name);
                meta["order"] = json!(order_kind);
                meta["policy"] = json!(pol.describe());
                meta["ref_policy"] = json!(per_lane.describe());
                let res = (|| {
                    let refs = cache.get(rig, reqs, &per_lane, false, &mut ctx.log)?;
                    concat_run(rig, reqs, &order, pol, refs, false, Control::None)?
                        .ok_or_else(|| "internal: concat reported not applicable".to_string())
                })();
                let first = res.as_ref().ok().and_then(|o| diffs_first_item(&o.diffs));
                let err = res.as_ref().err().cloned();
                let pass = ctx.record("stage-g0", &label, meta, res);
                if !pass && stage_first.is_none() {
                    stage_first = first.or(err.map(|m| format!("error: {m}")));
                }
                stage_pass &= pass;
            }
            table.push((
                name.clone(),
                if stage_pass {
                    "identical".to_string()
                } else {
                    format!("DIFFERS ({})", stage_first.unwrap_or_default())
                },
            ));
        }
        ctx.log
            .say("stage-g0 verdicts (reference: all-per-lane one-lane waves):".into());
        for (name, verdict) in &table {
            ctx.log.say(format!("  {name:<14} {verdict}"));
        }
        ctx.runs.push(json!({
            "mode": "stage-g0-table",
            "reference_policy": per_lane.describe(),
            "table": table.iter().map(|(n, v)| json!({"stage": n, "verdict": v})).collect::<Vec<_>>(),
        }));
        if table.first().is_some_and(|(_, v)| v != "identical") {
            ctx.log.say(
                "stage-g0: BASELINE all-per-lane differs, which must not happen by construction"
                    .into(),
            );
            ctx.all_pass = false;
        }
    }

    if args.controls {
        let order = lane_order("given", k);
        for control in [
            Control::SwapTokens,
            Control::SwapStates,
            Control::DupToken,
            Control::ShiftOne,
        ] {
            let mut meta = meta0.clone();
            meta["policy"] = json!(policy.describe());
            meta["order"] = json!("given");
            let res = (|| {
                let refs = cache.get(rig, reqs, &policy, false, &mut ctx.log)?;
                concat_run(rig, reqs, &order, &policy, refs, false, control)
            })();
            ctx.record_control(control.name(), meta, res);
        }
    }
    Ok(())
}

// ── chunk ────────────────────────────────────────────────────────────

/// `a+b+c` with an optional trailing `*` (the remainder).
fn parse_chunking(spec: &str) -> R<(Vec<usize>, bool)> {
    let mut parts = Vec::new();
    let mut star = false;
    let items: Vec<&str> = spec.split('+').map(str::trim).collect();
    for (i, it) in items.iter().enumerate() {
        if *it == "*" {
            if i + 1 != items.len() {
                return Err(format!("chunking '{spec}': '*' must be last"));
            }
            star = true;
        } else {
            let n: usize = it
                .parse()
                .map_err(|x| format!("chunking '{spec}': bad part '{it}': {x}"))?;
            if n == 0 {
                return Err(format!("chunking '{spec}': parts must be >= 1"));
            }
            parts.push(n);
        }
    }
    if parts.is_empty() && !star {
        return Err(format!("chunking '{spec}' is empty"));
    }
    Ok((parts, star))
}

/// The parts for a suffix of `len` rows; `None` when the spec does not fit.
fn resolve_parts(parts: &[usize], star: bool, len: usize) -> Option<Vec<usize>> {
    let fixed: usize = parts.iter().sum();
    if fixed > len || (!star && fixed != len) {
        return None;
    }
    let mut out = parts.to_vec();
    if star && len > fixed {
        out.push(len - fixed);
    }
    Some(out)
}

/// One request's suffix split into consecutive one-lane waves on pool state 0.
/// Without `all` only the last part's logits row is compared; with `all`
/// every part's rows are compared against the unsplit all-rows reference.
fn split_single(
    rig: &mut Rig,
    req: &Req,
    parts: &[usize],
    policy: &LaneStagePolicy,
    rf: &RefReq,
    all: bool,
) -> R<(Diff, f64)> {
    let vocab = rig.w.vocab;
    let ctx = format!("req{}", req.idx);
    rig.prep(0, &req.prefix)?;
    let mut diff = Diff::default();
    let mut secs = 0.0;
    let mut off = 0usize;
    let mut last: Option<LaneOut> = None;
    for (j, &p) in parts.iter().enumerate() {
        let specs = [LaneSpec {
            state: 0,
            tokens: &req.suffix[off..off + p],
        }];
        secs += rig.wave(&specs, all, policy)?;
        let out = rig.outputs(&specs, all)?.remove(0);
        if all {
            let row_bytes = vocab * 4;
            diff.check(
                &format!("{ctx} part{j} rows {off}..{} logits", off + p),
                &rf.out.logits[off * row_bytes..(off + p) * row_bytes],
                &out.logits,
                true,
            );
            compare_top1(
                &mut diff,
                &format!("{ctx} part{j}"),
                &rf.out.top1[off..off + p],
                &out.top1,
            );
        } else {
            last = Some(out);
        }
        off += p;
    }
    if let Some(out) = last {
        diff.check(&format!("{ctx} logits"), &rf.out.logits, &out.logits, true);
        compare_top1(&mut diff, &ctx, &rf.out.top1, &out.top1);
    }
    let items = rig.capture_state(0)?;
    compare_ckpt(&mut diff, &ctx, &rf.ckpt, &items)?;
    Ok((diff, secs))
}

/// The split parts of every applicable request packed wave by wave (wave `j`
/// = every request's part `j`), each request on its own state.
fn split_packed(
    rig: &mut Rig,
    reqs: &[Req],
    parts: &[Option<Vec<usize>>],
    policy: &LaneStagePolicy,
    refs: &RefSet,
) -> R<RunOut> {
    let active: Vec<usize> = (0..reqs.len()).filter(|&r| parts[r].is_some()).collect();
    rig.ensure_pool(reqs.len())?;
    for &r in &active {
        rig.prep(r, &reqs[r].prefix)?;
    }
    let max_parts = active
        .iter()
        .map(|&r| parts[r].as_ref().unwrap().len())
        .max()
        .unwrap_or(0);
    let mut offs = vec![0usize; reqs.len()];
    let mut finals: Vec<Option<LaneOut>> = (0..reqs.len()).map(|_| None).collect();
    let (mut secs, mut rows) = (0.0, 0usize);
    for j in 0..max_parts {
        let lanes: Vec<(usize, usize)> = active
            .iter()
            .filter_map(|&r| parts[r].as_ref().unwrap().get(j).map(|&p| (r, p)))
            .collect();
        let specs: Vec<LaneSpec<'_>> = lanes
            .iter()
            .map(|&(r, p)| LaneSpec {
                state: r,
                tokens: &reqs[r].suffix[offs[r]..offs[r] + p],
            })
            .collect();
        secs += rig.wave(&specs, false, policy)?;
        rows += lanes.iter().map(|&(_, p)| p).sum::<usize>();
        let mut outs = rig.outputs(&specs, false)?;
        for (lane, &(r, p)) in lanes.iter().enumerate() {
            if j + 1 == parts[r].as_ref().unwrap().len() {
                finals[r] = Some(std::mem::replace(
                    &mut outs[lane],
                    LaneOut {
                        logits: Vec::new(),
                        top1: Vec::new(),
                    },
                ));
            }
            offs[r] += p;
        }
    }
    let mut diffs = Vec::new();
    for &r in &active {
        let out = finals[r]
            .take()
            .ok_or_else(|| format!("req{r} produced no final row"))?;
        let items = rig.capture_state(r)?;
        diffs.push((
            r,
            compare_one(&format!("req{r}"), &refs.reqs[r], &out, &items)?,
        ));
    }
    Ok(RunOut { diffs, secs, rows })
}

fn run_chunk(
    args: &Args,
    ctx: &mut Ctx,
    rig: &mut Rig,
    cache: &mut RefCache,
    reqs: &[Req],
) -> R<()> {
    let policy = parse_policy(&args.policy)?;
    let meta0 = mix_depths_meta(args, reqs);
    for spec in &args.chunkings {
        let (fixed, star) = parse_chunking(spec)?;
        let parts: Vec<Option<Vec<usize>>> = reqs
            .iter()
            .map(|r| resolve_parts(&fixed, star, r.suffix.len()))
            .collect();
        for (r, p) in parts.iter().enumerate() {
            if p.is_none() {
                ctx.log.say(format!(
                    "SKIP chunk {spec} req{r}: suffix of {} rows does not fit the chunking",
                    reqs[r].suffix.len()
                ));
            }
        }
        for all in [false, true] {
            if all && !args.all_logits {
                continue;
            }
            // Single-request split runs.
            for (r, req) in reqs.iter().enumerate() {
                let Some(p) = parts[r].clone() else { continue };
                if all && req.suffix.len() > ALL_LOGITS_MAX_ROWS {
                    ctx.log.say(format!(
                        "SKIP chunk {spec} req{r} all_logits: suffix of {} rows exceeds {ALL_LOGITS_MAX_ROWS}",
                        req.suffix.len()
                    ));
                    continue;
                }
                let label = format!(
                    "chunking={spec} req={r} parts={p:?} single policy=[{}] all_logits={all}",
                    policy.describe()
                );
                let mut meta = meta0.clone();
                meta["chunking"] = json!(spec);
                meta["parts"] = json!(p);
                meta["request"] = json!(r);
                meta["packed"] = json!(false);
                meta["policy"] = json!(policy.describe());
                meta["all_logits"] = json!(all);
                let res = (|| -> R<RunOut> {
                    let refs = cache.get(rig, reqs, &policy, all, &mut ctx.log)?;
                    let (d, secs) = split_single(rig, req, &p, &policy, &refs.reqs[r], all)?;
                    Ok(RunOut {
                        diffs: vec![(r, d)],
                        secs,
                        rows: req.suffix.len(),
                    })
                })();
                ctx.record("chunk", &label, meta, res);
            }
            // Packed co-tenant run (Final head only).
            let active = parts.iter().filter(|p| p.is_some()).count();
            if !all && reqs.len() > 1 && active > 1 {
                let label = format!("chunking={spec} packed policy=[{}]", policy.describe());
                let mut meta = meta0.clone();
                meta["chunking"] = json!(spec);
                meta["packed"] = json!(true);
                meta["policy"] = json!(policy.describe());
                let res = (|| {
                    let refs = cache.get(rig, reqs, &policy, false, &mut ctx.log)?;
                    split_packed(rig, reqs, &parts, &policy, refs)
                })();
                ctx.record("chunk", &label, meta, res);
            }
        }
    }
    Ok(())
}

// ── singleton ────────────────────────────────────────────────────────

fn run_singleton(args: &Args, ctx: &mut Ctx, rig: &mut Rig, reqs: &[Req]) -> R<()> {
    let policy = parse_policy(&args.policy)?;
    let meta0 = mix_depths_meta(args, reqs);
    for all in [false, true] {
        if all && !args.all_logits {
            continue;
        }
        for req in reqs {
            if all && req.suffix.len() > ALL_LOGITS_MAX_ROWS {
                ctx.log.say(format!(
                    "SKIP singleton req{} all_logits: suffix of {} rows exceeds {ALL_LOGITS_MAX_ROWS}",
                    req.idx,
                    req.suffix.len()
                ));
                continue;
            }
            let label = format!(
                "req={} depth={} rows={} policy=[{}] all_logits={all}",
                req.idx,
                req.depth,
                req.suffix.len(),
                policy.describe()
            );
            let mut meta = meta0.clone();
            meta["request"] = json!(req.idx);
            meta["policy"] = json!(policy.describe());
            meta["all_logits"] = json!(all);
            let res = (|| -> R<RunOut> {
                // Reference: the product route on the resident state.
                let (ref_out, _) = rig.product_run(req, all)?;
                let ref_items = rig.capture_state_resident()?;
                let dir = rig_ref_dir(args).join(format!("singleton_req{}", req.idx));
                let ckpt = write_ckpt(&dir, &ref_items)?;
                if args.keep_raw {
                    // Product-route bytes for cross-build identity (diff two builds' OUT_DIR/ref).
                    fs::write(dir.join("logits.bin"), &ref_out.logits).map_err(e)?;
                    let ids: Vec<u8> = ref_out.top1.iter().flat_map(|v| v.to_le_bytes()).collect();
                    fs::write(dir.join("top1.bin"), ids).map_err(e)?;
                }
                let rf = RefReq { out: ref_out, ckpt };
                // Candidate: the same request as a one-lane wave.
                rig.prep(0, &req.prefix)?;
                let specs = [LaneSpec {
                    state: 0,
                    tokens: &req.suffix,
                }];
                let secs = rig.wave(&specs, all, &policy)?;
                let out = rig.outputs(&specs, all)?.remove(0);
                let items = rig.capture_state(0)?;
                let d = compare_one(&format!("req{}", req.idx), &rf, &out, &items)?;
                Ok(RunOut {
                    diffs: vec![(req.idx, d)],
                    secs,
                    rows: req.suffix.len(),
                })
            })();
            ctx.record("singleton", &label, meta, res);
        }
    }
    Ok(())
}

fn rig_ref_dir(args: &Args) -> PathBuf {
    args.out.join("ref")
}

impl Rig {
    /// State bytes of the resident (product-route) state.
    fn capture_state_resident(&self) -> R<Vec<Item>> {
        capture(&self.w.gpu, &self.w.bundle.state)
    }
}

// ── bench ────────────────────────────────────────────────────────────

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn run_bench(args: &Args, ctx: &mut Ctx, rig: &mut Rig, reqs: &[Req]) -> R<()> {
    let policy = parse_policy(&args.policy)?;
    let k = reqs.len();
    let total: usize = reqs.iter().map(|r| r.suffix.len()).sum();
    if total > rig.tile {
        return Err(format!(
            "bench concat wave of {total} rows exceeds the {}-row tile",
            rig.tile
        ));
    }
    rig.ensure_pool(k)?;
    let (mut concat, mut one_lane, mut singleton) = (Vec::new(), Vec::new(), Vec::new());
    for rep in 0..=args.reps {
        // Concatenated wave: fresh prefixes, one launch group.
        for r in 0..k {
            rig.prep(r, &reqs[r].prefix)?;
        }
        let specs: Vec<LaneSpec<'_>> = (0..k)
            .map(|r| LaneSpec {
                state: r,
                tokens: &reqs[r].suffix,
            })
            .collect();
        let a = rig.wave(&specs, false, &policy)?;
        // Sum of one-lane waves: fresh prefixes again.
        for r in 0..k {
            rig.prep(r, &reqs[r].prefix)?;
        }
        let mut b = 0.0;
        for r in 0..k {
            b += rig.wave(
                &[LaneSpec {
                    state: r,
                    tokens: &reqs[r].suffix,
                }],
                false,
                &policy,
            )?;
        }
        // Sum of product singleton suffix calls (prefix outside the timing).
        let mut c = 0.0;
        for req in reqs {
            c += rig.product_run(req, false)?.1;
        }
        let tag = if rep == 0 { "warmup" } else { "rep" };
        ctx.log.say(format!(
            "bench {tag} {rep}: concat {a:.4}s ({:.0} rows/s)  one-lane waves {b:.4}s ({:.0} rows/s)  singleton {c:.4}s ({:.0} rows/s)",
            total as f64 / a,
            total as f64 / b,
            total as f64 / c
        ));
        if rep > 0 {
            concat.push(a);
            one_lane.push(b);
            singleton.push(c);
        }
    }
    let arm = |name: &str, v: &[f64]| {
        let m = median(v);
        json!({"arm": name, "seconds": v, "median_seconds": m, "median_rows_per_s": total as f64 / m})
    };
    let meta = json!({
        "mode": "bench",
        "mix": reqs.iter().map(|r| r.suffix.len()).collect::<Vec<_>>(),
        "depths": reqs.iter().map(|r| r.depth).collect::<Vec<_>>(),
        "policy": policy.describe(),
        "reps": args.reps,
        "rows": total,
        "verdict": "BENCH",
        "arms": [arm("concat_wave", &concat), arm("sum_one_lane_waves", &one_lane), arm("sum_product_singletons", &singleton)],
    });
    for a in meta["arms"].as_array().unwrap() {
        ctx.log.say(format!(
            "bench {}: median {:.4}s = {:.0} rows/s over {} rows",
            a["arm"].as_str().unwrap(),
            a["median_seconds"].as_f64().unwrap(),
            a["median_rows_per_s"].as_f64().unwrap(),
            total
        ));
    }
    ctx.runs.push(meta);
    Ok(())
}

// ── main ─────────────────────────────────────────────────────────────

fn gdn_prefill_name() -> &'static str {
    match qwen4_gdn_prefill() {
        Qwen4GdnPrefill::Wy => "wy",
        Qwen4GdnPrefill::Persistent => "persistent",
        Qwen4GdnPrefill::Canon => "canon",
    }
}

fn main() {
    match real_main() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(m) => {
            eprintln!("fn_prefill_g0: {m}");
            std::process::exit(2);
        }
    }
}

fn real_main() -> R<bool> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out).map_err(e)?;

    let longest = args
        .mix
        .iter()
        .enumerate()
        .map(|(i, &n)| args.depths[i % args.depths.len()] + n)
        .max()
        .unwrap();
    let max_seq = args.max_seq.unwrap_or((longest + 64).div_ceil(1024) * 1024);
    if longest + 2 > max_seq {
        return Err(format!(
            "--max-seq {max_seq} cannot hold depth + rows = {longest}"
        ));
    }

    let mut ctx = Ctx {
        log: Log::default(),
        runs: Vec::new(),
        all_pass: true,
    };
    let file_size = fs::metadata(&args.model).map_err(e)?.len();
    let (_source, w, tokenizer) = load(&args.model, max_seq)?;
    let arch = w.gpu.arch.to_string();
    let invariant_env = prefill_pm_invariant_enabled();
    let invariant_effective = invariant_env && arch == "gfx1151";
    ctx.log.say(format!("gpu arch: {arch}"));
    ctx.log.say(format!(
        "model: {} ({file_size} bytes)",
        args.model.display()
    ));
    ctx.log.say(format!(
        "env: HIPFIRE_PREFILL_PM_INVARIANT={} (effective={invariant_effective}) HIPFIRE_QWEN4_GDN_PREFILL={}",
        invariant_env,
        gdn_prefill_name()
    ));
    let policy_text = parse_policy(&args.policy)?.describe();
    ctx.log.say(format!(
        "args: mode={} mix={:?} depths={:?} orders={:?} chunkings={:?} policy={} ({policy_text}) stage_g0={} controls={} all_logits={} reps={} max_seq={max_seq}",
        args.mode, args.mix, args.depths, args.orders, args.chunkings, args.policy, args.stage_g0,
        args.controls, args.all_logits, args.reps
    ));

    let text = fs::read_to_string(&args.text).map_err(e)?;
    let toks = tokenizer.encode(&text);
    if toks.is_empty() {
        return Err("TEXT tokenized to nothing".into());
    }
    if toks.len() < longest + 1 {
        ctx.log.say(format!(
            "warning: TEXT has {} tokens, fewer than the longest request {longest}; prompts wrap",
            toks.len()
        ));
    }
    let reqs = build_reqs(&toks, &args.mix, &args.depths);

    // Output buffers: Final rows are one per lane; all-rows logits only with
    // --all-logits and a small total.
    let vocab = w.vocab;
    let total_rows: usize = reqs.iter().map(|r| r.suffix.len()).sum();
    let max_len = reqs.iter().map(|r| r.suffix.len()).max().unwrap();
    let all_rows = if !args.all_logits {
        0
    } else if args.mode == "concat" {
        total_rows
    } else {
        max_len
    };
    if args.all_logits && all_rows > ALL_LOGITS_MAX_ROWS {
        return Err(format!(
            "--all-logits refused: {all_rows} rows exceed {ALL_LOGITS_MAX_ROWS} (the all-rows logits buffer is rows*vocab F32)"
        ));
    }
    let mut w = w;
    let final_logits = w
        .gpu
        .zeros(&[reqs.len().max(1) * vocab], DType::F32)
        .map_err(e)?;
    let top1 = w
        .gpu
        .zeros(
            &[reqs.len().max(all_rows).max(1) * std::mem::size_of::<i32>()],
            DType::Raw,
        )
        .map_err(e)?;
    let all = if all_rows > 0 {
        Some(w.gpu.zeros(&[all_rows * vocab], DType::F32).map_err(e)?)
    } else {
        None
    };
    let tile = TILE_ROWS.min(w.bundle.spec_chunk_rows().unwrap_or(TILE_ROWS));
    let mut rig = Rig {
        w,
        pool: Vec::new(),
        sc: Scratch {
            final_logits,
            top1,
            all,
        },
        tile,
        prefix_policy: LaneStagePolicy::all_per_lane(),
    };
    let mut cache = RefCache {
        dir: args.out.join("ref"),
        keep_raw: args.keep_raw,
        sets: HashMap::new(),
    };

    let started = Instant::now();
    let result = match args.mode.as_str() {
        "concat" => run_concat(&args, &mut ctx, &mut rig, &mut cache, &reqs),
        "chunk" => run_chunk(&args, &mut ctx, &mut rig, &mut cache, &reqs),
        "singleton" => run_singleton(&args, &mut ctx, &mut rig, &reqs),
        "bench" => run_bench(&args, &mut ctx, &mut rig, &reqs),
        other => Err(format!("unknown mode {other}")),
    };
    let is_bench = args.mode == "bench";
    let pass = match &result {
        Ok(()) => is_bench || ctx.all_pass,
        Err(m) => {
            ctx.log.say(format!("error: {m}"));
            false
        }
    };
    ctx.log.say(format!(
        "{} {} runs in {:.1}s",
        if pass { "OVERALL PASS" } else { "OVERALL FAIL" },
        ctx.runs.len(),
        started.elapsed().as_secs_f64()
    ));

    let summary = json!({
        "tool": "fn_prefill_g0",
        "mode": args.mode,
        "model": args.model.display().to_string(),
        "text": args.text.display().to_string(),
        "arch": arch,
        "env": {
            "HIPFIRE_PREFILL_PM_INVARIANT": std::env::var("HIPFIRE_PREFILL_PM_INVARIANT").ok(),
            "HIPFIRE_QWEN4_GDN_PREFILL": std::env::var("HIPFIRE_QWEN4_GDN_PREFILL").ok(),
            "invariant_enabled": invariant_env,
            "invariant_effective": invariant_effective,
            "gdn_prefill": gdn_prefill_name(),
        },
        "args": {
            "mix": args.mix,
            "depths": args.depths,
            "orders": args.orders,
            "chunkings": args.chunkings,
            "policy": args.policy,
            "stage_g0": args.stage_g0,
            "controls": args.controls,
            "all_logits": args.all_logits,
            "reps": args.reps,
            "max_seq": max_seq,
            "tile_rows": tile,
        },
        "pass": pass,
        "error": result.as_ref().err(),
        "runs": ctx.runs,
        "log": ctx.log.lines,
    });
    fs::write(
        args.out.join("summary.json"),
        serde_json::to_vec_pretty(&summary).map_err(e)?,
    )
    .map_err(e)?;
    if !args.keep_raw {
        let _ = fs::remove_dir_all(args.out.join("ref"));
    }
    Ok(pass)
}
