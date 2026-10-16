// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next exact AR lane oracle (fn-batch PLAN §2, §6 S2 G0).
//!
//! fn_lane_oracle MODEL.mq4 TEXT.txt OUT_DIR [--ks 1,2,4,8] [--contexts 512,2048,8192]
//!     [--steps 64] [--max-seq 16384] [--policy production|per-lane|<spec>]
//!     [--g0] [--controls] [--bench N] [--spec ar|mtp] [--k K] [--routes floor|cycle|batched] [--promote P]
//!
//! The model loads once. Request `i` gets a unique prompt of exactly
//! `contexts[i % len]` tokens (TEXT tokens from offset `i * 997`, wrapping).
//!
//! Singleton reference (resident state, `prefill_final` +
//! `forward_token_or_argmax`, retained replay included): per request the
//! prefill logits row (`logits_0.bin`), the logits row after each of `steps`
//! greedy tokens (`logits_{s}.bin`), the fed token ids, and the full state
//! bytes at the checkpoints (after prefill, after step 1, after the last step)
//! are written under `OUT/ref/req{i}/`.
//!
//! Lane phase: the same requests run as lanes (`admit_lane`, hand-built
//! `BatchStepPlan`s through `lane_executor`): one prefill tile per step for
//! the first unfinished lane plus every decoding lane's AR row, so lanes
//! finish prefill on different steps and prefill tiles share steps with AR
//! rows. A lane may run at most three AR steps ahead before every lane has
//! finished prefill; after that all lanes decode together. Every AR row's
//! logits row, every committed id and every checkpoint's state bytes are
//! compared with the singleton reference (first differing byte reported).
//!
//! `--g0`: per-stage probe (each non-stateful stage alone at once-over-all-rows,
//! k=2 and k=8, `min(steps, 16)` steps) and the combined policy at every k.
//! `--controls`: negative controls that MUST be detected (lane swap, wrong
//! position). `--bench N`: lane step vs singleton step wall time.
//!
//! `--spec mtp [--k 3] [--routes cycle]`: MTP lanes (fn-batch S5). Requires
//! `HIPFIRE_MTP_INCREMENTAL=0` so the singleton reference window
//! `mtp_step(k)` is the batched route at depth `k`. The lanes run first
//! (prefill tiles, then every decoding lane calls `draft_step` and runs one
//! window per step: Verify rows `[seed, drafts..]` for a `Batched` route, one
//! Ar row for `ArSynced`/`ArRetired`); per lane per window the route, drafts
//! and committed ids are recorded and the lane's trunk + head + pending
//! hidden + draft request state are written under `OUT/lanes/req{i}/` after
//! prefill, window 1 and the last window. The reference phase replays every
//! lane on the resident singleton (`Qwen4MtpDrafter`, greedy: `mtp_prefill`,
//! then per recorded window `mtp_step` at `k_max` or `lab_ar_window`) and
//! compares committed ids per window and the checkpoint bytes item by item.
//! After an `ArRetired` window the head is not compared (head-free route).
//! Every lane's committed stream must also equal an AR singleton run's ids.
//! `--controls` re-runs a two-lane cycle case with ring rollback skipped (must
//! fail) next to the positive cycle control (rejected windows must match);
//! `--bench N` times lane steps (draft + verify) beside the AR numbers.
//!
//! `--spec mtp --promote P` (fn-batch S6, P >= 1): request 0 does not start as
//! a lane. It runs on the resident singleton (fresh `Qwen4MtpDrafter`,
//! `mtp_prefill`, then P windows of `mtp_step` at depth `k`, recorded as
//! `Batched { k_max: k }` windows), while the other requests are admitted and
//! prefilled as lanes. At the first step where every other lane has finished
//! prefill `prepare_promotion` + `commit_promotion` move request 0 into a lane
//! and it joins the remaining windows. Request 0's checkpoints are taken right
//! after the promotion, after its first lane window and after the last window;
//! the reference phase replays its whole window list (singleton windows, then
//! lane windows) on a fresh singleton. `--controls` adds a 2-lane run that
//! zeroes the promoted lane's pending hidden right after the promotion (must
//! fail the comparison).
//!
//! `--spec ar --promote P` (P >= 0, AR lanes beside a loaded MTP head: serve
//! with `serve.batch_spec = false`): after the AR matrix, for every k request
//! 0 runs on the resident native-MTP singleton (`mtp_prefill`, then P windows
//! of `mtp_step` at depth `k`), is moved into an AR lane by
//! `prepare_promotion` + `commit_promotion`, and the other k-1 requests are
//! admitted and prefilled as AR lanes. The promoted stream (singleton ids,
//! then every lane pick) must equal the AR singleton reference ids; at P = 0
//! (promotion right after the prefill pick, as serve promotes a request that
//! a peer queues behind) every lane step's logits row and the step-1 and
//! last-step trunk states are compared byte for byte too. `--controls` adds a
//! 2-lane run that skips `commit_promotion` (the lane keeps its fresh trunk
//! state; must fail the comparison).
//!
//! Exit 0 only when every compared run passes and every control is detected.
//! Nothing is written outside OUT_DIR.
//!
//! Model load follows `qwen4_kld`: admission on the GPU architecture
//! (`admit_hfqm_artifact_on`), slab layer arenas served by
//! `expert_arena::recipe_arena_payload`, and `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N`
//! (a fixed count; `auto` is refused) keeping layers 0..N resident with the
//! rest host-mapped and borrowed from the resident source bytes.
//!
//! Split bases resolve existing PLE sidecars using `HIPFIRE_PLE_FORMAT`.
//! Like the replay gate, this tool never derives missing q4 rows.

use hipfire_arch_qwen4::admit_hfqm_artifact_on;
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::gpu_forward_lanes::{LaneStagePolicy, StageName, StageShare};
use hipfire_arch_qwen4::lane::{LabRoutes, LaneSpecConfig};
use hipfire_arch_qwen4::lane_exec::{LaneDraft, LaneRoute};
use hipfire_arch_qwen4::mtp_spec::Qwen4MtpDrafter;
use hipfire_arch_qwen4::state::Qwen4State;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::{ModelSource as _, SourcePayload};
use hipfire_runtime::ple_sidecar::{PleRowsChoice, PleRowsOrigin};
use hipfire_runtime::slot_batch::{
    BatchStepPlan, RequestAdvance, RequestEpoch, RequestRows, RequestStepKind, RowRange, SlotBatch,
};
use hipfire_runtime::spec::{MtpDrafter, SpecRequestConfig};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::slot_pool::SlotId;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

type R<T> = Result<T, String>;

fn e<E: std::fmt::Display>(x: E) -> String {
    x.to_string()
}

/// A lane may run this many AR steps before every lane has finished prefill.
const EARLY_AR_STEPS: usize = 3;

// ── arguments ────────────────────────────────────────────────────────

struct Args {
    model: PathBuf,
    text: PathBuf,
    out: PathBuf,
    ks: Vec<usize>,
    contexts: Vec<usize>,
    steps: usize,
    max_seq: usize,
    policy: String,
    g0: bool,
    controls: bool,
    bench: usize,
    spec_mtp: bool,
    k: usize,
    routes: RoutesArg,
    promote: Option<usize>,
    /// Staged lane capacity override; `None` stages exactly the lanes the run drives.
    lane_capacity: Option<usize>,
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

fn parse_args() -> R<Args> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    parse_args_from(&raw)
}

fn parse_args_from(raw: &[String]) -> R<Args> {
    let usage = "usage: fn_lane_oracle MODEL.mq4 TEXT.txt OUT_DIR [--ks 1,2,4,8] [--contexts 512,2048,8192] \
                 [--steps 64] [--max-seq 16384] [--policy production|per-lane|<spec>] [--g0] [--controls] [--bench N] \
                 [--spec ar|mtp] [--k K] [--routes floor|cycle|batched] [--promote P] [--lane-capacity N]";
    let mut pos = Vec::new();
    let mut a = Args {
        model: PathBuf::new(),
        text: PathBuf::new(),
        out: PathBuf::new(),
        ks: vec![1, 2, 4, 8],
        contexts: vec![512, 2048, 8192],
        steps: 64,
        max_seq: 16384,
        policy: "production".into(),
        g0: false,
        controls: false,
        bench: 0,
        spec_mtp: false,
        k: 3,
        routes: RoutesArg::Cycle,
        promote: None,
        lane_capacity: None,
    };
    let mut i = 0;
    let value = |i: &mut usize, flag: &str| -> R<String> {
        *i += 1;
        raw.get(*i)
            .cloned()
            .ok_or_else(|| format!("{flag} needs a value\n{usage}"))
    };
    while i < raw.len() {
        match raw[i].as_str() {
            "--ks" => a.ks = parse_list(&value(&mut i, "--ks")?, "--ks")?,
            "--contexts" => a.contexts = parse_list(&value(&mut i, "--contexts")?, "--contexts")?,
            "--steps" => a.steps = value(&mut i, "--steps")?.parse().map_err(e)?,
            "--max-seq" => a.max_seq = value(&mut i, "--max-seq")?.parse().map_err(e)?,
            "--policy" => a.policy = value(&mut i, "--policy")?,
            "--bench" => a.bench = value(&mut i, "--bench")?.parse().map_err(e)?,
            "--spec" => {
                a.spec_mtp = match value(&mut i, "--spec")?.as_str() {
                    "ar" => false,
                    "mtp" => true,
                    other => return Err(format!("--spec must be ar|mtp, got '{other}'\n{usage}")),
                }
            }
            "--k" => a.k = value(&mut i, "--k")?.parse().map_err(e)?,
            "--routes" => a.routes = RoutesArg::parse(&value(&mut i, "--routes")?)?,
            "--promote" => a.promote = Some(value(&mut i, "--promote")?.parse().map_err(e)?),
            "--lane-capacity" => {
                a.lane_capacity = Some(value(&mut i, "--lane-capacity")?.parse().map_err(e)?)
            }
            "--g0" => a.g0 = true,
            "--controls" => a.controls = true,
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}\n{usage}")),
            other => pos.push(other.to_string()),
        }
        i += 1;
    }
    if pos.len() != 3 {
        return Err(usage.to_string());
    }
    a.model = pos[0].clone().into();
    a.text = pos[1].clone().into();
    a.out = pos[2].clone().into();
    if a.ks.is_empty() || a.ks.contains(&0) {
        return Err("--ks must be non-empty and positive".into());
    }
    if a.contexts.is_empty() || a.contexts.contains(&0) {
        return Err("--contexts must be non-empty and positive".into());
    }
    if a.lane_capacity == Some(0) {
        return Err("--lane-capacity must be >= 1".into());
    }
    if a.steps == 0 {
        return Err("--steps must be >= 1".into());
    }
    if (a.spec_mtp || a.promote.is_some()) && (a.k == 0 || a.k > 7) {
        return Err("--k must be 1..=7".into());
    }
    if a.spec_mtp && a.g0 {
        return Err("--g0 probes the AR lane stages; use it with --spec ar".into());
    }
    if let Some(p) = a.promote {
        if a.spec_mtp && p == 0 {
            return Err("--spec mtp --promote must be >= 1".into());
        }
        if p >= a.steps {
            return Err(format!(
                "--promote {p} must be below --steps {} (the lane needs a window)",
                a.steps
            ));
        }
    }
    Ok(a)
}

/// Most lanes one lane store holds (`MAX_LANES` in `lane.rs`, private there).
const LANE_STORE_MAX_LANES: usize = 64;

/// Logical lanes the run drives: the active `--ks` maximum, 8 for `--g0`, 2 for
/// `--controls`. This is the singleton-reference count and never depends on
/// `--lane-capacity`.
fn required_lanes(args: &Args) -> usize {
    let mut max_k = *args.ks.iter().max().unwrap();
    if args.g0 {
        max_k = max_k.max(8);
    }
    if args.controls {
        max_k = max_k.max(2);
    }
    max_k
}

/// Lanes (and row budget) the store is staged with: the override, else
/// `required`. An override below the lanes the run drives, or above what a
/// lane store holds, is refused before any GPU work.
fn resolve_lane_capacity(requested: Option<usize>, required: usize) -> R<usize> {
    let capacity = requested.unwrap_or(required);
    if capacity < required {
        return Err(format!(
            "--lane-capacity {capacity} is below the {required} lanes the active ks/controls/g0 need"
        ));
    }
    if capacity > LANE_STORE_MAX_LANES {
        return Err(format!(
            "--lane-capacity {capacity} exceeds the lane store maximum {LANE_STORE_MAX_LANES}"
        ));
    }
    Ok(capacity)
}

// ── log / summary ────────────────────────────────────────────────────

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

// ── byte comparison ──────────────────────────────────────────────────

#[derive(Default)]
struct Diff {
    items: usize,
    bytes: u64,
    bad: usize,
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
            self.note(format!("{label}: token id ref {reference} got {got}"));
        }
    }

    fn counts(&mut self, label: &str, reference: usize, got: usize) {
        self.items += 1;
        self.bytes += 8;
        if reference != got {
            self.note(format!("{label}: ref {reference} got {got}"));
        }
    }

    /// Compare two token id lists; the first differing index is reported.
    fn id_lists(&mut self, label: &str, reference: &[u32], got: &[u32]) {
        self.items += 1;
        self.bytes += 4 * reference.len() as u64;
        if reference == got {
            return;
        }
        match reference.iter().zip(got).position(|(a, b)| a != b) {
            Some(i) => self.note(format!(
                "{label}: first differing id at index {i} (ref {} got {}); ref {reference:?} got {got:?}",
                reference[i], got[i]
            )),
            None => self.note(format!("{label}: length ref {} got {}; ref {reference:?} got {got:?}", reference.len(), got.len())),
        }
    }
}

// ── device reads / state capture ─────────────────────────────────────

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
/// the family's layer list.
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
    step: usize,
    path: PathBuf,
    /// (name, offset, len) of every item in the `.bin`.
    index: Vec<(String, u64, u64)>,
}

struct RefReq {
    prompt: Vec<u32>,
    /// Token fed at step `s` (1-based): `ids[s - 1]`.
    ids: Vec<u32>,
    dir: PathBuf,
    ckpts: Vec<RefCkpt>,
}

fn ckpt_tag(step: usize) -> String {
    if step == 0 {
        "prefill".into()
    } else {
        format!("step{step}")
    }
}

fn write_ckpt(dir: &Path, step: usize, items: &[Item]) -> R<RefCkpt> {
    let tag = ckpt_tag(step);
    let path = dir.join(format!("state_{tag}.bin"));
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
        dir.join(format!("state_{tag}.json")),
        serde_json::to_vec_pretty(&idx).map_err(e)?,
    )
    .map_err(e)?;
    Ok(RefCkpt { step, path, index })
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

// ── world: model, singleton reference, lane plumbing ────────────────

struct World {
    gpu: Gpu,
    bundle: Qwen4Bundle,
    vocab: usize,
    /// Singleton logits row.
    row: GpuTensor,
    max_lanes: usize,
}

fn read_row(w: &World) -> R<Vec<u8>> {
    w.gpu.hip.device_synchronize().map_err(e)?;
    read_prefix(&w.gpu, &w.row, w.vocab * 4)
}

fn record_ref(
    w: &mut World,
    i: usize,
    prompt: Vec<u32>,
    steps: usize,
    ckpt_steps: &[usize],
    dir: &Path,
) -> R<RefReq> {
    fs::create_dir_all(dir).map_err(e)?;
    let mut rq = RefReq {
        prompt,
        ids: Vec::with_capacity(steps),
        dir: dir.to_path_buf(),
        ckpts: Vec::new(),
    };
    w.bundle.reset(&mut w.gpu).map_err(e)?;
    let prompt = rq.prompt.clone();
    w.bundle
        .prefill_final(&mut w.gpu, &prompt, 0, &w.row)
        .map_err(e)?;
    fs::write(dir.join("logits_0.bin"), read_row(w)?).map_err(e)?;
    let items = capture(&w.gpu, &w.bundle.state)?;
    rq.ckpts.push(write_ckpt(dir, 0, &items)?);
    for s in 1..=steps {
        let id = w
            .bundle
            .forward_token_or_argmax(&mut w.gpu, None, &w.row)
            .map_err(|x| format!("request {i} singleton step {s}: {x}"))?;
        rq.ids.push(id);
        fs::write(dir.join(format!("logits_{s}.bin")), read_row(w)?).map_err(e)?;
        if ckpt_steps.contains(&s) {
            let items = capture(&w.gpu, &w.bundle.state)?;
            rq.ckpts.push(write_ckpt(dir, s, &items)?);
        }
    }
    fs::write(
        dir.join("ids.txt"),
        rq.ids
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .map_err(e)?;
    Ok(rq)
}

// ── plans ────────────────────────────────────────────────────────────

struct RowSpec {
    slot: usize,
    epoch: RequestEpoch,
    tokens: Vec<u32>,
    start: usize,
    kind: RequestStepKind,
}

/// One step's plan: per-slot rows for every slot `0..max_lanes` (idle slots
/// empty), requests in slot order, absolute positions.
fn make_plan(max_lanes: usize, mut rows: Vec<RowSpec>) -> BatchStepPlan {
    rows.sort_by_key(|r| r.slot);
    let mut per: Vec<(SlotId, Vec<u32>, usize)> =
        (0..max_lanes).map(|s| (SlotId(s), Vec::new(), 0)).collect();
    for r in &rows {
        per[r.slot] = (SlotId(r.slot), r.tokens.clone(), r.start);
    }
    let triples: Vec<(SlotId, &[u32], usize)> =
        per.iter().map(|(s, t, p)| (*s, t.as_slice(), *p)).collect();
    let batch = SlotBatch::build(&triples);
    let mut plan = BatchStepPlan {
        batch,
        ..Default::default()
    };
    let mut begin = 0;
    for r in &rows {
        plan.requests.push(RequestRows {
            epoch: r.epoch,
            rows: RowRange {
                begin,
                len: r.tokens.len(),
            },
            kind: r.kind,
        });
        match r.kind {
            RequestStepKind::Ar => plan.decode_rows += r.tokens.len(),
            RequestStepKind::Prefill => plan.prefill_rows += r.tokens.len(),
            RequestStepKind::Verify { .. } => plan.verify_rows += r.tokens.len(),
            RequestStepKind::Forced => plan.forced_rows += r.tokens.len(),
        }
        begin += r.tokens.len();
    }
    plan
}

struct StepErr {
    stage: &'static str,
    msg: String,
}

impl StepErr {
    fn text(&self) -> String {
        format!("{} failed: {}", self.stage, self.msg)
    }
}

struct Stepped {
    /// Logits row of every AR request and every prompt-completing prefill tile.
    logits: Vec<(RequestEpoch, Vec<u8>)>,
    advances: Vec<RequestAdvance>,
    /// Packed per-row target picks of the step.
    picks: Vec<u32>,
}

fn read_step_logits(
    gpu: &Gpu,
    bundle: &Qwen4Bundle,
    plan: &BatchStepPlan,
    picks: &[u32],
    vocab: usize,
    max_lanes: usize,
) -> R<Vec<(RequestEpoch, Vec<u8>)>> {
    gpu.hip.device_synchronize().map_err(e)?;
    let store = bundle.lanes().ok_or("lane store vanished")?;
    let t = store.logits();
    let mut out = Vec::new();
    let (mut ar, mut pre) = (0usize, 0usize);
    for rr in &plan.requests {
        match rr.kind {
            RequestStepKind::Ar => {
                out.push((rr.epoch, read_view(gpu, t, ar * vocab, vocab)?));
                ar += 1;
            }
            RequestStepKind::Prefill => {
                if picks.get(rr.rows.end() - 1).is_some_and(|&p| p != u32::MAX) {
                    out.push((
                        rr.epoch,
                        read_view(gpu, t, (max_lanes + pre) * vocab, vocab)?,
                    ));
                }
                pre += 1;
            }
            _ => {}
        }
    }
    Ok(out)
}

/// provision + forward + (logits read) + commit of one plan.
fn exec_step(w: &mut World, plan: &BatchStepPlan, read_logits: bool) -> Result<Stepped, StepErr> {
    let World {
        gpu,
        bundle,
        vocab,
        max_lanes,
        ..
    } = w;
    let se = |stage: &'static str, msg: String| StepErr { stage, msg };
    let mut ex = bundle
        .lane_executor()
        .map_err(|x| se("provision", x.to_string()))?;
    ex.provision_step(gpu, plan)
        .map_err(|m| se("provision", m))?;
    let out = match ex.forward_step(gpu, plan) {
        Ok(o) => o,
        Err(m) => {
            ex.abort_step(plan);
            return Err(se("forward", m));
        }
    };
    let picks = out.target_picks.clone();
    drop(ex);
    let logits = if read_logits {
        match read_step_logits(gpu, bundle, plan, &out.target_picks, *vocab, *max_lanes) {
            Ok(l) => l,
            Err(m) => {
                if let Ok(mut ex) = bundle.lane_executor() {
                    ex.abort_step(plan);
                }
                return Err(se("logits read", m));
            }
        }
    } else {
        Vec::new()
    };
    let mut ex = bundle
        .lane_executor()
        .map_err(|x| se("commit", x.to_string()))?;
    match ex.commit_step(gpu, plan, out) {
        Ok(advances) => Ok(Stepped {
            logits,
            advances,
            picks,
        }),
        Err(m) => {
            ex.abort_step(plan);
            Err(se("commit", m))
        }
    }
}

// ── lane runs ────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Control {
    None,
    /// Two AR rows carry each other's tokens.
    SwapTokens,
    /// Two AR rows carry each other's row_slot.
    SwapRowSlot,
    /// Two AR rows swap token, position and row_slot (each request's rows
    /// hold the other lane's step).
    SwapRows,
    /// One lane runs an extra AR row before a step (position ahead).
    WrongPosition,
}

impl Control {
    fn name(self) -> &'static str {
        match self {
            Control::None => "none",
            Control::SwapTokens => "lane_swap_tokens",
            Control::SwapRowSlot => "lane_swap_row_slot",
            Control::SwapRows => "lane_swap_rows",
            Control::WrongPosition => "wrong_position",
        }
    }
}

struct Lane {
    req: usize,
    epoch: RequestEpoch,
    slot: usize,
    prompt_len: usize,
    prefilled: usize,
    committed: Vec<u32>,
    ar_steps: usize,
    /// Compare committed ids only (a lane promoted after native MTP windows:
    /// its trunk rows came from verify forwards, not the AR reference's).
    ids_only: bool,
}

struct Run {
    lanes: Vec<Lane>,
    diff: Diff,
    /// Mismatches attributable to the prefill (logits row 0, first id, prefill state).
    pre_bad: usize,
    pre_checked: usize,
    lane_steps: usize,
    ckpts: usize,
    error: Option<String>,
    control: Control,
    fired: bool,
    refused: Option<String>,
    control_note: Option<String>,
    release_errors: Vec<String>,
}

struct RunOpts {
    prefill_ckpt: bool,
}

fn ar_spec(w: &World, l: &Lane) -> R<RowSpec> {
    let lane = w
        .bundle
        .lanes()
        .and_then(|s| s.lane(&l.epoch))
        .ok_or_else(|| format!("lane of request {} vanished", l.req))?;
    let seed = lane
        .host
        .pending_seed
        .ok_or_else(|| format!("lane of request {} has no pending seed", l.req))?;
    Ok(RowSpec {
        slot: l.slot,
        epoch: l.epoch,
        tokens: vec![seed],
        start: lane.state.position,
        kind: RequestStepKind::Ar,
    })
}

fn ref_logits(rq: &RefReq, s: usize) -> R<Vec<u8>> {
    fs::read(rq.dir.join(format!("logits_{s}.bin"))).map_err(e)
}

fn lane_items(w: &World, l: &Lane) -> R<Vec<Item>> {
    let lane = w
        .bundle
        .lanes()
        .and_then(|s| s.lane(&l.epoch))
        .ok_or_else(|| format!("lane of request {} vanished", l.req))?;
    capture(&w.gpu, &lane.state)
}

/// Compare one committed step with the references.
fn absorb(
    w: &World,
    refs: &[RefReq],
    run: &mut Run,
    plan: &BatchStepPlan,
    res: &Stepped,
    steps: usize,
    ckpt_steps: &[usize],
    opts: &RunOpts,
) -> R<()> {
    if res.advances.len() != plan.requests.len() {
        return Err(format!(
            "{} advances for {} plan requests",
            res.advances.len(),
            plan.requests.len()
        ));
    }
    for (rr, adv) in plan.requests.iter().zip(&res.advances) {
        let idx = run
            .lanes
            .iter()
            .position(|l| l.epoch == rr.epoch)
            .ok_or("plan epoch not in this run")?;
        if adv.epoch != rr.epoch {
            return Err("advance order differs from the plan".into());
        }
        let (req, prompt_len) = (run.lanes[idx].req, run.lanes[idx].prompt_len);
        let rq = &refs[req];
        let logits_of = |epoch: RequestEpoch| {
            res.logits
                .iter()
                .find(|(e2, _)| *e2 == epoch)
                .map(|(_, v)| v)
        };
        match rr.kind {
            RequestStepKind::Prefill => {
                run.lanes[idx].prefilled += rr.rows.len;
                if run.lanes[idx].prefilled < prompt_len {
                    if !adv.committed_ids.is_empty() {
                        run.diff.note(format!(
                            "req {req}: non-final prefill tile committed {:?}",
                            adv.committed_ids
                        ));
                    }
                    continue;
                }
                let before = run.diff.bad;
                let got = logits_of(rr.epoch).ok_or("prompt-completing tile has no logits row")?;
                run.diff.check(
                    &format!("req {req} prefill logits"),
                    &ref_logits(rq, 0)?,
                    got,
                    true,
                );
                match adv.committed_ids.as_slice() {
                    [id] => {
                        run.diff
                            .ids(&format!("req {req} first token"), rq.ids[0], *id);
                        run.lanes[idx].committed.push(*id);
                    }
                    other => run.diff.note(format!(
                        "req {req}: prefill commit returned {other:?}, expected 1 id"
                    )),
                }
                if opts.prefill_ckpt {
                    let items = lane_items(w, &run.lanes[idx])?;
                    let rc = rq
                        .ckpts
                        .iter()
                        .find(|c| c.step == 0)
                        .ok_or("no prefill checkpoint")?;
                    compare_ckpt(&mut run.diff, &format!("req {req} prefill"), rc, &items)?;
                    run.ckpts += 1;
                }
                run.pre_checked += 1;
                run.pre_bad += run.diff.bad - before;
            }
            RequestStepKind::Ar => {
                run.lanes[idx].ar_steps += 1;
                let s = run.lanes[idx].ar_steps;
                let got = logits_of(rr.epoch).ok_or("AR row has no logits row")?;
                let ids_only = run.lanes[idx].ids_only;
                if !ids_only {
                    run.diff.check(
                        &format!("req {req} step {s} logits"),
                        &ref_logits(rq, s)?,
                        got,
                        true,
                    );
                }
                run.lane_steps += 1;
                match adv.committed_ids.as_slice() {
                    [id] => {
                        // Commit of step `s` is the token fed at step `s + 1`.
                        if s < steps {
                            run.diff
                                .ids(&format!("req {req} step {s} pick"), rq.ids[s], *id);
                        }
                        run.lanes[idx].committed.push(*id);
                    }
                    other => run.diff.note(format!(
                        "req {req} step {s}: AR commit returned {other:?}, expected 1 id"
                    )),
                }
                if adv.committed_position != prompt_len + s {
                    run.diff.note(format!(
                        "req {req} step {s}: committed position {} expected {}",
                        adv.committed_position,
                        prompt_len + s
                    ));
                }
                if ckpt_steps.contains(&s) && !ids_only {
                    let items = lane_items(w, &run.lanes[idx])?;
                    let rc = rq
                        .ckpts
                        .iter()
                        .find(|c| c.step == s)
                        .ok_or("missing reference checkpoint")?;
                    compare_ckpt(&mut run.diff, &format!("req {req} step {s}"), rc, &items)?;
                    run.ckpts += 1;
                }
            }
            other => run
                .diff
                .note(format!("req {req}: unexpected plan row kind {other:?}")),
        }
    }
    Ok(())
}

fn drive(
    w: &mut World,
    refs: &[RefReq],
    run: &mut Run,
    steps: usize,
    ckpt_steps: &[usize],
    opts: &RunOpts,
) -> R<()> {
    let early_cap = EARLY_AR_STEPS.min(steps);
    loop {
        if run.diff.bad > 0 || run.error.is_some() {
            break;
        }
        let all_prefilled = run.lanes.iter().all(|l| l.prefilled == l.prompt_len);
        let pre = run.lanes.iter().position(|l| l.prefilled < l.prompt_len);
        let ar: Vec<usize> = run
            .lanes
            .iter()
            .enumerate()
            .filter(|(_, l)| {
                l.prefilled == l.prompt_len
                    && l.ar_steps < steps
                    && (all_prefilled || l.ar_steps < early_cap)
            })
            .map(|(i, _)| i)
            .collect();
        if pre.is_none() && ar.is_empty() {
            break;
        }
        let mut specs = Vec::new();
        if let Some(i) = pre {
            let l = &run.lanes[i];
            let len = w
                .bundle
                .lane_executor()
                .map_err(e)?
                .exact_prefill_chunk_len(&l.epoch)?;
            if len == 0 || l.prefilled + len > l.prompt_len {
                return Err(format!(
                    "req {}: prefill tile of {len} rows at {}/{}",
                    l.req, l.prefilled, l.prompt_len
                ));
            }
            specs.push(RowSpec {
                slot: l.slot,
                epoch: l.epoch,
                tokens: refs[l.req].prompt[l.prefilled..l.prefilled + len].to_vec(),
                start: l.prefilled,
                kind: RequestStepKind::Prefill,
            });
        }
        for &i in &ar {
            specs.push(ar_spec(w, &run.lanes[i])?);
        }

        // Wrong position: one extra AR row for lane 0 before its third step.
        if run.control == Control::WrongPosition && !run.fired {
            if let Some(&i) = ar
                .iter()
                .find(|&&i| run.lanes[i].req == 0 && run.lanes[i].ar_steps == 2)
            {
                let solo = make_plan(w.max_lanes, vec![ar_spec(w, &run.lanes[i])?]);
                run.fired = true;
                match exec_step(w, &solo, false) {
                    Ok(_) => {
                        run.control_note = Some(format!(
                            "request 0 ran one extra AR row before step 3 (position {} instead of {})",
                            run.lanes[i].prompt_len + 3,
                            run.lanes[i].prompt_len + 2
                        ));
                    }
                    Err(x) => {
                        run.error = Some(format!("extra AR row: {}", x.text()));
                    }
                }
                continue;
            }
        }

        let mut plan = make_plan(w.max_lanes, specs);

        // Lane swap controls fire on the first plan with two AR rows past step 2.
        let swap = matches!(
            run.control,
            Control::SwapTokens | Control::SwapRowSlot | Control::SwapRows
        );
        let mut swapped = false;
        if swap && !run.fired {
            let eligible: Vec<usize> = ar
                .iter()
                .copied()
                .filter(|&i| run.lanes[i].ar_steps >= 2)
                .collect();
            if eligible.len() >= 2 {
                let find = |lane: &Lane| {
                    plan.requests
                        .iter()
                        .position(|r| r.epoch == lane.epoch)
                        .unwrap()
                };
                let (ra, rb) = (find(&run.lanes[eligible[0]]), find(&run.lanes[eligible[1]]));
                let (ba, bb) = (plan.requests[ra].rows.begin, plan.requests[rb].rows.begin);
                if plan.batch.tokens[ba] != plan.batch.tokens[bb] {
                    let b = &mut plan.batch;
                    if matches!(run.control, Control::SwapTokens | Control::SwapRows) {
                        b.tokens.swap(ba, bb);
                    }
                    if matches!(run.control, Control::SwapRowSlot | Control::SwapRows) {
                        b.row_slot.swap(ba, bb);
                    }
                    if run.control == Control::SwapRows {
                        b.positions.swap(ba, bb);
                    }
                    swapped = true;
                    run.fired = true;
                    run.control_note = Some(format!(
                        "requests {} and {} swapped ({}) at their step {}",
                        run.lanes[eligible[0]].req,
                        run.lanes[eligible[1]].req,
                        run.control.name(),
                        run.lanes[eligible[0]].ar_steps + 1
                    ));
                }
            }
        }

        let res = match exec_step(w, &plan, true) {
            Ok(r) => r,
            Err(x) if swapped && x.stage == "provision" => {
                // Refusal at provision is the detection; nothing ran, the
                // run continues and its comparisons cover "state untouched".
                run.refused = Some(x.msg);
                continue;
            }
            Err(x) => {
                run.error = Some(x.text());
                if swapped {
                    run.control_note = Some(format!(
                        "{} (executor error counts as detection)",
                        run.control_note.clone().unwrap_or_default()
                    ));
                }
                break;
            }
        };
        absorb(w, refs, run, &plan, &res, steps, ckpt_steps, opts)?;
    }
    if run.diff.bad == 0 && run.error.is_none() {
        for l in &run.lanes {
            if l.prefilled != l.prompt_len
                || l.ar_steps != steps
                || l.committed.len() != steps + 1
                || l.committed[..steps] != refs[l.req].ids[..steps]
            {
                run.diff.note(format!(
                    "req {}: run ended at prefill {}/{}, {}/{steps} AR steps, {} committed ids (expected {})",
                    l.req,
                    l.prefilled,
                    l.prompt_len,
                    l.ar_steps,
                    l.committed.len(),
                    steps + 1
                ));
            }
        }
    }
    Ok(())
}

/// `--spec ar --promote`: request 0 runs `windows` native-MTP windows at
/// depth `k` on the resident singleton, then moves into an AR lane.
#[derive(Clone, Copy)]
struct ArPromote {
    windows: usize,
    k: usize,
    /// Negative control: skip `commit_promotion` (the lane keeps its fresh
    /// trunk state, the resident is reset), the pre-fix failure.
    skip_commit: bool,
}

/// Run `rq`'s request on the resident native-MTP singleton for `p.windows`
/// windows, then `prepare_promotion` (AR store: no drafter parts) +
/// `commit_promotion` into lane `epoch`. The returned lane is Decoding with
/// the singleton's committed ids (the last one is the pending seed) counted
/// as AR steps, so the lane's next AR row is reference step `ids.len()`.
fn promote_ar(
    w: &mut World,
    rq: &RefReq,
    epoch: RequestEpoch,
    p: ArPromote,
    max_seq: usize,
    steps: usize,
) -> R<Lane> {
    let prompt = rq.prompt.clone();
    let (drafter, seed, wins) = singleton_phase(w, &prompt, p.k, max_seq, p.windows)?;
    let mut stream = vec![seed];
    for win in &wins {
        stream.extend_from_slice(&win.committed);
    }
    let moved = (|| -> R<usize> {
        if stream.len() > steps {
            return Err(format!(
                "promote: the singleton committed {} ids, more than --steps {steps}",
                stream.len()
            ));
        }
        let pending = *stream
            .last()
            .ok_or("the promoted request has no pending seed")?;
        let position = w.bundle.state.position;
        let expect = prompt.len() + stream.len() - 1;
        if position != expect {
            return Err(format!(
                "promote: singleton position {position} expected {expect}"
            ));
        }
        let slot = w
            .bundle
            .prepare_promotion(
                &mut w.gpu,
                epoch,
                prompt.clone(),
                stream.clone(),
                pending,
                position,
                None,
            )
            .map_err(|x| format!("prepare_promotion: {x}"))?;
        let committed = if p.skip_commit {
            w.bundle
                .reset(&mut w.gpu)
                .map_err(|x| format!("control reset: {x}"))
        } else {
            w.bundle
                .commit_promotion(&mut w.gpu, &epoch)
                .map_err(|x| format!("commit_promotion: {x}"))
        };
        if let Err(m) = committed {
            let _ = w.bundle.release_lane(&epoch);
            return Err(m);
        }
        Ok(slot)
    })();
    Box::new(drafter).mtp_free(&mut w.gpu);
    let slot = moved?;
    Ok(Lane {
        req: 0,
        epoch,
        slot,
        prompt_len: prompt.len(),
        prefilled: prompt.len(),
        ar_steps: stream.len() - 1,
        committed: stream,
        ids_only: p.windows > 0,
    })
}

struct RunReport {
    k: usize,
    label: String,
    pass: bool,
    steps: usize,
    lane_steps: usize,
    ckpts: usize,
    first: Option<String>,
    prefill_pass: bool,
    json: Value,
    run: Run,
}

struct App {
    w: World,
    refs: Vec<RefReq>,
    steps: usize,
    ckpt_steps: Vec<usize>,
    tag: u64,
    log: Log,
    /// Every request's prompt (spec mode has no AR reference store).
    prompts: Vec<Vec<u32>>,
    max_seq: usize,
    /// `OUT/lanes`: the MTP lanes' checkpoints.
    lanes_dir: PathBuf,
    /// AR singleton ids per request (spec mode, route-independent greedy ids).
    ar_ids: Vec<Vec<u32>>,
}

impl App {
    /// Admit `k` lanes, drive them against the references, release them.
    fn run_lanes(
        &mut self,
        k: usize,
        steps: usize,
        policy: LaneStagePolicy,
        label: &str,
        control: Control,
        opts: &RunOpts,
    ) -> RunReport {
        self.run_lanes_with(k, steps, policy, label, control, opts, None)
    }

    /// [`Self::run_lanes`]; with `promote`, request 0 starts on the resident
    /// native-MTP singleton and is promoted into an AR lane ([`promote_ar`])
    /// before requests 1..k are admitted.
    #[allow(clippy::too_many_arguments)]
    fn run_lanes_with(
        &mut self,
        k: usize,
        steps: usize,
        policy: LaneStagePolicy,
        label: &str,
        control: Control,
        opts: &RunOpts,
        promote: Option<ArPromote>,
    ) -> RunReport {
        let started = Instant::now();
        let mut run = Run {
            lanes: Vec::new(),
            diff: Diff::default(),
            pre_bad: 0,
            pre_checked: 0,
            lane_steps: 0,
            ckpts: 0,
            error: None,
            control,
            fired: false,
            refused: None,
            control_note: None,
            release_errors: Vec::new(),
        };
        let res = (|| -> R<()> {
            self.w
                .bundle
                .lanes_mut()
                .ok_or("lane store is not staged")?
                .set_stage_policy(policy);
            let mut first = 0;
            if let Some(p) = promote {
                self.tag += 1;
                let epoch = RequestEpoch {
                    request_tag: self.tag,
                    owner_generation: 1,
                };
                let lane = promote_ar(&mut self.w, &self.refs[0], epoch, p, self.max_seq, steps)?;
                run.diff.id_lists(
                    "req 0 singleton ids before the promotion vs the AR singleton",
                    &self.refs[0].ids[..lane.committed.len()],
                    &lane.committed,
                );
                run.lanes.push(lane);
                first = 1;
            }
            for req in first..k {
                self.tag += 1;
                let epoch = RequestEpoch {
                    request_tag: self.tag,
                    owner_generation: 1,
                };
                let prompt = self.refs[req].prompt.clone();
                let prompt_len = prompt.len();
                let slot = self
                    .w
                    .bundle
                    .admit_lane(&mut self.w.gpu, epoch, prompt)
                    .map_err(|x| format!("admit request {req}: {x}"))?;
                run.lanes.push(Lane {
                    req,
                    epoch,
                    slot,
                    prompt_len,
                    prefilled: 0,
                    committed: Vec::new(),
                    ar_steps: 0,
                    ids_only: false,
                });
            }
            drive(
                &mut self.w,
                &self.refs,
                &mut run,
                steps,
                &self.ckpt_steps,
                opts,
            )
        })();
        if let Err(m) = res {
            run.error = Some(m);
        }
        for l in &run.lanes {
            if let Err(x) = self.w.bundle.release_lane(&l.epoch) {
                run.release_errors
                    .push(format!("release request {}: {x}", l.req));
            }
        }
        let clean = run.diff.bad == 0 && run.error.is_none() && run.release_errors.is_empty();
        let first = run
            .error
            .clone()
            .map(|m| format!("executor error: {m}"))
            .or_else(|| run.diff.first.clone())
            .or_else(|| run.release_errors.first().cloned());
        let prefilled_lanes = k - usize::from(promote.is_some());
        let prefill_pass =
            run.pre_bad == 0 && run.pre_checked == prefilled_lanes && run.error.is_none();
        let json = json!({
            "k": k,
            "policy": label,
            "promote": promote.map(|p| json!({"windows": p.windows, "k": p.k, "skip_commit": p.skip_commit})),
            "control": control.name(),
            "pass": clean,
            "steps": steps,
            "lane_steps_compared": run.lane_steps,
            "state_checkpoints_compared": run.ckpts,
            "items_compared": run.diff.items,
            "bytes_compared": run.diff.bytes,
            "differences": run.diff.bad,
            "first_divergence": first,
            "first_divergences": run.diff.shown,
            "executor_error": run.error,
            "provision_refusal": run.refused,
            "control_note": run.control_note,
            "release_errors": run.release_errors,
            "prefill_pass": prefill_pass,
            "wall_s": started.elapsed().as_secs_f64(),
        });
        RunReport {
            k,
            label: label.to_string(),
            pass: clean,
            steps,
            lane_steps: run.lane_steps,
            ckpts: run.ckpts,
            first,
            prefill_pass,
            json,
            run,
        }
    }

    fn say_run(&mut self, r: &RunReport) {
        let verdict = if r.pass { "PASS" } else { "FAIL" };
        let mut line = format!(
            "run k={} policy={}: {verdict} (steps {}, lane-steps compared {}, state checkpoints {})",
            r.k, r.label, r.steps, r.lane_steps, r.ckpts
        );
        if let Some(first) = &r.first {
            line += &format!("\n    first divergence: {first}");
        }
        self.log.say(line);
    }
}

// ── bench ────────────────────────────────────────────────────────────

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

impl App {
    /// Wall time per lane step at `k` lanes (production policy), median over `n`.
    fn bench_lanes(&mut self, k: usize, n: usize) -> R<f64> {
        self.w
            .bundle
            .lanes_mut()
            .ok_or("lane store is not staged")?
            .set_stage_policy(LaneStagePolicy::production());
        let mut lanes: Vec<Lane> = Vec::new();
        let result = (|| -> R<f64> {
            for req in 0..k {
                self.tag += 1;
                let epoch = RequestEpoch {
                    request_tag: self.tag,
                    owner_generation: 1,
                };
                let prompt = self.prompts[req].clone();
                let prompt_len = prompt.len();
                let slot = self
                    .w
                    .bundle
                    .admit_lane(&mut self.w.gpu, epoch, prompt)
                    .map_err(e)?;
                lanes.push(Lane {
                    req,
                    epoch,
                    slot,
                    prompt_len,
                    prefilled: 0,
                    committed: Vec::new(),
                    ar_steps: 0,
                    ids_only: false,
                });
            }
            while let Some(i) = lanes.iter().position(|l| l.prefilled < l.prompt_len) {
                let l = &lanes[i];
                let len = self
                    .w
                    .bundle
                    .lane_executor()
                    .map_err(e)?
                    .exact_prefill_chunk_len(&l.epoch)?;
                let plan = make_plan(
                    self.w.max_lanes,
                    vec![RowSpec {
                        slot: l.slot,
                        epoch: l.epoch,
                        tokens: self.prompts[l.req][l.prefilled..l.prefilled + len].to_vec(),
                        start: l.prefilled,
                        kind: RequestStepKind::Prefill,
                    }],
                );
                exec_step(&mut self.w, &plan, false).map_err(|x| x.text())?;
                lanes[i].prefilled += len;
            }
            let mut times = Vec::with_capacity(n);
            for it in 0..n + 3 {
                let specs = lanes
                    .iter()
                    .map(|l| ar_spec(&self.w, l))
                    .collect::<R<Vec<_>>>()?;
                let plan = make_plan(self.w.max_lanes, specs);
                let t = Instant::now();
                exec_step(&mut self.w, &plan, false).map_err(|x| x.text())?;
                self.w.gpu.hip.device_synchronize().map_err(e)?;
                if it >= 3 {
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
            Ok(median(&mut times))
        })();
        for l in &lanes {
            let _ = self.w.bundle.release_lane(&l.epoch);
        }
        result
    }

    /// Wall time per singleton `forward_token_or_argmax`, median over `n`.
    fn bench_singleton(&mut self, n: usize) -> R<f64> {
        let prompt = self.prompts[0].clone();
        self.w.bundle.reset(&mut self.w.gpu).map_err(e)?;
        self.w
            .bundle
            .prefill_final(&mut self.w.gpu, &prompt, 0, &self.w.row)
            .map_err(e)?;
        let mut times = Vec::with_capacity(n);
        for it in 0..n + 3 {
            self.w.gpu.hip.device_synchronize().map_err(e)?;
            let t = Instant::now();
            self.w
                .bundle
                .forward_token_or_argmax(&mut self.w.gpu, None, &self.w.row)
                .map_err(e)?;
            self.w.gpu.hip.device_synchronize().map_err(e)?;
            if it >= 3 {
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        Ok(median(&mut times))
    }
}

// ── MTP lanes (--spec mtp) ───────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RoutesArg {
    Floor,
    Cycle,
    Batched,
}

impl RoutesArg {
    fn parse(s: &str) -> R<Self> {
        match s {
            "floor" => Ok(RoutesArg::Floor),
            "cycle" => Ok(RoutesArg::Cycle),
            "batched" => Ok(RoutesArg::Batched),
            other => Err(format!(
                "--routes must be floor|cycle|batched, got '{other}'"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            RoutesArg::Floor => "floor",
            RoutesArg::Cycle => "cycle",
            RoutesArg::Batched => "batched",
        }
    }

    fn lab(self) -> LabRoutes {
        match self {
            RoutesArg::Floor => LabRoutes::Floor,
            RoutesArg::Cycle => LabRoutes::Cycle,
            RoutesArg::Batched => LabRoutes::Batched,
        }
    }
}

fn route_name(r: LaneRoute) -> &'static str {
    match r {
        LaneRoute::Ar => "ar",
        LaneRoute::ArSynced => "ar_synced",
        LaneRoute::ArRetired => "ar_retired",
        LaneRoute::Batched { .. } => "batched",
    }
}

/// One lane window as the lane step ran it.
struct SpecWin {
    route: LaneRoute,
    drafts: Vec<u32>,
    committed: Vec<u32>,
    accepted: usize,
    /// Target picks of the window's rows (verify picks for `Batched`).
    picks: Vec<u32>,
    /// Drafts the window offered: `drafts.len()` for a lane window; a
    /// singleton window records only the count.
    offered: usize,
    /// Ran on the resident singleton before the promotion.
    singleton: bool,
}

/// Request 0 under `--promote P`: `windows` windows ran on the resident
/// singleton, then the request moved into a lane at `position`.
struct Promote {
    windows: usize,
    position: usize,
    done: bool,
}

struct SpecLane {
    req: usize,
    epoch: RequestEpoch,
    slot: usize,
    prompt_len: usize,
    prefilled: usize,
    /// The prompt-completing tile's pick.
    first: Option<u32>,
    wins: Vec<SpecWin>,
    /// Lane checkpoints on disk: step = window number (0 = after prefill).
    ckpts: Vec<RefCkpt>,
    promote: Option<Promote>,
}

impl SpecLane {
    /// The lane's committed token stream: the prompt pick, then every window's ids.
    fn stream(&self) -> Vec<u32> {
        let mut s: Vec<u32> = self.first.into_iter().collect();
        for w in &self.wins {
            s.extend_from_slice(&w.committed);
        }
        s
    }

    /// An `ArRetired` window ran within the first `windows` windows: the head
    /// is frozen from then on and not compared.
    fn retired_by(&self, windows: usize) -> bool {
        self.wins[..windows.min(self.wins.len())]
            .iter()
            .any(|w| w.route == LaneRoute::ArRetired)
    }

    /// Request 0 under `--promote`, still on the resident singleton.
    fn pending_promotion(&self) -> bool {
        self.promote.as_ref().is_some_and(|p| !p.done)
    }

    /// `window` is the first lane window after the promotion.
    fn first_after_promotion(&self, window: usize) -> bool {
        self.promote
            .as_ref()
            .is_some_and(|p| p.windows + 1 == window)
    }
}

struct SpecRun {
    lanes: Vec<SpecLane>,
    diff: Diff,
    error: Option<String>,
    release_errors: Vec<String>,
    steps_run: usize,
    /// Request 0's resident singleton drafter until its promotion.
    drafter: Option<Qwen4MtpDrafter>,
    /// Negative control: zero the promoted lane's pending hidden right after the promotion.
    zero_pending: bool,
}

#[derive(Default, Clone, Copy)]
struct SpecStats {
    windows: usize,
    batched: usize,
    ar_synced: usize,
    ar_retired: usize,
    /// Batched windows that rejected at least one draft.
    rejected: usize,
    drafts: usize,
    accepted: usize,
    committed: usize,
}

fn spec_stats(lanes: &[SpecLane]) -> SpecStats {
    let mut s = SpecStats::default();
    for l in lanes {
        for w in &l.wins {
            s.windows += 1;
            s.committed += w.committed.len();
            match w.route {
                LaneRoute::Batched { .. } => {
                    s.batched += 1;
                    s.drafts += w.offered;
                    s.accepted += w.accepted;
                    if w.accepted < w.offered {
                        s.rejected += 1;
                    }
                }
                LaneRoute::ArSynced => s.ar_synced += 1,
                LaneRoute::ArRetired => s.ar_retired += 1,
                LaneRoute::Ar => {}
            }
        }
    }
    s
}

/// Trunk state bytes plus the lane's head, draft request state and pending
/// hidden (`mtp/` prefixed).
fn spec_lane_items(w: &World, l: &SpecLane) -> R<Vec<Item>> {
    let store = w.bundle.lanes().ok_or("lane store vanished")?;
    let lane = store
        .lane(&l.epoch)
        .ok_or_else(|| format!("lane of request {} vanished", l.req))?;
    let mut items = capture(&w.gpu, &lane.state)?;
    for (name, bytes) in store.lab_lane_items(&w.gpu, &l.epoch)? {
        items.push(Item {
            name: format!("mtp/{name}"),
            bytes,
        });
    }
    Ok(items)
}

/// The resident singleton's counterpart of [`spec_lane_items`].
fn singleton_items(gpu: &Gpu, bundle: &Qwen4Bundle, drafter: &Qwen4MtpDrafter) -> R<Vec<Item>> {
    let mut items = capture(gpu, &bundle.state)?;
    for (name, bytes) in bundle.lab_resident_head_items(gpu)? {
        items.push(Item {
            name: format!("mtp/{name}"),
            bytes,
        });
    }
    if !items.iter().any(|i| i.name == "mtp/pending_hidden") {
        items.push(Item {
            name: "mtp/pending_hidden".into(),
            bytes: drafter.lab_pending_hidden_bytes(gpu)?,
        });
    }
    Ok(items)
}

fn lane_checkpoint(w: &World, dir: &Path, l: &mut SpecLane, window: usize) -> R<()> {
    let items = spec_lane_items(w, l)?;
    let d = dir.join(format!("req{}", l.req));
    fs::create_dir_all(&d).map_err(e)?;
    let c = write_ckpt(&d, window, &items)?;
    l.ckpts.push(c);
    Ok(())
}

/// Compare a lane checkpoint file with the live singleton items by name
/// (`reference` = singleton, `got` = lane). `skip_head` leaves out the
/// `mtp/` items (the lane ran a head-free window).
fn compare_live(
    diff: &mut Diff,
    ctx: &str,
    lane: &RefCkpt,
    live: &[Item],
    skip_head: bool,
) -> R<()> {
    let keep = |name: &str| !(skip_head && name.starts_with("mtp/"));
    let mut f = File::open(&lane.path).map_err(e)?;
    let lane_names: HashSet<&str> = lane.index.iter().map(|(n, _, _)| n.as_str()).collect();
    for item in live.iter().filter(|i| keep(&i.name)) {
        if !lane_names.contains(item.name.as_str()) {
            diff.note(format!(
                "{ctx}: singleton item '{}' has no lane counterpart",
                item.name
            ));
        }
    }
    for (name, offset, len) in lane.index.iter().filter(|(n, _, _)| keep(n)) {
        let Some(r) = live.iter().find(|i| &i.name == name) else {
            diff.note(format!(
                "{ctx}: lane item '{name}' has no singleton counterpart"
            ));
            continue;
        };
        let mut buf = vec![0u8; *len as usize];
        f.seek(SeekFrom::Start(*offset)).map_err(e)?;
        f.read_exact(&mut buf).map_err(e)?;
        diff.check(&format!("{ctx} state {name}"), &r.bytes, &buf, false);
    }
    Ok(())
}

/// The lane's next row(s) for its drafted window.
fn spec_row(
    w: &World,
    req: usize,
    slot: usize,
    epoch: RequestEpoch,
    drafts: &[LaneDraft],
) -> R<RowSpec> {
    let d = drafts
        .iter()
        .find(|d| d.epoch == epoch)
        .ok_or_else(|| format!("draft_step returned no window for request {req}"))?;
    let lane = w
        .bundle
        .lanes()
        .and_then(|s| s.lane(&epoch))
        .ok_or_else(|| format!("lane of request {req} vanished"))?;
    let seed = lane
        .host
        .pending_seed
        .ok_or_else(|| format!("lane of request {req} has no pending seed"))?;
    let start = lane.state.position;
    match d.route {
        LaneRoute::Batched { .. } => {
            if d.drafts.is_empty() {
                return Err(format!(
                    "request {req}: a Batched window came back with no drafts"
                ));
            }
            let mut tokens = Vec::with_capacity(1 + d.drafts.len());
            tokens.push(seed);
            tokens.extend_from_slice(&d.drafts);
            Ok(RowSpec {
                slot,
                epoch,
                tokens,
                start,
                kind: RequestStepKind::Verify {
                    draft_len: d.drafts.len(),
                },
            })
        }
        LaneRoute::ArSynced | LaneRoute::ArRetired => Ok(RowSpec {
            slot,
            epoch,
            tokens: vec![seed],
            start,
            kind: RequestStepKind::Ar,
        }),
        LaneRoute::Ar => Err(format!(
            "request {req}: a spec lane was routed as a plain Ar row"
        )),
    }
}

/// Record one committed step of the lane phase.
fn absorb_spec(
    w: &World,
    run: &mut SpecRun,
    plan: &BatchStepPlan,
    res: &Stepped,
    drafts: &[LaneDraft],
    ckpt_windows: &[usize],
    dir: &Path,
) -> R<()> {
    if res.advances.len() != plan.requests.len() {
        return Err(format!(
            "{} advances for {} plan requests",
            res.advances.len(),
            plan.requests.len()
        ));
    }
    for (rr, adv) in plan.requests.iter().zip(&res.advances) {
        let idx = run
            .lanes
            .iter()
            .position(|l| l.epoch == rr.epoch)
            .ok_or("plan epoch not in this run")?;
        if adv.epoch != rr.epoch {
            return Err("advance order differs from the plan".into());
        }
        let SpecRun { lanes, diff, .. } = &mut *run;
        let l = &mut lanes[idx];
        match rr.kind {
            RequestStepKind::Prefill => {
                l.prefilled += rr.rows.len;
                if l.prefilled < l.prompt_len {
                    if !adv.committed_ids.is_empty() {
                        diff.note(format!(
                            "req {}: non-final prefill tile committed {:?}",
                            l.req, adv.committed_ids
                        ));
                    }
                    continue;
                }
                match adv.committed_ids.as_slice() {
                    [id] => l.first = Some(*id),
                    other => diff.note(format!(
                        "req {}: prefill commit returned {other:?}, expected 1 id",
                        l.req
                    )),
                }
                if ckpt_windows.contains(&0) {
                    lane_checkpoint(w, dir, l, 0)?;
                }
            }
            RequestStepKind::Ar | RequestStepKind::Verify { .. } => {
                let d = drafts
                    .iter()
                    .find(|d| d.epoch == rr.epoch)
                    .ok_or_else(|| format!("req {}: window without a draft_step entry", l.req))?;
                let win = SpecWin {
                    route: d.route,
                    drafts: d.drafts.clone(),
                    committed: adv.committed_ids.clone(),
                    accepted: adv.accepted_drafts,
                    picks: res
                        .picks
                        .get(rr.rows.begin..rr.rows.end())
                        .map(|s| s.to_vec())
                        .unwrap_or_default(),
                    offered: d.drafts.len(),
                    singleton: false,
                };
                let n = win.committed.len();
                let ok = match rr.kind {
                    RequestStepKind::Verify { draft_len } => n >= 1 && n <= draft_len + 1,
                    _ => n == 1,
                };
                if !ok {
                    diff.note(format!(
                        "req {} window {}: committed {n} ids for a {:?} window with {} drafts",
                        l.req,
                        l.wins.len() + 1,
                        win.route,
                        win.offered
                    ));
                }
                l.wins.push(win);
                let total: usize = l.wins.iter().map(|x| x.committed.len()).sum();
                if adv.committed_position != l.prompt_len + total {
                    diff.note(format!(
                        "req {} window {}: committed position {} expected {}",
                        l.req,
                        l.wins.len(),
                        adv.committed_position,
                        l.prompt_len + total
                    ));
                }
                let n = l.wins.len();
                if ckpt_windows.contains(&n) || l.first_after_promotion(n) {
                    lane_checkpoint(w, dir, l, n)?;
                }
            }
            other => diff.note(format!("req {}: unexpected plan row kind {other:?}", l.req)),
        }
    }
    Ok(())
}

/// Request 0 on the resident singleton: fresh greedy `Qwen4MtpDrafter`,
/// `mtp_prefill`, then `windows` windows of `mtp_step` at depth `k`, recorded as
/// `Batched { k_max: k }`. Returns the live drafter, the prompt pick and the windows.
fn singleton_phase(
    w: &mut World,
    prompt: &[u32],
    k: usize,
    max_seq: usize,
    windows: usize,
) -> R<(Qwen4MtpDrafter, u32, Vec<SpecWin>)> {
    let World { gpu, bundle, .. } = w;
    bundle.reset(gpu).map_err(e)?;
    let mut drafter = Qwen4MtpDrafter::new(k, max_seq, None);
    drafter.configure_request(SpecRequestConfig::default());
    let result = (|| -> R<(u32, Vec<SpecWin>)> {
        let seed = drafter
            .mtp_prefill(gpu, &mut *bundle, prompt, prompt, 0, false, &|| false)?
            .into_first_token()?;
        let eos = bundle.config.eos_token_id;
        let mut emitted = vec![seed];
        let mut pending = seed;
        let mut wins = Vec::with_capacity(windows);
        for n in 1..=windows {
            let position = bundle.state.position;
            let got =
                drafter.mtp_step(gpu, &mut *bundle, position, pending, &emitted, k, eos, None)?;
            pending = *got
                .committed
                .last()
                .ok_or_else(|| format!("singleton window {n} committed nothing"))?;
            emitted.extend_from_slice(&got.committed);
            wins.push(SpecWin {
                route: LaneRoute::Batched { k_max: k },
                drafts: Vec::new(),
                committed: got.committed,
                accepted: got.accepted,
                picks: Vec::new(),
                offered: got.drafts_generated,
                singleton: true,
            });
        }
        Ok((seed, wins))
    })();
    match result {
        Ok((seed, wins)) => Ok((drafter, seed, wins)),
        Err(m) => {
            Box::new(drafter).mtp_free(gpu);
            Err(m)
        }
    }
}

/// Promote request 0 (lane `idx`) from the resident singleton into its lane:
/// `prepare_promotion` + `commit_promotion`, then the checkpoint right after
/// the promotion (after the optional pending-hidden zeroing control).
fn promote_lane(
    w: &mut World,
    run: &mut SpecRun,
    idx: usize,
    prompt: Vec<u32>,
    dir: &Path,
) -> R<()> {
    let drafter = run
        .drafter
        .take()
        .ok_or("the promoted request has no singleton drafter")?;
    let zero = run.zero_pending;
    let l = &mut run.lanes[idx];
    let stream = l.stream();
    let epoch = l.epoch;
    let moved = (|| -> R<(usize, usize)> {
        let pending = *stream
            .last()
            .ok_or("the promoted request has no pending seed")?;
        let position = w.bundle.state.position;
        let expect = l.prompt_len + l.wins.iter().map(|x| x.committed.len()).sum::<usize>();
        if position != expect {
            return Err(format!(
                "req {}: singleton position {position} expected {expect} at the promotion",
                l.req
            ));
        }
        let parts = drafter.promotion_parts()?;
        let slot = w
            .bundle
            .prepare_promotion(
                &mut w.gpu,
                epoch,
                prompt,
                stream,
                pending,
                position,
                Some(parts),
            )
            .map_err(|x| format!("prepare_promotion: {x}"))?;
        w.bundle
            .commit_promotion(&mut w.gpu, &epoch)
            .map_err(|x| format!("commit_promotion: {x}"))?;
        Ok((slot, position))
    })();
    Box::new(drafter).mtp_free(&mut w.gpu);
    let (slot, position) = moved?;
    l.slot = slot;
    let windows = {
        let p = l
            .promote
            .as_mut()
            .ok_or("the promoted lane has no promotion record")?;
        p.done = true;
        p.position = position;
        p.windows
    };
    if zero {
        w.bundle
            .lanes_mut()
            .ok_or("lane store vanished")?
            .lab_zero_pending_hidden(&mut w.gpu, &l.epoch)
            .map_err(e)?;
    }
    lane_checkpoint(w, dir, l, windows)
}

/// The lane phase: prefill tiles (one per step, first unfinished lane) with
/// every decoding lane's window (`draft_step` + Ar / Verify rows).
fn drive_spec(
    w: &mut World,
    prompts: &[Vec<u32>],
    run: &mut SpecRun,
    steps: usize,
    k: usize,
    ckpt_windows: &[usize],
    dir: &Path,
) -> R<()> {
    let early_cap = EARLY_AR_STEPS.min(steps);
    loop {
        if run.diff.bad > 0 || run.error.is_some() {
            break;
        }
        let all_prefilled = run.lanes.iter().all(|l| l.prefilled == l.prompt_len);
        let pre = run.lanes.iter().position(|l| l.prefilled < l.prompt_len);
        if pre.is_none() {
            if let Some(i) = run.lanes.iter().position(|l| l.pending_promotion()) {
                let prompt = prompts[run.lanes[i].req].clone();
                promote_lane(w, run, i, prompt, dir)?;
            }
        }
        let dec: Vec<usize> = run
            .lanes
            .iter()
            .enumerate()
            .filter(|(_, l)| {
                l.prefilled == l.prompt_len
                    && !l.pending_promotion()
                    && l.wins.len() < steps
                    && (all_prefilled || l.wins.len() < early_cap)
            })
            .map(|(i, _)| i)
            .collect();
        if pre.is_none() && dec.is_empty() {
            break;
        }
        let mut specs = Vec::new();
        if let Some(i) = pre {
            let l = &run.lanes[i];
            let len = w
                .bundle
                .lane_executor()
                .map_err(e)?
                .exact_prefill_chunk_len(&l.epoch)?;
            if len == 0 || l.prefilled + len > l.prompt_len {
                return Err(format!(
                    "req {}: prefill tile of {len} rows at {}/{}",
                    l.req, l.prefilled, l.prompt_len
                ));
            }
            specs.push(RowSpec {
                slot: l.slot,
                epoch: l.epoch,
                tokens: prompts[l.req][l.prefilled..l.prefilled + len].to_vec(),
                start: l.prefilled,
                kind: RequestStepKind::Prefill,
            });
        }
        let mut drafts: Vec<LaneDraft> = Vec::new();
        if !dec.is_empty() {
            let reqs: Vec<(RequestEpoch, usize)> =
                dec.iter().map(|&i| (run.lanes[i].epoch, k)).collect();
            drafts = w
                .bundle
                .lane_executor()
                .map_err(e)?
                .draft_step(&mut w.gpu, &reqs)
                .map_err(|m| format!("draft_step failed: {m}"))?;
            for &i in &dec {
                let l = &run.lanes[i];
                specs.push(spec_row(w, l.req, l.slot, l.epoch, &drafts)?);
            }
        }
        let plan = make_plan(w.max_lanes, specs);
        let res = match exec_step(w, &plan, false) {
            Ok(r) => r,
            Err(x) => {
                run.error = Some(x.text());
                break;
            }
        };
        run.steps_run += 1;
        absorb_spec(w, run, &plan, &res, &drafts, ckpt_windows, dir)?;
    }
    if run.diff.bad == 0 && run.error.is_none() {
        for l in &run.lanes {
            if l.prefilled != l.prompt_len || l.first.is_none() || l.wins.len() != steps {
                run.diff.note(format!(
                    "req {}: run ended at prefill {}/{}, {}/{steps} windows",
                    l.req,
                    l.prefilled,
                    l.prompt_len,
                    l.wins.len()
                ));
            }
        }
    }
    Ok(())
}

/// `OUT/lanes/req{i}/windows.json`: the lane's prompt pick and every window
/// (route, drafts, committed ids, accepted drafts, verify picks).
fn write_windows(dir: &Path, l: &SpecLane) -> R<()> {
    let d = dir.join(format!("req{}", l.req));
    fs::create_dir_all(&d).map_err(e)?;
    let wins: Vec<Value> = l
        .wins
        .iter()
        .map(|x| {
            json!({
                "route": route_name(x.route),
                "k_max": match x.route { LaneRoute::Batched { k_max } => k_max, _ => 0 },
                "drafts": x.drafts,
                "offered": x.offered,
                "singleton": x.singleton,
                "committed": x.committed,
                "accepted": x.accepted,
                "picks": x.picks,
            })
        })
        .collect();
    let doc = json!({"first": l.first, "windows": wins});
    fs::write(
        d.join("windows.json"),
        serde_json::to_vec_pretty(&doc).map_err(e)?,
    )
    .map_err(e)?;
    Ok(())
}

struct SpecReport {
    pass: bool,
    first: Option<String>,
    json: Value,
    stats: SpecStats,
    run: SpecRun,
    line: String,
}

impl App {
    /// The first `n` ids of the AR singleton run for request `req`
    /// (computed once up to `bound` tokens, then reused).
    fn singleton_ids(&mut self, req: usize, n: usize, bound: usize) -> R<Vec<u32>> {
        if self.ar_ids.len() <= req {
            self.ar_ids.resize(req + 1, Vec::new());
        }
        if self.ar_ids[req].len() < n {
            let want = bound.max(n);
            let prompt = self.prompts[req].clone();
            let World {
                gpu, bundle, row, ..
            } = &mut self.w;
            bundle.reset(gpu).map_err(e)?;
            bundle.prefill_final(gpu, &prompt, 0, row).map_err(e)?;
            let mut ids = Vec::with_capacity(want);
            for s in 1..=want {
                ids.push(
                    bundle
                        .forward_token_or_argmax(gpu, None, row)
                        .map_err(|x| format!("request {req} singleton step {s}: {x}"))?,
                );
            }
            self.ar_ids[req] = ids;
        }
        Ok(self.ar_ids[req][..n].to_vec())
    }

    /// Replay one lane's recorded windows on the resident singleton and
    /// compare ids and checkpoints. Returns the checkpoints compared.
    fn replay_lane(&mut self, diff: &mut Diff, l: &SpecLane, k: usize) -> R<usize> {
        let prompt = self.prompts[l.req].clone();
        let max_seq = self.max_seq;
        let World { gpu, bundle, .. } = &mut self.w;
        bundle.reset(gpu).map_err(e)?;
        let mut drafter = Qwen4MtpDrafter::new(k, max_seq, None);
        drafter.configure_request(SpecRequestConfig::default());
        let mut compared = 0usize;
        let before = diff.bad;
        let result = (|| -> R<()> {
            let req = l.req;
            let seed = drafter
                .mtp_prefill(gpu, &mut *bundle, &prompt, &prompt, 0, false, &|| false)?
                .into_first_token()?;
            let lane_first = l
                .first
                .ok_or_else(|| format!("req {req}: the lane never produced its first token"))?;
            diff.ids(&format!("req {req} prompt pick"), seed, lane_first);
            if let Some(c) = l.ckpts.iter().find(|c| c.step == 0) {
                let live = singleton_items(gpu, bundle, &drafter)?;
                compare_live(diff, &format!("req {req} prefill"), c, &live, false)?;
                compared += 1;
            }
            let eos = bundle.config.eos_token_id;
            let mut emitted = vec![seed];
            let mut pending = seed;
            for (i, win) in l.wins.iter().enumerate() {
                if diff.bad > before {
                    break;
                }
                let n = i + 1;
                let position = bundle.state.position;
                let got = match win.route {
                    LaneRoute::Batched { k_max } => drafter.mtp_step(
                        gpu,
                        &mut *bundle,
                        position,
                        pending,
                        &emitted,
                        k_max,
                        eos,
                        None,
                    )?,
                    LaneRoute::ArSynced => {
                        drafter.lab_ar_window(gpu, &mut *bundle, position, pending, true)?
                    }
                    LaneRoute::ArRetired => {
                        drafter.lab_ar_window(gpu, &mut *bundle, position, pending, false)?
                    }
                    LaneRoute::Ar => {
                        return Err(format!(
                            "req {req} window {n}: plain Ar route in a spec run"
                        ))
                    }
                };
                let label = format!("req {req} window {n} ({})", route_name(win.route));
                diff.id_lists(
                    &format!("{label} committed"),
                    &got.committed,
                    &win.committed,
                );
                if matches!(win.route, LaneRoute::Batched { .. }) {
                    diff.counts(
                        &format!("{label} drafts offered"),
                        got.drafts_generated,
                        win.offered,
                    );
                    diff.counts(
                        &format!("{label} drafts accepted"),
                        got.accepted,
                        win.accepted,
                    );
                }
                if diff.bad > before {
                    break;
                }
                emitted.extend_from_slice(&got.committed);
                pending = *got.committed.last().ok_or("empty singleton window")?;
                if let Some(c) = l.ckpts.iter().find(|c| c.step == n) {
                    let live = singleton_items(gpu, bundle, &drafter)?;
                    compare_live(
                        diff,
                        &format!("req {req} window {n}"),
                        c,
                        &live,
                        l.retired_by(n),
                    )?;
                    compared += 1;
                }
            }
            Ok(())
        })();
        Box::new(drafter).mtp_free(gpu);
        result.map(|_| compared)
    }

    /// The lane's committed stream against the AR singleton's ids.
    fn compare_ar_ids(&mut self, diff: &mut Diff, l: &SpecLane, k: usize, steps: usize) -> R<()> {
        let stream = l.stream();
        let ids = self.singleton_ids(l.req, stream.len(), steps * (k + 1) + 1)?;
        diff.id_lists(
            &format!("req {} committed ids vs the AR singleton", l.req),
            &ids,
            &stream,
        );
        Ok(())
    }

    /// Admit `lanes` lanes, run `steps` windows each against the lane store
    /// (`routes`, optional `skip_rollback` negative control), release them and
    /// replay every lane on the singleton.
    #[allow(clippy::too_many_arguments)]
    fn run_spec_lanes(
        &mut self,
        lanes: usize,
        steps: usize,
        k: usize,
        policy: LaneStagePolicy,
        label: &str,
        routes: RoutesArg,
        skip_rollback: bool,
        promote: Option<usize>,
        zero_pending: bool,
    ) -> SpecReport {
        let started = Instant::now();
        let mut run = SpecRun {
            lanes: Vec::new(),
            diff: Diff::default(),
            error: None,
            release_errors: Vec::new(),
            steps_run: 0,
            drafter: None,
            zero_pending,
        };
        let mut ckpt_windows = vec![0, 1, steps];
        ckpt_windows.sort_unstable();
        ckpt_windows.dedup();
        let res = (|| -> R<()> {
            let store = self
                .w
                .bundle
                .lanes_mut()
                .ok_or("lane store is not staged")?;
            store.set_stage_policy(policy);
            store.lab_set_routes(routes.lab());
            store.lab_set_skip_rollback(skip_rollback);
            let mut first_lane = 0;
            if let Some(p) = promote {
                self.tag += 1;
                let epoch = RequestEpoch {
                    request_tag: self.tag,
                    owner_generation: 1,
                };
                let prompt_len = self.prompts[0].len();
                let (drafter, seed, wins) =
                    singleton_phase(&mut self.w, &self.prompts[0], k, self.max_seq, p)?;
                run.drafter = Some(drafter);
                run.lanes.push(SpecLane {
                    req: 0,
                    epoch,
                    slot: 0,
                    prompt_len,
                    prefilled: prompt_len,
                    first: Some(seed),
                    wins,
                    ckpts: Vec::new(),
                    promote: Some(Promote {
                        windows: p,
                        position: 0,
                        done: false,
                    }),
                });
                first_lane = 1;
            }
            for req in first_lane..lanes {
                self.tag += 1;
                let epoch = RequestEpoch {
                    request_tag: self.tag,
                    owner_generation: 1,
                };
                let prompt = self.prompts[req].clone();
                let prompt_len = prompt.len();
                let slot = self
                    .w
                    .bundle
                    .admit_lane(&mut self.w.gpu, epoch, prompt)
                    .map_err(|x| format!("admit request {req}: {x}"))?;
                run.lanes.push(SpecLane {
                    req,
                    epoch,
                    slot,
                    prompt_len,
                    prefilled: 0,
                    first: None,
                    wins: Vec::new(),
                    ckpts: Vec::new(),
                    promote: None,
                });
            }
            drive_spec(
                &mut self.w,
                &self.prompts,
                &mut run,
                steps,
                k,
                &ckpt_windows,
                &self.lanes_dir,
            )
        })();
        if let Err(m) = res {
            run.error = Some(m);
        }
        if let Some(d) = run.drafter.take() {
            Box::new(d).mtp_free(&mut self.w.gpu);
        }
        for l in &run.lanes {
            if let Err(m) = write_windows(&self.lanes_dir, l) {
                run.release_errors
                    .push(format!("windows.json of request {}: {m}", l.req));
            }
        }
        let lane_stats: Vec<Value> = run
            .lanes
            .iter()
            .map(|l| {
                self.w
                    .bundle
                    .lanes()
                    .and_then(|s| s.lane_spec_stats(&l.epoch))
                    .map_or(Value::Null, |s| {
                        json!({
                            "windows": s.windows, "ar_windows": s.ar_windows, "drafts": s.drafts,
                            "accepted": s.accepted, "retired": s.retired,
                        })
                    })
            })
            .collect();
        for l in run.lanes.iter().filter(|l| !l.pending_promotion()) {
            if let Err(x) = self.w.bundle.release_lane(&l.epoch) {
                run.release_errors
                    .push(format!("release request {}: {x}", l.req));
            }
        }
        if let Some(store) = self.w.bundle.lanes_mut() {
            store.lab_set_skip_rollback(false);
        }

        // Reference phase: the singleton replays every lane's windows.
        let mut ckpts = 0usize;
        if run.error.is_none() && run.diff.bad == 0 {
            let mut diff = std::mem::take(&mut run.diff);
            let lanes_done = std::mem::take(&mut run.lanes);
            for l in &lanes_done {
                match self.replay_lane(&mut diff, l, k) {
                    Ok(c) => ckpts += c,
                    Err(m) => {
                        run.error = Some(format!("replay of request {}: {m}", l.req));
                        break;
                    }
                }
                if let Err(m) = self.compare_ar_ids(&mut diff, l, k, steps) {
                    run.error = Some(format!("AR ids of request {}: {m}", l.req));
                    break;
                }
            }
            run.diff = diff;
            run.lanes = lanes_done;
        }

        let stats = spec_stats(&run.lanes);
        let clean = run.diff.bad == 0 && run.error.is_none() && run.release_errors.is_empty();
        let first = run
            .error
            .clone()
            .map(|m| format!("executor error: {m}"))
            .or_else(|| run.diff.first.clone())
            .or_else(|| run.release_errors.first().cloned());
        let tau: Vec<f64> = run
            .lanes
            .iter()
            .map(|l| {
                let n: usize = l.wins.iter().map(|w| w.committed.len()).sum();
                if l.wins.is_empty() {
                    0.0
                } else {
                    n as f64 / l.wins.len() as f64
                }
            })
            .collect();
        let head_free: Vec<usize> = run
            .lanes
            .iter()
            .filter(|l| l.retired_by(l.wins.len()))
            .map(|l| l.req)
            .collect();
        let control = if skip_rollback {
            "spec_skip_rollback"
        } else {
            "none"
        };
        let json = json!({
            "k": lanes,
            "spec": "mtp",
            "spec_k": k,
            "promote": promote,
            "promotion_position": run.lanes.first().and_then(|l| l.promote.as_ref()).filter(|p| p.done).map(|p| p.position),
            "routes": routes.name(),
            "policy": label,
            "control": control,
            "pass": clean,
            "steps": steps,
            "windows": stats.windows,
            "windows_by_route": {
                "batched": stats.batched, "ar_synced": stats.ar_synced, "ar_retired": stats.ar_retired,
            },
            "rejected_windows": stats.rejected,
            "drafts": stats.drafts,
            "accepted": stats.accepted,
            "committed_tokens": stats.committed,
            "tau_per_lane": tau,
            "lane_stats": lane_stats,
            "head_not_compared_after_retirement_reqs": head_free,
            "lane_steps_run": run.steps_run,
            "state_checkpoints_compared": ckpts,
            "items_compared": run.diff.items,
            "bytes_compared": run.diff.bytes,
            "differences": run.diff.bad,
            "first_divergence": first,
            "first_divergences": run.diff.shown,
            "executor_error": run.error,
            "release_errors": run.release_errors,
            "wall_s": started.elapsed().as_secs_f64(),
        });
        let promo = match run.lanes.first().and_then(|l| l.promote.as_ref()) {
            Some(p) if p.done => format!(" promote={}@{}", p.windows, p.position),
            Some(p) => format!(" promote={} (not reached)", p.windows),
            None => String::new(),
        };
        let mut line = format!(
            "run lanes={lanes} spec=mtp k={k} routes={} policy={label}{promo}{}: {} (windows {} [batched {}, ar_synced {}, \
             ar_retired {}], rejected {}, drafts {} accepted {}, tau {:?}, state checkpoints {})",
            routes.name(),
            if skip_rollback { " control=skip_rollback" } else { "" },
            if clean { "PASS" } else { "FAIL" },
            stats.windows,
            stats.batched,
            stats.ar_synced,
            stats.ar_retired,
            stats.rejected,
            stats.drafts,
            stats.accepted,
            tau.iter().map(|t| (t * 100.0).round() / 100.0).collect::<Vec<_>>(),
            ckpts
        );
        if !head_free.is_empty() {
            line += &format!("\n    head not compared after retirement on requests {head_free:?} (head-free ArRetired windows)");
        }
        if let Some(f) = &first {
            line += &format!("\n    first divergence: {f}");
        }
        SpecReport {
            pass: clean,
            first,
            json,
            stats,
            run,
            line,
        }
    }

    /// Lane step (draft + verify) wall time at `lanes` lanes, median over
    /// `n`; returns (step ms, committed tokens per second, committed tokens
    /// per lane step). `budget` 0 times the AR rows of spec lanes.
    fn bench_spec_lanes(
        &mut self,
        lanes_n: usize,
        n: usize,
        budget: usize,
        routes: RoutesArg,
    ) -> R<(f64, f64, f64)> {
        {
            let store = self
                .w
                .bundle
                .lanes_mut()
                .ok_or("lane store is not staged")?;
            store.set_stage_policy(LaneStagePolicy::production());
            store.lab_set_routes(routes.lab());
            store.lab_set_skip_rollback(false);
        }
        let mut lanes: Vec<Lane> = Vec::new();
        let result = (|| -> R<(f64, f64, f64)> {
            for req in 0..lanes_n {
                self.tag += 1;
                let epoch = RequestEpoch {
                    request_tag: self.tag,
                    owner_generation: 1,
                };
                let prompt = self.prompts[req].clone();
                let prompt_len = prompt.len();
                let slot = self
                    .w
                    .bundle
                    .admit_lane(&mut self.w.gpu, epoch, prompt)
                    .map_err(e)?;
                lanes.push(Lane {
                    req,
                    epoch,
                    slot,
                    prompt_len,
                    prefilled: 0,
                    committed: Vec::new(),
                    ar_steps: 0,
                    ids_only: false,
                });
            }
            while let Some(i) = lanes.iter().position(|l| l.prefilled < l.prompt_len) {
                let l = &lanes[i];
                let len = self
                    .w
                    .bundle
                    .lane_executor()
                    .map_err(e)?
                    .exact_prefill_chunk_len(&l.epoch)?;
                let plan = make_plan(
                    self.w.max_lanes,
                    vec![RowSpec {
                        slot: l.slot,
                        epoch: l.epoch,
                        tokens: self.prompts[l.req][l.prefilled..l.prefilled + len].to_vec(),
                        start: l.prefilled,
                        kind: RequestStepKind::Prefill,
                    }],
                );
                exec_step(&mut self.w, &plan, false).map_err(|x| x.text())?;
                lanes[i].prefilled += len;
            }
            let mut times = Vec::with_capacity(n);
            let mut tokens = 0usize;
            for it in 0..n + 3 {
                let reqs: Vec<(RequestEpoch, usize)> =
                    lanes.iter().map(|l| (l.epoch, budget)).collect();
                self.w.gpu.hip.device_synchronize().map_err(e)?;
                let t = Instant::now();
                let drafts = self
                    .w
                    .bundle
                    .lane_executor()
                    .map_err(e)?
                    .draft_step(&mut self.w.gpu, &reqs)
                    .map_err(|m| format!("draft_step failed: {m}"))?;
                let specs = lanes
                    .iter()
                    .map(|l| spec_row(&self.w, l.req, l.slot, l.epoch, &drafts))
                    .collect::<R<Vec<_>>>()?;
                let plan = make_plan(self.w.max_lanes, specs);
                let res = exec_step(&mut self.w, &plan, false).map_err(|x| x.text())?;
                self.w.gpu.hip.device_synchronize().map_err(e)?;
                if it >= 3 {
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                    tokens += res
                        .advances
                        .iter()
                        .map(|a| a.committed_ids.len())
                        .sum::<usize>();
                }
            }
            let total_ms: f64 = times.iter().sum();
            let per_step = tokens as f64 / (n * lanes_n) as f64;
            Ok((
                median(&mut times),
                tokens as f64 * 1000.0 / total_ms,
                per_step,
            ))
        })();
        for l in &lanes {
            let _ = self.w.bundle.release_lane(&l.epoch);
        }
        result
    }
}

/// `--spec mtp`: lane windows against the singleton `Qwen4MtpDrafter`.
fn spec_main(
    args: &Args,
    w: World,
    log: Log,
    toks: Vec<u32>,
    max_k: usize,
    lane_capacity: usize,
    file_size: u64,
    primary: LaneStagePolicy,
) -> R<bool> {
    let k = args.k;
    let prompts: Vec<Vec<u32>> = (0..max_k)
        .map(|i| prompt_for(&toks, i, args.contexts[i % args.contexts.len()]))
        .collect();
    let mut app = App {
        w,
        refs: Vec::new(),
        steps: args.steps,
        ckpt_steps: Vec::new(),
        tag: 0,
        log,
        prompts,
        max_seq: args.max_seq,
        lanes_dir: args.out.join("lanes"),
        ar_ids: Vec::new(),
    };
    let short_steps = args.steps.min(16);
    let mut all_pass = true;
    let mut runs: Vec<Value> = Vec::new();

    // ── main matrix ────────────────────────────────────────────────
    for &lanes in &args.ks {
        let r = app.run_spec_lanes(
            lanes,
            args.steps,
            k,
            primary,
            &args.policy,
            args.routes,
            false,
            args.promote,
            false,
        );
        app.log.say(r.line.clone());
        all_pass &= r.pass;
        runs.push(r.json);
    }

    // ── controls ───────────────────────────────────────────────────
    let mut controls_json = Vec::new();
    if args.controls {
        if short_steps < 4 {
            return Err("--controls needs --steps >= 4".into());
        }
        // Positive: cycle routes with rejected verify windows and ring rollback must match.
        let r = app.run_spec_lanes(
            2,
            short_steps,
            k,
            primary,
            &args.policy,
            RoutesArg::Cycle,
            false,
            None,
            false,
        );
        app.log.say(r.line.clone());
        let exercised = r.stats.rejected > 0 && r.stats.batched > 0;
        let ok = r.pass && exercised;
        all_pass &= ok;
        app.log.say(format!(
            "control spec_positive_cycle: {} ({} rejected Batched windows of {} with ring rollback)",
            if ok { "matched" } else { "FAILED" },
            r.stats.rejected,
            r.stats.batched
        ));
        controls_json.push(json!({
            "control": "spec_positive_cycle",
            "matched": ok,
            "rejected_windows": r.stats.rejected,
            "exercised": exercised,
            "run": r.json,
        }));
        // Negative: a rejected verify keeps its full rows; the comparison must fail.
        let r = app.run_spec_lanes(
            2,
            short_steps,
            k,
            primary,
            &args.policy,
            RoutesArg::Cycle,
            true,
            None,
            false,
        );
        app.log.say(r.line.clone());
        let exercised = r.stats.rejected > 0;
        let diverged = r.run.diff.bad > 0 || r.run.error.is_some();
        let (detected, how) = if !exercised {
            (
                false,
                "control was not exercised (no rejected verify window)".to_string(),
            )
        } else if diverged {
            (
                true,
                format!("comparison failed: {}", r.first.clone().unwrap_or_default()),
            )
        } else {
            (false, "comparison still passed".to_string())
        };
        all_pass &= detected;
        app.log.say(format!(
            "control spec_skip_rollback: {} ({how})",
            if detected { "detected" } else { "MISSED" }
        ));
        controls_json.push(json!({
            "control": "spec_skip_rollback",
            "detected": detected,
            "how": how,
            "rejected_windows": r.stats.rejected,
            "run": r.json,
        }));
        if let Some(p) = args.promote {
            // Negative: the promoted lane's pending hidden is zeroed right after the promotion; the comparison must fail.
            let p = p.min(short_steps - 1);
            let r = app.run_spec_lanes(
                2,
                short_steps,
                k,
                primary,
                &args.policy,
                args.routes,
                false,
                Some(p),
                true,
            );
            app.log.say(r.line.clone());
            let promoted = r
                .run
                .lanes
                .first()
                .and_then(|l| l.promote.as_ref())
                .is_some_and(|x| x.done);
            let diverged = r.run.diff.bad > 0 || r.run.error.is_some();
            let (detected, how) = if !promoted {
                (
                    false,
                    "control was not exercised (the promotion did not happen)".to_string(),
                )
            } else if diverged {
                (
                    true,
                    format!("comparison failed: {}", r.first.clone().unwrap_or_default()),
                )
            } else {
                (false, "comparison still passed".to_string())
            };
            all_pass &= detected;
            app.log.say(format!(
                "control promote_zero_pending: {} ({how})",
                if detected { "detected" } else { "MISSED" }
            ));
            controls_json.push(json!({
                "control": "promote_zero_pending",
                "detected": detected,
                "how": how,
                "promote": p,
                "run": r.json,
            }));
        }
    }

    // ── bench ──────────────────────────────────────────────────────
    let mut bench_json = Vec::new();
    if args.bench > 0 {
        let n = args.bench;
        let singleton_ms = app.bench_singleton(n)?;
        let singleton_tok_s = 1000.0 / singleton_ms;
        app.log.say(format!("bench singleton AR: {singleton_ms:.3} ms/step, {singleton_tok_s:.2} tok/s (median of {n}, request 0)"));
        for &lanes in &args.ks {
            let (ar_ms, ar_tok_s, _) = app.bench_spec_lanes(lanes, n, 0, RoutesArg::Floor)?;
            app.log.say(format!(
                "bench lanes={lanes} AR rows (spec lanes, draft budget 0): step_ms={ar_ms:.3}, tok_s={ar_tok_s:.2}, \
                 vs singleton {:.3}x",
                ar_tok_s / singleton_tok_s
            ));
            bench_json.push(json!({
                "k": lanes, "spec": "ar_budget0", "step_ms": ar_ms, "tok_s": ar_tok_s,
                "singleton_ms": singleton_ms, "speedup_vs_singleton": ar_tok_s / singleton_tok_s,
            }));
            for routes in [RoutesArg::Batched, RoutesArg::Floor] {
                let (ms, tok_s, per_lane_step) = app.bench_spec_lanes(lanes, n, k, routes)?;
                app.log.say(format!(
                    "bench lanes={lanes} spec=mtp k={k} routes={}: step_ms={ms:.3}, tok_s={tok_s:.2}, \
                     tokens_per_lane_step={per_lane_step:.3}, vs AR lanes {:.3}x, vs singleton {:.3}x",
                    routes.name(),
                    tok_s / ar_tok_s,
                    tok_s / singleton_tok_s
                ));
                bench_json.push(json!({
                    "k": lanes, "spec": "mtp", "spec_k": k, "routes": routes.name(), "step_ms": ms,
                    "tok_s": tok_s, "tokens_per_lane_step": per_lane_step, "ar_lane_tok_s": ar_tok_s,
                    "singleton_ms": singleton_ms, "speedup_vs_ar_lanes": tok_s / ar_tok_s,
                    "speedup_vs_singleton": tok_s / singleton_tok_s,
                }));
            }
        }
    }

    app.log.say(format!(
        "fn_lane_oracle: {}",
        if all_pass { "PASS" } else { "FAIL" }
    ));
    let summary = json!({
        "pass": all_pass,
        "gpu_arch": app.w.gpu.arch,
        "model": args.model.display().to_string(),
        "model_bytes": file_size,
        "args": {
            "ks": args.ks, "contexts": args.contexts, "steps": args.steps, "max_seq": args.max_seq,
            "lane_capacity": lane_capacity,
            "policy": args.policy, "policy_described": primary.describe(), "g0": false,
            "controls": args.controls, "bench": args.bench,
            "spec": "mtp", "k": args.k, "routes": args.routes.name(), "promote": args.promote,
        },
        "policy_evidence": primary.evidence(),
        "requests": app.prompts.iter().map(|p| json!({"context": p.len()})).collect::<Vec<_>>(),
        "runs": runs,
        "controls": controls_json,
        "bench": bench_json,
        "log": app.log.lines,
    });
    fs::write(
        args.out.join("summary.json"),
        serde_json::to_vec_pretty(&summary).map_err(e)?,
    )
    .map_err(e)?;
    Ok(all_pass)
}

// ── model load ───────────────────────────────────────────────────────

/// The PLE rows source of `path`, attached to `hfq`: the shared validation
/// entrypoint `attach_validation_ple_rows` under the same layout gate as the serve
/// loader (`resolve_qwen4_ple_rows_hfq`). A base without the Flash-Next PLE
/// layout (a developer profile's embedded rows, or a profile without PLE) and
/// a stock-FP8 base under `auto` have no sidecar to resolve: `None`. A missing
/// q4 sidecar is never derived here.
fn resolve_ple_rows(
    hfq: &mut HfqFile,
    path: &Path,
    request: hipfire_config::memory::PleFormatRequest,
    gpu_arch: &str,
) -> R<Option<PleRowsChoice>> {
    use hipfire_arch_qwen4::weights::is_ple_shard_name;
    if hfq.ple_rows_marker().is_none() && !hfq.tensors().iter().any(|t| is_ple_shard_name(&t.name))
    {
        return Ok(None);
    }
    if hfq
        .tensors()
        .iter()
        .any(|t| t.quant_type == 56 && is_ple_shard_name(&t.name))
    {
        return match request {
            hipfire_config::memory::PleFormatRequest::Auto => Ok(None),
            request => Err(format!(
                "qwen4: memory.ple_format = {request:?} requests PLE sidecar rows, but this stock-FP8 \
                 artifact reads its embedded FP8E4M3PT PLE rows; unset memory.ple_format"
            )),
        };
    }
    hipfire_runtime::ple_sidecar::attach_validation_ple_rows(hfq, path, request, gpu_arch)
        .map(Some)
        .map_err(|error| format!("qwen4 PLE rows: {error}"))
}

fn load(
    path: &Path,
    max_seq: usize,
    lane_capacity: usize,
    spec: Option<LaneSpecConfig>,
    attach_mtp: bool,
) -> R<(HfqModelSource, World, Tokenizer)> {
    let mut hfq = HfqFile::open(path).map_err(e)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(e)?;
    let mut gpu = Gpu::init().map_err(e)?;
    // The PLE rows source (`memory.ple_format` / `HIPFIRE_PLE_FORMAT`), bound
    // while the base is open and before admission, exactly as `qwen4_kld`,
    // `qwen4_greedy_ids` and the serve loader do. Never derives a missing q4
    // table (`hipfire ple derive` first); a sidecar that does not open and bind
    // is refused, never replaced by the embedded rows.
    let choice = resolve_ple_rows(
        &mut hfq,
        path,
        hipfire_config::memory::ple_format(),
        &gpu.arch,
    )?;
    // Admission is GPU-arch aware: an expert-slab artifact admits only on a
    // supported arch. Initialising the GPU allocates no model memory.
    let receipt = admit_hfqm_artifact_on(&hfq, Some(&gpu.arch)).map_err(e)?;
    match &choice {
        Some(choice) => {
            eprintln!(
                "fn_lane_oracle: ple rows: {} from {}",
                choice.format.name(),
                match &choice.origin {
                    PleRowsOrigin::Embedded => "embedded model file".to_string(),
                    PleRowsOrigin::Sidecar(sidecar) => sidecar.display().to_string(),
                }
            );
            if let Some(note) = &choice.note {
                eprintln!("fn_lane_oracle: ple rows: {note}");
            }
        }
        None => match &receipt.ple_rows {
            hipfire_arch_qwen4::Qwen4PleRows::Absent => eprintln!(
                "fn_lane_oracle: ple rows: none (profile {} declares no PLE)",
                receipt.config.profile.id()
            ),
            _ => eprintln!(
                "fn_lane_oracle: ple rows: embedded rows (no q4/q8 sidecar for this artifact)"
            ),
        },
    }
    // The loader's forward scratch plan, decided from the actual payload
    // while the mapping is readable.
    let scratch_mode = hipfire_arch_qwen4::artifact::source_forward_scratch_mode(
        &gpu,
        &hfq,
        &receipt.config,
        &receipt.manifest,
    )
    .map_err(e)?;
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
                    "fn_lane_oracle: routed experts: layers 0..{layers} in VRAM, {moved} tensors host-mapped"
                );
            }
            Some(hipfire_arch_qwen4::expert_residency::ExpertVramLayers::Auto) => {
                return Err("fn_lane_oracle takes a fixed expert layer count, not auto".into());
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
    let mut bundle = Qwen4Bundle::assemble_with_ple_rows(
        receipt.config,
        &receipt.manifest.identity.recipe,
        transaction,
        &receipt.placements,
        &mut gpu,
        max_seq,
        receipt.ple,
        receipt.ple_rows,
        state_format,
        backend,
        None,
    )
    .map_err(e)?;
    bundle
        .attach_forward(&mut gpu, max_seq, scratch_mode)
        .map_err(e)?;
    if attach_mtp {
        bundle.attach_mtp(&mut gpu, max_seq).map_err(e)?;
    }
    bundle
        .stage_lanes(&mut gpu, lane_capacity, lane_capacity, spec)
        .map_err(e)?;
    let row = gpu.zeros(&[vocab], DType::F32).map_err(e)?;
    let max_lanes = bundle
        .lanes()
        .map(|s| s.max_lanes())
        .ok_or("lane store is not staged")?;
    Ok((
        source,
        World {
            gpu,
            bundle,
            vocab,
            row,
            max_lanes,
        },
        tokenizer,
    ))
}

fn prompt_for(tokens: &[u32], i: usize, ctx: usize) -> Vec<u32> {
    (0..ctx)
        .map(|j| tokens[(i * 997 + j) % tokens.len()])
        .collect()
}

fn stage_rust_line(stage: StageName, share: StageShare) -> String {
    format!("    (StageName::{stage:?}, StageShare::{share:?}),")
}

// ── main ─────────────────────────────────────────────────────────────

fn main() {
    match real_main() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(m) => {
            eprintln!("fn_lane_oracle: {m}");
            std::process::exit(2);
        }
    }
}

fn real_main() -> R<bool> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out).map_err(e)?;
    let primary = LaneStagePolicy::parse(&args.policy)?;
    let spec_cfg = if args.spec_mtp {
        if !matches!(
            hipfire_config::developer_var("HIPFIRE_MTP_INCREMENTAL").as_deref(),
            Ok("0")
        ) {
            return Err(
                "--spec mtp requires HIPFIRE_MTP_INCREMENTAL=0 so the singleton reference window runs the batched \
                 route at depth k"
                    .into(),
            );
        }
        Some(LaneSpecConfig {
            k: args.k,
            end_of_turn: None,
        })
    } else {
        None
    };
    let max_k = required_lanes(&args);
    let lane_capacity = resolve_lane_capacity(args.lane_capacity, max_k)?;
    let max_ctx = *args.contexts.iter().max().unwrap();
    let per_window = if args.spec_mtp { args.k + 1 } else { 1 };
    let longest_run = args.steps.max(args.bench + 3) * per_window;
    if max_ctx + longest_run + 2 > args.max_seq {
        return Err(format!(
            "--max-seq {} cannot hold the longest context {max_ctx} + {longest_run} decode steps",
            args.max_seq
        ));
    }

    let mut log = Log::default();
    let file_size = fs::metadata(&args.model).map_err(e)?.len();
    // `--spec ar --promote` keeps the MTP head resident beside AR lanes, as serve does with `serve.batch_spec = false`.
    let attach_mtp = args.spec_mtp || args.promote.is_some();
    let (_source, mut w, tokenizer) = load(
        &args.model,
        args.max_seq,
        lane_capacity,
        spec_cfg,
        attach_mtp,
    )?;
    log.say(format!("gpu arch: {}", w.gpu.arch));
    log.say(format!(
        "model: {} ({file_size} bytes)",
        args.model.display()
    ));
    {
        let store = w.bundle.lanes().ok_or("lane store is not staged")?;
        let r = store.receipt(&w.gpu);
        log.say(format!(
            "lanes: kv_backend={} qsa={} gdn={} ring_rows={} max_lanes={} row_budget={} mapped_bytes={} stage_policy={} evidence={}",
            r.kv_backend, r.qsa_format, r.gdn_format, r.ring_rows, r.max_lanes, r.row_budget, r.mapped_bytes,
            r.stage_policy, r.stage_evidence
        ));
        if args.spec_mtp {
            log.say(format!("lanes spec: spec={} spec_k={}", r.spec, r.spec_k));
        }
    }
    log.say(format!(
        "args: ks={:?} contexts={:?} steps={} max_seq={} lane_capacity={} policy={} ({}) g0={} controls={} bench={}",
        args.ks, args.contexts, args.steps, args.max_seq, lane_capacity, args.policy, primary.describe(), args.g0,
        args.controls, args.bench
    ));
    if args.spec_mtp {
        log.say(format!(
            "spec args: spec=mtp k={} routes={} HIPFIRE_MTP_INCREMENTAL=0",
            args.k,
            args.routes.name()
        ));
    }
    if !primary.evidence().is_empty() {
        log.say(format!("policy evidence: {}", primary.evidence()));
    }

    let text = fs::read_to_string(&args.text).map_err(e)?;
    let toks = tokenizer.encode(&text);
    if toks.is_empty() {
        return Err("TEXT tokenized to nothing".into());
    }
    if toks.len() < max_ctx {
        log.say(format!(
            "warning: TEXT has {} tokens, fewer than the longest context {max_ctx}; prompts wrap",
            toks.len()
        ));
    }
    if args.spec_mtp {
        return spec_main(
            &args,
            w,
            log,
            toks,
            max_k,
            lane_capacity,
            file_size,
            primary,
        );
    }

    // Checkpoints: prefill, step 1, the shortened g0/control run, the last step.
    let short_steps = args.steps.min(16);
    let mut ckpt_steps = vec![1, args.steps];
    if args.g0 || args.controls {
        ckpt_steps.push(short_steps);
    }
    ckpt_steps.sort_unstable();
    ckpt_steps.dedup();

    // ── singleton references ───────────────────────────────────────
    let mut refs = Vec::with_capacity(max_k);
    for i in 0..max_k {
        let ctx = args.contexts[i % args.contexts.len()];
        let prompt = prompt_for(&toks, i, ctx);
        let dir = args.out.join("ref").join(format!("req{i}"));
        let t = Instant::now();
        let rq = record_ref(&mut w, i, prompt, args.steps, &ckpt_steps, &dir)?;
        log.say(format!(
            "reference req{i}: context {ctx}, {} steps, ids[0..4]={:?} ({:.1}s)",
            args.steps,
            &rq.ids[..rq.ids.len().min(4)],
            t.elapsed().as_secs_f64()
        ));
        refs.push(rq);
    }

    let prompts: Vec<Vec<u32>> = refs.iter().map(|r| r.prompt.clone()).collect();
    let mut app = App {
        w,
        refs,
        steps: args.steps,
        ckpt_steps,
        tag: 0,
        log,
        prompts,
        max_seq: args.max_seq,
        lanes_dir: args.out.join("lanes"),
        ar_ids: Vec::new(),
    };
    let full = RunOpts { prefill_ckpt: true };
    let mut all_pass = true;
    let mut runs: Vec<Value> = Vec::new();
    let mut s1: Option<bool> = None;

    // ── main matrix ────────────────────────────────────────────────
    for &k in &args.ks {
        let r = app.run_lanes(k, args.steps, primary, &args.policy, Control::None, &full);
        app.say_run(&r);
        all_pass &= r.pass;
        if k == 2 && s1.is_none() {
            s1 = Some(r.prefill_pass);
        }
        runs.push(r.json);
    }
    if s1.is_none() {
        let steps = app.steps.min(1);
        let r = app.run_lanes(2, steps, primary, &args.policy, Control::None, &full);
        s1 = Some(r.prefill_pass);
        all_pass &= r.pass;
        runs.push(r.json);
    }
    let s1_pass = s1.unwrap_or(false);
    all_pass &= s1_pass;
    app.log.say(format!(
        "S1 prefill parity k=2: {}",
        if s1_pass { "PASS" } else { "FAIL" }
    ));

    // ── G0 per-stage probe ─────────────────────────────────────────
    let mut g0_json = Value::Null;
    if args.g0 {
        let probe = RunOpts {
            prefill_ckpt: false,
        };
        let mut verdicts = Vec::new();
        let mut combined = LaneStagePolicy::all_per_lane();
        let mut identical_stages = Vec::new();
        for stage in StageName::ALL {
            if stage.stateful() {
                app.log.say(format!(
                    "g0 stage {}: per lane by construction (stateful)",
                    stage.key()
                ));
                verdicts
                    .push(json!({"stage": stage.key(), "verdict": "stateful", "identical": false}));
                continue;
            }
            let policy = match LaneStagePolicy::all_per_lane().with(stage, StageShare::OnceAllRows)
            {
                Ok(p) => p,
                Err(m) => {
                    app.log
                        .say(format!("g0 stage {}: policy refused: {m}", stage.key()));
                    verdicts.push(json!({"stage": stage.key(), "verdict": format!("refused: {m}"), "identical": false}));
                    continue;
                }
            };
            let mut identical = true;
            let mut first: Option<String> = None;
            let mut per_k = Vec::new();
            for k in [2usize, 8] {
                let label = format!("g0:{}=once", stage.key());
                let r = app.run_lanes(k, short_steps, policy, &label, Control::None, &probe);
                if !r.pass {
                    identical = false;
                    first.get_or_insert_with(|| {
                        format!("k={k}: {}", r.first.clone().unwrap_or_default())
                    });
                }
                per_k.push(r.json);
            }
            if identical {
                app.log.say(format!("g0 stage {}: identical", stage.key()));
                identical_stages.push(stage);
                combined = combined.with(stage, StageShare::OnceAllRows)?;
            } else {
                app.log.say(format!(
                    "g0 stage {}: differs (first divergence: {})",
                    stage.key(),
                    first.clone().unwrap_or_default()
                ));
            }
            verdicts.push(json!({
                "stage": stage.key(),
                "verdict": if identical { "identical" } else { "differs" },
                "identical": identical,
                "first_divergence": first,
                "runs": per_k,
            }));
        }
        app.log
            .say(format!("g0 combined policy: {}", combined.describe()));
        app.log.say("g0 combined STAGE_SHARE:".to_string());
        let mut table = Vec::new();
        for stage in StageName::ALL {
            let line = stage_rust_line(stage, combined.share(stage));
            app.log.say(line.clone());
            table.push(line);
        }
        let mut combined_runs = Vec::new();
        for &k in &args.ks {
            let r = app.run_lanes(k, args.steps, combined, "g0:combined", Control::None, &full);
            app.say_run(&r);
            all_pass &= r.pass;
            combined_runs.push(r.json);
        }
        g0_json = json!({
            "stages": verdicts,
            "identical_stages": identical_stages.iter().map(|s| s.key()).collect::<Vec<_>>(),
            "combined_policy": combined.describe(),
            "combined_stage_share": table,
            "combined_runs": combined_runs,
        });
    }

    // ── negative controls ──────────────────────────────────────────
    let mut controls_json = Vec::new();
    if args.controls {
        if short_steps < 4 {
            return Err("--controls needs --steps >= 4".into());
        }
        let probe = RunOpts { prefill_ckpt: true };
        for control in [
            Control::SwapTokens,
            Control::SwapRowSlot,
            Control::SwapRows,
            Control::WrongPosition,
        ] {
            let r = app.run_lanes(2, short_steps, primary, &args.policy, control, &probe);
            let refused = r.run.refused.clone();
            let diverged = r.run.diff.bad > 0 || r.run.error.is_some();
            let (detected, how) = if !r.run.fired {
                (
                    false,
                    "control was not exercised (no eligible step)".to_string(),
                )
            } else if let Some(m) = &refused {
                (true, format!("refused at provision: {m}"))
            } else if diverged {
                (
                    true,
                    format!("comparison failed: {}", r.first.clone().unwrap_or_default()),
                )
            } else {
                (false, "comparison still passed".to_string())
            };
            // A refused plan must leave every lane byte-exact for the rest of the run.
            let clean_after_refusal = refused.is_none() || r.pass;
            let ok = detected && clean_after_refusal;
            all_pass &= ok;
            let mut line = format!(
                "control {}: {}",
                control.name(),
                if detected { "detected" } else { "MISSED" }
            );
            line += &format!(" ({how})");
            if !clean_after_refusal {
                line += &format!(
                    "; FAIL: the refusal left lane state modified: {}",
                    r.first.clone().unwrap_or_default()
                );
            }
            app.log.say(line);
            controls_json.push(json!({
                "control": control.name(),
                "detected": detected,
                "how": how,
                "refused_at_provision": refused,
                "clean_after_refusal": clean_after_refusal,
                "note": r.run.control_note,
                "run": r.json,
            }));
        }
        app.log.say(
            "control lane_swap (device level): no test-visible path exchanges two lanes' Qwen4State; \
             the plan-level swaps above are the lane-swap controls"
                .to_string(),
        );
    }

    // ── promotion: native-MTP singleton → AR lane ──────────────────
    let mut promote_json = Vec::new();
    if let Some(windows) = args.promote {
        let promote = ArPromote {
            windows,
            k: args.k,
            skip_commit: false,
        };
        let label = format!("promote:{windows}@k{}", args.k);
        for &k in &args.ks {
            let r = app.run_lanes_with(
                k,
                args.steps,
                primary,
                &label,
                Control::None,
                &full,
                Some(promote),
            );
            app.say_run(&r);
            all_pass &= r.pass;
            promote_json.push(r.json);
        }
        if args.controls {
            // Negative: the lane keeps its fresh trunk state (no commit); the comparison must fail.
            let control = ArPromote {
                windows: 0,
                k: args.k,
                skip_commit: true,
            };
            let r = app.run_lanes_with(
                2,
                short_steps,
                primary,
                "promote:skip_commit",
                Control::None,
                &full,
                Some(control),
            );
            let detected = r.run.diff.bad > 0 || r.run.error.is_some();
            all_pass &= detected;
            let how = if detected {
                format!("comparison failed: {}", r.first.clone().unwrap_or_default())
            } else {
                "comparison still passed".to_string()
            };
            app.log.say(format!(
                "control promote_skip_commit: {} ({how})",
                if detected { "detected" } else { "MISSED" }
            ));
            controls_json.push(json!({
                "control": "promote_skip_commit",
                "detected": detected,
                "how": how,
                "run": r.json,
            }));
        }
    }

    // ── bench ──────────────────────────────────────────────────────
    let mut bench_json = Vec::new();
    if args.bench > 0 {
        let n = args.bench;
        let singleton_ms = app.bench_singleton(n)?;
        app.log.say(format!(
            "bench singleton: {singleton_ms:.3} ms/step (median of {n}, request 0)"
        ));
        for &k in &args.ks {
            let step_ms = app.bench_lanes(k, n)?;
            let tok_s = k as f64 * 1000.0 / step_ms;
            let speedup = tok_s / (1000.0 / singleton_ms);
            app.log.say(format!(
                "bench k={k}, step_ms={step_ms:.3}, tok_s={tok_s:.2}, singleton_ms={singleton_ms:.3}, speedup={speedup:.3}x"
            ));
            bench_json.push(json!({
                "k": k, "step_ms": step_ms, "tok_s": tok_s, "singleton_ms": singleton_ms, "speedup": speedup
            }));
        }
    }

    app.log.say(format!(
        "fn_lane_oracle: {}",
        if all_pass { "PASS" } else { "FAIL" }
    ));
    let summary = json!({
        "pass": all_pass,
        "gpu_arch": app.w.gpu.arch,
        "model": args.model.display().to_string(),
        "model_bytes": file_size,
        "args": {
            "ks": args.ks, "contexts": args.contexts, "steps": args.steps, "max_seq": args.max_seq,
            "lane_capacity": lane_capacity,
            "policy": args.policy, "policy_described": primary.describe(), "g0": args.g0,
            "controls": args.controls, "bench": args.bench, "spec": "ar",
            "promote": args.promote, "k": args.promote.map(|_| args.k),
        },
        "policy_evidence": primary.evidence(),
        "requests": app.refs.iter().map(|r| json!({"context": r.prompt.len(), "dir": r.dir.display().to_string()})).collect::<Vec<_>>(),
        "s1_prefill_parity_k2": s1_pass,
        "runs": runs,
        "promote_runs": promote_json,
        "g0": g0_json,
        "controls": controls_json,
        "bench": bench_json,
        "log": app.log.lines,
    });
    fs::write(
        args.out.join("summary.json"),
        serde_json::to_vec_pretty(&summary).map_err(e)?,
    )
    .map_err(e)?;
    Ok(all_pass)
}

#[cfg(test)]
mod ple_rows_tests {
    use super::*;
    use hipfire_config::memory::{ple_sidecar_path, PleFormat, PleFormatRequest};
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
    use hipfire_runtime::ple_sidecar::ple_shard_name;

    /// Removes its temp directory on drop, pass or fail.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("fn_lane_oracle_ple_{}_{tag}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// A monolithic PLE base: three embedded q8 shard tensors (4 rows x 64).
    fn write_monolithic(path: &Path) {
        let tensors: Vec<HfqMemTensor> = (0..3)
            .map(|s| HfqMemTensor {
                name: ple_shard_name(s),
                quant_type: 3,
                shape: vec![4, 64],
                group_size: 32,
                data: vec![0; 4 * (64 / 32 * 34)],
            })
            .collect();
        write_hfqm_package_mem(path, 16, "{}", &tensors).unwrap();
    }

    fn resolve(
        path: &Path,
        request: PleFormatRequest,
        arch: &str,
    ) -> (R<Option<PleRowsChoice>>, bool) {
        let mut hfq = HfqFile::open(path).unwrap();
        let out = resolve_ple_rows(&mut hfq, path, request, arch);
        (out, hfq.ple_rows().is_some())
    }

    #[test]
    fn explicit_q4_without_its_sidecar_refuses_and_never_derives() {
        let dir = Scratch::new("missing");
        let base = dir.0.join("m.hfqm");
        write_monolithic(&base);
        let (out, attached) = resolve(&base, PleFormatRequest::Q4, "gfx1100");
        let err = out.unwrap_err();
        assert!(err.starts_with("qwen4 PLE rows:"), "{err}");
        assert!(
            err.contains("missing and derivation was not requested"),
            "{err}"
        );
        assert!(!attached);
        assert!(
            !ple_sidecar_path(&base, PleFormat::Q4).exists(),
            "the oracle must not derive"
        );
    }

    #[test]
    fn a_sidecar_that_does_not_open_and_bind_is_refused_not_replaced_by_embedded_rows() {
        let dir = Scratch::new("damaged");
        let base = dir.0.join("m.hfqm");
        write_monolithic(&base);
        fs::write(ple_sidecar_path(&base, PleFormat::Q4), b"not a ple table").unwrap();
        // An explicit q4 request, and `auto` on the q4-preferring gfx1151.
        for (request, arch) in [
            (PleFormatRequest::Q4, "gfx1100"),
            (PleFormatRequest::Auto, "gfx1151"),
        ] {
            let (out, attached) = resolve(&base, request, arch);
            let err = out.unwrap_err();
            assert!(err.starts_with("qwen4 PLE rows:"), "{err}");
            assert!(!attached, "{request:?} on {arch}");
        }
    }

    #[test]
    fn embedded_rows_default_is_unchanged() {
        let dir = Scratch::new("embedded");
        let base = dir.0.join("m.hfqm");
        write_monolithic(&base);
        // The q4 sidecar is not consulted when q8 is wanted, even a bad one.
        fs::write(ple_sidecar_path(&base, PleFormat::Q4), b"not a ple table").unwrap();
        for request in [PleFormatRequest::Auto, PleFormatRequest::Q8] {
            let (out, attached) = resolve(&base, request, "gfx1100");
            let choice = out.unwrap().expect("a PLE base resolves a choice");
            assert_eq!(
                (choice.format, &choice.origin),
                (PleFormat::Q8, &PleRowsOrigin::Embedded)
            );
            assert!(!choice.derived && choice.note.is_none() && !attached);
        }
        // `auto` on gfx1151 with no q4 sidecar falls back to the embedded rows, with a note.
        fs::remove_file(ple_sidecar_path(&base, PleFormat::Q4)).unwrap();
        let (out, attached) = resolve(&base, PleFormatRequest::Auto, "gfx1151");
        let choice = out.unwrap().unwrap();
        assert_eq!(
            (choice.format, &choice.origin),
            (PleFormat::Q8, &PleRowsOrigin::Embedded)
        );
        assert!(choice.note.is_some() && !attached);
    }

    #[test]
    fn a_base_without_the_ple_layout_resolves_nothing() {
        let dir = Scratch::new("plain");
        let base = dir.0.join("plain.hfqm");
        write_hfqm_package_mem(&base, 16, "{}", &[]).unwrap();
        for request in [
            PleFormatRequest::Auto,
            PleFormatRequest::Q4,
            PleFormatRequest::Q8,
        ] {
            let (out, attached) = resolve(&base, request, "gfx1151");
            assert_eq!(out.unwrap(), None);
            assert!(!attached);
        }
    }
}

#[cfg(test)]
mod lane_capacity_tests {
    use super::*;

    fn parse(extra: &[&str]) -> R<Args> {
        let mut raw: Vec<String> = ["m.mq4", "t.txt", "out"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        raw.extend(extra.iter().map(|s| s.to_string()));
        parse_args_from(&raw)
    }

    #[test]
    fn flag_parses_and_defaults_to_none() {
        assert_eq!(parse(&[]).unwrap().lane_capacity, None);
        assert_eq!(
            parse(&["--lane-capacity", "8"]).unwrap().lane_capacity,
            Some(8)
        );
        assert!(parse(&["--lane-capacity"])
            .err()
            .expect("missing value must fail")
            .contains("needs a value"));
        assert!(parse(&["--lane-capacity", "x"]).is_err());
        assert!(parse(&["--lane-capacity", "0"])
            .err()
            .expect("zero capacity must fail")
            .contains(">= 1"));
    }

    #[test]
    fn default_capacity_is_the_required_lanes() {
        // Full run: --ks 1,2,4,8 stages 8; the default smoke (--ks 2) stages 2.
        let full = parse(&[]).unwrap();
        assert_eq!(
            resolve_lane_capacity(full.lane_capacity, required_lanes(&full)).unwrap(),
            8
        );
        let smoke = parse(&["--ks", "2"]).unwrap();
        assert_eq!(
            resolve_lane_capacity(smoke.lane_capacity, required_lanes(&smoke)).unwrap(),
            2
        );
        let g0 = parse(&["--ks", "2", "--g0"]).unwrap();
        assert_eq!(required_lanes(&g0), 8);
    }

    #[test]
    fn capacity_bounds_are_refused() {
        assert_eq!(resolve_lane_capacity(Some(8), 2).unwrap(), 8);
        assert_eq!(resolve_lane_capacity(Some(2), 2).unwrap(), 2);
        assert_eq!(
            resolve_lane_capacity(Some(LANE_STORE_MAX_LANES), 2).unwrap(),
            LANE_STORE_MAX_LANES
        );
        assert!(resolve_lane_capacity(Some(1), 2)
            .unwrap_err()
            .contains("below"));
        assert!(resolve_lane_capacity(Some(LANE_STORE_MAX_LANES + 1), 2)
            .unwrap_err()
            .contains("exceeds"));
        // Below the g0 requirement of 8 lanes.
        let g0 = parse(&["--ks", "2", "--g0", "--lane-capacity", "4"]).unwrap();
        assert!(resolve_lane_capacity(g0.lane_capacity, required_lanes(&g0)).is_err());
    }

    #[test]
    fn selected_k1_with_capacity_8_keeps_two_references_and_ks() {
        let a = parse(&["--ks", "1", "--controls", "--lane-capacity", "8"]).unwrap();
        assert_eq!(a.ks, vec![1]);
        // The singleton references are generated for `required_lanes` (2: k=1 plus the controls), not for the capacity.
        let required = required_lanes(&a);
        assert_eq!(required, 2);
        assert_eq!(resolve_lane_capacity(a.lane_capacity, required).unwrap(), 8);
        assert_eq!(a.ks, vec![1]);
        let k4 = parse(&["--ks", "4", "--lane-capacity", "8"]).unwrap();
        assert_eq!((k4.ks.clone(), required_lanes(&k4)), (vec![4], 4));
    }
}
