// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! SPEC-AR-FLOOR S2a gate 3 (as amended in SPEC-AR-FLOOR-VERDICT §7.8): a
//! qwen35 DFlash request the driver-owned floor moves between speculative
//! windows, AR tokens and forced advances.
//!
//! A verify window never writes the same bits as one-token AR steps, so state
//! is not compared against pure AR after a spec window. The gate is:
//!
//! - **(a)** candidate vs incumbent binary on the same scripted schedule:
//!   identical token ids and identical digests on every surface at every
//!   checkpoint.
//! - **(b)** at each spec→AR boundary the AR steps equal pure AR started from
//!   that boundary's state: the arm `prefix ++ AR×n ++ AR to the end` (the
//!   request retiring at the boundary) gives the same ids over the segment
//!   and the same digests at its end as the schedule. A terminal AR segment
//!   is its own retire arm, so only AR segments followed by more work are
//!   checked.
//! - **(c)** sabotage control: one AR token fed a wrong seed is detected.
//! - **(d)** AR→spec oracle: the schedule with every scripted `AR ×n`
//!   replaced by the same `n` tokens committed by forced advance (one token
//!   per call, the `ar_step` contract: rows committed exactly as a forced
//!   advance does) gives the same ids and digests on every surface at every
//!   checkpoint, through the following spec windows.
//! - **(e)** ids against pure AR on the four committed prompts, ≥256 decode
//!   tokens. A mismatch the incumbent reproduces identically (same index,
//!   same ids, same schedule) is pre-existing verify numerics: reported as a
//!   ticket, not a failure.
//!
//! Every arm runs in its own process (a fresh speculator and slot; no ring
//! cursor or prefix state is inherited) and writes a trace; a CPU verdict
//! test compares the traces.
//!
//! Schedules (op grammar: `sN` N spec windows, `aN` N AR tokens, `fN` N AR
//! tokens of the oracle source committed by forced advance, `F` a think-close
//! forced advance (two tokens), `S` / `A` spec windows / AR tokens until the
//! decode reaches `SPEC_FLOOR_HW_TOKENS`, default 256):
//!
//! - `s3,a6,S`: an AR segment between windows (spec ×3 → AR ×6 → spec);
//! - `s8,A`: a retirement at the 8-window checkpoint, no re-entry inside the
//!   request;
//! - `s8,a200,s3,A`: a retirement, a re-entry epoch after the AR gap
//!   (shortened to 200 to fit the decode) and a second retirement at its
//!   3-window checkpoint;
//! - `s3,a6,s1,a20,s6,F,s2,A` and its reverse `F,a2,s2,a20,s1,a6,s3,S`;
//! - `A`: pure AR; `R` (one per schedule with an `F`): pure AR with that
//!   schedule's think-close forced advances at the same decode indices, the
//!   (e) reference of every arm of that schedule (a forced close changes the
//!   text, so plain pure AR is no reference there).
//!
//! Driving one arm (GPU, `#[ignore]`d):
//!
//! ```text
//! SPEC_FLOOR_HW_PROMPT=benchmarks/prompts/prose_river_short.txt \
//! SPEC_FLOOR_HW_OPS=s3,a6,S SPEC_FLOOR_HW_OUT=<dir>/<arm>.trace \
//! cargo test --release -p hipfire-generate --test spec_floor_schedule_hw -- \
//!   --ignored --exact qwen35_dflash_floor_schedule_arm
//! ```
//!
//! - `SPEC_FLOOR_HW_TARGET` / `SPEC_FLOOR_HW_DRAFT`: model file names under
//!   `~/.hipfire/models` or absolute paths (default `qwen3.8-27b.mq4-xts` +
//!   `qwen3.8-27b-dflash.mq4v2-sym.hfq`).
//! - `SPEC_FLOOR_HW_ROUTE=pm4|hip` (default `pm4`): the load's AR executor.
//! - `SPEC_FLOOR_HW_ORACLE`: the trace of the schedule an `f` arm replays.
//! - `SPEC_FLOOR_HW_SABOTAGE=N`: feed AR token number N a wrong seed.
//! - `SPEC_FLOOR_HW_CALIBRATE`: before the schedule, `1` runs the load-time
//!   floor calibration exactly as the daemon does
//!   (`hipfire_generate::spec_floor_load::prepare_spec_floor`: calibrate →
//!   reset); `skip-drafter` / `skip-slot` run the calibration and the reset
//!   sequence minus the drafter's load reset (`Speculator::reset_to_load`
//!   replaced by `reset`) or minus the target slot reset (gate 22's
//!   sabotage arms).
//!
//! `spec_floor_schedule_arms` lists every arm (`ARM <ops> <oracle ops|->`,
//! oracle sources first); `spec_floor_schedule_verdict` reads
//! `SPEC_FLOOR_HW_TRACES/{cand,inc}/<prompt>/<arm>.trace` and applies (a)–(e).
//!
//! **Gate 22, calibration neutrality (§7.11).** The calibrated arm of a
//! schedule (`SPEC_FLOOR_HW_CALIBRATE=1`) must give the identical ids and
//! identical digests on every surface at every checkpoint as the same
//! schedule on a fresh load, and a sabotage arm that skips one reset step
//! must differ. `spec_floor_calibration_arms` lists the arms (`CAL <ops>`,
//! `CALSABOTAGE <ops> <mode>`); `spec_floor_calibration_verdict` reads
//! `SPEC_FLOOR_HW_TRACES/{fresh,calibrated}/<prompt>/<arm>.trace` (sabotage
//! traces `calibrated/<prompt>/<arm>+<mode>.trace`).

use hipfire_arch_qwen35::dflash_spec::DflashSpeculator;
use hipfire_arch_qwen35::speculative::ModelSlot;
use hipfire_runtime::loader_api::SpecLoadCfg;
use hipfire_runtime::spec::{FloorControl, PrefillOutcome, SpecTarget, Speculator};
use rdna_compute::GpuTensor;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The floor's schedules and the forced-advance interleavings, pure AR last.
const SCHEDULES: [&str; 6] = [
    "s3,a6,S",
    "s8,A",
    "s8,a200,s3,A",
    "s3,a6,s1,a20,s6,F,s2,A",
    "F,a2,s2,a20,s1,a6,s3,S",
    "A",
];
/// The sabotage arm: the AR-segment schedule, AR token 2 fed a wrong seed.
const SABOTAGE: (&str, usize) = ("s3,a6,S", 2);
/// Gate 22's schedules: an AR segment between windows, a retirement with a
/// re-entry, the forced-advance interleaving, and pure AR.
const CALIBRATION_SCHEDULES: [&str; 4] = ["s3,a6,S", "s8,a200,s3,A", "F,a2,s2,a20,s1,a6,s3,S", "A"];
/// Gate 22's sabotage arms: the first schedule with one reset step skipped.
const CALIBRATION_SABOTAGE: (&str, [&str; 2]) = ("s3,a6,S", ["skip-drafter", "skip-slot"]);
const PROMPTS: usize = 4;
const DEFAULT_TOKENS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    Spec(usize),
    Ar(usize),
    ForcedAr(usize),
    Close,
    SpecUntil,
    ArUntil,
    /// Pure AR to the end, with the oracle source's think-close forced
    /// advances at the same decode indices: the (e) reference of a schedule
    /// with an `F`.
    Mirror,
}

fn parse_ops(ops: &str) -> Vec<Op> {
    ops.split(',')
        .map(|op| {
            let n = || {
                op[1..]
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("op {op}"))
            };
            match &op[..1] {
                "s" => Op::Spec(n()),
                "a" => Op::Ar(n()),
                "f" => Op::ForcedAr(n()),
                "F" if op.len() == 1 => Op::Close,
                "S" if op.len() == 1 => Op::SpecUntil,
                "A" if op.len() == 1 => Op::ArUntil,
                "R" if op.len() == 1 => Op::Mirror,
                _ => panic!("op {op}"),
            }
        })
        .collect()
}

/// (b): for every scripted AR segment that directly follows spec windows and
/// is followed by more ops, the arm retiring there.
fn boundary_arms(ops: &str) -> Vec<(String, usize)> {
    let parts: Vec<&str> = ops.split(',').collect();
    let parsed = parse_ops(ops);
    (1..parsed.len().saturating_sub(1))
        .filter(|&i| {
            matches!(parsed[i], Op::Ar(_)) && matches!(parsed[i - 1], Op::Spec(_) | Op::SpecUntil)
        })
        .map(|i| (format!("{},A", parts[..=i].join(",")), i))
        .collect()
}

/// (d): the schedule with every scripted `aN` committed by forced advance.
fn oracle_arm(ops: &str) -> Option<String> {
    if !ops.split(',').any(|op| op.starts_with('a')) {
        return None;
    }
    Some(
        ops.split(',')
            .map(|op| match op.strip_prefix('a') {
                Some(n) => format!("f{n}"),
                None => op.to_owned(),
            })
            .collect::<Vec<_>>()
            .join(","),
    )
}

/// Every arm, oracle sources before the arms replaying them: (ops, source).
fn arms() -> Vec<(String, Option<String>)> {
    let mut out: Vec<(String, Option<String>)> = Vec::new();
    let mut push = |ops: String, src: Option<String>| {
        if !out.iter().any(|(o, s)| *o == ops && *s == src) {
            out.push((ops, src));
        }
    };
    for s in SCHEDULES {
        push(s.to_owned(), None);
        for (b, _) in boundary_arms(s) {
            push(b, None);
        }
    }
    for s in SCHEDULES {
        if let Some(f) = oracle_arm(s) {
            push(f, Some(s.to_owned()));
        }
        if s.split(',').any(|op| op == "F") {
            push("R".to_owned(), Some(s.to_owned()));
        }
    }
    out
}

/// The schedule with an `F` whose mirrored pure AR (`R`) is the (e)
/// reference of `ops` (the schedule itself, a boundary arm or its oracle arm).
fn close_family(ops: &str) -> Option<&'static str> {
    if !ops.split(',').any(|op| op == "F") {
        return None;
    }
    SCHEDULES.into_iter().find(|s| {
        *s == ops
            || boundary_arms(s).iter().any(|(b, _)| b == ops)
            || oracle_arm(s).as_deref() == Some(ops)
    })
}

/// Decode indices `i` where tokens `i+1, i+2` are the think-close pair (a
/// forced close committed after the token at `i`).
fn close_indices(tokens: &[u32], close: [u32; 2]) -> Vec<usize> {
    (0..tokens.len().saturating_sub(2))
        .filter(|&i| tokens[i + 1] == close[0] && tokens[i + 2] == close[1])
        .collect()
}

fn arm_file(ops: &str, src: Option<&str>, sabotage: Option<usize>) -> String {
    let base = match (ops, src) {
        ("R", Some(src)) => format!("R~{}", src.replace(',', "-")),
        _ => ops.replace(',', "-"),
    };
    match sabotage {
        Some(n) => format!("{base}+x{n}.trace"),
        None => format!("{base}.trace"),
    }
}

// ─── Trace ──────────────────────────────────────────────────────────────────

#[derive(Default, Debug)]
struct Trace {
    prompt_len: usize,
    /// Token committed at position `prompt_len + i` (the prefill's first token
    /// first).
    tokens: Vec<u32>,
    /// (op index, position) → surface → digest; op index `usize::MAX` is the
    /// prefill.
    checkpoints: Vec<(usize, usize, BTreeMap<String, String>)>,
}

impl Trace {
    fn write(&self, path: &Path, header: &str) {
        let mut s = format!("{header}\nprompt_len {}\n", self.prompt_len);
        s.push_str("tokens");
        for t in &self.tokens {
            s.push_str(&format!(" {t}"));
        }
        s.push('\n');
        for (op, pos, surfaces) in &self.checkpoints {
            let op = if *op == usize::MAX {
                "prefill".to_owned()
            } else {
                op.to_string()
            };
            for (k, v) in surfaces {
                s.push_str(&format!("ck {op} {pos} {k} {v}\n"));
            }
        }
        std::fs::write(path, s).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }

    fn read(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let mut t = Trace::default();
        let mut last: Option<(usize, usize)> = None;
        for line in text.lines() {
            let mut it = line.split(' ');
            match it.next() {
                Some("prompt_len") => t.prompt_len = it.next()?.parse().ok()?,
                Some("tokens") => t.tokens = it.map(|x| x.parse().unwrap()).collect(),
                Some("ck") => {
                    let op = match it.next()? {
                        "prefill" => usize::MAX,
                        n => n.parse().ok()?,
                    };
                    let pos: usize = it.next()?.parse().ok()?;
                    let key = it.next()?.to_owned();
                    let val = it.collect::<Vec<_>>().join(" ");
                    if last != Some((op, pos)) {
                        t.checkpoints.push((op, pos, BTreeMap::new()));
                        last = Some((op, pos));
                    }
                    t.checkpoints.last_mut()?.2.insert(key, val);
                }
                _ => {}
            }
        }
        Some(t)
    }

    fn checkpoint(&self, op: usize) -> Option<&(usize, usize, BTreeMap<String, String>)> {
        self.checkpoints.iter().find(|(o, _, _)| *o == op)
    }
}

// ─── GPU arm ────────────────────────────────────────────────────────────────

fn model(env: &str, default: &str) -> Option<String> {
    let name = std::env::var(env).unwrap_or_else(|_| default.to_owned());
    let path = if name.starts_with('/') {
        PathBuf::from(&name)
    } else {
        PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join(".hipfire/models")
            .join(&name)
    };
    if !path.is_file() {
        eprintln!("skip: {} missing", path.display());
        return None;
    }
    Some(path.to_string_lossy().into_owned())
}

fn special(tok: &hipfire_runtime::tokenizer::Tokenizer, s: &str) -> Vec<u32> {
    tok.special_token_id(s)
        .map_or_else(|| tok.encode(s), |id| vec![id])
}

/// The prompt in the chat frame with reasoning off (an empty think block).
fn chat_prompt(tok: &hipfire_runtime::tokenizer::Tokenizer, text: &str) -> Vec<u32> {
    let mut ids = special(tok, "<|im_start|>");
    ids.extend(tok.encode(&format!("user\n{}", text.trim_end())));
    ids.extend(special(tok, "<|im_end|>"));
    ids.extend(tok.encode("\n"));
    ids.extend(special(tok, "<|im_start|>"));
    ids.extend(tok.encode("assistant\n"));
    ids.extend(special(tok, "<think>"));
    ids.extend(tok.encode("\n\n"));
    ids.extend(special(tok, "</think>"));
    ids.extend(tok.encode("\n\n"));
    ids
}

fn h64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn download(gpu: &rdna_compute::Gpu, t: &GpuTensor) -> Vec<u8> {
    let mut bytes = vec![0u8; t.buf.size()];
    gpu.hip.memcpy_dtoh(&mut bytes, &t.buf).expect("dtoh");
    bytes
}

/// `rows` rows of `row_bytes` from a row-major buffer, row `r` at `index(r)`.
fn rows_digest(
    bytes: &[u8],
    row_bytes: usize,
    rows: impl Iterator<Item = usize>,
    index: impl Fn(usize) -> usize,
) -> u64 {
    let mut h = 0u64;
    for r in rows {
        let at = index(r) * row_bytes;
        h = h.rotate_left(7) ^ h64(&bytes[at..at + row_bytes]);
    }
    h
}

/// Every state surface at committed position `pos`.
fn digest(
    gpu: &rdna_compute::Gpu,
    slot: &ModelSlot,
    spec: &DflashSpeculator,
    pos: usize,
) -> BTreeMap<String, String> {
    gpu.hip.device_synchronize().expect("sync");
    let mut out = BTreeMap::new();
    let kv = &slot.kv_cache;
    let cap = kv.physical_cap.max(1);
    for (name, tensors) in [
        ("kv.k", &kv.k_gpu),
        ("kv.v", &kv.v_gpu),
        ("kv.k_scales", &kv.k_scales),
        ("kv.v_scales", &kv.v_scales),
    ] {
        for (layer, t) in tensors.iter().enumerate() {
            let size = t.buf.size();
            if size == 0 {
                continue;
            }
            let bytes = download(gpu, t);
            // Position-major rows: only the committed prefix is state (a
            // verify writes rejected-draft rows past it).
            let h = if size % cap == 0 {
                rows_digest(&bytes, size / cap, 0..pos.min(cap), |r| r)
            } else {
                h64(&bytes)
            };
            out.insert(format!("{name}[{layer:02}]"), format!("{h:016x}"));
        }
    }
    let dn = &slot.dn_state;
    for (name, tensors) in [
        ("dn.s", &dn.s_matrices),
        ("dn.scale", &dn.s_scales),
        ("dn.conv", &dn.conv_states),
        ("dn.ef", &dn.s_ef_residual),
    ] {
        for (layer, t) in tensors.iter().enumerate() {
            if t.buf.size() > 0 {
                out.insert(
                    format!("{name}[{layer:02}]"),
                    format!("{:016x}", h64(&download(gpu, t))),
                );
            }
        }
    }
    let df = spec.lane_state().expect("DFlash lane state");
    let rb = &df.hidden_rb;
    let live = rb.written.min(rb.max_positions);
    let row = rb.hidden_dim * 4;
    for (layer, t) in rb.layer_bufs.iter().enumerate() {
        let bytes = download(gpu, t);
        let h = rows_digest(&bytes, row, 0..live, |r| {
            (rb.head + rb.max_positions - live + r) % rb.max_positions
        });
        out.insert(format!("ring[{layer}]"), format!("{h:016x}"));
    }
    out.insert("ring.cursor".into(), format!("{}/{}", rb.head, rb.written));
    let feature = rb.hidden_dim * rb.extract_layers.len() * 4;
    let th = &df.draft_scratch.target_hidden;
    if feature > 0 && th.buf.size() % feature == 0 {
        let modulus = th.buf.size() / feature;
        let bytes = download(gpu, th);
        let from = pos.saturating_sub(modulus);
        let h = rows_digest(&bytes, feature, from..pos, |p| p % modulus);
        out.insert("interleaved".into(), format!("{h:016x}"));
    }
    out.insert(
        "thlog".into(),
        format!("{:?}", df.draft_scratch.thlog.mark()).replace(' ', ""),
    );
    out
}

struct Rig<'a> {
    gpu: &'a mut rdna_compute::Gpu,
    spec: &'a mut Box<dyn Speculator>,
    target: &'a mut dyn SpecTarget,
    prompt: Vec<u32>,
    close: [u32; 2],
}

impl Rig<'_> {
    fn digest(&mut self, pos: usize) -> BTreeMap<String, String> {
        let slot = self
            .target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .expect("qwen35 ModelSlot");
        let spec = self
            .spec
            .drafter_any_mut()
            .and_then(|any| any.downcast_ref::<DflashSpeculator>())
            .expect("DflashSpeculator");
        digest(self.gpu, slot, spec, pos)
    }

    fn forced(&mut self, pos: usize, tokens: &[u32]) {
        let handled = self
            .spec
            .on_forced_advance(self.gpu, self.target, tokens, pos, &|| false)
            .unwrap_or_else(|e| panic!("forced advance at {pos}: {e}"));
        assert!(
            handled,
            "the DFlash speculator advances forced tokens itself"
        );
    }

    /// One arm from prefill. `oracle`: the tokens an `f` op commits;
    /// `sabotage`: feed AR token number N a wrong seed.
    fn run(
        &mut self,
        ops: &[Op],
        until: usize,
        oracle: Option<&Trace>,
        sabotage: Option<usize>,
    ) -> Trace {
        let toks = self.prompt.clone();
        let first =
            match self
                .spec
                .prefill(self.gpu, self.target, &toks, &toks, 0, false, None, &|| {
                    false
                }) {
                Ok(PrefillOutcome::Ready { first_token }) => first_token,
                other => panic!("prefill: {other:?}"),
            };
        let start = toks.len();
        let end = start + until;
        let mut pos = start;
        let mut seed = first;
        let mut emitted = vec![first];
        let mut trace = Trace {
            prompt_len: start,
            tokens: vec![first],
            checkpoints: Vec::new(),
        };
        let d = self.digest(pos);
        trace.checkpoints.push((usize::MAX, pos, d));
        let mut ar_count = 0;
        for (index, &op) in ops.iter().enumerate() {
            let (windows, ar) = match op {
                Op::Spec(n) => (n, 0),
                Op::SpecUntil => (usize::MAX, 0),
                Op::Ar(n) => (0, n),
                Op::ArUntil => (0, usize::MAX),
                Op::ForcedAr(n) => {
                    let src = oracle.expect("an f op needs SPEC_FLOOR_HW_ORACLE");
                    for _ in 0..n {
                        let i = pos - start;
                        assert_eq!(
                            src.tokens.get(i),
                            Some(&seed),
                            "oracle source diverged before its AR segment (index {i})"
                        );
                        self.forced(pos, &[seed]);
                        pos += 1;
                        seed = src.tokens[i + 1];
                        emitted.push(seed);
                        trace.tokens.push(seed);
                    }
                    (0, 0)
                }
                Op::Close => {
                    self.forced(pos, &[seed, self.close[0]]);
                    pos += 2;
                    emitted.extend_from_slice(&self.close);
                    trace.tokens.extend_from_slice(&self.close);
                    seed = self.close[1];
                    (0, 0)
                }
                Op::Mirror => {
                    let src = oracle.expect("an R op needs SPEC_FLOOR_HW_ORACLE");
                    let closes = close_indices(&src.tokens, self.close);
                    while pos < end {
                        if closes.contains(&(pos - start)) {
                            self.forced(pos, &[seed, self.close[0]]);
                            pos += 2;
                            emitted.extend_from_slice(&self.close);
                            trace.tokens.extend_from_slice(&self.close);
                            seed = self.close[1];
                            continue;
                        }
                        let next = self
                            .spec
                            .ar_step(self.gpu, self.target, pos, seed, &emitted)
                            .unwrap_or_else(|e| panic!("ar_step at {pos}: {e}"));
                        pos += 1;
                        emitted.push(next);
                        trace.tokens.push(next);
                        seed = next;
                    }
                    (0, 0)
                }
            };
            let mut w = 0;
            while w < windows && (windows != usize::MAX || pos < end) {
                let step = self
                    .spec
                    .step(
                        self.gpu,
                        self.target,
                        pos,
                        seed,
                        &emitted,
                        None,
                        0.0,
                        usize::MAX,
                    )
                    .unwrap_or_else(|e| panic!("step at {pos}: {e}"));
                assert!(!step.emit.is_empty(), "verify emitted nothing at {pos}");
                pos += step.emit.len();
                emitted.extend_from_slice(&step.emit);
                trace.tokens.extend_from_slice(&step.emit);
                seed = step.next_seed;
                w += 1;
            }
            let mut a = 0;
            while a < ar && (ar != usize::MAX || pos < end) {
                ar_count += 1;
                let fed = if sabotage == Some(ar_count) {
                    seed ^ 1
                } else {
                    seed
                };
                let next = self
                    .spec
                    .ar_step(self.gpu, self.target, pos, fed, &emitted)
                    .unwrap_or_else(|e| panic!("ar_step at {pos}: {e}"));
                pos += 1;
                emitted.push(next);
                trace.tokens.push(next);
                seed = next;
                a += 1;
            }
            let d = self.digest(pos);
            trace.checkpoints.push((index, pos, d));
        }
        trace
    }
}

#[test]
#[ignore = "needs a HIP GPU and a qwen3.8-27b target with its DFlash draft (SPEC_FLOOR_HW_*)"]
fn qwen35_dflash_floor_schedule_arm() {
    let (Some(path), Some(draft)) = (
        model("SPEC_FLOOR_HW_TARGET", "qwen3.8-27b.mq4-xts"),
        model("SPEC_FLOOR_HW_DRAFT", "qwen3.8-27b-dflash.mq4v2-sym.hfq"),
    ) else {
        return;
    };
    let ops_text = std::env::var("SPEC_FLOOR_HW_OPS").expect("SPEC_FLOOR_HW_OPS");
    let ops = parse_ops(&ops_text);
    let prompt_path = std::env::var("SPEC_FLOOR_HW_PROMPT").expect("SPEC_FLOOR_HW_PROMPT");
    let out = PathBuf::from(std::env::var("SPEC_FLOOR_HW_OUT").expect("SPEC_FLOOR_HW_OUT"));
    let until = std::env::var("SPEC_FLOOR_HW_TOKENS")
        .ok()
        .map_or(DEFAULT_TOKENS, |v| v.parse().expect("SPEC_FLOOR_HW_TOKENS"));
    let sabotage = std::env::var("SPEC_FLOOR_HW_SABOTAGE")
        .ok()
        .map(|v| v.parse::<usize>().expect("SPEC_FLOOR_HW_SABOTAGE"));
    let oracle = std::env::var("SPEC_FLOOR_HW_ORACLE").ok().map(|p| {
        Trace::read(Path::new(&p)).unwrap_or_else(|| panic!("oracle trace {p} unreadable"))
    });
    assert_eq!(
        ops.iter()
            .any(|op| matches!(op, Op::ForcedAr(_) | Op::Mirror)),
        oracle.is_some(),
        "SPEC_FLOOR_HW_ORACLE is set exactly for an arm with an f op"
    );
    let text = std::fs::read_to_string(&prompt_path).expect("prompt file");
    let pm4 = std::env::var("SPEC_FLOOR_HW_ROUTE").map_or(true, |route| route != "hip");

    let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
    let spec_cfg = SpecLoadCfg {
        dflash: Some(true),
        mtp: Some(false),
        dspark: Some(false),
        ..SpecLoadCfg::default()
    };
    let mut m = hipfire_loader::load_model(
        &path,
        2048,
        Some(&draft),
        None,
        None,
        None,
        1,
        spec_cfg,
        &mut gpu,
    )
    .expect("load");
    let tok = m.tokenizer.as_ref().expect("tokenizer");
    let prompt = chat_prompt(tok, &text);
    let close_id = tok.special_token_id("</think>").expect("</think> token");
    let close = [close_id, tok.encode("\n\n")[0]];
    assert!(prompt.len() + until + 256 <= 2048, "prompt too long");
    let enabled = gpu.replay.configure_model_default(pm4);
    if pm4 && gpu.arch == "gfx1201" {
        gpu.replay
            .set_pm4_gfx12_dispatch_pacing(rdna_compute::replay::gfx1201_pm4_pacing_from_config());
    }
    let calibrate = std::env::var("SPEC_FLOOR_HW_CALIBRATE").ok();
    let calibrated = calibrate
        .as_deref()
        .map(|mode| calibrate_load(&mut m, &mut gpu, mode));
    let header = format!(
        "arm {ops_text} prompt {prompt_path} sabotage {sabotage:?} arch {} route {} retained {enabled} transport {} calibrate {} snapshot {}",
        gpu.arch,
        if pm4 { "pm4" } else { "hip" },
        gpu.replay.transport_name(),
        calibrate.as_deref().unwrap_or("-"),
        calibrated.as_deref().unwrap_or("-"),
    );
    eprintln!(
        "spec_floor_schedule_hw: {header} prompt_tokens={}",
        prompt.len()
    );
    let spec = m.speculator.as_mut().expect("speculator loaded");
    assert_eq!(spec.name(), "dflash", "speculator route");
    assert!(spec.supports_ar_step(), "DFlash AR step");
    let carrier = hipfire_loader::carrier_for(m.arch_id).expect("carrier");
    let mut guard = carrier
        .spec_target_guard(&mut m.state)
        .expect("spec target guard");
    let target = guard.slot().expect("spec target");
    let mut rig = Rig {
        gpu: &mut gpu,
        spec,
        target,
        prompt,
        close,
    };
    let trace = rig.run(&ops, until, oracle.as_ref(), sabotage);
    let replay = rig.gpu.replay.replay_observation();
    let header = format!(
        "{header} replay_count {} replay_failed {}",
        replay.count, replay.failed
    );
    eprintln!(
        "spec_floor_schedule_hw: {} tokens, {} checkpoints, {header}",
        trace.tokens.len(),
        trace.checkpoints.len()
    );
    assert!(!replay.failed, "replay failed");
    trace.write(&out, &header);
}

/// The target half of the reset sequence (KV cache and DeltaNet state
/// cleared, then synchronized), for gate 22's sabotage arms.
fn reset_slot(gpu: &mut rdna_compute::Gpu, slot: &mut ModelSlot) {
    slot.kv_cache.clear_gpu(gpu).expect("KV clear");
    slot.kv_cache.compact_offset = 0;
    slot.dn_state.reset(gpu).expect("DeltaNet reset");
    gpu.hip.device_synchronize().expect("sync");
}

/// Gate 22: the load-time floor calibration before the schedule. `1` runs
/// the daemon's own `prepare_spec_floor` (calibrate → reset); `skip-drafter`
/// and `skip-slot` run the calibration, then the reset sequence with one
/// step skipped. Returns the calibration's pin for the trace header.
fn calibrate_load(
    m: &mut hipfire_loader::LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    mode: &str,
) -> String {
    use hipfire_generate::spec_floor_load as load;
    let line = |prices: Option<hipfire_runtime::spec_prices::PinnedPrices>| {
        prices.map_or_else(
            || "none".to_owned(),
            |p| {
                let full = p.window(p.full_rows);
                format!(
                    "ar_us={},window_us={},b={:.4}",
                    p.ar_us,
                    full,
                    full / p.ar_us
                )
            },
        )
    };
    if mode == "1" {
        load::prepare_spec_floor(m, gpu).expect("calibration reset");
        let spec = m.speculator.as_mut().expect("speculator");
        let FloorControl::Driver(df) = spec.floor_control() else {
            panic!("the DFlash speculator owns no floor");
        };
        let (prices, source) = df.snapshot().expect("the calibration pinned");
        return format!("{},source={}", line(Some(prices)), source.name());
    }
    let prompt = m
        .tokenizer
        .as_ref()
        .expect("tokenizer")
        .encode(load::CALIBRATION_TEXT);
    let spec = m.speculator.as_mut().expect("speculator");
    let capacity = spec.ctx_capacity().min(m.physical_cap);
    let mut guard =
        hipfire_loader::spec_build::Qwen35SlotGuard::take(&mut m.state).expect("slot guard");
    let slot = guard.model_slot().expect("model slot");
    let calibration = load::run_spec_floor_calibration(gpu, spec.as_mut(), slot, &prompt, capacity)
        .expect("calibration");
    match mode {
        "skip-drafter" => {
            spec.reset(gpu).expect("reset");
            reset_slot(gpu, slot);
        }
        "skip-slot" => spec.reset_to_load(gpu).expect("reset_to_load"),
        other => panic!("SPEC_FLOOR_HW_CALIBRATE={other}"),
    }
    drop(guard);
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    line(calibration.snapshot)
}

// ─── CPU: arm list and verdict ──────────────────────────────────────────────

#[test]
#[ignore = "lists the gate-3 arms for the driver"]
fn spec_floor_schedule_arms() {
    for (ops, src) in arms() {
        println!("ARM {ops} {}", src.as_deref().unwrap_or("-"));
    }
    println!("SABOTAGE {} {}", SABOTAGE.0, SABOTAGE.1);
}

/// Surface class of a digest key, for the report.
fn class(key: &str) -> &str {
    key.split(['[', '.']).next().unwrap_or(key)
}

/// Token-id differences `(index, a, b)` over the common length.
fn id_diffs(a: &Trace, b: &Trace, range: std::ops::Range<usize>) -> Vec<(usize, u32, u32)> {
    range
        .filter_map(|i| match (a.tokens.get(i), b.tokens.get(i)) {
            (Some(x), Some(y)) if x != y => Some((i, *x, *y)),
            _ => None,
        })
        .collect()
}

/// Digest differences at op checkpoint `op` (or every checkpoint): one line
/// per surface, plus a per-class count.
fn surface_diffs(a: &Trace, b: &Trace, op: Option<usize>) -> Vec<String> {
    let mut out = Vec::new();
    for (o, pos, sa) in &a.checkpoints {
        if op.is_some_and(|x| x != *o) {
            continue;
        }
        let Some((_, pos_b, sb)) = b.checkpoint(*o) else {
            out.push(format!("op {o}: missing checkpoint"));
            continue;
        };
        if pos != pos_b {
            out.push(format!("op {o}: position {pos} vs {pos_b}"));
            continue;
        }
        for (k, va) in sa {
            match sb.get(k) {
                Some(vb) if vb == va => {}
                vb => out.push(format!("op {o} pos {pos} {k}: {va} vs {vb:?}")),
            }
        }
        for k in sb.keys().filter(|k| !sa.contains_key(*k)) {
            out.push(format!("op {o} pos {pos} {k}: missing"));
        }
    }
    out
}

fn summarize(label: &str, lines: &[String]) {
    if lines.is_empty() {
        return;
    }
    let mut classes: BTreeMap<String, usize> = BTreeMap::new();
    for l in lines {
        let key = l.split(' ').nth(4).unwrap_or("?").trim_end_matches(':');
        *classes.entry(class(key).to_owned()).or_default() += 1;
    }
    eprintln!(
        "{label}: {} surface mismatches by class {classes:?}",
        lines.len()
    );
    eprintln!("{label}: first: {}", lines[0]);
    for l in lines.iter().take(12) {
        eprintln!("    {l}");
    }
}

/// Decode position range `[from, to)` (token indices) of op `op` in `t`.
fn op_range(t: &Trace, op: usize) -> std::ops::Range<usize> {
    let before = if op == 0 {
        t.checkpoint(usize::MAX)
    } else {
        t.checkpoint(op - 1)
    };
    let from = before.map_or(0, |c| c.1 - t.prompt_len);
    let to = t.checkpoint(op).map_or(from, |c| c.1 - t.prompt_len);
    from..to
}

#[test]
#[ignore = "reads SPEC_FLOOR_HW_TRACES written by qwen35_dflash_floor_schedule_arm"]
fn spec_floor_schedule_verdict() {
    let root = PathBuf::from(std::env::var("SPEC_FLOOR_HW_TRACES").expect("SPEC_FLOOR_HW_TRACES"));
    let until = std::env::var("SPEC_FLOOR_HW_TOKENS")
        .ok()
        .map_or(DEFAULT_TOKENS, |v| v.parse().expect("SPEC_FLOOR_HW_TOKENS"));
    let mut prompts: Vec<String> = std::fs::read_dir(root.join("cand"))
        .expect("cand traces")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    prompts.sort();
    let load = |side: &str, prompt: &str, ops: &str, src: Option<&str>, sab: Option<usize>| {
        Trace::read(&root.join(side).join(prompt).join(arm_file(ops, src, sab)))
    };
    let mut failures = 0usize;
    let mut tickets = 0usize;
    let mut sabotage_seen = false;
    let mut sabotage_detected = true;
    for prompt in &prompts {
        eprintln!("== prompt {prompt}");
        let get = |side: &str, ops: &str, src: Option<&str>| {
            load(side, prompt, ops, src, None)
                .unwrap_or_else(|| panic!("missing {side}/{prompt}/{}", arm_file(ops, src, None)))
        };
        for (ops, src) in arms() {
            let c = get("cand", &ops, src.as_deref());
            let i = get("inc", &ops, src.as_deref());
            // Every arm reaches the decode length.
            if c.tokens.len() <= until || i.tokens.len() <= until {
                eprintln!(
                    "FAIL {prompt} {ops}: short decode {} / {}",
                    c.tokens.len(),
                    i.tokens.len()
                );
                failures += 1;
            }
            // (a) candidate vs incumbent: ids and every surface, every checkpoint.
            let ids = id_diffs(&c, &i, 0..c.tokens.len().max(i.tokens.len()));
            let surf = surface_diffs(&c, &i, None);
            let a_fail = !ids.is_empty() || !surf.is_empty() || c.tokens.len() != i.tokens.len();
            eprintln!(
                "(a) {prompt} {ops}: {} ids, {} surfaces differ cand vs inc -> {}",
                ids.len(),
                surf.len(),
                if a_fail { "FAIL" } else { "pass" }
            );
            if a_fail {
                if let Some(d) = ids.first() {
                    eprintln!(
                        "(a) {prompt} {ops}: first id diff at index {} cand {} inc {}",
                        d.0, d.1, d.2
                    );
                }
                summarize(&format!("(a) {prompt} {ops}"), &surf);
                failures += 1;
            }
            // (e) ids against pure AR (with the same think-close forced
            // advances for a schedule with an `F`), ticketed when the
            // incumbent reproduces the mismatch identically.
            let n = until + 1;
            // `A` and `R` are the references themselves.
            let (pure_c, pure_i) = match close_family(&ops) {
                _ if ops == "R" => (
                    get("cand", "R", src.as_deref()),
                    get("inc", "R", src.as_deref()),
                ),
                Some(family) => (
                    get("cand", "R", Some(family)),
                    get("inc", "R", Some(family)),
                ),
                None => (get("cand", "A", None), get("inc", "A", None)),
            };
            let ec = id_diffs(&c, &pure_c, 0..n);
            if !ec.is_empty() {
                let ei = id_diffs(&i, &pure_i, 0..n);
                if ec == ei {
                    eprintln!(
                        "(e) {prompt} {ops}: TICKET (pre-existing, incumbent identical): {} ids differ from pure AR, first {:?}",
                        ec.len(),
                        ec[0]
                    );
                    tickets += 1;
                } else {
                    eprintln!(
                        "(e) {prompt} {ops}: FAIL {} ids differ from pure AR (incumbent {}), first {:?}",
                        ec.len(),
                        ei.len(),
                        ec[0]
                    );
                    failures += 1;
                }
            } else {
                eprintln!("(e) {prompt} {ops}: ids equal pure AR over {n} tokens -> pass");
            }
            // (b) retire-at-boundary arms: ids over the segment, digests at its end.
            for (b_ops, op) in boundary_arms(&ops).into_iter().filter(|(b, _)| *b != ops) {
                for (side, sched) in [("cand", &c), ("inc", &i)] {
                    let b = get(side, &b_ops, None);
                    let range = op_range(sched, op);
                    let ids = id_diffs(sched, &b, 0..range.end + 1);
                    let surf = surface_diffs(sched, &b, Some(op));
                    let fail = !ids.is_empty() || !surf.is_empty();
                    eprintln!(
                        "(b) {side} {prompt} {ops} @op{op} tokens {range:?} vs {b_ops}: {} ids, {} surfaces -> {}",
                        ids.len(),
                        surf.len(),
                        if fail { "FAIL" } else { "pass" }
                    );
                    if fail && side == "cand" {
                        summarize(&format!("(b) {prompt} {ops} @op{op}"), &surf);
                        failures += 1;
                    }
                }
            }
            // (d) AR ×n vs forced advance ×n, through the following windows.
            if let Some(src) = src.as_deref().filter(|_| ops != "R") {
                for (side, f) in [("cand", &c), ("inc", &i)] {
                    let s = get(side, src, None);
                    let n = s.tokens.len().max(f.tokens.len());
                    let ids = id_diffs(f, &s, 0..n);
                    let surf = surface_diffs(f, &s, None);
                    let fail =
                        !ids.is_empty() || !surf.is_empty() || f.tokens.len() != s.tokens.len();
                    eprintln!(
                        "(d) {side} {prompt} {ops} vs {src}: {} ids, {} surfaces -> {}",
                        ids.len(),
                        surf.len(),
                        if fail { "FAIL" } else { "pass" }
                    );
                    if fail {
                        if let Some(d) = ids.first() {
                            eprintln!("(d) {side} {prompt} {ops}: first id diff at index {} forced {} ar {}", d.0, d.1, d.2);
                        }
                        summarize(&format!("(d) {side} {prompt} {ops}"), &surf);
                        if side == "cand" {
                            failures += 1;
                        }
                    }
                }
            }
        }
        // (c) sabotage.
        if let Some(sab) = load("cand", prompt, SABOTAGE.0, None, Some(SABOTAGE.1)) {
            sabotage_seen = true;
            let base = get("cand", SABOTAGE.0, None);
            let ids = id_diffs(&sab, &base, 0..sab.tokens.len().max(base.tokens.len()));
            let surf = surface_diffs(&sab, &base, None);
            eprintln!(
                "(c) {prompt} sabotage: {} ids, {} surfaces differ -> {}",
                ids.len(),
                surf.len(),
                if ids.is_empty() && surf.is_empty() {
                    "NOT DETECTED"
                } else {
                    "detected"
                }
            );
            sabotage_detected &= !ids.is_empty() || !surf.is_empty();
        }
    }
    eprintln!(
        "spec_floor_schedule_verdict: {} prompts, {failures} failures, {tickets} tickets, sabotage seen={sabotage_seen} detected={sabotage_detected}",
        prompts.len()
    );
    assert!(
        prompts.len() >= PROMPTS,
        "need the {PROMPTS} committed prompts"
    );
    assert!(
        sabotage_seen && sabotage_detected,
        "the sabotage control was not detected"
    );
    assert_eq!(failures, 0, "gate 3 mismatches");
}

#[test]
fn spec_floor_schedule_arm_catalog() {
    let all = arms();
    let names: Vec<&str> = all.iter().map(|(o, _)| o.as_str()).collect();
    assert!(names.contains(&"s3,a6,A"));
    assert!(names.contains(&"s8,a200,A"));
    assert!(names.contains(&"s3,a6,s1,a20,A"));
    assert!(names.contains(&"F,a2,s2,a20,A"));
    assert!(names.contains(&"F,a2,s2,a20,s1,a6,A"));
    assert!(names.contains(&"s3,f6,S"));
    assert!(names.contains(&"s8,f200,s3,A"));
    assert!(names.contains(&"F,f2,s2,f20,s1,f6,s3,S"));
    // `F,a2` follows a forced advance, not spec: no boundary arm.
    assert!(!names.contains(&"F,a2,A"));
    // `s8,A` scripts no AR count: no boundary or oracle arm.
    assert_eq!(names.iter().filter(|n| n.starts_with("s8,A")).count(), 1);
    let mirrors: Vec<_> = all.iter().filter(|(o, _)| o == "R").collect();
    assert_eq!(mirrors.len(), 2, "one mirrored pure-AR arm per F schedule");
    assert_eq!(close_family("F,a2,s2,a20,A"), Some(SCHEDULES[4]));
    assert_eq!(close_family("s3,f6,s1,f20,s6,F,s2,A"), Some(SCHEDULES[3]));
    assert_eq!(close_family("s3,a6,s1,a20,A"), None);
    for (ops, src) in &all {
        parse_ops(ops);
        if let Some(src) = src {
            let at = names.iter().position(|n| n == src).unwrap();
            let me = names.iter().position(|n| n == ops).unwrap();
            assert!(at < me, "oracle source runs first");
        }
    }
}

// ─── CPU: gate 22 (calibration neutrality) ──────────────────────────────────

fn calibration_sabotage_file(ops: &str, mode: &str) -> String {
    format!("{}+{mode}.trace", ops.replace(',', "-"))
}

#[test]
#[ignore = "lists the gate-22 arms for the driver"]
fn spec_floor_calibration_arms() {
    for ops in CALIBRATION_SCHEDULES {
        println!("CAL {ops}");
    }
    for mode in CALIBRATION_SABOTAGE.1 {
        println!("CALSABOTAGE {} {mode}", CALIBRATION_SABOTAGE.0);
    }
}

#[test]
#[ignore = "reads SPEC_FLOOR_HW_TRACES/{fresh,calibrated} written by qwen35_dflash_floor_schedule_arm"]
fn spec_floor_calibration_verdict() {
    let root = PathBuf::from(std::env::var("SPEC_FLOOR_HW_TRACES").expect("SPEC_FLOOR_HW_TRACES"));
    let mut prompts: Vec<String> = std::fs::read_dir(root.join("fresh"))
        .expect("fresh traces")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    prompts.sort();
    let read = |side: &str, prompt: &str, file: &str| {
        let path = root.join(side).join(prompt).join(file);
        Trace::read(&path).unwrap_or_else(|| panic!("missing {}", path.display()))
    };
    let mut failures = 0usize;
    let mut required_detected = 0usize;
    for prompt in &prompts {
        for ops in CALIBRATION_SCHEDULES {
            let file = arm_file(ops, None, None);
            let fresh = read("fresh", prompt, &file);
            let calibrated = read("calibrated", prompt, &file);
            let n = fresh.tokens.len().max(calibrated.tokens.len());
            let ids = id_diffs(&calibrated, &fresh, 0..n);
            let surf = surface_diffs(&calibrated, &fresh, None);
            let fail = !ids.is_empty()
                || !surf.is_empty()
                || calibrated.tokens.len() != fresh.tokens.len()
                || calibrated.checkpoints.len() != fresh.checkpoints.len();
            eprintln!(
                "(22) {prompt} {ops}: {} ids, {} surfaces over {} checkpoints differ calibrated vs fresh -> {}",
                ids.len(),
                surf.len(),
                fresh.checkpoints.len(),
                if fail { "FAIL" } else { "pass" }
            );
            if fail {
                summarize(&format!("(22) {prompt} {ops}"), &surf);
                failures += 1;
            }
        }
        let (ops, modes) = CALIBRATION_SABOTAGE;
        let fresh = read("fresh", prompt, &arm_file(ops, None, None));
        for mode in modes {
            let sab = read("calibrated", prompt, &calibration_sabotage_file(ops, mode));
            let n = fresh.tokens.len().max(sab.tokens.len());
            let ids = id_diffs(&sab, &fresh, 0..n);
            let surf = surface_diffs(&sab, &fresh, None);
            let detected = !ids.is_empty() || !surf.is_empty();
            eprintln!(
                "(22) {prompt} sabotage {mode}: {} ids, {} surfaces differ -> {}",
                ids.len(),
                surf.len(),
                if detected { "detected" } else { "NOT DETECTED" }
            );
            summarize(&format!("(22) {prompt} sabotage {mode}"), &surf);
            required_detected += usize::from(mode == "skip-drafter" && detected);
        }
    }
    eprintln!(
        "spec_floor_calibration_verdict: {} prompts, {failures} failures, skip-drafter detected on {required_detected}",
        prompts.len()
    );
    assert!(!prompts.is_empty(), "no prompt traces");
    assert_eq!(
        required_detected,
        prompts.len(),
        "the skip-drafter sabotage arm was not detected on every prompt"
    );
    assert_eq!(failures, 0, "gate 22 mismatches");
}
