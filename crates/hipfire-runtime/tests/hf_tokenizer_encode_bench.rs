// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Opt-in, release-only, single-thread encode benchmark:
//! `hipfire_runtime::tokenizer::Tokenizer::encode` versus the pinned HF
//! `tokenizers =0.22.2` reference (both are dev-dependencies already used by
//! `hf_tokenizer_differential`; nothing here ships in a binary).
//!
//! It is an `#[ignore]`d integration test rather than an example so it adds no
//! `[[example]]` entry (the ungated-example count is ratcheted by
//! `scripts/leanup-ratchets.sh`) and never runs in the normal test sweep.
//!
//! ```text
//! cargo test --release -p hipfire-runtime --test hf_tokenizer_encode_bench \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Reference: `tests/fixtures/hf_tokenizer_corpus/tokenizer.json`
//! (Qwen/Qwen3.8-27B, sha256 pinned by that directory's `manifest.json`;
//! normalizer `{"type":"NFC"}`). Workloads (all encoded with
//! `add_special_tokens=false`, one `encode` call per request, full rendered
//! text each time):
//!
//! * `hermes138`, `tc55`, `tc14` — exactly the committed corpus groups of
//!   `hf_tokenizer_corpus/cases.jsonl` (counts asserted, text sha256 verified,
//!   HF IDs asserted equal to the committed reference IDs).
//! * `doc64k`, `doc256k` — one document each, built by cycling the sorted
//!   `benchmarks/prompts/*.txt` (joined with a blank line) and cutting at the
//!   shortest char-boundary prefix whose HF token count reaches 65 536 /
//!   262 144.
//! * `doc64k_nfc_nonascii` — 64K tokens of already-NFC non-ASCII text
//!   (accents, Hangul, CJK, Cyrillic, emoji): isolates the NFC quick-check cost
//!   on text where the ASCII shortcut cannot help.
//! * `doc64k_nfc_adversarial` — 64K tokens of text whose NFC form differs
//!   (combining sequences, Hangul jamo, reordering marks): isolates the cost
//!   of real normalization.
//!
//! "NFC on" loads `tokenizer.json` unchanged. "NFC off" loads the same JSON
//! with `normalizer` set to `null`, for both engines, in memory only; this
//! disables normalization, not merely the quick path, and therefore only
//! preserves IDs for texts NFC leaves unchanged. "NFC eager" is the direct
//! quick-path-off comparison: fully normalize the input using HF's Unicode 9
//! NFC tables, then encode with the no-normalizer hipfire tokenizer. Its IDs
//! must equal HF NFC-on; unlike no-normalization, eager NFC preserves meaning.
//! Each line reports `nfc_changed_requests` and `ids_equal` compares against
//! HF in the corresponding normalization mode.
//!
//! Engines: `hf` = `Tokenizer::encode` (what the differential oracle uses,
//! builds tokens/offsets), `hf_fast` = `Tokenizer::encode_fast` (IDs only; the
//! fairer throughput comparison), `hipfire` = `Tokenizer::encode`.
//!
//! Output: one JSON object per line on stdout (`record` = `encode` |
//! `hermes_estimate`), progress on stderr. `elapsed_ns` is the median over
//! `reps` timed passes after one untimed warm-up pass; a pass is the sum of
//! per-request `Instant` deltas, outputs go through `black_box`.
//!
//! `hermes_estimate` models the Hermes pass: the 138 RC3 requests are the real
//! corpus. RC4c issued 115 requests (461,078 prompt tokens incl. cached;
//! `/home/kaden/qcal/release-0.4.1/hermes-p1-rc4c-breakdown.md`, lines 20, 50,
//! 103), whose bodies are not in the corpus, so the 115-request cost is given
//! three ways: `first115_ms` (first 115 corpus requests; caveat: not the
//! requests RC4c actually sent), `scaled115_requests_ms`
//! (`pass_138 × 115/138`) and `scaled115_tokens_ms`
//! (`pass_138 × 461 078 / corpus hermes tokens`). Each is compared with RC4c
//! wall 896.882 s / prefill 130.274 s and RC3 wall 1145.377 s.
//!
//! Env: `HIPFIRE_TOKBENCH_REPS` (default 7), `HIPFIRE_TOKBENCH_ONLY`
//! (comma-separated workload names), `HIPFIRE_QCAL_BREAKDOWN` (path of the
//! breakdown report; if readable its headline numbers are cross-checked).

use std::collections::BTreeMap;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use hipfire_runtime::tokenizer::Tokenizer as Ours;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer as Hf;

const MODEL: &str = "qwen38";
const DEFAULT_REPS: usize = 7;
const DEFAULT_BREAKDOWN: &str = "/home/kaden/qcal/release-0.4.1/hermes-p1-rc4c-breakdown.md";

// hermes-p1-rc4c-breakdown.md (lines 20, 50, 103 and the RC3 comparison).
const RC4C_REQUESTS: f64 = 115.0;
const RC4C_PROMPT_TOKENS: f64 = 461_078.0;
const RC4C_WALL_S: f64 = 896.882;
const RC4C_PREFILL_S: f64 = 130.274;
const RC3_REQUESTS: f64 = 138.0;
const RC3_PROMPT_TOKENS: f64 = 659_385.0;
const RC3_WALL_S: f64 = 1145.377;
/// Strings that must appear in the breakdown report if it is present.
const BREAKDOWN_MARKERS: [&str; 5] = ["896.882", "461,078", "130.274", "1145.377", "659,385"];

const ADVERSARIAL_SNIPPET: &str = "Cafe\u{301} re\u{301}sume\u{301} nai\u{308}ve A\u{30A}ngstro\u{308}m \
    \u{1112}\u{1161}\u{11AB}\u{1100}\u{1173}\u{11AF} q\u{307}\u{323}x \u{3b1}\u{342}\u{345} \
    <|im_start|>user\nde\u{301}ja\u{300} vu: o\u{302}\u{323} fn main() { println!(\"e\u{301}\"); }\n\
    <|im_end|>\n\u{0958}\u{0915}\u{093C} 12345 \n\t  ";
const NONASCII_NFC_SNIPPET: &str = "Caf\u{e9} r\u{e9}sum\u{e9} na\u{ef}ve \u{c5}ngstr\u{f6}m \
    \u{d55c}\u{ae00} \u{65e5}\u{672c}\u{8a9e}\u{306e}\u{30c6}\u{30ad}\u{30b9}\u{30c8} \u{4e2d}\u{6587}\u{6587}\u{672c} \
    \u{41f}\u{440}\u{438}\u{432}\u{435}\u{442} \u{43c}\u{438}\u{440} \u{1F600}\u{1F44D}\u{1F3FD} \
    \u{e9}\u{e8}\u{ea} fn main() { println!(\"\u{e9}\"); }\n\n";

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("hf_tokenizer_corpus")
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn median(v: &mut [u64]) -> u64 {
    v.sort_unstable();
    v[v.len() / 2]
}

// ---------------------------------------------------------------- engines --

#[derive(Clone, Copy)]
enum Engine<'a> {
    Hf(&'a Hf),
    HfFast(&'a Hf),
    Ours(&'a Ours),
    OursEagerNfc(&'a Ours),
}

impl Engine<'_> {
    fn name(&self) -> &'static str {
        match self {
            Engine::Hf(_) => "hf",
            Engine::HfFast(_) => "hf_fast",
            Engine::Ours(_) => "hipfire",
            Engine::OursEagerNfc(_) => "hipfire_eager_nfc",
        }
    }

    /// Encode and return the number of tokens; the output is `black_box`ed.
    fn encode_len(&self, text: &str) -> usize {
        match self {
            Engine::Hf(t) => {
                let enc = t.encode(black_box(text), false).expect("hf encode");
                let ids = black_box(enc.get_ids());
                ids.len()
            }
            Engine::HfFast(t) => {
                let enc = t
                    .encode_fast(black_box(text), false)
                    .expect("hf encode_fast");
                let ids = black_box(enc.get_ids());
                ids.len()
            }
            Engine::Ours(t) => {
                let ids = black_box(t.encode(black_box(text)));
                ids.len()
            }
            Engine::OursEagerNfc(t) => {
                use unicode_normalization_alignments::UnicodeNormalization;
                // Deliberately allocate an NFC string every time, including
                // already-normalized requests: this is the no-fast-path arm.
                let normalized: String = black_box(text).nfc().map(|(c, _)| c).collect();
                let ids = black_box(t.encode(black_box(&normalized)));
                ids.len()
            }
        }
    }

    fn ids(&self, text: &str) -> Vec<u32> {
        match self {
            Engine::Hf(t) => t.encode(text, false).expect("hf encode").get_ids().to_vec(),
            Engine::HfFast(t) => t
                .encode_fast(text, false)
                .expect("hf encode_fast")
                .get_ids()
                .to_vec(),
            Engine::Ours(t) => t.encode(text),
            Engine::OursEagerNfc(t) => {
                use unicode_normalization_alignments::UnicodeNormalization;
                let normalized: String = text.nfc().map(|(c, _)| c).collect();
                t.encode(&normalized)
            }
        }
    }
}

struct Measured {
    /// Median pass time over the timed reps.
    elapsed_ns: u64,
    min_ns: u64,
    tokens: u64,
    /// Per-request median (over reps) of the encode time, in ns.
    per_request_ns: Vec<u64>,
}

fn measure(engine: Engine, texts: &[String], reps: usize) -> Measured {
    let mut per_rep: Vec<Vec<u64>> = Vec::with_capacity(reps);
    let mut totals: Vec<u64> = Vec::with_capacity(reps);
    let mut tokens = 0u64;
    // Rep 0 is the untimed warm-up pass.
    for rep in 0..=reps {
        let mut times = Vec::with_capacity(texts.len());
        let mut toks = 0u64;
        for t in texts {
            let start = Instant::now();
            let n = engine.encode_len(t);
            let dt = start.elapsed().as_nanos() as u64;
            times.push(dt);
            toks += n as u64;
        }
        if rep > 0 {
            totals.push(times.iter().sum());
            per_rep.push(times);
            tokens = toks;
        }
    }
    let min_ns = *totals.iter().min().expect("reps >= 1");
    let per_request_ns = (0..texts.len())
        .map(|i| {
            let mut col: Vec<u64> = per_rep.iter().map(|r| r[i]).collect();
            median(&mut col)
        })
        .collect();
    Measured {
        elapsed_ns: median(&mut totals),
        min_ns,
        tokens,
        per_request_ns,
    }
}

// -------------------------------------------------------------- workloads --

struct Workload {
    group: String,
    texts: Vec<String>,
    /// Committed reference IDs (corpus groups only), for fixture integrity.
    reference: Option<Vec<Vec<u32>>>,
}

impl Workload {
    fn bytes(&self) -> u64 {
        self.texts.iter().map(|t| t.len() as u64).sum()
    }
}

fn load_corpus_groups(dir: &std::path::Path) -> Vec<Workload> {
    const REQUIRED: [(&str, usize); 3] = [("hermes", 138), ("tc55", 55), ("tc14", 14)];
    let raw = std::fs::read_to_string(dir.join("cases.jsonl")).expect("read cases.jsonl");
    let mut by_group: BTreeMap<String, Workload> = BTreeMap::new();
    for (name, _) in REQUIRED {
        let group = if name == "hermes" { "hermes138" } else { name };
        by_group.insert(
            name.to_string(),
            Workload {
                group: group.to_string(),
                texts: Vec::new(),
                reference: Some(Vec::new()),
            },
        );
    }
    for line in raw.lines().filter(|l| !l.is_empty()) {
        let v: Value = serde_json::from_str(line).expect("cases.jsonl line");
        let group = v["group"].as_str().expect("group");
        let Some(w) = by_group.get_mut(group) else {
            continue;
        };
        let id = v["id"].as_str().expect("id");
        let text = v["text"].as_str().expect("text");
        assert_eq!(
            sha256_hex(text.as_bytes()),
            v["text_sha256"].as_str().expect("text_sha256"),
            "{id}: committed text does not match its sha256"
        );
        let ids: Vec<u32> = v["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|x| x.as_u64().expect("id") as u32)
            .collect();
        w.texts.push(text.to_string());
        w.reference.as_mut().expect("reference").push(ids);
    }
    REQUIRED
        .iter()
        .map(|(name, n)| {
            let w = by_group.remove(*name).expect("group");
            assert_eq!(
                w.texts.len(),
                *n,
                "required group `{name}` must have {n} cases"
            );
            w
        })
        .collect()
}

fn prompt_pieces() -> Vec<String> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("benchmarks")
        .join("prompts");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    paths.sort();
    let pieces: Vec<String> = paths
        .iter()
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .filter(|s| !s.trim().is_empty())
        .collect();
    assert!(!pieces.is_empty(), "no benchmarks/prompts/*.txt pieces");
    pieces
}

/// Cycle `pieces` (joined with `sep`) and cut at the shortest char-boundary
/// prefix whose HF token count is >= `target`.
fn fit_doc(pieces: &[String], sep: &str, target: usize, hf: &Hf) -> String {
    let count = |s: &str| {
        hf.encode_fast(s, false)
            .expect("hf encode_fast")
            .get_ids()
            .len()
    };
    // Pass 1: whole pieces from per-piece counts.
    let mut text = String::new();
    let mut est = 0usize;
    let mut i = 0usize;
    while est < target {
        let p = &pieces[i % pieces.len()];
        est += count(p).max(1);
        text.push_str(p);
        text.push_str(sep);
        i += 1;
    }
    // Concatenation can merge differently than the parts; top up on real counts.
    while count(&text) < target {
        text.push_str(&pieces[i % pieces.len()]);
        text.push_str(sep);
        i += 1;
    }
    // Pass 2: bisect over char boundaries.
    let mut bounds: Vec<usize> = text.char_indices().map(|(b, _)| b).collect();
    bounds.push(text.len());
    let (mut lo, mut hi) = (0usize, bounds.len() - 1); // count(lo) < target <= count(hi)
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if count(&text[..bounds[mid]]) >= target {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    text.truncate(bounds[hi]);
    let got = count(&text);
    assert!(
        got >= target && got <= target + target / 100,
        "doc fit: {got} tokens for target {target}"
    );
    text
}

// ---------------------------------------------------------------- loading --

fn load_pair(json: &str, nfc_on: bool) -> (Hf, Ours) {
    let mut v: Value = serde_json::from_str(json).expect("parse tokenizer.json");
    if !nfc_on {
        v["normalizer"] = Value::Null;
    }
    let s = serde_json::to_string(&v).expect("serialize tokenizer.json");
    let hf = Hf::from_bytes(s.as_bytes()).expect("HF Tokenizer::from_bytes");
    let ours = Ours::from_hf_json(&s).expect("Tokenizer::from_hf_json");
    (hf, ours)
}

fn emit(v: Value) {
    println!("{v}");
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    sorted[(((sorted.len() - 1) as f64) * p).round() as usize]
}

struct Cell {
    on: bool,
    engine: &'static str,
    m: Measured,
}

#[test]
#[ignore = "opt-in benchmark; run with --release -- --ignored --nocapture --test-threads=1"]
fn bench_hf_tokenizer_encode() {
    let reps: usize = std::env::var("HIPFIRE_TOKBENCH_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_REPS)
        .max(1);
    let only: Option<Vec<String>> = std::env::var("HIPFIRE_TOKBENCH_ONLY")
        .ok()
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
    let selected = |g: &str| only.as_ref().map_or(true, |o| o.iter().any(|x| x == g));

    if cfg!(debug_assertions) {
        eprintln!("WARNING: debug build; timings are meaningless. Use --release.");
    }

    // --- qcal breakdown cross-check (optional) ---
    let breakdown_path =
        std::env::var("HIPFIRE_QCAL_BREAKDOWN").unwrap_or_else(|_| DEFAULT_BREAKDOWN.to_string());
    let breakdown_checked = match std::fs::read_to_string(&breakdown_path) {
        Ok(s) => {
            for m in BREAKDOWN_MARKERS {
                assert!(s.contains(m), "{breakdown_path}: marker `{m}` not found");
            }
            true
        }
        Err(_) => false,
    };

    // --- reference tokenizer, pinned ---
    let dir = corpus_dir();
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest"))
            .expect("parse manifest.json");
    let tok_bytes = std::fs::read(dir.join("tokenizer.json")).expect("read tokenizer.json");
    assert_eq!(
        sha256_hex(&tok_bytes),
        manifest["reference"]["tokenizer_json_sha256"]
            .as_str()
            .expect("sha"),
        "committed tokenizer.json does not match the manifest"
    );
    let tok_json = std::str::from_utf8(&tok_bytes).expect("tokenizer.json is UTF-8");
    let (hf_on, ours_on) = load_pair(tok_json, true);
    let (hf_off, ours_off) = load_pair(tok_json, false);

    // --- workloads ---
    let mut workloads = load_corpus_groups(&dir);
    let wl_filter: Vec<bool> = workloads.iter().map(|w| selected(&w.group)).collect();
    let mut keep = wl_filter.iter();
    workloads.retain(|_| *keep.next().expect("flag"));

    let docs: [(&str, usize, u8); 4] = [
        ("doc64k", 65_536, 0),
        ("doc256k", 262_144, 0),
        ("doc64k_nfc_nonascii", 65_536, 1),
        ("doc64k_nfc_adversarial", 65_536, 2),
    ];
    let mut prompts: Option<Vec<String>> = None;
    for (name, target, kind) in docs {
        if !selected(name) {
            continue;
        }
        eprintln!("building {name} ({target} tokens)");
        let text = match kind {
            0 => fit_doc(
                prompts.get_or_insert_with(prompt_pieces),
                "\n\n",
                target,
                &hf_on,
            ),
            1 => fit_doc(&[NONASCII_NFC_SNIPPET.to_string()], "", target, &hf_on),
            _ => fit_doc(&[ADVERSARIAL_SNIPPET.to_string()], "", target, &hf_on),
        };
        workloads.push(Workload {
            group: name.to_string(),
            texts: vec![text],
            reference: None,
        });
    }

    let mut all_equal = true;

    for w in &workloads {
        let requests = w.texts.len();
        let bytes = w.bytes();

        // Untimed verification: ids equality per NFC mode, fixture integrity,
        // and how many requests NFC actually changes.
        let hf_on_ids: Vec<Vec<u32>> = w.texts.iter().map(|t| Engine::Hf(&hf_on).ids(t)).collect();
        let hf_off_ids: Vec<Vec<u32>> =
            w.texts.iter().map(|t| Engine::Hf(&hf_off).ids(t)).collect();
        if let Some(reference) = &w.reference {
            assert_eq!(
                &hf_on_ids, reference,
                "{}: pinned HF IDs differ from the committed reference IDs",
                w.group
            );
        }
        let nfc_changed = hf_on_ids
            .iter()
            .zip(&hf_off_ids)
            .filter(|(a, b)| a != b)
            .count();

        let mut cells: Vec<Cell> = Vec::new();
        let mut ids_equal_map: BTreeMap<(bool, &'static str), usize> = BTreeMap::new();
        for on in [true, false] {
            let (hf, ours, hf_ids) = if on {
                (&hf_on, &ours_on, &hf_on_ids)
            } else {
                (&hf_off, &ours_off, &hf_off_ids)
            };
            let mut engines = vec![Engine::Hf(hf), Engine::HfFast(hf), Engine::Ours(ours)];
            if on {
                engines.push(Engine::OursEagerNfc(&ours_off));
            }
            for engine in engines {
                let mismatches = w
                    .texts
                    .iter()
                    .zip(hf_ids)
                    .filter(|(t, want)| &engine.ids(t) != *want)
                    .count();
                ids_equal_map.insert((on, engine.name()), mismatches);
                if mismatches != 0 {
                    all_equal = false;
                }
                eprintln!(
                    "measuring {} nfc={} {} ({} requests, {} bytes)",
                    w.group,
                    if on { "on" } else { "off" },
                    engine.name(),
                    requests,
                    bytes
                );
                cells.push(Cell {
                    on,
                    engine: engine.name(),
                    m: measure(engine, &w.texts, reps),
                });
            }
        }

        for c in &cells {
            let nfc = if c.on { "on" } else { "off" };
            let mismatches = ids_equal_map[&(c.on, c.engine)];
            let hf_ns = cells
                .iter()
                .find(|h| h.on == c.on && h.engine == "hf")
                .map(|h| h.m.elapsed_ns)
                .expect("hf cell");
            let hf_fast_ns = cells
                .iter()
                .find(|h| h.on == c.on && h.engine == "hf_fast")
                .map(|h| h.m.elapsed_ns)
                .expect("hf_fast cell");
            let ns = c.m.elapsed_ns.max(1) as f64;
            emit(json!({
                "record": "encode",
                "group": w.group,
                "model": MODEL,
                "nfc": nfc,
                "engine": c.engine,
                "requests": requests,
                "total_utf8_bytes": bytes,
                "total_output_tokens": c.m.tokens,
                "reps": reps,
                "elapsed_ns": c.m.elapsed_ns,
                "min_elapsed_ns": c.m.min_ns,
                "mb_per_s": bytes as f64 / ns * 1e3,
                "tokens_per_s": c.m.tokens as f64 / ns * 1e9,
                "ms_per_request": ns / 1e6 / requests as f64,
                "speedup_vs_hf": hf_ns as f64 / ns,
                "speedup_vs_hf_fast": hf_fast_ns as f64 / ns,
                "ids_equal_requests": requests - mismatches,
                "ids_equal": mismatches == 0,
                "nfc_changed_requests": nfc_changed,
            }));
        }

        // Hermes pass model.
        if w.group == "hermes138" {
            let corpus_tokens: u64 = hf_on_ids.iter().map(|i| i.len() as u64).sum();
            for c in &cells {
                let nfc = if c.on { "on" } else { "off" };
                let ns = &c.m.per_request_ns;
                let pass_ns: u64 = ns.iter().sum();
                let first115_ns: u64 = ns.iter().take(115).sum();
                let scaled_req_ns = pass_ns as f64 * RC4C_REQUESTS / RC3_REQUESTS;
                let scaled_tok_ns = pass_ns as f64 * RC4C_PROMPT_TOKENS / corpus_tokens as f64;
                let mut sorted = ns.clone();
                sorted.sort_unstable();
                let pct = |est_ns: f64, wall_s: f64| est_ns / 1e9 / wall_s * 100.0;
                emit(json!({
                    "record": "hermes_estimate",
                    "group": "hermes138",
                    "model": MODEL,
                    "nfc": nfc,
                    "engine": c.engine,
                    "basis": "sum of per-request medians; one full rendered encode per request",
                    "breakdown_source": DEFAULT_BREAKDOWN,
                    "breakdown_checked": breakdown_checked,
                    "corpus_requests": requests,
                    "corpus_prompt_tokens": corpus_tokens,
                    "rc3_usage_prompt_tokens": RC3_PROMPT_TOKENS,
                    "rc4c_requests": RC4C_REQUESTS,
                    "rc4c_prompt_tokens": RC4C_PROMPT_TOKENS,
                    "pass138_ms": pass_ns as f64 / 1e6,
                    "first115_ms": first115_ns as f64 / 1e6,
                    "scaled115_requests_ms": scaled_req_ns / 1e6,
                    "scaled115_tokens_ms": scaled_tok_ns / 1e6,
                    "first115_caveat": "first 115 corpus requests, not the 115 RC4c sent",
                    "request_ms_p50": percentile(&sorted, 0.5) as f64 / 1e6,
                    "request_ms_p95": percentile(&sorted, 0.95) as f64 / 1e6,
                    "request_ms_max": *sorted.last().expect("requests") as f64 / 1e6,
                    "rc3_wall_s": RC3_WALL_S,
                    "rc4c_wall_s": RC4C_WALL_S,
                    "rc4c_prefill_s": RC4C_PREFILL_S,
                    "pass138_pct_of_rc3_wall": pct(pass_ns as f64, RC3_WALL_S),
                    "scaled115_requests_pct_of_rc4c_wall": pct(scaled_req_ns, RC4C_WALL_S),
                    "scaled115_tokens_pct_of_rc4c_wall": pct(scaled_tok_ns, RC4C_WALL_S),
                    "scaled115_tokens_pct_of_rc4c_prefill": pct(scaled_tok_ns, RC4C_PREFILL_S),
                }));
            }
        }
    }

    assert!(
        all_equal,
        "hipfire/HF IDs differ for at least one workload (see `ids_equal`/`ids_equal_requests`)"
    );
}
