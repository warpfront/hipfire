// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Differential test: `hipfire_runtime::tokenizer::Tokenizer::from_hf_json`
//! against the HuggingFace `tokenizers` crate (pure-Rust `fancy-regex`
//! backend, **dev-dependency only** — no C++/onig, no Python, no network).
//!
//! Every manifest entry in `tests/fixtures/hf_tokenizer_differential/manifest.json`
//! (`[{family, file, md5, source, revision}]`, `file` relative to that
//! directory) is MD5-verified, loaded by HF (`Tokenizer::from_bytes`) and by
//! hipfire (`Tokenizer::from_hf_json`), then both encode the same inputs with
//! `add_special_tokens = false` and the IDs must be identical:
//!
//! * all 945 cases of `tests/fixtures/hf_tokenizer_corpus/cases.jsonl`
//!   (their *text* only; the live HF oracle supplies the reference IDs so every
//!   family is compared, not just Qwen);
//! * every `benchmarks/prompts/*.txt`, sorted by file name;
//! * a deterministic seeded adversarial corpus, `HIPFIRE_TOKENIZER_FUZZ_CASES`
//!   (default 50000) per tokenizer: whitespace runs, tabs, newlines, CRLF,
//!   unicode spaces, digit runs of every length, combining marks, emoji / ZWJ
//!   sequences, CJK and punctuation, code, JSON, plus each tokenizer's added /
//!   special tokens at word, whitespace and string boundaries.
//!
//! Where HF `decode(ids, false) == input` the test additionally requires
//! hipfire `decode(hf_ids) == input` and `encode(decode(our_ids)) == our_ids`.
//!
//! `tests/fixtures/hf_tokenizer_differential/regressions.jsonl` (must exist; may
//! be empty) holds fixed repros, one `{"family","text","ids"[,"id"]}` per line.
//! For that family each is also run through the comparison above and
//! additionally pinned: HF and hipfire must both produce exactly `ids`
//! (finding kind `regression_pin`). Commit a line whenever a runtime fix lands.
//!
//! The test never stops at the first mismatch. Every mismatch is recorded,
//! each unique signature is minimised (Unicode-safe delta reduction), and
//! complete JSONL reports are written to `HIPFIRE_TOKENIZER_DIFF_REPORT_DIR`
//! (default: `<temp_dir>/hipfire_tokenizer_differential`):
//!
//! * `<family>.failures.jsonl`  — every failing (case, kind), full input;
//! * `<family>.minimized.jsonl` — one line per unique mismatch signature with
//!   its minimal repro and all contributing case sources;
//! * `<family>.summary.json`    — counts and per-root-signature minimal repros.
//!
//! Each family is its own `#[test]` (a failure in one never skips another);
//! `unlisted_manifest_families` covers any manifest family without a dedicated
//! test, and `manifest_integrity` checks MD5s / coverage. A mismatch makes the
//! family test fail *after* all reports are written.
//!
//! ```text
//! cargo test --release -p hipfire-runtime --test hf_tokenizer_differential
//! HIPFIRE_TOKENIZER_FUZZ_CASES=200000 HIPFIRE_TOKENIZER_DIFF_REPORT_DIR=/tmp/tokdiff \
//!     cargo test --release -p hipfire-runtime --test hf_tokenizer_differential qwen35 -- --nocapture
//! ```
//!
//! Environment knobs: `HIPFIRE_TOKENIZER_FUZZ_CASES` (cases per tokenizer,
//! default 50000), `HIPFIRE_TOKENIZER_FUZZ_SEED` (u64, decimal or `0x` hex),
//! `HIPFIRE_TOKENIZER_DIFF_REPORT_DIR`, `HIPFIRE_TOKENIZER_DIFF_MIN_PROBES`
//! (per-signature minimisation budget, default 5000),
//! `HIPFIRE_TOKENIZER_DIFF_MIN_LIMIT` (max signatures minimised per family,
//! default 2000; the rest are still reported, marked `skipped_limit`),
//! `HIPFIRE_TOKENIZER_DIFF_PRINT_LIMIT` (root signatures printed, default 50).

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::Once;

use hipfire_runtime::tokenizer::Tokenizer as OursTokenizer;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer as HfTokenizer;

const DEFAULT_FUZZ_CASES: usize = 50_000;
const DEFAULT_FUZZ_SEED: u64 = 0x4849_5046_4952_4554; // "HIPFIRET"
const DEFAULT_MIN_PROBES: usize = 5_000;
const DEFAULT_MIN_LIMIT: usize = 2_000;
const DEFAULT_PRINT_LIMIT: usize = 50;
/// Added tokens exercised systematically per tokenizer (strided if more).
const MAX_ADDED_TOKENS_SYSTEMATIC: usize = 1024;
/// Cap on a generated fuzz case, in chars.
const MAX_FUZZ_CHARS: usize = 4096;
/// The committed corpus must not silently shrink.
const CORPUS_CASES: usize = 945;

// ───────────────────────────── environment / paths ─────────────────────────

fn env_u64(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Ok(v) if !v.trim().is_empty() => {
            let t = v.trim();
            let parsed = if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
                u64::from_str_radix(&h.replace('_', ""), 16)
            } else {
                t.replace('_', "").parse::<u64>()
            };
            parsed.unwrap_or_else(|e| panic!("{name}={v:?} is not a u64: {e}"))
        }
        _ => default,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env_u64(name, default as u64) as usize
}

fn differential_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("hf_tokenizer_differential")
}

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("hf_tokenizer_corpus")
}

fn prompts_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("benchmarks")
        .join("prompts")
}

fn report_dir() -> PathBuf {
    let dir = match std::env::var_os("HIPFIRE_TOKENIZER_DIFF_REPORT_DIR") {
        Some(d) if !d.is_empty() => PathBuf::from(d),
        _ => std::env::temp_dir().join("hipfire_tokenizer_differential"),
    };
    std::fs::create_dir_all(&dir)
        .unwrap_or_else(|e| panic!("create report dir {}: {e}", dir.display()));
    dir
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(64);
    for b in Sha256::digest(bytes) {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn md5_hex(bytes: &[u8]) -> String {
    format!("{:x}", md5::compute(bytes))
}

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ───────────────────────────────── manifest ────────────────────────────────

#[derive(Clone, Debug)]
struct ManifestEntry {
    family: String,
    file: String,
    md5: String,
    source: String,
    revision: String,
}

fn load_manifest() -> Vec<ManifestEntry> {
    let path = differential_dir().join("manifest.json");
    let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let v: Value =
        serde_json::from_slice(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    let arr = v
        .as_array()
        .unwrap_or_else(|| panic!("{} must be a JSON array", path.display()));
    let field = |e: &Value, k: &str| -> String {
        e.get(k)
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("manifest entry {e} lacks string `{k}`"))
            .to_string()
    };
    arr.iter()
        .map(|e| ManifestEntry {
            family: field(e, "family"),
            file: field(e, "file"),
            md5: field(e, "md5"),
            source: field(e, "source"),
            revision: field(e, "revision"),
        })
        .collect()
}

/// Read the entry's tokenizer.json and verify its MD5. Returns the bytes.
fn read_verified(entry: &ManifestEntry) -> Result<Vec<u8>, String> {
    let path = differential_dir().join(&entry.file);
    let bytes = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let got = md5_hex(&bytes);
    if got != entry.md5 {
        return Err(format!(
            "{}: md5 {} != manifest {} ({})",
            entry.family,
            got,
            entry.md5,
            path.display()
        ));
    }
    Ok(bytes)
}

// ─────────────────────────── panic-guarded oracle ──────────────────────────

thread_local! {
    static QUIET: Cell<bool> = const { Cell::new(false) };
}
static HOOK: Once = Once::new();

fn install_quiet_hook() {
    HOOK.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !QUIET.with(|q| q.get()) {
                prev(info);
            }
        }));
    });
}

/// Run `f`, converting a panic into `Err(message)` without console spam.
fn guarded<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    QUIET.with(|q| q.set(true));
    let r = catch_unwind(AssertUnwindSafe(f));
    QUIET.with(|q| q.set(false));
    r.map_err(|p| {
        let msg = if let Some(s) = p.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = p.downcast_ref::<String>() {
            s.clone()
        } else {
            "non-string panic".to_string()
        };
        format!("panic: {msg}")
    })
}

type Enc = Result<Vec<u32>, String>;

struct Oracle {
    hf: HfTokenizer,
    ours: OursTokenizer,
}

#[derive(Default)]
struct Eval {
    findings: Vec<Finding>,
    hf_tokens: usize,
    decode_checked: bool,
}

struct Finding {
    kind: &'static str,
    ours_ids: Option<Vec<u32>>,
    hf_ids: Option<Vec<u32>>,
    detail: String,
    /// Coarse dedupe key computed before minimisation.
    pre_sig: String,
    extra: Value,
}

fn norm_err(e: &str) -> String {
    let s: String = e
        .chars()
        .map(|c| if c.is_ascii_digit() { '#' } else { c })
        .take(96)
        .collect();
    s
}

fn first_divergence(a: &[u32], b: &[u32]) -> usize {
    a.iter()
        .zip(b.iter())
        .position(|(x, y)| x != y)
        .unwrap_or_else(|| a.len().min(b.len()))
}

impl Oracle {
    fn hf_encode(&self, t: &str) -> Enc {
        guarded(|| {
            self.hf
                .encode(t, false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| e.to_string())
        })
        .and_then(|r| r)
    }

    fn hf_decode(&self, ids: &[u32]) -> Result<String, String> {
        guarded(|| self.hf.decode(ids, false).map_err(|e| e.to_string())).and_then(|r| r)
    }

    fn ours_encode(&self, t: &str) -> Enc {
        guarded(|| self.ours.encode(t))
    }

    fn ours_decode(&self, ids: &[u32]) -> Result<String, String> {
        guarded(|| self.ours.decode(ids))
    }

    fn tok(&self, id: u32) -> String {
        self.hf
            .id_to_token(id)
            .unwrap_or_else(|| format!("<unk-id-{id}>"))
    }

    fn window(&self, ids: &[u32], at: usize, n: usize) -> Vec<String> {
        ids.iter().skip(at).take(n).map(|&i| self.tok(i)).collect()
    }

    fn encode_finding(&self, hf: &[u32], ours: &[u32]) -> Finding {
        let at = first_divergence(ours, hf);
        let ow = self.window(ours, at, 2);
        let hw = self.window(hf, at, 2);
        Finding {
            kind: "encode_ids",
            ours_ids: Some(ours.to_vec()),
            hf_ids: Some(hf.to_vec()),
            detail: format!(
                "ours={} hf={} tokens, first divergence at {at}",
                ours.len(),
                hf.len()
            ),
            pre_sig: format!(
                "encode_ids|ours:{ow:?}|hf:{hw:?}|len:{:?}",
                ours.len().cmp(&hf.len())
            ),
            extra: json!({
                "first_divergence": at,
                "ours_window": ow,
                "hf_window": hw,
                "ours_window_ids": ours.iter().skip(at).take(2).collect::<Vec<_>>(),
                "hf_window_ids": hf.iter().skip(at).take(2).collect::<Vec<_>>(),
            }),
        }
    }

    /// Full evaluation of one input. `only` restricts to one finding kind
    /// (used by the minimiser's predicate).
    fn evaluate(&self, text: &str, only: Option<&str>) -> Eval {
        let mut ev = Eval::default();
        let want = |k: &str| only.map_or(true, |o| o == k);
        let hf = self.hf_encode(text);
        let ours = self.ours_encode(text);
        match (&hf, &ours) {
            (Ok(h), Ok(o)) => {
                ev.hf_tokens = h.len();
                if h != o && want("encode_ids") {
                    ev.findings.push(self.encode_finding(h, o));
                }
            }
            (Ok(h), Err(e)) => {
                ev.hf_tokens = h.len();
                if want("ours_error") {
                    ev.findings.push(Finding {
                        kind: "ours_error",
                        ours_ids: None,
                        hf_ids: Some(h.clone()),
                        detail: e.clone(),
                        pre_sig: format!("ours_error|{}", norm_err(e)),
                        extra: Value::Null,
                    });
                }
            }
            (Err(e), Ok(o)) => {
                if want("hf_error") {
                    ev.findings.push(Finding {
                        kind: "hf_error",
                        ours_ids: Some(o.clone()),
                        hf_ids: None,
                        detail: e.clone(),
                        pre_sig: format!("hf_error|{}", norm_err(e)),
                        extra: Value::Null,
                    });
                }
            }
            (Err(e1), Err(e2)) => {
                if want("both_error") {
                    ev.findings.push(Finding {
                        kind: "both_error",
                        ours_ids: None,
                        hf_ids: None,
                        detail: format!("hf: {e1}; ours: {e2}"),
                        pre_sig: format!("both_error|{}|{}", norm_err(e1), norm_err(e2)),
                        extra: Value::Null,
                    });
                }
            }
        }

        let decode_kinds = ["decode_ours", "decode_error", "reencode_ours"];
        let decode_wanted = only.map_or(true, |o| decode_kinds.contains(&o));
        if !decode_wanted {
            return ev;
        }
        let Ok(h) = &hf else { return ev };
        // Lossless-roundtrip gate: only inputs HF itself decodes back to the
        // input are held to the decode / re-encode contract.
        match self.hf_decode(h) {
            Ok(d) if d == text => {}
            _ => return ev,
        }
        ev.decode_checked = true;
        match self.ours_decode(h) {
            Ok(s) => {
                if s != text && want("decode_ours") {
                    let at = s
                        .chars()
                        .zip(text.chars())
                        .position(|(a, b)| a != b)
                        .unwrap_or_else(|| s.chars().count().min(text.chars().count()));
                    let around: String = text.chars().skip(at.saturating_sub(1)).take(3).collect();
                    ev.findings.push(Finding {
                        kind: "decode_ours",
                        ours_ids: ours.as_ref().ok().cloned(),
                        hf_ids: Some(h.clone()),
                        detail: format!(
                            "ours.decode(hf_ids) != input; first differing char index {at}; ours_decoded={:?}",
                            clip(&s, 160)
                        ),
                        pre_sig: format!("decode_ours|{}", shape(&around)),
                        extra: json!({ "first_differing_char": at, "ours_decoded": s }),
                    });
                }
            }
            Err(e) => {
                if want("decode_error") {
                    ev.findings.push(Finding {
                        kind: "decode_error",
                        ours_ids: None,
                        hf_ids: Some(h.clone()),
                        detail: e.clone(),
                        pre_sig: format!("decode_error|{}", norm_err(&e)),
                        extra: Value::Null,
                    });
                }
            }
        }
        if want("reencode_ours") {
            if let Ok(o) = &ours {
                if let Ok(s) = self.ours_decode(o) {
                    match self.ours_encode(&s) {
                        Ok(r) if &r != o => {
                            let at = first_divergence(&r, o);
                            ev.findings.push(Finding {
                                kind: "reencode_ours",
                                ours_ids: Some(o.clone()),
                                hf_ids: Some(h.clone()),
                                detail: format!(
                                    "encode(decode(our_ids)) != our_ids: {} vs {} tokens, first divergence at {at}",
                                    r.len(),
                                    o.len()
                                ),
                                pre_sig: format!(
                                    "reencode_ours|re:{:?}|orig:{:?}",
                                    self.window(&r, at, 2),
                                    self.window(o, at, 2)
                                ),
                                extra: json!({ "reencoded_ids": r, "decoded": s }),
                            });
                        }
                        Ok(_) => {}
                        Err(e) => ev.findings.push(Finding {
                            kind: "reencode_ours",
                            ours_ids: Some(o.clone()),
                            hf_ids: Some(h.clone()),
                            detail: format!("re-encode failed: {e}"),
                            pre_sig: format!("reencode_ours|err|{}", norm_err(&e)),
                            extra: Value::Null,
                        }),
                    }
                }
            }
        }
        ev
    }
}

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(n).collect();
        t.push('…');
        t
    }
}

// ─────────────────────────── minimisation (ddmin) ──────────────────────────

fn is_extender(c: char) -> bool {
    matches!(
        c as u32,
        0x0300..=0x036F
            | 0x0483..=0x0489
            | 0x0591..=0x05BD
            | 0x064B..=0x065F
            | 0x0900..=0x0903
            | 0x093A..=0x094F
            | 0x1AB0..=0x1AFF
            | 0x1DC0..=0x1DFF
            | 0x200D
            | 0x20D0..=0x20FF
            | 0x3099
            | 0x309A
            | 0xFE00..=0xFE0F
            | 0xFE20..=0xFE2F
            | 0x1F3FB..=0x1F3FF
            | 0xE0100..=0xE01EF
    )
}

/// Split into grapheme-ish clusters: base char + combining marks / variation
/// selectors / skin tones, and ZWJ joins (so ZWJ sequences stay whole in the
/// first reduction stage).
fn clusters(chars: &[char]) -> Vec<Vec<char>> {
    let mut out: Vec<Vec<char>> = Vec::new();
    let mut prev_zwj = false;
    for &c in chars {
        if out.is_empty() || !(is_extender(c) || prev_zwj) {
            out.push(vec![c]);
        } else {
            out.last_mut().unwrap().push(c);
        }
        prev_zwj = c == '\u{200D}';
    }
    out
}

fn flatten(units: &[Vec<char>]) -> String {
    units.iter().flat_map(|u| u.iter()).collect()
}

struct MinState {
    probes: usize,
    budget: usize,
}

/// Classic ddmin over complements. Every candidate is a concatenation of whole
/// `char`s, hence always valid UTF-8.
fn ddmin(
    mut units: Vec<Vec<char>>,
    st: &mut MinState,
    pred: &mut dyn FnMut(&str) -> bool,
) -> Vec<Vec<char>> {
    let mut n = 2usize;
    while units.len() >= 2 && st.probes < st.budget {
        let len = units.len();
        let chunk = len.div_ceil(n);
        let mut reduced = false;
        let mut start = 0;
        while start < len {
            let end = (start + chunk).min(len);
            let cand: Vec<Vec<char>> = units[..start]
                .iter()
                .chain(units[end..].iter())
                .cloned()
                .collect();
            if !cand.is_empty() {
                st.probes += 1;
                if pred(&flatten(&cand)) {
                    units = cand;
                    n = n.saturating_sub(1).max(2);
                    reduced = true;
                    break;
                }
                if st.probes >= st.budget {
                    break;
                }
            }
            start = end;
        }
        if !reduced {
            if n >= len {
                break;
            }
            n = (n * 2).min(len);
        }
    }
    units
}

struct Minimized {
    text: String,
    probes: usize,
    status: &'static str,
}

fn minimize(text: &str, budget: usize, pred: &mut dyn FnMut(&str) -> bool) -> Minimized {
    if !pred(text) {
        return Minimized {
            text: text.to_string(),
            probes: 1,
            status: "original_not_reproducible",
        };
    }
    let mut st = MinState { probes: 1, budget };
    let chars: Vec<char> = text.chars().collect();
    let stage1 = ddmin(clusters(&chars), &mut st, pred);
    let flat: Vec<char> = stage1.iter().flat_map(|u| u.iter().copied()).collect();
    let stage2 = ddmin(flat.iter().map(|&c| vec![c]).collect(), &mut st, pred);
    Minimized {
        text: flatten(&stage2),
        probes: st.probes,
        status: if st.probes >= st.budget {
            "budget_exhausted"
        } else {
            "converged"
        },
    }
}

/// Character-class shape of a (minimal) repro: the root-cause signature.
fn shape(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i >= 64 {
            out.push_str(&format!("…({} chars)", s.chars().count()));
            break;
        }
        let u = c as u32;
        match c {
            ' ' => out.push('_'),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            'a'..='z' => out.push('a'),
            'A'..='Z' => out.push('A'),
            '0'..='9' => out.push('0'),
            _ if c.is_ascii_punctuation() => out.push(c),
            _ if c.is_ascii_control() => out.push_str(&format!("^{u:02X}")),
            '\u{200D}' => out.push('J'),
            _ if is_extender(c) => out.push('M'),
            _ if c.is_whitespace() || matches!(u, 0x200B..=0x200C | 0x2060 | 0xFEFF | 0x180E) => {
                out.push_str(&format!("S{u:04X}"))
            }
            _ if matches!(u, 0x1F000..=0x1FAFF | 0x2600..=0x27BF) => out.push('E'),
            _ if matches!(u, 0x4E00..=0x9FFF | 0x3040..=0x30FF | 0xAC00..=0xD7A3) => out.push('H'),
            _ if c.is_alphabetic() => out.push('L'),
            _ if c.is_numeric() => out.push('N'),
            _ => out.push_str(&format!("U+{u:X}")),
        }
    }
    out
}

// ─────────────────────────────── fuzz generation ───────────────────────────

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn range(&mut self, lo: usize, hi_incl: usize) -> usize {
        lo + self.below(hi_incl - lo + 1)
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

const ASCII_WS: &[&str] = &[
    " ", "  ", "   ", "    ", "        ", "\t", "\t\t", "\t\t\t", "\n", "\n\n", "\n\n\n",
    "\n\n\n\n", "\r\n", "\r\n\r\n", "\r", "\r\r", " \n", "\n ", " \n ", "\t\n", "\n\t", " \t ",
    "\t \t", "\r\n ", " \r\n", "  \n  ", "\n    ", "\n\t\t", "\u{000B}", "\u{000C}", "\u{001C}",
    "\u{001F}",
];

const UNI_WS: &[&str] = &[
    "\u{0085}", "\u{00A0}", "\u{00AD}", "\u{1680}", "\u{180E}", "\u{2000}", "\u{2001}", "\u{2002}",
    "\u{2003}", "\u{2004}", "\u{2005}", "\u{2006}", "\u{2007}", "\u{2008}", "\u{2009}", "\u{200A}",
    "\u{200B}", "\u{200C}", "\u{200D}", "\u{2028}", "\u{2029}", "\u{202F}", "\u{205F}", "\u{2060}",
    "\u{3000}", "\u{FEFF}",
];

const COMBINING_BASES: &[&str] = &[
    "e", "a", "o", "n", "A", "E", "z", "i", "u", "s", "и", "а", "α", "k", "क", "ا", "ก", "한",
];

const COMBINING_MARKS: &[&str] = &[
    "\u{0300}", "\u{0301}", "\u{0302}", "\u{0303}", "\u{0308}", "\u{030A}", "\u{0327}", "\u{0338}",
    "\u{0345}", "\u{0360}", "\u{036F}", "\u{0483}", "\u{0591}", "\u{064B}", "\u{0651}", "\u{093C}",
    "\u{094D}", "\u{0E31}", "\u{1AB0}", "\u{1DC0}", "\u{20D0}", "\u{20E3}", "\u{3099}", "\u{FE00}",
    "\u{FE0E}", "\u{FE0F}", "\u{FE20}",
];

const EMOJI: &[&str] = &[
    "😀",
    "😂",
    "🙂",
    "😍",
    "🥲",
    "🤔",
    "👍",
    "👍🏽",
    "👋🏻",
    "🔥",
    "✨",
    "❤️",
    "❤",
    "☺️",
    "✔️",
    "©️",
    "🇺🇸",
    "🇯🇵",
    "🇩🇪🇫🇷",
    "🏳️‍🌈",
    "🏴‍☠️",
    "👨‍👩‍👧‍👦",
    "👩‍💻",
    "🧑🏽‍🚀",
    "🤦🏼‍♂️",
    "1️⃣",
    "#️⃣",
    "*️⃣",
    "🐍",
    "🦀",
    "💩",
    "🎉",
    "🚀",
    "🧠",
    "𝟙",
    "𝕏",
    "🅰",
    "🈶",
    "⌚",
    "\u{1F9D1}\u{200D}\u{1F91D}\u{200D}\u{1F9D1}",
    "\u{E0067}\u{E0062}",
    "😀😀😀",
    "👨‍👩‍👧‍👦👨‍👩‍👧‍👦",
];

const CJK: &[&str] = &[
    "中",
    "文",
    "中文",
    "你好，世界",
    "这是一个测试。",
    "日本語のテキスト",
    "ひらがなカタカナ",
    "한국어 텍스트",
    "안녕하세요",
    "東京都",
    "漢字",
    "𠮷",
    "𠀀",
    "々",
    "ー",
    "ｱｲｳ",
    "ＡＢＣ１２３",
    "「引用」",
    "『二重』",
    "（括弧）",
    "【見出し】",
    "、。！？：；",
    "…",
    "——",
    "·",
    "〜",
    "￥",
    "％",
];

const CJK_PUNCT: &[&str] = &[
    "。", "，", "、", "！", "？", "：", "；", "「", "」", "『", "』", "（", "）", "【", "】", "《",
    "》", "〈", "〉", "…", "—", "～", "·", "“", "”", "‘", "’", "«", "»", "¿", "¡", "‽", "※",
];

const CONTRACTIONS: &[&str] = &[
    "don't",
    "I'm",
    "it's",
    "we'll",
    "they've",
    "you'd",
    "can't",
    "won't",
    "'s",
    "'S",
    "'T",
    "'re",
    "'RE",
    "'ve",
    "'ll",
    "'d",
    "'m",
    "’s",
    "’t",
    "O'Neil",
    "rock'n'roll",
    "''",
    "'''",
    "'tis",
];

const CODE: &[&str] = &[
    "fn main() {\n    println!(\"Hello, {}!\", name);\n}\n",
    "def f(x):\n\treturn x  # tab-indented\n\n\nclass A:\n    pass\n",
    "if (a && b || !c) { x += 1; } else { y -= 2; }",
    "#include <stdio.h>\nint main(void) {\n\tprintf(\"%d\\n\", 42);\n\treturn 0;\n}\n",
    "const f = async (a, b) => { await g(a ?? b); };",
    "SELECT a, b FROM t WHERE x >= 10 AND y <> 'z' ORDER BY 1 DESC;",
    "<div class=\"a\" id='b'>\n  <p>text &amp; more</p>\n</div>",
    "```python\nprint('hi')\n```\n",
    "# Heading\n\n- item 1\n- item 2\n  - nested\n\n> quote\n",
    "a=b\n\n\n    c=d   \n\te=f\t\n",
    "x = [1,2,3]; y = {a:1,b:2}; z = (1+2)*3/4%5",
    "/* comment */ // line\n#!/bin/bash\necho \"$HOME\" | grep -E '^/home/[a-z]+$'",
    "$ cargo test --release -p foo -- --nocapture\n",
    "\\begin{equation}\\alpha^2+\\beta_i=\\sum_{k=0}^{n}\\frac{1}{k!}\\end{equation}",
    "https://example.com/a/b?c=d&e=f#frag",
    "/usr/local/bin/python3.11",
    "user@example.com",
    "0x7fffffffe000 0xDEADBEEF 1e-10 3.14159 -42 +7 1,234,567.89 12:34:56 2026-10-07T12:34:56Z",
];

const JSON_SNIPPETS: &[&str] = &[
    "{\"a\": [1, 2, {\"b\": null}], \"c\": \"\\u00e9\\n\"}",
    "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"München\",\"units\":\"metric\"}}",
    "[{\"id\":1,\"ok\":true},{\"id\":2,\"ok\":false,\"msg\":\"a \\\"quoted\\\" word\"}]",
    "{\n  \"k\": \"v\",\n  \"list\": [\n    1,\n    2\n  ]\n}",
    "{\"emoji\":\"😀\",\"cjk\":\"中文\",\"num\":12345678901234567890}",
    "\"\\\\\"",
    "{}",
    "[]",
    "[[[[[[]]]]]]",
    "{\"a\":{\"b\":{\"c\":{\"d\":1}}}}",
    "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>",
];

const OTHER_SCRIPTS: &[&str] = &[
    "привет мир",
    "Ёлка",
    "Ελληνικά",
    "שלום עולם",
    "مرحبا بالعالم",
    "สวัสดีครับ",
    "नमस्ते दुनिया",
    "বাংলা",
    "தமிழ்",
    "ქართული",
    "հայերեն",
    "ᓀᐦᐃᔭᐍᐏᐣ",
    "ꦧꦱꦗꦮ",
    "ÀÉÎÕÜ àéîõü ß ñ ç",
    "İstanbul ıi",
    "Ǆǅǆ ǈ ǋ",
    "ﬁﬂ ﬃ",
    "Å",
    "ℌ ℝ ℂ",
    "①②③ ½ ¼ ² ³",
    "٠١٢٣٤٥٦٧٨٩",
    "०१२३४५६७८९",
    "௧௨௩",
];

const ASCII_PUNCT: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

fn random_scalar(rng: &mut Rng) -> char {
    let v = match rng.below(8) {
        0 => rng.range(0, 0x7F),
        1 => rng.range(0x80, 0x7FF),
        2 => rng.range(0x800, 0xD7FF),
        3 => rng.range(0xE000, 0xFFFF),
        4 => rng.range(0x10000, 0x10FFFF),
        5 => rng.range(0x2000, 0x2BFF),
        6 => rng.range(0x4E00, 0x9FFF),
        _ => rng.range(0x1F300, 0x1FAFF),
    };
    char::from_u32(v as u32).unwrap_or('\u{FFFD}')
}

struct FuzzGen {
    rng: Rng,
    added: Vec<String>,
}

impl FuzzGen {
    fn word(&mut self) -> String {
        let n = self.rng.range(1, 12);
        let upper_first = self.rng.below(4) == 0;
        let all_upper = self.rng.below(12) == 0;
        let mut s = String::new();
        for i in 0..n {
            let mut c = (b'a' + self.rng.below(26) as u8) as char;
            if all_upper || (i == 0 && upper_first) {
                c = c.to_ascii_uppercase();
            }
            s.push(c);
        }
        s
    }

    fn digits(&mut self) -> String {
        let n = self.rng.range(1, 24);
        let mut s = String::new();
        if self.rng.below(5) == 0 {
            s.push('0');
        }
        for _ in 0..n {
            s.push((b'0' + self.rng.below(10) as u8) as char);
        }
        match self.rng.below(6) {
            0 => s.insert(s.len() / 2, '.'),
            1 => s.insert(s.len() / 2, ','),
            2 => s.insert(0, '-'),
            _ => {}
        }
        s
    }

    fn piece(&mut self) -> String {
        match self.rng.below(19) {
            0 => {
                let n = self.rng.range(1, 3);
                (0..n).map(|_| *self.rng.pick(ASCII_WS)).collect()
            }
            1 => {
                let n = self.rng.range(1, 3);
                (0..n).map(|_| *self.rng.pick(UNI_WS)).collect()
            }
            2 => self.digits(),
            3 => self.word(),
            4 => (*self.rng.pick(CONTRACTIONS)).to_string(),
            5 => {
                let mut s = (*self.rng.pick(COMBINING_BASES)).to_string();
                for _ in 0..self.rng.range(1, 4) {
                    s.push_str(self.rng.pick(COMBINING_MARKS));
                }
                s
            }
            6 => {
                let n = self.rng.range(1, 3);
                (0..n).map(|_| *self.rng.pick(EMOJI)).collect()
            }
            7 => {
                if self.rng.below(2) == 0 {
                    (*self.rng.pick(CJK)).to_string()
                } else {
                    let n = self.rng.range(1, 6);
                    (0..n)
                        .map(|_| {
                            let v = match self.rng.below(3) {
                                0 => self.rng.range(0x4E00, 0x9FFF),
                                1 => self.rng.range(0x3040, 0x30FF),
                                _ => self.rng.range(0xAC00, 0xD7A3),
                            };
                            char::from_u32(v as u32).unwrap_or('?')
                        })
                        .collect()
                }
            }
            8 => {
                if self.rng.below(3) == 0 {
                    (*self.rng.pick(CJK_PUNCT)).to_string()
                } else {
                    let n = self.rng.range(1, 6);
                    let bytes = ASCII_PUNCT.as_bytes();
                    (0..n)
                        .map(|_| bytes[self.rng.below(bytes.len())] as char)
                        .collect()
                }
            }
            9 => (*self.rng.pick(CODE)).to_string(),
            10 => (*self.rng.pick(JSON_SNIPPETS)).to_string(),
            11 => (*self.rng.pick(OTHER_SCRIPTS)).to_string(),
            12 => {
                let n = self.rng.range(1, 4);
                (0..n).map(|_| random_scalar(&mut self.rng)).collect()
            }
            13 => {
                if self.added.is_empty() {
                    self.word()
                } else {
                    let i = self.rng.below(self.added.len());
                    self.added[i].clone()
                }
            }
            14 => {
                if self.added.is_empty() {
                    self.word()
                } else {
                    let i = self.rng.below(self.added.len());
                    let t = &self.added[i];
                    let n = t.chars().count();
                    let keep = self.rng.range(1, n.max(2) - 1).min(n);
                    t.chars().take(keep).collect()
                }
            }
            15 => {
                let mut s = "\n".to_string();
                for _ in 0..self.rng.range(0, 16) {
                    s.push(if self.rng.below(6) == 0 { '\t' } else { ' ' });
                }
                s.push_str(&self.word());
                s
            }
            16 => {
                let c = match self.rng.below(4) {
                    0 => (b'a' + self.rng.below(26) as u8) as char,
                    1 => (b'0' + self.rng.below(10) as u8) as char,
                    2 => ASCII_PUNCT.as_bytes()[self.rng.below(ASCII_PUNCT.len())] as char,
                    _ => random_scalar(&mut self.rng),
                };
                std::iter::repeat(c).take(self.rng.range(2, 40)).collect()
            }
            17 => {
                let mut s = String::new();
                let n = self.rng.range(1, 4);
                for i in 0..n {
                    if i > 0 {
                        s.push(['/', '.', '-', '_', ':'][self.rng.below(5)]);
                    }
                    s.push_str(&self.word());
                }
                s
            }
            _ => {
                // word + number glued, e.g. "gpt4", "v2beta", "x86_64"
                let mut s = self.word();
                s.push_str(&self.digits());
                if self.rng.below(2) == 0 {
                    s.push_str(&self.word());
                }
                s
            }
        }
    }

    fn random_case(&mut self) -> String {
        let mut s = String::new();
        if self.rng.below(6) == 0 {
            s.push_str(self.rng.pick(ASCII_WS));
        }
        let n = self.rng.range(1, 10);
        for i in 0..n {
            if i > 0 {
                match self.rng.below(5) {
                    0 | 1 => s.push(' '),
                    2 => s.push_str(self.rng.pick(ASCII_WS)),
                    _ => {}
                }
            }
            s.push_str(&self.piece());
            if s.chars().count() > MAX_FUZZ_CHARS {
                break;
            }
        }
        if self.rng.below(6) == 0 {
            s.push_str(self.rng.pick(ASCII_WS));
        }
        s.chars().take(MAX_FUZZ_CHARS).collect()
    }
}

/// Deterministic structured prefix of the adversarial corpus.
fn systematic_cases(added: &[String]) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    let ws_all: Vec<&str> = ASCII_WS.iter().chain(UNI_WS.iter()).copied().collect();
    for w in &ws_all {
        v.push((*w).to_string());
        v.push(format!("a{w}b"));
        v.push(format!("{w}a"));
        v.push(format!("a{w}"));
        v.push(format!("a {w}b"));
        v.push(format!("a{w} b"));
        v.push(format!("a{w}\n"));
        v.push(format!("\n{w}a"));
        v.push(format!("{w}{w}{w}"));
        v.push(format!("x{w}{w}y"));
        v.push(format!("  {w}  "));
    }
    // Whitespace run lengths 1..=40 for the common separators, between words
    // and alone.
    for w in [
        " ", "\t", "\n", "\r\n", "\n ", " \n", "\u{00A0}", "\u{3000}", "\u{2003}",
    ] {
        for n in 1..=40usize {
            let run = w.repeat(n);
            v.push(run.clone());
            v.push(format!("x{run}y"));
            v.push(format!("{run}x"));
            v.push(format!("x{run}"));
            v.push(format!("x{run}\n{run}y"));
        }
    }
    // Digit runs: every length 1..=48, bare / embedded / punctuated.
    for n in 1..=48usize {
        let d: String = (0..n).map(|i| (b'0' + (i % 10) as u8) as char).collect();
        let z: String = "0".repeat(n);
        let nines: String = "9".repeat(n);
        for s in [&d, &z, &nines] {
            v.push(s.clone());
            v.push(format!(" {s}"));
            v.push(format!("a{s}b"));
            v.push(format!("{s}.{s}"));
            v.push(format!("{s},{s}"));
            v.push(format!("-{s}\n"));
            v.push(format!("x = {s};"));
        }
    }
    // Combining marks on every base.
    for b in COMBINING_BASES {
        for m in COMBINING_MARKS {
            v.push(format!("{b}{m}"));
            v.push(format!("x {b}{m}{m} y"));
            v.push(format!("{m}{b}"));
        }
    }
    for e in EMOJI {
        v.push((*e).to_string());
        v.push(format!("a{e}b"));
        v.push(format!(" {e}"));
        v.push(format!("{e}\n"));
        v.push(format!("{e}{e}"));
        v.push(format!("12{e}34"));
    }
    for c in CJK
        .iter()
        .chain(CJK_PUNCT.iter())
        .chain(OTHER_SCRIPTS.iter())
    {
        v.push((*c).to_string());
        v.push(format!("a{c}b"));
        v.push(format!("a {c} b"));
        v.push(format!("{c}\n{c}"));
        v.push(format!("{c}123"));
    }
    for c in CONTRACTIONS {
        v.push((*c).to_string());
        v.push(format!("x {c} y"));
        v.push(format!("{c}{c}"));
    }
    for c in CODE.iter().chain(JSON_SNIPPETS.iter()) {
        v.push((*c).to_string());
        v.push(format!("{c}\n"));
        v.push(format!("\n{c}"));
        v.push(c.replace('\n', "\r\n"));
        v.push(c.replace("    ", "\t"));
    }
    // NFC edge cases: Unicode-table-sensitive canonical reorderings, plus every
    // ordered pair of the common combining marks on a base letter.
    for seq in [
        "\u{0302}\u{089B}",
        "\u{089B}\u{0302}",
        "\u{065D}\u{1DF7}",
        "\u{1DF7}\u{065D}",
    ] {
        v.push(seq.to_string());
        v.push(format!("e{seq}"));
        v.push(format!("a{seq}b"));
        v.push(format!(" {seq} "));
        v.push(format!("x\n{seq}\ty"));
        v.push(format!("\u{0627}{seq}"));
    }
    for a in COMBINING_MARKS {
        for b in COMBINING_MARKS {
            v.push(format!("e{a}{b}"));
        }
    }
    // Added / special token boundaries.
    for t in added {
        v.push(t.clone());
        v.push(format!("a{t}b"));
        v.push(format!(" {t} "));
        v.push(format!("{t}\n"));
        v.push(format!("\n{t}"));
        v.push(format!("{t}{t}"));
        v.push(format!("{t}a"));
        v.push(format!("a {t}"));
        v.push(format!("1{t}2"));
        v.push(format!("中{t}文"));
        v.push(format!("\t{t}\t"));
        v.push(format!("  {t}  "));
        v.push(format!("{t}\u{00A0}{t}"));
        let n = t.chars().count();
        if n > 2 {
            let cut: String = t.chars().take(n - 1).collect();
            v.push(cut.clone());
            v.push(format!("{cut} x"));
            let cut2: String = t.chars().take(n / 2).collect();
            v.push(format!("{cut2}{t}"));
            let tail: String = t.chars().skip(1).collect();
            v.push(format!("x{tail}"));
        }
    }
    v
}

fn added_tokens(hf: &HfTokenizer) -> Vec<String> {
    let mut all: Vec<(u32, String)> = hf
        .get_added_tokens_decoder()
        .iter()
        .map(|(id, t)| (*id, t.content.clone()))
        .filter(|(_, c)| !c.is_empty())
        .collect();
    all.sort();
    let contents: Vec<String> = all.into_iter().map(|(_, c)| c).collect();
    if contents.len() <= MAX_ADDED_TOKENS_SYSTEMATIC {
        return contents;
    }
    let stride = contents.len().div_ceil(MAX_ADDED_TOKENS_SYSTEMATIC);
    contents.into_iter().step_by(stride).collect()
}

// ─────────────────────────────────── corpora ───────────────────────────────

struct Case {
    source: String,
    text: String,
    /// Expected IDs for pinned regression cases (HF and hipfire must both match).
    pinned: Option<Vec<u32>>,
}

/// `regressions.jsonl` in the differential fixture dir: one
/// `{"family", "text", "ids"[, "id"]}` per line, committed with each runtime
/// fix. The file must exist (it may be empty). Returns `(family, case)`.
fn load_regression_cases() -> Vec<(String, Case)> {
    let path = differential_dir().join("regressions.jsonl");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {} (must exist, may be empty): {e}", path.display()));
    let mut out = Vec::new();
    for (n, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("regressions.jsonl line {}: {e}", n + 1));
        let family = v["family"]
            .as_str()
            .unwrap_or_else(|| panic!("regressions.jsonl line {}: `family`", n + 1))
            .to_string();
        let text = v["text"]
            .as_str()
            .unwrap_or_else(|| panic!("regressions.jsonl line {}: `text`", n + 1));
        let ids: Vec<u32> = v["ids"]
            .as_array()
            .unwrap_or_else(|| panic!("regressions.jsonl line {}: `ids`", n + 1))
            .iter()
            .map(|x| x.as_u64().expect("regression id is integer") as u32)
            .collect();
        let label = v
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("line{}", n + 1));
        out.push((
            family.clone(),
            Case {
                source: format!("regression:{family}:{label}"),
                text: text.to_string(),
                pinned: Some(ids),
            },
        ));
    }
    out
}

fn load_corpus_cases() -> Vec<Case> {
    let dir = corpus_dir();
    let raw = std::fs::read_to_string(dir.join("cases.jsonl"))
        .unwrap_or_else(|e| panic!("read cases.jsonl: {e}"));
    let mut out = Vec::new();
    for (n, line) in raw.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("cases.jsonl line {}: {e}", n + 1));
        let id = v["id"].as_str().expect("case id");
        let text = v["text"].as_str().expect("case text");
        assert_eq!(
            sha256_hex(text.as_bytes()),
            v["text_sha256"].as_str().expect("text_sha256"),
            "{id}: committed text does not match its recorded sha256"
        );
        out.push(Case {
            source: format!("corpus:{id}"),
            text: text.to_string(),
            pinned: None,
        });
    }
    assert_eq!(
        out.len(),
        CORPUS_CASES,
        "hf_tokenizer_corpus/cases.jsonl must keep all {CORPUS_CASES} cases"
    );
    out
}

fn load_prompt_cases() -> Vec<Case> {
    let dir = prompts_dir();
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".txt"))
        .collect();
    names.sort();
    assert!(!names.is_empty(), "no benchmarks/prompts/*.txt found");
    names
        .into_iter()
        .map(|n| {
            let bytes = std::fs::read(dir.join(&n)).unwrap_or_else(|e| panic!("read {n}: {e}"));
            let (text, lossy) = match String::from_utf8(bytes) {
                Ok(s) => (s, false),
                Err(e) => (String::from_utf8_lossy(e.as_bytes()).into_owned(), true),
            };
            Case {
                source: format!("prompt:{n}{}", if lossy { " (lossy-utf8)" } else { "" }),
                text,
                pinned: None,
            }
        })
        .collect()
}

impl Oracle {
    /// A pinned regression case must produce exactly `pinned` in both HF and hipfire.
    fn pin_finding(&self, text: &str, pinned: &[u32], source: &str) -> Option<Finding> {
        let hf = self.hf_encode(text);
        let ours = self.ours_encode(text);
        let hf_ok = hf.as_deref().map_or(false, |h| h == pinned);
        let ours_ok = ours.as_deref().map_or(false, |o| o == pinned);
        if hf_ok && ours_ok {
            return None;
        }
        Some(Finding {
            kind: "regression_pin",
            ours_ids: ours.as_ref().ok().cloned(),
            hf_ids: hf.as_ref().ok().cloned(),
            detail: format!(
                "pinned ids {}; hf_matches_pin={hf_ok} ours_matches_pin={ours_ok}",
                pinned.len()
            ),
            pre_sig: format!("regression_pin|{source}"),
            extra: json!({ "pinned_ids": pinned }),
        })
    }
}

// ───────────────────────────────── reporting ───────────────────────────────

struct SigEntry {
    kind: &'static str,
    count: usize,
    sources: Vec<String>,
    /// Shortest failing input seen for this signature.
    rep_text: String,
}

#[derive(Default)]
struct RootAgg {
    kind: String,
    cases: usize,
    pre_sigs: usize,
    repro: String,
    ours_ids: Vec<u32>,
    hf_ids: Vec<u32>,
    status: String,
}

#[derive(Default)]
struct FamilyOutcome {
    family: String,
    total_cases: usize,
    failing_cases: usize,
    failure_records: usize,
    errors: Vec<String>,
    summary_path: PathBuf,
}

fn wline(w: &mut BufWriter<File>, v: &Value) {
    serde_json::to_writer(&mut *w, v).expect("write jsonl");
    w.write_all(b"\n").expect("write jsonl");
}

fn create(path: &Path) -> BufWriter<File> {
    BufWriter::new(File::create(path).unwrap_or_else(|e| panic!("create {}: {e}", path.display())))
}

fn run_entry(entry: &ManifestEntry) -> FamilyOutcome {
    install_quiet_hook();
    let mut out = FamilyOutcome {
        family: entry.family.clone(),
        ..Default::default()
    };
    let rdir = report_dir();
    let base = rdir.join(&entry.family);
    let fail_path = base.with_extension("failures.jsonl");
    let min_path = base.with_extension("minimized.jsonl");
    let sum_path = base.with_extension("summary.json");
    out.summary_path = sum_path.clone();
    // `with_extension` on a family containing '.' would clobber; families are
    // plain identifiers, but be explicit:
    assert!(
        !entry.family.contains('.'),
        "family name must not contain '.'"
    );

    let fuzz_cases = env_usize("HIPFIRE_TOKENIZER_FUZZ_CASES", DEFAULT_FUZZ_CASES);
    let seed = env_u64("HIPFIRE_TOKENIZER_FUZZ_SEED", DEFAULT_FUZZ_SEED) ^ fnv1a(&entry.family);
    let min_budget = env_usize("HIPFIRE_TOKENIZER_DIFF_MIN_PROBES", DEFAULT_MIN_PROBES);
    let min_limit = env_usize("HIPFIRE_TOKENIZER_DIFF_MIN_LIMIT", DEFAULT_MIN_LIMIT);
    let print_limit = env_usize("HIPFIRE_TOKENIZER_DIFF_PRINT_LIMIT", DEFAULT_PRINT_LIMIT);

    let header = json!({
        "family": entry.family, "file": entry.file, "md5": entry.md5,
        "source": entry.source, "revision": entry.revision,
        "fuzz_cases": fuzz_cases, "fuzz_seed": seed,
    });

    // --- load (md5 verified); load failures are reported, not hidden ---
    let bytes = match read_verified(entry) {
        Ok(b) => b,
        Err(e) => {
            out.errors.push(e);
            write_summary(
                &sum_path,
                &header,
                &out,
                &BTreeMap::new(),
                &BTreeMap::new(),
                0,
                0,
                0,
            );
            return out;
        }
    };
    let hf = match HfTokenizer::from_bytes(&bytes) {
        Ok(t) => Some(t),
        Err(e) => {
            out.errors
                .push(format!("HF tokenizers failed to load {}: {e}", entry.file));
            None
        }
    };
    let ours = match std::str::from_utf8(&bytes) {
        Ok(s) => match guarded(|| OursTokenizer::from_hf_json(s)) {
            Ok(Ok(t)) => Some(t),
            Ok(Err(e)) => {
                out.errors.push(format!(
                    "hipfire from_hf_json failed to load {}: {e}",
                    entry.file
                ));
                None
            }
            Err(p) => {
                out.errors.push(format!(
                    "hipfire from_hf_json panicked on {}: {p}",
                    entry.file
                ));
                None
            }
        },
        Err(e) => {
            out.errors.push(format!("{} is not UTF-8: {e}", entry.file));
            None
        }
    };
    let (Some(hf), Some(ours)) = (hf, ours) else {
        write_summary(
            &sum_path,
            &header,
            &out,
            &BTreeMap::new(),
            &BTreeMap::new(),
            0,
            0,
            0,
        );
        eprintln!(
            "hf_tokenizer_differential[{}]: LOAD FAILURE: {}",
            entry.family,
            out.errors.join("; ")
        );
        return out;
    };
    let added = added_tokens(&hf);
    let oracle = Oracle { hf, ours };

    // --- cases ---
    let mut cases = load_corpus_cases();
    let n_corpus = cases.len();
    cases.extend(load_prompt_cases());
    let n_prompts = cases.len() - n_corpus;
    let mut n_regressions = 0usize;
    for (fam, c) in load_regression_cases() {
        if fam == entry.family {
            cases.push(c);
            n_regressions += 1;
        }
    }

    let systematic = systematic_cases(&added);
    let mut fz = FuzzGen {
        rng: Rng(seed),
        added: added.clone(),
    };

    let total_planned = cases.len() + fuzz_cases;
    let mut failures_w = create(&fail_path);
    let mut sigs: BTreeMap<String, SigEntry> = BTreeMap::new();
    let mut by_kind: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut decode_checked = 0usize;
    let mut hf_tokens: u64 = 0;

    let mut process = |case_index: usize,
                       source: &str,
                       text: &str,
                       pinned: Option<&[u32]>,
                       out: &mut FamilyOutcome| {
        let mut ev = oracle.evaluate(text, None);
        if let Some(p) = pinned {
            if let Some(f) = oracle.pin_finding(text, p, source) {
                ev.findings.push(f);
            }
        }
        out.total_cases += 1;
        hf_tokens += ev.hf_tokens as u64;
        if ev.decode_checked {
            decode_checked += 1;
        }
        if !ev.findings.is_empty() {
            out.failing_cases += 1;
        }
        for f in ev.findings {
            out.failure_records += 1;
            *by_kind.entry(f.kind).or_default() += 1;
            let mut rec = json!({
                "family": entry.family,
                "case_index": case_index,
                "source": source,
                "kind": f.kind,
                "signature": f.pre_sig,
                "detail": f.detail,
                "text_bytes": text.len(),
                "text_chars": text.chars().count(),
                "text": text,
                "ours_ids": f.ours_ids,
                "hf_ids": f.hf_ids,
            });
            if let (Some(obj), Some(extra)) = (rec.as_object_mut(), f.extra.as_object()) {
                for (k, v) in extra {
                    obj.insert(k.clone(), v.clone());
                }
            }
            wline(&mut failures_w, &rec);
            let e = sigs.entry(f.pre_sig.clone()).or_insert_with(|| SigEntry {
                kind: f.kind,
                count: 0,
                sources: Vec::new(),
                rep_text: text.to_string(),
            });
            e.count += 1;
            e.sources.push(source.to_string());
            if text.len() < e.rep_text.len() {
                e.rep_text = text.to_string();
            }
        }
    };

    for (i, c) in cases.iter().enumerate() {
        process(i, &c.source, &c.text, c.pinned.as_deref(), &mut out);
    }
    for i in 0..fuzz_cases {
        let (label, text) = if i < systematic.len() {
            (format!("fuzz:sys:{i}"), systematic[i].clone())
        } else {
            (format!("fuzz:{i}"), fz.random_case())
        };
        process(cases.len() + i, &label, &text, None, &mut out);
    }
    failures_w.flush().expect("flush failures");
    drop(failures_w);

    // --- minimise each unique signature ---
    let mut order: Vec<(&String, &SigEntry)> = sigs.iter().collect();
    order.sort_by(|a, b| b.1.count.cmp(&a.1.count).then(a.0.cmp(b.0)));
    let mut min_w = create(&min_path);
    let mut roots: BTreeMap<String, RootAgg> = BTreeMap::new();
    for (rank, (sig, e)) in order.iter().enumerate() {
        let (minimized, status, probes) = if e.kind == "regression_pin" {
            // Pinned regression repros are already minimal by construction.
            (None, "pinned_regression", 0)
        } else if rank < min_limit {
            let kind = e.kind;
            let mut pred = |t: &str| !oracle.evaluate(t, Some(kind)).findings.is_empty();
            let m = minimize(&e.rep_text, min_budget, &mut pred);
            (Some(m.text), m.status, m.probes)
        } else {
            (None, "skipped_limit", 0)
        };
        let (root_sig, repro, ours_ids, hf_ids) = match &minimized {
            Some(t) => {
                let ev = oracle.evaluate(t, Some(e.kind));
                let (o, h) = ev
                    .findings
                    .first()
                    .map(|f| {
                        (
                            f.ours_ids.clone().unwrap_or_default(),
                            f.hf_ids.clone().unwrap_or_default(),
                        )
                    })
                    .unwrap_or_default();
                (format!("{}|{}", e.kind, shape(t)), t.clone(), o, h)
            }
            None => (
                format!("unminimized|{sig}"),
                e.rep_text.clone(),
                Vec::new(),
                Vec::new(),
            ),
        };
        let agg = roots.entry(root_sig.clone()).or_default();
        if agg.cases == 0 || repro.len() < agg.repro.len() {
            agg.kind = e.kind.to_string();
            agg.repro = repro.clone();
            agg.ours_ids = ours_ids.clone();
            agg.hf_ids = hf_ids.clone();
            agg.status = status.to_string();
        }
        agg.cases += e.count;
        agg.pre_sigs += 1;
        wline(
            &mut min_w,
            &json!({
                "family": entry.family,
                "kind": e.kind,
                "signature": sig,
                "root_signature": root_sig,
                "cases": e.count,
                "case_sources": e.sources,
                "original_bytes": e.rep_text.len(),
                "original_text": e.rep_text,
                "minimized_text": minimized,
                "minimized_codepoints": minimized.as_ref().map(|t| t.chars().map(|c| format!("U+{:04X}", c as u32)).collect::<Vec<_>>()),
                "ours_ids": ours_ids,
                "hf_ids": hf_ids,
                "ours_tokens": ours_ids.iter().map(|&i| oracle.tok(i)).collect::<Vec<_>>(),
                "hf_tokens": hf_ids.iter().map(|&i| oracle.tok(i)).collect::<Vec<_>>(),
                "probes": probes,
                "status": status,
            }),
        );
    }
    min_w.flush().expect("flush minimized");
    drop(min_w);

    write_summary(
        &sum_path,
        &header,
        &out,
        &roots,
        &by_kind
            .iter()
            .map(|(k, v)| ((*k).to_string(), *v))
            .collect::<BTreeMap<_, _>>(),
        decode_checked,
        hf_tokens,
        sigs.len(),
    );

    // --- console report ---
    eprintln!(
        "hf_tokenizer_differential[{fam}]: {total}/{planned} cases ({n_corpus} corpus + {n_prompts} prompts + {n_regressions} pinned regressions + {fuzz_cases} adversarial, seed {seed:#x}), {hf_tokens} HF tokens, {decode_checked} decode-checked | failing cases {fail} ({recs} records) | unique signatures {nsig} | root signatures {nroot} | kinds {kinds:?}\n  reports: {fp} | {mp} | {sp}",
        fam = entry.family,
        total = out.total_cases,
        planned = total_planned,
        fail = out.failing_cases,
        recs = out.failure_records,
        nsig = sigs.len(),
        nroot = roots.len(),
        kinds = by_kind,
        fp = fail_path.display(),
        mp = min_path.display(),
        sp = sum_path.display(),
    );
    let mut root_order: Vec<(&String, &RootAgg)> = roots.iter().collect();
    root_order.sort_by(|a, b| b.1.cases.cmp(&a.1.cases).then(a.0.cmp(b.0)));
    for (i, (root, agg)) in root_order.iter().enumerate() {
        if i >= print_limit {
            eprintln!(
                "  … {} more root signatures in {} (raise HIPFIRE_TOKENIZER_DIFF_PRINT_LIMIT)",
                root_order.len() - print_limit,
                min_path.display()
            );
            break;
        }
        eprintln!(
            "  [{root}] {} cases, {} sigs, {} ({}): repro {:?}\n      ours={:?}\n      hf  ={:?}",
            agg.cases,
            agg.pre_sigs,
            agg.kind,
            agg.status,
            clip(&agg.repro, 200),
            clip(&format!("{:?}", agg.ours_ids), 200),
            clip(&format!("{:?}", agg.hf_ids), 200),
        );
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn write_summary(
    path: &Path,
    header: &Value,
    out: &FamilyOutcome,
    roots: &BTreeMap<String, RootAgg>,
    by_kind: &BTreeMap<String, usize>,
    decode_checked: usize,
    hf_tokens: u64,
    unique_sigs: usize,
) {
    let mut root_list: Vec<(&String, &RootAgg)> = roots.iter().collect();
    root_list.sort_by(|a, b| b.1.cases.cmp(&a.1.cases).then(a.0.cmp(b.0)));
    let v = json!({
        "run": header,
        "total_cases": out.total_cases,
        "failing_cases": out.failing_cases,
        "failure_records": out.failure_records,
        "decode_checked_cases": decode_checked,
        "hf_tokens": hf_tokens,
        "failures_by_kind": by_kind,
        "unique_signatures": unique_sigs,
        "root_signatures": root_list.iter().map(|(r, a)| json!({
            "root_signature": r,
            "kind": a.kind,
            "cases": a.cases,
            "signatures": a.pre_sigs,
            "minimal_repro": a.repro,
            "ours_ids": a.ours_ids,
            "hf_ids": a.hf_ids,
            "status": a.status,
        })).collect::<Vec<_>>(),
        "errors": out.errors,
    });
    std::fs::write(path, serde_json::to_vec_pretty(&v).expect("summary json"))
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn assert_outcomes(outcomes: &[FamilyOutcome]) {
    let mut msgs = Vec::new();
    for o in outcomes {
        if !o.errors.is_empty() {
            msgs.push(format!("{}: {}", o.family, o.errors.join("; ")));
        }
        if o.failure_records > 0 {
            msgs.push(format!(
                "{}: {} failing cases / {} total ({} failure records); full reports: {}",
                o.family,
                o.failing_cases,
                o.total_cases,
                o.failure_records,
                o.summary_path.display()
            ));
        }
    }
    assert!(
        msgs.is_empty(),
        "hipfire tokenizer differs from the HF reference:\n{}",
        msgs.join("\n")
    );
}

fn run_family_test(family: &str) {
    let entries: Vec<ManifestEntry> = load_manifest()
        .into_iter()
        .filter(|e| e.family == family)
        .collect();
    assert!(
        !entries.is_empty(),
        "family `{family}` is not in tests/fixtures/hf_tokenizer_differential/manifest.json"
    );
    let outcomes: Vec<FamilyOutcome> = entries.iter().map(run_entry).collect();
    assert_outcomes(&outcomes);
}

// ──────────────────────────────────── tests ────────────────────────────────

macro_rules! family_tests {
    ($($name:ident),* $(,)?) => {
        /// Families with a dedicated `#[test]`.
        const DEDICATED_FAMILIES: &[&str] = &[$(stringify!($name)),*];
        $(
            #[test]
            fn $name() {
                run_family_test(stringify!($name));
            }
        )*
    };
}

family_tests!(
    qwen3,
    qwen35,
    qwen36,
    qwen38,
    qwen38_flash_next,
    dots_ocr,
    deepseek4,
    minimax,
    minimax27,
    lfm2,
    lfm25,
    llama_bpe,
    muse_glimmer,
    qwen25,
    north_mini_code,
    maple,
    ornith,
    vibethinker,
    lfm25_moe,
);

/// Manifest hygiene: every file exists with the recorded MD5, family names are
/// unique, every dedicated family is present, the corpus/prompts load.
#[test]
fn manifest_integrity() {
    let manifest = load_manifest();
    let mut errors = Vec::new();
    let mut seen = BTreeSet::new();
    for e in &manifest {
        if !seen.insert(e.family.clone()) {
            errors.push(format!("duplicate family `{}`", e.family));
        }
        if e.source.is_empty() || e.revision.is_empty() {
            errors.push(format!("{}: empty source/revision", e.family));
        }
        if let Err(msg) = read_verified(e) {
            errors.push(msg);
        }
    }
    for fam in DEDICATED_FAMILIES {
        if !seen.contains(*fam) {
            errors.push(format!("dedicated test `{fam}` has no manifest entry"));
        }
    }
    for (fam, _) in load_regression_cases() {
        if !seen.contains(&fam) {
            errors.push(format!("regressions.jsonl names unknown family `{fam}`"));
        }
    }
    assert_eq!(load_corpus_cases().len(), CORPUS_CASES);
    assert!(!load_prompt_cases().is_empty());
    assert!(
        errors.is_empty(),
        "manifest problems:\n{}",
        errors.join("\n")
    );
}

/// Runs every manifest family that has no dedicated `#[test]` above, so adding
/// a manifest entry can never silently skip the comparison.
#[test]
fn unlisted_manifest_families() {
    let entries: Vec<ManifestEntry> = load_manifest()
        .into_iter()
        .filter(|e| !DEDICATED_FAMILIES.contains(&e.family.as_str()))
        .collect();
    let outcomes: Vec<FamilyOutcome> = entries.iter().map(run_entry).collect();
    assert_outcomes(&outcomes);
}
