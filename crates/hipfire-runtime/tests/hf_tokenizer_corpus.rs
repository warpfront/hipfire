// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact token-ID equality of `hipfire_runtime::tokenizer::Tokenizer::encode`
//! against the HuggingFace reference tokenizer, over a committed corpus.
//!
//! Everything lives in `tests/fixtures/hf_tokenizer_corpus/` and is read
//! relative to `CARGO_MANIFEST_DIR`: the reference `tokenizer.json`
//! (md5 `085e10165f24499bb9d8fcee7e17f9ee`), `cases.jsonl` (rendered input text
//! + HF reference IDs) and `manifest.json` (provenance + per-group counts). No
//! network, no Python, no files outside the repository.
//!
//! Groups: `hermes` (all 138 Hermes request bodies, chat-template rendered),
//! `tc55` (complete TC s123 request corpus), `tc14` (swarm TC captures
//! 00013..00026), `prompts` (`benchmarks/prompts/*.txt`) and `synthetic`
//! (tabs, mixed indentation, trailing newline+space, unicode whitespace,
//! combining marks, digits, contractions, special tokens, seeded fuzz).
//!
//! Inputs are pinned by SHA-256 (the committed `text_sha256` per case and the
//! tokenizer's SHA-256 in the manifest), so a corrupted or hand-edited
//! fixture fails loudly instead of silently re-baselining. The md5s in the
//! fixtures are provenance only. Regenerate offline with
//! `scripts/gen_hf_tokenizer_corpus.py` (see its `--help`).
//!
//! ```text
//! cargo test --release -p hipfire-runtime --test hf_tokenizer_corpus
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

use hipfire_runtime::tokenizer::Tokenizer;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Groups whose case count is part of the contract (the corpus must not be
/// silently shrunk): (group, expected cases).
const REQUIRED_GROUPS: [(&str, usize); 3] = [("hermes", 138), ("tc55", 55), ("tc14", 14)];

/// Max failing cases itemised in the assertion message.
const MAX_REPORTED: usize = 20;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("hf_tokenizer_corpus")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn str_field<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("fixture field `{key}` missing or not a string"))
}

fn u64_field(v: &Value, key: &str) -> u64 {
    v.get(key)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("fixture field `{key}` missing or not an integer"))
}

/// First index where the two ID streams differ (or the shorter length).
fn first_divergence(a: &[u32], b: &[u32]) -> usize {
    a.iter()
        .zip(b.iter())
        .position(|(x, y)| x != y)
        .unwrap_or_else(|| a.len().min(b.len()))
}

#[derive(Default)]
struct GroupTally {
    cases: usize,
    tokens: u64,
    text_bytes: u64,
    failed: usize,
    count_differs: usize,
}

#[test]
fn encode_matches_hf_reference_exactly() {
    let dir = fixture_dir();

    let manifest: Value = serde_json::from_slice(
        &std::fs::read(dir.join("manifest.json")).expect("read manifest.json"),
    )
    .expect("parse manifest.json");

    // --- reference tokenizer: pinned by sha256 ---
    let tok_bytes = std::fs::read(dir.join("tokenizer.json")).expect("read tokenizer.json");
    let want_tok_sha = manifest["reference"]["tokenizer_json_sha256"]
        .as_str()
        .expect("manifest.reference.tokenizer_json_sha256");
    assert_eq!(
        sha256_hex(&tok_bytes),
        want_tok_sha,
        "committed tokenizer.json does not match the manifest (HF md5 {})",
        manifest["reference"]["tokenizer_json_md5"]
    );
    let tok_json = std::str::from_utf8(&tok_bytes).expect("tokenizer.json is UTF-8");
    let tokenizer = Tokenizer::from_hf_json(tok_json).expect("Tokenizer::from_hf_json");

    // --- cases ---
    let cases_raw = std::fs::read_to_string(dir.join("cases.jsonl")).expect("read cases.jsonl");

    let mut groups: BTreeMap<String, GroupTally> = BTreeMap::new();
    let mut failures: Vec<String> = Vec::new();
    let mut seen_ids = std::collections::BTreeSet::new();
    let mut total_cases = 0usize;
    let mut total_tokens = 0u64;

    for (lineno, line) in cases_raw.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let case: Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("cases.jsonl line {}: {e}", lineno + 1));
        let id = str_field(&case, "id");
        assert!(seen_ids.insert(id.to_string()), "duplicate case id {id}");
        let group = str_field(&case, "group");
        let text = str_field(&case, "text");
        let want: Vec<u32> = case["ids"]
            .as_array()
            .unwrap_or_else(|| panic!("{id}: `ids` missing"))
            .iter()
            .map(|x| x.as_u64().expect("id is integer") as u32)
            .collect();

        // Fixture integrity: input and expected-ID bookkeeping must be intact.
        assert_eq!(
            sha256_hex(text.as_bytes()),
            str_field(&case, "text_sha256"),
            "{id}: committed text does not match its recorded sha256"
        );
        assert_eq!(
            text.len() as u64,
            u64_field(&case, "text_bytes"),
            "{id}: text_bytes mismatch"
        );
        assert_eq!(
            want.len() as u64,
            u64_field(&case, "n_tokens"),
            "{id}: n_tokens != ids.len()"
        );

        let got = tokenizer.encode(text);

        let tally = groups.entry(group.to_string()).or_default();
        tally.cases += 1;
        tally.tokens += want.len() as u64;
        tally.text_bytes += text.len() as u64;
        total_cases += 1;
        total_tokens += want.len() as u64;

        if got != want {
            tally.failed += 1;
            if got.len() != want.len() {
                tally.count_differs += 1;
            }
            if failures.len() < MAX_REPORTED {
                let at = first_divergence(&got, &want);
                let lo = at.saturating_sub(3);
                let got_hi = (at + 4).min(got.len());
                let want_hi = (at + 4).min(want.len());
                failures.push(format!(
                    "{id}: ours={} hf={} tokens, first divergence at {at}\n    ours[{lo}..{got_hi}]={:?} {:?}\n    hf  [{lo}..{want_hi}]={:?} {:?}",
                    got.len(),
                    want.len(),
                    &got[lo.min(got.len())..got_hi],
                    tokenizer.decode(&got[lo.min(got.len())..got_hi]),
                    &want[lo.min(want.len())..want_hi],
                    tokenizer.decode(&want[lo.min(want.len())..want_hi]),
                ));
            }
        }
    }

    // --- corpus shape must match the committed manifest (no silent shrink) ---
    assert_eq!(
        total_cases as u64,
        u64_field(&manifest, "n_cases"),
        "case count != manifest.n_cases"
    );
    assert_eq!(
        total_tokens,
        u64_field(&manifest, "n_tokens"),
        "token count != manifest.n_tokens"
    );
    let mgroups = manifest["groups"].as_object().expect("manifest.groups");
    assert_eq!(
        mgroups.len(),
        groups.len(),
        "group set differs from the manifest"
    );
    for (name, tally) in &groups {
        let m = mgroups
            .get(name)
            .unwrap_or_else(|| panic!("group `{name}` absent from manifest"));
        assert_eq!(
            tally.cases as u64,
            u64_field(m, "n_cases"),
            "{name}: n_cases"
        );
        assert_eq!(tally.tokens, u64_field(m, "n_tokens"), "{name}: n_tokens");
        assert_eq!(
            tally.text_bytes,
            u64_field(m, "n_text_bytes"),
            "{name}: n_text_bytes"
        );
    }
    for (name, want_n) in REQUIRED_GROUPS {
        let have = groups.get(name).map_or(0, |g| g.cases);
        assert_eq!(
            have, want_n,
            "required group `{name}` must have {want_n} cases"
        );
    }

    // --- the actual contract: exact ID equality ---
    let summary: Vec<String> = groups
        .iter()
        .map(|(n, g)| {
            format!(
                "{n}: {}/{} exact ({} count-differ)",
                g.cases - g.failed,
                g.cases,
                g.count_differs
            )
        })
        .collect();
    eprintln!(
        "hf_tokenizer_corpus: {total_cases} cases, {total_tokens} tokens | {}",
        summary.join(" | ")
    );
    let failed: usize = groups.values().map(|g| g.failed).sum();
    assert!(
        failed == 0,
        "{failed}/{total_cases} cases differ from the HF reference ({}):\n{}",
        summary.join(" | "),
        failures.join("\n")
    );
}
