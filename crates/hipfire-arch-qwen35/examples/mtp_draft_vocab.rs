// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Build (or check) the compressed MTP draft vocabulary of
//! `hipfire_arch_qwen35::draft_vocab` from files committed to this repository.
//! CPU only.
//!
//! ```text
//! cargo run --release -p hipfire-arch-qwen35 --example mtp_draft_vocab -- \
//!     [--repo <root>] [--k 65536] [--prior P] [--pool W] [--out <file.bin>] \
//!     [--model <model.hfq>]... [--eval-text <file>]... [--eval-json <file>]...
//! ```
//!
//! Without `--out` it rebuilds the subset from the committed sources and
//! compares it with the embedded `data/qwen35-draft-vocab-65536.bin`. With
//! `--out` it writes the file (that is how the committed file was made, with
//! the defaults). `--model` reports whether a model's tokenizer matches the
//! subset (the runtime check). `--eval-text` (plain text) and `--eval-json`
//! (any JSON; every string under a `text`, `content` or `reasoning_content` key) report the share of their
//! tokens inside the subset: an out-of-subset token is a draft the verify
//! must reject, so this bounds the tau cost. `--subset-json <file>` scores
//! another subset (`{"draft_to_full": [...]}`) on the same eval files.

use hipfire_arch_qwen35::draft_vocab::{
    committed_sources, rank_draft_vocab, tokenizer_identity, DraftVocab, RankParams,
    QWEN35_DRAFT_VOCAB, QWEN35_LM_HEAD_ROWS, SOURCE_IDS_JSONL, SOURCE_TEXTS, SOURCE_TOKENIZER,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};

fn values(args: &[String], key: &str) -> Vec<String> {
    args.windows(2)
        .filter(|w| w[0] == key)
        .map(|w| w[1].clone())
        .collect()
}

fn json_texts(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::Object(m) => {
            for (k, x) in m {
                match x {
                    serde_json::Value::String(s)
                        if matches!(k.as_str(), "text" | "content" | "reasoning_content") =>
                    {
                        out.push(s.clone())
                    }
                    _ => json_texts(x, out),
                }
            }
        }
        serde_json::Value::Array(a) => a.iter().for_each(|x| json_texts(x, out)),
        _ => {}
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let repo = values(&args, "--repo")
        .pop()
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."));
    let k: usize = values(&args, "--k")
        .pop()
        .map(|v| v.parse().expect("--k"))
        .unwrap_or(hipfire_arch_qwen35::draft_vocab::QWEN35_DRAFT_VOCAB_K);
    let read = |rel: &str| {
        std::fs::read_to_string(repo.join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"))
    };
    let t0 = std::time::Instant::now();
    let tok = Tokenizer::from_hf_json(&read(SOURCE_TOKENIZER)).expect("tokenizer.json");
    let texts = SOURCE_TEXTS.map(read);
    let sources = committed_sources(&tok, QWEN35_LM_HEAD_ROWS, &texts, &read(SOURCE_IDS_JSONL))
        .expect("sources");
    let mut params = RankParams::COMMITTED;
    if let Some(v) = values(&args, "--prior").pop() {
        params.merge_rank_prior = v.parse().expect("--prior");
    }
    if let Some(v) = values(&args, "--pool").pop() {
        params.variant_pool = v.parse().expect("--pool");
    }
    let ids = rank_draft_vocab(&tok, QWEN35_LM_HEAD_ROWS, &sources, k, params);
    let built = DraftVocab {
        rows: QWEN35_LM_HEAD_ROWS,
        tokenizer_sha256: tokenizer_identity(&tok),
        ids,
    };
    let runs = 1 + built.ids.windows(2).filter(|w| w[1] != w[0] + 1).count();
    eprintln!(
        "built k={} of {} rows in {:.1}s; {runs} row runs",
        built.ids.len(),
        built.rows,
        t0.elapsed().as_secs_f64(),
    );
    let names = SOURCE_TEXTS.iter().chain([&SOURCE_IDS_JSONL]);
    for (name, src) in names.zip(&sources) {
        let n = src.total().max(1);
        let inside: u64 = built.ids.iter().map(|&i| src.counts[i as usize]).sum();
        let distinct = src.counts.iter().filter(|&&c| c > 0).count();
        eprintln!(
            "  source {name}: {n} tokens, {:.3}% inside, {distinct} distinct ids",
            100.0 * inside as f64 / n as f64
        );
    }
    let bytes = built.to_bytes();
    match values(&args, "--out").pop() {
        Some(out) => {
            std::fs::write(&out, &bytes).expect("write --out");
            eprintln!("wrote {out} ({} B)", bytes.len());
        }
        None => {
            let same = bytes.as_slice() == QWEN35_DRAFT_VOCAB;
            eprintln!(
                "embedded subset {}",
                if same {
                    "MATCHES the rebuild"
                } else {
                    "DIFFERS from the rebuild"
                }
            );
            if !same {
                std::process::exit(1);
            }
        }
    }

    // `--subset-json` evaluates another subset (a `build_mtp_vocab_sidecar.py`
    // style `{"draft_to_full": [...]}` file) instead of the built one.
    let eval_ids: Vec<u32> = match values(&args, "--subset-json").pop() {
        Some(p) => {
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&p).expect("subset json"))
                    .expect("json");
            let ids: Vec<u32> = v["draft_to_full"]
                .as_array()
                .expect("draft_to_full")
                .iter()
                .map(|x| x.as_u64().expect("id") as u32)
                .collect();
            eprintln!(
                "evaluating {p} ({} ids) instead of the built subset",
                ids.len()
            );
            ids
        }
        None => built.ids.clone(),
    };
    let mut inside = vec![false; built.rows];
    eval_ids.iter().for_each(|&i| inside[i as usize] = true);
    let dump = args.iter().any(|a| a == "--dump-missing");
    let coverage = |name: &str, texts: &[String]| {
        let (mut n, mut hit) = (0u64, 0u64);
        let mut missing = std::collections::BTreeMap::<u32, u64>::new();
        for t in texts {
            for id in tok.encode(t) {
                n += 1;
                let ok = inside.get(id as usize).copied().unwrap_or(false);
                hit += u64::from(ok);
                if !ok {
                    *missing.entry(id).or_default() += 1;
                }
            }
        }
        if dump {
            let mut m: Vec<(u64, u32)> = missing.iter().map(|(&i, &c)| (c, i)).collect();
            m.sort_by(|a, b| b.cmp(a));
            for (c, i) in m.iter().take(40) {
                eprintln!("    missing id {i} x{c} {:?}", tok.decode(&[*i]));
            }
        }
        eprintln!(
            "  eval {name}: {} texts, {n} tokens, {:.3}% inside, {} outside",
            texts.len(),
            100.0 * hit as f64 / n.max(1) as f64,
            n - hit
        );
    };
    for p in values(&args, "--eval-text") {
        coverage(&p, &[std::fs::read_to_string(&p).expect("eval text")]);
    }
    for p in values(&args, "--eval-json") {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&p).expect("eval json")).expect("json");
        let mut texts = Vec::new();
        json_texts(&v, &mut texts);
        coverage(&p, &texts);
    }
    for p in values(&args, "--model") {
        let hfq = HfqFile::open(Path::new(&p)).expect("open model");
        let mt = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("model tokenizer");
        let same = tokenizer_identity(&mt) == built.tokenizer_sha256;
        eprintln!(
            "  model {p}: tokenizer {} the subset's (max id {:?}, vocab {})",
            if same { "matches" } else { "does NOT match" },
            mt.max_token_id(),
            mt.vocab_size()
        );
    }
}
