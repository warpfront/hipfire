// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.
//! Gemma-4 logit oracle.
//!
//! Runs `forward::decode_step` tokenwise over a token-id list and dumps the
//! final-position top-k logits as JSON. Token IDs are read from a file so the
//! HF reference (`scripts/oracle_gemma4.py`) and hipfire compare BYTE-IDENTICAL
//! inputs (no tokenizer differences enter).
//!
//! The state mirrors the production bundle: Q8 sliding tier and an asym3 full
//! tier (Q8 with `HIPFIRE_ORACLE_KV_Q8=1`), both position-indexed over
//! `max_seq`. This example bypasses the production admission/control plane.
//!
//! Usage: gemma4_oracle <model.hfq> <ids_file> [out.json]

use hipfire_arch_gemma4::gemma4::FullKvTier;
use hipfire_arch_gemma4::{forward, Gemma4Config, Gemma4State, Gemma4Weights};
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: gemma4_oracle <model.hfq> <ids_file> [out.json]");
        std::process::exit(1);
    }
    let model_path = Path::new(&args[1]);
    let ids: Vec<u32> = std::fs::read_to_string(&args[2])
        .expect("read ids file")
        .replace(',', " ")
        .split_whitespace()
        .map(|x| x.parse().expect("parse id"))
        .collect();
    assert!(!ids.is_empty(), "no ids");

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let hfq = HfqFile::open(model_path).expect("open model");
    let config = Gemma4Config::from_hfq(&hfq).expect("config");
    let weights = Gemma4Weights::load(&hfq, &config, &mut gpu).expect("load weights");
    let max_seq = ids.len().max(128);
    // The unified state has no F32 KV tier; Q8 is the closest exact-write tier.
    let tier = if std::env::var("HIPFIRE_ORACLE_KV_Q8").ok().as_deref() == Some("1") {
        FullKvTier::Q8
    } else {
        FullKvTier::LegacyAsym3
    };
    eprintln!("oracle: max_seq={max_seq} sliding=q8 full={tier:?}");
    let mut state =
        Gemma4State::new_with_full_tier(&mut gpu, &config, max_seq, tier).expect("state");

    let mut logits = Vec::new();
    for (i, &tok) in ids.iter().enumerate() {
        logits = forward::decode_step(&config, &weights, &mut state, &mut gpu, tok, i as u32)
            .unwrap_or_else(|e| panic!("decode_step at pos {i}: {e:?}"));
    }
    let vocab = config.vocab_size.min(logits.len());
    let mut idx: Vec<usize> = (0..vocab).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let topk = 20.min(vocab);
    let top: Vec<(usize, f32)> = idx[..topk].iter().map(|&i| (i, logits[i])).collect();

    eprintln!(
        "argmax: {} top5: {:?}",
        top[0].0,
        top.iter()
            .take(5)
            .map(|(i, v)| (*i, (v * 1e4).round() / 1e4))
            .collect::<Vec<_>>()
    );
    let json = format!(
        "{{\"n_ids\":{},\"logit_argmax\":{},\"logits_topk\":[{}]}}",
        ids.len(),
        top[0].0,
        top.iter()
            .map(|(i, v)| format!("[{},{:.4}]", i, v))
            .collect::<Vec<_>>()
            .join(",")
    );
    if let Some(out) = args.get(3) {
        std::fs::write(out, &json).expect("write out");
        eprintln!("wrote {}", out);
    }
    println!("{json}");
}
