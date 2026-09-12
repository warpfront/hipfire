// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.
//! Gemma-4 logit oracle.
//!
//! Runs `gemma4::forward_scratch` tokenwise over a token-id list and dumps the
//! final-position top-k logits as JSON. Token IDs are read from a file so the
//! HF reference (`scripts/oracle_gemma4.py`) and hipfire compare BYTE-IDENTICAL
//! inputs (no tokenizer differences enter).
//!
//! KV geometry mirrors the production lowered bundle (`load_gemma4_bundle`):
//! sliding Q8 ring over the actual sliding layers with physical capacity
//! `min(sliding_window, max_seq)`, full asym3 over the actual full layers with
//! capacity `max_seq`, scratch sized from the same `max_seq`. This example
//! still bypasses the production admission/control plane — it calls
//! `forward_scratch` directly with no daemon sliding-window guard.
//!
//! Usage: gemma4_oracle <model.hfq> <ids_file> [out.json]
//!
//! Common short-token case (fewer than 128 ids, e.g. the 15-token parity
//! prompt): `max_seq` floors at 128 (one flash tile) and both caches cover
//! every requested position, so no ring wrap occurs. Rollover case: only when
//! the id count exceeds the sliding physical capacity (`min(sliding_window,
//! max_seq)`) do later positions wrap the Q8 ring; shorter runs never wrap.

use hipfire_arch_gemma4::lowered::{self as gemma4, Gemma4Scratch};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
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
    let mut hfq = HfqFile::open(model_path).expect("open model");
    let config = gemma4::config_from_hfq(&hfq).expect("config_from_hfq");
    let weights = gemma4::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");
    // Logical authority: cover every requested position, floored at one flash
    // tile (128) so the scratch partials stay scratch-compatible.
    let max_seq = ids.len().max(128);
    // Actual per-type layer counts, mirroring the production bundle — NOT
    // config.n_layers for both caches.
    let (n_sliding_layers, n_full_layers) =
        config
            .layer_types
            .iter()
            .fold((0, 0), |(sliding, full), layer_type| match layer_type {
                gemma4::LayerType::Sliding => (sliding + 1, full),
                gemma4::LayerType::Full => (sliding, full + 1),
            });
    // Bounded sliding physical capacity: the ring holds at most
    // min(sliding_window, max_seq) slots (production passes sliding_window
    // against a much larger daemon max_seq; here max_seq can be smaller).
    let sliding_cap = config.sliding_window.min(max_seq);
    eprintln!(
        "oracle: max_seq={max_seq} sliding=q8-ring({n_sliding_layers} layers,cap={sliding_cap}) full=asym3({n_full_layers} layers,cap={max_seq}) wrap={}",
        ids.len() > sliding_cap
    );
    let mut scratch = Gemma4Scratch::new(&mut gpu, &config, max_seq).expect("scratch");
    gemma4::init_scratch_constants(&mut gpu, &scratch, config.full_head_dim)
        .expect("init_scratch_constants");
    let mut kv_sliding = KvCache::new_gpu_q8_capped(
        &mut gpu,
        n_sliding_layers,
        config.sliding_n_kv_heads,
        config.sliding_head_dim,
        max_seq,
        sliding_cap,
    )
    .expect("kv sliding alloc");
    let mut kv_full = if std::env::var("HIPFIRE_ORACLE_KV_F32").ok().as_deref() == Some("1") {
        eprintln!("oracle: FULL KV = F32 (HIPFIRE_ORACLE_KV_F32=1)");
        KvCache::new_gpu(
            &mut gpu,
            n_full_layers,
            config.full_n_kv_heads,
            config.full_head_dim,
            max_seq,
        )
        .expect("kv full alloc f32")
    } else {
        KvCache::new_gpu_asym3_gemma4(
            &mut gpu,
            n_full_layers,
            config.full_n_kv_heads,
            config.full_head_dim,
            max_seq,
        )
        .expect("kv full alloc")
    };

    for (i, &tok) in ids.iter().enumerate() {
        gemma4::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            tok,
            i,
            &mut kv_sliding,
            &mut kv_full,
            &mut scratch,
        )
        .unwrap_or_else(|e| panic!("forward_scratch at pos {i}: {e:?}"));
    }

    let logits = gpu.download_f32(&scratch.logits).expect("download logits");
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
