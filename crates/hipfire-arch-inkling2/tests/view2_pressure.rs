// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Upstream #1192 repro: under OOM-retry pressure the pool can hand a
//! LARGER buffer to an exact-size request; the old exact-size `view2`
//! assert killed the daemon (`view extent`: 15704 B vs 12288 B).
//!
//! Kaden's sequence, one process (one pool): two 128-token requests, then
//! a 4096-token prompt, each driven token-by-token through
//! `forward_token_gpu` on a fresh session. Bar: the run completes — clean
//! `Ok` or clean `Err` (503) — and never panics. A panic fails the test
//! process, so reaching the final eprintln IS the gate.
//!
//! Second gate: two identical 128-token sessions are bit-identical,
//! pinning that the `>=` prefix views did not move normal-path numerics.
//!
//! ```powershell
//! $env:HIPFIRE_HOME = 'D:\hipfire-models\inkling2-home'
//! $env:HIPFIRE_GRAPH = '0'
//! cargo test -p hipfire-arch-inkling2 --test view2_pressure -- --ignored --nocapture
//! ```
//!
//! The artifact path defaults to the Windows layout below; set
//! `INKLING2_MQ4` to point at it elsewhere.

use hipfire_arch_inkling2::arch_model::{Inkling2Bundle, Inkling2State};
use hipfire_arch_inkling2::config::Inkling2Config;
use hipfire_arch_inkling2::forward_token_gpu;
use hipfire_arch_inkling2::gpu::{upload, Inkling2GpuState, Inkling2GpuWeights};
use hipfire_arch_inkling2::weights::Inkling2Weights;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

const ART: &str = r"D:\hipfire-models\inkling2-mini-mq4.hfq";

fn ensure_home() {
    if cfg!(windows) && std::env::var("HIPFIRE_HOME").is_err() {
        std::env::set_var("HIPFIRE_HOME", r"D:\hipfire-models\inkling2-home");
    }
    assert_eq!(
        std::env::var("HIPFIRE_GRAPH").as_deref(),
        Ok("0"),
        "gates require HIPFIRE_GRAPH=0"
    );
}

fn open_artifact() -> (String, Inkling2Config, Inkling2Weights) {
    let art = match std::env::var("INKLING2_MQ4") {
        Ok(p) => p,
        Err(_) if cfg!(windows) => ART.to_string(),
        Err(_) => panic!("set INKLING2_MQ4 to the inkling2-mini-mq4.hfq path"),
    };
    let hfq = HfqFile::open(std::path::Path::new(&art))
        .unwrap_or_else(|e| panic!("open artifact {art}: {e}"));
    let cfg = Inkling2Config::from_metadata_json(&hfq.metadata_json).expect("parse config");
    let weights = Inkling2Weights::load(&hfq, &cfg).expect("index weights");
    (art, cfg, weights)
}

fn make_prompt(n: usize, vocab: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 7919) % vocab) as u32).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

fn make_bundle(
    cfg: Inkling2Config,
    weights: Inkling2Weights,
    gw: Inkling2GpuWeights,
    gs: Inkling2GpuState,
) -> Inkling2Bundle {
    Inkling2Bundle {
        config: cfg,
        weights,
        state: Inkling2State { seq_pos: 0 },
        eos_tok: 200006,
        gpu_weights: Some(gw),
        gpu_state: Some(gs),
    }
}

fn reset(bundle: &mut Inkling2Bundle, gpu: &mut Gpu) {
    bundle
        .gpu_state
        .as_mut()
        .unwrap()
        .reset_session_state(gpu)
        .expect("reset");
    bundle.state.seq_pos = 0;
}

/// Kaden's sequence in one pool: 128, 128, then 4096 tokens. Completes
/// clean (Ok or Err) — a view-cut panic would abort the test process.
#[test]
#[ignore = "needs the 14 GB MQ4 artifact + GPU"]
fn kaden_sequence_no_panic() {
    let (art, cfg, weights) = open_artifact();
    eprintln!("artifact: {art}");
    let vocab = cfg.unpadded_vocab_size;
    let mut gpu = {
        ensure_home();
        Gpu::init().expect("gpu init")
    };
    // Same session/pool for all three, like one daemon serving in order.
    let (gw, gs) = upload(&weights, &cfg, &mut gpu, 4608).expect("gpu upload");
    let mut bundle = make_bundle(cfg, weights, gw, gs);
    for (i, n) in [128usize, 128, 4096].iter().enumerate() {
        reset(&mut bundle, &mut gpu);
        let prompt = make_prompt(*n, vocab);
        let mut ok = true;
        let mut last_argmax = 0;
        for (pos, &tok) in prompt.iter().enumerate() {
            match forward_token_gpu(&mut bundle, &mut gpu, tok, pos) {
                Ok(logits) => {
                    assert!(
                        logits.iter().all(|v| v.is_finite()),
                        "req{i} pos {pos}: non-finite logits"
                    );
                    last_argmax = argmax(&logits);
                }
                Err(e) => {
                    eprintln!("req{i}: n={n} clean error at pos {pos} (no panic): {e}");
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            eprintln!("req{i}: n={n} ok argmax={last_argmax}");
        }
    }
    eprintln!("kaden_sequence_no_panic: PASS (process survived)");
}

/// Isolator: a single 4096-token session on a fresh pool (no prior
/// requests). Distinguishes pool-pressure failure (only after the 2×128)
/// from position-dependent failure (also fails here).
#[test]
#[ignore = "needs the 14 GB MQ4 artifact + GPU"]
fn single_4096_no_panic() {
    let (art, cfg, weights) = open_artifact();
    eprintln!("artifact: {art}");
    let vocab = cfg.unpadded_vocab_size;
    let mut gpu = {
        ensure_home();
        Gpu::init().expect("gpu init")
    };
    let (gw, gs) = upload(&weights, &cfg, &mut gpu, 4608).expect("gpu upload");
    let mut bundle = make_bundle(cfg, weights, gw, gs);
    reset(&mut bundle, &mut gpu);
    let prompt = make_prompt(4096, vocab);
    for (pos, &tok) in prompt.iter().enumerate() {
        match forward_token_gpu(&mut bundle, &mut gpu, tok, pos) {
            Ok(logits) => {
                assert!(
                    logits.iter().all(|v| v.is_finite()),
                    "pos {pos}: non-finite logits"
                );
            }
            Err(e) => {
                eprintln!("single4096: clean error at pos {pos} (no panic): {e}");
                return;
            }
        }
    }
    eprintln!("single_4096_no_panic: PASS (all 4096 ok)");
}

/// Normal-path numerics anchor: two identical 128-token sessions are
/// bit-identical. Proves the `>=` prefix views move nothing when sizes
/// match.
#[test]
#[ignore = "needs the 14 GB MQ4 artifact + GPU"]
fn prefix_views_keep_loop_parity_128() {
    let (art, cfg, weights) = open_artifact();
    eprintln!("artifact: {art}");
    let vocab = cfg.unpadded_vocab_size;
    let prompt = make_prompt(128, vocab);
    let mut gpu = {
        ensure_home();
        Gpu::init().expect("gpu init")
    };
    let (gw, gs) = upload(&weights, &cfg, &mut gpu, 1024).expect("gpu upload");
    let mut bundle = make_bundle(cfg, weights, gw, gs);
    let mut runs: Vec<Vec<f32>> = Vec::new();
    for _ in 0..2 {
        reset(&mut bundle, &mut gpu);
        let mut last = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            last = forward_token_gpu(&mut bundle, &mut gpu, tok, pos).expect("loop step");
        }
        runs.push(last);
    }
    assert_eq!(runs[0].len(), runs[1].len());
    let md = runs[0]
        .iter()
        .zip(&runs[1])
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "parity128: max|d|={md:.3e} argmax={} (run0) / {} (run1)",
        argmax(&runs[0]),
        argmax(&runs[1])
    );
    assert_eq!(md, 0.0, "prefix views moved normal-path numerics");
    eprintln!("prefix_views_keep_loop_parity_128: PASS (bit-identical)");
}
