// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched-vs-per-token prefill parity harness for Gemma 4.
//!
//! Runs the SAME prompt through (A) per-token `decode_step` and
//! (B) `forward_prefill_batch` over all but the last token plus one
//! `decode_step`, with fresh state each; compares the prompt's last logits
//! (max-abs diff, argmax) and an N-token greedy continuation (per-token decode
//! from both prefill states).
//!
//! Usage:
//!   prefill_parity_gemma4 --model <hfq> [--prompt <text>] [--decode N]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_gemma4::{forward, Gemma4Config, Gemma4State, Gemma4Weights};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompt =
        "The capital of France is a city with many famous museums and lovely streets".to_string();
    let mut decode_n: usize = 24;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--prompt" => {
                prompt = argv[i + 1].clone();
                i += 2;
            }
            "--prompt-file" => {
                prompt = std::fs::read_to_string(&argv[i + 1]).expect("read prompt file");
                i += 2;
            }
            "--decode" => {
                decode_n = argv[i + 1].parse().expect("--decode");
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(1);
            }
        }
    }
    let model = model.expect("--model required");

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("arch = {}", gpu.arch);
    let hfq = HfqFile::open(&model).expect("open model");
    let cfg = Gemma4Config::from_hfq(&hfq).expect("config");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let weights = Gemma4Weights::load(&hfq, &cfg, &mut gpu).expect("weights");
    let mut ids = tok.encode(&prompt);
    if ids.first() != Some(&cfg.bos_token) {
        ids.insert(0, cfg.bos_token);
    }
    let max_seq = (ids.len() + decode_n + 16).max(cfg.sliding_window + 1);

    eprintln!("prompt tokens = {}", ids.len());

    let fnv = |bytes: &[u8]| -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    };
    let argmax = |v: &[f32]| -> (usize, f32) {
        let mut bi = 0;
        let mut bv = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                bi = i;
            }
        }
        (bi, bv)
    };

    let mut run = |label: &str, batched: bool| -> (Vec<f32>, Vec<u32>) {
        let mut state = Gemma4State::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("state");
        let t0 = std::time::Instant::now();
        if batched {
            let last = ids.len() - 1;
            forward::forward_prefill_batch(&cfg, &weights, &mut state, &mut gpu, &ids[..last], 0)
                .expect("batched prefill");
            forward::decode_step(&cfg, &weights, &mut state, &mut gpu, ids[last], last as u32)
                .expect("last prompt token");
        } else {
            for (p, &t) in ids.iter().enumerate() {
                forward::decode_step(&cfg, &weights, &mut state, &mut gpu, t, p as u32)
                    .expect("per-token prefill");
            }
        }
        eprintln!(
            "[{label}] prefill {} tok in {:.3}s",
            ids.len(),
            t0.elapsed().as_secs_f64()
        );
        let logits = gpu.download_f32(&state.logits).expect("logits dl");
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite {label} logits"
        );
        let (am, av) = argmax(&logits);
        let lh = fnv(unsafe {
            std::slice::from_raw_parts(logits.as_ptr() as *const u8, logits.len() * 4)
        });
        eprintln!(
            "[{label}] prefill logits: argmax={am} ({:?}) val={av:.4} fnv=0x{lh:016x}",
            tok.decode(&[am as u32])
        );
        // Greedy continuation, per-token decode in BOTH runs (isolates prefill).
        let mut cont = Vec::new();
        let mut pos = ids.len();
        let mut next = am as u32;
        for _ in 0..decode_n {
            cont.push(next);
            let l = forward::decode_step(&cfg, &weights, &mut state, &mut gpu, next, pos as u32)
                .expect("decode");
            next = argmax(&l).0 as u32;
            pos += 1;
        }
        eprintln!("[{label}] cont ids:  {:?}", cont);
        eprintln!("[{label}] cont text: {:?}", tok.decode(&cont));
        state.free_gpu(&mut gpu);
        (logits, cont)
    };

    let (la, ca) = run("per-token", false);
    let (lb, cb) = run("batched  ", true);

    let mut max_abs = 0f32;
    let mut max_i = 0usize;
    let mut n_diff = 0usize;
    for i in 0..la.len() {
        let d = (la[i] - lb[i]).abs();
        if d > 0.0 {
            n_diff += 1;
        }
        if d > max_abs {
            max_abs = d;
            max_i = i;
        }
    }
    println!(
        "logits: n_diff={n_diff}/{} max_abs={max_abs:.6} at idx {max_i} (A={:.4} B={:.4})",
        la.len(),
        la[max_i],
        lb[max_i]
    );
    println!(
        "cont match: {}",
        if ca == cb { "IDENTICAL" } else { "DIVERGED" }
    );
    if ca != cb {
        let first = ca
            .iter()
            .zip(&cb)
            .position(|(a, b)| a != b)
            .unwrap_or(ca.len());
        println!("first cont divergence at token {first}");
        std::process::exit(2);
    }
    if n_diff == 0 {
        println!("logits BYTE-IDENTICAL");
    }
}
