// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! D1 GPU gates: bitwise equality vs pre-D1 path, H2D counter, tok/s.
//!
//! One ignored test, one 14 GB load, all three gates' data in a single
//! fresh process. UNTRACKED helper (not part of the D1 diff): copy this file
//! into any worktree whose `upload` / `forward_token_gpu` API matches and run:
//!
//! ```powershell
//! $env:HIPFIRE_HOME = 'D:\hipfire-models\inkling2-home'
//! $env:HIPFIRE_GRAPH = '0'
//! $env:INKLING2_D1_OUT = '<outdir>'
//! cargo test -p hipfire-arch-inkling2 --test d1_gates -- --ignored --nocapture
//! ```
//!
//! Output in $OUT (default `target/d1-out`): `logits_tok0.hex` (token-0 GPU
//! logits, hex f32 bits, one per line), `chain_ids.txt` (9 greedy ids,
//! token-0 input id 0 then argmax feedback), `h2d_per_token.txt`
//! (`htod_uploads()` delta per chain token), `timing_ms.txt` (wall ms per
//! chain token). Gate 1 compares `logits_tok0.hex` byte-for-byte and
//! `chain_ids.txt` across arms; gate 2 asserts every chain-token delta <= 1
//! (the embed-row upload, L7a scope); gate 3 uses `timing_ms.txt`.

use hipfire_arch_inkling2::arch_model::Inkling2Bundle;
use hipfire_arch_inkling2::arch_model::Inkling2State;
use hipfire_arch_inkling2::config::Inkling2Config;
use hipfire_arch_inkling2::forward_token_gpu;
use hipfire_arch_inkling2::gpu::{upload, Inkling2GpuState, Inkling2GpuWeights};
use hipfire_arch_inkling2::weights::Inkling2Weights;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

const ART: &str = r"D:\hipfire-models\inkling2-mini-mq4.hfq";
const CHAIN: usize = 9;

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32
}

fn test_gpu() -> Option<Gpu> {
    if std::env::var("HIPFIRE_HOME").is_err() {
        std::env::set_var("HIPFIRE_HOME", r"D:\hipfire-models\inkling2-home");
    }
    assert_eq!(
        std::env::var("HIPFIRE_GRAPH").as_deref(),
        Ok("0"),
        "D1 gates require HIPFIRE_GRAPH=0"
    );
    let gpu = match Gpu::init() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("skip: no usable GPU ({e:?})");
            return None;
        }
    };
    if !matches!(gpu.arch.as_str(), "gfx1100" | "gfx1151") {
        eprintln!("skip: requires gfx1100/gfx1151, found {}", gpu.arch);
        return None;
    }
    Some(gpu)
}

#[test]
#[ignore = "needs the 14 GB MQ4 artifact + gfx1100/gfx1151"]
fn d1_steady() {
    let out = std::env::var("INKLING2_D1_OUT").unwrap_or_else(|_| "target/d1-out".into());
    std::fs::create_dir_all(&out).expect("out dir");
    let Some(mut gpu) = test_gpu() else {
        return;
    };

    let hfq = HfqFile::open(std::path::Path::new(ART)).expect("open artifact");
    let cfg = Inkling2Config::from_metadata_json(&hfq.metadata_json).expect("parse config");
    let weights = Inkling2Weights::load(&hfq, &cfg).expect("index weights");
    let (gw, gs): (Inkling2GpuWeights, Inkling2GpuState) =
        upload(&weights, &cfg, &mut gpu, 64).expect("gpu upload");
    let mut bundle = Inkling2Bundle {
        config: cfg,
        weights,
        state: Inkling2State { seq_pos: 0 },
        eos_tok: 200006,
        gpu_weights: Some(gw),
        gpu_state: Some(gs),
    };

    // Warmup (JIT + DPM settle), then a fresh session for the measured chain.
    let _ = forward_token_gpu(&mut bundle, &mut gpu, 0, 0).expect("warmup");
    bundle
        .gpu_state
        .as_mut()
        .unwrap()
        .reset_session_state(&mut gpu)
        .expect("reset");
    bundle.state.seq_pos = 0;

    let mut ids: Vec<u32> = Vec::with_capacity(CHAIN);
    let mut dts: Vec<f64> = Vec::with_capacity(CHAIN);
    let mut h2ds: Vec<usize> = Vec::with_capacity(CHAIN);
    let mut tok0: Vec<f32> = Vec::new();
    let mut tok = 0u32;
    for pos in 0..CHAIN {
        let h_before = Gpu::htod_uploads();
        let t0 = std::time::Instant::now();
        let logits = forward_token_gpu(&mut bundle, &mut gpu, tok, pos).expect("chain step");
        let dt = t0.elapsed();
        let h_after = Gpu::htod_uploads();
        assert_eq!(logits.len(), bundle.config.unpadded_vocab_size);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "pos {pos}: non-finite"
        );
        if pos == 0 {
            tok0 = logits.clone();
        }
        tok = argmax(&logits);
        ids.push(tok);
        dts.push(dt.as_secs_f64() * 1000.0);
        h2ds.push(h_after - h_before);
        eprintln!(
            "d1: pos {pos} id {tok} {:7.1} ms h2d+{}",
            dts[pos], h2ds[pos]
        );
    }

    let mut s = String::with_capacity(tok0.len() * 9);
    for x in &tok0 {
        s.push_str(&format!("{:08x}\n", x.to_bits()));
    }
    std::fs::write(format!("{out}/logits_tok0.hex"), &s).expect("write logits");
    std::fs::write(
        format!("{out}/chain_ids.txt"),
        ids.iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .expect("write ids");
    std::fs::write(
        format!("{out}/h2d_per_token.txt"),
        h2ds.iter()
            .map(|h| h.to_string())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .expect("write h2d");
    std::fs::write(
        format!("{out}/timing_ms.txt"),
        dts.iter()
            .map(|d| format!("{d:.3}"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .expect("write timing");
    let med = {
        let mut v = dts.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    eprintln!(
        "d1: WROTE {out} ids={ids:?} h2d={h2ds:?} median_ms={med:.1} tokps={:.2}",
        1000.0 / med
    );
}
