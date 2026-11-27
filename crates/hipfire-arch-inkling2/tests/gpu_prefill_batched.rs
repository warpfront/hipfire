// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Beta-port F1 gates: blocked multi-row prefill for Inkling-2-Mini.
//!
//! Baseline is beta's per-token eager loop (`forward_token_gpu` per
//! position) — beta has no chunked-prefill seam (brief 11 not on beta).
//! All GPU tests are `#[ignore]`d: they need the 14 GB MQ4 artifact, a
//! gfx1100/gfx1151 GPU, and the lock at
//! `C:\Users\Jan\AppData\Local\Temp\opencode\gpu-lock`
//! (claim before any run: write `U2 <pid>`, release after).
//! Run fresh-process, one gate at a time:
//!
//! ```powershell
//! $env:HIPFIRE_HOME = 'D:\hipfire-models\inkling2-home'
//! $env:HIPFIRE_GRAPH = '0'
//! cargo test -p hipfire-arch-inkling2 --test gpu_prefill_batched -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Gate order (stop at the first red):
//! 1. Blocked entry == eager loop at 2048 tokens (incl. SWA wrap: with
//!    max_seq 4096 the 1024-row SWA ring wraps past position 1024).
//!    Bar: the auditor's agreement bar (argmax equal, KL <= 0.005), NOT
//!    bit-parity: the widen-step carries inherent F16 rounding.
//! 1b. Same at 128 tokens (no wrap; isolates the wiring delta).
//! 1c. Same at 128 tokens on the typed mq4-xts layout
//!    (`INKLING2_XTS_ARTIFACT=PATH`), covering per-expert AWQ.
//! 2. TTFT 2048 before/after (`F1_ARM=seam|batched` single-arm mode for
//!    interleaved fresh-process ABBA). No bar — the ratio is recorded
//!    honestly for upstream to judge; correctness gates are the hard
//!    requirement.
//! 4. Long prompt (8192) prefills without OOM.

use hipfire_arch_inkling2::arch_model::{Inkling2Bundle, Inkling2State};
use hipfire_arch_inkling2::config::Inkling2Config;
use hipfire_arch_inkling2::gpu::{upload, Inkling2GpuState, Inkling2GpuWeights};
use hipfire_arch_inkling2::weights::Inkling2Weights;
use hipfire_arch_inkling2::{forward_token_gpu, prefill_prompt_gpu_blocked};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::time::Instant;

const ART: &str = r"D:\hipfire-models\inkling2-mini-mq4.hfq";

fn ensure_home() {
    if std::env::var("HIPFIRE_HOME").is_err() {
        std::env::set_var("HIPFIRE_HOME", r"D:\hipfire-models\inkling2-home");
    }
    assert_eq!(
        std::env::var("HIPFIRE_GRAPH").as_deref(),
        Ok("0"),
        "gates require HIPFIRE_GRAPH=0"
    );
}

fn open_artifact_at(path: &std::path::Path) -> (Inkling2Config, Inkling2Weights) {
    let hfq = HfqFile::open(path).expect("open artifact");
    let cfg = Inkling2Config::from_metadata_json(&hfq.metadata_json).expect("parse config");
    let weights = Inkling2Weights::load(&hfq, &cfg).expect("index weights");
    (cfg, weights)
}

fn open_artifact() -> (Inkling2Config, Inkling2Weights) {
    let path = std::env::var_os("INKLING2_ARTIFACT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(ART));
    open_artifact_at(&path)
}

fn test_gpu() -> Option<Gpu> {
    ensure_home();
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

fn fresh_bundle(gpu: &mut Gpu, max_seq: usize) -> Inkling2Bundle {
    let (cfg, weights) = open_artifact();
    let (gw, gs) = upload(&weights, &cfg, gpu, max_seq).expect("gpu upload");
    make_bundle(cfg, weights, gw, gs)
}

fn fresh_bundle_at(gpu: &mut Gpu, max_seq: usize, path: &std::path::Path) -> Inkling2Bundle {
    let (cfg, weights) = open_artifact_at(path);
    let (gw, gs) = upload(&weights, &cfg, gpu, max_seq).expect("gpu upload");
    make_bundle(cfg, weights, gw, gs)
}

/// Free a bundle's GPU memory (2 resident bundles exceed card VRAM).
fn free_bundle_gpu(b: &mut Inkling2Bundle, gpu: &mut Gpu) {
    if let Some(gw) = b.gpu_weights.take() {
        gw.free_gpu(gpu);
    }
    if let Some(gs) = b.gpu_state.take() {
        gs.free_gpu(gpu);
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Deterministic prompt: strided ids inside the unpadded vocab (same
/// generator as the stack's fixture: md5 a97120e03dc554c804c11ba8d334bc5
/// for the 2048-token prompt).
fn make_prompt(n: usize, vocab: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 7919) % vocab) as u32).collect()
}

/// Beta's eager baseline: one `forward_token_gpu` per position.
fn eager_prefill(bundle: &mut Inkling2Bundle, gpu: &mut Gpu, token_ids: &[u32]) -> Vec<f32> {
    let mut last = Vec::new();
    for (i, tok) in token_ids.iter().enumerate() {
        last = forward_token_gpu(bundle, gpu, *tok, i).expect("eager step");
    }
    last
}

fn argmax(xs: &[f32]) -> usize {
    let mut bi = 0;
    for (i, &x) in xs.iter().enumerate() {
        if x > xs[bi] {
            bi = i;
        }
    }
    bi
}

/// KL(eager || blocked) in nats over the full logit distribution.
fn kl_p_q(p_logits: &[f32], q_logits: &[f32]) -> f32 {
    assert_eq!(p_logits.len(), q_logits.len());
    let max_p = p_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let max_q = q_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum_p = 0.0f64;
    let mut sum_q = 0.0f64;
    for (&p, &q) in p_logits.iter().zip(q_logits.iter()) {
        sum_p += ((p - max_p) as f64).exp();
        sum_q += ((q - max_q) as f64).exp();
    }
    let lp = sum_p.ln();
    let lq = sum_q.ln();
    let mut kl = 0.0f64;
    for (&p, &q) in p_logits.iter().zip(q_logits.iter()) {
        let log_p = (p - max_p) as f64 - lp;
        let log_q = (q - max_q) as f64 - lq;
        kl += log_p.exp() * (log_p - log_q);
    }
    kl as f32
}

/// Gate 1: blocked entry == eager loop at 2048 tokens (incl. SWA wrap).
/// Bar: the auditor's agreement bar (argmax equal, KL <= 0.005).
///
/// `F1_ARM` (test-only, for interleaved fresh-process TTFT A/B): `seam`
/// runs only the eager arm, `batched` only the batched arm (each prints
/// its own `f1 arm:` line with dt); unset runs both arms plus the parity
/// asserts (default gate behavior).
#[test]
#[ignore = "needs MQ4 artifact + gfx1100/gfx1151 + lock"]
fn f1_gate1_batched_matches_eager_2048() {
    let arm = std::env::var("F1_ARM").unwrap_or_default();
    let Some(mut gpu) = test_gpu() else {
        return;
    };
    let prompt = {
        let (cfg, _) = open_artifact();
        make_prompt(2048, cfg.unpadded_vocab_size)
    };
    let mut eager: Option<Vec<f32>> = None;
    let mut dt_eager = Default::default();
    if arm.is_empty() || arm == "seam" {
        let mut b0 = fresh_bundle(&mut gpu, 4096);
        let t = Instant::now();
        eager = Some(eager_prefill(&mut b0, &mut gpu, &prompt));
        dt_eager = t.elapsed();
        eprintln!("f1 arm: seam dt={dt_eager:?}");
        // Free the seam bundle before the batched arm: two resident bundles
        // exceed card VRAM (2 x 13.3 GiB weights > 25.8 GiB) and contaminate
        // the second arm's timing. Single-arm mode never holds both.
        free_bundle_gpu(&mut b0, &mut gpu);
        drop(b0);
    }
    let mut batched: Option<Vec<f32>> = None;
    let mut dt_batched = Default::default();
    if arm.is_empty() || arm == "batched" {
        let mut b1 = fresh_bundle(&mut gpu, 4096);
        let t = Instant::now();
        batched = Some(prefill_prompt_gpu_blocked(&mut b1, &mut gpu, &prompt, 0).unwrap());
        dt_batched = t.elapsed();
        eprintln!("f1 arm: batched dt={dt_batched:?}");
    }
    if arm.is_empty() {
        let eager = eager.unwrap();
        let batched = batched.unwrap();
        let m = max_abs(&eager, &batched);
        let a_eager = argmax(&eager);
        let a_block = argmax(&batched);
        let kl = kl_p_q(&eager, &batched);
        eprintln!(
            "f1 gate1: max|d|={m:e} argmax {a_eager} vs {a_block} KL={kl:e} eager={dt_eager:?} batched={dt_batched:?}"
        );
        // Bar per supervisor ruling (2026-11-24): the auditor's agreement bar
        // (argmax equal, KL <= 0.005), NOT 1e-5. The <=1e-5 bar governs
        // math-identical paths; an fp16-output visit kernel was never its
        // subject — the widen step carries inherent S1 rounding. max|d| is
        // recorded, not gated.
        assert_eq!(
            a_eager, a_block,
            "batched argmax {a_block} != eager {a_eager} (agreement bar)"
        );
        assert!(
            kl <= 0.005,
            "batched KL {kl:e} exceeds the 0.005 agreement bar"
        );
    }
}

/// Gate 1b: short-prompt parity of the blocked path (128 tokens, no SWA
/// wrap). Bar is the auditor's agreement bar.
#[test]
#[ignore = "needs MQ4 artifact + gfx1100/gfx1151 + lock"]
fn f1_gate1b_short_parity_128() {
    let Some(mut gpu) = test_gpu() else {
        return;
    };
    let n_vocab = open_artifact().0.unpadded_vocab_size;
    let prompt = make_prompt(128, n_vocab);
    let mut b0 = fresh_bundle(&mut gpu, 4096);
    let eager = eager_prefill(&mut b0, &mut gpu, &prompt);
    free_bundle_gpu(&mut b0, &mut gpu);
    drop(b0);
    let mut b1 = fresh_bundle(&mut gpu, 4096);
    let t = Instant::now();
    let blocked = prefill_prompt_gpu_blocked(&mut b1, &mut gpu, &prompt, 0).unwrap();
    let dt = t.elapsed();
    assert_eq!(eager.len(), blocked.len(), "logit len {}", eager.len());
    let m = max_abs(&eager, &blocked);
    let a_eager = argmax(&eager);
    let a_block = argmax(&blocked);
    let kl = kl_p_q(&eager, &blocked);
    eprintln!("f1 gate1b: max|d|={m:e} argmax {a_eager} vs {a_block} KL={kl:e} blocked={dt:?}");
    assert_eq!(
        a_eager, a_block,
        "blocked argmax {a_block} != eager {a_eager} (agreement bar)"
    );
    assert!(
        kl <= 0.005,
        "blocked KL {kl:e} exceeds the 0.005 agreement bar"
    );
}

/// Gate 1c: the default mq4-xts layout keeps its per-expert AWQ path through
/// both eager decode and blocked prefill. Set `INKLING2_XTS_ARTIFACT` to the
/// typed XTS HFQ; the agreement bar matches gate 1b.
#[test]
#[ignore = "needs mq4-xts artifact + gfx1100/gfx1151 + lock"]
fn f1_gate1c_xts_parity_128() {
    let Some(mut gpu) = test_gpu() else {
        return;
    };
    let path = std::env::var_os("INKLING2_XTS_ARTIFACT")
        .map(std::path::PathBuf::from)
        .expect("set INKLING2_XTS_ARTIFACT to the mq4-xts HFQ");
    let mut b0 = fresh_bundle_at(&mut gpu, 4096, &path);
    let prompt = make_prompt(128, b0.config.unpadded_vocab_size);
    let eager = eager_prefill(&mut b0, &mut gpu, &prompt);
    free_bundle_gpu(&mut b0, &mut gpu);
    drop(b0);
    let mut b1 = fresh_bundle_at(&mut gpu, 4096, &path);
    let blocked = prefill_prompt_gpu_blocked(&mut b1, &mut gpu, &prompt, 0).unwrap();
    let m = max_abs(&eager, &blocked);
    let a_eager = argmax(&eager);
    let a_block = argmax(&blocked);
    let kl = kl_p_q(&eager, &blocked);
    eprintln!("f1 gate1c xts: max|d|={m:e} argmax {a_eager} vs {a_block} KL={kl:e}");
    assert_eq!(a_eager, a_block, "XTS blocked argmax differs from eager");
    assert!(kl <= 0.005, "XTS blocked KL {kl:e} exceeds 0.005");
}

/// Gate 2: TTFT 2048 before/after. No bar — the ratio is recorded honestly
/// for upstream to judge; correctness gates are the hard requirement.
/// Baseline is beta's per-token eager loop (unmodified beta build), same
/// prompt bytes, fresh processes, throwaway first, interleaved ABBA.
#[test]
#[ignore = "needs MQ4 artifact + gfx1100/gfx1151 + lock"]
fn f1_gate2_ttft() {
    let Some(mut gpu) = test_gpu() else {
        return;
    };
    let prompt = {
        let (cfg, _) = open_artifact();
        make_prompt(2048, cfg.unpadded_vocab_size)
    };
    let mut b0 = fresh_bundle(&mut gpu, 4096);
    let t = Instant::now();
    let _ = eager_prefill(&mut b0, &mut gpu, &prompt);
    let dt_eager = t.elapsed();
    free_bundle_gpu(&mut b0, &mut gpu);
    drop(b0);
    let mut b1 = fresh_bundle(&mut gpu, 4096);
    let t = Instant::now();
    let _ = prefill_prompt_gpu_blocked(&mut b1, &mut gpu, &prompt, 0).unwrap();
    let dt_batched = t.elapsed();
    let ratio = dt_eager.as_secs_f64() / dt_batched.as_secs_f64();
    eprintln!(
        "f1 gate2: eager={dt_eager:?} batched={dt_batched:?} ratio={ratio:.2}x (recorded, no bar)"
    );
}

/// Gate 4: 8192-token prompt prefills without OOM.
#[test]
#[ignore = "needs MQ4 artifact + gfx1100/gfx1151 + lock"]
fn f1_gate4_long_prompt_no_oom() {
    let Some(mut gpu) = test_gpu() else {
        return;
    };
    let prompt = {
        let (cfg, _) = open_artifact();
        make_prompt(8192, cfg.unpadded_vocab_size)
    };
    let mut b = fresh_bundle(&mut gpu, 8192);
    let t = Instant::now();
    let logits = prefill_prompt_gpu_blocked(&mut b, &mut gpu, &prompt, 0).unwrap();
    eprintln!(
        "f1 gate4: 8192 tokens ok in {:?}, logits len {}",
        t.elapsed(),
        logits.len()
    );
    assert!(!logits.is_empty());
}
