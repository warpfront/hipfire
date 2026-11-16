// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! A speculative generation must not open the model file.
//!
//! Every Qwen3.5 spec request borrows the target through the carrier's
//! `spec_target_guard` (the daemon's per-request seam). It used to reopen the
//! model (`HfqFile::open`) on each generation, and a symmetric MQ3 open checks
//! every qt49 group, so spec TTFT paid ~1 s per request on a 27B. Each arm
//! runs two generations the way the daemon does (guard, prefill, verify steps,
//! guard drop) and requires [`HfqFile::open_count`] to stay unchanged, with a
//! positive control that one explicit open moves it.
//!
//! `#[ignore]`d: needs a real HIP GPU and the models under `~/.hipfire/models`
//! (skipped when missing). Each arm loads a 27B; run one arm per process:
//!
//! ```text
//! cargo test --release -p hipfire-generate --test spec_no_reopen_hw -- --ignored --exact qwen35_mtp_generation_opens_no_file
//! cargo test --release -p hipfire-generate --test spec_no_reopen_hw -- --ignored --exact qwen35_dflash_generation_opens_no_file
//! ```
//!
//! - MTP: `qwen3.8-27b.mq4-xts` with its `qwen3.8-27b.mtp` sidecar;
//! - DFlash: `qwen3.8-27b.mq4-xts` + `qwen3.8-27b-dflash.mq4v2-sym.hfq`.

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::loader_api::SpecLoadCfg;
use hipfire_runtime::spec::PrefillOutcome;
use std::path::{Path, PathBuf};

const PROMPT_LEN: usize = 64;
const GEN_TOKENS: usize = 32;

fn model(name: &str) -> Option<String> {
    let path = PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".hipfire/models")
        .join(name);
    if !path.is_file() {
        eprintln!("skip: {} missing", path.display());
        return None;
    }
    Some(path.to_string_lossy().into_owned())
}

fn prompt(seed: u64) -> Vec<u32> {
    let mut state = seed;
    (0..PROMPT_LEN)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (1000 + (state >> 33) % 99_000) as u32
        })
        .collect()
}

/// One greedy speculative generation through the daemon's per-request seam:
/// borrow the target, prefill, verify until `GEN_TOKENS` are emitted, and
/// return the bundle on guard drop.
fn generate(m: &mut hipfire_loader::LoadedModel, gpu: &mut rdna_compute::Gpu, seed: u64) {
    let carrier = hipfire_loader::carrier_for(m.arch_id).expect("carrier");
    let spec = m.speculator.as_mut().expect("speculator loaded");
    let mut guard = carrier
        .spec_target_guard(&mut m.state)
        .expect("spec target guard");
    let target = guard.slot().expect("spec target");
    let tokens = prompt(seed);
    let mut seed_token =
        match spec.prefill(gpu, target, &tokens, &tokens, 0, false, None, &|| false) {
            Ok(PrefillOutcome::Ready { first_token }) => first_token,
            other => panic!("{}: prefill: {other:?}", spec.name()),
        };
    let mut pos = tokens.len();
    let mut emitted = vec![seed_token];
    while emitted.len() < GEN_TOKENS {
        let step = spec
            .step(
                gpu,
                target,
                pos,
                seed_token,
                &emitted,
                None,
                0.0,
                usize::MAX,
            )
            .unwrap_or_else(|e| panic!("{}: verify: {e}", spec.name()));
        assert!(
            !step.emit.is_empty(),
            "{}: verify emitted nothing",
            spec.name()
        );
        pos += step.emit.len();
        emitted.extend_from_slice(&step.emit);
        seed_token = step.next_seed;
    }
    drop(guard);
    assert!(m.state.is_some(), "the guard must restore the bundle");
}

fn check(path: &str, draft: Option<&str>, spec_cfg: SpecLoadCfg, expected: &str) {
    let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
    let mut m = hipfire_loader::load_model(
        path,
        PROMPT_LEN + GEN_TOKENS + 512,
        draft,
        None,
        None,
        None,
        1,
        spec_cfg,
        &mut gpu,
    )
    .expect("load");
    assert_eq!(
        m.speculator.as_ref().map(|s| s.name()),
        Some(expected),
        "speculator route"
    );

    let before = HfqFile::open_count();
    generate(&mut m, &mut gpu, 0x0e11);
    generate(&mut m, &mut gpu, 0x0e12);
    assert_eq!(
        HfqFile::open_count(),
        before,
        "{expected}: a speculative generation opened an HFQ file"
    );

    // Positive control: the counter sees an open of the same model.
    let _ = HfqFile::open(Path::new(path)).expect("control open");
    assert_eq!(HfqFile::open_count(), before + 1, "open counter is live");
}

#[test]
#[ignore = "needs a HIP GPU and qwen3.8-27b.mq4-xts with its .mtp sidecar"]
fn qwen35_mtp_generation_opens_no_file() {
    let (Some(path), Some(_)) = (model("qwen3.8-27b.mq4-xts"), model("qwen3.8-27b.mtp")) else {
        return;
    };
    let spec = SpecLoadCfg {
        mtp: Some(true),
        dflash: Some(false),
        dspark: Some(false),
        ..SpecLoadCfg::default()
    };
    check(&path, None, spec, "mtp");
}

#[test]
#[ignore = "needs a HIP GPU, qwen3.8-27b.mq4-xts and qwen3.8-27b-dflash.mq4v2-sym.hfq"]
fn qwen35_dflash_generation_opens_no_file() {
    let (Some(path), Some(draft)) = (
        model("qwen3.8-27b.mq4-xts"),
        model("qwen3.8-27b-dflash.mq4v2-sym.hfq"),
    ) else {
        return;
    };
    let spec = SpecLoadCfg {
        dflash: Some(true),
        mtp: Some(false),
        dspark: Some(false),
        ..SpecLoadCfg::default()
    };
    check(&path, Some(&draft), spec, "dflash");
}
