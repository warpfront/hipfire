// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! An abort polled during an MTP-family speculative prefill is a cancel, not
//! an error (warpfront/hipfire#822).
//!
//! `Speculator::prefill` must return `Ok(PrefillOutcome::Aborted)` for a client
//! abort: the daemon's spec route rolls back and emits the `aborted` +
//! `done(aborted)` pair on that arm, while every `Err` becomes a
//! `class=validation` error terminal. A real failure (an empty prompt) must
//! stay an `Err`.
//!
//! `#[ignore]`d: each test needs a real HIP GPU and its model under
//! `~/.hipfire/models` (skipped when missing):
//!
//! ```text
//! cargo test --release -p hipfire-generate --test spec_prefill_abort_hw -- --ignored --test-threads=1
//! ```
//!
//! - DSpark: `qwen3-8b.mq4` with its `-dspark` sidecar;
//! - Qwen4 native MTP: `qwen3.8-flash-next-gptq3.mq4`;
//! - Qwen3.5 MTP: `qwen3.8-27b.mq4-xt` with its `.mtp` sidecar.

use hipfire_runtime::loader_api::{CaskConfig, SpecLoadCfg};
use hipfire_runtime::spec::PrefillOutcome;
use std::cell::Cell;
use std::path::PathBuf;

/// Long enough that every route polls the abort more than once.
const PROMPT_LEN: usize = 2048;

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

fn prompt() -> Vec<u32> {
    let mut state = 0x822u64;
    (0..PROMPT_LEN)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (1000 + (state >> 33) % 99_000) as u32
        })
        .collect()
}

/// Load `path` with `spec`, then: an abort raised at the second poll (after
/// the prefill started) is `Ok(Aborted)`; with `empty_prompt_errs`, an empty
/// prompt (a real failure) is `Err`.
fn check(path: &str, spec: SpecLoadCfg, expected: &str, empty_prompt_errs: bool) {
    let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
    let mut m = hipfire_loader::load_model(
        path,
        PROMPT_LEN + 512,
        None,
        None,
        None,
        None,
        &CaskConfig::default(),
        1,
        spec,
        &mut gpu,
    )
    .expect("load");
    let speculator = m.speculator.as_mut().expect("speculator loaded");
    assert_eq!(speculator.name(), expected, "speculator route");
    let carrier = hipfire_loader::carrier_for(m.arch_id).expect("carrier");
    let mut guard = carrier
        .spec_target_guard(&mut m.state, &m.model_path)
        .expect("spec target guard");
    let target = guard.slot().expect("spec target");

    let tokens = prompt();
    let polls = Cell::new(0usize);
    let abort = || {
        polls.set(polls.get() + 1);
        polls.get() > 1
    };
    let outcome = speculator.prefill(&mut gpu, target, &tokens, &tokens, 0, false, None, &abort);
    assert_eq!(
        outcome,
        Ok(PrefillOutcome::Aborted),
        "{expected}: an abort mid-prefill must be a cancel, not an error"
    );
    assert!(polls.get() > 1, "{expected}: the abort was never raised");

    if empty_prompt_errs {
        let failure = speculator.prefill(&mut gpu, target, &[], &[], 0, false, None, &|| false);
        assert!(
            failure.is_err(),
            "{expected}: an empty prompt must stay an error, got {failure:?}"
        );
    }
}

#[test]
#[ignore = "needs a HIP GPU and qwen3-8b.mq4 with its DSpark sidecar"]
fn dspark_prefill_abort_is_a_cancel() {
    let Some(path) = model("qwen3-8b.mq4") else {
        return;
    };
    let spec = SpecLoadCfg {
        dspark: Some(true),
        mtp: Some(false),
        dflash: Some(false),
        ..SpecLoadCfg::default()
    };
    check(&path, spec, "dspark", true);
}

#[test]
#[ignore = "needs a HIP GPU and qwen3.8-flash-next-gptq3.mq4"]
fn qwen4_native_mtp_prefill_abort_is_a_cancel() {
    let Some(path) = model("qwen3.8-flash-next-gptq3.mq4") else {
        return;
    };
    let spec = SpecLoadCfg {
        mtp: Some(true),
        dspark: Some(false),
        dflash: Some(false),
        ..SpecLoadCfg::default()
    };
    check(&path, spec, "mtp", true);
}

#[test]
#[ignore = "needs a HIP GPU and qwen3.8-27b.mq4-xt with its .mtp sidecar"]
fn qwen35_mtp_prefill_abort_is_a_cancel() {
    let Some(path) = model("qwen3.8-27b.mq4-xt") else {
        return;
    };
    let spec = SpecLoadCfg {
        mtp: Some(true),
        dspark: Some(false),
        dflash: Some(false),
        ..SpecLoadCfg::default()
    };
    // The Qwen3.5 drafter does not reject an empty prompt (it returns a seed
    // from stale logits), so this arm checks the abort only.
    check(&path, spec, "mtp", false);
}
