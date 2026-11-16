// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! A forced-token advance after the prompt prefill (think-cap close tokens,
//! `hipfire bench` with reasoning off) must leave the Qwen3.5 GDN stream
//! sealed, so the following DFlash/MTP verify and the AR step can read the
//! recurrent state (CHUNK-CANONICAL-DESIGN §3.5). The DFlash forced advance
//! seeds the forced tokens through the prompt-cache suffix prefill, which
//! opened the stream and never sealed it: the next verify failed with "the
//! GDN stream is Open".
//!
//! Each arm: prefill → forced advance (the daemon's call order: the
//! speculator hook, else the target's plain `spec_advance`) → one spec step →
//! a second forced advance → one AR step (DFlash only). The first step's
//! greedy tokens must equal those of the same tokens prefilled as one prompt.
//!
//! `#[ignore]`d: needs a real HIP GPU and the models under `~/.hipfire/models`
//! (skipped when missing). Each arm loads a 27B; run one arm per process
//! (in one process the second arm's load failed on a 32 GB gfx1201):
//!
//! ```text
//! cargo test --release -p hipfire-generate --test spec_forced_advance_seal_hw -- --ignored --exact qwen35_dflash_forced_advance_leaves_the_stream_sealed
//! cargo test --release -p hipfire-generate --test spec_forced_advance_seal_hw -- --ignored --exact qwen35_mtp_forced_advance_leaves_the_stream_sealed
//! ```
//!
//! - DFlash: `qwen3.8-27b.mq4-xts` + `qwen3.8-27b-dflash.mq4v2-sym.hfq`;
//! - MTP: `qwen3.8-27b.mq4-xts` with its `qwen3.8-27b.mtp` sidecar.

use hipfire_runtime::loader_api::SpecLoadCfg;
use hipfire_runtime::spec::{PrefillOutcome, SpecAdvance, SpecTarget, Speculator};
use std::path::PathBuf;

const PROMPT_LEN: usize = 300;

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
    let mut state = 0x5ea1u64;
    (0..PROMPT_LEN)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (1000 + (state >> 33) % 99_000) as u32
        })
        .collect()
}

/// The daemon's forced advance: the speculator hook first, else the target's
/// plain advance.
fn forced_advance(
    gpu: &mut rdna_compute::Gpu,
    spec: &mut dyn Speculator,
    target: &mut dyn SpecTarget,
    tokens: &[u32],
    start: usize,
) {
    let handled = spec
        .on_forced_advance(gpu, target, tokens, start, &|| false)
        .unwrap_or_else(|e| panic!("{}: forced-token advance: {e}", spec.name()));
    if !handled {
        match target.spec_advance(gpu, tokens, start, false, &|| false, None) {
            Ok(SpecAdvance::Ready { .. }) => {}
            other => panic!("{}: plain forced advance: {other:?}", spec.name()),
        }
    }
}

fn check(path: &str, draft: Option<&str>, spec_cfg: SpecLoadCfg, expected: &str) {
    let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
    let mut m = hipfire_loader::load_model(
        path,
        PROMPT_LEN + 512,
        draft,
        None,
        None,
        None,
        1,
        spec_cfg,
        &mut gpu,
    )
    .expect("load");
    let spec = m.speculator.as_mut().expect("speculator loaded");
    assert_eq!(spec.name(), expected, "speculator route");
    let carrier = hipfire_loader::carrier_for(m.arch_id).expect("carrier");
    let mut guard = carrier
        .spec_target_guard(&mut m.state)
        .expect("spec target guard");
    let target = guard.slot().expect("spec target");

    let tokens = prompt();
    let prefill = |spec: &mut Box<dyn Speculator>,
                   gpu: &mut rdna_compute::Gpu,
                   target: &mut dyn SpecTarget,
                   toks: &[u32]| {
        match spec.prefill(gpu, target, toks, toks, 0, false, None, &|| false) {
            Ok(PrefillOutcome::Ready { first_token }) => first_token,
            other => panic!("{expected}: prefill: {other:?}"),
        }
    };
    let first = prefill(spec, &mut gpu, target, &tokens);
    // Think-cap close: commit the pending seed and all forced tokens but the
    // last, which stays the next seed.
    let forced = [first, 151668, 271];

    // Reference: the same tokens as one prompt. Greedy, so the forced route's
    // continuation must start with the same tokens (no near-tie on this
    // fixed prompt: the beta failure diverged at the first token).
    let mut whole = tokens.clone();
    whole.extend_from_slice(&forced);
    let ref_first = prefill(spec, &mut gpu, target, &whole);
    let ref_step = spec
        .step(
            &mut gpu,
            target,
            whole.len(),
            ref_first,
            &[ref_first],
            None,
            0.0,
            usize::MAX,
        )
        .unwrap_or_else(|e| panic!("{expected}: reference verify: {e}"));
    let mut reference = vec![ref_first];
    reference.extend_from_slice(&ref_step.emit);

    let first = prefill(spec, &mut gpu, target, &tokens);
    assert_eq!(first, forced[0], "{expected}: prefill is not deterministic");
    let mut pos = tokens.len();
    let mut emitted = vec![first];
    forced_advance(&mut gpu, spec.as_mut(), target, &forced[..2], pos);
    pos += 2;
    let seed = forced[2];
    emitted.extend_from_slice(&forced[1..]);
    let step = spec
        .step(&mut gpu, target, pos, seed, &emitted, None, 0.0, usize::MAX)
        .unwrap_or_else(|e| panic!("{expected}: verify after forced advance: {e}"));
    assert!(!step.emit.is_empty(), "{expected}: verify emitted nothing");
    let n = step.emit.len().min(reference.len());
    assert_eq!(
        &step.emit[..n],
        &reference[..n],
        "{expected}: the forced route's greedy continuation differs from the same tokens prefilled as one prompt"
    );
    pos += step.emit.len();
    emitted.extend_from_slice(&step.emit);

    // A second forced advance mid-decode (pending seed + one forced token),
    // then the AR step (spec retire) or another verify.
    if emitted.last() != Some(&step.next_seed) {
        emitted.push(step.next_seed);
    }
    forced_advance(
        &mut gpu,
        spec.as_mut(),
        target,
        &[step.next_seed, 151668],
        pos,
    );
    pos += 2;
    emitted.extend_from_slice(&[151668, 271]);
    if spec.supports_ar_step() {
        spec.ar_step(&mut gpu, target, pos, 271, &emitted)
            .unwrap_or_else(|e| panic!("{expected}: AR step after forced advance: {e}"));
    } else {
        spec.step(&mut gpu, target, pos, 271, &emitted, None, 0.0, usize::MAX)
            .unwrap_or_else(|e| panic!("{expected}: second verify: {e}"));
    }
}

#[test]
#[ignore = "needs a HIP GPU, qwen3.8-27b.mq4-xts and qwen3.8-27b-dflash.mq4v2-sym.hfq"]
fn qwen35_dflash_forced_advance_leaves_the_stream_sealed() {
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

#[test]
#[ignore = "needs a HIP GPU and qwen3.8-27b.mq4-xts with its .mtp sidecar"]
fn qwen35_mtp_forced_advance_leaves_the_stream_sealed() {
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
