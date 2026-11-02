// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Rows that would end past the KV cache are refused before any launch.
//!
//! The daemon refuses a prompt of `max_seq` tokens or more and stops decode at
//! the cap, but a verify block or prefill chunk that starts below the cap can
//! end past it. The legacy KV backend has no mapping guard, so the sandwich
//! attention's cache bound is the refusal: the call returns an error and the
//! caller's request fails instead of writing past the cache.

use hipfire_arch_muse_glimmer::config::GlimmerConfig;
use hipfire_arch_muse_glimmer::forward::{prefill_with_capture, verify_block_with_capture};
use hipfire_arch_muse_glimmer::glimmer::{GlimmerState, GlimmerWeights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_backend::KvBackend;
use rdna_compute::Gpu;
use std::path::PathBuf;

const FIXTURE: &str = "muse-glimmer-30b.mq4";
const MAX_SEQ: usize = 64;

/// `$HIPFIRE_DENSE_FIXTURE/<name>`, else `~/.hipfire/models/<name>`.
fn fixture() -> Option<PathBuf> {
    let dir = std::env::var_os("HIPFIRE_DENSE_FIXTURE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".hipfire/models")))?;
    let path = dir.join(FIXTURE);
    path.exists().then_some(path)
}

#[test]
#[ignore = "needs a HIP GPU and the muse-glimmer-30b.mq4 fixture"]
fn rows_past_max_seq_are_an_error() {
    let Some(path) = fixture() else {
        eprintln!("skip: {FIXTURE} not found");
        return;
    };
    let mut gpu = Gpu::init().expect("gpu");
    let hfq = HfqFile::open(&path).expect("open");
    let cfg = GlimmerConfig::from_hfq(&hfq).expect("cfg");
    let weights = GlimmerWeights::load(&hfq, &cfg, &mut gpu).expect("weights");
    let mut state =
        GlimmerState::new_with_max_seq_backend(&mut gpu, &cfg, MAX_SEQ, KvBackend::Legacy)
            .expect("state");
    let tokens: Vec<u32> = (0..MAX_SEQ as u32).map(|t| 1000 + t).collect();
    let mut hidden = Vec::new();

    let start = MAX_SEQ - 8;
    prefill_with_capture(
        &cfg,
        &weights,
        &mut state,
        &mut gpu,
        &tokens[..start],
        0,
        &[],
        &mut hidden,
    )
    .expect("prefill below the cap");

    fn refused<T>(r: Result<T, String>) {
        match r {
            Err(e) => assert!(e.contains("outside a cache of 64 rows"), "{e}"),
            Ok(_) => panic!("rows past max_seq ran"),
        }
    }
    // A verify block and a prefill chunk that start below the cap and end past it.
    refused(verify_block_with_capture(
        &cfg,
        &weights,
        &mut state,
        &mut gpu,
        &tokens[..16],
        start as u32,
        &[],
        &mut hidden,
        None,
    ));
    refused(prefill_with_capture(
        &cfg,
        &weights,
        &mut state,
        &mut gpu,
        &tokens[..9],
        start as u32,
        &[],
        &mut hidden,
    ));

    // The request that fits still runs at the same position.
    let picks = verify_block_with_capture(
        &cfg,
        &weights,
        &mut state,
        &mut gpu,
        &tokens[..8],
        start as u32,
        &[],
        &mut hidden,
        None,
    )
    .expect("verify up to the cap");
    assert_eq!(picks.len(), 8);
}
