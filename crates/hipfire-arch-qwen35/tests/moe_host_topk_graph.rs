// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! #1149 GPU repro: a Qwen3.5-family MoE whose decode falls back to host top-K
//! (`MoeResolution::use_gpu_topk == false`) must not enter the AR decode
//! hipGraph capture. Before the fix the capture recorded the router-output D2H
//! and failed with hipError 906.
//!
//! The condition is constructed on a real A3B fixture by running decode at a
//! router width the GPU top-K kernel cannot select (k = 6), which sends every
//! MoE layer down the host top-K fallback, the same route a non-indexable
//! routed dtype (the issue's qt13+AWQ experts) takes.
//!
//! Three arms, one process, fresh KV/DeltaNet state per arm:
//! - eager reference (`experimental.graph.ar` off);
//! - default graph policy: must run (eager by the load-time decision) and match
//!   the reference greedy ids;
//! - plain AR capture mode set (no replay capture): the sealed MoE guard must
//!   refuse the host top-K call instead of issuing the D2H.
//!
//! ```bash
//! HIPFIRE_MOE_HOST_TOPK_FIXTURE=~/.hipfire/models/ornith-1.5-35b-a3b.mq4r \
//! cargo test --release -p hipfire-arch-qwen35 --test moe_host_topk_graph \
//!   -- --ignored --nocapture --test-threads=1
//! ```

use hipfire_arch_qwen35::qwen35::{
    config_from_hfq, forward_scratch, load_weights, DeltaNetState, HfqSource, LayerType, Layout,
    Qwen35Config, Qwen35Scratch, Qwen35Weights,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;
use std::sync::Arc;

const FIXTURE_ENV: &str = "HIPFIRE_MOE_HOST_TOPK_FIXTURE";
const DEFAULT_FIXTURE: &str = "ornith-1.5-35b-a3b.mq4r";
const CTX: usize = 256;
const STEPS: usize = 24;
/// A router width the GPU top-K kernel does not select (it needs k = 8).
const HOST_TOPK_K: usize = 6;
const PROMPT: [u32; 4] = [760, 6722, 314, 9625];

fn fresh_state(gpu: &mut Gpu, config: &Qwen35Config) -> (KvCache, DeltaNetState) {
    let is_kv_layer: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == LayerType::FullAttention)
        .collect();
    let kv =
        KvCache::new_gpu_q8_filtered(gpu, &is_kv_layer, config.n_kv_heads, config.head_dim, CTX)
            .expect("q8 kv");
    let dn = DeltaNetState::new(gpu, config).expect("dn state");
    (kv, dn)
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        assert!(x.is_finite(), "non-finite logit at {i}");
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Teacher-forced prompt, then greedy continuation; returns every argmax.
fn greedy(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    scratch: &Qwen35Scratch,
) -> Result<Vec<u32>, String> {
    let (mut kv, mut dn) = fresh_state(gpu, config);
    let mut ids = Vec::with_capacity(STEPS);
    let mut token = PROMPT[0];
    for pos in 0..STEPS {
        forward_scratch(gpu, weights, config, token, pos, &mut kv, &mut dn, scratch)
            .map_err(|e| format!("pos {pos}: {e:?}"))?;
        let next = argmax(&gpu.download_f32(&scratch.logits).expect("logits"));
        ids.push(next);
        token = PROMPT.get(pos + 1).copied().unwrap_or(next);
    }
    Ok(ids)
}

fn set_graph_ar(gpu: &mut Gpu, on: bool) {
    let mut flags = (*gpu.flags).clone();
    flags.graph_ar = on;
    gpu.flags = Arc::new(flags);
}

#[test]
#[ignore = "needs a real A3B fixture and a GPU; run with --ignored"]
fn host_topk_moe_decode_never_enters_ar_graph_capture() {
    let path = std::env::var(FIXTURE_ENV).unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.hipfire/models/{DEFAULT_FIXTURE}")
    });
    let mut gpu = Gpu::init().expect("Gpu::init");
    let mut hfq = HfqFile::open(std::path::Path::new(&path)).expect("open fixture");
    let mut config = config_from_hfq(&hfq).expect("config");
    assert!(config.num_experts > 0, "fixture must be MoE");
    let weights = {
        let mut src = HfqSource::new(&mut hfq, &config);
        load_weights(
            &mut src,
            std::slice::from_mut(&mut gpu),
            &Layout::single(config.n_layers),
        )
        .expect("load weights")
    };
    // Construct the host top-K condition (see the module docs).
    config.num_experts_per_tok = HOST_TOPK_K;
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).expect("scratch");
    eprintln!(
        "[1149] arch={} graph_forward={:?} graph_ar={} graph_moe={} replay={}",
        gpu.arch,
        gpu.flags.graph_forward,
        gpu.flags.graph_ar,
        gpu.flags.graph_moe,
        gpu.replay.is_enabled(),
    );
    let default_graph_ar = gpu.flags.graph_ar;

    set_graph_ar(&mut gpu, false);
    let eager = greedy(&mut gpu, &weights, &config, &scratch).expect("eager arm");
    eprintln!("[1149] eager ids {eager:?}");

    set_graph_ar(&mut gpu, default_graph_ar);
    let graph = greedy(&mut gpu, &weights, &config, &scratch)
        .expect("default graph arm must run eager on host top-K, not fail capture");
    eprintln!("[1149] graph ids {graph:?}");
    assert_eq!(
        graph, eager,
        "default graph policy changed host top-K greedy output"
    );

    // Plain AR hipGraph capture mode, no replay capture: the sealed MoE guard
    // fails closed instead of issuing the router-output D2H.
    set_graph_ar(&mut gpu, false);
    let (mut kv, mut dn) = fresh_state(&mut gpu, &config);
    assert!(gpu.graphs.replay.capturing.is_none());
    gpu.graphs.capture_mode = true;
    let refused = forward_scratch(
        &mut gpu, &weights, &config, PROMPT[0], 0, &mut kv, &mut dn, &scratch,
    );
    gpu.graphs.capture_mode = false;
    let err = format!(
        "{:?}",
        refused.expect_err("host top-K MoE ran inside capture mode")
    );
    eprintln!("[1149] capture-mode refusal: {err}");
    assert!(err.contains("cpu-topk-fallback-not-capture-safe"), "{err}");
}
