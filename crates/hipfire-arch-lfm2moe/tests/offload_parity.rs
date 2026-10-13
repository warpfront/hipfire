// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Greedy-decode parity between a fully resident LFM2.5 load and host-placed
//! (partial-offload) loads of the same artifact.
//!
//! The offload contract for this arch: placement decides WHERE a weight lives
//! (`hipHostMalloc(Mapped)` vs VRAM) and nothing else. The kernels dereference
//! the device alias of a host-mapped buffer over PCIe, so with `pcie` exec
//! (the only mode arch 11 admits) the arithmetic per token is identical and
//! greedy decode must be **byte-identical** — the same go/no-go oracle the
//! Qwen3.5 broadening used (`docs/plans/partial-gpu-offload-design.md` §9).
//!
//! Three arms over one artifact:
//!
//! 1. resident baseline (`Placement::all_device`);
//! 2. whole-layer spill — the first `SPILL_LAYERS` layers' own weights host
//!    (their routed experts follow the layer, per `Placement::dense_prefix`);
//! 3. routed-expert spill — experts host on the first `SPILL_LAYERS` MoE
//!    layers while every layer's own weights stay resident (the
//!    `memory.moe_expert_budget` tier; dense fixtures have no MoE layers and
//!    skip the arm).
//!
//! Each arm additionally asserts the locality it claims (host-located where
//! the placement says host, device where it says device) and that the model
//! really host-mapped (`Gpu::host_mapped_count`), so a placement that
//! silently fell back to device uploads cannot pass as a parity win.
//!
//! `#[ignore]`d: needs an RDNA GPU with a working HIP toolchain plus an HFQ
//! artifact. Both fixtures of the ladder work:
//!
//!   # dense (no expert arm):
//!   HIPFIRE_LFM2_OFFLOAD_PARITY_MODEL=$HOME/.hipfire/models/lfm2.5-1.2b.mq4 \
//!     cargo test -p hipfire-arch-lfm2moe --release --test offload_parity -- --ignored --nocapture
//!   # MoE (all three arms):
//!   HIPFIRE_LFM2_OFFLOAD_PARITY_MODEL=$HOME/.hipfire/models/lfm2.5-8b-a1b.mq4 \
//!     cargo test -p hipfire-arch-lfm2moe --release --test offload_parity -- --ignored --nocapture

use std::env;

use hipfire_arch_lfm2moe::lfm2moe::{Ffn, Lfm2MoeState, Lfm2MoeWeights};
use hipfire_arch_lfm2moe::Lfm2MoeConfig;
use hipfire_runtime::offload::{ExpertResidency, Placement, WeightResidency};
use rdna_compute::Gpu;

/// Whole layers host-placed in arm 2 / expert layers host-placed in arm 3.
/// Enough to cross every tensor class (conv, attention, dense FFN, MoE) on
/// both fixtures without being so large the host-side upload dominates.
const SPILL_LAYERS: usize = 3;

/// Greedy decode length past the seed prompt.
const STEPS: usize = 32;

/// Deterministic seed tokens (a parity oracle, not a coherence probe — the
/// ids only need to exercise the vocab gather and every mixer class).
fn seed_ids(vocab_size: usize) -> Vec<u32> {
    let mut state = 0x5EED_5EEDu64;
    (0..8)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32) % vocab_size as u32
        })
        .collect()
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0u32, f32::NEG_INFINITY), |(best, bv), (i, &v)| {
            if v > bv {
                (i as u32, v)
            } else {
                (best, bv)
            }
        })
        .0
}

/// Greedy-decode `STEPS` tokens from `seed`, returning the emitted token ids.
/// `decode_step(cfg, .., token, position)` feeds one token at its absolute
/// position and returns that position's logits, so the last seed token's
/// logits emit the first generated token.
fn greedy_decode(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    gpu: &mut Gpu,
    seed: &[u32],
) -> Vec<u32> {
    let mut state = Lfm2MoeState::new(gpu, cfg).expect("state alloc");
    let mut emitted = Vec::with_capacity(STEPS);
    let mut pos = 0u32;
    let mut last_logits = Vec::new();
    for &token in seed {
        last_logits = hipfire_arch_lfm2moe::forward::decode_step(
            cfg, weights, &mut state, gpu, token, pos,
        )
        .expect("decode step");
        pos += 1;
    }
    emitted.push(argmax(&last_logits));
    while emitted.len() < STEPS {
        let token = *emitted.last().unwrap();
        let logits = hipfire_arch_lfm2moe::forward::decode_step(
            cfg, weights, &mut state, gpu, token, pos,
        )
        .expect("decode step");
        pos += 1;
        emitted.push(argmax(&logits));
    }
    state.free_gpu(gpu);
    emitted
}

#[test]
#[ignore = "needs an RDNA GPU + HIPFIRE_LFM2_OFFLOAD_PARITY_MODEL=<hfq path>"]
fn spilled_lfm2moe_decodes_byte_identical_to_resident() {
    let Some(path) = env::var_os("HIPFIRE_LFM2_OFFLOAD_PARITY_MODEL") else {
        eprintln!("skipped: HIPFIRE_LFM2_OFFLOAD_PARITY_MODEL is unset");
        return;
    };
    let mut gpu = match Gpu::init() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("skipped: no GPU ({e:?})");
            return;
        }
    };
    let mut hfq = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(&path))
        .expect("open HFQ");
    let cfg = Lfm2MoeConfig::from_hfq(&hfq).expect("parse config");
    let seed = seed_ids(cfg.vocab_size);
    let n = cfg.num_hidden_layers;

    // ── Arm 1: resident baseline ───────────────────────────────────────────
    let weights = Lfm2MoeWeights::load(&mut hfq, &cfg, &mut gpu).expect("resident load");
    let baseline = greedy_decode(&cfg, &weights, &mut gpu, &seed);
    assert_eq!(baseline.len(), STEPS);
    assert_eq!(gpu.host_mapped_count(), 0, "resident load host-mapped nothing");
    weights.free_gpu(&mut gpu);

    // ── Arm 2: whole-layer spill (dense_prefix: layer + its experts) ───────
    let placement = Placement::dense_prefix(n, SPILL_LAYERS.min(n));
    let weights =
        Lfm2MoeWeights::load_with_placement(&mut hfq, &cfg, &mut gpu, &placement).expect("spilled load");
    assert_locality(&cfg, &weights, &gpu, &placement);
    let spilled = greedy_decode(&cfg, &weights, &mut gpu, &seed);
    weights.free_gpu(&mut gpu);
    assert!(
        gpu.host_mapped_count() == 0,
        "free_gpu returned every host-mapped owner (left: {})",
        gpu.host_mapped_count()
    );
    assert_eq!(
        baseline, spilled,
        "whole-layer spill changed the greedy token sequence"
    );

    // ── Arm 3: routed-expert spill, layer weights resident ─────────────────
    if cfg.num_experts == 0 {
        eprintln!("skipped: dense fixture has no routed experts");
        return;
    }
    let placement = Placement {
        layer_residency: vec![WeightResidency::Resident; n],
        expert_residency: (0..n)
            .map(|l| {
                if l < SPILL_LAYERS.min(n) {
                    ExpertResidency::HostMapped
                } else {
                    ExpertResidency::Device
                }
            })
            .collect(),
    };
    let weights =
        Lfm2MoeWeights::load_with_placement(&mut hfq, &cfg, &mut gpu, &placement).expect("expert-spilled load");
    assert_locality(&cfg, &weights, &gpu, &placement);
    let expert_spilled = greedy_decode(&cfg, &weights, &mut gpu, &seed);
    weights.free_gpu(&mut gpu);
    assert!(
        gpu.host_mapped_count() == 0,
        "free_gpu returned every host-mapped owner (left: {})",
        gpu.host_mapped_count()
    );
    assert_eq!(
        baseline, expert_spilled,
        "routed-expert spill changed the greedy token sequence"
    );
}

/// Assert the load honoured the placement: a host layer's own weights are
/// host-located, a device layer's are not, and a host expert layer's experts
/// are host-located while its router stays where its layer tier says.
fn assert_locality(cfg: &Lfm2MoeConfig, weights: &Lfm2MoeWeights, gpu: &Gpu, placement: &Placement) {
    let mut host_layers = 0usize;
    let mut host_expert_layers = 0usize;
    for (l, layer) in weights.layers.iter().enumerate() {
        let layer_host = placement.layer(l) == WeightResidency::HostMapped;
        assert_eq!(
            gpu.host_located(&layer.operator_norm),
            layer_host,
            "layer {l} operator_norm locality"
        );
        if layer_host {
            host_layers += 1;
        }
        if let Ffn::Moe(moe) = &layer.ffn {
            let experts_host = placement.experts(l) == ExpertResidency::HostMapped;
            assert_eq!(
                gpu.host_located(&moe.experts[0].gate_up.buf),
                experts_host,
                "layer {l} expert gate_up locality"
            );
            assert_eq!(
                gpu.host_located(&moe.experts[0].down.buf),
                experts_host,
                "layer {l} expert down locality"
            );
            if experts_host {
                host_expert_layers += 1;
            }
        }
    }
    eprintln!(
        "locality: {host_layers} host layers, {host_expert_layers} host-expert layers, \
         {} host-mapped owners",
        gpu.host_mapped_count()
    );
    assert!(
        gpu.host_mapped_count() > 0,
        "the placement host-mapped nothing — this test proves nothing"
    );
    // The globals never move.
    assert!(
        !gpu.host_located(&weights.embedding_norm),
        "embedding_norm is always resident"
    );
    assert!(
        !gpu.host_located(&weights.lm_head.buf),
        "lm_head is always resident"
    );
    let _ = cfg.max_position_embeddings;
}
