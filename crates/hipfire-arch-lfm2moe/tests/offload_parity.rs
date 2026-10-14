// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Greedy-decode parity between a fully resident LFM2.5 load and host-placed
//! (partial-offload) loads of the same artifact.
//!
//! The offload contract for this arch: placement decides WHERE a weight lives
//! (`hipHostMalloc(Mapped)` vs VRAM) and nothing else. Kernels dereference the
//! device alias of a host-mapped buffer over PCIe, so with `pcie` exec (the
//! only mode arch 11 admits) the per-token arithmetic is identical and greedy
//! decode must be **byte-identical** — the same go/no-go oracle the Qwen3.5
//! broadening used (`docs/plans/partial-gpu-offload-design.md` §9).
//!
//! # One model load per process
//!
//! Each invocation loads the model **once**. Two loads in one process do not
//! reliably fit: `free_gpu` returns weight buffers to `GpuPool`'s free lists,
//! while the next load's `upload_raw` allocates from raw `hip.malloc`, which
//! cannot see the pool (only `GpuPool::alloc` drains it on OOM, and weight
//! uploads bypass `alloc`). The first load is therefore still charged to the
//! device when the second asks for its bytes. Measured on this box: two 8B-A1B
//! loads (4.5 GB each) OOM at ~layer 19 with a co-tenant process holding
//! 1.27 GB, even though a single load fits comfortably.
//!
//! So the baseline is recorded once, under one placement, and compared from a
//! second process under another:
//!
//!   M=$HOME/.hipfire/models/lfm2.5-8b-a1b.mq4
//!   B=/tmp/lfm25-baseline.txt
//!   record() { HIPFIRE_LFM2_OFFLOAD_PARITY_MODEL=$M \
//!              HIPFIRE_LFM2_OFFLOAD_PARITY_PLACEMENT=$1 \
//!              HIPFIRE_LFM2_OFFLOAD_PARITY_RECORD=1 \
//!              HIPFIRE_LFM2_OFFLOAD_PARITY_BASELINE=$B cargo test \
//!              -p hipfire-arch-lfm2moe --release --test offload_parity \
//!              -- --ignored --nocapture --exact spill_parity_vs_baseline; }
//!   check()  { HIPFIRE_LFM2_OFFLOAD_PARITY_MODEL=$M \
//!              HIPFIRE_LFM2_OFFLOAD_PARITY_PLACEMENT=$1 \
//!              HIPFIRE_LFM2_OFFLOAD_PARITY_BASELINE=$B cargo test ... }
//!   record resident && check layer && check expert
//!
//! The recorded placement is whatever `*_PLACEMENT` names, so for a model that
//! fits resident, record `resident` and check `layer` / `expert` — that is the
//! full-strength oracle. For a model larger than the card
//! (LFM2.5-24B-A2B at MQ4 is 12.76 GB against ~10 GB free here) no resident
//! load exists, so record one spill depth and check another. That comparison
//! is weaker (a placement error identical at both depths passes) but it still
//! exercises two different host/device splits of every tensor class, and both
//! depths must fit one-at-a-time.
//!
//! `two_spill_depths_agree` does the same depth-vs-depth comparison in one
//! process (no baseline file), which is convenient when two loads fit. For a
//! near-card-size model they do not — use the record/check flow above.
//!
//! Env:
//!   HIPFIRE_LFM2_OFFLOAD_PARITY_MODEL      HFQ path (required)
//!   HIPFIRE_LFM2_OFFLOAD_PARITY_PLACEMENT  resident | layer | expert (default layer)
//!   HIPFIRE_LFM2_OFFLOAD_PARITY_RECORD     1 = write the baseline instead of comparing
//!   HIPFIRE_LFM2_OFFLOAD_PARITY_BASELINE   baseline file (required by the test)
//!   HIPFIRE_LFM2_OFFLOAD_PARITY_LAYERS     spill depth (default 3)
//!   HIPFIRE_LFM2_OFFLOAD_PARITY_DEEP       depth for two_spill_depths_agree (default 3x layers)
//!   HIPFIRE_LFM2_OFFLOAD_PARITY_STEPS      greedy steps (default 32)

use std::env;

use hipfire_arch_lfm2moe::lfm2moe::{Ffn, Lfm2MoeState, Lfm2MoeWeights};
use hipfire_arch_lfm2moe::Lfm2MoeConfig;
use hipfire_runtime::offload::{ExpertResidency, Placement, WeightResidency};
use rdna_compute::Gpu;

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Seed tokens: a REAL prompt encoded by the model's own tokenizer.
///
/// A synthetic pseudo-random id sequence is not good enough here. Feeding
/// random ids to this model drove greedy decode into a single-token attractor
/// (32 x token 900), and a saturated decode matches under *any* placement — the
/// comparison would pass while proving nothing (the exact failure mode
/// AGENTS.md §0 warns about). A real prompt keeps the distribution varied, and
/// [`assert_not_degenerate`] fails loudly if it ever collapses again.
fn prompt_ids(hfq: &hipfire_runtime::hfq::HfqFile) -> Vec<u32> {
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer from HFQ metadata");
    let ids = tokenizer.encode(PROMPT);
    assert!(
        ids.len() >= 4,
        "prompt tokenized to {} ids — too short to drive a meaningful decode",
        ids.len()
    );
    ids
}

/// A fixed, committed prompt in the model's own chat framing, so a recorded
/// baseline stays comparable across runs (CLAUDE.md: prompt structure dictates
/// the distribution, so any comparison needs identical prompt bytes).
///
/// The framing matters. These are "always thinks" chat models whose template
/// opens the assistant turn with `<think>` (see the checkpoint's
/// `chat_template.jinja`); a bare raw-completion prompt drives greedy decode
/// into a single-token attractor on the 8B/24B, which would make the parity
/// comparison vacuous. Verified on the whole ladder that this prompt keeps the
/// decode varied enough for [`assert_not_degenerate`] to hold.
const PROMPT: &str = "<|im_start|>user\nWrite a Python function that returns the nth \
                      Fibonacci number.<|im_end|>\n<|im_start|>assistant\n<think>\n";

/// Refuse to accept a degenerate decode as parity evidence.
///
/// A greedy sequence that collapses to one repeated token is invariant to
/// weight placement, so it cannot distinguish a correct placement from a
/// broken one. Minimum distinct tokens over the window, not a statistical
/// test: this is a floor on the oracle's discriminating power.
fn assert_not_degenerate(tokens: &[u32], label: &str) {
    let distinct = {
        let mut seen = tokens.to_vec();
        seen.sort_unstable();
        seen.dedup();
        seen.len()
    };
    assert!(
        distinct >= 4,
        "{label}: greedy decode emitted only {distinct} distinct token(s) over {} steps \
         (first ids: {:?}) — a saturated decode is placement-invariant, so this run cannot \
         prove placement parity. Use a prompt that does not collapse.",
        tokens.len(),
        &tokens[..tokens.len().min(12)]
    );
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

/// Greedy-decode `steps` tokens from `seed`. `decode_step(cfg, .., token,
/// position)` feeds one token at its absolute position and returns that
/// position's logits, so the last seed token's logits emit the first generated
/// token.
fn greedy_decode(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    gpu: &mut Gpu,
    seed: &[u32],
    steps: usize,
) -> Vec<u32> {
    let mut state = Lfm2MoeState::new(gpu, cfg).expect("state alloc");
    let mut emitted = Vec::with_capacity(steps);
    let mut pos = 0u32;
    let mut last_logits = Vec::new();
    for &token in seed {
        last_logits =
            hipfire_arch_lfm2moe::forward::decode_step(cfg, weights, &mut state, gpu, token, pos)
                .expect("decode step");
        pos += 1;
    }
    emitted.push(argmax(&last_logits));
    while emitted.len() < steps {
        let token = *emitted.last().unwrap();
        let logits =
            hipfire_arch_lfm2moe::forward::decode_step(cfg, weights, &mut state, gpu, token, pos)
                .expect("decode step");
        pos += 1;
        emitted.push(argmax(&logits));
    }
    state.free_gpu(gpu);
    emitted
}

/// `resident` | `layer` | `expert` | `auto` → the placement to load under.
///
/// * `resident` — `Placement::all_device` (the zero-diff default).
/// * `layer` — `Placement::dense_prefix`: host-places the first N *whole*
///   layers' own weights. NOTE: this is the DENSE tier — every routed expert
///   stays on the device. On LFM2.5-MoE the experts are ~98% of the bytes, so
///   this tier frees almost nothing; it is meaningful only on the dense
///   fixtures.
/// * `expert` — host-places the routed experts of the first N MoE layers
///   (`memory.moe_expert_budget`). This is the tier that matters for an
///   MoE model.
/// * `auto` — the production resolver (`carrier::resolve_partial_offload`),
///   which needs `HIPFIRE_GPU_LAYER_BUDGET` / `HIPFIRE_MOE_EXPERT_BUDGET` set
///   (e.g. `auto`). Spills routed experts before whole layers, so it is the
///   route that makes a model larger than the card fit.
fn placement_from_env(
    hfq: &hipfire_runtime::hfq::HfqFile,
    cfg: &Lfm2MoeConfig,
    gpu: &mut Gpu,
) -> (Placement, String) {
    let layers = cfg.num_hidden_layers;
    let spill = env_usize("HIPFIRE_LFM2_OFFLOAD_PARITY_LAYERS", 3).min(layers);
    match env::var("HIPFIRE_LFM2_OFFLOAD_PARITY_PLACEMENT").as_deref() {
        Ok("resident") => (Placement::all_device(layers), "resident".to_string()),
        Ok("auto") => {
            // Mirror `Lfm2MoeState::new`'s max_seq so the KV reservation the
            // resolver charges is the one the state will allocate.
            let max_seq = cfg.max_position_embeddings.min(8192);
            let free_vram = gpu.hip.get_vram_info().expect("hipMemGetInfo").0 as u64;
            let placement = hipfire_arch_lfm2moe::carrier::resolve_partial_offload(
                hfq, cfg, free_vram, max_seq, 0,
            )
            .unwrap_or_else(|e| panic!("auto placement refusal: {e}"));
            match placement {
                Some(p) => {
                    let label = format!(
                        "auto(host_layers={},host_expert_layers={})",
                        p.host_layers(),
                        p.host_expert_layers()
                    );
                    (p, label)
                }
                None => (
                    Placement::all_device(layers),
                    "auto(resident: model fits)".to_string(),
                ),
            }
        }
        Ok("expert") => (
            Placement {
                layer_residency: vec![WeightResidency::Resident; layers],
                expert_residency: (0..layers)
                    .map(|l| {
                        if l < spill {
                            ExpertResidency::HostMapped
                        } else {
                            ExpertResidency::Device
                        }
                    })
                    .collect(),
            },
            format!("expert-spill({spill})"),
        ),
        // `layer` and any unset/unknown value: the whole-layer (dense) tier.
        _ => (
            Placement::dense_prefix(layers, spill),
            format!("layer-spill({spill})"),
        ),
    }
}

/// Assert the load honoured the placement: a host layer's own weights are
/// host-located, a device layer's are not, and a host expert layer's experts
/// are host-located. Also asserts the allocator agrees with the claim in both
/// directions, so a placement that silently fell back to device uploads cannot
/// pass as a parity win, and the zero-diff default cannot silently spill.
fn assert_locality(
    cfg: &Lfm2MoeConfig,
    weights: &Lfm2MoeWeights,
    gpu: &Gpu,
    placement: &Placement,
    label: &str,
) {
    let mut host_layers = 0usize;
    let mut host_expert_layers = 0usize;
    for (l, layer) in weights.layers.iter().enumerate() {
        let layer_host = placement.layer(l) == WeightResidency::HostMapped;
        assert_eq!(
            gpu.host_located(&layer.operator_norm),
            layer_host,
            "{label}: layer {l} operator_norm locality"
        );
        if layer_host {
            host_layers += 1;
        }
        if let Ffn::Moe(moe) = &layer.ffn {
            let experts_host = placement.experts(l) == ExpertResidency::HostMapped;
            if experts_host {
                // A host-placed layer packs its experts into two host-mapped
                // blob owners with per-expert `Borrowed` views. The views
                // therefore report NOT host-located (`DeviceBuffer::
                // is_host_mapped` is about ownership, and they must not be
                // freed), so locality is asserted on the owners — and the
                // views are asserted to be non-owning, which is what keeps
                // `free_gpu` from double-freeing the blobs.
                assert!(
                    !moe.expert_blob_owners.is_empty(),
                    "{label}: layer {l} experts are host-placed but carry no packed blob owner"
                );
                for (i, owner) in moe.expert_blob_owners.iter().enumerate() {
                    assert!(
                        gpu.host_located(owner),
                        "{label}: layer {l} packed expert blob owner {i} is not host-mapped"
                    );
                }
                for (i, e) in moe.experts.iter().enumerate() {
                    assert!(
                        !gpu.host_located(&e.gate_up.buf),
                        "{label}: layer {l} expert {i} gate_up view claims ownership \
                         (must stay Borrowed into the blob)"
                    );
                }
            } else {
                // Device-placed experts are packed too (one blob per
                // projection per layer, so the placement census matches what is
                // really allocated). Their views are Borrowed regardless, so
                // locality is asserted on the owners.
                for (i, owner) in moe.expert_blob_owners.iter().enumerate() {
                    assert!(
                        !gpu.host_located(owner),
                        "{label}: layer {l} device-placed expert blob owner {i} is host-mapped"
                    );
                }
                assert!(
                    !gpu.host_located(&moe.experts[0].gate_up.buf),
                    "{label}: layer {l} expert gate_up is not device-placed"
                );
                assert!(
                    !gpu.host_located(&moe.experts[0].down.buf),
                    "{label}: layer {l} expert down is not device-placed"
                );
            }
            if experts_host {
                host_expert_layers += 1;
            }
        }
    }
    eprintln!(
        "{label}: {host_layers} host layers, {host_expert_layers} host-expert layers, \
         {} host-mapped owners",
        gpu.host_mapped_count()
    );
    if placement.is_fully_resident() {
        assert_eq!(
            gpu.host_mapped_count(),
            0,
            "{label}: a fully resident placement host-mapped something"
        );
    } else {
        assert!(
            gpu.host_mapped_count() > 0,
            "{label}: the placement host-mapped nothing — this proves nothing"
        );
    }
    // The globals never move.
    assert!(
        !gpu.host_located(&weights.embedding_norm),
        "{label}: embedding_norm is always resident"
    );
    assert!(
        !gpu.host_located(&weights.lm_head.buf),
        "{label}: lm_head is always resident"
    );
    let _ = cfg.max_position_embeddings;
}

/// Open the env-named HFQ, parse the config, init the GPU. `None` = skip.
fn setup() -> Option<(hipfire_runtime::hfq::HfqFile, Lfm2MoeConfig, Gpu)> {
    let path = env::var_os("HIPFIRE_LFM2_OFFLOAD_PARITY_MODEL")?;
    let mut gpu = match Gpu::init() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("skipped: no GPU ({e:?})");
            return None;
        }
    };
    let mut hfq =
        hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(&path)).expect("open HFQ");
    let cfg = Lfm2MoeConfig::from_hfq(&hfq).expect("parse config");
    let _ = &mut gpu;
    Some((hfq, cfg, gpu))
}

fn read_baseline(path: &str) -> Vec<u32> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read baseline {path}: {e}"))
        .split_whitespace()
        .map(|t| t.parse::<u32>().expect("baseline token id"))
        .collect()
}

/// One load per process: `resident` records the baseline, `layer` / `expert`
/// compare against it. See the module docs for the invocation.
#[test]
#[ignore = "needs an RDNA GPU + HIPFIRE_LFM2_OFFLOAD_PARITY_MODEL=<hfq path>"]
fn spill_parity_vs_baseline() {
    let Some((mut hfq, cfg, mut gpu)) = setup() else {
        return;
    };
    let baseline_path = env::var("HIPFIRE_LFM2_OFFLOAD_PARITY_BASELINE")
        .expect("HIPFIRE_LFM2_OFFLOAD_PARITY_BASELINE must name the baseline file");
    let steps = env_usize("HIPFIRE_LFM2_OFFLOAD_PARITY_STEPS", 32);
    let record = env::var("HIPFIRE_LFM2_OFFLOAD_PARITY_RECORD").is_ok_and(|v| v != "0");
    let seed = prompt_ids(&hfq);
    let (placement, label) = placement_from_env(&hfq, &cfg, &mut gpu);
    eprintln!(
        "model: {} layers  placement: {label}  record: {record}  seed: {seed:?}  steps: {steps}",
        cfg.num_hidden_layers
    );

    let weights = Lfm2MoeWeights::load_with_placement(&mut hfq, &cfg, &mut gpu, &placement)
        .unwrap_or_else(|e| panic!("{label} load: {e}"));
    assert_locality(&cfg, &weights, &gpu, &placement, &label);
    let tokens = greedy_decode(&cfg, &weights, &mut gpu, &seed, steps);
    weights.free_gpu(&mut gpu);
    assert_eq!(
        gpu.host_mapped_count(),
        0,
        "{label}: free_gpu must return every host-mapped owner"
    );
    assert_eq!(tokens.len(), steps);
    assert_not_degenerate(&tokens, &label);

    if record {
        let text: String = tokens
            .iter()
            .map(|t| format!("{t}\n"))
            .collect::<Vec<_>>()
            .join("");
        std::fs::write(&baseline_path, text)
            .unwrap_or_else(|e| panic!("write baseline {baseline_path}: {e}"));
        eprintln!("recorded {steps} {label} tokens to {baseline_path}");
        return;
    }

    let baseline = read_baseline(&baseline_path);
    assert_eq!(
        baseline.len(),
        tokens.len(),
        "baseline has {} tokens, this run emitted {}",
        baseline.len(),
        tokens.len()
    );
    if baseline != tokens {
        let at = baseline
            .iter()
            .zip(&tokens)
            .position(|(a, b)| a != b)
            .unwrap_or(baseline.len());
        panic!(
            "{label} diverged from the baseline at token {at} of {} (baseline was recorded \
             by the placement named in its run's log):\n  baseline: {:?}\n  {label}: {:?}",
            baseline.len(),
            &baseline[at..(at + 8).min(baseline.len())],
            &tokens[at..(at + 8).min(tokens.len())],
        );
    }
    eprintln!("{label} matches the baseline over {steps} tokens");
}

/// The no-resident oracle: two different spill depths must agree. Each load is
/// a different host/device split of the same bytes, so this holds a model that
/// cannot be loaded resident at all to the same standard.
#[test]
#[ignore = "needs an RDNA GPU + HIPFIRE_LFM2_OFFLOAD_PARITY_MODEL=<hfq path>"]
fn two_spill_depths_agree() {
    let Some((mut hfq, cfg, mut gpu)) = setup() else {
        return;
    };
    let steps = env_usize("HIPFIRE_LFM2_OFFLOAD_PARITY_STEPS", 32);
    let seed = prompt_ids(&hfq);
    let layers = cfg.num_hidden_layers;
    let shallow = env_usize("HIPFIRE_LFM2_OFFLOAD_PARITY_LAYERS", 3).min(layers);
    // Default deep = 3x shallow, which fits small fixtures outright. A model
    // near card size needs both depths chosen explicitly (each load must fit
    // beside whatever the other leaves behind), hence the override.
    let deep = env_usize("HIPFIRE_LFM2_OFFLOAD_PARITY_DEEP", shallow * 3).min(layers);
    assert!(
        deep > shallow,
        "two_spill_depths_agree needs two distinct depths: got shallow={shallow} \
         deep={deep} on a {layers}-layer model (set \
         HIPFIRE_LFM2_OFFLOAD_PARITY_DEEP above HIPFIRE_LFM2_OFFLOAD_PARITY_LAYERS)"
    );

    let mut decode_at = |depth: usize, gpu: &mut Gpu| {
        let label = format!("spill({depth})");
        let placement = Placement::dense_prefix(layers, depth);
        let weights = Lfm2MoeWeights::load_with_placement(&mut hfq, &cfg, gpu, &placement)
            .unwrap_or_else(|e| panic!("{label} load: {e}"));
        assert_locality(&cfg, &weights, gpu, &placement, &label);
        let tokens = greedy_decode(&cfg, &weights, gpu, &seed, steps);
        weights.free_gpu(gpu);
        assert_eq!(
            gpu.host_mapped_count(),
            0,
            "{label}: free_gpu must return every host-mapped owner"
        );
        tokens
    };

    let a = decode_at(shallow, &mut gpu);
    let b = decode_at(deep, &mut gpu);
    assert_eq!(
        a, b,
        "spill depth {shallow} and {deep} disagreed on the greedy token sequence"
    );
    assert_not_degenerate(&a, &format!("spill({shallow})"));
    eprintln!("spill depths {shallow} and {deep} agree over {steps} tokens");
}

/// Print the placement census for the env-named artifact. No GPU: this reads
/// the HFQ index only, so it is the cheap way to check whether the census
/// agrees with the artifact size before trusting a placement decision.
///
///   HIPFIRE_LFM2_OFFLOAD_PARITY_MODEL=<hfq> \
///     cargo test -p hipfire-arch-lfm2moe --test offload_parity census_report -- --nocapture
#[test]
fn census_report() {
    let Some(path) = env::var_os("HIPFIRE_LFM2_OFFLOAD_PARITY_MODEL") else {
        eprintln!("skipped: HIPFIRE_LFM2_OFFLOAD_PARITY_MODEL is unset");
        return;
    };
    let hfq = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(&path)).expect("open HFQ");
    let cfg = Lfm2MoeConfig::from_hfq(&hfq).expect("parse config");
    let bytes = hipfire_arch_lfm2moe::lfm2moe::lfm2_layer_bytes(&hfq, &cfg);
    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
    eprintln!(
        "census: {} layers, {} MoE layers (num_dense_layers={})",
        cfg.num_hidden_layers,
        cfg.num_hidden_layers - cfg.num_dense_layers,
        cfg.num_dense_layers
    );
    eprintln!(
        "  always_resident {:.1} MiB, non_expert total {:.1} MiB, expert total {:.1} MiB, \
         weight_total {:.1} MiB ({} bytes)",
        mib(bytes.always_resident),
        mib(bytes.non_expert.iter().sum()),
        mib(bytes.expert_total()),
        mib(bytes.weight_total()),
        bytes.weight_total()
    );
    for l in 0..cfg.num_hidden_layers {
        if bytes.expert[l] > 0 || l < 6 {
            eprintln!(
                "  layer {l:>2}: non_expert {:>8.1} MiB  expert {:>8.1} MiB  experts={}",
                mib(bytes.non_expert[l]),
                mib(bytes.expert[l]),
                if bytes.expert[l] == 0 {
                    0
                } else {
                    cfg.num_experts
                }
            );
        }
    }
    let on_disk = std::fs::metadata(std::path::Path::new(&path))
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!("  artifact file bytes {on_disk}");

    // Tokenizer sanity: the LFM family is BOS-trained, and a prompt that misses
    // BOS drives greedy decode into a single-token attractor (see
    // `Tokenizer::from_hfq_metadata`). Print the encoded ids so a degenerate
    // parity run can be diagnosed rather than guessed at.
    let meta: serde_json::Value =
        serde_json::from_str(&hfq.metadata_json).expect("metadata_json is JSON");
    eprintln!(
        "  metadata.architecture = {:?}",
        meta.get("architecture").and_then(|v| v.as_str())
    );
    let ids = prompt_ids(&hfq);
    eprintln!(
        "  prompt -> {} ids, first 12: {:?}",
        ids.len(),
        &ids[..ids.len().min(12)]
    );
}

/// Candidate parity prompts. One loads the model ONCE and reports how varied
/// greedy decode is under each, so a prompt that saturates can be identified
/// without a model reload per candidate.
const PROBE_PROMPTS: &[&str] = &[
    // 0: the committed PROMPT (chat-framed, thinking turn open).
    PROMPT,
    // 1: same shape, different task.
    "<|im_start|>user\nReverse a string in Python.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    // 2: with an explicit system turn.
    "<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n<|im_start|>user\nName three primary colors.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    // 3: assistant turn left closed (no think tag).
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
    // 4: no chat framing at all (the known-degenerate control).
    "Write a Python function that returns the nth Fibonacci number.",
];

/// Report greedy-decode variety per candidate prompt under one placement.
/// No baseline: this is a diagnostic for choosing the parity prompt.
#[test]
#[ignore = "needs an RDNA GPU + HIPFIRE_LFM2_OFFLOAD_PARITY_MODEL=<hfq path>"]
fn prompt_probe() {
    let Some((mut hfq, cfg, mut gpu)) = setup() else {
        return;
    };
    let steps = env_usize("HIPFIRE_LFM2_OFFLOAD_PARITY_STEPS", 48);
    let (placement, label) = placement_from_env(&hfq, &cfg, &mut gpu);
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer from HFQ metadata");
    let weights = Lfm2MoeWeights::load_with_placement(&mut hfq, &cfg, &mut gpu, &placement)
        .unwrap_or_else(|e| panic!("{label} load: {e}"));
    eprintln!("prompt_probe under placement {label}, {steps} greedy steps:");
    for (i, p) in PROBE_PROMPTS.iter().enumerate() {
        let ids = tokenizer.encode(p);
        let tokens = greedy_decode(&cfg, &weights, &mut gpu, &ids, steps);
        let mut sorted = tokens.clone();
        sorted.sort_unstable();
        sorted.dedup();
        eprintln!(
            "  [{i}] {} prompt ids -> {} distinct of {steps}: {:?}",
            ids.len(),
            sorted.len(),
            &tokens[..tokens.len().min(16)]
        );
    }
    weights.free_gpu(&mut gpu);
}
