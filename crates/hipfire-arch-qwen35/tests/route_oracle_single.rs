// SPDX-License-Identifier: MIT
// Copyright (c) 2026 alpineq
// hipfire — see LICENSE and NOTICE in the project root.

//! G1 single-route numerical/state oracle (`g1-route-oracles-nonllama`).
//!
//! Two ignored, env-gated GPU tests — one per single-GPU Qwen35 route:
//!   * `qwen35_dense_single_route_oracle` on `HIPFIRE_QWEN35_FIXTURE` (dense 27B)
//!   * `qwen35_moe_single_route_oracle` on `HIPFIRE_ORNITH_FIXTURE` (A3B MoE)
//!
//! Structure follows the LLaMA template oracle
//! (`crates/hipfire-arch-llama/src/carrier.rs`
//! `pinned_fixture_manifest_legacy_parity_oracle`) and the pp_parity load
//! pattern (`run_pp1`): each oracle drives the SAME single route TWICE over
//! the same committed prompt from a fresh load each time (fresh load A, fresh
//! load B), greedy, same pinned KV tier (`fwht3` — never `auto`), and at
//! every committed position asserts: token id `assert_eq`; full-vocab logits
//! `assert_eq` bitwise (same route, same op order — no tolerance); KV
//! geometry + byte extents; position counter (`compact_offset == 0`, same
//! position count); DeltaNet recurrent/conv state bytes; and, for MoE, the
//! last-layer router top-k selection (per-layer is NOT OBSERVABLE — see the
//! evidence block). Then prints an evidence block.
//!
//! This proves the Single route is deterministic and its state identities
//! are reproducible from a fresh load — the precondition every later mesh
//! comparison depends on.
//!
//! Anti-vacuity: load B with a different KV tier via
//! `HIPFIRE_ROUTE_ORACLE_KV_B` (e.g. `q8`) and both oracles go RED on the KV
//! geometry assertion. Run explicitly:
//!
//!   HIPFIRE_QWEN35_FIXTURE=$HOME/.hipfire/models/qwen3.8-27b.mq4-xt \
//!   HIPFIRE_ORNITH_FIXTURE=$HOME/.hipfire/models/ornith-1.5-35b-a3b.mq4r \
//!   HIP_VISIBLE_DEVICES=0 flock -w 3600 /tmp/hipfire-gpu.lock \
//!       cargo test -p hipfire-arch-qwen35 --locked --test route_oracle_single \
//!           -- --ignored --test-threads=1

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, LayerType, Qwen35Scratch, StateQuant};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::path::Path;
use std::sync::Mutex;

/// Serializes the two GPU oracles: both drive device 0 with multi-GB loads,
/// so parallel execution in one process would OOM / perturb timings.
static GPU_ORACLE_LOCK: Mutex<()> = Mutex::new(());

/// Committed prompt, byte-identical for load A and load B.
const PROMPT: &str = "The capital of France is located in";
/// Greedy decode steps after the prompt (matches the LLaMA template's 12).
const N_GENERATE: usize = 12;
/// KV / scratch position budget. Must exceed prompt + generated.
const MAX_SEQ: usize = 128;
const REPEAT_WINDOW: usize = 64;
/// Pinned KV tier for both loads. `auto` resolves to Fwht3 today but the
/// oracle never relies on that — the tier is constructed explicitly.
const PINNED_KV_TIER: &str = "fwht3";
/// Env knob selecting load B's KV tier. Unset (or `fwht3`) = same pinned
/// tier as A (green). Set to `q8`/`asym3` for the anti-vacuity red run.
const KV_TIER_B_ENV: &str = "HIPFIRE_ROUTE_ORACLE_KV_B";

// Fixture identities per `/home/kaden/hipfire-g4-val/G5-FIXTURE-LOCK.md`
// (full-file sha256, exact bytes, verified on both hosts).
const DENSE_SIZE: u64 = 14_980_361_216;
const DENSE_SHA256: &str = "9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7";
const MOE_SIZE: u64 = 18_700_570_368;
const MOE_SHA256: &str = "84103fcc8ade42aa2ac8ec01176df7a4ead5e94810597c9fae2f6763152a3ac6";

fn fixture_or_skip(env: &str, def: &str, role: &str) -> Option<String> {
    let path = std::env::var(env).unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.hipfire/models/{def}")
    });
    if !Path::new(&path).is_file() {
        eprintln!("skip: {role} fixture absent ({path}); set {env}");
        return None;
    }
    Some(path)
}

/// Full-file sha256, streamed through the external `sha256sum` so the digest
/// is read from the file itself — never a hardcoded string. (`sha2` is not a
/// direct dependency of this crate and the oracle must not touch Cargo.toml.)
fn fixture_sha256(path: &str) -> Option<String> {
    let out = std::process::Command::new("sha256sum")
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    std::str::from_utf8(&out.stdout)
        .ok()?
        .split_whitespace()
        .next()
        .map(|s| s.to_string())
}

fn build_kv(
    gpu: &mut Gpu,
    is_kv_layer: &[bool],
    n_kv_heads: usize,
    head_dim: usize,
    tier: &str,
) -> KvCache {
    match tier {
        "fwht3" => KvCache::new_gpu_fwht3_capped_filtered(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            MAX_SEQ,
            MAX_SEQ,
        )
        .expect("fwht3 KV"),
        "asym3" => KvCache::new_gpu_asym3_capped_filtered(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            MAX_SEQ,
            MAX_SEQ,
        )
        .expect("asym3 KV"),
        "q8" => KvCache::new_gpu_q8_capped_filtered(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            MAX_SEQ,
            MAX_SEQ,
        )
        .expect("q8 KV"),
        other => panic!("{KV_TIER_B_ENV}='{other}' unknown (expected fwht3|asym3|q8)"),
    }
}

/// Plain-data KV geometry snapshot: comparable after the owning cache is freed.
#[derive(Debug)]
struct KvGeometry {
    quant_q8: bool,
    quant_int8: bool,
    quant_asym3: bool,
    quant_fwht: bool,
    kv_dim: usize,
    max_seq: usize,
    physical_cap: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_layers: usize,
    k_bytes: Vec<usize>,
    v_bytes: Vec<usize>,
    compact_offset: usize,
}

fn kv_geometry(kv: &KvCache) -> KvGeometry {
    KvGeometry {
        quant_q8: kv.quant_q8,
        quant_int8: kv.quant_int8,
        quant_asym3: kv.quant_asym3,
        quant_fwht: kv.quant_fwht,
        kv_dim: kv.kv_dim,
        max_seq: kv.max_seq,
        physical_cap: kv.physical_cap,
        n_kv_heads: kv.n_kv_heads,
        head_dim: kv.head_dim,
        n_layers: kv.k_gpu.len(),
        k_bytes: kv.k_gpu.iter().map(|t| t.byte_size()).collect(),
        v_bytes: kv.v_gpu.iter().map(|t| t.byte_size()).collect(),
        compact_offset: kv.compact_offset,
    }
}

fn assert_kv_geometry_eq(a: &KvGeometry, b: &KvGeometry, tag: &str, tier_a: &str, tier_b: &str) {
    assert_eq!(
        (a.quant_q8, a.quant_int8, a.quant_asym3, a.quant_fwht),
        (b.quant_q8, b.quant_int8, b.quant_asym3, b.quant_fwht),
        "{tag}: KV tier flags differ (A tier={tier_a} flags={:?}, B tier={tier_b} flags={:?})",
        (a.quant_q8, a.quant_int8, a.quant_asym3, a.quant_fwht),
        (b.quant_q8, b.quant_int8, b.quant_asym3, b.quant_fwht),
    );
    assert_eq!(a.kv_dim, b.kv_dim, "{tag}: kv_dim mismatch");
    assert_eq!(a.max_seq, b.max_seq, "{tag}: max_seq mismatch");
    assert_eq!(
        a.physical_cap, b.physical_cap,
        "{tag}: physical_cap mismatch"
    );
    assert_eq!(a.n_kv_heads, b.n_kv_heads, "{tag}: n_kv_heads mismatch");
    assert_eq!(a.head_dim, b.head_dim, "{tag}: head_dim mismatch");
    assert_eq!(a.n_layers, b.n_layers, "{tag}: layer count mismatch");
    assert_eq!(a.k_bytes.len(), b.k_bytes.len(), "{tag}: K extent count");
    assert_eq!(a.v_bytes.len(), b.v_bytes.len(), "{tag}: V extent count");
    for (i, (x, y)) in a.k_bytes.iter().zip(&b.k_bytes).enumerate() {
        assert_eq!(x, y, "{tag}: layer {i} K byte extent mismatch");
    }
    for (i, (x, y)) in a.v_bytes.iter().zip(&b.v_bytes).enumerate() {
        assert_eq!(x, y, "{tag}: layer {i} V byte extent mismatch");
    }
    assert_eq!(
        a.compact_offset, b.compact_offset,
        "{tag}: compact_offset (position counter) mismatch"
    );
}

fn download_bytes(gpu: &Gpu, t: &rdna_compute::GpuTensor) -> Vec<u8> {
    let mut buf = vec![0u8; t.byte_size()];
    gpu.hip
        .memcpy_dtoh(&mut buf, &t.buf)
        .expect("download bytes");
    buf
}

/// Byte image of the DeltaNet recurrent + conv state.
///
/// NOTE on the Q8 extent: `DeltaNetState::new_batched_with_quant` stores the
/// int8 S matrix in a `GpuTensor` whose shape/dtype claim F32
/// (`weights.rs`), so `byte_size()` over-reports 4x. The true extent is one
/// byte per element — `n_heads * s_dim * s_dim` per delta layer — computed
/// from the config with the same formula as the constructor.
fn dn_state_bytes(
    gpu: &Gpu,
    dn: &DeltaNetState,
    config: &hipfire_arch_qwen35::qwen35::Qwen35Config,
) -> Vec<Vec<u8>> {
    assert_eq!(
        dn.quant,
        StateQuant::Q8,
        "oracle pins DeltaNet Q8 state on both loads"
    );
    let s_size =
        config.linear_num_value_heads * config.linear_key_head_dim * config.linear_key_head_dim;
    let mut out = Vec::new();
    for t in &dn.s_matrices {
        let mut buf = vec![0u8; s_size];
        gpu.hip.memcpy_dtoh(&mut buf, &t.buf).expect("dn S matrix");
        out.push(buf);
    }
    for t in dn
        .s_scales
        .iter()
        .chain(&dn.conv_states)
        .chain(&dn.s_ef_residual)
    {
        out.push(download_bytes(gpu, t));
    }
    out
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Full record of one fresh-load decode run. Everything GPU-owned is
/// downloaded to host before the run frees device memory, so run B can be
/// compared after run A is gone (two resident 27B/35B loads would OOM one GPU).
struct RunRecord {
    tier: String,
    fed: Vec<u32>,
    choices: Vec<u32>,
    logits: Vec<Vec<f32>>,
    kv_geo: KvGeometry,
    kv_payload: Vec<Vec<u8>>,
    dn_payload: Vec<Vec<u8>>,
    dn_ef_on: bool,
    alias: bool,
    /// Last-layer router state per position (MoE only; `None` for dense).
    router_logits: Option<Vec<Vec<f32>>>,
    router_topk_idx: Option<Vec<Vec<u8>>>,
    router_topk_w: Option<Vec<Vec<f32>>>,
}

fn decode_run(path: &str, kv_tier: &str, prompt_tokens: &[u32], label: &str) -> RunRecord {
    let mut hfq = HfqFile::open(Path::new(path)).expect("open hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let is_kv_layer: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == LayerType::FullAttention)
        .collect();
    let mut gpu = Gpu::init().expect("Gpu::init");
    let layout = qwen35::Layout::single(config.n_layers);
    let mut src = qwen35::HfqSource::new(&mut hfq, &config);
    let weights = qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
        .expect("load_weights");
    drop(src);
    drop(hfq);
    let mut kv = build_kv(
        &mut gpu,
        &is_kv_layer,
        config.n_kv_heads,
        config.head_dim,
        kv_tier,
    );
    let mut dn = DeltaNetState::new_with_quant(&mut gpu, &config, StateQuant::Q8).expect("dn");
    let scratch =
        Qwen35Scratch::new_with_kv_max(&mut gpu, &config, REPEAT_WINDOW, MAX_SEQ).expect("scratch");

    let total = prompt_tokens.len() + N_GENERATE;
    assert!(
        total <= MAX_SEQ,
        "{label}: position budget {total} exceeds MAX_SEQ={MAX_SEQ}"
    );
    let mut fed = Vec::with_capacity(total);
    let mut choices = Vec::with_capacity(total);
    let mut logits_all = Vec::with_capacity(total);
    let mut router_logits = Vec::new();
    let mut router_topk_idx = Vec::new();
    let mut router_topk_w = Vec::new();
    let mut next_token: u32 = 0;
    for pos in 0..total {
        let token = if pos < prompt_tokens.len() {
            prompt_tokens[pos]
        } else {
            next_token
        };
        fed.push(token);
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, token, pos, &mut kv, &mut dn, &scratch,
        )
        .expect("forward_scratch");
        let logits = gpu.download_f32(&scratch.logits).expect("download logits");
        assert_eq!(
            logits.len(),
            config.vocab_size,
            "{label}: logit width must equal vocab at pos {pos}"
        );
        next_token = argmax(&logits);
        choices.push(next_token);
        logits_all.push(logits);
        // MoE only: snapshot the shared router scratch. These buffers are
        // threaded through EVERY MoE layer (MoeScratchRef::from_scratch), so
        // after a forward only the LAST MoE layer's values survive — that is
        // what is compared. Per-layer selection is NOT OBSERVABLE (see below).
        if config.num_experts > 0 {
            let rl = scratch
                .moe_router_logits
                .as_ref()
                .expect("MoE scratch must carry router logits");
            router_logits.push(gpu.download_f32(rl).expect("router logits"));
            let ti = scratch
                .moe_topk_indices
                .as_ref()
                .expect("MoE scratch must carry topk indices");
            router_topk_idx.push(download_bytes(&gpu, ti));
            let tw = scratch
                .moe_topk_weights
                .as_ref()
                .expect("MoE scratch must carry topk weights");
            router_topk_w.push(gpu.download_f32(tw).expect("topk weights"));
        }
    }

    let kv_geo = kv_geometry(&kv);
    let mut kv_payload = Vec::with_capacity(2 * kv.k_gpu.len() + 2);
    for t in kv.k_gpu.iter().chain(kv.v_gpu.iter()) {
        kv_payload.push(download_bytes(&gpu, t));
    }
    // Rotation tables (FWHT signs / Givens cos-sin): fixed-seed state, part of
    // the reproducible identity. Absent on the q8 tier — record length-tagged.
    for t in [&kv.givens_cos, &kv.givens_sin].into_iter().flatten() {
        kv_payload.push(download_bytes(&gpu, t));
    }
    let dn_payload = dn_state_bytes(&gpu, &dn, &config);
    let dn_ef_on = !dn.s_ef_residual.is_empty();
    let alias = weights.lm_head_aliases_embd;
    let is_moe = config.num_experts > 0;

    scratch.free_gpu(&mut gpu).expect("free scratch");
    dn.free_gpu(&mut gpu);
    let _ = kv.free_gpu(&mut gpu);
    weights.free_gpu(&mut gpu);
    gpu.drain_pool();

    RunRecord {
        tier: kv_tier.to_string(),
        fed,
        choices,
        logits: logits_all,
        kv_geo,
        kv_payload,
        dn_payload,
        dn_ef_on,
        alias,
        router_logits: is_moe.then_some(router_logits),
        router_topk_idx: is_moe.then_some(router_topk_idx),
        router_topk_w: is_moe.then_some(router_topk_w),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_oracle(
    tag: &str,
    fixture_env: &str,
    fixture_default: &str,
    expect_moe: bool,
    pinned_size: u64,
    pinned_sha256: &str,
) {
    let Some(path) = fixture_or_skip(fixture_env, fixture_default, tag) else {
        return;
    };
    let Ok(meta) = std::fs::metadata(&path) else {
        eprintln!("skip: {tag}: cannot stat fixture ({path})");
        return;
    };
    assert_eq!(
        meta.len(),
        pinned_size,
        "{tag}: fixture size mismatch — not the pinned artifact"
    );
    let sha = fixture_sha256(&path).expect("sha256sum fixture");
    assert_eq!(
        sha, pinned_sha256,
        "{tag}: fixture content mismatch — not the pinned artifact"
    );
    if Gpu::init().is_err() {
        eprintln!("skip: {tag}: no GPU");
        return;
    }

    // Pin the route class from the config before any decode: dense must have
    // no experts, MoE must.
    let hfq = HfqFile::open(Path::new(&path)).expect("open hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let is_moe_cfg = config.num_experts > 0;
    assert_eq!(
        is_moe_cfg, expect_moe,
        "{tag}: route-class mismatch (num_experts={})",
        config.num_experts
    );
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("fixture tokenizer");
    let prompt_tokens = tokenizer.encode(PROMPT);
    let (n_layers, dim, n_exp, k_top, vocab, n_kv_heads, head_dim) = (
        config.n_layers,
        config.dim,
        config.num_experts,
        config.num_experts_per_tok,
        config.vocab_size,
        config.n_kv_heads,
        config.head_dim,
    );
    drop(hfq);
    drop(config);
    eprintln!("{tag}: fixture={path} size={} sha256={sha}", meta.len());
    eprintln!("{tag}: prompt={PROMPT:?} tokens={prompt_tokens:?}");

    let tier_b = std::env::var(KV_TIER_B_ENV).unwrap_or_else(|_| PINNED_KV_TIER.to_string());
    let a = decode_run(&path, PINNED_KV_TIER, &prompt_tokens, &format!("{tag}[A]"));
    let b = decode_run(&path, &tier_b, &prompt_tokens, &format!("{tag}[B]"));
    assert_eq!(
        &a.fed[..prompt_tokens.len()],
        &prompt_tokens,
        "{tag}: load-A prompt inputs must equal the committed prompt"
    );
    assert_eq!(
        b.fed, a.fed,
        "{tag}: load-B inputs must equal load-A committed inputs"
    );

    // Route identity: single-device Layout + forward_scratch on both loads.
    eprintln!(
        "{tag}: route identity — single (Layout::single + forward_scratch) on both loads; \
         n_layers={n_layers} dim={dim} experts={n_exp} k_top={k_top} vocab={vocab} \
         n_kv_heads={n_kv_heads} head_dim={head_dim}"
    );
    // Alias identity.
    assert_eq!(
        a.alias, b.alias,
        "{tag}: lm_head_aliases_embd must match between fresh loads"
    );
    eprintln!(
        "{tag}: lm_head aliases embed_tokens = {} (both loads)",
        a.alias
    );

    // KV geometry + byte extents (fires first on a tier-mismatched B).
    assert_kv_geometry_eq(&a.kv_geo, &b.kv_geo, tag, &a.tier, &b.tier);
    assert_eq!(
        a.kv_geo.compact_offset, 0,
        "{tag}: position counter moved — plain decode must not compact"
    );
    assert_eq!(
        b.kv_geo.compact_offset, 0,
        "{tag}: position counter moved — plain decode must not compact"
    );
    eprintln!(
        "{tag}: kv geometry — tier A={} B={} kv_dim={} max_seq={} cap={} n_heads={} head_dim={} layers={}",
        a.tier,
        b.tier,
        a.kv_geo.kv_dim,
        a.kv_geo.max_seq,
        a.kv_geo.physical_cap,
        a.kv_geo.n_kv_heads,
        a.kv_geo.head_dim,
        a.kv_geo.n_layers,
    );

    // Greedy lockstep over the committed positions: token ids bitwise-equal,
    // full-vocab logits bitwise-equal (same route, same op order — no tolerance).
    assert_eq!(a.logits.len(), b.logits.len(), "{tag}: position count");
    let total = a.logits.len();
    let mut worst_logit_diff: f32 = 0.0;
    for (pos, (la, lb)) in a.logits.iter().zip(&b.logits).enumerate() {
        assert_eq!(
            la.len(),
            lb.len(),
            "{tag}: logit width mismatch at committed position {pos}"
        );
        let mut diff: f32 = 0.0;
        for (x, y) in la.iter().zip(lb.iter()) {
            diff = diff.max((x - y).abs());
        }
        worst_logit_diff = worst_logit_diff.max(diff);
        assert_eq!(
            la, lb,
            "{tag}: logits differ at committed position {pos} (max abs diff {diff})"
        );
        assert_eq!(
            a.choices[pos], b.choices[pos],
            "{tag}: token-id mismatch at committed position {pos}"
        );
        eprintln!(
            "{tag}: pos {:>2} fed {:>6} choice {:>6}",
            pos, a.fed[pos], a.choices[pos]
        );
    }
    assert_eq!(
        worst_logit_diff, 0.0,
        "{tag}: worst logit diff must be exactly 0.0 for identical routes"
    );

    // End-state KV payload parity (all layers, K+V, plus rotation tables).
    assert_eq!(
        a.kv_payload.len(),
        b.kv_payload.len(),
        "{tag}: KV payload tensor count"
    );
    let mut kv_bytes = 0usize;
    for (i, (x, y)) in a.kv_payload.iter().zip(&b.kv_payload).enumerate() {
        kv_bytes += x.len();
        assert_eq!(x, y, "{tag}: KV payload tensor {i} not byte-identical");
    }
    eprintln!("{tag}: KV payload — {kv_bytes} bytes compared, 0 byte diffs");

    // DeltaNet recurrent/conv state parity.
    assert_eq!(
        a.dn_payload.len(),
        b.dn_payload.len(),
        "{tag}: DeltaNet state tensor count"
    );
    assert_eq!(
        a.dn_ef_on, b.dn_ef_on,
        "{tag}: DN error-feedback state diverged"
    );
    let mut dn_bytes = 0usize;
    for (i, (x, y)) in a.dn_payload.iter().zip(&b.dn_payload).enumerate() {
        dn_bytes += x.len();
        assert_eq!(x, y, "{tag}: DeltaNet state tensor {i} not byte-identical");
    }
    eprintln!(
        "{tag}: DeltaNet state (Q8, ef={}) — {dn_bytes} bytes compared, 0 byte diffs",
        if a.dn_ef_on { "on" } else { "off" }
    );

    // Router selection.
    match (&a.router_logits, &b.router_logits) {
        (None, None) => {
            assert!(
                !expect_moe,
                "{tag}: dense route must not produce router state"
            );
            eprintln!("{tag}: router: N/A (dense, moe_router_logits=None on both loads)");
        }
        (Some(ra), Some(rb)) => {
            assert!(expect_moe, "{tag}: unexpected router state on dense route");
            assert_eq!(ra.len(), total, "{tag}: router position count (A)");
            assert_eq!(rb.len(), total, "{tag}: router position count (B)");
            let ia = a.router_topk_idx.as_ref().expect("A topk indices");
            let ib = b.router_topk_idx.as_ref().expect("B topk indices");
            let wa = a.router_topk_w.as_ref().expect("A topk weights");
            let wb = b.router_topk_w.as_ref().expect("B topk weights");
            for pos in 0..total {
                assert_eq!(
                    ra[pos], rb[pos],
                    "{tag}: last-layer router logits differ at pos {pos}"
                );
                assert_eq!(
                    ia[pos], ib[pos],
                    "{tag}: last-layer router top-k differs at pos {pos}"
                );
                assert_eq!(
                    wa[pos], wb[pos],
                    "{tag}: last-layer router top-k weights differ at pos {pos}"
                );
            }
            eprintln!(
                "{tag}: router: last-layer top-k (k={k_top}) + router logits \
                 compared at all {total} positions, 0 diffs; per-layer router top-k: \
                 NOT OBSERVABLE from a test — moe_router_logits / moe_topk_* are \
                 shared scratch threaded through every MoE layer \
                 (MoeScratchRef::from_scratch), so only the last layer survives; \
                 expert_stats_enabled() is private and dump_expert_stats() is \
                 aggregate-only. Nearest seam: a pub per-layer router snapshot \
                 hook in forward_scratch_layers (or make moe_ffn_decode_with_scratch pub)."
            );
        }
        _ => panic!("{tag}: router presence diverged between fresh loads"),
    }

    eprintln!(
        "{tag}: PASS — {total} committed positions ({} prompt + {N_GENERATE} greedy), \
         worst logit diff {worst_logit_diff:.1}, tokens {:?}",
        prompt_tokens.len(),
        a.fed,
    );
}

#[test]
#[ignore]
fn qwen35_dense_single_route_oracle() {
    let _guard = GPU_ORACLE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    run_oracle(
        "g1-single-oracle[dense]",
        "HIPFIRE_QWEN35_FIXTURE",
        "qwen3.8-27b.mq4-xt",
        false,
        DENSE_SIZE,
        DENSE_SHA256,
    );
}

#[test]
#[ignore]
fn qwen35_moe_single_route_oracle() {
    let _guard = GPU_ORACLE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    run_oracle(
        "g1-single-oracle[moe]",
        "HIPFIRE_ORNITH_FIXTURE",
        "ornith-1.5-35b-a3b.mq4r",
        true,
        MOE_SIZE,
        MOE_SHA256,
    );
}
