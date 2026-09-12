// SPDX-License-Identifier: MIT
// Copyright (c) 2026 alpineq
// hipfire — see LICENSE and NOTICE in the project root.

//! #666 G1 mesh-route numerical/state oracles (env-gated, ignored by default).
//!
//! Each test stands up the single-GPU route and one mesh route from FRESH loads
//! of the SAME pinned fixture, pins the SAME KV tier (Fwht3) and DeltaNet state
//! quant (Q8) on both sides, then compares both routes at every committed
//! greedy position — interleaved lockstep where VRAM allows (PP/TP), sequential
//! record/replay for the MoE EP tests (two stacked meshes exceed device VRAM;
//! deterministic kernels make the two equivalent) — asserting token ids,
//! logits, KV geometry/byte extents, the position counter, alias identity and
//! route identity, then printing an evidence block. Pattern follows the template
//! oracle (`hipfire-arch-llama` `pinned_fixture_manifest_legacy_parity_oracle`)
//! and goes beyond `pp_parity.rs` (which compares argmax tokens only).
//!
//! Relationship decisions (verified from code, not assumed):
//! * PP2-vs-single — BOUNDED (tokens `assert_eq`, max abs logit diff <= 2.0).
//!   PP is a whole-layer split with no op reordering, but the Q8 DeltaNet
//!   state takes different requant paths per route: the single route uses the
//!   error-feedback sigma-delta requant (default-ON, deterministic), while the
//!   multi route's `new_with_quant_multi` leaves EF residuals empty so the
//!   kernel falls back to the legacy stochastic requant (`weights.rs`).
//!   Measured: 2.1e-3 at pos 0 growing to 8.1e-1 over 27 positions,
//!   bit-identical across fresh processes, tokens exact throughout. The
//!   `pp_parity.rs` bit-exact floor does not transfer to fwht3+Q8 on 27B
//!   (validated there on asym3+0.8B).
//! * TP2-vs-single — BOUNDED (tokens `assert_eq`, max abs logit diff <= 1.0).
//!   Dense TP shards heads/projections per rank and sums the row-parallel
//!   partials through the deterministic rooted peer reduce
//!   (`dense_tp_all_reduce_sum_f32` → `all_reduce_sum_f32_peer_rooted`,
//!   `forward.rs:3664`), a different f32 summation order than the single
//!   route's direct accumulation — so `assert_eq` on logits is unwarranted. On
//!   top of the order term, per-step Q8 recurrent-state writes diverge by
//!   quantization LSBs and the barely-contractive DN recurrence (α≈0.99)
//!   random-walks the difference up: measured pos0 2.141e-3 — bit-identical
//!   across PP/TP/EF-on/EF-off runs, i.e. a deterministic multi-standup
//!   first-write effect — growing to worst 4.6e-1 over 27 positions EF-on
//!   (3.0 with `HIPFIRE_DN_STATE_EF=0`). Bound 1.0 is 2x measured worst;
//!   tokens exact throughout (3 runs). Precedent: `qwen_dense_tp2_parity.rs`
//!   claims argmax-exact + relative error < 3e-3, never bit-exact.
//! * EP2-vs-single (MoE) — MEASURING-BUT-RED (bound 1e-3 asserted, test fails
//!   honestly). Snapshot lockstep pins KV + DeltaNet + the FULL scratch (30
//!   fixed + 14 MoE-opt incl. top-k indices/weights and `moe_down_expanded` +
//!   pos_bufs, all by true `DeviceBuffer::size`) yet the run holds ~1e-5 for
//!   17 positions then trips a deterministic 3.599e-2 cliff at pos 17 —
//!   bit-identical to the earlier KV/DN-only snapshot run, so the carrier
//!   lives OUTSIDE the snapshot set (suspects: `EpMesh.partials`, RCCL/peer
//!   staging, rank streams, per-rank expert shard/dummy state, or a
//!   routing-dependent dropped/duplicated expert; top-k is discrete and
//!   1e-5-scale grouping noise flips experts under interleaved forcing —
//!   measured 6.5e-2 cliff at pos 2, token flip at pos 5 — so lockstep, not
//!   interleaving, is the design). The bound is NOT widened to hide it.
//! * EP4-vs-EP2 (MoE) — MEASURING-BUT-RED (bound 1e-3 asserted, test fails
//!   honestly). Same snapshot-lockstep design (EP2 rank-0 bytes are the
//!   reference); pos 0–1 hold ~1e-5, then a deterministic 2.695e-3 cliff at
//!   pos 2 — a DIFFERENT position and magnitude than EP2-vs-single's
//!   pos-17/3.6e-2, so the trip is route-pair-specific (stride `e%4` vs `e%2`
//!   partial contents + RCCL summation order), not a universal step counter.
//!   Same out-of-snapshot suspect set as EP2. The rooted-peer order
//!   (`multi_gpu.rs:1759`) is NOT the decode path — opt-in via
//!   `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1` only — so `assert_eq` is unwarranted
//!   and the bound is NOT widened to hide the trip.
//!
//! Loader note: `load_model_ep` (loader `lib.rs`) has NO MoE serve path —
//! `qwen35_ep_moe_refusal` refuses `num_experts > 0` ("use TP or single-GPU")
//! and the MoE branch additionally requires `tp == 4`. The EP tests therefore
//! drive the retained qwen35-level EP substrate directly (replicated
//! `load_weights` under the `set_ep_expert_shard` TLS context +
//! `forward_ep`), which is the exact pattern of the in-tree
//! `hipfire-runtime/examples/ep_decode_parity.rs` validation. No `pub` seam
//! is missing: every symbol used here is already `pub`. Nothing is NOT WIRED.
//!
//! Run (EVERY GPU command flock-wrapped):
//! ```sh
//! HIPFIRE_QWEN35_FIXTURE=$HOME/.hipfire/models/qwen3.8-27b.mq4-xt \
//! HIPFIRE_ORNITH_FIXTURE=$HOME/.hipfire/models/ornith-1.5-35b-a3b.mq4r \
//! HIPFIRE_HAVE_2_GPU=1 HIP_VISIBLE_DEVICES=0,1,2,3 \
//! flock -w 3600 /tmp/hipfire-gpu.lock \
//! cargo test -p hipfire-arch-qwen35 --locked --test route_oracle_mesh \
//! -- --ignored --test-threads=1
//! ```

use hipfire_arch_qwen35::qwen35::{
    self, DeltaNetState, HfqSource, Layout, Qwen35Config, Qwen35Scratch, Qwen35ScratchSet,
    Qwen35Weights, StateQuant,
};
use hipfire_runtime::ep::ensure_rank_streams;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{KvCache, KvCacheExt};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{LazyLock, Mutex};

const KV_MAX: usize = 4096;
const SCRATCH_WINDOW: usize = 64;
const N_DECODE: usize = 12;
const TIER: &str = "fwht3";
const STATE_QUANT: &str = "q8";
const PROMPT: &str = "Write a one-sentence greeting.";
const DENSE_SIZE: u64 = 14_980_361_216;
const DENSE_SHA256: &str = "9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7";
const MOE_SIZE: u64 = 18_700_570_368;
const MOE_SHA256: &str = "84103fcc8ade42aa2ac8ec01176df7a4ead5e94810597c9fae2f6763152a3ac6";

// Absolute logit bounds for the bounded relationships. Set from measured
// worst-case per-position diffs (see evidence output) at ~2x headroom, rounded
// up; a wrong expert / wrong reduction / wrong band would move logits by
// orders of magnitude more, and any token flip inside a bound is a hard
// failure. TP measured worst 4.6e-1 (EF-on, 27 positions) → 1.0; PP measured
// worst 8.1e-1 → 2.0.
const BOUND_PP_ABS: f32 = 2.0;
const BOUND_TP_ABS: f32 = 1.0;
const BOUND_EP_ABS: f32 = 1e-3;

// ── gates / fixtures ─────────────────────────────────────────────────────────

fn have_mesh() -> bool {
    std::env::var("HIPFIRE_HAVE_2_GPU").as_deref() == Ok("1")
}

fn dense_fixture() -> String {
    std::env::var("HIPFIRE_QWEN35_FIXTURE").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.hipfire/models/qwen3.8-27b.mq4-xt")
    })
}

fn moe_fixture() -> String {
    std::env::var("HIPFIRE_ORNITH_FIXTURE").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.hipfire/models/ornith-1.5-35b-a3b.mq4r")
    })
}

static SHA256_CACHE: LazyLock<Mutex<HashMap<String, String>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Full-file sha256 via the system `sha256sum`, cached per process so the four
/// tests hash each fixture once. `qwen35` has no hash crate in its deps and the
/// oracle must not touch `Cargo.toml`.
fn sha256_of(path: &str) -> Option<String> {
    if let Some(hit) = SHA256_CACHE.lock().unwrap().get(path) {
        return Some(hit.clone());
    }
    let out = std::process::Command::new("sha256sum")
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let hex = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_string();
    if hex.len() != 64 {
        return None;
    }
    SHA256_CACHE
        .lock()
        .unwrap()
        .insert(path.to_string(), hex.clone());
    Some(hex)
}

fn pin_fixture(role: &str, path: &str, expect_bytes: u64, expect_sha256: &str) -> String {
    let meta = std::fs::metadata(path)
        .unwrap_or_else(|e| panic!("{role} oracle: cannot stat fixture {path}: {e}"));
    assert_eq!(
        meta.len(),
        expect_bytes,
        "{role} oracle: fixture size mismatch — not the pinned artifact"
    );
    match sha256_of(path) {
        Some(actual) => assert_eq!(
            actual, expect_sha256,
            "{role} oracle: fixture content mismatch — not the pinned artifact"
        ),
        None => {
            eprintln!(
                "{role} oracle: WARN sha256sum unavailable, identity rests on exact size only"
            )
        }
    }
    sha256_of(path).unwrap_or_else(|| "<unhashed>".to_string())
}

// ── numerics helpers ─────────────────────────────────────────────────────────

fn argmax(xs: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn chatml_prompt(tok: &Tokenizer) -> Vec<u32> {
    let im_start = tok.encode("<|im_start|>");
    let im_end = tok.encode("<|im_end|>");
    let nl = tok.encode("\n");
    let user = tok.encode("user");
    let asst = tok.encode("assistant");
    let q = tok.encode(PROMPT);
    let mut t = Vec::new();
    t.extend_from_slice(&im_start);
    t.extend_from_slice(&user);
    t.extend_from_slice(&nl);
    t.extend_from_slice(&q);
    t.extend_from_slice(&im_end);
    t.extend_from_slice(&nl);
    t.extend_from_slice(&im_start);
    t.extend_from_slice(&asst);
    t.extend_from_slice(&nl);
    t
}

/// FullAttention-layer mask: only FA layers hold KV (DeltaNet layers hold
/// recurrent state). Matches the loader's Mask-dim policy for qwen35.
fn kv_mask(config: &Qwen35Config) -> Vec<bool> {
    config
        .layer_types
        .iter()
        .map(|t| *t == qwen35::LayerType::FullAttention)
        .collect()
}

/// DeltaNet state precision for BOTH routes. `HIPFIRE_ORACLE_STATE_FP32=1`
/// runs the comparison on f32 recurrent state — the maintainer's diagnostic
/// for whether a divergence is carried by Q8 state requant at all.
fn oracle_state_quant() -> StateQuant {
    if std::env::var("HIPFIRE_ORACLE_STATE_FP32").as_deref() == Ok("1") {
        StateQuant::FP32
    } else {
        StateQuant::Q8
    }
}

fn state_quant_name(q: StateQuant) -> &'static str {
    if q == StateQuant::Q8 {
        "q8"
    } else if q == StateQuant::FP32 {
        "fp32"
    } else {
        "other"
    }
}

// ── single-GPU route ─────────────────────────────────────────────────────────

struct Single {
    gpu: Gpu,
    config: Qwen35Config,
    weights: Qwen35Weights,
    kv: KvCache,
    dn: DeltaNetState,
    scratch: Qwen35Scratch,
}

fn load_single(path: &str) -> Option<Single> {
    let mut hfq = HfqFile::open(Path::new(path)).expect("open fixture");
    let config = qwen35::config_from_hfq(&hfq).expect("fixture config");
    let Ok(mut gpu) = Gpu::init() else {
        eprintln!("skip: single route Gpu::init failed");
        return None;
    };
    let mut src = HfqSource::new(&mut hfq, &config);
    let weights = qwen35::load_weights(
        &mut src,
        std::slice::from_mut(&mut gpu),
        &Layout::single(config.n_layers),
    )
    .expect("single load_weights");
    let mask = kv_mask(&config);
    let kv = KvCache::new_gpu_fwht3_capped_filtered(
        &mut gpu,
        &mask,
        config.n_kv_heads,
        config.head_dim,
        KV_MAX,
        KV_MAX,
    )
    .expect("single fwht3 kv");
    let dn =
        DeltaNetState::new_with_quant(&mut gpu, &config, oracle_state_quant()).expect("single dn");
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, SCRATCH_WINDOW, KV_MAX)
        .expect("single scratch");
    Some(Single {
        gpu,
        config,
        weights,
        kv,
        dn,
        scratch,
    })
}

impl Single {
    fn step(&mut self, token: u32, pos: usize) -> Vec<f32> {
        qwen35::forward_scratch(
            &mut self.gpu,
            &self.weights,
            &self.config,
            token,
            pos,
            &mut self.kv,
            &mut self.dn,
            &self.scratch,
        )
        .expect("single forward_scratch");
        self.gpu.bind_thread().expect("single bind");
        self.gpu
            .download_f32(&self.scratch.logits)
            .expect("single logits")
    }

    fn free(self) {
        let mut gpu = self.gpu;
        let _ = self.scratch.free_gpu(&mut gpu);
        self.dn.free_gpu(&mut gpu);
        let _ = self.kv.free_gpu(&mut gpu);
        self.weights.free_gpu(&mut gpu);
        gpu.drain_pool();
    }
}

fn log_kv(tag: &str, kv: &KvCache) {
    eprintln!(
        "{tag}: kv tier=fwht3 fwht={} asym3={} kv_dim={} max_seq={} cap={} n_heads={} head_dim={} k_layers={} v_layers={} k0_bytes={} v0_bytes={}",
        kv.quant_fwht,
        kv.quant_asym3,
        kv.kv_dim,
        kv.max_seq,
        kv.physical_cap,
        kv.n_kv_heads,
        kv.head_dim,
        kv.k_gpu.len(),
        kv.v_gpu.len(),
        kv.k_gpu.first().map(|t| t.byte_size()).unwrap_or(0),
        kv.v_gpu.first().map(|t| t.byte_size()).unwrap_or(0),
    );
}

fn check_dn(tag: &str, dn: &DeltaNetState) {
    assert!(
        dn.quant == oracle_state_quant(),
        "{tag}: DeltaNet state must be Q8-pinned"
    );
    assert!(
        !dn.s_matrices.is_empty(),
        "{tag}: DeltaNet S matrices must be present"
    );
    eprintln!(
        "{tag}: dn quant={} s_matrices={} s0_bytes={} conv_states={}",
        state_quant_name(dn.quant),
        dn.s_matrices.len(),
        dn.s_matrices[0].byte_size(),
        dn.conv_states.len(),
    );
}

#[allow(clippy::too_many_arguments)]
fn evidence(
    test: &str,
    fixture: &str,
    sha: &str,
    ranks: &str,
    positions: usize,
    worst: f32,
    bound: Option<f32>,
    relation: &str,
    reason: &str,
) {
    eprintln!(
        "[{test}] EVIDENCE fixture={fixture} sha256={sha} tier={TIER} state={STATE_QUANT} \
         ranks={ranks} positions={positions} worst_abs_diff={worst:.3e} bound={} relation={relation}",
        bound.map(|b| format!("{b:.1e}")).unwrap_or_else(|| "-".to_string()),
    );
    eprintln!("[{test}] EVIDENCE reason: {reason}");
}

// ── 1. PP2 vs single (dense, BOUNDED) ─────────────────────────────────────────

/// `qwen35_pp2_vs_single_oracle` — dense 27B, PP over 2 GPUs vs single GPU.
///
/// Relationship: BOUNDED (tokens `assert_eq`, max abs logit diff <= 2.0).
/// Whole-layer split with no op reordering, but Q8 DeltaNet requant differs
/// per route (single: error-feedback sigma-delta; multi: legacy stochastic —
/// see module docs). Measured 2.1e-3 at pos 0 → 8.1e-1 worst over 27
/// positions, bit-identical across fresh processes, tokens exact throughout.
#[test]
#[ignore]
fn qwen35_pp2_vs_single_oracle() {
    const TEST: &str = "qwen35-pp2-oracle";
    if !have_mesh() {
        eprintln!("skip: {TEST} needs HIPFIRE_HAVE_2_GPU=1 and 2+ GPUs");
        return;
    }
    let path = dense_fixture();
    let sha = pin_fixture(TEST, &path, DENSE_SIZE, DENSE_SHA256);

    // Mesh first: `init_uniform` preflights VRAM uniformity across devices, so
    // the mesh must claim its devices before the ~15 GiB single route lands on
    // dev 0. (Load order is numerically irrelevant — deterministic kernels —
    // this only satisfies the capacity gate.)
    let tokenizer = {
        let hfq = HfqFile::open(Path::new(&path)).expect("reopen for tokenizer");
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer")
    };
    let prompt = chatml_prompt(&tokenizer);
    let total = prompt.len() + N_DECODE;
    assert!(total <= KV_MAX, "position budget vs KV max_seq");
    eprintln!(
        "{TEST}: prompt tokens = {} (prompt {PROMPT:?})",
        prompt.len()
    );

    // Mesh route: mirror `pp_parity.rs::run_pp2`, tier lifted asym3 → fwht3.
    let mut hfq = HfqFile::open(Path::new(&path)).expect("open fixture (pp)");
    let config = qwen35::config_from_hfq(&hfq).expect("pp config");
    let mut gpus = match Gpus::init_uniform(2, config.n_layers) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skip: {TEST} init_uniform(2) failed: {e:?}");
            return;
        }
    };
    let layout = Layout::from_gpus(&gpus, config.n_layers);
    let mut hfq_src = HfqSource::new(&mut hfq, &config);
    let weights_pp = qwen35::load_weights(&mut hfq_src, &mut gpus.devices, &layout)
        .expect("pp load_weights_multi");
    let scratch_pp =
        Qwen35ScratchSet::new_with_kv_max_multi(&mut gpus, &config, SCRATCH_WINDOW, KV_MAX)
            .expect("pp scratch_set");
    let mut kv_pp = KvCache::new_gpu_fwht3_capped_multi(
        &mut gpus,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        KV_MAX,
        KV_MAX,
    )
    .expect("pp fwht3 kv");
    let (mut dn_pp, la_to_device) =
        DeltaNetState::new_with_quant_multi(&mut gpus, &config, oracle_state_quant())
            .expect("pp dn");
    let _ = gpus.enable_peer_all().expect("pp enable_peer_all");
    let dev_last = gpus.output_device;
    // Single route second (see mesh-first note above).
    let Some(mut single) = load_single(&path) else {
        eprintln!("skip: {TEST} single-route GPU init failed");
        scratch_pp.free_gpu_multi(&mut gpus);
        dn_pp.free_gpu_multi(&mut gpus, &la_to_device);
        kv_pp.free_gpu_multi(&mut gpus);
        weights_pp.free_gpu_multi(&mut gpus);
        return;
    };
    assert_eq!(
        config.n_layers, single.config.n_layers,
        "route config n_layers"
    );
    assert_eq!(config.dim, single.config.dim, "route config dim");

    // Route identity + alias identity.
    eprintln!(
        "{TEST}: route identity — single=forward_scratch/Layout::single, mesh=forward_scratch_multi/pp=2"
    );
    assert_eq!(
        weights_pp.lm_head_aliases_embd, single.weights.lm_head_aliases_embd,
        "alias identity must match between routes"
    );

    // KV geometry parity.
    log_kv(TEST, &single.kv);
    assert!(
        single.kv.quant_fwht,
        "single route must carry the fwht3 tier"
    );
    assert_eq!(kv_pp.n_kv_heads, single.kv.n_kv_heads);
    assert_eq!(kv_pp.head_dim, single.kv.head_dim);
    assert_eq!(kv_pp.max_seq, single.kv.max_seq);
    assert_eq!(kv_pp.physical_cap, single.kv.physical_cap);
    assert!(kv_pp.quant_fwht, "mesh route must carry the fwht3 tier");
    assert_eq!(
        kv_pp.k_gpu.len(),
        single.kv.k_gpu.len(),
        "K layer count parity"
    );
    assert_eq!(
        kv_pp.v_gpu.len(),
        single.kv.v_gpu.len(),
        "V layer count parity"
    );
    // Extent parity at the first KV layer: the single route's filtered cache
    // holds 4-byte dummies on non-KV (DeltaNet) layers while the unfiltered
    // multi cache allocates every layer (`pp_parity.rs` precedent) — so layer
    // 0 (non-KV) is not comparable; the first FullAttention layer is.
    let kv_first = kv_mask(&config)
        .iter()
        .position(|&keep| keep)
        .expect("at least one KV layer");
    assert_eq!(
        kv_pp.k_gpu[kv_first].byte_size(),
        single.kv.k_gpu[kv_first].byte_size(),
        "first-KV-layer K byte extent parity"
    );
    assert_eq!(
        kv_pp.v_gpu[kv_first].byte_size(),
        single.kv.v_gpu[kv_first].byte_size(),
        "first-KV-layer V byte extent parity"
    );
    check_dn(TEST, &single.dn);
    assert!(
        dn_pp.quant == oracle_state_quant(),
        "mesh DeltaNet state must be Q8-pinned"
    );
    eprintln!(
        "{TEST}: mesh dn s_matrices={} (single {})",
        dn_pp.s_matrices.len(),
        single.dn.s_matrices.len()
    );

    // Greedy lockstep at every committed position; bounded logits, exact tokens.
    let mut next_token: u32 = 0;
    let mut worst: f32 = 0.0;
    for pos in 0..total {
        let token = if pos < prompt.len() {
            prompt[pos]
        } else {
            next_token
        };
        let single_logits = single.step(token, pos);
        qwen35::forward_scratch_multi(
            &mut gpus,
            &weights_pp,
            &config,
            token,
            pos,
            &mut kv_pp,
            &mut dn_pp,
            &scratch_pp,
        )
        .expect("pp forward_scratch_multi");
        gpus.devices[dev_last].bind_thread().expect("pp bind");
        let pp_logits = gpus.devices[dev_last]
            .download_f32(&scratch_pp.per_device[dev_last].logits)
            .expect("pp logits");
        assert_eq!(
            single_logits.len(),
            pp_logits.len(),
            "logit width at pos {pos}"
        );
        let diff = max_abs_diff(&single_logits, &pp_logits);
        worst = worst.max(diff);
        assert!(
            diff <= BOUND_PP_ABS,
            "PP-vs-single logit bound breach at committed position {pos}: {diff:.3e} > {BOUND_PP_ABS:.1e}"
        );
        let choice = argmax(&single_logits);
        assert_eq!(argmax(&pp_logits), choice, "token-id mismatch at pos {pos}");
        next_token = choice;
        eprintln!(
            "{TEST}: pos {pos:>2} token {token:>6} max-logit-diff {diff:.3e} (choice {choice})"
        );
    }

    evidence(
        TEST,
        &path,
        &sha,
        "pp=2",
        total,
        worst,
        Some(BOUND_PP_ABS),
        "bounded",
        "whole-layer split, no op reordering; Q8 DeltaNet requant differs per route (single EF sigma-delta vs multi legacy stochastic): measured 2.1e-3 → 8.1e-1 over 27 positions, deterministic, tokens exact",
    );
    eprintln!("{TEST}: PASS — {total} committed positions, worst logit diff {worst:.3e}");

    scratch_pp.free_gpu_multi(&mut gpus);
    dn_pp.free_gpu_multi(&mut gpus, &la_to_device);
    kv_pp.free_gpu_multi(&mut gpus);
    weights_pp.free_gpu_multi(&mut gpus);
    single.free();
}

// ── 2. TP2 vs single (dense, BOUNDED) ────────────────────────────────────────

/// Relationship: BOUNDED (tokens `assert_eq`, max abs logit diff <= 1.0).
/// Dense TP shards heads/projections per rank
/// (`load_weights_dense_tp_rank` over `dense_tp_rank_layouts`) and sums the
/// row-parallel partials through the deterministic rooted peer reduce
/// (`dense_tp_all_reduce_sum_f32` → `all_reduce_sum_f32_peer_rooted`,
/// `forward.rs:3664`), a different f32 summation order than the single route's
/// direct accumulation — so `assert_eq` on logits is unwarranted. On top of the
/// order term, the per-step Q8 recurrent-state writes diverge by quantization
/// LSBs (reduce-rounding-perturbed inputs flip quantize decisions in later DN
/// layers) and the barely-contractive DN recurrence (α≈0.99) random-walks the
/// difference up over positions: measured pos0 2.141e-3 — bit-identical across
/// PP/TP/EF-on/EF-off runs, i.e. a deterministic multi-standup first-write
/// effect, NOT f32 order — growing to worst 4.6e-1 over 27 positions EF-on
/// (3.0 with `HIPFIRE_DN_STATE_EF=0`, so EF sigma-delta damps ~6x). Bound 1.0
/// is 2x measured worst. Precedent: `qwen_dense_tp2_parity.rs` claims
/// argmax-exact + relative error < 3e-3, never bit-exact. No new `pub` seam
/// needed: every symbol below is already exported from
/// `hipfire_arch_qwen35::qwen35`.
#[test]
#[ignore]
fn qwen35_tp2_vs_single_oracle() {
    const TEST: &str = "qwen35-tp2-oracle";
    if !have_mesh() {
        eprintln!("skip: {TEST} needs HIPFIRE_HAVE_2_GPU=1 and 2+ GPUs");
        return;
    }
    let path = dense_fixture();
    let sha = pin_fixture(TEST, &path, DENSE_SIZE, DENSE_SHA256);

    // Mesh first: `init_tp` preflights VRAM uniformity like `init_uniform`,
    // so the mesh must claim its devices before the ~15 GiB single route
    // lands on dev 0. (Load order is numerically irrelevant — deterministic
    // kernels — this only satisfies the capacity gate.)
    let tokenizer = {
        let hfq = HfqFile::open(Path::new(&path)).expect("reopen for tokenizer");
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer")
    };
    let prompt = chatml_prompt(&tokenizer);
    let total = prompt.len() + N_DECODE;
    assert!(total <= KV_MAX, "position budget vs KV max_seq");
    eprintln!(
        "{TEST}: prompt tokens = {} (prompt {PROMPT:?})",
        prompt.len()
    );

    // Mesh route: dense-TP standup per the loader (`load_model_tp_qwen35_dense`)
    // and `qwen_dense_tp2_parity.rs`, tier pinned fwht3.
    const TP: usize = 2;
    let hfq = HfqFile::open(Path::new(&path)).expect("open fixture (tp)");
    let global = qwen35::config_from_hfq(&hfq).expect("tp config");
    // (Route-config cross-checks run after the single route loads below.)
    let shard = ShardConfig::new(TP, false, 0, ExpertAssign::Stride).expect("tp ShardConfig");
    let layouts = qwen35::dense_tp_rank_layouts(&global, &shard).expect("tp layouts");
    qwen35::preflight_weights_dense_tp(&hfq, &global, &shard).expect("tp preflight");
    drop(hfq);
    let configs: Vec<Qwen35Config> = layouts
        .iter()
        .map(|l| qwen35::local_dense_tp_config(&global, l))
        .collect();
    let mut gpus = match Gpus::init_tp(TP, global.n_layers) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skip: {TEST} init_tp(2) failed: {e:?}");
            return;
        }
    };
    let mut weights = Vec::with_capacity(TP);
    let mut kvs = Vec::with_capacity(TP);
    let mut dns = Vec::with_capacity(TP);
    let mut scratches = Vec::with_capacity(TP);
    for rank in 0..TP {
        let mut rank_hfq =
            HfqFile::open(Path::new(&path)).unwrap_or_else(|e| panic!("reopen rank {rank}: {e}"));
        gpus.devices[rank].bind_thread().expect("tp bind");
        weights.push(
            qwen35::load_weights_dense_tp_rank(
                &mut rank_hfq,
                &global,
                &mut gpus.devices[rank],
                &layouts[rank],
            )
            .unwrap_or_else(|e| panic!("tp load rank {rank}: {e:?}")),
        );
        let mask = kv_mask(&global);
        kvs.push(
            KvCache::new_gpu_fwht3_capped_filtered(
                &mut gpus.devices[rank],
                &mask,
                configs[rank].n_kv_heads,
                configs[rank].head_dim,
                KV_MAX,
                KV_MAX,
            )
            .unwrap_or_else(|e| panic!("tp kv rank {rank}: {e:?}")),
        );
        dns.push(
            DeltaNetState::new_with_quant(
                &mut gpus.devices[rank],
                &configs[rank],
                oracle_state_quant(),
            )
            .unwrap_or_else(|e| panic!("tp dn rank {rank}: {e:?}")),
        );
        scratches.push(
            Qwen35Scratch::new_with_kv_max(
                &mut gpus.devices[rank],
                &configs[rank],
                SCRATCH_WINDOW,
                KV_MAX,
            )
            .unwrap_or_else(|e| panic!("tp scratch rank {rank}: {e:?}")),
        );
    }
    let peer = gpus.can_access_peer_all().expect("tp can_access_peer_all");
    if peer {
        assert!(
            gpus.enable_peer_all().expect("tp enable_peer_all"),
            "complete P2P topology must enable peers"
        );
    }
    ensure_rank_streams(&mut gpus).expect("tp ensure_rank_streams");
    eprintln!("{TEST}: tp={TP} peer_access={peer} (rooted reduce iff peer)");
    // Single route second (see mesh-first note above).
    let Some(mut single) = load_single(&path) else {
        eprintln!("skip: {TEST} single-route GPU init failed");
        let _ = gpus.free_peer_reduce_scratch();
        for (rank, kv) in kvs.into_iter().enumerate() {
            let _ = kv.free_gpu(&mut gpus.devices[rank]);
        }
        for (rank, dn) in dns.into_iter().enumerate() {
            dn.free_gpu(&mut gpus.devices[rank]);
        }
        for (rank, s) in scratches.into_iter().enumerate() {
            let _ = s.free_gpu(&mut gpus.devices[rank]);
        }
        for (rank, w) in weights.into_iter().enumerate() {
            w.free_gpu(&mut gpus.devices[rank]);
        }
        return;
    };
    assert_eq!(
        global.n_layers, single.config.n_layers,
        "route config n_layers"
    );
    assert_eq!(global.dim, single.config.dim, "route config dim");

    eprintln!(
        "{TEST}: route identity — single=forward_scratch/Layout::single, mesh=forward_scratch_dense_tp/tp=2"
    );
    assert_eq!(
        weights[0].lm_head_aliases_embd, single.weights.lm_head_aliases_embd,
        "alias identity must match between routes"
    );

    // KV geometry: per-rank head sharding is expected; tier, caps and DN
    // quant must match. Whatever is reachable per rank is asserted per rank.
    log_kv(TEST, &single.kv);
    assert!(
        single.kv.quant_fwht,
        "single route must carry the fwht3 tier"
    );
    for rank in 0..TP {
        assert_eq!(kvs[rank].n_kv_heads, configs[rank].n_kv_heads);
        assert_eq!(kvs[rank].head_dim, single.kv.head_dim);
        assert_eq!(kvs[rank].max_seq, single.kv.max_seq);
        assert!(
            kvs[rank].quant_fwht,
            "rank {rank} must carry the fwht3 tier"
        );
        assert!(!kvs[rank].k_gpu.is_empty() && !kvs[rank].v_gpu.is_empty());
        assert!(
            dns[rank].quant == oracle_state_quant(),
            "rank {rank} DN must be Q8"
        );
        assert_eq!(dns[rank].s_matrices.len(), single.dn.s_matrices.len());
        eprintln!(
            "{TEST}: rank {rank} kv_heads={} k0_bytes={} dn_s0_bytes={}",
            kvs[rank].n_kv_heads,
            kvs[rank].k_gpu[0].byte_size(),
            dns[rank].s_matrices[0].byte_size(),
        );
    }
    check_dn(TEST, &single.dn);

    // Greedy lockstep; bounded logits, exact tokens.
    let mut next_token: u32 = 0;
    let mut worst: f32 = 0.0;
    for pos in 0..total {
        let token = if pos < prompt.len() {
            prompt[pos]
        } else {
            next_token
        };
        // Anti-vacuity knob (default OFF): `HIPFIRE_ORACLE_DIVERGE=1` forces the
        // mesh side onto a wrong token while the single route stays on the true
        // one, driving a real numerical divergence through the forward path so a
        // red run proves the bound/token asserts can fire. Committed runs leave
        // it unset.
        let mesh_token = if std::env::var("HIPFIRE_ORACLE_DIVERGE").as_deref() == Ok("1") {
            token.wrapping_add(1)
        } else {
            token
        };
        let single_logits = single.step(token, pos);
        qwen35::forward_scratch_dense_tp(
            &mut gpus, &shard, &weights, &configs, mesh_token, pos, &mut kvs, &mut dns, &scratches,
        )
        .expect("tp forward_scratch_dense_tp");
        gpus.devices[0].bind_thread().expect("tp bind0");
        let tp_logits = gpus.devices[0]
            .download_f32(&scratches[0].logits)
            .expect("tp logits");
        assert_eq!(
            single_logits.len(),
            tp_logits.len(),
            "logit width at pos {pos}"
        );
        let diff = max_abs_diff(&single_logits, &tp_logits);
        worst = worst.max(diff);
        assert!(
            diff <= BOUND_TP_ABS,
            "TP-vs-single logit bound breach at committed position {pos}: {diff:.3e} > {BOUND_TP_ABS:.1e}"
        );
        let choice = argmax(&single_logits);
        assert_eq!(argmax(&tp_logits), choice, "token-id mismatch at pos {pos}");
        next_token = choice;
        eprintln!(
            "{TEST}: pos {pos:>2} token {token:>6} max-logit-diff {diff:.3e} (choice {choice})"
        );
    }

    evidence(
        TEST,
        &path,
        &sha,
        "tp=2",
        total,
        worst,
        Some(BOUND_TP_ABS),
        "bounded",
        "per-rank head/projection shards + rooted peer reduce (forward.rs:3664 dense_tp_all_reduce_sum_f32 -> all_reduce_sum_f32_peer_rooted): different f32 summation order than single-route accumulation (qwen_dense_tp2_parity.rs precedent), plus per-step Q8 recurrent-state LSB divergence random-walking up through the barely-contractive DN recurrence (pos0 2.141e-3 identical across PP/TP/EF-on/EF-off; EF-on worst 4.6e-1 vs EF-off 3.0)",
    );
    eprintln!("{TEST}: PASS — {total} committed positions, worst logit diff {worst:.3e}");

    let _ = gpus.free_peer_reduce_scratch();
    for (rank, kv) in kvs.into_iter().enumerate() {
        let _ = kv.free_gpu(&mut gpus.devices[rank]);
    }
    for (rank, dn) in dns.into_iter().enumerate() {
        dn.free_gpu(&mut gpus.devices[rank]);
    }
    for (rank, s) in scratches.into_iter().enumerate() {
        let _ = s.free_gpu(&mut gpus.devices[rank]);
    }
    for (rank, w) in weights.into_iter().enumerate() {
        w.free_gpu(&mut gpus.devices[rank]);
    }
    single.free();
}

// ── state snapshot / restore (EP oracles) ──────────────────────────────────────
//
// MoE top-k is a discrete decision: 1e-5-scale grouping noise in the hidden
// state can flip an expert, injecting expert-scale perturbations that persist
// in KV/recurrent state (measured on the production fwht3+Q8 config:
// deterministic 6.5e-2 cliff at pos 2, token flip at pos 5, on two prompts).
// position the oracle snapshots the reference route's KV + DeltaNet + full
// scratch bytes BEFORE its step (scratch temporaries are reused across layers
// *and* steps, so their history must travel too), steps the reference,
// restores every mesh rank to those exact bytes, then steps the mesh forced
// with the same token. The discrete router then sees bit-identical inputs on
// both sides and only grouping-rounding (~1e-6) remains. `pos` is passed
// explicitly per step. All bytes move through already-`pub` APIs
// (`k_gpu`/`s_matrices`/scratch fields + `hip.memcpy_*`), the same pattern
// the template oracle uses for its K/V payload asserts.
struct StateSnap {
    kv_k: Vec<Vec<u8>>,
    kv_v: Vec<Vec<u8>>,
    dn_s: Vec<Vec<u8>>,
    dn_sc: Vec<Vec<u8>>,
    dn_c: Vec<Vec<u8>>,
    dn_ef: Vec<Vec<u8>>,
    scratch: ScratchSnap,
}

fn download_tensors(gpu: &mut Gpu, tensors: &[GpuTensor]) -> Vec<Vec<u8>> {
    // NOTE: `GpuTensor::byte_size` reports the nominal shape×dtype extent,
    // which over-reports for sub-byte state (Q8 DeltaNet S carries an int8
    // backing under a nominal-F32 shape). `DeviceBuffer::size` is the true
    // allocation length and the exact state extent.
    tensors
        .iter()
        .map(|t| {
            let mut bytes = vec![0u8; t.buf.size()];
            gpu.hip
                .memcpy_dtoh(&mut bytes, &t.buf)
                .expect("snapshot download");
            bytes
        })
        .collect()
}

fn upload_tensors(gpu: &mut Gpu, tensors: &[GpuTensor], blobs: &[Vec<u8>]) {
    assert_eq!(tensors.len(), blobs.len(), "snapshot tensor count parity");
    for (t, b) in tensors.iter().zip(blobs.iter()) {
        assert_eq!(t.buf.size(), b.len(), "snapshot tensor byte parity");
        gpu.hip
            .memcpy_htod(&t.buf, b.as_slice())
            .expect("snapshot upload");
    }
}

fn snap_state(
    gpu: &mut Gpu,
    kv: &KvCache,
    dn: &DeltaNetState,
    scratch: &Qwen35Scratch,
) -> StateSnap {
    gpu.bind_thread().expect("snapshot bind");
    StateSnap {
        kv_k: download_tensors(gpu, &kv.k_gpu),
        kv_v: download_tensors(gpu, &kv.v_gpu),
        dn_s: download_tensors(gpu, &dn.s_matrices),
        dn_sc: download_tensors(gpu, &dn.s_scales),
        dn_c: download_tensors(gpu, &dn.conv_states),
        dn_ef: download_tensors(gpu, &dn.s_ef_residual),
        scratch: snap_scratch(gpu, scratch),
    }
}

fn restore_state(
    gpu: &mut Gpu,
    kv: &KvCache,
    dn: &DeltaNetState,
    scratch: &mut Qwen35Scratch,
    snap: &StateSnap,
) {
    gpu.bind_thread().expect("restore bind");
    upload_tensors(gpu, &kv.k_gpu, &snap.kv_k);
    upload_tensors(gpu, &kv.v_gpu, &snap.kv_v);
    upload_tensors(gpu, &dn.s_matrices, &snap.dn_s);
    upload_tensors(gpu, &dn.s_scales, &snap.dn_sc);
    upload_tensors(gpu, &dn.conv_states, &snap.dn_c);
    upload_tensors(gpu, &dn.s_ef_residual, &snap.dn_ef);
    restore_scratch(gpu, scratch, &snap.scratch);
}
/// Scratch snapshot: every `Qwen35Scratch` device tensor in a fixed order.
/// Scratch temporaries are reused across layers *and* across steps; whatever
/// history they carry must travel with the KV/DN snapshot or the mesh replays
/// from a different micro-state than the reference.
struct ScratchSnap {
    fixed: Vec<Vec<u8>>,
    opt: Vec<Option<Vec<u8>>>,
    pos_buf: Vec<u8>,
    pos_buf3: Vec<u8>,
}

fn scratch_fixed(s: &Qwen35Scratch) -> [&GpuTensor; 30] {
    [
        &s.x,
        &s.tmp,
        &s.dn_qkv,
        &s.dn_z,
        &s.dn_alpha,
        &s.dn_beta,
        &s.dn_conv_out,
        &s.dn_q,
        &s.dn_k,
        &s.dn_v,
        &s.dn_q_raw,
        &s.dn_k_raw,
        &s.dn_attn_out,
        &s.dn_normed,
        &s.fa_q_full,
        &s.fa_q,
        &s.fa_gate,
        &s.fa_k,
        &s.fa_v,
        &s.fa_attn_out,
        &s.o,
        &s.gate_ffn,
        &s.up,
        &s.ffn_hidden,
        &s.ffn_out,
        &s.logits,
        &s.sample_buf,
        &s.repeat_buf,
        &s.x_rot,
        &s.flash_partials,
    ]
}

fn scratch_fixed_mut(s: &mut Qwen35Scratch) -> [&mut GpuTensor; 30] {
    [
        &mut s.x,
        &mut s.tmp,
        &mut s.dn_qkv,
        &mut s.dn_z,
        &mut s.dn_alpha,
        &mut s.dn_beta,
        &mut s.dn_conv_out,
        &mut s.dn_q,
        &mut s.dn_k,
        &mut s.dn_v,
        &mut s.dn_q_raw,
        &mut s.dn_k_raw,
        &mut s.dn_attn_out,
        &mut s.dn_normed,
        &mut s.fa_q_full,
        &mut s.fa_q,
        &mut s.fa_gate,
        &mut s.fa_k,
        &mut s.fa_v,
        &mut s.fa_attn_out,
        &mut s.o,
        &mut s.gate_ffn,
        &mut s.up,
        &mut s.ffn_hidden,
        &mut s.ffn_out,
        &mut s.logits,
        &mut s.sample_buf,
        &mut s.repeat_buf,
        &mut s.x_rot,
        &mut s.flash_partials,
    ]
}

fn scratch_opt(s: &Qwen35Scratch) -> [Option<&GpuTensor>; 14] {
    [
        s.moe_router_logits.as_ref(),
        s.moe_scalar_buf.as_ref(),
        s.moe_x_rot.as_ref(),
        s.moe_gate_up_buf.as_ref(),
        s.moe_gate_buf.as_ref(),
        s.moe_up_buf.as_ref(),
        s.moe_ffn_hidden.as_ref(),
        s.moe_ffn_out.as_ref(),
        s.moe_gate_batch.as_ref(),
        s.moe_up_batch.as_ref(),
        s.moe_rot_batch.as_ref(),
        s.moe_topk_indices.as_ref(),
        s.moe_topk_weights.as_ref(),
        s.moe_down_expanded.as_ref(),
    ]
}

#[allow(clippy::too_many_arguments)]
fn scratch_opt_mut(s: &mut Qwen35Scratch) -> [Option<&mut GpuTensor>; 14] {
    [
        s.moe_router_logits.as_mut(),
        s.moe_scalar_buf.as_mut(),
        s.moe_x_rot.as_mut(),
        s.moe_gate_up_buf.as_mut(),
        s.moe_gate_buf.as_mut(),
        s.moe_up_buf.as_mut(),
        s.moe_ffn_hidden.as_mut(),
        s.moe_ffn_out.as_mut(),
        s.moe_gate_batch.as_mut(),
        s.moe_up_batch.as_mut(),
        s.moe_rot_batch.as_mut(),
        s.moe_topk_indices.as_mut(),
        s.moe_topk_weights.as_mut(),
        s.moe_down_expanded.as_mut(),
    ]
}

fn snap_scratch(gpu: &mut Gpu, s: &Qwen35Scratch) -> ScratchSnap {
    gpu.bind_thread().expect("scratch snapshot bind");
    let mut pos_buf = vec![0u8; s.pos_buf.size()];
    gpu.hip
        .memcpy_dtoh(&mut pos_buf, &s.pos_buf)
        .expect("snapshot pos_buf");
    let mut pos_buf3 = vec![0u8; s.pos_buf3.size()];
    gpu.hip
        .memcpy_dtoh(&mut pos_buf3, &s.pos_buf3)
        .expect("snapshot pos_buf3");
    ScratchSnap {
        fixed: scratch_fixed(s)
            .iter()
            .map(|t| {
                let mut bytes = vec![0u8; t.buf.size()];
                gpu.hip
                    .memcpy_dtoh(&mut bytes, &t.buf)
                    .expect("snapshot scratch");
                bytes
            })
            .collect(),
        opt: scratch_opt(s)
            .iter()
            .map(|t| {
                t.map(|t| {
                    let mut bytes = vec![0u8; t.buf.size()];
                    gpu.hip
                        .memcpy_dtoh(&mut bytes, &t.buf)
                        .expect("snapshot scratch opt");
                    bytes
                })
            })
            .collect(),
        pos_buf,
        pos_buf3,
    }
}

fn restore_scratch(gpu: &mut Gpu, s: &mut Qwen35Scratch, snap: &ScratchSnap) {
    gpu.bind_thread().expect("scratch restore bind");
    gpu.hip
        .memcpy_htod(&s.pos_buf, snap.pos_buf.as_slice())
        .expect("restore pos_buf");
    gpu.hip
        .memcpy_htod(&s.pos_buf3, snap.pos_buf3.as_slice())
        .expect("restore pos_buf3");
    let fixed = scratch_fixed_mut(s);
    assert_eq!(fixed.len(), snap.fixed.len(), "scratch fixed count");
    for (t, b) in fixed.into_iter().zip(snap.fixed.iter()) {
        assert_eq!(t.buf.size(), b.len(), "scratch fixed bytes");
        gpu.hip
            .memcpy_htod(&t.buf, b.as_slice())
            .expect("restore scratch");
    }
    let opt = scratch_opt_mut(s);
    assert_eq!(opt.len(), snap.opt.len(), "scratch opt count");
    for (t, b) in opt.into_iter().zip(snap.opt.iter()) {
        match (t, b) {
            (Some(t), Some(b)) => {
                assert_eq!(t.buf.size(), b.len(), "scratch opt bytes");
                gpu.hip
                    .memcpy_htod(&t.buf, b.as_slice())
                    .expect("restore scratch opt");
            }
            (None, None) => {}
            _ => panic!("scratch opt presence mismatch between routes"),
        }
    }
}

/// Verify a restore round-trips byte-identically (restore then re-download
/// must equal the snapshot). Catches silent memcpy/stream-ordering faults in
/// the restore path itself, distinct from route-math divergence.
fn assert_restore_roundtrip(
    gpu: &mut Gpu,
    kv: &KvCache,
    dn: &DeltaNetState,
    scratch: &mut Qwen35Scratch,
    snap: &StateSnap,
) {
    restore_state(gpu, kv, dn, scratch, snap);
    let again = snap_state(gpu, kv, dn, scratch);
    assert_eq!(again.kv_k, snap.kv_k, "roundtrip kv_k");
    assert_eq!(again.kv_v, snap.kv_v, "roundtrip kv_v");
    assert_eq!(again.dn_s, snap.dn_s, "roundtrip dn_s");
    assert_eq!(again.dn_sc, snap.dn_sc, "roundtrip dn_sc");
    assert_eq!(again.dn_c, snap.dn_c, "roundtrip dn_c");
    assert_eq!(again.dn_ef, snap.dn_ef, "roundtrip dn_ef");
    assert_eq!(
        again.scratch.fixed, snap.scratch.fixed,
        "roundtrip scratch fixed"
    );
    assert_eq!(again.scratch.opt, snap.scratch.opt, "roundtrip scratch opt");
    assert_eq!(
        again.scratch.pos_buf, snap.scratch.pos_buf,
        "roundtrip pos_buf"
    );
    assert_eq!(
        again.scratch.pos_buf3, snap.scratch.pos_buf3,
        "roundtrip pos_buf3"
    );
}

// ── EP mesh standup (shared by tests 3–4) ────────────────────────────────────

struct EpMesh {
    gpus: Gpus,
    config: Qwen35Config,
    weights: Vec<Qwen35Weights>,
    kvs: Vec<KvCache>,
    dns: Vec<DeltaNetState>,
    scratches: Vec<Qwen35Scratch>,
    partials: Vec<GpuTensor>,
}

fn load_ep_mesh(test: &str, path: &str, tp: usize) -> Option<EpMesh> {
    let hfq0 = HfqFile::open(Path::new(path)).expect("open fixture (ep config)");
    let config = qwen35::config_from_hfq(&hfq0).expect("ep config");
    drop(hfq0);
    assert!(
        config.num_experts > 0,
        "{test}: EP oracle expects a MoE (A3B) fixture"
    );
    let mut gpus = match Gpus::init_tp(tp, config.n_layers) {
        Ok(g) if g.devices.len() == tp => g,
        Ok(g) => {
            eprintln!(
                "skip: {test} init_tp({tp}) gave {} devices",
                g.devices.len()
            );
            return None;
        }
        Err(e) => {
            eprintln!("skip: {test} init_tp({tp}) failed: {e:?}");
            return None;
        }
    };
    // Replicated load with per-rank expert sharding via the TLS context, then
    // stride assignment (rank r owns experts e%tp==r) — the
    // `ep_decode_parity.rs` pattern. Non-owned experts read load-time
    // zero-dummies and contribute 0 to the routed partial.
    let shard = ShardConfig::new(tp, true, config.num_experts, ExpertAssign::Stride)
        .expect("ep ShardConfig");
    let mask = kv_mask(&config);
    let mut weights = Vec::with_capacity(tp);
    let mut kvs = Vec::with_capacity(tp);
    let mut dns = Vec::with_capacity(tp);
    let mut scratches = Vec::with_capacity(tp);
    let mut partials = Vec::with_capacity(tp);
    for r in 0..tp {
        gpus.devices[r].bind_thread().expect("ep bind");
        let mut hfq =
            HfqFile::open(Path::new(path)).unwrap_or_else(|e| panic!("reopen rank {r}: {e}"));
        qwen35::set_ep_expert_shard(Some((shard.clone(), r)));
        let w = {
            let mut src = HfqSource::new(&mut hfq, &config);
            let layout = Layout::single(config.n_layers);
            qwen35::load_weights(
                &mut src,
                std::slice::from_mut(&mut gpus.devices[r]),
                &layout,
            )
        };
        qwen35::set_ep_expert_shard(None);
        weights.push(w.unwrap_or_else(|e| panic!("ep load rank {r}: {e:?}")));
        kvs.push(
            KvCache::new_gpu_fwht3_filtered(
                &mut gpus.devices[r],
                &mask,
                config.n_kv_heads,
                config.head_dim,
                KV_MAX,
            )
            .unwrap_or_else(|e| panic!("ep kv rank {r}: {e:?}")),
        );
        dns.push(
            DeltaNetState::new_with_quant(&mut gpus.devices[r], &config, oracle_state_quant())
                .unwrap_or_else(|e| panic!("ep dn rank {r}: {e:?}")),
        );
        scratches.push(
            Qwen35Scratch::new_with_kv_max(&mut gpus.devices[r], &config, SCRATCH_WINDOW, KV_MAX)
                .unwrap_or_else(|e| panic!("ep scratch rank {r}: {e:?}")),
        );
        partials.push(
            gpus.devices[r]
                .zeros(&[config.dim], DType::F32)
                .unwrap_or_else(|e| panic!("ep partial rank {r}: {e:?}")),
        );
    }
    if tp > 1 {
        let _ = gpus.enable_peer_all().expect("ep enable_peer_all");
    }
    ensure_rank_streams(&mut gpus).expect("ep ensure_rank_streams");
    Some(EpMesh {
        gpus,
        config,
        weights,
        kvs,
        dns,
        scratches,
        partials,
    })
}

impl EpMesh {
    /// One decode position; returns rank-0 logits.
    fn step(&mut self, token: u32, pos: usize) -> Vec<f32> {
        qwen35::forward_ep(
            &mut self.gpus,
            &self.weights,
            &self.config,
            token,
            pos,
            &mut self.kvs,
            &self.dns,
            &self.scratches,
            &self.partials,
        )
        .expect("ep forward_ep");
        self.gpus.devices[0].bind_thread().expect("ep bind0");
        self.gpus.devices[0]
            .download_f32(&self.scratches[0].logits)
            .expect("ep logits")
    }

    fn free(self) {
        let mut gpus = self.gpus;
        for t in self.partials {
            let _ = gpus.devices[0].free_tensor(t);
        }
        for (r, kv) in self.kvs.into_iter().enumerate() {
            let _ = kv.free_gpu(&mut gpus.devices[r]);
        }
        for (r, dn) in self.dns.into_iter().enumerate() {
            dn.free_gpu(&mut gpus.devices[r]);
        }
        for (r, s) in self.scratches.into_iter().enumerate() {
            let _ = s.free_gpu(&mut gpus.devices[r]);
        }
        for (r, w) in self.weights.into_iter().enumerate() {
            w.free_gpu(&mut gpus.devices[r]);
        }
        for dev in gpus.devices.iter_mut() {
            dev.drain_pool();
        }
    }
}

// ── 3. EP2 vs single (MoE, MEASURING-BUT-RED) ───────────────────────────────────

/// `qwen35_ep2_vs_single_oracle` — MoE A3B (256 experts, top_k=8),
/// expert-parallel EP=2 vs single.
///
/// Relationship: MEASURING-BUT-RED (tokens `assert_eq`, max abs logit diff
/// <= 1e-3, asserted and FAILING honestly — do not widen). Router/top-k run
/// replicated on every rank (`run_moe_ep` contract), but each rank accumulates
/// only its owned experts into a zeroed partial and the partials are summed by
/// RCCL `ncclAllReduce` (the `all_reduce_sum_f32_decode` default path).
/// Comparison is snapshot lockstep: the reference single route's KV + DeltaNet
/// + FULL scratch (30 fixed + 14 MoE-opt + pos_bufs, true `DeviceBuffer::size`)
/// are snapshotted pre-step and restored into both ranks, so the discrete
/// router sees identical inputs and only grouping rounding (~1e-5) remains.
/// MEASURED: pos 0–16 hold 3e-6..2e-5, then a deterministic 3.599e-2 cliff at
/// pos 17 — bit-identical to the earlier KV/DN-only snapshot run, so the full-
/// scratch extension changed NOTHING and the carrier lives OUTSIDE the
/// snapshot set. Out-of-snapshot state (maintainer suspect list): per-rank
/// `EpMesh.partials` routed accumulators, RCCL/peer-reduce staging and rank
/// streams, per-rank expert shard/zero-dummy weights (a routing-dependent
/// dropped/duplicated expert fits: deterministic, snapshot-immune, trips when
/// first selected). Token-exactness past pos 16 is UNPROVEN (the bound fires
/// first). Also observed: solo EP=2 greedy (the EP4 reference run) flips off
/// the single trajectory at pos 5 (78937 vs 325) — unpinned recurrence
/// amplifies the ~1e-5 per-step route math within 5 steps.
#[test]
#[ignore]
fn qwen35_ep2_vs_single_oracle() {
    const TEST: &str = "qwen35-ep2-oracle";
    if !have_mesh() {
        eprintln!("skip: {TEST} needs HIPFIRE_HAVE_2_GPU=1 and 2+ GPUs");
        return;
    }
    let path = moe_fixture();
    let sha = pin_fixture(TEST, &path, MOE_SIZE, MOE_SHA256);

    // Snapshot lockstep (see helpers): the reference single route runs first,
    // snapshotting its pre-step KV+DeltaNet bytes at every committed position;
    // after it is freed, the mesh replays each position restored to those
    // exact bytes, forced with the same token. Sequential because a resident
    // EP mesh (~13 GiB/rank) plus the full single MoE (~19 GiB) would leave no
    // headroom together on 34 GiB devices.
    let tokenizer = {
        let hfq = HfqFile::open(Path::new(&path)).expect("reopen for tokenizer");
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer")
    };
    let prompt = chatml_prompt(&tokenizer);
    let total = prompt.len() + N_DECODE;
    assert!(total <= KV_MAX, "position budget vs KV max_seq");

    let Some(mut single) = load_single(&path) else {
        eprintln!("skip: {TEST} single-route GPU init failed");
        return;
    };
    assert!(
        single.config.num_experts > 0,
        "{TEST}: MoE fixture must carry routed experts"
    );
    eprintln!(
        "{TEST}: prompt tokens = {} experts={} top_k={} (prompt {PROMPT:?})",
        prompt.len(),
        single.config.num_experts,
        single.config.num_experts_per_tok
    );

    // Record single-route geometry for the cross-check after the mesh loads.
    let s_alias = single.weights.lm_head_aliases_embd;
    let s_nheads = single.kv.n_kv_heads;
    let s_headdim = single.kv.head_dim;
    let s_maxseq = single.kv.max_seq;
    let s_nklayers = single.kv.k_gpu.len();
    let s_nvlayers = single.kv.v_gpu.len();
    let s_k0 = single.kv.k_gpu[0].byte_size();
    let s_v0 = single.kv.v_gpu[0].byte_size();
    let s_dn_len = single.dn.s_matrices.len();
    let s_dn0 = single.dn.s_matrices[0].byte_size();
    log_kv(TEST, &single.kv);
    assert!(
        single.kv.quant_fwht,
        "single route must carry the fwht3 tier"
    );
    check_dn(TEST, &single.dn);

    // Reference run: greedy decode, keeping every committed token, its logits,
    // and its pre-step state snapshot for the mesh replay below.
    let mut committed = Vec::with_capacity(total);
    let mut recorded: Vec<Vec<f32>> = Vec::with_capacity(total);
    let mut snaps: Vec<StateSnap> = Vec::with_capacity(total);
    let mut next_token: u32 = 0;
    for pos in 0..total {
        let token = if pos < prompt.len() {
            prompt[pos]
        } else {
            next_token
        };
        snaps.push(snap_state(
            &mut single.gpu,
            &single.kv,
            &single.dn,
            &single.scratch,
        ));
        let logits = single.step(token, pos);
        let choice = argmax(&logits);
        eprintln!("{TEST}: single pos {pos:>2} token {token:>6} (choice {choice})");
        next_token = choice;
        committed.push(token);
        recorded.push(logits);
    }
    let s_width = recorded[0].len();
    single.free();

    let Some(mut mesh) = load_ep_mesh(TEST, &path, 2) else {
        return;
    };

    eprintln!(
        "{TEST}: route identity — single=forward_scratch/Layout::single, mesh=forward_ep/ep=2 stride-sharded"
    );
    assert_eq!(
        mesh.weights[0].lm_head_aliases_embd, s_alias,
        "alias identity must match between routes"
    );

    // KV geometry: replicated on every rank — exact parity expected.
    for r in 0..2 {
        assert_eq!(mesh.kvs[r].n_kv_heads, s_nheads);
        assert_eq!(mesh.kvs[r].head_dim, s_headdim);
        assert_eq!(mesh.kvs[r].max_seq, s_maxseq);
        assert!(mesh.kvs[r].quant_fwht, "rank {r} must carry the fwht3 tier");
        assert_eq!(mesh.kvs[r].k_gpu.len(), s_nklayers);
        assert_eq!(mesh.kvs[r].v_gpu.len(), s_nvlayers);
        assert_eq!(
            mesh.kvs[r].k_gpu[0].byte_size(),
            s_k0,
            "rank {r} layer-0 K byte extent parity"
        );
        assert_eq!(
            mesh.kvs[r].v_gpu[0].byte_size(),
            s_v0,
            "rank {r} layer-0 V byte extent parity"
        );
        assert!(
            mesh.dns[r].quant == oracle_state_quant(),
            "rank {r} DN must be Q8"
        );
        assert_eq!(mesh.dns[r].s_matrices.len(), s_dn_len);
        assert_eq!(
            mesh.dns[r].s_matrices[0].byte_size(),
            s_dn0,
            "rank {r} S-matrix byte extent parity"
        );
    }

    // Snapshot-lockstep replay: restore every rank to the reference pre-step
    // bytes, step forced with the same token, compare. Only grouping rounding
    // remains; any systematic route difference shows at every position.
    let mut worst: f32 = 0.0;
    for pos in 0..total {
        {
            let EpMesh {
                gpus,
                kvs,
                dns,
                scratches,
                ..
            } = &mut mesh;
            for r in 0..2 {
                restore_state(
                    &mut gpus.devices[r],
                    &kvs[r],
                    &dns[r],
                    &mut scratches[r],
                    &snaps[pos],
                );
            }
            if pos == 0 {
                // Restore-path fidelity probe (once): re-download rank 0 and
                // require byte-identity with the snapshot before any compute.
                let EpMesh {
                    gpus,
                    kvs,
                    dns,
                    scratches,
                    ..
                } = &mut mesh;
                assert_restore_roundtrip(
                    &mut gpus.devices[0],
                    &kvs[0],
                    &dns[0],
                    &mut scratches[0],
                    &snaps[0],
                );
                eprintln!("{TEST}: restore roundtrip pos0 OK");
            }
        }
        let ep_logits = mesh.step(committed[pos], pos);
        assert_eq!(ep_logits.len(), s_width, "logit width at pos {pos}");
        let diff = max_abs_diff(&recorded[pos], &ep_logits);
        worst = worst.max(diff);
        assert!(
            diff <= BOUND_EP_ABS,
            "EP-vs-single logit bound breach at committed position {pos}: {diff:.3e} > {BOUND_EP_ABS:.1e}"
        );
        let choice = argmax(&recorded[pos]);
        assert_eq!(argmax(&ep_logits), choice, "token-id mismatch at pos {pos}");
        eprintln!(
            "{TEST}: pos {pos:>2} token {token:>6} max-logit-diff {diff:.3e} (choice {choice})",
            token = committed[pos]
        );
    }

    evidence(
        TEST,
        &path,
        &sha,
        "ep=2",
        total,
        worst,
        Some(BOUND_EP_ABS),
        "bounded",
        "router/top-k replicated but per-rank owned-expert partials summed via RCCL all-reduce (stride e%2 grouping differs from direct accumulation); snapshot lockstep isolates route math from discrete top-k chaos",
    );
    eprintln!("{TEST}: PASS — {total} committed positions, worst logit diff {worst:.3e}");

    mesh.free();
}

// ── 4. EP4 vs EP2 (MoE, MEASURING-BUT-RED) ──────────────────────────────────────

/// `qwen35_ep4_vs_ep2_oracle` — MoE A3B, expert-parallel EP=4 vs EP=2.
///
/// Relationship: MEASURING-BUT-RED (tokens `assert_eq`, max abs logit diff
/// <= 1e-3, asserted and FAILING honestly — do not widen). Both sides run the
/// same replicated router/top-k and the same RCCL decode reduce, but the
/// expert-to-rank groupings differ (stride `e%4` vs `e%2`), so the partial
/// contents and the RCCL summation order differ. MEASURED: pos 0–1 hold
/// ~1e-5, then a deterministic 2.695e-3 cliff at pos 2 — a DIFFERENT position
/// and magnitude than EP2-vs-single's pos-17/3.6e-2, so the trip is
/// route-pair-specific, not a universal step counter. Same out-of-snapshot
/// suspect set as EP2 (partials, RCCL/peer staging, streams, shard/dummy
/// state, routing-dependent dropped/duplicated expert). The rooted-peer order
/// (`multi_gpu.rs:1759`) would fix the reduction order, but it is NOT the
/// decode path — opt-in via `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1` only — so
/// `assert_eq` on these logits is unwarranted under the production default.
#[test]
#[ignore]
fn qwen35_ep4_vs_ep2_oracle() {
    const TEST: &str = "qwen35-ep4-vs-ep2-oracle";
    if !have_mesh() {
        eprintln!("skip: {TEST} needs HIPFIRE_HAVE_2_GPU=1 and 4 GPUs");
        return;
    }
    let path = moe_fixture();
    let sha = pin_fixture(TEST, &path, MOE_SIZE, MOE_SHA256);

    let tokenizer = {
        let hfq = HfqFile::open(Path::new(&path)).expect("reopen for tokenizer");
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer")
    };
    let prompt = chatml_prompt(&tokenizer);
    let total = prompt.len() + N_DECODE;
    assert!(total <= KV_MAX, "position budget vs KV max_seq");

    // Snapshot lockstep (see helpers): EP=2 runs first, snapshotting its rank-0
    // pre-step bytes at every committed position (KV/DN are replicated, so
    // rank 0 is authoritative); after it is freed, EP=4 replays each position
    // restored to those exact bytes. Sequential because two stacked meshes
    // exceed device VRAM. The discrete router then sees identical inputs and
    // only grouping rounding remains.
    let Some(mut ep2) = load_ep_mesh(TEST, &path, 2) else {
        return;
    };
    eprintln!(
        "{TEST}: prompt tokens = {} experts={} top_k={} (prompt {PROMPT:?})",
        prompt.len(),
        ep2.config.num_experts,
        ep2.config.num_experts_per_tok
    );

    // Record EP=2 geometry for the cross-check after EP=4 loads.
    let r_alias = ep2.weights[0].lm_head_aliases_embd;
    let r_nheads = ep2.kvs[0].n_kv_heads;
    let r_k0 = ep2.kvs[0].k_gpu[0].byte_size();
    let r_dn_len = ep2.dns[0].s_matrices.len();
    let r_dim = ep2.config.dim;
    let r_nexp = ep2.config.num_experts;

    // Reference run: EP=2 commits greedily, keeping tokens, logits, and
    // pre-step state snapshots for the EP=4 replay below.
    let mut committed = Vec::with_capacity(total);
    let mut recorded: Vec<Vec<f32>> = Vec::with_capacity(total);
    let mut snaps: Vec<StateSnap> = Vec::with_capacity(total);
    let mut next_token: u32 = 0;
    for pos in 0..total {
        let token = if pos < prompt.len() {
            prompt[pos]
        } else {
            next_token
        };
        snaps.push(snap_state(
            &mut ep2.gpus.devices[0],
            &ep2.kvs[0],
            &ep2.dns[0],
            &ep2.scratches[0],
        ));
        let logits = ep2.step(token, pos);
        let choice = argmax(&logits);
        eprintln!("{TEST}: ep2 pos {pos:>2} token {token:>6} (choice {choice})");
        next_token = choice;
        committed.push(token);
        recorded.push(logits);
    }
    let r_width = recorded[0].len();
    ep2.free();

    let Some(mut ep4) = load_ep_mesh(TEST, &path, 4) else {
        eprintln!("skip: {TEST} needs 4 visible GPUs for the EP=4 side");
        return;
    };
    assert_eq!(ep4.config.dim, r_dim, "route config dim");
    assert_eq!(ep4.config.num_experts, r_nexp, "route expert count");

    eprintln!("{TEST}: route identity — ref=forward_ep/ep=2 stride, mesh=forward_ep/ep=4 stride");
    assert_eq!(
        ep4.weights[0].lm_head_aliases_embd, r_alias,
        "alias identity must match between routes"
    );

    // KV geometry: replicated on every rank of both meshes — exact parity.
    for r in 0..4 {
        assert_eq!(ep4.kvs[r].n_kv_heads, r_nheads);
        assert_eq!(
            ep4.kvs[r].k_gpu[0].byte_size(),
            r_k0,
            "ep4 rank {r} layer-0 K byte extent parity vs recorded ep2"
        );
        assert!(
            ep4.kvs[r].quant_fwht,
            "ep4 rank {r} must carry the fwht3 tier"
        );
        assert!(
            ep4.dns[r].quant == oracle_state_quant(),
            "ep4 rank {r} DN must be Q8"
        );
        assert_eq!(ep4.dns[r].s_matrices.len(), r_dn_len);
    }

    // Snapshot-lockstep replay: restore all four ranks to the EP=2 pre-step
    // bytes, step forced with the same token, compare.
    let mut worst: f32 = 0.0;
    for pos in 0..total {
        {
            let EpMesh {
                gpus,
                kvs,
                dns,
                scratches,
                ..
            } = &mut ep4;
            for r in 0..4 {
                restore_state(
                    &mut gpus.devices[r],
                    &kvs[r],
                    &dns[r],
                    &mut scratches[r],
                    &snaps[pos],
                );
            }
        }
        let logits4 = ep4.step(committed[pos], pos);
        assert_eq!(logits4.len(), r_width, "logit width at pos {pos}");
        let diff = max_abs_diff(&recorded[pos], &logits4);
        worst = worst.max(diff);
        assert!(
            diff <= BOUND_EP_ABS,
            "EP4-vs-EP2 logit bound breach at committed position {pos}: {diff:.3e} > {BOUND_EP_ABS:.1e}"
        );
        let choice = argmax(&recorded[pos]);
        assert_eq!(argmax(&logits4), choice, "token-id mismatch at pos {pos}");
        eprintln!(
            "{TEST}: pos {pos:>2} token {token:>6} max-logit-diff {diff:.3e} (choice {choice})",
            token = committed[pos]
        );
    }

    evidence(
        TEST,
        &path,
        &sha,
        "ep=4 vs ep=2",
        total,
        worst,
        Some(BOUND_EP_ABS),
        "bounded",
        "same replicated router/top-k and same RCCL decode reduce, but stride e%4 vs e%2 groupings change partial contents and RCCL summation order; rooted-peer order is opt-in only, not the production decode path; snapshot lockstep isolates route math from discrete top-k chaos",
    );
    eprintln!("{TEST}: PASS — {total} committed positions, worst logit diff {worst:.3e}");

    ep4.free();
}
