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
//! deterministic kernels make the two equivalent). The dense PP/TP tests assert
//! token ids, bounded logits, KV geometry/byte extents, the position counter,
//! alias identity and route identity, then print an evidence block. The EP
//! tests assert cross-rank route/state agreement inside the mesh and strict
//! byte equality of the recorded rank-0 logits at every committed position.
//! `HIPFIRE_HAVE_2_GPU=1` gates physical multi-GPU runs; the snapshotted
//! `HIPFIRE_EMULATE_GPUS >= 2` runtime policy also admits logical EP runs on
//! one physical device. Emulation is accepted only for the exact four-logical
//! rank, single-gfx1151 batch shape below; it is diagnostic evidence, not
//! physical EP transport, synchronization, throughput, or product admission.
//! Pattern follows the template oracle
//! (`hipfire-arch-llama` `pinned_fixture_manifest_legacy_parity_oracle`) and
//! goes beyond `pp_parity.rs` (which compares argmax tokens only).
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
//! * EP2-vs-single (MoE) — ROUTE/STATE AGREEMENT + strict single parity.
//!   Production decode is root-routed partial
//!   (`EpMoeCombineMode::RootRoutedPartial` / `indexed-decode-routed-partial`):
//!   the root authorizes a GPU top-K route (`MoeRouteProducerProof` + route IDs);
//!   non-roots install those IDs (no local router authority). The repaired
//!   path gathers the global expert slots at the root and then uses the
//!   ordinary canonical slot-order combine. The asserted EP invariants are
//!   therefore cross-RANK, inside the mesh: after every `forward_ep` (which
//!   ends with a per-rank `device_synchronize`), each rank's last-layer
//!   `moe_topk_indices` + `moe_topk_weights` bytes equal the root's, and each
//!   rank's post-collective residual `s.x` float-bits equal the root's (the
//!   final norm reads `s.x` into `s.tmp` on rank 0 only, so `s.x` — not `s.tmp`
//!   — is the comparable residual). Compared logits must be finite and their
//!   complete f32 bit vectors must equal the single route at every position.
//!   Snapshot lockstep pins KV + DeltaNet + the FULL scratch (30 fixed + 14
//!   MoE-opt incl. top-k indices/weights and `moe_down_expanded` + pos_bufs,
//!   all by true `DeviceBuffer::size`) so the discrete router sees identical
//!   inputs. Lockstep (not interleaving) is the design: top-k is discrete and
//!   small grouping noise under interleaved forcing can flip experts.
//! * EP4-vs-EP2 (MoE) — ROUTE/STATE AGREEMENT + strict cross-mesh parity.
//!   Both sides gather global expert slots at the root and then use the
//!   ordinary canonical slot-order combine; the repaired association no longer
//!   depends on the expert-to-rank stride (`e%4` vs `e%2`). Asserted:
//!   per-rank route-byte and post-collective residual agreement inside the
//!   EP4 mesh, finite logits, and complete byte equality between every recorded
//!   EP2 rank-0 logit vector and the EP4 rank-0 vector.
//!
//! Substrate note: the EP tests drive the qwen35-level EP substrate directly
//! (sealed per-rank `load_weights_ep_rank` over the `init_ep` mesh +
//! `forward_ep` root-routed partial driver) because only the direct substrate
//! exposes the per-rank scratch/KV/DeltaNet state these snapshot/forced-state
//! comparisons restore and download. The product TP4 EP serve path is
//! validated separately through the daemon. The batch oracle uses one
//! read-only diagnostic observation helper for its private rank outputs; no
//! user-facing config or CLI seam is involved.
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
//!
//! For the single-device diagnostic EP shape, leave `HIPFIRE_HAVE_2_GPU`
//! unset, set `HIP_VISIBLE_DEVICES=0` and `HIPFIRE_EMULATE_GPUS=2` or `4`,
//! and select the required peer-allreduce transport externally.

use hipfire_arch_qwen35::qwen35::{
    self, DeltaNetState, HfqSource, Layout, Qwen35BatchLoadConfig, Qwen35Config,
    Qwen35DecodeBatchEpState, Qwen35EpBatchReceipt, Qwen35Scratch, Qwen35ScratchSet, Qwen35Weights,
    StateQuant,
};
use hipfire_runtime::ep::ensure_rank_streams;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{KvCache, KvCacheExt};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
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
// Absolute logit bounds for the bounded dense relationships. Set from measured
// worst-case per-position diffs (see evidence output) at ~2x headroom, rounded
// up; a wrong expert / wrong reduction / wrong band would move logits by
// orders of magnitude more, and any token flip inside a bound is a hard
// failure. The EP oracles assert cross-rank route/state agreement and strict
// rank-0 logit equality, so they carry no logit bound.
const BOUND_PP_ABS: f32 = 2.0;
const BOUND_TP_ABS: f32 = 1.0;

// ── gates / fixtures ─────────────────────────────────────────────────────────

fn emulation_enabled() -> bool {
    hipfire_runtime::config::get().emulate_gpus.is_some()
}

fn have_mesh() -> bool {
    std::env::var("HIPFIRE_HAVE_2_GPU").as_deref() == Ok("1") || emulation_enabled()
}

/// Exact diagnostic exception for four logical ranks aliased to one Halo GPU.
fn emulated_single_gfx1151_ep4(gpus: &Gpus) -> bool {
    if !emulation_enabled() || gpus.devices.len() != 4 {
        return false;
    }
    let Some(first) = gpus.devices.first() else {
        return false;
    };
    first.arch == "gfx1151"
        && gpus
            .devices
            .iter()
            .all(|gpu| gpu.arch == "gfx1151" && gpu.device_id == first.device_id)
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
fn record_f32_bit_mismatch(
    tag: &str,
    pos: usize,
    expected: &[f32],
    actual: &[f32],
    failures: &mut Vec<String>,
) {
    if expected.len() != actual.len() {
        failures.push(format!(
            "pos {pos}: {tag} width {} != {}",
            actual.len(),
            expected.len()
        ));
        return;
    }
    for (lane, (&want, &got)) in expected.iter().zip(actual).enumerate() {
        if want.to_bits() != got.to_bits() {
            failures.push(format!(
                "pos {pos}: {tag} lane {lane}: expected {:08x} != actual {:08x}",
                want.to_bits(),
                got.to_bits()
            ));
            return;
        }
    }
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

/// Optional post-step EP observation sink. The files intentionally contain
/// only comparable bytes and the separately recorded input token/position;
/// fixture/build identities belong in the surrounding evidence directory.
struct EpOracleDump {
    root: PathBuf,
}

impl EpOracleDump {
    fn from_env(topology: &str) -> Option<Self> {
        let root = std::env::var_os("HIPFIRE_EP_ORACLE_DUMP")?;
        if root.is_empty() {
            return None;
        }
        let root = PathBuf::from(root).join(topology);
        std::fs::create_dir_all(&root)
            .unwrap_or_else(|e| panic!("EP oracle dump root {}: {e}", root.display()));
        Some(Self { root })
    }

    fn write(path: PathBuf, bytes: &[u8]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .unwrap_or_else(|e| panic!("EP oracle dump dir {}: {e}", parent.display()));
        }
        std::fs::write(&path, bytes)
            .unwrap_or_else(|e| panic!("EP oracle dump file {}: {e}", path.display()));
    }

    fn position_root(&self, pos: usize) -> PathBuf {
        self.root.join(format!("position-{pos}"))
    }

    fn rank_position(&self, rank: usize, pos: usize) -> PathBuf {
        self.root
            .join(format!("rank-{rank}"))
            .join(format!("position-{pos}"))
    }

    fn write_f32(path: PathBuf, values: &[f32]) {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Self::write(path, &bytes);
    }

    fn write_agreement(
        &self,
        rank: usize,
        pos: usize,
        route_indices: &[f32],
        route_weights: &[f32],
        residual_x: &[f32],
    ) {
        let dir = self.rank_position(rank, pos);
        Self::write_f32(dir.join("route-indices.f32"), route_indices);
        Self::write_f32(dir.join("route-weights.f32"), route_weights);
        Self::write_f32(dir.join("residual-x.f32"), residual_x);
    }

    fn write_snapshot(&self, rank: usize, pos: usize, snap: &StateSnap) {
        let dir = self.rank_position(rank, pos);
        for (layer, bytes) in snap.kv_k.iter().enumerate() {
            Self::write(dir.join(format!("kv-k-{layer}.bin")), bytes);
        }
        for (layer, bytes) in snap.kv_v.iter().enumerate() {
            Self::write(dir.join(format!("kv-v-{layer}.bin")), bytes);
        }
        for (layer, bytes) in snap.dn_s.iter().enumerate() {
            Self::write(dir.join(format!("dn-s-{layer}.bin")), bytes);
        }
        for (layer, bytes) in snap.dn_sc.iter().enumerate() {
            Self::write(dir.join(format!("dn-scales-{layer}.bin")), bytes);
        }
        for (layer, bytes) in snap.dn_c.iter().enumerate() {
            Self::write(dir.join(format!("dn-conv-{layer}.bin")), bytes);
        }
        for (layer, bytes) in snap.dn_ef.iter().enumerate() {
            Self::write(dir.join(format!("dn-ef-{layer}.bin")), bytes);
        }
        for (field, bytes) in snap.scratch.fixed.iter().enumerate() {
            Self::write(dir.join(format!("scratch-fixed-{field}.bin")), bytes);
        }
        for (field, bytes) in snap.scratch.opt.iter().enumerate() {
            if let Some(bytes) = bytes {
                Self::write(dir.join(format!("scratch-opt-{field}.bin")), bytes);
            }
        }
        Self::write(dir.join("scratch-pos.bin"), &snap.scratch.pos_buf);
        Self::write(dir.join("scratch-pos3.bin"), &snap.scratch.pos_buf3);
    }

    fn write_observation(&self, mesh: &mut EpMesh, token: u32, pos: usize, root_logits: &[f32]) {
        let position = self.position_root(pos);
        Self::write(position.join("token.u32"), &token.to_le_bytes());
        Self::write(position.join("position.u64"), &(pos as u64).to_le_bytes());
        Self::write_f32(
            self.rank_position(0, pos).join("root-logits.f32"),
            root_logits,
        );
        for rank in 0..mesh.gpus.devices.len() {
            let snap = snap_state(
                &mut mesh.gpus.devices[rank],
                &mesh.kvs[rank],
                &mesh.dns[rank],
                &mesh.scratches[rank],
            );
            self.write_snapshot(rank, pos, &snap);
        }
    }
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
    let mut gpus = match Gpus::init_ep(tp, config.n_layers) {
        Ok(g) if g.devices.len() == tp => g,
        Ok(g) => {
            eprintln!(
                "skip: {test} init_ep({tp}) gave {} devices",
                g.devices.len()
            );
            return None;
        }
        Err(e) => {
            eprintln!("skip: {test} init_ep({tp}) failed: {e:?}");
            return None;
        }
    };
    // Sealed per-rank streaming load: each rank owns its plan-derived compact
    // experts with layout-specific zero dummies elsewhere. The cloned mesh +
    // physical list keep the plan mesh and the runtime topology in agreement.
    let mesh = gpus.mesh.clone();
    let physical_devices: Vec<i32> = gpus.devices.iter().map(|dev| dev.device_id).collect();
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
        let w = qwen35::load_weights_ep_rank(
            &mut hfq,
            &mut gpus.devices[r],
            &config,
            &mesh,
            &physical_devices,
            shard.clone(),
            r,
        );
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
        self.step_with_lease(token, pos, None)
    }

    /// Exercise the same public EP decode with a caller-owned peer lease.
    fn step_with_lease(
        &mut self,
        token: u32,
        pos: usize,
        peer_lease: Option<&hipfire_runtime::multi_gpu::PeerReduceScratchLease>,
    ) -> Vec<f32> {
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
            peer_lease,
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

// ── EP cross-rank route/state agreement ─────────────────────────────────────────
//
// Production EP decode is root-routed partial: the root routes IDs+weights
// once on the GPU top-K path and every rank consumes exact copies, then the
// weighted partials reduce through the fixed root fold and the same
// post-collective residual lands on all ranks. The observable contract is
// therefore cross-RANK, inside the mesh — not single-parity:
//   * every rank's last-layer `moe_topk_indices` + `moe_topk_weights` bytes
//     equal the root's (wrong non-root route/weights fail here);
//   * every rank's post-collective residual `s.x` float-bits equal the
//     root's (a non-converged partial all-reduce fails here). The final norm
//     reads `s.x` into `s.tmp` on rank 0 only (`forward_ep` §3 in
//     `ep_batch.rs`), so `s.x` — not `s.tmp` — is the comparable residual,
//     and the compared residual must be finite (bit-equality alone would
//     admit identical NaN payloads).
// Consumer observations only (`bind_thread` + `download_f32`, the same calls
// the step functions already use); no new prod/test hooks. Returns failure
// lines instead of panicking so the caller can free the mesh BEFORE the
// terminal assert — a mid-loop panic would leak GPU owners and silently
// starve the next ignored test under `--test-threads=1`.
fn assert_ep_rank_agreement(
    gpus: &mut Gpus,
    scratches: &[Qwen35Scratch],
    pos: usize,
    dump: Option<&EpOracleDump>,
) -> Vec<String> {
    fn cmp_bits(
        tag: &str,
        pos: usize,
        rank: usize,
        root: &[f32],
        other: &[f32],
        failures: &mut Vec<String>,
    ) {
        if root.len() != other.len() {
            failures.push(format!(
                "pos {pos}: rank {rank} {tag} len {} != root len {}",
                other.len(),
                root.len()
            ));
            return;
        }
        for (lane, (a, b)) in root.iter().zip(other.iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                failures.push(format!(
                    "pos {pos}: rank {rank} {tag} lane {lane}: root {:08x} != rank {:08x}",
                    a.to_bits(),
                    b.to_bits()
                ));
                return;
            }
        }
    }

    let mut failures = Vec::new();
    let n = gpus.devices.len();
    if scratches.len() != n {
        failures.push(format!(
            "pos {pos}: scratch count {} != rank count {n}",
            scratches.len()
        ));
        return failures;
    }
    // `forward_ep` ends with a per-rank `device_synchronize`, so these bound
    // downloads observe settled state.
    gpus.devices[0]
        .bind_thread()
        .expect("ep agreement bind root");
    let Some(root_ids) = scratches[0].moe_topk_indices.as_ref() else {
        failures.push(format!("pos {pos}: root moe_topk_indices missing"));
        return failures;
    };
    let Some(root_weights) = scratches[0].moe_topk_weights.as_ref() else {
        failures.push(format!("pos {pos}: root moe_topk_weights missing"));
        return failures;
    };
    let root_ids = gpus.devices[0]
        .download_f32(root_ids)
        .expect("ep agreement root ids");
    let root_weights = gpus.devices[0]
        .download_f32(root_weights)
        .expect("ep agreement root weights");
    let root_x = gpus.devices[0]
        .download_f32(&scratches[0].x)
        .expect("ep agreement root residual");
    if let Some(dump) = dump {
        dump.write_agreement(0, pos, &root_ids, &root_weights, &root_x);
    }
    if root_x.iter().any(|v| !v.is_finite()) {
        failures.push(format!(
            "pos {pos}: root post-collective residual non-finite"
        ));
    }
    for r in 1..n {
        gpus.devices[r]
            .bind_thread()
            .expect("ep agreement bind rank");
        let (Some(ids), Some(weights)) = (
            scratches[r].moe_topk_indices.as_ref(),
            scratches[r].moe_topk_weights.as_ref(),
        ) else {
            failures.push(format!("pos {pos}: rank {r} moe_topk route missing"));
            continue;
        };
        let ids = gpus.devices[r]
            .download_f32(ids)
            .expect("ep agreement rank ids");
        let weights = gpus.devices[r]
            .download_f32(weights)
            .expect("ep agreement rank weights");
        let x = gpus.devices[r]
            .download_f32(&scratches[r].x)
            .expect("ep agreement rank residual");
        if let Some(dump) = dump {
            dump.write_agreement(r, pos, &ids, &weights, &x);
        }
        cmp_bits("moe_topk_indices", pos, r, &root_ids, &ids, &mut failures);
        cmp_bits(
            "moe_topk_weights",
            pos,
            r,
            &root_weights,
            &weights,
            &mut failures,
        );
        cmp_bits(
            "post-collective residual x",
            pos,
            r,
            &root_x,
            &x,
            &mut failures,
        );
        if x.iter().any(|v| !v.is_finite()) {
            failures.push(format!(
                "pos {pos}: rank {r} post-collective residual non-finite"
            ));
        }
    }
    failures
}

// ── 3. EP2 vs single (MoE, route/state agreement + strict parity) ────────────────

/// `qwen35_ep2_vs_single_oracle` — MoE A3B (256 experts, top_k=8),
/// expert-parallel EP=2 vs single.
///
/// Relationship: ROUTE/STATE AGREEMENT inside the mesh plus strict byte-exact
/// single parity. Asserted after every `forward_ep`: all ranks' last-layer
/// `moe_topk_indices` + `moe_topk_weights` bytes equal the root's, all ranks'
/// post-collective residual `s.x` float-bits equal the root's, compared logits
/// are finite, and the complete rank-0 logit vector equals the recorded single
/// vector at every committed position (see `assert_ep_rank_agreement`). The
/// repaired association gathers global expert slots at the root and then uses
/// the ordinary canonical slot-order combine, so the strict comparison is
/// intentional rather than a diagnostic bound.
/// Production MoE EP decode is root-routed partial: root issues
/// `MoeRouteProducerProof` + route IDs on the GPU top-K path; non-roots
/// install those IDs. Snapshot lockstep restores the reference single route's
/// KV + DeltaNet + FULL scratch (30 fixed + 14 MoE-opt + pos_bufs, true
/// `DeviceBuffer::size`) before each forced mesh step, so the discrete router
/// sees identical inputs.
#[test]
#[ignore]
fn qwen35_ep2_vs_single_oracle() {
    const TEST: &str = "qwen35-ep2-oracle";
    if !have_mesh() {
        eprintln!(
            "skip: {TEST} needs HIPFIRE_HAVE_2_GPU=1 with physical ranks or runtime emulation"
        );
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
    let dump = EpOracleDump::from_env("ep2");

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
    // bytes, step forced with the same token, then check the actual EP
    // contract — cross-rank route/state agreement plus strict single-route
    // rank-0 logit equality. Failures accumulate across all positions and the
    // mesh is freed BEFORE the terminal assert, so a real cross-rank or
    // cross-route failure cannot leak GPU owners into the next test.
    let mut worst: f32 = 0.0;
    let mut agreement_failures: Vec<String> = Vec::new();
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
        if let Some(dump) = dump.as_ref() {
            dump.write_observation(&mut mesh, committed[pos], pos, &ep_logits);
        }
        if ep_logits.len() != s_width {
            agreement_failures.push(format!(
                "pos {pos}: logit width {} != single width {s_width}",
                ep_logits.len()
            ));
            continue;
        }
        if recorded[pos].iter().any(|v| !v.is_finite()) {
            agreement_failures.push(format!("pos {pos}: single logits non-finite"));
        }
        if ep_logits.iter().any(|v| !v.is_finite()) {
            agreement_failures.push(format!("pos {pos}: EP logits non-finite"));
        }
        {
            let EpMesh {
                gpus, scratches, ..
            } = &mut mesh;
            agreement_failures.extend(assert_ep_rank_agreement(
                gpus,
                scratches,
                pos,
                dump.as_ref(),
            ));
        }
        record_f32_bit_mismatch(
            "single-vs-ep rank-0 logits",
            pos,
            &recorded[pos],
            &ep_logits,
            &mut agreement_failures,
        );
        let diff = max_abs_diff(&recorded[pos], &ep_logits);
        worst = worst.max(diff);
        eprintln!(
            "{TEST}: pos {pos:>2} token {token:>6} exact-logit-check max-abs-diff {diff:.3e}",
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
        None,
        "root-gathered-global-slots-canonical-combine",
        "root-gathered global expert slots followed by the ordinary canonical slot-order combine; per-rank route-byte and post-combine residual agreement asserted, logits finite, and every single-vs-EP2 rank-0 logit vector compared byte-exactly at each committed position",
    );
    mesh.free();
    assert!(
        agreement_failures.is_empty(),
        "{TEST}: EP cross-rank route/state agreement failures:\n{}",
        agreement_failures.join("\n")
    );
    eprintln!(
        "{TEST}: PASS — {total} committed positions, strict single-vs-ep rank-0 logit equality (worst abs diff {worst:.3e})"
    );
}

// ── 4. EP4 vs EP2 (MoE, route/state agreement + strict parity) ───────────────────

/// `qwen35_ep4_vs_ep2_oracle` — MoE A3B, expert-parallel EP=4 vs EP=2.
///
/// Relationship: ROUTE/STATE AGREEMENT inside the EP4 mesh plus strict
/// cross-mesh parity. Asserted after every `forward_ep`: all four ranks'
/// last-layer `moe_topk_indices` + `moe_topk_weights` bytes equal the EP4
/// root's, all four ranks' post-collective residual `s.x` float-bits equal
/// the EP4 root's, compared logits are finite, and each EP4 rank-0 logit
/// vector equals the recorded EP2 rank-0 vector byte-for-byte. Both sides
/// gather global expert slots at the root and then use the ordinary canonical
/// slot-order combine; the `e%4`/`e%2` residency stride is not a numerical
/// association difference. Snapshot lockstep restores EP2 rank-0 pre-step
/// bytes before each forced EP4 step, so the discrete router sees identical
/// inputs.
#[test]
#[ignore]
fn qwen35_ep4_vs_ep2_oracle() {
    const TEST: &str = "qwen35-ep4-vs-ep2-oracle";
    if !have_mesh() {
        eprintln!(
            "skip: {TEST} needs HIPFIRE_HAVE_2_GPU=1 with physical ranks or runtime emulation"
        );
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
    let dump = EpOracleDump::from_env("ep4");
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
    // bytes, step forced with the same token, then check the actual EP
    // contract — cross-rank route/state agreement inside the EP4 mesh plus
    // strict EP2 rank-0 logit equality. Failures accumulate across all
    // positions and the mesh is freed BEFORE the terminal assert, so a real
    // cross-rank or cross-route failure cannot leak GPU owners into the next
    // test.
    let mut worst: f32 = 0.0;
    let mut agreement_failures: Vec<String> = Vec::new();
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
        if let Some(dump) = dump.as_ref() {
            dump.write_observation(&mut ep4, committed[pos], pos, &logits4);
        }
        if logits4.len() != r_width {
            agreement_failures.push(format!(
                "pos {pos}: logit width {} != ep2 width {r_width}",
                logits4.len()
            ));
            continue;
        }
        if recorded[pos].iter().any(|v| !v.is_finite()) {
            agreement_failures.push(format!("pos {pos}: EP2 logits non-finite"));
        }
        if logits4.iter().any(|v| !v.is_finite()) {
            agreement_failures.push(format!("pos {pos}: EP4 logits non-finite"));
        }
        {
            let EpMesh {
                gpus, scratches, ..
            } = &mut ep4;
            agreement_failures.extend(assert_ep_rank_agreement(
                gpus,
                scratches,
                pos,
                dump.as_ref(),
            ));
        }
        record_f32_bit_mismatch(
            "ep2-vs-ep4 rank-0 logits",
            pos,
            &recorded[pos],
            &logits4,
            &mut agreement_failures,
        );
        let diff = max_abs_diff(&recorded[pos], &logits4);
        worst = worst.max(diff);
        eprintln!(
            "{TEST}: pos {pos:>2} token {token:>6} exact-logit-check max-abs-diff {diff:.3e}",
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
        None,
        "root-gathered-global-slots-canonical-combine",
        "root-gathered global expert slots followed by the ordinary canonical slot-order combine on both sides; per-rank route-byte and post-combine residual agreement asserted inside EP4, logits finite, and every EP2-vs-EP4 rank-0 logit vector compared byte-exactly at each committed position",
    );
    ep4.free();
    assert!(
        agreement_failures.is_empty(),
        "{TEST}: EP cross-rank route/state agreement failures:\n{}",
        agreement_failures.join("\n")
    );
    eprintln!(
        "{TEST}: PASS — {total} committed positions, strict ep2-vs-ep4 rank-0 logit equality (worst abs diff {worst:.3e})"
    );
}

#[derive(Debug, Serialize)]
struct BatchReceiptTrace {
    epoch: u64,
    rank_count: u8,
    rank_mask: u64,
    rows: u32,
    moe_collectives: u32,
    reduce: String,
    parallelism: String,
}

fn batch_receipt_trace(receipt: &Qwen35EpBatchReceipt) -> BatchReceiptTrace {
    BatchReceiptTrace {
        epoch: receipt.epoch(),
        rank_count: receipt.rank_count(),
        rank_mask: receipt.rank_mask(),
        rows: receipt.rows(),
        moe_collectives: receipt.moe_collectives(),
        reduce: format!("{:?}", receipt.reduce()),
        parallelism: format!("{:?}", receipt.parallelism()),
    }
}

#[derive(Debug, Serialize)]
struct BatchTraceEvent {
    phase: &'static str,
    tick: usize,
    active_mask: u64,
    tokens: [u32; 2],
    positions: [usize; 2],
    sampled: [Option<u32>; 2],
    rng_states: [Option<u32>; 2],
    receipt: Option<BatchReceiptTrace>,
    state_epoch: u64,
    poison_mask: u64,
    expected_positions: [Option<usize>; 2],
    error: Option<String>,
}

fn batch_trace_event(
    batch: &Qwen35DecodeBatchEpState,
    phase: &'static str,
    tick: usize,
    active_mask: u64,
    tokens: [u32; 2],
    positions: [usize; 2],
    sampled: [Option<u32>; 2],
    rng_states: [Option<u32>; 2],
    receipt: Option<&Qwen35EpBatchReceipt>,
    expected_positions: [Option<usize>; 2],
    error: Option<String>,
) -> BatchTraceEvent {
    BatchTraceEvent {
        phase,
        tick,
        active_mask,
        tokens,
        positions,
        sampled,
        rng_states,
        receipt: receipt.map(batch_receipt_trace),
        state_epoch: batch.epoch(),
        poison_mask: batch.poison_mask(),
        expected_positions,
        error,
    }
}
fn observe_batch_rank_outputs(
    batch: &Qwen35DecodeBatchEpState,
    gpus: &mut Gpus,
    operation: &str,
    failures: &mut Vec<String>,
) {
    let (root_logits, residuals) = match batch.download_rank_outputs(gpus) {
        Ok(outputs) => outputs,
        Err(error) => {
            failures.push(format!(
                "{operation}: rank-output download failed: {error:?}"
            ));
            return;
        }
    };
    if root_logits.iter().any(|value| !value.is_finite()) {
        failures.push(format!("{operation}: rank-0 logits non-finite"));
    }
    if residuals.len() != 4 {
        failures.push(format!(
            "{operation}: residual rank count {} != 4",
            residuals.len()
        ));
        return;
    }
    if residuals[0].iter().any(|value| !value.is_finite()) {
        failures.push(format!(
            "{operation}: rank-0 post-combine residual non-finite"
        ));
    }
    for rank in 1..residuals.len() {
        if residuals[rank].iter().any(|value| !value.is_finite()) {
            failures.push(format!(
                "{operation}: rank {rank} post-combine residual non-finite"
            ));
        }
        record_f32_bit_mismatch(
            &format!("{operation} post-combine residual rank {rank}"),
            0,
            &residuals[0],
            &residuals[rank],
            failures,
        );
    }
}
#[derive(Debug, Serialize)]
struct BatchTraceFile {
    kind: &'static str,
    topology: &'static str,
    max_batch: usize,
    lane_capacity: usize,
    repeat_capacity: usize,
    prefill_chunk: usize,
    events: Vec<BatchTraceEvent>,
}

fn write_batch_trace(load_cfg: Qwen35BatchLoadConfig, events: Vec<BatchTraceEvent>) {
    let Some(root) = std::env::var_os("HIPFIRE_EP_ORACLE_DUMP") else {
        return;
    };
    if root.is_empty() {
        return;
    }
    let trace = BatchTraceFile {
        kind: "batch-behavior-trace",
        topology: "ep4-batch",
        max_batch: load_cfg.max_batch,
        lane_capacity: load_cfg.lane_capacity,
        repeat_capacity: load_cfg.repeat_capacity,
        prefill_chunk: load_cfg.prefill_chunk,
        events,
    };
    let bytes = serde_json::to_vec_pretty(&trace).expect("serialize EP batch behavior trace");
    EpOracleDump::write(
        PathBuf::from(root)
            .join("ep4-batch")
            .join("batch-behavior-trace.json"),
        &bytes,
    );
}

fn seed_ep_batch(
    batch: &mut Qwen35DecodeBatchEpState,
    mesh: &mut EpMesh,
    prompt: &[u32],
) -> ([u32; 2], [u32; 2], [Qwen35EpBatchReceipt; 2]) {
    batch.reset_all(&mut mesh.gpus).expect("EP batch reset_all");
    let receipt0 = batch
        .prefill_lane(&mut mesh.gpus, &mesh.weights, &mesh.config, 0, prompt)
        .expect("EP batch prefill lane 0");
    let receipt1 = batch
        .prefill_lane(&mut mesh.gpus, &mesh.weights, &mesh.config, 1, prompt)
        .expect("EP batch prefill lane 1");
    let (token0, rng0) = batch
        .sample_lane(&mut mesh.gpus, &mesh.config, 0, 0.0, 1.0, None, 1)
        .expect("EP batch initial sample lane 0");
    let (token1, rng1) = batch
        .sample_lane(&mut mesh.gpus, &mesh.config, 1, 0.0, 1.0, None, 1)
        .expect("EP batch initial sample lane 1");
    ([token0, token1], [rng0, rng1], [receipt0, receipt1])
}

#[test]
#[ignore]
fn qwen35_ep4_batch_step_recipe_oracle() {
    const TEST: &str = "qwen35-ep4-batch-step-recipe-oracle";
    if !have_mesh() {
        eprintln!(
            "skip: {TEST} needs HIPFIRE_HAVE_2_GPU=1 with physical ranks or runtime emulation"
        );
        return;
    }
    let path = moe_fixture();
    let _sha = pin_fixture(TEST, &path, MOE_SIZE, MOE_SHA256);
    let tokenizer = {
        let hfq = HfqFile::open(Path::new(&path)).expect("reopen for tokenizer");
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer")
    };
    let prompt = chatml_prompt(&tokenizer);
    let load_cfg = Qwen35BatchLoadConfig::new(2, 256, 128, 8);
    assert!(
        prompt.len() < load_cfg.lane_capacity,
        "{TEST}: chatml prompt must leave decode capacity"
    );

    let Some(mut mesh) = load_ep_mesh(TEST, &path, 4) else {
        return;
    };
    let emulated_halo = emulated_single_gfx1151_ep4(&mesh.gpus);
    let mut physical_ids: Vec<i32> = mesh.gpus.devices.iter().map(|gpu| gpu.device_id).collect();
    let all_gfx1201 = mesh.gpus.devices.iter().all(|gpu| gpu.arch == "gfx1201");
    physical_ids.sort_unstable();
    let distinct_devices = physical_ids.windows(2).all(|pair| pair[0] != pair[1]);
    if !emulated_halo && (mesh.gpus.devices.len() != 4 || !all_gfx1201 || !distinct_devices) {
        eprintln!(
            "skip: {TEST} requires four distinct gfx1201 devices (got {:?}, archs {:?})",
            physical_ids,
            mesh.gpus
                .devices
                .iter()
                .map(|gpu| gpu.arch.as_str())
                .collect::<Vec<_>>()
        );
        mesh.free();
        return;
    }

    let mut batch =
        match Qwen35DecodeBatchEpState::new(&mut mesh.gpus, &mesh.weights, &mesh.config, &load_cfg)
        {
            Ok(batch) => batch,
            Err(error) => {
                mesh.free();
                panic!("{TEST}: four-device batch admission failed: {error:?}");
            }
        };
    let mut failures = Vec::new();
    let mut events = Vec::new();

    // Exercise the sequential prefill fallback through the batch owner's live
    // lease and borrowed seed scratch, then one ordinary decode under that
    // same lease. This is deliberately not part of the raw-state dump.
    {
        let (pbs, seed_partials, peer_lease) = batch.sequential_prefill_scratch_mut();
        for (chunk_idx, chunk) in prompt.chunks(8).enumerate() {
            qwen35::forward_prefill_batch_ep(
                &mut mesh.gpus,
                &mesh.weights,
                &mesh.config,
                chunk,
                chunk_idx * 8,
                &mut mesh.kvs,
                &mut mesh.dns,
                &mesh.scratches,
                pbs,
                seed_partials,
                peer_lease,
            )
            .unwrap_or_else(|error| panic!("{TEST}: sequential EP prefill: {error:?}"));
        }
    }
    let peer_lease = batch.peer_reduce_lease();
    let _ = mesh.step_with_lease(
        *prompt.last().expect("chatml prompt is non-empty"),
        prompt.len(),
        peer_lease,
    );

    let prompt_pos = prompt.len();
    let (mut reference_inputs, mut reference_rng, reference_receipts) =
        seed_ep_batch(&mut batch, &mut mesh, &prompt);
    for lane in 0..2 {
        events.push(batch_trace_event(
            &batch,
            if lane == 0 {
                "reference-seed-lane-0"
            } else {
                "reference-seed-lane-1"
            },
            0,
            1u64 << lane,
            [reference_inputs[lane], 0],
            [prompt_pos, prompt_pos],
            [Some(reference_inputs[lane]), None],
            [Some(reference_rng[lane]), None],
            Some(&reference_receipts[lane]),
            [Some(prompt_pos), Some(prompt_pos)],
            None,
        ));
    }
    let mut reference_inputs_by_tick = Vec::with_capacity(8);
    let mut reference_samples = Vec::with_capacity(8);
    for tick in 0..8 {
        let inputs = reference_inputs;
        let positions = [prompt_pos + tick, prompt_pos + tick];
        reference_inputs_by_tick.push(inputs);
        let receipt = batch
            .forward_tick(
                &mut mesh.gpus,
                &mesh.weights,
                &mesh.config,
                0b11,
                &inputs,
                &positions,
            )
            .unwrap_or_else(|error| panic!("{TEST}: reference forward tick {tick}: {error:?}"));
        observe_batch_rank_outputs(
            &batch,
            &mut mesh.gpus,
            &format!("reference active tick {tick}"),
            &mut failures,
        );
        let mut sampled = [0u32; 2];
        let mut next_rng = [0u32; 2];
        for lane in 0..2 {
            (sampled[lane], next_rng[lane]) = batch
                .sample_lane(
                    &mut mesh.gpus,
                    &mesh.config,
                    lane,
                    0.0,
                    1.0,
                    None,
                    reference_rng[lane],
                )
                .unwrap_or_else(|error| {
                    panic!("{TEST}: reference sample {tick}/{lane}: {error:?}")
                });
        }
        reference_samples.push(sampled);
        events.push(batch_trace_event(
            &batch,
            "reference-two-active",
            tick,
            0b11,
            inputs,
            positions,
            [Some(sampled[0]), Some(sampled[1])],
            [Some(next_rng[0]), Some(next_rng[1])],
            Some(&receipt),
            [Some(prompt_pos + tick + 1), Some(prompt_pos + tick + 1)],
            None,
        ));
        reference_inputs = sampled;
        reference_rng = next_rng;
    }

    // Refusal preflight: neither an empty mask nor a wrong active-lane
    // position may advance epoch or poison state. A valid retry immediately
    // follows both refusals, proving their lane positions also stayed intact.
    let (check_inputs, check_rng, check_receipts) = seed_ep_batch(&mut batch, &mut mesh, &prompt);
    for lane in 0..2 {
        events.push(batch_trace_event(
            &batch,
            if lane == 0 {
                "refusal-check-seed-lane-0"
            } else {
                "refusal-check-seed-lane-1"
            },
            0,
            1u64 << lane,
            [check_inputs[lane], 0],
            [prompt_pos, prompt_pos],
            [Some(check_inputs[lane]), None],
            [Some(check_rng[lane]), None],
            Some(&check_receipts[lane]),
            [Some(prompt_pos), Some(prompt_pos)],
            None,
        ));
    }
    let before_empty_epoch = batch.epoch();
    let before_empty_poison = batch.poison_mask();
    let empty_error = batch
        .forward_tick(
            &mut mesh.gpus,
            &mesh.weights,
            &mesh.config,
            0,
            &check_inputs,
            &[prompt_pos, prompt_pos],
        )
        .err();
    if empty_error.is_none() {
        failures.push("empty active mask was accepted".to_string());
    }
    if batch.epoch() != before_empty_epoch || batch.poison_mask() != before_empty_poison {
        failures.push("empty active mask mutated epoch or poison state".to_string());
    }
    events.push(batch_trace_event(
        &batch,
        "empty-mask-refusal",
        0,
        0,
        check_inputs,
        [prompt_pos, prompt_pos],
        [None, None],
        [None, None],
        None,
        [Some(prompt_pos), Some(prompt_pos)],
        empty_error.map(|error| format!("{error:?}")),
    ));

    let before_wrong_epoch = batch.epoch();
    let before_wrong_poison = batch.poison_mask();
    let wrong_error = batch
        .forward_tick(
            &mut mesh.gpus,
            &mesh.weights,
            &mesh.config,
            0b11,
            &check_inputs,
            &[prompt_pos + 1, prompt_pos],
        )
        .err();
    if wrong_error.is_none() {
        failures.push("wrong active-lane position was accepted".to_string());
    }
    if batch.epoch() != before_wrong_epoch || batch.poison_mask() != before_wrong_poison {
        failures.push("wrong position mutated epoch or poison state".to_string());
    }
    events.push(batch_trace_event(
        &batch,
        "wrong-position-refusal",
        0,
        0b11,
        check_inputs,
        [prompt_pos + 1, prompt_pos],
        [None, None],
        [None, None],
        None,
        [Some(prompt_pos), Some(prompt_pos)],
        wrong_error.map(|error| format!("{error:?}")),
    ));

    let valid_receipt = batch
        .forward_tick(
            &mut mesh.gpus,
            &mesh.weights,
            &mesh.config,
            0b11,
            &check_inputs,
            &[prompt_pos, prompt_pos],
        )
        .unwrap_or_else(|error| panic!("{TEST}: valid retry failed: {error:?}"));
    observe_batch_rank_outputs(
        &batch,
        &mut mesh.gpus,
        "valid retry active tick",
        &mut failures,
    );
    let mut valid_sampled = [0u32; 2];
    let mut valid_rng = [0u32; 2];
    for lane in 0..2 {
        (valid_sampled[lane], valid_rng[lane]) = batch
            .sample_lane(
                &mut mesh.gpus,
                &mesh.config,
                lane,
                0.0,
                1.0,
                None,
                check_rng[lane],
            )
            .unwrap_or_else(|error| panic!("{TEST}: valid retry sample {lane}: {error:?}"));
    }
    events.push(batch_trace_event(
        &batch,
        "valid-retry",
        0,
        0b11,
        check_inputs,
        [prompt_pos, prompt_pos],
        [Some(valid_sampled[0]), Some(valid_sampled[1])],
        [Some(valid_rng[0]), Some(valid_rng[1])],
        Some(&valid_receipt),
        [Some(prompt_pos + 1), Some(prompt_pos + 1)],
        None,
    ));

    // Reseed before the inactive-lane probe. Lane one must retain the exact
    // expected position while lane zero takes one extra successful tick.
    let (replay_seed_inputs, replay_seed_rng, replay_seed_receipts) =
        seed_ep_batch(&mut batch, &mut mesh, &prompt);
    for lane in 0..2 {
        events.push(batch_trace_event(
            &batch,
            if lane == 0 {
                "inactive-probe-seed-lane-0"
            } else {
                "inactive-probe-seed-lane-1"
            },
            0,
            1u64 << lane,
            [replay_seed_inputs[lane], 0],
            [prompt_pos, prompt_pos],
            [Some(replay_seed_inputs[lane]), None],
            [Some(replay_seed_rng[lane]), None],
            Some(&replay_seed_receipts[lane]),
            [Some(prompt_pos), Some(prompt_pos)],
            None,
        ));
    }
    if replay_seed_inputs[1] != reference_inputs_by_tick[0][1] {
        failures.push("lane 1 initial sample changed after reseed".to_string());
    }
    let inactive_receipt = batch
        .forward_tick(
            &mut mesh.gpus,
            &mesh.weights,
            &mesh.config,
            0b01,
            &[replay_seed_inputs[0], 0],
            &[prompt_pos, prompt_pos],
        )
        .unwrap_or_else(|error| panic!("{TEST}: lane-0 inactive probe: {error:?}"));
    observe_batch_rank_outputs(
        &batch,
        &mut mesh.gpus,
        "inactive lane-0-only tick",
        &mut failures,
    );
    let (lane0_after_inactive, lane0_after_inactive_rng) = batch
        .sample_lane(
            &mut mesh.gpus,
            &mesh.config,
            0,
            0.0,
            1.0,
            None,
            replay_seed_rng[0],
        )
        .unwrap_or_else(|error| panic!("{TEST}: lane-0 inactive sample: {error:?}"));
    events.push(batch_trace_event(
        &batch,
        "lane-0-only-extra-tick",
        0,
        0b01,
        [replay_seed_inputs[0], 0],
        [prompt_pos, prompt_pos],
        [Some(lane0_after_inactive), None],
        [Some(lane0_after_inactive_rng), None],
        Some(&inactive_receipt),
        [Some(prompt_pos + 1), Some(prompt_pos)],
        None,
    ));

    let mut replay_rng = [lane0_after_inactive_rng, replay_seed_rng[1]];
    for tick in 0..8 {
        let inputs = reference_inputs_by_tick[tick];
        let positions = [prompt_pos + 1 + tick, prompt_pos + tick];
        let receipt = batch
            .forward_tick(
                &mut mesh.gpus,
                &mesh.weights,
                &mesh.config,
                0b11,
                &inputs,
                &positions,
            )
            .unwrap_or_else(|error| panic!("{TEST}: replay forward tick {tick}: {error:?}"));
        observe_batch_rank_outputs(
            &batch,
            &mut mesh.gpus,
            &format!("replay active tick {tick}"),
            &mut failures,
        );
        let mut sampled = [0u32; 2];
        let mut next_rng = [0u32; 2];
        for lane in 0..2 {
            (sampled[lane], next_rng[lane]) = batch
                .sample_lane(
                    &mut mesh.gpus,
                    &mesh.config,
                    lane,
                    0.0,
                    1.0,
                    None,
                    replay_rng[lane],
                )
                .unwrap_or_else(|error| panic!("{TEST}: replay sample {tick}/{lane}: {error:?}"));
        }
        if sampled[1] != reference_samples[tick][1] {
            failures.push(format!(
                "lane 1 sample changed at replay tick {tick}: {} != {}",
                sampled[1], reference_samples[tick][1]
            ));
        }
        if next_rng[1] != reference_rng[1] && tick + 1 == 8 {
            // `reference_rng` is the final reference state; only the final
            // replay draw is compared here because lane-one token equality
            // above is the behavior contract for every tick.
            failures.push(format!(
                "lane 1 final RNG changed: {} != {}",
                next_rng[1], reference_rng[1]
            ));
        }
        events.push(batch_trace_event(
            &batch,
            "replay-offset-lane-0",
            tick,
            0b11,
            inputs,
            positions,
            [Some(sampled[0]), Some(sampled[1])],
            [Some(next_rng[0]), Some(next_rng[1])],
            Some(&receipt),
            [Some(prompt_pos + 2 + tick), Some(prompt_pos + 1 + tick)],
            None,
        ));
        replay_rng = next_rng;
    }

    if let Err(error) = batch.free_gpu(&mut mesh.gpus) {
        failures.push(format!("free EP batch state: {error:?}"));
    }
    mesh.free();
    write_batch_trace(load_cfg, events);
    assert!(
        failures.is_empty(),
        "{TEST} behavior failures:\n{}",
        failures.join("\n")
    );
}
