// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab oracle for the MoE GDN fused preamble
//! (`HIPFIRE_GDN_MOE_PRE_FUSED`, default ON; `0`/`false` kills it): the MoE sibling's decomposed
//! 4-launch preamble against the single `gdn_pre_batched_gfx1201` launch.
//!
//! Shape (A3B): 16 key heads, 32 value heads (ratio 2), head_dim 128, conv
//! kernel width 4 (3-tap ring), `k_dim = 2048`, `v_dim = 4096`,
//! `qkv_dim = 8192`, `rows` in {64, 512, 4096}.
//!
//! Chunk start offsets (the launchers take no position argument; the offset
//! only decides which conv ring a chunk starts from and which input slice it
//! sees):
//! - `start_pos = 0`: first chunk of a sequence, conv ring all zero;
//! - `start_pos = 4096`: a later chunk, conv ring carries nonzero state.
//!
//! Arms, per (rows, start_pos):
//! - A (decomposed, MoE sibling order): `fused_sigmoid_alpha_gate_f32_batched`,
//!   `conv1d_silu_split_f32_n`, `fused_qk_l2_norm_scale_f32_batched` (in place
//!   on the raw Q/K planes), `repeat_interleave_qk_f32_batched` into the
//!   `[rows, 32, 128]` Q/K buffers;
//! - B (fused): one `gdn_pre_batched_gfx1201` launch.
//!
//! Compared as raw bytes (f32 words by bits): interleaved Q, interleaved K,
//! V, alpha, beta, updated conv state. Every buffer reports mismatch count,
//! first mismatching word, max abs diff and max ULP distance (zeros when
//! identical). Q, K, V, alpha and beta additionally report PAIRED PER-ROW
//! metrics (row r of the decomposed arm vs row r of the fused arm: rows
//! differing, first/last differing row, worst row by abs and by ULP, and
//! p50/p90/p99/max of the per-row max abs / max ULP).
//!
//! Assertions (these alone decide pass/fail): alpha, beta and the conv ring
//! must be byte-identical (same gate formula; the ring copies the raw last
//! three qkv rows). Q, K and V are REPORTED ONLY, not asserted: the fused
//! conv starts its fma tree as `fmaf(w3, x, w2 * s0)` where the legacy
//! certified kernel uses `fmaf(w2, s0, w3 * x)`, so identity is not claimed.
//! A route default change still needs a paired KLD. `q_raw`/`k_raw` are NOT
//! compared: arm A normalises them in place while the fused kernel keeps the
//! raw conv outputs there, by design. Also asserted: inputs (qkv, conv
//! weight, dt_bias, a_log) unchanged, outputs finite and written (outputs are
//! poisoned with NaN words before each arm), the conv ring actually advanced,
//! and a second fused launch from identical state reproducing the first.
//!
//! Timing: hipEvent pair around the launches only, 10 warmups + 50 reps,
//! median / p10 / p90 in microseconds per arm. Alpha, beta and conv state
//! are restored from pristine device copies (device-to-device, bracketed by
//! device synchronisation) OUTSIDE every event pair; q/k/v are fully
//! overwritten each launch so need no reset. Arms alternate order per rep so
//! neither systematically follows the other's cache footprint.
//!
//! stdout: one JSON object per case (`"record":"case"`) followed by one
//! `"record":"summary"` object; progress goes to stderr. Exit 1 only if an
//! asserted check fails (alpha/beta/conv-ring bytes, sanity), 2 on usage
//! error. Q/K/V differences never fail the run. Non-gfx1201 prints a skipped
//! summary and exits 0. This oracle calls the launchers directly, so it is
//! independent of the runtime flag state; a pass is evidence for the arm
//! comparison on the machine that ran it, nothing more.
//!
//! `deltanet` is not a default feature and `lab` does not imply it; the
//! launchers are `deltanet`-gated, so the example is registered with
//! `required-features = ["lab", "deltanet"]`:
//!
//! ```text
//! cargo run --release -p rdna-compute --features lab,deltanet \
//!     --example gdn_moe_pre_fused_oracle -- [--plan] [--rows 64,512,4096]
//! ```
//! `--plan` prints the matrix JSON without initializing the GPU.

use rdna_compute::{DType, Gpu, GpuTensor};

const NK: usize = 16;
const NV: usize = 32;
const RATIO: usize = NV / NK;
const HD: usize = 128;
const CONV_WIDTH: usize = 4;
const K_DIM: usize = NK * HD;
const V_DIM: usize = NV * HD;
const QKV_DIM: usize = 2 * K_DIM + V_DIM;
const ROWS: [usize; 3] = [64, 512, 4096];
const START_POS: [usize; 2] = [0, 4096];
const WARMUPS: usize = 10;
const REPS: usize = 50;
const EPS: f32 = 1e-6;
const POISON_BYTE: u8 = 0xFF;

struct Args {
    plan: bool,
    rows: Vec<usize>,
}

fn parse_args() -> Args {
    let mut args = Args {
        plan: false,
        rows: ROWS.to_vec(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--plan" => args.plan = true,
            "--rows" => {
                let list = it.next().unwrap_or_default();
                let parsed: Option<Vec<usize>> = list
                    .split(',')
                    .map(|s| s.trim().parse::<usize>().ok().filter(|r| ROWS.contains(r)))
                    .collect();
                match parsed {
                    Some(v) if !v.is_empty() => args.rows = v,
                    _ => {
                        eprintln!("--rows expects a comma list drawn from {ROWS:?}, got {list:?}");
                        std::process::exit(2);
                    }
                }
            }
            other => {
                eprintln!("unknown argument {other:?}; usage: [--plan] [--rows 64,512,4096]");
                std::process::exit(2);
            }
        }
    }
    args
}

fn plan_json(args: &Args) -> serde_json::Value {
    let cases: Vec<_> = args
        .rows
        .iter()
        .flat_map(|&rows| {
            START_POS.iter().map(move |&sp| {
                serde_json::json!({
                    "rows": rows,
                    "start_pos": sp,
                    "conv_state": if sp == 0 { "zero" } else { "nonzero" },
                })
            })
        })
        .collect();
    serde_json::json!({
        "oracle": "gdn_moe_pre_fused_oracle",
        "record": "plan",
        "plan_only": true,
        "n_key_heads": NK,
        "n_value_heads": NV,
        "qk_head_ratio": RATIO,
        "head_dim": HD,
        "conv_kernel": CONV_WIDTH,
        "timing": { "warmups": WARMUPS, "reps": REPS, "stats": ["median", "p10", "p90"], "clock": "hipEvent", "unit": "us" },
        "arms": {
            "decomposed": "fused_sigmoid_alpha_gate_f32_batched + conv1d_silu_split_f32_n + fused_qk_l2_norm_scale_f32_batched + repeat_interleave_qk_f32_batched",
            "fused": "gdn_pre_batched_gfx1201",
        },
        "compare": ["q (interleaved)", "k (interleaved)", "v", "alpha", "beta", "conv_state"],
        "asserted_byte_equal": ["alpha", "beta", "conv_state"],
        "reported_only": ["q", "k", "v"],
        "per_row_paired": ["q", "k", "v", "alpha", "beta"],
        "not_compared": ["q_raw", "k_raw"],
        "cases": cases,
    })
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    fn sym(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn fill(rng: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.sym() * scale).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn words(b: &[u8]) -> impl Iterator<Item = u32> + '_ {
    b.chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
}

struct Host {
    qkv: Vec<f32>,
    conv_w: Vec<f32>,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    conv_state: Vec<f32>,
}

fn make_host(rows: usize, start_pos: usize) -> Host {
    let mut rng = Rng(0xA3B0_9F00_0000_0000 ^ ((rows as u64) << 20) ^ start_pos as u64);
    let qkv = fill(&mut rng, rows * QKV_DIM, 0.6);
    let conv_w = fill(&mut rng, QKV_DIM * CONV_WIDTH, 0.25);
    let dt_bias = fill(&mut rng, NV, 1.0);
    let a_log = fill(&mut rng, NV, 0.5);
    let alpha = fill(&mut rng, rows * NV, 2.0);
    let beta = fill(&mut rng, rows * NV, 2.0);
    let conv_state = if start_pos == 0 {
        vec![0.0f32; QKV_DIM * (CONV_WIDTH - 1)]
    } else {
        fill(&mut rng, QKV_DIM * (CONV_WIDTH - 1), 0.2)
    };
    Host {
        qkv,
        conv_w,
        dt_bias,
        a_log,
        alpha,
        beta,
        conv_state,
    }
}

/// Per-arm mutable outputs. Conv ring, alpha and beta are in/out.
struct Arm {
    beta: GpuTensor,
    alpha: GpuTensor,
    q_raw: GpuTensor,
    k_raw: GpuTensor,
    v: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    conv_state: GpuTensor,
}

impl Arm {
    fn alloc(gpu: &mut Gpu, rows: usize) -> Self {
        let mut a = |n: usize, name: &str| gpu.alloc_tensor(&[n], DType::F32).expect(name);
        Self {
            beta: a(rows * NV, "beta"),
            alpha: a(rows * NV, "alpha"),
            q_raw: a(rows * K_DIM, "q_raw"),
            k_raw: a(rows * K_DIM, "k_raw"),
            v: a(rows * V_DIM, "v"),
            q: a(rows * V_DIM, "q"),
            k: a(rows * V_DIM, "k"),
            conv_state: a(QKV_DIM * (CONV_WIDTH - 1), "conv_state"),
        }
    }

    fn free(self, gpu: &mut Gpu) {
        for t in [
            self.beta,
            self.alpha,
            self.q_raw,
            self.k_raw,
            self.v,
            self.q,
            self.k,
            self.conv_state,
        ] {
            gpu.free_tensor(t).expect("free arm tensor");
        }
    }
}

/// Inputs shared by both arms plus pristine copies of the mutable state.
struct Shared {
    qkv: GpuTensor,
    conv_w: GpuTensor,
    dt_bias: GpuTensor,
    a_log: GpuTensor,
    alpha0: GpuTensor,
    beta0: GpuTensor,
    conv0: GpuTensor,
}

fn upload(gpu: &mut Gpu, v: &[f32]) -> GpuTensor {
    gpu.upload_f32(v, &[v.len()]).expect("upload")
}

fn poison_outputs(gpu: &Gpu, arm: &Arm) {
    for t in [&arm.q_raw, &arm.k_raw, &arm.v, &arm.q, &arm.k] {
        gpu.hip
            .memcpy_htod(&t.buf, &vec![POISON_BYTE; t.byte_size()])
            .expect("poison");
    }
}

/// Restore alpha / beta / conv ring from the pristine device copies. Always
/// called outside timed regions.
fn reset_state(gpu: &Gpu, arm: &Arm, s: &Shared) {
    gpu.bind_thread().expect("bind");
    gpu.hip.device_synchronize().expect("sync before reset");
    for (dst, src) in [
        (&arm.alpha, &s.alpha0),
        (&arm.beta, &s.beta0),
        (&arm.conv_state, &s.conv0),
    ] {
        gpu.hip
            .memcpy_dtod(&dst.buf, &src.buf, src.byte_size())
            .expect("reset dtod");
    }
    gpu.hip.device_synchronize().expect("sync after reset");
}

fn q_scale() -> f32 {
    1.0 / (HD as f32).sqrt()
}

fn run_decomposed(gpu: &mut Gpu, arm: &Arm, s: &Shared, rows: usize) {
    gpu.fused_sigmoid_alpha_gate_f32_batched(&arm.beta, &arm.alpha, &s.dt_bias, &s.a_log, NV, rows)
        .expect("sigmoid alpha");
    gpu.conv1d_silu_split_f32_n(
        &arm.q_raw,
        &arm.k_raw,
        &arm.v,
        &s.qkv,
        &s.conv_w,
        &arm.conv_state,
        K_DIM,
        V_DIM,
        rows,
    )
    .expect("conv split");
    gpu.fused_qk_l2_norm_scale_f32_batched(&arm.q_raw, &arm.k_raw, NK, HD, q_scale(), EPS, rows)
        .expect("norm scale");
    gpu.repeat_interleave_qk_f32_batched(
        &arm.q_raw, &arm.k_raw, &arm.q, &arm.k, NK, RATIO, HD, rows,
    )
    .expect("repeat");
}

fn run_fused(gpu: &mut Gpu, arm: &Arm, s: &Shared, rows: usize) {
    gpu.gdn_pre_batched_gfx1201(
        &arm.beta,
        &arm.alpha,
        &s.dt_bias,
        &s.a_log,
        &s.qkv,
        &s.conv_w,
        &arm.conv_state,
        &arm.q_raw,
        &arm.k_raw,
        &arm.v,
        &arm.q,
        &arm.k,
        NV,
        NK,
        RATIO,
        K_DIM,
        V_DIM,
        rows,
        q_scale(),
        EPS,
    )
    .expect("fused preamble");
}

struct Snap {
    q: Vec<u8>,
    k: Vec<u8>,
    v: Vec<u8>,
    alpha: Vec<u8>,
    beta: Vec<u8>,
    conv_state: Vec<u8>,
}

fn snapshot(gpu: &Gpu, arm: &Arm) -> Snap {
    gpu.hip.device_synchronize().expect("sync before download");
    let d = |t: &GpuTensor| gpu.download_raw_bytes(t).expect("download");
    Snap {
        q: d(&arm.q),
        k: d(&arm.k),
        v: d(&arm.v),
        alpha: d(&arm.alpha),
        beta: d(&arm.beta),
        conv_state: d(&arm.conv_state),
    }
}

impl Snap {
    fn named(&self) -> [(&'static str, &Vec<u8>); 6] {
        [
            ("q", &self.q),
            ("k", &self.k),
            ("v", &self.v),
            ("alpha", &self.alpha),
            ("beta", &self.beta),
            ("conv_state", &self.conv_state),
        ]
    }
}

/// Monotonic integer key for f32 bits: adjacent representable floats differ
/// by 1; +0 and -0 share key 0.
fn ulp_key(bits: u32) -> i64 {
    if bits >> 31 == 1 {
        -((bits & 0x7fff_ffff) as i64)
    } else {
        bits as i64
    }
}

/// Bitwise comparison of two f32 buffers. Abs / ULP statistics cover only
/// word pairs whose bits differ and are both finite (zero when identical);
/// non-finite differing pairs are counted separately.
fn compare_buf(a: &[u8], b: &[u8]) -> serde_json::Value {
    let mut mismatched = 0usize;
    let mut nonfinite_diff = 0usize;
    let mut first: Option<usize> = None;
    let mut max_abs = 0.0f64;
    let mut max_abs_idx = 0usize;
    let mut max_ulp = 0u64;
    let mut max_ulp_idx = 0usize;
    for (i, (x, y)) in words(a).zip(words(b)).enumerate() {
        if x == y {
            continue;
        }
        mismatched += 1;
        first.get_or_insert(i);
        let (fx, fy) = (f32::from_bits(x), f32::from_bits(y));
        if !fx.is_finite() || !fy.is_finite() {
            nonfinite_diff += 1;
            continue;
        }
        let abs = (fx as f64 - fy as f64).abs();
        if abs > max_abs {
            max_abs = abs;
            max_abs_idx = i;
        }
        let ulp = ulp_key(x).abs_diff(ulp_key(y));
        if ulp > max_ulp {
            max_ulp = ulp;
            max_ulp_idx = i;
        }
    }
    let len_equal = a.len() == b.len();
    let equal = len_equal && mismatched == 0;
    let mut j = serde_json::json!({
        "bytes": a.len(),
        "words": a.len() / 4,
        "equal": equal,
        "len_equal": len_equal,
        "mismatched_words": mismatched,
        "nonfinite_diff_words": nonfinite_diff,
        "max_abs": max_abs,
        "max_abs_word": max_abs_idx,
        "max_ulp": max_ulp,
        "max_ulp_word": max_ulp_idx,
    });
    if let Some(i) = first {
        let at = |buf: &[u8]| words(buf).nth(i).unwrap_or(0);
        j["first_mismatch_word"] = i.into();
        j["first_decomposed_bits"] = format!("0x{:08x}", at(a)).into();
        j["first_fused_bits"] = format!("0x{:08x}", at(b)).into();
    }
    j
}

fn pct_json(sorted: &[f64]) -> serde_json::Value {
    serde_json::json!({
        "p50": percentile(sorted, 0.50),
        "p90": percentile(sorted, 0.90),
        "p99": percentile(sorted, 0.99),
        "max": sorted.last().copied().unwrap_or(0.0),
    })
}

/// Paired per-row comparison: row r of `a` (decomposed) vs row r of `b`
/// (fused), rows `row_words` f32 words wide. Statistics follow `compare_buf`
/// (finite differing pairs only for abs / ULP).
fn per_row(a: &[u8], b: &[u8], row_words: usize) -> serde_json::Value {
    let row_bytes = row_words * 4;
    let mut abs_r: Vec<f64> = Vec::new();
    let mut ulp_r: Vec<f64> = Vec::new();
    let mut mism_r: Vec<f64> = Vec::new();
    let mut differing = 0usize;
    let mut nonfinite_rows = 0usize;
    let mut first_row: Option<usize> = None;
    let mut last_row = 0usize;
    let mut worst_abs = (0usize, 0.0f64);
    let mut worst_ulp = (0usize, 0u64);
    for (r, (ra, rb)) in a
        .chunks_exact(row_bytes)
        .zip(b.chunks_exact(row_bytes))
        .enumerate()
    {
        let (mut m_abs, mut m_ulp, mut mism, mut nf) = (0.0f64, 0u64, 0usize, false);
        for (x, y) in words(ra).zip(words(rb)) {
            if x == y {
                continue;
            }
            mism += 1;
            let (fx, fy) = (f32::from_bits(x), f32::from_bits(y));
            if !fx.is_finite() || !fy.is_finite() {
                nf = true;
                continue;
            }
            m_abs = m_abs.max((fx as f64 - fy as f64).abs());
            m_ulp = m_ulp.max(ulp_key(x).abs_diff(ulp_key(y)));
        }
        if mism > 0 {
            differing += 1;
            first_row.get_or_insert(r);
            last_row = r;
        }
        nonfinite_rows += nf as usize;
        if m_abs > worst_abs.1 {
            worst_abs = (r, m_abs);
        }
        if m_ulp > worst_ulp.1 {
            worst_ulp = (r, m_ulp);
        }
        abs_r.push(m_abs);
        ulp_r.push(m_ulp as f64);
        mism_r.push(mism as f64);
    }
    for v in [&mut abs_r, &mut ulp_r, &mut mism_r] {
        v.sort_by(|x, y| x.total_cmp(y));
    }
    serde_json::json!({
        "rows": abs_r.len(),
        "row_words": row_words,
        "rows_differing": differing,
        "rows_with_nonfinite_diff": nonfinite_rows,
        "first_differing_row": first_row,
        "last_differing_row": if differing > 0 { Some(last_row) } else { None },
        "worst_row_by_abs": { "row": worst_abs.0, "max_abs": worst_abs.1 },
        "worst_row_by_ulp": { "row": worst_ulp.0, "max_ulp": worst_ulp.1 },
        "per_row_max_abs": pct_json(&abs_r),
        "per_row_max_ulp": pct_json(&ulp_r),
        "per_row_mismatched_words": pct_json(&mism_r),
    })
}

/// Row width (f32 words) for buffers with a token axis; `None` for the ring.
fn row_words_of(name: &str) -> Option<usize> {
    match name {
        "q" | "k" | "v" => Some(V_DIM),
        "alpha" | "beta" => Some(NV),
        _ => None,
    }
}

const ASSERTED: [&str; 3] = ["alpha", "beta", "conv_state"];

fn nonfinite_words(b: &[u8]) -> usize {
    words(b).filter(|&w| !f32::from_bits(w).is_finite()).count()
}

fn nonzero_words(b: &[u8]) -> usize {
    words(b).filter(|&w| w & 0x7fff_ffff != 0).count()
}

fn event_us(gpu: &mut Gpu, launch: impl FnOnce(&mut Gpu)) -> f64 {
    let start = gpu.hip.event_create().expect("start event");
    let stop = gpu.hip.event_create().expect("stop event");
    gpu.hip
        .event_record(&start, gpu.active_stream.as_ref())
        .expect("record start");
    launch(gpu);
    gpu.hip
        .event_record(&stop, gpu.active_stream.as_ref())
        .expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("synchronize stop");
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed");
    gpu.hip.event_destroy(start).expect("destroy start");
    gpu.hip.event_destroy(stop).expect("destroy stop");
    ms as f64 * 1_000.0
}

/// Linear-interpolation-free nearest-rank percentile over a sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn stats(samples: &[f64]) -> serde_json::Value {
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    let median = (s[(n - 1) / 2] + s[n / 2]) * 0.5;
    serde_json::json!({
        "median_us": median,
        "p10_us": percentile(&s, 0.10),
        "p90_us": percentile(&s, 0.90),
        "min_us": s[0],
        "max_us": s[n - 1],
    })
}

fn run_case(gpu: &mut Gpu, rows: usize, start_pos: usize) -> (serde_json::Value, bool) {
    let h = make_host(rows, start_pos);
    let shared = Shared {
        qkv: upload(gpu, &h.qkv),
        conv_w: upload(gpu, &h.conv_w),
        dt_bias: upload(gpu, &h.dt_bias),
        a_log: upload(gpu, &h.a_log),
        alpha0: upload(gpu, &h.alpha),
        beta0: upload(gpu, &h.beta),
        conv0: upload(gpu, &h.conv_state),
    };
    let dec = Arm::alloc(gpu, rows);
    let fus = Arm::alloc(gpu, rows);
    let fus2 = Arm::alloc(gpu, rows);

    // Correctness: identical starting state, poisoned outputs, one pass each.
    for arm in [&dec, &fus, &fus2] {
        poison_outputs(gpu, arm);
        reset_state(gpu, arm, &shared);
    }
    run_decomposed(gpu, &dec, &shared, rows);
    run_fused(gpu, &fus, &shared, rows);
    run_fused(gpu, &fus2, &shared, rows);
    let a = snapshot(gpu, &dec);
    let b = snapshot(gpu, &fus);
    let b2 = snapshot(gpu, &fus2);

    let mut buffers = serde_json::Map::new();
    let mut asserted_equal = true;
    let mut qkv_equal = true;
    for ((name, x), (_, y)) in a.named().into_iter().zip(b.named()) {
        let mut c = compare_buf(x, y);
        if let Some(rw) = row_words_of(name) {
            c["per_row"] = per_row(x, y, rw);
        }
        let eq = c["equal"] == true;
        if ASSERTED.contains(&name) {
            asserted_equal &= eq;
        } else {
            qkv_equal &= eq;
        }
        c["asserted"] = ASSERTED.contains(&name).into();
        buffers.insert(name.to_string(), c);
    }
    let repeat_equal = b
        .named()
        .into_iter()
        .zip(b2.named())
        .all(|((_, x), (_, y))| x == y);

    let nonfinite: usize = a
        .named()
        .into_iter()
        .chain(b.named())
        .map(|(_, x)| nonfinite_words(x))
        .sum();
    let outputs_written = [&b.q, &b.k, &b.v, &b.alpha, &b.beta]
        .iter()
        .all(|x| nonzero_words(x) > 0);
    let ring_advanced = b.conv_state != f32_bytes(&h.conv_state);
    let inputs_unchanged = [
        (&shared.qkv, f32_bytes(&h.qkv)),
        (&shared.conv_w, f32_bytes(&h.conv_w)),
        (&shared.dt_bias, f32_bytes(&h.dt_bias)),
        (&shared.a_log, f32_bytes(&h.a_log)),
    ]
    .iter()
    .all(|(t, want)| &gpu.download_raw_bytes(t).expect("download input") == want);
    // Q/K/V differences are reported, never asserted (see the module docs).
    let pass = asserted_equal
        && repeat_equal
        && nonfinite == 0
        && outputs_written
        && ring_advanced
        && inputs_unchanged;

    // Timing: arms interleaved (order flipped each rep); state restored
    // outside the event pair.
    let mut dec_us = Vec::with_capacity(REPS);
    let mut fus_us = Vec::with_capacity(REPS);
    for iter in 0..WARMUPS + REPS {
        let (mut d_t, mut f_t) = (0.0, 0.0);
        for arm_is_dec in if iter % 2 == 0 {
            [true, false]
        } else {
            [false, true]
        } {
            if arm_is_dec {
                reset_state(gpu, &dec, &shared);
                d_t = event_us(gpu, |g| run_decomposed(g, &dec, &shared, rows));
            } else {
                reset_state(gpu, &fus, &shared);
                f_t = event_us(gpu, |g| run_fused(g, &fus, &shared, rows));
            }
        }
        if iter >= WARMUPS {
            dec_us.push(d_t);
            fus_us.push(f_t);
        }
    }
    let dec_stats = stats(&dec_us);
    let fus_stats = stats(&fus_us);
    let dec_med = dec_stats["median_us"].as_f64().unwrap_or(f64::NAN);
    let fus_med = fus_stats["median_us"].as_f64().unwrap_or(f64::NAN);

    dec.free(gpu);
    fus.free(gpu);
    fus2.free(gpu);
    for t in [
        shared.qkv,
        shared.conv_w,
        shared.dt_bias,
        shared.a_log,
        shared.alpha0,
        shared.beta0,
        shared.conv0,
    ] {
        gpu.free_tensor(t).expect("free shared");
    }

    let json = serde_json::json!({
        "oracle": "gdn_moe_pre_fused_oracle",
        "record": "case",
        "rows": rows,
        "start_pos": start_pos,
        "conv_state": if start_pos == 0 { "zero" } else { "nonzero" },
        "pass": pass,
        "asserted_byte_equal": asserted_equal,
        "qkv_byte_equal": qkv_equal,
        "buffers": buffers,
        "fused_repeat_equal": repeat_equal,
        "sanity": {
            "nonfinite_words": nonfinite,
            "outputs_written": outputs_written,
            "conv_ring_advanced": ring_advanced,
            "inputs_unchanged": inputs_unchanged,
        },
        "timing": {
            "warmups": WARMUPS,
            "reps": REPS,
            "decomposed": dec_stats,
            "fused": fus_stats,
            "speedup_x": dec_med / fus_med,
            "saved_us": dec_med - fus_med,
            "decomposed_samples_us": dec_us,
            "fused_samples_us": fus_us,
        },
    });
    (json, pass)
}

fn main() {
    let args = parse_args();
    if args.plan {
        println!("{}", plan_json(&args));
        return;
    }
    let mut gpu = Gpu::init().expect("Gpu::init");
    gpu.bind_thread().expect("bind");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP: gdn_moe_pre_fused_oracle requires exact gfx1201");
        println!(
            "{}",
            serde_json::json!({
                "oracle": "gdn_moe_pre_fused_oracle",
                "record": "summary",
                "skipped": true,
                "arch": gpu.arch,
                "pass": serde_json::Value::Null,
            })
        );
        return;
    }

    let mut brief = Vec::new();
    let mut all_pass = true;
    let mut qkv_all_equal = true;
    let mut qkv_max_abs = 0.0f64;
    let mut qkv_max_ulp = 0u64;
    let mut asserted_max_abs = 0.0f64;
    let mut asserted_max_ulp = 0u64;
    for &rows in &args.rows {
        for &sp in &START_POS {
            eprintln!("case rows={rows} start_pos={sp}");
            let (json, pass) = run_case(&mut gpu, rows, sp);
            all_pass &= pass;
            qkv_all_equal &= json["qkv_byte_equal"] == true;
            for (name, c) in json["buffers"].as_object().into_iter().flatten() {
                let abs = c["max_abs"].as_f64().unwrap_or(0.0);
                let ulp = c["max_ulp"].as_u64().unwrap_or(0);
                if ASSERTED.contains(&name.as_str()) {
                    asserted_max_abs = asserted_max_abs.max(abs);
                    asserted_max_ulp = asserted_max_ulp.max(ulp);
                } else {
                    qkv_max_abs = qkv_max_abs.max(abs);
                    qkv_max_ulp = qkv_max_ulp.max(ulp);
                }
            }
            let rows_differing = |n: &str| json["buffers"][n]["per_row"]["rows_differing"].clone();
            brief.push(serde_json::json!({
                "rows": rows,
                "start_pos": sp,
                "pass": pass,
                "asserted_byte_equal": json["asserted_byte_equal"],
                "qkv_byte_equal": json["qkv_byte_equal"],
                "q_rows_differing": rows_differing("q"),
                "k_rows_differing": rows_differing("k"),
                "v_rows_differing": rows_differing("v"),
                "q_max_ulp": json["buffers"]["q"]["max_ulp"],
                "k_max_ulp": json["buffers"]["k"]["max_ulp"],
                "v_max_ulp": json["buffers"]["v"]["max_ulp"],
                "decomposed_median_us": json["timing"]["decomposed"]["median_us"],
                "fused_median_us": json["timing"]["fused"]["median_us"],
                "speedup_x": json["timing"]["speedup_x"],
            }));
            println!("{json}");
        }
    }

    let summary = serde_json::json!({
        "oracle": "gdn_moe_pre_fused_oracle",
        "record": "summary",
        "skipped": false,
        "arch": gpu.arch,
        "n_key_heads": NK,
        "n_value_heads": NV,
        "qk_head_ratio": RATIO,
        "head_dim": HD,
        "conv_kernel": CONV_WIDTH,
        "timing": { "warmups": WARMUPS, "reps": REPS, "stats": ["median", "p10", "p90"], "clock": "hipEvent", "unit": "us" },
        "cases": brief.len(),
        "pass": all_pass,
        "pass_means": "alpha/beta/conv_state byte-identical + sanity checks; q/k/v reported only",
        "qkv_all_byte_equal": qkv_all_equal,
        "qkv_max_abs": qkv_max_abs,
        "qkv_max_ulp": qkv_max_ulp,
        "asserted_max_abs": asserted_max_abs,
        "asserted_max_ulp": asserted_max_ulp,
        "per_case": brief,
    });
    println!("{summary}");
    if !all_pass {
        std::process::exit(1);
    }
}
