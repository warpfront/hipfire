// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MQ6G256V2 prefill GEMM microbench + correctness gate: the exact-INT8 route
//! (gfx1201 and gfx1151) and the native-FP8 route (exact gfx1201 only);
//! developer example.
//!
//! Per (shape, N) the timed arms are
//!   * `inc`   `Gpu::rotate_x_mq_batched_f16` (shared F16 rotation) then
//!     `Gpu::gemm_mq6g256v2_xf16` (the incumbent PM/X-LDS route of the Qwen4
//!     forward on either arch; `n/a` when `gemm_mq6g256v2_xf16_applies` is
//!     false; the launched kernel name is recorded in `--json`),
//!   * `i8`    `Gpu::mq6_i8_prepare_x` (FWHT into F32 scratch + A8
//!     `block_i8_128` quantization) then `Gpu::gemm_mq6g256v2_i8_prequant`,
//!   * `i8pm`  the same prep then the certified-builder (PeaceMaker) twin of the
//!     i8 GEMM (`HIPFIRE_MQ6_I8_PM=1`, the product default: `gemm_mq6g256v2_mmq_i8_pm_{set,set_bf16}_gfx1201`;
//!     gfx1201 only; the `i8` arm is pinned to hipcc so the pair is always HIP vs PM),
//!   * `fp8_<arm>[_db]` `Gpu::mq6_fp8_prepare_x` then
//!     `Gpu::gemm_mq6g256v2_fp8_prepared` (gfx1201 only).
//! Times are GPU event times of one launch (median of `--iters`, after
//! `--warmup` launches); TFLOP/s = 2*M*K*N / time. Achieved GB/s = bytes / time
//! with GEMM bytes = weights `m*(k/256)*200` + activations (inc `n*k*2`; i8
//! `n*(k/128)*136`; fp8 `n*k + n*(k/128)*4 + n*4`) + output `n*m*4`; prep GB/s:
//! inc read `n*k*4` + write `n*k*2`; i8 rotate read/write `n*k*8` + quantizer
//! read `n*k*4` + write blocks; fp8 read `n*k*4` + write planes. `vs_inc` is
//! the incumbent prep+gemm time over the arm's.
//!
//! The incumbent is admitted like the Qwen4 forward does: `Gpu::qwen4_scope` is
//! set and `HIPFIRE_QWEN4_MQ6_X4_GFX1201` defaults to `1` for this process (set
//! it to `0` to see the memset+residual route skipped as `n/a` on gfx1201).
//!
//! Correctness (unless `--no-check`), for N in {512, 4000} and every shape
//! (M = 48 and the non-multiple-of-128 N exercise the OOB paths); 32 sampled
//! tokens (first 8, last 8, 16 seeded random) x all M rows are checked against
//! an f64 CPU reference with exactly decoded weights; the output is
//! NaN-prefilled (an unwritten element fails):
//!   * i8: the `block_i8_128` blocks the producer emitted are read back
//!     (`d * qs`, nibble-pair order undone); passes iff every element has
//!     `|y - ref| <= 1e-4 * sum_k |w_k d qx_k| + 1e-30` and the whole output is
//!     finite, and the BF16 entry equals bf16_rne(0.0 + F32 entry) bit for bit
//!     over the whole output; rel-L2(incumbent F32, i8) is reported;
//!   * fp8: the E4M3 bytes and per-row scales the producer emitted are read
//!     back; SPLIT passes iff every element has
//!     `|y - ref| <= 1e-4 * a_t * sum_k |w_k x8_k| + 1e-30` (the weight side is
//!     bit-exact, so only f32 summation noise remains) and the whole output is
//!     finite; ROUND reports rel-L2 vs the exact reference (no pass bar);
//!     rel-L2(incumbent F32, SPLIT) is reported (expected ~1e-2: E4M3
//!     activations vs F16); on (10240, 2560) the SPLIT BF16 variant must equal
//!     bf16_rne(0.0 + SPLIT F32) bit for bit.
//!   * i8 producer: for every shape at N in {512, 4000} the blocks of the timed
//!     one-pass producer (`mq6_i8_prepare_x` = `rotate_x_mq_i8_batched`) must equal,
//!     byte for byte over the whole `[K/128][N]` block array, the former two-pass
//!     chain (`rotate_x_mq_batched` into an F32 scratch, then `ensure_int8_mmq_x`);
//!     the producer's launched kernel name is printed and recorded in `--json`;
//!   * i8pm: for every shape at N in {512, 4000} the PM twin's F32 and BF16
//!     outputs (NaN-prefilled, same producer blocks) must equal the hipcc
//!     kernel's byte for byte over the WHOLE output, and the kernel the PM
//!     launch recorded must be the `_pm_` entry (a silent hipcc fallback
//!     fails); both launched kernel names are printed and recorded in `--json`.
//! Exit status 1 on any i8, i8 producer, i8pm, SPLIT or BF16 failure.
//!
//! Run: `cargo run --release -p rdna-compute --features lab --example
//! mq6_fp8_gemm_bench` (gated behind `lab`, like the other developer probes,
//! so the default build does not pay for it).
//!
//! CLI: `[--shapes fn] [--shape M,K]...` (a `--shape` replaces the FN list)
//! `[--n 64,128,...] [--iters 20] [--warmup 3] [--arm i8|i8pm|split|round|both|all]`
//! (default: `i8`, `i8pm`, `split` and `round` on gfx1201; `both` = split+round,
//! `all` = i8+i8pm+split+round; FP8 arms are refused on gfx1151)
//! `[--db 0|1|both]` (default both: every FP8 (arm, db) combination is timed as
//! `fp8_<arm>` / `fp8_<arm>_db`, and the SPLIT exactness + BF16
//! checks run for the DB variant too) `[--json PATH] [--no-check] [--seed N]`.

use half::f16;
use rdna_compute::fp8_e4m3::{bf16_rne, e4m3fn_to_f32};
use rdna_compute::{DType, Gpu, GpuTensor, Mq6Fp8Arm};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};
use std::ffi::c_void;
use std::sync::Arc;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const GROUP_BYTES: usize = 200;
/// Flash-Next trunk MQ6G256V2 shapes (M rows, K).
const FN_SHAPES: [(usize, usize); 7] = [
    (10240, 2560),
    (6144, 2560),
    (48, 2560),
    (12288, 2560),
    (512, 2560),
    (640, 2560),
    (2560, 6144),
];
const DEFAULT_NS: [usize; 8] = [64, 128, 256, 512, 1024, 2048, 4096, 8192];
const CHECK_NS: [usize; 2] = [512, 4000];
const BF16_SHAPE: (usize, usize) = (10240, 2560);

// ---------------------------------------------------------------- host math

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// Standard normal (Box-Muller).
    fn normal(&mut self) -> f64 {
        let u1 = ((self.next() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// `m` rows x `k/256` MQ6G256V2 groups in the exact byte format: per half
/// `s = f16(1e-4 + |N(0,1)|*1e-3)`, `z = f16(-32 s + N(0,1) s)`, uniform 6-bit
/// codes (random payload bytes), header `[s0 z0 s1 z1]` F16 LE.
fn gen_weights(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed ^ 0x6d71_3600_0000_0001);
    let mut w = vec![0u8; m * (k / 256) * GROUP_BYTES];
    for g in w.chunks_exact_mut(GROUP_BYTES) {
        for h in 0..2 {
            let s = f16::from_f32((1e-4 + r.normal().abs() * 1e-3) as f32);
            let sf = s.to_f32() as f64;
            let z = f16::from_f32((-32.0 * sf + r.normal() * sf) as f32);
            g[4 * h..4 * h + 2].copy_from_slice(&s.to_bits().to_le_bytes());
            g[4 * h + 2..4 * h + 4].copy_from_slice(&z.to_bits().to_le_bytes());
        }
        for c in g[8..].chunks_mut(8) {
            let v = r.next().to_le_bytes();
            c.copy_from_slice(&v[..c.len()]);
        }
    }
    w
}

/// `n x k` F32 activations ~ N(0,1) with 0.1% outliers x30.
fn gen_x(n: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed ^ 0x5858_0000_0000_0001);
    (0..n * k)
        .map(|_| {
            let x = r.normal();
            (if r.below(1000) == 0 { x * 30.0 } else { x }) as f32
        })
        .collect()
}

/// Exact f64 decode of one weight row (`gpr` groups of 200 B) into `out[k]`.
fn decode_row(wrow: &[u8], gpr: usize, out: &mut [f64]) {
    for g in 0..gpr {
        let gb = &wrow[g * GROUP_BYTES..(g + 1) * GROUP_BYTES];
        let hdr = [
            u32::from_le_bytes([gb[0], gb[1], gb[2], gb[3]]),
            u32::from_le_bytes([gb[4], gb[5], gb[6], gb[7]]),
        ];
        let payload = &gb[8..];
        for h in 0..2 {
            let s = f16::from_bits(hdr[h] as u16).to_f32() as f64;
            let z = f16::from_bits((hdr[h] >> 16) as u16).to_f32() as f64;
            for i in 0..128 {
                let c = h * 128 + i;
                let bit = 6 * c;
                let (byte, sh) = (bit / 8, bit % 8);
                let lo = payload[byte] as u32;
                let hi = if byte + 1 < payload.len() {
                    payload[byte + 1] as u32
                } else {
                    0
                };
                let q = (((lo | (hi << 8)) >> sh) & 63) as f64;
                out[g * 256 + c] = s * q + z;
            }
        }
    }
}

fn dec_e4m3(b: u8) -> f64 {
    e4m3fn_to_f32(b).unwrap_or(0.0) as f64
}

/// `(ref, mag)` laid out `[r * T + t]`: `ref = a_t * sum_k w_r(k) x8_t(k)`,
/// `mag = a_t * sum_k |w_r(k) x8_t(k)|`, f64.
fn reference(
    wbytes: &[u8],
    m: usize,
    k: usize,
    x8: &[Vec<f64>],
    a: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let t_n = x8.len();
    let gpr = k / 256;
    let mut refv = vec![0.0f64; m * t_n];
    let mut magv = vec![0.0f64; m * t_n];
    let nth = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
        .min(32);
    let rows_per = m.div_ceil(nth);
    std::thread::scope(|sc| {
        for (ci, (rc, mc)) in refv
            .chunks_mut(rows_per * t_n)
            .zip(magv.chunks_mut(rows_per * t_n))
            .enumerate()
        {
            sc.spawn(move || {
                let mut wrow = vec![0.0f64; k];
                let r0 = ci * rows_per;
                for lr in 0..rc.len() / t_n {
                    let r = r0 + lr;
                    decode_row(
                        &wbytes[r * gpr * GROUP_BYTES..(r + 1) * gpr * GROUP_BYTES],
                        gpr,
                        &mut wrow,
                    );
                    for t in 0..t_n {
                        let (mut s, mut sa) = (0.0f64, 0.0f64);
                        for (w, x) in wrow.iter().zip(&x8[t]) {
                            let p = w * x;
                            s += p;
                            sa += p.abs();
                        }
                        rc[lr * t_n + t] = a[t] * s;
                        mc[lr * t_n + t] = a[t] * sa;
                    }
                }
            });
        }
    });
    (refv, magv)
}

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        num += (x - y) * (x - y);
        den += y * y;
    }
    (num / den.max(1e-300)).sqrt()
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

// ------------------------------------------------------------------ device

fn dtoh(gpu: &Gpu, ptr: *mut c_void, bytes: usize) -> Res<Vec<u8>> {
    // SAFETY: `ptr` is a live scratch pointer of the producer with at least
    // `bytes` bytes; the borrowed wrapper does not own or free it.
    let buf = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, bytes) };
    let mut v = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut v, &buf)?;
    Ok(v)
}

/// One kernel variant: weight-plane arm and the double-buffered (DB) pipeline.
type Var = (Mq6Fp8Arm, bool);

fn set_var(gpu: &mut Gpu, (arm, db): Var) {
    let mut f = (*gpu.flags).clone();
    f.gfx12_mq6_fp8_arm = arm;
    f.gfx12_mq6_fp8_db = db;
    gpu.flags = Arc::new(f);
}

/// Pin `HIPFIRE_MQ6_I8_PM` for the launches that follow (`Some(true)`: PM twin,
/// `Some(false)`: hipcc), whatever the process environment says.
fn set_i8_pm(gpu: &mut Gpu, pm: bool) {
    let mut f = (*gpu.flags).clone();
    f.mq6_i8_pm = Some(pm);
    gpu.flags = Arc::new(f);
}

fn arm_name(arm: Mq6Fp8Arm) -> &'static str {
    match arm {
        Mq6Fp8Arm::Split => "split",
        Mq6Fp8Arm::Round => "round",
    }
}

/// `split` / `round`, `_db` appended for the double-buffered variant.
fn var_name((arm, db): Var) -> String {
    format!("{}{}", arm_name(arm), if db { "_db" } else { "" })
}

/// Median-able per-launch GPU times (ms) of `f`, `iters` samples after `warmup`.
fn time_ms(
    gpu: &mut Gpu,
    warmup: usize,
    iters: usize,
    mut f: impl FnMut(&mut Gpu) -> Res<()>,
) -> Res<Vec<f64>> {
    for _ in 0..warmup {
        f(gpu)?;
    }
    gpu.hip.device_synchronize()?;
    let start = gpu.hip.event_create()?;
    let stop = gpu.hip.event_create()?;
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        gpu.hip.event_record(&start, None)?;
        f(gpu)?;
        gpu.hip.event_record(&stop, None)?;
        gpu.hip.event_synchronize(&stop)?;
        samples.push(gpu.hip.event_elapsed_ms(&start, &stop)? as f64);
    }
    gpu.hip.event_destroy(start)?;
    gpu.hip.event_destroy(stop)?;
    Ok(samples)
}

fn tflops(m: usize, k: usize, n: usize, ms: f64) -> f64 {
    2.0 * m as f64 * k as f64 * n as f64 / (ms * 1e-3) / 1e12
}

fn nan_filled(gpu: &mut Gpu, elems: usize, dtype: DType) -> Res<GpuTensor> {
    let t = gpu.zeros(&[elems], dtype)?;
    gpu.hip.memset(&t.buf, 0xFF, t.byte_size())?;
    Ok(t)
}

// -------------------------------------------------------------- correctness

#[derive(Default)]
struct CheckOut {
    split_pass: bool,
    split_bad: usize,
    split_max_ratio: f64,
    split_nonfinite: usize,
    round_rel_l2: f64,
    inc_vs_split_rel_l2: Option<f64>,
    bf16: Option<(bool, usize)>,
    tokens: usize,
}

#[allow(clippy::too_many_arguments)]
fn check_case(
    gpu: &mut Gpu,
    w: &GpuTensor,
    wbytes: &[u8],
    xf: &[f32],
    m: usize,
    k: usize,
    n: usize,
    seed: u64,
    vars: &[Var],
) -> Res<CheckOut> {
    let xt = gpu.upload_f32(&xf[..n * k], &[n * k])?;
    let prepared = gpu.mq6_fp8_prepare_x(&xt, k, n)?;
    let x8_bytes = dtoh(gpu, prepared.x_fp8, n * k)?;
    let sc_bytes = dtoh(gpu, prepared.row_scales, n * 4)?;
    let scales: Vec<f64> = sc_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect();

    // Sampled tokens: first 8, last 8, 16 seeded random.
    let toks = sample_tokens(n, seed);
    let x8: Vec<Vec<f64>> = toks
        .iter()
        .map(|&t| {
            x8_bytes[t * k..(t + 1) * k]
                .iter()
                .map(|&b| dec_e4m3(b))
                .collect()
        })
        .collect();
    let a: Vec<f64> = toks.iter().map(|&t| scales[t]).collect();
    let (refv, magv) = reference(wbytes, m, k, &x8, &a);
    let t_n = toks.len();
    let mut out = CheckOut {
        tokens: t_n,
        ..Default::default()
    };

    let sample = |y: &[f32]| -> Vec<f64> {
        let mut s = Vec::with_capacity(m * t_n);
        for r in 0..m {
            for &t in &toks {
                s.push(y[t * m + r] as f64);
            }
        }
        s
    };

    let mut split_f32: HashMap<bool, Vec<f32>> = HashMap::new();
    let mut any_split = false;
    let mut split_pass = true;
    for &(arm, db) in vars {
        set_var(gpu, (arm, db));
        let y = nan_filled(gpu, n * m, DType::F32)?;
        gpu.gemm_mq6g256v2_fp8_prepared(w, &prepared, &y, m, k, n)?;
        let yh = gpu.download_f32(&y)?;
        gpu.free_tensor(y)?;
        let ys = sample(&yh);
        let vname = var_name((arm, db));
        match arm {
            Mq6Fp8Arm::Split => {
                let mut bad = 0usize;
                let mut max_ratio = 0.0f64;
                for i in 0..ys.len() {
                    let err = (ys[i] - refv[i]).abs();
                    let tol = 1e-4 * magv[i] + 1e-30;
                    max_ratio = max_ratio.max(err / tol);
                    // `!(<=)` also fails NaN.
                    if !(err <= tol) {
                        bad += 1;
                    }
                }
                let nonfinite = yh.iter().filter(|v| !v.is_finite()).count();
                println!(
                    "  variant {vname}: split bad={bad} nonfinite={nonfinite} max_err/tol={max_ratio:.3}"
                );
                out.split_bad += bad;
                out.split_nonfinite += nonfinite;
                out.split_max_ratio = out.split_max_ratio.max(max_ratio);
                split_pass &= bad == 0 && nonfinite == 0;
                any_split = true;
                split_f32.insert(db, yh);
            }
            Mq6Fp8Arm::Round => {
                let r = rel_l2(&ys, &refv);
                println!("  variant {vname}: round rel_l2={r:.3e}");
                out.round_rel_l2 = out.round_rel_l2.max(r);
            }
        }
    }
    out.split_pass = any_split && split_pass;

    // Incumbent F32 (sanity that rotation/layout agree).
    if let Some(ysplit) = split_f32.get(&false).or_else(|| split_f32.values().next()) {
        if gpu.gemm_mq6g256v2_xf16_applies(k, n) {
            let x16 = gpu.rotate_x_mq_batched_f16(&xt, k, n)?;
            let y = nan_filled(gpu, n * m, DType::F32)?;
            gpu.gemm_mq6g256v2_xf16(w, &x16, &y, m, k, n)?;
            let yh = gpu.download_f32(&y)?;
            gpu.free_tensor(y)?;
            out.inc_vs_split_rel_l2 = Some(rel_l2(&sample(&yh), &sample(ysplit)));
        }
    }

    // BF16 variant == bf16_rne(F32 SPLIT) bit for bit, per DB setting run.
    if (m, k) == BF16_SHAPE {
        let mut ran = false;
        let mut ok = true;
        let mut bad_total = 0usize;
        for (&db, ysplit) in &split_f32 {
            set_var(gpu, (Mq6Fp8Arm::Split, db));
            let y = nan_filled(gpu, n * m, DType::BF16)?;
            gpu.gemm_mq6g256v2_fp8_prepared(w, &prepared, &y, m, k, n)?;
            let raw = gpu.download_raw_bytes(&y)?;
            gpu.free_tensor(y)?;
            let mut mismatches = 0usize;
            for (i, c) in raw.chunks_exact(2).enumerate().take(n * m) {
                let got = u16::from_le_bytes([c[0], c[1]]);
                let v = 0.0f32 + ysplit[i];
                let want = if v.is_finite() {
                    bf16_rne(v)
                } else {
                    (v.to_bits() >> 16) as u16
                };
                if got != want {
                    mismatches += 1;
                }
            }
            println!(
                "  variant {}: bf16 mismatches={mismatches}",
                var_name((Mq6Fp8Arm::Split, db))
            );
            ran = true;
            ok &= mismatches == 0;
            bad_total += mismatches;
        }
        if ran {
            out.bf16 = Some((ok, bad_total));
        }
    }
    gpu.free_tensor(xt)?;
    Ok(out)
}

// ---------------------------------------------------------- bytes / reporting

/// One timed new-route arm of a (shape, N) point.
struct ArmRow {
    name: String,
    prep: Vec<f64>,
    prep_bytes: usize,
    gemm: Vec<f64>,
    /// Activation bytes the GEMM reads.
    act_bytes: usize,
    kernel: Option<String>,
}

/// GEMM traffic: weights `m*(k/256)*200` + activations + output `n*m*4`.
fn gemm_bytes(m: usize, k: usize, act_bytes: usize, n: usize) -> usize {
    m * (k / 256) * GROUP_BYTES + act_bytes + n * m * 4
}
/// Incumbent GEMM activations: F16 `n*k*2`.
fn inc_act_bytes(n: usize, k: usize) -> usize {
    n * k * 2
}
/// i8 GEMM activations: `block_i8_128` (136 B per 128 K) per token.
fn i8_act_bytes(n: usize, k: usize) -> usize {
    n * (k / 128) * 136
}
/// FP8 GEMM activations: E4M3 bytes + per-half sums + per-row scale.
fn fp8_act_bytes(n: usize, k: usize) -> usize {
    n * k + n * (k / 128) * 4 + n * 4
}
/// Incumbent prep: read F32 `n*k*4`, write F16 `n*k*2`.
fn inc_prep_bytes(n: usize, k: usize) -> usize {
    n * k * 4 + n * k * 2
}
/// i8 prep: FWHT rotate read + write F32 (`n*k*8`), quantizer read `n*k*4`, write blocks.
fn i8_prep_bytes(n: usize, k: usize) -> usize {
    n * k * 8 + n * k * 4 + i8_act_bytes(n, k)
}
/// FP8 prep: read F32 `n*k*4`, write the E4M3 planes (in-register producer).
fn fp8_prep_bytes(n: usize, k: usize) -> usize {
    n * k * 4 + fp8_act_bytes(n, k)
}
fn gbps(bytes: usize, ms: f64) -> f64 {
    bytes as f64 / (ms * 1e-3) / 1e9
}

/// First 8, last 8 and 16 seeded random tokens of `n`.
fn sample_tokens(n: usize, seed: u64) -> Vec<usize> {
    let mut toks: BTreeSet<usize> = (0..n.min(8)).chain(n.saturating_sub(8)..n).collect();
    let mut r = Rng(seed ^ 0x7467_6b00 ^ n as u64);
    for _ in 0..16 {
        toks.insert(r.below(n as u64) as usize);
    }
    toks.into_iter().collect()
}

/// Undo the A8 nibble-pair order: per 8-element group `g`, `qs[8g+0..3] =
/// q[8g+0,2,4,6]` and `qs[8g+4..7] = q[8g+1,3,5,7]`.
fn unpack_a8_qs(qs: &[u8], out: &mut [i32; 128]) {
    for g in 0..16 {
        for j in 0..4 {
            out[8 * g + 2 * j] = qs[8 * g + j] as i8 as i32;
            out[8 * g + 2 * j + 1] = qs[8 * g + 4 + j] as i8 as i32;
        }
    }
}

struct I8Check {
    pass: bool,
    bad: usize,
    nonfinite: usize,
    max_ratio: f64,
    inc_rel_l2: Option<f64>,
    /// F32-vs-BF16 entry mismatches over the whole output.
    bf16_bad: usize,
    tokens: usize,
}

/// Exactness gate of the i8 route: the `block_i8_128` blocks the producer
/// emitted are read back (`d * qs`), weights are decoded exactly on the CPU
/// (f64), and 32 sampled tokens x all M rows must satisfy
/// `|y - ref| <= 1e-4 * sum|w * d * qx| + 1e-30` (the integer core and the
/// fold are exact, so only f32 fold rounding remains); the NaN-prefilled output
/// must be fully finite, and the BF16 entry must equal bf16_rne(0.0 + F32
/// entry) bit for bit over the whole output.
#[allow(clippy::too_many_arguments)]
fn check_i8_case(
    gpu: &mut Gpu,
    w: &GpuTensor,
    wbytes: &[u8],
    xf: &[f32],
    m: usize,
    k: usize,
    n: usize,
    seed: u64,
) -> Res<I8Check> {
    let xt = gpu.upload_f32(&xf[..n * k], &[n * k])?;
    let xq = gpu.mq6_i8_prepare_x(&xt, k, n)?;
    let blocks = dtoh(gpu, xq, (k / 128) * n * 136)?;
    let toks = sample_tokens(n, seed);
    let t_n = toks.len();
    let xv: Vec<Vec<f64>> = toks
        .iter()
        .map(|&t| {
            let mut v = vec![0.0f64; k];
            let mut q = [0i32; 128];
            for kb in 0..k / 128 {
                let b = &blocks[(kb * n + t) * 136..(kb * n + t + 1) * 136];
                let d = f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64;
                unpack_a8_qs(&b[8..136], &mut q);
                for i in 0..128 {
                    v[kb * 128 + i] = d * q[i] as f64;
                }
            }
            v
        })
        .collect();
    let (refv, magv) = reference(wbytes, m, k, &xv, &vec![1.0; t_n]);
    let sample = |y: &[f32]| -> Vec<f64> {
        let mut s = Vec::with_capacity(m * t_n);
        for r in 0..m {
            for &t in &toks {
                s.push(y[t * m + r] as f64);
            }
        }
        s
    };

    let y = nan_filled(gpu, n * m, DType::F32)?;
    gpu.gemm_mq6g256v2_i8_prequant(w, xq, &y, m, k, n)?;
    let yh = gpu.download_f32(&y)?;
    gpu.free_tensor(y)?;
    let ys = sample(&yh);
    let (mut bad, mut max_ratio) = (0usize, 0.0f64);
    for i in 0..ys.len() {
        let err = (ys[i] - refv[i]).abs();
        let tol = 1e-4 * magv[i] + 1e-30;
        max_ratio = max_ratio.max(err / tol);
        // `!(<=)` also fails NaN.
        if !(err <= tol) {
            bad += 1;
        }
    }
    let nonfinite = yh.iter().filter(|v| !v.is_finite()).count();

    let y16 = nan_filled(gpu, n * m, DType::BF16)?;
    gpu.gemm_mq6g256v2_i8_prequant(w, xq, &y16, m, k, n)?;
    let raw = gpu.download_raw_bytes(&y16)?;
    gpu.free_tensor(y16)?;
    let mut bf16_bad = 0usize;
    for (i, c) in raw.chunks_exact(2).enumerate().take(n * m) {
        let got = u16::from_le_bytes([c[0], c[1]]);
        let v = 0.0f32 + yh[i];
        let want = if v.is_finite() {
            bf16_rne(v)
        } else {
            (v.to_bits() >> 16) as u16
        };
        if got != want {
            bf16_bad += 1;
        }
    }

    let inc_rel_l2 = if gpu.gemm_mq6g256v2_xf16_applies(k, n) {
        let x16 = gpu.rotate_x_mq_batched_f16(&xt, k, n)?;
        let yi = nan_filled(gpu, n * m, DType::F32)?;
        gpu.gemm_mq6g256v2_xf16(w, &x16, &yi, m, k, n)?;
        let ih = gpu.download_f32(&yi)?;
        gpu.free_tensor(yi)?;
        Some(rel_l2(&sample(&ih), &ys))
    } else {
        None
    };
    gpu.free_tensor(xt)?;
    Ok(I8Check {
        pass: bad == 0 && nonfinite == 0,
        bad,
        nonfinite,
        max_ratio,
        inc_rel_l2,
        bf16_bad,
        tokens: t_n,
    })
}

/// Result of the one-pass vs two-pass producer gate of one (K, N).
struct ProdCheck {
    /// Differing bytes over the whole `block_i8_128` array.
    diff_bytes: usize,
    bytes: usize,
    /// The one-pass launch ran the `mq_rotate_x_i8_*` entry.
    kernel: String,
}

/// The `block_i8_128` blocks of `mq6_i8_prepare_x` (one pass) against the two-pass
/// chain they replace, whole array, same F32 input.
fn check_producer_case(gpu: &mut Gpu, xf: &[f32], k: usize, n: usize) -> Res<ProdCheck> {
    let xt = gpu.upload_f32(&xf[..n * k], &[n * k])?;
    let bytes = (k / 128) * n * 136;
    // Both producers write the shared A8 scratch: copy the one-pass blocks out first.
    let one = gpu.mq6_i8_prepare_x(&xt, k, n)?;
    let kernel = gpu.last_launched_kernel().unwrap_or("?").to_owned();
    let one = dtoh(gpu, one, bytes)?;
    let x_rot = gpu.zeros(&[n * k], DType::F32)?;
    gpu.rotate_x_mq_batched(&xt, &x_rot, k, n)?;
    let two = gpu.ensure_int8_mmq_x(&x_rot, n, k)?;
    let two = dtoh(gpu, two, bytes)?;
    gpu.free_tensor(x_rot)?;
    gpu.free_tensor(xt)?;
    let diff_bytes = one.iter().zip(&two).filter(|(a, b)| a != b).count();
    Ok(ProdCheck {
        diff_bytes,
        bytes,
        kernel,
    })
}

/// Result of the PM-vs-hipcc bitwise gate of one (shape, N).
struct I8PmCheck {
    /// Differing bytes of the F32 entries over the whole output.
    f32_bad: usize,
    /// Differing bytes of the BF16 entries over the whole output.
    bf16_bad: usize,
    hip_kernels: [String; 2],
    pm_kernels: [String; 2],
    /// Both PM launches ran a `_pm_` entry (no silent hipcc fallback).
    pm_ran: bool,
}

impl I8PmCheck {
    fn pass(&self) -> bool {
        self.f32_bad == 0 && self.bf16_bad == 0 && self.pm_ran
    }
}

/// Bitwise equality of the PM twin and the hipcc kernel on the same producer
/// blocks, both output dtypes, NaN-prefilled outputs, the whole `n * m` extent.
fn check_i8pm_case(
    gpu: &mut Gpu,
    w: &GpuTensor,
    xf: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Res<I8PmCheck> {
    let xt = gpu.upload_f32(&xf[..n * k], &[n * k])?;
    let xq = gpu.mq6_i8_prepare_x(&xt, k, n)?;
    let mut raw: [Vec<Vec<u8>>; 2] = [Vec::new(), Vec::new()];
    let mut names: [Vec<String>; 2] = [Vec::new(), Vec::new()];
    for (side, pm) in [false, true].into_iter().enumerate() {
        set_i8_pm(gpu, pm);
        for dtype in [DType::F32, DType::BF16] {
            let y = nan_filled(gpu, n * m, dtype)?;
            gpu.gemm_mq6g256v2_i8_prequant(w, xq, &y, m, k, n)?;
            names[side].push(gpu.last_launched_kernel().unwrap_or("?").to_owned());
            raw[side].push(gpu.download_raw_bytes(&y)?);
            gpu.free_tensor(y)?;
        }
    }
    set_i8_pm(gpu, false);
    gpu.free_tensor(xt)?;
    let diff = |a: &[u8], b: &[u8]| -> usize {
        a.len().abs_diff(b.len()) + a.iter().zip(b).filter(|(x, y)| x != y).count()
    };
    let pm_ran = names[1].iter().all(|n| n.contains("_pm_"));
    Ok(I8PmCheck {
        f32_bad: diff(&raw[0][0], &raw[1][0]),
        bf16_bad: diff(&raw[0][1], &raw[1][1]),
        hip_kernels: [names[0][0].clone(), names[0][1].clone()],
        pm_kernels: [names[1][0].clone(), names[1][1].clone()],
        pm_ran,
    })
}

// ---------------------------------------------------------------------- cli

/// Arms selected by `--arm`: the exact-INT8 route and/or the FP8 weight-plane arms.
struct ArmSel {
    i8: bool,
    i8pm: bool,
    fp8: Vec<Mq6Fp8Arm>,
}

struct Args {
    shapes: Vec<(usize, usize)>,
    ns: Vec<usize>,
    iters: usize,
    warmup: usize,
    /// `None` = the arch default: `i8`, plus `split` and `round` on gfx1201.
    arms: Option<ArmSel>,
    dbs: Vec<bool>,
    json: Option<String>,
    check: bool,
    seed: u64,
}

impl Args {
    /// `(run the i8 arm, run the i8pm arm, FP8 weight-plane arms)` for `arch`;
    /// the FP8 and i8pm arms exist on gfx1201 only.
    fn resolve_arms(&self, arch: &str) -> Res<(bool, bool, Vec<Mq6Fp8Arm>)> {
        match &self.arms {
            None if arch == "gfx1201" => Ok((true, true, vec![Mq6Fp8Arm::Split, Mq6Fp8Arm::Round])),
            None => Ok((true, false, vec![])),
            Some(sel) if arch != "gfx1201" && (!sel.fp8.is_empty() || sel.i8pm) => {
                Err(format!("FP8 and i8pm arms need exact gfx1201, device is {arch}").into())
            }
            Some(sel) => Ok((sel.i8, sel.i8pm, sel.fp8.clone())),
        }
    }

    /// Every (arm, db) FP8 combination to time and check, arm-major.
    fn vars(&self, fp8: &[Mq6Fp8Arm]) -> Vec<Var> {
        fp8.iter()
            .flat_map(|&a| self.dbs.iter().map(move |&d| (a, d)))
            .collect()
    }
}

fn parse_args() -> Res<Args> {
    let mut a = Args {
        shapes: Vec::new(),
        ns: DEFAULT_NS.to_vec(),
        iters: 20,
        warmup: 3,
        arms: None,
        dbs: vec![false, true],
        json: None,
        check: true,
        seed: 1,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = |name: &str| -> Res<String> {
            it.next()
                .ok_or_else(|| format!("{name} needs a value").into())
        };
        match flag.as_str() {
            "--shapes" => match val("--shapes")?.as_str() {
                "fn" => {}
                other => return Err(format!("unknown --shapes set {other} (only `fn`)").into()),
            },
            "--shape" => {
                let v = val("--shape")?;
                let (m, k) = v.split_once(',').ok_or("--shape wants M,K")?;
                a.shapes.push((m.trim().parse()?, k.trim().parse()?));
            }
            "--n" => {
                a.ns = val("--n")?
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<Result<_, _>>()?;
            }
            "--iters" => a.iters = val("--iters")?.parse()?,
            "--warmup" => a.warmup = val("--warmup")?.parse()?,
            "--arm" => {
                a.arms = Some(match val("--arm")?.as_str() {
                    "i8" => ArmSel {
                        i8: true,
                        i8pm: false,
                        fp8: vec![],
                    },
                    "i8pm" => ArmSel {
                        i8: false,
                        i8pm: true,
                        fp8: vec![],
                    },
                    "split" => ArmSel {
                        i8: false,
                        i8pm: false,
                        fp8: vec![Mq6Fp8Arm::Split],
                    },
                    "round" => ArmSel {
                        i8: false,
                        i8pm: false,
                        fp8: vec![Mq6Fp8Arm::Round],
                    },
                    "both" => ArmSel {
                        i8: false,
                        i8pm: false,
                        fp8: vec![Mq6Fp8Arm::Split, Mq6Fp8Arm::Round],
                    },
                    "all" => ArmSel {
                        i8: true,
                        i8pm: true,
                        fp8: vec![Mq6Fp8Arm::Split, Mq6Fp8Arm::Round],
                    },
                    other => return Err(format!("unknown --arm {other}").into()),
                })
            }
            "--db" => {
                a.dbs = match val("--db")?.as_str() {
                    "0" => vec![false],
                    "1" => vec![true],
                    "both" => vec![false, true],
                    other => return Err(format!("unknown --db {other} (0|1|both)").into()),
                }
            }
            "--json" => a.json = Some(val("--json")?),
            "--no-check" => a.check = false,
            "--seed" => a.seed = val("--seed")?.parse()?,
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    if a.shapes.is_empty() {
        a.shapes = FN_SHAPES.to_vec();
    }
    for &(m, k) in &a.shapes {
        if m == 0 || k == 0 || k % 256 != 0 {
            return Err(format!("shape {m},{k}: need M>0 and K%256==0").into());
        }
    }
    if a.iters == 0 || a.ns.iter().any(|&n| n == 0) {
        return Err("--iters and every --n must be > 0".into());
    }
    Ok(a)
}

fn git_rev() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|| "unknown".into())
}

fn main() -> Res<()> {
    let args = parse_args()?;
    // The incumbent PM/X-LDS route is admitted as the Qwen4 forward does.
    if std::env::var_os("HIPFIRE_QWEN4_MQ6_X4_GFX1201").is_none() {
        std::env::set_var("HIPFIRE_QWEN4_MQ6_X4_GFX1201", "1");
    }
    let mut gpu = Gpu::init()?;
    if !matches!(gpu.arch.as_str(), "gfx1201" | "gfx1151") {
        return Err(format!(
            "mq6_fp8_gemm_bench needs exact gfx1201 or gfx1151, device is {}",
            gpu.arch
        )
        .into());
    }
    gpu.qwen4_scope = true;
    let (run_i8, run_i8pm, fp8_arms) = args.resolve_arms(&gpu.arch)?;
    let fp8_on = !fp8_arms.is_empty();

    // Largest N needed per K (activations are generated once, prefixes reused).
    let mut kmax: HashMap<usize, usize> = HashMap::new();
    for &(_, k) in &args.shapes {
        let e = kmax.entry(k).or_insert(0);
        *e = (*e).max(args.ns.iter().copied().max().unwrap_or(0));
        if args.check {
            *e = (*e).max(CHECK_NS.iter().copied().max().unwrap());
        }
    }
    let mut xcache: HashMap<usize, Vec<f32>> = HashMap::new();

    let vars = args.vars(&fp8_arms);
    let arm_names: Vec<String> = run_i8
        .then(|| "i8".to_owned())
        .into_iter()
        .chain(run_i8pm.then(|| "i8pm".to_owned()))
        .chain(vars.iter().map(|v| format!("fp8_{}", var_name(*v))))
        .collect();
    println!(
        "mq6_fp8_gemm_bench arch={} rev={} iters={} warmup={} arms={:?}",
        gpu.arch,
        git_rev(),
        args.iters,
        args.warmup,
        arm_names
    );
    println!(
        "{:>6}x{:<5} {:>5} | {:<16} | {:>9} {:>8} | {:>9} {:>7} {:>8} | {:>9} {:>7}",
        "M",
        "K",
        "N",
        "arm",
        "prep_ms",
        "prepGB/s",
        "gemm_ms",
        "TF/s",
        "GB/s",
        "total_ms",
        "vs_inc"
    );

    let mut runs: Vec<Value> = Vec::new();
    let mut checks: Vec<Value> = Vec::new();
    let mut failed = false;

    for &(m, k) in &args.shapes {
        let wbytes = gen_weights(m, k, args.seed ^ ((m as u64) << 24) ^ k as u64);
        let w = gpu.upload_raw(&wbytes, &[wbytes.len()])?;
        let xfull = xcache
            .entry(k)
            .or_insert_with(|| gen_x(kmax[&k], k, args.seed));

        for &n in &args.ns {
            let xt = gpu.upload_f32(&xfull[..n * k], &[n * k])?;
            let y_inc = gpu.zeros(&[n * m], DType::F32)?;
            let y_new = gpu.zeros(&[n * m], DType::F32)?;

            let inc_prep = time_ms(&mut gpu, args.warmup, args.iters, |g| {
                g.rotate_x_mq_batched_f16(&xt, k, n)?;
                Ok(())
            })?;
            let inc_ok = gpu.gemm_mq6g256v2_xf16_applies(k, n);
            let mut inc_kernel: Option<String> = None;
            let inc_gemm = if inc_ok {
                let x16 = gpu.rotate_x_mq_batched_f16(&xt, k, n)?;
                let s = time_ms(&mut gpu, args.warmup, args.iters, |g| {
                    g.gemm_mq6g256v2_xf16(&w, &x16, &y_inc, m, k, n)?;
                    Ok(())
                })?;
                inc_kernel = gpu.last_launched_kernel().map(str::to_owned);
                Some(s)
            } else {
                None
            };
            let inc_prep_ms = median(&inc_prep);
            let inc_total = inc_gemm.as_ref().map(|s| inc_prep_ms + median(s));

            // (name, prep samples, prep bytes, gemm samples, act bytes, kernel)
            let mut rows_out: Vec<ArmRow> = Vec::new();
            for (name, pm) in [("i8", false), ("i8pm", true)] {
                if !(if pm { run_i8pm } else { run_i8 }) {
                    continue;
                }
                // `i8` is always hipcc, `i8pm` always the PM twin, whatever the environment says.
                if gpu.arch == "gfx1201" {
                    set_i8_pm(&mut gpu, pm);
                }
                let prep = time_ms(&mut gpu, args.warmup, args.iters, |g| {
                    g.mq6_i8_prepare_x(&xt, k, n)?;
                    Ok(())
                })?;
                let xq = gpu.mq6_i8_prepare_x(&xt, k, n)?;
                let s = time_ms(&mut gpu, args.warmup, args.iters, |g| {
                    g.gemm_mq6g256v2_i8_prequant(&w, xq, &y_new, m, k, n)?;
                    Ok(())
                })?;
                let kernel = gpu.last_launched_kernel().map(str::to_owned);
                if pm && !kernel.as_deref().is_some_and(|k| k.contains("_pm_")) {
                    println!(
                        "WARNING i8pm M={m} K={k} N={n} ran {kernel:?}, not a PM entry (fell back)"
                    );
                    failed = true;
                }
                rows_out.push(ArmRow {
                    name: name.into(),
                    prep,
                    prep_bytes: i8_prep_bytes(n, k),
                    gemm: s,
                    act_bytes: i8_act_bytes(n, k),
                    kernel,
                });
            }
            if gpu.arch == "gfx1201" {
                set_i8_pm(&mut gpu, false);
            }
            if fp8_on {
                let prep = time_ms(&mut gpu, args.warmup, args.iters, |g| {
                    g.mq6_fp8_prepare_x(&xt, k, n)?;
                    Ok(())
                })?;
                let prepared = gpu.mq6_fp8_prepare_x(&xt, k, n)?;
                for &v in &vars {
                    set_var(&mut gpu, v);
                    let s = time_ms(&mut gpu, args.warmup, args.iters, |g| {
                        g.gemm_mq6g256v2_fp8_prepared(&w, &prepared, &y_new, m, k, n)?;
                        Ok(())
                    })?;
                    rows_out.push(ArmRow {
                        name: format!("fp8_{}", var_name(v)),
                        prep: prep.clone(),
                        prep_bytes: fp8_prep_bytes(n, k),
                        gemm: s,
                        act_bytes: fp8_act_bytes(n, k),
                        kernel: gpu.last_launched_kernel().map(str::to_owned),
                    });
                }
            }

            let fmt_ms = |v: Option<f64>| v.map_or("n/a".to_owned(), |v| format!("{v:.3}"));
            let inc_gemm_ms = inc_gemm.as_ref().map(|s| median(s));
            println!(
                "{:>6}x{:<5} {:>5} | {:<16} | {:>9.3} {:>8.1} | {:>9} {:>7} {:>8} | {:>9} {:>7}",
                m,
                k,
                n,
                "inc",
                inc_prep_ms,
                gbps(inc_prep_bytes(n, k), inc_prep_ms),
                fmt_ms(inc_gemm_ms),
                inc_gemm_ms.map_or("n/a".to_owned(), |v| format!("{:.1}", tflops(m, k, n, v))),
                inc_gemm_ms.map_or("n/a".to_owned(), |v| format!(
                    "{:.0}",
                    gbps(gemm_bytes(m, k, inc_act_bytes(n, k), n), v)
                )),
                fmt_ms(inc_total),
                "1.00x"
            );
            for r in &rows_out {
                let (pm, gm) = (median(&r.prep), median(&r.gemm));
                println!(
                    "{:>6} {:<5} {:>5} | {:<16} | {:>9.3} {:>8.1} | {:>9.3} {:>7.1} {:>8.0} | {:>9.3} {:>7}",
                    "",
                    "",
                    "",
                    r.name,
                    pm,
                    gbps(r.prep_bytes, pm),
                    gm,
                    tflops(m, k, n, gm),
                    gbps(gemm_bytes(m, k, r.act_bytes, n), gm),
                    pm + gm,
                    inc_total.map_or("n/a".to_owned(), |t| format!("{:.2}x", t / (pm + gm)))
                );
            }

            let mut arms_json = serde_json::Map::new();
            arms_json.insert(
                "inc_prep".into(),
                json!({
                    "median_ms": inc_prep_ms, "samples_ms": inc_prep,
                    "gbps": gbps(inc_prep_bytes(n, k), inc_prep_ms),
                }),
            );
            arms_json.insert(
                "inc_gemm".into(),
                match &inc_gemm {
                    Some(s) => json!({
                        "median_ms": median(s), "tflops": tflops(m, k, n, median(s)),
                        "gbps": gbps(gemm_bytes(m, k, inc_act_bytes(n, k), n), median(s)),
                        "samples_ms": s, "kernel": inc_kernel,
                    }),
                    None => json!("n/a"),
                },
            );
            for r in &rows_out {
                let (pm, gm) = (median(&r.prep), median(&r.gemm));
                arms_json.insert(
                    format!("{}_prep", r.name),
                    json!({
                        "median_ms": pm, "samples_ms": r.prep,
                        "gbps": gbps(r.prep_bytes, pm),
                    }),
                );
                arms_json.insert(
                    format!("{}_gemm", r.name),
                    json!({
                        "median_ms": gm, "tflops": tflops(m, k, n, gm),
                        "gbps": gbps(gemm_bytes(m, k, r.act_bytes, n), gm),
                        "samples_ms": r.gemm, "kernel": r.kernel,
                        "vs_inc": inc_total.map(|t| t / (pm + gm)),
                    }),
                );
            }
            runs.push(json!({"m": m, "k": k, "n": n, "arms": Value::Object(arms_json)}));

            gpu.free_tensor(y_inc)?;
            gpu.free_tensor(y_new)?;
            gpu.free_tensor(xt)?;
        }

        if args.check {
            for &n in &CHECK_NS {
                if run_i8 || run_i8pm {
                    let c = check_producer_case(&mut gpu, xfull, k, n)?;
                    let ran = c.kernel.contains("mq_rotate_x_i8");
                    println!(
                        "check i8 producer M={m} K={k} N={n} one-pass==two-pass blocks={} (diff bytes {} of {}) kernel={}",
                        if c.diff_bytes == 0 && ran { "PASS" } else { "FAIL" },
                        c.diff_bytes,
                        c.bytes,
                        c.kernel,
                    );
                    if c.diff_bytes != 0 || !ran {
                        failed = true;
                    }
                    checks.push(json!({
                        "arm": "i8_producer", "m": m, "k": k, "n": n,
                        "pass": c.diff_bytes == 0 && ran,
                        "diff_bytes": c.diff_bytes, "bytes": c.bytes, "kernel": c.kernel,
                    }));
                }
                if run_i8pm {
                    let c = check_i8pm_case(&mut gpu, &w, xfull, m, k, n)?;
                    println!(
                        "check i8pm M={m} K={k} N={n} bitwise PM==HIP f32={} (diff bytes {}) bf16={} (diff bytes {}) kernels hip={:?} pm={:?}",
                        if c.f32_bad == 0 { "PASS" } else { "FAIL" },
                        c.f32_bad,
                        if c.bf16_bad == 0 { "PASS" } else { "FAIL" },
                        c.bf16_bad,
                        c.hip_kernels,
                        c.pm_kernels,
                    );
                    if !c.pass() {
                        failed = true;
                    }
                    checks.push(json!({
                        "arm": "i8pm", "m": m, "k": k, "n": n,
                        "pass": c.pass(), "f32_diff_bytes": c.f32_bad,
                        "bf16_diff_bytes": c.bf16_bad, "pm_ran": c.pm_ran,
                        "hip_kernels": c.hip_kernels, "pm_kernels": c.pm_kernels,
                    }));
                }
                if run_i8 {
                    set_i8_pm(&mut gpu, false);
                    let c = check_i8_case(&mut gpu, &w, &wbytes, xfull, m, k, n, args.seed)?;
                    let bf16_ok = c.bf16_bad == 0;
                    println!(
                        "check i8 M={m} K={k} N={n} tokens={} i8={} (bad={} nonfinite={} max_err/tol={:.3}) inc_vs_i8_rel_l2={} bf16={}",
                        c.tokens,
                        if c.pass { "PASS" } else { "FAIL" },
                        c.bad,
                        c.nonfinite,
                        c.max_ratio,
                        c.inc_rel_l2.map_or("n/a".to_owned(), |v| format!("{v:.3e}")),
                        if bf16_ok { "PASS".to_owned() } else { format!("FAIL({})", c.bf16_bad) },
                    );
                    if !c.pass || !bf16_ok {
                        failed = true;
                    }
                    checks.push(json!({
                        "arm": "i8", "m": m, "k": k, "n": n, "tokens": c.tokens,
                        "pass": c.pass, "bad": c.bad, "nonfinite": c.nonfinite,
                        "max_err_over_tol": c.max_ratio,
                        "inc_vs_i8_rel_l2": c.inc_rel_l2,
                        "bf16_pass": bf16_ok, "bf16_mismatches": c.bf16_bad,
                    }));
                }
                if fp8_on {
                    let c = check_case(&mut gpu, &w, &wbytes, xfull, m, k, n, args.seed, &vars)?;
                    let bf16_ok = c.bf16.map_or(true, |(ok, _)| ok);
                    println!(
                        "check M={m} K={k} N={n} tokens={} split={} (bad={} nonfinite={} max_err/tol={:.3}) round_rel_l2={:.3e} inc_vs_split_rel_l2={} bf16={}",
                        c.tokens,
                        if c.split_pass { "PASS" } else { "FAIL" },
                        c.split_bad,
                        c.split_nonfinite,
                        c.split_max_ratio,
                        c.round_rel_l2,
                        c.inc_vs_split_rel_l2
                            .map_or("n/a".to_owned(), |v| format!("{v:.3e}")),
                        c.bf16.map_or("n/a".to_owned(), |(ok, bad)| {
                            if ok { "PASS".to_owned() } else { format!("FAIL({bad})") }
                        }),
                    );
                    // A SPLIT pass is only meaningful when SPLIT ran.
                    let split_ran = vars.iter().any(|(a, _)| *a == Mq6Fp8Arm::Split);
                    if (split_ran && !c.split_pass) || !bf16_ok {
                        failed = true;
                    }
                    checks.push(json!({
                        "arm": "fp8", "m": m, "k": k, "n": n, "tokens": c.tokens,
                        "split_ran": split_ran,
                        "split_pass": c.split_pass, "split_bad": c.split_bad,
                        "split_nonfinite": c.split_nonfinite,
                        "split_max_err_over_tol": c.split_max_ratio,
                        "round_rel_l2": c.round_rel_l2,
                        "inc_vs_split_rel_l2": c.inc_vs_split_rel_l2,
                        "bf16_pass": c.bf16.map(|(ok, _)| ok),
                        "bf16_mismatches": c.bf16.map(|(_, b)| b),
                    }));
                }
            }
        }
        gpu.free_tensor(w)?;
    }

    if let Some(path) = &args.json {
        let doc = json!({
            "git_rev": git_rev(),
            "gpu_arch": gpu.arch,
            "iters": args.iters,
            "warmup": args.warmup,
            "seed": args.seed,
            "arms": arm_names,
            "variants": vars.iter().map(|v| var_name(*v)).collect::<Vec<_>>(),
            "runs": runs,
            "checks": checks,
            "failed": failed,
        });
        std::fs::write(path, serde_json::to_string_pretty(&doc)?)?;
        println!("wrote {path}");
    }
    if failed {
        println!("RESULT FAIL");
        std::process::exit(1);
    }
    println!("RESULT OK");
    Ok(())
}
