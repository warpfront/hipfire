// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU correctness gate for the gfx1201 lane-WMMA launchers (`lane_wmma_gemm`,
//! `lane_wmma_read_up`; kernels `qwen4_lane_wmma.gfx1201.hip`) against CPU references:
//! BF16 HC-down shape with the `hc_act` epilogue, dense 2-term shapes, planar W8 weights, the
//! fused HC read-up mix, and split-K determinism (including the counters being left at zero).
//!
//! `#[ignore]`d: needs an exact gfx1201 device, no model files.
//!
//!   cargo test -p rdna-compute --release --test lane_wmma_hw -- --ignored --nocapture

use rdna_compute::lane_wmma::{
    lane_wmma_gemm, lane_wmma_read_up, LaneWmmaCfg, LaneWmmaScratch, LaneWmmaWeight,
    LANE_WMMA_SCRATCH_COUNTERS, LANE_WMMA_SCRATCH_PARTIAL_FLOATS,
};
use rdna_compute::{DType, Gpu, GpuTensor};

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform in (0, 1).
    fn unit(&mut self) -> f64 {
        ((self.next_u32() as f64) + 0.5) / 2147483648.0
    }
    fn normal(&mut self) -> f32 {
        let u1 = self.unit();
        let u2 = self.unit();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

/// Round-to-nearest-even through BF16, returned as f32.
fn bf16r(v: f32) -> f32 {
    f32::from_bits((bf16_bits(v) as u32) << 16)
}
fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    if v.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    let r = 0x7FFFu32 + ((b >> 16) & 1);
    (b.wrapping_add(r) >> 16) as u16
}
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
/// Distance in BF16 ulps (monotone ordered-integer mapping, so it is sign-safe).
fn bf16_ulps(a: f32, b: f32) -> u32 {
    fn ord(v: f32) -> i32 {
        let bits = bf16_bits(v) as i32;
        if bits & 0x8000 != 0 {
            -(bits & 0x7FFF)
        } else {
            bits
        }
    }
    (ord(a) - ord(b)).unsigned_abs()
}

/// f32 -> f16 bits, round-to-nearest-even, finite normal/subnormal inputs of modest magnitude.
fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let man = b & 0x7F_FFFF;
    if v == 0.0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - exp) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let up = rem > mid || (rem == mid && (half & 1) == 1);
        return sign | (half + up as u32) as u16;
    }
    let half = ((exp as u32) << 10) | (man >> 13);
    let rem = man & 0x1FFF;
    let up = rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1);
    sign | (half + up as u32) as u16
}
fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1F) as i32;
    let man = (h & 0x3FF) as f64;
    if exp == 0 {
        sign * man * 2f64.powi(-24)
    } else {
        sign * (1.0 + man / 1024.0) * 2f64.powi(exp - 15)
    }
}

fn upload_bytes(gpu: &mut Gpu, dtype: DType, numel: usize, data: &[u8]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[numel], dtype).expect("alloc");
    gpu.hip.memcpy_htod(&t.buf, data).expect("htod");
    t
}
fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    upload_bytes(gpu, DType::F32, data.len(), bytes)
}
fn upload_bf16(gpu: &mut Gpu, bits: &[u16]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(bits.as_ptr() as *const u8, bits.len() * 2) };
    upload_bytes(gpu, DType::BF16, bits.len(), bytes)
}
fn download_f32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
    out
}
fn bits_of(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn gpu_1201() -> Option<Gpu> {
    let gpu = Gpu::init().expect("GPU required");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP: requires exact gfx1201.");
        return None;
    }
    Some(gpu)
}

fn new_scratch(gpu: &mut Gpu) -> LaneWmmaScratch {
    LaneWmmaScratch::new(gpu, LANE_WMMA_SCRATCH_PARTIAL_FLOATS, LANE_WMMA_SCRATCH_COUNTERS)
        .expect("scratch")
}

/// Random BF16 matrix as (bits, f32 values), N(0,1) * `scale`.
fn rand_bf16(rng: &mut Lcg, n: usize, scale: f32) -> (Vec<u16>, Vec<f32>) {
    let bits: Vec<u16> = (0..n).map(|_| bf16_bits(rng.normal() * scale)).collect();
    let vals = bits.iter().map(|&b| bf16_to_f32(b)).collect();
    (bits, vals)
}

/// f64 `X . W^T`: `x [rows, k]`, `w [m, k]` -> `[rows, m]`.
fn ref_gemm(w: &[f64], x: &[f32], m: usize, k: usize, rows: usize) -> Vec<f64> {
    let mut out = vec![0f64; rows * m];
    for r in 0..rows {
        let xr = &x[r * k..(r + 1) * k];
        for c in 0..m {
            let wr = &w[c * k..(c + 1) * k];
            let mut s = 0f64;
            for i in 0..k {
                s += wr[i] * xr[i] as f64;
            }
            out[r * m + c] = s;
        }
    }
    out
}

fn hc_act_ref(sum: f64, scale: f32) -> f32 {
    let mut v = bf16r(sum as f32);
    v *= scale;
    v = bf16r(v);
    v = v / (1.0 + (-v).exp());
    bf16r(v)
}

fn max_rel_err(got: &[f32], want: &[f64]) -> f64 {
    let amax = want.iter().fold(0f64, |a, &b| a.max(b.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0f64, |a, (&g, &w)| a.max((g as f64 - w).abs()));
    err / amax
}

struct HcDown {
    w_bits: Vec<u16>,
    w_t: GpuTensor,
}
const HC_M: usize = 320;
const HC_K: usize = 10240;

fn hc_down_setup(gpu: &mut Gpu) -> HcDown {
    let mut rng = Lcg(0x1A4E_0001);
    let (w_bits, _) = rand_bf16(&mut rng, HC_M * HC_K, 0.05);
    let w_t = upload_bf16(gpu, &w_bits);
    HcDown { w_bits, w_t }
}

fn bf16_exact_x(rng: &mut Lcg, n: usize) -> Vec<f32> {
    (0..n).map(|_| bf16r(rng.normal())).collect()
}

fn run_gemm(
    gpu: &mut Gpu,
    w: &LaneWmmaWeight<'_>,
    x: &[f32],
    m: usize,
    k: usize,
    rows: usize,
    terms: usize,
    act: Option<f32>,
    scratch: &LaneWmmaScratch,
) -> Vec<f32> {
    let x_t = upload_f32(gpu, x);
    let y_t = gpu.zeros(&[rows * m], DType::F32).expect("y");
    lane_wmma_gemm(
        gpu,
        w,
        &x_t,
        &y_t,
        m,
        k,
        rows,
        terms,
        act,
        LaneWmmaCfg::default(),
        scratch,
    )
    .expect("lane_wmma_gemm");
    gpu.hip.device_synchronize().expect("sync");
    download_f32(gpu, &y_t, rows * m)
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn lane_wmma_gemm_bf16_hc_act_matches_cpu_hw() {
    let Some(mut gpu) = gpu_1201() else { return };
    let scratch = new_scratch(&mut gpu);
    let hc = hc_down_setup(&mut gpu);
    let w64: Vec<f64> = hc.w_bits.iter().map(|&b| bf16_to_f32(b) as f64).collect();
    for rows in [16usize, 33, 48, 64] {
        let mut rng = Lcg(0x1A4E_1000 + rows as u64);
        let x = bf16_exact_x(&mut rng, rows * HC_K);
        let got = run_gemm(
            &mut gpu,
            &LaneWmmaWeight::Bf16(&hc.w_t),
            &x,
            HC_M,
            HC_K,
            rows,
            1,
            Some(0.25),
            &scratch,
        );
        let want: Vec<f32> = ref_gemm(&w64, &x, HC_M, HC_K, rows)
            .into_iter()
            .map(|s| hc_act_ref(s, 0.25))
            .collect();
        let mut diff_bits = 0usize;
        let mut worst = 0u32;
        for (g, w) in got.iter().zip(&want) {
            if bf16_bits(*g) != bf16_bits(*w) {
                diff_bits += 1;
            }
            worst = worst.max(bf16_ulps(*g, *w));
        }
        eprintln!(
            "hc_act rows={rows}: worst {worst} ulp, {diff_bits}/{} differ",
            got.len()
        );
        assert!(worst <= 1, "rows={rows}: worst {worst} BF16 ulps");
        assert!(
            diff_bits * 100 <= got.len(),
            "rows={rows}: {diff_bits}/{} outputs differ in bits",
            got.len()
        );
    }
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn lane_wmma_gemm_dense_two_term_matches_cpu_hw() {
    let Some(mut gpu) = gpu_1201() else { return };
    let scratch = new_scratch(&mut gpu);
    for (m, k) in [(384usize, 2560usize), (2560, 384)] {
        let mut rng = Lcg(0x1A4E_2000 + m as u64);
        let (w_bits, w_vals) = rand_bf16(&mut rng, m * k, 0.05);
        let w_t = upload_bf16(&mut gpu, &w_bits);
        let w64: Vec<f64> = w_vals.iter().map(|&v| v as f64).collect();
        for rows in [16usize, 48] {
            let x: Vec<f32> = (0..rows * k).map(|_| rng.normal()).collect();
            let want = ref_gemm(&w64, &x, m, k, rows);
            let w = LaneWmmaWeight::Bf16(&w_t);
            let got2 = run_gemm(&mut gpu, &w, &x, m, k, rows, 2, None, &scratch);
            let got1 = run_gemm(&mut gpu, &w, &x, m, k, rows, 1, None, &scratch);
            let e2 = max_rel_err(&got2, &want);
            let e1 = max_rel_err(&got1, &want);
            eprintln!("dense m={m} k={k} rows={rows}: terms2 {e2:.3e}, terms1 {e1:.3e}");
            assert!(e2 <= 2e-5, "m={m} k={k} rows={rows}: terms=2 rel err {e2:.3e}");
            assert!(
                e1 > e2,
                "m={m} k={k} rows={rows}: terms=1 ({e1:.3e}) not worse than terms=2 ({e2:.3e})"
            );
        }
    }
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn lane_wmma_gemm_w8_matches_cpu_hw() {
    let Some(mut gpu) = gpu_1201() else { return };
    let scratch = new_scratch(&mut gpu);
    let (m, k, rows) = (HC_M, HC_K, 16usize);
    let mut rng = Lcg(0x1A4E_3000);
    let (_, w_vals) = rand_bf16(&mut rng, m * k, 0.05);
    let mut q = vec![0i8; m * k];
    let mut d_bits = vec![0u16; m * k / 32];
    let mut w_deq = vec![0f64; m * k];
    for g in 0..m * k / 32 {
        let blk = &w_vals[g * 32..(g + 1) * 32];
        let amax = blk.iter().fold(0f32, |a, &b| a.max(b.abs()));
        let dh = f16_bits(amax / 127.0);
        let d = f16_to_f64(dh);
        d_bits[g] = dh;
        for (i, &v) in blk.iter().enumerate() {
            let qi = if d == 0.0 {
                0.0
            } else {
                (v as f64 / d).round().clamp(-127.0, 127.0)
            };
            q[g * 32 + i] = qi as i8;
            w_deq[g * 32 + i] = qi * d;
        }
    }
    // The i8 bytes are uploaded raw; the launcher only checks byte extents.
    let q_bytes: Vec<u8> = q.iter().map(|&v| v as u8).collect();
    let q_t = upload_bytes(&mut gpu, DType::F16, m * k / 2, &q_bytes);
    let d_bytes: Vec<u8> = d_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let d_t = upload_bytes(&mut gpu, DType::F16, d_bits.len(), &d_bytes);
    let x = bf16_exact_x(&mut rng, rows * k);
    let got = run_gemm(
        &mut gpu,
        &LaneWmmaWeight::W8 { q: &q_t, d: &d_t },
        &x,
        m,
        k,
        rows,
        1,
        None,
        &scratch,
    );
    let want = ref_gemm(&w_deq, &x, m, k, rows);
    let e = max_rel_err(&got, &want);
    eprintln!("w8 rel err {e:.3e}");
    assert!(e <= 1e-5, "w8 rel err {e:.3e}");
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn lane_wmma_read_up_matches_cpu_hw() {
    let Some(mut gpu) = gpu_1201() else { return };
    let (hidden, low_rank) = (2560usize, 320usize);
    let wide = 4 * hidden;
    let mut rng = Lcg(0x1A4E_4000);
    let (w_bits, w_vals) = rand_bf16(&mut rng, wide * low_rank, 0.05);
    let w_t = upload_bf16(&mut gpu, &w_bits);
    let w64: Vec<f64> = w_vals.iter().map(|&v| v as f64).collect();
    for rows in [16usize, 48] {
        let low = bf16_exact_x(&mut rng, rows * low_rank);
        let norm = bf16_exact_x(&mut rng, rows * wide);
        let low_t = upload_f32(&mut gpu, &low);
        let norm_t = upload_f32(&mut gpu, &norm);
        let mixed_t = gpu.zeros(&[rows * hidden], DType::F32).expect("mixed");
        lane_wmma_read_up(
            &mut gpu,
            &LaneWmmaWeight::Bf16(&w_t),
            &low_t,
            &norm_t,
            &mixed_t,
            rows,
            hidden,
            low_rank,
            1,
        )
        .expect("lane_wmma_read_up");
        gpu.hip.device_synchronize().expect("sync");
        let got = download_f32(&gpu, &mixed_t, rows * hidden);
        // logits[r, br * hidden + j] = low[r] . W[br * hidden + j]
        let logits = ref_gemm(&w64, &low, wide, low_rank, rows);
        let mut diff_bits = 0usize;
        let mut worst = 0u32;
        for r in 0..rows {
            for j in 0..hidden {
                let mut value = 0.0f32;
                for br in 0..4 {
                    let h = logits[r * wide + br * hidden + j] as f32;
                    let gate_input = bf16r(h);
                    let gate = bf16r(1.0 / (1.0 + (-gate_input).exp()));
                    let normed = bf16r(norm[r * wide + br * hidden + j]);
                    value += bf16r(gate * normed);
                }
                let want = bf16r(value / 4.0);
                let g = got[r * hidden + j];
                if bf16_bits(g) != bf16_bits(want) {
                    diff_bits += 1;
                }
                worst = worst.max(bf16_ulps(g, want));
            }
        }
        let n = rows * hidden;
        eprintln!("read_up rows={rows}: worst {worst} ulp, {diff_bits}/{n} differ");
        assert!(worst <= 2, "rows={rows}: worst {worst} BF16 ulps");
        assert!(
            diff_bits * 100 <= n,
            "rows={rows}: {diff_bits}/{n} outputs differ in bits"
        );
    }
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn lane_wmma_gemm_split_k_deterministic_hw() {
    let Some(mut gpu) = gpu_1201() else { return };
    let scratch = new_scratch(&mut gpu);
    let hc = hc_down_setup(&mut gpu);
    let rows = 48usize;
    let mut rng = Lcg(0x1A4E_5000);
    let x = bf16_exact_x(&mut rng, rows * HC_K);
    let w = LaneWmmaWeight::Bf16(&hc.w_t);
    let a = run_gemm(&mut gpu, &w, &x, HC_M, HC_K, rows, 1, Some(0.25), &scratch);
    let b = run_gemm(&mut gpu, &w, &x, HC_M, HC_K, rows, 1, Some(0.25), &scratch);
    assert_eq!(bits_of(&a), bits_of(&b), "back-to-back runs differ");
    // A different-shape call in between must leave the counters at zero for the next run.
    let (m2, k2, rows2) = (384usize, 2560usize, 16usize);
    let w2: Vec<u16> = (0..m2 * k2).map(|_| bf16_bits(rng.normal() * 0.05)).collect();
    let w2_t = upload_bf16(&mut gpu, &w2);
    let x2: Vec<f32> = (0..rows2 * k2).map(|_| rng.normal()).collect();
    let _ = run_gemm(
        &mut gpu,
        &LaneWmmaWeight::Bf16(&w2_t),
        &x2,
        m2,
        k2,
        rows2,
        2,
        None,
        &scratch,
    );
    let c = run_gemm(&mut gpu, &w, &x, HC_M, HC_K, rows, 1, Some(0.25), &scratch);
    assert_eq!(bits_of(&a), bits_of(&c), "run after a different shape differs");
    let counters = download_f32(&gpu, &scratch.counters, LANE_WMMA_SCRATCH_COUNTERS);
    assert!(
        counters.iter().all(|c| c.to_bits() == 0),
        "split-K counters not left at zero"
    );
}
