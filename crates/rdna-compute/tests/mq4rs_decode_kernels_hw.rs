// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte-exactness gates for the 0.4.4 mq4rs decode kernels (A3B, gfx1201):
//!
//! * `gemv_mq4g128v2_moe_ninepath_d4` vs the four-kernel sequence it replaces
//!   (`silu_mul_f32` -> `rotate_x_mq_128_v2` ->
//!   `gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded` ->
//!   `moe_down_combine_k8_batched`), raw f32 bits of the accumulated residual.
//! * `gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2` (two rows per wave) vs
//!   the one-row `gemv_mq4g256v2_residual_sigmoid_scaled_k512`, raw f32 bits.
//!
//! Both compare the *real* production launchers on randomized inputs; the
//! compiler's FMA contraction of the originals is not reproducible by a CPU
//! model, so the originals are the oracle.
//!
//! `#[ignore]`d: needs an exact gfx1201 device, no model files.
//!
//!   cargo test -p rdna-compute --release --test mq4rs_decode_kernels_hw -- --ignored --nocapture

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
    /// Full-mantissa f32 in [-1, 1).
    fn f32_signed(&mut self) -> f32 {
        let a = self.next_u32() & 0x7F_FFFF;
        let sign = self.next_u32() & 1;
        let v = f32::from_bits(0x3F80_0000 | a) - 1.0;
        if sign == 1 {
            -v
        } else {
            v
        }
    }
    /// Normal f16 bit pattern with biased exponent in `lo..=hi`.
    fn f16_normal(&mut self, lo: u32, hi: u32, signed: bool) -> u16 {
        let exp = lo + self.next_u32() % (hi - lo + 1);
        let mant = self.next_u32() & 0x3FF;
        let sign = if signed { self.next_u32() & 1 } else { 0 };
        ((sign << 15) | (exp << 10) | mant) as u16
    }
}

fn f16_normal_to_f32(h: u16) -> f32 {
    let (sign, exp, mant) = (
        (h >> 15) as u32,
        ((h >> 10) & 0x1F) as u32,
        (h & 0x3FF) as u32,
    );
    assert!(exp != 0 && exp != 0x1F, "fixture emits normal f16 only");
    f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
}

fn f32_to_f16_bits_exact(v: f32) -> u16 {
    // Only called with values built from f16 normals scaled by a power of two.
    let b = v.to_bits();
    let sign = (b >> 31) as u16;
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    assert!((1..31).contains(&exp), "f16 exponent out of range");
    let mant = ((b >> 13) & 0x3FF) as u16;
    (sign << 15) | ((exp as u16) << 10) | mant
}

fn upload_u8(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::Raw)
        .expect("alloc u8");
    gpu.hip.memcpy_htod(&t.buf, data).expect("htod u8");
    t
}
fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len()], DType::F32)
        .expect("alloc f32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod f32");
    t
}
fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len() * 4], DType::Raw)
        .expect("alloc i32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod i32");
    t
}
fn upload_u64(gpu: &mut Gpu, data: &[u64]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
    let t = gpu
        .alloc_tensor(&[data.len() * 8], DType::Raw)
        .expect("alloc u64");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod u64");
    t
}
fn download_f32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
    out
}

struct Cmp {
    mismatches: usize,
    max_abs: f32,
    first: Option<(usize, u32, u32)>,
}
fn compare(got: &[f32], want: &[f32]) -> Cmp {
    let mut c = Cmp {
        mismatches: 0,
        max_abs: 0.0,
        first: None,
    };
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g.to_bits() != w.to_bits() {
            c.mismatches += 1;
            c.first.get_or_insert((i, g.to_bits(), w.to_bits()));
            let d = (g - w).abs();
            if d > c.max_abs || d.is_nan() {
                c.max_abs = d;
            }
        }
    }
    c
}

// ---------------------------------------------------------------------------
// gemv_mq4g128v2_moe_ninepath_d4
// ---------------------------------------------------------------------------

const DOWN_M: usize = 2048;
const DOWN_K: usize = 512;
const MI: usize = 512;
const N_EXP: usize = 24;
const K_TOP: usize = 8;
const ROW_BYTES: usize = 4 * 68;

/// One MQ4G128V2 (qt=53) expert: `DOWN_M` rows x 4 groups of 68 B
/// `[scale:f16][zero:f16][64 payload bytes]`. Half of the experts use the
/// symmetric header (`zero = -8 * scale`), the rest an arbitrary affine pair.
fn synth_qt53_expert(seed: u64, symmetric: bool) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let mut out = vec![0u8; DOWN_M * ROW_BYTES];
    for row in 0..DOWN_M {
        for g in 0..4 {
            let off = row * ROW_BYTES + g * 68;
            let sc = rng.f16_normal(9, 13, false); // 2^-6 .. 2^-2
            let zp = if symmetric {
                f32_to_f16_bits_exact(-8.0 * f16_normal_to_f32(sc))
            } else {
                rng.f16_normal(8, 14, true)
            };
            out[off..off + 2].copy_from_slice(&sc.to_le_bytes());
            out[off + 2..off + 4].copy_from_slice(&zp.to_le_bytes());
            for w in 0..16 {
                let pk = rng.next_u32() ^ (rng.next_u32() << 16);
                out[off + 4 + 4 * w..off + 8 + 4 * w].copy_from_slice(&pk.to_le_bytes());
            }
        }
    }
    out
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn mq4g128v2_ninepath_matches_four_kernel_sequence_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP — requires exact gfx1201.");
        return;
    }
    let mut keep: Vec<GpuTensor> = Vec::new();
    let mut ptrs: Vec<u64> = Vec::new();
    for e in 0..N_EXP {
        let blob = synth_qt53_expert(0x5133_0000 + e as u64, e % 2 == 0);
        let t = upload_u8(&mut gpu, &blob);
        ptrs.push(t.buf.as_ptr() as u64);
        keep.push(t);
    }
    let expert_ptrs = upload_u64(&mut gpu, &ptrs);

    let mut total_mismatch = 0usize;
    // trial 0..5: random gate/up of growing magnitude; trial 5 plants invalid
    // expert ids (negative and >= n_exp) to exercise the zero-row guard;
    // trial 6 plants exact zeros and a large-magnitude tail.
    for trial in 0..7u64 {
        let mut rng = Lcg(0xA3B0_4400 ^ trial.wrapping_mul(0x9E37_79B9));
        let mag = [0.5f32, 1.0, 2.0, 4.0, 8.0, 3.0, 6.0][trial as usize];
        let mut gate: Vec<f32> = (0..K_TOP * MI).map(|_| rng.f32_signed() * mag).collect();
        let mut up: Vec<f32> = (0..K_TOP * MI).map(|_| rng.f32_signed() * mag).collect();
        if trial == 6 {
            for i in (0..K_TOP * MI).step_by(7) {
                gate[i] = 0.0;
            }
            for i in (3..K_TOP * MI).step_by(11) {
                up[i] = 0.0;
                gate[i] = rng.f32_signed() * 40.0; // saturating silu
            }
        }
        let mut topk: Vec<i32> = Vec::new();
        while topk.len() < K_TOP {
            let e = (rng.next_u32() as usize % N_EXP) as i32;
            if !topk.contains(&e) {
                topk.push(e);
            }
        }
        if trial == 5 {
            topk[2] = -1;
            topk[6] = N_EXP as i32 + 3;
        }
        let mut weights: Vec<f32> = (0..K_TOP).map(|_| rng.f32_signed().abs() + 0.01).collect();
        let wsum: f32 = weights.iter().sum();
        for w in &mut weights {
            *w /= wsum;
        }
        let resid: Vec<f32> = (0..DOWN_M).map(|_| rng.f32_signed() * 3.0).collect();

        let gate_t = upload_f32(&mut gpu, &gate);
        let up_t = upload_f32(&mut gpu, &up);
        let topk_t = upload_i32(&mut gpu, &topk);
        let w_t = upload_f32(&mut gpu, &weights);

        // Oracle: the production four-launch sequence.
        let rot_t = upload_f32(&mut gpu, &vec![0.0f32; K_TOP * MI]);
        let expanded_t = upload_f32(&mut gpu, &vec![0.0f32; K_TOP * DOWN_M]);
        let out_ref = upload_f32(&mut gpu, &resid);
        gpu.silu_mul_f32(&gate_t, &up_t, &rot_t).expect("silu_mul");
        gpu.rotate_x_mq_128_v2(&rot_t, &rot_t, MI, K_TOP)
            .expect("rotate");
        gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
            &expert_ptrs,
            &topk_t,
            &rot_t,
            &expanded_t,
            DOWN_M,
            DOWN_K,
            1,
            N_EXP,
            K_TOP,
        )
        .expect("expanded down");
        gpu.moe_down_combine_k8_batched(&expanded_t, &w_t, &out_ref, DOWN_M, K_TOP, 1)
            .expect("combine");

        // Candidate, with a canary row past `out` to catch overruns.
        let mut cand = resid.clone();
        cand.extend(std::iter::repeat(f32::from_bits(0x7FC0_DEAD)).take(64));
        let out_new = upload_f32(&mut gpu, &cand);
        gpu.gemv_mq4g128v2_moe_ninepath_d4(
            &expert_ptrs,
            &topk_t,
            &w_t,
            &gate_t,
            &up_t,
            &out_new,
            DOWN_M,
            DOWN_K,
            N_EXP,
        )
        .expect("ninepath");
        gpu.hip.device_synchronize().expect("sync");

        let want = download_f32(&gpu, &out_ref, DOWN_M);
        let got_all = download_f32(&gpu, &out_new, DOWN_M + 64);
        let canary_ok = got_all[DOWN_M..].iter().all(|v| v.to_bits() == 0x7FC0_DEAD);
        let cmp = compare(&got_all[..DOWN_M], &want);
        // Non-vacuity: the fold must have moved the residual.
        let moved = want
            .iter()
            .zip(&resid)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!(
            "ninepath trial {trial}: mismatches={}/{DOWN_M} max_abs={:e} first={:x?} canary_ok={canary_ok} residual_moved={moved}",
            cmp.mismatches, cmp.max_abs, cmp.first
        );
        assert!(canary_ok, "trial {trial}: ninepath wrote past out[down_m]");
        assert!(
            moved > DOWN_M / 2,
            "trial {trial}: oracle did not move the residual"
        );
        total_mismatch += cmp.mismatches;
    }
    assert_eq!(
        total_mismatch, 0,
        "gemv_mq4g128v2_moe_ninepath_d4 is not byte-exact with the four-kernel sequence"
    );
}

// ---------------------------------------------------------------------------
// gemv_mq4g256v2_residual_sigmoid_scaled_k512 (two-row vs one-row)
// ---------------------------------------------------------------------------

const SH_M: usize = 2048;
const SH_K: usize = 512;

/// MQ4G256V2 rows: `SH_M` x 2 groups of 136 B
/// `[sc0:f16][zp0:f16][sc1:f16][zp1:f16][32 x u32 nibbles]`.
fn synth_mq4v2_rows(seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let mut out = vec![0u8; SH_M * 2 * 136];
    for row in 0..SH_M {
        for g in 0..2 {
            let off = (row * 2 + g) * 136;
            for half in 0..2 {
                let sc = rng.f16_normal(8, 13, false);
                let zp = rng.f16_normal(8, 14, true);
                out[off + 4 * half..off + 4 * half + 2].copy_from_slice(&sc.to_le_bytes());
                out[off + 4 * half + 2..off + 4 * half + 4].copy_from_slice(&zp.to_le_bytes());
            }
            for w in 0..32 {
                let pk = rng.next_u32() ^ (rng.next_u32() << 16);
                out[off + 8 + 4 * w..off + 12 + 4 * w].copy_from_slice(&pk.to_le_bytes());
            }
        }
    }
    out
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn mq4g256v2_k512_two_row_matches_one_row_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP — requires exact gfx1201.");
        return;
    }
    let mut total_mismatch = 0usize;
    let gates = [0.37f32, -2.5, 11.0, -9.0, 0.0, 6.5];
    for (trial, &cval) in gates.iter().enumerate() {
        let mut rng = Lcg(0x4B35_1200 ^ (trial as u64) << 12);
        let a = synth_mq4v2_rows(0x7700 + trial as u64);
        let a_t = upload_u8(&mut gpu, &a);
        let x: Vec<f32> = (0..SH_K).map(|_| rng.f32_signed() * 3.0).collect();
        let y0: Vec<f32> = (0..SH_M).map(|_| rng.f32_signed() * 2.0).collect();
        let x_t = upload_f32(&mut gpu, &x);
        let c_t = upload_f32(&mut gpu, &[cval]);
        let y_ref = upload_f32(&mut gpu, &y0);
        let mut cand = y0.clone();
        cand.extend(std::iter::repeat(f32::from_bits(0x7FC0_DEAD)).take(64));
        let y_new = upload_f32(&mut gpu, &cand);
        gpu.gemv_mq4g256v2_residual_sigmoid_scaled_k512_one_row(
            &a_t, &x_t, &y_ref, &c_t, SH_M, SH_K,
        )
        .expect("one-row");
        gpu.gemv_mq4g256v2_residual_sigmoid_scaled_k512_two_row(
            &a_t, &x_t, &y_new, &c_t, SH_M, SH_K,
        )
        .expect("two-row");
        gpu.hip.device_synchronize().expect("sync");
        let want = download_f32(&gpu, &y_ref, SH_M);
        let got_all = download_f32(&gpu, &y_new, SH_M + 64);
        let canary_ok = got_all[SH_M..].iter().all(|v| v.to_bits() == 0x7FC0_DEAD);
        let c_after = download_f32(&gpu, &c_t, 1)[0];
        let cmp = compare(&got_all[..SH_M], &want);
        let moved = want
            .iter()
            .zip(&y0)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!(
            "k512 trial {trial} c={cval}: mismatches={}/{SH_M} max_abs={:e} first={:x?} canary_ok={canary_ok} c_intact={} residual_moved={moved}",
            cmp.mismatches,
            cmp.max_abs,
            cmp.first,
            c_after.to_bits() == cval.to_bits()
        );
        assert!(canary_ok, "trial {trial}: two-row wrote past y[M]");
        assert_eq!(c_after.to_bits(), cval.to_bits(), "c_buf mutated");
        assert!(moved > SH_M / 2, "trial {trial}: oracle did not move y");
        total_mismatch += cmp.mismatches;
    }
    assert_eq!(
        total_mismatch, 0,
        "gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2 is not byte-exact with the one-row kernel"
    );
}

// ---------------------------------------------------------------------------
// Launch-loop driver for `rocprofv3 --kernel-trace` (per-kernel microseconds).
// ---------------------------------------------------------------------------

/// Cold-weight launch loop: a 256-expert pool (142 MB, past the infinity
/// cache) with a different top-8 every iteration, old four-kernel sequence and
/// the fused kernel interleaved. Run under `rocprofv3 --kernel-trace` and read
/// the per-kernel durations; prints nothing useful by itself.
#[test]
#[ignore = "profiling driver; needs an exact gfx1201 GPU"]
fn mq4rs_decode_kernels_profile_driver_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP — requires exact gfx1201.");
        return;
    }
    const POOL: usize = 256;
    const ITERS: usize = 300;
    let mut keep: Vec<GpuTensor> = Vec::new();
    let mut ptrs: Vec<u64> = Vec::new();
    for e in 0..POOL {
        let blob = synth_qt53_expert(0x6100_0000 + e as u64, true);
        let t = upload_u8(&mut gpu, &blob);
        ptrs.push(t.buf.as_ptr() as u64);
        keep.push(t);
    }
    let expert_ptrs = upload_u64(&mut gpu, &ptrs);
    let mut rng = Lcg(0xBE4C_0000);
    let gate: Vec<f32> = (0..K_TOP * MI).map(|_| rng.f32_signed() * 2.0).collect();
    let up: Vec<f32> = (0..K_TOP * MI).map(|_| rng.f32_signed() * 2.0).collect();
    let weights: Vec<f32> = vec![0.125; K_TOP];
    let mut topk_all: Vec<f32> = Vec::new();
    for _ in 0..ITERS {
        let mut sel: Vec<u32> = Vec::new();
        while sel.len() < K_TOP {
            let e = rng.next_u32() % POOL as u32;
            if !sel.contains(&e) {
                sel.push(e);
            }
        }
        topk_all.extend(sel.iter().map(|&e| f32::from_bits(e)));
    }
    let topk_t = upload_f32(&mut gpu, &topk_all);
    let gate_t = upload_f32(&mut gpu, &gate);
    let up_t = upload_f32(&mut gpu, &up);
    let w_t = upload_f32(&mut gpu, &weights);
    let rot_t = upload_f32(&mut gpu, &vec![0.0f32; K_TOP * MI]);
    let expanded_t = upload_f32(&mut gpu, &vec![0.0f32; K_TOP * DOWN_M]);
    let out_ref = upload_f32(&mut gpu, &vec![0.0f32; DOWN_M]);
    let out_new = upload_f32(&mut gpu, &vec![0.0f32; DOWN_M]);
    // A 2048x512 shared-expert weight for the k512 pair.
    let sh_a = upload_u8(&mut gpu, &synth_mq4v2_rows(0x7711));
    let sh_x = upload_f32(&mut gpu, &vec![0.25f32; SH_K]);
    let sh_c = upload_f32(&mut gpu, &[0.5f32]);
    let sh_y1 = upload_f32(&mut gpu, &vec![0.0f32; SH_M]);
    let sh_y2 = upload_f32(&mut gpu, &vec![0.0f32; SH_M]);
    for it in 0..ITERS {
        let tk = topk_t.sub_offset(it * K_TOP, K_TOP);
        gpu.silu_mul_f32(&gate_t, &up_t, &rot_t).unwrap();
        gpu.rotate_x_mq_128_v2(&rot_t, &rot_t, MI, K_TOP).unwrap();
        gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
            &expert_ptrs,
            &tk,
            &rot_t,
            &expanded_t,
            DOWN_M,
            DOWN_K,
            1,
            POOL,
            K_TOP,
        )
        .unwrap();
        gpu.moe_down_combine_k8_batched(&expanded_t, &w_t, &out_ref, DOWN_M, K_TOP, 1)
            .unwrap();
        gpu.gemv_mq4g128v2_moe_ninepath_d4(
            &expert_ptrs,
            &tk,
            &w_t,
            &gate_t,
            &up_t,
            &out_new,
            DOWN_M,
            DOWN_K,
            POOL,
        )
        .unwrap();
        gpu.gemv_mq4g256v2_residual_sigmoid_scaled_k512_one_row(
            &sh_a, &sh_x, &sh_y1, &sh_c, SH_M, SH_K,
        )
        .unwrap();
        gpu.gemv_mq4g256v2_residual_sigmoid_scaled_k512_two_row(
            &sh_a, &sh_x, &sh_y2, &sh_c, SH_M, SH_K,
        )
        .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
}
