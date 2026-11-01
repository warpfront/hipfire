// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! T0 experiment (throwaway): gfx11 MQ4V2-Lloyd prefill twins.
//!
//! Differential of each LUT twin against its uniform BT kernel with an
//! identity codebook (must be bit-identical) plus wall-time ratio at
//! N=384/511/512 (plan docs/plans/2026-09-15-lloyd-gfx11-port.md §T0 gate:
//! rel-L2 vs uniform baseline within noise, twin within +10% of uniform).
//! Covers residual, gate/up, qkv and qkvza BT twins.
//!
//! Run: `cargo run -p rdna-compute --example tmp_lloyd_gfx11_t0`
//! with TheRock on PATH (see repo notes for `HIPFIRE_HIPCC_EXTRA_FLAGS`).

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const K: usize = 2048;
const ITERS: usize = 11;

fn f16_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

fn identity_lut() -> [u32; 8] {
    let mut lut = [0u32; 8];
    for q in 0..16 {
        lut[q / 2] |= (f16_bits(q as f32) as u32) << ((q % 2) * 16);
    }
    lut
}

/// Identity C16 codebook for the MMQ-LUT arm (level q holds value q):
/// code = round(16*(q-7.5)), i8 bytes packed LE — mirrors
/// `lloyd_lut_c16_from_levels` in hipfire-runtime.
fn identity_c16() -> [u32; 4] {
    let mut out = [0u32; 4];
    for q in 0..16 {
        let code = (16 * q as i32 - 120) as i8 as u8 as u32;
        out[q / 4] |= code << ((q % 4) * 8);
    }
    out
}

/// Centered identity LUT for Lloyd-vs-Lloyd comparisons: level q holds
/// f16(q-7.5), matching the production wire contract (`w = sc*C[q] + zp'`,
/// `C[q] = L[q]-7.5`) that the MMQ-LUT kernel hardcodes (sc/16 * code).
/// The plain [`identity_lut`] stays for uniform-vs-Lloyd sections, where
/// both arms must decode the same values under different formulas.
fn identity_lut_centered() -> [u32; 8] {
    let mut lut = [0u32; 8];
    for q in 0..16 {
        lut[q / 2] |= (f16_bits(q as f32 - 7.5) as u32) << ((q % 2) * 16);
    }
    lut
}

/// Synthetic uniform MQ4V2 weight: sc=1, zp=0 both halves, random nibbles.
fn synth_weight(gpu: &mut Gpu, rows: usize, kk: usize, seed: u64) -> rdna_compute::GpuTensor {
    let groups = kk / 256;
    let row_bytes = groups * 136;
    let mut w = vec![0u8; rows * row_bytes];
    let sc_bits = f16_bits(1.0);
    let zp_bits = f16_bits(0.0);
    let mut s = seed;
    let mut rnd = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };
    for r in 0..rows {
        let base = r * row_bytes;
        for g in 0..groups {
            let o = base + g * 136;
            w[o..o + 2].copy_from_slice(&sc_bits.to_le_bytes());
            w[o + 2..o + 4].copy_from_slice(&zp_bits.to_le_bytes());
            w[o + 4..o + 6].copy_from_slice(&sc_bits.to_le_bytes());
            w[o + 6..o + 8].copy_from_slice(&zp_bits.to_le_bytes());
            for i in 0..128 {
                w[o + 8 + i] = (rnd() & 0xFF) as u8;
            }
        }
    }
    let t = gpu.alloc_tensor(&[w.len()], DType::Raw).expect("alloc w");
    gpu.hip.memcpy_htod(&t.buf, &w).expect("upload w");
    t
}

fn f32_input(gpu: &mut Gpu, n: usize, kk: usize, seed: u64) -> rdna_compute::GpuTensor {
    let mut xb = Vec::with_capacity(n * kk * 4);
    let mut s = seed;
    for _ in 0..n * kk {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        xb.extend_from_slice(&v.to_le_bytes());
    }
    let t = gpu.alloc_tensor(&[n * kk], DType::F32).expect("alloc x");
    gpu.hip.memcpy_htod(&t.buf, &xb).expect("upload x");
    t
}

fn zero_f32(gpu: &mut Gpu, n: usize) -> rdna_compute::GpuTensor {
    let t = gpu.alloc_tensor(&[n], DType::F32).expect("alloc y");
    gpu.hip.memset(&t.buf, 0, n * 4).expect("zero y");
    t
}

fn report(gpu: &Gpu, tag: (&str, usize, usize), tu: f64, tl: f64, ab: (&[f32], &[f32])) {
    gpu.hip.device_synchronize().expect("sync");
    let (label, n, bt) = tag;
    let (a, b) = ab;
    assert_eq!(a.len(), b.len());
    let mut max_abs: f32 = 0.0;
    let (mut num, mut den) = (0f64, 0f64);
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        num += (a[i] - b[i]) as f64 * (a[i] - b[i]) as f64;
        den += a[i] as f64 * a[i] as f64;
    }
    let rel = (num / den).sqrt();
    eprintln!(
        "{label} N={n} bt={bt}: uniform {tu:.6}s lloyd {tl:.6}s delta {:+.1}% max_abs={max_abs:.3e} rel_l2={rel:.3e}",
        (tl / tu - 1.0) * 100.0
    );
}

/// BT-vs-MMQ variant of [`report`]: first arg is the F16-BT arm, second the
/// X128-MMQ-LUT arm. Same math, different question: delta is BT-vs-MMQ
/// (negative = BT faster), and the rel_l2 gets a screen verdict — the arms
/// legitimately differ by Q8-activation noise (~1e-3), an order more than
/// that means the routes disagree beyond quantization and needs a look
/// before any dispatch change.
fn report_btvmmq(gpu: &Gpu, label: &str, n: usize, tbt: f64, tmmq: f64, ab: (&[f32], &[f32])) {
    gpu.hip.device_synchronize().expect("sync");
    let (a, b) = ab;
    assert_eq!(a.len(), b.len());
    let mut max_abs: f32 = 0.0;
    let (mut num, mut den) = (0f64, 0f64);
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        num += (a[i] - b[i]) as f64 * (a[i] - b[i]) as f64;
        den += a[i] as f64 * a[i] as f64;
    }
    let rel = (num / den).sqrt();
    let verdict = if rel < 5e-3 {
        "SCREEN-PASS"
    } else {
        "SCREEN-CHECK"
    };
    eprintln!(
        "{label} N={n}: BT {tbt:.6}s MMQ {tmmq:.6}s delta {:+.1}% max_abs={max_abs:.3e} rel_l2={rel:.3e} {verdict}",
        (tbt / tmmq - 1.0) * 100.0
    );
}

/// Paired-round timing: round 1 runs uniform then lloyd, round 2 runs lloyd
/// then uniform, so DPM/thermal drift cancels instead of biasing one side.
/// Reports the slower side's delta to stay conservative.
fn time_paired(iters: usize, mut f: impl FnMut(bool)) -> (f64, f64) {
    let mut tu = 0.0;
    let mut tl = 0.0;
    for _ in 0..iters {
        let t0 = Instant::now();
        f(false);
        tu += t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        f(true);
        tl += t0.elapsed().as_secs_f64();
    }
    for _ in 0..iters {
        let t0 = Instant::now();
        f(true);
        tl += t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        f(false);
        tu += t0.elapsed().as_secs_f64();
    }
    (tu / (2 * iters) as f64, tl / (2 * iters) as f64)
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("t0 arch: {}", gpu.arch);
    let lut = identity_lut();
    let luts3 = [lut; 3];
    let luts4 = [lut; 4];
    let only_small = std::env::var_os("HIPFIRE_T0_ONLY_SMALL").is_some();

    // Residual: M=5120.
    if !only_small {
        {
            const M: usize = 5120;
            let w = synth_weight(&mut gpu, M, K, 0x12345678);
            for &n in &[384usize, 511, 512] {
                let x = f32_input(&mut gpu, n, K, 0xdeadbeef);
                for &bt in &[4usize, 6, 8] {
                    let yu = zero_f32(&mut gpu, n * M);
                    let yl = zero_f32(&mut gpu, n * M);
                    gpu.gemm_mq4g256v2_residual_wmma_gfx1100_bt(&w, &x, &yu, M, K, n, bt)
                        .expect("uniform warmup");
                    gpu.gemm_mq4g256v2_residual_wmma_gfx1100_bt_lloyd(
                        &w, &x, &yl, M, K, n, bt, lut,
                    )
                    .expect("lloyd warmup");
                    gpu.hip.device_synchronize().expect("sync");
                    let (tu, tl) = time_paired(ITERS, |is_lloyd: bool| {
                        if is_lloyd {
                            gpu.gemm_mq4g256v2_residual_wmma_gfx1100_bt_lloyd(
                                &w, &x, &yl, M, K, n, bt, lut,
                            )
                            .expect("lloyd");
                            gpu.hip.device_synchronize().expect("iter sync");
                        } else {
                            gpu.gemm_mq4g256v2_residual_wmma_gfx1100_bt(&w, &x, &yu, M, K, n, bt)
                                .expect("uniform");
                            gpu.hip.device_synchronize().expect("iter sync");
                        }
                    });
                    let a = gpu.download_f32(&yu).expect("dl u");
                    let b = gpu.download_f32(&yl).expect("dl l");
                    report(&gpu, ("residual", n, bt), tu, tl, (&a, &b));
                }
            }
        }

        // Gate/up: gate=up=5120.
        {
            const G: usize = 5120;
            const U: usize = 5120;
            let wg = synth_weight(&mut gpu, G, K, 0x1111);
            let wu = synth_weight(&mut gpu, U, K, 0x2222);
            for &n in &[384usize, 511, 512] {
                let x = f32_input(&mut gpu, n, K, 0xdeadbeef);
                for &bt in &[6usize, 12] {
                    let yg = zero_f32(&mut gpu, n * G);
                    let yu = zero_f32(&mut gpu, n * U);
                    let lg = zero_f32(&mut gpu, n * G);
                    let lu = zero_f32(&mut gpu, n * U);
                    gpu.gemm_gate_up_mq4g256v2_wmma_gfx1100_bt(
                        &wg, &wu, &x, &yg, &yu, G, U, K, n, bt,
                    )
                    .expect("uniform warmup");
                    gpu.gemm_gate_up_mq4g256v2_wmma_gfx1100_bt_lloyd(
                        &wg, &wu, &x, &lg, &lu, G, U, K, n, bt, lut, lut,
                    )
                    .expect("lloyd warmup");
                    gpu.hip.device_synchronize().expect("sync");
                    let (tu, tl) = time_paired(ITERS, |is_lloyd: bool| {
                        if is_lloyd {
                            gpu.gemm_gate_up_mq4g256v2_wmma_gfx1100_bt_lloyd(
                                &wg, &wu, &x, &lg, &lu, G, U, K, n, bt, lut, lut,
                            )
                            .expect("lloyd");
                            gpu.hip.device_synchronize().expect("iter sync");
                        } else {
                            gpu.gemm_gate_up_mq4g256v2_wmma_gfx1100_bt(
                                &wg, &wu, &x, &yg, &yu, G, U, K, n, bt,
                            )
                            .expect("uniform");
                            gpu.hip.device_synchronize().expect("iter sync");
                        }
                    });
                    let mut a = gpu.download_f32(&yg).expect("dl ug");
                    a.extend_from_slice(&gpu.download_f32(&yu).expect("dl uu"));
                    let mut b = gpu.download_f32(&lg).expect("dl lg");
                    b.extend_from_slice(&gpu.download_f32(&lu).expect("dl lu"));
                    report(&gpu, ("gate_up", n, bt), tu, tl, (&a, &b));
                }
            }
        }

        // QKV: q=k=v=1024 (small, keeps the harness fast; shape-agnostic check).
        {
            const Q: usize = 1024;
            let wq = synth_weight(&mut gpu, Q, K, 0x3333);
            let wk = synth_weight(&mut gpu, Q, K, 0x4444);
            let wv = synth_weight(&mut gpu, Q, K, 0x5555);
            for &n in &[384usize, 511, 512] {
                let x = f32_input(&mut gpu, n, K, 0xdeadbeef);
                for &bt in &[4usize, 12] {
                    let yq = zero_f32(&mut gpu, n * Q);
                    let yk = zero_f32(&mut gpu, n * Q);
                    let yv = zero_f32(&mut gpu, n * Q);
                    let lq = zero_f32(&mut gpu, n * Q);
                    let lk = zero_f32(&mut gpu, n * Q);
                    let lv = zero_f32(&mut gpu, n * Q);
                    gpu.gemm_qkv_mq4g256v2_wmma_gfx1100_bt(
                        &wq, &wk, &wv, &x, &yq, &yk, &yv, Q, Q, Q, K, n, bt,
                    )
                    .expect("uniform warmup");
                    gpu.gemm_qkv_mq4g256v2_wmma_gfx1100_bt_lloyd(
                        &wq, &wk, &wv, &x, &lq, &lk, &lv, Q, Q, Q, K, n, bt, luts3,
                    )
                    .expect("lloyd warmup");
                    gpu.hip.device_synchronize().expect("sync");
                    let (tu, tl) = time_paired(ITERS, |is_lloyd: bool| {
                        if is_lloyd {
                            gpu.gemm_qkv_mq4g256v2_wmma_gfx1100_bt_lloyd(
                                &wq, &wk, &wv, &x, &lq, &lk, &lv, Q, Q, Q, K, n, bt, luts3,
                            )
                            .expect("lloyd");
                            gpu.hip.device_synchronize().expect("iter sync");
                        } else {
                            gpu.gemm_qkv_mq4g256v2_wmma_gfx1100_bt(
                                &wq, &wk, &wv, &x, &yq, &yk, &yv, Q, Q, Q, K, n, bt,
                            )
                            .expect("uniform");
                            gpu.hip.device_synchronize().expect("iter sync");
                        }
                    });
                    let mut a = gpu.download_f32(&yq).expect("dl");
                    a.extend_from_slice(&gpu.download_f32(&yk).expect("dl"));
                    a.extend_from_slice(&gpu.download_f32(&yv).expect("dl"));
                    let mut b = gpu.download_f32(&lq).expect("dl");
                    b.extend_from_slice(&gpu.download_f32(&lk).expect("dl"));
                    b.extend_from_slice(&gpu.download_f32(&lv).expect("dl"));
                    report(&gpu, ("qkv", n, bt), tu, tl, (&a, &b));
                }
            }
        }

        // QKVZA: qkv=1024, z=beta=alpha=256.
        {
            const QM: usize = 1024;
            const Z: usize = 256;
            let wq = synth_weight(&mut gpu, QM, K, 0x6666);
            let wz = synth_weight(&mut gpu, Z, K, 0x7777);
            let wb = synth_weight(&mut gpu, Z, K, 0x8888);
            let wa = synth_weight(&mut gpu, Z, K, 0x9999);
            for &n in &[384usize, 511, 512] {
                let x = f32_input(&mut gpu, n, K, 0xdeadbeef);
                for &bt in &[4usize, 12] {
                    let yq = zero_f32(&mut gpu, n * QM);
                    let yz = zero_f32(&mut gpu, n * Z);
                    let yb = zero_f32(&mut gpu, n * Z);
                    let ya = zero_f32(&mut gpu, n * Z);
                    let lq = zero_f32(&mut gpu, n * QM);
                    let lz = zero_f32(&mut gpu, n * Z);
                    let lb = zero_f32(&mut gpu, n * Z);
                    let la = zero_f32(&mut gpu, n * Z);
                    gpu.gemm_qkvza_mq4g256v2_wmma_gfx1100_bt(
                        &wq, &wz, &wb, &wa, &x, &yq, &yz, &yb, &ya, QM, Z, Z, Z, K, n, bt,
                    )
                    .expect("uniform warmup");
                    gpu.gemm_qkvza_mq4g256v2_wmma_gfx1100_bt_lloyd(
                        &wq, &wz, &wb, &wa, &x, &lq, &lz, &lb, &la, QM, Z, Z, Z, K, n, bt, luts4,
                    )
                    .expect("lloyd warmup");
                    gpu.hip.device_synchronize().expect("sync");
                    let (tu, tl) = time_paired(ITERS, |is_lloyd: bool| {
                        if is_lloyd {
                            gpu.gemm_qkvza_mq4g256v2_wmma_gfx1100_bt_lloyd(
                                &wq, &wz, &wb, &wa, &x, &lq, &lz, &lb, &la, QM, Z, Z, Z, K, n, bt,
                                luts4,
                            )
                            .expect("lloyd");
                            gpu.hip.device_synchronize().expect("iter sync");
                        } else {
                            gpu.gemm_qkvza_mq4g256v2_wmma_gfx1100_bt(
                                &wq, &wz, &wb, &wa, &x, &yq, &yz, &yb, &ya, QM, Z, Z, Z, K, n, bt,
                            )
                            .expect("uniform");
                            gpu.hip.device_synchronize().expect("iter sync");
                        }
                    });
                    let mut a = gpu.download_f32(&yq).expect("dl");
                    a.extend_from_slice(&gpu.download_f32(&yz).expect("dl"));
                    a.extend_from_slice(&gpu.download_f32(&yb).expect("dl"));
                    a.extend_from_slice(&gpu.download_f32(&ya).expect("dl"));
                    let mut b = gpu.download_f32(&lq).expect("dl");
                    b.extend_from_slice(&gpu.download_f32(&lz).expect("dl"));
                    b.extend_from_slice(&gpu.download_f32(&lb).expect("dl"));
                    b.extend_from_slice(&gpu.download_f32(&la).expect("dl"));
                    report(&gpu, ("qkvza", n, bt), tu, tl, (&a, &b));
                }
            }
        }
    } // end if !only_small

    // Small-N decider (real dense dims): uniform base vs uniform BT4 vs
    // Lloyd BT1 at N=24. Settles whether the F16-BT structure (masking,
    // sb-indirection) or the LUT itself costs the end-to-end gap.
    {
        const M: usize = 5120;
        const KK: usize = 5120;
        const N: usize = 24;
        let lut1 = identity_lut();
        let w = synth_weight(&mut gpu, M, KK, 0xabcd);
        let x = f32_input(&mut gpu, N, KK, 0xdcba);
        // Zero all outputs (residual add accumulates). Correctness first:
        // one timed shot each is compared bit-wise below after the perf loop.
        let yu = zero_f32(&mut gpu, N * M);
        let yl = zero_f32(&mut gpu, N * M);
        let yb = zero_f32(&mut gpu, N * M);
        gpu.gemm_mq4g256v2_residual_wmma_base(&w, &x, &yu, M, KK, N)
            .expect("uniform warmup");
        gpu.gemm_mq4g256v2_residual_wmma_gfx1100_bt_lloyd(&w, &x, &yl, M, KK, N, 1, lut1)
            .expect("lloyd warmup");
        gpu.gemm_mq4g256v2_residual_wmma_base(&w, &x, &yb, M, KK, N)
            .expect("base warmup");
        gpu.hip.device_synchronize().expect("sync");
        let (tu, tl) = time_paired(ITERS, |is_lloyd: bool| {
            if is_lloyd {
                gpu.gemm_mq4g256v2_residual_wmma_gfx1100_bt_lloyd(&w, &x, &yl, M, KK, N, 1, lut1)
                    .expect("lloyd");
                gpu.hip.device_synchronize().expect("iter sync");
            } else {
                gpu.gemm_mq4g256v2_residual_wmma_base(&w, &x, &yu, M, KK, N)
                    .expect("uniform");
                gpu.hip.device_synchronize().expect("iter sync");
            }
        });
        let a = gpu.download_f32(&yu).expect("dl u");
        let b = gpu.download_f32(&yl).expect("dl l");
        report(&gpu, ("resid24-base", N, 1), tu, tl, (&a, &b));
        // Same grid, uniform arithmetic both arms: isolates BT structure cost.
        let ybt = zero_f32(&mut gpu, N * M);
        let ybu = zero_f32(&mut gpu, N * M);
        gpu.gemm_mq4g256v2_residual_wmma_gfx1100_bt(&w, &x, &ybt, M, KK, N, 4)
            .expect("bt warmup");
        gpu.gemm_mq4g256v2_residual_wmma_base(&w, &x, &ybu, M, KK, N)
            .expect("base warmup");
        gpu.hip.device_synchronize().expect("sync");
        let (tt, tb2) = time_paired(ITERS, |is_bt: bool| {
            if is_bt {
                gpu.gemm_mq4g256v2_residual_wmma_gfx1100_bt(&w, &x, &ybt, M, KK, N, 4)
                    .expect("bt");
                gpu.hip.device_synchronize().expect("iter sync");
            } else {
                gpu.gemm_mq4g256v2_residual_wmma_base(&w, &x, &ybu, M, KK, N)
                    .expect("base");
                gpu.hip.device_synchronize().expect("iter sync");
            }
        });
        let _ = (tt, tb2);
        eprintln!("residual N=24 structure check: uniform-bt4 {tt:.6}s uniform-base {tb2:.6}s");
        // Same BT4 tile both arms: isolates the LUT path cost at small N.
        let yl4 = zero_f32(&mut gpu, N * M);
        let yu4 = zero_f32(&mut gpu, N * M);
        gpu.gemm_mq4g256v2_residual_wmma_gfx1100_bt_lloyd(&w, &x, &yl4, M, KK, N, 4, lut1)
            .expect("lloyd-bt4 warmup");
        gpu.gemm_mq4g256v2_residual_wmma_gfx1100_bt(&w, &x, &yu4, M, KK, N, 4)
            .expect("uniform-bt4 warmup");
        gpu.hip.device_synchronize().expect("sync");
        let (tu4, tl4) = time_paired(ITERS, |is_lloyd: bool| {
            if is_lloyd {
                gpu.gemm_mq4g256v2_residual_wmma_gfx1100_bt_lloyd(&w, &x, &yl4, M, KK, N, 4, lut1)
                    .expect("lloyd");
                gpu.hip.device_synchronize().expect("iter sync");
            } else {
                gpu.gemm_mq4g256v2_residual_wmma_gfx1100_bt(&w, &x, &yu4, M, KK, N, 4)
                    .expect("uniform");
                gpu.hip.device_synchronize().expect("iter sync");
            }
        });
        let e = gpu.download_f32(&yl4).expect("dl l4");
        let f = gpu.download_f32(&yu4).expect("dl u4");
        let same4 = e == f;
        eprintln!(
            "residual N=24 bt4-vs-bt4: uniform {tu4:.6}s lloyd {tl4:.6}s bit-identical {same4}"
        );
    }

    // Large-N shootout (follow-up screen): Lloyd F16-BT via the auto-tile
    // `*_gfx11_lloyd` launchers (the v1 route, called here past its N<64
    // dispatch cap) vs Lloyd X128-MMQ-LUT (the current N>=64 production
    // route). Same synthetic weights, identity codebooks in both forms.
    // Fewer iters: N=2048 launches run seconds, not ms.
    {
        const ITERS_BIG: usize = 3;
        let c16 = identity_c16();
        // Centered codebook + largest available BT tile per family (the fair
        // large-N cell: the auto launcher would fall back to bt4/bt6 up here
        // because the frozen policy has no large-N rows for these twins).
        let lutc = identity_lut_centered();
        let luts3c = [lutc; 3];
        let luts4c = [lutc; 4];
        // Residual: M=5120, max tile bt8.
        {
            const M: usize = 5120;
            let w = synth_weight(&mut gpu, M, K, 0x5eed);
            for &n in &[512usize, 2048] {
                let x = f32_input(&mut gpu, n, K, 0xbeef);
                let yb = zero_f32(&mut gpu, n * M);
                let ym = zero_f32(&mut gpu, n * M);
                gpu.gemm_mq4g256v2_residual_wmma_gfx1100_bt_lloyd(&w, &x, &yb, M, K, n, 8, lutc)
                    .expect("bt warmup");
                gpu.gemm_hfq4g256_residual_mmq_lloyd(&w, &x, &ym, M, K, n, c16)
                    .expect("mmq warmup");
                gpu.hip.device_synchronize().expect("sync");
                let (tbt, tmmq) = time_paired(ITERS_BIG, |is_mmq: bool| {
                    if is_mmq {
                        gpu.gemm_hfq4g256_residual_mmq_lloyd(&w, &x, &ym, M, K, n, c16)
                            .expect("mmq");
                        gpu.hip.device_synchronize().expect("iter sync");
                    } else {
                        gpu.gemm_mq4g256v2_residual_wmma_gfx1100_bt_lloyd(
                            &w, &x, &yb, M, K, n, 8, lutc,
                        )
                        .expect("bt");
                        gpu.hip.device_synchronize().expect("iter sync");
                    }
                });
                let a = gpu.download_f32(&yb).expect("dl bt");
                let b = gpu.download_f32(&ym).expect("dl mmq");
                report_btvmmq(&gpu, "residual", n, tbt, tmmq, (&a, &b));
            }
        }
        // Gate/up: gate=up=5120.
        {
            const G: usize = 5120;
            const U: usize = 5120;
            let wg = synth_weight(&mut gpu, G, K, 0x1111);
            let wu = synth_weight(&mut gpu, U, K, 0x2222);
            for &n in &[512usize, 2048] {
                let x = f32_input(&mut gpu, n, K, 0xdeadbeef);
                let bg = zero_f32(&mut gpu, n * G);
                let bu = zero_f32(&mut gpu, n * U);
                let mg = zero_f32(&mut gpu, n * G);
                let mu = zero_f32(&mut gpu, n * U);
                gpu.gemm_gate_up_mq4g256v2_wmma_gfx1100_bt_lloyd(
                    &wg, &wu, &x, &bg, &bu, G, U, K, n, 12, lutc, lutc,
                )
                .expect("bt warmup");
                gpu.gemm_gate_up_mq4g256v2_mmq_lloyd(&wg, &wu, &x, &mg, &mu, G, U, K, n, c16, c16)
                    .expect("mmq warmup");
                gpu.hip.device_synchronize().expect("sync");
                let (tbt, tmmq) = time_paired(ITERS_BIG, |is_mmq: bool| {
                    if is_mmq {
                        gpu.gemm_gate_up_mq4g256v2_mmq_lloyd(
                            &wg, &wu, &x, &mg, &mu, G, U, K, n, c16, c16,
                        )
                        .expect("mmq");
                        gpu.hip.device_synchronize().expect("iter sync");
                    } else {
                        gpu.gemm_gate_up_mq4g256v2_wmma_gfx1100_bt_lloyd(
                            &wg, &wu, &x, &bg, &bu, G, U, K, n, 12, lutc, lutc,
                        )
                        .expect("bt");
                        gpu.hip.device_synchronize().expect("iter sync");
                    }
                });
                let mut a = gpu.download_f32(&bg).expect("dl btg");
                a.extend_from_slice(&gpu.download_f32(&bu).expect("dl btu"));
                let mut b = gpu.download_f32(&mg).expect("dl mmqg");
                b.extend_from_slice(&gpu.download_f32(&mu).expect("dl mmqu"));
                report_btvmmq(&gpu, "gate_up", n, tbt, tmmq, (&a, &b));
            }
        }
        // QKV: q=k=v=1024.
        {
            const Q: usize = 1024;
            let wq = synth_weight(&mut gpu, Q, K, 0x3333);
            let wk = synth_weight(&mut gpu, Q, K, 0x4444);
            let wv = synth_weight(&mut gpu, Q, K, 0x5555);
            for &n in &[512usize, 2048] {
                let x = f32_input(&mut gpu, n, K, 0xdeadbeef);
                let bq = zero_f32(&mut gpu, n * Q);
                let bk = zero_f32(&mut gpu, n * Q);
                let bv = zero_f32(&mut gpu, n * Q);
                let mq = zero_f32(&mut gpu, n * Q);
                let mk = zero_f32(&mut gpu, n * Q);
                let mv = zero_f32(&mut gpu, n * Q);
                gpu.gemm_qkv_mq4g256v2_wmma_gfx1100_bt_lloyd(
                    &wq, &wk, &wv, &x, &bq, &bk, &bv, Q, Q, Q, K, n, 12, luts3c,
                )
                .expect("bt warmup");
                gpu.gemm_qkv_mq4g256v2_mmq_lloyd(
                    &wq, &wk, &wv, &x, &mq, &mk, &mv, Q, Q, Q, K, n, c16, c16, c16,
                )
                .expect("mmq warmup");
                gpu.hip.device_synchronize().expect("sync");
                let (tbt, tmmq) = time_paired(ITERS_BIG, |is_mmq: bool| {
                    if is_mmq {
                        gpu.gemm_qkv_mq4g256v2_mmq_lloyd(
                            &wq, &wk, &wv, &x, &mq, &mk, &mv, Q, Q, Q, K, n, c16, c16, c16,
                        )
                        .expect("mmq");
                        gpu.hip.device_synchronize().expect("iter sync");
                    } else {
                        gpu.gemm_qkv_mq4g256v2_wmma_gfx1100_bt_lloyd(
                            &wq, &wk, &wv, &x, &bq, &bk, &bv, Q, Q, Q, K, n, 12, luts3c,
                        )
                        .expect("bt");
                        gpu.hip.device_synchronize().expect("iter sync");
                    }
                });
                let mut a = gpu.download_f32(&bq).expect("dl");
                a.extend_from_slice(&gpu.download_f32(&bk).expect("dl"));
                a.extend_from_slice(&gpu.download_f32(&bv).expect("dl"));
                let mut b = gpu.download_f32(&mq).expect("dl");
                b.extend_from_slice(&gpu.download_f32(&mk).expect("dl"));
                b.extend_from_slice(&gpu.download_f32(&mv).expect("dl"));
                report_btvmmq(&gpu, "qkv", n, tbt, tmmq, (&a, &b));
            }
        }
        // QKVZA: qkv=1024, z=beta=alpha=256.
        {
            const QM: usize = 1024;
            const Z: usize = 256;
            let wq = synth_weight(&mut gpu, QM, K, 0x6666);
            let wz = synth_weight(&mut gpu, Z, K, 0x7777);
            let wb = synth_weight(&mut gpu, Z, K, 0x8888);
            let wa = synth_weight(&mut gpu, Z, K, 0x9999);
            for &n in &[512usize, 2048] {
                let x = f32_input(&mut gpu, n, K, 0xdeadbeef);
                let bq = zero_f32(&mut gpu, n * QM);
                let bz = zero_f32(&mut gpu, n * Z);
                let bb = zero_f32(&mut gpu, n * Z);
                let ba = zero_f32(&mut gpu, n * Z);
                let mq = zero_f32(&mut gpu, n * QM);
                let mz = zero_f32(&mut gpu, n * Z);
                let mb = zero_f32(&mut gpu, n * Z);
                let ma = zero_f32(&mut gpu, n * Z);
                gpu.gemm_qkvza_mq4g256v2_wmma_gfx1100_bt_lloyd(
                    &wq, &wz, &wb, &wa, &x, &bq, &bz, &bb, &ba, QM, Z, Z, Z, K, n, 12, luts4c,
                )
                .expect("bt warmup");
                gpu.gemm_qkvza_mq4g256v2_mmq_lloyd(
                    &wq, &wz, &wb, &wa, &x, &mq, &mz, &mb, &ma, QM, Z, Z, Z, K, n, c16, c16, c16,
                    c16,
                )
                .expect("mmq warmup");
                gpu.hip.device_synchronize().expect("sync");
                let (tbt, tmmq) = time_paired(ITERS_BIG, |is_mmq: bool| {
                    if is_mmq {
                        gpu.gemm_qkvza_mq4g256v2_mmq_lloyd(
                            &wq, &wz, &wb, &wa, &x, &mq, &mz, &mb, &ma, QM, Z, Z, Z, K, n, c16,
                            c16, c16, c16,
                        )
                        .expect("mmq");
                        gpu.hip.device_synchronize().expect("iter sync");
                    } else {
                        gpu.gemm_qkvza_mq4g256v2_wmma_gfx1100_bt_lloyd(
                            &wq, &wz, &wb, &wa, &x, &bq, &bz, &bb, &ba, QM, Z, Z, Z, K, n, 12,
                            luts4c,
                        )
                        .expect("bt");
                        gpu.hip.device_synchronize().expect("iter sync");
                    }
                });
                let mut a = gpu.download_f32(&bq).expect("dl");
                a.extend_from_slice(&gpu.download_f32(&bz).expect("dl"));
                a.extend_from_slice(&gpu.download_f32(&bb).expect("dl"));
                a.extend_from_slice(&gpu.download_f32(&ba).expect("dl"));
                let mut b = gpu.download_f32(&mq).expect("dl");
                b.extend_from_slice(&gpu.download_f32(&mz).expect("dl"));
                b.extend_from_slice(&gpu.download_f32(&mb).expect("dl"));
                b.extend_from_slice(&gpu.download_f32(&ma).expect("dl"));
                report_btvmmq(&gpu, "qkvza", n, tbt, tmmq, (&a, &b));
            }
        }
    }
}
