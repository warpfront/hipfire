// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! T2 differential (throwaway): Lloyd A8 twin vs X128-MMQ-LUT + per-128 oracle.
//!
//! Residual-only vehicle for §6.1' of
//! `docs/plans/local/2026-10-30-lloyd-mmq-iu4-lut-final.md`: the twin
//! (`gemm_mq4g256v2_mmq_add_prequant_i8_lloyd`, int8 X) against the incumbent
//! X128-MMQ-LUT arm (Q8 X, same identity codebook + headers) and against a
//! CPU per-128-int8 oracle (same d/s the GPU `quantize_int8_mmq_ds128`
//! producer emits, same fold order). Synthetic weights (sc=1, zp=0) are
//! uncentered on both arms identically — agreement is valid, absolute values
//! are not production numbers.
//!
//! Bars: twin-vs-oracle rel_l2 <= 1e-5 (kernel bug otherwise); twin-vs-X128 is
//! a coarse band (different X quant: Q8 vs int8) + paired timing, never the
//! 1e-5 gate.
//!
//! Run: `cargo run -p rdna-compute --example tmp_lloyd_i8_twin_t0`
//! with TheRock on PATH (see repo notes for `HIPFIRE_HIPCC_EXTRA_FLAGS`).

use hip_bridge::DeviceBuffer;
use rdna_compute::{DType, Gpu};
use std::time::Instant;

const K: usize = 2048;
const M: usize = 2048; // N=2048 -> t256 tiles (8x8=64 >= 48 CUs); smaller N -> tile128
const ITERS: usize = 3;

fn f16_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

/// Identity C16 codebook (level q holds value q): code = round(16*(q-7.5)).
fn identity_c16() -> [u32; 4] {
    let mut out = [0u32; 4];
    for q in 0..16 {
        let code = (16 * q as i32 - 120) as i8 as u8 as u32;
        out[q / 4] |= code << ((q % 4) * 8);
    }
    out
}

fn c16_codes(c16: [u32; 4]) -> [i8; 16] {
    let mut codes = [0i8; 16];
    for q in 0..16 {
        codes[q] = ((c16[q / 4] >> ((q % 4) * 8)) & 0xFF) as u8 as i8;
    }
    codes
}

/// Rotated identity codebook (level q holds value of level (q+5)%16):
/// proves the twin's gate/up table routing — a swap would fail the oracle.
fn identity_c16_rot() -> [u32; 4] {
    let mut out = [0u32; 4];
    for q in 0..16 {
        let code = (16 * ((q + 5) % 16) as i32 - 120) as i8 as u8 as u32;
        out[q / 4] |= code << ((q % 4) * 8);
    }
    out
}

/// Synthetic MQ4V2 weight: sc=1, zp=0 both halves, random nibbles.
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

fn host_input(n: usize, kk: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut x = Vec::with_capacity(n * kk);
    for _ in 0..n * kk {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        x.push((s >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0);
    }
    x
}

fn upload_f32(gpu: &mut Gpu, x: &[f32]) -> rdna_compute::GpuTensor {
    let t = gpu.alloc_tensor(&[x.len()], DType::F32).expect("alloc x");
    let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    gpu.hip.memcpy_htod(&t.buf, &bytes).expect("upload x");
    t
}

fn zero_f32(gpu: &mut Gpu, n: usize) -> rdna_compute::GpuTensor {
    let t = gpu.alloc_tensor(&[n], DType::F32).expect("alloc y");
    gpu.hip.memset(&t.buf, 0, n * 4).expect("zero y");
    t
}

/// CPU per-128-int8 oracle for the twin: same producer contract as
/// `quantize_int8_mmq_ds128` (amax, d=amax/127, q=clamp(rint(x/d)),
/// s=sum q), Lloyd decode (codes from c16, sc'=sc/16, zp' as stored),
/// fold per K128 in order: acc = fma(sc'*d, C, acc + zp'*d*s).
/// Output [N][M] token-major.
fn oracle_lloyd_i8(x: &[f32], w: &[u8], m: usize, k: usize, n: usize, codes: [i8; 16]) -> Vec<f32> {
    let kbn = k / 128;
    let gpr = k / 256;
    let row_bytes = gpr * 136;
    let mut y = vec![0f32; n * m];
    for t in 0..n {
        for r in 0..m {
            let mut acc = 0f32;
            for kb in 0..kbn {
                // X block.
                let src = &x[t * k + kb * 128..t * k + kb * 128 + 128];
                let amax = src.iter().fold(0f32, |a, &v| a.max(v.abs()));
                let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
                // Weight half + codes.
                let (g, h) = (kb / 2, kb % 2);
                let o = r * row_bytes + g * 136;
                let sc = f32::from(half::f16::from_bits(u16::from_le_bytes([w[o + 4 * h], w[o + 4 * h + 1]])));
                let zp = f32::from(half::f16::from_bits(u16::from_le_bytes([w[o + 4 * h + 2], w[o + 4 * h + 3]])));
                let scp = sc * 0.0625;
                let mut dot = 0i32;
                let mut s = 0i32;
                for (e, &v) in src.iter().enumerate() {
                    let qi = if amax == 0.0 {
                        0
                    } else {
                        (v / d).round_ties_even().clamp(-127.0, 127.0) as i32
                    };
                    s += qi;
                    let byte = w[o + 8 + h * 64 + e / 2];
                    let nib = if e % 2 == 0 { byte & 15 } else { byte >> 4 };
                    dot += codes[nib as usize] as i32 * qi;
                }
                if std::env::var_os("HIPFIRE_TWIN_DUMP").is_some() && t == 0 && r == 0 && kb == 0 {
                    eprintln!("orac kb0: amax={amax:.6} d={d:.8} s={s} dot={dot} sc={sc} zp={zp}");
                    eprintln!("orac x[0..4]: {:?}", &x[0..4]);
                    eprintln!("orac wbytes: {:02x?}", &w[0..16]);
                }
                acc = scp * d * dot as f32 + (acc + zp * d * s as f32);
            }
            y[t * m + r] = acc;
        }
    }
    y
}

/// CPU per-128-int8 oracle for the gate/up twin: same fold per arm, then
/// h = silu(g) * u in f32. Separate codebooks (proves table routing).
fn oracle_lloyd_i8_gateup(
    x: &[f32],
    wg: &[u8],
    wu: &[u8],
    m: usize,
    k: usize,
    n: usize,
    codes_g: [i8; 16],
    codes_u: [i8; 16],
) -> Vec<f32> {
    fn arm(
        x: &[f32],
        w: &[u8],
        m: usize,
        k: usize,
        n: usize,
        codes: [i8; 16],
    ) -> Vec<f32> {
        oracle_lloyd_i8(x, w, m, k, n, codes)
    }
    let g = arm(x, wg, m, k, n, codes_g);
    let u = arm(x, wu, m, k, n, codes_u);
    g.iter()
        .zip(u.iter())
        .map(|(&gv, &uv)| gv / (1.0 + (-gv).exp()) * uv)
        .collect()
}

fn rel_l2(a: &[f32], b: &[f32]) -> (f32, f64) {
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
    (max_abs, (num / den).sqrt())
}

fn time_paired(iters: usize, mut f: impl FnMut(bool)) -> (f64, f64) {
    let (mut t0v, mut t1v) = (0.0, 0.0);
    for _ in 0..iters {
        let t = Instant::now();
        f(false);
        t0v += t.elapsed().as_secs_f64();
        let t = Instant::now();
        f(true);
        t1v += t.elapsed().as_secs_f64();
    }
    for _ in 0..iters {
        let t = Instant::now();
        f(true);
        t1v += t.elapsed().as_secs_f64();
        let t = Instant::now();
        f(false);
        t0v += t.elapsed().as_secs_f64();
    }
    (t0v / (2 * iters) as f64, t1v / (2 * iters) as f64)
}

fn main() {
    // Box note: raw HIP ordinal 0 is the gfx1036 iGPU here (no KFD on native
    // Windows, so HIPFIRE_DEVICES/HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES do
    // not select; verified). The daemon resolves arch->ordinal itself; this
    // throwaway takes the ordinal from the env (default 0).
    let dev: i32 = std::env::var("HIPFIRE_T0_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(-1);
    let mut gpu = if dev >= 0 {
        Gpu::init_with_device(dev).expect("gpu init")
    } else {
        // Auto: first HIP ordinal whose arch is gfx1100 (Windows HIP order is
        // unstable across processes and env selection is unsupported).
        let mut found = None;
        let mut seen = Vec::new();
        for ordinal in 0..8 {
            match Gpu::init_with_device(ordinal) {
                Ok(g) => {
                    seen.push(format!("{ordinal}:{}", g.arch));
                    if g.arch == "gfx1100" {
                        found = Some(g);
                        break;
                    }
                }
                Err(e) => {
                    seen.push(format!("{ordinal}:err({e:?})"));
                    break;
                }
            }
        }
        eprintln!("seen devices: {}", seen.join(" "));
        found.expect("no gfx1100 among HIP ordinals")
    };
    eprintln!("twin-t0 arch: {}", gpu.arch);
    let c16 = if std::env::var_os("HIPFIRE_TWIN_C16ZERO").is_some() {
        [0u32; 4]
    } else {
        identity_c16()
    };
    let codes = c16_codes(c16);
    let w = synth_weight(&mut gpu, M, K, 0x5eed);
    // Host copy of the same weights for the oracle.
    let groups = K / 256;
    let row_bytes = groups * 136;
    let mut whost = vec![0u8; M * row_bytes];
    {
        let mut s = 0x5eedu64;
        let mut rnd = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u32
        };
        let sc_bits = f16_bits(1.0);
        let zp_bits = f16_bits(0.0);
        for r in 0..M {
            for g in 0..groups {
                let o = r * row_bytes + g * 136;
                whost[o..o + 2].copy_from_slice(&sc_bits.to_le_bytes());
                whost[o + 2..o + 4].copy_from_slice(&zp_bits.to_le_bytes());
                whost[o + 4..o + 6].copy_from_slice(&sc_bits.to_le_bytes());
                whost[o + 6..o + 8].copy_from_slice(&zp_bits.to_le_bytes());
                for i in 0..128 {
                    whost[o + 8 + i] = (rnd() & 0xFF) as u8;
                }
            }
        }
    }
    let mut fails = 0;
    let only_n: Option<usize> = std::env::var("HIPFIRE_TWIN_N")
        .ok()
        .and_then(|v| v.parse().ok());
    for &n in &[128usize, 512, 2048] {
        if let Some(only) = only_n {
            if n != only {
                continue;
            }
        }
        let xh = host_input(n, K, 0xbeef);
        let x = upload_f32(&mut gpu, &xh);
        let yx = zero_f32(&mut gpu, n * M);
        let yt = zero_f32(&mut gpu, n * M);
        gpu.gemm_hfq4g256_residual_mmq_lloyd(&w, &x, &yx, M, K, n, c16)
            .expect("x128 warmup");
        let xq = gpu.ensure_int8_mmq_x(&x, n, K).expect("int8 x");
        gpu.gemm_mq4g256v2_mmq_add_prequant_i8_lloyd(&w, xq, &yt, M, K, n, c16)
            .expect("twin warmup");
        gpu.hip.device_synchronize().expect("sync");
        let (tx, tt) = time_paired(ITERS, |is_twin: bool| {
            if is_twin {
                gpu.hip.memset(&yt.buf, 0, n * M * 4).expect("zero y");
                let xq = gpu.ensure_int8_mmq_x(&x, n, K).expect("int8 x");
                gpu.gemm_mq4g256v2_mmq_add_prequant_i8_lloyd(&w, xq, &yt, M, K, n, c16)
                    .expect("twin");
                gpu.hip.device_synchronize().expect("iter sync");
            } else {
                gpu.hip.memset(&yx.buf, 0, n * M * 4).expect("zero y");
                gpu.gemm_hfq4g256_residual_mmq_lloyd(&w, &x, &yx, M, K, n, c16)
                    .expect("x128");
                gpu.hip.device_synchronize().expect("iter sync");
            }
        });
        let a = gpu.download_f32(&yx).expect("dl x128");
        let b = gpu.download_f32(&yt).expect("dl twin");
        let o = oracle_lloyd_i8(&xh, &whost, M, K, n, codes);
        let (max_to, rel_to) = rel_l2(&b, &o);
        let (max_tx, rel_tx) = rel_l2(&b, &a);
        if std::env::var_os("HIPFIRE_TWIN_DUMP").is_some() {
            eprintln!("twin[0..6]: {:?}", &b[0..6]);
            eprintln!("orac[0..6]: {:?}", &o[0..6]);
            eprintln!("x128[0..6]: {:?}", &a[0..6]);
            for t in [0usize, 64] {
                for r in [0usize, 2, 130, 500] {
                    eprintln!(
                        "t={t} r={r} twin/orac/x128: {:?} / {:?} / {:?} (ratio {:.4})",
                        b[t * M + r],
                        o[t * M + r],
                        a[t * M + r],
                        b[t * M + r] / o[t * M + r]
                    );
                }
            }
            // Sidecar ground truth: download the GPU-produced block_i8_128
            // for (t=0, kb=0) and compare (d, s, qs) against the oracle.
            let xq = gpu.ensure_int8_mmq_x(&x, n, K).expect("int8 x");
            let nbytes = (K / 128) * n * 136;
            let view = unsafe { DeviceBuffer::from_raw(xq as *mut std::ffi::c_void, nbytes) };
            let mut raw = vec![0u8; nbytes];
            gpu.hip.memcpy_dtoh(&mut raw, &view).expect("dl sidecar");
            std::mem::forget(view);
            let d_gpu = f32::from_le_bytes(raw[0..4].try_into().unwrap());
            let s_gpu = i32::from_le_bytes(raw[4..8].try_into().unwrap());
            eprintln!("sidecar blk(t=0,kb=0): d_gpu={d_gpu:.8} s_gpu={s_gpu}");
            // Oracle recompute for the same block.
            let src0 = &xh[0..128];
            let amax0 = src0.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let d0 = if amax0 == 0.0 { 0.0 } else { amax0 / 127.0 };
            let mut s0 = 0i32;
            let mut q0 = [0i8; 128];
            for (e, &v) in src0.iter().enumerate() {
                let qi = if amax0 == 0.0 {
                    0
                } else {
                    (v / d0).round_ties_even().clamp(-127.0, 127.0) as i32
                };
                s0 += qi;
                q0[e] = qi as i8;
            }
            eprintln!("oracle blk(t=0,kb=0): d_or={d0:.8} s_or={s0}");
            // Unshuffle sidecar nibble-pair order to natural order.
            let mut mism = 0;
            for g in 0..16 {
                for i in 0..8 {
                    let sq = raw[8 + 8 * g + i] as i8;
                    let e = 8 * g + if i < 4 { 2 * i } else { 2 * (i - 4) + 1 };
                    if sq != q0[e] {
                        if mism < 4 {
                            eprintln!("qs mismatch e={e}: gpu={sq} oracle={}", q0[e]);
                        }
                        mism += 1;
                    }
                }
            }
            eprintln!("qs mismatches: {mism}/128");
        }
        let v_to = if rel_to <= 1e-5 { "TWIN-PASS" } else { fails += 1; "TWIN-CHECK" };
        let v_tx = if rel_tx < 5e-2 { "BAND-PASS" } else { "BAND-CHECK" };
        eprintln!(
            "residual N={n}: twin {tt:.6}s x128 {tx:.6}s delta {:+.1}% | twin-vs-oracle max_abs={max_to:.3e} rel_l2={rel_to:.3e} {v_to} | twin-vs-x128 rel_l2={rel_tx:.3e} {v_tx}",
            (tt / tx - 1.0) * 100.0
        );
    }
    if fails > 0 {
        eprintln!("TWIN-CHECK on {fails} shape(s)");
        std::process::exit(1);
    }

    // Gate/up SiLU twin (T3 vehicle): G=U=1024, dual codebooks (identity gate,
    // rotated up — a table swap fails the oracle).
    {
        const G: usize = 1024;
        let c16u = identity_c16_rot();
        let codes_u = c16_codes(c16u);
        let wg = synth_weight(&mut gpu, G, K, 0x1111);
        let wu = synth_weight(&mut gpu, G, K, 0x2222);
        let mut whost_g = vec![0u8; G * (K / 256) * 136];
        let mut whost_u = vec![0u8; G * (K / 256) * 136];
        for (dst, seed) in [(&mut whost_g, 0x1111u64), (&mut whost_u, 0x2222u64)] {
            let mut s = seed;
            let mut rnd = move || {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as u32
            };
            let sc_bits = f16_bits(1.0);
            let zp_bits = f16_bits(0.0);
            for r in 0..G {
                for ggg in 0..K / 256 {
                    let o = r * (K / 256) * 136 + ggg * 136;
                    dst[o..o + 2].copy_from_slice(&sc_bits.to_le_bytes());
                    dst[o + 2..o + 4].copy_from_slice(&zp_bits.to_le_bytes());
                    dst[o + 4..o + 6].copy_from_slice(&sc_bits.to_le_bytes());
                    dst[o + 6..o + 8].copy_from_slice(&zp_bits.to_le_bytes());
                    for i in 0..128 {
                        dst[o + 8 + i] = (rnd() & 0xFF) as u8;
                    }
                }
            }
        }
        for &n in &[128usize, 512] {
            if let Some(only) = only_n {
                if n != only {
                    continue;
                }
            }
            let xh = host_input(n, K, 0xdeadbeef);
            let x = upload_f32(&mut gpu, &xh);
            let hg = zero_f32(&mut gpu, n * G);
            let hu = zero_f32(&mut gpu, n * G);
            let ht = zero_f32(&mut gpu, n * G);
            gpu.gemm_gate_up_mq4g256v2_mmq_lloyd(&wg, &wu, &x, &hg, &hu, G, G, K, n, c16, c16u)
                .expect("x128 gate_up warmup");
            let xq = gpu.ensure_int8_mmq_x(&x, n, K).expect("int8 x");
            gpu.gemm_gate_up_silu_mq4g256v2_i8_lloyd(&wg, &wu, xq, &ht, G, K, n, c16, c16u)
                .expect("twin gate_up warmup");
            gpu.hip.device_synchronize().expect("sync");
            let (tx, tt) = time_paired(ITERS, |is_twin: bool| {
                if is_twin {
                    gpu.hip.memset(&ht.buf, 0, n * G * 4).expect("zero h");
                    let xq = gpu.ensure_int8_mmq_x(&x, n, K).expect("int8 x");
                    gpu.gemm_gate_up_silu_mq4g256v2_i8_lloyd(&wg, &wu, xq, &ht, G, K, n, c16, c16u)
                        .expect("twin");
                    gpu.hip.device_synchronize().expect("iter sync");
                } else {
                    gpu.hip.memset(&hg.buf, 0, n * G * 4).expect("zero g");
                    gpu.hip.memset(&hu.buf, 0, n * G * 4).expect("zero u");
                    gpu.gemm_gate_up_mq4g256v2_mmq_lloyd(&wg, &wu, &x, &hg, &hu, G, G, K, n, c16, c16u)
                        .expect("x128");
                    gpu.hip.device_synchronize().expect("iter sync");
                }
            });
            let ag = gpu.download_f32(&hg).expect("dl x128 g");
            let au = gpu.download_f32(&hu).expect("dl x128 u");
            let b = gpu.download_f32(&ht).expect("dl twin gu");
            let a: Vec<f32> = ag
                .iter()
                .zip(au.iter())
                .map(|(&gv, &uv)| gv / (1.0 + (-gv).exp()) * uv)
                .collect();
            let o = oracle_lloyd_i8_gateup(&xh, &whost_g, &whost_u, G, K, n, codes, codes_u);
            let (max_to, rel_to) = rel_l2(&b, &o);
            let (max_tx, rel_tx) = rel_l2(&b, &a);
            let v_to = if rel_to <= 1e-5 { "TWIN-PASS" } else { fails += 1; "TWIN-CHECK" };
            eprintln!(
                "gate_up N={n}: twin {tt:.6}s x128 {tx:.6}s delta {:+.1}% | twin-vs-oracle max_abs={max_to:.3e} rel_l2={rel_to:.3e} {v_to} | twin-vs-x128 rel_l2={rel_tx:.3e}",
                (tt / tx - 1.0) * 100.0
            );
        }
    }
    if fails > 0 {
        eprintln!("TWIN-CHECK on {fails} shape(s)");
        std::process::exit(1);
    }
}
