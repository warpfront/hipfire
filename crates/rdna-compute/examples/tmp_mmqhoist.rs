// SPDX-License-Identifier: Apache-2.0
// THROWAWAY slice-A harness for the gfx11 MMQ half-hoist (V1/V2).
// DELETE BEFORE COMMIT (plan slice A1/A2 owns it; never lands).
//
// Modes:
//   dump   --label L --out y.bin        run default path once, write Y f32 bin
//   parity --label L --ref y.bin        run, relL2/maxAbs vs ref bin + CPU f64 check
//   bench  --label L                    quant-once, 3 warm + 30 timed set calls
//
// --variant v1 : ensure_q8_1_mmq_x + gemm_mq4g256v2_mmq_set_prequant
//                (== compiled-in default: CURRENT in baseline tree, V1 in hoist tree)
// --variant v2 : x128 quantize + full_set_x128 via direct launch (hoist tree only)
// Shapes default M=8192 K=5120 N=512 (full_set path: M%128==0, N%128==0, K%256==0).

use hip_bridge::KernargBlob;
use rdna_compute::kv_slots::half_from_f32;
use rdna_compute::{Gpu, GpuTensor};
use std::ffi::c_void;
use std::time::Instant;

const GROUP: usize = 256;
const HALF: usize = 128;
const GROUP_BYTES: usize = 136;

const K_SRC: &str = include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq.hip");
const K_MODULE: &str = "gemm_mq4g256v2_residual_mmq";
const X128_Q: &str = "quantize_q8_1_mmq_ds4_x128";
const X128_SET: &str = "gemm_mq4g256v2_residual_mmq_full_set_x128";

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let mut exp = ((bits >> 10) & 0x1f) as u32;
    let mut mant = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            exp = 127 - 15 + 1;
            while mant & 0x0400 == 0 {
                mant <<= 1;
                exp -= 1;
            }
            sign | (exp << 23) | ((mant & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

/// Box-Muller normal from two uniform prng streams.
fn randn(i: usize) -> f32 {
    let u1 = prng(i, 0xA11CE_001).clamp(1e-9, 1.0 - 1e-9);
    let u2 = prng(i, 0xA11CE_002).clamp(0.0, 1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Synthetic weights: disjoint dual-fp16 headers (half0 ~ [-1,1], half1 ~
/// [96,160]) + zero-scale halves (rows %8==7: half1 constant; %13==12: half0 zero).
fn build_w(m: usize, k: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            let salt = (r * 7919 + g * 104_729) as u32;
            if r % 13 == 12 {
                for i in 0..HALF {
                    w[base + i] = 0.0;
                }
            } else {
                for i in 0..HALF {
                    w[base + i] = prng(base + i, salt) * 2.0 - 1.0;
                }
            }
            if r % 8 == 7 {
                for i in HALF..GROUP {
                    w[base + i] = 100.0;
                }
            } else {
                for i in HALF..GROUP {
                    w[base + i] = 96.0 + prng(base + i, salt ^ 0xA5A5_A5A5) * 64.0;
                }
            }
        }
    }
    w
}

/// MQ4V2 wire: 136 B/G256 = [fp16 s0,z0,s1,z1] + 128 B contiguous nibbles.
/// w = s*q + z per 128-half. Constant half -> s=0, codes stay 0.
fn pack_mq4g256v2(w: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % GROUP, 0);
    assert_eq!(w.len(), m * k);
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * GROUP_BYTES];
    for r in 0..m {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let dst = (r * gpr + g) * GROUP_BYTES;
            let mut codes = [0u8; GROUP];
            for h in 0..2 {
                let off = h * HALF;
                let slice = &w[src + off..src + off + HALF];
                let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let step = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
                let s_bits = if hi == lo { 0u16 } else { half_from_f32(step) };
                let z_bits = half_from_f32(lo);
                blob[dst + h * 4..dst + h * 4 + 2].copy_from_slice(&s_bits.to_le_bytes());
                blob[dst + h * 4 + 2..dst + h * 4 + 4].copy_from_slice(&z_bits.to_le_bytes());
                let s_rt = f16_to_f32(s_bits);
                let z_rt = f16_to_f32(z_bits);
                if s_rt == 0.0 {
                    continue;
                }
                let inv = 1.0 / s_rt;
                for i in 0..HALF {
                    let q = ((slice[i] - z_rt) * inv + 0.5).floor().clamp(0.0, 15.0);
                    codes[off + i] = q as u8;
                }
            }
            for i in 0..HALF {
                blob[dst + 8 + i] = (codes[2 * i] & 0xF) | ((codes[2 * i + 1] & 0xF) << 4);
            }
        }
    }
    blob
}

fn round_half_even(v: f32) -> f32 {
    let f = v.floor();
    let d = v - f;
    if d < 0.5 {
        f
    } else if d > 0.5 {
        f + 1.0
    } else if (f as i32) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

fn rel_rms(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in got.iter().zip(want.iter()) {
        let d = *x as f64 - *y as f64;
        num += d * d;
        den += (*y as f64) * (*y as f64);
    }
    if den == 0.0 {
        if num == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        (num / den).sqrt()
    }
}

/// CPU Q8_1 block model mirroring quantize_q8_1_mmq_ds4 (per-32 amax+sum),
/// then exact kernel block math in f32. `v1_assoc`: V1 (facc/sB) vs baseline
/// (per-block fma into acc) association. Returns acc for output rows 0..r_sub.
fn cpu_block_model(
    x: &[f32],
    blob: &[u8],
    m_sub: usize,
    m: usize,
    k: usize,
    n: usize,
    v1_assoc: bool,
) -> Vec<f32> {
    let gpr = k / GROUP;
    let mut y = vec![0.0f32; n * m_sub];
    for mm in 0..m_sub {
        for nn in 0..n {
            let mut acc = 0.0f32;
            for g in 0..gpr {
                let gp = (mm * gpr + g) * GROUP_BYTES;
                let s0 = f16_to_f32(u16::from_le_bytes([blob[gp], blob[gp + 1]]));
                let z0 = f16_to_f32(u16::from_le_bytes([blob[gp + 2], blob[gp + 3]]));
                let s1 = f16_to_f32(u16::from_le_bytes([blob[gp + 4], blob[gp + 5]]));
                let z1 = f16_to_f32(u16::from_le_bytes([blob[gp + 6], blob[gp + 7]]));
                for h in 0..2 {
                    let (sc, zp) = if h == 0 { (s0, z0) } else { (s1, z1) };
                    // quantize this 128-block per-32 sub-blocks
                    let mut dd = [0.0f32; 4];
                    let mut ss = [0.0f32; 4];
                    let mut qq = [[0i32; 32]; 4];
                    for b in 0..4 {
                        let mut amax = 0.0f32;
                        let mut s = 0.0f32;
                        for t in 0..32 {
                            let xi = x[nn * k + g * GROUP + h * HALF + b * 32 + t];
                            amax = amax.max(xi.abs());
                            s += xi;
                        }
                        let d = if amax > 1.0e-20 { amax / 127.0 } else { 0.0 };
                        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                        dd[b] = d;
                        ss[b] = s;
                        for t in 0..32 {
                            let xi = x[nn * k + g * GROUP + h * HALF + b * 32 + t];
                            qq[b][t] =
                                round_half_even((xi * id).max(-127.0).min(127.0)) as i32;
                        }
                    }
                    // weight nibbles for this half
                    let mut qw = [0i32; 128];
                    for t in 0..128 {
                        let byte = blob[gp + 8 + h * 64 + t / 2];
                        qw[t] = if t % 2 == 0 {
                            (byte & 0xF) as i32
                        } else {
                            ((byte >> 4) & 0xF) as i32
                        };
                    }
                    if v1_assoc {
                        // V1 association: facc/sB over the 4 blocks, then
                        // acc += sc*facc + zp*sB (f32, block order).
                        let mut f: f32 = 0.0;
                        let mut s_acc: f32 = 0.0;
                        for b in 0..4 {
                            let mut c: i64 = 0;
                            for t in 0..32 {
                                c += qw[b * 32 + t] as i64 * qq[b][t] as i64;
                            }
                            f = dd[b] * c as f32 + f;
                            s_acc += ss[b];
                        }
                        acc = sc * f + acc;
                        acc = zp * s_acc + acc;
                    } else {
                        for b in 0..4 {
                            let mut c: i64 = 0;
                            for t in 0..32 {
                                c += qw[b * 32 + t] as i64 * qq[b][t] as i64;
                            }
                            acc = sc * dd[b] * c as f32 + acc;
                            acc = zp * ss[b] + acc;
                        }
                    }
                }
            }
            y[nn * m_sub + mm] = acc;
        }
    }
    let _ = m;
    y
}

struct Cfg {
    variant: String,
    mode: String,
    label: String,
    out: String,
    rref: String,
    m: usize,
    k: usize,
    n: usize,
}

fn parse_args() -> Cfg {
    let mut c = Cfg {
        variant: "v1".into(),
        mode: "bench".into(),
        label: "x".into(),
        out: String::new(),
        rref: String::new(),
        m: 8192,
        k: 5120,
        n: 512,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--variant" => c.variant = it.next().expect("--variant value"),
            "--mode" => c.mode = it.next().expect("--mode value"),
            "--label" => c.label = it.next().expect("--label value"),
            "--out" => c.out = it.next().expect("--out value"),
            "--ref" => c.rref = it.next().expect("--ref value"),
            "--m" => c.m = it.next().expect("--m value").parse().unwrap(),
            "--k" => c.k = it.next().expect("--k value").parse().unwrap(),
            "--n" => c.n = it.next().expect("--n value").parse().unwrap(),
            x => panic!("unknown arg {x}"),
        }
    }
    c
}

/// Run the default compiled-in path (CURRENT in baseline tree, V1 in hoist tree).
fn run_default(
    gpu: &mut Gpu,
    d_a: &GpuTensor,
    d_x: &GpuTensor,
    d_y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
    quant_only: bool,
) -> (f64, f64) {
    let t0 = Instant::now();
    let xq = gpu.ensure_q8_1_mmq_x(d_x, n, k).expect("ensure_q8_1_mmq_x");
    gpu.hip.device_synchronize().unwrap();
    let q_us = t0.elapsed().as_secs_f64() * 1e6;
    if quant_only {
        return (q_us, 0.0);
    }
    let t1 = Instant::now();
    gpu.gemm_mq4g256v2_mmq_set_prequant(d_a, xq, d_y, m, k, n)
        .expect("set_prequant");
    gpu.hip.device_synchronize().unwrap();
    (q_us, t1.elapsed().as_secs_f64() * 1e6)
}

/// Run the V2 x128 path via direct launch into caller-owned Xq/Y.
fn run_v2(
    gpu: &mut Gpu,
    d_a: &GpuTensor,
    d_x: &GpuTensor,
    d_xq: &GpuTensor,
    d_y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> (f64, f64) {
    gpu.ensure_kernel_public(K_MODULE, K_SRC, X128_Q)
        .expect("ensure x128 quant");
    gpu.ensure_kernel_public(K_MODULE, K_SRC, X128_SET)
        .expect("ensure x128 set");
    let x_ptr = d_x.buf.as_ptr() as *const c_void;
    let xq_ptr = d_xq.buf.as_ptr() as *const c_void;
    let t0 = Instant::now();
    let mut kb = KernargBlob::new();
    kb.push_ptr(x_ptr);
    kb.push_ptr(xq_ptr);
    kb.push_i32(k as i32);
    kb.push_i32(n as i32);
    kb.pad_to(16);
    gpu.launch_kernel_blob(
        X128_Q,
        [((k + 1023) / 1024) as u32, n as u32, 1],
        [256, 1, 1],
        0,
        kb.as_mut_slice(),
    )
    .expect("launch x128 quant");
    gpu.hip.device_synchronize().unwrap();
    let q_us = t0.elapsed().as_secs_f64() * 1e6;
    let t1 = Instant::now();
    let mut kb = KernargBlob::new();
    kb.push_ptr(d_a.buf.as_ptr() as *const c_void);
    kb.push_ptr(xq_ptr);
    kb.push_ptr(d_y.buf.as_ptr() as *const c_void);
    kb.push_i32(m as i32);
    kb.push_i32(k as i32);
    kb.push_i32(n as i32);
    kb.push_i32(0);
    kb.pad_to(16);
    gpu.launch_kernel_blob(
        X128_SET,
        [(m / 128) as u32, (n / 128) as u32, 1],
        [32, 8, 1],
        ((128 * 36 + 128 * 76) * 4) as u32,
        kb.as_mut_slice(),
    )
    .expect("launch x128 set");
    gpu.hip.device_synchronize().unwrap();
    (q_us, t1.elapsed().as_secs_f64() * 1e6)
}

fn checksum(v: &[f32]) -> (u64, f32, f32) {
    let mut acc = 0u64;
    for &x in v {
        acc ^= (x.to_bits() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
    (acc, mn, mx)
}

fn main() {
    let cfg = parse_args();
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    assert!(
        matches!(arch.as_str(), "gfx1100" | "gfx1151"),
        "needs exact gfx1100/gfx1151, got {arch}"
    );
    assert_eq!(cfg.k % 256, 0);
    assert_eq!(cfg.m % 128, 0);
    assert_eq!(cfg.n % 128, 0);
    let (m, k, n) = (cfg.m, cfg.k, cfg.n);
    eprintln!(
        "HI label={} variant={} mode={} arch={} M={} K={} N={}",
        cfg.label, cfg.variant, cfg.mode, arch, m, k, n
    );

    let w = build_w(m, k);
    let blob = pack_mq4g256v2(&w, m, k);
    let x: Vec<f32> = (0..n * k).map(randn).collect();
    let d_a = gpu.upload_raw(&blob, &[blob.len()]).expect("upload A");
    let d_x = gpu.upload_f32(&x, &[n * k]).expect("upload X");
    let d_y = gpu
        .upload_f32(&vec![0.0f32; n * m], &[n * m])
        .expect("upload Y");
    gpu.hip.device_synchronize().unwrap();
    // V2-owned Xq buffer (also used as scratch check in v1 mode: untouched).
    let xq_bytes = (k / 128) * n * 144;
    let d_xq = gpu
        .upload_raw(&vec![0u8; xq_bytes], &[xq_bytes])
        .expect("upload Xq");

    let flop = 2.0 * m as f64 * n as f64 * k as f64;
    let run = |gpu: &mut Gpu| -> (f64, f64) {
        if cfg.variant == "v2" {
            run_v2(gpu, &d_a, &d_x, &d_xq, &d_y, m, k, n)
        } else {
            run_default(gpu, &d_a, &d_x, &d_y, m, k, n, false)
        }
    };

    match cfg.mode.as_str() {
        "dump" => {
            assert!(!cfg.out.is_empty(), "--out required for dump");
            let (q_us, s_us) = run(&mut gpu);
            let y = gpu.download_f32(&d_y).expect("download Y");
            assert_eq!(y.len(), n * m);
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(y.as_ptr() as *const u8, y.len() * 4)
            };
            std::fs::write(&cfg.out, bytes).expect("write bin");
            let (ck, mn, mx) = checksum(&y);
            eprintln!("DUMP label={} q_us={q_us:.1} set_us={s_us:.1} ck={ck:016x} min={mn:.4} max={mx:.4} y[0..4]={:?}", cfg.label, &y[..4]);
        }
        "parity" => {
            assert!(!cfg.rref.is_empty(), "--ref required for parity");
            let (q_us, s_us) = run(&mut gpu);
            let y = gpu.download_f32(&d_y).expect("download Y");
            let raw = std::fs::read(&cfg.rref).expect("read ref");
            assert_eq!(raw.len(), y.len() * 4);
            let r: &[f32] = unsafe {
                std::slice::from_raw_parts(raw.as_ptr() as *const f32, y.len())
            };
            let rr = rel_rms(&y, r);
            let mut mx = 0.0f32;
            let mut mn = f32::INFINITY;
            for (a, b) in y.iter().zip(r.iter()) {
                let d = (a - b).abs();
                if d > mx {
                    mx = d;
                }
                let denom = b.abs().max(1e-3);
                let rl = d / denom;
                if rl < mn {
                    mn = rl;
                }
            }
            let (ck, ymin, ymax) = checksum(&y);
            eprintln!("PARITY label={} q_us={q_us:.1} set_us={s_us:.1} relL2={rr:.3e} maxAbs={mx:.3e} ck={ck:016x} ymin={ymin:.4} ymax={ymax:.4}", cfg.label);
            // CPU block-model check on first 256 rows (same Q8_1 X algorithm).
            let m_sub = 256.min(m);
            let t = Instant::now();
            let yc = cpu_block_model(&x, &blob, m_sub, m, k, n, cfg.variant != "current");
            let dt = t.elapsed().as_secs_f64();
            let mut yg = vec![0.0f32; n * m_sub];
            for nn in 0..n {
                for mm in 0..m_sub {
                    yg[nn * m_sub + mm] = y[nn * m + mm];
                }
            }
            let rc = rel_rms(&yg, &yc);
            eprintln!("CPU label={} rows={m_sub} cpu_s={dt:.1} relL2_cpu={rc:.3e}", cfg.label);
        }
        "bench" => {
            // quant once outside the timed region (steady-state kernel timing)
            if cfg.variant == "v2" {
                let _ = run_v2(&mut gpu, &d_a, &d_x, &d_xq, &d_y, m, k, n);
            } else {
                let xq = gpu.ensure_q8_1_mmq_x(&d_x, n, k).expect("ensure");
                gpu.gemm_mq4g256v2_mmq_set_prequant(&d_a, xq, &d_y, m, k, n)
                    .expect("warm0");
            }
            gpu.hip.device_synchronize().unwrap();
            for _ in 0..3 {
                if cfg.variant == "v2" {
                    let _ = run_v2(&mut gpu, &d_a, &d_x, &d_xq, &d_y, m, k, n);
                } else {
                    let xq = gpu.ensure_q8_1_mmq_x(&d_x, n, k).expect("ensure");
                    gpu.gemm_mq4g256v2_mmq_set_prequant(&d_a, xq, &d_y, m, k, n)
                        .expect("warm");
                }
            }
            gpu.hip.device_synchronize().unwrap();
            // NOTE: warm loop re-quantizes each iter (ensure always launches);
            // the timed region below reuses ONE prelude for kernel-only timing.
            let xq = if cfg.variant == "v2" {
                std::ptr::null_mut()
            } else {
                gpu.ensure_q8_1_mmq_x(&d_x, n, k).expect("ensure")
            };
            // re-zero Y so `set` starts clean (full coverage anyway)
            let d_y2 = gpu
                .upload_f32(&vec![0.0f32; n * m], &[n * m])
                .expect("upload Y2");
            gpu.hip.device_synchronize().unwrap();
            let runs = 30;
            let t0 = Instant::now();
            for _ in 0..runs {
                if cfg.variant == "v2" {
                    let mut kb = KernargBlob::new();
                    kb.push_ptr(d_a.buf.as_ptr() as *const c_void);
                    kb.push_ptr(d_xq.buf.as_ptr() as *const c_void);
                    kb.push_ptr(d_y2.buf.as_ptr() as *const c_void);
                    kb.push_i32(m as i32);
                    kb.push_i32(k as i32);
                    kb.push_i32(n as i32);
                    kb.push_i32(0);
                    kb.pad_to(16);
                    gpu.launch_kernel_blob(
                        X128_SET,
                        [(m / 128) as u32, (n / 128) as u32, 1],
                        [32, 8, 1],
                        ((128 * 36 + 128 * 76) * 4) as u32,
                        kb.as_mut_slice(),
                    )
                    .expect("timed x128");
                } else {
                    gpu.gemm_mq4g256v2_mmq_set_prequant(&d_a, xq, &d_y2, m, k, n)
                        .expect("timed");
                }
            }
            gpu.hip.device_synchronize().unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / runs as f64;
            let tflops = flop / (us * 1e-6) / 1e12;
            let y = gpu.download_f32(&d_y2).expect("download Y");
            let (ck, ymin, ymax) = checksum(&y);
            eprintln!("BENCH label={} us={us:.2} tflops={tflops:.2} ck={ck:016x} ymin={ymin:.4} ymax={ymax:.4}", cfg.label);
        }
        x => panic!("unknown mode {x}"),
    }
}
