// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Node bench + numerics probe for the Raon grouped W4A4 expert GEMM
//! (`Gpu::gemm_mq4g256v2_moe_grouped_iu4_sym`, `kernels/src/moe_grouped_qt44_iu4_sym.gfx12.hip`) against the
//! shipped F16 grouped WMMA route (`Gpu::gemm_mq4g256v2_moe_grouped_wmma_k2` -> `..._wmma_gfx12`),
//! at the real Raon-1-mini-preview routed-expert shapes: 128 experts, top-6, gate/up `[2176, 2560]`,
//! down `[2560, 1280]` (both qt44, symmetric headers `zp = -8 sc`).
//!
//! Usage: `bench_moe_grouped_qt44_iu4 [--tokens 512,4096] [--reps 15] [--route uniform|skew] [--out F.json]`
//!
//! Per (tokens, projection) it reports median/min device time (device_synchronize bracketed, one
//! 379 MB / 223 MB weight set > the 96 MB LLC so every call streams) of the F16 route (with and without
//! its per-call F32->F16 activation convert), the activation quantizer, and every built IU4 entry; the
//! error of every IU4 entry against the F16 route's output at the live slots; the bitwise identity of
//! every entry against the first; and an exact CPU recomputation of sampled outputs from the downloaded
//! `block_i4_128` records (must be bit-equal to the kernel: same fold order and rounding).
//!
//! Weights are synthetic, symmetric by construction; activations are N(0, 1) F32 (what the rotated basis
//! feeds). The expert routing is a Gumbel-top-6 over lognormal popularities (`skew`, sigma 1) or equal
//! popularities (`uniform`). This is a node bench, not a model quality gate.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f32 {
        let (u1, u2) = (self.unit(), self.unit());
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

fn f32_to_f16(v: f32) -> u16 {
    // round-to-nearest-even, finite inputs only (the bench never produces others)
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7fffff;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = man | 0x800000;
        let shift = (14 - exp) as u32;
        let mut h = (m >> shift) as u16;
        let rem = m & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if rem > half || (rem == half && (h & 1) == 1) {
            h += 1;
        }
        return sign | h;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    let mut h = sign | ((exp as u16) << 10) | (man >> 13) as u16;
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    h
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign << 31
        } else {
            let mut e = -1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | (((e + 127 - 14 + 1 - 1 + 1) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        (sign << 31) | 0x7f800000 | (man << 13)
    } else {
        (sign << 31) | (((exp - 15 + 127) as u32) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// One qt44 weight matrix `[rows, k]`: 136 B per 256 K, symmetric headers.
fn build_qt44(rows: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let groups = k / 256;
    let mut out = vec![0u8; rows * groups * 136];
    for g in 0..rows * groups {
        let o = g * 136;
        for half in 0..2 {
            let sc = 0.004 + 0.012 * rng.unit() as f32;
            let sch = f32_to_f16(sc);
            let zph = f32_to_f16(-8.0 * f16_to_f32(sch));
            out[o + 4 * half..o + 4 * half + 2].copy_from_slice(&sch.to_le_bytes());
            out[o + 4 * half + 2..o + 4 * half + 4].copy_from_slice(&zph.to_le_bytes());
        }
        let mut i = 0;
        while i < 128 {
            let r = rng.next();
            out[o + 8 + i..o + 8 + i + 8].copy_from_slice(&r.to_le_bytes());
            i += 8;
        }
    }
    out
}

struct Routing {
    topk: Vec<i32>,
}

fn route(tokens: usize, experts: usize, top_k: usize, sigma: f32, seed: u64) -> Routing {
    let mut rng = Rng(seed);
    let logp: Vec<f32> = (0..experts).map(|_| sigma * rng.normal()).collect();
    let mut topk = Vec::with_capacity(tokens * top_k);
    for _ in 0..tokens {
        let mut sc: Vec<(f32, usize)> = (0..experts)
            .map(|e| (logp[e] - (-(rng.unit().ln())).ln() as f32, e))
            .collect();
        sc.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for s in sc.iter().take(top_k) {
            topk.push(s.1 as i32);
        }
    }
    Routing { topk }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_us<F: FnMut(&mut Gpu)>(gpu: &mut Gpu, reps: usize, mut f: F) -> (f64, f64) {
    for _ in 0..3 {
        f(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    let mut t = Vec::new();
    for _ in 0..reps {
        let t0 = Instant::now();
        f(gpu);
        gpu.hip.device_synchronize().unwrap();
        t.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    let mn = t.iter().cloned().fold(f64::MAX, f64::min);
    (median(&mut t), mn)
}

fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn dl_i32(gpu: &Gpu, t: &rdna_compute::GpuTensor) -> Vec<i32> {
    gpu.download_raw_bytes(t)
        .unwrap()
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

struct Stats {
    max_abs: f64,
    mean_abs: f64,
    rms_ref: f64,
    rms_err: f64,
}

fn compare(reference: &[f32], got: &[f32], m: usize, sorted: &[i32]) -> Stats {
    let (mut mx, mut sa, mut sr, mut se, mut n) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (pos, s) in sorted.iter().enumerate() {
        if *s < 0 {
            continue;
        }
        for r in 0..m {
            let (a, b) = (reference[pos * m + r] as f64, got[pos * m + r] as f64);
            let d = (a - b).abs();
            mx = mx.max(d);
            sa += d;
            sr += a * a;
            se += d * d;
            n += 1.0;
        }
    }
    Stats {
        max_abs: mx,
        mean_abs: sa / n,
        rms_ref: (sr / n).sqrt(),
        rms_err: (se / n).sqrt(),
    }
}

/// One benched expert-set shape family: `e` experts, `top_k` routing, and two GEMMs
/// (`gate_up`-style: x rows = tokens, x_row_div = top_k; `down`-style: x rows = slots, x_row_div = 1).
struct Site {
    name: &'static str,
    e: usize,
    top_k: usize,
    projs: [(&'static str, usize, usize); 2],
}

const SITES: [Site; 3] = [
    Site {
        name: "routed",
        e: 128,
        top_k: 6,
        projs: [("gate_up", 2176, 2560), ("down", 2560, 1280)],
    },
    // Single-expert sites run through the same grouped kernels with one expert (node proxy).
    Site {
        name: "shared",
        e: 1,
        top_k: 1,
        projs: [("gate_up", 1088, 2560), ("down", 2560, 1280)],
    },
    Site {
        name: "dense",
        e: 1,
        top_k: 1,
        projs: [("gate_up", 12032, 2560), ("down", 2560, 12032)],
    },
];

/// Finite e4m3 (OCP E4M3FN) code values; codes 0x7F / 0xFF are NaN and never produced.
fn e4m3_value(code: u8) -> f64 {
    let s = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((code >> 3) & 15) as i32;
    let m = (code & 7) as f64;
    s * if e == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
}

/// Round-to-nearest-even e4m3 of `v`, saturating at +-448 (the producer contract).
fn e4m3_encode_rne(mag: &[f64; 127], v: f32) -> u8 {
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0 };
    let a = (v.abs() as f64).min(448.0);
    let (mut lo, mut hi) = (0usize, 0x7Eusize);
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if mag[mid] <= a {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let c = if lo == 0x7E {
        lo
    } else {
        let (dl, dh) = (a - mag[lo], mag[lo + 1] - a);
        if dl < dh {
            lo
        } else if dl > dh {
            lo + 1
        } else if lo % 2 == 0 {
            lo
        } else {
            lo + 1
        }
    };
    sign | c as u8
}

/// Byte position of K element `i` (0..128) inside a 128-code block ("nibble-pair order").
fn pair_order(i: usize) -> usize {
    let (g, j) = (i / 8, i % 8);
    8 * g + if j % 2 == 0 { j / 2 } else { 4 + j / 2 }
}

struct TableRow {
    site: &'static str,
    proj: &'static str,
    tokens: usize,
    m: usize,
    k: usize,
    m_total: usize,
    f16_incl: f64,
    f16_gemm: f64,
    a8_quant: f64,
    a8: Vec<((usize, usize), f64)>,
    fp8_quant: f64,
    fp8: Vec<((usize, usize), f64)>,
}

fn best(v: &[((usize, usize), f64)]) -> ((usize, usize), f64) {
    v.iter()
        .cloned()
        .fold(((0, 0), f64::MAX), |b, x| if x.1 < b.1 { x } else { b })
}

fn at(v: &[((usize, usize), f64)], e: (usize, usize)) -> f64 {
    v.iter().find(|x| x.0 == e).map(|x| x.1).unwrap_or(f64::NAN)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str, d: &str| -> String {
        args.iter()
            .position(|a| a == k)
            .map(|i| args[i + 1].clone())
            .unwrap_or_else(|| d.into())
    };
    let tokens: Vec<usize> = arg("--tokens", "512,4096")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let reps: usize = arg("--reps", "15").parse().unwrap();
    let sigma = if arg("--route", "skew") == "uniform" {
        0.0
    } else {
        1.0
    };
    let out_path = arg("--out", "");
    let site_sel = arg("--sites", "routed");
    let run_a4 = arg("--a4", "1") == "1";

    let mut gpu = Gpu::init_with_device(0).unwrap();
    if gpu.arch != "gfx1201" {
        println!("skip: arch {} is not gfx1201", gpu.arch);
        return;
    }
    let mut mag = [0f64; 127];
    for (c, v) in mag.iter_mut().enumerate() {
        *v = e4m3_value(c as u8);
    }
    let dec: Vec<f64> = (0..256usize)
        .map(|c| {
            if c & 0x7F == 0x7F {
                f64::NAN
            } else {
                e4m3_value(c as u8)
            }
        })
        .collect();
    let mut results = Vec::new();
    let mut table: Vec<TableRow> = Vec::new();
    let mut fp8_fail = 0usize;

    for site in SITES.iter() {
        if !site_sel.split(',').any(|s| s == site.name) {
            continue;
        }
        let (e_n, top_k) = (site.e, site.top_k);
        let mut wrng = Rng(0x1234_5678_9abc_def1);
        let mut wt: [Vec<rdna_compute::GpuTensor>; 2] = [Vec::new(), Vec::new()];
        let mut wp: [Vec<u8>; 2] = [Vec::new(), Vec::new()];
        for _ in 0..e_n {
            for p in 0..2 {
                let (_, m, k) = site.projs[p];
                let g = build_qt44(m, k, &mut wrng);
                let t = gpu.upload_raw(&g, &[g.len()]).unwrap();
                wp[p].extend_from_slice(&(t.buf.as_ptr() as u64).to_le_bytes());
                wt[p].push(t);
            }
        }
        let ptr_t: Vec<rdna_compute::GpuTensor> = (0..2)
            .map(|p| gpu.upload_raw(&wp[p], &[wp[p].len()]).unwrap())
            .collect();

        for &rows in &tokens {
            let slots = rows * top_k;
            let bound = {
                let live = slots.min(e_n);
                (slots + live * 15).div_ceil(16) * 16
            };
            let rt = route(rows, e_n, top_k, sigma, 0x6D6F_655F_726F_7574);
            let topk = gpu.upload_raw(&i32s(&rt.topk), &[slots * 4]).unwrap();
            let counts = gpu.zeros(&[e_n], DType::F32).unwrap();
            let offsets = gpu.zeros(&[e_n + 1], DType::F32).unwrap();
            let sorted_t = gpu.zeros(&[bound], DType::F32).unwrap();
            let tiles_t = gpu.zeros(&[bound / 16], DType::F32).unwrap();
            let inv = gpu.zeros(&[slots], DType::F32).unwrap();
            gpu.moe_scatter_fused_k8(
                &topk, &counts, &offsets, &sorted_t, &tiles_t, &inv, slots, e_n, bound, 16,
            )
            .unwrap();
            let sorted = dl_i32(&gpu, &sorted_t);
            let tiles = dl_i32(&gpu, &tiles_t);
            let offs = dl_i32(&gpu, &offsets);
            let m_total = offs[e_n] as usize;
            let live_tiles = tiles.iter().filter(|&&t| t >= 0).count();
            println!("== site {} tokens {rows} slots {slots} bound {bound} m_total {m_total} live_tiles {live_tiles} sigma {sigma}", site.name);

            for p in 0..2 {
                let (proj, m, k) = site.projs[p];
                let ptrs = &ptr_t[p];
                let (x_row_div, x_rows) = if p == 0 {
                    (top_k, rows)
                } else {
                    (1usize, slots)
                };
                let mut xr = Rng(0xABCD_EF01 ^ rows as u64 ^ m as u64);
                let xf: Vec<f32> = (0..x_rows * k).map(|_| xr.normal()).collect();
                let x32 = gpu.upload_f32(&xf, &[x_rows, k]).unwrap();
                let xh: Vec<u16> = xf.iter().map(|&v| f32_to_f16(v)).collect();
                let x16 = gpu.upload_f16_bits(&xh, &[x_rows, k]).unwrap();
                let y_f16 = gpu.zeros(&[bound, m], DType::F32).unwrap();

                // F16 route, with and without its per-call convert.
                let (f16_med, f16_min) = time_us(&mut gpu, reps, |gpu| {
                    gpu.gemm_mq4g256v2_moe_grouped_wmma_k2(
                        ptrs, &tiles_t, &sorted_t, &x32, &y_f16, m, k, x_row_div, bound, x_rows,
                    )
                    .unwrap();
                });
                let (f16g_med, f16g_min) = time_us(&mut gpu, reps, |gpu| {
                    gpu.gemm_mq4g256v2_moe_grouped_wmma_k2(
                        ptrs, &tiles_t, &sorted_t, &x16, &y_f16, m, k, x_row_div, bound, x_rows,
                    )
                    .unwrap();
                });
                let yref = gpu.download_f32(&y_f16).unwrap();

                let mut first: Option<Vec<f32>> = None;
                let mut entry_rows = Vec::new();
                let (mut q_med, mut q_min) = (0.0, 0.0);
                let mut xq_host: Vec<u8> = Vec::new();
                if run_a4 {
                    // Activation quantizer.
                    (q_med, q_min) = time_us(&mut gpu, reps, |gpu| {
                        gpu.ensure_int4_mmq_x(&x32, x_rows, k).unwrap();
                    });
                    let xq = gpu.ensure_int4_mmq_x(&x32, x_rows, k).unwrap();
                    gpu.hip.device_synchronize().unwrap();
                    xq_host = vec![0u8; (k / 128) * x_rows * 72];
                    gpu.hip
                        .memcpy_dtoh(
                            &mut xq_host,
                            gpu.scratch.int4_mmq_x_scratch.as_ref().unwrap(),
                        )
                        .unwrap();
                    for &(nt, rf) in &[(1usize, 1usize), (2, 1), (4, 1), (1, 2), (2, 2)] {
                        if m % (64 * rf) != 0 {
                            continue;
                        }
                        let y = gpu.zeros(&[bound, m], DType::F32).unwrap();
                        let r = gpu.gemm_mq4g256v2_moe_grouped_iu4_sym(
                            ptrs, &tiles_t, &offsets, &sorted_t, xq, &y, m, k, x_row_div, bound,
                            x_rows, nt, rf,
                        );
                        if let Err(e) = r {
                            println!("  {proj} nt{nt}r{rf}: launch error {e:?}");
                            continue;
                        }
                        let (med, mn) = time_us(&mut gpu, reps, |gpu| {
                            gpu.gemm_mq4g256v2_moe_grouped_iu4_sym(
                                ptrs, &tiles_t, &offsets, &sorted_t, xq, &y, m, k, x_row_div,
                                bound, x_rows, nt, rf,
                            )
                            .unwrap();
                        });
                        let yv = gpu.download_f32(&y).unwrap();
                        let identical = match &first {
                            None => {
                                first = Some(yv.clone());
                                true
                            }
                            Some(f) => {
                                f.iter().zip(yv.iter()).enumerate().all(|(i, (a, b))| {
                                    sorted[i / m] < 0 || a.to_bits() == b.to_bits()
                                })
                            }
                        };
                        let st = compare(&yref, &yv, m, &sorted);
                        let flops = 2.0 * (m_total as f64) * m as f64 * k as f64;
                        println!(
                            "  {proj} nt{nt}r{rf}: med {med:.0} us min {mn:.0} us  ({:.1} TOPS on m_total)  vs f16: max|d| {:.4e} mean|d| {:.4e} rms_ref {:.4e} rms_err/rms_ref {:.4} bit-identical-to-first {identical}",
                            flops / med / 1e6,
                            st.max_abs, st.mean_abs, st.rms_ref, st.rms_err / st.rms_ref
                        );
                        entry_rows.push(serde_json::json!({
                            "nt": nt, "rf": rf, "med_us": med, "min_us": mn, "max_abs": st.max_abs, "mean_abs": st.mean_abs,
                            "rms_ref": st.rms_ref, "rms_err": st.rms_err, "identical_to_first": identical,
                        }));
                    }
                }
                // W4A8 twin: int8 activations, same grouped tables.
                let (q8_med, q8_min) = time_us(&mut gpu, reps, |gpu| {
                    gpu.ensure_int8_mmq_x(&x32, x_rows, k).unwrap();
                });
                let xq8 = gpu.ensure_int8_mmq_x(&x32, x_rows, k).unwrap();
                gpu.hip.device_synchronize().unwrap();
                let mut xq8_host = vec![0u8; (k / 128) * x_rows * 136];
                gpu.hip
                    .memcpy_dtoh(
                        &mut xq8_host,
                        gpu.scratch.int8_mmq_x_scratch.as_ref().unwrap(),
                    )
                    .unwrap();
                let mut a8_first: Option<Vec<f32>> = None;
                let mut a8_rows = Vec::new();
                let mut a8_times: Vec<((usize, usize), f64)> = Vec::new();
                for &(nt, rf) in &rdna_compute::moe_grouped_qt44_iu4::A8_ENTRIES {
                    if m % (64 * rf) != 0 {
                        continue;
                    }
                    let y = gpu.zeros(&[bound, m], DType::F32).unwrap();
                    if let Err(e) = gpu.gemm_mq4g256v2_moe_grouped_a8_sym(
                        ptrs, &tiles_t, &offsets, &sorted_t, xq8, &y, m, k, x_row_div, bound,
                        x_rows, nt, rf,
                    ) {
                        println!("  {proj} a8 nt{nt}r{rf}: launch error {e:?}");
                        continue;
                    }
                    let (med, mn) = time_us(&mut gpu, reps, |gpu| {
                        gpu.gemm_mq4g256v2_moe_grouped_a8_sym(
                            ptrs, &tiles_t, &offsets, &sorted_t, xq8, &y, m, k, x_row_div, bound,
                            x_rows, nt, rf,
                        )
                        .unwrap();
                    });
                    let yv = gpu.download_f32(&y).unwrap();
                    let identical = match &a8_first {
                        None => {
                            a8_first = Some(yv.clone());
                            true
                        }
                        Some(f) => f
                            .iter()
                            .zip(yv.iter())
                            .enumerate()
                            .all(|(i, (a, b))| sorted[i / m] < 0 || a.to_bits() == b.to_bits()),
                    };
                    let st = compare(&yref, &yv, m, &sorted);
                    let ops = 2.0 * (m_total as f64) * m as f64 * k as f64;
                    println!(
                        "  {proj} A8 nt{nt}r{rf}: med {med:.0} us min {mn:.0} us  ({:.1} TOPS on m_total)  vs f16: max|d| {:.4e} mean|d| {:.4e} rms_ref {:.4e} rms_err/rms_ref {:.4} bit-identical-to-first {identical}",
                        ops / med / 1e6, st.max_abs, st.mean_abs, st.rms_ref, st.rms_err / st.rms_ref
                    );
                    a8_times.push(((nt, rf), med));
                    a8_rows.push(serde_json::json!({
                        "nt": nt, "rf": rf, "med_us": med, "min_us": mn, "max_abs": st.max_abs, "mean_abs": st.mean_abs,
                        "rms_ref": st.rms_ref, "rms_err": st.rms_err, "identical_to_first": identical,
                    }));
                }

                // W4FP8: e4m3 activations (overwrites the shared int8 sidecar scratch; the A8 host copy
                // above stays valid for the A8 CPU check).
                let (qf_med, qf_min) = time_us(&mut gpu, reps, |gpu| {
                    gpu.ensure_fp8_mmq_x(&x32, x_rows, k).unwrap();
                });
                let xqf = gpu.ensure_fp8_mmq_x(&x32, x_rows, k).unwrap();
                gpu.hip.device_synchronize().unwrap();
                let mut xqf_host = vec![0u8; (k / 128) * x_rows * 136];
                gpu.hip
                    .memcpy_dtoh(
                        &mut xqf_host,
                        gpu.scratch.int8_mmq_x_scratch.as_ref().unwrap(),
                    )
                    .unwrap();
                // Producer check: every block against the CPU contract (d bit-exact; codes compared by
                // decoded value so -0 == +0).
                let (mut pd_bad, mut pc_bad, mut pa_bad, mut p_blocks) =
                    (0usize, 0usize, 0usize, 0usize);
                for h in 0..k / 128 {
                    for t in 0..x_rows {
                        let o = (h * x_rows + t) * 136;
                        let blk = &xqf_host[o..o + 136];
                        let xs = &xf[t * k + h * 128..t * k + h * 128 + 128];
                        let amax = xs.iter().fold(0f32, |a, &v| a.max(v.abs()));
                        let d = amax / 448.0f32;
                        let zero = d == 0.0;
                        let dg = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
                        p_blocks += 1;
                        if dg.to_bits() != d.to_bits() {
                            pd_bad += 1;
                        }
                        if blk[4..8] != [0, 0, 0, 0] {
                            pa_bad += 1;
                        }
                        let mut bad = false;
                        for i in 0..128 {
                            let want = if zero {
                                0u8
                            } else {
                                e4m3_encode_rne(&mag, (xs[i] / d).clamp(-448.0, 448.0))
                            };
                            let got = blk[8 + pair_order(i)];
                            if dec[got as usize] != dec[want as usize] {
                                bad = true;
                            }
                        }
                        if bad {
                            pc_bad += 1;
                        }
                    }
                }
                println!(
                    "  {proj}: FP8 producer vs CPU contract: {p_blocks} blocks, d mismatches {pd_bad}, code-block mismatches {pc_bad}, aux!=0 {pa_bad}; FP8 quantize {qf_med:.0} us (min {qf_min:.0})"
                );
                if pd_bad + pc_bad + pa_bad != 0 {
                    fp8_fail += 1;
                }

                let mut fp8_first: Option<Vec<f32>> = None;
                let mut fp8_rows = Vec::new();
                let mut fp8_times: Vec<((usize, usize), f64)> = Vec::new();
                for &(nt, rf) in &rdna_compute::moe_grouped_qt44_iu4::FP8_ENTRIES {
                    if m % (64 * rf) != 0 {
                        continue;
                    }
                    let y = gpu.zeros(&[bound, m], DType::F32).unwrap();
                    if let Err(e) = gpu.gemm_mq4g256v2_moe_grouped_fp8_sym(
                        ptrs, &tiles_t, &offsets, &sorted_t, xqf, &y, m, k, x_row_div, bound,
                        x_rows, nt, rf,
                    ) {
                        println!("  {proj} fp8 nt{nt}r{rf}: launch error {e:?}");
                        fp8_fail += 1;
                        continue;
                    }
                    let (med, mn) = time_us(&mut gpu, reps, |gpu| {
                        gpu.gemm_mq4g256v2_moe_grouped_fp8_sym(
                            ptrs, &tiles_t, &offsets, &sorted_t, xqf, &y, m, k, x_row_div, bound,
                            x_rows, nt, rf,
                        )
                        .unwrap();
                    });
                    let yv = gpu.download_f32(&y).unwrap();
                    let identical = match &fp8_first {
                        None => {
                            fp8_first = Some(yv.clone());
                            true
                        }
                        Some(f) => f
                            .iter()
                            .zip(yv.iter())
                            .enumerate()
                            .all(|(i, (a, b))| sorted[i / m] < 0 || a.to_bits() == b.to_bits()),
                    };
                    let st = compare(&yref, &yv, m, &sorted);
                    let st8 = compare(a8_first.as_deref().unwrap_or(&yref), &yv, m, &sorted);
                    let ops = 2.0 * (m_total as f64) * m as f64 * k as f64;
                    println!(
                        "  {proj} FP8 nt{nt}r{rf}: med {med:.0} us min {mn:.0} us  ({:.1} TOPS on m_total)  vs f16: max|d| {:.4e} rms_err/rms_ref {:.4}  vs a8: max|d| {:.4e} rms_err/rms_ref {:.4}  bit-identical-to-first {identical}",
                        ops / med / 1e6, st.max_abs, st.rms_err / st.rms_ref, st8.max_abs, st8.rms_err / st8.rms_ref
                    );
                    fp8_times.push(((nt, rf), med));
                    fp8_rows.push(serde_json::json!({
                        "nt": nt, "rf": rf, "med_us": med, "min_us": mn, "max_abs": st.max_abs, "mean_abs": st.mean_abs,
                        "rms_ref": st.rms_ref, "rms_err": st.rms_err, "max_abs_vs_a8": st8.max_abs,
                        "rms_err_vs_a8": st8.rms_err, "identical_to_first": identical,
                    }));
                    if !identical {
                        fp8_fail += 1;
                    }
                }

                let wblob = |e: usize, row: usize, g: usize, buf: &mut Vec<u8>| {
                    let t = &wt[p][e];
                    let rb = (k / 256) * 136;
                    let mut tmp = vec![0u8; 136];
                    gpu.hip
                        .memcpy_dtoh_at(&mut tmp, &t.buf, row * rb + g * 136)
                        .unwrap();
                    buf.clear();
                    buf.extend_from_slice(&tmp);
                };
                let mut gb = Vec::new();
                // exact CPU check of sampled outputs (first A4 entry's values)
                let mut checked = 0usize;
                let mut exact = 0usize;
                if let Some(yf) = first.as_ref() {
                    let mut srng = Rng(99);
                    while checked < 200 {
                        let pos = (srng.next() as usize) % m_total;
                        if sorted[pos] < 0 {
                            continue;
                        }
                        let row = (srng.next() as usize) % m;
                        let e = tiles[pos / 16] as usize;
                        let xrow = sorted[pos] as usize / x_row_div;
                        let mut sum = 0f32;
                        for h in 0..k / 128 {
                            wblob(e, row, h / 2, &mut gb);
                            let sc = f16_to_f32(u16::from_le_bytes([
                                gb[4 * (h & 1)],
                                gb[4 * (h & 1) + 1],
                            ]));
                            let blk = &xq_host
                                [((h * x_rows + xrow) * 72)..((h * x_rows + xrow) * 72 + 72)];
                            let d = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
                            let mut c = 0i32;
                            for i in 0..128 {
                                let wb = gb[8 + 64 * (h & 1) + i / 2];
                                let wn = if i % 2 == 0 { wb & 15 } else { wb >> 4 } as i32 - 8;
                                let xb = blk[8 + i / 2];
                                let xn = if i % 2 == 0 { xb & 15 } else { xb >> 4 } as i32;
                                let xn = if xn >= 8 { xn - 16 } else { xn };
                                c += wn * xn;
                            }
                            sum = (sc * d).mul_add(c as f32, sum);
                        }
                        checked += 1;
                        if sum.to_bits() == yf[pos * m + row].to_bits() {
                            exact += 1;
                        }
                    }
                    println!(
                        "  {proj}: exact CPU recompute {exact}/{checked} sampled outputs bit-equal"
                    );
                }
                let mut a8_exact = (0usize, 0usize);
                {
                    let yf8 = a8_first.clone().unwrap_or_default();
                    let mut srng = Rng(77);
                    let (mut checked8, mut exact8) = (0usize, 0usize);
                    while checked8 < 200 && !yf8.is_empty() {
                        let pos = (srng.next() as usize) % m_total;
                        if sorted[pos] < 0 {
                            continue;
                        }
                        let row = (srng.next() as usize) % m;
                        let e = tiles[pos / 16] as usize;
                        let xrow = sorted[pos] as usize / x_row_div;
                        let mut sum = 0f32;
                        for h in 0..k / 128 {
                            wblob(e, row, h / 2, &mut gb);
                            let sc = f16_to_f32(u16::from_le_bytes([
                                gb[4 * (h & 1)],
                                gb[4 * (h & 1) + 1],
                            ]));
                            let o = (h * x_rows + xrow) * 136;
                            let blk = &xq8_host[o..o + 136];
                            let d = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
                            let mut c = 0i32;
                            for i in 0..128 {
                                let wb = gb[8 + 64 * (h & 1) + i / 2];
                                let wn = if i % 2 == 0 { wb & 15 } else { wb >> 4 } as i32 - 8;
                                c += wn * (blk[8 + pair_order(i)] as i8 as i32);
                            }
                            sum = (sc * d).mul_add(c as f32, sum);
                        }
                        checked8 += 1;
                        if sum.to_bits() == yf8[pos * m + row].to_bits() {
                            exact8 += 1;
                        }
                    }
                    println!("  {proj}: A8 exact CPU recompute {exact8}/{checked8} sampled outputs bit-equal; A8 quantize {q8_med:.0} us (min {q8_min:.0})");
                    a8_exact = (exact8, checked8);
                }
                // FP8: CPU emulation of the contract. C_h is the exact sum of the e4m3 x (u - 8)
                // products (f64, exact: dyadic span << 53 bits) rounded once to F32; the WMMA
                // accumulates in F32 in an unspecified internal order, so bit-equality is expected
                // only when every partial sum is F32-exact; the rest is reported as max|d|.
                let fp8_cpu: (usize, usize);
                let (
                    mut fp8_max_vs_cpu,
                    mut fp8_max_vs_f64,
                    mut fp8_max_a8_vs_f64,
                    mut fp8_max_ref,
                ) = (0f64, 0f64, 0f64, 0f64);
                {
                    let yff = fp8_first.clone().unwrap_or_default();
                    let mut srng = Rng(55);
                    let (mut ck, mut ex) = (0usize, 0usize);
                    while ck < 200 && !yff.is_empty() {
                        let pos = (srng.next() as usize) % m_total;
                        if sorted[pos] < 0 {
                            continue;
                        }
                        let row = (srng.next() as usize) % m;
                        let e = tiles[pos / 16] as usize;
                        let xrow = sorted[pos] as usize / x_row_div;
                        let mut sum = 0f32;
                        let mut ref64 = 0f64;
                        let mut a8ref64 = 0f64;
                        for h in 0..k / 128 {
                            wblob(e, row, h / 2, &mut gb);
                            let sc = f16_to_f32(u16::from_le_bytes([
                                gb[4 * (h & 1)],
                                gb[4 * (h & 1) + 1],
                            ]));
                            let o = (h * x_rows + xrow) * 136;
                            let blk = &xqf_host[o..o + 136];
                            let d = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
                            let blk8 = &xq8_host[o..o + 136];
                            let d8 = f32::from_le_bytes([blk8[0], blk8[1], blk8[2], blk8[3]]);
                            let mut c = 0f64;
                            let mut c8 = 0f64;
                            for i in 0..128 {
                                let wb = gb[8 + 64 * (h & 1) + i / 2];
                                let wn =
                                    (if i % 2 == 0 { wb & 15 } else { wb >> 4 } as i32 - 8) as f64;
                                c += wn * dec[blk[8 + pair_order(i)] as usize];
                                c8 += wn * (blk8[8 + pair_order(i)] as i8 as f64);
                            }
                            sum = (sc * d).mul_add(c as f32, sum);
                            ref64 += sc as f64 * d as f64 * c;
                            a8ref64 += sc as f64 * d8 as f64 * c8;
                        }
                        ck += 1;
                        let got = yff[pos * m + row];
                        if sum.to_bits() == got.to_bits() {
                            ex += 1;
                        }
                        fp8_max_ref = fp8_max_ref.max(ref64.abs());
                        fp8_max_vs_cpu = fp8_max_vs_cpu.max((sum as f64 - got as f64).abs());
                        fp8_max_vs_f64 = fp8_max_vs_f64.max((ref64 - got as f64).abs());
                        fp8_max_a8_vs_f64 = fp8_max_a8_vs_f64.max((ref64 - a8ref64).abs());
                    }
                    println!(
                        "  {proj}: FP8 CPU emulation {ex}/{ck} sampled outputs bit-equal; max|kernel-cpu_f32| {fp8_max_vs_cpu:.3e}, max|kernel-f64| {fp8_max_vs_f64:.3e}, max|f64_fp8-f64_a8| {fp8_max_a8_vs_f64:.3e} (max|ref| {fp8_max_ref:.3e})"
                    );
                    fp8_cpu = (ex, ck);
                    if ck > 0 && fp8_max_vs_f64 > 1e-5 * fp8_max_ref.max(1.0) {
                        fp8_fail += 1;
                    }
                }
                println!(
                    "  {proj}: F16 route {f16_med:.0} us (min {f16_min:.0}) incl. convert; gemm-only {f16g_med:.0} us (min {f16g_min:.0}); A4 quantize {q_med:.0} us (min {q_min:.0})"
                );
                table.push(TableRow {
                    site: site.name,
                    proj,
                    tokens: rows,
                    m,
                    k,
                    m_total,
                    f16_incl: f16_med,
                    f16_gemm: f16g_med,
                    a8_quant: q8_med,
                    a8: a8_times,
                    fp8_quant: qf_med,
                    fp8: fp8_times,
                });
                results.push(serde_json::json!({
                    "site": site.name, "tokens": rows, "proj": proj, "m": m, "k": k, "m_total": m_total,
                    "f16_incl_convert_us": f16_med, "f16_incl_convert_min_us": f16_min,
                    "f16_gemm_us": f16g_med, "f16_gemm_min_us": f16g_min, "quantize_us": q_med, "quantize_min_us": q_min,
                    "exact_cpu": [exact, checked], "entries": entry_rows,
                    "a8_quantize_us": q8_med, "a8_quantize_min_us": q8_min, "a8_exact_cpu": [a8_exact.0, a8_exact.1], "a8_entries": a8_rows,
                    "fp8_quantize_us": qf_med, "fp8_quantize_min_us": qf_min, "fp8_cpu_bit_equal": [fp8_cpu.0, fp8_cpu.1],
                    "fp8_max_abs_vs_cpu_f32": fp8_max_vs_cpu, "fp8_max_abs_vs_f64": fp8_max_vs_f64,
                    "fp8_producer_blocks": p_blocks, "fp8_producer_d_bad": pd_bad, "fp8_producer_code_bad": pc_bad,
                    "fp8_entries": fp8_rows,
                }));
            }
        }
    }

    println!("\n## node table (median us; GEMM only, quantize separate; default = select_entry_{{a8,fp8}})");
    println!("| site | proj | tokens | m | k | m_total | F16 incl.cvt | F16 gemm | A8 quant | A8 default | A8 best (entry) | FP8 quant | FP8 default | FP8 best (entry) |");
    println!("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|");
    for r in &table {
        let (a8e, a8t) = best(&r.a8);
        let (fpe, fpt) = best(&r.fp8);
        println!(
            "| {} | {} | {} | {} | {} | {} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} (nt{}r{}) | {:.0} | {:.0} | {:.0} (nt{}r{}) |",
            r.site, r.proj, r.tokens, r.m, r.k, r.m_total, r.f16_incl, r.f16_gemm, r.a8_quant,
            at(&r.a8, rdna_compute::moe_grouped_qt44_iu4::select_entry_a8(r.m_total, r.m)),
            a8t, a8e.0, a8e.1, r.fp8_quant,
            at(&r.fp8, rdna_compute::moe_grouped_qt44_iu4::select_entry_fp8(r.m_total, r.m)),
            fpt, fpe.0, fpe.1
        );
    }
    println!("FP8 {}", if fp8_fail == 0 { "PASS" } else { "FAIL" });
    if !out_path.is_empty() {
        std::fs::write(
            &out_path,
            serde_json::to_string_pretty(&serde_json::json!({"sigma": sigma, "results": results}))
                .unwrap(),
        )
        .unwrap();
    }
    if fp8_fail != 0 {
        std::process::exit(1);
    }
}
