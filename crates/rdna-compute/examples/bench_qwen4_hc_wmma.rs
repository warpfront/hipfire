// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Oracle + timing: the Qwen4 HC GEMMs on the F16 WMMA route against the
//! kernels the route replaces, on this GPU's arch.
//!
//! * BF16 projections (`gemm_bf16_xf32_f16_wmma_qwen4`: F32 -> F16 input
//!   conversion, model-lifetime F16 weight shadow, LDS WMMA GEMM) vs
//!   `gemm_bf16_xf32_multirow`, both vs an F64 CPU reference on sampled rows.
//! * HC read tail (`hyper_read_up_wmma`) vs `hyper_read_up_fused`.
//!
//! Neither pair is bit-exact (F16 operands, WMMA F32 summation order); the
//! tool reports max abs / rel error and the bit-identical fraction.
//!
//! `MODE=sweep TILES=64_128_32_64_k64_p_s4,...`: the split-K kernel's tiles
//! (gfx1201/gfx1151) against the production `gemm_bf16_xf16_f16_wmma`, F16 X.
//! `MODE=stress ITERS=n`: every iteration reruns the route and the WMMA read
//! on ragged shapes and counts outputs that differ bitwise from iteration 0
//! or leave the multirow reference's tolerance; run several processes at
//! once on one card.  With `CAPTURE=1` each case is also issued once under
//! hipGraph capture — the route must refuse there, leaving the caller's
//! multirow / fused fallback, which is captured — and the graph is replayed
//! every iteration against the eager fallback bitwise.  Exits 1 on any
//! mismatch.
//!
//! Run: cargo run --release --features lab --example bench_qwen4_hc_wmma -p rdna-compute
//! `SHAPES=MxKxB,...` (projections), `READS=HIDDENxROWS,...`, `REPS=n`.
//! On gfx1201 the route is opt-in: set `HIPFIRE_QWEN4_F16_WMMA_GFX1201=1`
//! (the gfx1201 legs of the `gemm`/`tensor_ops` route unit tests need it too).

use rdna_compute::gemm::LdsTileSplitK;
use rdna_compute::tensor_ops::{hyper_read_up_fused, hyper_read_up_wmma, HyperReadUpFused};
use rdna_compute::{DType, Gpu, GpuTensor};

fn lcg(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0) * scale
        })
        .collect()
}

fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_val(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// F16 bits of a BF16-exact value in the F16 range: exact for normals (7
/// mantissa bits fit in 10); below 2^-14 rounded to the nearest F16 subnormal
/// (ties to even), as the GPU's F32 -> F16 conversion does.
fn f16_of_bf16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    if b & 0x7fff_ffff == 0 {
        return sign;
    }
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    assert!(exp < 31, "value {v} outside the F16 range");
    if exp < 1 {
        // |v| < 2^-14: units of 2^-24 (exact in f64), round half to even.
        let q = (v.abs() as f64) * (1u64 << 24) as f64;
        let r = q.round_ties_even() as u16;
        return sign | r;
    }
    sign | ((exp as u16) << 10) | ((b >> 13) & 0x3ff) as u16
}

fn upload_u16(gpu: &mut Gpu, v: &[u16], dtype: DType) -> GpuTensor {
    let bytes: Vec<u8> = v.iter().flat_map(|h| h.to_le_bytes()).collect();
    let mut t = gpu.upload_raw(&bytes, &[v.len()]).expect("upload");
    t.dtype = dtype;
    t
}

fn time_ms(gpu: &mut Gpu, reps: usize, mut f: impl FnMut(&mut Gpu)) -> f64 {
    for _ in 0..3 {
        f(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..reps {
        f(gpu);
    }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 / reps as f64
}

fn parse(var: &str, default: &str) -> Vec<Vec<usize>> {
    std::env::var(var)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.split('x').map(|t| t.parse().unwrap()).collect())
        .collect()
}

fn projections(gpu: &mut Gpu, reps: usize) {
    for s in parse("SHAPES", "320x20480x1536,320x20480x512,320x10240x1900,320x10240x1131,4x20480x1536") {
        let (m, k, b) = (s[0], s[1], s[2]);
        let wb: Vec<u16> = lcg(m * k, 11, 0.05).iter().map(|&v| bf16_bits(v)).collect();
        let xr: Vec<f32> = lcg(b * k, 23, 2.0).iter().map(|&v| bf16_val(bf16_bits(v))).collect();
        let wr: Vec<f32> = wb.iter().map(|&h| bf16_val(h)).collect();
        let weight = upload_u16(gpu, &wb, DType::BF16);
        let x = gpu.upload_f32(&xr, &[b * k]).unwrap();
        let y_old = gpu.zeros(&[b * m], DType::F32).unwrap();
        let y_new = gpu.zeros(&[b * m], DType::F32).unwrap();
        let rows: Vec<usize> = (0..b).step_by((b / 24).max(1)).chain([b - 1]).collect();
        let reference: Vec<f64> = rows
            .iter()
            .flat_map(|&bi| {
                let (xr, wr) = (&xr, &wr);
                (0..m).map(move |mi| {
                    (0..k).map(|kk| wr[mi * k + kk] as f64 * xr[bi * k + kk] as f64).sum::<f64>()
                })
            })
            .collect();
        let max_ref = reference.iter().fold(0f64, |a, r| a.max(r.abs()));
        let err = |y: &[f32]| -> f64 {
            let mut worst = 0f64;
            for (ri, &bi) in rows.iter().enumerate() {
                for mi in 0..m {
                    worst = worst.max((y[bi * m + mi] as f64 - reference[ri * m + mi]).abs());
                }
            }
            worst
        };
        let flop = 2.0 * (m * k * b) as f64;
        let t_old = time_ms(gpu, reps, |g| {
            g.gemm_bf16_xf32_multirow(&weight, &x, &y_old, m, k, b).unwrap()
        });
        let mut admitted = false;
        let t_new = time_ms(gpu, reps, |g| {
            admitted = g
                .gemm_bf16_xf32_f16_wmma_qwen4(&[(&weight, &y_new, m)], &x, k, b)
                .unwrap()
        });
        let old = gpu.download_f32(&y_old).unwrap();
        let new = gpu.download_f32(&y_new).unwrap();
        let (e_old, e_new) = (err(&old), err(&new));
        let mut dmax = 0f64;
        let mut same = 0usize;
        let mut omax = 0f64;
        for i in 0..b * m {
            dmax = dmax.max((new[i] - old[i]).abs() as f64);
            omax = omax.max(old[i].abs() as f64);
            same += (new[i].to_bits() == old[i].to_bits()) as usize;
        }
        println!(
            "proj M{m:<5} K{k:<5} B{b:<4} admitted={admitted} multirow {t_old:7.3} ms ({:5.1} TF/s) | route {t_new:7.3} ms ({:5.1} TF/s) x{:5.2} | f64-ref max|y| {max_ref:.3e}: multirow abs {e_old:.2e} rel {:.2e}, route abs {e_new:.2e} rel {:.2e} | route-vs-multirow max abs {dmax:.3e} rel {:.2e} bit-identical {:.2}%",
            flop / t_old / 1e9,
            flop / t_new / 1e9,
            t_old / t_new,
            e_old / max_ref,
            e_new / max_ref,
            dmax / omax,
            100.0 * same as f64 / (b * m) as f64,
        );
    }
}

fn reads(gpu: &mut Gpu, reps: usize) {
    let low_rank = 320usize;
    for s in parse("READS", "5120x1536,5120x512,2560x1900,2560x1131") {
        let (hidden, rows) = (s[0], s[1]);
        let wide = 4 * hidden;
        let up: Vec<u16> = lcg(wide * low_rank, 5, 0.1).iter().map(|&v| bf16_bits(v)).collect();
        // hc_activation's output: silu of BF16 values, BF16-rounded.
        let low: Vec<f32> = lcg(rows * low_rank, 7, 3.0)
            .iter()
            .map(|&v| bf16_val(bf16_bits(v / (1.0 + (-v).exp()))))
            .collect();
        // hyper_norm's rows, BF16-rounded, kept inside the F16 normal range.
        let norm: Vec<f32> = lcg(rows * wide, 9, 2.0)
            .iter()
            .map(|&v| bf16_val(bf16_bits(if v.abs() < 1e-3 { 1e-3 } else { v })))
            .collect();
        let up_weight = upload_u16(gpu, &up, DType::BF16);
        let low_f32 = gpu.upload_f32(&low, &[rows * low_rank]).unwrap();
        let low_bf16 =
            upload_u16(gpu, &low.iter().map(|&v| bf16_bits(v)).collect::<Vec<_>>(), DType::BF16);
        let norm_f32 = gpu.upload_f32(&norm, &[rows * wide]).unwrap();
        let norm_f16 =
            upload_u16(gpu, &norm.iter().map(|&v| f16_of_bf16(v)).collect::<Vec<_>>(), DType::F16);
        let mixed_old = gpu.zeros(&[rows * hidden], DType::F32).unwrap();
        let mixed_new = gpu.zeros(&[rows * hidden], DType::F32).unwrap();
        let old = HyperReadUpFused {
            up_weight: &up_weight,
            low: &low_f32,
            normalized: &norm_f32,
            mixed: &mixed_old,
            rows,
            hidden,
            low_rank,
            normalized_bf16: false,
        };
        let new = HyperReadUpFused {
            up_weight: &up_weight,
            low: &low_bf16,
            normalized: &norm_f16,
            mixed: &mixed_new,
            rows,
            hidden,
            low_rank,
            normalized_bf16: false,
        };
        let t_old = time_ms(gpu, reps, |g| hyper_read_up_fused(g, &old).unwrap());
        let t_new = time_ms(gpu, reps, |g| hyper_read_up_wmma(g, &new).unwrap());
        let a = gpu.download_f32(&mixed_old).unwrap();
        let b = gpu.download_f32(&mixed_new).unwrap();
        let mut dmax = 0f64;
        let mut amax = 0f64;
        let mut differ = 0usize;
        let mut steps = 0u32;
        for i in 0..a.len() {
            dmax = dmax.max((a[i] - b[i]).abs() as f64);
            amax = amax.max(a[i].abs() as f64);
            if a[i].to_bits() != b[i].to_bits() {
                differ += 1;
                // Both are BF16 values: distance in BF16 steps.
                let (x, y) = ((a[i].to_bits() >> 16) as i32, (b[i].to_bits() >> 16) as i32);
                steps = steps.max((x - y).unsigned_abs());
            }
        }
        let flop = 2.0 * (wide * low_rank * rows) as f64;
        println!(
            "read hidden{hidden:<5} rows{rows:<5} fused_f32 {t_old:7.3} ms ({:5.1} TF/s) | wmma {t_new:7.3} ms ({:5.1} TF/s) x{:5.2} | max abs {dmax:.3e} rel {:.2e}, differing {:.3}%, max {steps} BF16 step(s)",
            flop / t_old / 1e9,
            flop / t_new / 1e9,
            t_old / t_new,
            dmax / amax,
            100.0 * differ as f64 / a.len() as f64,
        );
    }
}

/// `bm_bn_wm_wn_k64[_p][_sN]` -> tile.
fn tile_of(name: &str) -> LdsTileSplitK {
    let parts: Vec<&str> = name.split('_').collect();
    let n = |i: usize| parts[i].parse::<usize>().unwrap();
    let pipe = parts.contains(&"p");
    let split = parts
        .iter()
        .find_map(|p| p.strip_prefix('s').and_then(|v| v.parse().ok()))
        .unwrap_or(1);
    LdsTileSplitK::new(n(0), n(1), n(2), n(3), pipe, split)
}

fn sweep(gpu: &mut Gpu, reps: usize) {
    let tiles: Vec<LdsTileSplitK> = std::env::var("TILES")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(tile_of)
        .collect();
    for s in parse("SHAPES", "320x20480x1536,320x10240x1900,320x10240x1131") {
        let (m, k, b) = (s[0], s[1], s[2]);
        let wb: Vec<u16> = lcg(m * k, 11, 0.05).iter().map(|&v| bf16_bits(v)).collect();
        let xr: Vec<f32> = lcg(b * k, 23, 2.0).iter().map(|&v| bf16_val(bf16_bits(v))).collect();
        let weight = upload_u16(gpu, &wb, DType::BF16);
        let w16: Vec<u16> = wb.iter().map(|&h| f16_of_bf16(bf16_val(h))).collect();
        let w_f16 = upload_u16(gpu, &w16, DType::F16);
        let x_f16 = upload_u16(gpu, &xr.iter().map(|&v| f16_of_bf16(v)).collect::<Vec<_>>(), DType::F16);
        let y_prod = gpu.zeros(&[b * m], DType::F32).unwrap();
        let y = gpu.zeros(&[b * m], DType::F32).unwrap();
        let flop = 2.0 * (m * k * b) as f64;
        let t_prod = time_ms(gpu, reps, |g| {
            g.gemm_bf16_xf16_f16_wmma(&weight, &x_f16, &y_prod, m, k, b).unwrap()
        });
        let prod = gpu.download_f32(&y_prod).unwrap();
        let scale = prod.iter().fold(0f32, |a, v| a.max(v.abs())) as f64;
        println!(
            "sweep M{m:<5} K{k:<5} B{b:<4} production {t_prod:7.3} ms ({:5.1} TF/s)",
            flop / t_prod / 1e9
        );
        for tile in &tiles {
            if k % (64 * tile.split) != 0 {
                continue;
            }
            let t = time_ms(gpu, reps, |g| {
                g.gemm_f16_x_f16_wmma_lds_splitk(&w_f16, &x_f16, &y, m, k, b, *tile).unwrap()
            });
            let got = gpu.download_f32(&y).unwrap();
            let d = got.iter().zip(&prod).fold(0f64, |a, (x, p)| a.max((x - p).abs() as f64));
            println!(
                "sweep M{m:<5} K{k:<5} B{b:<4} {:36} {t:7.3} ms ({:5.1} TF/s) x{:5.2} vs production, max |diff| / max |y| {:.2e}",
                tile.entry(),
                flop / t / 1e9,
                t_prod / t,
                d / scale
            );
        }
    }
}

/// Every output of every iteration against iteration 0 (bitwise) and the
/// multirow / fused reference (tolerance).
fn stress(gpu: &mut Gpu) -> bool {
    let iters: usize = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(50);
    let pid = std::process::id();
    let shapes = parse(
        "SHAPES",
        "320x20480x1536,320x10240x1131,4x20480x1536,63x2560x641,65x2560x639,319x2624x513,321x2560x1535,320x2496x1131,320x2816x1131,511x1280x640,513x1280x767,1023x640x769,1025x640x513,5120x2560x1537",
    );
    let reads = parse("READS", "5120x1536,2560x1131,2560x513,2560x1900");
    let capture = std::env::var("CAPTURE").is_ok_and(|v| v == "1");
    struct Case {
        run: Box<dyn Fn(&mut Gpu)>,
        /// Issued under capture: the route's refusal, then the caller's fallback into `cap`.
        fallback: Box<dyn Fn(&mut Gpu)>,
        out: GpuTensor,
        cap: GpuTensor,
        reference: Vec<f32>,
        tol: f32,
        first: Option<Vec<u32>>,
        name: String,
    }
    let alias = |t: &GpuTensor| GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(t.buf.as_ptr(), t.numel() * 4) },
        shape: t.shape.clone(),
        dtype: t.dtype,
    };
    let mut cases: Vec<Case> = Vec::new();
    for s in &shapes {
        let (m, k, b) = (s[0], s[1], s[2]);
        let wb: Vec<u16> = lcg(m * k, 11 + m as u64, 0.05).iter().map(|&v| bf16_bits(v)).collect();
        let xr: Vec<f32> = lcg(b * k, 23 + b as u64, 2.0).iter().map(|&v| bf16_val(bf16_bits(v))).collect();
        let weight = std::rc::Rc::new(upload_u16(gpu, &wb, DType::BF16));
        let x = std::rc::Rc::new(gpu.upload_f32(&xr, &[b * k]).unwrap());
        let exact = gpu.zeros(&[b * m], DType::F32).unwrap();
        gpu.gemm_bf16_xf32_multirow(&weight, &x, &exact, m, k, b).unwrap();
        let reference = gpu.download_f32(&exact).unwrap();
        let scale = reference.iter().fold(0f32, |a, v| a.max(v.abs()));
        let out = gpu.zeros(&[b * m], DType::F32).unwrap();
        let o = alias(&out);
        let cap = gpu.zeros(&[b * m], DType::F32).unwrap();
        let c = alias(&cap);
        let (w2, x2) = (weight.clone(), x.clone());
        cases.push(Case {
            run: Box::new(move |g: &mut Gpu| {
                assert!(g.gemm_bf16_xf32_f16_wmma_qwen4(&[(&weight, &o, m)], &x, k, b).unwrap());
            }),
            fallback: Box::new(move |g: &mut Gpu| {
                assert!(!g.gemm_bf16_xf32_f16_wmma_qwen4(&[(&w2, &c, m)], &x2, k, b).unwrap());
                g.gemm_bf16_xf32_multirow(&w2, &x2, &c, m, k, b).unwrap();
            }),
            out,
            cap,
            reference,
            tol: 1e-4 * scale,
            first: None,
            name: format!("proj {m}x{k}x{b}"),
        });
    }
    let low_rank = 320usize;
    for s in &reads {
        let (hidden, rows) = (s[0], s[1]);
        let wide = 4 * hidden;
        let up: Vec<u16> = lcg(wide * low_rank, 5, 0.1).iter().map(|&v| bf16_bits(v)).collect();
        let low: Vec<f32> = lcg(rows * low_rank, 7, 3.0)
            .iter()
            .map(|&v| bf16_val(bf16_bits(v / (1.0 + (-v).exp()))))
            .collect();
        let norm: Vec<f32> = lcg(rows * wide, 9, 2.0)
            .iter()
            .map(|&v| bf16_val(bf16_bits(if v.abs() < 1e-3 { 1e-3 } else { v })))
            .collect();
        let up_weight = std::rc::Rc::new(upload_u16(gpu, &up, DType::BF16));
        let low_f32 = gpu.upload_f32(&low, &[rows * low_rank]).unwrap();
        let low_bf16 = upload_u16(gpu, &low.iter().map(|&v| bf16_bits(v)).collect::<Vec<_>>(), DType::BF16);
        let norm_f32 = gpu.upload_f32(&norm, &[rows * wide]).unwrap();
        let norm_f16 = upload_u16(gpu, &norm.iter().map(|&v| f16_of_bf16(v)).collect::<Vec<_>>(), DType::F16);
        let fused = gpu.zeros(&[rows * hidden], DType::F32).unwrap();
        hyper_read_up_fused(
            gpu,
            &HyperReadUpFused {
                up_weight: &up_weight,
                low: &low_f32,
                normalized: &norm_f32,
                mixed: &fused,
                rows,
                hidden,
                low_rank,
                normalized_bf16: false,
            },
        )
        .unwrap();
        let reference = gpu.download_f32(&fused).unwrap();
        let out = gpu.zeros(&[rows * hidden], DType::F32).unwrap();
        let o = alias(&out);
        let cap = gpu.zeros(&[rows * hidden], DType::F32).unwrap();
        let c = alias(&cap);
        let up2 = up_weight.clone();
        cases.push(Case {
            run: Box::new(move |g: &mut Gpu| {
                hyper_read_up_wmma(
                    g,
                    &HyperReadUpFused {
                        up_weight: &up_weight,
                        low: &low_bf16,
                        normalized: &norm_f16,
                        mixed: &o,
                        rows,
                        hidden,
                        low_rank,
                        normalized_bf16: false,
                    },
                )
                .unwrap()
            }),
            // The forward keeps F32 HC streams (and the fused SIMT read) when
            // `qwen4_bf16_streams` is false, which capture forces.
            fallback: Box::new(move |g: &mut Gpu| {
                assert!(!g.qwen4_bf16_streams(rows));
                hyper_read_up_fused(
                    g,
                    &HyperReadUpFused {
                        up_weight: &up2,
                        low: &low_f32,
                        normalized: &norm_f32,
                        mixed: &c,
                        rows,
                        hidden,
                        low_rank,
                        normalized_bf16: false,
                    },
                )
                .unwrap()
            }),
            out,
            cap,
            // One BF16 step of the fused read (the gate rounding), relative.
            reference,
            tol: -1.0,
            first: None,
            name: format!("read {hidden}x{rows}"),
        });
    }
    if capture {
        // Warmup-first, as the forwards do: size every scratch the eager
        // route grows (F16 X, K-split partials) before capturing, because
        // scratch growth invalidates captured graphs.
        for case in &cases {
            (case.run)(gpu);
        }
        if gpu.active_stream.is_none() {
            gpu.active_stream = Some(gpu.hip.stream_create().unwrap());
        }
        gpu.hip.device_synchronize().unwrap();
        let stream = gpu.active_stream.take().unwrap();
        gpu.graphs.begin_graph_capture(&gpu.hip, gpu.device_id, &stream).unwrap();
        gpu.active_stream = Some(stream);
        for case in &cases {
            (case.fallback)(gpu);
        }
        let stream = gpu.active_stream.take().unwrap();
        gpu.graphs.end_graph_capture(&gpu.hip, gpu.device_id, &stream).unwrap();
        gpu.active_stream = Some(stream);
    }
    let (mut unstable, mut wrong, mut outputs, mut cap_differ) = (0usize, 0usize, 0usize, 0usize);
    for it in 0..iters {
        if capture {
            let stream = gpu.active_stream.take().unwrap();
            gpu.graphs.graph_launch(&gpu.hip, gpu.device_id, &stream).unwrap();
            gpu.active_stream = Some(stream);
        }
        for case in cases.iter_mut() {
            (case.run)(gpu);
            let got = gpu.download_f32(&case.out).unwrap();
            outputs += got.len();
            let bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            match &case.first {
                None => {
                    let far = if case.tol >= 0.0 {
                        got.iter().zip(&case.reference).filter(|(g, r)| (*g - *r).abs() > case.tol).count()
                    } else {
                        got.iter()
                            .zip(&case.reference)
                            .filter(|(g, r)| (*g - *r).abs() > r.abs().max(g.abs()) / 64.0 + 1e-6)
                            .count()
                    };
                    if far > 0 {
                        println!("[{pid}] it {it} {}: {far} outputs outside the reference tolerance", case.name);
                    }
                    wrong += far;
                    case.first = Some(bits);
                }
                Some(first) => {
                    let differ = first.iter().zip(&bits).filter(|(a, b)| a != b).count();
                    if differ > 0 {
                        println!("[{pid}] it {it} {}: {differ} outputs differ from iteration 0", case.name);
                    }
                    unstable += differ;
                }
            }
            if capture {
                gpu.hip.device_synchronize().unwrap();
                let got = gpu.download_f32(&case.cap).unwrap();
                let differ = got.iter().zip(&case.reference).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                if differ > 0 {
                    println!("[{pid}] it {it} {}: {differ} captured-fallback outputs differ from eager", case.name);
                }
                cap_differ += differ;
            }
        }
    }
    println!(
        "[{pid}] stress {} iterations x {} cases (capture {}): {outputs} outputs checked, {unstable} differ from iteration 0, {wrong} outside reference tolerance, {cap_differ} captured-fallback outputs differ from eager",
        iters,
        cases.len(),
        if capture { "on" } else { "off" }
    );
    unstable == 0 && wrong == 0 && cap_differ == 0
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch {}", gpu.arch);
    let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    match std::env::var("MODE").as_deref() {
        Ok("sweep") => sweep(&mut gpu, reps),
        Ok("stress") => {
            if !stress(&mut gpu) {
                std::process::exit(1);
            }
        }
        _ => {
            projections(&mut gpu, reps);
            reads(&mut gpu, reps);
        }
    }
}
