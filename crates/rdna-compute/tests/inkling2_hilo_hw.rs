// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Hardware gate (gfx12) for the Inkling-2 blocked-prefill hi/lo-F16 WMMA projections
//! (`rdna_compute::inkling2_hilo`) at the model's shapes, against f64 references:
//!
//! * `inkling2_gemm_hilo_f32`: controller down (64 x 4608, three problems), dynamic conv
//!   taps (32 x 2304, two problems sharing x), sinks (24 x 2304, ragged M); rows 1, 37,
//!   300, 512. Error bar: `|y - ref| <= 1e-5 * sum_k |w x|` (F32 class; the worst normalised
//!   error is printed beside a sequential CPU F32 sum's for scale);
//! * `inkling2_control_up_blend_hilo`: gain/forget (2304 x 64, three problems) and the
//!   blend tail against an f64 controller tail;
//! * no write past the output, bitwise run-to-run, and refused shapes;
//! * information: median ms per launch at 512 rows.
//!
//! Operands span magnitudes (a few at 3e-6 and at 900) so the lo plane's pre-scale and the
//! hi plane's range both matter.
//!
//! `#[ignore]`d: needs a gfx12 GPU. Run explicitly (through the GPU fabric):
//!
//!   cargo test -p rdna-compute --release --test inkling2_hilo_hw -- --ignored --nocapture

use rdna_compute::inkling2_hilo::{Inkling2HiloBlend, Inkling2HiloGemm};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const SENTINEL: u32 = 0x7fc0_dead;

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32 as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let v = self.unit();
                match i % 97 {
                    0 => v * 3.0e-6,
                    1 => v * 900.0 * scale.min(1.0),
                    _ => v * scale,
                }
            })
            .collect()
    }
}

fn up(gpu: &mut Gpu, v: &[f32]) -> GpuTensor {
    gpu.upload_f32(v, &[v.len()]).expect("upload")
}

fn sentinel(gpu: &mut Gpu, n: usize) -> GpuTensor {
    up(gpu, &vec![f32::from_bits(SENTINEL); n])
}

fn median_ms(gpu: &mut Gpu, mut f: impl FnMut(&mut Gpu)) -> f64 {
    let mut samples = Vec::new();
    for round in 0..12 {
        let t = Instant::now();
        for _ in 0..20 {
            f(gpu);
        }
        gpu.hip.device_synchronize().unwrap();
        if round >= 2 {
            samples.push(t.elapsed().as_secs_f64() * 1e3 / 20.0);
        }
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// (f64 dot, sum |w x|, sequential F32 dot) of row `r` of x against row `o` of w.
fn dot(w: &[f32], x: &[f32], o: usize, r: usize, k: usize) -> (f64, f64, f32) {
    let (wr, xr) = (&w[o * k..(o + 1) * k], &x[r * k..(r + 1) * k]);
    let mut d = 0f64;
    let mut a = 0f64;
    let mut s = 0f32;
    for (&p, &q) in wr.iter().zip(xr) {
        d += f64::from(p) * f64::from(q);
        a += (f64::from(p) * f64::from(q)).abs();
        s += p * q;
    }
    (d, a, s)
}

/// `problems` independent (w, x) pairs (x shared when `shared_x`) of `[m, k]` x `[n, k]`.
fn gemm_case(
    gpu: &mut Gpu,
    g: &mut Lcg,
    problems: usize,
    m: usize,
    k: usize,
    n: usize,
    shared_x: bool,
    time: bool,
) {
    let ws: Vec<Vec<f32>> = (0..problems).map(|_| g.vec(m * k, 0.03)).collect();
    let xs: Vec<Vec<f32>> = (0..problems)
        .map(|p| {
            if shared_x && p > 0 {
                Vec::new()
            } else {
                g.vec(n * k, 1.5)
            }
        })
        .collect();
    let wt: Vec<GpuTensor> = ws.iter().map(|w| up(gpu, w)).collect();
    let xt: Vec<GpuTensor> = xs
        .iter()
        .filter(|x| !x.is_empty())
        .map(|x| up(gpu, x))
        .collect();
    let yt: Vec<GpuTensor> = (0..problems).map(|_| sentinel(gpu, n * m + m)).collect();
    let xi = |p: usize| if shared_x { 0 } else { p };
    let probs: Vec<Inkling2HiloGemm<'_>> = (0..problems)
        .map(|p| Inkling2HiloGemm {
            w: &wt[p],
            x: &xt[xi(p)],
            y: &yt[p],
        })
        .collect();
    gpu.inkling2_gemm_hilo_f32(&probs, m, k, n)
        .expect("hilo gemm");
    gpu.hip.device_synchronize().expect("sync");
    let first: Vec<Vec<f32>> = yt.iter().map(|y| gpu.download_f32(y).unwrap()).collect();
    // Run to run: bitwise.
    gpu.inkling2_gemm_hilo_f32(&probs, m, k, n)
        .expect("hilo gemm");
    gpu.hip.device_synchronize().expect("sync");
    let (mut worst, mut worst_f32) = (0f64, 0f64);
    for p in 0..problems {
        let again = gpu.download_f32(&yt[p]).unwrap();
        let x = &xs[xi(p)];
        for (i, (a, b)) in first[p].iter().zip(&again).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "m {m} k {k} n {n} p {p}: run-to-run differs at {i}"
            );
        }
        for (i, v) in first[p].iter().enumerate().skip(n * m) {
            assert_eq!(
                v.to_bits(),
                SENTINEL,
                "m {m} k {k} n {n} p {p}: wrote past the output at {i}"
            );
        }
        for r in 0..n {
            for o in 0..m {
                let (d, a, s) = dot(&ws[p], x, o, r, k);
                let got = f64::from(first[p][r * m + o]);
                let e = (got - d).abs() / a.max(1e-30);
                assert!(
                    e <= 1e-5,
                    "m {m} k {k} n {n} p {p} row {r} out {o}: {got} vs f64 {d} (normalised error {e:.2e})"
                );
                worst = worst.max(e);
                worst_f32 = worst_f32.max((f64::from(s) - d).abs() / a.max(1e-30));
            }
        }
    }
    eprintln!(
        "gemm  P {problems} m {m:5} k {k:5} n {n:4}: worst |y - f64| / sum|wx| {worst:.2e} (sequential F32 {worst_f32:.2e})"
    );
    if time {
        let ms = median_ms(gpu, |gg| {
            gg.inkling2_gemm_hilo_f32(&probs, m, k, n).unwrap()
        });
        eprintln!("    ms per launch {ms:.4}");
    }
    drop(probs);
    for t in wt.into_iter().chain(xt).chain(yt) {
        gpu.free_tensor(t).unwrap();
    }
}

fn blend_case(
    gpu: &mut Gpu,
    g: &mut Lcg,
    problems: usize,
    m: usize,
    k: usize,
    n: usize,
    time: bool,
) {
    let gw: Vec<Vec<f32>> = (0..problems).map(|_| g.vec(m * k, 0.05)).collect();
    let fw: Vec<Vec<f32>> = (0..problems).map(|_| g.vec(m * k, 0.05)).collect();
    let gs: Vec<Vec<f32>> = (0..problems).map(|_| g.vec(n * k, 4.0)).collect();
    let curs: Vec<Vec<f32>> = (0..problems).map(|_| g.vec(n * m, 2.0)).collect();
    let msgs: Vec<Vec<f32>> = (0..problems).map(|_| g.vec(n * m, 2.0)).collect();
    let gwt: Vec<GpuTensor> = gw.iter().map(|v| up(gpu, v)).collect();
    let fwt: Vec<GpuTensor> = fw.iter().map(|v| up(gpu, v)).collect();
    let gt: Vec<GpuTensor> = gs.iter().map(|v| up(gpu, v)).collect();
    let ct: Vec<GpuTensor> = curs.iter().map(|v| up(gpu, v)).collect();
    let mt: Vec<GpuTensor> = msgs.iter().map(|v| up(gpu, v)).collect();
    let ot: Vec<GpuTensor> = (0..problems).map(|_| sentinel(gpu, n * m + m)).collect();
    let probs: Vec<Inkling2HiloBlend<'_>> = (0..problems)
        .map(|p| Inkling2HiloBlend {
            gain_w: &gwt[p],
            forget_w: &fwt[p],
            g: &gt[p],
            cur: &ct[p],
            msg: &mt[p],
            out: &ot[p],
        })
        .collect();
    gpu.inkling2_control_up_blend_hilo(&probs, m, k, n)
        .expect("blend");
    gpu.hip.device_synchronize().expect("sync");
    let mut worst = 0f64;
    for p in 0..problems {
        let out = gpu.download_f32(&ot[p]).unwrap();
        for (i, v) in out.iter().enumerate().skip(n * m) {
            assert_eq!(
                v.to_bits(),
                SENTINEL,
                "blend m {m} n {n} p {p}: wrote past the output at {i}"
            );
        }
        for r in 0..n {
            for o in 0..m {
                let (a, aa, _) = dot(&gw[p], &gs[p], o, r, k);
                let (f, fa, _) = dot(&fw[p], &gs[p], o, r, k);
                let i = r * m + o;
                let (c, ms) = (f64::from(curs[p][i]), f64::from(msgs[p][i]));
                let want = c - f.tanh() * c + a.tanh() * ms;
                // tanh' <= 1: the projection error bar carries through, plus F32 rounding of the tail.
                let bar = 1e-5 * (fa * c.abs() + aa * ms.abs())
                    + 4e-7 * (c.abs() + ms.abs() + want.abs());
                let got = f64::from(out[i]);
                assert!(
                    (got - want).abs() <= bar,
                    "blend m {m} k {k} n {n} p {p} row {r} out {o}: {got} vs f64 {want} (bar {bar:.2e})"
                );
                worst = worst.max((got - want).abs() / bar);
            }
        }
    }
    eprintln!("blend P {problems} m {m:5} k {k:5} n {n:4}: worst error / bar {worst:.3}");
    if time {
        let ms = median_ms(gpu, |gg| {
            gg.inkling2_control_up_blend_hilo(&probs, m, k, n).unwrap()
        });
        eprintln!("    ms per launch {ms:.4}");
    }
    drop(probs);
    for t in gwt
        .into_iter()
        .chain(fwt)
        .chain(gt)
        .chain(ct)
        .chain(mt)
        .chain(ot)
    {
        gpu.free_tensor(t).unwrap();
    }
}

#[test]
#[ignore = "needs a gfx12 GPU"]
fn inkling2_hilo_projections_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    if !gpu.arch_caps.has_wmma_w32_gfx12() {
        eprintln!("skip: {} has no gfx12 WMMA", gpu.arch);
        return;
    }
    let mut g = Lcg(0x1d10_6e44);
    // Controller down: three controllers per launch.
    for n in [1, 37, 512] {
        gemm_case(&mut gpu, &mut g, 3, 64, 4608, n, false, n == 512);
    }
    // The 32-row tiling (m >= 64) off its tile grid: ragged M and N, one split-K wave.
    gemm_case(&mut gpu, &mut g, 2, 80, 256, 45, false, false);
    // Dynamic conv taps (K and V share the normed input on SWA-free layers) and sinks.
    gemm_case(&mut gpu, &mut g, 2, 32, 2304, 512, true, true);
    gemm_case(&mut gpu, &mut g, 2, 32, 2304, 300, false, false);
    gemm_case(&mut gpu, &mut g, 1, 24, 2304, 1, false, false);
    gemm_case(&mut gpu, &mut g, 1, 24, 2304, 512, false, true);
    // Controller gain/forget + blend.
    for n in [1, 37, 512] {
        blend_case(&mut gpu, &mut g, 3, 2304, 64, n, n == 512);
    }
    blend_case(&mut gpu, &mut g, 1, 2304, 64, 300, false);

    // Refused, never mis-launched.
    let t = gpu.alloc_tensor(&[64 * 64], DType::F32).unwrap();
    let p = || Inkling2HiloGemm {
        w: &t,
        x: &t,
        y: &t,
    };
    assert!(
        gpu.inkling2_gemm_hilo_f32(&[p()], 8, 24, 8).is_err(),
        "k % 16 != 0"
    );
    assert!(
        gpu.inkling2_gemm_hilo_f32(&[], 16, 16, 16).is_err(),
        "no problems"
    );
    assert!(
        gpu.inkling2_gemm_hilo_f32(&[p(), p(), p(), p(), p()], 16, 16, 16)
            .is_err(),
        "five problems"
    );
    assert!(
        gpu.inkling2_gemm_hilo_f32(&[p()], 64, 64, 100).is_err(),
        "short operands"
    );
    gpu.free_tensor(t).unwrap();
}
