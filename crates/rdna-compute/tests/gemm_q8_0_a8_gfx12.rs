// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx12 W8A8 Q8_0 GEMM (`gemm_q8_0_a8_batched`, `kernels/src/gemm_q8_0_mmq_i8.gfx12.hip`)
//! against a CPU emulation of its exact contract:
//!
//!   amax = max |x| over a 32-K block; d = RN(amax / 127); q = clamp(rint(x / d), +-127)
//!   (zero block: d = 0, q = 0); per (row, token) acc = 0; for blocks b ascending:
//!   c = sum_i W8 * X8 (exact int); acc = fma(RN(ws * xs), float(c), acc).
//!
//! The integer dot is exact, so GPU and emulation must agree bit for bit for every
//! tile variant, both weight read paths (4-byte aligned K % 64 == 0 `FAST`, and the
//! 2-byte aligned `_slow` path used for K % 64 == 32 or a misaligned weight base)
//! and all M / N tails. An F64 reference bounds the quantization error.
//!
//! `#[ignore]`d: needs a gfx12 GPU. Run through fabric-run.sh:
//!
//!   cargo test -p rdna-compute --release --test gemm_q8_0_a8_gfx12 -- --ignored

use rdna_compute::{DType, Gpu, GpuTensor, Q8A8Tile};

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    fn gauss(&mut self) -> f32 {
        let s: f32 = (0..4).map(|_| self.unit()).sum();
        (s - 2.0) * 1.732_050_8
    }
}

fn bf16_round(x: f32) -> f32 {
    let u = x.to_bits();
    f32::from_bits(u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff_0000)
}

/// Returns (Q8_0 bytes, int8 quants [m*k], f32 scales [m * k/32]).
fn make_weights(m: usize, k: usize, rng: &mut Rng) -> (Vec<u8>, Vec<i8>, Vec<f32>) {
    let nb = k / 32;
    let mut bytes = Vec::new();
    let (mut qs, mut ss) = (Vec::new(), Vec::new());
    for _ in 0..m * nb {
        let sc = half::f16::from_f32(0.004 * (1.0 + 0.5 * rng.gauss().abs()));
        bytes.extend_from_slice(&sc.to_bits().to_le_bytes());
        ss.push(sc.to_f32());
        for _ in 0..32 {
            let q = (rng.gauss() * 40.0).round().clamp(-128.0, 127.0) as i8;
            bytes.push(q as u8);
            qs.push(q);
        }
    }
    (bytes, qs, ss)
}

fn emulate(wq: &[i8], ws: &[f32], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let nb = k / 32;
    let mut xq = vec![0i8; n * k];
    let mut xs = vec![0f32; n * nb];
    for t in 0..n {
        for b in 0..nb {
            let blk = &x[t * k + b * 32..t * k + b * 32 + 32];
            let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
            if amax == 0.0 {
                continue;
            }
            let d = amax / 127.0;
            xs[t * nb + b] = d;
            for (i, &v) in blk.iter().enumerate() {
                xq[t * k + b * 32 + i] = (v / d).round_ties_even().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    let mut y = vec![0f32; n * m];
    for t in 0..n {
        for r in 0..m {
            let mut acc = 0f32;
            for b in 0..nb {
                let c: i32 = (0..32)
                    .map(|i| wq[r * k + b * 32 + i] as i32 * xq[t * k + b * 32 + i] as i32)
                    .sum();
                let scale = ws[r * nb + b] * xs[t * nb + b];
                acc = scale.mul_add(c as f32, acc);
            }
            y[t * m + r] = acc;
        }
    }
    y
}

fn run_case(gpu: &mut Gpu, m: usize, k: usize, n: usize, misalign_base: bool, seed: u64) {
    let mut rng = Rng(seed);
    let (wbytes, wq, ws) = make_weights(m, k, &mut rng);
    // Massive channels + hot rows exercise the per-block scale range.
    let hot: Vec<f32> = (0..k)
        .map(|_| if rng.unit() < 0.02 { 30.0 } else { 1.0 })
        .collect();
    let x: Vec<f32> = (0..n)
        .flat_map(|t| {
            let row = if t % 7 == 3 { 9.0 } else { 1.0 };
            let g: Vec<f32> = (0..k).map(|_| rng.gauss()).collect();
            g.into_iter()
                .zip(hot.iter())
                .map(move |(v, h)| bf16_round(v * h * row))
                .collect::<Vec<_>>()
        })
        .collect();
    // An all-zero token block exercises the d = 0 path.
    let mut x = x;
    for v in &mut x[..32.min(k)] {
        *v = 0.0;
    }

    // Weight upload: optionally shifted by 2 bytes so the base is only 2-aligned.
    let (w, _owner): (GpuTensor, Option<GpuTensor>) = if misalign_base {
        let mut padded = vec![0u8; 2];
        padded.extend_from_slice(&wbytes);
        let owner = gpu.upload_raw(&padded, &[padded.len()]).expect("w");
        let view = owner.sub_offset(2, wbytes.len());
        (view, Some(owner))
    } else {
        (gpu.upload_raw(&wbytes, &[wbytes.len()]).expect("w"), None)
    };
    let xt = gpu.upload_f32(&x, &[n * k]).expect("x");
    let want = emulate(&wq, &ws, &x, m, k, n);

    let check = |gpu: &Gpu, y: &GpuTensor, what: &str| {
        let got = gpu.download_f32(y).expect("download");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{what} m={m} k={k} n={n} misalign={misalign_base}: Y[{}][{}] gpu {g} emu {w}",
                i / m,
                i % m
            );
        }
    };

    let acts = gpu.q8_a8_quantize_x(&xt, n, k).expect("quantize");
    for tile in Q8A8Tile::ALL {
        // NaN-poisoned output so a missed cell fails the bit compare.
        let y = gpu.upload_f32(&vec![f32::NAN; n * m], &[n * m]).expect("y");
        gpu.gemm_q8_0_a8_gemm_tile(&w, &acts, &y, m, tile)
            .unwrap_or_else(|e| panic!("{}: {e:?}", tile.label()));
        check(gpu, &y, tile.label());
    }
    let y = gpu.upload_f32(&vec![f32::NAN; n * m], &[n * m]).expect("y");
    gpu.gemm_q8_0_a8_batched(&w, &xt, &y, m, k, n)
        .expect("batched");
    check(gpu, &y, "batched");
}

#[test]
#[ignore = "needs a gfx12 GPU"]
fn w8a8_q8_0_matches_cpu_emulation_bitwise() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert!(gpu.q8_a8_supported(32), "needs gfx12, got {}", gpu.arch);
    // (m, k, n): tails in M and N; K = 576 / 512 (Raon MLA), K % 64 == 32 (slow
    // path), single-slab K, a wider multi-tile case.
    for (i, &(m, k, n)) in [
        (130usize, 640usize, 70usize),
        (97, 96, 33),
        (200, 576, 130),
        (64, 512, 64),
        (129, 32, 5),
        (260, 1280, 257),
    ]
    .iter()
    .enumerate()
    {
        run_case(&mut gpu, m, k, n, false, 0xa8a8_0000 + i as u64);
    }
    // Misaligned weight base takes the 2-byte `_slow` kernels even at K % 64 == 0.
    run_case(&mut gpu, 130, 640, 70, true, 0xa8a8_1000);
}

#[test]
#[ignore = "needs a gfx12 GPU"]
fn w8a8_q8_0_error_vs_f64_is_activation_quantization_sized() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert!(gpu.q8_a8_supported(32), "needs gfx12, got {}", gpu.arch);
    let (m, k, n) = (192usize, 2560usize, 96usize);
    let mut rng = Rng(0xf64f_a8a8);
    let (wbytes, wq, ws) = make_weights(m, k, &mut rng);
    let x: Vec<f32> = (0..n * k).map(|_| bf16_round(rng.gauss())).collect();
    let w = gpu.upload_raw(&wbytes, &[wbytes.len()]).expect("w");
    let xt = gpu.upload_f32(&x, &[n * k]).expect("x");
    let y = gpu.alloc_tensor(&[n * m], DType::F32).expect("y");
    gpu.gemm_q8_0_a8_batched(&w, &xt, &y, m, k, n).expect("a8");
    let got = gpu.download_f32(&y).expect("download");
    let nb = k / 32;
    let (mut se, mut sr) = (0f64, 0f64);
    for t in 0..n {
        for r in 0..m {
            let mut s = 0f64;
            for c in 0..k {
                s += ws[r * nb + c / 32] as f64 * wq[r * k + c] as f64 * x[t * k + c] as f64;
            }
            se += (got[t * m + r] as f64 - s).abs();
            sr += s.abs();
        }
    }
    let rel = se / sr;
    // int8 activations at amax/127 on Gaussian blocks: ~0.5% mean relative error
    // (measured 5.1e-3). A structural error shows up as >1e-2.
    assert!(rel < 1.5e-2, "mean relative error vs F64 {rel:.3e}");
}

// ---------------------------------------------------------------------------------------
// FP8 twin (`gemm_q8_0_fp8_batched`, `kernels/src/gemm_q8_0_mmq_fp8.gfx12.hip`).
//
// Contract emulated on the CPU:
//   producer: amax = max |x| over a 32-K block; d = RN(amax / 448); code = e4m3_rne_sat(
//             clamp(RN(x / d), +-448)) (zero block: d = 0, code = 0);
//   weights : int8 q -> e4m3 RNE (lossy for |q| > 16);
//   GEMM    : per K32 block two chained fp8 WMMAs (K16 each) from a zero accumulator,
//             c = RN(RN(sum over K{0-7,16-23}) + sum over K{8-15,24-31}) (exact products),
//             acc = fma(RN(ws * xs), c, acc), blocks ascending.
// ---------------------------------------------------------------------------------------

/// F32 -> OCP E4M3 (finite, saturating at 448), round-to-nearest-even. `v` is finite.
fn e4m3_encode(v: f32) -> u8 {
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0 };
    let a = v.abs().min(448.0);
    if a == 0.0 {
        return 0;
    }
    if a < 0.015625 {
        // Subnormal: multiples of 2^-9; a carry to 8 is the smallest normal (code 0x08).
        let m = (a * 512.0).round_ties_even() as u8;
        return sign | m;
    }
    let e = ((a.to_bits() >> 23) as i32) - 127;
    let frac = a / f32::from_bits(((e + 127) as u32) << 23); // [1, 2)
    let mut mant = ((frac - 1.0) * 8.0).round_ties_even() as i32;
    let mut e = e;
    if mant == 8 {
        mant = 0;
        e += 1;
    }
    let code = (((e + 7) as u8) << 3) | mant as u8;
    sign | code.min(0x7e)
}

fn e4m3_decode(c: u8) -> f32 {
    let s = if c & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((c >> 3) & 15) as i32;
    let m = (c & 7) as f32;
    if e == 0 {
        s * m * (1.0 / 512.0)
    } else {
        s * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// Returns `(y, mag)`: the emulated output and, per output, `sum_b |scale_b * c_b|` (the
/// scale of the terms folded into it, used for the WMMA-rounding tolerance).
fn emulate_fp8(
    wq: &[i8],
    ws: &[f32],
    x: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, Vec<f64>) {
    let nb = k / 32;
    let wd: Vec<f64> = wq
        .iter()
        .map(|&q| e4m3_decode(e4m3_encode(q as f32)) as f64)
        .collect();
    let mut xd = vec![0f64; n * k];
    let mut xs = vec![0f32; n * nb];
    for t in 0..n {
        for b in 0..nb {
            let blk = &x[t * k + b * 32..t * k + b * 32 + 32];
            let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
            if amax == 0.0 {
                continue;
            }
            let d = amax / 448.0;
            xs[t * nb + b] = d;
            for (i, &v) in blk.iter().enumerate() {
                let q = (v / d).clamp(-448.0, 448.0);
                xd[t * k + b * 32 + i] = e4m3_decode(e4m3_encode(q)) as f64;
            }
        }
    }
    let mut y = vec![0f32; n * m];
    let mut mag = vec![0f64; n * m];
    for t in 0..n {
        for r in 0..m {
            let mut acc = 0f32;
            let mut mg = 0f64;
            for b in 0..nb {
                let p = |i: usize| wd[r * k + b * 32 + i] * xd[t * k + b * 32 + i];
                let h1: f64 = (0..8).chain(16..24).map(p).sum();
                let h2: f64 = (8..16).chain(24..32).map(p).sum();
                let s1 = h1 as f32;
                let c = (s1 as f64 + h2) as f32;
                let scale = ws[r * nb + b] * xs[t * nb + b];
                acc = scale.mul_add(c, acc);
                mg += (scale as f64 * c as f64).abs();
            }
            y[t * m + r] = acc;
            mag[t * m + r] = mg;
        }
    }
    (y, mag)
}

/// GPU vs emulation. The fp8 WMMA's internal F32 accumulation of a block's 16 exact products
/// (plus the carried C) is not bit-reproducible by an exactly-rounded sum: the hardware
/// aligns/truncates inside the matrix core. Measured on gfx1201 it differs from the exact
/// RNE sum in a few 1e-4 of the outputs by 1-3 ulp. The contract checked here is therefore
/// (a) >= 99% of outputs bit-identical and (b) every output within `8 * 2^-23 * mag` of the
/// emulation, `mag = sum_b |scale_b * c_b|`.
struct BitDiff {
    mismatched: usize,
    total: usize,
    max_rel: f64,
    max_over_tol: f64,
    first: Option<(usize, f32, f32)>,
}

fn bit_diff(got: &[f32], want: &[f32], mag: &[f64]) -> BitDiff {
    let mut d = BitDiff {
        mismatched: 0,
        total: got.len(),
        max_rel: 0.0,
        max_over_tol: 0.0,
        first: None,
    };
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if g.to_bits() != w.to_bits() {
            d.mismatched += 1;
            if d.first.is_none() {
                d.first = Some((i, g, w));
            }
            let diff = (g as f64 - w as f64).abs();
            d.max_rel = d.max_rel.max(diff / (w as f64).abs().max(1e-30));
            let tol = 8.0 * 2f64.powi(-23) * mag[i];
            d.max_over_tol = d.max_over_tol.max(diff / tol.max(1e-30));
        }
    }
    d
}

fn run_case_fp8(gpu: &mut Gpu, m: usize, k: usize, n: usize, misalign_base: bool, seed: u64) {
    let mut rng = Rng(seed);
    let (wbytes, wq, ws) = make_weights(m, k, &mut rng);
    let hot: Vec<f32> = (0..k)
        .map(|_| if rng.unit() < 0.02 { 30.0 } else { 1.0 })
        .collect();
    let mut x: Vec<f32> = (0..n)
        .flat_map(|t| {
            let row = if t % 7 == 3 { 9.0 } else { 1.0 };
            let g: Vec<f32> = (0..k).map(|_| rng.gauss()).collect();
            g.into_iter()
                .zip(hot.iter())
                .map(move |(v, h)| bf16_round(v * h * row))
                .collect::<Vec<_>>()
        })
        .collect();
    for v in &mut x[..32.min(k)] {
        *v = 0.0;
    }

    let (w, _owner): (GpuTensor, Option<GpuTensor>) = if misalign_base {
        let mut padded = vec![0u8; 2];
        padded.extend_from_slice(&wbytes);
        let owner = gpu.upload_raw(&padded, &[padded.len()]).expect("w");
        let view = owner.sub_offset(2, wbytes.len());
        (view, Some(owner))
    } else {
        (gpu.upload_raw(&wbytes, &[wbytes.len()]).expect("w"), None)
    };
    let xt = gpu.upload_f32(&x, &[n * k]).expect("x");
    let (want, mag) = emulate_fp8(&wq, &ws, &x, m, k, n);

    let check = |gpu: &Gpu, y: &GpuTensor, what: &str| {
        let got = gpu.download_f32(y).expect("download");
        let d = bit_diff(&got, &want, &mag);
        println!(
            "fp8 {what} m={m} k={k} n={n} misalign={misalign_base}: {}/{} outputs differ bitwise from the emulation, max rel {:.3e}, max diff / tolerance {:.3}",
            d.mismatched, d.total, d.max_rel, d.max_over_tol
        );
        assert!(
            got.iter().all(|v| v.is_finite()),
            "{what}: non-finite output (missed cell)"
        );
        assert!(
            d.max_over_tol <= 1.0,
            "{what} m={m} k={k} n={n} misalign={misalign_base}: beyond WMMA-rounding tolerance, first mismatch {:?}",
            d.first
        );
        assert!(
            d.mismatched * 100 <= d.total,
            "{what} m={m} k={k} n={n} misalign={misalign_base}: {}/{} outputs differ bitwise",
            d.mismatched,
            d.total
        );
    };

    let acts = gpu.q8_fp8_quantize_x(&xt, n, k).expect("quantize");
    for tile in Q8A8Tile::ALL {
        let y = gpu.upload_f32(&vec![f32::NAN; n * m], &[n * m]).expect("y");
        gpu.gemm_q8_0_fp8_gemm_tile(&w, &acts, &y, m, tile)
            .unwrap_or_else(|e| panic!("{}: {e:?}", tile.label()));
        check(gpu, &y, tile.label());
    }
    let y = gpu.upload_f32(&vec![f32::NAN; n * m], &[n * m]).expect("y");
    gpu.gemm_q8_0_fp8_batched(&w, &xt, &y, m, k, n)
        .expect("batched");
    check(gpu, &y, "batched");
}

#[test]
#[ignore = "needs a gfx12 GPU"]
fn fp8_q8_0_matches_cpu_emulation() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert!(gpu.q8_fp8_supported(32), "needs gfx12, got {}", gpu.arch);
    for (i, &(m, k, n)) in [
        (130usize, 640usize, 70usize),
        (97, 96, 33),
        (200, 576, 130),
        (64, 512, 64),
        (129, 32, 5),
        (260, 1280, 257),
    ]
    .iter()
    .enumerate()
    {
        run_case_fp8(&mut gpu, m, k, n, false, 0xf8f8_0000 + i as u64);
    }
    run_case_fp8(&mut gpu, 130, 640, 70, true, 0xf8f8_1000);
}

/// BF16-valued activations: unit Gaussian bulk, ~0.8% persistent massive channels
/// (x25..x60) and ~1.5% hot rows (x8) — the bench generator.
fn make_acts_massive(n: usize, k: usize, rng: &mut Rng) -> Vec<f32> {
    let hot_ch: Vec<f32> = (0..k)
        .map(|_| {
            if rng.unit() < 0.008 {
                25.0 + 35.0 * rng.unit()
            } else {
                1.0
            }
        })
        .collect();
    let mut x = Vec::with_capacity(n * k);
    for _ in 0..n {
        let row = if rng.unit() < 0.015 { 8.0 } else { 1.0 };
        for c in 0..k {
            x.push(bf16_round(rng.gauss() * row * hot_ch[c]));
        }
    }
    x
}

/// Error vs F64 of the int8 (A8) and FP8 routes on Gaussian BF16 X and on massive-channel X.
#[test]
#[ignore = "needs a gfx12 GPU"]
fn fp8_q8_0_error_vs_f64_next_to_a8() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert!(gpu.q8_fp8_supported(32), "needs gfx12, got {}", gpu.arch);
    let (m, k, n) = (192usize, 2560usize, 96usize);
    let mut rng = Rng(0xf64f_f8f8);
    let (wbytes, wq, ws) = make_weights(m, k, &mut rng);
    let w = gpu.upload_raw(&wbytes, &[wbytes.len()]).expect("w");
    let nb = k / 32;
    let gauss: Vec<f32> = (0..n * k).map(|_| bf16_round(rng.gauss())).collect();
    let massive = make_acts_massive(n, k, &mut rng);
    println!("| X | route | max abs | mean abs | rel (sum|d|/sum|ref|) |");
    println!("|---|---|---:|---:|---:|");
    let mut fp8_rel = Vec::new();
    for (label, x) in [("gaussian", &gauss), ("massive", &massive)] {
        let xt = gpu.upload_f32(x, &[n * k]).expect("x");
        let y8 = gpu.alloc_tensor(&[n * m], DType::F32).expect("y8");
        let yf = gpu.alloc_tensor(&[n * m], DType::F32).expect("yf");
        gpu.gemm_q8_0_a8_batched(&w, &xt, &y8, m, k, n).expect("a8");
        gpu.gemm_q8_0_fp8_batched(&w, &xt, &yf, m, k, n)
            .expect("fp8");
        let g8 = gpu.download_f32(&y8).expect("download");
        let gf = gpu.download_f32(&yf).expect("download");
        let mut reference = vec![0f64; n * m];
        for t in 0..n {
            for r in 0..m {
                let mut s = 0f64;
                for c in 0..k {
                    s += ws[r * nb + c / 32] as f64 * wq[r * k + c] as f64 * x[t * k + c] as f64;
                }
                reference[t * m + r] = s;
            }
        }
        for (route, got) in [("A8", &g8), ("FP8", &gf)] {
            let (mut mx, mut se, mut sr) = (0f64, 0f64, 0f64);
            for (g, r) in got.iter().zip(&reference) {
                let e = (*g as f64 - r).abs();
                mx = mx.max(e);
                se += e;
                sr += r.abs();
            }
            let rel = se / sr;
            println!(
                "| {label} | {route} | {mx:.4e} | {:.4e} | {rel:.3e} |",
                se / got.len() as f64
            );
            if route == "FP8" {
                fp8_rel.push(rel);
            }
        }
    }
    // E4M3 has 3 mantissa bits on both operands (~3-4% per-element relative error that
    // averages down over the K-sum); a structural error shows up as O(1).
    for rel in fp8_rel {
        assert!(rel < 0.1, "FP8 mean relative error vs F64 {rel:.3e}");
    }
}
