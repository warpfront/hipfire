// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity + microbenchmark for the LDS-tiled VAE im2col and the fused
//! GroupNorm+SiLU, at the channel counts the real FLUX VAE decoder uses.
//!
//! Everything here is checked against the kernel it replaces, never against
//! itself:
//!
//!  1. **im2col map exactness** — `vae_im2col_f16_lds` vs the previous
//!     channel-fastest gather (`map = "c"`), byte-for-byte over the whole
//!     column matrix. Both produce f16 from the same f32 input, so any
//!     difference is a bug, not rounding.
//!  2. **im2col banding exactness** — the banded launch (`y0`, `rows`)
//!     reassembled band by band must equal the whole-image launch, including
//!     the top/bottom halo rows that only banding can get wrong.
//!  3. **conv route parity** — im2col + WMMA f16 GEMM + transpose vs the f32
//!     direct convolution `vae_conv3x3_f32`, at a relative-L2 tolerance that
//!     admits the f16 operands but not a wrong answer.
//!  4. **fused GroupNorm+SiLU exactness** — `vae_groupnorm_silu_f32` vs
//!     `vae_groupnorm_f32` followed by `silu_f32`, bit-for-bit.
//!
//! Then `--bench` times the im2col at the three real FLUX conv shapes
//! (512ch@128x128, 256ch@512x512, 128ch@1024x1024) for both maps and reports
//! effective GB/s against the machine's DRAM roof.
//!
//! Build: `cargo run --release --features lab --example test_vae_lds -p rdna-compute`
//! Needs a GPU — take `scripts/gpu-lock.sh` first.

use rdna_compute::{DType, Gpu, GpuTensor};

/// f16 operands, so parity against the f32 direct conv is a magnitude check.
/// A wrong tap or a wrong column order lands orders of magnitude above this.
const CONV_REL_L2_TOL: f64 = 2e-3;

fn download_u16(gpu: &Gpu, t: &GpuTensor) -> Vec<u16> {
    let numel: usize = t.shape.iter().product();
    let mut out = vec![0u16; numel];
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, numel * 2) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh f16");
    out
}

/// Deterministic pseudo-random fill — no rand dependency, and the same bytes
/// on every run so a failure is reproducible.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        })
        .collect()
}

fn upload_f16(gpu: &mut Gpu, data: &[f32], shape: &[usize]) -> GpuTensor {
    let staged = gpu.upload_f32(data, shape).expect("upload f32 staging");
    let t = gpu.alloc_tensor(shape, DType::F16).expect("alloc f16");
    gpu.cast_f32_to_f16(&staged, &t).expect("cast f16");
    gpu.free_tensor(staged).expect("free staging");
    t
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (*x as f64) - (*y as f64);
        num += d * d;
        den += (*y as f64) * (*y as f64);
    }
    if den == 0.0 {
        return num.sqrt();
    }
    (num / den).sqrt()
}

struct Shape {
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
}

/// A faithful mirror of `vae_gpu::Run::conv3x3`'s banded loop — same band
/// walk, same GEMM selection, same banded transpose with a non-zero
/// `dst_off`. `band = h` is the single-band case. Returns the channel-major
/// `[c_out][h*w]` result.
///
/// This is the only place the banded assembly is exercised outside the
/// product path, so it has to track `conv3x3` exactly; if that loop changes,
/// this must change with it.
#[allow(clippy::too_many_arguments)]
fn conv_route_banded(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w16: &GpuTensor,
    bias: &GpuTensor,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    band: usize,
) -> Vec<f32> {
    let hw = h * w;
    let k = c_in * 9;
    let y = gpu
        .alloc_tensor(&[c_out * hw], DType::F32)
        .expect("alloc y");
    let cols = gpu
        .alloc_tensor(&[band * w, k], DType::F16)
        .expect("alloc cols");
    let pos = gpu
        .alloc_tensor(&[band * w, c_out], DType::F32)
        .expect("alloc pos");
    let mut y0 = 0usize;
    while y0 < h {
        let rows = band.min(h - y0);
        let m = rows * w;
        gpu.vae_im2col_f16_band(x, &cols, c_in, h, w, y0, rows)
            .expect("im2col band");
        if k % 64 == 0 {
            gpu.gemm_f16_x_f16_wmma_lds_auto(w16, &cols, &pos, Some(bias), c_out, k, m)
                .expect("gemm lds");
        } else {
            gpu.gemm_f16_x_f16_wmma(w16, &cols, &pos, c_out, k, m)
                .expect("gemm 16");
            gpu.bias_add_f32(&pos, bias, m, c_out).expect("bias add");
        }
        gpu.vae_transpose_f32_banded(&pos, &y, m, c_out, hw, y0 * w)
            .expect("transpose banded");
        y0 += rows;
    }
    let out = gpu.download_f32(&y).expect("download y");
    for t in [y, cols, pos] {
        gpu.free_tensor(t).expect("free conv route");
    }
    out
}

const PARITY_SHAPES: &[Shape] = &[
    // conv_in: 16 latent channels -> 512, K = 144 (the %64 != 0 route that
    // takes the 16-step GEMM plus a separate bias_add).
    Shape {
        c_in: 16,
        c_out: 512,
        h: 16,
        w: 16,
    },
    // The deepest real conv, K = 4608.
    Shape {
        c_in: 512,
        c_out: 512,
        h: 24,
        w: 24,
    },
    Shape {
        c_in: 256,
        c_out: 256,
        h: 32,
        w: 32,
    },
    Shape {
        c_in: 128,
        c_out: 128,
        h: 48,
        w: 64,
    },
    // Ragged spatial dims: partial LDS tiles on both axes, and a c_out that
    // is not the c_in.
    Shape {
        c_in: 128,
        c_out: 64,
        h: 37,
        w: 53,
    },
];

fn main() {
    let bench = std::env::args().any(|a| a == "--bench");
    let mut gpu = Gpu::init().expect("GPU init failed");
    // The conv-route arms need the gfx11 wave32 WMMA GEMM (the gfx11
    // intrinsic hipcc rejects on gfx12). On other archs they skip —
    // announced here — while the im2col/transpose/gnorm arms still run.
    let gfx11_wmma_w32 = gpu.arch_caps.has_wmma_w32();
    if !gfx11_wmma_w32 {
        for k in ["gemm_f16_x_f16_wmma_lds", "gemm_f16_x_f16_wmma"] {
            println!("skip: {k} is gfx11 wave32 WMMA only (arch={})", gpu.arch);
        }
    }
    let mut failures = 0usize;

    // ── 1 + 2: im2col map and banding exactness ─────────────────────────
    for s in PARITY_SHAPES {
        let k = s.c_in * 9;
        let hw = s.h * s.w;
        let x = gpu
            .upload_f32(&fill(s.c_in * hw, 0x12c0 ^ s.c_in as u64), &[s.c_in * hw])
            .expect("upload x");

        let ref_cols = gpu.alloc_tensor(&[hw, k], DType::F16).expect("alloc ref");
        gpu.vae_im2col_f16_variant(&x, &ref_cols, s.c_in, s.h, s.w, 0, s.h, "c")
            .expect("im2col c");
        let want = download_u16(&gpu, &ref_cols);

        let lds_cols = gpu.alloc_tensor(&[hw, k], DType::F16).expect("alloc lds");
        gpu.vae_im2col_f16_variant(&x, &lds_cols, s.c_in, s.h, s.w, 0, s.h, "lds")
            .expect("im2col lds");
        let got = download_u16(&gpu, &lds_cols);

        let diff = want.iter().zip(&got).filter(|(a, b)| a != b).count();
        let ok = diff == 0;
        println!(
            "im2col map    c_in={:<4} {:>4}x{:<4} K={:<5} lds vs c: {} mismatching halves of {}  {}",
            s.c_in, s.h, s.w, k, diff, want.len(), if ok { "PASS" } else { "FAIL" }
        );
        failures += usize::from(!ok);

        // Banding: the same output assembled from 3-row bands. The band's own
        // top/bottom rows must still read the real neighbouring image rows,
        // not padding, which is the one thing banding can silently get wrong.
        let band = 3usize;
        let band_buf = gpu
            .alloc_tensor(&[band * s.w, k], DType::F16)
            .expect("alloc band");
        let mut band_diff = 0usize;
        let mut y0 = 0usize;
        while y0 < s.h {
            let rows = band.min(s.h - y0);
            gpu.vae_im2col_f16_variant(&x, &band_buf, s.c_in, s.h, s.w, y0, rows, "lds")
                .expect("im2col band");
            let got_band = download_u16(&gpu, &band_buf);
            let base = y0 * s.w * k;
            band_diff += (0..rows * s.w * k)
                .filter(|i| got_band[*i] != want[base + *i])
                .count();
            y0 += rows;
        }
        let ok = band_diff == 0;
        println!(
            "im2col band   c_in={:<4} {:>4}x{:<4} rows={band}: {} mismatching halves  {}",
            s.c_in,
            s.h,
            s.w,
            band_diff,
            if ok { "PASS" } else { "FAIL" }
        );
        failures += usize::from(!ok);

        // ── 3: full conv route vs the f32 direct convolution ────────────
        // Needs the gfx11 wave32 WMMA GEMM (skipped on other archs, announced
        // at startup); the im2col/banding checks above run everywhere.
        if gfx11_wmma_w32 {
            let wdata = fill(s.c_out * k, 0xc04f ^ s.c_out as u64);
            let bdata = fill(s.c_out, 0xb1a5);
            let w32 = gpu.upload_f32(&wdata, &[s.c_out, k]).expect("upload w f32");
            let bias = gpu.upload_f32(&bdata, &[s.c_out]).expect("upload bias");
            let direct = gpu
                .alloc_tensor(&[s.c_out * hw], DType::F32)
                .expect("alloc direct");
            gpu.vae_conv3x3_f32(&x, &w32, &bias, &direct, s.c_in, s.c_out, s.h, s.w)
                .expect("direct conv");
            let want_conv = gpu.download_f32(&direct).expect("download direct");

            let w16 = upload_f16(&mut gpu, &wdata, &[s.c_out, k]);
            let pos = gpu
                .alloc_tensor(&[hw, s.c_out], DType::F32)
                .expect("alloc pos");
            if k % 64 == 0 {
                gpu.gemm_f16_x_f16_wmma_lds_auto(
                    &w16,
                    &lds_cols,
                    &pos,
                    Some(&bias),
                    s.c_out,
                    k,
                    hw,
                )
                .expect("gemm lds");
            } else {
                gpu.gemm_f16_x_f16_wmma(&w16, &lds_cols, &pos, s.c_out, k, hw)
                    .expect("gemm 16");
                gpu.bias_add_f32(&pos, &bias, hw, s.c_out)
                    .expect("bias add");
            }
            let gemm_out = gpu
                .alloc_tensor(&[s.c_out * hw], DType::F32)
                .expect("alloc gemm out");
            gpu.vae_transpose_f32(&pos, &gemm_out, hw, s.c_out)
                .expect("transpose");
            let got_conv = gpu.download_f32(&gemm_out).expect("download gemm");
            let r = rel_l2(&got_conv, &want_conv);
            let ok = r <= CONV_REL_L2_TOL && r.is_finite();
            println!(
                "conv route    c_in={:<4} -> {:<4} {:>4}x{:<4} rel_l2 vs direct f32 = {r:.3e}  {}",
                s.c_in,
                s.c_out,
                s.h,
                s.w,
                if ok { "PASS" } else { "FAIL" }
            );
            failures += usize::from(!ok);
            for (t, what) in [
                (w32, "w32"),
                (w16, "w16"),
                (bias, "bias"),
                (direct, "direct"),
                (pos, "pos"),
                (gemm_out, "gemm_out"),
            ] {
                gpu.free_tensor(t)
                    .unwrap_or_else(|e| panic!("free {what}: {e:?}"));
            }
        }

        for (t, what) in [
            (x, "x"),
            (ref_cols, "ref"),
            (lds_cols, "lds"),
            (band_buf, "band"),
        ] {
            gpu.free_tensor(t)
                .unwrap_or_else(|e| panic!("free {what}: {e:?}"));
        }
    }

    // ── 3b: banded conv assembly, the product path's actual loop ────────
    //
    // Two things only banding can break, neither covered above: the banded
    // transpose writing at a non-zero `dst_off`, and a band's halo rows
    // reading the neighbouring band's image rows rather than padding. Each
    // shape below is chosen so the band count is several and the LAST band
    // is ragged.
    // Skipped with section 3: the banded assembly runs the WMMA GEMM.
    if gfx11_wmma_w32 {
        for (c_in, c_out, h, w, band) in [
            (128usize, 128usize, 37usize, 53usize, 8usize), // 5 bands, last = 5
            (256, 256, 20, 64, 6),                          // 4 bands, last = 2
            (512, 512, 15, 32, 4),                          // 4 bands, last = 3
        ] {
            let k = c_in * 9;
            let hw = h * w;
            let x = gpu
                .upload_f32(&fill(c_in * hw, 0xba7d ^ c_in as u64), &[c_in * hw])
                .expect("upload x");
            let wdata = fill(c_out * k, 0xba7e ^ c_out as u64);
            let w16 = upload_f16(&mut gpu, &wdata, &[c_out, k]);
            let bias = gpu
                .upload_f32(&fill(c_out, 0xba7f), &[c_out])
                .expect("upload bias");

            // Single band = the whole image in one GEMM, i.e. the unbanded route.
            let whole = conv_route_banded(&mut gpu, &x, &w16, &bias, c_in, c_out, h, w, h);
            let banded = conv_route_banded(&mut gpu, &x, &w16, &bias, c_in, c_out, h, w, band);

            let bands = h.div_ceil(band);
            let diff = whole
                .iter()
                .zip(&banded)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            // Not asserted bit-exact by construction: the LDS GEMM picks its
            // macro-tile from the batch size, which IS the band's row count, and
            // candidate tiles differ in k-step. Bit-exactness here is a measured
            // property of these shapes. The gate is the magnitude either way.
            let r = rel_l2(&banded, &whole);
            let ok = r <= CONV_REL_L2_TOL && r.is_finite();
            println!(
                "conv banded   c_in={c_in:<4} -> {c_out:<4} {h:>3}x{w:<3} band={band} ({bands} bands, \
                 last {}): {diff} of {} f32 differ, rel_l2 {r:.3e}  {}",
                h - band * (bands - 1),
                whole.len(),
                if ok { "PASS" } else { "FAIL" }
            );
            failures += usize::from(!ok);

            for t in [x, w16, bias] {
                gpu.free_tensor(t).expect("free banded conv");
            }
        }
    }

    // ── 3c: banded transpose in isolation — pure data movement, so this
    // one IS bit-exact by construction and a deviation is a bug. ─────────
    for (m, n, band) in [
        (37 * 53usize, 128usize, 8 * 53usize),
        (20 * 64, 256, 6 * 64),
    ] {
        let src = gpu
            .upload_f32(&fill(m * n, 0x7ab5), &[m, n])
            .expect("upload src");
        let whole_t = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc whole");
        gpu.vae_transpose_f32(&src, &whole_t, m, n)
            .expect("transpose whole");
        let want = gpu.download_f32(&whole_t).expect("download whole");

        let banded_t = gpu
            .alloc_tensor(&[n * m], DType::F32)
            .expect("alloc banded");
        let mut off = 0usize;
        while off < m {
            let rows = band.min(m - off);
            // The band's own source rows start at `off*n`; a sub-view is not
            // expressible as a GpuTensor here, so drive it through a copy of
            // the same launch the product path makes by offsetting the
            // destination only and slicing the source with a fresh upload.
            let sub = gpu
                .upload_f32(
                    &gpu.download_f32(&src).expect("download src")[off * n..(off + rows) * n],
                    &[rows, n],
                )
                .expect("upload sub");
            gpu.vae_transpose_f32_banded(&sub, &banded_t, rows, n, m, off)
                .expect("transpose banded");
            gpu.free_tensor(sub).expect("free sub");
            off += rows;
        }
        let got = gpu.download_f32(&banded_t).expect("download banded");
        let diff = want
            .iter()
            .zip(&got)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let ok = diff == 0;
        println!(
            "transpose band m={m:<6} n={n:<4} band={band}: {diff} of {} f32 differ  {}",
            want.len(),
            if ok { "PASS" } else { "FAIL" }
        );
        failures += usize::from(!ok);
        for t in [src, whole_t, banded_t] {
            gpu.free_tensor(t).expect("free transpose band");
        }
    }

    // ── 4: fused GroupNorm+SiLU vs the separate pair ────────────────────
    for (c, hw, groups) in [(512usize, 128 * 128usize, 32usize), (128, 64 * 64, 32)] {
        let x = gpu
            .upload_f32(&fill(c * hw, 0x9a05), &[c * hw])
            .expect("upload gn x");
        let gamma = gpu.upload_f32(&fill(c, 0x9a11), &[c]).expect("gamma");
        let beta = gpu.upload_f32(&fill(c, 0x9a22), &[c]).expect("beta");

        let n = gpu.alloc_tensor(&[c * hw], DType::F32).expect("alloc n");
        let sep = gpu.alloc_tensor(&[c * hw], DType::F32).expect("alloc sep");
        gpu.vae_groupnorm_f32(&x, &gamma, &beta, &n, c, hw, groups, 1e-6)
            .expect("groupnorm");
        gpu.silu_f32(&n, &sep).expect("silu");
        let want = gpu.download_f32(&sep).expect("download sep");

        let fused = gpu
            .alloc_tensor(&[c * hw], DType::F32)
            .expect("alloc fused");
        gpu.vae_groupnorm_silu_f32(&x, &gamma, &beta, &fused, c, hw, groups, 1e-6)
            .expect("groupnorm+silu");
        let got = gpu.download_f32(&fused).expect("download fused");

        let diff = want
            .iter()
            .zip(&got)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let ok = diff == 0;
        println!(
            "gnorm+silu    c={c:<4} hw={hw:<8} groups={groups}: {diff} differing f32 of {}  {}",
            want.len(),
            if ok { "PASS" } else { "FAIL" }
        );
        failures += usize::from(!ok);
        for t in [x, gamma, beta, n, sep, fused] {
            gpu.free_tensor(t).expect("free gn");
        }
    }

    if bench {
        bench_im2col(&mut gpu);
    }

    if failures > 0 {
        eprintln!("\n{failures} subtest(s) FAILED");
        std::process::exit(1);
    }
    println!("\nall subtests PASS");
}

/// DRAM roof of the box this is expected to run on (gfx1150 / Strix Point,
/// DDR5-5600 2x64-bit). Reported alongside the measurement so a number can be
/// read as a fraction of the roof rather than in isolation; override for
/// another machine with `HIPFIRE_DRAM_GBS`.
fn dram_roof_gbs() -> f64 {
    std::env::var("HIPFIRE_DRAM_GBS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(89.6)
}

fn bench_im2col(gpu: &mut Gpu) {
    // The three real FLUX VAE 3x3 conv shapes, deepest to widest.
    let shapes = [
        (512usize, 128usize, 128usize),
        (256, 512, 512),
        (128, 1024, 1024),
    ];
    let iters = 5;
    let roof = dram_roof_gbs();
    println!("\nim2col microbench (gfx1150 roof {roof} GB/s), {iters} timed iterations");
    println!(
        "{:<26} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "shape", "cols MB", "c ms", "lds ms", "lds GB/s", "of roof"
    );
    for (c_in, h, w) in shapes {
        let k = c_in * 9;
        let hw = h * w;
        let cols_bytes = hw * k * 2;
        let x = match gpu.upload_f32(&fill(c_in * hw, 0xbe0c), &[c_in * hw]) {
            Ok(t) => t,
            Err(e) => {
                println!("{c_in}ch @ {h}x{w}: input alloc failed ({e:?}) — skipped");
                continue;
            }
        };
        let cols = match gpu.alloc_tensor(&[hw, k], DType::F16) {
            Ok(t) => t,
            Err(e) => {
                println!(
                    "{c_in}ch @ {h}x{w}: {} MB column matrix alloc failed ({e:?}) — skipped",
                    cols_bytes / (1024 * 1024)
                );
                let _ = gpu.free_tensor(x);
                continue;
            }
        };
        // Useful traffic: every input element read once (what the LDS tiling
        // makes true) plus the 18 bytes of f16 taps it produces. The scalar
        // gather reads each element nine times, so it is charged the same
        // useful bytes and simply shows a lower effective rate.
        let bytes = (c_in * hw * 4 + hw * k * 2) as f64;
        let mut ms = [0.0f64; 2];
        for (slot, map) in ["c", "lds"].iter().enumerate() {
            // Warm the JIT for this (entry, shape) cell before timing.
            gpu.vae_im2col_f16_variant(&x, &cols, c_in, h, w, 0, h, map)
                .expect("im2col warm");
            gpu.hip.device_synchronize().expect("sync");
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                gpu.vae_im2col_f16_variant(&x, &cols, c_in, h, w, 0, h, map)
                    .expect("im2col timed");
            }
            gpu.hip.device_synchronize().expect("sync");
            ms[slot] = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        }
        let gbs = bytes / (ms[1] / 1000.0) / 1e9;
        println!(
            "{:<26} {:>10} {:>10.2} {:>10.2} {:>10.1} {:>7.0}%",
            format!("{c_in}ch @ {h}x{w}"),
            cols_bytes / (1024 * 1024),
            ms[0],
            ms[1],
            gbs,
            100.0 * gbs / roof
        );
        gpu.free_tensor(cols).expect("free cols");
        gpu.free_tensor(x).expect("free x");
    }
}
