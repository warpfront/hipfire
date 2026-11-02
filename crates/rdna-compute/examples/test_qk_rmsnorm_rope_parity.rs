// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity test for the fused `qk_rmsnorm_rope_flux` kernels against the
//! unfused launch pair they replace.
//!
//! Reference is the GPU path the FLUX MMDiT forward runs today, NOT a CPU
//! re-derivation: `Gpu::rmsnorm_batched` over `[n_all*heads, head_dim]`,
//! then `Gpu::rope_2d_flux_f32` with `row_offset = n_txt` (which dispatches
//! the tuned `rope_2d_flux_f32_fast` kernel by default). That pins the test
//! to the thing the fusion has to be indistinguishable from, including the
//! text rows the rope launch deliberately does not touch.
//!
//! Unlike the LayerNorm fusion, this one is NOT bit-identical and is not
//! claimed to be: the RMS reduction is 32 lanes x 4 values with a `shfl_xor`
//! butterfly where `rmsnorm_f32` is 128 threads x 1 value with an LDS halving
//! tree, and float addition is not associative. The rotation math is
//! bit-identical. Bars, per the task brief: f32 -> f32 within 1e-6 relative,
//! any f16 leg within 2e-3.
//!
//! The f16-input legs feed the reference the values the f16 tensor actually
//! holds — the host data rounded through f16 and widened back — so the two
//! paths see the same numbers and the comparison measures the kernel, not the
//! input quantization.
//!
//! Cases: real FLUX geometry (n_txt 512, n_img 4096, 24 heads, head_dim 128,
//! axes [16, 56, 56], theta 10000, 64x64 grid) in all four dtype
//! combinations, plus a ragged text-free case (n_txt = 0, an image row count
//! that is not a multiple of grid_w, and a unit count that is not a multiple
//! of the workgroup's wave count), a text-only case (n_img = 0, so every wave
//! takes the no-rotation path), and head_dim = 256, the launcher's ceiling,
//! where the per-lane pair buffer is exactly full.
//!
//! Run: cargo run --release -p rdna-compute --features lab \
//!          --example test_qk_rmsnorm_rope_parity

use rdna_compute::{DType, Gpu, GpuTensor};

const EPS: f32 = 1e-6;

struct Geom {
    n_txt: usize,
    n_img: usize,
    heads: usize,
    hd: usize,
    grid_w: usize,
    axes: [usize; 4],
    theta: f64,
}

impl Geom {
    fn n_all(&self) -> usize {
        self.n_txt + self.n_img
    }
    fn elems(&self) -> usize {
        self.n_all() * self.heads * self.hd
    }
}

fn host_x(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let a = (((i * 7919) % 1021) as f32 - 510.0) * 0.004;
            let b = (((i * 3571) % 89) as f32 - 44.0) * 0.011;
            a + b
        })
        .collect()
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let man = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        if man == 0 {
            sign << 31
        } else {
            // Subnormal: shift the leading 1 up into the implicit position.
            let mut e = -1i32;
            let mut m = man;
            loop {
                e += 1;
                m <<= 1;
                if m & 0x400 != 0 {
                    break;
                }
            }
            (sign << 31) | (((127 - 15 - e) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        (sign << 31) | 0x7f80_0000 | (man << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(out)
}

fn download_f16_as_f32(gpu: &Gpu, t: &GpuTensor) -> Vec<f32> {
    let numel = t.numel();
    let mut raw = vec![0u8; numel * 2];
    gpu.hip.memcpy_dtoh(&mut raw, &t.buf).expect("dtoh f16");
    raw.chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Upload `host` as F16 by way of the device cast, and return both the F16
/// tensor and the f32 values it actually holds.
fn upload_as_f16(gpu: &mut Gpu, host: &[f32], shape: &[usize]) -> (GpuTensor, Vec<f32>) {
    let src = gpu.upload_f32(host, shape).unwrap();
    let dst = gpu.zeros(shape, DType::F16).unwrap();
    gpu.cast_f32_to_f16(&src, &dst).unwrap();
    let rounded = download_f16_as_f32(gpu, &dst);
    gpu.free_tensor(src).unwrap();
    (dst, rounded)
}

/// `rmsnorm_batched` + `rope_2d_flux_f32` over `x`, on device, f32 throughout.
fn reference(gpu: &mut Gpu, g: &Geom, x: &[f32], scale: &GpuTensor) -> Vec<f32> {
    let rows = g.n_all() * g.heads;
    let g_x = gpu.upload_f32(x, &[rows, g.hd]).unwrap();
    let g_out = gpu.zeros(&[rows, g.hd], DType::F32).unwrap();
    gpu.rmsnorm_batched(&g_x, scale, &g_out, rows, g.hd, EPS)
        .unwrap();
    // `rope_2d_flux_f32` rejects n_img == 0, and rightly so — there is nothing
    // to rotate. The text-only reference is the norm alone, which is exactly
    // what the fused kernel must reduce to on that input.
    if g.n_img > 0 {
        gpu.rope_2d_flux_f32(
            &g_out, g.n_txt, g.n_img, g.heads, g.hd, g.grid_w, g.axes, g.theta, None,
        )
        .unwrap();
    }
    let out = gpu.download_f32(&g_out).unwrap();
    gpu.free_tensor(g_x).unwrap();
    gpu.free_tensor(g_out).unwrap();
    out
}

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    let max_want = want.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let max_err = got
        .iter()
        .zip(want.iter())
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    max_err / max_want
}

fn check(label: &str, got: &[f32], want: &[f32], tol: f32) -> usize {
    let rel = rel_err(got, want);
    if rel > tol || !rel.is_finite() {
        eprintln!("{label}: FAIL rel={rel:.3e} (tol {tol:.0e})");
        1
    } else {
        println!("{label}: ok rel={rel:.3e} (tol {tol:.0e})");
        0
    }
}

fn run_geom(gpu: &mut Gpu, g: &Geom, name: &str) -> usize {
    let n = g.elems();
    let x = host_x(n);
    let scale_host: Vec<f32> = (0..g.hd)
        .map(|i| 1.0 + (((i * 7013) % 103) as f32 - 51.0) * 0.004)
        .collect();
    let g_scale = gpu.upload_f32(&scale_host, &[g.hd]).unwrap();

    let rows = g.n_all() * g.heads;
    let (g_x16, x_rounded) = upload_as_f16(gpu, &x, &[rows, g.hd]);
    let g_x32 = gpu.upload_f32(&x, &[rows, g.hd]).unwrap();

    // Two references: the f32-input legs see `x`, the f16-input legs see the
    // f16-rounded values their tensor actually holds.
    let want32 = reference(gpu, g, &x, &g_scale);
    let want16 = reference(gpu, g, &x_rounded, &g_scale);

    let mut fails = 0;
    let fused = |gpu: &mut Gpu, x_t: &GpuTensor, out_dtype: DType| -> GpuTensor {
        let out = gpu.zeros(&[rows, g.hd], out_dtype).unwrap();
        gpu.qk_rmsnorm_rope_flux(
            x_t, &g_scale, &out, g.n_txt, g.n_img, g.heads, g.hd, g.grid_w, g.axes, g.theta, None,
        )
        .unwrap();
        out
    };

    let o = fused(gpu, &g_x32, DType::F32);
    fails += check(
        &format!("{name} f32->f32"),
        &gpu.download_f32(&o).unwrap(),
        &want32,
        1e-6,
    );
    gpu.free_tensor(o).unwrap();

    let o = fused(gpu, &g_x32, DType::F16);
    fails += check(
        &format!("{name} f32->f16"),
        &download_f16_as_f32(gpu, &o),
        &want32,
        2e-3,
    );
    gpu.free_tensor(o).unwrap();

    let o = fused(gpu, &g_x16, DType::F32);
    fails += check(
        &format!("{name} f16->f32"),
        &gpu.download_f32(&o).unwrap(),
        &want16,
        2e-3,
    );
    gpu.free_tensor(o).unwrap();

    let o = fused(gpu, &g_x16, DType::F16);
    fails += check(
        &format!("{name} f16->f16"),
        &download_f16_as_f32(gpu, &o),
        &want16,
        2e-3,
    );
    gpu.free_tensor(o).unwrap();

    gpu.free_tensor(g_x32).unwrap();
    gpu.free_tensor(g_x16).unwrap();
    gpu.free_tensor(g_scale).unwrap();
    fails
}

fn main() {
    eprintln!("=== test_qk_rmsnorm_rope_parity ===");
    let mut gpu = Gpu::init().expect("GPU init failed");
    let mut fails = 0;

    fails += run_geom(
        &mut gpu,
        &Geom {
            n_txt: 512,
            n_img: 4096,
            heads: 24,
            hd: 128,
            grid_w: 64,
            axes: [16, 56, 56, 0],
            theta: 10000.0,
        },
        "flux",
    );

    // n_txt = 0 (no text rows at all), an image row count that is not a
    // multiple of grid_w, and 51 (row, head) units against 8 waves per block.
    fails += run_geom(
        &mut gpu,
        &Geom {
            n_txt: 0,
            n_img: 17,
            heads: 3,
            hd: 8,
            grid_w: 5,
            axes: [2, 2, 4, 0],
            theta: 10000.0,
        },
        "ragged",
    );

    // Text-only: no image rows at all, so every wave takes the no-rotation
    // path and the kernel must reduce to a plain QK-RMSNorm. `grid_w` is a
    // dummy (it is never read when n_img == 0) but the launcher still
    // requires it non-zero.
    fails += run_geom(
        &mut gpu,
        &Geom {
            n_txt: 40,
            n_img: 0,
            heads: 3,
            hd: 8,
            grid_w: 1,
            axes: [2, 2, 4, 0],
            theta: 10000.0,
        },
        "text-only",
    );

    // head_dim at the launcher's ceiling: 128 pairs over 32 lanes fills the
    // MAX_PAIRS_PER_LANE = 4 register buffer exactly, so this is the shape
    // that would spill first if the buffer were ever mis-sized.
    fails += run_geom(
        &mut gpu,
        &Geom {
            n_txt: 8,
            n_img: 16,
            heads: 2,
            hd: 256,
            grid_w: 4,
            axes: [32, 112, 112, 0],
            theta: 10000.0,
        },
        "hd256",
    );

    if fails > 0 {
        eprintln!("FAIL: {fails} subtests failed");
        std::process::exit(1);
    }
    println!("PASS: qk_rmsnorm_rope_flux matches rmsnorm_batched + rope_2d_flux_f32");
}
