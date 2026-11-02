// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-reference parity test for the rope_2d_flux_f32 kernel (FLUX.1 MMDiT 2D
//! axial RoPE). Verifies the GPU in-place image-row rotation
//! matches the CPU reference `flux::rope_2d` across the axial configs the
//! MMDiT needs.
//!
//! Cases:
//!  1. Real FLUX axes_dim [16,56,56], head_dim 128, text-first concat with
//!     img rows at the end (row_offset = n_text), grid (8, 8) → 64 img rows.
//!  2. Real FLUX layout, rectangular grid (shaper, 8×16 → 128 img rows).
//!  3. Lab geometry [8,8,16]=32, heads 12 — exercises a head_dim split.
//!  4. row_offset = 0 (all-image, no text stream).
//!
//! The CPU reference reproduces the rotation verbatim; both use the same
//! cosf/sinf/pow, so parity should be near-bit-exact. Tolerance relative to
//! output magnitude at 1e-5. Any subtest over tolerance → exit 1.
//!
//! Build: `cargo run --release --example test_rope_2d_flux -p rdna-compute`

use rdna_compute::Gpu;

const TOL: f32 = 1e-5;

/// CPU replica of `flux::rope_2d` (axial positions, interleaved pairs).
fn cpu_ref(
    mut x: Vec<f32>,
    row_offset: usize,
    n_img: usize,
    heads: usize,
    hd: usize,
    grid: (usize, usize),
    axes_dim: [usize; 4],
    theta: f64,
) -> Vec<f32> {
    let (_grid_h, grid_w) = grid;
    let mut pair_regions = [0usize; 4];
    let mut acc = 0usize;
    for (i, d) in axes_dim.iter().enumerate() {
        pair_regions[i] = acc;
        acc += d / 2;
    }
    let stride = heads * hd;
    for t in 0..n_img {
        let (row, col) = (t / grid_w, t % grid_w);
        let pos = [0.0f64, row as f64, col as f64, 0.0f64];
        for h in 0..heads {
            let base = ((row_offset + t) * heads + h) * hd;
            for (axis, &d_axis) in axes_dim.iter().enumerate() {
                for p in 0..d_axis / 2 {
                    let angle = pos[axis] / theta.powf(2.0 * p as f64 / d_axis as f64);
                    let (c, s) = (angle.cos() as f32, angle.sin() as f32);
                    let i = base + 2 * (pair_regions[axis] + p);
                    let (a, b) = (x[i], x[i + 1]);
                    x[i] = a * c - b * s;
                    x[i + 1] = a * s + b * c;
                }
            }
        }
    }
    let _ = stride;
    x
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    gpu: &mut Gpu,
    row_offset: usize,
    n_img: usize,
    heads: usize,
    hd: usize,
    grid: (usize, usize),
    axes_dim: [usize; 4],
    theta: f64,
    label: &str,
) -> usize {
    let n_all = row_offset + n_img;
    let data: Vec<f32> = (0..n_all * heads * hd)
        .map(|i| (((i * 7919) % 509) as f32 - 254.0) * 0.01)
        .collect();
    let want = cpu_ref(
        data.clone(),
        row_offset,
        n_img,
        heads,
        hd,
        grid,
        axes_dim,
        theta,
    );

    let g_x = gpu.upload_f32(&data, &[n_all, heads * hd]).unwrap();
    gpu.rope_2d_flux_f32(
        &g_x, row_offset, n_img, heads, hd, grid.1, axes_dim, theta, None,
    )
    .unwrap();
    let got = gpu.download_f32(&g_x).unwrap();

    let max_want = want.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-12);
    let mut max_err: f32 = 0.0;
    for (a, b) in got.iter().zip(want.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    let rel = max_err / max_want;
    if rel > TOL {
        eprintln!("{label}: FAIL max_err={max_err:.3e} rel={rel:.3e} tol={TOL} (max|want|={max_want:.3e})");
        1
    } else {
        println!("{label}: ok row_off={row_offset} img={n_img} {heads}x{hd} grid={grid:?} ax={axes_dim:?} rel={rel:.3e}");
        0
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    let mut fails = 0;
    let theta = 10000.0f64;

    // 1. Real FLUX [16,56,56,0]/128, text-first concat (n_text=64), 8x8 grid.
    fails += run_case(
        &mut gpu,
        64,
        64,
        24,
        128,
        (8, 8),
        [16, 56, 56, 0],
        theta,
        "flux-dev-8x8",
    );
    // 2. Real FLUX axes, rectangular 8x16 grid (128 img rows).
    fails += run_case(
        &mut gpu,
        32,
        128,
        24,
        128,
        (8, 16),
        [16, 56, 56, 0],
        theta,
        "flux-dev-8x16",
    );
    // 3. Lab geometry [8,8,16,0]=32, 12 heads.
    fails += run_case(
        &mut gpu,
        16,
        48,
        12,
        32,
        (8, 6),
        [8, 8, 16, 0],
        theta,
        "lab-12heads-32",
    );
    // 4. row_offset = 0 (all-image, no text).
    fails += run_case(
        &mut gpu,
        0,
        64,
        24,
        128,
        (8, 8),
        [16, 56, 56, 0],
        theta,
        "row-offset-0",
    );

    if fails > 0 {
        eprintln!("FAIL: {fails}/4 subtests failed");
        std::process::exit(1);
    }
    println!("PASS: rope_2d_flux_f32 parity vs CPU reference");
}
