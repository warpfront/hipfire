// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity test for the fused `layernorm_modulate` kernels against the
//! unfused launch chain they replace.
//!
//! Reference is the GPU path the FLUX MMDiT forward runs today, NOT a CPU
//! re-derivation: `Gpu::layernorm_batched` with gamma = 1 / beta = 0, then
//! `Gpu::modulate_f32`, then (for the f16 entry) `Gpu::cast_f32_to_f16`.
//! Comparing against a CPU formula would only show that both are "about
//! right"; comparing against the chain is what proves the fusion is a no-op
//! on the numbers, which is what the golden-latent gate needs.
//!
//! The bar is therefore BIT-IDENTITY, not a tolerance: every f32 output word
//! must match the chain's f32 output word, and every f16 output word must
//! match the chain's cast output word. See
//! `kernels/src/layernorm_modulate_f32.hip` for why that is achievable
//! (matching reduction tree, matching expression order).
//!
//! Shapes: the FLUX image-token stream [4608, 3072] and the text stream
//! [512, 3072], then a ladder of small shapes chosen to land in each of
//! `ln_block_sum`'s block-size regimes — ragged strided accumulation, idle
//! threads, and the sub-wave butterfly widths. See the comments in `main`;
//! the launcher's `min(256, d).next_power_of_two()` block size is what
//! selects the regime, so the shapes are picked by `d`, not by row count.
//!
//! Run: cargo run --release -p rdna-compute --features lab \
//!          --example test_layernorm_modulate_parity

use rdna_compute::{DType, Gpu, GpuTensor};

const EPS: f32 = 1e-6;

/// Deterministic, sign-mixed, non-trivial magnitudes — a constant fill would
/// make the variance zero and hide any reduction-order difference.
fn host_x(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let a = (((i * 7919 + salt * 104_729) % 1021) as f32 - 510.0) * 0.011;
            let b = (((i * 3571 + salt * 7717) % 97) as f32 - 48.0) * 0.003;
            a + b
        })
        .collect()
}

fn host_vec(d: usize, mul: usize, m: usize, off: f32, k: f32) -> Vec<f32> {
    (0..d).map(|i| (((i * mul) % m) as f32 - off) * k).collect()
}

/// Raw f16 words of a device tensor. `download_f32` would read 4 bytes per
/// element out of a 2-byte-per-element buffer, so go through the runtime.
fn download_f16_bits(gpu: &Gpu, t: &GpuTensor) -> Vec<u16> {
    let numel = t.numel();
    let mut raw = vec![0u8; numel * 2];
    gpu.hip.memcpy_dtoh(&mut raw, &t.buf).expect("dtoh f16");
    raw.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn first_mismatch<T: PartialEq>(a: &[T], b: &[T]) -> Option<usize> {
    a.iter().zip(b.iter()).position(|(x, y)| x != y)
}

fn run_case(gpu: &mut Gpu, n_rows: usize, d: usize, label: &str) -> usize {
    let x = host_x(n_rows * d, n_rows + d);
    let shift = host_vec(d, 2027, 97, 48.0, 0.01);
    let scale = host_vec(d, 7013, 103, 51.0, 0.02);
    let ones = vec![1.0f32; d];
    let zeros = vec![0.0f32; d];

    let g_x = gpu.upload_f32(&x, &[n_rows, d]).unwrap();
    let g_shift = gpu.upload_f32(&shift, &[d]).unwrap();
    let g_scale = gpu.upload_f32(&scale, &[d]).unwrap();
    let g_ones = gpu.upload_f32(&ones, &[d]).unwrap();
    let g_zeros = gpu.upload_f32(&zeros, &[d]).unwrap();

    // Reference: the three-launch chain the forward runs today.
    let g_ref = gpu.zeros(&[n_rows, d], DType::F32).unwrap();
    gpu.layernorm_batched(&g_x, &g_ones, &g_zeros, &g_ref, n_rows, d, EPS)
        .unwrap();
    gpu.modulate_f32(&g_ref, &g_shift, &g_scale, &g_ref, n_rows, d)
        .unwrap();
    let want_f32 = gpu.download_f32(&g_ref).unwrap();

    let g_ref16 = gpu.zeros(&[n_rows, d], DType::F16).unwrap();
    gpu.cast_f32_to_f16(&g_ref, &g_ref16).unwrap();
    let want_f16 = download_f16_bits(gpu, &g_ref16);

    let mut fails = 0;

    // Fused, f32 out.
    let g_out = gpu.zeros(&[n_rows, d], DType::F32).unwrap();
    gpu.layernorm_modulate(&g_x, &g_shift, &g_scale, &g_out, n_rows, d, EPS)
        .unwrap();
    let got_f32 = gpu.download_f32(&g_out).unwrap();
    match first_mismatch(
        &got_f32.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        &want_f32.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
    ) {
        None => {
            println!("{label} f32: ok {n_rows}x{d} bit-identical to layernorm_batched+modulate_f32")
        }
        Some(i) => {
            let (g, w) = (got_f32[i], want_f32[i]);
            let rel = (g - w).abs() / w.abs().max(1e-12);
            eprintln!(
                "{label} f32: FAIL at {i} got {g:e} ({:08x}) want {w:e} ({:08x}) rel={rel:.3e}",
                g.to_bits(),
                w.to_bits()
            );
            fails += 1;
        }
    }

    // Fused, f16 out.
    let g_out16 = gpu.zeros(&[n_rows, d], DType::F16).unwrap();
    gpu.layernorm_modulate(&g_x, &g_shift, &g_scale, &g_out16, n_rows, d, EPS)
        .unwrap();
    let got_f16 = download_f16_bits(gpu, &g_out16);
    match first_mismatch(&got_f16, &want_f16) {
        None => println!(
            "{label} f16: ok {n_rows}x{d} bit-identical to the same chain + cast_f32_to_f16"
        ),
        Some(i) => {
            eprintln!(
                "{label} f16: FAIL at {i} got {:04x} want {:04x}",
                got_f16[i], want_f16[i]
            );
            fails += 1;
        }
    }

    for t in [
        g_x, g_shift, g_scale, g_ones, g_zeros, g_ref, g_ref16, g_out, g_out16,
    ] {
        gpu.free_tensor(t).unwrap();
    }
    fails
}

fn main() {
    eprintln!("=== test_layernorm_modulate_parity ===");
    let mut gpu = Gpu::init().expect("GPU init failed");
    let mut fails = 0;

    // Real FLUX image-token stream, hidden width 3072.
    fails += run_case(&mut gpu, 4608, 3072, "img-4608");
    // Real FLUX text stream.
    fails += run_case(&mut gpu, 512, 3072, "txt-512");
    // Below is a ladder over `ln_block_sum`'s block-size regimes. The block
    // is `min(256, d).next_power_of_two()`, and the reduction splits at the
    // wave boundary: LDS halving while `s >= 32`, `shfl_xor` butterfly below,
    // with the butterfly width `tail = min(blockDim.x, 32)`. Each rung below
    // lands in a different regime, and each is asserted bit-identical.

    // One row, block 64 = two wave32: one LDS level (s = 32), then a 32-wide
    // butterfly. Exercises the multi-wave path at a non-FLUX width.
    fails += run_case(&mut gpu, 1, 64, "1x64-block64");

    // d not a multiple of the block: 300 over a 256-thread block gives
    // threads 0..43 two elements and 44..255 one, so the strided accumulation
    // loop runs a ragged tail and the per-thread partials are unequal.
    fails += run_case(&mut gpu, 5, 300, "5x300-ragged-stride");

    // d < blockDim: 100 rounds up to a 128-thread block, so threads 100..127
    // contribute nothing and their LDS slots must still reduce as zeros.
    fails += run_case(&mut gpu, 3, 100, "3x100-idle-threads");

    // Genuinely sub-wave: 12 rounds up to a 16-thread block, so `tail` is 16,
    // not 32, and the butterfly must narrow with it. This is the branch
    // `ln_block_sum` writes `tail` for; nothing above reaches it.
    fails += run_case(&mut gpu, 2, 12, "2x12-subwave16");

    // Narrower still: 5 rounds up to an 8-thread block, `tail` = 8.
    fails += run_case(&mut gpu, 4, 5, "4x5-subwave8");

    if fails > 0 {
        eprintln!("FAIL: {fails} subtests failed");
        std::process::exit(1);
    }
    println!("PASS: layernorm_modulate f32/f16 bit-identical to the unfused chain");
}
