// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `gemm_q8_0_batched` above the kernel's MAX_BATCH = 64 (#732): any row count
//! must run, and row r of a B-row call must hold the exact bytes of a one-row
//! call on row r (the kernel's per-row reduction does not depend on B). Rows
//! cross the 64-row boundary (64/65/70) and several chunks (128/129/256). The
//! output starts as a NaN sentinel with one canary row past the end, so a row
//! the call skips, misplaces or writes past the end fails.
//!
//! `#[ignore]`d: needs a HIP GPU. Run explicitly:
//!
//!   cargo test -p rdna-compute --release --test gemm_q8_0_batched_rows -- --ignored

use rdna_compute::{Gpu, GpuTensor};

const M: usize = 97;
const K: usize = 256;
const Q8_ROW_BYTES: usize = K / 32 * 34;
const SENTINEL: u32 = 0x7fc0_dead;

fn next(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 24
}

fn sentinel(gpu: &mut Gpu, n: usize) -> GpuTensor {
    gpu.upload_f32(&vec![f32::from_bits(SENTINEL); n], &[n])
        .expect("upload sentinel")
}

fn bits(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.download_f32(t)
        .expect("download")
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

#[test]
#[ignore = "needs a HIP GPU"]
fn gemm_q8_0_batched_rows_past_max_batch_match_one_row_calls() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut seed = 0x0732_0064u64;
    // f16 scale in [2^-5, 2^-3) and 32 int8 quants per 34-byte block.
    let mut w = Vec::with_capacity(M * Q8_ROW_BYTES);
    for _ in 0..M * (K / 32) {
        w.extend_from_slice(&(0x2800u16 + (next(&mut seed) % 0x0800) as u16).to_le_bytes());
        w.extend((0..32).map(|_| next(&mut seed) as u8));
    }
    let w = gpu.upload_raw(&w, &[M * Q8_ROW_BYTES]).expect("upload w");

    for b in [1usize, 63, 64, 65, 70, 128, 129, 256] {
        let x_data: Vec<f32> = (0..b * K)
            .map(|_| (next(&mut seed) as u32 & 0xff_ffff) as f32 / (1u32 << 24) as f32 * 4.0 - 2.0)
            .collect();
        let x = gpu.upload_f32(&x_data, &[b * K]).expect("upload x");

        let y = sentinel(&mut gpu, (b + 1) * M);
        gpu.gemm_q8_0_batched(&w, &x, &y, M, K, b)
            .unwrap_or_else(|e| panic!("B={b}: {e:?}"));
        let got = bits(&gpu, &y);
        assert!(
            got[b * M..].iter().all(|&v| v == SENTINEL),
            "B={b}: wrote past the last row"
        );

        let y1 = sentinel(&mut gpu, M);
        for r in 0..b {
            gpu.gemm_q8_0_batched(&w, &x.sub_offset(r * K, K), &y1, M, K, 1)
                .expect("one-row call");
            let want = bits(&gpu, &y1);
            assert!(
                want.iter().all(|&v| v != SENTINEL),
                "B={b} row {r}: one-row call left a hole"
            );
            assert_eq!(&got[r * M..(r + 1) * M], &want[..], "B={b} row {r}");
        }
    }
}
