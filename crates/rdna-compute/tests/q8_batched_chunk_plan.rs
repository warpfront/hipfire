// SPDX-License-Identifier: Apache-2.0
// #732: the scalar `gemm_q8_0_batched` kernel takes at most 64 activation rows
// per launch; `Gpu::gemm_q8_0_batched` splits larger calls with
// `q8_0_batched_chunks`. CPU-only: no HIP, no GPU.
use rdna_compute::gemm::{q8_0_batched_chunks, Q8_0_BATCHED_MAX_BATCH};

const BOUNDARY_ROWS: [usize; 5] = [63, 64, 65, 70, 129];

/// Every launch is within the kernel cap; launches are consecutive, start at
/// row 0 and end at `n`, so each row is computed exactly once. Also checks the
/// element offsets the wrapper derives (`off*k` into x, `off*m` into y) stay in
/// the `[n, k]` / `[n, m]` buffers.
fn assert_plan(n: usize, expect: &[(usize, usize)]) {
    let plan: Vec<(usize, usize)> = q8_0_batched_chunks(n).collect();
    assert_eq!(plan, expect, "plan for n={n}");
    let mut next = 0;
    for &(off, take) in &plan {
        assert_eq!(off, next, "n={n}: launch must start where the last ended");
        assert!(
            (1..=Q8_0_BATCHED_MAX_BATCH).contains(&take),
            "n={n}: launch of {take} rows breaks the kernel cap"
        );
        next = off + take;
    }
    assert_eq!(next, n, "n={n}: rows covered");
    let (k, m) = (2560, 248_320);
    for &(off, take) in &plan {
        assert!((off + take) * k <= n * k, "n={n}: x window past the input");
        assert!((off + take) * m <= n * m, "n={n}: y window past the output");
    }
}

#[test]
fn kernel_cap_is_64() {
    assert_eq!(Q8_0_BATCHED_MAX_BATCH, 64);
}

#[test]
fn boundary_row_counts_plan() {
    assert_plan(63, &[(0, 63)]);
    assert_plan(64, &[(0, 64)]);
    assert_plan(65, &[(0, 64), (64, 1)]);
    assert_plan(70, &[(0, 64), (64, 6)]);
    assert_plan(129, &[(0, 64), (64, 64), (128, 1)]);
    for n in BOUNDARY_ROWS {
        assert!(q8_0_batched_chunks(n).all(|(_, t)| t <= 64));
    }
}

#[test]
fn small_and_empty_row_counts_plan() {
    assert_plan(0, &[]);
    assert_plan(1, &[(0, 1)]);
    assert_plan(128, &[(0, 64), (64, 64)]);
    assert_plan(256, &[(0, 64), (64, 64), (128, 64), (192, 64)]);
}

#[test]
fn every_row_count_to_1024_is_covered_once() {
    for n in 0..=1024usize {
        let mut seen = vec![0u8; n];
        for (off, take) in q8_0_batched_chunks(n) {
            assert!((1..=64).contains(&take));
            for row in &mut seen[off..off + take] {
                *row += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "n={n}");
    }
}
