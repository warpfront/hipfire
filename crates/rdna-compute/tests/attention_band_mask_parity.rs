// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The embeddinggemma2 (arch 18) sliding-window eager attention is two
//! kernels composed: `attention_scores_band_mask_f32` writes a large finite
//! negative over every score outside `|row - col| <= window`, then
//! `attention_rows_softmax_f16` softmaxes each row over `[valid_start,
//! valid_end)`. Both halves of that contract are silent-failure shaped:
//!
//! - the window is an **inclusive radius** (`abs(q - kv) <= window`, the HF
//!   `sliding_window_bidirectional_overlay` predicate). An off-by-one here
//!   keeps or drops exactly one column per row and changes the embedding by
//!   an O(1) amount with no error — this asserts the boundary cell directly.
//! - the sentinel must be **finite** (`-3.0e38`, not `-inf`): the row softmax
//!   computes `exp(x - rowmax)`, so a `-inf` sentinel yields `inf - inf` = NaN
//!   on any degenerate row. The test asserts the masked columns come back as
//!   exactly 0 probability and the rows still sum to 1.
//!
//! `#[ignore]`d: needs a GPU with a working HIP toolchain. Run explicitly:
//!
//!   cargo test -p rdna-compute --release --test attention_band_mask_parity \
//!       -- --ignored --test-threads=1

use rdna_compute::{DType, Gpu};

/// Rows == cols and >= 256 so the softmax block caps at 256 (production
/// shape); a power of two so no column is ever left half-full.
const N: usize = 512;
/// The arch's window: an inclusive radius of 512, which at N=512 means the
/// band covers the whole matrix — so the boundary is exercised with a
/// *smaller* window where `|i-j| == window` is an interior cell.
const WINDOW: usize = 7;

const SENTINEL: f32 = -3.0e38;

fn fill(n: usize, scale: f32) -> Vec<f32> {
    // Deterministic, distinct per cell so an off-by-one changes a value.
    (0..n)
        .map(|i| (((i * 2654435761) % 1000) as f32 / 1000.0 - 0.5) * 2.0 * scale)
        .collect()
}

#[test]
#[ignore]
fn band_mask_is_an_inclusive_radius_with_a_finite_sentinel() {
    let mut gpu = Gpu::init().expect("GPU init");
    let scores = gpu.upload_f32(&fill(N * N, 4.0), &[N, N]).expect("upload");
    let before = gpu.download_f32(&scores).expect("download");

    gpu.attention_scores_band_mask_f32(&scores, N, N, WINDOW)
        .expect("band mask");
    let after = gpu.download_f32(&scores).expect("download");

    for row in 0..N {
        for col in 0..N {
            let d = (row as i64 - col as i64).abs() as usize;
            let got = after[row * N + col];
            if d <= WINDOW {
                // In band: the score must be untouched, bit for bit.
                assert_eq!(
                    got.to_bits(),
                    before[row * N + col].to_bits(),
                    "in-band ({row},{col}) d={d} altered"
                );
            } else {
                assert_eq!(got, SENTINEL, "out-of-band ({row},{col}) d={d}");
            }
        }
    }

    // The exact boundary cells: |i-j| == window is IN band (inclusive radius);
    // |i-j| == window+1 is OUT. A `<` vs `<=` slip flips both.
    let at = |r: usize, c: usize| after[r * N + c];
    for row in 0..N {
        for col in [row + WINDOW, row + WINDOW + 1] {
            if col >= N {
                continue;
            }
            if col == row + WINDOW {
                assert!(at(row, col) != SENTINEL, "d==window must be in band");
            } else {
                assert_eq!(at(row, col), SENTINEL, "d==window+1 must be masked");
            }
        }
    }

    // The sentinel is finite and far enough below the band that exp underflows
    // to exactly 0 in the row softmax, so a fully-except-diagonal row is not
    // NaN. (The diagonal is always in band, so no row is ever fully masked.)
    assert!(SENTINEL.is_finite() && SENTINEL < f32::MIN / 2.0);

    // Composition: the row softmax over [0, N) must give the masked columns
    // exactly 0 and each row must sum to 1 with no NaN.
    let probs = gpu.alloc_tensor(&[N, N], DType::F16).expect("alloc f16");
    gpu.attention_rows_softmax_f16(&scores, &probs, N, N, 0, N)
        .expect("softmax");
    let bits = gpu.download_f16_bits(&probs).expect("download f16");
    let p: Vec<f32> = bits
        .iter()
        .map(|b| half::f16::from_bits(*b).to_f32())
        .collect();

    for row in 0..N {
        let mut sum = 0.0f32;
        for col in 0..N {
            let v = p[row * N + col];
            assert!(v.is_finite(), "non-finite probability at ({row},{col})");
            if (row as i64 - col as i64).abs() as usize > WINDOW {
                assert_eq!(v, 0.0, "masked column ({row},{col}) leaked {v}");
            }
            sum += v;
        }
        assert!((sum - 1.0).abs() < 5e-3, "row {row} probability sum {sum}");
    }

    for t in [scores, probs] {
        let _ = gpu.free_tensor(t);
    }
}
