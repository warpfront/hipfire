// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Host-side epilogues for CPU-executed steps.
//!
//! Only the residual accumulate lives here today, because it is the only
//! epilogue the CPU seam needs: the SiLU that feeds a fused down-projection is
//! still computed by the GPU (`silu_mul_f32`) and only the weight-reading matmul
//! moves to the CPU, so a `silu_mul` here would be a second implementation with
//! no caller. The gated/sigmoid-scaled epilogues are the MoE shared-expert and
//! DeltaNet-output forms, which this path does not cover.

/// `acc[j] += delta[j]`.
///
/// `acc` may be longer than `delta` (the seam hands it the full residual row);
/// the tail is left alone. A `delta` longer than `acc` is a caller bug — a
/// silent partial add is exactly the kind of thing that reads as a coherent
/// model until it does not — so it panics.
pub fn residual_add(acc: &mut [f32], delta: &[f32]) {
    assert!(
        acc.len() >= delta.len(),
        "residual_add: acc has {} elements, delta has {}",
        acc.len(),
        delta.len()
    );
    for (a, d) in acc.iter_mut().zip(delta) {
        *a += d;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn residual_add_accumulates_and_leaves_the_tail_alone() {
        let mut acc = [1.0f32, -2.0, 0.5, 100.0, 200.0];
        residual_add(&mut acc, &[0.25, 0.25, -1.5]);
        assert_eq!(acc, [1.25, -1.75, -1.0, 100.0, 200.0]);
    }

    #[test]
    fn residual_add_is_exact_for_representable_values() {
        // Same magnitudes the seam hands over: an f32 add per element, no
        // reassociation, so a small-integer fixture is exact by construction.
        let delta = [1.0f32, 2.0, 4.0, 8.0];
        let mut acc = [0.0f32; 4];
        for _ in 0..4 {
            residual_add(&mut acc, &delta);
        }
        assert_eq!(acc, [4.0, 8.0, 16.0, 32.0]);
    }

    #[test]
    #[should_panic(expected = "delta has 3")]
    fn residual_add_rejects_a_longer_delta() {
        let mut acc = [0.0f32; 2];
        residual_add(&mut acc, &[1.0, 2.0, 3.0]);
    }
}
