// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Host-side epilogues for CPU-executed steps.
//!
//! [`residual_add`] is the epilogue a CPU-executed `Step::GemvResidual` needs.
//! [`silu_mul`] is the epilogue a CPU-executed **routed-expert** FFN needs: the
//! pair of GEMVs that a `Step::Moe` fuses runs on the CPU for a spilled expert
//! blob, so the SwiGLU between them cannot stay on the GPU the way it does for a
//! dense layer (whose weight-reading matmul is the only half that moves). Both
//! mirror the device kernels (`silu_mul_f32`) element-for-element. The
//! gated/sigmoid-scaled epilogues are the MoE *shared*-expert and DeltaNet-output
//! forms, which this path does not cover.

/// `out[i] = silu(gate_up[i]) * gate_up[mi + i]` — the SwiGLU an MoE expert's
/// fused gate/up projection feeds its down projection with.
///
/// `gate_up` holds the concatenated `[gate(0..mi) | up(mi..2*mi)]` rows a fused
/// `gate_up` GEMV produces; `out` receives the `mi`-long hidden. Panics on a
/// length mismatch: a silently short hidden is a partial FFN, which reads as a
/// coherent model until it does not.
pub fn silu_mul(gate_up: &[f32], out: &mut [f32]) {
    let mi = out.len();
    assert!(
        gate_up.len() >= 2 * mi,
        "silu_mul: gate_up has {} elements, needs {}",
        gate_up.len(),
        2 * mi
    );
    for (o, (g, u)) in out
        .iter_mut()
        .zip(gate_up[..mi].iter().zip(gate_up[mi..2 * mi].iter()))
    {
        *o = (*g / (1.0 + (-*g).exp())) * *u;
    }
}

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

    #[test]
    fn silu_mul_matches_the_device_kernel_form() {
        // Same expression as `silu_mul_f32`: (v / (1 + exp(-v))) * up[i].
        let gate_up = [-1.0f32, 0.0, 1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 3];
        silu_mul(&gate_up, &mut out);
        let expect = |v: f32, u: f32| (v / (1.0 + (-v).exp())) * u;
        assert_eq!(out[0], expect(-1.0, 2.0));
        assert_eq!(out[1], expect(0.0, 3.0));
        assert_eq!(out[2], expect(1.0, 4.0));
    }

    #[test]
    #[should_panic(expected = "needs 8")]
    fn silu_mul_rejects_a_short_gate_up() {
        let mut out = [0.0f32; 4];
        silu_mul(&[1.0, 2.0, 3.0], &mut out);
    }
}
