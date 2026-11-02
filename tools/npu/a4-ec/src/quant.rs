// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! CPU A4 quantizer (c2 candidates; copied from tools/npu/fold-model/src/activations.rs::quantize)
//! plus the A8 second-plane residual used by the selective-A8 correction.

/// Quantize one 128-block. Returns (d, q[128]).
pub fn quantize_block(chunk: &[f32], q_out: &mut [i8]) -> f32 {
    debug_assert_eq!(chunk.len(), 128);
    let max = chunk.iter().map(|x| x.abs()).fold(0f32, f32::max);
    if max == 0.0 {
        q_out.fill(0);
        return 1.0;
    }
    let base = (max / 7.0) * 0.5;
    let mut best_err = f32::INFINITY;
    let mut best_d = 0.0f32;
    let mut q = [0i8; 128];
    for multiplier in [f32::from_bits(0x3fdb6db7), 2.0] {
        let d = base * multiplier;
        for i in 0..128 {
            q[i] = (chunk[i] / d).round_ties_even().clamp(-8.0, 7.0) as i8;
        }
        let mut errors = [0.0f32; 32];
        for lane in 0..32 {
            for j in 0..4 {
                let i = lane * 4 + j;
                let error = (-(q[i] as f32)).mul_add(d, chunk[i]);
                errors[lane] = error.mul_add(error, errors[lane]);
            }
        }
        for delta in [16, 8, 4, 2, 1] {
            let previous = errors;
            for lane in 0..32 {
                errors[lane] = previous[lane] + previous[lane ^ delta];
            }
        }
        if errors[0] < best_err {
            best_err = errors[0];
            best_d = d;
            q_out.copy_from_slice(&q);
        }
    }
    best_d
}

/// A8 residual plane for one 128-block of r': d_lo = max|r|/7, q_lo = clamp(rint(r/d_lo),-8,7),
/// r'' = r - d_lo*q_lo. Writes r'' into `out`; returns d_lo (0 for an all-zero block).
pub fn a8_residual(r: &[f32], out: &mut [f32]) -> f32 {
    let max = r.iter().map(|x| x.abs()).fold(0f32, f32::max);
    if max == 0.0 {
        out.fill(0.0);
        return 0.0;
    }
    let d = max / 7.0;
    for i in 0..r.len() {
        let q = (r[i] / d).round_ties_even().clamp(-8.0, 7.0);
        out[i] = (-q).mul_add(d, r[i]);
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a8_residual_is_bounded_and_shrinks_energy() {
        let mut s = 12345u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32
        };
        let r: Vec<f32> = (0..128)
            .map(|i| rnd() * if i == 5 { 8.0 } else { 1.0 })
            .collect();
        let mut out = vec![0f32; 128];
        let d = a8_residual(&r, &mut out);
        let max = r.iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!((d - max / 7.0).abs() < 1e-7);
        // every value rounds to the nearest grid point -> |r''| <= d/2 (grid covers +-max exactly)
        assert!(
            out.iter().all(|v| v.abs() <= d * 0.5 + 1e-6),
            "residual exceeds d/2"
        );
        // reconstruction: r = d*q + r'' with integer q in [-8,7]
        for i in 0..128 {
            let q = (r[i] - out[i]) / d;
            assert!((q - q.round()).abs() < 1e-3 && q.round() >= -8.0 && q.round() <= 7.0);
        }
        let e0: f32 = r.iter().map(|x| x * x).sum();
        let e1: f32 = out.iter().map(|x| x * x).sum();
        assert!(
            e1 <= 128.0 * (d * 0.5) * (d * 0.5) * 1.001 && e1 < e0 * 0.25,
            "A8 residual energy bound: {e0} -> {e1}"
        );
        let mut z = vec![1f32; 128];
        assert_eq!(a8_residual(&[0.0; 128], &mut z), 0.0);
        assert!(z.iter().all(|v| *v == 0.0));
    }
    #[test]
    fn c2_clips_with_candidate_indices() {
        // same producer pins as fold-model: all-zero block -> d=1; q within [-8,7]
        let mut q = [1i8; 128];
        assert_eq!(quantize_block(&[0.0; 128], &mut q), 1.0);
        assert!(q.iter().all(|v| *v == 0));
        let mut x = [0.0f32; 128];
        x[0] = 7.0;
        x[1] = -7.0;
        let d = quantize_block(&x, &mut q);
        assert!(q.iter().all(|v| (-8..=7).contains(v)) && d > 0.0);
    }
}
