// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU oracle helpers mirroring the scalar parts of llama.cpp's
//! `inkling2.cpp` graph. These are correctness references, not GPU kernels.

/// RMSNorm with no learned gain (all Inkling-2 norms are gamma-less).
pub fn rmsnorm(x: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter().map(|v| v * scale).collect()
}

/// Row L2 clip to `threshold`, after an optional per-channel scale.
/// Mirrors `llama_model_inkling2::graph::clip`: eps is added to the norm and
/// the factor `threshold / norm` is clamped to [0, 1].
pub fn clip(cur: &[f32], scale: Option<&[f32]>, threshold: f32) -> Vec<f32> {
    let scaled: Vec<f32> = match scale {
        Some(s) => cur.iter().zip(s.iter()).map(|(v, k)| v * k).collect(),
        None => cur.to_vec(),
    };
    let norm = scaled.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-6;
    let factor = (threshold / norm).clamp(0.0, 1.0);
    scaled.iter().map(|v| v * factor).collect()
}

/// Log-scaled position multiplier applied to Q and R on full-attention layers.
/// Mirrors `tau = 1 + alpha * log(max(1, (pos+1)/floor))`.
pub fn tau(pos: usize, floor: usize, alpha: f32) -> f32 {
    let ratio = ((pos + 1) as f64 / floor as f64).max(1.0);
    1.0 + alpha * ratio.ln() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn rmsnorm_is_gamma_less() {
        // RMS = sqrt((9 + 16)/2) = 3.5355..., not the L2 norm 5.
        let out = rmsnorm(&[3.0, 4.0], 1e-6);
        assert_eq!(out.len(), 2);
        assert!(close(out[0], 0.8485281), "{}", out[0]);
        assert!(close(out[1], 1.1313708), "{}", out[1]);
    }

    #[test]
    fn clip_scales_down_only_when_above_threshold() {
        let out = clip(&[3.0, 4.0], None, 2.0);
        assert!(close(out[0], 1.2), "{}", out[0]);
        assert!(close(out[1], 1.6), "{}", out[1]);
        let out = clip(&[3.0, 4.0], None, 10.0);
        assert!(close(out[0], 3.0) && close(out[1], 4.0));
    }

    #[test]
    fn clip_applies_optional_channel_scale_first() {
        let out = clip(&[3.0, 4.0], Some(&[2.0, 0.5]), 10.0);
        assert!(close(out[0], 6.0), "{}", out[0]);
        assert!(close(out[1], 2.0), "{}", out[1]);
    }

    #[test]
    fn tau_clamps_floor_and_scales_log() {
        assert_eq!(tau(0, 8192, 0.1), 1.0);
        assert_eq!(tau(8191, 8192, 0.1), 1.0);
        let t = tau(16383, 8192, 0.1);
        assert!(close(t, 1.0 + 0.1 * 2.0f32.ln()), "{}", t);
    }
}
