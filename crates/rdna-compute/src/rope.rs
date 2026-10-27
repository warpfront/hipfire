// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Rotary embedding description shared by every architecture.
//!
//! A family declares its RoPE as a [`Rope`] (base + [`RopeScaling`], parsed
//! once from the checkpoint by the engine) and hands it to the launchers;
//! [`Rope::yarn`] lowers it for one rotary width into the [`RopeYarn`] kernel
//! argument (`kernels/src/rope_yarn.h`). Scaling math lives here and in that
//! header only, so a family never implements YaRN itself.

/// A rotary embedding as the checkpoint declares it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rope {
    pub theta: f64,
    pub scaling: RopeScaling,
}

/// Long-context RoPE scaling (`rope_parameters.rope_type`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum RopeScaling {
    #[default]
    None,
    Yarn(Yarn),
}

/// YaRN parameters, with HF `_compute_yarn_parameters` defaults resolved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yarn {
    pub factor: f64,
    pub original_max_position_embeddings: usize,
    pub beta_fast: f64,
    pub beta_slow: f64,
    /// cos/sin multiplier (HF `attention_factor`; default `0.1 ln(factor) + 1`).
    pub attention_factor: f64,
}

impl Yarn {
    /// HF defaults: `beta_fast` 32, `beta_slow` 1, and `attention_factor`
    /// from `factor` (plus `mscale`/`mscale_all_dim` when both are given).
    /// `factor` must be finite and >= 1: below 1 the correction ramp inverts
    /// and [`RopeScaling::context_limit`] shrinks below the original context.
    pub fn new(
        factor: f64,
        original_max_position_embeddings: usize,
        beta_fast: Option<f64>,
        beta_slow: Option<f64>,
        attention_factor: Option<f64>,
        mscale: Option<(f64, f64)>,
    ) -> Self {
        assert!(
            factor.is_finite() && factor >= 1.0,
            "YaRN factor {factor} must be finite and >= 1"
        );
        let get_mscale = |m: f64| {
            if factor <= 1.0 {
                1.0
            } else {
                0.1 * m * factor.ln() + 1.0
            }
        };
        let attention_factor = attention_factor.unwrap_or_else(|| match mscale {
            Some((m, all)) if m != 0.0 && all != 0.0 => get_mscale(m) / get_mscale(all),
            _ => get_mscale(1.0),
        });
        Self {
            factor,
            original_max_position_embeddings,
            beta_fast: beta_fast.unwrap_or(32.0),
            beta_slow: beta_slow.unwrap_or(1.0),
            attention_factor,
        }
    }
}

impl RopeScaling {
    /// Positions the scaled embedding covers: `factor x original` under YaRN,
    /// otherwise the checkpoint's `max_position_embeddings`.
    pub fn context_limit(&self, max_position_embeddings: usize) -> usize {
        match self {
            Self::None => max_position_embeddings,
            Self::Yarn(y) => (y.original_max_position_embeddings as f64 * y.factor) as usize,
        }
    }
}

/// By-value kernel argument; layout matches `struct RopeYarn` in
/// `kernels/src/rope_yarn.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeYarn {
    pub freq_scale: f32,
    pub corr_low: f32,
    pub corr_high: f32,
    pub mscale: f32,
}

impl RopeYarn {
    /// Plain RoPE: the kernels leave angle and sin/cos untouched.
    pub const OFF: Self = Self {
        freq_scale: 1.0,
        corr_low: 0.0,
        corr_high: 0.0,
        mscale: 1.0,
    };

    /// Append the four fields in declaration order.
    pub fn push(&self, args: &mut hip_bridge::KernargBlob) {
        for v in [self.freq_scale, self.corr_low, self.corr_high, self.mscale] {
            args.push_f32(v);
        }
    }

    /// Host twin of `rope_yarn_sincos` (`kernels/src/rope_yarn.h`), same f32
    /// operation order: `(sin, cos)` of `angle` (= position * inv_freq[pair]).
    /// [`Self::OFF`] is `angle.sin_cos()`, so unscaled host references keep
    /// their results. Not bit-equal to HF, which scales `inv_freq` before the
    /// position multiply.
    pub fn sin_cos(&self, mut angle: f32, pair: usize) -> (f32, f32) {
        if self.freq_scale != 1.0 {
            let span = self.corr_high - self.corr_low;
            let span = if span == 0.0 { 0.001 } else { span };
            let ramp = ((pair as f32 - self.corr_low) / span).clamp(0.0, 1.0);
            angle *= ramp * self.freq_scale + (1.0 - ramp);
        }
        let (sine, cosine) = angle.sin_cos();
        if self.mscale != 1.0 {
            (sine * self.mscale, cosine * self.mscale)
        } else {
            (sine, cosine)
        }
    }
}

impl Rope {
    /// Kernel argument for a rotary width of `rotary_dim` channels
    /// (`rotary_dim / 2` frequency pairs).
    pub fn yarn(&self, rotary_dim: usize) -> RopeYarn {
        let RopeScaling::Yarn(y) = self.scaling else {
            return RopeYarn::OFF;
        };
        // HF find_correction_range (truncate = true).
        let dim = rotary_dim as f64;
        let correction_dim = |rotations: f64| {
            dim * (y.original_max_position_embeddings as f64
                / (rotations * 2.0 * std::f64::consts::PI))
                .ln()
                / (2.0 * self.theta.ln())
        };
        let low = correction_dim(y.beta_fast).floor().max(0.0);
        let high = correction_dim(y.beta_slow).ceil().min(dim - 1.0);
        RopeYarn {
            freq_scale: (1.0 / y.factor) as f32,
            corr_low: low as f32,
            corr_high: high as f32,
            mscale: y.attention_factor as f32,
        }
    }

    /// Host reference of the scaled inverse frequencies (HF
    /// `_compute_yarn_parameters` / default RoPE) and the cos/sin multiplier.
    pub fn inv_freq(&self, rotary_dim: usize) -> (Vec<f64>, f64) {
        let args = self.yarn(rotary_dim);
        let freqs = (0..rotary_dim / 2)
            .map(|pair| {
                let extrapolation = self.theta.powf(-((2 * pair) as f64) / rotary_dim as f64);
                let RopeScaling::Yarn(y) = self.scaling else {
                    return extrapolation;
                };
                let (low, high) = (args.corr_low as f64, args.corr_high as f64);
                let span = if high == low { 0.001 } else { high - low };
                let ramp = ((pair as f64 - low) / span).clamp(0.0, 1.0);
                extrapolation / y.factor * ramp + extrapolation * (1.0 - ramp)
            })
            .collect();
        let mscale = match self.scaling {
            RopeScaling::None => 1.0,
            RopeScaling::Yarn(y) => y.attention_factor,
        };
        (freqs, mscale)
    }
}

/// Llama-3 `rope_type: "llama3"` frequency remap (HF
/// `_compute_llama3_parameters`) as kernel scalars for `rope_f32` /
/// `rope_batched_f32`: with `s = clamp(inv_freq * smooth_mul + smooth_add, 0, 1)`,
/// `inv_freq' = inv_freq * (s + (1 - s) * inv_factor)`. `s` is linear in
/// `inv_freq`, so the clamp reproduces HF's three wavelength bands exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeLlama3 {
    pub inv_factor: f32,
    pub smooth_mul: f32,
    pub smooth_add: f32,
}

impl RopeLlama3 {
    /// Plain RoPE: the kernels leave `inv_freq` untouched.
    pub const OFF: Self = Self {
        inv_factor: 1.0,
        smooth_mul: 0.0,
        smooth_add: 1.0,
    };

    /// Requires `factor >= 1` and `high_freq_factor > low_freq_factor`.
    pub fn new(
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_position_embeddings: usize,
    ) -> Self {
        assert!(factor.is_finite() && factor >= 1.0 && high_freq_factor > low_freq_factor);
        let span = high_freq_factor - low_freq_factor;
        Self {
            inv_factor: (1.0 / factor) as f32,
            smooth_mul: (original_max_position_embeddings as f64
                / (2.0 * std::f64::consts::PI * span)) as f32,
            smooth_add: (-low_freq_factor / span) as f32,
        }
    }

    /// Host twin of the kernel remap.
    pub fn scale(&self, inv_freq: f32) -> f32 {
        if self.inv_factor == 1.0 {
            return inv_freq;
        }
        let s = (inv_freq * self.smooth_mul + self.smooth_add).clamp(0.0, 1.0);
        inv_freq * (s + (1.0 - s) * self.inv_factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen38_yarn() -> Rope {
        Rope {
            theta: 10_000_000.0,
            scaling: RopeScaling::Yarn(Yarn::new(4.0, 262_144, None, None, None, None)),
        }
    }

    #[test]
    fn unscaled_rope_lowers_to_the_identity_argument() {
        let rope = Rope {
            theta: 10_000_000.0,
            scaling: RopeScaling::None,
        };
        assert_eq!(rope.yarn(64), RopeYarn::OFF);
        assert_eq!(RopeScaling::None.context_limit(262_144), 262_144);
    }

    #[test]
    fn qwen38_flash_next_yarn_matches_hf_reference_values() {
        // transformers `_compute_yarn_parameters` for the Qwen3.8 README
        // override (theta 1e7, head_dim 256 x partial 0.25, factor 4, original
        // 262144): correction dims 14.24 / 21.12 truncate to (14, 22);
        // attention factor 0.1 ln 4 + 1.
        let rope = qwen38_yarn();
        let args = rope.yarn(64);
        assert_eq!((args.corr_low, args.corr_high), (14.0, 22.0));
        assert_eq!(args.freq_scale, 0.25);
        assert!((args.mscale as f64 - (0.1 * 4f64.ln() + 1.0)).abs() < 1e-7);
        assert_eq!(rope.scaling.context_limit(262_144), 1_048_576);
        let (freqs, _) = rope.inv_freq(64);
        let plain = |i: usize| 1e7f64.powf(-((2 * i) as f64) / 64.0);
        // High-frequency pairs extrapolate, low-frequency pairs interpolate.
        assert_eq!(freqs[0], plain(0));
        assert_eq!(freqs[14], plain(14));
        assert!((freqs[22] - plain(22) / 4.0).abs() < 1e-18);
        assert!((freqs[31] - plain(31) / 4.0).abs() < 1e-18);
        assert!(freqs[18] < plain(18) && freqs[18] > plain(18) / 4.0);
    }

    #[test]
    fn explicit_mscale_pair_sets_the_attention_factor() {
        // DeepSeek-V3 style: mscale == mscale_all_dim cancels to 1.
        let y = Yarn::new(40.0, 4096, None, None, None, Some((1.0, 1.0)));
        assert_eq!(y.attention_factor, 1.0);
        let y = Yarn::new(4.0, 4096, None, None, Some(1.5), None);
        assert_eq!(y.attention_factor, 1.5);
    }

    #[test]
    fn llama3_remap_matches_hf_bands() {
        // Llama-3.2-1B: theta 5e5, head_dim 64, factor 32, low 1, high 4, original 8192.
        let l3 = RopeLlama3::new(32.0, 1.0, 4.0, 8192);
        for i in 0..32 {
            let inv = 500_000f64.powf(-((2 * i) as f64) / 64.0);
            let wavelen = 2.0 * std::f64::consts::PI / inv;
            let hf = if wavelen < 8192.0 / 4.0 {
                inv
            } else if wavelen > 8192.0 {
                inv / 32.0
            } else {
                let s = (8192.0 / wavelen - 1.0) / 3.0;
                (1.0 - s) * inv / 32.0 + s * inv
            };
            let got = l3.scale(inv as f32) as f64;
            assert!((got - hf).abs() <= hf * 1e-5, "pair {i}: {got} vs {hf}");
        }
        assert_eq!(RopeLlama3::OFF.scale(0.25), 0.25);
    }
}
