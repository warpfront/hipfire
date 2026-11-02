// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! FLUX Flow-Match Euler scheduler, pinned to diffusers
//! `FlowMatchEulerDiscreteScheduler` 0.40 semantics (the golden source):
//!
//! - the pipeline feeds `sigmas = linspace(1.0, 1/steps, steps)`; the
//!   scheduler applies the shift transform `σ' = shift·σ/(1+(shift−1)σ)`
//!   (identity at the FLUX default `shift=1`), derives
//!   `timesteps = σ·num_train_timesteps`, and appends terminal `σ=0`.
//! - one Euler step: `x_{i+1} = x_i + (σ_{i+1} − σ_i)·ε_θ`.
//! - `calculate_shift` (resolution-dependent `mu`) is only *applied* when
//!   `use_dynamic_shifting` is set; the tenant config keeps it off.
//!
//! Latent packing replicates `FluxPipeline._pack_latents/_unpack_latents`:
//! `[B,C,H,W]` ↔ `[B,(H/2)(W/2),C·4]` packed patches.

use crate::vae::LatentNorm;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShiftRule {
    Fixed(f32),
    Empirical,
}

/// Empirical mu schedule: resolution-dependent shift coefficient for exponential sigma transform.
/// Based on Klein FLUX.2 formula (spec section 2.4).
pub fn empirical_mu(image_seq_len: usize, num_steps: usize) -> f32 {
    let (a1, b1) = (8.73809524e-05f32, 1.89833333f32);
    let (a2, b2) = (0.00016927f32, 0.45666666f32);
    let len = image_seq_len as f32;
    if image_seq_len > 4300 {
        return a2 * len + b2;
    }
    let m_200 = a2 * len + b2;
    let m_10 = a1 * len + b1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * num_steps as f32 + b
}

fn linspace_sigmas(steps: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![1.0];
    }
    (0..steps)
        .map(|i| 1.0 + (1.0 / steps as f32 - 1.0) * i as f32 / (steps - 1) as f32)
        .collect()
}

/// Sigma pairs with configurable shift rule (fixed or empirical).
pub fn sigma_pairs_ruled(steps: usize, rule: ShiftRule, image_seq_len: usize) -> Vec<(f32, f32)> {
    let mut sigmas = linspace_sigmas(steps);
    match rule {
        ShiftRule::Fixed(shift) => {
            for s in sigmas.iter_mut() {
                *s = shift * *s / (1.0 + (shift - 1.0) * *s);
            }
        }
        ShiftRule::Empirical => {
            let e = empirical_mu(image_seq_len, steps).exp();
            for s in sigmas.iter_mut() {
                *s = e / (e + 1.0 / *s - 1.0);
            }
        }
    }
    sigmas.push(0.0);
    (0..steps).map(|i| (sigmas[i], sigmas[i + 1])).collect()
}

/// Fractional sigma schedule with per-step pairs `[(σ_i, σ_{i+1})]`.
pub fn sigma_pairs(steps: usize, shift: f32) -> Vec<(f32, f32)> {
    sigma_pairs_ruled(steps, ShiftRule::Fixed(shift), 0)
}

/// The timestep (×1000 fraction) the transformer sees per step — equals σ·1000.
pub fn timestep_for_sigma(sigma: f32, num_train_timesteps: f32) -> f32 {
    sigma * num_train_timesteps
}

/// Resolution-dependent shift coefficient (diffusers `calculate_shift`).
pub fn calculate_shift(
    image_seq_len: usize,
    base_image_seq_len: usize,
    max_image_seq_len: usize,
    base_shift: f32,
    max_shift: f32,
) -> f32 {
    let m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len) as f32;
    let b = base_shift - m * base_image_seq_len as f32;
    image_seq_len as f32 * m + b
}

/// One flow-Euler step: `x' = x + (σ_next − σ)·ε`.
pub fn euler_step(x: &[f32], eps: &[f32], sigma: f32, sigma_next: f32) -> Vec<f32> {
    let dt = sigma_next - sigma;
    x.iter().zip(eps).map(|(x, e)| x + dt * e).collect()
}

/// Pack `[C][H][W]` latents into `[(H/2)(W/2)][C·4]` patches
/// (diffusers `_pack_latents` for batch 1):
/// `view(1, C, H/2, 2, W/2, 2) → permute(0, 2, 4, 1, 3, 5) → reshape`.
pub fn pack_latents(x: &[f32], c: usize, h: usize, w: usize) -> (Vec<f32>, usize) {
    let hh = h / 2;
    let ww = w / 2;
    let mut out = vec![0f32; hh * ww * c * 4];
    for ph in 0..hh {
        for pw in 0..ww {
            for ch in 0..c {
                for oh in 0..2 {
                    for ow in 0..2 {
                        let src = x[ch * h * w + (2 * ph + oh) * w + 2 * pw + ow];
                        let dst = (ph * ww + pw) * (c * 4) + ch * 4 + oh * 2 + ow;
                        out[dst] = src;
                    }
                }
            }
        }
    }
    (out, hh * ww)
}

/// Unpack `[tokens][C·4]` back to `[C][H][W]` (diffusers `_unpack_latents`).
pub fn unpack_latents(x: &[f32], tokens: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
    let hh = h / 2;
    let ww = w / 2;
    debug_assert_eq!(tokens, hh * ww);
    let mut out = vec![0f32; c * h * w];
    for ph in 0..hh {
        for pw in 0..ww {
            for ch in 0..c {
                for oh in 0..2 {
                    for ow in 0..2 {
                        let src = x[(ph * ww + pw) * (c * 4) + ch * 4 + oh * 2 + ow];
                        out[ch * h * w + (2 * ph + oh) * w + 2 * pw + ow] = src;
                    }
                }
            }
        }
    }
    out
}

/// VAE input scaling applied pre-decode: `x/scaling_factor + shift_factor`.
pub fn scale_latents(x: &[f32], scaling_factor: f32, shift_factor: f32) -> Vec<f32> {
    x.iter()
        .map(|v| v / scaling_factor + shift_factor)
        .collect()
}

/// Latent denormalization applied pre-decode, on PACKED latents (`[tokens]
/// [width]`, `width` = the transformer's packed-latent column count). For
/// [`LatentNorm::ScaleShift`] this is [`scale_latents`] (the per-element rule
/// commutes with `unpack_latents`, so applying it here vs. after unpack is
/// equivalent — the FLUX.1 path is byte-identical either way). For
/// [`LatentNorm::BatchNorm`] each packed column `c = i % width` carries its
/// own running statistics: `x[t*width + c] * std[c] + mean[c]`.
pub fn denormalize_packed(x: &[f32], tokens: usize, width: usize, norm: &LatentNorm) -> Vec<f32> {
    match norm {
        LatentNorm::ScaleShift { scaling, shift } => scale_latents(x, *scaling, *shift),
        LatentNorm::BatchNorm { mean, std } => {
            assert_eq!(
                mean.len(),
                width,
                "denormalize_packed: latent BatchNorm mean has {} channels but the packed width is {width}",
                mean.len()
            );
            assert_eq!(
                std.len(),
                width,
                "denormalize_packed: latent BatchNorm std has {} channels but the packed width is {width}",
                std.len()
            );
            assert_eq!(
                x.len(),
                tokens * width,
                "denormalize_packed: input has {} elements but tokens*width is {}",
                x.len(),
                tokens * width
            );
            x.iter()
                .enumerate()
                .map(|(i, v)| v * std[i % width] + mean[i % width])
                .collect()
        }
    }
}

/// Inverse of [`denormalize_packed`] — the encode-side latent normalization.
pub fn normalize_packed(x: &[f32], tokens: usize, width: usize, norm: &LatentNorm) -> Vec<f32> {
    match norm {
        LatentNorm::ScaleShift { scaling, shift } => {
            x.iter().map(|v| (v - shift) * scaling).collect()
        }
        LatentNorm::BatchNorm { mean, std } => {
            assert_eq!(
                mean.len(),
                width,
                "normalize_packed: latent BatchNorm mean has {} channels but the packed width is {width}",
                mean.len()
            );
            assert_eq!(
                std.len(),
                width,
                "normalize_packed: latent BatchNorm std has {} channels but the packed width is {width}",
                std.len()
            );
            assert_eq!(
                x.len(),
                tokens * width,
                "normalize_packed: input has {} elements but tokens*width is {}",
                x.len(),
                tokens * width
            );
            x.iter()
                .enumerate()
                .map(|(i, v)| (v - mean[i % width]) / std[i % width])
                .collect()
        }
    }
}

/// xorshift64* PRNG (same generator family as
/// `hipfire_arch_deepseek4::sampling::Xorshift`) — the latent-noise source.
/// Zero-dep, seed-reproducible; seed 0 splashes to the golden-ratio constant
/// so every seed yields a distinct stream.
struct NoiseRng(u64);

impl NoiseRng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in `(0, 1)` — never exactly 0 or 1 (24-bit mantissa fill).
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 + 0.5) / ((1u64 << 24) as f32)
    }
}

/// Deterministic standard-normal latent noise for txt2img init
/// (Box–Muller transform of the xorshift64* stream).
///
/// This is hipfire's own generator, NOT torch-compatible: diffusers pipes
/// seed latents from `torch.randn`, so byte-parity with a diffusers run is
/// only meaningful when the caller supplies the golden init latents (the
/// parity harness does). What this guarantees is the product determinism
/// contract: same seed → same noise → byte-identical PNG, any process.
pub fn seeded_gaussian(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = NoiseRng::new(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = rng.next_f32();
        let u2 = rng.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        out.push(r * theta.cos());
        if out.len() < n {
            out.push(r * theta.sin());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_2_steps_matches_diffusers() {
        // np.linspace(1.0, 0.5, 2) == [1.0, 0.5]; shift 1 → identity.
        assert_eq!(sigma_pairs(2, 1.0), vec![(1.0, 0.5), (0.5, 0.0)]);
    }

    #[test]
    fn euler_step_matches_diffusers_prev_sample() {
        let x0 = vec![1.0f32, 2.0];
        let eps = vec![2.0f32, -1.0];
        let x1 = euler_step(&x0, &eps, 1.0, 0.5);
        assert_eq!(x1, vec![0.0, 2.5]);
    }

    #[test]
    fn pack_unpack_round_trips() {
        let c = 1;
        let h = 4;
        let w = 2;
        let x: Vec<f32> = (0..c * h * w).map(|i| i as f32).collect();
        let (packed, tokens) = pack_latents(&x, c, h, w);
        assert_eq!(tokens, 2);
        assert_eq!(packed.len(), 2 * c * 4);
        let back = unpack_latents(&packed, tokens, c, h, w);
        assert_eq!(back, x);
    }

    #[test]
    fn calculate_shift_at_base_seq_len_is_base_shift() {
        assert!((calculate_shift(256, 256, 4096, 0.5, 1.15) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn seeded_gaussian_is_seed_deterministic() {
        let a = seeded_gaussian(1024, 42);
        let b = seeded_gaussian(1024, 42);
        assert_eq!(a, b, "same seed must produce byte-identical noise");
        let c = seeded_gaussian(1024, 43);
        assert_ne!(a, c, "different seeds must differ");
    }

    #[test]
    fn seeded_gaussian_is_standard_normal_ish() {
        // Loose sanity: mean ≈ 0, stddev ≈ 1 for a large draw. The exact
        // values are pinned by determinism, this only guards against a
        // broken transform (e.g. forgetting the sqrt or the 2π).
        let n = 20_000;
        let x = seeded_gaussian(n, 7);
        let mean = x.iter().sum::<f32>() / n as f32;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.1, "variance {var}");
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn empirical_mu_matches_the_klein_pipeline_formula() {
        // Values computed by hand from the published formula (spec section 2.4).
        let close = |a: f32, b: f32| (a - b).abs() < 1e-5;
        // image_seq_len > 4300: mu = a2*len + b2
        assert!(close(
            empirical_mu(6400, 4),
            0.00016927 * 6400.0 + 0.45666666
        ));
        // 4096 tokens, 4 steps
        let m200 = 0.00016927f32 * 4096.0 + 0.45666666;
        let m10 = 8.73809524e-05f32 * 4096.0 + 1.89833333;
        let a = (m200 - m10) / 190.0;
        let b = m200 - 200.0 * a;
        assert!(close(empirical_mu(4096, 4), a * 4.0 + b));
        assert!(close(empirical_mu(4096, 20), a * 20.0 + b));
        let m200 = 0.00016927f32 * 1024.0 + 0.45666666;
        let m10 = 8.73809524e-05f32 * 1024.0 + 1.89833333;
        let a = (m200 - m10) / 190.0;
        let b = m200 - 200.0 * a;
        assert!(close(empirical_mu(1024, 4), a * 4.0 + b));
    }

    #[test]
    fn exponential_shift_maps_sigma_one_to_one_and_keeps_order() {
        let pairs = sigma_pairs_ruled(4, ShiftRule::Empirical, 4096);
        assert_eq!(pairs.len(), 4);
        assert!((pairs[0].0 - 1.0).abs() < 1e-6, "sigma_0 stays 1");
        assert_eq!(pairs[3].1, 0.0, "terminal sigma is 0");
        for w in pairs.windows(2) {
            assert!(w[0].0 > w[1].0);
        }
        // sigma' = e^mu / (e^mu + 1/sigma - 1) for sigma = 1/4 at the last step
        let mu = empirical_mu(4096, 4);
        let expect = mu.exp() / (mu.exp() + 4.0 - 1.0);
        assert!((pairs[3].0 - expect).abs() < 1e-5);
    }

    #[test]
    fn fixed_rule_equals_the_legacy_sigma_pairs() {
        assert_eq!(
            sigma_pairs_ruled(8, ShiftRule::Fixed(1.0), 0),
            sigma_pairs(8, 1.0)
        );
        assert_eq!(
            sigma_pairs_ruled(8, ShiftRule::Fixed(3.0), 0),
            sigma_pairs(8, 3.0)
        );
    }

    #[test]
    fn batchnorm_normalize_denormalize_round_trip_on_packed_latents() {
        let width = 8;
        let tokens = 3;
        let norm = LatentNorm::BatchNorm {
            mean: (0..width).map(|c| c as f32 * 0.1).collect(),
            std: (0..width).map(|c| 1.0 + c as f32 * 0.05).collect(),
        };
        let x: Vec<f32> = (0..tokens * width).map(|i| (i as f32).sin()).collect();
        let n = normalize_packed(&x, tokens, width, &norm);
        let d = denormalize_packed(&n, tokens, width, &norm);
        for (a, b) in x.iter().zip(&d) {
            assert!((a - b).abs() < 1e-5);
        }
        assert!((n[width + 2] - (x[width + 2] - 0.2) / 1.1).abs() < 1e-6);
    }

    #[test]
    fn scaleshift_denormalize_equals_scale_latents() {
        let x = vec![0.5, -1.0, 2.0];
        let norm = LatentNorm::ScaleShift {
            scaling: 0.3611,
            shift: 0.1159,
        };
        assert_eq!(
            denormalize_packed(&x, 1, 3, &norm),
            scale_latents(&x, 0.3611, 0.1159)
        );
    }
}
