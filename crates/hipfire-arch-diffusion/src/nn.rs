// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared f32 CPU math helpers for the diffusion reference components
//! (T5 / CLIP encoders, VAE decoder, scheduler glue).
//!
//! Everything here is naive row-major f32 with deterministic summation
//! order. It is NOT a performance path — it exists so the pipeline can be
//! numerically validated against diffusers/transformers goldens before any
//! GPU kernel work. Small matrices mean the naive loops
//! are also the *reference* the GPU kernels will later be checked against.
//!
//! [`linear`] is row-parallel, which keeps the summation order per output
//! element exactly as written and so stays bit-identical to the serial form.
//! It is parallel only because the matrices stopped being small: a real
//! FLUX prompt puts ~4.7 TFLOP of T5-XXL through this function.

use crate::flux::Tensor;
use rayon::prelude::*;

/// `out = a × b` where `a` is `[n, k]` and `b` is `[k, m]`, row-major.
/// Deterministic: rows/cols summed in order.
pub fn matmul(a: &[f32], n: usize, k: usize, b: &[f32], m: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * m];
    for r in 0..n {
        for c in 0..m {
            let mut acc = 0f32;
            for t in 0..k {
                acc += a[r * k + t] * b[t * m + c];
            }
            out[r * m + c] = acc;
        }
    }
    out
}

/// Affine `Linear` forward: `y = x Wᵀ + b`, `x` is `[rows, in]`,
/// `weight` is `[out, in]`.
pub fn linear(
    x: &[f32],
    rows: usize,
    in_dim: usize,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Vec<f32> {
    debug_assert_eq!(weight.rows, weight.data.len() / weight.cols);
    debug_assert_eq!(weight.cols, in_dim);
    let out_dim = weight.rows;
    let mut y = vec![0f32; rows * out_dim];
    // Row-parallel. Every output row is independent, so this is a pure
    // scheduling change with bit-identical results — each accumulator still
    // sums the same terms in the same order. It matters because this function
    // carries the whole T5-XXL / CLIP conditioning: ~4.7 TFLOP for one FLUX
    // prompt, which as a single-threaded scalar loop is tens of minutes and
    // dwarfs the GPU denoise it feeds.
    let wdata = &weight.data;
    let bdata = bias.map(|b| b.data.as_slice());
    y.par_chunks_mut(out_dim)
        .enumerate()
        .for_each(|(r, out_row)| {
            let x_row = &x[r * in_dim..(r + 1) * in_dim];
            for (o, slot) in out_row.iter_mut().enumerate() {
                let w_row = &wdata[o * in_dim..(o + 1) * in_dim];
                let mut acc = bdata.map(|b| b[o]).unwrap_or(0.0);
                for i in 0..in_dim {
                    acc += x_row[i] * w_row[i];
                }
                *slot = acc;
            }
        });
    y
}

/// Affine LayerNorm (mean/var over the last axis).
pub fn layernorm_affine(
    x: &[f32],
    rows: usize,
    dim: usize,
    w: &[f32],
    b: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean: f32 = row.iter().sum::<f32>() / dim as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..dim {
            y[r * dim + c] = (row[c] - mean) * inv * w[c] + b[c];
        }
    }
    y
}

/// Weightless LayerNorm (FLUX/T5 block norms).
pub fn layernorm_plain(x: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    layernorm_affine(x, rows, dim, &vec![1.0; dim], &vec![0.0; dim], eps)
}

/// RMSNorm with per-channel scale (no mean subtraction, T5-style).
pub fn rmsnorm_scale(x: &[f32], rows: usize, dim: usize, scale: &[f32], eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for c in 0..dim {
            y[r * dim + c] = row[c] * inv * scale[c];
        }
    }
    y
}

/// GroupNorm over a `[channels][h][w]` tensor with `groups` groups.
pub fn groupnorm(
    x: &[f32],
    c: usize,
    h: usize,
    w: usize,
    groups: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    let ch_per_group = c / groups;
    let group_elems = ch_per_group * h * w;
    for g in 0..groups {
        let start = g * group_elems;
        let end = start + group_elems;
        let mean: f32 = x[start..end].iter().sum::<f32>() / group_elems as f32;
        let var: f32 = x[start..end]
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f32>()
            / group_elems as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in start..end {
            let ch = i / (h * w); // channel index for gamma/beta
            y[i] = (x[i] - mean) * inv * gamma[ch] + beta[ch];
        }
    }
    y
}

/// 2-D convolution, padding=`pad`, stride 1, no dilation, no groups.
/// Input `[c_in][h][w]`, weight `[c_out][c_in][kh][kw]`.
pub fn conv2d(
    x: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    c_out: usize,
    kh: usize,
    kw: usize,
    bias: Option<&[f32]>,
    pad: usize,
) -> (Vec<f32>, usize, usize) {
    let out_h = (h as isize + 2 * pad as isize - kh as isize + 1).max(0) as usize;
    let out_w = (w as isize + 2 * pad as isize - kw as isize + 1).max(0) as usize;
    let mut y = vec![0f32; c_out * out_h * out_w];
    for co in 0..c_out {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut acc = bias.map(|b| b[co]).unwrap_or(0.0);
                for ci in 0..c_in {
                    for khh in 0..kh {
                        for kww in 0..kw {
                            let ih = oh as isize + khh as isize - pad as isize;
                            let iw = ow as isize + kww as isize - pad as isize;
                            if ih >= 0 && iw >= 0 && (ih as usize) < h && (iw as usize) < w {
                                acc += x[ci * h * w + ih as usize * w + iw as usize]
                                    * weight[co * c_in * kh * kw + ci * kh * kw + khh * kw + kww];
                            }
                        }
                    }
                }
                y[co * out_h * out_w + oh * out_w + ow] = acc;
            }
        }
    }
    (y, out_h, out_w)
}

/// 2-D convolution with independent stride and asymmetric padding.
/// Input `[c_in][h][w]`, weight `[c_out][c_in][kh][kw]`. `conv2d` is the
/// stride-1/symmetric-pad special case of this (kept separate since it is
/// the hot loop for every VAE resnet conv).
pub fn conv2d_strided(
    x: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    c_out: usize,
    kh: usize,
    kw: usize,
    bias: Option<&[f32]>,
    stride: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
) -> (Vec<f32>, usize, usize) {
    let out_h = (h + pad_top + pad_bottom - kh) / stride + 1;
    let out_w = (w + pad_left + pad_right - kw) / stride + 1;
    let mut y = vec![0.0f32; c_out * out_h * out_w];
    y.par_chunks_mut(out_h * out_w)
        .enumerate()
        .for_each(|(co, plane)| {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut acc = bias.map_or(0.0, |b| b[co]);
                    for ci in 0..c_in {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = (oy * stride + ky) as isize - pad_top as isize;
                                let ix = (ox * stride + kx) as isize - pad_left as isize;
                                if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                                    continue;
                                }
                                acc += weight[((co * c_in + ci) * kh + ky) * kw + kx]
                                    * x[(ci * h + iy as usize) * w + ix as usize];
                            }
                        }
                    }
                    plane[oy * out_w + ox] = acc;
                }
            }
        });
    (y, out_h, out_w)
}

/// Nearest-neighbour 2× upsampling (`[c][h][w]` → `[c][2h][2w]`).
pub fn upsample_nearest2x(x: &[f32], c: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let (oh, ow) = (2 * h, 2 * w);
    let mut y = vec![0f32; c * oh * ow];
    for ch in 0..c {
        for ih in 0..h {
            for iw in 0..w {
                let v = x[ch * h * w + ih * w + iw];
                y[ch * oh * ow + (2 * ih) * ow + 2 * iw] = v;
                y[ch * oh * ow + (2 * ih) * ow + 2 * iw + 1] = v;
                y[ch * oh * ow + (2 * ih + 1) * ow + 2 * iw] = v;
                y[ch * oh * ow + (2 * ih + 1) * ow + 2 * iw + 1] = v;
            }
        }
    }
    (y, oh, ow)
}

/// Row-wise softmax over `[rows][cols]`.
pub fn softmax_rows(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        let mut exps = vec![0f32; cols];
        for c in 0..cols {
            let e = (row[c] - max).exp();
            exps[c] = e;
            sum += e;
        }
        for c in 0..cols {
            y[r * cols + c] = exps[c] / sum;
        }
    }
    y
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU with tanh approximation (`nn.GELU(approximate="tanh")`).
///
/// The inner constant is `√(2/π) ≈ 0.7978845608` — NOT `√2·√(2/π)`. A stray
/// `√2` over-activates every FFN lane (≈9% at x=1, ≈46% at x=-1) and silently
/// diverges the T5 conditioning from every reference (torch `F.gelu`
/// approximate="tanh", transformers T5-v1.1 gated GELU, ComfyUI
/// `gelu_pytorch_tanh`); the step-1 T5 bisect against ComfyUI found it.
pub fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_560_8f32 * (x + 0.044_715 * x * x * x)).tanh())
}

/// Exact GELU via erf (`nn.functional.gelu`, CLIP's `hidden_act: "gelu"`).
pub fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x / std::f32::consts::SQRT_2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv2d_small_pads_correctly() {
        // 1×2×2 input, 1×1×3×3 weight = identity-ish edge probe: value lands
        // only where the window fully covers (pad 0 keeps corners).
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let (y, oh, ow) = conv2d(&x, 1, 2, 2, &w, 1, 3, 3, None, 0);
        assert_eq!((oh, ow), (0, 0)); // 2+0-3+1 = 0 → no output positions
        assert!(y.is_empty());
        let (y, oh, ow) = conv2d(&x, 1, 2, 2, &w, 1, 3, 3, None, 1);
        assert_eq!((oh, ow), (2, 2));
        // center-only kernel + pad 1 = identity conv
        assert_eq!(y, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn groupnorm_single_group_is_full_normalization() {
        let x = vec![1.0, 3.0, 5.0, 7.0];
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let beta = vec![0.0; 4];
        let y = groupnorm(&x, 4, 1, 1, 1, &gamma, &beta, 1e-5);
        let mean = 4.0f32;
        let var = (5.0f32) / 1.0; // (9+1+1+9)/4
        let inv = 1.0 / (var + 1e-5).sqrt();
        assert!((y[0] - (1.0 - mean) * inv).abs() < 1e-4);
        assert!((y[3] - (7.0 - mean) * inv).abs() < 1e-4);
    }

    #[test]
    fn conv2d_strided_matches_conv2d_at_stride_one() {
        let (c_in, h, w, c_out) = (2, 4, 5, 3);
        let x: Vec<f32> = (0..c_in * h * w).map(|i| (i as f32 * 0.3).sin()).collect();
        let wt: Vec<f32> = (0..c_out * c_in * 9)
            .map(|i| (i as f32 * 0.7).cos())
            .collect();
        let a = conv2d(&x, c_in, h, w, &wt, c_out, 3, 3, None, 1);
        let b = conv2d_strided(&x, c_in, h, w, &wt, c_out, 3, 3, None, 1, 1, 1, 1, 1);
        assert_eq!(a, b);
    }

    #[test]
    fn conv2d_strided_downsample_uses_asymmetric_pad_0_1_0_1() {
        // diffusers Downsample2D: pad (0,1,0,1) then 3x3 stride 2, pad 0 → out = floor((h+1-3)/2)+1 = h/2
        let (c_in, h, w, c_out) = (1, 4, 6, 1);
        let x: Vec<f32> = (0..h * w).map(|i| i as f32).collect();
        let wt = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // centre tap only
        let (y, oh, ow) = conv2d_strided(&x, c_in, h, w, &wt, c_out, 3, 3, None, 2, 0, 0, 1, 1);
        assert_eq!((oh, ow), (2, 3));
        // centre tap at output (oy, ox) reads input (2*oy+1, 2*ox+1)
        assert_eq!(y[0], x[1 * w + 1]);
        assert_eq!(y[1 * ow + 2], x[3 * w + 5]);
    }

    #[test]
    fn gelu_tanh_matches_torch_reference_values() {
        // torch `F.gelu(approximate="tanh")` spot values. A stray √2 inside
        // the tanh argument (√(2/π)·√2 instead of √(2/π)) makes these come
        // out 0.9135 / -0.0865 — this test pins the correct 0.8412 / -0.1588.
        let got1 = gelu_tanh(1.0);
        let gotm1 = gelu_tanh(-1.0);
        assert!(
            (got1 - 0.841_192).abs() < 1e-4,
            "gelu_tanh(1) = {got1}, want 0.841192"
        );
        assert!(
            (gotm1 - (-0.158_808)).abs() < 1e-4,
            "gelu_tanh(-1) = {gotm1}, want -0.158808"
        );
    }
}
