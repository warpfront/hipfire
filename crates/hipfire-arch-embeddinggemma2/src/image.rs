// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The Gemma-4 image processor, ported (`Gemma4ImageProcessorPil`,
//! transformers 5.19.0 `models/gemma4/image_processing_pil_gemma4.py`).
//!
//! Per image, in this order:
//!
//! 1. **Aspect-preserving resize** to the largest size that (a) stays within
//!    the patch budget `max_soft_tokens * pooling_kernel_size²` and (b) has
//!    both sides divisible by `pooling_kernel_size * patch_size` (= 48). The
//!    resize is the processor's own resampler — torchvision's
//!    `tvF.resize(..., BICUBIC, antialias=True)` on a uint8 tensor,
//!    transcribed bit-for-bit in [`aten_bicubic_resize`] — and is skipped when
//!    the target equals the source.
//! 2. **Rescale × 1/255** into `[0, 1]`. There is **no normalization** — this
//!    tower was trained on raw `[0,1]` pixels (`do_normalize = false`).
//! 3. **Patchify**: `patch_size × patch_size` non-overlapping tiles, rows
//!    row-major over `(grid_h, grid_w)`, each patch flattened
//!    **h → w → channel** (`((dy * patch_size) + dx) * 3 + c`) — the layout
//!    `siglip2.convert_image_to_patches` produces and the patch-embedder GEMM
//!    expects.
//! 4. **Positions**: `(x = column, y = row)` per patch, same row-major order.
//! 5. **Token count**: `N = (grid_h / k) * (grid_w / k)`, `k = 3`.
//!
//! Differences from HF, both deliberate and load-bearing:
//!
//! - HF pads the patch list to `max_patches` (2 520) with zero patches and
//!   positions `-1`, and the tower then masks/zeroes them. We keep only the
//!   REAL patches: rows beyond the real grid cannot influence a real row
//!   (their keys are masked in attention, their pooled contribution is zero),
//!   so the surviving rows are identical and the sequence is 9× shorter for a
//!   small image. `patch_embedder`'s pad-zeroing has nothing to zero here.
//! - HF reports `num_soft_tokens_per_image = num_patches // k²`; we compute
//!   the pooler's real bin count `(grid_h/k) * (grid_w/k)`. The two agree
//!   because the resize makes both grid axes multiples of `k`.

use crate::config::VisionConfig;

/// Maximum source pixel count, checked from the format header BEFORE the
/// pixel buffer is allocated (decompression-bomb guard, the same value and
/// contract as `hipfire-arch-qwen35-vl`): anything a 50000×50000 PNG would
/// expand to is refused, while any real photo passes and is downscaled by
/// the aspect-preserving resize.
pub const MAX_DIMENSION_PIXELS: usize = 16_777_216;

// ─── torchvision's uint8 bicubic resize ───────────────────────────────────
//
// This resize is a *transcription* of the kernel HF's processor actually
// runs, not an approximation of it, and that is load-bearing. The installed
// processor backend is torchvision (`Gemma4ImageProcessor`), whose resize is
// `tvF.resize(image, size, interpolation=BICUBIC, antialias=True)` →
// `torch.nn.functional.interpolate` on a **uint8** tensor. ATen has a
// dedicated separable kernel for that dtype
// (`aten/src/ATen/native/cpu/UpSampleKernel.cpp`:
// `HelperInterpCubic::_compute_index_ranges_int16_weights` +
// `basic_loop_separable_1d_{horizontal,vertical}<uint8_t>`):
//
// - double-precision Catmull-Rom weights (`a = -0.5`, support
//   `2 * max(1, scale)`, centre `(i + 0.5) * scale - 0.5` in sample
//   coordinates — the same filter Pillow uses),
// - normalized to sum 1, then quantized to **int16** at a *dynamic*
//   precision `p` (the first `p` whose scaled maximum weight stops fitting
//   int16, over that axis' largest normalized weight),
// - applied as an integer multiply-accumulate with a `1 << (p - 1)` bias and a
//   `>> p` rounding, clamped to `[0, 255]`,
// - HORIZONTAL (width) pass first, into a uint8 intermediate, then the
//   vertical pass (the same order Pillow uses).
//
// Nothing else agrees with it bit-for-bit: `image`'s
// `imageops::resize(.., CatmullRom)` runs the passes the other way round with
// an f32 intermediate and a single rounding (up to 25/255 off on a noise
// texture), Pillow's own resampler uses 22-bit int32 weights (1 LSB off on 6%
// of a smooth gradient's pixels), and even ATen's *float* path computes
// different values (f32 weights, no int16 quantization). The vision tower
// amplifies sub-LSB pixel differences well past the media parity band, so
// bit-exactness here is what makes the image legs match the float32 reference.

/// ATen's `weights_precision` search bound for uint8 (`< 22`).
const ATEN_MAX_WEIGHT_BITS: i32 = 22;
/// ATen's Catmull-Rom parameter for `antialias=True`
/// (`HelperInterpCubic::aa_filter<double, true>`; `-0.75` is the
/// antialias=False "opencv" variant).
const ATEN_CUBIC_A: f64 = -0.5;

/// ATen's `cubic_convolution1` / `cubic_convolution2` (`UpSample.h`), the
/// exact expressions, with `a = -0.5`.
fn aten_cubic(x: f64) -> f64 {
    let a = ATEN_CUBIC_A;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
    } else {
        0.0
    }
}

/// One axis' int16 weight table
/// (`HelperInterpCubic::_compute_index_ranges_int16_weights`).
struct AtenAxis {
    /// Source extent this table samples from.
    in_size: usize,
    /// Weights per output index (ATen's `max_interp_size`).
    ksize: usize,
    /// First source index feeding each output index.
    xmin: Vec<usize>,
    /// Number of source samples feeding each output index.
    count: Vec<usize>,
    /// `[out_size, ksize]` int16 fixed-point weights.
    k: Vec<i16>,
    /// Fractional bits of `k` (ATen's dynamic `weights_precision`).
    precision: i32,
}

impl AtenAxis {
    fn new(in_size: usize, out_size: usize) -> Self {
        let scale = in_size as f64 / out_size as f64;
        // ATen: `support = (scale >= 1) ? interp_size * 0.5 * scale
        //                              : interp_size * 0.5`, `interp_size = 4`.
        let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
        let ksize = support.ceil() as usize * 2 + 1;
        let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
        let mut axis = Self {
            in_size,
            ksize,
            xmin: vec![0; out_size],
            count: vec![0; out_size],
            k: vec![0; out_size * ksize],
            precision: ATEN_MAX_WEIGHT_BITS,
        };

        // Pass 1: the normalized double weights, and this axis' largest one.
        let mut weights = vec![0.0f64; out_size * ksize];
        let mut wt_max = 0.0f64;
        for i in 0..out_size {
            // ATen's centre is the pixel *edge*; the `+ 0.5` in the kernel
            // argument turns it back into the sample centre.
            let center = scale * (i as f64 + 0.5);
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64);
            let count = (xmax - xmin as i64).max(0).min(ksize as i64) as usize;
            let mut total = 0.0f64;
            for j in 0..count {
                let w = aten_cubic((j as f64 + xmin as f64 - center + 0.5) * invscale);
                weights[i * ksize + j] = w;
                total += w;
            }
            if total != 0.0 {
                for j in 0..count {
                    weights[i * ksize + j] /= total;
                    wt_max = wt_max.max(weights[i * ksize + j]);
                }
            }
            axis.xmin[i] = xmin;
            axis.count[i] = count;
        }

        // ATen's dynamic precision: the first `p` whose scaled maximum weight
        // stops fitting int16, so the table's largest weight always fits. C's
        // loop leaves the variable at 22 when it never breaks.
        for p in 0..ATEN_MAX_WEIGHT_BITS {
            let next_value = (0.5 + wt_max * (1i64 << (p + 1)) as f64) as i32;
            if next_value >= (1 << 15) {
                axis.precision = p;
                break;
            }
        }

        // Pass 2: quantize (`(int)(±0.5 + w * 2^p)`, a half-biased truncation).
        let scaled = (1i64 << axis.precision) as f64;
        for i in 0..out_size {
            for j in 0..ksize {
                let v = weights[i * ksize + j] * scaled;
                axis.k[i * ksize + j] = if v < 0.0 {
                    (-0.5 + v) as i16
                } else {
                    (0.5 + v) as i16
                };
            }
        }
        axis
    }
}

/// ATen's uint8 rounding: the accumulator's high bits, clamped to `[0, 255]`
/// (`std::clamp(output >> weights_precision, 0, 255)`).
#[inline]
fn aten_clip8(v: i32, precision: i32) -> u8 {
    (v >> precision).clamp(0, 255) as u8
}

/// `basic_loop_separable_1d_horizontal<uint8_t>`: one row of `out_w` pixels
/// per input row, accumulated in the same int32 fixed point as ATen.
fn aten_resample_horizontal(src: &[u8], h: usize, out_w: usize, ax: &AtenAxis) -> Vec<u8> {
    let in_w = ax.in_size;
    let mut out = vec![0u8; out_w * h * 3];
    let bias = 1i32 << (ax.precision - 1); // ATen's `+0.5` part
    for y in 0..h {
        let row = &src[y * in_w * 3..(y + 1) * in_w * 3];
        let orow = &mut out[y * out_w * 3..(y + 1) * out_w * 3];
        for xx in 0..out_w {
            let k = &ax.k[xx * ax.ksize..xx * ax.ksize + ax.count[xx]];
            let base = ax.xmin[xx] * 3;
            let mut ss = [bias; 3];
            for (x, kx) in k.iter().enumerate() {
                let p = &row[base + x * 3..base + x * 3 + 3];
                ss[0] += p[0] as i32 * *kx as i32;
                ss[1] += p[1] as i32 * *kx as i32;
                ss[2] += p[2] as i32 * *kx as i32;
            }
            orow[xx * 3] = aten_clip8(ss[0], ax.precision);
            orow[xx * 3 + 1] = aten_clip8(ss[1], ax.precision);
            orow[xx * 3 + 2] = aten_clip8(ss[2], ax.precision);
        }
    }
    out
}

/// `basic_loop_separable_1d_vertical<uint8_t>`: the same reduction down
/// columns, over the uint8 intermediate.
fn aten_resample_vertical(src: &[u8], w: usize, out_h: usize, ax: &AtenAxis) -> Vec<u8> {
    let mut out = vec![0u8; w * out_h * 3];
    let bias = 1i32 << (ax.precision - 1);
    let stride = w * 3;
    for yy in 0..out_h {
        let k = &ax.k[yy * ax.ksize..yy * ax.ksize + ax.count[yy]];
        let base = ax.xmin[yy] * stride;
        let orow = &mut out[yy * stride..(yy + 1) * stride];
        for xx in 0..w {
            let mut ss = [bias; 3];
            for (y, ky) in k.iter().enumerate() {
                let p = &src[base + y * stride + xx * 3..base + y * stride + xx * 3 + 3];
                ss[0] += p[0] as i32 * *ky as i32;
                ss[1] += p[1] as i32 * *ky as i32;
                ss[2] += p[2] as i32 * *ky as i32;
            }
            orow[xx * 3] = aten_clip8(ss[0], ax.precision);
            orow[xx * 3 + 1] = aten_clip8(ss[1], ax.precision);
            orow[xx * 3 + 2] = aten_clip8(ss[2], ax.precision);
        }
    }
    out
}

/// `tvF.resize(image, (out_w, out_h), BICUBIC, antialias=True)` for an 8-bit
/// RGB image: `upsample_separable_Nd_kernel_impl`'s width-then-height order,
/// with the uint8 intermediate ATen's `at::empty(temp_oshape, input.options())`
/// creates.
fn aten_bicubic_resize(rgb: &image::RgbImage, out_w: usize, out_h: usize) -> image::RgbImage {
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    let src = rgb.as_raw();
    let mid_w = if out_w != w { out_w } else { w };
    let mid = if out_w != w {
        aten_resample_horizontal(src, h, out_w, &AtenAxis::new(w, out_w))
    } else {
        src.clone()
    };
    let out = if out_h != h {
        aten_resample_vertical(&mid, mid_w, out_h, &AtenAxis::new(h, out_h))
    } else {
        mid
    };
    image::RgbImage::from_raw(out_w as u32, out_h as u32, out)
        .expect("resize output has the requested dimensions")
}

/// One image's CPU-side processor output: what the tower forward needs.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessedImage {
    /// `[n_patches, patch_pixels]` row-major, pixels in `[0, 1]`,
    /// per-patch layout h → w → channel.
    pub patches: Vec<f32>,
    /// `[n_patches, 2]` row-major `(x, y)` patch coordinates.
    pub positions: Vec<i32>,
    /// Patch grid height (`height / patch_size`).
    pub grid_h: usize,
    /// Patch grid width.
    pub grid_w: usize,
    /// `grid_h * grid_w`.
    pub n_patches: usize,
    /// Soft tokens: `(grid_h / k) * (grid_w / k)` — the placeholder count.
    pub n_tokens: usize,
}

impl ProcessedImage {
    /// Positions for patch `i` as `(x, y)`.
    pub fn position(&self, i: usize) -> (i32, i32) {
        (self.positions[i * 2], self.positions[i * 2 + 1])
    }
}

/// HF `get_aspect_ratio_preserving_size`: the largest `(height, width)`
/// within the patch budget whose sides are multiples of
/// `pooling_kernel_size * patch_size`.
///
/// The single-side edge cases are HF's verbatim, including the asymmetry that
/// the surviving side is recomputed from the ORIGINAL ratio (`floor(w/h)`)
/// and clamped to `(max_patches / k²) * side_mult`, not from `ideal_*`.
/// A genuinely zero-sided input is refused before this is reached.
pub fn aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize), String> {
    if height == 0 || width == 0 {
        return Err(format!(
            "gemma4 image processor: degenerate image {height}x{width} (a side is zero)"
        ));
    }
    let side_mult = pooling_kernel_size * patch_size;
    let target_px = max_patches as f64 * (patch_size * patch_size) as f64;
    let total_px = (height as f64) * (width as f64);
    let factor = (target_px / total_px).sqrt();
    let ideal_h = factor * height as f64;
    let ideal_w = factor * width as f64;

    let mut target_h = ((ideal_h / side_mult as f64).floor() as usize) * side_mult;
    let mut target_w = ((ideal_w / side_mult as f64).floor() as usize) * side_mult;

    if target_h == 0 && target_w == 0 {
        // Unreachable for any real (both sides ≥ 1) image — HF raises here
        // too, so we mirror the refusal rather than invent a size.
        return Err(format!(
            "gemma4 image processor: {height}x{width} rounds to 0x0 at multiples of \
             {side_mult} — nothing to resize to"
        ));
    }

    let max_side_length = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if target_h == 0 {
        target_h = side_mult;
        target_w = (width / height) * side_mult;
        target_w = target_w.min(max_side_length);
    } else if target_w == 0 {
        target_w = side_mult;
        target_h = (height / width) * side_mult;
        target_h = target_h.min(max_side_length);
    }

    if target_h * target_w > max_patches * patch_size * patch_size {
        return Err(format!(
            "gemma4 image processor: resizing {height}x{width} to {target_h}x{target_w} \
             exceeds {max_patches} patches at patch_size {patch_size}"
        ));
    }
    Ok((target_h, target_w))
}

/// `siglip2.convert_image_to_patches`: `patch_size²` tiles, row-major over
/// the patch grid, each flattened h → w → channel, rescaled into `[0, 1]`.
///
/// Returns `(patches, grid_h, grid_w)`.
pub(crate) fn patchify(
    rgb: &image::RgbImage,
    patch_size: usize,
    rescale: f32,
) -> (Vec<f32>, usize, usize) {
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    let grid_h = h / patch_size;
    let grid_w = w / patch_size;
    let mut out = vec![0.0f32; grid_h * grid_w * 3 * patch_size * patch_size];
    let src = rgb.as_raw();
    for py in 0..grid_h {
        for px in 0..grid_w {
            let patch = py * grid_w + px;
            let dst = patch * 3 * patch_size * patch_size;
            for dy in 0..patch_size {
                for dx in 0..patch_size {
                    let y = py * patch_size + dy;
                    let x = px * patch_size + dx;
                    let s = (y * w + x) * 3;
                    let d = dst + ((dy * patch_size) + dx) * 3;
                    out[d] = src[s] as f32 * rescale;
                    out[d + 1] = src[s + 1] as f32 * rescale;
                    out[d + 2] = src[s + 2] as f32 * rescale;
                }
            }
        }
    }
    (out, grid_h, grid_w)
}

/// Run the processor on an already-decoded RGB image.
pub fn process_rgb8(
    mut rgb: image::RgbImage,
    cfg: &VisionConfig,
) -> Result<ProcessedImage, String> {
    let (src_h, src_w) = (rgb.height() as usize, rgb.width() as usize);
    if src_h == 0 || src_w == 0 {
        return Err("gemma4 image processor: decoded image has a zero side".into());
    }
    let (target_h, target_w) = aspect_ratio_preserving_size(
        src_h,
        src_w,
        cfg.patch_size,
        cfg.max_patches(),
        cfg.pooling_kernel_size,
    )?;
    if target_h != src_h || target_w != src_w {
        // torchvision's uint8 bicubic, transcribed — see the block at the top
        // of this file for why neither `image`'s `CatmullRom` nor Pillow's own
        // resampler is the same filter.
        rgb = aten_bicubic_resize(&rgb, target_w, target_h);
    }

    let (patches, grid_h, grid_w) = patchify(&rgb, cfg.patch_size, 1.0 / 255.0);
    let k = cfg.pooling_kernel_size;
    if grid_h % k != 0 || grid_w % k != 0 {
        return Err(format!(
            "gemma4 image processor: patch grid {grid_h}x{grid_w} is not divisible by the \
             pooling kernel {k} (the resize should have guaranteed it)"
        ));
    }
    let n_patches = grid_h * grid_w;
    let n_tokens = n_patches / (k * k);

    let mut positions = Vec::with_capacity(n_patches * 2);
    for py in 0..grid_h {
        for px in 0..grid_w {
            positions.push(px as i32);
            positions.push(py as i32);
        }
    }

    Ok(ProcessedImage {
        patches,
        positions,
        grid_h,
        grid_w,
        n_patches,
        n_tokens,
    })
}

/// Decompression-bomb guard: refuse header dimensions that would expand past
/// [`MAX_DIMENSION_PIXELS`], BEFORE any pixel buffer is allocated. Only a
/// guard — anything under it is legitimately handled by the resize.
fn guard_dimensions(w: usize, h: usize) -> Result<(), String> {
    if w.saturating_mul(h) > MAX_DIMENSION_PIXELS {
        return Err(format!(
            "image dimensions ({w}x{h}) exceed maximum ({MAX_DIMENSION_PIXELS} pixels)"
        ));
    }
    Ok(())
}

/// Decode container bytes (PNG/JPEG) and run the processor.
///
/// Dimensions are read from the format header first so a decompression bomb
/// is refused before any pixel buffer exists.
pub fn process_image(bytes: &[u8], cfg: &VisionConfig) -> Result<ProcessedImage, String> {
    let (w, h) = hipfire_runtime::imagedec::probe_dimensions(bytes)?;
    guard_dimensions(w as usize, h as usize)?;
    let rgb = hipfire_runtime::imagedec::decode_rgb8(bytes)?;
    process_rgb8(rgb, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> VisionConfig {
        // The checkpoint's vision_config, trimmed to what the processor reads.
        VisionConfig {
            hidden_size: 768,
            num_layers: 16,
            intermediate_size: 3072,
            n_heads: 12,
            n_kv_heads: 12,
            head_dim: 64,
            rms_norm_eps: 1e-6,
            patch_size: 16,
            pooling_kernel_size: 3,
            position_embedding_size: 10240,
            rope_theta: 100.0,
            max_soft_tokens: 280,
        }
    }

    fn size(h: usize, w: usize) -> (usize, usize) {
        let c = cfg();
        aspect_ratio_preserving_size(h, w, c.patch_size, c.max_patches(), c.pooling_kernel_size)
            .expect("size")
    }

    fn tokens(h: usize, w: usize) -> usize {
        let (th, tw) = size(h, w);
        ((th / 16) / 3) * ((tw / 16) / 3)
    }

    #[test]
    fn square_1024_is_the_documented_768_by_768_256_tokens() {
        // factor = sqrt(2520*256/1024²) = 0.7844 → ideal 803.3 →
        // floor(803.3/48)*48 = 768; (768/16)²/9 = 256.
        assert_eq!(size(1024, 1024), (768, 768));
        assert_eq!(tokens(1024, 1024), 256);
    }

    #[test]
    fn sides_are_always_multiples_of_48_and_within_the_patch_budget() {
        for (h, w) in [
            (1024, 1024),
            (2000, 500),
            (500, 2000),
            (4000, 100),
            (100, 4000),
            (48, 48),
            (10000, 10),
            (10, 10000),
            (1920, 1080),
            (1080, 1920),
            (3840, 2160),
            (300, 300),
            (1, 4000),
            (4000, 1),
        ] {
            let (th, tw) = size(h, w);
            assert_eq!(th % 48, 0, "{h}x{w} → {th}x{tw}");
            assert_eq!(tw % 48, 0, "{h}x{w} → {th}x{tw}");
            let patches = (th / 16) * (tw / 16);
            assert!(patches <= 2520, "{h}x{w} → {th}x{tw}: {patches} patches");
            assert!(tokens(h, w) <= 280, "{h}x{w}: {} tokens", tokens(h, w));
        }
    }

    #[test]
    fn wide_and_tall_keep_the_aspect_and_the_token_budget() {
        // 2048x512 (4:1): factor 0.7844 → 384x1584 → 24*99/9 = 264 tokens.
        assert_eq!(size(512, 2048), (384, 1584));
        assert_eq!(tokens(512, 2048), 264);
        // Same pixels transposed → the transposed resize.
        assert_eq!(size(2048, 512), (1584, 384));
        assert_eq!(tokens(2048, 512), 264);
    }

    #[test]
    fn a_side_that_rounds_to_zero_uses_the_ratio_fallback() {
        // The fallback branch needs ideal_h < 48, i.e. h/w < 1/280.
        // 2x1000: factor = 17.96 → ideal_h = 35.9 → 0 → th = 48 in the zero
        // branch, and tw is recomputed from the ORIGINAL ratio, clamped to
        // 280*48 = 13440: (1000/2)*48 = 24000 → clamped.
        assert_eq!(size(2, 1000), (48, 13440));
        assert_eq!(tokens(2, 1000), 280); // (3/3) * (840/3)
                                          // 1x10000: 10000*48 = 480000 > 13440 → the same clamp.
        assert_eq!(size(1, 10000), (48, 13440));
        assert_eq!(tokens(1, 10000), 280);
        // 3x3000 = 0.001 < 1/280 → the fallback branch as well.
        assert_eq!(size(3, 3000), (48, 13440));
        // Just above the threshold the generic path keeps the ideal width.
        assert_eq!(size(5, 1000), (48, 11328));
        assert_eq!(tokens(5, 1000), 236); // (3/3) * (708/3)
    }

    #[test]
    fn a_zero_sided_image_is_refused() {
        let c = cfg();
        let err = aspect_ratio_preserving_size(0, 100, c.patch_size, c.max_patches(), 3);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("zero"));
        assert!(aspect_ratio_preserving_size(100, 0, c.patch_size, c.max_patches(), 3).is_err());
    }

    #[test]
    fn patch_layout_is_row_major_h_w_channel() {
        // A 48x48 image → a 3x3 patch grid with no resize (each patch is one
        // 16x16 tile). Paint pixel (x,y) = (r=x*5, g=y*3, b=7) so every
        // element of the patch buffer is a known function of its position.
        let mut img = image::RgbImage::new(48, 48);
        for y in 0..48u32 {
            for x in 0..48u32 {
                img.put_pixel(x, y, image::Rgb([(x * 5) as u8, (y * 3) as u8, 7]));
            }
        }
        let (patches, gh, gw) = patchify(&img, 16, 1.0 / 255.0);
        assert_eq!((gh, gw), (3, 3));
        assert_eq!(patches.len(), 9 * 16 * 16 * 3);
        let px = |patch: usize, dy: usize, dx: usize, c: usize| -> f32 {
            patches[patch * 3 * 256 + ((dy * 16) + dx) * 3 + c]
        };
        for py in 0..3usize {
            for pxi in 0..3usize {
                let patch = py * 3 + pxi; // row-major over (grid_h, grid_w)
                for dy in 0..16usize {
                    for dx in 0..16usize {
                        let x = (pxi * 16 + dx) as u32;
                        let y = (py * 16 + dy) as u32;
                        let near = |got: f32, want: f32| {
                            assert!((got - want).abs() < 1e-7, "got {got}, want {want}");
                        };
                        near(px(patch, dy, dx, 0), (x * 5) as f32 * (1.0 / 255.0));
                        near(px(patch, dy, dx, 1), (y * 3) as f32 * (1.0 / 255.0));
                        near(px(patch, dy, dx, 2), 7.0 * (1.0 / 255.0));
                    }
                }
            }
        }
    }

    #[test]
    fn positions_are_column_then_row_row_major() {
        // 48x96 (PIL/RgbImage::new(width, height) = 48 wide, 96 tall)
        // resizes to 528x1104 → a 69x33 patch grid (h x w).
        let img = image::RgbImage::new(48, 96);
        let p = process_rgb8(img, &cfg()).expect("process");
        assert_eq!((p.grid_h, p.grid_w), (69, 33));
        assert_eq!(p.n_patches, 69 * 33);
        assert_eq!(p.n_tokens, 69 * 33 / 9);
        for i in 0..p.n_patches {
            let (x, y) = p.position(i);
            assert_eq!(x as usize, i % 33, "patch {i} column");
            assert_eq!(y as usize, i / 33, "patch {i} row");
        }
    }

    #[test]
    fn process_image_decodes_a_png_and_hits_the_token_budget() {
        // 64x64 synthetic PNG → the resize upscales it to 768x768 (256
        // tokens): the token count depends on the aspect, not the source
        // resolution.
        let mut img = image::RgbImage::new(64, 64);
        for y in 0..64u32 {
            for x in 0..64u32 {
                img.put_pixel(x, y, image::Rgb([(x * 4) as u8, (y * 4) as u8, 128]));
            }
        }
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut png, image::ImageFormat::Png)
            .expect("encode png");
        let bytes = png.into_inner();

        let c = cfg();
        let p = process_image(&bytes, &c).expect("process png");
        assert_eq!((p.grid_h, p.grid_w), (48, 48));
        assert_eq!(p.n_tokens, 256);
        assert_eq!(p.patches.len(), p.n_patches * c.patch_pixels());
        assert_eq!(p.positions.len(), p.n_patches * 2);
        for v in &p.patches {
            assert!((0.0..=1.0).contains(v), "rescaled pixel out of [0,1]: {v}");
        }
        // A 1xN strip is decoded, not rejected: the aspect resize takes it.
        assert_eq!(size(64, 64), (768, 768));
    }

    #[test]
    fn the_bomb_guard_refuses_by_header_dimensions() {
        // 16_777_216 pixels exactly is the contract's ceiling; 4096² sits on it.
        assert!(guard_dimensions(4096, 4096).is_ok());
        assert!(guard_dimensions(3000, 5000).is_ok());
        let err = guard_dimensions(4096, 4097).unwrap_err();
        assert!(err.contains("exceed maximum"), "{err}");
        let err = guard_dimensions(50_000, 50_000).unwrap_err();
        assert!(err.contains("exceed maximum"), "{err}");
        let err = guard_dimensions(20_000, 1000).unwrap_err();
        assert!(err.contains("exceed maximum"), "{err}");
    }

    #[test]
    fn a_truncated_container_is_refused_without_panicking() {
        // Probing fails before any pixel buffer exists; the message is the
        // decoder's, not a panic.
        let err =
            process_image(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A], &cfg()).unwrap_err();
        assert!(!err.is_empty());
        assert!(process_image(b"not an image", &cfg()).is_err());
    }

    /// The source of every resize pin: the deterministic 8-bit RGB pattern
    /// `(31x + 17y + 7xy) mod 256` (the generator's `_pattern` family), for a
    /// given height/width. Asymmetric in both axes, so a transposed or
    /// pass-swapped resampler cannot reproduce it.
    fn pin_source(h: usize, w: usize) -> image::RgbImage {
        let mut img = image::RgbImage::new(w as u32, h as u32);
        for y in 0..h as u32 {
            for x in 0..w as u32 {
                let a = ((x as i64 * 31 + y as i64 * 17 + x as i64 * y as i64 * 7) % 256) as u8;
                img.put_pixel(
                    x,
                    y,
                    image::Rgb([a, a.wrapping_mul(3).wrapping_add(11), 255 - a]),
                );
            }
        }
        img
    }

    /// `aten_bicubic_resize` must reproduce the reference processor's resize
    /// **byte for byte** — the expected bytes below are
    /// `torch.nn.functional.interpolate(uint8, mode="bicubic",
    /// antialias=True, align_corners=False)`'s output for [`pin_source`],
    /// which is what `tvF.resize` and therefore
    /// `Gemma4ImageProcessor` produce on this checkpoint. The tower amplifies
    /// sub-LSB pixel differences, so "close enough" is not: this is the rule
    /// the image leg's parity rests on.
    #[test]
    fn aten_bicubic_resize_matches_torch_byte_for_byte() {
        let cases: [(usize, usize, usize, usize, &[u8]); 3] = [
            // h, w, out_w, out_h — upscale/upscale, downscale/downscale (the
            // widened-support branch), and x-up/y-down.
            (
                3,
                4,
                7,
                5,
                &[
                    0, 0, 255, 7, 31, 248, 27, 91, 228, 45, 172, 211, 62, 204, 193, 82, 85, 173,
                    92, 11, 163, 4, 19, 251, 14, 57, 241, 36, 121, 219, 55, 141, 200, 74, 133, 181,
                    96, 97, 159, 107, 76, 148, 14, 53, 241, 27, 102, 228, 52, 175, 203, 74, 105,
                    181, 96, 35, 159, 121, 108, 134, 134, 157, 121, 25, 87, 230, 40, 141, 215, 69,
                    225, 186, 93, 166, 162, 118, 82, 137, 146, 63, 109, 161, 61, 94, 31, 106, 224,
                    47, 162, 208, 78, 251, 177, 104, 211, 152, 129, 124, 126, 160, 33, 95, 176, 0,
                    79,
                ],
            ),
            (
                7,
                9,
                4,
                3,
                &[
                    37, 112, 218, 119, 91, 136, 162, 121, 93, 138, 156, 117, 92, 153, 163, 157,
                    115, 98, 84, 117, 171, 138, 151, 117, 147, 109, 108, 135, 154, 120, 139, 152,
                    116, 112, 94, 143,
                ],
            ),
            (
                4,
                7,
                11,
                2,
                &[
                    8, 33, 247, 23, 91, 232, 48, 158, 207, 71, 128, 184, 92, 99, 163, 115, 80, 140,
                    146, 71, 109, 163, 94, 92, 163, 148, 92, 185, 155, 70, 202, 149, 53, 37, 132,
                    218, 58, 145, 197, 93, 162, 162, 124, 164, 131, 159, 128, 96, 185, 74, 70, 145,
                    66, 110, 94, 75, 161, 48, 93, 207, 70, 119, 185, 97, 134, 158,
                ],
            ),
        ];
        for (h, w, tw, th, want) in cases {
            let got = aten_bicubic_resize(&pin_source(h, w), tw, th);
            assert_eq!(got.dimensions(), (tw as u32, th as u32));
            assert_eq!(got.as_raw().as_slice(), want, "{w}x{h} -> {tw}x{th}");
        }
    }

    /// The rule above is *load-bearing*: `image`'s `CatmullRom` resize — what
    /// this crate used before the transcription — shares the kernel and the
    /// subpixel centres but runs the two passes in the opposite order with a
    /// float intermediate, and it disagrees with the reference on the same
    /// input.
    #[test]
    fn the_image_crate_catmullrom_resize_is_not_the_reference_resize() {
        let src = pin_source(3, 4);
        let reference = aten_bicubic_resize(&src, 7, 5);
        let image = image::imageops::resize(&src, 7, 5, image::imageops::FilterType::CatmullRom);
        let diff = reference
            .as_raw()
            .iter()
            .zip(image.as_raw())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            diff > 0,
            "`image`'s CatmullRom now reproduces the processor's resize on this input; \
             re-check whether the transcription is still needed"
        );
    }
}
