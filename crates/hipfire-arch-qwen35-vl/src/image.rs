// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 nickfinease
// hipfire — see LICENSE and NOTICE in the project root.

//! Image loading and preprocessing for Qwen3.5-VL vision encoder.
//! Loads PNG/JPEG, resizes to target resolution, normalizes to [-1, 1].

use std::path::Path;

/// Maximum total pixel count, checked from format-header dimensions BEFORE
/// the pixel buffer is allocated (decompression bomb guard). This is ONLY a
/// bomb guard — it must stay well above [`VISION_MAX_PIXELS`], because any
/// image between the two is legitimately handled by downscaling in
/// [`smart_resize`], not by rejection.
///
/// It used to be 4M, which was the same order as the input budget and so
/// *rejected* ordinary documents: a 300 DPI A4 scan is ~8.7 MP and was
/// refused outright instead of being downscaled.
///
/// The value is the model's own `preprocessor_config.json`
/// `size.longest_edge`, giving a clean contract: accept anything the model's
/// config contemplates, then downscale to what our attention kernel can
/// actually hold ([`VISION_MAX_PIXELS`]). That covers 300–400 DPI A4 scans
/// (8.7–15.5 MP) while still rejecting a malicious 50000×50000 PNG (2500 MP)
/// before any pixel buffer is allocated.
const MAX_DIMENSION_PIXELS: usize = 16_777_216;

/// Lower bound on resized total pixels, from the model's own
/// `preprocessor_config.json` (`size.shortest_edge`). Images below this are
/// upscaled. Previously hardcoded to `56*56 = 3136` — a stale Qwen2-VL
/// constant — which produced a SMALLER patch grid than HF for small images
/// (measured 2026-07-26: a 224×288 input gave hipfire 252 patches vs HF 320).
const VISION_MIN_PIXELS: usize = 65_536;

/// Upper bound on resized total pixels fed to the vision tower.
///
/// The model's `preprocessor_config.json` allows `size.longest_edge =
/// 16_777_216`, but we CANNOT use that value: `vit_attention_f32` holds its
/// per-query score row in LDS, sized `(N + block_size) * 4` bytes. With a
/// 64 KB LDS budget and `block_size <= 256` that caps N at
/// `65536/4 - 256 = 16_128` patches. At `patch_size = 16` each patch covers
/// 256 px, so the hard ceiling is `16_128 * 256 = 4_128_768` px. We take
/// 4.0 MP for margin (15_625 patches).
///
/// The previous value was `14*14*4*1280 = 1_003_520` — a Qwen2-VL patch-14
/// constant that starved this patch-16 model of resolution. For dense
/// document OCR that is the difference between ~10 px and ~20 px tall body
/// text.
///
/// 2.0 MP is chosen from measurement, not from the ceiling. Sweep on the
/// dense-table OCR page (gfx1201, OvisOCR2 q8, 2026-07-26):
///
/// | budget | visual tokens | vision encode | content recall |
/// |--------|--------------:|--------------:|---------------:|
/// | 1.0 MP |         1_019 |        2.61 s |            5/9 |
/// | 2.0 MP |         1_947 |        7.76 s |        **8/9** |
/// | 4.0 MP |         3_757 |       28.67 s |            5/9 |
///
/// Vision attention is O(N^2), so cost grows ~13x from 1->4 MP while quality
/// peaks in the middle: 4 MP is both slower AND worse. Raising this further
/// needs a tiled/flash vision attention kernel that does not keep all N
/// scores in LDS; until then 2 MP is the quality optimum.
///
/// Override with `HIPFIRE_VL_MAX_PIXELS` for experiments; values above the
/// LDS ceiling are clamped rather than trusted.
const VISION_MAX_PIXELS: usize = 2_000_000;

/// LDS-derived hard ceiling on patch count for `vit_attention_f32`.
const VISION_LDS_MAX_PIXELS: usize = 4_128_768;

/// Resolve the vision input pixel budget, honouring `HIPFIRE_VL_MAX_PIXELS`
/// but never exceeding what the attention kernel's LDS can hold.
fn vision_max_pixels() -> usize {
    hipfire_config::developer_var("HIPFIRE_VL_MAX_PIXELS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .map(|v| v.min(VISION_LDS_MAX_PIXELS))
        .unwrap_or(VISION_MAX_PIXELS)
}

/// Smart resize matching HuggingFace Qwen2_5_VLImageProcessor.
///
/// `factor` MUST equal `patch_size * spatial_merge_size`. With that constraint
/// the returned (h, w) are multiples of `patch_size * sms`, which guarantees
/// (1) clean patch extraction at `patch_size` stride and (2) a patch grid
/// divisible by `sms` so the spatial merger does not silently truncate a
/// row/column. Passing any other factor (e.g. the legacy `28` from Qwen2-VL
/// when patch_size=16) yields odd patch grids on small images and a
/// merger/LM token-count mismatch downstream.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    // Use u64 for pixel-count arithmetic to avoid usize overflow on
    // 32-bit targets where height * width could exceed 2^32-1.
    // (Defense-in-depth; callers are already gated by
    // MAX_DIMENSION_PIXELS, but the arithmetic should be correct
    // regardless of pointer width.)
    let h_bar = ((height as f64 / factor as f64).round() as usize) * factor;
    let w_bar = ((width as f64 / factor as f64).round() as usize) * factor;
    let hw = (height as u64) * (width as u64);
    let hbar_wbar = (h_bar as u64) * (w_bar as u64);

    if hbar_wbar > max_pixels as u64 {
        let beta = (hw as f64 / max_pixels as f64).sqrt();
        let h_bar = factor.max(((height as f64 / beta / factor as f64).floor() as usize) * factor);
        let w_bar = factor.max(((width as f64 / beta / factor as f64).floor() as usize) * factor);
        (h_bar, w_bar)
    } else if hbar_wbar < min_pixels as u64 {
        let beta = (min_pixels as f64 / hw as f64).sqrt();
        let h_bar = factor.max(((height as f64 * beta / factor as f64).ceil() as usize) * factor);
        let w_bar = factor.max(((width as f64 * beta / factor as f64).ceil() as usize) * factor);
        (h_bar, w_bar)
    } else {
        (h_bar, w_bar)
    }
}

/// Shared preprocessing logic that takes an already-loaded `DynamicImage`.
/// Returns (CHW data, height, width) where height and width are multiples of
/// `patch_size * spatial_merge_size`.
fn preprocess_dynamic_image(
    img: image::DynamicImage,
    patch_size: usize,
    spatial_merge_size: usize,
) -> (Vec<f32>, usize, usize) {
    let (orig_w, orig_h) = (img.width() as usize, img.height() as usize);

    let factor = patch_size * spatial_merge_size;
    let min_pixels = VISION_MIN_PIXELS;
    let max_pixels = vision_max_pixels();
    let (final_h, final_w) = smart_resize(orig_h, orig_w, factor, min_pixels, max_pixels);

    // HF's `Qwen2VLImageProcessorFast` uses PIL's BICUBIC (`resample=3`).
    // `FilterType::CatmullRom` is the `image` crate's bicubic filter; this
    // closes the rel-L1 residual measured in the May 2026 vs-HF diff (was
    // 0.002 with bilinear Triangle; CatmullRom drops it below model
    // sensitivity). See `benchmarks/vision/comparison-2026-05-23.md`.
    let img = img.resize_exact(
        final_w as u32,
        final_h as u32,
        image::imageops::FilterType::CatmullRom,
    );

    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);

    // CHW float in straight [R, G, B] order, normalize: pixel / 127.5 - 1.0.
    //
    // History: previously had a deliberate B<->G swap (storing as [R, B, G])
    // because pure-color PNG tests said red/green/blue were misnamed without
    // it. That diagnosis was wrong — the real cause was the (T,C,H,W) vs
    // (C,T,H,W) per-patch transpose in `extract_patches` below, which on a
    // single-color image happens to be re-fixable by any single channel
    // permutation. On natural images the two bugs compound and the swap
    // makes things strictly worse. Verified byte-identical to HF's
    // Qwen2VLImageProcessorFast on `barney_cigar.jpg` once the layout is
    // straight RGB AND extract_patches uses (C,T,H,W). See
    // benchmarks/vision/comparison-2026-05-23.md.
    let mut out = vec![0.0f32; 3 * h * w];
    let plane = h * w;
    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            let idx = y * w + x;
            out[idx] = pixel[0] as f32 / 127.5 - 1.0; // channel 0 = R
            out[plane + idx] = pixel[1] as f32 / 127.5 - 1.0; // channel 1 = G
            out[2 * plane + idx] = pixel[2] as f32 / 127.5 - 1.0; // channel 2 = B
        }
    }
    (out, h, w)
}

/// Load an image from a filesystem path, smart-resize, normalize.
///
/// Returns an error string instead of panicking so the daemon's `ImageSource::Path`
/// dispatch can surface a clean error to the client rather than crashing the
/// process on a missing file or corrupt header. Tests and examples that want
/// to abort on error can chain `.expect("...")`.
pub fn load_and_preprocess(
    path: &Path,
    patch_size: usize,
    spatial_merge_size: usize,
) -> Result<(Vec<f32>, usize, usize), String> {
    let img =
        image::open(path).map_err(|e| format!("failed to open image {}: {e}", path.display()))?;
    Ok(preprocess_dynamic_image(
        img,
        patch_size,
        spatial_merge_size,
    ))
}

/// Load an image from raw bytes (PNG or JPEG), smart-resize, normalize.
/// Returns `Result` so callers can surface decode errors.
///
/// Reads dimensions from the format header BEFORE decoding pixels so a
/// decompression-bomb image (e.g. 50000×50000 PNG that compresses to ~50 KB
/// but expands to multi-GB raw) is rejected before allocation. Format
/// rejection (non-PNG/JPEG) is via `ImageError::Unsupported` rather than
/// substring matching so the error surface is stable across `image` crate
/// versions.
pub fn load_and_preprocess_from_bytes(
    data: &[u8],
    patch_size: usize,
    spatial_merge_size: usize,
) -> Result<(Vec<f32>, usize, usize), String> {
    let reader = image::ImageReader::new(std::io::Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| format!("failed to read image: {e}"))?;

    let (orig_w, orig_h) = reader.into_dimensions().map_err(map_image_err)?;
    let (orig_w, orig_h) = (orig_w as usize, orig_h as usize);
    if orig_w * orig_h > MAX_DIMENSION_PIXELS {
        return Err(format!(
            "image dimensions ({orig_w}x{orig_h}) exceed maximum ({MAX_DIMENSION_PIXELS} pixels)"
        ));
    }

    let img = image::load_from_memory(data).map_err(map_image_err)?;
    Ok(preprocess_dynamic_image(
        img,
        patch_size,
        spatial_merge_size,
    ))
}

fn map_image_err(e: image::ImageError) -> String {
    match e {
        image::ImageError::Unsupported(_) => {
            "unsupported image format — supported: png, jpeg".to_string()
        }
        other => format!("failed to decode image: {other}"),
    }
}

/// Extract non-overlapping patches from a CHW image.
///
/// Input: `[C, H, W]` where H and W are divisible by `patch_size *
/// spatial_merge_size` (enforced by [`smart_resize`]). Output:
/// `[N, temporal_patch_size * C * patch_size * patch_size]` with
/// `N = (H/patch_size) * (W/patch_size)`, ordered to match HuggingFace
/// `Qwen2VLImageProcessorFast`:
///
///   * Patches are emitted in **`(spatial_merge_size × spatial_merge_size)`
///     spatial-merge-grouped order**: outer `(gy, gx)` row-major over
///     `(ph/SMS, pw/SMS)`, inner `(sy, sx)` row-major over `(SMS, SMS)`. This
///     means `SMS²` consecutive patches in the output buffer form one
///     spatial-merge output token.
///   * Per-patch layout is `(C, T, patch_h, patch_w)` flat —
///     **channel-outer, temporal-inner**.
///
/// `temporal_patch_size > 1` duplicates the same frame across the T slots
/// (this is image, not video — video pipelines must call a separate API).
///
/// Pre-2026-05-23 this function used row-major patch order and `(T, C, h, w)`
/// per-patch layout, both of which disagree with HF. Combined with the
/// `[R, B, G]` channel swap in [`preprocess_dynamic_image`] (also reverted in
/// that change) the vision tower received patches that were scrambled in
/// three independent dimensions — model perceived every image as "vertically
/// stretched / low-resolution / blurry". Diagnosed by element-wise diff
/// against HF reference on `barney_cigar.jpg`: rel-L1 dropped from 0.35
/// (pre-fix) to 0.002 (post-fix, residual is the resize-filter difference).
/// See `benchmarks/vision/comparison-2026-05-23.md` and `diff_dumps.py`.
pub fn extract_patches(
    chw: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
) -> Vec<f32> {
    let ph = height / patch_size;
    let pw = width / patch_size;
    let n_patches = ph * pw;
    let patch_elems = temporal_patch_size * channels * patch_size * patch_size;
    let mut patches = vec![0.0f32; n_patches * patch_elems];

    assert!(
        spatial_merge_size >= 1 && ph % spatial_merge_size == 0 && pw % spatial_merge_size == 0,
        "patch grid {ph}x{pw} not divisible by spatial_merge_size={spatial_merge_size} — \
         smart_resize should guarantee this",
    );
    let gw = pw / spatial_merge_size;

    for py in 0..ph {
        for px in 0..pw {
            let gy = py / spatial_merge_size;
            let gx = px / spatial_merge_size;
            let sy = py % spatial_merge_size;
            let sx = px % spatial_merge_size;
            // SMS×SMS-block-grouped row-major: ((gy, gx), (sy, sx)) flattened.
            let patch_out_idx =
                ((gy * gw + gx) * spatial_merge_size + sy) * spatial_merge_size + sx;
            let out_base = patch_out_idx * patch_elems;

            for c in 0..channels {
                for t in 0..temporal_patch_size {
                    let _ = t; // same frame duplicated for both temporal slots
                    for dy in 0..patch_size {
                        for dx in 0..patch_size {
                            let y = py * patch_size + dy;
                            let x = px * patch_size + dx;
                            let src_idx = c * height * width + y * width + x;
                            let dst_idx = out_base
                                + c * temporal_patch_size * patch_size * patch_size
                                + t * patch_size * patch_size
                                + dy * patch_size
                                + dx;
                            patches[dst_idx] = chw[src_idx];
                        }
                    }
                }
            }
        }
    }
    patches
}

/// VCN JPEG decode path (`image.decode = vcn|auto`, behind the `vcn-jpeg`
/// cargo feature, default off). Mirrors `vision.mode` resolution via
/// `hipfire-config`: `cpu` never touches VCN, `vcn`/`auto` attempt the pooled
/// libva decode and fall back to the CPU path on anything unexpected (a VCN
/// attempt NEVER fails the request — the CPU path is always correct).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ImageDecode {
    Cpu,
    Vcn,
    Auto,
}

/// Resolve `image.decode` (`HIPFIRE_IMAGE_DECODE` compat) from the process
/// snapshot. Default `cpu`; unrecognized values fail safe to `cpu`.
pub fn resolve_image_decode() -> ImageDecode {
    match hipfire_config::process_value("HIPFIRE_IMAGE_DECODE").as_deref() {
        Some("vcn") => ImageDecode::Vcn,
        Some("auto") => ImageDecode::Auto,
        _ => ImageDecode::Cpu,
    }
}

#[cfg(feature = "vcn-jpeg")]
use rdna_compute::{DType, Gpu, GpuTensor};
#[cfg(feature = "vcn-jpeg")]
pub use va_bridge::VcnFrame;

/// Device-resident patches from the VCN path, ready for
/// [`crate::qwen35_vl::vision_forward_patches`] (no upload, no CPU pixels).
#[cfg(feature = "vcn-jpeg")]
pub struct VcnPatches {
    pub patches: GpuTensor,
    pub img_h: usize,
    pub img_w: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

/// A pooled VCN decode plus its resized target dims — no GPU allocation
/// (the mapping is session-pooled), so this can run before the daemon's
/// capacity checks. [`vcn_to_patches`] does the kernel launches after them.
#[cfg(feature = "vcn-jpeg")]
pub struct VcnDecoded {
    pub frame: VcnFrame,
    pub img_h: usize,
    pub img_w: usize,
}

#[cfg(feature = "vcn-jpeg")]
const VL_YUV_PREPROCESS_SRC: &str =
    include_str!("../../../kernels/src/vl_yuv_preprocess.hip");
#[cfg(feature = "vcn-jpeg")]
const VL_RGB_KERNEL: &str = "vl_nv12_to_rgb_norm";
#[cfg(feature = "vcn-jpeg")]
const VL_PATCH_KERNEL: &str = "vl_extract_patches";
#[cfg(feature = "vcn-jpeg")]
const FOURCC_444P: u32 = 0x5034_3434;

/// Log-once gate for the expected `auto`-on-CPU-host fallback.
#[cfg(feature = "vcn-jpeg")]
static VCN_UNAVAILABLE_LOGGED: std::sync::Once = std::sync::Once::new();

/// Pooled VCN decode + resized target dims, no GPU allocation. Returns
/// `None` when the CPU path should be used (`image.decode = cpu`,
/// non-JPEG input, VCN-unsupported streams, missing hardware, or a fourcc
/// the preprocess kernels have no arm for). A `None` here is never an
/// error — the CPU path is always correct.
#[cfg(feature = "vcn-jpeg")]
pub fn vcn_decode(
    data: &[u8],
    patch_size: usize,
    spatial_merge_size: usize,
) -> Option<VcnDecoded> {
    let mode = resolve_image_decode();
    if mode == ImageDecode::Cpu {
        return None;
    }
    let frame = match va_bridge::VaSession::shared_decode_jpeg(data) {
        Ok(va_bridge::DecodeOutcome::Decoded(f)) => f,
        Ok(va_bridge::DecodeOutcome::Unsupported(reason)) => {
            if mode == ImageDecode::Vcn {
                eprintln!("[vl-vcn] VCN unsupported ({reason}) — CPU fallback");
            }
            return None;
        }
        Err(e) => {
            let unavailable = matches!(
                e,
                va_bridge::VaError::NoRenderNode
                    | va_bridge::VaError::Dlopen { .. }
                    | va_bridge::VaError::MissingSymbol { .. }
            );
            if mode == ImageDecode::Vcn || !unavailable {
                eprintln!("[vl-vcn] VCN decode failed ({e}) — CPU fallback");
            } else {
                VCN_UNAVAILABLE_LOGGED.call_once(|| {
                    eprintln!("[vl-vcn] VCN unavailable ({e}) — CPU fallback");
                });
            }
            return None;
        }
    };
    if frame.fourcc != va_bridge::VA_FOURCC_NV12 && frame.fourcc != FOURCC_444P {
        if mode == ImageDecode::Vcn {
            eprintln!(
                "[vl-vcn] fourcc 0x{:08x} has no kernel arm — CPU fallback",
                frame.fourcc
            );
        }
        return None;
    }
    // Same resize contract as the CPU path (`preprocess_dynamic_image`).
    let factor = patch_size * spatial_merge_size;
    let (t_h, t_w) = smart_resize(
        frame.height as usize,
        frame.width as usize,
        factor,
        VISION_MIN_PIXELS,
        vision_max_pixels(),
    );
    Some(VcnDecoded {
        frame,
        img_h: t_h,
        img_w: t_w,
    })
}

/// Chroma addressing for [`vcn_to_patches`]: NV12 interleaved or planar 444.
#[cfg(feature = "vcn-jpeg")]
fn vcn_chroma(frame: &VcnFrame) -> Option<(u32, u32, u32, u32, u32)> {
    if frame.fourcc == va_bridge::VA_FOURCC_NV12 {
        Some((frame.uv_offset(), frame.uv_offset() + 1, 2, 1, 1))
    } else if frame.fourcc == FOURCC_444P && frame.num_layers >= 3 {
        Some((
            frame.layers[1].offset[0],
            frame.layers[2].offset[0],
            1,
            0,
            0,
        ))
    } else {
        None
    }
}

/// Kernel launches for a [`vcn_decode`] result: NV12/planar → CHW f32 →
/// device patches. Joins the `Gpu` stream, ordered before the vision tower.
/// `Err` is GPU-side failure only (decode fallbacks already returned `None`
/// above); the caller falls back to a CPU decode of the retained bytes.
#[cfg(feature = "vcn-jpeg")]
pub fn vcn_to_patches(
    gpu: &mut Gpu,
    dec: &VcnDecoded,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
) -> Result<VcnPatches, String> {
    let (t_h, t_w) = (dec.img_h, dec.img_w);
    let frame = &dec.frame;
    let Some((u_off, v_off, step, sh_x, sh_y)) = vcn_chroma(frame) else {
        return Err(format!(
            "vcn fourcc 0x{:08x} has no kernel arm",
            frame.fourcc
        ));
    };
    let n_elem =
        (t_h / patch_size) * (t_w / patch_size) * temporal_patch_size * 3 * patch_size * patch_size;
    if n_elem == 0 {
        return Err("vcn empty patch grid".to_string());
    }
    let map_err = |op: &'static str| move |e: hip_bridge::HipError| format!("vcn {op}: {e}");
    gpu.ensure_kernel_public("vl_yuv_preprocess", VL_YUV_PREPROCESS_SRC, VL_RGB_KERNEL)
        .map_err(map_err("ensure rgb kernel"))?;
    gpu.ensure_kernel_public("vl_yuv_preprocess", VL_YUV_PREPROCESS_SRC, VL_PATCH_KERNEL)
        .map_err(map_err("ensure patch kernel"))?;
    let n_chw = 3 * t_h * t_w;
    let d_chw = gpu
        .alloc_tensor(&[n_chw], DType::F32)
        .map_err(map_err("alloc chw"))?;
    let mut b1 = hip_bridge::KernargBlob::new();
    b1.push_ptr(frame.device_ptr() as *const std::ffi::c_void);
    for v in [
        frame.y_pitch(),
        frame.uv_pitch(),
        u_off,
        v_off,
        step,
        frame.width,
        frame.height,
        sh_x,
        sh_y,
        t_w as u32,
        t_h as u32,
    ] {
        b1.push_u32(v);
    }
    b1.push_ptr(d_chw.buf.as_ptr() as *const std::ffi::c_void);
    b1.pad_to(16);
    gpu.launch_kernel_blob(
        VL_RGB_KERNEL,
        [(t_w as u32 + 15) / 16, (t_h as u32 + 15) / 16, 1],
        [16, 16, 1],
        0,
        b1.as_mut_slice(),
    )
    .map_err(map_err("launch rgb kernel"))?;
    let patches = gpu
        .alloc_tensor(&[n_elem], DType::F32)
        .map_err(map_err("alloc patches"))?;
    let mut b2 = hip_bridge::KernargBlob::new();
    b2.push_ptr(d_chw.buf.as_ptr() as *const std::ffi::c_void);
    for v in [
        t_h as u32,
        t_w as u32,
        patch_size as u32,
        temporal_patch_size as u32,
        spatial_merge_size as u32,
    ] {
        b2.push_u32(v);
    }
    b2.push_ptr(patches.buf.as_ptr() as *const std::ffi::c_void);
    b2.pad_to(16);
    gpu.launch_kernel_blob(
        VL_PATCH_KERNEL,
        [(n_elem as u32 + 255) / 256, 1, 1],
        [256, 1, 1],
        0,
        b2.as_mut_slice(),
    )
    .map_err(map_err("launch patch kernel"))?;
    gpu.free_tensor(d_chw).map_err(map_err("free chw"))?;
    eprintln!(
        "[vl-vcn] VCN decode {}x{} -> {}x{} ({} patches)",
        frame.width,
        frame.height,
        t_w,
        t_h,
        (t_h / patch_size) * (t_w / patch_size)
    );
    Ok(VcnPatches {
        patches,
        img_h: t_h,
        img_w: t_w,
        grid_h: t_h / patch_size,
        grid_w: t_w / patch_size,
    })
}

/// One-call VCN JPEG → device patches: [`vcn_decode`] then
/// [`vcn_to_patches`]. `Ok(None)` = take the CPU path; `Err` = GPU-side
/// failure after a successful decode.
#[cfg(feature = "vcn-jpeg")]
pub fn try_vcn_preprocess(
    gpu: &mut Gpu,
    data: &[u8],
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
) -> Result<Option<VcnPatches>, String> {
    let Some(dec) = vcn_decode(data, patch_size, spatial_merge_size) else {
        return Ok(None);
    };
    vcn_to_patches(gpu, &dec, patch_size, temporal_patch_size, spatial_merge_size).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic CHW image with distinguishable per-pixel values so
    /// any patch-order or per-patch-layout regression produces a wrong byte
    /// at a known output index.
    ///
    /// Encoding: `chw[c * H * W + y * W + x] = c * 10_000 + y * 100 + x`.
    fn synthetic_chw(channels: usize, h: usize, w: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; channels * h * w];
        for c in 0..channels {
            for y in 0..h {
                for x in 0..w {
                    out[c * h * w + y * w + x] = (c * 10_000 + y * 100 + x) as f32;
                }
            }
        }
        out
    }

    /// extract_patches in (C, T, ph, pw)-per-patch + 2x2-grouped patch order
    /// on a 4×4 image (1 patch grid 2×2 ⇒ 1 spatial-merge block). Locks the
    /// permutation that the May 2026 fix put in place — any future revert to
    /// row-major patch order or to (T, C, ph, pw) per-patch fails fast here.
    ///
    /// 4×4 image, patch_size=2, T=2, SMS=2: ph=pw=2, n_patches=4, all 4
    /// patches in one merge block. Patch out_idx for (gy=0, gx=0, sy, sx) is
    /// (sy*SMS + sx).
    #[test]
    fn extract_patches_locks_layout_and_order_4x4() {
        let chw = synthetic_chw(3, 4, 4);
        let patches = extract_patches(
            &chw, 3, 4, 4, /*patch_size=*/ 2, /*T=*/ 2, /*SMS=*/ 2,
        );
        // Per-patch element count: T * C * ph * pw = 2 * 3 * 2 * 2 = 24.
        // Total: 4 patches × 24 = 96.
        assert_eq!(patches.len(), 96);

        // Patch (py=0, px=1) — top-right patch. It maps to out_idx = sy=0, sx=1 = 1.
        // Per-patch base in the output buffer:
        //   patch_out_idx 1 → out_base = 1 * 24 = 24
        // Per-patch layout (C, T, ph, pw): for c=0, t=0, dy=0, dx=0 the source
        // pixel is at (py*ps+dy, px*ps+dx) = (0, 2) so value = 0*10000 + 0*100 + 2 = 2.
        let v = patches[24];
        assert_eq!(
            v, 2.0,
            "patch_out_idx=1, c=0,t=0,dy=0,dx=0 should hold (0,2)=2"
        );
        // Same patch, c=2 (B), t=1, dy=1, dx=1:
        //   src = (0*4 + 1) = y=1, x=2*2+1=3, c=2 → 2*10000 + 1*100 + 3 = 20103.
        //   dst offset within patch = 2 * 8 + 1 * 4 + 1 * 2 + 1 = 23.
        let v = patches[24 + 23];
        assert_eq!(
            v, 20103.0,
            "patch_out_idx=1, c=2,t=1,dy=1,dx=1 should hold (2,1,3)"
        );

        // Patch (py=1, px=0) — bottom-left. out_idx = sy=1, sx=0 = 2.
        // Per-patch c=1, t=0, dy=0, dx=0: src = (y=2, x=0, c=1) → 10000 + 200 + 0 = 10200.
        // dst offset = 1*8 + 0*4 + 0*2 + 0 = 8.
        let v = patches[2 * 24 + 8];
        assert_eq!(
            v, 10200.0,
            "patch_out_idx=2, c=1,t=0,dy=0,dx=0 should hold (1,2,0)"
        );
    }

    /// 4×6 image, patch_size=2, SMS=2: ph=2, pw=3 — non-square. ph%SMS=0,
    /// pw%SMS != 0 ⇒ should panic. This guards the assertion contract.
    #[test]
    #[should_panic(expected = "not divisible by spatial_merge_size")]
    fn extract_patches_rejects_indivisible_grid() {
        let chw = synthetic_chw(3, 4, 6);
        let _ = extract_patches(
            &chw, 3, 4, 6, /*patch_size=*/ 2, /*T=*/ 2, /*SMS=*/ 2,
        );
    }

    /// SMS=4 on a 8×8 image: ph=pw=4, divisible. Spot-check the 4x4-grouping
    /// math at a non-2 merge size — defends the new `spatial_merge_size`
    /// parameter against being silently re-hardcoded.
    #[test]
    fn extract_patches_supports_sms_4() {
        let chw = synthetic_chw(3, 8, 8);
        let patches = extract_patches(
            &chw, 3, 8, 8, /*patch_size=*/ 2, /*T=*/ 1, /*SMS=*/ 4,
        );
        // ph=pw=4, n=16 patches, patch_elems = 1*3*2*2 = 12.
        assert_eq!(patches.len(), 16 * 12);
        // With SMS=4 the entire 4×4 grid is ONE merge block (mh=mw=1).
        // patch (py, px) maps to out_idx = ((0,0), (py, px)) = py * 4 + px.
        // So patch (py=2, px=3) → out_idx = 11.
        // c=0, dy=0, dx=0 of that patch: src y=4, x=6 → 0 + 400 + 6 = 406.
        let v = patches[11 * 12];
        assert_eq!(v, 406.0, "SMS=4 patch ordering: (py=2, px=3) → out_idx=11");
    }
}
