// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Reference image preprocessing for FLUX.2 Klein image-editing conditioning:
//! decode, area-capped resize, floor-to-multiple-of-16 snap, and mapping to
//! the `[-1, 1]` channel-major float tensor the VAE encoder expects.

use std::path::Path;

/// Maximum reference image area (pixels) before downscaling kicks in.
pub const MAX_REF_AREA: u64 = 1024 * 1024;
/// Both output sides are floored to a multiple of this (VAE/patchify grid).
pub const REF_MULTIPLE: usize = 16;

/// Target size: scale down when area > [`MAX_REF_AREA`], then floor each
/// side to a multiple of [`REF_MULTIPLE`] (minimum one multiple per side).
pub fn target_size(w: usize, h: usize) -> (usize, usize) {
    let area = (w as u64) * (h as u64);
    let (mut tw, mut th) = (w as f64, h as f64);
    if area > MAX_REF_AREA {
        let s = ((MAX_REF_AREA as f64) / (area as f64)).sqrt();
        tw *= s;
        th *= s;
    }
    let snap = |v: f64| ((v.floor() as usize) / REF_MULTIPLE * REF_MULTIPLE).max(REF_MULTIPLE);
    (snap(tw), snap(th))
}

/// A decoded, resized reference image: `pixels` is `[3][height][width]`
/// (channel-major) in `[-1, 1]`.
pub struct RefImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<f32>,
}

/// Resize (if needed) to [`target_size`] and map to `[-1, 1]`
/// channel-major — the testable core of [`load_reference`].
pub fn prepare_reference(rgb: &image::RgbImage) -> RefImage {
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let (tw, th) = target_size(w, h);
    let resized = if (tw, th) == (w, h) {
        rgb.clone()
    } else {
        image::imageops::resize(
            rgb,
            tw as u32,
            th as u32,
            image::imageops::FilterType::Lanczos3,
        )
    };
    let mut pixels = vec![0.0f32; 3 * tw * th];
    for (x, y, p) in resized.enumerate_pixels() {
        for c in 0..3 {
            pixels[c * tw * th + y as usize * tw + x as usize] = p[c] as f32 / 127.5 - 1.0;
        }
    }
    RefImage {
        width: tw,
        height: th,
        pixels,
    }
}

/// Largest encoded reference image the daemon decodes. A 1 MP PNG is under
/// 4 MB; the cap bounds what one request can make the decoder allocate.
pub const MAX_REFERENCE_BYTES: usize = 32 << 20;

/// Decode an encoded reference image (PNG/JPEG bytes) that arrived over the
/// wire, resize to [`target_size`], and map to `[-1, 1]` channel-major.
/// This is the only entry the daemon uses: the request carries the bytes,
/// never a server-side path, so a client cannot make the server read a file.
pub fn decode_reference(bytes: &[u8]) -> Result<RefImage, String> {
    if bytes.len() > MAX_REFERENCE_BYTES {
        return Err(format!(
            "reference image is {} bytes; at most {MAX_REFERENCE_BYTES} are accepted",
            bytes.len()
        ));
    }
    let rgb = hipfire_runtime::imagedec::decode_rgb8(bytes)
        .map_err(|e| format!("reference image: {e}"))?;
    Ok(prepare_reference(&rgb))
}

/// [`decode_reference`] on a file — for the lab gates and examples, which
/// read fixtures from disk. Not a daemon path.
pub fn load_reference(path: &Path) -> Result<RefImage, String> {
    let rgb = hipfire_runtime::imagedec::decode_rgb8_path(path)
        .map_err(|e| format!("{e} ({})", path.display()))?;
    Ok(prepare_reference(&rgb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_size_keeps_small_images_and_floors_to_16() {
        assert_eq!(target_size(1024, 1024), (1024, 1024));
        assert_eq!(target_size(1000, 700), (992, 688));
        assert_eq!(target_size(100, 100), (96, 96));
    }

    #[test]
    fn target_size_scales_large_images_to_the_area_cap() {
        let (w, h) = target_size(2048, 1024);
        assert!((w as u64) * (h as u64) <= MAX_REF_AREA);
        assert_eq!(w % 16, 0);
        assert_eq!(h % 16, 0);
        // sqrt(1/2) scale: 2048*0.7071=1448 → 1440, 1024*0.7071=724 → 720
        assert_eq!((w, h), (1440, 720));
    }

    #[test]
    fn prepare_reference_maps_pixels_to_minus_one_one_channel_major() {
        let mut img = image::RgbImage::new(32, 16);
        for p in img.pixels_mut() {
            *p = image::Rgb([0, 128, 255]);
        }
        let r = prepare_reference(&img);
        assert_eq!((r.width, r.height), (32, 16));
        assert_eq!(r.pixels.len(), 3 * 32 * 16);
        assert!((r.pixels[0] + 1.0).abs() < 1e-6); // R
        assert!((r.pixels[32 * 16] - (128.0 / 127.5 - 1.0)).abs() < 1e-6); // G
        assert!((r.pixels[2 * 32 * 16] - 1.0).abs() < 1e-6); // B
    }

    /// The wire decoder is what the daemon trusts with client bytes: it must
    /// bound its input and fail closed on anything that is not an image.
    #[test]
    fn decode_reference_caps_size_and_rejects_non_images() {
        let too_big = vec![0u8; MAX_REFERENCE_BYTES + 1];
        let err = decode_reference(&too_big)
            .err()
            .expect("oversized input must fail");
        assert!(err.contains("at most"), "{err}");
        let err = decode_reference(b"/etc/passwd")
            .err()
            .expect("non-image bytes must fail");
        assert!(err.starts_with("reference image:"), "{err}");
        let mut png = Vec::new();
        image::RgbImage::from_fn(20, 40, |_, _| image::Rgb([1, 2, 3]))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let r = decode_reference(&png).unwrap();
        assert_eq!((r.width, r.height), (16, 32));
    }
}
