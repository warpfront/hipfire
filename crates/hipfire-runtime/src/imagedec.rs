// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared image decoding for every vision carrier.
//!
//! JPEG bytes (sniffed by the `FF D8` SOI marker) decode through
//! `libjpeg-turbo-rs` — `decompress_to(.., PixelFormat::Rgb)` — whose
//! pixels are byte-identical to C libjpeg-turbo / PIL. Anything else
//! (PNG, …) goes through `image::load_from_memory`, exactly as before,
//! so PNG alpha handling in the carriers is untouched. The `image`
//! crate is built WITHOUT its `jpeg` feature: zune-jpeg is out of the
//! lock and no vision carrier can silently route JPEG through it.
//!
//! Only the byte→pixel step moved here. Downstream processing in each
//! carrier (smart_resize, CatmullRom resize, normalize, patchify) is
//! untouched.

use std::path::Path;

/// JPEG start-of-image marker. Sniffed before touching either decoder so
/// PNG (and its alpha channel) never routes through the JPEG path.
pub fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8
}

fn map_image_err(e: image::ImageError) -> String {
    match e {
        image::ImageError::Unsupported(_) => {
            "unsupported image format — supported: png, jpeg".to_string()
        }
        other => format!("failed to decode image: {other}"),
    }
}

fn jpeg_dimensions(bytes: &[u8]) -> Result<(u32, u32), String> {
    match libjpeg_turbo_rs::probe(bytes) {
        Ok(info) => {
            let w = u32::try_from(info.width)
                .map_err(|_| format!("failed to decode image: width {} overflows u32", info.width))?;
            let h = u32::try_from(info.height)
                .map_err(|_| format!("failed to decode image: height {} overflows u32", info.height))?;
            Ok((w, h))
        }
        // Header unreadable after SOI matched: same wording the carriers
        // used when `into_dimensions` failed on a detected-but-corrupt JPEG.
        Err(e) => Err(format!("failed to decode image: {e}")),
    }
}

/// `(width, height)` from the format header WITHOUT decoding pixels, so
/// decompression-bomb images are rejected before allocation.
/// JPEG → turbo header probe; anything else → `image` header inspect.
pub fn probe_dimensions(bytes: &[u8]) -> Result<(u32, u32), String> {
    if is_jpeg(bytes) {
        return jpeg_dimensions(bytes);
    }
    let reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("failed to read image: {e}"))?;
    reader.into_dimensions().map_err(map_image_err)
}

/// [`probe_dimensions`] on a file.
pub fn probe_dimensions_path(path: &Path) -> Result<(u32, u32), String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("failed to open image {}: {e}", path.display()))?;
    probe_dimensions(&bytes)
}

/// Decode image bytes to an RGB8 buffer. JPEG → libjpeg-turbo-rs;
/// anything else → `image::load_from_memory(..).to_rgb8()`.
pub fn decode_rgb8(bytes: &[u8]) -> Result<image::RgbImage, String> {
    if is_jpeg(bytes) {
        let img = libjpeg_turbo_rs::decompress_to(bytes, libjpeg_turbo_rs::PixelFormat::Rgb)
            .map_err(|e| format!("failed to decode image: {e}"))?;
        let w = u32::try_from(img.width)
            .map_err(|_| format!("failed to decode image: width {} overflows u32", img.width))?;
        let h = u32::try_from(img.height)
            .map_err(|_| format!("failed to decode image: height {} overflows u32", img.height))?;
        return image::RgbImage::from_raw(w, h, img.data)
            .ok_or_else(|| "failed to decode image: pixel buffer size mismatches dimensions".to_string());
    }
    image::load_from_memory(bytes)
        .map(|dyn_img| dyn_img.to_rgb8())
        .map_err(map_image_err)
}

/// [`decode_rgb8`] on a file.
pub fn decode_rgb8_path(path: &Path) -> Result<image::RgbImage, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("failed to open image {}: {e}", path.display()))?;
    decode_rgb8(&bytes)
}

/// Decode image bytes to a `DynamicImage`, preserving the pixel variant
/// for non-JPEG inputs (notably `ImageRgba8`, which dots.ocr composites
/// onto white). JPEG (always RGB8/YUV) arrives as `ImageRgb8` — the same
/// variant `image::load_from_memory` produced for it.
pub fn decode_dynamic(bytes: &[u8]) -> Result<image::DynamicImage, String> {
    if is_jpeg(bytes) {
        return decode_rgb8(bytes).map(image::DynamicImage::ImageRgb8);
    }
    image::load_from_memory(bytes).map_err(map_image_err)
}

/// [`decode_dynamic`] on a file.
pub fn decode_dynamic_path(path: &Path) -> Result<image::DynamicImage, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("failed to open image {}: {e}", path.display()))?;
    decode_dynamic(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;

    fn doge_bytes() -> Vec<u8> {
        let path = format!(
            "{}/../../benchmarks/vision/images/doge.jpeg",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::read(&path).unwrap_or_else(|e| panic!("doge.jpeg fixture missing at {path}: {e}"))
    }

    #[test]
    fn is_jpeg_sniffs_soi_only() {
        assert!(is_jpeg(&[0xFF, 0xD8, 0xFF, 0xE0]));
        assert!(!is_jpeg(&[]));
        assert!(!is_jpeg(&[0xFF]));
        // PNG magic must NOT route to the JPEG path.
        assert!(!is_jpeg(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]));
        assert!(!is_jpeg(b"/etc/passwd"));
    }

    /// PIL reference (run from the workspace root):
    /// `python3 -c "from PIL import Image;import hashlib,numpy as np;im=np.asarray(Image.open('benchmarks/vision/images/doge.jpeg').convert('RGB'));print(im.shape,hashlib.sha256(im.tobytes()).hexdigest())"`
    /// → `(529, 537, 3) 45bb7423193c00359695e1c967676d86e82bd3f5d55aa1679a85df6c74a9cf55`
    #[test]
    fn doge_jpeg_is_byte_identical_to_pil() {
        let bytes = doge_bytes();
        assert!(is_jpeg(&bytes));
        let rgb = decode_rgb8(&bytes).expect("doge.jpeg must decode");
        assert_eq!((rgb.width(), rgb.height()), (537, 529));
        let digest = sha2::Sha256::digest(rgb.as_raw());
        assert_eq!(
            format!("{digest:x}"),
            "45bb7423193c00359695e1c967676d86e82bd3f5d55aa1679a85df6c74a9cf55",
            "turbo-decoded RGB buffer must match PIL/libjpeg-turbo exactly"
        );
    }

    #[test]
    fn doge_header_probe_matches_decode_without_pixels() {
        let bytes = doge_bytes();
        let (w, h) = probe_dimensions(&bytes).expect("doge.jpeg header must probe");
        assert_eq!((w, h), (537, 529));
        let dyn_img = decode_dynamic(&bytes).expect("doge.jpeg must decode");
        assert!(matches!(dyn_img, image::DynamicImage::ImageRgb8(_)));
        assert_eq!((dyn_img.width(), dyn_img.height()), (537, 529));
    }

    #[test]
    fn png_keeps_dynamic_variant_and_garbage_fails_closed() {
        // RGBA PNG must stay a DynamicImage::ImageRgba8 so carriers that
        // composite alpha (dots.ocr) see the same variant as before.
        let rgba = image::RgbaImage::from_fn(3, 2, |x, y| {
            image::Rgba([(x * 40) as u8, (y * 100) as u8, 7, 128])
        });
        let mut png = Vec::new();
        image::DynamicImage::ImageRgba8(rgba)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let decoded = decode_dynamic(&png).expect("PNG must decode");
        assert!(
            matches!(decoded, image::DynamicImage::ImageRgba8(_)),
            "PNG must keep its DynamicImage variant, got {:?}",
            decoded.color()
        );
        let rgb = decode_rgb8(&png).expect("PNG must decode to RGB8");
        assert_eq!((rgb.width(), rgb.height()), (3, 2));

        let (w, h) = probe_dimensions(&png).expect("PNG header must probe");
        assert_eq!((w, h), (3, 2));

        for bad in [&[][..], &[0xDE, 0xAD, 0xBE, 0xEF][..], &b"/etc/passwd"[..]] {
            let msg = decode_rgb8(bad).unwrap_err().to_lowercase();
            assert!(
                msg.contains("failed to decode image") || msg.contains("unsupported"),
                "garbage must fail closed, got: {msg}"
            );
        }
        assert!(decode_rgb8(&[]).is_err());
        assert!(probe_dimensions(&[]).is_err());
    }
}
