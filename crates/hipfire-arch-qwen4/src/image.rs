// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next shares Qwen image equations and patch layout, while owning the
//! bounded patch grid priced by its resident tower scratch reserve. The
//! existing Qwen35-VL path-input contract is intentionally broader.

use std::path::Path;

pub use hipfire_arch_qwen35_vl::image::{extract_patches, smart_resize};

fn admit_resized_grid(
    height: usize,
    width: usize,
    patch_size: usize,
    spatial_merge_size: usize,
) -> Result<(), String> {
    let (height, width) = hipfire_arch_qwen35_vl::image::resized_dimensions(
        height,
        width,
        patch_size,
        spatial_merge_size,
    )?;
    let patches = (height / patch_size)
        .checked_mul(width / patch_size)
        .ok_or("qwen4 vision: image patch count overflow")?;
    if patches > crate::vision::MAX_IMAGE_PATCHES {
        return Err(format!(
            "qwen4 vision: resized image patch count {patches} exceeds the reserved limit {}; reduce the aspect ratio",
            crate::vision::MAX_IMAGE_PATCHES
        ));
    }
    Ok(())
}

/// Flash-Next uses identical validation for file and byte image inputs.
pub fn load_and_preprocess(
    path: &Path,
    patch_size: usize,
    spatial_merge_size: usize,
) -> Result<(Vec<f32>, usize, usize), String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("failed to open image {}: {error}", path.display()))?;
    load_and_preprocess_from_bytes(&bytes, patch_size, spatial_merge_size)
}

/// Check the actual rounded patch grid before decoding or allocating pixels.
/// The shared byte decoder also retains its format-header bomb guard.
pub fn load_and_preprocess_from_bytes(
    data: &[u8],
    patch_size: usize,
    spatial_merge_size: usize,
) -> Result<(Vec<f32>, usize, usize), String> {
    let (width, height) = hipfire_runtime::imagedec::probe_dimensions(data)?;
    admit_resized_grid(
        height as usize,
        width as usize,
        patch_size,
        spatial_merge_size,
    )?;
    hipfire_arch_qwen35_vl::image::load_and_preprocess_from_bytes(
        data,
        patch_size,
        spatial_merge_size,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn red_png(width: u32, height: u32) -> Vec<u8> {
        let image = image::RgbImage::from_pixel(width, height, image::Rgb([255, 0, 0]));
        let mut data = std::io::Cursor::new(Vec::new());
        image.write_to(&mut data, image::ImageFormat::Png).unwrap();
        data.into_inner()
    }

    #[test]
    fn ordinary_image_matches_shared_processor_pixels_and_grid() {
        let bytes = red_png(32, 32);
        let actual = load_and_preprocess_from_bytes(&bytes, 16, 2).unwrap();
        let shared =
            hipfire_arch_qwen35_vl::image::load_and_preprocess_from_bytes(&bytes, 16, 2).unwrap();
        assert_eq!(actual, shared);
        assert_eq!((actual.1, actual.2), (256, 256));
    }

    #[test]
    fn reserved_patch_budget_refuses_valid_extreme_images() {
        for (width, height) in [(1, 1_048_576), (1_048_576, 1)] {
            let bytes = red_png(width, height);
            let error = load_and_preprocess_from_bytes(&bytes, 16, 2).unwrap_err();
            assert!(error.contains("patch count 32768"), "{error}");
            assert!(error.contains("reserved limit 16128"), "{error}");
        }
    }

    #[test]
    fn path_and_bytes_keep_flash_next_header_guard() {
        let bytes = red_png(4097, 4097);
        let byte_error = load_and_preprocess_from_bytes(&bytes, 16, 2).unwrap_err();
        assert!(byte_error.contains("exceed maximum"), "{byte_error}");
        let dir = std::env::temp_dir().join(format!(
            "hipfire-qwen4-image-cap-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("large.png");
        std::fs::write(&path, &bytes).unwrap();
        let result = load_and_preprocess(&path, 16, 2);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(result.unwrap_err(), byte_error);
    }

    #[test]
    fn default_resize_can_reach_the_full_reserved_patch_budget() {
        // A narrow image enters the minimum-area branch before its short
        // axis is clamped to 32. The default 2MP target is therefore not a
        // bound on the final rounded grid: the full reserve remains usable.
        let dimensions =
            hipfire_arch_qwen35_vl::image::resized_dimensions(1, 254_016, 16, 2).unwrap();
        assert_eq!(dimensions, (32, 129_024));
        assert_eq!((dimensions.0 / 16) * (dimensions.1 / 16), 16_128);
        assert!(admit_resized_grid(1, 254_016, 16, 2).is_ok());
    }

    #[test]
    fn invalid_patch_geometry_cannot_bypass_resource_admission() {
        for (patch, merge) in [(0, 2), (16, 0), (usize::MAX, 2)] {
            assert!(admit_resized_grid(32, 32, patch, merge).is_err());
        }
    }
}
