// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The video leg's frame frontend: HF `EmbeddingGemma2VideoProcessor`'s
//! sampling rule plus the frame pipeline, which is the IMAGE pipeline with
//! the video processor's own patch budget.
//!
//! What HF does with one video (`transformers 5.19.0`):
//!
//! 1. **Sample** (`video_processing_embedding_gemma2.py:316-360`,
//!    `sample_frames`). A decoded frame list carries no capture rate, so the
//!    `fps`-based pass is skipped exactly as HF skips it — `metadata.fps` is
//!    `None` for a frame list, HF logs its warning and sets `fps = None` —
//!    leaving `indices = arange(total_num_frames)`. Then the `max_frames`
//!    budget (`overflow_strategy`): ≤ `max_frames` frames are all kept, more
//!    are reduced with `np.linspace(0, total-1, max_frames, dtype=int)`
//!    (`"uniform"`) or cut to the first `max_frames` (`"truncate"`).
//!    `do_sample_frames = false` means HF never calls the sampler at all, so
//!    every supplied frame is used.
//! 2. **Preprocess one frame at a time**, through the same resize/rescale the
//!    image leg runs (`get_aspect_ratio_preserving_size` +
//!    `tvF.resize(..., BICUBIC, antialias=True)` on the uint8 frame, then
//!    `× 1/255`; `do_normalize` is false on this checkpoint and mean 0 /
//!    std 1 would make it an identity anyway) — but with the VIDEO
//!    processor's `max_soft_tokens` (140 here, so 1 260 patches) instead of
//!    the image processor's 280. HF resizes the whole `(T, C, H, W)` clip at
//!    once; every frame of a decoded clip has the same shape, so the target —
//!    and the pixel math, per frame — is the same as resizing each frame.
//!    Frames that do NOT share a shape cannot be represented at all (HF
//!    stacks them with `np.stack`), so they are refused by name.
//! 3. **Run the tower over every frame's real patches** as independent
//!    sequences: `(total_frames, max_patches, patch_pixels)` with
//!    `attention_mask = ~padding_positions` (`modeling_gemma4.py:2003-2062`),
//!    i.e. one batch element per frame — frames never attend each other.
//!    `VisionTower::encode_image` per frame is that same isolation (see
//!    [`crate::vision`]), so no new tower code exists here.
//!
//! The container rule mirrors the audio leg's WAV-only rule: a frame that is
//! not a PNG/JPEG is refused BY NAME with the frame extraction to run, never
//! silently decoded as something else.

use crate::config::{VideoFrontendConfig, VisionConfig};
use crate::image::{process_image, ProcessedImage};

/// One frame's bytes must be one of the two lossless/simple still-image
/// containers this leg accepts. Anything else (a container, a raw pixel
/// dump, a truncated file) is refused with the `ffmpeg` extraction to run —
/// the same posture as the audio leg's RIFF/WAVE requirement.
fn is_frame_image(bytes: &[u8]) -> bool {
    bytes.starts_with(b"\x89PNG\r\n\x1a\n") || bytes.starts_with(b"\xff\xd8\xff")
}

/// The container a non-image frame looks like, for the refusal text.
fn container_name(bytes: &[u8]) -> &'static str {
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") {
        if &bytes[8..12] == b"WEBP" {
            "a WebP image (this leg takes PNG/JPEG frames)"
        } else {
            "a RIFF container (WAV/AVI)"
        }
    } else if bytes.len() > 11 && &bytes[4..8] == b"ftyp" {
        "an ISO base media container (MP4/MOV)"
    } else if bytes.starts_with(b"\x1aE\xdf\xa3") {
        "a Matroska/WebM container"
    } else if bytes.starts_with(b"OggS") {
        "an OGG stream"
    } else if bytes.starts_with(b"GIF8") {
        "a GIF image (this leg takes PNG/JPEG frames)"
    } else if bytes.starts_with(b"#!") || bytes.starts_with(b"<") {
        "not an image at all"
    } else {
        "not a PNG/JPEG image"
    }
}

/// HF `EmbeddingGemma2VideoProcessor.sample_frames` for a decoded frame list:
/// the indices the clip's `max_frames` budget selects, in order.
///
/// `fps` is deliberately not consulted — a decoded frame list carries no
/// capture rate, which is the case where HF itself skips the rate pass
/// (`metadata.fps is None` → `fps = None`, warning) and keeps every frame as
/// a candidate. `do_sample_frames = false` skips the budget entirely (HF
/// never calls the sampler).
pub fn sample_indices(total: usize, cfg: &VideoFrontendConfig) -> Vec<usize> {
    let all = || (0..total).collect::<Vec<usize>>();
    if !cfg.do_sample_frames || total <= cfg.max_frames {
        return all();
    }
    let num = cfg.max_frames;
    match cfg.overflow {
        crate::config::OverflowStrategy::Truncate => (0..num).collect(),
        crate::config::OverflowStrategy::Uniform => {
            // `np.linspace(0, total-1, num, dtype=int)`: the values are
            // `arange(num) * (delta/div)` computed in f64 and then cast with a
            // truncating conversion, and the LAST entry is pinned to `total-1`
            // instead of trusting the accumulated step. `num == 1` is numpy's
            // own special case and returns `[start]` = index 0.
            let last = total - 1;
            if num == 1 {
                return vec![0];
            }
            let step = last as f64 / (num - 1) as f64;
            (0..num)
                .map(|i| {
                    if i + 1 == num {
                        last
                    } else {
                        ((i as f64) * step) as usize
                    }
                })
                .collect()
        }
    }
}

/// Decode + sample + preprocess one clip's frames: the frames the tower must
/// run, in the order their placeholder blocks appear in the prompt.
///
/// `index` is the clip's position in the request, used to name it in every
/// refusal. Frames are validated (container, shared shape) BEFORE the sample,
/// so a request that names a container can never half-succeed; only the
/// SELECTED frames are decoded (the resize is the expensive half).
pub fn process_video(
    frames: &[Vec<u8>],
    index: usize,
    cfg: &VideoFrontendConfig,
    vcfg: &VisionConfig,
) -> Result<Vec<ProcessedImage>, String> {
    if frames.is_empty() {
        return Err(format!(
            "video {index} carries no frames — send a decoded clip as a list of frames \
             (PNG/JPEG, in display order), e.g. `ffmpeg -i clip.mp4 -vf fps=1 \
             frame_%04d.png`"
        ));
    }
    let extraction = "this leg embeds decoded FRAMES (PNG/JPEG), not containers — extract \
                      them first, e.g. `ffmpeg -i clip.mp4 -vf fps=1 frame_%04d.png`";
    let mut dims: Vec<(u32, u32)> = Vec::with_capacity(frames.len());
    for (i, bytes) in frames.iter().enumerate() {
        if !is_frame_image(bytes) {
            return Err(format!(
                "video {index} frame {i} is {} — {extraction}",
                container_name(bytes)
            ));
        }
        let (w, h) = hipfire_runtime::imagedec::probe_dimensions(bytes)
            .map_err(|e| format!("video {index} frame {i}: {e} — {extraction}"))?;
        if w == 0 || h == 0 {
            return Err(format!("video {index} frame {i} has a zero side ({w}x{h})"));
        }
        dims.push((w, h));
    }
    // HF stacks a clip's frames into one `(T, H, W, C)` array
    // (`video_utils.convert_pil_frames_to_video`), so ragged frames have no
    // representation at all — and the sampler's single per-video soft-token
    // count assumes one shared frame geometry.
    let (w0, h0) = dims[0];
    for (i, &(w, h)) in dims.iter().enumerate() {
        if (w, h) != (w0, h0) {
            return Err(format!(
                "video {index} frame {i} is {w}x{h} but frame 0 is {w0}x{h0} — a decoded \
                 clip's frames share one size (HF stacks them into a single array); pass \
                 frames from one clip"
            ));
        }
    }

    let indices = sample_indices(frames.len(), cfg);
    let mut out = Vec::with_capacity(indices.len());
    for &i in &indices {
        out.push(
            process_image(&frames[i], vcfg).map_err(|e| format!("video {index} frame {i}: {e}"))?,
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> VideoFrontendConfig {
        VideoFrontendConfig::default()
    }

    /// The checkpoint's video processor: 32 frames, uniform overflow.
    #[test]
    fn a_short_clip_keeps_every_frame() {
        let c = cfg();
        for n in 1..=32usize {
            assert_eq!(sample_indices(n, &c), (0..n).collect::<Vec<_>>(), "n={n}");
        }
    }

    /// `np.linspace(0, total-1, 32, dtype=int)` — the values the float32
    /// reference itself sampled (verified against numpy: 77 frames →
    /// 0,2,4,7,9,12,…,73,76).
    #[test]
    fn a_long_clip_is_sampled_with_hfs_linspace() {
        let c = cfg();
        assert_eq!(
            sample_indices(77, &c),
            vec![
                0, 2, 4, 7, 9, 12, 14, 17, 19, 22, 24, 26, 29, 31, 34, 36, 39, 41, 44, 46, 49, 51,
                53, 56, 58, 61, 63, 66, 68, 71, 73, 76
            ]
        );
        // The interior values are the plain stride `i * (total-1)/(max_frames-1)`
        // truncated toward zero; the last is pinned to `total-1`. Both edges
        // are pinned against numpy (`np.linspace(0, total-1, 32, dtype=int)`).
        assert_eq!(
            sample_indices(33, &c),
            vec![
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 32
            ]
        );
        assert_eq!(
            sample_indices(63, &c),
            vec![
                0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42,
                44, 46, 48, 50, 52, 54, 56, 58, 60, 62
            ]
        );
        // One past a whole stride: the cast truncates, only the last entry is
        // nudged onto the final frame.
        assert_eq!(sample_indices(64, &c).last(), Some(&63));
        assert_eq!(sample_indices(1000, &c).len(), 32);
        assert_eq!(&sample_indices(1000, &c)[..3], &[0, 32, 64]);
        assert_eq!(sample_indices(1000, &c).last(), Some(&999));
    }

    #[test]
    fn truncate_and_no_sampling_are_the_other_two_branches() {
        let mut c = cfg();
        c.overflow = crate::config::OverflowStrategy::Truncate;
        assert_eq!(sample_indices(77, &c), (0..32).collect::<Vec<_>>());
        let mut c = cfg();
        c.do_sample_frames = false;
        assert_eq!(sample_indices(77, &c), (0..77).collect::<Vec<_>>());
    }

    #[test]
    fn a_single_frame_budget_takes_the_first_frame() {
        let mut c = cfg();
        c.max_frames = 1;
        assert_eq!(sample_indices(9, &c), vec![0]);
        assert_eq!(sample_indices(1, &c), vec![0]);
    }

    /// The container rule: bytes that are not a PNG/JPEG are refused by name
    /// with the extraction to run, before any decode.
    #[test]
    fn a_container_is_refused_with_the_extraction_to_run() {
        let vcfg = VisionConfig {
            hidden_size: 768,
            num_layers: 1,
            intermediate_size: 3072,
            n_heads: 12,
            n_kv_heads: 12,
            head_dim: 64,
            rms_norm_eps: 1e-6,
            patch_size: 16,
            pooling_kernel_size: 3,
            position_embedding_size: 10240,
            rope_theta: 100.0,
            max_soft_tokens: 140,
        };
        let mp4 = {
            let mut v = b"\x00\x00\x00\x18ftypisom".to_vec();
            v.extend_from_slice(&[0u8; 16]);
            v
        };
        let err = process_video(&[mp4], 0, &cfg(), &vcfg).unwrap_err();
        assert!(err.contains("ISO base media container"), "{err}");
        assert!(err.contains("ffmpeg"), "{err}");
        assert!(err.contains("video 0 frame 0"), "{err}");

        let err = process_video(&[], 3, &cfg(), &vcfg).unwrap_err();
        assert!(err.contains("video 3 carries no frames"), "{err}");

        // A PNG signature that is not a decodable PNG reaches the decoder's
        // own refusal, still naming the clip and frame.
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&[0u8; 8]);
        let err = process_video(&[png], 1, &cfg(), &vcfg).unwrap_err();
        assert!(err.contains("video 1 frame 0"), "{err}");
    }

    #[test]
    fn a_declared_frame_image_is_not_a_container() {
        assert!(is_frame_image(b"\x89PNG\r\n\x1a\n"));
        assert!(is_frame_image(b"\xff\xd8\xff\xe0"));
        assert!(!is_frame_image(b"RIFF"));
        assert!(!is_frame_image(b"\x1aE\xdf\xa3"));
        assert_eq!(
            container_name(b"GIF89a"),
            "a GIF image (this leg takes PNG/JPEG frames)"
        );
        assert_eq!(container_name(b"nope"), "not a PNG/JPEG image");
    }

    /// The checkpoint's vision geometry with the VIDEO processor's budget: the
    /// per-frame soft-token counts must be the ones the float32 reference
    /// emitted for the same frame shapes (140 = the cap at 240×336, 121 at
    /// 192×192 and 96×96, 126 at 480×144, 135 at 168×288).
    fn video_vision() -> VisionConfig {
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
            max_soft_tokens: 140,
        }
    }

    fn png(h: u32, w: u32, tint: u8) -> Vec<u8> {
        let mut img = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.put_pixel(x, y, image::Rgb([(x % 251) as u8, (y % 251) as u8, tint]));
            }
        }
        let mut buf = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .expect("encode png");
        buf.into_inner()
    }

    #[test]
    fn frames_are_resized_against_the_video_processors_patch_budget() {
        let c = cfg();
        let vcfg = video_vision();
        assert_eq!(c.max_patches(), 1260);
        for (h, w, want) in [
            (240u32, 336u32, 140usize), // at the 140 cap
            (192, 192, 121),
            (96, 96, 121),
            (480, 144, 126),
            (168, 288, 135),
        ] {
            let frames = vec![png(h, w, 7), png(h, w, 9)];
            let out = process_video(&frames, 0, &c, &vcfg).expect("process clip");
            assert_eq!(out.len(), 2, "{w}x{h}");
            for f in &out {
                assert_eq!(f.n_tokens, want, "{w}x{h}");
                assert_eq!(f.grid_h * f.grid_w, f.n_patches);
                assert!(f.n_patches <= c.max_patches());
            }
        }
    }

    #[test]
    fn the_sampler_decides_which_frames_run() {
        let c = cfg();
        let vcfg = video_vision();
        // 40 frames of one shape: the 32-frame budget strides over them.
        let mut frames: Vec<Vec<u8>> = (0..40).map(|i| png(96, 96, (i * 6) as u8)).collect();
        let out = process_video(&frames, 0, &c, &vcfg).expect("process long clip");
        assert_eq!(out.len(), 32);
        // The container check runs over EVERY supplied frame, not just the
        // sampled ones — frame 4 is one the sampler drops (linspace selects
        // 0,1,2,3,5,…), and a clip is still a clip.
        assert!(!sample_indices(40, &c).contains(&4));
        frames[4] = b"\x00\x00\x00\x18ftypmp42".to_vec();
        let err = process_video(&frames, 0, &c, &vcfg).unwrap_err();
        assert!(err.contains("video 0 frame 4"), "{err}");
        assert!(err.contains("ffmpeg"), "{err}");
    }

    #[test]
    fn a_clip_whose_frames_differ_in_size_is_refused() {
        let frames = vec![png(96, 96, 1), png(96, 48, 2)];
        let err = process_video(&frames, 2, &cfg(), &video_vision()).unwrap_err();
        assert!(err.contains("video 2 frame 1 is 48x96"), "{err}");
        assert!(err.contains("share one size"), "{err}");
    }
}
