// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-request image embeddings and three-axis positions. Image requests use
//! cold, eager AR and never publish a text-token-only prefix-cache entry.

use super::{BundleError, Qwen4Bundle};
use crate::gpu_forward::Qwen4GpuForwardError;
use crate::vision::{IMAGE_TOKEN_ID, VISION_END_TOKEN_ID, VISION_START_TOKEN_ID};
use hipfire_arch_qwen35_vl::mrope::{build_mrope_positions, MropePositions};
use rdna_compute::qsa_mrope::QsaMrope;
use rdna_compute::{Gpu, GpuTensor};

pub use hipfire_arch_qwen35_vl::mrope::ImageSpan;

pub(crate) struct ImageContext {
    prompt: Vec<u32>,
    span: ImageSpan,
    positions: GpuTensor,
    visual: GpuTensor,
    delta: i32,
}

fn invalid(message: impl Into<String>) -> BundleError {
    BundleError::Forward(format!("qwen4 image: {}", message.into()))
}

fn positions_for_image(
    prompt: &[u32],
    span: ImageSpan,
    merge: usize,
) -> Result<MropePositions, BundleError> {
    let end = span
        .start
        .checked_add(span.len)
        .ok_or_else(|| invalid("image span overflows"))?;
    let grid_tokens = span
        .grid_h
        .checked_mul(span.grid_w)
        .and_then(|n| merge.checked_mul(merge).filter(|&m| m > 0).map(|m| n / m));
    if prompt.len() > i32::MAX as usize / 12
        || span.start == 0
        || span.len == 0
        || end >= prompt.len()
        || merge == 0
        || span.grid_h == 0
        || span.grid_w == 0
        || span.grid_h % merge != 0
        || span.grid_w % merge != 0
        || grid_tokens != Some(span.len)
    {
        return Err(invalid("invalid image span or merged grid"));
    }
    if prompt[span.start - 1] != VISION_START_TOKEN_ID
        || prompt[end] != VISION_END_TOKEN_ID
        || prompt[span.start..end]
            .iter()
            .any(|&id| id != IMAGE_TOKEN_ID)
        || prompt[..span.start].contains(&IMAGE_TOKEN_ID)
        || prompt[end..].contains(&IMAGE_TOKEN_ID)
        || prompt.contains(&248057)
    {
        return Err(invalid(
            "expected exactly one framed image and no video tokens",
        ));
    }
    let positions = build_mrope_positions(prompt.len(), &[span], merge);
    if positions.positions.len() != prompt.len() {
        return Err(invalid("image position count differs from prompt tokens"));
    }
    Ok(positions)
}

impl ImageContext {
    pub(crate) fn mrope(&self) -> QsaMrope<'_> {
        QsaMrope {
            positions: &self.positions,
            prompt_rows: self.prompt.len(),
            delta: self.delta,
            section: [11, 11, 10],
        }
    }

    pub(crate) fn splice(
        &self,
        gpu: &mut Gpu,
        embeddings: &GpuTensor,
        position: usize,
        rows: usize,
        dim: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        let start = position.max(self.span.start);
        let end = (position + rows).min(self.span.start + self.span.len);
        if start < end {
            gpu.memcpy_dtod_at_auto(
                &embeddings.buf,
                (start - position) * dim * 4,
                &self.visual.buf,
                (start - self.span.start) * dim * 4,
                (end - start) * dim * 4,
            )?;
        }
        Ok(())
    }

    pub(crate) fn free(self, gpu: &mut Gpu) -> Result<(), hip_bridge::HipError> {
        let first = gpu.free_tensor(self.positions);
        let second = gpu.free_tensor(self.visual);
        first.and(second)
    }
}

impl Qwen4Bundle {
    /// Image state cannot be promoted to text-only lanes or prefix snapshots.
    pub fn image_context_active(&self) -> bool {
        self.image_context.is_some()
    }

    /// Attach one image to a freshly reset singleton. Neither cache admission
    /// nor state reset is implicit: a caller must cold-reset before preparing.
    pub fn prepare_image_context(
        &mut self,
        gpu: &mut Gpu,
        prompt: &[u32],
        visual_rows: &[f32],
        span: ImageSpan,
    ) -> Result<(), BundleError> {
        if self.state.position != 0 || self.image_context.is_some() {
            return Err(invalid("reset the request before preparing an image"));
        }
        if self.ep.is_some() || self.pp.is_some() {
            return Err(invalid("image requests require the singleton GPU route"));
        }
        if gpu.graphs.capture_mode || gpu.replay.is_recording() {
            return Err(invalid(
                "image requests require eager HIP outside graph capture",
            ));
        }
        let (vision, _) = self
            .vision
            .as_ref()
            .ok_or_else(|| invalid("vision sidecar is not loaded"))?;
        if prompt.is_empty()
            || prompt.len() > self.state.max_seq_len
            || self.config.hidden_size != vision.out_hidden_size
            || span.len.checked_mul(self.config.hidden_size) != Some(visual_rows.len())
            || visual_rows.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "visual embedding width, values or context length is invalid",
            ));
        }
        let built = positions_for_image(prompt, span, vision.spatial_merge_size)?;
        let bytes: Vec<u8> = built
            .positions
            .iter()
            .flat_map(|p| p.iter())
            .flat_map(|p| p.to_ne_bytes())
            .collect();
        let positions = gpu
            .upload_raw(&bytes, &[bytes.len()])
            .map_err(BundleError::Hip)?;
        let visual = match gpu.upload_f32(visual_rows, &[span.len, self.config.hidden_size]) {
            Ok(visual) => visual,
            Err(error) => {
                let _ = gpu.free_tensor(positions);
                return Err(BundleError::Hip(error));
            }
        };
        self.image_context = Some(ImageContext {
            prompt: prompt.to_vec(),
            span,
            positions,
            visual,
            delta: built.rope_delta,
        });
        // Reuse the existing per-request route override (also used by DFlash2
        // capture). A following reset starts a fresh NPU request normally.
        if let Some(forward) = self.execution.as_mut() {
            forward.npu_trunk_request_gpu_only("image embeddings and three-axis positions");
        }
        gpu.replay.set_forward_eligible(false);
        Ok(())
    }

    /// Prefill only the image-conditioned cold prompt, with final-row logits.
    /// No text prefix/session cache is consulted or populated.
    pub fn prefill_image_final(
        &mut self,
        gpu: &mut Gpu,
        prompt: &[u32],
        logits: &GpuTensor,
    ) -> Result<(), BundleError> {
        let image = self
            .image_context
            .as_ref()
            .ok_or_else(|| invalid("image context is not prepared"))?;
        if self.state.position != 0 || image.prompt != prompt {
            return Err(invalid("image prefill requires its exact cold prompt"));
        }
        self.forward_chunk_final_incumbent(gpu, prompt, logits, None)
    }

    pub(crate) fn clear_image_context(&mut self, gpu: &mut Gpu) -> Result<(), BundleError> {
        // free_tensor returns allocations to a reusable pool. A failed forward
        // may still have in-flight readers; retain ownership if the fence fails.
        if self.image_context.is_some() {
            gpu.hip.device_synchronize().map_err(BundleError::Hip)?;
        }
        if let Some(image) = self.image_context.take() {
            image.free(gpu).map_err(BundleError::Hip)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Vec<u32>, ImageSpan) {
        let mut prompt = vec![1, VISION_START_TOKEN_ID];
        prompt.extend([IMAGE_TOKEN_ID; 6]);
        prompt.extend([VISION_END_TOKEN_ID, 2, 3]);
        (
            prompt,
            ImageSpan {
                start: 2,
                len: 6,
                grid_h: 4,
                grid_w: 6,
            },
        )
    }
    #[test]
    fn rectangular_image_positions_and_decode_cursor() {
        let (prompt, span) = fixture();
        let p = positions_for_image(&prompt, span, 2).unwrap();
        assert_eq!(
            &p.positions[2..8],
            &[
                [2, 2, 2],
                [2, 2, 3],
                [2, 2, 4],
                [2, 3, 2],
                [2, 3, 3],
                [2, 3, 4]
            ]
        );
        assert_eq!(&p.positions[8..], &[[5; 3], [6; 3], [7; 3]]);
        assert_eq!(p.rope_delta, -3);
        assert_eq!(prompt.len() as i32 + p.rope_delta, 8);
    }
    #[test]
    fn malformed_image_spans_fail_before_allocation() {
        let (prompt, span) = fixture();
        for bad in [
            ImageSpan { len: 5, ..span },
            ImageSpan { grid_h: 3, ..span },
            ImageSpan {
                start: usize::MAX,
                ..span
            },
            ImageSpan { start: 0, ..span },
        ] {
            assert!(positions_for_image(&prompt, bad, 2).is_err());
        }
        assert!(positions_for_image(&prompt, span, 0).is_err());
        let mut bad = prompt.clone();
        bad[4] = 7;
        assert!(positions_for_image(&bad, span, 2).is_err());
        let mut bad = prompt;
        bad[0] = 248057;
        assert!(positions_for_image(&bad, span, 2).is_err());
    }
    #[test]
    fn pooled_key_position_uses_original_block_start() {
        let (prompt, span) = fixture();
        let p = positions_for_image(&prompt, span, 2).unwrap();
        // A four-token group that begins within the picture uses its three
        // grid coordinates, even when later rows straddle the image boundary.
        assert_eq!(p.positions[4], [2, 2, 4]);
        assert_ne!(p.positions[4], [4; 3]);
        assert_eq!(p.positions[8], [5; 3]);
    }
}
