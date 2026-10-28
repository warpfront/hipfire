// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next single-image generation. Image turns are cold AR requests:
//! their image identity is never confused with a text-only prefix cache key.
//! The ordinary AR producer owns sampling, typed emission and cancellation.
//! Flash-Next tower admission is resident-only: `auto` and `on` use the same
//! resident backend as `resident`; `on` requires a tower at load.

use base64::Engine;
use hipfire_arch_qwen35_vl::mrope::ImageSpan;
use hipfire_arch_qwen4::image;
use hipfire_loader::LoadedModel;
use rdna_compute::Gpu;

use crate::vision::{GenerateVLParams, ImageSource};

/// Standard AR request controls, kept separate from the legacy VL contract
/// so other architecture producers retain their existing API.
pub struct Qwen4VisionControls<'a> {
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub stop: &'a [String],
    pub enable_thinking: bool,
    pub reasoning_effort: Option<&'a str>,
}

/// Expand exactly one framed image placeholder. Refuse ambiguous markers
/// instead of conditioning only part of a prompt on an image.
fn expand_image_placeholder(
    tokens: &[u32],
    image_id: u32,
    start_id: u32,
    end_id: u32,
    visual_tokens: usize,
) -> Result<(Vec<u32>, usize), String> {
    if visual_tokens == 0 {
        return Err("qwen4 vision: image has no visual tokens".into());
    }
    let mut markers = tokens.iter().enumerate().filter(|(_, id)| **id == image_id);
    let start = markers
        .next()
        .map(|(i, _)| i)
        .ok_or_else(|| "qwen4 vision: prompt has no image placeholder".to_string())?;
    if markers.next().is_some()
        || start == 0
        || tokens.get(start - 1) != Some(&start_id)
        || tokens.get(start + 1) != Some(&end_id)
    {
        return Err("qwen4 vision: expected one framed image placeholder".into());
    }
    let capacity = tokens
        .len()
        .checked_add(visual_tokens - 1)
        .ok_or_else(|| "qwen4 vision: prompt length overflow".to_string())?;
    let mut expanded = Vec::with_capacity(capacity);
    expanded.extend_from_slice(&tokens[..start]);
    expanded.extend(std::iter::repeat_n(image_id, visual_tokens));
    expanded.extend_from_slice(&tokens[start + 1..]);
    Ok((expanded, start))
}

fn refuse(stdout: &mut std::io::Stdout, id: &str, message: &str) {
    crate::ar::emit_active_route_error(stdout, Some(id), message, "validation", false, false);
}

pub fn generate_qwen4_vl(
    model: &mut LoadedModel,
    gpu: &mut Gpu,
    stdout: &mut std::io::Stdout,
    params: &GenerateVLParams<'_>,
    controls: &Qwen4VisionControls<'_>,
) {
    let id = params.id;
    hipfire_runtime::llama::reset_cpu_sampler_rng(params.seed);
    let Some(bundle) = model.qwen4() else {
        refuse(stdout, id, "qwen4 vision requires a Flash-Next model");
        return;
    };
    let Some((vision, _)) = bundle.vision.as_ref() else {
        refuse(
            stdout,
            id,
            "qwen4 vision tower is not loaded; provide its sidecar and reload with vision_mode=resident",
        );
        return;
    };
    let vision = vision.clone();
    let vocab_size = bundle.config.vocab_size;
    let eos_token = bundle.config.eos_token_id;
    if params.max_tokens == 0 {
        refuse(stdout, id, "max_tokens must be > 0");
        return;
    }
    // Read file and base64 inputs through the same dimension-checked decoder.
    let bytes = match &params.image_source {
        ImageSource::Path(path) => std::fs::read(path).map_err(|e| e.to_string()),
        ImageSource::Base64(encoded) => base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|e| format!("invalid image base64: {e}")),
    };
    let prepared = bytes.and_then(|bytes| {
        image::load_and_preprocess_from_bytes(&bytes, vision.patch_size, vision.spatial_merge_size)
    });
    let (pixels, height, width) = match prepared {
        Ok(value) => value,
        Err(error) => {
            refuse(stdout, id, &error);
            return;
        }
    };
    let grid_h = height / vision.patch_size;
    let grid_w = width / vision.patch_size;
    let visual_tokens = (grid_h / vision.spatial_merge_size) * (grid_w / vision.spatial_merge_size);
    let framed_prompt = format!(
        "<|vision_start|><|image_pad|><|vision_end|>{}",
        params.prompt
    );
    let (cold, started_in_think) = match crate::qwen::render_qwen4_prompt_cold(
        model,
        crate::qwen::Qwen4PromptRoute::Ar,
        params.system_prompt,
        &framed_prompt,
        params.assistant_prefix,
        None,
        None,
        controls.enable_thinking,
        controls.reasoning_effort,
    ) {
        Ok(value) => value,
        Err((error, _)) => {
            refuse(stdout, id, &error);
            return;
        }
    };
    let (prompt, image_start) = match expand_image_placeholder(
        &cold,
        hipfire_arch_qwen4::vision::IMAGE_TOKEN_ID,
        hipfire_arch_qwen4::vision::VISION_START_TOKEN_ID,
        hipfire_arch_qwen4::vision::VISION_END_TOKEN_ID,
        visual_tokens,
    ) {
        Ok(value) => value,
        Err(error) => {
            refuse(stdout, id, &error);
            return;
        }
    };
    let max_tokens =
        crate::common::fit_max_tokens(params.max_tokens, prompt.len() + 1, model.max_seq);
    if prompt
        .len()
        .checked_add(max_tokens)
        .and_then(|n| n.checked_add(1))
        .is_none_or(|n| n > model.max_seq)
    {
        refuse(
            stdout,
            id,
            "qwen4 image and text exceed the configured context window",
        );
        return;
    }
    let reset = crate::common::production_fail_closed_rollback(model, gpu, None, None);
    if !reset.rolled_back {
        crate::common::emit_fail_closed_error(
            stdout,
            Some(id),
            "qwen4 vision reset failed",
            "gpu",
            false,
            &reset,
        );
        return;
    }
    if hipfire_engine::terminal::check_abort(id) {
        crate::common::emit_spec_cancel_after_rollback(stdout, id, 0, &reset);
        return;
    }
    let span = ImageSpan {
        start: image_start,
        len: visual_tokens,
        grid_h,
        grid_w,
    };
    crate::ar::generate_ar_with_forward(
        model,
        gpu,
        stdout,
        id,
        crate::ar::GenerationRoute::Qwen4Ar,
        &prompt,
        vocab_size,
        eos_token,
        params.temp,
        params.top_p,
        controls.top_k,
        controls.min_p,
        max_tokens,
        params.repeat_penalty,
        params.repeat_window,
        controls.presence_penalty,
        controls.frequency_penalty,
        params.max_think_tokens,
        started_in_think,
        controls.stop,
        false,
        0,
        false,
        |model, gpu, prompt, logits| {
            let bundle = model.qwen4_mut().ok_or("qwen4 vision bundle disappeared")?;
            let (config, weights) = bundle
                .vision
                .as_ref()
                .ok_or("qwen4 vision tower disappeared")?;
            let rows =
                hipfire_arch_qwen4::vision::encode(gpu, weights, config, &pixels, height, width)
                    .map_err(|e| e.to_string())?;
            bundle
                .prepare_image_context(gpu, prompt, &rows, span)
                .map_err(|e| e.to_string())?;
            bundle
                .prefill_image_final(gpu, prompt, logits)
                .map_err(|e| e.to_string())
        },
        |model, gpu, token, logits| {
            model
                .qwen4_mut()
                .ok_or("qwen4 vision bundle disappeared")?
                .forward_token_or_argmax(gpu, token, logits)
                .map_err(|e| e.to_string())
        },
        // Images are not keyed by the text session cache. Never publish their
        // recurrent state/checkpoints under an image-placeholder token key.
        |_| {},
    );
    // A subsequent text turn must not inherit image mRoPE or recurrent state.
    // The generic producer has already emitted exactly one terminal here.
    let reset = crate::common::production_fail_closed_rollback(model, gpu, None, None);
    if !reset.rolled_back {
        eprintln!("qwen4 vision cleanup failed: {:?}", reset.context);
    }
}

#[cfg(test)]
mod tests {
    use super::expand_image_placeholder;

    #[test]
    fn expansion_preserves_framing_and_following_text() {
        let (tokens, start) = expand_image_placeholder(&[7, 10, 11, 12, 8], 11, 10, 12, 4).unwrap();
        assert_eq!(tokens, [7, 10, 11, 11, 11, 11, 12, 8]);
        assert_eq!(start, 2);
    }

    #[test]
    fn ambiguous_or_unframed_images_are_refused() {
        for tokens in [
            &[11][..],
            &[10, 11, 12, 10, 11, 12],
            &[10, 11, 9],
            &[10, 12],
        ] {
            assert!(expand_image_placeholder(tokens, 11, 10, 12, 2).is_err());
        }
        assert!(expand_image_placeholder(&[10, 11, 12], 11, 10, 12, 0).is_err());
    }
}
