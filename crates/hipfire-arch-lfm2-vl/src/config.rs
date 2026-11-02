// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Vision-side configuration for LFM2-VL, parsed from the HFQ artifact's
//! embedded checkpoint config.
//!
//! The quantizer embeds the source `config.json` verbatim under
//! `metadata.config` (nested `text_config` included — see the arch-11
//! runtime's `text_config` flatten) and additively merges the processor
//! pixel-budget keys plus `image_mean` / `image_std` / `resample` into
//! `metadata.config.vision_config`. Tower params come from that
//! `vision_config` object; projector and splitting params are top-level
//! checkpoint keys with pinned defaults from LiquidAI/LFM2.5-VL-3B when
//! absent. Processor mean/std/resample default to that checkpoint's
//! `processor_config.json` (`image_processor`) only when the keys are
//! missing; present values are parsed and rejected if malformed.

use hipfire_runtime::hfq::HfqFile;

#[derive(Debug, Clone)]
pub struct VisionConfig {
    // ── tower ────────────────────────────────────────────────────────────
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub mlp_dim: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    /// Learned position-embedding table entries; NaFlex reshapes to
    /// `sqrt(n) × sqrt(n)` and bilinearly resizes per sub-image.
    pub num_position_embeddings: usize,
    pub norm_eps: f32,

    // ── projector ────────────────────────────────────────────────────────
    pub downsample_factor: usize,
    pub projector_hidden_size: usize,
    pub out_hidden_size: usize,

    // ── preprocessing / splitting budget ────────────────────────────────
    pub do_image_splitting: bool,
    pub tile_size: usize,
    pub min_tiles: usize,
    pub max_tiles: usize,
    pub use_thumbnail: bool,
    pub min_image_tokens: usize,
    pub max_image_tokens: usize,
    pub max_pixels_tolerance: f32,

    // ── processor pixel contract (HF `processor_config.json`) ────────────
    /// Per-channel mean after rescale-to-[0,1]. Checkpoint pin is SigLIP2
    /// `[0.5, 0.5, 0.5]`, not IMAGENET_STANDARD.
    pub image_mean: [f32; 3],
    /// Per-channel std after rescale-to-[0,1]. Checkpoint pin `[0.5, 0.5, 0.5]`.
    pub image_std: [f32; 3],
    /// Pillow / HF `resample` code. `3` = bicubic (`FilterType::CatmullRom`);
    /// `2` = bilinear (`FilterType::Triangle`). Other values are rejected
    /// at parse time.
    pub resample: u32,
}

impl VisionConfig {
    /// Sub-image token count after 2×2 spatial downsampling for a patch grid
    /// `(gh, gw)` — the length of this sub-image's contribution to the
    /// `<image>` placeholder run. Shared by the image splitter and the
    /// forward so prompt building and embedding splicing cannot disagree.
    pub fn tokens_for_grid(&self, gh: usize, gw: usize) -> usize {
        let f = self.downsample_factor.max(1);
        (gh / f) * (gw / f)
    }

    /// Tokens produced by one full-res tile (`tile_size // patch_size` patches
    /// per axis before downsampling).
    pub fn tokens_per_tile(&self) -> usize {
        let patches = self.tile_size / self.patch_size;
        self.tokens_for_grid(patches, patches)
    }
}

impl Default for VisionConfig {
    fn default() -> Self {
        // Pinned from LiquidAI/LFM2.5-VL-3B config.json + processor_config.json
        // (2026-08-27 fetch; processor re-verified 2026-08-30).
        let (hidden_size, num_heads) = (1152, 16);
        Self {
            hidden_size,
            num_heads,
            head_dim: hidden_size / num_heads,
            num_layers: 27,
            mlp_dim: 4304,
            patch_size: 16,
            num_channels: 3,
            num_position_embeddings: 256,
            norm_eps: 1e-6,
            downsample_factor: 2,
            projector_hidden_size: 2048,
            out_hidden_size: 2048,
            do_image_splitting: true,
            tile_size: 512,
            min_tiles: 1,
            max_tiles: 10,
            use_thumbnail: true,
            min_image_tokens: 64,
            max_image_tokens: 256,
            max_pixels_tolerance: 2.0,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            resample: 3,
        }
    }
}

/// Parse the vision config from HFQ metadata. Returns `None` when the
/// artifact carries no vision config at all (plain text model). Malformed
/// processor mean/std/resample values are refused (None after logging) so a
/// tower-bearing artifact fails closed at the existing loader match rather
/// than silently applying the wrong normalization.
pub fn vision_config_from_hfq(hfq: &HfqFile) -> Option<VisionConfig> {
    match vision_config_from_metadata_json(&hfq.metadata_json) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("lfm2-vl: refusing vision_config: {e}");
            None
        }
    }
}

/// Parse HFQ `metadata_json` into a vision config. `Ok(None)` = no vision
/// object (text-only). `Err` = vision object present but processor fields
/// are malformed / unsupported.
fn vision_config_from_metadata_json(json: &str) -> Result<Option<VisionConfig>, String> {
    let meta: serde_json::Value = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let Some(config) = meta.get("config") else {
        return Ok(None);
    };
    // Text-only lfm2 checkpoints have a `config` but no vision_config object
    // AND no tower-budget keys at any level — treat as "no vision".
    let Some(vc) = config.get("vision_config") else {
        return Ok(None);
    };

    let mut c = VisionConfig::default();

    // Tower params (nested object).
    if let Some(o) = vc.as_object() {
        c.hidden_size = o
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(c.hidden_size as u64) as usize;
        c.num_heads = o
            .get("num_attention_heads")
            .or_else(|| o.get("num_heads"))
            .and_then(|v| v.as_u64())
            .unwrap_or(c.num_heads as u64) as usize;
        c.num_layers = o
            .get("num_hidden_layers")
            .or_else(|| o.get("depth"))
            .and_then(|v| v.as_u64())
            .unwrap_or(c.num_layers as u64) as usize;
        c.mlp_dim = o
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(c.mlp_dim as u64) as usize;
        c.patch_size = o
            .get("patch_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(c.patch_size as u64) as usize;
        c.num_channels = o
            .get("num_channels")
            .and_then(|v| v.as_u64())
            .unwrap_or(c.num_channels as u64) as usize;
        c.num_position_embeddings =
            o.get("num_patches")
                .and_then(|v| v.as_u64())
                .unwrap_or(c.num_position_embeddings as u64) as usize;
        c.norm_eps = o
            .get("layer_norm_eps")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(c.norm_eps);
    }
    c.head_dim = c.hidden_size / c.num_heads;

    // Projector + splitting live at checkpoint top level. The quantizer's
    // budget merge may also have landed some of them inside vision_config;
    // prefer nested values there, then top level.
    fn pick_u64<'a>(
        vc: &'a serde_json::Value,
        cfg: &'a serde_json::Value,
        key: &str,
    ) -> Option<u64> {
        vc.get(key)
            .and_then(|v| v.as_u64())
            .or_else(|| cfg.get(key).and_then(|v| v.as_u64()))
    }
    fn pick_bool(vc: &serde_json::Value, cfg: &serde_json::Value, key: &str) -> Option<bool> {
        vc.get(key)
            .and_then(|v| v.as_bool())
            .or_else(|| cfg.get(key).and_then(|v| v.as_bool()))
    }
    fn pick_f64(vc: &serde_json::Value, cfg: &serde_json::Value, key: &str) -> Option<f64> {
        vc.get(key)
            .and_then(|v| v.as_f64())
            .or_else(|| cfg.get(key).and_then(|v| v.as_f64()))
    }
    fn pick<'a>(
        vc: &'a serde_json::Value,
        cfg: &'a serde_json::Value,
        key: &str,
    ) -> Option<&'a serde_json::Value> {
        vc.get(key).or_else(|| cfg.get(key))
    }

    c.downsample_factor =
        pick_u64(vc, config, "downsample_factor").unwrap_or(c.downsample_factor as u64) as usize;
    c.projector_hidden_size = pick_u64(vc, config, "projector_hidden_size")
        .unwrap_or(c.projector_hidden_size as u64) as usize;
    c.out_hidden_size = config
        .get("text_config")
        .and_then(|tc| tc.get("hidden_size"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(c.out_hidden_size);
    c.do_image_splitting =
        pick_bool(vc, config, "do_image_splitting").unwrap_or(c.do_image_splitting);
    c.tile_size = pick_u64(vc, config, "tile_size").unwrap_or(c.tile_size as u64) as usize;
    c.min_tiles = pick_u64(vc, config, "min_tiles").unwrap_or(c.min_tiles as u64) as usize;
    c.max_tiles = pick_u64(vc, config, "max_tiles").unwrap_or(c.max_tiles as u64) as usize;
    c.use_thumbnail = pick_bool(vc, config, "use_thumbnail").unwrap_or(c.use_thumbnail);
    c.min_image_tokens =
        pick_u64(vc, config, "min_image_tokens").unwrap_or(c.min_image_tokens as u64) as usize;
    c.max_image_tokens =
        pick_u64(vc, config, "max_image_tokens").unwrap_or(c.max_image_tokens as u64) as usize;
    c.max_pixels_tolerance = pick_f64(vc, config, "max_pixels_tolerance")
        .map(|v| v as f32)
        .unwrap_or(c.max_pixels_tolerance);

    if let Some(v) = pick(vc, config, "image_mean") {
        c.image_mean = parse_rgb3(v, "image_mean")?;
    }
    if let Some(v) = pick(vc, config, "image_std") {
        c.image_std = parse_rgb3(v, "image_std")?;
    }
    if let Some(v) = pick(vc, config, "resample") {
        c.resample = parse_resample(v)?;
    }

    Ok(Some(c))
}

fn parse_rgb3(v: &serde_json::Value, key: &str) -> Result<[f32; 3], String> {
    let arr = v
        .as_array()
        .ok_or_else(|| format!("processor {key} must be an array of 3 finite floats, got {v}"))?;
    if arr.len() != 3 {
        return Err(format!(
            "processor {key} length {} is invalid; expected 3 channels",
            arr.len()
        ));
    }
    let mut out = [0.0f32; 3];
    for (i, item) in arr.iter().enumerate() {
        let x = item
            .as_f64()
            .ok_or_else(|| format!("processor {key}[{i}] is not a finite number: {item}"))?;
        if !x.is_finite() {
            return Err(format!("processor {key}[{i}] is non-finite ({x})"));
        }
        out[i] = x as f32;
    }
    Ok(out)
}

fn parse_resample(v: &serde_json::Value) -> Result<u32, String> {
    let n = v
        .as_u64()
        .or_else(|| {
            v.as_f64()
                .filter(|x| x.is_finite() && *x >= 0.0 && x.fract() == 0.0)
                .map(|x| x as u64)
        })
        .ok_or_else(|| format!("processor resample must be an integer, got {v}"))?;
    match n {
        2 | 3 => Ok(n as u32),
        other => Err(format!(
            "unsupported processor resample={other} (Pillow/HF); supported: 2 (bilinear), 3 (bicubic)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_pinned_checkpoint() {
        let c = VisionConfig::default();
        assert_eq!(c.hidden_size, 1152);
        assert_eq!(c.head_dim, 72);
        assert_eq!(c.num_layers, 27);
        assert_eq!(c.mlp_dim, 4304);
        assert_eq!(c.downsample_factor, 2);
        assert_eq!(c.tokens_per_tile(), 256); // (512/16/2)^2
        assert_eq!(c.tokens_for_grid(32, 32), 256);
        assert_eq!(c.tokens_for_grid(10, 26), 65);
        assert_eq!(c.image_mean, [0.5, 0.5, 0.5]);
        assert_eq!(c.image_std, [0.5, 0.5, 0.5]);
        assert_eq!(c.resample, 3);
    }

    fn meta_with_vision(vision: serde_json::Value) -> String {
        serde_json::json!({ "config": { "vision_config": vision } }).to_string()
    }

    #[test]
    fn processor_fields_round_trip_from_checkpoint_processor_config() {
        // Nested `image_processor` keys as the quantizer copies them into
        // `config.vision_config` (LiquidAI/LFM2.5-VL-3B processor_config.json).
        let json = meta_with_vision(serde_json::json!({
            "hidden_size": 1152,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "resample": 3,
            "max_image_tokens": 256,
        }));
        let c = vision_config_from_metadata_json(&json)
            .expect("parse")
            .expect("vision present");
        assert_eq!(c.image_mean, [0.5, 0.5, 0.5]);
        assert_eq!(c.image_std, [0.5, 0.5, 0.5]);
        assert_eq!(c.resample, 3);
        assert_eq!(c.max_image_tokens, 256);
    }

    #[test]
    fn absent_processor_fields_keep_defaults() {
        let json = meta_with_vision(serde_json::json!({ "hidden_size": 1152 }));
        let c = vision_config_from_metadata_json(&json)
            .expect("parse")
            .expect("vision present");
        assert_eq!(c.image_mean, [0.5, 0.5, 0.5]);
        assert_eq!(c.image_std, [0.5, 0.5, 0.5]);
        assert_eq!(c.resample, 3);
    }

    #[test]
    fn present_processor_fields_override_defaults() {
        let json = meta_with_vision(serde_json::json!({
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "resample": 2,
        }));
        let c = vision_config_from_metadata_json(&json)
            .expect("parse")
            .expect("vision present");
        assert!((c.image_mean[0] - 0.485).abs() < 1e-6);
        assert!((c.image_std[1] - 0.224).abs() < 1e-6);
        assert_eq!(c.resample, 2);
    }

    #[test]
    fn malformed_mean_length_is_rejected() {
        let json = meta_with_vision(serde_json::json!({ "image_mean": [0.5, 0.5] }));
        let err = vision_config_from_metadata_json(&json).unwrap_err();
        assert!(err.contains("image_mean"), "{err}");
        assert!(err.contains("length 2"), "{err}");
    }

    #[test]
    fn malformed_std_non_number_is_rejected() {
        let json = meta_with_vision(serde_json::json!({ "image_std": [0.5, 0.5, "nan"] }));
        let err = vision_config_from_metadata_json(&json).unwrap_err();
        assert!(err.contains("image_std"), "{err}");
    }

    #[test]
    fn unsupported_resample_is_rejected() {
        let json = meta_with_vision(serde_json::json!({ "resample": 1 }));
        let err = vision_config_from_metadata_json(&json).unwrap_err();
        assert!(err.contains("resample=1"), "{err}");
        assert!(err.contains("bicubic"), "{err}");
    }

    #[test]
    fn text_only_metadata_is_none() {
        let json = serde_json::json!({ "config": { "hidden_size": 2048 } }).to_string();
        assert!(vision_config_from_metadata_json(&json).unwrap().is_none());
    }
}
