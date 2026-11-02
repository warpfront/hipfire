// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! FLUX diffusion transformer config, parsed from a safetensors
//! `config.json` (same JSON tree the HF repo publishes; also the metadata
//! shape a future `.hfq` checkpoint bundle would carry). Accepts the BFL / HF
//! field spellings plus the diffusers equivalents (`attention_head_dim`,
//! `axes_dims_rope`, `joint_attention_dim`, `in_channels`).
//!
//! Defaults below are the FLUX.1 architecture as published
//! (`black-forest-labs/FLUX.1-schnell` model card). They were
//! **verified against the actual config.json**; the parser
//! prefers JSON keys when present and only falls back to these defaults, so
//! a verified config always wins. The one hard invariant enforced here is
//! `axes_dim` summing to `head_dim` (the 2D RoPE budget per head).
//!
//! There is deliberately NO image-resolution field: FLUX's token count comes
//! from the latent grid passed at forward time (the denoise loop requests
//! width/height), not from a config key. The CPU reference takes the grid
//! explicitly (`flux.rs`).
//!
//! `qk_norm` and `norm_type` are parsed for validation and diagnostics only:
//! the BFL-2024 architecture ALWAYS applies the per-head QK RMSNorm and
//! weightless LayerNorms for the block/`norm_final` paths — see `src/flux.rs`.

use crate::manifest::TS_EMBED_DIM;
use serde_json::Value;

/// Which FLUX generation a config describes. `Flux1` covers the BFL/HF
/// FLUX.1 dev/schnell architecture (bias-carrying blocks, per-stream
/// modulation, 3-axis RoPE with a zero-padded fourth slot). `Flux2` covers
/// the Klein 4B/9B distilled architecture (`Flux2Transformer2DModel`:
/// bias-free blocks, modulation shared across streams, native 4-axis RoPE).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluxFamily {
    Flux1,
    Flux2,
}

/// Fully-parsed diffusion config for the FLUX.1 / FLUX.2 (Klein) MMDiT
/// family.
#[derive(Debug, Clone, PartialEq)]
pub struct FluxDiffusionConfig {
    /// Which FLUX generation this config describes.
    pub family: FluxFamily,
    /// Latent-patch embedding width (`hidden_size`).
    pub hidden_size: usize,
    /// Number of double (dual-stream) blocks (`num_layers`).
    pub num_layers: usize,
    /// Number of single (fused-stream) blocks (`num_single_layers`).
    pub num_single_layers: usize,
    /// Attention head count (`num_attention_heads`).
    pub num_attention_heads: usize,
    /// Attention head width (`head_dim`).
    pub head_dim: usize,
    /// Latent patch size (`patch_size`, 2).
    pub patch_size: usize,
    /// Text token ceiling (`max_sequence_length`, 512).
    pub max_sequence_length: usize,
    /// Guidance embedding width; 0 = guidance-free (FLUX.1-schnell).
    pub guidance_embed_dim: usize,
    /// Pooled text-conditioning width (`pooled_projection_dim`, 768).
    pub pooled_projection_dim: usize,
    /// 2D/4D RoPE per-axis dims (`axes_dim` / `axes_dims_rope`); must sum to
    /// `head_dim`. FLUX.1 uses 3 axes with a zero-padded fourth slot; FLUX.2
    /// (Klein) uses all four axes natively.
    pub axes_dim: [usize; 4],
    /// RoPE base frequency (`theta` / `rope_theta`; 10000.0 for FLUX.1,
    /// 2000.0 for FLUX.2).
    pub theta: f64,
    /// QK RMSNorm (`qk_norm`).
    pub qk_norm: bool,
    /// Norm variant (`norm_type`, "rms_norm").
    pub norm_type: String,
    /// Latent channel count (`latent_channels` / `in_channels`; 16 for the
    /// FLUX.1 VAE family; 128 for FLUX.2 Klein's packed token width. The
    /// patched `img_in`/`x_embedder` input width is
    /// `patch_size² × latent_channels`).
    pub latent_channels: usize,
    /// Text-token hidden width (`joint_attention_dim`; the T5-XXL width 4096
    /// for FLUX.1, overridable for tiny parity fixtures).
    pub txt_hidden_dim: usize,
    /// MLP expansion ratio (`mlp_ratio`; 4.0 for FLUX.1, 3.0 for FLUX.2).
    pub mlp_ratio: f32,
    /// Whether linear/attention layers carry a bias (true for FLUX.1, false
    /// for FLUX.2 Klein).
    pub bias: bool,
    /// Whether AdaLN modulation is shared across the double-stream blocks
    /// (false for FLUX.1's per-stream modulation, true for FLUX.2 Klein).
    pub shared_modulation: bool,
}

impl FluxDiffusionConfig {
    /// MLP hidden width: `hidden_size * mlp_ratio`.
    pub fn mlp_width(&self) -> usize {
        (self.hidden_size as f32 * self.mlp_ratio) as usize
    }

    /// `x_embedder`/`img_in` input width: `patch_size² × latent_channels`.
    pub fn patch_in(&self) -> usize {
        self.patch_size * self.patch_size * self.latent_channels
    }

    /// True when this config describes the FLUX.2 (Klein) family.
    pub fn is_flux2(&self) -> bool {
        self.family == FluxFamily::Flux2
    }

    /// Architecture-level default step count for txt2img: 28 for a
    /// guidance-distilled checkpoint (FLUX.1-dev, `guidance_embed_dim > 0`),
    /// 4 for a step-distilled one (FLUX.1-schnell, no guidance embedder).
    /// The CLI/TUI and HTTP layer read this instead of hardcoding steps.
    pub fn default_steps(&self) -> u32 {
        if self.guidance_embed_dim > 0 {
            28
        } else {
            4
        }
    }

    /// Parse from a HF-style `config.json` value. Unknown-typed or
    /// contradictory fields fail closed with a named reason.
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let get_usize = |key: &str, default: usize| -> Result<usize, String> {
            match v.get(key) {
                None => Ok(default),
                Some(x) => x
                    .as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| format!("flux config: `{key}` is not an integer: {x}")),
            }
        };
        // FLUX.2 (Klein) publishes `_class_name: "Flux2Transformer2DModel"`;
        // a leaner `model_type: "flux2"` alias is also recognised for hand
        // -written / test configs.
        let family = if v.get("_class_name").and_then(|c| c.as_str())
            == Some("Flux2Transformer2DModel")
            || v.get("model_type").and_then(|m| m.as_str()) == Some("flux2")
        {
            FluxFamily::Flux2
        } else {
            FluxFamily::Flux1
        };
        let num_attention_heads = get_usize("num_attention_heads", 24)?;
        // Explicit head dim wins; diffusers spellings accepted as aliases.
        let explicit_head_dim = get_usize("head_dim", 0)?.max(get_usize("attention_head_dim", 0)?);
        // hidden_size may be absent in diffusers-style configs (derived as
        // heads × head_dim); when present it must divide evenly by heads.
        let explicit_hidden = match v.get("hidden_size") {
            None => None,
            Some(x) => Some(
                x.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| format!("flux config: `hidden_size` is not an integer: {x}"))?,
            ),
        };
        let (hidden_size, head_dim) = match explicit_hidden {
            Some(hs) => {
                let hd = if explicit_head_dim > 0 {
                    explicit_head_dim
                } else {
                    hs / num_attention_heads
                };
                (hs, hd)
            }
            None => {
                let hd = if explicit_head_dim > 0 {
                    explicit_head_dim
                } else {
                    128
                };
                (num_attention_heads * hd, hd)
            }
        };
        if hidden_size / num_attention_heads != head_dim {
            return Err(format!(
                "flux config: hidden_size {hidden_size} / heads {num_attention_heads} != head_dim {head_dim}"
            ));
        }
        // RoPE budget: an explicit `axes_dim`/`axes_dims_rope` must sum to
        // head_dim (3 entries get a trailing 0 appended, so FLUX.1's 2D
        // convention and FLUX.2's native 4D convention both parse). Absent,
        // FLUX.1 falls back to the BFL default [16,56,56,0] (or a
        // proportional 3-way split off-128), while FLUX.2 requires either an
        // explicit key or head_dim == 128 (Klein's published [32,32,32,32]).
        let axes_dim = if v.get("axes_dim").is_some() || v.get("axes_dims_rope").is_some() {
            let a = parse_axes_dim(
                v.get("axes_dim")
                    .or_else(|| v.get("axes_dims_rope"))
                    .unwrap(),
            )?;
            if a.iter().sum::<usize>() != head_dim {
                return Err(format!(
                    "flux config: axes_dim {a:?} must sum to head_dim {head_dim}"
                ));
            }
            a
        } else {
            match family {
                FluxFamily::Flux1 => {
                    if head_dim == 128 {
                        [16, 56, 56, 0]
                    } else {
                        let base = head_dim / 3;
                        [base, base, head_dim - 2 * base, 0]
                    }
                }
                FluxFamily::Flux2 => {
                    if head_dim == 128 {
                        [32, 32, 32, 32]
                    } else {
                        return Err(
                            "flux2 config: axes_dim required when head_dim != 128".to_string()
                        );
                    }
                }
            }
        };
        let norm_type = v
            .get("norm_type")
            .and_then(|x| x.as_str())
            .unwrap_or("rms_norm")
            .to_string();
        if norm_type != "rms_norm" {
            return Err(format!("flux config: unsupported norm_type `{norm_type}`"));
        }
        // FLUX.1-dev's diffusers config carries `guidance_embeds: true` and
        // no `guidance_embed_dim` key; the BFL single-file layout stores the
        // `guidance_in.*` embedder weights and wants dim 256. Defaulting to 0
        // here silently DROPPED the guidance embedding on the real checkpoint
        // (the forward keys off guidance_embed_dim > 0), which pinned the
        // predicted velocity to the guidance-free stream and cost ~0.36 of
        // rel_l2 at step 20 vs ComfyUI. Only schnell (no flag, no keys)
        // stays guidance-free.
        let guidance_embed_dim = if let Ok(n) = get_usize("guidance_embed_dim", 0) {
            if n > 0 {
                n
            } else if bool_field(v, "guidance_embeds", false)? {
                TS_EMBED_DIM
            } else {
                0
            }
        } else {
            0
        };
        // FLUX.2 (Klein) has no architecture-implied default block count —
        // an absent key is a config error rather than a silent 19/38 guess.
        let (num_layers, num_single_layers) = match family {
            FluxFamily::Flux1 => (
                get_usize("num_layers", 19)?,
                get_usize("num_single_layers", 38)?,
            ),
            FluxFamily::Flux2 => {
                if v.get("num_layers").is_none() || v.get("num_single_layers").is_none() {
                    return Err(
                        "flux2 config: num_layers and num_single_layers are required".to_string(),
                    );
                }
                (
                    get_usize("num_layers", 0)?,
                    get_usize("num_single_layers", 0)?,
                )
            }
        };
        let mlp_ratio = v
            .get("mlp_ratio")
            .and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .unwrap_or(match family {
                FluxFamily::Flux1 => 4.0,
                FluxFamily::Flux2 => 3.0,
            });
        let pooled_projection_dim_default = match family {
            FluxFamily::Flux1 => 768,
            FluxFamily::Flux2 => 0,
        };
        Ok(FluxDiffusionConfig {
            family,
            hidden_size,
            num_layers,
            num_single_layers,
            num_attention_heads,
            head_dim,
            patch_size: get_usize("patch_size", 2)?,
            max_sequence_length: get_usize("max_sequence_length", 512)?,
            guidance_embed_dim,
            pooled_projection_dim: get_usize(
                "pooled_projection_dim",
                pooled_projection_dim_default,
            )?,
            axes_dim,
            theta: v
                .get("theta")
                .or_else(|| v.get("rope_theta"))
                .and_then(|x| x.as_f64())
                .unwrap_or(match family {
                    FluxFamily::Flux1 => 10000.0,
                    FluxFamily::Flux2 => 2000.0,
                }),
            qk_norm: bool_field(v, "qk_norm", true)?,
            norm_type,
            // latent_channels / in_channels: 16 is the FLUX.1-VAE family
            // value (128 for FLUX.2 Klein's packed token width); a config
            // that names either key wins, otherwise default 16.
            latent_channels: if v.get("latent_channels").is_some() || v.get("in_channels").is_some()
            {
                get_usize("latent_channels", 0)?.max(get_usize("in_channels", 0)?)
            } else {
                16
            },
            txt_hidden_dim: get_usize("joint_attention_dim", 4096)?,
            mlp_ratio,
            bias: family == FluxFamily::Flux1,
            shared_modulation: family == FluxFamily::Flux2,
        })
    }
}

fn parse_axes_dim(v: &Value) -> Result<[usize; 4], String> {
    let arr = v
        .as_array()
        .ok_or_else(|| format!("flux config: `axes_dim` is not an array: {v}"))?;
    if arr.len() != 3 && arr.len() != 4 {
        return Err(format!(
            "flux config: `axes_dim` must have 3 or 4 entries, got {}",
            arr.len()
        ));
    }
    let mut out = [0usize; 4];
    for (i, x) in arr.iter().enumerate() {
        out[i] = x
            .as_u64()
            .ok_or_else(|| format!("flux config: `axes_dim[{i}]` not an integer: {x}"))?
            as usize;
    }
    Ok(out)
}

fn bool_field(v: &Value, key: &str, default: bool) -> Result<bool, String> {
    match v.get(key) {
        None => Ok(default),
        Some(x) => x
            .as_bool()
            .ok_or_else(|| format!("flux config: `{key}` is not a bool: {x}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_match_flux1_schnell_blueprint() {
        let cfg = FluxDiffusionConfig::from_json(&json!({})).unwrap();
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_layers, 19);
        assert_eq!(cfg.num_single_layers, 38);
        assert_eq!(cfg.num_attention_heads, 24);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.patch_size, 2);
        assert_eq!(cfg.max_sequence_length, 512);
        assert_eq!(cfg.guidance_embed_dim, 0); // schnell: guidance-free
        assert_eq!(cfg.pooled_projection_dim, 768);
        assert_eq!(cfg.axes_dim, [16, 56, 56, 0]);
        assert_eq!(cfg.theta, 10000.0);
        assert!(cfg.qk_norm);
        assert_eq!(cfg.latent_channels, 16);
    }

    #[test]
    fn json_keys_preferred_over_defaults() {
        let cfg = FluxDiffusionConfig::from_json(&json!({
            "hidden_size": 64,
            "num_attention_heads": 8,
            "head_dim": 8,
            "axes_dim": [2, 3, 3],
            "num_layers": 2,
            "num_single_layers": 4,
            "patch_size": 4,
            "guidance_embed_dim": 256,
            "qk_norm": false,
        }))
        .unwrap();
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.head_dim, 8);
        assert_eq!(cfg.axes_dim, [2, 3, 3, 0]);
        assert_eq!(cfg.num_layers, 2);
        assert_eq!(cfg.num_single_layers, 4);
        assert_eq!(cfg.guidance_embed_dim, 256);
        assert!(!cfg.qk_norm);
        assert_eq!(cfg.theta, 10000.0);
    }

    #[test]
    fn guidance_defaults_from_embeds_flag_for_dev() {
        // The HF diffusers config for FLUX.1-dev says `guidance_embeds: true`
        // and has no `guidance_embed_dim` key. This is what gate 2026-09-02
        // measured failing: dim defaulted to 0, the guidance embedder was
        // never loaded or applied, and the velocity diverged ~0.36 rel_l2 at
        // step 20 vs ComfyUI (which always feeds guidance 3.5 to the
        // guidance-distilled dev model).
        let cfg = FluxDiffusionConfig::from_json(&json!({ "guidance_embeds": true })).unwrap();
        assert_eq!(cfg.guidance_embed_dim, 256);
        // Explicit dim still wins; schnell (no flag) stays guidance-free.
        let cfg = FluxDiffusionConfig::from_json(&json!({ "guidance_embed_dim": 32 })).unwrap();
        assert_eq!(cfg.guidance_embed_dim, 32);
        let cfg = FluxDiffusionConfig::from_json(&json!({ "guidance_embeds": false })).unwrap();
        assert_eq!(cfg.guidance_embed_dim, 0);
    }

    #[test]
    fn head_dim_derived_when_absent() {
        let cfg = FluxDiffusionConfig::from_json(&json!({
            "hidden_size": 128,
            "num_attention_heads": 4,
        }))
        .unwrap();
        assert_eq!(cfg.head_dim, 32);
    }

    #[test]
    fn axes_dim_must_sum_to_head_dim() {
        let err = FluxDiffusionConfig::from_json(&json!({ "axes_dim": [1, 1, 1] })).unwrap_err();
        assert!(err.contains("must sum to head_dim"), "{err}");
    }

    #[test]
    fn unknown_norm_type_fails_closed() {
        let err =
            FluxDiffusionConfig::from_json(&json!({ "norm_type": "layer_norm" })).unwrap_err();
        assert!(err.contains("unsupported norm_type"), "{err}");
    }

    #[test]
    fn bad_json_types_fail_closed() {
        let err = FluxDiffusionConfig::from_json(&json!({ "hidden_size": "big" })).unwrap_err();
        assert!(err.contains("`hidden_size` is not an integer"), "{err}");
    }

    #[test]
    fn klein_4b_config_parses_from_the_published_json() {
        let raw = include_str!("../tests/fixtures/klein/klein-4b-transformer.json");
        let cfg = FluxDiffusionConfig::from_json(&serde_json::from_str(raw).unwrap()).unwrap();
        assert_eq!(cfg.family, FluxFamily::Flux2);
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_attention_heads, 24);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.num_layers, 5);
        assert_eq!(cfg.num_single_layers, 20);
        assert_eq!(cfg.axes_dim, [32, 32, 32, 32]);
        assert_eq!(cfg.theta, 2000.0);
        assert_eq!(cfg.patch_size, 1);
        assert_eq!(cfg.latent_channels, 128);
        assert_eq!(cfg.patch_in(), 128);
        assert_eq!(cfg.txt_hidden_dim, 7680);
        assert_eq!(cfg.mlp_ratio, 3.0);
        assert_eq!(cfg.mlp_width(), 9216);
        assert!(!cfg.bias);
        assert!(cfg.shared_modulation);
        assert_eq!(cfg.guidance_embed_dim, 0);
        assert_eq!(cfg.pooled_projection_dim, 0);
    }

    #[test]
    fn klein_9b_config_parses_from_the_published_json() {
        let raw = include_str!("../tests/fixtures/klein/klein-9b-transformer.json");
        let cfg = FluxDiffusionConfig::from_json(&serde_json::from_str(raw).unwrap()).unwrap();
        assert_eq!(cfg.family, FluxFamily::Flux2);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_layers, 8);
        assert_eq!(cfg.num_single_layers, 24);
        assert_eq!(cfg.txt_hidden_dim, 12288);
        assert_eq!(cfg.mlp_width(), 12288);
    }

    #[test]
    fn flux1_defaults_keep_three_axes_and_a_zero_fourth() {
        let cfg = FluxDiffusionConfig::from_json(&json!({})).unwrap();
        assert_eq!(cfg.family, FluxFamily::Flux1);
        assert_eq!(cfg.axes_dim, [16, 56, 56, 0]);
        assert_eq!(cfg.mlp_ratio, 4.0);
        assert!(cfg.bias);
        assert!(!cfg.shared_modulation);
        assert_eq!(cfg.mlp_width(), 4 * 3072);
    }

    #[test]
    fn flux2_with_a_model_type_key_is_also_recognised() {
        let cfg = FluxDiffusionConfig::from_json(&json!({ "model_type": "flux2", "num_layers": 1, "num_single_layers": 1, "in_channels": 128, "joint_attention_dim": 7680, "patch_size": 1, "axes_dims_rope": [32,32,32,32], "rope_theta": 2000, "mlp_ratio": 3.0 })).unwrap();
        assert_eq!(cfg.family, FluxFamily::Flux2);
    }
}
