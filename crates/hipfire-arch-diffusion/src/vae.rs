// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU FLUX VAE decoder reference (latent → pixels), pinned to the
//! LDM / `taming-transformers` `Decoder` that BFL ships and ComfyUI runs
//! (`comfy/ldm/autoencoder.py`), NOT the diffusers `AutoencoderKL` naming:
//!
//! - `conv_in` (`decoder.conv_in`, latent→block_in, 3×3 pad 1), `mid`
//!   (`block_1` resnet → `attn_1` → `block_2` resnet), `up.{level}` blocks
//!   processed from the highest level down (`up.3` … `up.0`), `norm_out`
//!   (GroupNorm `norm_num_groups` groups, eps 1e-6), SiLU, `conv_out`
//!   (`decoder.conv_out`, → out_channels).
//! - `ResnetBlock`: norm1 → SiLU → conv1 → norm2 → SiLU → conv2 → + residual.
//!   When `in_channels != out_channels` the residual goes through a 1×1
//!   `nin_shortcut` conv (the default `use_conv_shortcut=False`). This is
//!   what the first resnet of every up block does — it halves the channel
//!   count — and is the reason the old channel-preserving decoder could not
//!   read real weights.
//! - `Upsample` (`decoder.up.{level}.upsample`): nearest 2× then a 3×3 pad-1
//!   conv. Present on every up block but level 0.
//! - mid `attn_1`: GroupNorm → 1×1 q/k/v → single-head attention
//!   (head_dim = channels, scale 1/√c) → 1×1 proj_out → residual.
//!
//! For FLUX: `block_out_channels [128,256,512,512]`, latent 16, layers_per_block
//! 2 (3 resnets/block), 32 norm groups → 4 levels, 3 upsamples ⇒ spatial ×8.

use crate::flux::Tensor;
use crate::nn;
use hipfire_runtime::model_source::ModelSource as ModelSourceTrait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    pub mid_block_add_attention: bool,
    /// FLUX.1: always present. FLUX.2 Klein: absent — the latent is
    /// normalized by [`LatentNorm::BatchNorm`] instead (see `bn.running_mean`
    /// / `bn.running_var` in [`VaeDecoderWeights::load`]).
    pub scaling_factor: Option<f32>,
    pub shift_factor: Option<f32>,
    pub use_quant_conv: bool,
    pub use_post_quant_conv: bool,
    /// Eps for `std = sqrt(var + eps)` on the latent BatchNorm stats.
    pub batch_norm_eps: f32,
    /// Latent-space patch cell packed by the transformer: `(1, 1)` for
    /// FLUX.1 (no extra latent patching beyond the VAE), `(2, 2)` for
    /// FLUX.2 Klein.
    pub latent_patch: (usize, usize),
}

/// How a raw model-space latent maps to/from VAE input space.
///
/// FLUX.1 uses a single scalar `scaling_factor`/`shift_factor` pair
/// (`ScaleShift`). FLUX.2 Klein normalizes the latent with a per-channel
/// BatchNorm instead (`BatchNorm`), fit over the packed latent's `latent *
/// patch.0 * patch.1` columns.
#[derive(Debug, Clone)]
pub enum LatentNorm {
    ScaleShift { scaling: f32, shift: f32 },
    BatchNorm { mean: Vec<f32>, std: Vec<f32> },
}

impl VaeConfig {
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let get_u = |k: &str| -> Result<usize, String> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| format!("vae config: missing `{k}`"))
        };
        let blocks: Vec<usize> = v
            .get("block_out_channels")
            .and_then(|a| a.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as usize)
                    .collect()
            })
            .ok_or("vae config: missing `block_out_channels`")?;
        Ok(Self {
            in_channels: get_u("in_channels")?,
            out_channels: get_u("out_channels")?,
            latent_channels: get_u("latent_channels")?,
            block_out_channels: blocks,
            layers_per_block: get_u("layers_per_block")?,
            norm_num_groups: get_u("norm_num_groups")?,
            mid_block_add_attention: v
                .get("mid_block_add_attention")
                .and_then(|x| x.as_bool())
                .unwrap_or(true),
            scaling_factor: v
                .get("scaling_factor")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32),
            shift_factor: v
                .get("shift_factor")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32),
            use_quant_conv: v
                .get("use_quant_conv")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            use_post_quant_conv: v
                .get("use_post_quant_conv")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            batch_norm_eps: v
                .get("batch_norm_eps")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(1e-4),
            latent_patch: v
                .get("patch_size")
                .and_then(|a| a.as_array())
                .filter(|a| a.len() == 2)
                .and_then(|a| Some((a[0].as_u64()? as usize, a[1].as_u64()? as usize)))
                .unwrap_or((1, 1)),
        })
    }
}

/// One taming `ResnetBlock`, channels possibly changing `in_ch → out_ch`.
/// norm1/conv1 see `in_ch`, norm2/conv2/residual-out see `out_ch`; the
/// residual is a 1×1 `nin_shortcut` when the channels differ.
#[derive(Debug, Clone)]
pub struct VaeResnet {
    pub in_ch: usize,
    pub out_ch: usize,
    pub norm1_w: Tensor, // [in_ch]
    pub norm1_b: Tensor,
    pub conv1_w: Tensor, // [out_ch, in_ch, 3, 3]
    pub conv1_b: Tensor,
    pub norm2_w: Tensor, // [out_ch]
    pub norm2_b: Tensor,
    pub conv2_w: Tensor, // [out_ch, out_ch, 3, 3]
    pub conv2_b: Tensor,
    pub nin_shortcut_w: Option<Tensor>, // [out_ch, in_ch, 1, 1]
    pub nin_shortcut_b: Option<Tensor>,
}

/// One `up.{level}` decoder block: `layers_per_block + 1` resnets, then an
/// optional nearest-2×-plus-conv upsample (absent only at level 0).
#[derive(Debug, Clone)]
pub struct UpBlock {
    pub channels: usize,
    pub resnets: Vec<VaeResnet>,
    pub upsample_w: Option<Tensor>, // [channels, channels, 3, 3]
    pub upsample_b: Option<Tensor>,
}

/// Mid-block self-attention (single head, head_dim = channels); the q/k/v/
/// proj weights are 1×1 convs, stored as `[c, c]` linears.
#[derive(Debug, Clone)]
pub struct MidAttn {
    pub group_norm_w: Tensor,
    pub group_norm_b: Tensor,
    pub q_w: Tensor,
    pub q_b: Tensor,
    pub k_w: Tensor,
    pub k_b: Tensor,
    pub v_w: Tensor,
    pub v_b: Tensor,
    pub out_w: Tensor,
    pub out_b: Tensor,
}

/// Decoder weights (row-major f32). Convs are `[c_out][c_in][kh][kw]`.
#[derive(Debug, Clone)]
pub struct VaeDecoderWeights {
    pub config: VaeConfig,
    pub conv_in_w: Tensor, // [block_in, latent_channels, 3, 3]
    pub conv_in_b: Tensor,
    /// `mid.block_1`, `mid.block_2` (channel-preserving at `block_in`).
    pub mid_resnet: Vec<VaeResnet>,
    /// `mid.attn_1` (present iff `mid_block_add_attention`).
    pub mid_attn: Option<MidAttn>,
    /// Indexed by level (0..n); decode walks from the highest level down.
    pub up_blocks: Vec<UpBlock>,
    pub conv_norm_out_w: Tensor, // [block_out_channels[0]]
    pub conv_norm_out_b: Tensor,
    pub conv_out_w: Tensor, // [out_channels, block_out_channels[0], 3, 3]
    pub conv_out_b: Tensor,
    /// FLUX.2 Klein's `post_quant_conv` (1×1, latent→latent), applied to `z`
    /// before `conv_in`. `None` for FLUX.1 (`use_post_quant_conv=false`).
    pub post_quant_conv: Option<(Tensor, Tensor)>,
    /// Latent normalization applied by the caller before `decode`/`decode_stages`.
    pub latent_norm: LatentNorm,
}

/// Build the [`LatentNorm`] for a VAE component source: FLUX.2 Klein's
/// top-level `bn.running_mean` / `bn.running_var` (NOT under `encoder.`/
/// `decoder.`) when present, else FLUX.1's `scaling_factor`/`shift_factor`
/// pair from `config`.
fn load_latent_norm(src: &dyn ModelSourceTrait, config: &VaeConfig) -> Result<LatentNorm, String> {
    if let Some((info, data)) = src.tensor_data("bn.running_mean") {
        let n = config.latent_channels * config.latent_patch.0 * config.latent_patch.1;
        let mean = crate::flux::decode_dtype(&info.dtype, &data)?;
        if mean.len() != n {
            return Err(format!(
                "vae: `bn.running_mean` has {} elems, expected {n}",
                mean.len()
            ));
        }
        let (var_info, var_data) = src
            .tensor_data("bn.running_var")
            .ok_or("vae: `bn.running_mean` present but `bn.running_var` missing")?;
        let var = crate::flux::decode_dtype(&var_info.dtype, &var_data)?;
        if var.len() != n {
            return Err(format!(
                "vae: `bn.running_var` has {} elems, expected {n}",
                var.len()
            ));
        }
        let std = var
            .iter()
            .map(|x| (x + config.batch_norm_eps).sqrt())
            .collect();
        Ok(LatentNorm::BatchNorm { mean, std })
    } else {
        Ok(LatentNorm::ScaleShift {
            scaling: config
                .scaling_factor
                .ok_or("vae: no scaling_factor and no bn stats")?,
            shift: config.shift_factor.unwrap_or(0.0),
        })
    }
}

/// Load one tensor by manifest name, checked against the expected `[rows,
/// cols]` shape. Shared by the decoder and encoder loaders.
fn load_tensor(
    src: &dyn ModelSourceTrait,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Tensor, String> {
    let (info, data) = src
        .tensor_data(name)
        .ok_or_else(|| format!("vae: missing tensor `{name}`"))?;
    let n = info.shape.iter().product::<usize>();
    if n != rows * cols {
        return Err(format!(
            "vae: tensor `{name}` has {n} elems, expected {}",
            rows * cols
        ));
    }
    Ok(Tensor {
        data: crate::flux::decode_dtype(&info.dtype, &data)?,
        rows,
        cols,
    })
}

/// One taming/diffusers `ResnetBlock`. norm1/conv1 over `in_ch`, norm2/conv2
/// over `out_ch`; a 1×1 residual projection (`shortcut` is `nin_shortcut` in
/// LDM naming, `conv_shortcut` in diffusers) only when the channels differ.
/// Shared by [`VaeDecoderWeights::load`] and [`VaeEncoderWeights::load`].
fn load_resnet(
    src: &dyn ModelSourceTrait,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    shortcut: &str,
) -> Result<VaeResnet, String> {
    let nin = if in_ch == out_ch {
        (None, None)
    } else {
        let w = load_tensor(src, &format!("{prefix}.{shortcut}.weight"), out_ch, in_ch)?;
        let b = load_tensor(src, &format!("{prefix}.{shortcut}.bias"), out_ch, 1)?;
        (Some(w), Some(b))
    };
    Ok(VaeResnet {
        in_ch,
        out_ch,
        norm1_w: load_tensor(src, &format!("{prefix}.norm1.weight"), in_ch, 1)?,
        norm1_b: load_tensor(src, &format!("{prefix}.norm1.bias"), in_ch, 1)?,
        conv1_w: load_tensor(src, &format!("{prefix}.conv1.weight"), out_ch, in_ch * 9)?,
        conv1_b: load_tensor(src, &format!("{prefix}.conv1.bias"), out_ch, 1)?,
        norm2_w: load_tensor(src, &format!("{prefix}.norm2.weight"), out_ch, 1)?,
        norm2_b: load_tensor(src, &format!("{prefix}.norm2.bias"), out_ch, 1)?,
        conv2_w: load_tensor(src, &format!("{prefix}.conv2.weight"), out_ch, out_ch * 9)?,
        conv2_b: load_tensor(src, &format!("{prefix}.conv2.bias"), out_ch, 1)?,
        nin_shortcut_w: nin.0,
        nin_shortcut_b: nin.1,
    })
}

impl VaeDecoderWeights {
    /// True for a [`Self::config_only`] load: the config is real but every
    /// weight is empty, so there is nothing to upload and `decode` must not
    /// be called.
    pub fn is_config_only(&self) -> bool {
        self.conv_in_w.data.is_empty()
    }

    /// Load the decoder half of a FLUX VAE component source, but leave every
    /// weight empty. `decode` must not be called on the result. `meta` only
    /// reads `scaling_factor` / `shift_factor`, both of which live in the
    /// config, so this lets the transformer path run and emit latents for an
    /// external decoder when callers set `HIPFIRE_VAE_CONFIG_ONLY=1`.
    pub fn config_only(src: &dyn ModelSourceTrait) -> Result<Self, String> {
        let v: serde_json::Value = serde_json::from_str(src.metadata_json())
            .map_err(|e| format!("vae: config.json invalid: {e}"))?;
        let v = v.get("config").cloned().unwrap_or(v);
        let config = VaeConfig::from_json(&v)?;
        let latent_norm = load_latent_norm(src, &config)?;
        let empty = || Tensor {
            data: vec![],
            rows: 0,
            cols: 0,
        };
        Ok(Self {
            config,
            conv_in_w: empty(),
            conv_in_b: empty(),
            mid_resnet: vec![],
            mid_attn: None,
            up_blocks: vec![],
            conv_norm_out_w: empty(),
            conv_norm_out_b: empty(),
            conv_out_w: empty(),
            conv_out_b: empty(),
            post_quant_conv: None,
            latent_norm,
        })
    }

    pub fn load(src: &dyn ModelSourceTrait) -> Result<Self, String> {
        let v: serde_json::Value = serde_json::from_str(src.metadata_json())
            .map_err(|e| format!("vae: config.json invalid: {e}"))?;
        let v = v.get("config").cloned().unwrap_or(v);
        let config = VaeConfig::from_json(&v)?;
        let latent_norm = load_latent_norm(src, &config)?;
        let nb = config.block_out_channels.len();
        let block_in = config.block_out_channels[nb - 1];
        let first_ch = config.block_out_channels[0];
        // Two naming conventions for the same decoder math: LDM/taming
        // (`decoder.up.{level}.block.{r}` — what BFL ships and ComfyUI runs)
        // and diffusers (`decoder.up_blocks.{i}.resnets.{r}`). Detect and map
        // both onto one processing-ordered block list; the arithmetic is
        // identical, only the tensor names differ (and the residual projection
        // is `nin_shortcut` in LDM, `conv_shortcut` in diffusers).
        let is_ldm = src
            .tensor_data("decoder.up.0.block.0.conv1.weight")
            .is_some();
        let shortcut = if is_ldm {
            "nin_shortcut"
        } else {
            "conv_shortcut"
        };
        let mut w = VaeDecoderWeights {
            config,
            conv_in_w: Tensor {
                data: vec![],
                rows: 0,
                cols: 0,
            },
            conv_in_b: Tensor {
                data: vec![],
                rows: 0,
                cols: 0,
            },
            mid_resnet: vec![],
            mid_attn: None,
            up_blocks: vec![],
            conv_norm_out_w: Tensor {
                data: vec![],
                rows: 0,
                cols: 0,
            },
            conv_norm_out_b: Tensor {
                data: vec![],
                rows: 0,
                cols: 0,
            },
            conv_out_w: Tensor {
                data: vec![],
                rows: 0,
                cols: 0,
            },
            conv_out_b: Tensor {
                data: vec![],
                rows: 0,
                cols: 0,
            },
            post_quant_conv: None,
            latent_norm,
        };
        let lc = w.config.latent_channels;
        w.conv_in_w = load_tensor(src, "decoder.conv_in.weight", block_in, lc * 9)?;
        w.conv_in_b = load_tensor(src, "decoder.conv_in.bias", block_in, 1)?;
        let (mid_r0, mid_r1) = if is_ldm {
            ("decoder.mid.block_1", "decoder.mid.block_2")
        } else {
            ("decoder.mid_block.resnets.0", "decoder.mid_block.resnets.1")
        };
        w.mid_resnet
            .push(load_resnet(src, mid_r0, block_in, block_in, shortcut)?);
        w.mid_resnet
            .push(load_resnet(src, mid_r1, block_in, block_in, shortcut)?);
        if w.config.mid_block_add_attention {
            let c = block_in;
            let (gn, q, k, v, o) = if is_ldm {
                (
                    "decoder.mid.attn_1.norm",
                    "decoder.mid.attn_1.q",
                    "decoder.mid.attn_1.k",
                    "decoder.mid.attn_1.v",
                    "decoder.mid.attn_1.proj_out",
                )
            } else {
                (
                    "decoder.mid_block.attentions.0.group_norm",
                    "decoder.mid_block.attentions.0.to_q",
                    "decoder.mid_block.attentions.0.to_k",
                    "decoder.mid_block.attentions.0.to_v",
                    "decoder.mid_block.attentions.0.to_out.0",
                )
            };
            w.mid_attn = Some(MidAttn {
                group_norm_w: load_tensor(src, &format!("{gn}.weight"), c, 1)?,
                group_norm_b: load_tensor(src, &format!("{gn}.bias"), c, 1)?,
                q_w: load_tensor(src, &format!("{q}.weight"), c, c)?,
                q_b: load_tensor(src, &format!("{q}.bias"), c, 1)?,
                k_w: load_tensor(src, &format!("{k}.weight"), c, c)?,
                k_b: load_tensor(src, &format!("{k}.bias"), c, 1)?,
                v_w: load_tensor(src, &format!("{v}.weight"), c, c)?,
                v_b: load_tensor(src, &format!("{v}.bias"), c, 1)?,
                out_w: load_tensor(src, &format!("{o}.weight"), c, c)?,
                out_b: load_tensor(src, &format!("{o}.bias"), c, 1)?,
            });
        }
        // Up blocks, stored in PROCESSING order (first-applied → last-applied).
        // `prev_out` is the running channel entering each block; the first
        // resnet of a block takes `prev_out → block_out` (a channel change
        // when they differ) and the rest preserve `block_out`. Both namings
        // describe the same walk: LDM levels high→low, diffusers `up_blocks`
        // 0→nb-1, so a single `step` index drives either.
        let mut blocks = Vec::with_capacity(nb);
        let mut prev_out = block_in;
        for step in 0..nb {
            let level = nb - 1 - step; // LDM level for this processing step
            let block_out = w.config.block_out_channels[level];
            let mut resnets = Vec::with_capacity(w.config.layers_per_block + 1);
            for r in 0..w.config.layers_per_block + 1 {
                let in_ch = if r == 0 { prev_out } else { block_out };
                let prefix = if is_ldm {
                    format!("decoder.up.{level}.block.{r}")
                } else {
                    format!("decoder.up_blocks.{step}.resnets.{r}")
                };
                resnets.push(load_resnet(src, &prefix, in_ch, block_out, shortcut)?);
                prev_out = block_out;
            }
            let has_upsample = level != 0; // == step != nb - 1
            let (uw, ub) = if has_upsample {
                let p = if is_ldm {
                    format!("decoder.up.{level}.upsample.conv")
                } else {
                    format!("decoder.up_blocks.{step}.upsamplers.0.conv")
                };
                (
                    Some(load_tensor(
                        src,
                        &format!("{p}.weight"),
                        block_out,
                        block_out * 9,
                    )?),
                    Some(load_tensor(src, &format!("{p}.bias"), block_out, 1)?),
                )
            } else {
                (None, None)
            };
            blocks.push(UpBlock {
                channels: block_out,
                resnets,
                upsample_w: uw,
                upsample_b: ub,
            });
        }
        w.up_blocks = blocks;
        let norm_out = if is_ldm {
            "decoder.norm_out"
        } else {
            "decoder.conv_norm_out"
        };
        w.conv_norm_out_w = load_tensor(src, &format!("{norm_out}.weight"), first_ch, 1)?;
        w.conv_norm_out_b = load_tensor(src, &format!("{norm_out}.bias"), first_ch, 1)?;
        w.conv_out_w = load_tensor(
            src,
            "decoder.conv_out.weight",
            w.config.out_channels,
            first_ch * 9,
        )?;
        w.conv_out_b = load_tensor(src, "decoder.conv_out.bias", w.config.out_channels, 1)?;
        if w.config.use_post_quant_conv {
            let pqw = load_tensor(src, "post_quant_conv.weight", lc, lc)?;
            let pqb = load_tensor(src, "post_quant_conv.bias", lc, 1)?;
            w.post_quant_conv = Some((pqw, pqb));
        }
        Ok(w)
    }
}

/// One taming `ResnetBlock` (spatial dims preserved; channels may change).
fn resnet_forward(
    x: &[f32],
    h: usize,
    w: usize,
    r: &VaeResnet,
    groups: usize,
    eps: f32,
) -> Vec<f32> {
    let in_ch = r.in_ch;
    let out_ch = r.out_ch;
    let n1 = nn::groupnorm(
        x,
        in_ch,
        h,
        w,
        groups,
        &r.norm1_w.data,
        &r.norm1_b.data,
        eps,
    );
    let a1: Vec<f32> = n1.iter().map(|&v| nn::silu(v)).collect();
    let (c1, _, _) = nn::conv2d(
        &a1,
        in_ch,
        h,
        w,
        &r.conv1_w.data,
        out_ch,
        3,
        3,
        Some(&r.conv1_b.data),
        1,
    );
    let n2 = nn::groupnorm(
        &c1,
        out_ch,
        h,
        w,
        groups,
        &r.norm2_w.data,
        &r.norm2_b.data,
        eps,
    );
    let a2: Vec<f32> = n2.iter().map(|&v| nn::silu(v)).collect();
    let (c2, _, _) = nn::conv2d(
        &a2,
        out_ch,
        h,
        w,
        &r.conv2_w.data,
        out_ch,
        3,
        3,
        Some(&r.conv2_b.data),
        1,
    );
    // Residual: identity when channels match, else a 1×1 nin_shortcut.
    let residual: Vec<f32> = if in_ch == out_ch {
        x.to_vec()
    } else {
        let sw = r
            .nin_shortcut_w
            .as_ref()
            .expect("channel change needs nin_shortcut");
        let sb = r
            .nin_shortcut_b
            .as_ref()
            .expect("channel change needs nin_shortcut");
        nn::conv2d(x, in_ch, h, w, &sw.data, out_ch, 1, 1, Some(&sb.data), 0).0
    };
    residual.iter().zip(c2.iter()).map(|(a, b)| a + b).collect()
}

/// taming `Upsample`: nearest 2× then a 3×3 pad-1 conv (resamp_with_conv).
fn upsample_conv(
    x: &[f32],
    c: usize,
    h: usize,
    w: usize,
    cw: &Tensor,
    cb: &Tensor,
) -> (Vec<f32>, usize, usize) {
    let (up, oh, ow) = nn::upsample_nearest2x(x, c, h, w);
    nn::conv2d(&up, c, oh, ow, &cw.data, c, 3, 3, Some(&cb.data), 1)
}

/// Stage-by-stage VAE decode (the parity-bisection dump and [`decode`] share
/// one implementation; `decode` only reads `out`).
pub struct VaeStages {
    pub conv_in: Vec<f32>,
    pub mid: Vec<f32>,
    pub mid_r0: Vec<f32>,
    pub mid_attn: Vec<f32>,
    pub mid_r1: Vec<f32>,
    pub up: Vec<f32>,
    pub out: Vec<f32>,
}

pub fn decode_stages(
    weights: &VaeDecoderWeights,
    z: &[f32],
    h_in: usize,
    w_in: usize,
) -> VaeStages {
    let cfg = &weights.config;
    let block_in = cfg.block_out_channels[cfg.block_out_channels.len() - 1];
    let eps = 1e-6;
    let groups = cfg.norm_num_groups;
    // FLUX.2 Klein applies a 1×1 latent→latent `post_quant_conv` before
    // `conv_in`; FLUX.1 has none (`weights.post_quant_conv` is `None`).
    let post_quant;
    let z = if let Some((pqw, pqb)) = &weights.post_quant_conv {
        let lc = cfg.latent_channels;
        let (out, _, _) = nn::conv2d(z, lc, h_in, w_in, &pqw.data, lc, 1, 1, Some(&pqb.data), 0);
        post_quant = out;
        &post_quant[..]
    } else {
        z
    };
    let (mut hidden, mut h, mut wd) = nn::conv2d(
        z,
        cfg.latent_channels,
        h_in,
        w_in,
        &weights.conv_in_w.data,
        block_in,
        3,
        3,
        Some(&weights.conv_in_b.data),
        1,
    );
    let conv_in = hidden.clone();
    hidden = resnet_forward(&hidden, h, wd, &weights.mid_resnet[0], groups, eps);
    let mid_r0 = hidden.clone();
    if let Some(a) = &weights.mid_attn {
        hidden = mid_attn_forward(&hidden, block_in, h, wd, a, groups, eps);
    }
    let mid_attn = hidden.clone();
    hidden = resnet_forward(&hidden, h, wd, &weights.mid_resnet[1], groups, eps);
    let mid_r1 = hidden.clone();
    let mid = hidden.clone();
    // Up blocks in processing order (the loader already resolved the
    // LDM high→low / diffusers 0→nb-1 walk into this order).
    for block in &weights.up_blocks {
        for r in &block.resnets {
            hidden = resnet_forward(&hidden, h, wd, r, groups, eps);
        }
        if let (Some(cw), Some(cb)) = (&block.upsample_w, &block.upsample_b) {
            let (up, oh, ow) = upsample_conv(&hidden, block.channels, h, wd, cw, cb);
            hidden = up;
            h = oh;
            wd = ow;
        }
    }
    let up = hidden.clone();
    let first_ch = cfg.block_out_channels[0];
    hidden = nn::groupnorm(
        &hidden,
        first_ch,
        h,
        wd,
        groups,
        &weights.conv_norm_out_w.data,
        &weights.conv_norm_out_b.data,
        eps,
    );
    hidden = hidden.iter().map(|&v| nn::silu(v)).collect();
    let (out, _, _) = nn::conv2d(
        &hidden,
        first_ch,
        h,
        wd,
        &weights.conv_out_w.data,
        cfg.out_channels,
        3,
        3,
        Some(&weights.conv_out_b.data),
        1,
    );
    VaeStages {
        conv_in,
        mid,
        mid_r0,
        mid_attn,
        mid_r1,
        up,
        out,
    }
}

/// VAE decode: `latents [latent_channels][h][w]` → `[out_channels][h][w]`.
pub fn decode(weights: &VaeDecoderWeights, z: &[f32], h_in: usize, w_in: usize) -> Vec<f32> {
    decode_stages(weights, z, h_in, w_in).out
}

/// Mid-block self-attention (single head, head_dim = channels).
///
/// taming feeds (B, C, H, W) to `AttnBlock`, which reshapes to (B, HW, C):
/// the 1×1 projections act on the CHANNEL axis per spatial position. The Rust
/// input here is channel-major `[c][h][w]`, so it is transposed to
/// position-major `[n][c]` before the linears, and the residual-add path goes
/// back through a (HW, C) → (C, HW) transpose.
fn mid_attn_forward(
    x: &[f32],
    c: usize,
    h: usize,
    w: usize,
    a: &MidAttn,
    groups: usize,
    eps: f32,
) -> Vec<f32> {
    let n = h * w;
    let normed = nn::groupnorm(
        x,
        c,
        h,
        w,
        groups,
        &a.group_norm_w.data,
        &a.group_norm_b.data,
        eps,
    );
    // channel-major [c][n] → position-major [n][c]
    let mut flat = vec![0f32; n * c];
    for ch in 0..c {
        for p in 0..n {
            flat[p * c + ch] = normed[ch * n + p];
        }
    }
    let q = nn::linear(&flat, n, c, &a.q_w, Some(&a.q_b));
    let k = nn::linear(&flat, n, c, &a.k_w, Some(&a.k_b));
    let v = nn::linear(&flat, n, c, &a.v_w, Some(&a.v_b));
    let scale = 1.0 / (c as f32).sqrt();
    let mut scores = vec![0f32; n * n];
    for qp in 0..n {
        for kp in 0..n {
            let mut acc = 0f32;
            for t in 0..c {
                acc += q[qp * c + t] * k[kp * c + t];
            }
            scores[qp * n + kp] = acc * scale;
        }
    }
    let probs = nn::softmax_rows(&scores, n, n);
    let mut ctx = vec![0f32; n * c];
    for qp in 0..n {
        for t in 0..c {
            let mut acc = 0f32;
            for kp in 0..n {
                acc += probs[qp * n + kp] * v[kp * c + t];
            }
            ctx[qp * c + t] = acc;
        }
    }
    let out = nn::linear(&ctx, n, c, &a.out_w, Some(&a.out_b));
    // (HW, C) → (C, HW) transpose before the residual add.
    let mut res = vec![0f32; x.len()];
    for p in 0..n {
        for ch in 0..c {
            res[ch * n + p] = x[ch * n + p] + out[p * c + ch];
        }
    }
    res
}

// ─── VAE encoder (diffusers `AutoencoderKL.Encoder` layout only) ───────────
//
// Pixels → latent moments: `conv_in` → `down_blocks.{i}` (each
// `layers_per_block` resnets, then a strided-conv downsampler for every
// block but the last) → `mid_block` (resnet, optional attention, resnet) →
// `conv_norm_out` → SiLU → `conv_out` (→ `2 * latent_channels`) → optional
// `quant_conv` (1×1, latent moments → latent moments). `encode` returns the
// mean half of the moments (argmax sample mode; it never draws a sample).
//
// The encoder is diffusers-only: FLUX ships an LDM decoder but a diffusers
// encoder, so there is no `encoder.down.*`/`nin_shortcut` naming to detect.

/// One `down_blocks.{i}`: `layers_per_block` resnets, then an optional
/// strided-conv downsampler (absent only on the last block).
#[derive(Debug, Clone)]
pub struct DownBlock {
    pub channels: usize,
    pub resnets: Vec<VaeResnet>,
    pub downsample_w: Option<Tensor>, // [channels, channels, 3, 3]
    pub downsample_b: Option<Tensor>,
}

/// Encoder weights (row-major f32), diffusers `AutoencoderKL.Encoder` naming.
#[derive(Debug, Clone)]
pub struct VaeEncoderWeights {
    pub config: VaeConfig,
    pub conv_in_w: Tensor, // [block_out[0], in_channels, 3, 3]
    pub conv_in_b: Tensor,
    /// Indexed 0..nb, processed in order (spatial ÷2 at every downsampler).
    pub down_blocks: Vec<DownBlock>,
    /// `mid_block.resnets.0/1` (channel-preserving at `block_out[nb-1]`).
    pub mid_resnet: Vec<VaeResnet>,
    /// `mid_block.attentions.0` (present iff `mid_block_add_attention`).
    pub mid_attn: Option<MidAttn>,
    pub conv_norm_out_w: Tensor, // [block_out[nb-1]]
    pub conv_norm_out_b: Tensor,
    pub conv_out_w: Tensor, // [2*latent, block_out[nb-1], 3, 3]
    pub conv_out_b: Tensor,
    /// Top-level `quant_conv` (1×1, `2*latent → 2*latent`), present iff
    /// `config.use_quant_conv`.
    pub quant_conv: Option<(Tensor, Tensor)>,
}

impl VaeEncoderWeights {
    /// Load the diffusers-layout encoder. Fails with a clear error if
    /// `encoder.conv_in.weight` is absent (an LDM-layout encoder is out of
    /// scope: FLUX ships an LDM *decoder* but a diffusers *encoder*).
    pub fn load(src: &dyn ModelSourceTrait) -> Result<Self, String> {
        let v: serde_json::Value = serde_json::from_str(src.metadata_json())
            .map_err(|e| format!("vae: config.json invalid: {e}"))?;
        let v = v.get("config").cloned().unwrap_or(v);
        let config = VaeConfig::from_json(&v)?;
        if src.tensor_data("encoder.conv_in.weight").is_none() {
            return Err(
                "vae encoder: missing `encoder.conv_in.weight` (only the diffusers \
                 encoder layout is supported, no LDM `encoder.down.*` fallback)"
                    .to_string(),
            );
        }
        let nb = config.block_out_channels.len();
        let first_ch = config.block_out_channels[0];
        let last_ch = config.block_out_channels[nb - 1];
        let lc = config.latent_channels;
        let two = 2 * lc;
        const SHORTCUT: &str = "conv_shortcut";

        let conv_in_w = load_tensor(
            src,
            "encoder.conv_in.weight",
            first_ch,
            config.in_channels * 9,
        )?;
        let conv_in_b = load_tensor(src, "encoder.conv_in.bias", first_ch, 1)?;

        let mut down_blocks = Vec::with_capacity(nb);
        for i in 0..nb {
            let out_ch = config.block_out_channels[i];
            let mut resnets = Vec::with_capacity(config.layers_per_block);
            for r in 0..config.layers_per_block {
                let in_ch = if r == 0 {
                    if i == 0 {
                        first_ch
                    } else {
                        config.block_out_channels[i - 1]
                    }
                } else {
                    out_ch
                };
                let prefix = format!("encoder.down_blocks.{i}.resnets.{r}");
                resnets.push(load_resnet(src, &prefix, in_ch, out_ch, SHORTCUT)?);
            }
            let has_downsample = i < nb - 1;
            let (dw, db) = if has_downsample {
                let p = format!("encoder.down_blocks.{i}.downsamplers.0.conv");
                (
                    Some(load_tensor(
                        src,
                        &format!("{p}.weight"),
                        out_ch,
                        out_ch * 9,
                    )?),
                    Some(load_tensor(src, &format!("{p}.bias"), out_ch, 1)?),
                )
            } else {
                (None, None)
            };
            down_blocks.push(DownBlock {
                channels: out_ch,
                resnets,
                downsample_w: dw,
                downsample_b: db,
            });
        }

        let mid_resnet = vec![
            load_resnet(
                src,
                "encoder.mid_block.resnets.0",
                last_ch,
                last_ch,
                SHORTCUT,
            )?,
            load_resnet(
                src,
                "encoder.mid_block.resnets.1",
                last_ch,
                last_ch,
                SHORTCUT,
            )?,
        ];

        let mid_attn = if config.mid_block_add_attention {
            let c = last_ch;
            let gn = "encoder.mid_block.attentions.0.group_norm";
            let q = "encoder.mid_block.attentions.0.to_q";
            let k = "encoder.mid_block.attentions.0.to_k";
            let v = "encoder.mid_block.attentions.0.to_v";
            let o = "encoder.mid_block.attentions.0.to_out.0";
            Some(MidAttn {
                group_norm_w: load_tensor(src, &format!("{gn}.weight"), c, 1)?,
                group_norm_b: load_tensor(src, &format!("{gn}.bias"), c, 1)?,
                q_w: load_tensor(src, &format!("{q}.weight"), c, c)?,
                q_b: load_tensor(src, &format!("{q}.bias"), c, 1)?,
                k_w: load_tensor(src, &format!("{k}.weight"), c, c)?,
                k_b: load_tensor(src, &format!("{k}.bias"), c, 1)?,
                v_w: load_tensor(src, &format!("{v}.weight"), c, c)?,
                v_b: load_tensor(src, &format!("{v}.bias"), c, 1)?,
                out_w: load_tensor(src, &format!("{o}.weight"), c, c)?,
                out_b: load_tensor(src, &format!("{o}.bias"), c, 1)?,
            })
        } else {
            None
        };

        let conv_norm_out_w = load_tensor(src, "encoder.conv_norm_out.weight", last_ch, 1)?;
        let conv_norm_out_b = load_tensor(src, "encoder.conv_norm_out.bias", last_ch, 1)?;
        let conv_out_w = load_tensor(src, "encoder.conv_out.weight", two, last_ch * 9)?;
        let conv_out_b = load_tensor(src, "encoder.conv_out.bias", two, 1)?;
        let quant_conv = if config.use_quant_conv {
            Some((
                load_tensor(src, "quant_conv.weight", two, two)?,
                load_tensor(src, "quant_conv.bias", two, 1)?,
            ))
        } else {
            None
        };

        Ok(VaeEncoderWeights {
            config,
            conv_in_w,
            conv_in_b,
            down_blocks,
            mid_resnet,
            mid_attn,
            conv_norm_out_w,
            conv_norm_out_b,
            conv_out_w,
            conv_out_b,
            quant_conv,
        })
    }
}

#[cfg(test)]
const VAE_ENCODER_SYNTH_SEED: u64 = 0xC0FF_EE00_0000_0042; // encoder, distinct from decoder/transformer seeds

#[cfg(test)]
impl VaeEncoderWeights {
    /// Deterministic synthetic weights matching [`Self::load`]'s tensor
    /// order and shapes exactly — the self-parity substrate (no real
    /// checkpoint needed), same `synth_val` sequence as `FluxWeights::synthetic`.
    pub fn synthetic(cfg: &VaeConfig) -> Self {
        struct Gen(u64);
        impl Gen {
            fn vec(&mut self, n: usize) -> Vec<f32> {
                let data: Vec<f32> = (0..n as u64)
                    .map(|i| crate::flux::synth_val(VAE_ENCODER_SYNTH_SEED, self.0 + i) * 0.05)
                    .collect();
                self.0 += n as u64;
                data
            }
            fn tensor(&mut self, rows: usize, cols: usize) -> Tensor {
                Tensor {
                    data: self.vec(rows * cols),
                    rows,
                    cols,
                }
            }
        }
        fn synth_resnet(g: &mut Gen, in_ch: usize, out_ch: usize) -> VaeResnet {
            let nin = if in_ch == out_ch {
                (None, None)
            } else {
                (Some(g.tensor(out_ch, in_ch)), Some(g.tensor(out_ch, 1)))
            };
            VaeResnet {
                in_ch,
                out_ch,
                norm1_w: g.tensor(in_ch, 1),
                norm1_b: g.tensor(in_ch, 1),
                conv1_w: g.tensor(out_ch, in_ch * 9),
                conv1_b: g.tensor(out_ch, 1),
                norm2_w: g.tensor(out_ch, 1),
                norm2_b: g.tensor(out_ch, 1),
                conv2_w: g.tensor(out_ch, out_ch * 9),
                conv2_b: g.tensor(out_ch, 1),
                nin_shortcut_w: nin.0,
                nin_shortcut_b: nin.1,
            }
        }

        let mut g = Gen(0);
        let nb = cfg.block_out_channels.len();
        let first_ch = cfg.block_out_channels[0];
        let last_ch = cfg.block_out_channels[nb - 1];
        let two = 2 * cfg.latent_channels;

        let conv_in_w = g.tensor(first_ch, cfg.in_channels * 9);
        let conv_in_b = g.tensor(first_ch, 1);

        let mut down_blocks = Vec::with_capacity(nb);
        for i in 0..nb {
            let out_ch = cfg.block_out_channels[i];
            let mut resnets = Vec::with_capacity(cfg.layers_per_block);
            for r in 0..cfg.layers_per_block {
                let in_ch = if r == 0 {
                    if i == 0 {
                        first_ch
                    } else {
                        cfg.block_out_channels[i - 1]
                    }
                } else {
                    out_ch
                };
                resnets.push(synth_resnet(&mut g, in_ch, out_ch));
            }
            let has_downsample = i < nb - 1;
            let (dw, db) = if has_downsample {
                (
                    Some(g.tensor(out_ch, out_ch * 9)),
                    Some(g.tensor(out_ch, 1)),
                )
            } else {
                (None, None)
            };
            down_blocks.push(DownBlock {
                channels: out_ch,
                resnets,
                downsample_w: dw,
                downsample_b: db,
            });
        }

        let mid_resnet = vec![
            synth_resnet(&mut g, last_ch, last_ch),
            synth_resnet(&mut g, last_ch, last_ch),
        ];

        let mid_attn = if cfg.mid_block_add_attention {
            let c = last_ch;
            Some(MidAttn {
                group_norm_w: g.tensor(c, 1),
                group_norm_b: g.tensor(c, 1),
                q_w: g.tensor(c, c),
                q_b: g.tensor(c, 1),
                k_w: g.tensor(c, c),
                k_b: g.tensor(c, 1),
                v_w: g.tensor(c, c),
                v_b: g.tensor(c, 1),
                out_w: g.tensor(c, c),
                out_b: g.tensor(c, 1),
            })
        } else {
            None
        };

        let conv_norm_out_w = g.tensor(last_ch, 1);
        let conv_norm_out_b = g.tensor(last_ch, 1);
        let conv_out_w = g.tensor(two, last_ch * 9);
        let conv_out_b = g.tensor(two, 1);
        let quant_conv = if cfg.use_quant_conv {
            Some((g.tensor(two, two), g.tensor(two, 1)))
        } else {
            None
        };

        VaeEncoderWeights {
            config: cfg.clone(),
            conv_in_w,
            conv_in_b,
            down_blocks,
            mid_resnet,
            mid_attn,
            conv_norm_out_w,
            conv_norm_out_b,
            conv_out_w,
            conv_out_b,
            quant_conv,
        }
    }
}

/// VAE encode: pixels `[in_channels][h][w]` (range `[-1, 1]`) → latent mean
/// `[latent_channels][h/8][w/8]`. Always returns the mean half of the
/// moments (argmax sample mode); never draws a sample from the logvar half.
pub fn encode(weights: &VaeEncoderWeights, x: &[f32], h: usize, w: usize) -> Vec<f32> {
    let cfg = &weights.config;
    let groups = cfg.norm_num_groups;
    let eps = 1e-6;
    let (mut cur, mut ch, mut hh, mut ww) = {
        let (y, oh, ow) = nn::conv2d(
            x,
            cfg.in_channels,
            h,
            w,
            &weights.conv_in_w.data,
            cfg.block_out_channels[0],
            3,
            3,
            Some(&weights.conv_in_b.data),
            1,
        );
        (y, cfg.block_out_channels[0], oh, ow)
    };
    for blk in &weights.down_blocks {
        for r in &blk.resnets {
            cur = resnet_forward(&cur, hh, ww, r, groups, eps);
            ch = r.out_ch;
        }
        if let (Some(dw), Some(db)) = (&blk.downsample_w, &blk.downsample_b) {
            // diffusers `Downsample2D`: asymmetric pad (0,1,0,1) then a 3×3
            // stride-2 conv, pad 0 elsewhere — halves H and W.
            let (y, oh, ow) = nn::conv2d_strided(
                &cur,
                ch,
                hh,
                ww,
                &dw.data,
                ch,
                3,
                3,
                Some(&db.data),
                2,
                0,
                0,
                1,
                1,
            );
            cur = y;
            hh = oh;
            ww = ow;
        }
    }
    cur = resnet_forward(&cur, hh, ww, &weights.mid_resnet[0], groups, eps);
    if let Some(a) = &weights.mid_attn {
        cur = mid_attn_forward(&cur, ch, hh, ww, a, groups, eps);
    }
    cur = resnet_forward(&cur, hh, ww, &weights.mid_resnet[1], groups, eps);
    let n = nn::groupnorm(
        &cur,
        ch,
        hh,
        ww,
        groups,
        &weights.conv_norm_out_w.data,
        &weights.conv_norm_out_b.data,
        eps,
    );
    let n: Vec<f32> = n.iter().map(|v| nn::silu(*v)).collect();
    let two = 2 * cfg.latent_channels;
    let (mut moments, _, _) = nn::conv2d(
        &n,
        ch,
        hh,
        ww,
        &weights.conv_out_w.data,
        two,
        3,
        3,
        Some(&weights.conv_out_b.data),
        1,
    );
    if let Some((qw, qb)) = &weights.quant_conv {
        moments = nn::conv2d(
            &moments,
            two,
            hh,
            ww,
            &qw.data,
            two,
            1,
            1,
            Some(&qb.data),
            0,
        )
        .0;
    }
    // mean = first half of the channels (the logvar half is dropped: argmax
    // sample mode never draws a sample).
    moments[..cfg.latent_channels * hh * ww].to_vec()
}

/// On-disk fixture writer for a synthetic VAE component — the encoder half,
/// the decoder half, both 1×1 quant convs and the FLUX.2 latent BatchNorm
/// statistics, under the exact names [`VaeEncoderWeights::load`] and
/// [`VaeDecoderWeights::load`] read.
///
/// It walks the SAME block/resnet structure those two loaders walk, so a
/// shape or naming change there fails this fixture loudly instead of leaving
/// a stale hand-written key list behind.
#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::VaeConfig;
    use crate::flux::test_fixtures::{bf16_blob, shape_of, NamedTensor};

    /// Distinct from the encoder/decoder synthetic seeds so a fixture pipe's
    /// VAE never coincidentally matches a `synthetic()` table.
    const VAE_FIXTURE_SEED: u64 = 0xC0FF_EE00_0000_0043;

    struct Writer {
        out: Vec<NamedTensor>,
        idx: u64,
    }

    impl Writer {
        fn mat(&mut self, name: String, rows: usize, cols: usize) {
            let data = bf16_blob(VAE_FIXTURE_SEED, &mut self.idx, rows * cols);
            self.out
                .push((name, data, "BF16".into(), shape_of(rows, cols)));
        }
        /// `weight [rows, cols]` + `bias [rows]`, the pair every conv and
        /// norm in this component ships.
        fn conv(&mut self, prefix: &str, rows: usize, cols: usize) {
            self.mat(format!("{prefix}.weight"), rows, cols);
            self.mat(format!("{prefix}.bias"), rows, 1);
        }
        /// One resnet, in `load_resnet`'s read order.
        fn resnet(&mut self, prefix: &str, in_ch: usize, out_ch: usize) {
            if in_ch != out_ch {
                self.conv(&format!("{prefix}.conv_shortcut"), out_ch, in_ch);
            }
            self.conv(&format!("{prefix}.norm1"), in_ch, 1);
            self.conv(&format!("{prefix}.conv1"), out_ch, in_ch * 9);
            self.conv(&format!("{prefix}.norm2"), out_ch, 1);
            self.conv(&format!("{prefix}.conv2"), out_ch, out_ch * 9);
        }
        /// The mid-block self-attention (`group_norm` + q/k/v/out 1×1s).
        fn mid_attn(&mut self, prefix: &str, c: usize) {
            self.conv(&format!("{prefix}.group_norm"), c, 1);
            for p in ["to_q", "to_k", "to_v", "to_out.0"] {
                self.conv(&format!("{prefix}.{p}"), c, c);
            }
        }
        /// An F32 vector written verbatim — the BatchNorm statistics are read
        /// as running moments, so they need real values, not noise.
        fn f32_vec(&mut self, name: &str, values: &[f32]) {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            self.out
                .push((name.into(), data, "F32".into(), vec![values.len()]));
        }
    }

    /// Every tensor a full (encoder + decoder + BatchNorm) VAE component
    /// carries for `cfg`.
    pub(crate) fn checkpoint_tensors(cfg: &VaeConfig) -> Vec<NamedTensor> {
        let mut w = Writer {
            out: Vec::new(),
            idx: 0,
        };
        let nb = cfg.block_out_channels.len();
        let first_ch = cfg.block_out_channels[0];
        let last_ch = cfg.block_out_channels[nb - 1];
        let lc = cfg.latent_channels;
        let two = 2 * lc;

        // ── encoder (diffusers layout) ────────────────────────────────
        w.conv("encoder.conv_in", first_ch, cfg.in_channels * 9);
        for i in 0..nb {
            let out_ch = cfg.block_out_channels[i];
            for r in 0..cfg.layers_per_block {
                let in_ch = match (r, i) {
                    (0, 0) => first_ch,
                    (0, _) => cfg.block_out_channels[i - 1],
                    _ => out_ch,
                };
                w.resnet(
                    &format!("encoder.down_blocks.{i}.resnets.{r}"),
                    in_ch,
                    out_ch,
                );
            }
            if i < nb - 1 {
                w.conv(
                    &format!("encoder.down_blocks.{i}.downsamplers.0.conv"),
                    out_ch,
                    out_ch * 9,
                );
            }
        }
        w.resnet("encoder.mid_block.resnets.0", last_ch, last_ch);
        w.resnet("encoder.mid_block.resnets.1", last_ch, last_ch);
        if cfg.mid_block_add_attention {
            w.mid_attn("encoder.mid_block.attentions.0", last_ch);
        }
        w.conv("encoder.conv_norm_out", last_ch, 1);
        w.conv("encoder.conv_out", two, last_ch * 9);
        if cfg.use_quant_conv {
            w.conv("quant_conv", two, two);
        }

        // ── decoder (diffusers layout: no `decoder.up.*` keys) ────────
        w.conv("decoder.conv_in", last_ch, lc * 9);
        w.resnet("decoder.mid_block.resnets.0", last_ch, last_ch);
        w.resnet("decoder.mid_block.resnets.1", last_ch, last_ch);
        if cfg.mid_block_add_attention {
            w.mid_attn("decoder.mid_block.attentions.0", last_ch);
        }
        let mut prev_out = last_ch;
        for step in 0..nb {
            let level = nb - 1 - step;
            let block_out = cfg.block_out_channels[level];
            for r in 0..cfg.layers_per_block + 1 {
                let in_ch = if r == 0 { prev_out } else { block_out };
                w.resnet(
                    &format!("decoder.up_blocks.{step}.resnets.{r}"),
                    in_ch,
                    block_out,
                );
                prev_out = block_out;
            }
            if level != 0 {
                w.conv(
                    &format!("decoder.up_blocks.{step}.upsamplers.0.conv"),
                    block_out,
                    block_out * 9,
                );
            }
        }
        w.conv("decoder.conv_norm_out", first_ch, 1);
        w.conv("decoder.conv_out", cfg.out_channels, first_ch * 9);
        if cfg.use_post_quant_conv {
            w.conv("post_quant_conv", lc, lc);
        }

        // ── FLUX.2 latent BatchNorm statistics ────────────────────────
        // Only when the config says the latent is BatchNorm-normalized
        // (`scaling_factor` absent) — a FLUX.1 fixture must NOT carry these,
        // or `load_latent_norm` would take the wrong branch.
        if cfg.scaling_factor.is_none() {
            let n = lc * cfg.latent_patch.0 * cfg.latent_patch.1;
            let mean: Vec<f32> = (0..n).map(|c| 0.05 * c as f32 - 0.1).collect();
            let var: Vec<f32> = (0..n).map(|c| 1.0 + 0.1 * c as f32).collect();
            w.f32_vec("bn.running_mean", &mean);
            w.f32_vec("bn.running_var", &var);
        }
        w.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flux::Tensor;

    fn t(rows: usize, cols: usize) -> Tensor {
        Tensor {
            data: vec![0.01; rows * cols],
            rows,
            cols,
        }
    }

    fn resnet(in_ch: usize, out_ch: usize, with_shortcut: bool) -> VaeResnet {
        VaeResnet {
            in_ch,
            out_ch,
            norm1_w: t(in_ch, 1),
            norm1_b: t(in_ch, 1),
            conv1_w: t(out_ch, in_ch * 9),
            conv1_b: t(out_ch, 1),
            norm2_w: t(out_ch, 1),
            norm2_b: t(out_ch, 1),
            conv2_w: t(out_ch, out_ch * 9),
            conv2_b: t(out_ch, 1),
            nin_shortcut_w: if with_shortcut {
                Some(t(out_ch, in_ch))
            } else {
                None
            },
            nin_shortcut_b: if with_shortcut {
                Some(t(out_ch, 1))
            } else {
                None
            },
        }
    }

    #[test]
    fn channel_preserving_resnet_is_shape_preserving() {
        let r = resnet(4, 4, false);
        let x = vec![0.5f32; 4 * 4 * 4];
        let y = resnet_forward(&x, 4, 4, &r, 1, 1e-6);
        assert_eq!(y.len(), 4 * 4 * 4);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn channel_halving_resnet_uses_nin_shortcut_and_keeps_spatial() {
        // First resnet of an up block: in 8 → out 4, spatial preserved.
        let r = resnet(8, 4, true);
        let x = vec![0.5f32; 8 * 6 * 6];
        let y = resnet_forward(&x, 6, 6, &r, 2, 1e-6);
        assert_eq!(y.len(), 4 * 6 * 6);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[should_panic(expected = "channel change needs nin_shortcut")]
    fn channel_change_without_shortcut_panics() {
        let r = resnet(8, 4, false);
        let x = vec![0.5f32; 8 * 4 * 4];
        let _ = resnet_forward(&x, 4, 4, &r, 2, 1e-6);
    }

    #[test]
    fn upsample_conv_doubles_spatial_and_keeps_channels() {
        let x = vec![0.25f32; 4 * 3 * 3];
        let (y, oh, ow) = upsample_conv(&x, 4, 3, 3, &t(4, 36), &t(4, 1));
        assert_eq!((oh, ow), (6, 6));
        assert_eq!(y.len(), 4 * 6 * 6);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn flux2_vae_config_parses_bn_and_quant_conv_flags() {
        let v = serde_json::json!({
            "in_channels": 3, "out_channels": 3, "latent_channels": 32,
            "block_out_channels": [128, 256, 512, 512], "layers_per_block": 2,
            "norm_num_groups": 32, "mid_block_add_attention": true,
            "batch_norm_eps": 0.0001, "patch_size": [2, 2],
            "use_quant_conv": true, "use_post_quant_conv": true
        });
        let cfg = VaeConfig::from_json(&v).unwrap();
        assert_eq!(cfg.latent_channels, 32);
        assert_eq!(cfg.scaling_factor, None);
        assert!(cfg.use_post_quant_conv);
        assert_eq!(cfg.latent_patch, (2, 2));
        assert_eq!(cfg.batch_norm_eps, 1e-4);
    }

    #[test]
    fn encoder_on_a_tiny_synthetic_config_halves_three_times_and_returns_the_mean() {
        let cfg = VaeConfig {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 4,
            block_out_channels: vec![8, 8, 16, 16],
            layers_per_block: 1,
            norm_num_groups: 4,
            mid_block_add_attention: true,
            scaling_factor: None,
            shift_factor: None,
            use_quant_conv: true,
            use_post_quant_conv: true,
            batch_norm_eps: 1e-4,
            latent_patch: (2, 2),
        };
        let w = VaeEncoderWeights::synthetic(&cfg);
        let (h, wd) = (16, 24);
        let x: Vec<f32> = (0..3 * h * wd).map(|i| ((i as f32) * 0.01).sin()).collect();
        let z = encode(&w, &x, h, wd);
        assert_eq!(z.len(), 4 * (h / 8) * (wd / 8));
        assert!(z.iter().all(|v| v.is_finite()));

        // Guard against a dead conv path: two different inputs must not
        // collapse to the same latent.
        let zeros = vec![0.0f32; 3 * h * wd];
        let halves = vec![0.5f32; 3 * h * wd];
        let z_zeros = encode(&w, &zeros, h, wd);
        let z_halves = encode(&w, &halves, h, wd);
        assert_ne!(z_zeros, z_halves);
    }

    #[test]
    fn flux1_vae_config_keeps_scale_and_shift() {
        let v = serde_json::json!({
            "in_channels": 3, "out_channels": 3, "latent_channels": 16,
            "block_out_channels": [128, 256, 512, 512], "layers_per_block": 2,
            "norm_num_groups": 32, "scaling_factor": 0.3611, "shift_factor": 0.1159
        });
        let cfg = VaeConfig::from_json(&v).unwrap();
        assert_eq!(cfg.scaling_factor, Some(0.3611));
        assert!(!cfg.use_post_quant_conv);
        assert_eq!(cfg.latent_patch, (1, 1));
    }
}
