// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU CLIP text encoder reference (FLUX `vec` conditioning = the pooled
//! output), pinned to `transformers` 5.16 CLIPTextModel semantics — the
//! golden source:
//!
//! - token embedding + learned position embedding (0..len−1), no type ids.
//! - 5 pre-norm encoder layers: causal self-attention (scale 1/√head_dim,
//!   additive mask: future keys AND padded keys → −inf) + MLP
//!   (fc1 → exact-erf GELU → fc2).
//! - `pooled_output` = last_hidden_state at `argmax(input_ids)` when
//!   `eos_token_id == 2` (transformers 5 semantics: the EOT position —
//!   there is NO `text_projection` in this model class/version).
//!
//! Reference fixture: `text_encoder/` (hf-internal-testing tiny CLIP):
//! hidden 32, heads 4, head_dim 8, depth 5, intermediate 37, seq 77,
//! vocab 1000.

use crate::flux::Tensor;
use crate::nn;
use hipfire_runtime::model_source::ModelSource as ModelSourceTrait;
use serde::{Deserialize, Serialize};

/// Mask value for future/padded keys (−inf in fp32).
pub const CLIP_MASK: f32 = f32::NEG_INFINITY;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f32,
    /// MLP activation: `quick_gelu` (OpenAI CLIP default, x·σ(1.702x)) or
    /// `gelu`/`gelu_erf` (exact erf). The FLUX clip_l config ships
    /// `quick_gelu`; using erf instead silently diverges the pooled vector.
    pub hidden_act: String,
}

impl ClipConfig {
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let get = |k: &str| -> Result<usize, String> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| format!("clip config: missing `{k}`"))
        };
        Ok(Self {
            hidden_size: get("hidden_size")?,
            intermediate_size: get("intermediate_size")?,
            num_attention_heads: get("num_attention_heads")?,
            num_hidden_layers: get("num_hidden_layers")?,
            vocab_size: get("vocab_size")?,
            max_position_embeddings: get("max_position_embeddings")?,
            layer_norm_eps: v
                .get("layer_norm_eps")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(1e-5),
            hidden_act: v
                .get("hidden_act")
                .and_then(|x| x.as_str())
                .unwrap_or("quick_gelu")
                .to_string(),
        })
    }
}

/// CLIP text encoder weights (row-major f32). All linears carry bias.
#[derive(Debug, Clone)]
pub struct ClipWeights {
    pub config: ClipConfig,
    pub token_embed: Tensor, // [vocab, hidden]
    pub pos_embed: Tensor,   // [max_pos, hidden]
    pub ln1_w: Vec<Tensor>,
    pub ln1_b: Vec<Tensor>,
    pub q_w: Vec<Tensor>,
    pub q_b: Vec<Tensor>,
    pub k_w: Vec<Tensor>,
    pub k_b: Vec<Tensor>,
    pub v_w: Vec<Tensor>,
    pub v_b: Vec<Tensor>,
    pub out_w: Vec<Tensor>,
    pub out_b: Vec<Tensor>,
    pub ln2_w: Vec<Tensor>,
    pub ln2_b: Vec<Tensor>,
    pub fc1_w: Vec<Tensor>,
    pub fc1_b: Vec<Tensor>,
    pub fc2_w: Vec<Tensor>,
    pub fc2_b: Vec<Tensor>,
    pub final_ln_w: Tensor,
    pub final_ln_b: Tensor,
}

impl ClipWeights {
    pub fn load(src: &dyn ModelSourceTrait) -> Result<Self, String> {
        let v: serde_json::Value = serde_json::from_str(src.metadata_json())
            .map_err(|e| format!("clip: config.json invalid: {e}"))?;
        let v = v.get("config").cloned().unwrap_or(v);
        let config = ClipConfig::from_json(&v)?;
        let h = config.hidden_size;
        let load = |name: &str, rows: usize, cols: usize| -> Result<Tensor, String> {
            let (info, data) = src
                .tensor_data(name)
                .ok_or_else(|| format!("clip: missing tensor `{name}`"))?;
            let n = info.shape.iter().product::<usize>();
            if n != rows * cols {
                return Err(format!(
                    "clip: tensor `{name}` has {n} elems, expected {}",
                    rows * cols
                ));
            }
            Ok(Tensor {
                data: crate::flux::decode_dtype(&info.dtype, &data)?,
                rows,
                cols,
            })
        };
        let num_layers = config.num_hidden_layers;
        let intermediate = config.intermediate_size;
        let mut w = ClipWeights {
            token_embed: load(
                "text_model.embeddings.token_embedding.weight",
                config.vocab_size,
                h,
            )?,
            pos_embed: load(
                "text_model.embeddings.position_embedding.weight",
                config.max_position_embeddings,
                h,
            )?,
            ln1_w: vec![],
            ln1_b: vec![],
            q_w: vec![],
            q_b: vec![],
            k_w: vec![],
            k_b: vec![],
            v_w: vec![],
            v_b: vec![],
            out_w: vec![],
            out_b: vec![],
            ln2_w: vec![],
            ln2_b: vec![],
            fc1_w: vec![],
            fc1_b: vec![],
            fc2_w: vec![],
            fc2_b: vec![],
            final_ln_w: load("text_model.final_layer_norm.weight", h, 1)?,
            final_ln_b: load("text_model.final_layer_norm.bias", h, 1)?,
            config,
        };
        for i in 0..num_layers {
            let p = format!("text_model.encoder.layers.{i}.");
            w.ln1_w.push(load(&format!("{p}layer_norm1.weight"), h, 1)?);
            w.ln1_b.push(load(&format!("{p}layer_norm1.bias"), h, 1)?);
            w.q_w
                .push(load(&format!("{p}self_attn.q_proj.weight"), h, h)?);
            w.q_b
                .push(load(&format!("{p}self_attn.q_proj.bias"), h, 1)?);
            w.k_w
                .push(load(&format!("{p}self_attn.k_proj.weight"), h, h)?);
            w.k_b
                .push(load(&format!("{p}self_attn.k_proj.bias"), h, 1)?);
            w.v_w
                .push(load(&format!("{p}self_attn.v_proj.weight"), h, h)?);
            w.v_b
                .push(load(&format!("{p}self_attn.v_proj.bias"), h, 1)?);
            w.out_w
                .push(load(&format!("{p}self_attn.out_proj.weight"), h, h)?);
            w.out_b
                .push(load(&format!("{p}self_attn.out_proj.bias"), h, 1)?);
            w.ln2_w.push(load(&format!("{p}layer_norm2.weight"), h, 1)?);
            w.ln2_b.push(load(&format!("{p}layer_norm2.bias"), h, 1)?);
            w.fc1_w
                .push(load(&format!("{p}mlp.fc1.weight"), intermediate, h)?);
            w.fc1_b
                .push(load(&format!("{p}mlp.fc1.bias"), intermediate, 1)?);
            w.fc2_w
                .push(load(&format!("{p}mlp.fc2.weight"), h, intermediate)?);
            w.fc2_b.push(load(&format!("{p}mlp.fc2.bias"), h, 1)?);
        }
        Ok(w)
    }
}

/// CLIP text forward (batch 1). Returns `(last_hidden_state [len][hidden],
/// pooled [hidden])` plus per-layer hidden states for golden debugging.
///
/// IMPORTANT: the diffusers FLUX pipeline calls `CLIPTextModel(input_ids)`
/// WITHOUT `attention_mask`, so padded CLIP positions attend freely — the
/// mask is CAUSAL ONLY (`kv <= q`), no key-padding masking. This is a
/// diffusers-specific conditioning quirk that the golden pins (a masked run
/// diverges at every padded row); real T5/CLIP text encoders elsewhere in
/// the workspace must not inherit this by accident — it lives here because
/// FLUX conditioning is what this crate serves.
pub fn encode(
    w: &ClipWeights,
    input_ids: &[u32],
    attention_mask: &[u8],
) -> (Vec<f32>, Vec<f32>, Vec<Vec<f32>>) {
    let _ = attention_mask;
    let cfg = &w.config;
    let len = input_ids.len();
    let h = cfg.hidden_size;
    let heads = cfg.num_attention_heads;
    let hd = h / heads;
    let eps = cfg.layer_norm_eps;

    // token + position embeddings
    let mut hidden: Vec<f32> = vec![0f32; len * h];
    for (r, &tok) in input_ids.iter().enumerate() {
        let tok = tok as usize;
        for c in 0..h {
            hidden[r * h + c] = w.token_embed.data[tok * h + c] + w.pos_embed.data[r * h + c];
        }
    }

    // causal-only additive mask (diffusers drops the padding mask)
    let mut mask = vec![0f32; len * len];
    for q in 0..len {
        for k in 0..len {
            if k > q {
                mask[q * len + k] = CLIP_MASK;
            }
        }
    }

    let mut layer_outs: Vec<Vec<f32>> = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        // pre-norm self-attention
        let n1 = nn::layernorm_affine(&hidden, len, h, &w.ln1_w[i].data, &w.ln1_b[i].data, eps);
        let q = nn::linear(&n1, len, h, &w.q_w[i], Some(&w.q_b[i]));
        let k = nn::linear(&n1, len, h, &w.k_w[i], Some(&w.k_b[i]));
        let v = nn::linear(&n1, len, h, &w.v_w[i], Some(&w.v_b[i]));
        let scale = 1.0 / (hd as f32).sqrt();

        let mut ctx = vec![0f32; len * h];
        for hh in 0..heads {
            let off = hh * hd;
            let mut scores = vec![0f32; len * len];
            for qpos in 0..len {
                for kpos in 0..len {
                    let mut acc = 0f32;
                    for t in 0..hd {
                        acc += q[qpos * h + off + t] * k[kpos * h + off + t];
                    }
                    scores[qpos * len + kpos] = acc * scale + mask[qpos * len + kpos];
                }
            }
            let probs = nn::softmax_rows(&scores, len, len);
            for qpos in 0..len {
                for t in 0..hd {
                    let mut acc = 0f32;
                    for kpos in 0..len {
                        acc += probs[qpos * len + kpos] * v[kpos * h + off + t];
                    }
                    ctx[qpos * h + off + t] = acc;
                }
            }
        }
        let attn = nn::linear(&ctx, len, h, &w.out_w[i], Some(&w.out_b[i]));
        for r in 0..len {
            for c in 0..h {
                hidden[r * h + c] += attn[r * h + c];
            }
        }
        // pre-norm MLP
        let n2 = nn::layernorm_affine(&hidden, len, h, &w.ln2_w[i].data, &w.ln2_b[i].data, eps);
        let fc1 = nn::linear(&n2, len, h, &w.fc1_w[i], Some(&w.fc1_b[i]));
        let act = |x: f32| match cfg.hidden_act.as_str() {
            "quick_gelu" => {
                // x·σ(1.702x) — OpenAI CLIP's default MLP activation.
                x * (1.0 / (1.0 + (-1.702f32 * x).exp()))
            }
            _ => nn::gelu_erf(x),
        };
        let gelu: Vec<f32> = fc1.iter().map(|&x| act(x)).collect();
        let fc2 = nn::linear(
            &gelu,
            len,
            cfg.intermediate_size,
            &w.fc2_w[i],
            Some(&w.fc2_b[i]),
        );
        for r in 0..len {
            for c in 0..h {
                hidden[r * h + c] += fc2[r * h + c];
            }
        }
        layer_outs.push(hidden.clone());
    }

    // final_layer_norm → last_hidden_state; pooled = row at argmax(input_ids)
    let last = nn::layernorm_affine(&hidden, len, h, &w.final_ln_w.data, &w.final_ln_b.data, eps);
    let mut eot = 0usize;
    let mut best = input_ids[0];
    for (i, &tok) in input_ids.iter().enumerate() {
        if tok > best {
            best = tok;
            eot = i;
        }
    }
    let pooled: Vec<f32> = last[eot * h..(eot + 1) * h].to_vec();
    (last, pooled, layer_outs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_mask_is_upper_triangle() {
        let len = 4usize;
        let mut m = vec![0f32; len * len];
        for q in 0..len {
            for k in 0..len {
                if k > q {
                    m[q * len + k] = CLIP_MASK;
                }
            }
        }
        assert!(m[0 * 4 + 1].is_infinite() && m[0 * 4 + 1] < 0.0);
        assert_eq!(m[1 * 4 + 0], 0.0);
        assert_eq!(m[3 * 4 + 3], 0.0);
    }
}
