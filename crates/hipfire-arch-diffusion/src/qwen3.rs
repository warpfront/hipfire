// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU Qwen3 reference encoder with hidden-state taps for FLUX.2 Klein
//! conditioning.
//!
//! Klein conditions on a Qwen3 causal LM text encoder: the residual stream
//! after layers 9, 18, 27 (1-based, [`KLEIN_TAPS`]) is concatenated per
//! token to build the `txt` conditioning sequence. This is the CPU
//! reference forward — dependency-free, obviously-correct f32, in the same
//! style as [`crate::t5`] — the GPU version lands on the same interfaces
//! later.
//!
//! Architecture notes (Qwen3 causal decoder):
//! - RMSNorm pre-norm, GQA attention with per-head Q/K RMSNorm applied
//!   *before* RoPE, half-split RoPE (`rotate_half`), causal + key-padding
//!   mask, SwiGLU MLP.
//! - `q_norm`/`k_norm` are per-head (`[head_dim]`), not per-projection.

use crate::flux::Tensor;
use crate::nn;
use hipfire_runtime::model_source::ModelSource as ModelSourceTrait;

/// 1-based layer indices whose post-block residual stream Klein conditions
/// on, concatenated per token in this order.
pub const KLEIN_TAPS: [usize; 3] = [9, 18, 27];

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub rope_theta: f64,
    pub eps: f32,
    pub vocab: usize,
    pub tie_embeddings: bool,
}

impl Qwen3Config {
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let get_usize = |k: &str| -> Result<usize, String> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| format!("qwen3 config: missing `{k}`"))
        };
        let get_f64 = |k: &str| -> Result<f64, String> {
            v.get(k)
                .and_then(|x| x.as_f64())
                .ok_or_else(|| format!("qwen3 config: missing `{k}`"))
        };
        let hidden = get_usize("hidden_size")?;
        let heads = get_usize("num_attention_heads")?;
        let head_dim = v
            .get("head_dim")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(hidden / heads);
        Ok(Self {
            hidden,
            layers: get_usize("num_hidden_layers")?,
            heads,
            kv_heads: get_usize("num_key_value_heads")?,
            head_dim,
            intermediate: get_usize("intermediate_size")?,
            rope_theta: get_f64("rope_theta")?,
            eps: v
                .get("rms_norm_eps")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(1e-6),
            vocab: get_usize("vocab_size")?,
            tie_embeddings: v
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(true),
        })
    }
}

/// One Qwen3 decoder layer's weights, row-major f32.
#[derive(Debug, Clone)]
pub struct Qwen3Layer {
    pub input_norm: Tensor,
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub o: Tensor,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub post_norm: Tensor,
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
}

/// Qwen3 causal LM weights (the text encoder Klein conditions on).
#[derive(Debug, Clone)]
pub struct Qwen3Weights {
    pub config: Qwen3Config,
    pub embed: Tensor, // [vocab, hidden]
    pub layers: Vec<Qwen3Layer>,
}

impl Qwen3Weights {
    /// True for a [`Qwen3Plan::materialize_light`] load: the embedding table
    /// is real but the decoder layers were left on the checkpoint (streaming
    /// mode). [`encode_taps`] on a light set would tap nothing, so every
    /// host-encode path materialises first — see
    /// `FluxPipeBundle::qwen3_host`.
    ///
    /// Mirrors `T5Weights::is_light`, which is what the FLUX.1 half of the
    /// same decision reads.
    pub fn is_light(&self) -> bool {
        self.layers.is_empty()
    }
}

/// Naming plan for a Qwen3 checkpoint: pure metadata, no tensor bytes read.
#[derive(Debug, Clone)]
pub struct Qwen3Plan {
    pub config: Qwen3Config,
}

impl Qwen3Plan {
    pub fn detect(src: &dyn ModelSourceTrait) -> Result<Self, String> {
        let config_str = src.metadata_json();
        let v: serde_json::Value = serde_json::from_str(config_str)
            .map_err(|e| format!("qwen3: config.json invalid: {e}"))?;
        // SafetensorsSource wraps the config under `config`; unwrap when present.
        let v = v.get("config").cloned().unwrap_or(v);
        let config = Qwen3Config::from_json(&v)?;
        Ok(Self { config })
    }

    /// The 11 tensor keys of layer `i`, `(name, rows, cols)`, in
    /// [`Qwen3Layer`] field order.
    pub fn layer_keys(&self, i: usize) -> [(String, usize, usize); 11] {
        let c = &self.config;
        let (h, hd) = (c.hidden, c.head_dim);
        let (heads, kvh, inter) = (c.heads, c.kv_heads, c.intermediate);
        let p = format!("model.layers.{i}");
        [
            (format!("{p}.input_layernorm.weight"), h, 1),
            (format!("{p}.self_attn.q_proj.weight"), heads * hd, h),
            (format!("{p}.self_attn.k_proj.weight"), kvh * hd, h),
            (format!("{p}.self_attn.v_proj.weight"), kvh * hd, h),
            (format!("{p}.self_attn.o_proj.weight"), h, heads * hd),
            (format!("{p}.self_attn.q_norm.weight"), hd, 1),
            (format!("{p}.self_attn.k_norm.weight"), hd, 1),
            (format!("{p}.post_attention_layernorm.weight"), h, 1),
            (format!("{p}.mlp.gate_proj.weight"), inter, h),
            (format!("{p}.mlp.up_proj.weight"), inter, h),
            (format!("{p}.mlp.down_proj.weight"), h, inter),
        ]
    }

    /// Decode one tensor to f32, checking its element count.
    pub fn tensor(
        &self,
        src: &dyn ModelSourceTrait,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Tensor, String> {
        let (info, bytes) = src
            .tensor_data(name)
            .ok_or_else(|| format!("qwen3: missing tensor `{name}`"))?;
        let n = info.shape.iter().product::<usize>();
        if n != rows * cols {
            return Err(format!(
                "qwen3: tensor `{name}` has {n} elems, expected {}",
                rows * cols
            ));
        }
        let data = crate::flux::decode_dtype(&info.dtype, bytes)?;
        Ok(Tensor { data, rows, cols })
    }

    /// Stage one tensor as f16 words for a direct device upload, never
    /// materialising it as f32.
    pub fn stage_f16<'s>(
        &self,
        src: &dyn ModelSourceTrait,
        name: &str,
        rows: usize,
        cols: usize,
        stage: &'s mut crate::f16_stage::F16Stage,
    ) -> Result<&'s [u16], String> {
        let (info, bytes) = src
            .tensor_data(name)
            .ok_or_else(|| format!("qwen3: missing tensor `{name}`"))?;
        stage.clear();
        let n = stage
            .push(&info.dtype, bytes)
            .map_err(|e| format!("qwen3: tensor `{name}`: {e}"))?;
        if n != rows * cols {
            return Err(format!(
                "qwen3: tensor `{name}` has {n} elems, expected {}",
                rows * cols
            ));
        }
        Ok(stage.words())
    }

    /// Hint that a tensor's bytes are done with. Best-effort.
    pub fn release(&self, src: &dyn ModelSourceTrait, name: &str) {
        src.release_tensor_pages(name);
    }

    /// The embedding table only, layers left empty. Not a valid `encode_taps`
    /// input.
    pub fn materialize_light(&self, src: &dyn ModelSourceTrait) -> Result<Qwen3Weights, String> {
        let d = self.config.hidden;
        Ok(Qwen3Weights {
            embed: self.tensor(src, "model.embed_tokens.weight", self.config.vocab, d)?,
            layers: vec![],
            config: self.config.clone(),
        })
    }

    /// Decode the whole causal LM into f32 host tables. `lm_head.weight`,
    /// present only when `tie_embeddings` is false, is ignored: the encoder
    /// never uses it.
    pub fn materialize(&self, src: &dyn ModelSourceTrait) -> Result<Qwen3Weights, String> {
        let mut out = self.materialize_light(src)?;
        for i in 0..self.config.layers {
            let keys = self.layer_keys(i);
            let [input_norm, q, k, v, o, q_norm, k_norm, post_norm, gate, up, down] =
                keys.map(|(name, rows, cols)| self.tensor(src, &name, rows, cols));
            out.layers.push(Qwen3Layer {
                input_norm: input_norm?,
                q: q?,
                k: k?,
                v: v?,
                o: o?,
                q_norm: q_norm?,
                k_norm: k_norm?,
                post_norm: post_norm?,
                gate: gate?,
                up: up?,
                down: down?,
            });
        }
        Ok(out)
    }
}

/// Half-split RoPE (HF `rotate_half`) applied in place over `[len, heads, hd]`.
fn rope_halfsplit(x: &mut [f32], len: usize, heads: usize, hd: usize, theta: f64) {
    let half = hd / 2;
    let inv_freq: Vec<f64> = (0..half)
        .map(|j| theta.powf(-2.0 * j as f64 / hd as f64))
        .collect();
    for pos in 0..len {
        for h in 0..heads {
            let base = (pos * heads + h) * hd;
            for j in 0..half {
                let angle = pos as f64 * inv_freq[j];
                let (sin, cos) = (angle.sin() as f32, angle.cos() as f32);
                let a = x[base + j];
                let b = x[base + j + half];
                x[base + j] = a * cos - b * sin;
                x[base + j + half] = a * sin + b * cos;
            }
        }
    }
}

/// Qwen3 causal-LM forward with hidden-state taps. Returns
/// `[len, taps.len() * hidden]`: the residual stream after each 1-based
/// layer index in `taps`, concatenated per token in `taps` order.
pub fn encode_taps(
    w: &Qwen3Weights,
    input_ids: &[u32],
    key_mask: &[u8],
    taps: &[usize],
) -> Vec<f32> {
    let cfg = &w.config;
    let d = cfg.hidden;
    let len = input_ids.len();
    let (heads, kvh, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
    let group = heads / kvh;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut hidden = vec![0.0f32; len * d];
    for (t, &id) in input_ids.iter().enumerate() {
        hidden[t * d..(t + 1) * d]
            .copy_from_slice(&w.embed.data[id as usize * d..(id as usize + 1) * d]);
    }

    let mut out = vec![0.0f32; len * taps.len() * d];
    for (li, layer) in w.layers.iter().enumerate() {
        let normed = nn::rmsnorm_scale(&hidden, len, d, &layer.input_norm.data, cfg.eps);
        let mut q = nn::linear(&normed, len, d, &layer.q, None); // [len, heads*hd]
        let mut k = nn::linear(&normed, len, d, &layer.k, None); // [len, kvh*hd]
        let v = nn::linear(&normed, len, d, &layer.v, None);
        q = nn::rmsnorm_scale(&q, len * heads, hd, &layer.q_norm.data, cfg.eps);
        k = nn::rmsnorm_scale(&k, len * kvh, hd, &layer.k_norm.data, cfg.eps);
        rope_halfsplit(&mut q, len, heads, hd, cfg.rope_theta);
        rope_halfsplit(&mut k, len, kvh, hd, cfg.rope_theta);

        let mut ctx = vec![0.0f32; len * heads * hd];
        for h in 0..heads {
            let kh = h / group;
            for qp in 0..len {
                let mut scores = vec![f32::NEG_INFINITY; len];
                for kp in 0..=qp {
                    if key_mask[kp] == 0 {
                        continue;
                    }
                    let mut acc = 0.0;
                    for t in 0..hd {
                        acc += q[(qp * heads + h) * hd + t] * k[(kp * kvh + kh) * hd + t];
                    }
                    scores[kp] = acc * scale;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                if !m.is_finite() {
                    continue; // fully masked row (a pad query with no visible key)
                }
                let mut sum = 0.0;
                let probs: Vec<f32> = scores
                    .iter()
                    .map(|s| {
                        let e = (s - m).exp();
                        sum += e;
                        e
                    })
                    .collect();
                for kp in 0..=qp {
                    let p = probs[kp] / sum;
                    if p == 0.0 {
                        continue;
                    }
                    for t in 0..hd {
                        ctx[(qp * heads + h) * hd + t] += p * v[(kp * kvh + kh) * hd + t];
                    }
                }
            }
        }
        let o = nn::linear(&ctx, len, heads * hd, &layer.o, None);
        for i in 0..len * d {
            hidden[i] += o[i];
        }
        let n2 = nn::rmsnorm_scale(&hidden, len, d, &layer.post_norm.data, cfg.eps);
        let g = nn::linear(&n2, len, d, &layer.gate, None);
        let u = nn::linear(&n2, len, d, &layer.up, None);
        let a: Vec<f32> = g.iter().zip(&u).map(|(g, u)| nn::silu(*g) * u).collect();
        let dn = nn::linear(&a, len, cfg.intermediate, &layer.down, None);
        for i in 0..len * d {
            hidden[i] += dn[i];
        }
        if let Some(ti) = taps.iter().position(|&t| t == li + 1) {
            for t in 0..len {
                out[(t * taps.len() + ti) * d..(t * taps.len() + ti + 1) * d]
                    .copy_from_slice(&hidden[t * d..(t + 1) * d]);
            }
        }
    }
    out
}

#[cfg(test)]
const QWEN3_SYNTH_SEED: u64 = 0xC0FF_EE00_0000_0051; // "…51" for the Qwen3 encoder

#[cfg(test)]
impl Qwen3Weights {
    /// Deterministic synthetic weights matching `cfg`'s shapes — same
    /// `synth_val` scheme as [`crate::flux::FluxWeights::synthetic`], filled
    /// in field order (embed first, then each layer's 11 tensors in
    /// [`Qwen3Layer`] field order).
    pub fn synthetic(cfg: &Qwen3Config) -> Self {
        use crate::flux::synth_val;
        let mut idx = 0u64;
        let mut fill = |n: usize| -> Vec<f32> {
            let data: Vec<f32> = (0..n as u64)
                .map(|i| synth_val(QWEN3_SYNTH_SEED, idx + i) * 0.05)
                .collect();
            idx += n as u64;
            data
        };
        let d = cfg.hidden;
        let embed = Tensor {
            data: fill(cfg.vocab * d),
            rows: cfg.vocab,
            cols: d,
        };
        let mut layers = Vec::with_capacity(cfg.layers);
        let (heads, kvh, hd, inter) = (cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.intermediate);
        for _ in 0..cfg.layers {
            layers.push(Qwen3Layer {
                input_norm: Tensor {
                    data: fill(d),
                    rows: d,
                    cols: 1,
                },
                q: Tensor {
                    data: fill(heads * hd * d),
                    rows: heads * hd,
                    cols: d,
                },
                k: Tensor {
                    data: fill(kvh * hd * d),
                    rows: kvh * hd,
                    cols: d,
                },
                v: Tensor {
                    data: fill(kvh * hd * d),
                    rows: kvh * hd,
                    cols: d,
                },
                o: Tensor {
                    data: fill(d * heads * hd),
                    rows: d,
                    cols: heads * hd,
                },
                q_norm: Tensor {
                    data: fill(hd),
                    rows: hd,
                    cols: 1,
                },
                k_norm: Tensor {
                    data: fill(hd),
                    rows: hd,
                    cols: 1,
                },
                post_norm: Tensor {
                    data: fill(d),
                    rows: d,
                    cols: 1,
                },
                gate: Tensor {
                    data: fill(inter * d),
                    rows: inter,
                    cols: d,
                },
                up: Tensor {
                    data: fill(inter * d),
                    rows: inter,
                    cols: d,
                },
                down: Tensor {
                    data: fill(d * inter),
                    rows: d,
                    cols: inter,
                },
            });
        }
        Qwen3Weights {
            config: cfg.clone(),
            embed,
            layers,
        }
    }
}

/// On-disk fixture writer for a synthetic Qwen3 checkpoint — the text half
/// of the tiny Klein pipe `pipeline.rs` builds.
#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::{Qwen3Plan, QWEN3_SYNTH_SEED};
    use crate::flux::test_fixtures::{bf16_blob, shape_of, NamedTensor};

    /// Every tensor [`Qwen3Plan::materialize`] reads, BF16, in the same value
    /// order `Qwen3Weights::synthetic` uses (embed, then each layer's 11
    /// tensors in [`super::Qwen3Layer`] field order) — so a checkpoint written
    /// here decodes to the synthetic tables modulo the bf16 truncation.
    pub(crate) fn checkpoint_tensors(plan: &Qwen3Plan) -> Vec<NamedTensor> {
        let c = &plan.config;
        let mut idx = 0u64;
        let mut out: Vec<NamedTensor> = vec![(
            "model.embed_tokens.weight".to_string(),
            bf16_blob(QWEN3_SYNTH_SEED, &mut idx, c.vocab * c.hidden),
            "BF16".to_string(),
            vec![c.vocab, c.hidden],
        )];
        for i in 0..c.layers {
            for (name, rows, cols) in plan.layer_keys(i) {
                let data = bf16_blob(QWEN3_SYNTH_SEED, &mut idx, rows * cols);
                out.push((name, data, "BF16".to_string(), shape_of(rows, cols)));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_4b_config_parses() {
        let v = serde_json::json!({ "hidden_size": 2560, "num_hidden_layers": 36, "num_attention_heads": 32,
            "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 9728, "rope_theta": 1000000,
            "rms_norm_eps": 1e-6, "vocab_size": 151936, "tie_word_embeddings": true });
        let c = Qwen3Config::from_json(&v).unwrap();
        assert_eq!(
            (
                c.hidden,
                c.layers,
                c.heads,
                c.kv_heads,
                c.head_dim,
                c.intermediate
            ),
            (2560, 36, 32, 8, 128, 9728)
        );
        assert_eq!(c.rope_theta, 1e6);
        assert!(c.tie_embeddings);
    }

    fn tiny_cfg() -> Qwen3Config {
        Qwen3Config {
            hidden: 16,
            layers: 4,
            heads: 2,
            kv_heads: 1,
            head_dim: 8,
            intermediate: 24,
            rope_theta: 1e6,
            eps: 1e-6,
            vocab: 32,
            tie_embeddings: true,
        }
    }

    #[test]
    fn taps_are_causal_and_pad_invariant() {
        // Tiny config: prefix tokens must not change when pad tokens are appended (causal + key mask).
        let cfg = tiny_cfg();
        let w = Qwen3Weights::synthetic(&cfg);
        let ids = [3u32, 7, 11];
        let a = encode_taps(&w, &ids, &[1, 1, 1], &[1, 2, 4]);
        let ids_p = [3u32, 7, 11, 0, 0];
        let b = encode_taps(&w, &ids_p, &[1, 1, 1, 0, 0], &[1, 2, 4]);
        assert_eq!(a.len(), 3 * 3 * 16);
        assert_eq!(b.len(), 5 * 3 * 16);
        for (x, y) in a.iter().zip(&b[..a.len()]) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn tap_order_is_layer_major_per_token() {
        let cfg = tiny_cfg();
        let w = Qwen3Weights::synthetic(&cfg);
        let single = encode_taps(&w, &[5, 6], &[1, 1], &[2]);
        let both = encode_taps(&w, &[5, 6], &[1, 1], &[1, 2]);
        // token 0: [tap1 (16) | tap2 (16)]; tap2 of `both` equals `single`
        assert_eq!(&both[16..32], &single[0..16]);
        assert_eq!(&both[48..64], &single[16..32]);
    }
}
