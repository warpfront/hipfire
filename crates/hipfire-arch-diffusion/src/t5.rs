// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU T5 encoder reference (conditioning for FLUX: `txt` hidden states),
//! pinned to `transformers` 5.16 eager semantics — the golden source:
//!
//! - T5LayerNorm (RMS over last axis, no mean subtraction), eps 1e-6.
//! - T5Attention: q/k/v/o linear (bias-free), per-head dim `d_kv`,
//!   **`scaling = 1.0`** (transformers 5 folds the relative bias in and
//!   does not scale the dot product — the golden was captured against this),
//!   relative-position buckets (`bidirectional=True`), additive key-padding
//!   mask (pad *keys* → −3.4e38; pad queries still attend to real keys).
//! - One relative-bias embedding on layer 0, reused by every layer.
//! - FFN: pre-norm → ReLU — note `d_ff` 37 here is NOT a multiple of 2
//!   (tiny fixture; the real T5-XXL uses 10240/8192): shapes are read from
//!   the config, nothing is hardcoded.
//! - `last_hidden_state` = final_layer_norm at the stack tail.
//!
//! Reference fixture: `text_encoder_2/` (hf-internal-testing/tiny-random-t5):
//! d_model 32, d_ff 37, d_kv 8, heads 4, depth 5, vocab 1103.

use crate::flux::Tensor;
use crate::nn;
use hipfire_runtime::model_source::ModelSource as ModelSourceTrait;
use serde::{Deserialize, Serialize};

/// Additive mask value for padded keys (torch fp32 `-inf` equivalent).
pub const T5_MASK: f32 = -3.4e38;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T5Config {
    pub d_model: usize,
    pub d_ff: usize,
    pub d_kv: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub layer_norm_epsilon: f32,
}

impl T5Config {
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let get = |k: &str| -> Result<usize, String> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| format!("t5 config: missing `{k}`"))
        };
        Ok(Self {
            d_model: get("d_model")?,
            d_ff: get("d_ff")?,
            d_kv: get("d_kv")?,
            num_heads: get("num_heads")?,
            num_layers: get("num_layers")?,
            vocab_size: get("vocab_size")?,
            relative_attention_num_buckets: get("relative_attention_num_buckets")?,
            relative_attention_max_distance: get("relative_attention_max_distance")?,
            layer_norm_epsilon: v
                .get("layer_norm_epsilon")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(1e-6),
        })
    }
}

/// T5 encoder weights (row-major f32; all linears bias-free).
#[derive(Debug, Clone)]
pub struct T5Weights {
    pub config: T5Config,
    pub embed: Tensor, // [vocab, d_model]
    pub q: Vec<Tensor>,
    pub k: Vec<Tensor>,
    pub v: Vec<Tensor>,
    pub o: Vec<Tensor>,
    pub attn_norm: Vec<Tensor>,
    pub wi: Vec<Tensor>,
    /// Gate projection, present only for the **gated** FFN variant
    /// (`feed_forward_proj: "gated-gelu"`, i.e. T5 v1.1). FLUX conditions on
    /// T5-XXL v1.1, which is gated: its FFN is
    /// `wo(gelu(wi_0(x)) * wi_1(x))`, not `wo(relu(wi(x)))`. When this is
    /// empty the layer uses the original non-gated ReLU form, so both
    /// checkpoint families load through the same struct.
    pub wi_gate: Vec<Tensor>,
    pub wo: Vec<Tensor>,
    pub ffn_norm: Vec<Tensor>,
    pub rel_bias: Tensor, // [buckets, num_heads] (layer 0 only)
    pub final_norm: Tensor,
}

/// Naming plan for a T5 encoder checkpoint: the config, and which FFN form
/// the tensor names say it uses.
///
/// Pure metadata — building one reads no tensor bytes. It exists so the GPU
/// upload can stream tensor-by-tensor out of the mmap
/// ([`crate::t5_gpu::GpuT5Weights::from_stream`]) instead of first decoding
/// T5-XXL into f32 host tables, which is ~18.5 GB held for the life of the
/// bundle even though the GPU encoder only ever reads the embedding table
/// back from the host.
#[derive(Debug, Clone)]
pub struct T5Plan {
    pub config: T5Config,
    /// T5 **v1.1** gated FFN (`wo(gelu(wi_0 x) * wi_1 x)`), what FLUX
    /// conditions on. `false` is the original v1.0 `wo(relu(wi x))`.
    ///
    /// Detected from the tensor names rather than from `feed_forward_proj`,
    /// so a checkpoint whose config is trimmed — the ComfyUI single-file text
    /// encoders often are — still loads.
    pub gated: bool,
}

impl T5Plan {
    pub fn detect(src: &dyn ModelSourceTrait) -> Result<Self, String> {
        let config_str = src.metadata_json();
        let v: serde_json::Value = serde_json::from_str(config_str)
            .map_err(|e| format!("t5: config.json invalid: {e}"))?;
        // SafetensorsSource wraps the config under `config`; unwrap when present.
        let v = v.get("config").cloned().unwrap_or(v);
        let config = T5Config::from_json(&v)?;
        let gated = src
            .tensor_info("encoder.block.0.layer.1.DenseReluDense.wi_0.weight")
            .is_some();
        Ok(Self { config, gated })
    }

    /// The FFN input-projection key(s) of layer `i`, in `[wi, wi_gate]` order.
    pub fn ffn_in(&self, i: usize) -> Vec<String> {
        if self.gated {
            vec![
                format!("encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight"),
                format!("encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight"),
            ]
        } else {
            vec![format!(
                "encoder.block.{i}.layer.1.DenseReluDense.wi.weight"
            )]
        }
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
            .ok_or_else(|| format!("t5: missing tensor `{name}`"))?;
        let n = info.shape.iter().product::<usize>();
        if n != rows * cols {
            return Err(format!(
                "t5: tensor `{name}` has {n} elems, expected {}",
                rows * cols
            ));
        }
        let data = crate::flux::decode_dtype(&info.dtype, bytes)?;
        Ok(Tensor { data, rows, cols })
    }

    /// Stage one tensor as f16 words for a direct device upload, never
    /// materialising it as f32. Bit-identical to `tensor()` followed by the
    /// device `(_Float16)` cast — see [`crate::f16_stage`].
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
            .ok_or_else(|| format!("t5: missing tensor `{name}`"))?;
        stage.clear();
        let n = stage
            .push(&info.dtype, bytes)
            .map_err(|e| format!("t5: tensor `{name}`: {e}"))?;
        if n != rows * cols {
            return Err(format!(
                "t5: tensor `{name}` has {n} elems, expected {}",
                rows * cols
            ));
        }
        Ok(stage.words())
    }

    /// Hint that a tensor's bytes are done with. Best-effort.
    pub fn release(&self, src: &dyn ModelSourceTrait, name: &str) {
        src.release_tensor_pages(name);
    }

    /// The tables the GPU encoder still reads from the host after its weights
    /// are resident: the token embedding (gathered per prompt) and the
    /// relative-attention bias (expanded per sequence length). Everything
    /// else — the 24 layers of q/k/v/o/wi/wi_gate/wo, i.e. all but ~0.5 GB of
    /// T5-XXL — is left empty and never decoded.
    ///
    /// The result is NOT a valid input to the host [`encode`]; the caller must
    /// materialise the full set for that (see
    /// `FluxPipeBundle::t5_host`).
    pub fn materialize_light(&self, src: &dyn ModelSourceTrait) -> Result<T5Weights, String> {
        let d = self.config.d_model;
        Ok(T5Weights {
            embed: self.tensor(src, "shared.weight", self.config.vocab_size, d)?,
            rel_bias: self.tensor(
                src,
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                self.config.relative_attention_num_buckets,
                self.config.num_heads,
            )?,
            final_norm: self.tensor(src, "encoder.final_layer_norm.weight", d, 1)?,
            q: vec![],
            k: vec![],
            v: vec![],
            o: vec![],
            attn_norm: vec![],
            wi: vec![],
            wi_gate: vec![],
            wo: vec![],
            ffn_norm: vec![],
            config: self.config.clone(),
        })
    }

    /// Decode the WHOLE encoder into f32 host tables (~18.5 GB for T5-XXL).
    /// The host reference [`encode`] path's input, and nothing else.
    pub fn materialize(&self, src: &dyn ModelSourceTrait) -> Result<T5Weights, String> {
        let d = self.config.d_model;
        let d_ff = self.config.d_ff;
        let mut out = self.materialize_light(src)?;
        for i in 0..self.config.num_layers {
            out.q.push(self.tensor(
                src,
                &format!("encoder.block.{i}.layer.0.SelfAttention.q.weight"),
                d,
                d,
            )?);
            out.k.push(self.tensor(
                src,
                &format!("encoder.block.{i}.layer.0.SelfAttention.k.weight"),
                d,
                d,
            )?);
            out.v.push(self.tensor(
                src,
                &format!("encoder.block.{i}.layer.0.SelfAttention.v.weight"),
                d,
                d,
            )?);
            out.o.push(self.tensor(
                src,
                &format!("encoder.block.{i}.layer.0.SelfAttention.o.weight"),
                d,
                d,
            )?);
            out.attn_norm.push(self.tensor(
                src,
                &format!("encoder.block.{i}.layer.0.layer_norm.weight"),
                d,
                1,
            )?);
            let ffn_in = self.ffn_in(i);
            out.wi.push(self.tensor(src, &ffn_in[0], d_ff, d)?);
            if self.gated {
                out.wi_gate.push(self.tensor(src, &ffn_in[1], d_ff, d)?);
            }
            out.wo.push(self.tensor(
                src,
                &format!("encoder.block.{i}.layer.1.DenseReluDense.wo.weight"),
                d,
                d_ff,
            )?);
            out.ffn_norm.push(self.tensor(
                src,
                &format!("encoder.block.{i}.layer.1.layer_norm.weight"),
                d,
                1,
            )?);
        }
        Ok(out)
    }
}

impl T5Weights {
    /// Load from a component source dir (e.g. `<pipe>/text_encoder_2/`).
    ///
    /// A thin wrapper over [`T5Plan::detect`] + [`T5Plan::materialize`]: one
    /// decode implementation serves both the eager host tables and the
    /// streaming GPU upload, so the two cannot drift.
    pub fn load(src: &dyn ModelSourceTrait) -> Result<Self, String> {
        T5Plan::detect(src)?.materialize(src)
    }

    /// True when only the tables the GPU encoder needs are present — the
    /// linears were streamed straight to the device and never decoded here.
    /// The host [`encode`] cannot run on such a set.
    pub fn is_light(&self) -> bool {
        self.q.is_empty()
    }
}

/// Relative-position bucket (transformers `T5Attention._relative_position_bucket`,
/// `bidirectional=True`, exact log-bucket math).
pub fn relative_position_bucket(
    relative_position: isize,
    num_buckets: usize,
    max_distance: usize,
) -> usize {
    // transformers halves the bucket count for the bidirectional case, then
    // adds `half` for positive offsets: negative side 0..half−1, positive
    // side half..2·half−1 (NOT symmetric — sign lives in the bucket index).
    let half = num_buckets / 2;
    let rel = relative_position.unsigned_abs();
    let max_exact = half / 2;
    let bucket = if rel < max_exact {
        rel
    } else {
        let large = max_exact
            + (((rel as f32 / max_exact as f32).ln())
                / ((max_distance as f32) / max_exact as f32).ln()
                * (half - max_exact) as f32) as usize;
        large.min(half - 1)
    };
    if relative_position > 0 {
        bucket + half
    } else {
        bucket
    }
}

/// Per-head relative bias, `[heads][q][k]`, from the layer-0 embedding.
/// `rel_bias.rows` = buckets, `rel_bias.cols` = heads.
pub fn compute_rel_bias(
    query_len: usize,
    key_len: usize,
    rel_bias: &Tensor,
    max_distance: usize,
) -> Vec<f32> {
    let heads = rel_bias.cols;
    let buckets = rel_bias.rows;
    let mut bias = vec![0f32; heads * query_len * key_len];
    for q in 0..query_len {
        for k in 0..key_len {
            let rel = k as isize - q as isize;
            let bucket = relative_position_bucket(rel, buckets, max_distance);
            for h in 0..heads {
                bias[h * query_len * key_len + q * key_len + k] = rel_bias.data[bucket * heads + h];
            }
        }
    }
    bias
}

/// T5 encoder forward (batch 1). Returns `(last_hidden_state [len][d_model],
/// per-layer hidden states [layers][len][d_model])` — the latter consumed
/// by the golden unit tests.
pub fn encode(
    w: &T5Weights,
    input_ids: &[u32],
    attention_mask: &[u8],
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let cfg = &w.config;
    let len = input_ids.len();
    let d = cfg.d_model;
    let heads = cfg.num_heads;
    let hd = cfg.d_kv;
    let eps = cfg.layer_norm_epsilon;

    // token embeddings
    let mut hidden: Vec<f32> = vec![0f32; len * d];
    for (r, &tok) in input_ids.iter().enumerate() {
        let tok = tok as usize;
        hidden[r * d..(r + 1) * d].copy_from_slice(&w.embed.data[tok * d..(tok + 1) * d]);
    }

    // relative bias (layer 0) reused by all layers.
    //
    // NOTE — no key-padding mask: ComfyUI's FLUX T5-XXL is constructed with
    // `enable_attention_masks=False` (T5XXLModel default, comfy/text_encoders/
    // sd3_clip.py), so the golden fixture it produced attends over the pad
    // rows too. transformers attends pads masked; ComfyUI does not, and
    // matching ComfyUI is what the gate measures. A masked T5 hidden diverges
    // from the reference already at layer 0 (cos ~0.68) and ends the 24-layer
    // stack near-orthogonal to it. `attention_mask` is accepted for API
    // stability but intentionally unused.
    let _ = attention_mask;
    let rel_bias = compute_rel_bias(len, len, &w.rel_bias, cfg.relative_attention_max_distance);

    let mut layer_outs: Vec<Vec<f32>> = Vec::with_capacity(cfg.num_layers);
    for i in 0..cfg.num_layers {
        // ── self-attention (pre-norm) ────────────────────────────────
        let normed = nn::rmsnorm_scale(&hidden, len, d, &w.attn_norm[i].data, eps);
        let q = nn::linear(&normed, len, d, &w.q[i], None);
        let k = nn::linear(&normed, len, d, &w.k[i], None);
        let v = nn::linear(&normed, len, d, &w.v[i], None);

        let mut attn_out = vec![0f32; len * d];
        let mut context = vec![0f32; len * d];
        for h in 0..heads {
            let off = h * hd;
            let mut scores = vec![0f32; len * len];
            for qpos in 0..len {
                for kpos in 0..len {
                    let mut acc = 0f32;
                    for t in 0..hd {
                        acc += q[qpos * d + off + t] * k[kpos * d + off + t];
                    }
                    scores[qpos * len + kpos] = acc + rel_bias[h * len * len + qpos * len + kpos];
                }
            }
            let probs = nn::softmax_rows(&scores, len, len);
            for qpos in 0..len {
                for t in 0..hd {
                    let mut acc = 0f32;
                    for kpos in 0..len {
                        acc += probs[qpos * len + kpos] * v[kpos * d + off + t];
                    }
                    context[qpos * d + off + t] = acc;
                }
            }
        }
        // o projection + residual
        attn_out = nn::linear(&context, len, d, &w.o[i], None);
        for r in 0..len {
            for c in 0..d {
                hidden[r * d + c] += attn_out[r * d + c];
            }
        }
        // ── FFN (pre-norm) ───────────────────────────────────────────
        // Two variants. T5 v1.0: `wo(relu(wi(x)))`. T5 v1.1 — which is what
        // FLUX conditions on — is GATED: `wo(gelu(wi_0(x)) * wi_1(x))`, with
        // the tanh ("gelu_new") GELU the reference uses. Picking the wrong one
        // does not fail loudly; it silently produces a wrong text embedding and
        // therefore a plausible-looking but wrong image.
        let ffn_in = nn::rmsnorm_scale(&hidden, len, d, &w.ffn_norm[i].data, eps);
        let wi_out = nn::linear(&ffn_in, len, d, &w.wi[i], None);
        let act: Vec<f32> = if w.wi_gate.is_empty() {
            wi_out.iter().map(|x| x.max(0.0)).collect()
        } else {
            let gate = nn::linear(&ffn_in, len, d, &w.wi_gate[i], None);
            wi_out
                .iter()
                .zip(gate.iter())
                .map(|(a, g)| nn::gelu_tanh(*a) * *g)
                .collect()
        };
        let wo_out = nn::linear(&act, len, cfg.d_ff, &w.wo[i], None);
        for r in 0..len {
            for c in 0..d {
                hidden[r * d + c] += wo_out[r * d + c];
            }
        }
        layer_outs.push(hidden.clone());
    }

    let last = nn::rmsnorm_scale(&hidden, len, d, &w.final_norm.data, eps);
    (last, layer_outs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flux::Tensor;

    #[test]
    fn bucket_boundaries_match_transformers() {
        // transformers `_relative_position_bucket` with (8, 128):
        let c = |rel: isize| relative_position_bucket(rel, 8, 128);
        assert_eq!(c(0), 0);
        assert_eq!(c(-1), 1); // negative exact increment
        assert_eq!(c(1), 4 + 1); // positive side = +half
        assert_eq!(c(2), 4 + 2); // >= max_exact(2) → log bucket on +side
        assert_eq!(c(-2), 0 + 2);
        assert_eq!(c(127), 4 + 3); // big positive → last positive bucket
        assert_eq!(c(128), 4 + 3); // clamps at max_distance
        assert_eq!(c(-128), 3); // negative clamps at half−1
    }

    #[test]
    fn rel_bias_embeds_sign_by_half_bucket_offset() {
        // The per-head bias matrix must come straight from the embedding:
        // value(q,k) == embed[bucket(k−q)] per head. Sign lives in the
        // bucket id (asymmetric, T5 semantics).
        let buckets = 8usize;
        let heads = 4usize;
        let mut data = Vec::with_capacity(buckets * heads);
        for b in 0..buckets {
            for h in 0..heads {
                data.push((b * 10 + h) as f32);
            }
        }
        let rb = Tensor {
            data,
            rows: buckets,
            cols: heads,
        };
        let bias = compute_rel_bias(4, 4, &rb, 128);
        for h in 0..heads {
            for q in 0..4 {
                for k in 0..4 {
                    let rel = k as isize - q as isize;
                    let bucket = relative_position_bucket(rel, 8, 128);
                    let expect = rb.data[bucket * heads + h];
                    let got = bias[h * 16 + q * 4 + k];
                    assert!(
                        (got - expect).abs() < 1e-6,
                        "h={h} q={q} k={k}: {got} vs {expect}"
                    );
                }
            }
        }
    }
}
#[cfg(test)]
mod gated_ffn_tests {
    use super::*;
    use crate::nn;

    /// FLUX conditions on T5-XXL **v1.1**, whose FFN is gated:
    /// `wo(gelu(wi_0 x) * wi_1 x)`. The original T5 v1.0 form is
    /// `wo(relu(wi x))`. Choosing the wrong one does not fail — it yields a
    /// wrong text embedding and therefore a wrong image, so the two branches
    /// are pinned here.
    ///
    /// `wi_gate` non-empty is what selects the gated path, and `T5Weights::load`
    /// populates it from the presence of `...DenseReluDense.wi_0.weight`.
    #[test]
    fn gated_and_ungated_ffn_differ_and_match_their_formulas() {
        let d = 2usize;
        let d_ff = 3usize;
        let x = vec![0.7f32, -0.4];
        let wi = Tensor {
            data: vec![0.5, -0.25, 0.75, 0.1, -0.6, 0.2],
            rows: d_ff,
            cols: d,
        };
        let gate = Tensor {
            data: vec![0.2, 0.9, -0.3, 0.4, 0.6, -0.1],
            rows: d_ff,
            cols: d,
        };

        let wi_out = nn::linear(&x, 1, d, &wi, None);
        let g_out = nn::linear(&x, 1, d, &gate, None);

        // v1.0: ReLU, no gate.
        let ungated: Vec<f32> = wi_out.iter().map(|v| v.max(0.0)).collect();
        // v1.1: GELU-tanh of the first projection, multiplied by the second.
        let gated: Vec<f32> = wi_out
            .iter()
            .zip(g_out.iter())
            .map(|(a, g)| nn::gelu_tanh(*a) * *g)
            .collect();

        assert_ne!(
            ungated, gated,
            "the two FFN forms must differ, else this test proves nothing"
        );
        // Spot-check the gated formula element-wise against its definition.
        for i in 0..d_ff {
            let expect = nn::gelu_tanh(wi_out[i]) * g_out[i];
            assert!(
                (gated[i] - expect).abs() < 1e-6,
                "gated FFN lane {i}: {} != {expect}",
                gated[i]
            );
        }
    }
}

/// Streaming-load coverage for the T5 encoder: the plan must name the same
/// tensors the eager loader read, stage them bit-identically to
/// decode-then-RNE, and be able to hand back either the light set (what the
/// GPU encoder needs from the host) or the full one (what the CPU reference
/// encoder needs). No GPU.
#[cfg(test)]
mod stream_tests {
    use super::*;
    use crate::f16_stage::{f32_to_f16_rne, F16Stage};
    use serde_json::json;
    use std::io::Write;
    use std::path::{Path, PathBuf};

    fn test_cfg() -> serde_json::Value {
        // Tiny but structurally complete: >1 layer so the per-layer key
        // formatting is exercised, d_ff != d_model so a transposed `wo` would
        // fail the element count.
        json!({
            "d_model": 8,
            "d_ff": 12,
            "d_kv": 4,
            "num_heads": 2,
            "num_layers": 2,
            "vocab_size": 16,
            "relative_attention_num_buckets": 4,
            "relative_attention_max_distance": 32,
        })
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hipfire-t5-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn bf16_blob(seed: u64, n: usize) -> Vec<u8> {
        // Deterministic bf16 words spanning both signs and a range of
        // exponents, all inside f16's normal range as real weights are.
        (0..n as u64)
            .flat_map(|i| {
                let mut x = (seed ^ i).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                x ^= x >> 31;
                let sign = ((x >> 3) & 1) as u16;
                let exp = (120 + (x % 14)) as u16; // 2^-7 .. 2^6
                let man = ((x >> 17) & 0x7F) as u16;
                ((sign << 15) | (exp << 7) | man).to_le_bytes()
            })
            .collect()
    }

    /// Write a synthetic T5 encoder checkpoint. `gated` picks the v1.1
    /// (`wi_0`/`wi_1`) or v1.0 (`wi`) FFN naming, which is the thing
    /// `T5Plan::detect` sniffs.
    fn write_t5(dir: &Path, cfg: &T5Config, gated: bool) {
        std::fs::write(dir.join("config.json"), test_cfg().to_string()).unwrap();
        let (d, d_ff) = (cfg.d_model, cfg.d_ff);
        let mut named: Vec<(String, Vec<usize>)> = vec![
            ("shared.weight".into(), vec![cfg.vocab_size, d]),
            (
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight".into(),
                vec![cfg.relative_attention_num_buckets, cfg.num_heads],
            ),
            ("encoder.final_layer_norm.weight".into(), vec![d]),
        ];
        for i in 0..cfg.num_layers {
            let b = format!("encoder.block.{i}");
            for p in ["q", "k", "v", "o"] {
                named.push((format!("{b}.layer.0.SelfAttention.{p}.weight"), vec![d, d]));
            }
            named.push((format!("{b}.layer.0.layer_norm.weight"), vec![d]));
            if gated {
                named.push((
                    format!("{b}.layer.1.DenseReluDense.wi_0.weight"),
                    vec![d_ff, d],
                ));
                named.push((
                    format!("{b}.layer.1.DenseReluDense.wi_1.weight"),
                    vec![d_ff, d],
                ));
            } else {
                named.push((
                    format!("{b}.layer.1.DenseReluDense.wi.weight"),
                    vec![d_ff, d],
                ));
            }
            named.push((
                format!("{b}.layer.1.DenseReluDense.wo.weight"),
                vec![d, d_ff],
            ));
            named.push((format!("{b}.layer.1.layer_norm.weight"), vec![d]));
        }

        let mut header = serde_json::Map::new();
        let mut blobs: Vec<Vec<u8>> = Vec::new();
        let mut offset = 0usize;
        for (seed, (name, shape)) in named.iter().enumerate() {
            let n: usize = shape.iter().product();
            let data = bf16_blob(seed as u64 + 1, n);
            let mut meta = serde_json::Map::new();
            meta.insert("dtype".into(), "BF16".into());
            meta.insert(
                "shape".into(),
                serde_json::Value::Array(shape.iter().map(|&s| s.into()).collect()),
            );
            meta.insert("data_offsets".into(), json!([offset, offset + data.len()]));
            offset += data.len();
            header.insert(name.clone(), meta.into());
            blobs.push(data);
        }
        let header_json = serde_json::Value::Object(header).to_string();
        let mut f = std::fs::File::create(dir.join("model.safetensors")).unwrap();
        f.write_all(&(header_json.len() as u64).to_le_bytes())
            .unwrap();
        f.write_all(header_json.as_bytes()).unwrap();
        for b in &blobs {
            f.write_all(b).unwrap();
        }
    }

    fn open(dir: &Path) -> hipfire_runtime::safetensors_source::SafetensorsSource {
        hipfire_runtime::safetensors_source::SafetensorsSource::open(dir)
            .unwrap_or_else(|e| panic!("open synthetic T5: {e}"))
    }

    /// The streamed f16 words must equal the old path's output exactly:
    /// decode to f32, then the device's round-to-nearest-even cast. Anything
    /// else moves the T5 conditioning for a reason no diff explains.
    #[test]
    fn stage_f16_is_bit_identical_to_decode_then_rne() {
        let cfg = T5Config::from_json(&test_cfg()).unwrap();
        let dir = temp_dir("stream-gated");
        write_t5(&dir, &cfg, true);
        let src = open(&dir);
        let plan = T5Plan::detect(&src).unwrap();
        assert!(
            plan.gated,
            "wi_0/wi_1 naming must be detected as v1.1 gated"
        );

        let (d, d_ff) = (cfg.d_model, cfg.d_ff);
        let mut linears: Vec<(String, usize, usize)> = Vec::new();
        for i in 0..cfg.num_layers {
            let b = format!("encoder.block.{i}");
            for p in ["q", "k", "v", "o"] {
                linears.push((format!("{b}.layer.0.SelfAttention.{p}.weight"), d, d));
            }
            for name in plan.ffn_in(i) {
                linears.push((name, d_ff, d));
            }
            linears.push((format!("{b}.layer.1.DenseReluDense.wo.weight"), d, d_ff));
        }
        assert_eq!(linears.len(), cfg.num_layers * 7, "q,k,v,o,wi,wi_gate,wo");

        let mut stage = F16Stage::new();
        for (name, rows, cols) in &linears {
            let want: Vec<u16> = plan
                .tensor(&src, name, *rows, *cols)
                .unwrap()
                .data
                .iter()
                .map(|v| f32_to_f16_rne(*v))
                .collect();
            let got = plan
                .stage_f16(&src, name, *rows, *cols, &mut stage)
                .unwrap();
            assert_eq!(got, &want[..], "staged words differ for `{name}`");
        }
    }

    /// The light set is exactly what the GPU encoder reads back from the host
    /// (embedding gather + relative bias) and nothing else, and it must agree
    /// element-for-element with the full eager load.
    #[test]
    fn materialize_light_matches_the_full_load_and_omits_the_linears() {
        let cfg = T5Config::from_json(&test_cfg()).unwrap();
        let dir = temp_dir("stream-light");
        write_t5(&dir, &cfg, true);
        let src = open(&dir);
        let plan = T5Plan::detect(&src).unwrap();

        let light = plan.materialize_light(&src).unwrap();
        let full = T5Weights::load(&src).unwrap();

        assert!(light.is_light(), "light set must report is_light");
        assert!(!full.is_light(), "full set must not report is_light");
        assert_eq!(light.embed.data, full.embed.data, "embedding table drift");
        assert_eq!(light.rel_bias.data, full.rel_bias.data, "rel bias drift");
        assert_eq!(light.final_norm.data, full.final_norm.data);
        for v in [
            &light.q,
            &light.k,
            &light.v,
            &light.o,
            &light.wi,
            &light.wi_gate,
            &light.wo,
            &light.attn_norm,
            &light.ffn_norm,
        ] {
            assert!(v.is_empty(), "light set must not decode any linear");
        }
        assert_eq!(full.q.len(), cfg.num_layers);
        assert_eq!(full.wi_gate.len(), cfg.num_layers);
    }

    /// An ungated (v1.0) checkpoint must still plan and load — the GPU path
    /// declines it, the host encoder serves it — and `ffn_in` must name the
    /// single `wi` rather than the two v1.1 projections.
    #[test]
    fn ungated_checkpoint_plans_the_single_wi() {
        let cfg = T5Config::from_json(&test_cfg()).unwrap();
        let dir = temp_dir("stream-ungated");
        write_t5(&dir, &cfg, false);
        let src = open(&dir);
        let plan = T5Plan::detect(&src).unwrap();
        assert!(!plan.gated);
        assert_eq!(
            plan.ffn_in(1),
            vec!["encoder.block.1.layer.1.DenseReluDense.wi.weight".to_string()]
        );
        let full = plan.materialize(&src).unwrap();
        assert_eq!(full.wi.len(), cfg.num_layers);
        assert!(full.wi_gate.is_empty());
    }

    #[test]
    fn a_missing_tensor_is_named_in_the_error() {
        let cfg = T5Config::from_json(&test_cfg()).unwrap();
        let dir = temp_dir("stream-missing");
        write_t5(&dir, &cfg, true);
        let src = open(&dir);
        let plan = T5Plan::detect(&src).unwrap();
        let mut stage = F16Stage::new();
        let err = plan
            .stage_f16(
                &src,
                "encoder.block.9.layer.0.SelfAttention.q.weight",
                8,
                8,
                &mut stage,
            )
            .unwrap_err();
        assert!(err.contains("encoder.block.9"), "{err}");
        // A shape the checkpoint disagrees with is caught, not silently
        // uploaded at the wrong size.
        let err = plan
            .stage_f16(&src, "encoder.final_layer_norm.weight", 9, 1, &mut stage)
            .unwrap_err();
        assert!(err.contains("expected 9"), "{err}");
    }
}
