// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 EAGLE / MTP drafter (`google/gemma-4-12B-it-assistant`,
//! `model_type = gemma4_unified_assistant`, arch_id = 22).
//!
//! A single-block speculative-decode draft head for the arch-13 Gemma 4 target.
//! It is NOT a standalone LM: every draft step reads the TARGET's per-step
//! embedding table + the TARGET's shared KV (last sliding-layer slot + last
//! full-layer slot) and a TARGET-provided hidden half. Architecture
//! (code-traced against transformers-main
//! `models/gemma4_assistant/modeling_gemma4_assistant.py` +
//! `generation/candidate_generator.py::SinglePositionMultiTokenCandidateGenerator`):
//!
//!   inputs_embeds = concat[ target_embed(prev_token)·√backbone_hidden  ‖  hidden_backbone ]   # 2·3840 = 7680
//!   x = pre_projection · inputs_embeds                                                          # 7680 → 1024, no bias
//!   for each drafter layer (4: [sliding,sliding,sliding,full]):
//!     # standard Gemma-4 sandwich-norm block, but the layer is a KV-SHARED
//!     # layer (num_kv_shared_layers = num_hidden_layers): it has q_proj /
//!     # q_norm / o_proj ONLY — no k/v projections. K,V come VERBATIM from the
//!     # target's shared cache (already k_norm'd + RoPE'd + v_norm'd).
//!     residual = x
//!     n1 = input_layernorm(x); q = q_proj(n1); per-head q_norm; q·√head_dim
//!     RoPE the drafter Q only (sliding: θ=1e4 full rotate-half over hd=256;
//!       full: θ=1e6 proportional-0.25 over hd=512) at the CONSTANT query_pos
//!     attend Q against the target's shared K/V (sliding layers → last sliding
//!       slot, SWA window 1024; full layer → last full slot, full ctx; MQA);
//!       scale 1.0 (q pre-scaled → kernel 1/√hd cancels)
//!     attn = o_proj; post_attention_layernorm; x = residual + attn
//!     residual = x; n2 = pre_feedforward_layernorm(x)
//!     ffn = down_proj( gelu_tanh(gate_proj(n2)) * up_proj(n2) ); post_ffn norm
//!     x = residual + ffn; x ·= layer_scalar
//!   n = model.norm(x)                                                                          # 1024
//!   logits = n · embed_tokens.T   (tied lm_head, full vocab, NO final softcap) → argmax
//!   post_projection_out = post_projection · n                                                  # 1024 → 3840
//!
//! No drafter KV cache, no cache writes: the whole step is a single-token
//! forward at a FIXED position. `use_ordered_embeddings`/centroid masking is
//! disabled (the shipped head ties lm_head directly).
//!
//! A draft step is one declarative step list: the target-embedding lookup and
//! its √backbone scale into the first concat half, `Gemv(pre_projection)`, one
//! query-only `[SandwichAttention, SandwichMlp, Scale?]` block per layer over
//! the target's last cache slot of the matching type (`crate::program`, the
//! same ops the target runs), the final norm, and `post_projection` straight
//! into the second concat half (the next step's hidden input). The shared
//! `DraftHead` ranks the tied `lm_head` on the GPU and returns the draft. The
//! target's weights and state are READ-ONLY here.

use crate::config::{Gemma4Config, LayerType, RopeType};
use crate::gemma4::{Gemma4State, Gemma4Weights};
use hipfire_dispatch::pipeline::{DraftHead, DraftHeadLayout, DraftHeadPolicy};
use hipfire_runtime::hfq::{load_awq_scale, HfqFile};
use hipfire_runtime::llama::{f16_to_f32, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

pub const DRAFTER_ARCH_ID: u32 = 22;

// ─── Config ─────────────────────────────────────────────────────────────

/// Typed Gemma 4 drafter (`gemma4_unified_assistant`) shape constants. Parsed
/// from the same `metadata.config` envelope as the target: `text_config.*` for
/// the per-block dims plus top-level `backbone_hidden_size` / `num_centroids` /
/// `centroid_intermediate_top_k`.
#[derive(Debug, Clone)]
pub struct Gemma4DrafterConfig {
    pub hidden: usize,   // text_config.hidden_size  (1024 on the 12B drafter)
    pub n_layers: usize, // text_config.num_hidden_layers (4)
    pub vocab_size: usize,
    pub norm_eps: f32,

    pub n_heads: usize,            // num_attention_heads (16)
    pub sliding_head_dim: usize,   // head_dim (256)
    pub sliding_n_kv_heads: usize, // num_key_value_heads (8)
    pub sliding_rope_theta: f32,   // 1e4
    pub sliding_window: usize,     // 1024

    pub full_head_dim: usize,            // global_head_dim (512)
    pub full_n_kv_heads: usize,          // num_global_key_value_heads (1, MQA)
    pub full_rope_theta: f32,            // 1e6
    pub full_rope_type: RopeType,        // Proportional
    pub full_partial_rotary_factor: f32, // 0.25

    pub hidden_dim: usize, // intermediate_size (8192)

    /// Target hidden width: the hidden half + the projection in/out widths.
    pub backbone_hidden: usize, // backbone_hidden_size (3840)
    pub final_logit_softcapping: f32, // null on the drafter → 0.0 (NO softcap)

    /// Clustered-head params (stored, UNUSED — head is plain tied lm_head).
    pub num_centroids: usize,
    pub centroid_top_k: usize,
    pub use_ordered_embeddings: bool,

    pub layer_types: Vec<LayerType>,
    pub norm_plus_one: bool,
}

impl Gemma4DrafterConfig {
    /// Parse from the HFQ metadata envelope (arch_id 22). Mirrors
    /// `Gemma4Config::from_hfq`; the drafter config nests the per-block dims
    /// under `text_config` and keeps `backbone_hidden_size` / `num_centroids` /
    /// `centroid_intermediate_top_k` / `use_ordered_embeddings` at top level.
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("gemma4-drafter: metadata_json not valid JSON: {e}"))?;
        let config = meta
            .get("config")
            .ok_or_else(|| "gemma4-drafter: metadata_json missing `config` wrapper".to_string())?;
        let tc = config.get("text_config").unwrap_or(config);

        let getu = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_u64());
        let getf = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_f64());
        let getb = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_bool());

        let hidden = getu(tc, "hidden_size").ok_or("gemma4-drafter: missing hidden_size")? as usize;
        let n_layers = getu(tc, "num_hidden_layers")
            .ok_or("gemma4-drafter: missing num_hidden_layers")? as usize;
        let vocab_size =
            getu(tc, "vocab_size").ok_or("gemma4-drafter: missing vocab_size")? as usize;
        let norm_eps = getf(tc, "rms_norm_eps").unwrap_or(1e-6) as f32;

        let n_heads = getu(tc, "num_attention_heads")
            .ok_or("gemma4-drafter: missing num_attention_heads")? as usize;

        let sliding_head_dim = getu(tc, "head_dim")
            .map(|v| v as usize)
            .unwrap_or(hidden / n_heads);
        let sliding_n_kv_heads = getu(tc, "num_key_value_heads").unwrap_or(n_heads as u64) as usize;
        let sliding_window = getu(tc, "sliding_window").unwrap_or(1024) as usize;

        let full_head_dim = getu(tc, "global_head_dim")
            .map(|v| v as usize)
            .unwrap_or(sliding_head_dim);
        let full_n_kv_heads =
            getu(tc, "num_global_key_value_heads").unwrap_or(sliding_n_kv_heads as u64) as usize;

        let rope_params = tc.get("rope_parameters");
        let sliding_rope = rope_params.and_then(|r| r.get("sliding_attention"));
        let full_rope = rope_params.and_then(|r| r.get("full_attention"));
        let sliding_rope_theta = sliding_rope
            .and_then(|r| getf(r, "rope_theta"))
            .unwrap_or(10_000.0) as f32;
        let full_rope_theta = full_rope
            .and_then(|r| getf(r, "rope_theta"))
            .unwrap_or(1_000_000.0) as f32;
        let full_rope_type = match full_rope
            .and_then(|r| r.get("rope_type"))
            .and_then(|v| v.as_str())
        {
            Some("proportional") => RopeType::Proportional,
            _ => RopeType::Default,
        };
        let full_partial_rotary_factor = full_rope
            .and_then(|r| getf(r, "partial_rotary_factor"))
            .unwrap_or(1.0) as f32;

        let hidden_dim = getu(tc, "intermediate_size")
            .ok_or("gemma4-drafter: missing intermediate_size")? as usize;

        // text_config softcap is null on the drafter; default 0.0 = NO softcap.
        let final_logit_softcapping = getf(tc, "final_logit_softcapping").unwrap_or(0.0) as f32;

        // Top-level drafter-specific fields.
        let backbone_hidden = getu(config, "backbone_hidden_size")
            .map(|v| v as usize)
            .ok_or("gemma4-drafter: missing backbone_hidden_size")?;
        let num_centroids = getu(config, "num_centroids").unwrap_or(2048) as usize;
        let centroid_top_k = getu(config, "centroid_intermediate_top_k").unwrap_or(32) as usize;
        let use_ordered_embeddings = getb(config, "use_ordered_embeddings").unwrap_or(false);

        let layer_types: Vec<LayerType> = tc
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| match v.as_str().unwrap_or("sliding_attention") {
                        "full_attention" => LayerType::Full,
                        _ => LayerType::Sliding,
                    })
                    .collect()
            })
            .unwrap_or_else(|| {
                // Drafter default: 3 sliding + 1 full.
                let mut v = vec![LayerType::Sliding; n_layers];
                if let Some(last) = v.last_mut() {
                    *last = LayerType::Full;
                }
                v
            });

        let norm_plus_one = hipfire_config::developer_var("HIPFIRE_GEMMA4_NORM_PLUS_ONE")
            .ok()
            .as_deref()
            == Some("1");

        Ok(Gemma4DrafterConfig {
            hidden,
            n_layers,
            vocab_size,
            norm_eps,
            n_heads,
            sliding_head_dim,
            sliding_n_kv_heads,
            sliding_rope_theta,
            sliding_window,
            full_head_dim,
            full_n_kv_heads,
            full_rope_theta,
            full_rope_type,
            full_partial_rotary_factor,
            hidden_dim,
            backbone_hidden,
            final_logit_softcapping,
            num_centroids,
            centroid_top_k,
            use_ordered_embeddings,
            layer_types,
            norm_plus_one,
        })
    }

    /// Concatenated pre-projection input width (2 · backbone_hidden = 7680).
    pub fn pre_proj_in(&self) -> usize {
        2 * self.backbone_hidden
    }

    /// Max q projection width across layer types (scratch sizing).
    pub fn max_q_dim(&self) -> usize {
        (self.n_heads * self.sliding_head_dim).max(self.n_heads * self.full_head_dim)
    }

    /// Max head_dim across the two attention flavours (scratch sizing).
    pub fn max_head_dim(&self) -> usize {
        self.sliding_head_dim.max(self.full_head_dim)
    }
}

// ─── HFQ load helpers (drafter-scoped; flat `model.*` names) ──────────────

fn load_f32_vec(hfq: &HfqFile, name: &str, expected_n: usize) -> Result<Vec<f32>, String> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| format!("gemma4-drafter: tensor not found: {name}"))?;
    let n: usize = info.shape.iter().map(|&s| s as usize).product();
    if expected_n != 0 && n != expected_n {
        return Err(format!(
            "gemma4-drafter: shape mismatch for {name}: expected {expected_n}, got {n}"
        ));
    }
    let f32_data: Vec<f32> = match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        16 => data
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        qt => {
            return Err(format!(
                "gemma4-drafter: expected F16/F32 for {name}, got qt={qt}"
            ))
        }
    };
    Ok(f32_data)
}

fn load_norm(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    dim: usize,
    plus_one: bool,
) -> Result<GpuTensor, String> {
    let mut f32_data = load_f32_vec(hfq, name, dim)?;
    if plus_one {
        for v in f32_data.iter_mut() {
            *v += 1.0;
        }
    }
    gpu.upload_f32(&f32_data, &[dim])
        .map_err(|e| format!("gemma4-drafter: upload norm {name}: {e:?}"))
}

fn load_layer_scalar(hfq: &HfqFile, name: &str) -> f32 {
    match load_f32_vec(hfq, name, 1) {
        Ok(v) => v[0],
        Err(_) => 1.0,
    }
}

/// Projection weight loader — mirrors `gemma4::load_wt` (F16/F32/Q8/MQ* paths).
fn load_wt(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| format!("gemma4-drafter: tensor not found: {name}"))?;
    if info.quant_type == 1 || info.quant_type == 16 {
        if info.quant_type == 16 && rdna_compute::calib_force_bf16() {
            // Native BF16 teacher — keep raw BF16 for consistency (drafter decode is GEMV, not batched MFMA).
            let buf = gpu
                .upload_raw(data, &[m, k])
                .map_err(|e| format!("gemma4-drafter: upload BF16 {name}: {e:?}"))?;
            return Ok(WeightTensor {
                buf,
                gpu_dtype: DType::BF16,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            });
        }
        let f32_data: Vec<f32> = if info.quant_type == 16 {
            data.chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect()
        } else {
            data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        };
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
        };
        let buf = gpu
            .upload_raw(bytes, &[m, k])
            .map_err(|e| format!("gemma4-drafter: upload F32 {name}: {e:?}"))?;
        return Ok(WeightTensor {
            buf,
            gpu_dtype: DType::F32,
            m,
            k,
            row_stride: 0,
            paro: None,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        });
    }
    let dtype = match info.quant_type {
        2 => {
            let buf = gpu
                .upload_raw(data, &[m, k])
                .map_err(|e| format!("gemma4-drafter: upload F32 {name}: {e:?}"))?;
            return Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            });
        }
        3 => DType::Q8_0,
        4 => DType::Q4K,
        6 => DType::HFQ4G256,
        7 => DType::HFQ4G128,
        8 => DType::HFQ6G256,
        9 => DType::HFQ2G256,
        11 => DType::HFQ3G256,
        13 => DType::MQ4G256,
        15 => DType::MQ6G256,
        19 => DType::MQ4G256,
        qt => {
            return Err(format!(
                "gemma4-drafter: unsupported quant_type {qt} for {name}"
            ))
        }
    };
    let buf = gpu
        .upload_raw(data, &[data.len()])
        .map_err(|e| format!("gemma4-drafter: upload {name}: {e:?}"))?;
    let awq_scale = if dtype.supports_awq_sidecar() {
        load_awq_scale(hfq, gpu, name, k)
    } else {
        None
    };
    Ok(WeightTensor {
        buf,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    })
}

// ─── Weights ──────────────────────────────────────────────────────────────

/// One drafter layer. KV-shared (num_kv_shared_layers == n_layers) → no
/// k/v projections, no k/v norms; K/V supplied by the target cache.
pub struct DrafterLayerWeights {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    pub pre_feedforward_layernorm: GpuTensor,
    pub post_feedforward_layernorm: GpuTensor,
    pub layer_scalar_host: f32,

    pub q_proj: WeightTensor,
    pub o_proj: WeightTensor,
    pub q_norm: GpuTensor, // [head_dim]

    pub gate_proj: WeightTensor,
    pub up_proj: WeightTensor,
    pub down_proj: WeightTensor,
}

pub struct Gemma4DrafterWeights {
    /// Own token embedding [vocab, hidden=1024]; aliased as the drafter lm_head.
    pub lm_head: WeightTensor, // tied to embed_tokens, dim = hidden
    pub embed_tokens: GpuTensor,
    pub final_norm: GpuTensor, // model.norm [hidden]

    pub pre_projection: WeightTensor,  // [hidden, 2·backbone_hidden]
    pub post_projection: WeightTensor, // [backbone_hidden, hidden]

    pub layers: Vec<DrafterLayerWeights>,
}

impl Gemma4DrafterWeights {
    /// Load the 48 drafter tensors (FLAT `model.*` names + two top-level
    /// projections). embd format follows the quant_type of `model.embed_tokens`.
    pub fn load(hfq: &HfqFile, cfg: &Gemma4DrafterConfig, gpu: &mut Gpu) -> Result<Self, String> {
        let hidden = cfg.hidden;
        let plus_one = cfg.norm_plus_one;

        // Embedding (tied lm_head).
        let embed_name = "model.embed_tokens.weight";
        let (embed_info, embed_data) = hfq
            .tensor_data(embed_name)
            .ok_or_else(|| "gemma4-drafter: embed_tokens not found".to_string())?;
        let (embed_tokens, embd_dtype) = match embed_info.quant_type {
            3 => (
                gpu.upload_raw(embed_data, &[embed_data.len()])
                    .map_err(|e| format!("gemma4-drafter: upload embed: {e:?}"))?,
                DType::Q8_0,
            ),
            1 => {
                let f32_data: Vec<f32> = embed_data
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                (
                    gpu.upload_f32(&f32_data, &[cfg.vocab_size, hidden])
                        .map_err(|e| format!("gemma4-drafter: upload embed f32: {e:?}"))?,
                    DType::F32,
                )
            }
            16 => {
                let f32_data: Vec<f32> = embed_data
                    .chunks_exact(2)
                    .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                    .collect();
                (
                    gpu.upload_f32(&f32_data, &[cfg.vocab_size, hidden])
                        .map_err(|e| format!("gemma4-drafter: upload embed f32: {e:?}"))?,
                    DType::F32,
                )
            }
            qt => return Err(format!("gemma4-drafter: unsupported embed quant_type {qt}")),
        };
        let lm_head = {
            let alias_buf = unsafe { embed_tokens.buf.alias() };
            let alias_tensor = GpuTensor {
                buf: alias_buf,
                shape: embed_tokens.shape.clone(),
                dtype: embd_dtype,
            };
            WeightTensor {
                buf: alias_tensor,
                gpu_dtype: embd_dtype,
                m: cfg.vocab_size,
                k: hidden,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            }
        };

        let final_norm = load_norm(hfq, gpu, "model.norm.weight", hidden, plus_one)?;

        // Two top-level projections (no bias).
        let pre_projection = load_wt(hfq, gpu, "pre_projection.weight", hidden, cfg.pre_proj_in())?;
        let post_projection = load_wt(
            hfq,
            gpu,
            "post_projection.weight",
            cfg.backbone_hidden,
            hidden,
        )?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("model.layers.{i}");
            let hd = match cfg.layer_types[i] {
                LayerType::Sliding => cfg.sliding_head_dim,
                LayerType::Full => cfg.full_head_dim,
            };
            let q_dim = cfg.n_heads * hd;
            layers.push(DrafterLayerWeights {
                input_layernorm: load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.input_layernorm.weight"),
                    hidden,
                    plus_one,
                )?,
                post_attention_layernorm: load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.post_attention_layernorm.weight"),
                    hidden,
                    plus_one,
                )?,
                pre_feedforward_layernorm: load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.pre_feedforward_layernorm.weight"),
                    hidden,
                    plus_one,
                )?,
                post_feedforward_layernorm: load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.post_feedforward_layernorm.weight"),
                    hidden,
                    plus_one,
                )?,
                layer_scalar_host: load_layer_scalar(hfq, &format!("{p}.layer_scalar")),
                q_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.q_proj.weight"),
                    q_dim,
                    hidden,
                )?,
                o_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.o_proj.weight"),
                    hidden,
                    q_dim,
                )?,
                q_norm: load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.q_norm.weight"),
                    hd,
                    plus_one,
                )?,
                gate_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.gate_proj.weight"),
                    cfg.hidden_dim,
                    hidden,
                )?,
                up_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.up_proj.weight"),
                    cfg.hidden_dim,
                    hidden,
                )?,
                down_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.down_proj.weight"),
                    hidden,
                    cfg.hidden_dim,
                )?,
            });
        }

        Ok(Gemma4DrafterWeights {
            lm_head,
            embed_tokens,
            final_norm,
            pre_projection,
            post_projection,
            layers,
        })
    }

    /// Return all GPU weight buffers to the pool (drained by the daemon's
    /// `unload_model`). Consumes self.
    ///
    /// `lm_head` is NOT freed here: it is always an alias of `embed_tokens`'
    /// allocation (tied embedding — see the alias construction in `load`).
    /// `embed_tokens` owns the bytes and is freed exactly once below; dropping
    /// the alias is a no-op (`DeviceBuffer` has no `Drop`). Same rule as
    /// `Gemma4Weights::free_gpu` on the target.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.embed_tokens);
        let _ = gpu.free_tensor(self.final_norm);
        self.pre_projection.free_all(gpu);
        self.post_projection.free_all(gpu);
        for l in self.layers {
            let _ = gpu.free_tensor(l.input_layernorm);
            let _ = gpu.free_tensor(l.post_attention_layernorm);
            let _ = gpu.free_tensor(l.pre_feedforward_layernorm);
            let _ = gpu.free_tensor(l.post_feedforward_layernorm);
            l.q_proj.free_all(gpu);
            l.o_proj.free_all(gpu);
            let _ = gpu.free_tensor(l.q_norm);
            l.gate_proj.free_all(gpu);
            l.up_proj.free_all(gpu);
            l.down_proj.free_all(gpu);
        }
    }
}

// ─── Scratch ───────────────────────────────────────────────────────────────

/// Per-draft-step GPU scratch. No KV cache (the drafter reads the target's).
pub struct Gemma4DrafterScratch {
    /// `[2·backbone_hidden]`: target embedding half ‖ hidden half. The hidden
    /// half holds the target hidden for a round's first step, then each
    /// step's `post_projection` output for the next.
    pub concat: GpuTensor,
    pub x: GpuTensor,        // [hidden] residual stream
    pub residual: GpuTensor, // [hidden]
    pub tmp: GpuTensor,      // [hidden] norm / o_proj scratch
    /// FWHT rotation scratch for MagnumQuant projections, `[max(q, hidden, bb)]`.
    pub x_rot: GpuTensor,
    pub q: GpuTensor,          // [max_q_dim]
    pub attn_out: GpuTensor,   // [max_q_dim]
    pub gate_ffn: GpuTensor,   // [hidden_dim]
    pub up_ffn: GpuTensor,     // [hidden_dim]
    pub ffn_hidden: GpuTensor, // [hidden_dim]
    pub ffn_out: GpuTensor,    // [hidden]
    pub normed: GpuTensor,     // [hidden] final-norm output (drives lm_head + post_proj)
    /// `[1]` i32 input token of the step.
    pub token_ids: GpuTensor,
    /// device i32 query position scalar (constant across a round).
    pub pos_buf: hip_bridge::DeviceBuffer,
    /// Ranks the tied `lm_head` and returns the draft token.
    pub head: DraftHead,
}

impl Gemma4DrafterScratch {
    pub fn new(
        gpu: &mut Gpu,
        cfg: &Gemma4DrafterConfig,
        weights: &Gemma4DrafterWeights,
    ) -> Result<Self, String> {
        let mut owned: Vec<GpuTensor> = Vec::new();
        let mut alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            let t = g
                .zeros(&[n], DType::F32)
                .map_err(|e| format!("gemma4-drafter: alloc {label}: {e:?}"))?;
            // SAFETY: rollback-only duplicate owner; see `crate::gemma4`.
            owned.push(GpuTensor {
                buf: unsafe { std::ptr::read(&t.buf) },
                shape: t.shape.clone(),
                dtype: t.dtype,
            });
            Ok(t)
        };
        let built = (|| -> Result<_, String> {
            let concat = alloc(gpu, cfg.pre_proj_in(), "concat")?;
            let x = alloc(gpu, cfg.hidden, "x")?;
            let residual = alloc(gpu, cfg.hidden, "residual")?;
            let tmp = alloc(gpu, cfg.hidden, "tmp")?;
            let rot = cfg.max_q_dim().max(cfg.hidden).max(cfg.backbone_hidden);
            let x_rot = alloc(gpu, rot, "x_rot")?;
            let q = alloc(gpu, cfg.max_q_dim(), "q")?;
            let attn_out = alloc(gpu, cfg.max_q_dim(), "attn_out")?;
            let gate_ffn = alloc(gpu, cfg.hidden_dim, "gate_ffn")?;
            let up_ffn = alloc(gpu, cfg.hidden_dim, "up_ffn")?;
            let ffn_hidden = alloc(gpu, cfg.hidden_dim, "ffn_hidden")?;
            let ffn_out = alloc(gpu, cfg.hidden, "ffn_out")?;
            let normed = alloc(gpu, cfg.hidden, "normed")?;
            let token_ids = alloc(gpu, 1, "token_ids")?;
            Ok((
                concat, x, residual, tmp, x_rot, q, attn_out, gate_ffn, up_ffn, ffn_hidden,
                ffn_out, normed, token_ids,
            ))
        })();
        let rollback = |gpu: &mut Gpu, owned: Vec<GpuTensor>| {
            for t in owned {
                let _ = gpu.free_tensor(t);
            }
        };
        let (
            concat,
            x,
            residual,
            tmp,
            x_rot,
            q,
            attn_out,
            gate_ffn,
            up_ffn,
            ffn_hidden,
            ffn_out,
            normed,
            token_ids,
        ) = match built {
            Ok(v) => v,
            Err(e) => {
                rollback(gpu, owned);
                return Err(e);
            }
        };
        let layout = DraftHeadLayout {
            vocab: cfg.vocab_size,
            hidden: cfg.hidden,
            front: 0,
            special: cfg.vocab_size,
            full_hold: 0,
        };
        let policy = DraftHeadPolicy {
            copy: None,
            rescore: false,
        };
        let head = match DraftHead::new(gpu, &weights.lm_head.buf, layout, policy) {
            Ok(head) => head,
            Err(e) => {
                rollback(gpu, owned);
                return Err(format!("gemma4-drafter: draft head: {e}"));
            }
        };
        let pos_buf = match gpu.hip.malloc(4) {
            Ok(buf) => buf,
            Err(e) => {
                let _ = head.free_gpu(gpu);
                rollback(gpu, owned);
                return Err(format!("gemma4-drafter: pos_buf malloc: {e:?}"));
            }
        };
        Ok(Gemma4DrafterScratch {
            concat,
            x,
            residual,
            tmp,
            x_rot,
            q,
            attn_out,
            gate_ffn,
            up_ffn,
            ffn_hidden,
            ffn_out,
            normed,
            token_ids,
            pos_buf,
            head,
        })
    }

    /// Return all scratch buffers to the pool. Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.concat,
            self.x,
            self.residual,
            self.tmp,
            self.x_rot,
            self.q,
            self.attn_out,
            self.gate_ffn,
            self.up_ffn,
            self.ffn_hidden,
            self.ffn_out,
            self.normed,
            self.token_ids,
        ] {
            let _ = gpu.free_tensor(t);
        }
        let _ = self.head.free_gpu(gpu);
        let _ = gpu.hip.free(self.pos_buf);
    }

    /// Start a round at the constant `query_pos` with the target `hidden`
    /// (`[backbone_hidden]`) as the first step's hidden half.
    pub fn begin_round(
        &self,
        gpu: &mut Gpu,
        hidden: &GpuTensor,
        query_pos: usize,
    ) -> Result<(), String> {
        let bb = self.concat.numel() / 2;
        gpu.memcpy_htod_auto(&self.pos_buf, &(query_pos as i32).to_ne_bytes())
            .map_err(|e| format!("gemma4-drafter: htod pos: {e:?}"))?;
        gpu.hip
            .memcpy_dtod_at(&self.concat.buf, bb * 4, &hidden.buf, 0, bb * 4)
            .map_err(|e| format!("gemma4-drafter: stage hidden half: {e:?}"))
    }
}

/// The target embedding table as a format-tagged tensor the embedding step
/// reads; `None` for formats without a batched lookup (F32 oracle tables).
pub fn target_embedding_table(target: &Gemma4Weights) -> Option<GpuTensor> {
    use hipfire_runtime::llama::EmbeddingFormat;
    let dtype = match target.embd_format {
        EmbeddingFormat::Q8_0 => DType::Q8_0,
        EmbeddingFormat::HFQ4G256 => DType::HFQ4G256,
        EmbeddingFormat::HFQ4G128 => DType::HFQ4G128,
        EmbeddingFormat::F32 | EmbeddingFormat::Q4K => return None,
    };
    Some(GpuTensor {
        // SAFETY: a view of the target-owned table for one step list.
        buf: unsafe { target.embed_tokens.buf.alias() },
        shape: target.embed_tokens.shape.clone(),
        dtype,
    })
}

// ─── One draft step ─────────────────────────────────────────────────────────

/// One draft step from `prev_token` at the round's query position: reads the
/// hidden half staged in `ds.concat`, leaves the next step's hidden half
/// there, and returns the draft token. Reads `target_weights` and
/// `target_state` only.
#[allow(clippy::too_many_arguments)]
pub fn draft_step(
    gpu: &mut Gpu,
    dw: &Gemma4DrafterWeights,
    dcfg: &Gemma4DrafterConfig,
    ds: &mut Gemma4DrafterScratch,
    target_weights: &Gemma4Weights,
    target_state: &Gemma4State,
    target_cfg: &Gemma4Config,
    prev_token: u32,
    query_pos: usize,
) -> Result<u32, String> {
    gpu.memcpy_htod_auto(&ds.token_ids.buf, &(prev_token as i32).to_ne_bytes())
        .map_err(|e| format!("gemma4-drafter: htod token: {e:?}"))?;
    run_step(
        gpu,
        dw,
        dcfg,
        ds,
        target_weights,
        target_state,
        target_cfg,
        query_pos,
        true,
    )?;
    ds.head
        .draft(gpu, &dw.lm_head.buf, &ds.normed)
        .map_err(|e| format!("gemma4-drafter: draft head: {e}"))
}

/// The step list. With `embed`, the first concat half is the target
/// embedding of `ds.token_ids` (× √backbone); without, the caller staged the
/// whole concat (validation harness).
#[allow(clippy::too_many_arguments)]
fn run_step(
    gpu: &mut Gpu,
    dw: &Gemma4DrafterWeights,
    dcfg: &Gemma4DrafterConfig,
    ds: &mut Gemma4DrafterScratch,
    target_weights: &Gemma4Weights,
    target_state: &Gemma4State,
    target_cfg: &Gemma4Config,
    pos: usize,
    embed: bool,
) -> Result<(), String> {
    use crate::program::{Geometry, LayerKv, LayerRefs, LayerScratch, ProgramBinding, Resident};
    use hipfire_dispatch::pipeline::sandwich::ScaleOp;
    use hipfire_dispatch::pipeline::EmbeddingOp;
    use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};

    let bb = dcfg.backbone_hidden;
    // Each block attends the target's LAST cache slot of its type.
    let last_slot = |n: usize, kind: &str| {
        n.checked_sub(1)
            .ok_or_else(|| format!("gemma4-drafter: target has no {kind} layers to share KV from"))
    };
    let sliding_slot = last_slot(target_cfg.n_sliding_layers(), "sliding")?;
    let full_slot = last_slot(target_cfg.n_full_layers(), "full")?;
    let table = target_embedding_table(target_weights)
        .ok_or("gemma4-drafter: target embedding format has no batched lookup")?;
    let embed_half = ds.concat.sub_offset(0, bb);
    let hidden_half = ds.concat.sub_offset(bb, bb);
    let binding = ProgramBinding {
        geo: Geometry::drafter(dcfg),
        resident: Resident {
            kv_sliding: &target_state.kv_sliding,
            kv_full: &target_state.kv_full,
            pos_buf: &ds.pos_buf,
            v_norm_ones: &target_state.v_norm_ones,
            flash_partials: &target_state.q8_flash_partials,
        },
        rows: 1,
        position: pos,
        positions: None,
        scratch: LayerScratch {
            x: &ds.x,
            residual: &ds.residual,
            normed: &ds.tmp,
            attn_rot: &ds.x_rot,
            mlp_rot: &ds.x_rot,
            q: &ds.q,
            k: &ds.q,
            v: &ds.q,
            attn_out: &ds.attn_out,
            gate: &ds.gate_ffn,
            up: &ds.up_ffn,
            act: &ds.ffn_hidden,
            mlp_out: &ds.ffn_out,
            ple: None,
            moe: None,
        },
    };

    let pre = dw.pre_projection.dispatch_ref();
    let post = dw.post_projection.dispatch_ref();
    let mut steps = Vec::with_capacity(3 * dcfg.n_layers + 5);
    if embed {
        steps.push(Step::Embed(EmbeddingOp {
            table: &table,
            rotated: &ds.x_rot,
            token_ids: &ds.token_ids,
            output: &embed_half,
            rows: 1,
            dim: bb,
        }));
        // The target's ScaledWordEmbedding bakes ·√backbone into its forward.
        steps.push(Step::Scale(ScaleOp {
            x: &embed_half,
            factor: (bb as f32).sqrt(),
        }));
    }
    steps.push(Step::Gemv {
        w: &pre,
        input: GemvInput::Raw(&ds.concat),
        out: &ds.x,
    });
    for (layer_idx, layer) in dw.layers.iter().enumerate() {
        let layer_type = dcfg.layer_types[layer_idx];
        let slot = match layer_type {
            LayerType::Sliding => sliding_slot,
            LayerType::Full => full_slot,
        };
        binding.layer(
            layer_idx,
            LayerRefs::drafter(layer, layer_type, dcfg.hidden_dim),
            LayerKv {
                slot,
                writes: false,
            },
            &mut steps,
        )?;
    }
    steps.push(Step::RmsnormAutomatic {
        x: &ds.x,
        norm_weight: &dw.final_norm,
        x_plain: &ds.normed,
        out: &ds.normed,
        awq_scale: None,
        k: dcfg.hidden,
        eps: dcfg.norm_eps,
        rotation: hipfire_dispatch::types::RotationPlan::None,
    });
    // The next step's hidden half; pre_projection already consumed this one.
    steps.push(Step::Gemv {
        w: &post,
        input: GemvInput::Raw(&ds.normed),
        out: &hidden_half,
    });
    let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);
    execute_steps(gpu, &ctx, &steps).map_err(|e| format!("gemma4-drafter: {e}"))
}

/// Oracle outputs of one validation step: the draft, its full logits, the
/// final normed hidden and the post-projected hidden half.
pub struct DrafterStepOut {
    pub argmax: u32,
    pub post_proj_hidden: Vec<f32>,
    pub normed_hidden: Vec<f32>,
    pub logits: Vec<f32>,
}

/// Validation entry: run one step from a PRE-BUILT `concat` (HF-exact inputs
/// isolate the backbone and heads from the embedding/feedback path) at
/// `query_pos` and download every intermediate the oracle compares.
#[allow(clippy::too_many_arguments)]
pub fn drafter_step_from_concat(
    gpu: &mut Gpu,
    dw: &Gemma4DrafterWeights,
    dcfg: &Gemma4DrafterConfig,
    ds: &mut Gemma4DrafterScratch,
    target_weights: &Gemma4Weights,
    target_state: &Gemma4State,
    target_cfg: &Gemma4Config,
    concat: &[f32],
    query_pos: usize,
) -> Result<DrafterStepOut, String> {
    if concat.len() != dcfg.pre_proj_in() {
        return Err(format!(
            "gemma4-drafter: concat len {} != {}",
            concat.len(),
            dcfg.pre_proj_in()
        ));
    }
    gpu.memcpy_htod_auto(&ds.pos_buf, &(query_pos as i32).to_ne_bytes())
        .map_err(|e| format!("gemma4-drafter: htod pos: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&ds.concat.buf, unsafe {
            std::slice::from_raw_parts(concat.as_ptr() as *const u8, concat.len() * 4)
        })
        .map_err(|e| format!("gemma4-drafter: upload concat: {e:?}"))?;
    run_step(
        gpu,
        dw,
        dcfg,
        ds,
        target_weights,
        target_state,
        target_cfg,
        query_pos,
        false,
    )?;
    let argmax = ds
        .head
        .draft(gpu, &dw.lm_head.buf, &ds.normed)
        .map_err(|e| format!("gemma4-drafter: draft head: {e}"))?;
    let download = |gpu: &mut Gpu, t: &GpuTensor, what: &str| {
        gpu.download_f32(t)
            .map_err(|e| format!("gemma4-drafter: download {what}: {e:?}"))
    };
    Ok(DrafterStepOut {
        argmax,
        post_proj_hidden: download(
            gpu,
            &ds.concat
                .sub_offset(dcfg.backbone_hidden, dcfg.backbone_hidden),
            "post_proj",
        )?,
        normed_hidden: download(gpu, &ds.normed, "normed")?,
        logits: download(gpu, ds.head.logits(), "logits")?,
    })
}
