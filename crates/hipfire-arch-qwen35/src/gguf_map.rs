// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GGUF -> hipfire tensor-name map and hyper-parameters for the dense
//! Qwen3.5 / Qwen3.6 / Qwen3.8 hybrid family (GatedDeltaNet + gated full
//! attention, hipfire arch 5; GGUF `general.architecture = "qwen35"`).
//!
//! The table is ours, written from what the upstream exporter emits (the
//! converter's `Qwen3_5TextModel` / `Qwen3NextModel` tensor edits, the
//! `gguf-py` tensor-name tables and the architecture's tensor list) and
//! from the shapes its loader creates; nothing is copied.
//!
//! Naming: a GGUF tensor `blk.N.<role>.weight` maps to the hipfire HFQ name
//! `model.language_model.layers.N.<hf role>.weight` (the first candidate of
//! `qwen35_tensor_name_candidates`); `token_embd` / `output_norm` / `output`
//! map to `model.language_model.embed_tokens.weight` / `.norm.weight` /
//! `lm_head.weight`. The MTP block (`blk.<n_layers>.*`, present when the
//! export carries `qwen35.nextn_predict_layers`) maps to the bare names of
//! the `.mtp` sidecar (`wq`, `eh_proj`, `shared_head_norm`, ...) under an
//! `mtp.` prefix, because hipfire's MTP head is a sidecar contract, not
//! trunk layer `n_layers`.
//!
//! **The names are only half of the contract.** The exporter also rewrites
//! tensor *data*; each [`ExpectedTensor`] carries a [`GgufTransform`] that
//! records it so the compute slice cannot read GGUF bytes as if they were
//! HF bytes:
//! - every RMSNorm weight except `linear_attn.norm` has `+1` folded in
//!   (the HFQ loader adds `QWEN35_NORM_BIAS = 1.0` at dequant, so a GGUF
//!   norm must be loaded with bias `0`);
//! - `A_log` is stored as `-exp(A_log)` (tensor `ssm_a`);
//! - the depthwise conv kernel is squeezed to `[conv_dim, K]`;
//! - when `linear_num_key_heads != linear_num_value_heads` the value-head
//!   axis of `in_proj_qkv` (V rows), `in_proj_z`, `in_proj_a/b`, `A_log`,
//!   `dt_bias`, the V channels of `conv1d` and the columns of `out_proj` is
//!   permuted from grouped-by-key-head to tiled order.
//!
//! Missing or extra tensors are the caller's refusal to raise, naming the
//! tensor; this module only states what a canonical export contains.

use crate::qwen35::{config_from_metadata_json, LayerType, Qwen35Config};
use hipfire_runtime::gguf::{MetaValue, TensorInfo};
use std::collections::HashMap;

/// `general.architecture` of the dense family's GGUF export.
pub const GGUF_ARCHITECTURE: &str = "qwen35";

const KEY_PREFIX: &str = "qwen35";

/// Hyper-parameters read from the GGUF KV metadata.
#[derive(Debug, Clone)]
pub struct Qwen35GgufHparams {
    /// The same struct the HFQ / safetensors loaders produce.
    pub config: Qwen35Config,
    /// `block_count - nextn_predict_layers`: trunk layers (`config.n_layers`).
    pub n_layers: usize,
    /// NextN/MTP blocks appended after the trunk (`blk.<n_layers>..`).
    pub nextn_layers: usize,
}

fn key(name: &str) -> String {
    format!("{KEY_PREFIX}.{name}")
}

fn req_usize(meta: &HashMap<String, MetaValue>, name: &str) -> Result<usize, String> {
    let k = key(name);
    meta.get(&k)
        .and_then(MetaValue::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| {
            format!(
                "GGUF metadata key {k} is missing or not an integer; this file is not a \
                 canonical qwen35 export"
            )
        })
}

fn opt_usize(meta: &HashMap<String, MetaValue>, name: &str) -> Result<Option<usize>, String> {
    let k = key(name);
    match meta.get(&k) {
        None => Ok(None),
        Some(v) => v
            .as_u64()
            .map(|v| Some(v as usize))
            .ok_or_else(|| format!("GGUF metadata key {k} is not an integer")),
    }
}

fn req_f64(meta: &HashMap<String, MetaValue>, name: &str) -> Result<f64, String> {
    let k = key(name);
    meta.get(&k)
        .and_then(MetaValue::as_f64)
        .ok_or_else(|| format!("GGUF metadata key {k} is missing or not a float"))
}

/// A KV head count that is a scalar, or a per-layer array whose non-zero
/// entries all agree (hipfire's full-attention layers share one KV width).
fn kv_head_count(meta: &HashMap<String, MetaValue>) -> Result<usize, String> {
    let k = key("attention.head_count_kv");
    match meta.get(&k) {
        None => Err(format!(
            "GGUF metadata key {k} is missing; this file is not a canonical qwen35 export"
        )),
        Some(MetaValue::Array(items)) => {
            let mut found: Option<usize> = None;
            for item in items {
                let v = item
                    .as_u64()
                    .ok_or_else(|| format!("GGUF metadata key {k} holds a non-integer entry"))?
                    as usize;
                if v == 0 {
                    continue;
                }
                match found {
                    None => found = Some(v),
                    Some(prev) if prev == v => {}
                    Some(prev) => {
                        return Err(format!(
                            "GGUF metadata key {k} varies per layer ({prev} and {v}); hipfire's \
                             Qwen3.5 attention layers share one KV head count"
                        ))
                    }
                }
            }
            found.ok_or_else(|| format!("GGUF metadata key {k} has no non-zero entry"))
        }
        Some(v) => v
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| format!("GGUF metadata key {k} is not an integer")),
    }
}

/// Parse the dense Qwen3.5 hyper-parameters. `tensors` supplies the
/// vocabulary size (the `token_embd.weight` row count is authoritative).
pub fn hparams_from_gguf(
    meta: &HashMap<String, MetaValue>,
    tensors: &[TensorInfo],
) -> Result<Qwen35GgufHparams, String> {
    if let Some(experts) = opt_usize(meta, "expert_count")? {
        if experts > 0 {
            return Err(format!(
                "GGUF metadata says expert_count={experts}: this is a mixture-of-experts file, \
                 and the dense qwen35 carrier does not admit it"
            ));
        }
    }
    let block_count = req_usize(meta, "block_count")?;
    let nextn = opt_usize(meta, "nextn_predict_layers")?.unwrap_or(0);
    if nextn > block_count {
        return Err(format!(
            "GGUF metadata nextn_predict_layers={nextn} exceeds block_count={block_count}"
        ));
    }
    let n_layers = block_count - nextn;
    if n_layers == 0 {
        return Err("GGUF metadata block_count leaves no trunk layers".to_string());
    }
    let dim = req_usize(meta, "embedding_length")?;
    let ff = req_usize(meta, "feed_forward_length")?;
    let heads = req_usize(meta, "attention.head_count")?;
    let kv_heads = kv_head_count(meta)?;
    if dim == 0 || heads == 0 || kv_heads == 0 {
        return Err("GGUF metadata has a zero embedding length or head count".to_string());
    }
    let head_dim = opt_usize(meta, "attention.key_length")?.unwrap_or(dim / heads);
    if let Some(value_len) = opt_usize(meta, "attention.value_length")? {
        if value_len != head_dim {
            return Err(format!(
                "GGUF metadata attention.value_length={value_len} differs from key_length=\
                 {head_dim}; hipfire's Qwen3.5 attention uses one head dimension"
            ));
        }
    }
    let eps = req_f64(meta, "attention.layer_norm_rms_epsilon")?;
    let rope_dim = req_usize(meta, "rope.dimension_count")?;
    if head_dim == 0 || rope_dim == 0 || rope_dim > head_dim {
        return Err(format!(
            "GGUF metadata rope.dimension_count={rope_dim} is not within head_dim={head_dim}"
        ));
    }
    let partial_rotary_factor = rope_dim as f64 / head_dim as f64;
    let rope_theta = meta
        .get(&key("rope.freq_base"))
        .and_then(MetaValue::as_f64)
        .unwrap_or(10_000_000.0);
    let mut mrope_section = [11u64, 11, 10];
    if let Some(items) = meta
        .get(&key("rope.dimension_sections"))
        .and_then(MetaValue::as_array)
    {
        for (dst, src) in mrope_section.iter_mut().zip(items.iter().take(3)) {
            *dst = src
                .as_u64()
                .ok_or_else(|| "GGUF rope.dimension_sections holds a non-integer".to_string())?;
        }
    }

    let conv_kernel = req_usize(meta, "ssm.conv_kernel")?;
    let key_head_dim = req_usize(meta, "ssm.state_size")?;
    let num_k_heads = req_usize(meta, "ssm.group_count")?;
    let num_v_heads = req_usize(meta, "ssm.time_step_rank")?;
    let inner = req_usize(meta, "ssm.inner_size")?;
    if num_v_heads == 0 || num_k_heads == 0 || inner % num_v_heads != 0 {
        return Err(format!(
            "GGUF metadata ssm.inner_size={inner} is not a multiple of ssm.time_step_rank=\
             {num_v_heads}"
        ));
    }
    let value_head_dim = inner / num_v_heads;

    let recurrent: Vec<bool> = match meta
        .get(&key("attention.recurrent_layers"))
        .and_then(MetaValue::as_array)
    {
        Some(items) => {
            if items.len() < n_layers {
                return Err(format!(
                    "GGUF metadata attention.recurrent_layers has {} entries for {n_layers} \
                     trunk layers",
                    items.len()
                ));
            }
            items
                .iter()
                .take(n_layers)
                .map(|v| {
                    v.as_bool()
                        .or_else(|| v.as_u64().map(|n| n != 0))
                        .ok_or_else(|| {
                            "GGUF metadata attention.recurrent_layers holds a non-boolean"
                                .to_string()
                        })
                })
                .collect::<Result<_, _>>()?
        }
        None => {
            let interval = opt_usize(meta, "full_attention_interval")?.unwrap_or(4);
            if interval == 0 {
                return Err("GGUF metadata full_attention_interval is zero".to_string());
            }
            (0..n_layers).map(|i| (i + 1) % interval != 0).collect()
        }
    };
    let layer_types: Vec<&str> = recurrent
        .iter()
        .map(|&r| {
            if r {
                "linear_attention"
            } else {
                "full_attention"
            }
        })
        .collect();

    let vocab = tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .and_then(|t| t.shape.get(1).copied())
        .ok_or_else(|| {
            "GGUF has no token_embd.weight (or it is not 2-D), so the vocabulary size is unknown"
                .to_string()
        })?;

    let mut config = serde_json::json!({
        "hidden_size": dim,
        "num_hidden_layers": n_layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "vocab_size": vocab,
        "intermediate_size": ff,
        "rms_norm_eps": eps,
        "layer_types": layer_types,
        "linear_num_key_heads": num_k_heads,
        "linear_num_value_heads": num_v_heads,
        "linear_key_head_dim": key_head_dim,
        "linear_value_head_dim": value_head_dim,
        "linear_conv_kernel_dim": conv_kernel,
        "rope_parameters": {
            "rope_theta": rope_theta,
            "partial_rotary_factor": partial_rotary_factor,
            // The exporter documents that Qwen3.5 always uses interleaved MRoPE.
            "mrope_interleaved": true,
            "mrope_section": mrope_section,
        },
    });
    if let Some(eos) = meta
        .get("tokenizer.ggml.eos_token_id")
        .and_then(MetaValue::as_u64)
    {
        config["eos_token_id"] = serde_json::json!(eos);
    }
    let envelope = serde_json::json!({ "architecture": GGUF_ARCHITECTURE, "config": config });
    let config = config_from_metadata_json(&envelope.to_string())?;
    Ok(Qwen35GgufHparams {
        n_layers: config.n_layers,
        config,
        nextn_layers: nextn,
    })
}

/// Which part of the file a tensor belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorGroup {
    /// The trunk (embedding, layers, final norm, LM head).
    Trunk,
    /// The NextN/MTP block. A file may be trunk-only: the MTP tensors are
    /// then absent as a group, never half-present.
    Mtp,
}

/// What a tensor is, for admission (matrices carry the quantised bytes; the
/// small ones are F32/F16/BF16 in every real export).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorClass {
    Matrix,
    Norm,
    Vector,
    Conv,
}

/// How the value-head axis was re-ordered by the exporter (only when
/// `linear_num_key_heads != linear_num_value_heads`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VHeadOrder {
    /// No re-ordering applies to this tensor.
    Native,
    /// All output rows are value-head major (`in_proj_z`, `in_proj_a/b`).
    TiledRows { head_dim: usize },
    /// Only the V rows of the fused `in_proj_qkv` (after the Q and K rows).
    TiledQkvValueRows { head_dim: usize },
    /// Input columns (`out_proj`).
    TiledCols { head_dim: usize },
    /// Only the V channels of the conv kernel (after the Q and K channels).
    TiledConvValueChannels { head_dim: usize },
    /// A 1-D per-value-head vector (`A_log`, `dt_bias`).
    TiledVector,
}

/// Data edits the exporter applied; the GGUF bytes are not the HF bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufTransform {
    /// `+1` folded into the stored RMSNorm weight.
    pub norm_plus_one: bool,
    /// Stored as `-exp(A_log)`.
    pub a_log_neg_exp: bool,
    /// Conv kernel squeezed from `[conv_dim, 1, K]` to `[conv_dim, K]`.
    pub conv_squeezed: bool,
    pub v_order: VHeadOrder,
}

impl GgufTransform {
    pub const AS_STORED: Self = Self {
        norm_plus_one: false,
        a_log_neg_exp: false,
        conv_squeezed: false,
        v_order: VHeadOrder::Native,
    };

    /// Whether the tensor's bytes are in the checkpoint's own layout.
    pub fn is_as_stored(&self) -> bool {
        *self == Self::AS_STORED
    }
}

/// One tensor a canonical export contains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedTensor {
    /// GGUF name, e.g. `blk.3.ssm_out.weight`.
    pub gguf_name: String,
    /// hipfire name (see the module docs).
    pub hipfire_name: String,
    /// Trunk layer index for trunk per-layer tensors; `None` for the
    /// embedding, final norm, LM head and every MTP tensor.
    pub layer: Option<usize>,
    /// Shape in GGUF (ggml `ne`) order: `ne[0]` is the contiguous axis.
    pub ggml_shape: Vec<usize>,
    pub group: TensorGroup,
    /// Absent is legal (tied LM head; MTP embedding/head that share the trunk's).
    pub optional: bool,
    pub class: TensorClass,
    pub transform: GgufTransform,
}

/// The full tensor contract of one dense Qwen3.5 export.
#[derive(Debug, Clone)]
pub struct Qwen35GgufMap {
    pub tensors: Vec<ExpectedTensor>,
    by_gguf_name: HashMap<String, usize>,
}

impl Qwen35GgufMap {
    pub fn new(hp: &Qwen35GgufHparams) -> Self {
        let tensors = expected_tensors(hp);
        let by_gguf_name = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.gguf_name.clone(), i))
            .collect();
        Self {
            tensors,
            by_gguf_name,
        }
    }

    pub fn lookup(&self, gguf_name: &str) -> Option<&ExpectedTensor> {
        self.by_gguf_name.get(gguf_name).map(|&i| &self.tensors[i])
    }
}

struct Builder<'a> {
    out: Vec<ExpectedTensor>,
    cfg: &'a Qwen35Config,
}

impl Builder<'_> {
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        gguf_name: String,
        hipfire_name: String,
        layer: Option<usize>,
        shape: &[usize],
        group: TensorGroup,
        optional: bool,
        class: TensorClass,
        transform: GgufTransform,
    ) {
        self.out.push(ExpectedTensor {
            gguf_name,
            hipfire_name,
            layer,
            ggml_shape: shape.to_vec(),
            group,
            optional,
            class,
            transform,
        });
    }

    /// One trunk or MTP block tensor. `blk` is the GGUF block index, `hf` the
    /// name relative to the layer (or the sidecar's bare name for MTP).
    fn layer_tensor(
        &mut self,
        blk: usize,
        group: TensorGroup,
        gguf_suffix: &str,
        hf_suffix: &str,
        shape: &[usize],
        class: TensorClass,
        transform: GgufTransform,
    ) {
        let (hipfire_name, layer) = match group {
            TensorGroup::Trunk => (
                format!("model.language_model.layers.{blk}.{hf_suffix}"),
                Some(blk),
            ),
            TensorGroup::Mtp => (format!("mtp.{hf_suffix}"), None),
        };
        self.push(
            format!("blk.{blk}.{gguf_suffix}"),
            hipfire_name,
            layer,
            shape,
            group,
            false,
            class,
            transform,
        );
    }

    /// The tensors shared by trunk full-attention layers and the MTP block.
    /// `trunk_hf` names the HF role; `mtp_hf` the sidecar's bare name.
    fn full_attention_block(&mut self, blk: usize, group: TensorGroup) {
        let c = self.cfg;
        let (dim, h, kvh, hd) = (c.dim, c.n_heads, c.n_kv_heads, c.head_dim);
        let plus_one = GgufTransform {
            norm_plus_one: true,
            ..GgufTransform::AS_STORED
        };
        let mtp = group == TensorGroup::Mtp;
        let pick = |trunk: &'static str, bare: &'static str| if mtp { bare } else { trunk };
        use TensorClass::{Matrix, Norm};
        self.layer_tensor(
            blk,
            group,
            "attn_norm.weight",
            pick("input_layernorm.weight", "attn_norm"),
            &[dim],
            Norm,
            plus_one,
        );
        self.layer_tensor(
            blk,
            group,
            "post_attention_norm.weight",
            pick("post_attention_layernorm.weight", "attn_post_norm"),
            &[dim],
            Norm,
            plus_one,
        );
        // q_proj carries the output gate: 2 * heads * head_dim rows.
        self.layer_tensor(
            blk,
            group,
            "attn_q.weight",
            pick("self_attn.q_proj.weight", "wq"),
            &[dim, 2 * h * hd],
            Matrix,
            GgufTransform::AS_STORED,
        );
        self.layer_tensor(
            blk,
            group,
            "attn_k.weight",
            pick("self_attn.k_proj.weight", "wk"),
            &[dim, kvh * hd],
            Matrix,
            GgufTransform::AS_STORED,
        );
        self.layer_tensor(
            blk,
            group,
            "attn_v.weight",
            pick("self_attn.v_proj.weight", "wv"),
            &[dim, kvh * hd],
            Matrix,
            GgufTransform::AS_STORED,
        );
        self.layer_tensor(
            blk,
            group,
            "attn_output.weight",
            pick("self_attn.o_proj.weight", "wo"),
            &[h * hd, dim],
            Matrix,
            GgufTransform::AS_STORED,
        );
        self.layer_tensor(
            blk,
            group,
            "attn_q_norm.weight",
            pick("self_attn.q_norm.weight", "attn_q_norm"),
            &[hd],
            Norm,
            plus_one,
        );
        self.layer_tensor(
            blk,
            group,
            "attn_k_norm.weight",
            pick("self_attn.k_norm.weight", "attn_k_norm"),
            &[hd],
            Norm,
            plus_one,
        );
        let (ff, dim) = (c.hidden_dim, c.dim);
        self.layer_tensor(
            blk,
            group,
            "ffn_gate.weight",
            pick("mlp.gate_proj.weight", "ffn_gate"),
            &[dim, ff],
            Matrix,
            GgufTransform::AS_STORED,
        );
        self.layer_tensor(
            blk,
            group,
            "ffn_up.weight",
            pick("mlp.up_proj.weight", "ffn_up"),
            &[dim, ff],
            Matrix,
            GgufTransform::AS_STORED,
        );
        self.layer_tensor(
            blk,
            group,
            "ffn_down.weight",
            pick("mlp.down_proj.weight", "ffn_down"),
            &[ff, dim],
            Matrix,
            GgufTransform::AS_STORED,
        );
    }

    fn delta_net_block(&mut self, blk: usize) {
        let c = self.cfg;
        let (dim, ff) = (c.dim, c.hidden_dim);
        let (nk, nv) = (c.linear_num_key_heads, c.linear_num_value_heads);
        let (kd, vd) = (c.linear_key_head_dim, c.linear_value_head_dim);
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = 2 * key_dim + value_dim;
        let reordered = nk != nv;
        let order = |o: VHeadOrder| if reordered { o } else { VHeadOrder::Native };
        let plus_one = GgufTransform {
            norm_plus_one: true,
            ..GgufTransform::AS_STORED
        };
        let with_order = |o: VHeadOrder| GgufTransform {
            v_order: order(o),
            ..GgufTransform::AS_STORED
        };
        use TensorClass::{Conv, Matrix, Norm, Vector};
        let g = TensorGroup::Trunk;
        self.layer_tensor(
            blk,
            g,
            "attn_norm.weight",
            "input_layernorm.weight",
            &[dim],
            Norm,
            plus_one,
        );
        self.layer_tensor(
            blk,
            g,
            "post_attention_norm.weight",
            "post_attention_layernorm.weight",
            &[dim],
            Norm,
            plus_one,
        );
        self.layer_tensor(
            blk,
            g,
            "attn_qkv.weight",
            "linear_attn.in_proj_qkv.weight",
            &[dim, conv_dim],
            Matrix,
            with_order(VHeadOrder::TiledQkvValueRows { head_dim: vd }),
        );
        self.layer_tensor(
            blk,
            g,
            "attn_gate.weight",
            "linear_attn.in_proj_z.weight",
            &[dim, value_dim],
            Matrix,
            with_order(VHeadOrder::TiledRows { head_dim: vd }),
        );
        self.layer_tensor(
            blk,
            g,
            "ssm_conv1d.weight",
            "linear_attn.conv1d.weight",
            &[c.conv_kernel_dim, conv_dim],
            Conv,
            GgufTransform {
                conv_squeezed: true,
                v_order: order(VHeadOrder::TiledConvValueChannels { head_dim: vd }),
                ..GgufTransform::AS_STORED
            },
        );
        self.layer_tensor(
            blk,
            g,
            "ssm_dt.bias",
            "linear_attn.dt_bias",
            &[nv],
            Vector,
            with_order(VHeadOrder::TiledVector),
        );
        self.layer_tensor(
            blk,
            g,
            "ssm_a",
            "linear_attn.A_log",
            &[nv],
            Vector,
            GgufTransform {
                a_log_neg_exp: true,
                v_order: order(VHeadOrder::TiledVector),
                ..GgufTransform::AS_STORED
            },
        );
        self.layer_tensor(
            blk,
            g,
            "ssm_beta.weight",
            "linear_attn.in_proj_b.weight",
            &[dim, nv],
            Matrix,
            with_order(VHeadOrder::TiledRows { head_dim: 1 }),
        );
        self.layer_tensor(
            blk,
            g,
            "ssm_alpha.weight",
            "linear_attn.in_proj_a.weight",
            &[dim, nv],
            Matrix,
            with_order(VHeadOrder::TiledRows { head_dim: 1 }),
        );
        // `linear_attn.norm` is the one norm the exporter leaves un-folded.
        self.layer_tensor(
            blk,
            g,
            "ssm_norm.weight",
            "linear_attn.norm.weight",
            &[vd],
            Norm,
            GgufTransform::AS_STORED,
        );
        self.layer_tensor(
            blk,
            g,
            "ssm_out.weight",
            "linear_attn.out_proj.weight",
            &[value_dim, dim],
            Matrix,
            with_order(VHeadOrder::TiledCols { head_dim: vd }),
        );
        self.layer_tensor(
            blk,
            g,
            "ffn_gate.weight",
            "mlp.gate_proj.weight",
            &[dim, ff],
            Matrix,
            GgufTransform::AS_STORED,
        );
        self.layer_tensor(
            blk,
            g,
            "ffn_up.weight",
            "mlp.up_proj.weight",
            &[dim, ff],
            Matrix,
            GgufTransform::AS_STORED,
        );
        self.layer_tensor(
            blk,
            g,
            "ffn_down.weight",
            "mlp.down_proj.weight",
            &[ff, dim],
            Matrix,
            GgufTransform::AS_STORED,
        );
    }
}

/// Every tensor a canonical dense Qwen3.5 export contains, in file order
/// (global tensors, then trunk layers, then the MTP block).
pub fn expected_tensors(hp: &Qwen35GgufHparams) -> Vec<ExpectedTensor> {
    let c = &hp.config;
    let mut b = Builder {
        out: Vec::new(),
        cfg: c,
    };
    let plus_one = GgufTransform {
        norm_plus_one: true,
        ..GgufTransform::AS_STORED
    };
    let (dim, vocab) = (c.dim, c.vocab_size);
    b.push(
        "token_embd.weight".into(),
        "model.language_model.embed_tokens.weight".into(),
        None,
        &[dim, vocab],
        TensorGroup::Trunk,
        false,
        TensorClass::Matrix,
        GgufTransform::AS_STORED,
    );
    b.push(
        "output_norm.weight".into(),
        "model.language_model.norm.weight".into(),
        None,
        &[dim],
        TensorGroup::Trunk,
        false,
        TensorClass::Norm,
        plus_one,
    );
    // Absent = tied to the embedding.
    b.push(
        "output.weight".into(),
        "lm_head.weight".into(),
        None,
        &[dim, vocab],
        TensorGroup::Trunk,
        true,
        TensorClass::Matrix,
        GgufTransform::AS_STORED,
    );
    for (blk, kind) in c.layer_types.iter().enumerate() {
        match kind {
            LayerType::FullAttention => b.full_attention_block(blk, TensorGroup::Trunk),
            LayerType::LinearAttention => b.delta_net_block(blk),
        }
    }
    for i in 0..hp.nextn_layers {
        let blk = hp.n_layers + i;
        b.full_attention_block(blk, TensorGroup::Mtp);
        let g = TensorGroup::Mtp;
        let pre = format!("blk.{blk}.nextn");
        for (suffix, bare, shape, class, optional) in [
            (
                "eh_proj.weight",
                "eh_proj",
                vec![2 * dim, dim],
                TensorClass::Matrix,
                false,
            ),
            ("enorm.weight", "enorm", vec![dim], TensorClass::Norm, false),
            ("hnorm.weight", "hnorm", vec![dim], TensorClass::Norm, false),
            (
                "shared_head_norm.weight",
                "shared_head_norm",
                vec![dim],
                TensorClass::Norm,
                true,
            ),
            (
                "embed_tokens.weight",
                "embed_tokens",
                vec![dim, vocab],
                TensorClass::Matrix,
                true,
            ),
            (
                "shared_head_head.weight",
                "shared_head_head",
                vec![dim, vocab],
                TensorClass::Matrix,
                true,
            ),
        ] {
            let transform = if class == TensorClass::Norm {
                plus_one
            } else {
                GgufTransform::AS_STORED
            };
            b.push(
                format!("{pre}.{suffix}"),
                format!("mtp.{bare}"),
                None,
                &shape,
                g,
                optional,
                class,
                transform,
            );
        }
    }
    b.out
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::gguf::GgmlType;

    fn meta(entries: &[(&str, MetaValue)]) -> HashMap<String, MetaValue> {
        entries
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    /// A small hybrid config: 2 layers (GDN, full attention), grouped value
    /// heads (nk=2, nv=4) so the value-head re-ordering applies.
    pub(crate) fn tiny_meta(nextn: usize) -> HashMap<String, MetaValue> {
        let mut m = meta(&[
            ("qwen35.block_count", MetaValue::U32(2 + nextn as u32)),
            ("qwen35.embedding_length", MetaValue::U32(256)),
            ("qwen35.feed_forward_length", MetaValue::U32(512)),
            ("qwen35.attention.head_count", MetaValue::U32(4)),
            ("qwen35.attention.head_count_kv", MetaValue::U32(2)),
            ("qwen35.attention.key_length", MetaValue::U32(64)),
            ("qwen35.attention.value_length", MetaValue::U32(64)),
            (
                "qwen35.attention.layer_norm_rms_epsilon",
                MetaValue::F32(1e-6),
            ),
            ("qwen35.rope.dimension_count", MetaValue::U32(16)),
            ("qwen35.rope.freq_base", MetaValue::F32(10_000_000.0)),
            (
                "qwen35.rope.dimension_sections",
                MetaValue::Array(vec![
                    MetaValue::I32(2),
                    MetaValue::I32(3),
                    MetaValue::I32(3),
                    MetaValue::I32(0),
                ]),
            ),
            ("qwen35.ssm.conv_kernel", MetaValue::U32(4)),
            ("qwen35.ssm.state_size", MetaValue::U32(32)),
            ("qwen35.ssm.group_count", MetaValue::U32(2)),
            ("qwen35.ssm.time_step_rank", MetaValue::U32(4)),
            ("qwen35.ssm.inner_size", MetaValue::U32(128)),
            ("qwen35.full_attention_interval", MetaValue::U32(2)),
            ("tokenizer.ggml.eos_token_id", MetaValue::U32(7)),
        ]);
        if nextn > 0 {
            m.insert(
                "qwen35.nextn_predict_layers".to_string(),
                MetaValue::U32(nextn as u32),
            );
        }
        m
    }

    pub(crate) fn embd_info(vocab: usize) -> TensorInfo {
        TensorInfo {
            name: "token_embd.weight".to_string(),
            shape: vec![256, vocab],
            dtype: GgmlType::Q8_0,
            offset: 0,
        }
    }

    #[test]
    fn hparams_fill_the_shared_config_struct() {
        let hp = hparams_from_gguf(&tiny_meta(0), &[embd_info(1000)]).unwrap();
        let c = &hp.config;
        assert_eq!((c.dim, c.n_layers, c.vocab_size), (256, 2, 1000));
        assert_eq!((c.n_heads, c.n_kv_heads, c.head_dim), (4, 2, 64));
        assert_eq!(c.hidden_dim, 512);
        assert_eq!((c.linear_num_key_heads, c.linear_num_value_heads), (2, 4));
        assert_eq!((c.linear_key_head_dim, c.linear_value_head_dim), (32, 32));
        assert_eq!(c.conv_kernel_dim, 4);
        assert_eq!(c.partial_rotary_factor, 0.25);
        assert_eq!(c.mrope_section, [2, 3, 3]);
        assert!(c.mrope_interleaved);
        assert_eq!(c.eos_token, 7);
        assert_eq!(c.num_experts, 0);
        // interval 2 => layers 0 linear, 1 full.
        assert_eq!(
            c.layer_types,
            vec![LayerType::LinearAttention, LayerType::FullAttention]
        );
        assert_eq!(hp.nextn_layers, 0);
    }

    #[test]
    fn recurrent_layers_array_overrides_the_interval() {
        let mut m = tiny_meta(0);
        m.insert(
            "qwen35.attention.recurrent_layers".to_string(),
            MetaValue::Array(vec![MetaValue::Bool(false), MetaValue::Bool(true)]),
        );
        let hp = hparams_from_gguf(&m, &[embd_info(10)]).unwrap();
        assert_eq!(
            hp.config.layer_types,
            vec![LayerType::FullAttention, LayerType::LinearAttention]
        );
    }

    #[test]
    fn missing_or_moe_metadata_refuses_naming_the_key() {
        let mut m = tiny_meta(0);
        m.remove("qwen35.ssm.state_size");
        let e = hparams_from_gguf(&m, &[embd_info(10)]).unwrap_err();
        assert!(e.contains("qwen35.ssm.state_size"), "{e}");

        let mut m = tiny_meta(0);
        m.insert("qwen35.expert_count".to_string(), MetaValue::U32(8));
        let e = hparams_from_gguf(&m, &[embd_info(10)]).unwrap_err();
        assert!(e.contains("expert_count=8"), "{e}");

        let e = hparams_from_gguf(&tiny_meta(0), &[]).unwrap_err();
        assert!(e.contains("token_embd.weight"), "{e}");
    }

    #[test]
    fn expected_tensor_set_for_two_layers_and_mtp() {
        let hp = hparams_from_gguf(&tiny_meta(1), &[embd_info(1000)]).unwrap();
        assert_eq!(hp.n_layers, 2);
        assert_eq!(hp.nextn_layers, 1);
        let map = Qwen35GgufMap::new(&hp);
        // 3 globals + GDN layer (14) + full layer (10 + ... ) + MTP.
        let names: Vec<&str> = map.tensors.iter().map(|t| t.gguf_name.as_str()).collect();
        for want in [
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.attn_gate.weight",
            "blk.0.ssm_a",
            "blk.0.ssm_dt.bias",
            "blk.0.ssm_conv1d.weight",
            "blk.0.ssm_beta.weight",
            "blk.0.ssm_alpha.weight",
            "blk.0.ssm_norm.weight",
            "blk.0.ssm_out.weight",
            "blk.0.ffn_down.weight",
            "blk.1.attn_q.weight",
            "blk.1.attn_k_norm.weight",
            "blk.1.attn_output.weight",
            "blk.2.attn_q.weight",
            "blk.2.nextn.eh_proj.weight",
            "blk.2.nextn.shared_head_norm.weight",
        ] {
            assert!(names.contains(&want), "{want} missing from {names:?}");
        }
        // No duplicate names on either side of the map.
        let mut g = names.clone();
        g.sort_unstable();
        g.dedup();
        assert_eq!(g.len(), names.len());
        let mut h: Vec<&str> = map
            .tensors
            .iter()
            .map(|t| t.hipfire_name.as_str())
            .collect();
        h.sort_unstable();
        h.dedup();
        assert_eq!(h.len(), names.len());

        let q = map.lookup("blk.0.attn_qkv.weight").unwrap();
        assert_eq!(
            q.hipfire_name,
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        );
        // conv_dim = 2 * (2*32) + 4*32 = 256
        assert_eq!(q.ggml_shape, vec![256, 256]);
        assert_eq!(
            q.transform.v_order,
            VHeadOrder::TiledQkvValueRows { head_dim: 32 }
        );
        let a = map.lookup("blk.0.ssm_a").unwrap();
        assert_eq!(
            a.hipfire_name,
            "model.language_model.layers.0.linear_attn.A_log"
        );
        assert!(a.transform.a_log_neg_exp);
        let n = map.lookup("blk.0.ssm_norm.weight").unwrap();
        assert!(n.transform.is_as_stored(), "linear_attn.norm is not folded");
        assert!(
            map.lookup("blk.0.attn_norm.weight")
                .unwrap()
                .transform
                .norm_plus_one
        );
        let m = map.lookup("blk.2.nextn.eh_proj.weight").unwrap();
        assert_eq!((m.hipfire_name.as_str(), m.layer), ("mtp.eh_proj", None));
        assert_eq!(m.ggml_shape, vec![512, 256]);
        assert_eq!(
            map.lookup("blk.2.attn_q.weight").unwrap().hipfire_name,
            "mtp.wq"
        );
        assert!(map.lookup("output.weight").unwrap().optional);
        assert_eq!(map.lookup("blk.9.attn_q.weight"), None);
    }

    #[test]
    fn equal_key_and_value_heads_need_no_reordering() {
        let mut m = tiny_meta(0);
        m.insert("qwen35.ssm.group_count".to_string(), MetaValue::U32(4));
        let hp = hparams_from_gguf(&m, &[embd_info(10)]).unwrap();
        let map = Qwen35GgufMap::new(&hp);
        assert_eq!(
            map.lookup("blk.0.attn_gate.weight")
                .unwrap()
                .transform
                .v_order,
            VHeadOrder::Native
        );
    }
}
