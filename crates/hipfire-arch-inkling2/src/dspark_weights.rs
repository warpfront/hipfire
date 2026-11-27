// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Inkling-2-Mini native DSpark draft sidecar (`<trunk-stem>-dspark.hfq`).
//!
//! Frozen v1 format (`inkling/MTP-PLAN.md` §3): the 54 source tensors of the
//! pinned `mtp.safetensors`, names and BF16 bytes unchanged, followed by the
//! `hidden_out` fold into the paired trunk's effective unembed. The HFQ carries
//! arch id 24 (the paired trunk arch) and the mandatory `dspark_body`
//! discriminator, so it is never a standalone trunk.
//!
//! The sidecar binds to one trunk: source repo/revision and the source trunk
//! and draft checkpoint digests, the tokenizer, the fixed loop schedule, the
//! trunk layout, and the embed / embed norm bytes it reuses. The fold is keyed
//! separately to the unembed bytes it was derived from, the `hidden_out` bytes
//! and the fold recipe. [`InklingDsparkWeights::load`] checks every one of
//! these plus the exact tensor layout, and fails closed on any mismatch. The
//! same manifest ([`InklingDsparkConfig::manifest`]) drives the packer.

use crate::weights::RawTensor;
use hipfire_runtime::hfq::HfqFile;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::borrow::Cow;

/// Mandatory body discriminator (`config.dspark_body`).
pub const DSPARK_BODY: &str = "inkling2-loop-rnn-hidden-v1";
/// Version of the `hipfire_inkling2_dspark` manifest object.
pub const DSPARK_FORMAT_VERSION: u64 = 1;
/// Top-level metadata key of the binding manifest.
pub const DSPARK_MANIFEST_KEY: &str = "hipfire_inkling2_dspark";
/// Top-level `architecture` tag of the sidecar.
pub const DSPARK_ARCHITECTURE: &str = "inkling2_dspark";
/// Fold derivation: `fold = U · W_hidden_out`, `fold_bias = U · b_hidden_out`
/// with `U` the trunk unembed decoded to F32, F64 accumulation over the hidden
/// index in ascending order, one rounding to BF16 (nearest-even).
pub const FOLD_RECIPE: &str = "unembed-x-hidden_out-f64-ascending-bf16-rne-v1";
/// Folded `W_lm · W_hidden_out`, `[vocab, markov_rank]`, BF16.
pub const FOLD_WEIGHT: &str = "markov_head.hidden_out_fold.weight";
/// Folded `W_lm · b_hidden_out`, `[vocab]`, BF16.
pub const FOLD_BIAS: &str = "markov_head.hidden_out_fold.bias";
/// Trunk tensors the draft reuses.
pub const TRUNK_EMBED: &str = "model.llm.embed.weight";
pub const TRUNK_EMBED_NORM: &str = "model.llm.embed_norm.weight";
pub const TRUNK_UNEMBED: &str = "model.llm.unembed.weight";
/// Trunk layout key (absent = the legacy incumbent layout).
pub const TRUNK_LAYOUT_KEY: &str = "hipfire_inkling2_mq4_layout";
pub const TRUNK_LAYOUT_LEGACY: &str = "legacy";
pub const TRUNK_LAYOUT_XTS: &str = "qt44-padded-awq-v1";
/// Physical block capacity (anchor + mask rows) and proposal capacity.
pub const BLOCK_CAPACITY: usize = 8;
pub const PROPOSAL_CAPACITY: usize = 7;
/// HFQ quant type byte of raw BF16.
pub const QT_BF16: u8 = 16;

/// Pinned source of the draft and the trunk it was trained against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InklingDsparkSource {
    pub repo: String,
    pub revision: String,
    /// Source trunk checkpoint file name and its SHA-256.
    pub trunk_file: String,
    pub trunk_sha256: String,
    /// `mtp.safetensors` whole-file SHA-256 and its JSON header SHA-256.
    pub draft_sha256: String,
    pub draft_header_sha256: String,
}

impl InklingDsparkSource {
    /// `thinkingmachines/Inkling-2-Mini@c9d02830`.
    pub fn mini() -> Self {
        Self {
            repo: "thinkingmachines/Inkling-2-Mini".into(),
            revision: "c9d028302cd0f035fdf8e7e91f089a4ce94d2195".into(),
            trunk_file: "model-00001-of-00001.safetensors".into(),
            trunk_sha256: "65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1".into(),
            draft_sha256: "7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2".into(),
            draft_header_sha256: "753b21c80ea24bce28d63e40a64e49fc27b4a9c59b855837b99e563868dc4bc1"
                .into(),
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "repo": self.repo,
            "revision": self.revision,
            "trunk_file": self.trunk_file,
            "trunk_sha256": self.trunk_sha256,
            "draft_sha256": self.draft_sha256,
            "draft_header_sha256": self.draft_header_sha256,
        })
    }

    pub fn from_json(v: &Value) -> Result<Self, String> {
        Ok(Self {
            repo: str_field(v, "repo")?,
            revision: str_field(v, "revision")?,
            trunk_file: str_field(v, "trunk_file")?,
            trunk_sha256: str_field(v, "trunk_sha256")?,
            draft_sha256: str_field(v, "draft_sha256")?,
            draft_header_sha256: str_field(v, "draft_header_sha256")?,
        })
    }
}

/// Typed draft geometry (`config.mtp_config` plus the shared target vocab and
/// softcap). Block capacity 8 and proposal capacity 7 are separate fields.
#[derive(Debug, Clone, PartialEq)]
pub struct InklingDsparkConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub local_layer_ids: Vec<usize>,
    pub sliding_window_size: usize,
    pub max_position_embeddings: usize,
    pub block_size: usize,
    pub proposal_capacity: usize,
    pub markov_rank: usize,
    pub markov_head_type: String,
    pub target_layer_ids: Vec<usize>,
    pub vocab_size: usize,
    pub unpadded_vocab_size: usize,
    pub final_logit_softcapping: f64,
}

impl InklingDsparkConfig {
    /// Parse from a source/trunk `config` object (`config.json`, or the HFQ
    /// metadata `config` the trunk quantizer embeds verbatim).
    pub fn from_source_config(cfg: &Value) -> Result<Self, String> {
        let m = cfg
            .get("mtp_config")
            .ok_or("inkling2 dspark: config has no mtp_config")?;
        let t = cfg
            .get("text_config")
            .ok_or("inkling2 dspark: config has no text_config")?;
        for key in ["q_bias", "o_bias"] {
            if m.get(key).and_then(Value::as_bool) != Some(false) {
                return Err(format!("inkling2 dspark: mtp_config.{key} must be false"));
            }
        }
        let block_size = usize_field(m, "block_size")?;
        let c = Self {
            hidden_size: usize_field(m, "hidden_size")?,
            num_hidden_layers: usize_field(m, "num_hidden_layers")?,
            num_attention_heads: usize_field(m, "num_attention_heads")?,
            num_key_value_heads: usize_field(m, "num_key_value_heads")?,
            head_dim: usize_field(m, "head_dim")?,
            intermediate_size: usize_field(m, "intermediate_size")?,
            rms_norm_eps: f64_field(m, "rms_norm_eps")?,
            rope_theta: f64_field(m, "rope_theta")?,
            local_layer_ids: usize_list(m, "local_layer_ids")?,
            sliding_window_size: usize_field(m, "sliding_window_size")?,
            max_position_embeddings: usize_field(m, "max_position_embeddings")?,
            block_size,
            proposal_capacity: block_size.saturating_sub(1),
            markov_rank: usize_field(m, "markov_rank")?,
            markov_head_type: str_field(m, "markov_head_type")?,
            target_layer_ids: usize_list(m, "target_layer_ids")?,
            vocab_size: usize_field(t, "vocab_size")?,
            unpadded_vocab_size: usize_field(t, "unpadded_vocab_size")?,
            final_logit_softcapping: f64_field(t, "final_logit_softcapping")?,
        };
        if usize_field(t, "hidden_size")? != c.hidden_size {
            return Err("inkling2 dspark: mtp_config.hidden_size differs from the trunk".into());
        }
        c.validate()?;
        Ok(c)
    }

    /// Invariants of the v1 body.
    fn validate(&self) -> Result<(), String> {
        let fail = |what: String| Err(format!("inkling2 dspark: {what}"));
        if self.block_size != BLOCK_CAPACITY || self.proposal_capacity != PROPOSAL_CAPACITY {
            return fail(format!(
                "block {} / proposals {} (v1 freezes {BLOCK_CAPACITY} / {PROPOSAL_CAPACITY})",
                self.block_size, self.proposal_capacity
            ));
        }
        if self.markov_head_type != "rnn_hidden" {
            return fail(format!("markov_head_type {:?}", self.markov_head_type));
        }
        if self.num_hidden_layers == 0 || self.target_layer_ids.is_empty() {
            return fail("empty draft tower or tap list".into());
        }
        if self.target_layer_ids.windows(2).any(|w| w[0] >= w[1]) {
            return fail(format!("taps {:?} not increasing", self.target_layer_ids));
        }
        if self
            .local_layer_ids
            .iter()
            .any(|&l| l >= self.num_hidden_layers)
        {
            return fail(format!("local layers {:?}", self.local_layer_ids));
        }
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.num_attention_heads % self.num_key_value_heads != 0
            || self.unpadded_vocab_size == 0
            || self.unpadded_vocab_size > self.vocab_size
            || self.markov_rank == 0
        {
            return fail("inconsistent head/vocab/rank geometry".into());
        }
        Ok(())
    }

    /// `config` keys of the sidecar (DSpark sidecar convention: `dspark_*`).
    pub fn to_metadata(&self) -> Value {
        json!({
            "dspark_body": DSPARK_BODY,
            "dspark_block_size": self.block_size,
            "dspark_proposal_capacity": self.proposal_capacity,
            "dspark_target_layer_ids": self.target_layer_ids,
            "dspark_num_targets": self.target_layer_ids.len(),
            "dspark_markov_rank": self.markov_rank,
            "dspark_markov_head_type": self.markov_head_type,
            "dspark_hidden_size": self.hidden_size,
            "dspark_num_hidden_layers": self.num_hidden_layers,
            "dspark_num_attention_heads": self.num_attention_heads,
            "dspark_num_key_value_heads": self.num_key_value_heads,
            "dspark_head_dim": self.head_dim,
            "dspark_intermediate_size": self.intermediate_size,
            "dspark_rms_norm_eps": self.rms_norm_eps,
            "dspark_rope_theta": self.rope_theta,
            "dspark_local_layer_ids": self.local_layer_ids,
            "dspark_sliding_window_size": self.sliding_window_size,
            "dspark_max_position_embeddings": self.max_position_embeddings,
            "dspark_vocab_size": self.vocab_size,
            "dspark_unpadded_vocab_size": self.unpadded_vocab_size,
            "dspark_final_logit_softcapping": self.final_logit_softcapping,
        })
    }

    /// Inverse of [`Self::to_metadata`]; requires the body discriminator.
    pub fn from_metadata(cfg: &Value) -> Result<Self, String> {
        match cfg.get("dspark_body").and_then(Value::as_str) {
            Some(DSPARK_BODY) => {}
            Some(other) => {
                return Err(format!(
                    "inkling2 dspark: dspark_body {other:?} is not {DSPARK_BODY:?}"
                ))
            }
            None => {
                return Err(
                    "inkling2 dspark: not an Inkling DSpark sidecar (no config.dspark_body)".into(),
                )
            }
        }
        let c = Self {
            hidden_size: usize_field(cfg, "dspark_hidden_size")?,
            num_hidden_layers: usize_field(cfg, "dspark_num_hidden_layers")?,
            num_attention_heads: usize_field(cfg, "dspark_num_attention_heads")?,
            num_key_value_heads: usize_field(cfg, "dspark_num_key_value_heads")?,
            head_dim: usize_field(cfg, "dspark_head_dim")?,
            intermediate_size: usize_field(cfg, "dspark_intermediate_size")?,
            rms_norm_eps: f64_field(cfg, "dspark_rms_norm_eps")?,
            rope_theta: f64_field(cfg, "dspark_rope_theta")?,
            local_layer_ids: usize_list(cfg, "dspark_local_layer_ids")?,
            sliding_window_size: usize_field(cfg, "dspark_sliding_window_size")?,
            max_position_embeddings: usize_field(cfg, "dspark_max_position_embeddings")?,
            block_size: usize_field(cfg, "dspark_block_size")?,
            proposal_capacity: usize_field(cfg, "dspark_proposal_capacity")?,
            markov_rank: usize_field(cfg, "dspark_markov_rank")?,
            markov_head_type: str_field(cfg, "dspark_markov_head_type")?,
            target_layer_ids: usize_list(cfg, "dspark_target_layer_ids")?,
            vocab_size: usize_field(cfg, "dspark_vocab_size")?,
            unpadded_vocab_size: usize_field(cfg, "dspark_unpadded_vocab_size")?,
            final_logit_softcapping: f64_field(cfg, "dspark_final_logit_softcapping")?,
        };
        if usize_field(cfg, "dspark_num_targets")? != c.target_layer_ids.len() {
            return Err("inkling2 dspark: dspark_num_targets disagrees with the tap list".into());
        }
        c.validate()?;
        Ok(c)
    }

    /// The source tensors in canonical (= sidecar storage) order, with shapes.
    /// For the Mini geometry this is exactly the 54 BF16 tensors of
    /// `mtp.safetensors`.
    pub fn manifest(&self) -> Vec<(String, Vec<u32>)> {
        let d = self.hidden_size as u32;
        let hd = self.head_dim as u32;
        let q = self.num_attention_heads as u32 * hd;
        let kv = self.num_key_value_heads as u32 * hd;
        let ff = self.intermediate_size as u32;
        let r = self.markov_rank as u32;
        let fuse_rows = self.num_hidden_layers as u32 + 1;
        let mut out = vec![(
            "fuse_weight".to_string(),
            vec![fuse_rows, self.target_layer_ids.len() as u32],
        )];
        for i in 0..self.num_hidden_layers {
            let p = format!("layers.{i}.");
            for (leaf, shape) in [
                ("dflare_norm.weight", vec![d]),
                ("input_layernorm.weight", vec![d]),
                ("post_attention_layernorm.weight", vec![d]),
                ("mlp.down_proj.weight", vec![d, ff]),
                ("mlp.gate_proj.weight", vec![ff, d]),
                ("mlp.up_proj.weight", vec![ff, d]),
                ("self_attn.k_norm.weight", vec![hd]),
                ("self_attn.k_norm_ctx.weight", vec![hd]),
                ("self_attn.k_proj.weight", vec![kv, d]),
                ("self_attn.k_proj_ctx.weight", vec![kv, d]),
                ("self_attn.o_proj.weight", vec![d, q]),
                ("self_attn.q_norm.weight", vec![hd]),
                ("self_attn.q_proj.weight", vec![q, d]),
                ("self_attn.v_proj.weight", vec![kv, d]),
                ("self_attn.v_proj_ctx.weight", vec![kv, d]),
            ] {
                out.push((format!("{p}{leaf}"), shape));
            }
        }
        for (name, shape) in [
            ("markov_head.dflare_norm.weight", vec![d]),
            ("markov_head.hidden_out.bias", vec![d]),
            ("markov_head.hidden_out.weight", vec![d, r]),
            (
                "markov_head.markov_w1.weight",
                vec![self.vocab_size as u32, r],
            ),
            ("markov_head.rnn_proj.bias", vec![3 * r]),
            ("markov_head.rnn_proj.weight", vec![3 * r, 2 * r + 2 * d]),
            ("mask_embed", vec![d]),
            ("norm.weight", vec![d]),
        ] {
            out.push((name.to_string(), shape));
        }
        out
    }

    /// The full sidecar tensor list: [`Self::manifest`] then the fold.
    pub fn sidecar_manifest(&self) -> Vec<(String, Vec<u32>)> {
        let mut out = self.manifest();
        out.push((
            FOLD_WEIGHT.to_string(),
            vec![self.vocab_size as u32, self.markov_rank as u32],
        ));
        out.push((FOLD_BIAS.to_string(), vec![self.vocab_size as u32]));
        out
    }
}

/// Encoding and SHA-256 of one tensor's stored bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InklingDsparkTensorDigest {
    pub name: String,
    pub quant_type: u8,
    pub shape: Vec<u32>,
    pub sha256: String,
}

impl InklingDsparkTensorDigest {
    pub fn of(name: &str, quant_type: u8, shape: &[u32], bytes: &[u8]) -> Self {
        Self {
            name: name.to_string(),
            quant_type,
            shape: shape.to_vec(),
            sha256: sha256_hex(bytes),
        }
    }

    fn of_hfq(hfq: &HfqFile, name: &str) -> Result<Self, String> {
        let (info, bytes) = tensor_bytes(hfq, name)?;
        Ok(Self::of(name, info.quant_type, &info.shape, &bytes))
    }

    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "quant_type": self.quant_type,
            "shape": self.shape,
            "sha256": self.sha256,
        })
    }

    pub fn from_json(v: &Value) -> Result<Self, String> {
        let quant_type = usize_field(v, "quant_type")?;
        Ok(Self {
            name: str_field(v, "name")?,
            quant_type: u8::try_from(quant_type)
                .map_err(|_| format!("inkling2 dspark: quant_type {quant_type}"))?,
            shape: usize_list(v, "shape")?
                .into_iter()
                .map(|d| u32::try_from(d).map_err(|_| format!("inkling2 dspark: dim {d}")))
                .collect::<Result<_, _>>()?,
            sha256: str_field(v, "sha256")?,
        })
    }
}

/// What the sidecar must match on the trunk side. The carrier derives it from
/// the loaded trunk with [`Self::from_trunk`] and the pinned source.
#[derive(Debug, Clone, PartialEq)]
pub struct InklingDsparkTargetIdentity {
    pub source: InklingDsparkSource,
    /// [`TRUNK_LAYOUT_LEGACY`] or [`TRUNK_LAYOUT_XTS`].
    pub layout: String,
    /// SHA-256 over canonical (key-sorted) JSON of the trunk's `tokenizer`,
    /// a NUL byte, then its `tokenizer_config`.
    pub tokenizer_sha256: String,
    /// Physical layer of every visit, in visit order.
    pub loop_schedule: Vec<usize>,
    /// Draft geometry the trunk's `config.mtp_config` declares.
    pub draft_config: InklingDsparkConfig,
    pub embed: InklingDsparkTensorDigest,
    pub embed_norm: InklingDsparkTensorDigest,
    pub unembed: InklingDsparkTensorDigest,
}

impl InklingDsparkTargetIdentity {
    /// Derive the identity of an Inkling-2 trunk HFQ (legacy or XTS layout).
    /// Reads metadata and the three bound tensors only.
    pub fn from_trunk(hfq: &HfqFile, source: InklingDsparkSource) -> Result<Self, String> {
        if hfq.arch_id != crate::config::ARCH_ID {
            return Err(format!(
                "inkling2 dspark: trunk arch id {} is not {}",
                hfq.arch_id,
                crate::config::ARCH_ID
            ));
        }
        let meta: Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("inkling2 dspark: bad trunk metadata JSON: {e}"))?;
        if meta["config"].get("dspark_body").is_some() {
            return Err("inkling2 dspark: the trunk is itself a DSpark sidecar".into());
        }
        let layout = match meta.get(TRUNK_LAYOUT_KEY) {
            None => TRUNK_LAYOUT_LEGACY.to_string(),
            Some(Value::String(s)) if s == TRUNK_LAYOUT_XTS => s.clone(),
            Some(other) => {
                return Err(format!(
                    "inkling2 dspark: unknown trunk {TRUNK_LAYOUT_KEY} {other}"
                ))
            }
        };
        let trunk_cfg = crate::config::Inkling2Config::from_metadata_json(&hfq.metadata_json)?;
        let draft_config = InklingDsparkConfig::from_source_config(&meta["config"])?;
        if draft_config.vocab_size != trunk_cfg.vocab_size
            || draft_config.unpadded_vocab_size != trunk_cfg.unpadded_vocab_size
            || draft_config.hidden_size != trunk_cfg.n_embd
        {
            return Err("inkling2 dspark: mtp_config geometry differs from the trunk".into());
        }
        // The fold is derived from the decoded unembed alone; an activation
        // scale on it would change the effective matrix.
        let awq = "model.llm.unembed.awq_scale.weight";
        if hfq.tensor_infos().iter().any(|t| t.name == awq) {
            return Err(format!(
                "inkling2 dspark: trunk carries {awq}; the v1 fold assumes a plain unembed"
            ));
        }
        let mut tok = Vec::new();
        for key in ["tokenizer", "tokenizer_config"] {
            let v = meta
                .get(key)
                .ok_or_else(|| format!("inkling2 dspark: trunk metadata has no {key}"))?;
            canonical_json(v, &mut tok);
            tok.push(0);
        }
        tok.pop();
        Ok(Self {
            source,
            layout,
            tokenizer_sha256: sha256_hex(&tok),
            loop_schedule: trunk_cfg.loop_cfg.visit_schedule(),
            draft_config,
            embed: InklingDsparkTensorDigest::of_hfq(hfq, TRUNK_EMBED)?,
            embed_norm: InklingDsparkTensorDigest::of_hfq(hfq, TRUNK_EMBED_NORM)?,
            unembed: InklingDsparkTensorDigest::of_hfq(hfq, TRUNK_UNEMBED)?,
        })
    }
}

/// Fold key and payload digests recorded by the packer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InklingDsparkFold {
    pub recipe: String,
    pub unembed: InklingDsparkTensorDigest,
    pub hidden_out_weight_sha256: String,
    pub hidden_out_bias_sha256: String,
    pub weight_sha256: String,
    pub bias_sha256: String,
}

impl InklingDsparkFold {
    pub fn to_json(&self) -> Value {
        json!({
            "recipe": self.recipe,
            "unembed": self.unembed.to_json(),
            "hidden_out_weight_sha256": self.hidden_out_weight_sha256,
            "hidden_out_bias_sha256": self.hidden_out_bias_sha256,
            "weight_sha256": self.weight_sha256,
            "bias_sha256": self.bias_sha256,
        })
    }

    pub fn from_json(v: &Value) -> Result<Self, String> {
        Ok(Self {
            recipe: str_field(v, "recipe")?,
            unembed: InklingDsparkTensorDigest::from_json(&v["unembed"])?,
            hidden_out_weight_sha256: str_field(v, "hidden_out_weight_sha256")?,
            hidden_out_bias_sha256: str_field(v, "hidden_out_bias_sha256")?,
            weight_sha256: str_field(v, "weight_sha256")?,
            bias_sha256: str_field(v, "bias_sha256")?,
        })
    }
}

/// The complete sidecar metadata JSON for a pack. `payload_sha256` covers the
/// source tensors' bytes concatenated in [`InklingDsparkConfig::manifest`]
/// order.
pub fn sidecar_metadata(
    identity: &InklingDsparkTargetIdentity,
    payload_sha256: &str,
    fold: &InklingDsparkFold,
) -> String {
    let cfg = &identity.draft_config;
    json!({
        "architecture": DSPARK_ARCHITECTURE,
        "config": cfg.to_metadata(),
        DSPARK_MANIFEST_KEY: {
            "format_version": DSPARK_FORMAT_VERSION,
            "dspark_body": DSPARK_BODY,
            "source": identity.source.to_json(),
            "target": {
                "layout": identity.layout,
                "tokenizer_sha256": identity.tokenizer_sha256,
                "loop_schedule": identity.loop_schedule,
                "embed": identity.embed.to_json(),
                "embed_norm": identity.embed_norm.to_json(),
            },
            "fold": fold.to_json(),
            "payload_sha256": payload_sha256,
            "tensor_count": cfg.sidecar_manifest().len(),
        },
    })
    .to_string()
}

/// A validated Inkling DSpark sidecar: config, provenance, and host copies of
/// every tensor (raw BF16 bytes) keyed by their sidecar names.
pub struct InklingDsparkWeights {
    pub cfg: InklingDsparkConfig,
    pub source: InklingDsparkSource,
    pub fold: InklingDsparkFold,
    /// In storage order: the 54 source tensors, then [`FOLD_WEIGHT`] and
    /// [`FOLD_BIAS`].
    pub tensors: Vec<(String, RawTensor)>,
}

impl InklingDsparkWeights {
    /// Load and validate a sidecar against the trunk it is paired with.
    pub fn load(hfq: &HfqFile, identity: &InklingDsparkTargetIdentity) -> Result<Self, String> {
        let err = |m: String| format!("inkling2 dspark sidecar {}: {m}", hfq.path().display());
        if hfq.arch_id != crate::config::ARCH_ID {
            return Err(err(format!(
                "arch id {} is not the paired trunk's {}",
                hfq.arch_id,
                crate::config::ARCH_ID
            )));
        }
        let meta: Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| err(format!("bad metadata JSON: {e}")))?;
        let cfg = InklingDsparkConfig::from_metadata(&meta["config"]).map_err(err)?;
        let man = meta
            .get(DSPARK_MANIFEST_KEY)
            .ok_or_else(|| err(format!("no {DSPARK_MANIFEST_KEY} manifest")))?;
        if man["format_version"].as_u64() != Some(DSPARK_FORMAT_VERSION)
            || man["dspark_body"].as_str() != Some(DSPARK_BODY)
        {
            return Err(err(format!(
                "manifest version/body {}/{} (expected {DSPARK_FORMAT_VERSION}/{DSPARK_BODY})",
                man["format_version"], man["dspark_body"]
            )));
        }

        // ── Provenance and target binding ──────────────────────────────────
        let source = InklingDsparkSource::from_json(&man["source"]).map_err(err)?;
        let want = &identity.source;
        if source.repo != want.repo
            || source.revision != want.revision
            || source.trunk_file != want.trunk_file
            || source.trunk_sha256 != want.trunk_sha256
        {
            return Err(err(format!(
                "wrong source: built from {}@{} ({} {}), target is {}@{} ({} {})",
                source.repo,
                source.revision,
                source.trunk_file,
                source.trunk_sha256,
                want.repo,
                want.revision,
                want.trunk_file,
                want.trunk_sha256
            )));
        }
        if source.draft_sha256 != want.draft_sha256
            || source.draft_header_sha256 != want.draft_header_sha256
        {
            return Err(err(format!(
                "wrong draft checkpoint: {} (header {}), expected {} (header {})",
                source.draft_sha256,
                source.draft_header_sha256,
                want.draft_sha256,
                want.draft_header_sha256
            )));
        }
        if cfg != identity.draft_config {
            return Err(err(format!(
                "draft config {cfg:?} differs from the trunk's mtp_config {:?}",
                identity.draft_config
            )));
        }
        let target = &man["target"];
        let tok = str_field(target, "tokenizer_sha256").map_err(err)?;
        if tok != identity.tokenizer_sha256 {
            return Err(err(format!(
                "wrong tokenizer: {tok}, trunk has {}",
                identity.tokenizer_sha256
            )));
        }
        let schedule = usize_list(target, "loop_schedule").map_err(err)?;
        if schedule != identity.loop_schedule {
            return Err(err(format!(
                "wrong loop schedule: {schedule:?}, trunk runs {:?}",
                identity.loop_schedule
            )));
        }
        // Taps index visits; the last is the final norm after visit n_slot-1.
        let n_slot = identity.loop_schedule.len();
        if cfg.target_layer_ids.last() != Some(&n_slot) {
            return Err(err(format!(
                "taps {:?} do not end at the final norm (visit count {n_slot})",
                cfg.target_layer_ids
            )));
        }
        let layout = str_field(target, "layout").map_err(err)?;
        if layout != identity.layout {
            return Err(err(format!(
                "built for trunk layout {layout:?}, trunk is {:?}",
                identity.layout
            )));
        }
        for (key, want) in [
            ("embed", &identity.embed),
            ("embed_norm", &identity.embed_norm),
        ] {
            let got = InklingDsparkTensorDigest::from_json(&target[key]).map_err(err)?;
            if &got != want {
                return Err(err(format!(
                    "bound {key} {got:?} differs from the trunk's {want:?}"
                )));
            }
        }
        let fold = InklingDsparkFold::from_json(&man["fold"]).map_err(err)?;
        if fold.recipe != FOLD_RECIPE || fold.unembed != identity.unembed {
            return Err(err(format!(
                "stale fold: made by {:?} from unembed {:?}; trunk unembed is {:?} ({FOLD_RECIPE})",
                fold.recipe, fold.unembed, identity.unembed
            )));
        }

        // ── Exact layout: names, order, dtypes, shapes, sizes, offsets ─────
        let expected = cfg.sidecar_manifest();
        if man["tensor_count"].as_u64() != Some(expected.len() as u64) {
            return Err(err(format!(
                "manifest tensor_count {} (expected {})",
                man["tensor_count"],
                expected.len()
            )));
        }
        let infos = hfq.tensor_infos();
        if infos.len() != expected.len() {
            return Err(err(format!(
                "{} tensors (expected {})",
                infos.len(),
                expected.len()
            )));
        }
        let mut cursor = hfq.data_offset() as usize;
        for (info, (name, shape)) in infos.iter().zip(&expected) {
            if &info.name != name {
                return Err(err(format!(
                    "tensor {:?} where {name:?} is expected",
                    info.name
                )));
            }
            let numel: usize = shape.iter().map(|&d| d as usize).product();
            if info.quant_type != QT_BF16
                || &info.shape != shape
                || info.group_size != 0
                || info.data_size != 2 * numel
            {
                return Err(err(format!(
                    "{name}: qt {} shape {:?} group {} bytes {} (expected BF16 {shape:?}, {} bytes)",
                    info.quant_type,
                    info.shape,
                    info.group_size,
                    info.data_size,
                    2 * numel
                )));
            }
            if info.data_offset != cursor {
                return Err(err(format!(
                    "{name}: data offset {} (expected {cursor})",
                    info.data_offset
                )));
            }
            cursor += info.data_size;
        }
        if cursor as u64 != hfq.file_len() {
            return Err(err(format!(
                "tensor data ends at {cursor}, file is {} bytes",
                hfq.file_len()
            )));
        }

        // ── Bytes ──────────────────────────────────────────────────────────
        let mut tensors = Vec::with_capacity(expected.len());
        let mut payload = Sha256::new();
        let n_source = expected.len() - 2;
        for (i, (name, _)) in expected.iter().enumerate() {
            let (info, bytes) = tensor_bytes(hfq, name).map_err(err)?;
            if i < n_source {
                payload.update(&bytes);
            }
            tensors.push((
                name.clone(),
                RawTensor {
                    info: info.clone(),
                    data: bytes.into_owned(),
                },
            ));
        }
        let payload_sha = hex(&payload.finalize());
        if man["payload_sha256"].as_str() != Some(payload_sha.as_str()) {
            return Err(err(format!(
                "source tensor bytes sha256 {payload_sha} differ from the manifest {}",
                man["payload_sha256"]
            )));
        }
        let by_name = |n: &str| -> &[u8] {
            &tensors
                .iter()
                .find(|(name, _)| name == n)
                .expect("manifest name")
                .1
                .data
        };
        for (what, name, want) in [
            (
                "stale fold: hidden_out.weight",
                "markov_head.hidden_out.weight",
                &fold.hidden_out_weight_sha256,
            ),
            (
                "stale fold: hidden_out.bias",
                "markov_head.hidden_out.bias",
                &fold.hidden_out_bias_sha256,
            ),
            ("fold weight", FOLD_WEIGHT, &fold.weight_sha256),
            ("fold bias", FOLD_BIAS, &fold.bias_sha256),
        ] {
            let got = sha256_hex(by_name(name));
            if &got != want {
                return Err(err(format!("{what} sha256 {got}, manifest {want}")));
            }
        }
        Ok(Self {
            cfg,
            source,
            fold,
            tensors,
        })
    }

    pub fn get(&self, name: &str) -> Option<&RawTensor> {
        self.tensors.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }
}

fn tensor_bytes<'a>(
    hfq: &'a HfqFile,
    name: &str,
) -> Result<(&'a hipfire_runtime::hfq::HfqTensorInfo, Cow<'a, [u8]>), String> {
    if let Some((info, bytes)) = hfq.tensor_data(name) {
        return Ok((info, Cow::Borrowed(bytes)));
    }
    hfq.tensor_data_vec(name)
        .map(|(info, v)| (info, Cow::Owned(v)))
        .ok_or_else(|| format!("cannot read tensor {name}"))
}

/// Lower-case hex SHA-256.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(d: &[u8]) -> String {
    use std::fmt::Write;
    d.iter().fold(String::with_capacity(64), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// Key-sorted compact JSON, independent of serde_json's map ordering feature.
pub fn canonical_json(v: &Value, out: &mut Vec<u8>) {
    match v {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push(b'{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                serde_json::to_writer(&mut *out, k).expect("serialize key");
                out.push(b':');
                canonical_json(&map[k], out);
            }
            out.push(b'}');
        }
        Value::Array(items) => {
            out.push(b'[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                canonical_json(item, out);
            }
            out.push(b']');
        }
        scalar => serde_json::to_writer(&mut *out, scalar).expect("serialize scalar"),
    }
}

fn str_field(v: &Value, key: &str) -> Result<String, String> {
    v.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("inkling2 dspark: missing or non-string {key}"))
}

fn usize_field(v: &Value, key: &str) -> Result<usize, String> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .ok_or_else(|| format!("inkling2 dspark: missing or non-integer {key}"))
}

fn f64_field(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("inkling2 dspark: missing or non-numeric {key}"))
}

fn usize_list(v: &Value, key: &str) -> Result<Vec<usize>, String> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("inkling2 dspark: missing or non-array {key}"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| format!("inkling2 dspark: non-integer entry in {key}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mini_config() -> Value {
        let mut v: Value = serde_json::from_str(crate::TEST_METADATA).unwrap();
        v["config"]["mtp_config"] = json!({
            "hidden_size": 2304, "num_hidden_layers": 3, "num_attention_heads": 24,
            "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 9216,
            "max_position_embeddings": 1048576, "torch_dtype": "bfloat16", "q_bias": false,
            "o_bias": false, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0,
            "local_layer_ids": [0, 2], "sliding_window_size": 1024, "block_size": 8,
            "markov_rank": 256, "markov_head_type": "rnn_hidden",
            "target_layer_ids": [5, 23, 41, 48]
        });
        v["config"].take()
    }

    #[test]
    fn mini_manifest_is_the_54_tensor_header_inventory() {
        let cfg = InklingDsparkConfig::from_source_config(&mini_config()).unwrap();
        assert_eq!((cfg.block_size, cfg.proposal_capacity), (8, 7));
        let m = cfg.manifest();
        assert_eq!(m.len(), 54);
        let get = |n: &str| m.iter().find(|(name, _)| name == n).unwrap().1.clone();
        assert_eq!(get("fuse_weight"), vec![4, 4]);
        assert_eq!(get("layers.2.mlp.down_proj.weight"), vec![2304, 9216]);
        assert_eq!(get("layers.1.self_attn.o_proj.weight"), vec![2304, 3072]);
        assert_eq!(
            get("layers.0.self_attn.k_proj_ctx.weight"),
            vec![1024, 2304]
        );
        assert_eq!(get("markov_head.markov_w1.weight"), vec![201024, 256]);
        assert_eq!(get("markov_head.rnn_proj.weight"), vec![768, 5120]);
        assert_eq!(get("markov_head.hidden_out.weight"), vec![2304, 256]);
        // Payload bytes of the published header: 635,795,744.
        let bytes: u64 = m
            .iter()
            .map(|(_, s)| 2 * s.iter().map(|&d| d as u64).product::<u64>())
            .sum();
        assert_eq!(bytes, 635_795_744);
        let names: std::collections::HashSet<_> = m.iter().map(|(n, _)| n).collect();
        assert_eq!(names.len(), 54);
        assert_eq!(cfg.sidecar_manifest().len(), 56);
    }

    #[test]
    fn config_metadata_roundtrips_and_requires_the_body() {
        let cfg = InklingDsparkConfig::from_source_config(&mini_config()).unwrap();
        let meta = cfg.to_metadata();
        assert_eq!(InklingDsparkConfig::from_metadata(&meta).unwrap(), cfg);
        let mut wrong = meta.clone();
        wrong["dspark_body"] = json!("qwen3-dspark");
        assert!(InklingDsparkConfig::from_metadata(&wrong).is_err());
        let mut none = meta;
        none.as_object_mut().unwrap().remove("dspark_body");
        assert!(InklingDsparkConfig::from_metadata(&none)
            .unwrap_err()
            .contains("not an Inkling DSpark sidecar"));
        // The legacy DSpark parser never accepts it: no synthetic noise id.
        assert!(
            hipfire_runtime::dspark_core::DsparkConfig::from_metadata_json(
                &json!({"config": cfg.to_metadata()}).to_string()
            )
            .is_none()
        );
    }

    #[test]
    fn v1_body_invariants_are_enforced() {
        let mut c = mini_config();
        c["mtp_config"]["block_size"] = json!(4);
        assert!(InklingDsparkConfig::from_source_config(&c).is_err());
        let mut c = mini_config();
        c["mtp_config"]["markov_head_type"] = json!("gru");
        assert!(InklingDsparkConfig::from_source_config(&c).is_err());
        let mut c = mini_config();
        c["mtp_config"]["q_bias"] = json!(true);
        assert!(InklingDsparkConfig::from_source_config(&c).is_err());
    }

    #[test]
    fn canonical_json_sorts_keys() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        canonical_json(&json!({"b": [1, {"y": 2, "x": "s"}], "a": null}), &mut a);
        canonical_json(
            &serde_json::from_str::<Value>(r#"{"a":null,"b":[1,{"x":"s","y":2}]}"#).unwrap(),
            &mut b,
        );
        assert_eq!(a, b);
        assert_eq!(a, br#"{"a":null,"b":[1,{"x":"s","y":2}]}"#);
    }
}
