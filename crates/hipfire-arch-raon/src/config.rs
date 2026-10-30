// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Validated Raon-1-mini configuration (one-based rda/mla lists → zero-based layer kinds). Owner: slice A.
//!
//! Ground truth: `configuration_raon1_mini.py` / `config.json` of
//! KRAFTON/Raon-1-mini-preview (PLAN §1.1, §3.1). The lists inside
//! `rda_config` are **one-based** (`(layer_idx + 1) in rda_layers`); this
//! module parses them once into [`LayerKind`] vectors indexed by zero-based
//! layer and keeps the one-based source lists only as metadata.
//!
//! The config JSON carries no `eos_token_id` (the Python class defaults it to
//! `[259202, 259200]`); `generation_config.json` is folded in from the HFQ
//! metadata envelope by [`RaonConfig::from_hfq`].

use hipfire_runtime::hfq::HfqFile;
use serde_json::Value;

/// Token mixer of one decoder layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Raon Delta Attention: vector-decay gated delta rule + three short convs.
    Rda,
    /// NoPE multi-head latent attention with output gate.
    Mla,
}

/// Feed-forward block of one decoder layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnKind {
    /// Dense SiTU-GLU MLP (`model.layers.N.mlp.*`).
    Dense,
    /// Sigmoid top-k routed experts + ungated shared expert (`model.layers.N.moe.*`).
    Moe,
}

/// Sampling defaults from `generation_config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationDefaults {
    pub do_sample: bool,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub presence_penalty: f32,
}

impl Default for GenerationDefaults {
    fn default() -> Self {
        Self {
            do_sample: true,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            presence_penalty: 1.5,
        }
    }
}

/// Released-checkpoint geometry. The production kernels are specialized to
/// these shapes, so a config that differs is refused by name, not mis-loaded.
pub const HIDDEN: usize = 2560;
pub const N_HEADS: usize = 40;
pub const HEAD_DIM: usize = 128;
pub const RDA_CONV_KERNEL: usize = 4;
pub const RDA_DECAY_RANK: usize = 128;
pub const MHC_STREAMS: usize = 4;
pub const Q_LORA: usize = 576;
pub const KV_LORA: usize = 512;
pub const QK_NOPE: usize = 128;
pub const QK_ROPE: usize = 64;
pub const V_HEAD: usize = 128;
pub const N_EXPERTS: usize = 128;
pub const TOP_K: usize = 6;
pub const MOE_INTER: usize = 1088;
pub const DENSE_INTER: usize = 12032;
/// qt44 / MQ4G256V2 input-group width: physical K is padded to this.
pub const QT44_GROUP: usize = 256;

/// Default `rda_config` lists of the released checkpoint (one-based).
pub const RELEASED_RDA_LAYERS: [usize; 18] = [
    1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23,
];
pub const RELEASED_MLA_LAYERS: [usize; 7] = [4, 8, 12, 16, 20, 24, 25];
/// Default EOS ids of the Python config class (`<|im_end|>`-class ids).
pub const DEFAULT_EOS: [u32; 2] = [259202, 259200];

/// Typed Raon-1-mini shape constants and per-layer schedule.
#[derive(Clone, Debug)]
pub struct RaonConfig {
    pub hidden: usize,
    pub n_layers: usize,
    pub vocab: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub max_position: usize,
    pub tie_word_embeddings: bool,

    // Norm epsilons (PLAN §1.3).
    /// Layer input / post-attention / RDA output / final norm epsilon.
    pub rms_eps: f32,
    /// MLA latent norm (576 and 512 channels) epsilon.
    pub mla_latent_eps: f32,
    /// mHC weightless selector RMS epsilon.
    pub mhc_selector_eps: f32,
    /// Q/K L2-normalization clamp (`x / max(||x||, eps)`).
    pub qk_l2_eps: f32,

    // RDA.
    /// `n_heads * head_dim` (5120).
    pub rda_proj: usize,
    pub rda_conv_kernel: usize,
    pub rda_decay_rank: usize,
    pub rda_full_rank_gate: bool,
    /// `Some(-5.0)` for the released checkpoint (`log_alpha = lb * sigmoid(u)`).
    pub rda_gate_lower_bound: Option<f32>,

    // MLA.
    pub q_lora: usize,
    pub kv_lora: usize,
    pub qk_nope: usize,
    pub qk_rope: usize,
    pub v_head: usize,

    // MoE / FFN.
    pub n_experts: usize,
    pub top_k: usize,
    pub n_shared_experts: usize,
    /// Logical routed/shared expert intermediate width (1088).
    pub moe_inter: usize,
    /// Physical down-projection K, `moe_inter` rounded up to a qt44 group (1280).
    pub moe_down_k: usize,
    pub dense_inter: usize,
    pub first_k_dense_replace: usize,
    pub moe_layer_freq: usize,
    pub router_renormalize: bool,
    pub routed_scale: f32,
    /// Router renormalization denominator epsilon (`sum(chosen) + 1e-20`).
    pub router_eps: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,

    // mHC.
    pub mhc_streams: usize,
    pub mhc_post_factor: f32,

    // Tokens / generation.
    pub eos_token_ids: Vec<u32>,
    pub pad_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub generation: GenerationDefaults,

    /// One-based source lists, preserved as metadata only.
    pub rda_layers_one_based: Vec<usize>,
    pub mla_layers_one_based: Vec<usize>,
    /// Zero-based per-layer mixer.
    pub layer_kinds: Vec<LayerKind>,
    /// Zero-based per-layer feed-forward block.
    pub ffn_kinds: Vec<FfnKind>,
}

// ───────────────────────── JSON helpers ─────────────────────────

fn opt_usize(obj: &Value, key: &str, default: usize) -> Result<usize, String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v.as_u64().map(|n| n as usize).ok_or_else(|| {
            format!("raon1_mini config: `{key}` must be a non-negative integer, got {v}")
        }),
    }
}

fn opt_f32(obj: &Value, key: &str, default: f32) -> Result<f32, String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_f64()
            .map(|n| n as f32)
            .ok_or_else(|| format!("raon1_mini config: `{key}` must be a number, got {v}")),
    }
}

fn opt_bool(obj: &Value, key: &str, default: bool) -> Result<bool, String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| format!("raon1_mini config: `{key}` must be a boolean, got {v}")),
    }
}

fn opt_str<'a>(obj: &'a Value, key: &str, default: &'a str) -> Result<&'a str, String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_str()
            .ok_or_else(|| format!("raon1_mini config: `{key}` must be a string, got {v}")),
    }
}

fn layer_list(rda: &Value, key: &str, released: &[usize]) -> Result<Vec<usize>, String> {
    match rda.get(key) {
        None => Err(format!("raon1_mini config: rda_config is missing `{key}`")),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_u64().map(|n| n as usize).ok_or_else(|| {
                    format!(
                        "raon1_mini config: rda_config.{key} entry {v} is not a positive integer"
                    )
                })
            })
            .collect(),
        Some(Value::Null) => Ok(released.to_vec()),
        Some(v) => Err(format!(
            "raon1_mini config: rda_config.{key} must be a list, got {v}"
        )),
    }
}

fn eos_from_value(v: &Value) -> Result<Vec<u32>, String> {
    let one = |x: &Value| {
        x.as_u64()
            .filter(|n| *n <= u32::MAX as u64)
            .map(|n| n as u32)
            .ok_or_else(|| format!("raon1_mini config: eos_token_id entry {x} is not a token id"))
    };
    match v {
        Value::Array(items) => items.iter().map(one).collect(),
        other => Ok(vec![one(other)?]),
    }
}

/// Parse and validate the one-based `rda`/`mla` lists into zero-based
/// per-layer kinds. Refuses duplicates, overlaps, out-of-range entries and
/// missing coverage.
pub fn parse_layer_kinds(
    n_layers: usize,
    rda_one_based: &[usize],
    mla_one_based: &[usize],
) -> Result<Vec<LayerKind>, String> {
    let mut kinds: Vec<Option<LayerKind>> = vec![None; n_layers];
    for (name, list, kind) in [
        ("rda_layers", rda_one_based, LayerKind::Rda),
        ("mla_layers", mla_one_based, LayerKind::Mla),
    ] {
        for &one in list {
            if one == 0 || one > n_layers {
                return Err(format!(
                    "raon1_mini config: rda_config.{name} entry {one} out of range \
                     (lists are one-based, valid 1..={n_layers})"
                ));
            }
            let slot = &mut kinds[one - 1];
            match *slot {
                None => *slot = Some(kind),
                Some(prev) if prev == kind => {
                    return Err(format!(
                        "raon1_mini config: rda_config.{name} lists layer {one} more than once"
                    ))
                }
                Some(_) => {
                    return Err(format!(
                        "raon1_mini config: layer {one} appears in both rda_layers and mla_layers"
                    ))
                }
            }
        }
    }
    let missing: Vec<usize> = kinds
        .iter()
        .enumerate()
        .filter(|(_, k)| k.is_none())
        .map(|(i, _)| i + 1)
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "raon1_mini config: layers {missing:?} (one-based) are in neither rda_layers nor mla_layers"
        ));
    }
    Ok(kinds.into_iter().map(|k| k.unwrap()).collect())
}

fn require_eq<T: PartialEq + std::fmt::Display>(what: &str, got: T, want: T) -> Result<(), String> {
    if got == want {
        Ok(())
    } else {
        Err(format!(
            "raon1_mini config: unsupported {what} {got} (the released geometry is {want})"
        ))
    }
}

impl RaonConfig {
    /// Parse from the HFQ `metadata_json` envelope: the source `config.json`
    /// under `config`, plus optional `generation_config`.
    pub fn from_hfq(hfq: &HfqFile) -> Result<RaonConfig, String> {
        Self::from_metadata_json(&hfq.metadata_json)
    }

    /// [`Self::from_hfq`] on the raw metadata string (also the unit-test seam).
    pub fn from_metadata_json(metadata_json: &str) -> Result<RaonConfig, String> {
        let wrapper: Value = serde_json::from_str(metadata_json)
            .map_err(|e| format!("raon1_mini: metadata_json not valid JSON: {e}"))?;
        let inner = wrapper
            .get("config")
            .ok_or_else(|| "raon1_mini: metadata_json missing `config` wrapper".to_string())?;
        let mut cfg = Self::from_json(inner)?;
        if let Some(gen) = wrapper.get("generation_config").filter(|g| g.is_object()) {
            cfg.apply_generation_config(gen)?;
        }
        Ok(cfg)
    }

    /// Parse the source `config.json` object (model_type `raon1_mini`).
    pub fn from_json(config: &Value) -> Result<RaonConfig, String> {
        if !config.is_object() {
            return Err("raon1_mini config: not a JSON object".into());
        }
        // A VL-style wrapper is not part of this family; text config is the object itself.
        match config.get("model_type").and_then(Value::as_str) {
            Some("raon1_mini") | None => {}
            Some(other) => {
                return Err(format!(
                    "raon1_mini config: model_type {other:?} is not raon1_mini"
                ))
            }
        }

        let hidden = opt_usize(config, "hidden_size", HIDDEN)?;
        let n_layers = opt_usize(config, "num_hidden_layers", 25)?;
        let n_heads = opt_usize(config, "num_attention_heads", N_HEADS)?;
        let head_dim = match config.get("head_dim") {
            // Python default argument is 128; an explicit null derives hidden / heads.
            None => HEAD_DIM,
            Some(Value::Null) => hidden
                .checked_div(n_heads)
                .ok_or_else(|| "raon1_mini config: num_attention_heads is zero".to_string())?,
            Some(v) => v
                .as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| format!("raon1_mini config: head_dim {v} invalid"))?,
        };
        let vocab = opt_usize(config, "vocab_size", 259_840)?;

        require_eq("hidden_size", hidden, HIDDEN)?;
        require_eq("num_attention_heads", n_heads, N_HEADS)?;
        require_eq("head_dim", head_dim, HEAD_DIM)?;
        if n_layers == 0 {
            return Err("raon1_mini config: num_hidden_layers is zero".into());
        }
        if vocab == 0 {
            return Err("raon1_mini config: vocab_size is zero".into());
        }

        let hidden_act = opt_str(config, "hidden_act", "situ")?;
        if hidden_act != "situ" {
            return Err(format!(
                "raon1_mini config: hidden_act {hidden_act:?} unsupported (only \"situ\")"
            ));
        }
        let tie = opt_bool(config, "tie_word_embeddings", false)?;
        if tie {
            return Err(
                "raon1_mini config: tie_word_embeddings=true unsupported (untied lm_head)".into(),
            );
        }

        // RDA block.
        let released_rda = RELEASED_RDA_LAYERS.to_vec();
        let released_mla = RELEASED_MLA_LAYERS.to_vec();
        let default_rda = serde_json::json!({
            "num_heads": n_heads,
            "head_dim": head_dim,
            "conv_kernel_size": RDA_CONV_KERNEL,
            "full_rank_gate": true,
            "gate_lower_bound": -5.0,
            "rda_layers": released_rda,
            "mla_layers": released_mla,
        });
        let rda_cfg = match config.get("rda_config") {
            None | Some(Value::Null) => &default_rda,
            Some(v) if v.is_object() => v,
            Some(v) => {
                return Err(format!(
                    "raon1_mini config: rda_config must be an object, got {v}"
                ))
            }
        };
        let rda_heads = opt_usize(rda_cfg, "num_heads", n_heads)?;
        let rda_head_dim = opt_usize(rda_cfg, "head_dim", head_dim)?;
        if rda_heads != n_heads || rda_head_dim != head_dim {
            return Err(format!(
                "raon1_mini config: rda_config heads/head_dim {rda_heads}x{rda_head_dim} \
                 differ from the model {n_heads}x{head_dim}"
            ));
        }
        let rda_conv_kernel = opt_usize(rda_cfg, "conv_kernel_size", RDA_CONV_KERNEL)?;
        require_eq("rda conv_kernel_size", rda_conv_kernel, RDA_CONV_KERNEL)?;
        let rda_full_rank_gate = opt_bool(rda_cfg, "full_rank_gate", true)?;
        if !rda_full_rank_gate {
            return Err("raon1_mini config: rda_config.full_rank_gate=false unsupported".into());
        }
        let rda_gate_lower_bound =
            match rda_cfg.get("gate_lower_bound") {
                None => Some(-5.0f32),
                Some(Value::Null) => None,
                Some(v) => Some(v.as_f64().ok_or_else(|| {
                    format!("raon1_mini config: gate_lower_bound {v} not a number")
                })? as f32),
            };
        match rda_gate_lower_bound {
            Some(lb) if lb == -5.0 => {}
            other => {
                return Err(format!(
                    "raon1_mini config: rda gate_lower_bound {other:?} unsupported \
                     (the released vector-decay gate uses -5.0)"
                ))
            }
        }
        let rda_decay_rank = opt_usize(config, "rda_decay_rank", RDA_DECAY_RANK)?;
        require_eq("rda_decay_rank", rda_decay_rank, RDA_DECAY_RANK)?;

        let rda_layers = layer_list(rda_cfg, "rda_layers", &RELEASED_RDA_LAYERS)?;
        let mla_layers = layer_list(rda_cfg, "mla_layers", &RELEASED_MLA_LAYERS)?;
        let layer_kinds = parse_layer_kinds(n_layers, &rda_layers, &mla_layers)?;

        // MLA.
        let q_lora = opt_usize(config, "q_lora_rank", Q_LORA)?;
        let kv_lora = opt_usize(config, "kv_lora_rank", KV_LORA)?;
        let qk_nope = opt_usize(config, "qk_nope_head_dim", QK_NOPE)?;
        let qk_rope = opt_usize(config, "qk_rope_head_dim", QK_ROPE)?;
        let v_head = opt_usize(config, "v_head_dim", V_HEAD)?;
        require_eq("q_lora_rank", q_lora, Q_LORA)?;
        require_eq("kv_lora_rank", kv_lora, KV_LORA)?;
        require_eq("qk_nope_head_dim", qk_nope, QK_NOPE)?;
        require_eq("qk_rope_head_dim", qk_rope, QK_ROPE)?;
        require_eq("v_head_dim", v_head, V_HEAD)?;
        if !opt_bool(config, "mla_use_nope", true)? {
            return Err("raon1_mini config: mla_use_nope=false (rotary MLA) unsupported".into());
        }
        if !opt_bool(config, "mla_use_output_gate", true)? {
            return Err("raon1_mini config: mla_use_output_gate=false unsupported".into());
        }

        // MoE / FFN.
        let n_experts = match config.get("num_experts") {
            Some(Value::Null) | None => N_EXPERTS,
            Some(_) => opt_usize(config, "num_experts", N_EXPERTS)?,
        };
        let top_k = opt_usize(config, "num_experts_per_token", TOP_K)?;
        let n_shared = opt_usize(config, "num_shared_experts", 1)?;
        let moe_inter = opt_usize(config, "moe_intermediate_size", MOE_INTER)?;
        let dense_inter = opt_usize(config, "intermediate_size", DENSE_INTER)?;
        require_eq("num_experts", n_experts, N_EXPERTS)?;
        require_eq("num_experts_per_token", top_k, TOP_K)?;
        require_eq("num_shared_experts", n_shared, 1)?;
        require_eq("moe_intermediate_size", moe_inter, MOE_INTER)?;
        require_eq("intermediate_size", dense_inter, DENSE_INTER)?;
        let router_act = opt_str(config, "moe_router_activation_func", "sigmoid")?;
        if router_act != "sigmoid" {
            return Err(format!(
                "raon1_mini config: moe_router_activation_func {router_act:?} unsupported (only sigmoid)"
            ));
        }
        let router_renormalize = opt_bool(config, "moe_renormalize", true)?;
        if !router_renormalize {
            return Err("raon1_mini config: moe_renormalize=false unsupported".into());
        }
        let first_k_dense = opt_usize(config, "first_k_dense_replace", 1)?;
        let moe_layer_freq = opt_usize(config, "moe_layer_freq", 1)?;
        if moe_layer_freq == 0 {
            return Err("raon1_mini config: moe_layer_freq is zero".into());
        }
        let ffn_kinds: Vec<FfnKind> = (0..n_layers)
            .map(|l| {
                if l >= first_k_dense && l % moe_layer_freq == 0 {
                    FfnKind::Moe
                } else {
                    FfnKind::Dense
                }
            })
            .collect();

        // mHC.
        let mhc_streams = opt_usize(config, "raon_mhc_streams", MHC_STREAMS)?;
        require_eq("raon_mhc_streams", mhc_streams, MHC_STREAMS)?;

        let eos_token_ids = match config.get("eos_token_id") {
            None | Some(Value::Null) => DEFAULT_EOS.to_vec(),
            Some(v) => eos_from_value(v)?,
        };

        let cfg = RaonConfig {
            hidden,
            n_layers,
            vocab,
            n_heads,
            head_dim,
            max_position: opt_usize(config, "max_position_embeddings", 262_144)?,
            tie_word_embeddings: false,
            rms_eps: opt_f32(config, "rms_norm_eps", 1e-5)?,
            mla_latent_eps: opt_f32(config, "mla_latent_norm_eps", 1e-6)?,
            mhc_selector_eps: opt_f32(config, "raon_mhc_selector_eps", 1e-5)?,
            qk_l2_eps: 1e-6,
            rda_proj: n_heads * head_dim,
            rda_conv_kernel,
            rda_decay_rank,
            rda_full_rank_gate,
            rda_gate_lower_bound,
            q_lora,
            kv_lora,
            qk_nope,
            qk_rope,
            v_head,
            n_experts,
            top_k,
            n_shared_experts: n_shared,
            moe_inter,
            moe_down_k: moe_inter.div_ceil(QT44_GROUP) * QT44_GROUP,
            dense_inter,
            first_k_dense_replace: first_k_dense,
            moe_layer_freq,
            router_renormalize,
            routed_scale: opt_f32(config, "routed_scaling_factor", 1.0)?,
            router_eps: 1e-20,
            situ_beta: opt_f32(config, "activation_situ_beta", 4.0)?,
            situ_linear_beta: opt_f32(config, "activation_situ_linear_beta", 25.0)?,
            mhc_streams,
            mhc_post_factor: opt_f32(config, "raon_mhc_post_factor", 2.0)?,
            eos_token_ids,
            pad_token_id: config
                .get("pad_token_id")
                .and_then(Value::as_u64)
                .map(|n| n as u32)
                .or(Some(259_200)),
            bos_token_id: config
                .get("bos_token_id")
                .and_then(Value::as_u64)
                .map(|n| n as u32),
            generation: GenerationDefaults::default(),
            rda_layers_one_based: rda_layers,
            mla_layers_one_based: mla_layers,
            layer_kinds,
            ffn_kinds,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Fold `generation_config.json` into the config: `eos_token_id`
    /// (int or list), `pad_token_id`, and the sampling defaults.
    pub fn apply_generation_config(&mut self, gen: &Value) -> Result<(), String> {
        if let Some(v) = gen.get("eos_token_id").filter(|v| !v.is_null()) {
            self.eos_token_ids = eos_from_value(v)?;
        }
        if let Some(n) = gen.get("pad_token_id").and_then(Value::as_u64) {
            self.pad_token_id = Some(n as u32);
        }
        if let Some(n) = gen.get("bos_token_id").and_then(Value::as_u64) {
            self.bos_token_id = Some(n as u32);
        }
        let g = &mut self.generation;
        g.do_sample = opt_bool(gen, "do_sample", g.do_sample)?;
        g.temperature = opt_f32(gen, "temperature", g.temperature)?;
        g.top_p = opt_f32(gen, "top_p", g.top_p)?;
        g.top_k = opt_usize(gen, "top_k", g.top_k as usize)? as u32;
        g.presence_penalty = opt_f32(gen, "presence_penalty", g.presence_penalty)?;
        self.validate()
    }

    /// Cross-field consistency checks, shared by every construction path.
    pub fn validate(&self) -> Result<(), String> {
        if self.layer_kinds.len() != self.n_layers || self.ffn_kinds.len() != self.n_layers {
            return Err(format!(
                "raon1_mini config: layer schedule length {}/{} != num_hidden_layers {}",
                self.layer_kinds.len(),
                self.ffn_kinds.len(),
                self.n_layers
            ));
        }
        if self.rda_proj != self.n_heads * self.head_dim {
            return Err("raon1_mini config: rda projection width != heads*head_dim".into());
        }
        if self.qk_nope + self.qk_rope != 192 {
            return Err("raon1_mini config: MLA q head dim (nope+rope) != 192".into());
        }
        if self.moe_down_k % QT44_GROUP != 0 || self.moe_down_k < self.moe_inter {
            return Err(
                "raon1_mini config: physical down K is not a qt44 group multiple >= logical".into(),
            );
        }
        if self.dense_inter % QT44_GROUP != 0 {
            return Err(format!(
                "raon1_mini config: dense intermediate {} is not a multiple of {QT44_GROUP}",
                self.dense_inter
            ));
        }
        if self.eos_token_ids.is_empty() {
            return Err("raon1_mini config: no eos token ids".into());
        }
        for &id in &self.eos_token_ids {
            if id as usize >= self.vocab {
                return Err(format!(
                    "raon1_mini config: eos id {id} outside vocab {}",
                    self.vocab
                ));
            }
        }
        Ok(())
    }

    pub fn is_rda(&self, layer: usize) -> bool {
        self.layer_kinds[layer] == LayerKind::Rda
    }
    pub fn is_mla(&self, layer: usize) -> bool {
        self.layer_kinds[layer] == LayerKind::Mla
    }
    pub fn is_moe(&self, layer: usize) -> bool {
        self.ffn_kinds[layer] == FfnKind::Moe
    }
    pub fn n_rda(&self) -> usize {
        self.layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::Rda)
            .count()
    }
    pub fn n_mla(&self) -> usize {
        self.layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::Mla)
            .count()
    }
    /// Zero-based layer indices of the RDA layers, ascending.
    pub fn rda_layers(&self) -> Vec<usize> {
        (0..self.n_layers).filter(|&l| self.is_rda(l)).collect()
    }
    /// Zero-based layer indices of the MLA layers, ascending.
    pub fn mla_layers(&self) -> Vec<usize> {
        (0..self.n_layers).filter(|&l| self.is_mla(l)).collect()
    }
    /// Dense index of `layer` among RDA layers (its recurrent-state slot).
    pub fn rda_index(&self, layer: usize) -> Option<usize> {
        self.is_rda(layer).then(|| {
            self.layer_kinds[..layer]
                .iter()
                .filter(|k| **k == LayerKind::Rda)
                .count()
        })
    }
    /// Dense index of `layer` among MLA layers (its latent-cache slot).
    pub fn mla_index(&self, layer: usize) -> Option<usize> {
        self.is_mla(layer).then(|| {
            self.layer_kinds[..layer]
                .iter()
                .filter(|k| **k == LayerKind::Mla)
                .count()
        })
    }
    /// Per-head MLA query/key width (`nope + rope` = 192).
    pub fn mla_qk_head(&self) -> usize {
        self.qk_nope + self.qk_rope
    }
    /// mHC control width: `n` pre + `n` post + `(n-1)^2` residual-chart (17).
    pub fn mhc_controls(&self) -> usize {
        2 * self.mhc_streams + (self.mhc_streams - 1) * (self.mhc_streams - 1)
    }
    /// Shared-expert intermediate width (`moe_inter * n_shared`).
    pub fn shared_inter(&self) -> usize {
        self.moe_inter * self.n_shared_experts
    }
    /// Physical (qt44-padded) K of the shared-expert down projection.
    pub fn shared_down_k(&self) -> usize {
        self.shared_inter().div_ceil(QT44_GROUP) * QT44_GROUP
    }

    /// Dimension census for [`crate::state::RaonState`] / prefill scratch.
    pub fn state_dims(&self) -> crate::state::RaonStateDims {
        crate::state::RaonStateDims {
            hidden: self.hidden,
            n_streams: self.mhc_streams,
            n_layers: self.n_layers,
            n_rda: self.n_rda(),
            n_mla: self.n_mla(),
            heads: self.n_heads,
            k_dim: self.head_dim,
            v_dim: self.head_dim,
            conv_channels: 3 * self.rda_proj,
            conv_hist: self.rda_conv_kernel - 1,
            latent: self.kv_lora,
            tail: self.qk_rope,
            vocab: self.vocab,
            dense_inter: self.dense_inter,
            moe_inter: self.moe_inter,
            moe_down_k: self.moe_down_k,
            n_experts: self.n_experts,
            top_k: self.top_k,
            shared_inter: self.shared_inter(),
            q_lora: self.q_lora,
            mla_q_out: self.n_heads * self.mla_qk_head(),
            mla_kv_b_out: self.n_heads * (self.qk_nope + self.v_head),
            mla_qk_head: self.mla_qk_head(),
            mla_v_head: self.v_head,
            mhc_controls: self.mhc_controls(),
            decay_bottleneck: self.rda_decay_rank,
            // An unparsable value is reported by `MlaPolicy::from_env` at load.
            mirror_q8: crate::mla::MlaPolicy::from_env()
                .map_or(crate::mla::MlaPolicy::DEFAULT.mirror_q8, |p| p.mirror_q8),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn released_json() -> Value {
        serde_json::from_str(
            r#"{
            "model_type": "raon1_mini",
            "hidden_size": 2560,
            "num_hidden_layers": 25,
            "num_attention_heads": 40,
            "head_dim": 128,
            "vocab_size": 259840,
            "tie_word_embeddings": false,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-5,
            "hidden_act": "situ",
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "intermediate_size": 12032,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "num_experts": 128,
            "num_experts_per_token": 6,
            "moe_intermediate_size": 1088,
            "num_shared_experts": 1,
            "moe_router_activation_func": "sigmoid",
            "moe_renormalize": true,
            "routed_scaling_factor": 1.0,
            "q_lora_rank": 576,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "mla_use_nope": true,
            "mla_use_output_gate": true,
            "raon_mhc_streams": 4,
            "rda_config": {
                "num_heads": 40, "head_dim": 128, "conv_kernel_size": 4,
                "full_rank_gate": true, "gate_lower_bound": -5.0,
                "mla_layers": [4, 8, 12, 16, 20, 24, 25],
                "rda_layers": [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23]
            }
        }"#,
        )
        .unwrap()
    }

    const RDA0: [usize; 18] = [
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22,
    ];
    const MLA0: [usize; 7] = [3, 7, 11, 15, 19, 23, 24];

    #[test]
    fn released_schedule_is_exact_25_layer_vector() {
        let cfg = RaonConfig::from_json(&released_json()).unwrap();
        assert_eq!(cfg.n_layers, 25);
        assert_eq!(cfg.rda_layers(), RDA0.to_vec());
        assert_eq!(cfg.mla_layers(), MLA0.to_vec());
        // Layer 0 is RDA with a dense MLP; layers 1..=24 are MoE.
        assert_eq!(cfg.layer_kinds[0], LayerKind::Rda);
        assert_eq!(cfg.ffn_kinds[0], FfnKind::Dense);
        assert!(cfg.ffn_kinds[1..].iter().all(|k| *k == FfnKind::Moe));
        // The final two layers are both MLA; there is no layer 25.
        assert_eq!(cfg.layer_kinds[23], LayerKind::Mla);
        assert_eq!(cfg.layer_kinds[24], LayerKind::Mla);
        assert_eq!(cfg.n_rda(), 18);
        assert_eq!(cfg.n_mla(), 7);
        assert_eq!(cfg.rda_index(0), Some(0));
        assert_eq!(cfg.rda_index(22), Some(17));
        assert_eq!(cfg.rda_index(3), None);
        assert_eq!(cfg.mla_index(3), Some(0));
        assert_eq!(cfg.mla_index(24), Some(6));
        assert_eq!(cfg.mla_index(0), None);
        // One-based source lists are preserved as metadata.
        assert_eq!(cfg.mla_layers_one_based, vec![4, 8, 12, 16, 20, 24, 25]);
    }

    #[test]
    fn released_scalars() {
        let cfg = RaonConfig::from_json(&released_json()).unwrap();
        assert_eq!(cfg.hidden, 2560);
        assert_eq!(cfg.vocab, 259_840);
        assert_eq!((cfg.n_heads, cfg.head_dim, cfg.rda_proj), (40, 128, 5120));
        assert_eq!((cfg.rda_conv_kernel, cfg.rda_decay_rank), (4, 128));
        assert_eq!(cfg.rda_gate_lower_bound, Some(-5.0));
        assert_eq!((cfg.q_lora, cfg.kv_lora), (576, 512));
        assert_eq!((cfg.qk_nope, cfg.qk_rope, cfg.v_head), (128, 64, 128));
        assert_eq!((cfg.n_experts, cfg.top_k), (128, 6));
        assert_eq!(
            (cfg.moe_inter, cfg.moe_down_k, cfg.dense_inter),
            (1088, 1280, 12032)
        );
        assert_eq!((cfg.situ_beta, cfg.situ_linear_beta), (4.0, 25.0));
        assert_eq!(cfg.mhc_streams, 4);
        assert_eq!(cfg.mhc_controls(), 17);
        assert_eq!(cfg.mla_qk_head(), 192);
        assert_eq!(cfg.shared_inter(), 1088);
        assert_eq!(cfg.shared_down_k(), 1280);
        assert_eq!(cfg.rms_eps, 1e-5);
        assert_eq!(cfg.mla_latent_eps, 1e-6);
        assert_eq!(cfg.mhc_selector_eps, 1e-5);
        assert_eq!(cfg.mhc_post_factor, 2.0);
        assert_eq!(cfg.routed_scale, 1.0);
        assert_eq!(cfg.router_eps, 1e-20);
        assert_eq!(cfg.eos_token_ids, vec![259202, 259200]);
        assert_eq!(cfg.generation, GenerationDefaults::default());
        assert_eq!(cfg.generation.top_k, 20);
    }

    #[test]
    fn missing_rda_config_uses_released_lists() {
        let mut j = released_json();
        j.as_object_mut().unwrap().remove("rda_config");
        let cfg = RaonConfig::from_json(&j).unwrap();
        assert_eq!(cfg.rda_layers(), RDA0.to_vec());
        assert_eq!(cfg.mla_layers(), MLA0.to_vec());
    }

    #[test]
    fn hfq_metadata_folds_generation_config() {
        let meta = json!({
            "architecture": "raon1_mini",
            "config": released_json(),
            "generation_config": {
                "eos_token_id": [259202, 259200], "pad_token_id": 259200,
                "do_sample": true, "temperature": 1.0, "top_p": 0.95,
                "top_k": 20, "presence_penalty": 1.5
            }
        })
        .to_string();
        let cfg = RaonConfig::from_metadata_json(&meta).unwrap();
        assert_eq!(cfg.eos_token_ids, vec![259202, 259200]);
        assert_eq!(cfg.pad_token_id, Some(259200));
        assert_eq!(cfg.generation.presence_penalty, 1.5);
        // Single-int eos is accepted.
        let meta2 =
            json!({"config": released_json(), "generation_config": {"eos_token_id": 259202}})
                .to_string();
        assert_eq!(
            RaonConfig::from_metadata_json(&meta2)
                .unwrap()
                .eos_token_ids,
            vec![259202]
        );
        // Missing wrapper and eos outside vocab are refused.
        assert!(RaonConfig::from_metadata_json("{}")
            .unwrap_err()
            .contains("`config`"));
        let bad = json!({"config": released_json(), "generation_config": {"eos_token_id": 999999}})
            .to_string();
        assert!(RaonConfig::from_metadata_json(&bad)
            .unwrap_err()
            .contains("outside vocab"));
    }

    fn with_rda(rda: Vec<usize>, mla: Vec<usize>) -> Value {
        let mut j = released_json();
        j["rda_config"]["rda_layers"] = json!(rda);
        j["rda_config"]["mla_layers"] = json!(mla);
        j
    }

    fn released_rda() -> Vec<usize> {
        RELEASED_RDA_LAYERS.to_vec()
    }
    fn released_mla() -> Vec<usize> {
        RELEASED_MLA_LAYERS.to_vec()
    }

    #[test]
    fn duplicate_overlap_missing_and_range_are_refused() {
        // duplicate inside one list
        let mut rda = released_rda();
        rda.push(1);
        let e = RaonConfig::from_json(&with_rda(rda, released_mla())).unwrap_err();
        assert!(e.contains("more than once"), "{e}");
        // overlap: layer 4 in both
        let mut mla = released_mla();
        mla.retain(|l| *l != 4);
        let mut rda = released_rda();
        rda.push(4);
        // layer 4 now only in rda -> no overlap, but put it in both:
        let mut mla2 = mla.clone();
        mla2.push(4);
        let e = RaonConfig::from_json(&with_rda(rda, mla2)).unwrap_err();
        assert!(e.contains("both"), "{e}");
        // missing coverage: drop layer 25
        let mut mla = released_mla();
        mla.retain(|l| *l != 25);
        let e = RaonConfig::from_json(&with_rda(released_rda(), mla)).unwrap_err();
        assert!(e.contains("neither") && e.contains("25"), "{e}");
        // out of range: zero (not one-based) and 26
        let mut rda = released_rda();
        rda[0] = 0;
        let e = RaonConfig::from_json(&with_rda(rda, released_mla())).unwrap_err();
        assert!(e.contains("out of range"), "{e}");
        let mut mla = released_mla();
        mla.push(26);
        let e = RaonConfig::from_json(&with_rda(released_rda(), mla)).unwrap_err();
        assert!(e.contains("out of range"), "{e}");
    }

    #[test]
    fn wrong_geometry_and_variants_are_refused() {
        let mut j = released_json();
        j["hidden_size"] = json!(2048);
        assert!(RaonConfig::from_json(&j)
            .unwrap_err()
            .contains("hidden_size"));
        let mut j = released_json();
        j["head_dim"] = json!(64);
        assert!(RaonConfig::from_json(&j).unwrap_err().contains("head_dim"));
        let mut j = released_json();
        j["num_attention_heads"] = json!(32);
        assert!(RaonConfig::from_json(&j)
            .unwrap_err()
            .contains("num_attention_heads"));
        let mut j = released_json();
        j["hidden_act"] = json!("silu");
        assert!(RaonConfig::from_json(&j)
            .unwrap_err()
            .contains("hidden_act"));
        let mut j = released_json();
        j["tie_word_embeddings"] = json!(true);
        assert!(RaonConfig::from_json(&j)
            .unwrap_err()
            .contains("tie_word_embeddings"));
        let mut j = released_json();
        j["moe_router_activation_func"] = json!("softmax");
        assert!(RaonConfig::from_json(&j).unwrap_err().contains("sigmoid"));
        let mut j = released_json();
        j["rda_config"]["gate_lower_bound"] = json!(null);
        assert!(RaonConfig::from_json(&j)
            .unwrap_err()
            .contains("gate_lower_bound"));
        let mut j = released_json();
        j["model_type"] = json!("qwen3_5");
        assert!(RaonConfig::from_json(&j)
            .unwrap_err()
            .contains("model_type"));
        let mut j = released_json();
        j["rda_config"]["rda_layers"] = json!("nope");
        assert!(RaonConfig::from_json(&j).is_err());
        let mut j = released_json();
        j["rda_config"]
            .as_object_mut()
            .unwrap()
            .remove("mla_layers");
        assert!(RaonConfig::from_json(&j)
            .unwrap_err()
            .contains("mla_layers"));
    }

    #[test]
    fn small_schedules_parse_for_fixtures() {
        // 4-layer toy: layer1 dense RDA, layer2 MLA, ...
        let mut j = released_json();
        j["num_hidden_layers"] = json!(4);
        j["rda_config"]["rda_layers"] = json!([1, 2, 4]);
        j["rda_config"]["mla_layers"] = json!([3]);
        let cfg = RaonConfig::from_json(&j).unwrap();
        assert_eq!(
            cfg.layer_kinds,
            vec![
                LayerKind::Rda,
                LayerKind::Rda,
                LayerKind::Mla,
                LayerKind::Rda
            ]
        );
        assert_eq!(
            cfg.ffn_kinds,
            vec![FfnKind::Dense, FfnKind::Moe, FfnKind::Moe, FfnKind::Moe]
        );
    }

    #[test]
    fn moe_layer_freq_follows_python_predicate() {
        let mut j = released_json();
        j["moe_layer_freq"] = json!(2);
        let cfg = RaonConfig::from_json(&j).unwrap();
        for l in 0..25 {
            let want = l >= 1 && l % 2 == 0;
            assert_eq!(cfg.is_moe(l), want, "layer {l}");
        }
    }
}
