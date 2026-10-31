// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon-1-mini quantization recipe: MXFP8 E8M0 dequant, class map, 1088→1280 padding. Owner: slice Q.
//!
//! Producer for `raon-1-mini-preview.sym-mq4v2*.hfq` (arch_id 19, `raon1_mini`):
//! the symmetric qt44 (MQ4G256V2) RTN control artifact for the experts/dense MLP
//! with every attention projection (and `lm_head`) at Q8F16; the same layout the
//! XTS calibration/refit lane replaces in place for the qt44 classes.
//!
//! Class map (PLAN §3.2, original HF tensor names preserved):
//!
//! | class | storage |
//! |---|---|
//! | embedding; `lm_head` (default, `HIPFIRE_RAON_LM_HEAD=qt44` overrides); every attention projection: MLA `q_a/q_b/kv_a/kv_b/g/o`, RDA `q/k/v/o/g/f_a/f_b/b_proj/rda_w_proj` | Q8F16 (qt3), no AWQ sidecar |
//! | routed/shared/dense gate,up,down | symmetric qt44 |
//! | routed + shared `down_proj` (natural K1088) | qt44 `[N,1280]`, tail zero-padded **before** FWHT |
//! | RMS gammas, conv, mHC alpha/b/phi | lossless: F16 when every value round-trips, else F32 |
//! | router, `A_log`, `dt_bias`, correction bias, mHC `eta` | F32 |
//!
//! Every qt44 tensor gets a unit F16 (qt1) `<stem>.awq_scale.weight` sidecar of
//! the full physical input width. The routed-expert source is MXFP8: E4M3
//! `[N,K]` plus raw E8M0 exponent bytes `[N,K/32]` in `.weight_scale_inv`;
//! `value = float32(E4M3) * 2^(u8-127)` (modeling `M:496-511`).
//!
//! `HIPFIRE_RAON_SHARED_FFN` (`qt44` default | `q8f16` | `bf16`) selects the storage of the
//! always-active FFN weights (shared expert gate/up/down of every MoE layer and the dense
//! layer-0 gate/up/down). `q8f16` (qt3, group 32) and `bf16` (qt16, source-exact) carry no AWQ
//! sidecar; the shared `down_proj` keeps the 1280 physical K (zero-padded columns) for both.
//! `--raon-shared-ffn <q8f16|bf16>` builds such a variant from a shipped qt44 artifact
//! without re-running calibration (see [`swap_shared_ffn`] / [`raon_hfq_diff`]).

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use hipfire_quantize::float16::{f16_to_f32, f32_to_f16};
use hipfire_quantize::safetensors_file::{SafetensorsFile, TensorMeta};
use hipfire_runtime::hfq::HfqFile as RuntimeHfqFile;
use rayon::prelude::*;

use crate::dequant::{dequantize_mxfp8_e8m0_strict, to_f32};
use crate::hfq::{write_hfq, write_hfq_streamed, HfqIndexEntry, HfqTensor, QuantType, TensorSpill};
use crate::quant_fwht::{gen_fwht_signs, quantize_mq4g256v2_symmetric, MQ4V2_GROUP_BYTES};
use crate::quant_q4::quantize_q8f16;

pub(crate) const RAON_ARCH_ID: u32 = 19;
pub(crate) const RAON_MODEL_TYPE: &str = "raon1_mini";
/// Pinned official checkpoint revision this recipe was written against.
pub(crate) const SOURCE_REVISION: &str = "8c9b802acacdbda9df16606689fcf437cf7262fb";
pub(crate) const RECIPE: &str = "raon-mq4v2-symmetric-qt44-experts-q8f16-attn/v2";
pub(crate) const MXFP8_BLOCK_K: usize = 32;
/// Rotation group width of MQ4G256V2; physical K of padded tensors rounds up to it.
const GROUP: usize = 256;
/// Rows per rayon work unit inside one tensor.
const ROWS_PER_UNIT: usize = 32;
/// Bytes per Q8F16 block of 32 weights: F16 scale + 32 × i8.
const Q8F16_BLOCK_BYTES: usize = 34;
/// Streaming window for pass-through copy and payload comparison.
const COPY_CHUNK: usize = 4 * 1024 * 1024;

// ─── class map ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Storage {
    /// Q8F16 (qt3), 32-element blocks.
    Q8F16,
    /// Q8F16 with natural K zero-padded to a multiple of 256 (1088→1280).
    Q8F16Padded,
    /// Symmetric qt44, natural K already divisible by 256.
    Qt44,
    /// Symmetric qt44 with natural K zero-padded to a multiple of 256 (1088→1280).
    Qt44Padded,
    /// BF16 (qt16): raw little-endian bfloat16 of the source values.
    Bf16,
    /// BF16 with natural K zero-padded to a multiple of 256 (1088→1280).
    Bf16Padded,
    /// F16 when exactly representable, else F32.
    Lossless,
    /// Always F32.
    F32,
}

impl Storage {
    pub(crate) fn is_qt44(self) -> bool {
        matches!(self, Storage::Qt44 | Storage::Qt44Padded)
    }

    pub(crate) fn is_padded(self) -> bool {
        matches!(
            self,
            Storage::Qt44Padded | Storage::Q8F16Padded | Storage::Bf16Padded
        )
    }
}

/// Storage of the always-active FFN weights (shared expert gate/up/down and the
/// dense layer-0 gate/up/down).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SharedFfn {
    Qt44,
    Q8F16,
    Bf16,
}

impl SharedFfn {
    pub(crate) fn parse(s: &str) -> Result<Self, String> {
        match s {
            "qt44" => Ok(SharedFfn::Qt44),
            "q8f16" => Ok(SharedFfn::Q8F16),
            "bf16" => Ok(SharedFfn::Bf16),
            other => Err(format!(
                "unknown shared-FFN storage `{other}` (expected qt44|q8f16|bf16)"
            )),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            SharedFfn::Qt44 => "qt44",
            SharedFfn::Q8F16 => "q8f16",
            SharedFfn::Bf16 => "bf16",
        }
    }

    /// Class of one always-active FFN weight; `padded` for the shared `down_proj`.
    fn storage(self, padded: bool) -> Storage {
        match (self, padded) {
            (SharedFfn::Qt44, false) => Storage::Qt44,
            (SharedFfn::Qt44, true) => Storage::Qt44Padded,
            (SharedFfn::Q8F16, false) => Storage::Q8F16,
            (SharedFfn::Q8F16, true) => Storage::Q8F16Padded,
            (SharedFfn::Bf16, false) => Storage::Bf16,
            (SharedFfn::Bf16, true) => Storage::Bf16Padded,
        }
    }
}

/// `model.layers.<n>.<rest>` → `(n, rest)`.
fn layer_rest(name: &str) -> Option<(usize, &str)> {
    let tail = name.strip_prefix("model.layers.")?;
    let (digits, rest) = tail.split_once('.')?;
    Some((digits.parse().ok()?, rest))
}

/// `moe.experts.<e>.<proj>.weight` → `(e, proj)`.
fn routed_expert(rest: &str) -> Option<(usize, &str)> {
    let tail = rest.strip_prefix("moe.experts.")?;
    let (digits, proj) = tail.split_once('.')?;
    let e: usize = digits.parse().ok()?;
    matches!(
        proj,
        "gate_proj.weight" | "up_proj.weight" | "down_proj.weight"
    )
    .then_some((e, proj))
}

/// True for `model.layers.N.moe.experts.E.{gate,up,down}_proj.weight`.
pub(crate) fn is_routed_expert_weight(name: &str) -> bool {
    layer_rest(name).is_some_and(|(_, rest)| routed_expert(rest).is_some())
}

/// `lm_head.weight` storage: Q8F16 by default; `HIPFIRE_RAON_LM_HEAD=qt44`
/// keeps the symmetric qt44 head (smaller artifact).
pub(crate) fn lm_head_storage() -> Storage {
    match hipfire_config::developer_var("HIPFIRE_RAON_LM_HEAD").as_deref() {
        Ok("qt44") => Storage::Qt44,
        _ => Storage::Q8F16,
    }
}

/// Always-active FFN storage: `HIPFIRE_RAON_SHARED_FFN=qt44` (default, unset) | `q8f16` |
/// `bf16`. An unknown value is an error, never a silent default.
pub(crate) fn shared_ffn_storage() -> Result<SharedFfn, String> {
    match hipfire_config::developer_var("HIPFIRE_RAON_SHARED_FFN") {
        Ok(v) => SharedFfn::parse(&v).map_err(|e| format!("HIPFIRE_RAON_SHARED_FFN: {e}")),
        Err(_) => Ok(SharedFfn::Qt44),
    }
}

/// Which always-active FFN tensors a swap covers: both classes, only the shared experts
/// of the MoE layers, or only the dense layer-0 MLP.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scope {
    All,
    Shared,
    /// Only the shared experts' gate/up weights; shared down and the dense MLP stay qt44.
    SharedGateUp,
    Dense,
}

impl Scope {
    pub(crate) fn parse(s: &str) -> Result<Self, String> {
        match s {
            "all" => Ok(Scope::All),
            "shared" => Ok(Scope::Shared),
            "shared-gateup" => Ok(Scope::SharedGateUp),
            "dense" => Ok(Scope::Dense),
            other => Err(format!(
                "unknown shared-FFN scope `{other}` (expected all|shared|shared-gateup|dense)"
            )),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Scope::All => "all",
            Scope::Shared => "shared",
            Scope::SharedGateUp => "shared-gateup",
            Scope::Dense => "dense",
        }
    }

    /// Is `name` an always-active FFN weight inside this scope?
    pub(crate) fn covers(self, name: &str) -> bool {
        match ffn_class_index(name) {
            Some(CLASS_DENSE) => matches!(self, Scope::All | Scope::Dense),
            Some(CLASS_GATE_UP) => self != Scope::Dense,
            Some(CLASS_DOWN) => matches!(self, Scope::All | Scope::Shared),
            _ => false,
        }
    }
}

/// `hipfire_raon.shared_ffn` value: `qt44`, `q8f16`, `bf16` (scope all) or
/// `<storage>:<scope>` for a mixed file.
fn storage_tag(shared: SharedFfn, scope: Scope) -> String {
    if shared == SharedFfn::Qt44 || scope == Scope::All {
        shared.as_str().to_string()
    } else {
        format!("{}:{}", shared.as_str(), scope.as_str())
    }
}

/// Recipe string of an artifact whose always-active FFN carries `tag` ([`storage_tag`]).
fn recipe_with_shared(base: &str, tag: &str) -> String {
    if tag == "qt44" {
        base.to_string()
    } else {
        format!("{base}+shared-ffn={tag}")
    }
}

/// [`classify_with`] for an artifact whose swap covers only `scope`.
fn classify_scoped(
    name: &str,
    lm_head: Storage,
    shared: SharedFfn,
    scope: Scope,
) -> Result<Storage, String> {
    let eff = if scope.covers(name) {
        shared
    } else {
        SharedFfn::Qt44
    };
    classify_with(name, lm_head, eff)
}

/// Storage class for a source tensor name; unknown names are an error, never
/// a silent default.
pub(crate) fn classify(name: &str) -> Result<Storage, String> {
    classify_with(name, lm_head_storage(), shared_ffn_storage()?)
}

/// [`classify`] with the two developer choices passed in (pure).
pub(crate) fn classify_with(
    name: &str,
    lm_head: Storage,
    shared: SharedFfn,
) -> Result<Storage, String> {
    match name {
        "model.embed_tokens.weight" => return Ok(Storage::Q8F16),
        "lm_head.weight" => return Ok(lm_head),
        "model.norm.weight" => return Ok(Storage::Lossless),
        _ => {}
    }
    let (_, rest) = layer_rest(name).ok_or_else(|| format!("unrecognised Raon tensor `{name}`"))?;
    if let Some((_, proj)) = routed_expert(rest) {
        return Ok(if proj == "down_proj.weight" {
            Storage::Qt44Padded
        } else {
            Storage::Qt44
        });
    }
    if let Some(site) = rest
        .strip_prefix("self_attn_raon_mhc.")
        .or_else(|| rest.strip_prefix("mlp_raon_mhc."))
    {
        return match site {
            "alpha_post" | "alpha_pre" | "b_post" | "b_pre" | "phi_post" | "phi_pre"
            | "res.alpha_z" | "res.b_z" | "res.phi_z" => Ok(Storage::Lossless),
            "res.eta" => Ok(Storage::F32),
            _ => Err(format!("unrecognised Raon mHC tensor `{name}`")),
        };
    }
    let class = match rest {
        "input_layernorm.weight" | "post_attention_layernorm.weight" => Storage::Lossless,
        "mlp.gate_proj.weight" | "mlp.up_proj.weight" | "mlp.down_proj.weight" => {
            shared.storage(false)
        }
        "moe.gate.weight" | "moe.gate.e_score_correction_bias" => Storage::F32,
        "moe.shared_experts.gate_proj.weight" | "moe.shared_experts.up_proj.weight" => {
            shared.storage(false)
        }
        "moe.shared_experts.down_proj.weight" => shared.storage(true),
        "self_attn.q_proj.weight"
        | "self_attn.k_proj.weight"
        | "self_attn.v_proj.weight"
        | "self_attn.g_proj.weight"
        | "self_attn.o_proj.weight"
        | "self_attn.q_a_proj.weight"
        | "self_attn.kv_a_proj_with_mqa.weight"
        | "self_attn.kv_b_proj.weight"
        | "self_attn.q_b_proj.weight"
        | "self_attn.f_a_proj.weight"
        | "self_attn.f_b_proj.weight"
        | "self_attn.b_proj.weight"
        | "self_attn.rda_w_proj.weight" => Storage::Q8F16,
        "self_attn.A_log" | "self_attn.dt_bias" => Storage::F32,
        "self_attn.q_conv1d.weight"
        | "self_attn.k_conv1d.weight"
        | "self_attn.v_conv1d.weight"
        | "self_attn.o_norm.weight"
        | "self_attn.q_a_layernorm.weight"
        | "self_attn.kv_a_layernorm.weight" => Storage::Lossless,
        _ => return Err(format!("unrecognised Raon tensor `{name}`")),
    };
    Ok(class)
}

/// True for the always-active FFN weights: shared expert gate/up/down of a MoE layer and
/// the dense layer gate/up/down.
pub(crate) fn is_shared_ffn_weight(name: &str) -> bool {
    layer_rest(name).is_some_and(|(_, rest)| {
        matches!(
            rest,
            "moe.shared_experts.gate_proj.weight"
                | "moe.shared_experts.up_proj.weight"
                | "moe.shared_experts.down_proj.weight"
                | "mlp.gate_proj.weight"
                | "mlp.up_proj.weight"
                | "mlp.down_proj.weight"
        )
    })
}

/// `<stem>.awq_scale.weight` → `<stem>.weight` when that is an always-active FFN weight.
fn shared_ffn_sidecar_parent(name: &str) -> Option<String> {
    let weight = format!("{}.weight", name.strip_suffix(".awq_scale.weight")?);
    is_shared_ffn_weight(&weight).then_some(weight)
}

pub(crate) fn awq_sidecar_name(weight_name: &str) -> String {
    match weight_name.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.awq_scale.weight"),
        None => format!("{weight_name}.awq_scale.weight"),
    }
}

/// Physical input width of a tensor with natural width `k`.
pub(crate) fn physical_k(storage: Storage, k: usize) -> usize {
    if storage.is_padded() {
        k.div_ceil(GROUP) * GROUP
    } else {
        k
    }
}

/// Is this a Raon source checkpoint (config.json `model_type == raon1_mini`)?
pub(crate) fn is_raon_source_dir(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| {
            v.get("model_type")
                .and_then(|m| m.as_str())
                .map(str::to_owned)
        })
        .as_deref()
        == Some(RAON_MODEL_TYPE)
}

// ─── expected tensor set ────────────────────────────────────────────────────

fn cfg_usize(cfg: &serde_json::Value, key: &str) -> Result<usize, String> {
    cfg.get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| format!("config.json: missing or non-integer `{key}`"))
}

/// Every non-scale tensor name the checkpoint must contain. The attention
/// kind of a layer (MLA vs RDA) is read from which tensors it carries, then
/// the full set for that kind is required.
fn expected_names(
    cfg: &serde_json::Value,
    present: &BTreeSet<&str>,
) -> Result<BTreeSet<String>, String> {
    let layers = cfg_usize(cfg, "num_hidden_layers")?;
    let experts = cfg_usize(cfg, "num_experts")?;
    let first_dense = cfg_usize(cfg, "first_k_dense_replace")?;
    if cfg_usize(cfg, "num_shared_experts")? != 1 {
        return Err("config.json: only num_shared_experts == 1 is supported".to_string());
    }
    let mut want: BTreeSet<String> = [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for l in 0..layers {
        let p = format!("model.layers.{l}.");
        let mut add = |rest: &str| {
            want.insert(format!("{p}{rest}"));
        };
        add("input_layernorm.weight");
        add("post_attention_layernorm.weight");
        for site in ["self_attn_raon_mhc", "mlp_raon_mhc"] {
            for t in [
                "alpha_post",
                "alpha_pre",
                "b_post",
                "b_pre",
                "phi_post",
                "phi_pre",
                "res.alpha_z",
                "res.b_z",
                "res.eta",
                "res.phi_z",
            ] {
                add(&format!("{site}.{t}"));
            }
        }
        let mla = present.contains(format!("{p}self_attn.q_a_proj.weight").as_str());
        let attn: &[&str] = if mla {
            &[
                "q_a_proj.weight",
                "q_a_layernorm.weight",
                "q_b_proj.weight",
                "kv_a_proj_with_mqa.weight",
                "kv_a_layernorm.weight",
                "kv_b_proj.weight",
                "g_proj.weight",
                "o_proj.weight",
            ]
        } else {
            &[
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "g_proj.weight",
                "o_proj.weight",
                "f_a_proj.weight",
                "f_b_proj.weight",
                "b_proj.weight",
                "rda_w_proj.weight",
                "A_log",
                "dt_bias",
                "q_conv1d.weight",
                "k_conv1d.weight",
                "v_conv1d.weight",
                "o_norm.weight",
            ]
        };
        for t in attn {
            add(&format!("self_attn.{t}"));
        }
        if l < first_dense {
            for t in ["gate_proj", "up_proj", "down_proj"] {
                add(&format!("mlp.{t}.weight"));
            }
        } else {
            add("moe.gate.weight");
            add("moe.gate.e_score_correction_bias");
            for t in ["gate_proj", "up_proj", "down_proj"] {
                add(&format!("moe.shared_experts.{t}.weight"));
                for e in 0..experts {
                    add(&format!("moe.experts.{e}.{t}.weight"));
                }
            }
        }
    }
    Ok(want)
}

// ─── source index ───────────────────────────────────────────────────────────

pub(crate) struct SourceIndex {
    files: Vec<SafetensorsFile>,
    loc: BTreeMap<String, usize>,
}

impl SourceIndex {
    /// Open every `*.safetensors` shard of `dir`; a tensor name appearing in
    /// two shards is an error.
    pub(crate) fn open(dir: &Path) -> Result<Self, String> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| format!("read {}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(format!("{} contains no .safetensors shards", dir.display()));
        }
        let mut files = Vec::new();
        let mut loc = BTreeMap::new();
        for (i, p) in paths.iter().enumerate() {
            let f = SafetensorsFile::open(p).map_err(|e| format!("open {}: {e}", p.display()))?;
            for name in f.tensor_names() {
                if let Some(prev) = loc.insert(name.to_string(), i) {
                    return Err(format!(
                        "tensor `{name}` appears in both {} and {}",
                        paths[prev].display(),
                        p.display()
                    ));
                }
            }
            files.push(f);
        }
        Ok(Self { files, loc })
    }

    fn get(&self, name: &str) -> Option<(&TensorMeta, &[u8])> {
        self.files[*self.loc.get(name)?].tensor_data(name)
    }

    /// Validate the whole checkpoint against the config and the MXFP8
    /// contract; returns the sorted weight jobs (scale tensors excluded).
    pub(crate) fn validate(
        &self,
        cfg: &serde_json::Value,
    ) -> Result<Vec<(String, Storage)>, String> {
        if cfg.get("model_type").and_then(|v| v.as_str()) != Some(RAON_MODEL_TYPE) {
            return Err(format!("config.json model_type is not `{RAON_MODEL_TYPE}`"));
        }
        let (scales, weights): (Vec<&str>, Vec<&str>) = self
            .loc
            .keys()
            .map(String::as_str)
            .partition(|n| n.ends_with(".weight_scale_inv"));
        let present: BTreeSet<&str> = weights.iter().copied().collect();

        let want = expected_names(cfg, &present)?;
        let missing: Vec<&String> = want
            .iter()
            .filter(|n| !present.contains(n.as_str()))
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "{} required tensor(s) missing, first: {:?}",
                missing.len(),
                &missing[..missing.len().min(6)]
            ));
        }
        let extra: Vec<&&str> = weights.iter().filter(|n| !want.contains(**n)).collect();
        if !extra.is_empty() {
            return Err(format!(
                "{} unexpected tensor(s), first: {:?}",
                extra.len(),
                &extra[..extra.len().min(6)]
            ));
        }

        // MXFP8 contract keyed on config + dtype + shape, never the model name.
        let routed: Vec<&str> = weights
            .iter()
            .copied()
            .filter(|n| is_routed_expert_weight(n))
            .collect();
        if !routed.is_empty() {
            let q = cfg
                .get("quantization_config")
                .ok_or("config.json: routed experts are FP8 but quantization_config is absent")?;
            if q.get("quant_method").and_then(|v| v.as_str()) != Some("mxfp8") {
                return Err("quantization_config.quant_method must be `mxfp8`".to_string());
            }
            let block: Vec<u64> = q
                .get("weight_block_size")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
                .unwrap_or_default();
            if block != [1, MXFP8_BLOCK_K as u64] {
                return Err(format!(
                    "quantization_config.weight_block_size must be [1,{MXFP8_BLOCK_K}], got {block:?}"
                ));
            }
            if q.get("dequantize").and_then(|v| v.as_bool()) == Some(true) {
                return Err(
                    "quantization_config.dequantize=true is not an MXFP8 source".to_string()
                );
            }
        }
        let scale_set: BTreeSet<&str> = scales.iter().copied().collect();
        for s in &scales {
            let stem = s
                .strip_suffix(".weight_scale_inv")
                .expect("partitioned on suffix");
            let w = format!("{stem}.weight");
            if !is_routed_expert_weight(&w) || !present.contains(w.as_str()) {
                return Err(format!(
                    "orphan scale tensor `{s}` (no routed-expert weight `{w}`)"
                ));
            }
        }
        for name in &weights {
            let (meta, _) = self.get(name).expect("indexed");
            if is_routed_expert_weight(name) {
                let sname = format!("{}_scale_inv", name);
                if !scale_set.contains(sname.as_str()) {
                    return Err(format!("`{name}` has no `.weight_scale_inv` scale tensor"));
                }
                if meta.dtype != "F8_E4M3"
                    || meta.shape.len() != 2
                    || meta.shape[1] % MXFP8_BLOCK_K != 0
                {
                    return Err(format!(
                        "`{name}`: routed expert must be F8_E4M3 [N,K%{MXFP8_BLOCK_K}==0], got {} {:?}",
                        meta.dtype, meta.shape
                    ));
                }
                let (sm, _) = self.get(&sname).expect("indexed");
                if !matches!(sm.dtype.as_str(), "U8" | "F8_E8M0")
                    || sm.shape != [meta.shape[0], meta.shape[1] / MXFP8_BLOCK_K]
                {
                    return Err(format!(
                        "`{sname}`: must be raw E8M0 bytes (U8/F8_E8M0) [{},{}], got {} {:?}",
                        meta.shape[0],
                        meta.shape[1] / MXFP8_BLOCK_K,
                        sm.dtype,
                        sm.shape
                    ));
                }
            } else if !matches!(meta.dtype.as_str(), "BF16" | "F16" | "F32") {
                return Err(format!(
                    "`{name}`: unsupported source dtype {} (expected BF16/F16/F32)",
                    meta.dtype
                ));
            }
        }
        weights
            .iter()
            .map(|n| classify(n).map(|c| (n.to_string(), c)))
            .collect()
    }

    /// Decode one source tensor to f32 (MXFP8 experts via the strict E8M0 path).
    fn load_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let (meta, bytes) = self.get(name).ok_or_else(|| format!("`{name}` vanished"))?;
        let data = if is_routed_expert_weight(name) {
            let sname = format!("{name}_scale_inv");
            let (sm, sbytes) = self
                .get(&sname)
                .ok_or_else(|| format!("`{sname}` missing"))?;
            dequantize_mxfp8_e8m0_strict(bytes, &meta.shape, sbytes, &sm.shape, MXFP8_BLOCK_K)
                .map_err(|e| format!("{name}: {e}"))?
        } else {
            to_f32(bytes, &meta.dtype)
        };
        if let Some(i) = data.par_iter().position_first(|v| !v.is_finite()) {
            return Err(format!("{name}: non-finite source value at flat index {i}"));
        }
        let shape = if meta.shape.is_empty() {
            vec![1]
        } else {
            meta.shape.clone()
        };
        if data.len() != shape.iter().product::<usize>() {
            return Err(format!(
                "{name}: element count does not match shape {shape:?}"
            ));
        }
        Ok((data, shape))
    }
}

// ─── conversion ─────────────────────────────────────────────────────────────

#[derive(Default)]
struct JobOut {
    tensors: Vec<HfqTensor>,
    /// `(weight name, logical K, physical K)` for padded tensors.
    padded: Option<(String, usize, usize)>,
}

/// bfloat16 bits of `v`, round-to-nearest-even; exact for values that are already bf16.
/// `v` must be finite (the source loader rejects non-finite values).
fn f32_to_bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// `[rows, k]` → `[rows, phys]` with zero tail columns.
fn pad_rows(data: &[f32], rows: usize, k: usize, phys: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; rows * phys];
    v.par_chunks_mut(phys)
        .zip(data.par_chunks(k))
        .for_each(|(dst, src)| dst[..k].copy_from_slice(src));
    v
}

fn f16_bytes_unit(n: usize) -> Vec<u8> {
    let one = f32_to_f16(1.0).to_le_bytes();
    let mut v = Vec::with_capacity(2 * n);
    for _ in 0..n {
        v.extend_from_slice(&one);
    }
    v
}

/// Every half header of a packed qt44 buffer must be symmetric:
/// `z_bits == f16(-8 * d)` bit-for-bit (signed zero included) and `d` finite.
pub(crate) fn verify_symmetric_headers(name: &str, data: &[u8]) -> Result<usize, String> {
    if data.len() % MQ4V2_GROUP_BYTES != 0 {
        return Err(format!(
            "{name}: qt44 byte count {} not a multiple of {MQ4V2_GROUP_BYTES}",
            data.len()
        ));
    }
    let mut halves = 0usize;
    for (g, group) in data.chunks_exact(MQ4V2_GROUP_BYTES).enumerate() {
        for half in 0..2 {
            let d = u16::from_le_bytes([group[4 * half], group[4 * half + 1]]);
            let z = u16::from_le_bytes([group[4 * half + 2], group[4 * half + 3]]);
            let df = f16_to_f32(d);
            if !df.is_finite() {
                return Err(format!(
                    "{name}: group {g} half {half} has non-finite scale 0x{d:04x}"
                ));
            }
            let want = f32_to_f16(-8.0 * df);
            if z != want {
                return Err(format!(
                    "{name}: group {g} half {half} header not symmetric: d=0x{d:04x} z=0x{z:04x} expected 0x{want:04x}"
                ));
            }
            halves += 1;
        }
    }
    Ok(halves)
}

fn quantize_qt44(
    name: &str,
    data: &[f32],
    rows: usize,
    k: usize,
    phys: usize,
) -> Result<Vec<u8>, String> {
    if phys % GROUP != 0 {
        return Err(format!(
            "{name}: physical K {phys} is not a multiple of {GROUP}"
        ));
    }
    let s1 = gen_fwht_signs(42, GROUP);
    let s2 = gen_fwht_signs(1042, GROUP);
    // Natural-K zero padding happens BEFORE the rotation, row by row.
    let units: Vec<Vec<u8>> = (0..rows.div_ceil(ROWS_PER_UNIT))
        .into_par_iter()
        .map(|u| {
            let r0 = u * ROWS_PER_UNIT;
            let r1 = (r0 + ROWS_PER_UNIT).min(rows);
            if phys == k {
                quantize_mq4g256v2_symmetric(&data[r0 * k..r1 * k], r1 - r0, phys, &s1, &s2)
            } else {
                let mut padded = vec![0.0f32; (r1 - r0) * phys];
                for (i, r) in (r0..r1).enumerate() {
                    padded[i * phys..i * phys + k].copy_from_slice(&data[r * k..(r + 1) * k]);
                }
                quantize_mq4g256v2_symmetric(&padded, r1 - r0, phys, &s1, &s2)
            }
        })
        .collect();
    let out: Vec<u8> = units.concat();
    verify_symmetric_headers(name, &out)?;
    Ok(out)
}

fn convert(src: &SourceIndex, name: &str, storage: Storage) -> Result<JobOut, String> {
    let (data, shape) = src.load_f32(name)?;
    let shape_u32: Vec<u32> = shape.iter().map(|&d| d as u32).collect();
    let mut out = JobOut::default();
    match storage {
        Storage::Qt44 | Storage::Qt44Padded => {
            if shape.len() != 2 {
                return Err(format!("{name}: qt44 needs a 2-D weight, got {shape:?}"));
            }
            let (rows, k) = (shape[0], shape[1]);
            let phys = physical_k(storage, k);
            let packed = quantize_qt44(name, &data, rows, k, phys)?;
            out.tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::MQ4G256V2,
                shape: vec![rows as u32, phys as u32],
                group_size: GROUP as u32,
                data: packed,
                spilled_len: 0,
            });
            out.tensors.push(HfqTensor {
                name: awq_sidecar_name(name),
                quant_type: QuantType::F16,
                shape: vec![phys as u32],
                group_size: 0,
                data: f16_bytes_unit(phys),
                spilled_len: 0,
            });
            if phys != k {
                out.padded = Some((name.to_string(), k, phys));
            }
        }
        Storage::Q8F16 | Storage::Q8F16Padded => {
            if shape.len() != 2 {
                return Err(format!("{name}: Q8F16 needs a 2-D weight, got {shape:?}"));
            }
            let (rows, k) = (shape[0], shape[1]);
            let phys = physical_k(storage, k);
            if phys == 0 || phys % 32 != 0 {
                return Err(format!("{name}: Q8F16 needs [N, K%32==0], got {shape:?}"));
            }
            let padded;
            let flat: &[f32] = if phys == k {
                &data
            } else {
                padded = pad_rows(&data, rows, k, phys);
                &padded
            };
            let bytes: Vec<u8> = flat
                .par_chunks(ROWS_PER_UNIT * phys)
                .map(quantize_q8f16)
                .collect::<Vec<_>>()
                .concat();
            out.tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::Q8F16,
                shape: vec![rows as u32, phys as u32],
                group_size: 32,
                data: bytes,
                spilled_len: 0,
            });
            if phys != k {
                out.padded = Some((name.to_string(), k, phys));
            }
        }
        Storage::Bf16 | Storage::Bf16Padded => {
            if shape.len() != 2 || shape[1] == 0 {
                return Err(format!("{name}: BF16 needs a 2-D weight, got {shape:?}"));
            }
            let (rows, k) = (shape[0], shape[1]);
            let phys = physical_k(storage, k);
            let mut bytes = vec![0u8; rows * phys * 2];
            bytes
                .par_chunks_mut(phys * 2)
                .zip(data.par_chunks(k))
                .for_each(|(dst, src)| {
                    for (d, &v) in dst.chunks_exact_mut(2).zip(src) {
                        d.copy_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
                    }
                });
            out.tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::BF16,
                shape: vec![rows as u32, phys as u32],
                group_size: 0,
                data: bytes,
                spilled_len: 0,
            });
            if phys != k {
                out.padded = Some((name.to_string(), k, phys));
            }
        }
        Storage::Lossless => {
            let exact = data
                .par_iter()
                .all(|&v| f16_to_f32(f32_to_f16(v)).to_bits() == v.to_bits());
            let (qt, bytes): (QuantType, Vec<u8>) = if exact {
                (
                    QuantType::F16,
                    data.iter()
                        .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                        .collect(),
                )
            } else {
                (
                    QuantType::F32,
                    data.iter().flat_map(|v| v.to_le_bytes()).collect(),
                )
            };
            out.tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: qt,
                shape: shape_u32,
                group_size: 0,
                data: bytes,
                spilled_len: 0,
            });
        }
        Storage::F32 => {
            out.tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::F32,
                shape: shape_u32,
                group_size: 0,
                data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
                spilled_len: 0,
            });
        }
    }
    Ok(out)
}

// ─── artifact ───────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub(crate) struct RaonSummary {
    pub(crate) tensors: usize,
    pub(crate) qt44: usize,
    pub(crate) sidecars: usize,
    pub(crate) padded: usize,
    pub(crate) halves_verified: usize,
    pub(crate) by_qt: BTreeMap<String, usize>,
    pub(crate) bytes: u64,
    /// Storage of the always-active FFN weights recorded in the artifact metadata.
    pub(crate) shared_ffn: String,
}

fn sha256_hex(path: &Path) -> Result<String, String> {
    crate::calibration::sha256_file_hex(path)
}

fn read_json(path: &Path) -> Result<Option<serde_json::Value>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let s = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&s)
        .map(Some)
        .map_err(|e| format!("parse {}: {e}", path.display()))
}

/// The only recipe Raon has: `--format mq4 --mq4v2-symmetric --tier xt`.
/// Anything else is refused instead of being silently reinterpreted.
pub(crate) fn run_recipe(
    input: &Path,
    output: &Path,
    format: &str,
    symmetric: bool,
    tier: Option<&str>,
) -> Result<RaonSummary, String> {
    if format != "mq4" || !symmetric || tier != Some("xt") {
        return Err(format!(
            "Raon-1-mini is produced only with `--format mq4 --mq4v2-symmetric --tier xt` \
             (got format={format}, mq4v2-symmetric={symmetric}, tier={tier:?})"
        ));
    }
    run_cli(input, output)
}

/// Convert the official Raon-1-mini checkpoint in `input` into the symmetric
/// qt44 control artifact at `output`, then read the artifact back and verify it.
pub(crate) fn run_cli(input: &Path, output: &Path) -> Result<RaonSummary, String> {
    let shared = shared_ffn_storage()?;
    let recipe = recipe_with_shared(RECIPE, shared.as_str());
    let cfg_path = input.join("config.json");
    let cfg = read_json(&cfg_path)?.ok_or_else(|| format!("{} not found", cfg_path.display()))?;
    let src = SourceIndex::open(input)?;
    let jobs = src.validate(&cfg)?;
    eprintln!(
        "raon: {} source weight tensors validated ({} routed MXFP8)",
        jobs.len(),
        jobs.iter()
            .filter(|(n, _)| is_routed_expert_weight(n))
            .count()
    );

    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if !parent.is_dir() {
        return Err(format!(
            "output parent directory does not exist: {}",
            parent.display()
        ));
    }
    let mut spill = TensorSpill::new(parent).map_err(|e| format!("create spill file: {e}"))?;
    let mut tensors: Vec<HfqTensor> = Vec::new();
    let mut padding: BTreeMap<String, [usize; 2]> = BTreeMap::new();
    let mut census: BTreeMap<String, usize> = BTreeMap::new();
    let mut halves_verified = 0usize;
    let started = std::time::Instant::now();
    const BATCH: usize = 48;
    for (bi, batch) in jobs.chunks(BATCH).enumerate() {
        let outs: Vec<Result<JobOut, String>> = batch
            .par_iter()
            .map(|(name, storage)| convert(&src, name, *storage))
            .collect();
        for out in outs {
            let out = out?;
            if let Some((n, k, phys)) = out.padded {
                padding.insert(n, [k, phys]);
            }
            for mut t in out.tensors {
                if t.quant_type == QuantType::MQ4G256V2 {
                    halves_verified += t.data.len() / MQ4V2_GROUP_BYTES * 2;
                }
                *census.entry(format!("{:?}", t.quant_type)).or_insert(0) += 1;
                let n = spill
                    .spill(&t.data)
                    .map_err(|e| format!("spill {}: {e}", t.name))?;
                t.data = Vec::new();
                t.spilled_len = n;
                tensors.push(t);
            }
        }
        if bi % 8 == 0 {
            eprintln!(
                "raon: {}/{} tensors, {:.0}s",
                ((bi + 1) * BATCH).min(jobs.len()),
                jobs.len(),
                started.elapsed().as_secs_f64()
            );
        }
    }

    let tokenizer = std::fs::read_to_string(input.join("tokenizer.json"))
        .map_err(|e| format!("read tokenizer.json: {e}"))?;
    let mut tokenizer_config =
        read_json(&input.join("tokenizer_config.json"))?.unwrap_or_else(|| serde_json::json!({}));
    let jinja = std::fs::read_to_string(input.join("chat_template.jinja"))
        .map_err(|e| format!("read chat_template.jinja: {e}"))?;
    if tokenizer_config
        .get("chat_template")
        .is_none_or(|v| v.is_null())
    {
        tokenizer_config["chat_template"] = jinja.into();
    }
    let generation_config = read_json(&input.join("generation_config.json"))?
        .ok_or("generation_config.json not found")?;
    let mut source_sha = serde_json::Map::new();
    for f in [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "chat_template.jinja",
    ] {
        if input.join(f).exists() {
            source_sha.insert(f.to_string(), sha256_hex(&input.join(f))?.into());
        }
    }
    let padding_json: serde_json::Map<String, serde_json::Value> = padding
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                serde_json::json!({"logical_k": v[0], "physical_k": v[1]}),
            )
        })
        .collect();
    let metadata = serde_json::json!({
        "architecture": RAON_MODEL_TYPE,
        "config": cfg,
        "tokenizer": tokenizer,
        "tokenizer_config": tokenizer_config,
        "generation_config": generation_config,
        "mq4v2.symmetric": 1,
        "hipfire_product_tier": "xt",
        "hipfire_base_format": "mq4v2",
        "hipfire_symmetric_verified": true,
        "hipfire_quant_recipe": recipe,
        "hipfire_lm_head": if lm_head_storage() == Storage::Q8F16 { "q8f16" } else { "qt44" },
        "hipfire_source_revision": SOURCE_REVISION,
        "hipfire_recipe_census": census,
        "hipfire_raon": {
            "arch_id": RAON_ARCH_ID,
            "model_type": RAON_MODEL_TYPE,
            "source_revision": SOURCE_REVISION,
            "quant_recipe": recipe,
            "shared_ffn": shared.as_str(),
            "symmetric_verified": true,
            "awq_sidecars": "unit-f16-physical-width",
            "mxfp8": {"block": [1, MXFP8_BLOCK_K], "scale": "e8m0-raw-u8", "applies_to": "routed experts"},
            "padded_tensors": padding_json,
            "source_sha256": source_sha,
        },
    });
    let metadata_json = serde_json::to_string(&metadata).map_err(|e| format!("metadata: {e}"))?;
    write_hfq(
        output,
        RAON_ARCH_ID,
        &metadata_json,
        &tensors,
        Some(&mut spill),
    )
    .map_err(|e| format!("write {}: {e}", output.display()))?;
    spill.cleanup();
    eprintln!(
        "raon: wrote {} in {:.0}s; verifying",
        output.display(),
        started.elapsed().as_secs_f64()
    );

    match verify_artifact(output, true) {
        Ok(mut summary) => {
            summary.halves_verified = summary.halves_verified.max(halves_verified);
            Ok(summary)
        }
        Err(e) => {
            let _ = std::fs::remove_file(output);
            Err(format!(
                "artifact verification failed (output removed): {e}"
            ))
        }
    }
}

/// Read an artifact back and verify the class map, padding, sidecars and every
/// qt44 half header. `unit_sidecars` additionally requires the sidecars to be
/// the unit (all 1.0) RTN-control values.
pub(crate) fn verify_artifact(path: &Path, unit_sidecars: bool) -> Result<RaonSummary, String> {
    let f = RuntimeHfqFile::open_with_reap_plan(path, None)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    if f.arch_id != RAON_ARCH_ID {
        return Err(format!("arch_id {} != {RAON_ARCH_ID}", f.arch_id));
    }
    let meta: serde_json::Value =
        serde_json::from_str(&f.metadata_json).map_err(|e| format!("metadata JSON: {e}"))?;
    let m = |k: &str| meta.get(k);
    if m("architecture").and_then(|v| v.as_str()) != Some(RAON_MODEL_TYPE)
        || m("hipfire_symmetric_verified").and_then(|v| v.as_bool()) != Some(true)
        || m("mq4v2.symmetric").and_then(|v| v.as_u64()) != Some(1)
        || m("hipfire_source_revision").and_then(|v| v.as_str()) != Some(SOURCE_REVISION)
        || m("hipfire_quant_recipe").and_then(|v| v.as_str()).is_none()
        || m("hipfire_product_tier").and_then(|v| v.as_str()) != Some("xt")
        || m("hipfire_base_format").and_then(|v| v.as_str()) != Some("mq4v2")
    {
        return Err("artifact metadata lacks the Raon identity/symmetric markers".to_string());
    }
    for key in [
        "config",
        "tokenizer",
        "tokenizer_config",
        "generation_config",
    ] {
        if m(key).is_none() {
            return Err(format!("artifact metadata lacks `{key}`"));
        }
    }
    if m("tokenizer_config")
        .and_then(|t| t.get("chat_template"))
        .and_then(|v| v.as_str())
        .is_none()
    {
        return Err("artifact metadata lacks the chat template".to_string());
    }
    let (shared, scope) = shared_ffn_of_metadata(&meta)?;
    let padding = meta
        .pointer("/hipfire_raon/padded_tensors")
        .and_then(|v| v.as_object())
        .ok_or("metadata lacks hipfire_raon.padded_tensors")?;

    let tensors = f.tensors();
    let by_name: BTreeMap<&str, &hipfire_runtime::hfq::HfqTensorInfo> =
        tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    if by_name.len() != tensors.len() {
        return Err("duplicate tensor names in artifact".to_string());
    }
    let mut s = RaonSummary::default();
    let mut padded_seen = 0usize;
    for t in tensors {
        s.tensors += 1;
        *s.by_qt.entry(format!("qt{}", t.quant_type)).or_insert(0) += 1;
        s.bytes += t.data_size as u64;
        if let Some(stem) = t.name.strip_suffix(".awq_scale.weight") {
            let parent = format!("{stem}.weight");
            let p = by_name
                .get(parent.as_str())
                .ok_or_else(|| format!("orphan sidecar {}", t.name))?;
            if t.quant_type != QuantType::F16 as u8 || t.shape != [p.shape[1]] || p.quant_type != 44
            {
                return Err(format!(
                    "sidecar {} is not F16[physical K] of a qt44 parent",
                    t.name
                ));
            }
            if unit_sidecars {
                let (_, bytes) = f.tensor_data(&t.name).ok_or("sidecar data missing")?;
                let one = f32_to_f16(1.0).to_le_bytes();
                if bytes.chunks_exact(2).any(|c| c != one) {
                    return Err(format!("sidecar {} is not unit", t.name));
                }
            }
            s.sidecars += 1;
            continue;
        }
        let storage = match (t.name.as_str(), t.quant_type) {
            // The head is admitted at either codec; the producer records its choice.
            ("lm_head.weight", 3) => Storage::Q8F16,
            ("lm_head.weight", 44) => Storage::Qt44,
            _ => classify_scoped(&t.name, Storage::Q8F16, shared, scope)?,
        };
        let ok_qt = match storage {
            Storage::Q8F16 | Storage::Q8F16Padded => t.quant_type == 3,
            Storage::Qt44 | Storage::Qt44Padded => t.quant_type == 44,
            Storage::Bf16 | Storage::Bf16Padded => t.quant_type == 16,
            Storage::Lossless => matches!(t.quant_type, 1 | 2),
            Storage::F32 => t.quant_type == 2,
        };
        if !ok_qt {
            return Err(format!(
                "{}: qt{} violates class {storage:?}",
                t.name, t.quant_type
            ));
        }
        // Geometry is checked for qt44 tensors and for every always-active FFN weight
        // (whatever codec the artifact stores them in).
        if storage.is_qt44() || is_shared_ffn_weight(&t.name) {
            if t.shape.len() != 2 {
                return Err(format!("{}: malformed {storage:?} entry", t.name));
            }
            let want_group = match storage {
                Storage::Qt44 | Storage::Qt44Padded => GROUP as u32,
                Storage::Q8F16 | Storage::Q8F16Padded => 32,
                _ => 0,
            };
            if t.group_size != want_group {
                return Err(format!(
                    "{}: group_size {} != {want_group} for {storage:?}",
                    t.name, t.group_size
                ));
            }
            let (rows, phys) = (t.shape[0] as usize, t.shape[1] as usize);
            if let Some(entry) = padding.get(&t.name) {
                let (lk, pk) = (
                    entry["logical_k"].as_u64().unwrap_or(0) as usize,
                    entry["physical_k"].as_u64().unwrap_or(0) as usize,
                );
                if !storage.is_padded() || pk != phys || physical_k(storage, lk) != pk {
                    return Err(format!(
                        "{}: padding record {entry} inconsistent with shape",
                        t.name
                    ));
                }
                padded_seen += 1;
            } else if storage.is_padded() && phys % GROUP != 0 {
                return Err(format!(
                    "{}: padded class but physical K {phys} not a multiple of {GROUP}",
                    t.name
                ));
            }
            let has_sidecar = by_name.contains_key(awq_sidecar_name(&t.name).as_str());
            if storage.is_qt44() {
                if phys % GROUP != 0 {
                    return Err(format!(
                        "{}: physical K {phys} not a multiple of {GROUP}",
                        t.name
                    ));
                }
                if !has_sidecar {
                    return Err(format!("{}: missing AWQ sidecar", t.name));
                }
                let (_, bytes) = f.tensor_data(&t.name).ok_or("qt44 data missing")?;
                if bytes.len() != rows * (phys / GROUP) * MQ4V2_GROUP_BYTES {
                    return Err(format!("{}: qt44 byte size does not match shape", t.name));
                }
                s.halves_verified += verify_symmetric_headers(&t.name, bytes)?;
                s.qt44 += 1;
            } else {
                if has_sidecar {
                    return Err(format!(
                        "{}: {storage:?} storage must not carry an AWQ sidecar",
                        t.name
                    ));
                }
                let want_bytes = match storage {
                    Storage::Q8F16 | Storage::Q8F16Padded => {
                        if phys % 32 != 0 {
                            return Err(format!("{}: Q8F16 physical K {phys} % 32 != 0", t.name));
                        }
                        rows * (phys / 32) * Q8F16_BLOCK_BYTES
                    }
                    _ => rows * phys * 2,
                };
                if t.data_size != want_bytes {
                    return Err(format!(
                        "{}: {storage:?} byte size {} does not match shape [{rows},{phys}] ({want_bytes})",
                        t.name, t.data_size
                    ));
                }
            }
        }
    }
    if padded_seen != padding.len() {
        return Err(format!(
            "padding metadata names {} tensors but {padded_seen} are present and padded",
            padding.len()
        ));
    }
    s.padded = padded_seen;
    s.shared_ffn = storage_tag(shared, scope);
    if s.sidecars != s.qt44 {
        return Err(format!(
            "{} qt44 tensors but {} sidecars",
            s.qt44, s.sidecars
        ));
    }
    Ok(s)
}

/// `hipfire_raon.shared_ffn` of an artifact; artifacts that predate the key store qt44.
fn shared_ffn_of_metadata(meta: &serde_json::Value) -> Result<(SharedFfn, Scope), String> {
    let Some(v) = meta.pointer("/hipfire_raon/shared_ffn") else {
        return Ok((SharedFfn::Qt44, Scope::All));
    };
    let tag = v
        .as_str()
        .ok_or("metadata hipfire_raon.shared_ffn is not a string")?;
    let ctx = |e: String| format!("metadata hipfire_raon.shared_ffn: {e}");
    let (shared, scope) = match tag.split_once(':') {
        Some((s, sc)) => (
            SharedFfn::parse(s).map_err(ctx)?,
            Scope::parse(sc).map_err(ctx)?,
        ),
        None => (SharedFfn::parse(tag).map_err(ctx)?, Scope::All),
    };
    if storage_tag(shared, scope) != tag {
        return Err(format!(
            "metadata hipfire_raon.shared_ffn `{tag}` is not a canonical storage tag"
        ));
    }
    Ok((shared, scope))
}

// ─── always-active FFN swap / verify routes ─────────────────────────────────

/// `--raon-shared-ffn` outcome: tensor-level accounting of one swap.
#[derive(Debug, Default, Clone)]
pub(crate) struct SwapSummary {
    pub(crate) target: String,
    pub(crate) replaced: usize,
    pub(crate) dropped_sidecars: usize,
    pub(crate) passthrough: usize,
    pub(crate) passthrough_bytes: u64,
    pub(crate) replaced_bytes_before: u64,
    pub(crate) replaced_bytes_after: u64,
    pub(crate) dropped_sidecar_bytes: u64,
    pub(crate) input_bytes: u64,
    pub(crate) output_bytes: u64,
    pub(crate) verified: RaonSummary,
}

impl SwapSummary {
    pub(crate) fn report(&self) -> String {
        format!(
            "raon swap -> {}: {} tensors replaced ({} -> {} data bytes), {} AWQ sidecars dropped \
             ({} bytes), {} tensors byte-identical pass-through ({} bytes); file {} -> {} bytes \
             (delta {:+}); verified {} tensors ({} qt44, {} sidecars, {} padded, {} symmetric \
             half headers), qt census {:?}",
            self.target,
            self.replaced,
            self.replaced_bytes_before,
            self.replaced_bytes_after,
            self.dropped_sidecars,
            self.dropped_sidecar_bytes,
            self.passthrough,
            self.passthrough_bytes,
            self.input_bytes,
            self.output_bytes,
            self.output_bytes as i128 - self.input_bytes as i128,
            self.verified.tensors,
            self.verified.qt44,
            self.verified.sidecars,
            self.verified.padded,
            self.verified.halves_verified,
            self.verified.by_qt,
        )
    }
}

/// Row index of the always-active FFN class tables used by the diff route.
const CLASS_PASS: usize = 0;
const CLASS_GATE_UP: usize = 1;
const CLASS_DOWN: usize = 2;
const CLASS_DENSE: usize = 3;
const CLASS_SIDECAR: usize = 4;
const CLASS_META: usize = 5;
const CLASS_NAMES: [&str; 6] = [
    "pass-through",
    "shared gate/up",
    "shared down",
    "dense mlp",
    "dropped sidecar",
    "metadata",
];

/// Table row of an always-active FFN weight (`CLASS_GATE_UP|DOWN|DENSE`).
fn ffn_class_index(name: &str) -> Option<usize> {
    let (_, rest) = layer_rest(name)?;
    match rest {
        "moe.shared_experts.gate_proj.weight" | "moe.shared_experts.up_proj.weight" => {
            Some(CLASS_GATE_UP)
        }
        "moe.shared_experts.down_proj.weight" => Some(CLASS_DOWN),
        "mlp.gate_proj.weight" | "mlp.up_proj.weight" | "mlp.down_proj.weight" => Some(CLASS_DENSE),
        _ => None,
    }
}

fn census_of<'a>(
    tensors: impl Iterator<Item = (&'a str, u8)>,
) -> Result<BTreeMap<String, usize>, String> {
    let mut census = BTreeMap::new();
    for (name, qt) in tensors {
        let q = QuantType::from_u8(qt).ok_or_else(|| format!("{name}: unknown qt{qt}"))?;
        *census.entry(format!("{q:?}")).or_insert(0usize) += 1;
    }
    Ok(census)
}

/// Expected payload bytes of a swapped (Q8F16 / BF16) `[rows, phys]` tensor.
fn swapped_bytes(target: SharedFfn, rows: usize, phys: usize) -> usize {
    match target {
        SharedFfn::Q8F16 => rows * (phys / 32) * Q8F16_BLOCK_BYTES,
        _ => rows * phys * 2,
    }
}

fn swapped_qt(target: SharedFfn) -> u8 {
    match target {
        SharedFfn::Q8F16 => QuantType::Q8F16 as u8,
        _ => QuantType::BF16 as u8,
    }
}

fn swapped_group(target: SharedFfn) -> u32 {
    match target {
        SharedFfn::Q8F16 => 32,
        _ => 0,
    }
}

/// Build an always-active-FFN variant of a shipped qt44 Raon artifact without
/// re-running calibration: every tensor outside the swap set is copied
/// byte-identically (streamed through a 4 MiB buffer, original index order); the
/// shared/dense gate/up/down weights are re-quantized from the source checkpoint at
/// `target` (same `convert` path as a full conversion) at the same index position;
/// their AWQ sidecars are dropped; the metadata is updated. Written atomically, then
/// verified with [`verify_artifact`] (non-unit sidecars, as for any XTS artifact).
pub(crate) fn swap_shared_ffn(
    input: &Path,
    source: &Path,
    output: &Path,
    target: SharedFfn,
    scope: Scope,
) -> Result<SwapSummary, String> {
    if target == SharedFfn::Qt44 {
        return Err(
            "--raon-shared-ffn target must be q8f16 or bf16 (qt44 is the shipped storage)"
                .to_string(),
        );
    }
    if let (Ok(a), Ok(b)) = (input.canonicalize(), output.canonicalize()) {
        if a == b {
            return Err("--output must differ from --input".to_string());
        }
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if !parent.is_dir() {
        return Err(format!(
            "output parent directory does not exist: {}",
            parent.display()
        ));
    }

    let f = RuntimeHfqFile::open_with_reap_plan(input, None)
        .map_err(|e| format!("open {}: {e}", input.display()))?;
    if f.arch_id != RAON_ARCH_ID {
        return Err(format!(
            "input arch_id {} != {RAON_ARCH_ID} ({RAON_MODEL_TYPE})",
            f.arch_id
        ));
    }
    let mut meta: serde_json::Value =
        serde_json::from_str(&f.metadata_json).map_err(|e| format!("metadata JSON: {e}"))?;
    if meta.get("architecture").and_then(|v| v.as_str()) != Some(RAON_MODEL_TYPE) {
        return Err(format!(
            "input metadata architecture is not `{RAON_MODEL_TYPE}`"
        ));
    }
    if meta.get("hipfire_source_revision").and_then(|v| v.as_str()) != Some(SOURCE_REVISION) {
        return Err(format!(
            "input was not built from source revision {SOURCE_REVISION}"
        ));
    }
    let tag = storage_tag(target, scope);
    let (current, current_scope) = shared_ffn_of_metadata(&meta)?;
    if current == target && current_scope == scope {
        return Err(format!(
            "input already stores the always-active FFN as {tag}; nothing to swap"
        ));
    }
    if current != SharedFfn::Qt44 {
        return Err(format!(
            "input stores the always-active FFN as {}; the swap starts from a qt44 artifact",
            storage_tag(current, current_scope)
        ));
    }
    let cfg = meta
        .get("config")
        .cloned()
        .ok_or("input metadata lacks `config`")?;
    let src = SourceIndex::open(source)?;
    src.validate(&cfg)?;
    if let Some(want) = meta
        .pointer("/hipfire_raon/source_sha256/config.json")
        .and_then(|v| v.as_str())
    {
        let got = sha256_hex(&source.join("config.json"))?;
        if got != want {
            return Err(format!(
                "--raon-source config.json sha256 {got} != the artifact's recorded source {want}"
            ));
        }
    }

    let tensors = f.tensors();
    let by_name: BTreeMap<&str, &hipfire_runtime::hfq::HfqTensorInfo> =
        tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    if by_name.len() != tensors.len() {
        return Err("duplicate tensor names in input artifact".to_string());
    }
    let present: BTreeSet<&str> = by_name
        .keys()
        .copied()
        .filter(|n| !n.ends_with(".awq_scale.weight"))
        .collect();
    let want: Vec<String> = expected_names(&cfg, &present)?
        .into_iter()
        .filter(|n| scope.covers(n))
        .collect();
    if want.is_empty() {
        return Err("config names no always-active FFN tensors".to_string());
    }
    let targets: BTreeSet<&str> = want.iter().map(String::as_str).collect();
    if let Some(extra) = present
        .iter()
        .copied()
        .find(|n| scope.covers(n) && !targets.contains(*n))
    {
        return Err(format!("unexpected always-active FFN tensor `{extra}`"));
    }
    for n in &want {
        let t = by_name
            .get(n.as_str())
            .ok_or_else(|| format!("target tensor `{n}` is missing from the input artifact"))?;
        if t.quant_type != 44 {
            return Err(format!(
                "target tensor `{n}` is qt{} (the swap needs the shipped qt44)",
                t.quant_type
            ));
        }
        if t.shape.len() != 2 {
            return Err(format!("target tensor `{n}` is not 2-D: {:?}", t.shape));
        }
        if !by_name.contains_key(awq_sidecar_name(n).as_str()) {
            return Err(format!("target tensor `{n}` has no AWQ sidecar"));
        }
    }

    let jobs: Vec<(&str, Storage)> = want
        .iter()
        .map(|n| classify_with(n, Storage::Q8F16, target).map(|s| (n.as_str(), s)))
        .collect::<Result<_, _>>()?;
    eprintln!(
        "raon swap: re-quantizing {} always-active FFN weights from {} to {}",
        jobs.len(),
        source.display(),
        target.as_str()
    );
    let outs: Vec<JobOut> = jobs
        .par_iter()
        .map(|(n, s)| convert(&src, n, *s))
        .collect::<Result<_, _>>()?;
    let mut repl: Vec<HfqTensor> = Vec::new();
    let mut repl_idx: BTreeMap<String, usize> = BTreeMap::new();
    let mut padding: BTreeMap<String, [usize; 2]> = BTreeMap::new();
    for out in outs {
        if let Some((n, k, phys)) = out.padded {
            padding.insert(n, [k, phys]);
        }
        for t in out.tensors {
            repl_idx.insert(t.name.clone(), repl.len());
            repl.push(t);
        }
    }
    for t in &repl {
        let a = by_name[t.name.as_str()];
        if t.shape != a.shape {
            return Err(format!(
                "{}: source re-quantizes to shape {:?} but the artifact holds {:?}",
                t.name, t.shape, a.shape
            ));
        }
    }

    #[derive(Clone, Copy)]
    enum Piece {
        Copy(usize),
        Repl(usize),
    }
    let mut entries: Vec<HfqIndexEntry> = Vec::with_capacity(tensors.len());
    let mut plan: Vec<Piece> = Vec::with_capacity(tensors.len());
    let mut s = SwapSummary {
        target: tag.clone(),
        ..SwapSummary::default()
    };
    for (i, t) in tensors.iter().enumerate() {
        if let Some(parent) = shared_ffn_sidecar_parent(&t.name).filter(|p| scope.covers(p)) {
            if !targets.contains(parent.as_str()) {
                return Err(format!("orphan sidecar {}", t.name));
            }
            s.dropped_sidecars += 1;
            s.dropped_sidecar_bytes += t.data_size as u64;
        } else if let Some(&ri) = repl_idx.get(&t.name) {
            let r = &repl[ri];
            entries.push(HfqIndexEntry {
                name: r.name.clone(),
                quant_type: r.quant_type as u8,
                shape: r.shape.clone(),
                group_size: r.group_size,
                data_len: r.data.len() as u64,
            });
            plan.push(Piece::Repl(ri));
            s.replaced += 1;
            s.replaced_bytes_before += t.data_size as u64;
            s.replaced_bytes_after += r.data.len() as u64;
        } else {
            entries.push(HfqIndexEntry {
                name: t.name.clone(),
                quant_type: t.quant_type,
                shape: t.shape.clone(),
                group_size: t.group_size,
                data_len: t.data_size as u64,
            });
            plan.push(Piece::Copy(i));
            s.passthrough += 1;
            s.passthrough_bytes += t.data_size as u64;
        }
    }
    if s.replaced != repl.len() || s.dropped_sidecars != repl.len() {
        return Err(format!(
            "swap accounting: {} replaced, {} sidecars dropped, {} converted",
            s.replaced,
            s.dropped_sidecars,
            repl.len()
        ));
    }

    let base_recipe = meta
        .get("hipfire_quant_recipe")
        .and_then(|v| v.as_str())
        .ok_or("input metadata lacks hipfire_quant_recipe")?
        .to_string();
    {
        let raon = meta
            .get_mut("hipfire_raon")
            .and_then(|v| v.as_object_mut())
            .ok_or("input metadata lacks the hipfire_raon object")?;
        raon.insert("shared_ffn".to_string(), tag.as_str().into());
        raon.insert(
            "shared_ffn_swap".to_string(),
            serde_json::json!({
                "from": SharedFfn::Qt44.as_str(),
                "to": target.as_str(),
                "scope": scope.as_str(),
                "source_revision": SOURCE_REVISION,
                "replaced_tensors": s.replaced,
                "dropped_sidecars": s.dropped_sidecars,
            }),
        );
        let pt = raon
            .get_mut("padded_tensors")
            .and_then(|v| v.as_object_mut())
            .ok_or("input metadata lacks hipfire_raon.padded_tensors")?;
        for n in &want {
            pt.remove(n);
        }
        for (n, [k, phys]) in &padding {
            pt.insert(
                n.clone(),
                serde_json::json!({"logical_k": k, "physical_k": phys}),
            );
        }
        if let Some(r) = raon
            .get("quant_recipe")
            .and_then(|v| v.as_str())
            .map(|r| recipe_with_shared(r, &tag))
        {
            raon.insert("quant_recipe".to_string(), r.into());
        }
    }
    meta["hipfire_quant_recipe"] = recipe_with_shared(&base_recipe, &tag).into();
    meta["hipfire_recipe_census"] = serde_json::to_value(census_of(
        entries.iter().map(|e| (e.name.as_str(), e.quant_type)),
    )?)
    .map_err(|e| format!("census: {e}"))?;
    let metadata_json = serde_json::to_string(&meta).map_err(|e| format!("metadata: {e}"))?;

    let mut src_file = File::open(input).map_err(|e| format!("open {}: {e}", input.display()))?;
    let mut buf = vec![0u8; COPY_CHUNK];
    write_hfq_streamed(
        output,
        RAON_ARCH_ID,
        &metadata_json,
        &entries,
        |i, out| match plan[i] {
            Piece::Repl(ri) => out.write_all(&repl[ri].data),
            Piece::Copy(ti) => {
                let t = &tensors[ti];
                src_file.seek(SeekFrom::Start(t.data_offset as u64))?;
                let mut left = t.data_size;
                while left > 0 {
                    let n = left.min(buf.len());
                    src_file.read_exact(&mut buf[..n])?;
                    out.write_all(&buf[..n])?;
                    left -= n;
                }
                Ok(())
            }
        },
    )
    .map_err(|e| format!("write {}: {e}", output.display()))?;
    s.input_bytes = std::fs::metadata(input)
        .map_err(|e| format!("stat {}: {e}", input.display()))?
        .len();
    s.output_bytes = std::fs::metadata(output)
        .map_err(|e| format!("stat {}: {e}", output.display()))?
        .len();
    match verify_artifact(output, false) {
        Ok(v) => s.verified = v,
        Err(e) => {
            let _ = std::fs::remove_file(output);
            return Err(format!("variant failed verification (output removed): {e}"));
        }
    }
    Ok(s)
}

/// One row of the [`raon_hfq_diff`] table.
#[derive(Debug, Clone, Default)]
pub(crate) struct DiffRow {
    pub(crate) class: &'static str,
    pub(crate) tensors: usize,
    pub(crate) a_qt: BTreeSet<u8>,
    pub(crate) b_qt: BTreeSet<u8>,
    pub(crate) a_bytes: u64,
    pub(crate) b_bytes: u64,
    pub(crate) violations: usize,
}

#[derive(Debug, Default)]
pub(crate) struct DiffReport {
    pub(crate) rows: Vec<DiffRow>,
    pub(crate) violations: Vec<String>,
}

impl DiffReport {
    pub(crate) fn ok(&self) -> bool {
        self.violations.is_empty()
    }

    fn fail(&mut self, class: usize, msg: String) {
        self.violations.push(msg);
        self.rows[class].violations += 1;
    }

    pub(crate) fn table(&self) -> String {
        let qts = |s: &BTreeSet<u8>| {
            if s.is_empty() {
                "-".to_string()
            } else {
                s.iter()
                    .map(|q| format!("qt{q}"))
                    .collect::<Vec<_>>()
                    .join(",")
            }
        };
        let mut out = format!(
            "{:<16} {:>8} {:>10} {:>10} {:>15} {:>15}  {}\n",
            "class", "tensors", "A qt", "B qt", "A bytes", "B bytes", "verdict"
        );
        for r in &self.rows {
            out.push_str(&format!(
                "{:<16} {:>8} {:>10} {:>10} {:>15} {:>15}  {}\n",
                r.class,
                r.tensors,
                qts(&r.a_qt),
                qts(&r.b_qt),
                r.a_bytes,
                r.b_bytes,
                if r.violations == 0 {
                    "OK".to_string()
                } else {
                    format!("FAIL({})", r.violations)
                }
            ));
        }
        out
    }
}

/// First (dotted) key at which two JSON objects differ.
fn first_meta_diff(a: &serde_json::Value, b: &serde_json::Value) -> Option<String> {
    let (Some(oa), Some(ob)) = (a.as_object(), b.as_object()) else {
        return (a != b).then(|| "<root>".to_string());
    };
    for k in oa.keys().chain(ob.keys()) {
        match (oa.get(k), ob.get(k)) {
            (Some(x), Some(y)) if x == y => {}
            (Some(x), Some(y)) if x.is_object() && y.is_object() => {
                return Some(first_meta_diff(x, y).map_or(k.clone(), |s| format!("{k}.{s}")));
            }
            _ => return Some(k.clone()),
        }
    }
    None
}

/// Read-only proof that `b` is exactly `a` with the always-active FFN swapped to
/// `target`: same arch and tensor order; every non-target tensor equal in
/// (name, qt, shape, group, size) and payload bytes (streamed in 4 MiB windows,
/// parallel across tensors); target tensors differ only as expected (qt, group,
/// size; same shape); exactly the targets' AWQ sidecars dropped; metadata equal
/// modulo the updated keys. `Err` only for I/O or unreadable containers; every
/// contract violation lands in [`DiffReport::violations`].
pub(crate) fn raon_hfq_diff(
    a: &Path,
    b: &Path,
    target: SharedFfn,
    scope: Scope,
) -> Result<DiffReport, String> {
    if target == SharedFfn::Qt44 {
        return Err("the variant storage must be q8f16 or bf16".to_string());
    }
    let tag = storage_tag(target, scope);
    let open = |p: &Path| {
        RuntimeHfqFile::open_with_reap_plan(p, None)
            .map_err(|e| format!("open {}: {e}", p.display()))
    };
    let fa = open(a)?;
    let fb = open(b)?;
    let mut rep = DiffReport {
        rows: CLASS_NAMES
            .iter()
            .map(|c| DiffRow {
                class: *c,
                ..DiffRow::default()
            })
            .collect(),
        violations: Vec::new(),
    };
    for (label, f) in [("A", &fa), ("B", &fb)] {
        if f.arch_id != RAON_ARCH_ID {
            rep.fail(
                CLASS_META,
                format!("{label}: arch_id {} != {RAON_ARCH_ID}", f.arch_id),
            );
        }
    }
    let (ta, tb) = (fa.tensors(), fb.tensors());

    // The B stream implied by A: sidecars of target weights vanish.
    let b_names: BTreeSet<&str> = tb.iter().map(|t| t.name.as_str()).collect();
    let mut kept: Vec<(&hipfire_runtime::hfq::HfqTensorInfo, usize)> = Vec::new();
    let mut n_targets = 0usize;
    for t in ta {
        if shared_ffn_sidecar_parent(&t.name).is_some_and(|p| scope.covers(&p)) {
            let r = &mut rep.rows[CLASS_SIDECAR];
            r.tensors += 1;
            r.a_qt.insert(t.quant_type);
            r.a_bytes += t.data_size as u64;
            if b_names.contains(t.name.as_str()) {
                rep.fail(CLASS_SIDECAR, format!("sidecar {} survives in B", t.name));
            }
            continue;
        }
        let class = if scope.covers(&t.name) {
            ffn_class_index(&t.name).unwrap_or(CLASS_PASS)
        } else {
            CLASS_PASS
        };
        if class != CLASS_PASS {
            n_targets += 1;
        }
        kept.push((t, class));
    }
    if rep.rows[CLASS_SIDECAR].tensors != n_targets {
        rep.fail(
            CLASS_SIDECAR,
            format!(
                "A has {n_targets} always-active FFN weights but {} of their sidecars",
                rep.rows[CLASS_SIDECAR].tensors
            ),
        );
    }
    if kept.len() != tb.len() {
        rep.fail(
            CLASS_PASS,
            format!(
                "B has {} tensors, expected {} (A minus dropped sidecars)",
                tb.len(),
                kept.len()
            ),
        );
    }

    let want_qt = swapped_qt(target);
    let want_group = swapped_group(target);
    let mut pass: Vec<(&str, &[u8], &[u8])> = Vec::new();
    for (idx, ((x, class), y)) in kept.iter().zip(tb.iter()).enumerate() {
        if x.name != y.name {
            rep.fail(
                CLASS_PASS,
                format!(
                    "tensor order/name differs at B index {idx}: expected `{}`, found `{}`",
                    x.name, y.name
                ),
            );
            break;
        }
        let class = *class;
        {
            let r = &mut rep.rows[class];
            r.tensors += 1;
            r.a_qt.insert(x.quant_type);
            r.b_qt.insert(y.quant_type);
            r.a_bytes += x.data_size as u64;
            r.b_bytes += y.data_size as u64;
        }
        if class == CLASS_PASS {
            if x.quant_type != y.quant_type
                || x.shape != y.shape
                || x.group_size != y.group_size
                || x.data_size != y.data_size
            {
                rep.fail(
                    CLASS_PASS,
                    format!(
                        "{}: index entry differs (A qt{} {:?} g{} {}B, B qt{} {:?} g{} {}B)",
                        x.name,
                        x.quant_type,
                        x.shape,
                        x.group_size,
                        x.data_size,
                        y.quant_type,
                        y.shape,
                        y.group_size,
                        y.data_size
                    ),
                );
                continue;
            }
            match (fa.tensor_data(&x.name), fb.tensor_data(&y.name)) {
                (Some((_, da)), Some((_, db))) => pass.push((x.name.as_str(), da, db)),
                _ => rep.fail(CLASS_PASS, format!("{}: payload unreadable", x.name)),
            }
            continue;
        }
        if x.quant_type != 44 {
            rep.fail(
                class,
                format!("{}: A is qt{}, expected qt44", x.name, x.quant_type),
            );
        }
        if y.quant_type != want_qt {
            rep.fail(
                class,
                format!("{}: B is qt{}, expected qt{want_qt}", y.name, y.quant_type),
            );
        }
        if y.shape != x.shape || y.shape.len() != 2 {
            rep.fail(
                class,
                format!("{}: B shape {:?} != A shape {:?}", y.name, y.shape, x.shape),
            );
            continue;
        }
        if y.group_size != want_group {
            rep.fail(
                class,
                format!("{}: B group_size {} != {want_group}", y.name, y.group_size),
            );
        }
        let (rows, phys) = (y.shape[0] as usize, y.shape[1] as usize);
        if class == CLASS_DOWN && phys % GROUP != 0 {
            rep.fail(
                class,
                format!("{}: B physical K {phys} not a multiple of {GROUP}", y.name),
            );
        }
        if target == SharedFfn::Q8F16 && phys % 32 != 0 {
            rep.fail(class, format!("{}: B physical K {phys} % 32 != 0", y.name));
            continue;
        }
        let want_bytes = swapped_bytes(target, rows, phys);
        if y.data_size != want_bytes {
            rep.fail(
                class,
                format!(
                    "{}: B size {} != expected {want_bytes}",
                    y.name, y.data_size
                ),
            );
        }
    }

    // Pass-through payloads, streamed in COPY_CHUNK windows.
    let bad: Vec<String> = pass
        .par_iter()
        .filter_map(|&(name, x, y)| {
            if x.len() != y.len() {
                return Some(format!("{name}: payload length differs"));
            }
            if x.par_chunks(COPY_CHUNK)
                .zip(y.par_chunks(COPY_CHUNK))
                .all(|(p, q)| p == q)
            {
                return None;
            }
            let at = x
                .iter()
                .zip(y.iter())
                .position(|(p, q)| p != q)
                .unwrap_or(0);
            Some(format!("{name}: payload differs (first byte {at})"))
        })
        .collect();
    for msg in bad {
        rep.fail(CLASS_PASS, msg);
    }

    // Metadata: equal modulo the keys the swap updates, and those carry the right values.
    let parse = |label: &str, f: &RuntimeHfqFile| {
        serde_json::from_str::<serde_json::Value>(&f.metadata_json)
            .map_err(|e| format!("{label} metadata JSON: {e}"))
    };
    let (ma, mb) = (parse("A", &fa)?, parse("B", &fb)?);
    let strip = |m: &serde_json::Value| {
        let mut m = m.clone();
        if let Some(o) = m.as_object_mut() {
            o.remove("hipfire_recipe_census");
            o.remove("hipfire_quant_recipe");
            if let Some(r) = o.get_mut("hipfire_raon").and_then(|v| v.as_object_mut()) {
                for k in ["shared_ffn", "shared_ffn_swap", "quant_recipe"] {
                    r.remove(k);
                }
            }
        }
        m
    };
    let mut checks = 0usize;
    let mut meta_check = |rep: &mut DiffReport, ok: bool, msg: String| {
        checks += 1;
        if !ok {
            rep.fail(CLASS_META, msg);
        }
    };
    meta_check(
        &mut rep,
        strip(&ma) == strip(&mb),
        format!(
            "metadata differs outside the updated keys at `{}`",
            first_meta_diff(&strip(&ma), &strip(&mb)).unwrap_or_default()
        ),
    );
    meta_check(
        &mut rep,
        shared_ffn_of_metadata(&ma) == Ok((SharedFfn::Qt44, Scope::All)),
        "A metadata hipfire_raon.shared_ffn is not qt44".to_string(),
    );
    meta_check(
        &mut rep,
        shared_ffn_of_metadata(&mb) == Ok((target, scope)),
        format!("B metadata hipfire_raon.shared_ffn is not {tag}"),
    );
    for (label, ptr) in [
        ("hipfire_quant_recipe", "/hipfire_quant_recipe"),
        ("hipfire_raon.quant_recipe", "/hipfire_raon/quant_recipe"),
    ] {
        if let Some(ra) = ma.pointer(ptr).and_then(|v| v.as_str()) {
            meta_check(
                &mut rep,
                mb.pointer(ptr).and_then(|v| v.as_str())
                    == Some(recipe_with_shared(ra, &tag).as_str()),
                format!("B {label} is not A's recipe + shared-ffn={tag}",),
            );
        } else if ptr == "/hipfire_quant_recipe" {
            meta_check(
                &mut rep,
                false,
                "A metadata lacks hipfire_quant_recipe".to_string(),
            );
        }
    }
    let census = census_of(tb.iter().map(|t| (t.name.as_str(), t.quant_type)))
        .ok()
        .and_then(|c| serde_json::to_value(c).ok());
    meta_check(
        &mut rep,
        census.is_some() && mb.get("hipfire_recipe_census") == census.as_ref(),
        "B hipfire_recipe_census does not match B's tensor index".to_string(),
    );
    let swap = mb.pointer("/hipfire_raon/shared_ffn_swap");
    meta_check(
        &mut rep,
        swap.and_then(|s| s.get("source_revision"))
            .and_then(|v| v.as_str())
            == Some(SOURCE_REVISION)
            && swap.and_then(|s| s.get("to")).and_then(|v| v.as_str()) == Some(target.as_str())
            && swap.and_then(|s| s.get("from")).and_then(|v| v.as_str()) == Some("qt44"),
        "B hipfire_raon.shared_ffn_swap lacks the expected source_revision/from/to".to_string(),
    );
    rep.rows[CLASS_META].tensors = checks;
    Ok(rep)
}

#[cfg(test)]
mod tests;
