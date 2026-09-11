// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::dspark_body::Qwen3DrafterAssets;
use crate::Llama;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::dspark_core::DsparkWeights;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCacheExt;
use hipfire_runtime::llama::{
    EmbeddingFormat, ForwardScratch, KvCache, KvDims, KvLayers, KvTarget, LayerWeights,
    LlamaConfig, LlamaWeights, WeightTensor,
};
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use hipfire_runtime::model_source::ModelSource as ModelSourceTrait;
use hipfire_runtime::weight_backend::hfq_weight_dtype;
use hipfire_runtime::weight_manifest::{plan_manifest, ManifestPlan, WeightEntry};
use hipfire_runtime::weight_store::{
    TakenWeight, WeightHandle, WeightLoadTransaction, WeightOrigin, WeightStoreAssembly,
    WeightStoreAssemblyGuard, WeightStoreError,
};
use rdna_compute::{DType, GpuTensor};
use std::collections::HashMap;

pub struct LlamaBundle {
    pub config: LlamaConfig,
    pub weights: LlamaWeights,
    pub scratch: ForwardScratch,
    pub kv: KvCache,
    /// The admitted mesh that owns this plan and the attached store origin.
    pub(crate) mesh: DeviceMesh,
    /// Pure declaration/placement plan captured at load time. The plan has no
    /// GPU handles and is immutable after publication.
    pub manifest_plan: ManifestPlan,
    /// A pilot store is attached only after its handles are assembled under
    /// this bundle. It is crate-visible so callers cannot create an independent
    /// unload owner; `ArchModel::free_gpu` is the sole release path. The
    /// attached store owns no allocation: resident weights belong to
    /// [`LlamaWeights`], scratch and KV stay bundle fields. It retains the
    /// validated projection/alias provenance plus value-only descriptors of
    /// the bundle-owned scratch/KV attachments — provenance, not ownership,
    /// and never manifest fulfillment entries.
    pub(crate) weight_store: Option<AttachedWeightStore>,
    /// Exact target identity captured before publication. The attached store
    /// binds this identity into its private drain capability, so teardown
    /// cannot encounter an origin mismatch.
    pub(crate) weight_origin: WeightOrigin,
    /// Decoder-layer indices whose residual hidden states a hidden-conditioned
    /// drafter (DFlash / EAGLE) wants captured, ascending order. Empty = no
    /// capture (the `SpecTarget::dflash_extract_layers` default of `None`). The
    /// speculator sets the real `target_layer_ids` via
    /// [`LlamaBundle::set_dflash_extract_layers`].
    pub dflash_extract_layers: Vec<usize>,
    /// Loaded DSpark drafter sidecar globals. `None` when no `-dspark` sidecar
    /// was found or speculation was disabled. Task-10 wires the speculator build.
    pub dspark_weights: Option<DsparkWeights>,
    /// Loaded DSpark drafter body assets (5-layer dense-GQA transformer +

    /// block-only KvCache/scratch). `None` when `dspark_weights` is `None`.
    pub dspark_assets: Option<Qwen3DrafterAssets>,
}

/// Crate-private attached owner for the manifest transaction.
///
/// The runtime transaction stays public only long enough for the load carrier
/// to assemble or roll it back. Once wrapped here, the only consuming path is
/// the crate's [`hipfire_runtime::arch_model::ArchModel::free_gpu`] implementation.
///
/// Single-owner truth: [`LlamaWeights`] owns every weight allocation and
/// frees it; scratch and KV are freed from the bundle fields in the existing
/// order. This wrapper retains the validated projection/alias provenance and
/// the scratch/KV attachment descriptors for post-publication description.
/// It frees nothing twice: assembly took every handle, so drain-time rollback
/// releases zero residents and the retained census drops without GPU work.
pub(crate) struct AttachedWeightStore {
    transaction: WeightLoadTransaction,
    attachments: AttachmentDescriptors,
}

/// Value-only descriptors of the bundle-owned scratch/KV attachments,
/// captured when the manifest transaction attaches. No handles move here and
/// nothing here is freed: scratch and KV were never manifest fulfillment
/// entries and stay owned (and torn down) by the bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AttachmentDescriptors {
    /// Owned scratch output width (logits shape) at attach.
    pub scratch_logits_shape: Vec<usize>,
    /// Owned KV geometry at attach.
    pub kv_dim: usize,
    pub kv_max_seq: usize,
    pub kv_physical_cap: usize,
    pub kv_n_kv_heads: usize,
    pub kv_head_dim: usize,
    pub kv_n_layers: usize,
}

impl AttachmentDescriptors {
    fn capture(scratch: &ForwardScratch, kv: &KvCache) -> Self {
        Self {
            scratch_logits_shape: scratch.logits.shape.clone(),
            kv_dim: kv.kv_dim,
            kv_max_seq: kv.max_seq,
            kv_physical_cap: kv.physical_cap,
            kv_n_kv_heads: kv.n_kv_heads,
            kv_head_dim: kv.head_dim,
            kv_n_layers: kv.k_gpu.len(),
        }
    }
}

impl AttachedWeightStore {
    fn from_transaction(
        transaction: WeightLoadTransaction,
        expected: WeightOrigin,
        attachments: AttachmentDescriptors,
    ) -> Result<Self, (WeightLoadTransaction, WeightStoreError)> {
        if let Err(error) = transaction.validate_origin_value(expected) {
            return Err((transaction, error));
        }
        Ok(Self {
            transaction,
            attachments,
        })
    }

    /// Consume the attached owner at unload. Assembly took every handle for
    /// the typed weights, so rollback releases zero residents; the retained
    /// provenance and attachment descriptors drop with the owner.
    pub(crate) fn drain(self, gpu: &mut rdna_compute::Gpu) -> hip_bridge::HipResult<()> {
        self.transaction.rollback(gpu)
    }

    /// Tied-source edge retained in provenance. Owns nothing.
    pub(crate) fn alias_source(
        &self,
        name: &str,
        layer: Option<usize>,
        device: usize,
    ) -> Option<&str> {
        self.transaction.alias_source(name, layer, device)
    }

    /// Value-only descriptors of the bundle-owned scratch/KV attachments.
    pub(crate) fn attachments(&self) -> &AttachmentDescriptors {
        &self.attachments
    }
}

fn with_weight_rollback_error(reason: String, rollback: hip_bridge::HipResult<()>) -> String {
    match rollback {
        Ok(()) => reason,
        Err(error) => format!("{reason}; resident rollback failed: {error}"),
    }
}

fn plan_single(
    config: &LlamaConfig,
    has_separate_lm_head: bool,
) -> Result<(DeviceMesh, ManifestPlan), String> {
    let mesh = DeviceMesh::single().map_err(|error| format!("llama: device mesh: {error}"))?;
    let manifest = Llama::weight_manifest_for_hfq(config, has_separate_lm_head);
    let state = Llama::state_manifest(config);
    let plan = plan_manifest(&manifest, &state, &mesh, config.n_layers)
        .map_err(|e| format!("llama: manifest planning failed: {e}"))?;
    Ok((mesh, plan))
}

fn llama_kv_dims(config: &LlamaConfig, max_seq: usize, physical_cap: Option<usize>) -> KvDims {
    KvDims {
        layers: KvLayers::Flat(config.n_layers),
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        max_seq,
        physical_cap,
    }
}

fn hfq_layer_names(layer: usize, relative: &str) -> Vec<String> {
    vec![
        format!("model.layers.{layer}.{relative}.weight"),
        format!("layers.{layer}.{relative}.weight"),
    ]
}
const HFQ_LM_HEAD_NAMES: &[&str] = &[
    "lm_head.weight",
    "model.lm_head.weight",
    "model.language_model.lm_head.weight",
];

fn hfq_has_separate_lm_head(hfq: &HfqFile) -> bool {
    HFQ_LM_HEAD_NAMES
        .iter()
        .any(|name| hfq.find_tensor_info(name).is_some())
}

fn hfq_entry_names(entry: &WeightEntry) -> Result<Vec<String>, String> {
    let names = match (entry.name.as_str(), entry.layer) {
        ("token_embd", None) => vec!["model.embed_tokens.weight".to_string()],
        ("output_norm", None) => vec!["model.norm.weight".to_string()],
        ("lm_head", None) => HFQ_LM_HEAD_NAMES
            .iter()
            .map(|name| (*name).to_string())
            .collect(),
        ("wq", Some(layer)) => hfq_layer_names(layer, "self_attn.q_proj"),
        ("wk", Some(layer)) => hfq_layer_names(layer, "self_attn.k_proj"),
        ("wv", Some(layer)) => hfq_layer_names(layer, "self_attn.v_proj"),
        ("wo", Some(layer)) => hfq_layer_names(layer, "self_attn.o_proj"),
        ("ffn_gate", Some(layer)) => hfq_layer_names(layer, "mlp.gate_proj"),
        ("ffn_up", Some(layer)) => hfq_layer_names(layer, "mlp.up_proj"),
        ("ffn_down", Some(layer)) => hfq_layer_names(layer, "mlp.down_proj"),
        ("attn_norm", Some(layer)) => hfq_layer_names(layer, "input_layernorm"),
        ("ffn_norm", Some(layer)) => hfq_layer_names(layer, "post_attention_layernorm"),
        ("q_norm", Some(layer)) => hfq_layer_names(layer, "self_attn.q_norm"),
        ("k_norm", Some(layer)) => hfq_layer_names(layer, "self_attn.k_norm"),
        (name, layer) => {
            return Err(format!(
                "llama: manifest entry {name}[layer {layer:?}] has no HFQ source mapping"
            ));
        }
    };
    Ok(names)
}

fn hfq_entry_data(hfq: &HfqFile, entry: &WeightEntry) -> Result<(Vec<u8>, u8), String> {
    for name in hfq_entry_names(entry)? {
        if let Some((info, data)) = hfq.tensor_data_vec(&name) {
            if !matches!(
                entry.name.as_str(),
                "token_embd" | "output_norm" | "attn_norm" | "ffn_norm" | "q_norm" | "k_norm"
            ) {
                let sidecar = match name.strip_suffix(".weight") {
                    Some(stem) => format!("{stem}.awq_scale.weight"),
                    None => format!("{name}.awq_scale.weight"),
                };
                if hfq.find_tensor_info(&sidecar).is_some() {
                    return Err(format!(
                        "llama: AWQ sidecar {sidecar} is not represented by the manifest pilot"
                    ));
                }
            }
            return Ok((data, info.quant_type));
        }
    }
    if entry.name == "lm_head" && entry.layer.is_none() {
        if let Some((info, data)) = hfq.tensor_data_vec("model.embed_tokens.weight") {
            return Ok((data, info.quant_type));
        }
    }
    Err(format!(
        "llama: source tensor for {}[layer {:?}] is missing",
        entry.name, entry.layer
    ))
}

fn f32_bytes_from_hfq(quant_type: u8, data: &[u8], name: &str) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::with_capacity(match quant_type {
        1 | 16 => data.len() * 2,
        2 => data.len(),
        _ => 0,
    });
    match quant_type {
        1 => {
            let chunks = data.chunks_exact(2);
            if !chunks.remainder().is_empty() {
                return Err(format!("{name}: truncated F16 payload"));
            }
            for chunk in chunks {
                bytes.extend_from_slice(
                    &hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]))
                        .to_le_bytes(),
                );
            }
        }
        2 => {
            if !data.len().is_multiple_of(4) {
                return Err(format!("{name}: truncated F32 payload"));
            }
            bytes.extend_from_slice(data);
        }
        16 => {
            let chunks = data.chunks_exact(2);
            if !chunks.remainder().is_empty() {
                return Err(format!("{name}: truncated BF16 payload"));
            }
            for chunk in chunks {
                bytes.extend_from_slice(
                    &f32::from_bits(u16::from_le_bytes([chunk[0], chunk[1]]) as u32 * (1 << 16))
                        .to_le_bytes(),
                );
            }
        }
        other => {
            return Err(format!(
                "{name}: quant_type={other} is not a host float payload"
            ));
        }
    }
    Ok(bytes)
}

fn hfq_source(hfq: &HfqFile, entry: &WeightEntry) -> Result<(Vec<u8>, DType), String> {
    let (data, quant_type) = hfq_entry_data(hfq, entry)?;
    let name = format!("{}[layer {:?}]", entry.name, entry.layer);
    if entry.name == "token_embd" {
        return match quant_type {
            1 | 2 | 16 => Ok((f32_bytes_from_hfq(quant_type, &data, &name)?, DType::F32)),
            3 => Ok((data, DType::Q8_0)),
            4 => Ok((data, DType::Q4K)),
            6 => Ok((data, DType::HFQ4G256)),
            7 => Ok((data, DType::HFQ4G128)),
            other => Err(format!(
                "{name}: quant_type={other} is unsupported for a LLaMA embedding"
            )),
        };
    }
    if matches!(
        entry.name.as_str(),
        "output_norm" | "attn_norm" | "ffn_norm" | "q_norm" | "k_norm"
    ) {
        return Ok((f32_bytes_from_hfq(quant_type, &data, &name)?, DType::F32));
    }
    match quant_type {
        1 | 2 | 16 => Ok((f32_bytes_from_hfq(quant_type, &data, &name)?, DType::F32)),
        other => hfq_weight_dtype(other)
            .map(|dtype| (data, dtype))
            .ok_or_else(|| format!("{name}: unsupported HFQ quant_type={other}")),
    }
}

fn take_slot(
    assembly: &mut WeightStoreAssembly<'_>,
    slots: &mut HashMap<(String, Option<usize>), usize>,
    name: &str,
    layer: Option<usize>,
) -> Result<(), String> {
    let slot = assembly
        .take(name, layer, 0)
        .ok_or_else(|| format!("llama: fulfilled store is missing {name}[layer {layer:?}]"))?;
    slots.insert((name.to_string(), layer), slot);
    Ok(())
}

fn require_materialized(
    assembly: &WeightStoreAssemblyGuard<'_>,
    name: &str,
    layer: Option<usize>,
    slot: usize,
) -> Result<(), String> {
    match assembly.get(slot) {
        Some(WeightHandle::Resident(_)) => Ok(()),
        Some(WeightHandle::Alias(source))
            if name == "lm_head" && layer.is_none() && source == "token_embd" =>
        {
            Ok(())
        }
        Some(WeightHandle::Alias(source)) => Err(format!(
            "llama: {name}[layer {layer:?}] aliases {source}; only lm_head may tie token_embd"
        )),
        None => Err(format!(
            "llama: {name}[layer {layer:?}] assembly slot {slot} is missing"
        )),
    }
}

fn resident_cell(
    cells: &mut HashMap<(String, Option<usize>), TakenWeight>,
    name: &str,
    layer: Option<usize>,
) -> GpuTensor {
    match cells.remove(&(name.to_string(), layer)) {
        Some(TakenWeight {
            handle: WeightHandle::Resident(tensor),
            ..
        }) => tensor,
        _ => unreachable!("validated LLaMA assembly lost resident {name}[layer {layer:?}]"),
    }
}

fn resident_weight(
    cells: &mut HashMap<(String, Option<usize>), TakenWeight>,
    name: &str,
    layer: Option<usize>,
    m: usize,
    k: usize,
) -> WeightTensor {
    let tensor = resident_cell(cells, name, layer);
    let dtype = tensor.dtype;
    WeightTensor {
        buf: tensor,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: dtype.row_stride(k),
        paro: None,
        awq_scale: None,
    }
}

fn tied_weight(
    cells: &mut HashMap<(String, Option<usize>), TakenWeight>,
    token_embd: &GpuTensor,
    embd_format: EmbeddingFormat,
    name: &str,
    layer: Option<usize>,
    m: usize,
    k: usize,
) -> WeightTensor {
    match cells.remove(&(name.to_string(), layer)) {
        Some(TakenWeight {
            handle: WeightHandle::Alias(source),
            ..
        }) if source == "token_embd" => {
            hipfire_runtime::weight_backend::tied_lm_head_alias(token_embd, embd_format, m, k)
        }
        _ => unreachable!("validated LLaMA assembly lost tied {name}[layer {layer:?}]"),
    }
}

fn embedding_format(dtype: DType) -> Result<EmbeddingFormat, String> {
    match dtype {
        DType::F32 => Ok(EmbeddingFormat::F32),
        DType::Q4K => Ok(EmbeddingFormat::Q4K),
        DType::HFQ4G256 => Ok(EmbeddingFormat::HFQ4G256),
        DType::HFQ4G128 => Ok(EmbeddingFormat::HFQ4G128),
        DType::Q8_0 => Ok(EmbeddingFormat::Q8_0),
        other => Err(format!(
            "llama: unsupported assembled embedding dtype {other:?}"
        )),
    }
}

fn assemble_llama_weights(
    config: &LlamaConfig,
    transaction: &mut WeightLoadTransaction,
) -> Result<LlamaWeights, String> {
    let mut assembly = transaction.begin_assembly();
    let mut slots = HashMap::new();
    let mut take =
        |name: &str, layer: Option<usize>| take_slot(&mut assembly, &mut slots, name, layer);

    take("token_embd", None)?;
    take("output_norm", None)?;
    take("lm_head", None)?;
    for layer in 0..config.n_layers {
        for name in [
            "wq",
            "wk",
            "wv",
            "wo",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
            "attn_norm",
            "ffn_norm",
        ] {
            take(name, Some(layer))?;
        }
        if config.has_qk_norm {
            take("q_norm", Some(layer))?;
            take("k_norm", Some(layer))?;
        }
    }

    drop(take);
    let guard = assembly.commit();
    for ((name, layer), slot) in &slots {
        require_materialized(&guard, name, *layer, *slot)?;
    }
    let token_slot = slots[&("token_embd".to_string(), None)];
    let token_dtype = match guard.get(token_slot) {
        Some(WeightHandle::Resident(tensor)) => tensor.dtype,
        _ => unreachable!("validated token_embd is not resident"),
    };
    let embd_format = embedding_format(token_dtype)?;
    let cells: HashMap<_, _> = guard
        .finalize()
        .into_iter()
        .map(|taken| ((taken.key.name.clone(), taken.key.layer), taken))
        .collect();
    let mut cells = cells;
    let token_embd = resident_cell(&mut cells, "token_embd", None);
    let output_norm = resident_cell(&mut cells, "output_norm", None);
    let lm_head_aliases_embd = matches!(
        cells.get(&("lm_head".to_string(), None)),
        Some(TakenWeight {
            handle: WeightHandle::Alias(_),
            ..
        })
    );
    let output = if lm_head_aliases_embd {
        tied_weight(
            &mut cells,
            &token_embd,
            embd_format,
            "lm_head",
            None,
            config.vocab_size,
            config.dim,
        )
    } else {
        resident_weight(&mut cells, "lm_head", None, config.vocab_size, config.dim)
    };
    let mut layers = Vec::with_capacity(config.n_layers);
    for layer in 0..config.n_layers {
        let q_norm = if config.has_qk_norm {
            Some(resident_cell(&mut cells, "q_norm", Some(layer)))
        } else {
            None
        };
        let k_norm = if config.has_qk_norm {
            Some(resident_cell(&mut cells, "k_norm", Some(layer)))
        } else {
            None
        };
        layers.push(LayerWeights {
            attn_norm: resident_cell(&mut cells, "attn_norm", Some(layer)),
            wq: resident_weight(
                &mut cells,
                "wq",
                Some(layer),
                config.n_heads * config.head_dim,
                config.dim,
            ),
            wk: resident_weight(
                &mut cells,
                "wk",
                Some(layer),
                config.n_kv_heads * config.head_dim,
                config.dim,
            ),
            wv: resident_weight(
                &mut cells,
                "wv",
                Some(layer),
                config.n_kv_heads * config.head_dim,
                config.dim,
            ),
            wo: resident_weight(
                &mut cells,
                "wo",
                Some(layer),
                config.dim,
                config.n_heads * config.head_dim,
            ),
            q_norm,
            k_norm,
            ffn_norm: resident_cell(&mut cells, "ffn_norm", Some(layer)),
            w_gate: resident_weight(
                &mut cells,
                "ffn_gate",
                Some(layer),
                config.hidden_dim,
                config.dim,
            ),
            w_up: resident_weight(
                &mut cells,
                "ffn_up",
                Some(layer),
                config.hidden_dim,
                config.dim,
            ),
            w_down: resident_weight(
                &mut cells,
                "ffn_down",
                Some(layer),
                config.dim,
                config.hidden_dim,
            ),
        });
    }
    debug_assert!(cells.is_empty(), "validated LLaMA assembly left cells");
    Ok(LlamaWeights {
        token_embd,
        embd_format,
        output_norm,
        output,
        layers,
        lm_head_aliases_embd,
    })
}

/// Build the LLaMA GPU bundle from an HFQ or safetensors-directory source.
///
/// The HFQ plain-LLaMA Single path is the production manifest pilot: planning
/// and source admission happen first, fulfillment uploads transactionally, and
/// typed handles are moved into `LlamaWeights` before the committed remainder
/// is published beneath this bundle's owner. The directory path remains on its
/// existing ParoQuant loader until that source has an equivalent representation
/// resolver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HfqLoadRoute {
    /// Plain, non-AWQ files admitted to the manifest/typed-assembly pilot.
    ManifestPlainLlama,
    /// Files carrying AWQ scale sidecars retain the established loader until
    /// sidecar ownership is represented by the manifest transaction.
    LegacyAwq,
}

fn classify_hfq_route(hfq: &HfqFile) -> HfqLoadRoute {
    if hfq.has_awq_sidecars() {
        HfqLoadRoute::LegacyAwq
    } else {
        HfqLoadRoute::ManifestPlainLlama
    }
}

pub fn load_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<LlamaBundle, String> {
    let (config, weights, kv, scratch, manifest_plan, weight_store, mesh, weight_origin) = match src
    {
        ModelSource::Hfq(hfq) => {
            let config =
                <Llama as Architecture>::config_from_hfq(&hfq).map_err(|e| e.to_string())?;
            // Admission and route classification are pure source checks.
            // They must run before any manifest fulfillment or GPU upload.
            hipfire_runtime::hfq::validate_llama_hfq_admission(&hfq).map_err(|e| e.to_string())?;
            let has_separate_lm_head = hfq_has_separate_lm_head(&hfq);
            let route = classify_hfq_route(&hfq);
            eprintln!("llama: HFQ source route = {route:?}");
            let (mesh, manifest_plan) = plan_single(&config, has_separate_lm_head)?;
            let weight_origin = WeightOrigin::for_single(&mesh, ctx.gpu);
            let (weights, mut weight_store) = match route {
                HfqLoadRoute::LegacyAwq => {
                    let weights = hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, ctx.gpu)
                        .map_err(|e| format!("llama: load_weights_hfq failed: {e:?}"))?;
                    (weights, None)
                }
                HfqLoadRoute::ManifestPlainLlama => {
                    let manifest = Llama::weight_manifest_for_hfq(&config, has_separate_lm_head);
                    // `weight_origin` was admitted with the plan above; binding
                    // it here fails before the first upload when the runtime
                    // mesh/GPU disagree with the plan identity.
                    let mut transaction = hipfire_runtime::weight_store::fulfill_manifest(
                        &manifest,
                        &mesh,
                        config.n_layers,
                        ctx.gpu,
                        weight_origin,
                        |entry| hfq_source(&hfq, entry),
                    )
                    .map_err(|e| format!("llama: {e}"))?;
                    let weights = match assemble_llama_weights(&config, &mut transaction) {
                        Ok(weights) => weights,
                        Err(error) => {
                            return Err(with_weight_rollback_error(
                                error,
                                transaction.rollback(ctx.gpu),
                            ));
                        }
                    };
                    (weights, Some(transaction))
                }
            };
            hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
            // The plain LLaMA path has no independent cap resolver. PR
            // #661's physical-cap behavior is owned by the existing
            // upstream KV plan.
            let scratch = match ForwardScratch::new_with_max_seq(ctx.gpu, &config, ctx.max_seq) {
                Ok(scratch) => scratch,
                Err(error) => {
                    let rollback = if let Some(transaction) = weight_store.take() {
                        transaction.rollback(ctx.gpu)
                    } else {
                        Ok(())
                    };
                    weights.free_gpu(ctx.gpu);
                    return Err(with_weight_rollback_error(
                        format!("llama: ForwardScratch::new_with_max_seq failed: {error:?}"),
                        rollback,
                    ));
                }
            };
            let dims = llama_kv_dims(&config, ctx.max_seq, None);
            let kv = match <KvCache as KvCacheExt>::from_mode(
                hipfire_runtime::kv_mode::resolve(
                    ctx.kv_mode_override.unwrap_or(""),
                    &hipfire_runtime::kv_mode::LLAMA_HFQ_POLICY,
                )
                .mode,
                KvTarget::Single(ctx.gpu),
                &dims,
            ) {
                Ok(kv) => kv,
                Err(error) => {
                    scratch.free_gpu(ctx.gpu);
                    let rollback = if let Some(transaction) = weight_store.take() {
                        transaction.rollback(ctx.gpu)
                    } else {
                        Ok(())
                    };
                    weights.free_gpu(ctx.gpu);
                    return Err(with_weight_rollback_error(
                        format!("llama: <KvCache as KvCacheExt>::from_mode failed: {error}"),
                        rollback,
                    ));
                }
            };
            (
                config,
                weights,
                kv,
                scratch,
                manifest_plan,
                weight_store,
                mesh,
                weight_origin,
            )
        }
        ModelSource::Dir(source) => {
            let config = hipfire_runtime::hfq::config_from_safetensors_llama(&source)
                .map_err(|e| format!("failed to parse LLaMA/Qwen3 config from config.json: {e}"))?;
            let (mesh, manifest_plan) =
                plan_single(&config, source.tensor_info("lm_head.weight").is_some())?;
            let weight_origin = WeightOrigin::for_single(&mesh, ctx.gpu);
            let weights =
                hipfire_runtime::hfq::load_weights_paroquant_llama(&source, &config, ctx.gpu)
                    .map_err(|e| format!("load_weights_paroquant_llama: {e:?}"))?;
            hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
            let kv_mode_str = ctx
                .kv_mode_override
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .unwrap_or_else(|| hipfire_runtime::config::get().kv_mode.clone());
            let rr = hipfire_runtime::kv_mode::resolve(
                &kv_mode_str,
                &hipfire_runtime::kv_mode::DIR_SAFETENSORS_POLICY,
            );
            if let Some(w) = rr.warning {
                eprintln!(
                    "  KV cache: {w} (site {})",
                    hipfire_runtime::kv_mode::DIR_SAFETENSORS_POLICY.site
                );
            }
            let dims = llama_kv_dims(&config, ctx.max_seq, Some(ctx.max_seq));
            let kv =
                match <KvCache as KvCacheExt>::from_mode(rr.mode, KvTarget::Single(ctx.gpu), &dims)
                {
                    Ok(kv) => kv,
                    Err(error) => {
                        weights.free_gpu(ctx.gpu);
                        return Err(format!("KvCache: {error}"));
                    }
                };
            let scratch = match ForwardScratch::new_with_max_seq(ctx.gpu, &config, ctx.max_seq) {
                Ok(scratch) => scratch,
                Err(error) => {
                    let _ = kv.free_gpu(ctx.gpu);
                    weights.free_gpu(ctx.gpu);
                    return Err(format!("ForwardScratch::new_with_max_seq: {error:?}"));
                }
            };
            (
                config,
                weights,
                kv,
                scratch,
                manifest_plan,
                None,
                mesh,
                weight_origin,
            )
        }
    };

    let mut bundle = LlamaBundle {
        config,
        weights,
        scratch,
        kv,
        manifest_plan,
        weight_store: None,
        weight_origin,
        mesh,
        dflash_extract_layers: Vec::new(),
        dspark_weights: None,
        dspark_assets: None,
    };
    if let Some(transaction) = weight_store {
        if let Err((transaction, error)) = bundle.attach_weight_store(transaction) {
            let LlamaBundle {
                weights,
                scratch,
                kv,
                ..
            } = bundle;
            let rollback = transaction.rollback(ctx.gpu);
            scratch.free_gpu(ctx.gpu);
            weights.free_gpu(ctx.gpu);
            let _ = kv.free_gpu(ctx.gpu);
            return Err(with_weight_rollback_error(error, rollback));
        }
    }
    Ok(bundle)
}

/// Alias matching the `load_<arch>_bundle` naming convention in the task.
pub use load_bundle as load_llama_bundle;

impl LlamaBundle {
    /// target identity. The resulting owner is crate-private and can only be
    /// consumed by `ArchModel::free_gpu`.
    ///
    /// The attached owner retains the transaction's validated
    /// projection/alias provenance plus value-only descriptors of the
    /// bundle-owned scratch/KV attachments. It owns no allocation: weights
    /// belong to [`LlamaWeights`], scratch and KV stay bundle fields in the
    /// existing teardown order.
    fn attach_weight_store(
        &mut self,
        transaction: WeightLoadTransaction,
    ) -> Result<(), (WeightLoadTransaction, String)> {
        if self.weight_store.is_some() {
            return Err((transaction, "llama: weight store already attached".into()));
        }
        let attachments = AttachmentDescriptors::capture(&self.scratch, &self.kv);
        let attached = match AttachedWeightStore::from_transaction(
            transaction,
            self.weight_origin,
            attachments,
        ) {
            Ok(attached) => attached,
            Err((transaction, error)) => {
                return Err((
                    transaction,
                    format!("llama: weight store origin rejected: {error}"),
                ));
            }
        };
        self.weight_store = Some(attached);
        Ok(())
    }

    /// The immutable mesh identity used by this bundle's manifest plan.
    /// Callers that run the Single pilot must pass this exact mesh to
    /// `fulfill_manifest`; constructing a fresh `DeviceMesh::single()` would
    /// intentionally fail the origin check.
    pub fn manifest_mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    /// Set the decoder-layer indices whose residual hidden states the
    /// hidden-conditioned drafter wants captured (ascending order). The
    /// speculator calls this with `dflash::DflashConfig::target_layer_ids`.
    pub fn set_dflash_extract_layers(&mut self, layers: Vec<usize>) {
        debug_assert!(
            layers.windows(2).all(|w| w[0] < w[1]),
            "dflash extract layers must be strictly ascending: {layers:?}"
        );
        self.dflash_extract_layers = layers;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serializes the pinned-fixture GPU evidence tests: hipMemGetInfo is
    /// device-global, so parallel GPU tests in the same process would corrupt
    /// the VRAM measurements and add noise to the load/unload cycle floors.
    static GPU_EVIDENCE_LOCK: Mutex<()> = Mutex::new(());

    use hipfire_runtime::arch_model::ArchModel;
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqFile, HfqMemTensor};
    use hipfire_runtime::kv_backend::KvBackend;
    use hipfire_runtime::kv_mode::KvMode;
    use hipfire_runtime::llama::ModelArch;
    use hipfire_runtime::llama::{
        forward_scratch_compute, forward_scratch_embed, KvCache, KvCacheExt, KvDims, KvLayers,
        KvTarget,
    };
    use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
    use hipfire_runtime::weight_manifest::ShardPolicy;
    use hipfire_runtime::weight_store::test_support;
    use hipfire_runtime::weight_store::{
        WeightLoadTransaction, WeightOrigin, WeightProjection, WeightProjectionKind, WeightStore,
    };
    use std::path::Path;

    fn hfq_tensor(name: &str, shape: &[u32], quant_type: u8, bytes: usize) -> HfqMemTensor {
        HfqMemTensor {
            name: name.into(),
            quant_type,
            shape: shape.to_vec(),
            group_size: 0,
            data: vec![0; bytes],
        }
    }

    fn f32_hfq_tensor(name: &str, shape: &[u32], malformed: bool) -> HfqMemTensor {
        let elements = shape.iter().map(|&dim| dim as usize).product::<usize>();
        let data = if malformed {
            vec![0; 4]
        } else {
            (0..elements)
                .flat_map(|value| ((value as f32) + 1.0).to_le_bytes())
                .collect()
        };
        HfqMemTensor {
            name: name.into(),
            quant_type: 2,
            shape: shape.to_vec(),
            group_size: 0,
            data,
        }
    }
    fn f16_hfq_tensor(name: &str, shape: &[u32]) -> HfqMemTensor {
        let elements = shape.iter().map(|&dim| dim as usize).product::<usize>();
        HfqMemTensor {
            name: name.into(),
            quant_type: 1,
            shape: shape.to_vec(),
            group_size: 0,
            data: (0..elements)
                .flat_map(|index| {
                    let bits = if index % 2 == 0 { 0x3c00u16 } else { 0x3800u16 };
                    bits.to_le_bytes()
                })
                .collect(),
        }
    }

    /// Owned synthetic HFQ fixture writer: each call owns a
    /// `tempfile::NamedTempFile` (unique path per call, file removed on drop —
    /// the same ownership pattern as the `tempfile::tempdir` fixtures in
    /// `hipfire-runtime`), so concurrent tests can never share or delete each
    /// other's fixture. Hold the returned owner across every open/read —
    /// including the legacy reopen — and never unlink its path by hand.
    fn fixture_file(
        metadata: &str,
        tensors: &[HfqMemTensor],
    ) -> (tempfile::NamedTempFile, HfqFile) {
        let fixture = tempfile::NamedTempFile::new().expect("unique HFQ fixture file");
        write_hfqm_package_mem(fixture.path(), 0, metadata, tensors).expect("write HFQ fixture");
        let hfq = HfqFile::open(fixture.path()).expect("open HFQ fixture");
        (fixture, hfq)
    }

    fn fixture_hfq(
        with_awq_sidecar: bool,
        with_q_proj_bias: bool,
        malformed_output_norm: bool,
        separate_lm_head: bool,
    ) -> (tempfile::NamedTempFile, HfqFile) {
        fixture_hfq_with_lm_head(
            with_awq_sidecar,
            with_q_proj_bias,
            malformed_output_norm,
            separate_lm_head.then_some("lm_head.weight"),
        )
    }

    fn fixture_hfq_with_lm_head(
        with_awq_sidecar: bool,
        with_q_proj_bias: bool,
        malformed_output_norm: bool,
        lm_head_name: Option<&str>,
    ) -> (tempfile::NamedTempFile, HfqFile) {
        let mut tensors = vec![
            f32_hfq_tensor("model.embed_tokens.weight", &[2, 32], false),
            f32_hfq_tensor("model.norm.weight", &[32], false),
            f16_hfq_tensor("model.layers.0.self_attn.q_proj.weight", &[32, 32]),
            f16_hfq_tensor("model.layers.0.self_attn.k_proj.weight", &[32, 32]),
            f16_hfq_tensor("model.layers.0.self_attn.v_proj.weight", &[32, 32]),
            f16_hfq_tensor("model.layers.0.self_attn.o_proj.weight", &[32, 32]),
            f16_hfq_tensor("model.layers.0.mlp.gate_proj.weight", &[64, 32]),
            f16_hfq_tensor("model.layers.0.mlp.up_proj.weight", &[64, 32]),
            f16_hfq_tensor("model.layers.0.mlp.down_proj.weight", &[32, 64]),
            f32_hfq_tensor("model.layers.0.input_layernorm.weight", &[32], false),
            f32_hfq_tensor(
                "model.layers.0.post_attention_layernorm.weight",
                &[32],
                false,
            ),
        ];
        if malformed_output_norm {
            tensors[1] = f32_hfq_tensor("model.norm.weight", &[32], true);
        }
        if with_awq_sidecar {
            tensors.push(hfq_tensor(
                "model.layers.0.self_attn.q_proj.awq_scale.weight",
                &[32],
                1,
                32 * 2,
            ));
        }
        if with_q_proj_bias {
            tensors.push(hfq_tensor(
                "model.layers.0.self_attn.q_proj.bias",
                &[32],
                1,
                32 * 2,
            ));
        }
        if let Some(lm_head_name) = lm_head_name {
            tensors.push(f32_hfq_tensor(lm_head_name, &[2, 32], false));
        }
        let metadata = r#"{
            "config": {
                "model_type": "llama",
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "intermediate_size": 64,
                "vocab_size": 2,
                "head_dim": 32,
                "rms_norm_eps": 0.00001,
                "max_position_embeddings": 8,
                "rope_theta": 10000.0
            }
        }"#;
        fixture_file(metadata, &tensors)
    }

    /// Constant-valued MQ4G256 trunk (quant_type 13) for AWQ math tests.
    /// Layout transcribed from `kernels/src/gemv_mq4g256.hip`: 136 B per
    /// 256-group row — `[0..4)` f32 scale, `[4..8)` f32 zero, `[8..136)`
    /// 128 B nibbles low-first — decoded as `w = scale * nibble + zero`.
    /// Every weight decodes to `scale * nibble + zero`, so the trunk is an
    /// exact known constant, not an opaque blob.
    fn mq4g256_const_tensor(
        name: &str,
        m: usize,
        k: usize,
        scale: f32,
        nibble: u8,
    ) -> HfqMemTensor {
        assert_eq!(k % 256, 0, "MQ4G256 fixture geometry needs K % 256 == 0");
        assert!(nibble < 16, "one nibble per weight");
        let groups = k / 256;
        let byte = nibble | (nibble << 4);
        let mut data = Vec::with_capacity(m * groups * 136);
        for _ in 0..m * groups {
            data.extend_from_slice(&scale.to_le_bytes());
            data.extend_from_slice(&0f32.to_le_bytes());
            data.extend_from_slice(&vec![byte; 128]);
        }
        HfqMemTensor {
            name: name.into(),
            quant_type: 13,
            shape: vec![m as u32, k as u32],
            group_size: 0,
            data,
        }
    }

    /// Synthetic AWQ fixture on a *supported* quantized path: the o_proj trunk
    /// is MQ4G256 (quant_type 13, in `DType::supports_awq_sidecar`) with the
    /// same 1D-F16 length-K sidecar a real quantizer emits. The legacy loader
    /// uploads MQ4 verbatim and attaches the sidecar; an F16 trunk would be
    /// host-widened to F32, which is outside the allow-list, so the sidecar
    /// would attach to nothing and the test would prove no AWQ math path.
    ///
    /// Valid kernel geometry: K = 256 satisfies the FWHT rotation granularity
    /// the AWQ input-rotate path assumes. Nontrivial scales: the trunk is the
    /// constant 1.0 (as a pre-scaled `(W·s)` stand-in) and the sidecar holds
    /// caller-chosen non-unit F16. A unit sidecar would only prove loader
    /// neutrality, not that the divide shapes numerics.
    ///
    /// `o_sidecar`: uniform F16 scale replicated across the o_proj sidecar
    /// (attachment coverage on a mid-block projection). `lm_scales`: when
    /// `Some`, the separate lm_head is itself a constant-1.0 MQ4 trunk with
    /// exactly these K per-channel F16 scales — the post-`output_norm` divide
    /// oracle (`lm_head` has no normalization downstream, so a uniform
    /// 2.0-vs-4.0 pair must forward at an exact 2:1 logit ratio). When
    /// `None`, the head stays F16 (distinct-head loader coverage).
    fn fixture_awq_mq4_hfq(
        o_sidecar: Option<u16>,
        lm_scales: Option<Vec<u16>>,
    ) -> (tempfile::NamedTempFile, HfqFile) {
        const K: usize = 256;
        let mut tensors = vec![
            f32_hfq_tensor("model.embed_tokens.weight", &[2, 256], false),
            f32_hfq_tensor("model.norm.weight", &[256], false),
            f16_hfq_tensor("model.layers.0.self_attn.q_proj.weight", &[256, 256]),
            f16_hfq_tensor("model.layers.0.self_attn.k_proj.weight", &[256, 256]),
            f16_hfq_tensor("model.layers.0.self_attn.v_proj.weight", &[256, 256]),
            // Constant 1.0 trunk: scale 0.5, nibble 2, zero 0.0. The sidecar
            // sits here (not q_proj) because single-token decode attends
            // over one key, making q mathematically irrelevant, while o
            // shapes every output token.
            mq4g256_const_tensor("model.layers.0.self_attn.o_proj.weight", 256, 256, 0.5, 2),
            f16_hfq_tensor("model.layers.0.mlp.gate_proj.weight", &[256, 256]),
            f16_hfq_tensor("model.layers.0.mlp.up_proj.weight", &[256, 256]),
            f16_hfq_tensor("model.layers.0.mlp.down_proj.weight", &[256, 256]),
            f32_hfq_tensor("model.layers.0.input_layernorm.weight", &[256], false),
            f32_hfq_tensor(
                "model.layers.0.post_attention_layernorm.weight",
                &[256],
                false,
            ),
        ];
        if let Some(bits) = o_sidecar {
            tensors.push(HfqMemTensor {
                name: "model.layers.0.self_attn.o_proj.awq_scale.weight".into(),
                quant_type: 1,
                shape: vec![K as u32],
                group_size: 0,
                // Caller-chosen F16 scale replicated across K.
                data: (0..K).flat_map(|_| bits.to_le_bytes()).collect(),
            });
        }
        match lm_scales {
            // Quantized post-norm head: the same constant-1.0 MQ4 trunk as
            // o_proj, so the per-channel sidecar divide is the only
            // difference between same-route forwards. Loads through
            // `hfq::load_weight_tensor` (raw codec passthrough) with the
            // centralized sidecar attach, exactly like o_proj.
            Some(scales) => {
                assert_eq!(scales.len(), K, "lm_head sidecar must cover K channels");
                tensors.push(mq4g256_const_tensor("lm_head.weight", 2, 256, 0.5, 2));
                tensors.push(HfqMemTensor {
                    name: "lm_head.awq_scale.weight".into(),
                    quant_type: 1,
                    shape: vec![K as u32],
                    group_size: 0,
                    data: scales.iter().flat_map(|bits| bits.to_le_bytes()).collect(),
                });
            }
            // F16, not F32: a separate lm_head loads through
            // `hfq::load_weight_tensor`, which host-decodes qt1 and passes raw
            // codecs through but has no qt2 arm. The supported F16 separate head
            // keeps distinct-head coverage on a loader-supported dtype.
            None => tensors.push(f16_hfq_tensor("lm_head.weight", &[2, 256])),
        }
        let metadata = r#"{
            "config": {
                "model_type": "llama",
                "hidden_size": 256,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "intermediate_size": 256,
                "vocab_size": 2,
                "head_dim": 256,
                "rms_norm_eps": 0.00001,
                "max_position_embeddings": 32,
                "rope_theta": 10000.0
            }
        }"#;
        fixture_file(metadata, &tensors)
    }

    fn load_ctx<'a>(
        path: &'a Path,
        gpu: &'a mut rdna_compute::Gpu,
        cask: &'a CaskConfig,
    ) -> LoadCtx<'a> {
        LoadCtx {
            path: path.to_str().expect("fixture path is UTF-8"),
            max_seq: 8,
            deepseek4_compute_placement: Default::default(),
            deepseek4_experts_per_token: None,
            draft_path: None,
            kv_mode_override: Some("q8"),
            kv_backend: KvBackend::Contiguous,
            kv_adaptive_override: None,
            state_quant_override: None,
            vision_path: None,
            cask,
            pp: 1,
            spec: SpecLoadCfg::default(),
            gpu,
            gemma4_drafter_path: None,
            gemma4_draft_len: 3,
        }
    }

    fn config() -> LlamaConfig {
        LlamaConfig {
            arch: ModelArch::Llama,
            dim: 4,
            hidden_dim: 8,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            vocab_size: 8,
            head_dim: 4,
            norm_eps: 1e-5,
            max_seq_len: 32,
            rope_freq_base: 10_000.0,
            bos_token: 1,
            eos_token: 2,
            has_qk_norm: false,
        }
    }

    fn alias_projection() -> WeightProjection {
        WeightProjection {
            kind: WeightProjectionKind::Static,
            axis: None,
            rank: 0,
            world_size: 1,
            logical_shape: vec![1],
            dtype: DType::F32,
        }
    }

    #[test]
    fn single_plan_covers_every_typed_llama_handle() {
        let (mesh, plan) = plan_single(&config(), true).unwrap();
        let manifest = Llama::weight_manifest(&config());
        assert_eq!(mesh.n_devices(), 1);
        assert_eq!(plan.weights.len(), 12);
        assert_eq!(plan.state.len(), 1);
        assert!(plan
            .collective_schedule
            .iter()
            .any(|entry| entry.name == "wo"));
        assert!(manifest[0].dtype_constraint.accepts(DType::HFQ4G256));
        assert!(manifest[1].dtype_constraint.accepts(DType::MQ4G256));
        assert!(manifest[9].dtype_constraint.accepts(DType::F32));
        assert!(!manifest[9].dtype_constraint.accepts(DType::F16));
    }

    /// Exported-manifest output placement on a PP mesh: the final norm and
    /// both the separate and tied language head must resolve to the final
    /// pipeline stage, while the embedding stays on stage zero. The Single
    /// production pilot maps every stage to device 0 and cannot see this.
    #[test]
    fn output_tensors_pin_to_final_pipeline_stage() {
        use hipfire_runtime::device_mesh::DimKind;
        use hipfire_runtime::weight_manifest::{placement_devices, PinTarget, PlacementHint};
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2)])
            .expect("two-stage pipeline mesh construction cannot overflow");
        let manifest = Llama::weight_manifest(&config());
        let devices = |name: &str| {
            let entry = manifest
                .iter()
                .find(|entry| entry.name == name && entry.layer.is_none())
                .expect("model-scope entry exists");
            placement_devices(entry, &mesh, config().n_layers)
        };
        assert_eq!(devices("token_embd"), vec![0]);
        assert_eq!(devices("output_norm"), vec![1]);
        assert_eq!(devices("lm_head"), vec![1]);
        let tied = Llama::weight_manifest_for_hfq(&config(), false);
        let head = tied.last().expect("manifest has lm_head");
        assert_eq!(head.placement, PlacementHint::Pin(PinTarget::Output));
        assert_eq!(
            placement_devices(head, &mesh, config().n_layers),
            vec![1],
            "tying lm_head must not move it to stage 0"
        );
    }

    #[test]
    fn typed_assembly_rolls_back_when_a_cell_is_not_resident() {
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let origin = WeightOrigin::from_parts(mesh.epoch(), 0, 0);
        let mut store = WeightStore::with_origin(origin);
        for name in ["token_embd", "output_norm", "lm_head"] {
            store
                .stage_alias(name, None, 0, "source", alias_projection())
                .unwrap();
        }
        let mut transaction = WeightLoadTransaction::new(store);
        let error = match assemble_llama_weights(
            &LlamaConfig {
                n_layers: 0,
                ..config()
            },
            &mut transaction,
        ) {
            Ok(_) => panic!("alias unexpectedly assembled as typed weights"),
            Err(error) => error,
        };
        assert!(error.contains("alias"));
        assert_eq!(transaction.len(), 3);
        assert!(transaction.contains("token_embd", None, 0));
        assert!(transaction.projection("lm_head", None, 0).is_some());
    }

    #[test]
    fn hfq_float_widening_matches_legacy_f32_representation() {
        let f16_one = [0x00, 0x3c, 0x00, 0xc0];
        let actual = f32_bytes_from_hfq(1, &f16_one, "test").unwrap();
        let expected = [1.0f32, -2.0f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn manifest_constraints_admit_every_pilot_representation() {
        let manifest = Llama::weight_manifest(&config());
        assert!(manifest[0].dtype_constraint.accepts(DType::HFQ4G256));
        assert!(manifest[1].dtype_constraint.accepts(DType::MQ4G256));
        assert!(manifest[9].dtype_constraint.accepts(DType::F32));
        assert!(!manifest[9].dtype_constraint.accepts(DType::F16));
    }
    #[test]
    fn physical_cap_remains_separate_from_configured_max_seq() {
        let dims = llama_kv_dims(&config(), 32_768, Some(4_096));
        assert_eq!(dims.max_seq, 32_768);
        assert_eq!(dims.physical_cap, Some(4_096));
    }

    #[test]
    fn missing_lm_head_manifest_declares_a_tied_embedding_alias() {
        let manifest = Llama::weight_manifest_for_hfq(&config(), false);
        let token = &manifest[0];
        let output = manifest.last().expect("manifest has lm_head");
        assert!(matches!(
            output.policy,
            ShardPolicy::Tied { ref source } if source == "token_embd"
        ));
        assert!(token
            .dtype_constraint
            .same_source_set(&output.dtype_constraint));
    }

    #[test]
    fn production_hfq_single_route_aliases_missing_lm_head_without_second_allocation() {
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            return;
        };
        let (fixture, hfq) = fixture_hfq(false, false, false, false);
        let cask = CaskConfig::default();
        let mut ctx = load_ctx(fixture.path(), &mut gpu, &cask);
        let bundle = load_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("load plain HFQ fixture");
        drop(ctx);
        assert!(bundle.weights.lm_head_aliases_embd);
        assert_eq!(
            bundle.weights.output.buf.buf.as_ptr(),
            bundle.weights.token_embd.buf.as_ptr()
        );
        // Attached provenance: the tied alias edge survives assembly, and the
        // descriptors match the bundle-owned KV/scratch they describe. The
        // store owns no allocation; these records are descriptive only.
        let attached = bundle.weight_store.as_ref().expect("attached store");
        assert_eq!(
            attached.alias_source("lm_head", None, 0),
            Some("token_embd")
        );
        assert_eq!(attached.attachments().kv_n_kv_heads, bundle.kv.n_kv_heads);
        assert_eq!(attached.attachments().kv_max_seq, bundle.kv.max_seq);
        assert_eq!(
            attached.attachments().scratch_logits_shape,
            bundle.scratch.logits.shape
        );
        Box::new(bundle).free_gpu(&mut gpu);
    }

    #[test]
    fn production_awq_sidecar_selects_legacy_loader() {
        let (_fixture, hfq) = fixture_hfq(true, false, false, false);
        assert_eq!(classify_hfq_route(&hfq), HfqLoadRoute::LegacyAwq);
    }

    #[test]
    fn alternate_explicit_lm_head_names_are_not_tied() {
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            return;
        };
        for name in &HFQ_LM_HEAD_NAMES[1..] {
            let (fixture, hfq) = fixture_hfq_with_lm_head(false, false, false, Some(name));
            assert!(hfq_has_separate_lm_head(&hfq));
            let cask = CaskConfig::default();
            let mut ctx = load_ctx(fixture.path(), &mut gpu, &cask);
            let bundle =
                load_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("load explicit lm_head");
            drop(ctx);
            assert!(!bundle.weights.lm_head_aliases_embd);
            Box::new(bundle).free_gpu(&mut gpu);
        }
    }

    #[test]
    fn production_biased_hfq_is_rejected_before_manifest_upload() {
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            return;
        };
        let (fixture, hfq) = fixture_hfq(false, true, false, false);
        let cask = CaskConfig::default();
        let mut ctx = load_ctx(fixture.path(), &mut gpu, &cask);
        let error = match load_bundle(ModelSource::Hfq(hfq), &mut ctx) {
            Ok(_) => panic!("biased HFQ unexpectedly loaded"),
            Err(error) => error,
        };
        drop(ctx);
        assert!(error.contains("q_proj.bias"));
        assert!(error.contains("refusing to load Qwen2"));
    }

    #[test]
    fn production_post_resident_failure_reclaims_every_uploaded_allocation() {
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            return;
        };
        let (fixture, hfq) = fixture_hfq(false, false, false, false);
        test_support::reset();
        test_support::arm_fail_after_upload(1);
        let cask = CaskConfig::default();
        let mut ctx = load_ctx(fixture.path(), &mut gpu, &cask);
        let error = match load_bundle(ModelSource::Hfq(hfq), &mut ctx) {
            Ok(_) => panic!("post-upload fault unexpectedly succeeded"),
            Err(error) => error,
        };
        drop(ctx);
        test_support::clear_faults();
        assert!(error.contains("test fault injected after resident upload"));
        let allocations = test_support::resident_allocations();
        assert!(allocations > 0, "fault must follow a resident upload");
        assert_eq!(
            allocations,
            test_support::resident_releases(),
            "every resident allocation must be reclaimed on load failure"
        );
    }

    #[test]
    fn production_manifest_matches_legacy_forward_logits() {
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            return;
        };
        let (fixture, hfq) = fixture_hfq(false, false, false, false);
        let cask = CaskConfig::default();
        let mut ctx = load_ctx(fixture.path(), &mut gpu, &cask);
        let mut bundle =
            load_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("load plain HFQ fixture");
        drop(ctx);

        let manifest_logits = {
            forward_scratch_embed(
                &mut gpu,
                &bundle.weights,
                &bundle.config,
                1,
                0,
                &bundle.scratch,
            )
            .expect("manifest embedding forward");
            forward_scratch_compute(
                &mut gpu,
                &bundle.weights,
                &bundle.config,
                0,
                &mut bundle.kv,
                &bundle.scratch,
            )
            .expect("manifest model forward");
            gpu.download_f32(&bundle.scratch.logits)
                .expect("download manifest logits")
        };
        Box::new(bundle).free_gpu(&mut gpu);

        let hfq = HfqFile::open(fixture.path()).expect("reopen HFQ fixture");
        let config = <Llama as Architecture>::config_from_hfq(&hfq).expect("fixture config");
        let legacy = hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, &mut gpu)
            .expect("load legacy HFQ fixture");
        let scratch = ForwardScratch::new_with_max_seq(&mut gpu, &config, 8)
            .expect("allocate legacy forward scratch");
        let dims = llama_kv_dims(&config, 8, None);
        let mut kv =
            <KvCache as KvCacheExt>::from_mode(KvMode::Q8, KvTarget::Single(&mut gpu), &dims)
                .expect("allocate legacy KV cache");
        forward_scratch_embed(&mut gpu, &legacy, &config, 1, 0, &scratch)
            .expect("legacy embedding forward");
        forward_scratch_compute(&mut gpu, &legacy, &config, 0, &mut kv, &scratch)
            .expect("legacy model forward");
        let legacy_logits = gpu
            .download_f32(&scratch.logits)
            .expect("download legacy logits");
        scratch.free_gpu(&mut gpu);
        let _ = kv.free_gpu(&mut gpu);
        legacy.free_gpu(&mut gpu);

        assert_eq!(manifest_logits.len(), legacy_logits.len());
        for (index, (manifest, legacy)) in manifest_logits.iter().zip(&legacy_logits).enumerate() {
            assert!(
                (manifest - legacy).abs() <= 1e-5,
                "logit mismatch at index {index}: manifest={manifest} legacy={legacy}"
            );
        }
    }

    #[test]
    fn physical_cap_is_honored_by_upstream_kv_constructor() {
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            return;
        };
        let dims = KvDims {
            layers: KvLayers::Flat(1),
            n_kv_heads: 1,
            head_dim: 32,
            max_seq: 8,
            physical_cap: Some(4),
        };
        let cache =
            <KvCache as KvCacheExt>::from_mode(KvMode::Q8, KvTarget::Single(&mut gpu), &dims)
                .expect("upstream Q8 constructor");
        assert_eq!(cache.max_seq, 8);
        assert_eq!(cache.physical_cap, 4);
        let _ = cache.free_gpu(&mut gpu);
    }

    /// #666 G3 pinned-fixture parity oracle.
    ///
    /// Runs on the registry fixture `qwen3:0.6b` (plain LLaMA-family HFQ,
    /// canonical local file `~/.hipfire/models/qwen3-0.6b.hf4`). This is a
    /// distinct acceptance fixture: the historic `qwen3-0.6b-llama.mq4` pin
    /// is unavailable, and no equivalence with it is claimed. The path is
    /// taken from `G3_FIXTURE` when set, else the canonical
    /// `~/.hipfire/models` location. Skips silently when the file or a GPU is
    /// absent (no-GPU / no-fixture batteries stay green); fails loudly on a
    /// size or route-class mismatch so a substituted artifact cannot pass as
    /// the pinned fixture.
    ///
    /// Two routes are loaded from equivalent cloned state:
    ///   * production manifest route — `load_bundle` classifies this plain
    ///     file as `ManifestPlainLlama` and publishes through manifest
    ///     planning, transactional fulfillment, and typed assembly;
    ///   * validation-only reference — the legacy loader entry the manifest
    ///     route replaces (`hfq::load_weights_hfq` plus the same scratch and
    ///     KV constructors).
    ///
    /// Both then decode the same committed prompt greedily (argmax), and at
    /// every committed position the oracle records and asserts: token IDs,
    /// logits (max absolute difference), KV geometry / byte extents, the
    /// position counter, alias identity, and route identity. The evidence
    /// block is printed for the evidence run.
    #[test]
    fn pinned_fixture_manifest_legacy_parity_oracle() {
        let _gpu_evidence_guard = GPU_EVIDENCE_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let fixture = std::env::var("G3_FIXTURE").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            format!("{home}/.hipfire/models/qwen3-0.6b.hf4")
        });
        // Fixture lock: the registry artifact identity, verified by content
        // hash. Size alone cannot detect a same-length substitution, and the
        // printed digest below is the freshly computed file hash, never a
        // hardcoded string. Route classification below additionally refuses
        // non-plain / mis-tagged artifacts.
        const PINNED_SIZE: u64 = 436_006_912;
        const PINNED_MD5: &str = "0d1055bf8f9492df2e2374d0e8bf787f";
        let Ok(meta) = std::fs::metadata(&fixture) else {
            eprintln!("g3-oracle: fixture absent ({fixture}); skipping");
            return;
        };
        assert_eq!(
            meta.len(),
            PINNED_SIZE,
            "g3-oracle: fixture size mismatch — not the pinned qwen3:0.6b artifact (md5 {PINNED_MD5})"
        );
        let actual_md5 = fixture_md5_hex(&fixture).expect("hash pinned fixture");
        assert_eq!(
            actual_md5, PINNED_MD5,
            "g3-oracle: fixture content mismatch — not the pinned qwen3:0.6b artifact"
        );
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            eprintln!("g3-oracle: no GPU; skipping");
            return;
        };
        let prompt = "The capital of France is located in";
        let max_seq = 64usize;
        eprintln!(
            "g3-oracle: fixture={fixture} size={} md5={actual_md5}",
            meta.len()
        );
        eprintln!("g3-oracle: prompt={prompt:?} (prompt md5 recorded by the evidence run)");

        let hfq = HfqFile::open(std::path::Path::new(&fixture)).expect("open pinned fixture");
        assert_eq!(
            classify_hfq_route(&hfq),
            HfqLoadRoute::ManifestPlainLlama,
            "pinned fixture must take the production manifest route"
        );
        assert!(
            !hfq.has_awq_sidecars(),
            "pinned fixture must be a plain (sidecar-free) LLaMA-family HFQ"
        );
        let tokenizer =
            hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                .expect("pinned fixture tokenizer");
        let prompt_tokens = tokenizer.encode(prompt);
        eprintln!(
            "g3-oracle: prompt tokens = {prompt_tokens:?} ({} incl. BOS)",
            prompt_tokens.len()
        );
        let has_separate_lm_head = hfq_has_separate_lm_head(&hfq);
        eprintln!("g3-oracle: separate lm_head in fixture = {has_separate_lm_head}");

        let cask = CaskConfig::default();
        let mut ctx = load_ctx(std::path::Path::new(&fixture), &mut gpu, &cask);
        ctx.max_seq = max_seq;
        let mut bundle = load_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("production load");
        assert_eq!(
            ctx.kv_mode_override,
            Some("q8"),
            "oracle runs both routes in the same Q8 KV mode"
        );
        assert!(bundle.weight_store.is_some(), "production store attached");
        drop(ctx);

        // Validation-only reference: the legacy loader entry, same scratch/KV.
        let hfq = HfqFile::open(std::path::Path::new(&fixture)).expect("reopen pinned fixture");
        let config = <Llama as Architecture>::config_from_hfq(&hfq).expect("reference config");
        let legacy = hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, &mut gpu)
            .expect("legacy reference load");
        let scratch = ForwardScratch::new_with_max_seq(&mut gpu, &config, max_seq)
            .expect("reference forward scratch");
        let dims = llama_kv_dims(&config, max_seq, None);
        let mut legacy_kv =
            <KvCache as KvCacheExt>::from_mode(KvMode::Q8, KvTarget::Single(&mut gpu), &dims)
                .expect("reference KV cache");

        eprintln!(
            "g3-oracle: route identity — production=ManifestPlainLlama, reference=legacy load_weights_hfq"
        );

        // Alias identity parity.
        assert_eq!(
            bundle.weights.lm_head_aliases_embd, legacy.lm_head_aliases_embd,
            "alias identity must match between routes"
        );
        eprintln!(
            "g3-oracle: lm_head aliases embed_tokens = {} (both routes)",
            legacy.lm_head_aliases_embd
        );

        // KV geometry parity (mode flags, dims, byte extents per layer).
        {
            let pkv = &bundle.kv;
            eprintln!(
                "g3-oracle: kv geometry — prod q8={} qint8={} kv_dim={} max_seq={} cap={} n_heads={} head_dim={}",
                pkv.quant_q8, pkv.quant_int8, pkv.kv_dim, pkv.max_seq, pkv.physical_cap,
                pkv.n_kv_heads, pkv.head_dim
            );
            assert_eq!(pkv.quant_q8, legacy_kv.quant_q8);
            assert_eq!(pkv.quant_int8, legacy_kv.quant_int8);
            assert_eq!(pkv.kv_dim, legacy_kv.kv_dim);
            assert_eq!(pkv.max_seq, legacy_kv.max_seq);
            assert_eq!(pkv.physical_cap, legacy_kv.physical_cap);
            assert_eq!(pkv.n_kv_heads, legacy_kv.n_kv_heads);
            assert_eq!(pkv.head_dim, legacy_kv.head_dim);
            assert_eq!(pkv.k_gpu.len(), legacy_kv.k_gpu.len());
            for layer in 0..pkv.k_gpu.len() {
                assert_eq!(
                    pkv.k_gpu[layer].byte_size(),
                    legacy_kv.k_gpu[layer].byte_size(),
                    "layer {layer} K byte extent parity"
                );
                assert_eq!(
                    pkv.v_gpu[layer].byte_size(),
                    legacy_kv.v_gpu[layer].byte_size(),
                    "layer {layer} V byte extent parity"
                );
            }
        }

        // Greedy decode in lockstep; compare at every committed position.
        let mut next_token: u32 = 0;
        let mut worst_logit_diff: f32 = 0.0;
        let mut tokens: Vec<u32> = Vec::new();
        let generated = 12usize;
        let total = prompt_tokens.len() + generated;
        assert!(total <= max_seq, "position budget vs KV max_seq");
        for pos in 0..total {
            let token = if pos < prompt_tokens.len() {
                prompt_tokens[pos]
            } else {
                next_token
            };
            forward_scratch_embed(
                &mut gpu,
                &bundle.weights,
                &bundle.config,
                token,
                pos,
                &bundle.scratch,
            )
            .expect("production embed");
            forward_scratch_compute(
                &mut gpu,
                &bundle.weights,
                &bundle.config,
                0,
                &mut bundle.kv,
                &bundle.scratch,
            )
            .expect("production compute");
            let prod_logits = gpu
                .download_f32(&bundle.scratch.logits)
                .expect("production logits");
            forward_scratch_embed(&mut gpu, &legacy, &config, token, pos, &scratch)
                .expect("reference embed");
            forward_scratch_compute(&mut gpu, &legacy, &config, 0, &mut legacy_kv, &scratch)
                .expect("reference compute");
            let legacy_logits = gpu.download_f32(&scratch.logits).expect("reference logits");
            assert_eq!(
                prod_logits.len(),
                legacy_logits.len(),
                "logit width parity at position {pos}"
            );
            let mut diff: f32 = 0.0;
            for (p, l) in prod_logits.iter().zip(&legacy_logits) {
                diff = diff.max((p - l).abs());
            }
            worst_logit_diff = worst_logit_diff.max(diff);
            assert!(
                diff <= 1e-5,
                "logit mismatch at committed position {pos}: max abs diff {diff}"
            );
            let prod_choice = argmax_index(&prod_logits) as u32;
            let legacy_choice = argmax_index(&legacy_logits) as u32;
            assert_eq!(
                prod_choice, legacy_choice,
                "token-id mismatch at committed position {pos}"
            );
            next_token = prod_choice;
            tokens.push(token);
            eprintln!(
                "g3-oracle: pos {pos:>2} token {token:>6} max-logit-diff {diff:.3e} (choice {prod_choice})"
            );
        }

        // End-state KV payload parity on layer 0 (full written extent).
        let mut prod_k = vec![0u8; bundle.kv.k_gpu[0].byte_size()];
        let mut ref_k = vec![0u8; legacy_kv.k_gpu[0].byte_size()];
        gpu.hip
            .memcpy_dtoh(&mut prod_k, &bundle.kv.k_gpu[0].buf)
            .expect("download prod K");
        gpu.hip
            .memcpy_dtoh(&mut ref_k, &legacy_kv.k_gpu[0].buf)
            .expect("download ref K");
        let k_diffs = prod_k.iter().zip(&ref_k).filter(|(a, b)| a != b).count();
        eprintln!(
            "g3-oracle: layer-0 K payload — {} bytes compared, {k_diffs} byte diffs",
            prod_k.len()
        );
        assert_eq!(prod_k, ref_k, "layer-0 K payload must be byte-identical");
        let mut prod_v = vec![0u8; bundle.kv.v_gpu[0].byte_size()];
        let mut ref_v = vec![0u8; legacy_kv.v_gpu[0].byte_size()];
        gpu.hip
            .memcpy_dtoh(&mut prod_v, &bundle.kv.v_gpu[0].buf)
            .expect("download prod V");
        gpu.hip
            .memcpy_dtoh(&mut ref_v, &legacy_kv.v_gpu[0].buf)
            .expect("download ref V");
        let v_diffs = prod_v.iter().zip(&ref_v).filter(|(a, b)| a != b).count();
        eprintln!(
            "g3-oracle: layer-0 V payload — {} bytes compared, {v_diffs} byte diffs",
            prod_v.len()
        );
        assert_eq!(prod_v, ref_v, "layer-0 V payload must be byte-identical");

        eprintln!(
            "g3-oracle: PASS — {total} committed positions, worst logit diff {worst_logit_diff:.3e}, tokens {tokens:?}"
        );
        Box::new(bundle).free_gpu(&mut gpu);
        scratch.free_gpu(&mut gpu);
        let _ = legacy_kv.free_gpu(&mut gpu);
        legacy.free_gpu(&mut gpu);
    }

    /// #666 G3 pinned-fixture lifecycle evidence.
    ///
    /// On the same tracker fixture as the parity oracle: production load
    /// through the manifest route, decode, existing-reset smoke (decode
    /// again after `reset_session_state` with identical output), unload via
    /// the sole consuming owner (`ArchModel::free_gpu`), immediate reload
    /// with identical decode, a deterministic post-upload fault whose
    /// rollback returns every resident store allocation (store accounting:
    /// allocations == releases on the failed path), and an immediate retry
    /// that decodes identically. Skips cleanly when the fixture or a GPU is
    /// absent.
    #[test]
    fn pinned_fixture_lifecycle_fault_retry_reload() {
        let _gpu_evidence_guard = GPU_EVIDENCE_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(fixture) = pinned_fixture_path() else {
            return;
        };
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            eprintln!("g3-lifecycle: no GPU; skipping");
            return;
        };
        let prompt = "The capital of France is located in";
        let max_seq = 64usize;
        let hfq = HfqFile::open(std::path::Path::new(&fixture)).expect("open pinned fixture");
        let tokenizer =
            hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                .expect("pinned fixture tokenizer");
        let prompt_tokens = tokenizer.encode(prompt);
        let cask = CaskConfig::default();

        // First production load + decode (warms store resident accounting).
        test_support::reset();
        let mut ctx = load_ctx(std::path::Path::new(&fixture), &mut gpu, &cask);
        ctx.max_seq = max_seq;
        let hfq = HfqFile::open(std::path::Path::new(&fixture)).expect("reopen for first load");
        let mut bundle = load_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("production load");
        drop(ctx);
        let allocations = test_support::resident_allocations();
        assert!(
            allocations > 0,
            "production load must publish resident store allocations"
        );
        eprintln!("g3-lifecycle: warm-baseline resident allocations = {allocations}");
        let baseline = greedy_decode(
            &mut gpu,
            &bundle.weights,
            &bundle.config,
            &mut bundle.kv,
            &bundle.scratch,
            &prompt_tokens,
            8,
        );
        eprintln!("g3-lifecycle: first decode = {baseline:?}");

        // Existing-reset smoke: reset_session_state leaves the model reusable
        // and the next decode is byte-identical.
        hipfire_runtime::arch_model::ArchModel::reset_session_state(&mut bundle, &mut gpu)
            .expect("existing reset smoke");
        let after_reset = greedy_decode(
            &mut gpu,
            &bundle.weights,
            &bundle.config,
            &mut bundle.kv,
            &bundle.scratch,
            &prompt_tokens,
            8,
        );
        assert_eq!(after_reset, baseline, "reset must not change decode output");

        // Unload through the sole consuming owner (ArchModel::free_gpu drains
        // the attached store and frees weights/scratch/KV).
        Box::new(bundle).free_gpu(&mut gpu);
        eprintln!("g3-lifecycle: unloaded via ArchModel::free_gpu");

        // Immediate reload decodes identically.
        let mut ctx = load_ctx(std::path::Path::new(&fixture), &mut gpu, &cask);
        ctx.max_seq = max_seq;
        let hfq = HfqFile::open(std::path::Path::new(&fixture)).expect("reopen for reload");
        let mut bundle = load_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("immediate reload");
        drop(ctx);
        let after_reload = greedy_decode(
            &mut gpu,
            &bundle.weights,
            &bundle.config,
            &mut bundle.kv,
            &bundle.scratch,
            &prompt_tokens,
            8,
        );
        assert_eq!(after_reload, baseline, "immediate reload decode parity");

        // Deterministic post-upload fault on the next load: it must fail and
        // roll back every resident allocation (no legacy fallback). Store
        // accounting is reset so the failed path alone is measured:
        // allocations == releases after the rollback.
        test_support::reset();
        test_support::arm_fail_after_upload(1);
        let mut ctx = load_ctx(std::path::Path::new(&fixture), &mut gpu, &cask);
        ctx.max_seq = max_seq;
        let hfq = HfqFile::open(std::path::Path::new(&fixture)).expect("reopen for fault");
        let error = match load_bundle(ModelSource::Hfq(hfq), &mut ctx) {
            Ok(_) => panic!("post-upload fault unexpectedly succeeded"),
            Err(error) => error,
        };
        drop(ctx);
        test_support::clear_faults();
        assert!(
            error.contains("test fault injected after resident upload"),
            "unexpected error: {error}"
        );
        assert_eq!(
            test_support::resident_allocations(),
            test_support::resident_releases(),
            "fault rollback must return every resident allocation (zero-free)"
        );
        eprintln!("g3-lifecycle: deterministic fault rolled back — error {error:?}");

        // Immediate retry after the fault decodes identically.
        let mut ctx = load_ctx(std::path::Path::new(&fixture), &mut gpu, &cask);
        ctx.max_seq = max_seq;
        let hfq = HfqFile::open(std::path::Path::new(&fixture)).expect("reopen for retry");
        let mut bundle = load_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("immediate retry");
        drop(ctx);
        let after_retry = greedy_decode(
            &mut gpu,
            &bundle.weights,
            &bundle.config,
            &mut bundle.kv,
            &bundle.scratch,
            &prompt_tokens,
            8,
        );
        assert_eq!(after_retry, baseline, "immediate retry decode parity");
        Box::new(bundle).free_gpu(&mut gpu);
        eprintln!(
            "g3-lifecycle: PASS — load/reset/unload/reload/fault/retry all decode identically"
        );
    }

    /// AWQ-sidecar HFQ on a supported quantized trunk retains the legacy
    /// loader end to end on GPU: the source is classified `LegacyAwq`, no
    /// manifest store is attached, the MQ4G256 o_proj keeps its quantized
    /// dtype (not widened), and its AWQ scale sidecar is attached through the
    /// `DType::supports_awq_sidecar` gate. Forward runs finite nonzero
    /// logits. The numerical proof is a post-`output_norm` oracle: a
    /// quantized lm_head with a uniform 2.0-vs-4.0 sidecar pair must forward
    /// at an exact 2:1 logit ratio (the divide-then-linear chain is exactly
    /// linear and no normalization sits downstream of lm_head), plus a
    /// nonuniform sidecar that must differ from both uniform runs. A global
    /// sidecar on a pre-norm projection cannot serve as the oracle —
    /// RMSNorm erases it — so the o_proj pair only records attachment.
    /// Unload via the sole consuming owner followed by an immediate reload
    /// re-attaches the sidecar with bitwise-identical numerics.
    #[test]
    fn production_awq_sidecar_loads_and_decodes_on_gpu_through_legacy_route() {
        fn forward_logits(gpu: &mut rdna_compute::Gpu, bundle: &mut LlamaBundle) -> Vec<f32> {
            forward_scratch_embed(gpu, &bundle.weights, &bundle.config, 0, 0, &bundle.scratch)
                .expect("AWQ embed");
            forward_scratch_compute(
                gpu,
                &bundle.weights,
                &bundle.config,
                0,
                &mut bundle.kv,
                &bundle.scratch,
            )
            .expect("AWQ compute");
            gpu.download_f32(&bundle.scratch.logits)
                .expect("AWQ logits")
        }
        let _gpu_evidence_guard = GPU_EVIDENCE_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            return;
        };
        let (fixture, hfq) = fixture_awq_mq4_hfq(Some(0x4000), None);
        assert_eq!(classify_hfq_route(&hfq), HfqLoadRoute::LegacyAwq);
        assert!(hfq.has_awq_sidecars());
        let cask = CaskConfig::default();
        let mut ctx = load_ctx(fixture.path(), &mut gpu, &cask);
        let mut bundle = load_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("AWQ legacy GPU load");
        drop(ctx);
        assert!(
            bundle.weight_store.is_none(),
            "AWQ-sidecar sources must not attach a manifest store"
        );
        assert_eq!(
            bundle.weights.layers[0].wo.gpu_dtype,
            DType::MQ4G256,
            "supported AWQ trunk must keep its quantized dtype, not widen"
        );
        assert!(
            bundle.weights.layers[0].wo.awq_scale.is_some(),
            "AWQ scale sidecar must attach on the supported dtype path"
        );
        let awq_logits = forward_logits(&mut gpu, &mut bundle);
        assert_eq!(awq_logits.len(), 2, "fixture vocab width");
        assert!(
            awq_logits.iter().all(|value| value.is_finite()),
            "AWQ forward must produce finite logits, got {awq_logits:?}"
        );
        assert!(
            awq_logits.iter().any(|value| value.abs() > 1e-3),
            "AWQ forward must compute nonzero output, got {awq_logits:?}"
        );
        Box::new(bundle).free_gpu(&mut gpu);
        // Post-norm divide oracle: a quantized lm_head (constant-1.0 MQ4
        // trunk) with a uniform 2.0 sidecar vs a uniform 4.0 sidecar through
        // the same legacy route and kernels. The chain after `output_norm`
        // is divide-by-scale, FWHT, GEMV — exactly linear in 1/s with no
        // normalization downstream of lm_head — so the 4.0 logits must equal
        // the 2.0 logits halved, to fp tolerance. A bypassed sidecar would
        // forward bit-identically (deviation 1.0); the 1e-4 relative bound
        // discriminates a live divide from a bypass by four orders of
        // magnitude. A third alternating 2.0/4.0 sidecar must differ from
        // the uniform run beyond fp noise, proving per-channel application
        // rather than a global fudge factor.
        fn load_lm_head_pair(
            path: &Path,
            hfq: HfqFile,
            gpu: &mut rdna_compute::Gpu,
            cask: &CaskConfig,
        ) -> LlamaBundle {
            let mut ctx = load_ctx(path, gpu, cask);
            let bundle =
                load_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("AWQ lm_head legacy load");
            drop(ctx);
            bundle
        }
        const LM_K: usize = 256;
        let (lm2_fixture, lm2_hfq) = fixture_awq_mq4_hfq(None, Some(vec![0x4000; LM_K]));
        assert_eq!(classify_hfq_route(&lm2_hfq), HfqLoadRoute::LegacyAwq);
        let mut pair2 = load_lm_head_pair(lm2_fixture.path(), lm2_hfq, &mut gpu, &cask);
        assert_eq!(
            pair2.weights.output.gpu_dtype,
            DType::MQ4G256,
            "quantized lm_head must keep its MQ4 dtype, not widen"
        );
        assert!(
            pair2.weights.output.awq_scale.is_some(),
            "lm_head AWQ scale sidecar must attach on the supported dtype path"
        );
        let logits2 = forward_logits(&mut gpu, &mut pair2);
        assert_eq!(logits2.len(), 2, "fixture vocab width");
        assert!(
            logits2.iter().all(|value| value.is_finite()),
            "lm_head AWQ forward must produce finite logits, got {logits2:?}"
        );
        assert!(
            logits2.iter().any(|value| value.abs() > 1e-3),
            "lm_head AWQ forward must compute nonzero output, got {logits2:?}"
        );
        Box::new(pair2).free_gpu(&mut gpu);
        let (lm4_fixture, lm4_hfq) = fixture_awq_mq4_hfq(None, Some(vec![0x4400; LM_K]));
        let mut pair4 = load_lm_head_pair(lm4_fixture.path(), lm4_hfq, &mut gpu, &cask);
        assert!(pair4.weights.output.awq_scale.is_some());
        let logits4 = forward_logits(&mut gpu, &mut pair4);
        assert!(logits4.iter().all(|value| value.is_finite()));
        Box::new(pair4).free_gpu(&mut gpu);
        assert_eq!(logits2.len(), logits4.len(), "same trunk, same vocab width");
        for (index, (reference, halved)) in logits2.iter().zip(logits4.iter()).enumerate() {
            let deviation = (halved * 2.0 - reference).abs() / reference.abs().max(1e-6);
            assert!(
                deviation < 1e-4,
                "lm_head AWQ divide ratio broken at logit {index}: s2={reference} s4={halved} (expected {halved}*2 == {reference}); a bypassed sidecar gives deviation 1.0"
            );
        }
        eprintln!("awq-numerics: lm2={logits2:?} lm4={logits4:?} divide-ratio-2-holds");
        let mut mixed = vec![0u16; LM_K];
        for (index, slot) in mixed.iter_mut().enumerate() {
            *slot = if index % 2 == 0 { 0x4000 } else { 0x4400 };
        }
        let (mix_fixture, mix_hfq) = fixture_awq_mq4_hfq(None, Some(mixed));
        let mut pair_mix = load_lm_head_pair(mix_fixture.path(), mix_hfq, &mut gpu, &cask);
        let logits_mix = forward_logits(&mut gpu, &mut pair_mix);
        assert!(logits_mix.iter().all(|value| value.is_finite()));
        Box::new(pair_mix).free_gpu(&mut gpu);
        let peak = logits2
            .iter()
            .fold(0.0f32, |best, value| best.max(value.abs()));
        let shift = logits2
            .iter()
            .zip(logits_mix.iter())
            .fold(0.0f32, |best, (plain, varied)| {
                best.max((varied - plain).abs())
            });
        assert!(
            shift / peak.max(1e-6) > 1e-6,
            "nonuniform lm_head sidecar must reshape numerics per channel, got shift {shift} at peak {peak}"
        );
        eprintln!("awq-numerics: lm_mix={logits_mix:?} per-channel-shift {shift}");
        // Sidecar-free trunk through the manifest route: proves the same MQ4
        // trunk loads and forwards on the G3 production path. No numeric
        // comparison across routes is drawn (different kernels per route).
        let (plain_fixture, plain_hfq) = fixture_awq_mq4_hfq(None, None);
        let mut ctx = load_ctx(plain_fixture.path(), &mut gpu, &cask);
        let mut plain = load_bundle(ModelSource::Hfq(plain_hfq), &mut ctx).expect("plain MQ4 load");
        drop(ctx);
        let plain_logits = forward_logits(&mut gpu, &mut plain);
        assert!(plain_logits.iter().all(|value| value.is_finite()));
        Box::new(plain).free_gpu(&mut gpu);
        // Immediate reload of the sidecar file re-attaches the scale with
        // bitwise-identical numerics: unload released the scale-carrying
        // weight exactly once with no manifest store involved.
        let hfq = HfqFile::open(fixture.path()).expect("reopen AWQ fixture");
        let mut ctx = load_ctx(fixture.path(), &mut gpu, &cask);
        let mut bundle = load_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("AWQ legacy reload");
        drop(ctx);
        assert_eq!(bundle.weights.layers[0].wo.gpu_dtype, DType::MQ4G256);
        assert!(bundle.weights.layers[0].wo.awq_scale.is_some());
        let reload_logits = forward_logits(&mut gpu, &mut bundle);
        assert_eq!(reload_logits, awq_logits, "AWQ reload decode parity");
        Box::new(bundle).free_gpu(&mut gpu);
    }

    /// Repeated production load/unload cycles on the pinned fixture must not
    /// leak: every cycle records the same journal provenance upload count,
    /// decodes identically, and holds post-teardown free VRAM at the
    /// post-warmup plateau within 256 MiB slack. Manifest uploads are
    /// pool-backed and teardown returns every buffer to the pool, so the
    /// pool-hit counters and driver free VRAM both stabilize after warmup.
    #[test]
    fn pinned_fixture_repeated_load_unload_cycles_leak_nothing() {
        let _gpu_evidence_guard = GPU_EVIDENCE_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(fixture) = pinned_fixture_path() else {
            return;
        };
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            eprintln!("g3-cycles: no GPU; skipping");
            return;
        };
        const SLACK: usize = 256 * 1024 * 1024;
        let prompt = "The capital of France is located in";
        let max_seq = 64usize;
        let hfq = HfqFile::open(std::path::Path::new(&fixture)).expect("open pinned fixture");
        let tokenizer =
            hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                .expect("pinned fixture tokenizer");
        let prompt_tokens = tokenizer.encode(prompt);
        let cask = CaskConfig::default();
        test_support::reset();
        let mut baseline_tokens: Option<Vec<u32>> = None;
        let mut provenance_baseline = 0usize;
        let mut stabilized_free: usize = 0;
        for cycle in 0..4 {
            let (free_before, _) = gpu.hip.get_vram_info().expect("vram before cycle");
            let (pool_new_before, pool_reused_before, _) = gpu.pool_stats();
            let alloc_before = test_support::resident_allocations();
            let mut ctx = load_ctx(std::path::Path::new(&fixture), &mut gpu, &cask);
            ctx.max_seq = max_seq;
            let hfq = HfqFile::open(std::path::Path::new(&fixture)).expect("reopen for cycle");
            let mut bundle = load_bundle(ModelSource::Hfq(hfq), &mut ctx)
                .unwrap_or_else(|e| panic!("cycle {cycle} production load: {e}"));
            drop(ctx);
            // Journal provenance count: the immutable per-cycle upload tally
            // (one record per fulfilled manifest entry). It is not an
            // ownership claim — ownership lives with the typed weights owner.
            let provenance = test_support::resident_allocations() - alloc_before;
            if cycle == 0 {
                provenance_baseline = provenance;
                assert!(
                    provenance_baseline > 0,
                    "cycle must record fulfilled manifest uploads"
                );
            } else {
                assert_eq!(
                    provenance, provenance_baseline,
                    "cycle {cycle} journal provenance must match the warm baseline upload count ({provenance_baseline})"
                );
            }
            let tokens = greedy_decode(
                &mut gpu,
                &bundle.weights,
                &bundle.config,
                &mut bundle.kv,
                &bundle.scratch,
                &prompt_tokens,
                8,
            );
            match &baseline_tokens {
                None => baseline_tokens = Some(tokens.clone()),
                Some(baseline) => assert_eq!(&tokens, baseline, "cycle {cycle} decode parity"),
            }
            Box::new(bundle).free_gpu(&mut gpu);
            let (free_after, _) = gpu.hip.get_vram_info().expect("vram after unload");
            let (pool_new_after, pool_reused_after, pool_bytes) = gpu.pool_stats();
            eprintln!(
                "g3-cycles: cycle {cycle} — provenance {provenance} uploads, free VRAM {free_before} -> {free_after}, pool new {pool_new_before}->{pool_new_after} reused {pool_reused_before}->{pool_reused_after} bytes_new {pool_bytes}"
            );
            // Cycle 0 pays one-time context cost (kernel modules, stream and
            // driver-side arena growth); cycle 1 sets the post-warmup
            // plateau level. Later cycles must hold that plateau within a
            // small slack. Pool-backed uploads plateau: freed weight buffers
            // return to the pool and the next cycle reuses them, so
            // `pool_new` stays flat and driver free VRAM holds. Steady
            // per-cycle growth of `pool_new` with flat `pool_reused` would
            // mean uploads bypass the pool while frees feed it.
            if cycle == 0 {
                stabilized_free = free_after;
            } else if cycle == 1 {
                stabilized_free = free_after;
            } else {
                assert!(
                    free_after + SLACK >= stabilized_free,
                    "cycle {cycle} broke post-warmup VRAM plateau: {stabilized_free} -> {free_after}"
                );
            }
        }
        assert!(baseline_tokens.is_some());
        eprintln!(
            "g3-cycles: PASS — 4 load/unload cycles, per-cycle provenance {provenance_baseline} uploads, decode identical, post-warmup VRAM plateau held"
        );
    }

    /// Legacy-route load/unload cycles on the pinned fixture, measured with
    /// the same VRAM floor as the manifest-route cycle test. gfx1151 UMA
    /// pooling means hipMemGetInfo does not credit driver-pooled frees
    /// in-cycle on either route; this test pins that the manifest route
    /// behaves no worse than the legacy loader it replaces.
    #[test]
    fn pinned_fixture_legacy_route_cycles_vram_bounded() {
        let _gpu_evidence_guard = GPU_EVIDENCE_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(fixture) = pinned_fixture_path() else {
            return;
        };
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            eprintln!("g3-legacy-vram: no GPU; skipping");
            return;
        };
        const SLACK: usize = 256 * 1024 * 1024;
        const PER_CYCLE_ALLOWANCE: usize = 1024 * 1024 * 1024;
        let mut stabilized_free: usize = 0;
        for cycle in 0..3 {
            let (free0, _) = gpu.hip.get_vram_info().expect("vram before cycle");
            let hfq = HfqFile::open(std::path::Path::new(&fixture)).expect("open pinned fixture");
            let config = <Llama as Architecture>::config_from_hfq(&hfq).expect("fixture config");
            let legacy = hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, &mut gpu)
                .expect("legacy load");
            let scratch =
                ForwardScratch::new_with_max_seq(&mut gpu, &config, 64).expect("legacy scratch");
            let dims = llama_kv_dims(&config, 64, None);
            let kv =
                <KvCache as KvCacheExt>::from_mode(KvMode::Q8, KvTarget::Single(&mut gpu), &dims)
                    .expect("legacy KV");
            let (free1, _) = gpu.hip.get_vram_info().expect("vram after load");
            scratch.free_gpu(&mut gpu);
            let _ = kv.free_gpu(&mut gpu);
            legacy.free_gpu(&mut gpu);
            let (free2, _) = gpu.hip.get_vram_info().expect("vram after unload");
            eprintln!(
                "g3-legacy-vram: cycle {cycle} free {free0} -> loaded {free1} -> unloaded {free2}"
            );
            if cycle == 0 {
                stabilized_free = free2;
            } else {
                assert!(
                    free2 + SLACK + (cycle as usize) * PER_CYCLE_ALLOWANCE >= stabilized_free,
                    "legacy cycle {cycle} exceeded VRAM floor: {stabilized_free} -> {free2}"
                );
            }
        }
        eprintln!("g3-legacy-vram: PASS — legacy-route cycles within the same VRAM floor");
    }

    /// Stream a file's actual MD5 without a whole-file allocation. The G3
    /// tracker identity is an MD5, so the fixture lock must hash content:
    /// size alone cannot detect a same-length substitution and printing a
    /// hardcoded digest proves nothing.
    fn fixture_md5_hex(path: &str) -> std::io::Result<String> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut context = md5::Context::new();
        let mut chunk = vec![0u8; 8 << 20];
        loop {
            let read = file.read(&mut chunk)?;
            if read == 0 {
                break;
            }
            context.consume(&chunk[..read]);
        }
        Ok(format!("{:x}", context.finalize()))
    }

    /// Distinct acceptance fixture (see the oracle docs): registry
    /// `qwen3:0.6b` = `qwen3-0.6b.hf4`. No equivalence with the historic
    /// `.mq4` pin is claimed.
    fn pinned_fixture_path() -> Option<String> {
        let fixture = std::env::var("G3_FIXTURE").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            format!("{home}/.hipfire/models/qwen3-0.6b.hf4")
        });
        const PINNED_SIZE: u64 = 436_006_912;
        const PINNED_MD5: &str = "0d1055bf8f9492df2e2374d0e8bf787f";
        let Ok(meta) = std::fs::metadata(&fixture) else {
            eprintln!("g3: fixture absent ({fixture}); skipping");
            return None;
        };
        assert_eq!(
            meta.len(),
            PINNED_SIZE,
            "fixture size mismatch — not the pinned qwen3:0.6b artifact (md5 {PINNED_MD5})"
        );
        let actual = fixture_md5_hex(&fixture).expect("hash pinned fixture");
        assert_eq!(
            actual, PINNED_MD5,
            "fixture content mismatch — not the pinned qwen3:0.6b artifact"
        );
        Some(fixture)
    }

    /// CPU-only route qualification for the distinct registry acceptance
    /// fixture: the file must open as HFQ, parse as a llama-family config,
    /// classify `ManifestPlainLlama` with no AWQ sidecars, and carry a
    /// tokenizer. No GPU is touched: this is the header gate the hardware
    /// evidence legs build on.
    #[test]
    fn registry_fixture_qualifies_for_manifest_route() {
        let Some(fixture) = pinned_fixture_path() else {
            return;
        };
        let hfq = HfqFile::open(std::path::Path::new(&fixture)).expect("open registry fixture");
        let config = <Llama as Architecture>::config_from_hfq(&hfq)
            .expect("registry fixture parses as llama-family");
        assert_eq!(
            classify_hfq_route(&hfq),
            HfqLoadRoute::ManifestPlainLlama,
            "registry fixture must take the production manifest route"
        );
        assert!(
            !hfq.has_awq_sidecars(),
            "registry fixture must be a plain (sidecar-free) LLaMA-family HFQ"
        );
        let _tokenizer =
            hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                .expect("registry fixture tokenizer");
        eprintln!(
            "g3-qualify: {fixture} n_layers={} dim={} route=ManifestPlainLlama",
            config.n_layers, config.dim
        );
    }

    /// Greedy argmax decode over `prompt_tokens` followed by `generated`
    /// self-generated tokens, one token per committed position.
    fn greedy_decode(
        gpu: &mut rdna_compute::Gpu,
        weights: &LlamaWeights,
        config: &LlamaConfig,
        kv: &mut KvCache,
        scratch: &ForwardScratch,
        prompt_tokens: &[u32],
        generated: usize,
    ) -> Vec<u32> {
        let mut next_token: u32 = 0;
        let mut out = Vec::new();
        let total = prompt_tokens.len() + generated;
        assert!(total <= kv.max_seq, "position budget vs KV max_seq");
        for pos in 0..total {
            let token = if pos < prompt_tokens.len() {
                prompt_tokens[pos]
            } else {
                next_token
            };
            forward_scratch_embed(gpu, weights, config, token, pos, scratch).expect("embed");
            forward_scratch_compute(gpu, weights, config, 0, kv, scratch).expect("compute");
            let logits = gpu.download_f32(&scratch.logits).expect("logits");
            next_token = argmax_index(&logits) as u32;
            out.push(token);
        }
        out
    }

    fn argmax_index(logits: &[f32]) -> usize {
        let mut best = 0usize;
        for (index, value) in logits.iter().enumerate() {
            if value > &logits[best] {
                best = index;
            }
        }
        best
    }
}
