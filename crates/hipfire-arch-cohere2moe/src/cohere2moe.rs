// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cohere2-MoE (North-Mini-Code) weights + decode state.
//!
//! HFQ files carry RAW HF tensor names; the loader looks each up by exact name
//! (no rename). Mirrors the LFM2.5-MoE / MiniMax loaders (shared `WeightTensor`,
//! `KvCache`, indexed-MoE GEMV kernels) but reflects the Cohere2 structure:
//!   * ONE `input_layernorm` per layer (parallel block — feeds both attention
//!     and the FFN; there is no `post_attention_layernorm`).
//!   * No QK-norm, no attention bias.
//!   * Per-layer FFN split: the first `first_k_dense_replace` layers are dense
//!     SwiGLU MLPs (`mlp.{gate,up,down}_proj`, intermediate 3072); the rest are
//!     128-expert MoE (`mlp.experts.{j}.{gate,up,down}_proj` + `mlp.gate`), no
//!     routing bias, no shared expert.
//!   * Tied embeddings: `lm_head` reuses `model.embed_tokens.weight`.
//!
//! Expert weights ship pre-split (gate_proj/up_proj/down_proj); the loader
//! byte-fuses gate_proj‖up_proj into the per-expert `gate_up` blob the indexed
//! GEMV kernels expect. Per-expert buffers are retained (not just a packed
//! blob) so the shared sealed-MoE executor can select indexed kernels for
//! MQ4/MQ6 and the host-expert operation for F16/Q8 tiers.

use std::hash::{Hash, Hasher};

use crate::config::{AttnKind, Cohere2MoeConfig};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, KvCache, WeightTensor};
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::sealed_moe::{ExpertExecutionPlan, ExpertSourceMetadata};
use hipfire_runtime::tp_shard::ExpertAssign;
use hipfire_runtime::weight_manifest::{
    ExpertGroupSpec, ExpertParallelism, ExpertProjectionResources, ExpertResourceRequirements,
    ExpertSourceLayout, ShardPolicy, WeightEntry,
};
use rdna_compute::{DType, Gpu, GpuTensor};

// ───────────────────────── HFQ load helpers ─────────────────────────

fn read_tensor(hfq: &HfqFile, name: &str) -> Result<(u8, Vec<u8>), String> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("cohere2moe: tensor not found in HFQ: {name}"))?;
    Ok((info.quant_type, data))
}
fn load_wt(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    wt_from_raw(gpu, qt, &data, m, k).map_err(|e| format!("cohere2moe: load_wt {name}: {e}"))
}

fn hfq_source_metadata(
    hfq: &HfqFile,
    name: &str,
    m: usize,
    k: usize,
    fingerprint: &str,
) -> Result<ExpertSourceMetadata, String> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("cohere2moe: tensor not found in HFQ: {name}"))?;
    let expected_shape = vec![m, k];
    let shape = info
        .shape
        .iter()
        .map(|&dimension| dimension as usize)
        .collect::<Vec<_>>();
    if shape != expected_shape {
        return Err(format!(
            "cohere2moe: source {name} shape {:?} != {:?}",
            shape, expected_shape
        ));
    }
    let (dtype, row_stride) = validate_raw_layout(info.quant_type, data.len(), m, k)?;
    let mut metadata = ExpertSourceMetadata::new(
        info.name.clone(),
        fingerprint.to_string(),
        shape,
        dtype,
        data.len(),
        row_stride,
        1,
        format!("hfq:qt{}:g{}", info.quant_type, info.group_size),
        format!("{:?}", hipfire_dispatch::types::dtype_rotation_plan(dtype)),
    );
    metadata.sidecar_source_names = Vec::new();
    Ok(metadata)
}

/// Load a 1D/raw F16/F32/Q8 vector → F32 GpuTensor with the given shape. Used
/// for the per-layer + final **RMSNorm** weights (cohere2_moe uses RMSNorm at
/// `rms_norm_eps`, NOT base Cohere2's mean-centered LayerNorm). RMSNorm has a
/// learned weight (gamma) and no bias — this loads the gamma vector.
fn load_f32(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    shape: &[usize],
) -> Result<GpuTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    let f32_data: Vec<f32> = match qt {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        3 => dequant_q8_0(&data),
        _ => {
            return Err(format!(
                "cohere2moe: expected F16/F32/Q8 for {name}, got qt={qt}"
            ))
        }
    };
    gpu.upload_f32(&f32_data, shape)
        .map_err(|e| format!("cohere2moe: upload {name}: {e:?}"))
}

/// Minimal Q8_0 dequant (32-elem blocks: little-endian f16 scale + 32 int8).
fn dequant_q8_0(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 34 * 32);
    for blk in data.chunks_exact(34) {
        let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for &q in &blk[2..34] {
            out.push((q as i8) as f32 * scale);
        }
    }
    out
}

/// Quant type → GPU dtype mapping (mirrors the loader's dispatch mapping).
fn dtype_for_quant_type(qt: u8) -> Result<DType, String> {
    match qt {
        1 => Ok(DType::F16),
        2 => Ok(DType::F32),
        16 => Ok(DType::BF16), // native bf16 reference (oracle tier)
        3 => Ok(DType::Q8_0),
        6 => Ok(DType::HFQ4G256),
        8 => Ok(DType::HFQ6G256),
        13 => Ok(DType::MQ4G256),
        15 => Ok(DType::MQ6G256),
        17 => Ok(DType::MQ3G256),
        18 => Ok(DType::MQ2G256),
        19 => Ok(DType::MQ2G256Lloyd),
        20 => Ok(DType::MQ3G256Lloyd),
        30 => Ok(DType::MQ4G256Lloyd),
        other => Err(format!("unsupported quant_type {other}")),
    }
}

/// Return the encoded byte count for one logical row.
///
/// `WeightTensor::row_stride` is a byte stride for raw weights, not a count of
/// logical columns.  In particular, the G256 formats encode 256 columns in
/// 136/200 bytes, so comparing the encoded stride with `k` rejects valid
/// tensors.  Keep the layout formulas here in lockstep with the corresponding
/// `DType`/quantizer contracts and check every multiplication.
fn encoded_row_bytes(dtype: DType, k: usize) -> Result<usize, String> {
    let grouped = |group_k: usize, group_bytes: usize| {
        if k % group_k != 0 {
            return Err(format!(
                "{dtype:?} weights require K divisible by {group_k}, got K={k}"
            ));
        }
        (k / group_k).checked_mul(group_bytes).ok_or_else(|| {
            format!("{dtype:?} encoded row byte length overflows: (K/{group_k})*{group_bytes}")
        })
    };

    match dtype {
        DType::F32 => k
            .checked_mul(DType::F32.size())
            .ok_or_else(|| format!("{dtype:?} encoded row byte length overflows: K*4")),
        DType::F16 | DType::BF16 => k
            .checked_mul(2)
            .ok_or_else(|| format!("{dtype:?} encoded row byte length overflows: K*2")),
        DType::Q8_0 => grouped(32, 34),
        DType::HFQ4G256 | DType::MQ4G256 => grouped(256, 136),
        DType::HFQ6G256 | DType::MQ6G256 => grouped(256, 200),
        DType::MQ3G256 => grouped(256, 104),
        DType::MQ2G256 | DType::MQ2G256Lloyd => grouped(256, 72),
        DType::MQ3G256Lloyd => grouped(256, 112),
        DType::MQ4G256Lloyd => grouped(256, 160),
        // Paro's repacked projection layout is HFQ4-G128: 72 bytes per
        // 128 logical columns.  `wt_from_raw` is HFQ-only today, but keeping
        // the authoritative row formula here prevents a future raw Paro
        // entry from falling back to one-byte-per-column reasoning.
        DType::ParoQ4G128 => grouped(128, 72),
        other => Err(format!("no encoded row byte formula for dtype {other:?}")),
    }
}

/// Validate a raw tensor's dimensions and encoded length before any GPU upload.
/// Returns the native dtype and exact byte stride of one encoded row.
fn validate_raw_layout(
    qt: u8,
    data_len: usize,
    m: usize,
    k: usize,
) -> Result<(DType, usize), String> {
    let dtype = dtype_for_quant_type(qt)?;
    if m == 0 || k == 0 {
        return Err(format!(
            "{dtype:?} weights require non-zero dimensions, got m={m} k={k}"
        ));
    }
    let row_stride = encoded_row_bytes(dtype, k)?;
    let expected = row_stride.checked_mul(m).ok_or_else(|| {
        format!("{dtype:?} encoded weight byte length overflows: m={m} row_stride={row_stride}")
    })?;
    if data_len % m != 0 {
        return Err(format!(
            "{dtype:?} encoded weight bytes {data_len} do not divide evenly into {m} rows"
        ));
    }
    if data_len != expected {
        return Err(format!(
            "{dtype:?} encoded weight bytes {data_len} do not match exact \
             layout {m} rows × {row_stride} bytes ({expected} total)"
        ));
    }
    Ok((dtype, row_stride))
}

fn wt_from_raw(
    gpu: &mut Gpu,
    qt: u8,
    data: &[u8],
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (dtype, row_stride) = validate_raw_layout(qt, data.len(), m, k)?;
    let buf = gpu
        .upload_raw(data, &[data.len()])
        .map_err(|e| format!("upload_raw: {e:?}"))?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: dtype,
        m,
        k,
        row_stride,
        paro: None,
        awq_scale: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_rows_use_encoded_stride_not_logical_columns() {
        for &(qt, dtype, row_stride) in
            &[(13, DType::MQ4G256, 2 * 136), (15, DType::MQ6G256, 2 * 200)]
        {
            let (actual_dtype, actual_stride) =
                validate_raw_layout(qt, 2 * row_stride, 2, 512).unwrap();
            assert_eq!(actual_dtype, dtype);
            assert_eq!(actual_stride, row_stride);
            assert!(actual_stride < 512);
        }
    }

    #[test]
    fn quantized_rows_require_exact_encoded_length() {
        for &(qt, row_stride) in &[(13, 2 * 136), (15, 2 * 200)] {
            let valid = 2 * row_stride;
            assert!(validate_raw_layout(qt, valid, 2, 512).is_ok());
            assert!(validate_raw_layout(qt, valid - 1, 2, 512).is_err());
            assert!(validate_raw_layout(qt, valid + 1, 2, 512).is_err());
            assert!(validate_raw_layout(qt, valid + row_stride, 2, 512).is_err());
        }
    }

    #[test]
    fn dense_rows_keep_exact_element_width() {
        for &(qt, element_bytes) in &[(1, 2), (2, 4), (16, 2)] {
            let valid = 2 * 8 * element_bytes;
            assert!(validate_raw_layout(qt, valid, 2, 8).is_ok());
            assert!(validate_raw_layout(qt, valid - 1, 2, 8).is_err());
            assert!(validate_raw_layout(qt, valid + element_bytes, 2, 8).is_err());
        }
    }

    #[test]
    fn grouped_formats_reject_ragged_k() {
        assert!(validate_raw_layout(13, 136, 1, 257).is_err());
        assert!(validate_raw_layout(15, 200, 1, 257).is_err());
        assert_eq!(encoded_row_bytes(DType::ParoQ4G128, 128).unwrap(), 72);
    }
}

// ──────────────────────────── Weights ────────────────────────────
/// Immutable, load-time proof for one Cohere MoE layer.  The dispatch table
/// owns only CPU metadata; the pointer tables and expert buffers remain owned
/// by this layer's `MoeFfn`.
pub(crate) struct MoeStaticBinding {
    pub plan: ExpertExecutionPlan,
    pub table: hipfire_dispatch::pipeline::sealed_moe::ExpertTable,
    pub cache: hipfire_dispatch::pipeline::sealed_moe::ExpertBindingCache,
    pub gate_up_dtypes: Box<[DType]>,
    pub down_dtypes: Box<[DType]>,
}

pub(crate) fn source_fingerprint(source: &dyn ModelSource, names: &[String]) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source.path().to_string_lossy().hash(&mut hasher);
    source.metadata_json().hash(&mut hasher);
    for name in names {
        name.hash(&mut hasher);
        if let Some(info) = source.tensor_info(name) {
            info.name.hash(&mut hasher);
            info.dtype.hash(&mut hasher);
            info.shape.hash(&mut hasher);
            info.quant_type.hash(&mut hasher);
            info.data_offset.hash(&mut hasher);
            info.data_size.hash(&mut hasher);
        }
    }
    format!("{:016x}", hasher.finish())
}

pub(crate) fn source_row_stride(bytes: usize, rows: usize) -> Result<usize, String> {
    if rows == 0 {
        return Err("encoded source row count is zero".to_string());
    }
    bytes
        .checked_div(rows)
        .filter(|&stride| stride > 0 && stride * rows == bytes)
        .ok_or_else(|| format!("encoded source bytes {bytes} do not form {rows} rows"))
}
pub(crate) fn model_source_metadata(
    source: &dyn ModelSource,
    name: &str,
    logical_shape: Vec<usize>,
    encoded_bytes: usize,
    row_stride: usize,
    dtype: DType,
    fingerprint: &str,
    quant_tag: String,
    basis: String,
    sidecars: Vec<String>,
) -> Result<ExpertSourceMetadata, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("cohere2moe: source tensor not found: {name}"))?;
    if logical_shape.is_empty() || encoded_bytes == 0 || row_stride == 0 {
        return Err(format!(
            "cohere2moe: invalid source layout for {name}: shape={logical_shape:?} bytes={encoded_bytes} stride={row_stride}"
        ));
    }
    let rows = logical_shape[0];
    let expected = row_stride
        .checked_mul(rows)
        .ok_or_else(|| format!("cohere2moe: source row capacity overflows for {name}"))?;
    if expected != encoded_bytes {
        return Err(format!(
            "cohere2moe: source {name} bytes {encoded_bytes} != rows {rows} × stride {row_stride} ({expected})"
        ));
    }
    let mut metadata = ExpertSourceMetadata::new(
        info.name.clone(),
        fingerprint.to_string(),
        logical_shape,
        dtype,
        encoded_bytes,
        row_stride,
        1,
        quant_tag,
        basis,
    );
    metadata.sidecar_source_names = sidecars;
    Ok(metadata)
}

/// Build the one runtime plan used by both Cohere carriers.  The source
/// metadata vectors are assembled before the corresponding expert upload;
/// this function never reads GPU state or constructs dispatch metadata itself.
pub(crate) fn build_moe_static_binding(
    router: ExpertSourceMetadata,
    gates: Vec<ExpertSourceMetadata>,
    ups: Vec<ExpertSourceMetadata>,
    downs: Vec<ExpertSourceMetadata>,
    sidecars: Vec<ExpertSourceMetadata>,
    layer_idx: usize,
    n_layers: usize,
    physical_device: i32,
) -> Result<MoeStaticBinding, String> {
    if gates.is_empty() || gates.len() != ups.len() || gates.len() != downs.len() {
        return Err(format!(
            "expert binding metadata count mismatch (gate {}, up {}, down {})",
            gates.len(),
            ups.len(),
            downs.len()
        ));
    }
    let n_experts = gates.len();
    let mut source_by_name = std::collections::BTreeMap::<String, ExpertSourceMetadata>::new();
    let mut add_source = |source: ExpertSourceMetadata| -> Result<(), String> {
        if let Some(previous) = source_by_name.get(&source.name) {
            if previous != &source {
                return Err(format!(
                    "expert source '{}' metadata is inconsistent",
                    source.name
                ));
            }
        } else {
            source_by_name.insert(source.name.clone(), source);
        }
        Ok(())
    };
    add_source(router.clone())?;
    for source in sidecars {
        add_source(source)?;
    }
    for source in gates.iter().chain(ups.iter()).chain(downs.iter()) {
        add_source(source.clone())?;
    }
    let first_gate = gates
        .first()
        .ok_or_else(|| "expert binding has no gate metadata".to_string())?;
    let first_up = ups
        .first()
        .ok_or_else(|| "expert binding has no up metadata".to_string())?;
    let first_down = downs
        .first()
        .ok_or_else(|| "expert binding has no down metadata".to_string())?;
    if first_gate.logical_shape.len() != 2
        || first_up.logical_shape != first_gate.logical_shape
        || first_down.logical_shape.len() != 2
    {
        return Err("expert source projection geometry mismatch".to_string());
    }
    for ((gate, up), down) in gates.iter().zip(&ups).zip(&downs) {
        if gate.logical_shape != first_gate.logical_shape
            || up.logical_shape != first_up.logical_shape
            || down.logical_shape != first_down.logical_shape
            || gate.dtype != up.dtype
            || gate.alignment != up.alignment
            || gate.alignment != down.alignment
        {
            return Err(
                "expert source projection geometry, dtype, or alignment mismatch".to_string(),
            );
        }
    }
    let mut manifest = Vec::with_capacity(4 + n_experts * 3 + source_by_name.len());
    manifest.push(WeightEntry::layer(
        router.name.clone(),
        layer_idx,
        router.logical_shape.clone(),
        router.dtype,
        ShardPolicy::Replicate,
    ));
    let gate_names: Vec<String> = gates.iter().map(|source| source.name.clone()).collect();
    let up_names: Vec<String> = ups.iter().map(|source| source.name.clone()).collect();
    let down_names: Vec<String> = downs.iter().map(|source| source.name.clone()).collect();
    for source in &gates {
        manifest.push(WeightEntry::layer(
            source.name.clone(),
            layer_idx,
            vec![n_experts, source.logical_shape[0], source.logical_shape[1]],
            source.dtype,
            ShardPolicy::Replicate,
        ));
    }
    for source in &ups {
        manifest.push(WeightEntry::layer(
            source.name.clone(),
            layer_idx,
            vec![n_experts, source.logical_shape[0], source.logical_shape[1]],
            source.dtype,
            ShardPolicy::Replicate,
        ));
    }
    for source in &downs {
        manifest.push(WeightEntry::layer(
            source.name.clone(),
            layer_idx,
            vec![n_experts, source.logical_shape[0], source.logical_shape[1]],
            source.dtype,
            ShardPolicy::Replicate,
        ));
    }
    let sidecar_names = gates
        .iter()
        .chain(ups.iter())
        .chain(downs.iter())
        .flat_map(|source| source.sidecar_source_names.iter().cloned())
        .collect::<std::collections::BTreeSet<_>>();
    for name in &sidecar_names {
        let source = source_by_name
            .get(name)
            .ok_or_else(|| format!("expert sidecar metadata '{name}' is absent"))?;
        manifest.push(WeightEntry::layer(
            name.clone(),
            layer_idx,
            source.logical_shape.clone(),
            source.dtype,
            ShardPolicy::Replicate,
        ));
    }
    let resources = gates
        .iter()
        .zip(&ups)
        .zip(&downs)
        .map(|((gate, up), down)| {
            if gate.alignment != up.alignment || gate.alignment != down.alignment {
                return Err("expert projection alignment mismatch".to_string());
            }
            Ok(ExpertProjectionResources {
                gate_bytes: gate.encoded_bytes,
                up_bytes: up.encoded_bytes,
                down_bytes: down.encoded_bytes,
                alignment: gate.alignment,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let spec = ExpertGroupSpec {
        group: format!("cohere2moe.layer.{layer_idx}"),
        layer: Some(layer_idx),
        n_experts,
        parallelism: ExpertParallelism::Single,
        assignment: ExpertAssign::Stride,
        source_layout: ExpertSourceLayout::PerExpertSeparate {
            gate: gate_names,
            up: up_names,
            down: down_names,
            sidecars: sidecar_names.into_iter().collect(),
        },
        resources: ExpertResourceRequirements::new(resources),
        router: router.name,
        execution: "grouped-prefill-single".to_string(),
    };
    let sources: Vec<ExpertSourceMetadata> = source_by_name.into_values().collect();
    let (plan, table, cache) = hipfire_runtime::sealed_moe::plan_single_expert_execution(
        &manifest,
        &spec,
        &sources,
        n_layers,
        physical_device,
    )?;
    Ok(MoeStaticBinding {
        plan,
        table,
        cache,
        gate_up_dtypes: gates.iter().map(|source| source.dtype).collect(),
        down_dtypes: downs.iter().map(|source| source.dtype).collect(),
    })
}

enum PendingOwner {
    Tensor(GpuTensor),
    Weight(WeightTensor),
    Expert(ExpertWeights),
    Sidecars(Cohere2MoeParoSidecars),
    Moe(MoeFfn),
    Layer(Cohere2MoeLayerWeights),
}

/// Transactional owner registry for model loading.  Every successful GPU
/// allocation is registered before a later source, table, or metadata step
/// can fail, so partial loads cannot leak or publish an unbound layer.
pub(crate) struct LoadTransaction {
    gpu: *mut Gpu,
    owners: Vec<Option<PendingOwner>>,
    committed: bool,
}

impl LoadTransaction {
    pub(crate) fn new(gpu: &mut Gpu) -> Self {
        Self {
            gpu,
            owners: Vec::new(),
            committed: false,
        }
    }

    fn hold(&mut self, owner: PendingOwner) -> usize {
        self.owners.push(Some(owner));
        self.owners.len() - 1
    }

    pub(crate) fn hold_tensor(&mut self, tensor: GpuTensor) -> usize {
        self.hold(PendingOwner::Tensor(tensor))
    }

    pub(crate) fn hold_weight(&mut self, weight: WeightTensor) -> usize {
        self.hold(PendingOwner::Weight(weight))
    }

    pub(crate) fn hold_expert(&mut self, expert: ExpertWeights) -> usize {
        self.hold(PendingOwner::Expert(expert))
    }

    pub(crate) fn hold_sidecars(&mut self, sidecars: Cohere2MoeParoSidecars) -> usize {
        self.hold(PendingOwner::Sidecars(sidecars))
    }

    pub(crate) fn hold_moe(&mut self, moe: MoeFfn) -> usize {
        self.hold(PendingOwner::Moe(moe))
    }

    pub(crate) fn hold_layer(&mut self, layer: Cohere2MoeLayerWeights) -> usize {
        self.hold(PendingOwner::Layer(layer))
    }

    fn take_owner(&mut self, slot: usize) -> Result<PendingOwner, String> {
        self.owners
            .get_mut(slot)
            .and_then(Option::take)
            .ok_or_else(|| format!("load transaction owner slot {slot} is empty"))
    }

    pub(crate) fn take_tensor(&mut self, slot: usize) -> Result<GpuTensor, String> {
        match self.take_owner(slot)? {
            PendingOwner::Tensor(tensor) => Ok(tensor),
            _ => Err(format!(
                "load transaction owner slot {slot} is not a tensor"
            )),
        }
    }

    pub(crate) fn take_weight(&mut self, slot: usize) -> Result<WeightTensor, String> {
        match self.take_owner(slot)? {
            PendingOwner::Weight(weight) => Ok(weight),
            _ => Err(format!(
                "load transaction owner slot {slot} is not a weight"
            )),
        }
    }

    pub(crate) fn take_expert(&mut self, slot: usize) -> Result<ExpertWeights, String> {
        match self.take_owner(slot)? {
            PendingOwner::Expert(expert) => Ok(expert),
            _ => Err(format!(
                "load transaction owner slot {slot} is not an expert"
            )),
        }
    }

    pub(crate) fn take_sidecars(&mut self, slot: usize) -> Result<Cohere2MoeParoSidecars, String> {
        match self.take_owner(slot)? {
            PendingOwner::Sidecars(sidecars) => Ok(sidecars),
            _ => Err(format!(
                "load transaction owner slot {slot} is not sidecars"
            )),
        }
    }

    pub(crate) fn take_moe(&mut self, slot: usize) -> Result<MoeFfn, String> {
        match self.take_owner(slot)? {
            PendingOwner::Moe(moe) => Ok(moe),
            _ => Err(format!("load transaction owner slot {slot} is not an MoE")),
        }
    }

    pub(crate) fn moe_mut(&mut self, slot: usize) -> Option<&mut MoeFfn> {
        match self.owners.get_mut(slot).and_then(Option::as_mut) {
            Some(PendingOwner::Moe(moe)) => Some(moe),
            _ => None,
        }
    }

    pub(crate) fn take_layer(&mut self, slot: usize) -> Result<Cohere2MoeLayerWeights, String> {
        match self.take_owner(slot)? {
            PendingOwner::Layer(layer) => Ok(layer),
            _ => Err(format!("load transaction owner slot {slot} is not a layer")),
        }
    }
    pub(crate) fn expert_ref(&self, slot: usize) -> Option<&ExpertWeights> {
        match self.owners.get(slot).and_then(Option::as_ref) {
            Some(PendingOwner::Expert(expert)) => Some(expert),
            _ => None,
        }
    }

    pub(crate) fn sidecars_ref(&self, slot: usize) -> Option<&Cohere2MoeParoSidecars> {
        match self.owners.get(slot).and_then(Option::as_ref) {
            Some(PendingOwner::Sidecars(sidecars)) => Some(sidecars),
            _ => None,
        }
    }

    pub(crate) fn tensor_ref(&self, slot: usize) -> Option<&GpuTensor> {
        match self.owners.get(slot).and_then(Option::as_ref) {
            Some(PendingOwner::Tensor(tensor)) => Some(tensor),
            _ => None,
        }
    }

    pub(crate) fn weight_ref(&self, slot: usize) -> Option<&WeightTensor> {
        match self.owners.get(slot).and_then(Option::as_ref) {
            Some(PendingOwner::Weight(weight)) => Some(weight),
            _ => None,
        }
    }
    pub(crate) fn commit(&mut self) {
        self.committed = true;
        self.owners.clear();
    }
}

impl Drop for LoadTransaction {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        let gpu = unsafe { &mut *self.gpu };
        for owner in self.owners.drain(..).rev().flatten() {
            match owner {
                PendingOwner::Tensor(tensor) => {
                    let _ = gpu.free_tensor(tensor);
                }
                PendingOwner::Weight(weight) => weight.free_all(gpu),
                PendingOwner::Expert(expert) => expert.free_gpu(gpu),
                PendingOwner::Sidecars(sidecars) => sidecars.free_gpu(gpu),
                PendingOwner::Moe(moe) => moe.free_gpu(gpu),
                PendingOwner::Layer(layer) => layer.free_gpu(gpu),
            }
        }
    }
}

/// Dense SwiGLU MLP (the `first_k_dense_replace` prefix layers; layer 0 here).
pub struct DenseFfn {
    pub gate: WeightTensor, // mlp.gate_proj [dense_inter, hidden]
    pub up: WeightTensor,   // mlp.up_proj   [dense_inter, hidden]
    pub down: WeightTensor, // mlp.down_proj [hidden, dense_inter]
}

/// One MoE expert: fused gate(gate_proj)‖up(up_proj) and down(down_proj).
pub struct ExpertWeights {
    pub gate_up: WeightTensor, // [2*moe_inter, hidden]
    pub down: WeightTensor,    // [hidden, moe_inter]
}

/// Per-layer shared ParoQuant rotation sidecars (Dir/safetensors path only).
/// All experts in a MoE layer reference these via non-owning `ParoRotation`
/// aliases, so the owner must outlive the experts — `MoeFfn` holds it.
pub struct Cohere2MoeParoSidecars {
    pub gate_up_pairs: GpuTensor,
    pub gate_up_theta: GpuTensor,
    pub gate_up_channel_scales: GpuTensor,
    pub down_pairs: GpuTensor,
    pub down_theta: GpuTensor,
    pub down_channel_scales: GpuTensor,
}

/// 128-expert MoE FFN (sigmoid selection, no bias, no shared expert).
pub struct MoeFfn {
    pub router: WeightTensor,           // mlp.gate.weight [n_exp, hidden]
    pub experts: Vec<ExpertWeights>,    // per-expert buffers (owned here)
    pub expert_gate_up_ptrs: GpuTensor, // [2*n_exp] F32 = n_exp u64 device ptrs
    pub expert_down_ptrs: GpuTensor,
    /// Load-time source/layout/dtype proof. This field is mandatory for every
    /// published MoE owner; forward code has no raw-execution fallback.
    pub(crate) sealed: MoeStaticBinding,
    /// Owned per-layer PARO sidecars (Dir/safetensors path only). The experts'
    /// `ParoRotation` aliases reference these — keep them alive.
    pub paro_shared: Option<Cohere2MoeParoSidecars>,
}

struct ResidentMoeExperts<'a>(&'a [ExpertWeights]);

impl hipfire_dispatch::families::moe::RoutedExpertWeights for ResidentMoeExperts<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn get(
        &self,
        expert_idx: usize,
    ) -> Option<(
        hipfire_dispatch::families::gemv::WeightRef<'_>,
        hipfire_dispatch::families::gemv::WeightRef<'_>,
    )> {
        self.0
            .get(expert_idx)
            .map(|expert| (expert.gate_up.dispatch_ref(), expert.down.dispatch_ref()))
    }
}

impl MoeFfn {
    pub(crate) fn bind_live(&mut self) -> Result<(), String> {
        self.sealed
            .cache
            .bind_live(
                &self.sealed.table,
                &ResidentMoeExperts(&self.experts),
                &self.expert_gate_up_ptrs,
                &self.expert_down_ptrs,
                None,
                None,
            )
            .map_err(|error| format!("cohere2moe: bind live expert resources: {error:?}"))
    }
}
impl hipfire_dispatch::families::moe::RoutedExpertWeights for MoeFfn {
    fn len(&self) -> usize {
        self.experts.len()
    }

    fn get(
        &self,
        expert_idx: usize,
    ) -> Option<(
        hipfire_dispatch::families::gemv::WeightRef<'_>,
        hipfire_dispatch::families::gemv::WeightRef<'_>,
    )> {
        self.experts
            .get(expert_idx)
            .map(|expert| (expert.gate_up.dispatch_ref(), expert.down.dispatch_ref()))
    }
}

pub enum Ffn {
    Dense(DenseFfn),
    Moe(MoeFfn),
}

pub struct Cohere2MoeLayerWeights {
    /// The SINGLE `input_layernorm` (RMSNorm gamma, [hidden]). Feeds
    /// both the attention and FFN branches (parallel block).
    pub input_norm: GpuTensor,
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    pub ffn: Ffn,
    /// full_attention (global, NoPE) vs sliding_attention (window, RoPE).
    pub attn_kind: AttnKind,
}

pub struct Cohere2MoeWeights {
    pub embed: GpuTensor,      // model.embed_tokens.weight (raw bytes)
    pub embed_dtype: DType,    // dtype of `embed` (drives the lookup path)
    pub final_norm: GpuTensor, // model.norm.weight (RMSNorm gamma)
    pub lm_head: WeightTensor, // tied = embed_tokens
    pub layers: Vec<Cohere2MoeLayerWeights>,
}
impl Cohere2MoeWeights {
    pub fn load(hfq: &mut HfqFile, cfg: &Cohere2MoeConfig, gpu: &mut Gpu) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let dense_inter = cfg.dense_intermediate_size;
        let moe_inter = cfg.moe_intermediate_size;
        let n_exp = cfg.num_experts;
        let gate_up_rows = moe_inter
            .checked_mul(2)
            .ok_or_else(|| "cohere2moe: gate/up dimension overflows".to_string())?;
        let ptr_count = n_exp
            .checked_mul(2)
            .ok_or_else(|| "cohere2moe: expert pointer-table length overflows".to_string())?;

        let mut tx = LoadTransaction::new(gpu);

        // Register every global allocation before the next fallible load.
        let (_embed_qt, embed_bytes) = read_tensor(hfq, "model.embed_tokens.weight")?;
        let embed_slot = tx.hold_tensor(
            gpu.upload_raw(&embed_bytes, &[embed_bytes.len()])
                .map_err(|e| format!("cohere2moe: upload embed: {e:?}"))?,
        );
        let lm_head_slot = tx.hold_weight(load_wt(
            hfq,
            gpu,
            "model.embed_tokens.weight",
            cfg.vocab_size,
            hidden,
        )?);
        let embed_dtype = tx
            .weight_ref(lm_head_slot)
            .map(|weight| weight.gpu_dtype)
            .ok_or_else(|| "cohere2moe: lm_head transaction slot is not a weight".to_string())?;
        let final_norm_slot = tx.hold_tensor(load_f32(hfq, gpu, "model.norm.weight", &[hidden])?);

        let mut layer_slots = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            let input_norm_slot = tx.hold_tensor(load_f32(
                hfq,
                gpu,
                &format!("{p}.input_layernorm.weight"),
                &[hidden],
            )?);
            let wq_slot = tx.hold_weight(load_wt(
                hfq,
                gpu,
                &format!("{p}.self_attn.q_proj.weight"),
                q_dim,
                hidden,
            )?);
            let wk_slot = tx.hold_weight(load_wt(
                hfq,
                gpu,
                &format!("{p}.self_attn.k_proj.weight"),
                kv_dim,
                hidden,
            )?);
            let wv_slot = tx.hold_weight(load_wt(
                hfq,
                gpu,
                &format!("{p}.self_attn.v_proj.weight"),
                kv_dim,
                hidden,
            )?);
            let wo_slot = tx.hold_weight(load_wt(
                hfq,
                gpu,
                &format!("{p}.self_attn.o_proj.weight"),
                hidden,
                q_dim,
            )?);

            let ffn = if cfg.is_dense_ffn(l) {
                let gate_slot = tx.hold_weight(load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.gate_proj.weight"),
                    dense_inter,
                    hidden,
                )?);
                let up_slot = tx.hold_weight(load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.up_proj.weight"),
                    dense_inter,
                    hidden,
                )?);
                let down_slot = tx.hold_weight(load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.down_proj.weight"),
                    hidden,
                    dense_inter,
                )?);
                Ffn::Dense(DenseFfn {
                    gate: tx.take_weight(gate_slot)?,
                    up: tx.take_weight(up_slot)?,
                    down: tx.take_weight(down_slot)?,
                })
            } else {
                let router_name = format!("{p}.mlp.gate.weight");
                let source_capacity = n_exp
                    .checked_mul(3)
                    .and_then(|count| count.checked_add(1))
                    .ok_or_else(|| format!("cohere2moe L{l}: source-name capacity overflows"))?;
                let mut source_names = Vec::with_capacity(source_capacity);
                source_names.push(router_name.clone());
                for e in 0..n_exp {
                    let ep = format!("{p}.mlp.experts.{e}");
                    source_names.push(format!("{ep}.gate_proj.weight"));
                    source_names.push(format!("{ep}.up_proj.weight"));
                    source_names.push(format!("{ep}.down_proj.weight"));
                }
                let fingerprint = source_fingerprint(&*hfq, &source_names);
                let router_source =
                    hfq_source_metadata(hfq, &router_name, n_exp, hidden, &fingerprint)?;
                let mut gate_sources = Vec::with_capacity(n_exp);
                let mut up_sources = Vec::with_capacity(n_exp);
                let mut down_sources = Vec::with_capacity(n_exp);
                for e in 0..n_exp {
                    let ep = format!("{p}.mlp.experts.{e}");
                    gate_sources.push(hfq_source_metadata(
                        hfq,
                        &format!("{ep}.gate_proj.weight"),
                        moe_inter,
                        hidden,
                        &fingerprint,
                    )?);
                    up_sources.push(hfq_source_metadata(
                        hfq,
                        &format!("{ep}.up_proj.weight"),
                        moe_inter,
                        hidden,
                        &fingerprint,
                    )?);
                    down_sources.push(hfq_source_metadata(
                        hfq,
                        &format!("{ep}.down_proj.weight"),
                        hidden,
                        moe_inter,
                        &fingerprint,
                    )?);
                }
                let sealed = build_moe_static_binding(
                    router_source,
                    gate_sources,
                    up_sources,
                    down_sources,
                    Vec::new(),
                    l,
                    cfg.num_hidden_layers,
                    gpu.device_id,
                )
                .map_err(|error| format!("cohere2moe L{l}: {error}"))?;
                let router_slot = tx.hold_weight(load_wt(hfq, gpu, &router_name, n_exp, hidden)?);
                let mut expert_slots = Vec::with_capacity(n_exp);
                for e in 0..n_exp {
                    let ep = format!("{p}.mlp.experts.{e}");
                    let gate_name = format!("{ep}.gate_proj.weight");
                    let up_name = format!("{ep}.up_proj.weight");
                    let down_name = format!("{ep}.down_proj.weight");
                    let (qt_g, g) = read_tensor(hfq, &gate_name)?;
                    let (qt_u, u) = read_tensor(hfq, &up_name)?;
                    if qt_g != qt_u {
                        return Err(format!(
                            "cohere2moe L{l}E{e}: gate/up dtype mismatch ({qt_g:?} vs {qt_u:?}) — cannot byte-fuse gate_up"
                        ));
                    }
                    let (qt_d, d) = read_tensor(hfq, &down_name)?;
                    let gate_up_len = g.len().checked_add(u.len()).ok_or_else(|| {
                        format!("cohere2moe L{l}E{e}: gate/up byte length overflows")
                    })?;
                    let mut gate_up_bytes = Vec::with_capacity(gate_up_len);
                    gate_up_bytes.extend_from_slice(&g);
                    gate_up_bytes.extend_from_slice(&u);
                    let gate_up_slot = tx.hold_weight(
                        wt_from_raw(gpu, qt_g, &gate_up_bytes, gate_up_rows, hidden)
                            .map_err(|e2| format!("cohere2moe: fuse gate_up L{l}E{e}: {e2}"))?,
                    );
                    let down_slot = tx.hold_weight(
                        wt_from_raw(gpu, qt_d, &d, hidden, moe_inter)
                            .map_err(|e2| format!("cohere2moe: down L{l}E{e}: {e2}"))?,
                    );
                    let gate_up = tx.take_weight(gate_up_slot)?;
                    let down = tx.take_weight(down_slot)?;
                    let expert_slot = tx.hold_expert(ExpertWeights { gate_up, down });
                    expert_slots.push(expert_slot);
                }

                let gu_bytes: Vec<u8> = expert_slots
                    .iter()
                    .map(|slot| {
                        tx.expert_ref(*slot)
                            .ok_or_else(|| {
                                format!("cohere2moe L{l}: missing expert transaction slot")
                            })
                            .map(|expert| (expert.gate_up.buf.buf.as_ptr() as u64).to_ne_bytes())
                    })
                    .collect::<Result<Vec<[u8; 8]>, String>>()?
                    .into_iter()
                    .flatten()
                    .collect();
                let dn_bytes: Vec<u8> = expert_slots
                    .iter()
                    .map(|slot| {
                        tx.expert_ref(*slot)
                            .ok_or_else(|| {
                                format!("cohere2moe L{l}: missing expert transaction slot")
                            })
                            .map(|expert| (expert.down.buf.buf.as_ptr() as u64).to_ne_bytes())
                    })
                    .collect::<Result<Vec<[u8; 8]>, String>>()?
                    .into_iter()
                    .flatten()
                    .collect();
                let expert_gate_up_ptrs_slot = tx.hold_tensor(
                    gpu.alloc_tensor(&[ptr_count], DType::F32)
                        .map_err(|e| format!("cohere2moe: alloc gu_ptrs L{l}: {e:?}"))?,
                );
                let expert_down_ptrs_slot = tx.hold_tensor(
                    gpu.alloc_tensor(&[ptr_count], DType::F32)
                        .map_err(|e| format!("cohere2moe: alloc dn_ptrs L{l}: {e:?}"))?,
                );
                let gate_up_ptrs = tx
                    .tensor_ref(expert_gate_up_ptrs_slot)
                    .ok_or_else(|| format!("cohere2moe L{l}: missing gate/up pointer table"))?;
                gpu.hip
                    .memcpy_htod(&gate_up_ptrs.buf, &gu_bytes)
                    .map_err(|e| format!("cohere2moe: htod gu_ptrs L{l}: {e:?}"))?;
                let down_ptrs = tx
                    .tensor_ref(expert_down_ptrs_slot)
                    .ok_or_else(|| format!("cohere2moe L{l}: missing down pointer table"))?;
                gpu.hip
                    .memcpy_htod(&down_ptrs.buf, &dn_bytes)
                    .map_err(|e| format!("cohere2moe: htod dn_ptrs L{l}: {e:?}"))?;

                let router = tx.take_weight(router_slot)?;
                let mut experts = Vec::with_capacity(expert_slots.len());
                for slot in expert_slots {
                    experts.push(tx.take_expert(slot)?);
                }
                let expert_gate_up_ptrs = tx.take_tensor(expert_gate_up_ptrs_slot)?;
                let expert_down_ptrs = tx.take_tensor(expert_down_ptrs_slot)?;
                let moe_slot = tx.hold_moe(MoeFfn {
                    router,
                    experts,
                    expert_gate_up_ptrs,
                    expert_down_ptrs,
                    sealed,
                    paro_shared: None,
                });
                tx.moe_mut(moe_slot)
                    .ok_or_else(|| format!("cohere2moe L{l}: missing staged MoE owner"))?
                    .bind_live()?;
                Ffn::Moe(tx.take_moe(moe_slot)?)
            };

            let input_norm = tx.take_tensor(input_norm_slot)?;
            let wq = tx.take_weight(wq_slot)?;
            let wk = tx.take_weight(wk_slot)?;
            let wv = tx.take_weight(wv_slot)?;
            let wo = tx.take_weight(wo_slot)?;
            let layer_slot = tx.hold_layer(Cohere2MoeLayerWeights {
                input_norm,
                wq,
                wk,
                wv,
                wo,
                ffn,
                attn_kind: cfg.attn_kind(l),
            });
            layer_slots.push(layer_slot);
        }

        let mut layers = Vec::with_capacity(layer_slots.len());
        for slot in layer_slots {
            layers.push(tx.take_layer(slot)?);
        }
        let embed = tx.take_tensor(embed_slot)?;
        let final_norm = tx.take_tensor(final_norm_slot)?;
        let lm_head = tx.take_weight(lm_head_slot)?;
        tx.commit();
        Ok(Cohere2MoeWeights {
            embed,
            embed_dtype,
            final_norm,
            lm_head,
            layers,
        })
    }
}

impl DenseFfn {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let DenseFfn { gate, up, down } = self;
        gate.free_all(gpu);
        up.free_all(gpu);
        down.free_all(gpu);
    }
}

impl ExpertWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let ExpertWeights { gate_up, down } = self;
        gate_up.free_all(gpu);
        down.free_all(gpu);
    }
}

impl Cohere2MoeParoSidecars {
    pub(crate) fn free_gpu(self, gpu: &mut Gpu) {
        let Cohere2MoeParoSidecars {
            gate_up_pairs,
            gate_up_theta,
            gate_up_channel_scales,
            down_pairs,
            down_theta,
            down_channel_scales,
        } = self;
        let _ = gpu.free_tensor(gate_up_pairs);
        let _ = gpu.free_tensor(gate_up_theta);
        let _ = gpu.free_tensor(gate_up_channel_scales);
        let _ = gpu.free_tensor(down_pairs);
        let _ = gpu.free_tensor(down_theta);
        let _ = gpu.free_tensor(down_channel_scales);
    }
}

impl MoeFfn {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let MoeFfn {
            router,
            experts,
            expert_gate_up_ptrs,
            expert_down_ptrs,
            sealed: _,
            paro_shared,
        } = self;
        router.free_all(gpu);
        for expert in experts {
            expert.free_gpu(gpu);
        }
        let _ = gpu.free_tensor(expert_gate_up_ptrs);
        let _ = gpu.free_tensor(expert_down_ptrs);
        if let Some(sidecars) = paro_shared {
            sidecars.free_gpu(gpu);
        }
    }
}

impl Ffn {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        match self {
            Ffn::Dense(d) => d.free_gpu(gpu),
            Ffn::Moe(m) => m.free_gpu(gpu),
        }
    }
}

impl Cohere2MoeLayerWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let Cohere2MoeLayerWeights {
            input_norm,
            wq,
            wk,
            wv,
            wo,
            ffn,
            attn_kind: _,
        } = self;
        let _ = gpu.free_tensor(input_norm);
        wq.free_all(gpu);
        wk.free_all(gpu);
        wv.free_all(gpu);
        wo.free_all(gpu);
        ffn.free_gpu(gpu);
    }
}

impl Cohere2MoeWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let Cohere2MoeWeights {
            embed,
            embed_dtype: _,
            final_norm,
            lm_head,
            layers,
        } = self;
        let _ = gpu.free_tensor(embed);
        let _ = gpu.free_tensor(final_norm);
        lm_head.free_all(gpu);
        for layer in layers {
            layer.free_gpu(gpu);
        }
    }
}

// ──────────────────────────── State ────────────────────────────

/// Per-decode GPU scratch + KV cache (one slot per layer — every Cohere2 layer
/// is attention). Buffers are eager-allocated.
pub struct Cohere2MoeState {
    pub kv: KvCache,
    pub pos_buf: hip_bridge::DeviceBuffer, // device i32 position scalar
    pub max_seq: usize,
    pub n_tokens: usize,

    // residual + parallel-block normed input
    pub h: GpuTensor,      // [hidden] residual stream
    pub normed: GpuTensor, // [hidden] input_rmsnorm(h) — fed to BOTH branches

    // attention scratch
    pub fa_q: GpuTensor,        // [q_dim]
    pub fa_k: GpuTensor,        // [kv_dim]
    pub fa_v: GpuTensor,        // [kv_dim]
    pub fa_attn_out: GpuTensor, // [q_dim]

    // dense-ffn scratch (layer 0)
    pub dense_gate: GpuTensor, // [dense_inter]
    pub dense_up: GpuTensor,   // [dense_inter]
    pub dense_act: GpuTensor,  // [dense_inter] silu(gate)*up

    // moe scratch
    pub ffn_x_rot: GpuTensor, // [hidden] FWHT(normed) for MQ4/MQ6 experts
    pub router_logits: GpuTensor, // [n_exp]
    pub topk_indices: GpuTensor, // [k_top] i32-in-F32
    pub topk_weights: GpuTensor, // [k_top]
    pub gate_batch: GpuTensor, // [k_top*moe_inter]
    pub up_batch: GpuTensor,  // [k_top*moe_inter]
    pub rot_batch: GpuTensor, // [k_top*moe_inter]
    pub down_expanded: GpuTensor, // [k_top*hidden]
    /// Per-expert scratch for the F16/Q8 (non-indexed) MoE path.
    pub expert_gate_up: GpuTensor, // [2*moe_inter]
    pub expert_act: GpuTensor, // [moe_inter] silu(gate)*up
    pub expert_down: GpuTensor, // [hidden]

    // head
    pub final_norm_buf: GpuTensor, // [hidden]
    pub logits: GpuTensor,         // [vocab]
    /// Flash-attention online-softmax partials, [n_heads · ceil(max_seq/128) ·
    /// (2+head_dim) · FLASH_PREFILL_SUBBATCH]. Enables the tiled (O(1)-LDS) flash
    /// Q8 attention used for BOTH decode and prefill — no seq-bound shared-memory
    /// ceiling, so long-context (file reads) no longer crashes the LDS-bound
    /// legacy kernel. The trailing factor IS the batched-prefill flash sub-batch
    /// size at full context (the wrapper computes sub_batch = capacity/per_pos =
    /// factor·max_tiles_alloc / max_tiles_actual): at full context that is the
    /// factor itself, so a too-small factor serializes long prefill to 1 query
    /// per launch. FLASH_PREFILL_SUBBATCH=64 keeps it batched (~68 MB for North).
    pub flash_partials: GpuTensor,
}

/// Batched-prefill flash sub-batch size at full context — the trailing factor of
/// the `flash_partials` allocation. Larger = fewer, bigger flash launches during
/// long prefill (see `flash_partials` doc); 64 ≈ 68 MB for North-Mini-Code.
const FLASH_PREFILL_SUBBATCH: usize = 64;

/// Default KV-cache window when the caller doesn't request one. North supports
/// `max_position_embeddings` = 500k, but the KV is allocated up front (~53 KB /
/// token), so we default GENEROUSLY but not to the full 500k (≈26 GB): 32k
/// (~1.7 GB) handles typical agentic long-context (large file reads) out of the
/// box. The daemon honours an explicit larger `max_seq` up to MAX_REQUESTED_SEQ
/// (512k) via `new_with_max_seq`. (The old 8k default was a bring-up-era cap.)
const DEFAULT_MAX_SEQ: usize = 32_768;

impl Cohere2MoeState {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let Cohere2MoeState {
            kv,
            pos_buf,
            max_seq: _,
            n_tokens: _,
            h,
            normed,
            fa_q,
            fa_k,
            fa_v,
            fa_attn_out,
            dense_gate,
            dense_up,
            dense_act,
            ffn_x_rot,
            router_logits,
            topk_indices,
            topk_weights,
            gate_batch,
            up_batch,
            rot_batch,
            down_expanded,
            expert_gate_up,
            expert_act,
            expert_down,
            final_norm_buf,
            logits,
            flash_partials,
        } = self;
        let _ = kv.free_gpu(gpu);
        let _ = gpu.hip.free(pos_buf);
        for t in [
            h,
            normed,
            fa_q,
            fa_k,
            fa_v,
            fa_attn_out,
            dense_gate,
            dense_up,
            dense_act,
            ffn_x_rot,
            router_logits,
            topk_indices,
            topk_weights,
            gate_batch,
            up_batch,
            rot_batch,
            down_expanded,
            expert_gate_up,
            expert_act,
            expert_down,
            final_norm_buf,
            logits,
            flash_partials,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }

    pub fn new(gpu: &mut Gpu, cfg: &Cohere2MoeConfig) -> Result<Self, String> {
        let max_seq = cfg.max_position_embeddings.min(DEFAULT_MAX_SEQ);
        Self::new_with_max_seq(gpu, cfg, max_seq)
    }

    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &Cohere2MoeConfig,
        max_seq: usize,
    ) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let dense_inter = cfg.dense_intermediate_size;
        let moe_inter = cfg.moe_intermediate_size;
        let n_exp = cfg.num_experts;
        let k = cfg.num_experts_per_tok;

        // FWHT sign LUT must exist before any rotate_x_mq / fused rotate kernel.
        gpu.ensure_mq_signs()
            .map_err(|e| format!("cohere2moe: ensure_mq_signs: {e:?}"))?;

        // One KV slot per layer (every layer is attention).
        let kv = KvCache::new_gpu_q8(
            gpu,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            max_seq,
        )
        .map_err(|e| format!("cohere2moe: kv cache: {e:?}"))?;
        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|e| format!("cohere2moe: pos_buf malloc: {e:?}"))?;

        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.alloc_tensor(&[n], DType::F32)
                .map_err(|e| format!("cohere2moe: alloc {label}: {e:?}"))
        };

        Ok(Cohere2MoeState {
            kv,
            pos_buf,
            max_seq,
            n_tokens: 0,
            h: alloc(gpu, hidden, "h")?,
            normed: alloc(gpu, hidden, "normed")?,
            fa_q: alloc(gpu, q_dim, "fa_q")?,
            fa_k: alloc(gpu, kv_dim, "fa_k")?,
            fa_v: alloc(gpu, kv_dim, "fa_v")?,
            fa_attn_out: alloc(gpu, q_dim, "fa_attn_out")?,
            dense_gate: alloc(gpu, dense_inter, "dense_gate")?,
            dense_up: alloc(gpu, dense_inter, "dense_up")?,
            dense_act: alloc(gpu, dense_inter, "dense_act")?,
            ffn_x_rot: alloc(gpu, hidden, "ffn_x_rot")?,
            router_logits: alloc(gpu, n_exp, "router_logits")?,
            topk_indices: alloc(gpu, k, "topk_indices")?,
            topk_weights: alloc(gpu, k, "topk_weights")?,
            gate_batch: alloc(gpu, k * moe_inter, "gate_batch")?,
            up_batch: alloc(gpu, k * moe_inter, "up_batch")?,
            rot_batch: alloc(gpu, k * moe_inter, "rot_batch")?,
            down_expanded: alloc(gpu, k * hidden, "down_expanded")?,
            expert_gate_up: alloc(gpu, 2 * moe_inter, "expert_gate_up")?,
            expert_act: alloc(gpu, moe_inter, "expert_act")?,
            expert_down: alloc(gpu, hidden, "expert_down")?,
            final_norm_buf: alloc(gpu, hidden, "final_norm_buf")?,
            logits: alloc(gpu, cfg.vocab_size, "logits")?,
            flash_partials: alloc(
                gpu,
                cfg.num_attention_heads
                    * ((max_seq + 127) / 128)
                    * (2 + cfg.head_dim)
                    * FLASH_PREFILL_SUBBATCH,
                "flash_partials",
            )?,
        })
    }

    /// Reset for a fresh conversation. Rewinds the KV cursor AND zeros the KV
    /// buffers. The cursor rewind alone is sufficient for correctness (cohere2moe
    /// is pure attention with no recurrent/compressed state — unlike lfm2moe's
    /// `conv_states` — so the next cold prefill overwrites every attended slot
    /// and the stale tail is never read), but zeroing the buffers makes the reset
    /// holistic: no prior-conversation KV can survive even under a future
    /// window/LCP edge. Every daemon `reset()` call site also clears
    /// `conversation_tokens`, so a zeroed slot can never be stale-LCP-reused.
    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.n_tokens = 0;
        self.kv
            .clear_gpu(gpu)
            .map_err(|e| format!("cohere2moe reset: clear kv: {e:?}"))?;
        Ok(())
    }
}
