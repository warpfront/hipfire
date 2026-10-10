// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 weights + per-decode state, one stack for every variant: the
//! dense 12B, E2B/E4B (per-layer embeddings, KV sharing, double-wide MLP
//! tails behind the strict topology contract in [`Gemma4Config`]) and the
//! 26B-A4B MoE (routed experts in parallel with every dense FFN). Multimodal
//! towers remain out of scope.

use crate::config::{Gemma4Config, LayerType};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, EmbeddingFormat, KvCache, WeightTensor};
use hipfire_runtime::weight_backend::load_embedding;
use hipfire_runtime::weight_store::upload_pooled_bytes;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Rows per batched forward.
pub(crate) const GEMMA4_FORWARD_BATCH_MAX: usize = 256;
/// Query rows the flash partials are sized for; batched attention launches
/// split larger batches into sub-batches of this many rows.
const FLASH_PARTIALS_ROWS: usize = 64;

// ───────────────────────── Allocation ledger ─────────────────────────

/// Every device allocation of one load, as second owner handles. A failed
/// load frees them all; a successful one forgets them (the published owners
/// hold the bytes). Neither `GpuTensor` nor `DeviceBuffer` has a `Drop`, so a
/// forgotten or dropped handle frees nothing.
pub(crate) struct Ledger<'g> {
    pub gpu: &'g mut Gpu,
    owners: Vec<GpuTensor>,
    ops: usize,
    /// Fail the n-th allocation (rollback tests).
    fail_at: Option<usize>,
}

/// A second owner handle of `tensor`'s allocation (a non-owning alias would
/// be refused by `free_tensor`).
fn owner_handle(tensor: &GpuTensor) -> GpuTensor {
    GpuTensor {
        // SAFETY: `DeviceBuffer` is plain data without `Drop`; exactly one of
        // the two handles is ever freed — the ledger's on rollback (the
        // partially built owners are then dropped unfreed), else the
        // published owner's.
        buf: unsafe { std::ptr::read(&tensor.buf) },
        shape: tensor.shape.clone(),
        dtype: tensor.dtype,
    }
}

impl<'g> Ledger<'g> {
    pub fn new(gpu: &'g mut Gpu, fail_at: Option<usize>) -> Self {
        Self {
            gpu,
            owners: Vec::new(),
            ops: 0,
            fail_at,
        }
    }

    /// Run one allocation and record its owner.
    pub fn alloc(
        &mut self,
        label: &str,
        f: impl FnOnce(&mut Gpu) -> hip_bridge::HipResult<GpuTensor>,
    ) -> Result<GpuTensor, String> {
        self.ops += 1;
        if self.fail_at == Some(self.ops) {
            return Err(format!(
                "gemma4: injected load fault at {label} (op {})",
                self.ops
            ));
        }
        let tensor = f(self.gpu).map_err(|e| format!("gemma4: alloc {label}: {e:?}"))?;
        self.owners.push(owner_handle(&tensor));
        Ok(tensor)
    }

    /// A zeroed F32 `[n]` tensor.
    pub fn zeros(&mut self, n: usize, label: &str) -> Result<GpuTensor, String> {
        self.alloc(label, |g| g.zeros(&[n], DType::F32))
    }

    /// Build a KV cache and record every tensor it owns.
    pub fn kv(
        &mut self,
        label: &str,
        f: impl FnOnce(&mut Gpu) -> hip_bridge::HipResult<KvCache>,
    ) -> Result<KvCache, String> {
        self.ops += 1;
        if self.fail_at == Some(self.ops) {
            return Err(format!(
                "gemma4: injected load fault at {label} (op {})",
                self.ops
            ));
        }
        let kv = f(self.gpu).map_err(|e| format!("gemma4: {label}: {e:?}"))?;
        let tensors = kv
            .k_gpu
            .iter()
            .chain(&kv.v_gpu)
            .chain(&kv.k_scales)
            .chain(&kv.v_scales)
            .chain(&kv.givens_cos)
            .chain(&kv.givens_sin);
        self.owners.extend(tensors.map(owner_handle));
        Ok(kv)
    }

    /// Free every recorded owner and hand the GPU back.
    pub fn rollback(self) -> &'g mut Gpu {
        for tensor in self.owners.into_iter().rev() {
            let _ = self.gpu.free_tensor(tensor);
        }
        self.gpu
    }
}

// ───────────────────────── HFQ load helpers ─────────────────────────

fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and any byte pattern is a valid u8.
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
}

/// Decode a shape-[n] F16/F32/BF16 tensor into an F32 host Vec.
fn load_f32_vec(hfq: &HfqFile, name: &str, expected_n: usize) -> Result<Vec<f32>, String> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| format!("gemma4: tensor not found: {name}"))?;
    let n: usize = info.shape.iter().map(|&s| s as usize).product();
    if expected_n != 0 && n != expected_n {
        return Err(format!(
            "gemma4: shape mismatch for {name}: expected {expected_n}, got {n}"
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
        qt => return Err(format!("gemma4: expected F16/F32 for {name}, got qt={qt}")),
    };
    Ok(f32_data)
}

/// Load a Gemma 4 RMSNorm weight — plain `x * w` form (NO +1 shift), unless
/// `plus_one` (`HIPFIRE_GEMMA4_NORM_PLUS_ONE=1`) bakes the Gemma-2/3
/// `x * (1 + w)` convention at load time.
fn load_norm(
    l: &mut Ledger,
    hfq: &HfqFile,
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
    l.alloc(name, |g| g.upload_f32(&f32_data, &[dim]))
}

/// Load the learned per-layer `layer_scalar` (shape-[1]); returns the host f32
/// value so decode can scale with no D2H round-trip. Returns `1.0` (no-op)
/// when the tensor is absent — the 12B may not ship it.
fn load_layer_scalar(hfq: &HfqFile, name: &str) -> f32 {
    match load_f32_vec(hfq, name, 1) {
        Ok(v) => v[0],
        Err(_) => 1.0,
    }
}

/// Quantized projection formats. 19 and 30 are Lloyd formats now (the old
/// MG4 aliases), which no Gemma 4 kernel reads, so they refuse.
fn quantized_dtype(quant_type: u8) -> Option<DType> {
    Some(match quant_type {
        3 => DType::Q8_0,
        4 => DType::Q4K,
        6 => DType::HFQ4G256,
        7 => DType::HFQ4G128,
        8 => DType::HFQ6G256,
        9 => DType::HFQ2G256,
        10 => DType::HFQ2G128,
        11 => DType::HFQ3G256,
        12 => DType::HFQ3G128,
        13 => DType::MQ4G256,
        14 => DType::MQ8G256,
        15 => DType::MQ6G256,
        17 => DType::MQ3G256,
        18 => DType::MQ2G256,
        44 => DType::MQ4G256V2,
        _ => return None,
    })
}

/// HFQ4-G128 rows each start a fresh 72-byte group; a trailing partial group
/// is padded. Files quantized before the 2-D packer (`b4846285e`) packed
/// groups across rows whenever `K % 128 != 0` (Gemma 4 expert `down_proj`,
/// K = 704), which every per-row kernel reads as garbage. Refuse them.
fn check_hfq4g128_rows(name: &str, m: usize, k: usize, bytes: usize) -> Result<(), String> {
    let rows = m * k.div_ceil(128) * 72;
    if bytes == rows {
        return Ok(());
    }
    let reason = if bytes == (m * k).div_ceil(128) * 72 {
        "packs HFQ4-G128 groups across rows (quantized before hipfire-quantize \
         b4846285e); requantize the model"
    } else {
        "has an unexpected byte size"
    };
    Err(format!(
        "{name} [{m} x {k}] {reason} ({bytes} bytes, expected {rows})"
    ))
}

/// AWQ per-channel scale sidecar (`<weight>.awq_scale.weight`, F16 `[k]`).
/// Absent or malformed sidecars are skipped; a present, well-formed one whose
/// upload fails fails the load instead of silently computing `(W·s)·x`.
fn load_awq_scale(
    l: &mut Ledger,
    hfq: &HfqFile,
    weight_name: &str,
    k: usize,
) -> Result<Option<GpuTensor>, String> {
    let sidecar_name = match weight_name.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.awq_scale.weight"),
        None => format!("{weight_name}.awq_scale.weight"),
    };
    let Some((info, data)) = hfq.tensor_data_vec(&sidecar_name) else {
        return Ok(None);
    };
    if info.quant_type != 1 || info.shape.len() != 1 || info.shape[0] as usize != k {
        eprintln!(
            "warning: AWQ sidecar {sidecar_name} (qt={}, shape {:?}) is not F16 [{k}]; skipping",
            info.quant_type, info.shape
        );
        return Ok(None);
    }
    let scale: Vec<f32> = data
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let bytes = f32_bytes(&scale);
    l.alloc(&sidecar_name, |g| {
        upload_pooled_bytes(g, bytes, &[bytes.len()])
    })
    .map(Some)
}

/// Load one `[m, k]` projection weight (plus its AWQ sidecar).
fn load_wt(
    l: &mut Ledger,
    hfq: &HfqFile,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| format!("gemma4: tensor not found: {name}"))?;
    let widen = |bf16: bool| -> Vec<f32> {
        data.chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                if bf16 {
                    f32::from_bits((bits as u32) << 16)
                } else {
                    f16_to_f32(bits)
                }
            })
            .collect()
    };
    let (buf, gpu_dtype) = match info.quant_type {
        // BF16 stays BF16 for the calibration teacher's MFMA GEMM.
        16 if rdna_compute::calib_force_bf16() => (
            l.alloc(name, |g| upload_pooled_bytes(g, data, &[m, k]))?,
            DType::BF16,
        ),
        qt @ (1 | 16) => {
            let values = widen(qt == 16);
            let bytes = f32_bytes(&values);
            (
                l.alloc(name, |g| upload_pooled_bytes(g, bytes, &[m, k]))?,
                DType::F32,
            )
        }
        2 => (
            l.alloc(name, |g| upload_pooled_bytes(g, data, &[m, k]))?,
            DType::F32,
        ),
        qt => {
            let dtype = quantized_dtype(qt)
                .ok_or_else(|| format!("gemma4: unsupported quant_type {qt} for {name}"))?;
            if dtype == DType::HFQ4G128 {
                check_hfq4g128_rows(name, m, k, data.len())?;
            }
            (
                l.alloc(name, |g| upload_pooled_bytes(g, data, &[data.len()]))?,
                dtype,
            )
        }
    };
    let awq_scale = if gpu_dtype.supports_awq_sidecar() {
        load_awq_scale(l, hfq, name, k)?
    } else {
        None
    };
    Ok(WeightTensor {
        buf,
        gpu_dtype,
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

/// A tied weight view over `table` (no ownership: the table frees the bytes).
fn tied_head(table: &GpuTensor, format: EmbeddingFormat, m: usize, k: usize) -> WeightTensor {
    let dtype = hipfire_runtime::weight_backend::embedding_format_dtype(format);
    WeightTensor {
        buf: GpuTensor {
            // SAFETY: `embed_tokens` owns the allocation and outlives the head.
            buf: unsafe { table.buf.alias() },
            shape: table.shape.clone(),
            dtype,
        },
        gpu_dtype: dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

// ──────────────────────────── Weights ────────────────────────────

/// Per-layer weights for a SLIDING layer (head_dim 256, full RoPE, own v_proj).
pub struct SlidingLayerWeights {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    pub pre_feedforward_layernorm: GpuTensor,
    pub post_feedforward_layernorm: GpuTensor,
    pub layer_scalar_host: f32,

    pub q_proj: WeightTensor,
    pub k_proj: WeightTensor,
    pub v_proj: WeightTensor,
    pub o_proj: WeightTensor,
    pub q_norm: GpuTensor, // [head_dim]
    pub k_norm: GpuTensor, // [head_dim]

    pub gate_proj: WeightTensor,
    pub up_proj: WeightTensor,
    pub down_proj: WeightTensor,
    pub ffn_hidden_dim: usize,
    pub per_layer: Option<PerLayerBranchWeights>,
    pub moe: Option<MoeLayerExtras>,
}

/// Per-layer weights for a FULL layer (head_dim 512, partial RoPE).
///
/// `attention_k_eq_v == true` (12B, 26B): no `v_proj`; V is the PRE-k_norm
/// output of k_proj, renormed by a weight-less (ones) RMSNorm. When false
/// (E-series), a separate `v_proj` is loaded.
pub struct FullLayerWeights {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    pub pre_feedforward_layernorm: GpuTensor,
    pub post_feedforward_layernorm: GpuTensor,
    pub layer_scalar_host: f32,

    pub q_proj: WeightTensor,
    pub k_proj: WeightTensor,
    pub v_proj: Option<WeightTensor>, // None when attention_k_eq_v
    pub o_proj: WeightTensor,
    pub q_norm: GpuTensor, // [head_dim]
    pub k_norm: GpuTensor, // [head_dim]
    // no v_norm weight — v is no-scale (ones buffer passed at decode time)
    pub gate_proj: WeightTensor,
    pub up_proj: WeightTensor,
    pub down_proj: WeightTensor,
    pub ffn_hidden_dim: usize,
    pub per_layer: Option<PerLayerBranchWeights>,
    pub moe: Option<MoeLayerExtras>,
}

pub enum LayerWeights {
    Sliding(SlidingLayerWeights),
    Full(FullLayerWeights),
}

impl LayerWeights {
    pub fn moe(&self) -> Option<&MoeLayerExtras> {
        match self {
            LayerWeights::Sliding(l) => l.moe.as_ref(),
            LayerWeights::Full(l) => l.moe.as_ref(),
        }
    }
}

/// Per-layer residual branch driven by the token-level PLE source.
pub struct PerLayerBranchWeights {
    pub input_gate: WeightTensor,
    pub projection: WeightTensor,
    pub post_input_norm: GpuTensor,
}

/// Token-level PLE weights shared by all decoder layers.
pub struct PerLayerInputWeights {
    pub embed_tokens: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub model_projection: WeightTensor,
    pub projection_norm: GpuTensor,
}

/// The routed-expert branch of one MoE layer (26B-A4B). Each expert kind lives
/// in one pooled allocation (one hipMalloc per kind and layer instead of one
/// per expert, which fragmented the heap); the indexed kernels find expert
/// `e` through the `[n_experts]` u64 device pointer tables.
pub struct MoeLayerExtras {
    /// `[n_experts, dim]` router projection.
    pub router_proj: WeightTensor,
    /// `[dim]` multiplicative router-input scale (`router.scale`).
    pub router_scale: GpuTensor,
    /// `[n_experts]` per-expert post-`down_proj` scale.
    pub per_expert_scale: GpuTensor,
    /// `[dim]` RMSNorm on the expert input.
    pub pre_feedforward_layernorm_2: GpuTensor,
    /// `[dim]` RMSNorm on the dense FFN output before the sum.
    pub post_feedforward_layernorm_1: GpuTensor,
    /// `[dim]` RMSNorm on the expert output before the sum.
    pub post_feedforward_layernorm_2: GpuTensor,
    pub n_experts: usize,
    pub gate_up_dtype: DType,
    pub down_dtype: DType,
    pub experts_gate_up_pool: GpuTensor,
    pub experts_down_pool: GpuTensor,
    pub experts_gate_up_ptrs: GpuTensor,
    pub experts_down_ptrs: GpuTensor,
}

/// Every `{p}.experts.{x}.{kind}.weight` of one layer in a single pooled
/// allocation; returns `(pool, dtype, bytes_per_expert)`.
fn load_expert_pool(
    l: &mut Ledger,
    hfq: &HfqFile,
    p: &str,
    kind: &str,
    n_experts: usize,
) -> Result<(GpuTensor, DType, usize), String> {
    let first_name = format!("{p}.experts.0.{kind}.weight");
    let (first_info, first_data) = hfq
        .tensor_data(&first_name)
        .ok_or_else(|| format!("gemma4: MoE expert tensor not found: {first_name}"))?;
    let bytes_per_expert = first_data.len();
    let dtype = quantized_dtype(first_info.quant_type).ok_or_else(|| {
        format!(
            "gemma4: unsupported MoE expert quant_type {} for {first_name}",
            first_info.quant_type
        )
    })?;
    if dtype == DType::HFQ4G128 {
        let [m, k] = first_info.shape[..] else {
            return Err(format!(
                "{first_name}: expected a 2-D shape, got {:?}",
                first_info.shape
            ));
        };
        check_hfq4g128_rows(&first_name, m as usize, k as usize, bytes_per_expert)?;
    }
    let mut concat = Vec::with_capacity(bytes_per_expert * n_experts);
    concat.extend_from_slice(first_data);
    for x in 1..n_experts {
        let name = format!("{p}.experts.{x}.{kind}.weight");
        let (info, data) = hfq
            .tensor_data(&name)
            .ok_or_else(|| format!("gemma4: MoE expert tensor not found: {name}"))?;
        if data.len() != bytes_per_expert || info.quant_type != first_info.quant_type {
            return Err(format!(
                "gemma4: MoE expert {name} (qt={}, {} bytes) differs from expert 0 (qt={}, {bytes_per_expert} bytes)",
                info.quant_type,
                data.len(),
                first_info.quant_type
            ));
        }
        concat.extend_from_slice(data);
    }
    let pool = l.alloc(&format!("{p} {kind} pool"), |g| {
        upload_pooled_bytes(g, &concat, &[concat.len()])
    })?;
    Ok((pool, dtype, bytes_per_expert))
}

fn load_moe_layer_extras(
    l: &mut Ledger,
    hfq: &HfqFile,
    cfg: &Gemma4Config,
    p: &str,
    plus_one: bool,
) -> Result<MoeLayerExtras, String> {
    let n_exp = cfg.num_experts;
    let dim = cfg.dim;
    let router_proj = load_wt(l, hfq, &format!("{p}.router.proj.weight"), n_exp, dim)?;
    // `router.scale` and `router.per_expert_scale` ship WITHOUT the `.weight`
    // suffix in HF's 26B-A4B safetensors. Neither is an RMSNorm weight.
    let router_scale = load_norm(l, hfq, &format!("{p}.router.scale"), dim, false)?;
    let per_expert = load_f32_vec(hfq, &format!("{p}.router.per_expert_scale"), n_exp)?;
    let per_expert_bytes = f32_bytes(&per_expert);
    let per_expert_scale = l.alloc("per_expert_scale", |g| {
        upload_pooled_bytes(g, per_expert_bytes, &[n_exp])
    })?;
    let norm =
        |l: &mut Ledger, rel: &str| load_norm(l, hfq, &format!("{p}.{rel}.weight"), dim, plus_one);
    let pre_feedforward_layernorm_2 = norm(l, "pre_feedforward_layernorm_2")?;
    let post_feedforward_layernorm_1 = norm(l, "post_feedforward_layernorm_1")?;
    let post_feedforward_layernorm_2 = norm(l, "post_feedforward_layernorm_2")?;
    let (gate_up_pool, gate_up_dtype, gate_up_bpe) =
        load_expert_pool(l, hfq, p, "gate_up_proj", n_exp)?;
    let (down_pool, down_dtype, down_bpe) = load_expert_pool(l, hfq, p, "down_proj", n_exp)?;
    // Pool allocations never move, so the device pointer tables stay valid
    // for the model's lifetime.
    let ptrs = |pool: &GpuTensor, bpe: usize| -> Vec<u8> {
        (0..n_exp)
            .flat_map(|x| (pool.buf.as_ptr() as u64 + (x * bpe) as u64).to_ne_bytes())
            .collect()
    };
    let gate_up_ptr_bytes = ptrs(&gate_up_pool, gate_up_bpe);
    let down_ptr_bytes = ptrs(&down_pool, down_bpe);
    // Each u64 occupies two f32 slots; the kernels cast the buffer to u64*.
    let experts_gate_up_ptrs = l.alloc("gate_up ptrs", |g| {
        upload_pooled_bytes(g, &gate_up_ptr_bytes, &[n_exp * 2])
    })?;
    let experts_down_ptrs = l.alloc("down ptrs", |g| {
        upload_pooled_bytes(g, &down_ptr_bytes, &[n_exp * 2])
    })?;
    Ok(MoeLayerExtras {
        router_proj,
        router_scale,
        per_expert_scale,
        pre_feedforward_layernorm_2,
        post_feedforward_layernorm_1,
        post_feedforward_layernorm_2,
        n_experts: n_exp,
        gate_up_dtype,
        down_dtype,
        experts_gate_up_pool: gate_up_pool,
        experts_down_pool: down_pool,
        experts_gate_up_ptrs,
        experts_down_ptrs,
    })
}

fn load_per_layer_branch(
    l: &mut Ledger,
    hfq: &HfqFile,
    cfg: &Gemma4Config,
    prefix: &str,
    plus_one: bool,
) -> Result<Option<PerLayerBranchWeights>, String> {
    let ple_dim = cfg.hidden_size_per_layer_input;
    if ple_dim == 0 {
        return Ok(None);
    }
    Ok(Some(PerLayerBranchWeights {
        input_gate: load_wt(
            l,
            hfq,
            &format!("{prefix}.per_layer_input_gate.weight"),
            ple_dim,
            cfg.dim,
        )?,
        projection: load_wt(
            l,
            hfq,
            &format!("{prefix}.per_layer_projection.weight"),
            cfg.dim,
            ple_dim,
        )?,
        post_input_norm: load_norm(
            l,
            hfq,
            &format!("{prefix}.post_per_layer_input_norm.weight"),
            cfg.dim,
            plus_one,
        )?,
    }))
}

fn load_layer(
    l: &mut Ledger,
    hfq: &HfqFile,
    cfg: &Gemma4Config,
    name_root: &str,
    i: usize,
) -> Result<LayerWeights, String> {
    let dim = cfg.dim;
    let plus_one = cfg.norm_plus_one;
    let p = format!("{name_root}.layers.{i}");
    let ffn_hd = cfg.ffn_hidden_dim_for_layer(i);
    let (hd, n_kv) = match cfg.layer_types[i] {
        LayerType::Sliding => (cfg.sliding_head_dim, cfg.sliding_n_kv_heads),
        LayerType::Full => (cfg.full_head_dim, cfg.full_n_kv_heads),
    };
    let kv_dim = n_kv * hd;
    let q_dim = cfg.n_heads * hd;
    let norm = |l: &mut Ledger, rel: &str, n: usize| {
        load_norm(l, hfq, &format!("{p}.{rel}.weight"), n, plus_one)
    };
    let wt = |l: &mut Ledger, rel: &str, m: usize, k: usize| {
        load_wt(l, hfq, &format!("{p}.{rel}.weight"), m, k)
    };
    let input_layernorm = norm(l, "input_layernorm", dim)?;
    let post_attention_layernorm = norm(l, "post_attention_layernorm", dim)?;
    let pre_feedforward_layernorm = norm(l, "pre_feedforward_layernorm", dim)?;
    let post_feedforward_layernorm = norm(l, "post_feedforward_layernorm", dim)?;
    let layer_scalar_host = load_layer_scalar(hfq, &format!("{p}.layer_scalar"));
    let q_proj = wt(l, "self_attn.q_proj", q_dim, dim)?;
    let k_proj = wt(l, "self_attn.k_proj", kv_dim, dim)?;
    let has_v = cfg.layer_types[i] == LayerType::Sliding || !cfg.attention_k_eq_v;
    let v_proj = if has_v {
        Some(wt(l, "self_attn.v_proj", kv_dim, dim)?)
    } else {
        None
    };
    let o_proj = wt(l, "self_attn.o_proj", dim, q_dim)?;
    let q_norm = norm(l, "self_attn.q_norm", hd)?;
    let k_norm = norm(l, "self_attn.k_norm", hd)?;
    let gate_proj = wt(l, "mlp.gate_proj", ffn_hd, dim)?;
    let up_proj = wt(l, "mlp.up_proj", ffn_hd, dim)?;
    let down_proj = wt(l, "mlp.down_proj", dim, ffn_hd)?;
    let per_layer = load_per_layer_branch(l, hfq, cfg, &p, plus_one)?;
    let moe = if cfg.enable_moe_block {
        Some(load_moe_layer_extras(l, hfq, cfg, &p, plus_one)?)
    } else {
        None
    };
    Ok(match (cfg.layer_types[i], v_proj) {
        (LayerType::Sliding, Some(v_proj)) => LayerWeights::Sliding(SlidingLayerWeights {
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            layer_scalar_host,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            gate_proj,
            up_proj,
            down_proj,
            ffn_hidden_dim: ffn_hd,
            per_layer,
            moe,
        }),
        (_, v_proj) => LayerWeights::Full(FullLayerWeights {
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            layer_scalar_host,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            gate_proj,
            up_proj,
            down_proj,
            ffn_hidden_dim: ffn_hd,
            per_layer,
            moe,
        }),
    })
}

pub struct Gemma4Weights {
    /// Token embedding [vocab, dim]; aliased as lm_head when tied.
    pub embed_tokens: GpuTensor,
    pub embd_format: EmbeddingFormat,
    /// LM head (shares bytes with `embed_tokens` when tied).
    pub lm_head: WeightTensor,
    pub final_norm: GpuTensor,
    pub per_layer_input: Option<PerLayerInputWeights>,
    pub layers: Vec<LayerWeights>,
}

impl Gemma4Weights {
    pub fn load(hfq: &HfqFile, cfg: &Gemma4Config, gpu: &mut Gpu) -> Result<Self, String> {
        Self::load_with_fault(hfq, cfg, gpu, None)
    }

    /// [`Self::load`] that fails its `fail_at`-th allocation: every staged
    /// owner, completed layers and AWQ sidecars included, is freed.
    pub(crate) fn load_with_fault(
        hfq: &HfqFile,
        cfg: &Gemma4Config,
        gpu: &mut Gpu,
        fail_at: Option<usize>,
    ) -> Result<Self, String> {
        if cfg.hidden_size_per_layer_input != 0 || cfg.num_kv_shared_layers != 0 {
            cfg.e_series_variant()?;
        }
        let mut l = Ledger::new(gpu, fail_at);
        match Self::load_into(&mut l, hfq, cfg) {
            Ok(weights) => Ok(weights),
            Err(e) => {
                l.rollback();
                Err(e)
            }
        }
    }

    fn load_into(l: &mut Ledger, hfq: &HfqFile, cfg: &Gemma4Config) -> Result<Self, String> {
        let dim = cfg.dim;
        let plus_one = cfg.norm_plus_one;
        let name_root = ["model.language_model", "language_model", "model"]
            .into_iter()
            .find(|root| {
                hfq.tensor_data(&format!("{root}.embed_tokens.weight"))
                    .is_some()
            })
            .ok_or_else(|| {
                "gemma4: embed_tokens not found under model.language_model, language_model, or model"
                    .to_string()
            })?;

        let (embed_info, embed_data) = hfq
            .tensor_data(&format!("{name_root}.embed_tokens.weight"))
            .ok_or_else(|| "gemma4: embed_tokens not found in HFQ".to_string())?;
        let mut embd_format = EmbeddingFormat::F32;
        let embed_tokens = l.alloc("embed_tokens", |g| {
            load_embedding(g, embed_info.quant_type, embed_data, cfg.vocab_size, dim).map(
                |(table, format)| {
                    embd_format = format;
                    table
                },
            )
        })?;
        let lm_head = tied_head(&embed_tokens, embd_format, cfg.vocab_size, dim);
        let final_norm = load_norm(l, hfq, &format!("{name_root}.norm.weight"), dim, plus_one)?;

        let per_layer_input = if cfg.hidden_size_per_layer_input != 0 {
            let ple_dim = cfg.hidden_size_per_layer_input;
            let packed_dim = cfg.n_layers * ple_dim;
            let name = format!("{name_root}.embed_tokens_per_layer.weight");
            let (info, data) = hfq
                .tensor_data(&name)
                .ok_or_else(|| format!("gemma4: tensor not found: {name}"))?;
            let mut ple_format = EmbeddingFormat::F32;
            let embed_tokens = l.alloc(&name, |g| {
                load_embedding(
                    g,
                    info.quant_type,
                    data,
                    cfg.vocab_size_per_layer_input,
                    packed_dim,
                )
                .map(|(table, format)| {
                    ple_format = format;
                    table
                })
            })?;
            Some(PerLayerInputWeights {
                embed_tokens,
                embd_format: ple_format,
                model_projection: load_wt(
                    l,
                    hfq,
                    &format!("{name_root}.per_layer_model_projection.weight"),
                    packed_dim,
                    dim,
                )?,
                projection_norm: load_norm(
                    l,
                    hfq,
                    &format!("{name_root}.per_layer_projection_norm.weight"),
                    ple_dim,
                    plus_one,
                )?,
            })
        } else {
            None
        };

        let layers = (0..cfg.n_layers)
            .map(|i| load_layer(l, hfq, cfg, name_root, i))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Gemma4Weights {
            embed_tokens,
            embd_format,
            lm_head,
            final_norm,
            per_layer_input,
            layers,
        })
    }

    /// Return all GPU weight buffers to the pool. Consumes self.
    ///
    /// `lm_head` is NOT freed here: it aliases `embed_tokens`' allocation
    /// (tied embeddings), which is freed exactly once below.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.embed_tokens);
        let _ = gpu.free_tensor(self.final_norm);
        if let Some(ple) = self.per_layer_input {
            let _ = gpu.free_tensor(ple.embed_tokens);
            ple.model_projection.free_all(gpu);
            let _ = gpu.free_tensor(ple.projection_norm);
        }
        let free_branch = |gpu: &mut Gpu, ple: Option<PerLayerBranchWeights>| {
            if let Some(ple) = ple {
                ple.input_gate.free_all(gpu);
                ple.projection.free_all(gpu);
                let _ = gpu.free_tensor(ple.post_input_norm);
            }
        };
        let free_moe = |gpu: &mut Gpu, moe: Option<MoeLayerExtras>| {
            if let Some(moe) = moe {
                moe.router_proj.free_all(gpu);
                for t in [
                    moe.router_scale,
                    moe.per_expert_scale,
                    moe.pre_feedforward_layernorm_2,
                    moe.post_feedforward_layernorm_1,
                    moe.post_feedforward_layernorm_2,
                    moe.experts_gate_up_pool,
                    moe.experts_down_pool,
                    moe.experts_gate_up_ptrs,
                    moe.experts_down_ptrs,
                ] {
                    let _ = gpu.free_tensor(t);
                }
            }
        };
        for l in self.layers {
            match l {
                LayerWeights::Sliding(l) => {
                    for t in [
                        l.input_layernorm,
                        l.post_attention_layernorm,
                        l.pre_feedforward_layernorm,
                        l.post_feedforward_layernorm,
                        l.q_norm,
                        l.k_norm,
                    ] {
                        let _ = gpu.free_tensor(t);
                    }
                    for w in [
                        l.q_proj,
                        l.k_proj,
                        l.v_proj,
                        l.o_proj,
                        l.gate_proj,
                        l.up_proj,
                        l.down_proj,
                    ] {
                        w.free_all(gpu);
                    }
                    free_branch(gpu, l.per_layer);
                    free_moe(gpu, l.moe);
                }
                LayerWeights::Full(l) => {
                    for t in [
                        l.input_layernorm,
                        l.post_attention_layernorm,
                        l.pre_feedforward_layernorm,
                        l.post_feedforward_layernorm,
                        l.q_norm,
                        l.k_norm,
                    ] {
                        let _ = gpu.free_tensor(t);
                    }
                    for w in [
                        l.q_proj,
                        l.k_proj,
                        l.o_proj,
                        l.gate_proj,
                        l.up_proj,
                        l.down_proj,
                    ]
                    .into_iter()
                    .chain(l.v_proj)
                    {
                        w.free_all(gpu);
                    }
                    free_branch(gpu, l.per_layer);
                    free_moe(gpu, l.moe);
                }
            }
        }
    }
}

// ──────────────────────────── State ────────────────────────────

/// Storage of the full-attention KV tier. The sliding tier is always Q8.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FullKvTier {
    Q8,
    /// FWHT-512 3-bit K + Q8_0 V (developer examples only).
    Fwht3,
    /// Deprecated Givens asym3 (`kv_cache=legacy-asym3`).
    LegacyAsym3,
}

/// Single-row routed-expert scratch of a MoE checkpoint; batched forwards
/// route row by row through it.
pub struct MoeScratch {
    pub cur_mlp: GpuTensor,       // [dim] post_norm_1(dense FFN out)
    pub pre2: GpuTensor,          // [dim] expert input
    pub pre2_rot: GpuTensor,      // [dim] FWHT-rotated expert input
    pub router_in: GpuTensor,     // [dim]
    pub router_logits: GpuTensor, // [n_experts]
    pub topk_indices: GpuTensor,  // [top_k] i32 in f32 slots
    pub topk_weights: GpuTensor,  // [top_k]
    pub cur_moe: GpuTensor,       // [dim] expert accumulator
    pub gate_batch: GpuTensor,    // [top_k * mi]
    pub up_batch: GpuTensor,      // [top_k * mi]
    pub hidden_batch: GpuTensor,  // [top_k * mi]
}

impl MoeScratch {
    fn new(l: &mut Ledger, cfg: &Gemma4Config) -> Result<Self, String> {
        let dim = cfg.dim;
        let k_mi = cfg.top_k_experts * cfg.moe_intermediate_size;
        Ok(Self {
            cur_mlp: l.zeros(dim, "moe cur_mlp")?,
            pre2: l.zeros(dim, "moe pre2")?,
            pre2_rot: l.zeros(dim, "moe pre2_rot")?,
            router_in: l.zeros(dim, "moe router_in")?,
            router_logits: l.zeros(cfg.num_experts, "moe router_logits")?,
            topk_indices: l.zeros(cfg.top_k_experts, "moe topk_indices")?,
            topk_weights: l.zeros(cfg.top_k_experts, "moe topk_weights")?,
            cur_moe: l.zeros(dim, "moe cur_moe")?,
            gate_batch: l.zeros(k_mi, "moe gate_batch")?,
            up_batch: l.zeros(k_mi, "moe up_batch")?,
            hidden_batch: l.zeros(k_mi, "moe hidden_batch")?,
        })
    }

    fn tensors(self) -> [GpuTensor; 11] {
        [
            self.cur_mlp,
            self.pre2,
            self.pre2_rot,
            self.router_in,
            self.router_logits,
            self.topk_indices,
            self.topk_weights,
            self.cur_moe,
            self.gate_batch,
            self.up_batch,
            self.hidden_batch,
        ]
    }
}

/// Per-decode GPU scratch + the two KV caches (sliding + full).
///
/// Scratch is sized once against the MAX of sliding/full attention dims so a
/// single buffer set works across both layer types. `kv_slot_for_layer` maps
/// each layer ordinal to its slot within the per-type cache. Both caches are
/// position-indexed (`physical_cap == max_seq`, no ring): a row below the
/// current position is never rewritten, which speculative verify and session
/// snapshots rely on.
pub struct Gemma4State {
    /// Sliding-window KV cache (Q8), one slot per sliding layer.
    pub kv_sliding: KvCache,
    /// Full-attention KV cache, one slot per full layer.
    pub kv_full: KvCache,
    /// Per-layer slot index into the matching per-type cache.
    pub kv_slot_for_layer: Vec<usize>,

    /// Device i32 position scalar.
    pub pos_buf: GpuTensor,
    /// Stable host source for the device position scalar. The hipGraph decode
    /// path captures a `memcpy_htod_auto` from these bytes; the captured node
    /// re-reads this heap-stable `Box` on every replay (see
    /// `decode_step_with_graph`). Updated host-side before each `graph_launch`.
    pub pos_host: Box<[i32]>,
    pub max_seq: usize,
    pub n_tokens: usize,
    /// hipGraph warmup gate: the first decode after a fresh load runs eager
    /// (no capture) to JIT-compile kernels + settle DPM, then the next call
    /// captures. Survives turn resets (the graph stays valid for the same model
    /// — only weight pointers + device buffers are baked, and those are stable).
    pub ar_warmed_up: bool,

    // residual stream + scratch
    pub x: GpuTensor,        // [dim]
    pub residual: GpuTensor, // [dim]
    pub tmp: GpuTensor,      // [dim] norm / o_proj scratch
    /// FWHT-rotated rmsnorm output for MagnumQuant projections. [max(dim, q)]
    pub tmp_rot: GpuTensor,

    // attention scratch (sized to max over layer types)
    pub q: GpuTensor,        // [max_q_dim]
    pub k: GpuTensor,        // [max_kv_dim]
    pub v: GpuTensor,        // [max_kv_dim]
    pub attn_out: GpuTensor, // [max_q_dim]
    /// Shared tile/reduce workspace for eager and batched attention. Sized
    /// for the larger attention geometry and the admitted forward batch cap.
    pub q8_flash_partials: GpuTensor,

    /// Ones-filled weight buffer for the weight-less V RMSNorm on full layers.
    pub v_norm_ones: GpuTensor, // [max_head_dim]

    // FFN scratch
    pub gate_ffn: GpuTensor,   // [hidden_dim]
    pub up_ffn: GpuTensor,     // [hidden_dim]
    pub ffn_hidden: GpuTensor, // [hidden_dim]
    pub ffn_out: GpuTensor,    // [dim]

    // Per-layer input (PLE) scratch. Dense and MoE checkpoints leave these absent.
    pub ple_token_inputs: Option<GpuTensor>, // [n_layers * ple_dim]
    pub ple_projection_all: Option<GpuTensor>, // [n_layers * ple_dim]
    pub ple_gate: Option<GpuTensor>,         // [ple_dim]
    pub ple_hidden: Option<GpuTensor>,       // [ple_dim]
    pub ple_out: Option<GpuTensor>,          // [dim]

    /// Routed-expert scratch; MoE checkpoints only.
    pub moe: Option<MoeScratch>,

    // head
    pub logits: GpuTensor, // [vocab]
}

fn build_kv_slot_map(cfg: &Gemma4Config) -> Result<Vec<usize>, String> {
    let mut slots = vec![usize::MAX; cfg.n_layers];
    let mut sliding = 0usize;
    let mut full = 0usize;
    for layer_idx in 0..cfg.n_layers {
        if let Some(source_layer) = cfg.kv_shared_source_layer_idx(layer_idx) {
            let source_slot = slots[source_layer];
            if source_slot == usize::MAX {
                return Err(format!(
                    "gemma4: KV source layer {source_layer} for layer {layer_idx} has no slot"
                ));
            }
            slots[layer_idx] = source_slot;
            continue;
        }
        if cfg.is_kv_shared_layer(layer_idx) {
            return Err(format!(
                "gemma4: shared KV layer {layer_idx} has no preceding same-type source"
            ));
        }
        match cfg.layer_types[layer_idx] {
            LayerType::Sliding => {
                slots[layer_idx] = sliding;
                sliding += 1;
            }
            LayerType::Full => {
                slots[layer_idx] = full;
                full += 1;
            }
        }
    }
    Ok(slots)
}

/// Single-query asym3 hd512 flash partials: `n_heads * ceil(max_seq/128) *
/// (2 + head_dim)` floats at the asym3 kernel's fixed 128-row tile.
fn asym3_flash_partials_len(cfg: &Gemma4Config, max_seq: usize) -> usize {
    cfg.n_heads * max_seq.div_ceil(128) * (2 + cfg.full_head_dim)
}

impl Gemma4State {
    pub fn new(gpu: &mut Gpu, cfg: &Gemma4Config) -> Result<Self, String> {
        // Cap the KV cache so the 262144-ctx config doesn't OOM.
        let max_seq = cfg.max_position_embeddings.min(8192);
        Self::new_with_max_seq(gpu, cfg, max_seq)
    }

    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &Gemma4Config,
        max_seq: usize,
    ) -> Result<Self, String> {
        Self::new_with_full_tier(gpu, cfg, max_seq, FullKvTier::Q8)
    }

    pub fn new_with_fwht3_max_seq(
        gpu: &mut Gpu,
        cfg: &Gemma4Config,
        max_seq: usize,
    ) -> Result<Self, String> {
        Self::new_with_full_tier(gpu, cfg, max_seq, FullKvTier::Fwht3)
    }

    pub fn new_with_full_tier(
        gpu: &mut Gpu,
        cfg: &Gemma4Config,
        max_seq: usize,
        tier: FullKvTier,
    ) -> Result<Self, String> {
        // FWHT sign LUT must exist before any fused_rmsnorm_rotate_mq /
        // fused_gate_up_hfq4g256 launch (the MQ4 fused FFN path).
        gpu.ensure_mq_signs()
            .map_err(|e| format!("gemma4: ensure_mq_signs: {e:?}"))?;
        let kv_slot_for_layer = build_kv_slot_map(cfg)?;
        let mut l = Ledger::new(gpu, None);
        match Self::alloc(&mut l, cfg, max_seq, tier, kv_slot_for_layer) {
            Ok(state) => Ok(state),
            Err(e) => {
                l.rollback();
                Err(e)
            }
        }
    }

    fn alloc(
        l: &mut Ledger,
        cfg: &Gemma4Config,
        max_seq: usize,
        tier: FullKvTier,
        kv_slot_for_layer: Vec<usize>,
    ) -> Result<Self, String> {
        let kv_sliding = l.kv("sliding kv cache", |g| {
            KvCache::new_gpu_q8(
                g,
                cfg.n_sliding_kv_slots(),
                cfg.sliding_n_kv_heads,
                cfg.sliding_head_dim,
                max_seq,
            )
        })?;
        let n_full = cfg.n_full_kv_slots();
        let (heads, hd) = (cfg.full_n_kv_heads, cfg.full_head_dim);
        let kv_full = l.kv("full kv cache", |g| match tier {
            FullKvTier::Q8 => KvCache::new_gpu_q8(g, n_full, heads, hd, max_seq),
            FullKvTier::Fwht3 => KvCache::new_gpu_fwht3_capped_filtered_gemma4(
                g,
                &vec![true; n_full],
                heads,
                hd,
                max_seq,
                max_seq,
            ),
            FullKvTier::LegacyAsym3 => KvCache::new_gpu_asym3_gemma4(g, n_full, heads, hd, max_seq),
        })?;
        let dim = cfg.dim;
        let max_q = cfg.max_q_dim();
        let max_kv = cfg.max_kv_dim();
        let max_hd = cfg.max_head_dim();
        let ple_dim = cfg.hidden_size_per_layer_input;
        let ple_packed = cfg.n_layers * ple_dim;
        let mut partials = crate::program::q8_flash_partials_len(
            l.gpu,
            &crate::program::Geometry::eager(cfg),
            max_seq,
            FLASH_PARTIALS_ROWS,
        );
        if tier != FullKvTier::Q8 {
            partials = partials.max(asym3_flash_partials_len(cfg, max_seq));
        }

        let pos_buf = l.alloc("pos_buf", |g| g.zeros(&[1], DType::F32))?;
        let v_norm_ones = l.alloc("v_norm_ones", |g| {
            g.upload_f32(&vec![1.0; max_hd], &[max_hd])
        })?;
        let x = l.zeros(dim, "x")?;
        let residual = l.zeros(dim, "residual")?;
        let tmp = l.zeros(dim, "tmp")?;
        let tmp_rot = l.zeros(dim.max(max_q), "tmp_rot")?;
        let q = l.zeros(max_q, "q")?;
        let k = l.zeros(max_kv, "k")?;
        let v = l.zeros(max_kv, "v")?;
        let attn_out = l.zeros(max_q, "attn_out")?;
        let q8_flash_partials = l.zeros(partials, "flash_partials")?;
        let gate_ffn = l.zeros(cfg.max_ffn_hidden_dim(), "gate_ffn")?;
        let up_ffn = l.zeros(cfg.max_ffn_hidden_dim(), "up_ffn")?;
        let ffn_hidden = l.zeros(cfg.max_ffn_hidden_dim(), "ffn_hidden")?;
        let ffn_out = l.zeros(dim, "ffn_out")?;
        let mut ple = |n: usize, label: &str| -> Result<Option<GpuTensor>, String> {
            (ple_dim != 0).then(|| l.zeros(n, label)).transpose()
        };
        let ple_token_inputs = ple(ple_packed, "ple_token_inputs")?;
        let ple_projection_all = ple(ple_packed, "ple_projection_all")?;
        let ple_gate = ple(ple_dim, "ple_gate")?;
        let ple_hidden = ple(ple_dim, "ple_hidden")?;
        let ple_out = ple(dim, "ple_out")?;
        let moe = if cfg.enable_moe_block {
            Some(MoeScratch::new(l, cfg)?)
        } else {
            None
        };
        let logits = l.zeros(cfg.vocab_size, "logits")?;
        Ok(Gemma4State {
            kv_sliding,
            kv_full,
            kv_slot_for_layer,
            pos_buf,
            pos_host: vec![0i32; 1].into_boxed_slice(),
            max_seq,
            n_tokens: 0,
            ar_warmed_up: false,
            x,
            residual,
            tmp,
            tmp_rot,
            q,
            k,
            v,
            attn_out,
            q8_flash_partials,
            v_norm_ones,
            gate_ffn,
            up_ffn,
            ffn_hidden,
            ffn_out,
            ple_token_inputs,
            ple_projection_all,
            ple_gate,
            ple_hidden,
            ple_out,
            moe,
            logits,
        })
    }

    pub fn reset(&mut self) {
        self.n_tokens = 0;
    }

    /// Every scratch buffer (not the KV caches), for whole-surface resets.
    pub fn scratch_tensors(&self) -> Vec<&GpuTensor> {
        let mut out = vec![
            &self.x,
            &self.residual,
            &self.tmp,
            &self.tmp_rot,
            &self.q,
            &self.k,
            &self.v,
            &self.attn_out,
            &self.q8_flash_partials,
            &self.gate_ffn,
            &self.up_ffn,
            &self.ffn_hidden,
            &self.ffn_out,
            &self.logits,
        ];
        out.extend(
            [
                &self.ple_token_inputs,
                &self.ple_projection_all,
                &self.ple_gate,
                &self.ple_hidden,
                &self.ple_out,
            ]
            .into_iter()
            .flatten(),
        );
        if let Some(m) = &self.moe {
            out.extend([
                &m.cur_mlp,
                &m.pre2,
                &m.pre2_rot,
                &m.router_in,
                &m.router_logits,
                &m.topk_indices,
                &m.topk_weights,
                &m.cur_moe,
                &m.gate_batch,
                &m.up_batch,
                &m.hidden_batch,
            ]);
        }
        out
    }

    /// Return all GPU state buffers (both KV caches, the device position
    /// scalar, and the per-decode scratch tensors) to the pool. Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = self.kv_sliding.free_gpu(gpu);
        let _ = self.kv_full.free_gpu(gpu);
        let tensors = [
            self.pos_buf,
            self.x,
            self.residual,
            self.tmp,
            self.tmp_rot,
            self.q,
            self.k,
            self.v,
            self.attn_out,
            self.q8_flash_partials,
            self.v_norm_ones,
            self.gate_ffn,
            self.up_ffn,
            self.ffn_hidden,
            self.ffn_out,
            self.logits,
        ];
        let ple = [
            self.ple_token_inputs,
            self.ple_projection_all,
            self.ple_gate,
            self.ple_hidden,
            self.ple_out,
        ];
        for t in tensors
            .into_iter()
            .chain(ple.into_iter().flatten())
            .chain(self.moe.into_iter().flat_map(MoeScratch::tensors))
        {
            let _ = gpu.free_tensor(t);
        }
    }
}

#[cfg(test)]
mod load_rollback_tests {
    use super::*;
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};

    fn tiny_config(moe: bool) -> Gemma4Config {
        let meta = serde_json::json!({"config": {"text_config": {
            "hidden_size": 16, "num_hidden_layers": 2, "vocab_size": 16,
            "num_attention_heads": 2, "head_dim": 8, "num_key_value_heads": 2,
            "global_head_dim": 8, "num_global_key_value_heads": 2,
            "attention_k_eq_v": true, "intermediate_size": 32,
            "layer_types": ["sliding_attention", "full_attention"],
            "enable_moe_block": moe, "moe_intermediate_size": 8,
            "num_experts": 2, "top_k_experts": 2,
        }}});
        Gemma4Config::from_metadata_json(&meta.to_string()).expect("tiny config")
    }

    fn mem(name: String, quant_type: u8, shape: Vec<u32>, data: Vec<u8>) -> HfqMemTensor {
        HfqMemTensor {
            name,
            quant_type,
            shape,
            group_size: 0,
            data,
        }
    }

    fn ones(n: usize) -> Vec<u8> {
        f32_bytes(&vec![1.0f32; n]).to_vec()
    }

    /// An MQ4G256 blob plus its F16 AWQ sidecar, so rollback must reclaim
    /// sidecars too.
    fn proj(name: String, m: usize, k: usize) -> Vec<HfqMemTensor> {
        let awq = name.replace(".weight", ".awq_scale.weight");
        vec![
            mem(name, 13, vec![m as u32, k as u32], vec![0u8; 256]),
            mem(awq, 1, vec![k as u32], vec![0u8; k * 2]),
        ]
    }

    fn fixture(cfg: &Gemma4Config) -> Vec<HfqMemTensor> {
        let root = "model.language_model";
        let dim = cfg.dim;
        let mut v = vec![
            mem(
                format!("{root}.embed_tokens.weight"),
                2,
                vec![cfg.vocab_size as u32, dim as u32],
                ones(cfg.vocab_size * dim),
            ),
            mem(
                format!("{root}.norm.weight"),
                2,
                vec![dim as u32],
                ones(dim),
            ),
        ];
        for (i, ty) in cfg.layer_types.iter().enumerate() {
            let p = format!("{root}.layers.{i}");
            for rel in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                v.push(mem(
                    format!("{p}.{rel}.weight"),
                    2,
                    vec![dim as u32],
                    ones(dim),
                ));
            }
            let (hd, kv) = match ty {
                LayerType::Sliding => (cfg.sliding_head_dim, cfg.sliding_n_kv_heads),
                LayerType::Full => (cfg.full_head_dim, cfg.full_n_kv_heads),
            };
            let (q_dim, kv_dim) = (cfg.n_heads * hd, kv * hd);
            v.extend(proj(format!("{p}.self_attn.q_proj.weight"), q_dim, dim));
            v.extend(proj(format!("{p}.self_attn.k_proj.weight"), kv_dim, dim));
            if *ty == LayerType::Sliding {
                v.extend(proj(format!("{p}.self_attn.v_proj.weight"), kv_dim, dim));
            }
            v.extend(proj(format!("{p}.self_attn.o_proj.weight"), dim, q_dim));
            v.extend(proj(
                format!("{p}.mlp.gate_proj.weight"),
                cfg.hidden_dim,
                dim,
            ));
            v.extend(proj(format!("{p}.mlp.up_proj.weight"), cfg.hidden_dim, dim));
            v.extend(proj(
                format!("{p}.mlp.down_proj.weight"),
                dim,
                cfg.hidden_dim,
            ));
            for rel in ["q_norm", "k_norm"] {
                v.push(mem(
                    format!("{p}.self_attn.{rel}.weight"),
                    2,
                    vec![hd as u32],
                    ones(hd),
                ));
            }
            if cfg.enable_moe_block {
                let n = cfg.num_experts;
                v.extend(proj(format!("{p}.router.proj.weight"), n, dim));
                v.push(mem(
                    format!("{p}.router.scale"),
                    2,
                    vec![dim as u32],
                    ones(dim),
                ));
                v.push(mem(
                    format!("{p}.router.per_expert_scale"),
                    2,
                    vec![n as u32],
                    ones(n),
                ));
                for rel in [
                    "pre_feedforward_layernorm_2",
                    "post_feedforward_layernorm_1",
                    "post_feedforward_layernorm_2",
                ] {
                    v.push(mem(
                        format!("{p}.{rel}.weight"),
                        2,
                        vec![dim as u32],
                        ones(dim),
                    ));
                }
                for x in 0..n {
                    for kind in ["gate_up_proj", "down_proj"] {
                        v.push(mem(
                            format!("{p}.experts.{x}.{kind}.weight"),
                            3,
                            vec![64],
                            vec![0u8; 64],
                        ));
                    }
                }
            }
        }
        v
    }

    /// Fail every allocation of a load in turn: each failure is the injected
    /// one and returns the pool to its warm level (completed layers, MoE pools
    /// and AWQ sidecars included); the first clean load then frees everything.
    fn sweep(tag: &str, moe: bool) {
        let mut gpu = match Gpu::init() {
            Ok(gpu) => gpu,
            Err(e) => return eprintln!("skip: no GPU ({e:?})"),
        };
        let cfg = tiny_config(moe);
        let path =
            std::env::temp_dir().join(format!("gemma4_ledger_{tag}_{}.hfq", std::process::id()));
        write_hfqm_package_mem(&path, 13, "{}", &fixture(&cfg)).expect("write fixture");
        let hfq = HfqFile::open(&path).expect("open fixture");
        for _ in 0..2 {
            Gemma4Weights::load(&hfq, &cfg, &mut gpu)
                .expect("warm load")
                .free_gpu(&mut gpu);
        }
        let warm = gpu.pool_stats().0;
        let mut fail_at = 1;
        loop {
            match Gemma4Weights::load_with_fault(&hfq, &cfg, &mut gpu, Some(fail_at)) {
                Ok(weights) => {
                    assert!(moe == weights.layers[0].moe().is_some());
                    weights.free_gpu(&mut gpu);
                    break;
                }
                Err(e) => {
                    assert!(e.contains("injected"), "fail_at={fail_at}: {e}");
                    assert_eq!(gpu.pool_stats().0, warm, "fail_at={fail_at} leaked");
                }
            }
            fail_at += 1;
        }
        assert!(fail_at > 10, "the sweep must cover every staged owner");
        assert_eq!(gpu.pool_stats().0, warm, "clean load leaked");
        gpu.drain_pool();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[ignore = "requires an AMD GPU; exercises ledger rollback across layers and AWQ sidecars"]
    fn dense_load_failure_reclaims_every_owner() {
        sweep("dense", false);
    }

    #[test]
    #[ignore = "requires an AMD GPU; exercises MoE pool / pointer-table rollback"]
    fn moe_load_failure_reclaims_every_owner() {
        sweep("moe", true);
    }
}
