// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
#![cfg(target_os = "linux")]

//! Shared dense Qwen3.5 weight export/import (weight-host S3).
//!
//! The weight host runs the **existing** loader once (every load transform —
//! norm bias, AWQ repack, reciprocal planes, Z|beta|alpha fold — is reused, not
//! rewritten), then [`export_dense_weights`] promotes every unique final owner
//! to an exportable read-only VMM allocation, one owner at a time, and builds a
//! typed [`WeightManifest`]. A tenant calls [`import_dense_weights`], which
//! validates the complete manifest (config, geometry, sidecars, Lloyd LUTs,
//! alias topology, allocation extents) **before** mapping any FD, maps one
//! allocation per unique owner and hydrates the loader's final typed
//! [`Qwen35Weights`] through `Qwen35Weights::from_dense_seal`.
//!
//! Identity model:
//! * owner id == the typed-traversal role of the (single) owning tensor, e.g.
//!   `layers.0003.wqkv.buf`; roles sort bytewise == numeric layer order because
//!   layer indices are zero padded;
//! * alias views (a tied `output.buf`) carry `alias_of == owner_id`,
//!   `byte_offset`/`byte_extent` inside the owner;
//! * the deterministic binary seal produced by [`encode_dense_seal`] (fixed
//!   tags, little-endian u64 lengths, no `serde`, no native `usize`, no
//!   address) is `WeightManifest::dispatch_metadata` (encoded and hashed
//!   verbatim by the runtime's layout seal);
//! * `WeightManifest::config` is [`manifest_config`]: exact float bits, layer
//!   topology and the resolved recipe flags.
//!
//! Nothing here launches a kernel; GPU work is limited to the S2 allocation
//! primitives.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::c_void;
use std::fmt;
use std::os::fd::{BorrowedFd, OwnedFd};

use hip_bridge::{DeviceBuffer, HipError};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::EmbeddingFormat;
use hipfire_runtime::weight_host::{
    CanonicalFields, FieldValue, LayoutRecipe, MatrixDesc, QuantDesc, SharedAllocation,
    WeightHostLease, WeightManifest, WeightView,
};
use rdna_compute::{DType, ExportedWeightAllocation, Gpu, GpuTensor, ImportedWeightAllocationDesc};

use crate::qwen35::config::{f16_lm_head_mode_from_config, F16LmHeadMode, LayerType, Qwen35Config};
use crate::qwen35::weights::{
    DeltaNetLayerWeights, FullAttnLayerWeights, GpuTensorDescriptor, LayerWeights, Qwen35LayerSeal,
    Qwen35RankSeal, Qwen35Weights, WeightTensorDescriptor,
};

// ─── Constants ──────────────────────────────────────────────────────────

/// `WeightManifest::arch` of every dense Qwen3.5 shared set.
pub const DENSE_MANIFEST_ARCH: &str = "hipfire.qwen35.dense";
/// Only device architecture admitted by v1.
pub const SUPPORTED_DEVICE_ARCH: &str = "gfx1201";
/// Zeroed tail mapped behind every owner (mirrors rdna-compute's
/// `EXPORT_WEIGHT_TAIL_PAD_BYTES`): kernels over-read past `buf.size()`.
pub const EXPORT_WEIGHT_TAIL_PAD_BYTES: u64 = 64 * 1024;
/// `SharedAllocation::representation_tag` of every owner.
pub const DENSE_REPRESENTATION_TAG: &str =
    "qwen35-dense/typed-seal-v1/vmm-ro-posix-fd/min-granule/tail64k";
/// `LayoutRecipe::schema` of the dense recipe.
pub const DENSE_RECIPE_SCHEMA: &str = "hipfire.qwen35.dense.recipe.v1";

const SEAL_MAGIC: &[u8; 8] = b"HQ35DSL1";
const SEAL_VERSION: u32 = 1;
/// Hard decode bounds (checked before any allocation they would drive).
pub const MAX_SEAL_BYTES: usize = 8 << 20;
const MAX_LAYERS: u64 = 4096;
const MAX_RANK: u64 = 8;
/// Block-count of a Z-fold tile search on gfx11 (see `load.rs` fold).
const GFX11_FOLD_N: usize = 512;

// ─── Errors ─────────────────────────────────────────────────────────────

/// Failure of an export/import transaction. When cleanup of GPU state also
/// failed the failures are carried in [`SharedWeightError::Cleanup`]; GPU
/// arenas whose release failed stay tracked by the `Gpu` orphan table.
#[derive(Debug)]
pub enum SharedWeightError {
    /// Model/route/config outside the v1 dense allowlist.
    Unsupported(String),
    /// Structural/validation failure (mismatching manifest, bad extents …).
    Invalid(String),
    /// Malformed or truncated seal blob.
    Codec(String),
    /// Manifest construction/validation failure reported by the wire module.
    Manifest(String),
    /// A HIP/allocator call failed.
    Hip { context: String, error: HipError },
    /// The primary failure plus failures observed while cleaning up after it.
    Cleanup {
        cause: Box<SharedWeightError>,
        failures: Vec<String>,
    },
}

impl SharedWeightError {
    fn invalid(msg: impl Into<String>) -> Self {
        Self::Invalid(msg.into())
    }
    fn unsupported(msg: impl Into<String>) -> Self {
        Self::Unsupported(msg.into())
    }
    fn codec(msg: impl Into<String>) -> Self {
        Self::Codec(msg.into())
    }
    fn hip(context: impl Into<String>, error: HipError) -> Self {
        Self::Hip {
            context: context.into(),
            error,
        }
    }
    fn with_cleanup(self, failures: Vec<String>) -> Self {
        if failures.is_empty() {
            self
        } else {
            Self::Cleanup {
                cause: Box::new(self),
                failures,
            }
        }
    }
}

impl fmt::Display for SharedWeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(m) => write!(f, "shared dense weights: unsupported: {m}"),
            Self::Invalid(m) => write!(f, "shared dense weights: invalid: {m}"),
            Self::Codec(m) => write!(f, "shared dense weights: seal codec: {m}"),
            Self::Manifest(m) => write!(f, "shared dense weights: manifest: {m}"),
            Self::Hip { context, error } => write!(f, "shared dense weights: {context}: {error}"),
            Self::Cleanup { cause, failures } => {
                write!(f, "{cause}; cleanup failures: {}", failures.join("; "))
            }
        }
    }
}

impl std::error::Error for SharedWeightError {}

type SwResult<T> = Result<T, SharedWeightError>;

fn hip_err(context: &'static str) -> impl FnOnce(HipError) -> SharedWeightError {
    move |e| SharedWeightError::hip(context, e)
}

fn manifest_err<E: fmt::Display>(context: &'static str) -> impl FnOnce(E) -> SharedWeightError {
    move |e| SharedWeightError::Manifest(format!("{context}: {e}"))
}

// ─── DType / embedding tags ─────────────────────────────────────────────

/// One table drives tag, inverse and name. `dtype_tag` is an exhaustive match,
/// so a new `DType` variant fails to compile here until it is given a fixed,
/// never-reused tag. Tags are wire constants.
macro_rules! dtype_table {
    ($($v:ident = $tag:literal),+ $(,)?) => {
        /// Fixed wire tag of `dtype`.
        pub fn dtype_tag(dtype: DType) -> u8 {
            match dtype { $(DType::$v => $tag,)+ }
        }
        /// Inverse of [`dtype_tag`]; `None` for an unassigned tag.
        pub fn dtype_from_tag(tag: u8) -> Option<DType> {
            match tag { $($tag => Some(DType::$v),)+ _ => None }
        }
        /// Stable textual name (wire constant, not `Debug`).
        pub fn dtype_name(dtype: DType) -> &'static str {
            match dtype { $(DType::$v => stringify!($v),)+ }
        }
    };
}

dtype_table! {
    F32 = 1, F16 = 2, BF16 = 3, Q4K = 4, Q6K = 5, Q8_0 = 6, Q4F16G64 = 7, Q4F16G32 = 8,
    Q8HFQ = 9, HFQ4G256 = 10, HFQ4G128 = 11, HFQ3G256 = 12, HFQ3G128 = 13, MQ4G256 = 14,
    MQ4G256V2 = 15, MQ4G256V2Lloyd = 16, MQ4CG256 = 17, MQ6G256V2 = 18, MQ5G256V2 = 19,
    MQ4G128V2 = 20, MQ4G128 = 21, MQ3G256V2 = 22, MQ2G256V2 = 23, MQ8G256 = 24, MQ6G256 = 25,
    MQ5G256 = 26, MQ3G256 = 27, MQ2G256 = 28, MQ2G256Lloyd = 29, MQ2G256LloydU = 30,
    MQ3G256Lloyd = 31, MQ4G256Lloyd = 32, MQ2G256GL = 33, MQ3G256GL = 34, HFP4G32 = 35,
    MFP4G32 = 36, MFP4G32Lloyd = 37, MFP4G32P = 38, MFP4G32E8 = 39, MFP4G32E8SOA = 40,
    MFP3G32E8 = 41, MFP2G32E8 = 42, MFP4G32E8G128 = 43, HFQ2G256 = 44, HFQ2G128 = 45,
    TQ2G128 = 46, BQ1G128 = 47, HFQ6G256 = 48, ParoQ4G128 = 49, Raw = 50, FP8E4M3B128 = 51,
    FP8E4M3PT = 52,
    MQ3I4G128R256 = 53, MQ3I4G128R128 = 54, MQ2I4G128R256 = 55, MQ2I4G128R128 = 56,
    E8PI4G128R256 = 57, E8PI4G128R128 = 58, ParoQ4G64 = 59,
}

fn embd_tag(f: EmbeddingFormat) -> u8 {
    match f {
        EmbeddingFormat::F32 => 1,
        EmbeddingFormat::Q4K => 2,
        EmbeddingFormat::HFQ4G256 => 3,
        EmbeddingFormat::HFQ4G128 => 4,
        EmbeddingFormat::Q8_0 => 5,
    }
}

fn embd_from_tag(tag: u8) -> Option<EmbeddingFormat> {
    Some(match tag {
        1 => EmbeddingFormat::F32,
        2 => EmbeddingFormat::Q4K,
        3 => EmbeddingFormat::HFQ4G256,
        4 => EmbeddingFormat::HFQ4G128,
        5 => EmbeddingFormat::Q8_0,
        _ => return None,
    })
}

/// `(group elems, block elems, block bytes, packing)` of the dense quant
/// dtypes whose per-row bytes are `ceil(k/block) * block_bytes`.
fn quant_block(dtype: DType) -> Option<(u64, u64, u64, &'static str)> {
    Some(match dtype {
        DType::Q8_0 => (32, 32, 34, "q8_0"),
        DType::HFQ4G256 => (256, 256, 136, "hfq4-fp32-scale-zero"),
        DType::MQ4G256 => (256, 256, 136, "mq4-fwht-fp32-scale-zero"),
        DType::MQ4G256V2 => (256, 256, 136, "mq4v2-fwht-dual-fp16-header"),
        DType::MQ4G256V2Lloyd => (256, 256, 136, "mq4v2-lloyd-dual-fp16-header-host-lut"),
        DType::MQ4G128V2 => (128, 128, 68, "mq4g128v2"),
        DType::HFQ6G256 | DType::MQ6G256 | DType::MQ6G256V2 => (256, 256, 200, "6bit-g256"),
        DType::MQ3G256 => (256, 256, 104, "3bit-g256"),
        DType::MQ2G256 | DType::MQ2G256Lloyd => (256, 256, 72, "2bit-g256"),
        DType::MQ3G256Lloyd => (256, 256, 112, "3bit-g256-lloyd"),
        DType::MQ4G256Lloyd => (256, 256, 160, "4bit-g256-lloyd"),
        DType::ParoQ4G128 => (128, 128, 72, "paroq4g128"),
        _ => return None,
    })
}

// ─── Resolved recipe / dispatch flags ───────────────────────────────────

/// Every load-affecting dispatch fact the loader resolved from live `Gpu`
/// predicates (never from environment strings).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseRecipeFlags {
    /// `hfq.mq4v2_symmetric()`; restored on `gpu.mq4v2_symmetric` by the importer.
    pub mq4v2_symmetric: bool,
    /// `gpu.a4_rms_fdiv_planes_wanted()` — AWQ reciprocal planes `[a][R][Rlo]`.
    pub a4_rms_fdiv_planes: bool,
    /// `gpu.iu4_gdn_fold_active(..)` — gfx12 Z|beta|alpha fold (+128 rows).
    pub gdn_fold_gfx12: bool,
    /// `gpu.mq4v2_fold_betaalpha_active(..)` — gfx11 fold.
    pub gdn_fold_gfx11: bool,
    /// `HIPFIRE_LM_HEAD_F16` resolved mode (lm_head F16 widened to F32).
    pub lm_head_f32: bool,
}

/// DeltaNet projection dims the fold predicates are evaluated on.
fn delta_dims(c: &Qwen35Config) -> SwResult<([usize; 4], usize)> {
    let overflow = || SharedWeightError::invalid("DeltaNet dimension overflow");
    let qkv = c
        .linear_num_key_heads
        .checked_mul(c.linear_key_head_dim)
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| {
            c.linear_num_value_heads
                .checked_mul(c.linear_value_head_dim)
                .and_then(|w| v.checked_add(w))
        })
        .ok_or_else(overflow)?;
    let z = c
        .linear_num_value_heads
        .checked_mul(c.linear_value_head_dim)
        .ok_or_else(overflow)?;
    Ok((
        [qkv, z, c.linear_num_value_heads, c.linear_num_value_heads],
        c.dim,
    ))
}

/// Resolve [`DenseRecipeFlags`] with the exact predicates `load.rs` uses
/// (`attach_awq_rcp_planes`: `a4_rms_fdiv_planes_wanted`; `append_betaalpha_to_z`:
/// `iu4_gdn_fold_active` / `mq4v2_fold_betaalpha_active(.., 512)`). The result
/// depends on `gpu.mq4v2_symmetric`, so apply
/// [`apply_source_dispatch_flags`] first.
pub fn resolve_dense_flags(gpu: &Gpu, config: &Qwen35Config) -> SwResult<DenseRecipeFlags> {
    let (dims, k) = delta_dims(config)?;
    Ok(DenseRecipeFlags {
        mq4v2_symmetric: gpu.mq4v2_symmetric,
        a4_rms_fdiv_planes: gpu.a4_rms_fdiv_planes_wanted(),
        gdn_fold_gfx12: gpu.iu4_gdn_fold_active(dims, k),
        gdn_fold_gfx11: gpu.mq4v2_fold_betaalpha_active(
            dims[0],
            dims[1],
            dims[2],
            dims[3],
            k,
            GFX11_FOLD_N,
        ),
        lm_head_f32: f16_lm_head_mode_from_config() == F16LmHeadMode::F32,
    })
}

/// Same source flag `model_load::load_weights_inner` stamps on every device
/// before a load: `device.mq4v2_symmetric = source.mq4v2_symmetric()`.
/// Call this on the host/tenant `Gpu` before [`resolve_dense_recipe`].
pub fn apply_source_dispatch_flags(gpu: &mut Gpu, hfq: &HfqFile) -> bool {
    let symmetric = hfq.mq4v2_symmetric();
    gpu.mq4v2_symmetric = symmetric;
    symmetric
}

fn fv_u64(v: u64) -> FieldValue {
    FieldValue::U64(v)
}
fn fv_bool(v: bool) -> FieldValue {
    FieldValue::Bool(v)
}
fn fv_str(v: &str) -> FieldValue {
    FieldValue::Str(v.to_owned())
}
fn fv_bytes(v: Vec<u8>) -> FieldValue {
    FieldValue::Bytes(v)
}

fn recipe_fields(arch: &str, flags: &DenseRecipeFlags) -> CanonicalFields {
    let mut f = CanonicalFields::new();
    f.insert("arch".into(), fv_str(arch));
    f.insert("mq4v2_symmetric".into(), fv_bool(flags.mq4v2_symmetric));
    f.insert(
        "a4_rms_fdiv_planes".into(),
        fv_bool(flags.a4_rms_fdiv_planes),
    );
    f.insert("gdn_fold_gfx12".into(), fv_bool(flags.gdn_fold_gfx12));
    f.insert("gdn_fold_gfx11".into(), fv_bool(flags.gdn_fold_gfx11));
    f.insert("lm_head_f32".into(), fv_bool(flags.lm_head_f32));
    f.insert(
        "weight_tail_pad_bytes".into(),
        fv_u64(EXPORT_WEIGHT_TAIL_PAD_BYTES),
    );
    f.insert(
        "export_granularity".into(),
        fv_str("hipMemAllocationGranularityMinimum"),
    );
    f.insert(
        "representation_tag".into(),
        fv_str(DENSE_REPRESENTATION_TAG),
    );
    f.insert(
        "load_transforms".into(),
        fv_str("reuse:qwen35.load:norm-bias+awq-rcp-planes+z-beta-alpha-fold"),
    );
    f
}

/// Resolve the dense layout recipe from live GPU predicates. Defaults are
/// unchanged: nothing here toggles a loader transform.
pub fn resolve_dense_recipe(gpu: &Gpu, config: &Qwen35Config) -> SwResult<LayoutRecipe> {
    check_dense_config(config)?;
    refuse_host_mapped_weights(gpu)?;
    let flags = resolve_dense_flags(gpu, config)?;
    Ok(LayoutRecipe {
        schema: DENSE_RECIPE_SCHEMA.to_owned(),
        fields: recipe_fields(&gpu.arch, &flags),
    })
}

/// The load-time half of the v1 allowlist that the config can no longer state:
/// partial GPU offload is resolved from the budgets at load, and a model with
/// any host-mapped weight cannot share them — the shared seal describes
/// device-resident bytes.
///
/// The test is the live host-mapped allocation count, not the CPU-exec policy
/// flag (`owns_host_cpu_weights`): the allowlist refused *any* partial offload
/// (beta's `i_gpu_start != 0`), and a `pcie` layer spill registers host-mapped
/// owners without setting that flag. `validate_dense_seal` refuses host-mapped
/// owners again at seal validation; this is the early, named refusal.
fn refuse_host_mapped_weights(gpu: &Gpu) -> SwResult<()> {
    if gpu.host_mapped_count() != 0 {
        return Err(SharedWeightError::unsupported(
            "partial GPU offload (host-mapped weights)",
        ));
    }
    Ok(())
}

// ─── Config identity ────────────────────────────────────────────────────

/// Reject every config outside the v1 dense allowlist.
pub fn check_dense_config(c: &Qwen35Config) -> SwResult<()> {
    if c.num_experts != 0 {
        return Err(SharedWeightError::unsupported("MoE (num_experts != 0)"));
    }
    if c.paged_experts {
        return Err(SharedWeightError::unsupported("weight pager"));
    }
    if c.reap_keep.is_some() {
        return Err(SharedWeightError::unsupported("REAP keep-map"));
    }
    // Partial GPU offload is no longer a config fact: placement is resolved at
    // load time from the budgets, so the refusal lives at the Gpu-bearing
    // entries below (`resolve_dense_recipe` / `export_prepare`) and reads
    // `Gpu::owns_host_cpu_weights()`.
    if c.n_layers == 0 || c.layer_types.len() != c.n_layers || c.n_layers as u64 > MAX_LAYERS {
        return Err(SharedWeightError::invalid(format!(
            "layer topology: n_layers {} vs layer_types {}",
            c.n_layers,
            c.layer_types.len()
        )));
    }
    for (name, v) in [
        ("dim", c.dim),
        ("hidden_dim", c.hidden_dim),
        ("vocab_size", c.vocab_size),
        ("n_heads", c.n_heads),
        ("n_kv_heads", c.n_kv_heads),
        ("head_dim", c.head_dim),
        ("linear_num_key_heads", c.linear_num_key_heads),
        ("linear_num_value_heads", c.linear_num_value_heads),
        ("linear_key_head_dim", c.linear_key_head_dim),
        ("linear_value_head_dim", c.linear_value_head_dim),
        ("conv_kernel_dim", c.conv_kernel_dim),
    ] {
        if v == 0 {
            return Err(SharedWeightError::invalid(format!("config {name} is zero")));
        }
    }
    Ok(())
}

/// Exact config identity: every primitive of `Qwen35Config`, floats as raw
/// bits, layer topology as one tag byte per layer.
pub fn dense_config_fields(c: &Qwen35Config) -> CanonicalFields {
    let mut f = CanonicalFields::new();
    let mut u = |k: &str, v: u64| {
        f.insert(k.to_owned(), fv_u64(v));
    };
    u("dim", c.dim as u64);
    u("n_layers", c.n_layers as u64);
    u("vocab_size", c.vocab_size as u64);
    u("norm_eps_bits", u64::from(c.norm_eps.to_bits()));
    u("eos_token", u64::from(c.eos_token));
    u("n_heads", c.n_heads as u64);
    u("n_kv_heads", c.n_kv_heads as u64);
    u("head_dim", c.head_dim as u64);
    u("rope_theta_bits", u64::from(c.rope_theta.to_bits()));
    u(
        "partial_rotary_factor_bits",
        u64::from(c.partial_rotary_factor.to_bits()),
    );
    u("mrope_section_0", c.mrope_section[0] as u64);
    u("mrope_section_1", c.mrope_section[1] as u64);
    u("mrope_section_2", c.mrope_section[2] as u64);
    u("linear_num_key_heads", c.linear_num_key_heads as u64);
    u("linear_num_value_heads", c.linear_num_value_heads as u64);
    u("linear_key_head_dim", c.linear_key_head_dim as u64);
    u("linear_value_head_dim", c.linear_value_head_dim as u64);
    u("conv_kernel_dim", c.conv_kernel_dim as u64);
    u("hidden_dim", c.hidden_dim as u64);
    u("num_experts", c.num_experts as u64);
    u("num_experts_per_tok", c.num_experts_per_tok as u64);
    u("moe_intermediate_size", c.moe_intermediate_size as u64);
    u(
        "shared_expert_intermediate_size",
        c.shared_expert_intermediate_size as u64,
    );
    u("vram_budget_bytes", c.vram_budget_bytes);
    f.insert("is_vl_text".into(), fv_bool(c.is_vl_text));
    f.insert("mrope_interleaved".into(), fv_bool(c.mrope_interleaved));
    f.insert("has_shared_expert".into(), fv_bool(c.has_shared_expert));
    f.insert("norm_topk_prob".into(), fv_bool(c.norm_topk_prob));
    f.insert("paged_experts".into(), fv_bool(c.paged_experts));
    f.insert("has_reap_keep".into(), fv_bool(c.reap_keep.is_some()));
    f.insert(
        "layer_types".into(),
        fv_bytes(
            c.layer_types
                .iter()
                .map(|t| match t {
                    LayerType::LinearAttention => 1u8,
                    LayerType::FullAttention => 2u8,
                })
                .collect(),
        ),
    );
    f
}

/// Complete `WeightManifest::config`: [`dense_config_fields`] + resolved
/// recipe flags. The importer rebuilds it from its local config and the
/// decoded flags and requires exact equality.
pub fn manifest_config(config: &Qwen35Config, flags: &DenseRecipeFlags) -> CanonicalFields {
    let mut f = dense_config_fields(config);
    f.insert(
        "flag.mq4v2_symmetric".into(),
        fv_bool(flags.mq4v2_symmetric),
    );
    f.insert(
        "flag.a4_rms_fdiv_planes".into(),
        fv_bool(flags.a4_rms_fdiv_planes),
    );
    f.insert("flag.gdn_fold_gfx12".into(), fv_bool(flags.gdn_fold_gfx12));
    f.insert("flag.gdn_fold_gfx11".into(), fv_bool(flags.gdn_fold_gfx11));
    f.insert("flag.lm_head_f32".into(), fv_bool(flags.lm_head_f32));
    f.insert(
        "weight_tail_pad_bytes".into(),
        fv_u64(EXPORT_WEIGHT_TAIL_PAD_BYTES),
    );
    f
}

// ─── Deterministic seal codec ───────────────────────────────────────────

/// Decoded dense seal: resolved flags plus the complete typed weight seal.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseSealMeta {
    pub flags: DenseRecipeFlags,
    pub seal: Qwen35RankSeal,
}

struct Enc(Vec<u8>);

impl Enc {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn bool(&mut self, v: bool) {
        self.0.push(u8::from(v));
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn usize(&mut self, v: usize) -> SwResult<()> {
        self.u64(u64::try_from(v).map_err(|_| SharedWeightError::codec("usize exceeds u64"))?);
        Ok(())
    }
    fn gtd(&mut self, d: &GpuTensorDescriptor) -> SwResult<()> {
        if d.shape.len() as u64 > MAX_RANK {
            return Err(SharedWeightError::codec("tensor rank exceeds bound"));
        }
        self.u64(d.shape.len() as u64);
        for &s in &d.shape {
            self.usize(s)?;
        }
        self.u8(dtype_tag(d.dtype));
        self.usize(d.byte_len)
    }
    fn lut<const N: usize>(&mut self, l: &Option<[u32; N]>) {
        match l {
            None => self.u8(0),
            Some(words) => {
                self.u8(1);
                for w in words {
                    self.u32(*w);
                }
            }
        }
    }
    fn wtd(&mut self, w: &WeightTensorDescriptor) -> SwResult<()> {
        if w.paro.is_some() {
            return Err(SharedWeightError::unsupported("Paro sidecar"));
        }
        self.u8(dtype_tag(w.gpu_dtype));
        self.usize(w.m)?;
        self.usize(w.k)?;
        self.usize(w.row_stride)?;
        self.gtd(&w.buf)?;
        match &w.awq_scale {
            None => self.u8(0),
            Some(a) => {
                self.u8(1);
                self.gtd(a)?;
            }
        }
        self.u8(0); // paro: explicit absent tag
        self.lut(&w.lloyd_lut_e4m3);
        self.lut(&w.lloyd_lut_f16);
        self.lut(&w.lloyd_lut_c16);
        Ok(())
    }
}

/// Deterministic binary encoding of a complete dense seal.
pub fn encode_dense_seal(flags: &DenseRecipeFlags, seal: &Qwen35RankSeal) -> SwResult<Vec<u8>> {
    if seal.has_pager
        || seal.moe_has_mq6
        || seal.global_expert_dtypes.iter().any(|v| !v.is_empty())
        || seal.local_expert_descriptors.iter().any(|v| !v.is_empty())
    {
        return Err(SharedWeightError::unsupported(
            "MoE/pager/expert metadata in seal",
        ));
    }
    if seal.layer_seals.len() as u64 > MAX_LAYERS {
        return Err(SharedWeightError::codec("layer count exceeds bound"));
    }
    let mut e = Enc(Vec::with_capacity(1 << 16));
    e.0.extend_from_slice(SEAL_MAGIC);
    e.u32(SEAL_VERSION);
    e.bool(flags.mq4v2_symmetric);
    e.bool(flags.a4_rms_fdiv_planes);
    e.bool(flags.gdn_fold_gfx12);
    e.bool(flags.gdn_fold_gfx11);
    e.bool(flags.lm_head_f32);
    e.u8(embd_tag(seal.embd_format));
    e.bool(seal.lm_head_aliases_embd);
    e.gtd(&seal.token_embd)?;
    e.gtd(&seal.output_norm)?;
    e.wtd(&seal.output)?;
    e.u64(seal.layer_seals.len() as u64);
    for layer in &seal.layer_seals {
        match layer {
            Qwen35LayerSeal::DeltaNet {
                attn_norm,
                wqkv,
                wz,
                w_alpha,
                w_beta,
                a_log,
                dt_bias,
                conv_weight,
                norm_weight,
                wo,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
            } => {
                e.u8(1);
                e.gtd(attn_norm)?;
                e.wtd(wqkv)?;
                e.wtd(wz)?;
                e.wtd(w_alpha)?;
                e.wtd(w_beta)?;
                e.gtd(a_log)?;
                e.gtd(dt_bias)?;
                e.gtd(conv_weight)?;
                e.gtd(norm_weight)?;
                e.wtd(wo)?;
                e.gtd(ffn_norm)?;
                e.wtd(w_gate)?;
                e.wtd(w_up)?;
                e.wtd(w_down)?;
            }
            Qwen35LayerSeal::FullAttn {
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
            } => {
                e.u8(2);
                e.gtd(attn_norm)?;
                e.wtd(wq)?;
                e.wtd(wk)?;
                e.wtd(wv)?;
                e.wtd(wo)?;
                e.gtd(q_norm)?;
                e.gtd(k_norm)?;
                e.gtd(ffn_norm)?;
                e.wtd(w_gate)?;
                e.wtd(w_up)?;
                e.wtd(w_down)?;
            }
            Qwen35LayerSeal::DeltaNetMoe { .. } | Qwen35LayerSeal::FullAttnMoe { .. } => {
                return Err(SharedWeightError::unsupported("MoE layer seal"));
            }
        }
    }
    if e.0.len() > MAX_SEAL_BYTES {
        return Err(SharedWeightError::codec("encoded seal exceeds bound"));
    }
    Ok(e.0)
}

struct Dec<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Dec<'a> {
    fn take(&mut self, n: usize) -> SwResult<&'a [u8]> {
        let end = self
            .p
            .checked_add(n)
            .filter(|&end| end <= self.b.len())
            .ok_or_else(|| SharedWeightError::codec("truncated seal"))?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u8(&mut self) -> SwResult<u8> {
        Ok(self.take(1)?[0])
    }
    fn bool(&mut self) -> SwResult<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            t => Err(SharedWeightError::codec(format!("invalid bool tag {t}"))),
        }
    }
    fn u32(&mut self) -> SwResult<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn u64(&mut self) -> SwResult<u64> {
        let s = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Ok(u64::from_le_bytes(a))
    }
    fn usize(&mut self) -> SwResult<usize> {
        usize::try_from(self.u64()?).map_err(|_| SharedWeightError::codec("value exceeds usize"))
    }
    fn dtype(&mut self) -> SwResult<DType> {
        let t = self.u8()?;
        dtype_from_tag(t).ok_or_else(|| SharedWeightError::codec(format!("unknown dtype tag {t}")))
    }
    fn gtd(&mut self) -> SwResult<GpuTensorDescriptor> {
        let rank = self.u64()?;
        if rank > MAX_RANK {
            return Err(SharedWeightError::codec("tensor rank exceeds bound"));
        }
        let mut shape = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            shape.push(self.usize()?);
        }
        let dtype = self.dtype()?;
        let byte_len = self.usize()?;
        Ok(GpuTensorDescriptor {
            shape,
            dtype,
            byte_len,
        })
    }
    fn lut<const N: usize>(&mut self) -> SwResult<Option<[u32; N]>> {
        match self.u8()? {
            0 => Ok(None),
            1 => {
                let mut w = [0u32; N];
                for slot in &mut w {
                    *slot = self.u32()?;
                }
                Ok(Some(w))
            }
            t => Err(SharedWeightError::codec(format!("invalid LUT tag {t}"))),
        }
    }
    fn wtd(&mut self) -> SwResult<WeightTensorDescriptor> {
        let gpu_dtype = self.dtype()?;
        let m = self.usize()?;
        let k = self.usize()?;
        let row_stride = self.usize()?;
        let buf = self.gtd()?;
        let awq_scale = match self.u8()? {
            0 => None,
            1 => Some(self.gtd()?),
            t => return Err(SharedWeightError::codec(format!("invalid AWQ tag {t}"))),
        };
        match self.u8()? {
            0 => {}
            _ => return Err(SharedWeightError::unsupported("Paro sidecar in seal")),
        }
        let lloyd_lut_e4m3 = self.lut::<4>()?;
        let lloyd_lut_f16 = self.lut::<8>()?;
        let lloyd_lut_c16 = self.lut::<4>()?;
        Ok(WeightTensorDescriptor {
            gpu_dtype,
            m,
            k,
            row_stride,
            buf,
            awq_scale,
            paro: None,
            lloyd_lut_e4m3,
            lloyd_lut_f16,
            lloyd_lut_c16,
        })
    }
}

/// Checked, bounded inverse of [`encode_dense_seal`]. Rejects truncation,
/// trailing bytes, unknown tags, oversized ranks/layer counts and Paro/MoE.
pub fn decode_dense_seal(bytes: &[u8]) -> SwResult<DenseSealMeta> {
    if bytes.len() > MAX_SEAL_BYTES {
        return Err(SharedWeightError::codec("seal exceeds size bound"));
    }
    let mut d = Dec { b: bytes, p: 0 };
    if d.take(SEAL_MAGIC.len())? != SEAL_MAGIC {
        return Err(SharedWeightError::codec("bad magic"));
    }
    let version = d.u32()?;
    if version != SEAL_VERSION {
        return Err(SharedWeightError::codec(format!(
            "unsupported seal version {version}"
        )));
    }
    let flags = DenseRecipeFlags {
        mq4v2_symmetric: d.bool()?,
        a4_rms_fdiv_planes: d.bool()?,
        gdn_fold_gfx12: d.bool()?,
        gdn_fold_gfx11: d.bool()?,
        lm_head_f32: d.bool()?,
    };
    let tag = d.u8()?;
    let embd_format = embd_from_tag(tag)
        .ok_or_else(|| SharedWeightError::codec(format!("unknown embedding tag {tag}")))?;
    let lm_head_aliases_embd = d.bool()?;
    let token_embd = d.gtd()?;
    let output_norm = d.gtd()?;
    let output = d.wtd()?;
    let n_layers = d.u64()?;
    if n_layers > MAX_LAYERS {
        return Err(SharedWeightError::codec("layer count exceeds bound"));
    }
    let mut layer_seals = Vec::with_capacity(n_layers as usize);
    for _ in 0..n_layers {
        layer_seals.push(match d.u8()? {
            1 => Qwen35LayerSeal::DeltaNet {
                attn_norm: d.gtd()?,
                wqkv: d.wtd()?,
                wz: d.wtd()?,
                w_alpha: d.wtd()?,
                w_beta: d.wtd()?,
                a_log: d.gtd()?,
                dt_bias: d.gtd()?,
                conv_weight: d.gtd()?,
                norm_weight: d.gtd()?,
                wo: d.wtd()?,
                ffn_norm: d.gtd()?,
                w_gate: d.wtd()?,
                w_up: d.wtd()?,
                w_down: d.wtd()?,
            },
            2 => Qwen35LayerSeal::FullAttn {
                attn_norm: d.gtd()?,
                wq: d.wtd()?,
                wk: d.wtd()?,
                wv: d.wtd()?,
                wo: d.wtd()?,
                q_norm: d.gtd()?,
                k_norm: d.gtd()?,
                ffn_norm: d.gtd()?,
                w_gate: d.wtd()?,
                w_up: d.wtd()?,
                w_down: d.wtd()?,
            },
            t => return Err(SharedWeightError::codec(format!("unknown layer tag {t}"))),
        });
    }
    if d.p != bytes.len() {
        return Err(SharedWeightError::codec("trailing bytes after seal"));
    }
    let n = layer_seals.len();
    Ok(DenseSealMeta {
        flags,
        seal: Qwen35RankSeal {
            token_embd,
            embd_format,
            output_norm,
            output,
            moe_has_mq6: false,
            has_pager: false,
            lm_head_aliases_embd,
            layer_seals,
            global_expert_dtypes: vec![Vec::new(); n],
            local_expert_descriptors: vec![Vec::new(); n],
        },
    })
}

// ─── Seal role entries (naming == typed traversal) ──────────────────────

struct Entry<'a> {
    role: String,
    desc: &'a GpuTensorDescriptor,
    /// Set for a weight's `.buf` role.
    weight: Option<&'a WeightTensorDescriptor>,
    /// Set for an AWQ companion: the weight `.buf` role it belongs to.
    awq_of: Option<(String, &'a WeightTensorDescriptor)>,
    /// Owner role when this entry is a borrowed alias.
    alias_of: Option<String>,
    kind: &'static str,
}

fn push_weight<'a>(
    out: &mut Vec<Entry<'a>>,
    prefix: &str,
    w: &'a WeightTensorDescriptor,
    alias_of: Option<&str>,
) {
    let buf_role = format!("{prefix}.buf");
    out.push(Entry {
        role: buf_role.clone(),
        desc: &w.buf,
        weight: Some(w),
        awq_of: None,
        alias_of: alias_of.map(str::to_owned),
        kind: "weight",
    });
    if let Some(a) = &w.awq_scale {
        out.push(Entry {
            role: format!("{prefix}.awq_scale"),
            desc: a,
            weight: None,
            awq_of: Some((buf_role, w)),
            alias_of: None,
            kind: "awq_scale",
        });
    }
}

fn push_plain<'a>(
    out: &mut Vec<Entry<'a>>,
    role: String,
    d: &'a GpuTensorDescriptor,
    kind: &'static str,
) {
    out.push(Entry {
        role,
        desc: d,
        weight: None,
        awq_of: None,
        alias_of: None,
        kind,
    });
}

/// Every role of the seal in canonical traversal order (same names and order
/// as `Qwen35Weights::for_each_dense_tensor`).
fn seal_entries(seal: &Qwen35RankSeal) -> Vec<Entry<'_>> {
    let mut out = Vec::new();
    push_plain(&mut out, "token_embd".into(), &seal.token_embd, "embedding");
    push_plain(&mut out, "output_norm".into(), &seal.output_norm, "norm");
    push_weight(
        &mut out,
        "output",
        &seal.output,
        seal.lm_head_aliases_embd.then_some("token_embd"),
    );
    for (i, layer) in seal.layer_seals.iter().enumerate() {
        let p = format!("layers.{i:04}");
        let name = |n: &str| format!("{p}.{n}");
        match layer {
            Qwen35LayerSeal::DeltaNet {
                attn_norm,
                wqkv,
                wz,
                w_alpha,
                w_beta,
                a_log,
                dt_bias,
                conv_weight,
                norm_weight,
                wo,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
            } => {
                push_plain(&mut out, name("attn_norm"), attn_norm, "norm");
                push_weight(&mut out, &name("wqkv"), wqkv, None);
                push_weight(&mut out, &name("wz"), wz, None);
                push_weight(&mut out, &name("w_alpha"), w_alpha, None);
                push_weight(&mut out, &name("w_beta"), w_beta, None);
                push_plain(&mut out, name("a_log"), a_log, "raw_f32");
                push_plain(&mut out, name("dt_bias"), dt_bias, "raw_f32");
                push_plain(&mut out, name("conv_weight"), conv_weight, "raw_f32");
                push_plain(&mut out, name("norm_weight"), norm_weight, "raw_f32");
                push_weight(&mut out, &name("wo"), wo, None);
                push_plain(&mut out, name("ffn_norm"), ffn_norm, "norm");
                push_weight(&mut out, &name("w_gate"), w_gate, None);
                push_weight(&mut out, &name("w_up"), w_up, None);
                push_weight(&mut out, &name("w_down"), w_down, None);
            }
            Qwen35LayerSeal::FullAttn {
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
            } => {
                push_plain(&mut out, name("attn_norm"), attn_norm, "norm");
                push_weight(&mut out, &name("wq"), wq, None);
                push_weight(&mut out, &name("wk"), wk, None);
                push_weight(&mut out, &name("wv"), wv, None);
                push_weight(&mut out, &name("wo"), wo, None);
                push_plain(&mut out, name("q_norm"), q_norm, "norm");
                push_plain(&mut out, name("k_norm"), k_norm, "norm");
                push_plain(&mut out, name("ffn_norm"), ffn_norm, "norm");
                push_weight(&mut out, &name("w_gate"), w_gate, None);
                push_weight(&mut out, &name("w_up"), w_up, None);
                push_weight(&mut out, &name("w_down"), w_down, None);
            }
            // Dense validation rejects these before entries are consumed.
            Qwen35LayerSeal::DeltaNetMoe { .. } | Qwen35LayerSeal::FullAttnMoe { .. } => {}
        }
    }
    out
}

// ─── Seal geometry / sidecar / LUT validation ───────────────────────────

fn row_bytes_checked(dtype: DType, k: usize) -> Option<usize> {
    dtype.row_bytes(k)
}

fn lut_bytes<const N: usize>(l: &[u32; N]) -> Vec<u8> {
    l.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn numel_checked(role: &str, shape: &[usize]) -> SwResult<usize> {
    shape
        .iter()
        .try_fold(1usize, |a, &s| a.checked_mul(s))
        .ok_or_else(|| SharedWeightError::invalid(format!("{role}: shape overflow")))
}

/// Typed bytes of a tensor whose logical content is exactly `shape` elements
/// of `dtype` (byte-level dtypes count one byte per element). Valid only for
/// plain tensors, byte blobs and `[bytes]`-shaped uploads; packed projection
/// buffers use [`typed_weight_bytes`].
fn generic_typed_bytes(role: &str, d: &GpuTensorDescriptor) -> SwResult<usize> {
    numel_checked(role, &d.shape)?
        .checked_mul(d.dtype.size())
        .ok_or_else(|| SharedWeightError::invalid(format!("{role}: size overflow")))
}

/// Rows of a DeltaNet layer's Z owner after the loader's beta/alpha fold
/// (`append_betaalpha_to_z`): `Some(total rows)` when the layer is folded.
/// Mirrors the loader's predicate: resolved fold flag, all four projections
/// MQ4G256V2, equal K.
fn z_fold_rows(
    flags: &DenseRecipeFlags,
    wqkv: &WeightTensorDescriptor,
    wz: &WeightTensorDescriptor,
    w_beta: &WeightTensorDescriptor,
    w_alpha: &WeightTensorDescriptor,
) -> SwResult<Option<usize>> {
    if !(flags.gdn_fold_gfx12 || flags.gdn_fold_gfx11) {
        return Ok(None);
    }
    let all_mq4v2 = [
        wqkv.gpu_dtype,
        wz.gpu_dtype,
        w_beta.gpu_dtype,
        w_alpha.gpu_dtype,
    ] == [DType::MQ4G256V2; 4];
    if !all_mq4v2 || [wz.k, w_beta.k, w_alpha.k] != [wqkv.k; 3] {
        return Ok(None);
    }
    if flags.gdn_fold_gfx11 {
        return Err(SharedWeightError::unsupported(format!(
            "gfx11 beta/alpha fold (v1 admits {SUPPORTED_DEVICE_ARCH} only)"
        )));
    }
    Ok(Some(Gpu::iu4_gdn_fold_rows(wz.m).0))
}

/// Typed bytes of a projection's `.buf`: `m` packed rows (the folded row
/// count for a folded Z, whose upload shape is `[rows * row_bytes]`).
fn typed_weight_bytes(
    role: &str,
    w: &WeightTensorDescriptor,
    fold_rows: Option<usize>,
) -> SwResult<usize> {
    if w.row_stride == 0 {
        if let Some(rb) = row_bytes_checked(w.gpu_dtype, w.k) {
            let n = rb
                .checked_mul(fold_rows.unwrap_or(w.m))
                .ok_or_else(|| SharedWeightError::invalid(format!("{role}: row bytes overflow")))?;
            // A folded Z is always a `[bytes]` Raw upload; any other Raw
            // byte blob must also be exactly its typed bytes.
            let blob = w.buf.dtype == DType::Raw && w.buf.shape.len() == 1;
            if (fold_rows.is_some() || blob) && w.buf.shape.as_slice() != [n] {
                return Err(SharedWeightError::invalid(format!(
                    "{role}: raw packed shape {:?} != [{n}] (fold rows {fold_rows:?})",
                    w.buf.shape
                )));
            }
            return Ok(n);
        }
    }
    generic_typed_bytes(role, &w.buf)
}

/// Typed bytes of an AWQ sidecar: `K x 3` f32 reciprocal planes (shape
/// `[K, AWQ_RCP_ROWS]`) or the plain scale blob.
fn typed_awq_bytes(
    role: &str,
    w: &WeightTensorDescriptor,
    d: &GpuTensorDescriptor,
) -> SwResult<usize> {
    if d.shape.as_slice() == [w.k, rdna_compute::gemv::AWQ_RCP_ROWS] {
        return w
            .k
            .checked_mul(rdna_compute::gemv::AWQ_RCP_ROWS * 4)
            .ok_or_else(|| SharedWeightError::invalid(format!("{role}: size overflow")));
    }
    generic_typed_bytes(role, d)
}

/// Typed (semantic representation) bytes of every seal role, canonical order.
/// A captured seal carries the accessible source size per role (the
/// allocator may round it up); a normalized seal carries exactly this value.
fn typed_lengths(
    seal: &Qwen35RankSeal,
    flags: &DenseRecipeFlags,
) -> SwResult<Vec<(String, usize)>> {
    let mut folded: HashMap<String, usize> = HashMap::new();
    for (i, layer) in seal.layer_seals.iter().enumerate() {
        if let Qwen35LayerSeal::DeltaNet {
            wqkv,
            wz,
            w_alpha,
            w_beta,
            ..
        } = layer
        {
            if let Some(rows) = z_fold_rows(flags, wqkv, wz, w_beta, w_alpha)? {
                folded.insert(format!("layers.{i:04}.wz.buf"), rows);
            }
        }
    }
    let entries = seal_entries(seal);
    let mut out = Vec::with_capacity(entries.len());
    for e in &entries {
        let n = if let Some((_, w)) = &e.awq_of {
            typed_awq_bytes(&e.role, w, e.desc)?
        } else if e.alias_of.is_some() {
            generic_typed_bytes("token_embd", &seal.token_embd)?
        } else if let Some(w) = e.weight {
            typed_weight_bytes(&e.role, w, folded.get(&e.role).copied())?
        } else {
            generic_typed_bytes(&e.role, e.desc)?
        };
        out.push((e.role.clone(), n));
    }
    Ok(out)
}

fn visit_weight_desc_mut(
    prefix: &str,
    w: &mut WeightTensorDescriptor,
    f: &mut dyn FnMut(String, &mut GpuTensorDescriptor),
) {
    f(format!("{prefix}.buf"), &mut w.buf);
    if let Some(a) = w.awq_scale.as_mut() {
        f(format!("{prefix}.awq_scale"), a);
    }
}

/// Visit every tensor descriptor of the seal by role (same names as
/// [`seal_entries`]).
fn for_each_desc_mut(
    seal: &mut Qwen35RankSeal,
    f: &mut dyn FnMut(String, &mut GpuTensorDescriptor),
) {
    f("token_embd".into(), &mut seal.token_embd);
    f("output_norm".into(), &mut seal.output_norm);
    visit_weight_desc_mut("output", &mut seal.output, f);
    for (i, layer) in seal.layer_seals.iter_mut().enumerate() {
        let p = format!("layers.{i:04}");
        let name = |n: &str| format!("{p}.{n}");
        match layer {
            Qwen35LayerSeal::DeltaNet {
                attn_norm,
                wqkv,
                wz,
                w_alpha,
                w_beta,
                a_log,
                dt_bias,
                conv_weight,
                norm_weight,
                wo,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
            } => {
                f(name("attn_norm"), attn_norm);
                visit_weight_desc_mut(&name("wqkv"), wqkv, f);
                visit_weight_desc_mut(&name("wz"), wz, f);
                visit_weight_desc_mut(&name("w_alpha"), w_alpha, f);
                visit_weight_desc_mut(&name("w_beta"), w_beta, f);
                f(name("a_log"), a_log);
                f(name("dt_bias"), dt_bias);
                f(name("conv_weight"), conv_weight);
                f(name("norm_weight"), norm_weight);
                visit_weight_desc_mut(&name("wo"), wo, f);
                f(name("ffn_norm"), ffn_norm);
                visit_weight_desc_mut(&name("w_gate"), w_gate, f);
                visit_weight_desc_mut(&name("w_up"), w_up, f);
                visit_weight_desc_mut(&name("w_down"), w_down, f);
            }
            Qwen35LayerSeal::FullAttn {
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
            } => {
                f(name("attn_norm"), attn_norm);
                visit_weight_desc_mut(&name("wq"), wq, f);
                visit_weight_desc_mut(&name("wk"), wk, f);
                visit_weight_desc_mut(&name("wv"), wv, f);
                visit_weight_desc_mut(&name("wo"), wo, f);
                f(name("q_norm"), q_norm);
                f(name("k_norm"), k_norm);
                f(name("ffn_norm"), ffn_norm);
                visit_weight_desc_mut(&name("w_gate"), w_gate, f);
                visit_weight_desc_mut(&name("w_up"), w_up, f);
                visit_weight_desc_mut(&name("w_down"), w_down, f);
            }
            // Dense validation rejects these before descriptors are consumed.
            Qwen35LayerSeal::DeltaNetMoe { .. } | Qwen35LayerSeal::FullAttnMoe { .. } => {}
        }
    }
}

/// Canonicalize a captured dense seal for sharing: every descriptor's
/// `byte_len` becomes the typed length of its role, never the accessible
/// source size (pool buckets and the 256 B minimum round that up). A source
/// smaller than its typed extent, or typed geometry that disagrees with the
/// config/flags, is rejected. The result passes [`validate_dense_seal`].
fn normalize_dense_seal(
    source: &Qwen35RankSeal,
    flags: &DenseRecipeFlags,
    config: &Qwen35Config,
) -> SwResult<Qwen35RankSeal> {
    check_dense_config(config)?;
    source
        .validate_dense()
        .map_err(hip_err("typed dense seal"))?;
    let lens = typed_lengths(source, flags)?;
    let entries = seal_entries(source);
    for ((role, typed), e) in lens.iter().zip(&entries) {
        if e.desc.byte_len < *typed {
            return Err(SharedWeightError::invalid(format!(
                "{role}: accessible source {} B < typed {typed} B",
                e.desc.byte_len
            )));
        }
    }
    let typed: HashMap<&str, usize> = lens.iter().map(|(r, n)| (r.as_str(), *n)).collect();
    let mut sealed = source.clone();
    let mut unmatched = Vec::new();
    for_each_desc_mut(&mut sealed, &mut |role, d| match typed.get(role.as_str()) {
        Some(&n) => d.byte_len = n,
        None => unmatched.push(role),
    });
    if !unmatched.is_empty() || typed.len() != entries.len() {
        return Err(SharedWeightError::invalid(format!(
            "typed length roles drifted from seal roles (unmatched {unmatched:?})"
        )));
    }
    validate_dense_seal(&sealed, flags, config)?;
    Ok(sealed)
}

fn normalized_seal_of(
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    flags: &DenseRecipeFlags,
) -> SwResult<Qwen35RankSeal> {
    let source = Qwen35RankSeal::capture_dense(weights).map_err(hip_err("dense seal capture"))?;
    normalize_dense_seal(&source, flags, config)
}

/// Typed (semantic representation) byte length of every dense tensor role of
/// a loaded graph, resolved against the live GPU's recipe flags: packed
/// projections `m * row_bytes`, a folded Z `(m + fold rows) * row_bytes`, AWQ
/// reciprocal planes `K * 3 * 4`, plain tensors `numel * dtype.size()`. Never
/// the (possibly larger) accessible `buf.size()`; fails when any source
/// buffer is smaller than its typed length. Includes a tied head's alias role.
pub fn dense_typed_tensor_lengths(
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    gpu: &Gpu,
) -> Result<BTreeMap<String, usize>, SharedWeightError> {
    let flags = resolve_dense_flags(gpu, config)?;
    let seal = normalized_seal_of(weights, config, &flags)?;
    Ok(seal_entries(&seal)
        .into_iter()
        .map(|e| (e.role, e.desc.byte_len))
        .collect())
}

/// Typed `elems`-wide F32 tensor; `d.byte_len` is its exact typed size.
fn check_plain(role: &str, d: &GpuTensorDescriptor, elems: usize) -> SwResult<()> {
    let bytes = elems
        .checked_mul(4)
        .ok_or_else(|| SharedWeightError::invalid(format!("{role}: size overflow")))?;
    let numel = numel_checked(role, &d.shape)?;
    if d.dtype != DType::F32 || numel != elems || d.byte_len != bytes {
        return Err(SharedWeightError::invalid(format!(
            "{role}: expected F32 x{elems} ({bytes} B), found {:?} shape {:?} {} B",
            d.dtype, d.shape, d.byte_len
        )));
    }
    Ok(())
}

struct WeightRule {
    m: usize,
    k: usize,
}

fn check_weight(
    role: &str,
    w: &WeightTensorDescriptor,
    rule: &WeightRule,
    flags: &DenseRecipeFlags,
) -> SwResult<()> {
    let bad = |msg: String| SharedWeightError::invalid(format!("{role}: {msg}"));
    if w.paro.is_some() {
        return Err(SharedWeightError::unsupported(format!(
            "{role}: Paro sidecar"
        )));
    }
    if w.m != rule.m || w.k != rule.k {
        return Err(bad(format!(
            "m/k {}x{} != config geometry {}x{}",
            w.m, w.k, rule.m, rule.k
        )));
    }
    if w.buf.byte_len == 0 {
        return Err(bad("empty buffer".into()));
    }
    // Lloyd LUTs: MQ4G256V2Lloyd needs all three arrays; no other dtype may carry any.
    let luts = usize::from(w.lloyd_lut_e4m3.is_some())
        + usize::from(w.lloyd_lut_f16.is_some())
        + usize::from(w.lloyd_lut_c16.is_some());
    if w.gpu_dtype == DType::MQ4G256V2Lloyd {
        if luts != 3 {
            return Err(SharedWeightError::unsupported(format!(
                "{role}: MQ4G256V2Lloyd weight is missing Lloyd LUTs ({luts} of 3)"
            )));
        }
    } else if luts != 0 {
        return Err(bad(format!("{:?} weight carries Lloyd LUTs", w.gpu_dtype)));
    }
    if let Some(a) = &w.awq_scale {
        if !w.gpu_dtype.supports_awq_sidecar() {
            return Err(bad(format!(
                "{:?} cannot carry an AWQ sidecar",
                w.gpu_dtype
            )));
        }
        let k4 = w.k.checked_mul(4);
        let k12 = w.k.checked_mul(12);
        let ok = Some(a.byte_len) == k4 || (Some(a.byte_len) == k12 && flags.a4_rms_fdiv_planes);
        if !ok {
            return Err(bad(format!(
                "AWQ sidecar {} B for K={} (planes enabled: {})",
                a.byte_len, w.k, flags.a4_rms_fdiv_planes
            )));
        }
    }
    Ok(())
}

/// Validate a decoded/normalized dense seal against the local config and
/// flags. Pure: no GPU, no FD. Covers layer topology, every tensor's geometry,
/// sidecar sizes, Lloyd LUT completeness and tied-head alias rules. Every
/// descriptor `byte_len` must equal the typed length of its role (see
/// [`dense_typed_tensor_lengths`]); the exporter canonicalizes a captured
/// seal before validating.
pub fn validate_dense_seal(
    seal: &Qwen35RankSeal,
    flags: &DenseRecipeFlags,
    config: &Qwen35Config,
) -> SwResult<()> {
    check_dense_config(config)?;
    seal.validate_dense().map_err(hip_err("typed dense seal"))?;
    if seal.layer_seals.len() != config.n_layers {
        return Err(SharedWeightError::invalid(format!(
            "seal has {} layers, config {}",
            seal.layer_seals.len(),
            config.n_layers
        )));
    }
    let mul = |a: usize, b: usize| {
        a.checked_mul(b)
            .ok_or_else(|| SharedWeightError::invalid("config dimension overflow"))
    };
    let (dims, _) = delta_dims(config)?;
    let (qkv_dim, d_inner) = (dims[0], dims[1]);
    let q_out = mul(mul(config.n_heads, config.head_dim)?, 2)?;
    let kv_dim = mul(config.n_kv_heads, config.head_dim)?;
    let o_in = mul(config.n_heads, config.head_dim)?;
    let conv_elems = mul(qkv_dim, config.conv_kernel_dim)?;
    let (dim, hidden) = (config.dim, config.hidden_dim);

    // Globals.
    let tok = &seal.token_embd;
    if tok.byte_len == 0 {
        return Err(SharedWeightError::invalid("token_embd: empty"));
    }
    if seal.embd_format == EmbeddingFormat::F32
        && tok.byte_len != mul(mul(config.vocab_size, dim)?, 4)?
    {
        return Err(SharedWeightError::invalid(
            "token_embd: F32 table size != vocab*dim*4",
        ));
    }
    check_plain("output_norm", &seal.output_norm, dim)?;
    check_weight(
        "output",
        &seal.output,
        &WeightRule {
            m: config.vocab_size,
            k: dim,
        },
        flags,
    )?;
    if seal.lm_head_aliases_embd && seal.output.buf.byte_len != tok.byte_len {
        return Err(SharedWeightError::invalid(
            "tied output.buf extent != token_embd extent",
        ));
    }

    for (i, (layer, ty)) in seal.layer_seals.iter().zip(&config.layer_types).enumerate() {
        let r = |n: &str| format!("layers.{i:04}.{n}");
        let w = |m: usize, k: usize| WeightRule { m, k };
        match (layer, ty) {
            (
                Qwen35LayerSeal::DeltaNet {
                    attn_norm,
                    wqkv,
                    wz,
                    w_alpha,
                    w_beta,
                    a_log,
                    dt_bias,
                    conv_weight,
                    norm_weight,
                    wo,
                    ffn_norm,
                    w_gate,
                    w_up,
                    w_down,
                },
                LayerType::LinearAttention,
            ) => {
                let heads = config.linear_num_value_heads;
                check_plain(&r("attn_norm"), attn_norm, dim)?;
                check_weight(&r("wqkv"), wqkv, &w(qkv_dim, dim), flags)?;
                check_weight(&r("wz"), wz, &w(d_inner, dim), flags)?;
                check_weight(&r("w_alpha"), w_alpha, &w(heads, dim), flags)?;
                check_weight(&r("w_beta"), w_beta, &w(heads, dim), flags)?;
                check_plain(&r("a_log"), a_log, heads)?;
                check_plain(&r("dt_bias"), dt_bias, heads)?;
                check_plain(&r("conv_weight"), conv_weight, conv_elems)?;
                check_plain(&r("norm_weight"), norm_weight, config.linear_value_head_dim)?;
                check_weight(&r("wo"), wo, &w(dim, d_inner), flags)?;
                check_plain(&r("ffn_norm"), ffn_norm, dim)?;
                check_weight(&r("w_gate"), w_gate, &w(hidden, dim), flags)?;
                check_weight(&r("w_up"), w_up, &w(hidden, dim), flags)?;
                check_weight(&r("w_down"), w_down, &w(dim, hidden), flags)?;
            }
            (
                Qwen35LayerSeal::FullAttn {
                    attn_norm,
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                    ffn_norm,
                    w_gate,
                    w_up,
                    w_down,
                },
                LayerType::FullAttention,
            ) => {
                check_plain(&r("attn_norm"), attn_norm, dim)?;
                check_weight(&r("wq"), wq, &w(q_out, dim), flags)?;
                check_weight(&r("wk"), wk, &w(kv_dim, dim), flags)?;
                check_weight(&r("wv"), wv, &w(kv_dim, dim), flags)?;
                check_weight(&r("wo"), wo, &w(dim, o_in), flags)?;
                check_plain(&r("q_norm"), q_norm, config.head_dim)?;
                check_plain(&r("k_norm"), k_norm, config.head_dim)?;
                check_plain(&r("ffn_norm"), ffn_norm, dim)?;
                check_weight(&r("w_gate"), w_gate, &w(hidden, dim), flags)?;
                check_weight(&r("w_up"), w_up, &w(hidden, dim), flags)?;
                check_weight(&r("w_down"), w_down, &w(dim, hidden), flags)?;
            }
            _ => {
                return Err(SharedWeightError::invalid(format!(
                    "layer {i}: seal kind does not match config layer type {ty:?}"
                )));
            }
        }
    }
    // Every descriptor extent is the typed length of its role.
    for ((role, typed), e) in typed_lengths(seal, flags)?.iter().zip(seal_entries(seal)) {
        if e.desc.byte_len != *typed {
            return Err(SharedWeightError::invalid(format!(
                "{role}: descriptor {} B != typed {typed} B (flags {flags:?})",
                e.desc.byte_len
            )));
        }
    }
    Ok(())
}

// ─── Views ──────────────────────────────────────────────────────────────

fn row_major_strides(shape: &[usize]) -> SwResult<Vec<u64>> {
    let mut strides = vec![0u64; shape.len()];
    let mut acc: u64 = 1;
    for (i, &s) in shape.iter().enumerate().rev() {
        strides[i] = acc;
        acc = acc
            .checked_mul(s as u64)
            .ok_or_else(|| SharedWeightError::invalid("stride overflow"))?;
    }
    Ok(strides)
}

fn build_view(e: &Entry<'_>) -> SwResult<WeightView> {
    let owner_id = e.alias_of.clone().unwrap_or_else(|| e.role.clone());
    let mut meta = CanonicalFields::new();
    meta.insert("kind".into(), fv_str(e.kind));
    meta.insert(
        "dtype_tag".into(),
        fv_u64(u64::from(dtype_tag(e.desc.dtype))),
    );
    let (matrix, quant, companions) = if let Some(w) = e.weight {
        meta.insert("gpu_dtype".into(), fv_str(dtype_name(w.gpu_dtype)));
        meta.insert(
            "gpu_dtype_tag".into(),
            fv_u64(u64::from(dtype_tag(w.gpu_dtype))),
        );
        meta.insert("weight_m".into(), fv_u64(w.m as u64));
        meta.insert("weight_k".into(), fv_u64(w.k as u64));
        meta.insert("weight_row_stride".into(), fv_u64(w.row_stride as u64));
        if let Some(l) = &w.lloyd_lut_e4m3 {
            meta.insert("lloyd_lut_e4m3".into(), fv_bytes(lut_bytes(l)));
        }
        if let Some(l) = &w.lloyd_lut_f16 {
            meta.insert("lloyd_lut_f16".into(), fv_bytes(lut_bytes(l)));
        }
        if let Some(l) = &w.lloyd_lut_c16 {
            meta.insert("lloyd_lut_c16".into(), fv_bytes(lut_bytes(l)));
        }
        let mut companions = BTreeMap::new();
        if w.awq_scale.is_some() {
            companions.insert(
                "awq_scale".to_owned(),
                e.role.trim_end_matches(".buf").to_owned() + ".awq_scale",
            );
        }
        let quant = quant_block(w.gpu_dtype).map(|(group, elems, bytes, packing)| QuantDesc {
            format: dtype_name(w.gpu_dtype).to_owned(),
            group_size: group as u32,
            block_elems: elems as u32,
            block_bytes: bytes as u32,
            packing: packing.to_owned(),
        });
        // Matrix view only when the effective row stride is known and the
        // rows fit the extent (the runtime validates `m * stride <= extent`);
        // m/k/row_stride are always carried in the metadata fields too.
        let stride = if w.row_stride != 0 {
            Some(w.row_stride)
        } else {
            row_bytes_checked(w.gpu_dtype, w.k)
        };
        let matrix = stride
            .filter(|&s| {
                s != 0
                    && w.m != 0
                    && w.k != 0
                    && w.m.checked_mul(s).is_some_and(|n| n <= e.desc.byte_len)
            })
            .map(|s| MatrixDesc {
                m: w.m as u64,
                k: w.k as u64,
                row_stride_bytes: s as u64,
            });
        (matrix, quant, companions)
    } else {
        (None, None, BTreeMap::new())
    };
    if let Some((buf_role, w)) = &e.awq_of {
        meta.insert("awq_for".into(), fv_str(buf_role));
        meta.insert("awq_k".into(), fv_u64(w.k as u64));
        meta.insert(
            "awq_reciprocal_planes".into(),
            fv_bool(e.desc.byte_len == w.k * 12),
        );
    }
    Ok(WeightView {
        role: e.role.clone(),
        owner_id: owner_id.clone(),
        byte_offset: 0,
        byte_extent: e.desc.byte_len as u64,
        dtype: dtype_name(e.desc.dtype).to_owned(),
        shape: e.desc.shape.iter().map(|&s| s as u64).collect(),
        strides: row_major_strides(&e.desc.shape)?,
        matrix,
        quant,
        companions,
        alias_of: e.alias_of.clone(),
        metadata: meta,
    })
}

/// Deterministic view table of a seal, sorted by role (manifest order).
pub fn views_from_seal(seal: &Qwen35RankSeal) -> SwResult<Vec<WeightView>> {
    let mut views = seal_entries(seal)
        .iter()
        .map(build_view)
        .collect::<SwResult<Vec<_>>>()?;
    views.sort_by(|a, b| a.role.cmp(&b.role));
    Ok(views)
}

/// Owner ids (sorted bytewise, FD order) with their logical byte size.
fn owners_from_seal(seal: &Qwen35RankSeal) -> Vec<(String, usize)> {
    let mut owners: Vec<(String, usize)> = seal_entries(seal)
        .into_iter()
        .filter(|e| e.alias_of.is_none())
        .map(|e| (e.role, e.desc.byte_len))
        .collect();
    owners.sort_by(|a, b| a.0.cmp(&b.0));
    owners
}

// ─── Allocation extents ─────────────────────────────────────────────────

/// `round_up(logical + tail, granularity)` — the only legal physical size.
pub fn expected_physical_bytes(logical: u64, granularity: u64) -> Option<u64> {
    if granularity == 0 {
        return None;
    }
    let padded = logical.checked_add(EXPORT_WEIGHT_TAIL_PAD_BYTES)?;
    padded.div_ceil(granularity).checked_mul(granularity)
}

fn check_extents(owner: &str, physical: u64, logical: u64, granularity: u64) -> SwResult<()> {
    if logical == 0 || granularity == 0 || physical % granularity != 0 {
        return Err(SharedWeightError::invalid(format!(
            "{owner}: bad extents physical {physical} logical {logical} granularity {granularity}"
        )));
    }
    if physical < logical.saturating_add(EXPORT_WEIGHT_TAIL_PAD_BYTES)
        || Some(physical) != expected_physical_bytes(logical, granularity)
    {
        return Err(SharedWeightError::invalid(format!(
            "{owner}: physical {physical} != round_up(logical {logical} + {EXPORT_WEIGHT_TAIL_PAD_BYTES}, {granularity})"
        )));
    }
    Ok(())
}

// ─── Live graph census ──────────────────────────────────────────────────

struct Rec {
    role: String,
    order: usize,
    ptr: usize,
    size: usize,
    borrowed: bool,
    /// Typed (semantic) bytes of the role; `size` is the accessible
    /// allocation, which may exceed it (allocator slack).
    typed: usize,
    host: bool,
    vmm: bool,
    shape: Vec<usize>,
    dtype: DType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CensusOwner {
    /// Canonical owner id == typed-traversal role.
    pub owner_id: String,
    /// Traversal position (canonical order, not sorted order).
    pub order: usize,
    /// Typed (semantic representation) bytes: what is sealed, exported and
    /// imported. Never the accessible source size.
    pub logical_bytes: u64,
    /// Accessible bytes of the live allocation (`logical_bytes` plus allocator
    /// slack; equal once the owner is an exported VMM allocation).
    pub source_bytes: u64,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CensusAlias {
    pub role: String,
    pub order: usize,
    pub owner_id: String,
    pub byte_offset: u64,
    pub byte_extent: u64,
}

/// Complete unique-owner census of a dense graph from typed traversal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseCensus {
    /// Owners in canonical traversal order.
    pub owners: Vec<CensusOwner>,
    pub aliases: Vec<CensusAlias>,
    /// Sum of owners' typed bytes.
    pub logical_bytes: u64,
    pub largest_owner_bytes: u64,
    /// Sum of owners' accessible allocation bytes.
    pub source_bytes: u64,
    /// `source_bytes - logical_bytes`: allocator slack carried by the source
    /// allocations, never sealed or exported.
    pub source_slack_bytes: u64,
}

impl DenseCensus {
    /// Owner ids sorted bytewise: manifest allocation / FD order.
    pub fn sorted_owner_ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self.owners.iter().map(|o| o.owner_id.as_str()).collect();
        ids.sort_unstable();
        ids
    }
}

fn collect_records(weights: &Qwen35Weights, typed: &HashMap<String, usize>) -> SwResult<Vec<Rec>> {
    let mut recs = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    weights
        .for_each_dense_tensor(&mut |role, t| {
            let typed_bytes = match typed.get(role) {
                Some(&n) => n,
                None => {
                    missing.push(role.to_owned());
                    0
                }
            };
            recs.push(Rec {
                role: role.to_owned(),
                order: recs.len(),
                ptr: t.buf.as_ptr() as usize,
                size: t.buf.size(),
                borrowed: t.buf.is_borrowed(),
                typed: typed_bytes,
                host: t.buf.is_host_mapped(),
                vmm: t.buf.is_vmm_owner(),
                shape: t.shape.clone(),
                dtype: t.dtype,
            });
            Ok(())
        })
        .map_err(hip_err("typed dense traversal"))?;
    if !missing.is_empty() {
        return Err(SharedWeightError::invalid(format!(
            "live roles without a typed length: {missing:?}"
        )));
    }
    Ok(recs)
}

fn census_from_records(recs: &[Rec], require_vmm: bool) -> SwResult<DenseCensus> {
    let mut sorted: Vec<&Rec> = Vec::new();
    for r in recs.iter().filter(|r| !r.borrowed) {
        if r.host {
            return Err(SharedWeightError::unsupported(format!(
                "{}: host-mapped owner",
                r.role
            )));
        }
        if r.ptr == 0 || r.size == 0 {
            return Err(SharedWeightError::invalid(format!(
                "{}: null/empty owner",
                r.role
            )));
        }
        if r.size < r.typed {
            return Err(SharedWeightError::invalid(format!(
                "{}: accessible owner {} B < typed {} B",
                r.role, r.size, r.typed
            )));
        }
        if require_vmm && !r.vmm {
            return Err(SharedWeightError::invalid(format!(
                "{}: owner is not a VMM allocation",
                r.role
            )));
        }
        sorted.push(r);
    }
    sorted.sort_by_key(|r| r.ptr);
    for pair in sorted.windows(2) {
        let end = pair[0]
            .ptr
            .checked_add(pair[0].size)
            .ok_or_else(|| SharedWeightError::invalid("owner range overflow"))?;
        if end > pair[1].ptr {
            return Err(SharedWeightError::invalid(format!(
                "owners {} and {} overlap or share an address",
                pair[0].role, pair[1].role
            )));
        }
    }
    let mut aliases = Vec::new();
    for r in recs.iter().filter(|r| r.borrowed) {
        if r.ptr == 0 || r.size == 0 {
            return Err(SharedWeightError::invalid(format!(
                "{}: null/empty alias",
                r.role
            )));
        }
        if r.size < r.typed {
            return Err(SharedWeightError::invalid(format!(
                "{}: accessible alias {} B < typed {} B",
                r.role, r.size, r.typed
            )));
        }
        let idx = sorted.partition_point(|o| o.ptr <= r.ptr);
        let owner = idx
            .checked_sub(1)
            .map(|i| sorted[i])
            .ok_or_else(|| SharedWeightError::invalid(format!("{}: alias has no owner", r.role)))?;
        let alias_end = r
            .ptr
            .checked_add(r.size)
            .ok_or_else(|| SharedWeightError::invalid("alias range overflow"))?;
        if alias_end > owner.ptr.saturating_add(owner.size) {
            return Err(SharedWeightError::invalid(format!(
                "{}: alias range exceeds owner {}",
                r.role, owner.role
            )));
        }
        if r.order < owner.order {
            return Err(SharedWeightError::unsupported(format!(
                "{}: alias precedes its owner {} in canonical order",
                r.role, owner.role
            )));
        }
        aliases.push(CensusAlias {
            role: r.role.clone(),
            order: r.order,
            owner_id: owner.role.clone(),
            byte_offset: (r.ptr - owner.ptr) as u64,
            byte_extent: r.typed as u64,
        });
    }
    let mut owners: Vec<CensusOwner> = sorted
        .iter()
        .map(|r| CensusOwner {
            owner_id: r.role.clone(),
            order: r.order,
            logical_bytes: r.typed as u64,
            source_bytes: r.size as u64,
            shape: r.shape.clone(),
            dtype: r.dtype,
        })
        .collect();
    owners.sort_by_key(|o| o.order);
    let logical_bytes: u64 = owners.iter().map(|o| o.logical_bytes).sum();
    let largest_owner_bytes = owners.iter().map(|o| o.logical_bytes).max().unwrap_or(0);
    let source_bytes: u64 = owners.iter().map(|o| o.source_bytes).sum();
    Ok(DenseCensus {
        owners,
        aliases,
        logical_bytes,
        largest_owner_bytes,
        source_bytes,
        source_slack_bytes: source_bytes - logical_bytes,
    })
}

fn typed_map_of_seal(seal: &Qwen35RankSeal) -> HashMap<String, usize> {
    seal_entries(seal)
        .into_iter()
        .map(|e| (e.role, e.desc.byte_len))
        .collect()
}

/// Complete unique-owner/alias census via typed traversal (no pointer
/// scanning). Public so the oracle can hash/compare owners by id. Owner
/// `logical_bytes` are typed lengths (see [`dense_typed_tensor_lengths`]);
/// accessible allocation bytes and their slack are reported separately.
pub fn census_dense_weights(
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    gpu: &Gpu,
) -> SwResult<DenseCensus> {
    let flags = resolve_dense_flags(gpu, config)?;
    let seal = normalized_seal_of(weights, config, &flags)?;
    census_from_records(&collect_records(weights, &typed_map_of_seal(&seal))?, false)
}

/// Census equals the seal's role table: same roles/order, extents, shapes,
/// dtypes and alias topology (catches any traversal/seal naming drift).
fn check_census_matches_seal(
    recs: &[Rec],
    census: &DenseCensus,
    seal: &Qwen35RankSeal,
) -> SwResult<()> {
    let entries = seal_entries(seal);
    if entries.len() != recs.len() {
        return Err(SharedWeightError::invalid(format!(
            "seal has {} roles, live graph {}",
            entries.len(),
            recs.len()
        )));
    }
    for (e, r) in entries.iter().zip(recs) {
        if e.role != r.role
            || e.desc.byte_len != r.typed
            || e.desc.shape != r.shape
            || e.desc.dtype != r.dtype
            || e.alias_of.is_some() != r.borrowed
        {
            return Err(SharedWeightError::invalid(format!(
                "live tensor {} disagrees with seal role {}",
                r.role, e.role
            )));
        }
    }
    for a in &census.aliases {
        let owner = census
            .owners
            .iter()
            .find(|o| o.owner_id == a.owner_id)
            .ok_or_else(|| SharedWeightError::invalid("alias owner missing from census"))?;
        // Hydration rebuilds a tied head as a full-extent view of its owner.
        if a.byte_offset != 0 || a.byte_extent != owner.logical_bytes {
            return Err(SharedWeightError::unsupported(format!(
                "{}: only a full-extent offset-0 alias can be hydrated",
                a.role
            )));
        }
        let entry = entries
            .iter()
            .find(|e| e.role == a.role)
            .ok_or_else(|| SharedWeightError::invalid("alias role missing from seal"))?;
        if entry.alias_of.as_deref() != Some(a.owner_id.as_str()) {
            return Err(SharedWeightError::invalid(format!(
                "{}: live alias owner {} != sealed {:?}",
                a.role, a.owner_id, entry.alias_of
            )));
        }
    }
    Ok(())
}

// ─── Cleanup ────────────────────────────────────────────────────────────

fn empty_tensor() -> GpuTensor {
    GpuTensor {
        // SAFETY: a null, zero-sized borrowed descriptor never reaches HIP.
        buf: unsafe { DeviceBuffer::from_raw(std::ptr::null_mut(), 0) },
        shape: vec![0],
        dtype: DType::Raw,
    }
}

/// Release one tensor through the path matching its real ownership.
/// Borrowed aliases / empty tombstones own nothing and are skipped.
fn release_tensor(gpu: &mut Gpu, role: &str, t: GpuTensor, failures: &mut Vec<String>) {
    if t.buf.is_borrowed() {
        return;
    }
    let result = if t.buf.is_hip_allocation() {
        gpu.release_tensor_immediate(t)
    } else {
        gpu.free_tensor(t)
    };
    if let Err(e) = result {
        failures.push(format!("{role}: {e}"));
    }
}

/// Move every tensor out of `weights` without validation (works on partially
/// promoted graphs) and release the unique owners exactly once.
fn release_all_weights(gpu: &mut Gpu, mut weights: Qwen35Weights, failures: &mut Vec<String>) {
    let take = |role: String, t: &mut GpuTensor, out: &mut Vec<(String, GpuTensor)>| {
        out.push((role, std::mem::replace(t, empty_tensor())));
    };
    let mut taken: Vec<(String, GpuTensor)> = Vec::new();
    take("token_embd".into(), &mut weights.token_embd, &mut taken);
    take("output_norm".into(), &mut weights.output_norm, &mut taken);
    take_weight(&mut weights.output, "output", &mut taken);
    let layers = std::mem::take(&mut weights.layers);
    let mut foreign = Vec::new();
    for (i, layer) in layers.into_iter().enumerate() {
        let p = format!("layers.{i:04}");
        match layer {
            LayerWeights::DeltaNet(mut l) => {
                let l: &mut DeltaNetLayerWeights = &mut l;
                take(format!("{p}.attn_norm"), &mut l.attn_norm, &mut taken);
                take_weight(&mut l.wqkv, &format!("{p}.wqkv"), &mut taken);
                take_weight(&mut l.wz, &format!("{p}.wz"), &mut taken);
                take_weight(&mut l.w_alpha, &format!("{p}.w_alpha"), &mut taken);
                take_weight(&mut l.w_beta, &format!("{p}.w_beta"), &mut taken);
                take(format!("{p}.a_log"), &mut l.a_log, &mut taken);
                take(format!("{p}.dt_bias"), &mut l.dt_bias, &mut taken);
                take(format!("{p}.conv_weight"), &mut l.conv_weight, &mut taken);
                take(format!("{p}.norm_weight"), &mut l.norm_weight, &mut taken);
                take_weight(&mut l.wo, &format!("{p}.wo"), &mut taken);
                take(format!("{p}.ffn_norm"), &mut l.ffn_norm, &mut taken);
                take_weight(&mut l.w_gate, &format!("{p}.w_gate"), &mut taken);
                take_weight(&mut l.w_up, &format!("{p}.w_up"), &mut taken);
                take_weight(&mut l.w_down, &format!("{p}.w_down"), &mut taken);
            }
            LayerWeights::FullAttn(mut l) => {
                let l: &mut FullAttnLayerWeights = &mut l;
                take(format!("{p}.attn_norm"), &mut l.attn_norm, &mut taken);
                take_weight(&mut l.wq, &format!("{p}.wq"), &mut taken);
                take_weight(&mut l.wk, &format!("{p}.wk"), &mut taken);
                take_weight(&mut l.wv, &format!("{p}.wv"), &mut taken);
                take_weight(&mut l.wo, &format!("{p}.wo"), &mut taken);
                take(format!("{p}.q_norm"), &mut l.q_norm, &mut taken);
                take(format!("{p}.k_norm"), &mut l.k_norm, &mut taken);
                take(format!("{p}.ffn_norm"), &mut l.ffn_norm, &mut taken);
                take_weight(&mut l.w_gate, &format!("{p}.w_gate"), &mut taken);
                take_weight(&mut l.w_up, &format!("{p}.w_up"), &mut taken);
                take_weight(&mut l.w_down, &format!("{p}.w_down"), &mut taken);
            }
            other => foreign.push(other),
        }
    }
    // Non-dense layers cannot occur after the dense gate; free them with the
    // incumbent routine rather than leak if a caller bypassed it.
    for layer in foreign {
        layer.free_gpu(gpu);
    }
    if let Some(pager) = weights.pager.take() {
        pager.into_inner().free_all(gpu);
    }
    for (role, t) in taken {
        release_tensor(gpu, &role, t, failures);
    }
}

fn take_weight(
    w: &mut hipfire_runtime::llama::WeightTensor,
    prefix: &str,
    out: &mut Vec<(String, GpuTensor)>,
) {
    out.push((
        format!("{prefix}.buf"),
        std::mem::replace(&mut w.buf, empty_tensor()),
    ));
    if let Some(a) = w.awq_scale.as_mut() {
        out.push((
            format!("{prefix}.awq_scale"),
            std::mem::replace(a, empty_tensor()),
        ));
    }
    if let Some(p) = w.paro.as_mut() {
        if !p.is_alias {
            for (n, t) in [
                ("pairs", &mut p.pairs),
                ("theta", &mut p.theta),
                ("channel_scales", &mut p.channel_scales),
            ] {
                out.push((
                    format!("{prefix}.paro.{n}"),
                    std::mem::replace(t, empty_tensor()),
                ));
            }
        }
    }
}

// ─── Export ─────────────────────────────────────────────────────────────

/// Exported unique owners plus everything the host needs to publish them.
/// Dropping it without [`ExportedDenseWeights::free_gpu`] leaks the GPU owners
/// (a `GpuTensor` drop is not a release).
pub struct ExportedDenseWeights {
    /// Final typed graph: every owner is a registered read-only VMM
    /// allocation; tied aliases are borrowed views of their owner.
    pub weights: Qwen35Weights,
    /// Complete manifest (canonicalized and validated).
    pub manifest: WeightManifest,
    /// One FD per manifest allocation, same index order as
    /// `manifest.allocations` (sorted by owner id).
    pub allocation_fds: Vec<OwnedFd>,
    /// Census of the exported graph (traversal order).
    pub census: DenseCensus,
    /// Resolved flags the manifest/seal were built under.
    pub flags: DenseRecipeFlags,
    /// Sum of mapped physical bytes of all allocations.
    pub physical_bytes: u64,
}

impl ExportedDenseWeights {
    /// Release every owner of the exported graph (host teardown) and close the
    /// FDs. Failures are reported, never swallowed; failed VMM arenas stay in
    /// the `Gpu` orphan table.
    pub fn free_gpu(self, gpu: &mut Gpu) -> SwResult<()> {
        let mut failures = Vec::new();
        let Self {
            weights,
            allocation_fds,
            ..
        } = self;
        drop(allocation_fds);
        release_all_weights(gpu, weights, &mut failures);
        if failures.is_empty() {
            Ok(())
        } else {
            Err(SharedWeightError::Cleanup {
                cause: Box::new(SharedWeightError::invalid(
                    "exported dense weights teardown",
                )),
                failures,
            })
        }
    }
}

struct Promoted {
    owner_id: String,
    fd: OwnedFd,
    physical: u64,
    logical: u64,
    granularity: u64,
}

#[derive(Default)]
struct ExportCtx {
    promoted: Vec<Promoted>,
    /// Tensors owned by neither the graph nor a finished release: preserved
    /// sources after a sync failure, or a new owner whose extents were
    /// rejected. Released during cleanup.
    stranded: Vec<(String, GpuTensor)>,
}

struct Prepared {
    flags: DenseRecipeFlags,
    seal: Qwen35RankSeal,
    census: DenseCensus,
}

fn export_prepare(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    recipe: &LayoutRecipe,
) -> SwResult<Prepared> {
    check_dense_config(config)?;
    refuse_host_mapped_weights(gpu)?;
    if gpu.arch != SUPPORTED_DEVICE_ARCH {
        return Err(SharedWeightError::unsupported(format!(
            "device arch {} (v1 admits {SUPPORTED_DEVICE_ARCH})",
            gpu.arch
        )));
    }
    let flags = resolve_dense_flags(gpu, config)?;
    let expected = recipe_fields(&gpu.arch, &flags);
    if recipe.schema != DENSE_RECIPE_SCHEMA || recipe.fields != expected {
        return Err(SharedWeightError::invalid(
            "layout recipe does not match the recipe resolved from live GPU state",
        ));
    }
    // The captured seal carries accessible source sizes; the shared seal,
    // census and every extent below use the typed lengths instead.
    let seal = normalized_seal_of(weights, config, &flags)?;
    let recs = collect_records(weights, &typed_map_of_seal(&seal))?;
    let census = census_from_records(&recs, false)?;
    check_census_matches_seal(&recs, &census, &seal)?;
    Ok(Prepared {
        flags,
        seal,
        census,
    })
}

/// Promote every unique owner to an exportable VMM allocation, one owner at a
/// time, in a single typed traversal: create+copy → swap the slot → rebind every
/// alias → release the old owner immediately. Any failure stops the traversal;
/// `ctx` retains everything that still needs cleanup.
fn export_promote(
    gpu: &mut Gpu,
    weights: &mut Qwen35Weights,
    census: &DenseCensus,
    ctx: &mut ExportCtx,
) -> SwResult<()> {
    let owner_ids: HashSet<&str> = census.owners.iter().map(|o| o.owner_id.as_str()).collect();
    let owner_typed: HashMap<&str, usize> = census
        .owners
        .iter()
        .map(|o| (o.owner_id.as_str(), o.logical_bytes as usize))
        .collect();
    let mut alias_to_owner: HashMap<&str, &str> = HashMap::new();
    let mut alias_extent: HashMap<&str, usize> = HashMap::new();
    let mut pending_aliases: HashMap<&str, usize> = HashMap::new();
    for a in &census.aliases {
        alias_to_owner.insert(a.role.as_str(), a.owner_id.as_str());
        alias_extent.insert(a.role.as_str(), a.byte_extent as usize);
        *pending_aliases.entry(a.owner_id.as_str()).or_insert(0) += 1;
    }
    let mut new_ptr: HashMap<String, usize> = HashMap::new();
    let mut old_hold: HashMap<String, GpuTensor> = HashMap::new();
    let mut stage_error: Option<SharedWeightError> = None;

    let traversal = weights.for_each_dense_tensor_mut(&mut |role, slot| {
        let fail = |stage_error: &mut Option<SharedWeightError>,
                    e: SharedWeightError|
         -> Result<(), HipError> {
            *stage_error = Some(e);
            Err(HipError::new(0, "shared dense export aborted"))
        };
        if let Some(&owner) = alias_to_owner.get(role) {
            let Some(&ptr) = new_ptr.get(owner) else {
                return fail(
                    &mut stage_error,
                    SharedWeightError::invalid(format!(
                        "{role}: alias visited before owner {owner}"
                    )),
                );
            };
            // The rebound view spans the typed extent of the new owner.
            let size = alias_extent.get(role).copied().unwrap_or(0);
            *slot = GpuTensor {
                // SAFETY: re-derived from the registered new VMM owner of `owner`.
                buf: unsafe { DeviceBuffer::from_raw(ptr as *mut c_void, size) },
                shape: slot.shape.clone(),
                dtype: slot.dtype,
            };
            let left = pending_aliases.get_mut(owner).map(|c| {
                *c -= 1;
                *c
            });
            if left == Some(0) {
                if let Some(mut old) = old_hold.remove(owner) {
                    if let Err(e) = release_old(gpu, owner, &mut old, ctx) {
                        return fail(&mut stage_error, e);
                    }
                }
            }
            return Ok(());
        }
        if !owner_ids.contains(role) {
            return fail(
                &mut stage_error,
                SharedWeightError::invalid(format!("{role}: traversal role missing from census")),
            );
        }
        let typed_bytes = owner_typed.get(role).copied().unwrap_or(0);
        if typed_bytes == 0 || typed_bytes > slot.buf.size() {
            return fail(
                &mut stage_error,
                SharedWeightError::invalid(format!(
                    "{role}: typed {typed_bytes} B vs accessible source {} B",
                    slot.buf.size()
                )),
            );
        }
        // Copy only the typed prefix: a non-owning view bounded to
        // `typed_bytes` of the live source owner, which stays in the graph
        // (released once by `release_old`). The allocator slack beyond it is
        // never exported; the new owner's tail past the typed bytes is zeroed.
        let source_view = GpuTensor {
            // SAFETY: `slot` owns at least `typed_bytes` valid bytes at this
            // pointer and outlives the copy; a borrowed buffer never frees.
            buf: unsafe { DeviceBuffer::from_raw(slot.buf.as_ptr() as *mut c_void, typed_bytes) },
            shape: Vec::new(),
            dtype: slot.dtype,
        };
        let exp = match gpu.create_exportable_weight_copy(&source_view) {
            Ok(exp) => exp,
            Err(e) => {
                return fail(
                    &mut stage_error,
                    SharedWeightError::hip(format!("export copy of {role}"), e),
                );
            }
        };
        let ExportedWeightAllocation {
            owner,
            fd,
            physical_bytes,
            logical_bytes,
            granularity,
        } = exp;
        let (physical, logical, gran) = (
            physical_bytes as u64,
            logical_bytes as u64,
            granularity as u64,
        );
        if let Err(e) = check_extents(role, physical, logical, gran).and_then(|()| {
            if logical_bytes == typed_bytes
                && owner.buf.size() == logical_bytes
                && owner.buf.is_vmm_owner()
            {
                Ok(())
            } else {
                Err(SharedWeightError::invalid(format!(
                    "{role}: exported owner extent {} != typed {typed_bytes}",
                    owner.buf.size()
                )))
            }
        }) {
            ctx.stranded.push((format!("{role}.exported"), owner));
            return fail(&mut stage_error, e);
        }
        let typed = GpuTensor {
            buf: owner.buf,
            shape: slot.shape.clone(),
            dtype: slot.dtype,
        };
        new_ptr.insert(role.to_owned(), typed.buf.as_ptr() as usize);
        let mut old = std::mem::replace(slot, typed);
        ctx.promoted.push(Promoted {
            owner_id: role.to_owned(),
            fd,
            physical,
            logical,
            granularity: gran,
        });
        if pending_aliases.get(role).copied().unwrap_or(0) == 0 {
            if let Err(e) = release_old(gpu, role, &mut old, ctx) {
                return fail(&mut stage_error, e);
            }
        } else {
            old_hold.insert(role.to_owned(), old);
        }
        Ok(())
    });
    // Sources still waiting for an alias rebind were never released: keep
    // them for cleanup so nothing is dropped live.
    for (owner, old) in old_hold {
        ctx.stranded.push((format!("{owner}.source"), old));
    }
    match traversal {
        Ok(()) => stage_error.map_or(Ok(()), Err),
        Err(e) => {
            Err(stage_error.unwrap_or_else(|| SharedWeightError::hip("typed export traversal", e)))
        }
    }
}

/// Release a promoted-away source. A synchronization failure leaves `old`
/// intact (kept for cleanup); a free failure is terminal (descriptor already
/// emptied by the primitive) and is reported without retry.
fn release_old(
    gpu: &mut Gpu,
    owner: &str,
    old: &mut GpuTensor,
    ctx: &mut ExportCtx,
) -> SwResult<()> {
    match gpu.release_weight_source_immediate(old) {
        Ok(()) => Ok(()),
        Err(e) => {
            if !old.buf.is_borrowed() {
                let preserved = std::mem::replace(old, empty_tensor());
                ctx.stranded.push((format!("{owner}.source"), preserved));
            }
            Err(SharedWeightError::hip(
                format!("release of promoted source {owner}"),
                e,
            ))
        }
    }
}

fn build_manifest(
    gpu: &Gpu,
    config: &Qwen35Config,
    prepared: &Prepared,
    promoted: &[Promoted],
) -> SwResult<WeightManifest> {
    let views = views_from_seal(&prepared.seal)?;
    let blob = encode_dense_seal(&prepared.flags, &prepared.seal)?;
    let allocations = promoted
        .iter()
        .map(|p| SharedAllocation {
            owner_id: p.owner_id.clone(),
            physical_bytes: p.physical,
            logical_bytes: p.logical,
            granularity: p.granularity,
            device_arch: gpu.arch.clone(),
            representation_tag: DENSE_REPRESENTATION_TAG.to_owned(),
        })
        .collect();
    Ok(WeightManifest {
        arch: DENSE_MANIFEST_ARCH.to_owned(),
        config: manifest_config(config, &prepared.flags),
        dispatch_metadata: blob,
        allocations,
        views,
    })
}

fn export_finish(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    prepared: &Prepared,
    ctx: &mut ExportCtx,
) -> SwResult<(WeightManifest, DenseCensus)> {
    // The promoted graph must still match the seal and own only VMM owners.
    let recs = collect_records(weights, &typed_map_of_seal(&prepared.seal))?;
    let census = census_from_records(&recs, true)?;
    check_census_matches_seal(&recs, &census, &prepared.seal)?;
    if census.owners.len() != prepared.census.owners.len()
        || census.aliases != prepared.census.aliases
    {
        return Err(SharedWeightError::invalid(
            "owner/alias topology changed during promotion",
        ));
    }
    if ctx.promoted.len() != census.owners.len() {
        return Err(SharedWeightError::invalid(format!(
            "{} promoted allocations for {} owners",
            ctx.promoted.len(),
            census.owners.len()
        )));
    }
    ctx.promoted
        .sort_by(|a, b| a.owner_id.as_bytes().cmp(b.owner_id.as_bytes()));
    let mut manifest = build_manifest(gpu, config, prepared, &ctx.promoted)?;
    manifest.canonicalize();
    manifest.validate().map_err(manifest_err("validate"))?;
    for (alloc, p) in manifest.allocations.iter().zip(&ctx.promoted) {
        if alloc.owner_id != p.owner_id {
            return Err(SharedWeightError::invalid(
                "canonical allocation order != FD order",
            ));
        }
    }
    Ok((manifest, census))
}

/// Promote a loaded dense graph into shared VMM owners and describe it.
///
/// On any error every GPU resource is released immediately (graph owners,
/// preserved sources, unattached new owners) and FDs are closed; nothing is
/// published. Cleanup failures ride in [`SharedWeightError::Cleanup`].
pub fn export_dense_weights(
    gpu: &mut Gpu,
    weights: Qwen35Weights,
    config: &Qwen35Config,
    recipe: &LayoutRecipe,
) -> Result<ExportedDenseWeights, SharedWeightError> {
    let mut weights = weights;
    let mut ctx = ExportCtx::default();
    let outcome = (|| -> SwResult<(Prepared, WeightManifest, DenseCensus)> {
        let prepared = export_prepare(gpu, &weights, config, recipe)?;
        export_promote(gpu, &mut weights, &prepared.census, &mut ctx)?;
        let (manifest, census) = export_finish(gpu, &weights, config, &prepared, &mut ctx)?;
        Ok((prepared, manifest, census))
    })();
    match outcome {
        Ok((prepared, manifest, census)) => {
            let physical_bytes = ctx.promoted.iter().map(|p| p.physical).sum();
            let allocation_fds = ctx.promoted.into_iter().map(|p| p.fd).collect();
            debug_assert!(ctx.stranded.is_empty());
            Ok(ExportedDenseWeights {
                weights,
                manifest,
                allocation_fds,
                census,
                flags: prepared.flags,
                physical_bytes,
            })
        }
        Err(error) => {
            let mut failures = Vec::new();
            for (role, t) in std::mem::take(&mut ctx.stranded) {
                release_tensor(gpu, &role, t, &mut failures);
            }
            release_all_weights(gpu, weights, &mut failures);
            drop(ctx); // closes every exported FD
            Err(error.with_cleanup(failures))
        }
    }
}

// ─── Import ─────────────────────────────────────────────────────────────

/// Everything validated about a lease before any FD is mapped.
struct ImportPlan {
    meta: DenseSealMeta,
    /// (owner id, descriptor, extents) in FD order.
    owners: Vec<(String, GpuTensorDescriptor, ImportedWeightAllocationDesc)>,
}

fn to_usize(v: u64, what: &str) -> SwResult<usize> {
    usize::try_from(v).map_err(|_| SharedWeightError::invalid(format!("{what} {v} exceeds usize")))
}

/// Pure plan: manifest/config/seal/geometry/sidecar/LUT/alias/extent checks.
/// Requires `gpu.mq4v2_symmetric` already set to the sealed value so the
/// recipe predicates evaluate under the host's dispatch facts.
fn plan_import(
    gpu: &Gpu,
    config: &Qwen35Config,
    manifest: &WeightManifest,
    fd_count: usize,
    meta: DenseSealMeta,
) -> SwResult<ImportPlan> {
    if manifest.arch != DENSE_MANIFEST_ARCH {
        return Err(SharedWeightError::invalid(format!(
            "manifest arch {:?}",
            manifest.arch
        )));
    }
    if manifest.config != manifest_config(config, &meta.flags) {
        return Err(SharedWeightError::invalid(
            "manifest config differs from local Qwen35Config/flags",
        ));
    }
    if encode_dense_seal(&meta.flags, &meta.seal)? != manifest.dispatch_metadata {
        return Err(SharedWeightError::invalid(
            "dispatch metadata is not the canonical encoding of its seal",
        ));
    }
    if gpu.arch != SUPPORTED_DEVICE_ARCH {
        return Err(SharedWeightError::unsupported(format!(
            "device arch {}",
            gpu.arch
        )));
    }
    let local = resolve_dense_flags(gpu, config)?;
    if local != meta.flags {
        return Err(SharedWeightError::invalid(format!(
            "resolved recipe flags differ: host {:?} local {local:?}",
            meta.flags
        )));
    }
    validate_dense_seal(&meta.seal, &meta.flags, config)?;

    let mut expected_views = views_from_seal(&meta.seal)?;
    let mut actual_views = manifest.views.clone();
    expected_views.sort_by(|a, b| a.role.cmp(&b.role));
    actual_views.sort_by(|a, b| a.role.cmp(&b.role));
    if expected_views != actual_views {
        return Err(SharedWeightError::invalid(
            "manifest views (roles/extents/shapes/strides/LUT & sidecar metadata) differ from the sealed typed graph",
        ));
    }

    let owners = owners_from_seal(&meta.seal);
    if manifest.allocations.len() != owners.len() || fd_count != owners.len() {
        return Err(SharedWeightError::invalid(format!(
            "{} allocations / {} FDs for {} unique owners",
            manifest.allocations.len(),
            fd_count,
            owners.len()
        )));
    }
    let descs: HashMap<String, GpuTensorDescriptor> = seal_entries(&meta.seal)
        .into_iter()
        .filter(|e| e.alias_of.is_none())
        .map(|e| (e.role, e.desc.clone()))
        .collect();
    let mut plan = Vec::with_capacity(owners.len());
    for (alloc, (owner_id, bytes)) in manifest.allocations.iter().zip(&owners) {
        if &alloc.owner_id != owner_id {
            return Err(SharedWeightError::invalid(format!(
                "allocation order/owner {:?} != sealed owner {owner_id:?}",
                alloc.owner_id
            )));
        }
        if alloc.device_arch != gpu.arch || alloc.representation_tag != DENSE_REPRESENTATION_TAG {
            return Err(SharedWeightError::invalid(format!(
                "{owner_id}: allocation arch/representation {:?}/{:?}",
                alloc.device_arch, alloc.representation_tag
            )));
        }
        if alloc.logical_bytes != *bytes as u64 {
            return Err(SharedWeightError::invalid(format!(
                "{owner_id}: logical {} != sealed {bytes}",
                alloc.logical_bytes
            )));
        }
        check_extents(
            owner_id,
            alloc.physical_bytes,
            alloc.logical_bytes,
            alloc.granularity,
        )?;
        let desc = ImportedWeightAllocationDesc {
            physical_bytes: to_usize(alloc.physical_bytes, "physical_bytes")?,
            logical_bytes: to_usize(alloc.logical_bytes, "logical_bytes")?,
            granularity: to_usize(alloc.granularity, "granularity")?,
        };
        let tensor = descs
            .get(owner_id)
            .ok_or_else(|| SharedWeightError::invalid("owner missing from seal"))?
            .clone();
        plan.push((owner_id.clone(), tensor, desc));
    }
    Ok(ImportPlan { meta, owners: plan })
}

/// Bytes a successful import maps (sum of physical bytes) — for `ImportAck`.
pub fn dense_imported_physical_bytes(manifest: &WeightManifest) -> u64 {
    manifest.allocations.iter().map(|a| a.physical_bytes).sum()
}

fn lease_fd(lease: &WeightHostLease, i: usize) -> SwResult<BorrowedFd<'_>> {
    lease
        .allocation_fd(i)
        .ok_or_else(|| SharedWeightError::invalid(format!("lease has no FD {i}")))
}

/// Import a published dense set as typed `Qwen35Weights`.
///
/// Order: (1) decode + validate everything pure — config, flags, seal geometry,
/// sidecars, LUTs, views, alias topology, allocation extents, FD count; (2)
/// restore `gpu.mq4v2_symmetric` from the sealed source flag; (3) map one
/// allocation per unique owner; (4) hydrate via `from_dense_seal`; (5) verify
/// the hydrated graph. Any failure after (1) unmaps (frees) every imported
/// owner exactly once and restores `gpu.mq4v2_symmetric`. Does not acknowledge
/// the lease: the caller sends `ack_import(dense_imported_physical_bytes(..))`
/// after its own setup succeeds.
pub fn import_dense_weights(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    lease: &WeightHostLease,
) -> Result<Qwen35Weights, SharedWeightError> {
    let previous = gpu.mq4v2_symmetric;
    let result = import_inner(gpu, config, lease);
    if result.is_err() {
        gpu.mq4v2_symmetric = previous;
    }
    result
}

fn import_inner(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    lease: &WeightHostLease,
) -> SwResult<Qwen35Weights> {
    check_dense_config(config)?;
    let manifest = lease.manifest();
    manifest.validate().map_err(manifest_err("validate"))?;
    let meta = decode_dense_seal(&manifest.dispatch_metadata)?;
    // Same stamp model_load applies before a load; predicates read it.
    gpu.mq4v2_symmetric = meta.flags.mq4v2_symmetric;
    let plan = plan_import(gpu, config, manifest, lease.allocation_fds().len(), meta)?;

    // ── map: one allocation per unique owner ──
    let mut mapped: HashMap<String, GpuTensor> = HashMap::with_capacity(plan.owners.len());
    let mut failures = Vec::new();
    let mut map_error: Option<SharedWeightError> = None;
    for (i, (owner_id, desc, alloc)) in plan.owners.iter().enumerate() {
        let step = lease_fd(lease, i).and_then(|fd| {
            gpu.import_weight_allocation(fd, alloc)
                .map_err(|e| SharedWeightError::hip(format!("import of {owner_id}"), e))
        });
        match step {
            Ok(raw) => {
                if raw.buf.size() != desc.byte_len || !raw.buf.is_vmm_owner() {
                    let e = SharedWeightError::invalid(format!(
                        "{owner_id}: imported {} B (vmm owner: {}), sealed {} B",
                        raw.buf.size(),
                        raw.buf.is_vmm_owner(),
                        desc.byte_len
                    ));
                    release_tensor(gpu, owner_id, raw, &mut failures);
                    map_error = Some(e);
                    break;
                }
                mapped.insert(
                    owner_id.clone(),
                    GpuTensor {
                        buf: raw.buf,
                        shape: desc.shape.clone(),
                        dtype: desc.dtype,
                    },
                );
            }
            Err(e) => {
                map_error = Some(e);
                break;
            }
        }
    }
    if let Some(e) = map_error {
        for (role, t) in mapped {
            release_tensor(gpu, &role, t, &mut failures);
        }
        return Err(e.with_cleanup(failures));
    }

    // ── hydrate ──
    let hydrated = Qwen35Weights::from_dense_seal(&plan.meta.seal, &mut |role, _desc| {
        mapped
            .remove(role)
            .ok_or_else(|| HipError::new(0, &format!("no imported owner for role {role}")))
    });
    let weights = match hydrated {
        Ok(w) => w,
        Err(err) => {
            let mut failures = Vec::new();
            for (role, t) in err.acquired {
                release_tensor(gpu, &role, t, &mut failures);
            }
            for (role, t) in mapped {
                release_tensor(gpu, &role, t, &mut failures);
            }
            return Err(SharedWeightError::hip("typed hydration", err.error).with_cleanup(failures));
        }
    };

    // ── verify: every owner consumed once, graph == seal ──
    let verify = (|| -> SwResult<()> {
        if !mapped.is_empty() {
            return Err(SharedWeightError::invalid(format!(
                "{} imported owner(s) not consumed by hydration",
                mapped.len()
            )));
        }
        let recs = collect_records(&weights, &typed_map_of_seal(&plan.meta.seal))?;
        let census = census_from_records(&recs, true)?;
        check_census_matches_seal(&recs, &census, &plan.meta.seal)?;
        if census.owners.len() != plan.owners.len() {
            return Err(SharedWeightError::invalid(
                "hydrated owner count != allocation count",
            ));
        }
        Ok(())
    })();
    match verify {
        Ok(()) => Ok(weights),
        Err(e) => {
            let mut failures = Vec::new();
            for (role, t) in mapped {
                release_tensor(gpu, &role, t, &mut failures);
            }
            release_all_weights(gpu, weights, &mut failures);
            Err(e.with_cleanup(failures))
        }
    }
}

// ─── Tests (pure: no HIP) ───────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn config(layer_types: Vec<LayerType>) -> Qwen35Config {
        Qwen35Config {
            dim: 256,
            n_layers: layer_types.len(),
            vocab_size: 32,
            norm_eps: 1e-6,
            eos_token: 7,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 64,
            rope_theta: 1.0e7,
            partial_rotary_factor: 0.25,
            is_vl_text: false,
            mrope_interleaved: false,
            mrope_section: [0, 0, 0],
            linear_num_key_heads: 1,
            linear_num_value_heads: 2,
            linear_key_head_dim: 64,
            linear_value_head_dim: 64,
            conv_kernel_dim: 4,
            hidden_dim: 512,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            has_shared_expert: false,
            norm_topk_prob: false,
            layer_types,
            paged_experts: false,
            vram_budget_bytes: u64::MAX,
            reap_keep: None,
        }
    }

    fn flags() -> DenseRecipeFlags {
        DenseRecipeFlags {
            mq4v2_symmetric: true,
            a4_rms_fdiv_planes: true,
            gdn_fold_gfx12: true,
            gdn_fold_gfx11: false,
            lm_head_f32: false,
        }
    }

    fn f32d(n: usize) -> GpuTensorDescriptor {
        GpuTensorDescriptor {
            shape: vec![n],
            dtype: DType::F32,
            byte_len: n * 4,
        }
    }

    fn wd(
        dtype: DType,
        m: usize,
        k: usize,
        extra_rows: usize,
        awq: bool,
        planes: bool,
    ) -> WeightTensorDescriptor {
        let rb = dtype.row_bytes(k).expect("row bytes");
        let bytes = (m + extra_rows) * rb;
        WeightTensorDescriptor {
            gpu_dtype: dtype,
            m,
            k,
            row_stride: 0,
            buf: GpuTensorDescriptor {
                shape: vec![bytes],
                dtype: DType::Raw,
                byte_len: bytes,
            },
            awq_scale: awq.then(|| GpuTensorDescriptor {
                shape: if planes { vec![k, 3] } else { vec![k * 4] },
                dtype: DType::Raw,
                byte_len: if planes { k * 12 } else { k * 4 },
            }),
            paro: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        }
    }

    fn seal(c: &Qwen35Config, tied: bool) -> Qwen35RankSeal {
        let (dims, _) = delta_dims(c).unwrap();
        let d = DType::MQ4G256V2;
        let (dim, hid) = (c.dim, c.hidden_dim);
        let layers = c
            .layer_types
            .iter()
            .map(|t| match t {
                LayerType::LinearAttention => Qwen35LayerSeal::DeltaNet {
                    attn_norm: f32d(dim),
                    wqkv: wd(d, dims[0], dim, 0, true, true),
                    wz: wd(d, dims[1], dim, 128, true, false),
                    w_alpha: wd(d, dims[2], dim, 0, true, false),
                    w_beta: wd(d, dims[3], dim, 0, true, false),
                    a_log: f32d(dims[2]),
                    dt_bias: f32d(dims[2]),
                    conv_weight: f32d(dims[0] * c.conv_kernel_dim),
                    norm_weight: f32d(c.linear_value_head_dim),
                    wo: wd(d, dim, dims[1], 0, true, false),
                    ffn_norm: f32d(dim),
                    w_gate: wd(d, hid, dim, 0, true, true),
                    w_up: wd(d, hid, dim, 0, true, false),
                    w_down: wd(d, dim, hid, 0, true, false),
                },
                LayerType::FullAttention => Qwen35LayerSeal::FullAttn {
                    attn_norm: f32d(dim),
                    wq: wd(d, c.n_heads * c.head_dim * 2, dim, 0, true, true),
                    wk: wd(d, c.n_kv_heads * c.head_dim, dim, 0, true, false),
                    wv: wd(d, c.n_kv_heads * c.head_dim, dim, 0, true, false),
                    wo: wd(d, dim, c.n_heads * c.head_dim, 0, true, false),
                    q_norm: f32d(c.head_dim),
                    k_norm: f32d(c.head_dim),
                    ffn_norm: f32d(dim),
                    w_gate: wd(d, hid, dim, 0, true, true),
                    w_up: wd(d, hid, dim, 0, true, false),
                    w_down: wd(d, dim, hid, 0, true, false),
                },
            })
            .collect::<Vec<_>>();
        let n = layers.len();
        let token_embd = GpuTensorDescriptor {
            shape: vec![c.vocab_size * dim],
            dtype: DType::F32,
            byte_len: c.vocab_size * dim * 4,
        };
        let mut output = wd(DType::F32, c.vocab_size, dim, 0, false, false);
        if tied {
            output.buf = token_embd.clone();
        }
        Qwen35RankSeal {
            token_embd,
            embd_format: EmbeddingFormat::F32,
            output_norm: f32d(dim),
            output,
            moe_has_mq6: false,
            has_pager: false,
            lm_head_aliases_embd: tied,
            layer_seals: layers,
            global_expert_dtypes: vec![Vec::new(); n],
            local_expert_descriptors: vec![Vec::new(); n],
        }
    }

    fn cfg() -> Qwen35Config {
        config(vec![LayerType::LinearAttention, LayerType::FullAttention])
    }

    #[test]
    fn dtype_tags_round_trip_and_are_unique() {
        let mut seen = HashSet::new();
        for tag in 0..=255u8 {
            if let Some(d) = dtype_from_tag(tag) {
                assert_eq!(dtype_tag(d), tag);
                assert!(seen.insert(dtype_name(d)));
            }
        }
    }

    #[test]
    fn codec_round_trips_deterministically() {
        for tied in [false, true] {
            let s = seal(&cfg(), tied);
            let a = encode_dense_seal(&flags(), &s).unwrap();
            let b = encode_dense_seal(&flags(), &s).unwrap();
            assert_eq!(a, b);
            let m = decode_dense_seal(&a).unwrap();
            assert_eq!(m.flags, flags());
            assert_eq!(m.seal, s);
            assert_eq!(encode_dense_seal(&m.flags, &m.seal).unwrap(), a);
        }
    }

    #[test]
    fn codec_rejects_truncation_trailing_and_bad_tags() {
        let blob = encode_dense_seal(&flags(), &seal(&cfg(), false)).unwrap();
        for cut in 0..blob.len() {
            assert!(
                decode_dense_seal(&blob[..cut]).is_err(),
                "prefix {cut} decoded"
            );
        }
        let mut extra = blob.clone();
        extra.push(0);
        assert!(decode_dense_seal(&extra).is_err());
        let mut bad_magic = blob.clone();
        bad_magic[0] ^= 1;
        assert!(decode_dense_seal(&bad_magic).is_err());
        let mut bad_version = blob.clone();
        bad_version[8] = 9;
        assert!(decode_dense_seal(&bad_version).is_err());
        // embedding tag byte follows magic(8)+version(4)+5 flags.
        let mut bad_embd = blob.clone();
        bad_embd[17] = 0xEE;
        assert!(decode_dense_seal(&bad_embd).is_err());
        let mut bad_bool = blob.clone();
        bad_bool[12] = 2;
        assert!(decode_dense_seal(&bad_bool).is_err());
    }

    #[test]
    fn codec_bounds_huge_counts_without_allocating() {
        let mut blob = Vec::new();
        blob.extend_from_slice(SEAL_MAGIC);
        blob.extend_from_slice(&SEAL_VERSION.to_le_bytes());
        blob.extend_from_slice(&[1, 1, 1, 0, 0, 1, 0]);
        blob.extend_from_slice(&u64::MAX.to_le_bytes()); // token_embd rank
        assert!(decode_dense_seal(&blob).is_err());
        assert!(decode_dense_seal(&vec![0u8; MAX_SEAL_BYTES + 1]).is_err());
    }

    #[test]
    fn codec_rejects_paro_and_moe() {
        let mut s = seal(&cfg(), false);
        s.output.paro = Some(crate::qwen35::weights::ParoDescriptor {
            krot: 1,
            group_size: 128,
            is_alias: false,
            pairs: f32d(1),
            theta: f32d(1),
            channel_scales: f32d(1),
        });
        assert!(encode_dense_seal(&flags(), &s).is_err());
        let mut s = seal(&cfg(), false);
        s.has_pager = true;
        assert!(encode_dense_seal(&flags(), &s).is_err());
        let blob = encode_dense_seal(&flags(), &seal(&cfg(), false)).unwrap();
        assert!(decode_dense_seal(&blob).is_ok());
    }

    #[test]
    fn lloyd_luts_must_be_complete_and_exclusive() {
        let c = cfg();
        let mut s = seal(&c, false);
        if let Qwen35LayerSeal::DeltaNet { wqkv, wz, .. } = &mut s.layer_seals[0] {
            wqkv.gpu_dtype = DType::MQ4G256V2Lloyd;
            wqkv.awq_scale = None;
            // The loader folds Z only when all four projections are uniform V2.
            let z_bytes = wz.m * wz.gpu_dtype.row_bytes(wz.k).unwrap();
            wz.buf.byte_len = z_bytes;
            wz.buf.shape = vec![z_bytes];
        }
        assert!(matches!(
            validate_dense_seal(&s, &flags(), &c),
            Err(SharedWeightError::Unsupported(_)) | Err(SharedWeightError::Hip { .. })
        ));
        if let Qwen35LayerSeal::DeltaNet { wqkv, .. } = &mut s.layer_seals[0] {
            wqkv.lloyd_lut_e4m3 = Some([1, 2, 3, 4]);
            wqkv.lloyd_lut_f16 = Some([1; 8]);
        }
        assert!(validate_dense_seal(&s, &flags(), &c).is_err()); // partial
        if let Qwen35LayerSeal::DeltaNet { wqkv, .. } = &mut s.layer_seals[0] {
            wqkv.lloyd_lut_c16 = Some([9, 9, 9, 9]);
        }
        validate_dense_seal(&s, &flags(), &c).expect("complete LUTs");
        // LUT payload is identity: change one word → different view metadata + blob.
        let v1 = views_from_seal(&s).unwrap();
        let b1 = encode_dense_seal(&flags(), &s).unwrap();
        if let Qwen35LayerSeal::DeltaNet { wqkv, .. } = &mut s.layer_seals[0] {
            wqkv.lloyd_lut_f16 = Some([1, 1, 1, 1, 1, 1, 1, 2]);
        }
        assert_ne!(v1, views_from_seal(&s).unwrap());
        assert_ne!(b1, encode_dense_seal(&flags(), &s).unwrap());
        // A uniform dtype must not carry LUTs.
        let mut u = seal(&c, false);
        if let Qwen35LayerSeal::DeltaNet { wqkv, .. } = &mut u.layer_seals[0] {
            wqkv.lloyd_lut_c16 = Some([0; 4]);
        }
        assert!(validate_dense_seal(&u, &flags(), &c).is_err());
    }

    #[test]
    fn geometry_and_sidecar_mismatches_reject() {
        let c = cfg();
        validate_dense_seal(&seal(&c, false), &flags(), &c).expect("baseline");
        validate_dense_seal(&seal(&c, true), &flags(), &c).expect("tied baseline");

        // Wrong layer topology.
        let swapped = config(vec![LayerType::FullAttention, LayerType::LinearAttention]);
        assert!(validate_dense_seal(&seal(&c, false), &flags(), &swapped).is_err());
        // Fold bytes only legal when the fold flag was resolved.
        let mut no_fold = flags();
        no_fold.gdn_fold_gfx12 = false;
        assert!(validate_dense_seal(&seal(&c, false), &no_fold, &c).is_err());
        // Reciprocal planes only legal when planes were resolved.
        let mut no_planes = flags();
        no_planes.a4_rms_fdiv_planes = false;
        assert!(validate_dense_seal(&seal(&c, false), &no_planes, &c).is_err());
        // Wrong norm width.
        let mut s = seal(&c, false);
        s.output_norm = f32d(c.dim + 1);
        assert!(validate_dense_seal(&s, &flags(), &c).is_err());
        // Wrong packed bytes.
        let mut s = seal(&c, false);
        if let Qwen35LayerSeal::FullAttn { wk, .. } = &mut s.layer_seals[1] {
            wk.buf.byte_len += 136;
        }
        assert!(validate_dense_seal(&s, &flags(), &c).is_err());
        // Tied head must match the embedding extent.
        let mut s = seal(&c, true);
        s.output.buf.byte_len += 4;
        assert!(validate_dense_seal(&s, &flags(), &c).is_err());
    }

    /// Source seal as captured from a pool-allocated graph: every accessible
    /// `byte_len` exceeds its typed length.
    fn padded_source(typed: &Qwen35RankSeal) -> Qwen35RankSeal {
        let mut s = typed.clone();
        for_each_desc_mut(&mut s, &mut |_, d| d.byte_len += 256);
        s
    }

    #[test]
    fn f32_semantic_extent_ignores_allocator_slack() {
        let slack = GpuTensorDescriptor {
            shape: vec![48],
            dtype: DType::F32,
            byte_len: 256,
        };
        assert_eq!(generic_typed_bytes("a_log", &slack).unwrap(), 192);
        // A sealed (typed) descriptor never carries slack.
        assert!(check_plain("a_log", &slack, 48).is_err());
        check_plain(
            "a_log",
            &GpuTensorDescriptor {
                byte_len: 192,
                ..slack.clone()
            },
            48,
        )
        .unwrap();
    }

    #[test]
    fn typed_lengths_cover_fold_planes_and_alias() {
        let c = cfg();
        let (dims, _) = delta_dims(&c).unwrap();
        let rb = DType::MQ4G256V2.row_bytes(c.dim).unwrap();
        let lens: HashMap<String, usize> = typed_lengths(&seal(&c, false), &flags())
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(lens["layers.0000.wz.buf"], (dims[1] + 128) * rb);
        assert_eq!(lens["layers.0000.wqkv.buf"], dims[0] * rb);
        assert_eq!(lens["layers.0000.wqkv.awq_scale"], c.dim * 12);
        assert_eq!(lens["layers.0000.wz.awq_scale"], c.dim * 4);
        assert_eq!(lens["layers.0000.a_log"], dims[2] * 4);
        let tied: HashMap<String, usize> = typed_lengths(&seal(&c, true), &flags())
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(tied["output.buf"], tied["token_embd"]);
    }

    #[test]
    fn normalize_accepts_allocator_padding_and_seals_typed_bytes() {
        let c = cfg();
        for tied in [false, true] {
            let typed = seal(&c, tied);
            let src = padded_source(&typed);
            assert!(validate_dense_seal(&src, &flags(), &c).is_err());
            let n = normalize_dense_seal(&src, &flags(), &c).expect("padded source normalizes");
            assert_eq!(n, typed);
        }
    }

    #[test]
    fn normalize_rejects_short_sources_and_wrong_metadata() {
        let c = cfg();
        let typed = seal(&c, false);
        // Every role: one byte short of its typed length is rejected.
        for (role, n) in typed_lengths(&typed, &flags()).unwrap() {
            let mut s = padded_source(&typed);
            for_each_desc_mut(&mut s, &mut |r, d| {
                if r == role {
                    d.byte_len = n - 1;
                }
            });
            assert!(normalize_dense_seal(&s, &flags(), &c).is_err(), "{role}");
        }
        // Folded Z whose shape is not its folded byte count.
        let mut s = padded_source(&typed);
        if let Qwen35LayerSeal::DeltaNet { wz, .. } = &mut s.layer_seals[0] {
            wz.buf.shape = vec![wz.buf.shape[0] - 136];
        }
        assert!(normalize_dense_seal(&s, &flags(), &c).is_err());
        // Fold rows present but the fold flag was not resolved.
        let mut no_fold = flags();
        no_fold.gdn_fold_gfx12 = false;
        assert!(normalize_dense_seal(&padded_source(&typed), &no_fold, &c).is_err());
        // Reciprocal planes present but the planes flag was not resolved.
        let mut no_planes = flags();
        no_planes.a4_rms_fdiv_planes = false;
        assert!(normalize_dense_seal(&padded_source(&typed), &no_planes, &c).is_err());
        // Wrong geometry.
        let mut s = padded_source(&typed);
        if let Qwen35LayerSeal::FullAttn { wk, .. } = &mut s.layer_seals[1] {
            wk.m += 1;
        }
        assert!(normalize_dense_seal(&s, &flags(), &c).is_err());
    }

    #[test]
    fn config_identity_is_exact_bits_and_topology() {
        let a = cfg();
        let base = dense_config_fields(&a);
        assert_eq!(base, dense_config_fields(&a.clone()));
        let mut b = a.clone();
        b.norm_eps = f32::from_bits(a.norm_eps.to_bits() + 1);
        assert_ne!(base, dense_config_fields(&b));
        let mut b = a.clone();
        b.rope_theta = f32::from_bits(a.rope_theta.to_bits() ^ 1);
        assert_ne!(base, dense_config_fields(&b));
        let mut b = a.clone();
        b.layer_types.swap(0, 1);
        assert_ne!(base, dense_config_fields(&b));
        let mut b = a.clone();
        b.mrope_interleaved = true;
        assert_ne!(base, dense_config_fields(&b));
        let mut b = a.clone();
        b.num_experts = 1;
        assert!(check_dense_config(&b).is_err());
        b = a.clone();
        b.paged_experts = true;
        assert!(check_dense_config(&b).is_err());
        check_dense_config(&a).expect("dense");
    }

    #[test]
    fn views_describe_aliases_and_owners() {
        let c = cfg();
        let tied = seal(&c, true);
        let views = views_from_seal(&tied).unwrap();
        let out = views.iter().find(|v| v.role == "output.buf").unwrap();
        assert_eq!(out.alias_of.as_deref(), Some("token_embd"));
        assert_eq!(out.owner_id, "token_embd");
        assert_eq!(out.byte_offset, 0);
        let owners = owners_from_seal(&tied);
        assert!(owners.iter().all(|(id, _)| id != "output.buf"));
        let roles: HashSet<_> = views.iter().map(|v| v.role.as_str()).collect();
        assert_eq!(roles.len(), views.len());
        assert!(roles.contains("layers.0000.wz.buf") && roles.contains("layers.0001.wq.awq_scale"));
        // Sorted bytewise; layer zero-padding keeps numeric order.
        let sorted: Vec<_> = views.iter().map(|v| v.role.clone()).collect();
        let mut check = sorted.clone();
        check.sort();
        assert_eq!(sorted, check);
        // Untied output is its own owner.
        let untied = owners_from_seal(&seal(&c, false));
        assert!(untied.iter().any(|(id, _)| id == "output.buf"));
    }

    #[test]
    fn census_records_slack_separately_from_typed_bytes() {
        let mut slack = rec("token_embd", 0, 0x1000, 0x140, false);
        slack.typed = 0x100;
        let c = census_from_records(&[slack], false).unwrap();
        assert_eq!(
            (c.logical_bytes, c.source_bytes, c.source_slack_bytes),
            (0x100, 0x140, 0x40)
        );
        // An accessible allocation shorter than its typed length is rejected.
        let mut short = rec("token_embd", 0, 0x1000, 0x100, false);
        short.typed = 0x101;
        assert!(census_from_records(&[short], false).is_err());
    }

    fn rec(role: &str, order: usize, ptr: usize, size: usize, borrowed: bool) -> Rec {
        Rec {
            role: role.to_owned(),
            order,
            ptr,
            size,
            borrowed,
            typed: size,
            host: false,
            vmm: !borrowed,
            shape: vec![size],
            dtype: DType::Raw,
        }
    }

    #[test]
    fn census_alias_ranges_and_owner_uniqueness() {
        let ok = vec![
            rec("token_embd", 0, 0x1000, 0x100, false),
            rec("output_norm", 1, 0x2000, 0x10, false),
            rec("output.buf", 2, 0x1000, 0x100, true),
        ];
        let c = census_from_records(&ok, true).unwrap();
        assert_eq!(c.owners.len(), 2);
        assert_eq!(c.aliases[0].owner_id, "token_embd");
        assert_eq!(
            (c.aliases[0].byte_offset, c.aliases[0].byte_extent),
            (0, 0x100)
        );
        assert_eq!(c.logical_bytes, 0x110);
        assert_eq!(c.largest_owner_bytes, 0x100);
        assert_eq!(c.sorted_owner_ids(), vec!["output_norm", "token_embd"]);

        let outside = vec![
            rec("a", 0, 0x1000, 0x100, false),
            rec("v", 1, 0x1080, 0x100, true),
        ];
        assert!(census_from_records(&outside, false).is_err());
        let orphan = vec![
            rec("a", 0, 0x1000, 0x100, false),
            rec("v", 1, 0x500, 0x10, true),
        ];
        assert!(census_from_records(&orphan, false).is_err());
        let before = vec![
            rec("v", 0, 0x1000, 0x10, true),
            rec("a", 1, 0x1000, 0x100, false),
        ];
        assert!(census_from_records(&before, false).is_err());
        let dup = vec![
            rec("a", 0, 0x1000, 0x100, false),
            rec("b", 1, 0x1000, 0x100, false),
        ];
        assert!(census_from_records(&dup, false).is_err());
        let overlap = vec![
            rec("a", 0, 0x1000, 0x100, false),
            rec("b", 1, 0x1080, 0x100, false),
        ];
        assert!(census_from_records(&overlap, false).is_err());
        let mut host = rec("h", 0, 0x1000, 0x100, false);
        host.host = true;
        assert!(census_from_records(&[host], false).is_err());
        let mut not_vmm = rec("n", 0, 0x1000, 0x100, false);
        not_vmm.vmm = false;
        assert!(census_from_records(&[not_vmm], true).is_err());
        // Offset alias is representable but cannot be hydrated.
        let off = vec![
            rec("a", 0, 0x1000, 0x100, false),
            rec("v", 1, 0x1040, 0x40, true),
        ];
        let c = census_from_records(&off, false).unwrap();
        assert_eq!(c.aliases[0].byte_offset, 0x40);
    }

    #[test]
    fn allocation_extents_follow_tail_and_granularity() {
        assert_eq!(expected_physical_bytes(1, 4096), Some(69632));
        assert_eq!(expected_physical_bytes(4096, 4096), Some(69632));
        assert_eq!(expected_physical_bytes(1, 0), None);
        check_extents("o", 69632, 4096, 4096).unwrap();
        assert!(check_extents("o", 4096, 4096, 4096).is_err()); // no tail
        assert!(check_extents("o", 69633, 4096, 4096).is_err()); // unaligned
        assert!(check_extents("o", 73728, 4096, 4096).is_err()); // slack granule
        assert!(check_extents("o", 69632, 0, 4096).is_err());
        assert!(check_extents("o", 69632, 4096, 0).is_err());
    }
}
