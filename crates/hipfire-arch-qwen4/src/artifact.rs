// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Checked Qwen4 HFQM artifact admission.
//!
//! This is the one source/index validator for Qwen4.  Loader, evaluator, and
//! any future carrier must consume [`admit_hfqm_artifact`] rather than parsing
//! the HFQM index or re-implementing quant/PLE geometry locally.  The receipt
//! is value-only: it owns validated declarations and a source seal, but never
//! allocates GPU memory or starts a reader worker.

use crate::config::{Qwen4Config, Qwen4PleConfig, ARCH_ID};
use crate::expert_hotset::HOTSET_ENV;
use crate::expert_slab_table::{build_expert_slab_table_with_keep, ExpertSlabTableError};
use crate::expert_slabs::{self, parse_expert_entry_name, ExpertSlabsError};
use crate::gpu_forward::{qwen4_forward_scratch_mode, Qwen4ForwardScratchMode};
use crate::ple::PleHashMetadata;
use crate::weights::{
    is_ple_shard_name_within, Qwen4BandScope, Qwen4Manifest, Qwen4Placement, Qwen4TierRecipe,
    WeightError, RECIPE_METADATA_KEY,
};
use hipfire_runtime::external_rows::RowEncoding;
use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use hipfire_runtime::model_source::{SourceFormat, SourceIdentity, SourceRangeDescriptor};
use hipfire_runtime::ple_hot_pack::PackOrder;
use hipfire_runtime::ple_hot_store::HotRowStore;
use hipfire_runtime::ple_q4::{lm4_codebook_bits, lm4_scale_grid_bits};
use hipfire_runtime::verified_cache::StatKey;
use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry, WeightResidency};
use hipfire_runtime::weight_store::external_row_stride;
use rdna_compute::{DType, Gpu};
use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

const QWEN4_FORMAT_VERSION: u32 = 1;
const QWEN4_QT_BF16: u8 = 16;
const QWEN4_QT_I64: u8 = 54;
const QWEN4_QT_MQ4G256V2: u8 = 44;
const QWEN4_QT_MQ4G128V2: u8 = 53;
/// MQ6G256V2 (qt=47): aligned-K 256 groups, 200 B/group (6-bit payload).
const QWEN4_QT_MQ6G256V2: u8 = 47;
/// Q8F16 (qt=3): f16 scale + 32 int8 per block, 34 bytes per 32 weights.  Not
/// FWHT-rotated.  The MoE fixed classes (embed, lm_head) ship at this tier.
const QWEN4_QT_Q8F16: u8 = 3;
/// MFP4G32E8SOA (qt=35): E8-lattice SoA rows, 4.25 bpw, 32-weight blocks with a
/// 16-byte-aligned padded scale block.
const QWEN4_QT_MFP4G32E8SOA: u8 = 35;
/// FP8E4M3B128 (qt=55): stacked routed-expert blocks of OCP e4m3 codes and
/// BF16 128x128 `s_inv` (stock-FP8 recipe); group_size 128.
const QWEN4_QT_FP8E4M3B128: u8 = 55;
/// FP8E4M3PT (qt=56): 16-byte per-tensor BF16 scale header then one e4m3 code
/// per value (stock-FP8 PLE rows); group_size 0.
const QWEN4_QT_FP8E4M3PT: u8 = 56;
/// One FWHT-256 segment: the unit the encoder rotates and the decode GEMV/GEMM
/// assert (`assert!(k % 256 == 0)`).  A payload that only satisfied the
/// 32-weight block size would pass every byte count here and then either panic
/// or read unrotated activations.
const QWEN4_MFP4G32E8SOA_K_ALIGNMENT: usize = 256;

/// PLE metadata is deliberately tiny.  Refuse an unexpectedly large record
/// before allocating a host read buffer, even if a corrupt index advertises a
/// matching shape with a huge byte extent.
const MAX_PLE_METADATA_BYTES: usize = 2048;

/// Zero-based layer PLE shard tensors and I64 records live under.
fn ple_tensor_prefix(layer: usize) -> String {
    format!("model.language_model.layers.{layer}.ple")
}

/// Name of PLE shard `shard` of the PLE block at `layer`.
fn ple_shard_tensor_name(layer: usize, shard: usize) -> String {
    format!(
        "{}.ple_embedding.ngram_embedding.shard_{shard}.weight",
        ple_tensor_prefix(layer)
    )
}

/// The three I64 PLE records of the PLE block at `layer`:
/// `[layer_multipliers, ngram_heads_offsets, ngram_heads_vocab_sizes]`.
fn ple_record_names(layer: usize) -> [String; 3] {
    let base = format!("{}.ple_embedding", ple_tensor_prefix(layer));
    [
        format!("{base}.layer_multipliers"),
        format!("{base}.ngram_heads_offsets"),
        format!("{base}.ngram_heads_vocab_sizes"),
    ]
}

/// Where an admitted artifact's PLE n-gram rows come from.
///
/// The rows are the only part of a Qwen4 model that can live outside the HFQM
/// file the loader opens: [`Self::Sidecar`] is a `<model>.ple-q4` /
/// `<model>.ple-q8` file attached with
/// `hipfire_runtime::ple_sidecar::resolve_and_attach_ple_rows`.
#[derive(Clone, Debug)]
pub enum Qwen4PleRows {
    /// The profile declares no PLE: there are no rows, no shard tensors, no
    /// I64 records and no sidecar.
    Absent,
    /// The shard tensors of the admitted HFQM file itself (the monolithic
    /// artifact, read exactly as before sidecars existed).
    Embedded,
    /// An attached, validated sidecar rows file.
    Sidecar {
        path: PathBuf,
        encoding: RowEncoding,
        rows: SidecarRows,
    },
}

/// How a sidecar's rows are served.
#[derive(Clone)]
pub enum SidecarRows {
    /// Row-order table: the 128 ordered shard ranges under one sealed
    /// single-file identity, read through the bounded row store like the
    /// embedded shards.
    Descriptors(Vec<SourceRangeDescriptor>),
    /// Rank-order table: the rows are stored hottest-first with an embedded
    /// rank, so the file is its own hot-row pack; the store serves rows
    /// directly and the row store is built on top of it.
    Table(Arc<HotRowStore>),
}

impl fmt::Debug for SidecarRows {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Descriptors(descriptors) => f
                .debug_tuple("Descriptors")
                .field(&descriptors.len())
                .finish(),
            Self::Table(_) => f.write_str("Table"),
        }
    }
}

impl Qwen4PleRows {
    /// The rank-order table store, when the rows come from one.
    pub fn rank_table(&self) -> Option<&Arc<HotRowStore>> {
        match self {
            Self::Sidecar {
                rows: SidecarRows::Table(store),
                ..
            } => Some(store),
            _ => None,
        }
    }
}

/// All declarations needed by the executable carrier and source evaluator.
///
/// The receipt contains no GPU handle and no mutable source reader.  The
/// `source_identity` seal is retained so an evaluator can bind any later
/// range reads to the exact artifact that was admitted.
#[derive(Clone, Debug)]
pub struct Qwen4HfqmArtifact {
    pub config: Qwen4Config,
    /// The weight manifest.  With a sidecar the PLE shard entries are
    /// removed ([`Qwen4Manifest::without_ple_shards`]): their rows are read
    /// from the sidecar, never fulfilled from the HFQM file.
    pub manifest: Qwen4Manifest,
    /// The validated PLE hash metadata; `None` exactly when the profile
    /// declares no PLE.
    pub ple: Option<PleHashMetadata>,
    pub placements: Vec<Qwen4Placement>,
    pub source_tensor_count: usize,
    pub source_identity: Arc<SourceIdentity>,
    /// The PLE rows source the bundle must be assembled with
    /// (`Qwen4Bundle::assemble_with_ple_rows`); [`Qwen4PleRows::Absent`] when
    /// the profile has no PLE.
    pub ple_rows: Qwen4PleRows,
}

impl Qwen4HfqmArtifact {
    /// Project this already-admitted inventory before rank-local fulfillment.
    /// Source sealing and codec admission are never repeated or weakened.
    pub fn for_band(
        mut self,
        scope: &Qwen4BandScope,
        device: usize,
    ) -> Result<Self, Qwen4ArtifactError> {
        self.placements = self
            .manifest
            .placements_for_scope(&self.config, scope, device)
            .map_err(|error| Qwen4ArtifactError::new(error.to_string()))?;
        self.manifest = self
            .manifest
            .with_band_scope(&self.config, scope)
            .map_err(|error| Qwen4ArtifactError::new(error.to_string()))?;
        if !scope.input {
            self.ple = None;
            self.ple_rows = Qwen4PleRows::Absent;
        }
        Ok(self)
    }
}

/// A checked artifact boundary failure.  Keeping the error owned and typed
/// lets loader/evaluator callers preserve one diagnostic convention without
/// exposing private parser helpers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Qwen4ArtifactError {
    /// Existing admission diagnostics retain their exact text.
    Message(String),
    ExpertSlabsMetadata(ExpertSlabsError),
    ExpertSlabTable(ExpertSlabTableError),
    ExpertSlabsMetadataRequired {
        name: String,
        quant_type: u8,
    },
    ExpertSlabQuantTypeUnavailable {
        name: String,
        quant_type: u8,
    },
    ExpertSlabKernelsUnavailable {
        w3: usize,
        w2: usize,
        e8p: usize,
        mq4: usize,
    },
}

impl Qwen4ArtifactError {
    fn new(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

impl fmt::Display for Qwen4ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(message) => f.write_str(message),
            Self::ExpertSlabsMetadata(error) => write!(f, "qwen4: expert slab metadata: {error}"),
            Self::ExpertSlabTable(error) => write!(f, "qwen4: expert slab table: {error}"),
            Self::ExpertSlabsMetadataRequired { name, quant_type } => write!(
                f,
                "qwen4: new I4 expert types or per-expert slab names without \
                 qwen4.expert_slabs v1 metadata are not admitted: {name} (qt={quant_type})"
            ),
            Self::ExpertSlabQuantTypeUnavailable { name, quant_type } => write!(
                f,
                "qwen4: tensor {name} qt={quant_type} is a planar I4 expert type; \
                 routed GPU kernels are unavailable in this build"
            ),
            Self::ExpertSlabKernelsUnavailable { w3, w2, e8p, mq4 } => write!(
                f,
                "qwen4: per-expert slab artifact recognized and validated \
                 (W3={w3}, W2={w2}, E8P={e8p}, MQ4={mq4}); MQ3-I4/MQ2-I4/E8P-I4/pointer-addressed MQ4 \
                 routed kernels are not available on this device in this build, refusing to load"
            ),
        }
    }
}

impl std::error::Error for Qwen4ArtifactError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ExpertSlabsMetadata(error) => Some(error),
            Self::ExpertSlabTable(error) => Some(error),
            _ => None,
        }
    }
}

/// Recognize the new layout before legacy config/manifest admission. No
/// payload reads or GPU work: scale validation is a separate callable API.
fn refuse_expert_slabs(
    metadata_json: &str,
    tensors: &[HfqTensorInfo],
) -> Result<(), Qwen4ArtifactError> {
    let metadata = match expert_slabs::parse(metadata_json) {
        Ok(metadata) => metadata,
        // Invalid legacy JSON still receives the existing config diagnostic.
        Err(ExpertSlabsError::InvalidJson(_) | ExpertSlabsError::MetadataNotObject) => None,
        Err(error) => return Err(Qwen4ArtifactError::ExpertSlabsMetadata(error)),
    };
    if let Some(metadata) = metadata {
        let value: serde_json::Value = serde_json::from_str(metadata_json)
            .map_err(|error| Qwen4ArtifactError::new(format!("qwen4: metadata: {error}")))?;
        let keep = crate::ExpertKeep::from_metadata(&value).map_err(Qwen4ArtifactError::new)?;
        let table = build_expert_slab_table_with_keep(&metadata, tensors, keep.as_ref())
            .map_err(Qwen4ArtifactError::ExpertSlabTable)?;
        let counts = table.counts();
        return Err(Qwen4ArtifactError::ExpertSlabKernelsUnavailable {
            w3: counts.w3,
            w2: counts.w2,
            e8p: counts.e8p,
            mq4: counts.mq4,
        });
    }
    if let Some(info) = tensors.iter().find(|info| {
        matches!(info.quant_type, 131..=136) || parse_expert_entry_name(&info.name).is_some()
    }) {
        return Err(Qwen4ArtifactError::ExpertSlabsMetadataRequired {
            name: info.name.clone(),
            quant_type: info.quant_type,
        });
    }
    Ok(())
}

/// The artifact-load receipts flags and the gfx1151 E8P switch are owned by
/// [`rdna_compute::expert_capabilities`] (the one capability table); they are re-exported here
/// for the artifact gate. `SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED` is `true`: the Halo G1 receipts
/// named in [`rdna_compute::slab_admission`] are pre-rc5 history for
/// `e8p_i4_moe_gemv_pm_gfx1151` and `qwen4_moe_iu4_sym_e8p_pm_gfx1151`; the current rc5 bounded
/// images (96-byte K1, 64-byte K2) are not claimed GPU qualified by it. `false` would hold the
/// gfx1151 E8P cells (`Hold::Gfx1151E8pReceiptsPending`) and refuse MQ4 + E8P layers with the
/// arena `ArchRefused` error; it does not affect [`slab_route_admitted`] or any other cell.
pub use rdna_compute::expert_capabilities::{
    SLAB_ROUTE_GFX1151_RECEIPTS_LANDED, SLAB_ROUTE_RECEIPTS_LANDED,
};
pub use rdna_compute::slab_admission::SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED;

/// Whether the sealed expert-slab route may load on GPU architecture `arch`:
/// [`rdna_compute::expert_capabilities::slab_artifact_route_admitted`] (the architecture serves
/// MQ4 and its own receipts flag is set: gfx1100, gfx1151 and gfx1201). Which cold codec rides
/// next to MQ4 is per architecture, checked per expert by
/// [`crate::expert_arena::ExpertArenaPlan::check_executable`].
pub fn slab_route_admitted(arch: &str) -> bool {
    rdna_compute::expert_capabilities::slab_artifact_route_admitted(arch)
}

/// The arena plan of a version-1 expert-slab artifact when the route is
/// admitted on `device`; otherwise the unchanged fail-closed refusal
/// ([`refuse_expert_slabs`]). `Ok(None)` is an ordinary (non-slab) artifact.
fn expert_slab_plan(
    metadata_json: &str,
    tensors: &[HfqTensorInfo],
    device: Option<&str>,
) -> Result<Option<Arc<crate::expert_arena::ExpertArenaPlan>>, Qwen4ArtifactError> {
    if !device.is_some_and(slab_route_admitted) {
        refuse_expert_slabs(metadata_json, tensors)?;
        return Ok(None);
    }
    let metadata = match expert_slabs::parse(metadata_json) {
        Ok(metadata) => metadata,
        Err(ExpertSlabsError::InvalidJson(_) | ExpertSlabsError::MetadataNotObject) => None,
        Err(error) => return Err(Qwen4ArtifactError::ExpertSlabsMetadata(error)),
    };
    let Some(metadata) = metadata else {
        refuse_expert_slabs(metadata_json, tensors)?;
        return Ok(None);
    };
    let value: serde_json::Value = serde_json::from_str(metadata_json)
        .map_err(|error| Qwen4ArtifactError::new(format!("qwen4: metadata: {error}")))?;
    let keep = crate::ExpertKeep::from_metadata(&value).map_err(Qwen4ArtifactError::new)?;
    let table = build_expert_slab_table_with_keep(&metadata, tensors, keep.as_ref())
        .map_err(Qwen4ArtifactError::ExpertSlabTable)?;
    // W2 has no kernels on any architecture.
    let counts = table.counts();
    if counts.w2 != 0 {
        return Err(Qwen4ArtifactError::ExpertSlabKernelsUnavailable {
            w3: counts.w3,
            w2: counts.w2,
            e8p: counts.e8p,
            mq4: counts.mq4,
        });
    }
    // The arena plan names every expert's codec; the architecture's admitted codecs (gfx1100 MQ4 + E8P,
    // gfx1151 MQ4 + W3 + E8P (E8P via SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED), gfx1201 MQ4 + W3 + E8P)
    // and one cold codec per layer refuse the rest.
    let plan = crate::expert_arena::ExpertArenaPlan::from_table(&table)
        .map_err(|error| Qwen4ArtifactError::new(format!("qwen4: expert slab arenas: {error}")))?;
    plan.check_executable(device.unwrap_or_default())
        .map_err(|error| Qwen4ArtifactError::new(format!("qwen4: expert slab arenas: {error}")))?;
    Ok(Some(Arc::new(plan)))
}

/// `manifest` (declared under the packed recipe) re-declared under
/// [`crate::weights::Qwen4TierRecipe::FlashNextSlab`] with `plan`'s arenas.
fn slab_manifest(
    config: &Qwen4Config,
    manifest: &Qwen4Manifest,
    plan: &Arc<crate::expert_arena::ExpertArenaPlan>,
) -> Result<Qwen4Manifest, Qwen4ArtifactError> {
    let crate::weights::Qwen4TierRecipe::FlashNextPacked {
        trunk_tier,
        mtp_source,
    } = &manifest.identity.recipe
    else {
        return Err(Qwen4ArtifactError::new(
            "qwen4: an expert-slab artifact is admitted under the packed Flash-Next recipe only",
        ));
    };
    Qwen4Manifest::build_with_recipe(
        config,
        crate::weights::Qwen4TierRecipe::FlashNextSlab {
            trunk_tier: *trunk_tier,
            mtp_source: *mtp_source,
            arenas: plan.clone(),
        },
    )
    .map_err(|error| {
        Qwen4ArtifactError::new(format!("qwen4: slab manifest admission failed: {error}"))
    })
}

/// Escape hatch for deliberately loading an artifact whose manifest target is
/// quantized while its bytes are BF16 (the unpacked decode path).
fn unpacked_quantized_target_allowed() -> bool {
    hipfire_config::developer_var("HIPFIRE_ALLOW_BF16_QUANTIZED_TARGET")
        .is_ok_and(|value| value == "1")
}

fn source_dtype_name(quant_type: u8) -> Option<&'static str> {
    match quant_type {
        QWEN4_QT_BF16 => Some("BF16"),
        QWEN4_QT_I64 => Some("I64"),
        QWEN4_QT_MQ4G256V2 => Some("MQ4G256V2"),
        QWEN4_QT_MQ4G128V2 => Some("MQ4G128V2"),
        QWEN4_QT_MQ6G256V2 => Some("MQ6G256V2"),
        // The effective-source seal is composed by the runtime from the artifact's
        // own index through `quant_type_to_dtype`, which spells qt=3 "Q8_0". This
        // table is compared against that seal, so the spelling has to match it.
        QWEN4_QT_Q8F16 => Some("Q8_0"),
        QWEN4_QT_MFP4G32E8SOA => Some("MFP4G32E8SOA"),
        QWEN4_QT_FP8E4M3B128 => Some("FP8E4M3B128"),
        QWEN4_QT_FP8E4M3PT => Some("FP8E4M3PT"),
        _ => None,
    }
}

fn shape_elements(shape: &[u32], name: &str) -> Result<usize, Qwen4ArtifactError> {
    if shape.is_empty() || shape.iter().any(|&dimension| dimension == 0) {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: source tensor {name} has an empty or zero-sized shape {shape:?}"
        )));
    }
    shape.iter().try_fold(1usize, |elements, &dimension| {
        let dimension = usize::try_from(dimension).map_err(|_| {
            Qwen4ArtifactError::new(format!("qwen4: source tensor {name} shape overflows usize"))
        })?;
        elements.checked_mul(dimension).ok_or_else(|| {
            Qwen4ArtifactError::new(format!(
                "qwen4: source tensor {name} element count overflows usize"
            ))
        })
    })
}

/// Row stride and byte extent of a packed `[rows..., K]` tensor, refusing a
/// `K` that is not a multiple of `k_alignment` (1 = any `K`).
fn quantized_extent(
    shape: &[u32],
    name: &str,
    dtype: DType,
    k_alignment: usize,
) -> Result<(usize, usize), Qwen4ArtifactError> {
    if shape.len() < 2 {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: quantized source tensor {name} shape {shape:?} has no row/K dimensions"
        )));
    }
    let k = usize::try_from(*shape.last().expect("shape length checked")).map_err(|_| {
        Qwen4ArtifactError::new(format!(
            "qwen4: source tensor {name} K dimension overflows usize"
        ))
    })?;
    if k == 0 || k % k_alignment != 0 {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: {dtype:?} tensor {name} needs a nonzero K multiple of {k_alignment}, got {k}"
        )));
    }
    let rows = shape[..shape.len() - 1]
        .iter()
        .try_fold(1usize, |rows, &dimension| {
            rows.checked_mul(usize::try_from(dimension).ok()?)
        })
        .ok_or_else(|| {
            Qwen4ArtifactError::new(format!(
                "qwen4: source tensor {name} row count overflows usize"
            ))
        })?;
    let row_stride = dtype.row_bytes(k).ok_or_else(|| {
        Qwen4ArtifactError::new(format!(
            "qwen4: source tensor {name} has no {dtype:?} row layout for K={k}"
        ))
    })?;
    let extent = rows.checked_mul(row_stride).ok_or_else(|| {
        Qwen4ArtifactError::new(format!(
            "qwen4: source tensor {name} data extent overflows usize"
        ))
    })?;
    Ok((row_stride, extent))
}

fn validate_weight_geometry(
    entry: &WeightEntry,
    info: &HfqTensorInfo,
) -> Result<(), Qwen4ArtifactError> {
    if matches!(info.quant_type, 131..=136) {
        return Err(Qwen4ArtifactError::ExpertSlabQuantTypeUnavailable {
            name: info.name.clone(),
            quant_type: info.quant_type,
        });
    }
    let expected_shape = entry
        .logical_shape
        .iter()
        .copied()
        .map(|dimension| {
            u32::try_from(dimension).map_err(|_| {
                Qwen4ArtifactError::new(format!(
                    "qwen4: manifest dimension for {} does not fit HFQ shape",
                    entry.name
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if info.shape != expected_shape {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: source tensor {} shape {:?} does not match manifest {:?}",
            entry.name, info.shape, entry.logical_shape
        )));
    }

    let (source_dtype, expected_extent, row_stride) = match info.quant_type {
        QWEN4_QT_BF16 => {
            if info.group_size != 0 {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: BF16 tensor {} has group_size={}, expected 0",
                    entry.name, info.group_size
                )));
            }
            let elements = shape_elements(&info.shape, &entry.name)?;
            let extent = elements.checked_mul(2).ok_or_else(|| {
                Qwen4ArtifactError::new(format!(
                    "qwen4: BF16 tensor {} data extent overflows usize",
                    entry.name
                ))
            })?;
            (Some(DType::BF16), extent, 0usize)
        }
        QWEN4_QT_MQ4G256V2 => {
            if info.group_size != 256 {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: tensor {} qt44 group_size={}, expected 256",
                    entry.name, info.group_size
                )));
            }
            let (row_stride, extent) =
                quantized_extent(&info.shape, &entry.name, DType::MQ4G256V2, 256)?;
            (Some(DType::MQ4G256V2), extent, row_stride)
        }
        QWEN4_QT_MQ6G256V2 => {
            if info.group_size != 256 {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: tensor {} qt47 group_size={}, expected 256",
                    entry.name, info.group_size
                )));
            }
            let (row_stride, extent) =
                quantized_extent(&info.shape, &entry.name, DType::MQ6G256V2, 256)?;
            (Some(DType::MQ6G256V2), extent, row_stride)
        }
        QWEN4_QT_MFP4G32E8SOA => {
            if info.group_size != 32 {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: tensor {} qt35 group_size={}, expected 32",
                    entry.name, info.group_size
                )));
            }
            let (row_stride, extent) = quantized_extent(
                &info.shape,
                &entry.name,
                DType::MFP4G32E8SOA,
                QWEN4_MFP4G32E8SOA_K_ALIGNMENT,
            )?;
            (Some(DType::MFP4G32E8SOA), extent, row_stride)
        }
        QWEN4_QT_Q8F16 => {
            if info.group_size != 32 {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: tensor {} qt3 group_size={}, expected 32",
                    entry.name, info.group_size
                )));
            }
            let (row_stride, extent) = quantized_extent(&info.shape, &entry.name, DType::Q8_0, 1)?;
            (Some(DType::Q8_0), extent, row_stride)
        }
        QWEN4_QT_MQ4G128V2 => {
            if info.group_size != 128 {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: tensor {} qt53 group_size={}, expected 128",
                    entry.name, info.group_size
                )));
            }
            let (row_stride, extent) =
                quantized_extent(&info.shape, &entry.name, DType::MQ4G128V2, 1)?;
            (Some(DType::MQ4G128V2), extent, row_stride)
        }
        QWEN4_QT_FP8E4M3B128 => {
            if info.group_size != 128 {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: tensor {} qt55 group_size={}, expected 128",
                    entry.name, info.group_size
                )));
            }
            let dims = info
                .shape
                .iter()
                .map(|&dimension| dimension as usize)
                .collect::<Vec<_>>();
            let (experts, rows, k) = match dims[..] {
                [experts, rows, k] => (experts, rows, k),
                [rows, k] => (1, rows, k),
                _ => {
                    return Err(Qwen4ArtifactError::new(format!(
                        "qwen4: FP8E4M3B128 tensor {} shape {:?} is not [experts, rows, K]",
                        entry.name, info.shape
                    )))
                }
            };
            let block = rdna_compute::fp8_e4m3::b128_block_stride(rows, k);
            let extent = rdna_compute::fp8_e4m3::b128_tensor_bytes(experts, rows, k)
                .filter(|_| experts != 0 && rows != 0 && k != 0)
                .zip(block)
                .ok_or_else(|| {
                    Qwen4ArtifactError::new(format!(
                        "qwen4: FP8E4M3B128 tensor {} shape {:?} has no exact extent",
                        entry.name, info.shape
                    ))
                })?;
            (Some(DType::FP8E4M3B128), extent.0, extent.1)
        }
        QWEN4_QT_FP8E4M3PT => {
            if info.group_size != 0 {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: tensor {} qt56 group_size={}, expected 0",
                    entry.name, info.group_size
                )));
            }
            let [rows, k] = info.shape[..] else {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: FP8E4M3PT tensor {} shape {:?} is not [rows, K]",
                    entry.name, info.shape
                )));
            };
            let extent = rdna_compute::fp8_e4m3::pt_tensor_bytes(rows as usize, k as usize)
                .ok_or_else(|| {
                    Qwen4ArtifactError::new(format!(
                        "qwen4: FP8E4M3PT tensor {} extent overflows usize",
                        entry.name
                    ))
                })?;
            (Some(DType::FP8E4M3PT), extent, k as usize)
        }
        other => {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: tensor {} has unsupported HFQM quant_type={other}",
                entry.name
            )));
        }
    };

    match entry.dtype {
        // The stock-FP8 tiers declare exactly one representation: the tag must
        // be the declared dtype, never a BF16 or packed stand-in.
        DType::FP8E4M3B128 | DType::FP8E4M3PT => {
            if source_dtype != Some(entry.dtype) || !entry.dtype_constraint.accepts(entry.dtype) {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: tensor {} is declared {:?} by the manifest but carries quant_type={}",
                    entry.name, entry.dtype, info.quant_type
                )));
            }
        }
        DType::MQ4G256V2
        | DType::MQ4G128V2
        | DType::MQ6G256V2
        | DType::MFP4G32E8SOA
        | DType::Q8_0 => {
            // An artifact whose manifest target is quantized but whose bytes are
            // BF16 decodes that matrix through the unpacked path.  Supporting
            // that silently is how a packed trunk regresses to BF16 without
            // anyone noticing, so it takes an explicit opt-in — *unless* the
            // entry's own source contract admits BF16, which is the case for a
            // table that ships in two tiers and whose artifact declares which
            // one it carries (Qwen4 PLE rows: Q8F16 or BF16).
            if source_dtype == Some(DType::BF16)
                && !entry.dtype_constraint.accepts(DType::BF16)
                && !unpacked_quantized_target_allowed()
            {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: tensor {} is declared {:?} by the manifest but the artifact \
                     carries BF16 bytes; this artifact would decode it unpacked. \
                     Re-publish with the packed trunk, or set \
                     HIPFIRE_ALLOW_BF16_QUANTIZED_TARGET=1 to load it anyway",
                    entry.name, entry.dtype
                )));
            }
            // A converted artifact may retain BF16 source bytes for a matrix
            // whose manifest target is quantized.  When it is already packed,
            // the tag must name a representation the entry's *own* source
            // contract admits — not the target this build happens to declare.
            // The trunk is the class where those differ: its contract names
            // both MQ6G256V2 (qt=47) and Q8F16 (qt=3) because the artifact
            // chooses the tier, and every extent above was computed from the
            // tag the artifact actually carries.  This is the same question the
            // runtime's weight store asks at fulfillment, so the two layers
            // admit the same artifacts.
            if source_dtype != Some(DType::BF16)
                && !entry
                    .dtype_constraint
                    .accepts(source_dtype.expect("every admitted tag names a dtype"))
            {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: tensor {} source quant tag {} is not an admitted source for manifest dtype {:?}",
                    entry.name, info.quant_type, entry.dtype
                )));
            }
        }
        DType::BF16 => {
            if source_dtype != Some(DType::BF16) {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: native BF16 tensor {} must use HFQM quant_type=16, got {}",
                    entry.name, info.quant_type
                )));
            }
        }
        dtype => {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: manifest tensor {} has unsupported target dtype {dtype:?}",
                entry.name
            )));
        }
    }

    if let WeightResidency::ExternalRows {
        row_bytes,
        valid_rows,
    } = entry.residency
    {
        let physical_rows = usize::try_from(*info.shape.first().expect("shape is non-empty"))
            .map_err(|_| {
                Qwen4ArtifactError::new(format!(
                    "qwen4: external tensor {} row count overflows usize",
                    entry.name
                ))
            })?;
        let width = info.shape[1..]
            .iter()
            .try_fold(1usize, |width, &dimension| {
                let dimension = usize::try_from(dimension).map_err(|_| {
                    Qwen4ArtifactError::new(format!(
                        "qwen4: external tensor {} width overflows usize",
                        entry.name
                    ))
                })?;
                width.checked_mul(dimension).ok_or_else(|| {
                    Qwen4ArtifactError::new(format!(
                        "qwen4: external tensor {} row width overflows usize",
                        entry.name
                    ))
                })
            })?;
        // A row-addressed table declares its tier through the artifact: the
        // shard's own quant tag names the representation, its own dtype-derived
        // stride must be the one the manifest declared, and the entry's source
        // contract has to admit that tier. Qwen4 PLE rows ship as BF16 and as
        // Q8F16, so both are answered here rather than one being privileged.
        let declared = match info.quant_type {
            QWEN4_QT_BF16 => DType::BF16,
            QWEN4_QT_Q8F16 => DType::Q8_0,
            QWEN4_QT_FP8E4M3PT => DType::FP8E4M3PT,
            other => {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: external tensor {} has quant_type={other}, which is not a \
                     row-addressed tier",
                    entry.name
                )));
            }
        };
        external_row_stride(declared, width).ok_or_else(|| {
            Qwen4ArtifactError::new(format!(
                "qwen4: external tensor {} width {width} is not addressable as {declared:?} rows",
                entry.name
            ))
        })?;
        // `row_bytes` is the stride the declaration defaults to. A table may
        // admit more than one row tier (Qwen4 PLE rows ship as BF16 and Q8F16),
        // so the declaration only has to name a stride that is coherent for this
        // width; the artifact's own tag and extent below pin the real geometry.
        let declared_stride_is_a_tier = [DType::BF16, DType::Q8_0, DType::FP8E4M3PT]
            .into_iter()
            .any(|tier| external_row_stride(tier, width) == Some(row_bytes));
        if !declared_stride_is_a_tier {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: external tensor {} declares row_stride={}, which is not a row tier \
                 for width {width}",
                entry.name, row_bytes
            )));
        }
        if !entry.dtype_constraint.accepts(declared) {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: external tensor {} carries {declared:?} rows, which its manifest \
                 declaration does not admit",
                entry.name
            )));
        }
        if valid_rows == 0 || valid_rows > physical_rows {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: external tensor {} valid_rows={} outside 1..={physical_rows}",
                entry.name, valid_rows
            )));
        }
        let expected_group = if declared == DType::Q8_0 { 32 } else { 0 };
        if info.group_size != expected_group {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: external tensor {} has quant_type={}, group_size={}, expected \
                 {declared:?} rows",
                entry.name, info.quant_type, info.group_size
            )));
        }
        // The extent is pinned by the tag and shape above and compared against
        // the index below; the declaration's stride only has to be a coherent
        // tier for this width, because a two-tier table cannot declare one
        // stride for both.
    } else if row_stride != 0 && info.data_size != expected_extent {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: tensor {} row_stride={} data_size={} expected {}",
            entry.name, row_stride, info.data_size, expected_extent
        )));
    }
    if info.data_size != expected_extent {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: tensor {} data_size={} expected {}",
            entry.name, info.data_size, expected_extent
        )));
    }
    Ok(())
}

fn validate_metadata_geometry(
    metadata: &crate::weights::MetadataTensor,
    info: &HfqTensorInfo,
) -> Result<(), Qwen4ArtifactError> {
    let expected_shape = metadata
        .shape
        .iter()
        .copied()
        .map(|dimension| {
            u32::try_from(dimension).map_err(|_| {
                Qwen4ArtifactError::new(format!(
                    "qwen4: metadata dimension for {} does not fit HFQ shape",
                    metadata.name
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if info.shape != expected_shape {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: metadata tensor {} shape {:?} does not match {:?}",
            metadata.name, info.shape, metadata.shape
        )));
    }
    if metadata.source_dtype != "I64" || info.quant_type != QWEN4_QT_I64 || info.group_size != 0 {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: metadata tensor {} must use HFQM I64 (qt54/group0), got qt{}/group{}",
            metadata.name, info.quant_type, info.group_size
        )));
    }
    let expected_extent = shape_elements(&info.shape, &metadata.name)?
        .checked_mul(8)
        .ok_or_else(|| {
            Qwen4ArtifactError::new(format!(
                "qwen4: metadata tensor {} data extent overflows usize",
                metadata.name
            ))
        })?;
    if info.data_size != expected_extent {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: metadata tensor {} data_size={} expected {} (I64)",
            metadata.name, info.data_size, expected_extent
        )));
    }
    if expected_extent > MAX_PLE_METADATA_BYTES {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: metadata tensor {} exceeds bounded PLE record size {}",
            metadata.name, MAX_PLE_METADATA_BYTES
        )));
    }
    Ok(())
}

fn is_vision_tensor(name: &str) -> bool {
    [
        "model.visual.",
        "model.vision_tower.",
        "model.vision_projection.",
        "model.multi_modal_projector.",
        "vision_tower.",
        "visual.",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}

/// `slab`: the artifact's per-expert entries (qt 44/53/131/132, named
/// `...experts.{N}.gate_up_proj`) are validated end to end by the span table
/// and read as layer arenas; the per-tensor identity loop skips them (it knows
/// only the dtypes of the ordinary source tensors).
fn validate_source_identity(
    hfq: &HfqFile,
    slab: bool,
) -> Result<Arc<SourceIdentity>, Qwen4ArtifactError> {
    let effective = hfq.source_identity().map_err(|error| {
        Qwen4ArtifactError::new(format!("qwen4: source identity seal failed: {error}"))
    })?;
    if effective.format != SourceFormat::Hfq {
        return Err(Qwen4ArtifactError::new("qwen4: source identity is not HFQ"));
    }
    if effective.files.len() != 1 {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: exact HFQM source requires one backing file, got {}",
            effective.files.len()
        )));
    }
    let loaded = hfq.load_identity().map_err(|error| {
        Qwen4ArtifactError::new(format!("qwen4: source identity stat failed: {error}"))
    })?;
    if loaded.arch_id != ARCH_ID
        || effective.canonical_path != loaded.canonical_path
        || effective.metadata_json != loaded.metadata_json
    {
        return Err(Qwen4ArtifactError::new(
            "qwen4: source identity metadata/path does not match opened HFQM",
        ));
    }
    let file = &effective.files[0];
    if file.canonical_path != loaded.canonical_path
        || file.dev != loaded.dev
        || file.ino != loaded.ino
        || file.len != loaded.len
        || file.mtime_secs != loaded.mtime_secs
        || file.mtime_nanos != loaded.mtime_nanos
    {
        return Err(Qwen4ArtifactError::new(
            "qwen4: source file identity changed while admitting HFQM",
        ));
    }
    if effective.manifest.len() != hfq.tensors().len()
        || loaded.manifest.len() != hfq.tensors().len()
    {
        return Err(Qwen4ArtifactError::new(
            "qwen4: source identity manifest length does not match HFQM index",
        ));
    }
    for (index, info) in hfq.tensors().iter().enumerate() {
        if slab && parse_expert_entry_name(&info.name).is_some() {
            continue;
        }
        let range_identity = &effective.manifest[index];
        let loaded_identity = &loaded.manifest[index];
        let expected_dtype = source_dtype_name(info.quant_type).ok_or_else(|| {
            Qwen4ArtifactError::new(format!(
                "qwen4: source identity contains unknown HFQM quant_type={} for {}",
                info.quant_type, info.name
            ))
        })?;
        let logical_shape = info
            .shape
            .iter()
            .map(|&dimension| dimension as usize)
            .collect::<Vec<_>>();
        if range_identity.name != info.name
            || range_identity.file_index != 0
            || range_identity.offset != info.data_offset as u64
            || range_identity.length != info.data_size as u64
            || range_identity.dtype != expected_dtype
            || range_identity.logical_shape != logical_shape
            || loaded_identity.name != info.name
            || loaded_identity.quant_type != info.quant_type
            || loaded_identity.shape != info.shape
            || loaded_identity.group_size != info.group_size
            || loaded_identity.data_offset != info.data_offset
            || loaded_identity.data_size != info.data_size
        {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: source identity geometry for {} does not match HFQM index",
                info.name
            )));
        }
        let range = hfq
            .tensor_range(&info.name)
            .map_err(|error| {
                Qwen4ArtifactError::new(format!(
                    "qwen4: source range seal failed for {}: {error}",
                    info.name
                ))
            })?
            .ok_or_else(|| {
                Qwen4ArtifactError::new(format!("qwen4: source range missing for {}", info.name))
            })?;
        if range.source_identity() != effective.as_ref()
            || range.offset != info.data_offset as u64
            || range.length != info.data_size as u64
            || range.dtype() != expected_dtype
            || range.logical_shape() != logical_shape
        {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: source range identity for {} does not match HFQM geometry",
                info.name
            )));
        }
    }
    Ok(effective)
}

/// The base file's PLE shard tensors against what the rows source allows.
///
/// Returns the number of PLE shard tensors the index carries (and that the
/// manifest therefore must not declare).  `sidecar` is whether a PLE rows
/// sidecar is attached: then the base is either a split base (marker present,
/// no shards) or the monolithic file (all shards, shadowed by the sidecar and
/// never read).  Without a sidecar the manifest declares the shards, so a
/// split base fails here with the actionable message instead of a bare
/// missing-tensor error.
///
/// The shard count is the profile's: a profile without PLE has no shards, no
/// split-base marker and no sidecar, and an artifact that carries either is
/// refused rather than ignored.
fn validate_ple_shard_presence(
    hfq: &HfqFile,
    sidecar: bool,
    ple: Option<&Qwen4PleConfig>,
) -> Result<usize, Qwen4ArtifactError> {
    let Some(ple) = ple else {
        if sidecar || hfq.ple_rows_marker().is_some() {
            return Err(Qwen4ArtifactError::new(
                "qwen4: this profile declares no PLE, but the artifact carries a PLE rows \
                 sidecar or split-base marker",
            ));
        }
        return Ok(0);
    };
    let shard_count = ple.shard_count;
    let present = hfq
        .tensors()
        .iter()
        .filter(|tensor| is_ple_shard_name_within(&tensor.name, shard_count))
        .count();
    let split = present == 0 && hfq.ple_rows_marker().is_some();
    if !sidecar {
        if split {
            return Err(Qwen4ArtifactError::new(
                "qwen4: this is a split base (its PLE n-gram rows live in a sidecar) but no PLE \
                 rows file is attached; put <model>.ple-q4 or <model>.ple-q8 next to it \
                 (`hipfire pull <model> --ple q4|q8`) and select the format with \
                 memory.ple_format",
            ));
        }
        return Ok(0);
    }
    if ple.layer != 1 {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: PLE rows sidecars address the PLE block at layer 1; this profile's PLE is \
             at layer {}",
            ple.layer
        )));
    }
    if present != 0 && present != shard_count {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: source carries {present} of {shard_count} PLE shard tensors; a PLE rows \
             sidecar needs all of them (monolithic) or none (split base)"
        )));
    }
    if present == 0 && !split {
        return Err(Qwen4ArtifactError::new(
            "qwen4: source has no PLE shard tensors and no hipfire_ple_rows split-base marker; \
             a PLE rows sidecar cannot bind to it",
        ));
    }
    Ok(present)
}

/// The complete expected inventory against an HFQ tensor index: exact names,
/// shapes, quant tags, group sizes and byte extents.  It reads only index
/// records, so it runs identically over a real container and over a
/// header-only dry inventory, and every refusal happens before any allocation
/// proportional to tensor payloads.
///
/// `shadowed_ple_shards` is how many index records a PLE rows sidecar shadows
/// (`shadow_shard_count` is the profile's shard count for recognising them).
fn validate_manifest_inventory(
    tensors: &[HfqTensorInfo],
    manifest: &Qwen4Manifest,
    shadowed_ple_shards: usize,
    shadow_shard_count: usize,
) -> Result<usize, Qwen4ArtifactError> {
    let mut expected = BTreeMap::<String, Vec<u32>>::new();
    for entry in &manifest.weights {
        if matches!(&entry.policy, ShardPolicy::Tied { .. }) {
            continue;
        }
        let shape = entry
            .logical_shape
            .iter()
            .copied()
            .map(|dimension| {
                u32::try_from(dimension).map_err(|_| {
                    Qwen4ArtifactError::new(format!(
                        "qwen4: manifest dimension for {} does not fit HFQ shape",
                        entry.name
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if expected.insert(entry.name.clone(), shape).is_some() {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: manifest contains duplicate source tensor {}",
                entry.name
            )));
        }
    }
    for entry in &manifest.metadata {
        let shape = entry
            .shape
            .iter()
            .copied()
            .map(|dimension| {
                u32::try_from(dimension).map_err(|_| {
                    Qwen4ArtifactError::new(format!(
                        "qwen4: metadata dimension for {} does not fit HFQ shape",
                        entry.name
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if expected.insert(entry.name.clone(), shape).is_some() {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: manifest metadata duplicates source tensor {}",
                entry.name
            )));
        }
    }
    let manifest_count = manifest.source_tensor_count();
    if expected.len() != manifest_count {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: manifest source inventory has {} descriptors but contract declares {manifest_count}",
            expected.len()
        )));
    }
    let hfq_declared = tensors
        .len()
        .checked_sub(shadowed_ple_shards)
        .ok_or_else(|| {
            Qwen4ArtifactError::new("qwen4: more shadowed PLE shards than index records")
        })?;
    if hfq_declared != manifest_count {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: HFQ source inventory has {hfq_declared} tensors but manifest declares {manifest_count}",
        )));
    }
    for tensor in tensors {
        if is_vision_tensor(&tensor.name) {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: vision tensor {} is not admitted by the text-only source boundary",
                tensor.name
            )));
        }
        if shadowed_ple_shards != 0 && is_ple_shard_name_within(&tensor.name, shadow_shard_count) {
            // Shadowed by the attached sidecar: `HfqFile::attach_ple_rows`
            // already bound their geometry to the sidecar header.
            continue;
        }
        let Some(shape) = expected.remove(&tensor.name) else {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: source tensor {} is not declared by the manifest",
                tensor.name
            )));
        };
        if tensor.shape != shape {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: source tensor {} has shape {:?}, expected manifest shape {:?}",
                tensor.name, tensor.shape, shape
            )));
        }
    }
    if let Some((name, _)) = expected.into_iter().next() {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: source inventory is missing manifest tensor {name}"
        )));
    }
    // Every index name is unique and declared at this point, so a by-name
    // index is exact.
    let by_name: BTreeMap<&str, &HfqTensorInfo> = tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();
    for entry in &manifest.weights {
        if matches!(&entry.policy, ShardPolicy::Tied { .. }) {
            continue;
        }
        let info = by_name.get(entry.name.as_str()).copied().ok_or_else(|| {
            Qwen4ArtifactError::new(format!(
                "qwen4: source tensor {} disappeared while validating geometry",
                entry.name
            ))
        })?;
        validate_weight_geometry(entry, info)?;
    }
    for metadata in &manifest.metadata {
        let info = by_name
            .get(metadata.name.as_str())
            .copied()
            .ok_or_else(|| {
                Qwen4ArtifactError::new(format!(
                    "qwen4: metadata tensor {} disappeared while validating geometry",
                    metadata.name
                ))
            })?;
        validate_metadata_geometry(metadata, info)?;
    }
    Ok(manifest_count)
}

/// The PLE hash metadata of `metadata_json` against the profile's PLE.
///
/// A profile without PLE has no `qwen4_ple` object (one present is refused).
/// Flash-Next must equal the pinned canonical metadata exactly; a synthetic
/// profile is checked against its own explicit geometry.
fn admit_ple_metadata(
    metadata_json: &str,
    config: &Qwen4Config,
) -> Result<Option<PleHashMetadata>, Qwen4ArtifactError> {
    let Some(ple_config) = config.ple.as_ref() else {
        let value: serde_json::Value = serde_json::from_str(metadata_json).map_err(|error| {
            Qwen4ArtifactError::new(format!("qwen4: PLE metadata admission failed: {error}"))
        })?;
        if value.get("qwen4_ple").is_some() {
            return Err(Qwen4ArtifactError::new(
                "qwen4: this profile declares no PLE, but the metadata carries a qwen4_ple object",
            ));
        }
        return Ok(None);
    };
    if config.profile.is_flash_next() {
        let ple = PleHashMetadata::from_json(metadata_json).map_err(|error| {
            Qwen4ArtifactError::new(format!("qwen4: PLE metadata admission failed: {error}"))
        })?;
        if ple != PleHashMetadata::qwen4() {
            return Err(Qwen4ArtifactError::new(
                "qwen4: PLE metadata does not match the pinned manifest contract",
            ));
        }
        return Ok(Some(ple));
    }
    let ple = PleHashMetadata::from_json_for(metadata_json, ple_config).map_err(|error| {
        Qwen4ArtifactError::new(format!("qwen4: PLE metadata admission failed: {error}"))
    })?;
    ple_config.check_hash_metadata(&ple).map_err(|error| {
        Qwen4ArtifactError::new(format!(
            "qwen4: PLE metadata does not match the profile's PLE geometry: {error}"
        ))
    })?;
    Ok(Some(ple))
}

/// Production admission parses with the production parser, which refuses every
/// `hipfire_profile` marker.  Only a build with the explicit
/// `qwen4-dev-profiles` feature routes the developer marker to the developer
/// parser; there is no environment or config key that does.
fn parse_artifact_config(metadata_json: &str) -> Result<Qwen4Config, Qwen4ArtifactError> {
    #[cfg(feature = "qwen4-dev-profiles")]
    let parsed = Qwen4Config::from_developer_metadata_json(metadata_json);
    #[cfg(not(feature = "qwen4-dev-profiles"))]
    let parsed = Qwen4Config::from_metadata_json(metadata_json);
    parsed.map_err(|error| {
        Qwen4ArtifactError::new(format!("qwen4: config admission failed: {error}"))
    })
}

/// Receipt of a header-only inventory admission: no container, payload,
/// source seal or GPU is involved, so it states exactly one thing — the
/// declared index is the profile's exact inventory.
#[derive(Clone, Debug)]
pub struct Qwen4InventoryReceipt {
    pub config: Qwen4Config,
    pub manifest: Qwen4Manifest,
    pub ple: Option<PleHashMetadata>,
    pub source_tensor_count: usize,
}

/// The config and manifest an artifact's metadata declares.
///
/// A Flash-Next config's `qwen4_recipe` object, which predates recipe ids,
/// carries no id for the packed recipe (its manifest comes from
/// [`Qwen4Manifest::build`]); the one Flash-Next id is
/// [`crate::weights::STOCK_FP8_RECIPE_ID`], which selects
/// [`Qwen4TierRecipe::StockFp8`] and the AR-only profile
/// ([`Qwen4Config::into_ar_only`]): the artifact carries no MTP head, so the
/// admitted config carries none either.  A synthetic developer profile
/// declares its recipe by `qwen4_recipe.id`: absent means source-exact BF16,
/// the only recipe before packed synthetic artifacts existed.  An unknown or
/// non-string id is refused rather than guessed.  The declared recipe is
/// explicit and exact: the manifest then admits only the formats and extents
/// that recipe declares.
///
/// `drop_head` (the trunk shard's AR-only load,
/// [`crate::weights::trunk_shard::trunk_shard_drops_mtp_head`]) admits a packed
/// Flash-Next artifact headless: the config becomes
/// [`crate::config::Qwen4Profile::FlashNextHeadless`] (no MTP head, the packed recipe
/// kept) and the manifest declares no `mtp.*` tensor, so nothing downstream (weights,
/// census, state, lanes, spec) references a head. The artifact's own `mtp.*` records stay
/// in the file, unread ([`head_free_tensors`]).
fn manifest_for_metadata(
    config: Qwen4Config,
    metadata_json: &str,
    drop_head: bool,
) -> Result<(Qwen4Config, Qwen4Manifest), Qwen4ArtifactError> {
    let value: serde_json::Value = serde_json::from_str(metadata_json).map_err(|error| {
        Qwen4ArtifactError::new(format!("qwen4: recipe metadata admission failed: {error}"))
    })?;
    let declared = value.get(RECIPE_METADATA_KEY);
    let id = match declared {
        None => None,
        Some(recipe) => match recipe.get("id") {
            None => None,
            Some(id) => Some(id.as_str().ok_or_else(|| {
                Qwen4ArtifactError::new(format!("qwen4: {RECIPE_METADATA_KEY}.id must be a string"))
            })?),
        },
    };
    let manifest_error = |error: WeightError| {
        Qwen4ArtifactError::new(format!("qwen4: manifest admission failed: {error}"))
    };
    if config.profile.is_flash_next() {
        let Some(id) = id else {
            let config = if drop_head {
                config.into_headless().map_err(|error| {
                    Qwen4ArtifactError::new(format!(
                        "qwen4: headless config admission failed: {error}"
                    ))
                })?
            } else {
                config
            };
            let manifest = Qwen4Manifest::build(&config).map_err(manifest_error)?;
            return Ok((config, manifest));
        };
        let recipe = Qwen4TierRecipe::from_flash_next_id(id).ok_or_else(|| {
            Qwen4ArtifactError::new(format!(
                "qwen4: {RECIPE_METADATA_KEY}.id {id:?} is not a Flash-Next recipe; the \
                 Flash-Next profile has exactly one recipe per id (none: packed, \"{}\": stock FP8)",
                crate::weights::STOCK_FP8_RECIPE_ID
            ))
        })?;
        let config = config.into_ar_only().map_err(|error| {
            Qwen4ArtifactError::new(format!("qwen4: AR-only config admission failed: {error}"))
        })?;
        let manifest = Qwen4Manifest::build_with_recipe(&config, recipe).map_err(manifest_error)?;
        return Ok((config, manifest));
    }
    let recipe = match id {
        None => Qwen4TierRecipe::SyntheticSourceBf16,
        Some(id) => Qwen4TierRecipe::from_synthetic_id(id).ok_or_else(|| {
            Qwen4ArtifactError::new(format!(
                "qwen4: {RECIPE_METADATA_KEY}.id {id:?} is not a synthetic-profile recipe"
            ))
        })?,
    };
    let manifest = Qwen4Manifest::build_with_recipe(&config, recipe).map_err(manifest_error)?;
    Ok((config, manifest))
}

/// The HFQ index records the manifest of `config` is checked against: every record, unless
/// the config is the head-dropped profile, whose `mtp.*` records (the native MTP head the
/// artifact still carries) are ignored: nothing declares them and nothing reads them.
fn head_free_tensors<'a>(
    config: &Qwen4Config,
    tensors: &'a [HfqTensorInfo],
) -> std::borrow::Cow<'a, [HfqTensorInfo]> {
    if matches!(
        config.profile,
        crate::config::Qwen4Profile::FlashNextHeadless
    ) {
        std::borrow::Cow::Owned(
            tensors
                .iter()
                .filter(|info| !info.name.starts_with("mtp."))
                .cloned()
                .collect(),
        )
    } else {
        std::borrow::Cow::Borrowed(tensors)
    }
}

fn admit_inventory(
    arch_id: u32,
    config: Qwen4Config,
    metadata_json: &str,
    tensors: &[HfqTensorInfo],
    drop_head: bool,
) -> Result<Qwen4InventoryReceipt, Qwen4ArtifactError> {
    if arch_id != ARCH_ID {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: source arch_id={arch_id} does not match reserved arch_id={ARCH_ID}"
        )));
    }
    let (config, manifest) = manifest_for_metadata(config, metadata_json, drop_head)?;
    let source_tensor_count =
        validate_manifest_inventory(&head_free_tensors(&config, tensors), &manifest, 0, 0)?;
    let ple = admit_ple_metadata(metadata_json, &config)?;
    Ok(Qwen4InventoryReceipt {
        config,
        manifest,
        ple,
        source_tensor_count,
    })
}

/// Header-only dry inventory admission under the **production** parser: the
/// declared index of a Flash-Next artifact must be the exact pinned inventory
/// (names, shapes, quant tags, group sizes, byte extents).  A config carrying
/// a developer profile marker is refused.
pub fn admit_inventory_headers(
    arch_id: u32,
    metadata_json: &str,
    tensors: &[HfqTensorInfo],
) -> Result<Qwen4InventoryReceipt, Qwen4ArtifactError> {
    refuse_expert_slabs(metadata_json, tensors)?;
    let config = Qwen4Config::from_metadata_json(metadata_json).map_err(|error| {
        Qwen4ArtifactError::new(format!("qwen4: config admission failed: {error}"))
    })?;
    admit_inventory(arch_id, config, metadata_json, tensors, false)
}

/// Explicit developer admission of a header-only dry inventory: Flash-Next
/// (no marker) or a synthetic developer profile (explicit marker).  The index
/// must still be the profile's exact inventory.  It proves inventory only; it
/// does not make a synthetic profile executable.
#[cfg(feature = "qwen4-dev-profiles")]
pub fn admit_developer_inventory_headers(
    arch_id: u32,
    metadata_json: &str,
    tensors: &[HfqTensorInfo],
) -> Result<Qwen4InventoryReceipt, Qwen4ArtifactError> {
    refuse_expert_slabs(metadata_json, tensors)?;
    let config = Qwen4Config::from_developer_metadata_json(metadata_json).map_err(|error| {
        Qwen4ArtifactError::new(format!("qwen4: config admission failed: {error}"))
    })?;
    admit_inventory(arch_id, config, metadata_json, tensors, false)
}

/// Validate one exact Qwen4 HFQM artifact before any GPU allocation.
///
/// The check covers the immutable container/index identity, the complete
/// manifest inventory, quantized row geometry, PLE external rows, and the
/// three signed-I64 PLE metadata records.  It intentionally accepts only the
/// executable text HFQM contract; safetensors, overlays, vision records, and
/// unknown quant tags fail closed.
///
/// Feature boundary: a default build (no `qwen4-dev-profiles`) is
/// production-strict — the production parser refuses every config carrying a
/// `hipfire_profile` marker, so a synthetic artifact never reaches the loader.
/// A build compiled with the explicit `qwen4-dev-profiles` feature routes the
/// same call through the developer parser, which additionally admits a
/// synthetic profile that carries the explicit developer marker (never one
/// inferred from geometry), still under its exact inventory and extents.  No
/// environment variable or config key crosses this boundary.
pub fn admit_hfqm_artifact(hfq: &HfqFile) -> Result<Qwen4HfqmArtifact, Qwen4ArtifactError> {
    admit_hfqm_artifact_on(hfq, None)
}

/// [`admit_hfqm_artifact`] for a known GPU architecture: an expert-slab
/// artifact is admitted only where [`slab_route_admitted`] holds for `arch`;
/// every other arch (and `None`) keeps the `ExpertSlabKernelsUnavailable`
/// refusal.
pub fn admit_hfqm_artifact_on(
    hfq: &HfqFile,
    slab_device: Option<&str>,
) -> Result<Qwen4HfqmArtifact, Qwen4ArtifactError> {
    admit_hfqm_artifact_with(
        hfq,
        slab_device,
        crate::weights::trunk_shard::trunk_shard_drops_mtp_head(),
    )
}

/// [`admit_hfqm_artifact_on`] with the head policy passed in: `drop_head` admits a packed
/// Flash-Next artifact headless ([`manifest_for_metadata`]). The one place the trunk shard's
/// switch reaches admission; the CPU tests drive it with an explicit value.
pub(crate) fn admit_hfqm_artifact_with(
    hfq: &HfqFile,
    slab_device: Option<&str>,
    drop_head: bool,
) -> Result<Qwen4HfqmArtifact, Qwen4ArtifactError> {
    if hfq.arch_id != ARCH_ID {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: source arch_id={} does not match reserved arch_id={ARCH_ID}",
            hfq.arch_id
        )));
    }
    if hfq.format_version != QWEN4_FORMAT_VERSION {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: unsupported HFQM format version {} (expected {})",
            hfq.format_version, QWEN4_FORMAT_VERSION
        )));
    }
    if hfq.has_overlay() {
        return Err(Qwen4ArtifactError::new(
            "qwen4: REAP/head overlays are not admitted for exact HFQM inventory",
        ));
    }
    let slab = expert_slab_plan(&hfq.metadata_json, hfq.tensors(), slab_device)?;
    let config = parse_artifact_config(&hfq.metadata_json)?;
    let (config, mut manifest) = manifest_for_metadata(config, &hfq.metadata_json, drop_head)?;
    if let Some(plan) = &slab {
        manifest = slab_manifest(&config, &manifest, plan)?;
    }
    let ple_config = config.ple.as_ref();
    let sidecar = hfq.ple_rows().is_some();
    if sidecar && manifest.identity.recipe.is_stock_fp8() {
        // The stock-FP8 PLE rows are the artifact's own qt=56 shards; a Q8/Q4
        // sidecar would serve different values than the FP8 table it shadows.
        return Err(Qwen4ArtifactError::new(
            "qwen4: the stock-FP8 artifact reads its embedded qt=56 FP8E4M3PT PLE rows; a PLE \
             rows sidecar is not admitted for it (remove the sidecar or unset memory.ple_format)",
        ));
    }
    let shadowed_ple_shards = validate_ple_shard_presence(hfq, sidecar, ple_config)?;
    if sidecar {
        manifest = manifest.without_ple_shards();
    }
    let slab_inventory;
    let head_free;
    let (inventory_tensors, inventory_manifest): (&[HfqTensorInfo], &Qwen4Manifest) =
        if slab.is_some() {
            // The 49,152 per-expert entries were validated end to end by the
            // span table; the manifest declares them as per-layer arenas.
            // The remaining tensors face the ordinary exact-inventory check
            // against the manifest with the arenas taken out.
            slab_inventory = (
                head_free_tensors(&config, hfq.tensors())
                    .iter()
                    .filter(|info| parse_expert_entry_name(&info.name).is_none())
                    .cloned()
                    .collect::<Vec<_>>(),
                {
                    let mut without_arenas = manifest.clone();
                    without_arenas
                        .weights
                        .retain(|entry| !entry.name.ends_with(crate::expert_arena::ARENA_SUFFIX));
                    without_arenas
                },
            );
            (&slab_inventory.0, &slab_inventory.1)
        } else {
            head_free = head_free_tensors(&config, hfq.tensors());
            (&head_free[..], &manifest)
        };
    let source_tensor_count = validate_manifest_inventory(
        inventory_tensors,
        inventory_manifest,
        shadowed_ple_shards,
        ple_config.map_or(0, |ple| ple.shard_count),
    )?;
    let ple = admit_ple_metadata(&hfq.metadata_json, &config)?;

    // Seal every descriptor before reading the three tiny records.  This
    // keeps the bounded host reads tied to the same source identity as all
    // later external-row reads.
    let source_identity = validate_source_identity(hfq, slab.is_some())?;
    let ple_rows = match (ple_config, &ple) {
        (Some(ple_config), Some(ple)) => {
            validate_ple_records(hfq, ple_config, ple)?;
            admit_ple_rows(hfq, ple_config, ple)?
        }
        _ => Qwen4PleRows::Absent,
    };

    let placements = manifest
        .placements_for_scope(&config, &Qwen4BandScope::single(&config), 0)
        .map_err(|error| Qwen4ArtifactError::new(error.to_string()))?;
    Ok(Qwen4HfqmArtifact {
        config,
        manifest,
        ple,
        placements,
        source_tensor_count,
        source_identity,
        ple_rows,
    })
}

/// Suffix of a trunk layer's packed routed gate/up expert tensor.
const ROUTED_GATE_UP_SUFFIX: &str = ".experts.gate_up_proj";
/// Suffix of a trunk layer's packed routed down expert tensor.
const ROUTED_DOWN_SUFFIX: &str = ".experts.down_proj";

/// Decode an IEEE binary16 bit pattern exactly (the GPU checker's
/// `(float)__builtin_bit_cast(_Float16, ..)`).
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1f);
    let frac = u32::from(bits & 0x3ff);
    match exp {
        0 => {
            // Zero and subnormals: frac * 2^-24 is exact in f32.
            let magnitude = frac as f32 * (1.0 / 16_777_216.0);
            if sign != 0 {
                -magnitude
            } else {
                magnitude
            }
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (frac << 13)),
        _ => f32::from_bits(sign | ((exp + 112) << 23) | (frac << 13)),
    }
}

/// The symmetric-grid predicate of `qwen4_moe_sym_check.gfx1201.hip` for one
/// K128 header (`sc` = low half, `zp` = high half of the little-endian word):
/// both finite and `zp == -8 * sc` in f32.
fn header_symmetric(scale_bits: u16, zero_bits: u16) -> bool {
    let scale = f16_bits_to_f32(scale_bits);
    let zero = f16_bits_to_f32(zero_bits);
    scale.is_finite() && zero.is_finite() && zero == -8.0f32 * scale
}

/// Every K128 header of a packed payload of `group_bytes` groups: QT44 groups
/// (136 B) carry two headers at byte 0 and 4, QT53 groups (68 B) one at byte 0.
fn payload_symmetric(bytes: &[u8], group_bytes: usize) -> bool {
    let headers = group_bytes / 68;
    bytes.chunks_exact(group_bytes).all(|group| {
        (0..headers).all(|header| {
            let at = 4 * header;
            header_symmetric(
                u16::from_le_bytes([group[at], group[at + 1]]),
                u16::from_le_bytes([group[at + 2], group[at + 3]]),
            )
        })
    })
}

/// Scan the payloads on every available core, stopping at the first
/// violation. `jobs` are `(payload, group_bytes)`.
fn payloads_symmetric(jobs: &[(&[u8], usize)]) -> bool {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let next = AtomicUsize::new(0);
    let symmetric = AtomicBool::new(true);
    let workers = std::thread::available_parallelism()
        .map_or(1, |count| count.get())
        .min(jobs.len())
        .max(1);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                while symmetric.load(Ordering::Relaxed) {
                    let Some(&(bytes, group_bytes)) =
                        jobs.get(next.fetch_add(1, Ordering::Relaxed))
                    else {
                        break;
                    };
                    if !payload_symmetric(bytes, group_bytes) {
                        symmetric.store(false, Ordering::Relaxed);
                    }
                }
            });
        }
    });
    symmetric.load(Ordering::Relaxed)
}

/// [`routed_headers_symmetric`] over the named packed routed tensors.
fn routed_tensors_symmetric(hfq: &HfqFile, names: &[&str]) -> Result<bool, Qwen4ArtifactError> {
    let mut gate_up = 0usize;
    let mut down = 0usize;
    let mut jobs: Vec<(&[u8], usize)> = Vec::with_capacity(names.len());
    for &name in names {
        // QT44 gate/up reduces over K256 groups of 136 bytes, QT53 down over
        // K128 groups of 68 bytes: the only formats the symmetric route reads.
        let (quant_type, values_per_group, group_bytes) = if name.ends_with(ROUTED_GATE_UP_SUFFIX) {
            gate_up += 1;
            (QWEN4_QT_MQ4G256V2, 256usize, 136usize)
        } else if name.ends_with(ROUTED_DOWN_SUFFIX) {
            down += 1;
            (QWEN4_QT_MQ4G128V2, 128usize, 68usize)
        } else {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: {name} is not a packed routed expert tensor"
            )));
        };
        let info = hfq.find_tensor_info(name).ok_or_else(|| {
            Qwen4ArtifactError::new(format!("qwen4: routed expert tensor {name} is missing"))
        })?;
        if info.quant_type != quant_type {
            // BF16 or another tier: not the symmetric grid the route reads.
            return Ok(false);
        }
        let [experts, rows, k] = info.shape[..] else {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: routed expert tensor {name} shape {:?} is not [experts, rows, K]",
                info.shape
            )));
        };
        let (experts, rows, k) = (experts as usize, rows as usize, k as usize);
        let expected = k
            .checked_div(values_per_group)
            .filter(|_| k % values_per_group == 0)
            .and_then(|groups| experts.checked_mul(rows)?.checked_mul(groups))
            .and_then(|groups| groups.checked_mul(group_bytes))
            .ok_or_else(|| {
                Qwen4ArtifactError::new(format!(
                    "qwen4: routed expert tensor {name} shape {:?} is not a whole number of \
                     {values_per_group}-value groups",
                    info.shape
                ))
            })?;
        let (_, bytes) = hfq.tensor_data(name).ok_or_else(|| {
            Qwen4ArtifactError::new(format!(
                "qwen4: routed expert tensor {name} payload is not readable"
            ))
        })?;
        if bytes.len() != expected {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: routed expert tensor {name} payload is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        jobs.push((bytes, group_bytes));
    }
    if gate_up == 0 || gate_up != down {
        return Ok(false);
    }
    Ok(payloads_symmetric(&jobs))
}

/// Whether every trunk routed expert header of the artifact is on the
/// symmetric grid (`zp == -8 * sc`, both finite) the symmetric IU4 MoE route
/// requires, read from the actual payload bytes, not metadata labels.  Mirrors
/// `kernels/src/qwen4_moe_sym_check.gfx1201.hip` exactly: QT44 gate/up and QT53
/// down only, both QT44 subheaders of every group, every expert and row.
///
/// A short or malformed payload is an error; a finite asymmetric header, a
/// non-QT44/QT53 tier, or an inventory without routed experts returns `false`.
pub fn routed_headers_symmetric(
    hfq: &HfqFile,
    manifest: &Qwen4Manifest,
) -> Result<bool, Qwen4ArtifactError> {
    routed_tensors_symmetric(hfq, &routed_scan_names(manifest))
}

/// The routed trunk gate/up and down tensor names [`routed_headers_symmetric`]
/// scans, in manifest order.
fn routed_scan_names(manifest: &Qwen4Manifest) -> Vec<&str> {
    manifest
        .weights
        .iter()
        .filter(|entry| {
            entry.layer.is_some()
                && (entry.name.ends_with(ROUTED_GATE_UP_SUFFIX)
                    || entry.name.ends_with(ROUTED_DOWN_SUFFIX))
        })
        .map(|entry| entry.name.as_str())
        .collect()
}

/// Lowercase hex SHA-256 over the sorted names, quant types, shapes and
/// payload byte ranges of the tensors the scan reads; `None` when a name is
/// missing from the container (the scan itself reports that error).
fn routed_names_digest(hfq: &HfqFile, names: &[&str]) -> Option<String> {
    use sha2::{Digest, Sha256};
    let mut sorted = names.to_vec();
    sorted.sort_unstable();
    let mut digest = Sha256::new();
    digest.update((sorted.len() as u64).to_le_bytes());
    for name in sorted {
        let info = hfq.find_tensor_info(name)?;
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update([info.quant_type]);
        digest.update((info.shape.len() as u64).to_le_bytes());
        for dim in &info.shape {
            digest.update(dim.to_le_bytes());
        }
        digest.update((info.data_offset as u64).to_le_bytes());
        digest.update((info.data_size as u64).to_le_bytes());
    }
    Some(format!("{:x}", digest.finalize()))
}

/// [`routed_headers_symmetric`] with its verdict cached per model file (see
/// [`crate::sym_verdict_cache`]): an in-process memo, and, when `home` is the
/// hipfire home, the verdict file under it. A hit needs the same canonical
/// `path`, stat key, scan version and routed tensor layout. An artifact with an
/// overlay, or a path that cannot be canonicalized or stat'ed, scans every
/// time. Logs one stderr line per file hit or scan; a memo hit is silent.
pub fn routed_headers_symmetric_cached(
    hfq: &HfqFile,
    manifest: &Qwen4Manifest,
    path: &std::path::Path,
    home: Option<&std::path::Path>,
) -> Result<bool, Qwen4ArtifactError> {
    use crate::sym_verdict_cache::{self as cache, Identity, Origin};
    let names = routed_scan_names(manifest);
    let identity = (!hfq.has_overlay())
        .then(|| {
            let path = std::fs::canonicalize(path).ok()?;
            let stat = StatKey::of(&std::fs::metadata(&path).ok()?);
            let names_digest = routed_names_digest(hfq, &names)?;
            Some(Identity {
                path,
                stat,
                names_digest,
            })
        })
        .flatten();
    let Some(identity) = identity else {
        return routed_tensors_symmetric(hfq, &names);
    };
    let file = home.map(cache::cache_file_in_home);
    let (symmetric, origin) = cache::verdict(
        file.as_deref(),
        Some(&cache::PROCESS_MEMO),
        &identity,
        || routed_tensors_symmetric(hfq, &names),
    )?;
    let shown = identity.path.display();
    match origin {
        Origin::Memo => {}
        Origin::File => eprintln!("  qwen4 symmetric IU4 verdict: cached ({shown})"),
        Origin::Scanned {
            seconds,
            stored: true,
        } => {
            eprintln!("  qwen4 symmetric IU4 verdict: scanned in {seconds:.1} s, cached ({shown})")
        }
        Origin::Scanned {
            seconds,
            stored: false,
        } => eprintln!(
            "  qwen4 symmetric IU4 verdict: scanned in {seconds:.1} s, not cached ({shown})"
        ),
    }
    Ok(symmetric)
}

/// [`routed_headers_symmetric_cached`] for the artifact's own path and the
/// hipfire home ([`hipfire_config::hipfire_home`]: `HIPFIRE_HOME`, else
/// `$HOME/.hipfire`). The one entry every scan site calls.
/// `HIPFIRE_QWEN4_SYM_VERDICT_CACHE=0` disables the file and the memo: every
/// call scans.
pub fn routed_headers_symmetric_verdict(
    hfq: &HfqFile,
    manifest: &Qwen4Manifest,
) -> Result<bool, Qwen4ArtifactError> {
    if !hipfire_config::developer_bool(crate::sym_verdict_cache::VERDICT_CACHE_ENV, true) {
        return routed_headers_symmetric(hfq, manifest);
    }
    let home = hipfire_config::hipfire_home().ok();
    routed_headers_symmetric_cached(hfq, manifest, hfq.path(), home.as_deref())
}

/// DESIGN F1: repack the symmetric MQ4-XTS routed experts without their zero
/// plane while each expert-parallel rank compacts them to the device
/// (`2,611,200 -> 2,534,400` bytes per expert). `=0` keeps the shipped layout.
pub const EXPERT_DROP_ZERO_ENV: &str = "HIPFIRE_QWEN4_EXPERT_DROP_ZERO";

/// Default of [`EXPERT_DROP_ZERO_ENV`] when it is unset.
pub const EXPERT_DROP_ZERO_DEFAULT: bool = false;

/// Whether the drop-zero repack is requested for this process.
pub fn expert_drop_zero_requested() -> bool {
    hipfire_config::developer_bool(EXPERT_DROP_ZERO_ENV, EXPERT_DROP_ZERO_DEFAULT)
}

/// What [`apply_expert_drop_zero`] did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpertDropZero {
    /// Not requested: the manifest is untouched.
    Off,
    /// `entries` routed expert stacks were redeclared planar.
    Applied { entries: usize },
    /// Requested but not applicable; the manifest is untouched.
    Declined(String),
}

impl ExpertDropZero {
    /// Whether any entry was redeclared.
    pub fn applied(&self) -> bool {
        matches!(self, Self::Applied { .. })
    }
}

/// Declare the routed expert stacks of an expert-parallel `manifest` drop-zero
/// planar when `requested` and every header of the stacks to repack is proven
/// symmetric (`zp == -8*sc`, both finite) from the artifact's own bytes: the
/// trunk layers by the cached whole-artifact verdict (the one the symmetric IU4
/// route uses), the MTP head's stacks by a direct scan. The planar kernels
/// exist on exact gfx1201 only. Anything else (other arch, non-packed recipe,
/// not expert-parallel, an asymmetric or non-QT44/QT53 trunk or MTP stack)
/// declines and leaves the whole manifest shipped: the launchers bind one
/// layout per device, so the repack is all-or-nothing.
pub fn apply_expert_drop_zero(
    manifest: Qwen4Manifest,
    hfq: &HfqFile,
    arch: &str,
    requested: bool,
) -> Result<(Qwen4Manifest, ExpertDropZero), Qwen4ArtifactError> {
    if !requested {
        return Ok((manifest, ExpertDropZero::Off));
    }
    let decline = |manifest: Qwen4Manifest, why: String| {
        eprintln!("  qwen4 expert drop-zero: declined ({why})");
        Ok((manifest, ExpertDropZero::Declined(why)))
    };
    if arch != "gfx1201" {
        return decline(
            manifest,
            format!("the planar expert kernels exist on gfx1201 only, this is {arch}"),
        );
    }
    // D1 per-expert EP units compose: the planner derives the unit geometry from the planar
    // aggregate sizes declared below, and the unit-arena gather repacks the shipped stacks
    // planar (`EpUnitLayerMap::gather_arena_mq4_planar`).
    if !matches!(
        manifest.identity.recipe,
        crate::weights::Qwen4TierRecipe::FlashNextPacked { .. }
    ) {
        return decline(
            manifest,
            "only the packed aggregate Flash-Next recipe repacks".to_string(),
        );
    }
    let routed = |entry: &WeightEntry| {
        entry.name.ends_with(ROUTED_GATE_UP_SUFFIX) || entry.name.ends_with(ROUTED_DOWN_SUFFIX)
    };
    if !manifest
        .weights
        .iter()
        .filter(|entry| routed(entry))
        .all(|entry| matches!(entry.policy, ShardPolicy::ExpertSharded { .. }))
    {
        return decline(
            manifest,
            "the routed experts are not expert-parallel sharded".to_string(),
        );
    }
    if !routed_headers_symmetric_verdict(hfq, &manifest)? {
        return decline(
            manifest,
            "a trunk routed header is not symmetric (zp != -8*sc)".to_string(),
        );
    }
    let mtp_names: Vec<&str> = manifest
        .weights
        .iter()
        .filter(|entry| entry.layer.is_none() && routed(entry))
        .map(|entry| entry.name.as_str())
        .collect();
    if !mtp_names.is_empty() && !routed_tensors_symmetric(hfq, &mtp_names)? {
        return decline(
            manifest,
            "an MTP head routed stack is not a symmetric QT44/QT53 payload".to_string(),
        );
    }
    let (manifest, entries) = manifest
        .with_expert_drop_zero(|_| true)
        .map_err(|error| Qwen4ArtifactError::new(format!("qwen4: {error}")))?;
    Ok((manifest, ExpertDropZero::Applied { entries }))
}

fn hotset_refusal(reason: impl fmt::Display) -> String {
    format!(
        "qwen4: {HOTSET_ENV} requires the packed Flash-Next aggregate MQ4-XTS routed experts \
         (48 trunk layers x 512 experts, top-10, qt44 gate_up + qt53 down): {reason}"
    )
}

/// Header-only check, with no payload read, that the artifact's layout can
/// back [`hotset_aggregate_slabs`]: the legacy aggregate routed tensors of
/// the packed recipe.  Refuses (naming [`HOTSET_ENV`]) an expert-slab
/// artifact (qwen4.expert_slabs metadata, per-expert entry names, qt131..134
/// planar I4 experts, i.e. the slab arena / K4 route), a stock-FP8 artifact
/// (qt55 routed experts), any trunk routed tensor that is not qt44/qt53, and
/// an overlaid source.  This is an optional eligibility check, called from
/// [`hotset_aggregate_slabs`] during hotset preparation after
/// [`admit_hfqm_artifact_on`]'s generic admission has already succeeded; a
/// failure here means the hotset is not eligible and the caller must fall
/// back to the exact whole-layer placement, not refuse the load.
pub fn hotset_source_preflight(hfq: &HfqFile) -> Result<(), String> {
    if hfq.arch_id != ARCH_ID {
        return Err(hotset_refusal(format!(
            "source arch_id={} is not the Qwen4 arch_id={ARCH_ID}",
            hfq.arch_id
        )));
    }
    if hfq.has_overlay() {
        return Err(hotset_refusal(
            "REAP/head overlays are not admitted (the hotset reads the base mmap directly)",
        ));
    }
    match expert_slabs::parse(&hfq.metadata_json) {
        Ok(None) | Err(ExpertSlabsError::InvalidJson(_) | ExpertSlabsError::MetadataNotObject) => {}
        Ok(Some(_)) => {
            return Err(hotset_refusal(
                "this is a per-expert slab artifact (qwen4.expert_slabs v1 metadata: slab arena / \
                 K4 route), not an aggregate artifact",
            ));
        }
        Err(error) => {
            return Err(hotset_refusal(format!(
                "the artifact carries invalid qwen4.expert_slabs metadata ({error})"
            )));
        }
    }
    for info in hfq.tensors() {
        if parse_expert_entry_name(&info.name).is_some() {
            return Err(hotset_refusal(format!(
                "per-expert slab entry {} is present (slab arena / K4 route), not an aggregate \
                 artifact",
                info.name
            )));
        }
        if matches!(info.quant_type, 131..=136) {
            return Err(hotset_refusal(format!(
                "tensor {} is a planar I4 expert type (qt={}), not aggregate MQ4-XTS",
                info.name, info.quant_type
            )));
        }
        let routed_gate_up = info.name.ends_with(ROUTED_GATE_UP_SUFFIX);
        let routed_down = info.name.ends_with(ROUTED_DOWN_SUFFIX);
        if !(routed_gate_up || routed_down) {
            continue;
        }
        if info.quant_type == QWEN4_QT_FP8E4M3B128 {
            return Err(hotset_refusal(format!(
                "routed tensor {} is qt=55 FP8E4M3B128 (the stock-FP8 artifact keeps every \
                 routed expert resident on gfx942 and has no MQ4 experts)",
                info.name
            )));
        }
        if !info.name.starts_with("model.language_model.layers.") {
            // The MTP head's experts stay host-mapped legacy tensors, outside the hotset.
            continue;
        }
        let expected = if routed_gate_up {
            QWEN4_QT_MQ4G256V2
        } else {
            QWEN4_QT_MQ4G128V2
        };
        if info.quant_type != expected {
            return Err(hotset_refusal(format!(
                "trunk routed tensor {} is qt={}, expected qt={expected}",
                info.name, info.quant_type
            )));
        }
    }
    Ok(())
}

/// The per-expert `(layer * 512 + expert)` source slabs of the artifact's
/// aggregate MQ4-XTS routed experts, for the load-time hotset.  Each span is
/// an **absolute** artifact mmap offset (`HfqTensorInfo::data_offset` plus
/// the expert stride), readable through [`HfqFile::data_range`]; no payload
/// is read, hashed or copied.
///
/// Strict contract: the packed Flash-Next recipe (not stock FP8, not the
/// slab arena), `config.num_hidden_layers == 48`, 512 experts, top-10 (the
/// authoritative `config.ffn` and its scalar mirrors); every trunk layer has
/// qt44 `[512, 1280, 2560]` gate_up and qt53 `[512, 2560, 640]` down of the
/// exact byte length, fully inside the mapping.  Every violation is a
/// refusal naming [`HOTSET_ENV`].  `hfq` and `manifest` must be the admitted
/// artifact (`admit_hfqm_artifact_on`) the slabs are used with: the slab
/// list is bound to the source by structure (names, qt, shape, length,
/// absolute offsets of the sealed index), not by a payload digest.
pub fn hotset_aggregate_slabs(
    hfq: &HfqFile,
    config: &Qwen4Config,
    manifest: &Qwen4Manifest,
) -> Result<Vec<crate::expert_slab_table::ExpertSlab>, String> {
    use crate::expert_slab_aggregate::{
        slabs_from_aggregate_mq4, AggregateLayer, AggregateProjection,
    };
    use crate::expert_slabs::{ExpertFormat, EXPERTS_PER_LAYER, TRUNK_LAYERS};

    if config.expert_keep.is_some() {
        return Err(hotset_refusal(
            "qwen4.expert_keep artifacts do not support expert-hotset/tier placement",
        ));
    }
    hotset_source_preflight(hfq)?;
    if !config.profile.is_packed_flash_next() {
        return Err(hotset_refusal(format!(
            "profile {} is not the packed Flash-Next profile",
            config.profile.id()
        )));
    }
    if !matches!(
        manifest.identity.recipe,
        Qwen4TierRecipe::FlashNextPacked { .. }
    ) {
        return Err(hotset_refusal(
            "the manifest recipe is not the packed Flash-Next recipe (stock-FP8 and slab-arena \
             recipes are not aggregate MQ4)",
        ));
    }
    let layers = TRUNK_LAYERS as usize;
    let experts = EXPERTS_PER_LAYER as usize;
    if config.num_hidden_layers != layers
        || config.num_experts != experts
        || config.num_experts_per_tok != 10
        || !matches!(
            config.ffn,
            crate::config::Qwen4FfnConfig::Moe {
                num_experts: 512,
                top_k: 10,
                ..
            }
        )
    {
        return Err(hotset_refusal(format!(
            "config has {} layers / {} experts / top-{} (need {layers} / {experts} / top-10)",
            config.num_hidden_layers, config.num_experts, config.num_experts_per_tok
        )));
    }

    let (gate_up_qt, down_qt) = ExpertFormat::Mq4Xts.pair();
    let gate_up_len = ExpertFormat::Mq4Xts.gate_up_bytes() * experts;
    let down_len = ExpertFormat::Mq4Xts.down_bytes() * experts;
    let projection = |layer: usize,
                      suffix: &str,
                      qt: u8,
                      shape: [u32; 3],
                      len: usize|
     -> Result<AggregateProjection, String> {
        let name = format!("model.language_model.layers.{layer}.mlp{suffix}");
        if !manifest
            .weights
            .iter()
            .any(|entry| entry.name == name && entry.layer == Some(layer))
        {
            return Err(hotset_refusal(format!(
                "the manifest declares no aggregate routed tensor {name} (nonaggregate layout)"
            )));
        }
        let info = hfq
            .find_tensor_info(&name)
            .ok_or_else(|| hotset_refusal(format!("routed tensor {name} is missing")))?;
        if info.quant_type != qt {
            return Err(hotset_refusal(format!(
                "routed tensor {name} is qt={}, expected qt={qt}",
                info.quant_type
            )));
        }
        if info.shape != shape {
            return Err(hotset_refusal(format!(
                "routed tensor {name} shape {:?}, expected {shape:?}",
                info.shape
            )));
        }
        if info.data_size != len {
            return Err(hotset_refusal(format!(
                "routed tensor {name} is {} bytes, expected {len}",
                info.data_size
            )));
        }
        if hfq.data_range(info.data_offset, info.data_size).is_none() {
            return Err(hotset_refusal(format!(
                "routed tensor {name} bytes [{}, +{}) are not inside the artifact mmap",
                info.data_offset, info.data_size
            )));
        }
        Ok(AggregateProjection {
            offset: info.data_offset as u64,
            len: info.data_size as u64,
            quant_type: info.quant_type,
            shape: info.shape.iter().map(|&dim| dim as usize).collect(),
        })
    };
    let aggregates = (0..layers)
        .map(|layer| -> Result<AggregateLayer, String> {
            Ok(AggregateLayer {
                gate_up: projection(
                    layer,
                    ROUTED_GATE_UP_SUFFIX,
                    gate_up_qt,
                    [512, 1280, 2560],
                    gate_up_len,
                )?,
                down: projection(
                    layer,
                    ROUTED_DOWN_SUFFIX,
                    down_qt,
                    [512, 2560, 640],
                    down_len,
                )?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    slabs_from_aggregate_mq4(&aggregates)
        .map_err(|error| hotset_refusal(format!("aggregate adapter: {error}")))
}

/// Whether every PLE key/value projection runs as BF16 weights: the manifest
/// declares them BF16 and no `HIPFIRE_QWEN4_REQUANT` rule matches them (the
/// load-time requantization would turn them into a quantized matrix).
pub fn ple_key_value_bf16(manifest: &Qwen4Manifest) -> bool {
    let requant = hipfire_config::developer_var("HIPFIRE_QWEN4_REQUANT").ok();
    let mut seen = false;
    for entry in &manifest.weights {
        if !entry.name.contains(".ple.")
            || !(entry.name.ends_with(".key_proj.weight")
                || entry.name.ends_with(".value_proj.weight"))
        {
            continue;
        }
        seen = true;
        if entry.dtype != DType::BF16 {
            return false;
        }
        let requantized = requant.as_deref().is_some_and(|spec| {
            spec.split(';')
                .filter(|rule| !rule.trim().is_empty())
                .filter_map(|rule| rule.split_once('='))
                .any(|(pattern, _)| entry.name.contains(pattern.trim()))
        });
        if requantized {
            return false;
        }
    }
    seen
}

/// The forward scratch plan one artifact is first admitted to, decided before
/// any placement or allocation and without touching the payload: the aggregate
/// (non-slab) route is optimistic (the artifact is assumed symmetric IU4), so
/// the plan is compact exactly when the device, flag and profile terms allow
/// it.  The proof is the GPU symmetry check in `Qwen4GpuForward::new_inner`;
/// a compact attach it refuses is re-planned Full by `attach_forward`.  A
/// manifest whose recipe is `FlashNextSlab` is an expert-slab artifact:
/// [`crate::weights`] replaces every trunk routed aggregate by its layer's
/// arena or refuses the manifest, so every routed trunk layer is a slab arena
/// (the MTP head's experts live in the separate MTP forward).  Every
/// real-model caller (the loader and the oracle examples) charges and attaches
/// exactly this mode.
pub fn source_forward_scratch_mode(
    gpu: &Gpu,
    _hfq: &HfqFile,
    config: &Qwen4Config,
    manifest: &Qwen4Manifest,
) -> Result<Qwen4ForwardScratchMode, Qwen4ArtifactError> {
    let ple_bf16 = ple_key_value_bf16(manifest);
    let slab = matches!(
        manifest.identity.recipe,
        crate::weights::Qwen4TierRecipe::FlashNextSlab { .. }
    );
    Ok(qwen4_forward_scratch_mode(
        gpu, config, true, slab, ple_bf16,
    ))
}

/// The geometry facts of a PLE table header that admission checks against the
/// canonical contract (a plain value so the checks are testable without
/// building a container).
#[derive(Clone, Debug, PartialEq, Eq)]
struct SidecarGeometry {
    rows_per_shard: u64,
    row_width: u64,
    padded_rows: u64,
    encoding: RowEncoding,
    encoded_row_bytes: u64,
    codebook_f32_bits: Option<[u32; 16]>,
    scale_grid_f32_bits: Option<[u32; 9]>,
}

impl SidecarGeometry {
    fn from_header(
        header: &hipfire_runtime::ple_hot_pack::PleHotPackHeader,
    ) -> Result<Self, Qwen4ArtifactError> {
        let table = header.table.as_ref().ok_or_else(|| {
            Qwen4ArtifactError::new(
                "qwen4: PLE rows sidecar geometry: the file is a hot-row pack, not a PLE table",
            )
        })?;
        Ok(Self {
            rows_per_shard: header.rows_per_shard,
            row_width: header.row_width,
            padded_rows: header.padded_rows,
            encoding: header.row_encoding,
            encoded_row_bytes: header.encoded_row_bytes,
            codebook_f32_bits: table.codebook_f32_bits,
            scale_grid_f32_bits: table.scale_grid_f32_bits,
        })
    }
}

/// The sidecar's geometry against the profile's PLE contract, and the row
/// encoding it declares.
///
/// A sidecar is accepted only at the exact geometry the profile's metadata
/// hashes into: `ple.shard_count` shards of `ple.shard_rows` rows of
/// `ple.row_width` values, summing to the metadata's padded rows (for
/// Flash-Next 128 x 2,500,012 x 160).  Q4LM must carry the exact LM4 codebook
/// and scale grid the decoder is compiled with (the decoder has no way to read
/// another), and Q8F16 carries neither.
fn validate_sidecar_geometry(
    geometry: &SidecarGeometry,
    ple: &Qwen4PleConfig,
    canonical: &PleHashMetadata,
) -> Result<RowEncoding, Qwen4ArtifactError> {
    let fail =
        |what: String| Qwen4ArtifactError::new(format!("qwen4: PLE rows sidecar geometry: {what}"));
    if geometry.rows_per_shard != ple.shard_rows as u64
        || geometry.row_width != ple.row_width as u64
    {
        return Err(fail(format!(
            "{} rows per shard x {} values, expected {} x {}",
            geometry.rows_per_shard, geometry.row_width, ple.shard_rows, ple.row_width
        )));
    }
    // Rows per shard times the shard count is the metadata's padded row
    // count, so equality here pins the shard count too.
    let declared_padded = (ple.shard_rows as u64).checked_mul(ple.shard_count as u64);
    if geometry.padded_rows != canonical.padded_rows()
        || Some(geometry.padded_rows) != declared_padded
    {
        return Err(fail(format!(
            "{} padded rows, PLE metadata has {} ({} shards of {})",
            geometry.padded_rows,
            canonical.padded_rows(),
            ple.shard_count,
            ple.shard_rows
        )));
    }
    let encoding = geometry.encoding;
    // FP8E4M3PT rows are a tier of the artifact's own qt=56 shards (admitted
    // by `validate_weight_geometry`): their scale lives in each shard's
    // header, which a sidecar table does not carry.
    if encoding == RowEncoding::Fp8PerTensor {
        return Err(fail(
            "FP8E4M3PT rows are read from the artifact's embedded qt=56 shards, not a sidecar \
             (a table carries no per-shard scale)"
                .to_string(),
        ));
    }
    if !matches!(encoding, RowEncoding::Q8F16 | RowEncoding::Q4Lm) {
        return Err(fail(format!(
            "row encoding {} is not Q8F16 or Q4LM",
            encoding.dtype_name()
        )));
    }
    let row_bytes = encoding.encoded_row_bytes(ple.row_width) as u64;
    if geometry.encoded_row_bytes != row_bytes {
        return Err(fail(format!(
            "{} rows are {} bytes, expected {row_bytes}",
            encoding.dtype_name(),
            geometry.encoded_row_bytes
        )));
    }
    let (book, grid) = match encoding {
        RowEncoding::Q4Lm => (Some(lm4_codebook_bits()), Some(lm4_scale_grid_bits())),
        _ => (None, None),
    };
    if geometry.codebook_f32_bits != book || geometry.scale_grid_f32_bits != grid {
        return Err(fail(format!(
            "{} codebook/scale grid do not match this build's Lloyd-Max tables",
            encoding.dtype_name()
        )));
    }
    Ok(encoding)
}

/// The PLE rows source of `hfq`: its attached sidecar (validated) or, with
/// none attached, its own embedded shard tensors.
fn admit_ple_rows(
    hfq: &HfqFile,
    ple: &Qwen4PleConfig,
    canonical: &PleHashMetadata,
) -> Result<Qwen4PleRows, Qwen4ArtifactError> {
    let Some(rows) = hfq.ple_rows() else {
        return Ok(Qwen4PleRows::Embedded);
    };
    let path = rows.path();
    let geometry = SidecarGeometry::from_header(rows.header())?;
    let encoding = validate_sidecar_geometry(&geometry, ple, canonical)?;
    let sidecar_rows = match rows.order() {
        PackOrder::Row => {
            let descriptors = rows.shard_descriptors().map_err(|error| {
                Qwen4ArtifactError::new(format!(
                    "qwen4: PLE rows sidecar {} descriptors could not be sealed: {error}",
                    path.display()
                ))
            })?;
            if descriptors.len() != ple.shard_count {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: PLE rows sidecar {} yields {} shard descriptors, expected {}",
                    path.display(),
                    descriptors.len(),
                    ple.shard_count
                )));
            }
            SidecarRows::Descriptors(descriptors)
        }
        PackOrder::Rank => SidecarRows::Table(rows.open_table_store().map_err(|error| {
            Qwen4ArtifactError::new(format!(
                "qwen4: PLE rows sidecar {} could not be opened: {error}",
                path.display()
            ))
        })?),
    };
    Ok(Qwen4PleRows::Sidecar {
        path: path.to_path_buf(),
        encoding,
        rows: sidecar_rows,
    })
}

impl Qwen4HfqmArtifact {
    /// The recipe the artifact was admitted under (selected by its metadata,
    /// see [`RECIPE_METADATA_KEY`]).  [`Self::manifest`] is already built under
    /// it; a consumer must take the manifest from here, never rebuild one from
    /// `config` with [`Qwen4Manifest::build`], which declares the default recipe
    /// of the profile and would misread a synthetic-packed artifact.
    pub fn recipe(&self) -> &Qwen4TierRecipe {
        &self.manifest.identity.recipe
    }

    /// The ordered shard descriptors of the PLE rows source (one per
    /// `config.ple.shard_count`): the row-order sidecar's, or the artifact's
    /// own shard tensors (`hfq` must be the file this receipt was admitted
    /// from).  A rank-order table has none: it is its own hot-row pack and is
    /// read through its store ([`Qwen4PleRows::rank_table`]).  A profile
    /// without PLE has no rows source and yields no descriptors.
    pub fn ple_row_descriptors(
        &self,
        hfq: &HfqFile,
    ) -> Result<Vec<SourceRangeDescriptor>, Qwen4ArtifactError> {
        match &self.ple_rows {
            Qwen4PleRows::Absent => Ok(Vec::new()),
            Qwen4PleRows::Sidecar {
                rows: SidecarRows::Descriptors(descriptors),
                ..
            } => Ok(descriptors.clone()),
            Qwen4PleRows::Sidecar {
                path,
                rows: SidecarRows::Table(_),
                ..
            } => Err(Qwen4ArtifactError::new(format!(
                "qwen4: PLE rows {} are a rank-order table; it has no row-order shard ranges",
                path.display()
            ))),
            Qwen4PleRows::Embedded => {
                let ple = self.config.ple.as_ref().ok_or_else(|| {
                    Qwen4ArtifactError::new("qwen4: embedded PLE rows without a PLE profile")
                })?;
                (0..ple.shard_count)
                    .map(|shard| {
                        let name = ple_shard_tensor_name(ple.layer, shard);
                        hfq.tensor_range(&name)
                            .map_err(|error| {
                                Qwen4ArtifactError::new(format!(
                                    "qwen4: PLE shard range {name} could not be sealed: {error}"
                                ))
                            })?
                            .ok_or_else(|| {
                                Qwen4ArtifactError::new(format!(
                                    "qwen4: PLE shard {name} is missing"
                                ))
                            })
                    })
                    .collect()
            }
        }
    }

    /// The file the PLE rows are read from and its length: the sidecar, or
    /// the model file itself (path as opened).  A hot-row pack is cut from
    /// and verified against exactly this file.
    pub fn ple_rows_source(&self, hfq: &HfqFile) -> Result<(PathBuf, u64), Qwen4ArtifactError> {
        match &self.ple_rows {
            Qwen4PleRows::Absent => Err(Qwen4ArtifactError::new(
                "qwen4: this profile has no PLE rows source",
            )),
            Qwen4PleRows::Embedded => {
                let file = self.source_identity.files.first().ok_or_else(|| {
                    Qwen4ArtifactError::new("qwen4: source identity has no backing file")
                })?;
                Ok((hfq.path().to_path_buf(), file.len))
            }
            Qwen4PleRows::Sidecar { path, rows, .. } => {
                let bytes = match rows {
                    SidecarRows::Descriptors(descriptors) => descriptors
                        .first()
                        .and_then(|descriptor| descriptor.source_identity().files.first())
                        .map(|file| file.len)
                        .ok_or_else(|| {
                            Qwen4ArtifactError::new("qwen4: PLE rows sidecar has no backing file")
                        })?,
                    SidecarRows::Table(_) => std::fs::metadata(path)
                        .map_err(|error| {
                            Qwen4ArtifactError::new(format!(
                                "qwen4: stat PLE rows {}: {error}",
                                path.display()
                            ))
                        })?
                        .len(),
                };
                Ok((path.clone(), bytes))
            }
        }
    }
}

fn read_i64_record(
    hfq: &HfqFile,
    name: &str,
    expected_values: usize,
) -> Result<Vec<i64>, Qwen4ArtifactError> {
    let expected_bytes = expected_values.checked_mul(8).ok_or_else(|| {
        Qwen4ArtifactError::new(format!(
            "qwen4: PLE record {name} byte count overflows usize"
        ))
    })?;
    if expected_bytes > MAX_PLE_METADATA_BYTES {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: PLE record {name} exceeds bounded read size {MAX_PLE_METADATA_BYTES}"
        )));
    }
    let info = hfq.find_tensor_info(name).ok_or_else(|| {
        Qwen4ArtifactError::new(format!("qwen4: PLE metadata record {name} is missing"))
    })?;
    if info.quant_type != QWEN4_QT_I64 || info.group_size != 0 || info.data_size != expected_bytes {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: PLE metadata record {name} has qt={}, group={}, bytes={}, expected qt54/group0/{}",
            info.quant_type, info.group_size, info.data_size, expected_bytes
        )));
    }
    let range = hfq
        .tensor_range(name)
        .map_err(|error| {
            Qwen4ArtifactError::new(format!(
                "qwen4: PLE metadata range {name} could not be sealed: {error}"
            ))
        })?
        .ok_or_else(|| {
            Qwen4ArtifactError::new(format!("qwen4: PLE metadata range {name} is missing"))
        })?;
    if range.length != expected_bytes as u64 {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: PLE metadata range {name} length={} expected {expected_bytes}",
            range.length
        )));
    }
    let mut bytes = vec![0u8; expected_bytes];
    range.read_exact(&mut bytes).map_err(|error| {
        Qwen4ArtifactError::new(format!(
            "qwen4: bounded PLE metadata read {name} failed: {error}"
        ))
    })?;
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("chunks_exact(8)")))
        .collect())
}

fn validate_ple_records(
    hfq: &HfqFile,
    ple: &Qwen4PleConfig,
    canonical: &PleHashMetadata,
) -> Result<(), Qwen4ArtifactError> {
    let [multipliers_name, offsets_name, vocab_sizes_name] = ple_record_names(ple.layer);
    let multipliers = read_i64_record(hfq, &multipliers_name, ple.multiplier_count)?;
    if multipliers.as_slice() != canonical.multipliers().as_slice() {
        return Err(Qwen4ArtifactError::new(
            "qwen4: PLE I64 multipliers payload does not match canonical metadata",
        ));
    }
    let offsets = read_i64_record(hfq, &offsets_name, ple.head_count)?;
    let expected_offsets = canonical
        .head_offsets()
        .iter()
        .map(|&value| i64::try_from(value).expect("canonical PLE offset fits i64"))
        .collect::<Vec<_>>();
    if offsets != expected_offsets {
        return Err(Qwen4ArtifactError::new(
            "qwen4: PLE I64 head_offsets payload does not match canonical metadata",
        ));
    }
    let vocab_sizes = read_i64_record(hfq, &vocab_sizes_name, ple.head_count)?;
    let expected_vocab_sizes = canonical
        .head_vocab_sizes()
        .iter()
        .map(|&value| i64::try_from(value).expect("canonical PLE vocab size fits i64"))
        .collect::<Vec<_>>();
    if vocab_sizes != expected_vocab_sizes {
        return Err(Qwen4ArtifactError::new(
            "qwen4: PLE I64 head_vocab_sizes payload does not match canonical metadata",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::{
        Qwen4ManifestIdentity, Qwen4TierRecipe, PLE_ROW_WIDTH, PLE_SHARD_COUNT, PLE_SHARD_ROWS,
    };
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
    use hipfire_runtime::ple_sidecar::ple_shard_name;
    use serde_json::json;
    use std::path::{Path, PathBuf};

    #[test]
    fn source_scratch_selection_is_optimistic_for_every_route() {
        use crate::gpu_forward::CompactScratchTerms;

        let mut cases = 0;
        for knob in [None, Some(false), Some(true)] {
            for bits in 0..64 {
                let terms = CompactScratchTerms {
                    knob,
                    exact_gfx1201: bits & 1 != 0,
                    grouped_gemm: bits & 2 != 0,
                    sym_iu4_requested: bits & 4 != 0,
                    source_slab: bits & 8 != 0,
                    ple_key_value_bf16: bits & 16 != 0,
                    flash_next_profile: bits & 32 != 0,
                    source_symmetric: true,
                };
                // The aggregate path selects with the artifact assumed
                // symmetric; the slab path ignores symmetry entirely.
                let optimistic = terms.mode();
                let pessimistic = CompactScratchTerms {
                    source_symmetric: false,
                    ..terms
                }
                .mode();
                if terms.source_slab {
                    assert_eq!(optimistic, pessimistic, "{terms:?}");
                } else if optimistic != Qwen4ForwardScratchMode::Full {
                    assert_eq!(pessimistic, Qwen4ForwardScratchMode::Full, "{terms:?}");
                }
                cases += 1;
            }
        }
        assert_eq!(cases, 192);
    }

    fn slab_metadata() -> String {
        json!({
            (expert_slabs::EXPERT_SLABS_METADATA_KEY): expert_slabs::ExpertSlabsV1::fn_v1().to_json_value()
        })
        .to_string()
    }

    fn container_refusal(
        fixture: &str,
        metadata: &str,
        tensors: &[HfqMemTensor],
    ) -> Qwen4ArtifactError {
        let path = fixture_path(fixture);
        write_hfqm_package_mem(&path, ARCH_ID, metadata, tensors).unwrap();
        let hfq = HfqFile::open(&path).unwrap();
        let error = admit_hfqm_artifact(&hfq).expect_err("CPU admission must refuse slabs");
        drop(hfq);
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(path.parent().unwrap()).unwrap();
        error
    }

    #[test]
    fn slab_container_invalid_table_refuses_before_config_admission() {
        assert!(matches!(
            container_refusal("slab-empty-table", &slab_metadata(), &[]),
            Qwen4ArtifactError::ExpertSlabTable(_)
        ));
    }

    #[test]
    fn slab_container_i4_without_metadata_refuses_each_new_qtype() {
        for quant_type in 131..=136 {
            let tensor = HfqMemTensor {
                name: "stray.weight".into(),
                quant_type,
                shape: vec![1, 256],
                group_size: 128,
                data: vec![],
            };
            assert!(matches!(
                container_refusal(&format!("slab-no-key-{quant_type}"), "{}", &[tensor]),
                Qwen4ArtifactError::ExpertSlabsMetadataRequired {
                    quant_type: found, ..
                } if found == quant_type
            ));
        }
    }

    #[test]
    fn slab_container_per_expert_mq4_name_without_metadata_refuses() {
        let tensor = HfqMemTensor {
            name: expert_slabs::expert_entry_name(0, 0, expert_slabs::ExpertProj::GateUp),
            quant_type: 44,
            shape: vec![1, 256],
            group_size: 256,
            data: vec![],
        };
        assert!(matches!(
            container_refusal("slab-name-no-key", "{}", &[tensor]),
            Qwen4ArtifactError::ExpertSlabsMetadataRequired { quant_type: 44, .. }
        ));
    }

    #[test]
    fn slab_container_malformed_metadata_is_a_typed_refusal() {
        let metadata = json!({(expert_slabs::EXPERT_SLABS_METADATA_KEY): {"version": 2}});
        assert!(matches!(
            container_refusal("slab-bad-key", &metadata.to_string(), &[]),
            Qwen4ArtifactError::ExpertSlabsMetadata(ExpertSlabsError::Mismatch {
                field: "version",
                ..
            })
        ));
    }

    #[test]
    fn slab_valid_full_inventory_refuses_unavailable_kernels_with_counts() {
        use expert_slabs::ExpertFormat;
        let headers = crate::expert_slab_table::full_fn_headers(|layer, expert| {
            match (layer * 512 + expert) % 4 {
                0 => ExpertFormat::Mq3I4,
                1 => ExpertFormat::Mq2I4,
                2 => ExpertFormat::E8pI4,
                _ => ExpertFormat::Mq4Xts,
            }
        });
        let error = refuse_expert_slabs(&slab_metadata(), &headers).unwrap_err();
        assert_eq!(
            error,
            Qwen4ArtifactError::ExpertSlabKernelsUnavailable {
                w3: 6144,
                w2: 6144,
                e8p: 6144,
                mq4: 6144,
            }
        );
        assert!(error.to_string().contains("not available on this device"));
    }

    /// The sealed route admits per (arch, cold codec): gfx1100 MQ4 + E8P,
    /// gfx1151 MQ4 + W3 + E8P (E8P via SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED), gfx1201 MQ4 + W3 + E8P,
    /// W2 nowhere, one cold codec per layer (layers may differ), and every other architecture
    /// (or none) keeps the unchanged fail-closed refusal.
    #[test]
    fn slab_plan_admits_only_the_arch_cold_codec_pair() {
        use crate::expert_arena::{CODEC_E8P, CODEC_MQ4, CODEC_W3};
        use expert_slabs::ExpertFormat::{self, E8pI4, Mq2I4, Mq3I4, Mq4Xts};
        let headers = |cold: ExpertFormat| {
            crate::expert_slab_table::full_fn_headers(move |layer, expert| {
                if (layer * 512 + expert) % 2 == 0 {
                    cold
                } else {
                    Mq4Xts
                }
            })
        };
        // Cold codec of one layer, MQ4 on the odd experts.
        for (arch, cold, cold_codec) in [
            ("gfx1201", Mq3I4, CODEC_W3),
            ("gfx1201", E8pI4, CODEC_E8P),
            ("gfx1100", E8pI4, CODEC_E8P),
            ("gfx1151", Mq3I4, CODEC_W3),
        ] {
            assert!(slab_route_admitted(arch));
            let plan = expert_slab_plan(&slab_metadata(), &headers(cold), Some(arch))
                .unwrap()
                .expect("mixture admitted");
            for layer in plan.layers() {
                let table = layer.codec_table().unwrap();
                assert!(
                    table.iter().all(|c| *c == CODEC_MQ4 || *c == cold_codec),
                    "{arch}"
                );
                assert!(
                    table.contains(&CODEC_MQ4) && table.contains(&cold_codec),
                    "{arch}"
                );
                assert_eq!(layer.cold_codec(arch), Ok(cold_codec));
            }
        }
        // W3 on gfx1100 is an arch refusal; W2 is refused everywhere (typed).
        let error =
            expert_slab_plan(&slab_metadata(), &headers(Mq3I4), Some("gfx1100")).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("no sealed-route kernels on gfx1100"),
            "{error}"
        );
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            assert!(matches!(
                expert_slab_plan(&slab_metadata(), &headers(Mq2I4), Some(arch)),
                Err(Qwen4ArtifactError::ExpertSlabKernelsUnavailable { .. })
            ));
        }
        // MQ4-only: admitted on all three, no cold experts.
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            let plan = expert_slab_plan(&slab_metadata(), &headers(Mq4Xts), Some(arch))
                .unwrap()
                .expect("MQ4-only admitted");
            assert!(plan
                .layers()
                .iter()
                .all(|layer| layer.codec_counts() == (512, 0, 0)));
        }
        // One layer with both W3 and E8P is refused; layers choosing different cold codecs are not.
        let mixed_layer = |layer_a: usize| {
            crate::expert_slab_table::full_fn_headers(move |layer, expert| match expert % 3 {
                0 => Mq4Xts,
                1 => Mq3I4,
                _ if layer == layer_a => E8pI4,
                _ => Mq3I4,
            })
        };
        let error =
            expert_slab_plan(&slab_metadata(), &mixed_layer(0), Some("gfx1201")).unwrap_err();
        assert!(error.to_string().contains("two cold codecs"), "{error}");
        assert!(expert_slab_plan(&slab_metadata(), &mixed_layer(0), Some("gfx1100")).is_err());
        let alternating = crate::expert_slab_table::full_fn_headers(|layer, expert| {
            match (layer % 2, expert % 2) {
                (_, 0) => Mq4Xts,
                (0, _) => Mq3I4,
                _ => E8pI4,
            }
        });
        let plan = expert_slab_plan(&slab_metadata(), &alternating, Some("gfx1201"))
            .unwrap()
            .expect("per-layer cold codecs admitted");
        assert_eq!(plan.layers()[0].cold_codec("gfx1201"), Ok(CODEC_W3));
        assert_eq!(plan.layers()[1].cold_codec("gfx1201"), Ok(CODEC_E8P));
        assert!(expert_slab_plan(&slab_metadata(), &alternating, Some("gfx1100")).is_err());
        // gfx1151 admits MQ4 + E8P exactly when SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED (else an
        // arena arch refusal); a W3 + E8P layer is refused either way (the arch refusal, or the
        // two-cold-codecs refusal when admitted).
        let e8p_on_gfx1151 = expert_slab_plan(&slab_metadata(), &headers(E8pI4), Some("gfx1151"));
        assert_eq!(
            e8p_on_gfx1151.is_ok(),
            SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED
        );
        if !SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED {
            let error = e8p_on_gfx1151.unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("no sealed-route kernels on gfx1151"),
                "{error}"
            );
        }
        assert!(expert_slab_plan(&slab_metadata(), &mixed_layer(0), Some("gfx1151")).is_err());
        assert_eq!(
            expert_slab_plan(&slab_metadata(), &alternating, Some("gfx1151")).is_ok(),
            SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED
        );
        for arch in [Some("gfx942"), Some("gfx1200"), None] {
            assert!(!arch.is_some_and(slab_route_admitted));
            assert!(matches!(
                expert_slab_plan(&slab_metadata(), &headers(E8pI4), arch),
                Err(Qwen4ArtifactError::ExpertSlabKernelsUnavailable { .. })
            ));
        }
    }

    #[test]
    fn slab_pruned_inventory_preserves_cpu_refusal_and_gpu_route() {
        use expert_slabs::ExpertFormat;
        let keep = crate::ExpertKeep::new(512, vec![(16..32).collect(); 48], None, 10).unwrap();
        let mut headers = crate::expert_slab_table::full_fn_headers(|_, _| ExpertFormat::Mq3I4);
        headers.retain(|info| {
            parse_expert_entry_name(&info.name)
                .map_or(true, |(layer, expert, _)| keep.is_kept(layer, expert))
        });
        let mut cursor = 4096;
        for info in &mut headers {
            info.data_offset = cursor;
            cursor += info.data_size;
        }
        let mut metadata: serde_json::Value = serde_json::from_str(&slab_metadata()).unwrap();
        metadata["qwen4.expert_keep"] = keep.to_metadata_value();
        let metadata = metadata.to_string();
        let expected = Qwen4ArtifactError::ExpertSlabKernelsUnavailable {
            w3: keep.total_kept(),
            w2: 0,
            e8p: 0,
            mq4: 0,
        };
        assert_eq!(
            refuse_expert_slabs(&metadata, &headers).unwrap_err(),
            expected
        );
        assert_eq!(
            expert_slab_plan(&metadata, &headers, None).unwrap_err(),
            expected
        );
        // gfx1100 serves MQ4 + E8P only: pruned W3 experts are an arch refusal, not a count refusal.
        let error = expert_slab_plan(&metadata, &headers, Some("gfx1100")).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("no sealed-route kernels on gfx1100"),
            "{error}"
        );
        let table = build_expert_slab_table_with_keep(
            &expert_slabs::ExpertSlabsV1::fn_v1(),
            &headers,
            Some(&keep),
        )
        .unwrap();
        for expert in [0, 15, 32, 511] {
            let slab = table.slab(0, expert).unwrap();
            assert_eq!(slab.gate_up.len, 0);
            assert_eq!(slab.down.len, 0);
        }
        assert_ne!(table.slab(0, 16).unwrap().gate_up.len, 0);
        assert_ne!(table.slab(0, 31).unwrap().down.len, 0);
        for arch in ["gfx1201", "gfx1151"] {
            let plan = expert_slab_plan(&metadata, &headers, Some(arch))
                .unwrap()
                .unwrap();
            assert_eq!(
                plan.physical_bytes(),
                (keep.total_kept() * ExpertFormat::Mq3I4.slab_bytes()) as u64,
            );
        }
        // An absent (zero-byte) expert carries the MQ4 codec and never names a cold
        // codec: a pruned E8P layer is admitted on gfx1100 (which has no W3 route).
        let mut e8p = crate::expert_slab_table::full_fn_headers(|_, _| ExpertFormat::E8pI4);
        e8p.retain(|info| {
            parse_expert_entry_name(&info.name)
                .map_or(true, |(layer, expert, _)| keep.is_kept(layer, expert))
        });
        let mut cursor = 4096;
        for info in &mut e8p {
            info.data_offset = cursor;
            cursor += info.data_size;
        }
        for arch in ["gfx1100", "gfx1201"] {
            let plan = expert_slab_plan(&metadata, &e8p, Some(arch))
                .unwrap()
                .unwrap();
            for layer in plan.layers() {
                assert_eq!(layer.cold_codec(arch), Ok(crate::expert_arena::CODEC_E8P));
                assert_eq!(layer.codec_counts(), (0, 0, 16));
                let table = layer.codec_table().unwrap();
                assert!(table.iter().all(|c| *c == crate::expert_arena::CODEC_MQ4
                    || *c == crate::expert_arena::CODEC_E8P));
            }
        }
    }

    #[test]
    fn slab_route_admission_follows_each_arch_receipts() {
        assert_eq!(slab_route_admitted("gfx1201"), SLAB_ROUTE_RECEIPTS_LANDED);
        assert_eq!(
            slab_route_admitted("gfx1151"),
            SLAB_ROUTE_GFX1151_RECEIPTS_LANDED
        );
        assert_eq!(slab_route_admitted("gfx1100"), SLAB_ROUTE_RECEIPTS_LANDED);
        for other in ["gfx1200", "gfx1101", "gfx1151x", "GFX1151", "gfx942", ""] {
            assert!(!slab_route_admitted(other), "{other:?}");
        }
    }

    /// `slab_route_admitted` is the central MQ4 capability AND the arch's own receipts flag.
    #[test]
    fn slab_route_admission_is_the_mq4_matrix_gate_and_the_arch_receipts_flag() {
        for arch in ["gfx1100", "gfx1151", "gfx1201", "gfx1200", "gfx942", ""] {
            let flag = match arch {
                "gfx1151" => SLAB_ROUTE_GFX1151_RECEIPTS_LANDED,
                "gfx1100" | "gfx1201" => SLAB_ROUTE_RECEIPTS_LANDED,
                _ => false,
            };
            assert_eq!(
                slab_route_admitted(arch),
                rdna_compute::expert_capabilities::slab_serves(
                    arch,
                    rdna_compute::expert_capabilities::QT_MQ4_XTS
                ) && flag,
                "{arch:?}"
            );
        }
    }

    #[test]
    fn slab_recognition_leaves_legacy_headers_and_invalid_json_to_existing_checks() {
        refuse_expert_slabs(&flash_next_json(), &flash_next_headers()).unwrap();
        refuse_expert_slabs("invalid JSON", &[]).unwrap();
        refuse_expert_slabs("[]", &[]).unwrap();
    }

    #[test]
    fn weight_geometry_explicitly_refuses_each_planar_i4_qtype() {
        let manifest = Qwen4Manifest::build(&crate::config::compact_test_config()).unwrap();
        let entry = &manifest.weights[0];
        for quant_type in 131..=136 {
            let info = HfqTensorInfo {
                name: entry.name.clone(),
                quant_type,
                shape: entry.logical_shape.iter().map(|&d| d as u32).collect(),
                group_size: 128,
                data_offset: 4096,
                data_size: 0,
            };
            assert!(matches!(
                validate_weight_geometry(entry, &info),
                Err(Qwen4ArtifactError::ExpertSlabQuantTypeUnavailable {
                    quant_type: found, ..
                }) if found == quant_type
            ));
        }
    }

    const PLE_MULTIPLIERS_NAME: &str =
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers";
    const PLE_OFFSETS_NAME: &str =
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets";
    const PLE_VOCAB_SIZES_NAME: &str =
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes";

    /// The Flash-Next PLE geometry as the typed config declares it.
    fn canonical_ple() -> Qwen4PleConfig {
        crate::config::compact_test_config()
            .ple
            .expect("Flash-Next declares PLE")
    }

    fn i64_bytes(values: &[i64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn metadata_json() -> String {
        let canonical = PleHashMetadata::qwen4();
        json!({
            "qwen4_ple": {
                "version": 1,
                "multipliers": canonical.multipliers(),
                "head_vocab_sizes": canonical.head_vocab_sizes(),
                "head_offsets": canonical.head_offsets(),
                "padded_rows": canonical.padded_rows(),
            }
        })
        .to_string()
    }

    fn fixture_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-qwen4-artifact-{name}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("artifact.hfq")
    }

    fn write_ple_fixture(path: &Path, multipliers: &[i64]) {
        let canonical = PleHashMetadata::qwen4();
        let tensors = vec![
            HfqMemTensor {
                name: PLE_MULTIPLIERS_NAME.to_string(),
                quant_type: QWEN4_QT_I64,
                shape: vec![3],
                group_size: 0,
                data: i64_bytes(multipliers),
            },
            HfqMemTensor {
                name: PLE_OFFSETS_NAME.to_string(),
                quant_type: QWEN4_QT_I64,
                shape: vec![16],
                group_size: 0,
                data: i64_bytes(
                    &canonical
                        .head_offsets()
                        .iter()
                        .map(|&value| value as i64)
                        .collect::<Vec<_>>(),
                ),
            },
            HfqMemTensor {
                name: PLE_VOCAB_SIZES_NAME.to_string(),
                quant_type: QWEN4_QT_I64,
                shape: vec![16],
                group_size: 0,
                data: i64_bytes(
                    &canonical
                        .head_vocab_sizes()
                        .iter()
                        .map(|&value| value as i64)
                        .collect::<Vec<_>>(),
                ),
            },
        ];
        write_hfqm_package_mem(path, ARCH_ID, &metadata_json(), &tensors).unwrap();
    }

    #[test]
    fn bounded_ple_i64_validation_rejects_payload_mismatch() {
        let path = fixture_path("mismatch");
        let mut multipliers = PleHashMetadata::qwen4().multipliers().to_vec();
        multipliers[1] += 1;
        write_ple_fixture(&path, &multipliers);
        let hfq = HfqFile::open(&path).unwrap();
        let error =
            validate_ple_records(&hfq, &canonical_ple(), &PleHashMetadata::qwen4()).unwrap_err();
        assert!(
            error.to_string().contains("multipliers"),
            "payload mismatch must identify the signed I64 record: {error}"
        );
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(path.parent().unwrap());
    }

    #[test]
    fn bounded_ple_i64_validation_accepts_canonical_payloads() {
        let path = fixture_path("canonical");
        write_ple_fixture(&path, PleHashMetadata::qwen4().multipliers());
        let hfq = HfqFile::open(&path).unwrap();
        validate_ple_records(&hfq, &canonical_ple(), &PleHashMetadata::qwen4()).unwrap();
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(path.parent().unwrap());
    }

    /// A routed expert block is rank-3 on disk and rows-by-K in the format, so
    /// the loader's extent must flatten the leading dimensions exactly as the
    /// producer's row count does — and refuse a K the decode kernels assert on.
    #[test]
    fn mfp4e8soa_extent_flattens_expert_blocks_and_requires_fwht_alignment() {
        let soa = |shape: &[u32], name: &str| {
            quantized_extent(
                shape,
                name,
                DType::MFP4G32E8SOA,
                QWEN4_MFP4G32E8SOA_K_ALIGNMENT,
            )
        };
        let (stride, extent) = soa(&[512, 1280, 2560], "gate_up").unwrap();
        assert_eq!(stride, 16 + 80 + 80 * 16);
        assert_eq!(extent, 512 * 1280 * stride);

        // The same rows as a flat rank-2 matrix describe the same payload: the
        // flattening is exactly the leading-dimension product.
        let (_flat_stride, flat) = soa(&[512 * 1280, 2560], "gate_up_flat").unwrap();
        assert_eq!(flat, extent);

        // `moe_intermediate_size = 640` is not one FWHT-256 segment, so the
        // routed down projection cannot be carried by this format at all: the
        // boundary must say so by name instead of admitting a payload whose
        // decode would drop the unaligned tail.
        let error = soa(&[512, 2560, 640], "down").unwrap_err();
        assert!(
            error.to_string().contains("multiple of 256"),
            "a K below one FWHT-256 segment must be refused by name, got: {error}"
        );
        let error = soa(&[512, 1280, 128], "gate_up").unwrap_err();
        assert!(
            error.to_string().contains("multiple of 256"),
            "a K below one FWHT-256 segment must be refused by name, got: {error}"
        );
    }

    /// GPU-free admission probe over a real artifact.
    ///
    ///   HIPFIRE_PROBE_MODEL=<artifact.hfq> cargo test -p hipfire-arch-qwen4 --lib \
    ///     admits_a_sealed_artifact_with_its_declared_ple_tier --ignored --nocapture
    ///
    /// Reads headers and the index only. Prints the admitted PLE tier, its row
    /// stride and the external byte count, so a sealed artifact written before
    /// the Q8F16 tier and one written with it can both be checked without a GPU.
    #[test]
    #[ignore = "artifact admission probe; set HIPFIRE_PROBE_MODEL to an HFQ artifact"]
    fn admits_a_sealed_artifact_with_its_declared_ple_tier() {
        let Ok(model) = hipfire_config::developer_var("HIPFIRE_PROBE_MODEL") else {
            println!("admit-probe: skipped, HIPFIRE_PROBE_MODEL is unset");
            return;
        };
        let hfq = hipfire_runtime::hfq::HfqFile::open(Path::new(&model)).expect("open artifact");
        let admitted = admit_hfqm_artifact(&hfq).expect("artifact must be admitted");
        let mut shards = 0usize;
        let mut stride = 0usize;
        let mut external_bytes = 0u64;
        let mut tier = None;
        for entry in admitted.manifest.weights.iter() {
            if !entry.name.contains(".ngram_embedding.shard_") {
                continue;
            }
            shards += 1;
            if let hipfire_runtime::weight_manifest::WeightResidency::ExternalRows {
                row_bytes,
                ..
            } = entry.residency
            {
                stride = row_bytes;
            }
            tier = Some(entry.dtype);
            let info = hfq
                .find_tensor_info(&entry.name)
                .expect("PLE shard index entry");
            external_bytes += info.data_size as u64;
        }
        println!(
            "admit-probe {model}: admitted, {shards} PLE shards, declared {tier:?} tier, \
             stride={stride}, external_bytes={external_bytes}"
        );
        assert_eq!(shards, 128, "every PLE shard must be admitted");
        assert!(stride == 320 || stride == 170, "declared stride {stride}");
    }

    #[test]
    fn external_row_geometry_admits_each_declared_ple_tier() {
        // A PLE shard declaration admits BF16 rows and Q8F16 rows; the artifact
        // decides which one it carries, and the geometry gate must answer both
        // without privileging either.
        let mut entry = WeightEntry::layer(
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
            1,
            vec![4, 160],
            DType::Q8_0,
            ShardPolicy::Replicate,
        )
        .external_rows(170, 4);
        entry.dtype_constraint =
            hipfire_runtime::weight_manifest::DTypeConstraint::source_from_sources(vec![
                DType::Q8_0,
                DType::BF16,
            ]);

        let info =
            |quant_type: u8, group_size: u32, data_size: u64| hipfire_runtime::hfq::HfqTensorInfo {
                name: entry.name.clone(),
                quant_type,
                shape: vec![4, 160],
                group_size,
                data_offset: 0,
                data_size: data_size.try_into().unwrap(),
            };

        validate_weight_geometry(&entry, &info(QWEN4_QT_BF16, 0, 4 * 320))
            .expect("a sealed BF16 shard must be admitted");
        validate_weight_geometry(&entry, &info(QWEN4_QT_Q8F16, 32, 4 * 170))
            .expect("a Q8F16 shard must be admitted");

        // A stride that is no tier for this width, a mismatched index extent,
        // and a tag the declaration excludes are each refused by name.
        let mut nonsense = entry.clone();
        nonsense.residency =
            hipfire_runtime::weight_manifest::WeightResidency::external_rows(640, 4);
        let error = validate_weight_geometry(&nonsense, &info(QWEN4_QT_BF16, 0, 4 * 320))
            .expect_err("a stride that is no row tier must be refused");
        assert!(error.to_string().contains("not a row tier"), "{error}");

        let error = validate_weight_geometry(&entry, &info(QWEN4_QT_BF16, 0, 4 * 320 - 1))
            .expect_err("an extent that disagrees with the index must be refused");
        assert!(error.to_string().contains("data_size"), "{error}");

        let mut bf16_only = entry.clone();
        bf16_only.dtype_constraint =
            hipfire_runtime::weight_manifest::DTypeConstraint::source_exact(DType::BF16);
        let error = validate_weight_geometry(&bf16_only, &info(QWEN4_QT_Q8F16, 32, 4 * 170))
            .expect_err("a tier the declaration excludes must be refused");
        assert!(
            error.to_string().contains("not an admitted source"),
            "{error}"
        );
    }

    fn metadata_tensors() -> Vec<crate::weights::MetadataTensor> {
        [
            (PLE_MULTIPLIERS_NAME, 3),
            (PLE_OFFSETS_NAME, 16),
            (PLE_VOCAB_SIZES_NAME, 16),
        ]
        .into_iter()
        .map(|(name, values)| crate::weights::MetadataTensor {
            name: name.to_string(),
            shape: vec![values],
            source_dtype: "I64".to_string(),
        })
        .collect()
    }

    /// A manifest that declares only the three PLE metadata records, i.e.
    /// what a sidecar-fed manifest reduces to once the 128 shards are trimmed
    /// from a base whose other tensors are not under test.
    fn trimmed_manifest() -> Qwen4Manifest {
        Qwen4Manifest {
            expert_keep: None,
            identity: Qwen4ManifestIdentity {
                profile_id: "qwen3.8-flash-next".to_string(),
                developer_only: false,
                equation_source: "pinned-upstream",
                ffn_kind: "moe",
                recipe: Qwen4TierRecipe::FlashNextPacked {
                    trunk_tier: DType::Q8_0,
                    mtp_source: false,
                },
            },
            weights: Vec::new(),
            state: Vec::new(),
            metadata: metadata_tensors(),
        }
    }

    /// Metadata JSON with (`split`) or without the split-base marker.
    fn marked_metadata_json(split: bool) -> String {
        let mut value: serde_json::Value = serde_json::from_str(&metadata_json()).unwrap();
        if split {
            value[hipfire_runtime::ple_sidecar::PLE_BASE_MARKER_KEY] = json!({
                "version": 1,
                "location": "sidecar",
                "shard_count": PLE_SHARD_COUNT,
                "rows_per_shard": PLE_SHARD_ROWS,
                "row_width": PLE_ROW_WIDTH,
                "q8_rows_sha256": "0".repeat(64),
            });
        }
        value.to_string()
    }

    /// The three metadata records plus `shards` tiny Q8F16 shard tensors under
    /// the canonical shard names (their geometry is not what the sidecar
    /// seams under test look at).
    fn write_base_fixture(path: &Path, split: bool, shards: usize) {
        let canonical = PleHashMetadata::qwen4();
        let i64s = |values: Vec<i64>| i64_bytes(&values);
        let mut tensors = (0..shards)
            .map(|shard| HfqMemTensor {
                name: ple_shard_name(shard),
                quant_type: QWEN4_QT_Q8F16,
                shape: vec![1, 32],
                group_size: 0,
                data: vec![0u8; 34],
            })
            .collect::<Vec<_>>();
        tensors.extend([
            HfqMemTensor {
                name: PLE_MULTIPLIERS_NAME.to_string(),
                quant_type: QWEN4_QT_I64,
                shape: vec![3],
                group_size: 0,
                data: i64s(canonical.multipliers().to_vec()),
            },
            HfqMemTensor {
                name: PLE_OFFSETS_NAME.to_string(),
                quant_type: QWEN4_QT_I64,
                shape: vec![16],
                group_size: 0,
                data: i64s(canonical.head_offsets().iter().map(|&v| v as i64).collect()),
            },
            HfqMemTensor {
                name: PLE_VOCAB_SIZES_NAME.to_string(),
                quant_type: QWEN4_QT_I64,
                shape: vec![16],
                group_size: 0,
                data: i64s(
                    canonical
                        .head_vocab_sizes()
                        .iter()
                        .map(|&v| v as i64)
                        .collect(),
                ),
            },
        ]);
        write_hfqm_package_mem(path, ARCH_ID, &marked_metadata_json(split), &tensors).unwrap();
    }

    fn with_base_fixture<T>(
        name: &str,
        split: bool,
        shards: usize,
        check: impl FnOnce(&HfqFile) -> T,
    ) -> T {
        let path = fixture_path(name);
        write_base_fixture(&path, split, shards);
        let hfq = HfqFile::open(&path).unwrap();
        let result = check(&hfq);
        drop(hfq);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(path.parent().unwrap());
        result
    }

    #[test]
    fn split_base_without_sidecar_names_the_sidecar_and_the_setting() {
        with_base_fixture("split-bare", true, 0, |hfq| {
            let error = validate_ple_shard_presence(hfq, false, Some(&canonical_ple()))
                .unwrap_err()
                .to_string();
            for needle in [".ple-q4", ".ple-q8", "memory.ple_format", "split base"] {
                assert!(error.contains(needle), "{needle} missing from: {error}");
            }
        });
    }

    #[test]
    fn plain_base_without_shards_keeps_the_missing_tensor_diagnostic() {
        // No marker, no shards, no sidecar: not a split base; the inventory
        // check (not this one) refuses it as before.
        with_base_fixture("plain-bare", false, 0, |hfq| {
            assert_eq!(
                validate_ple_shard_presence(hfq, false, Some(&canonical_ple())).unwrap(),
                0
            );
        });
    }

    #[test]
    fn sidecar_requires_all_shards_or_a_split_marker() {
        with_base_fixture("split-sidecar", true, 0, |hfq| {
            assert_eq!(
                validate_ple_shard_presence(hfq, true, Some(&canonical_ple())).unwrap(),
                0
            );
        });
        with_base_fixture("mono-sidecar", false, PLE_SHARD_COUNT, |hfq| {
            assert_eq!(
                validate_ple_shard_presence(hfq, true, Some(&canonical_ple())).unwrap(),
                PLE_SHARD_COUNT
            );
            // Without a sidecar the same file is the grandfathered monolith.
            assert_eq!(
                validate_ple_shard_presence(hfq, false, Some(&canonical_ple())).unwrap(),
                0
            );
        });
        with_base_fixture("partial-sidecar", false, 5, |hfq| {
            let error = validate_ple_shard_presence(hfq, true, Some(&canonical_ple()))
                .unwrap_err()
                .to_string();
            assert!(error.contains("5 of 128"), "{error}");
        });
        with_base_fixture("bare-sidecar", false, 0, |hfq| {
            let error = validate_ple_shard_presence(hfq, true, Some(&canonical_ple()))
                .unwrap_err()
                .to_string();
            assert!(error.contains("hipfire_ple_rows"), "{error}");
        });
    }

    #[test]
    fn inventory_accepts_a_split_base_and_a_shadowed_monolith_under_a_sidecar() {
        let manifest = trimmed_manifest();
        assert_eq!(manifest.source_tensor_count(), 3);
        // Split base: the index is exactly the trimmed manifest.
        with_base_fixture("inv-split", true, 0, |hfq| {
            let shadowed = validate_ple_shard_presence(hfq, true, Some(&canonical_ple())).unwrap();
            assert_eq!(
                validate_manifest_inventory(hfq.tensors(), &manifest, shadowed, PLE_SHARD_COUNT)
                    .unwrap(),
                3
            );
        });
        // Monolith + sidecar: the 128 shard tensors are in the index, shadowed
        // and ignored.
        with_base_fixture("inv-mono", false, PLE_SHARD_COUNT, |hfq| {
            let shadowed = validate_ple_shard_presence(hfq, true, Some(&canonical_ple())).unwrap();
            assert_eq!(shadowed, PLE_SHARD_COUNT);
            assert_eq!(
                validate_manifest_inventory(hfq.tensors(), &manifest, shadowed, PLE_SHARD_COUNT)
                    .unwrap(),
                3
            );
            // The same index against the trimmed manifest without the shadow
            // allowance is an inventory mismatch, never a silent pass.
            let error = validate_manifest_inventory(hfq.tensors(), &manifest, 0, PLE_SHARD_COUNT)
                .unwrap_err();
            assert!(error.to_string().contains("131 tensors"), "{error}");
        });
    }

    #[test]
    fn inventory_still_refuses_an_undeclared_tensor_next_to_a_sidecar() {
        let path = fixture_path("inv-foreign");
        let canonical = PleHashMetadata::qwen4();
        let record = |name: &str, shape: u32, values: Vec<i64>| HfqMemTensor {
            name: name.to_string(),
            quant_type: QWEN4_QT_I64,
            shape: vec![shape],
            group_size: 0,
            data: i64_bytes(&values),
        };
        let tensors = vec![
            HfqMemTensor {
                name: "model.unknown.weight".to_string(),
                quant_type: QWEN4_QT_BF16,
                shape: vec![1],
                group_size: 0,
                data: vec![0, 0],
            },
            record(PLE_MULTIPLIERS_NAME, 3, canonical.multipliers().to_vec()),
            record(
                PLE_OFFSETS_NAME,
                16,
                canonical.head_offsets().iter().map(|&v| v as i64).collect(),
            ),
            record(
                PLE_VOCAB_SIZES_NAME,
                16,
                canonical
                    .head_vocab_sizes()
                    .iter()
                    .map(|&v| v as i64)
                    .collect(),
            ),
        ];
        write_hfqm_package_mem(&path, ARCH_ID, &marked_metadata_json(true), &tensors).unwrap();
        let hfq = HfqFile::open(&path).unwrap();
        let error =
            validate_manifest_inventory(hfq.tensors(), &trimmed_manifest(), 0, 0).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("4 tensors but manifest declares 3"),
            "{error}"
        );
        drop(hfq);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(path.parent().unwrap());
    }

    fn sidecar_geometry(encoding: RowEncoding) -> SidecarGeometry {
        let q4 = encoding == RowEncoding::Q4Lm;
        SidecarGeometry {
            rows_per_shard: PLE_SHARD_ROWS as u64,
            row_width: PLE_ROW_WIDTH as u64,
            padded_rows: PleHashMetadata::qwen4().padded_rows(),
            encoding,
            encoded_row_bytes: encoding.encoded_row_bytes(PLE_ROW_WIDTH) as u64,
            codebook_f32_bits: q4.then(lm4_codebook_bits),
            scale_grid_f32_bits: q4.then(lm4_scale_grid_bits),
        }
    }

    #[test]
    fn sidecar_geometry_admits_the_canonical_q8_and_q4_tables() {
        let canonical = PleHashMetadata::qwen4();
        for encoding in [RowEncoding::Q8F16, RowEncoding::Q4Lm] {
            assert_eq!(
                validate_sidecar_geometry(
                    &sidecar_geometry(encoding),
                    &canonical_ple(),
                    &canonical
                )
                .unwrap(),
                encoding
            );
        }
        assert_eq!(RowEncoding::Q4Lm.encoded_row_bytes(PLE_ROW_WIDTH), 90);
    }

    #[test]
    fn sidecar_geometry_refuses_each_deviation() {
        let canonical = PleHashMetadata::qwen4();
        let refused = |geometry: SidecarGeometry, needle: &str| {
            let error = validate_sidecar_geometry(&geometry, &canonical_ple(), &canonical)
                .expect_err(needle)
                .to_string();
            assert!(error.contains(needle), "{needle} missing from: {error}");
        };
        let q8 = || sidecar_geometry(RowEncoding::Q8F16);
        let q4 = || sidecar_geometry(RowEncoding::Q4Lm);
        let mut geometry = q8();
        geometry.rows_per_shard += 1;
        refused(geometry, "rows per shard");
        let mut geometry = q8();
        geometry.row_width = 128;
        refused(geometry, "rows per shard");
        let mut geometry = q8();
        geometry.padded_rows += 1;
        refused(geometry, "padded rows");
        let mut geometry = q8();
        geometry.padded_rows /= 2;
        refused(geometry, "padded rows");
        let mut geometry = q8();
        geometry.encoding = RowEncoding::Bf16;
        refused(geometry, "not Q8F16 or Q4LM");
        let mut geometry = q8();
        geometry.encoded_row_bytes = 171;
        refused(geometry, "bytes, expected 170");
        let mut geometry = q4();
        geometry.encoded_row_bytes = 170;
        refused(geometry, "bytes, expected 90");
        let mut geometry = q4();
        geometry.codebook_f32_bits = None;
        refused(geometry, "Lloyd-Max tables");
        let mut geometry = q4();
        geometry.codebook_f32_bits.as_mut().unwrap()[3] ^= 1;
        refused(geometry, "Lloyd-Max tables");
        let mut geometry = q4();
        geometry.scale_grid_f32_bits.as_mut().unwrap()[0] ^= 1;
        refused(geometry, "Lloyd-Max tables");
        // A Q8F16 table must not carry a codebook.
        let mut geometry = q8();
        geometry.codebook_f32_bits = Some(lm4_codebook_bits());
        refused(geometry, "Lloyd-Max tables");
    }

    #[test]
    fn ple_fingerprint_helper_matches_the_metadata_fingerprint() {
        let expected = PleHashMetadata::qwen4()
            .fingerprint()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            hipfire_runtime::ple_sidecar::ple_fingerprint_from_metadata(&metadata_json()).unwrap(),
            expected
        );
    }

    /// Index records of `manifest`'s exact inventory, with every byte extent
    /// from a literal row formula (Q8F16 34 B per 32 weights, MQ4G256V2 136 B
    /// per 256, MQ4G128V2 68 B per 128, MQ6G256V2 200 B per 256, BF16 2 B, I64
    /// 8 B) rather than from the admission code under test.
    fn oracle_headers(manifest: &Qwen4Manifest) -> Vec<HfqTensorInfo> {
        let mut headers = Vec::new();
        for entry in &manifest.weights {
            if matches!(entry.policy, ShardPolicy::Tied { .. }) {
                continue;
            }
            let (&k, lead) = entry.logical_shape.split_last().expect("non-empty shape");
            let rows: usize = lead.iter().product();
            let (quant_type, group_size, bytes) = match entry.dtype {
                DType::BF16 => (16u8, 0u32, rows * k * 2),
                DType::Q8_0 => (3, 32, rows * (k / 32) * 34),
                DType::MQ4G256V2 => (44, 256, rows * (k / 256) * 136),
                DType::MQ4G128V2 => (53, 128, rows * (k / 128) * 68),
                DType::MQ6G256V2 => (47, 256, rows * (k / 256) * 200),
                // qt=55: `E` blocks of round_up(R*K + 2*ceil(R/128)*ceil(K/128), 256).
                DType::FP8E4M3B128 => {
                    let [experts, r] = lead[..] else {
                        panic!("FP8E4M3B128 oracle needs [E, R, K]")
                    };
                    let block = r * k + 2 * r.div_ceil(128) * k.div_ceil(128);
                    (55, 128, experts * block.div_ceil(256) * 256)
                }
                // qt=56: 16-byte scale header, then one code per value.
                DType::FP8E4M3PT => (56, 0, 16 + rows * k),
                other => panic!("oracle has no byte formula for {other:?}"),
            };
            headers.push(HfqTensorInfo {
                name: entry.name.clone(),
                quant_type,
                shape: entry.logical_shape.iter().map(|&d| d as u32).collect(),
                group_size,
                data_offset: 0,
                data_size: bytes,
            });
        }
        for record in &manifest.metadata {
            headers.push(HfqTensorInfo {
                name: record.name.clone(),
                quant_type: 54,
                shape: record.shape.iter().map(|&d| d as u32).collect(),
                group_size: 0,
                data_offset: 0,
                data_size: record.shape.iter().product::<usize>() * 8,
            });
        }
        headers
    }

    fn flash_next_json() -> String {
        crate::config::compact_test_config().developer_metadata_json()
    }

    fn flash_next_headers() -> Vec<HfqTensorInfo> {
        let config = crate::config::compact_test_config();
        oracle_headers(&Qwen4Manifest::build(&config).expect("pinned manifest"))
    }

    #[test]
    fn flash_next_header_inventory_admits_with_exact_extents() {
        let headers = flash_next_headers();
        assert_eq!(headers.len(), 1325, "1,658 source records minus 333 vision");
        let receipt = admit_inventory_headers(ARCH_ID, &flash_next_json(), &headers)
            .expect("the exact pinned inventory is admitted");
        assert_eq!(receipt.source_tensor_count, 1325);
        assert_eq!(receipt.ple, Some(PleHashMetadata::qwen4()));
        assert_eq!(receipt.manifest.identity.profile_id, "qwen3.8-flash-next");
        assert!(!receipt.manifest.identity.developer_only);
        // Literal spot checks of the extents the oracle produced.
        let extent = |name: &str| {
            headers
                .iter()
                .find(|header| header.name == name)
                .unwrap_or_else(|| panic!("missing {name}"))
                .data_size
        };
        assert_eq!(
            extent("model.language_model.embed_tokens.weight"),
            248320 * (2560 / 32) * 34
        );
        assert_eq!(extent("lm_head.weight"), 248320 * (2560 / 256) * 200);
        assert_eq!(
            extent("model.language_model.layers.0.mlp.experts.gate_up_proj"),
            512 * 1280 * (2560 / 256) * 136
        );
        assert_eq!(
            extent("model.language_model.layers.0.mlp.experts.down_proj"),
            512 * 2560 * (640 / 128) * 68
        );
        assert_eq!(
            extent(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight"
            ),
            2_500_012 * 170
        );
    }

    #[test]
    fn flash_next_header_inventory_refuses_each_deviation_before_any_allocation() {
        let json = flash_next_json();
        let refused = |mutate: &dyn Fn(&mut Vec<HfqTensorInfo>), needle: &str| {
            let mut headers = flash_next_headers();
            mutate(&mut headers);
            let error = admit_inventory_headers(ARCH_ID, &json, &headers)
                .expect_err(needle)
                .to_string();
            assert!(error.contains(needle), "{needle} missing from: {error}");
        };
        let position = |headers: &[HfqTensorInfo], name: &str| {
            headers
                .iter()
                .position(|header| header.name == name)
                .unwrap_or_else(|| panic!("missing {name}"))
        };
        let q = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        let experts = "model.language_model.layers.3.mlp.experts.gate_up_proj";
        refused(
            &|headers| {
                headers.remove(position(headers, q));
            },
            "tensors but manifest declares",
        );
        refused(
            &|headers| {
                let index = position(headers, q);
                headers[index].name = "model.language_model.layers.0.linear_attn.qkv.weight".into();
            },
            "is not declared by the manifest",
        );
        refused(
            &|headers| {
                let index = position(headers, q);
                headers[index].shape[0] += 1;
            },
            "has shape",
        );
        refused(
            &|headers| {
                let index = position(headers, q);
                headers[index].data_size += 1;
            },
            "data_size",
        );
        refused(
            &|headers| {
                let index = position(headers, q);
                headers[index].group_size = 64;
            },
            "group_size",
        );
        refused(
            &|headers| {
                let index = position(headers, experts);
                headers[index].quant_type = 99;
            },
            "unsupported HFQM quant_type=99",
        );
        // gate_up (K=2560) retagged MQ4G128V2 is byte-for-byte the same extent
        // as MQ4G256V2 (20 x 68 == 10 x 136) and its source set admits both,
        // so that is a legitimately equal payload, not a deviation.  down
        // (K=640) is not a whole number of 256-groups: retagging it qt44 must
        // be refused by the aligned-K rule.
        refused(
            &|headers| {
                let index = position(
                    headers,
                    "model.language_model.layers.3.mlp.experts.down_proj",
                );
                headers[index].quant_type = QWEN4_QT_MQ4G256V2;
                headers[index].group_size = 256;
            },
            "multiple of 256",
        );
        refused(
            &|headers| {
                // A vision record in place of a declared tensor: the count is
                // right, so only the text-only boundary can refuse it.
                let index = position(headers, q);
                headers[index].name = "model.visual.patch_embed.weight".into();
            },
            "vision tensor",
        );
        refused(
            &|headers| {
                let index = position(headers, "lm_head.weight");
                headers[index].data_size = usize::MAX;
            },
            "data_size",
        );
        // A different identity is refused before the inventory is looked at.
        let error = admit_inventory_headers(ARCH_ID + 1, &json, &flash_next_headers())
            .expect_err("wrong arch id")
            .to_string();
        assert!(error.contains("arch_id"), "{error}");
        let error = admit_inventory_headers(ARCH_ID, "{\"config\": {}}", &flash_next_headers())
            .expect_err("wrong identity")
            .to_string();
        assert!(error.contains("config admission failed"), "{error}");
    }

    fn admit_with_head_policy(
        json: &str,
        headers: &[HfqTensorInfo],
        drop_head: bool,
    ) -> Result<Qwen4InventoryReceipt, Qwen4ArtifactError> {
        let config = Qwen4Config::from_metadata_json(json).expect("pinned config");
        admit_inventory(ARCH_ID, config, json, headers, drop_head)
    }

    /// The shipped packed artifact (head records in the file) admits under the trunk
    /// shard's head policy as the headless profile: packed recipe, no `mtp.*` in the
    /// manifest, the head records ignored, every other record still held to the exact
    /// inventory. Without the policy the same file admits unchanged, with the head.
    #[test]
    fn packed_artifact_admits_headless_when_the_shard_drops_the_head() {
        let headers = flash_next_headers();
        let head_records = headers
            .iter()
            .filter(|header| header.name.starts_with("mtp."))
            .count();
        assert!(head_records > 0);
        let kept = admit_with_head_policy(&flash_next_json(), &headers, false).unwrap();
        assert!(kept.config.profile == crate::config::Qwen4Profile::FlashNext);
        assert!(kept.config.mtp.is_some());
        assert_eq!(kept.source_tensor_count, headers.len());

        let headless = admit_with_head_policy(&flash_next_json(), &headers, true)
            .expect("the head-dropped inventory is admitted");
        assert_eq!(
            headless.config.profile,
            crate::config::Qwen4Profile::FlashNextHeadless
        );
        assert!(headless.config.mtp.is_none());
        assert_eq!(headless.config.mtp_num_hidden_layers, 0);
        assert!(matches!(
            headless.manifest.identity.recipe,
            Qwen4TierRecipe::FlashNextPacked { .. }
        ));
        assert_eq!(
            headless.manifest.identity.recipe,
            kept.manifest.identity.recipe
        );
        assert_eq!(
            headless.manifest.identity.profile_id,
            "qwen3.8-flash-next-headless"
        );
        assert!(headless
            .manifest
            .weights
            .iter()
            .all(|entry| !entry.name.starts_with("mtp.")));
        assert_eq!(headless.ple, kept.ple);
        // the manifest declares none of the head's records; they stay in the file, ignored
        assert_eq!(headless.source_tensor_count, headers.len() - head_records);

        // a file that never carried the head admits the same headless manifest
        let no_head: Vec<_> = headers
            .iter()
            .filter(|header| !header.name.starts_with("mtp."))
            .cloned()
            .collect();
        let bare = admit_with_head_policy(&flash_next_json(), &no_head, true).unwrap();
        assert_eq!(bare.manifest, headless.manifest);
        // ... but the head-bearing profile refuses it, and a missing / reshaped trunk record
        // is still refused under the head policy
        assert!(admit_with_head_policy(&flash_next_json(), &no_head, false).is_err());
        let mut missing = headers.clone();
        let at = missing
            .iter()
            .position(|header| header.name == "lm_head.weight")
            .unwrap();
        missing.remove(at);
        assert!(admit_with_head_policy(&flash_next_json(), &missing, true).is_err());
        let mut reshaped = headers.clone();
        let at = reshaped
            .iter()
            .position(|header| header.name == "lm_head.weight")
            .unwrap();
        reshaped[at].shape[0] += 1;
        assert!(admit_with_head_policy(&flash_next_json(), &reshaped, true).is_err());
        // an undeclared extra record that is not the head is refused
        let mut extra = headers.clone();
        let mut stray = extra[0].clone();
        stray.name = "model.language_model.stray.weight".into();
        extra.push(stray);
        assert!(admit_with_head_policy(&flash_next_json(), &extra, true).is_err());
        // the stock-FP8 recipe id is its own (already head-free) profile: the policy is moot
        let stock = stock_fp8_json("stock-fp8");
        let stock_headers = stock_fp8_headers();
        let under_policy = admit_with_head_policy(&stock, &stock_headers, true).unwrap();
        assert_eq!(
            under_policy.config.profile,
            crate::config::Qwen4Profile::FlashNextArOnly
        );
        assert_eq!(
            under_policy.manifest.identity.recipe,
            Qwen4TierRecipe::StockFp8
        );
    }

    fn stock_fp8_json(id: &str) -> String {
        let mut value: serde_json::Value = serde_json::from_str(&flash_next_json()).unwrap();
        value[RECIPE_METADATA_KEY] = json!({"version": 3, "id": id});
        value.to_string()
    }

    fn stock_fp8_headers() -> Vec<HfqTensorInfo> {
        let config = crate::config::compact_test_config().into_ar_only().unwrap();
        oracle_headers(
            &Qwen4Manifest::build_with_recipe(&config, Qwen4TierRecipe::StockFp8)
                .expect("stock-FP8 manifest"),
        )
    }

    /// A stock-FP8 inventory (`qwen4_recipe.id = "stock-fp8"`) admits under
    /// the AR-only profile with qt=55 experts, qt=56 PLE rows, a BF16 trunk
    /// and no MTP records, at the contract's exact extents.
    #[test]
    fn stock_fp8_header_inventory_admits_ar_only_with_exact_extents() {
        let headers = stock_fp8_headers();
        let receipt = admit_inventory_headers(ARCH_ID, &stock_fp8_json("stock-fp8"), &headers)
            .expect("the stock-FP8 inventory is admitted");
        assert_eq!(receipt.manifest.identity.recipe, Qwen4TierRecipe::StockFp8);
        assert!(receipt.config.profile.is_ar_only() && receipt.config.profile.is_flash_next());
        assert!(receipt.config.mtp.is_none());
        assert_eq!(receipt.config.mtp_num_hidden_layers, 0);
        assert_eq!(
            receipt.manifest.identity.profile_id,
            "qwen3.8-flash-next-ar"
        );
        assert_eq!(receipt.ple, Some(PleHashMetadata::qwen4()));
        assert_eq!(receipt.source_tensor_count, headers.len());
        assert!(headers
            .iter()
            .all(|header| !header.name.starts_with("mtp.")));
        let flash = flash_next_headers();
        let mtp = flash
            .iter()
            .filter(|header| header.name.starts_with("mtp."))
            .count();
        assert!(mtp > 0);
        assert_eq!(headers.len(), flash.len() - mtp);
        let header = |name: &str| {
            headers
                .iter()
                .find(|header| header.name == name)
                .unwrap_or_else(|| panic!("missing {name}"))
                .clone()
        };
        let gate_up = header("model.language_model.layers.0.mlp.experts.gate_up_proj");
        assert_eq!((gate_up.quant_type, gate_up.group_size), (55, 128));
        assert_eq!(gate_up.data_size, 512 * 3_277_312);
        let down = header("model.language_model.layers.0.mlp.experts.down_proj");
        assert_eq!((down.quant_type, down.data_size), (55, 512 * 1_638_656));
        let shard = header(
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight",
        );
        assert_eq!((shard.quant_type, shard.group_size), (56, 0));
        assert_eq!(shard.data_size, 16 + 2_500_012 * 160);
        for name in [
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.3.self_attn.q_proj.weight",
            "model.language_model.layers.0.mlp.gate.weight",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
        ] {
            assert_eq!(header(name).quant_type, 16, "{name}");
        }
        for entry in receipt.manifest.external_entries() {
            assert_eq!(entry.dtype, DType::FP8E4M3PT);
            assert!(matches!(
                entry.residency,
                WeightResidency::ExternalRows { row_bytes: 160, .. }
            ));
        }
    }

    /// The stock-FP8 inventory admits exactly its own representation: a
    /// packed or BF16 stand-in for an FP8 tensor, a stray MTP record, a wrong
    /// extent or group, and the inventory under any other recipe id are all
    /// refused.
    #[test]
    fn stock_fp8_header_inventory_refuses_each_deviation() {
        let json = stock_fp8_json("stock-fp8");
        let refused = |json: &str, mutate: &dyn Fn(&mut Vec<HfqTensorInfo>), needle: &str| {
            let mut headers = stock_fp8_headers();
            mutate(&mut headers);
            let error = admit_inventory_headers(ARCH_ID, json, &headers)
                .expect_err(needle)
                .to_string();
            assert!(error.contains(needle), "{needle} missing from: {error}");
        };
        let position = |headers: &[HfqTensorInfo], name: &str| {
            headers
                .iter()
                .position(|header| header.name == name)
                .unwrap_or_else(|| panic!("missing {name}"))
        };
        let gate_up = "model.language_model.layers.3.mlp.experts.gate_up_proj";
        let shard =
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_5.weight";
        refused(
            &json,
            &|headers| {
                let index = position(headers, gate_up);
                headers[index].quant_type = QWEN4_QT_MQ4G256V2;
                headers[index].group_size = 256;
                headers[index].data_size = 512 * 1280 * 10 * 136;
            },
            "declared FP8E4M3B128",
        );
        refused(
            &json,
            &|headers| {
                let index = position(headers, gate_up);
                headers[index].data_size -= 256;
            },
            "data_size",
        );
        refused(
            &json,
            &|headers| {
                let index = position(headers, gate_up);
                headers[index].group_size = 256;
            },
            "qt55 group_size",
        );
        refused(
            &json,
            &|headers| {
                let index = position(headers, shard);
                headers[index].quant_type = QWEN4_QT_Q8F16;
                headers[index].group_size = 32;
                headers[index].data_size = 2_500_012 * 170;
            },
            "FP8E4M3PT",
        );
        refused(
            &json,
            &|headers| {
                let index = position(headers, shard);
                headers[index].data_size = 2_500_012 * 160;
            },
            "data_size",
        );
        refused(
            &json,
            &|headers| {
                let index = position(headers, "lm_head.weight");
                headers[index].quant_type = QWEN4_QT_MQ6G256V2;
                headers[index].group_size = 256;
                headers[index].data_size = 248320 * 10 * 200;
            },
            "native BF16 tensor lm_head.weight",
        );
        refused(
            &json,
            &|headers| {
                headers.push(HfqTensorInfo {
                    name: "mtp.fc_hidden.weight".into(),
                    quant_type: QWEN4_QT_BF16,
                    shape: vec![2560, 2560],
                    group_size: 0,
                    data_offset: 0,
                    data_size: 2560 * 2560 * 2,
                });
            },
            "manifest declares",
        );
        // The same bytes under the packed (no id) or an unknown recipe id.
        refused(&flash_next_json(), &|_| {}, "manifest declares");
        refused(
            &stock_fp8_json("stock-fp9"),
            &|_| {},
            "not a Flash-Next recipe",
        );
    }

    #[test]
    fn extent_arithmetic_reports_overflow_instead_of_panicking() {
        let overflowing = [u32::MAX; 4];
        let error = quantized_extent(&overflowing, "x", DType::Q8_0, 1)
            .expect_err("row count overflow")
            .to_string();
        assert!(error.contains("overflows usize"), "{error}");
        let error = shape_elements(&[u32::MAX; 5], "x")
            .expect_err("element count overflow")
            .to_string();
        assert!(error.contains("overflows usize"), "{error}");
        let error = shape_elements(&[4, 0], "x")
            .expect_err("zero dim")
            .to_string();
        assert!(error.contains("zero-sized"), "{error}");
    }

    #[cfg(feature = "qwen4-dev-profiles")]
    mod developer_profiles {
        use super::*;
        use crate::config::{Qwen4Config, Qwen4FfnConfig};
        use crate::weights::{Qwen4TierRecipe, RECIPE_METADATA_KEY};

        fn dense() -> Qwen4FfnConfig {
            Qwen4FfnConfig::Dense {
                intermediate_size: 17_408,
            }
        }

        fn small_moe(shared: Option<usize>) -> Qwen4FfnConfig {
            Qwen4FfnConfig::Moe {
                num_experts: 8,
                top_k: 2,
                intermediate_size: 64,
                shared_intermediate_size: shared,
                norm_topk_prob: true,
            }
        }

        fn admit(config: &Qwen4Config) -> Result<Qwen4InventoryReceipt, Qwen4ArtifactError> {
            let manifest = Qwen4Manifest::build(config).expect("developer manifest");
            admit_developer_inventory_headers(
                ARCH_ID,
                &config.developer_metadata_json(),
                &oracle_headers(&manifest),
            )
        }

        #[test]
        fn flash_next_is_admitted_identically_by_the_developer_path() {
            let production =
                admit_inventory_headers(ARCH_ID, &flash_next_json(), &flash_next_headers())
                    .expect("production admission");
            let developer = admit_developer_inventory_headers(
                ARCH_ID,
                &flash_next_json(),
                &flash_next_headers(),
            )
            .expect("developer admission");
            assert_eq!(
                developer.source_tensor_count,
                production.source_tensor_count
            );
            assert_eq!(developer.manifest, production.manifest);
            assert_eq!(developer.ple, production.ple);
        }

        #[test]
        fn dense_64_layer_hidden_5120_header_inventories_admit_with_and_without_ple_mtp() {
            for (ple, mtp) in [(false, false), (true, false), (false, true), (true, true)] {
                let config = crate::config::synthetic_test_config(dense(), 5120, 64, ple, mtp);
                let receipt = admit(&config).expect("exact dense inventory");
                assert_eq!(receipt.manifest.identity.ffn_kind, "dense");
                assert!(receipt.manifest.identity.developer_only);
                assert_eq!(receipt.ple.is_some(), ple);
                assert_eq!(
                    receipt.source_tensor_count,
                    receipt.manifest.source_tensor_count()
                );
                // The dry admission is honest about what it proves: roles are
                // declared, nothing is bound.
                assert!(receipt
                    .manifest
                    .weights
                    .iter()
                    .any(|entry| entry.name.ends_with(".mlp.down_proj.weight")));
            }
        }

        fn packed_manifest(config: &Qwen4Config) -> Qwen4Manifest {
            Qwen4Manifest::build_with_recipe(config, Qwen4TierRecipe::SyntheticPacked)
                .expect("synthetic packed manifest")
        }

        fn dtype_of(manifest: &Qwen4Manifest, name: &str) -> DType {
            manifest
                .weights
                .iter()
                .find(|entry| entry.name == name)
                .unwrap_or_else(|| panic!("manifest declares {name}"))
                .dtype
        }

        fn metadata_with_recipe(config: &Qwen4Config, id: Option<serde_json::Value>) -> String {
            let mut value: serde_json::Value =
                serde_json::from_str(&config.developer_metadata_json()).unwrap();
            if let Some(id) = id {
                value[RECIPE_METADATA_KEY] = serde_json::json!({ "id": id });
            }
            value.to_string()
        }

        /// The packed recipe is selected by `qwen4_recipe.id` and by nothing
        /// else: the same packed inventory is refused without the id, admitted
        /// with it, and every other id spelling is refused.
        #[test]
        fn packed_recipe_admits_only_under_its_explicit_metadata_id() {
            let config = crate::config::synthetic_test_config(dense(), 5120, 4, true, true);
            let manifest = packed_manifest(&config);
            assert_eq!(manifest.identity.recipe, Qwen4TierRecipe::SyntheticPacked);
            assert!(manifest
                .weights
                .iter()
                .any(|entry| entry.dtype != DType::BF16));
            let headers = oracle_headers(&manifest);

            let error = admit_developer_inventory_headers(
                ARCH_ID,
                &metadata_with_recipe(&config, None),
                &headers,
            )
            .expect_err("packed tags without the recipe id are refused");
            assert!(error.to_string().contains("tensor"), "{error}");

            let receipt = admit_developer_inventory_headers(
                ARCH_ID,
                &metadata_with_recipe(
                    &config,
                    Some(serde_json::json!(
                        crate::weights::SYNTHETIC_PACKED_RECIPE_ID
                    )),
                ),
                &headers,
            )
            .expect("packed inventory under its recipe id");
            assert_eq!(receipt.manifest, manifest);

            let source_exact = admit_developer_inventory_headers(
                ARCH_ID,
                &metadata_with_recipe(
                    &config,
                    Some(serde_json::json!(
                        crate::weights::SYNTHETIC_SOURCE_BF16_RECIPE_ID
                    )),
                ),
                &oracle_headers(&Qwen4Manifest::build(&config).unwrap()),
            )
            .expect("explicit source-exact id");
            assert_eq!(
                source_exact.manifest.identity.recipe,
                Qwen4TierRecipe::SyntheticSourceBf16
            );

            for bad in [
                serde_json::json!("flash-next"),
                serde_json::json!("Synthetic-Packed"),
                serde_json::json!(7),
            ] {
                let error = admit_developer_inventory_headers(
                    ARCH_ID,
                    &metadata_with_recipe(&config, Some(bad)),
                    &headers,
                )
                .expect_err("unknown recipe id")
                .to_string();
                assert!(error.contains(RECIPE_METADATA_KEY), "{error}");
            }
        }

        /// Flash-Next has exactly one recipe: a recipe id on its metadata is
        /// refused instead of being ignored.
        #[test]
        fn flash_next_refuses_a_recipe_id() {
            let mut value: serde_json::Value = serde_json::from_str(&flash_next_json()).unwrap();
            value[RECIPE_METADATA_KEY] = serde_json::json!({
                "id": crate::weights::SYNTHETIC_PACKED_RECIPE_ID
            });
            let error = admit_developer_inventory_headers(
                ARCH_ID,
                &value.to_string(),
                &flash_next_headers(),
            )
            .expect_err("Flash-Next must not take a recipe selector")
            .to_string();
            assert!(error.contains("exactly one recipe"), "{error}");
        }

        /// Each role packs at its own reduction width, and the routed gate/up
        /// and down matrices pack together or not at all.
        #[test]
        fn packed_recipe_decides_per_role_and_packs_routed_experts_as_a_pair() {
            let moe = |intermediate: usize| Qwen4FfnConfig::Moe {
                num_experts: 4,
                top_k: 2,
                intermediate_size: intermediate,
                shared_intermediate_size: Some(intermediate),
                norm_topk_prob: true,
            };
            let layer = |role: &str| format!("model.language_model.layers.0.mlp.{role}");
            // hidden 256, expert width 128: both routed matrices pack.
            let config = crate::config::synthetic_test_config(moe(128), 256, 4, true, true);
            let manifest = packed_manifest(&config);
            assert_eq!(
                dtype_of(&manifest, &layer("experts.gate_up_proj")),
                DType::MQ4G256V2
            );
            assert_eq!(
                dtype_of(&manifest, &layer("experts.down_proj")),
                DType::MQ4G128V2
            );
            assert_eq!(
                dtype_of(&manifest, "model.language_model.embed_tokens.weight"),
                DType::Q8_0
            );
            assert_eq!(dtype_of(&manifest, "lm_head.weight"), DType::MQ6G256V2);
            // The router, shared expert and hyper-connection mixers stay BF16.
            assert_eq!(dtype_of(&manifest, &layer("gate.weight")), DType::BF16);
            assert_eq!(
                dtype_of(&manifest, &layer("shared_expert.gate_proj.weight")),
                DType::BF16
            );
            assert_eq!(
                dtype_of(
                    &manifest,
                    "model.language_model.hyper_connection_mixer.hc_norm.weight"
                ),
                DType::BF16
            );
            assert_eq!(dtype_of(&manifest, "mtp.fc_embedding.weight"), DType::BF16);
            // Expert width 64 is not a MQ4G128V2 group: gate/up must not pack alone.
            let config = crate::config::synthetic_test_config(moe(64), 256, 4, true, true);
            let manifest = packed_manifest(&config);
            assert_eq!(
                dtype_of(&manifest, &layer("experts.gate_up_proj")),
                DType::BF16
            );
            assert_eq!(
                dtype_of(&manifest, &layer("experts.down_proj")),
                DType::BF16
            );
            // Hidden 128 is not a MQ4G256V2 group: neither packs, down included.
            let config = crate::config::synthetic_test_config(moe(128), 128, 4, true, true);
            let manifest = packed_manifest(&config);
            assert_eq!(
                dtype_of(&manifest, &layer("experts.gate_up_proj")),
                DType::BF16
            );
            assert_eq!(
                dtype_of(&manifest, &layer("experts.down_proj")),
                DType::BF16
            );
            // A dense FFN packs gate/up where K tiles Q8F16 and keeps a narrow down BF16.
            let config = crate::config::synthetic_test_config(
                Qwen4FfnConfig::Dense {
                    intermediate_size: 8,
                },
                256,
                4,
                false,
                false,
            );
            let manifest = packed_manifest(&config);
            assert_eq!(dtype_of(&manifest, &layer("gate_proj.weight")), DType::Q8_0);
            assert_eq!(dtype_of(&manifest, &layer("up_proj.weight")), DType::Q8_0);
            assert_eq!(dtype_of(&manifest, &layer("down_proj.weight")), DType::BF16);
            // Hidden 16: nothing tiles any packed format.
            let config = crate::config::synthetic_test_config(moe(8), 16, 4, true, true);
            let manifest = packed_manifest(&config);
            assert!(
                manifest
                    .weights
                    .iter()
                    .all(|entry| entry.dtype == DType::BF16),
                "no role of a width-16 profile can pack"
            );
        }

        #[test]
        fn small_moe_header_inventories_admit() {
            for shared in [Some(64usize), None] {
                let config =
                    crate::config::synthetic_test_config(small_moe(shared), 256, 4, true, true);
                let receipt = admit(&config).expect("exact small-MoE inventory");
                assert_eq!(receipt.manifest.identity.ffn_kind, "moe");
                assert!(receipt.ple.is_some());
            }
        }

        #[test]
        fn developer_inventories_refuse_missing_extra_and_mis_sized_tensors() {
            let config = crate::config::synthetic_test_config(dense(), 5120, 64, true, true);
            let manifest = Qwen4Manifest::build(&config).expect("manifest");
            let json = config.developer_metadata_json();
            let refuse = |mutate: &dyn Fn(&mut Vec<HfqTensorInfo>), needle: &str| {
                let mut headers = oracle_headers(&manifest);
                mutate(&mut headers);
                let error = admit_developer_inventory_headers(ARCH_ID, &json, &headers)
                    .expect_err(needle)
                    .to_string();
                assert!(error.contains(needle), "{needle} missing from: {error}");
            };
            let down = "model.language_model.layers.7.mlp.down_proj.weight";
            let find = |headers: &[HfqTensorInfo], name: &str| {
                headers.iter().position(|h| h.name == name).expect(name)
            };
            refuse(
                &|h| {
                    h.remove(find(h, down));
                },
                "tensors but manifest declares",
            );
            refuse(
                &|h| {
                    let index = find(h, down);
                    h[index].data_size -= 2;
                },
                "data_size",
            );
            refuse(
                &|h| {
                    let index = find(h, down);
                    h[index].quant_type = 99;
                },
                "unsupported HFQM quant_type=99",
            );
            // A packed tag on a synthetic role is refused: nothing but BF16 is
            // admitted until a role format is executable.
            refuse(
                &|h| {
                    let index = find(h, down);
                    h[index].quant_type = QWEN4_QT_Q8F16;
                    h[index].group_size = 32;
                    h[index].data_size = 5120 * (17_408 / 32) * 34;
                },
                "must use HFQM quant_type=16",
            );
            refuse(
                &|h| {
                    h.push(HfqTensorInfo {
                        name: "model.language_model.layers.0.mlp.gate.weight".into(),
                        quant_type: QWEN4_QT_BF16,
                        shape: vec![8, 5120],
                        group_size: 0,
                        data_offset: 0,
                        data_size: 8 * 5120 * 2,
                    });
                },
                "tensors but manifest declares",
            );
        }

        #[test]
        fn pleless_profile_refuses_a_ple_metadata_object_and_pled_one_requires_it() {
            let no_ple = crate::config::synthetic_test_config(dense(), 5120, 64, false, false);
            let headers = oracle_headers(&Qwen4Manifest::build(&no_ple).expect("manifest"));
            let mut value: serde_json::Value =
                serde_json::from_str(&no_ple.developer_metadata_json()).unwrap();
            value["qwen4_ple"] = serde_json::from_str::<serde_json::Value>(&metadata_json())
                .unwrap()["qwen4_ple"]
                .clone();
            let error = admit_developer_inventory_headers(ARCH_ID, &value.to_string(), &headers)
                .expect_err("a PLE-less profile must not carry PLE metadata")
                .to_string();
            assert!(error.contains("declares no PLE"), "{error}");

            let with_ple = crate::config::synthetic_test_config(dense(), 5120, 64, true, false);
            let headers = oracle_headers(&Qwen4Manifest::build(&with_ple).expect("manifest"));
            let mut value: serde_json::Value =
                serde_json::from_str(&with_ple.developer_metadata_json()).unwrap();
            value.as_object_mut().unwrap().remove("qwen4_ple");
            let error = admit_developer_inventory_headers(ARCH_ID, &value.to_string(), &headers)
                .expect_err("a PLE profile needs its hash metadata")
                .to_string();
            assert!(error.contains("PLE metadata admission failed"), "{error}");
        }

        /// Real container, real admission: a tiny PLE-less/MTP-less synthetic
        /// dense artifact is admitted through `admit_hfqm_artifact` when the
        /// feature is compiled in, and the absent PLE is typed absence.
        #[test]
        fn tiny_dense_container_without_ple_admits_with_absent_ple_rows() {
            let config = crate::config::synthetic_test_config(dense(), 256, 4, false, false);
            let manifest = Qwen4Manifest::build(&config).expect("manifest");
            let tensors = oracle_headers(&manifest)
                .into_iter()
                .map(|header| HfqMemTensor {
                    name: header.name,
                    quant_type: header.quant_type,
                    shape: header.shape,
                    group_size: header.group_size,
                    data: vec![0u8; header.data_size],
                })
                .collect::<Vec<_>>();
            let path = fixture_path("tiny-dense");
            write_hfqm_package_mem(&path, ARCH_ID, &config.developer_metadata_json(), &tensors)
                .unwrap();
            let hfq = HfqFile::open(&path).unwrap();
            let admitted = admit_hfqm_artifact(&hfq).expect("developer container admitted");
            assert!(matches!(admitted.ple_rows, Qwen4PleRows::Absent));
            assert!(admitted.ple.is_none());
            assert!(admitted.ple_row_descriptors(&hfq).unwrap().is_empty());
            assert!(admitted.ple_rows_source(&hfq).is_err());
            assert_eq!(admitted.placements.len(), admitted.manifest.weights.len());
            drop(hfq);
            let _ = std::fs::remove_file(&path);
            let _ = std::fs::remove_dir(path.parent().unwrap());
        }
    }

    /// Without the explicit feature a synthetic developer artifact never
    /// reaches the inventory: the production parser refuses its marker, for
    /// the header-only path and for a real container alike.
    #[cfg(not(feature = "qwen4-dev-profiles"))]
    #[test]
    fn default_build_refuses_a_synthetic_developer_artifact() {
        use crate::config::Qwen4FfnConfig;
        let config = crate::config::synthetic_test_config(
            Qwen4FfnConfig::Dense {
                intermediate_size: 17_408,
            },
            256,
            4,
            false,
            false,
        );
        let json = config.developer_metadata_json();
        assert!(
            json.contains("hipfire_profile"),
            "fixture must carry the marker"
        );
        let headers = oracle_headers(&Qwen4Manifest::build(&config).expect("manifest"));
        let error = admit_inventory_headers(ARCH_ID, &json, &headers)
            .expect_err("production header admission must refuse the marker")
            .to_string();
        assert!(error.contains("config admission failed"), "{error}");

        let path = fixture_path("default-refuses-synthetic");
        write_hfqm_package_mem(&path, ARCH_ID, &json, &[]).unwrap();
        let hfq = HfqFile::open(&path).unwrap();
        let error = admit_hfqm_artifact(&hfq)
            .expect_err("production container admission must refuse the marker")
            .to_string();
        assert!(error.contains("config admission failed"), "{error}");
        drop(hfq);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(path.parent().unwrap());
    }

    /// With the feature a marker is accepted (and then judged by its exact
    /// inventory), but a marker that is not the explicit developer marker, or
    /// a synthetic config smuggled without one, is still refused by the config
    /// parser, never inferred from geometry.
    #[cfg(feature = "qwen4-dev-profiles")]
    #[test]
    fn feature_build_judges_a_synthetic_container_by_inventory_not_by_marker() {
        use crate::config::Qwen4FfnConfig;
        let config = crate::config::synthetic_test_config(
            Qwen4FfnConfig::Dense {
                intermediate_size: 17_408,
            },
            256,
            4,
            false,
            false,
        );
        let path = fixture_path("feature-empty-synthetic");
        write_hfqm_package_mem(&path, ARCH_ID, &config.developer_metadata_json(), &[]).unwrap();
        let hfq = HfqFile::open(&path).unwrap();
        let error = admit_hfqm_artifact(&hfq)
            .expect_err("an empty index is not the profile's inventory")
            .to_string();
        assert!(
            !error.contains("config admission failed"),
            "the explicit marker must pass the parser: {error}"
        );
        assert!(error.contains("tensors but manifest declares"), "{error}");
        drop(hfq);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(path.parent().unwrap());

        // Marker stripped: the same geometry is not recognised as synthetic.
        let mut value: serde_json::Value =
            serde_json::from_str(&config.developer_metadata_json()).unwrap();
        value.as_object_mut().unwrap().remove("hipfire_profile");
        if let Some(inner) = value.get_mut("config").and_then(|c| c.as_object_mut()) {
            inner.remove("hipfire_profile");
        }
        let error = admit_developer_inventory_headers(ARCH_ID, &value.to_string(), &[])
            .expect_err("synthetic geometry without the marker is not admitted")
            .to_string();
        assert!(error.contains("config admission failed"), "{error}");
    }

    const SYM_GATE_UP: &str = "model.language_model.layers.0.mlp.experts.gate_up_proj";
    const SYM_DOWN: &str = "model.language_model.layers.0.mlp.experts.down_proj";
    /// f16 bit patterns: 1.0, -8.0, -1.0, 8.0, 2.0, -2.0, +inf, quiet NaN.
    const F16_ONE: u16 = 0x3C00;
    const F16_NEG_EIGHT: u16 = 0xC800;
    const F16_NEG_ONE: u16 = 0xBC00;
    const F16_EIGHT: u16 = 0x4800;
    const F16_NEG_TWO: u16 = 0xC000;
    const F16_INF: u16 = 0x7C00;
    const F16_NAN: u16 = 0x7E00;

    #[test]
    fn f16_decode_matches_ieee_binary16() {
        assert_eq!(f16_bits_to_f32(F16_ONE), 1.0);
        assert_eq!(f16_bits_to_f32(F16_NEG_EIGHT), -8.0);
        assert_eq!(f16_bits_to_f32(0x0001), 2.0f32.powi(-24));
        assert_eq!(f16_bits_to_f32(0x03FF), 1023.0 * 2.0f32.powi(-24));
        assert_eq!(f16_bits_to_f32(0x0400), 2.0f32.powi(-14));
        assert_eq!(f16_bits_to_f32(0x7BFF), 65504.0);
        assert!(f16_bits_to_f32(0x8000).is_sign_negative());
        assert_eq!(f16_bits_to_f32(0x8000), 0.0);
        assert!(f16_bits_to_f32(F16_INF).is_infinite());
        assert!(f16_bits_to_f32(F16_NAN).is_nan());
    }

    #[test]
    fn header_predicate_matches_the_gpu_checker() {
        assert!(header_symmetric(F16_ONE, F16_NEG_EIGHT));
        // A negative scale pairs with a positive zero point.
        assert!(header_symmetric(F16_NEG_ONE, F16_EIGHT));
        // Signed zero: zp != -8*sc is an IEEE compare, so +0/-0 both pass.
        assert!(header_symmetric(0x0000, 0x0000));
        assert!(header_symmetric(0x0000, 0x8000));
        assert!(header_symmetric(0x8000, 0x0000));
        assert!(header_symmetric(0x8000, 0x8000));
        // Finite but off the grid.
        assert!(!header_symmetric(F16_ONE, F16_NEG_TWO));
        assert!(!header_symmetric(F16_ONE, F16_ONE));
        // Non-finite scale or zero point, even where the product would match.
        assert!(!header_symmetric(F16_INF, F16_INF));
        assert!(!header_symmetric(F16_NAN, F16_NAN));
        assert!(!header_symmetric(F16_ONE, F16_INF));
        assert!(!header_symmetric(F16_NAN, F16_NEG_EIGHT));
    }

    /// One packed group: `headers` (scale, zero) pairs at the group start, then
    /// payload filler that must never be read as a header.
    fn sym_group(group_bytes: usize, headers: &[(u16, u16)]) -> Vec<u8> {
        let mut group = vec![0xA5u8; group_bytes];
        for (index, (scale, zero)) in headers.iter().enumerate() {
            group[4 * index..4 * index + 2].copy_from_slice(&scale.to_le_bytes());
            group[4 * index + 2..4 * index + 4].copy_from_slice(&zero.to_le_bytes());
        }
        group
    }

    const SYM_OK: (u16, u16) = (F16_ONE, F16_NEG_EIGHT);

    /// `[experts, rows, K]` gate/up (QT44, K=256: one two-header group per
    /// row) and down (QT53, K=256: two one-header groups per row), the groups
    /// supplied in storage order.
    fn sym_fixture(
        name: &str,
        gate_up_groups: &[Vec<u8>],
        down_groups: &[Vec<u8>],
        down_quant_type: u8,
    ) -> (PathBuf, HfqFile) {
        let path = fixture_path(name);
        let tensors = vec![
            HfqMemTensor {
                name: SYM_GATE_UP.to_string(),
                quant_type: QWEN4_QT_MQ4G256V2,
                shape: vec![2, 2, 256],
                group_size: 256,
                data: gate_up_groups.concat(),
            },
            HfqMemTensor {
                name: SYM_DOWN.to_string(),
                quant_type: down_quant_type,
                shape: vec![2, 2, 256],
                group_size: 128,
                data: down_groups.concat(),
            },
        ];
        write_hfqm_package_mem(&path, ARCH_ID, "{}", &tensors).unwrap();
        let hfq = HfqFile::open(&path).unwrap();
        (path, hfq)
    }

    fn sym_cleanup(path: PathBuf, hfq: HfqFile) {
        drop(hfq);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(path.parent().unwrap());
    }

    /// Four rows (2 experts x 2 rows) of gate/up, eight QT53 groups of down.
    fn sym_all_ok() -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        (
            vec![sym_group(136, &[SYM_OK, SYM_OK]); 4],
            vec![sym_group(68, &[SYM_OK]); 8],
        )
    }

    #[test]
    fn routed_header_scan_accepts_a_symmetric_payload() {
        let (gate_up, down) = sym_all_ok();
        let (path, hfq) = sym_fixture("sym-ok", &gate_up, &down, QWEN4_QT_MQ4G128V2);
        assert!(routed_tensors_symmetric(&hfq, &[SYM_GATE_UP, SYM_DOWN]).unwrap());
        sym_cleanup(path, hfq);
    }

    #[test]
    fn routed_header_scan_checks_both_qt44_subheaders_and_every_qt53_group() {
        // Only the second K128 header of the last gate/up group is off-grid.
        let (mut gate_up, down) = sym_all_ok();
        gate_up[3] = sym_group(136, &[SYM_OK, (F16_ONE, F16_NEG_TWO)]);
        let (path, hfq) = sym_fixture("sym-gate-up-sub", &gate_up, &down, QWEN4_QT_MQ4G128V2);
        assert!(!routed_tensors_symmetric(&hfq, &[SYM_GATE_UP, SYM_DOWN]).unwrap());
        sym_cleanup(path, hfq);

        // Only the last down group is off-grid.
        let (gate_up, mut down) = sym_all_ok();
        down[7] = sym_group(68, &[(F16_ONE, F16_NEG_TWO)]);
        let (path, hfq) = sym_fixture("sym-down-last", &gate_up, &down, QWEN4_QT_MQ4G128V2);
        assert!(!routed_tensors_symmetric(&hfq, &[SYM_GATE_UP, SYM_DOWN]).unwrap());
        sym_cleanup(path, hfq);
    }

    #[test]
    fn routed_header_scan_rejects_non_finite_headers_and_accepts_signed_zero() {
        let (mut gate_up, down) = sym_all_ok();
        gate_up[0] = sym_group(136, &[(F16_INF, F16_INF), SYM_OK]);
        let (path, hfq) = sym_fixture("sym-inf", &gate_up, &down, QWEN4_QT_MQ4G128V2);
        assert!(!routed_tensors_symmetric(&hfq, &[SYM_GATE_UP, SYM_DOWN]).unwrap());
        sym_cleanup(path, hfq);

        let (gate_up, mut down) = sym_all_ok();
        down[2] = sym_group(68, &[(F16_NAN, F16_NAN)]);
        let (path, hfq) = sym_fixture("sym-nan", &gate_up, &down, QWEN4_QT_MQ4G128V2);
        assert!(!routed_tensors_symmetric(&hfq, &[SYM_GATE_UP, SYM_DOWN]).unwrap());
        sym_cleanup(path, hfq);

        let (mut gate_up, mut down) = sym_all_ok();
        gate_up[1] = sym_group(136, &[(0x8000, 0x0000), (0x0000, 0x8000)]);
        down[5] = sym_group(68, &[(F16_NEG_ONE, F16_EIGHT)]);
        let (path, hfq) = sym_fixture("sym-zero", &gate_up, &down, QWEN4_QT_MQ4G128V2);
        assert!(routed_tensors_symmetric(&hfq, &[SYM_GATE_UP, SYM_DOWN]).unwrap());
        sym_cleanup(path, hfq);
    }

    #[test]
    fn routed_header_scan_refuses_short_payloads_and_other_tiers() {
        // One byte short on down: an invalid payload is an error, not `false`.
        let (gate_up, mut down) = sym_all_ok();
        down[7].pop();
        let (path, hfq) = sym_fixture("sym-short", &gate_up, &down, QWEN4_QT_MQ4G128V2);
        let error = routed_tensors_symmetric(&hfq, &[SYM_GATE_UP, SYM_DOWN])
            .expect_err("a short routed payload is invalid");
        assert!(error.to_string().contains("expected"), "{error}");
        sym_cleanup(path, hfq);

        // A finite tier other than QT53 down is not the symmetric route's.
        let (gate_up, down) = sym_all_ok();
        let (path, hfq) = sym_fixture("sym-tier", &gate_up, &down, QWEN4_QT_MQ6G256V2);
        assert!(!routed_tensors_symmetric(&hfq, &[SYM_GATE_UP, SYM_DOWN]).unwrap());
        sym_cleanup(path, hfq);

        // No routed experts, or gate/up without its down: nothing to prove.
        let (gate_up, down) = sym_all_ok();
        let (path, hfq) = sym_fixture("sym-unpaired", &gate_up, &down, QWEN4_QT_MQ4G128V2);
        assert!(!routed_tensors_symmetric(&hfq, &[]).unwrap());
        assert!(!routed_tensors_symmetric(&hfq, &[SYM_GATE_UP]).unwrap());
        sym_cleanup(path, hfq);
    }
}
