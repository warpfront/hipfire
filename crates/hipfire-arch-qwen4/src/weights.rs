// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed Qwen4 manifest declarations and GPU-owned assembly.
//!
//! The names in this module are the names emitted by the pinned
//! `Qwen/Qwen3.8-Flash-Next` checkpoint.  In particular, the decoder uses the
//! `model.language_model.*` namespace, QSA keeps separate q/k/v projections,
//! and the PLE table is a set of numerically suffixed external rows.  The
//! runtime's [`WeightLoadTransaction`] remains the only allocation owner until
//! this module finalizes handles into [`Qwen4Weights`].

use crate::config::{LayerType, Qwen4Config, Qwen4FfnConfig, Qwen4PleConfig, Qwen4Profile};
use hipfire_runtime::external_rows::RowEncoding;
use hipfire_runtime::mesh_placement::BandPlacement;
use hipfire_runtime::model_source::SourceRangeDescriptor;
use hipfire_runtime::tp_shard::ExpertAssign;
use hipfire_runtime::trunk_overlay::TrunkOverlayPayload;
use hipfire_runtime::weight_manifest::WeightResidency;
use hipfire_runtime::weight_manifest::{
    DTypeConstraint, ExpertSourceLayout, PinTarget, PlacementHint, ShardPolicy, StateEntry,
    StateKind, StateScope, WeightEntry,
};
use hipfire_runtime::weight_store::{TakenWeight, WeightHandle, WeightLoadTransaction};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::ops::Range;

/// Rank-local ownership projected from the admitted, absolute trunk partition.
/// This value owns neither devices nor an independent topology.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4BandScope {
    pub band: Range<usize>,
    pub input: bool,
    pub output: bool,
}

impl Qwen4BandScope {
    /// The historical whole-model allocation and assembly scope.
    pub fn single(config: &Qwen4Config) -> Self {
        Self {
            band: 0..config.num_hidden_layers,
            input: true,
            output: true,
        }
    }

    /// Project one admitted stage, refusing a PLE cut off the input owner.
    pub fn from_placement(
        config: &Qwen4Config,
        placement: &BandPlacement,
        stage: usize,
    ) -> Result<Self, String> {
        if placement.n_trunk_layers != config.num_hidden_layers || stage >= placement.stages() {
            return Err("Qwen4 band placement does not match the model or stage".to_string());
        }
        if let Some(ple) = &config.ple {
            if !placement.band(0).contains(&ple.layer) {
                return Err("Qwen4 PLE layer must belong to the input band".to_string());
            }
        }
        let scope = Self {
            band: placement.band(stage),
            input: stage == 0,
            output: stage + 1 == placement.stages(),
        };
        scope.validate(config)?;
        Ok(scope)
    }

    /// Check absolute geometry and endpoint ownership before side effects.
    pub fn validate(&self, config: &Qwen4Config) -> Result<(), String> {
        if self.band.is_empty()
            || self.band.end > config.num_hidden_layers
            || config.layer_types.len() != config.num_hidden_layers
            || self.input != (self.band.start == 0)
            || self.output != (self.band.end == config.num_hidden_layers)
        {
            return Err("invalid Qwen4 band or endpoint ownership".to_string());
        }
        if config
            .ple
            .as_ref()
            .is_some_and(|ple| self.input != self.owns_layer(ple.layer))
        {
            return Err("Qwen4 PLE layer must belong to the input band".to_string());
        }
        Ok(())
    }

    pub fn owns_layer(&self, layer: usize) -> bool {
        self.band.contains(&layer)
    }

    /// MTP is an output-pinned auxiliary namespace, never another trunk layer.
    pub fn owns_entry(&self, entry: &WeightEntry) -> bool {
        if entry.name.starts_with("mtp.") {
            return self.output;
        }
        if let Some(layer) = entry.layer {
            return self.owns_layer(layer);
        }
        match entry.placement {
            PlacementHint::Pin(PinTarget::Embed) => self.input,
            PlacementHint::Pin(PinTarget::Output) => self.output,
            PlacementHint::Policy => match entry.policy {
                ShardPolicy::Pin(PinTarget::Embed) => self.input,
                _ => self.output,
            },
        }
    }
}

/// The routed experts' declared targets: gate/up at the aligned-K group width
/// and down at the row-local one.
///
/// The E8 lattice format is deliberately *not* used here.  It rotates and
/// encodes in 256-wide FWHT segments, so `K % 256 == 0` is a precondition of
/// both the encoder and the decode kernels (`groups_per_row = K / 256`), and
/// `experts.down_proj` reduces over `moe_intermediate_size = 640` — three
/// quarters of a segment short.  The row-local MQ4G128V2 group exists for
/// exactly this case, and both tiers cost the same 4.25 bits per weight, so the
/// expert family keeps the pair the qwen4 typed route's kernels are built on.
pub const ROUTED_GATE_UP_DTYPE: DType = DType::MQ4G256V2;
pub const ROUTED_DOWN_DTYPE: DType = DType::MQ4G128V2;
pub const PLE_SHARD_ROWS: usize = 2_500_012;
pub use crate::ple::{PLE_HEAD_COUNT, PLE_ROW_WIDTH};
pub const PLE_SHARD_COUNT: usize = 128;
/// Upper bound on the PLE shard tensors a profile may declare, checked before
/// the manifest allocates anything proportional to the shard count.
const MAX_PLE_SHARD_TENSORS: usize = 65_536;

/// Packed formats this family's planner recognises.
///
/// A declaration outside this set is refused by name rather than demoted to
/// BF16, so the predicate is a recognition list, not the tier policy: the
/// policy lives in [`qwen4_matrix_dtype`].
fn qwen4_quantized_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::MQ4G256V2 | DType::MQ4G128V2 | DType::MQ6G256V2 | DType::MFP4G32E8SOA | DType::Q8_0
    )
}

/// The planar expert-slab dtypes (qt=131..=136) that no qwen4 trunk/dense route
/// has a kernel for.  The qwen4 planner never produces them, but a caller-supplied
/// [`DType`] could otherwise ride through [`qwen4_target_dtype`]'s
/// pass-through arm into a declaration and fail much later in a generic
/// consumer.
fn qwen4_expert_slab_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::MQ3I4G128R256
            | DType::MQ3I4G128R128
            | DType::MQ2I4G128R256
            | DType::MQ2I4G128R128
            | DType::E8PI4G128R256
            | DType::E8PI4G128R128
    )
}

/// Typed refusal of an expert-slab dtype at a result-returning boundary that
/// receives a caller-supplied [`DType`].
fn reject_expert_slab_dtype(dtype: DType) -> Result<(), WeightError> {
    if qwen4_expert_slab_dtype(dtype) {
        Err(WeightError::ExpertSlabDtypeKernelsUnavailable { dtype })
    } else {
        Ok(())
    }
}

/// Payloads that stay source-exact BF16 even though they are rank-2 matrices.
///
/// The Qwen4 family quantizes every matrix whose consumers read a quantized
/// weight — the trunk projections, the shared expert, the router, the PLE
/// key/value projections and the embedding/head — mirroring the canonical
/// `qwen3.6-35b-a3b` layout.  The MTP connector matrices are the exception:
/// their owner (`mtp_gpu`) still binds source BF16, so declaring them
/// quantized here would produce an artifact nothing can read.
const QWEN4_SOURCE_EXACT_SUFFIXES: &[&str] = &["mtp.fc_embedding.weight", "mtp.fc_hidden.weight"];

/// Payload classes that stay source-exact BF16 despite being rank-2 matrices.
///
/// The hyper-connection mixer is four output rows over a `hc_count * hidden`
/// reduction — the thinnest matrix in the network against the widest activation
/// — so it keeps source bytes rather than being packed for a rounding error's
/// worth of traffic.
const QWEN4_SOURCE_EXACT_MARKERS: &[&str] = &["hyper_connection"];

/// The classes the repo's MoE recipes hold at eight bits rather than four.
///
/// Every family's embedding arm ships Q8 (never MQ4), and the quantizer's MoE
/// default promotes the whole fixed tier — attention, lm_head, embed, router —
/// to Q8F16 (qt=3, 34 bytes per 32 weights) because it is quality-critical and
/// small relative to the routed experts.  `qwen3.6-35b-a3b.mq6` ships its
/// lm_head, embed and router at qt=3 while its experts carry the six-bit tier.
const QWEN4_Q8_MARKERS: &[&str] = &["embed_tokens.weight"];

/// The untied 248k-vocab head ships at the trunk's six-bit tier (MQ6G256V2):
/// 0.50 GB read per token instead of Q8F16's 0.68, measured on the decode
/// route at +0.0005 KLD against the Q8F16 head (2026-09-28).  Four bits cost
/// +0.014 KLD there, so six is the floor.
///
/// The other matrices decode streams at BF16 (HC read, shared expert) stay
/// BF16 here: decode reads load-time Q8_0 copies of them
/// (`Qwen4GpuForward::decode_q8`) while prefill keeps the BF16 source — Q8 in
/// the file cost the prefill route +0.0021 KLD (HC) and short-prompt prefill
/// 46% (shared expert, no fast exact multi-row Q8 GEMM) for no decode speed.
pub(crate) const QWEN4_LM_HEAD: &str = "lm_head.weight";

/// The input token embedding table (Q8_0 in a Flash-Next artifact). The MTP
/// head's `mtp.embed_tokens.weight` is a tied alias of it.
pub const QWEN4_EMBED: &str = "model.language_model.embed_tokens.weight";

fn qwen4_q8_dtype(name: &str) -> bool {
    QWEN4_Q8_MARKERS.iter().any(|marker| name.ends_with(marker))
}

fn qwen4_quantizable_matrix(name: &str, shape: &[usize]) -> bool {
    if shape.len() == 3 {
        return name.ends_with(".experts.down_proj") || name.ends_with(".experts.gate_up_proj");
    }
    if shape.len() != 2
        || QWEN4_SOURCE_EXACT_SUFFIXES
            .iter()
            .any(|s| name.ends_with(s))
        || QWEN4_SOURCE_EXACT_MARKERS
            .iter()
            .any(|marker| name.contains(marker))
    {
        return false;
    }
    // The packed set is the wide attention/GDN projections, which dominate
    // decode weight traffic.  The shared expert, the router, and the PLE
    // projections stay source-exact: the shared expert is as wide as any
    // trunk matrix but rides the sealed-MoE route, and the router decides
    // expert selection from 512 logits.
    if !name.contains(".linear_attn.") && !name.contains(".self_attn.") {
        return false;
    }
    // The eight-bit block format carries any K that is a multiple of its
    // 32-weight block, so alignment beyond the admission boundary above is not
    // a packing precondition.  A K that is not a multiple of 128 stays BF16.
    shape.last().is_some_and(|k| k % 128 == 0)
}

/// The tier each packed class ships at.
///
/// Routed experts keep the four-bit MQ4 family: their error is averaged over the
/// ten experts a token selects, they are where nearly all of the weight traffic
/// is (512 experts x two matrices per layer), and the qwen4 typed route's
/// grouped kernels are built on this exact qt44/qt53 pair.  The rank-2 trunk
/// carries eight bits instead — a dense projection writes its error straight
/// into the residual stream, and this family has already measured what a
/// lossier tier does there: a four-bit trunk opened a reasoning block the model
/// could not close.  Q8F16 costs 8.5 bpw but needs no rotation basis at all, and
/// the trunk is a rounding error next to the experts.
///
/// Invariant: this takes only a shape, so it can never receive an expert-slab
/// dtype (`MQ3I4G128R256`/`R128`, `MQ2I4G128R256`/`R128`) and always returns
/// one of the MQ4/Q8 tiers above.  The public entries that carry a caller
/// [`DType`] ([`TensorRef::new`], [`Qwen4Manifest::build_with_recipe`],
/// [`Qwen4Manifest::with_expert_parallel`] and [`Qwen4Weights::assemble`])
/// refuse those dtypes with [`WeightError::ExpertSlabDtypeKernelsUnavailable`]
/// before any mapping runs.
fn qwen4_matrix_dtype(shape: &[usize]) -> DType {
    let k = shape.last().copied().unwrap_or_default();
    if shape.len() == 3 {
        // gate/up reduces over `hidden` (256-aligned) and takes the aligned-K
        // group; down reduces over `moe_intermediate_size` and takes the
        // row-local one.
        if k % 256 == 0 {
            ROUTED_GATE_UP_DTYPE
        } else {
            ROUTED_DOWN_DTYPE
        }
    } else {
        DType::Q8_0
    }
}

/// Pure mapping: a `requested` dtype that is neither BF16 nor a recognised
/// packed format is returned unchanged, so this function would preserve an
/// expert-slab dtype.  It is infallible by design; every public caller that
/// can hold such a dtype rejects it first (see [`qwen4_matrix_dtype`]).
fn qwen4_target_dtype(name: &str, shape: &[usize], requested: DType) -> DType {
    if matches!(requested, DType::BF16 | DType::Q8_0) {
        if name == QWEN4_LM_HEAD {
            return DType::MQ6G256V2;
        }
        if qwen4_q8_dtype(name) {
            return DType::Q8_0;
        }
    }
    if qwen4_quantizable_matrix(name, shape)
        && (requested == DType::BF16 || qwen4_quantized_dtype(requested))
    {
        qwen4_matrix_dtype(shape)
    } else {
        requested
    }
}

/// Q8 targets accept a BF16 source (the checkpoint) or an already-converted
/// Q8 payload, and nothing else: any other packed family is a different
/// geometry and would be misread here.
fn qwen4_q8_source() -> DTypeConstraint {
    DTypeConstraint::source_from_sources(vec![DType::BF16, DType::Q8_0])
}

/// Routed experts accept the checkpoint's BF16 or the published four-bit
/// payload at either declared geometry.
fn qwen4_quant_source() -> DTypeConstraint {
    DTypeConstraint::source_from_sources(vec![DType::BF16, ROUTED_GATE_UP_DTYPE, ROUTED_DOWN_DTYPE])
}

/// The trunk's wide attention/GDN projections are the one packed class whose
/// tier the *artifact* chooses rather than this build.
///
/// Both rungs describe the same logical matrix and load as the same resident
/// handle: the six-bit rung carries MQ6G256V2 (qt=47) and the eight-bit rung
/// carries Q8F16 (qt=3), which is the tier this build declares.  The loader
/// reads the tier the artifact actually ships and derives every row geometry
/// from that tag, so the source contract admits either payload.  Naming only
/// the tier this build happens to write would make a published rung of the
/// other tier unloadable by the very tree that produced it.
///
/// A third rung is admitted for the dense IU4 trunk (opt-in through
/// `HIPFIRE_QWEN4_TRUNK_IU4`, or the default of an artifact carrying the
/// `qwen4_trunk_iu4_default` marker): symmetric MQ4G256V2 (qt=44), the aligned-K
/// 256 group the trunk's GEMV (`gemv_mq4g256v2`) and the dense IU4 SET
/// kernels read. Other four-bit families stay out: their group width is one
/// the trunk's decode kernels do not read, and their admission would only
/// move the failure to the artifact boundary.
fn qwen4_trunk_matrix_source() -> DTypeConstraint {
    DTypeConstraint::source_from_sources(vec![
        DType::BF16,
        DType::Q8_0,
        DType::MQ6G256V2,
        DType::MQ4G256V2,
    ])
}

/// True for the rank-2 trunk attention/GDN matrices — the class
/// [`qwen4_trunk_matrix_source`] governs.
fn qwen4_trunk_matrix(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2 && qwen4_quantizable_matrix(name, shape)
}

/// The packed tier the rank-2 trunk matrices ship at.
///
/// `HIPFIRE_QWEN4_TRUNK_TIER=mq6` puts the wide attention/GDN projections on
/// the aligned-K six-bit group (MQ6G256V2, qt=47, 6.25 bpw) instead of the
/// eight-bit recipe this build declares by default (Q8F16, qt=3, 8.5 bpw).
/// Any other value — an unset variable included — selects that default, so the
/// knob only ever adds the six-bit rung and every recipe and artifact written
/// so far keeps its meaning.
///
/// The producer and the loader reach the declaration through the same
/// [`Qwen4Manifest::build`], and the trunk's source contract
/// ([`qwen4_trunk_matrix_source`]) admits both rungs, so each build loads the
/// other's artifact.  The MTP namespace declares its own targets through
/// [`qwen4_target_dtype`] and never reads this.
fn qwen4_trunk_tier(value: Option<&str>) -> DType {
    match value {
        Some("mq6") => DType::MQ6G256V2,
        _ => DType::Q8_0,
    }
}

/// `HIPFIRE_QWEN4_MTP_TIER=source` declares the rank-2 MTP matrices at their
/// source's own BF16 instead of the eight-bit recipe the shared matrix
/// classifier otherwise picks for them.
///
/// Scratch measurements found no acceptance lift from this tier alone; it
/// remains an opt-in comparison against the published Q8F16 MTP projections.
/// Unset, the existing recipe is unchanged, and the producer and loader use
/// the same manifest declaration.
fn qwen4_mtp_source_tier(value: Option<&str>) -> bool {
    matches!(value, Some("source"))
}

thread_local! {
    /// Test-only override so a manifest assertion can exercise the knob without
    /// a process-global environment write.  Production always reads the env.
    static MTP_SOURCE_TIER_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// The predicate the manifest actually consults, so a test can prove the knob
/// *reaches* the five rank-2 MTP attention matrices rather than only parsing.
fn mtp_source_tier_active() -> bool {
    MTP_SOURCE_TIER_OVERRIDE
        .with(|value| value.get())
        .unwrap_or_else(|| {
            qwen4_mtp_source_tier(
                hipfire_config::developer_var("HIPFIRE_QWEN4_MTP_TIER")
                    .ok()
                    .as_deref(),
            )
        })
}

/// The declared target of one entry, with the selected trunk tier applied.
///
/// Only the rank-2 trunk attention/GDN class moves, and only when the request
/// was packed in the first place: a request that [`qwen4_target_dtype`] leaves
/// unpacked (F32, or a class this family keeps source-exact) keeps its dtype.
fn qwen4_trunk_target(name: &str, shape: &[usize], requested: DType, trunk_tier: DType) -> DType {
    let dtype = qwen4_target_dtype(name, shape, requested);
    if dtype == DType::Q8_0 && qwen4_trunk_matrix(name, shape) {
        trunk_tier
    } else {
        dtype
    }
}

/// Metadata key naming the recipe a developer artifact was written under.
/// A Flash-Next artifact's key predates the typed recipe and carries no `id`,
/// except the stock-FP8 artifact ([`STOCK_FP8_RECIPE_ID`]), which a Flash-Next
/// config may declare; a synthetic profile reads or requires one.
pub const RECIPE_METADATA_KEY: &str = "qwen4_recipe";
/// `id` of [`Qwen4TierRecipe::SyntheticSourceBf16`] in artifact metadata.
pub const SYNTHETIC_SOURCE_BF16_RECIPE_ID: &str = "synthetic-source-bf16";
/// `id` of [`Qwen4TierRecipe::SyntheticPacked`] in artifact metadata.
pub const SYNTHETIC_PACKED_RECIPE_ID: &str = "synthetic-packed";
/// `id` of [`Qwen4TierRecipe::StockFp8`] in artifact metadata (written by
/// `hipfire-quantize --recipe stock-fp8`).
pub const STOCK_FP8_RECIPE_ID: &str = "stock-fp8";

impl Qwen4TierRecipe {
    /// Stable artifact-metadata id of a synthetic recipe; `None` for the
    /// Flash-Next recipes.
    pub const fn synthetic_id(&self) -> Option<&'static str> {
        match self {
            Self::FlashNextPacked { .. } | Self::FlashNextSlab { .. } | Self::StockFp8 => None,
            Self::SyntheticSourceBf16 => Some(SYNTHETIC_SOURCE_BF16_RECIPE_ID),
            Self::SyntheticPacked => Some(SYNTHETIC_PACKED_RECIPE_ID),
        }
    }

    /// The synthetic recipe an artifact-metadata `id` names.
    pub fn from_synthetic_id(id: &str) -> Option<Self> {
        match id {
            SYNTHETIC_SOURCE_BF16_RECIPE_ID => Some(Self::SyntheticSourceBf16),
            SYNTHETIC_PACKED_RECIPE_ID => Some(Self::SyntheticPacked),
            _ => None,
        }
    }

    /// The Flash-Next recipe an artifact-metadata `id` names (only
    /// [`STOCK_FP8_RECIPE_ID`]; the packed recipe carries no id).
    pub fn from_flash_next_id(id: &str) -> Option<Self> {
        match id {
            STOCK_FP8_RECIPE_ID => Some(Self::StockFp8),
            _ => None,
        }
    }

    /// The stock-FP8 recipe ([`Self::StockFp8`]).
    pub const fn is_stock_fp8(&self) -> bool {
        matches!(self, Self::StockFp8)
    }

    /// Whether the Flash-Next trunk (every trunk/head/HC/shared-expert/router/
    /// embedding matrix) is source BF16 (qt=16) loaded raw — the
    /// `HIPFIRE_CALIB_BF16` behaviour as a recipe property: the forward keeps
    /// BF16 reads and builds no Q8 decode copies.  True for [`Self::StockFp8`]
    /// only (a synthetic profile never takes the Flash-Next decode recipe).
    pub const fn trunk_raw_bf16(&self) -> bool {
        self.is_stock_fp8()
    }

    /// Whether the routed experts are qt=55 `FP8E4M3B128` tables, which only
    /// a device with `Qwen4Capabilities::fp8_experts` (gfx942) may load.
    pub const fn fp8_experts(&self) -> bool {
        self.is_stock_fp8()
    }

    /// Refuses a recipe whose trunk tier names an expert-slab dtype
    /// (`MQ3I4G128R256`/`R128`, `MQ2I4G128R256`/`R128`,
    /// `E8PI4G128R256`/`R128`): no qwen4 trunk kernel exists for them.
    pub fn reject_expert_slab_dtypes(&self) -> Result<(), WeightError> {
        match self {
            Self::FlashNextPacked { trunk_tier, .. } | Self::FlashNextSlab { trunk_tier, .. } => {
                reject_expert_slab_dtype(*trunk_tier)
            }
            Self::SyntheticSourceBf16 | Self::SyntheticPacked | Self::StockFp8 => Ok(()),
        }
    }
}

/// What [`Qwen4TierRecipe::SyntheticPacked`] declares for one tensor, with the
/// role class and, when the role stays source-exact BF16, the reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyntheticPackDecision {
    pub dtype: DType,
    pub class: &'static str,
    /// Empty when the role is packed.
    pub reason: &'static str,
}

/// The synthetic packed recipe's per-role target: the Flash-Next formats at
/// the role's own reduction width `K` (the last dimension), declared only where
/// `K` tiles the format and a consumer reads it, BF16 otherwise.
///
/// - token embedding: Q8F16 when `K % 32 == 0`;
/// - language head: MQ6G256V2 when `K % 256 == 0`, else Q8F16 when `K % 32 == 0`;
/// - trunk attention/GDN projections (rank 2): Q8F16 when `K % 128 == 0`, the
///   Flash-Next trunk rule, without the `HIPFIRE_QWEN4_TRUNK_TIER` knob;
/// - dense FFN gate/up/down: Q8F16 when `K % 32 == 0`;
/// - routed experts: gate/up MQ4G256V2 and down MQ4G128V2, **both or neither**
///   (the sealed routed route reads the qt44/qt53 pair): packed only when the
///   hidden width is a multiple of 256 and the expert width of 128;
/// - everything else (hyper-connection mixers, router, shared expert, PLE
///   projections, MTP connectors, norms, convolutions) stays BF16 exactly as in
///   the Flash-Next recipe.
pub fn synthetic_packed_decision(name: &str, shape: &[usize]) -> SyntheticPackDecision {
    let k = shape.last().copied().unwrap_or(0);
    let packed = |dtype: DType, class: &'static str| SyntheticPackDecision {
        dtype,
        class,
        reason: "",
    };
    let exact = |class: &'static str, reason: &'static str| SyntheticPackDecision {
        dtype: DType::BF16,
        class,
        reason,
    };
    const Q8_BLOCK: &str = "K is not a multiple of the 32-weight Q8F16 block";
    if shape.len() == 2 && name == QWEN4_LM_HEAD {
        return if k % 256 == 0 {
            packed(DType::MQ6G256V2, "language head")
        } else if k % 32 == 0 {
            packed(DType::Q8_0, "language head")
        } else {
            exact("language head", Q8_BLOCK)
        };
    }
    if shape.len() == 2 && qwen4_q8_dtype(name) {
        return if k % 32 == 0 {
            packed(DType::Q8_0, "token embedding")
        } else {
            exact("token embedding", Q8_BLOCK)
        };
    }
    if shape.len() == 2 && name.contains(".ple_embedding.ngram_embedding.shard_") {
        return if synthetic_packed_ple_row_dtype(k) == DType::Q8_0 {
            packed(DType::Q8_0, "PLE rows")
        } else {
            exact("PLE rows", Q8_BLOCK)
        };
    }
    let gate_up = name.ends_with(".experts.gate_up_proj");
    let down = name.ends_with(".experts.down_proj");
    if shape.len() == 3 && (gate_up || down) {
        let class = if gate_up {
            "routed gate/up"
        } else {
            "routed down"
        };
        // gate_up is [experts, 2 * intermediate, hidden]; down is
        // [experts, hidden, intermediate].
        let (hidden, intermediate) = if gate_up {
            if shape[1] % 2 != 0 {
                return exact(class, "stacked gate/up rows are not an even pair");
            }
            (shape[2], shape[1] / 2)
        } else {
            (shape[1], shape[2])
        };
        return if hidden % 256 != 0 {
            exact(
                class,
                "hidden width is not a multiple of the 256-weight MQ4G256V2 group \
                 (routed gate/up and down pack together)",
            )
        } else if intermediate % 128 != 0 {
            exact(
                class,
                "expert intermediate width is not a multiple of the 128-weight MQ4G128V2 group \
                 (routed gate/up and down pack together)",
            )
        } else if gate_up {
            packed(ROUTED_GATE_UP_DTYPE, class)
        } else {
            packed(ROUTED_DOWN_DTYPE, class)
        };
    }
    if shape.len() == 2
        && !name.contains(".shared_expert.")
        && name.contains(".mlp.")
        && (name.ends_with(".mlp.gate_proj.weight")
            || name.ends_with(".mlp.up_proj.weight")
            || name.ends_with(".mlp.down_proj.weight"))
    {
        return if k % 32 == 0 {
            packed(DType::Q8_0, "dense FFN")
        } else {
            exact("dense FFN", Q8_BLOCK)
        };
    }
    if shape.len() == 2 && (name.contains(".linear_attn.") || name.contains(".self_attn.")) {
        return if k % 128 == 0 {
            packed(DType::Q8_0, "trunk projection")
        } else {
            exact(
                "trunk projection",
                "K is not a multiple of 128 (the Flash-Next trunk rule)",
            )
        };
    }
    const KEPT: &str = "the Flash-Next recipe keeps this role source-exact";
    let class = if name.contains("hyper_connection") {
        "hyper-connection mixer"
    } else if name.contains(".shared_expert") {
        "shared expert"
    } else if name.ends_with(".mlp.gate.weight") {
        "router"
    } else if name.contains(".ple.") {
        "PLE projection"
    } else if name.starts_with("mtp.fc_") {
        "MTP connector"
    } else {
        "source-exact parameter"
    };
    exact(class, KEPT)
}

/// Row tier of one synthetic-packed PLE table: Q8F16 rows when the row width
/// is a multiple of the 32-weight block (the reader's own addressing rule),
/// BF16 rows otherwise.
pub fn synthetic_packed_ple_row_dtype(row_width: usize) -> DType {
    if row_width % 32 == 0 {
        DType::Q8_0
    } else {
        DType::BF16
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum TensorRole {
    TokenEmbedding,
    LanguageHead,
    HyperConnection,
    HyperConnectionNorm,
    HyperConnectionBlockInject,
    HyperConnectionDown,
    HyperConnectionUp,
    GdnQkv,
    GdnZ,
    GdnA,
    GdnB,
    GdnDtBias,
    GdnALog,
    GdnNorm,
    GdnConv,
    GdnOutput,
    QsaQ,
    QsaK,
    QsaV,
    QsaOutput,
    QsaQNorm,
    QsaKNorm,
    QsaIndexerQk,
    QsaIndexerQNorm,
    QsaIndexerKNorm,
    Router,
    SharedExpertGateScalar,
    SharedExpertGate,
    SharedExpertUp,
    SharedExpertDown,
    RoutedGateUp,
    RoutedDown,
    /// A trunk layer's contiguous expert slab arena (`FlashNextSlab` recipe).
    RoutedArena,
    DenseFfnGate,
    DenseFfnUp,
    DenseFfnDown,
    PleShard,
    PleKey,
    PleValue,
    PleNorm,
    PleConv,
    PleMetadata,
    MtpEmbeddingProjection,
    MtpHiddenProjection,
    MtpNorm,
}

/// Typed logical reference.  The runtime manifest remains authoritative for
/// placement, aliases, source dtype, and residency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRef {
    pub name: String,
    pub role: TensorRole,
    pub layer: Option<usize>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl TensorRef {
    pub fn new(
        name: impl Into<String>,
        role: TensorRole,
        layer: Option<usize>,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Result<Self, WeightError> {
        let name = name.into();
        if name.is_empty() || shape.is_empty() || shape.iter().any(|&dim| dim == 0) {
            return Err(WeightError::InvalidShape { name, shape });
        }
        reject_expert_slab_dtype(dtype)?;
        let dtype = qwen4_target_dtype(&name, &shape, dtype);
        Ok(Self {
            name,
            role,
            layer,
            shape,
            dtype,
        })
    }
}

/// I64 metadata is retained as an exact source declaration rather than being
/// coerced into a floating-point `WeightEntry`.  The HFQM writer serializes
/// these arrays under its versioned `qwen4_ple` object; they never become GPU
/// resident weight handles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetadataTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub source_dtype: String,
}

impl MetadataTensor {
    fn i64(name: impl Into<String>, shape: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            shape,
            source_dtype: "I64".to_string(),
        }
    }
}

/// The tier recipe a manifest declares its targets and source constraints
/// under.  It is part of the manifest identity: two manifests with the same
/// names and shapes but different recipes are different artifacts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Qwen4TierRecipe {
    /// The pinned Flash-Next packed recipe: Q8/MQ6 trunk, MQ4 routed experts,
    /// Q8 embedding, MQ6 head, Q8F16 PLE rows.  `trunk_tier` and `mtp_source`
    /// are the two producer/loader-shared knobs, resolved once.
    FlashNextPacked { trunk_tier: DType, mtp_source: bool },
    /// [`Self::FlashNextPacked`] with the 48 trunk layers' routed experts held
    /// as per-layer slab arenas (version-1 expert-slab artifact): each layer
    /// declares ONE `slab_arena` entry (raw bytes, resident or host-mapped)
    /// instead of the two aggregate expert tensors, and `arenas` carries the
    /// validated per-expert format and offsets. The MTP head's experts stay
    /// the legacy aggregates.
    FlashNextSlab {
        trunk_tier: DType,
        mtp_source: bool,
        arenas: std::sync::Arc<crate::expert_arena::ExpertArenaPlan>,
    },
    /// The stock FP8 checkpoint (rev `236dfdf2`) converted byte-for-byte:
    /// routed experts qt=55 `FP8E4M3B128` (OCP e4m3 codes + BF16 `s_inv`
    /// 128x128 blocks, gate_up `[E, 1280, 2560]`, down `[E, 2560, 640]`),
    /// every other tensor source BF16 (qt=16) loaded raw, PLE rows qt=56
    /// `FP8E4M3PT`.  AR-only: the artifact carries no MTP head, so it is
    /// declared for [`crate::config::Qwen4Profile::FlashNextArOnly`] only.
    /// Selected by `qwen4_recipe.id = "stock-fp8"`; no environment knob
    /// moves it.
    StockFp8,
    /// Developer-only synthetic profiles stay conservative: every tensor is
    /// declared source-exact BF16 (and BF16 external PLE rows) until a role
    /// format has an executable consumer, a producer and a proof.  No
    /// environment knob moves this recipe.
    SyntheticSourceBf16,
    /// Developer-only synthetic profiles with the Flash-Next per-role packed
    /// formats, declared only where a role's reduction width tiles the format
    /// ([`synthetic_packed_decision`]); every other role stays source-exact
    /// BF16.  Each role admits exactly the one representation declared for it,
    /// so an artifact's recipe identity is its bytes, not a negotiation.
    SyntheticPacked,
}

/// Profile and recipe identity carried by every manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4ManifestIdentity {
    /// `Qwen4Profile::id()` of the config the manifest was built from.
    pub profile_id: String,
    pub developer_only: bool,
    /// `Qwen4EquationSource::as_str()` of the profile.
    pub equation_source: &'static str,
    /// `"dense"` or `"moe"`.
    pub ffn_kind: &'static str,
    pub recipe: Qwen4TierRecipe,
}

/// Pure logical Qwen4 manifest.  No GPU, source, or allocator is touched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4Manifest {
    pub identity: Qwen4ManifestIdentity,
    pub expert_keep: Option<crate::ExpertKeep>,
    pub weights: Vec<WeightEntry>,
    pub state: Vec<StateEntry>,
    pub metadata: Vec<MetadataTensor>,
}

fn positive(name: &str, value: usize) -> Result<usize, WeightError> {
    if value == 0 {
        return Err(WeightError::InvalidShape {
            name: name.to_string(),
            shape: vec![value],
        });
    }
    Ok(value)
}

fn checked_product(name: &str, a: usize, b: usize) -> Result<usize, WeightError> {
    a.checked_mul(b)
        .ok_or_else(|| WeightError::ShapeOverflow(name.to_string()))
}

impl Qwen4Manifest {
    /// The declaration both the producer and the loader read.
    ///
    /// Flash-Next takes its packed tier from `HIPFIRE_QWEN4_TRUNK_TIER`; the
    /// source-tier experiment affects only rank-2 MTP records.  The AR-only
    /// Flash-Next profile has exactly one recipe,
    /// [`Qwen4TierRecipe::StockFp8`].  A synthetic developer profile ignores
    /// both knobs and declares [`Qwen4TierRecipe::SyntheticSourceBf16`].
    pub fn build(config: &Qwen4Config) -> Result<Self, WeightError> {
        let tier = if config.profile.is_ar_only() {
            Qwen4TierRecipe::StockFp8
        } else if config.profile.is_flash_next() {
            Qwen4TierRecipe::FlashNextPacked {
                trunk_tier: qwen4_trunk_tier(
                    hipfire_config::developer_var("HIPFIRE_QWEN4_TRUNK_TIER")
                        .ok()
                        .as_deref(),
                ),
                mtp_source: mtp_source_tier_active(),
            }
        } else {
            Qwen4TierRecipe::SyntheticSourceBf16
        };
        Self::build_with_recipe(config, tier)
    }

    /// [`Self::build`] against an explicit trunk tier, so the selector's effect
    /// on the declaration is testable without a process-global environment.
    #[cfg(test)]
    fn build_with_trunk_tier(config: &Qwen4Config, trunk_tier: DType) -> Result<Self, WeightError> {
        Self::build_with_recipe(
            config,
            Qwen4TierRecipe::FlashNextPacked {
                trunk_tier,
                mtp_source: mtp_source_tier_active(),
            },
        )
    }

    /// The one inventory generator.  Names, logical shapes, optional PLE/MTP,
    /// the FFN family and the recipe identity all come from the typed profile
    /// in `config`; nothing here reintroduces a hard-coded Flash-Next count.
    pub fn build_with_recipe(
        config: &Qwen4Config,
        recipe: Qwen4TierRecipe,
    ) -> Result<Self, WeightError> {
        recipe.reject_expert_slab_dtypes()?;
        config.validate().map_err(WeightError::Config)?;
        let (packed, trunk_tier, mtp_source) = match &recipe {
            Qwen4TierRecipe::FlashNextPacked {
                trunk_tier,
                mtp_source,
            }
            | Qwen4TierRecipe::FlashNextSlab {
                trunk_tier,
                mtp_source,
                ..
            } => (true, *trunk_tier, *mtp_source),
            Qwen4TierRecipe::StockFp8
            | Qwen4TierRecipe::SyntheticSourceBf16
            | Qwen4TierRecipe::SyntheticPacked => (false, DType::BF16, false),
        };
        let synthetic_packed = matches!(recipe, Qwen4TierRecipe::SyntheticPacked);
        let stock_fp8 = recipe.is_stock_fp8();
        let profile_admits_recipe = match &recipe {
            Qwen4TierRecipe::FlashNextPacked { .. } | Qwen4TierRecipe::FlashNextSlab { .. } => {
                matches!(config.profile, Qwen4Profile::FlashNext)
            }
            Qwen4TierRecipe::StockFp8 => config.profile.is_ar_only(),
            Qwen4TierRecipe::SyntheticSourceBf16 | Qwen4TierRecipe::SyntheticPacked => {
                config.profile.is_synthetic()
            }
        };
        if !profile_admits_recipe {
            return Err(WeightError::UnsupportedProfile {
                profile: config.profile.id().to_string(),
                reason: format!("profile cannot be declared under the {recipe:?} recipe"),
            });
        }
        // The unpacked recipes declare one exact representation per tensor.
        let unpacked_dtype = |name: &str, shape: &[usize]| {
            if stock_fp8 {
                stock_fp8_dtype(name, shape)
            } else if synthetic_packed {
                synthetic_packed_decision(name, shape).dtype
            } else {
                DType::BF16
            }
        };
        let bf16 = DTypeConstraint::source_exact(DType::BF16);
        // Converted HFQM records may already carry either MQ4G256V2 (qt=44)
        // or MQ4G128V2 (qt=53).  The source checkpoint itself remains BF16.
        let quant_matrix = qwen4_quant_source();
        let quant_q8 = qwen4_q8_source();
        let quant_trunk = qwen4_trunk_matrix_source();
        // The head keeps the trunk's six/eight-bit rungs (artifacts written
        // with its earlier Q8F16 tier keep loading) but not the IU4 trunk's
        // four-bit one: four bits cost +0.014 KLD there.
        let quant_head =
            DTypeConstraint::source_from_sources(vec![DType::BF16, DType::Q8_0, DType::MQ6G256V2]);
        let model = |name: &str,
                     shape: Vec<usize>,
                     requested_dtype: DType,
                     policy: ShardPolicy,
                     source: &DTypeConstraint| {
            if !packed {
                let dtype = unpacked_dtype(name, &shape);
                return WeightEntry::model_with_dtype_constraint(
                    name,
                    shape,
                    dtype,
                    DTypeConstraint::source_exact(dtype),
                    policy,
                );
            }
            let dtype = qwen4_trunk_target(name, &shape, requested_dtype, trunk_tier);
            let source = if name == QWEN4_LM_HEAD {
                &quant_head
            } else if qwen4_trunk_matrix(name, &shape) {
                &quant_trunk
            } else if dtype == DType::Q8_0 {
                &quant_q8
            } else if qwen4_quantized_dtype(dtype) {
                &quant_matrix
            } else {
                source
            };
            WeightEntry::model_with_dtype_constraint(name, shape, dtype, source.clone(), policy)
        };
        let layer = |name: &str,
                     layer_idx: usize,
                     shape: Vec<usize>,
                     requested_dtype: DType,
                     policy: ShardPolicy,
                     source: &DTypeConstraint| {
            if !packed {
                let dtype = unpacked_dtype(name, &shape);
                return WeightEntry::layer_with_dtype_constraint(
                    name,
                    layer_idx,
                    shape,
                    dtype,
                    DTypeConstraint::source_exact(dtype),
                    policy,
                );
            }
            let dtype = qwen4_trunk_target(name, &shape, requested_dtype, trunk_tier);
            let source = if qwen4_trunk_matrix(name, &shape) {
                &quant_trunk
            } else if dtype == DType::Q8_0 {
                &quant_q8
            } else if qwen4_quantized_dtype(dtype) {
                &quant_matrix
            } else {
                source
            };
            WeightEntry::layer_with_dtype_constraint(
                name,
                layer_idx,
                shape,
                dtype,
                source.clone(),
                policy,
            )
        };
        let mtp_model = |name: &str,
                         shape: Vec<usize>,
                         requested_dtype: DType,
                         policy: ShardPolicy,
                         source: &DTypeConstraint| {
            mtp_entry(
                name,
                shape,
                requested_dtype,
                policy,
                source,
                packed,
                mtp_source,
                synthetic_packed,
            )
        };
        let mtp_layer = |name: &str,
                         _layer: usize,
                         shape: Vec<usize>,
                         requested_dtype: DType,
                         policy: ShardPolicy,
                         source: &DTypeConstraint| {
            // MTP names live in a separate model namespace, not the trunk's
            // layer placement scope.  Keep their manifest identity layerless.
            mtp_entry(
                name,
                shape,
                requested_dtype,
                policy,
                source,
                packed,
                mtp_source,
                synthetic_packed,
            )
        };
        let hidden = positive("hidden_size", config.hidden_size)?;
        let hc_wide = checked_product("hyper-connection width", config.hc_count, hidden)?;
        let hc_rank = config.hc_lowrank;
        let q_dim = checked_product("QSA q width", config.num_attention_heads, config.head_dim)?;
        let kv_dim = checked_product("QSA kv width", config.num_key_value_heads, config.head_dim)?;
        let indexer_heads = config
            .indexer_n_heads
            .checked_add(config.indexer_kv_heads)
            .ok_or_else(|| WeightError::ShapeOverflow("QSA indexer heads".to_string()))?;
        let indexer_qk_dim =
            checked_product("QSA indexer width", indexer_heads, config.indexer_head_dim)?;
        let gdn_qk = checked_product(
            "GDN key width",
            config.linear_num_key_heads,
            config.linear_key_head_dim,
        )?;
        let gdn_v = checked_product(
            "GDN value width",
            config.linear_num_value_heads,
            config.linear_value_head_dim,
        )?;
        let gdn_qkv = 2usize
            .checked_mul(gdn_qk)
            .and_then(|v| v.checked_add(gdn_v))
            .ok_or_else(|| WeightError::ShapeOverflow("GDN qkv width".to_string()))?;
        // Absent PLE declares no tensor, metadata record or state at all.
        let ple = config.ple.as_ref();
        if let Some(ple) = ple {
            if ple.shard_count == 0 || ple.shard_count > MAX_PLE_SHARD_TENSORS {
                return Err(WeightError::InvalidShape {
                    name: "PLE shard count".to_string(),
                    shape: vec![ple.shard_count],
                });
            }
        }
        // PLE channels are the hyper-connection wide width (hidden x branches);
        // the table embedding width (`embed_dim`) is only the projections' input
        // columns.  Flash-Next has embed_dim == hidden, so both coincide there.
        let ple_channels = ple.map(|_| hc_wide);
        if let Some(mtp) = &config.mtp {
            if mtp.num_hidden_layers != 1 {
                return Err(WeightError::Config(format!(
                    "native MTP inventory declares exactly one layer, got {}",
                    mtp.num_hidden_layers
                )));
            }
        }

        let mut weights = Vec::with_capacity(
            64usize
                .saturating_add(config.num_hidden_layers.saturating_mul(32))
                .saturating_add(ple.map_or(0, |ple| ple.shard_count)),
        );
        weights.push(
            model(
                QWEN4_EMBED,
                vec![config.vocab_size, hidden],
                DType::BF16,
                ShardPolicy::Pin(PinTarget::Embed),
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Embed)),
        );
        weights.push(
            model(
                "lm_head.weight",
                vec![config.vocab_size, hidden],
                DType::BF16,
                ShardPolicy::Pin(PinTarget::Output),
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );
        // MTP aliases exactly the input embedding.  The target has no MTP
        // lm_head tensor; the distinct trunk lm_head remains the output head.
        // A profile without an MTP head declares no alias.
        if config.mtp.is_some() {
            weights.push(
                model(
                    "mtp.embed_tokens.weight",
                    vec![config.vocab_size, hidden],
                    DType::BF16,
                    ShardPolicy::Tied {
                        source: QWEN4_EMBED.to_string(),
                    },
                    &bf16,
                )
                .with_placement(PlacementHint::Pin(PinTarget::Embed)),
            );
        }
        // Final HC mixer is the only trunk-level post-layer transform.  Its
        // hc_norm is the final normalization; no invented RMSNorm is added.
        weights.push(
            model(
                "model.language_model.hyper_connection_mixer.hc_norm.weight",
                vec![hc_wide],
                DType::BF16,
                ShardPolicy::Pin(PinTarget::Output),
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );
        weights.push(
            model(
                "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
                vec![hc_rank, hc_wide],
                DType::BF16,
                ShardPolicy::ColumnShard { axis: 0 },
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );
        weights.push(
            model(
                "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
                vec![hc_wide, hc_rank],
                DType::BF16,
                ShardPolicy::RowShard { axis: 1 },
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );

        for (layer_idx, kind) in config.layer_types.iter().copied().enumerate() {
            let prefix = format!("model.language_model.layers.{layer_idx}");
            push_hyper_connection_entries(
                &mut weights,
                &layer,
                &prefix,
                layer_idx,
                "attn_hyper_connection",
                config.hc_count,
                hc_wide,
                hc_rank,
                &bf16,
            );
            push_hyper_connection_entries(
                &mut weights,
                &layer,
                &prefix,
                layer_idx,
                "mlp_hyper_connection",
                config.hc_count,
                hc_wide,
                hc_rank,
                &bf16,
            );
            match kind {
                LayerType::LinearAttention => {
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.A_log"),
                        layer_idx,
                        vec![config.linear_num_value_heads],
                        DType::BF16,
                        ShardPolicy::Replicate,
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.conv1d.weight"),
                        layer_idx,
                        vec![gdn_qkv, 1, config.linear_conv_kernel_dim],
                        DType::BF16,
                        ShardPolicy::Replicate,
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.dt_bias"),
                        layer_idx,
                        vec![config.linear_num_value_heads],
                        DType::BF16,
                        ShardPolicy::Replicate,
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.in_proj_a.weight"),
                        layer_idx,
                        vec![config.linear_num_value_heads, hidden],
                        DType::BF16,
                        ShardPolicy::ColumnShard { axis: 0 },
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.in_proj_b.weight"),
                        layer_idx,
                        vec![config.linear_num_value_heads, hidden],
                        DType::BF16,
                        ShardPolicy::ColumnShard { axis: 0 },
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                        layer_idx,
                        vec![gdn_qkv, hidden],
                        DType::BF16,
                        ShardPolicy::ColumnShard { axis: 0 },
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.in_proj_z.weight"),
                        layer_idx,
                        vec![gdn_v, hidden],
                        DType::BF16,
                        ShardPolicy::ColumnShard { axis: 0 },
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.norm.weight"),
                        layer_idx,
                        vec![config.linear_value_head_dim],
                        DType::BF16,
                        ShardPolicy::Replicate,
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.out_proj.weight"),
                        layer_idx,
                        vec![hidden, gdn_v],
                        DType::BF16,
                        ShardPolicy::RowShard { axis: 1 },
                        &bf16,
                    ));
                }
                LayerType::FullAttention => {
                    push_qsa_entries(
                        &mut weights,
                        &layer,
                        &prefix,
                        layer_idx,
                        hidden,
                        q_dim,
                        kv_dim,
                        indexer_qk_dim,
                        config,
                        &bf16,
                    );
                }
            }
            push_ffn_entries(
                &mut weights,
                &layer,
                &prefix,
                layer_idx,
                hidden,
                &config.ffn,
                &bf16,
                &quant_matrix,
            )?;
            if let (Some(ple), Some(channels)) = (ple, ple_channels) {
                if ple.layer == layer_idx {
                    push_ple_entries(
                        &mut weights,
                        &layer,
                        &prefix,
                        layer_idx,
                        hidden,
                        channels,
                        ple,
                        packed,
                        synthetic_packed,
                        stock_fp8,
                        &bf16,
                    );
                }
            }
        }

        // Native MTP has one full-attention/FFN/HC layer and no PLE.  Keep the
        // source's exact names and shapes; fc_hidden is [H,H] in this
        // checkpoint even though its consumer widens the activation.  A profile
        // without an MTP head declares none of it.
        if config.mtp.is_some() {
            weights.push(mtp_model(
                "mtp.fc_embedding.weight",
                vec![hidden, hidden],
                DType::BF16,
                ShardPolicy::ColumnShard { axis: 0 },
                &bf16,
            ));
            weights.push(mtp_model(
                "mtp.fc_hidden.weight",
                vec![hidden, hidden],
                DType::BF16,
                ShardPolicy::ColumnShard { axis: 0 },
                &bf16,
            ));
            weights.push(mtp_model(
                "mtp.pre_fc_norm_embedding.weight",
                vec![hidden],
                DType::BF16,
                ShardPolicy::Replicate,
                &bf16,
            ));
            weights.push(mtp_model(
                "mtp.pre_fc_norm_hidden.weight",
                vec![hc_wide],
                DType::BF16,
                ShardPolicy::Replicate,
                &bf16,
            ));
            push_mtp_hyper_connection_entries(&mut weights, &mtp_model, hc_wide, hc_rank, &bf16);
            push_hyper_connection_entries(
                &mut weights,
                &mtp_layer,
                "mtp.layers.0",
                0,
                "attn_hyper_connection",
                config.hc_count,
                hc_wide,
                hc_rank,
                &bf16,
            );
            push_hyper_connection_entries(
                &mut weights,
                &mtp_layer,
                "mtp.layers.0",
                0,
                "mlp_hyper_connection",
                config.hc_count,
                hc_wide,
                hc_rank,
                &bf16,
            );
            push_qsa_entries(
                &mut weights,
                &mtp_layer,
                "mtp.layers.0",
                0,
                hidden,
                q_dim,
                kv_dim,
                indexer_qk_dim,
                config,
                &bf16,
            );
            push_ffn_entries(
                &mut weights,
                &mtp_layer,
                "mtp.layers.0",
                0,
                hidden,
                &config.ffn,
                &bf16,
                &quant_matrix,
            )?;
        }

        // These declarations preserve the exact I64 payload identities for
        // the qwen4_ple container.  They are deliberately not WeightEntry
        // values because no GPU tensor or float dtype may represent them.
        let metadata = match ple {
            Some(ple) => {
                let base = format!(
                    "model.language_model.layers.{}.ple.ple_embedding",
                    ple.layer
                );
                vec![
                    MetadataTensor::i64(
                        format!("{base}.layer_multipliers"),
                        vec![ple.multiplier_count],
                    ),
                    MetadataTensor::i64(
                        format!("{base}.ngram_heads_offsets"),
                        vec![ple.head_count],
                    ),
                    MetadataTensor::i64(
                        format!("{base}.ngram_heads_vocab_sizes"),
                        vec![ple.head_count],
                    ),
                ]
            }
            None => Vec::new(),
        };

        let mut state = Vec::new();
        for (layer_idx, kind) in config.layer_types.iter().copied().enumerate() {
            state.push(match kind {
                LayerType::LinearAttention => StateEntry::new(StateKind::Recurrent, layer_idx),
                // Full K/V plus raw/pooled index keys; their storage format is
                // resolved at load (`resolve_qsa_format`, memory.kv_cache).
                LayerType::FullAttention => StateEntry::new(
                    StateKind::Kv {
                        quant: "qsa-full-raw-pooled".into(),
                    },
                    layer_idx,
                ),
            });
            if kind == LayerType::LinearAttention {
                state.push(StateEntry::new(StateKind::Conv, layer_idx));
            }
        }
        // PLE is injected at its declared layer and retains its depthwise
        // convolution history, owned by the input band (the PLE layer is
        // admitted only inside band 0), so it is declared as input-pinned
        // state rather than a second Conv entry of that trunk layer.  A
        // profile without PLE declares no PLE state.
        if ple.is_some() {
            state.push(StateEntry::pinned(StateKind::Conv, PinTarget::Embed));
        }
        if config.expert_keep.is_some() {
            for entry in &mut weights {
                if entry.name.ends_with(".experts.gate_up_proj")
                    || entry.name.ends_with(".experts.down_proj")
                {
                    if let Some(ids) = config
                        .kept_expert_ids(entry.layer.unwrap_or(0), entry.name.starts_with("mtp."))
                    {
                        entry.logical_shape[0] = ids.len();
                    }
                }
            }
        }
        if let Qwen4TierRecipe::FlashNextSlab { arenas, .. } = &recipe {
            apply_expert_arenas(&mut weights, arenas)?;
        }
        let identity = Qwen4ManifestIdentity {
            profile_id: config.profile.id().to_string(),
            developer_only: config.profile.developer_only(),
            equation_source: config.profile.equation_source().as_str(),
            ffn_kind: match config.ffn {
                Qwen4FfnConfig::Dense { .. } => "dense",
                Qwen4FfnConfig::Moe { .. } => "moe",
            },
            recipe,
        };
        Ok(Self {
            expert_keep: config.expert_keep.clone(),
            identity,
            weights,
            state,
            metadata,
        })
    }

    /// Refuses a manifest (built here or assembled by hand from its public
    /// fields) that declares or admits an expert-slab dtype
    /// (`MQ3I4G128R256`/`R128`, `MQ2I4G128R256`/`R128`) for any weight or
    /// source constraint, or whose recipe names one.  Pure; no GPU is touched.
    pub fn reject_expert_slab_dtypes(&self) -> Result<(), WeightError> {
        self.identity.recipe.reject_expert_slab_dtypes()?;
        for entry in &self.weights {
            reject_expert_slab_dtype(entry.dtype)?;
            match &entry.dtype_constraint.source {
                hipfire_runtime::weight_manifest::SourceDType::Any => {}
                hipfire_runtime::weight_manifest::SourceDType::Exact(dtype) => {
                    reject_expert_slab_dtype(*dtype)?;
                }
                hipfire_runtime::weight_manifest::SourceDType::OneOf(dtypes) => {
                    for dtype in dtypes {
                        reject_expert_slab_dtype(*dtype)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn entry(&self, name: &str, layer: Option<usize>) -> Option<&WeightEntry> {
        self.weights
            .iter()
            .find(|entry| entry.name == name && entry.layer == layer)
    }

    pub fn external_entries(&self) -> impl Iterator<Item = &WeightEntry> {
        self.weights
            .iter()
            .filter(|entry| entry.residency.is_external())
    }

    pub fn resident_keys(&self) -> impl Iterator<Item = (&str, Option<usize>)> {
        self.weights
            .iter()
            .filter(|entry| !entry.residency.is_external())
            .map(|entry| (entry.name.as_str(), entry.layer))
    }

    /// Keep declaration order and source identities while removing every
    /// resource not owned by this band. Single is the unchanged declaration.
    pub fn with_band_scope(
        mut self,
        config: &Qwen4Config,
        scope: &Qwen4BandScope,
    ) -> Result<Self, WeightError> {
        scope.validate(config).map_err(WeightError::Config)?;
        if *scope == Qwen4BandScope::single(config) {
            return Ok(self);
        }
        if self.weights.iter().any(|entry| {
            matches!(
                entry.policy,
                ShardPolicy::ExpertSharded { .. } | ShardPolicy::ExpertTensorSharded { .. }
            )
        }) {
            return Err(WeightError::EpPlacement(
                "Qwen4 pipeline bands cannot be combined with expert parallelism".to_string(),
            ));
        }
        self.weights.retain(|entry| scope.owns_entry(entry));
        for entry in &mut self.weights {
            if entry.name.starts_with("mtp.") {
                entry.placement = PlacementHint::Pin(PinTarget::Output);
            }
        }
        self.state.retain(|entry| match entry.scope {
            StateScope::Layer(layer) => scope.owns_layer(layer),
            StateScope::Pin(PinTarget::Embed) => scope.input,
            StateScope::Pin(PinTarget::Output) => scope.output,
        });
        if !scope.input {
            self.metadata.clear();
        }
        Ok(self)
    }

    /// Exact ordered placement keys for an admitted local band.
    pub fn placements_for_scope(
        &self,
        config: &Qwen4Config,
        scope: &Qwen4BandScope,
        device: usize,
    ) -> Result<Vec<Qwen4Placement>, WeightError> {
        scope.validate(config).map_err(WeightError::Config)?;
        Ok(self
            .weights
            .iter()
            .filter(|entry| !entry.residency.is_external() && scope.owns_entry(entry))
            .map(|entry| Qwen4Placement {
                name: entry.name.clone(),
                layer: entry.layer,
                device,
            })
            .collect())
    }

    /// Number of source tensor records represented by this text manifest.
    /// The only non-source record is the tied MTP embedding alias (absent
    /// without an MTP head); the exact I64 PLE metadata arrays (absent
    /// without PLE) are counted from `metadata`.
    pub fn source_tensor_count(&self) -> usize {
        let aliases = self
            .weights
            .iter()
            .filter(|entry| matches!(entry.policy, ShardPolicy::Tied { .. }))
            .count();
        self.weights
            .len()
            .saturating_sub(aliases)
            .saturating_add(self.metadata.len())
    }

    /// This manifest without its external PLE n-gram shard tensors.
    ///
    /// A sidecar rows file (`hipfire_runtime::ple_sidecar`) supplies the PLE
    /// rows, so the source file's own shard tensors are neither declared nor
    /// fulfilled: the base may lack them (a split base) or carry them,
    /// shadowed and ignored (the monolithic artifact next to a derived q4
    /// sidecar).  The I64 PLE metadata records stay.  The shards are the
    /// only external-row entries of a Qwen4 manifest; their count comes from
    /// the declaration, not a constant.
    pub fn without_ple_shards(mut self) -> Self {
        self.weights.retain(|entry| !entry.residency.is_external());
        self
    }

    /// This manifest declared for one rank of an expert-parallel mesh: every
    /// routed expert stack (trunk layers and the MTP head, gate/up and down)
    /// becomes `ShardPolicy::ExpertSharded { n_experts, Stride }`, so rank `r`
    /// of `ranks` holds experts `e % ranks == r` compactly in ascending `e`
    /// (local index `e / ranks`). The entries keep their logical shape
    /// (`[n_experts, ..]`, which the manifest validator requires); the
    /// fulfilled tensor is `[ep.local_count(n_experts), ..]` (uneven splits
    /// give the low ranks one more expert). Everything else is
    /// unchanged, and a manifest that is never passed through here (ep = 1)
    /// stays byte-identical to the single-device declaration.
    pub fn with_expert_parallel(mut self, ep: Qwen4EpPlacement) -> Result<Self, WeightError> {
        if self.expert_keep.is_some() {
            return Err(WeightError::EpPlacement("qwen4.expert_keep artifacts do not support ExpertSharded/EP placement; use one device".into()));
        }
        if self.identity.recipe.is_stock_fp8() {
            return Err(WeightError::EpPlacement(format!(
                "the stock-FP8 recipe loads every routed expert fully resident on one device \
                 (P1); expert parallelism (rank {} of {}) is refused",
                ep.rank, ep.ranks
            )));
        }
        self.reject_expert_slab_dtypes()?;
        if ep.ranks < 2 || ep.rank >= ep.ranks {
            return Err(WeightError::EpPlacement(format!(
                "rank {} of {} is not an expert-parallel placement",
                ep.rank, ep.ranks
            )));
        }
        let mut sharded = 0usize;
        for entry in self
            .weights
            .iter_mut()
            .filter(|entry| is_routed_expert_tensor(&entry.name))
        {
            let experts = entry.logical_shape.first().copied().unwrap_or(0);
            if experts < ep.ranks {
                return Err(WeightError::EpPlacement(format!(
                    "{}: {experts} routed experts cannot cover {} ranks",
                    entry.name, ep.ranks
                )));
            }
            entry.policy = ShardPolicy::ExpertSharded {
                n_experts: experts,
                assign: ExpertAssign::Stride,
            };
            sharded += 1;
        }
        if sharded == 0 {
            return Err(WeightError::EpPlacement(
                "the manifest declares no routed expert tensors (dense FFN)".to_string(),
            ));
        }
        Ok(self)
    }

    /// This manifest declared for a `HIPFIRE_QWEN4_EXPERT_HOTSET` load: the
    /// 48 trunk layers' aggregate routed `gate_up`/`down` entries are owned by
    /// the expert-tier lease instead of the weight store, so they are removed
    /// (and so leave [`Self::resident_keys`]).  Everything else, the MTP
    /// head's aggregate experts included, is unchanged.
    ///
    /// Only the dense-aggregate Flash-Next declaration qualifies: exactly the
    /// expected 96 trunk entries (`[512,1280,2560]` MQ4G256V2 gate_up and
    /// `[512,2560,640]` MQ4G128V2 down, replicated, resident, one pair per
    /// layer) must be present.  The slab-arena and stock-FP8 recipes, a
    /// dense FFN, synthetic profiles and expert-parallel shards are refused.
    pub fn with_expert_tiers(mut self) -> Result<Self, WeightError> {
        self.check_expert_tiers()?;
        self.weights
            .retain(|entry| !is_trunk_routed_expert_entry(entry));
        Ok(self)
    }

    /// The validation half of [`Self::with_expert_tiers`], with no mutation:
    /// `Ok(())` exactly when `with_expert_tiers` would accept this manifest.
    pub fn check_expert_tiers(&self) -> Result<(), WeightError> {
        let refuse = |why: &str| WeightError::Config(format!("HIPFIRE_QWEN4_EXPERT_HOTSET: {why}"));
        if self.expert_keep.is_some() {
            return Err(refuse("qwen4.expert_keep artifacts do not support expert-hotset/tier placement; use the single-device packed or slab arena route"));
        }
        match &self.identity.recipe {
            Qwen4TierRecipe::FlashNextPacked { .. } => {}
            Qwen4TierRecipe::FlashNextSlab { .. } => {
                return Err(refuse(
                    "the slab-arena recipe already holds its experts in per-layer arenas",
                ))
            }
            Qwen4TierRecipe::StockFp8 => {
                return Err(refuse(
                    "the stock-FP8 recipe's qt=55 experts have no tier owner",
                ))
            }
            Qwen4TierRecipe::SyntheticSourceBf16 | Qwen4TierRecipe::SyntheticPacked => {
                return Err(refuse("synthetic profiles have no Flash-Next expert tiers"))
            }
        }
        if self.identity.ffn_kind != "moe" {
            return Err(refuse("the dense FFN declares no routed experts"));
        }
        let layers = crate::expert_slabs::TRUNK_LAYERS as usize;
        let trunk_routed = self
            .weights
            .iter()
            .filter(|entry| is_trunk_routed_expert_entry(entry))
            .count();
        if trunk_routed != 2 * layers {
            return Err(refuse(&format!(
                "the manifest declares {trunk_routed} trunk routed expert entries, expected {}",
                2 * layers
            )));
        }
        for layer in 0..layers {
            for (suffix, shape, dtype) in [
                (
                    "gate_up_proj",
                    crate::expert_slab_aggregate::GATE_UP_SHAPE,
                    ROUTED_GATE_UP_DTYPE,
                ),
                (
                    "down_proj",
                    crate::expert_slab_aggregate::DOWN_SHAPE,
                    ROUTED_DOWN_DTYPE,
                ),
            ] {
                let name = format!("model.language_model.layers.{layer}.mlp.experts.{suffix}");
                let Some(entry) = self.entry(&name, Some(layer)) else {
                    return Err(refuse(&format!("the manifest lacks {name}")));
                };
                if entry.logical_shape != shape
                    || entry.dtype != dtype
                    || entry.policy != ShardPolicy::Replicate
                    || entry.residency != WeightResidency::Resident
                {
                    return Err(refuse(&format!(
                        "{name} is not the dense aggregate {shape:?} {dtype:?} replicated \
                         resident entry (expert-parallel, host-mapped or reshaped declarations \
                         are refused)"
                    )));
                }
            }
        }
        Ok(())
    }

    /// This manifest declared for a per-expert EP load
    /// (`HIPFIRE_QWEN4_EP_EXPERT_HOTSET`, [`crate::expert_ep_units`]), for
    /// rank `ep.rank`: apply it AFTER [`Self::with_expert_parallel`] and the
    /// whole-layer placement (or on the unsharded declaration `assemble`
    /// rebuilds). Each trunk layer follows `plan`'s state of this rank:
    /// [`EpLayerState::Resident`] and [`EpLayerState::Host`] layers keep their
    /// two aggregate entries with the residency the state names (so a
    /// whole-layers plan declares exactly what `place_routed_experts` does);
    /// an [`EpLayerState::Mixed`] layer replaces them with two raw replicated
    /// arena entries, `...mlp.experts.unit_vram_arena` (resident) and
    /// `...unit_host_arena` (host-mapped), shaped `[arena entry bytes]` of the
    /// layer's slot map. The MTP head's experts are untouched (they keep the
    /// whole-layer placement's host residency).
    ///
    /// Only the dense MQ4 Flash-Next packed declaration qualifies; every other
    /// recipe, a dense FFN, kept-expert artifacts and rank/plan mismatches are
    /// refused.
    pub fn with_expert_units(
        mut self,
        ep: Qwen4EpPlacement,
        plan: &crate::expert_ep_units::EpUnitPlan,
    ) -> Result<Self, WeightError> {
        use crate::expert_ep_units::{
            EpLayerState, UnitTier, EP_EXPERT_HOTSET_ENV, UNIT_HOST_ARENA_SUFFIX,
            UNIT_VRAM_ARENA_SUFFIX,
        };
        let refuse =
            |why: String| WeightError::EpPlacement(format!("{EP_EXPERT_HOTSET_ENV}: {why}"));
        let geometry = *plan.geometry();
        if ep.ranks < 2 || ep.rank >= ep.ranks || geometry.ranks != ep.ranks {
            return Err(refuse(format!(
                "rank {} of {} does not match the plan's {} ranks",
                ep.rank, ep.ranks, geometry.ranks
            )));
        }
        if self.expert_keep.is_some() {
            return Err(refuse(
                "qwen4.expert_keep artifacts are not supported".into(),
            ));
        }
        if !matches!(
            self.identity.recipe,
            Qwen4TierRecipe::FlashNextPacked { .. }
        ) || self.identity.ffn_kind != "moe"
        {
            return Err(refuse(
                "only the dense MQ4 Flash-Next packed MoE recipe has per-expert units".into(),
            ));
        }
        let mut gate_up_seen = vec![false; geometry.layers];
        let mut down_seen = vec![false; geometry.layers];
        let mut weights = Vec::with_capacity(self.weights.len());
        for mut entry in self.weights.drain(..) {
            if !is_trunk_routed_expert_entry(&entry) {
                weights.push(entry);
                continue;
            }
            let layer = entry
                .layer
                .expect("a trunk routed expert entry has a layer");
            if layer >= geometry.layers {
                return Err(refuse(format!(
                    "{} names trunk layer {layer}, the plan covers {}",
                    entry.name, geometry.layers
                )));
            }
            let gate_up = entry.name.ends_with(".mlp.experts.gate_up_proj");
            let (dtype, seen) = if gate_up {
                (ROUTED_GATE_UP_DTYPE, &mut gate_up_seen)
            } else {
                (ROUTED_DOWN_DTYPE, &mut down_seen)
            };
            if entry.dtype != dtype
                || entry.logical_shape.first() != Some(&geometry.experts_per_layer)
                || std::mem::replace(&mut seen[layer], true)
            {
                return Err(refuse(format!(
                    "{} is not the {dtype:?} aggregate of {} experts (or is declared twice)",
                    entry.name, geometry.experts_per_layer
                )));
            }
            match plan.layer_state(ep.rank, layer) {
                EpLayerState::Resident => {
                    entry.residency = WeightResidency::Resident;
                    weights.push(entry);
                }
                EpLayerState::Host => {
                    entry.residency = WeightResidency::HostMapped;
                    weights.push(entry);
                }
                EpLayerState::Mixed { .. } => {
                    if !gate_up {
                        continue;
                    }
                    let prefix = entry
                        .name
                        .strip_suffix(".mlp.experts.gate_up_proj")
                        .expect("suffix checked above")
                        .to_string();
                    let map = plan.layer_map(ep.rank, layer);
                    for (suffix, tier, residency) in [
                        (
                            UNIT_VRAM_ARENA_SUFFIX,
                            UnitTier::Vram,
                            WeightResidency::Resident,
                        ),
                        (
                            UNIT_HOST_ARENA_SUFFIX,
                            UnitTier::Host,
                            WeightResidency::HostMapped,
                        ),
                    ] {
                        let mut arena = WeightEntry::layer_with_dtype_constraint(
                            format!("{prefix}{suffix}"),
                            layer,
                            vec![map.arena_entry_bytes(tier)],
                            DType::Raw,
                            DTypeConstraint::source_exact(DType::Raw),
                            ShardPolicy::Replicate,
                        );
                        arena.residency = residency;
                        weights.push(arena);
                    }
                }
            }
        }
        if let Some(layer) = (0..geometry.layers).find(|&l| !gate_up_seen[l] || !down_seen[l]) {
            return Err(refuse(format!(
                "the manifest lacks trunk layer {layer}'s routed expert pair"
            )));
        }
        self.weights = weights;
        Ok(self)
    }
}

/// A trunk layer's aggregate routed expert entry (`layers.{L}.mlp.experts.*`);
/// the MTP head's (`mtp.…`, no layer) is not one.
fn is_trunk_routed_expert_entry(entry: &WeightEntry) -> bool {
    entry.layer.is_some()
        && entry.name.starts_with("model.language_model.layers.")
        && is_routed_expert_tensor(&entry.name)
}

/// The placement set must be exactly the manifest's resident keys.
fn check_placement_set(
    manifest: &Qwen4Manifest,
    placements: &[Qwen4Placement],
) -> Result<(), WeightError> {
    let expected = manifest
        .resident_keys()
        .map(|(name, layer)| (name.to_string(), layer))
        .collect::<BTreeSet<_>>();
    let provided = placements
        .iter()
        .map(|placement| (placement.name.clone(), placement.layer))
        .collect::<BTreeSet<_>>();
    if expected != provided {
        return Err(WeightError::PlacementSetMismatch {
            expected: expected.len(),
            actual: provided.len(),
        });
    }
    Ok(())
}

fn mtp_entry(
    name: &str,
    shape: Vec<usize>,
    requested_dtype: DType,
    policy: ShardPolicy,
    source: &DTypeConstraint,
    packed: bool,
    mtp_source: bool,
    synthetic_packed: bool,
) -> WeightEntry {
    // A synthetic profile never takes the Flash-Next tier knobs: its MTP
    // tensors are source-exact BF16, or the synthetic packed recipe's per-role
    // decision when that recipe is declared.
    if !packed {
        let dtype = if synthetic_packed {
            synthetic_packed_decision(name, &shape).dtype
        } else {
            DType::BF16
        };
        return WeightEntry::model_with_dtype_constraint(
            name,
            shape,
            dtype,
            DTypeConstraint::source_exact(dtype),
            policy,
        );
    }
    // The existing source-tier experiment changes only rank-2 MTP entries;
    // the routed experts are rank-3, and their writer has no BF16 arm.
    if mtp_source && shape.len() == 2 {
        return WeightEntry::model_with_dtype_constraint(
            name,
            shape,
            DType::BF16,
            DTypeConstraint::source_exact(DType::BF16),
            policy,
        );
    }
    let dtype = qwen4_target_dtype(name, &shape, requested_dtype);
    // Mirror the trunk's `model` closure: the wide attention/GDN projections
    // ship at whichever tier the artifact declares (MQ6G256V2 or Q8), and a Q8
    // target otherwise takes the Q8 source set (BF16 checkpoint or an
    // already-converted Q8 payload).  `qwen4_quantized_dtype` is true for Q8_0
    // as well, so testing it first would hand every eight-bit MTP projection
    // the expert constraint (BF16 | MQ4G256V2 | MQ4G128V2) — families that are
    // a different geometry and would be misread there.
    let source = if qwen4_trunk_matrix(name, &shape) {
        qwen4_trunk_matrix_source()
    } else if dtype == DType::Q8_0 {
        qwen4_q8_source()
    } else if qwen4_quantized_dtype(dtype) {
        qwen4_quant_source()
    } else {
        source.clone()
    };
    WeightEntry::model_with_dtype_constraint(name, shape, dtype, source, policy)
}

fn push_hyper_connection_entries<F>(
    weights: &mut Vec<WeightEntry>,
    layer: &F,
    prefix: &str,
    layer_idx: usize,
    name: &str,
    hc_count: usize,
    hc_wide: usize,
    hc_rank: usize,
    bf16: &DTypeConstraint,
) where
    F: Fn(&str, usize, Vec<usize>, DType, ShardPolicy, &DTypeConstraint) -> WeightEntry,
{
    let prefix = format!("{prefix}.{name}");
    weights.push(layer(
        &format!("{prefix}.block_inject_weight.weight"),
        layer_idx,
        vec![hc_count, hc_wide],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{prefix}.hc_norm.weight"),
        layer_idx,
        vec![hc_wide],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{prefix}.input_mix_weight_down.weight"),
        layer_idx,
        vec![hc_rank, hc_wide],
        DType::BF16,
        ShardPolicy::ColumnShard { axis: 0 },
        bf16,
    ));
    weights.push(layer(
        &format!("{prefix}.input_mix_weight_up.weight"),
        layer_idx,
        vec![hc_wide, hc_rank],
        DType::BF16,
        ShardPolicy::RowShard { axis: 1 },
        bf16,
    ));
    // Final HC mixers (declared separately) have no inject operand.
}

fn push_mtp_hyper_connection_entries<M>(
    weights: &mut Vec<WeightEntry>,
    mtp_model: &M,
    hc_wide: usize,
    hc_rank: usize,
    bf16: &DTypeConstraint,
) where
    M: Fn(&str, Vec<usize>, DType, ShardPolicy, &DTypeConstraint) -> WeightEntry,
{
    weights.push(mtp_model(
        "mtp.hyper_connection_mixer.hc_norm.weight",
        vec![hc_wide],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(mtp_model(
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
        vec![hc_rank, hc_wide],
        DType::BF16,
        ShardPolicy::ColumnShard { axis: 0 },
        bf16,
    ));
    weights.push(mtp_model(
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
        vec![hc_wide, hc_rank],
        DType::BF16,
        ShardPolicy::RowShard { axis: 1 },
        bf16,
    ));
}

fn push_qsa_entries<F>(
    weights: &mut Vec<WeightEntry>,
    layer: &F,
    prefix: &str,
    layer_idx: usize,
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    indexer_qk_dim: usize,
    config: &Qwen4Config,
    bf16: &DTypeConstraint,
) where
    F: Fn(&str, usize, Vec<usize>, DType, ShardPolicy, &DTypeConstraint) -> WeightEntry,
{
    let push =
        |weights: &mut Vec<WeightEntry>, suffix: &str, shape: Vec<usize>, policy: ShardPolicy| {
            weights.push(layer(
                &format!("{prefix}.self_attn.{suffix}"),
                layer_idx,
                shape,
                DType::BF16,
                policy,
                bf16,
            ));
        };
    push(
        weights,
        "indexer.index_qk_proj.weight",
        vec![indexer_qk_dim, hidden],
        ShardPolicy::ColumnShard { axis: 0 },
    );
    push(
        weights,
        "indexer.k_layernorm.weight",
        vec![config.indexer_head_dim],
        ShardPolicy::Replicate,
    );
    push(
        weights,
        "indexer.q_layernorm.weight",
        vec![config.indexer_head_dim],
        ShardPolicy::Replicate,
    );
    push(
        weights,
        "k_norm.weight",
        vec![config.head_dim],
        ShardPolicy::Replicate,
    );
    push(
        weights,
        "k_proj.weight",
        vec![kv_dim, hidden],
        ShardPolicy::ColumnShard { axis: 0 },
    );
    push(
        weights,
        "o_proj.weight",
        vec![hidden, q_dim],
        ShardPolicy::RowShard { axis: 1 },
    );
    push(
        weights,
        "q_norm.weight",
        vec![config.head_dim],
        ShardPolicy::Replicate,
    );
    push(
        weights,
        "q_proj.weight",
        vec![2 * q_dim, hidden],
        ShardPolicy::ColumnShard { axis: 0 },
    );
    push(
        weights,
        "v_proj.weight",
        vec![kv_dim, hidden],
        ShardPolicy::ColumnShard { axis: 0 },
    );
}

/// The FFN inventory of one layer: the profile's single owner of which tensors
/// exist.  A dense FFN is three resident nonexpert matrices (gate/up/down) and
/// nothing else — no router, no expert stacks, no shared selector.  A MoE FFN
/// is the router and the two routed expert stacks, plus the gated shared
/// expert only when the profile declares one.
fn push_ffn_entries<F>(
    weights: &mut Vec<WeightEntry>,
    layer: &F,
    prefix: &str,
    layer_idx: usize,
    hidden: usize,
    ffn: &Qwen4FfnConfig,
    bf16: &DTypeConstraint,
    quant_matrix: &DTypeConstraint,
) -> Result<(), WeightError>
where
    F: Fn(&str, usize, Vec<usize>, DType, ShardPolicy, &DTypeConstraint) -> WeightEntry,
{
    let mlp = format!("{prefix}.mlp");
    match ffn {
        Qwen4FfnConfig::Dense {
            intermediate_size, ..
        } => {
            let intermediate = positive("dense intermediate_size", *intermediate_size)?;
            checked_product("dense FFN gate/up elements", intermediate, hidden)?;
            weights.push(layer(
                &format!("{mlp}.gate_proj.weight"),
                layer_idx,
                vec![intermediate, hidden],
                DType::BF16,
                ShardPolicy::ColumnShard { axis: 0 },
                bf16,
            ));
            weights.push(layer(
                &format!("{mlp}.up_proj.weight"),
                layer_idx,
                vec![intermediate, hidden],
                DType::BF16,
                ShardPolicy::ColumnShard { axis: 0 },
                bf16,
            ));
            weights.push(layer(
                &format!("{mlp}.down_proj.weight"),
                layer_idx,
                vec![hidden, intermediate],
                DType::BF16,
                ShardPolicy::RowShard { axis: 1 },
                bf16,
            ));
        }
        Qwen4FfnConfig::Moe {
            num_experts,
            intermediate_size,
            shared_intermediate_size,
            ..
        } => {
            let experts = positive("num_experts", *num_experts)?;
            let intermediate = positive("moe intermediate_size", *intermediate_size)?;
            let gate_up_rows = checked_product("routed gate_up rows", 2, intermediate)?;
            checked_product("routed gate_up elements", gate_up_rows, hidden).and_then(
                |per_expert| checked_product("routed expert elements", per_expert, experts),
            )?;
            weights.push(layer(
                &format!("{mlp}.experts.down_proj"),
                layer_idx,
                vec![experts, hidden, intermediate],
                ROUTED_DOWN_DTYPE,
                ShardPolicy::Replicate,
                quant_matrix,
            ));
            weights.push(layer(
                &format!("{mlp}.experts.gate_up_proj"),
                layer_idx,
                vec![experts, gate_up_rows, hidden],
                ROUTED_GATE_UP_DTYPE,
                ShardPolicy::Replicate,
                quant_matrix,
            ));
            weights.push(layer(
                &format!("{mlp}.gate.weight"),
                layer_idx,
                vec![experts, hidden],
                DType::BF16,
                ShardPolicy::Replicate,
                bf16,
            ));
            if let Some(shared) = shared_intermediate_size {
                let shared = positive("shared expert intermediate_size", *shared)?;
                weights.push(layer(
                    &format!("{mlp}.shared_expert.down_proj.weight"),
                    layer_idx,
                    vec![hidden, shared],
                    DType::BF16,
                    ShardPolicy::RowShard { axis: 1 },
                    bf16,
                ));
                weights.push(layer(
                    &format!("{mlp}.shared_expert.gate_proj.weight"),
                    layer_idx,
                    vec![shared, hidden],
                    DType::BF16,
                    ShardPolicy::ColumnShard { axis: 0 },
                    bf16,
                ));
                weights.push(layer(
                    &format!("{mlp}.shared_expert.up_proj.weight"),
                    layer_idx,
                    vec![shared, hidden],
                    DType::BF16,
                    ShardPolicy::ColumnShard { axis: 0 },
                    bf16,
                ));
                weights.push(layer(
                    &format!("{mlp}.shared_expert_gate.weight"),
                    layer_idx,
                    vec![1, hidden],
                    DType::BF16,
                    ShardPolicy::Replicate,
                    bf16,
                ));
            }
        }
    }
    Ok(())
}

/// One PLE block.  Every dimension (channel width, conv kernel, shard count,
/// shard rows, row width, valid rows) comes from the typed [`Qwen4PleConfig`];
/// the Flash-Next values are that config's canonical instance, not constants
/// here.  `packed` is the Flash-Next recipe: it ships Q8F16 rows and still
/// admits BF16 ones.  `synthetic_packed` ships exactly the one tier the row
/// width allows ([`synthetic_packed_ple_row_dtype`]); `stock_fp8` ships and
/// admits qt=56 `FP8E4M3PT` rows only; any other synthetic recipe ships and
/// admits BF16 rows only.
fn push_ple_entries<F>(
    weights: &mut Vec<WeightEntry>,
    layer: &F,
    prefix: &str,
    layer_idx: usize,
    hidden: usize,
    ple_channels: usize,
    config: &Qwen4PleConfig,
    packed: bool,
    synthetic_packed: bool,
    stock_fp8: bool,
    bf16: &DTypeConstraint,
) where
    F: Fn(&str, usize, Vec<usize>, DType, ShardPolicy, &DTypeConstraint) -> WeightEntry,
{
    let ple = format!("{prefix}.ple");
    weights.push(layer(
        &format!("{ple}.conv1d.weight"),
        layer_idx,
        vec![ple_channels, 1, config.conv_kernel_size],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.key_proj.weight"),
        layer_idx,
        vec![ple_channels, config.embed_dim],
        DType::BF16,
        ShardPolicy::ColumnShard { axis: 0 },
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.norm_conv.weight"),
        layer_idx,
        vec![ple_channels],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.norm_key.weight"),
        layer_idx,
        vec![ple_channels],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.norm_query.weight"),
        layer_idx,
        vec![ple_channels],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.value_proj.weight"),
        layer_idx,
        vec![hidden, config.embed_dim],
        DType::BF16,
        ShardPolicy::RowShard { axis: 1 },
        bf16,
    ));
    // The PLE row tier is declared by the artifact, not by this manifest: a
    // sealed artifact written before the Q8F16 tier carries BF16 rows and must
    // stay loadable, so both tiers are admitted and the reader derives its page
    // and read geometry from whichever one the shard declares.
    let (ple_rows, row_dtype, encoding) = if stock_fp8 {
        (
            DTypeConstraint::source_exact(DType::FP8E4M3PT),
            DType::FP8E4M3PT,
            RowEncoding::Fp8PerTensor,
        )
    } else if packed {
        (
            DTypeConstraint::source_from_sources(vec![DType::Q8_0, DType::BF16]),
            DType::Q8_0,
            RowEncoding::Q8F16,
        )
    } else if synthetic_packed && synthetic_packed_ple_row_dtype(config.row_width) == DType::Q8_0 {
        (
            DTypeConstraint::source_exact(DType::Q8_0),
            DType::Q8_0,
            RowEncoding::Q8F16,
        )
    } else {
        (bf16.clone(), DType::BF16, RowEncoding::Bf16)
    };
    for shard in 0..config.shard_count {
        let name = format!("{ple}.ple_embedding.ngram_embedding.shard_{shard}.weight");
        weights.push(
            layer(
                &name,
                layer_idx,
                vec![config.shard_rows, config.row_width],
                row_dtype,
                ShardPolicy::Replicate,
                &ple_rows,
            )
            // Q8F16 is the current writer's tier: five `[f16 scale][32 x i8]`
            // blocks per 160-value row, 170 bytes instead of the 320 a BF16 row
            // costs. The declared stride here is the default; a sealed BF16
            // shard is accepted by the same constraint and validated against its
            // own dtype-derived stride.
            .external_rows(
                encoding.encoded_row_bytes(config.row_width),
                config.shard_valid_rows(shard),
            ),
        );
    }
}

/// GPU placement requested by the typed Qwen4 assembler.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Qwen4Placement {
    pub name: String,
    pub layer: Option<usize>,
    pub device: usize,
}

#[derive(Clone, Debug)]
pub struct HyperConnectionReadWeights {
    pub hc_norm: TensorRef,
    pub input_mix_down: TensorRef,
    pub input_mix_up: TensorRef,
}

#[derive(Clone, Debug)]
pub struct HyperConnectionWeights {
    pub block_inject: TensorRef,
    pub hc_norm: TensorRef,
    pub input_mix_down: TensorRef,
    pub input_mix_up: TensorRef,
}

#[derive(Clone, Debug)]
pub struct GdnWeights {
    pub a_log: TensorRef,
    pub dt_bias: TensorRef,
    pub in_proj_a: TensorRef,
    pub in_proj_b: TensorRef,
    pub qkv: TensorRef,
    pub z: TensorRef,
    pub norm: TensorRef,
    pub conv: TensorRef,
    pub output: TensorRef,
}

#[derive(Clone, Debug)]
pub struct QsaWeights {
    pub indexer_qk: TensorRef,
    pub indexer_k_norm: TensorRef,
    pub indexer_q_norm: TensorRef,
    pub k_norm: TensorRef,
    pub k: TensorRef,
    pub output: TensorRef,
    pub q_norm: TensorRef,
    pub q: TensorRef,
    pub v: TensorRef,
}

#[derive(Clone, Debug)]
pub struct PleWeights {
    pub conv: TensorRef,
    pub key: TensorRef,
    pub norm_conv: TensorRef,
    pub norm_key: TensorRef,
    pub norm_query: TensorRef,
    pub value: TensorRef,
}

#[derive(Clone, Debug)]
pub struct MoeWeights {
    pub experts_down: TensorRef,
    pub experts_gate_up: TensorRef,
    /// `Some` on a `FlashNextSlab` trunk layer: the one raw arena holding all
    /// 512 experts (per-expert format and offsets live in the recipe's plan).
    /// `experts_gate_up` / `experts_down` then name no fulfilled tensor.
    pub expert_arena: Option<TensorRef>,
    /// `Some((vram arena, host arena))` on a MIXED layer of a per-expert EP
    /// load (`HIPFIRE_QWEN4_EP_EXPERT_HOTSET`): the two raw arenas holding this
    /// rank's owned experts of the layer. `experts_gate_up` / `experts_down`
    /// then name no fulfilled tensor (resolving them fails): ask
    /// [`Qwen4Weights::ep_unit_layer`] first. `None` everywhere else.
    pub unit_arenas: Option<(TensorRef, TensorRef)>,
    /// The pinned checkpoint carries one stacked gate/up source.  Gate and up
    /// remain distinct logical projection slices in the sealed runtime plan;
    /// do not rewrite the artifact merely to spell them as separate sources.
    pub expert_source_layout: ExpertSourceLayout,
    pub gate: TensorRef,
    pub shared_down: TensorRef,
    pub shared_gate: TensorRef,
    pub shared_up: TensorRef,
    pub shared_gate_scalar: TensorRef,
}

/// Dense gated-MLP projections: `down(silu(gate(x)) * up(x))`.
#[derive(Clone, Debug)]
pub struct DenseFfnWeights {
    pub gate: TensorRef,
    pub up: TensorRef,
    pub down: TensorRef,
}

/// The FFN block of one decoder layer (or of the MTP layer).
#[derive(Clone, Debug)]
pub enum Qwen4FfnWeights {
    Dense(DenseFfnWeights),
    Moe(MoeWeights),
}

impl Qwen4FfnWeights {
    pub fn moe(&self) -> Option<&MoeWeights> {
        match self {
            Self::Moe(moe) => Some(moe),
            Self::Dense(_) => None,
        }
    }

    pub fn dense(&self) -> Option<&DenseFfnWeights> {
        match self {
            Self::Dense(dense) => Some(dense),
            Self::Moe(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Qwen4LayerWeights {
    pub layer: usize,
    pub kind: LayerType,
    pub attn_hyper: HyperConnectionWeights,
    pub mlp_hyper: HyperConnectionWeights,
    pub attention: Option<QsaWeights>,
    pub gdn: Option<GdnWeights>,
    pub ple: Option<PleWeights>,
    pub ffn: Qwen4FfnWeights,
}

#[derive(Clone, Debug)]
pub struct Qwen4MtpWeights {
    pub fc_embedding: TensorRef,
    pub fc_hidden: TensorRef,
    pub pre_fc_norm_embedding: TensorRef,
    pub pre_fc_norm_hidden: TensorRef,
    pub attn_hyper: HyperConnectionWeights,
    pub mlp_hyper: HyperConnectionWeights,
    pub attention: QsaWeights,
    pub ffn: Qwen4FfnWeights,
    pub final_hyper: HyperConnectionReadWeights,
}

/// Typed trunk-level roots required by the native forward path.
///
/// The references are immutable logical identities.  Their resident buffers
/// are resolved through [`Qwen4Weights::resident`] so aliases remain owned by
/// the canonical load transaction and the forward path never performs source
/// I/O or an ad-hoc upload.
#[derive(Clone, Debug)]
pub struct Qwen4RootWeights {
    pub embedding: TensorRef,
    pub lm_head: TensorRef,
    pub final_hyper: HyperConnectionReadWeights,
}

/// Expert-parallel placement of one rank: routed experts are sharded by id,
/// expert `e` lives on rank `e % ranks` at compact local index `e / ranks`
/// (owned experts in ascending id order). Everything else is replicated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4EpPlacement {
    pub rank: usize,
    pub ranks: usize,
}

impl Qwen4EpPlacement {
    /// Whether this rank owns routed expert `expert`.
    pub fn owns(self, expert: usize) -> bool {
        expert % self.ranks == self.rank
    }

    /// Compact local index of an owned expert.
    pub fn local_index(self, expert: usize) -> usize {
        expert / self.ranks
    }

    /// Number of routed experts this rank owns out of `experts`
    /// (`e % ranks == rank`): the low ranks hold one more when the split is
    /// uneven (512 over 6 ranks = 86/86/85/85/85/85).
    pub fn local_count(self, experts: usize) -> usize {
        (experts + self.ranks - 1 - self.rank) / self.ranks
    }

    /// The largest per-rank expert count (rank 0's): the count a layer must
    /// budget on every rank so it is resident on all ranks or none.
    pub fn max_local_count(self, experts: usize) -> usize {
        experts.div_ceil(self.ranks)
    }
}

/// Device view of one resident stacked qt=55 `FP8E4M3B128` routed-expert
/// matrix `[experts, rows, k]` (the stock-FP8 recipe's gate_up `[512, 1280,
/// 2560]` — rows 0..640 gate_proj, 640..1280 up_proj — and down `[512, 2560,
/// 640]`).  Expert `e`'s block starts at `base + e * stride`
/// (`stride = S(R, K) = round_up(R*K + 2*ceil(R/128)*ceil(K/128), 256)`,
/// [`rdna_compute::fp8_e4m3::b128_block_stride`]): its OCP e4m3fn codes
/// `u8[R*K]` row-major at `+0`, its BF16 `s_inv` grid
/// `[ceil(R/128) * ceil(K/128)]` row-major at `+R*K`.  Addresses are device
/// pointers as `usize` (the `MoePointerEntries` convention).  Built only from
/// a fully resident (not host-mapped), 256-byte-aligned buffer that holds
/// every block, so every derived pointer is in bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8B128ExpertTable {
    /// Device address of expert 0's block (its codes).
    pub base: usize,
    pub experts: usize,
    /// Rows `R` of one expert's matrix (output features).
    pub rows: usize,
    /// Reduction width `K` of one row.
    pub k: usize,
    /// Expert block stride `S(R, K)` in bytes.
    pub stride: usize,
}

impl Fp8B128ExpertTable {
    /// Validate `tensor` (the resident handle of manifest entry `name`) as a
    /// stacked qt=55 table and describe it.
    pub fn from_tensor(name: &str, tensor: &GpuTensor) -> Result<Self, WeightError> {
        let refuse = |reason: String| {
            WeightError::DescriptorMismatch(format!("{name}: FP8E4M3B128 expert table: {reason}"))
        };
        if tensor.dtype != DType::FP8E4M3B128 {
            return Err(refuse(format!("resident dtype is {:?}", tensor.dtype)));
        }
        let [experts, rows, k] = tensor.shape[..] else {
            return Err(refuse(format!(
                "shape {:?} is not [experts, rows, K]",
                tensor.shape
            )));
        };
        if experts == 0 || rows == 0 || k == 0 {
            return Err(refuse(format!(
                "shape {:?} has a zero dimension",
                tensor.shape
            )));
        }
        let stride = rdna_compute::fp8_e4m3::b128_block_stride(rows, k)
            .ok_or_else(|| refuse("block stride overflows".to_string()))?;
        let bytes = rdna_compute::fp8_e4m3::b128_tensor_bytes(experts, rows, k)
            .ok_or_else(|| refuse("payload size overflows".to_string()))?;
        if tensor.buf.size() < bytes {
            return Err(refuse(format!(
                "buffer holds {} bytes, {experts} blocks of {stride} need {bytes}",
                tensor.buf.size()
            )));
        }
        if tensor.buf.is_host_mapped() {
            return Err(refuse(
                "it is host-mapped; the stock-FP8 recipe requires every routed expert fully \
                 resident in device memory (P1)"
                    .to_string(),
            ));
        }
        let base = tensor.buf.as_ptr() as usize;
        if base % rdna_compute::fp8_e4m3::B128_ALIGN != 0 {
            return Err(refuse(format!(
                "base 0x{base:x} is not {}-byte aligned",
                rdna_compute::fp8_e4m3::B128_ALIGN
            )));
        }
        Ok(Self {
            base,
            experts,
            rows,
            k,
            stride,
        })
    }

    /// Device address of expert `expert`'s block — its codes (`base + e*S`);
    /// `None` past the last expert.
    pub fn block_ptr(&self, expert: usize) -> Option<usize> {
        (expert < self.experts).then(|| self.base + expert * self.stride)
    }

    /// Device address of expert `expert`'s codes (same as [`Self::block_ptr`]).
    pub fn codes_ptr(&self, expert: usize) -> Option<usize> {
        self.block_ptr(expert)
    }

    /// Device address of expert `expert`'s BF16 `s_inv` grid (`block + R*K`).
    pub fn scales_ptr(&self, expert: usize) -> Option<usize> {
        Some(self.block_ptr(expert)? + self.scales_offset())
    }

    /// Byte offset of the scale grid inside one block (`R * K`).
    pub fn scales_offset(&self) -> usize {
        self.rows * self.k
    }

    /// `(ceil(R/128), ceil(K/128))`: the row-major `s_inv` grid of one block.
    pub fn scale_grid(&self) -> (usize, usize) {
        rdna_compute::fp8_e4m3::b128_scale_dims(self.rows, self.k)
    }

    /// Every expert's block address in expert order: the per-expert pointer
    /// table a grouped launch uploads (codes at `+0`, scales at `+R*K`).
    pub fn block_entries(&self) -> Vec<usize> {
        (0..self.experts)
            .map(|expert| self.base + expert * self.stride)
            .collect()
    }
}

/// The two qt=55 routed-expert tables of one MoE layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8MoeExpertTables {
    /// `[E, 2*intermediate, hidden]`: gate rows first, then up rows.
    pub gate_up: Fp8B128ExpertTable,
    /// `[E, hidden, intermediate]`.
    pub down: Fp8B128ExpertTable,
}

/// The per-expert EP state of one rank's weights: the mesh's plan, this rank
/// and the slot maps of its mixed trunk layers.
struct EpUnitState {
    plan: std::sync::Arc<crate::expert_ep_units::EpUnitPlan>,
    rank: usize,
    maps: HashMap<usize, crate::expert_ep_units::EpUnitLayerMap>,
}

/// One MIXED trunk layer of this rank ([`Qwen4Weights::ep_unit_layer`]).
pub struct EpUnitLayer<'a> {
    /// This rank's slot map for the layer.
    pub map: &'a crate::expert_ep_units::EpUnitLayerMap,
    /// The device arena (`unit_vram_arena`).
    pub vram: &'a GpuTensor,
    /// The pinned host-mapped arena (`unit_host_arena`).
    pub host: &'a GpuTensor,
}

/// Published resident ownership.  External PLE descriptors remain in the
/// attached transaction census; no fake handle is stored here.
pub struct Qwen4Weights {
    pub manifest: Qwen4Manifest,
    pub taken: Vec<TakenWeight>,
    /// `taken` slots by tensor name: `resident` runs ~1000 times per forward,
    /// and a linear scan over every taken weight cost ~3 ms per decode token.
    taken_by_name: HashMap<String, Vec<usize>>,
    pub root: Qwen4RootWeights,
    /// Compact owned layers in absolute order; use [`Self::layer`] for an
    /// absolute layer ID instead of indexing this vector.
    pub layer_refs: Vec<Qwen4LayerWeights>,
    /// `Some` iff native MTP is declared and this scope owns the output.
    pub mtp: Option<Qwen4MtpWeights>,
    /// `Some` iff the routed experts were uploaded expert-sharded for one
    /// rank of an expert-parallel mesh (compact tensors of `local_count(512)`
    /// experts); `None` on the single-device route.
    ep: Option<Qwen4EpPlacement>,
    pub expert_tiers: Option<crate::expert_tier_owners::ExpertTierLease>,
    /// `Some` on a per-expert EP load (`HIPFIRE_QWEN4_EP_EXPERT_HOTSET`): the
    /// mesh's plan, this rank and the slot maps of its mixed layers.
    ep_units: Option<EpUnitState>,
    scope: Qwen4BandScope,
    device: usize,
    /// `Some((rank, parts))` once [`Self::shard_dense_ffn`] kept that slice.
    tp_shard: Option<(usize, usize)>,
}

impl Qwen4Weights {
    /// Publish the resident weights of `tx` for `config` under the manifest
    /// the artifact was admitted with: `recipe` is `Qwen4HfqmArtifact::recipe()`
    /// (the manifest identity's recipe), so the declaration is rebuilt
    /// exactly as admission built it instead of from `config` alone.  For
    /// Flash-Next the recipe is the single `FlashNextPacked` carrying the
    /// trunk-tier and MTP-source knobs resolved at admission, which is what
    /// [`Qwen4Manifest::build`] resolved from the same environment before.  A
    /// recipe the profile cannot be declared under is refused by
    /// [`Qwen4Manifest::build_with_recipe`].
    pub fn assemble(
        tx: &mut WeightLoadTransaction,
        config: &Qwen4Config,
        recipe: &Qwen4TierRecipe,
        placements: &[Qwen4Placement],
        scope: &Qwen4BandScope,
    ) -> Result<Self, WeightError> {
        Self::assemble_impl(tx, config, recipe, placements, scope, None, None)
    }

    /// [`Self::assemble`] for a `HIPFIRE_QWEN4_EXPERT_HOTSET` load: the trunk
    /// layers' aggregate routed experts are owned by `lease`, not the
    /// transaction, so the manifest is [`Qwen4Manifest::with_expert_tiers`]
    /// and `placements` must cover exactly its remaining resident keys.  The
    /// lease is retained by the returned weights ([`Self::expert_tiers`]).
    pub fn assemble_with_expert_tiers(
        tx: &mut WeightLoadTransaction,
        config: &Qwen4Config,
        recipe: &Qwen4TierRecipe,
        placements: &[Qwen4Placement],
        scope: &Qwen4BandScope,
        lease: crate::expert_tier_owners::ExpertTierLease,
    ) -> Result<Self, WeightError> {
        Self::assemble_impl(tx, config, recipe, placements, scope, Some(lease), None)
    }

    /// [`Self::assemble`] for a per-expert EP load
    /// (`HIPFIRE_QWEN4_EP_EXPERT_HOTSET`): rank `rank` of `plan`'s mesh. The
    /// manifest is [`Qwen4Manifest::with_expert_units`] (each mixed trunk layer's
    /// aggregates replaced by its two arena entries) and `placements` must cover
    /// exactly its resident keys. The mesh's plan and the rank's slot maps are
    /// retained by the returned weights ([`Self::ep_unit_plan`],
    /// [`Self::ep_unit_layer`]). The EP placement is the plan's even when
    /// every trunk layer is mixed (no aggregate carries a projection tag then).
    pub fn assemble_with_ep_units(
        tx: &mut WeightLoadTransaction,
        config: &Qwen4Config,
        recipe: &Qwen4TierRecipe,
        placements: &[Qwen4Placement],
        scope: &Qwen4BandScope,
        plan: std::sync::Arc<crate::expert_ep_units::EpUnitPlan>,
        rank: usize,
    ) -> Result<Self, WeightError> {
        Self::assemble_impl(
            tx,
            config,
            recipe,
            placements,
            scope,
            None,
            Some((plan, rank)),
        )
    }

    fn assemble_impl(
        tx: &mut WeightLoadTransaction,
        config: &Qwen4Config,
        recipe: &Qwen4TierRecipe,
        placements: &[Qwen4Placement],
        scope: &Qwen4BandScope,
        expert_tiers: Option<crate::expert_tier_owners::ExpertTierLease>,
        ep_units: Option<(std::sync::Arc<crate::expert_ep_units::EpUnitPlan>, usize)>,
    ) -> Result<Self, WeightError> {
        let mut manifest = assembly_manifest(config, recipe)?;
        if expert_tiers.is_some() {
            manifest = manifest.with_expert_tiers()?;
        }
        if let Some((plan, rank)) = &ep_units {
            if expert_tiers.is_some() {
                return Err(WeightError::EpPlacement(
                    "per-expert EP units cannot be combined with the single-device expert tiers"
                        .to_string(),
                ));
            }
            if *scope != Qwen4BandScope::single(config) {
                return Err(WeightError::EpPlacement(
                    "Qwen4 pipeline bands cannot be combined with expert parallelism".to_string(),
                ));
            }
            manifest = manifest.with_expert_units(
                Qwen4EpPlacement {
                    rank: *rank,
                    ranks: plan.geometry().ranks,
                },
                plan,
            )?;
        }
        let placements = if *scope == Qwen4BandScope::single(config) {
            Cow::Borrowed(placements)
        } else {
            Cow::Owned(
                placements
                    .iter()
                    .filter(|placement| {
                        manifest
                            .entry(&placement.name, placement.layer)
                            .is_none_or(|entry| scope.owns_entry(entry))
                    })
                    .cloned()
                    .collect::<Vec<_>>(),
            )
        };
        let manifest = manifest.with_band_scope(config, scope)?;
        check_placement_set(&manifest, &placements)?;
        let device = placements.first().map_or(0, |placement| placement.device);
        if placements
            .iter()
            .any(|placement| placement.device != device)
        {
            return Err(WeightError::Config(
                "Qwen4 band assembly requires one owning device".to_string(),
            ));
        }
        let ep = ep_from_projections(&manifest, |name, layer| tx.projection(name, layer, device))?;
        let ep = match &ep_units {
            Some((plan, rank)) => {
                let want = Qwen4EpPlacement {
                    rank: *rank,
                    ranks: plan.geometry().ranks,
                };
                if ep.is_some_and(|found| found != want) {
                    return Err(WeightError::EpPlacement(format!(
                        "the fulfilled projections carry placement {ep:?}, the unit plan's is {want:?}"
                    )));
                }
                Some(want)
            }
            None => ep,
        };
        if ep.is_some() && *scope != Qwen4BandScope::single(config) {
            return Err(WeightError::EpPlacement(
                "Qwen4 pipeline bands cannot be combined with expert parallelism".to_string(),
            ));
        }
        if recipe.is_stock_fp8() {
            // Checked before anything is taken, so a refusal leaves every
            // handle with the transaction for its normal rollback.
            if let Some(ep) = ep {
                return Err(WeightError::EpPlacement(format!(
                    "the stock-FP8 recipe loads every routed expert on one device (P1); \
                     expert-parallel rank {} of {} is refused",
                    ep.rank, ep.ranks
                )));
            }
            for entry in manifest
                .weights
                .iter()
                .filter(|entry| is_routed_expert_tensor(&entry.name))
            {
                match tx.get(&entry.name, entry.layer, device) {
                    Some(WeightHandle::Resident(tensor)) => {
                        Fp8B128ExpertTable::from_tensor(&entry.name, tensor)?;
                    }
                    _ => {
                        return Err(WeightError::MissingResident {
                            name: entry.name.clone(),
                            layer: entry.layer,
                            device,
                        })
                    }
                }
            }
        }
        let mut assembly = tx.begin_assembly();
        for placement in placements.iter() {
            if assembly
                .take(&placement.name, placement.layer, placement.device)
                .is_none()
            {
                return Err(WeightError::MissingPlacement {
                    name: placement.name.clone(),
                    layer: placement.layer,
                    device: placement.device,
                });
            }
        }
        let assembly = assembly.commit();
        let mut root = build_root_refs(config)?;
        if !scope.input && scope.output && config.mtp.is_some() {
            root.embedding.name = "mtp.embed_tokens.weight".to_string();
        }
        let mut layer_refs = if *scope == Qwen4BandScope::single(config) {
            build_layer_refs(config)?
        } else {
            build_layer_refs_scoped(config, scope)?
        };
        if let Qwen4TierRecipe::FlashNextSlab { arenas, .. } = recipe {
            for layer in &mut layer_refs {
                let Qwen4FfnWeights::Moe(moe) = &mut layer.ffn else {
                    continue;
                };
                let plan = arenas.layer(layer.layer).ok_or_else(|| {
                    WeightError::DescriptorMismatch(format!(
                        "expert arena plan has no layer {}",
                        layer.layer
                    ))
                })?;
                let prefix = moe
                    .experts_gate_up
                    .name
                    .strip_suffix(".mlp.experts.gate_up_proj")
                    .ok_or_else(|| {
                        WeightError::DescriptorMismatch(moe.experts_gate_up.name.clone())
                    })?;
                moe.expert_arena = Some(TensorRef::new(
                    format!("{prefix}{}", crate::expert_arena::ARENA_SUFFIX),
                    TensorRole::RoutedArena,
                    Some(layer.layer),
                    vec![plan.bytes],
                    DType::Raw,
                )?);
            }
        }
        let ep_units = match ep_units {
            Some((plan, rank)) => {
                use crate::expert_ep_units::{UNIT_HOST_ARENA_SUFFIX, UNIT_VRAM_ARENA_SUFFIX};
                let mut maps = HashMap::new();
                for layer in &mut layer_refs {
                    let Qwen4FfnWeights::Moe(moe) = &mut layer.ffn else {
                        continue;
                    };
                    if layer.layer >= plan.geometry().layers
                        || !matches!(
                            plan.layer_state(rank, layer.layer),
                            crate::expert_ep_units::EpLayerState::Mixed { .. }
                        )
                    {
                        continue;
                    }
                    let map = plan.layer_map(rank, layer.layer);
                    let prefix = moe
                        .experts_gate_up
                        .name
                        .strip_suffix(".mlp.experts.gate_up_proj")
                        .ok_or_else(|| {
                            WeightError::DescriptorMismatch(moe.experts_gate_up.name.clone())
                        })?
                        .to_string();
                    let arena = |suffix: &str, tier| {
                        TensorRef::new(
                            format!("{prefix}{suffix}"),
                            TensorRole::RoutedArena,
                            Some(layer.layer),
                            vec![map.arena_entry_bytes(tier)],
                            DType::Raw,
                        )
                    };
                    moe.unit_arenas = Some((
                        arena(
                            UNIT_VRAM_ARENA_SUFFIX,
                            crate::expert_ep_units::UnitTier::Vram,
                        )?,
                        arena(
                            UNIT_HOST_ARENA_SUFFIX,
                            crate::expert_ep_units::UnitTier::Host,
                        )?,
                    ));
                    maps.insert(layer.layer, map);
                }
                Some(EpUnitState { plan, rank, maps })
            }
            None => None,
        };
        let mtp = config
            .mtp
            .as_ref()
            .filter(|_| scope.output)
            .map(|_| build_mtp_refs(config))
            .transpose()?;
        let taken = assembly.finalize();
        let mut taken_by_name: HashMap<String, Vec<usize>> = HashMap::new();
        for (slot, weight) in taken.iter().enumerate() {
            taken_by_name
                .entry(weight.key.name.clone())
                .or_default()
                .push(slot);
        }
        Ok(Self {
            manifest,
            taken,
            taken_by_name,
            root,
            layer_refs,
            mtp,
            ep,
            expert_tiers,
            scope: scope.clone(),
            device,
            tp_shard: None,
            ep_units,
        })
    }

    /// Absolute half-open trunk interval owned by this rank.
    pub fn owned_layers(&self) -> Range<usize> {
        self.scope.band.clone()
    }

    /// The validated endpoint and trunk ownership used during assembly.
    pub fn band_scope(&self) -> &Qwen4BandScope {
        &self.scope
    }

    /// `(rank, parts)` of the dense FFN slice these weights hold, `None` when
    /// unsharded.
    pub fn tp_shard(&self) -> Option<(usize, usize)> {
        self.tp_shard
    }

    /// Resolve an absolute layer ID to its compact rank-local reference.
    pub fn layer(&self, absolute: usize) -> Option<&Qwen4LayerWeights> {
        let local = absolute.checked_sub(self.scope.band.start)?;
        self.layer_refs
            .get(local)
            .filter(|layer| layer.layer == absolute)
    }

    pub fn expert_tiers(&self) -> Option<&crate::expert_tier_owners::ExpertTierLease> {
        self.expert_tiers.as_ref()
    }

    /// One tier layer through the canonical (`Resident`) read view.
    pub fn expert_tier_layer(
        &self,
        layer: usize,
    ) -> Option<crate::expert_tier_owners::LayerExpertTierView<'_>> {
        self.expert_tier_layer_view(layer, crate::expert_tier_owners::ExpertReadView::Resident)
    }

    /// One tier layer through the chosen read view; `None` without a tier
    /// lease, for an out-of-range layer, or for `BorrowPrefill` on a lease
    /// that does not share.
    pub fn expert_tier_layer_view(
        &self,
        layer: usize,
        view: crate::expert_tier_owners::ExpertReadView,
    ) -> Option<crate::expert_tier_owners::LayerExpertTierView<'_>> {
        self.expert_tiers.as_ref()?.layer_view(layer, view)
    }

    /// The expert-parallel placement these weights were assembled for, or
    /// `None` on the single-device route.
    pub fn ep_placement(&self) -> Option<Qwen4EpPlacement> {
        self.ep
    }

    /// The per-expert EP plan of the whole mesh (every rank) this load was
    /// assembled under, shared by all ranks of the load; `None` without
    /// `HIPFIRE_QWEN4_EP_EXPERT_HOTSET`.
    pub fn ep_unit_plan(&self) -> Option<&std::sync::Arc<crate::expert_ep_units::EpUnitPlan>> {
        self.ep_units.as_ref().map(|state| &state.plan)
    }

    /// One MIXED trunk layer of this rank: its slot map and the two arenas
    /// (device, pinned host). `Ok(None)` for a layer that is not mixed on this
    /// rank (whole-layer compact path) and for a load with no unit plan. The
    /// aggregate `MoeWeights::experts_*` of a mixed layer do not resolve, so
    /// every consumer asks here first.
    pub fn ep_unit_layer(&self, layer: usize) -> Result<Option<EpUnitLayer<'_>>, WeightError> {
        let Some(state) = &self.ep_units else {
            return Ok(None);
        };
        let Some(map) = state.maps.get(&layer) else {
            return Ok(None);
        };
        debug_assert_eq!(map.rank(), state.rank);
        let (vram, host) = self
            .layer(layer)
            .and_then(|weights| weights.ffn.moe())
            .and_then(|moe| moe.unit_arenas.as_ref())
            .ok_or_else(|| {
                WeightError::DescriptorMismatch(format!(
                    "mixed EP unit layer {layer} has no arena references"
                ))
            })?;
        Ok(Some(EpUnitLayer {
            map,
            vram: self.resident(vram)?,
            host: self.resident(host)?,
        }))
    }

    /// Resolve one immutable typed reference to its resident GPU buffer.
    ///
    /// Resolve only on the assembled band's owning logical device. Local ties
    /// follow their source; separately materialized endpoint ties resolve their
    /// own handle. Neither path performs I/O or allocation.
    pub fn resident(&self, reference: &TensorRef) -> Result<&GpuTensor, WeightError> {
        self.resident_named(&reference.name, reference.layer, self.device, 0)
    }

    /// [`Self::resident`] by manifest name and layer (the decode-share
    /// weight-shadow region plan names its sources, not typed references).
    pub fn resident_by_name(
        &self,
        name: &str,
        layer: Option<usize>,
    ) -> Result<&GpuTensor, WeightError> {
        self.resident_named(name, layer, self.device, 0)
    }

    fn resident_named(
        &self,
        name: &str,
        layer: Option<usize>,
        device: usize,
        depth: usize,
    ) -> Result<&GpuTensor, WeightError> {
        if depth > self.taken.len() {
            return Err(WeightError::DescriptorMismatch(format!(
                "{name}: resident alias cycle"
            )));
        }
        let Some(taken) = self
            .taken_by_name
            .get(name)
            .into_iter()
            .flatten()
            .map(|&slot| &self.taken[slot])
            .find(|taken| taken.key.layer == layer && taken.key.device == device)
        else {
            return Err(WeightError::MissingResident {
                name: name.to_string(),
                layer,
                device,
            });
        };
        match &taken.handle {
            WeightHandle::Resident(tensor) => Ok(tensor),
            WeightHandle::Alias(source) => {
                let source = source.clone();
                self.resident_named(&source, layer, device, depth + 1)
            }
        }
    }

    /// The resident qt=55 table behind a routed-expert reference
    /// (`MoeWeights::experts_gate_up` / `experts_down`) of a stock-FP8 load.
    pub fn fp8_expert_table(
        &self,
        reference: &TensorRef,
    ) -> Result<Fp8B128ExpertTable, WeightError> {
        Fp8B128ExpertTable::from_tensor(&reference.name, self.resident(reference)?)
    }

    /// Both qt=55 tables of one MoE layer of a stock-FP8 load.
    pub fn fp8_moe_expert_tables(
        &self,
        moe: &MoeWeights,
    ) -> Result<Fp8MoeExpertTables, WeightError> {
        Ok(Fp8MoeExpertTables {
            gate_up: self.fp8_expert_table(&moe.experts_gate_up)?,
            down: self.fp8_expert_table(&moe.experts_down)?,
        })
    }

    pub fn external_descriptor<'a>(
        tx: &'a WeightLoadTransaction,
        placement: &Qwen4Placement,
    ) -> Option<&'a SourceRangeDescriptor> {
        tx.external_descriptor(&placement.name, placement.layer, placement.device)
    }

    /// Explicit GPU teardown for handles finalized from the canonical
    /// transaction.  The attached transaction census is drained separately,
    /// after readers and these resident handles are gone.
    pub fn free_gpu(self, gpu: &mut Gpu) -> hip_bridge::HipResult<()> {
        free_taken_handles(self.taken, |tensor| gpu.free_tensor(tensor))
    }

    /// Load-time precision experiments: `HIPFIRE_QWEN4_REQUANT="pat=fmt;..."`
    /// replaces every resident rank-2 weight whose name contains `pat` (first
    /// matching rule wins) with a requantized copy, `fmt` = `mq2` .. `mq6`
    /// (MQ G256 V2) | `q8` (from BF16 only) | `qt44:DIR` (pre-encoded QT44
    /// bytes from `DIR/<tensor name>.bin` where that file exists, e.g. an
    /// offline GPTQ solve; it must hold exactly the `m x k` matrix). Decode
    /// and prefill both
    /// bind it.
    pub fn requant_from_env(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        let Ok(spec) = hipfire_config::developer_var("HIPFIRE_QWEN4_REQUANT") else {
            return Ok(());
        };
        let mut rules = Vec::new();
        for rule in spec.split(';').filter(|rule| !rule.trim().is_empty()) {
            let (pattern, format) = rule
                .split_once('=')
                .ok_or_else(|| format!("HIPFIRE_QWEN4_REQUANT rule {rule:?} is not pat=fmt"))?;
            let (target, dir) = match format.trim() {
                "mq2" => (DType::MQ2G256V2, None),
                "mq3" => (DType::MQ3G256V2, None),
                "mq4" => (DType::MQ4G256V2, None),
                "mq5" => (DType::MQ5G256V2, None),
                "mq6" => (DType::MQ6G256V2, None),
                "q8" => (DType::Q8_0, None),
                other => match other.strip_prefix("qt44:") {
                    Some(dir) => (DType::MQ4G256V2, Some(std::path::PathBuf::from(dir))),
                    None => return Err(format!("HIPFIRE_QWEN4_REQUANT format {other:?}")),
                },
            };
            rules.push((pattern.trim().to_string(), target, dir));
        }
        let (mut count, mut before, mut after) = (0usize, 0usize, 0usize);
        for taken in &mut self.taken {
            let WeightHandle::Resident(tensor) = &taken.handle else {
                continue;
            };
            let Some((_, target, dir)) = rules
                .iter()
                .find(|(pattern, _, _)| taken.key.name.contains(pattern.as_str()))
            else {
                continue;
            };
            let target = *target;
            let [m, k] = taken.projection.logical_shape[..] else {
                continue;
            };
            let replacement = if let Some(dir) = dir {
                // A matching tensor without a file keeps its payload (the
                // MTP attention shares the trunk's names).
                let path = dir.join(format!("{}.bin", taken.key.name));
                if !path.exists() {
                    continue;
                }
                let bytes =
                    std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
                if qt44_matrix_bytes(m, k) != Some(bytes.len()) {
                    return Err(format!(
                        "{}: {} bytes is not a {m}x{k} QT44 matrix",
                        path.display(),
                        bytes.len()
                    ));
                }
                Replacement::Qt44(Cow::Owned(bytes))
            } else {
                if tensor.dtype == target {
                    continue;
                }
                let produced = if target == DType::Q8_0 {
                    gpu.quantize_bf16_q8_0(tensor, m, k)
                } else {
                    gpu.requant_g256(tensor, m, k, target)
                }
                .map_err(|error| format!("requant {} -> {target:?}: {error}", taken.key.name))?;
                Replacement::Tensor(produced)
            };
            let (old_bytes, new_bytes) = replace_resident(gpu, taken, replacement)?;
            count += 1;
            before += old_bytes;
            after += new_bytes;
        }
        eprintln!(
            "[qwen4] HIPFIRE_QWEN4_REQUANT {spec:?}: {count} tensors, {before} -> {after} bytes"
        );
        Ok(())
    }

    /// Replace the QT44 targets of a verified trunk overlay
    /// (`hipfire_runtime::trunk_overlay`), exactly as the `qt44:` rule of
    /// [`Self::requant_from_env`] does (same upload, dtype and free).
    ///
    /// Everything is checked first: each payload must name exactly one
    /// resident weight whose logical shape is the payload's `[m, k]` and whose
    /// byte count is the QT44 size.  A failed check replaces nothing and is
    /// [`TrunkOverlayOutcome::Refused`] (the base weights run).  An upload or
    /// free error once replacement started is an `Err`.
    pub fn apply_trunk_overlay(
        &mut self,
        gpu: &mut Gpu,
        payloads: &[TrunkOverlayPayload],
    ) -> Result<TrunkOverlayOutcome, String> {
        let mut slots = Vec::with_capacity(payloads.len());
        for payload in payloads {
            let mut resident = self
                .taken_by_name
                .get(&payload.name)
                .into_iter()
                .flatten()
                .copied()
                .filter(|&slot| matches!(self.taken[slot].handle, WeightHandle::Resident(_)));
            let (Some(slot), None) = (resident.next(), resident.next()) else {
                return Ok(TrunkOverlayOutcome::Refused(format!(
                    "tensor {} does not match exactly one resident weight",
                    payload.name
                )));
            };
            let [m, k] = payload.shape;
            let projection = &self.taken[slot].projection;
            if projection.logical_shape != [m, k] {
                return Ok(TrunkOverlayOutcome::Refused(format!(
                    "tensor {}: the model's weight is {:?}, overlay is [{m}, {k}]",
                    payload.name, projection.logical_shape
                )));
            }
            if qt44_matrix_bytes(m, k) != Some(payload.bytes.len()) {
                return Ok(TrunkOverlayOutcome::Refused(format!(
                    "tensor {}: {} bytes is not a {m}x{k} QT44 matrix",
                    payload.name,
                    payload.bytes.len()
                )));
            }
            if slots.contains(&slot) {
                return Ok(TrunkOverlayOutcome::Refused(format!(
                    "tensor {} is listed twice",
                    payload.name
                )));
            }
            slots.push(slot);
        }
        let (mut before, mut after) = (0usize, 0usize);
        for (payload, &slot) in payloads.iter().zip(&slots) {
            let (old_bytes, new_bytes) = replace_resident(
                gpu,
                &mut self.taken[slot],
                Replacement::Qt44(Cow::Borrowed(&payload.bytes[..])),
            )?;
            before += old_bytes;
            after += new_bytes;
        }
        Ok(TrunkOverlayOutcome::Applied(TrunkOverlayApplied {
            tensors: payloads.len(),
            before_bytes: before,
            after_bytes: after,
        }))
    }

    /// Keep only TP rank `rank` of `parts` of every dense FFN layer: gate/up
    /// rows and down columns along the intermediate, as their manifest
    /// policies (`ColumnShard { axis: 0 }`, `RowShard { axis: 1 }`) declare,
    /// so the forward runs each layer at `intermediate / parts` and writes
    /// this rank's partial sum. Every other weight stays whole (replicated).
    /// The dense FFN is source-exact BF16 (unpacked), so a slice is plain
    /// bytes. Run after any load-time replacement (requant, trunk overlay:
    /// they read the source projection shape) and before the first forward
    /// (the Gpu keys its lazy F16 weight shadows on the device pointer).
    pub fn shard_dense_ffn(
        &mut self,
        gpu: &mut Gpu,
        rank: usize,
        parts: usize,
    ) -> Result<(), String> {
        if parts < 2 || rank >= parts || self.tp_shard.is_some() {
            return Err(format!(
                "TP rank {rank} of {parts} parts: needs two or more parts, a rank inside them \
                 and unsharded weights (already {:?})",
                self.tp_shard
            ));
        }
        for index in 0..self.layer_refs.len() {
            let Qwen4FfnWeights::Dense(dense) = &self.layer_refs[index].ffn else {
                continue;
            };
            let mut shards = Vec::with_capacity(3);
            for reference in [&dense.gate, &dense.up, &dense.down] {
                let context = &reference.name;
                let policy = &self
                    .manifest
                    .entry(&reference.name, reference.layer)
                    .ok_or_else(|| format!("{context}: no manifest entry"))?
                    .policy;
                let tensor = self.resident(reference).map_err(|e| e.to_string())?;
                let [rows, cols] = reference.shape[..] else {
                    return Err(format!(
                        "{context}: shape {:?} is not a matrix",
                        reference.shape
                    ));
                };
                if tensor.dtype != DType::BF16 {
                    return Err(format!("{context}: {:?} is not BF16", tensor.dtype));
                }
                let split = match policy {
                    ShardPolicy::ColumnShard { axis: 0 } => rows,
                    ShardPolicy::RowShard { axis: 1 } => cols,
                    other => {
                        return Err(format!("{context}: policy {other:?} is not an FFN split"))
                    }
                };
                if split % parts != 0 {
                    return Err(format!(
                        "{context}: intermediate {split} is not divisible by {parts}"
                    ));
                }
                let (start, end) = (rank * split / parts, (rank + 1) * split / parts);
                let bytes = gpu.download_raw_bytes(tensor).map_err(|e| e.to_string())?;
                let (bytes, shape) = if split == rows {
                    let rows_bytes =
                        hipfire_runtime::tp_shard::slice_uniform_rows(&bytes, rows, start, end)?;
                    (rows_bytes, vec![end - start, cols])
                } else {
                    let cols_bytes = hipfire_runtime::tp_shard::slice_uniform_group_cols(
                        &bytes, rows, cols, start, end, 1,
                    )?;
                    (cols_bytes, vec![rows, end - start])
                };
                let slot = self
                    .taken_by_name
                    .get(&reference.name)
                    .into_iter()
                    .flatten()
                    .copied()
                    .find(|&slot| {
                        let key = &self.taken[slot].key;
                        key.layer == reference.layer && key.device == self.device
                    })
                    .ok_or_else(|| format!("{context}: no resident handle"))?;
                shards.push((slot, bytes, shape));
            }
            // Every slice of the layer is checked before any is installed.
            for (slot, bytes, shape) in &shards {
                let mut tensor = gpu.upload_raw(bytes, shape).map_err(|e| e.to_string())?;
                tensor.dtype = DType::BF16;
                replace_resident(gpu, &mut self.taken[*slot], Replacement::Tensor(tensor))?;
            }
            if let Qwen4FfnWeights::Dense(dense) = &mut self.layer_refs[index].ffn {
                for (reference, (_, _, shape)) in [&mut dense.gate, &mut dense.up, &mut dense.down]
                    .into_iter()
                    .zip(shards)
                {
                    reference.shape = shape;
                }
            }
        }
        self.tp_shard = Some((rank, parts));
        Ok(())
    }
}

/// What [`Qwen4Weights::apply_trunk_overlay`] did.
#[derive(Debug)]
pub enum TrunkOverlayOutcome {
    Applied(TrunkOverlayApplied),
    /// Nothing was replaced; the reason is one line.
    Refused(String),
}

/// Tensors replaced by a trunk overlay and their device bytes before/after.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrunkOverlayApplied {
    pub tensors: usize,
    pub before_bytes: usize,
    pub after_bytes: usize,
}

/// Byte size of an `m x k` QT44 matrix; `None` unless K is a multiple of 256.
fn qt44_matrix_bytes(m: usize, k: usize) -> Option<usize> {
    (k % 256 == 0).then(|| m * (k / 256) * rdna_compute::MQ4V2_GROUP_BYTES)
}

/// Release every finalized handle through `free`, one owner each. A true
/// `Borrowed` resident is a view into an arena root that the transaction's
/// store owns (and releases last, once): it owns no bytes, so it is skipped
/// and never reaches the owner free path. Owned handles free normally; the
/// first error is reported after every handle was attempted.
fn free_taken_handles(
    taken: Vec<TakenWeight>,
    mut free: impl FnMut(GpuTensor) -> hip_bridge::HipResult<()>,
) -> hip_bridge::HipResult<()> {
    let mut first = None;
    for taken in taken {
        if let WeightHandle::Resident(tensor) = taken.handle {
            if tensor.buf.is_borrowed() {
                continue;
            }
            if let Err(error) = free(tensor) {
                if first.is_none() {
                    first = Some(error);
                }
            }
        }
    }
    first.map_or(Ok(()), Err)
}

/// What replaces a resident weight in `HIPFIRE_QWEN4_REQUANT` / a trunk
/// overlay: a kernel-produced tensor, or pre-encoded QT44 bytes (dtype
/// `MQ4G256V2`, shape `[len]`).
enum Replacement<'a> {
    Tensor(GpuTensor),
    Qt44(Cow<'a, [u8]>),
}

/// The device/owner operations of a replacement. [`GpuReplace`] is the only
/// production implementation; CPU tests substitute a recording fake.
trait ReplaceOps {
    /// Prepare an arena-registered weight for an in-place rewrite of its span:
    /// the Gpu quiesces the device exactly once (even with no cached shadow, a
    /// direct reader may be in flight), drops the source's cached shadow view
    /// without freeing any root or binding, and clears value-dependent weight
    /// metadata, so a stable pointer carrying new bytes/dtype cannot stale-hit.
    /// An error means nothing may be written.
    fn invalidate_arena_source(&mut self, ptr: usize) -> Result<bool, String>;
    /// Upload host bytes to the start of `dst`.
    fn write_host(&mut self, dst: &hip_bridge::DeviceBuffer, bytes: &[u8]) -> Result<(), String>;
    /// Copy `bytes` device bytes from the start of `src` to the start of `dst`.
    fn copy_device(
        &mut self,
        dst: &hip_bridge::DeviceBuffer,
        src: &hip_bridge::DeviceBuffer,
        bytes: usize,
    ) -> Result<(), String>;
    /// Free an owned tensor.
    fn free(&mut self, tensor: GpuTensor) -> Result<(), String>;
    /// Upload pre-encoded QT44 bytes as an owned resident tensor.
    fn upload_qt44(&mut self, bytes: &[u8]) -> Result<GpuTensor, String>;
}

struct GpuReplace<'g>(&'g mut Gpu);

impl ReplaceOps for GpuReplace<'_> {
    fn invalidate_arena_source(&mut self, ptr: usize) -> Result<bool, String> {
        self.0
            .invalidate_arena_weight_source(ptr)
            .map_err(|error| error.to_string())
    }

    fn write_host(&mut self, dst: &hip_bridge::DeviceBuffer, bytes: &[u8]) -> Result<(), String> {
        self.0
            .hip
            .memcpy_htod_offset(dst, 0, bytes)
            .map_err(|error| error.to_string())
    }

    fn copy_device(
        &mut self,
        dst: &hip_bridge::DeviceBuffer,
        src: &hip_bridge::DeviceBuffer,
        bytes: usize,
    ) -> Result<(), String> {
        self.0
            .hip
            .memcpy_dtod_at(dst, 0, src, 0, bytes)
            .map_err(|error| error.to_string())
    }

    fn free(&mut self, tensor: GpuTensor) -> Result<(), String> {
        self.0
            .free_tensor(tensor)
            .map_err(|error| error.to_string())
    }

    fn upload_qt44(&mut self, bytes: &[u8]) -> Result<GpuTensor, String> {
        let mut copy = self
            .0
            .upload_raw(bytes, &[bytes.len()])
            .map_err(|error| error.to_string())?;
        copy.dtype = DType::MQ4G256V2;
        Ok(copy)
    }
}

fn replace_resident(
    gpu: &mut Gpu,
    taken: &mut TakenWeight,
    replacement: Replacement<'_>,
) -> Result<(usize, usize), String> {
    replace_resident_with(&mut GpuReplace(gpu), taken, replacement)
}

/// Install `replacement` as `taken`'s resident weight and return the old and
/// new device bytes.
///
/// An owned resident is swapped for the replacement and the old tensor is
/// freed (the pre-arena behavior, unchanged). A true `Borrowed` resident is a
/// planned arena span: the replacement payload is written INTO that span (no
/// new permanent owner, the pointer the shadow arena bound stays valid), the
/// handle's dtype/shape are retagged, and any temporary owner is freed. A
/// replacement larger than the planned span is refused loudly.
fn replace_resident_with(
    ops: &mut dyn ReplaceOps,
    taken: &mut TakenWeight,
    replacement: Replacement<'_>,
) -> Result<(usize, usize), String> {
    let arena_backed = match &taken.handle {
        WeightHandle::Resident(old) => old.buf.is_borrowed(),
        WeightHandle::Alias(_) => {
            if let Replacement::Tensor(copy) = replacement {
                let _ = ops.free(copy);
            }
            return Err(format!("{}: weight is not resident", taken.key.name));
        }
    };
    if arena_backed {
        replace_in_arena_span(ops, taken, replacement)
    } else {
        replace_owned(ops, taken, replacement)
    }
}

fn replace_owned(
    ops: &mut dyn ReplaceOps,
    taken: &mut TakenWeight,
    replacement: Replacement<'_>,
) -> Result<(usize, usize), String> {
    let copy = match replacement {
        Replacement::Tensor(copy) => copy,
        Replacement::Qt44(bytes) => ops.upload_qt44(&bytes)?,
    };
    let WeightHandle::Resident(old) = &taken.handle else {
        let _ = ops.free(copy);
        return Err(format!("{}: weight is not resident", taken.key.name));
    };
    let sizes = (old.buf.size(), copy.buf.size());
    let WeightHandle::Resident(old) =
        std::mem::replace(&mut taken.handle, WeightHandle::Resident(copy))
    else {
        unreachable!()
    };
    ops.free(old)?;
    Ok(sizes)
}

/// Exact byte extent of an `[m, k]` replacement of `dtype`: the Q8_0 rows or
/// the `MQ{2..6}G256V2` groups (`8 + 32*bits` bytes per 256 values) the
/// requant kernels emit; `None` for any other dtype, a `k` that is not whole
/// groups, or overflow.
fn replacement_payload_bytes(dtype: DType, m: usize, k: usize) -> Option<usize> {
    let group_bytes = match dtype {
        DType::Q8_0 => return dtype.row_bytes(k).and_then(|row| row.checked_mul(m)),
        DType::MQ2G256V2 => rdna_compute::MQ2G256V2_GROUP_BYTES,
        DType::MQ3G256V2 => rdna_compute::MQ3G256V2_GROUP_BYTES,
        DType::MQ4G256V2 => rdna_compute::MQ4V2_GROUP_BYTES,
        DType::MQ5G256V2 => rdna_compute::MQ5G256V2_GROUP_BYTES,
        DType::MQ6G256V2 => rdna_compute::MQ6G256V2_GROUP_BYTES,
        _ => return None,
    };
    (k != 0 && k % 256 == 0)
        .then(|| m.checked_mul(k / 256)?.checked_mul(group_bytes))
        .flatten()
}

/// The retagged view of an arena span that now holds a replacement payload:
/// SAME pointer and span capacity as `old`, new dtype/shape. Refuses a payload
/// larger than the span.
fn arena_replacement_view(
    old: &GpuTensor,
    payload_bytes: usize,
    shape: &[usize],
    dtype: DType,
) -> Result<GpuTensor, String> {
    if !old.buf.is_borrowed() {
        return Err("arena replacement needs a borrowed arena view".to_string());
    }
    if payload_bytes > old.buf.size() {
        return Err(format!(
            "replacement payload of {payload_bytes} bytes exceeds the planned arena span of {} bytes; \
             an arena view cannot grow",
            old.buf.size()
        ));
    }
    Ok(GpuTensor {
        buf: old.buf.byte_view(0, old.buf.size()),
        shape: shape.to_vec(),
        dtype,
    })
}

/// Write a kernel-produced `copy` into the span `old` views.
fn write_tensor_into_span(
    ops: &mut dyn ReplaceOps,
    old: &GpuTensor,
    copy: &GpuTensor,
    m: usize,
    k: usize,
) -> Result<(GpuTensor, usize), String> {
    let payload = replacement_payload_bytes(copy.dtype, m, k)
        .filter(|&bytes| bytes <= copy.buf.size())
        .ok_or_else(|| {
            format!(
                "{:?} replacement of [{m}, {k}] has no exact payload extent within its {}-byte buffer",
                copy.dtype,
                copy.buf.size()
            )
        })?;
    let view = arena_replacement_view(old, payload, &copy.shape, copy.dtype)?;
    // One fence: the Gpu quiesces (even with no cached shadow), drops the
    // cached shadow view and value-dependent metadata, THEN we overwrite. A
    // failed fence writes nothing.
    ops.invalidate_arena_source(old.buf.as_ptr() as usize)?;
    ops.copy_device(&old.buf, &copy.buf, payload)?;
    Ok((view, payload))
}

fn replace_in_arena_span(
    ops: &mut dyn ReplaceOps,
    taken: &mut TakenWeight,
    replacement: Replacement<'_>,
) -> Result<(usize, usize), String> {
    let WeightHandle::Resident(old) = &taken.handle else {
        unreachable!("caller proved the handle is a borrowed resident")
    };
    let (view, before, after, temp) = match replacement {
        Replacement::Qt44(bytes) => {
            let view = arena_replacement_view(old, bytes.len(), &[bytes.len()], DType::MQ4G256V2)?;
            ops.invalidate_arena_source(old.buf.as_ptr() as usize)?;
            ops.write_host(&old.buf, &bytes)?;
            (view, old.buf.size(), bytes.len(), None)
        }
        Replacement::Tensor(copy) => {
            let [m, k] = taken.projection.logical_shape[..] else {
                let _ = ops.free(copy);
                return Err(format!(
                    "{}: replacement needs a rank-2 weight, got {:?}",
                    taken.key.name, taken.projection.logical_shape
                ));
            };
            match write_tensor_into_span(ops, old, &copy, m, k) {
                Ok((view, payload)) => (view, old.buf.size(), payload, Some(copy)),
                Err(error) => {
                    let _ = ops.free(copy);
                    return Err(error);
                }
            }
        }
    };
    taken.handle = WeightHandle::Resident(view);
    if let Some(copy) = temp {
        ops.free(copy)?;
    }
    Ok((before, after))
}

fn build_root_refs(config: &Qwen4Config) -> Result<Qwen4RootWeights, WeightError> {
    let hidden = config.hidden_size;
    let hc_wide = config.hc_count * hidden;
    let embedding = TensorRef::new(
        QWEN4_EMBED,
        TensorRole::TokenEmbedding,
        None,
        vec![config.vocab_size, hidden],
        DType::BF16,
    )?;
    let lm_head = TensorRef::new(
        "lm_head.weight",
        TensorRole::LanguageHead,
        None,
        vec![config.vocab_size, hidden],
        DType::BF16,
    )?;
    Ok(Qwen4RootWeights {
        embedding,
        lm_head,
        final_hyper: build_hyper_read_refs(
            "model.language_model.hyper_connection_mixer",
            None,
            hc_wide,
            config.hc_lowrank,
        )?,
    })
}

fn build_hyper_read_refs(
    prefix: &str,
    layer: Option<usize>,
    hc_wide: usize,
    hc_rank: usize,
) -> Result<HyperConnectionReadWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>| {
        TensorRef::new(
            format!("{prefix}.{suffix}"),
            role,
            layer,
            shape,
            DType::BF16,
        )
    };
    Ok(HyperConnectionReadWeights {
        hc_norm: tr(
            "hc_norm.weight",
            TensorRole::HyperConnectionNorm,
            vec![hc_wide],
        )?,
        input_mix_down: tr(
            "input_mix_weight_down.weight",
            TensorRole::HyperConnectionDown,
            vec![hc_rank, hc_wide],
        )?,
        input_mix_up: tr(
            "input_mix_weight_up.weight",
            TensorRole::HyperConnectionUp,
            vec![hc_wide, hc_rank],
        )?,
    })
}

fn build_hyper_refs(
    prefix: &str,
    layer: Option<usize>,
    hc_count: usize,
    hc_wide: usize,
    hc_rank: usize,
) -> Result<HyperConnectionWeights, WeightError> {
    let read = build_hyper_read_refs(prefix, layer, hc_wide, hc_rank)?;
    let block_inject = TensorRef::new(
        format!("{prefix}.block_inject_weight.weight"),
        TensorRole::HyperConnectionBlockInject,
        layer,
        vec![hc_count, hc_wide],
        DType::BF16,
    )?;
    Ok(HyperConnectionWeights {
        block_inject,
        hc_norm: read.hc_norm,
        input_mix_down: read.input_mix_down,
        input_mix_up: read.input_mix_up,
    })
}

fn build_layer_refs(config: &Qwen4Config) -> Result<Vec<Qwen4LayerWeights>, WeightError> {
    build_layer_refs_scoped(config, &Qwen4BandScope::single(config))
}

fn build_layer_refs_scoped(
    config: &Qwen4Config,
    scope: &Qwen4BandScope,
) -> Result<Vec<Qwen4LayerWeights>, WeightError> {
    scope.validate(config).map_err(WeightError::Config)?;
    let hidden = config.hidden_size;
    let hc_wide = config.hc_count * hidden;
    let hc_rank = config.hc_lowrank;
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let indexer_qk_dim =
        (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim;
    let gdn_qk = config.linear_num_key_heads * config.linear_key_head_dim;
    let gdn_v = config.linear_num_value_heads * config.linear_value_head_dim;
    let gdn_qkv = 2 * gdn_qk + gdn_v;
    let mut refs = Vec::with_capacity(scope.band.len());
    for layer in scope.band.clone() {
        let kind = config.layer_types[layer];
        let prefix = format!("model.language_model.layers.{layer}");
        let attn_hyper = build_hyper_refs(
            &format!("{prefix}.attn_hyper_connection"),
            Some(layer),
            config.hc_count,
            hc_wide,
            hc_rank,
        )?;
        let mlp_hyper = build_hyper_refs(
            &format!("{prefix}.mlp_hyper_connection"),
            Some(layer),
            config.hc_count,
            hc_wide,
            hc_rank,
        )?;
        let attention = if kind == LayerType::FullAttention {
            Some(build_qsa_refs(
                &format!("{prefix}.self_attn"),
                Some(layer),
                hidden,
                q_dim,
                kv_dim,
                indexer_qk_dim,
                config,
            )?)
        } else {
            None
        };
        let gdn = if kind == LayerType::LinearAttention {
            Some(build_gdn_refs(
                &format!("{prefix}.linear_attn"),
                Some(layer),
                hidden,
                gdn_qkv,
                gdn_v,
                config,
            )?)
        } else {
            None
        };
        let ple = match &config.ple {
            Some(ple_config) if ple_config.layer == layer => {
                let ple_channels = config
                    .hidden_size
                    .checked_mul(config.hc_count)
                    .ok_or_else(|| WeightError::ShapeOverflow("PLE channel width".to_string()))?;
                Some(build_ple_refs(
                    &format!("{prefix}.ple"),
                    Some(layer),
                    hidden,
                    ple_channels,
                    ple_config,
                )?)
            }
            _ => None,
        };
        let ffn = build_ffn_refs(&format!("{prefix}.mlp"), Some(layer), hidden, config)?;
        refs.push(Qwen4LayerWeights {
            layer,
            kind,
            attn_hyper,
            mlp_hyper,
            attention,
            gdn,
            ple,
            ffn,
        });
    }
    Ok(refs)
}

/// Bind the native MTP head; the caller binds it only when `config.mtp` is
/// declared.
fn build_mtp_refs(config: &Qwen4Config) -> Result<Qwen4MtpWeights, WeightError> {
    let hidden = config.hidden_size;
    let hc_wide = config.hc_count * hidden;
    let hc_rank = config.hc_lowrank;
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let indexer_qk_dim =
        (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim;
    Ok(Qwen4MtpWeights {
        fc_embedding: TensorRef::new(
            "mtp.fc_embedding.weight",
            TensorRole::MtpEmbeddingProjection,
            None,
            vec![hidden, hidden],
            DType::BF16,
        )?,
        fc_hidden: TensorRef::new(
            "mtp.fc_hidden.weight",
            TensorRole::MtpHiddenProjection,
            None,
            vec![hidden, hidden],
            DType::BF16,
        )?,
        pre_fc_norm_embedding: TensorRef::new(
            "mtp.pre_fc_norm_embedding.weight",
            TensorRole::MtpNorm,
            None,
            vec![hidden],
            DType::BF16,
        )?,
        pre_fc_norm_hidden: TensorRef::new(
            "mtp.pre_fc_norm_hidden.weight",
            TensorRole::MtpNorm,
            None,
            vec![hc_wide],
            DType::BF16,
        )?,
        attn_hyper: build_hyper_refs(
            "mtp.layers.0.attn_hyper_connection",
            None,
            config.hc_count,
            hc_wide,
            hc_rank,
        )?,
        mlp_hyper: build_hyper_refs(
            "mtp.layers.0.mlp_hyper_connection",
            None,
            config.hc_count,
            hc_wide,
            hc_rank,
        )?,
        attention: build_qsa_refs(
            "mtp.layers.0.self_attn",
            None,
            hidden,
            q_dim,
            kv_dim,
            indexer_qk_dim,
            config,
        )?,
        ffn: build_ffn_refs("mtp.layers.0.mlp", None, hidden, config)?,
        final_hyper: build_hyper_read_refs("mtp.hyper_connection_mixer", None, hc_wide, hc_rank)?,
    })
}

fn build_gdn_refs(
    prefix: &str,
    layer: Option<usize>,
    hidden: usize,
    qkv: usize,
    value: usize,
    config: &Qwen4Config,
) -> Result<GdnWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>| {
        TensorRef::new(
            format!("{prefix}.{suffix}"),
            role,
            layer,
            shape,
            DType::BF16,
        )
    };
    Ok(GdnWeights {
        a_log: tr(
            "A_log",
            TensorRole::GdnALog,
            vec![config.linear_num_value_heads],
        )?,
        dt_bias: tr(
            "dt_bias",
            TensorRole::GdnDtBias,
            vec![config.linear_num_value_heads],
        )?,
        in_proj_a: tr(
            "in_proj_a.weight",
            TensorRole::GdnA,
            vec![config.linear_num_value_heads, hidden],
        )?,
        in_proj_b: tr(
            "in_proj_b.weight",
            TensorRole::GdnB,
            vec![config.linear_num_value_heads, hidden],
        )?,
        qkv: tr("in_proj_qkv.weight", TensorRole::GdnQkv, vec![qkv, hidden])?,
        z: tr("in_proj_z.weight", TensorRole::GdnZ, vec![value, hidden])?,
        norm: tr(
            "norm.weight",
            TensorRole::GdnNorm,
            vec![config.linear_value_head_dim],
        )?,
        conv: tr(
            "conv1d.weight",
            TensorRole::GdnConv,
            vec![qkv, 1, config.linear_conv_kernel_dim],
        )?,
        output: tr(
            "out_proj.weight",
            TensorRole::GdnOutput,
            vec![hidden, value],
        )?,
    })
}

fn build_qsa_refs(
    prefix: &str,
    layer: Option<usize>,
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    indexer_qk_dim: usize,
    config: &Qwen4Config,
) -> Result<QsaWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>| {
        TensorRef::new(
            format!("{prefix}.{suffix}"),
            role,
            layer,
            shape,
            DType::BF16,
        )
    };
    Ok(QsaWeights {
        indexer_qk: tr(
            "indexer.index_qk_proj.weight",
            TensorRole::QsaIndexerQk,
            vec![indexer_qk_dim, hidden],
        )?,
        indexer_k_norm: tr(
            "indexer.k_layernorm.weight",
            TensorRole::QsaIndexerKNorm,
            vec![config.indexer_head_dim],
        )?,
        indexer_q_norm: tr(
            "indexer.q_layernorm.weight",
            TensorRole::QsaIndexerQNorm,
            vec![config.indexer_head_dim],
        )?,
        k_norm: tr("k_norm.weight", TensorRole::QsaKNorm, vec![config.head_dim])?,
        k: tr("k_proj.weight", TensorRole::QsaK, vec![kv_dim, hidden])?,
        output: tr("o_proj.weight", TensorRole::QsaOutput, vec![hidden, q_dim])?,
        q_norm: tr("q_norm.weight", TensorRole::QsaQNorm, vec![config.head_dim])?,
        q: tr("q_proj.weight", TensorRole::QsaQ, vec![2 * q_dim, hidden])?,
        v: tr("v_proj.weight", TensorRole::QsaV, vec![kv_dim, hidden])?,
    })
}

fn build_ple_refs(
    prefix: &str,
    layer: Option<usize>,
    hidden: usize,
    channels: usize,
    config: &Qwen4PleConfig,
) -> Result<PleWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>| {
        TensorRef::new(
            format!("{prefix}.{suffix}"),
            role,
            layer,
            shape,
            DType::BF16,
        )
    };
    Ok(PleWeights {
        conv: tr(
            "conv1d.weight",
            TensorRole::PleConv,
            vec![channels, 1, config.conv_kernel_size],
        )?,
        key: tr(
            "key_proj.weight",
            TensorRole::PleKey,
            vec![channels, config.embed_dim],
        )?,
        norm_conv: tr("norm_conv.weight", TensorRole::PleNorm, vec![channels])?,
        norm_key: tr("norm_key.weight", TensorRole::PleNorm, vec![channels])?,
        norm_query: tr("norm_query.weight", TensorRole::PleNorm, vec![channels])?,
        value: tr(
            "value_proj.weight",
            TensorRole::PleValue,
            vec![hidden, config.embed_dim],
        )?,
    })
}

/// Refuse, before any allocation or placement, every profile the typed
/// executor cannot run.  Header-only developer inventories are admitted by
/// `artifact` independently of this executor admission.
///
/// * Flash-Next: the pinned shared-expert MoE with PLE and the native MTP head.
/// * AR-only Flash-Next: the same, with no MTP head (the stock-FP8 recipe).
/// * Synthetic developer profiles: only under `cfg(test)` or the
///   `qwen4-dev-profiles` feature, and only a dense FFN (PLE and MTP optional
///   at this layer).  A synthetic MoE has no sealed route (see
///   [`admit_synthetic_profile`]).
pub(crate) fn admit_executable_profile(config: &Qwen4Config) -> Result<(), WeightError> {
    let unsupported = |reason: &str| WeightError::UnsupportedProfile {
        profile: config.profile.id().to_string(),
        reason: reason.to_string(),
    };
    match &config.profile {
        Qwen4Profile::FlashNext | Qwen4Profile::FlashNextArOnly => {
            if !matches!(
                &config.ffn,
                Qwen4FfnConfig::Moe {
                    shared_intermediate_size: Some(_),
                    ..
                }
            ) {
                return Err(unsupported(
                    "the Qwen4 executor binds only the shared-expert MoE FFN",
                ));
            }
            if config.ple.is_none() {
                return Err(unsupported("the Qwen4 executor requires PLE"));
            }
            match (config.profile.is_ar_only(), config.mtp.is_some()) {
                (false, false) => Err(unsupported(
                    "the Qwen4 executor requires the native MTP head",
                )),
                (true, true) => Err(unsupported(
                    "the AR-only Flash-Next profile carries no MTP head",
                )),
                _ => Ok(()),
            }
        }
        Qwen4Profile::Synthetic(_) => admit_synthetic_profile(config),
    }
}

/// The declaration [`Qwen4Weights::assemble`] publishes against: the profile
/// must be executable, and the manifest is the one `recipe` declares for it
/// (the admitted artifact's own recipe), never a rebuild from `config` alone.
fn assembly_manifest(
    config: &Qwen4Config,
    recipe: &Qwen4TierRecipe,
) -> Result<Qwen4Manifest, WeightError> {
    recipe.reject_expert_slab_dtypes()?;
    admit_executable_profile(config)?;
    Qwen4Manifest::build_with_recipe(config, recipe.clone())
}

/// Replace each trunk layer's two aggregate routed-expert entries with the one
/// `slab_arena` entry of its [`crate::expert_arena::ExpertArenaLayer`]: raw
/// bytes (`DType::Raw`, shape `[arena_bytes]`), replicated. The MTP head's
/// experts (`mtp.…`, not `layers.{L}.`) keep their aggregate entries.
fn apply_expert_arenas(
    weights: &mut Vec<WeightEntry>,
    arenas: &crate::expert_arena::ExpertArenaPlan,
) -> Result<(), WeightError> {
    use crate::expert_arena::ARENA_SUFFIX;
    let mut seen = 0usize;
    let mut out = Vec::with_capacity(weights.len());
    for entry in weights.drain(..) {
        let trunk = entry
            .layer
            .filter(|_| entry.name.starts_with("model.language_model.layers."));
        let projection = if entry.name.ends_with(".mlp.experts.gate_up_proj") {
            Some(".mlp.experts.gate_up_proj")
        } else if entry.name.ends_with(".mlp.experts.down_proj") {
            Some(".mlp.experts.down_proj")
        } else {
            None
        };
        let (Some(layer), Some(suffix)) = (trunk, projection) else {
            out.push(entry);
            continue;
        };
        if suffix == ".mlp.experts.down_proj" {
            continue;
        }
        let plan = arenas.layer(layer).ok_or_else(|| {
            WeightError::DescriptorMismatch(format!("expert arena plan has no layer {layer}"))
        })?;
        let prefix = entry
            .name
            .strip_suffix(suffix)
            .expect("suffix checked above");
        out.push(WeightEntry::layer_with_dtype_constraint(
            format!("{prefix}{ARENA_SUFFIX}"),
            layer,
            vec![plan.bytes],
            DType::Raw,
            DTypeConstraint::source_exact(DType::Raw),
            ShardPolicy::Replicate,
        ));
        seen += 1;
    }
    *weights = out;
    if seen != arenas.layers().len() {
        return Err(WeightError::DescriptorMismatch(format!(
            "expert arena plan has {} layers but the manifest declares {seen} trunk MoE layers",
            arenas.layers().len()
        )));
    }
    Ok(())
}

/// Whether `name` is a routed expert stack (trunk `…mlp.experts.*` or the MTP
/// head's `mtp.layers.0.mlp.experts.*`): the only tensors expert parallelism
/// shards.
fn is_routed_expert_tensor(name: &str) -> bool {
    name.ends_with(".mlp.experts.gate_up_proj") || name.ends_with(".mlp.experts.down_proj")
}

/// What [`Qwen4TierRecipe::StockFp8`] declares for one tensor: qt=55
/// `FP8E4M3B128` for the stacked routed experts, qt=56 `FP8E4M3PT` for the
/// PLE n-gram shards, source BF16 for everything else.
fn stock_fp8_dtype(name: &str, shape: &[usize]) -> DType {
    if shape.len() == 3 && is_routed_expert_tensor(name) {
        DType::FP8E4M3B128
    } else if is_ple_shard_name(name) {
        DType::FP8E4M3PT
    } else {
        DType::BF16
    }
}

/// The expert-parallel placement recorded in a transaction's projection
/// census (`lookup(name, layer)` is its device-0 projection): `Some` iff every
/// routed expert stack of `manifest` was fulfilled `ExpertCompact` for one
/// `(rank, ranks)`, `None` iff none was. A mixture, or disagreeing
/// placements, is a corrupt load and is refused before assembly takes handles.
///
/// A per-expert EP manifest ([`Qwen4Manifest::with_expert_units`]) replaces a
/// mixed layer's two aggregates by raw replicated arenas, which are not routed
/// expert tensors (no `gate_up_proj`/`down_proj` name) and carry no projection
/// tag: they are skipped, and the placement is recovered from the other trunk
/// layers and the MTP head (the caller falls back to the unit plan's own
/// placement when no tagged stack remains).
fn ep_from_projections<'a>(
    manifest: &Qwen4Manifest,
    lookup: impl Fn(&str, Option<usize>) -> Option<&'a hipfire_runtime::weight_store::WeightProjection>,
) -> Result<Option<Qwen4EpPlacement>, WeightError> {
    use hipfire_runtime::weight_store::WeightProjectionKind;
    let mut placement: Option<Qwen4EpPlacement> = None;
    let mut plain = 0usize;
    let mut compact = 0usize;
    for entry in manifest
        .weights
        .iter()
        .filter(|entry| is_routed_expert_tensor(&entry.name))
    {
        let Some(projection) = lookup(&entry.name, entry.layer) else {
            continue;
        };
        if projection.kind == WeightProjectionKind::ExpertCompact {
            compact += 1;
            let this = Qwen4EpPlacement {
                rank: projection.rank,
                ranks: projection.world_size,
            };
            if placement.is_some_and(|seen| seen != this) {
                return Err(WeightError::EpPlacement(format!(
                    "{}: placement {this:?} disagrees with {placement:?}",
                    entry.name
                )));
            }
            placement = Some(this);
        } else {
            plain += 1;
        }
    }
    if compact > 0 && plain > 0 {
        return Err(WeightError::EpPlacement(format!(
            "{compact} routed expert tensors were fulfilled compactly and {plain} in full"
        )));
    }
    Ok(placement)
}

/// Production builds never execute a developer profile.
#[cfg(not(any(test, feature = "qwen4-dev-profiles")))]
fn admit_synthetic_profile(config: &Qwen4Config) -> Result<(), WeightError> {
    Err(WeightError::UnsupportedProfile {
        profile: config.profile.id().to_string(),
        reason: "developer profiles execute only in builds with the qwen4-dev-profiles feature"
            .to_string(),
    })
}

#[cfg(any(test, feature = "qwen4-dev-profiles"))]
fn admit_synthetic_profile(config: &Qwen4Config) -> Result<(), WeightError> {
    match &config.ffn {
        Qwen4FfnConfig::Dense { .. } => Ok(()),
        Qwen4FfnConfig::Moe {
            num_experts,
            top_k,
            shared_intermediate_size: None,
            ..
        } => Err(WeightError::UnsupportedProfile {
            profile: config.profile.id().to_string(),
            reason: format!(
                "MoE without a shared expert has no executor (experts={num_experts}, top_k={top_k})"
            ),
        }),
        Qwen4FfnConfig::Moe {
            num_experts, top_k, ..
        } => Err(WeightError::UnsupportedProfile {
            profile: config.profile.id().to_string(),
            reason: format!(
                "no sealed MoE route exists for this geometry (experts={num_experts}, top_k={top_k}): \
                 the canonical qt44/qt53 route pins 512 experts, top-10, hidden 2560 and intermediate 640, \
                 and there is no BF16 routed-expert route or top-k other than 8/10"
            ),
        }),
    }
}

/// Bind one FFN block's typed references: dense gate/up/down, or the pinned
/// shared-expert MoE.
fn build_ffn_refs(
    prefix: &str,
    layer: Option<usize>,
    hidden: usize,
    config: &Qwen4Config,
) -> Result<Qwen4FfnWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>, dtype: DType| {
        TensorRef::new(format!("{prefix}.{suffix}"), role, layer, shape, dtype)
    };
    match &config.ffn {
        Qwen4FfnConfig::Dense { intermediate_size } => {
            let intermediate = *intermediate_size;
            Ok(Qwen4FfnWeights::Dense(DenseFfnWeights {
                gate: tr(
                    "gate_proj.weight",
                    TensorRole::DenseFfnGate,
                    vec![intermediate, hidden],
                    DType::BF16,
                )?,
                up: tr(
                    "up_proj.weight",
                    TensorRole::DenseFfnUp,
                    vec![intermediate, hidden],
                    DType::BF16,
                )?,
                down: tr(
                    "down_proj.weight",
                    TensorRole::DenseFfnDown,
                    vec![hidden, intermediate],
                    DType::BF16,
                )?,
            }))
        }
        Qwen4FfnConfig::Moe {
            num_experts,
            intermediate_size,
            shared_intermediate_size: Some(shared),
            ..
        } => {
            let (experts, intermediate, shared) = (*num_experts, *intermediate_size, *shared);
            let payload_experts = config
                .kept_expert_ids(layer.unwrap_or(0), prefix.starts_with("mtp."))
                .map_or(experts, <[u16]>::len);
            Ok(Qwen4FfnWeights::Moe(MoeWeights {
                expert_arena: None,
                unit_arenas: None,
                experts_down: tr(
                    "experts.down_proj",
                    TensorRole::RoutedDown,
                    vec![payload_experts, hidden, intermediate],
                    ROUTED_DOWN_DTYPE,
                )?,
                experts_gate_up: tr(
                    "experts.gate_up_proj",
                    TensorRole::RoutedGateUp,
                    vec![payload_experts, 2 * intermediate, hidden],
                    ROUTED_GATE_UP_DTYPE,
                )?,
                expert_source_layout: ExpertSourceLayout::PackedFused {
                    gate_up: format!("{prefix}.experts.gate_up_proj"),
                    down: format!("{prefix}.experts.down_proj"),
                    sidecars: Vec::new(),
                },
                gate: tr(
                    "gate.weight",
                    TensorRole::Router,
                    vec![experts, hidden],
                    DType::BF16,
                )?,
                shared_down: tr(
                    "shared_expert.down_proj.weight",
                    TensorRole::SharedExpertDown,
                    vec![hidden, shared],
                    DType::BF16,
                )?,
                shared_gate: tr(
                    "shared_expert.gate_proj.weight",
                    TensorRole::SharedExpertGate,
                    vec![shared, hidden],
                    DType::BF16,
                )?,
                shared_up: tr(
                    "shared_expert.up_proj.weight",
                    TensorRole::SharedExpertUp,
                    vec![shared, hidden],
                    DType::BF16,
                )?,
                shared_gate_scalar: tr(
                    "shared_expert_gate.weight",
                    TensorRole::SharedExpertGateScalar,
                    vec![1, hidden],
                    DType::BF16,
                )?,
            }))
        }
        Qwen4FfnConfig::Moe {
            shared_intermediate_size: None,
            ..
        } => Err(WeightError::UnsupportedProfile {
            profile: config.profile.id().to_string(),
            reason: "the Qwen4 executor binds only the shared-expert MoE FFN".to_string(),
        }),
    }
}

/// Whether `name` is one of the [`PLE_SHARD_COUNT`] canonical Flash-Next
/// n-gram PLE shard tensor names (`..ngram_embedding.shard_{0..127}.weight`,
/// no leading zeros).
pub fn is_ple_shard_name(name: &str) -> bool {
    is_ple_shard_name_within(name, PLE_SHARD_COUNT)
}

/// Whether `name` is a canonical PLE shard tensor name whose index is below
/// `shard_count` (the profile's declared [`Qwen4PleConfig::shard_count`]).
pub fn is_ple_shard_name_within(name: &str, shard_count: usize) -> bool {
    name.strip_prefix(hipfire_runtime::ple_sidecar::PLE_SHARD_PREFIX)
        .and_then(|name| name.strip_suffix(hipfire_runtime::ple_sidecar::PLE_SHARD_SUFFIX))
        .and_then(|digits| {
            let index = digits.parse::<usize>().ok()?;
            (index < shard_count && index.to_string() == digits).then_some(index)
        })
        .is_some()
}

/// Valid rows of Flash-Next PLE shard `shard`.  Profile-generic callers use
/// [`Qwen4PleConfig::shard_valid_rows`].
pub fn ple_valid_rows_for_shard(shard: usize) -> usize {
    crate::ple::PLE_VALID_ROWS
        .saturating_sub((shard as u64).saturating_mul(PLE_SHARD_ROWS as u64))
        .min(PLE_SHARD_ROWS as u64) as usize
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightError {
    Config(String),
    InvalidShape {
        name: String,
        shape: Vec<usize>,
    },
    ShapeOverflow(String),
    DescriptorMismatch(String),
    PlacementSetMismatch {
        expected: usize,
        actual: usize,
    },
    MissingPlacement {
        name: String,
        layer: Option<usize>,
        device: usize,
    },
    MissingExternalDescriptor {
        name: String,
        layer: Option<usize>,
        device: usize,
    },
    MissingResident {
        name: String,
        layer: Option<usize>,
        device: usize,
    },
    PleShardCount {
        expected: usize,
        actual: usize,
    },
    /// The profile is valid developer inventory (or a different identity) but
    /// the typed executor has no binder for it.
    UnsupportedProfile {
        profile: String,
        reason: String,
    },
    /// An expert-parallel placement that is malformed, or a transaction whose
    /// routed expert tensors do not agree on one.
    EpPlacement(String),
    /// A planar expert-slab dtype (`MQ3I4G128R256`/`R128`,
    /// `MQ2I4G128R256`/`R128`, `E8PI4G128R256`/`R128`) reached a declaration
    /// boundary.  The format is recognised but no generic qwen4 GPU kernel route
    /// exists for it.
    ExpertSlabDtypeKernelsUnavailable {
        dtype: DType,
    },
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(f, "Qwen4 manifest config: {message}"),
            Self::InvalidShape { name, shape } => write!(f, "{name}: invalid shape {shape:?}"),
            Self::ShapeOverflow(name) => write!(f, "{name}: shape overflow"),
            Self::DescriptorMismatch(name) => {
                write!(f, "{name}: external descriptor mismatch")
            }
            Self::PlacementSetMismatch { expected, actual } => {
                write!(f, "Qwen4 placement set expected {expected}, got {actual}")
            }
            Self::MissingPlacement {
                name,
                layer,
                device,
            } => write!(
                f,
                "missing resident placement {name}[layer {layer:?}] on device {device}"
            ),
            Self::MissingExternalDescriptor {
                name,
                layer,
                device,
            } => write!(
                f,
                "missing external descriptor {name}[layer {layer:?}] on device {device}"
            ),
            Self::MissingResident {
                name,
                layer,
                device,
            } => write!(
                f,
                "missing resident tensor {name}[layer {layer:?}] on device {device}"
            ),
            Self::PleShardCount { expected, actual } => {
                write!(f, "Qwen4 PLE shard count expected {expected}, got {actual}")
            }
            Self::UnsupportedProfile { profile, reason } => {
                write!(f, "Qwen4 profile {profile:?} is not executable: {reason}")
            }
            Self::ExpertSlabDtypeKernelsUnavailable { dtype } => write!(
                f,
                "Qwen4 expert-slab dtype {dtype:?} is unsupported: no GPU kernels exist for the \
                 MQ3I4G128R256/MQ3I4G128R128/MQ2I4G128R256/MQ2I4G128R128/E8PI4G128R256/E8PI4G128R128 \
                 planar formats at this declaration boundary"
            ),
            Self::EpPlacement(reason) => write!(f, "Qwen4 expert-parallel placement: {reason}"),
        }
    }
}

impl std::error::Error for WeightError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stock-FP8 declaration: qt=55 routed experts, qt=56 external PLE
    /// rows (160-byte rows), source BF16 for everything else, no MTP tensor;
    /// declared for the AR-only profile only, and the AR-only profile builds
    /// it by default.
    #[test]
    fn stock_fp8_manifest_declares_fp8_experts_ple_and_bf16_trunk() {
        let ar_only = pinned_config().into_ar_only().unwrap();
        let manifest = Qwen4Manifest::build_with_recipe(&ar_only, Qwen4TierRecipe::StockFp8)
            .expect("stock-FP8 manifest");
        assert_eq!(Qwen4Manifest::build(&ar_only).unwrap(), manifest);
        assert_eq!(manifest.identity.recipe, Qwen4TierRecipe::StockFp8);
        assert!(manifest.identity.recipe.trunk_raw_bf16());
        assert!(manifest.identity.recipe.fp8_experts());
        let (mut experts, mut shards) = (0, 0);
        for entry in &manifest.weights {
            assert!(!entry.name.starts_with("mtp."), "{}", entry.name);
            let want = if is_routed_expert_tensor(&entry.name) {
                experts += 1;
                DType::FP8E4M3B128
            } else if is_ple_shard_name(&entry.name) {
                shards += 1;
                assert_eq!(
                    entry.residency,
                    WeightResidency::external_rows(
                        160,
                        ar_only.ple.as_ref().unwrap().shard_valid_rows(
                            entry
                                .name
                                .rsplit("shard_")
                                .next()
                                .and_then(|tail| tail.strip_suffix(".weight"))
                                .and_then(|index| index.parse().ok())
                                .unwrap()
                        )
                    )
                );
                DType::FP8E4M3PT
            } else {
                DType::BF16
            };
            assert_eq!(entry.dtype, want, "{}", entry.name);
            assert_eq!(
                entry.dtype_constraint,
                DTypeConstraint::source_exact(want),
                "{}",
                entry.name
            );
        }
        assert_eq!(experts, 2 * 48);
        assert_eq!(shards, PLE_SHARD_COUNT);
        let gate_up = manifest
            .entry(
                "model.language_model.layers.0.mlp.experts.gate_up_proj",
                Some(0),
            )
            .unwrap();
        assert_eq!(gate_up.logical_shape, vec![512, 1280, 2560]);
        let down = manifest
            .entry(
                "model.language_model.layers.0.mlp.experts.down_proj",
                Some(0),
            )
            .unwrap();
        assert_eq!(down.logical_shape, vec![512, 2560, 640]);

        // Recipe and profile must agree both ways.
        let flash = pinned_config();
        assert!(matches!(
            Qwen4Manifest::build_with_recipe(&flash, Qwen4TierRecipe::StockFp8),
            Err(WeightError::UnsupportedProfile { .. })
        ));
        assert!(matches!(
            Qwen4Manifest::build_with_recipe(
                &ar_only,
                Qwen4TierRecipe::FlashNextPacked {
                    trunk_tier: DType::Q8_0,
                    mtp_source: false
                }
            ),
            Err(WeightError::UnsupportedProfile { .. })
        ));
        // P1: no expert parallelism for the stock-FP8 recipe.
        let error = manifest
            .with_expert_parallel(Qwen4EpPlacement { rank: 0, ranks: 2 })
            .unwrap_err();
        assert!(error.to_string().contains("stock-FP8"), "{error}");
        assert_eq!(
            Qwen4TierRecipe::from_flash_next_id(STOCK_FP8_RECIPE_ID),
            Some(Qwen4TierRecipe::StockFp8)
        );
        assert_eq!(Qwen4TierRecipe::StockFp8.synthetic_id(), None);
        assert_eq!(
            Qwen4TierRecipe::from_synthetic_id(STOCK_FP8_RECIPE_ID),
            None
        );
    }

    /// The AR-only profile is head-free everywhere an MTP entry point looks:
    /// the executor admits it without a head and refuses one bolted on.
    #[test]
    fn ar_only_profile_is_executable_without_mtp() {
        let ar_only = pinned_config().into_ar_only().unwrap();
        assert!(ar_only.mtp.is_none());
        admit_executable_profile(&ar_only).expect("AR-only Flash-Next is executable");
        let mut headed = ar_only.clone();
        headed.mtp = pinned_config().mtp;
        assert!(admit_executable_profile(&headed).is_err());
        assert!(admit_executable_profile(&pinned_config()).is_ok());
    }

    /// A fake device address (never dereferenced) for table-geometry tests.
    fn fake_tensor(base: usize, bytes: usize, shape: Vec<usize>, dtype: DType) -> GpuTensor {
        GpuTensor {
            // SAFETY: a borrowed (non-owning) wrapper the test never reads,
            // writes or frees; `from_tensor` only does address arithmetic.
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(base as *mut std::ffi::c_void, bytes)
            },
            shape,
            dtype,
        }
    }

    /// The qt=55 accessor: block/codes/scales addresses follow
    /// `base + e*S(R,K)` and `+R*K`; a wrong dtype/shape, a short or
    /// misaligned buffer is refused.
    #[test]
    fn fp8_expert_table_addresses_follow_the_contract_stride() {
        let base = 0x7f00_0000_0000usize;
        let bytes = 512 * 3_277_312;
        let gate_up = Fp8B128ExpertTable::from_tensor(
            "gate_up",
            &fake_tensor(base, bytes, vec![512, 1280, 2560], DType::FP8E4M3B128),
        )
        .unwrap();
        assert_eq!(
            gate_up,
            Fp8B128ExpertTable {
                base,
                experts: 512,
                rows: 1280,
                k: 2560,
                stride: 3_277_312
            }
        );
        assert_eq!(gate_up.block_ptr(0), Some(base));
        assert_eq!(gate_up.codes_ptr(7), Some(base + 7 * 3_277_312));
        assert_eq!(gate_up.scales_offset(), 1280 * 2560);
        assert_eq!(
            gate_up.scales_ptr(511),
            Some(base + 511 * 3_277_312 + 1280 * 2560)
        );
        assert_eq!(gate_up.block_ptr(512), None);
        assert_eq!(gate_up.scale_grid(), (10, 20));
        let entries = gate_up.block_entries();
        assert_eq!(entries.len(), 512);
        assert_eq!(entries[300], base + 300 * 3_277_312);
        let down = Fp8B128ExpertTable::from_tensor(
            "down",
            &fake_tensor(
                base,
                512 * 1_638_656,
                vec![512, 2560, 640],
                DType::FP8E4M3B128,
            ),
        )
        .unwrap();
        assert_eq!((down.stride, down.scale_grid()), (1_638_656, (20, 5)));

        for (tensor, needle) in [
            (
                fake_tensor(base, bytes, vec![512, 1280, 2560], DType::MQ4G256V2),
                "dtype",
            ),
            (
                fake_tensor(base, bytes, vec![655_360, 2560], DType::FP8E4M3B128),
                "shape",
            ),
            (
                fake_tensor(base, bytes - 1, vec![512, 1280, 2560], DType::FP8E4M3B128),
                "buffer",
            ),
            (
                fake_tensor(base + 16, bytes, vec![512, 1280, 2560], DType::FP8E4M3B128),
                "aligned",
            ),
        ] {
            let error = Fp8B128ExpertTable::from_tensor("t", &tensor).unwrap_err();
            assert!(error.to_string().contains(needle), "{needle}: {error}");
        }
    }

    const EXPERT_SLAB_DTYPES: [DType; 6] = [
        DType::MQ3I4G128R256,
        DType::MQ3I4G128R128,
        DType::MQ2I4G128R256,
        DType::MQ2I4G128R128,
        DType::E8PI4G128R256,
        DType::E8PI4G128R128,
    ];

    #[test]
    fn expert_slab_dtypes_are_rejected_by_tensor_ref() {
        for dtype in EXPERT_SLAB_DTYPES {
            for (name, role, shape) in [
                (
                    "model.language_model.layers.0.mlp.experts.gate_up_proj",
                    TensorRole::RoutedGateUp,
                    vec![512, 1280, 2560],
                ),
                (
                    "model.language_model.layers.0.mlp.experts.down_proj",
                    TensorRole::RoutedDown,
                    vec![512, 2560, 640],
                ),
                (
                    "model.language_model.layers.0.self_attn.q_proj.weight",
                    TensorRole::QsaQ,
                    vec![4096, 2560],
                ),
            ] {
                assert_eq!(
                    TensorRef::new(name, role, Some(0), shape, dtype),
                    Err(WeightError::ExpertSlabDtypeKernelsUnavailable { dtype }),
                    "{dtype:?} {name}"
                );
            }
        }
    }

    #[test]
    fn expert_slab_dtypes_are_rejected_by_recipe_guard() {
        let config = pinned_config();
        for dtype in EXPERT_SLAB_DTYPES {
            let recipe = Qwen4TierRecipe::FlashNextPacked {
                trunk_tier: dtype,
                mtp_source: false,
            };
            let expected = WeightError::ExpertSlabDtypeKernelsUnavailable { dtype };
            assert_eq!(recipe.reject_expert_slab_dtypes(), Err(expected.clone()));
            assert_eq!(
                Qwen4Manifest::build_with_recipe(&config, recipe.clone()).err(),
                Some(expected.clone()),
                "{dtype:?}"
            );
            assert_eq!(
                assembly_manifest(&config, &recipe).err(),
                Some(expected),
                "{dtype:?}"
            );
        }
        // The legacy tiers still build.
        for tier in [DType::Q8_0, DType::MQ6G256V2] {
            assert!(Qwen4Manifest::build_with_trunk_tier(&config, tier).is_ok());
        }
    }

    #[test]
    fn expert_slab_dtypes_are_rejected_in_raw_manifests() {
        use hipfire_runtime::weight_manifest::SourceDType;
        let base = Qwen4Manifest::build(&pinned_config()).unwrap();
        assert_eq!(base.reject_expert_slab_dtypes(), Ok(()));
        let placement = Qwen4EpPlacement { rank: 0, ranks: 2 };
        for dtype in EXPERT_SLAB_DTYPES {
            let expected = WeightError::ExpertSlabDtypeKernelsUnavailable { dtype };
            let mut target = base.clone();
            target.weights[0].dtype = dtype;
            assert_eq!(target.reject_expert_slab_dtypes(), Err(expected.clone()));
            assert_eq!(
                target.with_expert_parallel(placement).err(),
                Some(expected.clone())
            );
            for source in [
                SourceDType::Exact(dtype),
                SourceDType::OneOf(vec![DType::BF16, dtype]),
            ] {
                let mut constraint = base.clone();
                constraint.weights[0].dtype_constraint.source = source;
                assert_eq!(
                    constraint.reject_expert_slab_dtypes(),
                    Err(expected.clone())
                );
                assert_eq!(
                    constraint.with_expert_parallel(placement).err(),
                    Some(expected.clone())
                );
            }
            let mut recipe = base.clone();
            recipe.identity.recipe = Qwen4TierRecipe::FlashNextPacked {
                trunk_tier: dtype,
                mtp_source: false,
            };
            assert_eq!(recipe.reject_expert_slab_dtypes(), Err(expected));
        }
    }

    #[test]
    fn expert_slab_dtype_error_names_the_dtype() {
        for dtype in EXPERT_SLAB_DTYPES {
            let message = WeightError::ExpertSlabDtypeKernelsUnavailable { dtype }.to_string();
            assert!(message.contains(&format!("{dtype:?}")), "{message}");
            assert!(message.contains("unsupported"), "{message}");
        }
    }

    #[test]
    fn ple_shards_exclude_padding() {
        let total: usize = (0..PLE_SHARD_COUNT).map(ple_valid_rows_for_shard).sum();
        assert_eq!(total, 320_001_446);
        assert_eq!(ple_valid_rows_for_shard(127), 2_499_922);
    }

    #[test]
    fn sidecar_manifest_drops_exactly_the_ple_shards() {
        let full = Qwen4Manifest::build(&pinned_config()).unwrap();
        assert_eq!(full.external_entries().count(), PLE_SHARD_COUNT);
        let trimmed = full.clone().without_ple_shards();
        assert_eq!(full.weights.len() - trimmed.weights.len(), PLE_SHARD_COUNT);
        assert_eq!(trimmed.external_entries().count(), 0);
        assert_eq!(trimmed.metadata, full.metadata);
        assert_eq!(trimmed.state, full.state);
        assert_eq!(
            full.source_tensor_count() - trimmed.source_tensor_count(),
            PLE_SHARD_COUNT
        );
        // Every kept entry is untouched and none is a shard.
        assert!(trimmed
            .weights
            .iter()
            .all(|entry| !is_ple_shard_name(&entry.name)));
        assert!(trimmed
            .weights
            .iter()
            .all(|entry| full.weights.contains(entry)));
    }

    #[test]
    fn ple_shard_name_predicate_is_canonical() {
        let name = |index: &str| {
            format!(
                "{}{index}{}",
                hipfire_runtime::ple_sidecar::PLE_SHARD_PREFIX,
                hipfire_runtime::ple_sidecar::PLE_SHARD_SUFFIX
            )
        };
        assert!(is_ple_shard_name(&name("0")));
        assert!(is_ple_shard_name(&name("127")));
        assert!(!is_ple_shard_name(&name("128")));
        assert!(!is_ple_shard_name(&name("007")));
        assert!(!is_ple_shard_name(&name("")));
        assert!(!is_ple_shard_name("model.embed_tokens.weight"));
    }

    #[test]
    fn expert_parallel_manifest_shards_only_routed_experts() {
        use hipfire_runtime::device_mesh::DeviceMesh;
        let base = Qwen4Manifest::build(&pinned_config()).unwrap();
        // ep = 1 never passes through the transform: every expert stack stays
        // `Replicate` (the byte-identical single-device declaration).
        assert!(base
            .weights
            .iter()
            .filter(|entry| is_routed_expert_tensor(&entry.name))
            .all(|entry| entry.policy == ShardPolicy::Replicate));
        for ranks in [2usize, 3, 4, 5, 6, 8] {
            let rank = ranks - 1;
            let ep = base
                .clone()
                .with_expert_parallel(Qwen4EpPlacement { rank, ranks })
                .unwrap();
            assert_eq!(ep.identity, base.identity);
            assert_eq!(ep.state, base.state);
            assert_eq!(ep.metadata, base.metadata);
            assert_eq!(ep.weights.len(), base.weights.len());
            let mut sharded = 0;
            let mut mtp_sharded = 0;
            for (before, after) in base.weights.iter().zip(&ep.weights) {
                if is_routed_expert_tensor(&before.name) {
                    assert_eq!(
                        after.policy,
                        ShardPolicy::ExpertSharded {
                            n_experts: 512,
                            assign: ExpertAssign::Stride
                        },
                        "{}",
                        before.name
                    );
                    assert_eq!(after.logical_shape, before.logical_shape);
                    assert_eq!(after.logical_shape[0], 512);
                    sharded += 1;
                    mtp_sharded += usize::from(before.name.starts_with("mtp.layers.0."));
                } else {
                    assert_eq!(before, after, "{} must be untouched", before.name);
                }
            }
            // 48 trunk layers + the MTP head, gate_up and down each.
            assert_eq!(sharded, 2 * 49);
            assert_eq!(mtp_sharded, 2);
            // The sharded declaration is a valid single-mesh manifest.
            hipfire_runtime::weight_manifest::validate_manifest(
                &ep.weights,
                &DeviceMesh::single().unwrap(),
            )
            .unwrap();
        }
    }

    #[test]
    fn ep_placement_splits_experts_unevenly_and_densely() {
        for experts in [512usize, 8] {
            for ranks in [2usize, 3, 4, 5, 6, 8] {
                let mut total = 0;
                let mut owner_of = vec![usize::MAX; experts];
                for rank in 0..ranks {
                    let placement = Qwen4EpPlacement { rank, ranks };
                    let count = placement.local_count(experts);
                    assert!(count <= placement.max_local_count(experts));
                    assert_eq!(count, (0..experts).filter(|&e| placement.owns(e)).count());
                    // Owned experts fill slots 0..count densely, ascending.
                    let slots: Vec<usize> = (0..experts)
                        .filter(|&e| placement.owns(e))
                        .map(|e| placement.local_index(e))
                        .collect();
                    assert_eq!(slots, (0..count).collect::<Vec<_>>());
                    for e in (0..experts).filter(|&e| placement.owns(e)) {
                        assert_eq!(owner_of[e], usize::MAX, "expert {e} owned twice");
                        owner_of[e] = rank;
                    }
                    total += count;
                }
                assert_eq!(total, experts, "experts={experts} ranks={ranks}");
                assert!(owner_of.iter().all(|&o| o != usize::MAX));
            }
        }
        let six = |rank| Qwen4EpPlacement { rank, ranks: 6 };
        let counts: Vec<usize> = (0..6).map(|r| six(r).local_count(512)).collect();
        assert_eq!(counts, [86, 86, 85, 85, 85, 85]);
        assert_eq!(six(3).max_local_count(512), 86);
    }

    #[test]
    fn expert_parallel_manifest_refuses_bad_placements() {
        let base = Qwen4Manifest::build(&pinned_config()).unwrap();
        for (rank, ranks) in [(0, 1), (0, 0), (2, 2), (4, 4), (0, 513), (1, 600)] {
            assert!(
                matches!(
                    base.clone()
                        .with_expert_parallel(Qwen4EpPlacement { rank, ranks }),
                    Err(WeightError::EpPlacement(_))
                ),
                "rank {rank} of {ranks}"
            );
        }
    }

    #[test]
    fn ep_placement_is_recovered_from_the_projection_census() {
        use hipfire_runtime::weight_store::{WeightProjection, WeightProjectionKind};
        let manifest = Qwen4Manifest::build(&pinned_config()).unwrap();
        let projection = |kind, rank, world_size| WeightProjection {
            kind,
            axis: None,
            rank,
            world_size,
            logical_shape: vec![512, 1, 1],
            dtype: DType::MQ4G256V2,
        };
        let compact = projection(WeightProjectionKind::ExpertCompact, 1, 4);
        let plain = projection(WeightProjectionKind::Static, 0, 1);
        let other = projection(WeightProjectionKind::ExpertCompact, 2, 4);
        assert_eq!(
            ep_from_projections(&manifest, |_, _| Some(&compact)).unwrap(),
            Some(Qwen4EpPlacement { rank: 1, ranks: 4 })
        );
        assert_eq!(
            ep_from_projections(&manifest, |_, _| Some(&plain)).unwrap(),
            None
        );
        assert_eq!(ep_from_projections(&manifest, |_, _| None).unwrap(), None);
        // A mixture of compact and full expert stacks is refused.
        assert!(ep_from_projections(&manifest, |name, _| {
            if name.ends_with("down_proj") {
                Some(&plain)
            } else {
                Some(&compact)
            }
        })
        .is_err());
        // Two different placements are refused.
        assert!(ep_from_projections(&manifest, |_, layer| {
            if layer == Some(3) {
                Some(&other)
            } else {
                Some(&compact)
            }
        })
        .is_err());
    }

    /// A two-rank unit plan over the real Flash-Next geometry: trunk layers
    /// 0..3 fully resident, layer 3 half resident (mixed), layers 4.. host.
    fn ep_units_test_plan() -> crate::expert_ep_units::EpUnitPlan {
        use crate::expert_ep_units::{
            place_routed_experts_units, EpRanking, EpUnitGeometry, EpUnitSource,
            VRAM_ARENA_TAIL_PAD_BYTES,
        };
        let geometry = EpUnitGeometry {
            layers: 48,
            experts_per_layer: 512,
            ranks: 2,
            gate_up_bytes: 1280 * 10 * 136,
            down_bytes: 2560 * 5 * 68,
        };
        let ids: Vec<u32> = (0..48 * 512).collect();
        let budget = (3 * 256 + 128) * geometry.unit_bytes() + 4 * 2 * VRAM_ARENA_TAIL_PAD_BYTES;
        place_routed_experts_units(
            &geometry,
            &EpUnitSource::Ranking {
                ranking: EpRanking {
                    rank: &ids,
                    counts: None,
                    sha256: None,
                },
                rank_budget_bytes: vec![budget; 2],
            },
        )
        .unwrap()
    }

    /// `with_expert_parallel` + the whole-layer placement of `vram_layers`:
    /// what the loader's rank manifest is before the unit overlay.
    fn ep_placed_manifest(ep: Qwen4EpPlacement, vram_layers: usize) -> Qwen4Manifest {
        let mut manifest = Qwen4Manifest::build(&pinned_config())
            .unwrap()
            .with_expert_parallel(ep)
            .unwrap();
        crate::expert_residency::place_routed_experts(&mut manifest.weights, vram_layers);
        manifest
    }

    /// Flag off the loader never calls `with_expert_units`, so the manifest is
    /// the placed one. The overlay of a whole-layers plan (the same split) is
    /// that manifest byte for byte: nothing but arenas of MIXED layers ever
    /// changes a declaration.
    #[test]
    fn ep_units_flag_off_manifest_is_identical() {
        use crate::expert_ep_units::{place_routed_experts_units, EpUnitGeometry, EpUnitSource};
        for (rank, ranks, vram_layers) in [(0usize, 2usize, 0usize), (1, 2, 20), (2, 3, 48)] {
            let ep = Qwen4EpPlacement { rank, ranks };
            let today = ep_placed_manifest(ep, vram_layers);
            let plan = place_routed_experts_units(
                &EpUnitGeometry {
                    layers: 48,
                    experts_per_layer: 512,
                    ranks,
                    gate_up_bytes: 1280 * 10 * 136,
                    down_bytes: 2560 * 5 * 68,
                },
                &EpUnitSource::WholeLayers { vram_layers },
            )
            .unwrap();
            assert!(plan.mixed_layers(rank).is_empty());
            let overlaid = today.clone().with_expert_units(ep, &plan).unwrap();
            assert_eq!(
                overlaid, today,
                "rank {rank}/{ranks} vram_layers {vram_layers}"
            );
        }
    }

    #[test]
    fn ep_units_manifest_replaces_mixed_layers_with_arenas() {
        use crate::expert_ep_units::{UnitTier, UNIT_HOST_ARENA_SUFFIX, UNIT_VRAM_ARENA_SUFFIX};
        let plan = ep_units_test_plan();
        for rank in 0..2 {
            let ep = Qwen4EpPlacement { rank, ranks: 2 };
            let today = ep_placed_manifest(ep, 3);
            let units = today.clone().with_expert_units(ep, &plan).unwrap();
            assert_eq!(plan.mixed_layers(rank), vec![3]);
            let prefix = |layer: usize| format!("model.language_model.layers.{layer}");
            for layer in 0..48 {
                let gate_up = format!("{}.mlp.experts.gate_up_proj", prefix(layer));
                let down = format!("{}.mlp.experts.down_proj", prefix(layer));
                let vram = format!("{}{UNIT_VRAM_ARENA_SUFFIX}", prefix(layer));
                let host = format!("{}{UNIT_HOST_ARENA_SUFFIX}", prefix(layer));
                if layer == 3 {
                    assert!(units.entry(&gate_up, Some(layer)).is_none());
                    assert!(units.entry(&down, Some(layer)).is_none());
                    let map = plan.layer_map(rank, layer);
                    assert!(map.is_mixed());
                    for (name, tier, residency) in [
                        (&vram, UnitTier::Vram, WeightResidency::Resident),
                        (&host, UnitTier::Host, WeightResidency::HostMapped),
                    ] {
                        let entry = units.entry(name, Some(layer)).expect("arena entry");
                        assert_eq!(entry.dtype, DType::Raw);
                        assert_eq!(entry.logical_shape, vec![map.arena_entry_bytes(tier)]);
                        assert_eq!(entry.policy, ShardPolicy::Replicate);
                        assert_eq!(entry.residency, residency);
                    }
                } else {
                    assert!(units.entry(&vram, Some(layer)).is_none());
                    assert!(units.entry(&host, Some(layer)).is_none());
                    let want = if layer < 3 {
                        WeightResidency::Resident
                    } else {
                        WeightResidency::HostMapped
                    };
                    for name in [&gate_up, &down] {
                        let entry = units.entry(name, Some(layer)).expect("aggregate entry");
                        assert_eq!(entry.residency, want, "layer {layer}");
                        assert_eq!(Some(entry), today.entry(name, Some(layer)));
                    }
                }
            }
            // Everything that is not a trunk routed pair of layer 3 is today's,
            // the MTP head's host-mapped experts included.
            let mtp_routed = |entry: &&WeightEntry| {
                entry.name.starts_with("mtp.") && is_routed_expert_tensor(&entry.name)
            };
            assert!(today.weights.iter().filter(mtp_routed).count() > 0);
            for entry in today.weights.iter().filter(mtp_routed) {
                assert_eq!(entry.residency, WeightResidency::HostMapped);
                assert_eq!(units.entry(&entry.name, entry.layer), Some(entry));
            }
            assert_eq!(units.weights.len(), today.weights.len());
            for entry in &today.weights {
                if entry.layer == Some(3) && is_routed_expert_tensor(&entry.name) {
                    continue;
                }
                assert_eq!(units.entry(&entry.name, entry.layer), Some(entry));
            }
        }
    }

    #[test]
    fn ep_units_manifest_refuses_a_mismatched_rank_or_plan() {
        let plan = ep_units_test_plan();
        let ep = Qwen4EpPlacement { rank: 0, ranks: 2 };
        let placed = ep_placed_manifest(ep, 3);
        // The plan is a 2-rank plan.
        assert!(placed
            .clone()
            .with_expert_units(Qwen4EpPlacement { rank: 0, ranks: 3 }, &plan)
            .is_err());
        assert!(placed
            .clone()
            .with_expert_units(Qwen4EpPlacement { rank: 2, ranks: 2 }, &plan)
            .is_err());
        assert!(placed
            .clone()
            .with_expert_units(Qwen4EpPlacement { rank: 0, ranks: 1 }, &plan)
            .is_err());
        // A manifest without a trunk layer's routed pair is refused.
        let mut missing = placed.clone();
        missing
            .weights
            .retain(|entry| !(entry.layer == Some(7) && is_routed_expert_tensor(&entry.name)));
        assert!(missing.with_expert_units(ep, &plan).is_err());
        // The stock-FP8 recipe has no packed aggregate pair to overlay.
        let mut stock = placed.clone();
        stock.identity.recipe = Qwen4TierRecipe::StockFp8;
        assert!(stock.with_expert_units(ep, &plan).is_err());
    }

    /// Mixed layers declare no routed projection tag: the placement is
    /// recovered from the layers and the MTP head that still carry one, and a
    /// lookup is never made for an arena.
    #[test]
    fn ep_from_projections_tolerates_mixed_layer_arenas() {
        use hipfire_runtime::weight_store::{WeightProjection, WeightProjectionKind};
        let plan = ep_units_test_plan();
        let ep = Qwen4EpPlacement { rank: 1, ranks: 2 };
        let manifest = ep_placed_manifest(ep, 3)
            .with_expert_units(ep, &plan)
            .unwrap();
        assert!(manifest
            .weights
            .iter()
            .any(|entry| crate::expert_ep_units::is_unit_arena_name(&entry.name)));
        let compact = WeightProjection {
            kind: WeightProjectionKind::ExpertCompact,
            axis: None,
            rank: 1,
            world_size: 2,
            logical_shape: vec![512, 1, 1],
            dtype: DType::MQ4G256V2,
        };
        assert_eq!(
            ep_from_projections(&manifest, |name, _| {
                assert!(
                    !crate::expert_ep_units::is_unit_arena_name(name),
                    "{name} has no projection tag"
                );
                Some(&compact)
            })
            .unwrap(),
            Some(ep)
        );
        // Arena and MTP only: still recovered from the head.
        let mut arenas_and_mtp = manifest.clone();
        arenas_and_mtp
            .weights
            .retain(|entry| !is_trunk_routed_expert_entry(entry));
        assert_eq!(
            ep_from_projections(&arenas_and_mtp, |_, _| Some(&compact)).unwrap(),
            Some(ep)
        );
        // No tagged routed stack at all: no placement (the unit plan names it).
        arenas_and_mtp
            .weights
            .retain(|entry| !is_routed_expert_tensor(&entry.name));
        assert_eq!(
            ep_from_projections(&arenas_and_mtp, |_, _| Some(&compact)).unwrap(),
            None
        );
    }
    fn pinned_config() -> Qwen4Config {
        crate::config::compact_test_config()
    }

    fn put(
        out: &mut Vec<WeightEntry>,
        layer: Option<usize>,
        name: String,
        shape: &[usize],
        dtype: DType,
        constraint: DTypeConstraint,
        policy: ShardPolicy,
    ) {
        out.push(match layer {
            Some(layer) => WeightEntry::layer_with_dtype_constraint(
                &name,
                layer,
                shape.to_vec(),
                dtype,
                constraint,
                policy,
            ),
            None => WeightEntry::model_with_dtype_constraint(
                &name,
                shape.to_vec(),
                dtype,
                constraint,
                policy,
            ),
        });
    }

    fn oracle_bf16() -> DTypeConstraint {
        DTypeConstraint::source_exact(DType::BF16)
    }

    fn oracle_q8() -> DTypeConstraint {
        DTypeConstraint::source_from_sources(vec![DType::BF16, DType::Q8_0])
    }

    fn oracle_trunk() -> DTypeConstraint {
        DTypeConstraint::source_from_sources(vec![
            DType::BF16,
            DType::Q8_0,
            DType::MQ6G256V2,
            DType::MQ4G256V2,
        ])
    }

    fn oracle_hc(out: &mut Vec<WeightEntry>, layer: Option<usize>, prefix: &str) {
        for (suffix, shape, policy) in [
            (
                "block_inject_weight.weight",
                &[4usize, 10240][..],
                ShardPolicy::Replicate,
            ),
            ("hc_norm.weight", &[10240][..], ShardPolicy::Replicate),
            (
                "input_mix_weight_down.weight",
                &[320, 10240][..],
                ShardPolicy::ColumnShard { axis: 0 },
            ),
            (
                "input_mix_weight_up.weight",
                &[10240, 320][..],
                ShardPolicy::RowShard { axis: 1 },
            ),
        ] {
            put(
                out,
                layer,
                format!("{prefix}.{suffix}"),
                shape,
                DType::BF16,
                oracle_bf16(),
                policy,
            );
        }
    }

    fn oracle_qsa(out: &mut Vec<WeightEntry>, layer: Option<usize>, prefix: &str) {
        let col = ShardPolicy::ColumnShard { axis: 0 };
        let row = ShardPolicy::RowShard { axis: 1 };
        let rep = ShardPolicy::Replicate;
        for (suffix, shape, packed, policy) in [
            (
                "indexer.index_qk_proj.weight",
                &[640usize, 2560][..],
                true,
                col.clone(),
            ),
            ("indexer.k_layernorm.weight", &[128][..], false, rep.clone()),
            ("indexer.q_layernorm.weight", &[128][..], false, rep.clone()),
            ("k_norm.weight", &[256][..], false, rep.clone()),
            ("k_proj.weight", &[512, 2560][..], true, col.clone()),
            ("o_proj.weight", &[2560, 6144][..], true, row),
            ("q_norm.weight", &[256][..], false, rep),
            ("q_proj.weight", &[12288, 2560][..], true, col.clone()),
            ("v_proj.weight", &[512, 2560][..], true, col),
        ] {
            let (dtype, constraint) = if packed {
                (DType::Q8_0, oracle_trunk())
            } else {
                (DType::BF16, oracle_bf16())
            };
            put(
                out,
                layer,
                format!("{prefix}.self_attn.{suffix}"),
                shape,
                dtype,
                constraint,
                policy,
            );
        }
    }

    fn oracle_moe(out: &mut Vec<WeightEntry>, layer: Option<usize>, prefix: &str) {
        let experts = DTypeConstraint::source_from_sources(vec![
            DType::BF16,
            DType::MQ4G256V2,
            DType::MQ4G128V2,
        ]);
        let col = ShardPolicy::ColumnShard { axis: 0 };
        let rep = ShardPolicy::Replicate;
        let name = |suffix: &str| format!("{prefix}.mlp.{suffix}");
        put(
            out,
            layer,
            name("experts.down_proj"),
            &[512, 2560, 640],
            DType::MQ4G128V2,
            experts.clone(),
            rep.clone(),
        );
        put(
            out,
            layer,
            name("experts.gate_up_proj"),
            &[512, 1280, 2560],
            DType::MQ4G256V2,
            experts,
            rep.clone(),
        );
        put(
            out,
            layer,
            name("gate.weight"),
            &[512, 2560],
            DType::BF16,
            oracle_bf16(),
            rep.clone(),
        );
        put(
            out,
            layer,
            name("shared_expert.down_proj.weight"),
            &[2560, 640],
            DType::BF16,
            oracle_bf16(),
            ShardPolicy::RowShard { axis: 1 },
        );
        put(
            out,
            layer,
            name("shared_expert.gate_proj.weight"),
            &[640, 2560],
            DType::BF16,
            oracle_bf16(),
            col.clone(),
        );
        put(
            out,
            layer,
            name("shared_expert.up_proj.weight"),
            &[640, 2560],
            DType::BF16,
            oracle_bf16(),
            col,
        );
        put(
            out,
            layer,
            name("shared_expert_gate.weight"),
            &[1, 2560],
            DType::BF16,
            oracle_bf16(),
            rep,
        );
    }

    /// The complete pinned Flash-Next inventory, enumerated from literal names,
    /// shapes, dtypes, source constraints, shard policies and placements.  No
    /// tier/recipe helper or config field of this module derives any of it, so
    /// it is an independent oracle for the profile-owned generator.
    fn flash_next_oracle() -> Vec<WeightEntry> {
        let mut out = Vec::new();
        let head =
            DTypeConstraint::source_from_sources(vec![DType::BF16, DType::Q8_0, DType::MQ6G256V2]);
        out.push(
            WeightEntry::model_with_dtype_constraint(
                "model.language_model.embed_tokens.weight",
                vec![248320, 2560],
                DType::Q8_0,
                oracle_q8(),
                ShardPolicy::Pin(PinTarget::Embed),
            )
            .with_placement(PlacementHint::Pin(PinTarget::Embed)),
        );
        out.push(
            WeightEntry::model_with_dtype_constraint(
                "lm_head.weight",
                vec![248320, 2560],
                DType::MQ6G256V2,
                head,
                ShardPolicy::Pin(PinTarget::Output),
            )
            .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );
        out.push(
            WeightEntry::model_with_dtype_constraint(
                "mtp.embed_tokens.weight",
                vec![248320, 2560],
                DType::Q8_0,
                oracle_q8(),
                ShardPolicy::Tied {
                    source: "model.language_model.embed_tokens.weight".to_string(),
                },
            )
            .with_placement(PlacementHint::Pin(PinTarget::Embed)),
        );
        for (suffix, shape, policy) in [
            (
                "hc_norm.weight",
                &[10240usize][..],
                ShardPolicy::Pin(PinTarget::Output),
            ),
            (
                "input_mix_weight_down.weight",
                &[320, 10240][..],
                ShardPolicy::ColumnShard { axis: 0 },
            ),
            (
                "input_mix_weight_up.weight",
                &[10240, 320][..],
                ShardPolicy::RowShard { axis: 1 },
            ),
        ] {
            out.push(
                WeightEntry::model_with_dtype_constraint(
                    &format!("model.language_model.hyper_connection_mixer.{suffix}"),
                    shape.to_vec(),
                    DType::BF16,
                    oracle_bf16(),
                    policy,
                )
                .with_placement(PlacementHint::Pin(PinTarget::Output)),
            );
        }
        for layer in 0..48usize {
            let prefix = format!("model.language_model.layers.{layer}");
            oracle_hc(
                &mut out,
                Some(layer),
                &format!("{prefix}.attn_hyper_connection"),
            );
            oracle_hc(
                &mut out,
                Some(layer),
                &format!("{prefix}.mlp_hyper_connection"),
            );
            if layer % 4 == 3 {
                oracle_qsa(&mut out, Some(layer), &prefix);
            } else {
                let col = ShardPolicy::ColumnShard { axis: 0 };
                for (suffix, shape, packed, policy) in [
                    ("A_log", &[48usize][..], false, ShardPolicy::Replicate),
                    (
                        "conv1d.weight",
                        &[10240, 1, 4][..],
                        false,
                        ShardPolicy::Replicate,
                    ),
                    ("dt_bias", &[48][..], false, ShardPolicy::Replicate),
                    ("in_proj_a.weight", &[48, 2560][..], true, col.clone()),
                    ("in_proj_b.weight", &[48, 2560][..], true, col.clone()),
                    ("in_proj_qkv.weight", &[10240, 2560][..], true, col.clone()),
                    ("in_proj_z.weight", &[6144, 2560][..], true, col),
                    ("norm.weight", &[128][..], false, ShardPolicy::Replicate),
                    (
                        "out_proj.weight",
                        &[2560, 6144][..],
                        true,
                        ShardPolicy::RowShard { axis: 1 },
                    ),
                ] {
                    let (dtype, constraint) = if packed {
                        (DType::Q8_0, oracle_trunk())
                    } else {
                        (DType::BF16, oracle_bf16())
                    };
                    put(
                        &mut out,
                        Some(layer),
                        format!("{prefix}.linear_attn.{suffix}"),
                        shape,
                        dtype,
                        constraint,
                        policy,
                    );
                }
            }
            oracle_moe(&mut out, Some(layer), &prefix);
            if layer == 1 {
                let ple = format!("{prefix}.ple");
                let col = ShardPolicy::ColumnShard { axis: 0 };
                for (suffix, shape, policy) in [
                    (
                        "conv1d.weight",
                        &[10240usize, 1, 4][..],
                        ShardPolicy::Replicate,
                    ),
                    ("key_proj.weight", &[10240, 2560][..], col),
                    ("norm_conv.weight", &[10240][..], ShardPolicy::Replicate),
                    ("norm_key.weight", &[10240][..], ShardPolicy::Replicate),
                    ("norm_query.weight", &[10240][..], ShardPolicy::Replicate),
                    (
                        "value_proj.weight",
                        &[2560, 2560][..],
                        ShardPolicy::RowShard { axis: 1 },
                    ),
                ] {
                    put(
                        &mut out,
                        Some(layer),
                        format!("{ple}.{suffix}"),
                        shape,
                        DType::BF16,
                        oracle_bf16(),
                        policy,
                    );
                }
                for shard in 0..128usize {
                    let valid = 320_001_446usize
                        .saturating_sub(shard * 2_500_012)
                        .min(2_500_012);
                    out.push(
                        WeightEntry::layer_with_dtype_constraint(
                            &format!("{ple}.ple_embedding.ngram_embedding.shard_{shard}.weight"),
                            layer,
                            vec![2_500_012, 160],
                            DType::Q8_0,
                            oracle_q8(),
                            ShardPolicy::Replicate,
                        )
                        .external_rows(170, valid),
                    );
                }
            }
        }
        let mtp = |out: &mut Vec<WeightEntry>, name: &str, shape: &[usize], policy: ShardPolicy| {
            put(
                out,
                None,
                name.to_string(),
                shape,
                DType::BF16,
                oracle_bf16(),
                policy,
            );
        };
        mtp(
            &mut out,
            "mtp.fc_embedding.weight",
            &[2560, 2560],
            ShardPolicy::ColumnShard { axis: 0 },
        );
        mtp(
            &mut out,
            "mtp.fc_hidden.weight",
            &[2560, 2560],
            ShardPolicy::ColumnShard { axis: 0 },
        );
        mtp(
            &mut out,
            "mtp.pre_fc_norm_embedding.weight",
            &[2560],
            ShardPolicy::Replicate,
        );
        mtp(
            &mut out,
            "mtp.pre_fc_norm_hidden.weight",
            &[10240],
            ShardPolicy::Replicate,
        );
        mtp(
            &mut out,
            "mtp.hyper_connection_mixer.hc_norm.weight",
            &[10240],
            ShardPolicy::Replicate,
        );
        mtp(
            &mut out,
            "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
            &[320, 10240],
            ShardPolicy::ColumnShard { axis: 0 },
        );
        mtp(
            &mut out,
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
            &[10240, 320],
            ShardPolicy::RowShard { axis: 1 },
        );
        oracle_hc(&mut out, None, "mtp.layers.0.attn_hyper_connection");
        oracle_hc(&mut out, None, "mtp.layers.0.mlp_hyper_connection");
        oracle_qsa(&mut out, None, "mtp.layers.0");
        oracle_moe(&mut out, None, "mtp.layers.0");
        out
    }

    /// Flash-Next is an exact profile: the profile-owned generator reproduces
    /// the pinned inventory entry for entry, in order — names, layer keys,
    /// logical shapes, target dtypes, source constraints, shard policies,
    /// placements and external-row residencies — plus the exact state and I64
    /// metadata inventories and the manifest identity.
    #[test]
    fn flash_next_manifest_is_byte_identical_to_the_independent_oracle() {
        MTP_SOURCE_TIER_OVERRIDE.with(|value| value.set(Some(false)));
        let built = Qwen4Manifest::build_with_trunk_tier(&pinned_config(), DType::Q8_0);
        MTP_SOURCE_TIER_OVERRIDE.with(|value| value.set(None));
        let manifest = built.expect("pinned manifest");
        let oracle = flash_next_oracle();
        assert_eq!(
            manifest.weights.len(),
            1323,
            "weights (1325 source records)"
        );
        assert_eq!(oracle.len(), manifest.weights.len());
        for (index, (built, expected)) in manifest.weights.iter().zip(&oracle).enumerate() {
            assert_eq!(built, expected, "entry {index} {}", expected.name);
        }
        let mut state = Vec::new();
        for layer in 0..48usize {
            if layer % 4 == 3 {
                state.push(StateEntry::new(
                    StateKind::Kv {
                        quant: "qsa-full-raw-pooled".into(),
                    },
                    layer,
                ));
            } else {
                state.push(StateEntry::new(StateKind::Recurrent, layer));
                state.push(StateEntry::new(StateKind::Conv, layer));
            }
        }
        state.push(StateEntry::pinned(StateKind::Conv, PinTarget::Embed));
        assert_eq!(manifest.state, state);
        assert_eq!(manifest.state.len(), 85);
        let metadata_base = "model.language_model.layers.1.ple.ple_embedding";
        assert_eq!(
            manifest.metadata,
            vec![
                MetadataTensor::i64(format!("{metadata_base}.layer_multipliers"), vec![3]),
                MetadataTensor::i64(format!("{metadata_base}.ngram_heads_offsets"), vec![16]),
                MetadataTensor::i64(format!("{metadata_base}.ngram_heads_vocab_sizes"), vec![16]),
            ]
        );
        assert_eq!(
            manifest.identity,
            Qwen4ManifestIdentity {
                profile_id: "qwen3.8-flash-next".to_string(),
                developer_only: false,
                equation_source: "pinned-upstream",
                ffn_kind: "moe",
                recipe: Qwen4TierRecipe::FlashNextPacked {
                    trunk_tier: DType::Q8_0,
                    mtp_source: false,
                },
            }
        );
    }

    fn synthetic(
        ffn: Qwen4FfnConfig,
        hidden: usize,
        layers: usize,
        ple: bool,
        mtp: bool,
    ) -> Qwen4Config {
        crate::config::synthetic_test_config(ffn, hidden, layers, ple, mtp)
    }

    fn dense_ffn() -> Qwen4FfnConfig {
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

    /// Expected weight count of a synthetic inventory from closed-form per-role
    /// counts (HC 4+4, GDN 9 or QSA 9, FFN roles, MTP block, PLE block).
    fn expected_weight_count(config: &Qwen4Config) -> usize {
        let ffn = match &config.ffn {
            Qwen4FfnConfig::Dense { .. } => 3,
            Qwen4FfnConfig::Moe {
                shared_intermediate_size,
                ..
            } => {
                3 + if shared_intermediate_size.is_some() {
                    4
                } else {
                    0
                }
            }
        };
        // embedding, head, final HC mixer (3)
        let mut count = 5 + config.num_hidden_layers * (8 + 9 + ffn);
        if config.mtp.is_some() {
            // alias, fc x2, pre-fc norms x2, mixer x3, HC 8, QSA 9, FFN
            count += 1 + 4 + 3 + 8 + 9 + ffn;
        }
        if let Some(ple) = &config.ple {
            count += 6 + ple.shard_count;
        }
        count
    }

    fn assert_synthetic_bf16_inventory(config: &Qwen4Config, manifest: &Qwen4Manifest) {
        assert_eq!(manifest.weights.len(), expected_weight_count(config));
        assert_eq!(manifest.identity.profile_id, config.profile.id());
        assert!(manifest.identity.developer_only);
        assert_eq!(
            manifest.identity.equation_source,
            config.profile.equation_source().as_str()
        );
        assert_eq!(
            manifest.identity.recipe,
            Qwen4TierRecipe::SyntheticSourceBf16
        );
        for entry in &manifest.weights {
            assert_eq!(entry.dtype, DType::BF16, "{} must stay BF16", entry.name);
            assert!(
                entry.dtype_constraint.accepts(DType::BF16),
                "{}",
                entry.name
            );
            for packed in [
                DType::Q8_0,
                DType::MQ6G256V2,
                DType::MQ4G256V2,
                DType::MQ4G128V2,
            ] {
                assert!(
                    !entry.dtype_constraint.accepts(packed),
                    "{} must not admit {packed:?} before an executable role proof",
                    entry.name
                );
            }
        }
        assert_eq!(
            manifest.metadata.len(),
            if config.ple.is_some() { 3 } else { 0 }
        );
        assert_eq!(
            manifest.external_entries().count(),
            config.ple.as_ref().map_or(0, |ple| ple.shard_count)
        );
        let has = |needle: &str| manifest.weights.iter().any(|e| e.name.contains(needle));
        assert_eq!(has(".ple."), config.ple.is_some());
        assert_eq!(has("mtp."), config.mtp.is_some());
        let aliases = manifest
            .weights
            .iter()
            .filter(|entry| matches!(entry.policy, ShardPolicy::Tied { .. }))
            .count();
        assert_eq!(aliases, usize::from(config.mtp.is_some()));
        let recurrent_layers = config
            .layer_types
            .iter()
            .filter(|kind| **kind == LayerType::LinearAttention)
            .count();
        let full_layers = config.layer_types.len() - recurrent_layers;
        assert_eq!(
            manifest.state.len(),
            recurrent_layers * 2 + full_layers + usize::from(config.ple.is_some())
        );
    }

    #[test]
    fn dense_64_layer_hidden_5120_inventory_has_only_gate_up_down_roles() {
        for (ple, mtp) in [(false, false), (true, false), (false, true), (true, true)] {
            let config = synthetic(dense_ffn(), 5120, 64, ple, mtp);
            let manifest = Qwen4Manifest::build(&config).expect("dense manifest");
            assert_synthetic_bf16_inventory(&config, &manifest);
            assert_eq!(manifest.identity.ffn_kind, "dense");
            for layer in [0usize, 63] {
                let name =
                    |role: &str| format!("model.language_model.layers.{layer}.mlp.{role}.weight");
                for (role, shape) in [
                    ("gate_proj", [17_408usize, 5120]),
                    ("up_proj", [17_408, 5120]),
                    ("down_proj", [5120, 17_408]),
                ] {
                    let entry = manifest
                        .entry(&name(role), Some(layer))
                        .unwrap_or_else(|| panic!("missing dense {role} at layer {layer}"));
                    assert_eq!(entry.logical_shape, shape);
                    assert!(!entry.residency.is_external(), "dense roles are resident");
                }
            }
            for entry in &manifest.weights {
                for forbidden in [".experts.", "shared_expert", ".mlp.gate.weight"] {
                    assert!(
                        !entry.name.contains(forbidden),
                        "dense profile declared MoE tensor {}",
                        entry.name
                    );
                }
            }
            // The wide HC width at hidden 5120 is hc_count * hidden.
            let mixer = manifest
                .entry(
                    "model.language_model.hyper_connection_mixer.hc_norm.weight",
                    None,
                )
                .expect("final HC norm");
            assert_eq!(mixer.logical_shape, [config.hc_count * 5120]);
        }
    }

    #[test]
    fn absent_ple_and_mtp_leave_no_tensors_metadata_aliases_or_state() {
        let config = synthetic(dense_ffn(), 5120, 64, false, false);
        let manifest = Qwen4Manifest::build(&config).expect("manifest");
        assert!(manifest.metadata.is_empty());
        assert_eq!(manifest.external_entries().count(), 0);
        assert!(manifest
            .weights
            .iter()
            .all(|entry| !entry.name.starts_with("mtp.") && !entry.name.contains(".ple.")));
        assert!(manifest
            .weights
            .iter()
            .all(|entry| !matches!(entry.policy, ShardPolicy::Tied { .. })));
        assert_eq!(manifest.source_tensor_count(), manifest.weights.len());
        let full = Qwen4Manifest::build(&synthetic(dense_ffn(), 5120, 64, true, true))
            .expect("full manifest");
        assert_eq!(full.state.len(), manifest.state.len() + 1, "PLE conv state");
        assert!(full.source_tensor_count() > manifest.source_tensor_count());
    }

    #[test]
    fn small_moe_inventory_with_and_without_a_shared_expert() {
        for shared in [Some(64usize), None] {
            let config = synthetic(small_moe(shared), 256, 4, true, true);
            let manifest = Qwen4Manifest::build(&config).expect("small moe manifest");
            assert_synthetic_bf16_inventory(&config, &manifest);
            assert_eq!(manifest.identity.ffn_kind, "moe");
            let layer = |role: &str| format!("model.language_model.layers.0.mlp.{role}");
            let expert = manifest
                .entry(&layer("experts.gate_up_proj"), Some(0))
                .expect("routed gate_up");
            assert_eq!(expert.logical_shape, [8, 128, 256]);
            let down = manifest
                .entry(&layer("experts.down_proj"), Some(0))
                .expect("routed down");
            assert_eq!(down.logical_shape, [8, 256, 64]);
            assert_eq!(
                manifest
                    .entry(&layer("gate.weight"), Some(0))
                    .unwrap()
                    .logical_shape,
                [8, 256]
            );
            for role in [
                "shared_expert.gate_proj.weight",
                "shared_expert.up_proj.weight",
                "shared_expert.down_proj.weight",
                "shared_expert_gate.weight",
            ] {
                assert_eq!(
                    manifest.entry(&layer(role), Some(0)).is_some(),
                    shared.is_some(),
                    "{role}"
                );
            }
            assert!(manifest
                .entry(&layer("gate_proj.weight"), Some(0))
                .is_none());
        }
    }

    #[test]
    fn ple_shard_inventory_follows_the_typed_geometry() {
        let config = synthetic(small_moe(Some(64)), 256, 4, true, false);
        let ple = config.ple.clone().expect("synthetic PLE");
        let manifest = Qwen4Manifest::build(&config).expect("manifest");
        let shards = manifest.external_entries().collect::<Vec<_>>();
        assert_eq!(shards.len(), ple.shard_count);
        let mut total_valid = 0usize;
        for (index, entry) in shards.iter().enumerate() {
            assert_eq!(entry.layer, Some(ple.layer));
            assert_eq!(entry.logical_shape, [ple.shard_rows, ple.row_width]);
            let WeightResidency::ExternalRows {
                row_bytes,
                valid_rows,
            } = entry.residency
            else {
                panic!("PLE shard must be external rows");
            };
            assert_eq!(
                row_bytes,
                RowEncoding::Bf16.encoded_row_bytes(ple.row_width)
            );
            assert_eq!(valid_rows, ple.shard_valid_rows(index));
            total_valid += valid_rows;
        }
        assert_eq!(total_valid as u64, ple.valid_rows);
        let base = format!(
            "model.language_model.layers.{}.ple.ple_embedding",
            ple.layer
        );
        assert_eq!(
            manifest.metadata,
            vec![
                MetadataTensor::i64(
                    format!("{base}.layer_multipliers"),
                    vec![ple.multiplier_count]
                ),
                MetadataTensor::i64(format!("{base}.ngram_heads_offsets"), vec![ple.head_count]),
                MetadataTensor::i64(
                    format!("{base}.ngram_heads_vocab_sizes"),
                    vec![ple.head_count]
                ),
            ]
        );
        let trimmed = manifest.clone().without_ple_shards();
        assert_eq!(trimmed.external_entries().count(), 0);
        assert_eq!(
            manifest.weights.len() - trimmed.weights.len(),
            ple.shard_count
        );
        assert!(
            is_ple_shard_name_within(
                &shards[0]
                    .name
                    .replace(&format!("layers.{}.", ple.layer), "layers.1."),
                ple.shard_count
            ) || ple.layer != 1
        );
    }

    #[test]
    fn recipe_and_profile_must_agree_and_synthetic_ignores_tier_environment() {
        let synthetic_config = synthetic(small_moe(Some(64)), 256, 4, false, false);
        let error = Qwen4Manifest::build_with_recipe(
            &synthetic_config,
            Qwen4TierRecipe::FlashNextPacked {
                trunk_tier: DType::MQ6G256V2,
                mtp_source: false,
            },
        )
        .unwrap_err();
        assert!(
            matches!(error, WeightError::UnsupportedProfile { .. }),
            "{error}"
        );
        let error = Qwen4Manifest::build_with_recipe(
            &pinned_config(),
            Qwen4TierRecipe::SyntheticSourceBf16,
        )
        .unwrap_err();
        assert!(
            matches!(error, WeightError::UnsupportedProfile { .. }),
            "{error}"
        );
        // The MTP source-tier switch and the trunk tier do not move a
        // synthetic recipe.
        MTP_SOURCE_TIER_OVERRIDE.with(|value| value.set(Some(true)));
        let built = Qwen4Manifest::build(&synthetic(small_moe(Some(64)), 256, 4, false, true));
        MTP_SOURCE_TIER_OVERRIDE.with(|value| value.set(None));
        let manifest = built.expect("synthetic manifest");
        assert_eq!(
            manifest.identity.recipe,
            Qwen4TierRecipe::SyntheticSourceBf16
        );
        assert!(manifest
            .weights
            .iter()
            .all(|entry| entry.dtype == DType::BF16));
    }

    /// PLE channels are `hc_count * hidden` and the key/value projections read
    /// `embed_dim` columns: with hidden 5120 and a 16-wide embedding the conv
    /// and norms are 10240 wide, key is [10240, 16] and value is [5120, 16]
    /// (the S0 dense-5120 source shapes).  Flash-Next (embed == hidden) keeps
    /// [10240, 2560] / [2560, 2560], asserted by the FN oracle.
    #[test]
    fn ple_channels_follow_hidden_and_projections_follow_embed_dim() {
        let config = synthetic(dense_ffn(), 5120, 64, true, false);
        let ple = config.ple.clone().expect("PLE");
        assert_ne!(
            ple.embed_dim, config.hidden_size,
            "mixed hidden/embed fixture"
        );
        let channels = config.hc_count * config.hidden_size;
        let manifest = Qwen4Manifest::build(&config).expect("manifest");
        let shape = |suffix: &str| {
            manifest
                .entry(
                    &format!("model.language_model.layers.{}.ple.{suffix}", ple.layer),
                    Some(ple.layer),
                )
                .unwrap_or_else(|| panic!("missing {suffix}"))
                .logical_shape
                .clone()
        };
        assert_eq!(shape("conv1d.weight"), [channels, 1, ple.conv_kernel_size]);
        assert_eq!(shape("key_proj.weight"), [channels, ple.embed_dim]);
        assert_eq!(
            shape("value_proj.weight"),
            [config.hidden_size, ple.embed_dim]
        );
        for norm in ["norm_conv.weight", "norm_key.weight", "norm_query.weight"] {
            assert_eq!(shape(norm), [channels], "{norm}");
        }
    }

    /// The HC inject operand has `hc_count` rows (not a literal 4): a
    /// synthetic profile with `hc_count = 2` declares `[2, wide]` for the
    /// attention and FFN connections of every layer and of the MTP layer,
    /// and the final mixers (trunk and MTP) declare no inject operand at all.
    #[test]
    fn hyper_connection_inject_rows_follow_hc_count() {
        let config = synthetic(small_moe(Some(64)), 256, 4, false, true);
        assert_eq!(config.hc_count, 2, "fixture must exercise a non-FN count");
        let wide = config.hc_count * config.hidden_size;
        let manifest = Qwen4Manifest::build(&config).expect("manifest");
        let mut injects = 0usize;
        for entry in &manifest.weights {
            if entry.name.ends_with(".block_inject_weight.weight") {
                assert_eq!(
                    entry.logical_shape,
                    [config.hc_count, wide],
                    "{}",
                    entry.name
                );
                injects += 1;
            }
        }
        // attention + FFN per trunk layer, plus attention + FFN for MTP.
        assert_eq!(injects, 2 * config.num_hidden_layers + 2);
        for name in [
            "model.language_model.hyper_connection_mixer.block_inject_weight.weight",
            "mtp.hyper_connection_mixer.block_inject_weight.weight",
        ] {
            assert!(manifest.entry(name, None).is_none(), "{name}");
        }
        for (name, layer) in [
            (
                "model.language_model.layers.1.mlp_hyper_connection",
                Some(1),
            ),
            ("mtp.layers.0.attn_hyper_connection", None),
        ] {
            let shape = |suffix: &str| {
                manifest
                    .entry(&format!("{name}.{suffix}"), layer)
                    .unwrap_or_else(|| panic!("{name}.{suffix}"))
                    .logical_shape
                    .clone()
            };
            assert_eq!(shape("hc_norm.weight"), [wide]);
            assert_eq!(
                shape("input_mix_weight_down.weight"),
                [config.hc_lowrank, wide]
            );
            assert_eq!(
                shape("input_mix_weight_up.weight"),
                [wide, config.hc_lowrank]
            );
        }
    }

    #[test]
    fn ffn_shape_overflow_is_an_error_not_a_panic() {
        let layer = |name: &str,
                     layer_idx: usize,
                     shape: Vec<usize>,
                     dtype: DType,
                     policy: ShardPolicy,
                     source: &DTypeConstraint| {
            WeightEntry::layer_with_dtype_constraint(
                name,
                layer_idx,
                shape,
                dtype,
                source.clone(),
                policy,
            )
        };
        let bf16 = oracle_bf16();
        let quant = qwen4_quant_source();
        let mut weights = Vec::new();
        for ffn in [
            Qwen4FfnConfig::Dense {
                intermediate_size: usize::MAX,
            },
            Qwen4FfnConfig::Moe {
                num_experts: 8,
                top_k: 2,
                intermediate_size: usize::MAX / 2 + 1,
                shared_intermediate_size: None,
                norm_topk_prob: true,
            },
            Qwen4FfnConfig::Moe {
                num_experts: usize::MAX,
                top_k: 2,
                intermediate_size: 2,
                shared_intermediate_size: None,
                norm_topk_prob: true,
            },
        ] {
            let error = push_ffn_entries(&mut weights, &layer, "p", 0, 4096, &ffn, &bf16, &quant)
                .expect_err("overflowing FFN geometry must be refused");
            assert!(matches!(error, WeightError::ShapeOverflow(_)), "{error}");
        }
        assert!(weights.is_empty(), "nothing is declared past a refusal");
        let error = push_ffn_entries(
            &mut weights,
            &layer,
            "p",
            0,
            4096,
            &Qwen4FfnConfig::Dense {
                intermediate_size: 0,
            },
            &bf16,
            &quant,
        )
        .expect_err("a zero-width FFN is refused");
        assert!(matches!(error, WeightError::InvalidShape { .. }), "{error}");
    }

    #[test]
    fn executor_admits_flash_next_and_dense_and_refuses_synthetic_moe() {
        admit_executable_profile(&pinned_config()).expect("Flash-Next is executable");
        for config in [
            synthetic(dense_ffn(), 5120, 64, true, true),
            synthetic(dense_ffn(), 256, 4, false, false),
        ] {
            admit_executable_profile(&config).expect("dense developer profile is executable");
        }
        for shared in [Some(64), None] {
            let config = synthetic(small_moe(shared), 256, 4, true, true);
            let error = admit_executable_profile(&config).unwrap_err();
            let WeightError::UnsupportedProfile { profile, reason } = &error else {
                panic!("{error}");
            };
            assert_eq!(profile, config.profile.id());
            assert!(
                reason.contains("experts=") && reason.contains("top_k="),
                "{reason}"
            );
            if shared.is_some() {
                assert!(reason.contains("no sealed MoE route"), "{reason}");
            }
        }
    }

    #[test]
    fn assembly_declares_the_admitted_recipes_manifest() {
        // A packed synthetic artifact is declared packed, not as the
        // profile's default source-exact manifest `build` returns.
        let config = synthetic(dense_ffn(), 5120, 64, true, true);
        let default = Qwen4Manifest::build(&config).expect("default manifest");
        assert_eq!(
            default.identity.recipe,
            Qwen4TierRecipe::SyntheticSourceBf16
        );
        let packed = Qwen4Manifest::build_with_recipe(&config, Qwen4TierRecipe::SyntheticPacked)
            .expect("packed manifest");
        let assembled = assembly_manifest(&config, &packed.identity.recipe).expect("assembly");
        assert_eq!(assembled, packed);
        assert_eq!(assembled.identity.recipe, Qwen4TierRecipe::SyntheticPacked);
        assert_ne!(
            assembled.weights, default.weights,
            "the packed recipe declares packed formats"
        );
        // Same tensors, so the placement set is unchanged by the recipe.
        assert_eq!(
            assembled.resident_keys().collect::<Vec<_>>(),
            default.resident_keys().collect::<Vec<_>>()
        );
        // The source-exact recipe reproduces `build` exactly.
        let source = assembly_manifest(&config, &default.identity.recipe).expect("source assembly");
        assert_eq!(source, default);
        // A recipe the profile cannot be declared under is refused.
        let flash_next = Qwen4TierRecipe::FlashNextPacked {
            trunk_tier: DType::Q8_0,
            mtp_source: false,
        };
        let error = assembly_manifest(&config, &flash_next).unwrap_err();
        assert!(
            matches!(error, WeightError::UnsupportedProfile { .. }),
            "{error}"
        );
    }

    #[test]
    fn flash_next_assembly_manifest_is_exactly_the_default_build() {
        // The recipe admission recorded (carrying the trunk-tier and
        // MTP-source knobs) rebuilds the manifest `Qwen4Manifest::build`
        // declares, tier overrides included.
        let config = pinned_config();
        let built = Qwen4Manifest::build(&config).expect("pinned manifest");
        let assembled = assembly_manifest(&config, &built.identity.recipe).expect("assembly");
        assert_eq!(assembled, built);
        for tier in [DType::Q8_0, DType::MQ6G256V2] {
            let tiered = Qwen4Manifest::build_with_trunk_tier(&config, tier).expect("tiered");
            let assembled =
                assembly_manifest(&config, &tiered.identity.recipe).expect("tiered assembly");
            assert_eq!(assembled, tiered);
            assert_eq!(
                assembled.resident_keys().collect::<Vec<_>>(),
                built.resident_keys().collect::<Vec<_>>()
            );
        }
        // A synthetic recipe cannot declare Flash-Next.
        let error = assembly_manifest(&config, &Qwen4TierRecipe::SyntheticPacked).unwrap_err();
        assert!(
            matches!(error, WeightError::UnsupportedProfile { .. }),
            "{error}"
        );
    }

    #[test]
    fn expert_tier_manifest_removes_only_trunk_routed_entries() {
        let config = pinned_config();
        let recipe = Qwen4TierRecipe::FlashNextPacked {
            trunk_tier: DType::Q8_0,
            mtp_source: false,
        };
        let base = Qwen4Manifest::build_with_recipe(&config, recipe).expect("base manifest");
        let tiered = base.clone().with_expert_tiers().expect("tier manifest");
        assert_eq!(tiered.identity, base.identity);
        assert_eq!(tiered.weights.len() + 96, base.weights.len());
        assert_eq!(tiered.state, base.state);
        assert_eq!(tiered.metadata, base.metadata);
        let removed = |name: &str, layer: Option<usize>| {
            layer.is_some()
                && name.starts_with("model.language_model.layers.")
                && (name.ends_with(".mlp.experts.gate_up_proj")
                    || name.ends_with(".mlp.experts.down_proj"))
        };
        let kept = base
            .resident_keys()
            .filter(|(name, layer)| !removed(name, *layer))
            .collect::<Vec<_>>();
        assert_eq!(tiered.resident_keys().collect::<Vec<_>>(), kept);
        // The MTP head's aggregate experts stay declared.
        for name in [
            "mtp.layers.0.mlp.experts.gate_up_proj",
            "mtp.layers.0.mlp.experts.down_proj",
        ] {
            assert!(tiered.entry(name, None).is_some(), "{name}");
        }
        // A second application, expert parallelism and a stock/synthetic/dense
        // declaration are each refused under the HOTSET name.
        let refused = [
            tiered.clone().with_expert_tiers(),
            base.clone()
                .with_expert_parallel(Qwen4EpPlacement { rank: 0, ranks: 2 })
                .and_then(Qwen4Manifest::with_expert_tiers),
            Qwen4Manifest::build(&synthetic(dense_ffn(), 256, 4, true, true))
                .and_then(Qwen4Manifest::with_expert_tiers),
        ];
        for result in refused {
            let error = result.expect_err("tier manifest refusal");
            assert!(
                error.to_string().contains("HIPFIRE_QWEN4_EXPERT_HOTSET"),
                "{error}"
            );
        }
    }

    #[test]
    fn check_expert_tiers_accepts_base_and_refuses_without_mutation() {
        let config = pinned_config();
        let recipe = Qwen4TierRecipe::FlashNextPacked {
            trunk_tier: DType::Q8_0,
            mtp_source: false,
        };
        let base = Qwen4Manifest::build_with_recipe(&config, recipe).expect("base manifest");
        let base_keys = base.resident_keys().collect::<Vec<_>>();
        let base_len = base.weights.len();
        base.check_expert_tiers().expect("base manifest qualifies");
        assert_eq!(base.weights.len(), base_len);
        assert_eq!(base.resident_keys().collect::<Vec<_>>(), base_keys);

        let tiered = base.clone().with_expert_tiers().expect("tier manifest");
        let tiered_keys = tiered.resident_keys().collect::<Vec<_>>();
        let tiered_len = tiered.weights.len();
        let ep = base
            .clone()
            .with_expert_parallel(Qwen4EpPlacement { rank: 0, ranks: 2 })
            .expect("ep manifest");
        let ep_keys = ep.resident_keys().collect::<Vec<_>>();
        let ep_len = ep.weights.len();
        for (manifest, keys, len) in [(&tiered, &tiered_keys, tiered_len), (&ep, &ep_keys, ep_len)]
        {
            let error = manifest.check_expert_tiers().expect_err("refusal");
            assert!(
                error.to_string().contains("HIPFIRE_QWEN4_EXPERT_HOTSET"),
                "{error}"
            );
            assert_eq!(manifest.weights.len(), len);
            assert_eq!(&manifest.resident_keys().collect::<Vec<_>>(), keys);
        }
    }

    #[test]
    fn placement_set_accepts_missing_tier_entries_but_not_missing_nonexpert() {
        let config = pinned_config();
        let recipe = Qwen4TierRecipe::FlashNextPacked {
            trunk_tier: DType::Q8_0,
            mtp_source: false,
        };
        let base = Qwen4Manifest::build_with_recipe(&config, recipe).expect("base manifest");
        let tiered = base.clone().with_expert_tiers().expect("tier manifest");
        let placements = |manifest: &Qwen4Manifest| {
            manifest
                .resident_keys()
                .map(|(name, layer)| Qwen4Placement {
                    name: name.to_string(),
                    layer,
                    device: 0,
                })
                .collect::<Vec<_>>()
        };
        // The tier-owned entries are absent from the tier placement set.
        let tier_placements = placements(&tiered);
        check_placement_set(&tiered, &tier_placements).expect("tier set accepted");
        // A regular load still requires them.
        assert!(matches!(
            check_placement_set(&base, &tier_placements),
            Err(WeightError::PlacementSetMismatch { .. })
        ));
        // And a missing non-expert tensor is rejected under the tier manifest.
        let mut missing = tier_placements.clone();
        let at = missing
            .iter()
            .position(|placement| {
                !placement.name.contains(".mlp.experts.") && placement.layer.is_some()
            })
            .expect("a trunk non-expert placement");
        missing.remove(at);
        assert!(matches!(
            check_placement_set(&tiered, &missing),
            Err(WeightError::PlacementSetMismatch { .. })
        ));
        // A tier-owned tensor offered as a placement is not expected either.
        let mut extra = tier_placements;
        extra.push(Qwen4Placement {
            name: "model.language_model.layers.0.mlp.experts.gate_up_proj".to_string(),
            layer: Some(0),
            device: 0,
        });
        assert!(matches!(
            check_placement_set(&tiered, &extra),
            Err(WeightError::PlacementSetMismatch { .. })
        ));
    }

    #[test]
    fn assembly_refuses_a_synthetic_moe_with_the_executor_text() {
        let config = synthetic(small_moe(Some(64)), 256, 4, true, true);
        let error = assembly_manifest(&config, &Qwen4TierRecipe::SyntheticSourceBf16).unwrap_err();
        let WeightError::UnsupportedProfile { reason, .. } = &error else {
            panic!("{error}");
        };
        assert!(reason.contains("no sealed MoE route"), "{reason}");
    }

    #[test]
    fn dense_binders_follow_the_profile_geometry() {
        let config = synthetic(dense_ffn(), 256, 4, true, true);
        let Qwen4FfnConfig::Dense { intermediate_size } = config.ffn else {
            panic!("dense profile");
        };
        let manifest = Qwen4Manifest::build(&config).expect("dense manifest");
        let layers = build_layer_refs(&config).expect("dense layer refs");
        let mtp = build_mtp_refs(&config).expect("dense mtp refs");
        let check = |reference: &TensorRef| {
            let entry = manifest
                .entry(&reference.name, reference.layer)
                .unwrap_or_else(|| panic!("manifest lacks {}", reference.name));
            assert_eq!(entry.logical_shape, reference.shape, "{}", reference.name);
        };
        for layer in &layers {
            let dense = layer.ffn.dense().expect("dense FFN binding");
            assert!(layer.ffn.moe().is_none());
            assert_eq!(
                dense.gate.shape,
                vec![intermediate_size, config.hidden_size]
            );
            assert_eq!(
                dense.down.shape,
                vec![config.hidden_size, intermediate_size]
            );
            for reference in [&dense.gate, &dense.up, &dense.down] {
                check(reference);
            }
            check(&layer.mlp_hyper.block_inject);
            assert_eq!(
                layer.mlp_hyper.block_inject.shape,
                vec![config.hc_count, config.hc_count * config.hidden_size]
            );
        }
        let dense = mtp.ffn.dense().expect("dense MTP FFN binding");
        for reference in [&dense.gate, &dense.up, &dense.down] {
            check(reference);
        }
        check(&mtp.mlp_hyper.block_inject);
    }

    #[test]
    fn manifest_matches_pinned_text_inventory_and_rejects_legacy_names() {
        let manifest = Qwen4Manifest::build(&pinned_config()).expect("pinned config manifest");
        // The source has 1,658 records, of which 333 are vision records.  The
        // text manifest accounts for all 1,325 remaining records; its one MTP
        // embedding alias is excluded by source_tensor_count and its three
        // exact I64 PLE arrays are included through `metadata`.
        assert_eq!(manifest.source_tensor_count(), 1325);
        assert_eq!(manifest.source_tensor_count() + 333, 1658);
        let expect = |name: &str, layer: Option<usize>, shape: &[usize]| {
            let entry = manifest
                .entry(name, layer)
                .unwrap_or_else(|| panic!("missing exact inventory tensor {name} layer {layer:?}"));
            assert_eq!(entry.logical_shape, shape, "{name} shape");
        };
        expect(
            "model.language_model.layers.0.linear_attn.A_log",
            Some(0),
            &[48],
        );
        expect(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            Some(0),
            &[10240, 2560],
        );
        expect(
            "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight",
            Some(3),
            &[640, 2560],
        );
        expect(
            "model.language_model.layers.3.self_attn.q_proj.weight",
            Some(3),
            &[12288, 2560],
        );
        expect(
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight",
            Some(1),
            &[PLE_SHARD_ROWS, PLE_ROW_WIDTH],
        );
        expect("mtp.fc_hidden.weight", None, &[2560, 2560]);
        expect("mtp.layers.0.self_attn.q_proj.weight", None, &[12288, 2560]);
        expect(
            "model.language_model.hyper_connection_mixer.hc_norm.weight",
            None,
            &[10240],
        );
        assert!(manifest
            .entry("model.layers.0.linear_attn.in_proj_qkv.weight", Some(0))
            .is_none());
        assert!(manifest
            .entry(
                "model.language_model.layers.0.self_attn.qkv_proj.weight",
                Some(0)
            )
            .is_none());
        assert!(manifest
            .entry("model.language_model.layers.0.mlp.router.weight", Some(0))
            .is_none());
        assert!(manifest.metadata.iter().any(|tensor| {
            tensor.name == "model.language_model.layers.1.ple.ple_embedding.layer_multipliers"
                && tensor.source_dtype == "I64"
                && tensor.shape == [3]
        }));
        assert!(matches!(
            manifest
                .entry(
                    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight",
                    Some(1)
                )
                .expect("PLE shard"),
            WeightEntry {
                residency: WeightResidency::ExternalRows { .. },
                ..
            }
        ));
    }
    #[test]
    fn final_hyper_manifest_has_read_operands_only() {
        let config = pinned_config();
        let manifest = Qwen4Manifest::build(&config).expect("pinned config manifest");
        assert!(
            manifest
                .entry(
                    "model.language_model.hyper_connection_mixer.block_inject_weight.weight",
                    None,
                )
                .is_none(),
            "final HC must not invent a write operand"
        );
        let root = build_root_refs(&config).expect("root references");
        assert_eq!(
            root.final_hyper.hc_norm.name,
            "model.language_model.hyper_connection_mixer.hc_norm.weight"
        );
        let mtp = build_mtp_refs(&config).expect("MTP references");
        assert_eq!(
            mtp.final_hyper.input_mix_up.name,
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight"
        );
    }

    #[test]
    fn qwen4_moe_keeps_fused_source_replicated_for_single_execution() {
        let config = pinned_config();
        let manifest = Qwen4Manifest::build(&config).expect("pinned config manifest");
        for name in [
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            "model.language_model.layers.0.mlp.experts.down_proj",
        ] {
            let entry = manifest
                .entry(name, Some(0))
                .unwrap_or_else(|| panic!("missing Qwen4 expert source {name}"));
            assert_eq!(
                entry.policy,
                ShardPolicy::Replicate,
                "{name} must be replicated for the sealed Single route"
            );
        }

        let layers = build_layer_refs(&config).expect("Qwen4 layer references");
        let moe = layers[0].ffn.moe().expect("Flash-Next binds the MoE FFN");
        match &moe.expert_source_layout {
            ExpertSourceLayout::PackedFused {
                gate_up,
                down,
                sidecars,
            } => {
                assert_eq!(
                    gate_up,
                    "model.language_model.layers.0.mlp.experts.gate_up_proj"
                );
                assert_eq!(down, "model.language_model.layers.0.mlp.experts.down_proj");
                assert!(sidecars.is_empty());
            }
            other => panic!("pinned Qwen4 source inventory must stay PackedFused: {other:?}"),
        }
    }
    #[test]
    fn matrices_quantize_by_group_width_and_unrepresentable_k_stays_bf16() {
        let tensor = |name: &str, shape: &[usize]| {
            TensorRef::new(
                name,
                TensorRole::RoutedGateUp,
                Some(0),
                shape.to_vec(),
                DType::BF16,
            )
            .expect("tensor reference")
        };
        assert_eq!(
            tensor(
                "model.language_model.layers.0.mlp.experts.gate_up_proj",
                &[2, 4, 256],
            )
            .dtype,
            ROUTED_GATE_UP_DTYPE
        );
        assert_eq!(
            tensor(
                "model.language_model.layers.0.mlp.experts.down_proj",
                &[2, 4, 128],
            )
            .dtype,
            ROUTED_DOWN_DTYPE,
            "a K the aligned group cannot carry falls to the row-local one"
        );
        assert_eq!(
            tensor("mtp.layers.0.mlp.experts.gate_up_proj", &[2, 4, 256]).dtype,
            ROUTED_GATE_UP_DTYPE
        );
        assert_eq!(
            tensor("mtp.layers.0.mlp.experts.down_proj", &[2, 4, 128]).dtype,
            ROUTED_DOWN_DTYPE
        );

        // Rank-2 matrices the packed set admits all take the eight-bit class,
        // whatever their alignment above the 128-weight admission boundary:
        // the block format carries any K, so there is no fallback tier.
        for name in [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        ] {
            for k in [256usize, 384] {
                assert_eq!(
                    tensor(name, &[2, k]).dtype,
                    DType::Q8_0,
                    "rank-2 matrix {name} K={k} must ship eight-bit"
                );
            }
        }

        // The token embedding is eight-bit; the language head is six-bit.
        assert_eq!(
            tensor("model.language_model.embed_tokens.weight", &[2, 256]).dtype,
            DType::Q8_0
        );
        assert_eq!(tensor("lm_head.weight", &[2, 256]).dtype, DType::MQ6G256V2);

        // The thinnest-over-widest reduction keeps source bytes whatever its K:
        // the hyper-connection mixer is four rows over a
        // `hc_count * hidden` reduction.
        for name in [
            "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
            "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight",
            "model.language_model.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
        ] {
            assert_eq!(
                tensor(name, &[2, 256]).dtype,
                DType::BF16,
                "{name} keeps source bytes"
            );
        }

        // K = 320 is neither 256- nor 128-aligned (the hyper-connection up
        // projection), and K = 160 is the PLE row width; both stay source BF16.
        for (name, k) in [
            (
                "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
                320usize,
            ),
            (
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                160,
            ),
        ] {
            assert_eq!(
                tensor(name, &[2, k]).dtype,
                DType::BF16,
                "{name} K={k} must stay source-exact BF16"
            );
        }

        // The MTP connector matrices still have a source-BF16 owner.
        for name in ["mtp.fc_embedding.weight", "mtp.fc_hidden.weight"] {
            assert_eq!(
                tensor(name, &[2, 256]).dtype,
                DType::BF16,
                "{name} stays source-exact BF16 until its owner reads quantized weights"
            );
        }
    }

    /// The trunk recipe is a decision, not a default: the wide attention/GDN
    /// projections ship eight-bit because the paired candidate tiers are worse
    /// here — four bits on a dense projection (which writes straight into the
    /// residual stream) produced a model that opened a reasoning block it could
    /// not close, and BF16 is the unpacked path this packing exists to replace.
    /// Either fallback changes the dtype assertion or the count below.  The
    /// eight-bit tier is what an unset `HIPFIRE_QWEN4_TRUNK_TIER` declares; the
    /// six-bit rung is the selector's, and is asserted below.
    #[test]
    fn mtp_tier_knob_reaches_the_five_attention_matrices() {
        let config = pinned_config();
        // Measured on the published rung: these five ship qt=3 (Q8F16) while the
        // trunk carries the mq6 rung, and the knob must move exactly these.
        let mtp_attention = [
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
        ];
        let trunk_attention = "model.language_model.layers.2.linear_attn.in_proj_qkv.weight";
        MTP_SOURCE_TIER_OVERRIDE.with(|value| value.set(Some(false)));
        let shipped = Qwen4Manifest::build(&config).expect("default manifest");
        MTP_SOURCE_TIER_OVERRIDE.with(|value| value.set(Some(true)));
        let source = Qwen4Manifest::build(&config).expect("source-tier manifest");
        MTP_SOURCE_TIER_OVERRIDE.with(|value| value.set(None));
        for name in mtp_attention {
            let default_entry = shipped
                .entry(name, None)
                .expect("default MTP attention entry");
            let source_entry = source
                .entry(name, None)
                .expect("source-tier MTP attention entry");
            assert_ne!(default_entry.dtype, DType::BF16, "{name} ships quantized");
            assert_eq!(
                source_entry.dtype,
                DType::BF16,
                "{name} must follow HIPFIRE_QWEN4_MTP_TIER=source"
            );
        }
        // The rank-3 routed experts keep their packed tier: the expert writer
        // carries no BF16 arm, and the knob is scoped to rank-2 matrices.
        for name in [
            "mtp.layers.0.mlp.experts.gate_up_proj",
            "mtp.layers.0.mlp.experts.down_proj",
        ] {
            let default_entry = shipped.entry(name, None).expect("default expert entry");
            let source_entry = source.entry(name, None).expect("source-tier expert entry");
            assert_eq!(
                default_entry.dtype, source_entry.dtype,
                "{name} must not move with the matrix knob"
            );
        }
        let default_trunk = shipped
            .entry(trunk_attention, Some(2))
            .expect("trunk entry");
        let source_trunk = source.entry(trunk_attention, Some(2)).expect("trunk entry");
        assert_eq!(
            default_trunk.dtype, source_trunk.dtype,
            "trunk tier must not move"
        );
        let norm = source
            .entry("mtp.layers.0.self_attn.q_norm.weight", None)
            .expect("MTP norm entry");
        assert_eq!(
            norm.dtype,
            DType::BF16,
            "MTP norms are already source-exact"
        );
    }

    #[test]
    fn mtp_source_tier_knob_only_answers_to_its_own_value() {
        assert!(!qwen4_mtp_source_tier(None), "unset keeps the shipped tier");
        assert!(!qwen4_mtp_source_tier(Some("mq6")));
        assert!(qwen4_mtp_source_tier(Some("source")));
    }

    #[test]
    fn trunk_attention_projections_are_pinned_to_eight_bits() {
        let manifest = Qwen4Manifest::build(&pinned_config()).expect("pinned config manifest");
        let mut packed = 0usize;
        for entry in &manifest.weights {
            let trunk_attention =
                entry.name.contains(".linear_attn.") || entry.name.contains(".self_attn.");
            if !trunk_attention || entry.logical_shape.len() != 2 {
                continue;
            }
            let k = *entry.logical_shape.last().expect("rank checked above");
            if k % 128 != 0 {
                continue;
            }
            assert_eq!(
                entry.dtype,
                DType::Q8_0,
                "{} is a trunk attention projection and must ship at eight bits",
                entry.name
            );
            packed += 1;
        }
        // 36 linear-attention layers x (qkv, z, in_proj_a, in_proj_b, out_proj)
        // + 12 full-attention layers x (q, k, v, o, indexer qk)
        // + the MTP layer's five full-attention projections.
        assert_eq!(packed, 245, "trunk packing count");
    }

    /// The trunk's declared tier follows `HIPFIRE_QWEN4_TRUNK_TIER`, and only
    /// the trunk's does.
    ///
    /// The default the line above pins is the eight-bit recipe; `mq6` is the
    /// only value that selects the six-bit rung, and it moves the two hundred
    /// and forty rank-2 trunk attention/GDN matrices — not the MTP namespace,
    /// which reaches the manifest through its own constructor and keeps its own
    /// targets.
    #[test]
    fn trunk_tier_selector_moves_the_declared_trunk_target() {
        assert_eq!(qwen4_trunk_tier(None), DType::Q8_0, "unset knob default");
        assert_eq!(qwen4_trunk_tier(Some("mq6")), DType::MQ6G256V2);
        assert_eq!(
            qwen4_trunk_tier(Some("mfp4e8")),
            DType::Q8_0,
            "a retired value must not resurrect a third tier"
        );

        let config = pinned_config();
        let six = Qwen4Manifest::build_with_trunk_tier(&config, qwen4_trunk_tier(Some("mq6")))
            .expect("six-bit trunk manifest");
        let eight = Qwen4Manifest::build_with_trunk_tier(&config, qwen4_trunk_tier(None))
            .expect("eight-bit trunk manifest");
        assert_eq!(
            six.weights.len(),
            eight.weights.len(),
            "tier, not the entry set"
        );

        let mut moved = 0usize;
        let mut held = 0usize;
        for (six_entry, eight_entry) in six.weights.iter().zip(&eight.weights) {
            assert_eq!(six_entry.name, eight_entry.name, "entries must not move");
            let trunk_matrix = qwen4_trunk_matrix(&six_entry.name, &six_entry.logical_shape)
                && six_entry.layer.is_some();
            // The layer index is what separates the trunk declaration from the
            // MTP namespace's: MTP names are layerless by construction.
            if trunk_matrix {
                assert_eq!(
                    six_entry.dtype,
                    DType::MQ6G256V2,
                    "{} trunk target under HIPFIRE_QWEN4_TRUNK_TIER=mq6",
                    six_entry.name
                );
                assert_eq!(
                    eight_entry.dtype,
                    DType::Q8_0,
                    "{} trunk target with the knob unset",
                    eight_entry.name
                );
                moved += 1;
            } else {
                assert_eq!(
                    six_entry.dtype, eight_entry.dtype,
                    "{} must not follow the trunk selector",
                    six_entry.name
                );
            }
            // Under either declaration the trunk keeps admitting both rungs,
            // which is what keeps a published artifact of the other tier
            // loadable by this build.
            if trunk_matrix {
                for tier in [DType::BF16, DType::Q8_0, DType::MQ6G256V2] {
                    assert!(
                        six_entry.dtype_constraint.accepts(tier),
                        "{} must keep admitting {tier:?}",
                        six_entry.name
                    );
                }
            }
        }
        assert_eq!(moved, 240, "trunk matrices the selector moves");

        // The MTP namespace's five attention projections are the same class
        // under a separate declaration: Q8F16 under the six-bit build too.
        for entry in &six.weights {
            if entry.name.starts_with("mtp.")
                && qwen4_trunk_matrix(&entry.name, &entry.logical_shape)
            {
                assert_eq!(entry.dtype, DType::Q8_0, "{} MTP tier", entry.name);
                held += 1;
            }
        }
        assert_eq!(held, 5, "MTP attention projections left alone");
    }

    #[test]
    fn manifest_admits_quantized_matrices_and_keeps_source_exact_payloads_bf16() {
        let manifest = Qwen4Manifest::build(&pinned_config()).expect("pinned config manifest");
        for (name, layer, dtype) in [
            (
                "model.language_model.layers.3.mlp.experts.gate_up_proj",
                Some(3),
                ROUTED_GATE_UP_DTYPE,
            ),
            (
                "model.language_model.layers.3.mlp.experts.down_proj",
                Some(3),
                ROUTED_DOWN_DTYPE,
            ),
            (
                "mtp.layers.0.mlp.experts.gate_up_proj",
                None,
                ROUTED_GATE_UP_DTYPE,
            ),
            (
                "mtp.layers.0.mlp.experts.down_proj",
                None,
                ROUTED_DOWN_DTYPE,
            ),
        ] {
            let entry = manifest.entry(name, layer).expect("routed expert entry");
            assert_eq!(entry.dtype, dtype, "{name} target dtype");
            assert!(entry.dtype_constraint.accepts(DType::BF16));
            assert!(entry.dtype_constraint.accepts(dtype), "{name} source");
            assert!(
                !entry.dtype_constraint.accepts(DType::MQ6G256V2),
                "{name} must not accept an aligned-K family its geometry is not"
            );
        }

        // Dense matrices are quantized targets that still accept BF16 source
        // bytes, which is what lets an older artifact load against a newer
        // manifest.  The trunk's wide attention/GDN projections are also the
        // class whose packed tier the artifact chooses: the six-bit rung
        // carries MQ6G256V2 (qt47) there and the eight-bit rung Q8F16 (qt3).
        // Both must satisfy the entry's source contract, or a published rung
        // of one tier cannot be loaded by the tree that produced it.
        let assert_quantized = |name: &str, layer: Option<usize>| {
            let entry = manifest.entry(name, layer).expect("quantized matrix entry");
            assert_eq!(entry.dtype, DType::Q8_0, "{name} target dtype");
            assert!(entry.dtype_constraint.accepts(DType::BF16));
            assert!(entry.dtype_constraint.accepts(DType::Q8_0));
            assert!(
                entry.dtype_constraint.accepts(DType::MQ6G256V2),
                "{name} must admit the six-bit trunk rung"
            );
        };
        for (name, layer) in [
            (
                "model.language_model.layers.3.self_attn.q_proj.weight",
                Some(3),
            ),
            (
                "model.language_model.layers.2.linear_attn.in_proj_qkv.weight",
                Some(2),
            ),
            // The MTP layer's attention projections are the same class under
            // the model namespace, and reach the manifest through their own
            // constructor.
            ("mtp.layers.0.self_attn.q_proj.weight", None),
        ] {
            assert_quantized(name, layer);
        }

        let assert_bf16 = |name: &str, layer: Option<usize>| {
            let entry = manifest.entry(name, layer).expect("BF16 entry");
            assert_eq!(entry.dtype, DType::BF16, "{name} target dtype");
            assert!(entry.dtype_constraint.accepts(DType::BF16));
            assert!(!entry.dtype_constraint.accepts(DType::MQ4G256V2));
            assert!(!entry.dtype_constraint.accepts(DType::MQ4G128V2));
        };
        for name in ["mtp.fc_embedding.weight", "mtp.fc_hidden.weight"] {
            assert_bf16(name, None);
        }
        // The embedding: BF16 in the checkpoint, Q8 in the artifact.
        let entry = manifest
            .entry("model.language_model.embed_tokens.weight", None)
            .expect("embedding entry");
        assert_eq!(entry.dtype, DType::Q8_0);
        assert!(entry.dtype_constraint.accepts(DType::BF16));
        assert!(entry.dtype_constraint.accepts(DType::Q8_0));
        assert!(!entry.dtype_constraint.accepts(DType::MQ6G256V2));
        // The head ships six-bit and still admits the earlier Q8F16 tier.
        let entry = manifest.entry("lm_head.weight", None).expect("head entry");
        assert_eq!(entry.dtype, DType::MQ6G256V2);
        for dtype in [DType::BF16, DType::Q8_0, DType::MQ6G256V2] {
            assert!(
                entry.dtype_constraint.accepts(dtype),
                "head admits {dtype:?}"
            );
        }
        assert!(!entry.dtype_constraint.accepts(DType::MQ4G256V2));
        // The shared expert, the router, and the PLE projections stay
        // source-exact: they ride the sealed-MoE route or select among 512
        // experts, and neither is a wide attention projection.
        assert_bf16("model.language_model.layers.3.mlp.gate.weight", Some(3));
        assert_bf16(
            "model.language_model.layers.3.mlp.shared_expert.up_proj.weight",
            Some(3),
        );
        let ple = manifest
            .weights
            .iter()
            .find(|entry| entry.name.contains(".ple.") && entry.name.contains("key"))
            .map(|entry| (entry.name.clone(), entry.layer))
            .expect("PLE key projection entry");
        assert_bf16(&ple.0, ple.1);
        // The hyper-connection mixer and block-inject matrices are the thin
        // side of the widest reduction in the network; they keep source bytes.
        assert_bf16(
            "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
            None,
        );
        assert_bf16(
            "model.language_model.layers.3.attn_hyper_connection.block_inject_weight.weight",
            Some(3),
        );

        let ple = manifest
            .entry(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                Some(1),
            )
            .expect("PLE shard");
        assert_eq!(ple.dtype, DType::Q8_0);
        assert!(ple.dtype_constraint.accepts(DType::Q8_0));
        // A sealed artifact written before the Q8F16 tier carries BF16 rows and
        // must stay loadable: both tiers are admitted by the same declaration.
        assert!(ple.dtype_constraint.accepts(DType::BF16));
        assert!(!ple.dtype_constraint.accepts(DType::MQ4G128V2));
    }

    #[test]
    fn single_band_preserves_manifest_keys_references_and_assembly_order() {
        let config = pinned_config();
        let base = Qwen4Manifest::build_with_trunk_tier(&config, DType::Q8_0).unwrap();
        let scope = Qwen4BandScope::single(&config);
        let scoped = base.clone().with_band_scope(&config, &scope).unwrap();
        assert_eq!(scoped, base);
        let placements = scoped.placements_for_scope(&config, &scope, 0).unwrap();
        let keys: Vec<_> = base.resident_keys().collect();
        assert_eq!(
            placements
                .iter()
                .map(|p| (p.name.as_str(), p.layer))
                .collect::<Vec<_>>(),
            keys
        );
        let refs = build_layer_refs_scoped(&config, &scope).unwrap();
        assert_eq!(
            refs.iter().map(|layer| layer.layer).collect::<Vec<_>>(),
            (0..config.num_hidden_layers).collect::<Vec<_>>()
        );
        // Exercise the real assembly take order using the store's CPU-only
        // symbolic ownership seam. No GPU tensor is fabricated or allocated.
        let mut store = hipfire_runtime::weight_store::WeightStore::default();
        for entry in base
            .weights
            .iter()
            .filter(|entry| !entry.residency.is_external())
        {
            store
                .stage_alias(
                    &entry.name,
                    entry.layer,
                    0,
                    "test-source",
                    hipfire_runtime::weight_store::WeightProjection {
                        kind: hipfire_runtime::weight_store::WeightProjectionKind::Static,
                        axis: None,
                        rank: 0,
                        world_size: 1,
                        logical_shape: entry.logical_shape.clone(),
                        dtype: entry.dtype,
                    },
                )
                .unwrap();
        }
        let mut tx = WeightLoadTransaction::new(store);
        let weights =
            Qwen4Weights::assemble(&mut tx, &config, &base.identity.recipe, &placements, &scope)
                .unwrap();
        assert_eq!(
            weights
                .taken
                .iter()
                .map(|taken| (taken.key.name.as_str(), taken.key.layer))
                .collect::<Vec<_>>(),
            keys
        );
        assert!(tx.is_empty());
        for absolute in scope.band.clone() {
            assert_eq!(weights.layer(absolute).unwrap().layer, absolute);
        }
        assert!(weights.layer(config.num_hidden_layers).is_none());
    }

    #[test]
    fn pp2_and_pp6_partition_every_flash_next_key_exactly_once() {
        use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
        let config = pinned_config();
        let base = Qwen4Manifest::build_with_trunk_tier(&config, DType::Q8_0).unwrap();
        let expected: BTreeSet<_> = base
            .weights
            .iter()
            .map(|entry| (entry.name.clone(), entry.layer))
            .collect();
        for counts in [&[24, 24][..], &[9, 8, 8, 8, 8, 7][..]] {
            let mesh = DeviceMesh::rect(&[(DimKind::Pp, counts.len())]).unwrap();
            let placement = BandPlacement::from_counts(&mesh, 48, counts).unwrap();
            let mut union = BTreeSet::new();
            let mut states = Vec::new();
            for stage in 0..counts.len() {
                let scope = Qwen4BandScope::from_placement(&config, &placement, stage).unwrap();
                let local = base.clone().with_band_scope(&config, &scope).unwrap();
                let expected_order: Vec<_> = base
                    .weights
                    .iter()
                    .filter(|entry| scope.owns_entry(entry))
                    .map(|entry| (entry.name.as_str(), entry.layer))
                    .collect();
                assert_eq!(
                    local
                        .weights
                        .iter()
                        .map(|entry| (entry.name.as_str(), entry.layer))
                        .collect::<Vec<_>>(),
                    expected_order
                );
                let placements = base.placements_for_scope(&config, &scope, stage).unwrap();
                check_placement_set(&local, &placements).unwrap();
                assert!(placements.iter().all(|entry| entry.device == stage));
                assert_eq!(
                    build_layer_refs_scoped(&config, &scope)
                        .unwrap()
                        .iter()
                        .map(|layer| layer.layer)
                        .collect::<Vec<_>>(),
                    scope.band.clone().collect::<Vec<_>>()
                );
                for entry in &local.weights {
                    assert!(
                        union.insert((entry.name.clone(), entry.layer)),
                        "{}",
                        entry.name
                    );
                    if let Some(layer) = entry.layer {
                        assert!(scope.owns_layer(layer));
                    }
                    if entry.name.starts_with("mtp.") {
                        assert!(scope.output);
                        assert_eq!(entry.placement, PlacementHint::Pin(PinTarget::Output));
                    }
                }
                assert_eq!(local.metadata.is_empty(), !scope.input);
                states.extend(local.state);
            }
            assert_eq!(union, expected);
            assert_eq!(states.len(), base.state.len());
            for state in &base.state {
                assert_eq!(
                    states.iter().filter(|entry| *entry == state).count(),
                    base.state.iter().filter(|entry| *entry == state).count()
                );
            }
        }
    }

    /// A real 48-layer slab plan whose per-layer arena sizes all differ
    /// (layer `l` holds `(5 * l) % 64` MQ4 experts, the rest W3).
    fn nonuniform_slab_plan() -> std::sync::Arc<crate::expert_arena::ExpertArenaPlan> {
        use crate::expert_slabs::ExpertFormat;
        let tensors = crate::expert_slab_table::full_fn_headers(|layer, expert| {
            if expert < (layer * 5) % 64 {
                ExpertFormat::Mq4Xts
            } else {
                ExpertFormat::Mq3I4
            }
        });
        let table = crate::expert_slab_table::build_expert_slab_table(
            &crate::expert_slabs::ExpertSlabsV1::fn_v1(),
            &tensors,
        )
        .unwrap();
        std::sync::Arc::new(crate::expert_arena::ExpertArenaPlan::from_table(&table).unwrap())
    }

    #[test]
    fn slab_manifest_band_scope_keeps_own_arenas_and_pins_mtp_to_output() {
        use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
        let config = pinned_config();
        let plan = nonuniform_slab_plan();
        let base = Qwen4Manifest::build_with_recipe(
            &config,
            Qwen4TierRecipe::FlashNextSlab {
                trunk_tier: DType::Q8_0,
                mtp_source: false,
                arenas: plan.clone(),
            },
        )
        .unwrap();
        let arena_name =
            |layer: usize| format!("model.language_model.layers.{layer}.mlp.experts.slab_arena");
        let is_arena = |name: &str| name.ends_with(crate::expert_arena::ARENA_SUFFIX);
        let is_aggregate = |name: &str| {
            name.ends_with(".mlp.experts.gate_up_proj") || name.ends_with(".mlp.experts.down_proj")
        };
        // The whole-model declaration: one arena per trunk layer sized by
        // the plan, MTP aggregates kept, no trunk aggregates.
        let arena_bytes: BTreeSet<_> = plan.layers().iter().map(|layer| layer.bytes).collect();
        assert_eq!(arena_bytes.len(), 48, "every layer's arena size differs");
        for layer in 0..48 {
            let entry = base.entry(&arena_name(layer), Some(layer)).unwrap();
            assert_eq!(entry.dtype, DType::Raw);
            assert_eq!(entry.logical_shape, vec![plan.layer(layer).unwrap().bytes]);
        }
        let mtp_aggregates: Vec<_> = base
            .weights
            .iter()
            .filter(|entry| entry.name.starts_with("mtp.") && is_aggregate(&entry.name))
            .map(|entry| entry.name.clone())
            .collect();
        assert_eq!(
            mtp_aggregates,
            [
                "mtp.layers.0.mlp.experts.down_proj",
                "mtp.layers.0.mlp.experts.gate_up_proj"
            ]
        );
        assert!(base
            .weights
            .iter()
            .all(|entry| entry.name.starts_with("mtp.") || !is_aggregate(&entry.name)));

        for counts in [&[24, 24][..], &[16, 16, 16][..], &[9, 8, 8, 8, 8, 7][..]] {
            let mesh = DeviceMesh::rect(&[(DimKind::Pp, counts.len())]).unwrap();
            let placement = BandPlacement::from_counts(&mesh, 48, counts).unwrap();
            let mut arenas_seen = BTreeSet::new();
            let mut mtp_bands = 0;
            for stage in 0..counts.len() {
                let scope = Qwen4BandScope::from_placement(&config, &placement, stage).unwrap();
                let local = base.clone().with_band_scope(&config, &scope).unwrap();
                // Exactly this band's arenas, in layer order.
                let kept: Vec<_> = local
                    .weights
                    .iter()
                    .filter(|entry| is_arena(&entry.name))
                    .map(|entry| (entry.name.clone(), entry.layer.unwrap()))
                    .collect();
                let want: Vec<_> = scope
                    .band
                    .clone()
                    .map(|layer| (arena_name(layer), layer))
                    .collect();
                assert_eq!(kept, want, "{counts:?} band {:?}", scope.band);
                for (name, layer) in &kept {
                    assert!(arenas_seen.insert(name.clone()), "{name} on two bands");
                    let entry = local.entry(name, Some(*layer)).unwrap();
                    assert_eq!(entry.logical_shape, vec![plan.layer(*layer).unwrap().bytes]);
                }
                // No trunk aggregate survives; the MTP ones are output-only
                // and pinned there.
                let mtp: Vec<_> = local
                    .weights
                    .iter()
                    .filter(|entry| entry.name.starts_with("mtp.") && is_aggregate(&entry.name))
                    .collect();
                assert!(local
                    .weights
                    .iter()
                    .all(|entry| entry.name.starts_with("mtp.") || !is_aggregate(&entry.name)));
                if scope.output {
                    mtp_bands += 1;
                    assert_eq!(
                        mtp.iter()
                            .map(|entry| entry.name.as_str())
                            .collect::<Vec<_>>(),
                        mtp_aggregates
                    );
                    for entry in mtp {
                        assert_eq!(entry.placement, PlacementHint::Pin(PinTarget::Output));
                    }
                } else {
                    assert!(mtp.is_empty(), "{counts:?} band {:?}", scope.band);
                }
                // Placements name the same arenas the band keeps.
                let placements = base.placements_for_scope(&config, &scope, stage).unwrap();
                check_placement_set(&local, &placements).unwrap();
                assert_eq!(
                    placements
                        .iter()
                        .filter(|placement| is_arena(&placement.name))
                        .count(),
                    scope.band.len()
                );
            }
            assert_eq!(mtp_bands, 1);
            assert_eq!(arenas_seen.len(), 48);
        }
    }

    #[test]
    fn output_embedding_preserves_tied_source_without_input_allocation() {
        let config = pinned_config();
        let base = Qwen4Manifest::build_with_trunk_tier(&config, DType::Q8_0).unwrap();
        let scope = Qwen4BandScope {
            band: 24..48,
            input: false,
            output: true,
        };
        let local = base.with_band_scope(&config, &scope).unwrap();
        assert!(local
            .entry("model.language_model.embed_tokens.weight", None)
            .is_none());
        let tied = local.entry("mtp.embed_tokens.weight", None).unwrap();
        assert_eq!(
            tied.policy,
            ShardPolicy::Tied {
                source: "model.language_model.embed_tokens.weight".to_string()
            }
        );
        assert_eq!(tied.placement, PlacementHint::Pin(PinTarget::Output));
    }

    #[test]
    fn bands_refuse_invalid_endpoints_ple_cut_and_expert_parallelism() {
        use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
        let config = pinned_config();
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2)]).unwrap();
        let bad = BandPlacement::from_counts(&mesh, 48, &[1, 47]).unwrap();
        for stage in 0..2 {
            assert!(Qwen4BandScope::from_placement(&config, &bad, stage).is_err());
        }
        assert!(Qwen4BandScope::from_placement(&config, &bad, 2).is_err());
        assert!(Qwen4BandScope {
            band: 24..48,
            input: true,
            output: true
        }
        .validate(&config)
        .is_err());
        let ep = Qwen4Manifest::build_with_trunk_tier(&config, DType::Q8_0)
            .unwrap()
            .with_expert_parallel(Qwen4EpPlacement { rank: 0, ranks: 2 })
            .unwrap();
        let scope = Qwen4BandScope {
            band: 0..24,
            input: true,
            output: false,
        };
        assert!(matches!(
            ep.with_band_scope(&config, &scope),
            Err(WeightError::EpPlacement(_))
        ));
    }

    #[test]
    fn scoped_assembly_failure_returns_taken_handles_to_the_transaction() {
        let config = pinned_config();
        let manifest = Qwen4Manifest::build_with_trunk_tier(&config, DType::Q8_0).unwrap();
        let scope = Qwen4BandScope {
            band: 0..24,
            input: true,
            output: false,
        };
        let placements = manifest.placements_for_scope(&config, &scope, 0).unwrap();
        let mut store = hipfire_runtime::weight_store::WeightStore::default();
        for placement in placements.iter().take(5) {
            let entry = manifest.entry(&placement.name, placement.layer).unwrap();
            store
                .stage_alias(
                    &entry.name,
                    entry.layer,
                    0,
                    "test-source",
                    hipfire_runtime::weight_store::WeightProjection {
                        kind: hipfire_runtime::weight_store::WeightProjectionKind::Static,
                        axis: None,
                        rank: 0,
                        world_size: 1,
                        logical_shape: entry.logical_shape.clone(),
                        dtype: entry.dtype,
                    },
                )
                .unwrap();
        }
        let mut tx = WeightLoadTransaction::new(store);
        assert!(matches!(
            Qwen4Weights::assemble(
                &mut tx,
                &config,
                &manifest.identity.recipe,
                &placements,
                &scope,
            ),
            Err(WeightError::MissingPlacement { .. })
        ));
        assert_eq!(tx.len(), 5);
        for placement in placements.iter().take(5) {
            assert!(tx.contains(&placement.name, placement.layer, 0));
            assert!(tx.projection(&placement.name, placement.layer, 0).is_some());
        }
    }
}

/// CPU tests of the arena-aware teardown and in-place replacement. Device
/// buffers are fabricated descriptors that are never dereferenced; the fake
/// [`ReplaceOps`] records the exact operation order.
#[cfg(test)]
mod weight_store_arena_replacement {
    use super::*;
    use hip_bridge::DeviceBuffer;
    use hipfire_runtime::weight_store::{
        WeightPlacementKey, WeightProjection, WeightProjectionKind,
    };

    const ROOT_BASE: usize = 0x4000_0000;
    const TEMP_BASE: usize = 0x9000_0000;

    /// A borrowed arena view.
    fn view(ptr: usize, size: usize, dtype: DType, shape: &[usize]) -> GpuTensor {
        GpuTensor {
            // SAFETY: never dereferenced; tests only read the descriptor.
            buf: unsafe { DeviceBuffer::from_raw(ptr as *mut std::ffi::c_void, size) },
            shape: shape.to_vec(),
            dtype,
        }
    }

    /// An owned (non-borrowed) tensor descriptor. A VMM-owner tag is the one
    /// non-borrowed constructor that needs no device.
    fn owned(ptr: usize, size: usize, dtype: DType, shape: &[usize]) -> GpuTensor {
        GpuTensor {
            // SAFETY: never dereferenced or freed; the fake owner only logs.
            buf: unsafe { DeviceBuffer::from_vmm_owner(ptr as *mut std::ffi::c_void, size) },
            shape: shape.to_vec(),
            dtype,
        }
    }

    fn taken(handle: WeightHandle, m: usize, k: usize) -> TakenWeight {
        TakenWeight {
            key: WeightPlacementKey::new("w", Some(0), 0),
            handle,
            projection: WeightProjection {
                kind: WeightProjectionKind::Static,
                axis: None,
                rank: 0,
                world_size: 1,
                logical_shape: vec![m, k],
                dtype: DType::BF16,
            },
        }
    }

    #[derive(Default)]
    struct Fake {
        log: Vec<String>,
        fail_invalidate: bool,
        fail_copy: bool,
    }

    impl ReplaceOps for Fake {
        fn invalidate_arena_source(&mut self, ptr: usize) -> Result<bool, String> {
            self.log.push(format!("invalidate:{ptr:#x}"));
            if self.fail_invalidate {
                return Err("fake quiesce failure".into());
            }
            Ok(true)
        }
        fn write_host(&mut self, dst: &DeviceBuffer, bytes: &[u8]) -> Result<(), String> {
            self.log.push(format!(
                "write:{:#x}:{}",
                dst.as_ptr() as usize,
                bytes.len()
            ));
            Ok(())
        }
        fn copy_device(
            &mut self,
            dst: &DeviceBuffer,
            src: &DeviceBuffer,
            bytes: usize,
        ) -> Result<(), String> {
            self.log.push(format!(
                "copy:{:#x}<-{:#x}:{bytes}",
                dst.as_ptr() as usize,
                src.as_ptr() as usize
            ));
            if self.fail_copy {
                return Err("fake copy failure".into());
            }
            Ok(())
        }
        fn free(&mut self, tensor: GpuTensor) -> Result<(), String> {
            self.log
                .push(format!("free:{:#x}", tensor.buf.as_ptr() as usize));
            Ok(())
        }
        fn upload_qt44(&mut self, bytes: &[u8]) -> Result<GpuTensor, String> {
            self.log.push(format!("upload_qt44:{}", bytes.len()));
            Ok(owned(
                TEMP_BASE,
                bytes.len(),
                DType::MQ4G256V2,
                &[bytes.len()],
            ))
        }
    }

    fn ptr_of(taken: &TakenWeight) -> usize {
        let WeightHandle::Resident(tensor) = &taken.handle else {
            panic!("not resident")
        };
        tensor.buf.as_ptr() as usize
    }

    #[test]
    fn free_gpu_skips_borrowed_views_and_frees_owned_tensors() {
        let handles = vec![
            taken(
                WeightHandle::Resident(view(ROOT_BASE, 4096, DType::BF16, &[4, 512])),
                4,
                512,
            ),
            taken(
                WeightHandle::Resident(owned(TEMP_BASE, 64, DType::F32, &[16])),
                1,
                16,
            ),
            taken(WeightHandle::Alias("w".into()), 1, 1),
            taken(
                WeightHandle::Resident(view(ROOT_BASE + 4096, 256, DType::BF16, &[1, 128])),
                1,
                128,
            ),
        ];
        let mut freed = Vec::new();
        free_taken_handles(handles, |tensor| {
            freed.push(tensor.buf.as_ptr() as usize);
            Ok(())
        })
        .unwrap();
        assert_eq!(
            freed,
            vec![TEMP_BASE],
            "only the owned tensor reaches the free path"
        );
    }

    #[test]
    fn free_gpu_reports_the_first_owned_free_error_after_attempting_all() {
        let handles = vec![
            taken(
                WeightHandle::Resident(owned(TEMP_BASE, 64, DType::F32, &[16])),
                1,
                16,
            ),
            taken(
                WeightHandle::Resident(owned(TEMP_BASE + 64, 64, DType::F32, &[16])),
                1,
                16,
            ),
        ];
        let mut attempts = 0;
        let error = free_taken_handles(handles, |_| {
            attempts += 1;
            Err(hip_bridge::HipError::new(7, &format!("fail {attempts}")))
        })
        .unwrap_err();
        assert_eq!(attempts, 2);
        assert!(error.message.contains("fail 1"));
    }

    #[test]
    fn replaced_owned_weight_still_frees_the_old_tensor() {
        let mut weight = taken(
            WeightHandle::Resident(owned(TEMP_BASE, 4096, DType::BF16, &[4, 512])),
            4,
            512,
        );
        let copy = owned(TEMP_BASE + 0x1000, 544, DType::Q8_0, &[4, 512]);
        let mut ops = Fake::default();
        let sizes =
            replace_resident_with(&mut ops, &mut weight, Replacement::Tensor(copy)).unwrap();
        assert_eq!(sizes, (4096, 544));
        assert_eq!(ptr_of(&weight), TEMP_BASE + 0x1000, "the copy is installed");
        assert_eq!(ops.log, vec![format!("free:{TEMP_BASE:#x}")]);
    }

    #[test]
    fn every_requant_rule_replaces_in_place_keeping_pointer_and_span_capacity() {
        let (m, k) = (4usize, 512usize);
        let span = m * k * 2;
        for (target, expected) in [
            (DType::MQ2G256V2, m * (k / 256) * 72),
            (DType::MQ3G256V2, m * (k / 256) * 104),
            (DType::MQ4G256V2, m * (k / 256) * 136),
            (DType::MQ5G256V2, m * (k / 256) * 168),
            (DType::MQ6G256V2, m * (k / 256) * 200),
            (DType::Q8_0, m * (k / 32) * 34),
        ] {
            assert_eq!(replacement_payload_bytes(target, m, k), Some(expected));
            let mut weight = taken(
                WeightHandle::Resident(view(ROOT_BASE, span, DType::BF16, &[m, k])),
                m,
                k,
            );
            let copy = owned(TEMP_BASE, expected, target, &[m, k]);
            let mut ops = Fake::default();
            let (before, after) =
                replace_resident_with(&mut ops, &mut weight, Replacement::Tensor(copy)).unwrap();
            assert_eq!((before, after), (span, expected), "{target:?}");
            let WeightHandle::Resident(now) = &weight.handle else {
                panic!("not resident")
            };
            assert_eq!(
                now.buf.as_ptr() as usize,
                ROOT_BASE,
                "{target:?}: pointer stable"
            );
            assert!(now.buf.is_borrowed(), "{target:?}: still a borrowed view");
            assert_eq!(now.buf.size(), span, "{target:?}: span capacity preserved");
            assert_eq!(now.dtype, target);
            assert_eq!(now.shape, vec![m, k]);
            // One fence+invalidate, then write, then free only the temporary
            // owner. The planned span is never freed.
            assert_eq!(
                ops.log,
                vec![
                    format!("invalidate:{ROOT_BASE:#x}"),
                    format!("copy:{ROOT_BASE:#x}<-{TEMP_BASE:#x}:{expected}"),
                    format!("free:{TEMP_BASE:#x}"),
                ],
                "{target:?}"
            );
        }
    }

    #[test]
    fn qt44_replacement_writes_the_payload_directly_into_the_span() {
        let (m, k) = (4usize, 512usize);
        let span = m * k * 2;
        let payload = vec![7u8; m * (k / 256) * 136];
        let mut weight = taken(
            WeightHandle::Resident(view(ROOT_BASE, span, DType::BF16, &[m, k])),
            m,
            k,
        );
        let mut ops = Fake::default();
        let sizes = replace_resident_with(
            &mut ops,
            &mut weight,
            Replacement::Qt44(Cow::Borrowed(&payload[..])),
        )
        .unwrap();
        assert_eq!(sizes, (span, payload.len()));
        let WeightHandle::Resident(now) = &weight.handle else {
            panic!("not resident")
        };
        assert_eq!(now.buf.as_ptr() as usize, ROOT_BASE);
        assert_eq!(now.buf.size(), span);
        assert_eq!(now.dtype, DType::MQ4G256V2);
        assert_eq!(now.shape, vec![payload.len()]);
        assert_eq!(
            ops.log,
            vec![
                format!("invalidate:{ROOT_BASE:#x}"),
                format!("write:{ROOT_BASE:#x}:{}", payload.len()),
            ],
            "no temporary owner exists, so nothing is uploaded or freed"
        );
    }

    #[test]
    fn a_second_replacement_still_fits_the_original_span_capacity() {
        let (m, k) = (4usize, 512usize);
        let span = m * k * 2;
        let mut weight = taken(
            WeightHandle::Resident(view(ROOT_BASE, span, DType::BF16, &[m, k])),
            m,
            k,
        );
        let small = vec![1u8; m * (k / 256) * 72];
        let mut ops = Fake::default();
        replace_resident_with(
            &mut ops,
            &mut weight,
            Replacement::Qt44(Cow::Borrowed(&small)),
        )
        .unwrap();
        // A larger payload than the first, still within the planned span.
        let larger = vec![2u8; m * (k / 256) * 200];
        let mut ops = Fake::default();
        let (before, after) = replace_resident_with(
            &mut ops,
            &mut weight,
            Replacement::Qt44(Cow::Borrowed(&larger)),
        )
        .unwrap();
        assert_eq!((before, after), (span, larger.len()));
        assert_eq!(ptr_of(&weight), ROOT_BASE);
    }

    #[test]
    fn an_oversized_replacement_errors_before_any_write() {
        let span = 1024usize;
        let mut weight = taken(
            WeightHandle::Resident(view(ROOT_BASE, span, DType::BF16, &[4, 128])),
            4,
            256,
        );
        let oversized = vec![0u8; span + 1];
        let mut ops = Fake::default();
        let error = replace_resident_with(
            &mut ops,
            &mut weight,
            Replacement::Qt44(Cow::Borrowed(&oversized)),
        )
        .unwrap_err();
        assert!(error.contains("exceeds the planned arena span"), "{error}");
        assert!(
            ops.log.is_empty(),
            "no invalidate, write or fallback: {:?}",
            ops.log
        );
        assert_eq!(ptr_of(&weight), ROOT_BASE, "the original view is untouched");

        // A kernel-produced tensor too large for the span is refused the same
        // way, its temporary owner is freed, and nothing is written.
        let copy = owned(TEMP_BASE, 4 * 4 * 200, DType::MQ6G256V2, &[4, 1024]);
        let mut weight = taken(
            WeightHandle::Resident(view(ROOT_BASE, span, DType::BF16, &[4, 128])),
            4,
            1024,
        );
        let mut ops = Fake::default();
        let error =
            replace_resident_with(&mut ops, &mut weight, Replacement::Tensor(copy)).unwrap_err();
        assert!(error.contains("exceeds the planned arena span"), "{error}");
        assert_eq!(ops.log, vec![format!("free:{TEMP_BASE:#x}")]);
        assert_eq!(ptr_of(&weight), ROOT_BASE);
    }

    #[test]
    fn a_failed_fence_writes_nothing_and_keeps_the_original_view() {
        let (m, k) = (4usize, 512usize);
        let span = m * k * 2;
        let mut weight = taken(
            WeightHandle::Resident(view(ROOT_BASE, span, DType::BF16, &[m, k])),
            m,
            k,
        );
        let expected = m * (k / 256) * 136;
        let copy = owned(TEMP_BASE, expected, DType::MQ4G256V2, &[m, k]);
        let mut ops = Fake {
            fail_invalidate: true,
            ..Fake::default()
        };
        let error =
            replace_resident_with(&mut ops, &mut weight, Replacement::Tensor(copy)).unwrap_err();
        assert!(error.contains("fake quiesce failure"));
        assert_eq!(
            ops.log,
            vec![
                format!("invalidate:{ROOT_BASE:#x}"),
                format!("free:{TEMP_BASE:#x}")
            ],
            "no write; only the temporary owner is released"
        );
        let WeightHandle::Resident(now) = &weight.handle else {
            panic!("not resident")
        };
        assert_eq!(
            now.dtype,
            DType::BF16,
            "the weight keeps its original metadata"
        );

        let payload = vec![0u8; expected];
        let mut ops = Fake {
            fail_invalidate: true,
            ..Fake::default()
        };
        replace_resident_with(
            &mut ops,
            &mut weight,
            Replacement::Qt44(Cow::Borrowed(&payload)),
        )
        .unwrap_err();
        assert_eq!(
            ops.log,
            vec![format!("invalidate:{ROOT_BASE:#x}")],
            "no write on a failed fence"
        );
    }

    #[test]
    fn a_failed_copy_still_frees_the_temporary_and_keeps_the_view_metadata() {
        let (m, k) = (4usize, 512usize);
        let span = m * k * 2;
        let mut weight = taken(
            WeightHandle::Resident(view(ROOT_BASE, span, DType::BF16, &[m, k])),
            m,
            k,
        );
        let expected = m * (k / 256) * 136;
        let copy = owned(TEMP_BASE, expected, DType::MQ4G256V2, &[m, k]);
        let mut ops = Fake {
            fail_copy: true,
            ..Fake::default()
        };
        replace_resident_with(&mut ops, &mut weight, Replacement::Tensor(copy)).unwrap_err();
        assert_eq!(ops.log.last().unwrap(), &format!("free:{TEMP_BASE:#x}"));
        let WeightHandle::Resident(now) = &weight.handle else {
            panic!("not resident")
        };
        assert_eq!(now.dtype, DType::BF16);
    }

    #[test]
    fn replaced_view_does_not_free_the_root_at_teardown() {
        let (m, k) = (4usize, 512usize);
        let mut weight = taken(
            WeightHandle::Resident(view(ROOT_BASE, m * k * 2, DType::BF16, &[m, k])),
            m,
            k,
        );
        let mut ops = Fake::default();
        replace_resident_with(
            &mut ops,
            &mut weight,
            Replacement::Qt44(Cow::Owned(vec![0u8; m * (k / 256) * 136])),
        )
        .unwrap();
        let mut freed = 0;
        free_taken_handles(vec![weight], |_| {
            freed += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            freed, 0,
            "the retagged view is still borrowed: the root owner frees it"
        );
    }

    #[test]
    fn a_non_resident_weight_is_refused_and_the_temporary_freed() {
        let mut weight = taken(WeightHandle::Alias("src".into()), 4, 512);
        let copy = owned(TEMP_BASE, 544, DType::Q8_0, &[4, 512]);
        let mut ops = Fake::default();
        let error =
            replace_resident_with(&mut ops, &mut weight, Replacement::Tensor(copy)).unwrap_err();
        assert!(error.contains("not resident"));
        assert_eq!(ops.log, vec![format!("free:{TEMP_BASE:#x}")]);
    }
}
