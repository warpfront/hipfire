// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//! MoE kernel family: dispatching expert GEMM operations.
//!
//! Supports 3 variants:
//! - **IndexedGateUp**: gate+up projection for a single expert (indexed by token)
//! - **IndexedDown**: down projection for a single expert (indexed by token)
//! - **GroupedGemm**: batched grouped-expert GEMM (all experts in one launch)
//!
//! # Current status
//!
//! `run()` is the centralized single-token MoE decode entry — it delegates to
//! [`crate::pipeline::run_moe_decode`] (the GPU top-K fast path plus the generic
//! CPU-top-K fallback). The family owns resolution (`MoeDtypes` → `MoeResolution`);
//! the model passes only the dtype snapshot + k. One `DispatchCtx` is threaded
//! end-to-end from the call site through every inner GEMV. Scratch stays model-owned.
//! Grouped-GEMM prefill is a future arm (gated on `ShapeInfo.batch_size`).
//!
//! # qt=55 FP8E4M3B128 route (gfx942 only)
//!
//! qt=55 expert blocks are served by exactly one route: the sealed typed
//! [`MoeRouteCapability::Fp8B128Grouped`] route (gfx942 only; canonical
//! 512 experts / top-10 / hidden 2560 / routed + shared intermediate 640 / 640;
//! routed gate_up and down both qt=55, router and every shared projection
//! BF16). The architecture declares it through [`MoeRoutePolicy`] and
//! `pipeline::select_fp8_grouped_route` accepts it or fails closed with a
//! typed [`DispatchError`]; it never falls through to generic selection.
//!
//! Generic resolution ([`MoeResolution`], [`MoePrefillResolution`]) has no
//! qt=55 implementation. Given any qt=55 operand it returns a *refusal state*
//! — every route flag disabled, `fp8_b128_refused == true` — and the
//! `check_generic_route` method on each turns that state into the typed
//! refusal. Execution MUST call it (or `pipeline::reject_fp8_moe`) before
//! using a generic route for a layer whose route is not
//! [`MoeRouteCapability::Fp8B128Grouped`]; the generic prefill Path 0/1/2 and
//! decode gpu-top-k/CPU fallback therefore can never carry qt=55 bytes.

use rdna_compute::DType;
use rdna_compute::{Gpu, GpuTensor};

use crate::context::DispatchCtx;
use crate::families::gemv::{GivensRef, WeightRef};
use crate::pipeline::sealed_moe::PrefillRouteMode;
use crate::tables::KernelRegistry;
use crate::traits::KernelFamily;
use crate::types::*;

/// Grouped-MoE WMMA tile row count used by capacity bounds and scatter.
pub const MOE_GROUPED_BLOCK_M: usize = 16;

/// Host entries a binder uploaded into a routed-expert pointer table.
#[derive(Clone, Copy)]
pub struct MoePointerEntries<'a> {
    pub gate_up: &'a [usize],
    pub down: &'a [usize],
}

/// Placement byte: the expert's projections live in a device (VRAM) owner.
pub const EXPERT_TIER_DEVICE: u8 = 0;
/// Placement byte: the expert's projections live in a host-mapped owner.
pub const EXPERT_TIER_HOST: u8 = 1;

/// Neutral per-layer metadata of an explicit per-expert tier placement
/// (model-level device + host-mapped payload owners). Plain data only: the
/// loader that owns the tier converts its own view/lease into this, so no
/// arch-private type crosses the dispatch boundary.
///
/// `placement` and `codecs` hold one byte per layer-local expert id (the id the
/// kernels index with; model ids never reach dispatch). `any_host` is derived
/// from `placement`, never from a representative expert, and is for
/// conservative capability/path decisions only: it carries no arithmetic
/// meaning. `generation` is the owner-binding generation (a reload gets a new
/// one), `logical_digest` the placement plan's logical digest, and `keepalive`
/// the owner retention every captured or retained resource must hold until its
/// GPU work is retired.
#[derive(Clone)]
pub struct ExpertTierIdentity {
    generation: u64,
    logical_digest: [u8; 32],
    any_host: bool,
    placement: Box<[u8]>,
    codecs: Box<[u8]>,
    gate_up_entries: Box<[usize]>,
    down_entries: Box<[usize]>,
    keepalive: std::sync::Arc<dyn std::any::Any + Send + Sync>,
}

impl ExpertTierIdentity {
    /// `placement` entries must be [`EXPERT_TIER_DEVICE`] or
    /// [`EXPERT_TIER_HOST`]. Every table has one entry per layer-local expert
    /// and the same non-zero length; pointer entries are non-null absolute
    /// addresses (the frozen CPU mirrors of the uploaded device tables).
    pub fn new(
        generation: u64,
        logical_digest: [u8; 32],
        placement: &[u8],
        codecs: &[u8],
        gate_up_entries: &[usize],
        down_entries: &[usize],
        keepalive: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<Self, String> {
        let n = placement.len();
        if n == 0 || n != codecs.len() || n != gate_up_entries.len() || n != down_entries.len() {
            return Err(format!(
                "expert tier identity tables must be non-empty and equal length \
                 (placement {n}, codecs {}, gate/up entries {}, down entries {})",
                codecs.len(),
                gate_up_entries.len(),
                down_entries.len()
            ));
        }
        if let Some(index) = (0..n).find(|&i| gate_up_entries[i] == 0 || down_entries[i] == 0) {
            return Err(format!(
                "expert tier identity has a null pointer entry for expert {index}"
            ));
        }
        if let Some(index) = placement
            .iter()
            .position(|&tier| tier != EXPERT_TIER_DEVICE && tier != EXPERT_TIER_HOST)
        {
            return Err(format!(
                "expert tier identity placement byte {} for expert {index} is neither device nor host",
                placement[index]
            ));
        }
        Ok(Self {
            generation,
            logical_digest,
            any_host: placement.iter().any(|&tier| tier == EXPERT_TIER_HOST),
            placement: placement.into(),
            codecs: codecs.into(),
            gate_up_entries: gate_up_entries.into(),
            down_entries: down_entries.into(),
            keepalive,
        })
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn logical_digest(&self) -> &[u8; 32] {
        &self.logical_digest
    }

    /// True when at least one expert of the layer is host-mapped. Capability
    /// checks only; never select arithmetic from it.
    pub fn any_host(&self) -> bool {
        self.any_host
    }

    /// `(gate_up, down)` host-mapped residency of the layer's projections.
    /// Both are the layer-wide `any_host` bit: a tier places a pair together.
    pub fn host_projections(&self) -> (bool, bool) {
        (self.any_host, self.any_host)
    }

    pub fn placement(&self) -> &[u8] {
        &self.placement
    }

    pub fn codecs(&self) -> &[u8] {
        &self.codecs
    }

    /// Absolute GPU-visible gate/up projection addresses per layer-local
    /// expert: the frozen CPU mirror of the uploaded pointer table.
    pub fn gate_up_entries(&self) -> &[usize] {
        &self.gate_up_entries
    }

    /// Absolute GPU-visible down projection addresses (see
    /// [`Self::gate_up_entries`]).
    pub fn down_entries(&self) -> &[usize] {
        &self.down_entries
    }

    pub fn keepalive(&self) -> &std::sync::Arc<dyn std::any::Any + Send + Sync> {
        &self.keepalive
    }
}

impl PartialEq for ExpertTierIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.generation == other.generation
            && self.logical_digest == other.logical_digest
            && self.any_host == other.any_host
            && self.placement == other.placement
            && self.codecs == other.codecs
            && self.gate_up_entries == other.gate_up_entries
            && self.down_entries == other.down_entries
            && std::sync::Arc::ptr_eq(&self.keepalive, &other.keepalive)
    }
}

impl Eq for ExpertTierIdentity {}

impl std::fmt::Debug for ExpertTierIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertTierIdentity")
            .field("generation", &self.generation)
            .field("logical_digest", &self.logical_digest)
            .field("any_host", &self.any_host)
            .field("experts", &self.placement.len())
            .finish_non_exhaustive()
    }
}

/// `(gate_up, down)` host-mapped residency of a layer's routed experts.
/// An explicit tier binding wins over every adapter override: its answer is
/// the layer's `any_host`, never a representative expert's residency.
pub fn routed_host_mapped_projections(experts: &dyn RoutedExpertWeights) -> (bool, bool) {
    experts.tier_identity().map_or_else(
        || experts.host_mapped_projections(),
        ExpertTierIdentity::host_projections,
    )
}

/// Layer-wide host-mapped bit of a layer's routed experts; tier-first like
/// [`routed_host_mapped_projections`].
pub fn routed_host_mapped(experts: &dyn RoutedExpertWeights) -> bool {
    experts
        .tier_identity()
        .map_or_else(|| experts.host_mapped(), ExpertTierIdentity::any_host)
}

/// The complete semantic recipe selected by an architecture binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeRecipe {
    SoftmaxGatedShared {
        /// Keep the model's BF16 storage boundaries even when projections use F32 buffers.
        bf16_round_trip: bool,
        /// Accumulate routed experts before the shared residual.
        shared_after_combine: bool,
    },
    SigmoidRoutedNoShared,
}

impl MoeRecipe {
    pub(crate) fn bf16_round_trip(self) -> bool {
        matches!(
            self,
            Self::SoftmaxGatedShared {
                bf16_round_trip: true,
                ..
            }
        )
    }

    pub(crate) fn shared_after_combine(self) -> bool {
        matches!(
            self,
            Self::SoftmaxGatedShared {
                shared_after_combine: true,
                ..
            }
        )
    }
}

/// Selection bias of a sigmoid-style router (LFM2, MiniMax, DeepSeek-V4):
/// experts are chosen on `score + bias`, weighted by the unbiased scores,
/// normalized over the selection, then multiplied by `route_scale`.
#[derive(Clone, Copy)]
pub struct MoeSelectionBias<'a> {
    /// `[n_exp]` F32, read for selection only.
    pub bias: &'a GpuTensor,
    pub route_scale: f32,
    /// Where the selector's scores come from.
    pub scores: MoeScores,
}

/// Source of the scores the biased selector reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeScores {
    /// Sealed decode runs the router GEMV and `sigmoid_f32` itself.
    Sigmoid,
    /// The model already wrote activated scores into `MoeParams::router_logits`;
    /// sealed decode runs only the biased top-k over them.
    Precomputed,
}

/// Widest selection the biased sigmoid router and the k8 routed kernels take.
pub const MOE_BIASED_SIGMOID_MAX_K: usize = 8;

// ── Generic grouped MoE (sealed policies, kernel-limit admission) ──────────

pub use rdna_compute::ffn_activation::{
    CombineOrder, FfnActivation, RouterKind, FFN_ROUTER_MAX_EXPERTS, FFN_ROUTER_MAX_TOP_K,
};

/// Widest top-k the generic grouped-MoE program runs: the router selection and
/// the combine kernels carry up to [`FFN_ROUTER_MAX_TOP_K`] ranks, and the
/// indexed gate/up + down qt44 kernels and the grouped WMMA core take the rank
/// count as a runtime argument (`K_TOP` grid axis / `x_row_div`).
pub const MOE_GROUPED_MAX_TOP_K: usize = FFN_ROUTER_MAX_TOP_K;

/// Whether the generic grouped-MoE program (`pipeline::moe_program::GroupedMoeOp`)
/// can run `top_k` ranks over `n_experts` experts: `1 <= top_k <= 8`,
/// `top_k <= n_experts <= 1024`. Never pads the rank count with dummy experts.
#[inline]
pub fn moe_grouped_top_k_supported(top_k: usize, n_experts: usize) -> bool {
    (1..=MOE_GROUPED_MAX_TOP_K).contains(&top_k)
        && top_k <= n_experts
        && n_experts <= FFN_ROUTER_MAX_EXPERTS
}

/// The sealed Qwen softmax router (`moe_softmax_topk_renorm_k8*`,
/// `moe_router_softmax_topk_k8_*`) selects exactly eight experts: this is the
/// kernel contract of that router, not a model property. The generic
/// grouped-MoE program uses [`moe_grouped_top_k_supported`] instead.
#[inline]
pub fn moe_softmax_k8_router_top_k_supported(top_k: usize) -> bool {
    top_k == 8
}

/// Padded grouped-row bound of the WMMA grouped route: every touched expert
/// rounds its slot count up to [`MOE_GROUPED_BLOCK_M`]. Only experts that
/// receive at least one slot contribute padding.
pub fn grouped_moe_m_total_bound(total_slots: usize, n_experts: usize) -> usize {
    let live_expert_bound = total_slots.min(n_experts);
    let raw = total_slots + live_expert_bound * (MOE_GROUPED_BLOCK_M - 1);
    raw.div_ceil(MOE_GROUPED_BLOCK_M) * MOE_GROUPED_BLOCK_M
}

/// Logical versus physical widths of one generic grouped-MoE layer. Logical
/// `logical_mi` is the activation width; `down_k` is the physical (padded)
/// down-projection input width; the pad columns are exact zeros appended after
/// the activation and its cast, before the AWQ divide and the rotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupedMoeGeometry {
    pub n_experts: usize,
    pub top_k: usize,
    pub hidden: usize,
    /// Routed expert intermediate width (gate and up rows each).
    pub logical_mi: usize,
    /// Physical K of the routed down projection (`>= logical_mi`, `% 256 == 0`).
    pub down_k: usize,
    /// Shared expert intermediate width, `0` when the layer has no shared expert.
    pub shared_mi: usize,
    /// Physical K of the shared down projection (`0` without a shared expert).
    pub shared_down_k: usize,
}

impl GroupedMoeGeometry {
    fn refuse(variant: &'static str) -> DispatchError {
        DispatchError::UnsupportedVariant {
            family: "moe",
            variant,
            arch: "",
            quant: "",
        }
    }

    /// Capability check against the actual limits of the grouped kernels. Named
    /// refusal for every violated limit; never a silent reinterpretation.
    pub fn validate(&self) -> Result<(), DispatchError> {
        if !moe_grouped_top_k_supported(self.top_k, self.n_experts) {
            return Err(Self::refuse("grouped-op-top-k-outside-1..=8-or-experts"));
        }
        if self.hidden == 0 || self.hidden % 256 != 0 {
            return Err(Self::refuse("grouped-op-hidden-not-256-multiple"));
        }
        if self.logical_mi == 0 || self.down_k < self.logical_mi || self.down_k % 256 != 0 {
            return Err(Self::refuse("grouped-op-down-k-geometry"));
        }
        if self.shared_mi == 0 {
            if self.shared_down_k != 0 {
                return Err(Self::refuse("grouped-op-shared-geometry"));
            }
        } else if self.shared_down_k < self.shared_mi {
            return Err(Self::refuse("grouped-op-shared-geometry"));
        }
        Ok(())
    }
}

/// Architecture declaration, checked against both live operands and the
/// immutable specialized-kernel contract before any route is selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeRoutePolicy {
    pub capability: MoeRouteCapability,
    pub geometry: MoeRouteGeometry,
    pub formats: MoeRouteFormats,
    /// Sealed expert-slab layers only: the raw codec id of the layer's cold
    /// bucket (W3 = 1, E8P = 3), declared by the architecture from the
    /// layer's immutable slab metadata; `None` for a layer without cold
    /// experts (all MQ4) and for every other route. Selection validates it
    /// against the live dtype tables in its one pass, so lowering never
    /// rescans a table.
    pub slab_cold_codec: Option<u8>,
    /// Load-owned device address of E512 u8 membership, zero for the legacy route.
    pub expert_keep_mask: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeRouteGeometry {
    pub experts: usize,
    pub top_k: usize,
    pub hidden: usize,
    pub routed_intermediate: usize,
    pub shared_intermediate: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeRouteFormats {
    pub router: DType,
    pub shared_selector: DType,
    pub shared_gate: DType,
    pub shared_up: DType,
    pub shared_down: DType,
    pub routed_gate_up: DType,
    pub routed_down: DType,
}

/// QT44 gate/up plus QT53 down, with exactly 512 experts, top-10,
/// hidden=2560, routed/shared intermediate=640, and BF16 router/shared
/// projections (shared down may also be QT53). Arbitrary QT44/QT53
/// pairs are not admitted by this fixed-width router and combine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeRouteCapability {
    Qt44Qt53Grouped,
    /// The same fixed-shape route over an expert set whose every QT44/QT53
    /// header was verified symmetric (`zp == -8*sc`) on the device after
    /// load. Declared from that verification only, never from a dtype; it
    /// lowers exactly like [`Self::Qt44Qt53Grouped`] except that the opt-in
    /// symmetric IU4 prefill arm may take the routed experts.
    Qt44Qt53GroupedSymmetric,
    /// gfx942-only qt=55 FP8E4M3B128 grouped route over the same fixed
    /// 512/10/2560/640/640 geometry: routed gate_up and down are both qt=55,
    /// router and all shared projections BF16. Lowered only by the typed
    /// FP8 stages; admitted by `pipeline::select_fp8_grouped_route` and
    /// never by generic resolution.
    Fp8B128Grouped,
    /// Sealed expert-slab route (ctl3-hot): the same fixed
    /// 512/10/2560/640/640 geometry over experts whose codec is per expert,
    /// addressed through the layer's pointer tables. gfx1201 admits MQ4
    /// symmetric QT44/QT53, MQ3-I4 planar qt131/qt132 (W3) and E8P-I4
    /// planar qt135/qt136; gfx1151 admits MQ4 and W3, and E8P-I4 only once
    /// `rdna_compute::slab_admission::SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED` is true (refused
    /// today; its kernels are selectable, the one constant is the admission); gfx1100 admits
    /// MQ4 and E8P-I4 (not W3); W2 is
    /// refused on every arch. A layer carries at most one cold codec (W3 or E8P); see
    /// `rdna_compute::qwen4_slab_codec_admitted`. Decode rows
    /// (`route_rows <= 8`) run the K1 lane-union GEMV over the two codec
    /// buckets (cold = 0, MQ4 = 1); longer prefill runs the K2 grouped IU4
    /// kernels over codec-masked tile ids. Declared from the slab artifact's
    /// per-expert table only, never from a dtype.
    SlabCodecGrouped,
}

impl MoeRouteCapability {
    /// The source-bound QT44/QT53 route is admitted on any GPU when DeltaNet
    /// is enabled; the fixed geometry and format contract is checked separately.
    /// The FP8 route is arch-gated, so it is never admitted by this
    /// device-independent predicate (see [`Self::admitted_on_arch`]).
    pub(crate) fn admitted_on(self) -> bool {
        match self {
            Self::Qt44Qt53Grouped | Self::Qt44Qt53GroupedSymmetric | Self::SlabCodecGrouped => {
                cfg!(feature = "deltanet")
            }
            Self::Fp8B128Grouped => false,
        }
    }

    /// Device-aware admission: the QT44/QT53 routes keep their
    /// device-independent rule, the FP8 route requires gfx942.
    pub(crate) fn admitted_on_arch(self, arch: &rdna_compute::arch_caps::ArchCaps) -> bool {
        match self {
            Self::Qt44Qt53Grouped | Self::Qt44Qt53GroupedSymmetric => self.admitted_on(),
            Self::SlabCodecGrouped => {
                self.admitted_on() && rdna_compute::qwen4_slab_codec_admitted(arch.arch(), 0)
            }
            Self::Fp8B128Grouped => arch.is_gfx942(),
        }
    }

    /// The sealed expert-slab declaration (MQ4 plus the arch's cold codec: W3 or E8P-I4).
    pub fn is_slab(self) -> bool {
        matches!(self, Self::SlabCodecGrouped)
    }

    /// Any QT44/QT53 grouped declaration (the slab one included): they lower
    /// through the same fixed-shape stages. False for the FP8 route.
    pub fn is_qt44_qt53_grouped(self) -> bool {
        matches!(
            self,
            Self::Qt44Qt53Grouped | Self::Qt44Qt53GroupedSymmetric | Self::SlabCodecGrouped
        )
    }

    /// The gfx942 qt=55 FP8E4M3B128 grouped declaration.
    pub fn is_fp8_grouped(self) -> bool {
        matches!(self, Self::Fp8B128Grouped)
    }

    /// Any sealed grouped declaration: QT44/QT53 (either) or FP8E4M3B128.
    pub fn is_sealed_grouped(self) -> bool {
        self.is_qt44_qt53_grouped() || self.is_fp8_grouped()
    }
}

/// The one fixed expert/hidden geometry every grouped specialized route
/// (QT44/QT53 and FP8E4M3B128) is built for.
const FIXED_ROUTE_GEOMETRY: MoeRouteGeometry = MoeRouteGeometry {
    experts: 512,
    top_k: 10,
    hidden: 2560,
    routed_intermediate: 640,
    shared_intermediate: 640,
};

/// Exact geometry and wire-format contract of the available grouped kernels.
/// The route also requires `admitted_on` and live operands matching the policy.
#[allow(clippy::too_many_arguments)]
pub(crate) fn grouped_route_geometry_supported(
    policy: &MoeRoutePolicy,
    experts: usize,
    top_k: usize,
    hidden: usize,
    intermediate: usize,
    gate_up_k: usize,
    down_m: usize,
    down_k: usize,
    gate_up: DType,
    down: DType,
) -> bool {
    let geometry = policy.geometry;
    let formats = policy.formats;
    let fixed_geometry = FIXED_ROUTE_GEOMETRY;
    policy.capability.is_qt44_qt53_grouped()
        && geometry == fixed_geometry
        && experts == geometry.experts
        && top_k == geometry.top_k
        && hidden == geometry.hidden
        && intermediate == geometry.routed_intermediate
        && gate_up_k == geometry.hidden
        && down_m == geometry.hidden
        && down_k == geometry.routed_intermediate
        && formats.routed_gate_up == DType::MQ4G256V2
        && formats.routed_down == DType::MQ4G128V2
        && gate_up == formats.routed_gate_up
        && down == formats.routed_down
}

/// Whether `dtype` is a routed gate/up format of the sealed expert-slab route
/// on some arch (MQ4, W3 qt131, E8P-I4 qt135); per-arch membership is
/// [`slab_codec_pair`].
pub fn slab_gate_up_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::MQ4G256V2 | DType::MQ3I4G128R256 | DType::E8PI4G128R256
    )
}

/// Whether `dtype` is a routed down format of the sealed expert-slab route on
/// some arch (MQ4, W3 qt132, E8P-I4 qt136).
pub fn slab_down_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::MQ4G128V2 | DType::MQ3I4G128R128 | DType::E8PI4G128R128
    )
}

/// Slab codec byte of a routed `(gate_up, down)` pair (`0` = MQ4 symmetric
/// QT44/QT53, `1` = W3 qt131/qt132, `3` = E8P-I4 qt135/qt136), never crossed.
/// W2 has no codec byte: it is refused everywhere.
pub fn slab_codec_of_pair(gate_up: DType, down: DType) -> Option<u8> {
    match (gate_up, down) {
        (DType::MQ4G256V2, DType::MQ4G128V2) => Some(0),
        (DType::MQ3I4G128R256, DType::MQ3I4G128R128) => Some(1),
        (DType::E8PI4G128R256, DType::E8PI4G128R128) => Some(3),
        _ => None,
    }
}

/// Whether a routed `(gate_up, down)` pair is one slab codec that `arch`
/// admits (`rdna_compute::qwen4_slab_codec_admitted`).
pub fn slab_codec_pair(arch: &str, gate_up: DType, down: DType) -> bool {
    slab_codec_of_pair(gate_up, down)
        .is_some_and(|codec| rdna_compute::qwen4_slab_codec_admitted(arch, codec))
}

/// The K1/K2 codec bucket of an admitted slab dtype pair: the cold codec (W3
/// or E8P) is bucket 0, MQ4 bucket 1.
pub fn slab_bucket_of_pair(arch: &str, gate_up: DType, down: DType) -> Option<usize> {
    slab_bucket_of_pair_by(
        &|codec| rdna_compute::qwen4_slab_codec_admitted(arch, codec),
        gate_up,
        down,
    )
}

/// [`slab_bucket_of_pair`] over an explicit admission predicate (the pure decision: `admits`
/// is the architecture's [`rdna_compute::slab_admission::qwen4_slab_codec_admitted_with`]).
pub(crate) fn slab_bucket_of_pair_by(
    admits: &dyn Fn(u8) -> bool,
    gate_up: DType,
    down: DType,
) -> Option<usize> {
    let codec = slab_codec_of_pair(gate_up, down)?;
    admits(codec).then_some(usize::from(codec == 0))
}

/// The cold codec a layer's dtype metadata names, in one pass over a mixed
/// layer's per-expert pairs: `Some(Some(codec))` for a layer holding cold
/// experts of one codec, `Some(None)` for a uniform MQ4 layer, `None` when it
/// is not a slab layer (an unknown or crossed pair, a codec `admits` refuses,
/// W3 and E8P in one layer, or a mixed layer without both MQ4 and cold
/// experts).
fn layer_cold_codec(dtypes: &MoeDtypes<'_>, admits: &dyn Fn(u8) -> bool) -> Option<Option<u8>> {
    if dtypes.routed_has_mixed_experts {
        let (gate_up, down) = (dtypes.per_expert_gate_up?, dtypes.per_expert_down?);
        let (mut cold, mut mq4) = (None, false);
        // `admits` once per distinct codec, not per expert: it resolves the arch's
        // capability table and this pass runs per layer on every decode step.
        for (g, d) in gate_up.iter().zip(down) {
            let codec = slab_codec_of_pair(*g, *d)?;
            if codec == 0 {
                mq4 = true;
            } else if cold.map_or(true, |seen| seen == codec) {
                cold = Some(codec);
            } else {
                return None;
            }
        }
        (mq4 && admits(0) && cold.is_some_and(|codec| admits(codec))).then_some(cold)
    } else {
        let codec = slab_codec_of_pair(dtypes.routed_gate_up, dtypes.routed_down)
            .filter(|codec| admits(*codec))?;
        Some((codec != 0).then_some(codec))
    }
}

/// Raw `cold_codec` the runtime's K1 / tile-mask entries take for a layer that
/// declared `declared` ([`MoeRoutePolicy::slab_cold_codec`]): the declared
/// cold codec when `arch` admits it, or, for a layer without cold experts,
/// the arch's default cold codec
/// (`rdna_compute::qwen4_slab_default_cold_codec`; that bucket is never
/// launched). `None` when `arch` is not a slab arch or refuses the declared
/// codec.
pub fn slab_launch_cold_codec(arch: &str, declared: Option<u8>) -> Option<u8> {
    match declared {
        Some(codec) => rdna_compute::qwen4_slab_codec_admitted(arch, codec).then_some(codec),
        None => rdna_compute::qwen4_slab_default_cold_codec(arch),
    }
}

/// Exact geometry and operand contract of the sealed expert-slab route: the
/// fixed geometry, and routed experts that are either all one codec the
/// architecture admits (no per-expert tables) or a per-expert table of admitted codec
/// pairs, one entry per expert, holding MQ4 and one cold codec. The
/// policy's declared cold codec must be the one the dtypes name. `admits:
/// Some` (the architecture's `rdna_compute::qwen4_slab_codec_admitted` per codec) is
/// selection's one validating pass over the dtype tables; `None` is
/// the lowering-time shape check, which trusts that pass and never rescans
/// them (the stage that launches the kernels enforces the device).
#[allow(clippy::too_many_arguments)]
pub(crate) fn slab_route_geometry_supported(
    admits: Option<&dyn Fn(u8) -> bool>,
    policy: &MoeRoutePolicy,
    experts: usize,
    top_k: usize,
    hidden: usize,
    intermediate: usize,
    gate_up_k: usize,
    down_m: usize,
    down_k: usize,
    dtypes: &MoeDtypes<'_>,
    has_codec_table: bool,
) -> bool {
    let geometry = policy.geometry;
    let shape = policy.capability.is_slab()
        && geometry == FIXED_ROUTE_GEOMETRY
        && experts == geometry.experts
        && top_k == geometry.top_k
        && hidden == geometry.hidden
        && intermediate == geometry.routed_intermediate
        && gate_up_k == geometry.hidden
        && down_m == geometry.hidden
        && down_k == geometry.routed_intermediate;
    if !shape {
        return false;
    }
    if let Some(admits) = admits {
        if layer_cold_codec(dtypes, admits) != Some(policy.slab_cold_codec) {
            return false;
        }
    }
    if dtypes.routed_has_mixed_experts {
        let (Some(gate_up), Some(down)) = (dtypes.per_expert_gate_up, dtypes.per_expert_down)
        else {
            return false;
        };
        has_codec_table && gate_up.len() == experts && down.len() == experts
    } else {
        !has_codec_table && dtypes.per_expert_gate_up.is_none() && dtypes.per_expert_down.is_none()
    }
}

/// Exact geometry and wire-format contract of the gfx942 qt=55 grouped
/// kernels: the fixed geometry, routed gate_up and down both
/// [`DType::FP8E4M3B128`], and BF16 router/shared projections. The route also
/// requires `admitted_on_arch` and live operands matching the policy.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fp8_grouped_route_geometry_supported(
    policy: &MoeRoutePolicy,
    experts: usize,
    top_k: usize,
    hidden: usize,
    intermediate: usize,
    gate_up_k: usize,
    down_m: usize,
    down_k: usize,
    gate_up: DType,
    down: DType,
) -> bool {
    let geometry = policy.geometry;
    let formats = policy.formats;
    policy.capability.is_fp8_grouped()
        && geometry == FIXED_ROUTE_GEOMETRY
        && experts == geometry.experts
        && top_k == geometry.top_k
        && hidden == geometry.hidden
        && intermediate == geometry.routed_intermediate
        && gate_up_k == geometry.hidden
        && down_m == geometry.hidden
        && down_k == geometry.routed_intermediate
        && [
            formats.router,
            formats.shared_selector,
            formats.shared_gate,
            formats.shared_up,
            formats.shared_down,
        ]
        .iter()
        .all(|dtype| *dtype == DType::BF16)
        && formats.routed_gate_up == DType::FP8E4M3B128
        && formats.routed_down == DType::FP8E4M3B128
        && gate_up == formats.routed_gate_up
        && down == formats.routed_down
}

/// Typed refusal returned whenever a qt=55 operand would reach a generic
/// (non-FP8) MoE route.
pub(crate) fn fp8_b128_generic_refusal() -> DispatchError {
    DispatchError::UnsupportedVariant {
        family: "moe",
        variant: "fp8b128_specialized_route_only",
        arch: "",
        quant: "FP8E4M3B128",
    }
}

/// Source of the activation consumed by a MoE recipe.
#[derive(Clone, Copy)]
pub enum MoeNormalization<'a> {
    Provided,
    RmsNorm {
        weight: &'a GpuTensor,
        plain_out: &'a GpuTensor,
        eps: f32,
    },
}

/// Dtypes for the optional replicated/shared expert.
#[derive(Clone, Copy, Debug)]
pub struct MoeSharedDtypes {
    pub selector: DType,
    pub gate: DType,
    pub up: DType,
    pub down: DType,
}

/// Borrowed weights for the optional replicated/shared expert.
#[derive(Clone, Copy)]
pub struct MoeSharedWeights<'a> {
    pub selector: WeightRef<'a>,
    pub gate: WeightRef<'a>,
    pub up: WeightRef<'a>,
    pub down: WeightRef<'a>,
}

/// Decode scratch and weights for the optional replicated/shared expert.
#[derive(Clone, Copy)]
pub struct MoeSharedDecode<'a> {
    pub weights: MoeSharedWeights<'a>,
    pub intermediate: usize,
    pub scalar: &'a GpuTensor,
    pub gate_out: &'a GpuTensor,
    pub up_out: &'a GpuTensor,
}

/// Prefill scratch and weights for the optional replicated/shared expert.
#[derive(Clone, Copy)]
pub struct MoeSharedPrefill<'a> {
    pub weights: MoeSharedWeights<'a>,
    pub intermediate: usize,
    pub scalar: &'a GpuTensor,
    pub gate_out: &'a GpuTensor,
    pub up_out: &'a GpuTensor,
    pub rotated: &'a GpuTensor,
}

/// Router projection policy for Q8 prefill inputs.
#[derive(Clone, Copy)]
pub enum MoeQ8RouterPolicy<'a> {
    DispatcherEntry,
    FreshFp16Wmma {
        enabled: bool,
        scratch: &'a GpuTensor,
    },
}

/// Complete prefill prologue declaration.
#[derive(Clone, Copy)]
pub struct MoePrefillPrelude<'a> {
    pub normalization: MoeNormalization<'a>,
    pub router: WeightRef<'a>,
    pub router_logits: &'a GpuTensor,
    pub router_scores: &'a GpuTensor,
    pub norm_topk_prob: bool,
    pub route: PrefillRouteMode<'a>,
    pub shared: Option<MoeSharedPrefill<'a>>,
    pub q8_router_policy: MoeQ8RouterPolicy<'a>,
}

/// Whether the Qwen softmax/shared gate quartet has the exact V1 MQ4 layout
/// consumed by the fused gate-side kernel.
pub fn softmax_shared_gate_uniform_mq4(
    router: DType,
    selector: DType,
    gate: DType,
    up: DType,
) -> bool {
    [router, selector, gate, up]
        .into_iter()
        .all(|dtype| dtype == DType::MQ4G256)
}

/// Whether normalization may write the Qwen softmax/shared activation directly
/// in the prerotated basis consumed by the fused gate-side operation.
///
/// V1 is dtype-selected on every architecture, matching the established
/// `fused_qkvza_hfq4g256` route. V2 is restricted to its exact Ornith quartet,
/// no AWQ companions, and the two architectures with the matching fused kernel.
pub fn softmax_shared_gate_prerotated(
    ctx: &DispatchCtx,
    router: &WeightRef<'_>,
    shared: &MoeSharedWeights<'_>,
) -> bool {
    if softmax_shared_gate_uniform_mq4(
        router.dtype,
        shared.selector.dtype,
        shared.gate.dtype,
        shared.up.dtype,
    ) {
        return true;
    }
    if !(ctx.arch.is_gfx1100() || ctx.arch.is_gfx1201()) {
        return false;
    }
    softmax_shared_gate_mq4v2_layouts([
        (router.dtype, router.m, router.k, router.awq_scale.is_some()),
        (
            shared.selector.dtype,
            shared.selector.m,
            shared.selector.k,
            shared.selector.awq_scale.is_some(),
        ),
        (
            shared.gate.dtype,
            shared.gate.m,
            shared.gate.k,
            shared.gate.awq_scale.is_some(),
        ),
        (
            shared.up.dtype,
            shared.up.m,
            shared.up.k,
            shared.up.awq_scale.is_some(),
        ),
    ])
}

fn softmax_shared_gate_mq4v2_layouts(layouts: [(DType, usize, usize, bool); 4]) -> bool {
    let expected = [(256, 2_048), (1, 2_048), (512, 2_048), (512, 2_048)];
    layouts
        .into_iter()
        .zip(expected)
        .all(|((dtype, m, k, has_awq), (want_m, want_k))| {
            dtype == DType::MQ4G256V2 && m == want_m && k == want_k && !has_awq
        })
}
// ── MoE eligibility lattice ────────────────────────────

/// Routed-expert tiers the mixed-tier graded decode path can execute: the
/// tiers for which per-tier indexed gate_up/down GEMV kernels exist (served
/// on-device via `run_moe_decode`'s `expert_dtype_tags` branch). A per-expert
/// tier table containing any other DType
/// cannot be served by the mixed path and is rejected up front with a clear
/// error rather than failing deep in the per-bucket dispatch.
///
/// Includes V1 affine MQ4/MQ6, Paro, and the dual-half V2 layouts (qt44/qt47)
/// consumed by mixed kernel branch tags 7..18. V1/V2 must stay distinct —
/// collapsing either pair silently corrupts scale/zero decode.
pub const MIXED_SUPPORTED_TIERS: [DType; 5] = [
    DType::MQ4G256,
    DType::MQ6G256,
    DType::ParoQ4G128,
    DType::MQ4G256V2,
    DType::MQ6G256V2,
];

/// Per-layer dtype snapshot the MoE eligibility lattice reads. Built by the
/// model from its weight structs; kept dtype-only so this stays GPU-free and
/// the dispatch crate needs no dependency on any arch crate.
///
/// `experts_all_gate_up_mq4` mirrors the `ffn.experts.iter().all(..)` clause
/// the original `gate_side_mq4` check used (qwen35.rs:4598-4605). The routed
#[derive(Clone, Copy)]
pub struct MoeDtypes<'a> {
    pub router: DType,
    /// Optional shared-expert projection dtypes. `None` is a first-class
    /// no-shared recipe, independent of `has_paro_shared`.
    pub shared: Option<MoeSharedDtypes>,
    pub experts_all_gate_up_mq4: bool,
    pub routed_gate_up: DType, // ffn.experts[0].gate_up
    pub routed_down: DType,    // ffn.experts[0].down
    /// Per-expert mixed routed dtype: experts in one layer carry DIFFERENT
    /// gate_up and/or down dtypes (N-tier graded: MQ6 hot / MQ4 mid / MQ2L
    /// or MQ3L or E8-family cold), so `routed_gate_up` / `routed_down`
    /// (= experts[0]) are NOT representative.
    pub routed_has_mixed_experts: bool,
    pub has_paro_shared: bool, // ffn.paro_shared.is_some()
    /// Per-expert gate_up tiers for intra-layer mixed-tier dispatch.
    pub per_expert_gate_up: Option<&'a [DType]>,
    /// Per-expert down tiers (parallel to `per_expert_gate_up`).
    pub per_expert_down: Option<&'a [DType]>,
}

impl MoeDtypes<'_> {
    pub fn has_mq6_projection(&self) -> bool {
        let shared_has_mq6 = self.shared.is_some_and(|shared| {
            [shared.selector, shared.gate, shared.up, shared.down]
                .iter()
                .any(|dt| matches!(*dt, DType::MQ6G256 | DType::MQ6G256V2))
        });
        shared_has_mq6
            || [self.routed_gate_up, self.routed_down]
                .iter()
                .any(|dt| matches!(*dt, DType::MQ6G256 | DType::MQ6G256V2))
    }

    /// Any operand (router, shared projection, routed gate_up/down, or any
    /// per-expert tier) is qt=55 FP8E4M3B128. Generic routes refuse these.
    pub fn has_fp8_b128_projection(&self) -> bool {
        let is_fp8 = |dtype: DType| dtype == DType::FP8E4M3B128;
        self.shared.is_some_and(|shared| {
            [shared.selector, shared.gate, shared.up, shared.down]
                .into_iter()
                .any(is_fp8)
        }) || [self.router, self.routed_gate_up, self.routed_down]
            .into_iter()
            .any(is_fp8)
            || self
                .per_expert_gate_up
                .is_some_and(|values| values.iter().copied().any(is_fp8))
            || self
                .per_expert_down
                .is_some_and(|values| values.iter().copied().any(is_fp8))
    }

    /// The one narrow qwen35 qt53 routed-down typed fallback shape: uniform
    /// routed gate/up `MQ4G256V2` (qt44) + uniform routed down `MQ4G128V2`
    /// (qt53), no mixed tiers or per-expert tag shape, no Paro, and qt53 (or
    /// any planar MQ-I4 / qt=55) nowhere else (router, shared selector,
    /// shared gate/up/down). Pure dtype shape only; tags and AWQ sidecars are
    /// not part of this snapshot and are refused by the pipeline stages.
    /// Admission of this shape applies only to MoE without a route policy
    /// (callers reach it through `route.is_none()`).
    pub fn qt53_routed_down_fallback(&self) -> bool {
        let uniform = |table: Option<&[DType]>, expected: DType| {
            table.map_or(true, |values| values.iter().all(|dtype| *dtype == expected))
        };
        let qt53_or_mq_i4 = |dtype: DType| dtype == DType::MQ4G128V2 || dtype.is_mq_i4();
        self.routed_gate_up == DType::MQ4G256V2
            && self.routed_down == DType::MQ4G128V2
            && !self.routed_has_mixed_experts
            && !self.has_paro_shared
            && uniform(self.per_expert_gate_up, DType::MQ4G256V2)
            && uniform(self.per_expert_down, DType::MQ4G128V2)
            && !qt53_or_mq_i4(self.router)
            && !self.shared.is_some_and(|shared| {
                [shared.selector, shared.gate, shared.up, shared.down]
                    .into_iter()
                    .any(qt53_or_mq_i4)
            })
            && !self.has_fp8_b128_projection()
    }
}

/// Typed refusal for the qwen35 qt53 routed-down fallback: the shape is
/// admitted but the live device/route cannot launch it, so it is refused
/// before any launch (never a silent per-expert GEMV or CPU fallback).
pub(crate) fn qt53_fallback_unsupported(variant: &'static str) -> DispatchError {
    DispatchError::UnsupportedVariant {
        family: "moe",
        variant,
        arch: "",
        quant: "MQ4G128V2",
    }
}

/// Resolved fused-vs-fallback eligibility for one MoE decode layer. This IS the
/// routing-config logic, relocated from `moe_ffn_decode_impl` into one typed,
/// testable place (review finding #1). Pure function of `MoeDtypes` + k.
#[derive(Clone, Copy, Debug)]
pub struct MoeResolution {
    pub gate_side_mq4: bool,
    /// Router + shared expert gate/up are an exact-uniform MQ4G256 V1
    /// quartet. The fused gate path is independent of routed-expert dtype.
    /// MQ4G256V2 is admitted separately via `gate_fusable_mq4v2` (exact V2
    /// quartet → `fused_qkvza_hfq4g256_mq4v2`); mixed V1/V2 stays non-fusable.
    pub gate_fusable: bool,
    /// Router + shared scalar gate + shared expert gate/up are an
    /// exact-uniform MQ4G256V2 quartet. Independent of routed-expert dtype.
    /// Mixed V1/V2 gate-side dtypes never set this (or `gate_fusable`).
    pub gate_fusable_mq4v2: bool,
    pub routed_indexable_mq4: bool,
    pub routed_indexable_mq4v2: bool,
    pub routed_indexable_mq5: bool,
    pub routed_indexable_mq6: bool,
    /// Uniform all-MQ6G256V2 (qt47) routed experts. Gated on BOTH gate_up and
    /// down being V2 — qt47 dual-f16-grid header is incompatible with V1
    /// MQ6G256's f32 header; a split pairing must never claim either arm.
    pub routed_indexable_mq6v2: bool,
    /// Mixed routed experts: gate_up MQ4, down MQ6 (the "mq6-down" lever —
    /// promote only the sensitive residual-write projection to 6-bit while
    /// gate_up stays 4-bit). Indexable on the decode GPU-top-K path: gate_up
    /// uses the MQ4 indexed GEMV, down uses the MQ6 indexed GEMV, silu+rotate
    /// (optionally AWQ) is weight-agnostic. Decode-only (prefill Path-0 on
    /// gfx9* has no MQ6 down arm; eval scores per-token = decode).
    pub routed_indexable_mixed_gu4_dn6: bool,
    pub routed_indexable_paro: bool,
    /// Uniform all-MQ2-Lloyd routed experts (gate_up == down == MQ2G256Lloyd).
    /// Reuses the ds4/minimax indexed Lloyd MoE GEMVs on the decode GPU-top-K
    /// path: gate_up uses the MQ2-Lloyd indexed GEMV, down uses the MQ2-Lloyd
    /// atomic-residual GEMV (self-combining -> no separate down combine).
    pub routed_indexable_mq2lloyd: bool,
    /// Uniform all-MQ3-Lloyd routed experts (gate_up == down == MQ3G256Lloyd).
    /// Same indexed-Lloyd decode path as mq2lloyd, MQ3 launchers.
    pub routed_indexable_mq3lloyd: bool,
    /// Routed experts whose gate_up and down are each drawn, INDEPENDENTLY, from
    /// the codebook family `{MQ2G256Lloyd, MQ3G256Lloyd, MQ2G256GL, MQ3G256GL}` —
    /// the per-projection allocation (e.g. gate_up 2-bit, down 3-bit) that puts
    /// the cheap bits on the larger projection and the accurate ones on the
    /// residual write. Indexable because `run_moe_decode` already picks the
    /// gate_up and down GEMVs from their own dtypes rather than a coupled flag,
    /// and because ALL FOUR down kernels self-combine via atomicAdd — so
    /// `routed_down_self_combines` (keyed on `routed_down` alone) stays correct
    /// and the shared down-combine is skipped exactly once. silu+rotate is
    /// weight-agnostic (it reads activations only). Subsumes the two uniform
    /// Lloyd arms above and the uniform GL cases.
    ///
    /// Lloyd and GL are freely mixable across the two projections: both are
    /// FWHT-G256 formats consuming the same rotated activation, and each GEMV is
    /// selected from its own projection's dtype. The ONLY thing that differs is
    /// where the codebook comes from (per-group fp16 header vs scalar kernel
    /// args), which is entirely inside the launcher.
    ///
    /// Decode-only: batched prefill rejects MoE MQ3-Lloyd outright (see
    /// `moe_ffn_has_mq3_experts_uniform` in hipfire-arch-qwen35), which already
    /// blocks the pre-existing uniform MQ3-Lloyd path too; the GL dtypes are
    /// likewise not admitted by `moe_ffn_batched_admissible_for_dtypes`, so a
    /// GL model prefills through the per-token path.
    pub routed_indexable_mixed_lloyd: bool,
    /// Per-expert N-tier graded routed experts (MQ6 hot / MQ4 mid / MQ2L or
    /// MQ3L cold, applied to BOTH gate_up and down). Indexable on the decode
    /// GPU-top-K path via the merged dtype-tag-branched gate_up AND down
    /// kernels. The merged down writes the EXPANDED buffer for all dtypes →
    /// the single shared `moe_down_combine_k8_batched` runs (NOT Lloyd atomic
    /// self-combine). silu+rotate is weight-agnostic (unchanged).
    pub routed_indexable_mixed_per_expert: bool,
    /// Uniform UNROTATED Lloyd routed experts (MQ2G256LloydU, qt=51) on BOTH
    /// gate_up and down. Binds the same indexed MQ2-Lloyd kernels as qt19 but
    /// consumes x in the natural basis (`needs_x_rot_local == false`).
    pub routed_indexable_mq2lloyd_u: bool,
    pub use_gpu_topk: bool,
    pub needs_x_rot_local: bool,
    /// True when a per-expert tier table is `Some` AND contains >1 distinct
    /// DType — the layer's routed experts span multiple quant tiers and need
    /// the bucketed dispatch path (Task 3). `None` tables or all-equal `Some`
    /// tables leave this `false` ⇒ unchanged uniform fast path.
    pub mixed: bool,
    /// Refusal state: a qt=55 operand is present, so no generic route flag is
    /// set. Resolution stays infallible; [`Self::check_generic_route`] raises
    /// the typed refusal at execution.
    pub fp8_b128_refused: bool,
    /// Uniform qt44 gate/up + qt53 down without a route policy (see
    /// [`MoeDtypes::qt53_routed_down_fallback`]): lowers on the GPU top-K
    /// indexed path with the F32 FWHT128 down recipe. Decode requires `k == 8`.
    pub routed_indexable_qt53_f32: bool,
    /// Refusal state: the qt53 fallback shape is present but `k != 8` leaves no
    /// GPU top-K route; [`Self::check_generic_route`] raises the typed refusal
    /// (the per-expert CPU fallback has no qt53 GEMV).
    pub qt53_fallback_refused: bool,
}

impl MoeResolution {
    /// Arch-agnostic entry. The E8 indexed/grouped kernels exist on the RDNA3
    /// wave32-WMMA family (gfx11; `arch_has_e8_wmma`); passing `false` here routes
    /// E8 to the CPU-top-K fallback — preserving every existing caller + test.
    pub fn resolve(d: &MoeDtypes<'_>, k: usize) -> Self {
        Self::resolve_arch(d, k, false)
    }

    pub fn resolve_arch(d: &MoeDtypes<'_>, k: usize, arch_has_e8_wmma: bool) -> Self {
        if d.has_fp8_b128_projection() {
            return Self::fp8_b128_refused();
        }
        use DType::*;
        // The fused four-weight gate kernel is admitted only for the exact
        // MQ4G256 V1 quartet. The exact MQ4G256V2 quartet is admitted on a
        // separate predicate (`gate_fusable_mq4v2`) that routes to the V2
        // scalar fused launcher. Mixed V1/V2 gate-side stays on the generic
        // four-GEMV path. Independent of routed-expert dtype: all rotated MQ
        // families consume the same FwhtG256 activation.
        let shared = d.shared;
        // A no-shared recipe cannot enter any shared/gate-quartet fusion.
        // `has_paro_shared` remains independent metadata for routed Paro.
        let gate_fusable = shared.is_some_and(|shared| {
            d.router == MQ4G256
                && shared.selector == MQ4G256
                && shared.gate == MQ4G256
                && shared.up == MQ4G256
        });
        let gate_fusable_mq4v2 = shared.is_some_and(|shared| {
            d.router == MQ4G256V2
                && shared.selector == MQ4G256V2
                && shared.gate == MQ4G256V2
                && shared.up == MQ4G256V2
        });
        // gate_side_mq4 keeps the stricter all-MQ4 meaning (incl. routed experts)
        // for the rotate/AWQ branch + callers that assume a uniform-MQ4 FFN.
        let gate_side_mq4 = gate_fusable && d.experts_all_gate_up_mq4;

        let routed_gate_up_mq4 = d.routed_gate_up == MQ4G256;
        // qt44/qt45 are FWHT-G256 formats exactly like qt13 — their kernels read
        // the ROTATED activations. Omitting them from `needs_x_rot_local` below
        // feeds unrotated x into rotated weights, which is silent: the model
        // still emits fluent text. Measured on Ornith 1.5 35B-A3B, prefill KLD
        // was 0.993 against 0.044 on the per-token path for the SAME artifact.
        let routed_gate_up_mq4v2 = d.routed_gate_up == MQ4G256V2;
        let routed_gate_up_mq4c = d.routed_gate_up == MQ4CG256;
        let routed_gate_up_mq5 = d.routed_gate_up == MQ5G256;
        let routed_gate_up_mq6 = d.routed_gate_up == MQ6G256;
        // qt47. FWHT-G256 dual-half header like qt44 — kernels read rotated x.
        let routed_gate_up_mq6v2 = d.routed_gate_up == MQ6G256V2;
        let routed_gate_up_paro = d.routed_gate_up == ParoQ4G128 && d.has_paro_shared;
        let routed_gate_up_mq2lloyd = d.routed_gate_up == MQ2G256Lloyd;
        let routed_gate_up_mq3lloyd = d.routed_gate_up == MQ3G256Lloyd;
        // UNROTATED Lloyd sibling (qt=51). Same kernels, same byte layout —
        // the ONLY difference is that it must not rotate x.
        let routed_gate_up_mq2lloyd_u = d.routed_gate_up == MQ2G256LloydU;

        let routed_indexable_mq4 = (d.routed_down == MQ4G256) && routed_gate_up_mq4;
        // qt44. Gated on BOTH sides being MQ4G256V2, like every other uniform
        // pairing: the indexed GEMVs decode qt44's dual-f16-grid header, and
        // handing one a qt13 f32-header row reads the scale/zero as the two
        // halves of a float — silently wrong output, not a fault.
        let routed_indexable_mq4v2 = (d.routed_down == MQ4G256V2) && routed_gate_up_mq4v2;
        let routed_indexable_mq5 = (d.routed_down == MQ5G256) && routed_gate_up_mq5;
        let routed_indexable_mq6 = (d.routed_down == MQ6G256) && routed_gate_up_mq6;
        // qt47. BOTH sides MQ6G256V2 — same dual-half hazard as mq4v2: V1 MQ6
        // stores f32 scale/zero while V2 stores two f16 pairs; same 200 B
        // stride so a mis-pair is silent fluent garbage, not a fault.
        let routed_indexable_mq6v2 = (d.routed_down == MQ6G256V2) && routed_gate_up_mq6v2;
        let routed_indexable_mixed_gu4_dn6 = routed_gate_up_mq4 && (d.routed_down == MQ6G256);
        let routed_indexable_mq2lloyd = (d.routed_down == MQ2G256Lloyd) && routed_gate_up_mq2lloyd;
        // Both sides must be the UNROTATED dtype. A rotated/unrotated mix has
        // no coherent single rotation decision for the layer, so it must fall
        // out of the indexed path entirely rather than pick one and silently
        // corrupt the other projection.
        let routed_indexable_mq2lloyd_u =
            (d.routed_down == MQ2G256LloydU) && routed_gate_up_mq2lloyd_u;
        let routed_indexable_mq3lloyd = (d.routed_down == MQ3G256Lloyd) && routed_gate_up_mq3lloyd;
        // gate_up on one of the codebook (Lloyd / GL) formats — needed both for
        // the per-projection mix below and for `needs_x_rot_local` (all four are
        // FwhtG256 and consume the pre-rotated activation).
        let routed_gate_up_gl = matches!(d.routed_gate_up, MQ2G256GL | MQ3G256GL);
        // Per-projection codebook mix (e.g. gate_up MQ2-GL + down MQ3-GL, the
        // 2-bit-gate/3-bit-down allocation; or any Lloyd×GL cross). Subsumes the
        // two uniform Lloyd arms above and the uniform GL cases; the OR below
        // makes the overlap harmless.
        //
        // SAFETY INVARIANT: every dtype admitted here MUST have (a) an indexed
        // gate_up GEMV arm in `run_moe_decode`, (b) an ATOMIC SELF-COMBINING
        // down GEMV arm there, and (c) membership in the
        // `routed_down_self_combines` set in pipeline/mod.rs. Admitting a dtype
        // that misses (c) double-counts every MoE layer, silently.
        const CODEBOOK_INDEXABLE: [DType; 4] = [MQ2G256Lloyd, MQ3G256Lloyd, MQ2G256GL, MQ3G256GL];
        let routed_indexable_mixed_lloyd = CODEBOOK_INDEXABLE.contains(&d.routed_gate_up)
            && CODEBOOK_INDEXABLE.contains(&d.routed_down);
        let routed_indexable_paro =
            (d.routed_down == ParoQ4G128 && d.has_paro_shared) && routed_gate_up_paro;
        // Per-expert mixed: the model already verified the experts carry
        // different down dtypes and built the tag table (single source of
        // truth). gate_up stays uniform MQ4, so it pairs with the MQ4 indexed
        // gate_up GEMV; the merged dtype-tag kernel serves the down step.
        let routed_indexable_mixed_per_expert = d.routed_has_mixed_experts;
        // mfp4/mfp3/mfp2-E8 grouped experts (RDNA3 wave32-WMMA): uniform E8-family
        // gate_up + down → the gemv_mfp4g32_e8_moe_{gate_up,down}_k8_indexed kernels
        // (for uniform E8 models). FWHT-rotated (FwhtG256), same as MQ4, so the
        // shared silu+mul+rotate plumbing applies. Graded mixed-E8 uses the tag-table
        // path (routed_indexable_mixed_per_expert) rather than this uniform arm.
        let routed_gate_up_e8 = matches!(d.routed_gate_up, MFP4G32E8 | MFP3G32E8 | MFP2G32E8);
        let routed_indexable_e8 = arch_has_e8_wmma
            && routed_gate_up_e8
            && matches!(d.routed_down, MFP4G32E8 | MFP3G32E8 | MFP2G32E8);

        let routed_indexable_qt53_f32 = d.qt53_routed_down_fallback();

        let routed_dtype_indexable = routed_indexable_mq4
            || routed_indexable_mq4v2
            || routed_indexable_mq5
            || routed_indexable_mq6
            || routed_indexable_mq6v2
            || routed_indexable_mixed_gu4_dn6
            || routed_indexable_mixed_per_expert
            || routed_indexable_mq2lloyd
            || routed_indexable_mq2lloyd_u
            || routed_indexable_mq3lloyd
            || routed_indexable_mixed_lloyd
            || routed_indexable_paro
            || routed_indexable_e8
            || routed_indexable_qt53_f32;

        let use_gpu_topk = moe_softmax_k8_router_top_k_supported(k) && routed_dtype_indexable;
        let qt53_fallback_refused = routed_indexable_qt53_f32 && !use_gpu_topk;
        let needs_x_rot_local = gate_side_mq4
            || gate_fusable_mq4v2
            || routed_indexable_mixed_per_expert
            || routed_gate_up_mq4
            || routed_gate_up_mq4v2
            || routed_gate_up_mq4c
            || routed_gate_up_mq5
            || routed_gate_up_mq6
            || routed_gate_up_mq6v2
            || routed_gate_up_mq2lloyd
            || routed_gate_up_mq3lloyd
            // MQ2/MQ3-G256-GL are FWHT-G256 formats: their gate_up kernel reads
            // `x_rot`, so the local rotation MUST be produced. Missing this is a
            // silent garbage-output failure (unrotated x into a rotated weight).
            || routed_gate_up_gl
            || routed_gate_up_paro
            || routed_indexable_e8;
        // NOTE: `routed_gate_up_mq2lloyd_u` is DELIBERATELY ABSENT from the
        // chain above. MQ2G256LloydU is the unrotated sibling: its weights are
        // encoded in the natural basis, so producing x_rot and handing it to
        // the kernel would be the exact "unrotated x into a rotated weight"
        // failure the comment above warns about, only mirrored — and equally
        // silent. It is also deliberately NOT in `CODEBOOK_INDEXABLE`, because
        // membership there would let a rotated/unrotated cross-pair resolve via
        // `routed_indexable_mixed_lloyd` with no coherent rotation decision.
        // See docs/design/2026-08-22-maple-preview-20b-a1b.md.

        // A per-expert tier table is "mixed" only when it is Some AND spans more
        // than one distinct DType. A Some table that is all-equal collapses to
        // the uniform fast path (mixed = false), so existing arches — which pass
        // None for both tables — are always uniform and byte-identical to today.
        let table_varies = |t: &Option<&[DType]>| {
            t.as_ref()
                .and_then(|v| v.split_first())
                .map(|(first, rest)| rest.iter().any(|dt| dt != first))
                .unwrap_or(false)
        };
        let mixed = table_varies(&d.per_expert_gate_up) || table_varies(&d.per_expert_down);

        Self {
            gate_side_mq4,
            gate_fusable,
            gate_fusable_mq4v2,
            routed_indexable_mq4,
            routed_indexable_mq4v2,
            routed_indexable_mq5,
            routed_indexable_mq6,
            routed_indexable_mq6v2,
            routed_indexable_mixed_gu4_dn6,
            routed_indexable_mq2lloyd,
            routed_indexable_mq2lloyd_u,
            routed_indexable_mq3lloyd,
            routed_indexable_mixed_lloyd,
            routed_indexable_mixed_per_expert,
            routed_indexable_paro,
            use_gpu_topk,
            needs_x_rot_local,
            mixed,
            fp8_b128_refused: false,
            routed_indexable_qt53_f32,
            qt53_fallback_refused,
        }
    }

    /// Resolution for a layer sealed to [`MoeRouteCapability::Fp8B128Grouped`]:
    /// every generic route flag disabled and no refusal. The typed FP8 route
    /// lowers its own stages; nothing here selects a generic kernel.
    pub fn sealed_fp8() -> Self {
        Self {
            fp8_b128_refused: false,
            ..Self::fp8_b128_refused()
        }
    }

    /// All-disabled resolution for layers carrying a qt=55 operand.
    fn fp8_b128_refused() -> Self {
        Self {
            gate_side_mq4: false,
            gate_fusable: false,
            gate_fusable_mq4v2: false,
            routed_indexable_mq4: false,
            routed_indexable_mq4v2: false,
            routed_indexable_mq5: false,
            routed_indexable_mq6: false,
            routed_indexable_mq6v2: false,
            routed_indexable_mixed_gu4_dn6: false,
            routed_indexable_mq2lloyd: false,
            routed_indexable_mq2lloyd_u: false,
            routed_indexable_mq3lloyd: false,
            routed_indexable_mixed_lloyd: false,
            routed_indexable_mixed_per_expert: false,
            routed_indexable_paro: false,
            use_gpu_topk: false,
            needs_x_rot_local: false,
            mixed: false,
            fp8_b128_refused: true,
            routed_indexable_qt53_f32: false,
            qt53_fallback_refused: false,
        }
    }

    /// Typed execution guard for the generic decode route. `route` is the
    /// layer's selected specialized route: only
    /// [`MoeRouteCapability::Fp8B128Grouped`] may carry qt=55 operands.
    pub fn check_generic_route(
        &self,
        route: Option<MoeRouteCapability>,
    ) -> Result<(), DispatchError> {
        if self.fp8_b128_refused && !route.is_some_and(MoeRouteCapability::is_fp8_grouped) {
            return Err(fp8_b128_generic_refusal());
        }
        // qwen35 qt53 routed-down typed fallback: the specialized Flash-Next route
        // never consults this state.
        if self.qt53_fallback_refused && !route.is_some_and(MoeRouteCapability::is_sealed_grouped) {
            return Err(qt53_fallback_unsupported(
                "qt53-routed-down-fallback-needs-gpu-topk-k8",
            ));
        }
        Ok(())
    }

    pub fn routed_indexable(&self) -> bool {
        self.routed_indexable_mq4
            || self.routed_indexable_mq4v2
            || self.routed_indexable_mq5
            || self.routed_indexable_mq6
            || self.routed_indexable_mq6v2
            || self.routed_indexable_mixed_gu4_dn6
            || self.routed_indexable_mixed_per_expert
            || self.routed_indexable_mq2lloyd
            || self.routed_indexable_mq3lloyd
            || self.routed_indexable_mixed_lloyd
            || self.routed_indexable_paro
            || self.routed_indexable_qt53_f32
    }
}

/// Sealed decode mode for one indexed-MoE call. This is the model-declared
/// intent; the sealer (`pipeline::sealed_moe`) cross-checks it against the
/// bound expert table, the execution contract, and the process environment
/// before any launch, and invalid combinations fail there — never silently.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MoeEpMode {
    /// Single-GPU (or Single-owner) decode, including the existing
    /// next-layer deferred-combine experiment. Requires `routed_out=None`.
    None,
    /// Root-routed expert-parallel decode: per-rank zeroed partials
    /// (`routed_out=Some`, `defer_routed_combine=false`).
    /// The root routes on-GPU (SoftmaxTopK), seals the route-producer
    /// proof, and folds owned experts plus the shared expert once into
    /// its partial (`skip_shared=false`); non-root ranks skip the shared
    /// expert (`skip_shared=true`), adopt the broadcast root route, and
    /// fold owned experts (zero dummies read 0) into their partials.
    /// The driver then all-reduce-sums the partials and adds the result
    /// into each rank's residual once. Requires a plan-bound compact
    /// table carrying the root-routed execution contract.
    RootRoutedPartial,
}

/// True when the routed-down GEMV for `routed_down` materializes per-slot
/// rows into `down_expanded` (so a later fold can combine them).
///
/// The whole codebook family (`MQ2/MQ3-Lloyd` and their global-codebook
/// siblings `MQ2/MQ3-GL`) self-combines via atomic adds straight into the
/// residual target and writes NO expanded buffer — folding `down_expanded`
/// afterwards would double-count (or fold stale rows). Per-expert mixed
/// mode writes the expanded buffer for every tier, so it always returns
/// true here.
///
/// Dtype-only base of [`MoeParams::down_writes_expanded`]. Keep this dtype
/// list in lockstep with the decode down arms: miss a dtype here and a route
/// silently zeroes out or double-counts.
pub fn moe_down_writes_expanded(routed_down: DType, has_dtype_tags: bool) -> bool {
    has_dtype_tags
        || !matches!(
            routed_down,
            DType::MQ2G256Lloyd
                | DType::MQ2G256LloydU
                | DType::MQ3G256Lloyd
                | DType::MQ2G256GL
                | DType::MQ3G256GL
        )
}

impl MoeParams<'_> {
    /// Whether the selection bias carries precomputed (activated) router scores.
    pub fn router_scores_precomputed(&self) -> bool {
        self.selection_bias
            .is_some_and(|bias| bias.scores == MoeScores::Precomputed)
    }

    /// Whether the routed down writes the expanded buffer for a separate combine
    /// (`true`) or self-combines into the residual (`false`). A declared
    /// MQ2-Lloyd expert follows the model's down-combine policy.
    pub fn down_writes_expanded(&self) -> bool {
        match (&self.mq2_lloyd, self.dtypes.routed_down) {
            (Some(lloyd), DType::MQ2G256Lloyd) => lloyd.deterministic_down(),
            _ => {
                moe_down_writes_expanded(self.dtypes.routed_down, self.expert_dtype_tags.is_some())
            }
        }
    }
}

// ── Dispatch parameters ────────────────────────────────

/// Host-side routed weight views for generic CPU-top-K fallback and dispatch metadata.
///
/// Implementations return borrowed dispatch views on demand so model hot paths
/// do not rebuild an expert-reference vector for every token.
pub trait RoutedExpertWeights {
    fn len(&self) -> usize;

    fn get(&self, expert_idx: usize) -> Option<(WeightRef<'_>, WeightRef<'_>)>;

    /// Representative `(gate_up, down)` host-mapped residency, read from the
    /// owning allocations without querying HIP or traversing the expert table.
    ///
    /// The default supports owner-backed `get(0)` tensors and fails closed for
    /// empty sets and borrowed views. Packed adapters must override this using
    /// their allocation owners: a borrowed buffer is not residency metadata.
    fn host_mapped_projections(&self) -> (bool, bool) {
        if let Some(tier) = self.tier_identity() {
            return tier.host_projections();
        }
        self.get(0).map_or((false, false), |(gate_up, down)| {
            (
                gate_up.buf.buf.is_host_mapped(),
                down.buf.buf.is_host_mapped(),
            )
        })
    }

    /// CPU-splice payload for one selected expert: the packed byte spans plus
    /// their canonical CPU decoder. `None` (the default) means this adapter
    /// cannot resolve one, and the caller must supply its own resolver — the
    /// historical contract. Host-placed packed adapters override this using
    /// their allocation owners, exactly as they do for
    /// [`Self::host_mapped_projections`]: a borrowed per-expert view is not
    /// enough to find the owning bucket.
    fn cpu_splice_expert(
        &self,
        _gpu: &Gpu,
        _expert_idx: usize,
    ) -> Option<crate::cpu_exec::CpuMoeExpert<'_>> {
        None
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `Some(id)` promises that every value reporting `id` returns the same
    /// weights from `get` for as long as it lives (a set built once and never
    /// mutated, with `id` unique to that construction).  A live binding
    /// captured from such a set then re-checks the id per call instead of
    /// every expert.  `None` (the default) keeps the per-expert check.
    fn immutable_identity(&self) -> Option<u64> {
        None
    }

    /// Whether the routed experts live in host-mapped memory. Expert views are
    /// borrowed byte views, so the owner's placement must be carried here; a
    /// layer's experts share one residency. `false` by default; an explicit
    /// tier binding answers `any_host`.
    fn host_mapped(&self) -> bool {
        self.tier_identity()
            .is_some_and(ExpertTierIdentity::any_host)
    }

    /// Explicit per-expert tier placement of this layer (`None` for every
    /// legacy layer). When `Some`, residency is never inferred from expert 0,
    /// and the live binding retains and revalidates this identity.
    fn tier_identity(&self) -> Option<&ExpertTierIdentity> {
        None
    }

    /// Eager CPU-cold/GPU-hot split executor attached to this layer (the arch
    /// owns attachment, the immutable expert placement and the format/layout
    /// admission). `None` (the default) leaves every call in its ordinary
    /// mode. When `Some`, a call of at most
    /// [`crate::cpu_split::CPU_SPLIT_MAX_ROWS`] rows on the original grouped
    /// QT44/QT53 route is sealed in the distinct split mode: the routed GPU
    /// tail reads the executor's masked ids while the CPU computes the cold
    /// slots' unweighted outputs, and the canonical combine still reads the
    /// original route. Wider windows are ordinary wide prefill.
    fn cpu_split(&self) -> Option<&crate::cpu_split::CpuSplitExecutor> {
        None
    }
}

/// Expert-parallel row exchange for the indexed decode MoE and the few-row
/// grouped QT44/QT53 prefill.
///
/// Invoked once per MoE call after the routed down producer wrote `slots`
/// unweighted expanded rows (slot order, `slots = rows * k`) into
/// `down_expanded` and before the canonical combine reads them. The
/// implementation must leave every slot whose expert this rank owns untouched
/// and overwrite every other slot with the owning rank's bits (a copy, never
/// arithmetic), so the unchanged combine folds exactly the rows a single
/// device would have produced. Decode passes `slots = k`.
pub trait MoeRowGather {
    fn gather_rows(
        &self,
        gpu: &mut rdna_compute::Gpu,
        layer_idx: u16,
        slots: usize,
        down_expanded: &GpuTensor,
        topk_indices: &GpuTensor,
    ) -> Result<(), String>;

    /// Whether this call's layer uses device-assigned (balanced) ownership:
    /// the lowering then inserts `MoeStage::OwnerAssign` after routing and
    /// before any stage that reads the routed expert tables. `false` by
    /// default (static ownership, no stage).
    fn assigns_owners(&self, _layer_idx: u16) -> bool {
        false
    }

    /// Rewrite the layer's live expert tables and slot owners from
    /// `topk_indices` (`slots = rows * k`). Runs after routing, before the
    /// first routed-table reader of the same call.
    fn assign_owners(
        &self,
        _gpu: &mut rdna_compute::Gpu,
        _layer_idx: u16,
        _slots: usize,
        _topk_indices: &GpuTensor,
    ) -> Result<(), String> {
        Err("owner assignment not supported".into())
    }
}

/// Expert-parallel owner-routed chunk exchange of the grouped symmetric IU4
/// prefill (`HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`).
///
/// Every rank runs the same router, top-k and stable grouping, so the grouped
/// row layout (`expert_tile_ids`, `sorted_slot_index`, `inverse_perm`) is
/// identical on all ranks. Each rank then computes only the tiles of the
/// experts it owns: the lowering inserts `MoeStage::TileMask` after the
/// scatter (a copy of the tile ids with every non-owned tile set to the `-1`
/// sentinel the symmetric GEMMs already exit on, written to
/// [`MoeTileExchangeParams::tile_ids_owned`]; gate/up and down read that copy)
/// and `MoeStage::TileExchange` between the mutation fence and the combine,
/// where [`Self::exchange`] fills every non-owned live tile's BF16 grouped
/// rows of `y_down_grouped` from the owning rank (a copy, never arithmetic),
/// so the unchanged BF16-row combine reads exactly the rows a single device
/// produced.
///
/// Protocol (eager only; all ranks execute the same invocation sequence):
/// `exchange` records this rank's producer-complete event, passes a host
/// barrier, waits the peer's event, pulls, records its completion event and
/// passes a second host barrier; [`Self::before_down`] enqueues, right before
/// the NEXT invocation's routed down overwrites `y_down_grouped`, the wait for
/// the peer's completion of the previous pull (the peer reads this rank's
/// buffer only between its producer and completion events).
pub trait MoeTileExchange {
    /// This rank, the device table of the owner rank of every expert (`i32`
    /// per expert, at least `n_experts` words) and `n_experts`; the mask and
    /// the pull read them as inputs, never as constants.
    fn owners(&self) -> (usize, &GpuTensor, usize);

    /// Enqueue, on the launch stream, the wait that the peer finished pulling
    /// this rank's rows of the previous invocation. Called by the routed down
    /// stage immediately before it writes `y_down_grouped`; a no-op before the
    /// first invocation.
    fn before_down(&self, gpu: &mut rdna_compute::Gpu) -> Result<(), String>;

    /// Cross-rank host barrier (every rank calls it at the same point); used to
    /// order a rank-local drain (`ep_quiesce`) before any rank launches a chunk
    /// that reads peer VRAM.
    fn host_barrier(&self) -> Result<(), String>;

    /// Publish this rank's `y_down_grouped` and pull the peer-owned tiles into
    /// it. `tile_ids` is the UNMASKED scatter output (the pull's predicate is
    /// `tile_ids[t] >= 0 && owner(tile_ids[t]) != rank`), `tiles` the padded
    /// tile capacity (`grouped_rows / 16`).
    fn exchange(
        &self,
        gpu: &mut rdna_compute::Gpu,
        layer_idx: u16,
        tile_ids: &GpuTensor,
        y_down_grouped: &GpuTensor,
        tiles: usize,
    ) -> Result<(), String>;
}

/// The owner-routed chunk exchange of one MoE call: the transport and the
/// layer's masked tile id buffer (an `i32` per tile, at least the padded tile
/// capacity; never aliasing `expert_tile_ids`).
#[derive(Clone, Copy)]
pub struct MoeTileExchangeParams<'a> {
    pub exchange: &'a dyn MoeTileExchange,
    pub tile_ids_owned: &'a GpuTensor,
}

/// Everything the MoE decode executor arm reads, marshaled by the model from
/// its weight/config/scratch structs. Resolution is owned by the family
/// (the model passes only the dtype snapshot + k); the executor computes
/// [`MoeResolution`] from [`MoeDtypes`] on entry.
#[derive(Clone, Copy)]
pub struct MoeParams<'a> {
    pub dtypes: MoeDtypes<'a>,
    pub recipe: MoeRecipe,
    /// Optional specialized route declaration supplied by the architecture.
    pub route_policy: Option<MoeRoutePolicy>,
    pub normalization: MoeNormalization<'a>,
    /// Token-batch width. Decode = 1. >1 must route to grouped prefill (Step 8).
    pub batch_size: usize,
    // dims / config scalars
    pub hidden: usize,
    pub mi: usize,
    pub k: usize,
    pub n_exp: usize,
    pub norm_topk_prob: bool,
    /// `Some` for a biased sigmoid router (only with
    /// [`MoeRecipe::SigmoidRoutedNoShared`]); `None` keeps the unbiased top-8.
    pub selection_bias: Option<MoeSelectionBias<'a>>,
    pub x_rot_prerotated: bool,
    /// Single-GPU lowered-decode experiment: leave the atomic-free routed
    /// output expanded so the architecture layer can combine it into the
    /// residual while producing the next layer's normalized activation.
    pub defer_routed_combine: bool,
    /// CPU-deferred routed experts (`memory.offload_exec=cpu` on a host-placed
    /// packed layer). The sealed program keeps the GPU router, rotation, shared
    /// expert, and residual, but must NOT run the routed
    /// gate_up/activation/down/combine stages: a host-side batched splice
    /// (`cpu_exec::moe_cpu_experts`) supplies the routing-weighted routed
    /// contribution exactly once afterwards, accumulating into `x_residual`.
    /// Independent of `use_gpu_topk` (the router still runs on the GPU) and
    /// never a fallback to the per-expert host GEMV path.
    pub cpu_deferred_experts: bool,
    /// Sealed decode mode (see [`MoeEpMode`]).
    pub ep_mode: MoeEpMode,
    /// Safetensors layer index used by native GPTQ-on-E8 Hessian capture.
    pub layer_idx: u16,
    // activations / residual
    pub x_norm: &'a GpuTensor,
    pub x_residual: &'a GpuTensor,
    pub routed_out: Option<&'a GpuTensor>,
    /// EP: skip the shared-expert down projection on non-root ranks.
    pub skip_shared: bool,
    // gate-side weights
    pub router: WeightRef<'a>,
    pub shared: Option<MoeSharedDecode<'a>>,
    // routed expert pointer tables + dims
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    /// Host entries behind `expert_gate_up_ptrs` / `expert_down_ptrs`.
    ///
    /// Supplying them lets dispatch prove the table *contents* point at the live
    /// expert tensors instead of trusting the upload. `None` keeps the historical
    /// contract (contents unchecked); a retained body requires `Some`, because a
    /// replay dereferences those entries without re-uploading them.
    pub expert_ptrs_host: Option<MoePointerEntries<'a>>,
    /// Route A MoE-AWQ: per-routed-expert down `awq_scale` pointer table
    /// (`[2·n_exp]` f32 = n_exp `u64` ptrs → each expert's `[routed_down_k]`
    /// f32 scale). `Some` only when the `.hfq` carries per-expert
    /// `down_proj.awq_scale` sidecars; the executor then runs the AWQ-aware
    /// indexed silu+rotate (`x/s` before the FWHT). `None` (default) = the
    /// plain silu+rotate, byte-identical to pre-AWQ.
    pub expert_down_awq_ptrs: Option<&'a GpuTensor>,
    /// Per-expert mixed-precision decode: `[n_exp]` u8 (DType::Raw, 1 B/exp)
    /// dtype-tag table, `Some` iff any expert's gate_up or down dtype differs
    /// from experts[0] (N-tier graded files). The merged dtype-tag-branched
    /// gate_up AND down kernels read `dtype_tags[expert_id]` per block:
    ///   0=MQ6G256 (200 B/grp), 1=MQ2G256Lloyd (72 B/grp),
    ///   2=MQ4G256 (136 B/grp), 3=MQ3G256Lloyd (112 B/grp).
    /// `None` (default) ⇒ uniform path, byte-identical to pre-mixed.
    pub expert_dtype_tags: Option<&'a GpuTensor>,
    pub routed_gate_up_k: usize,
    pub routed_down_m: usize,
    pub routed_down_k: usize,
    /// Allocation-free access to resident per-expert weights for the generic
    /// CPU-top-K fallback (`!use_gpu_topk`). Empty under paged residency, where
    /// the indexed GPU-top-K path is the only supported mode.
    pub routed_experts: &'a dyn RoutedExpertWeights,
    // paro sidecars
    pub routed_gate_up_paro: Option<GivensRef<'a>>,
    pub routed_down_paro: Option<GivensRef<'a>>,
    // scratch buffers
    pub router_logits: &'a GpuTensor,
    pub x_rot_local: &'a GpuTensor,
    /// Fused [gate||up] scratch for a single routed expert in the host path.
    pub gate_up_buf: &'a GpuTensor,
    pub ffn_hidden: &'a GpuTensor,
    pub ffn_out: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub down_expanded: &'a GpuTensor,
    /// Expert-parallel decode: copy peer-owned expanded rows into
    /// `down_expanded` between the routed down producer and the combine.
    /// `None` (every single-device caller) lowers exactly the historical
    /// stage sequence.
    pub ep_row_gather: Option<&'a dyn MoeRowGather>,
    /// DeepSeek-V4 MQ2-Lloyd routed experts (k in 1..=8). `None` lowers the
    /// generic sealed sequence unchanged.
    pub mq2_lloyd: Option<MoeMq2Lloyd<'a>>,
}

/// Model-owned MQ2-Lloyd routed-expert declaration (DeepSeek-V4): sealed decode
/// runs these experts with the model's SwiGLU clamp and down-combine policy.
#[derive(Clone, Copy)]
pub struct MoeMq2Lloyd<'a> {
    /// SwiGLU clamp passed to `deepseek4_silu_mul_clamp_f32_batched`.
    pub swiglu_limit: f32,
    /// Atomic self-combining down (the gfx1151 policy). When `false` and
    /// `HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC!=0`, the expanded write plus
    /// `moe_down_combine_k8_batched` is used.
    pub uses_atomic_down: bool,
    /// Exact-device native backend, when the loader admitted one.
    pub native: Option<&'a dyn MoeBiasAwareMq2Backend>,
    /// Shared zero gate_up buffer for expert-parallel non-owned slots.
    pub nonowned_gate_up_dummy: Option<&'a GpuTensor>,
}

impl MoeMq2Lloyd<'_> {
    /// Expanded-down + combine policy (the default), read per call.
    pub fn deterministic_down(&self) -> bool {
        !self.uses_atomic_down
            && hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC").as_deref()
                != Ok("0")
    }
}

// ── MQ2-Lloyd routed-expert backend ───────────

/// Exact-device MQ2-Lloyd operations used by sealed decode for DeepSeek-V4
/// routed experts (`MoeMq2Lloyd::native`).
///
/// The dispatch crate deliberately has no architecture detection here. A
/// model crate may provide this capability only after its loader has admitted
/// a model-owned backend. The implementation must still fail closed when the
/// supplied [`Gpu`] is not the device proven by that backend.
pub trait MoeBiasAwareMq2Backend {
    #[allow(clippy::too_many_arguments)]
    fn gate_up(
        &self,
        gpu: &mut Gpu,
        expert_ptrs: &GpuTensor,
        nonowned_gate_up_dummy: Option<&GpuTensor>,
        topk_indices: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> Result<(), String>;

    fn rotate_x_batched(
        &self,
        gpu: &mut Gpu,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> Result<(), String>;

    #[allow(clippy::too_many_arguments)]
    fn down_expanded(
        &self,
        gpu: &mut Gpu,
        expert_ptrs: &GpuTensor,
        ownership_ptrs: &GpuTensor,
        nonowned_gate_up_dummy: Option<&GpuTensor>,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> Result<(), String>;

    #[allow(clippy::too_many_arguments)]
    fn down_residual_scaled(
        &self,
        gpu: &mut Gpu,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,
        residual: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> Result<(), String>;
}

/// Selected routed-expert decode subgraph. Route selection is intentionally
/// absent: callers must provide the exact normalized IDs and weights produced
/// by the model-owned router. This is useful for split ownership where the
/// router and expert weights cannot reside on the same device.
pub struct MoeSelectedParams<'a> {
    pub hidden: usize,
    pub mi: usize,
    pub k_top: usize,
    pub swiglu_limit: f32,
    pub uses_atomic_moe_down: bool,
    pub native_mq2_backend: Option<&'a dyn MoeBiasAwareMq2Backend>,
    pub nonowned_gate_up_dummy: Option<&'a GpuTensor>,
    pub batch_size: usize,
    pub x_rot: &'a GpuTensor,
    pub ffn_out: &'a GpuTensor,
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    pub down_expanded: &'a GpuTensor,
}

// ── DeepSeek-V4 batched/prefill MoE parameters ─────────

/// Router-selection mode for the batched/prefill MoE path. DeepSeek-V4 uses
/// static hash routing for the first `num_hash_layers` layers and bias-aware
/// top-k for the rest; the executor branches on this.
pub enum MoePrefillRouting<'a> {
    /// Bias-aware batched top-k (select on `scores + gate_bias`, weight on the
    /// unbiased `scores`, normalize, `*route_scale`).
    BiasAware { gate_bias: &'a GpuTensor },
    /// Static `tid2eid` hash routing (layers `0..num_hash_layers`). `tokens` is
    /// the device-side `[B]` i32 token-id buffer.
    Hash {
        tid2eid: &'a GpuTensor,
        tokens: &'a GpuTensor,
    },
}

/// Parameters for the deepseek4 batched/prefill MoE (k=6, MQ2-Lloyd). The
/// model owns RMSNorm, the shared expert, the router GEMV + `sqrt_softplus`
/// (producing `scores`); this arm runs routing → routed experts → combine,
/// accumulating into `ffn_out` (the shared expert already seeded it).
///
/// Picks the grouped-GEMM path when `batch_size >= HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE`
/// (default 128), else the scalar K4 indexed path — mirroring `ffn_batched`.
pub struct MoeBiasAwarePrefillParams<'a> {
    // dims / config scalars
    pub hidden: usize,
    pub mi: usize,
    pub n_exp: usize,
    pub k_top: usize,
    pub batch_size: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    /// Model-local dispatch policy. The DS4 loader derives this from the
    /// verified MQ2R backend; generic GPU state must not influence it.
    pub uses_atomic_moe_down: bool,
    pub layer_idx: usize, // for the optional HIPFIRE_DEEPSEEK4_DUMP_TOPK header
    // routing
    pub routing: MoePrefillRouting<'a>,
    pub scores: &'a GpuTensor, // post-sqrt_softplus moe_scores_batch [B, n_exp]
    pub topk_indices: &'a GpuTensor, // [B, k_top] (routing out, expert in)
    pub topk_weights: &'a GpuTensor, // [B, k_top]
    // routed expert pointer tables
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    // activation / residual
    pub x_rot: &'a GpuTensor,   // ffn_x_rot_batch [B, hidden]
    pub ffn_out: &'a GpuTensor, // ffn_out_batch [B, hidden] (accumulate target)
    // grouped-path scratch
    pub expert_token_counts: &'a GpuTensor,
    pub expert_offsets: &'a GpuTensor,
    pub sorted_slot_index: &'a GpuTensor,
    pub expert_tile_ids: &'a GpuTensor,
    pub inverse_perm: &'a GpuTensor,
    pub y_gate_up_grouped: &'a GpuTensor,
    pub y_down_grouped: &'a GpuTensor,
    // shared scratch (grouped + scalar)
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    // scalar-path scratch (expanded deterministic down)
    pub down_expert_outputs: &'a GpuTensor,
}

// ── Qwen3.5 softmax-top-k MoE prefill parameters (Ship 4.2) ──

/// Static VRAM pointer tables for a full-layer prefill DMA stage.
/// Live expert bindings still name the original mapped owners. The down
/// launcher releases this parity buffer before combine/shared work begins.
#[derive(Clone, Copy)]
pub struct MoeStageTables<'a> {
    pub gate_up: &'a GpuTensor,
    pub down: &'a GpuTensor,
    pub free: &'a hip_bridge::Event,
}

/// Parameters for the qwen35 batched/prefill MoE routed-expert block.
///
/// Distinct from [`MoeBiasAwarePrefillParams`] — qwen35 uses softmax top-k
/// routing (k=8) with MQ4/MQ6/Paro routed experts, a fused gate-side, and a
/// shared expert that seeds `x_batch` before this arm runs.
///
/// The model owns RMSNorm, the router GEMV + softmax top-k (producing
/// `topk_indices` / `topk_weights`), and the shared expert (which already
/// accumulated into `x_batch`). This arm runs scatter → gate_up → unscatter →
/// SwiGLU+rotate → down → combine, accumulating into `x_batch`.
///
/// All tensor refs are `&'a GpuTensor` (shared, not `&mut` — GpuTensor is Copy).
/// Scratch tensors are model-owned; the family holds only references.
#[derive(Clone, Copy)]
pub struct MoePrefillParams<'a> {
    // dtype snapshot
    pub dtypes: MoeDtypes<'a>,
    pub recipe: MoeRecipe,
    /// Optional specialized route declaration supplied by the architecture.
    pub route_policy: Option<MoeRoutePolicy>,
    /// Complete normalization/router/shared prelude. The routed tail below
    /// remains unchanged and consumes the declared route buffers.
    pub prelude: MoePrefillPrelude<'a>,
    // dims
    pub batch_size: usize,
    pub mi: usize,
    pub down_m: usize,
    pub down_k: usize,
    pub gate_up_k: usize,
    pub k_top: usize,
    pub n_exp: usize,
    /// m_total upper bound pre-computed by the model via
    /// `moe_grouped_m_total_bound(total_slots, n_exp)`. Used by Path 2
    /// scatter + grouped GEMM for grid sizing.
    pub m_total_max: usize,
    /// Model-level safety fence for promoted/mixed MQ6 checkpoints. When true,
    /// MQ4 grouped prefill calls use FP16 WMMA even for layers whose local
    /// routed dtype snapshot is pure MQ4. This keeps pure MQ4 models on the
    /// existing i8 default while avoiding mixed-checkpoint corruption.
    pub force_mq4_grouped_fp16: bool,
    /// CPU-deferred routed experts (see [`MoeParams::cpu_deferred_experts`]).
    /// Set only for narrow (`<= 4`-row) windows on a CPU-spliced host-placed
    /// layer: the sealed program keeps normalize/input-basis/router/shared, but
    /// must NOT run the routed scatter/gate_up/activation/down/combine stages.
    /// The arch's batched CPU splice supplies the routed contribution once
    /// afterwards, accumulating into `x_batch`. Wide prefill keeps the real GPU
    /// grouped read and stays `false`.
    pub cpu_deferred_experts: bool,
    // routing inputs (model-produced)
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    // destination = x_batch (residual; combine accumulates here)
    pub x_batch: &'a GpuTensor,
    // activation buffers
    pub x_norm_batch: &'a GpuTensor,
    pub x_rot_batch: &'a GpuTensor,
    // routed gate_up/down pointer tables
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    /// Prefill-only table override; decode and live-binding validation ignore it.
    pub expert_stage_ptrs: Option<MoeStageTables<'a>>,
    /// Exact live expert owners bound to the sealed load-time identity.
    pub routed_experts: &'a dyn RoutedExpertWeights,
    /// Route A MoE-AWQ: per-routed-expert down `awq_scale` pointer table (see
    /// [`MoeParams::expert_down_awq_ptrs`]). When `Some`, the prefill silu+rotate
    /// uses the indexed AWQ kernel (per-slot scale via `topk_indices`),
    /// superseding the single-scale `down_awq_scale` stub below for routed
    /// experts. `None` (default) = plain silu+rotate.
    pub expert_down_awq_ptrs: Option<&'a GpuTensor>,
    /// Per-expert mixed-precision prefill: `[n_exp]` u8 dtype-tag table,
    /// `Some` iff the routed experts carry mixed dtypes (graded T3-3L). Drives
    /// the merged grouped-WMMA prefill kernel. `None` ⇒ uniform path, byte-identical.
    pub expert_dtype_tags: Option<&'a GpuTensor>,
    // intermediate buffers
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    // Path 1 expanded-down scratch
    pub down_expanded: &'a GpuTensor,
    // Path 2 scatter scratch (model-owned)
    pub expert_token_counts: &'a GpuTensor,
    pub expert_offsets: &'a GpuTensor,
    pub sorted_slot_index: &'a GpuTensor,
    pub expert_tile_ids: &'a GpuTensor,
    pub inverse_perm: &'a GpuTensor,
    pub y_gate_up_grouped: &'a GpuTensor,
    pub y_down_grouped: &'a GpuTensor,
    // paro sidecars (per-layer shared Givens rotation tables)
    pub paro_gate_up: Option<GivensRef<'a>>,
    pub paro_down: Option<GivensRef<'a>>,
    /// AWQ scale for the routed down weight (experts[0].down.awq_scale).
    /// Used by the AWQ-aware silu+rotate step. `None` when the routed
    /// experts are non-AWQ (the common case for A3B).
    pub down_awq_scale: Option<&'a GpuTensor>,
    /// EP (Ship 6 substrate-EP prefill): when `Some`, the **routed** combine
    /// accumulates into this **zeroed** `[batch × dim]` partial instead of
    /// `x_batch`; the EP prefill driver then all-reduce-sums the partial across
    /// ranks and adds it into each rank's `x_batch`. The **shared** expert stays
    /// in `x_batch` (replicated per rank — added once to each rank's own copy,
    /// no all-reduce). `None` (the default) accumulates routed into `x_batch`,
    /// byte-identical to pre-EP behavior.
    pub routed_out: Option<&'a GpuTensor>,
    /// Few-row EP row gather (qwen4 EP speculation): when `Some`, the lowering
    /// admits only the grouped QT44/QT53 branch and inserts a `RowGather` stage
    /// between the mutation fence and the combine. `None` (the default) changes
    /// nothing.
    pub ep_row_gather: Option<&'a dyn MoeRowGather>,
    /// Where the unfused shared-expert down writes its F32 rows. `None` ⇒
    /// `down_expanded` (the shared down runs after the combine and would
    /// otherwise overwrite rows a peer may still be gathering). Required with
    /// `ep_row_gather`.
    pub shared_down_out: Option<&'a GpuTensor>,
    /// Layer index handed to `ep_row_gather` (it selects the gather buffer);
    /// read only when `ep_row_gather` is `Some`, `0` otherwise.
    pub ep_layer_idx: u16,

    /// Owner-routed EP chunk exchange (`HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`): when
    /// `Some`, the lowering admits only the grouped symmetric IU4 prefill (the
    /// verified-symmetric QT44/QT53 route at >= 512 rows), inserts a
    /// `TileMask` stage after the scatter and a `TileExchange` stage between the
    /// mutation fence and the combine, and the grouped gate/up and down read
    /// the masked tile ids. `None` (the default) changes nothing.
    pub ep_tile_exchange: Option<MoeTileExchangeParams<'a>>,
    /// Default-off expert IU4 prefill capability (`HIPFIRE_MOE_SYM_IU4`), bound
    /// only by an architecture whose loader verified the routed tensors: the
    /// loader's [`ExpertIu4Spec`](crate::families::moe_expert_iu4::ExpertIu4Spec)
    /// and a `bool` that is `true` only when the loader verified symmetric
    /// headers on every routed gate/up and down tensor of ALL model layers.
    /// `None` (every non-Qwen35 constructor, and Qwen35 with the flag off) means
    /// the route does not exist for this call and changes nothing.
    pub expert_iu4: Option<(crate::families::moe_expert_iu4::ExpertIu4Spec, bool)>,
}
/// Dtype-only admission snapshot for grouped MoE prefill.
///
/// This record intentionally has no architecture dependency. Model loaders
/// assemble it from their owned weights and pass it to the shared policy.
#[derive(Debug, Clone, Copy)]
pub struct MoePrefillDtypes {
    pub router: DType,
    pub shared_expert_scalar_gate: DType,
    pub shared_expert_gate: DType,
    pub shared_expert_up: DType,
    pub shared_expert_down: DType,
    pub expert_gate_up: DType,
    pub expert_down: DType,
    pub expert_gate_up_uniform: bool,
    pub expert_down_uniform: bool,
    pub routed_mixed_merged: bool,
}

impl MoePrefillDtypes {
    /// Construct a uniform prefill dtype snapshot.
    pub fn uniform(dtype: DType) -> Self {
        Self {
            router: dtype,
            shared_expert_scalar_gate: dtype,
            shared_expert_gate: dtype,
            shared_expert_up: dtype,
            shared_expert_down: dtype,
            expert_gate_up: dtype,
            expert_down: dtype,
            expert_gate_up_uniform: true,
            expert_down_uniform: true,
            routed_mixed_merged: false,
        }
    }
}

#[inline]
pub fn paro_batched_admit_enabled_from_env(value: Option<&str>) -> bool {
    value == Some("1")
}

#[inline]
pub fn moe_prefill_topk_shape_supported(k_top: usize, num_experts: usize) -> bool {
    moe_softmax_k8_router_top_k_supported(k_top) && num_experts <= FFN_ROUTER_MAX_EXPERTS
}

/// Quantized routed-expert formats with an implemented Cohere sigmoid
/// grouped-prefill path.
#[inline]
pub fn sigmoid_routed_prefill_supported(gate_up: DType) -> bool {
    matches!(
        gate_up,
        DType::MQ4G256 | DType::HFQ4G256 | DType::MQ6G256 | DType::HFQ6G256
    )
}

#[inline]
pub fn routed_codebook_grouped_supported(dt: DType) -> bool {
    matches!(
        dt,
        DType::MQ4G256 | DType::MQ2G256Lloyd | DType::MQ3G256Lloyd
    )
}

#[inline]
pub fn routed_uniform_mqv2_grouped_supported(dt: DType) -> bool {
    matches!(dt, DType::MQ4G256V2 | DType::MQ6G256V2)
}

#[inline]
pub fn routed_codebook_pair_batched_supported(gate_up: DType, down: DType) -> bool {
    routed_codebook_grouped_supported(gate_up)
        && routed_codebook_grouped_supported(down)
        && (matches!(gate_up, DType::MQ2G256Lloyd | DType::MQ3G256Lloyd)
            || matches!(down, DType::MQ2G256Lloyd | DType::MQ3G256Lloyd))
}

pub fn codebook_batched_admit_enabled_from_env(
    value: Option<&str>,
    grouped_gemm_value: Option<&str>,
    _arch: &str,
) -> bool {
    let grouped_gemm_on = !matches!(grouped_gemm_value, Some("0") | Some("off"));
    match value {
        Some("0") | Some("off") | Some("false") => false,
        Some("1") | Some("on") | Some("true") => grouped_gemm_on,
        _ => false,
    }
}

pub fn codebook_batched_admit_enabled(arch: &str) -> bool {
    codebook_batched_admit_enabled_from_env(
        hipfire_config::developer_var("HIPFIRE_MOE_CODEBOOK_BATCHED")
            .ok()
            .as_deref(),
        hipfire_config::developer_var("HIPFIRE_MOE_GROUPED_GEMM")
            .ok()
            .as_deref(),
        arch,
    )
}

/// Whether MQ6 grouped prefill is admitted by the process policy.
pub fn mq6_batched_admit_enabled_from_env(value: Option<&str>, arch: &str) -> bool {
    match value {
        Some("0") | Some("off") | Some("false") => false,
        Some("1") | Some("on") | Some("true") => true,
        _ => arch.starts_with("gfx11") || arch.starts_with("gfx12"),
    }
}

pub fn gated_moe_prefill_admissible_for_dtypes(
    dtypes: &MoePrefillDtypes,
    admit_mq6: bool,
    admit_paro: bool,
    admit_e8: bool,
    admit_codebook: bool,
) -> bool {
    let router_ok = matches!(
        dtypes.router,
        DType::MQ4G256 | DType::MQ4G256V2 | DType::Q8_0 | DType::F32
    );
    let shared_gate_ok = matches!(
        dtypes.shared_expert_scalar_gate,
        DType::MQ4G256 | DType::MQ4G256V2 | DType::Q8_0 | DType::F32
    );
    let routed_ok =
        dtypes.routed_mixed_merged || (dtypes.expert_gate_up_uniform && dtypes.expert_down_uniform);
    if !(router_ok && shared_gate_ok && routed_ok) {
        return false;
    }

    let is_e8_family = |dt| matches!(dt, DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8);
    if dtypes.routed_mixed_merged {
        let shared_gu_ok = (dtypes.shared_expert_gate == dtypes.shared_expert_up
            && matches!(dtypes.shared_expert_gate, DType::MQ4G256 | DType::MQ4G256V2))
            || (admit_mq6
                && matches!(
                    dtypes.shared_expert_gate,
                    DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2
                )
                && dtypes.shared_expert_up == dtypes.shared_expert_gate);
        let shared_dn_ok = matches!(dtypes.shared_expert_down, DType::MQ4G256 | DType::MQ4G256V2)
            || (admit_mq6
                && matches!(dtypes.shared_expert_down, DType::MQ6G256 | DType::MQ6G256V2));
        return shared_gu_ok && shared_dn_ok;
    }

    if admit_e8
        && dtypes.shared_expert_gate == DType::Q8_0
        && dtypes.shared_expert_up == DType::Q8_0
        && dtypes.shared_expert_down == DType::Q8_0
        && is_e8_family(dtypes.expert_gate_up)
        && is_e8_family(dtypes.expert_down)
    {
        return true;
    }

    if admit_e8
        && is_e8_family(dtypes.expert_gate_up)
        && is_e8_family(dtypes.expert_down)
        && dtypes.shared_expert_gate == dtypes.shared_expert_up
        && matches!(
            dtypes.shared_expert_gate,
            DType::Q8_0 | DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8
        )
        && matches!(
            dtypes.shared_expert_down,
            DType::Q8_0 | DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8
        )
    {
        return true;
    }

    if admit_codebook
        && routed_codebook_pair_batched_supported(dtypes.expert_gate_up, dtypes.expert_down)
    {
        let shared_gu_ok = (dtypes.shared_expert_gate == dtypes.shared_expert_up
            && matches!(dtypes.shared_expert_gate, DType::MQ4G256 | DType::MQ4G256V2))
            || (admit_mq6
                && matches!(
                    dtypes.shared_expert_gate,
                    DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2
                )
                && dtypes.shared_expert_up == dtypes.shared_expert_gate);
        let shared_dn_ok = matches!(dtypes.shared_expert_down, DType::MQ4G256 | DType::MQ4G256V2)
            || (admit_mq6
                && matches!(dtypes.shared_expert_down, DType::MQ6G256 | DType::MQ6G256V2));
        if shared_gu_ok && shared_dn_ok {
            return true;
        }
    }

    if admit_paro
        && dtypes.shared_expert_gate == DType::ParoQ4G128
        && dtypes.shared_expert_up == DType::ParoQ4G128
        && dtypes.shared_expert_down == DType::ParoQ4G128
        && dtypes.expert_gate_up == DType::ParoQ4G128
        && dtypes.expert_down == DType::ParoQ4G128
    {
        return true;
    }

    // qwen35 qt53 routed-down typed fallback: uniform qt44 gate/up + uniform qt53
    // down. The shared expert stays an MQ4/MQ6 (G256) dense projection pair.
    if dtypes.expert_gate_up == DType::MQ4G256V2 && dtypes.expert_down == DType::MQ4G128V2 {
        let shared_gu_dt = dtypes.shared_expert_gate;
        let shared_ok = |dtype: DType| {
            matches!(dtype, DType::MQ4G256 | DType::MQ4G256V2)
                || (admit_mq6 && matches!(dtype, DType::MQ6G256 | DType::MQ6G256V2))
        };
        return shared_ok(shared_gu_dt)
            && dtypes.shared_expert_up == shared_gu_dt
            && shared_ok(dtypes.shared_expert_down);
    }

    if admit_mq6 {
        let shared_gu_dt = dtypes.shared_expert_gate;
        let shared_gu_ok = matches!(
            shared_gu_dt,
            DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2
        ) && dtypes.shared_expert_up == shared_gu_dt;
        let shared_dn_ok = matches!(
            dtypes.shared_expert_down,
            DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2
        );
        let experts_ok = matches!(
            dtypes.expert_gate_up,
            DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2
        ) && matches!(
            dtypes.expert_down,
            DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2
        );
        shared_gu_ok && shared_dn_ok && experts_ok
    } else {
        dtypes.shared_expert_gate == dtypes.shared_expert_up
            && matches!(dtypes.shared_expert_gate, DType::MQ4G256 | DType::MQ4G256V2)
            && matches!(dtypes.shared_expert_down, DType::MQ4G256 | DType::MQ4G256V2)
            && matches!(dtypes.expert_gate_up, DType::MQ4G256 | DType::MQ4G256V2)
            && matches!(dtypes.expert_down, DType::MQ4G256 | DType::MQ4G256V2)
    }
}

/// Shared admission policy for Qwen-style softmax gated/shared prefill.
pub fn gated_moe_prefill_admissible(
    dtypes: &MoePrefillDtypes,
    admit_mq6: bool,
    arch: &str,
) -> bool {
    let admit_e8 = matches!(
        arch,
        "gfx1100"
            | "gfx1101"
            | "gfx1102"
            | "gfx1150"
            | "gfx1151"
            | "gfx1152"
            | "gfx1200"
            | "gfx1201"
    ) && hipfire_config::developer_var("HIPFIRE_E8_GFX12")
        .ok()
        .as_deref()
        == Some("1");
    let admit_paro = paro_batched_admit_enabled_from_env(
        hipfire_config::developer_var("HIPFIRE_PARO_BATCHED")
            .ok()
            .as_deref(),
    );
    let admit_codebook = codebook_batched_admit_enabled(arch);
    gated_moe_prefill_admissible_for_dtypes(dtypes, admit_mq6, admit_paro, admit_e8, admit_codebook)
}

/// Upper extent (rows) of the shared narrow-window width: the MTP verify
/// window (drafts + 1, capped at 4). Single declaration for every spelling of
/// this width — the CPU-splice gate (`n <= MAX`) and the verify/coalesced
/// gates (`2..=MAX`) below all read this. The verify lower bound of 2 is
/// load-bearing (a 1-row window is decode, not verify) and lives at each
/// verify use site, not here.
pub const NARROW_VERIFY_MAX_ROWS: usize = 4;

/// Actual storage bound (rows) of a CPU split executor: the single
/// [`crate::cpu_split::CPU_SPLIT_STORAGE_MAX_ROWS`] declaration (64) owned by
/// `cpu_split.rs`. Storage only, NOT a qualification or a default: the legacy
/// default route still admits at most [`NARROW_VERIFY_MAX_ROWS`] rows, and a
/// 5..64-row call additionally needs an explicit lab-wide qualification (the
/// routing/admission owners' `verify_rows > 8` scope). Never define a second
/// spelling of either width here.
pub use crate::cpu_split::CPU_SPLIT_STORAGE_MAX_ROWS;

/// Per-call lab-wide verify scope the routing slice measured for this call.
///
/// G2/G4 contract: BOTH explicit lab arms — `wide8` and `wide64` — execute the
/// FULL logical R1..64 CPU expert backend after actual 640-slot storage. The
/// `wide8` arm only closes the GPU few-row projection routes above 8 (the
/// incumbent prefill-kernel baseline); it MUST NOT leave a CPU backend hard
/// cap of 8/4 or force a whole-forward chunk.
///
/// Gate (presence, not value): the CPU experimental capacity is gated on
/// EXPLICIT existing-oracle-override presence plus the eager-verify scope and
/// actual allocated storage — never on `Gpu::few_row_limit` as a CPU physical
/// capacity (that is a GPU projection-route limit, not CPU storage). Concretely
/// `lab_armed` is true only when the caller observed the existing oracle
/// override present (the `HIPFIRE_QWEN4_WIDE_VERIFY_ROWS` developer var or the
/// runtime override) on an exact oracle arch alongside an eager (non-prefill,
/// non-captured) verify scope. Default (no lab override present) remains
/// qualified CPU 4; ordinary prefill and lanes are unchanged (they never arm).
/// Carried per call (never a global), so an ordinary prefill in the same
/// process keeps the default 4-row route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuWideVerifyScope {
    /// Explicit lab override observed present for this call.
    pub lab_armed: bool,
    /// Frozen row-exact verify rows measured for this call (the existing
    /// override value clamped `8..=64`, or the per-arch table default).
    /// Informational only: NEVER a CPU capacity (see `lab_armed`).
    pub verify_rows: usize,
}

impl CpuWideVerifyScope {
    /// Default product scope: no lab override observed.
    pub const fn default_scope() -> Self {
        Self {
            lab_armed: false,
            verify_rows: rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS,
        }
    }

    /// Scope from an observed override presence plus the measured
    /// `verify_rows` value (already clamped `8..=64` by the frozen getter).
    /// `lab_armed` MUST be the existing-override presence (env or runtime) on
    /// an exact oracle arch with an eager verify scope — never derived from
    /// `few_row_limit`, which is a GPU route limit, not CPU storage.
    pub const fn from_lab_presence(lab_armed: bool, verify_rows: usize) -> Self {
        Self {
            lab_armed,
            verify_rows,
        }
    }

    /// Scope from a measured `verify_rows` value alone (no presence signal).
    /// Presence-unknown callers MUST use this: a value above 8 does NOT arm
    /// the lab path by itself (a wide GPU arm is not CPU storage proof).
    /// Kept for back-compat; prefer [`Self::from_lab_presence`].
    pub const fn from_verify_rows(verify_rows: usize) -> Self {
        Self {
            lab_armed: false,
            verify_rows,
        }
    }

    /// Whether this call runs inside an explicit lab-wide eager-verify scope:
    /// the existing oracle override was observed present. Ordinary prefill,
    /// lanes and the default product route read `false`.
    pub const fn is_lab_wide(&self) -> bool {
        self.lab_armed
    }

    /// Qualified CPU rows for this call: 4 on the default route; inside an
    /// explicit lab-wide scope, `min(capacity, STORAGE)` — the ACTUAL
    /// allocated capacity, never the `verify_rows` clamp. `verify_rows` is the
    /// informational GPU arm width only; the wide8 baseline
    /// (`from_lab_presence(true, 8)`) with 64-row storage qualifies the FULL
    /// R1..64 CPU backend. The executor gate independently checks allocated
    /// capacity. Default (unarmed) stays qualified 4.
    pub const fn qualified_max_rows(&self, capacity_rows: usize) -> usize {
        if !self.lab_armed {
            return NARROW_VERIFY_MAX_ROWS;
        }
        if capacity_rows > CPU_SPLIT_STORAGE_MAX_ROWS {
            CPU_SPLIT_STORAGE_MAX_ROWS
        } else {
            capacity_rows
        }
    }
}

/// Prewarmed D1 host-slot scratch rows visible to the routing slice (the
/// admission owner's `reserve_moe_cpu_host_slot_scratch` reservation, in rows;
/// `0` until the admission owner prewarms 64-row storage). Room for the
/// admission owner to back this with its real reservation getter without a
/// routing-slice edit: routing calls ONLY this function for the D1 capacity
/// source, never a constant. D1 has no split executor, so the executor
/// capacity API does not apply here.
/// D1 capacity source: the binding minimum of actual prewarmed host-slot
/// scratch rows and actual pinned-transfer stage rows for this call's
/// geometry, on the calling thread. `dim/mi/top_k` are the call's expert
/// geometry; `device_id` is the sealing context's device. Returns `0`
/// (legacy `<= 4` windows only, fail-closed) when nothing is prewarmed, when
/// geometry is zero/overflows, or when the stage belongs to another device or
/// is pending. Routing calls ONLY this function for the D1 capacity source,
/// never a constant. The admission owner prewarms both (`admission` +
/// `reserve_cpu_transfer_values`); this function only reads actuals.
///
/// Stage formula authority (`cpu_exec.rs`): `stage_bytes = (n_x + slots) * 8`
/// with `n_x = R*dim` (activation + residual) and `slots = R*top_k` (indices +
/// weights), so `stage_rows = bytes / (8 * (dim + top_k))`. A `4*(dim+2*top_k)`
/// inverse would undercharge the stage and overstate capacity — never use it.
pub fn cpu_d1_prewarmed_scratch_rows(dim: usize, mi: usize, top_k: usize, device_id: i32) -> usize {
    let host_rows = crate::cpu_exec::moe_cpu_host_slot_scratch_capacity_rows(dim, mi, top_k);
    if host_rows == 0 {
        return 0;
    }
    let per_row = (dim as u128)
        .checked_add(top_k as u128)
        .unwrap_or(u128::MAX);
    if per_row == 0 {
        return 0;
    }
    let stage_bytes = crate::cpu_exec::cpu_transfer_capacity_bytes(device_id) as u128;
    let denom = per_row.checked_mul(8).unwrap_or(0);
    if denom == 0 {
        return 0;
    }
    let stage_rows = (stage_bytes / denom).min(usize::MAX as u128) as usize;
    host_rows.min(stage_rows)
}

/// Whether a D1 (`cpu_deferred_experts`) window of `rows` real rows over
/// `route_rows` routing rows takes the CPU splice under this call's lab-wide
/// scope with prewarmed host scratch: decode and a whole verify window of at
/// most `scope.qualified_max_rows(scratch_rows)` rows, and only when the real
/// rows are the whole routing window. Default (no lab override present, or no
/// prewarm yet) keeps the legacy 4-row gate; an explicit lab-wide eager-verify
/// scope with actual 64-row prewarmed scratch admits the whole logical 1..64
/// window. Ordinary prefill and lanes never arm (their `armed` is false or
/// their rows are not the whole routing window). No chunking, no new math.
pub const fn cpu_d1_wide_window_qualified(
    armed: bool,
    rows: usize,
    route_rows: usize,
    scope: CpuWideVerifyScope,
    scratch_rows: usize,
) -> bool {
    // The legacy `<= 4`-row windows never need prewarmed scratch (an armed
    // scope with nothing prewarmed is fail-closed to them, not to zero rows).
    let scoped = scope.qualified_max_rows(scratch_rows);
    let qualified = if scoped < NARROW_VERIFY_MAX_ROWS {
        NARROW_VERIFY_MAX_ROWS
    } else {
        scoped
    };
    crate::cpu_split::cpu_splice_window_qualified(armed, rows, route_rows, qualified)
}

/// Whether a call of `rows` real rows over `route_rows` routing rows takes the
/// CPU routed-expert window on a layer that is `armed`, under this call's
/// lab-wide scope and the executor's allocated capacity: decode and a verify
/// window of at most `scope.qualified_max_rows(capacity_rows)` rows, and only
/// when the real rows are the whole routing window. A call over a wider
/// routing window (a short tail of a long prefill whose `route_rows` is the
/// canonical chunk) takes the GPU route. No chunking: the whole logical
/// 1..64-row window seals one forward; GDN/QSA/PLE/capture keep full-window
/// state. `capacity_rows` is the executor's allocated
/// [`crate::cpu_split::CpuSplitExecutor::capacity_rows`]; callers MUST pass
/// the executor's own value, never a constant.
pub const fn cpu_wide_window_qualified(
    armed: bool,
    rows: usize,
    route_rows: usize,
    scope: CpuWideVerifyScope,
    capacity_rows: usize,
) -> bool {
    let qualified = scope.qualified_max_rows(capacity_rows);
    crate::cpu_split::cpu_splice_window_qualified(armed, rows, route_rows, qualified)
}

/// The layer's CPU split executor when a `rows`-row call takes the split mode
/// on the default route (`1..=NARROW_VERIFY_MAX_ROWS` rows and the owner
/// attached an executor with allocated capacity for them). Storage-only gate:
/// wider logical windows return `None` here and qualify only through
/// [`cpu_wide_window_qualified`] with an explicit lab-wide scope plus the
/// executor's allocated [`crate::cpu_split::CpuSplitExecutor::capacity_rows`].
/// The one predicate both sealing and execution read on the default path;
/// unqualified wider windows stay ordinary wide prefill.
pub fn cpu_split_executor(
    experts: &dyn RoutedExpertWeights,
    rows: usize,
) -> Option<&crate::cpu_split::CpuSplitExecutor> {
    if rows == 0 || rows > crate::cpu_split::CPU_SPLIT_MAX_ROWS {
        return None;
    }
    experts
        .cpu_split()
        .filter(|executor| rows <= executor.capacity_rows())
}

/// The layer's CPU split executor for a call inside an explicit lab-wide
/// verify scope: `1..=STORAGE`-row logical windows whose executor has
/// allocated capacity for them. The routing slice calls this ONLY after
/// [`cpu_wide_window_qualified`] admitted the window (armed layer, whole
/// routing window, qualified width, allocated capacity); the executor then
/// independently refuses any `rows` past its allocation. Unqualified windows
/// (ordinary prefill, lanes, default route) never reach this and keep the
/// [`cpu_split_executor`] default-4 gate. Full-window state/capture/ticket
/// stay intact: one forward, never chunked.
pub fn cpu_split_executor_wide(
    experts: &dyn RoutedExpertWeights,
    rows: usize,
) -> Option<&crate::cpu_split::CpuSplitExecutor> {
    if rows == 0 || rows > CPU_SPLIT_STORAGE_MAX_ROWS {
        return None;
    }
    experts
        .cpu_split()
        .filter(|executor| rows <= executor.capacity_rows())
}

impl<'a> MoeParams<'a> {
    /// A by-value view of this call whose routed ids are `ids`; every other
    /// operand, including the weights the canonical combine reads, is shared.
    /// Only the GPU routed gate/up, activation and down readers of the CPU
    /// split take it; the original route is never rewritten.
    pub fn with_topk_indices<'b>(&self, ids: &'b GpuTensor) -> MoeParams<'b>
    where
        'a: 'b,
    {
        let mut view: MoeParams<'b> = *self;
        view.topk_indices = ids;
        view
    }
}

impl<'a> MoePrefillParams<'a> {
    /// Prefill counterpart of [`MoeParams::with_topk_indices`] (also read by
    /// the grouped scatter).
    pub fn with_topk_indices<'b>(&self, ids: &'b GpuTensor) -> MoePrefillParams<'b>
    where
        'a: 'b,
    {
        let mut view: MoePrefillParams<'b> = *self;
        view.topk_indices = ids;
        view
    }
}

/// Resolved dispatch plan for the qwen35 batched MoE prefill routed block.
///
/// Distinct from [`MoeResolution`] (decode) — prefill adds the Path 0/1/2
/// grouped-vs-scalar down selection and the Paro i8/k8 levers.
/// Pure function of [`MoeDtypes`] + arch + [`FeatureFlags`].

pub struct MoePrefillResolution {
    /// Gate_up + down via grouped-GEMM scatter pipeline (Path 2).
    /// Requires WMMA-capable arch (gfx11/gfx12) + `moe_grouped_gemm` flag.
    pub use_path2: bool,
    /// Down uses atomic-accumulate GEMV (Path 0) instead of atomic-free
    /// expanded+combine (Path 1). gfx9* wave64 archs (gfx906/gfx908/gfx94x).
    pub down_path0: bool,
    /// gfx1151 Paro i8 MMQ grouped GEMM (Path 2 only).
    pub use_paro_i8: bool,
    /// gfx1151 Paro i8 MMQ k8 grouped GEMM (Path 2 only).
    pub use_paro_i8_k8: bool,
    /// Routed experts use ParoQ4G128 (determines SwiGLU+rotate kernel selection).
    pub paro_mode: bool,
    /// gfx1151's HFQ4 grouped-i8 path is correct for pure MQ4, but corrupts
    /// MQ6-promoted A3B MTP prefill when the same MoE layer mixes MQ4 and MQ6
    /// projections. Default mixed layers back to FP16 WMMA; explicit
    /// HIPFIRE_MOE_GROUPED_I8=1 still opts into the research path.
    pub force_mq4_grouped_fp16: bool,
    /// Refusal state: a qt=55 operand is present, so every path flag above is
    /// disabled. Resolution stays infallible; [`Self::check_generic_route`]
    /// raises the typed refusal at execution.
    pub fp8_b128_refused: bool,
    /// Refusal state: the qwen35 qt53 routed-down fallback shape is present on
    /// a Path 0 (gfx9 atomic-GEMV) device, which has no qt53 down; the
    /// typed refusal is raised before any launch. Never set for FP8 or for a
    /// layer on the specialized Flash-Next route (exempt in the check).
    pub qt53_refused: bool,
}

impl MoePrefillResolution {
    /// Resolve the prefill dispatch plan from dtypes, arch, and flags.
    ///
    /// Reads MoE prefill env levers from `flags` (parsed once at `Gpu::init`),
    /// not `std::env` — mid-prefill env mutation is not honored.
    pub fn resolve(
        d: &MoeDtypes<'_>,
        arch: &rdna_compute::arch_caps::ArchCaps,
        flags: &rdna_compute::feature_flags::FeatureFlags,
    ) -> Self {
        Self::resolve_with_batch_ctx(d, arch, flags, 0, false, false)
    }

    /// Batch- and workload-aware resolution. A narrow *verify* batch (MTP
    /// verify, n <= 4) whose routed projections are host-mapped takes the
    /// indexed Path 1. Prompt chunks keep the legacy route regardless of width,
    /// and so does a verify batch over device-resident experts: the Path 1
    /// evidence is host-spilled verify only, and Path 1 accumulates in a
    /// different order than the resident grouped route. `batch == 0` (unknown)
    /// preserves the legacy shape-blind behavior.
    ///
    /// `has_host_projection` is authoritative: the caller MUST set it only when
    /// the routed projections are actually host-mapped (see
    /// [`RoutedExpertWeights::host_mapped_projections`]). One exception inside
    /// the host-mapped case: a host-mapped uniform MQ4 gate_up+down on gfx1201
    /// with no mixed-expert layer is re-admitted to Path 2 (grouped-WMMA).
    /// `resolve` (batch0 legacy) always passes `false`.
    pub fn resolve_with_batch_ctx(
        d: &MoeDtypes<'_>,
        arch: &rdna_compute::arch_caps::ArchCaps,
        flags: &rdna_compute::feature_flags::FeatureFlags,
        batch: usize,
        is_verify: bool,
        has_host_projection: bool,
    ) -> Self {
        if d.has_fp8_b128_projection() {
            return Self {
                use_path2: false,
                down_path0: false,
                use_paro_i8: false,
                use_paro_i8_k8: false,
                paro_mode: false,
                force_mq4_grouped_fp16: false,
                fp8_b128_refused: true,
                qt53_refused: false,
            };
        }
        let paro_mode = d.routed_gate_up == DType::ParoQ4G128 && d.has_paro_shared;
        // Resident routed experts never take the narrow-verify route: only a
        // host-mapped verify window moves to Path 1.
        let narrow_verify =
            is_verify && has_host_projection && batch != 0 && batch <= NARROW_VERIFY_MAX_ROWS;
        // Host-mapped uniform MQ4 gate_up+down on a narrow gfx1201 verify:
        // re-admit Path 2 (grouped-WMMA) for this one offloaded case. The later
        // mixed / MQ6 / E8 logic below is unchanged and still governs every
        // other path. Lower bound of 2 is load-bearing: a 1-row window is
        // decode, not a verify window, so it must not take the coalesced/Path-2
        // re-admit.
        let offloaded_uniform_verify = narrow_verify
            && (2..=NARROW_VERIFY_MAX_ROWS).contains(&batch)
            && arch.is_gfx1201()
            && has_host_projection
            && !d.routed_has_mixed_experts
            && matches!(d.routed_gate_up, DType::MQ4G256 | DType::MQ4G256V2)
            && matches!(d.routed_down, DType::MQ4G256 | DType::MQ4G256V2);
        let use_path2 = flags.moe_grouped_gemm
            && arch.has_wmma()
            && (!narrow_verify || offloaded_uniform_verify);
        // MQ6 / MQ6V2 grouped-WMMA: gfx11 `_k2` kernel now exists (alongside the
        // gfx12 `_gfx12` / `mq6g256v2` sisters). Only suppress Path 2 on archs
        // that have NEITHER (gfx9*, gfx1010/1030, CDNA) — i.e. no wmma_w32 and
        // not gfx12. gfx1100/1101/1102/1103/1150/1151/1152 all have wmma_w32.
        // (Master's narrower gfx1151-only MQ6 admit (dfed8cc6) is subsumed by
        // this wider gfx11 widen (8d555fc6); master's mixed-checkpoint safety
        // is preserved separately via `force_mq4_grouped_fp16` below.)
        // qt47 (MQ6G256V2) shares the same gfx11/gfx12 grouped availability —
        // never collapse it onto the V1 MQ6G256 path (dual-half vs f32 header).
        let mq6_on_non_wmma = matches!(d.routed_gate_up, DType::MQ6G256 | DType::MQ6G256V2)
            && !arch.has_wmma_w32()
            && !(arch.is_gfx1200() || arch.is_gfx1201());
        let use_path2 = use_path2 && !mq6_on_non_wmma;
        // MQ5 grouped-WMMA (`gemm_hfq5g256_moe_grouped_wmma`) is gfx12-only
        // (same as MQ6) — fall back to Path 1 (indexed batched GEMV) on
        // gfx11/gfx9 to avoid the gfx12-only kernel panic.
        let mq5_on_non_gfx12 =
            d.routed_gate_up == DType::MQ5G256 && !(arch.is_gfx1200() || arch.is_gfx1201());
        let use_path2 = use_path2 && !mq5_on_non_gfx12;
        // Mixed per-expert: the merged grouped kernel covers all four dtype
        // tags on any WMMA arch (gfx11 _k2 or gfx12 .gfx12). The routed
        // representative dtype may be MQ6/MQ5 and trip the suppression above,
        // so re-admit Path 2 when the file is graded-mixed (tag table present).
        //
        // Load-bearing for CORRECTNESS, not just perf. Measured 2026-10-04 on
        // qwen3.6-35b-a3b.mq4p: forcing a graded narrow verify onto
        // both-indexed projections (mixed indexed gate_up AND down together)
        // drives MTP tau 2.34 -> 0.00 and degenerates the plain AR prompt
        // prefill (`hipfire run --spec off`) to a repeated-token attractor;
        // either indexed kernel alone is fine, so this is an all-indexed
        // interaction, not a single-kernel defect. The narrow-verify gate must
        // NOT override this re-admit.
        let use_path2 =
            use_path2 || (d.routed_has_mixed_experts && flags.moe_grouped_gemm && arch.has_wmma());
        // mfp4-E8 routed experts: use Path 2 (grouped-WMMA) on gfx1151 and gfx12
        // (RDNA4). Both have a native E8 grouped-WMMA GEMM kernel:
        //   gfx1151 → gemm_mfp4g32_e8_moe_grouped_wmma (gfx1151.hip)
        //   gfx12   → gemm_mfp4g32_e8_moe_grouped_wmma_gfx12 (gfx12.hip)
        // Other archs (gfx1100 dGPU, gfx9*/CDNA) have no grouped E8 sister → Path 1.
        // mfp4-E8 grouped-WMMA prefill on ALL WMMA arches (RDNA3 gfx11 + RDNA4
        // gfx12). The gfx1151 kernel uses the RDNA3 wave32-WMMA builtin and runs
        // correctly on gfx1100/1101/1102; gfx12 uses its .gfx12 sister. The prior
        // "gfx1151-only / gfx1100 wash" call rested on pp512 97.5-vs-97.6 — which is
        // DECODE tok/s, not prefill throughput (a prefill change can't move decode
        // tok/s). Real prefill throughput is what bench_sweep measures, so route
        // gfx1100 through Path 2 and re-measure. Only ever active under the
        // HIPFIRE_E8_GFX12 batched-prefill gate.
        let e8_no_grouped = matches!(
            d.routed_gate_up,
            DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8
        ) && !(arch.is_rdna3() || arch.is_rdna4());
        let use_path2 = use_path2 && !e8_no_grouped;
        // Path 0: gfx9* wave64 archs (gfx906/gfx908/gfx94x) — cheap HBM
        // atomics make the atomic GEMV pattern competitive vs expanded scratch.
        let down_path0 = arch.is_gcn5() || arch.is_cdna1() || arch.is_cdna3();
        let is_gfx1151 = arch.is_gfx1151();
        let use_paro_i8 = paro_mode && use_path2 && is_gfx1151 && flags.moe_paro_i8.unwrap_or(true);
        let use_paro_i8_k8 = use_paro_i8 && flags.moe_paro_i8_k8.unwrap_or(true);
        let force_mq4_grouped_fp16 =
            use_path2 && is_gfx1151 && d.has_mq6_projection() && flags.moe_grouped_i8.is_none();
        Self {
            use_path2,
            down_path0,
            use_paro_i8,
            use_paro_i8_k8,
            paro_mode,
            force_mq4_grouped_fp16,
            fp8_b128_refused: false,
            qt53_refused: d.qt53_routed_down_fallback() && down_path0 && !use_path2,
        }
    }

    /// Resolution for a layer sealed to [`MoeRouteCapability::Fp8B128Grouped`]:
    /// every generic path flag disabled (no Path 0/1/2) and no refusal. The
    /// typed FP8 route is a separate arm in the sealed selection.
    pub fn sealed_fp8() -> Self {
        Self {
            use_path2: false,
            down_path0: false,
            use_paro_i8: false,
            use_paro_i8_k8: false,
            paro_mode: false,
            force_mq4_grouped_fp16: false,
            fp8_b128_refused: false,
            qt53_refused: false,
        }
    }

    /// Typed execution guard for the generic prefill route (Paths 0/1/2).
    /// `route` is the layer's selected specialized route: only
    /// [`MoeRouteCapability::Fp8B128Grouped`] may carry qt=55 operands.
    pub fn check_generic_route(
        &self,
        route: Option<MoeRouteCapability>,
    ) -> Result<(), DispatchError> {
        if self.fp8_b128_refused && !route.is_some_and(MoeRouteCapability::is_fp8_grouped) {
            return Err(fp8_b128_generic_refusal());
        }
        if self.qt53_refused && !route.is_some_and(MoeRouteCapability::is_sealed_grouped) {
            return Err(qt53_fallback_unsupported(
                "qt53-routed-down-fallback-path0-unsupported",
            ));
        }
        Ok(())
    }
}

use crate::tables::moe_table;

// ── Family ─────────────────────────────────────────────

pub struct MoeFamily {
    registry: KernelRegistry,
}

impl MoeFamily {
    pub fn new() -> Self {
        let mut registry = KernelRegistry::new();
        moe_table::populate(&mut registry);
        registry
            .validate()
            .expect("moe kernel table has empty entries");
        Self { registry }
    }

    pub fn registry(&self) -> &KernelRegistry {
        &self.registry
    }

    /// Resolve the best kernel key for the given MoE variant.
    ///
    /// Applies arch gating through `KernelRegistry::resolve`.
    pub fn resolve(
        &self,
        variant: MoeVariant,
        ctx: &DispatchCtx,
        shape: Option<&ShapeInfo>,
    ) -> Result<&KernelVariant, DispatchError> {
        let key = match variant {
            MoeVariant::IndexedGateUp => KernelKey::MoeIndexedGateUpLloyd,
            MoeVariant::IndexedDown => KernelKey::MoeIndexedDownLloyd,
            MoeVariant::GroupedGemm => KernelKey::MoeGroupedGemm,
        };
        self.registry.resolve(key, ctx, shape)
    }

    /// Run only the selected-expert portion of the single-token DeepSeek4
    /// MQ2-Lloyd subgraph. The caller owns route selection and must already
    /// have populated `topk_indices` and `topk_weights`.
    pub fn run_selected(
        &self,
        gpu: &mut rdna_compute::Gpu,
        params: &MoeSelectedParams,
    ) -> Result<(), DispatchError> {
        crate::pipeline::run_moe_decode_selected(gpu, params)
    }

    /// Run a batched/prefill deepseek4 MoE step (k=6, MQ2-Lloyd): routing
    /// (bias-aware or hash) → routed experts (grouped GEMM when
    /// `batch_size >= gate`, else scalar K4 indexed) → combine, accumulating
    /// into `params.ffn_out`. Delegates to
    /// [`crate::pipeline::run_moe_prefill_bias_aware`]. The model owns RMSNorm,
    /// the shared expert, and the router GEMV + `sqrt_softplus`.
    pub fn run_bias_aware_prefill(
        &self,
        gpu: &mut rdna_compute::Gpu,
        params: &MoeBiasAwarePrefillParams,
    ) -> Result<(), DispatchError> {
        crate::pipeline::run_moe_prefill_bias_aware(gpu, params)
    }
}

impl KernelFamily for MoeFamily {
    fn name(&self) -> &'static str {
        "moe"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_mq4() -> MoeDtypes<'static> {
        MoeDtypes {
            router: DType::MQ4G256,
            shared: Some(MoeSharedDtypes {
                selector: DType::MQ4G256,
                gate: DType::MQ4G256,
                up: DType::MQ4G256,
                down: DType::MQ4G256,
            }),
            experts_all_gate_up_mq4: true,
            routed_gate_up: DType::MQ4G256,
            routed_down: DType::MQ4G256,
            routed_has_mixed_experts: false,
            has_paro_shared: false,
            per_expert_gate_up: None,
            per_expert_down: None,
        }
    }

    #[test]
    fn qwen_gate_prerotation_requires_exact_v1_or_ornith_v2_quartet() {
        assert!(softmax_shared_gate_uniform_mq4(
            DType::MQ4G256,
            DType::MQ4G256,
            DType::MQ4G256,
            DType::MQ4G256,
        ));
        assert!(!softmax_shared_gate_uniform_mq4(
            DType::MQ4G256V2,
            DType::MQ4G256V2,
            DType::MQ4G256V2,
            DType::MQ4G256V2,
        ));
        let exact = [
            (DType::MQ4G256V2, 256, 2_048, false),
            (DType::MQ4G256V2, 1, 2_048, false),
            (DType::MQ4G256V2, 512, 2_048, false),
            (DType::MQ4G256V2, 512, 2_048, false),
        ];
        assert!(softmax_shared_gate_mq4v2_layouts(exact));
        for slot in 0..4 {
            let mut mixed = exact;
            mixed[slot].0 = DType::MQ4G256;
            assert!(!softmax_shared_gate_mq4v2_layouts(mixed));
            let mut awq = exact;
            awq[slot].3 = true;
            assert!(!softmax_shared_gate_mq4v2_layouts(awq));
        }
        let mut wrong_shape = exact;
        wrong_shape[2].1 = 511;
        assert!(!softmax_shared_gate_mq4v2_layouts(wrong_shape));
    }

    #[test]
    fn resolve_none_per_expert_is_not_mixed() {
        let d = uniform_mq4();
        let r = MoeResolution::resolve(&d, 8);
        assert!(!r.mixed);
    }

    #[test]
    fn resolve_some_per_expert_with_varied_tiers_is_mixed() {
        let mut d = uniform_mq4();
        d.per_expert_gate_up = Some(&[DType::MQ4G256, DType::MQ6G256]); // varies
        d.per_expert_down = Some(&[DType::MQ4G256, DType::MQ6G256]);
        let r = MoeResolution::resolve(&d, 8);
        assert!(r.mixed);
    }

    #[test]
    fn resolve_empty_per_expert_table_is_not_mixed_and_does_not_panic() {
        // A degenerate empty table must not index v[0]; it collapses to uniform.
        let mut d = uniform_mq4();
        d.per_expert_gate_up = Some(&[]);
        d.per_expert_down = Some(&[]);
        let r = MoeResolution::resolve(&d, 8);
        assert!(!r.mixed);
    }

    #[test]
    fn resolve_some_per_expert_all_same_is_not_mixed() {
        // a per-expert table that is uniform should NOT trigger the mixed path
        let mut d = uniform_mq4();
        d.per_expert_gate_up = Some(&[DType::MQ4G256, DType::MQ4G256]);
        d.per_expert_down = Some(&[DType::MQ4G256, DType::MQ4G256]);
        let r = MoeResolution::resolve(&d, 8);
        assert!(
            !r.mixed,
            "a uniform per-expert table must take the fast uniform path"
        );
    }

    #[test]
    fn resolve_mq6v2_uniform_is_indexable() {
        // qt47 uniform: both projections MQ6G256V2 => indexable, GPU top-K on.
        let mut d = uniform_mq4();
        d.routed_gate_up = DType::MQ6G256V2;
        d.routed_down = DType::MQ6G256V2;
        d.experts_all_gate_up_mq4 = false;
        let r = MoeResolution::resolve(&d, 8);
        assert!(r.routed_indexable_mq6v2);
        assert!(!r.routed_indexable_mq6, "must not claim the V1 MQ6 arm");
        assert!(!r.routed_indexable_mq4);
        assert!(!r.routed_indexable_mq4v2);
        assert!(r.use_gpu_topk);
        assert!(r.needs_x_rot_local, "qt47 kernels read ROTATED activations");
        assert!(r.routed_indexable());
    }

    #[test]
    fn resolve_mq6v2_mixed_with_v1_is_not_indexable() {
        // Same dual-half hazard as mq4v2/qt13: V1 and V2 share the 200 B group
        // stride and 6-bit packing, differing ONLY in the 8-byte header. A
        // split pairing must NOT be indexable on either arm.
        for (gu, dn) in [
            (DType::MQ6G256V2, DType::MQ6G256),
            (DType::MQ6G256, DType::MQ6G256V2),
            (DType::MQ6G256V2, DType::MQ4G256),
            (DType::MQ4G256V2, DType::MQ6G256V2),
        ] {
            let mut d = uniform_mq4();
            d.routed_gate_up = gu;
            d.routed_down = dn;
            d.experts_all_gate_up_mq4 = false;
            let r = MoeResolution::resolve(&d, 8);
            assert!(!r.routed_indexable_mq6v2, "{gu:?}/{dn:?}");
            assert!(!r.routed_indexable_mq6, "{gu:?}/{dn:?}");
            assert!(
                !r.use_gpu_topk,
                "{gu:?}/{dn:?} must fall back, not guess a layout"
            );
        }
    }

    #[test]
    fn resolve_mq6_v1_still_indexable_without_mq6v2() {
        // Preserve V1: uniform MQ6G256 must keep the V1 arm and never claim V2.
        let mut d = uniform_mq4();
        d.routed_gate_up = DType::MQ6G256;
        d.routed_down = DType::MQ6G256;
        d.experts_all_gate_up_mq4 = false;
        let r = MoeResolution::resolve(&d, 8);
        assert!(r.routed_indexable_mq6);
        assert!(!r.routed_indexable_mq6v2);
        assert!(r.use_gpu_topk);
        assert!(r.needs_x_rot_local);
    }

    #[test]
    fn has_mq6_projection_recognizes_v1_and_v2() {
        let mut d = uniform_mq4();
        assert!(!d.has_mq6_projection());

        d.routed_down = DType::MQ6G256;
        assert!(
            d.has_mq6_projection(),
            "V1 MQ6 must trip has_mq6_projection"
        );

        d.routed_down = DType::MQ6G256V2;
        assert!(
            d.has_mq6_projection(),
            "V2 MQ6 must trip has_mq6_projection"
        );

        d.routed_down = DType::MQ4G256;
        d.shared = Some(MoeSharedDtypes {
            selector: DType::MQ4G256,
            gate: DType::MQ6G256V2,
            up: DType::MQ4G256,
            down: DType::MQ4G256,
        });
        assert!(
            d.has_mq6_projection(),
            "shared-expert MQ6V2 must trip has_mq6_projection"
        );

        d.shared = Some(MoeSharedDtypes {
            selector: DType::MQ4G256,
            gate: DType::MQ4G256V2,
            up: DType::MQ4G256,
            down: DType::MQ4G256,
        });
        assert!(
            !d.has_mq6_projection(),
            "MQ4V2 must not be treated as an MQ6 projection"
        );
    }

    #[test]
    fn mixed_supported_tiers_include_v1_and_v2_affine() {
        // Kernel branch tags 7..18 consume MQ4V2/MQ6V2; admission must list
        // them alongside the preserved V1 tiers. Exact membership — no MQ2/3/5V2.
        assert!(MIXED_SUPPORTED_TIERS.contains(&DType::MQ4G256));
        assert!(MIXED_SUPPORTED_TIERS.contains(&DType::MQ6G256));
        assert!(MIXED_SUPPORTED_TIERS.contains(&DType::ParoQ4G128));
        assert!(MIXED_SUPPORTED_TIERS.contains(&DType::MQ4G256V2));
        assert!(MIXED_SUPPORTED_TIERS.contains(&DType::MQ6G256V2));
        assert_eq!(MIXED_SUPPORTED_TIERS.len(), 5);
        assert!(!MIXED_SUPPORTED_TIERS.contains(&DType::MQ5G256V2));
        assert!(!MIXED_SUPPORTED_TIERS.contains(&DType::MQ2G256V2));
        assert!(!MIXED_SUPPORTED_TIERS.contains(&DType::MQ3G256V2));
    }

    #[test]
    fn resolve_mq6v2_k_ne_8_disables_gpu_topk() {
        let mut d = uniform_mq4();
        d.routed_gate_up = DType::MQ6G256V2;
        d.routed_down = DType::MQ6G256V2;
        d.experts_all_gate_up_mq4 = false;
        let r = MoeResolution::resolve(&d, 6);
        assert!(r.routed_indexable_mq6v2);
        assert!(!r.use_gpu_topk);
    }

    fn uniform_prefill_dtypes(dtype: DType) -> MoeDtypes<'static> {
        let mut d = uniform_mq4();
        d.routed_gate_up = dtype;
        d.routed_down = dtype;
        d.experts_all_gate_up_mq4 = dtype == DType::MQ4G256;
        d.has_paro_shared = dtype == DType::ParoQ4G128;
        let shared_dtype = match dtype {
            DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8 => DType::Q8_0,
            _ => dtype,
        };
        d.shared = Some(MoeSharedDtypes {
            selector: DType::MQ4G256,
            gate: shared_dtype,
            up: shared_dtype,
            down: shared_dtype,
        });
        d
    }

    fn prefill_resolution_fields(r: &MoePrefillResolution) -> [bool; 8] {
        // Exhaustive destructuring makes a new resolution field require an
        // explicit comparison here, without changing the production API.
        let MoePrefillResolution {
            use_path2,
            down_path0,
            use_paro_i8,
            use_paro_i8_k8,
            paro_mode,
            force_mq4_grouped_fp16,
            fp8_b128_refused,
            qt53_refused,
        } = r;
        [
            *use_path2,
            *down_path0,
            *use_paro_i8,
            *use_paro_i8_k8,
            *paro_mode,
            *force_mq4_grouped_fp16,
            *fp8_b128_refused,
            *qt53_refused,
        ]
    }

    fn assert_resident_verify_matches_legacy(d: &MoeDtypes<'_>) {
        for arch_name in [
            "gfx1201", "gfx1200", "gfx1151", "gfx1100", "gfx1030", "gfx942",
        ] {
            for flag_case in 0..4 {
                let mut flags = rdna_compute::feature_flags::FeatureFlags::for_test(arch_name);
                match flag_case {
                    1 => flags.moe_grouped_gemm = false,
                    2 => flags.moe_grouped_i8 = Some(true),
                    3 => {
                        flags.moe_paro_i8 = Some(false);
                        flags.moe_paro_i8_k8 = Some(false);
                    }
                    _ => {}
                }
                let flags = std::sync::Arc::new(flags);
                let arch = rdna_compute::arch_caps::ArchCaps::new(arch_name, flags.clone());
                let legacy = MoePrefillResolution::resolve(d, &arch, &flags);
                // Unknown width, decode, every narrow verify width, and the
                // first wide verify all retain the resident legacy route.
                for batch in 0..=NARROW_VERIFY_MAX_ROWS + 1 {
                    let resident = MoePrefillResolution::resolve_with_batch_ctx(
                        d, &arch, &flags, batch, true, false,
                    );
                    assert_eq!(
                        prefill_resolution_fields(&resident),
                        prefill_resolution_fields(&legacy),
                        "{arch_name}, {:?}/{:?}, mixed={}, flags={flag_case}, batch={batch}",
                        d.routed_gate_up,
                        d.routed_down,
                        d.routed_has_mixed_experts,
                    );
                }
            }
        }
    }

    #[test]
    fn prefill_resident_verify_uniform_matches_legacy_all_fields() {
        for dtype in [
            DType::MQ4G256,
            DType::MQ4G256V2,
            DType::MQ6G256,
            DType::MQ6G256V2,
            DType::MQ5G256,
            DType::MFP4G32E8,
            DType::MFP3G32E8,
            DType::MFP2G32E8,
            DType::ParoQ4G128,
        ] {
            assert_resident_verify_matches_legacy(&uniform_prefill_dtypes(dtype));
        }
    }

    #[test]
    fn prefill_resident_verify_mixed_experts_matches_legacy_all_fields() {
        for representative in [
            DType::MQ4G256,
            DType::MQ6G256,
            DType::MQ6G256V2,
            DType::MQ5G256,
        ] {
            let mut d = uniform_prefill_dtypes(representative);
            d.routed_has_mixed_experts = true;
            d.experts_all_gate_up_mq4 = false;
            d.per_expert_gate_up = Some(&[DType::MQ4G256, DType::MQ6G256V2]);
            d.per_expert_down = Some(&[DType::MQ6G256, DType::MQ4G256V2]);
            assert_resident_verify_matches_legacy(&d);
        }
    }

    #[test]
    fn prefill_host_spill_uniform_mq4_gfx1201_narrow_readmits_path2() {
        let ctx = crate::context::DispatchCtx::for_test("gfx1201");
        for dtype in [DType::MQ4G256, DType::MQ4G256V2] {
            let d = uniform_prefill_dtypes(dtype);
            let legacy = MoePrefillResolution::resolve(&d, &ctx.arch, &ctx.flags);
            assert!(legacy.use_path2);
            for batch in 2..=NARROW_VERIFY_MAX_ROWS {
                let host = MoePrefillResolution::resolve_with_batch_ctx(
                    &d, &ctx.arch, &ctx.flags, batch, true, true,
                );
                assert!(host.use_path2, "{dtype:?}, batch={batch}");
                assert_eq!(
                    prefill_resolution_fields(&host),
                    prefill_resolution_fields(&legacy),
                    "{dtype:?}, batch={batch}",
                );
            }
            let decode = MoePrefillResolution::resolve_with_batch_ctx(
                &d, &ctx.arch, &ctx.flags, 1, true, true,
            );
            assert!(
                !decode.use_path2,
                "one row must not take the Path 2 re-admit"
            );

            let mut flags = rdna_compute::feature_flags::FeatureFlags::for_test("gfx1201");
            flags.moe_grouped_gemm = false;
            for batch in 2..=NARROW_VERIFY_MAX_ROWS {
                let disabled = MoePrefillResolution::resolve_with_batch_ctx(
                    &d, &ctx.arch, &flags, batch, true, true,
                );
                assert!(
                    !disabled.use_path2,
                    "re-admit must respect grouped-GEMM opt-out"
                );
            }
        }
    }

    #[test]
    fn prefill_host_spill_narrow_verify_keeps_indexed_policy() {
        for arch_name in ["gfx1201", "gfx1200", "gfx1151", "gfx1100"] {
            let ctx = crate::context::DispatchCtx::for_test(arch_name);
            for dtype in [
                DType::MQ4G256,
                DType::MQ4G256V2,
                DType::MQ6G256,
                DType::MQ6G256V2,
                DType::MQ5G256,
                DType::MFP4G32E8,
                DType::MFP3G32E8,
                DType::MFP2G32E8,
                DType::ParoQ4G128,
            ] {
                let d = uniform_prefill_dtypes(dtype);
                let legacy = MoePrefillResolution::resolve(&d, &ctx.arch, &ctx.flags);
                for batch in 1..=NARROW_VERIFY_MAX_ROWS {
                    if arch_name == "gfx1201"
                        && matches!(dtype, DType::MQ4G256 | DType::MQ4G256V2)
                        && batch >= 2
                    {
                        continue; // Covered by the uniform MQ4 re-admit test.
                    }
                    let host = MoePrefillResolution::resolve_with_batch_ctx(
                        &d, &ctx.arch, &ctx.flags, batch, true, true,
                    );
                    assert_eq!(
                        prefill_resolution_fields(&host),
                        [
                            false,
                            false,
                            false,
                            false,
                            dtype == DType::ParoQ4G128,
                            false,
                            false,
                            false // qt53_refused: none of these dtypes is qt53
                        ],
                        "{arch_name}, {dtype:?}, batch={batch}",
                    );
                }
                for batch in [0, NARROW_VERIFY_MAX_ROWS + 1] {
                    let host = MoePrefillResolution::resolve_with_batch_ctx(
                        &d, &ctx.arch, &ctx.flags, batch, true, true,
                    );
                    assert_eq!(
                        prefill_resolution_fields(&host),
                        prefill_resolution_fields(&legacy),
                        "unknown/wide host verify: {arch_name}, {dtype:?}, batch={batch}",
                    );
                }
                for batch in 0..=NARROW_VERIFY_MAX_ROWS + 1 {
                    let prompt = MoePrefillResolution::resolve_with_batch_ctx(
                        &d, &ctx.arch, &ctx.flags, batch, false, true,
                    );
                    assert_eq!(
                        prefill_resolution_fields(&prompt),
                        prefill_resolution_fields(&legacy),
                        "host prompt: {arch_name}, {dtype:?}, batch={batch}",
                    );
                }
            }
        }
    }

    #[test]
    fn prefill_host_spill_mixed_narrow_verify_keeps_grouped_route() {
        for arch_name in ["gfx1201", "gfx1200", "gfx1151", "gfx1100"] {
            let ctx = crate::context::DispatchCtx::for_test(arch_name);
            let mut d = uniform_prefill_dtypes(DType::MQ6G256);
            d.routed_has_mixed_experts = true;
            d.per_expert_gate_up = Some(&[DType::MQ4G256, DType::MQ6G256]);
            d.per_expert_down = Some(&[DType::MQ4G256, DType::MQ6G256]);
            let legacy = MoePrefillResolution::resolve(&d, &ctx.arch, &ctx.flags);
            assert!(legacy.use_path2);
            for batch in 1..=NARROW_VERIFY_MAX_ROWS {
                let host = MoePrefillResolution::resolve_with_batch_ctx(
                    &d, &ctx.arch, &ctx.flags, batch, true, true,
                );
                assert_eq!(
                    prefill_resolution_fields(&host),
                    prefill_resolution_fields(&legacy),
                    "{arch_name}, mixed host verify, batch={batch}",
                );
            }
        }
    }

    #[test]
    fn narrow_window_extent_lower_bound_is_two() {
        // The shared narrow-window extent: bumping it is a deliberate edit.
        assert_eq!(NARROW_VERIFY_MAX_ROWS, 4);
        // The splice-width gate (`n <= MAX`) accepts the 1-row decode window.
        for n in [1, 2, NARROW_VERIFY_MAX_ROWS] {
            assert!(n <= NARROW_VERIFY_MAX_ROWS, "splice gate must accept {n}");
        }
        assert!(NARROW_VERIFY_MAX_ROWS + 1 > NARROW_VERIFY_MAX_ROWS);
        // The verify/coalesced gates (`2..=MAX`) admit 2..=MAX but never the
        // 1-row decode window — that lower bound of 2 is load-bearing for the
        // gfx1201 Path-2 re-admit and must not be collapsed to `n <= MAX`.
        assert!(!(2..=NARROW_VERIFY_MAX_ROWS).contains(&1));
        assert!(!(2..=NARROW_VERIFY_MAX_ROWS).contains(&0));
        for n in 2..=NARROW_VERIFY_MAX_ROWS {
            assert!(
                (2..=NARROW_VERIFY_MAX_ROWS).contains(&n),
                "verify range must admit {n}"
            );
        }
        assert!(!(2..=NARROW_VERIFY_MAX_ROWS).contains(&(NARROW_VERIFY_MAX_ROWS + 1)));
    }

    #[test]
    fn cpu_wide_scope_truth_table_is_1_to_64_wiring_only() {
        // Wiring only — MUST NOT be counted as numeric proof. The G1/G2
        // family gates (real-layer exact parity, long-prefill staging) still
        // qualify 5..64; this test only pins the routing/admission truth:
        // default scope admits 1..=4, lab-wide scope admits whole 1..64
        // windows with allocated capacity, tails never qualify.
        use crate::cpu_split::{cpu_splice_window_qualified, CPU_SPLIT_MAX_ROWS};
        assert_eq!(NARROW_VERIFY_MAX_ROWS, 4);
        assert_eq!(CPU_SPLIT_STORAGE_MAX_ROWS, 64);
        let default = CpuWideVerifyScope::default_scope();
        assert!(!default.is_lab_wide());
        assert_eq!(default.qualified_max_rows(64), 4);
        // Value alone never arms: a wide GPU arm without override presence is
        // not CPU storage proof (G2/G4: never gate on few_row_limit-as-value).
        for verify in 1..=CPU_SPLIT_STORAGE_MAX_ROWS {
            let scope = CpuWideVerifyScope::from_verify_rows(verify);
            assert!(!scope.is_lab_wide(), "verify={verify}");
            assert_eq!(scope.qualified_max_rows(64), 4, "verify={verify}");
        }
        // Presence arms both lab widths: wide8 AND wide64 execute the FULL
        // logical R1..64 backend after actual storage (wide8 only closes the
        // GPU few-row projection routes above 8; it must not cap the CPU at 8).
        // Qualification is actual capacity clamped to storage — NEVER the
        // verify_rows value (required wide8-baseline R64 CPU reachability).
        assert_eq!(
            CpuWideVerifyScope::from_lab_presence(true, 8).qualified_max_rows(64),
            64,
        );
        for verify in [8usize, 9, 16, 64] {
            let scope = CpuWideVerifyScope::from_lab_presence(true, verify);
            assert!(scope.is_lab_wide(), "verify={verify}");
            for capacity in [0usize, 1, 4, 8, 40, 64, 65, 512] {
                assert_eq!(
                    scope.qualified_max_rows(capacity),
                    capacity.min(CPU_SPLIT_STORAGE_MAX_ROWS),
                    "verify={verify} capacity={capacity}"
                );
            }
        }
        // Lab-wide scope at every width qualifies actual capacity (capped by
        // storage), independent of the verify width; unarmed stays 4.
        for verify in 1..=CPU_SPLIT_STORAGE_MAX_ROWS {
            let scope = CpuWideVerifyScope::from_lab_presence(true, verify);
            assert!(scope.is_lab_wide(), "verify={verify}");
            assert_eq!(scope.qualified_max_rows(64), 64, "verify={verify}");
            assert_eq!(scope.qualified_max_rows(4), 4, "verify={verify}");
        }
        // Whole-window predicate across the full logical 1..64 range: armed
        // whole windows qualify iff within the qualified width; tails
        // (`rows != route_rows`) and zero rows never do.
        for rows in 1..=CPU_SPLIT_STORAGE_MAX_ROWS {
            assert!(
                cpu_splice_window_qualified(true, rows, rows, CPU_SPLIT_MAX_ROWS) == (rows <= 4)
            );
            assert!(cpu_wide_window_qualified(true, rows, rows, default, 64) == (rows <= 4));
            // wide8 presence executes the FULL R1..64 backend (not capped at 8).
            let lab8 = CpuWideVerifyScope::from_lab_presence(true, 8);
            assert!(
                cpu_wide_window_qualified(true, rows, rows, lab8, 64),
                "wide8 rows={rows}"
            );
            let lab64 = CpuWideVerifyScope::from_lab_presence(true, 64);
            assert!(cpu_wide_window_qualified(true, rows, rows, lab64, 64));
            assert!(!cpu_wide_window_qualified(true, rows, rows, lab64, 4) || rows <= 4);
            assert!(!cpu_wide_window_qualified(true, rows, rows + 1, lab64, 64));
            assert!(!cpu_wide_window_qualified(true, rows, 64, lab64, 64) || rows == 64);
        }
        assert!(!cpu_wide_window_qualified(true, 0, 0, default, 64));
        assert!(!cpu_wide_window_qualified(false, 4, 4, default, 64));
        // Storage gate: beyond-64 never resolves, even lab-wide.
        assert!(cpu_split_executor_wide_is_none_past_storage());
    }

    #[test]
    fn cpu_d1_wide_truth_is_1_to_64_wiring_only() {
        // Wiring only — MUST NOT be counted as numeric proof. Default keeps the
        // legacy `<= 4` gate; lab presence with prewarmed scratch admits whole
        // logical 1..64 windows; tails/zero/unprewarmed refuse.
        let default = CpuWideVerifyScope::default_scope();
        let lab64 = CpuWideVerifyScope::from_lab_presence(true, 64);
        let lab8 = CpuWideVerifyScope::from_lab_presence(true, 8);
        for rows in 1..=CPU_SPLIT_STORAGE_MAX_ROWS {
            assert!(cpu_d1_wide_window_qualified(true, rows, rows, default, 64) == (rows <= 4));
            // No prewarm (0 rows): only legacy windows qualify, even when armed.
            assert!(cpu_d1_wide_window_qualified(true, rows, rows, lab64, 0) == (rows <= 4));
            // Prewarmed 64: full window qualifies (wide8 presence included).
            assert!(cpu_d1_wide_window_qualified(true, rows, rows, lab64, 64));
            assert!(cpu_d1_wide_window_qualified(true, rows, rows, lab8, 64));
            // Tails never qualify.
            assert!(!cpu_d1_wide_window_qualified(
                true,
                rows,
                rows + 1,
                lab64,
                64
            ));
            // Unprewarmed geometry reports 0 rows (fail-closed): zero dims and
            // an unreserved thread hold no rows.
            assert_eq!(cpu_d1_prewarmed_scratch_rows(0, 640, 10, 0), 0);
            assert_eq!(cpu_d1_prewarmed_scratch_rows(2560, 0, 10, 0), 0);
            assert_eq!(cpu_d1_prewarmed_scratch_rows(2560, 640, 0, 0), 0);
        }
        assert!(!cpu_d1_wide_window_qualified(true, 0, 0, lab64, 64));
        assert!(!cpu_d1_wide_window_qualified(false, 4, 4, lab64, 64));
    }

    #[cfg(test)]
    fn cpu_split_executor_wide_is_none_past_storage() -> bool {
        // The storage gate refuses 65+ structurally (no executor needed: the
        // row bound itself rejects). Proved by the bound, not by hardware.
        65 > CPU_SPLIT_STORAGE_MAX_ROWS
    }

    fn fp8_b128_refusal(err: DispatchError) -> bool {
        matches!(
            err,
            DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "fp8b128_specialized_route_only",
                quant: "FP8E4M3B128",
                ..
            }
        )
    }

    /// Every place a qt=55 operand can sit: the generic resolutions refuse
    /// each one with all path flags disabled.
    fn fp8_b128_operand_cases() -> Vec<MoeDtypes<'static>> {
        static FP8_TIERS: [DType; 2] = [DType::FP8E4M3B128, DType::FP8E4M3B128];
        static MQ4_TIERS: [DType; 2] = [DType::MQ4G256, DType::MQ4G256];
        let mut cases = Vec::new();
        let mut routed = uniform_mq4();
        routed.routed_gate_up = DType::FP8E4M3B128;
        routed.routed_down = DType::FP8E4M3B128;
        cases.push(routed);
        let mut down_only = uniform_mq4();
        down_only.routed_down = DType::FP8E4M3B128;
        cases.push(down_only);
        let mut router = uniform_mq4();
        router.router = DType::FP8E4M3B128;
        cases.push(router);
        let mut shared = uniform_mq4();
        shared.shared.as_mut().unwrap().down = DType::FP8E4M3B128;
        cases.push(shared);
        let mut tiers = uniform_mq4();
        tiers.per_expert_gate_up = Some(&MQ4_TIERS);
        tiers.per_expert_down = Some(&FP8_TIERS);
        cases.push(tiers);
        cases
    }

    #[test]
    fn fp8_capability_predicates_are_disjoint_from_qt44() {
        assert!(MoeRouteCapability::Fp8B128Grouped.is_fp8_grouped());
        assert!(!MoeRouteCapability::Fp8B128Grouped.is_qt44_qt53_grouped());
        for qt44 in [
            MoeRouteCapability::Qt44Qt53Grouped,
            MoeRouteCapability::Qt44Qt53GroupedSymmetric,
        ] {
            assert!(qt44.is_qt44_qt53_grouped());
            assert!(!qt44.is_fp8_grouped());
        }
    }

    #[test]
    fn fp8_b128_decode_resolution_is_a_typed_refusal_state() {
        for arch_name in ["gfx942", "gfx1151", "gfx1201", "gfx1100"] {
            let ctx = crate::context::DispatchCtx::for_test(arch_name);
            for d in fp8_b128_operand_cases() {
                assert!(d.has_fp8_b128_projection());
                let r = MoeResolution::resolve_arch(&d, 8, ctx.arch.has_wmma());
                assert!(r.fp8_b128_refused, "{arch_name}");
                assert!(!r.use_gpu_topk && !r.routed_indexable() && !r.mixed);
                assert!(!r.gate_side_mq4 && !r.gate_fusable && !r.gate_fusable_mq4v2);
                assert!(!r.needs_x_rot_local && !r.routed_indexable_mq2lloyd_u);
                assert!(fp8_b128_refusal(r.check_generic_route(None).unwrap_err()));
                assert!(fp8_b128_refusal(
                    r.check_generic_route(Some(MoeRouteCapability::Qt44Qt53Grouped))
                        .unwrap_err()
                ));
                r.check_generic_route(Some(MoeRouteCapability::Fp8B128Grouped))
                    .unwrap();
            }
        }
    }

    #[test]
    fn fp8_b128_prefill_resolution_disables_every_path() {
        for arch_name in ["gfx942", "gfx1151", "gfx1201", "gfx1100"] {
            let ctx = crate::context::DispatchCtx::for_test(arch_name);
            for d in fp8_b128_operand_cases() {
                for (batch, verify, host) in [
                    (0, false, false),
                    (1, false, false),
                    (2, true, true),
                    (4, true, false),
                ] {
                    let r = MoePrefillResolution::resolve_with_batch_ctx(
                        &d, &ctx.arch, &ctx.flags, batch, verify, host,
                    );
                    assert_eq!(
                        prefill_resolution_fields(&r),
                        // qwen35 qt53 routed-down typed fallback: eighth field is qt53_refused.
                        [false, false, false, false, false, false, true, false],
                        "{arch_name} batch={batch}",
                    );
                    assert!(fp8_b128_refusal(r.check_generic_route(None).unwrap_err()));
                    assert!(fp8_b128_refusal(
                        r.check_generic_route(Some(MoeRouteCapability::Qt44Qt53Grouped))
                            .unwrap_err()
                    ));
                    r.check_generic_route(Some(MoeRouteCapability::Fp8B128Grouped))
                        .unwrap();
                }
            }
        }
    }

    #[test]
    fn non_fp8_resolution_has_no_refusal_state() {
        let ctx = crate::context::DispatchCtx::for_test("gfx942");
        let d = uniform_mq4();
        assert!(!d.has_fp8_b128_projection());
        let decode = MoeResolution::resolve_arch(&d, 8, ctx.arch.has_wmma());
        assert!(!decode.fp8_b128_refused);
        decode.check_generic_route(None).unwrap();
        let prefill = MoePrefillResolution::resolve(&d, &ctx.arch, &ctx.flags);
        assert!(!prefill.fp8_b128_refused);
        prefill.check_generic_route(None).unwrap();
    }

    /// The cold codec comes from the layer's whole per-expert metadata, never
    /// from the first expert or the arch alone.
    #[test]
    fn slab_cold_codec_is_read_from_every_expert_pair() {
        const MQ4: (DType, DType) = (DType::MQ4G256V2, DType::MQ4G128V2);
        const W3: (DType, DType) = (DType::MQ3I4G128R256, DType::MQ3I4G128R128);
        const E8P: (DType, DType) = (DType::E8PI4G128R256, DType::E8PI4G128R128);
        const W2: (DType, DType) = (DType::MQ2I4G128R256, DType::MQ2I4G128R128);
        let layer = |pairs: &[(DType, DType)], admits: &dyn Fn(u8) -> bool| {
            let gate_up: Vec<DType> = pairs.iter().map(|pair| pair.0).collect();
            let down: Vec<DType> = pairs.iter().map(|pair| pair.1).collect();
            let mut dtypes = uniform_mq4();
            dtypes.routed_has_mixed_experts = true;
            dtypes.routed_gate_up = gate_up[0];
            dtypes.routed_down = down[0];
            layer_cold_codec(
                &MoeDtypes {
                    per_expert_gate_up: Some(&gate_up),
                    per_expert_down: Some(&down),
                    ..dtypes
                },
                admits,
            )
        };
        let all = |_: u8| true;
        // MQ4 first, one cold expert far behind it.
        assert_eq!(layer(&[MQ4, MQ4, MQ4, E8P], &all), Some(Some(3)));
        assert_eq!(layer(&[MQ4, W3, MQ4], &all), Some(Some(1)));
        assert_eq!(layer(&[E8P, MQ4], &all), Some(Some(3)));
        // Two cold codecs in one layer, W2 anywhere, an MQ4-only or cold-only
        // mixed table all fail closed.
        assert_eq!(layer(&[MQ4, W3, E8P], &all), None);
        assert_eq!(layer(&[MQ4, MQ4, W2], &all), None);
        assert_eq!(layer(&[MQ4, MQ4], &all), None);
        assert_eq!(layer(&[W3, W3], &all), None);
        // An arch that refuses the cold codec refuses the layer, even when
        // the first expert is MQ4.
        assert_eq!(layer(&[MQ4, MQ4, E8P], &|codec| codec == 0), None);
        assert_eq!(layer(&[MQ4, E8P], &|codec| codec != 3), None);
        // gfx1151's own admission, with the E8P receipts not landed and landed: the layer's cold
        // codec is read from every expert; E8P is refused (even far behind MQ4) until the one
        // switch flips, W3 is served either way, and W3 + E8P is never one layer.
        let gfx1151 = |landed: bool| {
            move |codec: u8| {
                rdna_compute::slab_admission::qwen4_slab_codec_admitted_with(
                    "gfx1151", codec, landed,
                )
            }
        };
        let (refused, landed) = (gfx1151(false), gfx1151(true));
        assert_eq!(layer(&[MQ4, MQ4, MQ4, E8P], &refused), None);
        assert_eq!(layer(&[E8P, MQ4], &refused), None);
        assert_eq!(layer(&[MQ4, MQ4, MQ4, E8P], &landed), Some(Some(3)));
        assert_eq!(layer(&[E8P, MQ4], &landed), Some(Some(3)));
        for admits in [&refused as &dyn Fn(u8) -> bool, &landed] {
            assert_eq!(layer(&[MQ4, W3, MQ4], admits), Some(Some(1)));
            assert_eq!(layer(&[MQ4, W3, E8P], admits), None);
            assert_eq!(layer(&[MQ4, MQ4, W2], admits), None);
        }
        let uniform_e8p = MoeDtypes {
            routed_gate_up: E8P.0,
            routed_down: E8P.1,
            ..uniform_mq4()
        };
        assert_eq!(layer_cold_codec(&uniform_e8p, &refused), None);
        assert_eq!(layer_cold_codec(&uniform_e8p, &landed), Some(Some(3)));
        // Uniform layers: MQ4 has no cold codec, W3 / E8P are their own.
        let uniform = |pair: (DType, DType)| {
            let dtypes = MoeDtypes {
                routed_gate_up: pair.0,
                routed_down: pair.1,
                ..uniform_mq4()
            };
            layer_cold_codec(&dtypes, &all)
        };
        assert_eq!(uniform(MQ4), Some(None));
        assert_eq!(uniform(W3), Some(Some(1)));
        assert_eq!(uniform(E8P), Some(Some(3)));
        assert_eq!(uniform(W2), None);
    }

    fn qt53_fallback_dtypes() -> MoeDtypes<'static> {
        MoeDtypes {
            routed_gate_up: DType::MQ4G256V2,
            routed_down: DType::MQ4G128V2,
            experts_all_gate_up_mq4: false,
            ..uniform_mq4()
        }
    }

    /// Pure shape table of the qwen35 qt53 routed-down typed fallback: only
    /// uniform qt44 gate/up + uniform qt53 down, qt53 nowhere else.
    #[test]
    fn qt53_fallback_dtype_shape_table() {
        static MIXED_DOWN: [DType; 2] = [DType::MQ4G128V2, DType::MQ4G256V2];
        static MIXED_GATE_UP: [DType; 2] = [DType::MQ4G256V2, DType::MQ4G256];
        static UNIFORM_GATE_UP: [DType; 3] = [DType::MQ4G256V2; 3];
        static UNIFORM_DOWN: [DType; 3] = [DType::MQ4G128V2; 3];
        let base = qt53_fallback_dtypes();
        assert!(base.qt53_routed_down_fallback());
        let accepted = MoeDtypes {
            per_expert_gate_up: Some(&UNIFORM_GATE_UP),
            per_expert_down: Some(&UNIFORM_DOWN),
            ..base
        };
        assert!(accepted.qt53_routed_down_fallback());
        let shared = |edit: fn(&mut MoeSharedDtypes)| {
            let mut d = qt53_fallback_dtypes();
            let mut s = d.shared.unwrap();
            edit(&mut s);
            d.shared = Some(s);
            d
        };
        let refused = [
            MoeDtypes {
                routed_down: DType::MQ4G256V2,
                ..base
            },
            MoeDtypes {
                routed_gate_up: DType::MQ4G256,
                ..base
            },
            MoeDtypes {
                routed_gate_up: DType::MQ4G128V2,
                routed_down: DType::MQ4G256V2,
                ..base
            },
            MoeDtypes {
                routed_gate_up: DType::MQ4G128V2,
                ..base
            },
            MoeDtypes {
                routed_has_mixed_experts: true,
                ..base
            },
            MoeDtypes {
                has_paro_shared: true,
                ..base
            },
            MoeDtypes {
                router: DType::MQ4G128V2,
                ..base
            },
            MoeDtypes {
                per_expert_down: Some(&MIXED_DOWN),
                ..base
            },
            MoeDtypes {
                per_expert_gate_up: Some(&MIXED_GATE_UP),
                ..base
            },
            MoeDtypes {
                shared: None,
                routed_down: DType::MQ6G256V2,
                ..base
            },
            shared(|s| s.selector = DType::MQ4G128V2),
            shared(|s| s.gate = DType::MQ4G128V2),
            shared(|s| s.up = DType::MQ4G128V2),
            shared(|s| s.down = DType::MQ4G128V2),
            shared(|s| s.down = DType::FP8E4M3B128),
            MoeDtypes {
                router: DType::FP8E4M3B128,
                ..base
            },
        ];
        for (index, d) in refused.iter().enumerate() {
            assert!(!d.qt53_routed_down_fallback(), "case {index}");
        }
        // No-shared recipe with the same routed pair stays in the shape.
        assert!(MoeDtypes {
            shared: None,
            ..base
        }
        .qt53_routed_down_fallback());
    }

    /// Decode resolution: the shape lowers on the GPU top-K indexed path only at
    /// k == 8; any other k is a typed refusal (no per-expert CPU fallback has a
    /// qt53 GEMV). qt53 never claims a V2 256 indexed flag.
    #[test]
    fn qt53_fallback_decode_resolution_requires_gpu_topk_k8() {
        let d = qt53_fallback_dtypes();
        for has_wmma in [false, true] {
            let r = MoeResolution::resolve_arch(&d, 8, has_wmma);
            assert!(r.routed_indexable_qt53_f32 && r.routed_indexable());
            assert!(r.use_gpu_topk && r.needs_x_rot_local && !r.qt53_fallback_refused);
            assert!(!r.routed_indexable_mq4v2 && !r.mixed && !r.fp8_b128_refused);
            r.check_generic_route(None).unwrap();
            for k in [1usize, 4, 10] {
                let r = MoeResolution::resolve_arch(&d, k, has_wmma);
                assert!(!r.use_gpu_topk && r.qt53_fallback_refused, "k={k}");
                assert!(matches!(
                    r.check_generic_route(None),
                    Err(DispatchError::UnsupportedVariant {
                        family: "moe",
                        variant: "qt53-routed-down-fallback-needs-gpu-topk-k8",
                        quant: "MQ4G128V2",
                        ..
                    })
                ));
            }
        }
        // A non-qt53 shape never carries the qt53 flags.
        let r = MoeResolution::resolve_arch(&uniform_mq4(), 8, false);
        assert!(!r.routed_indexable_qt53_f32 && !r.qt53_fallback_refused);
    }

    /// Prefill resolution: Path 0 (gfx9 atomic GEMV, no qt53 down) refuses the
    /// shape before any launch; Path 1 / Path 2 archs keep their legacy routing.
    #[test]
    fn qt53_fallback_prefill_resolution_refuses_only_path0_archs() {
        let d = qt53_fallback_dtypes();
        for (arch_name, path0, path2) in [
            ("gfx942", true, false),
            ("gfx906", true, false),
            ("gfx1030", false, false),
            ("gfx1100", false, true),
            ("gfx1151", false, true),
            ("gfx1201", false, true),
        ] {
            let ctx = crate::context::DispatchCtx::for_test(arch_name);
            let r = MoePrefillResolution::resolve(&d, &ctx.arch, &ctx.flags);
            assert_eq!((r.down_path0, r.use_path2), (path0, path2), "{arch_name}");
            assert_eq!(r.qt53_refused, path0, "{arch_name}");
            match r.check_generic_route(None) {
                Ok(()) => assert!(!path0, "{arch_name}"),
                Err(err) => assert!(
                    path0
                        && matches!(
                            err,
                            DispatchError::UnsupportedVariant {
                                family: "moe",
                                variant: "qt53-routed-down-fallback-path0-unsupported",
                                quant: "MQ4G128V2",
                                ..
                            }
                        ),
                    "{arch_name}"
                ),
            }
            // Every other dtype shape keeps the legacy fields exactly.
            let legacy = MoePrefillResolution::resolve(&uniform_mq4(), &ctx.arch, &ctx.flags);
            assert!(!legacy.qt53_refused, "{arch_name}");
        }
    }

    /// The Flash-Next route (`route_policy = Some(Qt44Qt53Grouped*)`) never
    /// consults the qt53 fallback refusal: its fixed top-10 / Path 0 shapes
    /// still pass the generic guards exactly as before.
    #[test]
    fn qt53_fallback_refusal_is_exempt_for_sealed_grouped_routes() {
        let d = qt53_fallback_dtypes();
        let decode = MoeResolution::resolve_arch(&d, 10, false);
        assert!(decode.qt53_fallback_refused && !decode.use_gpu_topk);
        let ctx = crate::context::DispatchCtx::for_test("gfx942");
        let prefill = MoePrefillResolution::resolve(&d, &ctx.arch, &ctx.flags);
        assert!(prefill.qt53_refused);
        for route in [
            MoeRouteCapability::Qt44Qt53Grouped,
            MoeRouteCapability::Qt44Qt53GroupedSymmetric,
            MoeRouteCapability::SlabCodecGrouped,
        ] {
            decode.check_generic_route(Some(route)).unwrap();
            prefill.check_generic_route(Some(route)).unwrap();
        }
        assert!(decode.check_generic_route(None).is_err());
        assert!(prefill.check_generic_route(None).is_err());
    }

    /// Batched-prefill dtype snapshot admission: only uniform qt44 gate/up +
    /// qt53 down over MQ4/MQ6 G256 shared projections; qt53 anywhere else or
    /// beside another gate/up stays refused.
    #[test]
    fn qt53_fallback_prefill_snapshot_admission_is_narrow() {
        let base = MoePrefillDtypes {
            expert_gate_up: DType::MQ4G256V2,
            expert_down: DType::MQ4G128V2,
            ..MoePrefillDtypes::uniform(DType::MQ4G256V2)
        };
        let admits = |d: &MoePrefillDtypes, mq6: bool| {
            gated_moe_prefill_admissible_for_dtypes(d, mq6, false, false, false)
        };
        assert!(admits(&base, false) && admits(&base, true));
        assert!(!admits(
            &MoePrefillDtypes {
                expert_gate_up: DType::MQ4G256,
                ..base
            },
            false
        ));
        assert!(!admits(
            &MoePrefillDtypes {
                expert_gate_up: DType::MQ6G256V2,
                ..base
            },
            true
        ));
        assert!(!admits(
            &MoePrefillDtypes {
                shared_expert_down: DType::MQ4G128V2,
                ..base
            },
            false
        ));
        assert!(!admits(
            &MoePrefillDtypes {
                shared_expert_gate: DType::MQ4G128V2,
                shared_expert_up: DType::MQ4G128V2,
                ..base
            },
            false
        ));
        assert!(!admits(
            &MoePrefillDtypes {
                router: DType::MQ4G128V2,
                ..base
            },
            false
        ));
        assert!(!admits(
            &MoePrefillDtypes {
                expert_down_uniform: false,
                ..base
            },
            false
        ));
        assert!(!admits(
            &MoePrefillDtypes {
                expert_gate_up_uniform: false,
                ..base
            },
            false
        ));
        // MQ6 shared projections are only admitted under the MQ6 policy switch.
        let mq6_shared = MoePrefillDtypes {
            shared_expert_gate: DType::MQ6G256V2,
            shared_expert_up: DType::MQ6G256V2,
            shared_expert_down: DType::MQ6G256V2,
            ..base
        };
        assert!(!admits(&mq6_shared, false) && admits(&mq6_shared, true));
        // The pre-existing qt53-down refusal for any other gate_up is unchanged.
        assert!(!admits(
            &MoePrefillDtypes {
                expert_down: DType::MQ4G128V2,
                ..MoePrefillDtypes::uniform(DType::MQ4G256)
            },
            false
        ));
    }
}
