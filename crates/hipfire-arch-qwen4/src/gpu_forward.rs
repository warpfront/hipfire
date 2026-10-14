// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Ordinary-HIP Qwen4 forward execution.
//!
//! This module is the production device path.  The CPU/reference equations in
//! reference_forward are intentionally separate and are never called
//! here.  All
//! learned projections consume the resident tensor supplied by the assembled
//! bundle; MQv2 projections are rotated and dispatched through their native
//! qt44/qt53 kernels, while recurrent/cache state remains resident on the GPU.

use crate::bundle::Qwen4Bundle;
use crate::config::{LayerType, Qwen4Config, Qwen4FfnConfig};
use crate::ple_stage::{
    ple_async_upload, ple_async_upload_enabled, PleHostStage, PLE_ASYNC_UPLOAD_ENV,
};
use crate::program::{
    execute_final_hyper, execute_lm_head, validate_final_hyper, validate_lm_head,
    Qwen4AttentionWeights, Qwen4DenseBinding, Qwen4FfnBinding, Qwen4FfnDims, Qwen4GdnWeights,
    Qwen4HyperReadWeights, Qwen4HyperWeights, Qwen4LayerDescription, Qwen4LayerScratch,
    Qwen4MoeBinding, Qwen4PleWeights, Qwen4ProgramDims, Qwen4QsaWeights, Qwen4ScratchLayout,
    QWEN4_QSA_KERNEL_ROPE_THETA,
};
use crate::projection::{row_stride, ProjectionView};
use crate::weights::{
    HyperConnectionReadWeights, HyperConnectionWeights, MoeWeights, Qwen4EpPlacement,
    Qwen4FfnWeights, Qwen4LayerWeights, Qwen4Weights, TensorRef, WeightError,
};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::moe::{
    MoeDtypes, MoeEpMode, MoeNormalization, MoeParams, MoeRecipe, MoeRouteCapability,
    MoeRouteFormats, MoeRouteGeometry, MoeRoutePolicy, MoeRowGather, MoeSharedDecode,
    MoeSharedDtypes, MoeSharedWeights, MoeStageTables, MoeTileExchange, MoeTileExchangeParams,
    RoutedExpertWeights,
};
use hipfire_dispatch::pipeline::sealed_moe::{
    retained_body_action, specialized_sealed_moe_retained_admission, RetainedBodyAction,
    SealedMoeCall,
};
use hipfire_dispatch::pipeline::{
    execute_embedding, execute_validated_steps, execute_validated_steps_with_moe_hooks,
    seal_decode, validate_steps, BoundMoeExperts, ClearOp, DenseFfnOp, DenseFfnRecipe, EmbeddingOp,
    ExpertBindingCache, ExpertMetadata, ExpertResource, ExpertResources, ExpertTable,
    GatedDeltaNetOp, GroupedDepthwiseOp, HyperReadOp, HyperWriteOp, IndexedAttentionMode,
    IndexedAttentionOp, IndexedAttentionState, MixerPhase, MoeStepHooks, Step,
};
use hipfire_dispatch::types::DispatchError;
use hipfire_dispatch::types::dtype_rotation_plan;
use hipfire_runtime::external_rows::{RowFetch, RowTicket};
use hipfire_runtime::weight_manifest::ExpertSourceLayout;
use rdna_compute::moe::Qwen4EpOwnerTables;
use rdna_compute::replay::{ReplayController, ReplayState, ShadowBodyRoute};
use rdna_compute::tensor_ops::{argmax_f32, ArgmaxF32};
use rdna_compute::{DType, Gpu, GpuTensor, TrunkFamily};
use smallvec::SmallVec;
use std::cell::{Cell, RefCell};
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

#[cfg(target_os = "linux")]
#[path = "npu_trunk/forward.rs"]
mod npu_forward;

pub(crate) const EPSILON: f32 = 1.0e-6;
/// Default rows of one Qwen4 prefill chunk (the reusable forward scratch).
///
/// Public serving calls may receive longer prompts; the forward owner tiles
/// those requests over this bounded capacity instead of allocating
/// prompt-sized grouped MoE buffers.  Every chunk re-streams all routed
/// expert weights, so pp throughput rises with the chunk until another
/// kernel's shape envelope ends.  Measured against the earlier 1536-row
/// chunk: gfx1151 pp8192 +10.2% at 8192 rows (resident experts).  On
/// gfx1201 under `auto` expert placement the larger chunk costs VRAM expert
/// layers, so the rung is the fastest pp8192 within 3% of the 2048-row
/// rung's tg64: 4096 rows (14 layers at max_seq 66,560) measured pp8192
/// +14.1% over 2048 rows (16 layers) at -2.9% tg64; 8192 rows (11 layers)
/// measured +22.6% at -6.9% tg64.  Other architectures keep the measured
/// 1536.
pub fn qwen4_prefill_chunk_default(arch: &str) -> usize {
    match arch {
        "gfx1151" => 8192,
        "gfx1201" => 4096,
        _ => 1536,
    }
}

/// Requested prefill chunk ceiling: `prefill.chunk_rows`
/// (`HIPFIRE_PREFILL_CHUNK_ROWS`) rounded down to whole 256-row GEMM tiles,
/// else [`qwen4_prefill_chunk_default`]; never above `max_seq`.  Load-time
/// admission ([`qwen4_prefill_chunk_rungs`]) may still pick a smaller rung.
pub fn qwen4_prefill_chunk_requested(arch: &str, max_seq: usize) -> usize {
    hipfire_config::developer_var("HIPFIRE_PREFILL_CHUNK_ROWS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .map(|rows| rows / 256 * 256)
        .filter(|&rows| rows >= 256)
        .unwrap_or_else(|| qwen4_prefill_chunk_default(arch))
        .min(max_seq)
}

/// Chunk rungs admission tries, largest first: `requested`, then each
/// smaller default rung down to 1536 (or `requested` itself when smaller).
pub(crate) fn qwen4_prefill_chunk_rungs(requested: usize) -> impl Iterator<Item = usize> {
    std::iter::once(requested).chain(
        [4096, 2048, 1536]
            .into_iter()
            .filter(move |&rows| rows < requested),
    )
}

/// Rows of the speculative logit and argmax buffers.  Only verify blocks
/// return every row's argmax (native MTP verifies at most `K + 1 = 11`
/// rows; block drafters on the shared `SpecTarget` contract get the rest);
/// prompt advances and MTP prefill read their final row only.  A
/// chunk-sized buffer cost `vocab * 4` bytes per chunk row.
pub(crate) const QWEN4_SPEC_VERIFY_ROWS: usize = 64;

pub(crate) fn qwen4_spec_logit_rows(max_chunk: usize) -> usize {
    QWEN4_SPEC_VERIFY_ROWS.min(max_chunk)
}

/// Allocation plan of one [`Qwen4GpuForwardScratch`].
///
/// `Full` is the historical plan: one dedicated allocation per field, every
/// MoE buffer sized for `max_chunk` rows of the widest (F32) route.
/// `FnSymGfx1201` is the proven-route plan of a gfx1201 Flash-Next artifact
/// whose routed headers are all symmetric: the MoE buffers are sized for the
/// symmetric IU4 prefill route plus a [`QWEN4_COMPACT_FALLBACK_ROWS`]-row F32
/// fallback, and eight phase-disjoint fields borrow their storage from an
/// owner (see [`Qwen4GpuForwardScratch::plan`]).  Both modes run identical
/// arithmetic; the mode only changes where intermediates live, so it must be
/// decided before expert placement and frozen for the forward's lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen4ForwardScratchMode {
    Full,
    FnSymGfx1201,
}

/// Identical native trunk padding scope for sequential and paired prefill.
fn native_trunk_prefill_scope(prefill: bool, iu4: bool, arch: &str, mode: Qwen4ForwardScratchMode) -> bool {
    prefill && iu4 && arch == "gfx1151" && mode == Qwen4ForwardScratchMode::Full
}

/// Rows the `FnSymGfx1201` plan sizes the F32 fallback MoE buffers for.  A
/// larger MoE invocation that is not on the symmetric IU4 route has no buffer
/// to run in; dispatch refuses it before its first launch.
pub const QWEN4_COMPACT_FALLBACK_ROWS: usize = 512;

/// Storage-only opt-in for the compact scratch (`1` on, default off; `0`
/// leaves the IU4 math enabled and only keeps the full allocation plan).
pub const QWEN4_COMPACT_SCRATCH_ENV: &str = "HIPFIRE_QWEN4_COMPACT_SCRATCH";

/// Pure predicate of [`qwen4_forward_scratch_mode`]: every term must hold
/// for the compact plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CompactScratchTerms {
    pub knob: bool,
    pub exact_gfx1201: bool,
    pub grouped_gemm: bool,
    pub sym_iu4_requested: bool,
    pub source_symmetric: bool,
    pub ple_key_value_bf16: bool,
    pub flash_next_profile: bool,
}

impl CompactScratchTerms {
    /// Exactly gfx1201: no other arch, and no family prefix match.
    pub(crate) fn is_exact_gfx1201(arch: &str) -> bool {
        arch == "gfx1201"
    }

    pub(crate) fn mode(self) -> Qwen4ForwardScratchMode {
        if self.knob
            && self.exact_gfx1201
            && self.grouped_gemm
            && self.sym_iu4_requested
            && self.source_symmetric
            && self.ple_key_value_bf16
            && self.flash_next_profile
        {
            Qwen4ForwardScratchMode::FnSymGfx1201
        } else {
            Qwen4ForwardScratchMode::Full
        }
    }
}

/// The device, flag and profile terms of [`qwen4_forward_scratch_mode`]
/// with the artifact-dependent terms assumed true.  `false` means the mode is
/// `Full` whatever the artifact holds, so a caller can skip its payload
/// header scan.
pub fn qwen4_compact_scratch_possible(gpu: &Gpu, config: &Qwen4Config) -> bool {
    qwen4_forward_scratch_mode(gpu, config, true, true) == Qwen4ForwardScratchMode::FnSymGfx1201
}

/// The scratch plan a load takes: `FnSymGfx1201` only for the opted-in exact
/// gfx1201 Flash-Next symmetric-IU4 route.  `source_symmetric` is the proof
/// from the artifact's actual routed payload headers
/// ([`crate::artifact::routed_headers_symmetric`]), not metadata;
/// `ple_key_value_bf16` states that the PLE key/value projections are BF16.
/// Cheap arch/flag checks come first so a caller may skip its header scan
/// when this would be `Full` regardless.  [`Qwen4GpuForward::new`] still
/// confirms the symmetric proof on the loaded payloads and refuses a
/// `FnSymGfx1201` forward the GPU check does not fully confirm.
pub fn qwen4_forward_scratch_mode(
    gpu: &Gpu,
    config: &Qwen4Config,
    source_symmetric: bool,
    ple_key_value_bf16: bool,
) -> Qwen4ForwardScratchMode {
    static KNOB: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| hipfire_config::developer_bool(QWEN4_COMPACT_SCRATCH_ENV, false));
    CompactScratchTerms {
        knob: *KNOB,
        exact_gfx1201: CompactScratchTerms::is_exact_gfx1201(&gpu.arch),
        grouped_gemm: gpu.flags.moe_grouped_gemm,
        sym_iu4_requested: gpu.qwen4_moe_sym_iu4_requested(),
        source_symmetric,
        ple_key_value_bf16,
        flash_next_profile: matches!(config.profile, crate::config::Qwen4Profile::FlashNext),
    }
    .mode()
}

/// Device bytes of the forward resources that scale with the chunk: the
/// reusable scratch for `rows` rows plus the speculative logits and argmax.
/// Load admission and the auto expert placement reserve size against this.
pub fn qwen4_forward_device_bytes(
    config: &Qwen4Config,
    rows: usize,
    mode: Qwen4ForwardScratchMode,
) -> Option<u64> {
    let scratch = Qwen4GpuForwardScratch::device_bytes(config, rows, mode).ok()?;
    let spec = qwen4_spec_logit_rows(rows)
        .checked_mul(config.vocab_size.checked_mul(4)?.checked_add(4)?)?;
    scratch.checked_add(spec as u64)
}

/// Free device memory a chunk rung must leave beside its chunk-sized
/// resources for the kernels' lazily sized workspaces (333-347 MiB at
/// pp8192 on gfx1201, measured as load-time free minus prefill-time free).
pub const QWEN4_FORWARD_HEADROOM_BYTES: u64 = 1 << 30;

const QWEN4_STEP_INLINE_CAPACITY: usize = 384;
const QWEN4_QSA_INLINE_CAPACITY: usize = 12;

/// Long-prefill host-expert staging (G2). `1`/`0` force it on/off; unset it
/// is on for gfx1201 only, the card it was measured on. Placement is not
/// touched: a staged prefill borrows the VRAM of the last two resident expert
/// layers as its two DMA stages, and restores them from pinned host copies
/// before the forward returns, so decode keeps today's residency.
pub const QWEN4_EXPERT_STAGE_ENV: &str = "HIPFIRE_QWEN4_EXPERT_STAGE";

/// Forwards of at least this many rows stage ([`QWEN4_EXPERT_STAGE_ENV`]);
/// smaller ones, decode and verify keep the mapped reads.
pub const QWEN4_EXPERT_STAGE_MIN_ROWS_ENV: &str = "HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS";

/// Whether a long forward that captures wide hidden rows (the native-MTP
/// prompt fill's trunk tiles) stages like any other long forward. `0` keeps
/// those tiles on the mapped reads; unset or `1` stages them. The capture
/// only copies the trunk's output streams after the program has run, on the
/// compute stream, and never reads an expert, so it is byte-identical either
/// way. Verify forwards capture too but stay below the row threshold.
pub const QWEN4_EXPERT_STAGE_CAPTURE_ENV: &str = "HIPFIRE_QWEN4_EXPERT_STAGE_CAPTURE";

/// A 1,336,934,400-byte layer copy takes 23.5 ms (card B, gfx1201). A
/// 2048-row forward already hides it: pp2048 went from 1057 to 1377 tok/s,
/// and the restore that was not hidden cost 18 ms of that chunk. Smaller
/// forwards keep the mapped path (K3: 1536 rows expose the DMA).
const QWEN4_EXPERT_STAGE_MIN_ROWS_DEFAULT: usize = 2048;

/// One Flash-Next routed expert's QT44 gate/up and QT53 down bytes.
const QWEN4_EXPERT_STAGE_GATE_UP_BYTES: usize = 1_740_800;
const QWEN4_EXPERT_STAGE_DOWN_BYTES: usize = 870_400;

fn qwen4_expert_stage_min_rows() -> usize {
    hipfire_config::developer_var(QWEN4_EXPERT_STAGE_MIN_ROWS_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&rows| rows >= 2)
        .unwrap_or(QWEN4_EXPERT_STAGE_MIN_ROWS_DEFAULT)
}

/// Which rows of a batched Qwen4 prefill write language-model logits.
///
/// `All` is the public chunk/capture contract.  `Final` keeps the complete
/// trunk batch but emits only its final row for ordinary autoregressive AR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen4OutputRows {
    All,
    Final,
}

impl Qwen4OutputRows {
    pub(crate) fn count(self, n: usize) -> usize {
        match self {
            Self::All => n,
            Self::Final => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Qwen4OutputPolicy {
    Rows(Qwen4OutputRows),
    None,
}

impl Qwen4OutputPolicy {
    fn requested_rows(self, n: usize) -> Option<usize> {
        match self {
            Self::Rows(rows) => Some(rows.count(n)),
            Self::None => None,
        }
    }
}

/// Host-side forward phases timed when profiling is enabled.
#[derive(Clone, Copy)]
pub(crate) enum Qwen4ProfilePhase {
    MoeSeal,
    PleWait,
    PleStage,
    PleUpload,
    PleApply,
}

impl Qwen4ProfilePhase {
    pub(crate) const ALL: [Self; 5] = [
        Self::MoeSeal,
        Self::PleWait,
        Self::PleStage,
        Self::PleUpload,
        Self::PleApply,
    ];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::MoeSeal => "moe_seal",
            Self::PleWait => "ple_wait",
            Self::PleStage => "ple_stage",
            Self::PleUpload => "ple_upload",
            Self::PleApply => "ple_apply",
        }
    }
}

/// Calls and host nanoseconds per [`Qwen4ProfilePhase`], indexed by phase.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Qwen4ProfileStats {
    pub(crate) calls: [u64; Qwen4ProfilePhase::ALL.len()],
    pub(crate) ns: [u64; Qwen4ProfilePhase::ALL.len()],
}

thread_local! {
    static QWEN4_PROFILE_ENABLED: Cell<bool> = const { Cell::new(false) };
    static QWEN4_PROFILE_STATS: RefCell<Qwen4ProfileStats> =
        RefCell::new(Qwen4ProfileStats::default());
}

#[inline(always)]
fn qwen4_profile_enabled() -> bool {
    QWEN4_PROFILE_ENABLED.with(|enabled| enabled.get())
}

pub(crate) fn qwen4_profile_enable(enabled: bool) {
    QWEN4_PROFILE_ENABLED.with(|current| current.set(enabled));
}

pub(crate) fn qwen4_profile_reset() {
    QWEN4_PROFILE_STATS.with(|stats| *stats.borrow_mut() = Qwen4ProfileStats::default());
}

pub(crate) fn qwen4_profile_snapshot() -> Qwen4ProfileStats {
    QWEN4_PROFILE_STATS.with(|stats| *stats.borrow())
}

#[inline(always)]
fn qwen4_profile_start() -> Option<Instant> {
    if qwen4_profile_enabled() {
        Some(Instant::now())
    } else {
        None
    }
}

#[inline(always)]
fn qwen4_profile_record(phase: Qwen4ProfilePhase, started: Option<Instant>) {
    let Some(started) = started else {
        return;
    };
    let elapsed = started.elapsed().as_nanos() as u64;
    QWEN4_PROFILE_STATS.with(|stats| {
        let mut stats = stats.borrow_mut();
        let slot = phase as usize;
        stats.calls[slot] = stats.calls[slot].saturating_add(1);
        stats.ns[slot] = stats.ns[slot].saturating_add(elapsed);
    });
}

/// Errors returned by the native ordinary-HIP path.
#[derive(Debug)]
pub enum Qwen4GpuForwardError {
    Hip(hip_bridge::HipError),
    Weights(WeightError),
    Invalid(String),
    Dispatch(String),
    Ple(String),
}

impl fmt::Display for Qwen4GpuForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hip(error) => write!(f, "Qwen4 GPU forward HIP error: {error}"),
            Self::Weights(error) => write!(f, "Qwen4 GPU forward weights error: {error}"),
            Self::Invalid(error) => write!(f, "Qwen4 GPU forward invalid input: {error}"),
            Self::Dispatch(error) => write!(f, "Qwen4 GPU forward dispatch error: {error}"),
            Self::Ple(error) => write!(f, "Qwen4 GPU forward PLE error: {error}"),
        }
    }
}

impl std::error::Error for Qwen4GpuForwardError {}

impl From<hip_bridge::HipError> for Qwen4GpuForwardError {
    fn from(error: hip_bridge::HipError) -> Self {
        Self::Hip(error)
    }
}

impl From<WeightError> for Qwen4GpuForwardError {
    fn from(error: WeightError) -> Self {
        Self::Weights(error)
    }
}

pub(crate) fn invalid(message: impl Into<String>) -> Qwen4GpuForwardError {
    Qwen4GpuForwardError::Invalid(message.into())
}

fn checked_bytes(rows: usize, stride: usize) -> Result<usize, Qwen4GpuForwardError> {
    crate::projection::checked_bytes(rows, stride)
        .ok_or_else(|| invalid("packed row byte size overflow"))
}

pub(crate) fn f32_view(tensor: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    debug_assert_eq!(tensor.dtype, DType::F32);
    tensor.sub_offset(offset, len)
}

pub(crate) fn view(tensor: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    tensor.sub_offset(offset, len)
}

/// Byte range `(offset, len)` in `scratch.token_ids` of `tokens` at row
/// `offset`, when the forward's upload (`uploaded_rows` rows mirrored in
/// `host_bytes`, native-endian i32) holds exactly those ids there.
fn uploaded_ids_range(
    uploaded_rows: usize,
    host_bytes: &[u8],
    offset: usize,
    tokens: &[u32],
) -> Option<(usize, usize)> {
    const ID_BYTES: usize = std::mem::size_of::<i32>();
    let end = offset.checked_add(tokens.len())?;
    if tokens.is_empty() || end > uploaded_rows {
        return None;
    }
    let start_byte = offset.checked_mul(ID_BYTES)?;
    let end_byte = end.checked_mul(ID_BYTES)?;
    let mirror = host_bytes.get(start_byte..end_byte)?;
    mirror
        .chunks_exact(ID_BYTES)
        .zip(tokens)
        .all(|(bytes, &token)| bytes == (token as i32).to_ne_bytes())
        .then_some((start_byte, end_byte - start_byte))
}

pub(crate) fn matrix_view(
    tensor: &GpuTensor,
    rows: usize,
    columns: usize,
) -> Result<GpuTensor, Qwen4GpuForwardError> {
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| invalid("Qwen4 matrix view shape overflows"))?;
    let mut view = tensor.sub_offset(0, elements);
    view.shape = vec![rows, columns];
    Ok(view)
}

fn weight_dims(reference: &TensorRef) -> Result<(usize, usize), Qwen4GpuForwardError> {
    if reference.shape.len() != 2 {
        return Err(invalid(format!(
            "projection {} has logical shape {:?}, expected rank two",
            reference.name, reference.shape
        )));
    }
    Ok((reference.shape[0], reference.shape[1]))
}
pub(crate) fn dense_ref<'a>(
    weights: &'a Qwen4Weights,
    reference: &TensorRef,
) -> Result<WeightRef<'a>, Qwen4GpuForwardError> {
    let tensor = weights.resident(reference)?;
    let (m, k) = weight_dims(reference)?;
    Ok(WeightRef {
        buf: tensor,
        dtype: tensor.dtype,
        m,
        k,
        row_stride: row_stride(tensor.dtype, k),
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    })
}

/// The admitted Qwen4 grouped route is declared by this architecture
/// constructor and consumed as data by shared sealing/lowering.  The shared
/// expert's formats are those of the bound weights: BF16, or the Q8_0 decode
/// copies (`Qwen4GpuForward::decode_q8`). `symmetric` (a prefill chunk over
/// a layer whose expert headers were verified symmetric) declares the
/// symmetric variant the opt-in IU4 arm requires.
fn qwen4_route_policy(
    config: &Qwen4Config,
    shared: &MoeSharedWeights<'_>,
    symmetric: bool,
) -> MoeRoutePolicy {
    MoeRoutePolicy {
        capability: if symmetric {
            MoeRouteCapability::Qt44Qt53GroupedSymmetric
        } else {
            MoeRouteCapability::Qt44Qt53Grouped
        },
        geometry: MoeRouteGeometry {
            experts: config.num_experts,
            top_k: config.num_experts_per_tok,
            hidden: config.hidden_size,
            routed_intermediate: config.moe_intermediate_size,
            shared_intermediate: config.shared_expert_intermediate_size,
        },
        formats: MoeRouteFormats {
            router: DType::BF16,
            shared_selector: DType::BF16,
            shared_gate: shared.gate.dtype,
            shared_up: shared.up.dtype,
            shared_down: shared.down.dtype,
            routed_gate_up: DType::MQ4G256V2,
            routed_down: DType::MQ4G128V2,
        },
    }
}
pub(crate) fn program_dims(config: &Qwen4Config) -> Qwen4ProgramDims {
    Qwen4ProgramDims {
        hidden: config.hidden_size,
        hc_count: config.hc_count,
        hc_lowrank: config.hc_lowrank,
        indexer_n_heads: config.indexer_n_heads,
        indexer_kv_heads: config.indexer_kv_heads,
        indexer_head_dim: config.indexer_head_dim,
        indexer_budget: config.indexer_budget,
        indexer_compress_ratio: config.indexer_compress_ratio,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        linear_num_key_heads: config.linear_num_key_heads,
        linear_num_value_heads: config.linear_num_value_heads,
        linear_key_head_dim: config.linear_key_head_dim,
        linear_value_head_dim: config.linear_value_head_dim,
        linear_conv_kernel_dim: config.linear_conv_kernel_dim,
        ffn: match &config.ffn {
            Qwen4FfnConfig::Dense { intermediate_size } => Qwen4FfnDims::Dense {
                intermediate: *intermediate_size,
            },
            Qwen4FfnConfig::Moe {
                num_experts,
                top_k,
                intermediate_size,
                shared_intermediate_size,
                ..
            } => Qwen4FfnDims::Moe {
                num_experts: *num_experts,
                experts_per_token: *top_k,
                intermediate: *intermediate_size,
                // A shared-expert-less MoE has no executor; `validate_dims`
                // rejects the zero extent.
                shared_intermediate: shared_intermediate_size.unwrap_or(0),
            },
        },
        ple_conv_kernel_dim: config.ple_conv_kernel_size,
        norm_eps: EPSILON,
    }
}

/// Columns of the router-logit scratch view (`num_experts`; one placeholder
/// column for a dense FFN, which routes nothing).
pub(crate) fn router_columns(dims: Qwen4ProgramDims) -> usize {
    match dims.ffn {
        Qwen4FfnDims::Moe { num_experts, .. } => num_experts,
        Qwen4FfnDims::Dense { .. } => 1,
    }
}

/// Q8_0 decode copies per layer (`Qwen4GpuForward::decode_q8`).
pub(crate) const DECODE_Q8_PER_LAYER: usize = 7;

/// A decode-shaped forward's projection (one token, or the few rows of a
/// speculative verify, which must match it bitwise): its Q8_0 decode copy
/// (`Qwen4GpuForward::decode_q8`) when one exists.
pub(crate) fn decode_copy<'a>(
    copies: &'a [GpuTensor],
    index: usize,
    rows: usize,
    source: WeightRef<'a>,
) -> WeightRef<'a> {
    match copies.get(index) {
        Some(buf) if rows <= 8 => WeightRef {
            buf,
            dtype: DType::Q8_0,
            row_stride: row_stride(DType::Q8_0, source.k),
            ..source
        },
        _ => source,
    }
}

pub(crate) fn hyper_read_desc<'a>(
    weights: &'a Qwen4Weights,
    hyper: &HyperConnectionReadWeights,
) -> Result<Qwen4HyperReadWeights<'a>, Qwen4GpuForwardError> {
    Ok(Qwen4HyperReadWeights {
        norm: weights.resident(&hyper.hc_norm)?,
        input_mix_down: dense_ref(weights, &hyper.input_mix_down)?,
        input_mix_up: dense_ref(weights, &hyper.input_mix_up)?,
    })
}

pub(crate) fn hyper_desc<'a>(
    weights: &'a Qwen4Weights,
    hyper: &HyperConnectionWeights,
) -> Result<Qwen4HyperWeights<'a>, Qwen4GpuForwardError> {
    let norm = weights.resident(&hyper.hc_norm)?;
    Ok(Qwen4HyperWeights {
        read: Qwen4HyperReadWeights {
            norm,
            input_mix_down: dense_ref(weights, &hyper.input_mix_down)?,
            input_mix_up: dense_ref(weights, &hyper.input_mix_up)?,
        },
        write: crate::program::Qwen4HyperWriteWeights {
            norm,
            block_inject: dense_ref(weights, &hyper.block_inject)?,
        },
    })
}

fn gdn_desc<'a>(
    weights: &'a Qwen4Weights,
    gdn: &crate::weights::GdnWeights,
) -> Result<Qwen4GdnWeights<'a>, Qwen4GpuForwardError> {
    Ok(Qwen4GdnWeights {
        qkv: dense_ref(weights, &gdn.qkv)?,
        conv: weights.resident(&gdn.conv)?,
        in_proj_a: dense_ref(weights, &gdn.in_proj_a)?,
        in_proj_b: dense_ref(weights, &gdn.in_proj_b)?,
        a_log: weights.resident(&gdn.a_log)?,
        dt_bias: weights.resident(&gdn.dt_bias)?,
        z: dense_ref(weights, &gdn.z)?,
        norm: weights.resident(&gdn.norm)?,
        output: dense_ref(weights, &gdn.output)?,
    })
}

pub(crate) fn qsa_desc<'a>(
    weights: &'a Qwen4Weights,
    qsa: &crate::weights::QsaWeights,
) -> Result<Qwen4QsaWeights<'a>, Qwen4GpuForwardError> {
    Ok(Qwen4QsaWeights {
        indexer_qk: dense_ref(weights, &qsa.indexer_qk)?,
        indexer_q_norm: weights.resident(&qsa.indexer_q_norm)?,
        indexer_k_norm: weights.resident(&qsa.indexer_k_norm)?,
        q: dense_ref(weights, &qsa.q)?,
        k: dense_ref(weights, &qsa.k)?,
        v: dense_ref(weights, &qsa.v)?,
        q_norm: weights.resident(&qsa.q_norm)?,
        k_norm: weights.resident(&qsa.k_norm)?,
        output: dense_ref(weights, &qsa.output)?,
    })
}
pub(crate) fn ple_desc<'a>(
    weights: &'a Qwen4Weights,
    ple: &crate::weights::PleWeights,
) -> Result<Qwen4PleWeights<'a>, Qwen4GpuForwardError> {
    Ok(Qwen4PleWeights {
        key: dense_ref(weights, &ple.key)?,
        value: dense_ref(weights, &ple.value)?,
        norm_key: weights.resident(&ple.norm_key)?,
        norm_query: weights.resident(&ple.norm_query)?,
        norm_conv: weights.resident(&ple.norm_conv)?,
        conv: weights.resident(&ple.conv)?,
    })
}

/// `HIPFIRE_QWEN4_PLE_FUSE`: whether a `rows`-row forward stages the PLE rows
/// straight as F16 for the BF16 key/value projections (the singleton chunk
/// forward and the prefill wave decide it with this one expression; under the
/// invariant prefill profile the F16 route applies at every row count).
pub(crate) fn qwen4_ple_rows_f16(
    gpu: &Gpu,
    bf16_state: bool,
    hc_count: usize,
    hidden_size: usize,
    ple: &Qwen4PleWeights<'_>,
    rows: usize,
) -> bool {
    bf16_state
        && gpu.flags.qwen4_ple_fuse_enabled()
        && hc_count <= 4
        && hidden_size % 32 == 0
        && ple.key.k == hidden_size
        && ple.value.k == hidden_size
        && gpu.qwen4_f16_wmma_applies(ple.key.buf, ple.key.k, rows)
        && gpu.qwen4_f16_wmma_applies(ple.value.buf, ple.value.k, rows)
}

/// `decode_q8` / `rows`: the forward's Q8_0 decode copies and row count; a
/// one-row forward binds this layer's shared-expert copies.  `moe` is the
/// layer's MoE runtime and is `Some` exactly for a MoE FFN.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_desc<'a>(
    weights: &'a Qwen4Weights,
    layer: &Qwen4LayerWeights,
    moe: Option<&'a Qwen4MoeLayerRuntime>,
    config: &Qwen4Config,
    decode_q8: &'a [GpuTensor],
    rows: usize,
    stage: Option<&'a Qwen4ExpertStage>,
) -> Result<Qwen4LayerDescription<'a>, Qwen4GpuForwardError> {
    let attention = match layer.kind {
        LayerType::LinearAttention => Qwen4AttentionWeights::Linear(gdn_desc(
            weights,
            layer
                .gdn
                .as_ref()
                .ok_or_else(|| invalid("GDN weights missing"))?,
        )?),
        LayerType::FullAttention => Qwen4AttentionWeights::Full(qsa_desc(
            weights,
            layer
                .attention
                .as_ref()
                .ok_or_else(|| invalid("QSA weights missing"))?,
        )?),
    };
    let ffn = match (&layer.ffn, moe) {
        (Qwen4FfnWeights::Dense(dense), None) => {
            let gate = dense_ref(weights, &dense.gate)?;
            Qwen4FfnBinding::Dense(Qwen4DenseBinding {
                intermediate: gate.m,
                gate,
                up: dense_ref(weights, &dense.up)?,
                down: dense_ref(weights, &dense.down)?,
            })
        }
        (Qwen4FfnWeights::Moe(_), Some(moe)) => {
            let copy = |slot: usize, source: &'a ProjectionView| {
                decode_copy(
                    decode_q8,
                    layer.layer * DECODE_Q8_PER_LAYER + slot,
                    rows,
                    source.dispatch_ref(),
                )
            };
            let shared = MoeSharedWeights {
                selector: moe.shared_scalar.dispatch_ref(),
                gate: copy(4, &moe.shared_gate),
                up: copy(5, &moe.shared_up),
                down: copy(6, &moe.shared_down),
            };
            Qwen4FfnBinding::Moe(Qwen4MoeBinding {
                route_policy: qwen4_route_policy(
                    config,
                    &shared,
                    moe.symmetric && rows >= rdna_compute::gemm::QWEN4_MOE_SYM_IU4_MIN_ROWS,
                ),
                table: &moe.table,
                cache: &moe.cache,
                routed_experts: moe,
                router: moe.router.dispatch_ref(),
                shared,
                intermediate: moe
                    .experts
                    .first()
                    .map(|expert| expert.gate_up.m)
                    .unwrap_or_default()
                    / 2,
                experts_all_gate_up_mq4: moe.experts_all_gate_up_mq4,
                expert_gate_up_ptrs: &moe.expert_gate_up_ptrs,
                expert_down_ptrs: &moe.expert_down_ptrs,
                expert_stage_ptrs: stage
                    .filter(|_| moe.host_mapped())
                    .map(|stage| stage.tables(layer.layer)),
                expert_gate_up_entries: &moe.expert_gate_up_entries,
                expert_down_entries: &moe.expert_down_entries,
                layer_idx: layer.layer as u16,
                norm_topk_prob: config.norm_topk_prob,
                row_gather: None,
                shared_down_out: None,
                tile_exchange: None,
            })
        }
        _ => {
            return Err(invalid(format!(
                "layer {} FFN weights and MoE runtime disagree",
                layer.layer
            )))
        }
    };
    Ok(Qwen4LayerDescription {
        attn_hyper: hyper_desc(weights, &layer.attn_hyper)?,
        mlp_hyper: hyper_desc(weights, &layer.mlp_hyper)?,
        attention,
        ffn,
    })
}

pub(crate) struct Qwen4ExpertView {
    gate_up: ProjectionView,
    down: ProjectionView,
}

struct ExpertViewSet<'a> {
    experts: &'a [Qwen4ExpertView],
    identity: u64,
    host_mapped: bool,
}

impl RoutedExpertWeights for ExpertViewSet<'_> {
    fn len(&self) -> usize {
        self.experts.len()
    }

    fn get(&self, expert_idx: usize) -> Option<(WeightRef<'_>, WeightRef<'_>)> {
        self.experts
            .get(expert_idx)
            .map(|expert| (expert.gate_up.dispatch_ref(), expert.down.dispatch_ref()))
    }

    fn immutable_identity(&self) -> Option<u64> {
        Some(self.identity)
    }

    fn host_mapped(&self) -> bool {
        self.host_mapped
    }
}

/// Source of [`Qwen4MoeLayerRuntime::identity`].
static NEXT_EXPERT_SET_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Architecture-owned sealed-MoE resources.  Pointer tables and expert views
/// are built once with the assembled resident weights and reused for every
/// token; no per-token resource or metadata allocation occurs.
pub(crate) struct Qwen4MoeLayerRuntime {
    table: ExpertTable,
    cache: ExpertBindingCache,
    router: ProjectionView,
    shared_scalar: ProjectionView,
    shared_gate: ProjectionView,
    shared_up: ProjectionView,
    shared_down: ProjectionView,
    /// `experts` is built once in `from_moe` and never mutated, so this
    /// construction-unique id is its [`RoutedExpertWeights::immutable_identity`].
    experts: Vec<Qwen4ExpertView>,
    identity: u64,
    experts_all_gate_up_mq4: bool,
    expert_gate_up_ptrs: GpuTensor,
    expert_down_ptrs: GpuTensor,
    /// Host entries the two tables above were uploaded from. Kept so dispatch can
    /// prove the table contents name the live expert tensors before a retained
    /// body records them.
    expert_gate_up_entries: Vec<usize>,
    expert_down_entries: Vec<usize>,
    /// Every routed QT44 gate/up and QT53 down header behind the two tables
    /// satisfies `zp == -8*sc`, checked on the device once after load
    /// ([`Self::verify_symmetric`]); only then may this layer declare
    /// [`MoeRouteCapability::Qt44Qt53GroupedSymmetric`]. False unless the
    /// opt-in route was requested.
    symmetric: bool,
    /// Whether this rank's own (local) resident expert tensors of the layer
    /// are host-mapped: a layer-level property, so an EP rank reports the
    /// same residency single would, never that of a dummy or a peer view.
    host_mapped: bool,
}

impl RoutedExpertWeights for Qwen4MoeLayerRuntime {
    fn len(&self) -> usize {
        self.experts.len()
    }

    fn get(&self, expert_idx: usize) -> Option<(WeightRef<'_>, WeightRef<'_>)> {
        self.experts
            .get(expert_idx)
            .map(|expert| (expert.gate_up.dispatch_ref(), expert.down.dispatch_ref()))
    }

    fn immutable_identity(&self) -> Option<u64> {
        Some(self.identity)
    }

    fn host_mapped(&self) -> bool {
        self.host_mapped
    }
}

/// Gather buffers one EP rank owns: parity `L % 2` over the trunk layers and
/// parity `2 + j % 2` for the MTP head, each `QWEN4_EP_GATHER_MAX_ROWS *
/// num_experts_per_tok * hidden` F32.
pub const QWEN4_EP_GATHER_BUFFERS: usize = 4;

/// Most rows a few-row forward (verify / replay / takeover) may carry through
/// the EP table + row gather; larger chunks run the full (P2P) table.
pub const QWEN4_EP_GATHER_MAX_ROWS: usize = 8;

/// Which EP gather buffer a MoE invocation uses: trunk layer `L` uses
/// `L % 2`; the MTP head (a layer index at or past the trunk layer count)
/// uses `2 + (layer_idx - trunk_layers) % 2`.
pub(crate) fn qwen4_ep_buffer_index(layer_idx: usize, trunk_layers: usize) -> usize {
    if layer_idx >= trunk_layers {
        2 + (layer_idx - trunk_layers) % 2
    } else {
        layer_idx % 2
    }
}

/// Device pointers and byte lengths of one rank's compact routed-expert
/// tensors of one MoE layer (`512 / ranks` experts, local index `e / ranks`).
/// The same shape describes the per-rank zero dummy expert (one expert).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4PeerExpertSource {
    pub gate_up_base: usize,
    pub gate_up_bytes: usize,
    pub down_base: usize,
    pub down_bytes: usize,
}

/// How one rank's per-layer expert pointer table is populated.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Qwen4ExpertTableLayout<'a> {
    /// Every expert from the layer's resident tensors (one device).
    Single,
    /// EP decode table: owned experts name the compact local tensors,
    /// every other expert names the rank's zero dummy expert.
    Ep {
        placement: Qwen4EpPlacement,
        dummy: Qwen4PeerExpertSource,
    },
    /// EP chunk table: owned experts name the local compact tensors, every
    /// other expert names the owning peer's compact tensors (P2P), `peers`
    /// indexed by rank.
    Full {
        placement: Qwen4EpPlacement,
        peers: &'a [Qwen4PeerExpertSource],
    },
}

/// The allocation one expert-table entry names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Qwen4ExpertOrigin {
    /// This rank's resident (compact) expert tensors.
    Local,
    /// This rank's zero dummy expert.
    Dummy,
    /// The compact expert tensors of the named peer rank.
    Peer(usize),
}

/// Host pointer entries of one expert in the gate/up and down tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Qwen4ExpertEntry {
    pub(crate) origin: Qwen4ExpertOrigin,
    pub(crate) gate_up_ptr: usize,
    pub(crate) down_ptr: usize,
}

fn validate_ep_placement(
    placement: Qwen4EpPlacement,
    num_experts: usize,
) -> Result<(), Qwen4GpuForwardError> {
    if placement.ranks == 0 || placement.rank >= placement.ranks {
        return Err(invalid(format!(
            "EP placement rank {} is outside {} ranks",
            placement.rank, placement.ranks
        )));
    }
    if num_experts % placement.ranks != 0 {
        return Err(invalid(format!(
            "{num_experts} experts do not shard evenly over {} EP ranks",
            placement.ranks
        )));
    }
    Ok(())
}

impl Qwen4ExpertTableLayout<'_> {
    /// Experts the layer's own resident tensors hold under this layout.
    fn local_experts(&self, num_experts: usize) -> Result<usize, Qwen4GpuForwardError> {
        match self {
            Self::Single => Ok(num_experts),
            Self::Ep { placement, .. } | Self::Full { placement, .. } => {
                validate_ep_placement(*placement, num_experts)?;
                Ok(num_experts / placement.ranks)
            }
        }
    }
}

/// Device address of expert slot `index` of a tensor `(base, total)` bytes
/// long holding `expert_bytes` per slot; the whole slot must lie inside the
/// tensor.
fn expert_slot_ptr(
    what: &str,
    base: usize,
    total: usize,
    index: usize,
    expert_bytes: usize,
) -> Result<usize, Qwen4GpuForwardError> {
    let offset = index
        .checked_mul(expert_bytes)
        .ok_or_else(|| invalid(format!("{what} expert {index} offset overflows")))?;
    let end = offset
        .checked_add(expert_bytes)
        .ok_or_else(|| invalid(format!("{what} expert {index} end overflows")))?;
    if base == 0 {
        return Err(invalid(format!("{what} base pointer is null")));
    }
    if end > total {
        return Err(invalid(format!(
            "{what} expert slot {index} [{offset}, {end}) exceeds the {total}-byte tensor"
        )));
    }
    base.checked_add(offset)
        .ok_or_else(|| invalid(format!("{what} expert {index} address overflows")))
}

/// Pure arithmetic behind every expert pointer table: the host entry of each
/// global expert under `layout`. `local` names this rank's own resident
/// tensors (all experts under `Single`, the compact `num_experts / ranks`
/// under EP). Nothing here touches a device.
pub(crate) fn qwen4_expert_table_entries(
    layout: &Qwen4ExpertTableLayout<'_>,
    local: Qwen4PeerExpertSource,
    num_experts: usize,
    gate_up_expert_bytes: usize,
    down_expert_bytes: usize,
) -> Result<Vec<Qwen4ExpertEntry>, Qwen4GpuForwardError> {
    let mut entries = Vec::with_capacity(num_experts);
    match layout {
        Qwen4ExpertTableLayout::Single => {
            for expert in 0..num_experts {
                entries.push(Qwen4ExpertEntry {
                    origin: Qwen4ExpertOrigin::Local,
                    gate_up_ptr: expert_slot_ptr(
                        "gate/up",
                        local.gate_up_base,
                        local.gate_up_bytes,
                        expert,
                        gate_up_expert_bytes,
                    )?,
                    down_ptr: expert_slot_ptr(
                        "down",
                        local.down_base,
                        local.down_bytes,
                        expert,
                        down_expert_bytes,
                    )?,
                });
            }
        }
        Qwen4ExpertTableLayout::Ep { placement, dummy } => {
            let per_rank = layout.local_experts(num_experts)?;
            check_exact_compact(local, per_rank, gate_up_expert_bytes, down_expert_bytes)?;
            if dummy.gate_up_bytes != gate_up_expert_bytes || dummy.down_bytes != down_expert_bytes
            {
                return Err(invalid(format!(
                    "EP dummy expert holds {}+{} bytes, one expert is {gate_up_expert_bytes}+{down_expert_bytes}",
                    dummy.gate_up_bytes, dummy.down_bytes
                )));
            }
            for expert in 0..num_experts {
                entries.push(if placement.owns(expert) {
                    let slot = placement.local_index(expert);
                    Qwen4ExpertEntry {
                        origin: Qwen4ExpertOrigin::Local,
                        gate_up_ptr: expert_slot_ptr(
                            "gate/up",
                            local.gate_up_base,
                            local.gate_up_bytes,
                            slot,
                            gate_up_expert_bytes,
                        )?,
                        down_ptr: expert_slot_ptr(
                            "down",
                            local.down_base,
                            local.down_bytes,
                            slot,
                            down_expert_bytes,
                        )?,
                    }
                } else {
                    Qwen4ExpertEntry {
                        origin: Qwen4ExpertOrigin::Dummy,
                        gate_up_ptr: expert_slot_ptr(
                            "dummy gate/up",
                            dummy.gate_up_base,
                            dummy.gate_up_bytes,
                            0,
                            gate_up_expert_bytes,
                        )?,
                        down_ptr: expert_slot_ptr(
                            "dummy down",
                            dummy.down_base,
                            dummy.down_bytes,
                            0,
                            down_expert_bytes,
                        )?,
                    }
                });
            }
        }
        Qwen4ExpertTableLayout::Full { placement, peers } => {
            let per_rank = layout.local_experts(num_experts)?;
            check_exact_compact(local, per_rank, gate_up_expert_bytes, down_expert_bytes)?;
            if peers.len() != placement.ranks {
                return Err(invalid(format!(
                    "EP full table needs {} peer sources, got {}",
                    placement.ranks,
                    peers.len()
                )));
            }
            if peers[placement.rank] != local {
                return Err(invalid(format!(
                    "EP peer source of rank {} does not name this rank's own expert tensors",
                    placement.rank
                )));
            }
            for expert in 0..num_experts {
                let owner = expert % placement.ranks;
                let slot = expert / placement.ranks;
                entries.push(if owner == placement.rank {
                    Qwen4ExpertEntry {
                        origin: Qwen4ExpertOrigin::Local,
                        gate_up_ptr: expert_slot_ptr(
                            "gate/up",
                            local.gate_up_base,
                            local.gate_up_bytes,
                            slot,
                            gate_up_expert_bytes,
                        )?,
                        down_ptr: expert_slot_ptr(
                            "down",
                            local.down_base,
                            local.down_bytes,
                            slot,
                            down_expert_bytes,
                        )?,
                    }
                } else {
                    let peer = peers[owner];
                    Qwen4ExpertEntry {
                        origin: Qwen4ExpertOrigin::Peer(owner),
                        gate_up_ptr: expert_slot_ptr(
                            "peer gate/up",
                            peer.gate_up_base,
                            peer.gate_up_bytes,
                            slot,
                            gate_up_expert_bytes,
                        )?,
                        down_ptr: expert_slot_ptr(
                            "peer down",
                            peer.down_base,
                            peer.down_bytes,
                            slot,
                            down_expert_bytes,
                        )?,
                    }
                });
            }
        }
    }
    Ok(entries)
}

/// An EP rank's resident tensors hold exactly `per_rank` experts.
fn check_exact_compact(
    local: Qwen4PeerExpertSource,
    per_rank: usize,
    gate_up_expert_bytes: usize,
    down_expert_bytes: usize,
) -> Result<(), Qwen4GpuForwardError> {
    let gate_up = checked_bytes(per_rank, gate_up_expert_bytes)?;
    let down = checked_bytes(per_rank, down_expert_bytes)?;
    if local.gate_up_bytes != gate_up || local.down_bytes != down {
        return Err(invalid(format!(
            "EP compact expert tensors hold {}+{} bytes, expected exactly {per_rank} experts = {gate_up}+{down}",
            local.gate_up_bytes, local.down_bytes
        )));
    }
    Ok(())
}

/// A non-owning whole-allocation window over a peer rank's expert tensor,
/// only ever cut into per-expert views (never freed by this rank).
fn foreign_packed_source(base: usize, bytes: usize, dtype: DType) -> GpuTensor {
    GpuTensor {
        // SAFETY: `base` names a live peer allocation of `bytes` bytes (the
        // peer's resident expert tensor, P2P-mapped before any launch), and
        // the wrapper is `Borrowed`, so neither this rank's `free_tensor`
        // nor a drop releases it.
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(base as *mut std::ffi::c_void, bytes) },
        shape: vec![bytes],
        dtype,
    }
}

/// One per-rank zero-filled expert in the routed format (QT44 gate/up, QT53
/// down), shared by every layer's EP decode table for the experts the rank
/// does not own.
///
/// All-zero bytes dequantize to exactly 0, so the producers (which run
/// unchanged on all ten slots) emit zero rows for non-owned slots, which the
/// row gather then overwrites with the owner's bits:
/// * gate/up (`qwen4_gemv_mq4g256v2_moe_gate_up_k8_indexed_batched.hip`): the
///   group header is f16 `sc`, f16 `zp` read from bytes `[0,4)` / `[4,8)`;
///   every weight is `sc * nibble + zp` = `0 * 0 + 0` = +0.0 in f32, so each
///   product is `+-0 * x` (x is a finite rotated activation) and every
///   accumulator (initialised to +0.0) stays +0.0;
/// * down (`gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded.hip`):
///   `scale`, `zero` are the f16 at group bytes `[0,2)` / `[2,4)`; the same
///   arithmetic gives `acc = +0.0`, written to the expanded row.
/// No division, reciprocal or NaN/inf source exists in either dequant.
pub(crate) struct Qwen4EpDummyExpert {
    gate_up: GpuTensor,
    down: GpuTensor,
}

impl Qwen4EpDummyExpert {
    /// Encoded bytes of one routed expert `(gate_up, down)`.
    fn expert_bytes(config: &Qwen4Config) -> Result<(usize, usize), Qwen4GpuForwardError> {
        let gate_up = checked_bytes(
            2 * config.moe_intermediate_size,
            row_stride(DType::MQ4G256V2, config.hidden_size),
        )?;
        let down = checked_bytes(
            config.hidden_size,
            row_stride(DType::MQ4G128V2, config.moe_intermediate_size),
        )?;
        if gate_up % 4 != 0 || down % 4 != 0 {
            return Err(invalid("EP dummy expert byte size is not a multiple of 4"));
        }
        Ok((gate_up, down))
    }

    fn new(gpu: &mut Gpu, config: &Qwen4Config) -> Result<Self, Qwen4GpuForwardError> {
        let (gate_up_bytes, down_bytes) = Self::expert_bytes(config)?;
        let gate_up = gpu.zeros(&[gate_up_bytes / 4], DType::F32)?;
        let down = match gpu.zeros(&[down_bytes / 4], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(gate_up);
                return Err(error.into());
            }
        };
        Ok(Self { gate_up, down })
    }

    fn source(&self) -> Qwen4PeerExpertSource {
        Qwen4PeerExpertSource {
            gate_up_base: self.gate_up.buf.as_ptr() as usize,
            gate_up_bytes: self.gate_up.buf.size(),
            down_base: self.down.buf.as_ptr() as usize,
            down_bytes: self.down.buf.size(),
        }
    }

    fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = None;
        for tensor in [self.gate_up, self.down] {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }
}

/// The forward's row gather and gather buffers, lent to the rank's MTP head
/// for one call (buffers `2 + j % 2` carry the head's MoE rows, see
/// [`qwen4_ep_buffer_index`]).
#[derive(Clone, Copy)]
pub(crate) struct MtpEpGather<'a> {
    pub(crate) hook: &'a (dyn MoeRowGather + Send + Sync),
    pub(crate) buffers: [&'a GpuTensor; QWEN4_EP_GATHER_BUFFERS],
}

/// `HIPFIRE_EP_BALANCED_OWNERS` (default on under EP; `=0` disables): per-call
/// balanced expert ownership on the host-mapped trunk MoE layers of an EP rank.
/// Only the exact value `0` is off (unset or any other value is on), and off
/// builds nothing (no tables, no buffer, no stage, no kernel load).
pub const QWEN4_EP_BALANCED_OWNERS_ENV: &str = "HIPFIRE_EP_BALANCED_OWNERS";

fn qwen4_ep_balanced_owners_enabled() -> bool {
    hipfire_config::developer_var(QWEN4_EP_BALANCED_OWNERS_ENV).as_deref() != Ok("0")
}

/// `HIPFIRE_EP_OWNER_ROUTED_PREFILL` (default ON under EP2; `=0` is the kill switch): owner-routed
/// chunk MoE (rank-generic code; EP4 is still refused at admission). A prefill chunk of at least
/// 512 rows whose layer verified symmetric on every rank runs the symmetric IU4 GEMMs on the
/// compact (owned-experts-only) table with the tiles of the other ranks' experts masked out, then
/// pulls each other-rank-owned tile's BF16 grouped down rows over P2P (`MoeTileExchange`) instead
/// of computing every expert on every rank. Every other chunk keeps the full-table route. Only
/// the exact value `0` is off (unset or any other value is on, like `HIPFIRE_EP_BALANCED_OWNERS`);
/// every other admission condition still applies, and off allocates nothing, lowers no stage and
/// loads no kernel.
pub const QWEN4_EP_OWNER_ROUTED_PREFILL_ENV: &str = "HIPFIRE_EP_OWNER_ROUTED_PREFILL";

/// The shared EP lever rule: default on, exact `0` off.
fn ep_lever_on(value: Option<&str>) -> bool {
    value != Some("0")
}

fn qwen4_ep_owner_routed_prefill_enabled() -> bool {
    ep_lever_on(hipfire_config::developer_var(QWEN4_EP_OWNER_ROUTED_PREFILL_ENV).ok().as_deref())
}

/// Rank counts the owner-routed chunk MoE admits: EP2 and EP4, matching the mesh
/// (`attach_ep_mesh`) and the four-source launcher ABI.
const fn tile_ranks_supported(ranks: usize) -> bool {
    matches!(ranks, 2 | 4)
}

/// The `experts` little-endian i32 owners (`expert % ranks`) the launcher reads.
fn tile_owner_table_bytes(experts: usize, ranks: usize) -> Vec<u8> {
    (0..experts)
        .flat_map(|expert| ((expert % ranks) as i32).to_le_bytes())
        .collect()
}

/// Per-rank donor mirror bytes of one G2 stage layer (`[gate/up, down]`): a rank hosts
/// `experts / ranks` experts of each kind.
fn stage_donor_bytes(experts: usize, ranks: usize) -> [usize; 2] {
    let local_experts = experts / ranks;
    [
        local_experts * QWEN4_EXPERT_STAGE_GATE_UP_BYTES,
        local_experts * QWEN4_EXPERT_STAGE_DOWN_BYTES,
    ]
}

/// Redirect the entries of the experts this rank owns (`e % ranks == rank`) into its donor
/// buffer at `base`, slot `e / ranks`; every other entry keeps its pointer.
fn redirect_owned_entries(
    placement: Qwen4EpPlacement,
    base: usize,
    stride: usize,
    entries: &[usize],
) -> Vec<u8> {
    entries.iter().enumerate().flat_map(|(expert, &pointer)| {
        let pointer = if expert % placement.ranks == placement.rank {
            base + (expert / placement.ranks) * stride
        } else {
            pointer
        };
        pointer.to_ne_bytes()
    }).collect()
}

/// One rank's owner-routed chunk MoE state. The buffers exist from forward construction
/// (before the mesh enables peers, so a peer-mapped allocation never lags the mapping);
/// the hook is installed once every rank agreed on the per-layer symmetric verdicts.
pub(crate) struct Qwen4TileRoute {
    /// Owner rank of every expert (`i32` per expert, `e % ranks`): the mask and the pull
    /// read it as an input.
    owner_table: GpuTensor,
    /// The masked tile ids gate/up and down read (`i32` per tile of the padded capacity).
    tile_ids_owned: GpuTensor,
    /// Padded tile capacity the exchange is registered for.
    max_tiles: usize,
    /// Per layer: the compact layer's symmetric verdict, taken on the full tables in
    /// `attach_ep`. Empty until then.
    verdicts: Vec<bool>,
    /// The installed transport (`None` until the mesh agreed).
    hook: Option<Arc<dyn MoeTileExchange + Send + Sync>>,
}

impl Qwen4TileRoute {
    /// Admit the lever for this load or log why not and return `None` (nothing allocated).
    #[allow(clippy::too_many_arguments)]
    fn admit(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        placement: Qwen4EpPlacement,
        scratch: &Qwen4GpuForwardScratch,
        mode: Qwen4ForwardScratchMode,
        max_chunk: usize,
        flash_next: bool,
    ) -> Result<Option<Self>, Qwen4GpuForwardError> {
        let refuse = |reason: &str| {
            eprintln!(
                "qwen4 EP rank {}/{}: owner-routed prefill not admitted ({reason}); chunks keep the full-table route",
                placement.rank, placement.ranks
            );
            Ok(None)
        };
        if !tile_ranks_supported(placement.ranks) {
            return refuse("needs 2 or 4 ranks");
        }
        if placement.ranks == 4 {
            // The transport and the 4-source pull are rank-generic and CPU-tested, but the EP4
            // exactness cells have not run on hardware: EP4 keeps the full-table route.
            return refuse("EP4 owner-routed prefill not yet gated");
        }
        if gpu.arch != "gfx1201" {
            return refuse("needs gfx1201");
        }
        if !flash_next {
            return refuse("needs the Flash-Next profile");
        }
        if !gpu.flags.moe_grouped_gemm || !gpu.qwen4_moe_sym_iu4_requested() {
            return refuse("the symmetric IU4 route is not requested");
        }
        if mode != Qwen4ForwardScratchMode::Full {
            // The compact plan lends `y_down_grouped` to a phase-disjoint field the peer
            // would be reading.
            return refuse("the compact scratch plan aliases y_down_grouped");
        }
        // Sym scratch before the mesh enables peers (the grouped-row buffer and the masked
        // ids are allocated now; ROCm does not map later allocations).
        gpu.reserve_qwen4_moe_sym(
            max_chunk,
            config.num_experts_per_tok,
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
        )?;
        let word = std::mem::size_of::<i32>();
        let tile_bytes = 16 * config.hidden_size * DType::BF16.size();
        let tile_capacity = scratch.moe_expert_tile_ids.buf.size() / word;
        let max_tiles = tile_capacity.min(scratch.moe_y_down_grouped.buf.size() / tile_bytes);
        if max_tiles == 0 {
            return refuse("no tile capacity");
        }
        let owners = tile_owner_table_bytes(config.num_experts, placement.ranks);
        let owner_table = gpu.upload_raw(&owners, &[owners.len()])?;
        let sentinels = vec![0xFFu8; tile_capacity * word];
        let tile_ids_owned = match gpu.upload_raw(&sentinels, &[sentinels.len()]) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(owner_table);
                return Err(error.into());
            }
        };
        Ok(Some(Self {
            owner_table,
            tile_ids_owned,
            max_tiles,
            verdicts: Vec::new(),
            hook: None,
        }))
    }

    fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = None;
        for tensor in [self.owner_table, self.tile_ids_owned] {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }
}

/// One host-mapped trunk MoE layer under balanced ownership: the rank's *live*
/// expert runtime. It is the layer's full (P2P) addressing
/// ([`Qwen4MoeLayerRuntime::from_moe_full`]); `qwen4_ep_owner_assign` rewrites
/// its two pointer tables every call (owned routed experts keep their full
/// address, every other entry names the rank's zero dummy expert), and the
/// routed producers read those tables in place of the static EP decode table.
pub(crate) struct Qwen4BalancedLayer {
    live: Qwen4MoeLayerRuntime,
    /// The layer's label (`layer.layer`, what the dispatch hook is called with).
    label: usize,
    tables: Qwen4EpOwnerTables,
}

/// What the row-gather hook needs of a rank's balanced layers: the owner-assign
/// table addresses per layer label and a non-owning view of the rank's
/// `slot_owner` buffer (owned by [`Qwen4ForwardEp`], freed with it).
pub(crate) struct Qwen4BalancedPlan {
    pub(crate) layers: Vec<(usize, Qwen4EpOwnerTables)>,
    pub(crate) slot_owner: GpuTensor,
}

/// Expert-parallel state of one rank's forward (present iff the bundle's
/// weights carry an EP placement).
struct Qwen4ForwardEp {
    placement: Qwen4EpPlacement,
    dummy: Qwen4EpDummyExpert,
    /// Per-layer full (P2P) tables for chunk forwards; empty until
    /// [`Qwen4GpuForward::attach_ep`].
    moe_full: Vec<Qwen4MoeLayerRuntime>,
    /// Expanded-row exchange buffers, see [`qwen4_ep_buffer_index`].
    buffers: [GpuTensor; QWEN4_EP_GATHER_BUFFERS],
    /// Row gather the one-row and few-row forwards install, once attached.
    gather: Option<Arc<dyn MoeRowGather + Send + Sync>>,
    /// Balanced-ownership state per MoE ordinal (`Some` for every host-mapped
    /// trunk layer); empty when `HIPFIRE_EP_BALANCED_OWNERS=0` at
    /// [`Qwen4GpuForward::attach_ep`].
    balanced: Vec<Option<Qwen4BalancedLayer>>,
    /// The rank's `QWEN4_EP_GATHER_MAX_ROWS * k` i32 slot owners, shared by the
    /// balanced layers (stream-ordered: a call's assignment writes it and the
    /// same call's gather reads it before the next layer's assignment).
    slot_owner: Option<GpuTensor>,
    /// Owner-routed chunk MoE state (`HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`, admitted at
    /// construction); `None` when the lever is off or refused.
    tile: Option<Qwen4TileRoute>,
}

impl Qwen4ForwardEp {
    fn new(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        placement: Qwen4EpPlacement,
    ) -> Result<Self, Qwen4GpuForwardError> {
        validate_ep_placement(placement, config.num_experts)?;
        let elements = QWEN4_EP_GATHER_MAX_ROWS
            .checked_mul(config.num_experts_per_tok)
            .and_then(|slots| slots.checked_mul(config.hidden_size))
            .ok_or_else(|| invalid("EP gather buffer size overflows"))?;
        let dummy = Qwen4EpDummyExpert::new(gpu, config)?;
        let mut buffers = Vec::with_capacity(QWEN4_EP_GATHER_BUFFERS);
        for _ in 0..QWEN4_EP_GATHER_BUFFERS {
            match gpu.zeros(&[elements], DType::F32) {
                Ok(tensor) => buffers.push(tensor),
                Err(error) => {
                    for tensor in buffers {
                        let _ = gpu.free_tensor(tensor);
                    }
                    let _ = dummy.free_gpu(gpu);
                    return Err(error.into());
                }
            }
        }
        let buffers: [GpuTensor; QWEN4_EP_GATHER_BUFFERS] = match buffers.try_into() {
            Ok(buffers) => buffers,
            Err(_) => return Err(invalid("EP gather buffer count")),
        };
        let state = Self {
            placement,
            dummy,
            moe_full: Vec::new(),
            buffers,
            gather: None,
            balanced: Vec::new(),
            slot_owner: None,
            tile: None,
        };
        // The zero fills are stream-ordered; peers read the buffers over P2P
        // on their own streams.
        if let Err(error) = gpu.hip.device_synchronize() {
            let _ = state.free_gpu(gpu);
            return Err(error.into());
        }
        Ok(state)
    }

    fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = None;
        for layer in self.moe_full {
            if let Some(error) = layer.free_gpu(gpu) {
                first.get_or_insert(error);
            }
        }
        for layer in self.balanced.into_iter().flatten() {
            if let Some(error) = layer.live.free_gpu(gpu) {
                first.get_or_insert(error);
            }
        }
        if let Some(route) = self.tile {
            if let Some(error) = route.free_gpu(gpu) {
                first.get_or_insert(error);
            }
        }
        if let Some(tensor) = self.slot_owner {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        for tensor in self.buffers {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        if let Some(error) = self.dummy.free_gpu(gpu) {
            first.get_or_insert(error);
        }
        first
    }
}

impl Qwen4MoeLayerRuntime {
    /// Single-device runtime: the layer's resident tensors hold every expert.
    pub(crate) fn from_moe(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        moe: &MoeWeights,
        layer_label: usize,
        config: &Qwen4Config,
    ) -> Result<Self, Qwen4GpuForwardError> {
        Self::build(
            gpu,
            weights,
            moe,
            layer_label,
            config,
            Qwen4ExpertTableLayout::Single,
            None,
        )
    }

    /// EP decode runtime: the resident expert tensors are this rank's compact
    /// `num_experts / ranks` experts; owned experts name them, every other
    /// expert names `dummy`.
    pub(crate) fn from_moe_ep(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        moe: &MoeWeights,
        layer_label: usize,
        config: &Qwen4Config,
        placement: Qwen4EpPlacement,
        dummy: &Qwen4EpDummyExpert,
    ) -> Result<Self, Qwen4GpuForwardError> {
        Self::build(
            gpu,
            weights,
            moe,
            layer_label,
            config,
            Qwen4ExpertTableLayout::Ep {
                placement,
                dummy: dummy.source(),
            },
            Some(dummy),
        )
    }

    /// EP chunk runtime: owned experts name the local compact tensors, every
    /// other expert names the owning peer's compact tensors
    /// (`peers_for_layer[rank]`).
    pub(crate) fn from_moe_full(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        moe: &MoeWeights,
        layer_label: usize,
        config: &Qwen4Config,
        placement: Qwen4EpPlacement,
        peers_for_layer: &[Qwen4PeerExpertSource],
    ) -> Result<Self, Qwen4GpuForwardError> {
        Self::build(
            gpu,
            weights,
            moe,
            layer_label,
            config,
            Qwen4ExpertTableLayout::Full {
                placement,
                peers: peers_for_layer,
            },
            None,
        )
    }

    /// Every table is one single-owner (rank 0 of 1) dispatch table whose
    /// views mirror the host pointer entries; EP tables differ only in which
    /// allocation an entry names, so dispatch validates and seals them like
    /// the single-device table.
    fn build(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        moe: &MoeWeights,
        layer_label: usize,
        config: &Qwen4Config,
        layout: Qwen4ExpertTableLayout<'_>,
        dummy: Option<&Qwen4EpDummyExpert>,
    ) -> Result<Self, Qwen4GpuForwardError> {
        let router_source = weights.resident(&moe.gate)?;
        let scalar_source = weights.resident(&moe.shared_gate_scalar)?;
        let shared_gate_source = weights.resident(&moe.shared_gate)?;
        let shared_up_source = weights.resident(&moe.shared_up)?;
        let shared_down_source = weights.resident(&moe.shared_down)?;
        let router =
            ProjectionView::from_source(router_source, config.num_experts, config.hidden_size);
        let shared_scalar = ProjectionView::from_source(scalar_source, 1, config.hidden_size);
        let shared_gate = ProjectionView::from_source(
            shared_gate_source,
            config.shared_expert_intermediate_size,
            config.hidden_size,
        );
        let shared_up = ProjectionView::from_source(
            shared_up_source,
            config.shared_expert_intermediate_size,
            config.hidden_size,
        );
        let shared_down = ProjectionView::from_source(
            shared_down_source,
            config.hidden_size,
            config.shared_expert_intermediate_size,
        );

        let gate_up_source = weights.resident(&moe.experts_gate_up)?;
        let down_source = weights.resident(&moe.experts_down)?;
        let gate_up_dtype = gate_up_source.dtype;
        let down_dtype = down_source.dtype;
        let gate_up_rows = 2 * config.moe_intermediate_size;
        let gate_up_row_stride = row_stride(gate_up_dtype, config.hidden_size);
        let down_row_stride = row_stride(down_dtype, config.moe_intermediate_size);
        let gate_up_expert_bytes = checked_bytes(gate_up_rows, gate_up_row_stride)?;
        let down_expert_bytes = checked_bytes(config.hidden_size, down_row_stride)?;
        let source_experts = layout.local_experts(config.num_experts)?;
        let expected_gate_up_bytes = checked_bytes(source_experts, gate_up_expert_bytes)?;
        let expected_down_bytes = checked_bytes(source_experts, down_expert_bytes)?;
        if gate_up_source.buf.size() < expected_gate_up_bytes {
            return Err(invalid(format!(
                "layer {} gate/up packed bytes {} < expected {}",
                layer_label,
                gate_up_source.buf.size(),
                expected_gate_up_bytes
            )));
        }
        if down_source.buf.size() < expected_down_bytes {
            return Err(invalid(format!(
                "layer {} down packed bytes {} < expected {}",
                layer_label,
                down_source.buf.size(),
                expected_down_bytes
            )));
        }
        // EP: owned experts keep this rank's residency. A host-mapped compact
        // tensor is device-visible exactly as in the single route, a non-owned
        // expert names the device dummy (decode) or the owner's compact tensor
        // (chunk). Peer pointers into pinned host memory rely on ROCm's UVA:
        // pinned host allocations are device-visible from every GPU, so a
        // peer's host-mapped compact tensor is read through the same address.
        let local_host_mapped = gate_up_source.buf.is_host_mapped();
        if matches!(layout, Qwen4ExpertTableLayout::Ep { .. })
            && (gate_up_dtype != DType::MQ4G256V2 || down_dtype != DType::MQ4G128V2)
        {
            return Err(invalid(format!(
                "layer {layer_label} EP dummy expert is QT44/QT53 but the experts are {gate_up_dtype:?}/{down_dtype:?}"
            )));
        }

        let (gate_up_source_name, down_source_name) = match &moe.expert_source_layout {
            ExpertSourceLayout::PackedFused { gate_up, down, .. } => {
                (gate_up.as_str(), down.as_str())
            }
            layout => {
                return Err(invalid(format!(
                    "layer {} uses unsupported expert source layout {layout:?}",
                    layer_label
                )));
            }
        };
        let local = Qwen4PeerExpertSource {
            gate_up_base: gate_up_source.buf.as_ptr() as usize,
            gate_up_bytes: gate_up_source.buf.size(),
            down_base: down_source.buf.as_ptr() as usize,
            down_bytes: down_source.buf.size(),
        };
        let table_entries = qwen4_expert_table_entries(
            &layout,
            local,
            config.num_experts,
            gate_up_expert_bytes,
            down_expert_bytes,
        )
        .map_err(|error| match error {
            Qwen4GpuForwardError::Invalid(message) => {
                invalid(format!("layer {layer_label}: {message}"))
            }
            other => other,
        })?;
        // Whole-tensor windows over each peer rank's compact experts; the
        // per-expert views below are cut from them. `qwen4_expert_table_entries`
        // proved every slot lies inside its window.
        let peer_sources: Vec<(GpuTensor, GpuTensor)> = match layout {
            Qwen4ExpertTableLayout::Full { peers, .. } => peers
                .iter()
                .map(|peer| {
                    (
                        foreign_packed_source(peer.gate_up_base, peer.gate_up_bytes, gate_up_dtype),
                        foreign_packed_source(peer.down_base, peer.down_bytes, down_dtype),
                    )
                })
                .collect(),
            _ => Vec::new(),
        };
        let mut experts = Vec::with_capacity(config.num_experts);
        let mut metadata = Vec::with_capacity(config.num_experts);
        for (expert, entry) in table_entries.iter().enumerate() {
            let (gate_up_backing, gate_up_offset, down_backing, down_offset) = match entry.origin {
                Qwen4ExpertOrigin::Local => (
                    gate_up_source,
                    entry.gate_up_ptr - local.gate_up_base,
                    down_source,
                    entry.down_ptr - local.down_base,
                ),
                Qwen4ExpertOrigin::Dummy => {
                    let dummy = dummy.ok_or_else(|| {
                        invalid(format!("layer {layer_label}: EP table without a dummy expert"))
                    })?;
                    (&dummy.gate_up, 0, &dummy.down, 0)
                }
                Qwen4ExpertOrigin::Peer(rank) => {
                    let (peer_gate_up, peer_down) = peer_sources.get(rank).ok_or_else(|| {
                        invalid(format!("layer {layer_label}: peer rank {rank} has no source"))
                    })?;
                    (
                        peer_gate_up,
                        entry.gate_up_ptr - peer_gate_up.buf.as_ptr() as usize,
                        peer_down,
                        entry.down_ptr - peer_down.buf.as_ptr() as usize,
                    )
                }
            };
            let gate_up = ProjectionView::from_packed(
                gate_up_backing,
                gate_up_offset,
                gate_up_expert_bytes,
                gate_up_dtype,
                gate_up_rows,
                config.hidden_size,
            );
            let down = ProjectionView::from_packed(
                down_backing,
                down_offset,
                down_expert_bytes,
                down_dtype,
                config.hidden_size,
                config.moe_intermediate_size,
            );
            if gate_up.tensor.buf.as_ptr() as usize != entry.gate_up_ptr
                || down.tensor.buf.as_ptr() as usize != entry.down_ptr
            {
                return Err(invalid(format!(
                    "layer {layer_label} expert {expert} view disagrees with its table entry"
                )));
            }
            experts.push(Qwen4ExpertView { gate_up, down });

            let gate_up_resource = ExpertResource::new(
                gate_up_source_name,
                "qwen4-resident-mqv2-v1",
                vec![gate_up_rows, config.hidden_size],
                gate_up_dtype,
                gate_up_expert_bytes,
                gate_up_row_stride,
                16,
                dtype_rotation_plan(gate_up_dtype),
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("expert resource: {error:?}"))
            })?;
            let down_resource = ExpertResource::new(
                down_source_name,
                "qwen4-resident-mqv2-v1",
                vec![config.hidden_size, config.moe_intermediate_size],
                down_dtype,
                down_expert_bytes,
                down_row_stride,
                16,
                dtype_rotation_plan(down_dtype),
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("expert resource: {error:?}"))
            })?;
            let resources =
                ExpertResources::fused(gate_up_resource, down_resource).map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!("expert resources: {error:?}"))
                })?;
            metadata.push(
                ExpertMetadata::new(expert, 0, expert, resources).map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!("expert metadata: {error:?}"))
                })?,
            );
        }
        let table = ExpertTable::new(metadata)
            .map_err(|error| Qwen4GpuForwardError::Dispatch(format!("expert table: {error:?}")))?;
        let mut cache = table
            .prepare_binding(0, 1, gpu.device_id)
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("expert binding: {error:?}"))
            })?;

        let gate_entries = experts
            .iter()
            .map(|expert| expert.gate_up.tensor.buf.as_ptr() as usize)
            .collect::<Vec<_>>();
        let down_entries = experts
            .iter()
            .map(|expert| expert.down.tensor.buf.as_ptr() as usize)
            .collect::<Vec<_>>();
        let gate_ptrs = gate_entries
            .iter()
            .flat_map(|entry| entry.to_ne_bytes())
            .collect::<Vec<_>>();
        let down_ptrs = down_entries
            .iter()
            .flat_map(|entry| entry.to_ne_bytes())
            .collect::<Vec<_>>();
        let expert_gate_up_ptrs = match gpu.alloc_tensor(&[2 * config.num_experts], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => return Err(error.into()),
        };
        if let Err(error) = gpu.memcpy_htod_auto(&expert_gate_up_ptrs.buf, &gate_ptrs) {
            let _ = gpu.free_tensor(expert_gate_up_ptrs);
            return Err(error.into());
        }
        let expert_down_ptrs = match gpu.alloc_tensor(&[2 * config.num_experts], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(expert_gate_up_ptrs);
                return Err(error.into());
            }
        };
        if let Err(error) = gpu.memcpy_htod_auto(&expert_down_ptrs.buf, &down_ptrs) {
            let _ = gpu.free_tensor(expert_gate_up_ptrs);
            let _ = gpu.free_tensor(expert_down_ptrs);
            return Err(error.into());
        }
        let identity = NEXT_EXPERT_SET_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let expert_views = ExpertViewSet {
            experts: &experts,
            identity,
            host_mapped: local_host_mapped,
        };
        if let Err(error) = cache.bind_live(
            &table,
            &expert_views,
            &expert_gate_up_ptrs,
            &expert_down_ptrs,
            None,
            None,
        ) {
            let _ = gpu.free_tensor(expert_gate_up_ptrs);
            let _ = gpu.free_tensor(expert_down_ptrs);
            return Err(Qwen4GpuForwardError::Dispatch(format!(
                "expert live binding: {error:?}"
            )));
        }
        Ok(Self {
            table,
            cache,
            router,
            shared_scalar,
            shared_gate,
            shared_up,
            shared_down,
            experts_all_gate_up_mq4: gate_up_dtype == DType::MQ4G256V2,
            experts,
            identity,
            expert_gate_up_ptrs,
            expert_down_ptrs,
            expert_gate_up_entries: gate_entries,
            expert_down_entries: down_entries,
            symmetric: false,
            host_mapped: local_host_mapped,
        })
    }

    /// Check every routed expert header of this layer for the symmetric grid
    /// the IU4 route needs and record the verdict. QT44 gate/up and QT53
    /// down only; one asymmetric or non-finite header refuses the layer.
    fn verify_symmetric(
        &mut self,
        gpu: &mut Gpu,
        config: &Qwen4Config,
    ) -> Result<bool, Qwen4GpuForwardError> {
        let formats = self.experts.iter().all(|expert| {
            expert.gate_up.dtype == DType::MQ4G256V2 && expert.down.dtype == DType::MQ4G128V2
        });
        self.symmetric = formats
            && gpu.qwen4_moe_sym_check(
                &self.expert_gate_up_ptrs,
                2 * config.moe_intermediate_size,
                config.hidden_size,
                config.num_experts,
                136,
            )?
            && gpu.qwen4_moe_sym_check(
                &self.expert_down_ptrs,
                config.hidden_size,
                config.moe_intermediate_size,
                config.num_experts,
                68,
            )?;
        Ok(self.symmetric)
    }

    /// Device bytes [`Self::from_moe`] allocates: the gate/up and down
    /// pointer tables, one device pointer per expert each.
    pub(crate) fn device_bytes(config: &Qwen4Config) -> Option<usize> {
        config
            .num_experts
            .checked_mul(2 * std::mem::size_of::<u64>())
    }

    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = None;
        for tensor in [self.expert_gate_up_ptrs, self.expert_down_ptrs] {
            if let Err(error) = gpu.free_tensor(tensor) {
                if first.is_none() {
                    first = Some(error);
                }
            }
        }
        first
    }
}

/// What a donor layer's VRAM holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DonorState {
    /// Its own experts, ordered for every later compute-stream launch.
    Resident,
    /// A host layer's experts are (or may be partly) copied over it.
    Overwritten,
    /// Its restore is issued on the copy stream and recorded on `ready`, but
    /// the compute stream has not yet waited on it.
    Restoring,
}

/// Long-prefill host-expert DMA pipeline (G2) on borrowed VRAM. The last
/// resident expert layer of each parity is that parity's stage: once its own
/// MoE has run, the host layers of the same parity are copied over it in
/// turn (L+2 refill), and after the last of them its bytes are restored
/// from a pinned host copy. Each fork and join is a recorded event, and
/// the forward makes the compute stream wait on the restores before it
/// returns. Any later launch on that stream (the active stream is installed
/// once, by `ensure_capture_stream`, and never replaced) therefore sees the
/// donor layers whole. Routing-independent: neither the copies nor the
/// pointer tables depend on the experts a row picks.
pub(crate) struct Qwen4ExpertStage {
    /// Forwards of at least this many rows stage.
    min_rows: usize,
    /// Long forwards with a wide-hidden capture stage too
    /// ([`QWEN4_EXPERT_STAGE_CAPTURE_ENV`]).
    capture: bool,
    /// Donor trunk layer per parity.
    donors: [usize; 2],
    /// Borrowed device views of each donor's gate/up and down experts,
    /// parity-major. Never freed here: the bundle's weights own them.
    donor_views: Vec<GpuTensor>,
    /// Pinned host copies of the same bytes: the restore source.
    mirrors: Vec<GpuTensor>,
    /// Static per-expert pointer tables into each donor's gate/up and down
    /// VRAM, parity-major. They never replace a layer's bound mapped tables.
    tables: Vec<GpuTensor>,
    /// EP tables preserve non-owned pointers from the runtime being bound.
    compact_tables: Vec<GpuTensor>,
    compact_keys: Vec<usize>,
    full_keys: Vec<std::cell::OnceCell<usize>>,
    compact_bound: Vec<std::cell::Cell<bool>>,
    first_host: usize,
    placement: Option<Qwen4EpPlacement>,
    bytes: [usize; 2],
    copy_stream: hip_bridge::Stream,
    ready: Vec<hip_bridge::Event>,
    free: Vec<hip_bridge::Event>,
    /// Sticky until the donors are back: a failed forward leaves it set,
    /// and every later forward restores first ([`Self::restore_blocking`]).
    state: std::cell::Cell<[DonorState; 2]>,
}

impl Qwen4ExpertStage {
    const EP_ENV: &'static str = "HIPFIRE_EP_EXPERT_STAGE";

    /// Per-forward admission: full EP (EP2/EP4) chunk tables read other ranks' donor VRAM
    /// and must never stage. S1 supplies true only when every trunk MoE layer
    /// binds the compact owner-routed route. Single-card admission is unchanged.
    ///
    /// Entering a compact staged chunk after a full chunk is safe: the donor
    /// overwrite waits on local `free`, recorded after this chunk's local
    /// waits on peer ready; peer ready follows the previous forward in the
    /// peer compute stream, so its earlier donor reads have completed.
    fn begin_forward(&self, ep_compact_all_layers: bool) -> bool {
        self.placement.is_none() || ep_compact_all_layers
    }

    /// Before any full-table EP chunk, S1 calls this on EVERY rank, then
    /// crosses its host barrier before ANY rank launches that chunk. `join`
    /// marks donors Resident after enqueueing restore waits, not host completion,
    /// so even Resident donors must be drained before peers can read them.
    fn ep_quiesce(&self, gpu: &Gpu) -> Result<(), Qwen4GpuForwardError> {
        if gpu.active_stream.as_ref().map(|stream| gpu.hip.stream_is_capturing(stream)).transpose()?.unwrap_or(false) {
            return Err(invalid("G2 EP quiescence cannot run during capture"));
        }
        gpu.hip.device_synchronize()?;
        self.restore_blocking(gpu)
    }

    fn new(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        moe: &[Qwen4MoeLayerRuntime],
    ) -> Result<Option<Self>, Qwen4GpuForwardError> {
        if !hipfire_config::developer_bool(QWEN4_EXPERT_STAGE_ENV, gpu.arch == "gfx1201") {
            return Ok(None);
        }
        let Some(first_host) = moe.iter().position(RoutedExpertWeights::host_mapped) else {
            return Ok(None);
        };
        let config = &bundle.config;
        // G2's stage layout is the frozen Flash-Next QT44/QT53 geometry.
        // Other quantized artifacts keep their existing mapped path.
        if config.num_experts != 512
            || config.hidden_size != 2560
            || config.moe_intermediate_size != 640
            || moe[first_host.saturating_sub(2)..].iter().any(|layer| {
                layer.experts.iter().any(|expert| {
                    expert.gate_up.tensor.dtype != DType::MQ4G256V2
                        || expert.down.tensor.dtype != DType::MQ4G128V2
                })
            })
        {
            return Ok(None);
        }
        let off = |reason: &str| {
            eprintln!("  qwen4 long-prefill expert staging: off ({reason})");
            Ok(None)
        };
        // Resident layers are a prefix (`place_routed_experts`), so each
        // donor's own MoE runs before every host layer of its parity.
        if first_host < 2 || !moe[first_host..].iter().all(RoutedExpertWeights::host_mapped) {
            return off("needs two resident expert layers ahead of every host layer");
        }
        let mut donors = [first_host - 2, first_host - 1];
        donors.sort_by_key(|layer| layer % 2);
        let placement = bundle.weights.ep_placement();
        let ranks = placement.map_or(1, |placement| placement.ranks);
        let bytes = stage_donor_bytes(config.num_experts, ranks);
        let layer_bytes = bytes.iter().sum::<usize>();
        let mirror_bytes = crate::expert_residency::expert_stage_mirror_bytes(layer_bytes, ranks)
            .map_err(invalid)?;
        let admitted = crate::expert_residency::check_host_ram(
            mirror_bytes,
            rdna_compute::kv_slots::mem_available_bytes(),
            crate::expert_residency::ttm_pool_estimate(),
        )
        .and_then(|()| {
            crate::expert_residency::check_gtt_cap(
                mirror_bytes,
                crate::expert_residency::gtt_budget(),
            )
        });
        if let Err(reason) = admitted {
            return off(&format!("restore copies: {reason}"));
        }
        let parts =
            |layer: usize| -> Result<[(&TensorRef, usize, usize); 2], Qwen4GpuForwardError> {
                let weights = bundle.weights.layer_refs[layer]
                    .ffn
                    .moe()
                    .ok_or_else(|| invalid("G2 stage layer has no MoE FFN"))?;
                Ok([
                    (
                        &weights.experts_gate_up,
                        bytes[0],
                        QWEN4_EXPERT_STAGE_GATE_UP_BYTES,
                    ),
                    (
                        &weights.experts_down,
                        bytes[1],
                        QWEN4_EXPERT_STAGE_DOWN_BYTES,
                    ),
                ])
            };
        for layer in first_host..moe.len() {
            for (source, bytes, _) in parts(layer)? {
                let owner = bundle.weights.resident(source)?;
                // Host addresses come only from owning tensors. Borrowed
                // expert views are never re-tagged as HostMapped (G1).
                if gpu.host_bytes(owner).is_none_or(|host| host.len() < bytes) {
                    return off("a host expert owner has no pinned host bytes of the stage size");
                }
            }
        }
        for layer in donors {
            for (source, bytes, _) in parts(layer)? {
                let owner = bundle.weights.resident(source)?;
                if owner.buf.is_host_mapped() || owner.buf.size() < bytes {
                    return off("a donor expert owner is not device memory of the stage size");
                }
            }
        }
        gpu.ensure_capture_stream()?;
        let mut stage = Self {
            min_rows: qwen4_expert_stage_min_rows(),
            capture: hipfire_config::developer_bool(QWEN4_EXPERT_STAGE_CAPTURE_ENV, true),
            donors,
            donor_views: Vec::with_capacity(4),
            mirrors: Vec::with_capacity(4),
            tables: Vec::with_capacity(4),
            compact_tables: Vec::new(),
            compact_keys: if placement.is_some() {
                moe.iter().map(|runtime| runtime.expert_gate_up_ptrs.buf.as_ptr() as usize).collect()
            } else { Vec::new() },
            full_keys: if placement.is_some() {
                (0..moe.len()).map(|_| std::cell::OnceCell::new()).collect()
            } else { Vec::new() },
            compact_bound: if placement.is_some() {
                (0..moe.len()).map(|_| std::cell::Cell::new(false)).collect()
            } else { Vec::new() },
            first_host,
            placement,
            bytes,
            copy_stream: gpu.hip.stream_create_non_blocking()?,
            ready: Vec::with_capacity(2),
            free: Vec::with_capacity(2),
            state: std::cell::Cell::new([DonorState::Resident; 2]),
        };
        let result = (|| {
            for _ in 0..2 {
                stage.ready.push(
                    gpu.hip
                        .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
                );
                stage.free.push(
                    gpu.hip
                        .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
                );
            }
            for layer in donors {
                for (source, bytes, stride) in parts(layer)? {
                    let owner = bundle.weights.resident(source)?;
                    let view = owner.sub_offset(0, bytes);
                    // The mirror reads the loaded bytes back once, at load.
                    let mut host = vec![0u8; bytes];
                    gpu.memcpy_dtoh_auto(&mut host, &view.buf)?;
                    stage.mirrors.push(gpu.upload_raw_host_mapped(&host, &[bytes])?);
                    drop(host);
                    let base = view.buf.as_ptr() as usize;
                    if placement.is_none() {
                        let entries = (0..config.num_experts)
                            .flat_map(|expert| (base + expert * stride).to_ne_bytes())
                            .collect::<Vec<_>>();
                        stage.tables.push(gpu.alloc_tensor(&[2 * config.num_experts], DType::F32)?);
                        gpu.memcpy_htod_auto(
                            &stage.tables.last().expect("table just allocated").buf,
                            &entries,
                        )?;
                    }
                    stage.donor_views.push(view);
                }
            }
            if placement.is_some() {
                for (layer, runtime) in moe.iter().enumerate().skip(first_host) {
                    for (kind, entries) in [&runtime.expert_gate_up_entries, &runtime.expert_down_entries].into_iter().enumerate() {
                        let redirected = stage.redirect_entries(layer, kind, entries);
                        stage.compact_tables.push(gpu.alloc_tensor(&[2 * config.num_experts], DType::F32)?);
                        gpu.memcpy_htod_auto(&stage.compact_tables.last().expect("allocated").buf, &redirected)?;
                        stage.tables.push(gpu.alloc_tensor(&[2 * config.num_experts], DType::F32)?);
                    }
                }
            }
            Ok::<(), Qwen4GpuForwardError>(())
        })();
        if let Err(error) = result {
            let _ = stage.free_gpu(gpu);
            return Err(error);
        }
        eprintln!(
            "  qwen4 long-prefill expert staging: forwards of >= {} rows stage host expert layers {first_host}..{} through resident layers {} and {} ({} MiB pinned restore copies; wide-hidden capture forwards {})",
            stage.min_rows,
            moe.len(),
            donors[0],
            donors[1],
            2 * layer_bytes >> 20,
            if stage.capture { "stage" } else { "mapped" }
        );
        Ok(Some(stage))
    }

    /// The stage tables a host `layer` reads from on a staged forward.
    fn tables(&self, layer: usize) -> MoeStageTables<'_> {
        let parity = layer % 2;
        if self.placement.is_some() && self.compact_bound[layer].get() {
            return self.tables_compact(layer);
        }
        let index = if self.placement.is_some() { layer - self.first_host } else { parity };
        MoeStageTables {
            gate_up: &self.tables[2 * index],
            down: &self.tables[2 * index + 1],
            free: &self.free[parity],
        }
    }

    /// S1 binds this accessor when its compact owner-routed runtime is used.
    fn tables_compact(&self, layer: usize) -> MoeStageTables<'_> {
        if self.placement.is_none() {
            return self.tables(layer);
        }
        let index = 2 * (layer - self.first_host);
        MoeStageTables {
            gate_up: &self.compact_tables[index],
            down: &self.compact_tables[index + 1],
            free: &self.free[layer % 2],
        }
    }

    fn redirect_entries(&self, layer: usize, kind: usize, entries: &[usize]) -> Vec<u8> {
        let placement = self.placement.expect("EP tables");
        let base = self.donor_views[2 * (layer % 2) + kind].buf.as_ptr() as usize;
        let stride = [QWEN4_EXPERT_STAGE_GATE_UP_BYTES, QWEN4_EXPERT_STAGE_DOWN_BYTES][kind];
        redirect_owned_entries(placement, base, stride, entries)
    }

    /// Initialize a full EP table once, before any capture/tape. Compact
    /// tables are built at load and remain separate from peer-mapped tables.
    fn prepare_runtime(
        &self,
        gpu: &Gpu,
        layer: usize,
        runtime: Option<&Qwen4MoeLayerRuntime>,
    ) -> Result<(), Qwen4GpuForwardError> {
        if self.placement.is_none() || layer < self.first_host {
            return Ok(());
        }
        let runtime = runtime.ok_or_else(|| invalid("G2 staged EP runtime missing"))?;
        let key = runtime.expert_gate_up_ptrs.buf.as_ptr() as usize;
        let compact = key == self.compact_keys[layer];
        self.compact_bound[layer].set(compact);
        if compact {
            return Ok(());
        }
        if let Some(&prepared) = self.full_keys[layer].get() {
            return if prepared == key { Ok(()) } else { Err(invalid("G2 full runtime changed")) };
        }
        if gpu.active_stream.as_ref().map(|stream| gpu.hip.stream_is_capturing(stream)).transpose()?.unwrap_or(false) {
            return Err(invalid("G2 EP stage table initialization cannot run during capture"));
        }
        for (kind, entries) in [&runtime.expert_gate_up_entries, &runtime.expert_down_entries].into_iter().enumerate() {
            let redirected = self.redirect_entries(layer, kind, entries);
            gpu.memcpy_htod_auto(&self.tables[2 * (layer - self.first_host) + kind].buf, &redirected)?;
        }
        self.full_keys[layer].set(key).map_err(|_| invalid("G2 full runtime initialized twice"))?;
        Ok(())
    }

    fn set_state(&self, parity: usize, value: DonorState) {
        let mut state = self.state.get();
        state[parity] = value;
        self.state.set(state);
    }

    /// Copy host `layer`'s experts over its parity's donor once the stage is
    /// free, recording `ready`.
    fn enqueue_layer(
        &self,
        gpu: &Gpu,
        bundle: &Qwen4Bundle,
        layer: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        let parity = layer % 2;
        let weights = bundle.weights.layer_refs[layer]
            .ffn
            .moe()
            .ok_or_else(|| invalid("G2 stage layer has no MoE FFN"))?;
        let gate = gpu
            .host_bytes(bundle.weights.resident(&weights.experts_gate_up)?)
            .ok_or_else(|| invalid("G2 gate/up pinned owner missing"))?;
        let down = gpu
            .host_bytes(bundle.weights.resident(&weights.experts_down)?)
            .ok_or_else(|| invalid("G2 down pinned owner missing"))?;
        gpu.hip
            .stream_wait_event(&self.copy_stream, &self.free[parity])?;
        // From the first byte issued, the donor no longer holds its experts.
        self.set_state(parity, DonorState::Overwritten);
        self.copy_into_donor(gpu, parity, [gate, down])
    }

    /// Copy the donor's own bytes back once the last host layer of its
    /// parity is done with the stage, recording `ready`.
    fn enqueue_restore(&self, gpu: &Gpu, parity: usize) -> Result<(), Qwen4GpuForwardError> {
        let [gate, down] = self.mirror_bytes(gpu, parity)?;
        gpu.hip
            .stream_wait_event(&self.copy_stream, &self.free[parity])?;
        self.copy_into_donor(gpu, parity, [gate, down])?;
        self.set_state(parity, DonorState::Restoring);
        Ok(())
    }

    fn copy_into_donor(
        &self,
        gpu: &Gpu,
        parity: usize,
        sources: [&[u8]; 2],
    ) -> Result<(), Qwen4GpuForwardError> {
        for (kind, (source, bytes)) in sources
            .into_iter()
            .zip(self.bytes)
            .enumerate()
        {
            gpu.hip.memcpy_htod_async(
                &self.donor_views[2 * parity + kind].buf,
                &source[..bytes],
                &self.copy_stream,
            )?;
        }
        gpu.hip
            .event_record(&self.ready[parity], Some(&self.copy_stream))?;
        Ok(())
    }

    fn mirror_bytes<'a>(
        &'a self,
        gpu: &Gpu,
        parity: usize,
    ) -> Result<[&'a [u8]; 2], Qwen4GpuForwardError> {
        let host = |kind: usize| {
            gpu.host_bytes(&self.mirrors[2 * parity + kind])
                .ok_or_else(|| invalid("G2 donor restore copy missing"))
        };
        Ok([host(0)?, host(1)?])
    }

    /// After `layer`'s MoE is issued on a staged forward: a donor frees its
    /// stage, and a donor or host layer refills its parity for the next host
    /// layer L+2.., or, past the last one, restores the donor.
    fn after_moe(
        &self,
        gpu: &Gpu,
        bundle: &Qwen4Bundle,
        moe: &[Qwen4MoeLayerRuntime],
        layer: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        let parity = layer % 2;
        let donor = self.donors[parity] == layer;
        if !donor && !moe[layer].host_mapped() {
            return Ok(());
        }
        if donor {
            // A host layer's routed down records `free` itself.
            let compute = gpu
                .active_stream
                .as_ref()
                .ok_or_else(|| invalid("G2 compute stream missing"))?;
            gpu.hip.event_record(&self.free[parity], Some(compute))?;
        }
        match (layer + 2..moe.len())
            .step_by(2)
            .find(|&next| moe[next].host_mapped())
        {
            Some(next) => self.enqueue_layer(gpu, bundle, next),
            None if self.state.get()[parity] == DonorState::Overwritten => {
                self.enqueue_restore(gpu, parity)
            }
            None => Ok(()),
        }
    }

    /// End of a staged forward: the compute stream waits on every restore,
    /// charging what the trunk did not hide to this forward.
    fn join(&self, gpu: &Gpu) -> Result<(), Qwen4GpuForwardError> {
        for parity in 0..2 {
            if self.state.get()[parity] == DonorState::Overwritten {
                self.enqueue_restore(gpu, parity)?;
            }
            if self.state.get()[parity] == DonorState::Restoring {
                let compute = gpu
                    .active_stream
                    .as_ref()
                    .ok_or_else(|| invalid("G2 compute stream missing"))?;
                gpu.hip.stream_wait_event(compute, &self.ready[parity])?;
                self.set_state(parity, DonorState::Resident);
            }
        }
        Ok(())
    }

    /// Recovery when a staged forward did not reach [`Self::join`]: drain the
    /// device, then copy every donor not known resident back synchronously.
    /// The state stays set on failure, so the next forward retries before
    /// anything reads the donors.
    fn restore_blocking(&self, gpu: &Gpu) -> Result<(), Qwen4GpuForwardError> {
        if self.state.get() == [DonorState::Resident; 2] {
            return Ok(());
        }
        if let Some(stream) = gpu.active_stream.as_ref() {
            if gpu.hip.stream_is_capturing(stream)? {
                return Err(invalid(
                    "G2 donor expert layers need a restore, which a stream capture cannot run",
                ));
            }
        }
        gpu.hip.device_synchronize()?;
        for parity in 0..2 {
            if self.state.get()[parity] == DonorState::Resident {
                continue;
            }
            for (kind, source) in self.mirror_bytes(gpu, parity)?.into_iter().enumerate() {
                gpu.hip
                    .memcpy_htod(&self.donor_views[2 * parity + kind].buf, source)?;
            }
            gpu.hip.device_synchronize()?;
            self.set_state(parity, DonorState::Resident);
        }
        Ok(())
    }

    fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = gpu.hip.stream_synchronize(&self.copy_stream).err();
        if let Err(error) = gpu.hip.stream_destroy(self.copy_stream) {
            first.get_or_insert(error);
        }
        for event in self.ready.into_iter().chain(self.free) {
            if let Err(error) = gpu.hip.event_destroy(event) {
                first.get_or_insert(error);
            }
        }
        for tensor in self.tables.into_iter().chain(self.compact_tables).chain(self.mirrors) {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }
}

/// The G2 stage's host work around each MoE call of a staged forward, run by
/// the step interpreter (so the whole layer program stays one interpreter
/// call): a host layer's compute waits for its staged experts, and after the
/// call the stage is freed / refilled ([`Qwen4ExpertStage::after_moe`]).
struct StagedMoeHooks<'s> {
    stage: &'s Qwen4ExpertStage,
    bundle: &'s Qwen4Bundle,
    moe: &'s [Qwen4MoeLayerRuntime],
    /// The layer of the next MoE call.
    layer: usize,
    /// The stage error behind a hook failure, returned instead of the
    /// interpreter's stringified one.
    error: Option<Qwen4GpuForwardError>,
}

impl StagedMoeHooks<'_> {
    fn fail(&mut self, result: Result<(), Qwen4GpuForwardError>) -> Result<(), DispatchError> {
        result.map_err(|error| {
            let message = error.to_string();
            self.error = Some(error);
            DispatchError::Hip(message)
        })
    }
}

impl MoeStepHooks for StagedMoeHooks<'_> {
    fn before_moe(&mut self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let result = match self.moe.get(self.layer) {
            None => Err(invalid("G2 staged forward ran more MoE calls than layers")),
            Some(moe) if !moe.host_mapped() => Ok(()),
            Some(_) => match gpu.active_stream.as_ref() {
                None => Err(invalid("G2 compute stream missing")),
                Some(compute) => gpu
                    .hip
                    .stream_wait_event(compute, &self.stage.ready[self.layer % 2])
                    .map_err(Qwen4GpuForwardError::from),
            },
        };
        self.fail(result)
    }

    fn after_moe(&mut self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        // Down recorded free before combine. Refill this parity for L+2 or
        // restore its donor; no routing results are involved.
        let result = self.stage.after_moe(gpu, self.bundle, self.moe, self.layer);
        self.layer += 1;
        self.fail(result)
    }
}

/// Scratch views required by one sealed Qwen4 MoE invocation.  The caller owns
/// these reusable buffers; the sealed expert table and live pointer resources
/// remain in [`Qwen4MoeLayerRuntime`].
pub(crate) struct Qwen4MoeScratch<'a> {
    pub(crate) router_logits: &'a GpuTensor,
    pub(crate) scalar_buf: &'a GpuTensor,
    pub(crate) x_rot_local: &'a GpuTensor,
    pub(crate) gate_up_buf: &'a GpuTensor,
    pub(crate) gate_buf: &'a GpuTensor,
    pub(crate) up_buf: &'a GpuTensor,
    pub(crate) ffn_hidden: &'a GpuTensor,
    pub(crate) ffn_out: &'a GpuTensor,
    pub(crate) gate_batch: &'a GpuTensor,
    pub(crate) up_batch: &'a GpuTensor,
    pub(crate) rot_batch: &'a GpuTensor,
    pub(crate) topk_indices: &'a GpuTensor,
    pub(crate) topk_weights: &'a GpuTensor,
    pub(crate) down_expanded: &'a GpuTensor,
}

/// Seal the one-row decode MoE of `runtime` for a caller's step list; the
/// caller clears `output` with a preceding `Step::Clear`.
pub(crate) fn seal_moe_decode<'a>(
    ctx: &'a DispatchCtx,
    config: &Qwen4Config,
    layer_index: usize,
    runtime: &'a Qwen4MoeLayerRuntime,
    input: &'a GpuTensor,
    output: &'a GpuTensor,
    scratch: Qwen4MoeScratch<'a>,
    row_gather: Option<&'a dyn MoeRowGather>,
) -> Result<SealedMoeCall<'a>, Qwen4GpuForwardError> {
    let dtypes = MoeDtypes {
        router: runtime.router.dtype,
        shared: Some(MoeSharedDtypes {
            selector: runtime.shared_scalar.dtype,
            gate: runtime.shared_gate.dtype,
            up: runtime.shared_up.dtype,
            down: runtime.shared_down.dtype,
        }),
        experts_all_gate_up_mq4: runtime
            .experts
            .iter()
            .all(|expert| expert.gate_up.dtype == DType::MQ4G256V2),
        routed_gate_up: runtime
            .experts
            .first()
            .map(|expert| expert.gate_up.dtype)
            .ok_or_else(|| invalid("MoE expert table is empty"))?,
        routed_down: runtime
            .experts
            .first()
            .map(|expert| expert.down.dtype)
            .ok_or_else(|| invalid("MoE expert table is empty"))?,
        routed_has_mixed_experts: false,
        has_paro_shared: false,
        per_expert_gate_up: None,
        per_expert_down: None,
    };
    let shared = MoeSharedWeights {
        selector: runtime.shared_scalar.dispatch_ref(),
        gate: runtime.shared_gate.dispatch_ref(),
        up: runtime.shared_up.dispatch_ref(),
        down: runtime.shared_down.dispatch_ref(),
    };
    let params = MoeParams {
        dtypes,
        recipe: MoeRecipe::SoftmaxGatedShared {
            bf16_round_trip: true,
            shared_after_combine: true,
        },
        // Single-token decode/MTP binding: never the symmetric IU4 prefill arm.
        route_policy: Some(qwen4_route_policy(config, &shared, false)),
        normalization: MoeNormalization::Provided,
        batch_size: 1,
        hidden: config.hidden_size,
        mi: config.moe_intermediate_size,
        k: config.num_experts_per_tok,
        n_exp: config.num_experts,
        norm_topk_prob: config.norm_topk_prob,
        selection_bias: None,
        x_rot_prerotated: false,
        defer_routed_combine: false,
        cpu_deferred_experts: false,
        ep_mode: MoeEpMode::None,
        layer_idx: layer_index as u16,
        x_norm: input,
        x_residual: output,
        routed_out: None,
        skip_shared: false,
        router: runtime.router.dispatch_ref(),
        shared: Some(MoeSharedDecode {
            weights: shared,
            intermediate: config.shared_expert_intermediate_size,
            scalar: scratch.scalar_buf,
            gate_out: scratch.gate_buf,
            up_out: scratch.up_buf,
        }),
        expert_gate_up_ptrs: &runtime.expert_gate_up_ptrs,
        expert_down_ptrs: &runtime.expert_down_ptrs,
        expert_ptrs_host: None,
        expert_down_awq_ptrs: None,
        expert_dtype_tags: None,
        routed_gate_up_k: config.hidden_size,
        routed_down_m: config.hidden_size,
        routed_down_k: config.moe_intermediate_size,
        routed_experts: runtime,
        routed_gate_up_paro: None,
        routed_down_paro: None,
        router_logits: scratch.router_logits,
        x_rot_local: scratch.x_rot_local,
        gate_up_buf: scratch.gate_up_buf,
        ffn_hidden: scratch.ffn_hidden,
        ffn_out: scratch.ffn_out,
        gate_batch: scratch.gate_batch,
        up_batch: scratch.up_batch,
        rot_batch: scratch.rot_batch,
        topk_indices: scratch.topk_indices,
        topk_weights: scratch.topk_weights,
        down_expanded: scratch.down_expanded,
        ep_row_gather: row_gather,
    };
    let bound = BoundMoeExperts::from_cache(&runtime.table, &runtime.cache)
        .map_err(|error| Qwen4GpuForwardError::Dispatch(format!("bound MoE experts: {error:?}")))?;
    let seal_started = qwen4_profile_start();
    let sealed_result = seal_decode(bound, ctx, params);
    qwen4_profile_record(Qwen4ProfilePhase::MoeSeal, seal_started);
    sealed_result
        .map_err(|error| Qwen4GpuForwardError::Dispatch(format!("seal Qwen4 MoE: {error:?}")))
}

/// Every device field of [`Qwen4GpuForwardScratch`], in allocation order.
/// The order is the plan's index: `plan` refuses to push a slot out of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScratchField {
    TokenIds,
    EmbeddingRot,
    Embeddings,
    Streams,
    HcNormalized,
    HcLow,
    HcUp,
    HcMixed,
    HcGates,
    Rotation,
    Projection,
    Projection2,
    GdnA,
    GdnB,
    GdnGate,
    GdnBeta,
    GdnRecurrentOutput,
    GdnBf16,
    GdnZ,
    GdnOutput,
    QsaIndex,
    QsaQgate,
    QsaK,
    QsaV,
    QsaOutput,
    QsaSelected,
    AttentionOutput,
    PleStaged,
    PleRows,
    PleKey,
    PleValue,
    PleQuery,
    PleGated,
    PleNormed,
    PleOutput,
    RouterLogits,
    MoeXRot,
    MoeGateUp,
    MoeGate,
    MoeUp,
    MoeHidden,
    FfnOutput,
    MoeSharedOutput,
    MoeGateBatch,
    MoeUpBatch,
    MoeRotBatch,
    MoeTopkIndices,
    MoeTopkWeights,
    MoeDownExpanded,
    MoeScalar,
    MoeExpertTokenCounts,
    MoeExpertOffsets,
    MoeSortedSlotIndex,
    MoeExpertTileIds,
    MoeInversePerm,
    MoeYGateUpGrouped,
    MoeYDownGrouped,
    Logits,
}

impl ScratchField {
    pub(crate) const COUNT: usize = Self::Logits as usize + 1;
    const ALL: [Self; Self::COUNT] = [
        Self::TokenIds, Self::EmbeddingRot, Self::Embeddings, Self::Streams,
        Self::HcNormalized, Self::HcLow, Self::HcUp, Self::HcMixed, Self::HcGates,
        Self::Rotation, Self::Projection, Self::Projection2,
        Self::GdnA, Self::GdnB, Self::GdnGate, Self::GdnBeta,
        Self::GdnRecurrentOutput, Self::GdnBf16, Self::GdnZ, Self::GdnOutput,
        Self::QsaIndex, Self::QsaQgate, Self::QsaK, Self::QsaV, Self::QsaOutput,
        Self::QsaSelected, Self::AttentionOutput, Self::PleStaged, Self::PleRows,
        Self::PleKey, Self::PleValue, Self::PleQuery, Self::PleGated,
        Self::PleNormed, Self::PleOutput, Self::RouterLogits, Self::MoeXRot,
        Self::MoeGateUp, Self::MoeGate, Self::MoeUp, Self::MoeHidden,
        Self::FfnOutput, Self::MoeSharedOutput, Self::MoeGateBatch,
        Self::MoeUpBatch, Self::MoeRotBatch, Self::MoeTopkIndices,
        Self::MoeTopkWeights, Self::MoeDownExpanded, Self::MoeScalar,
        Self::MoeExpertTokenCounts, Self::MoeExpertOffsets, Self::MoeSortedSlotIndex,
        Self::MoeExpertTileIds, Self::MoeInversePerm, Self::MoeYGateUpGrouped,
        Self::MoeYDownGrouped, Self::Logits,
    ];

    /// Pair execution splits before the mixer: `HcMixed` (the mixer input)
    /// and `HcGates` (the attention write's gates, from prepare's paired HC
    /// read or the previous fragment's MoE row fold; the BF16-stream write
    /// reads no `HcNormalized`) survive that split. Keep this exhaustive when
    /// adding fields.
    fn npu_bank_lifetime(self) -> NpuBankLifetime {
        use NpuBankLifetime as L;
        match self {
            Self::TokenIds | Self::EmbeddingRot | Self::Embeddings | Self::PleStaged => L::Initialize,
            Self::Streams => L::Residual,
            Self::HcMixed | Self::HcGates => L::PreparedInput,
            Self::PleRows => L::StagedRows,
            Self::HcNormalized | Self::HcLow | Self::HcUp | Self::Rotation | Self::GdnBf16
            | Self::PleKey | Self::PleValue | Self::PleQuery | Self::PleGated
            | Self::PleNormed | Self::PleOutput => L::PrepareAndFragment,
            Self::Projection | Self::Projection2
            | Self::GdnA | Self::GdnB | Self::GdnGate | Self::GdnBeta
            | Self::GdnRecurrentOutput | Self::GdnZ | Self::GdnOutput
            | Self::QsaIndex | Self::QsaQgate | Self::QsaK | Self::QsaV
            | Self::QsaOutput | Self::QsaSelected | Self::AttentionOutput
            | Self::RouterLogits
            | Self::MoeXRot | Self::MoeGateUp | Self::MoeGate | Self::MoeUp
            | Self::MoeHidden | Self::FfnOutput | Self::MoeSharedOutput
            | Self::MoeGateBatch | Self::MoeUpBatch | Self::MoeRotBatch
            | Self::MoeTopkIndices | Self::MoeTopkWeights | Self::MoeDownExpanded
            | Self::MoeScalar | Self::MoeExpertTokenCounts | Self::MoeExpertOffsets
            | Self::MoeSortedSlotIndex | Self::MoeExpertTileIds | Self::MoeInversePerm
            | Self::MoeYGateUpGrouped | Self::MoeYDownGrouped => L::Fragment,
            Self::Logits => L::Finish,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NpuBankLifetime {
    Initialize,
    Residual,
    PreparedInput,
    StagedRows,
    PrepareAndFragment,
    Fragment,
    Finish,
}

impl NpuBankLifetime {
    fn private(self) -> bool {
        matches!(self, Self::Residual | Self::PreparedInput | Self::StagedRows)
    }
}

/// Which pair executor owns bank B. Under `Npu` the in-projection outputs live
/// in the per-bank NPU C arenas across the switch; under `Emulated` the GPU
/// twin projection writes them into scratch one layer ahead of their Core, so
/// bank B keeps its own copies instead of sharing bank A's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NpuBankKind {
    Npu,
    Emulated,
}

impl NpuBankKind {
    /// Whether bank B owns `field` (rather than borrowing bank A's).
    fn owns(self, field: ScratchField) -> bool {
        use ScratchField as F;
        field.npu_bank_lifetime().private()
            || (self == Self::Emulated
                && matches!(
                    field,
                    F::Projection | F::GdnA | F::GdnB | F::GdnZ
                        | F::QsaIndex | F::QsaQgate | F::QsaK | F::QsaV
                ))
    }
}

/// One field of a scratch plan.
///
/// `owner == None`: the field owns a device allocation of `elements` of
/// `dtype`.  `owner == Some(o)`: the field is a borrowed, never-freed view of
/// `o`'s allocation from offset zero with the same `dtype` and an honest
/// `elements`; `o` itself always owns its allocation.
///
/// `pitch` is set for every member of an alias set (the owner included, except
/// a pure-capacity owner such as the grouped-down buffer): the row pitch in
/// elements that a lane segment base uses.  All members of one alias set share
/// the owner's pitch, so segments of different members that are live at once
/// occupy disjoint, identically addressed row ranges (see
/// [`Qwen4GpuForwardScratch::segment_view`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScratchSlot {
    pub elements: usize,
    pub dtype: DType,
    pub owner: Option<ScratchField>,
    pub pitch: Option<usize>,
}

impl ScratchSlot {
    fn bytes(&self) -> Result<usize, Qwen4GpuForwardError> {
        self.elements
            .checked_mul(self.dtype.size())
            .ok_or_else(|| invalid("scratch field byte size overflows"))
    }
}

fn scratch_mul(a: usize, b: usize) -> Result<usize, Qwen4GpuForwardError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("scratch size arithmetic overflows"))
}

/// `(offset, len)` in elements of lane segment `[begin, begin + rows)` of an
/// alias-set member whose owner row pitch is `pitch` (`logical_width <=
/// pitch` elements per row are live) inside an owner view of `owner_numel`
/// elements.  The base is `begin * pitch`, never `begin * logical_width`, so
/// two members of one set that are live for the same segment address the
/// same rows and segments of different sizes cannot overlap.  The returned
/// length is the segment's whole reservation, `rows * pitch`.
pub(crate) fn owner_pitch_extent(
    pitch: usize,
    owner_numel: usize,
    begin: usize,
    rows: usize,
    logical_width: usize,
) -> Result<(usize, usize), Qwen4GpuForwardError> {
    if logical_width > pitch {
        return Err(invalid(format!(
            "lane segment row width {logical_width} exceeds its owner row pitch {pitch}"
        )));
    }
    let offset = scratch_mul(begin, pitch)?;
    let len = scratch_mul(rows, pitch)?;
    let end = offset
        .checked_add(len)
        .ok_or_else(|| invalid("lane segment reservation overflows"))?;
    if end > owner_numel {
        return Err(invalid(format!(
            "lane segment rows {begin}..{} at owner pitch {pitch} exceed the {owner_numel}-element owner",
            begin.saturating_add(rows)
        )));
    }
    Ok((offset, len))
}

/// Device buffers shared by one forward object.  All allocations happen in
/// `new`/`new_npu_bank`; token/chunk execution only creates borrowed subviews.
///
/// In [`Qwen4ForwardScratchMode::FnSymGfx1201`] eight fields are borrowed
/// views of another field's allocation (see [`Self::plan`]); only owners are
/// allocated, charged by [`Self::device_bytes`] and freed.
pub struct Qwen4GpuForwardScratch {
    pub max_chunk: usize,
    pub mode: Qwen4ForwardScratchMode,
    /// The plan this scratch was built from, in [`ScratchField`] order.
    pub(crate) slots: Vec<ScratchSlot>,
    /// NPU bank B borrows every layer-local field from bank A; only the
    /// bank-private lifetimes (and, under `emulate`, the in-projection outputs
    /// its twin projection writes ahead of the consuming Core) are owned here.
    /// Bank A outlives this view.
    bank_kind: Option<NpuBankKind>,
    pub token_ids: GpuTensor,
    pub embedding_rot: GpuTensor,
    pub embeddings: GpuTensor,
    pub streams: GpuTensor,
    pub hc_normalized: GpuTensor,
    pub hc_low: GpuTensor,
    pub hc_up: GpuTensor,
    pub hc_mixed: GpuTensor,
    pub hc_gates: GpuTensor,
    pub rotation: GpuTensor,
    pub projection: GpuTensor,
    pub projection2: GpuTensor,
    pub gdn_a: GpuTensor,
    pub gdn_b: GpuTensor,
    pub gdn_gate: GpuTensor,
    pub gdn_beta: GpuTensor,
    pub gdn_recurrent_output: GpuTensor,
    pub gdn_bf16: GpuTensor,
    pub gdn_z: GpuTensor,
    pub gdn_output: GpuTensor,
    pub qsa_index: GpuTensor,
    pub qsa_qgate: GpuTensor,
    pub qsa_k: GpuTensor,
    pub qsa_v: GpuTensor,
    pub qsa_output: GpuTensor,
    pub qsa_selected: GpuTensor,
    pub attention_output: GpuTensor,
    pub ple_staged: GpuTensor,
    pub ple_rows: GpuTensor,
    pub ple_key: GpuTensor,
    pub ple_value: GpuTensor,
    pub ple_query: GpuTensor,
    pub ple_gated: GpuTensor,
    pub ple_normed: GpuTensor,
    pub ple_output: GpuTensor,
    pub router_logits: GpuTensor,
    pub moe_x_rot: GpuTensor,
    pub moe_gate_up: GpuTensor,
    pub moe_gate: GpuTensor,
    pub moe_up: GpuTensor,
    pub moe_hidden: GpuTensor,
    /// Separate shared-expert down output.  The public MoE output is the
    /// routed accumulator; aliasing these buffers makes the shared GEMV
    /// overwrite the routed contribution before the source-ordered add.
    pub moe_shared_output: GpuTensor,
    /// The layer's FFN output (MoE routed accumulator or dense down output)
    /// that the HC write reads.
    pub ffn_output: GpuTensor,
    pub moe_gate_batch: GpuTensor,
    pub moe_up_batch: GpuTensor,
    pub moe_rot_batch: GpuTensor,
    pub moe_topk_indices: GpuTensor,
    pub moe_topk_weights: GpuTensor,
    pub moe_down_expanded: GpuTensor,
    pub moe_scalar: GpuTensor,
    pub moe_expert_token_counts: GpuTensor,
    pub moe_expert_offsets: GpuTensor,
    pub moe_sorted_slot_index: GpuTensor,
    pub moe_expert_tile_ids: GpuTensor,
    pub moe_inverse_perm: GpuTensor,
    pub moe_y_gate_up_grouped: GpuTensor,
    pub moe_y_down_grouped: GpuTensor,
    pub logits: GpuTensor,
}

impl Qwen4GpuForwardScratch {
    /// Every field of a `max_chunk`-row scratch in [`ScratchField`] order
    /// (allocation or borrowed view, with its dtype and element count), plus
    /// the host PLE staging bytes.
    ///
    /// `Full` is one dedicated allocation per field.  `FnSymGfx1201` applies,
    /// on top of the same field list, with `n = max_chunk`, `c = min(n,
    /// QWEN4_COMPACT_FALLBACK_ROWS)` and `G(r)` the tile-padded grouped rows of
    /// `r` rows:
    /// - gate/up/rot batches: `c * top_k * moe_intermediate` F32;
    /// - expanded down: `max(c*top_k*hidden*4, n*hidden*4, n*top_k*8)` bytes
    ///   (the shared-down F32 fallback and combine-order bytes are reserved
    ///   regardless of dtype or flags);
    /// - grouped gate/up: `max(G(c)*2*moe*4, G(n)*moe*2)` bytes and grouped
    ///   down `max(G(c)*hidden*4, G(n)*hidden*2)` bytes, kept as F32-typed
    ///   facades of `bytes / 4` elements (the IU4 kernels read the pointers as
    ///   BF16 storage);
    /// - phase aliases: `hc_normalized`, `ple_query` -> `projection`;
    ///   `hc_up`, `ple_key` -> `projection2`; `ple_gated` -> `rotation`;
    ///   `ple_output` -> `moe_y_down_grouped`; `ple_value` ->
    ///   `moe_shared_output`; `gdn_recurrent_output` -> `qsa_qgate`.
    ///   `ple_normed` stays dedicated: partial verify rollback reads it after
    ///   the forward.
    fn plan(
        config: &Qwen4Config,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
    ) -> Result<(Vec<ScratchSlot>, usize), Qwen4GpuForwardError> {
        use ScratchField as F;
        if max_chunk == 0 {
            return Err(invalid("max_chunk is zero"));
        }
        let dims = program_dims(config);
        let layout = Qwen4ScratchLayout::for_rows(dims, max_chunk)
            .map_err(|error| invalid(error.to_string()))?;
        let hidden = layout.hidden;
        let wide = layout.wide;
        let q_width = layout.q_width;
        let qsa_qgate = 2 * q_width;
        let qsa_index = layout.index_width;
        let gdn_value = layout.gdn_value;
        let gdn_qkv = layout.gdn_qkv;
        let max_projection = wide.max(qsa_qgate).max(gdn_qkv).max(hidden);
        let max_rotation = wide.max(hidden).max(config.ffn.intermediate_size());
        // PLE channels are `hc_count * hidden` (the key projection's rows),
        // not `ple_embed_dim * hc_count`; the staged row width is
        // `head_count * row_width = ple_embed_dim`.
        let ple_embed_dim = config
            .ple
            .as_ref()
            .map_or(0, |ple| ple.head_count * ple.row_width);
        let ple_channels = wide;
        let ffn_batch_rows = layout.routed_rows.max(layout.dense_rows);
        let max_ple_bytes = max_chunk
            .checked_mul(ple_embed_dim)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| invalid("PLE staging size overflow"))?;
        let i32_bytes = std::mem::size_of::<i32>();
        let qsa_selected_bytes = max_chunk
            .checked_mul(config.qsa_selected_capacity())
            .and_then(|elements| elements.checked_mul(i32_bytes))
            .ok_or_else(|| invalid("QSA selected scratch size overflow"))?;
        let route_i32_bytes = layout
            .slots
            .checked_mul(i32_bytes)
            .ok_or_else(|| invalid("MoE route index scratch overflow"))?;
        let expert_i32_bytes = config
            .num_experts
            .checked_mul(i32_bytes)
            .ok_or_else(|| invalid("MoE expert index scratch overflow"))?;
        let expert_offsets_bytes = config
            .num_experts
            .checked_add(1)
            .and_then(|count| count.checked_mul(i32_bytes))
            .ok_or_else(|| invalid("MoE expert offset scratch overflow"))?;
        let grouped_bound = layout.grouped_rows;
        let grouped_i32_bytes = grouped_bound
            .checked_mul(i32_bytes)
            .ok_or_else(|| invalid("MoE grouped index scratch overflow"))?;
        let tile_i32_bytes = (grouped_bound / 16)
            .checked_mul(i32_bytes)
            .ok_or_else(|| invalid("MoE grouped tile scratch overflow"))?;
        let rows = |width: usize| scratch_mul(max_chunk, width);
        let routed_width = scratch_mul(config.num_experts_per_tok, config.moe_intermediate_size)?;
        let mut slots: Vec<ScratchSlot> = Vec::with_capacity(ScratchField::COUNT);
        let mut alloc = |field: ScratchField,
                         elements: usize,
                         dtype: DType|
         -> Result<(), Qwen4GpuForwardError> {
            if slots.len() != field as usize {
                return Err(invalid(format!(
                    "scratch plan field {field:?} is out of order"
                )));
            }
            // Every planned buffer holds at least one element: MoE-only
            // extents are zero for a dense FFN.
            slots.push(ScratchSlot {
                elements: elements.max(1),
                dtype,
                owner: None,
                pitch: None,
            });
            Ok(())
        };
        let result = (|| {
            alloc(F::TokenIds, rows(std::mem::size_of::<i32>())?, DType::Raw)?;
            alloc(F::EmbeddingRot, rows(hidden)?, DType::F32)?;
            alloc(F::Embeddings, rows(hidden)?, DType::F32)?;
            alloc(F::Streams, rows(wide)?, DType::F32)?;
            alloc(F::HcNormalized, rows(wide)?, DType::F32)?;
            alloc(F::HcLow, rows(config.hc_lowrank)?, DType::F32)?;
            alloc(F::HcUp, rows(wide)?, DType::F32)?;
            alloc(F::HcMixed, rows(hidden)?, DType::F32)?;
            alloc(F::HcGates, rows(config.hc_count)?, DType::F32)?;
            alloc(F::Rotation, rows(max_rotation)?, DType::F32)?;
            alloc(F::Projection, rows(max_projection)?, DType::F32)?;
            alloc(F::Projection2, rows(max_projection)?, DType::F32)?;
            alloc(F::GdnA, rows(config.linear_num_value_heads)?, DType::F32)?;
            alloc(F::GdnB, rows(config.linear_num_value_heads)?, DType::F32)?;
            alloc(F::GdnGate, rows(config.linear_num_value_heads)?, DType::F32)?;
            alloc(F::GdnBeta, rows(config.linear_num_value_heads)?, DType::F32)?;
            alloc(F::GdnRecurrentOutput, rows(gdn_value)?, DType::F32)?;
            alloc(F::GdnBf16, gdn_value.max(hidden).max(config.hc_lowrank), DType::BF16)?;
            alloc(F::GdnZ, rows(gdn_value)?, DType::F32)?;
            alloc(F::GdnOutput, rows(gdn_value)?, DType::F32)?;
            alloc(F::QsaIndex, rows(qsa_index)?, DType::F32)?;
            alloc(F::QsaQgate, rows(qsa_qgate)?, DType::F32)?;
            alloc(
                F::QsaK,
                rows(scratch_mul(config.num_key_value_heads, config.head_dim)?)?,
                DType::F32,
            )?;
            alloc(
                F::QsaV,
                rows(scratch_mul(config.num_key_value_heads, config.head_dim)?)?,
                DType::F32,
            )?;
            alloc(F::QsaOutput, rows(q_width)?, DType::F32)?;
            alloc(F::QsaSelected, qsa_selected_bytes, DType::Raw)?;
            alloc(F::AttentionOutput, rows(hidden)?, DType::F32)?;
            alloc(F::PleStaged, rows(ple_embed_dim)?, DType::BF16)?;
            alloc(F::PleRows, rows(ple_embed_dim.max(hidden))?, DType::F32)?;
            alloc(F::PleKey, rows(ple_channels)?, DType::F32)?;
            alloc(F::PleValue, rows(hidden)?, DType::F32)?;
            alloc(F::PleQuery, rows(ple_channels)?, DType::F32)?;
            alloc(F::PleGated, rows(ple_channels)?, DType::F32)?;
            alloc(F::PleNormed, rows(ple_channels)?, DType::F32)?;
            alloc(F::PleOutput, rows(ple_channels)?, DType::F32)?;
            // Viewed as `[rows, router_columns]` by every forward (one
            // placeholder column for a dense FFN).
            alloc(F::RouterLogits, rows(router_columns(dims))?, DType::F32)?;
            alloc(F::MoeXRot, rows(hidden)?, DType::F32)?;
            alloc(
                F::MoeGateUp,
                rows(2 * config.moe_intermediate_size)?,
                DType::F32,
            )?;
            alloc(
                F::MoeGate,
                rows(config.shared_expert_intermediate_size)?,
                DType::F32,
            )?;
            alloc(
                F::MoeUp,
                rows(config.shared_expert_intermediate_size)?,
                DType::F32,
            )?;
            alloc(
                F::MoeHidden,
                rows(config.shared_expert_intermediate_size)?,
                DType::F32,
            )?;
            alloc(F::FfnOutput, rows(hidden)?, DType::F32)?;
            alloc(F::MoeSharedOutput, rows(hidden)?, DType::F32)?;
            // Batched gate/up rows: routed slots * routed intermediate for a
            // MoE profile, rows * intermediate for a dense FFN (exactly one of
            // the two extents is nonzero); `Step::DenseFfn` borrows both.
            alloc(F::MoeGateBatch, ffn_batch_rows, DType::F32)?;
            alloc(F::MoeUpBatch, ffn_batch_rows, DType::F32)?;
            alloc(F::MoeRotBatch, rows(routed_width)?, DType::F32)?;
            alloc(F::MoeTopkIndices, route_i32_bytes, DType::Raw)?;
            alloc(
                F::MoeTopkWeights,
                rows(config.num_experts_per_tok)?,
                DType::F32,
            )?;
            alloc(
                F::MoeDownExpanded,
                rows(scratch_mul(config.num_experts_per_tok, hidden)?)?,
                DType::F32,
            )?;
            alloc(
                F::MoeScalar,
                rows(config.shared_expert_intermediate_size)?,
                DType::F32,
            )?;
            alloc(F::MoeExpertTokenCounts, expert_i32_bytes, DType::Raw)?;
            alloc(F::MoeExpertOffsets, expert_offsets_bytes, DType::Raw)?;
            alloc(F::MoeSortedSlotIndex, grouped_i32_bytes, DType::Raw)?;
            alloc(F::MoeExpertTileIds, tile_i32_bytes, DType::Raw)?;
            alloc(F::MoeInversePerm, route_i32_bytes, DType::Raw)?;
            alloc(
                F::MoeYGateUpGrouped,
                scratch_mul(grouped_bound, 2 * config.moe_intermediate_size)?,
                DType::F32,
            )?;
            alloc(
                F::MoeYDownGrouped,
                scratch_mul(grouped_bound, hidden)?,
                DType::F32,
            )?;
            alloc(F::Logits, config.vocab_size, DType::F32)?;
            Ok::<(), Qwen4GpuForwardError>(())
        })();
        result?;
        if slots.len() != ScratchField::COUNT {
            return Err(invalid("scratch plan does not cover every field"));
        }
        if mode == Qwen4ForwardScratchMode::FnSymGfx1201 {
            if ple_channels != wide {
                return Err(invalid(
                    "compact scratch requires PLE channels equal to the hyper-connection width",
                ));
            }
            let f32_bytes = std::mem::size_of::<f32>();
            let bf16_bytes = DType::BF16.size();
            let top_k = config.num_experts_per_tok;
            let moe = config.moe_intermediate_size;
            let fallback_rows = max_chunk.min(QWEN4_COMPACT_FALLBACK_ROWS);
            let fallback_grouped = crate::program::grouped_m_total_bound(
                scratch_mul(fallback_rows, top_k)?,
                config.num_experts,
            )
            .map_err(|error| invalid(error.to_string()))?;
            // Elements of an F32-typed facade over `bytes` of storage.
            let facade = |bytes: usize| bytes.div_ceil(f32_bytes);
            let fallback_batch = scratch_mul(fallback_rows, routed_width)?;
            for field in [F::MoeGateBatch, F::MoeUpBatch, F::MoeRotBatch] {
                slots[field as usize].elements = fallback_batch;
            }
            let hidden_f32_bytes = scratch_mul(hidden, f32_bytes)?;
            let expanded_bytes = scratch_mul(scratch_mul(fallback_rows, top_k)?, hidden_f32_bytes)?
                .max(scratch_mul(max_chunk, hidden_f32_bytes)?)
                .max(scratch_mul(scratch_mul(max_chunk, top_k)?, 8)?);
            slots[F::MoeDownExpanded as usize].elements = facade(expanded_bytes);
            let gate_up_f32 =
                scratch_mul(fallback_grouped, scratch_mul(2 * moe, f32_bytes)?)?;
            let gate_up_bf16 = scratch_mul(grouped_bound, scratch_mul(moe, bf16_bytes)?)?;
            slots[F::MoeYGateUpGrouped as usize].elements = facade(gate_up_f32.max(gate_up_bf16));
            let down_f32 = scratch_mul(fallback_grouped, hidden_f32_bytes)?;
            let down_bf16 = scratch_mul(grouped_bound, scratch_mul(hidden, bf16_bytes)?)?;
            slots[F::MoeYDownGrouped as usize].elements = facade(down_f32.max(down_bf16));

            // `owner_pitch`: row pitch of the owner's lane segments (`None`
            // for a pure-capacity owner); `view_pitch`: the views' pitch.
            let mut bind = |owner: F,
                            owner_pitch: Option<usize>,
                            view_pitch: usize,
                            views: &[F]|
             -> Result<(), Qwen4GpuForwardError> {
                let owner_slot = slots[owner as usize];
                if owner_slot.owner.is_some() {
                    return Err(invalid(format!(
                        "scratch alias owner {owner:?} is itself a view"
                    )));
                }
                if let Some(pitch) = owner_pitch {
                    if owner_slot.elements < scratch_mul(max_chunk, pitch)? {
                        return Err(invalid(format!(
                            "scratch alias owner {owner:?} is smaller than {max_chunk} rows of pitch {pitch}"
                        )));
                    }
                    slots[owner as usize].pitch = Some(pitch);
                }
                for &view in views {
                    let slot = slots[view as usize];
                    if slot.owner.is_some() || slot.dtype != owner_slot.dtype {
                        return Err(invalid(format!(
                            "scratch field {view:?} cannot alias {owner:?}"
                        )));
                    }
                    if slot.elements > owner_slot.elements {
                        return Err(invalid(format!(
                            "scratch field {view:?} ({} elements) exceeds its owner {owner:?} ({} elements)",
                            slot.elements, owner_slot.elements
                        )));
                    }
                    if slot.elements > scratch_mul(max_chunk, view_pitch)? {
                        return Err(invalid(format!(
                            "scratch field {view:?} is wider than its owner pitch {view_pitch}"
                        )));
                    }
                    slots[view as usize].owner = Some(owner);
                    slots[view as usize].pitch = Some(view_pitch);
                }
                Ok(())
            };
            bind(
                F::Projection,
                Some(max_projection),
                max_projection,
                &[F::HcNormalized, F::PleQuery],
            )?;
            bind(
                F::Projection2,
                Some(max_projection),
                max_projection,
                &[F::HcUp, F::PleKey],
            )?;
            bind(F::Rotation, Some(max_rotation), max_rotation, &[F::PleGated])?;
            bind(F::MoeYDownGrouped, None, ple_channels, &[F::PleOutput])?;
            bind(F::MoeSharedOutput, Some(hidden), hidden, &[F::PleValue])?;
            bind(
                F::QsaQgate,
                Some(qsa_qgate),
                qsa_qgate,
                &[F::GdnRecurrentOutput],
            )?;
        }
        Ok((slots, max_ple_bytes))
    }

    /// Device bytes [`Self::new`] allocates for `max_chunk` rows: the owning
    /// slots only; borrowed views are not charged.
    pub(crate) fn device_bytes(
        config: &Qwen4Config,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
    ) -> Result<u64, Qwen4GpuForwardError> {
        let (slots, _) = Self::plan(config, max_chunk, mode)?;
        slots
            .iter()
            .filter(|slot| slot.owner.is_none())
            .try_fold(0u64, |total, slot| {
                total
                    .checked_add(slot.bytes()? as u64)
                    .ok_or_else(|| invalid("scratch device bytes overflow"))
            })
    }

    pub fn new(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
    ) -> Result<(Self, Vec<u8>, Vec<u8>), Qwen4GpuForwardError> {
        Self::allocate(gpu, config, max_chunk, mode, None)
    }

    /// Allocate only state live across the A/B switch. All borrowed pointers
    /// remain owned by `bank_a`, which must outlive this bank and GPU work.
    fn new_npu_bank(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        bank_a: &Self,
        emulated: bool,
    ) -> Result<(Self, Vec<u8>, Vec<u8>), Qwen4GpuForwardError> {
        if bank_a.mode != Qwen4ForwardScratchMode::Full {
            return Err(invalid("NPU shared bank requires full scratch"));
        }
        let kind = if emulated { NpuBankKind::Emulated } else { NpuBankKind::Npu };
        Self::allocate(gpu, config, bank_a.max_chunk, bank_a.mode, Some((bank_a, kind)))
    }

    #[cfg(test)]
    fn npu_bank_device_bytes(
        config: &Qwen4Config,
        max_chunk: usize,
    ) -> Result<u64, Qwen4GpuForwardError> {
        let (slots, _) = Self::plan(config, max_chunk, Qwen4ForwardScratchMode::Full)?;
        Self::npu_bank_plan_bytes(&slots, false)
    }

    fn npu_bank_plan_bytes(slots: &[ScratchSlot], emulated: bool) -> Result<u64, Qwen4GpuForwardError> {
        let kind = if emulated { NpuBankKind::Emulated } else { NpuBankKind::Npu };
        slots.iter().enumerate().try_fold(0u64, |total, (index, slot)| {
            // The exhaustive field table is shared by allocation and freeing.
            let private = kind.owns(ScratchField::ALL[index]);
            total.checked_add(if private { slot.bytes()? as u64 } else { 0 })
                .ok_or_else(|| invalid("NPU bank device bytes overflow"))
        })
    }

    fn allocate(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
        bank: Option<(&Self, NpuBankKind)>,
    ) -> Result<(Self, Vec<u8>, Vec<u8>), Qwen4GpuForwardError> {
        let bank_a = bank.map(|(bank_a, _)| bank_a);
        let bank_kind = bank.map(|(_, kind)| kind);
        let (slots, max_ple_bytes) = Self::plan(config, max_chunk, mode)?;
        // Owners first; a view needs its owner's pointer.
        let mut tensors: Vec<Option<GpuTensor>> = Vec::with_capacity(slots.len());
        for (index, slot) in slots.iter().enumerate() {
            if slot.owner.is_some()
                || bank_kind.is_some_and(|kind| !kind.owns(ScratchField::ALL[index]))
            {
                tensors.push(None);
                continue;
            }
            match gpu.zeros(&[slot.elements], slot.dtype) {
                Ok(tensor) => tensors.push(Some(tensor)),
                Err(error) => {
                    // Only owners are present: each is freed exactly once.
                    for tensor in tensors.into_iter().flatten() {
                        let _ = gpu.free_tensor(tensor);
                    }
                    return Err(error.into());
                }
            }
        }
        for (index, slot) in slots.iter().enumerate() {
            if let Some(owner) = slot.owner {
                let view = tensors[owner as usize]
                    .as_ref()
                    .expect("plan validated that a view's owner owns its allocation")
                    .sub_offset(0, slot.elements);
                tensors[index] = Some(view);
            }
        }
        if let Some(bank_a) = bank_a {
            for (index, slot) in slots.iter().enumerate() {
                let field = ScratchField::ALL[index];
                if !bank_kind.expect("a borrowed bank has a kind").owns(field) {
                    tensors[index] = Some(bank_a.field_tensor(field).sub_offset(0, slot.elements));
                }
            }
        }
        let mut take = |field: ScratchField| {
            tensors[field as usize]
                .take()
                .expect("every scratch field is bound exactly once")
        };
        use ScratchField as F;
        let scratch = Self {
            max_chunk,
            mode,
            bank_kind,
            token_ids: take(F::TokenIds),
            embedding_rot: take(F::EmbeddingRot),
            embeddings: take(F::Embeddings),
            streams: take(F::Streams),
            hc_normalized: take(F::HcNormalized),
            hc_low: take(F::HcLow),
            hc_up: take(F::HcUp),
            hc_mixed: take(F::HcMixed),
            hc_gates: take(F::HcGates),
            rotation: take(F::Rotation),
            projection: take(F::Projection),
            projection2: take(F::Projection2),
            gdn_a: take(F::GdnA),
            gdn_b: take(F::GdnB),
            gdn_gate: take(F::GdnGate),
            gdn_beta: take(F::GdnBeta),
            gdn_recurrent_output: take(F::GdnRecurrentOutput),
            gdn_bf16: take(F::GdnBf16),
            gdn_z: take(F::GdnZ),
            gdn_output: take(F::GdnOutput),
            qsa_index: take(F::QsaIndex),
            qsa_qgate: take(F::QsaQgate),
            qsa_k: take(F::QsaK),
            qsa_v: take(F::QsaV),
            qsa_output: take(F::QsaOutput),
            qsa_selected: take(F::QsaSelected),
            attention_output: take(F::AttentionOutput),
            ple_staged: take(F::PleStaged),
            ple_rows: take(F::PleRows),
            ple_key: take(F::PleKey),
            ple_value: take(F::PleValue),
            ple_query: take(F::PleQuery),
            ple_gated: take(F::PleGated),
            ple_normed: take(F::PleNormed),
            ple_output: take(F::PleOutput),
            router_logits: take(F::RouterLogits),
            moe_x_rot: take(F::MoeXRot),
            moe_gate_up: take(F::MoeGateUp),
            moe_gate: take(F::MoeGate),
            moe_up: take(F::MoeUp),
            moe_hidden: take(F::MoeHidden),
            ffn_output: take(F::FfnOutput),
            moe_shared_output: take(F::MoeSharedOutput),
            moe_gate_batch: take(F::MoeGateBatch),
            moe_up_batch: take(F::MoeUpBatch),
            moe_rot_batch: take(F::MoeRotBatch),
            moe_topk_indices: take(F::MoeTopkIndices),
            moe_topk_weights: take(F::MoeTopkWeights),
            moe_down_expanded: take(F::MoeDownExpanded),
            moe_scalar: take(F::MoeScalar),
            moe_expert_token_counts: take(F::MoeExpertTokenCounts),
            moe_expert_offsets: take(F::MoeExpertOffsets),
            moe_sorted_slot_index: take(F::MoeSortedSlotIndex),
            moe_expert_tile_ids: take(F::MoeExpertTileIds),
            moe_inverse_perm: take(F::MoeInversePerm),
            moe_y_gate_up_grouped: take(F::MoeYGateUpGrouped),
            moe_y_down_grouped: take(F::MoeYDownGrouped),
            logits: take(F::Logits),
            slots,
        };
        Ok((
            scratch,
            vec![0; max_chunk * std::mem::size_of::<i32>()],
            vec![0; max_ple_bytes],
        ))
    }

    /// The tensor of `field`: an owner's allocation or a borrowed view.
    pub(crate) fn field_tensor(&self, field: ScratchField) -> &GpuTensor {
        use ScratchField as F;
        match field {
            F::TokenIds => &self.token_ids,
            F::EmbeddingRot => &self.embedding_rot,
            F::Embeddings => &self.embeddings,
            F::Streams => &self.streams,
            F::HcNormalized => &self.hc_normalized,
            F::HcLow => &self.hc_low,
            F::HcUp => &self.hc_up,
            F::HcMixed => &self.hc_mixed,
            F::HcGates => &self.hc_gates,
            F::Rotation => &self.rotation,
            F::Projection => &self.projection,
            F::Projection2 => &self.projection2,
            F::GdnA => &self.gdn_a,
            F::GdnB => &self.gdn_b,
            F::GdnGate => &self.gdn_gate,
            F::GdnBeta => &self.gdn_beta,
            F::GdnRecurrentOutput => &self.gdn_recurrent_output,
            F::GdnBf16 => &self.gdn_bf16,
            F::GdnZ => &self.gdn_z,
            F::GdnOutput => &self.gdn_output,
            F::QsaIndex => &self.qsa_index,
            F::QsaQgate => &self.qsa_qgate,
            F::QsaK => &self.qsa_k,
            F::QsaV => &self.qsa_v,
            F::QsaOutput => &self.qsa_output,
            F::QsaSelected => &self.qsa_selected,
            F::AttentionOutput => &self.attention_output,
            F::PleStaged => &self.ple_staged,
            F::PleRows => &self.ple_rows,
            F::PleKey => &self.ple_key,
            F::PleValue => &self.ple_value,
            F::PleQuery => &self.ple_query,
            F::PleGated => &self.ple_gated,
            F::PleNormed => &self.ple_normed,
            F::PleOutput => &self.ple_output,
            F::RouterLogits => &self.router_logits,
            F::MoeXRot => &self.moe_x_rot,
            F::MoeGateUp => &self.moe_gate_up,
            F::MoeGate => &self.moe_gate,
            F::MoeUp => &self.moe_up,
            F::MoeHidden => &self.moe_hidden,
            F::FfnOutput => &self.ffn_output,
            F::MoeSharedOutput => &self.moe_shared_output,
            F::MoeGateBatch => &self.moe_gate_batch,
            F::MoeUpBatch => &self.moe_up_batch,
            F::MoeRotBatch => &self.moe_rot_batch,
            F::MoeTopkIndices => &self.moe_topk_indices,
            F::MoeTopkWeights => &self.moe_topk_weights,
            F::MoeDownExpanded => &self.moe_down_expanded,
            F::MoeScalar => &self.moe_scalar,
            F::MoeExpertTokenCounts => &self.moe_expert_token_counts,
            F::MoeExpertOffsets => &self.moe_expert_offsets,
            F::MoeSortedSlotIndex => &self.moe_sorted_slot_index,
            F::MoeExpertTileIds => &self.moe_expert_tile_ids,
            F::MoeInversePerm => &self.moe_inverse_perm,
            F::MoeYGateUpGrouped => &self.moe_y_gate_up_grouped,
            F::MoeYDownGrouped => &self.moe_y_down_grouped,
            F::Logits => &self.logits,
        }
    }

    /// Rows `[begin, begin + rows)` of `field` for one lane segment, each
    /// row `logical_width` live elements wide.
    ///
    /// A dedicated field (no pitch) is addressed contiguously at its logical
    /// width from row `begin` to its end.  An alias-set member is addressed at
    /// its allocation owner's row pitch ([`owner_pitch_extent`]) and the view
    /// is the segment's whole `rows * pitch` reservation, so every member of a
    /// set maps one segment to the same rows of one owner.  The returned view
    /// is an ordinary contiguous tensor: kernels get no strided ABI.
    pub(crate) fn segment_view(
        &self,
        field: ScratchField,
        begin: usize,
        rows: usize,
        logical_width: usize,
    ) -> Result<GpuTensor, Qwen4GpuForwardError> {
        let tensor = self.field_tensor(field);
        let numel = tensor.numel();
        match self.slots[field as usize].pitch {
            None => {
                let offset = scratch_mul(begin, logical_width)?;
                if offset > numel {
                    return Err(invalid("lane row view starts past its scratch tensor"));
                }
                Ok(tensor.sub_offset(offset, numel - offset))
            }
            Some(pitch) => {
                let (offset, len) = owner_pitch_extent(pitch, numel, begin, rows, logical_width)?;
                Ok(tensor.sub_offset(offset, len))
            }
        }
    }

    /// Free the owners.  Borrowed views are never freed: their allocation is
    /// freed once, through its owner.
    fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let slots = self.slots;
        let bank_kind = self.bank_kind;
        let tensors = [
            self.token_ids,
            self.embedding_rot,
            self.embeddings,
            self.streams,
            self.hc_normalized,
            self.hc_low,
            self.hc_up,
            self.hc_mixed,
            self.hc_gates,
            self.rotation,
            self.projection,
            self.projection2,
            self.gdn_a,
            self.gdn_b,
            self.gdn_gate,
            self.gdn_beta,
            self.gdn_recurrent_output,
            self.gdn_bf16,
            self.gdn_z,
            self.gdn_output,
            self.qsa_index,
            self.qsa_qgate,
            self.qsa_k,
            self.qsa_v,
            self.qsa_output,
            self.qsa_selected,
            self.attention_output,
            self.ple_staged,
            self.ple_rows,
            self.ple_key,
            self.ple_value,
            self.ple_query,
            self.ple_gated,
            self.ple_normed,
            self.ple_output,
            self.router_logits,
            self.moe_x_rot,
            self.moe_gate_up,
            self.moe_gate,
            self.moe_up,
            self.moe_hidden,
            self.ffn_output,
            self.moe_shared_output,
            self.moe_gate_batch,
            self.moe_up_batch,
            self.moe_rot_batch,
            self.moe_topk_indices,
            self.moe_topk_weights,
            self.moe_down_expanded,
            self.moe_scalar,
            self.moe_expert_token_counts,
            self.moe_expert_offsets,
            self.moe_sorted_slot_index,
            self.moe_expert_tile_ids,
            self.moe_inverse_perm,
            self.moe_y_gate_up_grouped,
            self.moe_y_down_grouped,
            self.logits,
        ];
        debug_assert_eq!(tensors.len(), slots.len());
        let mut first = None;
        for (index, (tensor, slot)) in tensors.into_iter().zip(slots).enumerate() {
            if slot.owner.is_some()
                || bank_kind.is_some_and(|kind| !kind.owns(ScratchField::ALL[index]))
            {
                continue;
            }
            if let Err(error) = gpu.free_tensor(tensor) {
                if first.is_none() {
                    first = Some(error);
                }
            }
        }
        first
    }
}

pub(crate) fn layer_scratch<'a>(
    scratch: &'a Qwen4GpuForwardScratch,
    router_logits: &'a GpuTensor,
) -> Qwen4LayerScratch<'a> {
    Qwen4LayerScratch {
        streams: &scratch.streams,
        rotation: &scratch.rotation,
        hc_normalized: &scratch.hc_normalized,
        hc_low: &scratch.hc_low,
        hc_up: &scratch.hc_up,
        hc_mixed: &scratch.hc_mixed,
        hc_gates: &scratch.hc_gates,
        projection: &scratch.projection,
        projection2: &scratch.projection2,
        gdn_a: &scratch.gdn_a,
        gdn_b: &scratch.gdn_b,
        gdn_gate: &scratch.gdn_gate,
        gdn_beta: &scratch.gdn_beta,
        gdn_recurrent_output: &scratch.gdn_recurrent_output,
        gdn_bf16: &scratch.gdn_bf16,
        gdn_z: &scratch.gdn_z,
        gdn_output: &scratch.gdn_output,
        qsa_index: &scratch.qsa_index,
        qsa_qgate: &scratch.qsa_qgate,
        qsa_k: &scratch.qsa_k,
        qsa_v: &scratch.qsa_v,
        qsa_output: &scratch.qsa_output,
        attention_output: &scratch.attention_output,
        ffn_output: &scratch.ffn_output,
        moe_shared_output: &scratch.moe_shared_output,
        moe_router_logits: router_logits,
        moe_x_rot: &scratch.moe_x_rot,
        moe_gate_up: &scratch.moe_gate_up,
        moe_scalar: &scratch.moe_scalar,
        moe_gate: &scratch.moe_gate,
        moe_up: &scratch.moe_up,
        moe_hidden: &scratch.moe_hidden,
        moe_gate_batch: &scratch.moe_gate_batch,
        moe_up_batch: &scratch.moe_up_batch,
        moe_rot_batch: &scratch.moe_rot_batch,
        moe_topk_indices: &scratch.moe_topk_indices,
        moe_topk_weights: &scratch.moe_topk_weights,
        moe_down_expanded: &scratch.moe_down_expanded,
        moe_expert_token_counts: &scratch.moe_expert_token_counts,
        moe_expert_offsets: &scratch.moe_expert_offsets,
        moe_sorted_slot_index: &scratch.moe_sorted_slot_index,
        moe_expert_tile_ids: &scratch.moe_expert_tile_ids,
        moe_inverse_perm: &scratch.moe_inverse_perm,
        moe_y_gate_up_grouped: &scratch.moe_y_gate_up_grouped,
        moe_y_down_grouped: &scratch.moe_y_down_grouped,
    }
}

/// Parity-harness observer run after each QSA layer's step on the ordinary HIP
/// route: `(gpu, qsa_slot, op)`, with the op's scratch and state holding that
/// layer's inputs, selection and attention output for the forward's rows.
/// Only `reference-parity` builds can install one ([`Qwen4Bundle::set_qsa_tap`]);
/// a forward with a tap never records or replays a retained body.
pub type Qwen4QsaTap =
    Box<dyn FnMut(&mut Gpu, usize, &IndexedAttentionOp<'_>) -> Result<(), String> + Send>;

/// Reusable production forward owner.  Construct it once per assembled bundle
/// and use `forward_token`/`forward_chunk` repeatedly; both methods share the
/// exact same chunk implementation.
pub struct Qwen4GpuForward {
    pub scratch: Qwen4GpuForwardScratch,
    pub(crate) host_token_bytes: Vec<u8>,
    /// Rows of `scratch.token_ids` the last forward uploaded whole from
    /// `host_token_bytes` (its mirror): 0 for a HIP single-row body, a device
    /// argmax id or a captured upload.
    pub(crate) uploaded_id_rows: usize,
    /// Host staging for the PLE rows and the fence over its asynchronous
    /// upload (`HIPFIRE_QWEN4_PLE_ASYNC_UPLOAD`).
    pub(crate) host_ple: PleHostStage,
    /// Tokens of the prefill chunk that follows the next forward. A
    /// successful forward enqueues their PLE rows (hashed from the history it
    /// committed) as `ple_ahead` once its own lease is released, so the next
    /// chunk's reads and row materialization overlap this chunk's GPU work.
    /// Consumed (cleared) by every forward.
    ple_lookahead: Vec<u32>,
    /// Retained next-chunk PLE ticket. The next forward adopts it when its
    /// row ids and epoch match exactly, else drops it and fetches afresh.
    pub(crate) ple_ahead: Option<RowTicket>,
    /// Device-argmax token readback (`forward_token_or_argmax`), once used.
    token_readback: Option<TokenReadback>,
    pub(crate) moe: Vec<Qwen4MoeLayerRuntime>,
    expert_stage: Option<Qwen4ExpertStage>,
    /// Expert-parallel state (`Some` iff the bundle's weights carry an EP
    /// placement): the zero dummy expert, the full per-layer tables for chunk
    /// forwards, the row-gather buffers and the installed gather. Under EP
    /// `moe` holds the per-layer decode tables, there is no G2 expert stage,
    /// and no forward records or replays a retained body.
    ep: Option<Qwen4ForwardEp>,
    /// `DECODE_Q8_PER_LAYER` per layer [HC attn down, attn up, mlp down, mlp
    /// up, shared gate, up, down]: BF16 matrices requantized to Q8_0 for
    /// single-token forwards (gfx1151). Batch-1 decode streams every weight
    /// once per token, and these are ~1.7 GB of it; Q8_0 halves that at a
    /// decode KLD inside the noise band (HC 0.0743 -> 0.0754, shared +0.0007
    /// over 32 chunks). Prefill keeps the BF16 source for its WMMA routes and
    /// the exact multi-row arms short prompts take.
    pub(crate) decode_q8: Vec<GpuTensor>,
    /// `HIPFIRE_QWEN4_ROUTE_TRACE`: routed expert ids per layer of every
    /// single-token HIP forward, for expert-cache sizing. `None` when unset.
    route_trace: Option<RouteTrace>,
    pub(crate) qsa_tap: Option<Qwen4QsaTap>,
    /// `HIPFIRE_QWEN4_TRUNK_IU4`: per layer, the GDN Z|beta|alpha fold the
    /// IU4 trunk's V2B SET reads (see [`prepare_trunk_iu4`]); empty when the
    /// route is off or the trunk did not verify.
    pub(crate) trunk_zba: Vec<Option<GpuTensor>>,
    /// `HIPFIRE_QWEN4_TRUNK_IU4`: per layer, the [`TrunkFamily`] bits the
    /// mask selected for A4 (empty when the route is off or did not arm).
    pub(crate) trunk_a4: Vec<u16>,
    /// Set only while a lane's prefill tile runs the singleton chunk program
    /// on a lane state ([`Qwen4Bundle::lane_prefill_tile`]): the forward is
    /// then never a retained-body candidate, so it neither records nor
    /// replays the singleton's tape (whose pointers name the resident state).
    pub(crate) lane_scope: bool,
    /// Set only by the bundle's prefill entries (`forward_chunk*`,
    /// `spec_prefill_rows*`, `lane_prefill_tile`) around their forward call:
    /// with `HIPFIRE_PREFILL_PM_INVARIANT` on gfx1151 the forward then runs the
    /// batch-invariant prefill profile ([`Gpu::prefill_invariant_scope`]). Never
    /// set for decode, MTP verify or the MTP head.
    pub(crate) prefill_scope: bool,
    #[cfg(target_os = "linux")]
    npu_trunk: Option<npu_forward::NpuForward>,
}

/// An event after a forward's device argmax and a stream independent of the
/// legacy null stream to read the token id on.
struct TokenReadback {
    stream: hip_bridge::Stream,
    event: hip_bridge::Event,
}

/// Environment knob naming the routing-trace output file (appended).
pub const ROUTE_TRACE_ENV: &str = "HIPFIRE_QWEN4_ROUTE_TRACE";

/// Routing trace sink: one text line per single-token forward holding
/// `position` then `num_hidden_layers * num_experts_per_tok` expert ids in
/// layer order. Each layer's top-K ids are copied device-to-device right after
/// its MoE step, so tracing splits the step list at MoE barriers only.
struct RouteTrace {
    device: GpuTensor,
    host: Vec<u8>,
    out: std::io::BufWriter<std::fs::File>,
}

/// Dense IU4 trunk (`HIPFIRE_QWEN4_TRUNK_IU4=<mask>`, see
/// [`rdna_compute::Qwen4TrunkMask`]) over a trunk that may mix symmetric
/// MQ4G256V2 (QT44) and other tiers (MQ6 ...) per tensor: check on the device
/// that every MQ4G256V2 trunk projection (GDN qkv/z/a/b/out, QSA
/// indexer/q/k/v/o) is a symmetric QT44 matrix, then fold each GDN layer's Z,
/// beta (`in_proj_b`) and alpha (`in_proj_a`) rows plus zero rows into one
/// `z.m + 256`-row matrix for the dense V2B Z|beta|alpha SET, for the layers
/// where all three are MQ4 and in the mask (the rows are copied verbatim; the
/// original tensors stay for decode). Arms `gpu.qwen4_trunk_iu4` iff at least
/// one trunk tensor is MQ4G256V2 and in the mask (A4 projections > 0) and
/// every MQ4 one verified. Returns the folds
/// and the per-layer A4 mask bits (empty when not armed). The mask is the
/// `HIPFIRE_QWEN4_TRUNK_IU4` override when that variable is set (`0` = off),
/// else the artifact's embedded `qwen4_trunk_iu4_default` default on its exact
/// architecture ([`rdna_compute::qwen4_trunk_iu4_route`]); an unmarked
/// artifact is off unless the variable selects a mask. A bad mask value
/// fails the load even when the route cannot arm; nothing else runs unless
/// the route was requested.
fn prepare_trunk_iu4(
    gpu: &mut Gpu,
    bundle: &Qwen4Bundle,
) -> Result<(Vec<Option<GpuTensor>>, Vec<u16>), Qwen4GpuForwardError> {
    gpu.qwen4_trunk_iu4 = false;
    let route = rdna_compute::qwen4_trunk_iu4_route(&gpu.arch, bundle.config.trunk_iu4_default)
        .map_err(invalid)?;
    let mask = &route.mask;
    mask.check_layers(bundle.weights.layer_refs.len())
        .map_err(|error| invalid(format!("HIPFIRE_QWEN4_TRUNK_IU4: {error}")))?;
    let mut folds = Vec::new();
    if !gpu.qwen4_trunk_iu4_requested(mask) {
        // A marked artifact on its exact arch whose default the env switched
        // off (`0` / empty): say so, so the kill is visible at load. Unmarked
        // artifacts and other arches return silently, as before.
        if let Some(default) = bundle.config.trunk_iu4_default {
            if route.source == rdna_compute::TrunkIu4MaskSource::Env
                && mask.is_empty()
                && gpu.arch == default.arch()
            {
                eprintln!(
                    "  qwen4 IU4 trunk: mask=off (env override), artifact default {} disabled; QT44 trunk weights stay resident, stock route",
                    default.marker()
                );
            }
        }
        return Ok((folds, Vec::new()));
    }
    let weights = &bundle.weights;
    let (mut mq4, mut verified, mut a4, mut other) = (0usize, 0usize, 0usize, 0usize);
    for (index, layer) in weights.layer_refs.iter().enumerate() {
        let bits = mask.bits(index);
        let matrices = match (&layer.gdn, &layer.attention) {
            (Some(gdn), _) => {
                let g = gdn_desc(weights, gdn)?;
                [
                    (TrunkFamily::GdnQkv, g.qkv),
                    (TrunkFamily::GdnZ, g.z),
                    (TrunkFamily::GdnA, g.in_proj_a),
                    (TrunkFamily::GdnB, g.in_proj_b),
                    (TrunkFamily::GdnOut, g.output),
                ]
            }
            (None, Some(qsa)) => {
                let q = qsa_desc(weights, qsa)?;
                [
                    (TrunkFamily::QsaIdx, q.indexer_qk),
                    (TrunkFamily::QsaQ, q.q),
                    (TrunkFamily::QsaK, q.k),
                    (TrunkFamily::QsaV, q.v),
                    (TrunkFamily::QsaO, q.output),
                ]
            }
            (None, None) => return Err(invalid("trunk layer without attention weights")),
        };
        for (family, w) in matrices {
            if w.dtype != DType::MQ4G256V2 {
                other += 1;
                continue;
            }
            mq4 += 1;
            if gpu.qwen4_trunk_sym_check(w.buf, w.m, w.k)? {
                verified += 1;
            }
            if bits & family.bit() != 0 {
                a4 += 1;
            }
        }
    }
    eprintln!(
        "  qwen4 IU4 trunk: mask={} ({}) MQ4 {verified}/{mq4} verified symmetric, A4 {a4} projections, MQ6/other {other}",
        mask.canonical(),
        match route.source {
            rdna_compute::TrunkIu4MaskSource::Env => "env override".to_owned(),
            rdna_compute::TrunkIu4MaskSource::ArtifactDefault(default) =>
                format!("artifact default {}", default.marker()),
            rdna_compute::TrunkIu4MaskSource::Off => "off".to_owned(),
        }
    );
    if mq4 == 0 || verified != mq4 || a4 == 0 {
        return Ok((folds, Vec::new()));
    }
    let result = (|| {
        for (index, layer) in weights.layer_refs.iter().enumerate() {
            let Some(gdn) = &layer.gdn else {
                folds.push(None);
                continue;
            };
            let g = gdn_desc(weights, gdn)?;
            let k = g.z.k;
            let zba = TrunkFamily::GdnZ.bit() | TrunkFamily::GdnA.bit() | TrunkFamily::GdnB.bit();
            if mask.bits(index) & zba != zba
                || [g.z.dtype, g.in_proj_a.dtype, g.in_proj_b.dtype]
                    .iter()
                    .any(|d| *d != DType::MQ4G256V2)
                || g.in_proj_a.k != k
                || g.in_proj_b.k != k
                || g.in_proj_a.m != 48
                || g.in_proj_b.m != 48
            {
                folds.push(None);
                continue;
            }
            let row_bytes = k / 256 * rdna_compute::MQ4V2_GROUP_BYTES;
            let mut rows = vec![0u8; (g.z.m + 256) * row_bytes];
            for (w, start) in [(&g.z, 0), (&g.in_proj_b, g.z.m), (&g.in_proj_a, g.z.m + 48)] {
                let span = start * row_bytes..(start + w.m) * row_bytes;
                gpu.hip.memcpy_dtoh(&mut rows[span], &w.buf.buf)?;
            }
            folds.push(Some(gpu.upload_raw(&rows, &[rows.len()])?));
        }
        Ok::<(), Qwen4GpuForwardError>(())
    })();
    if let Err(error) = result {
        for tensor in folds.into_iter().flatten() {
            let _ = gpu.free_tensor(tensor);
        }
        return Err(error);
    }
    // Every MQ4V2 matrix the Qwen4 forward hands the dense IU4 GEMMs is a
    // trunk projection verified above (routed experts take their own verified
    // route), so the symmetric-grid contract those SET kernels read
    // (`mq4v2_symmetric`: V2B / X5 / symfold) holds even though Qwen4
    // artifacts record expert symmetry under their own metadata key.
    gpu.mq4v2_symmetric = true;
    gpu.qwen4_trunk_iu4 = true;
    let layer_bits = (0..weights.layer_refs.len()).map(|layer| mask.bits(layer)).collect();
    Ok((folds, layer_bits))
}

impl Qwen4GpuForward {
    /// Refuse, before any allocation, a profile this forward cannot run: a
    /// full-attention layer needs the QSA RoPE kernels' hard-coded base.  PLE
    /// is optional: a profile without it has no PLE step, rows, staging,
    /// prefetch or lookahead in the forward.
    pub(crate) fn admit_profile(config: &Qwen4Config) -> Result<(), Qwen4GpuForwardError> {
        let id = config.profile.id();
        if config.layer_types.contains(&LayerType::FullAttention)
            && config.rope_theta != QWEN4_QSA_KERNEL_ROPE_THETA
        {
            return Err(invalid(format!(
                "profile {id} rope_theta {} differs from the {} base hard-coded in the QSA RoPE kernels",
                config.rope_theta, QWEN4_QSA_KERNEL_ROPE_THETA
            )));
        }
        Ok(())
    }

    pub fn new(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
    ) -> Result<Self, Qwen4GpuForwardError> {
        Self::new_inner(
            gpu, bundle, max_chunk, mode,
            #[cfg(target_os = "linux")]
            None,
        )
    }

    /// Prepare the stock MQ6 trunk before the bundle swaps any overlay tensors.
    #[cfg(target_os = "linux")]
    pub(crate) fn prepare_npu_trunk(
        gpu: &mut Gpu, bundle: &Qwen4Bundle, pre_overlay: bool, rows: usize, mode: Qwen4ForwardScratchMode,
    ) -> Result<Option<npu_forward::PreparedTrunk>, Qwen4GpuForwardError> {
        npu_forward::NpuForward::prepare_stock(gpu, bundle, pre_overlay, rows, mode)
    }

    pub(crate) fn new_inner(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
        #[cfg(target_os = "linux")]
        npu_prepared: Option<Option<npu_forward::PreparedTrunk>>,
    ) -> Result<Self, Qwen4GpuForwardError> {
        Self::admit_profile(&bundle.config)?;
        let flash_next = bundle.config.profile.is_flash_next();
        let (scratch, host_token_bytes, host_ple_bytes) =
            Qwen4GpuForwardScratch::new(gpu, &bundle.config, max_chunk, mode)?;
        let mut moe = Vec::with_capacity(bundle.config.num_hidden_layers);
        let mut decode_q8 = Vec::new();
        let mut route_trace = None;
        let mut expert_stage = None;
        let mut ep: Option<Qwen4ForwardEp> = None;
        let mut trunk_zba = Vec::new();
        let mut trunk_a4 = Vec::new();
        #[cfg(target_os = "linux")]
        let mut npu_trunk = None;
        let result = (|| {
            if let Some(placement) = bundle.weights.ep_placement() {
                let state = ep.insert(Qwen4ForwardEp::new(gpu, &bundle.config, placement)?);
                for layer in &bundle.weights.layer_refs {
                    if let Some(weights) = layer.ffn.moe() {
                        moe.push(Qwen4MoeLayerRuntime::from_moe_ep(
                            gpu,
                            &bundle.weights,
                            weights,
                            layer.layer,
                            &bundle.config,
                            placement,
                            &state.dummy,
                        )?);
                    }
                }
            } else {
                for layer in &bundle.weights.layer_refs {
                    if let Some(weights) = layer.ffn.moe() {
                        moe.push(Qwen4MoeLayerRuntime::from_moe(
                            gpu,
                            &bundle.weights,
                            weights,
                            layer.layer,
                            &bundle.config,
                        )?);
                    }
                }
            }
            let compact = mode == Qwen4ForwardScratchMode::FnSymGfx1201;
            if compact {
                // The compact plan was admitted before any weight was
                // loaded; it is valid only for the exact route it sized.
                if gpu.arch != "gfx1201"
                    || !gpu.flags.moe_grouped_gemm
                    || !gpu.qwen4_moe_sym_iu4_requested()
                {
                    return Err(invalid(
                        "compact scratch requires gfx1201 with the grouped MoE GEMM and the \
                         symmetric IU4 route requested",
                    ));
                }
                for layer in &bundle.weights.layer_refs {
                    if let Some(ple) = layer.ple.as_ref() {
                        let ple = ple_desc(&bundle.weights, ple)?;
                        if ple.key.dtype != DType::BF16 || ple.value.dtype != DType::BF16 {
                            return Err(invalid(
                                "compact scratch requires BF16 PLE key/value projections",
                            ));
                        }
                    }
                }
            }
            // Symmetric IU4 MoE (`HIPFIRE_QWEN4_MOE_SYM_IU4`, default on for
            // gfx1151 and gfx1201, `=0` off): verify each layer's final
            // expert headers once, before any capture, and size the route's
            // scratch for the chunk cap. Layers that fail (every layer of an
            // asymmetric artifact) keep the F16 route; nothing runs (and no new
            // module loads) unless the route was requested.
            // (An EP rank verifies its full chunk tables in `attach_ep`, once
            // every peer's experts are mapped.)
            // Owner-routed chunk MoE (`HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`): its grouped-row
            // buffer, masked tile ids and sym scratch exist before the mesh enables peers.
            if let Some(state) = ep.as_mut() {
                if qwen4_ep_owner_routed_prefill_enabled() {
                    state.tile = Qwen4TileRoute::admit(
                        gpu,
                        &bundle.config,
                        state.placement,
                        &scratch,
                        mode,
                        max_chunk,
                        flash_next,
                    )?;
                }
            }
            if ep.is_none() && !moe.is_empty() && gpu.qwen4_moe_sym_iu4_requested() {
                let mut verified = 0usize;
                for layer in &mut moe {
                    verified += usize::from(layer.verify_symmetric(gpu, &bundle.config)?);
                }
                eprintln!(
                    "  qwen4 symmetric IU4 MoE: {verified}/{} layers verified symmetric",
                    moe.len()
                );
                // The compact scratch has no buffer for the F32 route of a
                // layer that is not symmetric: never run it silently.
                if compact && verified != moe.len() {
                    return Err(invalid(format!(
                        "compact scratch was selected from the artifact's source headers, but \
                         the GPU check verified only {verified}/{} layers symmetric",
                        moe.len()
                    )));
                }
                if verified > 0 {
                    gpu.reserve_qwen4_moe_sym(
                        max_chunk,
                        bundle.config.num_experts_per_tok,
                        bundle.config.num_experts,
                        bundle.config.hidden_size,
                        bundle.config.moe_intermediate_size,
                    )?;
                }
            }
            if flash_next {
                (trunk_zba, trunk_a4) = prepare_trunk_iu4(gpu, bundle)?;
                if ep.is_none()
                    || (ep_lever_on(hipfire_config::developer_var(Qwen4ExpertStage::EP_ENV).ok().as_deref())
                        && ep.as_ref().is_some_and(|state| state.tile.is_some()))
                {
                    expert_stage = Qwen4ExpertStage::new(gpu, bundle, &moe)?;
                }
            } else {
                gpu.qwen4_trunk_iu4 = false;
            }
            #[cfg(target_os = "linux")]
            { npu_trunk = npu_forward::NpuForward::load(gpu, bundle, &scratch, max_chunk, mode, npu_prepared)?; }
            // The Q8_0 decode copies are the Flash-Next decode recipe (they
            // re-quantize its shared-expert and HC matrices); a developer
            // profile decodes the source BF16 weights, so decode and prefill
            // share one numeric.
            let mut source_refs = Vec::new();
            if flash_next {
                for layer in &bundle.weights.layer_refs {
                    let moe = layer
                        .ffn
                        .moe()
                        .ok_or_else(|| invalid("Flash-Next layer without an MoE FFN"))?;
                    source_refs.extend([
                        &layer.attn_hyper.input_mix_down,
                        &layer.attn_hyper.input_mix_up,
                        &layer.mlp_hyper.input_mix_down,
                        &layer.mlp_hyper.input_mix_up,
                        &moe.shared_gate,
                        &moe.shared_up,
                        &moe.shared_down,
                    ]);
                }
            }
            let sources = source_refs
                .into_iter()
                .map(|reference| dense_ref(&bundle.weights, reference))
                .collect::<Result<Vec<_>, _>>()?;
            if gpu.arch_caps.is_gfx1151()
                && sources
                    .iter()
                    .all(|weight| weight.dtype == DType::BF16 && weight.k % 32 == 0)
            {
                for weight in sources {
                    decode_q8.push(gpu.quantize_bf16_q8_0(weight.buf, weight.m, weight.k)?);
                }
            }
            if let Some(path) = hipfire_config::developer_var(ROUTE_TRACE_ENV)
                .ok()
                .filter(|_| flash_next)
            {
                let slots = bundle.config.num_hidden_layers * bundle.config.num_experts_per_tok;
                let file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .map_err(|error| invalid(format!("{ROUTE_TRACE_ENV}={path}: {error}")))?;
                route_trace = Some(RouteTrace {
                    device: gpu.alloc_tensor(&[slots], DType::F32)?,
                    host: vec![0u8; slots * 4],
                    out: std::io::BufWriter::new(file),
                });
                eprintln!("  qwen4 routing trace -> {path}");
            }
            Ok::<(), Qwen4GpuForwardError>(())
        })();
        if let Err(error) = result {
            #[cfg(target_os = "linux")]
            if let Some(npu) = npu_trunk { let _ = npu.free(gpu); }
            if let Some(stage) = expert_stage {
                let _ = stage.free_gpu(gpu);
            }
            for layer in moe {
                let _ = layer.free_gpu(gpu);
            }
            if let Some(ep) = ep {
                let _ = ep.free_gpu(gpu);
            }
            gpu.qwen4_trunk_iu4 = false;
            for tensor in decode_q8.into_iter().chain(trunk_zba.into_iter().flatten()) {
                let _ = gpu.free_tensor(tensor);
            }
            if let Some(trace) = route_trace {
                let _ = gpu.free_tensor(trace.device);
            }
            let _ = scratch.free_gpu(gpu);
            return Err(error);
        }
        Ok(Self {
            expert_stage,
            ep,
            scratch,
            host_token_bytes,
            uploaded_id_rows: 0,
            host_ple: PleHostStage::new(
                host_ple_bytes,
                ple_async_upload_enabled(
                    hipfire_config::developer_var(PLE_ASYNC_UPLOAD_ENV)
                        .ok()
                        .as_deref(),
                ),
            ),
            ple_lookahead: Vec::new(),
            ple_ahead: None,
            token_readback: None,
            moe,
            decode_q8,
            route_trace,
            trunk_zba,
            trunk_a4,
            qsa_tap: None,
            lane_scope: false,
            prefill_scope: false,
            #[cfg(target_os = "linux")]
            npu_trunk,
        })
    }

    /// Whether planned pairs run layer-major through the pair executor (`on`,
    /// `auto`, and `emulate` with a bank-B scratch): the capture capacity covers
    /// both banks of a group.
    pub(crate) fn npu_trunk_pipelined(&self) -> bool {
        #[cfg(target_os = "linux")]
        { self.npu_trunk.as_ref().is_some_and(|n|n.pipelined()) }
        #[cfg(not(target_os = "linux"))]
        { false }
    }

    pub(crate) fn npu_trunk_planned(&self) -> bool {
        #[cfg(target_os = "linux")]
        { self.npu_trunk.is_some() }
        #[cfg(not(target_os = "linux"))]
        { false }
    }

    pub(crate) fn begin_npu_trunk_request(&mut self) {
        #[cfg(target_os = "linux")]
        if let Some(npu)=&mut self.npu_trunk { npu.begin_request(); }
    }

    pub(crate) fn plan_npu_trunk_request(&mut self, rows: usize, position: usize) {
        #[cfg(target_os = "linux")]
        if let Some(npu) = &mut self.npu_trunk { npu.plan_request(rows, position); }
        #[cfg(not(target_os = "linux"))]
        let _ = (rows, position);
    }

    /// Rows of banks A and B of the pair the plan starts at `position`.
    pub(crate) fn npu_trunk_pair(&self, position: usize, remaining: usize, chunk: usize) -> Option<[usize; 2]> {
        #[cfg(target_os = "linux")]
        { self.npu_trunk.as_ref().and_then(|n| n.selected_pair(position, remaining, chunk)) }
        #[cfg(not(target_os = "linux"))]
        { let _ = (position, remaining, chunk); None }
    }

    pub(crate) fn note_npu_trunk_pair(&mut self, rows: [usize; 2], position: usize) {
        #[cfg(target_os = "linux")]
        if let Some(npu)=&mut self.npu_trunk {npu.note_pair(rows,position);}
        #[cfg(not(target_os = "linux"))]
        let _=(rows,position);
    }

    /// Before a forward that is not this owner's chunk program (the lane
    /// step) reads any expert: restore the donor layers a failed staged
    /// forward may have left overwritten. A no-op without G2 staging.
    pub(crate) fn restore_expert_stage(&self, gpu: &mut Gpu) -> Result<(), Qwen4GpuForwardError> {
        match self.expert_stage.as_ref() {
            Some(stage) => stage.restore_blocking(gpu),
            None => Ok(()),
        }
    }

    /// This rank's expert-parallel placement, `None` on the single-device route.
    pub(crate) fn ep_placement(&self) -> Option<Qwen4EpPlacement> {
        self.ep.as_ref().map(|ep| ep.placement)
    }

    /// Device pointers of this rank's compact routed-expert tensors, one per
    /// trunk MoE layer in layer order (what every peer's full table names).
    pub(crate) fn ep_expert_sources(
        &self,
        bundle: &Qwen4Bundle,
    ) -> Result<Vec<Qwen4PeerExpertSource>, Qwen4GpuForwardError> {
        if self.ep.is_none() {
            return Err(invalid("EP expert sources of a forward without an EP placement"));
        }
        let mut sources = Vec::with_capacity(self.moe.len());
        for layer in &bundle.weights.layer_refs {
            if let Some(weights) = layer.ffn.moe() {
                let gate_up = bundle.weights.resident(&weights.experts_gate_up)?;
                let down = bundle.weights.resident(&weights.experts_down)?;
                sources.push(Qwen4PeerExpertSource {
                    gate_up_base: gate_up.buf.as_ptr() as usize,
                    gate_up_bytes: gate_up.buf.size(),
                    down_base: down.buf.as_ptr() as usize,
                    down_bytes: down.buf.size(),
                });
            }
        }
        if sources.len() != self.moe.len() {
            return Err(invalid(format!(
                "EP expert sources cover {} layers, the forward has {}",
                sources.len(),
                self.moe.len()
            )));
        }
        Ok(sources)
    }

    /// The expanded-row gather buffers, indexed by [`qwen4_ep_buffer_index`].
    pub(crate) fn ep_gather_buffers(&self) -> Option<[&GpuTensor; QWEN4_EP_GATHER_BUFFERS]> {
        self.ep
            .as_ref()
            .map(|ep| std::array::from_fn(|index| &ep.buffers[index]))
    }

    /// This rank's EP binding for a head MoE runtime: its placement and the
    /// zero dummy expert (`None` on the single-device route).
    pub(crate) fn ep_binding(&self) -> Option<(Qwen4EpPlacement, &Qwen4EpDummyExpert)> {
        self.ep.as_ref().map(|ep| (ep.placement, &ep.dummy))
    }

    /// What an EP rank's MTP head borrows from this forward at call time: the
    /// row gather and the gather buffers (`None` outside EP or before
    /// `attach_ep`).
    pub(crate) fn mtp_ep_gather(&self) -> Option<MtpEpGather<'_>> {
        let ep = self.ep.as_ref()?;
        Some(MtpEpGather {
            hook: ep.gather.as_deref()?,
            buffers: std::array::from_fn(|index| &ep.buffers[index]),
        })
    }

    /// What the rank's row-gather hook needs to run balanced ownership: `None`
    /// when `HIPFIRE_EP_BALANCED_OWNERS=0` was set at [`Self::attach_ep`]
    /// (default on under EP) or no trunk MoE layer is host-mapped. The
    /// `slot_owner` view is non-owning; this forward frees the buffer.
    pub(crate) fn balanced_plan(&self) -> Option<Qwen4BalancedPlan> {
        let ep = self.ep.as_ref()?;
        let owners = ep.slot_owner.as_ref()?;
        let layers: Vec<(usize, Qwen4EpOwnerTables)> = ep
            .balanced
            .iter()
            .flatten()
            .map(|layer| (layer.label, layer.tables))
            .collect();
        if layers.is_empty() {
            return None;
        }
        Some(Qwen4BalancedPlan {
            layers,
            slot_owner: GpuTensor {
                // SAFETY: a `Borrowed` alias of the buffer `Qwen4ForwardEp`
                // owns until `free_gpu`; the hook that holds it lives inside
                // that forward and is never used after the forward is freed.
                buf: unsafe { owners.buf.alias() },
                shape: owners.shape.clone(),
                dtype: owners.dtype,
            },
        })
    }

    /// After every rank is loaded and P2P is enabled on every rank: build this
    /// rank's full (P2P) chunk tables from all ranks' sources
    /// (`peers[rank][moe_layer]`), take the symmetric-IU4 verdict a single
    /// device takes at construction, and install the row gather the one-row and few-row
    /// forwards use.
    pub(crate) fn attach_ep(
        &mut self,
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        peers: &[Vec<Qwen4PeerExpertSource>],
        gather: Arc<dyn MoeRowGather + Send + Sync>,
    ) -> Result<(), Qwen4GpuForwardError> {
        let layers = self.moe.len();
        let max_chunk = self.scratch.max_chunk;
        // The EP decode runtime's route-selecting properties per MoE ordinal
        // (host-mapped, symmetric, all-MQ4 gate/up): a balanced layer's live
        // runtime must select the same sealed route.
        let moe_traits: Vec<(bool, bool, bool)> = self
            .moe
            .iter()
            .map(|moe| (moe.host_mapped, moe.symmetric, moe.experts_all_gate_up_mq4))
            .collect();
        let Some(ep) = self.ep.as_mut() else {
            return Err(invalid("attach_ep on a forward without an EP placement"));
        };
        if !ep.moe_full.is_empty() || ep.gather.is_some() {
            return Err(invalid("EP is already attached"));
        }
        let placement = ep.placement;
        if peers.len() != placement.ranks {
            return Err(invalid(format!(
                "attach_ep got {} peer source sets for {} ranks",
                peers.len(),
                placement.ranks
            )));
        }
        for (rank, sources) in peers.iter().enumerate() {
            if sources.len() != layers {
                return Err(invalid(format!(
                    "attach_ep rank {rank} lists {} MoE layers, expected {layers}",
                    sources.len()
                )));
            }
        }
        let config = &bundle.config;
        let balanced_on = qwen4_ep_balanced_owners_enabled();
        let dummy = ep.dummy.source();
        let layer_peers_of = |ordinal: usize| -> Result<Vec<Qwen4PeerExpertSource>, Qwen4GpuForwardError> {
            peers
                .iter()
                .map(|sources| {
                    sources.get(ordinal).copied().ok_or_else(|| {
                        invalid(format!("attach_ep is missing MoE layer {ordinal}"))
                    })
                })
                .collect()
        };
        let mut full: Vec<Qwen4MoeLayerRuntime> = Vec::with_capacity(layers);
        let mut balanced: Vec<Option<Qwen4BalancedLayer>> = Vec::new();
        let mut slot_owner: Option<GpuTensor> = None;
        let result = (|| -> Result<(), Qwen4GpuForwardError> {
            let mut ordinal = 0usize;
            for layer in &bundle.weights.layer_refs {
                if let Some(weights) = layer.ffn.moe() {
                    let layer_peers = layer_peers_of(ordinal)?;
                    full.push(Qwen4MoeLayerRuntime::from_moe_full(
                        gpu,
                        &bundle.weights,
                        weights,
                        layer.layer,
                        config,
                        placement,
                        &layer_peers,
                    )?);
                    ordinal += 1;
                }
            }
            // The single-device construction's symmetric-IU4 verdict, taken
            // on the full tables so a chunk forward takes the same route
            // (symmetric IU4 or F16) as one device.
            if !full.is_empty() && gpu.qwen4_moe_sym_iu4_requested() {
                let mut verified = 0usize;
                for layer in &mut full {
                    verified += usize::from(layer.verify_symmetric(gpu, config)?);
                }
                eprintln!(
                    "  qwen4 symmetric IU4 MoE (ep rank {}/{}): {verified}/{} layers verified symmetric",
                    placement.rank,
                    placement.ranks,
                    full.len()
                );
                if verified > 0 {
                    gpu.reserve_qwen4_moe_sym(
                        max_chunk,
                        config.num_experts_per_tok,
                        config.num_experts,
                        config.hidden_size,
                        config.moe_intermediate_size,
                    )?;
                }
            }
            // Balanced ownership (default on under EP; `HIPFIRE_EP_BALANCED_OWNERS=0` disables): every
            // host-mapped trunk layer gets a live runtime of the same full
            // addressing, whose tables the per-call owner assignment
            // rewrites. Nothing is built, allocated or loaded when off.
            if balanced_on {
                let mut ordinal = 0usize;
                for layer in &bundle.weights.layer_refs {
                    let Some(weights) = layer.ffn.moe() else {
                        continue;
                    };
                    let (host_mapped, symmetric, all_mq4) = moe_traits[ordinal];
                    if !host_mapped {
                        balanced.push(None);
                        ordinal += 1;
                        continue;
                    }
                    let owners = match slot_owner.as_ref() {
                        Some(owners) => owners.buf.as_ptr() as usize,
                        None => {
                            let tensor = gpu.zeros(
                                &[QWEN4_EP_GATHER_MAX_ROWS * config.num_experts_per_tok],
                                DType::F32,
                            )?;
                            let address = tensor.buf.as_ptr() as usize;
                            slot_owner = Some(tensor);
                            address
                        }
                    };
                    let live = Qwen4MoeLayerRuntime::from_moe_full(
                        gpu,
                        &bundle.weights,
                        weights,
                        layer.layer,
                        config,
                        placement,
                        &layer_peers_of(ordinal)?,
                    )?;
                    if live.symmetric != symmetric
                        || live.host_mapped != host_mapped
                        || live.experts_all_gate_up_mq4 != all_mq4
                    {
                        let _ = live.free_gpu(gpu);
                        return Err(invalid(format!(
                            "layer {} balanced runtime selects a different sealed route than the EP decode runtime",
                            layer.layer
                        )));
                    }
                    let tables = Qwen4EpOwnerTables {
                        full_gate_up: full[ordinal].expert_gate_up_ptrs.buf.as_ptr() as usize,
                        full_down: full[ordinal].expert_down_ptrs.buf.as_ptr() as usize,
                        live_gate_up: live.expert_gate_up_ptrs.buf.as_ptr() as usize,
                        live_down: live.expert_down_ptrs.buf.as_ptr() as usize,
                        slot_owner: owners,
                        dummy_gate_up: dummy.gate_up_base,
                        dummy_down: dummy.down_base,
                    };
                    balanced.push(Some(Qwen4BalancedLayer {
                        live,
                        label: layer.layer,
                        tables,
                    }));
                    ordinal += 1;
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            for layer in full {
                let _ = layer.free_gpu(gpu);
            }
            for layer in balanced.into_iter().flatten() {
                let _ = layer.live.free_gpu(gpu);
            }
            if let Some(tensor) = slot_owner {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(error);
        }
        if balanced_on {
            eprintln!(
                "qwen4 EP rank {}/{}: balanced ownership on {} host-mapped MoE layers",
                placement.rank,
                placement.ranks,
                balanced.iter().flatten().count()
            );
        }
        if let Some(route) = ep.tile.as_mut() {
            // The compact table carries no verdict of its own under EP (the verification
            // above ran on the full tables; same experts, same bytes): the mesh compares
            // all ranks' vectors before anything is propagated to the compact layers.
            route.verdicts = full.iter().map(|layer| layer.symmetric).collect();
        }
        ep.moe_full = full;
        ep.balanced = balanced;
        ep.slot_owner = slot_owner;
        ep.gather = Some(gather);
        Ok(())
    }

    /// This rank's grouped-row buffer (the peer reads it) and the tile capacity the
    /// exchange is registered for, or `None` when not admitted.
    pub(crate) fn ep_tile_registration(&self) -> Option<(&GpuTensor, usize)> {
        let route = self.ep.as_ref()?.tile.as_ref()?;
        Some((&self.scratch.moe_y_down_grouped, route.max_tiles))
    }

    /// The per-layer symmetric verdicts taken in `attach_ep` (`None` when not admitted).
    pub(crate) fn ep_tile_verdicts(&self) -> Option<Vec<bool>> {
        Some(self.ep.as_ref()?.tile.as_ref()?.verdicts.clone())
    }

    /// A non-owning view of the rank's owner table for the transport hook (owned by the
    /// forward, freed with it; the hook is never used after the forward is freed).
    pub(crate) fn ep_tile_owner_table_view(&self) -> Option<GpuTensor> {
        let table = &self.ep.as_ref()?.tile.as_ref()?.owner_table;
        Some(GpuTensor {
            // SAFETY: a `Borrowed` alias of the buffer `Qwen4TileRoute` owns until
            // `free_gpu`; the hook that holds it lives inside this forward.
            buf: unsafe { table.buf.alias() },
            shape: table.shape.clone(),
            dtype: table.dtype,
        })
    }

    /// Install the agreed transport: every layer's compact runtime takes the symmetric
    /// verdict (same experts, same bytes as the full tables it was taken on), so a chunk of
    /// at least 512 rows binds the compact table with the tile exchange. Refused unless
    /// every layer verified symmetric (the mesh checked every rank agrees).
    pub(crate) fn install_tile_exchange(
        &mut self,
        hook: Arc<dyn MoeTileExchange + Send + Sync>,
    ) -> Result<(), Qwen4GpuForwardError> {
        let layers = self.moe.len();
        let Some(route) = self.ep.as_mut().and_then(|ep| ep.tile.as_mut()) else {
            return Err(invalid("owner-routed prefill was not admitted on this rank"));
        };
        if route.verdicts.len() != layers || !route.verdicts.iter().all(|&symmetric| symmetric) {
            return Err(invalid(
                "owner-routed prefill needs every layer verified symmetric on the full tables",
            ));
        }
        route.hook = Some(hook);
        for layer in &mut self.moe {
            layer.symmetric = true;
        }
        Ok(())
    }

    /// Name the tokens that follow the next forward (at most one chunk is
    /// kept); see the `ple_lookahead` field.
    pub(crate) fn set_ple_lookahead(&mut self, tokens: &[u32]) {
        self.ple_lookahead.clear();
        self.ple_lookahead
            .extend_from_slice(&tokens[..tokens.len().min(self.scratch.max_chunk)]);
    }

    /// Teardown only: free the forward right before the bundle weights it
    /// ran on (`Qwen4Bundle::free_gpu`). After a failed staged forward
    /// whose recovery also failed, two resident expert layers may still hold
    /// another layer's bytes, and nothing here restores them.
    pub fn free_gpu(self, gpu: &mut Gpu) -> Result<(), hip_bridge::HipError> {
        let Qwen4GpuForward {
            scratch,
            host_ple,
            moe,
            expert_stage,
            ep,
            token_readback,
            decode_q8,
            route_trace,
            trunk_zba,
            #[cfg(target_os = "linux")]
            npu_trunk,
            ..
        } = self;
        // Drain both streams before releasing stage tables, events or scratch.
        // A donor a failed forward left overwritten is not restored: the
        // bundle attaches its forward once and frees it only right before
        // the weights (`Qwen4Bundle::free_gpu`), so nothing reads it again.
        if expert_stage.is_some() || ep.is_some() {
            gpu.hip.device_synchronize()?;
        }
        // The PLE staging bytes may still be read by an enqueued upload:
        // retire it before the device buffers and the host bytes go.
        let mut first = host_ple.free_gpu(gpu);
        #[cfg(target_os = "linux")]
        if let Some(npu) = npu_trunk {
            if let Err(e) = npu.free(gpu) { first.get_or_insert(e); }
        }
        if let Some(error) = scratch.free_gpu(gpu) {
            first.get_or_insert(error);
        }
        if let Some(stage) = expert_stage {
            if let Some(error) = stage.free_gpu(gpu) {
                first.get_or_insert(error);
            }
        }
        if let Some(trace) = route_trace {
            if let Err(error) = gpu.free_tensor(trace.device) {
                first.get_or_insert(error);
            }
        }
        gpu.qwen4_trunk_iu4 = false;
        for tensor in decode_q8.into_iter().chain(trunk_zba.into_iter().flatten()) {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        if let Some(readback) = token_readback {
            for result in [
                gpu.hip.stream_synchronize(&readback.stream),
                gpu.hip.stream_destroy(readback.stream),
                gpu.hip.event_destroy(readback.event),
            ] {
                if let Err(error) = result {
                    first.get_or_insert(error);
                }
            }
        }
        for layer in moe {
            if let Some(error) = layer.free_gpu(gpu) {
                if first.is_none() {
                    first = Some(error);
                }
            }
        }
        if let Some(ep) = ep {
            if let Some(error) = ep.free_gpu(gpu) {
                first.get_or_insert(error);
            }
        }
        first.map_or(Ok(()), Err)
    }

    /// Append the traced forward's routed expert ids (see [`RouteTrace`]).
    /// Synchronizes the device: tracing is a measurement mode.
    fn write_route_trace(
        &mut self,
        gpu: &mut Gpu,
        position: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        use std::io::Write;
        let Some(trace) = self.route_trace.as_mut() else {
            return Ok(());
        };
        gpu.hip.device_synchronize()?;
        gpu.hip.memcpy_dtoh(&mut trace.host, &trace.device.buf)?;
        let mut line = position.to_string();
        for id in trace.host.chunks_exact(4) {
            line.push(' ');
            line.push_str(&i32::from_ne_bytes([id[0], id[1], id[2], id[3]]).to_string());
        }
        line.push('\n');
        trace
            .out
            .write_all(line.as_bytes())
            .and_then(|_| trace.out.flush())
            .map_err(|error| invalid(format!("{ROUTE_TRACE_ENV}: {error}")))
    }

    /// Token-only request checks shared by every forward entry: non-empty,
    /// within context capacity, inside the embedding vocabulary.
    fn validate_tokens(
        &self,
        bundle: &Qwen4Bundle,
        tokens: &[u32],
    ) -> Result<(), Qwen4GpuForwardError> {
        if tokens.is_empty() {
            return Err(invalid("Qwen4 forward cannot process an empty token slice"));
        }
        let config = &bundle.config;
        let end_position = bundle
            .state
            .position
            .checked_add(tokens.len())
            .ok_or_else(|| invalid("Qwen4 forward position overflows"))?;
        if end_position > bundle.state.max_seq_len {
            return Err(invalid(format!(
                "Qwen4 forward end position {end_position} exceeds context capacity {}",
                bundle.state.max_seq_len
            )));
        }
        if tokens
            .iter()
            .copied()
            .any(|token| (token as usize) >= config.vocab_size)
        {
            return Err(invalid(
                "Qwen4 token id is outside the embedding vocabulary",
            ));
        }
        Ok(())
    }

    fn validate_request(
        &self,
        bundle: &Qwen4Bundle,
        tokens: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
        wide_hidden_capture: Option<&GpuTensor>,
        output_rows: Qwen4OutputRows,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.validate_tokens(bundle, tokens)?;
        let config = &bundle.config;
        let requested_rows = output_rows.count(tokens.len());
        let expected_logits = requested_rows
            .checked_mul(config.vocab_size)
            .ok_or_else(|| invalid("Qwen4 logits shape overflows"))?;
        if logits.dtype != DType::F32 || logits.numel() != expected_logits {
            return Err(invalid(format!(
                "logits must be F32 with {expected_logits} elements for {output_rows:?} output"
            )));
        }
        if let Some(top1) = top1 {
            let expected = requested_rows
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or_else(|| invalid("Qwen4 top1 shape overflows"))?;
            if top1.dtype != DType::Raw || top1.buf.size() < expected {
                return Err(invalid(
                    "top1 output must be Raw with one i32 per requested output row",
                ));
            }
        }
        if let Some(capture) = wide_hidden_capture {
            let expected = tokens
                .len()
                .checked_mul(program_dims(config).wide())
                .ok_or_else(|| invalid("Qwen4 wide hidden capture shape overflows"))?;
            if capture.dtype != DType::F32 || capture.numel() < expected {
                return Err(invalid(format!(
                    "wide hidden capture must be F32 with at least {expected} elements"
                )));
            }
        }
        Ok(())
    }

    fn preflight_output_resources(
        &self,
        bundle: &Qwen4Bundle,
        rows: usize,
        logits: &GpuTensor,
        requested_rows: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        let config = &bundle.config;
        let dims = program_dims(config);
        let final_hyper = hyper_read_desc(&bundle.weights, &bundle.weights.root.final_hyper)?;
        let lm_head = dense_ref(&bundle.weights, &bundle.weights.root.lm_head)?;
        let preflight_scratch = layer_scratch(&self.scratch, &self.scratch.router_logits);
        validate_final_hyper(
            dims,
            &final_hyper,
            &self.scratch.streams,
            &preflight_scratch,
            rows,
        )
        .map_err(|error| {
            Qwen4GpuForwardError::Dispatch(format!("validate Qwen4 final hyper: {error:?}"))
        })?;
        let hidden_elements = rows
            .checked_mul(config.hidden_size)
            .ok_or_else(|| invalid("Qwen4 final hidden shape overflows"))?;
        let final_hidden = view(&self.scratch.hc_mixed, 0, hidden_elements);
        validate_lm_head(&lm_head, &final_hidden, logits, rows, requested_rows).map_err(
            |error| Qwen4GpuForwardError::Dispatch(format!("validate Qwen4 LM head: {error:?}")),
        )?;
        Ok(())
    }

    pub fn forward_token(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        token: u32,
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.forward_chunk_inner(
            bundle,
            gpu,
            std::slice::from_ref(&token),
            None,
            Some(logits),
            top1,
            None,
            Qwen4OutputPolicy::Rows(Qwen4OutputRows::Final),
        )
    }

    /// Copy the single wide HC stream row the last ordinary one-token
    /// [`Self::forward_token`] left in `scratch.streams` into `destination`
    /// as F32, with the conversion `forward_chunk` applies for a wide hidden
    /// capture (BF16 stream bits widen, F32 streams copy). The final hyper
    /// reads `streams` into other scratch, so the row is the trunk's output on
    /// the HIP and the retained-replay route alike. `destination` must be F32
    /// with exactly `hc_count * hidden_size` elements. Calibration only; the
    /// row is valid until the next forward writes `streams`.
    pub(crate) fn copy_last_wide_hidden_to(
        &self,
        gpu: &mut Gpu,
        config: &Qwen4Config,
        destination: &GpuTensor,
    ) -> Result<(), Qwen4GpuForwardError> {
        let wide = program_dims(config).wide();
        if destination.dtype != DType::F32 || destination.numel() != wide {
            return Err(invalid(format!(
                "wide hidden destination must be F32 with exactly {wide} elements"
            )));
        }
        if self.scratch.streams.numel() < wide {
            return Err(invalid("Qwen4 stream scratch is narrower than one wide row"));
        }
        let source = view(&self.scratch.streams, 0, wide);
        let destination = f32_view(destination, 0, wide);
        if gpu.qwen4_bf16_streams(1) {
            rdna_compute::tensor_ops::hc_state_bf16_to_f32(gpu, &source, &destination, wide)?;
        } else {
            gpu.copy_d2d(&source, &destination, source.byte_size())?;
        }
        Ok(())
    }

    /// [`Self::forward_token`] of `token`, or (`None`) of the GPU argmax of
    /// `logits` as the previous forward left them; returns the token. The
    /// argmax is taken only after this forward's step program is built, and
    /// on the HIP body it stays on the device until the PLE layer, so neither
    /// the host build nor the readback leaves the GPU idle.
    pub fn forward_token_or_argmax(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        token: Option<u32>,
        logits: &GpuTensor,
    ) -> Result<u32, Qwen4GpuForwardError> {
        let placeholder = token.unwrap_or(0);
        self.forward_chunk_inner(
            bundle,
            gpu,
            std::slice::from_ref(&placeholder),
            token.is_none().then_some(logits),
            Some(logits),
            None,
            None,
            Qwen4OutputPolicy::Rows(Qwen4OutputRows::Final),
        )?;
        // The forward committed its token as the newest PLE history slot.
        Ok(bundle.state.ple_history.previous()[1])
    }

    /// Run bounded forward tiles, writing all logits rows or only the last.
    /// Final-only intermediate tiles still commit model state but skip the head.
    pub fn forward_chunk(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        tokens: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
        wide_hidden_capture: Option<&GpuTensor>,
        output_rows: Qwen4OutputRows,
    ) -> Result<(), Qwen4GpuForwardError> {
        // The caller's lookahead names what follows this whole request, so it
        // belongs to the final tile; earlier tiles look ahead to the next tile.
        let mut caller_lookahead = Some(std::mem::take(&mut self.ple_lookahead));
        self.validate_request(
            bundle,
            tokens,
            logits,
            top1,
            wide_hidden_capture,
            output_rows,
        )?;
        self.plan_npu_trunk_request(tokens.len(), bundle.state.position);
        // Start the first tile's PLE rows now, ahead of the output-resource
        // preflight.
        self.prefetch_first_tile_ple(bundle, tokens);
        let vocab = bundle.config.vocab_size;
        let max_chunk = self.scratch.max_chunk;
        let preflight_rows = if output_rows == Qwen4OutputRows::Final {
            (tokens.len() - 1) % max_chunk + 1
        } else {
            tokens.len().min(max_chunk)
        };
        let preflight_output_rows = output_rows.count(preflight_rows);
        let preflight_logits = logits.sub_offset(0, preflight_output_rows * vocab);
        self.preflight_output_resources(
            bundle,
            preflight_rows,
            &preflight_logits,
            preflight_output_rows,
        )?;
        let capture_width = wide_hidden_capture.map(|_| program_dims(&bundle.config).wide());
        let mut offset = 0usize;
        while offset < tokens.len() {
            // A planned pair (`on`, `auto` and `emulate` alike) runs layer-major
            // through the pair executor; everything else is an ordinary tile.
            #[cfg(target_os = "linux")]
            if let Some(banks)=self.npu_trunk_pair(bundle.state.position,tokens.len()-offset,max_chunk)
            {
                let rows=banks[0]+banks[1];
                self.note_npu_trunk_pair(banks,bundle.state.position);
                let final_pair=self.set_tile_ple_lookahead(tokens,offset,rows,&mut caller_lookahead);
                let output=if output_rows==Qwen4OutputRows::All || final_pair {
                    Qwen4OutputPolicy::Rows(output_rows)
                } else { Qwen4OutputPolicy::None };
                npu_forward::run_pair_request(self,bundle,gpu,&tokens[offset..offset+rows],offset,banks,Some(logits),top1,wide_hidden_capture,output)?;
                offset+=rows;
                continue;
            }
            let rows = (tokens.len() - offset).min(max_chunk);
            // The caller's lookahead belongs to the final tile; earlier tiles
            // look ahead to the next tile.
            let final_chunk =
                self.set_tile_ple_lookahead(tokens, offset, rows, &mut caller_lookahead);
            let selected_rows = output_rows.count(rows);
            let logits_offset = if output_rows == Qwen4OutputRows::All {
                offset
                    .checked_mul(vocab)
                    .ok_or_else(|| invalid("Qwen4 chunk logits offset overflows"))?
            } else {
                0
            };
            let logits_len = selected_rows
                .checked_mul(vocab)
                .ok_or_else(|| invalid("Qwen4 chunk logits shape overflows"))?;
            let logits_chunk = logits.sub_offset(logits_offset, logits_len);
            let top1_chunk = top1
                .filter(|_| output_rows == Qwen4OutputRows::All || final_chunk)
                .map(|destination| {
                    destination.sub_offset(
                        if output_rows == Qwen4OutputRows::All {
                            offset * std::mem::size_of::<i32>()
                        } else {
                            0
                        },
                        selected_rows * std::mem::size_of::<i32>(),
                    )
                });
            let capture_chunk = wide_hidden_capture
                .zip(capture_width)
                .map(|(capture, wide)| {
                    let capture_offset = offset
                        .checked_mul(wide)
                        .ok_or_else(|| invalid("Qwen4 wide capture offset overflows"))?;
                    let capture_len = rows
                        .checked_mul(wide)
                        .ok_or_else(|| invalid("Qwen4 wide capture shape overflows"))?;
                    Ok::<_, Qwen4GpuForwardError>(capture.sub_offset(capture_offset, capture_len))
                })
                .transpose()?;
            let output_policy = if output_rows == Qwen4OutputRows::All || final_chunk {
                Qwen4OutputPolicy::Rows(output_rows)
            } else {
                Qwen4OutputPolicy::None
            };
            self.forward_chunk_inner(
                bundle,
                gpu,
                &tokens[offset..offset + rows],
                None,
                Some(&logits_chunk),
                top1_chunk.as_ref(),
                capture_chunk.as_ref(),
                output_policy,
            )?;
            offset = offset
                .checked_add(rows)
                .ok_or_else(|| invalid("Qwen4 chunk offset overflows"))?;
        }
        Ok(())
    }

    /// Run bounded forward tiles that only commit model state: every tile
    /// takes [`Qwen4OutputPolicy::None`], so no head, logits or top-1 rows are
    /// produced. With `wide_hidden_capture` each tile still writes its wide
    /// hidden rows into the capture at `offset * wide` (the trunk's output,
    /// taken before the head, so identical to [`Self::forward_chunk`]'s).
    /// Tiles are `scratch.max_chunk` rows from the start of `tokens`, exactly
    /// as [`Self::forward_chunk`] tiles them, so a prefill split at tile
    /// multiples keeps the trunk numerics byte-identical.
    pub(crate) fn forward_chunk_silent(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        tokens: &[u32],
        wide_hidden_capture: Option<&GpuTensor>,
    ) -> Result<(), Qwen4GpuForwardError> {
        let mut caller_lookahead = Some(std::mem::take(&mut self.ple_lookahead));
        self.validate_tokens(bundle, tokens)?;
        self.plan_npu_trunk_request(tokens.len(), bundle.state.position);
        let capture_width = wide_hidden_capture.map(|_| program_dims(&bundle.config).wide());
        if let Some((capture, wide)) = wide_hidden_capture.zip(capture_width) {
            let expected = tokens
                .len()
                .checked_mul(wide)
                .ok_or_else(|| invalid("Qwen4 wide hidden capture shape overflows"))?;
            if capture.dtype != DType::F32 || capture.numel() < expected {
                return Err(invalid(format!(
                    "wide hidden capture must be F32 with at least {expected} elements"
                )));
            }
        }
        self.prefetch_first_tile_ple(bundle, tokens);
        let max_chunk = self.scratch.max_chunk;
        let mut offset = 0usize;
        while offset < tokens.len() {
            #[cfg(target_os = "linux")]
            if let Some(banks)=self.npu_trunk_pair(bundle.state.position,tokens.len()-offset,max_chunk)
            {
                let rows=banks[0]+banks[1];
                self.note_npu_trunk_pair(banks,bundle.state.position);
                self.set_tile_ple_lookahead(tokens,offset,rows,&mut caller_lookahead);
                npu_forward::run_pair_request(self,bundle,gpu,&tokens[offset..offset+rows],offset,banks,None,None,wide_hidden_capture,Qwen4OutputPolicy::None)?;
                offset+=rows;
                continue;
            }
            let rows = (tokens.len() - offset).min(max_chunk);
            self.set_tile_ple_lookahead(tokens, offset, rows, &mut caller_lookahead);
            let capture_chunk = wide_hidden_capture
                .zip(capture_width)
                .map(|(capture, wide)| {
                    let capture_offset = offset
                        .checked_mul(wide)
                        .ok_or_else(|| invalid("Qwen4 wide capture offset overflows"))?;
                    let capture_len = rows
                        .checked_mul(wide)
                        .ok_or_else(|| invalid("Qwen4 wide capture shape overflows"))?;
                    Ok::<_, Qwen4GpuForwardError>(capture.sub_offset(capture_offset, capture_len))
                })
                .transpose()?;
            self.forward_chunk_inner(
                bundle,
                gpu,
                &tokens[offset..offset + rows],
                None,
                None,
                None,
                capture_chunk.as_ref(),
                Qwen4OutputPolicy::None,
            )?;
            offset = offset
                .checked_add(rows)
                .ok_or_else(|| invalid("Qwen4 chunk offset overflows"))?;
        }
        Ok(())
    }

    /// Device view of `tokens` as the ids the last forward holds at row
    /// `offset` of its `scratch.token_ids` upload, or `None` when that forward
    /// did not upload them whole from the host or they differ from `tokens`.
    /// The view borrows `scratch.token_ids`: the next forward overwrites it.
    pub(crate) fn uploaded_token_ids(&self, offset: usize, tokens: &[u32]) -> Option<GpuTensor> {
        let (byte_offset, byte_len) =
            uploaded_ids_range(self.uploaded_id_rows, &self.host_token_bytes, offset, tokens)?;
        Some(view(&self.scratch.token_ids, byte_offset, byte_len))
    }

    /// Start the first tile's PLE rows now, unless a retained lookahead
    /// already names them. The tile's forward adopts this ticket (same epoch,
    /// same ids).
    fn prefetch_first_tile_ple(&mut self, bundle: &Qwen4Bundle, tokens: &[u32]) {
        // A profile without PLE has no rows to start.
        let Some(owner) = bundle.ple.as_ref() else {
            return;
        };
        let first_ids = bundle.state.ple_history.row_ids(
            &owner.metadata,
            &tokens[..tokens.len().min(self.scratch.max_chunk)],
        );
        if !self
            .ple_ahead
            .as_ref()
            .is_some_and(|ticket| owner.rows.ticket_matches(ticket, &first_ids))
        {
            drop(self.ple_ahead.take());
            self.ple_ahead = owner
                .rows
                .prefetch(owner.rows.current_epoch(), first_ids)
                .ok();
        }
    }

    /// Name what follows the tile at `tokens[offset..offset + rows]`: the
    /// caller's lookahead for the final tile, the next tile's rows otherwise.
    /// Returns whether this is the final tile.
    fn set_tile_ple_lookahead(
        &mut self,
        tokens: &[u32],
        offset: usize,
        rows: usize,
        caller_lookahead: &mut Option<Vec<u32>>,
    ) -> bool {
        let final_chunk = offset + rows == tokens.len();
        if final_chunk {
            self.ple_lookahead = caller_lookahead.take().unwrap_or_default();
        } else {
            let next = offset + rows;
            self.ple_lookahead.clear();
            self.ple_lookahead.extend_from_slice(
                &tokens[next..(next + self.scratch.max_chunk).min(tokens.len())],
            );
        }
        final_chunk
    }

    /// Every forward runs with [`Gpu::qwen4_scope`] set, which admits the
    /// Qwen4-only MQ6 X-LDS defaults (U2/U3); the previous value is restored.
    #[allow(clippy::too_many_arguments)]
    fn forward_chunk_inner(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        tokens: &[u32],
        argmax_of: Option<&GpuTensor>,
        logits: Option<&GpuTensor>,
        top1: Option<&GpuTensor>,
        wide_hidden_capture: Option<&GpuTensor>,
        output_policy: Qwen4OutputPolicy,
    ) -> Result<(), Qwen4GpuForwardError> {
        let scope = std::mem::replace(&mut gpu.qwen4_scope, true);
        // The batch-invariant prefill profile (see `rdna_compute::prefill_invariant`):
        // every row-count routing decision of this forward sees the canonical
        // chunk size; launches keep the real row count. Restored below on every
        // return path.
        let invariant = self.prefill_scope
            && rdna_compute::prefill_invariant::prefill_pm_invariant_enabled()
            && gpu.arch == "gfx1151";
        let prior_invariant = std::mem::replace(&mut gpu.prefill_invariant_scope, invariant);
        // The ordinary plan dedicates HcMixed/GdnZ/GdnA/GdnB to max_chunk
        // rows (plan allocations at 2677/2682/2683/2688). Only [0,n)
        // contains live tokens. Compact scratch may alias live fields: never
        // arm its in-place padding, even if a facade appears large enough.
        let ragged_scope = native_trunk_prefill_scope(
            self.prefill_scope, gpu.qwen4_trunk_iu4, &gpu.arch, self.scratch.mode,
        );
        let prior_ragged = std::mem::replace(
            &mut gpu.qwen4_trunk_iu4_prefill_scope, ragged_scope,
        );
        let result = self.forward_chunk_scoped(
            bundle,
            gpu,
            tokens,
            argmax_of,
            logits,
            top1,
            wide_hidden_capture,
            output_policy,
        );
        gpu.qwen4_scope = scope;
        gpu.prefill_invariant_scope = prior_invariant;
        gpu.qwen4_trunk_iu4_prefill_scope = prior_ragged;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_chunk_scoped(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        tokens: &[u32],
        argmax_of: Option<&GpuTensor>,
        logits: Option<&GpuTensor>,
        top1: Option<&GpuTensor>,
        wide_hidden_capture: Option<&GpuTensor>,
        output_policy: Qwen4OutputPolicy,
    ) -> Result<(), Qwen4GpuForwardError> {
        let ple_lookahead = std::mem::take(&mut self.ple_lookahead);
        if argmax_of.is_some() && tokens.len() != 1 {
            return Err(invalid("an argmax-token Qwen4 forward carries one token"));
        }
        let n = tokens.len();
        if n == 0 || n > self.scratch.max_chunk {
            return Err(invalid(format!(
                "chunk length {n} exceeds configured capacity {}",
                self.scratch.max_chunk
            )));
        }
        // Before anything can read a donor a failed staged forward left
        // overwritten (decode included).
        if let Some(stage) = self.expert_stage.as_ref() {
            stage.restore_blocking(gpu)?;
        }
        let config = bundle.config.clone();
        let end_position = bundle
            .state
            .position
            .checked_add(n)
            .ok_or_else(|| invalid("Qwen4 forward position overflows"))?;
        if end_position > bundle.state.max_seq_len {
            return Err(invalid(format!(
                "Qwen4 forward end position {end_position} exceeds context capacity {}",
                bundle.state.max_seq_len
            )));
        }
        if argmax_of.is_none()
            && tokens
                .iter()
                .copied()
                .any(|token| (token as usize) >= config.vocab_size)
        {
            return Err(invalid(
                "Qwen4 token id is outside the embedding vocabulary",
            ));
        }
        let dims = program_dims(&config);
        let requested_rows = output_policy.requested_rows(n);
        if let Some(requested_rows) = requested_rows {
            let logits = logits.ok_or_else(|| {
                invalid("Qwen4 head output policy requires a logits tensor")
            })?;
            let expected_logits = requested_rows
                .checked_mul(config.vocab_size)
                .ok_or_else(|| invalid("logit shape overflow"))?;
            if logits.dtype != DType::F32 || logits.numel() != expected_logits {
                return Err(invalid(format!(
                    "logits must be F32 with {expected_logits} elements for {output_policy:?} output"
                )));
            }
            if let Some(top1) = top1 {
                let expected = requested_rows
                    .checked_mul(std::mem::size_of::<i32>())
                    .ok_or_else(|| invalid("top1 shape overflow"))?;
                if top1.dtype != DType::Raw || top1.buf.size() < expected {
                    return Err(invalid(
                        "top1 output must be Raw with one i32 per requested output row",
                    ));
                }
            }
        } else if top1.is_some() {
            return Err(invalid("Qwen4 discarded output cannot receive top1 rows"));
        }
        if let Some(capture) = wide_hidden_capture.as_ref() {
            let expected = n
                .checked_mul(dims.wide())
                .ok_or_else(|| invalid("wide hidden capture shape overflow"))?;
            if capture.dtype != DType::F32 || capture.numel() < expected {
                return Err(invalid(format!(
                    "wide hidden capture must be F32 with at least {expected} elements"
                )));
            }
        }
        // Map the QSA context rows this forward writes and reads (every
        // arena through `end_position`) and, for a chunk the gathered prefill
        // attention takes, its F16 workspace over the same prefix: here,
        // before any view, capture or record exists. Neither grows while a
        // capture or record is armed; a failure leaves the position
        // unpublished and already mapped pages owned.
        // Session snapshots yield to context growth (only at a growth
        // boundary; no query on the steady path).
        bundle.release_session_for_growth(gpu, end_position);
        let mapped_before = bundle.state.mapped_context_tokens();
        bundle
            .state
            .ensure_mapped_capacity(gpu, end_position)
            .map_err(|error| {
                invalid(format!(
                    "map Qwen4 QSA context through {end_position} tokens: {error}"
                ))
            })?;
        if gpu.route_rows(n) >= rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS {
            if let Some(qsa) = bundle.state.qsa.first().filter(|qsa| qsa.f16_arena.is_none()) {
                rdna_compute::tensor_ops::ensure_qsa_gathered_wmma_workspace(
                    gpu,
                    qsa.format,
                    config.num_key_value_heads,
                    end_position,
                )?;
            }
        }
        if bundle.state.mapped_context_tokens() != mapped_before {
            if let Ok((target, mtp)) = bundle.qsa_context_committed_bytes(gpu) {
                eprintln!(
                    "  qwen4 QSA context mapped for {} tokens: {} MiB target + {} MiB MTP committed, {} MiB gather workspace",
                    bundle.state.mapped_context_tokens(),
                    target >> 20,
                    mtp >> 20,
                    gpu.qsa_gather_scratch_bytes() >> 20
                );
            }
        }
        // Without PLE (`config.ple` is `None`) there is no PLE layer, no row
        // geometry and no row fetch below.
        let ple_layer_index = if config.ple.is_some() {
            let index = config
                .ple_layer_ids
                .first()
                .and_then(|id| id.checked_sub(1))
                .ok_or_else(|| invalid("PLE layer id missing"))?;
            if index >= config.num_hidden_layers {
                return Err(invalid("PLE layer id is outside the trunk"));
            }
            Some(index)
        } else {
            None
        };
        let (ple_head_count, ple_row_width) = config
            .ple
            .as_ref()
            .map_or((0, 0), |ple| (ple.head_count, ple.row_width));
        let final_hyper = if requested_rows.is_some() {
            Some(hyper_read_desc(
                &bundle.weights,
                &bundle.weights.root.final_hyper,
            )?)
        } else {
            None
        };
        let lm_head = if requested_rows.is_some() {
            Some(dense_ref(&bundle.weights, &bundle.weights.root.lm_head)?)
        } else {
            None
        };
        if let Some(requested_rows) = requested_rows {
            let final_hyper = final_hyper
                .as_ref()
                .ok_or_else(|| invalid("Qwen4 final hyper preflight disappeared"))?;
            let lm_head = lm_head
                .as_ref()
                .ok_or_else(|| invalid("Qwen4 LM head preflight disappeared"))?;
            let logits = logits
                .ok_or_else(|| invalid("Qwen4 logits output disappeared before preflight"))?;
            let preflight_scratch = layer_scratch(&self.scratch, &self.scratch.router_logits);
            validate_final_hyper(
                dims,
                final_hyper,
                &self.scratch.streams,
                &preflight_scratch,
                n,
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("validate Qwen4 final hyper: {error:?}"))
            })?;
            let hidden_elements = n
                .checked_mul(config.hidden_size)
                .ok_or_else(|| invalid("Qwen4 final hidden shape overflow"))?;
            let final_hidden = view(&self.scratch.hc_mixed, 0, hidden_elements);
            validate_lm_head(lm_head, &final_hidden, logits, n, requested_rows).map_err(
                |error| {
                    Qwen4GpuForwardError::Dispatch(format!("validate Qwen4 LM head: {error:?}"))
                },
            )?;
        }

        // The PLE row fetch runs on the row store's worker while the step
        // program is built below; a matching retained lookahead is adopted.
        let retained = self.ple_ahead.take();
        let mut ple = match bundle.ple.as_ref() {
            None => {
                drop(retained);
                None
            }
            Some(owner) => Some(
                if argmax_of.is_none() {
                    RowFetch::begin_or_adopt(
                        &owner.rows,
                        retained,
                        bundle.state.ple_history.row_ids(&owner.metadata, tokens),
                    )
                } else {
                    drop(retained);
                    RowFetch::begin(&owner.rows)
                }
                .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?,
            ),
        };
        let router_logits = matrix_view(&self.scratch.router_logits, n, router_columns(dims))?;
        let scratch_desc = layer_scratch(&self.scratch, &router_logits);
        // Decided once: every op touching the HC streams in this forward agrees.
        let bf16_state = gpu.qwen4_bf16_streams(n);
        // The row count every routing decision below sees (`n` unless the
        // invariant prefill profile is on); launch extents keep `n`.
        let route_n = gpu.route_rows(n);
        // Expert parallelism picks the MoE table per forward by routing row
        // count, identically on every rank: one row, or a few-row forward
        // (`2..=QWEN4_EP_GATHER_MAX_ROWS` rows that are also the routing rows:
        // verify / replay / takeover), runs the rank's decode table with the
        // row gather installed and a parity gather buffer as the expanded-row
        // scratch; larger chunks run the full (P2P) table, no gather.
        let ep = self.ep.as_ref();
        let ep_few_rows = ep.is_some()
            && (2..=QWEN4_EP_GATHER_MAX_ROWS).contains(&route_n)
            && route_n == n;
        if let Some(ep) = ep {
            if route_n == 1 {
                if n != 1 {
                    return Err(invalid("an EP decode forward carries exactly one row"));
                }
                if ep.gather.is_none() {
                    return Err(invalid(
                        "EP decode forward before attach_ep: the row gather is not installed",
                    ));
                }
            } else if ep_few_rows {
                if ep.gather.is_none() {
                    return Err(invalid(
                        "EP few-row forward before attach_ep: the row gather is not installed",
                    ));
                }
            } else if ep.moe_full.len() != self.moe.len() {
                return Err(invalid(
                    "EP chunk forward before attach_ep: the full expert tables are not built",
                ));
            }
        }
        let ep_gathered = route_n == 1 || ep_few_rows;
        let ep_gather: Option<&dyn MoeRowGather> = ep
            .filter(|_| ep_gathered)
            .and_then(|ep| ep.gather.as_deref().map(|gather| gather as &dyn MoeRowGather));
        let ep_scratches: Option<[Qwen4LayerScratch<'_>; QWEN4_EP_GATHER_BUFFERS]> =
            ep.filter(|_| ep_gathered).map(|ep| {
                std::array::from_fn(|parity| Qwen4LayerScratch {
                    moe_down_expanded: &ep.buffers[parity],
                    ..layer_scratch(&self.scratch, &router_logits)
                })
            });
        // Owner-routed chunk MoE (`HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`, hook installed by
        // the mesh): a chunk on the symmetric IU4 route (>= 512 routing rows that are the
        // launch rows) binds each layer's compact table plus the tile exchange; every
        // other chunk keeps the full-table route exactly as without the lever.
        let tile_route = ep
            .filter(|_| route_n > 1 && !ep_few_rows)
            .and_then(|ep| ep.tile.as_ref())
            .filter(|route| route.hook.is_some());
        let tile_active =
            tile_route.is_some() && route_n == n && gpu.qwen4_moe_sym_iu4_applies(route_n);
        if let Some(route) = tile_route.filter(|_| !tile_active) {
            // A full-table chunk also writes `y_down_grouped`: it may start only after the
            // peer finished pulling this rank's rows of the last exchange.
            if let Some(hook) = route.hook.as_ref() {
                hook.before_down(gpu)
                    .map_err(|error| invalid(format!("EP tile exchange: {error}")))?;
            }
        }
        let tile_route = tile_route.filter(|_| tile_active);
        // G2 expert stage under EP: a full-table chunk reads another rank's donor VRAM, so every
        // rank drains its stage donors and crosses a host barrier before ANY rank launches it.
        if ep.is_some() && route_n > 1 && !ep_few_rows && !tile_active {
            // A stage is only ever USED (donors overwritten) on forwards that bind the compact
            // route, which needs the installed exchange; without its hook no forward staged, so
            // there is nothing to drain.
            let hook = ep
                .and_then(|ep| ep.tile.as_ref())
                .and_then(|route| route.hook.as_ref());
            if let (Some(stage), Some(hook)) = (self.expert_stage.as_ref(), hook) {
                stage.ep_quiesce(gpu)?;
                hook.host_barrier()
                    .map_err(|error| invalid(format!("EP stage quiesce barrier: {error}")))?;
            }
        }
        if config.num_hidden_layers.saturating_mul(7).saturating_add(1) > QWEN4_STEP_INLINE_CAPACITY
        {
            return Err(invalid(
                "Qwen4 generic layer program inline capacity exhausted",
            ));
        }
        // Decode, small chunks and MTP's few-row wide-hidden verify keep
        // mapped reads (below `min_rows`). A long forward that captures wide
        // hidden rows (the MTP prompt fill) stages unless
        // `HIPFIRE_QWEN4_EXPERT_STAGE_CAPTURE=0`: the capture is a copy of
        // the output streams after the program, on the compute stream. The
        // parity observer keeps its historical unsplit transport. A stream
        // capture keeps mapped reads too: a staged forward forks a copy
        // stream and its failure recovery synchronizes the device.
        let expert_stage = match self.expert_stage.as_ref().filter(|stage| stage.begin_forward(tile_active)) {
            Some(stage)
                if n >= stage.min_rows
                    && (stage.capture || wide_hidden_capture.is_none())
                    && self.qsa_tap.is_none() =>
            {
                let capturing = match gpu.active_stream.as_ref() {
                    Some(stream) => gpu.hip.stream_is_capturing(stream)?,
                    None => false,
                };
                (!capturing).then_some(stage)
            }
            _ => None,
        };
        let ctx = DispatchCtx::new(gpu);
        let mut steps: SmallVec<[Step<'_>; QWEN4_STEP_INLINE_CAPACITY]> = SmallVec::new();
        let mut gdn_slot = 0usize;
        let mut qsa_slot = 0usize;
        // `HIPFIRE_QWEN4_PLE_FUSE`: the PLE rows are staged straight as F16 for
        // the BF16 key/value projections (decided once; the step op and the
        // staging closure agree on it).
        let mut ple_rows_f16 = false;
        // `HIPFIRE_CHUNK_DUMP_DIR` (developer diagnostic; off by default).
        let chunk_dump = crate::chunk_dump::config().filter(|_| route_n > 1);
        let mut dump_ends: Vec<(usize, usize, bool)> = Vec::new();
        for layer_index in 0..config.num_hidden_layers {
            let layer = &bundle.weights.layer_refs[layer_index];
            if ple_layer_index == Some(layer_index) {
                let ple_weights = layer
                    .ple
                    .as_ref()
                    .ok_or_else(|| invalid("PLE weights missing"))?;
                if steps.len() >= QWEN4_STEP_INLINE_CAPACITY {
                    return Err(invalid(
                        "Qwen4 generic layer program inline capacity exhausted",
                    ));
                }
                let ple = ple_desc(&bundle.weights, ple_weights)?;
                ple_rows_f16 = qwen4_ple_rows_f16(
                    gpu,
                    bf16_state,
                    dims.hc_count,
                    config.hidden_size,
                    &ple,
                    n,
                );
                steps.push(Step::GroupedDepthwise(GroupedDepthwiseOp {
                    rotation: &self.scratch.rotation,
                    state_bf16: bf16_state,
                    rows_f16: ple_rows_f16,
                    key: ple.key,
                    value: ple.value,
                    norm_key: ple.norm_key,
                    norm_query: ple.norm_query,
                    norm_conv: ple.norm_conv,
                    conv: ple.conv,
                    state: bundle
                        .state
                        .ple_conv
                        .as_ref()
                        .ok_or_else(|| invalid("PLE conv state missing"))?,
                    streams: &self.scratch.streams,
                    rows_tensor: &self.scratch.ple_rows,
                    query: &self.scratch.ple_query,
                    key_scratch: &self.scratch.ple_key,
                    value_scratch: &self.scratch.ple_value,
                    gated: &self.scratch.ple_gated,
                    normed: &self.scratch.ple_normed,
                    output: &self.scratch.ple_output,
                    rows: n,
                    branches: dims.hc_count,
                    hidden: config.hidden_size,
                    embed_dim: ple_head_count * ple_row_width,
                    kernel_size: dims.ple_conv_kernel_dim,
                    dilation: config.ple_conv_dilation(),
                    epsilon: EPSILON,
                }));
            }

            // Balanced ownership: a host-mapped trunk layer of a gathered EP
            // forward (one row, or few rows) binds its live runtime, whose
            // tables the layer's owner assignment rewrites after routing.
            // Every other layer keeps the static EP table (or the full table
            // for chunk forwards). `ep.balanced` is empty unless the env
            // lever was on at attach, so off binds exactly what it always did.
            let balanced_live = ep
                .filter(|_| ep_gathered)
                .and_then(|ep| ep.balanced.get(layer_index))
                .and_then(Option::as_ref)
                .map(|balanced| &balanced.live);
            let layer_moe = match (balanced_live, ep) {
                (Some(live), _) => Some(live),
                (None, Some(ep)) if route_n > 1 && !ep_few_rows => {
                    if tile_route.is_some() {
                        self.moe.get(layer_index)
                    } else {
                        ep.moe_full.get(layer_index)
                    }
                }
                _ => self.moe.get(layer_index),
            };
            if let Some(stage) = expert_stage {
                stage.prepare_runtime(gpu, layer_index, layer_moe)?;
            }
            let mut description = layer_desc(
                &bundle.weights,
                layer,
                layer_moe,
                &config,
                &self.decode_q8,
                route_n,
                // Staged expert tables never replace the live table.
                expert_stage.filter(|_| balanced_live.is_none()),
            )?;
            if let (Some(gather), Qwen4FfnBinding::Moe(binding)) =
                (ep_gather, &mut description.ffn)
            {
                binding.row_gather = Some(gather);
                if ep_few_rows {
                    // The gather buffer is the combine's `down_expanded`; the
                    // unfused shared down writes the ordinary scratch.
                    binding.shared_down_out = Some(&self.scratch.moe_down_expanded);
                }
            }
            if let (Some(route), Qwen4FfnBinding::Moe(binding)) =
                (tile_route, &mut description.ffn)
            {
                if let Some(hook) = route.hook.as_deref() {
                    binding.tile_exchange = Some(MoeTileExchangeParams {
                        exchange: hook,
                        tile_ids_owned: &route.tile_ids_owned,
                    });
                }
            }
            let attn_read = &description.attn_hyper.read;
            if steps.len() >= QWEN4_STEP_INLINE_CAPACITY {
                return Err(invalid(
                    "Qwen4 generic layer program inline capacity exhausted",
                ));
            }
            steps.push(Step::HyperRead(HyperReadOp {
                rotation: &self.scratch.rotation,
                state_bf16: bf16_state,
                input: &self.scratch.streams,
                norm_weight: attn_read.norm,
                input_mix_down: decode_copy(
                    &self.decode_q8,
                    layer_index * DECODE_Q8_PER_LAYER,
                    route_n,
                    attn_read.input_mix_down,
                ),
                input_mix_up: decode_copy(
                    &self.decode_q8,
                    layer_index * DECODE_Q8_PER_LAYER + 1,
                    route_n,
                    attn_read.input_mix_up,
                ),
                normalized: &self.scratch.hc_normalized,
                low: &self.scratch.hc_low,
                up: &self.scratch.hc_up,
                mixed: &self.scratch.hc_mixed,
                bf16_scratch: &self.scratch.gdn_bf16,
                rows: n,
                branches: dims.hc_count,
                hidden: config.hidden_size,
                low_rank: dims.hc_lowrank,
            }));

            match layer.kind {
                LayerType::LinearAttention => {
                    let state = bundle
                        .state
                        .gdn
                        .get(gdn_slot)
                        .ok_or_else(|| invalid("GDN state slot missing"))?;
                    gdn_slot += 1;
                    let weights = match &description.attention {
                        Qwen4AttentionWeights::Linear(weights) => weights,
                        _ => return Err(invalid("GDN descriptor kind mismatch")),
                    };
                    steps.push(Step::GatedDeltaNet(GatedDeltaNetOp {
                        rotation: &self.scratch.rotation,
                        qkv: weights.qkv,
                        conv: weights.conv,
                        in_proj_a: weights.in_proj_a,
                        in_proj_b: weights.in_proj_b,
                        a_log: weights.a_log,
                        dt_bias: weights.dt_bias,
                        z: weights.z,
                        norm: weights.norm,
                        output: weights.output,
                        recurrent: &state.recurrent,
                        conv_state: &state.conv,
                        projection: &self.scratch.projection,
                        projection2: &self.scratch.projection2,
                        a: &self.scratch.gdn_a,
                        b: &self.scratch.gdn_b,
                        gate: &self.scratch.gdn_gate,
                        beta: &self.scratch.gdn_beta,
                        recurrent_output: &self.scratch.gdn_recurrent_output,
                        bf16_scratch: &self.scratch.gdn_bf16,
                        z_output: &self.scratch.gdn_z,
                        output_scratch: &self.scratch.gdn_output,
                        input: &self.scratch.hc_mixed,
                        output_tensor: &self.scratch.attention_output,
                        rows: n,
                        start_position: bundle.state.position,
                        key_heads: dims.linear_num_key_heads,
                        value_heads: dims.linear_num_value_heads,
                        key_dim: dims.linear_key_head_dim,
                        value_dim: dims.linear_value_head_dim,
                        conv_kernel: dims.linear_conv_kernel_dim,
                        input_width: dims.hidden,
                        row_capture: bundle.state.gdn_row_capture(gdn_slot - 1, n),
                        zba_fold: self.trunk_zba.get(layer_index).and_then(Option::as_ref),
                        trunk_a4: self.trunk_a4.get(layer_index).copied().unwrap_or(0),
                        phase: MixerPhase::Whole,
                    }));
                }
                LayerType::FullAttention => {
                    let state = bundle
                        .state
                        .qsa
                        .get(qsa_slot)
                        .ok_or_else(|| invalid("QSA state slot missing"))?;
                    qsa_slot += 1;
                    let weights = match &description.attention {
                        Qwen4AttentionWeights::Full(weights) => weights,
                        _ => return Err(invalid("QSA descriptor kind mismatch")),
                    };
                    steps.push(Step::IndexedAttention(IndexedAttentionOp {
                        rotation: &self.scratch.rotation,
                        mode: IndexedAttentionMode::Full,
                        indexer_qk: weights.indexer_qk,
                        indexer_q_norm: weights.indexer_q_norm,
                        indexer_k_norm: weights.indexer_k_norm,
                        q: weights.q,
                        k: weights.k,
                        v: weights.v,
                        q_norm: weights.q_norm,
                        k_norm: weights.k_norm,
                        output: weights.output,
                        state: IndexedAttentionState {
                            format: state.format,
                            full_keys: &state.full_keys,
                            full_values: &state.full_values,
                            raw_index_keys: &state.raw_index_keys,
                            pooled_keys: &state.pooled_keys,
                            selected_indices: &state.selected_indices,
                            // Decode and speculative verify never consume the arena.
                            f16_arena: if self.prefill_scope { state.f16_arena_ref() } else { None },
                            full_capacity: state.full_capacity,
                            raw_capacity: state.raw_capacity,
                            pooled_capacity: state.pooled_capacity,
                            selected_capacity: state.selected_capacity,
                            position_capacity: state.position_capacity,
                            full_len: state.full_len,
                            raw_len: state.raw_len,
                            pooled_len: state.pooled_len,
                            selected_len: state.selected_len,
                            position: state.position,
                        },
                        input: &self.scratch.hc_mixed,
                        index_scratch: &self.scratch.qsa_index,
                        qgate_scratch: &self.scratch.qsa_qgate,
                        k_scratch: &self.scratch.qsa_k,
                        v_scratch: &self.scratch.qsa_v,
                        qsa_output: &self.scratch.qsa_output,
                        selected_scratch: &self.scratch.qsa_selected,
                        attention_output: &self.scratch.attention_output,
                        bf16_scratch: &self.scratch.gdn_bf16,
                        rows: n,
                        index_heads: dims.indexer_n_heads,
                        index_kv_heads: dims.indexer_kv_heads,
                        index_dim: dims.indexer_head_dim,
                        budget: dims.indexer_budget,
                        compress: dims.indexer_compress_ratio,
                        heads: dims.num_attention_heads,
                        kv_heads: dims.num_key_value_heads,
                        head_dim: dims.head_dim,
                        rotary_dim: config.rotary_dim(),
                        rope: config.rope(),
                        input_width: dims.hidden,
                        trunk_a4: self.trunk_a4.get(layer_index).copied().unwrap_or(0),
                        phase: MixerPhase::Whole,
                    }));
                }
            }

            let attn_write = &description.attn_hyper.write;
            steps.push(Step::HyperWrite(HyperWriteOp {
                rotation: &self.scratch.rotation,
                state_bf16: bf16_state,
                input: &self.scratch.streams,
                norm_weight: attn_write.norm,
                block_inject: attn_write.block_inject,
                normalized: &self.scratch.hc_normalized,
                mixed: &self.scratch.attention_output,
                gates: &self.scratch.hc_gates,
                output: &self.scratch.streams,
                rows: n,
                branches: dims.hc_count,
                hidden: config.hidden_size,
            }));

            let mlp_read = &description.mlp_hyper.read;
            steps.push(Step::HyperRead(HyperReadOp {
                rotation: &self.scratch.rotation,
                state_bf16: bf16_state,
                input: &self.scratch.streams,
                norm_weight: mlp_read.norm,
                input_mix_down: decode_copy(
                    &self.decode_q8,
                    layer_index * DECODE_Q8_PER_LAYER + 2,
                    route_n,
                    mlp_read.input_mix_down,
                ),
                input_mix_up: decode_copy(
                    &self.decode_q8,
                    layer_index * DECODE_Q8_PER_LAYER + 3,
                    route_n,
                    mlp_read.input_mix_up,
                ),
                normalized: &self.scratch.hc_normalized,
                low: &self.scratch.hc_low,
                up: &self.scratch.hc_up,
                mixed: &self.scratch.hc_mixed,
                bf16_scratch: &self.scratch.gdn_bf16,
                rows: n,
                branches: dims.hc_count,
                hidden: config.hidden_size,
                low_rank: dims.hc_lowrank,
            }));

            match &description.ffn {
                Qwen4FfnBinding::Moe(moe) => {
                    let sealed = crate::program::seal_moe_for_layer(
                        &ctx,
                        dims,
                        *moe,
                        ep_scratches.as_ref().map_or(&scratch_desc, |parity| {
                            &parity[qwen4_ep_buffer_index(layer_index, config.num_hidden_layers)]
                        }),
                        n,
                        route_n,
                    )
                    .map_err(|error| {
                        Qwen4GpuForwardError::Dispatch(format!("seal Qwen4 MoE: {error:?}"))
                    })?;
                    steps.push(Step::Clear(ClearOp {
                        tensor: &self.scratch.ffn_output,
                        elements: n * config.hidden_size,
                    }));
                    steps.push(Step::Moe(sealed));
                }
                Qwen4FfnBinding::Dense(dense) => {
                    steps.push(Step::DenseFfn(DenseFfnOp {
                        recipe: DenseFfnRecipe::SiluBf16Source,
                        gate: dense.gate,
                        up: dense.up,
                        down: dense.down,
                        input: &self.scratch.hc_mixed,
                        gate_scratch: &self.scratch.moe_gate_batch,
                        up_scratch: &self.scratch.moe_up_batch,
                        output: &self.scratch.ffn_output,
                        rotation: &self.scratch.rotation,
                        rows: n,
                        hidden: config.hidden_size,
                        intermediate: dense.intermediate,
                    }));
                }
            }

            let mlp_write = &description.mlp_hyper.write;
            steps.push(Step::HyperWrite(HyperWriteOp {
                rotation: &self.scratch.rotation,
                state_bf16: bf16_state,
                input: &self.scratch.streams,
                norm_weight: mlp_write.norm,
                block_inject: mlp_write.block_inject,
                normalized: &self.scratch.hc_normalized,
                mixed: &self.scratch.ffn_output,
                gates: &self.scratch.hc_gates,
                output: &self.scratch.streams,
                rows: n,
                branches: dims.hc_count,
                hidden: config.hidden_size,
            }));
            if chunk_dump.is_some_and(|dump| dump.wants(layer_index)) {
                let bf16_grouped =
                    layer_moe.is_some_and(|moe| moe.symmetric) && gpu.qwen4_moe_sym_iu4_applies(route_n);
                dump_ends.push((layer_index, steps.len(), bf16_grouped));
            }
        }
        validate_steps(gpu, &steps).map_err(|error| {
            Qwen4GpuForwardError::Dispatch(format!("preflight Qwen4 typed program: {error:?}"))
        })?;
        // A retained replay does not execute host code inside its steps. Derive
        // the same QSA commit for either route before any forward-side effects.
        let mut qsa_commits: SmallVec<
            [(usize, usize, usize, usize, usize, usize); QWEN4_QSA_INLINE_CAPACITY],
        > = SmallVec::new();
        for step in &steps {
            if let Step::IndexedAttention(op) = step {
                if qsa_commits.len() >= QWEN4_QSA_INLINE_CAPACITY {
                    return Err(invalid("Qwen4 QSA commit inline capacity exhausted"));
                }
                let (full_len, raw_len, pooled_len, selected_len, position) =
                    op.next_lengths().map_err(|error| {
                        Qwen4GpuForwardError::Dispatch(format!("derive Qwen4 QSA state: {error:?}"))
                    })?;
                qsa_commits.push((
                    qsa_commits.len(),
                    full_len,
                    raw_len,
                    pooled_len,
                    selected_len,
                    position,
                ));
            }
        }
        // A HIP body: no retained tape records or routes this forward.
        let hip_body = n == 1
            && !self.lane_scope
            && !gpu.prefill_invariant_scope
            && matches!(gpu.replay.state(), ReplayState::Hip | ReplayState::Fallback);
        // `argmax_of` on a HIP body: the GPU argmax writes the token id and the
        // host reads it back only at the PLE layer, so the GPU runs this
        // forward's first layers while the host waits. Otherwise the host
        // takes the argmax now (the program above is built and nothing has
        // touched the device or the state yet).
        let device_token = argmax_of.filter(|_| hip_body);
        let host_argmax: [u32; 1];
        let tokens = match argmax_of {
            Some(previous) if !hip_body => {
                let token = rdna_compute::tensor_ops::argmax_f32_host(gpu, previous)?;
                if token as usize >= config.vocab_size {
                    return Err(invalid(
                        "Qwen4 token id is outside the embedding vocabulary",
                    ));
                }
                host_argmax = [token];
                if let (Some(ple), Some(owner)) = (ple.as_mut(), bundle.ple.as_ref()) {
                    ple.prefetch(
                        bundle
                            .state
                            .ple_history
                            .row_ids(&owner.metadata, &host_argmax),
                    )
                    .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
                }
                &host_argmax[..]
            }
            _ => tokens,
        };
        let ids = view(&self.scratch.token_ids, 0, n * std::mem::size_of::<i32>());
        // No claim on the device ids until this forward's own upload lands.
        self.uploaded_id_rows = 0;
        if let Some(previous) = device_token {
            argmax_f32(
                gpu,
                &ArgmaxF32 {
                    logits: previous,
                    indices: &ids,
                    rows: 1,
                    vocab: config.vocab_size,
                },
            )?;
            if self.token_readback.is_none() {
                self.token_readback = Some(TokenReadback {
                    stream: gpu.hip.stream_create_non_blocking()?,
                    event: gpu
                        .hip
                        .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
                });
            }
            if let Some(readback) = self.token_readback.as_ref() {
                gpu.hip
                    .event_record(&readback.event, gpu.active_stream.as_ref())?;
            }
        } else {
            for (index, token) in tokens.iter().copied().enumerate() {
                let bytes = &mut self.host_token_bytes[index * 4..index * 4 + 4];
                bytes.copy_from_slice(&(token as i32).to_ne_bytes());
            }
            if hip_body {
                // The one id is a stream-ordered fill, not a synchronous copy.
                let active = gpu.active_stream.as_ref();
                gpu.hip.memset_d32_async(
                    &self.scratch.token_ids.buf,
                    tokens[0] as i32,
                    1,
                    active,
                )?;
            } else {
                gpu.memcpy_htod_auto(
                    &self.scratch.token_ids.buf,
                    &self.host_token_bytes[..n * std::mem::size_of::<i32>()],
                )?;
                // Whole-row host upload on the default stream: the device ids
                // stay as uploaded until the next forward writes them.
                if !gpu.graphs.capture_mode {
                    self.uploaded_id_rows = n;
                }
            }
        }
        let embedding = bundle.weights.resident(&bundle.weights.root.embedding)?;
        let embedding_rot = view(&self.scratch.embedding_rot, 0, n * config.hidden_size);
        let embeddings = view(&self.scratch.embeddings, 0, n * config.hidden_size);
        execute_embedding(
            gpu,
            &EmbeddingOp {
                table: embedding,
                rotated: &embedding_rot,
                token_ids: &ids,
                output: &embeddings,
                rows: n,
                dim: config.hidden_size,
            },
        )
        .map_err(|e| Qwen4GpuForwardError::Dispatch(format!("Qwen4 embedding: {e:?}")))?;

        let next_history = bundle.state.ple_history;
        let next_position = bundle.state.position;
        // Boundary samples, hoisted so the post-body finalize outside this closure
        // can report them: they are taken where the window opens, which is what
        // makes the census an exact bracket of one recorded body.
        let mut device_read: Option<u32> = None;
        let mut diagnostic_capture = false;
        let mut launched_before = 0u64;
        let mut traced = false;
        let attempt = (|| -> Result<
            SmallVec<[(usize, usize, usize, usize, usize, usize); QWEN4_QSA_INLINE_CAPACITY]>,
            Qwen4GpuForwardError,
        > {
            let ple_rows = view(
                &self.scratch.ple_rows,
                0,
                n * ple_head_count * ple_row_width,
            );
            let staged = view(
                &self.scratch.ple_staged,
                0,
                n * ple_head_count * ple_row_width,
            );

            // Expand the embedding batch into the row-major stream basis with
            // one broadcast kernel.  The shared HC program consumes
            // [row, branch, hidden], so a launch-per-row/branch memcpy loop is
            // both unnecessary and visible in short AR prefill profiles.
            let wide = dims.wide();
            if bf16_state {
                gpu.hc_streams_init_from_embed_batched_bf16(
                    &embeddings,
                    &self.scratch.streams,
                    config.hidden_size as i32,
                    config.hc_count as i32,
                    n as i32,
                )?;
            } else {
                gpu.hc_streams_init_from_embed_batched(
                    &embeddings,
                    &self.scratch.streams,
                    config.hidden_size as i32,
                    config.hc_count as i32,
                    n as i32,
                )?;
            }

            // PLE rows: wait for the host fetch, stage, upload and widen them.
            // A prefill defers this until layer 0 is enqueued (the rows are first
            // read at the PLE layer), so the host fetch overlaps GPU work; a
            // single-token forward keeps it ahead of the retained-body boundary.
            let ple_host = &mut self.host_ple;
            let ple_split = steps
                .iter()
                .position(|step| matches!(step, Step::GroupedDepthwise(_)))
                .filter(|_| n > 1 || device_token.is_some());
            let deferred = ple_split.is_some();
            let readback = self.token_readback.as_ref();
            let token_ids = &self.scratch.token_ids;
            let history = &bundle.state.ple_history;
            let ple_metadata = bundle.ple.as_ref().map(|owner| &owner.metadata);
            let vocab = config.vocab_size;
            let read_token = &mut device_read;
            let mut stage_ple = |gpu: &mut Gpu| -> Result<(), Qwen4GpuForwardError> {
                if ple.as_ref().is_some_and(|ple| ple.lease().is_some()) {
                    return Ok(());
                }
                if device_token.is_some() {
                    // The argmax this forward enqueued first: wait for it alone
                    // and read its id on a stream independent of the layers
                    // enqueued since.
                    let readback =
                        readback.ok_or_else(|| invalid("Qwen4 token readback missing"))?;
                    gpu.hip.event_synchronize(&readback.event)?;
                    let mut bytes = [0u8; 4];
                    gpu.hip
                        .memcpy_dtoh_async(&mut bytes, &token_ids.buf, Some(&readback.stream))?;
                    gpu.hip.stream_synchronize(&readback.stream)?;
                    let token = i32::from_ne_bytes(bytes) as u32;
                    if token as usize >= vocab {
                        return Err(invalid(
                            "Qwen4 token id is outside the embedding vocabulary",
                        ));
                    }
                    *read_token = Some(token);
                    if let (Some(ple), Some(ple_metadata)) = (ple.as_mut(), ple_metadata) {
                        ple.prefetch(history.row_ids(ple_metadata, &[token]))
                            .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
                    }
                }
                // A profile without PLE has no rows to wait for, stage, upload
                // or gather (the token read above is all this forward needs).
                let Some(ple) = ple.as_mut() else {
                    return Ok(());
                };

                let wait_started = qwen4_profile_start();
                let lease_result = ple.wait();
                qwen4_profile_record(Qwen4ProfilePhase::PleWait, wait_started);
                let lease =
                    lease_result.map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
                let upload_len = lease
                    .as_bytes()
                    .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?
                    .len();
                // Retire any earlier asynchronous upload still reading the
                // staging bytes before the host copy overwrites them.
                let stage_started = qwen4_profile_start();
                let stage_result = match ple_host.stage_buffer(gpu, upload_len) {
                    Ok(buffer) => lease
                        .stage_into(buffer)
                        .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string())),
                    Err(error) => Err(error.into()),
                };
                qwen4_profile_record(Qwen4ProfilePhase::PleStage, stage_started);
                stage_result?;
                let upload_started = qwen4_profile_start();
                // After a device-token readback the host has synchronized past
                // the previous forward's upload, so the staging buffer is free.
                let upload_result = if device_token.is_some() {
                    ple_host.upload_unfenced(gpu, &staged.buf, upload_len)
                } else if ple_async_upload(
                    ple_host.enabled(),
                    false,
                    deferred,
                    gpu.graphs.capture_mode,
                ) {
                    // Deferred past layer 0: do not make the host wait for the
                    // layers already enqueued. The fence keeps the staging
                    // bytes unwritable until the copy has run.
                    ple_host.upload_async(gpu, &staged.buf, upload_len)
                } else {
                    ple_host.upload_blocking(gpu, &staged.buf, upload_len)
                };
                qwen4_profile_record(Qwen4ProfilePhase::PleUpload, upload_started);
                upload_result?;
                lease
                    .validate_after_upload()
                    .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
                let apply_started = qwen4_profile_start();
                if ple_rows_f16 {
                    gpu.grouped_gather_convert_bf16_f16(
                        &staged,
                        &ple_rows,
                        n,
                        ple_head_count,
                        ple_row_width,
                    )?;
                } else {
                    gpu.grouped_gather_convert_bf16(
                        &staged,
                        &ple_rows,
                        n,
                        ple_head_count,
                        ple_row_width,
                    )?;
                }
                qwen4_profile_record(Qwen4ProfilePhase::PleApply, apply_started);
                Ok(())
            };
            if ple_split.is_none() {
                stage_ple(gpu)?;
            }

            // ── Retained-body boundary ────────────────────────────────────────
            // Everything above materialises this forward's inputs — token ids,
            // embeddings, HC stream init, and the PLE staging/upload/gather — and
            // stays outside a retained tape, so it runs on both the HIP path and a
            // replay. Everything below is the body the tape covers, and this
            // boundary owns the whole per-forward retained-body lifecycle (arm,
            // route, close, prepare) so any driver — the generation loop, a
            // benchmark carrier — leaves the controller in a defined state.
            //
            // Eligible only for a plain single-token continuation: the wide-hidden
            // capture is the speculative-verify shape and must never enter the tape.
            // Developer profiles never record: their dense FFN's `silu_f32`
            // launch bypasses the Redline recorder (census 82 launched / 80
            // recorded on tiny-dense-dev-gpu, card B) and the per-row GDN conv
            // carries a position kernarg linear AQL cannot rebind.
            let eligible = n == 1
                && self.ep.is_none()
                && config.profile.is_flash_next()
                && !self.lane_scope
                && !gpu.prefill_invariant_scope
                && wide_hidden_capture.is_none()
                && self.qsa_tap.is_none()
                && !hipfire_dispatch::pipeline::qsa_projection_hook_installed();
            gpu.replay.set_forward_eligible(eligible);
            // A manual shadow controller states the executor directly: one prepared
            // tape is compared across the exact-kernarg HIP oracle, the retained
            // transport, and ordinary HIP. The shadow arms never record, so they do
            // not run the production lifecycle below.
            let shadow_route = if eligible {
                gpu.replay.shadow_body_route()
            } else {
                None
            };
            if shadow_route.is_none() {
                match retained_body_action(
                    eligible,
                    gpu.replay.state(),
                    specialized_sealed_moe_retained_admission(),
                ) {
                    RetainedBodyAction::Ineligible => {}
                    RetainedBodyAction::Refuse { reason } => {
                        // The route cannot be retained: poison before the body so this
                        // and every later forward runs on HIP.
                        gpu.replay.poison(reason);
                        eprintln!("[redline] qwen4 retained body unavailable: {reason}");
                    }
                    RetainedBodyAction::Arm { diagnostic } => {
                        diagnostic_capture = diagnostic;
                        // The fused GDN rotation's arrival counters are zeroed once,
                        // on first use; materialize them before the window opens so
                        // that memset is not state the tape would have to replay.
                        rdna_compute::tensor_ops::ensure_gdn_pair_counters(gpu)?;
                        launched_before = hip_bridge::launch_counters::launch_kernel::count();
                        gpu.replay
                            .begin_auto_capture_if_armed()
                            .map_err(|reason| invalid(reason))?;
                    }
                    RetainedBodyAction::Route => {}
                }
            }
            let pm4_route = match shadow_route {
                Some(ShadowBodyRoute::Plan) => gpu.replay.prepared_pm4_plan_ready(),
                Some(_) => false,
                None => gpu.replay.should_route_pm4(),
            };
            let aql_route = match shadow_route {
                Some(ShadowBodyRoute::Plan) => gpu.replay.prepared_aql_plan_ready(),
                Some(_) => false,
                None => gpu.replay.should_route_aql(),
            };
            if ple_split.is_some() && (pm4_route || aql_route || shadow_route.is_some()) {
                return Err(invalid("Qwen4 retained body with deferred PLE staging"));
            }
            let routed = if pm4_route {
                // Forced non-default retained route: HIP PLE staging (evidence
                // 54ba6c5a6) is outside the PM4 queue's ordering, so drain the
                // staging stream (not the device) before replay.
                let staging_sync = match gpu.active_stream.as_ref() {
                    Some(stream) => gpu.hip.stream_synchronize(stream),
                    None => gpu.hip.stream_synchronize_default(),
                };
                if let Err(error) = staging_sync {
                    gpu.replay
                        .poison(format!("Qwen4 PM4 staging-stream sync failed: {error}"));
                    return Err(invalid(format!(
                        "Qwen4 retained replay refused: staging-stream sync before PM4 replay failed: {error}"
                    )));
                }
                // SAFETY: the boundary above staged every host input the tape's
                // recorded launches read, and every pointer in the tape is owned by
                // this bundle for the plan's lifetime. `replay_pm4_routed` is the
                // plain PM4 replay unless railgun check mode
                // (`HIPFIRE_RAILGUN_CHECK`) is on.
                match unsafe { gpu.replay_pm4_routed(next_position) } {
                    Ok(_) => true,
                    Err(reason) => {
                        gpu.replay.poison(format!("Qwen4 PM4 replay failed: {reason}"));
                        return Err(invalid(format!("Qwen4 retained replay failed: {reason}")));
                    }
                }
            } else if shadow_route == Some(ShadowBodyRoute::HipOracle) {
                // The exact-kernarg oracle: re-execute the recorded HIP launch
                // sequence. This is the arm that proves the tape names the dispatches
                // ordinary HIP issues, not merely that some retained transport runs.
                let launches = gpu.replay.recorded_launches().len();
                match gpu.replay_recorded_hip_prefix_at(launches, next_position) {
                    Ok(_) => true,
                    Err(reason) => {
                        return Err(invalid(format!(
                            "Qwen4 recorded-HIP oracle failed: {reason}"
                        )));
                    }
                }
            } else if aql_route {
                // SAFETY: same contract as the PM4 route above.
                match unsafe { gpu.replay.replay_linear_aql(next_position) } {
                    Ok(_) => true,
                    Err(reason) => {
                        gpu.replay.poison(format!("Qwen4 AQL replay failed: {reason}"));
                        return Err(invalid(format!("Qwen4 retained replay failed: {reason}")));
                    }
                }
            } else {
                false
            };

            if !routed {
                // Parity harness only: a QSA projection hook counts slots from
                // each forward's first QSA step (no-op with no hook installed).
                hipfire_dispatch::pipeline::reset_qsa_projection_slot();
                let execute = |gpu: &mut Gpu, steps: &[Step<'_>]| {
                    execute_validated_steps(gpu, &ctx, steps).map_err(|error| {
                        Qwen4GpuForwardError::Dispatch(format!(
                            "execute Qwen4 layer program: {error:?}"
                        ))
                    })
                };
                if !dump_ends.is_empty() && expert_stage.is_none() && self.qsa_tap.is_none() {
                    // `HIPFIRE_CHUNK_DUMP_DIR`: run the chunk in segments that end after
                    // each selected layer and dump that boundary (diagnostic only).
                    let dump = chunk_dump.expect("dump ends imply the dump is enabled");
                    let mut start = 0;
                    for index in 0..steps.len() {
                        if ple_split == Some(index) {
                            execute(gpu, &steps[start..index])?;
                            stage_ple(gpu)?;
                            start = index;
                        }
                        if let Some(&(layer, _, bf16_grouped)) =
                            dump_ends.iter().find(|(_, end, _)| *end == index + 1)
                        {
                            execute(gpu, &steps[start..=index])?;
                            start = index + 1;
                            dump.dump_layer(
                                gpu,
                                &self.scratch,
                                &config,
                                layer,
                                bundle.state.position,
                                n,
                                bf16_grouped,
                            )
                            .map_err(|error| invalid(format!("chunk dump: {error}")))?;
                        }
                    }
                    execute(gpu, &steps[start..])?;
                } else if let Some(tap) = self.qsa_tap.as_mut() {
                    // Parity tap: run up to and including each QSA step, then
                    // let the observer read that layer's scratch and state.
                    let mut start = 0;
                    let mut qsa_slot = 0;
                    for (index, step) in steps.iter().enumerate() {
                        if ple_split == Some(index) {
                            execute(gpu, &steps[start..index])?;
                            stage_ple(gpu)?;
                            start = index;
                        }
                        if let Step::IndexedAttention(op) = step {
                            execute(gpu, &steps[start..=index])?;
                            start = index + 1;
                            tap(gpu, qsa_slot, op).map_err(invalid)?;
                            qsa_slot += 1;
                        }
                    }
                    execute(gpu, &steps[start..])?;
                } else if let Some(stage) = expert_stage {
                    // One interpreter call over the whole program: the stage
                    // waits and refills run as MoE hooks, so the fusions across
                    // a MoE call (HC read/write pairing, the combine zero-init,
                    // the shared-down fold) apply as on an unstaged forward.
                    let mut hooks = StagedMoeHooks {
                        stage,
                        bundle,
                        moe: &self.moe,
                        layer: 0,
                        error: None,
                    };
                    let mut run = |gpu: &mut Gpu, steps: &[Step<'_>]| {
                        execute_validated_steps_with_moe_hooks(gpu, &ctx, steps, &mut hooks)
                            .map_err(|error| {
                                hooks.error.take().unwrap_or_else(|| {
                                    Qwen4GpuForwardError::Dispatch(format!(
                                        "execute Qwen4 layer program: {error:?}"
                                    ))
                                })
                            })
                    };
                    match ple_split {
                        Some(split) => {
                            run(gpu, &steps[..split])?;
                            stage_ple(gpu)?;
                            run(gpu, &steps[split..])?;
                        }
                        None => run(gpu, &steps)?,
                    }
                } else {
                    let trace = self.route_trace.as_ref().filter(|_| n == 1);
                    match (ple_split, trace) {
                        (split, Some(trace)) => {
                            let k = config.num_experts_per_tok;
                            let mut start = 0usize;
                            let mut layer = 0usize;
                            for index in 0..steps.len() {
                                if split == Some(index) {
                                    if start < index {
                                        execute(gpu, &steps[start..index])?;
                                    }
                                    stage_ple(gpu)?;
                                    start = index;
                                }
                                if matches!(steps[index], Step::Moe(_)) {
                                    execute(gpu, &steps[start..=index])?;
                                    start = index + 1;
                                    gpu.copy_d2d(
                                        &self.scratch.moe_topk_indices,
                                        &trace.device.sub_offset(layer * k, k),
                                        k * std::mem::size_of::<i32>(),
                                    )?;
                                    layer += 1;
                                }
                            }
                            if start < steps.len() {
                                execute(gpu, &steps[start..])?;
                            }
                            traced = true;
                        }
                        (Some(split), None) => {
                            execute(gpu, &steps[..split])?;
                            stage_ple(gpu)?;
                            execute(gpu, &steps[split..])?;
                        }
                        (None, None) => execute(gpu, &steps)?,
                    }
                }
            }

            if let Some(capture) = wide_hidden_capture.as_ref() {
                let wide_elements = n * wide;
                let source = view(&self.scratch.streams, 0, wide_elements);
                let destination = f32_view(capture, 0, wide_elements);
                if bf16_state {
                    rdna_compute::tensor_ops::hc_state_bf16_to_f32(
                        gpu,
                        &source,
                        &destination,
                        wide_elements,
                    )?;
                } else {
                    gpu.copy_d2d(&source, &destination, source.byte_size())?;
                }
            }

            if let Some(requested_rows) = requested_rows {
                let final_hyper = final_hyper
                    .as_ref()
                    .ok_or_else(|| invalid("Qwen4 final hyper disappeared before execution"))?;
                let lm_head = lm_head
                    .as_ref()
                    .ok_or_else(|| invalid("Qwen4 LM head disappeared before execution"))?;
                let logits = logits.ok_or_else(|| {
                    invalid("Qwen4 logits output disappeared before execution")
                })?;
                execute_final_hyper(
                    gpu,
                    dims,
                    final_hyper,
                    &self.scratch.streams,
                    &scratch_desc,
                    n,
                    bf16_state,
                )
                .map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!("execute Qwen4 final hyper: {error:?}"))
                })?;
                let final_hidden = view(&self.scratch.hc_mixed, 0, n * config.hidden_size);
                // The head shares the trunk's basis scratch: each projection
                // consumes its rotated rows before the next one writes.
                let head_rotation =
                    view(&self.scratch.rotation, 0, n * config.hidden_size);
                execute_lm_head(
                    gpu,
                    lm_head,
                    &final_hidden,
                    logits,
                    n,
                    requested_rows,
                    Some(&head_rotation),
                )
                .map_err(
                    |error| {
                        Qwen4GpuForwardError::Dispatch(format!("execute Qwen4 LM head: {error:?}"))
                    },
                )?;

                if let Some(top1) = top1 {
                    argmax_f32(
                        gpu,
                        &ArgmaxF32 { logits,
                        indices: top1,
                        rows: requested_rows,
                        vocab: config.vocab_size, },
                    )?;
                }
            }

            // The restores join the compute stream last, so the head overlaps
            // them and anything not hidden is charged to this forward.
            if let Some(stage) = expert_stage {
                stage.join(gpu)?;
            }

            drop(steps);
            Ok(qsa_commits)
        })();
        let read_tokens: [u32; 1];
        let tokens = match device_read {
            Some(token) => {
                read_tokens = [token];
                &read_tokens[..]
            }
            None => tokens,
        };
        match attempt {
            Ok(qsa_commits) => {
                if let Some(ple) = ple.as_mut() {
                    ple.complete();
                }
                for (slot, full_len, raw_len, pooled_len, selected_len, position) in qsa_commits {
                    let state = bundle
                        .state
                        .qsa_mut(slot)
                        .ok_or_else(|| invalid("QSA state slot disappeared"))?;
                    state.full_len = full_len;
                    state.raw_len = raw_len;
                    state.pooled_len = pooled_len;
                    state.selected_len = selected_len;
                    state.position = position;
                    state.clamp_f16_valid();
                }
                let mut next_history = next_history;
                for token in tokens.iter().copied() {
                    next_history.push(token);
                }
                bundle.state.ple_history = next_history;
                // This forward's lease is released, so the reader has a free
                // staging buffer: enqueue the next chunk's rows, hashed from
                // the history just committed, while the GPU runs this chunk.
                // The next forward adopts the materialized ticket when its ids
                // match exactly (bytes are a pure function of the ids), else
                // drops it. Best effort: a refused enqueue only loses overlap.
                if let Some(owner) = bundle.ple.as_ref().filter(|_| !ple_lookahead.is_empty()) {
                    let fit = owner.rows.max_rows_per_prefetch() / ple_head_count;
                    let ahead = &ple_lookahead[..ple_lookahead.len().min(fit)];
                    self.ple_ahead = owner
                        .rows
                        .prefetch(
                            owner.rows.current_epoch(),
                            next_history.row_ids(&owner.metadata, ahead),
                        )
                        .ok();
                }
                bundle.state.commit_row_capture(n);
                bundle.state.position = next_position
                    .checked_add(n)
                    .ok_or_else(|| invalid("Qwen4 forward position overflows at commit"))?;
                // Close the retained window this forward opened, report the census
                // and install (or refuse) the prepared plan. This is the only place
                // that knows both ends of one recorded body, so the window can never
                // outlive the forward that opened it.
                if gpu.replay.should_auto_finalize_capture() {
                    finish_qwen4_capture(gpu, diagnostic_capture, launched_before);
                }
                if traced {
                    self.write_route_trace(gpu, next_position)?;
                }
                Ok(())
            }
            Err(error) => {
                bundle.state.invalidate_f16_arenas();
                // A staged forward that failed before its join may have left
                // a donor overwritten; nothing may read it until restored.
                let error = match self.expert_stage.as_ref().map(|stage| stage.restore_blocking(gpu)) {
                    Some(Err(restore)) => {
                        invalid(format!("{error}; G2 donor expert restore failed: {restore}"))
                    }
                    _ => error,
                };
                if gpu.replay.is_recording() {
                    // A capture window that failed inside the body must not keep
                    // failing every later forward.
                    gpu.replay
                        .poison("qwen4 capture failed; later forwards fall back to HIP");
                    eprintln!("[redline] qwen4 capture failed; later forwards fall back to HIP");
                }
                let aborted = match ple.as_mut() {
                    Some(ple) => ple.abort(),
                    None => Ok(0),
                };
                match aborted {
                    Ok(_) => Err(error),
                    Err(cleanup) => Err(Qwen4GpuForwardError::Ple(format!(
                        "forward attempt failed: {error}; PLE cleanup failed: {cleanup}"
                    ))),
                }
            }
        }
    }
}

/// A recorded capture body that cannot become a prepared tape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaptureCensusError {
    /// The recorder saw a different number of launches than the body issued
    /// (#397-style truncated tape).
    LaunchMismatch { launched: usize, recorded: usize },
}

impl fmt::Display for CaptureCensusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LaunchMismatch { launched, recorded } => write!(
                f,
                "qwen4 capture census mismatch: {launched} launched, {recorded} recorded \
                 (truncated tape refused)"
            ),
        }
    }
}

/// Fail closed on a census mismatch: poison the retained route (sticky HIP
/// fallback) so the tape is never closed, prepared or installed.
fn gate_capture_census(
    replay: &mut ReplayController,
    launched: usize,
    recorded: usize,
) -> Result<(), CaptureCensusError> {
    if launched == recorded {
        return Ok(());
    }
    let error = CaptureCensusError::LaunchMismatch { launched, recorded };
    replay.poison(error.to_string());
    Err(error)
}

/// Close a recorded Qwen4 body: census, then install the prepared plan, or refuse
/// it for a measurement-only capture.
///
/// Prepare failure after a successful HIP warmup poisons the route (sticky
/// fallback) rather than failing the request that already produced output.
fn finish_qwen4_capture(gpu: &mut Gpu, diagnostic_capture: bool, launched_before: u64) {
    let launched = hip_bridge::launch_counters::launch_kernel::count();
    let effects = gpu.replay.capture_window_effects();
    let summary = gpu.replay.capture_summary();
    let recorded = summary.launch_count;
    let computed = launched.saturating_sub(launched_before) as usize;
    eprintln!(
        "[redline] qwen4 capture census: computed={computed} recorded={recorded} \
         unique_kernels={} sequence_hash={:016x}",
        summary.unique_kernel_count, summary.sequence_hash
    );
    eprintln!(
        "[redline] qwen4 capture census: effects inside window — htod={} dtod={} dtoh={} memset={}",
        effects.htod, effects.dtod, effects.dtoh, effects.memset
    );
    if computed != recorded {
        eprintln!(
            "[redline] qwen4 capture census: MISMATCH — {computed} launched, {recorded} recorded \
             (#397-style truncated tape)"
        );
    }
    if !effects.replayable() {
        eprintln!(
            "[redline] qwen4 capture census: effect-incomplete — {} device copy(ies), {} readback(s) \
             and {} memset(s) inside the window are state the tape cannot replay; prepare refuses it",
            effects.dtod, effects.dtoh, effects.memset
        );
    }
    if diagnostic_capture {
        let reason =
            "diagnostic specialized-MoE capture only; no plan installed (the model runs on \
                      HIP)";
        gpu.replay.poison(reason);
        eprintln!("[redline] qwen4: {reason}");
        return;
    }
    // A truncated tape must never be prepared: refuse before the sync, close
    // and prepare below so no plan can be installed from it.
    if let Err(error) = gate_capture_census(&mut gpu.replay, computed, recorded) {
        eprintln!("[redline] falling back to HIP: {error}");
        return;
    }
    if let Err(reason) = gpu.hip.device_synchronize() {
        let reason = format!("qwen4 capture sync failed: {reason:?}");
        gpu.replay.poison(reason.clone());
        eprintln!("[redline] qwen4 capture could not be closed or prepared: {reason}");
        return;
    }
    if let Err(reason) = gpu.replay.finish_capture() {
        gpu.replay.poison(reason);
        eprintln!("[redline] qwen4 capture could not be closed or prepared: {reason}");
        return;
    }
    let launches = gpu.replay.recorded_launches().len();
    let prepared = if gpu.replay.uses_pm4_transport() {
        gpu.replay
            .prepare_pm4_prefix(gpu.device_id as usize, launches)
            .map(|_| ())
    } else {
        gpu.replay
            .prepare_linear_aql(gpu.device_id as usize)
            .map(|_| ())
    };
    match prepared {
        Ok(()) => eprintln!(
            "[redline] qwen4 retained body prepared: transport={} dispatches={launches}",
            gpu.replay.transport_name()
        ),
        Err(reason) => {
            let reason = format!("Redline prepare after warmup failed: {reason}");
            gpu.replay.poison(reason.clone());
            eprintln!("[redline] falling back to HIP: {reason}");
        }
    }
}

#[cfg(test)]
mod capture_census_tests {
    use super::{gate_capture_census, CaptureCensusError};
    use rdna_compute::replay::{ReplayBackendRequest, ReplayController, ReplayState};

    #[test]
    fn census_mismatch_poisons_route_and_refuses_tape() {
        let mut replay = ReplayController::new(ReplayBackendRequest::Auto);
        assert_eq!(replay.state(), ReplayState::RecordingWarmup);
        let error = gate_capture_census(&mut replay, 7, 5).unwrap_err();
        assert_eq!(
            error,
            CaptureCensusError::LaunchMismatch { launched: 7, recorded: 5 }
        );
        let text = error.to_string();
        assert!(text.contains("7 launched") && text.contains("5 recorded"), "{text}");
        // Poisoned: the route is on HIP, the capture can no longer be closed
        // into a tape, and the reason carries both counts.
        assert_eq!(replay.state(), ReplayState::Fallback);
        assert!(replay.finish_capture().is_err());
        let reason = replay.fallback_reason().expect("fallback reason");
        assert!(reason.contains("7 launched") && reason.contains("5 recorded"), "{reason}");
    }

    #[test]
    fn census_equal_is_accepted_without_touching_route() {
        let mut replay = ReplayController::new(ReplayBackendRequest::Auto);
        assert_eq!(gate_capture_census(&mut replay, 5, 5), Ok(()));
        assert_eq!(replay.state(), ReplayState::RecordingWarmup);
        assert!(replay.fallback_reason().is_none());
    }
}

#[cfg(test)]
mod ep_table_tests {
    use super::{
        qwen4_ep_buffer_index, qwen4_expert_table_entries, Qwen4EpPlacement, Qwen4ExpertEntry,
        Qwen4ExpertOrigin, Qwen4ExpertTableLayout, Qwen4GpuForwardError, Qwen4PeerExpertSource,
        QWEN4_EP_GATHER_BUFFERS, QWEN4_EP_GATHER_MAX_ROWS,
    };

    const EXPERTS: usize = 512;
    const GATE_UP: usize = 1_740_800;
    const DOWN: usize = 870_400;

    fn compact(base: usize, gate_up_base: usize, experts: usize) -> Qwen4PeerExpertSource {
        Qwen4PeerExpertSource {
            gate_up_base: gate_up_base + base,
            gate_up_bytes: experts * GATE_UP,
            down_base: 0x4000_0000 + base,
            down_bytes: experts * DOWN,
        }
    }

    fn dummy() -> Qwen4PeerExpertSource {
        Qwen4PeerExpertSource {
            gate_up_base: 0x9000_0000,
            gate_up_bytes: GATE_UP,
            down_base: 0x9800_0000,
            down_bytes: DOWN,
        }
    }

    fn entries(
        layout: &Qwen4ExpertTableLayout<'_>,
        local: Qwen4PeerExpertSource,
    ) -> Result<Vec<Qwen4ExpertEntry>, Qwen4GpuForwardError> {
        qwen4_expert_table_entries(layout, local, EXPERTS, GATE_UP, DOWN)
    }

    fn message(result: Result<Vec<Qwen4ExpertEntry>, Qwen4GpuForwardError>) -> String {
        match result {
            Err(Qwen4GpuForwardError::Invalid(message)) => message,
            other => panic!("expected an Invalid error, got {other:?}"),
        }
    }

    #[test]
    fn gather_buffer_index_alternates_over_the_trunk_and_isolates_the_mtp_head() {
        assert_eq!(QWEN4_EP_GATHER_BUFFERS, 4);
        assert_eq!(QWEN4_EP_GATHER_MAX_ROWS, 8);
        assert_eq!(qwen4_ep_buffer_index(0, 48), 0);
        assert_eq!(qwen4_ep_buffer_index(1, 48), 1);
        assert_eq!(qwen4_ep_buffer_index(2, 48), 0);
        assert_eq!(qwen4_ep_buffer_index(47, 48), 1);
        assert_eq!(qwen4_ep_buffer_index(48, 48), 2);
        assert_eq!(qwen4_ep_buffer_index(49, 48), 3);
        assert_eq!(qwen4_ep_buffer_index(50, 48), 2);
        assert_eq!(qwen4_ep_buffer_index(1001, 48), 3);
    }

    #[test]
    fn single_table_is_the_contiguous_expert_walk() {
        let local = compact(0, 0x1000_0000, EXPERTS);
        let table = entries(&Qwen4ExpertTableLayout::Single, local).unwrap();
        assert_eq!(table.len(), EXPERTS);
        for (expert, entry) in table.iter().enumerate() {
            assert_eq!(entry.origin, Qwen4ExpertOrigin::Local);
            assert_eq!(entry.gate_up_ptr, local.gate_up_base + expert * GATE_UP);
            assert_eq!(entry.down_ptr, local.down_base + expert * DOWN);
        }
    }

    #[test]
    fn single_table_refuses_a_short_tensor() {
        let mut local = compact(0, 0x1000_0000, EXPERTS);
        local.down_bytes -= 1;
        let text = message(entries(&Qwen4ExpertTableLayout::Single, local));
        assert!(text.contains("down expert slot 511") && text.contains("exceeds"), "{text}");
    }

    #[test]
    fn ep_table_maps_owned_to_compact_local_and_the_rest_to_the_dummy() {
        for (ranks, rank) in [(2usize, 0usize), (2, 1), (4, 0), (4, 3)] {
            let placement = Qwen4EpPlacement { rank, ranks };
            let local = compact(0, 0x1000_0000, EXPERTS / ranks);
            let layout = Qwen4ExpertTableLayout::Ep {
                placement,
                dummy: dummy(),
            };
            let table = entries(&layout, local).unwrap();
            assert_eq!(table.len(), EXPERTS);
            let mut owned = 0;
            for (expert, entry) in table.iter().enumerate() {
                if expert % ranks == rank {
                    owned += 1;
                    assert_eq!(entry.origin, Qwen4ExpertOrigin::Local);
                    assert_eq!(
                        entry.gate_up_ptr,
                        local.gate_up_base + (expert / ranks) * GATE_UP
                    );
                    assert_eq!(entry.down_ptr, local.down_base + (expert / ranks) * DOWN);
                } else {
                    assert_eq!(entry.origin, Qwen4ExpertOrigin::Dummy);
                    assert_eq!(entry.gate_up_ptr, dummy().gate_up_base);
                    assert_eq!(entry.down_ptr, dummy().down_base);
                }
            }
            assert_eq!(owned, EXPERTS / ranks);
        }
    }

    #[test]
    fn ep_table_requires_exactly_the_compact_expert_count_and_one_dummy_expert() {
        let placement = Qwen4EpPlacement { rank: 1, ranks: 4 };
        let layout = Qwen4ExpertTableLayout::Ep {
            placement,
            dummy: dummy(),
        };
        // A single-device tensor (all 512 experts) is not a compact source.
        let text = message(entries(&layout, compact(0, 0x1000_0000, EXPERTS)));
        assert!(text.contains("exactly 128 experts"), "{text}");
        // One expert short.
        let text = message(entries(&layout, compact(0, 0x1000_0000, 127)));
        assert!(text.contains("exactly 128 experts"), "{text}");
        let mut big_dummy = dummy();
        big_dummy.down_bytes += 4;
        let layout = Qwen4ExpertTableLayout::Ep {
            placement,
            dummy: big_dummy,
        };
        let text = message(entries(&layout, compact(0, 0x1000_0000, 128)));
        assert!(text.contains("dummy expert"), "{text}");
    }

    #[test]
    fn ep_placement_must_shard_evenly() {
        let local = compact(0, 0x1000_0000, EXPERTS / 4);
        for placement in [
            Qwen4EpPlacement { rank: 4, ranks: 4 },
            Qwen4EpPlacement { rank: 0, ranks: 0 },
            Qwen4EpPlacement { rank: 0, ranks: 3 },
        ] {
            let layout = Qwen4ExpertTableLayout::Ep {
                placement,
                dummy: dummy(),
            };
            assert!(entries(&layout, local).is_err(), "{placement:?}");
        }
    }

    fn peers(ranks: usize) -> Vec<Qwen4PeerExpertSource> {
        (0..ranks)
            .map(|rank| {
                let mut source = compact(0, 0x1000_0000 + rank * 0x1_0000_0000, EXPERTS / ranks);
                source.down_base += rank * 0x1_0000_0000;
                source
            })
            .collect()
    }

    #[test]
    fn full_table_maps_owned_local_and_non_owned_into_the_owner_peer() {
        for (ranks, rank) in [(2usize, 0usize), (2, 1), (4, 2)] {
            let placement = Qwen4EpPlacement { rank, ranks };
            let peers = peers(ranks);
            let layout = Qwen4ExpertTableLayout::Full {
                placement,
                peers: &peers,
            };
            let table = entries(&layout, peers[rank]).unwrap();
            assert_eq!(table.len(), EXPERTS);
            for (expert, entry) in table.iter().enumerate() {
                let owner = expert % ranks;
                let slot = expert / ranks;
                let expected_origin = if owner == rank {
                    Qwen4ExpertOrigin::Local
                } else {
                    Qwen4ExpertOrigin::Peer(owner)
                };
                assert_eq!(entry.origin, expected_origin, "expert {expert}");
                assert_eq!(
                    entry.gate_up_ptr,
                    peers[owner].gate_up_base + slot * GATE_UP,
                    "expert {expert}"
                );
                assert_eq!(
                    entry.down_ptr,
                    peers[owner].down_base + slot * DOWN,
                    "expert {expert}"
                );
            }
        }
    }

    #[test]
    fn full_table_refuses_inconsistent_or_short_peers() {
        let placement = Qwen4EpPlacement { rank: 0, ranks: 2 };
        let good = peers(2);
        let layout = Qwen4ExpertTableLayout::Full {
            placement,
            peers: &good,
        };
        // Peer list for the wrong rank count.
        let wrong_count = &good[..1];
        let layout_count = Qwen4ExpertTableLayout::Full {
            placement,
            peers: wrong_count,
        };
        let text = message(entries(&layout_count, good[0]));
        assert!(text.contains("needs 2 peer sources"), "{text}");
        // `peers[rank]` must be this rank's own tensors.
        let text = message(entries(&layout, good[1]));
        assert!(text.contains("does not name this rank's own"), "{text}");
        // A peer whose tensor is one expert short is caught by the slot bounds.
        let mut short = good.clone();
        short[1].gate_up_bytes -= GATE_UP;
        let layout_short = Qwen4ExpertTableLayout::Full {
            placement,
            peers: &short,
        };
        let text = message(entries(&layout_short, short[0]));
        assert!(text.contains("peer gate/up expert slot 255") && text.contains("exceeds"), "{text}");
        // A null peer base is refused rather than mapped.
        let mut null = good.clone();
        null[1].down_base = 0;
        let layout_null = Qwen4ExpertTableLayout::Full {
            placement,
            peers: &null,
        };
        let text = message(entries(&layout_null, null[0]));
        assert!(text.contains("null"), "{text}");
    }
}

#[cfg(test)]
mod tests {
    use super::{
        owner_pitch_extent, qwen4_forward_device_bytes, qwen4_spec_logit_rows, uploaded_ids_range,
        CompactScratchTerms, Qwen4ForwardScratchMode as Mode, Qwen4GpuForwardScratch as Scratch,
        ScratchField as F, ScratchSlot, QWEN4_COMPACT_FALLBACK_ROWS,
    };
    use crate::config::{compact_test_config, Qwen4Config};

    fn fn_config() -> Qwen4Config {
        compact_test_config()
    }

    fn scratch_bytes(rows: usize, mode: Mode) -> u64 {
        Scratch::device_bytes(&fn_config(), rows, mode).expect("FN scratch bytes")
    }

    fn plan(rows: usize, mode: Mode) -> Vec<ScratchSlot> {
        Scratch::plan(&fn_config(), rows, mode).expect("FN scratch plan").0
    }

    fn field_bytes(slots: &[ScratchSlot], field: F) -> u64 {
        slots[field as usize].bytes().expect("field bytes") as u64
    }

    fn owned_bytes(slots: &[ScratchSlot]) -> u64 {
        slots
            .iter()
            .filter(|slot| slot.owner.is_none())
            .map(|slot| slot.bytes().expect("slot bytes") as u64)
            .sum()
    }

    /// `G(r)`: tile-padded grouped rows of `r` rows, top-10 of 512 experts.
    fn grouped_rows(rows: u64) -> u64 {
        let slots = 10 * rows;
        (slots + 15 * slots.min(512)).div_ceil(16) * 16
    }

    #[test]
    fn npu_bank_sharing_covers_every_field_and_never_crosses_a_bank_switch() {
        use super::NpuBankLifetime as L;
        // Event order from pair_scoped, for two layers with PLE on layer 1:
        // initialize A/B, prepare(0,A/B), G(0,A), prepare(1,A), G(0,B),
        // prepare(1,B), G(1,A/B), finish A/B. Each event is a complete
        // same-stream sequence; an interval is one produced value's lifetime,
        // not the lifetime of its allocation.
        let banks = [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1];
        let local_init = [(0, 0), (1, 1)];
        let residual = [(0, 10), (1, 11)];
        let input = [(2, 4), (3, 6), (5, 8), (7, 9)];
        // The attention write's gates: prepare's paired read, or the MoE row
        // fold of the bank's previous fragment, to the next fragment.
        let gates = [(2, 4), (3, 6), (4, 8), (5, 8), (6, 9), (7, 9)];
        let ple_rows = [(0, 5), (1, 7)];
        let prepare_and_fragment = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
            (7, 7), (8, 8), (9, 9), (10, 10), (11, 11)];
        let fragments = [(4, 4), (6, 6), (8, 8), (9, 9)];
        let finish = [(10, 10), (11, 11)];
        // This independent data-flow table is also the full private/shared
        // inventory. HC gates cross from prepare (or the previous fragment's
        // row fold) to the bank's next fragment; fragment-local HC fusions
        // consume their gates before returning. PLE normed rows are retained
        // only for verify rollback, not prompt pairs; session capture copies
        // fixed state, not this field.
        let rules: &[(L, &[F], &[(usize, usize)])] = &[
            (L::Initialize, &[F::TokenIds, F::EmbeddingRot, F::Embeddings, F::PleStaged], &local_init),
            (L::Residual, &[F::Streams], &residual),
            (L::PreparedInput, &[F::HcMixed], &input),
            (L::PreparedInput, &[F::HcGates], &gates),
            (L::StagedRows, &[F::PleRows], &ple_rows),
            (L::PrepareAndFragment, &[F::HcNormalized, F::HcLow, F::HcUp,
                F::Rotation, F::GdnBf16, F::PleKey, F::PleValue, F::PleQuery,
                F::PleGated, F::PleNormed, F::PleOutput], &prepare_and_fragment),
            (L::Fragment, &[F::Projection, F::Projection2, F::GdnA,
                F::GdnB, F::GdnGate, F::GdnBeta, F::GdnRecurrentOutput, F::GdnZ,
                F::GdnOutput, F::QsaIndex, F::QsaQgate, F::QsaK, F::QsaV,
                F::QsaOutput, F::QsaSelected, F::AttentionOutput, F::RouterLogits,
                F::MoeXRot, F::MoeGateUp, F::MoeGate, F::MoeUp, F::MoeHidden,
                F::FfnOutput, F::MoeSharedOutput, F::MoeGateBatch, F::MoeUpBatch,
                F::MoeRotBatch, F::MoeTopkIndices, F::MoeTopkWeights,
                F::MoeDownExpanded, F::MoeScalar, F::MoeExpertTokenCounts,
                F::MoeExpertOffsets, F::MoeSortedSlotIndex, F::MoeExpertTileIds,
                F::MoeInversePerm, F::MoeYGateUpGrouped, F::MoeYDownGrouped], &fragments),
            (L::Finish, &[F::Logits], &finish),
        ];
        let mut seen = [false; F::COUNT];
        for &(lifetime, fields, intervals) in rules {
            let crosses_bank = intervals.iter().any(|&(first, last)|
                banks[first..=last].iter().any(|&bank| bank != banks[first]));
            for &field in fields {
                assert!(!std::mem::replace(&mut seen[field as usize], true), "{field:?} repeated");
                assert_eq!(field.npu_bank_lifetime(), lifetime, "{field:?}");
                assert_eq!(field.npu_bank_lifetime().private(), crosses_bank, "{field:?}");
            }
        }
        assert!(seen.into_iter().all(|field| field));
        for (index, field) in F::ALL.into_iter().enumerate() {
            assert_eq!(field as usize, index);
        }
        assert_eq!(F::ALL.into_iter().filter(|field| field.npu_bank_lifetime().private())
            .collect::<Vec<_>>(), [F::Streams, F::HcMixed, F::HcGates, F::PleRows]);
    }

    #[test]
    fn npu_private_bank_bytes_are_pinned_at_full_and_partial_rows() {
        let config = fn_config();
        for rows in [512, 8151, 8192] {
            let private = Scratch::npu_bank_device_bytes(&config, rows).unwrap();
            let slots = plan(rows, Mode::Full);
            let expected = [F::Streams, F::HcMixed, F::HcGates, F::PleRows].into_iter()
                .map(|field| field_bytes(&slots, field)).sum::<u64>();
            assert_eq!(private, expected);
            assert_eq!(private, rows as u64 * (4 * 2560 + 2560 + 4 + 2560) * 4);
        }
        let private = Scratch::npu_bank_device_bytes(&config, 8192).unwrap();
        assert_eq!(private, 503_447_552);
        assert_eq!(scratch_bytes(8192, Mode::Full) - private, 8_454_551_428);
    }

    #[test]
    fn fn_scratch_bytes_are_pinned() {
        assert_eq!(scratch_bytes(4096, Mode::Full), 4_538_503_044);
        assert_eq!(scratch_bytes(8192, Mode::Full), 8_957_998_980);
        assert_eq!(scratch_bytes(4096, Mode::FnSymGfx1201), 2_314_473_348);
        assert_eq!(scratch_bytes(8192, Mode::FnSymGfx1201), 4_511_905_668);
        assert_eq!(
            scratch_bytes(4096, Mode::Full) - scratch_bytes(4096, Mode::FnSymGfx1201),
            2_224_029_696
        );
        assert_eq!(
            scratch_bytes(8192, Mode::Full) - scratch_bytes(8192, Mode::FnSymGfx1201),
            4_446_093_312
        );
        // The 8192-row compact scratch is smaller than today's 4096-row one.
        assert!(scratch_bytes(8192, Mode::FnSymGfx1201) <= scratch_bytes(4096, Mode::Full));
        assert_eq!(
            scratch_bytes(4096, Mode::Full) - scratch_bytes(8192, Mode::FnSymGfx1201),
            26_597_376
        );
        // The estimator adds the same 64-row speculative logits to either plan.
        let config = fn_config();
        let spec = |rows: usize| qwen4_spec_logit_rows(rows) as u64 * (248_320 * 4 + 4);
        for (rows, mode) in [
            (4096, Mode::Full),
            (8192, Mode::Full),
            (4096, Mode::FnSymGfx1201),
            (8192, Mode::FnSymGfx1201),
        ] {
            assert_eq!(
                qwen4_forward_device_bytes(&config, rows, mode),
                Some(scratch_bytes(rows, mode) + spec(rows))
            );
        }
        assert!(
            qwen4_forward_device_bytes(&config, 8192, Mode::FnSymGfx1201)
                <= qwen4_forward_device_bytes(&config, 4096, Mode::Full)
        );
    }

    #[test]
    fn compact_route_buffers_follow_the_frozen_capacities() {
        // (rows, gate/up/rot batch, expanded down, grouped gate/up, grouped down)
        for (rows, batch, expanded, gate_up, down) in [
            (4096usize, 13_107_200u64, 52_428_800u64, 65_536_000u64, 249_036_800u64),
            (8192, 13_107_200, 83_886_080, 114_688_000, 458_752_000),
        ] {
            let slots = plan(rows, Mode::FnSymGfx1201);
            for field in [F::MoeGateBatch, F::MoeUpBatch, F::MoeRotBatch] {
                assert_eq!(field_bytes(&slots, field), batch, "{field:?} at {rows}");
            }
            assert_eq!(field_bytes(&slots, F::MoeDownExpanded), expanded);
            assert_eq!(field_bytes(&slots, F::MoeYGateUpGrouped), gate_up);
            assert_eq!(field_bytes(&slots, F::MoeYDownGrouped), down);
        }
        // Today's plan at 8192: 200/200/200/800/437.5/875 MiB.
        let full = plan(8192, Mode::Full);
        for field in [F::MoeGateBatch, F::MoeUpBatch, F::MoeRotBatch] {
            assert_eq!(field_bytes(&full, field), 209_715_200);
        }
        assert_eq!(field_bytes(&full, F::MoeDownExpanded), 838_860_800);
        assert_eq!(field_bytes(&full, F::MoeYGateUpGrouped), 458_752_000);
        assert_eq!(field_bytes(&full, F::MoeYDownGrouped), 917_504_000);
    }

    #[test]
    fn compact_aliases_borrow_one_owner_each_and_ple_normed_stays_dedicated() {
        let full = plan(8192, Mode::Full);
        assert_eq!(full.len(), F::COUNT);
        assert!(full.iter().all(|slot| slot.owner.is_none() && slot.pitch.is_none()));
        let compact = plan(8192, Mode::FnSymGfx1201);
        let owner = |field: F| compact[field as usize].owner;
        assert_eq!(owner(F::HcNormalized), Some(F::Projection));
        assert_eq!(owner(F::PleQuery), Some(F::Projection));
        assert_eq!(owner(F::HcUp), Some(F::Projection2));
        assert_eq!(owner(F::PleKey), Some(F::Projection2));
        assert_eq!(owner(F::PleGated), Some(F::Rotation));
        assert_eq!(owner(F::PleOutput), Some(F::MoeYDownGrouped));
        assert_eq!(owner(F::PleValue), Some(F::MoeSharedOutput));
        assert_eq!(owner(F::GdnRecurrentOutput), Some(F::QsaQgate));
        assert_eq!(owner(F::PleNormed), None);
        assert_eq!(compact.iter().filter(|slot| slot.owner.is_some()).count(), 8);
        // Only owners are charged; no view's owner is itself a view and every
        // view fits inside its owner.
        assert_eq!(owned_bytes(&compact), scratch_bytes(8192, Mode::FnSymGfx1201));
        assert_eq!(owned_bytes(&full), scratch_bytes(8192, Mode::Full));
        for slot in compact.iter().filter(|slot| slot.owner.is_some()) {
            let owner = compact[slot.owner.unwrap() as usize];
            assert!(owner.owner.is_none());
            assert_eq!(owner.dtype, slot.dtype);
            assert!(slot.elements <= owner.elements);
        }
        // Alias-set members share the owner's row pitch.
        let pitch = |field: F| compact[field as usize].pitch;
        for field in [F::Projection, F::HcNormalized, F::PleQuery] {
            assert_eq!(pitch(field), Some(12_288), "{field:?}");
        }
        for field in [F::Projection2, F::HcUp, F::PleKey] {
            assert_eq!(pitch(field), Some(12_288), "{field:?}");
        }
        for field in [F::QsaQgate, F::GdnRecurrentOutput] {
            assert_eq!(pitch(field), Some(12_288), "{field:?}");
        }
        for field in [F::Rotation, F::PleGated, F::PleOutput] {
            assert_eq!(pitch(field), Some(10_240), "{field:?}");
        }
        for field in [F::MoeSharedOutput, F::PleValue] {
            assert_eq!(pitch(field), Some(2_560), "{field:?}");
        }
        assert_eq!(pitch(F::MoeYDownGrouped), None);
        assert_eq!(pitch(F::PleNormed), None);
    }

    #[test]
    fn compact_layout_holds_at_the_boundary_rows() {
        for rows in [1usize, 8, 9, 64, 511, 512, 513, 4096, 8192] {
            let full = plan(rows, Mode::Full);
            let compact = plan(rows, Mode::FnSymGfx1201);
            assert_eq!(owned_bytes(&full), scratch_bytes(rows, Mode::Full), "rows {rows}");
            assert_eq!(
                owned_bytes(&compact),
                scratch_bytes(rows, Mode::FnSymGfx1201),
                "rows {rows}"
            );
            assert!(owned_bytes(&compact) < owned_bytes(&full), "rows {rows}");

            let n = rows as u64;
            let c = rows.min(QWEN4_COMPACT_FALLBACK_ROWS) as u64;
            for field in [F::MoeGateBatch, F::MoeUpBatch, F::MoeRotBatch] {
                assert_eq!(field_bytes(&compact, field), c * 10 * 640 * 4);
            }
            assert_eq!(
                field_bytes(&compact, F::MoeDownExpanded),
                (c * 10 * 2560 * 4).max(n * 2560 * 4).max(n * 80)
            );
            assert_eq!(
                field_bytes(&compact, F::MoeYGateUpGrouped),
                (grouped_rows(c) * 1280 * 4).max(grouped_rows(n) * 640 * 2)
            );
            let down = field_bytes(&compact, F::MoeYDownGrouped);
            assert_eq!(
                down,
                (grouped_rows(c) * 2560 * 4).max(grouped_rows(n) * 2560 * 2)
            );
            // The borrowed PLE output must fit inside the grouped-down owner.
            assert!(down >= n * 10_240 * 4, "rows {rows}");
            if rows <= QWEN4_COMPACT_FALLBACK_ROWS {
                for field in [
                    F::MoeGateBatch,
                    F::MoeUpBatch,
                    F::MoeRotBatch,
                    F::MoeDownExpanded,
                    F::MoeYGateUpGrouped,
                    F::MoeYDownGrouped,
                ] {
                    assert_eq!(
                        field_bytes(&compact, field),
                        field_bytes(&full, field),
                        "{field:?} at {rows} rows is the full F32 size"
                    );
                }
            }
        }
    }

    #[test]
    fn scratch_plan_refuses_zero_rows_and_overflow() {
        let config = fn_config();
        for mode in [Mode::Full, Mode::FnSymGfx1201] {
            assert!(Scratch::device_bytes(&config, 0, mode).is_err());
            assert!(Scratch::device_bytes(&config, usize::MAX, mode).is_err());
            assert!(Scratch::device_bytes(&config, usize::MAX / 2, mode).is_err());
            assert_eq!(qwen4_forward_device_bytes(&config, 0, mode), None);
            assert_eq!(qwen4_forward_device_bytes(&config, usize::MAX, mode), None);
        }
    }

    #[test]
    fn compact_mode_needs_every_term() {
        let all = CompactScratchTerms {
            knob: true,
            exact_gfx1201: true,
            grouped_gemm: true,
            sym_iu4_requested: true,
            source_symmetric: true,
            ple_key_value_bf16: true,
            flash_next_profile: true,
        };
        assert_eq!(all.mode(), Mode::FnSymGfx1201);
        let off = [
            CompactScratchTerms { knob: false, ..all },
            CompactScratchTerms { exact_gfx1201: false, ..all },
            CompactScratchTerms { grouped_gemm: false, ..all },
            CompactScratchTerms { sym_iu4_requested: false, ..all },
            CompactScratchTerms { source_symmetric: false, ..all },
            CompactScratchTerms { ple_key_value_bf16: false, ..all },
            CompactScratchTerms { flash_next_profile: false, ..all },
        ];
        for terms in off {
            assert_eq!(terms.mode(), Mode::Full, "{terms:?}");
        }
    }

    #[test]
    fn compact_mode_is_exact_gfx1201_only() {
        assert!(CompactScratchTerms::is_exact_gfx1201("gfx1201"));
        for arch in [
            "gfx906", "gfx908", "gfx1010", "gfx1011", "gfx1012", "gfx1030", "gfx1031", "gfx1032",
            "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200",
            "gfx940", "gfx941", "gfx942", "gfx12010", "gfx1201x", "gfx120", "",
        ] {
            assert!(!CompactScratchTerms::is_exact_gfx1201(arch), "{arch}");
        }
    }

    #[test]
    fn owner_pitch_segments_are_disjoint_and_bounded() {
        let pitch = 12_288usize;
        let numel = 8192 * pitch;
        // Many small invocations (aggregate well past 512 rows) of unequal
        // sizes, then one large segment, at nonzero row starts.
        let mut segments = Vec::new();
        let mut begin = 0usize;
        for index in 0..700usize {
            let rows = if index % 7 == 0 { 8 } else { 1 };
            segments.push((begin, rows));
            begin += rows;
        }
        segments.push((begin, 1500));
        assert!(begin + 1500 > QWEN4_COMPACT_FALLBACK_ROWS);
        for width in [12_288usize, 10_240, 6_144] {
            let mut previous_end = 0usize;
            for &(begin, rows) in &segments {
                let (offset, len) = owner_pitch_extent(pitch, numel, begin, rows, width).unwrap();
                assert_eq!(offset, begin * pitch);
                assert_eq!(len, rows * pitch);
                assert!(offset >= previous_end, "segment at row {begin} overlaps its predecessor");
                previous_end = offset + len;
            }
            assert!(previous_end <= numel);
        }
        // Past the owner, wider than the pitch, or overflowing: refused.
        assert!(owner_pitch_extent(pitch, numel, 8191, 2, 10_240).is_err());
        assert!(owner_pitch_extent(pitch, numel, 0, 1, pitch + 1).is_err());
        assert!(owner_pitch_extent(usize::MAX, usize::MAX, 2, 1, 1).is_err());
        assert!(owner_pitch_extent(pitch, usize::MAX, usize::MAX, 1, 1).is_err());
    }

    #[test]
    fn logical_width_bases_collide_where_owner_pitch_bases_do_not() {
        // Two members of one alias set with different logical widths (a
        // 12288-wide and a 10240-wide field) serving neighbouring segments
        // A = rows 9..12 and B = row 12.
        let (a_begin, a_rows, b_begin) = (9usize, 3usize, 12usize);
        // Negative control: bases at each field's own logical width overlap.
        let a_logical = (a_begin * 12_288, (a_begin + a_rows) * 12_288);
        let b_logical_start = b_begin * 10_240;
        assert!(a_logical.0 <= b_logical_start && b_logical_start < a_logical.1);
        // Owner pitch: the same segments are disjoint.
        let pitch = 12_288usize;
        let numel = 64 * pitch;
        let (a_offset, a_len) = owner_pitch_extent(pitch, numel, a_begin, a_rows, 12_288).unwrap();
        let (b_offset, _) = owner_pitch_extent(pitch, numel, b_begin, 1, 10_240).unwrap();
        assert!(a_offset + a_len <= b_offset);
    }

    fn bytes(ids: &[u32]) -> Vec<u8> {
        ids.iter()
            .flat_map(|&id| (id as i32).to_ne_bytes())
            .collect()
    }

    #[test]
    fn uploaded_ids_cover_sub_chunk_offsets() {
        let ids = [5u32, 6, 7, 8, 9, 10];
        let host = bytes(&ids);
        assert_eq!(uploaded_ids_range(6, &host, 0, &ids), Some((0, 24)));
        assert_eq!(uploaded_ids_range(6, &host, 2, &ids[2..5]), Some((8, 12)));
        assert_eq!(uploaded_ids_range(6, &host, 5, &ids[5..]), Some((20, 4)));
    }

    #[test]
    fn uploaded_ids_refuse_unclaimed_or_out_of_bounds_rows() {
        let ids = [5u32, 6, 7, 8];
        let host = bytes(&ids);
        // Nothing claimed (HIP single-row body, device argmax, graph capture).
        assert_eq!(uploaded_ids_range(0, &host, 0, &ids), None);
        // The range may not pass the uploaded rows.
        assert_eq!(uploaded_ids_range(4, &host, 0, &[5, 6, 7, 8, 9]), None);
        assert_eq!(uploaded_ids_range(4, &host, 3, &[8, 9]), None);
        assert_eq!(uploaded_ids_range(3, &host, 2, &[7, 8]), None);
        // The host mirror must hold the claimed rows.
        assert_eq!(uploaded_ids_range(4, &host[..12], 0, &ids), None);
        assert_eq!(uploaded_ids_range(usize::MAX, &host, usize::MAX, &[1]), None);
        assert_eq!(uploaded_ids_range(4, &host, 0, &[]), None);
    }

    #[test]
    fn uploaded_ids_refuse_different_tokens() {
        let ids = [5u32, 6, 7, 8];
        let host = bytes(&ids);
        assert_eq!(uploaded_ids_range(4, &host, 0, &[5, 6, 7, 9]), None);
        assert_eq!(uploaded_ids_range(4, &host, 1, &[5, 6]), None);
    }

    /// The pre-profile scratch plan, spelled out from the pinned scalars:
    /// the Flash-Next byte accounting must not move.
    fn flash_next_plan_bytes(config: &crate::config::Qwen4Config, r: usize) -> u64 {
        let (h, hc) = (config.hidden_size, config.hc_count);
        let w = h * hc;
        let q = config.num_attention_heads * config.head_dim;
        let kv = config.num_key_value_heads * config.head_dim;
        let gdn_value = config.linear_num_value_heads * config.linear_value_head_dim;
        let gdn_qkv = 2 * config.linear_num_key_heads * config.linear_key_head_dim + gdn_value;
        let index = (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim;
        let max_projection = w.max(2 * q).max(gdn_qkv).max(h);
        let max_rotation = w.max(h).max(config.moe_intermediate_size);
        let (k, e, mi, shared) = (
            config.num_experts_per_tok,
            config.num_experts,
            config.moe_intermediate_size,
            config.shared_expert_intermediate_size,
        );
        let slots = r * k;
        let grouped = crate::program::grouped_m_total_bound(slots, e).unwrap();
        let selected = r * config.qsa_selected_capacity() * 4;
        let f32_elements = r * h * 2
            + r * w * 2
            + r * config.hc_lowrank
            + r * w
            + r * h
            + r * hc
            + r * max_rotation
            + r * max_projection * 2
            + r * config.linear_num_value_heads * 4
            + r * gdn_value * 3
            + r * index
            + r * 2 * q
            + r * kv * 2
            + r * q
            + r * h
            + r * h
            + r * w
            + r * h
            + r * w * 4
            + r * e
            + r * h
            + r * 2 * mi
            + r * shared * 3
            + r * h * 2
            + slots * mi * 3
            + r * k
            + slots * h
            + r * shared
            + grouped * 2 * mi
            + grouped * h
            + config.vocab_size;
        let raw_bytes = r * 4
            + selected
            + slots * 4
            + e * 4
            + (e + 1) * 4
            + grouped * 4
            + grouped / 16 * 4
            + slots * 4;
        let bf16_bytes = 2 * gdn_value + 2 * r * 16 * 160;
        (f32_elements * 4 + raw_bytes + bf16_bytes) as u64
    }

    #[test]
    fn flash_next_scratch_bytes_are_unchanged() {
        let config = crate::config::compact_test_config();
        for rows in [1, 16, 1536] {
            assert_eq!(
                super::Qwen4GpuForwardScratch::device_bytes(&config, rows, super::Qwen4ForwardScratchMode::Full)
                    .unwrap(),
                flash_next_plan_bytes(&config, rows),
                "rows {rows}"
            );
        }
    }

    fn tiny_dense() -> crate::config::Qwen4Config {
        crate::config::synthetic_test_config(
            crate::config::Qwen4FfnConfig::Dense {
                intermediate_size: 8,
            },
            16,
            4,
            true,
            true,
        )
    }

    #[test]
    fn dense_scratch_is_profile_derived_and_never_empty() {
        let config = tiny_dense();
        let rows = 4;
        let (plan, ple_bytes) =
            super::Qwen4GpuForwardScratch::plan(&config, rows, super::Qwen4ForwardScratchMode::Full)
                .unwrap();
        assert!(plan.iter().all(|slot| slot.elements >= 1));
        let ple = config.ple.as_ref().unwrap();
        let embed = ple.head_count * ple.row_width;
        assert_eq!(ple_bytes, rows * embed * 2);
        use super::ScratchField as F;
        let slot = |field: F| (plan[field as usize].elements, plan[field as usize].dtype);
        let gdn_value = config.linear_num_value_heads * config.linear_value_head_dim;
        assert_eq!(
            slot(F::GdnBf16),
            (
                gdn_value.max(config.hidden_size).max(config.hc_lowrank),
                rdna_compute::DType::BF16
            )
        );
        assert_eq!(slot(F::PleStaged), (rows * embed, rdna_compute::DType::BF16));
        assert_eq!(slot(F::PleKey).0, rows * config.hidden_size * config.hc_count);
        assert_eq!(slot(F::MoeGateBatch).0, rows * 8);
        assert_eq!(slot(F::MoeUpBatch).0, rows * 8);
    }

    #[test]
    fn forward_refuses_a_rope_base_the_qsa_kernels_do_not_implement() {
        let mut config = tiny_dense();
        let error = super::Qwen4GpuForward::admit_profile(&config).unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains(&config.rope_theta.to_string())
                && message.contains(&super::QWEN4_QSA_KERNEL_ROPE_THETA.to_string()),
            "{message}"
        );
        config.rope_theta = super::QWEN4_QSA_KERNEL_ROPE_THETA;
        super::Qwen4GpuForward::admit_profile(&config).expect("kernel base admitted");
        // PLE is optional: a profile without it is admitted, and its scratch
        // plan holds no PLE staging bytes (every buffer still has an element).
        config.ple = None;
        super::Qwen4GpuForward::admit_profile(&config).expect("no PLE is admitted");
        let (plan, ple_bytes) =
            super::Qwen4GpuForwardScratch::plan(&config, 4, super::Qwen4ForwardScratchMode::Full)
                .unwrap();
        assert_eq!(ple_bytes, 0);
        assert!(plan.iter().all(|slot| slot.elements >= 1));
    }

    #[test]
    fn dense_ffn_program_dims_and_layout() {
        let config = tiny_dense();
        let dims = super::program_dims(&config);
        assert!(matches!(
            dims.ffn,
            crate::program::Qwen4FfnDims::Dense { intermediate: 8 }
        ));
        let layout = crate::program::Qwen4ScratchLayout::for_rows(dims, 4).unwrap();
        assert_eq!(layout.dense_rows, 32);
        assert_eq!(
            (
                layout.slots,
                layout.grouped_rows,
                layout.shared_rows,
                layout.routed_rows
            ),
            (0, 0, 0, 0)
        );
        let flash = super::program_dims(&crate::config::compact_test_config());
        let layout = crate::program::Qwen4ScratchLayout::for_rows(flash, 4).unwrap();
        assert_eq!(layout.dense_rows, 0);
        assert_eq!(layout.routed_rows, 4 * 10 * 640);
    }
}

/// The session cache's next pending snapshot boundary and the one after it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SessionBoundaries {
    pub next: Option<usize>,
    pub after: Option<usize>,
}

/// A shared deterministic singleton/MTP schedule. A planned pair covers
/// `min(remaining, 2 * chunk)` rows. One interior session boundary rides the
/// in-bank Core sub-launch; a second interior boundary ends the group at the
/// first one, as an ordinary sequential bank.
pub(crate) fn qwen4_prefill_segment_rows(
    position: usize, prompt_end: usize, chunk: usize,
    boundaries: SessionBoundaries, planned: bool,
) -> usize {
    let remaining = prompt_end.saturating_sub(position);
    let pair = planned && chunk == 8192 && remaining > chunk;
    let rows = if pair { remaining.min(2 * chunk) } else { remaining.min(chunk) };
    let Some(boundary) = boundaries.next else { return rows; };
    if boundary <= position { return remaining.min(chunk); }
    let distance = boundary - position;
    if pair {
        if distance < rows && boundaries.after.is_some_and(|next| next < position + rows) {
            return distance.min(chunk);
        }
        return rows;
    }
    rows.min(distance)
}

/// Where in a pair group the session snapshot row falls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CutSite {
    /// No interior snapshot row (the group end is published by the driver).
    None,
    /// Exactly the A/B boundary: bank A runs whole, state staged after it.
    Split,
    /// Inside `bank` at its local `row`: that bank's Core runs as two launches.
    Bank { bank: usize, row: usize },
}

/// Bank and sub-launch geometry of one pair group: two banks and the group-local
/// row of an interior session snapshot. Both the executor and the numerical twin
/// derive their chunking from this, so they cut at identical rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PairShape {
    pub rows: [usize; 2],
    pub cut: Option<usize>,
}

impl PairShape {
    /// `cut` is the session boundary `boundary` made group-local at `position`;
    /// only a boundary strictly inside the group is a cut.
    pub(crate) fn new(rows: [usize; 2], position: usize, boundary: Option<usize>) -> Self {
        let total = rows[0] + rows[1];
        let cut = boundary.filter(|&b| b > position && b < position + total).map(|b| b - position);
        Self { rows, cut }
    }

    pub(crate) fn total(&self) -> usize { self.rows[0] + self.rows[1] }

    pub(crate) fn site(&self) -> CutSite {
        match self.cut {
            None => CutSite::None,
            Some(cut) if cut == self.rows[0] => CutSite::Split,
            Some(cut) if cut < self.rows[0] => CutSite::Bank { bank: 0, row: cut },
            Some(cut) => CutSite::Bank { bank: 1, row: cut - self.rows[0] },
        }
    }

    /// Group-local row ranges of the sequential chunks the numerical twin runs:
    /// the banks, further split at the cut (a sub-launch is a chunk boundary).
    pub(crate) fn pieces(&self) -> Vec<std::ops::Range<usize>> {
        let mut points = vec![0, self.rows[0], self.total()];
        points.extend(self.cut);
        points.sort_unstable();
        points.dedup();
        points.windows(2).map(|w| w[0]..w[1]).collect()
    }
}

#[cfg(test)]
mod ep_rank_generic_tests {
    use super::*;

    /// Default on, exact `0` is the kill switch, anything else leaves it on.
    #[test]
    fn ep_levers_are_default_on_with_an_exact_zero_kill_switch() {
        assert!(ep_lever_on(None));
        assert!(ep_lever_on(Some("1")));
        assert!(ep_lever_on(Some("")));
        assert!(ep_lever_on(Some("true")));
        assert!(!ep_lever_on(Some("0")));
    }

    fn owners(table: &[u8]) -> Vec<i32> {
        table
            .chunks_exact(4)
            .map(|word| i32::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn owner_routed_prefill_admits_exactly_ep2_and_ep4() {
        for ranks in [2usize, 4] {
            assert!(tile_ranks_supported(ranks), "{ranks}");
        }
        for ranks in [0usize, 1, 3, 5, 6, 8] {
            assert!(!tile_ranks_supported(ranks), "{ranks}");
        }
    }

    #[test]
    fn owner_table_is_expert_modulo_ranks_and_balanced() {
        for ranks in [2usize, 4] {
            let table = owners(&tile_owner_table_bytes(512, ranks));
            assert_eq!(table.len(), 512);
            assert!(table.iter().enumerate().all(|(e, &o)| o == (e % ranks) as i32));
            for rank in 0..ranks {
                assert_eq!(table.iter().filter(|&&o| o == rank as i32).count(), 512 / ranks);
            }
        }
    }

    #[test]
    fn stage_donor_bytes_split_by_every_rank_and_total_the_full_layer() {
        let full = 512 * (QWEN4_EXPERT_STAGE_GATE_UP_BYTES + QWEN4_EXPERT_STAGE_DOWN_BYTES);
        for ranks in [1usize, 2, 4] {
            let [gate_up, down] = stage_donor_bytes(512, ranks);
            assert_eq!(gate_up, 512 / ranks * QWEN4_EXPERT_STAGE_GATE_UP_BYTES);
            assert_eq!(down, 512 / ranks * QWEN4_EXPERT_STAGE_DOWN_BYTES);
            assert_eq!((gate_up + down) * ranks, full);
        }
    }

    #[test]
    fn ep4_donor_redirect_covers_each_expert_once_and_keeps_peer_pointers() {
        let ranks = 4usize;
        let stride = QWEN4_EXPERT_STAGE_GATE_UP_BYTES;
        let base = 0x10_0000usize;
        let entries: Vec<usize> = (0..512).map(|e| 0x9000_0000 + e).collect();
        let mut claimed = vec![0usize; 512];
        for rank in 0..ranks {
            let placement = Qwen4EpPlacement { rank, ranks };
            let bytes = redirect_owned_entries(placement, base, stride, &entries);
            for (expert, word) in bytes.chunks_exact(std::mem::size_of::<usize>()).enumerate() {
                let pointer = usize::from_ne_bytes(word.try_into().unwrap());
                if expert % ranks == rank {
                    assert_eq!(pointer, base + (expert / ranks) * stride, "rank {rank} expert {expert}");
                    claimed[expert] += 1;
                } else {
                    assert_eq!(pointer, entries[expert], "rank {rank} expert {expert}");
                }
            }
        }
        assert!(claimed.iter().all(|&count| count == 1));
    }
}
