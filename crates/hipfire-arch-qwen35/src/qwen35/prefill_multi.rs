// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Multi-request batched prefill chunk: several independent requests' short
//! row blocks (e.g. MTP verify rows `[seed, drafts..]`) through ONE trunk
//! forward, per request byte-identical to the singleton
//! [`super::forward_prefill_batch`] over that request's rows alone.
//!
//! It runs the singleton chunk body's own stage functions
//! (`forward_batch_chunk_impl`): the row-local stages — embedding, the
//! norm/rotate + projection GEMM stages, the FFN, the final norm — run once
//! over all requests' rows; the stages that read request state — position
//! upload, DeltaNet pre-GDN + recurrence, FullAttention prep + KV write +
//! attend — run per request on that request's rows (a row view of the
//! shared scratch), KV and DeltaNet state. Exactness rests on the shared
//! stages being row-local and selecting the same kernels for the combined
//! row count as for one request's rows: true while every count is in
//! `2..64` (the gfx12 projection GEMMs are row-count independent there —
//! CbSpec probe — and every n-dependent route switch is at >= 64 rows).
//!
//! Aggregate 64..=128 rows (`HIPFIRE_CB_VERIFY_CHUNK128`, default on for exact gfx1201 with
//! a dense MQ4G256V2 target only, see [`multi_chunk_wide_admitted`]): the
//! default `>= 64`-row projection route (IU4/FP8 quantized activations) is not
//! the singleton's arithmetic, so the shared stages run with
//! `DenseBatchMath::SingletonWmma` — the F32 producers plus the explicit
//! exact verify GEMMs, a F16-input / f32-WMMA chain byte-identical to the
//! singleton at every row count. Every request stays below 64 rows on its
//! own (its state-sensitive stages and its singleton equivalent are
//! request-sized); only the combined row count may reach 128.
//!
//! DFlash chain-verify requests (`MultiChunkRequest::fusion == ChainVerify`,
//! with their own hidden ring and tape) keep every route that depends on the
//! request's own row count request-sized: GDN pre/tape fusion and FA prep use
//! the request's `fusion`. The ChainVerify S4 residual arm (`1..=16` rows) is
//! byte-identical to the `Off` route on exact gfx1201, so output projection +
//! FFN run once at the combined count there; on other arches a layer where
//! any request's singleton would take the S4 arm runs those stages per request
//! view instead.
//!
//! Lane-batched FullAttention (exact gfx1201, every request a ChainVerify lane
//! on native fp8 KV, see [`lane_fa_chunk_admitted`]): the per-request unfused
//! prep + fp8 KV write + VerifyAttn tile/reduce of each FA layer collapse to
//! the row-local prep once over all rows plus one launch per stage for all
//! lanes (`rdna_compute::lane_fa`), byte-identical per request;
//! `HIPFIRE_CB_LANE_FA_BATCH=0` keeps the per-request loop.
//!
//! Save-free DeltaNet verify (`MultiChunkRequest::dn_state_src`, exact gfx1201,
//! `HIPFIRE_CB_DN_SAVE_FREE`, DFlash CB verify only): the lane-batched GDN
//! route reads each lane's pre-window state from a source snapshot and writes
//! only the request's `dn_state` (a conv-ring copy A -> B, the in-place conv
//! kernel on B and
//! `gated_delta_net_q8_fast_lanes_varrows_from`), so the caller needs no SAVE copy of the
//! live state; byte-identical to the in-place route after a SAVE.

use super::*;

/// One request's rows of a multi-request chunk.
///
/// `fusion` is the request's own [`DflashFusionCtx`]: `Off` for MTP verify and
/// ordinary rows, `ChainVerify` for a DFlash chain verify block. Every
/// route-sensitive stage (GDN pre/tape fusion, FA prep; off gfx1201 also the
/// S4 residual arm) runs with this request's `fusion` and its own row count,
/// exactly as the singleton verify forward over these rows alone. `hidden_rb`, when set, is
/// this request's DFlash extraction ring: the post-layer residual rows of
/// each extract layer are staged and committed (head advanced by the
/// request's row count) exactly as the singleton verify does.
pub struct MultiChunkRequest<'a> {
    pub tokens: &'a [u32],
    pub start_pos: usize,
    pub kv_cache: &'a mut llama::KvCache,
    pub dn_state: &'a mut DeltaNetState,
    /// Rollback tape for this request's rows (the singleton verify's
    /// `gdn_tape`, offset 0); `None` = no capture.
    pub gdn_tape: Option<&'a crate::speculative::GdnTape>,
    /// `Off` (MTP / plain rows) or `ChainVerify` (DFlash chain verify).
    pub fusion: DflashFusionCtx,
    /// This request's hidden-state ring (DFlash extract layers); `None` = no
    /// capture. Its staging must hold the request's rows (`max_batch >= n`).
    pub hidden_rb: Option<&'a mut HiddenStateRingBuffer>,
    /// Save-free DeltaNet verify (`HIPFIRE_CB_DN_SAVE_FREE`): `Some(snap)` =
    /// `snap` holds this request's pre-window DeltaNet state (S, scales, conv
    /// ring, EF) as an IMMUTABLE source, and `dn_state` is the DESTINATION:
    /// the caller exchanged the owners of the two banks before the call
    /// ([`crate::speculative::DeltaNetSnapshot::exchange_buffers`]), so
    /// `dn_state` holds no valid state yet. The lane-batched GDN route then
    /// reads the source at each state's first use and writes only
    /// `dn_state`. `None` = today's in-place route (`dn_state` is both).
    /// All requests of one chunk must agree; a sourced chunk that cannot take
    /// the lane-batched GDN route is refused, never run in place.
    pub dn_state_src: Option<&'a crate::speculative::DeltaNetSnapshot>,
}

/// The largest aggregate row count the multi-request body supports (the
/// exact wide verify GEMMs cover 64..=128 rows). The route's EFFECTIVE cap is
/// [`multi_chunk_row_cap`]: 128 on exact gfx1201 unless `HIPFIRE_CB_VERIFY_CHUNK128=0`,
/// otherwise [`MULTI_CHUNK_PRODUCT_MAX_ROWS`].
pub const MULTI_CHUNK_MAX_ROWS: usize = 128;

/// The largest aggregate row count on the product arithmetic: below every
/// n-dependent kernel/route switch of the shared stages (`>= 64` rows).
pub const MULTI_CHUNK_PRODUCT_MAX_ROWS: usize = 63;

/// The largest single request: its singleton (`forward_prefill_batch`) over
/// 64+ rows would select the quantized `>= 64`-row projection route, whose
/// bytes the wide exact route does not reproduce, so a lone 64/128-row request
/// is never admitted as a window.
pub const MULTI_CHUNK_MAX_LANE_ROWS: usize = 63;

/// The effective aggregate row cap of this device/process:
/// `Gpu::mq4_verify_chunk_rows` (128 on exact gfx1201 with WMMA batch tiles unless
/// `HIPFIRE_CB_VERIFY_CHUNK128=0`, else 63) bounded to the
/// supported `MULTI_CHUNK_PRODUCT_MAX_ROWS..=MULTI_CHUNK_MAX_ROWS`.
pub fn multi_chunk_row_cap(gpu: &Gpu) -> usize {
    gpu.mq4_verify_chunk_rows()
        .clamp(MULTI_CHUNK_PRODUCT_MAX_ROWS, MULTI_CHUNK_MAX_ROWS)
}

/// May an aggregate chunk of 64+ rows run on this target? The wide route is
/// exact only when (a) the device's effective cap exceeds the product cap,
/// (b) every singleton arm it must match runs the one-tile F16-WMMA chain the
/// exact wide kernels reproduce ([`Gpu::mq4_verify_singleton_chain`]: refused
/// under `HIPFIRE_FP16=0`, `HIPFIRE_LM_HEAD_WMMA=0`,
/// `HIPFIRE_HFQ4G256_LDSSTAGE=1`), and (c) the trunk and head are dense
/// MQ4G256V2: every projection of every layer and the lm_head (no MoE,
/// Lloyd, MQ3/MQ6, Q8 or other dtypes), with `w_down.k == hidden_dim` (the
/// `SingletonWmma` FFN-down shape). Decided before any request state is
/// written; a refusal keeps the cap at [`MULTI_CHUNK_PRODUCT_MAX_ROWS`].
pub fn multi_chunk_wide_admitted(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
) -> bool {
    multi_chunk_wide_route(&gpu.arch, &gpu.flags) && multi_chunk_dense_mq4v2_target(weights, config)
}

/// The device/flag half of [`multi_chunk_wide_admitted`] (no device needed).
fn multi_chunk_wide_route(arch: &str, flags: &rdna_compute::FeatureFlags) -> bool {
    Gpu::mq4_verify_chunk_rows_for(arch, flags)
        .clamp(MULTI_CHUNK_PRODUCT_MAX_ROWS, MULTI_CHUNK_MAX_ROWS)
        > MULTI_CHUNK_PRODUCT_MAX_ROWS
        && Gpu::mq4_verify_singleton_chain_for(arch, flags)
}

/// IU4 DFlash verify for this target (the load's decision, default on at
/// `dflash_block` 8; `HIPFIRE_CB_VERIFY_IU4`): [`Gpu::iu4_verify_route_admitted`] and a dense MQ4G256V2
/// trunk + head. The DFlash verify call sites (singleton and multi-request)
/// open `Gpu::iu4_verify_scope` with this, so both run the same A4 producers
/// and IU4 GEMMs ([`DenseBatchMath::Iu4Verify`]).
pub fn dflash_iu4_verify_admitted(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
) -> bool {
    gpu.iu4_verify_route_admitted() && multi_chunk_dense_mq4v2_target(weights, config)
}

/// The target half of [`multi_chunk_wide_admitted`] (also the dense-target
/// half of the PM FP8 verify admission, `prefill::fp8_verify_target`).
pub(crate) fn multi_chunk_dense_mq4v2_target(
    weights: &Qwen35Weights,
    config: &Qwen35Config,
) -> bool {
    use rdna_compute::DType::MQ4G256V2;
    config.num_experts == 0
        && weights.output.gpu_dtype == MQ4G256V2
        && weights.layers.iter().all(|l| match l {
            LayerWeights::DeltaNet(d) => {
                d.w_down.k == config.hidden_dim
                    && [
                        &d.wqkv, &d.wz, &d.w_alpha, &d.w_beta, &d.wo, &d.w_gate, &d.w_up, &d.w_down,
                    ]
                    .iter()
                    .all(|w| w.gpu_dtype == MQ4G256V2)
            }
            LayerWeights::FullAttn(f) => {
                f.w_down.k == config.hidden_dim
                    && [&f.wq, &f.wk, &f.wv, &f.wo, &f.w_gate, &f.w_up, &f.w_down]
                        .iter()
                        .all(|w| w.gpu_dtype == MQ4G256V2)
            }
            _ => false,
        })
}

/// The rows one trunk chunk of this target packs: the scratch capacity
/// `cap` when the wide route is admitted for this target, else at most
/// [`MULTI_CHUNK_PRODUCT_MAX_ROWS`] (a scratch sized for the wide route may
/// serve a target that cannot take it).
pub fn multi_chunk_pack_cap(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    cap: usize,
) -> usize {
    pack_cap_for(multi_chunk_wide_admitted(gpu, weights, config), cap)
}

fn pack_cap_for(wide_admitted: bool, cap: usize) -> usize {
    cap.min(if wide_admitted {
        MULTI_CHUNK_MAX_ROWS
    } else {
        MULTI_CHUNK_PRODUCT_MAX_ROWS
    })
}

/// Rows a shared verify scratch planned for `requested` rows allocates on
/// this target: its effective packing cap ([`multi_chunk_pack_cap`]), at
/// least [`MIN_BATCH`]. A target the wide route does not admit allocates at
/// most [`MULTI_CHUNK_PRODUCT_MAX_ROWS`] rows, whatever the device cap.
pub fn multi_chunk_scratch_rows(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    requested: usize,
) -> usize {
    scratch_rows_for(multi_chunk_wide_admitted(gpu, weights, config), requested)
}

fn scratch_rows_for(wide_admitted: bool, requested: usize) -> usize {
    pack_cap_for(wide_admitted, requested).max(MIN_BATCH)
}

/// Pack whole lanes, in order, into trunk chunks of at most
/// `min(max_rows, MULTI_CHUNK_MAX_ROWS)` rows: `rows[i]` is lane `i`'s row
/// count and `max_rows` the caller's effective cap (63 on the product route;
/// 64..=128 only with the wide exact route admitted). `out` receives one
/// lane-index range per chunk (cleared first; no allocation when `out`
/// already has capacity). A lane is never split, padded or reordered, and no
/// lane may exceed [`MULTI_CHUNK_MAX_LANE_ROWS`]: e.g. eight 16-row lanes pack
/// as `[0..3, 3..6, 6..8]` (48 + 48 + 32 rows) at cap 63, `[0..4, 4..8]`
/// (64 + 64) at cap 64 and `[0..8]` (one 128-row chunk) at cap 128. Errors
/// (leaving `out` empty) when a lane has fewer than [`MIN_BATCH`] rows or more
/// rows than one chunk or one request can hold.
pub fn pack_whole_lanes(
    rows: &[usize],
    max_rows: usize,
    out: &mut Vec<std::ops::Range<usize>>,
) -> Result<(), String> {
    out.clear();
    let cap = max_rows.min(MULTI_CHUNK_MAX_ROWS);
    let lane_cap = cap.min(MULTI_CHUNK_MAX_LANE_ROWS);
    let mut start = 0usize;
    let mut acc = 0usize;
    for (i, &n) in rows.iter().enumerate() {
        if n < MIN_BATCH || n > lane_cap {
            out.clear();
            return Err(format!(
                "pack_whole_lanes: lane {i} has {n} rows, outside {MIN_BATCH}..={lane_cap}"
            ));
        }
        if acc + n > cap {
            out.push(start..i);
            start = i;
            acc = 0;
        }
        acc += n;
    }
    if start < rows.len() {
        out.push(start..rows.len());
    }
    Ok(())
}

/// Most requests (segments) one twin launch carries: every request has at
/// least [`MIN_BATCH`] rows.
const MULTI_CHUNK_MAX_SEGS: usize = MULTI_CHUNK_MAX_ROWS / MIN_BATCH;

use hipfire_dispatch::ops::verify_twins::{self as seg_twins, AttnFp8Seg};

/// `HIPFIRE_CB_SEG_TWINS=0` keeps every request on its own singleton
/// launches (A/B control for the segment twins).
static SEG_TWINS: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_CB_SEG_TWINS").map_or(true, |v| v.trim() != "0")
});

/// `HIPFIRE_CB_TAPE_COPY_BATCH=0` (developer kill switch): every request
/// snaps its GDN rollback tape with its own three `hipMemcpyDtoD` per layer
/// (the singleton's copies) instead of one batched descriptor-copy launch per
/// layer for all requests of the chunk ([`Gpu::tape_copy_batched`]). Default
/// on; the tape bytes are identical either way.
static TAPE_COPY_BATCH: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_CB_TAPE_COPY_BATCH").map_or(true, |v| v.trim() != "0")
});

/// `HIPFIRE_CB_LANE_GDN_BATCH=0` (developer kill switch): every request runs
/// its own DeltaNet pre-GDN sequence + recurrence per layer (4 launches per
/// lane) instead of one launch per stage per layer for all lanes of the chunk
/// (see [`lane_gdn_chunk_admitted`]). Default on; every output and state byte
/// is identical either way.
static LANE_GDN_BATCH: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_CB_LANE_GDN_BATCH").map_or(true, |v| v.trim() != "0")
});

/// `HIPFIRE_CB_LANE_GDN_VARROWS=0` (developer kill switch): restores the
/// equal-row-count-only admission of the lane-batched GDN route (both
/// [`lane_gdn_chunk_admitted`] and the CB save-free admission,
/// `dflash_cb::cb_dn_save_free_admitted`): a chunk whose requests do not all
/// carry the same row count keeps the per-request recurrence and the SAVE +
/// in-place verify. Default on: each lane may carry its own `1..=16` rows (the
/// recurrence reads each lane's row count and its prefix frame offset from
/// its table row); equal-count chunks are unaffected either way. Every output
/// and state byte is identical either way. Equal-count batching / save-free
/// stay on with this switch off (`HIPFIRE_CB_LANE_GDN_BATCH` /
/// `HIPFIRE_CB_DN_SAVE_FREE` keep their own meanings).
static LANE_GDN_VARROWS: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_CB_LANE_GDN_VARROWS").map_or(true, |v| v.trim() != "0")
});

/// Is the variable-row lane-batched GDN admission enabled for this process?
pub(crate) fn lane_gdn_varrows_enabled() -> bool {
    *LANE_GDN_VARROWS
}

/// Largest per-lane row count of an UNEQUAL (variable-row) lane-batched GDN
/// chunk. Equal-count chunks keep their existing admission (up to the
/// whole-forward row bound).
const LANE_GDN_VARROWS_MAX_N: usize = 16;

/// Default of `HIPFIRE_CB_DN_SAVE_FREE` when the variable is unset. The only
/// place to flip the default after measurement.
const CB_DN_SAVE_FREE_DEFAULT: bool = true;

/// `HIPFIRE_CB_DN_SAVE_FREE` (developer var; `0` is the kill switch):
/// whether the DFlash CB verify may swap the owners of each lane's live
/// DeltaNet state and its pre-verify snapshot instead of copying the live
/// state into the snapshot (see `dflash_cb::cb_dn_save_free_admitted`). Off
/// restores today's SAVE + in-place verify, byte for byte.
static CB_DN_SAVE_FREE: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(
        || match hipfire_config::developer_var("HIPFIRE_CB_DN_SAVE_FREE") {
            Ok(v) => v.trim() != "0",
            Err(_) => CB_DN_SAVE_FREE_DEFAULT,
        },
    );

/// Is the save-free CB verify enabled for this process? (The scratch arms its
/// from-table at construction only when it is.)
pub(crate) fn cb_dn_save_free_enabled() -> bool {
    *CB_DN_SAVE_FREE
}

/// `HIPFIRE_CB_UPLOAD_COALESCE` (developer var; `0` is the kill switch):
/// whether the continuous-batching step may skip or merge the tiny
/// synchronous H2D uploads it issues every step. Two sites read it: the VMM
/// executor's per-layer slot-descriptor tables (re-uploaded only when their
/// bytes differ from what the device already holds) and the multi-lane CB
/// verify's per-lane position uploads (one upload of every lane's positions).
/// The device bytes before every reading kernel are identical either way; off
/// restores one upload per layer / per lane. The draft-lane key positions land
/// in separate per-lane scratch buffers and are never coalesced, so this
/// switch does not touch `draft_forward_lanes`.
static CB_UPLOAD_COALESCE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_CB_UPLOAD_COALESCE").map_or(true, |v| v.trim() != "0")
});

/// Is the CB upload coalescing enabled for this process?
pub(crate) fn cb_upload_coalesce_enabled() -> bool {
    *CB_UPLOAD_COALESCE
}

/// `HIPFIRE_CB_LANE_FA_BATCH=0` (developer kill switch): every request runs
/// its own FullAttention prep + fp8 KV write + VerifyAttn tile + reduce per
/// layer (4+ launches per lane) instead of one launch per stage per layer
/// for all lanes of the chunk (see [`lane_fa_chunk_admitted`]). Default on;
/// every output and KV byte is identical either way.
static LANE_FA_BATCH: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_CB_LANE_FA_BATCH").map_or(true, |v| v.trim() != "0")
});

/// Largest lane-batched FA partials buffer (f32 elements, 1 GiB): a chunk whose
/// lanes would need more (very long contexts x many rows) keeps the per-lane
/// route, which reuses the model scratch's one-row-group partials.
const LANE_FA_PARTIALS_MAX_FLOATS: usize = (1usize << 30) / 4;

/// Growth granule of the lane-batched FA partials buffer (64 MiB of f32), so a
/// context creeping one tile at a time does not reallocate every step.
const LANE_FA_PARTIALS_GROW_FLOATS: usize = (64usize << 20) / 4;

/// Bytes of one layer's device GDN lane table (`CONV_LANES_MAX`
/// [`GdnLaneTable`](rdna_compute::lane_gdn::GdnLaneTable) rows, 80 B each): the
/// layer stride `gated_delta_net_q8_fast_lanes_varrows` is launched with.
const GDN_LANE_TABLE_LAYER_BYTES: usize = rdna_compute::lane_gdn::CONV_LANES_MAX
    * std::mem::size_of::<rdna_compute::lane_gdn::GdnLaneTable>();

/// Bytes of one layer's device snapshot-source GDN lane table
/// (`CONV_LANES_MAX` [`GdnLaneTableFrom`](rdna_compute::lane_gdn::GdnLaneTableFrom)
/// rows, 104 B each): the layer stride `gated_delta_net_q8_fast_lanes_varrows_from`
/// is launched with.
const GDN_LANE_FROM_TABLE_LAYER_BYTES: usize = rdna_compute::lane_gdn::CONV_LANES_MAX
    * std::mem::size_of::<rdna_compute::lane_gdn::GdnLaneTableFrom>();

/// The save-free route's lane tables: device `GdnLaneTableFrom` rows
/// `[delta_layer][lane]` plus the host copy of what each holds. Separate from
/// the base table so requests without a source keep their own cache. The
/// owner exchange flips every lane's live/snapshot banks each admitted step,
/// so the rows alternate between two configurations: the table is two-slot
/// (one device table + its host rows per slot, least-recently-used
/// replacement, [`TwoSlot`](crate::speculative::TwoSlot)), and both
/// configurations hit without an upload after the first two steps.
struct GdnLaneFromTable {
    devices: [GpuTensor; 2],
    host: std::sync::Mutex<
        crate::speculative::TwoSlot<Vec<rdna_compute::lane_gdn::GdnLaneTableFrom>>,
    >,
}

/// A HIP error that leaves the device context unusable (illegal address,
/// assert, illegal instruction, misaligned/invalid address, invalid PC,
/// launch failure): never swallowed as "route unavailable".
fn hip_context_fault(e: &HipError) -> bool {
    matches!(e.code, 700 | 710 | 715..=719)
}

/// Shared scratch of a multi-request chunk: the row scratch every request's
/// rows live in, the segment-twin tables (one per FullAttention layer) and the
/// lane-batched GDN recurrence table (`CONV_LANES_MAX` rows per DeltaNet layer).
pub struct MultiChunkScratch {
    pub pbs: PrefillBatchScratch,
    attn_segs: GpuTensor,
    /// Device `GdnLaneTable` rows, `[delta_layer][lane]`, read by
    /// `gated_delta_net_q8_fast_lanes_varrows` (grid.z = lane).
    gdn_lane_table: GpuTensor,
    /// What `gdn_lane_table` currently holds (same layout); the forward takes
    /// `&MultiChunkScratch`, hence the lock. The table is re-uploaded only
    /// when a chunk's rows differ (lane membership / order change).
    gdn_lane_table_host: std::sync::Mutex<Vec<rdna_compute::lane_gdn::GdnLaneTable>>,
    /// Snapshot-source twin of `gdn_lane_table` for the save-free DeltaNet
    /// verify (`GdnLaneTableFrom` rows read by `gated_delta_net_q8_fast_lanes_varrows_from`).
    /// Armed at construction only when `HIPFIRE_CB_DN_SAVE_FREE` is on and the
    /// device supports the lane-batched GDN route (the `_from` recurrence
    /// JIT'd, the table allocated); `None` otherwise, which
    /// keeps `Self::gdn_from_ready` false so no chunk is admitted to it.
    gdn_lane_from: Option<GdnLaneFromTable>,
    /// Disjoint per-lane VerifyAttn partials of the lane-batched FullAttention
    /// route (`Gpu::lane_fa_partials_len` f32 elements), grown on demand
    /// ([`Self::lane_fa_partials_acquire`]); `None` until a chunk takes the
    /// route. The model scratch's `flash_partials` holds only one row group
    /// of one singleton launch, so lanes cannot share it.
    lane_fa_partials: std::sync::Mutex<Option<GpuTensor>>,
}

impl MultiChunkScratch {
    /// `max_rows` is clamped to `MIN_BATCH..=multi_chunk_row_cap(gpu)` (the
    /// effective cap: 63 unless the wide verify route is on); the scratch
    /// stores that capacity ([`Self::max_rows`]).
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, max_rows: usize) -> HipResult<Self> {
        let rows = max_rows.clamp(MIN_BATCH, multi_chunk_row_cap(gpu));
        let pbs = PrefillBatchScratch::new(gpu, config, rows)?;
        let n_fa = config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count();
        let n_delta = config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count();
        let bytes = n_fa.max(1) * MULTI_CHUNK_MAX_SEGS * std::mem::size_of::<AttnFp8Seg>();
        let attn_segs = match gpu.zeros(&[bytes.div_ceil(4)], rdna_compute::DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = pbs.free_gpu(gpu);
                return Err(e);
            }
        };
        let gdn_bytes = n_delta.max(1) * GDN_LANE_TABLE_LAYER_BYTES;
        let gdn_lane_table = match gpu.zeros(&[gdn_bytes.div_ceil(4)], rdna_compute::DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(attn_segs);
                let _ = pbs.free_gpu(gpu);
                return Err(e);
            }
        };
        let gdn_lane_from = match Self::arm_gdn_lane_from(gpu, config, n_delta) {
            Ok(f) => f,
            Err(e) => {
                let _ = gpu.free_tensor(gdn_lane_table);
                let _ = gpu.free_tensor(attn_segs);
                let _ = pbs.free_gpu(gpu);
                return Err(e);
            }
        };
        Ok(Self {
            pbs,
            attn_segs,
            gdn_lane_table,
            gdn_lane_table_host: std::sync::Mutex::new(Vec::new()),
            gdn_lane_from,
            lane_fa_partials: std::sync::Mutex::new(None),
        })
    }

    /// Arm the save-free route's from-table (see the `gdn_lane_from` field):
    /// JIT the `_from` recurrence and allocate both slot tables, once, here,
    /// never in a decode step. `Ok(None)` when the route
    /// is off or unsupported, or when the JIT / allocation is merely
    /// unavailable (out of memory, compile failure): readiness stays false
    /// and every chunk keeps today's SAVE route. A device-context fault is
    /// propagated (the caller frees what it built).
    fn arm_gdn_lane_from(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        n_delta: usize,
    ) -> HipResult<Option<GdnLaneFromTable>> {
        if !cb_dn_save_free_enabled() || n_delta == 0 || !lane_gdn_batch_ready(gpu, config, 1) {
            return Ok(None);
        }
        let ready = gpu.ensure_lane_gdn_varrows_from();
        if let Err(e) = ready {
            return if hip_context_fault(&e) {
                Err(e)
            } else {
                Ok(None)
            };
        }
        let bytes = n_delta * GDN_LANE_FROM_TABLE_LAYER_BYTES;
        let fail = |e: HipError| {
            if hip_context_fault(&e) {
                Err(e)
            } else {
                Ok(None)
            }
        };
        let first = match gpu.zeros(&[bytes.div_ceil(4)], rdna_compute::DType::F32) {
            Ok(t) => t,
            Err(e) => return fail(e),
        };
        let second = match gpu.zeros(&[bytes.div_ceil(4)], rdna_compute::DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(first);
                return fail(e);
            }
        };
        Ok(Some(GdnLaneFromTable {
            devices: [first, second],
            host: std::sync::Mutex::new(crate::speculative::TwoSlot::new()),
        }))
    }

    /// Is the snapshot-source (save-free) lane table armed? Part of the CB
    /// admission: a sourced chunk without it is refused by the forward.
    pub(crate) fn gdn_from_ready(&self) -> bool {
        self.gdn_lane_from.is_some()
    }

    pub fn max_rows(&self) -> usize {
        self.pbs.max_batch
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> HipResult<()> {
        self.pbs.free_gpu(gpu)?;
        gpu.free_tensor(self.attn_segs)?;
        gpu.free_tensor(self.gdn_lane_table)?;
        if let Some(from) = self.gdn_lane_from {
            let [d0, d1] = from.devices;
            gpu.free_tensor(d0)?;
            gpu.free_tensor(d1)?;
        }
        match self.lane_fa_partials.into_inner() {
            Ok(Some(t)) => gpu.free_tensor(t),
            _ => Ok(()),
        }
    }
}

/// Process/device half of the lane-batched GDN admission: both kill switches
/// (`HIPFIRE_CB_LANE_GDN_BATCH`, and `HIPFIRE_CB_TAPE_COPY_BATCH` whose batched
/// descriptors the route queues), `1..=CONV_LANES_MAX` lanes and
/// [`Gpu::lane_gdn_batch_admitted`] for this config's head geometry (exact
/// gfx1201, fast GDN kernel, 128-wide heads, GQA layout, no graph capture /
/// replay recording / G0 observation). Shared by [`lane_gdn_chunk_admitted`]
/// and the CB save-free admission.
pub(crate) fn lane_gdn_batch_ready(gpu: &Gpu, config: &Qwen35Config, n_lanes: usize) -> bool {
    *LANE_GDN_BATCH
        && *TAPE_COPY_BATCH
        && (1..=rdna_compute::lane_gdn::CONV_LANES_MAX).contains(&n_lanes)
        && gpu.lane_gdn_batch_admitted(
            config.linear_num_value_heads,
            config.linear_num_key_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        )
}

/// May this chunk's DeltaNet layers run the lane-batched GDN route (one
/// launch per stage per layer for all requests instead of 4 per request)?
/// Process/device half: [`lane_gdn_batch_ready`]. Chunk half: every request a
/// ChainVerify verify carrying a rollback tape that holds its OWN rows at
/// offset 0 (the per-request route is then the unfused sequence the batched
/// stages reproduce: neither the gfx1100 capture fusion nor the gfx1201
/// Slice-P fusion, which needs `gdn_tape.is_none()`, applies); sequential
/// batch semantics; and the shape: all requests with the SAME row count (the
/// existing equal-count admission, unbounded here up to the whole-forward row
/// bound), or, unless `HIPFIRE_CB_LANE_GDN_VARROWS=0`, every request with its
/// own `1..=LANE_GDN_VARROWS_MAX_N` rows. The recurrence's stochastic-rounding
/// frame of lane `i` is `base + prefix_i` (`prefix_i` = rows of lanes `< i`,
/// carried in lane `i`'s table row), the per-request order for any row counts.
fn lane_gdn_chunk_admitted(
    gpu: &Gpu,
    config: &Qwen35Config,
    reqs: &[MultiChunkRequest<'_>],
    sem: BatchSemantics<'_>,
) -> bool {
    let n0 = match reqs.first() {
        Some(r) => r.tokens.len(),
        None => return false,
    };
    let equal = reqs.iter().all(|r| r.tokens.len() == n0);
    matches!(sem, BatchSemantics::Sequential)
        && lane_gdn_batch_ready(gpu, config, reqs.len())
        && (equal
            || (lane_gdn_varrows_enabled()
                && reqs
                    .iter()
                    .all(|r| (1..=LANE_GDN_VARROWS_MAX_N).contains(&r.tokens.len()))))
        && reqs.iter().all(|r| {
            r.fusion.is_chain_verify() && r.gdn_tape.is_some_and(|t| r.tokens.len() <= t.max_n)
        })
}

impl MultiChunkScratch {
    /// The lane-batched FA partials buffer holding at least `floats` f32
    /// elements, locked for the caller's use (the guard keeps a concurrent
    /// chunk from reallocating it). Grows by freeing the old buffer (an
    /// implicit device sync on the stream's earlier users) and allocating the
    /// requirement rounded up to [`LANE_FA_PARTIALS_GROW_FLOATS`]. `None` on a
    /// poisoned lock or when the (re)allocation fails: the chunk then keeps
    /// the per-lane route.
    fn lane_fa_partials_acquire(
        &self,
        gpu: &mut Gpu,
        floats: usize,
    ) -> Option<std::sync::MutexGuard<'_, Option<GpuTensor>>> {
        let mut guard = self.lane_fa_partials.lock().ok()?;
        if guard.as_ref().map_or(true, |t| t.numel() < floats) {
            if let Some(old) = guard.take() {
                gpu.free_tensor(old).ok()?;
            }
            let rounded = floats.next_multiple_of(LANE_FA_PARTIALS_GROW_FLOATS);
            *guard = Some(
                gpu.alloc_tensor(&[rounded], rdna_compute::DType::F32)
                    .ok()?,
            );
        }
        Some(guard)
    }
}

/// Does this lane's singleton route reach `kv_cache_write_fp8_e4m3_batched`
/// + the VerifyAttn tile (`AttnFp8E4m3KvBatchedMasked` with
/// `verify_tile_attend`)? [`attn_fp8_twin_eligible`] without its context /
/// fusion limits: the same tier-plan derivation `execute_fa_attend_step` does.
fn lane_fa_plan_ok(
    gpu: &Gpu,
    s: &Qwen35Scratch,
    kv: &llama::KvCache,
    start_pos: usize,
    n: usize,
) -> bool {
    KvTierPlan::derive(KvTierInputs {
        pos: start_pos,
        flash_mode: s.flash_mode as usize,
        capture_mode: gpu.graphs.capture_mode,
        batch_size: n,
        is_tree: false,
        ..kv.tier_inputs()
    })
    .is_ok_and(|p| {
        p.attend_key == hipfire_dispatch::types::KernelKey::AttnFp8E4m3KvBatchedMasked
            && p.write_key == hipfire_dispatch::types::KernelKey::KvWriteFp8E4m3Batched
    })
}

/// May this chunk's FullAttention layers run the lane-batched route (the
/// row-local prep launches once over all rows, then one fp8 KV write per
/// side, one VerifyAttn tile and one reduce for all requests, instead of the
/// per-request prep + write + attend)? The lane route reproduces exactly the
/// route each request's singleton takes, so every one of these must hold:
///
/// * process/device: `HIPFIRE_CB_LANE_FA_BATCH` not `0`, sequential batch
///   semantics, [`Gpu::lane_fa_batch_admitted`] (exact gfx1201, VerifyAttn on,
///   head_dim 256 GQA group 6 tile 128, no capture / recording / G0
///   observation, every request 1..=32 rows), at most `LANE_FA_MAX` requests,
///   no triattn tap (it would interleave per-request recording with the prep);
/// * no request on the segment-twin arm (`twin_attn`);
/// * per request: ChainVerify (prep unfused: `gfx12_fa_prep_admitted` false;
///   attend via `verify_tile_attend`), native fp8 KV with `compact_offset 0`
///   (the rope offset is then 0 for every row), `2..64` rows (below every
///   FA2 / packet / Q-resident arm, which start at 64 rows), not the Q8
///   multirow arm, the paired fp8 write + fp8 batched-masked attend tier
///   plan, and the model scratch's partials able to hold one singleton row
///   (the singleton's own VerifyAttn admission);
/// * the lanes' disjoint partials fit [`LANE_FA_PARTIALS_MAX_FLOATS`].
fn lane_fa_chunk_admitted(
    gpu: &Gpu,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    reqs: &[MultiChunkRequest<'_>],
    sem: BatchSemantics<'_>,
    twin_attn: &[bool],
) -> bool {
    if !*LANE_FA_BATCH
        || !matches!(sem, BatchSemantics::Sequential)
        || reqs.is_empty()
        || reqs.len() > rdna_compute::lane_fa::LANE_FA_MAX
        || twin_attn.iter().any(|&t| t)
        || hipfire_runtime::triattn::tap_enabled()
    {
        return false;
    }
    let rows: Vec<usize> = reqs.iter().map(|r| r.tokens.len()).collect();
    if !gpu.lane_fa_batch_admitted(config.n_heads, config.n_kv_heads, config.head_dim, &rows) {
        return false;
    }
    let tile = gpu.attn_tile_size();
    let mut shape = Vec::with_capacity(reqs.len());
    for r in reqs {
        let n = r.tokens.len();
        let kv = &*r.kv_cache;
        let max_ctx_len = r.start_pos + n;
        let multirow = q8_multirow_attn_admitted(
            gpu.arch_caps.arch(),
            kv.quant_q8,
            config.head_dim,
            n,
            max_ctx_len,
            fa_pertoken_min_ctx(gpu.arch_caps.arch()),
            false,
            false,
            false,
            false,
            false,
        );
        if !r.fusion.is_chain_verify()
            || !kv.quant_fp8
            || kv.compact_offset != 0
            || !(MIN_BATCH..64).contains(&n)
            || gfx12_fa_prep_admitted(
                gpu,
                &hybrid_dims(config),
                r.fusion,
                n,
                hipfire_runtime::triattn::tap_enabled(),
            )
            || multirow
            || !lane_fa_plan_ok(gpu, s, kv, r.start_pos, n)
            || s.flash_partials.numel()
                < config.n_heads * max_ctx_len.div_ceil(tile) * (2 + config.head_dim)
        {
            return false;
        }
        shape.push((n, max_ctx_len));
    }
    Gpu::lane_fa_partials_len(config.n_heads, &shape) <= LANE_FA_PARTIALS_MAX_FLOATS
}

/// Make the device GDN lane table hold this chunk's recurrence rows: row
/// `[delta_layer][lane]` is lane's row views of the shared scratch's
/// q/k/v/alpha(gate)/beta/attn-out buffers and its Q8 state, scale and EF
/// buffers for that DeltaNet layer (the exact tensors the per-request
/// `gated_delta_net_q8_batch_seq` takes), plus its own row count `n_tokens`
/// and its frame offset `frame_off` (`offs[i]`, the rows of the lanes before
/// it in this chunk: the lane's stochastic-rounding frame is the launch's
/// frame base plus this prefix, relative, never an absolute counter). Rows are
/// compared with the host cache (metadata included, so a row-count change
/// with unmoved pointers misses) and the table is re-uploaded (stream-ordered
/// on the active stream) only when one differs: in steady state the lanes keep
/// their buffers and shape, so a step uploads nothing.
fn stage_gdn_lane_table(
    gpu: &mut Gpu,
    scratch: &MultiChunkScratch,
    config: &Qwen35Config,
    reqs: &[MultiChunkRequest<'_>],
    views: &[std::mem::ManuallyDrop<PrefillBatchScratch>],
    offs: &[usize],
) -> HipResult<()> {
    use rdna_compute::dflash_gdn_replay::{table_bytes, GdnLayerTable};
    use rdna_compute::lane_gdn::{GdnLaneTable, CONV_LANES_MAX};
    let n_delta = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::LinearAttention)
        .count();
    let mut cache = scratch
        .gdn_lane_table_host
        .lock()
        .map_err(|_| HipError::new(0, "gdn lane table cache poisoned"))?;
    let mut dirty = false;
    if cache.len() != n_delta * CONV_LANES_MAX {
        let base = GdnLayerTable {
            q: 0,
            k: 0,
            v: 0,
            gate: 0,
            beta: 0,
            s_q8: 0,
            s_scales: 0,
            output: 0,
            ef: 0,
        };
        let zero = GdnLaneTable {
            base,
            n_tokens: 0,
            frame_off: 0,
        };
        cache.clear();
        cache.resize(n_delta * CONV_LANES_MAX, zero);
        dirty = true;
    }
    let ptr = |t: &GpuTensor| t.buf.as_ptr() as u64;
    for l in 0..n_delta {
        for (i, (r, view)) in reqs.iter().zip(views).enumerate() {
            let d = &*r.dn_state;
            let row = GdnLaneTable {
                base: GdnLayerTable {
                    q: ptr(&view.dn_q_batch),
                    k: ptr(&view.dn_k_batch),
                    v: ptr(&view.dn_v_batch),
                    gate: ptr(&view.dn_alpha_batch),
                    beta: ptr(&view.dn_beta_batch),
                    s_q8: ptr(&d.s_matrices[l]),
                    s_scales: ptr(&d.s_scales[l]),
                    output: ptr(&view.dn_attn_out_batch),
                    ef: d.ef_residual(l).map_or(0, ptr),
                },
                n_tokens: r.tokens.len() as u32,
                frame_off: offs[i] as u32,
            };
            let slot = &mut cache[l * CONV_LANES_MAX + i];
            if *slot != row {
                *slot = row;
                dirty = true;
            }
        }
    }
    if dirty {
        if let Err(e) = gpu.upload_gdn_lane_table(&scratch.gdn_lane_table, table_bytes(&cache)) {
            // The device table is in an unknown state: force a full re-stage.
            cache.clear();
            return Err(e);
        }
    }
    Ok(())
}

/// [`stage_gdn_lane_table`] for a sourced (save-free) chunk: each row is a
/// [`GdnLaneTableFrom`](rdna_compute::lane_gdn::GdnLaneTableFrom) (the sourced
/// `GdnLayerTableFrom` row plus the lane's own `n_tokens` and relative
/// `frame_off = offs[i]`, both part of the cache comparison).
/// Its base half is exactly the in-place row (scratch q/k/v/gate/beta/output
/// views) except that the state S/scales/EF are the DESTINATION `dn_state`
/// (bank B after the pre-verify exchange); the appended `s_q8_src` /
/// `s_scales_src` / `ef_src` are the lane's source snapshot (bank A,
/// [`DeltaNetSnapshot::layer_state_ptrs`](crate::speculative::DeltaNetSnapshot)).
/// Both the source and the destination pointers are compared every chunk: the
/// exchange flips the two banks of every lane each admitted step, so the rows
/// alternate between two configurations. The table is two-slot
/// ([`GdnLaneFromTable`]): a chunk whose rows match either slot's host rows
/// uses that slot's device table with no upload; a miss rewrites the least
/// recently used slot (stream-ordered and drained like the base table's).
/// Returns the slot whose device table holds this chunk's rows.
fn stage_gdn_lane_table_from(
    gpu: &mut Gpu,
    scratch: &MultiChunkScratch,
    config: &Qwen35Config,
    reqs: &[MultiChunkRequest<'_>],
    views: &[std::mem::ManuallyDrop<PrefillBatchScratch>],
    offs: &[usize],
) -> HipResult<usize> {
    use rdna_compute::dflash_gdn_replay::{table_bytes, GdnLayerTable, GdnLayerTableFrom};
    use rdna_compute::lane_gdn::{GdnLaneTableFrom, CONV_LANES_MAX};
    let Some(from) = scratch.gdn_lane_from.as_ref() else {
        return Err(HipError::new(
            0,
            "stage_gdn_lane_table_from: the snapshot-source lane table is not armed",
        ));
    };
    if reqs.iter().any(|r| r.dn_state_src.is_none()) {
        return Err(HipError::new(
            0,
            "stage_gdn_lane_table_from: request without a snapshot source",
        ));
    }
    let n_delta = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::LinearAttention)
        .count();
    let total = n_delta * CONV_LANES_MAX;
    let ptr = |t: &GpuTensor| t.buf.as_ptr() as u64;
    // The row of DeltaNet layer `l`, lane `i` (every request carries a source,
    // checked above).
    let row = |l: usize, i: usize| -> GdnLaneTableFrom {
        let (r, view) = (&reqs[i], &views[i]);
        let d = &*r.dn_state;
        let [s_src, scales_src, _conv_src, ef_src] =
            r.dn_state_src.map_or([0; 4], |src| src.layer_state_ptrs(l));
        GdnLaneTableFrom {
            base: GdnLayerTableFrom {
                base: GdnLayerTable {
                    q: ptr(&view.dn_q_batch),
                    k: ptr(&view.dn_k_batch),
                    v: ptr(&view.dn_v_batch),
                    gate: ptr(&view.dn_alpha_batch),
                    beta: ptr(&view.dn_beta_batch),
                    s_q8: ptr(&d.s_matrices[l]),
                    s_scales: ptr(&d.s_scales[l]),
                    output: ptr(&view.dn_attn_out_batch),
                    ef: d.ef_residual(l).map_or(0, ptr),
                },
                s_q8_src: s_src,
                s_scales_src: scales_src,
                ef_src,
            },
            n_tokens: r.tokens.len() as u32,
            frame_off: offs[i] as u32,
        }
    };
    let mut cache = from
        .host
        .lock()
        .map_err(|_| HipError::new(0, "gdn lane from-table cache poisoned"))?;
    // Only the chunk's own lanes are compared: rows of lanes beyond it are
    // never read by the launch.
    let hit = cache.lookup_by(|host: &Vec<GdnLaneTableFrom>| {
        host.len() == total
            && (0..n_delta)
                .all(|l| (0..reqs.len()).all(|i| host[l * CONV_LANES_MAX + i] == row(l, i)))
    });
    match hit {
        Ok(slot) => {
            cache.mark_used(slot);
            Ok(slot)
        }
        Err(victim) => {
            // Unkeyed while it is rewritten: a failed upload leaves the slot
            // empty (a full re-stage), never keyed to an unknown table.
            let mut host = cache.take(victim).unwrap_or_default();
            if host.len() != total {
                let z = GdnLayerTable {
                    q: 0,
                    k: 0,
                    v: 0,
                    gate: 0,
                    beta: 0,
                    s_q8: 0,
                    s_scales: 0,
                    output: 0,
                    ef: 0,
                };
                let zb = GdnLayerTableFrom {
                    base: z,
                    s_q8_src: 0,
                    s_scales_src: 0,
                    ef_src: 0,
                };
                let zero = GdnLaneTableFrom {
                    base: zb,
                    n_tokens: 0,
                    frame_off: 0,
                };
                host.clear();
                host.resize(total, zero);
            }
            for l in 0..n_delta {
                for i in 0..reqs.len() {
                    host[l * CONV_LANES_MAX + i] = row(l, i);
                }
            }
            gpu.upload_gdn_lane_table(&from.devices[victim], table_bytes(&host))?;
            cache.set(victim, host);
            Ok(victim)
        }
    }
}

/// Does this request's singleton attend run `attention_fp8_e4m3_kv_batched`
/// after `kv_cache_write_fp8_e4m3_batched` (the fp8 scalar-batched arm:
/// gfx1201, under 64 rows, context at most the 4096 crossover, not a
/// DFlash chain-verify block, which takes the tile + reduce —
/// `DispatchCtx::verify_tile_attend`)? Then the
/// segment twin reproduces it.
fn attn_fp8_twin_eligible(
    gpu: &Gpu,
    s: &Qwen35Scratch,
    kv: &llama::KvCache,
    start_pos: usize,
    n: usize,
    fusion: DflashFusionCtx,
) -> bool {
    if !(gpu.arch_caps.is_gfx1201()
        && kv.quant_fp8
        && (MIN_BATCH..64).contains(&n)
        && start_pos + n <= 4096
        && !fusion.is_chain_verify())
    {
        return false;
    }
    KvTierPlan::derive(KvTierInputs {
        pos: start_pos,
        flash_mode: s.flash_mode as usize,
        capture_mode: gpu.graphs.capture_mode,
        batch_size: n,
        is_tree: false,
        ..kv.tier_inputs()
    })
    .is_ok_and(|p| {
        p.attend_key == hipfire_dispatch::types::KernelKey::AttnFp8E4m3KvBatchedMasked
            && p.write_key == hipfire_dispatch::types::KernelKey::KvWriteFp8E4m3Batched
    })
}

/// Rows `[r0, r0 + n)` of `pbs` as a scratch of `n` rows. Only row-major
/// fields are re-pointed; they alias `pbs` (never freed through the view).
fn rows_view(
    pbs: &PrefillBatchScratch,
    config: &Qwen35Config,
    r0: usize,
    n: usize,
) -> std::mem::ManuallyDrop<PrefillBatchScratch> {
    let dim = config.dim;
    let hidden = config.hidden_dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let qkv_dim = 2 * k_dim + v_dim;
    let nv = config.linear_num_value_heads;
    let q_dim = config.n_heads * config.head_dim;
    let kv_dim = config.n_kv_heads * config.head_dim;
    let v = |t: &GpuTensor, w: usize| t.sub_offset(r0 * w, n * w);
    // SAFETY: a bitwise copy whose every row-major tensor is then replaced
    // by a non-owning view of the same buffer; `ManuallyDrop` keeps the copy
    // from ever releasing anything `pbs` owns.
    unsafe {
        let mut p = std::mem::ManuallyDrop::new(std::ptr::read(pbs));
        let set = |slot: &mut GpuTensor, t: GpuTensor| std::ptr::write(slot, t);
        p.max_batch = n;
        set(&mut p.x_batch, v(&pbs.x_batch, dim));
        set(&mut p.x_rot_batch, v(&pbs.x_rot_batch, dim));
        set(&mut p.x_norm_batch, v(&pbs.x_norm_batch, dim));
        set(&mut p.dn_qkv_batch, v(&pbs.dn_qkv_batch, qkv_dim));
        set(&mut p.dn_z_batch, v(&pbs.dn_z_batch, v_dim));
        set(&mut p.dn_z_fold_batch, v(&pbs.dn_z_fold_batch, v_dim + 256));
        set(&mut p.dn_alpha_batch, v(&pbs.dn_alpha_batch, nv));
        set(&mut p.dn_beta_batch, v(&pbs.dn_beta_batch, nv));
        set(&mut p.dn_q_raw_batch, v(&pbs.dn_q_raw_batch, k_dim));
        set(&mut p.dn_k_raw_batch, v(&pbs.dn_k_raw_batch, k_dim));
        set(&mut p.dn_v_batch, v(&pbs.dn_v_batch, v_dim));
        set(&mut p.dn_q_batch, v(&pbs.dn_q_batch, v_dim));
        set(&mut p.dn_k_batch, v(&pbs.dn_k_batch, v_dim));
        set(&mut p.dn_attn_out_batch, v(&pbs.dn_attn_out_batch, v_dim));
        set(&mut p.dn_normed_batch, v(&pbs.dn_normed_batch, v_dim));
        set(&mut p.gate_ffn_batch, v(&pbs.gate_ffn_batch, hidden));
        set(&mut p.up_batch, v(&pbs.up_batch, hidden));
        set(&mut p.ffn_hidden_batch, v(&pbs.ffn_hidden_batch, hidden));
        set(
            &mut p.dn_normed_rot_batch,
            v(&pbs.dn_normed_rot_batch, v_dim),
        );
        set(&mut p.positions, v(&pbs.positions, 1));
        set(&mut p.rope_positions, v(&pbs.rope_positions, 1));
        set(&mut p.pos3, v(&pbs.pos3, 3));
        set(&mut p.ext_emb_index, v(&pbs.ext_emb_index, 1));
        set(&mut p.tokens, v(&pbs.tokens, 1));
        set(&mut p.fa_q_full_batch, v(&pbs.fa_q_full_batch, 2 * q_dim));
        set(&mut p.fa_q_batch, v(&pbs.fa_q_batch, q_dim));
        set(&mut p.fa_gate_batch, v(&pbs.fa_gate_batch, q_dim));
        set(&mut p.fa_k_batch, v(&pbs.fa_k_batch, kv_dim));
        set(&mut p.fa_v_batch, v(&pbs.fa_v_batch, kv_dim));
        set(&mut p.fa_attn_out_batch, v(&pbs.fa_attn_out_batch, q_dim));
        set(
            &mut p.fa_attn_out_rot_batch,
            v(&pbs.fa_attn_out_rot_batch, q_dim),
        );
        p
    }
}

/// Must a layer's output projection and FFN run per request view because a
/// request's singleton takes the ChainVerify S4 residual arm for a consumer of
/// dtype `w_dtype` at its own row count (`s4_residual_fast` admits only
/// `1..=16` rows)?
///
/// Never on exact gfx1201: there the S4 arm is byte-identical to the `Off`
/// route at any row count `<= 63` (row-parallel F16 producers computing the
/// F32 pipeline's values in-register and casting with `convert_f32_to_f16`'s
/// cast; the same one-tile-arithmetic residual GEMM; every other n-dependent
/// route switch is at `>= 64` rows), so the stage runs once at the combined
/// row count. On exact gfx1100 the S4 residual GEMM picks ldsstage / split-K
/// tiers at `n <= 16` that the `Off` route does not reproduce at larger `n`
/// (not proven identical), so it keeps the split.
fn any_lane_s4(gpu: &Gpu, reqs: &[MultiChunkRequest<'_>], w_dtype: rdna_compute::DType) -> bool {
    !gpu.arch_caps.is_gfx1201()
        && reqs.iter().any(|r| {
            s4_residual_fast(
                gpu,
                r.fusion,
                w_dtype,
                &BatchEpilogue::Residual,
                r.tokens.len(),
            )
        })
}

/// Capture the post-layer residual rows of every request whose ring extracts
/// `layer_idx` (the singleton's post-FFN `write_chunk_rows` of `x_batch`).
fn capture_layer_rows(
    gpu: &mut Gpu,
    reqs: &[MultiChunkRequest<'_>],
    views: &[std::mem::ManuallyDrop<PrefillBatchScratch>],
    layer_idx: usize,
) -> HipResult<()> {
    for (r, view) in reqs.iter().zip(views) {
        if let Some(rb) = r.hidden_rb.as_deref() {
            if let Some(slot) = rb.extract_slot(layer_idx) {
                rb.write_chunk_rows(gpu, slot, &view.x_batch, r.tokens.len())?;
            }
        }
    }
    Ok(())
}

/// Run every request's rows through one trunk forward, per request
/// byte-identical to `forward_prefill_batch_with_pbs_opts` over its rows
/// with the request's own `fusion`, `gdn_tape` (offset 0) and `hidden_rb`.
/// With `hidden_out`, writes post-output-norm hidden rows `[total x dim]` in
/// request order (`HiddenCapture::Verify`, as the singleton verify).
///
/// Shared across requests: embedding, the norm/rotate + input projection
/// GEMMs, and the output projection + FFN (on exact gfx1201, including for
/// ChainVerify lanes, whose S4 arm equals the `Off` route; elsewhere only when
/// no request's singleton takes the ChainVerify S4 residual arm / gfx1100 F16
/// projection route at its own row count). Per request: positions, GDN
/// pre/tape + recurrence, FA prep/KV write/attend with that request's fusion
/// flags, and the ring capture. A layer where a non-gfx1201 request would
/// select a ChainVerify route at its own `n` runs those stages on every
/// request's row view with its own `fusion` and `n` (MTP views with `Off`),
/// never at the combined count.
///
/// Every precondition (rows `2..=63` per request, in total at most the
/// effective cap — `<= 63` on the product route, `64..=128` only when
/// [`multi_chunk_wide_admitted`] holds for this target — capacities,
/// ring/tape bounds, Q8+EF DeltaNet, uncompacted Q8/fp8 KV, no capture or
/// recording) is checked before any request state is mutated; the request is
/// refused before the first launch otherwise. A combined count of 64 or more
/// runs the shared stages with [`DenseBatchMath::SingletonWmma`], as does any
/// row count under the PM FP8 verify scope (`Gpu::fp8_verify_scope`, see
/// [`verify_dense_math`]: the `*VerifyExact` keys route to the FP8 verify
/// GEMMs at every row count); otherwise at most 63 rows keep
/// [`DenseBatchMath::Product`] exactly as before.
pub fn forward_prefill_batch_multi(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    scratch: &MultiChunkScratch,
    reqs: &mut [MultiChunkRequest<'_>],
    hidden_out: Option<&GpuTensor>,
) -> HipResult<()> {
    crate::qwen35::vl_prefill::refuse_active("multi-request lane prefill")?;
    let refuse = |why: &str| {
        Err(HipError::new(
            0,
            &format!("forward_prefill_batch_multi: {why}"),
        ))
    };
    let pbs = &scratch.pbs;
    let total: usize = reqs.iter().map(|r| r.tokens.len()).sum();
    if reqs.is_empty() || total > multi_chunk_row_cap(gpu) || total > pbs.max_batch || pbs.lean {
        return refuse("row count outside 1..=effective cap (63, or 128 on the wide verify route) / scratch capacity, or lean scratch");
    }
    // Aggregate 64+ rows: the default projection route is quantized and not
    // the singleton's arithmetic; only the exact verify GEMMs of a dense
    // MQ4G256V2 target whose singleton runs the one-tile F16-WMMA chain
    // reproduce it. Refused before any state write. Inside the IU4 verify
    // scope every row count (singleton included) runs the A4 producers +
    // IU4 GEMMs instead, whose rows do not depend on the count.
    let wide = total > MULTI_CHUNK_PRODUCT_MAX_ROWS;
    if wide && !multi_chunk_wide_admitted(gpu, weights, config) {
        return refuse("64+ rows need the exact wide verify route: a dense MQ4G256V2 gfx1201 target whose singleton runs the one-tile F16-WMMA chain");
    }
    let math = if gpu.iu4_verify_scope {
        DenseBatchMath::Iu4Verify
    } else if wide {
        DenseBatchMath::SingletonWmma
    } else {
        verify_dense_math(gpu)
    };
    // Never the plain-AR retained body (see `forward_prefill_batch`'s batched
    // path). Cleared before the refusal: a recording window left open by a
    // decode forward that returned early (RecordingWarmup with stale
    // eligibility) then no longer reads as recording this batched forward, which
    // runs eager; a genuine graph capture is still refused, and the next
    // plain-AR forward re-arms eligibility.
    gpu.replay.enter_batched_forward();
    if gpu.graphs.capture_mode || gpu.replay.is_recording() {
        return refuse("graph capture / replay recording is not supported");
    }
    let arch = gpu.arch.clone();
    let mut kv_ends = Vec::with_capacity(reqs.len());
    for r in reqs.iter() {
        let n = r.tokens.len();
        // The singleton runs n == 1 through forward_scratch, not this body.
        if n < MIN_BATCH {
            return refuse("every request needs >= 2 rows");
        }
        // A lone 64+-row request would be the singleton's quantized route.
        if n > MULTI_CHUNK_MAX_LANE_ROWS {
            return refuse("every request needs <= 63 rows");
        }
        if !prefill_batch_pbs_eligible(weights, config, r.dn_state, n, &arch, true) {
            return refuse("request not eligible for the batched prefill body");
        }
        let d = &r.dn_state;
        if d.quant != StateQuant::Q8 || d.s_ef_residual.len() != d.s_matrices.len() {
            return refuse("DeltaNet state must be Q8 with error feedback");
        }
        let kv = &r.kv_cache;
        if !(kv.quant_q8 || kv.quant_fp8) || kv.compact_offset != 0 {
            return refuse("KV must be uncompacted Q8 or fp8");
        }
        if r.gdn_tape.is_some_and(|t| t.max_n < n) {
            return refuse("GDN tape smaller than the request's rows");
        }
        if let Some(rb) = r.hidden_rb.as_deref() {
            // The singleton verify stages `n` rows per extract layer and
            // commits them; a ring it would write straight (n > staging) or
            // cannot hold the rows is not reproduced here.
            if rb.hidden_dim != config.dim
                || rb.layer_bufs.len() != rb.extract_layers.len()
                || rb.staging_bufs.len() != rb.extract_layers.len()
                || rb.extract_layers.iter().any(|&l| l >= config.n_layers)
                || n > rb.max_batch
                || n > rb.max_positions
            {
                return refuse(
                    "hidden ring layers/dimension/staging cannot hold the request's rows",
                );
            }
        }
        kv_ends.push(checked_kv_end(
            r.start_pos,
            n,
            "forward_prefill_batch_multi",
        )?);
    }
    if !weights.layers.iter().all(|l| match l {
        LayerWeights::DeltaNet(_) => true,
        LayerWeights::FullAttn(_) => qwen35_layer_batch_admissible(l, config, &arch).is_ok(),
        _ => false,
    }) {
        return refuse("only dense DeltaNet / batched FullAttention layers");
    }
    // Source mode (save-free verify), decided and validated BEFORE any request
    // state is written. Either every request carries a snapshot source (the
    // caller already exchanged the owners: `dn_state` is the not-yet-written
    // destination bank) or none does. A sourced chunk that cannot take the
    // lane-batched GDN route, whose from-table / snapshot is not exact, or
    // whose rows exceed the replay step bound is an error: the per-request
    // in-place route would read the uninitialised destination bank.
    let sem = BatchSemantics::Sequential;
    let lane_gdn = lane_gdn_chunk_admitted(gpu, config, reqs, sem);
    let n_sourced = reqs.iter().filter(|r| r.dn_state_src.is_some()).count();
    let lane_from: Option<&GdnLaneFromTable> = if n_sourced == 0 {
        None
    } else {
        if n_sourced != reqs.len() {
            return refuse(
                "save-free verify: every request of a chunk must carry a snapshot source, or none",
            );
        }
        if !lane_gdn {
            return refuse("save-free verify: a sourced chunk needs the lane-batched GDN route");
        }
        let Some(from) = scratch.gdn_lane_from.as_ref() else {
            return refuse("save-free verify: the snapshot-source lane table is not armed");
        };
        for r in reqs.iter() {
            if r.tokens.len() > rdna_compute::dflash_gdn_replay::DFLASH_GDN_REPLAY_MAX_STEPS {
                return refuse(
                    "save-free verify: more rows than the snapshot-source replay admits",
                );
            }
            if !r
                .dn_state_src
                .is_some_and(|src| src.cb_verify_owned_exact(&*r.dn_state, config))
            {
                return refuse(
                    "save-free verify: DeltaNet state / snapshot are not an exact owned pair",
                );
            }
        }
        Some(from)
    };
    // Capacity/mapping (warms mapped KV; no request state is written).
    for (r, &end) in reqs.iter_mut().zip(&kv_ends) {
        release_widened_pbs_for_kv_growth(gpu, r.kv_cache, config, s, end)?;
        r.kv_cache.ensure_mapped_capacity(gpu, end)?;
        r.kv_cache.require_mapped_capacity(end)?;
    }

    let dim = config.dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    // Fusion of every stage run once over the combined rows: those stages
    // read `fusion` only through the gfx1100 F16 projection route (checked
    // per request below, which switches them to per-request views) and the
    // S4 arm (checked per layer), so `Off` is the singleton's route for them.
    let shared_fusion = DflashFusionCtx::Off;
    // Request row ranges in the shared scratch.
    let mut offs = Vec::with_capacity(reqs.len());
    let mut off = 0usize;
    for r in reqs.iter() {
        offs.push(off);
        off += r.tokens.len();
    }
    let views: Vec<_> = reqs
        .iter()
        .zip(&offs)
        .map(|(r, &o)| rows_view(pbs, config, o, r.tokens.len()))
        .collect();
    // Any request on the ChainVerify gfx1100 F16 projection route at its own
    // row count: the projection/FFN input stages run per request view.
    let split_proj = reqs
        .iter()
        .any(|r| mq_f16_projection_fast_route(gpu, r.fusion, r.tokens.len(), dim));

    let tokens: Vec<u32> = reqs.iter().flat_map(|r| r.tokens.iter().copied()).collect();
    batch_chunk_embed_tokens(
        gpu,
        weights,
        &tokens,
        s,
        pbs,
        total,
        dim,
        dim * 4,
        true,
        false,
        false,
        None,
        None,
    )?;
    if cb_upload_coalesce_enabled() {
        // One upload of every lane's positions. Each per-lane call below (all
        // Sequential, no tree ctx, no active mask, not pre-uploaded) is a
        // single `memcpy_htod` of `start_pos..start_pos + n` into its row view
        // of `pbs.positions`, and the views are contiguous in lane order
        // (`offs` = cumulative rows), so the concatenation written at offset 0
        // of `pbs.positions` is the same bytes at the same addresses.
        // `rope_positions` is written only for a tree ctx (None here).
        let positions: Vec<i32> = reqs
            .iter()
            .flat_map(|r| (0..r.tokens.len()).map(move |i| (r.start_pos + i) as i32))
            .collect();
        debug_assert_eq!(positions.len(), total);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(positions.as_ptr() as *const u8, positions.len() * 4)
        };
        gpu.hip.memcpy_htod(&pbs.positions.buf, bytes)?;
    } else {
        for (r, view) in reqs.iter().zip(&views) {
            batch_chunk_upload_positions(gpu, view, sem, r.start_pos, r.tokens.len(), None, false)?;
        }
    }
    let q8_wmma_arch = q8_prefill_wmma_enabled(gpu);
    let arch_has_wmma = q8_wmma_arch;
    let tapes = reqs.iter().any(|r| r.gdn_tape.is_some());
    let workload = prefill_dispatch_workload(hidden_out.is_some(), tapes, false);
    // MTP / plain requests keep the attend crossover; a ChainVerify request
    // attends through the tile + reduce (`verify_tile_attend`).
    let ctx = DispatchCtx::new(gpu).with_workload(workload);
    let chain_ctx = DispatchCtx::new(gpu)
        .with_workload(workload)
        .with_verify_tile_attend(true);

    // Requests whose singleton attend is the fp8 scalar-batched arm run it
    // as one segment-twin launch per layer (tables staged up front). All
    // twin segments share the singleton `max_seq` (`physical_cap`) word.
    let mut twin_attn: Vec<bool> = reqs
        .iter()
        .map(|r| {
            *SEG_TWINS
                && attn_fp8_twin_eligible(gpu, s, r.kv_cache, r.start_pos, r.tokens.len(), r.fusion)
        })
        .collect();
    let twin_cap = reqs
        .iter()
        .zip(&twin_attn)
        .find(|(_, &t)| t)
        .map(|(r, _)| r.kv_cache.physical_cap);
    for (r, t) in reqs.iter().zip(twin_attn.iter_mut()) {
        *t &= Some(r.kv_cache.physical_cap) == twin_cap;
    }
    let twin_ctx: Vec<usize> = reqs
        .iter()
        .zip(&twin_attn)
        .filter(|(_, &t)| t)
        .map(|(r, _)| r.start_pos + r.tokens.len())
        .collect();
    let twin_rows = reqs
        .iter()
        .zip(&twin_attn)
        .filter(|(_, &t)| t)
        .map(|(r, _)| r.tokens.len())
        .max()
        .unwrap_or(0);
    if !twin_ctx.is_empty() {
        let mut fa_idx = 0usize;
        for (layer_idx, ty) in config.layer_types.iter().enumerate() {
            if *ty != LayerType::FullAttention {
                continue;
            }
            let segs: Vec<AttnFp8Seg> = reqs
                .iter()
                .zip(&views)
                .zip(&twin_attn)
                .filter(|(_, &t)| t)
                .map(|((r, view), _)| AttnFp8Seg {
                    q: view.fa_q_batch.buf.as_ptr() as u64,
                    k_cache: r.kv_cache.k_gpu[layer_idx].buf.as_ptr() as u64,
                    v_cache: r.kv_cache.v_gpu[layer_idx].buf.as_ptr() as u64,
                    out: view.fa_attn_out_batch.buf.as_ptr() as u64,
                    positions: view.positions.buf.as_ptr() as u64,
                    n_rows: r.tokens.len() as u64,
                })
                .collect();
            seg_twins::stage_attention_fp8_segs(
                gpu,
                &scratch.attn_segs,
                fa_idx * MULTI_CHUNK_MAX_SEGS,
                &segs,
                config.n_heads,
                config.head_dim,
            )?;
            fa_idx += 1;
        }
    }

    let mut delta_layer_idx = 0usize;
    let mut kv_layer_idx = 0usize;
    // Deferred tape-capture copies of one DeltaNet layer (all requests),
    // issued as one batched launch after the layer's per-request recurrence.
    let mut tape_copies: Vec<rdna_compute::tape_copy::TapeCopy> =
        Vec::with_capacity(if *TAPE_COPY_BATCH && tapes {
            (3 + usize::from(lane_from.is_some())) * reqs.len()
        } else {
            0
        });
    // Lane-batched GDN route (see `lane_gdn_chunk_admitted`, decided above
    // with the source mode): the batched tape copy (`TAPE_COPY_BATCH`), whose
    // descriptors it queues, is part of its admission, so each kill switch
    // keeps its own A/B meaning.
    // Each lane's own row count and prefix frame offset live in its table row
    // (`offs`, reset to zero per chunk); the recurrence launches reserve `total`.
    let mut lane_conv: Vec<(u64, usize, usize)> =
        Vec::with_capacity(if lane_gdn { reqs.len() } else { 0 });
    let mut lane_from_slot = 0usize;
    if lane_gdn {
        if lane_from.is_some() {
            lane_from_slot = stage_gdn_lane_table_from(gpu, scratch, config, reqs, &views, &offs)?;
        } else {
            stage_gdn_lane_table(gpu, scratch, config, reqs, &views, &offs)?;
        }
    }
    // Lane-batched FullAttention route (see `lane_fa_chunk_admitted`): decided
    // once per chunk; the disjoint per-lane partials are held (locked) for the
    // whole forward.
    let mut lane_fa_guard = None;
    if lane_fa_chunk_admitted(gpu, config, s, reqs, sem, &twin_attn) {
        let shape: Vec<(usize, usize)> = reqs
            .iter()
            .map(|r| (r.tokens.len(), r.start_pos + r.tokens.len()))
            .collect();
        lane_fa_guard = scratch
            .lane_fa_partials_acquire(gpu, Gpu::lane_fa_partials_len(config.n_heads, &shape));
    }
    let lane_fa_partials: Option<&GpuTensor> = lane_fa_guard.as_ref().and_then(|g| (**g).as_ref());
    let dims = hybrid_dims(config);
    let batch = pbs.view();
    let flash = flash_scratch(s);
    let tap_fn = batched_attention_tap(config);
    let tap = tap_fn.get();
    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                // batch_chunk_delta_net_attn, non chunk-scan arm (n < 64).
                if split_proj {
                    for (r, view) in reqs.iter().zip(&views) {
                        batch_chunk_delta_net_input_projection(
                            gpu,
                            &layer.mixer_view(),
                            &dims,
                            &view.view(),
                            r.tokens.len(),
                            dim,
                            q8_wmma_arch,
                            r.fusion,
                            None,
                            DenseBatchMath::Product,
                        )?;
                    }
                } else {
                    batch_chunk_delta_net_input_projection(
                        gpu,
                        &layer.mixer_view(),
                        &dims,
                        &batch,
                        total,
                        dim,
                        q8_wmma_arch,
                        shared_fusion,
                        None,
                        math,
                    )?;
                }
                if lane_gdn {
                    // Lane-batched pre-GDN + recurrence: one launch per stage
                    // for all requests. Byte-identical to the per-request loop
                    // below (admission: `lane_gdn_chunk_admitted`) because
                    //  * the sigmoid/alpha-gate and the Q/K norm-interleave
                    //    kernels are row-local (the row folds into grid.x and
                    //    each row's arithmetic is independent of the row
                    //    count), so one launch over all `total` rows equals a
                    //    launch per request over its row view;
                    //  * the conv launch runs each request's conv1d body
                    //    verbatim over its own rows and its own conv ring
                    //    (kernels/src/conv1d_silu_split_lanes.hip);
                    //  * the multi-lane recurrence is the single-lane body
                    //    per grid.z row; each lane's table row carries its
                    //    own row count and its prefix frame offset
                    //    (`offs[i]` = rows of the lanes before it), so lane
                    //    `i`'s stochastic-rounding frame is `base + offs[i]`:
                    //    the request-order frame sequence of per-request
                    //    launches for ANY row counts (the launch reserves
                    //    `total` frames once; the single final requant of a
                    //    lane uses its interval's START, `base + offs[i]`,
                    //    and EF consumes no RNG but reserves the same frames);
                    //  * requests own disjoint rows of the shared scratch and
                    //    disjoint DeltaNet/tape state, so interleaving their
                    //    work across launches changes no byte.
                    gdn_sigmoid_alpha_gate(gpu, &layer.mixer_view(), &batch, n_v_heads, total)?;
                    // DFlash tape capture (the pre-GDN sink branch's three
                    // descriptors per request): nothing below writes the
                    // source rows, and the next layer's producers have not
                    // run yet.
                    for (r, view) in reqs.iter().zip(&views) {
                        if let Some(tape) = r.gdn_tape {
                            push_gdn_tape_copies(
                                &mut tape_copies,
                                &gdn_tape_view(tape),
                                &view.view(),
                                n_v_heads,
                                0,
                                r.tokens.len(),
                                delta_layer_idx,
                            );
                        }
                    }
                    if lane_from.is_some() {
                        // Save-free verify (byte-exactness): each lane's live
                        // conv ring is the freshly exchanged, never-written
                        // bank B; its pre-window ring is in the snapshot bank
                        // A. Copy A -> B (pure bytes, the LOGICAL ring extent
                        // the kernel reads and writes), batched into the same
                        // descriptor launch as the tape copies, then run the
                        // UNCHANGED in-place conv kernel on B below: it reads
                        // B (== A's bytes) and writes B in today's order, so
                        // the conv output rows and the final ring are what the
                        // in-place verify of a SAVEd lane produces. A is never
                        // written, so it stays the pre-window state commit /
                        // terminal repair restore. A source/destination conv
                        // kernel twin was rejected: the macro-built twin
                        // compiles to different FMA contraction than the
                        // in-place kernel (1-8 ulp on gfx1201). The ring copy
                        // is ~4.9% of the state bytes the old SAVE moved
                        // (conv ring vs S + scales + EF).
                        let conv_bytes =
                            ((2 * k_dim + v_dim) * (config.conv_kernel_dim - 1) * 4) as u64;
                        for r in reqs.iter() {
                            let Some(src) = r.dn_state_src else {
                                return refuse(
                                    "save-free verify: request without a snapshot source",
                                );
                            };
                            tape_copies.push(rdna_compute::tape_copy::TapeCopy {
                                dst: r.dn_state.conv_states[delta_layer_idx].buf.as_ptr() as u64,
                                src: src.layer_state_ptrs(delta_layer_idx)[2],
                                bytes: conv_bytes,
                            });
                        }
                    }
                    if !tape_copies.is_empty() {
                        gpu.tape_copy_batched(&tape_copies)?;
                        tape_copies.clear();
                    }
                    lane_conv.clear();
                    lane_conv.extend(reqs.iter().zip(&offs).map(|(r, &o)| {
                        (
                            r.dn_state.conv_states[delta_layer_idx].buf.as_ptr() as u64,
                            o,
                            r.tokens.len(),
                        )
                    }));
                    gpu.conv1d_silu_split_f32_lanes(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch,
                        &layer.conv_weight,
                        &lane_conv,
                        k_dim,
                        v_dim,
                    )?;
                    gdn_qk_norm_interleave(gpu, &dims, &batch, n_v_heads, hd, total)?;
                    match lane_from {
                        Some(from) => {
                            // `_from` changes only the S / scales / EF LOAD
                            // addresses (bank A) of the unchanged recurrence:
                            // same frames, reductions and final requant; every
                            // output and every state element (S, scales, f16
                            // EF residual) is stored to the destination bank
                            // B (the EF residual is read from A at the final
                            // requant and the new residual written to B).
                            let table = (from.devices[lane_from_slot].buf.as_ptr() as *const u8)
                                .wrapping_add(delta_layer_idx * GDN_LANE_FROM_TABLE_LAYER_BYTES)
                                as *const std::ffi::c_void;
                            gpu.gated_delta_net_q8_fast_lanes_varrows_from(
                                table,
                                reqs.len(),
                                total,
                                n_v_heads,
                                config.linear_value_head_dim,
                            )?;
                        }
                        None => {
                            let table = (scratch.gdn_lane_table.buf.as_ptr() as *const u8)
                                .wrapping_add(delta_layer_idx * GDN_LANE_TABLE_LAYER_BYTES)
                                as *const std::ffi::c_void;
                            gpu.gated_delta_net_q8_fast_lanes_varrows(
                                table,
                                reqs.len(),
                                total,
                                n_v_heads,
                                config.linear_value_head_dim,
                            )?;
                        }
                    }
                } else {
                    for (r, view) in reqs.iter_mut().zip(&views) {
                        let n = r.tokens.len();
                        let sink = (*TAPE_COPY_BATCH).then_some(&mut tape_copies);
                        let parents = batch_chunk_delta_net_pre_gdn(
                            gpu,
                            &layer.mixer_view(),
                            &dims,
                            &view.view(),
                            &deltanet_state_view(r.dn_state),
                            n,
                            k_dim,
                            v_dim,
                            n_v_heads,
                            hd,
                            sem,
                            None,
                            r.gdn_tape.map(gdn_tape_view).as_ref(),
                            0,
                            delta_layer_idx,
                            r.fusion,
                            sink,
                        )?;
                        if parents.is_some() {
                            return refuse("tree recurrence in a linear verify");
                        }
                        let d = &*r.dn_state;
                        gpu.gated_delta_net_q8_batch_seq(
                            &view.dn_q_batch,
                            &view.dn_k_batch,
                            &view.dn_v_batch,
                            &view.dn_alpha_batch,
                            &view.dn_beta_batch,
                            &d.s_matrices[delta_layer_idx],
                            &d.s_scales[delta_layer_idx],
                            &view.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                            d.ef_residual(delta_layer_idx),
                        )?;
                    }
                }
                // The deferred tape copies read rows of dn_qkv/alpha/beta that
                // no kernel since each request's sigmoid has written; the next
                // layer's producers have not run yet.
                if !tape_copies.is_empty() {
                    gpu.tape_copy_batched(&tape_copies)?;
                    tape_copies.clear();
                }
                if split_proj
                    || any_lane_s4(gpu, reqs, layer.wo.gpu_dtype)
                    || any_lane_s4(gpu, reqs, layer.w_down.gpu_dtype)
                {
                    for (r, view) in reqs.iter().zip(&views) {
                        let n = r.tokens.len();
                        batch_chunk_delta_net_output_projection(
                            gpu,
                            &layer.mixer_view(),
                            &dims,
                            &view.view(),
                            n,
                            n_v_heads,
                            q8_wmma_arch,
                            arch_has_wmma,
                            BatchEpilogue::Residual,
                            r.fusion,
                            GdnScanOut::F32,
                            DenseBatchMath::Product,
                        )?;
                        layer.ffn_rows(
                            gpu,
                            config.norm_eps,
                            &view.view(),
                            n,
                            dim,
                            q8_wmma_arch,
                            arch_has_wmma,
                            MlpOutput::Residual,
                            r.fusion,
                            DenseBatchMath::Product,
                        )?;
                    }
                } else {
                    batch_chunk_delta_net_output_projection(
                        gpu,
                        &layer.mixer_view(),
                        &dims,
                        &batch,
                        total,
                        n_v_heads,
                        q8_wmma_arch,
                        arch_has_wmma,
                        BatchEpilogue::Residual,
                        shared_fusion,
                        GdnScanOut::F32,
                        math,
                    )?;
                    layer.ffn_rows(
                        gpu,
                        config.norm_eps,
                        &batch,
                        total,
                        dim,
                        q8_wmma_arch,
                        arch_has_wmma,
                        MlpOutput::Residual,
                        shared_fusion,
                        math,
                    )?;
                }
                capture_layer_rows(gpu, reqs, &views, layer_idx)?;
                delta_layer_idx += 1;
            }
            (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
                // batch_chunk_full_attn_attn with the per-request flags each
                // request's own singleton chunk computes (all n < 64).
                if split_proj {
                    for (r, view) in reqs.iter().zip(&views) {
                        batch_chunk_full_attn_input_projection(
                            gpu,
                            &layer.mixer_view(),
                            &dims,
                            &view.view(),
                            r.tokens.len(),
                            dim,
                            q8_wmma_arch,
                            r.fusion,
                            DenseBatchMath::Product,
                        )?;
                    }
                } else {
                    batch_chunk_full_attn_input_projection(
                        gpu,
                        &layer.mixer_view(),
                        &dims,
                        &batch,
                        total,
                        dim,
                        q8_wmma_arch,
                        shared_fusion,
                        math,
                    )?;
                }
                if let Some(lane_partials) = lane_fa_partials {
                    // Lane-batched route: the unfused prep's row-local stages
                    // once over all rows (every lane's rows are contiguous
                    // rows of the shared scratch, each position row uploaded
                    // with its own start_pos; `compact_offset` is 0 for every
                    // lane, see `lane_fa_chunk_admitted`), then one KV write
                    // per side and one VerifyAttn tile + reduce for all
                    // lanes. Per request this is exactly the unfused prep +
                    // `batch_chunk_fa_attend` (fp8 write, then the
                    // `verify_tile_attend` tile + reduce).
                    fa_prep_unfused_qk_norm(gpu, &layer.mixer_view(), &dims, &batch, total)?;
                    fa_prep_unfused_rope(gpu, &dims, &batch, &pbs.positions, total, 0)?;
                    let k_writes: Vec<rdna_compute::lane_fa::LaneFaWrite<'_>> = reqs
                        .iter()
                        .zip(&views)
                        .map(|(r, view)| rdna_compute::lane_fa::LaneFaWrite {
                            cache: &r.kv_cache.k_gpu[layer_idx],
                            src: &view.fa_k_batch,
                            positions: &view.positions,
                            rows: r.tokens.len(),
                        })
                        .collect();
                    gpu.kv_cache_write_fp8_e4m3_lanes(
                        &k_writes,
                        config.n_kv_heads,
                        config.head_dim,
                    )?;
                    let v_writes: Vec<rdna_compute::lane_fa::LaneFaWrite<'_>> = reqs
                        .iter()
                        .zip(&views)
                        .map(|(r, view)| rdna_compute::lane_fa::LaneFaWrite {
                            cache: &r.kv_cache.v_gpu[layer_idx],
                            src: &view.fa_v_batch,
                            positions: &view.positions,
                            rows: r.tokens.len(),
                        })
                        .collect();
                    gpu.kv_cache_write_fp8_e4m3_lanes(
                        &v_writes,
                        config.n_kv_heads,
                        config.head_dim,
                    )?;
                    let attends: Vec<rdna_compute::lane_fa::LaneFaAttend<'_>> = reqs
                        .iter()
                        .zip(&views)
                        .map(|(r, view)| rdna_compute::lane_fa::LaneFaAttend {
                            q: &view.fa_q_batch,
                            k_cache: &r.kv_cache.k_gpu[layer_idx],
                            v_cache: &r.kv_cache.v_gpu[layer_idx],
                            out: &view.fa_attn_out_batch,
                            positions: &view.positions,
                            rows: r.tokens.len(),
                            max_ctx_len: r.start_pos + r.tokens.len(),
                            physical_cap: r.kv_cache.physical_cap,
                        })
                        .collect();
                    // Lanes the singleton serves with the native-fp8 split-KV
                    // verifier keep their singleton launch over the model's
                    // own `flash_partials` (batched == singleton); the rest
                    // share one VerifyAttn tile + reduce.
                    gpu.attention_verify_fp8_lanes_routed(
                        &attends,
                        lane_partials,
                        &s.flash_partials,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                    )?;
                } else {
                    for ((r, view), &twin) in reqs.iter_mut().zip(&views).zip(&twin_attn) {
                        let n = r.tokens.len();
                        let max_ctx_len = r.start_pos + n;
                        let rctx = if r.fusion.is_chain_verify() {
                            &chain_ctx
                        } else {
                            &ctx
                        };
                        let gfx12_fa_prep =
                            gfx12_fa_prep_admitted(gpu, &dims, r.fusion, n, tap.is_some());
                        let multirow = q8_multirow_attn_admitted(
                            gpu.arch_caps.arch(),
                            r.kv_cache.quant_q8,
                            config.head_dim,
                            n,
                            r.start_pos + n,
                            fa_pertoken_min_ctx(gpu.arch_caps.arch()),
                            false,
                            false,
                            false,
                            false,
                            false,
                        );
                        batch_chunk_full_attn_prepare(
                            gpu,
                            multirow,
                            &layer.mixer_view(),
                            &dims,
                            &view.view(),
                            &flash,
                            &kv_view(r.kv_cache),
                            n,
                            r.start_pos,
                            max_ctx_len,
                            rctx,
                            sem,
                            None,
                            kv_layer_idx,
                            layer_idx,
                            r.fusion,
                            gfx12_fa_prep,
                            false,
                            false,
                            tap,
                        )?;
                        if twin {
                            // The singleton attend's paired write; the attention
                            // itself runs in the segment twin below.
                            for (cache, src) in [
                                (&r.kv_cache.k_gpu[layer_idx], &view.fa_k_batch),
                                (&r.kv_cache.v_gpu[layer_idx], &view.fa_v_batch),
                            ] {
                                gpu.kv_cache_write_fp8_e4m3_batched(
                                    cache,
                                    src,
                                    &view.positions,
                                    config.n_kv_heads,
                                    config.head_dim,
                                    n,
                                )?;
                            }
                        } else {
                            // Same S6 paired Q8 K/V-write admission as the singleton
                            // attend (byte-identical to the two batched writes).
                            let pair_q8_writes = dflash_fa_batch_fusion_admitted(
                                gpu,
                                &dims,
                                r.fusion,
                                n,
                                tap.is_some(),
                            );
                            batch_chunk_fa_attend(
                                gpu,
                                &dims,
                                &view.view(),
                                &flash,
                                &kv_view(r.kv_cache),
                                n,
                                r.start_pos,
                                max_ctx_len,
                                rctx,
                                sem,
                                None,
                                layer_idx,
                                multirow,
                                None,
                                false,
                                pair_q8_writes,
                            )?;
                        }
                    }
                }
                if let Some(cap) = twin_cap.filter(|_| !twin_ctx.is_empty()) {
                    seg_twins::attention_fp8_segs(
                        gpu,
                        &scratch.attn_segs,
                        kv_layer_idx * MULTI_CHUNK_MAX_SEGS,
                        twin_rows,
                        &twin_ctx,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        cap,
                    )?;
                }
                if split_proj
                    || any_lane_s4(gpu, reqs, layer.wo.gpu_dtype)
                    || any_lane_s4(gpu, reqs, layer.w_down.gpu_dtype)
                {
                    for (r, view) in reqs.iter().zip(&views) {
                        let n = r.tokens.len();
                        batch_chunk_full_attn_output_projection(
                            gpu,
                            &layer.mixer_view(),
                            &view.view(),
                            n,
                            q8_wmma_arch,
                            arch_has_wmma,
                            BatchEpilogue::Residual,
                            r.fusion,
                            false,
                            None,
                            DenseBatchMath::Product,
                        )?;
                        layer.ffn_rows(
                            gpu,
                            config.norm_eps,
                            &view.view(),
                            n,
                            dim,
                            q8_wmma_arch,
                            arch_has_wmma,
                            MlpOutput::Residual,
                            r.fusion,
                            DenseBatchMath::Product,
                        )?;
                    }
                } else {
                    batch_chunk_full_attn_output_projection(
                        gpu,
                        &layer.mixer_view(),
                        &batch,
                        total,
                        q8_wmma_arch,
                        arch_has_wmma,
                        BatchEpilogue::Residual,
                        shared_fusion,
                        false,
                        None,
                        math,
                    )?;
                    layer.ffn_rows(
                        gpu,
                        config.norm_eps,
                        &batch,
                        total,
                        dim,
                        q8_wmma_arch,
                        arch_has_wmma,
                        MlpOutput::Residual,
                        shared_fusion,
                        math,
                    )?;
                }
                capture_layer_rows(gpu, reqs, &views, layer_idx)?;
                kv_layer_idx += 1;
            }
            _ => return refuse("layer type mismatch"),
        }
    }
    batch_chunk_final_logits(
        gpu,
        weights,
        config,
        s,
        pbs,
        total,
        dim,
        dim * 4,
        hidden_out.map(|t| (t, 0, HiddenCapture::Verify)),
        false,
        true,
        &ctx,
    )?;
    // The singleton chunk loop's per-chunk ring finish: scatter this
    // request's staged rows to its ring head and advance by its row count.
    for r in reqs.iter_mut() {
        let n = r.tokens.len();
        if let Some(rb) = r.hidden_rb.as_deref_mut() {
            rb.finish_prefill_chunk(gpu, n)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod pack_tests {
    use super::{
        multi_chunk_wide_route, pack_whole_lanes, scratch_rows_for, MULTI_CHUNK_MAX_ROWS,
        MULTI_CHUNK_PRODUCT_MAX_ROWS,
    };
    use rdna_compute::FeatureFlags;

    /// gfx1201 defaults (`HIPFIRE_CB_VERIFY_CHUNK128` and
    /// `HIPFIRE_WMMA_BATCH_TILES` are both on by default on gfx1201; the
    /// `for_test` constructor keeps both off, so they are set here).
    fn chunk128_flags() -> FeatureFlags {
        let mut f = FeatureFlags::for_test("gfx1201");
        f.cb_verify_chunk128 = true;
        f.wmma_batch_tiles = true;
        f
    }

    /// Every switch that moves a singleton (`<= 63`-row) dense MQ4G256V2 arm
    /// off the one-tile F16-WMMA chain the wide kernels reproduce.
    const SINGLETON_ARM_SWITCHES: [(&str, fn(&mut FeatureFlags)); 4] = [
        ("HIPFIRE_FP16=0", |f| f.fp16_disabled = true),
        ("HIPFIRE_LM_HEAD_WMMA=0", |f| f.lm_head_wmma_disabled = true),
        ("HIPFIRE_HFQ4G256_LDSSTAGE=1", |f| {
            f.hfq4g256_ldsstage_wmma = true
        }),
        ("HIPFIRE_WMMA_BATCH_TILES=0", |f| f.wmma_batch_tiles = false),
    ];

    #[test]
    fn wide_route_admitted_by_default_refused_under_each_singleton_switch() {
        assert!(multi_chunk_wide_route("gfx1201", &chunk128_flags()));
        // The chunk128 flag off, or another arch, never opens the wide route.
        assert!(!multi_chunk_wide_route(
            "gfx1201",
            &FeatureFlags::for_test("gfx1201")
        ));
        assert!(!multi_chunk_wide_route("gfx1100", &chunk128_flags()));
        for (name, set) in SINGLETON_ARM_SWITCHES {
            let mut f = chunk128_flags();
            set(&mut f);
            assert!(
                !multi_chunk_wide_route("gfx1201", &f),
                "{name} must refuse wide admission"
            );
        }
    }

    #[test]
    fn scratch_rows_equal_the_effective_cap() {
        // Admitted: the planned rows, up to 128.
        assert_eq!(
            scratch_rows_for(true, MULTI_CHUNK_MAX_ROWS),
            MULTI_CHUNK_MAX_ROWS
        );
        assert_eq!(scratch_rows_for(true, 4096), MULTI_CHUNK_MAX_ROWS);
        assert_eq!(scratch_rows_for(true, 96), 96);
        // Not admitted (any refusing switch, a non-dense target, flag off):
        // exactly the product cap, as before the wide route existed.
        for (name, set) in SINGLETON_ARM_SWITCHES {
            let mut f = chunk128_flags();
            set(&mut f);
            let rows =
                scratch_rows_for(multi_chunk_wide_route("gfx1201", &f), MULTI_CHUNK_MAX_ROWS);
            assert_eq!(rows, MULTI_CHUNK_PRODUCT_MAX_ROWS, "{name}");
        }
        assert_eq!(
            scratch_rows_for(false, MULTI_CHUNK_MAX_ROWS),
            MULTI_CHUNK_PRODUCT_MAX_ROWS
        );
        assert_eq!(scratch_rows_for(false, 32), 32);
        // Never below the batched-prefill minimum.
        assert_eq!(scratch_rows_for(false, 0), 2);
        assert_eq!(scratch_rows_for(true, 1), 2);
    }

    fn pack(rows: &[usize], cap: usize) -> Vec<std::ops::Range<usize>> {
        let mut out = Vec::new();
        pack_whole_lanes(rows, cap, &mut out).expect("packs");
        out
    }

    #[test]
    fn c8_block16_packs_by_effective_cap() {
        let rows = [16usize; 8];
        // Flag off (product cap 63): 48 + 48 + 32.
        assert_eq!(
            pack(&rows, MULTI_CHUNK_PRODUCT_MAX_ROWS),
            vec![0..3, 3..6, 6..8]
        );
        // Cap 64: 64 + 64.
        assert_eq!(pack(&rows, 64), vec![0..4, 4..8]);
        // Cap 128: one chunk.
        assert_eq!(pack(&rows, MULTI_CHUNK_MAX_ROWS), vec![0..8]);
        // A cap above the supported aggregate is clamped to it.
        assert_eq!(pack(&rows, 4096), vec![0..8]);
    }

    #[test]
    fn ragged_totals_pack_whole_lanes_in_order() {
        // 4 x 16 rows then 6 x 15 rows (154 in all).
        let rows = [16, 16, 16, 16, 15, 15, 15, 15, 15, 15];
        assert_eq!(pack(&rows, 64), vec![0..4, 4..8, 8..10]);
        assert_eq!(pack(&rows, 96), vec![0..6, 6..10]);
        assert_eq!(pack(&rows, 128), vec![0..8, 8..10]);
        // A 63-row lane leaves no room for a second lane at cap 64.
        assert_eq!(pack(&[63, 2], 64), vec![0..1, 1..2]);
        assert_eq!(pack(&[63, 63], 128), vec![0..2]);
        // Nothing split, padded or reordered: every lane appears exactly once.
        let r = pack(&[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31], 64);
        assert_eq!(r.first().map(|x| x.start), Some(0));
        assert_eq!(r.last().map(|x| x.end), Some(11));
        assert!(r.windows(2).all(|w| w[0].end == w[1].start));
    }

    #[test]
    fn a_lane_over_63_rows_or_under_2_is_refused() {
        let mut out = vec![0..1];
        assert!(pack_whole_lanes(&[16, 64], 128, &mut out).is_err());
        assert!(out.is_empty());
        assert!(pack_whole_lanes(&[1, 16], 128, &mut out).is_err());
        assert!(out.is_empty());
        // At the product cap a 64-row lane is also refused.
        assert!(pack_whole_lanes(&[64], 63, &mut out).is_err());
    }
}
