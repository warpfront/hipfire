// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Per-request VMM continuous-batching executor (plan §4.1/§4.2, slice 1B).
//
// One resident `Qwen35Weights` (borrowed from the existing model owner, never
// a second load) drives a shared trunk forward over rows from independent
// requests. Every request owns its own `KvCache` VMM reservation and its own
// `DeltaNetState` (S matrices + conv rings); the step addresses them through
// per-KV-layer `VmmKvSlotDesc` tables (absolute per-request VAs) and a
// slot-indexed DeltaNet table. No legacy arena, no `SlotPool`.
//
// Takeover: a request's `kv`/`dn` are the SAME owner types the singleton
// carrier route runs on. `Qwen35RequestState::swap_with_bundle` exchanges them
// with `Qwen35Bundle::{kv_cache, dn_state}` (plain moves, no GPU copy, no
// replay), so a lonely request continues on the unchanged singleton fast/spec
// route with its exact state, and the bundle's resident state becomes a batch
// participant without reconstruction. A request that is lonely from arrival
// never enters this route (it runs the unchanged singleton route, G2); a
// batch that drains to one live request may finish it here (tail steps).
//
// Stage-1 scope: AR and Prefill rows; one pick per request's last head row
// through the singleton route's own `sampler::sample` with that request's
// config/history/RNG. Verify/Forced rows are slice-2 work and are refused
// at provision, never silently treated as AR. VL rows (pos3/ext_emb) are
// refused. Native fp8 (gfx1201 default) and Q8 KV have `_vmm` kernels;
// every other KV mode is refused rather than routed to a legacy arena.

use super::{
    embed_lookup_slots, final_logits_per_slot, lm_head_slots_admissible,
    require_batchable_deltanet_layer, require_batchable_deltanet_moe_layer,
    require_batchable_fullattn_layer, require_batchable_fullattn_moe_layer,
    require_batchable_moe_ffn, run_layers_slots, slots_embed_format_admitted, LayerKvAddr,
    SlotKvTier,
};
use crate::carrier::Qwen35Bundle;
use crate::qwen35::LayerWeights;
use crate::qwen35::{
    DeltaNetState, LayerType, PrefillBatchScratch, Qwen35Config, Qwen35Scratch, Qwen35Weights,
};
use hip_bridge::{HipError, HipResult};
use hipfire_dispatch::pipeline::batched::q8_prefill_wmma_enabled;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::kv_mode::KvMode;
use hipfire_runtime::llama::{EmbeddingFormat, KvCache, KvCacheExt, KvDims, KvLayers, KvTarget};
use hipfire_runtime::sampler::SamplerConfig;
use hipfire_runtime::slot_batch::{
    BatchStepPlan, RequestAdvance, RequestEpoch, RequestStepKind, StepOutput,
};
use rdna_compute::attention::{VmmFlashDecodePlan, VmmKvFormat, VmmKvSide};
use rdna_compute::kv_slots::{validate_vmm_rows, VmmKvSlotDesc};
use rdna_compute::{DType, Gpu, GpuTensor};
pub mod dflash;
mod exact;
pub mod spec;

/// Load-time admission for the VMM executor on this resident model: the
/// exact predicates `forward_step` enforces (Q8_0 embedding table, an
/// admitted lm_head dtype, every layer batchable by the shared slots body,
/// a `_vmm` KV route and a VMM flash decode plan). Stage the store only when
/// this is `Ok`; otherwise the request path stays on the singleton route.
pub fn vmm_executor_supports(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    kv: &KvCache,
    route: VmmRoute,
) -> Result<(), String> {
    if route == VmmRoute::Exact {
        exact::supports(gpu, weights, config)?;
    }
    slots_embed_format_admitted(weights.embd_format).map_err(|e| format!("VMM executor: {e}"))?;
    if !lm_head_slots_admissible(weights.output.gpu_dtype) {
        return Err(format!(
            "VMM executor: lm_head dtype {:?} not admitted by the slots head",
            weights.output.gpu_dtype
        ));
    }
    let arch = gpu.arch.as_str();
    for (i, (layer, lt)) in weights.layers.iter().zip(&config.layer_types).enumerate() {
        let r = match (layer, lt) {
            (LayerWeights::DeltaNet(l), LayerType::LinearAttention) => {
                require_batchable_deltanet_layer(l, arch).map(drop)
            }
            (LayerWeights::FullAttn(l), LayerType::FullAttention) => {
                require_batchable_fullattn_layer(l, arch).map(drop)
            }
            (LayerWeights::DeltaNetMoe(l), LayerType::LinearAttention) => {
                require_batchable_deltanet_moe_layer(l, arch)
                    .and_then(|_| require_batchable_moe_ffn(gpu, &l.ffn))
            }
            (LayerWeights::FullAttnMoe(l), LayerType::FullAttention) => {
                require_batchable_fullattn_moe_layer(l, arch)
                    .and_then(|_| require_batchable_moe_ffn(gpu, &l.ffn))
            }
            _ => return Err(format!("VMM executor: layer {i} weight/type mismatch")),
        };
        r.map_err(|e| format!("VMM executor: layer {i}: {e}"))?;
    }
    let format = vmm_format_of(kv)?;
    gpu.vmm_flash_decode_plan(
        format,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        kv.max_seq,
        kv.vmm_logical_bound(),
    )
    .map(drop)
    .map_err(|e| format!("VMM executor: {e}"))
}

/// One FullAttention layer's per-request VMM addressing for a step.
pub struct VmmLayerKv<'a> {
    pub format: VmmKvFormat,
    /// Device `[max_slots]` `VmmKvSlotDesc` table for this KV layer.
    pub descs: &'a GpuTensor,
    /// Device `[rows]` i32 row → slot map.
    pub row_slot: &'a GpuTensor,
    /// Row-batched flash decode plan (tile size of the singleton route for
    /// this model bound) and its `[rows × partial_floats_per_row]` partials.
    pub flash: &'a VmmFlashDecodePlan,
    pub partials: &'a GpuTensor,
}

/// Device budget of the flash-decode partials buffer. Decode partials scale
/// with context, so a `rows × bound` buffer is tens of GB at 256K; the
/// attention instead runs in row groups that fit this fixed buffer.
const VMM_FLASH_PARTIALS_BYTES: usize = 256 << 20;
/// Device memory provision_step never maps KV into: the exact route's
/// prefill runs on the singleton scratch, which may widen after mapping.
const VMM_MAP_DEVICE_RESERVE_BYTES: usize = 512 << 20;

/// Rows past `prompt + max_tokens` a lane may still write: the last
/// speculative window (DFlash block, at most 63 rows; MTP `k + 1`) and the
/// finish drain.
pub const VMM_RESERVE_SLACK_ROWS: usize = 64;

/// Device bytes a VMM KV map can use: HIP's free figure plus the buffers the
/// pool parks (a VMM map returns them to HIP before mapping,
/// `Gpu::vmm_pool_headroom`), less [`VMM_MAP_DEVICE_RESERVE_BYTES`].
pub fn vmm_map_room(device_free: usize, pool_parked: usize) -> usize {
    device_free
        .saturating_add(pool_parked)
        .saturating_sub(VMM_MAP_DEVICE_RESERVE_BYTES)
}

/// Device bytes one admitted lane holds and may still map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VmmLaneNeed {
    /// Bytes held now (mapped KV, DeltaNet state, speculation state); all
    /// of it is freed when the lane completes.
    pub held: usize,
    /// The mapped-KV part of `held` (charged to the staged KV budget).
    pub held_kv: usize,
    /// KV bytes the lane may still map before it completes.
    pub remaining: usize,
}

/// Kind of lane an admission check charges: AR (no speculation state), a
/// DFlash lane or an MTP lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VmmLaneSpec {
    Ar,
    Dflash,
    Mtp,
}

/// Device bytes of a lane's speculation state: nothing for AR, the probed
/// MTP lane state for MTP, the probed DFlash lane state plus `checkpoints`
/// DeltaNet snapshots for DFlash.
pub fn vmm_spec_bytes(
    kind: VmmLaneSpec,
    mtp_lane_bytes: usize,
    dflash_lane_bytes: usize,
    snapshot_bytes: usize,
    checkpoints: usize,
) -> usize {
    match kind {
        VmmLaneSpec::Ar => 0,
        VmmLaneSpec::Mtp => mtp_lane_bytes,
        VmmLaneSpec::Dflash => dflash_lane_bytes + checkpoints * snapshot_bytes,
    }
}

/// Slack on top of a new lane's charged owners (prompt KV, DeltaNet state,
/// lane state) when predicting the free device bytes of its prompt seed's
/// chunk-ceiling query: VMM granule rounding, small scratch and any
/// allocation the charge does not itemize.
pub const VMM_SEED_CEILING_MARGIN_BYTES: usize = 256 << 20;

/// Free bytes the lane's own allocations take between its admission check
/// and its prompt seed's chunk-ceiling query: `fixed_held` (the lane's
/// charged bytes before checkpoints) plus the margin.
pub fn vmm_seed_free_deduction(fixed_held: usize) -> usize {
    fixed_held.saturating_add(VMM_SEED_CEILING_MARGIN_BYTES)
}

/// Banker's-algorithm safety of a set of lanes: some completion order
/// exists in which every lane's remaining KV growth fits the device room
/// and the KV budget, each completed lane returning what it holds. Lanes
/// in ascending `remaining` order are such an order whenever one exists
/// (a completion only ever adds room).
pub fn vmm_lanes_safe(room: usize, kv_room: usize, lanes: &[VmmLaneNeed]) -> bool {
    let mut order: Vec<&VmmLaneNeed> = lanes.iter().collect();
    order.sort_by_key(|l| l.remaining);
    let (mut room, mut kv_room) = (room, kv_room);
    for lane in order {
        if lane.remaining > room || lane.remaining > kv_room {
            return false;
        }
        // room - remaining + (held + remaining): the lane maps its rest,
        // completes and frees everything it then holds.
        room = room.saturating_add(lane.held);
        kv_room = kv_room.saturating_add(lane.held_kv);
    }
    true
}

/// Whether a lane may map `grow` more KV bytes now: it fits the room and
/// leaves the lanes safe ([`vmm_lanes_safe`]) with the growth applied to
/// `lanes[lane]`. On `true`, `room`, `kv_room` and `lanes[lane]` carry the
/// growth. A zero growth is always granted.
pub fn vmm_grant_growth(
    room: &mut usize,
    kv_room: &mut usize,
    lanes: &mut [VmmLaneNeed],
    lane: usize,
    grow: usize,
) -> bool {
    if grow == 0 {
        return true;
    }
    if grow > *room || grow > *kv_room {
        return false;
    }
    let before = lanes[lane];
    lanes[lane] = VmmLaneNeed {
        held: before.held + grow,
        held_kv: before.held_kv + grow,
        remaining: before.remaining.saturating_sub(grow),
    };
    if vmm_lanes_safe(*room - grow, *kv_room - grow, lanes) {
        *room -= grow;
        *kv_room -= grow;
        true
    } else {
        lanes[lane] = before;
        false
    }
}

/// Admission of a new lane: it can take what it holds right after its
/// prompt fill (`need.held`) and the lanes stay safe with it added.
pub fn vmm_admission_fits(
    room: usize,
    kv_room: usize,
    lanes: &[VmmLaneNeed],
    need: VmmLaneNeed,
) -> bool {
    if need.held > room || need.held_kv > kv_room {
        return false;
    }
    let mut all = lanes.to_vec();
    all.push(need);
    vmm_lanes_safe(room - need.held, kv_room - need.held_kv, &all)
}

/// Device memory view of the store at a step boundary: the room a KV map
/// can use, the staged KV budget left, and every admitted lane's need.
pub struct VmmMemoryState {
    pub room: usize,
    pub kv_room: usize,
    pub lanes: Vec<(RequestEpoch, VmmLaneNeed)>,
}

/// Admission verdict of [`Qwen35VmmStore::admission_check`].
#[derive(Clone, Copy, Debug)]
pub struct VmmAdmission {
    pub fits: bool,
    /// The new lane's need (fixed state + KV after its prompt, KV to its end).
    pub need: VmmLaneNeed,
    pub room: usize,
    pub kv_room: usize,
    /// Remaining KV growth summed over the admitted lanes.
    pub committed: usize,
}

/// Where the flash decode leaves each row's attention output.
pub(super) enum VmmAttendOut<'a> {
    /// The reduced, ungated output in `pbs.fa_attn_out_batch` (the caller
    /// applies the output gate and the `wo` rotation).
    Plain,
    /// The singleton's fused epilogue (A3B): reduce + sigmoid(`gate`) output
    /// gate + `wo` FWHT rotate, row r's rotated `wo` input written to row r of
    /// `rot` (`[rows, n_heads * head_dim]`). `gate` rows are the per-row gates
    /// of the FA prep (`pbs.fa_gate_batch`).
    GatedRotate {
        gate: &'a GpuTensor,
        rot: &'a GpuTensor,
    },
}

/// KV write (K and V) and causal attend over independent request owners.
pub(super) fn vmm_kv_write_attend(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    layer: &VmmLayerKv,
    pbs: &PrefillBatchScratch,
    n: usize,
) -> HipResult<()> {
    vmm_kv_write_attend_out(gpu, config, layer, pbs, n, &VmmAttendOut::Plain)
}

/// [`vmm_kv_write_attend`] with an explicit output mode.
pub(super) fn vmm_kv_write_attend_out(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    layer: &VmmLayerKv,
    pbs: &PrefillBatchScratch,
    n: usize,
    out: &VmmAttendOut,
) -> HipResult<()> {
    for (side, src) in [
        (VmmKvSide::K, &pbs.fa_k_batch),
        (VmmKvSide::V, &pbs.fa_v_batch),
    ] {
        gpu.kv_cache_write_batched_vmm(
            layer.format,
            side,
            src,
            &pbs.positions,
            config.n_kv_heads,
            config.head_dim,
            n,
            layer.descs,
            layer.row_slot,
        )?;
    }
    // Row-batched flash decode: bitwise the singleton flash decode per row,
    // causal over each row's own absolute position, no LDS context cap.
    // Rows are independent, so the step runs in row groups sized to the
    // fixed partials buffer (exact: no cross-row reduction exists).
    let per_row = layer.flash.partial_floats_per_row.max(1);
    let group = (layer.partials.numel() / per_row).max(1);
    let q_row = config.n_heads * config.head_dim;
    let mut r0 = 0usize;
    while r0 < n {
        let len = group.min(n - r0);
        let q = pbs.fa_q_batch.sub_offset(r0 * q_row, len * q_row);
        let positions = pbs.positions.sub_offset(r0, len);
        let row_slot = layer.row_slot.sub_offset(r0, len);
        match out {
            VmmAttendOut::Plain => gpu.attention_flash_decode_vmm(
                layer.flash,
                &q,
                &pbs.fa_attn_out_batch.sub_offset(r0 * q_row, len * q_row),
                &positions,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                len,
                layer.partials,
                layer.descs,
                &row_slot,
            )?,
            VmmAttendOut::GatedRotate { gate, rot } => gpu
                .attention_flash_decode_vmm_gated_mq_rotate(
                    layer.flash,
                    &q,
                    &gate.sub_offset(r0 * q_row, len * q_row),
                    &rot.sub_offset(r0 * q_row, len * q_row),
                    &positions,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    len,
                    layer.partials,
                    layer.descs,
                    &row_slot,
                )?,
        }
        r0 += len;
    }
    Ok(())
}

/// Map a resolved KV owner to its validated `_vmm` format, or refuse
/// (legacy arena, rotated/bf16 modes and compacted owners have none).
fn vmm_format_of(kv: &KvCache) -> Result<VmmKvFormat, String> {
    kv.vmm_kv_format()
        .map_err(|e| format!("VMM executor: request KV has no _vmm route: {e}"))
}

fn mode_of(format: VmmKvFormat) -> KvMode {
    match format {
        VmmKvFormat::Fp8E4M3 => KvMode::Fp8,
        VmmKvFormat::Q8 => KvMode::Q8,
    }
}

/// Private mutable state of one admitted request.
pub struct Qwen35RequestState {
    pub epoch: RequestEpoch,
    /// Scheduler slot (`SlotBatch` slot index) this request occupies.
    pub slot: usize,
    pub kv: KvCache,
    pub dn: DeltaNetState,
    /// Committed target watermark: next absolute write position.
    pub position: usize,
    /// Last committed id not yet fed through the trunk (the next AR row).
    pub pending_seed: Option<u32>,
    /// KV positions this request may map before it completes (prompt +
    /// output budget + [`VMM_RESERVE_SLACK_ROWS`]), set by the admitting
    /// driver; the memory scheduler reserves growth up to it. `0` reserves
    /// nothing past the current position.
    pub reserve_end: usize,
    /// Set when a participating forward failed: device state is not trusted.
    pub poisoned: bool,
    /// Total prompt rows; the Prefill chunk ending here carries the head
    /// and commits the first generated id.
    pub prompt_len: usize,
    /// Request-local GatedDeltaNet stochastic-rounding frame (used only
    /// when error feedback is off; the exact route swaps it in around this
    /// request's recurrence so its frame sequence is its own).
    pub gdn_frame: u32,
    /// Ids that finish the request with `finish = Some("stop")`.
    pub stop_ids: Vec<u32>,
    /// The singleton route's sampler config for this request. The driver
    /// may update policy fields (e.g. blocked tokens) between steps.
    pub sampler: SamplerConfig,
    /// RNG state, advanced only by a committed pick.
    pub rng_state: u32,
    /// Penalty history (the singleton route's sampling scope). The executor
    /// appends every committed id.
    pub history: Vec<u32>,
    /// MTP drafter state of a spec lane (`spec::spec_prefill`); `None` for
    /// an AR lane.
    pub mtp: Option<crate::mtp_spec::MtpSpecState>,
    /// Draft of a planned Verify step (provision → commit/abort).
    pub(super) spec_draft: Option<crate::mtp_spec::MtpDraftOutput>,
    /// Verify outcome of a forwarded Verify step (forward → commit/abort).
    pub(super) spec_verified: Option<crate::mtp_spec::cb::MtpCbVerified>,
    /// DFlash lane state (`dflash::dflash_prefill` / `dflash_adopt`); `None`
    /// unless the request is a DFlash spec lane. A request holds MTP
    /// (`mtp`) or DFlash, never both.
    pub dflash: Option<crate::dflash_cb::DflashVmmLaneState>,
    /// Draft of a planned DFlash Verify step (provision → commit/abort);
    /// the verify outcome waits in `dflash.picks`.
    pub(super) dflash_draft: Option<crate::speculative::DflashCbDraft>,
}

/// Admission inputs of one request, as the singleton route would hold them.
pub struct VmmRequestInit {
    pub prompt_len: usize,
    pub stop_ids: Vec<u32>,
    pub sampler: SamplerConfig,
    pub rng_state: u32,
    pub history: Vec<u32>,
}

/// Mapped-prefix bytes of one owner (mapped token capacity × K+V stride ×
/// KV layers). Recomputed from the owner, never accumulated, so swapped-in
/// singleton state is accounted exactly like a fresh request owner.
fn owner_mapped_bytes(kv: &KvCache, n_kv_layers: usize) -> Result<usize, String> {
    let format = vmm_format_of(kv)?;
    let tokens = kv
        .mapped_token_capacity()
        .map_err(|e| format!("VMM executor: mapped capacity: {e}"))?
        .unwrap_or(0);
    Ok(tokens * format.bytes_per_token(kv.n_kv_heads, kv.head_dim) * 2 * n_kv_layers)
}

/// Device bytes of a DeltaNet state (S matrices, scales, conv rings,
/// error-feedback residuals) as allocated.
fn dn_state_bytes(dn: &DeltaNetState) -> usize {
    dn.s_matrices
        .iter()
        .chain(&dn.s_scales)
        .chain(&dn.conv_states)
        .chain(&dn.s_ef_residual)
        .map(|t| t.buf.size())
        .sum()
}

impl Qwen35RequestState {
    /// Allocate a fresh request owner shaped exactly like the resident
    /// singleton `template` (same resolved KV mode, logical bound and
    /// physical cap; same DeltaNet quant). VMM reserves VA only; physical
    /// granules are mapped on demand by `provision_step`.
    pub fn new_like(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        template_kv: &KvCache,
        template_dn: &DeltaNetState,
        epoch: RequestEpoch,
        slot: usize,
        init: VmmRequestInit,
    ) -> Result<Self, String> {
        let format = vmm_format_of(template_kv)?;
        let is_kv_layer: Vec<bool> = config
            .layer_types
            .iter()
            .map(|t| *t == LayerType::FullAttention)
            .collect();
        let dims = KvDims {
            layers: KvLayers::Mask(is_kv_layer),
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            max_seq: template_kv.max_seq,
            physical_cap: Some(template_kv.physical_cap),
        };
        let kv = <KvCache as KvCacheExt>::from_mode_with_backend(
            mode_of(format),
            KvBackend::Vmm,
            KvTarget::Single(gpu),
            &dims,
        )
        .map_err(|e| format!("VMM executor: request KV ({:?}): {e}", mode_of(format)))?;
        let dn = match DeltaNetState::new_with_quant(gpu, config, template_dn.quant) {
            Ok(dn) => dn,
            Err(e) => {
                let cleanup = kv.release_vmm_after(gpu, None);
                return Err(format!(
                    "VMM executor: request DeltaNet state: {e}; kv free: {cleanup:?}"
                ));
            }
        };
        Ok(Self {
            epoch,
            slot,
            kv,
            dn,
            position: 0,
            pending_seed: None,
            reserve_end: 0,
            gdn_frame: rdna_compute::norm::gdn_requant_frame_checkpoint(),
            poisoned: false,
            prompt_len: init.prompt_len,
            stop_ids: init.stop_ids,
            sampler: init.sampler,
            rng_state: init.rng_state,
            history: init.history,
            mtp: None,
            spec_draft: None,
            spec_verified: None,
            dflash: None,
            dflash_draft: None,
        })
    }

    /// Exchange this request's owners with the resident singleton bundle's.
    /// Moves only; the bundle then runs the singleton route on this request's
    /// exact KV/DN, and this state holds what the bundle held. `position` and
    /// `pending_seed` are the caller's to exchange with its singleton frontier.
    pub fn swap_with_bundle(&mut self, bundle: &mut Qwen35Bundle) {
        std::mem::swap(&mut self.kv, &mut bundle.kv_cache);
        std::mem::swap(&mut self.dn, &mut bundle.dn_state);
    }

    /// Copy a running singleton's DeltaNet state into this state's own
    /// tensors (promotion into the batch; the caller then moves the KV
    /// owner with `mem::swap`). The singleton keeps its DN objects, so
    /// anything sized from them (the speculator's DN snapshots) stays valid;
    /// a fresh DN's pooled buffers need not match their sizes. Does not
    /// modify the singleton. Synchronizes before returning.
    pub fn copy_dn_from(&mut self, gpu: &mut Gpu, dn: &DeltaNetState) -> Result<(), String> {
        let pairs = self
            .dn
            .s_matrices
            .iter()
            .zip(&dn.s_matrices)
            .chain(self.dn.s_scales.iter().zip(&dn.s_scales))
            .chain(self.dn.conv_states.iter().zip(&dn.conv_states))
            .chain(self.dn.s_ef_residual.iter().zip(&dn.s_ef_residual));
        if self.dn.s_matrices.len() != dn.s_matrices.len()
            || self.dn.s_scales.len() != dn.s_scales.len()
            || self.dn.conv_states.len() != dn.conv_states.len()
            || self.dn.s_ef_residual.len() != dn.s_ef_residual.len()
        {
            return Err("copy_dn_from: DeltaNet layouts differ".into());
        }
        for (dst, src) in pairs {
            // Same constructor and shape on both sides; allocations may be
            // pool-rounded differently, so copy the common buffer extent
            // (it covers every element).
            if dst.shape != src.shape {
                return Err("copy_dn_from: DeltaNet tensor shapes differ".into());
            }
            let n = dst.buf.size().min(src.buf.size());
            gpu.memcpy_dtod_at_auto(&dst.buf, 0, &src.buf, 0, n)
                .map_err(|e| format!("copy_dn_from: {e}"))?;
        }
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("copy_dn_from: sync: {e}"))?;
        Ok(())
    }

    /// Free this owner. Every forward that read it has completed (the
    /// executor synchronizes before `forward_step` returns).
    pub fn free_gpu(self, gpu: &mut Gpu) -> Result<(), String> {
        let r = self
            .kv
            .release_vmm_after(gpu, None)
            .map_err(|e| e.to_string());
        self.dn.free_gpu(gpu);
        if let Some(mtp) = self.mtp {
            mtp.free_gpu(gpu);
        }
        if let Some(lane) = self.dflash {
            lane.free_gpu(gpu);
        }
        r
    }
}

/// A finished conversation kept for prefix reuse: the owner that served it
/// (KV + DeltaNet at `tokens.len()`, every token forwarded including the
/// ChatML trailer), its token stream and its DeltaNet resume checkpoints —
/// exactly what the singleton route keeps resident in the bundle
/// (`conversation_tokens`, `prefill_checkpoints`) after a turn.
pub struct VmmPrefixEntry {
    pub state: Qwen35RequestState,
    pub tokens: Vec<u32>,
    pub checkpoints: Vec<(usize, crate::speculative::DeltaNetSnapshot)>,
    /// Attempt whose client commit is outstanding; not reusable (and dropped
    /// on abort) until it commits.
    pub pending: Option<(String, u64)>,
    /// Recency stamp (higher = more recent) for eviction.
    pub stamp: u64,
}

impl VmmPrefixEntry {
    pub fn free_gpu(self, gpu: &mut Gpu) -> Result<(), String> {
        for (_, snap) in self.checkpoints {
            snap.free_gpu(gpu);
        }
        self.state.free_gpu(gpu)
    }
}

/// Slot-indexed DeltaNet view for the shared layer body.
struct DnTable<'a>(Vec<Option<&'a DeltaNetState>>);

impl std::ops::Index<usize> for DnTable<'_> {
    type Output = DeltaNetState;
    fn index(&self, s: usize) -> &DeltaNetState {
        self.0[s].expect("VMM executor: row addresses a slot with no request state")
    }
}

/// Plan identity recorded by provision/forward and checked by commit.
#[derive(Clone, PartialEq, Eq)]
struct StepKey {
    rows: Vec<(RequestEpoch, usize, usize, RequestStepKind)>,
    tokens: Vec<u32>,
    positions: Vec<i32>,
}

impl StepKey {
    fn of(plan: &BatchStepPlan) -> Self {
        Self {
            rows: plan
                .requests
                .iter()
                .map(|r| (r.epoch, r.rows.begin, r.rows.len, r.kind))
                .collect(),
            tokens: plan.batch.tokens.clone(),
            positions: plan.batch.positions.clone(),
        }
    }
}

enum Phase {
    Idle,
    Provisioned(StepKey),
    Forwarded(StepKey, u64),
}

/// Persistent executor owner: request table plus width/row-budget-sized
/// stable scratch. Never holds weights; `executor()` borrows them per step.
pub struct Qwen35VmmStore {
    max_slots: usize,
    row_budget: usize,
    format: VmmKvFormat,
    /// Model layer index of each FullAttention layer, in order.
    kv_layer_ids: Vec<usize>,
    slots: Vec<Option<Qwen35RequestState>>,
    /// One device `[max_slots]` descriptor table per KV layer (stable VAs:
    /// graph-safe, contents re-uploaded per step).
    descs_dev: Vec<GpuTensor>,
    descs_host: Vec<VmmKvSlotDesc>,
    /// Bytes last uploaded into each `descs_dev[l]` (`None` = unknown: never
    /// uploaded, or an upload failed part-way). `forward_inner` skips a
    /// layer's upload when its `descs_host` bytes equal this mirror
    /// (`HIPFIRE_CB_UPLOAD_COALESCE`). `descs_dev` is written only by that
    /// upload (kernels read it as `const`; it is allocated once in `new` and
    /// freed only with the store), so the mirror is exactly the device content.
    descs_uploaded: Vec<Option<Vec<u8>>>,
    row_slot_dev: GpuTensor,
    flash_partials: GpuTensor,
    /// Model-resolved max_seq: fixes the flash tile size to the singleton's.
    model_max_seq: usize,
    logits: GpuTensor,
    pbs: PrefillBatchScratch,
    phase: Phase,
    next_step_id: u64,
    // Reused host staging (no per-step allocation after warm-up).
    row_gen: Vec<u64>,
    row_slot_host: Vec<i32>,
    skip: Vec<bool>,
    /// RNG states produced by the forwarded step, published by commit.
    pending_rng: Vec<(RequestEpoch, u32)>,
    /// Batched head sampler (`hipfire_runtime::batch_sampler`): device result
    /// slots, packed penalty-scope slab and pinned stage. Allocated on the
    /// first sampled pass (needs the scratch's `repeat_buf` capacity, which
    /// `new` does not see); freed with the store.
    sampler_arena: Option<hipfire_runtime::batch_sampler::BatchSamplerArena>,
    /// Plan-ordered pre-norm hidden rows of a split (decode + prefill) step.
    hidden_out: GpuTensor,
    /// Exact route: device row tables of the rows-batched twins (one
    /// region per twin, per-layer for the stateful ones), staged per step.
    pub(super) rows_tables: Option<GpuTensor>,
    /// `(slot, epoch)` per row of the tables last uploaded: request owners
    /// and batch rows are stable, so an unchanged row set reuses them.
    pub(super) rows_tables_key: Vec<(usize, RequestEpoch)>,
    hidden_split: bool,
    /// Resolved logical context bound shared by every request owner.
    max_seq_bound: usize,
    mapped_high_water: usize,
    /// Shared physical KV budget across every request owner.
    kv_budget_bytes: usize,
    route: VmmRoute,
    /// MTP engine of spec lanes (`install_spec`); `None` = AR lanes only.
    spec: Option<spec::VmmSpecEngine>,
    /// Retained conversations for prefix reuse (VMM route continuity); the
    /// generate crate owns the policy. Freed with the store.
    pub prefix_pool: Vec<VmmPrefixEntry>,
    /// DFlash engine of DFlash spec lanes (`install_dflash`); `None` = no
    /// DFlash lanes. May coexist with the MTP engine: a request is one or
    /// the other.
    dflash_engine: Option<dflash::VmmDflashEngine>,
    /// Staged `serve.batch_sampled`: sampled (temperature>0) requests may be
    /// admitted to this store's lanes (set by the loader at staging; default
    /// false = greedy-only).
    sampled: bool,
}

/// Arithmetic of a VMM batched step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmmRoute {
    /// Byte-identical per request to the default (lowered) singleton route:
    /// batched exact plain projections, every other super-op is the
    /// singleton binding per request, prefill is the singleton prefill.
    Exact,
    /// The shared slots body (WMMA projections, batched prefill). Opt-in
    /// only (HIPFIRE_SERVE_BATCH_NONEXACT); never default.
    Nonexact,
}

/// Loaded-ack receipt of the actual per-request KV owners.
#[derive(Debug, Clone)]
pub struct VmmBatchReceipt {
    pub kv_backend: &'static str,
    pub kv_mode: String,
    pub max_seq_bound: usize,
    pub mapped_bytes: u64,
    pub mapped_high_water: u64,
}

impl Qwen35VmmStore {
    /// Which arithmetic this store runs: [`VmmRoute::Exact`] (byte-identical
    /// per request to the default singleton route) or the explicitly
    /// nonexact slots body.
    pub fn route(&self) -> VmmRoute {
        self.route
    }

    /// Whether the store was staged with `serve.batch_sampled` on (sampled
    /// requests admitted to the batch route for provably singleton-exact lane
    /// modes).
    pub fn sampled(&self) -> bool {
        self.sampled
    }

    /// Staging: record `serve.batch_sampled` (loader, before first use).
    pub fn set_sampled(&mut self, sampled: bool) {
        self.sampled = sampled;
    }

    /// The singleton route's prefill chunk length for `epoch` with
    /// `remaining` prompt rows. On the exact route every Prefill row range
    /// must have exactly this length (the planner uses it as the request's
    /// chunk maximum); chunk boundaries change the DeltaNet requant cadence.
    pub fn exact_prefill_chunk_len(
        &self,
        gpu: &Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        epoch: &RequestEpoch,
        remaining: usize,
    ) -> Result<usize, String> {
        let st = self
            .request_state(epoch)
            .ok_or_else(|| format!("exact_prefill_chunk_len: unknown epoch {epoch:?}"))?;
        exact::prefill_chunk_len(gpu, weights, config, st, remaining)
    }

    /// `template_kv` is the resident singleton KV owner; it fixes the
    /// format every request owner uses (no override of mode or bound).
    pub fn new(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        template_kv: &KvCache,
        max_slots: usize,
        row_budget: usize,
        kv_budget_bytes: usize,
        route: VmmRoute,
    ) -> Result<Self, String> {
        if max_slots < 2 || row_budget < max_slots {
            return Err(format!(
                "VMM executor: width {max_slots} / row budget {row_budget} invalid (width>=2, budget>=width)"
            ));
        }
        let format = vmm_format_of(template_kv)?;
        let kv_layer_ids: Vec<usize> = config
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == LayerType::FullAttention)
            .map(|(i, _)| i)
            .collect();
        // Size flash partials for the worst step: every row at the bound.
        let worst = gpu
            .vmm_flash_decode_plan(
                format,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                template_kv.max_seq,
                template_kv.vmm_logical_bound(),
            )
            .map_err(|e| format!("VMM executor: {e}"))?;
        // The exact route's trunk rows are decode rows only (one per slot):
        // its prefill is the singleton prefill on the singleton's own scratch.
        // Sizing these for prefill rows would duplicate the singleton's PBS.
        let trunk_rows = match route {
            VmmRoute::Exact => max_slots,
            VmmRoute::Nonexact => row_budget,
        };
        let singleton_rows = route == VmmRoute::Exact && gpu.arch_caps.is_gfx1100();
        let result =
            (|| -> HipResult<(Vec<GpuTensor>, GpuTensor, GpuTensor, GpuTensor, GpuTensor)> {
                let mut descs = Vec::with_capacity(kv_layer_ids.len());
                for _ in &kv_layer_ids {
                    descs.push(gpu.zeros(&[max_slots * 32], DType::Raw)?);
                }
                // 4-byte elements so per-row-group sub-views address rows.
                let row_slot = gpu.zeros(&[trunk_rows], DType::F32)?;
                // Fixed partials budget, never rows × context: the attention runs
                // in row groups that fit it. At least one row at the bound.
                // gfx1100 Exact attends through each lane's singleton scratch.
                // Keep only a placeholder for this unused batched-attention owner.
                let floats = if singleton_rows {
                    1
                } else {
                    (VMM_FLASH_PARTIALS_BYTES / 4)
                        .min(trunk_rows * worst.partial_floats_per_row)
                        .max(worst.partial_floats_per_row)
                };
                let partials = gpu.zeros(&[floats], DType::F32)?;
                let logits = gpu.zeros(&[max_slots * config.vocab_size], DType::F32)?;
                let hidden_out = gpu.zeros(&[trunk_rows * config.dim], DType::F32)?;
                Ok((descs, row_slot, partials, logits, hidden_out))
            })();
        let (descs_dev, row_slot_dev, flash_partials, logits, hidden_out) =
            result.map_err(|e| format!("VMM executor scratch: {e}"))?;
        // Plain trunk rows only: no GDN S-tape (that is a verify/tree cost).
        let pbs = match PrefillBatchScratch::new_opt(gpu, config, trunk_rows, false) {
            Ok(p) => p,
            Err(e) => {
                for t in descs_dev {
                    let _ = gpu.free_tensor(t);
                }
                let _ = gpu.free_tensor(row_slot_dev);
                let _ = gpu.free_tensor(flash_partials);
                let _ = gpu.free_tensor(logits);
                let _ = gpu.free_tensor(hidden_out);
                return Err(format!("VMM executor pbs: {e}"));
            }
        };
        let n_dn = config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count();
        // GDN (72 B) + conv (40 B) rows per DeltaNet layer; gated norm (24 B)
        // and FA prep (40 B) rows are layer-independent.
        let table_bytes = max_slots * (n_dn * (72 + 40) + 24 + 40);
        let rows_tables = match route {
            VmmRoute::Exact if !singleton_rows => match gpu.zeros(&[table_bytes / 4], DType::F32) {
                Ok(t) => Some(t),
                Err(e) => {
                    let _ = pbs.free_gpu(gpu);
                    for t in descs_dev {
                        let _ = gpu.free_tensor(t);
                    }
                    let _ = gpu.free_tensor(row_slot_dev);
                    let _ = gpu.free_tensor(flash_partials);
                    let _ = gpu.free_tensor(logits);
                    let _ = gpu.free_tensor(hidden_out);
                    return Err(format!("VMM executor rows tables: {e}"));
                }
            },
            VmmRoute::Exact | VmmRoute::Nonexact => None,
        };
        let n_kv = kv_layer_ids.len();
        Ok(Self {
            max_slots,
            row_budget,
            format,
            kv_layer_ids,
            slots: (0..max_slots).map(|_| None).collect(),
            descs_dev,
            descs_host: vec![VmmKvSlotDesc::MASKED; n_kv * max_slots],
            descs_uploaded: vec![None; n_kv],
            row_slot_dev,
            flash_partials,
            model_max_seq: template_kv.max_seq,
            logits,
            pbs,
            phase: Phase::Idle,
            next_step_id: 1,
            row_gen: Vec::with_capacity(row_budget),
            row_slot_host: Vec::with_capacity(row_budget),
            skip: vec![false; max_slots],
            pending_rng: Vec::with_capacity(max_slots),
            sampler_arena: None,
            hidden_out,
            rows_tables,
            rows_tables_key: Vec::new(),
            hidden_split: false,
            max_seq_bound: template_kv.vmm_logical_bound(),
            mapped_high_water: 0,
            kv_budget_bytes,
            route,
            spec: None,
            prefix_pool: Vec::new(),
            dflash_engine: None,
            sampled: false,
        })
    }

    pub fn max_slots(&self) -> usize {
        self.max_slots
    }

    pub fn row_budget(&self) -> usize {
        self.row_budget
    }

    pub fn format(&self) -> VmmKvFormat {
        self.format
    }

    /// Number of admitted requests.
    pub fn occupancy(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Admit a request owner into its slot. Refused while a step is in
    /// flight, for an unadmitted epoch, an occupied slot, a duplicate epoch
    /// or a KV owner of another format.
    pub fn admit(&mut self, state: Qwen35RequestState) -> Result<(), (Qwen35RequestState, String)> {
        if !matches!(self.phase, Phase::Idle) {
            return Err((
                state,
                "VMM executor: admit during an uncommitted step".into(),
            ));
        }
        if !state.epoch.is_admitted() {
            return Err((state, "VMM executor: admit of an unadmitted epoch".into()));
        }
        if state.slot >= self.max_slots {
            let m = format!(
                "VMM executor: slot {} >= width {}",
                state.slot, self.max_slots
            );
            return Err((state, m));
        }
        if self.slots[state.slot].is_some() {
            let m = format!("VMM executor: slot {} occupied", state.slot);
            return Err((state, m));
        }
        if self.slots.iter().flatten().any(|s| s.epoch == state.epoch) {
            return Err((state, "VMM executor: duplicate epoch".into()));
        }
        match vmm_format_of(&state.kv) {
            Ok(f) if f == self.format => {}
            Ok(_) => {
                return Err((
                    state,
                    "VMM executor: request KV format differs from store".into(),
                ))
            }
            Err(e) => return Err((state, e)),
        }
        let slot = state.slot;
        self.slots[slot] = Some(state);
        Ok(())
    }

    /// Release a request owner (finish, cancel, poison or singleton
    /// takeover). Refused while a step is in flight: its leases are still
    /// referenced by uploaded descriptors until commit/abort.
    pub fn retire(&mut self, epoch: &RequestEpoch) -> Result<Qwen35RequestState, String> {
        if !matches!(self.phase, Phase::Idle) {
            return Err("VMM executor: retire during an uncommitted step; abort_step first".into());
        }
        let slot = self
            .slots
            .iter()
            .position(|s| s.as_ref().is_some_and(|s| s.epoch == *epoch))
            .ok_or_else(|| format!("VMM executor: retire of unknown epoch {epoch:?}"))?;
        for l in 0..self.kv_layer_ids.len() {
            self.descs_host[l * self.max_slots + slot] = VmmKvSlotDesc::MASKED;
        }
        Ok(self.slots[slot].take().expect("slot located above"))
    }

    pub fn request_state(&self, epoch: &RequestEpoch) -> Option<&Qwen35RequestState> {
        self.slots.iter().flatten().find(|s| s.epoch == *epoch)
    }

    pub fn request_state_mut(&mut self, epoch: &RequestEpoch) -> Option<&mut Qwen35RequestState> {
        self.slots.iter_mut().flatten().find(|s| s.epoch == *epoch)
    }

    /// Abandon a provisioned/forwarded step. A forwarded step already wrote
    /// device state, which is not atomic: every participant is poisoned and
    /// must be retired; its tokens are never committed. Verify drafts and
    /// outcomes of the plan are dropped (a provisioned-only draft touched
    /// no committed state, so its lane stays live).
    pub fn abort_step(&mut self, plan: &BatchStepPlan) {
        if let Phase::Forwarded(..) = self.phase {
            self.poison(plan);
        }
        self.spec_clear_planned(plan);
        self.phase = Phase::Idle;
    }

    fn poison(&mut self, plan: &BatchStepPlan) {
        for r in &plan.requests {
            if let Some(s) = self.request_state_mut(&r.epoch) {
                s.poisoned = true;
            }
        }
    }

    /// Pre-final-norm residual rows `[rows x dim]` f32 of the last forward
    /// (`pbs.x_batch`), valid until the next `forward_step`.
    pub fn hidden(&self) -> &GpuTensor {
        if self.hidden_split {
            &self.hidden_out
        } else {
            &self.pbs.x_batch
        }
    }

    /// Per-slot last-row logits `[max_slots x vocab]` f32 of the last
    /// forward; row `s` valid iff slot `s` had an AR row.
    pub fn logits(&self) -> &GpuTensor {
        &self.logits
    }

    /// Loaded-ack receipt of the actual per-request VMM owners: physically
    /// mapped bytes (reserved VA is not counted) and their high-water mark.
    pub fn receipt(&mut self) -> Result<VmmBatchReceipt, String> {
        let mapped = self.mapped_kv_bytes()?;
        self.mapped_high_water = self.mapped_high_water.max(mapped);
        Ok(VmmBatchReceipt {
            kv_backend: "vmm",
            kv_mode: format!("{:?}", mode_of(self.format)),
            max_seq_bound: self.max_seq_bound,
            mapped_bytes: mapped as u64,
            mapped_high_water: self.mapped_high_water as u64,
        })
    }

    /// Does this planned request carry a head (target pick) this step?
    /// AR rows always; a Prefill chunk only when it completes the prompt.
    fn wants_head(&self, r: &hipfire_runtime::slot_batch::RequestRows) -> bool {
        match r.kind {
            RequestStepKind::Ar => true,
            RequestStepKind::Prefill => self
                .request_state(&r.epoch)
                .is_some_and(|s| s.position + r.rows.len == s.prompt_len),
            _ => false,
        }
    }

    /// Physically mapped KV bytes summed over admitted request owners and
    /// retained prefix entries.
    pub fn mapped_kv_bytes(&self) -> Result<usize, String> {
        let n_kv = self.kv_layer_ids.len();
        let mut total = 0usize;
        for s in self.slots.iter().flatten() {
            total += owner_mapped_bytes(&s.kv, n_kv)?;
        }
        for e in &self.prefix_pool {
            total += owner_mapped_bytes(&e.state.kv, n_kv)?;
        }
        Ok(total)
    }

    /// What `st` holds on the device and may still map. Held: mapped KV,
    /// DeltaNet state, and a DFlash lane's state at the engine's probed
    /// footprint plus its checkpoints, or an MTP lane's state at the MTP
    /// engine's probed footprint. Remaining: KV growth up to
    /// `reserve_end` (never below the current position).
    fn lane_need(&self, gpu: &Gpu, st: &Qwen35RequestState) -> Result<VmmLaneNeed, String> {
        let held_kv = owner_mapped_bytes(&st.kv, self.kv_layer_ids.len())?;
        let spec = if let (Some(lane), Some(engine)) = (&st.dflash, &self.dflash_engine) {
            vmm_spec_bytes(
                VmmLaneSpec::Dflash,
                0,
                engine.lane_bytes(),
                engine.snapshot_bytes(),
                lane.checkpoints.len(),
            )
        } else if let (Some(_), Some(engine)) = (&st.mtp, &self.spec) {
            vmm_spec_bytes(VmmLaneSpec::Mtp, engine.lane_bytes(), 0, 0, 0)
        } else {
            0
        };
        let end = st
            .reserve_end
            .max(st.position)
            .min(st.kv.vmm_logical_bound());
        let remaining = st
            .kv
            .planned_mapped_growth_bytes(gpu, end)
            .map_err(|e| format!("VMM memory: planned growth of {:?}: {e}", st.epoch))?;
        Ok(VmmLaneNeed {
            held: held_kv + dn_state_bytes(&st.dn) + spec,
            held_kv,
            remaining,
        })
    }

    /// The store's memory view now (see [`VmmMemoryState`]). Retained
    /// prefix entries are outside the room and never complete, so they are
    /// not lanes.
    pub fn memory_state(&self, gpu: &Gpu) -> Result<VmmMemoryState, String> {
        let free = gpu
            .device_mem_info()
            .map_err(|e| format!("VMM memory: VRAM query: {e}"))?
            .0;
        let lanes = self
            .slots
            .iter()
            .flatten()
            .map(|st| Ok((st.epoch, self.lane_need(gpu, st)?)))
            .collect::<Result<Vec<_>, String>>()?;
        Ok(VmmMemoryState {
            room: vmm_map_room(free, gpu.pool_parked_bytes()),
            kv_room: self.kv_budget_bytes.saturating_sub(self.mapped_kv_bytes()?),
            lanes,
        })
    }

    /// Admission of a new request owner shaped like `template_kv` /
    /// `template_dn` (the resident singleton's) with `prompt_len` prompt
    /// rows that may grow to `reserve_end` positions, as a lane of kind
    /// `lane` (a DFlash lane is charged its lane state and a checkpoint
    /// bound, an MTP lane its probed MTP state). Fits when the room holds the
    /// lane after its prompt fill and every lane, the new one included, can
    /// still complete ([`vmm_admission_fits`]).
    ///
    /// DFlash checkpoint bound. The lane's prompt seed takes its checkpoints
    /// at the ends of outer chunks whose ceiling is memory-admitted at the
    /// seed's plan time (`dflash_seed_ceiling`), so the old
    /// `ceil(prompt/interval)` over-charges. Here the ceiling is queried now
    /// with `free_deduction` = the bytes charged for the lane before its
    /// checkpoints (prompt KV, DeltaNet state, lane state) plus
    /// [`VMM_SEED_CEILING_MARGIN_BYTES`]. Soundness: between this check and
    /// the seed's plan the driver allocates exactly those owners
    /// (`new_like`, KV provision, lane state; synchronous), so device free
    /// at plan time is at least free now minus the deduction; the byte
    /// admission is monotone in free bytes, so the actual ceiling is at least
    /// the queried one (`min_ceiling`) and, being a staged rung or
    /// `min_ceiling` itself, one of the candidates the bound maximizes over
    /// (the checkpoint count is not monotone in the ceiling). A failed
    /// ceiling query falls back to the whole-prompt bound.
    #[allow(clippy::too_many_arguments)]
    pub fn admission_check(
        &self,
        gpu: &Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        template_kv: &KvCache,
        template_dn: &DeltaNetState,
        prompt_len: usize,
        reserve_end: usize,
        lane: VmmLaneSpec,
    ) -> Result<VmmAdmission, String> {
        let bound = template_kv.vmm_logical_bound();
        let planned = |positions: usize| {
            template_kv
                .planned_mapped_bytes_from_empty(gpu, positions.min(bound))
                .map_err(|e| format!("VMM admission: planned KV for {positions} positions: {e}"))
        };
        let kv_prompt = planned(prompt_len)?;
        let kv_end = planned(reserve_end.max(prompt_len))?;
        let dn = dn_state_bytes(template_dn);
        let spec = match lane {
            VmmLaneSpec::Ar => 0,
            VmmLaneSpec::Mtp => self.spec.as_ref().map_or(0, |e| {
                vmm_spec_bytes(VmmLaneSpec::Mtp, e.lane_bytes(), 0, 0, 0)
            }),
            VmmLaneSpec::Dflash => match &self.dflash_engine {
                None => 0,
                Some(engine) => {
                    let lane_state = engine.lane_bytes();
                    let deduction = vmm_seed_free_deduction(kv_prompt + dn + lane_state);
                    let bound = match crate::speculative::dflash_seed_ceiling(
                        gpu,
                        weights,
                        config,
                        template_dn,
                        template_kv,
                        deduction,
                    ) {
                        Ok(min_ceiling) => engine.checkpoint_bound(prompt_len, min_ceiling),
                        Err(_) => engine.checkpoint_bound_unplanned(prompt_len),
                    };
                    vmm_spec_bytes(
                        VmmLaneSpec::Dflash,
                        0,
                        lane_state,
                        engine.snapshot_bytes(),
                        bound,
                    )
                }
            },
        };
        let need = VmmLaneNeed {
            held: kv_prompt + dn + spec,
            held_kv: kv_prompt,
            remaining: kv_end - kv_prompt,
        };
        let mem = self.memory_state(gpu)?;
        let lanes: Vec<VmmLaneNeed> = mem.lanes.iter().map(|(_, l)| *l).collect();
        Ok(VmmAdmission {
            fits: vmm_admission_fits(mem.room, mem.kv_room, &lanes, need),
            need,
            room: mem.room,
            kv_room: mem.kv_room,
            committed: lanes.iter().map(|l| l.remaining).sum(),
        })
    }

    /// Per-step KV growth grants. `granted` are `(epoch, end)` requests
    /// already granted this step (applied first, unconditionally);
    /// `candidates` are granted in order by [`vmm_grant_growth`]: a request
    /// whose mapping to `end` positions would leave the lanes unable to
    /// complete waits (`false`) with its state untouched, as a lane that
    /// does not fit the row budget does. Unknown epochs are not granted.
    pub fn step_grants(
        &self,
        gpu: &Gpu,
        granted: &[(RequestEpoch, usize)],
        candidates: &[(RequestEpoch, usize)],
    ) -> Result<Vec<bool>, String> {
        let VmmMemoryState {
            mut room,
            mut kv_room,
            lanes,
        } = self.memory_state(gpu)?;
        let epochs: Vec<RequestEpoch> = lanes.iter().map(|(e, _)| *e).collect();
        let mut needs: Vec<VmmLaneNeed> = lanes.into_iter().map(|(_, l)| l).collect();
        let growth = |epoch: &RequestEpoch, end: usize| -> Result<Option<(usize, usize)>, String> {
            let Some(i) = epochs.iter().position(|e| e == epoch) else {
                return Ok(None);
            };
            let st = self.request_state(epoch).expect("epoch listed above");
            let g = st
                .kv
                .planned_mapped_growth_bytes(gpu, end.min(st.kv.vmm_logical_bound()))
                .map_err(|e| format!("VMM memory: planned growth of {epoch:?}: {e}"))?;
            Ok(Some((i, g)))
        };
        for (epoch, end) in granted {
            if let Some((i, g)) = growth(epoch, *end)? {
                room = room.saturating_sub(g);
                kv_room = kv_room.saturating_sub(g);
                let n = needs[i];
                needs[i] = VmmLaneNeed {
                    held: n.held + g,
                    held_kv: n.held_kv + g,
                    remaining: n.remaining.saturating_sub(g),
                };
            }
        }
        candidates
            .iter()
            .map(|(epoch, end)| {
                Ok(match growth(epoch, *end)? {
                    Some((i, g)) => vmm_grant_growth(&mut room, &mut kv_room, &mut needs, i, g),
                    None => false,
                })
            })
            .collect()
    }

    /// One-line itemization of the staged KV budget and of what every
    /// admitted lane and retained prefix entry holds on the device (mapped
    /// KV, DeltaNet state, DFlash checkpoints, MTP/DFlash presence, reserved
    /// end), plus device free and the pool's parked bytes. Logged with a
    /// provision refusal.
    pub fn budget_report(&self, gpu: &Gpu) -> String {
        let n_kv = self.kv_layer_ids.len();
        let free = gpu.device_mem_info().map_or(0, |(f, _)| f);
        let (lane_bytes, snapshot_bytes) = self
            .dflash_engine
            .as_ref()
            .map_or((0, 0), |e| (e.lane_bytes(), e.snapshot_bytes()));
        let mtp_lane_bytes = self.spec.as_ref().map_or(0, |e| e.lane_bytes());
        let mut s = format!(
            "kv_budget={} mapped_kv={} device_free={free} pool_parked={} map_reserve={VMM_MAP_DEVICE_RESERVE_BYTES} dflash_lane_bytes={lane_bytes} mtp_lane_bytes={mtp_lane_bytes} dn_snapshot_bytes={snapshot_bytes} lanes=[",
            self.kv_budget_bytes,
            self.mapped_kv_bytes().unwrap_or(usize::MAX),
            gpu.pool_parked_bytes(),
        );
        for st in self.slots.iter().flatten() {
            s += &format!(
                " slot{}:pos={} reserve_end={} kv={} dn={} dflash={} dflash_ckpt={} mtp={};",
                st.slot,
                st.position,
                st.reserve_end,
                owner_mapped_bytes(&st.kv, n_kv).unwrap_or(usize::MAX),
                dn_state_bytes(&st.dn),
                st.dflash.is_some(),
                st.dflash.as_ref().map_or(0, |d| d.checkpoints.len()),
                st.mtp.is_some(),
            );
        }
        s += &format!(" ] prefix_pool={}:[", self.prefix_pool.len());
        for e in &self.prefix_pool {
            s += &format!(
                " kv={} dn={} ckpt={};",
                owner_mapped_bytes(&e.state.kv, n_kv).unwrap_or(usize::MAX),
                dn_state_bytes(&e.state.dn),
                e.checkpoints.len()
            );
        }
        s + " ]"
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> Result<(), String> {
        let mut first: Option<String> = None;
        for s in self.slots.into_iter().flatten() {
            if let Err(e) = s.free_gpu(gpu) {
                first.get_or_insert(e);
            }
        }
        for e in self.prefix_pool {
            if let Err(e) = e.free_gpu(gpu) {
                first.get_or_insert(e);
            }
        }
        for t in self.descs_dev {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.free_tensor(self.row_slot_dev);
        let _ = gpu.free_tensor(self.flash_partials);
        let _ = gpu.free_tensor(self.logits);
        let _ = gpu.free_tensor(self.hidden_out);
        if let Some(t) = self.rows_tables {
            let _ = gpu.free_tensor(t);
        }
        if let Some(arena) = self.sampler_arena {
            arena.free_gpu(gpu);
        }
        if let Some(engine) = self.spec {
            engine.free_gpu(gpu);
        }
        if let Some(engine) = self.dflash_engine {
            engine.free_gpu(gpu);
        }
        if let Err(e) = self.pbs.free_gpu(gpu) {
            first.get_or_insert(e.to_string());
        }
        first.map_or(Ok(()), Err)
    }

    /// Borrow the resident weights for this step's executor methods.
    pub fn executor<'a>(
        &'a mut self,
        weights: &'a Qwen35Weights,
        config: &'a Qwen35Config,
        scratch: &'a Qwen35Scratch,
    ) -> Qwen35VmmExecutor<'a> {
        Qwen35VmmExecutor {
            store: self,
            weights,
            config,
            scratch,
        }
    }
}

/// Per-step borrow view exposing the frozen §4.2 executor methods.
pub struct Qwen35VmmExecutor<'a> {
    store: &'a mut Qwen35VmmStore,
    weights: &'a Qwen35Weights,
    config: &'a Qwen35Config,
    scratch: &'a Qwen35Scratch,
}

impl Qwen35VmmExecutor<'_> {
    /// Validate the plan against the owner table and map every KV granule
    /// its rows will write. Mapping/growth happens here, never inside a
    /// forward or capture. Fails closed per request bound (no OOM midstep).
    pub fn provision_step(&mut self, gpu: &mut Gpu, plan: &BatchStepPlan) -> Result<(), String> {
        let (weights, config) = (self.weights, self.config);
        let st = &mut *self.store;
        if !matches!(st.phase, Phase::Idle) {
            return Err("provision_step: previous step not committed or aborted".into());
        }
        let b = &plan.batch;
        let n = b.total_rows();
        if n == 0 || plan.requests.is_empty() {
            return Err("provision_step: empty plan".into());
        }
        // Exact-route prefill rows run on the singleton's scratch (chunk
        // length checked below); only its decode rows use store scratch.
        let trunk_rows = match st.route {
            VmmRoute::Exact => plan
                .requests
                .iter()
                .filter(|r| r.kind != RequestStepKind::Prefill)
                .map(|r| r.rows.len)
                .sum(),
            VmmRoute::Nonexact => n,
        };
        if trunk_rows > st.row_budget {
            return Err(format!(
                "provision_step: {trunk_rows} rows exceed row budget {}",
                st.row_budget
            ));
        }
        if b.m_per_slot.len() > st.max_slots
            || b.tokens.len() != n
            || b.positions.len() != n
            || b.row_slot.len() != n
        {
            return Err("provision_step: malformed SlotBatch".into());
        }
        if !b.pos3.is_empty() || b.ext_emb.iter().any(|&e| e >= 0) {
            return Err("provision_step: VL rows are not admitted on the VMM executor".into());
        }
        let mut covered = vec![false; n];
        let mut seen_slots = vec![false; st.max_slots];
        for r in &plan.requests {
            match r.kind {
                RequestStepKind::Ar | RequestStepKind::Prefill => {}
                RequestStepKind::Verify { draft_len } => {
                    let seed = b.tokens.get(r.rows.begin).copied().unwrap_or(u32::MAX);
                    st.spec_check_verify(r, draft_len, seed)?;
                }
                RequestStepKind::Forced => {
                    return Err(
                        "provision_step: Forced rows are not admitted on the VMM executor".into(),
                    );
                }
            }
            let state = st
                .slots
                .iter()
                .flatten()
                .find(|s| s.epoch == r.epoch)
                .ok_or_else(|| format!("provision_step: stale or unknown epoch {:?}", r.epoch))?;
            if state.poisoned {
                return Err(format!("provision_step: epoch {:?} is poisoned", r.epoch));
            }
            if r.rows.len == 0 || r.rows.begin + r.rows.len > n {
                return Err(format!("provision_step: bad row range {:?}", r.rows));
            }
            if seen_slots[state.slot] {
                return Err(format!("provision_step: slot {} planned twice", state.slot));
            }
            seen_slots[state.slot] = true;
            if b.m_per_slot.get(state.slot).copied() != Some(r.rows.len) {
                return Err(format!(
                    "provision_step: m_per_slot[{}] disagrees with row range len {}",
                    state.slot, r.rows.len
                ));
            }
            if r.kind == RequestStepKind::Prefill && state.position + r.rows.len > state.prompt_len
            {
                return Err(format!(
                    "provision_step: prefill past prompt end {} for {:?}",
                    state.prompt_len, r.epoch
                ));
            }
            if st.route == VmmRoute::Exact && r.kind == RequestStepKind::Prefill {
                let want = exact::prefill_chunk_len(
                    gpu,
                    self.weights,
                    self.config,
                    state,
                    state.prompt_len - state.position,
                )?;
                if r.rows.len != want {
                    return Err(format!(
                        "provision_step: exact route prefill chunk for {:?} is {} rows, singleton route uses {want}",
                        r.epoch, r.rows.len
                    ));
                }
            }
            if r.kind == RequestStepKind::Ar && r.rows.len != 1 {
                return Err("provision_step: AR request must contribute one row".into());
            }
            for (j, row) in (r.rows.begin..r.rows.begin + r.rows.len).enumerate() {
                if covered[row] {
                    return Err(format!("provision_step: row {row} in two requests"));
                }
                covered[row] = true;
                if b.row_slot[row] != state.slot as i32 {
                    return Err(format!(
                        "provision_step: row {row} row_slot {} != owner slot {}",
                        b.row_slot[row], state.slot
                    ));
                }
                if b.positions[row] as i64 != (state.position + j) as i64 {
                    return Err(format!(
                        "provision_step: row {row} position {} != committed frontier {}",
                        b.positions[row],
                        state.position + j
                    ));
                }
            }
            let end = state.position + r.rows.len;
            if end > state.kv.vmm_logical_bound() {
                return Err(format!(
                    "provision_step: epoch {:?} needs {end} positions > logical bound {}",
                    r.epoch,
                    state.kv.vmm_logical_bound()
                ));
            }
        }
        if covered.iter().any(|c| !c) {
            return Err("provision_step: rows outside every request range".into());
        }
        // Map granules (outside any forward/capture) against the shared
        // physical budget AND the device memory a map can use (HIP free plus
        // the pool's parked buffers, which the map returns to HIP first,
        // less a reserve for the singleton prefill scratch the exact route
        // widens), so an overstated budget is a typed refusal, never a raw
        // HIP OOM. Refusal maps nothing for that request.
        let mut mapped_total = self.store.mapped_kv_bytes()?;
        let mut device_room = vmm_map_room(
            gpu.device_mem_info()
                .map_err(|e| format!("provision_step: VRAM query: {e}"))?
                .0,
            gpu.pool_parked_bytes(),
        );
        let st = &mut *self.store;
        for r in &plan.requests {
            let budget = st
                .kv_budget_bytes
                .saturating_sub(mapped_total)
                .min(device_room);
            let s = st
                .slots
                .iter_mut()
                .flatten()
                .find(|s| s.epoch == r.epoch)
                .expect("validated above");
            let end = s.position + r.rows.len;
            let grown = match s.kv.provision_vmm_positions(gpu, end, budget) {
                Ok(g) => g,
                Err(e) => {
                    let msg = format!("provision_step: map {end} positions for {:?}: {e}", r.epoch);
                    eprintln!("[vmm-budget] refused: {msg}; {}", st.budget_report(gpu));
                    return Err(msg);
                }
            };
            mapped_total += grown;
            device_room = device_room.saturating_sub(grown);
        }
        st.mapped_high_water = st.mapped_high_water.max(mapped_total);
        // Build descriptors (re-read after provision: views borrow) and the
        // per-row owner generation of the PLANNED request, then run the host
        // gate: a row whose row_slot points at another owner, or a stale
        // owner after abort/reuse, fails before any launch.
        let ms = st.max_slots;
        for d in st.descs_host.iter_mut() {
            *d = VmmKvSlotDesc::MASKED;
        }
        st.row_gen.clear();
        st.row_gen.resize(n, 0);
        for r in &plan.requests {
            let s = st
                .slots
                .iter()
                .flatten()
                .find(|s| s.epoch == r.epoch)
                .expect("validated above");
            let generation =
                s.kv.vmm_owner_generation(gpu)
                    .map_err(|e| format!("provision_step: {:?}: {e}", r.epoch))?;
            st.row_gen[r.rows.begin..r.rows.begin + r.rows.len].fill(generation);
            let descs =
                s.kv.vmm_slot_descs(gpu)
                    .map_err(|e| format!("provision_step: descriptors for {:?}: {e}", r.epoch))?;
            for (l, &layer) in st.kv_layer_ids.iter().enumerate() {
                // Idle requests stay MASKED: never fake participants.
                st.descs_host[l * ms + s.slot] = descs[layer];
            }
        }
        for l in 0..st.kv_layer_ids.len() {
            let table = &st.descs_host[l * ms..(l + 1) * ms];
            validate_vmm_rows(table, &b.row_slot, &b.positions, &st.row_gen)
                .map_err(|e| format!("provision_step: layer {}: {e}", st.kv_layer_ids[l]))?;
        }
        // Verify requests: their epoch-tagged drafts (singleton drafter).
        st.spec_draft_planned(gpu, weights, config, plan)?;
        st.phase = Phase::Provisioned(StepKey::of(plan));
        Ok(())
    }

    /// The step's trunk forward(s), then one pick per head request. AR rows
    /// and prefill rows run as two separate passes when both are present:
    /// the projection kernels select their arithmetic by row count, so a
    /// decode row co-batched with a prefill chunk would take the large-M
    /// tile route and its result would depend on another request's prompt.
    /// Waits for completion (the pick download synchronizes) so commit can
    /// release leases safely. On error the step is aborted and every
    /// participant poisoned.
    pub fn forward_step(
        &mut self,
        gpu: &mut Gpu,
        plan: &BatchStepPlan,
    ) -> Result<StepOutput, String> {
        let key = StepKey::of(plan);
        match &self.store.phase {
            Phase::Provisioned(k) if *k == key => {}
            _ => return Err("forward_step: plan was not provisioned".into()),
        }
        let step_id = self.store.next_step_id;
        self.store.phase = Phase::Forwarded(key, step_id);
        match self.forward_inner(gpu, plan) {
            Ok(picks) => {
                self.store.next_step_id += 1;
                Ok(StepOutput {
                    step_id,
                    target_picks: picks,
                })
            }
            Err(e) => {
                self.store.abort_step(plan);
                Err(format!("forward_step: {e}"))
            }
        }
    }

    fn forward_inner(&mut self, gpu: &mut Gpu, plan: &BatchStepPlan) -> HipResult<Vec<u32>> {
        let (weights, config, s) = (self.weights, self.config, self.scratch);
        let st = &mut *self.store;
        slots_embed_format_admitted(weights.embd_format)?;
        let ms = st.max_slots;
        let coalesce = crate::qwen35::prefill::multi::cb_upload_coalesce_enabled();
        for (l, dev) in st.descs_dev.iter().enumerate() {
            let bytes = VmmKvSlotDesc::as_bytes(&st.descs_host[l * ms..(l + 1) * ms]);
            // A layer whose table bytes equal what the device already holds
            // would upload identical bytes: skipping it leaves the same device
            // content before every attention kernel. Unknown mirror → upload.
            if coalesce && st.descs_uploaded[l].as_deref() == Some(bytes) {
                continue;
            }
            // Unknown until the copy returns Ok (a failed copy may be partial).
            let mirror = st.descs_uploaded[l].take();
            gpu.hip.memcpy_htod(&dev.buf, bytes)?;
            if coalesce {
                let mut m = mirror.unwrap_or_default();
                m.clear();
                m.extend_from_slice(bytes);
                st.descs_uploaded[l] = Some(m);
            }
        }
        let decode: Vec<usize> = (0..plan.requests.len())
            .filter(|&i| plan.requests[i].kind == RequestStepKind::Ar)
            .collect();
        let is_verify = |i: usize| matches!(plan.requests[i].kind, RequestStepKind::Verify { .. });
        let prefill: Vec<usize> = (0..plan.requests.len())
            .filter(|&i| plan.requests[i].kind != RequestStepKind::Ar && !is_verify(i))
            .collect();
        let split = !decode.is_empty() && !prefill.is_empty();
        st.hidden_split = split;
        st.pending_rng.clear();
        let mut picks = vec![u32::MAX; plan.total_rows()];
        if st.route == VmmRoute::Exact {
            st.hidden_split = false;
            exact::decode(gpu, st, weights, config, s, plan, &decode, &mut picks)?;
            for &i in &prefill {
                exact::prefill(gpu, st, weights, config, s, plan, i)?;
            }
            sample_heads(gpu, st, config, s, plan, &prefill, &mut picks)?;
        } else if split {
            run_pass(gpu, st, weights, config, s, plan, &decode, true, &mut picks)?;
            run_pass(
                gpu, st, weights, config, s, plan, &prefill, true, &mut picks,
            )?;
        } else if !decode.is_empty() || !prefill.is_empty() {
            let all: Vec<usize> = (0..plan.requests.len())
                .filter(|&i| !is_verify(i))
                .collect();
            run_pass(gpu, st, weights, config, s, plan, &all, false, &mut picks)?;
        }
        // Verify requests: their own shared verify trunk + head (singleton
        // verify arithmetic per request; picks stay with the request state).
        st.spec_verify_planned(gpu, weights, config, s, plan)?;
        gpu.hip.device_synchronize()?;
        Ok(picks)
    }
}

/// One trunk forward over the rows of `members` (indices into
/// `plan.requests`, slot order), then their heads. With `split`, the pass's
/// hidden rows are copied to their plan rows in `hidden_out`.
#[allow(clippy::too_many_arguments)]
fn run_pass(
    gpu: &mut Gpu,
    st: &mut Qwen35VmmStore,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    plan: &BatchStepPlan,
    members: &[usize],
    split: bool,
    picks: &mut [u32],
) -> HipResult<()> {
    let pb = &plan.batch;
    let n_slots = pb.m_per_slot.len();
    let dim = config.dim;
    // Sub-batch in slot order: (request index, row offset in this pass).
    let mut b = hipfire_runtime::slot_batch::SlotBatch::default();
    b.m_per_slot = vec![0; n_slots];
    let mut offs: Vec<(usize, usize)> = Vec::with_capacity(members.len());
    for &i in members {
        let r = &plan.requests[i];
        offs.push((i, b.tokens.len()));
        let rows = r.rows.begin..r.rows.begin + r.rows.len;
        b.tokens.extend_from_slice(&pb.tokens[rows.clone()]);
        b.positions.extend_from_slice(&pb.positions[rows.clone()]);
        b.row_slot.extend_from_slice(&pb.row_slot[rows]);
        b.m_per_slot[pb.row_slot[r.rows.begin] as usize] = r.rows.len;
    }
    let n = b.total_rows();
    let to_bytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_ne_bytes()).collect() };
    st.row_slot_host.clear();
    st.row_slot_host.extend(b.tokens.iter().map(|&t| t as i32));
    gpu.hip
        .memcpy_htod(&st.pbs.tokens.buf, &to_bytes(&st.row_slot_host))?;
    gpu.hip
        .memcpy_htod(&st.pbs.positions.buf, &to_bytes(&b.positions))?;
    gpu.hip
        .memcpy_htod(&st.row_slot_dev.buf, &to_bytes(&b.row_slot))?;
    embed_lookup_slots(
        gpu,
        weights.embd_format,
        &weights.token_embd,
        &st.pbs.x_batch,
        &st.pbs.tokens,
        n,
        dim,
    )?;

    let mut dn: Vec<Option<&DeltaNetState>> = vec![None; n_slots];
    for state in st.slots.iter().flatten() {
        if state.slot < n_slots && b.m_per_slot[state.slot] > 0 {
            dn[state.slot] = Some(&state.dn);
        }
    }
    let dn = DnTable(dn);
    let max_ctx_len = (b.positions.iter().copied().max().unwrap_or(0) as usize + 1).max(1);
    let format = st.format;
    let flash = gpu.vmm_flash_decode_plan(
        format,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        st.model_max_seq,
        max_ctx_len,
    )?;
    if flash.partial_floats_per_row > st.flash_partials.numel() {
        return Err(HipError::new(
            0,
            "VMM executor: flash partials undersized for this step",
        ));
    }
    let tier = SlotKvTier {
        mode: mode_of(format),
        givens_cos: None,
        givens_sin: None,
        fwht_signs1: None,
        fwht_signs2: None,
    };
    let q8_wmma_arch = q8_prefill_wmma_enabled(gpu);
    {
        let partials = &st.flash_partials;
        let descs_dev = &st.descs_dev;
        let row_slot_dev = &st.row_slot_dev;
        run_layers_slots(
            gpu,
            weights,
            config,
            &b,
            &dn,
            |kv_layer_idx| {
                LayerKvAddr::Vmm(VmmLayerKv {
                    format,
                    descs: &descs_dev[kv_layer_idx],
                    row_slot: row_slot_dev,
                    flash: &flash,
                    partials,
                })
            },
            &tier,
            &st.pbs,
            s,
            q8_wmma_arch,
            n,
            max_ctx_len,
            max_ctx_len,
            false,
            config.n_layers,
            None,
        )?;
    }
    drop(dn);
    if split {
        let row_bytes = dim * 4;
        for &(i, off) in &offs {
            let r = &plan.requests[i];
            gpu.memcpy_dtod_at_auto(
                &st.hidden_out.buf,
                r.rows.begin * row_bytes,
                &st.pbs.x_batch.buf,
                off * row_bytes,
                r.rows.len * row_bytes,
            )?;
        }
    }

    // Head only for rows that pick: AR rows and the chunk completing a
    // prompt. Other prefill chunks' last-row logits would be discarded
    // work that re-reads the whole lm_head.
    st.skip.clear();
    st.skip.resize(n_slots, true);
    for &i in members {
        let r = &plan.requests[i];
        if st.wants_head(r) {
            st.skip[pb.row_slot[r.rows.begin] as usize] = false;
        }
    }
    if st.skip.iter().all(|&x| x) {
        return Ok(());
    }
    final_logits_per_slot(gpu, weights, config, &b, &st.pbs, s, &st.logits, &st.skip)?;
    sample_heads(gpu, st, config, s, plan, members, picks)
}

/// Pick every head row of `members` (plan request indices) from its `logits`
/// row with ONE `batch_sampler::sample_batch` — byte-identical per request to
/// the singleton route's own `sampler::sample` (same kernels, same scratch
/// window cap = `s.repeat_buf`, same blocked-token/penalty order), each with
/// its own history/config and a COPY of its RNG; the advanced RNG is
/// published only by commit. Requests without a head this step are skipped.
fn sample_heads(
    gpu: &mut Gpu,
    st: &mut Qwen35VmmStore,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    plan: &BatchStepPlan,
    members: &[usize],
    picks: &mut [u32],
) -> HipResult<()> {
    let vocab = config.vocab_size;
    if !members.iter().any(|&i| st.wants_head(&plan.requests[i])) {
        return Ok(());
    }
    if st.sampler_arena.is_none() {
        st.sampler_arena = Some(hipfire_runtime::batch_sampler::BatchSamplerArena::new(
            gpu,
            st.max_slots,
            s.repeat_buf.buf.size() / 4,
        )?);
    }
    let mut heads: Vec<(usize, &Qwen35RequestState)> = Vec::with_capacity(members.len());
    for &i in members {
        let r = &plan.requests[i];
        if !st.wants_head(r) {
            continue;
        }
        let state = st
            .slots
            .iter()
            .flatten()
            .find(|x| x.epoch == r.epoch)
            .expect("provisioned epoch");
        heads.push((i, state));
    }
    let views: Vec<GpuTensor> = heads
        .iter()
        .map(|(_, state)| st.logits.sub_offset(state.slot * vocab, vocab))
        .collect();
    let jobs: Vec<hipfire_runtime::batch_sampler::BatchSampleJob<'_>> = heads
        .iter()
        .zip(&views)
        .map(
            |((_, state), view)| hipfire_runtime::batch_sampler::BatchSampleJob {
                logits: view,
                history: &state.history,
                cfg: &state.sampler,
                rng_state: state.rng_state,
            },
        )
        .collect();
    let arena = st.sampler_arena.as_mut().expect("allocated above");
    let sampled = hipfire_runtime::batch_sampler::sample_batch(gpu, arena, &jobs)?;
    for (&(i, _), pick) in heads.iter().zip(sampled) {
        let r = &plan.requests[i];
        if pick.token_id as usize >= vocab {
            return Err(HipError::new(
                0,
                &format!("VMM executor: pick {} out of vocab", pick.token_id),
            ));
        }
        picks[r.rows.begin + r.rows.len - 1] = pick.token_id;
        st.pending_rng.push((r.epoch, pick.advanced_rng));
    }
    Ok(())
}

impl Qwen35VmmExecutor<'_> {
    /// Publish the step's private frontiers. Validates the step id, the
    /// exact plan, and every epoch against the owner table before mutating
    /// anything. Returns one advance per AR request (prefill chunks advance
    /// the KV frontier only; the transport drains their prompt).
    pub fn commit_step(
        &mut self,
        gpu: &mut Gpu,
        plan: &BatchStepPlan,
        output: StepOutput,
    ) -> Result<Vec<RequestAdvance>, String> {
        let (weights, config, scratch) = (self.weights, self.config, self.scratch);
        let st = &mut *self.store;
        match &st.phase {
            Phase::Forwarded(k, id) if *k == StepKey::of(plan) && *id == output.step_id => {}
            _ => return Err("commit_step: output does not belong to the forwarded plan".into()),
        }
        if output.target_picks.len() != plan.total_rows() {
            return Err("commit_step: pick count != planned rows".into());
        }
        for r in &plan.requests {
            match st.slots.iter().flatten().find(|s| s.epoch == r.epoch) {
                Some(s) if !s.poisoned => {}
                _ => {
                    return Err(format!(
                        "commit_step: epoch {:?} no longer owns its slot",
                        r.epoch
                    ))
                }
            }
            if st.wants_head(r) && output.target_picks[r.rows.begin + r.rows.len - 1] == u32::MAX {
                return Err("commit_step: head row without a target pick".into());
            }
        }
        let mut out = Vec::with_capacity(plan.requests.len());
        for r in &plan.requests {
            if matches!(r.kind, RequestStepKind::Verify { .. }) {
                // Accept/repair then publish this request's window. A
                // failure leaves device state untrusted: poison the step.
                match st.spec_commit_verify(gpu, weights, config, scratch, &r.epoch) {
                    Ok(a) => {
                        out.push(a);
                        continue;
                    }
                    Err(e) => {
                        st.poison(plan);
                        st.spec_clear_planned(plan);
                        st.phase = Phase::Idle;
                        return Err(e);
                    }
                }
            }
            let head = st.wants_head(r);
            let s = st
                .slots
                .iter_mut()
                .flatten()
                .find(|s| s.epoch == r.epoch)
                .expect("validated above");
            s.position += r.rows.len;
            if head {
                let id = output.target_picks[r.rows.begin + r.rows.len - 1];
                s.pending_seed = Some(id);
                s.history.push(id);
                if let Some(&(_, rng)) = st.pending_rng.iter().find(|(e, _)| *e == r.epoch) {
                    s.rng_state = rng;
                }
                let finish = if s.stop_ids.contains(&id) {
                    Some("stop".to_string())
                } else if s.position >= s.kv.vmm_logical_bound() {
                    Some("length".to_string())
                } else {
                    None
                };
                out.push(RequestAdvance {
                    epoch: r.epoch,
                    committed_ids: vec![id],
                    committed_position: s.position,
                    accepted_drafts: 0,
                    verified_rows: 0,
                    finish,
                });
            }
        }
        st.phase = Phase::Idle;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1 << 20;
    /// One 2 MiB granule on each of the 27B's 32 KV tensors.
    const CHUNK: usize = 64 * MIB;

    fn lane(held: usize, held_kv: usize, remaining: usize) -> VmmLaneNeed {
        VmmLaneNeed {
            held,
            held_kv,
            remaining,
        }
    }

    /// The R9700 long8 K=8 refusal: 88 MiB HIP-free and 367 MiB parked are
    /// below the 512 MiB map reserve, so the map room is 0 and the staged
    /// KV budget (13.4 GB unused) never bound. Parked buffers count as room.
    #[test]
    fn map_room_counts_parked_pool_bytes_above_the_reserve() {
        assert_eq!(vmm_map_room(88_080_384, 384_518_400), 0);
        assert_eq!(vmm_map_room(600 * MIB, 0), 88 * MIB);
        assert_eq!(vmm_map_room(100 * MIB, 500 * MIB), 88 * MIB);
        assert_eq!(
            vmm_map_room(usize::MAX, usize::MAX),
            usize::MAX - VMM_MAP_DEVICE_RESERVE_BYTES
        );
    }

    /// Lanes that already hold their whole reservation complete first and
    /// return their footprint, so lanes still needing a chunk are safe even
    /// with no room left; with every lane needing growth they are not.
    #[test]
    fn lanes_safe_orders_completions_by_remaining_growth() {
        let done = lane(1_500 * MIB, 192 * MIB, 0);
        let grow = lane(1_400 * MIB, 128 * MIB, CHUNK);
        assert!(vmm_lanes_safe(0, usize::MAX, &[grow, done, grow, done]));
        assert!(!vmm_lanes_safe(0, usize::MAX, &[grow, grow]));
        assert!(vmm_lanes_safe(CHUNK, usize::MAX, &[grow, grow]));
        // The KV budget is a second resource: a released lane returns only
        // its mapped KV to it.
        assert!(!vmm_lanes_safe(usize::MAX, CHUNK - 1, &[grow]));
        assert!(vmm_lanes_safe(usize::MAX, CHUNK, &[grow, grow]));
        assert!(vmm_lanes_safe(0, 0, &[]));
    }

    #[test]
    fn grant_growth_keeps_some_completion_order() {
        let mut lanes = vec![lane(1_400 * MIB, 128 * MIB, 2 * CHUNK); 2];
        let (mut room, mut kv_room) = (2 * CHUNK, usize::MAX);
        // Lane 1 may take one chunk: it can then finish in the chunk left.
        assert!(vmm_grant_growth(
            &mut room,
            &mut kv_room,
            &mut lanes,
            1,
            CHUNK
        ));
        assert_eq!(room, CHUNK);
        assert_eq!(
            lanes[1],
            lane(1_400 * MIB + CHUNK, 128 * MIB + CHUNK, CHUNK)
        );
        // Lane 0 taking that chunk would leave both lanes a chunk short with
        // no room: it waits, and nothing changes.
        let before = (room, lanes.clone());
        assert!(!vmm_grant_growth(
            &mut room,
            &mut kv_room,
            &mut lanes,
            0,
            CHUNK
        ));
        assert_eq!((room, lanes.clone()), before);
        // Lane 1 taking it completes lane 1's reservation; its release then
        // covers lane 0.
        assert!(vmm_grant_growth(
            &mut room,
            &mut kv_room,
            &mut lanes,
            1,
            CHUNK
        ));
        assert_eq!(room, 0);
        // A step that maps nothing is always granted, even with no room.
        assert!(vmm_grant_growth(&mut room, &mut kv_room, &mut lanes, 0, 0));
        // Growth past the room is never granted.
        let mut lanes = vec![lane(0, 0, 0)];
        let (mut room, mut kv_room) = (CHUNK - 1, usize::MAX);
        assert!(!vmm_grant_growth(
            &mut room,
            &mut kv_room,
            &mut lanes,
            0,
            CHUNK
        ));
    }

    /// Measured long8 footprint per DFlash lane: lane state 1,038,090,240 B,
    /// one 150,994,944 B checkpoint, 120,324,096 B DeltaNet, 3 KV chunks
    /// after a ~4.2K prompt; the request needs no further chunk.
    #[test]
    fn admission_charges_the_whole_lane_and_keeps_lanes_completable() {
        let held = 1_038_090_240 + 150_994_944 + 120_324_096 + 3 * CHUNK;
        let running = vec![lane(held, 3 * CHUNK, 0); 6];
        let new = lane(held, 3 * CHUNK, 0);
        assert!(vmm_admission_fits(held, usize::MAX, &running, new));
        assert!(!vmm_admission_fits(held - 1, usize::MAX, &running, new));
        // The new lane fits now, but every lane (it included) then needs a
        // chunk the room no longer has: it waits.
        let growing = vec![lane(held, 2 * CHUNK, CHUNK); 6];
        let new = lane(held, 2 * CHUNK, CHUNK);
        assert!(!vmm_admission_fits(held, usize::MAX, &growing, new));
        assert!(vmm_admission_fits(held + CHUNK, usize::MAX, &growing, new));
        // Its prompt KV is charged to the staged KV budget as well.
        assert!(!vmm_admission_fits(usize::MAX, 2 * CHUNK - 1, &[], new));
        assert!(vmm_admission_fits(usize::MAX, 3 * CHUNK, &[], new));
    }

    #[test]
    fn spec_bytes_charge_the_lane_kind_state() {
        // AR lanes hold no speculation state, whatever the engines probed.
        assert_eq!(vmm_spec_bytes(VmmLaneSpec::Ar, 7, 11, 13, 3), 0);
        // MTP: the probed MTP lane state only.
        assert_eq!(vmm_spec_bytes(VmmLaneSpec::Mtp, 7, 11, 13, 3), 7);
        // DFlash: lane state plus one snapshot per checkpoint.
        assert_eq!(
            vmm_spec_bytes(VmmLaneSpec::Dflash, 7, 11, 13, 3),
            11 + 3 * 13
        );
        assert_eq!(vmm_spec_bytes(VmmLaneSpec::Dflash, 7, 11, 13, 0), 11);
    }

    #[test]
    fn seed_free_deduction_covers_the_lane_owners_and_a_margin() {
        let (kv, dn, lane_state) = (3 * CHUNK, 120_324_096, 1_038_090_240);
        assert_eq!(
            vmm_seed_free_deduction(kv + dn + lane_state),
            kv + dn + lane_state + 256 * MIB
        );
        assert_eq!(vmm_seed_free_deduction(usize::MAX), usize::MAX);
    }

    #[test]
    fn mtp_lane_charge_enters_admission() {
        // A running MTP lane holds its probed state; a new one is refused
        // when the room cannot hold it after its prompt.
        let mtp = 600 * MIB;
        let held = vmm_spec_bytes(VmmLaneSpec::Mtp, mtp, 0, 0, 0) + 120_324_096 + 3 * CHUNK;
        let new = lane(held, 3 * CHUNK, 0);
        assert!(vmm_admission_fits(held, usize::MAX, &[], new));
        assert!(!vmm_admission_fits(held - 1, usize::MAX, &[], new));
        // Uncharged (the old behaviour) it would have fit.
        assert!(held - mtp < held);
    }
}
