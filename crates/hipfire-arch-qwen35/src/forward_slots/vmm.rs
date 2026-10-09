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
    final_logits_per_slot, lm_head_slots_admissible, q8_prefill_wmma_enabled,
    require_batchable_deltanet_layer, require_batchable_deltanet_moe_layer,
    require_batchable_fullattn_layer, require_batchable_fullattn_moe_layer,
    require_batchable_moe_ffn, run_layers_slots, LayerKvAddr, SlotKvTier,
};
use crate::qwen35::LayerWeights;
use crate::carrier::Qwen35Bundle;
use crate::qwen35::{
    DeltaNetState, LayerType, PrefillBatchScratch, Qwen35Config, Qwen35Scratch, Qwen35Weights,
};
use hip_bridge::{HipError, HipResult};
use hipfire_runtime::kv_mode::KvMode;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::llama::{EmbeddingFormat, KvCache, KvCacheExt, KvDims, KvLayers, KvTarget};
use hipfire_runtime::slot_batch::{
    BatchStepPlan, RequestAdvance, RequestEpoch, RequestStepKind, StepOutput,
};
use rdna_compute::attention::{VmmFlashDecodePlan, VmmKvFormat, VmmKvSide};
use rdna_compute::kv_slots::{validate_vmm_rows, VmmKvSlotDesc};
use hipfire_runtime::sampler::SamplerConfig;
use rdna_compute::{DType, Gpu, GpuTensor};
mod exact;
pub mod spec;
pub mod dflash;


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
        exact::supports(gpu, weights)?;
    }
    if !matches!(weights.embd_format, EmbeddingFormat::Q8_0) {
        return Err("VMM executor: embedding table must be Q8_0".into());
    }
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
                require_batchable_deltanet_moe_layer(l, arch).and_then(|_| require_batchable_moe_ffn(gpu, &l.ffn))
            }
            (LayerWeights::FullAttnMoe(l), LayerType::FullAttention) => {
                require_batchable_fullattn_moe_layer(l, arch).and_then(|_| require_batchable_moe_ffn(gpu, &l.ffn))
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

/// KV write (K and V) and causal attend over independent request owners.
pub(super) fn vmm_kv_write_attend(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    layer: &VmmLayerKv,
    pbs: &PrefillBatchScratch,
    n: usize,
) -> HipResult<()> {
    for (side, src) in [(VmmKvSide::K, &pbs.fa_k_batch), (VmmKvSide::V, &pbs.fa_v_batch)] {
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
        gpu.attention_flash_decode_vmm(
            layer.flash,
            &pbs.fa_q_batch.sub_offset(r0 * q_row, len * q_row),
            &pbs.fa_attn_out_batch.sub_offset(r0 * q_row, len * q_row),
            &pbs.positions.sub_offset(r0, len),
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            len,
            layer.partials,
            layer.descs,
            &layer.row_slot.sub_offset(r0, len),
        )?;
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
                return Err(format!("VMM executor: request DeltaNet state: {e}; kv free: {cleanup:?}"));
            }
        };
        Ok(Self {
            epoch,
            slot,
            kv,
            dn,
            position: 0,
            pending_seed: None,
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
        let r = self.kv.release_vmm_after(gpu, None).map_err(|e| e.to_string());
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
        // its prefill is the singleton prefill on the singleton's own scratch
        // and its attention is the singleton binding (no shared partials).
        // Sizing these for prefill rows would duplicate the singleton's PBS.
        let trunk_rows = match route {
            VmmRoute::Exact => max_slots,
            VmmRoute::Nonexact => row_budget,
        };
        let result = (|| -> HipResult<(Vec<GpuTensor>, GpuTensor, GpuTensor, GpuTensor, GpuTensor)> {
            let mut descs = Vec::with_capacity(kv_layer_ids.len());
            for _ in &kv_layer_ids {
                descs.push(gpu.zeros(&[max_slots * 32], DType::Raw)?);
            }
            // 4-byte elements so per-row-group sub-views address rows.
            let row_slot = gpu.zeros(&[trunk_rows], DType::F32)?;
            // Fixed partials budget, never rows × context: the attention runs
            // in row groups that fit it. At least one row at the bound.
            let floats = (VMM_FLASH_PARTIALS_BYTES / 4)
                .min(trunk_rows * worst.partial_floats_per_row)
                .max(worst.partial_floats_per_row);
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
        let n_dn = config.layer_types.iter().filter(|t| **t == LayerType::LinearAttention).count();
        // GDN (72 B) + conv (40 B) rows per DeltaNet layer; gated norm (24 B)
        // and FA prep (40 B) rows are layer-independent.
        let table_bytes = max_slots * (n_dn * (72 + 40) + 24 + 40);
        let rows_tables = match route {
            VmmRoute::Exact => match gpu.zeros(&[table_bytes / 4], DType::F32) {
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
            VmmRoute::Nonexact => None,
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
            return Err((state, "VMM executor: admit during an uncommitted step".into()));
        }
        if !state.epoch.is_admitted() {
            return Err((state, "VMM executor: admit of an unadmitted epoch".into()));
        }
        if state.slot >= self.max_slots {
            let m = format!("VMM executor: slot {} >= width {}", state.slot, self.max_slots);
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
            Ok(_) => return Err((state, "VMM executor: request KV format differs from store".into())),
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
            return Err(format!("provision_step: {trunk_rows} rows exceed row budget {}", st.row_budget));
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
                    return Err("provision_step: Forced rows are not admitted on the VMM executor".into());
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
            if r.kind == RequestStepKind::Prefill && state.position + r.rows.len > state.prompt_len {
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
                    r.epoch, state.kv.vmm_logical_bound()
                ));
            }
        }
        if covered.iter().any(|c| !c) {
            return Err("provision_step: rows outside every request range".into());
        }
        // Map granules (outside any forward/capture) against the shared
        // physical budget AND the device's actual free memory (less a
        // reserve for the singleton prefill scratch the exact route widens),
        // so an overstated budget is a typed refusal, never a raw HIP OOM.
        // Refusal maps nothing for that request.
        let mut mapped_total = self.store.mapped_kv_bytes()?;
        let mut device_room = gpu
            .hip
            .get_vram_info()
            .map_err(|e| format!("provision_step: VRAM query: {e}"))?
            .0
            .saturating_sub(VMM_MAP_DEVICE_RESERVE_BYTES);
        let st = &mut *self.store;
        for r in &plan.requests {
            let budget = st.kv_budget_bytes.saturating_sub(mapped_total).min(device_room);
            let s = st
                .slots
                .iter_mut()
                .flatten()
                .find(|s| s.epoch == r.epoch)
                .expect("validated above");
            let end = s.position + r.rows.len;
            let grown = s
                .kv
                .provision_vmm_positions(gpu, end, budget)
                .map_err(|e| format!("provision_step: map {end} positions for {:?}: {e}", r.epoch))?;
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
            let generation = s
                .kv
                .vmm_owner_generation(gpu)
                .map_err(|e| format!("provision_step: {:?}: {e}", r.epoch))?;
            st.row_gen[r.rows.begin..r.rows.begin + r.rows.len].fill(generation);
            let descs = s
                .kv
                .vmm_slot_descs(gpu)
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
    pub fn forward_step(&mut self, gpu: &mut Gpu, plan: &BatchStepPlan) -> Result<StepOutput, String> {
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
        if !matches!(weights.embd_format, EmbeddingFormat::Q8_0) {
            return Err(HipError::new(0, "VMM executor: embedding table must be Q8_0"));
        }
        let ms = st.max_slots;
        for (l, dev) in st.descs_dev.iter().enumerate() {
            gpu.hip.memcpy_htod(
                &dev.buf,
                VmmKvSlotDesc::as_bytes(&st.descs_host[l * ms..(l + 1) * ms]),
            )?;
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
                exact::prefill(gpu, st, weights, config, s, plan, i, &mut picks)?;
            }
        } else if split {
            run_pass(gpu, st, weights, config, s, plan, &decode, true, &mut picks)?;
            run_pass(gpu, st, weights, config, s, plan, &prefill, true, &mut picks)?;
        } else if !decode.is_empty() || !prefill.is_empty() {
            let all: Vec<usize> = (0..plan.requests.len()).filter(|&i| !is_verify(i)).collect();
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
    gpu.hip.memcpy_htod(&st.pbs.tokens.buf, &to_bytes(&st.row_slot_host))?;
    gpu.hip.memcpy_htod(&st.pbs.positions.buf, &to_bytes(&b.positions))?;
    gpu.hip.memcpy_htod(&st.row_slot_dev.buf, &to_bytes(&b.row_slot))?;
    gpu.embedding_lookup_q8_batched(&weights.token_embd, &st.pbs.x_batch, &st.pbs.tokens, n, dim)?;

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
        return Err(HipError::new(0, "VMM executor: flash partials undersized for this step"));
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
    for &i in members {
        sample_head(gpu, st, config, s, plan, i, picks)?;
    }
    Ok(())
}

/// Pick plan request `i`'s head row from its `logits` row with the singleton
/// route's own sampler (`sampler::sample`: same kernel, same scratch window
/// cap, same penalty/blocked-token order), its own history/config and a COPY
/// of its RNG; the advanced RNG is published only by commit. No-op for a
/// request without a head this step.
fn sample_head(
    gpu: &mut Gpu,
    st: &mut Qwen35VmmStore,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    plan: &BatchStepPlan,
    i: usize,
    picks: &mut [u32],
) -> HipResult<()> {
    let r = &plan.requests[i];
    if !st.wants_head(r) {
        return Ok(());
    }
    let state = st
        .slots
        .iter()
        .flatten()
        .find(|x| x.epoch == r.epoch)
        .expect("provisioned epoch");
    let view = st
        .logits
        .sub_offset(state.slot * config.vocab_size, config.vocab_size);
    let mut rng = state.rng_state;
    let id = hipfire_runtime::sampler::sample(
        gpu,
        &view,
        &s.sample_buf,
        &s.repeat_buf,
        config.vocab_size,
        &state.history,
        &state.sampler,
        &mut rng,
    );
    if id as usize >= config.vocab_size {
        return Err(HipError::new(0, &format!("VMM executor: pick {id} out of vocab")));
    }
    picks[r.rows.begin + r.rows.len - 1] = id;
    st.pending_rng.push((r.epoch, rng));
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
                _ => return Err(format!("commit_step: epoch {:?} no longer owns its slot", r.epoch)),
            }
            if st.wants_head(r)
                && output.target_picks[r.rows.begin + r.rows.len - 1] == u32::MAX
            {
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
