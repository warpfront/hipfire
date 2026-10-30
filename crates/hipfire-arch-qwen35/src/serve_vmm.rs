// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
//! VMM KV storage for the multi-slot serve engine (`serve.multi_slot` with
//! `kv_backend=vmm`).
//!
//! The arena route keeps one `SlotPool` slab per KV layer for `n_slots ×
//! cap_tokens` positions, reserved at load. This module is the alternative:
//! every slot owns its own `KvCache` VMM reservation, mapped on demand and
//! addressed by `VmmKvSlotDesc` absolute VAs (the ABI the `_vmm` write and
//! flash-decode kernels read).
//!
//! # Owner lifecycle
//!
//! One owner per slot, created on the slot's first admission and **kept for
//! the engine's lifetime**. `VmmArena::release` never returns virtual address
//! space (it is retired against a process cap), so a per-request owner would
//! burn VA in proportion to request count. Reusing the owner costs a sticky
//! physical high-water mark per slot — the same trade the slab route makes —
//! and keeps the descriptor bytes (bases, generation) constant for a request's
//! lifetime, which is what makes the decode graph pointer-stable.
//!
//! # Mapping
//!
//! All mapping happens at admission, to `prompt + max_tokens + slack`
//! positions. [`VmmSlotKv::forward_step`] asserts the step's write frontier is
//! inside the mapped prefix, so an admitted request can never fail a step on a
//! map. (The VMM batch route maps inside `provision_step`; this engine keeps
//! its stronger "no OOM mid-step" guarantee instead.)

use hip_bridge::{HipError, HipResult};
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::kv_mode::KvMode;
use hipfire_runtime::llama::{KvCache, KvDims, KvLayers, VMode};
use hipfire_runtime::slot_batch::SlotBatch;
use hipfire_runtime::swap::snapshot::{checksum_of, SlotSnapshot, SnapshotStamp};
use hipfire_runtime::swap::SwapError;
use rdna_compute::attention::VmmKvFormat;
use rdna_compute::kv_slots::{validate_vmm_rows, VmmKvSlotDesc};
use rdna_compute::slot_pool::SlotPool;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::forward_slots::vmm::{vmm_map_room, VmmLayerKv, VMM_RESERVE_SLACK_ROWS};
use crate::forward_slots::{
    advance_slot_seq_lens, embed_lookup_slots, final_logits_per_slot, run_layers_slots,
    LayerKvAddr, SlotKvTier, SpecVerifyCapture,
};
use crate::qwen35::{
    DeltaNetState, ExtractHiddenCapture, LayerType, PrefillBatchScratch, Qwen35Config,
    Qwen35Scratch, Qwen35Weights,
};
use hipfire_dispatch::pipeline::batched::q8_prefill_wmma_enabled;

/// Device budget of the row-batched flash-decode partials (same fixed policy
/// as the VMM batch route: attention runs in row groups that fit it).
const VMM_FLASH_PARTIALS_BYTES: usize = 256 << 20;

/// Device memory never handed to KV maps: the prefill scratch may widen after
/// admission, and the graph/stream machinery needs headroom.
const VMM_MAP_DEVICE_RESERVE_BYTES: usize = 512 << 20;

/// Marker OR-ed into `SnapshotStamp::kv_dtype_tag` for VMM snapshots. The
/// arena and VMM payloads for the same `KvMode` have the same shape today, but
/// they are different storage routes; a snapshot must never cross.
pub const VMM_SNAPSHOT_TAG_BIT: u32 = 0x8000_0000;

/// Resolve the `_vmm` KV format for a slots tier. Rotated (asym/fwht) and
/// flat-2B tiers have no descriptor kernel, so they are refused rather than
/// silently routed through the arena.
pub fn vmm_format_for_mode(mode: KvMode) -> Result<VmmKvFormat, String> {
    match mode {
        KvMode::Q8 => Ok(VmmKvFormat::Q8),
        KvMode::Fp8 => Ok(VmmKvFormat::Fp8E4M3),
        other => Err(format!(
            "experimental multi-slot kv_backend=vmm supports q8 and fp8 KV only \
             (no `_vmm` descriptor kernel for {other:?}); use kv_backend=legacy \
             for the rotated/native tiers"
        )),
    }
}

fn mode_of(format: VmmKvFormat) -> KvMode {
    match format {
        VmmKvFormat::Q8 => KvMode::Q8,
        VmmKvFormat::Fp8E4M3 => KvMode::Fp8,
    }
}

/// Positions a request must be able to write, and therefore be mapped at
/// admission: prompt + generation budget + the speculative slack, clamped to
/// the context cap. Mapping this at admission is what lets the forward assert
/// (never grow) the frontier inside a step (invariant I5).
pub fn slot_reserve_end(prompt_len: usize, max_tokens: usize, cap_tokens: usize) -> usize {
    prompt_len
        .saturating_add(max_tokens.max(1))
        .saturating_add(VMM_RESERVE_SLACK_ROWS)
        .min(cap_tokens)
        .max(1)
}

/// Mapped physical bytes of one owner, recomputed from the owner itself.
fn owner_mapped_bytes(kv: &KvCache, n_kv_layers: usize) -> Result<usize, String> {
    let format = kv
        .vmm_kv_format()
        .map_err(|e| format!("VMM slot KV: owner format: {e}"))?;
    let tokens = kv
        .mapped_token_capacity()
        .map_err(|e| format!("VMM slot KV: mapped capacity: {e}"))?
        .unwrap_or(0);
    Ok(tokens * format.bytes_per_token(kv.n_kv_heads, kv.head_dim) * 2 * n_kv_layers)
}

/// One slot's VMM KV owner table plus the staging the `_vmm` forward reads.
pub struct VmmSlotKv {
    format: VmmKvFormat,
    mode: KvMode,
    n_kv_heads: usize,
    head_dim: usize,
    n_layers: usize,
    /// Model layer index of each FullAttention layer, in order.
    kv_layer_ids: Vec<usize>,
    /// Model-sized KV-carrying mask (`layer_types == FullAttention`).
    is_kv_layer: Vec<bool>,
    owners: Vec<Option<KvCache>>,
    /// Mapped-prefix guarantee per slot (positions an admitted request may use).
    reserve_end: Vec<usize>,
    /// One device `[n_slots]` descriptor table per KV layer (stable VAs).
    descs_dev: Vec<GpuTensor>,
    /// Layer-major host mirror: `l * n_slots + slot`.
    descs_host: Vec<VmmKvSlotDesc>,
    /// Bytes last uploaded per layer; `None` = unknown.
    descs_uploaded: Vec<Option<Vec<u8>>>,
    /// `[max_batch]` row → slot map for the `_vmm` kernels (F32 elements so
    /// `sub_offset` addresses rows).
    row_slot_dev: GpuTensor,
    /// Fixed-budget flash-decode partials (rows are grouped to fit it).
    flash_partials: GpuTensor,
    /// Engine-wide ceiling on mapped KV bytes.
    kv_budget_bytes: usize,
    mapped_high_water: usize,
    n_slots: usize,
    cap_tokens: usize,
    /// Per-slot GatedDeltaNet stochastic-rounding frame, used only when the
    /// slot's DN carries no error-feedback residuals (mirrors the VMM batch
    /// route's exact prefill around `forward_prefill_batch`).
    gdn_frame: Vec<u32>,
    /// hipGraph cache for pure-decode steps (one graph per shape bucket).
    graph: VmmDecodeGraph,
}

/// Context-length bucket for the VMM decode graph: a capture bakes the
/// attention context bound into kernargs, so it is reused until the real
/// context crosses the next bucket (one capture per this many tokens).
const VMM_DECODE_GRAPH_CTX_BUCKET: usize = 8192;

/// Identity of a captured VMM decode graph. Any change invalidates it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct VmmGraphKey {
    n_rows: usize,
    ctx_bucket: usize,
    /// FNV-1a over the per-slot row counts and the lm_head-skip mask: both
    /// change which kernels the captured body launches.
    m_hash: u64,
}

fn vmm_decode_graph_m_hash(m_per_slot: &[usize], lm_head_skip: &[bool]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &m in m_per_slot {
        h ^= m as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    for &skip in lm_head_skip {
        h ^= u64::from(skip);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// One captured decode graph plus its shape key. Owned by [`VmmSlotKv`] so a
/// storage route can never replay another route's graph.
#[derive(Default)]
struct VmmDecodeGraph {
    exec: Option<hip_bridge::GraphExec>,
    graph: Option<hip_bridge::Graph>,
    key: Option<VmmGraphKey>,
    captures: usize,
    replays: usize,
}

impl VmmDecodeGraph {
    /// Destroy captured handles before the owning tensors are freed. Safe to
    /// call when empty.
    fn release(&mut self, gpu: &Gpu) {
        if let Some(exec) = self.exec.take() {
            let _ = gpu.hip.graph_exec_destroy(exec);
        }
        if let Some(graph) = self.graph.take() {
            let _ = gpu.hip.graph_destroy(graph);
        }
        self.key = None;
    }

    /// `(captures, replays)` — the same health signal the arena cache exposes.
    pub fn stats(&self) -> (usize, usize) {
        (self.captures, self.replays)
    }
}

impl VmmSlotKv {
    /// Allocate the staging tables. No KV is reserved or mapped here; owners
    /// are created per slot on first admission ([`Self::ensure_slot`]).
    pub fn new(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        format: VmmKvFormat,
        n_slots: usize,
        cap_tokens: usize,
        max_batch: usize,
        kv_budget_bytes: usize,
    ) -> Result<Self, String> {
        if n_slots == 0 || max_batch < n_slots || cap_tokens == 0 {
            return Err(format!(
                "VMM slot KV: invalid geometry (n_slots={n_slots}, \
                 max_batch={max_batch}, cap_tokens={cap_tokens})"
            ));
        }
        let kv_layer_ids: Vec<usize> = config
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == LayerType::FullAttention)
            .map(|(i, _)| i)
            .collect();
        if kv_layer_ids.is_empty() {
            return Err("VMM slot KV: model has no FullAttention layer".to_string());
        }
        let worst = gpu
            .vmm_flash_decode_plan(
                format,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                cap_tokens,
                cap_tokens,
            )
            .map_err(|e| format!("VMM slot KV: flash plan: {e}"))?;
        let floats = (VMM_FLASH_PARTIALS_BYTES / 4)
            .min(max_batch * worst.partial_floats_per_row)
            .max(worst.partial_floats_per_row);
        // Allocation ledger: any failure frees what this ctor already took.
        let mut descs_dev: Vec<GpuTensor> = Vec::with_capacity(kv_layer_ids.len());
        let mut staging_error: Option<String> = None;
        for _ in &kv_layer_ids {
            match gpu.zeros(&[n_slots * 32], DType::Raw) {
                Ok(t) => descs_dev.push(t),
                Err(e) => {
                    staging_error = Some(format!("VMM slot KV desc table: {e}"));
                    break;
                }
            }
        }
        let row_slot_dev = if staging_error.is_none() {
            match gpu.zeros(&[max_batch], DType::F32) {
                Ok(t) => Some(t),
                Err(e) => {
                    staging_error = Some(format!("VMM slot KV row map: {e}"));
                    None
                }
            }
        } else {
            None
        };
        let flash_partials = if staging_error.is_none() {
            match gpu.zeros(&[floats], DType::F32) {
                Ok(t) => Some(t),
                Err(e) => {
                    staging_error = Some(format!("VMM slot KV partials: {e}"));
                    None
                }
            }
        } else {
            None
        };
        if let Some(e) = staging_error {
            for t in descs_dev
                .drain(..)
                .chain(row_slot_dev)
                .chain(flash_partials)
            {
                let _ = gpu.free_tensor(t);
            }
            return Err(e);
        }
        let row_slot_dev = row_slot_dev.expect("allocated above");
        let flash_partials = flash_partials.expect("allocated above");
        let is_kv_layer: Vec<bool> = config
            .layer_types
            .iter()
            .map(|t| *t == LayerType::FullAttention)
            .collect();
        let n_kv_layers = kv_layer_ids.len();
        Ok(Self {
            format,
            mode: mode_of(format),
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            n_layers: kv_layer_ids.len(),
            kv_layer_ids,
            is_kv_layer,
            owners: (0..n_slots).map(|_| None).collect(),
            reserve_end: vec![0; n_slots],
            descs_dev,
            descs_host: vec![VmmKvSlotDesc::MASKED; n_kv_layers * n_slots],
            descs_uploaded: vec![None; n_kv_layers],
            row_slot_dev,
            flash_partials,
            kv_budget_bytes,
            mapped_high_water: 0,
            n_slots,
            cap_tokens,
            gdn_frame: (0..n_slots)
                .map(|_| rdna_compute::norm::gdn_requant_frame_checkpoint())
                .collect(),
            graph: VmmDecodeGraph::default(),
        })
    }

    pub fn format(&self) -> VmmKvFormat {
        self.format
    }

    pub fn mode(&self) -> KvMode {
        self.mode
    }

    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    pub fn budget_bytes(&self) -> usize {
        self.kv_budget_bytes
    }

    pub fn mapped_high_water(&self) -> usize {
        self.mapped_high_water
    }

    /// `(captures, replays)` of the decode graph cache (telemetry/tests).
    pub fn graph_stats(&self) -> (usize, usize) {
        self.graph.stats()
    }

    /// Replace the engine-wide mapped-byte ceiling (called once at load, after
    /// every non-KV allocation is in place).
    pub fn set_budget_bytes(&mut self, bytes: usize) {
        self.kv_budget_bytes = bytes;
    }

    /// Sum of the physical bytes every owner currently has mapped.
    pub fn mapped_kv_bytes(&self) -> Result<usize, String> {
        let mut total = 0usize;
        for owner in self.owners.iter().flatten() {
            total = total.saturating_add(owner_mapped_bytes(owner, self.n_layers)?);
        }
        Ok(total)
    }

    pub fn has_owner(&self, slot: usize) -> bool {
        self.owners.get(slot).is_some_and(Option::is_some)
    }

    /// Mapped token capacity of a slot's owner (0 when it has none).
    pub fn mapped_positions(&self, slot: usize) -> usize {
        self.owners
            .get(slot)
            .and_then(Option::as_ref)
            .and_then(|kv| kv.mapped_token_capacity().ok().flatten())
            .unwrap_or(0)
    }

    /// Positions this slot's current request may write.
    pub fn reserve_end(&self, slot: usize) -> usize {
        self.reserve_end.get(slot).copied().unwrap_or(0)
    }

    /// Create the slot's owner (if absent) and map its KV prefix to cover
    /// `end` positions. Called at admission/restore only; never inside a step.
    ///
    /// Refusal is a typed capacity error: nothing is mapped for the request
    /// and the caller rejects it (retryable overload).
    pub fn ensure_slot(&mut self, gpu: &mut Gpu, slot: usize, end: usize) -> Result<(), String> {
        if slot >= self.n_slots {
            return Err(format!("VMM slot KV: slot {slot} >= {}", self.n_slots));
        }
        let end = end.min(self.cap_tokens).max(1);
        if self.owners[slot].is_none() {
            let dims = KvDims {
                layers: KvLayers::Mask(self.is_kv_layer.clone()),
                n_kv_heads: self.n_kv_heads,
                head_dim: self.head_dim,
                max_seq: self.cap_tokens,
                physical_cap: Some(self.cap_tokens),
            };
            let kv = KvCache::from_mode_with_backend(self.mode, KvBackend::Vmm, gpu, &dims)
                .map_err(|e| format!("VMM slot KV: slot {slot} owner ({:?}): {e}", self.mode))?;
            self.owners[slot] = Some(kv);
            self.reserve_end[slot] = 0;
        }
        let mapped = self.mapped_kv_bytes()?;
        let room = vmm_map_room(
            gpu.device_mem_info()
                .map_err(|e| format!("VMM slot KV: VRAM query: {e}"))?
                .0,
            gpu.pool_parked_bytes(),
        );
        let budget = self.kv_budget_bytes.saturating_sub(mapped).min(room);
        let kv = self.owners[slot]
            .as_mut()
            .expect("owner just created above");
        let need = kv
            .vmm_admission_bytes(gpu, end)
            .map_err(|e| format!("VMM slot KV: slot {slot} admission: {e}"))?;
        if need > budget {
            return Err(format!(
                "VMM slot KV: mapping {end} positions for slot {slot} needs \
                 {need} B, budget {budget} B (mapped {mapped} B of {}) — \
                 capacity exhausted",
                self.kv_budget_bytes
            ));
        }
        let grown = kv
            .provision_vmm_positions(gpu, end, budget)
            .map_err(|e| format!("VMM slot KV: slot {slot} map {end}: {e}"))?;
        self.mapped_high_water = self.mapped_high_water.max(mapped.saturating_add(grown));
        self.reserve_end[slot] = self.reserve_end[slot].max(end);
        self.refresh_descs(gpu)
    }

    /// Release a slot's request mapping guarantee. The owner (and its mapped
    /// prefix) is kept: it is reused by the next request in this slot.
    pub fn release_slot(&mut self, slot: usize) {
        self.reserve_end[slot] = 0;
    }

    /// Fail-closed check that this step's write frontier is inside the slot's
    /// mapped prefix. Never maps.
    pub fn check_frontier(&self, slot: usize, frontier: usize) -> Result<(), String> {
        let owner = self
            .owners
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                format!("VMM slot KV: slot {slot} has no owner for a {frontier}-position write")
            })?;
        let mapped = owner
            .mapped_token_capacity()
            .map_err(|e| format!("VMM slot KV: slot {slot} mapped capacity: {e}"))?
            .unwrap_or(0);
        if frontier > self.reserve_end(slot) || frontier > mapped {
            return Err(format!(
                "VMM slot KV: slot {slot} write frontier {frontier} outside the \
                 mapped prefix (reserve_end {}, mapped {mapped}) — mapping must \
                 happen at admission, never inside a step",
                self.reserve_end(slot)
            ));
        }
        Ok(())
    }

    /// Rebuild the host descriptor table and upload any layer whose bytes
    /// changed. Descriptor contents for a live request are constant (bases and
    /// owner generation are fixed, mapped_positions only grows), so uploads
    /// happen at admission and on restore.
    fn refresh_descs(&mut self, gpu: &Gpu) -> Result<(), String> {
        for d in self.descs_host.iter_mut() {
            *d = VmmKvSlotDesc::MASKED;
        }
        for (slot, owner) in self.owners.iter().enumerate() {
            let Some(owner) = owner else { continue };
            let descs = owner
                .vmm_slot_descs(gpu)
                .map_err(|e| format!("VMM slot KV: descriptors for slot {slot}: {e}"))?;
            for (l, &layer) in self.kv_layer_ids.iter().enumerate() {
                self.descs_host[l * self.n_slots + slot] = descs[layer];
            }
        }
        let ms = self.n_slots;
        for (l, dev) in self.descs_dev.iter().enumerate() {
            let bytes = VmmKvSlotDesc::as_bytes(&self.descs_host[l * ms..(l + 1) * ms]);
            if self.descs_uploaded[l].as_deref() == Some(bytes) {
                continue;
            }
            gpu.hip
                .memcpy_htod(&dev.buf, bytes)
                .map_err(|e| format!("VMM slot KV: desc upload layer {l}: {e}"))?;
            let mut mirror = self.descs_uploaded[l].take().unwrap_or_default();
            mirror.clear();
            mirror.extend_from_slice(bytes);
            self.descs_uploaded[l] = Some(mirror);
        }
        Ok(())
    }

    /// One batched decode/verify step for rows already addressed by `batch`.
    ///
    /// Prefill rows are **not** handled here: the engine prefills a slot's
    /// chunk on the singleton body against the slot's own owner (byte-identical
    /// to the serial route) and passes only decode/verify rows.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_step(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        batch: &SlotBatch,
        pool: &mut SlotPool,
        dn_states: &[DeltaNetState],
        pbs: &PrefillBatchScratch,
        s: &Qwen35Scratch,
        logits_out: &GpuTensor,
        lm_head_skip: &[bool],
        spec_capture: Option<&mut SpecVerifyCapture>,
    ) -> HipResult<()> {
        let n = batch.total_rows();
        if n == 0 {
            return Ok(());
        }
        // Validation and per-step inputs run OUTSIDE any capture: a captured
        // memcpy bakes its host source pointer.
        self.prepare_step(gpu, batch, pool, pbs)?;
        // A verify step fills the GDN tape and writes hidden rows the accept
        // path reads; it is never graph-captured (the tape would be stale on
        // replay). Verify steps also carry >1 row, so `pure_decode` excludes
        // them already; this is the defensive belt.
        let pure_decode = spec_capture.is_none() && batch.m_per_slot.iter().all(|&m| m <= 1);
        if !pure_decode {
            return self.step_body(
                gpu,
                weights,
                config,
                batch,
                pool,
                dn_states,
                pbs,
                s,
                logits_out,
                lm_head_skip,
                None,
                spec_capture,
            );
        }
        if !gpu.slots_decode_graph() {
            return self.step_body(
                gpu,
                weights,
                config,
                batch,
                pool,
                dn_states,
                pbs,
                s,
                logits_out,
                lm_head_skip,
                None,
                spec_capture,
            );
        }
        if gpu.owns_host_cpu_weights() {
            hipfire_dispatch::log_capture_disabled_once();
            return self.step_body(
                gpu,
                weights,
                config,
                batch,
                pool,
                dn_states,
                pbs,
                s,
                logits_out,
                lm_head_skip,
                None,
                spec_capture,
            );
        }
        self.graphed_step(
            gpu,
            weights,
            config,
            batch,
            pool,
            dn_states,
            pbs,
            s,
            logits_out,
            lm_head_skip,
            n,
        )
    }

    /// Host-side validation and the per-step H2D uploads. Every check that can
    /// reject a step runs here, before any launch or capture.
    fn prepare_step(
        &mut self,
        gpu: &mut Gpu,
        batch: &SlotBatch,
        pool: &SlotPool,
        pbs: &PrefillBatchScratch,
    ) -> HipResult<()> {
        let n = batch.total_rows();
        if !batch.pos3.is_empty() || batch.ext_emb.iter().any(|&e| e >= 0) {
            return Err(HipError::new(
                0,
                "VMM slot forward: VL/M-RoPE rows are not admitted on the VMM route",
            ));
        }
        let n_slots = self.n_slots;
        if batch.m_per_slot.len() != n_slots || pool.descriptors().len() != n_slots {
            return Err(HipError::new(
                0,
                &format!(
                    "VMM slot forward: shape mismatch (batch {}, pool {}) vs {n_slots}",
                    batch.m_per_slot.len(),
                    pool.descriptors().len()
                ),
            ));
        }
        if n > pbs.max_batch {
            return Err(HipError::new(
                0,
                &format!(
                    "VMM slot forward: {n} rows exceed pbs.max_batch {}",
                    pbs.max_batch
                ),
            ));
        }
        if batch.row_slot.len() != n || batch.positions.len() != n || batch.tokens.len() != n {
            return Err(HipError::new(0, "VMM slot forward: malformed SlotBatch"));
        }
        // Every row must address a live owner inside its mapped prefix.
        let mut row_gen = vec![0u64; n];
        for (row, row_generation) in row_gen.iter_mut().enumerate() {
            let slot = usize::try_from(batch.row_slot[row])
                .map_err(|_| HipError::new(0, "VMM slot forward: negative row_slot"))?;
            if slot >= n_slots {
                return Err(HipError::new(
                    0,
                    &format!("VMM slot forward: row {row} slot {slot} >= {n_slots}"),
                ));
            }
            let owner = self.owners[slot].as_ref().ok_or_else(|| {
                HipError::new(
                    0,
                    &format!("VMM slot forward: row {row} slot {slot} has no owner"),
                )
            })?;
            let pos = batch.positions[row];
            if pos < 0 {
                return Err(HipError::new(0, "VMM slot forward: negative position"));
            }
            let frontier = pos as usize + 1;
            if frontier > self.reserve_end[slot] {
                return Err(HipError::new(
                    0,
                    &format!(
                        "VMM slot forward: row {row} position {pos} outside slot {slot}'s \
                         reserved prefix {}",
                        self.reserve_end[slot]
                    ),
                ));
            }
            *row_generation = owner
                .vmm_owner_generation(gpu)
                .map_err(|e| HipError::new(0, &format!("VMM slot forward: generation: {e}")))?;
        }

        let tokens_host: Vec<i32> = batch.tokens.iter().map(|&t| t as i32).collect();
        let tokens_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
        gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
        let positions_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(batch.positions.as_ptr() as *const u8, n * 4) };
        gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
        let row_slot_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(batch.row_slot.as_ptr() as *const u8, n * 4) };
        gpu.hip
            .memcpy_htod(&self.row_slot_dev.buf, row_slot_bytes)?;

        // Descriptor table: refresh (no-op when unchanged) and validate every
        // row against its owner generation before any launch.
        self.refresh_descs(gpu).map_err(|e| HipError::new(0, &e))?;
        for l in 0..self.n_layers {
            let table = &self.descs_host[l * n_slots..(l + 1) * n_slots];
            validate_vmm_rows(table, &batch.row_slot, &batch.positions, &row_gen).map_err(|e| {
                HipError::new(
                    0,
                    &format!("VMM slot forward: layer {}: {e}", self.kv_layer_ids[l]),
                )
            })?;
        }
        Ok(())
    }

    /// The GPU half of a step. `ctx_override` fixes the attention context
    /// bound for a captured graph (rounded up to a bucket); over-scanning
    /// inside a bucket is safe because `positions[]` is the causal authority.
    #[allow(clippy::too_many_arguments)]
    fn step_body(
        &self,
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        batch: &SlotBatch,
        pool: &mut SlotPool,
        dn_states: &[DeltaNetState],
        pbs: &PrefillBatchScratch,
        s: &Qwen35Scratch,
        logits_out: &GpuTensor,
        lm_head_skip: &[bool],
        ctx_override: Option<usize>,
        mut spec_capture: Option<&mut SpecVerifyCapture>,
    ) -> HipResult<()> {
        let n = batch.total_rows();
        embed_lookup_slots(
            gpu,
            weights.embd_format,
            &weights.token_embd,
            &pbs.x_batch,
            &pbs.tokens,
            n,
            config.dim,
        )?;
        let max_ctx_len = ctx_override.unwrap_or_else(|| {
            (batch.positions.iter().copied().max().unwrap_or(0) as usize + 1)
                .min(self.cap_tokens)
                .max(1)
        });
        let flash = gpu
            .vmm_flash_decode_plan(
                self.format,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                self.cap_tokens,
                max_ctx_len,
            )
            .map_err(|e| HipError::new(0, &format!("VMM slot forward: flash plan: {e}")))?;
        if flash.partial_floats_per_row > self.flash_partials.numel() {
            return Err(HipError::new(
                0,
                "VMM slot forward: flash partials undersized for this step",
            ));
        }
        let tier = SlotKvTier {
            mode: self.mode,
            givens_cos: None,
            givens_sin: None,
            fwht_signs1: None,
            fwht_signs2: None,
        };
        let q8_wmma_arch = q8_prefill_wmma_enabled(gpu);
        let format = self.format;
        let row_slot_dev = &self.row_slot_dev;
        let partials = &self.flash_partials;
        let descs_dev = &self.descs_dev;
        run_layers_slots(
            gpu,
            weights,
            config,
            batch,
            dn_states,
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
            pbs,
            s,
            q8_wmma_arch,
            n,
            max_ctx_len,
            max_ctx_len,
            false,
            config.n_layers,
            spec_capture.as_deref_mut(),
        )?;
        final_logits_per_slot(
            gpu,
            weights,
            config,
            batch,
            pbs,
            s,
            logits_out,
            lm_head_skip,
        )?;
        advance_slot_seq_lens(batch, pool)
    }

    /// hipGraph capture/replay for pure-decode VMM steps. The descriptor
    /// tables and owner VAs are stable for a request's lifetime, and mapping
    /// never happens inside a step, so a captured step stays valid until the
    /// context bucket or batch composition changes.
    #[allow(clippy::too_many_arguments)]
    fn graphed_step(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        batch: &SlotBatch,
        pool: &mut SlotPool,
        dn_states: &[DeltaNetState],
        pbs: &PrefillBatchScratch,
        s: &Qwen35Scratch,
        logits_out: &GpuTensor,
        lm_head_skip: &[bool],
        n: usize,
    ) -> HipResult<()> {
        let true_ctx = (batch.positions.iter().copied().max().unwrap_or(0) as usize + 1)
            .min(self.cap_tokens)
            .max(1);
        let ctx_bucket = true_ctx
            .div_ceil(VMM_DECODE_GRAPH_CTX_BUCKET)
            .saturating_mul(VMM_DECODE_GRAPH_CTX_BUCKET)
            .min(self.cap_tokens)
            .max(1);
        let key = VmmGraphKey {
            n_rows: n,
            ctx_bucket,
            m_hash: vmm_decode_graph_m_hash(&batch.m_per_slot, lm_head_skip),
        };
        // Take the cache out so the body (which borrows `self` immutably) can
        // run while the graph handles are manipulated. Restored on every path.
        let mut graph = std::mem::take(&mut self.graph);
        let result = (|| -> HipResult<()> {
            if graph.key != Some(key) {
                graph.release(gpu);
                gpu.ensure_capture_stream()?;
                // Warm-up: the same kernels at the same shape, outside capture
                // (a kernel compile inside capture is forbidden).
                self.step_body(
                    gpu,
                    weights,
                    config,
                    batch,
                    pool,
                    dn_states,
                    pbs,
                    s,
                    logits_out,
                    lm_head_skip,
                    Some(ctx_bucket),
                    None,
                )?;
                // The warm-up advanced the slot lengths; the capture body
                // advances them again, so roll back first.
                crate::forward_slots::rewind_slot_seq_lens(batch, pool)?;
                gpu.begin_stream_capture()?;
                let captured = self.step_body(
                    gpu,
                    weights,
                    config,
                    batch,
                    pool,
                    dn_states,
                    pbs,
                    s,
                    logits_out,
                    lm_head_skip,
                    Some(ctx_bucket),
                    None,
                );
                let hip_graph = gpu.end_stream_capture()?;
                captured?;
                let exec = gpu.hip.graph_instantiate(&hip_graph)?;
                graph.graph = Some(hip_graph);
                graph.exec = Some(exec);
                graph.key = Some(key);
                graph.captures += 1;
                // Capture does not execute; the warm-up already ran the step.
                return Ok(());
            }
            let exec = graph
                .exec
                .take()
                .expect("cache key set implies an instantiated exec");
            let launched = gpu.launch_graph(&exec);
            graph.exec = Some(exec);
            launched?;
            graph.replays += 1;
            // Host bookkeeping a replay cannot do for itself.
            advance_slot_seq_lens(batch, pool)?;
            Ok(())
        })();
        self.graph = graph;
        result
    }

    /// Prefill one slot's prompt chunk on the singleton body, against the
    /// slot's own VMM owner. Writes the chunk's last-token logits into the
    /// slot's `logits_out` row so the engine can sample the first generated
    /// token exactly as the serial route does.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_chunk(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        scratch: &Qwen35Scratch,
        slot: usize,
        tokens: &[u32],
        start_pos: usize,
        dn: &mut DeltaNetState,
        logits_out: &GpuTensor,
        hidden_out: Option<&GpuTensor>,
        extract_hidden: Option<ExtractHiddenCapture<'_>>,
    ) -> HipResult<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        if slot >= self.n_slots {
            return Err(HipError::new(0, "VMM slot prefill: slot out of range"));
        }
        let frontier = start_pos + tokens.len();
        self.check_frontier(slot, frontier)
            .map_err(|e| HipError::new(0, &e))?;
        // Error-feedback off means the recurrence uses the process-global
        // stochastic-rounding frame; give this slot its own frame sequence
        // (the same protocol as the VMM batch route's exact prefill).
        let saved_frame = self.gdn_frame[slot];
        let restore_after = dn.s_ef_residual.is_empty().then(|| {
            let g = rdna_compute::norm::gdn_requant_frame_checkpoint();
            rdna_compute::norm::restore_gdn_requant_frame_checkpoint(saved_frame);
            g
        });
        let owner = self.owners[slot]
            .as_mut()
            .ok_or_else(|| HipError::new(0, "VMM slot prefill: slot has no owner"))?;
        // The MTP head prefill consumes post-output-norm trunk hidden rows;
        // the capture variant writes exactly those (`tokens.len() * dim`) and
        // keeps KV, DeltaNet state and logits identical to the plain prefill.
        let result = match (hidden_out, extract_hidden) {
            (Some(_), Some(_)) => Err(HipError::new(
                0,
                "VMM slot prefill: MTP and DFlash hidden captures are mutually exclusive",
            )),
            (Some(hidden), None) => crate::qwen35::forward_prefill_batch_capture_hidden(
                gpu, weights, config, tokens, start_pos, owner, dn, scratch, hidden,
            ),
            (None, Some(extract)) => crate::qwen35::forward_prefill_batch_capture_extract_hidden(
                gpu, weights, config, tokens, start_pos, owner, dn, scratch, extract,
            ),
            (None, None) => crate::qwen35::forward_prefill_batch(
                gpu, weights, config, tokens, start_pos, owner, dn, scratch, None, None, None, None,
            ),
        };
        if let Some(g) = restore_after {
            self.gdn_frame[slot] = rdna_compute::norm::gdn_requant_frame_checkpoint();
            rdna_compute::norm::restore_gdn_requant_frame_checkpoint(g);
        }
        result?;
        let view = logits_out.sub_offset(slot * config.vocab_size, config.vocab_size);
        gpu.memcpy_dtod_at_auto(&view.buf, 0, &scratch.logits.buf, 0, config.vocab_size * 4)
    }

    /// Capture one slot's live KV prefix plus its recurrent-state buffers into
    /// a swap snapshot. Unlike the arena routes this is a flat copy: every
    /// owner tensor is a linear VMM reservation.
    pub fn capture_slot(
        &self,
        gpu: &mut Gpu,
        slot: usize,
        seq_len: usize,
        dn_extras: &[&GpuTensor],
        tokens: &[u32],
        stamp: SnapshotStamp,
    ) -> Result<SlotSnapshot, SwapError> {
        let owner = self.owners[slot]
            .as_ref()
            .ok_or_else(|| SwapError::Gpu("VMM capture: slot has no owner".into()))?;
        let per_pos_k = stamp.per_pos_bytes as usize;
        let per_pos_v = stamp.per_pos_v_bytes as usize;
        let span_k = seq_len * per_pos_k;
        let span_v = seq_len * per_pos_v;
        // K and V are separate owners/arenas: each span is bounded by its own
        // mapped extent. Comparing their SUM against one buffer refused every
        // session past half its mapped prefix (seq_len <= mapped_tokens / 2),
        // so long conversations were never parked and re-prefilled cold.
        // Every KV layer maps the same extent; check them all so a shorter
        // later layer cannot be over-read.
        for &layer in &self.kv_layer_ids {
            let k_bytes = owner.k_gpu[layer].buf.size();
            let v_bytes = owner.v_gpu[layer].buf.size();
            if span_k > k_bytes || span_v > v_bytes {
                return Err(SwapError::Corrupt(format!(
                    "VMM capture: layer {layer} seq_len {seq_len} is past the \
                     mapped prefix (K {span_k}>{k_bytes}, V {span_v}>{v_bytes})"
                )));
            }
        }
        let dn_total: usize = dn_extras.iter().map(|t| t.buf.size()).sum();
        let mut payload = vec![0u8; self.n_layers * (span_k + span_v) + dn_total];
        let mut off = 0usize;
        for &layer in &self.kv_layer_ids {
            for (tensor, span) in [(&owner.k_gpu[layer], span_k), (&owner.v_gpu[layer], span_v)] {
                gpu.hip
                    .memcpy_dtoh_at(&mut payload[off..off + span], &tensor.buf, 0)
                    .map_err(|e| SwapError::Gpu(e.to_string()))?;
                off += span;
            }
        }
        for t in dn_extras {
            let bytes = t.buf.size();
            gpu.hip
                .memcpy_dtoh(&mut payload[off..off + bytes], &t.buf)
                .map_err(|e| SwapError::Gpu(e.to_string()))?;
            off += bytes;
        }
        let checksum = checksum_of(&payload);
        Ok(SlotSnapshot {
            stamp,
            seq_len,
            tokens: tokens.to_vec(),
            payload,
            checksum,
        })
    }

    /// Restore a swap snapshot into `slot`, mapping to `end` first (the
    /// restored request's own future growth) so no map happens in a later step.
    #[allow(clippy::too_many_arguments)]
    pub fn restore_slot(
        &mut self,
        gpu: &mut Gpu,
        slot: usize,
        snap: &SlotSnapshot,
        expect: SnapshotStamp,
        dn_extras: &[&GpuTensor],
        end: usize,
    ) -> Result<(), SwapError> {
        snap.validate(expect)?;
        self.ensure_slot(gpu, slot, end.max(snap.seq_len))
            .map_err(SwapError::Gpu)?;
        let owner = self.owners[slot]
            .as_mut()
            .ok_or_else(|| SwapError::Gpu("VMM restore: slot has no owner".into()))?;
        let per_pos_k = snap.stamp.per_pos_bytes as usize;
        let per_pos_v = snap.stamp.per_pos_v_bytes as usize;
        let span_k = snap.seq_len * per_pos_k;
        let span_v = snap.seq_len * per_pos_v;
        let dn_total: usize = dn_extras.iter().map(|t| t.buf.size()).sum();
        if self.n_layers * (span_k + span_v) + dn_total != snap.payload.len() {
            return Err(SwapError::Corrupt(
                "VMM restore: payload does not match this rig's KV and state sizes".to_string(),
            ));
        }
        let mut off = 0usize;
        for &layer in &self.kv_layer_ids {
            for (tensor, span) in [(&owner.k_gpu[layer], span_k), (&owner.v_gpu[layer], span_v)] {
                gpu.hip
                    .memcpy_htod_offset(&tensor.buf, 0, &snap.payload[off..off + span])
                    .map_err(|e| SwapError::Gpu(e.to_string()))?;
                off += span;
            }
        }
        for t in dn_extras {
            let bytes = t.buf.size();
            gpu.hip
                .memcpy_htod(&t.buf, &snap.payload[off..off + bytes])
                .map_err(|e| SwapError::Gpu(e.to_string()))?;
            off += bytes;
        }
        Ok(())
    }

    /// A `KvCache` view of this slot's own VMM owner, for the sequential VL
    /// writer (`vl_forward_remaining`). Tensors are shallow clones — writes
    /// land in the owner, which stays alive in this store. `None` when the
    /// slot has no owner.
    pub fn slot_view(&self, slot: usize) -> Option<KvCache> {
        let owner = self.owners.get(slot)?.as_ref()?;
        Some(KvCache {
            k_gpu: owner.k_gpu.iter().map(|t| t.shallow_clone()).collect(),
            v_gpu: owner.v_gpu.iter().map(|t| t.shallow_clone()).collect(),
            k_scales: Vec::new(),
            v_scales: Vec::new(),
            kv_dim: owner.kv_dim,
            max_seq: owner.max_seq,
            physical_cap: owner.physical_cap,
            n_kv_heads: owner.n_kv_heads,
            head_dim: owner.head_dim,
            quantized: owner.quantized,
            quant_q8: owner.quant_q8,
            quant_int8: owner.quant_int8,
            quant_hfq4: owner.quant_hfq4,
            quant_asym4: owner.quant_asym4,
            quant_asym3: owner.quant_asym3,
            quant_asym2: owner.quant_asym2,
            quant_fwht: owner.quant_fwht,
            quant_bf16: owner.quant_bf16,
            quant_fp8: owner.quant_fp8,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: Vec::new(),
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
    }

    /// Owner generation for `slot` (descriptor binding / diagnostics).
    pub fn owner_generation(&self, gpu: &Gpu, slot: usize) -> Option<u64> {
        let owner = self.owners.get(slot)?.as_ref()?;
        owner.vmm_owner_generation(gpu).ok()
    }

    /// Free every owner and staging buffer. Called only at engine teardown.
    pub fn free_gpu(mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.graph.release(gpu);
        let mut first = None;
        for t in self
            .descs_dev
            .into_iter()
            .chain([self.row_slot_dev, self.flash_partials])
        {
            if let Err(e) = gpu.free_tensor(t) {
                first.get_or_insert(e);
            }
        }
        for owner in self.owners.into_iter().flatten() {
            if let Err(e) = owner.free_gpu(gpu) {
                first.get_or_insert(e);
            }
        }
        match first {
            Some(e) => Err(format!("VMM slot KV teardown: {e}")),
            None => Ok(()),
        }
    }
}

/// Device budget a fresh VMM slot store may hand to KV maps: live free VRAM
/// less the reserve the prefill/graph machinery needs.
pub fn vmm_slot_budget(gpu: &Gpu) -> usize {
    gpu.device_mem_info()
        .map(|(free, _)| free.saturating_sub(VMM_MAP_DEVICE_RESERVE_BYTES))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vmm_format_gate_admits_only_descriptor_tiers() {
        assert_eq!(vmm_format_for_mode(KvMode::Q8), Ok(VmmKvFormat::Q8));
        assert_eq!(vmm_format_for_mode(KvMode::Fp8), Ok(VmmKvFormat::Fp8E4M3));
        for mode in [
            KvMode::Asym2,
            KvMode::Asym3,
            KvMode::Asym4,
            KvMode::Fwht2,
            KvMode::Fwht3,
            KvMode::Fwht4,
            KvMode::Bf16,
            KvMode::F16,
        ] {
            let err =
                vmm_format_for_mode(mode).expect_err("rotated/flat tiers have no _vmm kernel");
            assert!(err.contains("legacy"), "refusal must name the fix: {err}");
        }
    }

    #[test]
    fn reservation_covers_prompt_budget_and_slack_within_the_cap() {
        // prompt + max_tokens + slack.
        assert_eq!(slot_reserve_end(100, 50, 8192), 214);
        // Clamped to the context cap.
        assert_eq!(slot_reserve_end(100, 50, 200), 200);
        assert_eq!(slot_reserve_end(8190, 4096, 8192), 8192);
        // A zero generation budget still reserves one token.
        assert_eq!(slot_reserve_end(1, 0, 1024), 66);
        // Never zero, even for a degenerate cap.
        assert_eq!(slot_reserve_end(0, 0, 1), 1);
    }

    #[test]
    fn graph_key_distinguishes_every_shape_that_changes_launched_kernels() {
        let base = vmm_decode_graph_m_hash(&[1, 1, 0], &[false, false, false]);
        assert_eq!(
            base,
            vmm_decode_graph_m_hash(&[1, 1, 0], &[false, false, false]),
            "the same composition must produce the same key"
        );
        assert_ne!(
            base,
            vmm_decode_graph_m_hash(&[1, 0, 1], &[false, false, false]),
            "a different active-slot set launches a different body"
        );
        assert_ne!(
            base,
            vmm_decode_graph_m_hash(&[1, 1, 0], &[true, false, false]),
            "the lm_head-skip mask changes which GEMVs exist"
        );
        assert_ne!(
            base,
            vmm_decode_graph_m_hash(&[1, 1, 0, 1], &[false, false, false, false]),
            "row count is part of the shape"
        );
    }

    #[test]
    fn vmm_snapshot_tag_is_distinct_from_every_arena_tag() {
        // Arena tags are small discriminants; the VMM marker must flip a bit
        // above all of them so the same KvMode cannot restore across routes.
        for arena_tag in 1u32..=0x40 {
            let vmm_tag = arena_tag | VMM_SNAPSHOT_TAG_BIT;
            assert_ne!(vmm_tag, arena_tag);
            assert_eq!(vmm_tag & VMM_SNAPSHOT_TAG_BIT, VMM_SNAPSHOT_TAG_BIT);
        }
    }
}
