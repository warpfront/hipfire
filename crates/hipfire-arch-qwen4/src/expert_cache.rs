//! Decode-only exact LRU of immutable Flash-Next expert payloads.
//!
//! A token uses the tables acquired at its start. Misses execute from immutable
//! mapped host backing; asynchronous copies serve later tokens. Dirty short
//! prefill reads independent immutable core/host tables. Long prefill restores
//! the original donor bytes before using the frozen resident view.

use crate::expert_cache_config::ExpertCacheConfig;
use crate::expert_slabs::{EXPERTS_PER_LAYER, TRUNK_LAYERS};
use crate::expert_tier_owners::{ExpertCacheSlot, ExpertCacheTableLease, ExpertTierLease};
use crate::expert_tier_plan::Placement;
use hip_bridge::{DeviceBuffer, Event, HipRuntime, Stream, HIP_EVENT_DISABLE_TIMING};
use hipfire_dispatch::families::moe::{
    MutableExpertResidency, MutableExpertResidencySpec, ResidencyLocation, ResidencyPublication,
    ResidencySlot, ResidencySlotTicket, ResidencySnapshot, ResidencySpan, ResidencyTokenGuard,
};
use hipfire_runtime::weight_pager::{SlotCache, SlotCacheError, SlotReservation};
use rdna_compute::Gpu;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Instant;

const EXPERTS: usize = EXPERTS_PER_LAYER as usize;
const LAYERS: usize = TRUNK_LAYERS as usize;
const TOPK: usize = 10;
type Result<T> = std::result::Result<T, String>;
type Reservation = SlotReservation<usize, ()>;

/// Descriptors returned by tables are borrowed and must never be freed.
#[derive(Clone)]
pub(crate) struct ExpertCacheLayerBinding {
    pub tables: ExpertCacheTableLease,
    pub residency: MutableExpertResidency,
}

/// An unretired token poisons every capability and prevents freeing its owners.
pub(crate) struct ExpertCacheToken {
    guards: Vec<ResidencyTokenGuard>,
    outstanding: Arc<AtomicBool>,
}
impl ExpertCacheToken {
    /// # Safety
    /// All GPU reads under this token have completed, including PM4 work.
    /// A failed HIP wait cannot establish quiescence of an independent PM4 queue.
    pub(crate) unsafe fn retire_after_gpu_wait(self) -> Result<()> {
        let mut first_error = None;
        for guard in self.guards {
            // SAFETY: caller proves retirement of the entire token.
            if let Err(error) = unsafe { guard.retire_after_gpu_wait() } {
                first_error.get_or_insert(error);
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        self.outstanding.store(false, Ordering::Release);
        Ok(())
    }
}

/// Cumulative logical byte counters and host-observed wall durations. Timings
/// cover these manager methods, not the caller's route readback or GPU step.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(crate) struct ExpertCacheStats {
    pub tokens: u64,
    pub selected: u64,
    pub core_hits: u64,
    pub cache_hits: u64,
    pub host_misses: u64,
    pub miss_bytes: u64,
    pub fills_started: u64,
    pub fills_completed: u64,
    pub fill_bytes: u64,
    pub evictions: u64,
    pub skipped_full: u64,
    /// Calls and rows admitted to the independent immutable mapped-prefill
    /// view. Admission does not prove GPU execution or measure GPU time.
    pub mapped_prefill_calls: u64,
    pub mapped_prefill_rows: u64,
    pub prefill_restores: u64,
    pub restore_bytes: u64,
    pub peak_inflight: usize,
    /// Cumulative wall time, including HIP waits. These enclosing timings
    /// overlap publish_wall_ns and must not be summed as CPU time.
    pub begin_token_wall_ns: u64,
    pub finish_token_wall_ns: u64,
    /// Pair-publication wall time includes validation, upload and visibility wait.
    pub publish_wall_ns: u64,
    /// Publication bytes successfully enqueued; excludes initial canonical table
    /// construction and does not represent measured PCIe traffic.
    pub table_upload_bytes: u64,
    /// Attempted GU/DN publication transactions, including failed attempts.
    pub publish_calls: u64,
    /// Whole prepare_prefill wall time when entering with dirty donor bytes:
    /// drain, mapping withdrawal, restoration and waits, including failed attempts.
    pub prefill_restore_wall_ns: u64,
}

#[derive(Clone, Copy)]
struct HostPair {
    gate_up: (usize, usize),
    down: (usize, usize),
}
struct LayerCache {
    binding: ExpertCacheLayerBinding,
    slots: Vec<ExpertCacheSlot>,
    host: Vec<Option<HostPair>>,
    cache: Option<SlotCache<usize, ()>>,
    // Retain upload mirrors even when the second upload fails after the first.
    gate_up_upload: Vec<u8>,
    down_upload: Vec<u8>,
}
struct PendingFill {
    layer: usize,
    reservation: Reservation,
    done: Event,
}

pub(crate) struct ExpertCache {
    hip: Arc<HipRuntime>,
    device_id: i32,
    lease: Option<ExpertTierLease>,
    layers: Vec<LayerCache>,
    copy_stream: Option<Stream>,
    retire_event: Option<Event>,
    pending: VecDeque<PendingFill>,
    idle_events: Vec<Event>,
    max_inflight: usize,
    warm_order: Vec<Vec<usize>>,
    needs_seed: bool,
    dirty: bool,
    poisoned: Option<String>,
    outstanding: Arc<AtomicBool>,
    stats: ExpertCacheStats,
    next_layer: usize,
}
fn elapsed_wall_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn error(context: &str, value: impl std::fmt::Display) -> String {
    format!("exact expert cache {context}: {value}")
}
fn host_projection(
    alias: usize,
    bytes: usize,
    host_alias: (usize, usize),
    host_cpu: (usize, usize),
) -> Result<(usize, usize)> {
    if host_alias.1 != host_cpu.1 || bytes == 0 {
        return Err("expert cache invalid host owner extent".into());
    }
    let offset = alias
        .checked_sub(host_alias.0)
        .ok_or("expert cache host alias precedes owner")?;
    if offset
        .checked_add(bytes)
        .is_none_or(|end| end > host_alias.1)
    {
        return Err("expert cache host projection exceeds owner".into());
    }
    let cpu = host_cpu
        .0
        .checked_add(offset)
        .ok_or("expert cache host CPU address overflow")?;
    cpu.checked_add(bytes)
        .ok_or("expert cache host CPU end overflow")?;
    Ok((cpu, bytes))
}
/// Preserve selected order within each layer while giving every nonempty
/// layer its first fill before any layer receives a second one.
fn fair_misses(misses: &[Vec<usize>], start: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
    let layers = misses.len();
    let longest = misses.iter().map(Vec::len).max().unwrap_or(0);
    (0..longest).flat_map(move |ordinal| {
        (0..layers).filter_map(move |offset| {
            let layer = (start + offset) % layers;
            misses[layer]
                .get(ordinal)
                .copied()
                .map(|expert| (layer, expert))
        })
    })
}

fn warm_order(rank: &[u32]) -> Result<Vec<Vec<usize>>> {
    if rank.len() != LAYERS * EXPERTS {
        return Err("expert cache rank cardinality".into());
    }
    let mut seen = vec![false; rank.len()];
    let mut layers = (0..LAYERS)
        .map(|_| Vec::with_capacity(EXPERTS))
        .collect::<Vec<_>>();
    // Touch cold to hot, leaving the hottest resident donor most recent.
    for &id in rank.iter().rev() {
        let id = id as usize;
        if id >= seen.len() || std::mem::replace(&mut seen[id], true) {
            return Err("expert cache rank is not a permutation".into());
        }
        layers[id / EXPERTS].push(id % EXPERTS);
    }
    Ok(layers)
}
/// Count what the completed token actually read, independent of later LRU
/// reservations or copy completion. Reject malformed selection before counting.
fn account_routes(
    stats: &mut ExpertCacheStats,
    ids: &[i32],
    host: &[Option<HostPair>],
    mapping: &[ResidencyLocation],
) -> Result<()> {
    if host.len() != EXPERTS || mapping.len() != EXPERTS {
        return Err("expert cache route-accounting cardinality".into());
    }
    let mut seen = [false; EXPERTS];
    for &id in ids {
        let expert = usize::try_from(id).map_err(|_| "expert cache negative selected expert")?;
        if expert >= EXPERTS || std::mem::replace(&mut seen[expert], true) {
            return Err("expert cache invalid/duplicate selected expert".into());
        }
    }
    for &id in ids {
        let expert = id as usize;
        stats.selected = stats.selected.saturating_add(1);
        match host[expert] {
            None => stats.core_hits = stats.core_hits.saturating_add(1),
            Some(_) if matches!(mapping[expert], ResidencyLocation::Slot(_)) => {
                stats.cache_hits = stats.cache_hits.saturating_add(1);
            }
            Some(pair) => {
                stats.host_misses = stats.host_misses.saturating_add(1);
                stats.miss_bytes = stats
                    .miss_bytes
                    .saturating_add((pair.gate_up.1 + pair.down.1) as u64);
            }
        }
    }
    Ok(())
}

impl ExpertCache {
    pub(crate) fn new(
        gpu: &mut Gpu,
        lease: ExpertTierLease,
        config: ExpertCacheConfig,
        rank: &[u32],
    ) -> Result<Self> {
        if gpu.arch != "gfx1201" || config.max_inflight == 0 {
            return Err("expert cache requires gfx1201 and a bounded nonzero queue".into());
        }
        gpu.bind_thread().map_err(|e| error("bind", e))?;
        let warm_order = warm_order(rank)?;
        let ranges = lease
            .expert_cache_ranges()
            .ok_or("expert cache missing donor mirror")?;
        if ranges.donor_device.1 == 0 || ranges.donor_device.1 != ranges.mirror_cpu.1 {
            return Err("expert cache invalid restore extent".into());
        }
        let host_alias = lease
            .owner_range(Placement::HostMapped)
            .ok_or("expert cache missing host alias")?;
        let host_cpu = lease
            .host_cpu_range()
            .ok_or("expert cache missing host owner")?;
        let hip = Arc::new(HipRuntime::load().map_err(|e| error("HIP", e))?);
        hip.set_device(gpu.device_id)
            .map_err(|e| error("device", e))?;
        let mut result = Self {
            hip,
            device_id: gpu.device_id,
            lease: Some(lease.clone()),
            layers: Vec::with_capacity(LAYERS),
            copy_stream: None,
            retire_event: None,
            pending: VecDeque::new(),
            idle_events: Vec::with_capacity(config.max_inflight),
            max_inflight: config.max_inflight,
            warm_order,
            needs_seed: true,
            dirty: false,
            poisoned: None,
            outstanding: Arc::new(AtomicBool::new(false)),
            stats: ExpertCacheStats::default(),
            next_layer: 0,
        };
        for layer in 0..LAYERS {
            let view = lease
                .expert_cache_layer(layer)
                .ok_or("expert cache missing canonical layer")?;
            let slots = lease
                .expert_cache_slots(layer)
                .ok_or("expert cache missing slot metadata")?;
            let mut canonical_gate_up: Vec<ResidencySpan> = Vec::with_capacity(EXPERTS);
            let mut canonical_down: Vec<ResidencySpan> = Vec::with_capacity(EXPERTS);
            let mut host = Vec::with_capacity(EXPERTS);
            for expert in 0..EXPERTS {
                let entry = view.entry(expert).ok_or("expert cache missing expert")?;
                let gu = ResidencySpan {
                    address: view.gate_up_ptrs()[expert],
                    bytes: usize::try_from(entry.gate_up_len)
                        .map_err(|_| "expert cache GU length overflow")?,
                };
                let dn = ResidencySpan {
                    address: view.down_ptrs()[expert],
                    bytes: usize::try_from(entry.down_len)
                        .map_err(|_| "expert cache DN length overflow")?,
                };
                if expert > 0
                    && (gu.bytes != canonical_gate_up[0].bytes
                        || dn.bytes != canonical_down[0].bytes)
                {
                    return Err(
                        "expert cache requires equal projection extents within a layer".into(),
                    );
                }
                canonical_gate_up.push(gu);
                canonical_down.push(dn);
                host.push(if entry.tier == Placement::HostMapped {
                    Some(HostPair {
                        gate_up: host_projection(gu.address, gu.bytes, host_alias, host_cpu)?,
                        down: host_projection(dn.address, dn.bytes, host_alias, host_cpu)?,
                    })
                } else {
                    None
                });
            }
            let tables = lease
                .allocate_expert_cache_tables(gpu, layer)
                .map_err(|e| error("tables", e))?;
            let gu = tables.gate_up_table();
            let dn = tables.down_table();
            let slot_spans = slots
                .iter()
                .map(|slot| ResidencySlot {
                    gate_up: ResidencySpan {
                        address: slot.ranges.gate_up_device.0,
                        bytes: slot.ranges.gate_up_device.1,
                    },
                    down: ResidencySpan {
                        address: slot.ranges.down_device.0,
                        bytes: slot.ranges.down_device.1,
                    },
                })
                .collect();
            // SAFETY: leases retain every checked allocation. Table allocation
            // uploaded and synchronized canonical addresses; no token exists.
            let residency = unsafe {
                MutableExpertResidency::new(MutableExpertResidencySpec {
                    owner_generation: lease.generation(),
                    canonical_gate_up,
                    canonical_down,
                    gate_up_table: ResidencySpan {
                        address: gu.buf.as_ptr() as usize,
                        bytes: gu.buf.size(),
                    },
                    down_table: ResidencySpan {
                        address: dn.buf.as_ptr() as usize,
                        bytes: dn.buf.size(),
                    },
                    slots: slot_spans,
                    keepalive: tables.keepalive(),
                })
            }?;
            let cache = if slots.is_empty() {
                None
            } else {
                Some(SlotCache::new(slots.len()).map_err(|e| error("slots", e))?)
            };
            result.layers.push(LayerCache {
                binding: ExpertCacheLayerBinding { tables, residency },
                slots,
                host,
                cache,
                gate_up_upload: vec![0; EXPERTS * 8],
                down_upload: vec![0; EXPERTS * 8],
            });
        }
        result.copy_stream = Some(
            result
                .hip
                .stream_create_non_blocking()
                .map_err(|e| error("copy stream", e))?,
        );
        result.retire_event = Some(
            result
                .hip
                .event_create_with_flags(HIP_EVENT_DISABLE_TIMING)
                .map_err(|e| error("retire event", e))?,
        );
        for _ in 0..config.max_inflight {
            result.idle_events.push(
                result
                    .hip
                    .event_create_with_flags(HIP_EVENT_DISABLE_TIMING)
                    .map_err(|e| error("fill event pool", e))?,
            );
        }
        Ok(result)
    }
    pub(crate) fn layer_binding(&self, layer: usize) -> Option<ExpertCacheLayerBinding> {
        self.layers.get(layer).map(|layer| layer.binding.clone())
    }
    pub(crate) fn stats(&self) -> ExpertCacheStats {
        self.stats
    }
    pub(crate) fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Check readiness for a prefill that reads only the independent immutable
    /// canonical core/device and donor/host view. The caller must disable the
    /// Resident staging path and the retained decode route for this view.
    ///
    /// Copies may keep filling donor slots: neither those slots nor the mutable
    /// decode tables are read here. Keep pending tickets, LRU, mappings and dirty
    /// state intact so later decode publication or long-prefill restoration uses
    /// the same physical ownership and generation history.
    pub(crate) fn prepare_host_prefill(&mut self, gpu: &mut Gpu, rows: usize) -> Result<()> {
        self.check(gpu)?;
        if rows == 0 {
            return Err("expert cache mapped prefill requires at least one row".into());
        }
        self.stats.mapped_prefill_calls = self.stats.mapped_prefill_calls.saturating_add(1);
        self.stats.mapped_prefill_rows = self
            .stats
            .mapped_prefill_rows
            .saturating_add(u64::try_from(rows).unwrap_or(u64::MAX));
        Ok(())
    }

    fn emit_stats(&self) {
        match serde_json::to_string(&self.stats) {
            Ok(json) => eprintln!("  qwen4 exact expert cache stats: {json}"),
            Err(error) => {
                eprintln!("  qwen4 exact expert cache stats serialization failed: {error}")
            }
        }
    }

    fn check(&self, gpu: &Gpu) -> Result<()> {
        if let Some(reason) = &self.poisoned {
            return Err(error("poisoned", reason));
        }
        if self.outstanding.load(Ordering::Acquire) {
            return Err("expert cache token is unretired".into());
        }
        if gpu.device_id != self.device_id || self.copy_stream.is_none() {
            return Err("expert cache stopped or used on another GPU".into());
        }
        gpu.bind_thread().map_err(|e| error("bind", e))
    }
    pub(crate) fn begin_token(&mut self, gpu: &mut Gpu) -> Result<ExpertCacheToken> {
        let started = Instant::now();
        let result = self.begin_token_inner(gpu);
        self.stats.begin_token_wall_ns = self
            .stats
            .begin_token_wall_ns
            .saturating_add(elapsed_wall_ns(started));
        result
    }

    fn begin_token_inner(&mut self, gpu: &mut Gpu) -> Result<ExpertCacheToken> {
        self.check(gpu)?;
        let result: Result<_> = (|| {
            if self.needs_seed {
                self.seed(gpu)?;
            }
            self.publish_completed(gpu)?;
            let mut guards = Vec::with_capacity(LAYERS);
            for layer in &self.layers {
                let cap = &layer.binding.residency;
                match cap.begin_token(cap.snapshot()?.epoch) {
                    Ok(guard) => guards.push(guard),
                    Err(error) => {
                        // SAFETY: no launch has used any newly acquired guard.
                        for guard in guards {
                            unsafe { guard.retire_after_gpu_wait() }?;
                        }
                        return Err(error);
                    }
                }
            }
            self.outstanding.store(true, Ordering::Release);
            Ok(ExpertCacheToken {
                guards,
                outstanding: self.outstanding.clone(),
            })
        })();
        if let Err(reason) = &result {
            self.poisoned = Some(reason.clone());
        }
        result
    }
    /// # Safety
    /// Every GPU read under token has retired, including independent PM4 work.
    /// IDs are this tokens complete 48x10 device readback in selected order.
    pub(crate) unsafe fn finish_token(
        &mut self,
        gpu: &mut Gpu,
        token: ExpertCacheToken,
        ids: &[i32],
    ) -> Result<()> {
        let started = Instant::now();
        // SAFETY: the caller proves retirement of all token GPU reads, exactly
        // as required by finish_token_inner before releasing its guards.
        let result = unsafe { self.finish_token_inner(gpu, token, ids) };
        self.stats.finish_token_wall_ns = self
            .stats
            .finish_token_wall_ns
            .saturating_add(elapsed_wall_ns(started));
        if result.is_ok() && self.stats.tokens.is_multiple_of(64) {
            self.emit_stats();
        }
        result
    }

    unsafe fn finish_token_inner(
        &mut self,
        gpu: &mut Gpu,
        token: ExpertCacheToken,
        ids: &[i32],
    ) -> Result<()> {
        if !Arc::ptr_eq(&token.outstanding, &self.outstanding) {
            return Err("expert cache token belongs to another cache".into());
        }
        if gpu.device_id != self.device_id {
            return Err("expert cache token finished on another GPU".into());
        }
        gpu.bind_thread().map_err(|e| error("finish bind", e))?;
        let valid =
            ids.len() == LAYERS * TOPK && ids.iter().all(|id| (0..EXPERTS as i32).contains(id));
        let snapshots: Vec<_> = token
            .guards
            .iter()
            .map(|guard| guard.snapshot().clone())
            .collect();
        // SAFETY: caller proves all submitted token work has completed.
        unsafe { token.retire_after_gpu_wait() }?;
        if !valid {
            self.poisoned = Some("invalid route readback".into());
            return Err("expert cache invalid route readback".into());
        }
        let result = self.observe_and_fill(gpu, ids, &snapshots);
        if let Err(reason) = &result {
            self.poisoned = Some(reason.clone());
        }
        result
    }
    fn observe_and_fill(
        &mut self,
        gpu: &Gpu,
        ids: &[i32],
        snapshots: &[ResidencySnapshot],
    ) -> Result<()> {
        self.stats.tokens = self.stats.tokens.saturating_add(1);
        let mut misses = vec![Vec::new(); LAYERS];
        for layer in 0..LAYERS {
            let state = &mut self.layers[layer];
            let selected = &ids[layer * TOPK..(layer + 1) * TOPK];
            account_routes(
                &mut self.stats,
                selected,
                &state.host,
                &snapshots[layer].mapping,
            )?;
            // Touch every cached/pending hit before reserving any miss.
            for &id in selected {
                let expert = id as usize;
                if state.host[expert].is_none() {
                    continue;
                }
                if let Some(cache) = &mut state.cache {
                    if cache.resident_slot(expert).is_some()
                        || self
                            .pending
                            .iter()
                            .any(|fill| fill.layer == layer && fill.reservation.key == expert)
                    {
                        cache.reserve((), expert).map_err(|e| error("touch", e))?;
                    } else {
                        misses[layer].push(expert);
                    }
                }
            }
        }
        let mut fills = Vec::new();
        let mut mappings: Vec<_> = snapshots.iter().map(|s| s.mapping.to_vec()).collect();
        let mut changed = [false; LAYERS];
        // A layer-first pass lets a few miss-heavy layers consume the whole
        // bounded queue. Interleave layers instead; advance past the last
        // successful fill so the extra quota rotates fairly on the next token.
        let mut last_served = None;
        for (layer, expert) in fair_misses(&misses, self.next_layer) {
            let cache = self.layers[layer]
                .cache
                .as_mut()
                .expect("misses require slots");
            if self.pending.len() + fills.len() >= self.max_inflight {
                self.stats.skipped_full = self.stats.skipped_full.saturating_add(1);
                continue;
            }
            let reservation = match cache.reserve((), expert) {
                Ok(reservation) => reservation,
                Err(SlotCacheError::AllSlotsPending) => {
                    self.stats.skipped_full = self.stats.skipped_full.saturating_add(1);
                    continue;
                }
                Err(e) => return Err(error("reserve", e)),
            };
            if !reservation.needs_fill() {
                continue;
            }
            if let Some(victim) = reservation.victim() {
                mappings[layer][victim] = ResidencyLocation::Canonical;
                changed[layer] = true;
                self.stats.evictions = self.stats.evictions.saturating_add(1);
            }
            fills.push((layer, reservation));
            last_served = Some(layer);
        }
        self.next_layer = (last_served.unwrap_or(self.next_layer) + 1) % LAYERS;
        if fills.is_empty() {
            return Ok(());
        }
        for layer in 0..LAYERS {
            if changed[layer] {
                self.publish(gpu, layer, std::mem::take(&mut mappings[layer]), Vec::new())?;
            }
        }
        // Compute-stream publication precedes this event. An explicit copy-stream
        // dependency retires old pointers before SDMA overwrites a victim slot.
        let retire = self.retire_event.as_ref().expect("live retire event");
        let stream = self.copy_stream.as_ref().expect("live copy stream");
        self.hip
            .event_record(retire, gpu.active_stream.as_ref())
            .map_err(|e| error("retire record", e))?;
        self.hip
            .stream_wait_event(stream, retire)
            .map_err(|e| error("retire wait", e))?;
        for (layer, reservation) in fills {
            let state = &self.layers[layer];
            let source = state.host[reservation.key]
                .ok_or("expert cache attempted to replace immutable core")?;
            let slot = state.slots[reservation.slot].ranges;
            if source.gate_up.1 != slot.gate_up_device.1 || source.down.1 != slot.down_device.1 {
                return Err("expert cache source/slot extent mismatch".into());
            }
            let done = self
                .idle_events
                .pop()
                .ok_or("expert cache exhausted bounded event pool")?;
            // Install ownership before enqueue, including partial-pair errors.
            self.pending.push_back(PendingFill {
                layer,
                reservation,
                done,
            });
            self.dirty = true;
            self.copy_projection(source.gate_up, slot.gate_up_device, stream)?;
            self.copy_projection(source.down, slot.down_device, stream)?;
            let event = &self.pending.back().expect("installed pending fill").done;
            self.hip
                .event_record(event, Some(stream))
                .map_err(|e| error("fill record", e))?;
            self.stats.fills_started = self.stats.fills_started.saturating_add(1);
            self.stats.fill_bytes = self
                .stats
                .fill_bytes
                .saturating_add((source.gate_up.1 + source.down.1) as u64);
            self.stats.peak_inflight = self.stats.peak_inflight.max(self.pending.len());
        }
        Ok(())
    }
    fn copy_projection(
        &self,
        source: (usize, usize),
        destination: (usize, usize),
        stream: &Stream,
    ) -> Result<()> {
        if source.1 != destination.1 {
            return Err("expert cache unequal copy extents".into());
        }
        // SAFETY: the pinned immutable CPU source and the slot belong to checked
        // tier ranges. The owning lease outlives every pending event.
        let bytes = unsafe { std::slice::from_raw_parts(source.0 as *const u8, source.1) };
        let dst = unsafe { DeviceBuffer::from_raw(destination.0 as *mut c_void, destination.1) };
        self.hip
            .memcpy_htod_async(&dst, bytes, stream)
            .map_err(|e| error("payload copy", e))
    }
    fn publish_completed(&mut self, gpu: &Gpu) -> Result<()> {
        let mut completed = Vec::new();
        let mut query_error = None;
        // One FIFO copy stream: stop querying at its first incomplete pair.
        while let Some(front) = self.pending.front() {
            match self.hip.event_query(&front.done) {
                Ok(true) => completed.push(self.pending.pop_front().expect("queried front")),
                Ok(false) => break,
                Err(e) => {
                    query_error = Some(error("fill query", e));
                    break;
                }
            }
        }
        let result: Result<()> = (|| {
            if let Some(error) = query_error {
                return Err(error);
            }
            for layer in 0..LAYERS {
                let own: Vec<_> = completed
                    .iter()
                    .filter(|fill| fill.layer == layer)
                    .collect();
                if own.is_empty() {
                    continue;
                }
                let mut mapping = self.layers[layer]
                    .binding
                    .residency
                    .snapshot()?
                    .mapping
                    .to_vec();
                let tickets: Vec<_> = own
                    .iter()
                    .map(|fill| ResidencySlotTicket {
                        slot: fill.reservation.slot,
                        expert: fill.reservation.key,
                        generation: fill.reservation.generation(),
                    })
                    .collect();
                for &ticket in &tickets {
                    mapping[ticket.expert] = ResidencyLocation::Slot(ticket);
                }
                self.publish(gpu, layer, mapping, tickets)?;
                let cache = self.layers[layer]
                    .cache
                    .as_mut()
                    .expect("pending requires cache");
                for fill in own {
                    if !cache.commit(fill.reservation) {
                        return Err("expert cache stale fill commit".into());
                    }
                    self.stats.fills_completed = self.stats.fills_completed.saturating_add(1);
                }
            }
            Ok(())
        })();
        // These events are proven complete, including on a later query/upload
        // failure. Keep their owners for orderly shutdown instead of leaking
        // handles as a local error unwinds. The manager then fails closed.
        for fill in completed {
            self.idle_events.push(fill.done);
        }
        result
    }
    fn publish(
        &mut self,
        gpu: &Gpu,
        layer: usize,
        mapping: Vec<ResidencyLocation>,
        completed_slots: Vec<ResidencySlotTicket>,
    ) -> Result<()> {
        let started = Instant::now();
        self.stats.publish_calls = self.stats.publish_calls.saturating_add(1);
        let result = self.publish_inner(gpu, layer, mapping, completed_slots);
        self.stats.publish_wall_ns = self
            .stats
            .publish_wall_ns
            .saturating_add(elapsed_wall_ns(started));
        result
    }

    fn publish_inner(
        &mut self,
        gpu: &Gpu,
        layer: usize,
        mapping: Vec<ResidencyLocation>,
        completed_slots: Vec<ResidencySlotTicket>,
    ) -> Result<()> {
        let state = &mut self.layers[layer];
        let cap = state.binding.residency.clone();
        let epoch = cap.snapshot()?.epoch;
        let gu = state.binding.tables.gate_up_table();
        let dn = state.binding.tables.down_table();
        // SAFETY: no token guards are active, and prior reads retired. Completed
        // tickets passed event_query or name restored original donor payloads.
        // Both tables are uploaded and synchronized before host publication.
        unsafe {
            cap.publish_after_gpu_wait(
                ResidencyPublication {
                    expected_epoch: epoch,
                    mapping,
                    completed_slots,
                },
                |snapshot| {
                    for (pointers, upload, table) in [
                        (&snapshot.gate_up, &mut state.gate_up_upload, &gu),
                        (&snapshot.down, &mut state.down_upload, &dn),
                    ] {
                        for (&pointer, bytes) in pointers.iter().zip(upload.chunks_exact_mut(8)) {
                            bytes.copy_from_slice(&(pointer as u64).to_le_bytes());
                        }
                        self.hip
                            .memcpy_htod_async_on(&table.buf, upload, gpu.active_stream.as_ref())
                            .map_err(|e| error("table upload", e))?;
                        self.stats.table_upload_bytes = self
                            .stats
                            .table_upload_bytes
                            .saturating_add(u64::try_from(upload.len()).unwrap_or(u64::MAX));
                    }
                    match gpu.active_stream.as_ref() {
                        Some(stream) => self.hip.stream_synchronize(stream),
                        None => self.hip.stream_synchronize_default(),
                    }
                    .map_err(|e| error("table visibility", e))
                },
            )
        }?;
        Ok(())
    }
    fn seed(&mut self, gpu: &Gpu) -> Result<()> {
        for layer in 0..LAYERS {
            let state = &mut self.layers[layer];
            let Some(cache) = &mut state.cache else {
                continue;
            };
            let mut mapping = vec![ResidencyLocation::Canonical; EXPERTS];
            let mut tickets = Vec::with_capacity(state.slots.len());
            for (slot, payload) in state.slots.iter().enumerate() {
                let expert = payload.ranges.original_expert;
                let reservation = cache
                    .reserve((), expert)
                    .map_err(|e| error("seed reserve", e))?;
                if !reservation.needs_fill()
                    || reservation.slot != slot
                    || !cache.commit(reservation)
                {
                    return Err("expert cache warm seed slot mismatch".into());
                }
                let ticket = ResidencySlotTicket {
                    slot,
                    expert,
                    generation: reservation.generation(),
                };
                mapping[expert] = ResidencyLocation::Slot(ticket);
                tickets.push(ticket);
            }
            for &expert in &self.warm_order[layer] {
                if cache.resident_slot(expert).is_some() {
                    cache
                        .reserve((), expert)
                        .map_err(|e| error("seed recency", e))?;
                }
            }
            self.publish(gpu, layer, mapping, tickets)?;
        }
        self.needs_seed = false;
        Ok(())
    }
    /// Drain cache DMA, retire slot mappings, and restore the original contiguous
    /// donor tail before frozen prefill. Never reset the slot generation counter.
    pub(crate) fn prepare_prefill(&mut self, gpu: &mut Gpu) -> Result<()> {
        let started = Instant::now();
        let was_dirty = self.dirty;
        let result = self.prepare_prefill_inner(gpu);
        if was_dirty {
            self.stats.prefill_restore_wall_ns = self
                .stats
                .prefill_restore_wall_ns
                .saturating_add(elapsed_wall_ns(started));
            if result.is_ok() {
                self.emit_stats();
            }
        }
        result
    }

    fn prepare_prefill_inner(&mut self, gpu: &mut Gpu) -> Result<()> {
        self.check(gpu)?;
        let result: Result<()> = (|| {
            let stream = self.copy_stream.as_ref().expect("live copy stream");
            self.hip
                .stream_synchronize(stream)
                .map_err(|e| error("prefill drain", e))?;
            for layer in 0..LAYERS {
                let snapshot = self.layers[layer].binding.residency.snapshot()?;
                if snapshot
                    .mapping
                    .iter()
                    .any(|loc| *loc != ResidencyLocation::Canonical)
                {
                    self.publish(
                        gpu,
                        layer,
                        vec![ResidencyLocation::Canonical; EXPERTS],
                        Vec::new(),
                    )?;
                }
                if let Some(cache) = &mut self.layers[layer].cache {
                    cache.clear();
                }
            }
            while let Some(fill) = self.pending.pop_front() {
                self.idle_events.push(fill.done);
            }
            if self.dirty {
                let ranges = self
                    .lease
                    .as_ref()
                    .expect("live tier lease")
                    .expert_cache_ranges()
                    .ok_or("expert cache lost restore ranges")?;
                let stream = self.copy_stream.as_ref().expect("live copy stream");
                self.copy_projection(ranges.mirror_cpu, ranges.donor_device, stream)?;
                self.hip
                    .stream_synchronize(stream)
                    .map_err(|e| error("prefill restore", e))?;
                self.stats.prefill_restores = self.stats.prefill_restores.saturating_add(1);
                self.stats.restore_bytes = self
                    .stats
                    .restore_bytes
                    .saturating_add(ranges.donor_device.1 as u64);
                self.dirty = false;
            }
            self.needs_seed = true;
            Ok(())
        })();
        if let Err(reason) = &result {
            self.poisoned = Some(reason.clone());
        }
        result
    }
    /// Refuse unknown token quiescence: a HIP wait alone cannot retire PM4.
    pub(crate) fn shutdown(&mut self, gpu: &mut Gpu) -> Result<()> {
        if self.outstanding.load(Ordering::Acquire) {
            return Err("expert cache refuses shutdown with an unretired GPU token".into());
        }
        if gpu.device_id != self.device_id {
            return Err("expert cache shutdown on another GPU".into());
        }
        gpu.bind_thread().map_err(|e| error("shutdown bind", e))?;
        self.hip
            .device_synchronize()
            .map_err(|e| error("shutdown wait", e))?;
        self.destroy_queue()
    }
    fn destroy_queue(&mut self) -> Result<()> {
        while let Some(event) = self.idle_events.pop() {
            self.hip
                .event_destroy(event)
                .map_err(|e| error("shutdown pooled event", e))?;
        }
        while let Some(fill) = self.pending.pop_front() {
            self.hip
                .event_destroy(fill.done)
                .map_err(|e| error("shutdown event", e))?;
        }
        if let Some(event) = self.retire_event.take() {
            self.hip
                .event_destroy(event)
                .map_err(|e| error("shutdown retire event", e))?;
        }
        if let Some(stream) = self.copy_stream.take() {
            self.hip
                .stream_destroy(stream)
                .map_err(|e| error("shutdown stream", e))?;
        }
        Ok(())
    }
    fn retain_on_unknown_quiescence(&mut self) {
        // Retain canonical bytes, slots, tables and upload mirrors. A poisoned
        // token can still have independent PM4 reads that HIP cannot fence.
        std::mem::forget(std::mem::take(&mut self.layers));
        if let Some(lease) = self.lease.take() {
            std::mem::forget(lease);
        }
        std::mem::forget(self.hip.clone());
    }
}
impl Drop for ExpertCache {
    fn drop(&mut self) {
        if self.outstanding.load(Ordering::Acquire) {
            self.retain_on_unknown_quiescence();
            return;
        }
        // Destruction is outside decode. This also fences a partially uploaded
        // table after a failed compute-stream call.
        let previous = self.hip.current_device();
        if self
            .hip
            .set_device(self.device_id)
            .and_then(|()| self.hip.device_synchronize())
            .is_ok()
        {
            let _ = self.destroy_queue();
        } else {
            self.retain_on_unknown_quiescence();
        }
        if let Ok(device) = previous {
            if device != self.device_id {
                let _ = self.hip.set_device(device);
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expert_cache_token_drop_does_not_claim_unknown_gpu_retirement() {
        let marker = Arc::new(AtomicBool::new(true));
        drop(ExpertCacheToken {
            guards: Vec::new(),
            outstanding: marker.clone(),
        });
        assert!(marker.load(Ordering::Acquire));
        let token = ExpertCacheToken {
            guards: Vec::new(),
            outstanding: marker.clone(),
        };
        // SAFETY: this fixture submitted no work and owns no guards.
        unsafe { token.retire_after_gpu_wait() }.unwrap();
        assert!(!marker.load(Ordering::Acquire));
    }

    #[test]
    fn expert_cache_accounting_uses_token_start_mapping() {
        let pair = HostPair {
            gate_up: (0, 2),
            down: (0, 3),
        };
        let mut host = vec![Some(pair); EXPERTS];
        host[0] = None;
        host[1] = None;
        let mut mapping = vec![ResidencyLocation::Canonical; EXPERTS];
        for expert in [2, 3] {
            mapping[expert] = ResidencyLocation::Slot(ResidencySlotTicket {
                slot: expert - 2,
                expert,
                generation: 1,
            });
        }
        let ids: Vec<_> = (0..10).collect();
        let mut stats = ExpertCacheStats::default();
        account_routes(&mut stats, &ids, &host, &mapping).unwrap();
        assert_eq!(
            (
                stats.selected,
                stats.core_hits,
                stats.cache_hits,
                stats.host_misses,
                stats.miss_bytes
            ),
            (10, 2, 2, 6, 30)
        );
        // A fill after the token cannot retroactively turn its host read into a hit.
        mapping[4] = ResidencyLocation::Slot(ResidencySlotTicket {
            slot: 2,
            expert: 4,
            generation: 2,
        });
        assert_eq!(stats.host_misses, 6);
        let mut rejected = ExpertCacheStats::default();
        assert!(account_routes(&mut rejected, &[7, 7], &host, &mapping).is_err());
        assert_eq!(rejected.selected, 0);
        assert!(account_routes(&mut rejected, &[-1], &host, &mapping).is_err());
        assert!(account_routes(&mut rejected, &[EXPERTS as i32], &host, &mapping).is_err());
    }

    #[test]
    fn expert_cache_cpu_alias_translation_checks_both_ends() {
        assert_eq!(
            host_projection(0x1100, 128, (0x1000, 1024), (0x8000, 1024)).unwrap(),
            (0x8100, 128)
        );
        assert!(host_projection(0x0fff, 1, (0x1000, 1024), (0x8000, 1024)).is_err());
        assert!(host_projection(0x13ff, 2, (0x1000, 1024), (0x8000, 1024)).is_err());
        assert!(host_projection(0x1000, 1, (0x1000, 1024), (usize::MAX, 1024)).is_err());
        assert!(host_projection(0x1000, 1, (0x1000, 1024), (0x8000, 1023)).is_err());
    }
    #[test]
    fn expert_cache_warm_order_requires_complete_permutation() {
        let mut rank: Vec<_> = (0..(LAYERS * EXPERTS) as u32).collect();
        let order = warm_order(&rank).unwrap();
        assert_eq!(order[0][0], EXPERTS - 1);
        assert_eq!(order[0][EXPERTS - 1], 0);
        assert!(warm_order(&rank[..rank.len() - 1]).is_err());
        rank[1] = rank[0];
        assert!(warm_order(&rank).is_err());
        rank[1] = (LAYERS * EXPERTS) as u32;
        assert!(warm_order(&rank).is_err());
    }
    #[test]
    fn expert_cache_seed_reset_keeps_ticket_generation_monotonic() {
        let mut cache = SlotCache::new(2).unwrap();
        let a = cache.reserve((), 42usize).unwrap();
        assert!(cache.commit(a));
        let b = cache.reserve((), 17).unwrap();
        assert!(cache.commit(b));
        cache.reserve((), 42).unwrap();
        cache.clear();
        let reset = cache.reserve((), 42).unwrap();
        assert_eq!(reset.slot, 0);
        assert!(reset.generation() > b.generation());
        assert!(!cache.commit(a));
        assert!(cache.commit(reset));
    }
    #[test]
    fn expert_cache_bounded_queue_serves_every_layer_before_second_fills() {
        let misses = (0..LAYERS)
            .map(|_| (0..TOPK).collect())
            .collect::<Vec<Vec<usize>>>();
        let mut cursor = 0;
        let mut total = [0; LAYERS];
        for _ in 0..3 {
            let selected = fair_misses(&misses, cursor).take(64).collect::<Vec<_>>();
            let mut served = [0; LAYERS];
            for &(layer, expert) in &selected {
                assert_eq!(expert, served[layer], "within-layer priority changed");
                served[layer] += 1;
                total[layer] += 1;
            }
            assert_eq!(selected.len(), 64);
            assert!(
                served.iter().all(|&n| n == 1 || n == 2),
                "a layer monopolized the queue"
            );
            cursor = (selected.last().unwrap().0 + 1) % LAYERS;
        }
        assert!(total.iter().all(|&n| n == 4), "extra quota did not rotate");
    }

    #[test]
    fn expert_cache_fair_fill_order_skips_empty_layers() {
        let misses = vec![vec![], vec![7, 9], vec![], vec![3]];
        assert_eq!(
            fair_misses(&misses, 3).collect::<Vec<_>>(),
            [(3, 3), (1, 7), (1, 9)]
        );
        assert_eq!(fair_misses(&[], 0).count(), 0);
        assert_eq!(fair_misses(&[vec![], vec![]], 1).count(), 0);
    }
}
