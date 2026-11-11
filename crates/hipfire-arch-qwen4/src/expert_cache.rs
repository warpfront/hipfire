//! Exact LRU of immutable Flash-Next expert payloads for AR and bounded verify.
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
use railgun::sealed_table::generation::{
    acquire_window_owned, finish_window, BlockRegistry, Expert, FenceBackend, FenceFailure,
    FenceState, FillTicket, FillsNotDone, GenerationId, GenerationRetired, OwnedTableGeneration,
    PendingFills, SlotBank, SlotCell, SlotPhaseView, SlotSpan, Staging, TableBacking,
    WindowReadLease,
};
use railgun::window::Quiescence;
use railgun::Binding;
use rdna_compute::Gpu;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::Instant;

const EXPERTS: usize = EXPERTS_PER_LAYER as usize;
const LAYERS: usize = TRUNK_LAYERS as usize;
const TOPK: usize = 10;
type Result<T> = std::result::Result<T, String>;

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
    reader: Option<WindowReadLease<'static>>,
    rows: usize,
}
/// Cumulative logical byte counters and host-observed wall durations. Timings
/// cover these manager methods, not the caller's route readback or GPU step.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(crate) struct ExpertCacheStats {
    /// Executed target rows, including speculative/replayed rows, not emitted tokens.
    pub tokens: u64,
    pub verify_batches: u64,
    pub max_guard_rows: usize,
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
    slot_ids: Vec<usize>,
    index: Vec<Option<usize>>,
    // Retain upload mirrors even when the second upload fails after the first.
    gate_up_upload: Vec<u8>,
    down_upload: Vec<u8>,
}
struct PendingFill {
    batch: PendingFills,
}

pub(crate) struct ExpertCache {
    hip: Arc<HipRuntime>,
    device_id: i32,
    lease: Option<ExpertTierLease>,
    layers: Vec<LayerCache>,
    copy_stream: Option<Stream>,
    bank: Option<CacheBank>,
    pending: VecDeque<PendingFill>,
    events: CopyEvents,
    max_inflight: usize,
    verify_rows: usize,
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

/// Validate the entire readback before accounting or cache policy can change.
/// Expert IDs may repeat between rows, but never within one router's top-k.
fn validate_routes(rows: usize, ids: &[i32]) -> Result<()> {
    if !(1..=MutableExpertResidency::MAX_VERIFY_ROWS).contains(&rows)
        || ids.len() != LAYERS * rows * TOPK
    {
        return Err("expert cache route readback shape mismatch".into());
    }
    for selected in ids.chunks_exact(TOPK) {
        let mut seen = [false; EXPERTS];
        for &id in selected {
            let expert =
                usize::try_from(id).map_err(|_| "expert cache negative selected expert")?;
            if expert >= EXPERTS || std::mem::replace(&mut seen[expert], true) {
                return Err("expert cache invalid/duplicate selected expert".into());
            }
        }
    }
    Ok(())
}

fn touch_unique(order: &mut Vec<usize>, expert: usize) {
    if let Some(index) = order.iter().position(|&id| id == expert) {
        order.remove(index);
    }
    order.push(expert);
}

struct LayerVisits {
    misses: Vec<usize>,
    accesses: Vec<usize>,
}

/// The sole slot/fill-generation authority. Policy metadata selects a slot,
/// while every ownership transition is performed by the shared railgun bank.
struct CacheBank {
    bank: Option<SlotBank>,
    cells: Vec<Option<SlotCell>>,
    staging: Staging,
    backing: Option<TableBacking>,
    generation: Option<OwnedTableGeneration>,
    retired: Option<GenerationRetired>,
    meta: Vec<(usize, usize)>,
    last_used: Vec<u64>,
    clock: u64,
    epoch: u64,
}
impl CacheBank {
    fn new(
        block: (usize, usize),
        allocation_generation: u64,
        spans: Vec<Vec<SlotSpan>>,
        meta: Vec<(usize, usize)>,
        tables: Vec<Binding>,
    ) -> Result<Self> {
        let registry = BlockRegistry::new(0);
        // This is the owner's actual contiguous, exclusively lent donor tail,
        // not the min/max hull of unrelated allocations. The tier lease owns it.
        let block = registry
            .adopt::<Expert>(block.0 as u64, block.1 as u64, allocation_generation)
            .map_err(|e| e.to_string())?;
        let (mut bank, cells) =
            SlotBank::new(block, spans, false).map_err(|(_, e)| e.to_string())?;
        let staging = bank.staging();
        let backing = bank
            .register_table_backing(tables)
            .map_err(|e| e.to_string())?;
        let generation = bank
            .seal(&staging, backing, GenerationId { epoch: 0, sub: 0 })
            .map_err(|(_, e)| e.to_string())?;
        let n = cells.len();
        if meta.len() != n {
            return Err("expert cache slot metadata extent mismatch".into());
        }
        Ok(Self {
            bank: Some(bank),
            cells: cells.into_iter().map(|c| Some(SlotCell::Free(c))).collect(),
            staging,
            backing: None,
            generation: Some(generation),
            retired: None,
            meta,
            last_used: vec![0; n],
            clock: 0,
            epoch: 0,
        })
    }
    fn require_backing(&self) -> Result<()> {
        if self.backing.is_none() || self.generation.is_some() {
            return Err("expert cache pointer tables are leased to a reader generation".into());
        }
        Ok(())
    }
    fn retire_idle(&mut self) -> Result<()> {
        if let Some(generation) = self.generation.take() {
            match self
                .bank
                .as_mut()
                .expect("open bank")
                .retire(generation, None)
            {
                Ok((backing, retired)) => {
                    self.backing = Some(backing);
                    self.retired = Some(retired);
                }
                Err((generation, error)) => {
                    self.generation = Some(generation);
                    return Err(error.to_string());
                }
            }
        }
        self.require_backing()
    }
    fn seal_idle(&mut self) -> Result<()> {
        self.require_backing()?;
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or("expert cache generation exhausted")?;
        let backing = self.backing.take().expect("write backing");
        match self.bank.as_mut().expect("open bank").seal(
            &self.staging,
            backing,
            GenerationId { epoch, sub: 0 },
        ) {
            Ok(generation) => {
                self.generation = Some(generation);
                self.epoch = epoch;
                Ok(())
            }
            Err((backing, error)) => {
                self.backing = Some(backing);
                Err(error.to_string())
            }
        }
    }
    fn acquire_reader(&mut self) -> Result<WindowReadLease<'static>> {
        let generation = self
            .generation
            .take()
            .ok_or("expert cache generation already has a reader")?;
        match acquire_window_owned(generation) {
            Ok(reader) => Ok(reader),
            Err((generation, error)) => {
                self.generation = Some(generation);
                Err(error.to_string())
            }
        }
    }
    fn finish_reader(
        &mut self,
        reader: WindowReadLease<'static>,
        fence: &mut dyn FenceBackend,
    ) -> Result<()> {
        if self.generation.is_some() || self.backing.is_some() {
            return Err("expert cache reader owner state mismatch".into());
        }
        let pending = reader
            .enqueued(fence)
            .map_err(|(_reader, f)| error("reader fence record", f.message))?;
        let fenced =
            finish_window(pending, fence).map_err(|e| error("reader wait", format!("{e:?}")))?;
        self.generation = Some(
            fenced
                .into_generation()
                .ok_or("expert cache window lost owned generation")?,
        );
        Ok(())
    }
    fn phase(&self, slot: usize) -> SlotPhaseView {
        self.cells[slot].as_ref().expect("cell present").phase()
    }
    fn occupant(&self, slot: usize) -> Option<u64> {
        match self.cells[slot].as_ref().expect("cell present") {
            SlotCell::Resident(s) => Some(s.occupant()),
            SlotCell::Filling(s) => Some(s.occupant()),
            _ => None,
        }
    }
    fn touch(&mut self, slot: usize) {
        self.clock = self.clock.saturating_add(1);
        self.last_used[slot] = self.clock;
    }
    /// Visit every executed row before choosing any victim. Repeated misses
    /// request one fill; repeated reads still retain their last visit order.
    fn visit_routes(
        &mut self,
        ids: &[i32],
        host: &[Option<HostPair>],
        index: &[Option<usize>],
    ) -> LayerVisits {
        let mut visits = LayerVisits {
            misses: Vec::new(),
            accesses: Vec::new(),
        };
        for &id in ids {
            let expert = id as usize;
            if host[expert].is_none() {
                continue;
            }
            touch_unique(&mut visits.accesses, expert);
            if let Some(slot) = index[expert] {
                self.touch(slot);
            } else {
                touch_unique(&mut visits.misses, expert);
            }
        }
        visits
    }
    /// Reservations may evict a key visited earlier in this batch. Only the
    /// current index may affect recency; an old snapshot never revives a key.
    fn touch_survivors(&mut self, accesses: &[usize], index: &[Option<usize>]) {
        for &expert in accesses {
            if let Some(slot) = index[expert] {
                self.touch(slot);
            }
        }
    }
    fn choose(&self, slots: &[usize]) -> Option<usize> {
        slots
            .iter()
            .copied()
            .find(|&s| self.phase(s) == SlotPhaseView::Free)
            .or_else(|| {
                slots
                    .iter()
                    .copied()
                    .filter(|&s| self.phase(s) == SlotPhaseView::Resident)
                    .min_by_key(|&s| (self.last_used[s], s))
            })
    }
    fn start_fill(&mut self, slot: usize, occupant: u64) -> Result<FillTicket> {
        match self.cells[slot].take().expect("cell present") {
            SlotCell::Free(free) => match self
                .bank
                .as_mut()
                .expect("open bank")
                .start_fill(free, occupant)
            {
                Ok((filling, ticket)) => {
                    self.cells[slot] = Some(SlotCell::Filling(filling));
                    self.touch(slot);
                    Ok(ticket)
                }
                Err((free, error)) => {
                    self.cells[slot] = Some(SlotCell::Free(free));
                    Err(error.to_string())
                }
            },
            other => {
                self.cells[slot] = Some(other);
                Err("expert cache fill needs a free slot".into())
            }
        }
    }
    fn complete(
        &mut self,
        slot: usize,
        done: railgun::sealed_table::generation::FillComplete,
    ) -> Result<(u64, u64)> {
        match self.cells[slot].take().expect("cell present") {
            SlotCell::Filling(filling) => match self
                .bank
                .as_mut()
                .expect("open bank")
                .finish_fill(filling, done)
            {
                Ok(resident) => {
                    let identity = (resident.occupant(), resident.fill_generation());
                    let result = self
                        .bank
                        .as_mut()
                        .expect("open bank")
                        .publish(&mut self.staging, &resident);
                    self.cells[slot] = Some(SlotCell::Resident(resident));
                    result.map_err(|e| e.to_string())?;
                    Ok(identity)
                }
                Err((filling, _done, error)) => {
                    self.cells[slot] = Some(SlotCell::Filling(filling));
                    Err(error.to_string())
                }
            },
            other => {
                self.cells[slot] = Some(other);
                Err("expert cache completion needs a filling slot".into())
            }
        }
    }
    fn evict(&mut self, slot: usize) -> Result<()> {
        self.require_backing()?;
        let retired = self
            .retired
            .as_ref()
            .ok_or("cache missing generation retirement")?;
        let bank = self.bank.as_mut().expect("open bank");
        match self.cells[slot].take().expect("cell present") {
            SlotCell::Resident(resident) => match bank.unpublish(&mut self.staging, resident) {
                Ok(evicting) => match bank.reclaim(evicting, retired) {
                    Ok(free) => {
                        self.cells[slot] = Some(SlotCell::Free(free));
                        Ok(())
                    }
                    Err((evicting, error)) => {
                        self.cells[slot] = Some(SlotCell::Evicting(evicting));
                        Err(error.to_string())
                    }
                },
                Err((resident, error)) => {
                    self.cells[slot] = Some(SlotCell::Resident(resident));
                    Err(error.to_string())
                }
            },
            other => {
                self.cells[slot] = Some(other);
                Err("expert cache eviction needs a resident slot".into())
            }
        }
    }
}
fn checked_slot_span(block: (usize, usize), span: (usize, usize)) -> Result<SlotSpan> {
    let offset = span
        .0
        .checked_sub(block.0)
        .ok_or("cache slot precedes donor tail")?;
    if span.1 == 0 || offset.checked_add(span.1).is_none_or(|end| end > block.1) {
        return Err("cache slot exceeds donor tail".into());
    }
    Ok(SlotSpan {
        offset: offset as u64,
        bytes: span.1 as u64,
    })
}
fn copy_projection(
    hip: &HipRuntime,
    source: (usize, usize),
    destination: (usize, usize),
    stream: &Stream,
) -> Result<()> {
    if source.1 == 0 || source.1 != destination.1 {
        return Err("expert cache unequal copy extents".into());
    }
    // SAFETY: caller binds immutable pinned source and destination ranges to
    // the live tier lease and either a FillTicket or the exact ScratchLoan.
    let bytes = unsafe { std::slice::from_raw_parts(source.0 as *const u8, source.1) };
    let destination =
        unsafe { DeviceBuffer::from_raw(destination.0 as *mut c_void, destination.1) };
    hip.memcpy_htod_async(&destination, bytes, stream)
        .map_err(|e| error("payload copy", e))
}
fn copy_pair(hip: &HipRuntime, stream: &Stream, source: HostPair, spans: &[Binding]) -> Result<()> {
    if spans.len() != 2 {
        return Err("cache fill requires paired GU/DN ticket spans".into());
    }
    copy_projection(
        hip,
        source.gate_up,
        (spans[0].base as usize, spans[0].bytes as usize),
        stream,
    )?;
    copy_projection(
        hip,
        source.down,
        (spans[1].base as usize, spans[1].bytes as usize),
        stream,
    )
}
static NEXT_FENCE_INCAR: AtomicU64 = AtomicU64::new(1);
fn incarnation() -> u64 {
    NEXT_FENCE_INCAR.fetch_add(1, Ordering::Relaxed)
}
fn unknown(message: impl std::fmt::Display) -> FenceFailure {
    FenceFailure {
        message: message.to_string(),
        quiescence: Quiescence::Unknown,
    }
}
struct CopyEvents {
    incarnation: u64,
    next: u64,
    idle: Vec<Event>,
    live: BTreeMap<u64, Event>,
}
impl CopyEvents {
    fn new() -> Self {
        Self {
            incarnation: incarnation(),
            next: 1,
            idle: Vec::new(),
            live: BTreeMap::new(),
        }
    }
}
struct CopyFence<'a> {
    hip: &'a HipRuntime,
    stream: &'a Stream,
    events: &'a mut CopyEvents,
}
// SAFETY: one real HIP event follows every pair's writes on the immutable
// copy-stream incarnation. Errors never claim quiescence; the bank quarantines.
unsafe impl FenceBackend for CopyFence<'_> {
    fn incarnation(&self) -> u64 {
        self.events.incarnation
    }
    fn record(&mut self) -> std::result::Result<u64, FenceFailure> {
        let event = match self.events.idle.pop() {
            Some(e) => e,
            None => self
                .hip
                .event_create_with_flags(HIP_EVENT_DISABLE_TIMING)
                .map_err(unknown)?,
        };
        if let Err(e) = self.hip.event_record(&event, Some(self.stream)) {
            self.events.idle.push(event);
            return Err(unknown(e));
        }
        let id = self.events.next;
        self.events.next = id
            .checked_add(1)
            .ok_or_else(|| unknown("cache copy fence counter exhausted"))?;
        self.events.live.insert(id, event);
        Ok(id)
    }
    fn poll(&mut self, fence: u64) -> FenceState {
        match self.events.live.get(&fence) {
            None => FenceState::Failed(unknown("cache copy fence missing")),
            Some(event) => match self.hip.event_query(event) {
                Ok(true) => FenceState::Complete,
                Ok(false) => FenceState::Pending,
                Err(e) => FenceState::Failed(unknown(e)),
            },
        }
    }
    fn wait(&mut self, fence: u64) -> FenceState {
        match self.events.live.get(&fence) {
            None => FenceState::Failed(unknown("cache copy fence missing")),
            Some(event) => match self.hip.event_synchronize(event) {
                Ok(()) => FenceState::Complete,
                Err(e) => FenceState::Failed(unknown(e)),
            },
        }
    }
    fn recycle(&mut self, fence: u64) {
        if let Some(event) = self.events.live.remove(&fence) {
            self.events.idle.push(event);
        }
    }
}
struct ComputeFence<'a> {
    hip: &'a HipRuntime,
    stream: Option<&'a Stream>,
    incarnation: u64,
    recorded: u64,
    _thread: PhantomData<*const ()>,
}
impl<'a> ComputeFence<'a> {
    fn new(hip: &'a HipRuntime, stream: Option<&'a Stream>) -> Self {
        Self {
            hip,
            stream,
            incarnation: incarnation(),
            recorded: 0,
            _thread: PhantomData,
        }
    }
}
// SAFETY: record marks the already-enqueued final HIP consumer. A successful
// HIP wait covers those consumers. Callers separately prove independent PM4
// retirement before handing its window to this backend; Unknown never enters.
unsafe impl FenceBackend for ComputeFence<'_> {
    fn incarnation(&self) -> u64 {
        self.incarnation
    }
    fn record(&mut self) -> std::result::Result<u64, FenceFailure> {
        self.recorded += 1;
        Ok(self.recorded)
    }
    fn poll(&mut self, _: u64) -> FenceState {
        FenceState::Pending
    }
    fn wait(&mut self, fence: u64) -> FenceState {
        if fence == 0 || fence > self.recorded {
            return FenceState::Failed(unknown("cache compute fence missing"));
        }
        let result = match self.stream {
            Some(s) => self.hip.stream_synchronize(s),
            None => self.hip.device_synchronize(),
        };
        match result {
            Ok(()) => FenceState::Complete,
            Err(e) => FenceState::Failed(unknown(e)),
        }
    }
}

impl ExpertCache {
    pub(crate) fn new(
        gpu: &mut Gpu,
        lease: ExpertTierLease,
        config: ExpertCacheConfig,
        rank: &[u32],
    ) -> Result<Self> {
        if gpu.arch != "gfx1201"
            || config.max_inflight == 0
            || !(1..=MutableExpertResidency::MAX_VERIFY_ROWS).contains(&config.verify_rows)
        {
            return Err(
                "expert cache requires gfx1201, bounded verify rows and a nonzero queue".into(),
            );
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
            bank: None,
            pending: VecDeque::new(),
            events: CopyEvents::new(),
            max_inflight: config.max_inflight,
            verify_rows: config.verify_rows,
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
            result.layers.push(LayerCache {
                binding: ExpertCacheLayerBinding { tables, residency },
                slots,
                host,
                slot_ids: Vec::new(),
                index: vec![None; EXPERTS],
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
        let mut slot_meta = Vec::new();
        let mut slot_spans = Vec::new();
        let mut tables = Vec::new();
        for (layer, state) in result.layers.iter_mut().enumerate() {
            for (local, slot) in state.slots.iter().enumerate() {
                state.slot_ids.push(slot_meta.len());
                slot_meta.push((layer, local));
                slot_spans.push(vec![
                    checked_slot_span(ranges.donor_device, slot.ranges.gate_up_device)?,
                    checked_slot_span(ranges.donor_device, slot.ranges.down_device)?,
                ]);
            }
            for table in [
                state.binding.tables.gate_up_table(),
                state.binding.tables.down_table(),
            ] {
                tables.push(Binding {
                    base: table.buf.as_ptr() as u64,
                    bytes: table.buf.size() as u64,
                });
            }
        }
        result.bank = Some(CacheBank::new(
            ranges.donor_device,
            lease.generation(),
            slot_spans,
            slot_meta,
            tables,
        )?);
        for _ in 0..config.max_inflight {
            result.events.idle.push(
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
    pub(crate) fn verify_rows(&self) -> usize {
        self.verify_rows
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
        if tracing::enabled!(tracing::Level::DEBUG) {
            tracing::debug!(stats = ?self.stats, "qwen4 exact expert cache");
        }
    }

    pub(crate) fn check_readers_idle(&self) -> Result<()> {
        if let Some(reason) = &self.poisoned {
            return Err(error("poisoned", reason));
        }
        if self.outstanding.load(Ordering::Acquire) {
            return Err("expert cache token is unretired".into());
        }
        if self.copy_stream.is_none() {
            return Err("expert cache is stopped".into());
        }
        Ok(())
    }
    pub(crate) fn check_idle_for_state_op(&self, gpu: &Gpu) -> Result<()> {
        self.check_readers_idle()?;
        if gpu.device_id != self.device_id {
            return Err("expert cache used on another GPU".into());
        }
        Ok(())
    }
    fn check(&self, gpu: &Gpu) -> Result<()> {
        self.check_idle_for_state_op(gpu)?;
        gpu.bind_thread().map_err(|e| error("bind", e))
    }
    pub(crate) fn begin_token(&mut self, gpu: &mut Gpu) -> Result<ExpertCacheToken> {
        self.begin_rows(gpu, 1)
    }
    pub(crate) fn begin_rows(&mut self, gpu: &mut Gpu, rows: usize) -> Result<ExpertCacheToken> {
        if rows == 0 || rows > self.verify_rows {
            return Err("expert cache verify exceeds resolved row capacity".into());
        }
        let started = Instant::now();
        let result = self.begin_token_inner(gpu, rows);
        self.stats.begin_token_wall_ns = self
            .stats
            .begin_token_wall_ns
            .saturating_add(elapsed_wall_ns(started));
        result
    }

    fn begin_token_inner(&mut self, gpu: &mut Gpu, rows: usize) -> Result<ExpertCacheToken> {
        self.check(gpu)?;
        let result = (|| {
            if self.needs_seed {
                self.seed(gpu)?;
            }
            self.publish_completed(gpu, false)?;
            let mut guards = Vec::with_capacity(LAYERS);
            for layer in &self.layers {
                let cap = &layer.binding.residency;
                let acquired = cap.snapshot().and_then(|snapshot| {
                    if rows == 1 {
                        cap.begin_token(snapshot.epoch)
                    } else {
                        cap.begin_verify(snapshot.epoch, rows)
                    }
                });
                match acquired {
                    Ok(guard) => guards.push(guard),
                    Err(error) => {
                        // SAFETY: no GPU token operation has been submitted.
                        for guard in guards {
                            unsafe { guard.retire_after_gpu_wait() }?;
                        }
                        return Err(error);
                    }
                }
            }
            let reader = match self.bank.as_mut().expect("live bank").acquire_reader() {
                Ok(reader) => reader,
                Err(error) => {
                    // SAFETY: reader acquisition precedes every token submission.
                    for guard in guards {
                        unsafe { guard.retire_after_gpu_wait() }?;
                    }
                    return Err(error);
                }
            };
            self.outstanding.store(true, Ordering::Release);
            Ok(ExpertCacheToken {
                guards,
                reader: Some(reader),
                outstanding: self.outstanding.clone(),
                rows,
            })
        })();
        if let Err(reason) = &result {
            self.poisoned = Some(reason.clone());
        }
        result
    }

    /// Cancel without observing routes, changing recency or issuing payload copies.
    /// # Safety
    /// Every independent PM4 reader is proven retired (or never submitted).
    /// The HIP fence below covers the remaining HIP work; it cannot prove PM4 retirement.
    /// Unknown callers must retain/drop the token and quarantine its complete owner set.
    pub(crate) unsafe fn cancel_after_gpu_wait(
        &mut self,
        gpu: &mut Gpu,
        mut token: ExpertCacheToken,
    ) -> Result<()> {
        if !Arc::ptr_eq(&token.outstanding, &self.outstanding) || gpu.device_id != self.device_id {
            return Err("expert cache reader belongs to another cache or GPU".into());
        }
        gpu.bind_thread().map_err(|e| error("retire bind", e))?;
        let mut fence = ComputeFence::new(&self.hip, gpu.active_stream.as_ref());
        let reader = token
            .reader
            .take()
            .ok_or("expert cache reader lease missing")?;
        let result = self
            .bank
            .as_mut()
            .expect("live bank")
            .finish_reader(reader, &mut fence);
        if let Err(reason) = result {
            self.poisoned = Some(reason.clone());
            return Err(reason);
        }
        for guard in token.guards {
            // SAFETY: caller's independent-queue proof plus the HIP fence above.
            unsafe { guard.retire_after_gpu_wait() }?;
        }
        self.outstanding.store(false, Ordering::Release);
        Ok(())
    }

    /// # Safety
    /// Every independent PM4 reader is proven retired; IDs are this token's complete readback.
    pub(crate) unsafe fn finish_token(
        &mut self,
        gpu: &mut Gpu,
        token: ExpertCacheToken,
        ids: &[i32],
    ) -> Result<()> {
        // SAFETY: forwarded; the wrapper retains the singleton AR contract.
        unsafe { self.finish_rows(gpu, token, 1, ids) }
    }

    /// # Safety
    /// Every row's independent PM4 reader is proven retired. IDs are the complete
    /// [layer][row][top-k] readback under this one stable-table reader.
    pub(crate) unsafe fn finish_rows(
        &mut self,
        gpu: &mut Gpu,
        token: ExpertCacheToken,
        rows: usize,
        ids: &[i32],
    ) -> Result<()> {
        let started = Instant::now();
        let valid = if rows != token.rows || rows == 0 || rows > self.verify_rows {
            Err("expert cache completion rows differ from acquired reader".into())
        } else {
            validate_routes(rows, ids)
        };
        let snapshots: Vec<_> = token.guards.iter().map(|g| g.snapshot().clone()).collect();
        // SAFETY: forwarded from this function's caller; HIP work is fenced in cancellation.
        let result = unsafe { self.cancel_after_gpu_wait(gpu, token) }.and_then(|()| {
            valid?;
            self.observe_and_fill(gpu, rows, ids, &snapshots)
        });
        self.stats.finish_token_wall_ns = self
            .stats
            .finish_token_wall_ns
            .saturating_add(elapsed_wall_ns(started));
        if let Err(reason) = &result {
            self.poisoned = Some(reason.clone());
        }
        if result.is_ok() && self.stats.tokens.is_multiple_of(64) {
            self.emit_stats();
        }
        result
    }

    fn observe_and_fill(
        &mut self,
        gpu: &Gpu,
        rows: usize,
        ids: &[i32],
        snapshots: &[ResidencySnapshot],
    ) -> Result<()> {
        self.stats.tokens = self.stats.tokens.saturating_add(rows as u64);
        self.stats.max_guard_rows = self.stats.max_guard_rows.max(rows);
        if rows > 1 {
            self.stats.verify_batches = self.stats.verify_batches.saturating_add(1);
        }
        let bank = self.bank.as_mut().expect("live bank");
        let mut misses = vec![Vec::new(); LAYERS];
        let mut accesses = vec![Vec::new(); LAYERS];
        for layer in 0..LAYERS {
            let selected = &ids[layer * rows * TOPK..(layer + 1) * rows * TOPK];
            let state = &self.layers[layer];
            for row in selected.chunks_exact(TOPK) {
                account_routes(&mut self.stats, row, &state.host, &snapshots[layer].mapping)?;
            }
            let visits = bank.visit_routes(selected, &state.host, &state.index);
            accesses[layer] = visits.accesses;
            if !state.slot_ids.is_empty() {
                misses[layer] = visits.misses;
            }
        }
        bank.retire_idle()?;
        let mut mappings: Vec<_> = self
            .layers
            .iter()
            .map(|s| s.binding.residency.snapshot().map(|s| s.mapping.to_vec()))
            .collect::<Result<_>>()?;
        let mut changed = [false; LAYERS];
        let mut tickets = Vec::new();
        let pending = self
            .pending
            .iter()
            .map(|p| p.batch.fills().len())
            .sum::<usize>();
        let mut last_served = None;
        for (layer, expert) in fair_misses(&misses, self.next_layer) {
            if pending + tickets.len() >= self.max_inflight {
                self.stats.skipped_full += 1;
                continue;
            }
            let Some(slot) = bank.choose(&self.layers[layer].slot_ids) else {
                self.stats.skipped_full += 1;
                continue;
            };
            if let Some(victim) = bank.occupant(slot) {
                let victim = victim as usize % EXPERTS;
                self.layers[layer].index[victim] = None;
                mappings[layer][victim] = ResidencyLocation::Canonical;
                changed[layer] = true;
                bank.evict(slot)?;
                self.stats.evictions += 1;
            }
            let ticket = bank.start_fill(slot, (layer * EXPERTS + expert) as u64)?;
            self.layers[layer].index[expert] = Some(slot);
            tickets.push(ticket);
            last_served = Some(layer);
        }
        self.next_layer = (last_served.unwrap_or(self.next_layer) + 1) % LAYERS;
        if rows > 1 {
            for (layer, visits) in accesses.iter().enumerate() {
                bank.touch_survivors(visits, &self.layers[layer].index);
            }
        }
        for layer in 0..LAYERS {
            if changed[layer] {
                self.publish(gpu, layer, std::mem::take(&mut mappings[layer]), Vec::new())?;
            }
        }
        // The newly sealed membership excludes every victim before copy enqueue.
        self.bank.as_mut().expect("live bank").seal_idle()?;
        if tickets.is_empty() {
            return Ok(());
        }
        let pairs = tickets.len();
        let bytes = tickets
            .iter()
            .map(|t| t.write_spans().iter().map(|s| s.bytes).sum::<u64>())
            .sum::<u64>();
        let hip = &self.hip;
        let stream = self.copy_stream.as_ref().expect("live copy stream");
        let layers = &self.layers;
        let mut backend = CopyFence {
            hip,
            stream,
            events: &mut self.events,
        };
        let bank = self.bank.as_mut().expect("live bank");
        let result = bank.bank.as_mut().expect("open bank").issue_fills(
            tickets,
            &mut backend,
            &mut |ticket| {
                let layer = ticket.occupant() as usize / EXPERTS;
                let expert = ticket.occupant() as usize % EXPERTS;
                let source = layers[layer].host[expert].ok_or("fill attempted immutable core")?;
                copy_pair(hip, stream, source, ticket.write_spans())
            },
        );
        match result {
            Ok(batch) => self.pending.push_back(PendingFill { batch }),
            Err((tickets, failure)) => {
                bank.bank
                    .as_mut()
                    .expect("open bank")
                    .abandon_fills(tickets);
                return Err(error("fill enqueue", failure.message));
            }
        }
        self.dirty = true;
        self.stats.fills_started += pairs as u64;
        self.stats.fill_bytes += bytes;
        self.stats.peak_inflight = self.stats.peak_inflight.max(pending + pairs);
        Ok(())
    }
    fn publish_completed(&mut self, gpu: &Gpu, wait: bool) -> Result<()> {
        let mut completed = Vec::new();
        while let Some(fill) = self.pending.pop_front() {
            let mut copy = CopyFence {
                hip: &self.hip,
                stream: self.copy_stream.as_ref().expect("live copy stream"),
                events: &mut self.events,
            };
            match self
                .bank
                .as_mut()
                .expect("live bank")
                .bank
                .as_mut()
                .expect("open bank")
                .complete_fills(fill.batch, &mut copy, wait)
            {
                Ok(mut done) => completed.append(&mut done),
                Err(FillsNotDone::Pending(batch)) => {
                    self.pending.push_front(PendingFill { batch });
                    break;
                }
                Err(FillsNotDone::Failed { failure, .. }) => {
                    return Err(error("fill completion", failure.message))
                }
            }
        }
        if completed.is_empty() {
            return Ok(());
        }
        self.bank.as_mut().expect("live bank").retire_idle()?;
        let mut by_layer = vec![Vec::new(); LAYERS];
        for done in completed {
            let slot = done.slot() as usize;
            let bank = self.bank.as_mut().expect("live bank");
            let (layer, local) = bank.meta[slot];
            let (expert, generation) = bank.complete(slot, done)?;
            by_layer[layer].push(ResidencySlotTicket {
                slot: local,
                expert: expert as usize % EXPERTS,
                generation,
            });
            self.stats.fills_completed += 1;
        }
        for (layer, tickets) in by_layer.into_iter().enumerate() {
            if tickets.is_empty() {
                continue;
            }
            let mut mapping = self.layers[layer]
                .binding
                .residency
                .snapshot()?
                .mapping
                .to_vec();
            for &ticket in &tickets {
                mapping[ticket.expert] = ResidencyLocation::Slot(ticket);
            }
            self.publish(gpu, layer, mapping, tickets)?;
        }
        self.bank.as_mut().expect("live bank").seal_idle()
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
        self.bank.as_ref().expect("live bank").require_backing()?;
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
        let bank = self.bank.as_mut().expect("live bank");
        bank.retire_idle()?;
        let mut tickets = Vec::new();
        for (layer, state) in self.layers.iter_mut().enumerate() {
            for (local, slot) in state.slots.iter().enumerate() {
                let global = state.slot_ids[local];
                let expert = slot.ranges.original_expert;
                tickets.push(bank.start_fill(global, (layer * EXPERTS + expert) as u64)?);
                state.index[expert] = Some(global);
            }
        }
        // Initial load or completed long-prefill restore already installed the
        // exact original donor bytes. Still issue real bank completion fences.
        let mut copy = CopyFence {
            hip: &self.hip,
            stream: self.copy_stream.as_ref().expect("live copy stream"),
            events: &mut self.events,
        };
        let pending = match bank.bank.as_mut().expect("open bank").issue_fills(
            tickets,
            &mut copy,
            &mut |_| Ok(()),
        ) {
            Ok(pending) => pending,
            Err((tickets, failure)) => {
                bank.bank
                    .as_mut()
                    .expect("open bank")
                    .abandon_fills(tickets);
                return Err(error("seed issue", failure.message));
            }
        };
        let completed = bank
            .bank
            .as_mut()
            .expect("open bank")
            .complete_fills(pending, &mut copy, true)
            .map_err(|_| "expert cache seed fence did not complete")?;
        let mut layers = vec![Vec::new(); LAYERS];
        for done in completed {
            let global = done.slot() as usize;
            let (layer, local) = bank.meta[global];
            let (expert, generation) = bank.complete(global, done)?;
            layers[layer].push(ResidencySlotTicket {
                slot: local,
                expert: expert as usize % EXPERTS,
                generation,
            });
        }
        for (layer, tickets) in layers.into_iter().enumerate() {
            let mut mapping = vec![ResidencyLocation::Canonical; EXPERTS];
            for &ticket in &tickets {
                mapping[ticket.expert] = ResidencyLocation::Slot(ticket);
            }
            self.publish(gpu, layer, mapping, tickets)?;
            let bank = self.bank.as_mut().expect("live bank");
            for &expert in &self.warm_order[layer] {
                if let Some(slot) = self.layers[layer].index[expert] {
                    bank.touch(slot);
                }
            }
        }
        self.bank.as_mut().expect("live bank").seal_idle()?;
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
            self.publish_completed(gpu, true)?;
            self.bank.as_mut().expect("live bank").retire_idle()?;
            for layer in 0..LAYERS {
                let snapshot = self.layers[layer].binding.residency.snapshot()?;
                if snapshot
                    .mapping
                    .iter()
                    .any(|m| *m != ResidencyLocation::Canonical)
                {
                    self.publish(
                        gpu,
                        layer,
                        vec![ResidencyLocation::Canonical; EXPERTS],
                        Vec::new(),
                    )?;
                }
                self.layers[layer].index.fill(None);
            }
            let bank = self.bank.as_mut().expect("live bank");
            for slot in 0..bank.cells.len() {
                if bank.phase(slot) == SlotPhaseView::Resident {
                    bank.evict(slot)?;
                }
            }
            if self.dirty {
                let ranges = self
                    .lease
                    .as_ref()
                    .expect("live lease")
                    .expert_cache_ranges()
                    .ok_or("cache lost restore ranges")?;
                let mut device = ComputeFence::new(&self.hip, None);
                let q = bank
                    .bank
                    .as_mut()
                    .expect("open bank")
                    .quiesce(&mut device)
                    .map_err(|f| f.message)?;
                let loan = bank
                    .bank
                    .as_mut()
                    .expect("open bank")
                    .lend_scratch(q)
                    .map_err(|e| e.to_string())?;
                if loan.span()
                    != (Binding {
                        base: ranges.donor_device.0 as u64,
                        bytes: ranges.donor_device.1 as u64,
                    })
                {
                    return Err("cache scratch loan extent changed".into());
                }
                let stream = self.copy_stream.as_ref().expect("live copy stream");
                copy_projection(&self.hip, ranges.mirror_cpu, ranges.donor_device, stream)?;
                let mut copy = CopyFence {
                    hip: &self.hip,
                    stream,
                    events: &mut self.events,
                };
                let restore = bank
                    .bank
                    .as_mut()
                    .expect("open bank")
                    .issue_restore(&loan, &mut copy, &mut |_, _, _| {
                        Err("free cache slot unexpectedly occupied during restore".into())
                    })
                    .map_err(|f| f.message)?;
                bank.bank
                    .as_mut()
                    .expect("open bank")
                    .end_scratch(loan, restore, &mut copy)
                    .map_err(|(_, _, e)| e.to_string())?;
                self.stats.prefill_restores += 1;
                self.stats.restore_bytes += ranges.donor_device.1 as u64;
                self.dirty = false;
            }
            self.bank.as_mut().expect("live bank").seal_idle()?;
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
        if let Some(bank) = self.bank.as_mut() {
            if bank.bank.is_none() || bank.cells.iter().any(Option::is_none) {
                return Err("expert cache teardown authority is incomplete; retain owners".into());
            }
            bank.retire_idle()?;
            let mut device = ComputeFence::new(&self.hip, None);
            let q = bank
                .bank
                .as_mut()
                .expect("open bank")
                .quiesce(&mut device)
                .map_err(|f| f.message)?;
            let raw = bank.bank.take().expect("open bank");
            let cells = bank
                .cells
                .iter_mut()
                .map(|c| c.take().expect("cell present"))
                .collect();
            match raw.close(cells, q) {
                Ok((block, done)) => {
                    block.free(done).map_err(|(_, _, e)| e.to_string())?;
                }
                Err((raw, cells, error)) => {
                    bank.bank = Some(raw);
                    bank.cells = cells.into_iter().map(Some).collect();
                    return Err(error.to_string());
                }
            }
        }
        self.bank = None;
        self.pending.clear();
        for event in std::mem::take(&mut self.events.live)
            .into_values()
            .chain(self.events.idle.drain(..))
        {
            self.hip
                .event_destroy(event)
                .map_err(|e| error("shutdown event", e))?;
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
        if let Some(bank) = self.bank.take() {
            std::mem::forget(bank);
        }
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
            if self.destroy_queue().is_err() {
                self.retain_on_unknown_quiescence();
            }
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
    struct CpuFence {
        incarnation: u64,
        next: u64,
        unknown: bool,
    }
    // SAFETY: metadata-only tests enqueue no device operations. The explicit
    // unknown arm simulates a backend that cannot establish retirement.
    unsafe impl FenceBackend for CpuFence {
        fn incarnation(&self) -> u64 {
            self.incarnation
        }
        fn record(&mut self) -> std::result::Result<u64, FenceFailure> {
            self.next += 1;
            Ok(self.next)
        }
        fn poll(&mut self, _: u64) -> FenceState {
            if self.unknown {
                FenceState::Failed(unknown("injected unknown queue"))
            } else {
                FenceState::Complete
            }
        }
        fn wait(&mut self, f: u64) -> FenceState {
            self.poll(f)
        }
    }
    fn cpu_fence(unknown: bool) -> CpuFence {
        CpuFence {
            incarnation: incarnation(),
            next: 0,
            unknown,
        }
    }
    fn bank_fixture(base: usize) -> CacheBank {
        CacheBank::new(
            (base, 64),
            7,
            vec![vec![
                SlotSpan {
                    offset: 0,
                    bytes: 16,
                },
                SlotSpan {
                    offset: 16,
                    bytes: 16,
                },
            ]],
            vec![(0, 0)],
            vec![Binding {
                base: base as u64 + 0x1000,
                bytes: 16,
            }],
        )
        .unwrap()
    }
    fn cpu_completed(
        bank: &mut CacheBank,
        expert: u64,
    ) -> railgun::sealed_table::generation::FillComplete {
        let ticket = bank.start_fill(0, expert).unwrap();
        let mut fence = cpu_fence(false);
        let raw = bank.bank.as_mut().unwrap();
        let pending = raw
            .issue_fills(vec![ticket], &mut fence, &mut |_| Ok(()))
            .unwrap();
        raw.complete_fills(pending, &mut fence, true)
            .unwrap()
            .pop()
            .unwrap()
    }
    #[test]
    fn expert_cache_bank_reader_blocks_publication_and_known_cancellation_returns_authority() {
        let mut bank = bank_fixture(0x10000);
        bank.retire_idle().unwrap();
        let done = cpu_completed(&mut bank, 42);
        assert_eq!(bank.complete(0, done).unwrap(), (42, 1));
        bank.seal_idle().unwrap();
        let epoch = bank.epoch;
        let reader = bank.acquire_reader().unwrap();
        assert!(bank.retire_idle().is_err());
        assert!(bank.require_backing().is_err());
        assert!(bank.evict(0).is_err());
        assert_eq!(bank.phase(0), SlotPhaseView::Resident);
        bank.finish_reader(reader, &mut cpu_fence(false)).unwrap();
        assert_eq!(
            bank.epoch, epoch,
            "cancel must not publish or invent a generation"
        );
        bank.retire_idle().unwrap();
        bank.evict(0).unwrap();
        let done = cpu_completed(&mut bank, 17);
        assert_eq!(bank.complete(0, done).unwrap(), (17, 2));
    }
    #[test]
    fn expert_cache_bank_unknown_reader_and_dropped_token_never_reopen_backing() {
        for drop_token in [false, true] {
            let mut bank = bank_fixture(0x20000);
            let reader = bank.acquire_reader().unwrap();
            if drop_token {
                let outstanding = Arc::new(AtomicBool::new(true));
                drop(ExpertCacheToken {
                    guards: Vec::new(),
                    reader: Some(reader),
                    outstanding: outstanding.clone(),
                    rows: 1,
                });
                assert!(outstanding.load(Ordering::Acquire));
            } else {
                assert!(bank.finish_reader(reader, &mut cpu_fence(true)).is_err());
            }
            assert!(bank.retire_idle().is_err());
            assert!(bank.require_backing().is_err());
            assert_eq!(bank.bank.as_ref().unwrap().live_generations(), 1);
        }
    }
    #[test]
    fn expert_cache_bank_rejects_foreign_completion_and_partial_pair_enqueue() {
        let mut a = bank_fixture(0x30000);
        let mut b = bank_fixture(0x40000);
        a.retire_idle().unwrap();
        b.retire_idle().unwrap();
        let foreign = cpu_completed(&mut a, 8);
        let ticket = b.start_fill(0, 9).unwrap();
        assert!(b.complete(0, foreign).is_err());
        assert_eq!(b.phase(0), SlotPhaseView::Filling);
        let mut wrote_gu = false;
        let failure = b
            .bank
            .as_mut()
            .unwrap()
            .issue_fills(vec![ticket], &mut cpu_fence(false), &mut |t| {
                assert_eq!(t.write_spans().len(), 2);
                wrote_gu = true;
                Err("injected DN enqueue failure after GU".into())
            })
            .unwrap_err();
        assert!(wrote_gu);
        assert_eq!(failure.1.quiescence, Quiescence::Unknown);
        b.bank.as_mut().unwrap().abandon_fills(failure.0);
        assert!(b.bank.as_ref().unwrap().is_quarantined(0));
        assert!(!b.staging.contains(0));
        assert!(
            b.choose(&[0]).is_none(),
            "partial pair cannot be reused or published"
        );
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
