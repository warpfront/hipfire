//! Decode-only exact LRU of immutable Flash-Next expert payloads, on the shared runtime core.
//!
//! A token uses the tables acquired at its start. Misses execute from immutable mapped host
//! backing; asynchronous copies serve later tokens. Dirty short prefill reads independent
//! immutable core/host tables. Long prefill restores the original donor bytes before using the
//! frozen resident view.
//!
//! # What runs where
//!
//! The decisions and every slot / fill / generation transition belong to
//! [`hipfire_runtime::expert_cache`]: [`ExpertCachePolicy`] (LRU, candidates, fill budget,
//! pending-is-invisible, withdrawal, the retirement cadence over railgun's `SlotBank`) and
//! [`DeviceIo`] (copy stream, event pool, one event per paired GU+DN fill). This module is the
//! thin Qwen adapter:
//!
//! * the layout: [`ExpertKey`]s come from the admitted [`ExpertCatalog`]
//!   ([`crate::expert_cache_config::build_expert_catalog`], checked against the identity recorded
//!   at admission) and each layer's donor slots become one policy pool whose layout class the
//!   catalog proves ([`build_layout`]);
//! * the canonical sources: both projections of a cacheable expert are the segments of one
//!   [`BundleSource`] bundle, so a fill copies GU and DN under one event and never publishes half;
//! * the decode window: one window per AR token or per verify/repair window of `rows` rows
//!   (`begin_token` .. `finish_token`). The routed ids the forward read back, laid out
//!   `[layer][row][k]`, become the window's uses; a rejected draft row is a real read and counts
//!   as a use (PLAN §5.3), and a model-state rollback never touches residency. `published()` /
//!   `withdrawn()` are written into the per-layer device tables through the dispatch
//!   [`MutableExpertResidency`] capability (the table writes and the launch authority stay
//!   dispatch's);
//! * the donor protocol (§5.2, see [`ExpertCache::prepare_prefill`]): a dirty prompt shorter than
//!   the staged threshold reads the immutable core/host tables and leaves fills, mappings,
//!   tickets and recency alone; a prompt that needs the Resident donors back refuses while a
//!   token or window is open, drains every fill, rewrites each pool hull from the pinned mirror,
//!   records the re-seed behind those copies and publishes it over the canonical mapping, all
//!   before its first kernel. Prefill never observes routes, so it promotes no expert;
//! * teardown ([`ExpertCache::shutdown`], `Drop`): wait every fill, drop the pools once the
//!   device is idle; an unretired token or an unprovable retirement retains every owner.
//!
//! # Retirement cadence (A2.1)
//!
//! A window with nothing to publish and no victim to withdraw seals, retires and syncs nothing:
//! each pool's one lease of its live generation simply continues. Only a completed fill or a
//! withdrawn victim ([`ExpertCachePolicy::needs_retirement`]) fences the compute stream, retires
//! the dirty pools' generations, rewrites the tables of the changed layers and seals again.
//! Completed fills are published when `begin_token` polls them and when `finish_token` stages
//! them with the group decision, so one retirement serves both.
//!
//! # Decision differences from the #1022 private manager (placement only, never math)
//!
//! * Victims: least recently used by the core's stamp (a key the window used is stamped), the
//!   colder hot-set ranking first on a tie, then the slot index. #1022 broke ties by slot index
//!   and stamped slots whose fill was still pending.
//! * Candidates: missed keys ordered by reads, then hot-set ranking, then key. #1022 served the
//!   layers round-robin in selected order.
//! * Fill budget: the pool's `F` minus its fills in flight, one window per group. `F` is
//!   `ceil(HIPFIRE_QWEN4_EXPERT_CACHE_INFLIGHT / pools)` per layer pool (at least one) and the
//!   event pool holds `F x pools` events, so the global cap is that knob rounded up to a whole
//!   fill per layer pool; #1022 counted one global cap with no per-layer share.
//! * Pools: one pool per layer (a single layout class each, refused otherwise) instead of one pool
//!   per layout class. The dispatch capability publishes through a per-layer slot universe, so a
//!   layer's key can only occupy that layer's donor slots; sharing a class pool across layers
//!   would make every layer's capability carry every slot (its constructor checks every slot
//!   against every other slot and each publication clones all of them). S4c / S3b can lift this.
//! * Warm start: the donors' original experts fill the slots, all with the same stamp, so the
//!   colder hot-set ranking is evicted first (#1022 staggered the stamps by the ranking, the same
//!   order); the seed is published at load and after each restore instead of at the next token.
//! * The old per-token reader lease and compute-stream sync are gone: the window consumer joins
//!   on the caller's proof (the route readback wait, or `device_synchronize` on cancellation),
//!   and the stream is only synced when a generation retires.
//! * Slot fill generations restart when a restore rebuilds the pools, so a dispatch ticket's
//!   generation is `restore epoch << 32 | fill generation`: still strictly increasing per slot.

use crate::expert_cache_config::{build_expert_catalog, ExpertCacheConfig};
use crate::expert_slabs::{EXPERTS_PER_LAYER, TRUNK_LAYERS};
use crate::expert_tier_owners::{ExpertCacheTableLease, ExpertTierLease};
use crate::expert_tier_plan::Placement;
use hip_bridge::{HipRuntime, Stream};
use hipfire_dispatch::families::moe::{
    MutableExpertResidency, MutableExpertResidencySpec, ResidencyLocation, ResidencyPublication,
    ResidencySlot, ResidencySlotTicket, ResidencySnapshot, ResidencySpan, ResidencyTokenGuard,
};
use hipfire_runtime::expert_cache::{
    BundleSource, CacheStats, CancelOutcome, ConsumerId, CopyDevice, DeviceIo, ExpertCachePolicy,
    ExpertCatalog, ExpertKey, HipCopyBackend, HostRange, PolicyConfig, PoolConfig, PoolDest,
    PoolId, PoolInit, Published, Segment, WindowConsumerPhase, WindowUse, Withdrawn,
};
use railgun::sealed_table::generation::{
    BlockRegistry, Expert, FenceBackend, FenceFailure, FenceState, SlotSpan,
};
use railgun::window::Quiescence;
use railgun::Binding;
use rdna_compute::Gpu;
use std::marker::PhantomData;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::Instant;

const EXPERTS: usize = EXPERTS_PER_LAYER as usize;
const LAYERS: usize = TRUNK_LAYERS as usize;
const TOPK: usize = 10;
type Result<T, E = String> = std::result::Result<T, E>;

/// Descriptors returned by tables are borrowed and must never be freed.
#[derive(Clone)]
pub(crate) struct ExpertCacheLayerBinding {
    pub tables: ExpertCacheTableLease,
    pub residency: MutableExpertResidency,
}

/// An unretired token poisons every capability and prevents freeing its owners.
pub(crate) struct ExpertCacheToken {
    guards: Vec<ResidencyTokenGuard>,
    consumer: ConsumerId,
    outstanding: Arc<AtomicBool>,
}
/// Cumulative logical byte counters and host-observed wall durations. Timings
/// cover these manager methods, not the caller's route readback or GPU step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub(crate) struct ExpertCacheStats {
    /// Windows observed (one per AR token or per verify/repair window).
    pub tokens: u64,
    /// Rows observed across those windows (one per AR token).
    pub rows: u64,
    /// Windows that carried more than one row.
    pub verify_windows: u64,
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
    /// drain, restoration and waits, including failed attempts.
    pub prefill_restore_wall_ns: u64,
}

fn elapsed_wall_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn fail(context: &str, value: impl std::fmt::Display) -> String {
    format!("exact expert cache {context}: {value}")
}

/// What every state operation, mapped prefill and donor restore checks before it touches
/// anything: a poisoned cache, an unretired token and a stopped cache each refuse by name.
fn reader_gate(poisoned: Option<&str>, token_outstanding: bool, running: bool) -> Result<()> {
    if let Some(reason) = poisoned {
        return Err(fail("poisoned", reason));
    }
    if token_outstanding {
        return Err("expert cache token is unretired".into());
    }
    if !running {
        return Err("expert cache is stopped".into());
    }
    Ok(())
}

/// Count a prompt admitted to the immutable core/host view. Counters only: that view reads no
/// donor slot, so nothing waits for, publishes or re-judges anything here, and no use is
/// recorded (a scan-heavy prompt promotes no expert in the LRU).
fn note_mapped_prefill(stats: &mut ExpertCacheStats, rows: usize) -> Result<()> {
    if rows == 0 {
        return Err("expert cache mapped prefill requires at least one row".into());
    }
    stats.mapped_prefill_calls = stats.mapped_prefill_calls.saturating_add(1);
    stats.mapped_prefill_rows = stats
        .mapped_prefill_rows
        .saturating_add(u64::try_from(rows).unwrap_or(u64::MAX));
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Layout: keys, pools, canonical sources
// ---------------------------------------------------------------------------------------------

/// CPU range `(address, bytes)` of both canonical projections of one host-backed expert.
#[derive(Clone, Copy)]
struct HostPair {
    gate_up: (usize, usize),
    down: (usize, usize),
}

/// One expert of a layer as the cache view names it.
struct ExpertInput {
    gate_up: ResidencySpan,
    down: ResidencySpan,
    /// `Some` for every cacheable expert (host-mapped or a donor); `None` for the pinned core.
    host: Option<HostPair>,
}

/// One donor slot: the original expert whose bytes it holds and its device extents.
struct SlotInput {
    original_expert: usize,
    gate_up: (usize, usize),
    down: (usize, usize),
}

struct LayerInput {
    experts: Vec<ExpertInput>,
    /// In physical tail order.
    slots: Vec<SlotInput>,
    /// The layer's mutable GU and DN pointer tables.
    tables: [Binding; 2],
}

/// What one routed selection of a (layer, expert) is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Read {
    /// Pinned in VRAM by the admitted plan.
    Core,
    /// Cacheable and served by a pool of donor slots.
    Cache { pool: PoolId, rank: u32, bytes: u64 },
    /// Cacheable but the layer has no donor slot: always a host read.
    Host { bytes: u64 },
}

/// The physical shape of one pool: a layer's donor slots.
struct PoolGeometry {
    layer: usize,
    /// `(device address, bytes)` of the hull of the pool's slots inside the donor tail.
    block: (usize, usize),
    /// Block-relative `[gate/up, down]` write extents per slot, in physical order.
    spans: Vec<Vec<SlotSpan>>,
    tables: Vec<Binding>,
    /// Slot `i` starts out holding its donor's original expert.
    seed: Vec<(ExpertKey, u32)>,
}

struct Layout {
    owner_id: u64,
    owner_generation: u64,
    /// `layer * EXPERTS + expert`.
    reads: Vec<Read>,
    /// The two canonical host segments (GU, DN) of each cacheable expert of a pooled layer.
    bundles: Vec<Option<[Segment; 2]>>,
    /// The policy pool of each layer.
    layer_pool: Vec<Option<PoolId>>,
    pools: Vec<PoolGeometry>,
}

impl Layout {
    fn key(&self, layer: usize, expert: usize) -> ExpertKey {
        ExpertKey {
            owner_id: self.owner_id,
            owner_generation: self.owner_generation,
            group_id: layer as u32,
            expert_id: expert as u32,
        }
    }

    fn slots(&self) -> usize {
        self.pools.iter().map(|pool| pool.spans.len()).sum()
    }
}

impl BundleSource for Layout {
    fn bundle(&self, key: &ExpertKey, pool: PoolId) -> Option<&[Segment]> {
        if key.owner_id != self.owner_id || key.owner_generation != self.owner_generation {
            return None;
        }
        let layer = key.group_id as usize;
        let expert = key.expert_id as usize;
        if layer >= LAYERS || expert >= EXPERTS || self.layer_pool[layer] != Some(pool) {
            return None;
        }
        self.bundles[layer * EXPERTS + expert]
            .as_ref()
            .map(|bundle| &bundle[..])
    }
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

/// Ranking position (smaller = hotter) of every `layer * EXPERTS + expert`; the rank must be a
/// complete permutation.
fn rank_positions(rank: &[u32]) -> Result<Vec<u32>> {
    if rank.len() != LAYERS * EXPERTS {
        return Err("expert cache rank cardinality".into());
    }
    let mut position = vec![u32::MAX; rank.len()];
    for (at, &id) in rank.iter().enumerate() {
        let slot = position
            .get_mut(id as usize)
            .ok_or("expert cache rank is not a permutation")?;
        if std::mem::replace(slot, at as u32) != u32::MAX {
            return Err("expert cache rank is not a permutation".into());
        }
    }
    Ok(position)
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

/// Check the cache view against the admitted catalog and derive every pool, seed and canonical
/// bundle. Pure: no device, no lease.
fn build_layout(
    catalog: &ExpertCatalog,
    layers: &[LayerInput],
    rank_pos: &[u32],
) -> Result<Layout> {
    if layers.len() != LAYERS || rank_pos.len() != LAYERS * EXPERTS {
        return Err("expert cache layout cardinality".into());
    }
    let mut layout = Layout {
        owner_id: catalog.owner_id(),
        owner_generation: catalog.owner_generation(),
        reads: vec![Read::Core; LAYERS * EXPERTS],
        bundles: vec![None; LAYERS * EXPERTS],
        layer_pool: vec![None; LAYERS],
        pools: Vec::new(),
    };
    let mut prior_end = 0usize;
    for (layer, input) in layers.iter().enumerate() {
        if input.experts.len() != EXPERTS {
            return Err("expert cache layer expert cardinality".into());
        }
        let (gate_up_bytes, down_bytes) =
            (input.experts[0].gate_up.bytes, input.experts[0].down.bytes);
        let mut class = None;
        for (expert, e) in input.experts.iter().enumerate() {
            if e.gate_up.bytes != gate_up_bytes || e.down.bytes != down_bytes {
                return Err("expert cache requires equal projection extents within a layer".into());
            }
            let key = layout.key(layer, expert);
            let bundle = catalog.bundle(&key).ok_or_else(|| {
                format!("expert cache catalog lacks layer {layer} expert {expert}")
            })?;
            let segments = bundle.segments();
            if segments.len() != 2
                || segments[0].extent.len != e.gate_up.bytes as u64
                || segments[1].extent.len != e.down.bytes as u64
            {
                return Err(format!(
                    "expert cache catalog segments of layer {layer} expert {expert} differ from the cache view"
                ));
            }
            if bundle.pinned() != e.host.is_none() {
                return Err(format!(
                    "expert cache catalog and cache view disagree on whether layer {layer} expert {expert} is pinned"
                ));
            }
            if !bundle.pinned() {
                match class {
                    None => class = Some(bundle.pool()),
                    Some(bound) if bound != bundle.pool() => {
                        return Err(format!(
                            "expert cache layer {layer} mixes layout classes; one pool per layer needs one"
                        ));
                    }
                    Some(_) => {}
                }
            }
        }
        let pooled = !input.slots.is_empty();
        let pool = PoolId(layout.pools.len() as u16);
        if pooled {
            let class = class
                .and_then(|pool| catalog.class_of(pool))
                .ok_or_else(|| {
                    format!("expert cache layer {layer} has donors but no cacheable expert")
                })?;
            let [class_gate_up, class_down] = class.segments() else {
                return Err("expert cache layout class is not a GU + DN pair".into());
            };
            let mut start = usize::MAX;
            let mut end = 0usize;
            for slot in &input.slots {
                if slot.gate_up.1 as u64 != class_gate_up.len
                    || slot.down.1 as u64 != class_down.len
                {
                    return Err(format!(
                        "expert cache layer {layer} slot extents differ from its layout class"
                    ));
                }
                for (address, bytes) in [slot.gate_up, slot.down] {
                    start = start.min(address);
                    end = end.max(
                        address
                            .checked_add(bytes)
                            .ok_or("expert cache slot extent overflows")?,
                    );
                }
            }
            if start < prior_end {
                return Err(
                    "expert cache layer donor slots interleave with the previous layer's".into(),
                );
            }
            prior_end = end;
            let block = (start, end - start);
            let mut spans = Vec::with_capacity(input.slots.len());
            let mut seed: Vec<(ExpertKey, u32)> = Vec::with_capacity(input.slots.len());
            for slot in &input.slots {
                spans.push(vec![
                    checked_slot_span(block, slot.gate_up)?,
                    checked_slot_span(block, slot.down)?,
                ]);
                let expert = slot.original_expert;
                if expert >= EXPERTS
                    || input.experts[expert].host.is_none()
                    || seed.iter().any(|(key, _)| key.expert_id == expert as u32)
                {
                    return Err(format!(
                        "expert cache layer {layer} donor {expert} is not a distinct cacheable expert"
                    ));
                }
                seed.push((
                    layout.key(layer, expert),
                    rank_pos[layer * EXPERTS + expert],
                ));
            }
            layout.layer_pool[layer] = Some(pool);
            layout.pools.push(PoolGeometry {
                layer,
                block,
                spans,
                tables: input.tables.to_vec(),
                seed,
            });
        }
        for (expert, e) in input.experts.iter().enumerate() {
            let id = layer * EXPERTS + expert;
            let Some(host) = e.host else { continue };
            let bytes = (host.gate_up.1 + host.down.1) as u64;
            layout.reads[id] = if pooled {
                Read::Cache {
                    pool,
                    rank: rank_pos[id],
                    bytes,
                }
            } else {
                Read::Host { bytes }
            };
            if pooled {
                // SAFETY: canonical pinned host mirror of the lease the cache retains; nothing
                // writes it and it outlives every fill.
                let range =
                    |(address, len): (usize, usize)| unsafe { HostRange::new(address, len) };
                layout.bundles[id] = Some([
                    Segment::new(range(host.gate_up), 0, 0),
                    Segment::new(range(host.down), 1, 0),
                ]);
            }
        }
    }
    Ok(layout)
}

// ---------------------------------------------------------------------------------------------
// Route accounting
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RouteAccount {
    selected: u64,
    core_hits: u64,
    cache_hits: u64,
    host_misses: u64,
    miss_bytes: u64,
}

/// Count what the completed token actually read, independent of later LRU
/// reservations or copy completion. Reject malformed selection before counting.
fn account_route(
    layout: &Layout,
    layer: usize,
    ids: &[i32],
    mapping: &[ResidencyLocation],
) -> Result<RouteAccount> {
    if mapping.len() != EXPERTS {
        return Err("expert cache route-accounting cardinality".into());
    }
    let mut seen = [false; EXPERTS];
    for &id in ids {
        let expert = usize::try_from(id).map_err(|_| "expert cache negative selected expert")?;
        if expert >= EXPERTS || std::mem::replace(&mut seen[expert], true) {
            return Err("expert cache invalid/duplicate selected expert".into());
        }
    }
    let mut account = RouteAccount::default();
    for &id in ids {
        let expert = id as usize;
        account.selected += 1;
        match layout.reads[layer * EXPERTS + expert] {
            Read::Core => account.core_hits += 1,
            Read::Cache { bytes, .. } | Read::Host { bytes } => {
                if matches!(mapping[expert], ResidencyLocation::Slot(_)) {
                    account.cache_hits += 1;
                } else {
                    account.host_misses += 1;
                    account.miss_bytes += bytes;
                }
            }
        }
    }
    Ok(account)
}

/// Reject a route readback that is not exactly `rows` windows-rows of `LAYERS * TOPK` valid
/// expert ids, or that exceeds the rows the cache's use log was sized for.
fn validate_window_ids(ids: &[i32], rows: usize, max_rows: usize) -> Result<()> {
    let expected = LAYERS
        .checked_mul(TOPK)
        .and_then(|ids| ids.checked_mul(rows));
    if rows == 0
        || rows > max_rows
        || expected != Some(ids.len())
        || ids.iter().any(|id| !(0..EXPERTS as i32).contains(id))
    {
        return Err("expert cache invalid route readback".into());
    }
    Ok(())
}

/// Walk one window's routed ids layer -> row -> rank (ids are `[layer][row][k]`; for one row
/// this is today's layer-major order). Every row is accounted on its own `TOPK` slice, so one
/// expert in two rows is legal while a duplicate inside one row is refused. `record` receives
/// each cacheable read in that order. Returns the summed account of the window.
fn observe_window_routes<'a>(
    layout: &Layout,
    ids: &[i32],
    rows: usize,
    mapping_of: impl Fn(usize) -> &'a [ResidencyLocation],
    mut record: impl FnMut(WindowUse) -> Result<()>,
) -> Result<RouteAccount> {
    let mut total = RouteAccount::default();
    for layer in 0..LAYERS {
        let mapping = mapping_of(layer);
        for row in 0..rows {
            let start = (layer * rows + row) * TOPK;
            let selected = ids
                .get(start..start + TOPK)
                .ok_or("expert cache route readback extent")?;
            let account = account_route(layout, layer, selected, mapping)?;
            total.selected = total.selected.saturating_add(account.selected);
            total.core_hits = total.core_hits.saturating_add(account.core_hits);
            total.cache_hits = total.cache_hits.saturating_add(account.cache_hits);
            total.host_misses = total.host_misses.saturating_add(account.host_misses);
            total.miss_bytes = total.miss_bytes.saturating_add(account.miss_bytes);
            for &id in selected {
                let expert = id as usize;
                if let Read::Cache { pool, rank, .. } = layout.reads[layer * EXPERTS + expert] {
                    record(WindowUse {
                        key: layout.key(layer, expert),
                        pool,
                        reads: 1,
                        rank,
                    })?;
                }
            }
        }
    }
    Ok(total)
}

/// The dispatch ticket generation of a fill generation: slot fill generations restart when a
/// restore rebuilds the pools, the capability requires them never to go backwards.
fn ticket_generation(restore_epoch: u64, fill_generation: u64) -> Result<u64> {
    if fill_generation >= 1 << 32 {
        return Err("expert cache slot fill generation exhausted".into());
    }
    restore_epoch
        .checked_mul(1 << 32)
        .and_then(|base| base.checked_add(fill_generation))
        .ok_or_else(|| "expert cache restore epoch exhausted".into())
}

/// The experts one layer's table image drops and the slot tickets it gains.
struct LayerChange {
    withdrawn: Vec<usize>,
    published: Vec<ResidencySlotTicket>,
}

/// Group the policy's withdrawals and publications by layer. A ticket's generation is
/// `restore_epoch << 32 | fill generation` ([`ticket_generation`]).
fn layer_changes(
    withdrawn: &[Withdrawn],
    published: &[Published],
    restore_epoch: u64,
) -> Result<Vec<LayerChange>> {
    let mut changes: Vec<LayerChange> = (0..LAYERS)
        .map(|_| LayerChange {
            withdrawn: Vec::new(),
            published: Vec::new(),
        })
        .collect();
    for entry in withdrawn {
        changes
            .get_mut(entry.key.group_id as usize)
            .ok_or("expert cache withdrawal names an unknown layer")?
            .withdrawn
            .push(entry.key.expert_id as usize);
    }
    for entry in published {
        let ticket = ResidencySlotTicket {
            slot: entry.slot as usize,
            expert: entry.key.expert_id as usize,
            generation: ticket_generation(restore_epoch, entry.fill_generation)?,
        };
        changes
            .get_mut(entry.key.group_id as usize)
            .ok_or("expert cache publication names an unknown layer")?
            .published
            .push(ticket);
    }
    Ok(changes)
}

/// One layer's next mapping. `current` is the published mapping to edit; `None` (the re-seed of
/// a restore) starts from the canonical one, so an expert the earlier pools had filled into a
/// slot names that slot no more: only the tickets of this change survive.
fn next_mapping(
    current: Option<Vec<ResidencyLocation>>,
    change: &LayerChange,
) -> Vec<ResidencyLocation> {
    let mut mapping = current.unwrap_or_else(|| vec![ResidencyLocation::Canonical; EXPERTS]);
    for &expert in &change.withdrawn {
        mapping[expert] = ResidencyLocation::Canonical;
    }
    for &ticket in &change.published {
        mapping[ticket.expert] = ResidencyLocation::Slot(ticket);
    }
    mapping
}

/// The per-layer verdicts a prefill route reads from its runtime: whether the layer reads
/// host-mapped experts (it selects the stage and the host wait) and whether its routed headers
/// were verified symmetric (the >= 512-row IU4 route, whose scratch plan the F32 fallback would
/// overflow).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefillVerdict {
    pub host_mapped: bool,
    pub symmetric: bool,
}

/// The verdicts of a layer's cache runtime (the immutable core/host prompt view or the mutable
/// decode tables), given its Resident runtime's. Both name the Resident payloads byte for byte
/// (the donors' mirror, or a slot holding a copy of the canonical payload), so the symmetric
/// verdict verified on the Resident headers carries over. A host-mapped bit that differs means a
/// layer donates without holding a host expert (the plan never picks such a donor): a call through
/// it would take another route than the Resident one, so the pair is refused.
pub(crate) fn inherit_prefill_verdict(
    layer: usize,
    resident: PrefillVerdict,
    cache_host_mapped: bool,
) -> Result<PrefillVerdict> {
    if resident.host_mapped != cache_host_mapped {
        return Err(format!(
            "expert cache layer {layer}: host_mapped is {} in the Resident view and {cache_host_mapped} \
             in the cache view (a layer without a host expert must not donate)",
            resident.host_mapped
        ));
    }
    Ok(resident)
}

// ---------------------------------------------------------------------------------------------
// Device adapters
// ---------------------------------------------------------------------------------------------

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

/// The shared core's HIP copy backend.
type QwenCopy = HipCopyBackend<Arc<HipRuntime>>;

/// The copy fence of a warm seed: the seed moves no byte of its own (the donors hold their
/// experts already, or a restore's bulk copies were enqueued first), so every pool's batch
/// records copy event 0 behind everything on the copy stream and one synchronize covers them
/// all. Only valid while the io has no fill in flight, so event 0 is free.
struct SeedFence<'a, B: CopyDevice> {
    copy: &'a B,
    recorded: u64,
    synced: bool,
}
// SAFETY: record marks event 0 after every copy enqueued so far on the copy stream; a
// successful synchronize of it covers all of them. Seed batches enqueue nothing themselves.
unsafe impl<B: CopyDevice> FenceBackend for SeedFence<'_, B> {
    fn incarnation(&self) -> u64 {
        self.copy.incarnation()
    }
    fn record(&mut self) -> std::result::Result<u64, FenceFailure> {
        self.copy.record(0).map_err(unknown)?;
        self.recorded += 1;
        self.synced = false;
        Ok(self.recorded)
    }
    fn poll(&mut self, _: u64) -> FenceState {
        FenceState::Pending
    }
    fn wait(&mut self, fence: u64) -> FenceState {
        if fence == 0 || fence > self.recorded {
            return FenceState::Failed(unknown("seed copy fence missing"));
        }
        if !self.synced {
            if let Err(e) = self.copy.synchronize(0) {
                return FenceState::Failed(unknown(e));
            }
            self.synced = true;
        }
        FenceState::Complete
    }
}

/// The last-read fence of a retiring generation: every consumer of the closed window already
/// retired, so one synchronize of the compute stream covers all of them. One instance serves one
/// retirement step: nothing is enqueued between its waits, so after the first successful wait
/// every later fence of the step is already covered.
struct ComputeFence<'a> {
    hip: &'a HipRuntime,
    stream: Option<&'a Stream>,
    incarnation: u64,
    recorded: u64,
    synced: bool,
    _thread: PhantomData<*const ()>,
}
impl<'a> ComputeFence<'a> {
    fn new(hip: &'a HipRuntime, stream: Option<&'a Stream>) -> Self {
        Self {
            hip,
            stream,
            incarnation: incarnation(),
            recorded: 0,
            synced: false,
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
        if self.synced {
            return FenceState::Complete;
        }
        let result = match self.stream {
            Some(s) => self.hip.stream_synchronize(s),
            None => self.hip.device_synchronize(),
        };
        match result {
            Ok(()) => {
                self.synced = true;
                FenceState::Complete
            }
            Err(e) => FenceState::Failed(unknown(e)),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The cache
// ---------------------------------------------------------------------------------------------

struct LayerCache {
    binding: ExpertCacheLayerBinding,
    // Retain upload mirrors even when the second upload fails after the first.
    gate_up_upload: Vec<u8>,
    down_upload: Vec<u8>,
}

/// The shared core's policy over its device I/O.
struct Engine<B: CopyDevice> {
    policy: ExpertCachePolicy,
    io: DeviceIo<B>,
}

pub(crate) struct ExpertCache {
    hip: Arc<HipRuntime>,
    device_id: i32,
    lease: Option<ExpertTierLease>,
    layers: Vec<LayerCache>,
    layout: Layout,
    engine: Option<Engine<QwenCopy>>,
    policy_config: PolicyConfig,
    /// `F` of every pool: fills one window may start.
    fills_per_pool: usize,
    /// The donor tail a restore rewrites, and its pinned host mirror (equal lengths).
    donor_device: (usize, usize),
    mirror_cpu: (usize, usize),
    /// Most rows one window may carry; the use log is sized for it at construction.
    max_rows: usize,
    restore_epoch: u64,
    dirty: bool,
    poisoned: Option<String>,
    outstanding: Arc<AtomicBool>,
    stats: ExpertCacheStats,
}

/// Build the pools of a layout. Each build adopts its blocks from a fresh registry: the previous
/// build's ledger blocks are dropped with its policy, which is a registry capacity leak only.
fn build_policy(
    layout: &Layout,
    config: PolicyConfig,
    fills_per_pool: usize,
    allocation_generation: u64,
) -> Result<ExpertCachePolicy> {
    let registry = BlockRegistry::new(0);
    let mut pools = Vec::with_capacity(layout.pools.len());
    for geometry in &layout.pools {
        let block = registry
            .adopt::<Expert>(
                geometry.block.0 as u64,
                geometry.block.1 as u64,
                allocation_generation,
            )
            .map_err(|e| fail(&format!("layer {} pool block", geometry.layer), e))?;
        pools.push((
            PoolConfig {
                slots: geometry.spans.len(),
                fills_per_window: fills_per_pool,
            },
            PoolInit {
                block,
                spans: geometry.spans.clone(),
                tables: geometry.tables.clone(),
                seed: geometry.seed.clone(),
            },
        ));
    }
    ExpertCachePolicy::new(config, pools).map_err(|e| fail("policy", e))
}

/// Rewrite every pool hull from the pinned donor mirror: one bulk copy per pool on the copy
/// stream, so the seed events recorded after it complete only once the donors hold their
/// original bytes. The hulls lie inside the donor tail; the mirror maps it byte for byte.
fn restore_hulls<B: CopyDevice>(
    layout: &Layout,
    donor_device: (usize, usize),
    mirror_cpu: (usize, usize),
    copy: &B,
) -> Result<()> {
    for (index, pool) in layout.pools.iter().enumerate() {
        let (base, bytes) = pool.block;
        let offset = base
            .checked_sub(donor_device.0)
            .filter(|offset| {
                offset
                    .checked_add(bytes)
                    .is_some_and(|end| end <= donor_device.1)
            })
            .ok_or("expert cache pool hull leaves the donor tail")?;
        // SAFETY: the lease retains the pinned mirror, which nothing writes, past every copy.
        let src = unsafe { HostRange::new(mirror_cpu.0 + offset, bytes) };
        copy.enqueue_copy(PoolId(index as u16), base as u64, src)?;
    }
    Ok(())
}

/// Issue every seed fill as one batch per pool behind a single copy fence (`SeedFence`), then
/// wait for all of them: the completion tokens are in the policy, staged by the next plan.
fn issue_seed<B: CopyDevice>(engine: &mut Engine<B>) -> Result<()> {
    if engine.io.in_flight() != 0 {
        return Err("expert cache seed needs an idle copy engine".into());
    }
    let mut tickets: Vec<Vec<_>> = (0..engine.policy.pools()).map(|_| Vec::new()).collect();
    for op in engine.policy.take_fills() {
        tickets[usize::from(op.pool.0)].push(op.ticket);
    }
    let mut fence = SeedFence {
        copy: engine.io.backend(),
        recorded: 0,
        synced: false,
    };
    let mut batches = Vec::with_capacity(tickets.len());
    for (index, tickets) in tickets.into_iter().enumerate() {
        if tickets.is_empty() {
            continue;
        }
        let pool = PoolId(index as u16);
        let batch = engine
            .policy
            .issue_fills(pool, tickets, &mut fence, &mut |_| Ok(()))
            .map_err(|(back, failure)| {
                engine.policy.abandon_fills(pool, back);
                fail("seed fill", failure.message)
            })?;
        batches.push((pool, batch));
    }
    for (pool, batch) in batches {
        let done = engine
            .policy
            .complete_fills(pool, batch, &mut fence, true)
            .map_err(|_| "expert cache seed fence did not complete")?;
        for token in done {
            engine
                .policy
                .deliver_completion(pool, token)
                .map_err(|e| fail("seed completion", e))?;
        }
    }
    Ok(())
}

/// The donor tail a restore rewrites and its pinned host mirror (equal lengths, byte for byte).
#[derive(Clone, Copy)]
struct DonorTail {
    device: (usize, usize),
    mirror: (usize, usize),
}

/// A freshly seeded engine: its seed fills are issued (events recorded, completion tokens waiting
/// in the policy for the staging plan) but nothing is published yet.
struct Seeded<B: CopyDevice> {
    engine: Engine<B>,
    /// The first failure of the restore copies or the seed. The engine is returned regardless, so
    /// teardown sees it.
    issued: Result<()>,
}

/// Seed `policy`'s pools over `io`. With `restore`, every pool hull is first rewritten from the
/// mirror in one bulk copy per pool; the seed copies nothing, and its one fence per pool records
/// on the same copy stream, behind those copies, so a completed seed means the donors hold their
/// original experts. CPU and copy stream only: no compute stream and no table in here.
fn issue_seed_pools<B: CopyDevice>(
    layout: &Layout,
    policy: ExpertCachePolicy,
    io: DeviceIo<B>,
    restore: Option<DonorTail>,
) -> Seeded<B> {
    let mut engine = Engine { policy, io };
    let issued = match restore {
        Some(donor) => restore_hulls(layout, donor.device, donor.mirror, engine.io.backend()),
        None => Ok(()),
    };
    let issued = issued.and_then(|()| issue_seed(&mut engine));
    Seeded { engine, issued }
}

/// Why a restore did not happen.
#[derive(Debug, PartialEq, Eq)]
enum RestoreError {
    /// A window is open, so a reader may still be reading a donor slot. Nothing was drained,
    /// rewritten or recorded: the cache stays installed and usable, and the caller may retry once
    /// the reader retires.
    Busy,
    /// A drain, rebuild or copy failed: the donors may be half restored.
    Failed(String),
}

/// What a restore leaves on the copy stream.
struct Restored<B: CopyDevice> {
    /// Counters of the pools the restore retired.
    retired: CacheStats,
    seeded: Seeded<B>,
}

/// The policy and copy-stream half of a long-prefill restore of the engine in `slot`. Refused (an
/// open window, a fill that cannot be waited for, pools that cannot be rebuilt) the engine stays
/// installed. Otherwise every fill of the old pools is drained (no copy can land in a donor slot
/// after the rewrite), the new pools are built, the old ones are dropped (a completion they
/// issued can no longer name a slot of the new ones: their bank is another one), each pool hull
/// is rewritten from the mirror and the seed's events record behind those copies.
fn restore_pools<B: CopyDevice>(
    slot: &mut Option<Engine<B>>,
    layout: &Layout,
    config: PolicyConfig,
    fills_per_pool: usize,
    allocation_generation: u64,
    donor: DonorTail,
) -> Result<Restored<B>, RestoreError> {
    let old = slot
        .as_mut()
        .ok_or_else(|| RestoreError::Failed("expert cache is stopped".into()))?;
    if !old.policy.window().is_idle() {
        return Err(RestoreError::Busy);
    }
    old.io
        .poll(&mut old.policy, true)
        .map_err(|cause| RestoreError::Failed(fail("restore drain", cause)))?;
    let policy = build_policy(layout, config, fills_per_pool, allocation_generation)
        .map_err(RestoreError::Failed)?;
    let Engine {
        policy: old_policy,
        io,
    } = slot
        .take()
        .ok_or_else(|| RestoreError::Failed("expert cache is stopped".into()))?;
    let retired = old_policy.stats();
    drop(old_policy);
    Ok(Restored {
        retired,
        seeded: issue_seed_pools(layout, policy, io, Some(donor)),
    })
}

impl ExpertCache {
    pub(crate) fn new(
        gpu: &mut Gpu,
        lease: ExpertTierLease,
        config: ExpertCacheConfig,
        rank: &[u32],
        max_rows: usize,
    ) -> Result<Self> {
        if gpu.arch != "gfx1201" || config.max_inflight == 0 || max_rows == 0 {
            return Err("expert cache requires gfx1201, a bounded nonzero queue and at least one window row".into());
        }
        gpu.bind_thread().map_err(|e| fail("bind", e))?;
        let rank_pos = rank_positions(rank)?;
        let catalog = admitted_catalog(&lease)?;
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
        let hip = Arc::new(HipRuntime::load().map_err(|e| fail("HIP", e))?);
        hip.set_device(gpu.device_id)
            .map_err(|e| fail("device", e))?;
        let mut layers = Vec::with_capacity(LAYERS);
        let mut inputs = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            let view = lease
                .expert_cache_layer(layer)
                .ok_or("expert cache missing canonical layer")?;
            let slots = lease
                .expert_cache_slots(layer)
                .ok_or("expert cache missing slot metadata")?;
            let mut canonical_gate_up: Vec<ResidencySpan> = Vec::with_capacity(EXPERTS);
            let mut canonical_down: Vec<ResidencySpan> = Vec::with_capacity(EXPERTS);
            let mut experts = Vec::with_capacity(EXPERTS);
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
                let host = if entry.tier == Placement::HostMapped {
                    Some(HostPair {
                        gate_up: host_projection(gu.address, gu.bytes, host_alias, host_cpu)?,
                        down: host_projection(dn.address, dn.bytes, host_alias, host_cpu)?,
                    })
                } else {
                    None
                };
                canonical_gate_up.push(gu);
                canonical_down.push(dn);
                experts.push(ExpertInput {
                    gate_up: gu,
                    down: dn,
                    host,
                });
            }
            let tables = lease
                .allocate_expert_cache_tables(gpu, layer)
                .map_err(|e| fail("tables", e))?;
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
            let table_bindings = [&gu, &dn].map(|table| Binding {
                base: table.buf.as_ptr() as u64,
                bytes: table.buf.size() as u64,
            });
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
            inputs.push(LayerInput {
                experts,
                slots: slots
                    .iter()
                    .map(|slot| SlotInput {
                        original_expert: slot.ranges.original_expert,
                        gate_up: slot.ranges.gate_up_device,
                        down: slot.ranges.down_device,
                    })
                    .collect(),
                tables: table_bindings,
            });
            layers.push(LayerCache {
                binding: ExpertCacheLayerBinding { tables, residency },
                gate_up_upload: vec![0; EXPERTS * 8],
                down_upload: vec![0; EXPERTS * 8],
            });
        }
        let layout = build_layout(&catalog, &inputs, &rank_pos)?;
        let pools = layout.pools.len();
        if pools == 0 {
            return Err("expert cache has no evictable slot".into());
        }
        let fills_per_pool = config.max_inflight.div_ceil(pools).max(1);
        let events = fills_per_pool * pools;
        let slots = layout.slots();
        let use_log = LAYERS
            .checked_mul(TOPK)
            .and_then(|uses| uses.checked_mul(max_rows))
            .ok_or("expert cache route-use extent overflows")?;
        let policy_config = PolicyConfig {
            // One window per AR token or verify/repair window: every window closes its own
            // group. The log holds every distinct use of the widest window.
            retire_ids: 1,
            max_windows: 1,
            fill_group_cap: 1,
            use_log,
            // A warm seed's completions are all buffered before the one plan that stages them.
            completions: slots + events,
            parked: slots,
            consumers: 1,
            queues: 1,
        };
        let dests = layout
            .pools
            .iter()
            // SAFETY: the pool's hull lies inside the donor tail the lease retains.
            .map(|pool| unsafe { PoolDest::new(pool.block.0 as u64, pool.block.1 as u64) })
            .collect();
        let backend =
            HipCopyBackend::new(hip.clone(), dests, events).map_err(|e| fail("copy backend", e))?;
        let io = DeviceIo::new(backend).map_err(|e| fail("device io", e))?;
        let mut cache = Self {
            hip,
            device_id: gpu.device_id,
            max_rows,
            lease: Some(lease),
            layers,
            layout,
            engine: None,
            policy_config,
            fills_per_pool,
            donor_device: ranges.donor_device,
            mirror_cpu: ranges.mirror_cpu,
            restore_epoch: 0,
            dirty: false,
            poisoned: None,
            outstanding: Arc::new(AtomicBool::new(false)),
            stats: ExpertCacheStats::default(),
        };
        // The donor slots hold their original experts' bytes right now: warm seed, no copy.
        cache.install_seeded(gpu, io, false)?;
        Ok(cache)
    }
    pub(crate) fn layer_binding(&self, layer: usize) -> Option<ExpertCacheLayerBinding> {
        self.layers.get(layer).map(|layer| layer.binding.clone())
    }
    /// Cumulative counters: the adapter's own plus the live policy's and io's.
    pub(crate) fn stats(&self) -> ExpertCacheStats {
        let mut stats = self.stats;
        if let Some(engine) = &self.engine {
            let live = engine.policy.stats();
            stats.fills_started += live.fills_issued;
            stats.fills_completed += live.fills_published;
            stats.evictions += live.evictions;
            stats.skipped_full += live.skipped_full;
            stats.fill_bytes = engine.io.stats().bytes_enqueued;
        }
        stats
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
        note_mapped_prefill(&mut self.stats, rows)
    }

    fn emit_stats(&self) {
        if tracing::enabled!(tracing::Level::DEBUG) {
            tracing::debug!(stats = ?self.stats(), "qwen4 exact expert cache");
        }
    }

    pub(crate) fn check_readers_idle(&self) -> Result<()> {
        reader_gate(
            self.poisoned.as_deref(),
            self.outstanding.load(Ordering::Acquire),
            self.engine.is_some(),
        )
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
        gpu.bind_thread().map_err(|e| fail("bind", e))
    }

    /// Run `f` with the engine taken out of `self`, and put it back whatever `f` returns.
    fn with_engine<R>(
        &mut self,
        f: impl FnOnce(&mut Self, &mut Engine<QwenCopy>) -> Result<R>,
    ) -> Result<R> {
        let mut engine = self.engine.take().ok_or("expert cache is stopped")?;
        let result = f(self, &mut engine);
        self.engine = Some(engine);
        result
    }

    fn fold_policy_stats(&mut self, retired: CacheStats) {
        self.stats.fills_started += retired.fills_issued;
        self.stats.fills_completed += retired.fills_published;
        self.stats.evictions += retired.evictions;
        self.stats.skipped_full += retired.skipped_full;
    }

    pub(crate) fn begin_token(
        &mut self,
        gpu: &mut Gpu,
        phase: WindowConsumerPhase,
    ) -> Result<ExpertCacheToken> {
        let started = Instant::now();
        let result = self.begin_token_inner(gpu, phase);
        self.stats.begin_token_wall_ns = self
            .stats
            .begin_token_wall_ns
            .saturating_add(elapsed_wall_ns(started));
        result
    }

    fn begin_token_inner(
        &mut self,
        gpu: &mut Gpu,
        phase: WindowConsumerPhase,
    ) -> Result<ExpertCacheToken> {
        self.check(gpu)?;
        let gpu: &Gpu = gpu;
        let result = self.with_engine(|this, engine| {
            this.publish_completed(gpu, engine, false)?;
            let mut guards = Vec::with_capacity(LAYERS);
            for layer in &this.layers {
                let cap = &layer.binding.residency;
                let acquired = cap
                    .snapshot()
                    .and_then(|snapshot| cap.begin_token(snapshot.epoch));
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
            let opened = engine
                .policy
                .begin_window()
                .and_then(|_| engine.policy.register_consumer(0, phase, 0));
            let consumer = match opened {
                Ok(consumer) => consumer,
                Err(cause) => {
                    // The window never reached a queue: proven retired.
                    let _ = engine.policy.cancel_window(CancelOutcome::Retired);
                    // SAFETY: window and guards precede every token submission.
                    for guard in guards {
                        unsafe { guard.retire_after_gpu_wait() }?;
                    }
                    return Err(fail("window", cause));
                }
            };
            this.outstanding.store(true, Ordering::Release);
            Ok(ExpertCacheToken {
                guards,
                consumer,
                outstanding: this.outstanding.clone(),
            })
        });
        if let Err(reason) = &result {
            self.poisoned = Some(reason.clone());
        }
        result
    }

    /// Retire the token's guards and join (`finished`) or cancel its window.
    ///
    /// # Safety
    /// Every GPU reader of the token is proven retired: the caller's route readback wait for a
    /// finished token, a device wait for a cancelled one, and for every independent PM4 reader its
    /// own checked retirement (or it never submitted).
    unsafe fn retire_token(
        &mut self,
        gpu: &mut Gpu,
        token: ExpertCacheToken,
        finished: bool,
    ) -> Result<()> {
        if !Arc::ptr_eq(&token.outstanding, &self.outstanding) || gpu.device_id != self.device_id {
            return Err("expert cache reader belongs to another cache or GPU".into());
        }
        gpu.bind_thread().map_err(|e| fail("retire bind", e))?;
        let engine = self.engine.as_mut().ok_or("expert cache is stopped")?;
        let window = if finished {
            engine.policy.join_consumer(token.consumer)
        } else {
            engine.policy.cancel_window(CancelOutcome::Retired)
        };
        if let Err(cause) = window {
            let reason = fail("window", cause);
            self.poisoned = Some(reason.clone());
            return Err(reason);
        }
        for guard in token.guards {
            // SAFETY: caller's proof above.
            unsafe { guard.retire_after_gpu_wait() }?;
        }
        self.outstanding.store(false, Ordering::Release);
        Ok(())
    }

    /// Cancel without observing routes, changing recency or issuing payload copies.
    /// # Safety
    /// Every independent PM4 reader is proven retired (or never submitted) and the caller waited
    /// for the HIP work of the token.
    /// Unknown callers must retain/drop the token and quarantine its complete owner set.
    pub(crate) unsafe fn cancel_after_gpu_wait(
        &mut self,
        gpu: &mut Gpu,
        token: ExpertCacheToken,
    ) -> Result<()> {
        // SAFETY: forwarded from this function's caller.
        unsafe { self.retire_token(gpu, token, false) }
    }

    /// # Safety
    /// Every independent PM4 reader is proven retired; IDs are this window's complete readback
    /// (`rows` rows, `[layer][row][k]`), which the caller waited for.
    pub(crate) unsafe fn finish_token(
        &mut self,
        gpu: &mut Gpu,
        token: ExpertCacheToken,
        ids: &[i32],
        rows: usize,
    ) -> Result<()> {
        let started = Instant::now();
        let snapshots: Vec<_> = token.guards.iter().map(|g| g.snapshot().clone()).collect();
        // SAFETY: forwarded from this function's caller.
        let retired = unsafe { self.retire_token(gpu, token, true) };
        let result = retired.and_then(|()| {
            if validate_window_ids(ids, rows, self.max_rows).is_err() {
                // The consumer joined: the window is idle again once cancelled as retired.
                if let Some(engine) = self.engine.as_mut() {
                    let _ = engine.policy.cancel_window(CancelOutcome::Retired);
                }
                return Err("expert cache invalid route readback".into());
            }
            let gpu: &Gpu = gpu;
            self.with_engine(|this, engine| this.observe(gpu, engine, ids, rows, &snapshots))
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

    /// The window's routes become the group's uses; the group is judged, retired when it must be,
    /// and the fills it decided are issued.
    fn observe(
        &mut self,
        gpu: &Gpu,
        engine: &mut Engine<QwenCopy>,
        ids: &[i32],
        rows: usize,
        snapshots: &[ResidencySnapshot],
    ) -> Result<()> {
        let result = self.observe_inner(gpu, engine, ids, rows, snapshots);
        if result.is_err() {
            // The consumer joined; a still-open window is idle again once cancelled as retired.
            let _ = engine.policy.cancel_window(CancelOutcome::Retired);
        }
        result
    }

    fn observe_inner(
        &mut self,
        gpu: &Gpu,
        engine: &mut Engine<QwenCopy>,
        ids: &[i32],
        rows: usize,
        snapshots: &[ResidencySnapshot],
    ) -> Result<()> {
        self.stats.tokens = self.stats.tokens.saturating_add(1);
        self.stats.rows = self.stats.rows.saturating_add(rows as u64);
        if rows > 1 {
            self.stats.verify_windows = self.stats.verify_windows.saturating_add(1);
        }
        let account = observe_window_routes(
            &self.layout,
            ids,
            rows,
            |layer| &snapshots[layer].mapping[..],
            |used| {
                engine
                    .policy
                    .record_use(used)
                    .map_err(|e| fail("route use", e))
            },
        )?;
        self.stats.selected = self.stats.selected.saturating_add(account.selected);
        self.stats.core_hits = self.stats.core_hits.saturating_add(account.core_hits);
        self.stats.cache_hits = self.stats.cache_hits.saturating_add(account.cache_hits);
        self.stats.host_misses = self.stats.host_misses.saturating_add(account.host_misses);
        self.stats.miss_bytes = self.stats.miss_bytes.saturating_add(account.miss_bytes);
        let core = account.core_hits;
        let windows = engine
            .policy
            .end_window(ids.len() as u64)
            .map_err(|e| fail("window end", e))?
            .ok_or("expert cache cadence did not close its one-window group")?;
        // Deliver what the copy stream finished meanwhile; the plan stages those completions.
        engine
            .io
            .poll(&mut engine.policy, false)
            .map_err(|e| fail("fill completion", e))?;
        engine
            .policy
            .plan_group(windows, (core, core))
            .map_err(|e| fail("plan", e))?;
        // Only a staged completion or a withdrawn victim retires anything (A2.1).
        self.retire_and_publish(gpu, engine, false)?;
        // The newly sealed membership excludes every victim before copy enqueue.
        let issued = engine
            .io
            .issue_pending(&mut engine.policy, &self.layout)
            .map_err(|e| fail("fill enqueue", e))?;
        if issued > 0 {
            self.dirty = true;
            self.stats.peak_inflight = self.stats.peak_inflight.max(engine.io.in_flight());
        }
        Ok(())
    }

    /// Deliver the fills the copy stream completed and publish them (a publication-only plan).
    fn publish_completed(
        &mut self,
        gpu: &Gpu,
        engine: &mut Engine<QwenCopy>,
        wait: bool,
    ) -> Result<()> {
        engine
            .io
            .poll(&mut engine.policy, wait)
            .map_err(|e| fail("fill completion", e))?;
        if engine.policy.completion_room() < engine.policy.config().completions {
            engine
                .policy
                .plan_group(0, (0, 0))
                .map_err(|e| fail("publication plan", e))?;
            self.retire_and_publish(gpu, engine, false)?;
        }
        Ok(())
    }

    /// The retirement protocol of every pool with something pending: fence the compute stream,
    /// retire the generations, rewrite the tables, reclaim the withdrawn victims (which starts the
    /// fills deferred into them) and seal the next generation. A no-op when nothing is pending.
    fn retire_and_publish(
        &mut self,
        gpu: &Gpu,
        engine: &mut Engine<QwenCopy>,
        reseed: bool,
    ) -> Result<()> {
        if !engine.policy.needs_retirement() {
            return Ok(());
        }
        let mut dirty = Vec::new();
        {
            let mut fence = ComputeFence::new(&self.hip, gpu.active_stream.as_ref());
            for index in 0..engine.policy.pools() {
                let pool = PoolId(index as u16);
                if engine.policy.pool_needs_retirement(pool) {
                    engine
                        .policy
                        .retire_generation(pool, &mut fence)
                        .map_err(|e| fail("generation retirement", e))?;
                    dirty.push(pool);
                }
            }
        }
        self.write_tables(gpu, &engine.policy, reseed)?;
        for pool in dirty {
            engine
                .policy
                .finish_plan(pool)
                .map_err(|e| fail("reclaim", e))?;
            engine.policy.seal(pool).map_err(|e| fail("seal", e))?;
        }
        Ok(())
    }

    /// Write `withdrawn()` / `published()` into the layers' device tables. `reseed` rebuilds the
    /// whole mapping from the canonical one (the seed of a restore); otherwise the published
    /// mapping is edited.
    fn write_tables(&mut self, gpu: &Gpu, policy: &ExpertCachePolicy, reseed: bool) -> Result<()> {
        let changes = layer_changes(policy.withdrawn(), policy.published(), self.restore_epoch)?;
        for (layer, change) in changes.into_iter().enumerate() {
            if change.withdrawn.is_empty() && change.published.is_empty() {
                continue;
            }
            let current = if reseed {
                None
            } else {
                Some(
                    self.layers[layer]
                        .binding
                        .residency
                        .snapshot()?
                        .mapping
                        .to_vec(),
                )
            };
            let mapping = next_mapping(current, &change);
            self.publish(gpu, layer, mapping, change.published)?;
        }
        Ok(())
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
        // SAFETY: no token guards are active, and prior reads retired (the generation fence
        // above). Completed tickets passed event_query or name the restored original donor
        // payloads. Both tables are uploaded and synchronized before host publication.
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
                            .map_err(|e| fail("table upload", e))?;
                        self.stats.table_upload_bytes = self
                            .stats
                            .table_upload_bytes
                            .saturating_add(u64::try_from(upload.len()).unwrap_or(u64::MAX));
                    }
                    match gpu.active_stream.as_ref() {
                        Some(stream) => self.hip.stream_synchronize(stream),
                        None => self.hip.stream_synchronize_default(),
                    }
                    .map_err(|e| fail("table visibility", e))
                },
            )
        }?;
        Ok(())
    }

    fn allocation_generation(&self) -> u64 {
        self.lease.as_ref().map_or(0, ExpertTierLease::generation)
    }

    /// Replace the pools by a fresh build whose slots hold their donors' original experts,
    /// published into the tables. `restore`: first copy each pool hull back from the mirror in
    /// one bulk copy per pool (the restore itself); the seed fills then copy nothing and only
    /// record their events on the same copy stream, behind the bulk copies. The engine is always
    /// installed, so teardown sees it.
    fn install_seeded(&mut self, gpu: &Gpu, io: DeviceIo<QwenCopy>, restore: bool) -> Result<()> {
        let policy = build_policy(
            &self.layout,
            self.policy_config,
            self.fills_per_pool,
            self.allocation_generation(),
        )?;
        let donor = restore.then_some(DonorTail {
            device: self.donor_device,
            mirror: self.mirror_cpu,
        });
        let seeded = issue_seed_pools(&self.layout, policy, io, donor);
        self.publish_seeded(gpu, seeded)
    }

    /// Install a seeded engine and publish its seed over the canonical mapping: one plan stages
    /// every seed completion, the generation retires behind a compute-stream fence, the tables are
    /// rewritten from the canonical mapping (no slot of any earlier pool survives it) and the
    /// seed's membership is sealed. The engine is installed whether or not this succeeds.
    fn publish_seeded(&mut self, gpu: &Gpu, seeded: Seeded<QwenCopy>) -> Result<()> {
        let mut engine = seeded.engine;
        let result = seeded.issued.and_then(|()| {
            engine
                .policy
                .plan_group(0, (0, 0))
                .map_err(|e| fail("seed staging", e))?;
            self.retire_and_publish(gpu, &mut engine, true)?;
            for index in 0..engine.policy.pools() {
                engine
                    .policy
                    .check_seeded(PoolId(index as u16))
                    .map_err(|e| fail("seed", e))?;
            }
            Ok(())
        });
        self.engine = Some(engine);
        result
    }

    /// Make the Resident (original donor) weights readable for a prefill. A cache whose fills
    /// never reached a donor slot still holds them (`!dirty`: nothing to do). Otherwise, with no
    /// token and no window open (a refusal changes nothing and leaves the cache usable), every
    /// fill is drained, each pool hull is rewritten from the pinned mirror in one bulk copy, the
    /// rebuilt pools' seed records its events behind those copies, and the seed is published over
    /// the CANONICAL mapping: an expert the cache had filled names no slot any more, and the
    /// re-seeded tickets outrank every earlier one ([`ticket_generation`]). All of it completes
    /// before this returns, i.e. before the prefill's first kernel. Nothing reads a donor slot
    /// while it is rewritten: the caller holds `&mut` over the cache, no token or retained queue
    /// is outstanding, and a mapped prefill tile still in flight reads the immutable core/host
    /// tables, never a donor slot. A failure after the first byte moved poisons the cache (the
    /// donors may be half restored).
    ///
    /// Prefill never observes routes: it records no use, so a scan-heavy prompt promotes no
    /// expert, and the restore resets recency to the seed's (every donor stamped alike).
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
        if !self.dirty {
            return Ok(());
        }
        // Before anything moves: the epoch the re-seeded tickets carry.
        let next_epoch = self
            .restore_epoch
            .checked_add(1)
            .ok_or("expert cache restore epoch exhausted")?;
        let gpu: &Gpu = gpu;
        let allocation = self.allocation_generation();
        let donor = DonorTail {
            device: self.donor_device,
            mirror: self.mirror_cpu,
        };
        let restored = match restore_pools(
            &mut self.engine,
            &self.layout,
            self.policy_config,
            self.fills_per_pool,
            allocation,
            donor,
        ) {
            Ok(restored) => restored,
            Err(RestoreError::Busy) => {
                return Err("expert cache restore needs an idle window".into());
            }
            Err(RestoreError::Failed(reason)) => {
                self.poisoned = Some(reason.clone());
                return Err(reason);
            }
        };
        self.fold_policy_stats(restored.retired);
        self.restore_epoch = next_epoch;
        let result = self.publish_seeded(gpu, restored.seeded).map(|()| {
            self.stats.prefill_restores += 1;
            self.stats.restore_bytes += self.donor_device.1 as u64;
            self.dirty = false;
        });
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
        gpu.bind_thread().map_err(|e| fail("shutdown bind", e))?;
        self.hip
            .device_synchronize()
            .map_err(|e| fail("shutdown wait", e))?;
        self.destroy_engine()
    }
    /// Wait every fill (the io synchronizes each event), then drop the pools: the device is idle,
    /// so their generations and leases are CPU ledgers only. An open or poisoned window, or an
    /// event that cannot be waited, is a refusal and the caller retains every owner.
    fn destroy_engine(&mut self) -> Result<()> {
        if let Some(engine) = &self.engine {
            if !engine.policy.window().is_idle() {
                return Err("expert cache teardown has an unjoined window; retain owners".into());
            }
        }
        let Some(Engine { policy, io }) = self.engine.take() else {
            return Ok(());
        };
        self.fold_policy_stats(policy.stats());
        io.shutdown().map_err(|e| fail("shutdown", e))?;
        drop(policy);
        Ok(())
    }
    fn retain_on_unknown_quiescence(&mut self) {
        // Retain canonical bytes, slots, tables and upload mirrors. A poisoned
        // token can still have independent PM4 reads that HIP cannot fence.
        std::mem::forget(std::mem::take(&mut self.layers));
        if let Some(engine) = self.engine.take() {
            std::mem::forget(engine);
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
            if self.destroy_engine().is_err() {
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

/// The catalog of the lease's admitted plan, proven to be the one admission recorded: the keys
/// the cache uses name the admitted canonical experts, never a rebuilt or different layout.
fn admitted_catalog(lease: &ExpertTierLease) -> Result<ExpertCatalog> {
    let plan = lease.tier_plan();
    let catalog = build_expert_catalog(plan).map_err(|e| fail("catalog", e))?;
    let recorded = plan
        .expert_cache_identity()
        .ok_or("exact expert cache plan carries no admitted catalog identity")?;
    let identity = catalog
        .identity(&[])
        .map_err(|e| fail("catalog identity", e))?;
    if identity.as_bytes() != recorded {
        return Err(
            "exact expert cache catalog differs from the identity admitted with the plan".into(),
        );
    }
    Ok(catalog)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_slab_table::{ExpertSlab, ExpertSlabSpan};
    use crate::expert_slabs::ExpertFormat;
    use crate::expert_tier_plan::{plan_expert_tiers_from_slabs, ExpertTierPlan};
    use hipfire_runtime::expert_cache::{validate_bundle, Lookup, PlanSummary, SlotPhase};
    use std::cell::RefCell;
    use std::rc::Rc;

    const DEVICE: usize = 0x1_0000_0000;
    const HOST: usize = 0x9000_0000;
    /// Padded device bytes of one expert (both projections 256-aligned).
    const SPAN: u64 = 2_611_200;

    // -- fixtures -------------------------------------------------------------------------

    fn real_slabs() -> Vec<ExpertSlab> {
        let format = ExpertFormat::Mq4Xts;
        let mut cursor = 0;
        (0..(TRUNK_LAYERS * EXPERTS_PER_LAYER) as usize)
            .map(|_| {
                let gate_up = ExpertSlabSpan {
                    offset: cursor,
                    len: format.gate_up_bytes(),
                };
                cursor += format.gate_up_bytes();
                let down = ExpertSlabSpan {
                    offset: cursor,
                    len: format.down_bytes(),
                };
                cursor += format.down_bytes();
                ExpertSlab {
                    format,
                    gate_up,
                    down,
                }
            })
            .collect()
    }

    /// A plan of 16 device experts (layer 0) with the `donors` coldest of them donated.
    fn cached_plan(donors: u64) -> ExpertTierPlan {
        let rank: Vec<u32> = (0..16).collect();
        let plain = plan_expert_tiers_from_slabs(&real_slabs(), &rank, u64::MAX).unwrap();
        let (plan, tail) = plain.with_expert_cache_tail(&rank, donors * SPAN).unwrap();
        assert_eq!(tail.donors.len(), donors as usize);
        crate::expert_cache_config::record_catalog_identity(plan).unwrap()
    }

    fn inputs(plan: &ExpertTierPlan) -> Vec<LayerInput> {
        let tail = plan.expert_cache_tail().unwrap();
        (0..LAYERS)
            .map(|layer| {
                let experts = (0..EXPERTS)
                    .map(|expert| {
                        let entry = plan.entry(layer, expert).unwrap();
                        let id = layer * EXPERTS + expert;
                        let cacheable = entry.tier == Placement::HostMapped
                            || tail.donors.binary_search(&(id as u32)).is_ok();
                        let gate_up = entry.gate_up_len as usize;
                        let down = entry.down_len as usize;
                        ExpertInput {
                            gate_up: ResidencySpan {
                                address: 0x4000_0000 + id * 0x40_0000,
                                bytes: gate_up,
                            },
                            down: ResidencySpan {
                                address: 0x4020_0000 + id * 0x40_0000,
                                bytes: down,
                            },
                            host: cacheable.then(|| HostPair {
                                gate_up: (HOST + id * 0x40_0000, gate_up),
                                down: (HOST + id * 0x40_0000 + 0x20_0000, down),
                            }),
                        }
                    })
                    .collect();
                let slots = tail
                    .donors
                    .iter()
                    .filter(|&&id| id as usize / EXPERTS == layer)
                    .map(|&id| {
                        let entry = plan.entry(layer, id as usize % EXPERTS).unwrap();
                        SlotInput {
                            original_expert: id as usize % EXPERTS,
                            gate_up: (
                                DEVICE + entry.gate_up_offset as usize,
                                entry.gate_up_len as usize,
                            ),
                            down: (DEVICE + entry.down_offset as usize, entry.down_len as usize),
                        }
                    })
                    .collect();
                LayerInput {
                    experts,
                    slots,
                    tables: [0, 1].map(|table| Binding {
                        base: 0x7000_0000 + (layer * 2 + table) as u64 * 0x1000,
                        bytes: 4096,
                    }),
                }
            })
            .collect()
    }

    fn rank_pos() -> Vec<u32> {
        (0..(LAYERS * EXPERTS) as u32).collect()
    }

    fn layout_of(plan: &ExpertTierPlan) -> Layout {
        let catalog = build_expert_catalog(plan).unwrap();
        build_layout(&catalog, &inputs(plan), &rank_pos()).unwrap()
    }

    fn config() -> PolicyConfig {
        PolicyConfig {
            retire_ids: 1,
            max_windows: 1,
            fill_group_cap: 1,
            use_log: LAYERS * TOPK,
            completions: 64,
            parked: 64,
            consumers: 1,
            queues: 1,
        }
    }

    // -- a fake copy device ---------------------------------------------------------------

    #[derive(Default)]
    struct Trace {
        /// `copy (dst, len)` and `record` in submission order.
        log: RefCell<Vec<String>>,
        copies: RefCell<Vec<(u64, usize)>>,
        /// Every copy, record and wait (a query or a synchronize) in submission order.
        seq: RefCell<Vec<&'static str>>,
    }
    struct Fake {
        events: usize,
        trace: Rc<Trace>,
    }
    impl CopyDevice for Fake {
        fn incarnation(&self) -> u64 {
            1
        }
        fn events(&self) -> usize {
            self.events
        }
        fn enqueue_copy(
            &self,
            _: PoolId,
            dst: u64,
            src: HostRange,
        ) -> std::result::Result<(), String> {
            self.trace.copies.borrow_mut().push((dst, src.len()));
            self.trace.log.borrow_mut().push("copy".into());
            self.trace.seq.borrow_mut().push("copy");
            Ok(())
        }
        fn record(&self, _: usize) -> std::result::Result<(), String> {
            self.trace.log.borrow_mut().push("record".into());
            self.trace.seq.borrow_mut().push("record");
            Ok(())
        }
        fn query(&self, _: usize) -> std::result::Result<bool, String> {
            self.trace.seq.borrow_mut().push("wait");
            Ok(true)
        }
        fn synchronize(&self, _: usize) -> std::result::Result<(), String> {
            self.trace.seq.borrow_mut().push("wait");
            Ok(())
        }
        fn quarantine(&self) {}
    }

    struct CpuFence(u64);
    // SAFETY: metadata-only tests enqueue no device operation; every fence is complete.
    unsafe impl FenceBackend for CpuFence {
        fn incarnation(&self) -> u64 {
            7
        }
        fn record(&mut self) -> std::result::Result<u64, FenceFailure> {
            self.0 += 1;
            Ok(self.0)
        }
        fn poll(&mut self, _: u64) -> FenceState {
            FenceState::Complete
        }
        fn wait(&mut self, _: u64) -> FenceState {
            FenceState::Complete
        }
    }

    type Rig = Engine<Fake>;

    fn rig(layout: &Layout, events: usize) -> (Rig, Rc<Trace>) {
        let trace = Rc::new(Trace::default());
        let policy = build_policy(layout, config(), 4, 9).unwrap();
        let io = DeviceIo::new(Fake {
            events,
            trace: trace.clone(),
        })
        .unwrap();
        (Engine { policy, io }, trace)
    }

    /// Stage and publish what the policy completed (the table rewrite is the dispatch capability's
    /// job and is not under test): retire, reclaim, seal.
    fn retire_all(engine: &mut Rig) {
        let mut fence = CpuFence(0);
        let pools = engine.policy.pools();
        let dirty: Vec<_> = (0..pools)
            .map(|i| PoolId(i as u16))
            .filter(|&p| engine.policy.pool_needs_retirement(p))
            .collect();
        for &pool in &dirty {
            engine.policy.retire_generation(pool, &mut fence).unwrap();
        }
        for &pool in &dirty {
            engine.policy.finish_plan(pool).unwrap();
            engine.policy.seal(pool).unwrap();
        }
    }

    // -- ported protocol and helper tests -----------------------------------------------------

    #[test]
    fn route_accounting_uses_token_start_mapping() {
        let mut layout = layout_of(&cached_plan(5));
        // Layer 0: experts 0 and 1 are the pinned core, 2 and 3 are in slots, the rest miss.
        for expert in [0, 1] {
            layout.reads[expert] = Read::Core;
        }
        for expert in 2..10 {
            layout.reads[expert] = Read::Host { bytes: 5 };
        }
        let mut mapping = vec![ResidencyLocation::Canonical; EXPERTS];
        for expert in [2, 3] {
            mapping[expert] = ResidencyLocation::Slot(ResidencySlotTicket {
                slot: expert - 2,
                expert,
                generation: 1,
            });
        }
        let ids: Vec<_> = (0..10).collect();
        let account = account_route(&layout, 0, &ids, &mapping).unwrap();
        assert_eq!(
            (
                account.selected,
                account.core_hits,
                account.cache_hits,
                account.host_misses,
                account.miss_bytes
            ),
            (10, 2, 2, 6, 30)
        );
        // A fill after the token cannot retroactively turn its host read into a hit.
        mapping[4] = ResidencyLocation::Slot(ResidencySlotTicket {
            slot: 2,
            expert: 4,
            generation: 2,
        });
        assert_eq!(account.host_misses, 6);
        assert!(account_route(&layout, 0, &[7, 7], &mapping).is_err());
        assert!(account_route(&layout, 0, &[-1], &mapping).is_err());
        assert!(account_route(&layout, 0, &[EXPERTS as i32], &mapping).is_err());
    }

    #[test]
    fn window_ids_must_be_exactly_rows_by_layers_by_topk_in_range() {
        let ok = vec![0i32; LAYERS * TOPK * 3];
        assert!(validate_window_ids(&ok, 3, 64).is_ok());
        assert!(validate_window_ids(&ok[..LAYERS * TOPK], 1, 64).is_ok());
        // A wrong length for the claimed rows, zero rows and too many rows refuse.
        assert!(validate_window_ids(&ok, 2, 64).is_err());
        assert!(validate_window_ids(&ok, 4, 64).is_err());
        assert!(validate_window_ids(&ok[..LAYERS * TOPK], 0, 64).is_err());
        assert!(validate_window_ids(&ok, 3, 2).is_err());
        assert!(validate_window_ids(&ok, usize::MAX, usize::MAX).is_err());
        // Every id must be a valid expert.
        let mut bad = ok.clone();
        bad[7] = -1;
        assert!(validate_window_ids(&bad, 3, 64).is_err());
        bad[7] = EXPERTS as i32;
        assert!(validate_window_ids(&bad, 3, 64).is_err());
    }

    /// Every expert below 64 of every layer is a cacheable read ranked by its id.
    fn cacheable_layout() -> Layout {
        let mut layout = layout_of(&cached_plan(5));
        for layer in 0..LAYERS {
            for expert in 0..64 {
                layout.reads[layer * EXPERTS + expert] = Read::Cache {
                    pool: PoolId(0),
                    rank: expert as u32,
                    bytes: 5,
                };
            }
        }
        layout
    }

    /// `[layer][row][k]` ids: row `r` of every layer selects experts `r * 10 .. r * 10 + 10`.
    fn window_ids(rows: usize) -> Vec<i32> {
        let mut ids = Vec::with_capacity(LAYERS * rows * TOPK);
        for _layer in 0..LAYERS {
            for row in 0..rows {
                ids.extend((0..TOPK).map(|j| (row * TOPK + j) as i32));
            }
        }
        ids
    }

    fn observed(
        layout: &Layout,
        ids: &[i32],
        rows: usize,
    ) -> Result<(Vec<(u32, u32)>, RouteAccount)> {
        let mapping = vec![ResidencyLocation::Canonical; EXPERTS];
        let mut order = Vec::new();
        let account = observe_window_routes(
            layout,
            ids,
            rows,
            |_| &mapping[..],
            |used| {
                order.push((used.key.group_id, used.key.expert_id));
                Ok(())
            },
        )?;
        Ok((order, account))
    }

    #[test]
    fn a_one_row_window_observes_layer_major_as_before() {
        let layout = cacheable_layout();
        let (order, account) = observed(&layout, &window_ids(1), 1).unwrap();
        let expected: Vec<_> = (0..LAYERS as u32)
            .flat_map(|layer| (0..TOPK as u32).map(move |j| (layer, j)))
            .collect();
        assert_eq!(order, expected);
        assert_eq!(account.selected, (LAYERS * TOPK) as u64);
    }

    #[test]
    fn a_multi_row_window_observes_layer_then_row_then_rank() {
        let layout = cacheable_layout();
        let rows = 3;
        let (order, account) = observed(&layout, &window_ids(rows), rows).unwrap();
        let expected: Vec<_> = (0..LAYERS as u32)
            .flat_map(|layer| {
                (0..rows as u32).flat_map(move |row| {
                    (0..TOPK as u32).map(move |j| (layer, row * TOPK as u32 + j))
                })
            })
            .collect();
        assert_eq!(order, expected);
        assert_eq!(account.selected, (LAYERS * rows * TOPK) as u64);
    }

    #[test]
    fn one_expert_in_two_rows_is_accepted_but_a_duplicate_inside_a_row_is_refused() {
        let layout = cacheable_layout();
        let rows = 3;
        // Rows 0 and 1 select the same experts in every layer.
        let mut ids = window_ids(rows);
        for layer in 0..LAYERS {
            let row0 = (layer * rows) * TOPK;
            let row1 = (layer * rows + 1) * TOPK;
            for j in 0..TOPK {
                ids[row1 + j] = ids[row0 + j];
            }
        }
        let (order, account) = observed(&layout, &ids, rows).unwrap();
        assert_eq!(account.selected, (LAYERS * rows * TOPK) as u64);
        assert_eq!(order.len(), LAYERS * rows * TOPK);
        // A duplicate inside row 1 of layer 5 refuses.
        let mut dup = window_ids(rows);
        let row1 = (5 * rows + 1) * TOPK;
        dup[row1 + 3] = dup[row1];
        assert!(observed(&layout, &dup, rows).is_err());
        // A readback shorter than rows x layers x topk refuses.
        assert!(observed(&layout, &window_ids(2), rows).is_err());
    }

    #[test]
    fn cpu_alias_translation_checks_both_ends() {
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
    fn rank_positions_require_a_complete_permutation() {
        let mut rank: Vec<_> = (0..(LAYERS * EXPERTS) as u32).collect();
        rank.reverse();
        let position = rank_positions(&rank).unwrap();
        // The hottest id (rank[0]) has position 0.
        assert_eq!(position[rank[0] as usize], 0);
        assert_eq!(position[0], (LAYERS * EXPERTS - 1) as u32);
        assert!(rank_positions(&rank[..rank.len() - 1]).is_err());
        rank[1] = rank[0];
        assert!(rank_positions(&rank).is_err());
        rank[1] = (LAYERS * EXPERTS) as u32;
        assert!(rank_positions(&rank).is_err());
    }

    #[test]
    fn ticket_generations_never_go_backwards_across_restores() {
        // A restore rebuilds the bank, so fill generations restart at 1; the ticket must not.
        let before = ticket_generation(0, 41).unwrap();
        let after = ticket_generation(1, 1).unwrap();
        assert!(after > before && ticket_generation(0, 1).unwrap() >= 1);
        assert!(ticket_generation(0, 1 << 32).is_err());
        assert!(ticket_generation(u64::MAX, 1).is_err());
    }

    // -- layout ---------------------------------------------------------------------------

    #[test]
    fn layout_pools_the_donor_layer_and_names_every_cacheable_bundle() {
        let plan = cached_plan(5);
        let layout = layout_of(&plan);
        // Only layer 0 donates: one pool of its five slots, seeded with the donors' experts.
        assert_eq!(layout.pools.len(), 1);
        assert_eq!(layout.layer_pool[0], Some(PoolId(0)));
        assert!(layout.layer_pool[1..].iter().all(Option::is_none));
        let pool = &layout.pools[0];
        assert_eq!((pool.layer, pool.spans.len(), pool.seed.len()), (0, 5, 5));
        let tail = plan.expert_cache_tail().unwrap();
        for ((key, rank), &donor) in pool.seed.iter().zip(&tail.donors) {
            assert_eq!((key.group_id, key.expert_id), (0, donor % EXPERTS as u32));
            assert_eq!(*rank, donor);
        }
        // Every cacheable expert of the pooled layer has a GU + DN bundle covering a slot exactly;
        // the pinned core has none, and another pool's id is not its pool.
        let slot_spans: Vec<Binding> = pool.spans[0]
            .iter()
            .map(|span| Binding {
                base: pool.block.0 as u64 + span.offset,
                bytes: span.bytes,
            })
            .collect();
        for expert in 0..EXPERTS {
            let key = layout.key(0, expert);
            match layout.reads[expert] {
                Read::Core => assert!(layout.bundle(&key, PoolId(0)).is_none()),
                Read::Cache { pool, .. } => {
                    assert_eq!(pool, PoolId(0));
                    let bundle = layout.bundle(&key, pool).expect("cacheable bundle");
                    assert_eq!(bundle.len(), 2, "GU and DN travel as one bundle");
                    validate_bundle(bundle, &slot_spans).unwrap();
                    assert!(layout.bundle(&key, PoolId(1)).is_none());
                }
                Read::Host { .. } => panic!("a pooled layer has no unpooled cacheable expert"),
            }
        }
        // Layers without donors read the host and never fill.
        assert!(matches!(layout.reads[EXPERTS + 100], Read::Host { .. }));
        assert!(layout.bundle(&layout.key(1, 100), PoolId(0)).is_none());
    }

    #[test]
    fn layout_refuses_a_view_the_admitted_catalog_does_not_describe() {
        let plan = cached_plan(5);
        let catalog = build_expert_catalog(&plan).unwrap();
        // The cache view pins an expert the catalog marks cacheable.
        let mut bad = inputs(&plan);
        bad[0].experts[100].host = None;
        assert!(build_layout(&catalog, &bad, &rank_pos())
            .err()
            .expect("refused")
            .contains("pinned"));
        // A slot whose extents are not its layout class's.
        let mut bad = inputs(&plan);
        bad[0].slots[1].gate_up.1 -= 256;
        assert!(build_layout(&catalog, &bad, &rank_pos())
            .err()
            .expect("refused")
            .contains("layout class"));
        // Two donors of one expert would misalign the seed.
        let mut bad = inputs(&plan);
        bad[0].slots[1].original_expert = bad[0].slots[0].original_expert;
        assert!(build_layout(&catalog, &bad, &rank_pos())
            .err()
            .expect("refused")
            .contains("distinct"));
        // A layer whose experts differ in extent.
        let mut bad = inputs(&plan);
        bad[3].experts[7].down.bytes -= 256;
        assert!(build_layout(&catalog, &bad, &rank_pos())
            .err()
            .expect("refused")
            .contains("equal projection extents"));
    }

    // -- the adapter over the shared core ---------------------------------------------------

    #[test]
    fn a_seed_is_one_event_per_donor_and_a_restore_is_one_bulk_copy_per_pool() {
        let layout = layout_of(&cached_plan(5));
        let start = layout.pools.iter().map(|p| p.block.0).min().unwrap();
        let end = layout
            .pools
            .iter()
            .map(|p| p.block.0 + p.block.1)
            .max()
            .unwrap();
        let donor = (start, end - start);
        let mirror = (0x7000_0000, donor.1);
        // A hull that leaves the donor tail is refused.
        let (probe, _) = rig(&layout, 2);
        assert!(
            restore_hulls(&layout, (donor.0, donor.1 - 1), mirror, probe.io.backend())
                .unwrap_err()
                .contains("leaves the donor tail")
        );
        for restore in [false, true] {
            let (mut engine, trace) = rig(&layout, 2);
            if restore {
                restore_hulls(&layout, donor, mirror, engine.io.backend()).unwrap();
            }
            issue_seed(&mut engine).unwrap();
            let summary = engine.policy.plan_group(0, (0, 0)).unwrap();
            assert_eq!(summary.published, 5, "restore {restore}");
            assert!(summary.needs_retirement);
            assert_eq!(engine.io.in_flight(), 0);
            {
                // Five seed fills in each pool are one batch behind one fence and move no byte;
                // a restore copies each pool hull once, before every seed fence.
                let log = trace.log.borrow();
                let pools = layout.pools.len();
                let copies = if restore { pools } else { 0 };
                assert!(log[..copies].iter().all(|e| e == "copy"));
                assert_eq!(log[copies..], vec!["record"; pools][..]);
                let hulls: Vec<(u64, usize)> = layout.pools[..copies]
                    .iter()
                    .map(|p| (p.block.0 as u64, p.block.1))
                    .collect();
                assert_eq!(*trace.copies.borrow(), hulls);
            }
            retire_all(&mut engine);
            engine.policy.check_seeded(PoolId(0)).unwrap();
            for (slot, (key, _)) in layout.pools[0].seed.iter().enumerate() {
                assert_eq!(
                    engine.policy.lookup(*key),
                    Lookup::Published {
                        pool: PoolId(0),
                        slot: slot as u32,
                        fill_generation: 1
                    }
                );
            }
        }
    }

    /// Pools seeded and published (generation sealed), the trace cleared.
    fn seeded(layout: &Layout, events: usize) -> (Rig, Rc<Trace>) {
        let (mut engine, trace) = rig(layout, events);
        issue_seed(&mut engine).unwrap();
        engine.policy.plan_group(0, (0, 0)).unwrap();
        retire_all(&mut engine);
        trace.log.borrow_mut().clear();
        trace.copies.borrow_mut().clear();
        trace.seq.borrow_mut().clear();
        (engine, trace)
    }

    /// One decode window reading `keys`, judged as its own group the way `finish_token` does.
    fn window(engine: &mut Rig, layout: &Layout, keys: &[ExpertKey]) -> PlanSummary {
        engine.policy.begin_window().unwrap();
        let consumer = engine
            .policy
            .register_consumer(0, WindowConsumerPhase::Decode, 0)
            .unwrap();
        for &key in keys {
            let id = key.group_id as usize * EXPERTS + key.expert_id as usize;
            let Read::Cache { pool, rank, .. } = layout.reads[id] else {
                panic!("a cacheable expert");
            };
            engine
                .policy
                .record_use(WindowUse {
                    key,
                    pool,
                    reads: 1,
                    rank,
                })
                .unwrap();
        }
        engine.policy.join_consumer(consumer).unwrap();
        assert_eq!(
            engine.policy.end_window(keys.len() as u64).unwrap(),
            Some(1)
        );
        engine.policy.plan_group(1, (0, 0)).unwrap()
    }

    #[test]
    fn a_miss_fills_a_victim_with_one_paired_bundle_and_stays_invisible_until_staged() {
        let layout = layout_of(&cached_plan(5));
        let (mut engine, trace) = seeded(&layout, 4);
        // Two host experts (16 and 17 are not donors, so not seeded) are read.
        let missed = [layout.key(0, 16), layout.key(0, 17)];
        let summary = window(&mut engine, &layout, &missed);
        // Every slot is a resident seed: both candidates need a victim, and withdrawing one
        // retires the generation before its slot may be refilled.
        assert_eq!(
            (summary.misses, summary.withdrawn, summary.fills),
            (2, 2, 0)
        );
        assert!(summary.needs_retirement);
        assert_eq!(engine.policy.withdrawn().len(), 2);
        retire_all(&mut engine);
        // Reclaimed: both fills started, both slots Filling, neither key visible.
        assert_eq!(engine.policy.count(PoolId(0), SlotPhase::Filling), 2);
        for key in missed {
            assert!(matches!(engine.policy.lookup(key), Lookup::Pending { .. }));
        }
        assert_eq!(
            engine
                .io
                .issue_pending(&mut engine.policy, &layout)
                .unwrap(),
            2
        );
        // Two fills: GU and DN copies then one event each, and the copies land in the slot spans.
        let log = trace.log.borrow().clone();
        assert_eq!(log, ["copy", "copy", "record", "copy", "copy", "record"]);
        let block = layout.pools[0].block.0 as u64;
        let spans: Vec<(u64, u64)> = layout.pools[0]
            .spans
            .iter()
            .flat_map(|slot| slot.iter().map(move |s| (block + s.offset, s.bytes)))
            .collect();
        for &(dst, len) in trace.copies.borrow().iter() {
            assert!(
                spans.contains(&(dst, len as u64)),
                "copy outside a slot extent"
            );
        }
        // Still invisible until the completions are delivered AND staged.
        assert!(matches!(
            engine.policy.lookup(missed[0]),
            Lookup::Pending { .. }
        ));
        let polled = engine.io.poll(&mut engine.policy, false).unwrap();
        assert_eq!(polled.delivered, 2);
        assert!(matches!(
            engine.policy.lookup(missed[1]),
            Lookup::Pending { .. }
        ));
        let staged = engine.policy.plan_group(0, (0, 0)).unwrap();
        assert_eq!((staged.published, staged.stale), (2, 0));
        for key in missed {
            assert!(matches!(
                engine.policy.lookup(key),
                Lookup::Published { .. }
            ));
        }
    }

    #[test]
    fn a_pending_fill_is_never_evicted_by_a_later_window() {
        let layout = layout_of(&cached_plan(5));
        let (mut engine, _trace) = seeded(&layout, 4);
        let first = [layout.key(0, 16), layout.key(0, 17)];
        window(&mut engine, &layout, &first);
        retire_all(&mut engine);
        assert_eq!(engine.policy.count(PoolId(0), SlotPhase::Filling), 2);
        // Their fills are decided but nothing completed: the next window's misses may only take
        // the three seeds still resident, never the two slots that are filling.
        let second = [layout.key(0, 18), layout.key(0, 19)];
        let summary = window(&mut engine, &layout, &second);
        assert_eq!((summary.withdrawn, summary.skipped_full), (2, 0));
        assert!(engine
            .policy
            .withdrawn()
            .iter()
            .all(|w| !first.contains(&w.key)));
        retire_all(&mut engine);
        assert_eq!(engine.policy.count(PoolId(0), SlotPhase::Filling), 4);
        for key in first {
            assert!(matches!(engine.policy.lookup(key), Lookup::Pending { .. }));
        }
        // The fill budget (F = 4) is spent by the four fills in flight: a third window admits
        // nothing and touches no pending slot although a seed is still resident.
        let third = [layout.key(0, 20), layout.key(0, 21)];
        let summary = window(&mut engine, &layout, &third);
        assert_eq!(
            (summary.misses, summary.withdrawn, summary.fills),
            (2, 0, 0)
        );
        assert!(!summary.needs_retirement);
        assert_eq!(engine.policy.count(PoolId(0), SlotPhase::Filling), 4);
        assert_eq!(engine.policy.count(PoolId(0), SlotPhase::Live), 1);
    }

    #[test]
    fn a_completion_of_dropped_pools_never_publishes_into_rebuilt_ones() {
        let layout = layout_of(&cached_plan(5));
        // A restore rebuilds the pools: a completion the old bank issued cannot name a slot of
        // the new one, whatever its slot index and fill generation.
        let (mut old, _) = rig(&layout, 4);
        let (mut fresh, _) = rig(&layout, 4);
        let op = old.policy.take_fills().next().expect("a seed fill");
        let pool = op.pool;
        let mut fence = CpuFence(0);
        let pending = old
            .policy
            .issue_fills(pool, vec![op.ticket], &mut fence, &mut |_| Ok(()))
            .unwrap();
        let done = old
            .policy
            .complete_fills(pool, pending, &mut fence, true)
            .unwrap()
            .pop()
            .expect("one completion");
        fresh.policy.deliver_completion(pool, done).unwrap();
        let summary = fresh.policy.plan_group(0, (0, 0)).unwrap();
        assert_eq!((summary.published, summary.stale), (0, 1));
        assert!(!fresh.policy.needs_retirement());
    }

    // -- the donor protocol: what a long prompt does before its first kernel -----------------------

    fn donor_tail(layout: &Layout) -> DonorTail {
        let start = layout.pools.iter().map(|p| p.block.0).min().unwrap();
        let end = layout
            .pools
            .iter()
            .map(|p| p.block.0 + p.block.1)
            .max()
            .unwrap();
        DonorTail {
            device: (start, end - start),
            mirror: (0x7000_0000, end - start),
        }
    }

    fn restore(layout: &Layout, slot: &mut Option<Rig>) -> Result<Restored<Fake>, RestoreError> {
        restore_pools(slot, layout, config(), 4, 9, donor_tail(layout))
    }

    /// The tickets a mapping names, in expert order.
    fn held(mapping: &[ResidencyLocation]) -> Vec<ResidencySlotTicket> {
        mapping
            .iter()
            .filter_map(|location| match location {
                ResidencyLocation::Slot(ticket) => Some(*ticket),
                ResidencyLocation::Canonical => None,
            })
            .collect()
    }

    #[test]
    fn the_reader_gate_refuses_a_poisoned_unretired_or_stopped_cache() {
        assert!(reader_gate(None, false, true).is_ok());
        assert!(reader_gate(Some("boom"), false, true)
            .unwrap_err()
            .contains("poisoned: boom"));
        // An unretired token (another sequence's decode) refuses every state operation, and so
        // a restore, which starts with this gate.
        assert!(reader_gate(None, true, true)
            .unwrap_err()
            .contains("unretired"));
        assert!(reader_gate(None, false, false)
            .unwrap_err()
            .contains("stopped"));
        // A poisoned cache names its poison even with a token outstanding.
        assert!(reader_gate(Some("boom"), true, false)
            .unwrap_err()
            .contains("poisoned"));
    }

    #[test]
    fn a_mapped_prefill_counts_its_rows_and_touches_nothing_else() {
        let mut stats = ExpertCacheStats::default();
        assert!(note_mapped_prefill(&mut stats, 0).is_err());
        assert_eq!(stats, ExpertCacheStats::default());
        note_mapped_prefill(&mut stats, 511).unwrap();
        note_mapped_prefill(&mut stats, 1).unwrap();
        // No token, use, fill, restore or publication is counted: the prompt neither promotes an
        // expert nor disturbs the cache the decode that follows will read.
        assert_eq!(
            stats,
            ExpertCacheStats {
                mapped_prefill_calls: 2,
                mapped_prefill_rows: 512,
                ..ExpertCacheStats::default()
            }
        );
    }

    #[test]
    fn a_restore_refuses_an_open_window_and_changes_nothing() {
        let layout = layout_of(&cached_plan(5));
        let (mut engine, trace) = seeded(&layout, 8);
        // A token's consumer is registered in an open window: a reader may still be reading a
        // donor slot, so the restore must not drain, rewrite or record anything.
        engine.policy.begin_window().unwrap();
        engine
            .policy
            .register_consumer(0, WindowConsumerPhase::Decode, 0)
            .unwrap();
        let mut slot = Some(engine);
        assert_eq!(restore(&layout, &mut slot).err(), Some(RestoreError::Busy));
        assert!(trace.seq.borrow().is_empty());
        let engine = slot
            .as_mut()
            .expect("a refusal leaves the engine installed");
        for (key, _) in &layout.pools[0].seed {
            assert!(matches!(
                engine.policy.lookup(*key),
                Lookup::Published {
                    fill_generation: 1,
                    ..
                }
            ));
        }
        // Once the reader retires the same call proceeds.
        engine.policy.cancel_window(CancelOutcome::Retired).unwrap();
        assert!(restore(&layout, &mut slot).is_ok());
        assert!(slot.is_none());
    }

    #[test]
    fn a_restore_drains_every_fill_before_it_rewrites_a_hull() {
        let layout = layout_of(&cached_plan(5));
        let (mut engine, trace) = seeded(&layout, 8);
        // Two fills are in flight on the copy stream when the long prompt arrives.
        window(
            &mut engine,
            &layout,
            &[layout.key(0, 16), layout.key(0, 17)],
        );
        retire_all(&mut engine);
        assert_eq!(
            engine
                .io
                .issue_pending(&mut engine.policy, &layout)
                .unwrap(),
            2
        );
        assert_eq!(engine.io.in_flight(), 2);
        let issued_before = engine.policy.stats().fills_issued;
        trace.seq.borrow_mut().clear();
        trace.copies.borrow_mut().clear();
        let mut slot = Some(engine);
        let Ok(restored) = restore(&layout, &mut slot) else {
            panic!("an idle cache restores");
        };
        assert!(slot.is_none(), "the old engine is consumed");
        assert_eq!(restored.retired.fills_issued, issued_before);
        // Both in-flight fills are waited for first; then each pool hull is rewritten with one
        // copy; then each pool's seed batch records one fence behind it, and one wait covers all.
        let pools = layout.pools.len();
        let seq = trace.seq.borrow().clone();
        assert_eq!(seq[..2], ["wait", "wait"]);
        assert_eq!(seq[2..2 + pools], vec!["copy"; pools]);
        assert_eq!(seq[2 + pools..2 + 2 * pools], vec!["record"; pools]);
        assert_eq!(seq[2 + 2 * pools..], ["wait"]);
        let hulls: Vec<(u64, usize)> = layout
            .pools
            .iter()
            .map(|p| (p.block.0 as u64, p.block.1))
            .collect();
        assert_eq!(*trace.copies.borrow(), hulls);
        assert_eq!(restored.seeded.engine.io.in_flight(), 0);
        assert!(restored.seeded.issued.is_ok());
    }

    #[test]
    fn a_reseed_names_only_the_donors_originals_and_outranks_every_earlier_ticket() {
        let layout = layout_of(&cached_plan(5));
        let originals: Vec<usize> = layout.pools[0]
            .seed
            .iter()
            .map(|(key, _)| key.expert_id as usize)
            .collect();
        // Layer 0's table image as the capability would publish it: the seed at restore epoch 0,
        // then two fills replacing two donors' experts (fill generation 2 of their slots).
        let (mut engine, _) = rig(&layout, 8);
        issue_seed(&mut engine).unwrap();
        engine.policy.plan_group(0, (0, 0)).unwrap();
        let change = |engine: &Rig, epoch| {
            let mut changes =
                layer_changes(engine.policy.withdrawn(), engine.policy.published(), epoch).unwrap();
            changes.swap_remove(0)
        };
        let mut mapping = next_mapping(None, &change(&engine, 0));
        retire_all(&mut engine);
        window(
            &mut engine,
            &layout,
            &[layout.key(0, 16), layout.key(0, 17)],
        );
        mapping = next_mapping(Some(mapping), &change(&engine, 0));
        retire_all(&mut engine);
        assert_eq!(
            engine
                .io
                .issue_pending(&mut engine.policy, &layout)
                .unwrap(),
            2
        );
        engine.io.poll(&mut engine.policy, true).unwrap();
        engine.policy.plan_group(0, (0, 0)).unwrap();
        mapping = next_mapping(Some(mapping), &change(&engine, 0));
        retire_all(&mut engine);
        let before = held(&mapping);
        assert_eq!(before.len(), 5);
        assert!(before.iter().any(|t| t.expert == 16) && before.iter().any(|t| t.expert == 17));
        let newest = before.iter().map(|t| t.generation).max().unwrap();
        assert!(newest >= 2, "the refilled slots are past their first fill");

        let mut slot = Some(engine);
        let Ok(restored) = restore(&layout, &mut slot) else {
            panic!("an idle cache restores");
        };
        let Seeded {
            mut engine, issued, ..
        } = restored.seeded;
        issued.unwrap();
        engine.policy.plan_group(0, (0, 0)).unwrap();
        let reseed = change(&engine, 1);
        let after = next_mapping(None, &reseed);
        // Only the donors' own experts name a slot: the two experts the cache had filled are
        // canonical again, so no table entry points at a donor slot holding another expert.
        let mut named: Vec<usize> = held(&after).iter().map(|t| t.expert).collect();
        named.sort_unstable();
        let mut expected = originals.clone();
        expected.sort_unstable();
        assert_eq!(named, expected);
        for expert in [16, 17] {
            assert_eq!(after[expert], ResidencyLocation::Canonical);
        }
        // The rebuilt bank restarts its fill generations at 1, so a raw generation would go
        // backwards; the ticket carries the restore epoch and outranks every earlier one.
        assert!(engine
            .policy
            .published()
            .iter()
            .all(|p| p.fill_generation == 1));
        for ticket in held(&after) {
            let earlier = before.iter().find(|t| t.slot == ticket.slot).unwrap();
            assert!(
                ticket.generation > earlier.generation,
                "slot {} went backwards",
                ticket.slot
            );
            assert_eq!(ticket.generation, ticket_generation(1, 1).unwrap());
        }
    }

    #[test]
    fn the_prompt_view_inherits_the_resident_verdicts_or_is_refused() {
        let resident = PrefillVerdict {
            host_mapped: true,
            symmetric: true,
        };
        assert_eq!(
            inherit_prefill_verdict(3, resident, true).unwrap(),
            resident
        );
        let plain = PrefillVerdict {
            host_mapped: false,
            symmetric: false,
        };
        assert_eq!(inherit_prefill_verdict(3, plain, false).unwrap(), plain);
        // A layer that donates without a host expert would route its prompt differently.
        for (resident_bit, immutable_bit) in [(true, false), (false, true)] {
            let resident = PrefillVerdict {
                host_mapped: resident_bit,
                symmetric: true,
            };
            let error = inherit_prefill_verdict(3, resident, immutable_bit).unwrap_err();
            assert!(
                error.contains("layer 3") && error.contains("host_mapped"),
                "{error}"
            );
        }
    }

    #[test]
    fn the_mutable_runtime_takes_the_resident_route_on_prompt_tiles_and_windows() {
        use crate::gpu_forward::declared_route_capability as capability;
        use hipfire_dispatch::families::moe::MoeRouteCapability::{
            Qt44Qt53Grouped, Qt44Qt53GroupedSymmetric,
        };
        let floor = rdna_compute::gemm::QWEN4_MOE_SYM_IU4_MIN_ROWS;
        for resident in [
            PrefillVerdict {
                host_mapped: true,
                symmetric: true,
            },
            PrefillVerdict {
                host_mapped: true,
                symmetric: false,
            },
            PrefillVerdict {
                host_mapped: false,
                symmetric: true,
            },
        ] {
            let mutable = inherit_prefill_verdict(5, resident, resident.host_mapped).unwrap();
            assert_eq!(mutable, resident);
            // A canonical-chunk prompt tile and every window width declare the Resident route.
            for rows in (1..=64).chain([floor, 2047, 16384]) {
                assert_eq!(
                    capability(false, false, mutable.symmetric, rows, floor),
                    capability(false, false, resident.symmetric, rows, floor),
                    "{resident:?} {rows}"
                );
            }
        }
        // Windows never reach the IU4 floor, so the verdict is inert there; a prompt tile is not.
        for rows in 1..=64 {
            assert_eq!(capability(false, false, true, rows, floor), Qt44Qt53Grouped);
        }
        assert_eq!(
            capability(false, false, true, 16384, floor),
            Qt44Qt53GroupedSymmetric
        );
    }
}
