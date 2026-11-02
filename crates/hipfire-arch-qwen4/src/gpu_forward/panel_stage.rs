// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S1 slow-link layer-major prefill: expert panel transport.
//!
//! One prompt window sweeps every host layer's expert payload over the link
//! exactly once. This module owns the sweep's VRAM side: a bounded pool of
//! equally sized pages (two active 32-expert slots plus one next-layer
//! lookahead store cut into 32-expert pages), the copy stream that fills them,
//! and the ready/free events that order the copies against the compute
//! stream's readers. Nothing here replays the incumbent stage: no donor layer
//! is borrowed, no weight is transformed, and no hot resident weight is ever a
//! copy destination.
//!
//! A page's life is `Free -> Copying -> Ready -> InUse -> Retiring -> Free`.
//!
//! * [`ExpertPanelRing::enqueue`] picks a free page, makes the copy stream wait
//!   on the page's free event (recorded after its previous last reader), copies
//!   the group's expert extents (runs of adjacent host bytes coalesced), zeroes
//!   the tail pad the kernels' over-read can reach, uploads the two rebased
//!   per-expert pointer tables and records `ready`. `ready` therefore covers
//!   payload, tables and pad.
//! * [`ExpertPanelRing::acquire`] makes the compute stream wait on `ready` and
//!   lends D's dispatch the page as [`PanelBindingParts`].
//! * [`ExpertPanelRing::retire`] records the free event on the compute stream
//!   after the group's last reader was enqueued. Only then may the page be
//!   refilled; the CPU may enqueue the refill long before the free event
//!   completes because the copy stream waits on it.
//!
//! A ticket `(slot, epoch, layer, group, codec)` is checked on every acquire
//! and retire, so a stale ticket, a forged layer/codec or a double retire can
//! never touch a page that now holds other bytes at the same address.
//! Pinned host sources and the pinned table staging stay owned until the page's
//! previous copy is known complete (the next enqueue of that page waits for it
//! on the host) or until [`ExpertPanelRing::drain`]. A failed operation
//! quarantines the ring: every call but `drain` is refused and no owner is
//! released until the streams are drained.
//!
//! All state transitions live in [`RingCore`], generic over a [`PanelBackend`],
//! so the lifecycle is unit tested without HIP; [`HipPanelBackend`] is the real
//! one.

use super::*;
use crate::expert_arena::ARENA_TAIL_PAD_BYTES;
use std::any::Any;
use std::collections::HashMap;

type R<T> = Result<T, Qwen4GpuForwardError>;

/// Alignment the page payload offsets preserve of every source address.
const EXTENT_ALIGN: usize = 256;
/// Largest group a page holds (D's `PanelLimits::experts_per_slot`).
pub(crate) const MAX_GROUP_EXPERTS: usize = 32;
/// Largest per-layer expert table.
pub(crate) const MAX_TABLE_EXPERTS: usize = 1024;

fn align_up(value: usize, align: usize) -> Option<usize> {
    value.checked_add(align - 1).map(|v| v & !(align - 1))
}

// ── sources ──────────────────────────────────────────────────────────

/// Codec identity of a group's experts; one group never mixes codecs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PanelCodec(pub(crate) u32);

/// One projection's bytes in pinned host memory (CPU-readable address).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExpertExtent {
    pub(crate) addr: usize,
    pub(crate) len: usize,
}

/// One expert's host bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExpertSource {
    pub(crate) codec: PanelCodec,
    pub(crate) gate_up: ExpertExtent,
    pub(crate) down: ExpertExtent,
}

/// One expert as the placement gives it: device-visible alias addresses of the
/// host-mapped arena (the values of the layer's pointer entries).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AliasExpert {
    pub(crate) codec: PanelCodec,
    pub(crate) gate_up_addr: usize,
    pub(crate) gate_up_len: usize,
    pub(crate) down_addr: usize,
    pub(crate) down_len: usize,
}

/// A host layer's immutable expert sources. Indexed by the layer's local
/// expert id; `None` is an expert the layer does not hold (pruned).
pub(crate) struct LayerExpertSources {
    layer: usize,
    /// Host ranges `(base, bytes)`: one (gate/up and down share it) or two
    /// (gate/up, then down: separate pinned allocations).
    ranges: SmallVec<[(usize, usize); 2]>,
    experts: Vec<Option<ExpertSource>>,
    owner: Arc<dyn Any + Send + Sync>,
}

impl LayerExpertSources {
    /// One host range holding every extent.
    ///
    /// # Safety
    /// `[host_base, host_base + host_bytes)` must be readable pinned host
    /// memory that stays valid and unmodified for as long as `owner` (or a
    /// clone of it) lives. Every extent is checked to lie inside that range.
    pub(crate) unsafe fn new(
        layer: usize,
        host_base: usize,
        host_bytes: usize,
        experts: Vec<Option<ExpertSource>>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> R<Self> {
        let mut ranges = SmallVec::new();
        ranges.push((host_base, host_bytes));
        // SAFETY: forwarded from this function's contract.
        unsafe { Self::build(layer, ranges, experts, owner) }
    }

    /// Gate/up extents in `gate_up` and down extents in `down`: two separate
    /// pinned allocations. Extents of different ranges are never merged into
    /// one copy, even when the ranges happen to be adjacent.
    ///
    /// # Safety
    /// As [`Self::new`], for both `(base, bytes)` ranges.
    pub(crate) unsafe fn new_split(
        layer: usize,
        gate_up: (usize, usize),
        down: (usize, usize),
        experts: Vec<Option<ExpertSource>>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> R<Self> {
        let mut ranges = SmallVec::new();
        ranges.push(gate_up);
        ranges.push(down);
        // SAFETY: forwarded from this function's contract.
        unsafe { Self::build(layer, ranges, experts, owner) }
    }

    /// `ranges[0]` bounds gate/up extents; `ranges[last]` bounds down extents.
    unsafe fn build(
        layer: usize,
        ranges: SmallVec<[(usize, usize); 2]>,
        experts: Vec<Option<ExpertSource>>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> R<Self> {
        if experts.is_empty() || experts.len() > MAX_TABLE_EXPERTS {
            return Err(invalid(format!(
                "panel sources layer {layer}: {} experts is outside 1..={MAX_TABLE_EXPERTS}",
                experts.len()
            )));
        }
        let mut ends = SmallVec::<[usize; 2]>::new();
        for &(base, bytes) in &ranges {
            ends.push(base.checked_add(bytes).ok_or_else(|| {
                invalid(format!("panel sources layer {layer}: host range overflows"))
            })?);
        }
        if ranges.len() == 2 {
            let (a, b) = (ranges[0], ranges[1]);
            if a.0 < ends[1] && b.0 < ends[0] {
                return Err(invalid(format!(
                    "panel sources layer {layer}: gate/up and down host ranges overlap"
                )));
            }
        }
        for (id, expert) in experts.iter().enumerate() {
            let Some(expert) = expert else { continue };
            for (name, extent, range) in [
                ("gate_up", expert.gate_up, 0),
                ("down", expert.down, ranges.len() - 1),
            ] {
                let (host_base, host_bytes) = ranges[range];
                let end = extent.addr.checked_add(extent.len).ok_or_else(|| {
                    invalid(format!(
                        "panel sources layer {layer} expert {id} {name} extent overflows"
                    ))
                })?;
                if extent.len == 0 || extent.addr < host_base || end > ends[range] {
                    return Err(invalid(format!(
                        "panel sources layer {layer} expert {id} {name} extent [{:#x}, +{}) is outside the host owner [{host_base:#x}, +{host_bytes})",
                        extent.addr, extent.len
                    )));
                }
            }
        }
        Ok(Self {
            layer,
            ranges,
            experts,
            owner,
        })
    }

    /// Sources from the placement's device-visible alias addresses: each alias
    /// address must lie in `[alias_base, alias_base + host_bytes)` and maps to
    /// `cpu_base + (alias - alias_base)`.
    ///
    /// # Safety
    /// As [`Self::new`], for the CPU range `[cpu_base, cpu_base + host_bytes)`.
    pub(crate) unsafe fn from_alias(
        layer: usize,
        cpu_base: usize,
        alias_base: usize,
        host_bytes: usize,
        experts: &[Option<AliasExpert>],
        owner: Arc<dyn Any + Send + Sync>,
    ) -> R<Self> {
        let range = (cpu_base, alias_base, host_bytes);
        let out = translate_alias(layer, range, range, experts)?;
        // SAFETY: forwarded from this function's contract.
        unsafe { Self::new(layer, cpu_base, host_bytes, out, owner) }
    }

    /// As [`Self::from_alias`] for an aggregate layer whose gate/up and down
    /// payloads are two pinned allocations: `(cpu_base, alias_base, bytes)`
    /// each. Every projection is translated and bounds-checked against its own
    /// range only.
    ///
    /// # Safety
    /// As [`Self::new_split`], for both CPU ranges.
    pub(crate) unsafe fn from_alias_split(
        layer: usize,
        gate_up: (usize, usize, usize),
        down: (usize, usize, usize),
        experts: &[Option<AliasExpert>],
        owner: Arc<dyn Any + Send + Sync>,
    ) -> R<Self> {
        let out = translate_alias(layer, gate_up, down, experts)?;
        // SAFETY: forwarded from this function's contract.
        unsafe { Self::new_split(layer, (gate_up.0, gate_up.2), (down.0, down.2), out, owner) }
    }

    pub(crate) fn layer(&self) -> usize {
        self.layer
    }

    pub(crate) fn num_experts(&self) -> usize {
        self.experts.len()
    }

    pub(crate) fn expert(&self, id: usize) -> Option<&ExpertSource> {
        self.experts.get(id).and_then(Option::as_ref)
    }

    /// Provenance: the gate/up (or only) host range `(base, bytes)`.
    pub(crate) fn host_range(&self) -> (usize, usize) {
        self.ranges[0]
    }

    /// Every host range: one, or gate/up then down.
    pub(crate) fn host_ranges(&self) -> &[(usize, usize)] {
        &self.ranges
    }

    /// Range index of a projection's extents (0 gate/up; down is 1 when split).
    fn range_of(&self, down: bool) -> usize {
        if down {
            self.ranges.len() - 1
        } else {
            0
        }
    }

    /// The expert's own payload bytes (gate/up plus down), for receipt
    /// "touched" counts; `None` for an expert the layer does not hold.
    pub(crate) fn expert_extent_bytes(&self, id: usize) -> Option<usize> {
        self.expert(id)
            .and_then(|e| e.gate_up.len.checked_add(e.down.len))
    }

    fn owner(&self) -> &Arc<dyn Any + Send + Sync> {
        &self.owner
    }
}

/// Translate alias addresses to CPU addresses; each projection against its own
/// `(cpu_base, alias_base, bytes)` range.
fn translate_alias(
    layer: usize,
    gate_up: (usize, usize, usize),
    down: (usize, usize, usize),
    experts: &[Option<AliasExpert>],
) -> R<Vec<Option<ExpertSource>>> {
    let translate = |id: usize, addr: usize, (cpu, alias, bytes): (usize, usize, usize)| {
        addr.checked_sub(alias)
            .filter(|offset| *offset < bytes)
            .and_then(|offset| cpu.checked_add(offset))
            .ok_or_else(|| {
                invalid(format!(
                    "panel sources layer {layer} expert {id}: alias {addr:#x} is outside the host owner [{alias:#x}, +{bytes})"
                ))
            })
    };
    let mut out = Vec::with_capacity(experts.len());
    for (id, expert) in experts.iter().enumerate() {
        out.push(match expert {
            None => None,
            Some(e) => Some(ExpertSource {
                codec: e.codec,
                gate_up: ExpertExtent {
                    addr: translate(id, e.gate_up_addr, gate_up)?,
                    len: e.gate_up_len,
                },
                down: ExpertExtent {
                    addr: translate(id, e.down_addr, down)?,
                    len: e.down_len,
                },
            }),
        });
    }
    Ok(out)
}

// ── groups and layout ────────────────────────────────────────────────

/// One transfer unit: up to `experts_per_slot` ascending experts of one codec.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExpertGroupPlan {
    pub(crate) layer: usize,
    pub(crate) group: u32,
    pub(crate) codec: PanelCodec,
    pub(crate) experts: SmallVec<[usize; MAX_GROUP_EXPERTS]>,
}

impl ExpertGroupPlan {
    /// The immutable full-layer schedule: every present expert in ascending
    /// order, a new group at each codec change or every `experts_per_slot`.
    pub(crate) fn partition(sources: &LayerExpertSources, experts_per_slot: usize) -> R<Vec<Self>> {
        if experts_per_slot == 0 || experts_per_slot > MAX_GROUP_EXPERTS {
            return Err(invalid(format!(
                "panel group size {experts_per_slot} is outside 1..={MAX_GROUP_EXPERTS}"
            )));
        }
        let mut out: Vec<Self> = Vec::new();
        for id in 0..sources.num_experts() {
            let Some(expert) = sources.expert(id) else {
                continue;
            };
            match out.last_mut() {
                Some(last)
                    if last.codec == expert.codec && last.experts.len() < experts_per_slot =>
                {
                    last.experts.push(id)
                }
                _ => {
                    let group = u32::try_from(out.len())
                        .map_err(|_| invalid("panel group index overflows u32"))?;
                    let mut experts = SmallVec::new();
                    experts.push(id);
                    out.push(Self {
                        layer: sources.layer(),
                        group,
                        codec: expert.codec,
                        experts,
                    });
                }
            }
        }
        Ok(out)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CopyRun {
    src: usize,
    dst: usize,
    len: usize,
}

/// Where a group's bytes sit in a page.
#[derive(Debug, PartialEq, Eq)]
struct GroupLayout {
    runs: SmallVec<[CopyRun; 8]>,
    /// Page offset of each group expert's gate/up and down (parallel to the
    /// group's `experts`).
    gate_up: SmallVec<[usize; MAX_GROUP_EXPERTS]>,
    down: SmallVec<[usize; MAX_GROUP_EXPERTS]>,
    /// End of the payload; the zeroed tail pad starts here.
    used: usize,
    /// Sum of the experts' extent lengths.
    extent_bytes: usize,
}

impl GroupLayout {
    fn submitted_bytes(&self) -> usize {
        self.runs.iter().map(|run| run.len).sum()
    }
}

/// Lay out `experts` (ascending ids of `sources`). Extents are sorted by host
/// range then address and merged where they touch or overlap inside one range
/// (never across the gate/up and down allocations of a split layer); runs are
/// packed back to back so that each run's page offset is congruent to its
/// source address modulo [`EXTENT_ALIGN`], which preserves whatever alignment
/// the source extents had.
fn plan_group_layout(experts: &[usize], sources: &LayerExpertSources) -> R<GroupLayout> {
    // (host range, addr, len, index in `experts`, is_down)
    let mut items: SmallVec<[(usize, usize, usize, usize, bool); 2 * MAX_GROUP_EXPERTS]> =
        SmallVec::new();
    for (index, &id) in experts.iter().enumerate() {
        let expert = sources.expert(id).ok_or_else(|| {
            invalid(format!(
                "panel layout layer {}: expert {id} is not held",
                sources.layer()
            ))
        })?;
        items.push((
            sources.range_of(false),
            expert.gate_up.addr,
            expert.gate_up.len,
            index,
            false,
        ));
        items.push((
            sources.range_of(true),
            expert.down.addr,
            expert.down.len,
            index,
            true,
        ));
    }
    items.sort_unstable();
    let overflow = || invalid("panel layout overflows");
    // (host range, source start, source end, page offset)
    let mut runs: SmallVec<[(usize, usize, usize, usize); 8]> = SmallVec::new();
    let mut item_run: SmallVec<[usize; 2 * MAX_GROUP_EXPERTS]> = SmallVec::new();
    let mut cursor = 0usize;
    let mut extent_bytes = 0usize;
    for &(range, addr, len, _, _) in &items {
        let end = addr.checked_add(len).ok_or_else(overflow)?;
        extent_bytes = extent_bytes.checked_add(len).ok_or_else(overflow)?;
        match runs.last_mut() {
            Some(run) if run.0 == range && addr <= run.2 => run.2 = run.2.max(end),
            _ => {
                let dst = align_up(cursor, EXTENT_ALIGN)
                    .and_then(|aligned| aligned.checked_add(addr % EXTENT_ALIGN))
                    .ok_or_else(overflow)?;
                runs.push((range, addr, end, dst));
            }
        }
        item_run.push(runs.len() - 1);
        let run = runs.last().expect("run exists");
        cursor = run.3.checked_add(run.2 - run.1).ok_or_else(overflow)?;
    }
    let mut layout = GroupLayout {
        runs: runs
            .iter()
            .map(|&(_, start, end, dst)| CopyRun {
                src: start,
                dst,
                len: end - start,
            })
            .collect(),
        gate_up: SmallVec::from_elem(0, experts.len()),
        down: SmallVec::from_elem(0, experts.len()),
        used: cursor,
        extent_bytes,
    };
    for (&(_, addr, _, index, down), &run) in items.iter().zip(&item_run) {
        let (_, start, _, dst) = runs[run];
        let offset = dst + (addr - start);
        if down {
            layout.down[index] = offset;
        } else {
            layout.gate_up[index] = offset;
        }
    }
    Ok(layout)
}

/// The two little-endian per-expert pointer tables of a page: every group
/// expert rebased to `page_base + offset`, every other entry zero (the panel
/// never reads it).
fn fill_tables(
    layout: &GroupLayout,
    experts: &[usize],
    page_base: usize,
    gate_up: &mut [u64],
    down: &mut [u64],
) -> R<()> {
    gate_up.fill(0);
    down.fill(0);
    for (index, &id) in experts.iter().enumerate() {
        if id >= gate_up.len() || id >= down.len() {
            return Err(invalid(format!(
                "panel table: expert {id} is outside the {}-entry table",
                gate_up.len()
            )));
        }
        for (table, offset) in [
            (&mut *gate_up, layout.gate_up[index]),
            (&mut *down, layout.down[index]),
        ] {
            let address = page_base.checked_add(offset).ok_or_else(|| {
                invalid(format!("panel table: expert {id} rebased entry overflows"))
            })?;
            table[id] = u64::try_from(address)
                .map_err(|_| invalid(format!("panel table: expert {id} address exceeds u64")))?;
        }
    }
    Ok(())
}

/// Largest page payload any group of `layers` needs (the pool's
/// `payload_bytes`).
pub(crate) fn required_payload_bytes(
    layers: &[&LayerExpertSources],
    experts_per_slot: usize,
) -> R<usize> {
    let mut largest = 0usize;
    for sources in layers {
        for group in ExpertGroupPlan::partition(sources, experts_per_slot)? {
            largest = largest.max(plan_group_layout(&group.experts, sources)?.used);
        }
    }
    if largest == 0 {
        return Err(invalid("panel pool: no host experts to size a page"));
    }
    Ok(largest)
}

// ── pool geometry ────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PanelPoolSpec {
    /// Pages reserved for current-layer groups (two in S1).
    pub(crate) active_slots: usize,
    /// Additional pages of the one-layer lookahead store (at most 16 for MQ4).
    pub(crate) lookahead_pages: usize,
    pub(crate) experts_per_slot: usize,
    /// Entries of each per-layer pointer table.
    pub(crate) num_experts: usize,
    /// Largest group payload ([`required_payload_bytes`]); the tail pad is
    /// added by the page.
    pub(crate) payload_bytes: usize,
    /// Copy/compute interval records the receipt timeline keeps (0 = off).
    pub(crate) timeline_records: usize,
    /// Whole-layer compute marks the receipt timeline keeps.
    pub(crate) timeline_layers: usize,
}

/// Charged bytes of a pool (VRAM `payload` + `tables`, pinned host `staging`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PanelPoolReserve {
    pub(crate) pages: usize,
    pub(crate) page_bytes: usize,
    pub(crate) payload: usize,
    pub(crate) tables: usize,
    pub(crate) staging: usize,
    /// `payload + tables` (device).
    pub(crate) device_total: usize,
}

impl PanelPoolSpec {
    pub(crate) fn pages(&self) -> usize {
        self.active_slots + self.lookahead_pages
    }

    /// One page: payload, then the zeroed tail pad, rounded to 256.
    pub(crate) fn page_bytes(&self) -> Option<usize> {
        self.payload_bytes
            .checked_add(ARENA_TAIL_PAD_BYTES)
            .and_then(|bytes| align_up(bytes, EXTENT_ALIGN))
    }

    pub(crate) fn validate(&self) -> R<()> {
        if self.active_slots == 0 {
            return Err(invalid("panel pool needs at least one active slot"));
        }
        if self.experts_per_slot == 0 || self.experts_per_slot > MAX_GROUP_EXPERTS {
            return Err(invalid(format!(
                "panel pool group size {} is outside 1..={MAX_GROUP_EXPERTS}",
                self.experts_per_slot
            )));
        }
        if self.num_experts == 0 || self.num_experts > MAX_TABLE_EXPERTS {
            return Err(invalid(format!(
                "panel pool table of {} entries is outside 1..={MAX_TABLE_EXPERTS}",
                self.num_experts
            )));
        }
        if self.payload_bytes == 0 {
            return Err(invalid("panel pool payload is empty"));
        }
        self.reserve()
            .map(|_| ())
            .ok_or_else(|| invalid("panel pool byte size overflows"))
    }

    /// The bytes this pool charges, `None` on overflow.
    pub(crate) fn reserve(&self) -> Option<PanelPoolReserve> {
        let pages = self.pages();
        let page_bytes = self.page_bytes()?;
        let payload = page_bytes.checked_mul(pages)?;
        let table = self.num_experts.checked_mul(std::mem::size_of::<u64>())?;
        let tables = table.checked_mul(2)?.checked_mul(pages)?;
        Some(PanelPoolReserve {
            pages,
            page_bytes,
            payload,
            tables,
            staging: tables,
            device_total: payload.checked_add(tables)?,
        })
    }
}

/// Worst-case page bytes lost to run packing: each extent may start its own run
/// (64 runs for a 32-expert group), and each run loses at most
/// `(EXTENT_ALIGN - 1)` to rounding the cursor plus `(EXTENT_ALIGN - 1)` to
/// keeping the source's alignment phase.
pub(crate) const UNIFORM_RUN_SLACK: usize = 2 * MAX_GROUP_EXPERTS * 2 * (EXTENT_ALIGN - 1);

impl PanelPoolSpec {
    /// Spec for a layer set whose experts all have `gate_up_expert_bytes` of
    /// gate/up and `down_expert_bytes` of down payload. The page holds any
    /// group of `experts_per_slot` experts however the host extents are
    /// placed: `experts_per_slot * (gate_up + down)` plus [`UNIFORM_RUN_SLACK`].
    /// This is the single authority for the charged reserve: the census and
    /// the allocation both read [`Self::reserve`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn uniform(
        active_slots: usize,
        lookahead_pages: usize,
        experts_per_slot: usize,
        num_experts: usize,
        gate_up_expert_bytes: usize,
        down_expert_bytes: usize,
        timeline_records: usize,
        timeline_layers: usize,
    ) -> R<Self> {
        if gate_up_expert_bytes == 0 || down_expert_bytes == 0 {
            return Err(invalid("panel pool experts have an empty projection"));
        }
        let payload_bytes = gate_up_expert_bytes
            .checked_add(down_expert_bytes)
            .and_then(|per| per.checked_mul(experts_per_slot))
            .and_then(|bytes| bytes.checked_add(UNIFORM_RUN_SLACK))
            .ok_or_else(|| invalid("panel pool payload bytes overflow"))?;
        let spec = Self {
            active_slots,
            lookahead_pages,
            experts_per_slot,
            num_experts,
            payload_bytes,
            timeline_records,
            timeline_layers,
        };
        spec.validate()?;
        Ok(spec)
    }
}

// ── tickets, statistics, timeline ────────────────────────────────────

/// Names one filled page content: `(slot, epoch, layer, group, codec)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PanelTicket {
    pub(crate) slot: usize,
    /// Ring-unique and never reused: a page refilled at the same address has a
    /// new epoch.
    pub(crate) epoch: u64,
    pub(crate) layer: usize,
    pub(crate) group: u32,
    pub(crate) codec: PanelCodec,
}

/// Cumulative since the last [`RingCore::begin_window`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PanelStats {
    pub(crate) enqueued: u64,
    pub(crate) acquired: u64,
    pub(crate) retired: u64,
    /// Retired without ever being acquired (prefetched, then unused).
    pub(crate) discarded: u64,
    /// Bytes H2D-copied as payload (submitted).
    pub(crate) submitted_payload_bytes: u64,
    /// Sum of the experts' own extent lengths of those groups.
    pub(crate) extent_bytes: u64,
    pub(crate) table_bytes: u64,
    /// Payload bytes of groups the compute stream acquired.
    pub(crate) acquired_payload_bytes: u64,
    /// Payload bytes of groups retired unread.
    pub(crate) discarded_payload_bytes: u64,
    /// Enqueues refused for lack of a free page of the needed class.
    pub(crate) page_full_stalls: u64,
    /// Acquires whose copy had not completed when the compute stream got there.
    pub(crate) copy_starved_stalls: u64,
    pub(crate) timeline_overflow: u64,
    pub(crate) layer_mark_overflow: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum TimelineEvent {
    Base,
    CopyStart(usize),
    CopyEnd(usize),
    ComputeBegin(usize),
    ComputeEnd(usize),
    LayerBegin(usize),
    LayerEnd(usize),
}

/// One group's intervals in ms since the window base event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PanelSpanMs {
    pub(crate) layer: usize,
    pub(crate) group: u32,
    pub(crate) bytes: usize,
    pub(crate) copy_start: f64,
    pub(crate) copy_end: f64,
    /// Acquire..retire on the compute stream; `None` if never acquired.
    pub(crate) compute: Option<(f64, f64)>,
}

/// A whole layer's compute interval (`mark_layer_begin`..`mark_layer_end`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LayerSpanMs {
    pub(crate) layer: usize,
    pub(crate) begin: f64,
    pub(crate) end: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct OverlapSummary {
    /// Count of (layer L compute, next layer's H2D) interval pairs that intersect.
    pub(crate) cross_layer_intersections: u64,
    pub(crate) cross_layer_overlap_ms: f64,
    /// Gaps between consecutive copies (waiting on a free event or on work).
    pub(crate) copy_idle_ms: f64,
    pub(crate) copy_busy_ms: f64,
    pub(crate) compute_busy_ms: f64,
    pub(crate) page_full_stalls: u64,
    pub(crate) copy_starved_stalls: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct PanelTimeline {
    pub(crate) spans: Vec<PanelSpanMs>,
    pub(crate) layers: Vec<LayerSpanMs>,
    pub(crate) overlap: OverlapSummary,
    pub(crate) stats: PanelStats,
}

fn union_ms(mut intervals: Vec<(f64, f64)>) -> f64 {
    intervals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut total = 0.0;
    let mut open: Option<(f64, f64)> = None;
    for (start, end) in intervals {
        match open.as_mut() {
            Some(cur) if start <= cur.1 => cur.1 = cur.1.max(end),
            _ => {
                if let Some(cur) = open.take() {
                    total += cur.1 - cur.0;
                }
                open = Some((start, end));
            }
        }
    }
    if let Some(cur) = open {
        total += cur.1 - cur.0;
    }
    total
}

/// Receipt numbers from measured intervals only. Layer compute intervals come
/// from the whole-layer marks when the layer has one, else from that layer's
/// per-group acquire..retire spans; "next layer" is the next distinct layer in
/// ascending order that has copies.
pub(crate) fn summarize_overlap(
    spans: &[PanelSpanMs],
    layers: &[LayerSpanMs],
    stats: &PanelStats,
) -> OverlapSummary {
    let mut summary = OverlapSummary {
        page_full_stalls: stats.page_full_stalls,
        copy_starved_stalls: stats.copy_starved_stalls,
        ..OverlapSummary::default()
    };
    let mut by_copy: Vec<&PanelSpanMs> = spans.iter().collect();
    by_copy.sort_by(|a, b| {
        a.copy_start
            .partial_cmp(&b.copy_start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for pair in by_copy.windows(2) {
        summary.copy_idle_ms += (pair[1].copy_start - pair[0].copy_end).max(0.0);
    }
    summary.copy_busy_ms = union_ms(spans.iter().map(|s| (s.copy_start, s.copy_end)).collect());
    summary.compute_busy_ms = union_ms(spans.iter().filter_map(|s| s.compute).collect());
    let mut order: Vec<usize> = spans
        .iter()
        .map(|s| s.layer)
        .chain(layers.iter().map(|l| l.layer))
        .collect();
    order.sort_unstable();
    order.dedup();
    for window in order.windows(2) {
        let (current, next) = (window[0], window[1]);
        let copies: Vec<(f64, f64)> = spans
            .iter()
            .filter(|s| s.layer == next)
            .map(|s| (s.copy_start, s.copy_end))
            .collect();
        if copies.is_empty() {
            continue;
        }
        let mut compute: Vec<(f64, f64)> = layers
            .iter()
            .filter(|l| l.layer == current)
            .map(|l| (l.begin, l.end))
            .collect();
        if compute.is_empty() {
            compute = spans
                .iter()
                .filter(|s| s.layer == current)
                .filter_map(|s| s.compute)
                .collect();
        }
        for &(c0, c1) in &compute {
            for &(h0, h1) in &copies {
                let overlap = c1.min(h1) - c0.max(h0);
                if overlap > 0.0 {
                    summary.cross_layer_intersections += 1;
                    summary.cross_layer_overlap_ms += overlap;
                }
            }
        }
    }
    summary
}

// ── backend ──────────────────────────────────────────────────────────

/// Device operations of the ring. `Ctx` is `Gpu` for HIP and `()` for the CPU
/// model.
pub(crate) trait PanelBackend {
    type Ctx: ?Sized;

    /// Device address of the page payload (what the rebased tables point into).
    fn page_base(&self, slot: usize) -> usize;
    /// Host staging of the page's two pointer tables, `num_experts` entries each.
    /// Only written after [`Self::await_copy_done`] of the slot's previous fill.
    fn tables_mut(&mut self, slot: usize) -> (&mut [u64], &mut [u64]);
    /// Host-wait until the slot's last recorded `ready` completed: its payload
    /// and table copies no longer read host memory.
    fn await_copy_done(&mut self, ctx: &Self::Ctx, slot: usize) -> R<()>;
    /// Copy stream: wait the slot's free event (if one was ever recorded).
    fn copy_begin(
        &mut self,
        ctx: &Self::Ctx,
        slot: usize,
        wait_free: bool,
        rec: Option<usize>,
    ) -> R<()>;
    fn copy_payload(
        &mut self,
        ctx: &Self::Ctx,
        slot: usize,
        src: usize,
        dst: usize,
        len: usize,
    ) -> R<()>;
    fn copy_zero(&mut self, ctx: &Self::Ctx, slot: usize, dst: usize, len: usize) -> R<()>;
    fn copy_tables(&mut self, ctx: &Self::Ctx, slot: usize) -> R<()>;
    /// Copy stream: record `ready` after payload, pad and tables.
    fn copy_end(&mut self, ctx: &Self::Ctx, slot: usize, rec: Option<usize>) -> R<()>;
    /// Compute stream waits `ready`; returns whether the copy was still running.
    fn compute_acquire(&mut self, ctx: &Self::Ctx, slot: usize, rec: Option<usize>) -> R<bool>;
    /// Compute stream records the slot's free event (after the last reader).
    fn compute_release(&mut self, ctx: &Self::Ctx, slot: usize, rec: Option<usize>) -> R<()>;
    fn mark_base(&mut self, ctx: &Self::Ctx) -> R<()>;
    fn mark_layer(&mut self, ctx: &Self::Ctx, mark: usize, end: bool) -> R<()>;
    /// Synchronize the copy and compute streams.
    fn drain(&mut self, ctx: &Self::Ctx) -> R<()>;
    fn elapsed_ms(&self, ctx: &Self::Ctx, from: TimelineEvent, to: TimelineEvent) -> R<f64>;
}

// ── core state machine ───────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PageState {
    Free,
    Copying,
    Ready,
    InUse,
    Retiring,
}

struct Page {
    state: PageState,
    epoch: u64,
    layer: usize,
    group: u32,
    codec: PanelCodec,
    experts: SmallVec<[usize; MAX_GROUP_EXPERTS]>,
    host_base: usize,
    payload_bytes: usize,
    used: usize,
    acquired: bool,
    /// A fill was ever issued since the last drain: its `ready` exists.
    filled: bool,
    /// A free event was recorded since the last drain.
    free_recorded: bool,
    rec: Option<usize>,
    /// Pinned host source owner of the current/last fill.
    retained: Option<Arc<dyn Any + Send + Sync>>,
}

impl Page {
    fn new() -> Self {
        Self {
            state: PageState::Free,
            epoch: 0,
            layer: 0,
            group: 0,
            codec: PanelCodec(0),
            experts: SmallVec::new(),
            host_base: 0,
            payload_bytes: 0,
            used: 0,
            acquired: false,
            filled: false,
            free_recorded: false,
            rec: None,
            retained: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Current,
    Lookahead,
}

struct SpanRec {
    layer: usize,
    group: u32,
    bytes: usize,
    acquired: bool,
    released: bool,
}

struct LayerMark {
    layer: usize,
    ended: bool,
}

pub(crate) struct RingCore<B: PanelBackend> {
    spec: PanelPoolSpec,
    pages: Vec<Page>,
    /// Free pages in the order they were freed (FIFO): the page free the
    /// longest is refilled first, so its free event is the likeliest complete.
    free: Vec<usize>,
    next_epoch: u64,
    current: Option<usize>,
    next: Option<usize>,
    /// Last group enqueued per window layer.
    cursors: SmallVec<[(usize, u32); 2]>,
    poisoned: bool,
    stats: PanelStats,
    records: Vec<SpanRec>,
    marks: Vec<LayerMark>,
    /// Never dropped implicitly while device work may be queued: see the
    /// `Drop` impl.
    backend: std::mem::ManuallyDrop<B>,
    /// Any device operation was issued since the last SUCCESSFUL drain. A
    /// retired/free logical state is not GPU completion, so only a completed
    /// [`Self::drain`] clears this.
    dirty: bool,
    backend_taken: bool,
}

impl<B: PanelBackend> Drop for RingCore<B> {
    fn drop(&mut self) {
        if self.backend_taken {
            return;
        }
        if self.dirty {
            // Device work may still read the pool, tables and pinned sources
            // and the owner may not hold a `Gpu` to drain with: leak the
            // backend (never dropped: `ManuallyDrop`) and every retained
            // source owner rather than free memory in flight.
            for page in &mut self.pages {
                if let Some(owner) = page.retained.take() {
                    std::mem::forget(owner);
                }
            }
            return;
        }
        // SAFETY: not taken (checked above) and dropped exactly once, here;
        // no work was issued since the last completed drain (or ever).
        unsafe { std::mem::ManuallyDrop::drop(&mut self.backend) };
    }
}

impl<B: PanelBackend> RingCore<B> {
    /// Hand the backend to its owner for explicit release. Only after a
    /// completed drain; the core then never drops it.
    pub(crate) fn take_backend(&mut self) -> R<B> {
        if self.dirty || self.backend_taken {
            return Err(invalid(
                "panel ring backend can only be taken after a completed drain",
            ));
        }
        self.backend_taken = true;
        // SAFETY: taken at most once (flag above); `Drop` skips it afterwards.
        Ok(unsafe { std::mem::ManuallyDrop::take(&mut self.backend) })
    }
}

impl<B: PanelBackend> RingCore<B> {
    pub(crate) fn new(spec: PanelPoolSpec, backend: B) -> R<Self> {
        spec.validate()?;
        let pages = spec.pages();
        Ok(Self {
            spec,
            pages: (0..pages).map(|_| Page::new()).collect(),
            free: (0..pages).collect(),
            next_epoch: 1,
            current: None,
            next: None,
            cursors: SmallVec::new(),
            poisoned: false,
            stats: PanelStats::default(),
            records: Vec::with_capacity(spec.timeline_records),
            marks: Vec::with_capacity(spec.timeline_layers),
            backend: std::mem::ManuallyDrop::new(backend),
            dirty: false,
            backend_taken: false,
        })
    }

    pub(crate) fn stats(&self) -> PanelStats {
        self.stats
    }

    pub(crate) fn is_quarantined(&self) -> bool {
        self.poisoned
    }

    fn check_live(&self) -> R<()> {
        if self.poisoned {
            return Err(invalid(
                "panel ring is quarantined by an earlier failure; drain it first",
            ));
        }
        Ok(())
    }

    /// Start of an admitted prompt window: every page must be free. Resets the
    /// statistics, the timeline and the layer window, and stamps the base event.
    pub(crate) fn begin_window(&mut self, ctx: &B::Ctx) -> R<()> {
        self.check_live()?;
        if let Some(busy) = self.pages.iter().position(|p| p.state != PageState::Free) {
            return Err(invalid(format!(
                "panel window starts with page {busy} still {:?}",
                self.pages[busy].state
            )));
        }
        self.current = None;
        self.next = None;
        self.cursors.clear();
        self.stats = PanelStats::default();
        self.records.clear();
        self.marks.clear();
        self.run(|core| core.backend.mark_base(ctx))
    }

    /// Declare the layers whose groups may be enqueued: `current` (compute
    /// consumers) and `next` (the one lookahead layer). `current` is `None`
    /// before the first host layer so its payload can already prefetch.
    pub(crate) fn set_layers(&mut self, current: Option<usize>, next: Option<usize>) -> R<()> {
        self.check_live()?;
        if let (Some(c), Some(n)) = (current, next) {
            if n <= c {
                return Err(invalid(format!(
                    "panel window next layer {n} is not after current {c}"
                )));
            }
        }
        match (self.current, current) {
            (Some(_), None) => {
                return Err(invalid("panel window cannot return to no current layer"))
            }
            (Some(old), Some(new)) if new < old => {
                return Err(invalid(format!(
                    "panel window moves back from layer {old} to {new}"
                )))
            }
            _ => {}
        }
        if let (Some(old), Some(new)) = (self.next, next) {
            if new < old {
                return Err(invalid(format!(
                    "panel lookahead moves back from layer {old} to {new}"
                )));
            }
        }
        if let Some(page) = self.pages.iter().position(|p| {
            p.state != PageState::Free && Some(p.layer) != current && Some(p.layer) != next
        }) {
            return Err(invalid(format!(
                "panel page {page} of layer {} is still live outside the new window",
                self.pages[page].layer
            )));
        }
        self.current = current;
        self.next = next;
        self.cursors
            .retain(|(layer, _)| Some(*layer) == current || Some(*layer) == next);
        Ok(())
    }

    fn classify(&self, layer: usize, scheduled: bool) -> R<Class> {
        if Some(layer) == self.current {
            Ok(Class::Current)
        } else if Some(layer) == self.next {
            if self.spec.lookahead_pages == 0 && !scheduled {
                return Err(invalid("panel pool has no lookahead pages"));
            }
            Ok(Class::Lookahead)
        } else {
            Err(invalid(format!(
                "panel layer {layer} is outside the window (current {:?}, next {:?})",
                self.current, self.next
            )))
        }
    }

    /// Run a mutating device sequence; a failure quarantines the ring.
    fn run<T>(&mut self, f: impl FnOnce(&mut Self) -> R<T>) -> R<T> {
        // From here on the device may hold queued work that reads this
        // ring's pages, tables and pinned sources.
        self.dirty = true;
        let result = f(self);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    /// Enqueue `group`, or `Ok(None)` (counted) when no free page of its class
    /// exists yet. Every validation failure happens before any mutation.
    pub(crate) fn try_enqueue(
        &mut self,
        ctx: &B::Ctx,
        group: &ExpertGroupPlan,
        sources: &LayerExpertSources,
    ) -> R<Option<PanelTicket>> {
        self.try_enqueue_inner(ctx, group, sources, false)
    }

    /// [`Self::try_enqueue`] for [`PanelSchedule`]: its ordering guarantees the
    /// current layer is fully enqueued before any next-layer group, so the next
    /// layer may also take the active slots the current layer leaves idle (and a
    /// pool without lookahead pages still prefetches).
    fn try_enqueue_scheduled(
        &mut self,
        ctx: &B::Ctx,
        group: &ExpertGroupPlan,
        sources: &LayerExpertSources,
    ) -> R<Option<PanelTicket>> {
        self.try_enqueue_inner(ctx, group, sources, true)
    }

    fn try_enqueue_inner(
        &mut self,
        ctx: &B::Ctx,
        group: &ExpertGroupPlan,
        sources: &LayerExpertSources,
        scheduled: bool,
    ) -> R<Option<PanelTicket>> {
        self.check_live()?;
        let class = self.classify(group.layer, scheduled)?;
        if sources.layer() != group.layer {
            return Err(invalid(format!(
                "panel group layer {} enqueued with sources of layer {}",
                group.layer,
                sources.layer()
            )));
        }
        if sources.num_experts() != self.spec.num_experts {
            return Err(invalid(format!(
                "panel sources hold a {}-entry table, the pool {}",
                sources.num_experts(),
                self.spec.num_experts
            )));
        }
        if group.experts.is_empty() || group.experts.len() > self.spec.experts_per_slot {
            return Err(invalid(format!(
                "panel group of {} experts is outside 1..={}",
                group.experts.len(),
                self.spec.experts_per_slot
            )));
        }
        if group.experts.windows(2).any(|w| w[0] >= w[1]) {
            return Err(invalid("panel group experts are not strictly ascending"));
        }
        for &id in &group.experts {
            match sources.expert(id) {
                Some(expert) if expert.codec == group.codec => {}
                Some(_) => {
                    return Err(invalid(format!(
                        "panel group codec {:?} does not match expert {id}",
                        group.codec
                    )))
                }
                None => return Err(invalid(format!("panel group expert {id} is not held"))),
            }
        }
        let cursor = self.cursors.iter().find(|(layer, _)| *layer == group.layer);
        if let Some(&(_, last)) = cursor {
            if group.group <= last {
                return Err(invalid(format!(
                    "panel layer {} group {} enqueued again or out of order (last {last}); one transfer per group per window",
                    group.layer, group.group
                )));
            }
        }
        let layout = plan_group_layout(&group.experts, sources)?;
        let page_bytes = self
            .spec
            .page_bytes()
            .ok_or_else(|| invalid("panel page size overflows"))?;
        if layout
            .used
            .checked_add(ARENA_TAIL_PAD_BYTES)
            .map_or(true, |need| need > page_bytes)
        {
            return Err(invalid(format!(
                "panel layer {} group {} needs {} payload bytes plus pad, a page holds {page_bytes}",
                group.layer, group.group, layout.used
            )));
        }
        let position = match class {
            Class::Current => (!self.free.is_empty()).then_some(0),
            // The schedule enqueues a next-layer group only after every group of
            // the current layer holds a page, so no active slot is reserved for
            // pending current-layer work: any free page serves.
            Class::Lookahead if scheduled => (!self.free.is_empty()).then_some(0),
            Class::Lookahead => self
                .free
                .iter()
                .position(|&slot| slot >= self.spec.active_slots),
        };
        let Some(position) = position else {
            self.stats.page_full_stalls += 1;
            return Ok(None);
        };
        let slot = self.free.remove(position);
        let epoch = self.next_epoch;
        self.next_epoch += 1;
        let rec = if self.spec.timeline_records == 0 {
            None
        } else if self.records.len() < self.spec.timeline_records {
            self.records.push(SpanRec {
                layer: group.layer,
                group: group.group,
                bytes: layout.submitted_bytes(),
                acquired: false,
                released: false,
            });
            Some(self.records.len() - 1)
        } else {
            self.stats.timeline_overflow += 1;
            None
        };
        self.run(|core| core.fill(ctx, slot, epoch, group, sources, &layout, rec))?;
        match self
            .cursors
            .iter_mut()
            .find(|(layer, _)| *layer == group.layer)
        {
            Some(entry) => entry.1 = group.group,
            None => self.cursors.push((group.layer, group.group)),
        }
        Ok(Some(PanelTicket {
            slot,
            epoch,
            layer: group.layer,
            group: group.group,
            codec: group.codec,
        }))
    }

    pub(crate) fn enqueue(
        &mut self,
        ctx: &B::Ctx,
        group: &ExpertGroupPlan,
        sources: &LayerExpertSources,
    ) -> R<PanelTicket> {
        self.try_enqueue(ctx, group, sources)?.ok_or_else(|| {
            invalid(format!(
                "panel ring is full: no free page for layer {} group {}",
                group.layer, group.group
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn fill(
        &mut self,
        ctx: &B::Ctx,
        slot: usize,
        epoch: u64,
        group: &ExpertGroupPlan,
        sources: &LayerExpertSources,
        layout: &GroupLayout,
        rec: Option<usize>,
    ) -> R<()> {
        let (filled, wait_free) = {
            let page = &self.pages[slot];
            (page.filled, page.free_recorded)
        };
        // The previous fill's copies must be done before its pinned table
        // staging is overwritten and its source owner released.
        if filled {
            self.backend.await_copy_done(ctx, slot)?;
        }
        let base = self.backend.page_base(slot);
        {
            let page = &mut self.pages[slot];
            page.retained = Some(Arc::clone(sources.owner()));
            page.state = PageState::Copying;
            page.epoch = epoch;
            page.layer = group.layer;
            page.group = group.group;
            page.codec = group.codec;
            page.experts = group.experts.clone();
            page.host_base = sources.host_range().0;
            page.payload_bytes = layout.submitted_bytes();
            page.used = layout.used;
            page.acquired = false;
            page.filled = true;
            page.rec = rec;
        }
        let (gate_up, down) = self.backend.tables_mut(slot);
        fill_tables(layout, &group.experts, base, gate_up, down)?;
        self.backend.copy_begin(ctx, slot, wait_free, rec)?;
        for run in &layout.runs {
            self.backend
                .copy_payload(ctx, slot, run.src, run.dst, run.len)?;
        }
        self.backend
            .copy_zero(ctx, slot, layout.used, ARENA_TAIL_PAD_BYTES)?;
        self.backend.copy_tables(ctx, slot)?;
        self.backend.copy_end(ctx, slot, rec)?;
        self.pages[slot].state = PageState::Ready;
        self.stats.enqueued += 1;
        self.stats.submitted_payload_bytes += layout.submitted_bytes() as u64;
        self.stats.extent_bytes += layout.extent_bytes as u64;
        self.stats.table_bytes += (2 * self.spec.num_experts * std::mem::size_of::<u64>()) as u64;
        Ok(())
    }

    /// Resolve a ticket to its page, refusing stale or forged ones.
    fn checked_slot(&self, ticket: PanelTicket, allowed: &[PageState]) -> R<usize> {
        let page = self
            .pages
            .get(ticket.slot)
            .ok_or_else(|| invalid(format!("panel ticket slot {} is out of range", ticket.slot)))?;
        if page.epoch != ticket.epoch {
            return Err(invalid(format!(
                "stale panel ticket: slot {} epoch {} was refilled (now epoch {})",
                ticket.slot, ticket.epoch, page.epoch
            )));
        }
        if page.layer != ticket.layer {
            return Err(invalid(format!(
                "stale panel ticket: layer {} does not match the page's layer {}",
                ticket.layer, page.layer
            )));
        }
        if page.group != ticket.group {
            return Err(invalid(format!(
                "stale panel ticket: group {} does not match the page's group {}",
                ticket.group, page.group
            )));
        }
        if page.codec != ticket.codec {
            return Err(invalid(format!(
                "stale panel ticket: codec {:?} does not match the page's codec {:?}",
                ticket.codec, page.codec
            )));
        }
        if !allowed.contains(&page.state) {
            return Err(invalid(format!(
                "panel ticket slot {} epoch {} is {:?}, expected one of {allowed:?}",
                ticket.slot, ticket.epoch, page.state
            )));
        }
        Ok(ticket.slot)
    }

    /// `Ready -> InUse`: the compute stream waits for the copy.
    pub(crate) fn acquire(&mut self, ctx: &B::Ctx, ticket: PanelTicket) -> R<()> {
        self.check_live()?;
        let slot = self.checked_slot(ticket, &[PageState::Ready])?;
        let rec = self.pages[slot].rec;
        let starved = self.run(|core| core.backend.compute_acquire(ctx, slot, rec))?;
        if starved {
            self.stats.copy_starved_stalls += 1;
        }
        let page = &mut self.pages[slot];
        page.state = PageState::InUse;
        page.acquired = true;
        self.stats.acquired += 1;
        self.stats.acquired_payload_bytes += page.payload_bytes as u64;
        if let Some(record) = rec.and_then(|r| self.records.get_mut(r)) {
            record.acquired = true;
        }
        Ok(())
    }

    /// The page is leased: it must be `InUse` under exactly this ticket.
    fn leased(&self, ticket: PanelTicket) -> R<&Page> {
        let slot = self.checked_slot(ticket, &[PageState::InUse])?;
        Ok(&self.pages[slot])
    }

    /// `InUse|Ready -> Retiring -> Free`: call after the group's last reader
    /// was enqueued on the compute stream. A `Ready` page was prefetched and
    /// never read; it is released without a read and counted as discarded.
    pub(crate) fn retire(&mut self, ctx: &B::Ctx, ticket: PanelTicket) -> R<()> {
        self.check_live()?;
        let slot = self.checked_slot(ticket, &[PageState::InUse, PageState::Ready])?;
        let (acquired, rec, bytes) = {
            let page = &mut self.pages[slot];
            page.state = PageState::Retiring;
            (page.acquired, page.rec, page.payload_bytes)
        };
        let compute_rec = if acquired { rec } else { None };
        self.run(|core| core.backend.compute_release(ctx, slot, compute_rec))?;
        let page = &mut self.pages[slot];
        page.state = PageState::Free;
        page.free_recorded = true;
        self.free.push(slot);
        self.stats.retired += 1;
        if !acquired {
            self.stats.discarded += 1;
            self.stats.discarded_payload_bytes += bytes as u64;
        }
        if let Some(record) = rec.and_then(|r| self.records.get_mut(r)) {
            record.released = acquired;
        }
        Ok(())
    }

    /// Drain both streams, then force every page free, release every retained
    /// source, end the window and clear the quarantine. Outstanding tickets go
    /// stale. A failed drain keeps the quarantine and every owner alive.
    pub(crate) fn drain(&mut self, ctx: &B::Ctx) -> R<()> {
        if let Err(error) = self.backend.drain(ctx) {
            self.poisoned = true;
            return Err(error);
        }
        // Both streams completed: nothing queued by this ring is in flight.
        self.dirty = false;
        for page in &mut self.pages {
            page.state = PageState::Free;
            page.retained = None;
            page.filled = false;
            page.free_recorded = false;
            page.rec = None;
            page.acquired = false;
        }
        self.free.clear();
        self.free.extend(0..self.pages.len());
        self.current = None;
        self.next = None;
        self.cursors.clear();
        self.poisoned = false;
        Ok(())
    }

    pub(crate) fn mark_layer_begin(&mut self, ctx: &B::Ctx, layer: usize) -> R<()> {
        self.check_live()?;
        if self.spec.timeline_layers == 0 {
            return Ok(());
        }
        if self.marks.len() >= self.spec.timeline_layers {
            self.stats.layer_mark_overflow += 1;
            return Ok(());
        }
        let mark = self.marks.len();
        self.marks.push(LayerMark {
            layer,
            ended: false,
        });
        self.run(|core| core.backend.mark_layer(ctx, mark, false))
    }

    pub(crate) fn mark_layer_end(&mut self, ctx: &B::Ctx, layer: usize) -> R<()> {
        self.check_live()?;
        if self.spec.timeline_layers == 0 {
            return Ok(());
        }
        let Some(mark) = self
            .marks
            .iter()
            .rposition(|m| m.layer == layer && !m.ended)
        else {
            if self.stats.layer_mark_overflow > 0 {
                return Ok(());
            }
            return Err(invalid(format!("layer {layer} has no open compute mark")));
        };
        self.marks[mark].ended = true;
        self.run(|core| core.backend.mark_layer(ctx, mark, true))
    }

    /// Intervals and receipt numbers of this window. Call after [`Self::drain`]
    /// (or once the window's work completed): every event must have completed.
    pub(crate) fn timeline(&self, ctx: &B::Ctx) -> R<PanelTimeline> {
        let ms = |from, to| self.backend.elapsed_ms(ctx, from, to);
        let mut spans = Vec::with_capacity(self.records.len());
        for (index, record) in self.records.iter().enumerate() {
            let compute = if record.acquired && record.released {
                Some((
                    ms(TimelineEvent::Base, TimelineEvent::ComputeBegin(index))?,
                    ms(TimelineEvent::Base, TimelineEvent::ComputeEnd(index))?,
                ))
            } else {
                None
            };
            spans.push(PanelSpanMs {
                layer: record.layer,
                group: record.group,
                bytes: record.bytes,
                copy_start: ms(TimelineEvent::Base, TimelineEvent::CopyStart(index))?,
                copy_end: ms(TimelineEvent::Base, TimelineEvent::CopyEnd(index))?,
                compute,
            });
        }
        let mut layers = Vec::with_capacity(self.marks.len());
        for (index, mark) in self.marks.iter().enumerate() {
            if !mark.ended {
                continue;
            }
            layers.push(LayerSpanMs {
                layer: mark.layer,
                begin: ms(TimelineEvent::Base, TimelineEvent::LayerBegin(index))?,
                end: ms(TimelineEvent::Base, TimelineEvent::LayerEnd(index))?,
            });
        }
        let overlap = summarize_overlap(&spans, &layers, &self.stats);
        Ok(PanelTimeline {
            spans,
            layers,
            overlap,
            stats: self.stats,
        })
    }
}

// ── layer-ahead schedule ─────────────────────────────────────────────

/// The weight schedule a window runs against the page ring.
///
/// [`PanelSchedule`] is S1's page-order sweep (layer ahead, one transfer per
/// group, prefetching before the route exists); [`SpSchedule`] is the
/// short-prompt hook's selective schedule (one layer at a time, only the
/// groups the planner chose, no lookahead). Both drive the same [`RingCore`],
/// so page lifecycle, tickets, events, drain and quarantine are shared.
pub(crate) trait RingSchedule {
    /// Start of the window; returns the number of groups enqueued.
    fn prime<B: PanelBackend>(&mut self, core: &mut RingCore<B>, ctx: &B::Ctx) -> R<usize>;

    /// Compute reaches host layer `layer`; returns the groups enqueued.
    fn enter_layer<B: PanelBackend>(
        &mut self,
        core: &mut RingCore<B>,
        ctx: &B::Ctx,
        layer: usize,
    ) -> R<usize>;

    /// Enqueue whatever the schedule's policy allows (call after each retire).
    fn pump<B: PanelBackend>(&mut self, core: &mut RingCore<B>, ctx: &B::Ctx) -> R<usize>;

    /// The ticket of current-layer `group`, which compute needs now.
    fn require<B: PanelBackend>(
        &mut self,
        core: &mut RingCore<B>,
        ctx: &B::Ctx,
        layer: usize,
        group: u32,
    ) -> R<PanelTicket>;

    /// The ticket of `(layer, group)` once enqueued.
    fn ticket(&self, layer: usize, group: u32) -> Option<PanelTicket>;

    /// Payload bytes this window's schedule submits: the full sweep for
    /// [`PanelSchedule`], the bytes of the groups enqueued so far this window
    /// for [`SpSchedule`] (equal to the sum of their layouts' submitted bytes).
    fn submitted_bytes(&self) -> u64;
}

struct ScheduledLayer {
    sources: LayerExpertSources,
    groups: Vec<ExpertGroupPlan>,
    /// Next group index to enqueue.
    next_group: usize,
    tickets: Vec<Option<PanelTicket>>,
}

/// The frozen weight schedule of one window: every host layer's groups in
/// ascending layer/expert order, at most one layer ahead of compute, one
/// transfer per group. [`Self::pump`] enqueues missing current-layer groups
/// first, then the next layer's, until the pool is full.
pub(crate) struct PanelSchedule {
    layers: Vec<ScheduledLayer>,
    /// Index of the layer compute is on; `None` before the first host layer.
    current: Option<usize>,
    sweep_bytes: u64,
    extent_bytes: u64,
}

impl PanelSchedule {
    /// `layers` must be ascending by layer index. Plans every group's layout
    /// now so a malformed source fails before any transfer.
    pub(crate) fn new(layers: Vec<LayerExpertSources>, experts_per_slot: usize) -> R<Self> {
        let mut scheduled = Vec::with_capacity(layers.len());
        let (mut sweep_bytes, mut extent_bytes) = (0u64, 0u64);
        let mut previous: Option<usize> = None;
        for sources in layers {
            if previous.is_some_and(|p| sources.layer() <= p) {
                return Err(invalid("panel schedule layers are not strictly ascending"));
            }
            previous = Some(sources.layer());
            let groups = ExpertGroupPlan::partition(&sources, experts_per_slot)?;
            for group in &groups {
                let layout = plan_group_layout(&group.experts, &sources)?;
                sweep_bytes += layout.submitted_bytes() as u64;
                extent_bytes += layout.extent_bytes as u64;
            }
            scheduled.push(ScheduledLayer {
                tickets: vec![None; groups.len()],
                next_group: 0,
                groups,
                sources,
            });
        }
        Ok(Self {
            layers: scheduled,
            current: None,
            sweep_bytes,
            extent_bytes,
        })
    }

    /// Payload bytes one full sweep submits (all groups' coalesced runs).
    pub(crate) fn sweep_bytes(&self) -> u64 {
        self.sweep_bytes
    }

    /// Sum of the experts' own extents over the sweep.
    pub(crate) fn extent_bytes(&self) -> u64 {
        self.extent_bytes
    }

    /// Own payload bytes of `layer`'s expert `id` (receipt touched bytes).
    pub(crate) fn expert_extent_bytes(&self, layer: usize, id: usize) -> Option<usize> {
        self.layers
            .iter()
            .find(|l| l.sources.layer() == layer)
            .and_then(|l| l.sources.expert_extent_bytes(id))
    }

    pub(crate) fn layer_groups(&self, layer: usize) -> Option<&[ExpertGroupPlan]> {
        self.layers
            .iter()
            .find(|l| l.sources.layer() == layer)
            .map(|l| l.groups.as_slice())
    }

    fn next_index(&self) -> Option<usize> {
        let next = self.current.map_or(0, |c| c + 1);
        (next < self.layers.len()).then_some(next)
    }

    fn apply_window<B: PanelBackend>(&self, core: &mut RingCore<B>) -> R<()> {
        core.set_layers(
            self.current.map(|c| self.layers[c].sources.layer()),
            self.next_index().map(|n| self.layers[n].sources.layer()),
        )
    }

    /// The expert sources of host layer `layer`.
    pub(crate) fn sources_of(&self, layer: usize) -> Option<&LayerExpertSources> {
        self.layers
            .iter()
            .find(|l| l.sources.layer() == layer)
            .map(|l| &l.sources)
    }

    // The [`RingSchedule`] operations are also inherent, so callers that
    // drive a `PanelSchedule` directly need not import the trait.

    /// Start of the window: lookahead covers the first host layer while
    /// resident layers compute. Pumps immediately.
    pub(crate) fn prime<B: PanelBackend>(
        &mut self,
        core: &mut RingCore<B>,
        ctx: &B::Ctx,
    ) -> R<usize> {
        RingSchedule::prime(self, core, ctx)
    }

    /// Compute reaches host layer `layer` (the next in the schedule).
    pub(crate) fn enter_layer<B: PanelBackend>(
        &mut self,
        core: &mut RingCore<B>,
        ctx: &B::Ctx,
        layer: usize,
    ) -> R<usize> {
        RingSchedule::enter_layer(self, core, ctx, layer)
    }

    /// Enqueue as many groups as free pages allow: the current layer's
    /// missing groups first, then the next layer's.
    pub(crate) fn pump<B: PanelBackend>(
        &mut self,
        core: &mut RingCore<B>,
        ctx: &B::Ctx,
    ) -> R<usize> {
        RingSchedule::pump(self, core, ctx)
    }

    /// The ticket of `(layer, group)` once enqueued.
    pub(crate) fn ticket(&self, layer: usize, group: u32) -> Option<PanelTicket> {
        RingSchedule::ticket(self, layer, group)
    }

    /// The ticket of a current-layer group compute needs now, enqueueing the
    /// groups before it if the pool allowed none yet. Errors (no state
    /// change) when the pool is full: retire earlier tickets first.
    pub(crate) fn require<B: PanelBackend>(
        &mut self,
        core: &mut RingCore<B>,
        ctx: &B::Ctx,
        layer: usize,
        group: u32,
    ) -> R<PanelTicket> {
        RingSchedule::require(self, core, ctx, layer, group)
    }
}

impl RingSchedule for PanelSchedule {
    fn prime<B: PanelBackend>(&mut self, core: &mut RingCore<B>, ctx: &B::Ctx) -> R<usize> {
        self.current = None;
        for layer in &mut self.layers {
            layer.next_group = 0;
            layer.tickets.iter_mut().for_each(|t| *t = None);
        }
        core.begin_window(ctx)?;
        self.apply_window(core)?;
        RingSchedule::pump(self, core, ctx)
    }

    fn enter_layer<B: PanelBackend>(
        &mut self,
        core: &mut RingCore<B>,
        ctx: &B::Ctx,
        layer: usize,
    ) -> R<usize> {
        let expected = self.next_index().ok_or_else(|| {
            invalid(format!(
                "panel schedule has no layer after {:?}",
                self.current
            ))
        })?;
        if self.layers[expected].sources.layer() != layer {
            return Err(invalid(format!(
                "panel schedule expects layer {} next, compute entered {layer}",
                self.layers[expected].sources.layer()
            )));
        }
        self.current = Some(expected);
        self.apply_window(core)?;
        RingSchedule::pump(self, core, ctx)
    }

    fn pump<B: PanelBackend>(&mut self, core: &mut RingCore<B>, ctx: &B::Ctx) -> R<usize> {
        let mut issued = 0;
        for index in [self.current, self.next_index()].into_iter().flatten() {
            let layer = &mut self.layers[index];
            while layer.next_group < layer.groups.len() {
                let group = layer.next_group;
                match core.try_enqueue_scheduled(ctx, &layer.groups[group], &layer.sources)? {
                    Some(ticket) => {
                        layer.tickets[group] = Some(ticket);
                        layer.next_group += 1;
                        issued += 1;
                    }
                    None => return Ok(issued),
                }
            }
        }
        Ok(issued)
    }

    fn require<B: PanelBackend>(
        &mut self,
        core: &mut RingCore<B>,
        ctx: &B::Ctx,
        layer: usize,
        group: u32,
    ) -> R<PanelTicket> {
        if let Some(ticket) = RingSchedule::ticket(&*self, layer, group) {
            return Ok(ticket);
        }
        RingSchedule::pump(self, core, ctx)?;
        RingSchedule::ticket(&*self, layer, group).ok_or_else(|| {
            invalid(format!(
                "panel layer {layer} group {group} cannot be enqueued: the pool is full or the layer is not current"
            ))
        })
    }

    fn ticket(&self, layer: usize, group: u32) -> Option<PanelTicket> {
        self.layers
            .iter()
            .find(|l| l.sources.layer() == layer)
            .and_then(|l| l.tickets.get(group as usize).copied().flatten())
    }

    fn submitted_bytes(&self) -> u64 {
        self.sweep_bytes
    }
}

// ── selective (short-prompt) schedule ────────────────────────────────

/// One layer's planned groups and what the ring did with them.
struct SpPlannedLayer {
    layer: usize,
    groups: Vec<ExpertGroupPlan>,
    /// Per group: `(submitted payload bytes, payload end of its page layout)`.
    layouts: Vec<(u64, usize)>,
    tickets: Vec<Option<PanelTicket>>,
    /// `enter_layer` enqueued the groups.
    entered: bool,
}

/// The short-prompt hook's schedule: per layer, only the groups the planner
/// chose (sparse ascending experts), all enqueued at layer entry, never
/// prefetched and never waited for. Sources come from the window's
/// [`PanelSchedule`] (borrowed, read-only). One transfer per group per window.
///
/// Per layer the order is `plan_layer(L, groups)` then `enter_layer(L)`; every
/// group's tickets must be retired before the next layer is entered.
pub(crate) struct SpSchedule<'a> {
    sources: &'a PanelSchedule,
    plan: Option<SpPlannedLayer>,
    /// The last layer entered this window.
    current: Option<usize>,
    submitted: u64,
}

#[allow(dead_code)]
impl<'a> SpSchedule<'a> {
    pub(crate) fn new(sources: &'a PanelSchedule) -> Self {
        Self {
            sources,
            plan: None,
            current: None,
            submitted: 0,
        }
    }

    /// Plan host layer `layer`'s streamed set: `groups` are ascending expert
    /// ids, each group `1..=MAX_GROUP_EXPERTS` long, ascending across groups
    /// (so no expert streams twice), one codec per group. An empty list is a
    /// layer that streams nothing. Pure: no ring or device state is touched,
    /// and a refusal leaves the previous state unchanged.
    pub(crate) fn plan_layer(&mut self, layer: usize, groups: &[Vec<u16>]) -> R<()> {
        if let Some(open) = self.plan.as_ref().filter(|p| !p.entered) {
            return Err(invalid(format!(
                "sp schedule: layer {} is planned but not entered; layer {layer} cannot be planned over it",
                open.layer
            )));
        }
        if let Some(current) = self.current.filter(|&c| layer <= c) {
            return Err(invalid(format!(
                "sp schedule: layer {layer} is not after the entered layer {current}"
            )));
        }
        let sources = self
            .sources
            .sources_of(layer)
            .ok_or_else(|| invalid(format!("sp schedule: layer {layer} is not a host layer")))?;
        let mut plans = Vec::with_capacity(groups.len());
        let mut layouts = Vec::with_capacity(groups.len());
        let mut previous: Option<usize> = None;
        for (index, ids) in groups.iter().enumerate() {
            let group = u32::try_from(index)
                .map_err(|_| invalid("sp schedule: group index overflows u32"))?;
            if ids.is_empty() || ids.len() > MAX_GROUP_EXPERTS {
                return Err(invalid(format!(
                    "sp schedule layer {layer} group {group}: {} experts is outside 1..={MAX_GROUP_EXPERTS}",
                    ids.len()
                )));
            }
            let experts: SmallVec<[usize; MAX_GROUP_EXPERTS]> =
                ids.iter().map(|&id| usize::from(id)).collect();
            for &id in &experts {
                if previous.is_some_and(|p| id <= p) {
                    return Err(invalid(format!(
                        "sp schedule layer {layer} group {group}: expert {id} is not strictly ascending across the layer's groups"
                    )));
                }
                previous = Some(id);
            }
            let codec = sources
                .expert(experts[0])
                .ok_or_else(|| {
                    invalid(format!(
                        "sp schedule layer {layer} group {group}: expert {} is not held",
                        experts[0]
                    ))
                })?
                .codec;
            for &id in &experts {
                match sources.expert(id) {
                    Some(expert) if expert.codec == codec => {}
                    Some(_) => {
                        return Err(invalid(format!(
                            "sp schedule layer {layer} group {group}: expert {id} does not share codec {codec:?}"
                        )))
                    }
                    None => {
                        return Err(invalid(format!(
                            "sp schedule layer {layer} group {group}: expert {id} is not held"
                        )))
                    }
                }
            }
            let layout = plan_group_layout(&experts, sources)?;
            layouts.push((layout.submitted_bytes() as u64, layout.used));
            plans.push(ExpertGroupPlan {
                layer,
                group,
                codec,
                experts,
            });
        }
        self.plan = Some(SpPlannedLayer {
            layer,
            tickets: vec![None; plans.len()],
            groups: plans,
            layouts,
            entered: false,
        });
        Ok(())
    }

    /// The planned groups of `layer` (the layer last planned).
    pub(crate) fn planned_groups(&self, layer: usize) -> Option<&[ExpertGroupPlan]> {
        self.plan
            .as_ref()
            .filter(|p| p.layer == layer)
            .map(|p| p.groups.as_slice())
    }
}

impl RingSchedule for SpSchedule<'_> {
    /// `begin_window` plus `set_layers(None, None)`: zero enqueues.
    fn prime<B: PanelBackend>(&mut self, core: &mut RingCore<B>, ctx: &B::Ctx) -> R<usize> {
        self.plan = None;
        self.current = None;
        self.submitted = 0;
        core.begin_window(ctx)?;
        core.set_layers(None, None)?;
        Ok(0)
    }

    /// Enqueue every planned group of `layer` at once. Every refusal that can
    /// be decided from the plan (not planned, more groups than free pages, an
    /// oversized group) happens before the ring is touched. A group the ring
    /// then cannot take is a protocol bug: the ring is quarantined and the
    /// caller must drain; the host never waits for a page.
    fn enter_layer<B: PanelBackend>(
        &mut self,
        core: &mut RingCore<B>,
        ctx: &B::Ctx,
        layer: usize,
    ) -> R<usize> {
        core.check_live()?;
        let base = self.sources;
        let plan = match self.plan.as_mut() {
            Some(plan) if plan.layer == layer && !plan.entered => plan,
            Some(plan) if plan.layer == layer => {
                return Err(invalid(format!(
                    "sp schedule: layer {layer} was already entered this window"
                )))
            }
            other => {
                return Err(invalid(format!(
                    "sp schedule: layer {layer} entered without plan_layer (planned {:?})",
                    other.map(|p| p.layer)
                )))
            }
        };
        let sources = base
            .sources_of(layer)
            .ok_or_else(|| invalid(format!("sp schedule: layer {layer} is not a host layer")))?;
        if sources.num_experts() != core.spec.num_experts {
            return Err(invalid(format!(
                "sp schedule: sources hold a {}-entry table, the pool {}",
                sources.num_experts(),
                core.spec.num_experts
            )));
        }
        let groups = plan.groups.len();
        if groups > core.free.len() {
            return Err(invalid(format!(
                "sp schedule layer {layer}: {groups} groups need {groups} pages, the ring has {} free of {}",
                core.free.len(),
                core.pages.len()
            )));
        }
        let page_bytes = core
            .spec
            .page_bytes()
            .ok_or_else(|| invalid("panel page size overflows"))?;
        for (group, &(_, used)) in plan.groups.iter().zip(&plan.layouts) {
            if group.experts.len() > core.spec.experts_per_slot {
                return Err(invalid(format!(
                    "sp schedule layer {layer} group {}: {} experts exceed the pool's {} per page",
                    group.group,
                    group.experts.len(),
                    core.spec.experts_per_slot
                )));
            }
            if used
                .checked_add(ARENA_TAIL_PAD_BYTES)
                .map_or(true, |need| need > page_bytes)
            {
                return Err(invalid(format!(
                    "sp schedule layer {layer} group {} needs {used} payload bytes plus pad, a page holds {page_bytes}",
                    group.group
                )));
            }
        }
        core.set_layers(Some(layer), None)?;
        plan.entered = true;
        self.current = Some(layer);
        for index in 0..groups {
            match core.try_enqueue_scheduled(ctx, &plan.groups[index], sources)? {
                Some(ticket) => {
                    plan.tickets[index] = Some(ticket);
                    self.submitted += plan.layouts[index].0;
                }
                None => {
                    core.poisoned = true;
                    return Err(invalid(format!(
                        "sp schedule layer {layer} group {index}: no free page after {index} of {groups} groups although {groups} pages were free (protocol bug; drain the ring)"
                    )));
                }
            }
        }
        Ok(groups)
    }

    /// Nothing is prefetched or refilled: every group of the layer already
    /// holds its page.
    fn pump<B: PanelBackend>(&mut self, _core: &mut RingCore<B>, _ctx: &B::Ctx) -> R<usize> {
        Ok(0)
    }

    /// The ticket of an enqueued group; never enqueues or waits.
    fn require<B: PanelBackend>(
        &mut self,
        _core: &mut RingCore<B>,
        _ctx: &B::Ctx,
        layer: usize,
        group: u32,
    ) -> R<PanelTicket> {
        RingSchedule::ticket(&*self, layer, group).ok_or_else(|| {
            invalid(format!(
                "sp schedule layer {layer} group {group} has no ticket: it is not in the entered layer's plan"
            ))
        })
    }

    fn ticket(&self, layer: usize, group: u32) -> Option<PanelTicket> {
        self.plan
            .as_ref()
            .filter(|p| p.layer == layer && p.entered)
            .and_then(|p| p.tickets.get(group as usize).copied().flatten())
    }

    fn submitted_bytes(&self) -> u64 {
        self.submitted
    }
}

// ── HIP backend and public ring ──────────────────────────────────────

/// The pool's pinned host staging for the per-page pointer tables.
///
/// `hip_bridge::PinnedHostBuffer` holds a raw `hipHostMalloc` pointer and is
/// therefore not `Send`; `Qwen4GpuForward` (and with it the architecture model
/// object) must be, so the ring's owner asserts the narrow invariant here
/// instead of widening anything in `hip-bridge` or the bundle.
struct PinnedStaging(hip_bridge::PinnedHostBuffer);

// SAFETY: `PinnedStaging` is the unique owner of its hipHostMalloc allocation:
// nothing aliases or caches the pointer, and the only host access is
// `as_bytes_mut` through `&mut self`, so moving the owner to another thread
// cannot create concurrent access. The allocation does not move with its
// owner and the HIP library it frees through is thread-agnostic and kept
// alive by the buffer itself. Device-side reads of the staging (the async
// table uploads) are fenced by the ring: the next write to a page's staging
// waits on that page's `ready` event on the host (`await_copy_done`), and
// `drain` / `free_gpu` synchronize both streams before the buffer can be
// dropped (a failed drain leaks it rather than freeing it). Deliberately NOT
// `Sync`: concurrent host access is unsupported.
unsafe impl Send for PinnedStaging {}

impl PinnedStaging {
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.0.as_bytes_mut()
    }
}

const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<HipPanelBackend>();
    assert_send::<ExpertPanelRing>();
    assert_send::<PanelSchedule>();
};

/// Real device resources of the pool.
pub(crate) struct HipPanelBackend {
    copy: hip_bridge::Stream,
    payloads: Vec<GpuTensor>,
    gate_ups: Vec<GpuTensor>,
    downs: Vec<GpuTensor>,
    ready: Vec<hip_bridge::Event>,
    free: Vec<hip_bridge::Event>,
    /// Pinned host staging of every page's two pointer tables.
    staging: PinnedStaging,
    table_len: usize,
    base: Option<hip_bridge::Event>,
    /// Four per record: copy start/end, compute begin/end.
    timing: Vec<hip_bridge::Event>,
    /// Two per layer mark: begin/end.
    layer_timing: Vec<hip_bridge::Event>,
}

fn compute_stream(gpu: &Gpu) -> R<&hip_bridge::Stream> {
    gpu.active_stream
        .as_ref()
        .ok_or_else(|| invalid("panel ring: compute stream missing"))
}

impl HipPanelBackend {
    fn new(gpu: &mut Gpu, spec: &PanelPoolSpec) -> R<Self> {
        let reserve = spec
            .reserve()
            .ok_or_else(|| invalid("panel pool byte size overflows"))?;
        let staging = PinnedStaging(gpu.hip.host_buffer(reserve.staging)?);
        gpu.ensure_capture_stream()?;
        let copy = gpu.hip.stream_create_non_blocking()?;
        let mut backend = Self {
            copy,
            payloads: Vec::with_capacity(reserve.pages),
            gate_ups: Vec::with_capacity(reserve.pages),
            downs: Vec::with_capacity(reserve.pages),
            ready: Vec::with_capacity(reserve.pages),
            free: Vec::with_capacity(reserve.pages),
            staging,
            table_len: spec.num_experts,
            base: None,
            timing: Vec::new(),
            layer_timing: Vec::new(),
        };
        if let Err(error) = backend.populate(gpu, spec, &reserve) {
            let _ = backend.release(gpu);
            return Err(error);
        }
        Ok(backend)
    }

    fn populate(
        &mut self,
        gpu: &mut Gpu,
        spec: &PanelPoolSpec,
        reserve: &PanelPoolReserve,
    ) -> R<()> {
        let table_bytes = spec.num_experts * std::mem::size_of::<u64>();
        for _ in 0..reserve.pages {
            self.payloads
                .push(gpu.alloc_tensor(&[reserve.page_bytes], DType::Raw)?);
            for tables in [&mut self.gate_ups, &mut self.downs] {
                let tensor = gpu.alloc_tensor(&[2 * spec.num_experts], DType::F32)?;
                gpu.hip.memset(&tensor.buf, 0, table_bytes)?;
                tables.push(tensor);
            }
            self.ready.push(
                gpu.hip
                    .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
            );
            self.free.push(
                gpu.hip
                    .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
            );
        }
        if spec.timeline_records > 0 || spec.timeline_layers > 0 {
            self.base = Some(gpu.hip.event_create()?);
        }
        for _ in 0..4 * spec.timeline_records {
            self.timing.push(gpu.hip.event_create()?);
        }
        for _ in 0..2 * spec.timeline_layers {
            self.layer_timing.push(gpu.hip.event_create()?);
        }
        Ok(())
    }

    /// Free every device resource. Streams must be drained.
    fn release(mut self, gpu: &mut Gpu) -> R<()> {
        let mut first: Option<Qwen4GpuForwardError> = None;
        let mut note = |result: Result<(), hip_bridge::HipError>| {
            if let Err(error) = result {
                first.get_or_insert(error.into());
            }
        };
        for tensor in self
            .payloads
            .drain(..)
            .chain(self.gate_ups.drain(..))
            .chain(self.downs.drain(..))
        {
            note(gpu.free_tensor(tensor));
        }
        for event in self
            .ready
            .drain(..)
            .chain(self.free.drain(..))
            .chain(self.base.take())
            .chain(self.timing.drain(..))
            .chain(self.layer_timing.drain(..))
        {
            note(gpu.hip.event_destroy(event));
        }
        note(gpu.hip.stream_destroy(self.copy));
        match first {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn timeline_event(&self, event: TimelineEvent) -> R<&hip_bridge::Event> {
        let missing = || invalid(format!("panel timeline event {event:?} was not allocated"));
        match event {
            TimelineEvent::Base => self.base.as_ref().ok_or_else(missing),
            TimelineEvent::CopyStart(r) => self.timing.get(4 * r).ok_or_else(missing),
            TimelineEvent::CopyEnd(r) => self.timing.get(4 * r + 1).ok_or_else(missing),
            TimelineEvent::ComputeBegin(r) => self.timing.get(4 * r + 2).ok_or_else(missing),
            TimelineEvent::ComputeEnd(r) => self.timing.get(4 * r + 3).ok_or_else(missing),
            TimelineEvent::LayerBegin(m) => self.layer_timing.get(2 * m).ok_or_else(missing),
            TimelineEvent::LayerEnd(m) => self.layer_timing.get(2 * m + 1).ok_or_else(missing),
        }
    }
}

impl PanelBackend for HipPanelBackend {
    type Ctx = Gpu;

    fn page_base(&self, slot: usize) -> usize {
        self.payloads[slot].buf.as_ptr() as usize
    }

    fn tables_mut(&mut self, slot: usize) -> (&mut [u64], &mut [u64]) {
        let n = self.table_len;
        let stride = 2 * n * std::mem::size_of::<u64>();
        let region = &mut self.staging.as_bytes_mut()[slot * stride..(slot + 1) * stride];
        // SAFETY: hipHostMalloc storage is page aligned and the region starts
        // at a multiple of 8 bytes, so the u64 view has no unaligned head/tail
        // and covers exactly 2 * n entries.
        let (head, entries, tail) = unsafe { region.align_to_mut::<u64>() };
        assert!(head.is_empty() && tail.is_empty() && entries.len() == 2 * n);
        entries.split_at_mut(n)
    }

    fn await_copy_done(&mut self, gpu: &Gpu, slot: usize) -> R<()> {
        gpu.hip.event_synchronize(&self.ready[slot])?;
        Ok(())
    }

    fn copy_begin(&mut self, gpu: &Gpu, slot: usize, wait_free: bool, rec: Option<usize>) -> R<()> {
        if wait_free {
            gpu.hip.stream_wait_event(&self.copy, &self.free[slot])?;
        }
        if let Some(rec) = rec {
            gpu.hip.event_record(
                self.timeline_event(TimelineEvent::CopyStart(rec))?,
                Some(&self.copy),
            )?;
        }
        Ok(())
    }

    fn copy_payload(
        &mut self,
        gpu: &Gpu,
        slot: usize,
        src: usize,
        dst: usize,
        len: usize,
    ) -> R<()> {
        // SAFETY: `src..src+len` was checked against the layer's pinned host
        // range in `LayerExpertSources::new`; the ring retains the owner until
        // this copy is known complete.
        let source = unsafe { std::slice::from_raw_parts(src as *const u8, len) };
        gpu.hip.memcpy_htod_async(
            &self.payloads[slot].buf.byte_view(dst, len),
            source,
            &self.copy,
        )?;
        Ok(())
    }

    fn copy_zero(&mut self, gpu: &Gpu, slot: usize, dst: usize, len: usize) -> R<()> {
        gpu.hip.memset_async(
            &self.payloads[slot].buf.byte_view(dst, len),
            0,
            len,
            &self.copy,
        )?;
        Ok(())
    }

    fn copy_tables(&mut self, gpu: &Gpu, slot: usize) -> R<()> {
        let n = self.table_len;
        let table_bytes = n * std::mem::size_of::<u64>();
        let stride = 2 * table_bytes;
        let region = &self.staging.as_bytes_mut()[slot * stride..(slot + 1) * stride];
        let (gate_up, down) = region.split_at(table_bytes);
        gpu.hip
            .memcpy_htod_async(&self.gate_ups[slot].buf, gate_up, &self.copy)?;
        gpu.hip
            .memcpy_htod_async(&self.downs[slot].buf, down, &self.copy)?;
        Ok(())
    }

    fn copy_end(&mut self, gpu: &Gpu, slot: usize, rec: Option<usize>) -> R<()> {
        if let Some(rec) = rec {
            gpu.hip.event_record(
                self.timeline_event(TimelineEvent::CopyEnd(rec))?,
                Some(&self.copy),
            )?;
        }
        gpu.hip.event_record(&self.ready[slot], Some(&self.copy))?;
        Ok(())
    }

    fn compute_acquire(&mut self, gpu: &Gpu, slot: usize, rec: Option<usize>) -> R<bool> {
        let compute = compute_stream(gpu)?;
        let starved = !gpu.hip.event_query(&self.ready[slot])?;
        gpu.hip.stream_wait_event(compute, &self.ready[slot])?;
        if let Some(rec) = rec {
            gpu.hip.event_record(
                self.timeline_event(TimelineEvent::ComputeBegin(rec))?,
                Some(compute),
            )?;
        }
        Ok(starved)
    }

    fn compute_release(&mut self, gpu: &Gpu, slot: usize, rec: Option<usize>) -> R<()> {
        let compute = compute_stream(gpu)?;
        if let Some(rec) = rec {
            gpu.hip.event_record(
                self.timeline_event(TimelineEvent::ComputeEnd(rec))?,
                Some(compute),
            )?;
        }
        gpu.hip.event_record(&self.free[slot], Some(compute))?;
        Ok(())
    }

    fn mark_base(&mut self, gpu: &Gpu) -> R<()> {
        if let Some(base) = self.base.as_ref() {
            gpu.hip.event_record(base, Some(&self.copy))?;
        }
        Ok(())
    }

    fn mark_layer(&mut self, gpu: &Gpu, mark: usize, end: bool) -> R<()> {
        let event = self.timeline_event(if end {
            TimelineEvent::LayerEnd(mark)
        } else {
            TimelineEvent::LayerBegin(mark)
        })?;
        gpu.hip.event_record(event, Some(compute_stream(gpu)?))?;
        Ok(())
    }

    fn drain(&mut self, gpu: &Gpu) -> R<()> {
        gpu.hip.stream_synchronize(&self.copy)?;
        gpu.hip.stream_synchronize(compute_stream(gpu)?)?;
        Ok(())
    }

    fn elapsed_ms(&self, gpu: &Gpu, from: TimelineEvent, to: TimelineEvent) -> R<f64> {
        Ok(f64::from(gpu.hip.event_elapsed_ms(
            self.timeline_event(from)?,
            self.timeline_event(to)?,
        )?))
    }
}

/// D's dispatch input for one leased page. Everything borrows the ring, so it
/// cannot outlive the lease; the page is never freed through it.
pub(crate) struct PanelBindingParts<'a> {
    pub(crate) ticket: PanelTicket,
    /// Ascending global expert ids the page holds.
    pub(crate) experts: &'a [usize],
    /// Generation epoch: differs whenever the page is refilled at this address.
    pub(crate) epoch: u64,
    pub(crate) layer: usize,
    pub(crate) group: u32,
    pub(crate) codec: PanelCodec,
    /// The page payload allocation (tables point into it).
    pub(crate) payload: &'a GpuTensor,
    /// Bytes of `payload` holding expert data (the tail pad follows).
    pub(crate) payload_used: usize,
    /// 512-entry (pool `num_experts`) device pointer tables, global expert id
    /// indexed, rebased onto `payload`; entries outside `experts` are zero.
    pub(crate) gate_up_table: &'a GpuTensor,
    pub(crate) down_table: &'a GpuTensor,
    /// Source provenance: the layer's pinned host owner base address.
    pub(crate) source_host_base: usize,
}

/// A page checked out for compute; borrows the ring exclusively until dropped.
/// Drop it, then [`ExpertPanelRing::retire`] the ticket.
pub(crate) struct ExpertPanelLease<'a> {
    ring: &'a ExpertPanelRing,
    ticket: PanelTicket,
}

impl<'a> ExpertPanelLease<'a> {
    pub(crate) fn ticket(&self) -> PanelTicket {
        self.ticket
    }

    pub(crate) fn parts(&self) -> PanelBindingParts<'_> {
        let core = &self.ring.core;
        let page = &core.pages[self.ticket.slot];
        let slot = self.ticket.slot;
        PanelBindingParts {
            ticket: self.ticket,
            experts: &page.experts,
            epoch: page.epoch,
            layer: page.layer,
            group: page.group,
            codec: page.codec,
            payload: &core.backend.payloads[slot],
            payload_used: page.used,
            gate_up_table: &core.backend.gate_ups[slot],
            down_table: &core.backend.downs[slot],
            source_host_base: page.host_base,
        }
    }
}

/// The expert panel ring: `enqueue / acquire / retire / drain` over the page
/// pool, plus the window/layer/timeline controls of S1's schedule.
pub(crate) struct ExpertPanelRing {
    core: RingCore<HipPanelBackend>,
}

impl ExpertPanelRing {
    /// Allocate the pool (charged by [`PanelPoolSpec::reserve`]).
    pub(crate) fn new(gpu: &mut Gpu, spec: PanelPoolSpec) -> R<Self> {
        spec.validate()?;
        let backend = HipPanelBackend::new(gpu, &spec)?;
        Ok(Self {
            core: RingCore::new(spec, backend)?,
        })
    }

    pub(crate) fn spec(&self) -> &PanelPoolSpec {
        &self.core.spec
    }

    /// Start a window and prefetch the first host layer's groups (no compute
    /// layer yet). Returns the number of groups enqueued.
    pub(crate) fn prime<S: RingSchedule>(&mut self, gpu: &Gpu, schedule: &mut S) -> R<usize> {
        schedule.prime(&mut self.core, gpu)
    }

    /// Compute reached host layer `layer`.
    pub(crate) fn enter_layer<S: RingSchedule>(
        &mut self,
        gpu: &Gpu,
        schedule: &mut S,
        layer: usize,
    ) -> R<usize> {
        schedule.enter_layer(&mut self.core, gpu, layer)
    }

    /// Enqueue whatever the schedule allows (call after each retire).
    pub(crate) fn pump<S: RingSchedule>(&mut self, gpu: &Gpu, schedule: &mut S) -> R<usize> {
        schedule.pump(&mut self.core, gpu)
    }

    /// Ticket of current-layer `group`, enqueueing it if the schedule does so.
    pub(crate) fn require<S: RingSchedule>(
        &mut self,
        gpu: &Gpu,
        schedule: &mut S,
        layer: usize,
        group: u32,
    ) -> R<PanelTicket> {
        schedule.require(&mut self.core, gpu, layer, group)
    }

    pub(crate) fn begin_window(&mut self, gpu: &Gpu) -> R<()> {
        self.core.begin_window(gpu)
    }

    pub(crate) fn set_layers(&mut self, current: Option<usize>, next: Option<usize>) -> R<()> {
        self.core.set_layers(current, next)
    }

    pub(crate) fn try_enqueue(
        &mut self,
        gpu: &Gpu,
        group: &ExpertGroupPlan,
        source: &LayerExpertSources,
    ) -> R<Option<PanelTicket>> {
        self.core.try_enqueue(gpu, group, source)
    }

    pub(crate) fn enqueue(
        &mut self,
        gpu: &Gpu,
        group: &ExpertGroupPlan,
        source: &LayerExpertSources,
    ) -> R<PanelTicket> {
        self.core.enqueue(gpu, group, source)
    }

    pub(crate) fn acquire<'a>(
        &'a mut self,
        gpu: &Gpu,
        ticket: PanelTicket,
    ) -> R<ExpertPanelLease<'a>> {
        self.core.acquire(gpu, ticket)?;
        self.core.leased(ticket)?;
        Ok(ExpertPanelLease { ring: self, ticket })
    }

    pub(crate) fn retire(&mut self, gpu: &Gpu, ticket: PanelTicket) -> R<()> {
        self.core.retire(gpu, ticket)
    }

    pub(crate) fn drain(&mut self, gpu: &Gpu) -> R<()> {
        self.core.drain(gpu)
    }

    pub(crate) fn mark_layer_begin(&mut self, gpu: &Gpu, layer: usize) -> R<()> {
        self.core.mark_layer_begin(gpu, layer)
    }

    pub(crate) fn mark_layer_end(&mut self, gpu: &Gpu, layer: usize) -> R<()> {
        self.core.mark_layer_end(gpu, layer)
    }

    pub(crate) fn stats(&self) -> PanelStats {
        self.core.stats()
    }

    pub(crate) fn timeline(&self, gpu: &Gpu) -> R<PanelTimeline> {
        self.core.timeline(gpu)
    }

    /// Drain, then free the pool. A failed drain leaks the device and pinned
    /// memory rather than freeing what the device may still use.
    pub(crate) fn free_gpu(mut self, gpu: &mut Gpu) -> R<()> {
        // A failed drain returns here with `self` still dirty: its `Drop`
        // leaks the pool and every retained source instead of freeing them.
        self.core.drain(&*gpu)?;
        let backend = self.core.take_backend()?;
        backend.release(gpu)
    }
}

// ── diagnostic: payload / table identity self-test ───────────────────

/// What a downloaded reader image was compared against.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ImageCheck {
    pub(crate) payload_bytes: u64,
    pub(crate) table_entries: u64,
    pub(crate) pad_bytes: u64,
}

/// Check a page image read back from the device against the pinned host
/// sources: `image` is the page payload `[0, used + ARENA_TAIL_PAD_BYTES)`;
/// every group expert's gate/up and down bytes (located through its table
/// entry, rebased onto `page_base`) must equal the source extent byte for
/// byte, the tail pad must be zero, group entries must point inside the
/// payload and every other table entry must be zero.
fn verify_page_image(
    image: &[u8],
    used: usize,
    page_base: usize,
    experts: &[usize],
    sources: &LayerExpertSources,
    gate_up_table: &[u64],
    down_table: &[u64],
) -> R<ImageCheck> {
    let layer = sources.layer();
    let fail = |what: String| invalid(format!("panel image layer {layer}: {what}"));
    let need = used
        .checked_add(ARENA_TAIL_PAD_BYTES)
        .ok_or_else(|| fail("payload length overflows".into()))?;
    if image.len() < need {
        return Err(fail(format!(
            "image of {} bytes is shorter than {need}",
            image.len()
        )));
    }
    if gate_up_table.len() != sources.num_experts() || down_table.len() != sources.num_experts() {
        return Err(fail(format!(
            "tables of {}/{} entries, expected {}",
            gate_up_table.len(),
            down_table.len(),
            sources.num_experts()
        )));
    }
    let mut check = ImageCheck::default();
    if let Some(at) = image[used..need].iter().position(|&b| b != 0) {
        return Err(fail(format!("tail pad byte {at} is nonzero")));
    }
    check.pad_bytes = ARENA_TAIL_PAD_BYTES as u64;
    let mut in_group = vec![false; sources.num_experts()];
    for &id in experts {
        *in_group
            .get_mut(id)
            .ok_or_else(|| fail(format!("group expert {id} is outside the table")))? = true;
    }
    for (name, table) in [("gate_up", gate_up_table), ("down", down_table)] {
        for (id, &entry) in table.iter().enumerate() {
            if !in_group[id] && entry != 0 {
                return Err(fail(format!(
                    "{name} table entry {id} is nonzero outside the group"
                )));
            }
            check.table_entries += 1;
        }
    }
    for &id in experts {
        let expert = sources
            .expert(id)
            .ok_or_else(|| fail(format!("expert {id} is not held")))?;
        for (name, table, extent) in [
            ("gate_up", gate_up_table, expert.gate_up),
            ("down", down_table, expert.down),
        ] {
            let offset = usize::try_from(table[id])
                .ok()
                .and_then(|entry| entry.checked_sub(page_base))
                .ok_or_else(|| {
                    fail(format!(
                        "expert {id} {name} entry {:#x} is below the page",
                        table[id]
                    ))
                })?;
            let end = offset
                .checked_add(extent.len)
                .filter(|end| *end <= used)
                .ok_or_else(|| {
                    fail(format!(
                        "expert {id} {name} entry names [{offset}, +{}) outside the {used}-byte payload",
                        extent.len
                    ))
                })?;
            // SAFETY: the extent was bounds-checked against the pinned host
            // range of `sources`, which its owner keeps alive and unmodified.
            let source =
                unsafe { std::slice::from_raw_parts(extent.addr as *const u8, extent.len) };
            if let Some(at) = source
                .iter()
                .zip(&image[offset..end])
                .position(|(a, b)| a != b)
            {
                return Err(fail(format!(
                    "expert {id} {name} byte {at} differs from the pinned source"
                )));
            }
            check.payload_bytes += extent.len as u64;
        }
    }
    Ok(check)
}

/// What [`primitive_selftest`] proved on the device.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PrimitiveSelftestReport {
    pub(crate) layers: usize,
    pub(crate) groups_checked: u64,
    /// Expert gate/up + down bytes compared against the pinned sources.
    pub(crate) payload_bytes_compared: u64,
    pub(crate) table_entries_checked: u64,
    pub(crate) pad_bytes_checked: u64,
    /// Page refilled at the same address while the previous epoch's (delayed)
    /// reader was still queued, and that reader's bytes still matched.
    pub(crate) delayed_reader_reuses: u64,
    pub(crate) stale_ticket_rejections: u64,
    pub(crate) distinct_addresses: usize,
    /// Addresses that held at least two distinct epochs, each verified.
    pub(crate) repeated_address_epochs: u64,
    pub(crate) min_epochs_per_address: u64,
    pub(crate) max_epochs_per_address: u64,
    pub(crate) submitted_payload_bytes: u64,
}

fn selftest_epoch_summary(seen: &HashMap<usize, Vec<u64>>) -> PrimitiveSelftestReport {
    PrimitiveSelftestReport {
        distinct_addresses: seen.len(),
        repeated_address_epochs: seen.values().filter(|e| e.len() >= 2).count() as u64,
        min_epochs_per_address: seen.values().map(|e| e.len() as u64).min().unwrap_or(0),
        max_epochs_per_address: seen.values().map(|e| e.len() as u64).max().unwrap_or(0),
        ..PrimitiveSelftestReport::default()
    }
}

pub(crate) const SELFTEST_DELAY_BYTES: usize = 64 << 20;
pub(crate) const SELFTEST_DELAY_BUFFERS: usize = 2;

/// Device bytes [`primitive_selftest`] allocates for its duration: one scratch
/// holding a page image plus both pointer tables, and two delay buffers. The
/// ONE authority: the allocation and every census charge read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SelftestReserve {
    /// `page_bytes + 2 * num_experts * 8`.
    pub(crate) scratch_bytes: usize,
    pub(crate) delay_buffers: usize,
    pub(crate) delay_bytes_each: usize,
    pub(crate) total_device_bytes: usize,
}

impl PanelPoolSpec {
    /// The diagnostic's exact device allocation for this pool geometry,
    /// `None` on overflow. Charged only on the diagnostic path.
    pub(crate) fn selftest_reserve(&self) -> Option<SelftestReserve> {
        let tables = self
            .num_experts
            .checked_mul(std::mem::size_of::<u64>())?
            .checked_mul(2)?;
        let scratch_bytes = self.page_bytes()?.checked_add(tables)?;
        let delay_total = SELFTEST_DELAY_BYTES.checked_mul(SELFTEST_DELAY_BUFFERS)?;
        Some(SelftestReserve {
            scratch_bytes,
            delay_buffers: SELFTEST_DELAY_BUFFERS,
            delay_bytes_each: SELFTEST_DELAY_BYTES,
            total_device_bytes: scratch_bytes.checked_add(delay_total)?,
        })
    }
}

/// Allocator of the diagnostic buffers: HIP in production, a counting model in
/// CPU tests.
trait DiagMem {
    type Buf;
    fn alloc(&mut self, bytes: usize) -> R<Self::Buf>;
    fn free(&mut self, buf: Self::Buf) -> R<()>;
}

/// Plain device allocations, outside the shared `GpuPool`: charged equals
/// allocated, nothing is cached after release. `DeviceBuffer` has no `Drop`,
/// so [`release_diag_after_drain`] is the only thing that frees them.
struct HipDiagMem<'a>(&'a Gpu);

impl DiagMem for HipDiagMem<'_> {
    type Buf = hip_bridge::DeviceBuffer;

    fn alloc(&mut self, bytes: usize) -> R<Self::Buf> {
        self.0.bind_thread()?;
        Ok(self.0.hip.malloc(bytes)?)
    }

    fn free(&mut self, buf: Self::Buf) -> R<()> {
        self.0.bind_thread()?;
        Ok(self.0.hip.free(buf)?)
    }
}

struct DiagBuffers<B> {
    scratch: B,
    delay: Vec<B>,
}

/// Allocate exactly the reserve's buffers (scratch, then the delay buffers). A
/// failure frees whatever this call already allocated.
fn alloc_diag<M: DiagMem>(mem: &mut M, reserve: &SelftestReserve) -> R<DiagBuffers<M::Buf>> {
    let scratch = mem.alloc(reserve.scratch_bytes)?;
    let mut delay = Vec::with_capacity(reserve.delay_buffers);
    for _ in 0..reserve.delay_buffers {
        match mem.alloc(reserve.delay_bytes_each) {
            Ok(buffer) => delay.push(buffer),
            Err(error) => {
                let _ = mem.free(scratch);
                for buffer in delay {
                    let _ = mem.free(buffer);
                }
                return Err(error);
            }
        }
    }
    Ok(DiagBuffers { scratch, delay })
}

/// Release the diagnostic buffers only if the streams are proven idle
/// (`drained` succeeded). A failed drain returns its error and LEAKS the
/// buffers: the device may still read or write them. Every buffer is freed even
/// when an earlier free failed; the first free error is returned.
fn release_diag_after_drain<M: DiagMem>(
    mem: &mut M,
    buffers: DiagBuffers<M::Buf>,
    drained: R<()>,
) -> R<()> {
    if let Err(error) = drained {
        std::mem::forget(buffers);
        return Err(error);
    }
    let mut first: R<()> = Ok(());
    for buffer in std::iter::once(buffers.scratch).chain(buffers.delay) {
        if let Err(error) = mem.free(buffer) {
            if first.is_ok() {
                first = Err(error);
            }
        }
    }
    first
}

struct PendingReader {
    layer_index: usize,
    experts: SmallVec<[usize; MAX_GROUP_EXPERTS]>,
    used: usize,
    base: usize,
}

/// Real-device diagnostic of the transport primitives, driven through the
/// SAME `prime / enter_layer / require / acquire / retire / pump` path the
/// layer-major prefill uses, on the first `layers` host layers of `schedule`
/// (their real [`LayerExpertSources`]).
///
/// For every group the compute stream first runs a long delay, then a D2D
/// reader copies the page payload, tail pad and both pointer tables into a
/// scratch buffer; the page is retired and the schedule pumped, which refills
/// the freed page at the same address while that reader is still queued. Only
/// then are the streams synchronized, the scratch downloaded and compared
/// byte for byte with the pinned host sources ([`verify_page_image`]). A
/// transport that let the refill start before the reader finished, or whose
/// tables/pad/payload were not ready at acquire, fails here. The fixture must
/// hold at least two full pool sweeps (`2 * pages` groups) so every address
/// is reused; otherwise this is an error, not a pass.
///
/// Ends with the ring drained; the schedule must be `prime`d again before
/// real use. Diagnostic only: allocates one page-sized scratch and two
/// delay buffers for its duration.
pub(crate) fn primitive_selftest(
    gpu: &mut Gpu,
    ring: &mut ExpertPanelRing,
    schedule: &mut PanelSchedule,
    layers: usize,
    delay_repeats: usize,
) -> R<PrimitiveSelftestReport> {
    let spec = *ring.spec();
    let reserve = spec
        .reserve()
        .ok_or_else(|| invalid("panel selftest: pool byte size overflows"))?;
    if layers == 0 || layers > schedule.layers.len() {
        return Err(invalid(format!(
            "panel selftest: {layers} layers requested of {}",
            schedule.layers.len()
        )));
    }
    let groups: usize = schedule.layers[..layers]
        .iter()
        .map(|l| l.groups.len())
        .sum();
    if groups < 2 * reserve.pages {
        return Err(invalid(format!(
            "panel selftest: {groups} groups cannot reuse each of {} pages twice; need at least {} (use more layers)",
            reserve.pages,
            2 * reserve.pages
        )));
    }
    let diag = spec
        .selftest_reserve()
        .ok_or_else(|| invalid("panel selftest: diagnostic reserve overflows"))?;
    let table_bytes = spec.num_experts * std::mem::size_of::<u64>();
    let buffers = alloc_diag(&mut HipDiagMem(&*gpu), &diag)?;
    let result = run_selftest(
        gpu,
        ring,
        schedule,
        layers,
        delay_repeats,
        &buffers.scratch,
        &buffers.delay,
        table_bytes,
        reserve.page_bytes,
    );
    // Streams must be idle before the diagnostic buffers are released; a
    // failed drain leaks them (the device may still be using them).
    let drained = ring.drain(&*gpu);
    let released = release_diag_after_drain(&mut HipDiagMem(&*gpu), buffers, drained);
    let report = result?;
    released?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn run_selftest(
    gpu: &Gpu,
    ring: &mut ExpertPanelRing,
    schedule: &mut PanelSchedule,
    layers: usize,
    delay_repeats: usize,
    scratch: &hip_bridge::DeviceBuffer,
    delay: &[hip_bridge::DeviceBuffer],
    table_bytes: usize,
    page_bytes: usize,
) -> R<PrimitiveSelftestReport> {
    let mut report = PrimitiveSelftestReport {
        layers,
        ..PrimitiveSelftestReport::default()
    };
    let mut seen: HashMap<usize, Vec<u64>> = HashMap::new();
    let mut image = vec![0u8; page_bytes + 2 * table_bytes];
    let mut pending: Option<PendingReader> = None;

    // Verify the reader issued for the previous group: sync, download, compare.
    let verify = |ring: &mut ExpertPanelRing,
                  schedule: &PanelSchedule,
                  image: &mut Vec<u8>,
                  report: &mut PrimitiveSelftestReport,
                  reader: PendingReader|
     -> R<()> {
        ring.core.backend.drain(gpu)?;
        gpu.hip.memcpy_dtoh(image.as_mut_slice(), scratch)?;
        let decode = |bytes: &[u8]| -> Vec<u64> {
            bytes
                .chunks_exact(std::mem::size_of::<u64>())
                .map(|chunk| u64::from_ne_bytes(chunk.try_into().expect("8-byte chunk")))
                .collect()
        };
        let gate_up = decode(&image[page_bytes..page_bytes + table_bytes]);
        let down = decode(&image[page_bytes + table_bytes..page_bytes + 2 * table_bytes]);
        let sources = &schedule.layers[reader.layer_index].sources;
        let check = verify_page_image(
            &image[..page_bytes],
            reader.used,
            reader.base,
            &reader.experts,
            sources,
            &gate_up,
            &down,
        )?;
        report.groups_checked += 1;
        report.payload_bytes_compared += check.payload_bytes;
        report.table_entries_checked += check.table_entries;
        report.pad_bytes_checked += check.pad_bytes;
        Ok(())
    };

    ring.prime(gpu, schedule)?;
    for layer_index in 0..layers {
        let layer = schedule.layers[layer_index].sources.layer();
        ring.enter_layer(gpu, schedule, layer)?;
        let group_count = schedule.layers[layer_index].groups.len();
        for group in 0..group_count {
            let ticket = ring.require(gpu, schedule, layer, group as u32)?;
            if let Some(reader) = pending.take() {
                verify(ring, schedule, &mut image, &mut report, reader)?;
            }
            let reader = {
                let lease = ring.acquire(gpu, ticket)?;
                let parts = lease.parts();
                let base = parts.payload.buf.as_ptr() as usize;
                seen.entry(base).or_default().push(parts.epoch);
                let compute = compute_stream(gpu)?;
                for _ in 0..delay_repeats {
                    gpu.hip.memcpy_dtod_async_at(
                        &delay[1],
                        0,
                        &delay[0],
                        0,
                        SELFTEST_DELAY_BYTES,
                        compute,
                    )?;
                    gpu.hip.memcpy_dtod_async_at(
                        &delay[0],
                        0,
                        &delay[1],
                        0,
                        SELFTEST_DELAY_BYTES,
                        compute,
                    )?;
                }
                let image_bytes = parts.payload_used + ARENA_TAIL_PAD_BYTES;
                gpu.hip.memcpy_dtod_async_at(
                    scratch,
                    0,
                    &parts.payload.buf,
                    0,
                    image_bytes,
                    compute,
                )?;
                gpu.hip.memcpy_dtod_async_at(
                    scratch,
                    page_bytes,
                    &parts.gate_up_table.buf,
                    0,
                    table_bytes,
                    compute,
                )?;
                gpu.hip.memcpy_dtod_async_at(
                    scratch,
                    page_bytes + table_bytes,
                    &parts.down_table.buf,
                    0,
                    table_bytes,
                    compute,
                )?;
                PendingReader {
                    layer_index,
                    experts: SmallVec::from_slice(parts.experts),
                    used: parts.payload_used,
                    base,
                }
            };
            ring.retire(gpu, ticket)?;
            if ring.acquire(gpu, ticket).is_ok() {
                return Err(invalid(format!(
                    "panel selftest: retired ticket (slot {}, epoch {}) was acquired again",
                    ticket.slot, ticket.epoch
                )));
            }
            if ring.core.is_quarantined() {
                return Err(invalid(
                    "panel selftest: a refused stale ticket quarantined the ring",
                ));
            }
            report.stale_ticket_rejections += 1;
            // Refill the freed page while the delayed reader is still queued.
            ring.pump(gpu, schedule)?;
            let page = &ring.core.pages[ticket.slot];
            if page.epoch != ticket.epoch && page.state != PageState::Free {
                report.delayed_reader_reuses += 1;
            }
            pending = Some(reader);
        }
    }
    if let Some(reader) = pending.take() {
        verify(ring, schedule, &mut image, &mut report, reader)?;
    }
    let epochs = selftest_epoch_summary(&seen);
    report.distinct_addresses = epochs.distinct_addresses;
    report.repeated_address_epochs = epochs.repeated_address_epochs;
    report.min_epochs_per_address = epochs.min_epochs_per_address;
    report.max_epochs_per_address = epochs.max_epochs_per_address;
    report.submitted_payload_bytes = ring.stats().submitted_payload_bytes;
    if report.delayed_reader_reuses == 0 {
        return Err(invalid(
            "panel selftest: no page was refilled while a delayed reader was queued; the delayed-reader case was not exercised",
        ));
    }
    Ok(report)
}

#[cfg(test)]
mod slowlink_ring {
    use super::*;
    use std::collections::HashMap;

    fn counted() -> (RingCore<Mock>, Arc<std::sync::atomic::AtomicUsize>) {
        let mock = Mock::new(1);
        let drops = Arc::clone(&mock.dropped);
        (RingCore::new(spec(1, 0), mock).unwrap(), drops)
    }

    fn backend_drops(drops: &Arc<std::sync::atomic::AtomicUsize>) -> usize {
        drops.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// One group enqueued, read and retired: logically free, GPU unproven.
    fn used_ring(owner: &Arc<u8>) -> (RingCore<Mock>, Arc<std::sync::atomic::AtomicUsize>) {
        let any: Arc<dyn Any + Send + Sync> = owner.clone();
        let s = sources_with(0, any, |_| 1);
        let (mut ring, drops) = counted();
        ring.begin_window(&()).unwrap();
        ring.set_layers(Some(0), None).unwrap();
        let t = ring.enqueue(&(), &group(&s, 0), &s).unwrap();
        ring.acquire(&(), t).unwrap();
        ring.retire(&(), t).unwrap();
        (ring, drops)
    }

    #[test]
    fn slowlink_ring_implicit_drop_of_an_unused_ring_releases_the_backend() {
        let (ring, drops) = counted();
        drop(ring);
        assert_eq!(backend_drops(&drops), 1);
    }

    #[test]
    fn slowlink_ring_implicit_drop_after_undrained_work_leaks_backend_and_sources() {
        let owner = Arc::new(0u8);
        let (ring, drops) = used_ring(&owner);
        // Retired and logically free, but never proven complete on the GPU.
        assert_eq!(
            Arc::strong_count(&owner),
            2,
            "the page retains the source owner"
        );
        drop(ring);
        assert_eq!(
            backend_drops(&drops),
            0,
            "the pool/tables/staging were leaked"
        );
        assert_eq!(
            Arc::strong_count(&owner),
            2,
            "the retained source owner was leaked, not released"
        );
        // Even a window start (a queued event record) makes the ring dirty.
        let (mut started, drops) = counted();
        started.begin_window(&()).unwrap();
        drop(started);
        assert_eq!(backend_drops(&drops), 0);
    }

    #[test]
    fn slowlink_ring_implicit_drop_after_successful_drain_releases_everything() {
        let owner = Arc::new(0u8);
        let (mut ring, drops) = used_ring(&owner);
        ring.drain(&()).unwrap();
        drop(ring);
        assert_eq!(backend_drops(&drops), 1);
        assert_eq!(
            Arc::strong_count(&owner),
            1,
            "drain released the retained owner"
        );
        // Work after a drain makes it dirty again.
        let (mut again, drops) = counted();
        again.begin_window(&()).unwrap();
        again.drain(&()).unwrap();
        again.begin_window(&()).unwrap();
        drop(again);
        assert_eq!(backend_drops(&drops), 0);
    }

    #[test]
    fn slowlink_ring_implicit_drop_after_failed_drain_or_failed_enqueue_leaks() {
        let owner = Arc::new(0u8);
        let (mut ring, drops) = used_ring(&owner);
        ring.backend.fail_drain = true;
        assert!(ring.drain(&()).is_err());
        drop(ring);
        assert_eq!(backend_drops(&drops), 0);
        assert_eq!(Arc::strong_count(&owner), 2);
        // A device failure mid-enqueue (quarantine) leaks too.
        let any: Arc<dyn Any + Send + Sync> = owner.clone();
        let s = sources_with(0, any, |_| 1);
        let (mut ring, drops) = counted();
        ring.begin_window(&()).unwrap();
        ring.set_layers(Some(0), None).unwrap();
        ring.backend.fail_at = Some(ring.backend.ops.len() + 2);
        assert!(ring.enqueue(&(), &group(&s, 0), &s).is_err());
        assert!(ring.is_quarantined());
        drop(ring);
        assert_eq!(backend_drops(&drops), 0);
    }

    #[test]
    fn slowlink_ring_backend_is_taken_only_after_a_completed_drain() {
        let owner = Arc::new(0u8);
        let (mut ring, drops) = used_ring(&owner);
        assert!(
            ring.take_backend().is_err(),
            "dirty ring cannot hand out its backend"
        );
        ring.drain(&()).unwrap();
        let backend = ring.take_backend().unwrap();
        assert!(ring.take_backend().is_err(), "taken once");
        drop(ring);
        assert_eq!(
            backend_drops(&drops),
            0,
            "the core never drops a taken backend"
        );
        drop(backend);
        assert_eq!(
            backend_drops(&drops),
            1,
            "the new owner drops it exactly once"
        );
    }

    /// Counting model of the diagnostic allocator: every live allocation is
    /// tracked by id, so a leak (never freed) is observable.
    #[derive(Default)]
    struct DiagModel {
        next: u64,
        live: Vec<(u64, usize)>,
        peak: usize,
        allocs: Vec<usize>,
        frees: usize,
        fail_alloc_at: Option<usize>,
        fail_free_at: Option<usize>,
    }

    struct ModelBuf(u64);

    impl DiagMem for DiagModel {
        type Buf = ModelBuf;

        fn alloc(&mut self, bytes: usize) -> R<ModelBuf> {
            if self.fail_alloc_at == Some(self.allocs.len()) {
                self.fail_alloc_at = None;
                return Err(invalid("injected hipMalloc failure"));
            }
            self.next += 1;
            self.live.push((self.next, bytes));
            self.allocs.push(bytes);
            self.peak = self.peak.max(self.live.iter().map(|(_, b)| b).sum());
            Ok(ModelBuf(self.next))
        }

        fn free(&mut self, buf: ModelBuf) -> R<()> {
            if self.fail_free_at == Some(self.frees) {
                self.fail_free_at = None;
                self.frees += 1;
                self.live.retain(|(id, _)| *id != buf.0);
                return Err(invalid("injected hipFree failure"));
            }
            self.frees += 1;
            let before = self.live.len();
            self.live.retain(|(id, _)| *id != buf.0);
            assert_eq!(self.live.len() + 1, before, "double free");
            Ok(())
        }
    }

    fn mq4_spec() -> PanelPoolSpec {
        PanelPoolSpec::uniform(2, 16, 32, 512, 1_740_800, 870_400, 0, 0).unwrap()
    }

    #[test]
    fn slowlink_ring_selftest_reserve_is_the_single_authority_for_the_diagnostic_bytes() {
        let spec = mq4_spec();
        let reserve = spec.selftest_reserve().unwrap();
        let page = spec.reserve().unwrap().page_bytes;
        assert_eq!(page, 83_595_264);
        // Scratch holds the page image plus BOTH 512-entry pointer tables.
        assert_eq!(reserve.scratch_bytes, page + 2 * 512 * 8);
        assert_eq!(reserve.scratch_bytes, 83_603_456);
        assert_eq!(
            (reserve.delay_buffers, reserve.delay_bytes_each),
            (2, 64 << 20)
        );
        assert_eq!(reserve.total_device_bytes, 217_821_184);
        // The allocation is exactly the reserve: same sizes, same total.
        let mut model = DiagModel::default();
        let buffers = alloc_diag(&mut model, &reserve).unwrap();
        assert_eq!(
            model.allocs,
            vec![reserve.scratch_bytes, 64 << 20, 64 << 20]
        );
        assert_eq!(
            model.allocs.iter().sum::<usize>(),
            reserve.total_device_bytes
        );
        assert_eq!(model.peak, reserve.total_device_bytes);
        assert_eq!(buffers.delay.len(), reserve.delay_buffers);
        release_diag_after_drain(&mut model, buffers, Ok(())).unwrap();
        // A small geometry and an overflowing one.
        let small = spec_small_for_reserve();
        let r = small.selftest_reserve().unwrap();
        assert_eq!(
            r.total_device_bytes,
            small.page_bytes().unwrap() + 2 * small.num_experts * 8 + 2 * (64 << 20)
        );
        let overflow = PanelPoolSpec {
            num_experts: usize::MAX / 8,
            ..small
        };
        assert_eq!(overflow.selftest_reserve(), None);
        let huge = PanelPoolSpec {
            payload_bytes: usize::MAX - 8,
            ..small
        };
        assert_eq!(huge.selftest_reserve(), None);
    }

    fn spec_small_for_reserve() -> PanelPoolSpec {
        spec(2, 3)
    }

    #[test]
    fn slowlink_ring_selftest_three_back_to_back_cases_never_retain_or_exceed_the_reserve() {
        let reserve = mq4_spec().selftest_reserve().unwrap();
        let mut model = DiagModel::default();
        for run in 1..=3usize {
            let buffers = alloc_diag(&mut model, &reserve).unwrap();
            assert_eq!(
                model.live.iter().map(|(_, b)| b).sum::<usize>(),
                reserve.total_device_bytes,
                "run {run}: exactly the reserve is live"
            );
            release_diag_after_drain(&mut model, buffers, Ok(())).unwrap();
            assert!(model.live.is_empty(), "run {run}: nothing retained");
        }
        assert_eq!(model.peak, reserve.total_device_bytes, "peak is 1x, not 3x");
        assert_eq!(model.allocs.len(), 9);
        assert_eq!(model.frees, 9);
    }

    #[test]
    fn slowlink_ring_selftest_partial_allocation_failure_frees_what_it_allocated() {
        let reserve = mq4_spec().selftest_reserve().unwrap();
        for fail_at in [0usize, 1, 2] {
            let mut model = DiagModel {
                fail_alloc_at: Some(fail_at),
                ..DiagModel::default()
            };
            assert!(alloc_diag(&mut model, &reserve).is_err());
            assert!(
                model.live.is_empty(),
                "failure at allocation {fail_at} leaked"
            );
            assert_eq!(model.frees, fail_at);
            // The next attempt starts clean and succeeds.
            let buffers = alloc_diag(&mut model, &reserve).unwrap();
            release_diag_after_drain(&mut model, buffers, Ok(())).unwrap();
            assert!(model.live.is_empty());
        }
    }

    #[test]
    fn slowlink_ring_selftest_failed_drain_leaks_the_buffers_instead_of_freeing_them() {
        let reserve = mq4_spec().selftest_reserve().unwrap();
        let mut model = DiagModel::default();
        let buffers = alloc_diag(&mut model, &reserve).unwrap();
        let error = release_diag_after_drain(&mut model, buffers, Err(invalid("drain failed")))
            .unwrap_err()
            .to_string();
        assert!(error.contains("drain failed"), "{error}");
        assert_eq!(
            model.frees, 0,
            "nothing is freed while the device may use it"
        );
        assert_eq!(model.live.len(), 3, "all three buffers stay owned (leaked)");
    }

    #[test]
    fn slowlink_ring_selftest_free_failure_still_frees_every_other_buffer() {
        let reserve = mq4_spec().selftest_reserve().unwrap();
        let mut model = DiagModel {
            fail_free_at: Some(0),
            ..DiagModel::default()
        };
        let buffers = alloc_diag(&mut model, &reserve).unwrap();
        let error = release_diag_after_drain(&mut model, buffers, Ok(()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("hipFree"), "{error}");
        assert_eq!(model.frees, 3, "all three frees were attempted");
        assert!(model.live.is_empty());
    }

    // ── real E512 physical kinds (CPU logical binding proof) ─────────────
    //
    // Scalar metadata of the real artifact (`d2-default-gates/layout-proof.json`,
    // header sha256 7a23ec6f0b4aa8347b1efea289519f6d0a030f1197a28ecd18f3f65f45407162:
    // 512 experts, 48 aggregate gate/up (QT44) + 48 aggregate down (QT53) tensors, no
    // arena or slab tensors) and of the slab layout: per-expert gate/up G, down D,
    // interleaved stride P = G + D. No memory is allocated: every address below is
    // a fabricated 64-bit value that the layout and tables only do arithmetic on.
    const REAL_E: usize = 512;
    const REAL_G: usize = 1_740_800;
    const REAL_D: usize = 870_400;
    const REAL_P: usize = 2_611_200;
    /// 32 experts x P: the payload bound of one group page.
    const REAL_GROUP_PAYLOAD: usize = 83_558_400;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Proj {
        GateUp,
        Down,
    }

    /// The two physical placements of the same logical experts.
    #[derive(Clone, Copy)]
    enum RealKind {
        /// Two independent allocations: gate/up `A_G + e*G`, down `A_D + e*D`.
        Aggregate {
            cpu_g: usize,
            alias_g: usize,
            cpu_d: usize,
            alias_d: usize,
        },
        /// One interleaved arena: gate/up `S + e*P`, down `S + e*P + G`.
        Slab { cpu: usize, alias: usize },
    }

    impl RealKind {
        fn aggregate() -> Self {
            Self::Aggregate {
                cpu_g: 0x7000_0000_0000,
                alias_g: 0x7a00_0000_0000,
                cpu_d: 0x7200_0000_1000,
                alias_d: 0x7c00_0000_1000,
            }
        }

        fn slab() -> Self {
            Self::Slab {
                cpu: 0x7000_0000_0000,
                alias: 0x7a00_0000_0000,
            }
        }

        /// The alias (device-visible) expert table the loader would hand over.
        fn alias_experts(&self) -> Vec<Option<AliasExpert>> {
            (0..REAL_E)
                .map(|e| {
                    let (gate_up_addr, down_addr) = match *self {
                        Self::Aggregate {
                            alias_g, alias_d, ..
                        } => (alias_g + e * REAL_G, alias_d + e * REAL_D),
                        Self::Slab { alias, .. } => {
                            (alias + e * REAL_P, alias + e * REAL_P + REAL_G)
                        }
                    };
                    Some(AliasExpert {
                        codec: PanelCodec(1),
                        gate_up_addr,
                        gate_up_len: REAL_G,
                        down_addr,
                        down_len: REAL_D,
                    })
                })
                .collect()
        }

        fn sources(&self, owner: Arc<dyn Any + Send + Sync>) -> R<LayerExpertSources> {
            let experts = self.alias_experts();
            // SAFETY: the sources are only used for address arithmetic here; the
            // mock backend (and these tests) never dereference a source address.
            unsafe {
                match *self {
                    Self::Aggregate {
                        cpu_g,
                        alias_g,
                        cpu_d,
                        alias_d,
                    } => LayerExpertSources::from_alias_split(
                        7,
                        (cpu_g, alias_g, REAL_E * REAL_G),
                        (cpu_d, alias_d, REAL_E * REAL_D),
                        &experts,
                        owner,
                    ),
                    Self::Slab { cpu, alias } => LayerExpertSources::from_alias(
                        7,
                        cpu,
                        alias,
                        REAL_E * REAL_P,
                        &experts,
                        owner,
                    ),
                }
            }
        }

        /// Decode a CPU source address to `(expert, projection, byte offset in
        /// the projection)` from the scalar metadata alone: independent of every
        /// table the layout code consumes.
        fn decode(&self, addr: usize) -> Option<(usize, Proj, usize)> {
            let within =
                |base: usize, bytes: usize| addr.checked_sub(base).filter(|rel| *rel < bytes);
            match *self {
                Self::Aggregate { cpu_g, cpu_d, .. } => {
                    if let Some(rel) = within(cpu_g, REAL_E * REAL_G) {
                        Some((rel / REAL_G, Proj::GateUp, rel % REAL_G))
                    } else {
                        within(cpu_d, REAL_E * REAL_D)
                            .map(|rel| (rel / REAL_D, Proj::Down, rel % REAL_D))
                    }
                }
                Self::Slab { cpu, .. } => within(cpu, REAL_E * REAL_P).map(|rel| {
                    let (e, r) = (rel / REAL_P, rel % REAL_P);
                    if r < REAL_G {
                        (e, Proj::GateUp, r)
                    } else {
                        (e, Proj::Down, r - REAL_G)
                    }
                }),
            }
        }
    }

    /// Prove, for the real E512 geometry of `kind`, that the page layout and the
    /// rebased tables bind every pointer to the same logical
    /// `(expert, projection, byte offset)` the source metadata names, for the first,
    /// second and last 32-expert groups. Physical pointers are never compared.
    fn prove_real_kind(kind: RealKind, expected_runs: usize) {
        let owner = Arc::new(0u8);
        let any: Arc<dyn Any + Send + Sync> = owner.clone();
        let sources = kind.sources(any).unwrap();
        assert_eq!(Arc::strong_count(&owner), 2, "the sources retain the owner");
        assert_eq!(sources.num_experts(), REAL_E);
        assert_eq!(
            sources.host_ranges().len(),
            if matches!(kind, RealKind::Aggregate { .. }) {
                2
            } else {
                1
            }
        );

        let spec = PanelPoolSpec::uniform(2, 16, 32, REAL_E, REAL_G, REAL_D, 0, 0).unwrap();
        let page_bytes = spec.page_bytes().unwrap();
        assert_eq!(page_bytes, 83_595_264);
        // Every page allocation is ~84 MB: nothing near the 1.3 GB slab range.
        assert!(page_bytes < 100 << 20);
        assert_eq!(32 * REAL_P, REAL_GROUP_PAYLOAD);
        assert_eq!(REAL_P, REAL_G + REAL_D);

        let groups = ExpertGroupPlan::partition(&sources, 32).unwrap();
        assert_eq!(groups.len(), 16);
        let page_base = 0x2_0000_0000usize;
        for index in [0usize, 1, 15] {
            let group = &groups[index];
            let first = index * 32;
            assert_eq!(
                group.experts.as_slice(),
                (first..first + 32).collect::<Vec<_>>().as_slice(),
                "group {index} is [{first}, {})",
                first + 32
            );
            let layout = plan_group_layout(&group.experts, &sources).unwrap();
            // Payload bound: exactly 32 experts of G + D, no holes, no padding.
            assert_eq!(layout.extent_bytes, REAL_GROUP_PAYLOAD);
            assert_eq!(layout.submitted_bytes(), REAL_GROUP_PAYLOAD);
            assert_eq!(layout.used, REAL_GROUP_PAYLOAD);
            assert!(layout.used + ARENA_TAIL_PAD_BYTES <= page_bytes);
            assert_eq!(layout.runs.len(), expected_runs, "group {index} runs");
            // Alignment guard: every run starts 256-aligned at both ends, and
            // each run's first and last byte decode inside the source ranges.
            for run in &layout.runs {
                assert_eq!(run.src % EXTENT_ALIGN, 0);
                assert_eq!(run.dst % EXTENT_ALIGN, 0);
                assert!(
                    kind.decode(run.src).is_some(),
                    "run start outside the sources"
                );
                assert!(
                    kind.decode(run.src + run.len - 1).is_some(),
                    "run end outside the sources"
                );
            }
            // Poisoned tables: every outside entry must come out zero, every group
            // entry must be a real (non-poison, non-zero) pointer.
            let mut gate_up = vec![u64::MAX; REAL_E];
            let mut down = vec![u64::MAX; REAL_E];
            fill_tables(&layout, &group.experts, page_base, &mut gate_up, &mut down).unwrap();
            for id in 0..REAL_E {
                let inside = (first..first + 32).contains(&id);
                for (name, table) in [("gate_up", &gate_up), ("down", &down)] {
                    if inside {
                        assert!(
                            table[id] != 0 && table[id] != u64::MAX,
                            "{name} entry {id} of group {index} was not filled"
                        );
                    } else {
                        assert_eq!(table[id], 0, "{name} entry {id} outside group {index}");
                    }
                }
            }
            // Every pointer resolves, through the page layout, to the same logical
            // (expert, projection, byte offset): first and last byte of each extent.
            let resolve = |offset: usize| -> Option<(usize, Proj, usize)> {
                let run = layout
                    .runs
                    .iter()
                    .find(|r| r.dst <= offset && offset < r.dst + r.len)?;
                kind.decode(run.src + (offset - run.dst))
            };
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for &id in &group.experts {
                for (proj, table, len) in [
                    (Proj::GateUp, &gate_up, REAL_G),
                    (Proj::Down, &down, REAL_D),
                ] {
                    let entry = usize::try_from(table[id]).unwrap();
                    let offset = entry.checked_sub(page_base).expect("entry below the page");
                    assert_eq!(offset % EXTENT_ALIGN, 0, "expert {id} {proj:?} unaligned");
                    assert!(
                        offset + len <= layout.used,
                        "expert {id} {proj:?} past the payload"
                    );
                    assert_eq!(
                        resolve(offset),
                        Some((id, proj, 0)),
                        "expert {id} {proj:?} start"
                    );
                    assert_eq!(
                        resolve(offset + len - 1),
                        Some((id, proj, len - 1)),
                        "expert {id} {proj:?} end"
                    );
                    spans.push((offset, len));
                }
            }
            // Extents are disjoint in the page and tile the payload exactly.
            spans.sort_unstable();
            assert_eq!(spans.len(), 64);
            for pair in spans.windows(2) {
                assert!(pair[0].0 + pair[0].1 <= pair[1].0, "overlapping extents");
            }
            let last = spans.last().unwrap();
            assert_eq!(last.0 + last.1, layout.used);
            assert_eq!(spans.iter().map(|s| s.1).sum::<usize>(), layout.used);
        }
        // The sources' own range guard: an extent running one byte past its
        // range, or an alias outside every range, is refused.
        let mut experts = kind.alias_experts();
        if let Some(Some(last)) = experts.last_mut() {
            last.down_len += 1;
        }
        // SAFETY: construction must fail before anything is used.
        let overrun = unsafe {
            match kind {
                RealKind::Aggregate {
                    cpu_g,
                    alias_g,
                    cpu_d,
                    alias_d,
                } => LayerExpertSources::from_alias_split(
                    7,
                    (cpu_g, alias_g, REAL_E * REAL_G),
                    (cpu_d, alias_d, REAL_E * REAL_D),
                    &experts,
                    Arc::new(0u8),
                ),
                RealKind::Slab { cpu, alias } => LayerExpertSources::from_alias(
                    7,
                    cpu,
                    alias,
                    REAL_E * REAL_P,
                    &experts,
                    Arc::new(0u8),
                ),
            }
        };
        assert!(overrun.is_err(), "an extent past its range was accepted");
        drop(sources);
        assert_eq!(
            Arc::strong_count(&owner),
            1,
            "the owner is released with the sources"
        );
    }

    #[test]
    fn slowlink_ring_real_e512_aggregate_binds_every_pointer_to_its_logical_extent() {
        // Independent allocations: gate/up run then down run, never merged.
        prove_real_kind(RealKind::aggregate(), 2);
    }

    #[test]
    fn slowlink_ring_real_e512_interleaved_slab_binds_every_pointer_to_its_logical_extent() {
        // One interleaved arena: gate/up and down of consecutive experts touch,
        // so a whole group is one contiguous run.
        prove_real_kind(RealKind::slab(), 1);
    }

    #[test]
    fn slowlink_ring_real_e512_kinds_agree_on_logical_bindings_and_cross_kind_aliases_fail() {
        // The same group decodes to the same logical triples under both kinds,
        // although their physical runs, sizes and pointers differ.
        let (agg, slab) = (RealKind::aggregate(), RealKind::slab());
        let (sa, ss) = (
            agg.sources(Arc::new(0u8)).unwrap(),
            slab.sources(Arc::new(0u8)).unwrap(),
        );
        let ids: Vec<usize> = (480..512).collect();
        let (la, ls) = (
            plan_group_layout(&ids, &sa).unwrap(),
            plan_group_layout(&ids, &ss).unwrap(),
        );
        assert_eq!((la.runs.len(), ls.runs.len()), (2, 1));
        assert_eq!(la.used, ls.used);
        for (index, &id) in ids.iter().enumerate() {
            for (proj, len, a_off, s_off) in [
                (Proj::GateUp, REAL_G, la.gate_up[index], ls.gate_up[index]),
                (Proj::Down, REAL_D, la.down[index], ls.down[index]),
            ] {
                let find = |runs: &[CopyRun], off: usize| {
                    runs.iter()
                        .find(|r| r.dst <= off && off < r.dst + r.len)
                        .map(|r| r.src + (off - r.dst))
                        .unwrap()
                };
                let a = agg
                    .decode(find(la.runs.as_slice(), a_off + len - 1))
                    .unwrap();
                let s = slab
                    .decode(find(ls.runs.as_slice(), s_off + len - 1))
                    .unwrap();
                assert_eq!(a, (id, proj, len - 1));
                assert_eq!(a, s, "expert {id} {proj:?} binds differently per kind");
            }
        }
        // An aggregate alias table handed to the slab constructor (and the
        // converse) falls outside the declared range and is refused.
        let RealKind::Aggregate { alias_g, .. } = agg else {
            unreachable!()
        };
        let RealKind::Slab { cpu, .. } = slab else {
            unreachable!()
        };
        // SAFETY: construction must fail.
        let wrong = unsafe {
            LayerExpertSources::from_alias(
                7,
                cpu,
                alias_g + REAL_E * REAL_G,
                REAL_E * REAL_P,
                &agg.alias_experts(),
                Arc::new(0u8),
            )
        };
        assert!(wrong.is_err());
    }

    #[test]
    fn slowlink_ring_transport_owners_are_send() {
        fn assert_send<T: Send>() {}
        assert_send::<PinnedStaging>();
        assert_send::<HipPanelBackend>();
        assert_send::<ExpertPanelRing>();
        assert_send::<PanelSchedule>();
        assert_send::<LayerExpertSources>();
    }

    const GU: usize = 4096;
    const DN: usize = 2048;
    const HOST: usize = 0x7000_0000;
    const EXPERTS: usize = 64;

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Op {
        AwaitCopyDone(usize),
        Begin(usize, bool),
        Payload(usize, usize, usize, usize),
        Zero(usize, usize, usize),
        Tables(usize),
        End(usize),
        Acquire(usize),
        Release(usize),
        Drain,
        Base,
        Layer(usize, bool),
    }

    struct Mock {
        ops: Vec<Op>,
        tables: Vec<(Vec<u64>, Vec<u64>)>,
        clock: f64,
        times: HashMap<TimelineEvent, f64>,
        starved: bool,
        /// Fail the nth (0-based) op.
        fail_at: Option<usize>,
        fail_drain: bool,
        /// Counts backend drops: a leaked backend never increments it.
        dropped: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Drop for Mock {
        fn drop(&mut self) {
            self.dropped
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    impl Mock {
        fn new(pages: usize) -> Self {
            Self::sized(pages, EXPERTS)
        }

        /// A mock whose per-page pointer tables hold `experts` entries.
        fn sized(pages: usize, experts: usize) -> Self {
            Self {
                ops: Vec::new(),
                tables: (0..pages)
                    .map(|_| (vec![0; experts], vec![0; experts]))
                    .collect(),
                clock: 0.0,
                times: HashMap::new(),
                starved: false,
                fail_at: None,
                fail_drain: false,
                dropped: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }
        }

        fn op(&mut self, op: Op) -> R<()> {
            if self.fail_at == Some(self.ops.len()) {
                self.fail_at = None;
                return Err(invalid("injected device failure"));
            }
            self.ops.push(op);
            self.clock += 1.0;
            Ok(())
        }

        fn stamp(&mut self, event: TimelineEvent) {
            self.times.insert(event, self.clock);
        }
    }

    impl PanelBackend for Mock {
        type Ctx = ();

        fn page_base(&self, slot: usize) -> usize {
            0x2_0000_0000 + slot * 0x1000_0000
        }
        fn tables_mut(&mut self, slot: usize) -> (&mut [u64], &mut [u64]) {
            let (g, d) = &mut self.tables[slot];
            (g.as_mut_slice(), d.as_mut_slice())
        }
        fn await_copy_done(&mut self, _: &(), slot: usize) -> R<()> {
            self.op(Op::AwaitCopyDone(slot))
        }
        fn copy_begin(
            &mut self,
            _: &(),
            slot: usize,
            wait_free: bool,
            rec: Option<usize>,
        ) -> R<()> {
            self.op(Op::Begin(slot, wait_free))?;
            if let Some(r) = rec {
                self.stamp(TimelineEvent::CopyStart(r));
            }
            Ok(())
        }
        fn copy_payload(
            &mut self,
            _: &(),
            slot: usize,
            src: usize,
            dst: usize,
            len: usize,
        ) -> R<()> {
            self.op(Op::Payload(slot, src, dst, len))
        }
        fn copy_zero(&mut self, _: &(), slot: usize, dst: usize, len: usize) -> R<()> {
            self.op(Op::Zero(slot, dst, len))
        }
        fn copy_tables(&mut self, _: &(), slot: usize) -> R<()> {
            self.op(Op::Tables(slot))
        }
        fn copy_end(&mut self, _: &(), slot: usize, rec: Option<usize>) -> R<()> {
            self.op(Op::End(slot))?;
            if let Some(r) = rec {
                self.stamp(TimelineEvent::CopyEnd(r));
            }
            Ok(())
        }
        fn compute_acquire(&mut self, _: &(), slot: usize, rec: Option<usize>) -> R<bool> {
            self.op(Op::Acquire(slot))?;
            if let Some(r) = rec {
                self.stamp(TimelineEvent::ComputeBegin(r));
            }
            Ok(self.starved)
        }
        fn compute_release(&mut self, _: &(), slot: usize, rec: Option<usize>) -> R<()> {
            self.op(Op::Release(slot))?;
            if let Some(r) = rec {
                self.stamp(TimelineEvent::ComputeEnd(r));
            }
            Ok(())
        }
        fn mark_base(&mut self, _: &()) -> R<()> {
            self.op(Op::Base)?;
            self.stamp(TimelineEvent::Base);
            Ok(())
        }
        fn mark_layer(&mut self, _: &(), mark: usize, end: bool) -> R<()> {
            self.op(Op::Layer(mark, end))?;
            self.stamp(if end {
                TimelineEvent::LayerEnd(mark)
            } else {
                TimelineEvent::LayerBegin(mark)
            });
            Ok(())
        }
        fn drain(&mut self, _: &()) -> R<()> {
            if self.fail_drain {
                return Err(invalid("injected drain failure"));
            }
            self.op(Op::Drain)
        }
        fn elapsed_ms(&self, _: &(), from: TimelineEvent, to: TimelineEvent) -> R<f64> {
            let at = |e: &TimelineEvent| {
                self.times
                    .get(e)
                    .copied()
                    .ok_or_else(|| invalid(format!("no event {e:?}")))
            };
            Ok(at(&to)? - at(&from)?)
        }
    }

    fn spec(active: usize, lookahead: usize) -> PanelPoolSpec {
        PanelPoolSpec {
            active_slots: active,
            lookahead_pages: lookahead,
            experts_per_slot: 4,
            num_experts: EXPERTS,
            payload_bytes: 4 * (GU + DN),
            timeline_records: 64,
            timeline_layers: 8,
        }
    }

    fn core(active: usize, lookahead: usize) -> RingCore<Mock> {
        RingCore::new(spec(active, lookahead), Mock::new(active + lookahead)).unwrap()
    }

    /// Experts `0..EXPERTS` back to back (gate/up then down), codec per id.
    fn sources_with(
        layer: usize,
        owner: Arc<dyn Any + Send + Sync>,
        codec: impl Fn(usize) -> u32,
    ) -> LayerExpertSources {
        let experts = (0..EXPERTS)
            .map(|id| {
                let base = HOST + id * (GU + DN);
                Some(ExpertSource {
                    codec: PanelCodec(codec(id)),
                    gate_up: ExpertExtent {
                        addr: base,
                        len: GU,
                    },
                    down: ExpertExtent {
                        addr: base + GU,
                        len: DN,
                    },
                })
            })
            .collect();
        // SAFETY: the mock backend never dereferences a source address.
        unsafe { LayerExpertSources::new(layer, HOST, EXPERTS * (GU + DN), experts, owner) }
            .unwrap()
    }

    fn sources(layer: usize) -> LayerExpertSources {
        sources_with(layer, Arc::new(0u8), |_| 1)
    }

    fn group(sources: &LayerExpertSources, index: usize) -> ExpertGroupPlan {
        ExpertGroupPlan::partition(sources, 4).unwrap()[index].clone()
    }

    fn err_text(error: Qwen4GpuForwardError) -> String {
        error.to_string()
    }

    #[test]
    fn slowlink_ring_stale_epoch_layer_group_and_codec_are_rejected() {
        let mut ring = core(1, 0);
        ring.begin_window(&()).unwrap();
        ring.set_layers(Some(3), None).unwrap();
        let s = sources(3);
        let first = ring.enqueue(&(), &group(&s, 0), &s).unwrap();
        ring.acquire(&(), first).unwrap();
        ring.retire(&(), first).unwrap();
        // Same slot (one-page pool), new epoch: the old ticket is stale.
        let second = ring.enqueue(&(), &group(&s, 1), &s).unwrap();
        assert_eq!(first.slot, second.slot);
        assert!(second.epoch > first.epoch);
        let text = err_text(ring.acquire(&(), first).unwrap_err());
        assert!(
            text.contains("stale panel ticket") && text.contains("epoch"),
            "{text}"
        );
        assert!(ring.retire(&(), first).is_err());
        // Forged fields with the right epoch are each refused, state untouched.
        for (forged, needle) in [
            (PanelTicket { layer: 4, ..second }, "layer"),
            (PanelTicket { group: 9, ..second }, "group"),
            (
                PanelTicket {
                    codec: PanelCodec(7),
                    ..second
                },
                "codec",
            ),
        ] {
            let text = err_text(ring.acquire(&(), forged).unwrap_err());
            assert!(
                text.contains("stale panel ticket") && text.contains(needle),
                "{text}"
            );
        }
        ring.acquire(&(), second).unwrap();
        // Double retire / acquire of a retired ticket.
        ring.retire(&(), second).unwrap();
        assert!(ring.retire(&(), second).is_err());
        assert!(ring.acquire(&(), second).is_err());
    }

    #[test]
    fn slowlink_ring_delayed_free_is_never_reused_and_orders_reuse_after_release() {
        let mut ring = core(2, 0);
        ring.begin_window(&()).unwrap();
        ring.set_layers(Some(0), None).unwrap();
        let s = sources(0);
        let a = ring.enqueue(&(), &group(&s, 0), &s).unwrap();
        let b = ring.enqueue(&(), &group(&s, 1), &s).unwrap();
        ring.acquire(&(), a).unwrap();
        ring.acquire(&(), b).unwrap();
        // Both pages are in use: nothing may be refilled, however long it takes.
        assert_eq!(ring.try_enqueue(&(), &group(&s, 2), &s).unwrap(), None);
        assert_eq!(ring.stats().page_full_stalls, 1);
        let text = err_text(ring.enqueue(&(), &group(&s, 2), &s).unwrap_err());
        assert!(text.contains("full"), "{text}");
        // Only the retired page comes back, after its free event was recorded.
        ring.retire(&(), b).unwrap();
        let c = ring.enqueue(&(), &group(&s, 2), &s).unwrap();
        assert_eq!(c.slot, b.slot);
        let ops = &ring.backend.ops;
        let release = ops.iter().position(|o| *o == Op::Release(b.slot)).unwrap();
        let begin = ops
            .iter()
            .rposition(|o| matches!(o, Op::Begin(s, _) if *s == b.slot))
            .unwrap();
        assert!(release < begin);
        assert_eq!(
            ops[begin],
            Op::Begin(b.slot, true),
            "reuse waits the free event"
        );
        // The first fill of a page never waits a free event nobody recorded.
        assert!(ops.contains(&Op::Begin(a.slot, false)));
    }

    #[test]
    fn slowlink_ring_free_pages_are_reused_in_freed_order() {
        let mut ring = core(3, 0);
        ring.begin_window(&()).unwrap();
        ring.set_layers(Some(0), None).unwrap();
        let s = sources(0);
        let t: Vec<_> = (0..3)
            .map(|g| ring.enqueue(&(), &group(&s, g), &s).unwrap())
            .collect();
        for ticket in [t[2], t[0], t[1]] {
            ring.acquire(&(), ticket).unwrap();
            ring.retire(&(), ticket).unwrap();
        }
        let next: Vec<_> = (3..6)
            .map(|g| ring.enqueue(&(), &group(&s, g), &s).unwrap().slot)
            .collect();
        assert_eq!(next, vec![t[2].slot, t[0].slot, t[1].slot]);
    }

    #[test]
    fn slowlink_ring_layout_rebases_pointers_and_zeroes_absent_entries() {
        let s = sources(0);
        let g = group(&s, 2); // experts 8..12
        assert_eq!(g.experts.as_slice(), &[8, 9, 10, 11]);
        let layout = plan_group_layout(&g.experts, &s).unwrap();
        // Contiguous experts coalesce into one run copied once.
        assert_eq!(layout.runs.len(), 1);
        assert_eq!(layout.runs[0].src, HOST + 8 * (GU + DN));
        assert_eq!(layout.runs[0].len, 4 * (GU + DN));
        assert_eq!(layout.submitted_bytes(), layout.extent_bytes);
        assert_eq!(layout.used, layout.runs[0].dst + 4 * (GU + DN));
        assert_eq!(
            layout.runs[0].dst % EXTENT_ALIGN,
            layout.runs[0].src % EXTENT_ALIGN
        );
        let mut gu = vec![u64::MAX; EXPERTS];
        let mut dn = vec![u64::MAX; EXPERTS];
        let base = 0x2_0000_0000usize;
        fill_tables(&layout, &g.experts, base, &mut gu, &mut dn).unwrap();
        for (i, &id) in g.experts.iter().enumerate() {
            let dst = layout.runs[0].dst + i * (GU + DN);
            assert_eq!(gu[id], (base + dst) as u64);
            assert_eq!(dn[id], (base + dst + GU) as u64);
        }
        assert_eq!(gu.iter().filter(|&&v| v != 0).count(), 4);
        assert_eq!(dn.iter().filter(|&&v| v != 0).count(), 4);
        // Rebasing refuses to wrap the address space.
        assert!(fill_tables(&layout, &g.experts, usize::MAX - 8, &mut gu, &mut dn).is_err());
        // An expert id outside the table is refused.
        assert!(fill_tables(&layout, &[EXPERTS], base, &mut gu, &mut dn).is_err());
    }

    #[test]
    fn slowlink_ring_layout_packs_separate_runs_and_preserves_source_alignment() {
        // Experts 2 and 0 in descending host order with a hole between them.
        let experts = vec![
            Some(ExpertSource {
                codec: PanelCodec(1),
                gate_up: ExpertExtent {
                    addr: HOST + 10_000,
                    len: 1000,
                },
                down: ExpertExtent {
                    addr: HOST + 11_000,
                    len: 500,
                },
            }),
            None,
            Some(ExpertSource {
                codec: PanelCodec(1),
                gate_up: ExpertExtent {
                    addr: HOST + 100_000,
                    len: 777,
                },
                down: ExpertExtent {
                    addr: HOST + 200_001,
                    len: 33,
                },
            }),
        ];
        // SAFETY: never dereferenced.
        let s =
            unsafe { LayerExpertSources::new(0, HOST, 1 << 20, experts, Arc::new(0u8)) }.unwrap();
        let layout = plan_group_layout(&[0, 2], &s).unwrap();
        assert_eq!(layout.runs.len(), 3);
        for run in &layout.runs {
            assert_eq!(run.dst % EXTENT_ALIGN, run.src % EXTENT_ALIGN);
        }
        // Runs never overlap in the page and every extent lands inside a run.
        let mut sorted: Vec<_> = layout.runs.iter().copied().collect();
        sorted.sort_by_key(|r| r.dst);
        for pair in sorted.windows(2) {
            assert!(pair[0].dst + pair[0].len <= pair[1].dst);
        }
        assert_eq!(layout.used, sorted[2].dst + sorted[2].len);
        assert_eq!(layout.extent_bytes, 1000 + 500 + 777 + 33);
        assert_eq!(layout.submitted_bytes(), layout.extent_bytes);
    }

    #[test]
    fn slowlink_ring_layout_overlap_is_copied_once_and_overflow_is_refused() {
        let experts = vec![Some(ExpertSource {
            codec: PanelCodec(1),
            gate_up: ExpertExtent {
                addr: HOST,
                len: 1000,
            },
            down: ExpertExtent {
                addr: HOST + 600,
                len: 1000,
            },
        })];
        // SAFETY: never dereferenced.
        let s = unsafe { LayerExpertSources::new(0, HOST, 4096, experts, Arc::new(0u8)) }.unwrap();
        let layout = plan_group_layout(&[0], &s).unwrap();
        assert_eq!(layout.runs.len(), 1);
        assert_eq!(layout.submitted_bytes(), 1600);
        assert_eq!(layout.extent_bytes, 2000);
        assert_eq!(layout.down[0] - layout.gate_up[0], 600);
        // Extent length overflowing the address space.
        let bad = vec![Some(ExpertSource {
            codec: PanelCodec(1),
            gate_up: ExpertExtent {
                addr: usize::MAX - 4,
                len: 100,
            },
            down: ExpertExtent { addr: HOST, len: 8 },
        })];
        // SAFETY: construction must fail before anything is read.
        assert!(unsafe { LayerExpertSources::new(0, HOST, 4096, bad, Arc::new(0u8)) }.is_err());
    }

    #[test]
    fn slowlink_ring_sources_refuse_ranges_outside_the_host_owner() {
        let mk = |gate_up: ExpertExtent, down: ExpertExtent| {
            // SAFETY: construction is expected to fail before anything is read.
            unsafe {
                LayerExpertSources::new(
                    0,
                    HOST,
                    1 << 20,
                    vec![Some(ExpertSource {
                        codec: PanelCodec(1),
                        gate_up,
                        down,
                    })],
                    Arc::new(0u8),
                )
            }
        };
        let ok = ExpertExtent {
            addr: HOST,
            len: 256,
        };
        assert!(mk(
            ok,
            ExpertExtent {
                addr: HOST + 256,
                len: 256
            }
        )
        .is_ok());
        assert!(mk(
            ok,
            ExpertExtent {
                addr: HOST - 256,
                len: 256
            }
        )
        .is_err());
        assert!(mk(
            ok,
            ExpertExtent {
                addr: HOST + (1 << 20) - 255,
                len: 256
            }
        )
        .is_err());
        assert!(mk(ok, ExpertExtent { addr: HOST, len: 0 }).is_err());
        // SAFETY: as above.
        assert!(unsafe { LayerExpertSources::new(0, HOST, 1, Vec::new(), Arc::new(0u8)) }.is_err());
        // Host range overflow.
        assert!(unsafe {
            LayerExpertSources::new(0, usize::MAX - 8, 64, vec![None], Arc::new(0u8))
        }
        .is_err());
    }

    #[test]
    fn slowlink_ring_alias_translation_rebases_to_cpu_addresses() {
        let alias_base = 0x5000_0000usize;
        let e = |off: usize| AliasExpert {
            codec: PanelCodec(1),
            gate_up_addr: alias_base + off,
            gate_up_len: GU,
            down_addr: alias_base + off + GU,
            down_len: DN,
        };
        // SAFETY: never dereferenced.
        let s = unsafe {
            LayerExpertSources::from_alias(
                2,
                HOST,
                alias_base,
                1 << 20,
                &[Some(e(0)), None, Some(e(GU + DN))],
                Arc::new(0u8),
            )
        }
        .unwrap();
        assert_eq!(s.expert(0).unwrap().gate_up.addr, HOST);
        assert_eq!(s.expert(2).unwrap().down.addr, HOST + GU + DN + GU);
        assert!(s.expert(1).is_none());
        for bad in [alias_base - 1, alias_base + (1 << 20)] {
            let mut x = e(0);
            x.gate_up_addr = bad;
            // SAFETY: construction must fail.
            assert!(unsafe {
                LayerExpertSources::from_alias(
                    0,
                    HOST,
                    alias_base,
                    1 << 20,
                    &[Some(x)],
                    Arc::new(0u8),
                )
            }
            .is_err());
        }
    }

    #[test]
    fn slowlink_ring_partition_splits_on_codec_and_group_size() {
        let s = sources_with(5, Arc::new(0u8), |id| u32::from(id >= 6));
        let groups = ExpertGroupPlan::partition(&s, 4).unwrap();
        // 0..4, 4..6 (codec change), 6..10, ... 60..64.
        assert_eq!(groups[0].experts.as_slice(), &[0, 1, 2, 3]);
        assert_eq!(groups[1].experts.as_slice(), &[4, 5]);
        assert_eq!(groups[2].experts.as_slice(), &[6, 7, 8, 9]);
        assert!(groups
            .iter()
            .enumerate()
            .all(|(i, g)| g.group as usize == i && g.layer == 5));
        let all: Vec<usize> = groups
            .iter()
            .flat_map(|g| g.experts.iter().copied())
            .collect();
        assert_eq!(all, (0..EXPERTS).collect::<Vec<_>>());
        assert!(ExpertGroupPlan::partition(&s, 0).is_err());
        assert!(ExpertGroupPlan::partition(&s, 33).is_err());
        // A mixed-codec group is refused at enqueue.
        let mut ring = core(1, 0);
        ring.begin_window(&()).unwrap();
        ring.set_layers(Some(5), None).unwrap();
        let mut mixed = groups[1].clone();
        mixed.experts.push(6);
        assert!(ring.enqueue(&(), &mixed, &s).is_err());
        assert!(
            !ring.is_quarantined(),
            "a validation failure mutates nothing"
        );
        assert!(ring.backend.ops.len() == 1, "only the window base op ran");
    }

    #[test]
    fn slowlink_ring_retained_sources_live_until_copy_done_or_drain() {
        let owner = Arc::new(0u8);
        let any: Arc<dyn Any + Send + Sync> = owner.clone();
        let s = sources_with(0, any, |_| 1);
        assert_eq!(Arc::strong_count(&owner), 2);
        let mut ring = core(1, 0);
        ring.begin_window(&()).unwrap();
        ring.set_layers(Some(0), None).unwrap();
        let t = ring.enqueue(&(), &group(&s, 0), &s).unwrap();
        assert_eq!(
            Arc::strong_count(&owner),
            3,
            "the page retains the source owner"
        );
        ring.acquire(&(), t).unwrap();
        ring.retire(&(), t).unwrap();
        // Retired but not yet known complete on the host: still retained.
        assert_eq!(Arc::strong_count(&owner), 3);
        let s2_owner = Arc::new(1u8);
        let s2 = sources_with(0, s2_owner.clone(), |_| 1);
        let t2 = ring.enqueue(&(), &group(&s2, 1), &s2).unwrap();
        // The refill waited for the previous copy on the host before dropping it.
        assert_eq!(Arc::strong_count(&owner), 2);
        let wait = ring
            .backend
            .ops
            .iter()
            .rposition(|o| *o == Op::AwaitCopyDone(t.slot))
            .unwrap();
        let tables = ring
            .backend
            .ops
            .iter()
            .rposition(|o| *o == Op::Tables(t.slot))
            .unwrap();
        assert!(
            wait < tables,
            "staging is rewritten only after the old copy completed"
        );
        assert_eq!(Arc::strong_count(&s2_owner), 3);
        ring.acquire(&(), t2).unwrap();
        ring.retire(&(), t2).unwrap();
        ring.drain(&()).unwrap();
        assert_eq!(
            Arc::strong_count(&s2_owner),
            2,
            "drain releases the retained owner"
        );
    }

    #[test]
    fn slowlink_ring_failure_quarantines_and_drain_recovers() {
        let owner = Arc::new(0u8);
        let any: Arc<dyn Any + Send + Sync> = owner.clone();
        let s = sources_with(0, any, |_| 1);
        let mut ring = core(2, 0);
        ring.begin_window(&()).unwrap();
        ring.set_layers(Some(0), None).unwrap();
        let ok = ring.enqueue(&(), &group(&s, 0), &s).unwrap();
        // Fail in the middle of the second group's copy (after Begin, at the payload).
        let at = ring.backend.ops.len() + 2;
        ring.backend.fail_at = Some(at);
        assert!(ring.enqueue(&(), &group(&s, 1), &s).is_err());
        assert!(ring.is_quarantined());
        assert_eq!(
            Arc::strong_count(&owner),
            4,
            "owners stay held while quarantined"
        );
        for result in [
            ring.try_enqueue(&(), &group(&s, 2), &s).map(|_| ()),
            ring.acquire(&(), ok),
            ring.retire(&(), ok),
            ring.begin_window(&()),
            ring.set_layers(Some(0), None),
        ] {
            assert!(err_text(result.unwrap_err()).contains("quarantined"));
        }
        // A drain that fails keeps the quarantine and every owner.
        ring.backend.fail_drain = true;
        assert!(ring.drain(&()).is_err());
        assert!(ring.is_quarantined());
        assert_eq!(Arc::strong_count(&owner), 4);
        ring.backend.fail_drain = false;
        ring.drain(&()).unwrap();
        assert!(!ring.is_quarantined());
        assert_eq!(
            Arc::strong_count(&owner),
            2,
            "drain released every retained owner"
        );
        // Outstanding tickets are stale; the ring works again.
        assert!(ring.acquire(&(), ok).is_err());
        ring.begin_window(&()).unwrap();
        ring.set_layers(Some(0), None).unwrap();
        let again = ring.enqueue(&(), &group(&s, 0), &s).unwrap();
        assert!(again.epoch > ok.epoch);
        // After the drain no page waits a free event nobody recorded.
        assert!(matches!(ring.backend.ops.last(), Some(Op::End(_))));
        assert!(ring
            .backend
            .ops
            .iter()
            .rev()
            .find_map(|o| match o {
                Op::Begin(_, wait) => Some(!wait),
                _ => None,
            })
            .unwrap());
    }

    #[test]
    fn slowlink_ring_failure_during_acquire_or_retire_quarantines() {
        let s = sources(0);
        for stage in 0..2 {
            let mut ring = core(1, 0);
            ring.begin_window(&()).unwrap();
            ring.set_layers(Some(0), None).unwrap();
            let t = ring.enqueue(&(), &group(&s, 0), &s).unwrap();
            if stage == 0 {
                ring.backend.fail_at = Some(ring.backend.ops.len());
                assert!(ring.acquire(&(), t).is_err());
            } else {
                ring.acquire(&(), t).unwrap();
                ring.backend.fail_at = Some(ring.backend.ops.len());
                assert!(ring.retire(&(), t).is_err());
            }
            assert!(ring.is_quarantined());
            ring.drain(&()).unwrap();
            assert!(!ring.is_quarantined());
        }
    }

    #[test]
    fn slowlink_ring_lookahead_prefetches_before_counts_and_never_takes_active_slots() {
        let mut ring = core(2, 3);
        ring.begin_window(&()).unwrap();
        let l1 = sources(1);
        // Before the first host layer computes: no current layer, layer 1 next.
        ring.set_layers(None, Some(1)).unwrap();
        let mut tickets = Vec::new();
        for g in 0..3 {
            let t = ring.enqueue(&(), &group(&l1, g), &l1).unwrap();
            assert!(t.slot >= 2, "lookahead pages only");
            tickets.push(t);
        }
        // The lookahead store is full though both active slots are free.
        assert_eq!(ring.try_enqueue(&(), &group(&l1, 3), &l1).unwrap(), None);
        // Compute reaches layer 1: its remaining groups use the active slots and
        // already-ready pages are consumed in place, with no refetch.
        ring.set_layers(Some(1), Some(2)).unwrap();
        let t3 = ring.enqueue(&(), &group(&l1, 3), &l1).unwrap();
        assert!(t3.slot < 2);
        let ops_before = ring.backend.ops.len();
        ring.acquire(&(), tickets[0]).unwrap();
        assert_eq!(
            &ring.backend.ops[ops_before..],
            &[Op::Acquire(tickets[0].slot)]
        );
        // A group that was already transferred cannot be transferred again.
        let text = err_text(ring.enqueue(&(), &group(&l1, 1), &l1).unwrap_err());
        assert!(
            text.contains("again") || text.contains("out of order"),
            "{text}"
        );
        // Current layer enqueue without a current layer is refused up front.
        let mut early = core(2, 2);
        early.begin_window(&()).unwrap();
        early.set_layers(None, Some(1)).unwrap();
        let l2 = sources(2);
        assert!(early.enqueue(&(), &group(&l2, 0), &l2).is_err());
        // At most current plus one layer ahead.
        let l3 = sources(3);
        assert!(early.enqueue(&(), &group(&l3, 0), &l3).is_err());
    }

    #[test]
    fn slowlink_ring_window_moves_forward_only_and_refuses_stranded_pages() {
        let mut ring = core(2, 2);
        ring.begin_window(&()).unwrap();
        ring.set_layers(None, Some(4)).unwrap();
        let l4 = sources(4);
        let t = ring.enqueue(&(), &group(&l4, 0), &l4).unwrap();
        // Skipping past layer 4 with its page still ready is refused.
        assert!(ring.set_layers(Some(6), Some(7)).is_err());
        ring.set_layers(Some(4), Some(6)).unwrap();
        assert!(ring.set_layers(Some(3), Some(6)).is_err());
        assert!(ring.set_layers(Some(5), Some(5)).is_err());
        assert!(ring.set_layers(None, Some(6)).is_err());
        // A prefetched page whose experts routing never used is released
        // unread and counted, not refetched or leaked.
        ring.retire(&(), t).unwrap();
        let stats = ring.stats();
        assert_eq!((stats.discarded, stats.acquired), (1, 0));
        assert_eq!(stats.discarded_payload_bytes, stats.submitted_payload_bytes);
        ring.set_layers(Some(6), Some(7)).unwrap();
        // A new window needs every page free.
        let l6 = sources(6);
        let live = ring.enqueue(&(), &group(&l6, 0), &l6).unwrap();
        assert!(ring.begin_window(&()).is_err());
        ring.retire(&(), live).unwrap();
        ring.begin_window(&()).unwrap();
        assert_eq!(ring.stats(), PanelStats::default());
    }

    #[test]
    fn slowlink_ring_page_reused_across_layer_epochs_gets_fresh_tables() {
        let mut ring = core(1, 0);
        ring.begin_window(&()).unwrap();
        let (a, b) = (sources(1), sources_with(2, Arc::new(9u8), |_| 2));
        let mut seen = Vec::new();
        for (current, s, g) in [(1, &a, 0), (2, &b, 7)] {
            ring.set_layers(Some(current), None).unwrap();
            let t = ring.enqueue(&(), &group(s, g), s).unwrap();
            seen.push((
                t,
                ring.backend.tables[t.slot].0.clone(),
                ring.backend.page_base(t.slot),
            ));
            ring.acquire(&(), t).unwrap();
            ring.retire(&(), t).unwrap();
        }
        // Same slot, same device address, different epoch/layer/codec identity.
        assert_eq!(seen[0].0.slot, seen[1].0.slot);
        assert_eq!(seen[0].2, seen[1].2);
        assert_ne!(seen[0].0.epoch, seen[1].0.epoch);
        assert_ne!(seen[0].0.codec, seen[1].0.codec);
        assert_ne!(
            seen[0].1, seen[1].1,
            "the new group's tables name its own bytes"
        );
        // The old layer's ticket cannot be acquired at the same address.
        assert!(ring.acquire(&(), seen[0].0).is_err());
    }

    #[test]
    fn slowlink_ring_oversized_group_for_the_page_is_refused_before_mutation() {
        let mut small = spec(1, 0);
        small.payload_bytes = 2 * (GU + DN);
        let mut ring = RingCore::new(small, Mock::new(1)).unwrap();
        ring.begin_window(&()).unwrap();
        ring.set_layers(Some(0), None).unwrap();
        let s = sources(0);
        let ops = ring.backend.ops.len();
        let text = err_text(ring.enqueue(&(), &group(&s, 0), &s).unwrap_err());
        assert!(text.contains("page holds"), "{text}");
        assert_eq!(ring.backend.ops.len(), ops);
        assert!(!ring.is_quarantined());
        assert_eq!(ring.free.len(), 1, "the page stayed free");
    }

    /// A real host arena the sources point into (test memory, not a mock of
    /// the verifier): the verifier compares against these actual bytes.
    struct Arena {
        bytes: Vec<u8>,
    }

    impl Arena {
        fn new() -> Self {
            let mut state = 0x9E37_79B9_7F4A_7C15u64;
            let bytes = (0..EXPERTS * (GU + DN) + 4096)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (state >> 33) as u8 | 1
                })
                .collect();
            Self { bytes }
        }
        fn sources(&self, layer: usize) -> LayerExpertSources {
            let base = self.bytes.as_ptr() as usize;
            let experts = (0..EXPERTS)
                .map(|id| {
                    let at = base + id * (GU + DN);
                    Some(ExpertSource {
                        codec: PanelCodec(1),
                        gate_up: ExpertExtent { addr: at, len: GU },
                        down: ExpertExtent {
                            addr: at + GU,
                            len: DN,
                        },
                    })
                })
                .collect();
            // SAFETY: `self.bytes` outlives the sources in every test below.
            unsafe {
                LayerExpertSources::new(layer, base, EXPERTS * (GU + DN), experts, Arc::new(0u8))
            }
            .unwrap()
        }
    }

    /// The page image the copy stream would produce, plus its tables.
    fn page_image(
        s: &LayerExpertSources,
        experts: &[usize],
        base: usize,
    ) -> (Vec<u8>, usize, Vec<u64>, Vec<u64>) {
        let layout = plan_group_layout(experts, s).unwrap();
        let mut image = vec![0u8; layout.used + ARENA_TAIL_PAD_BYTES];
        for run in &layout.runs {
            // SAFETY: the run lies inside the arena the sources were built on.
            let src = unsafe { std::slice::from_raw_parts(run.src as *const u8, run.len) };
            image[run.dst..run.dst + run.len].copy_from_slice(src);
        }
        let (mut gu, mut dn) = (vec![0u64; EXPERTS], vec![0u64; EXPERTS]);
        fill_tables(&layout, experts, base, &mut gu, &mut dn).unwrap();
        (image, layout.used, gu, dn)
    }

    #[test]
    fn slowlink_ring_image_verifier_accepts_the_real_copy_and_counts_what_it_compared() {
        let arena = Arena::new();
        let s = arena.sources(4);
        let experts = [3usize, 4, 9, 40];
        let base = 0x2_0000_0000usize;
        let (image, used, gu, dn) = page_image(&s, &experts, base);
        let check = verify_page_image(&image, used, base, &experts, &s, &gu, &dn).unwrap();
        assert_eq!(check.payload_bytes, (4 * (GU + DN)) as u64);
        assert_eq!(check.table_entries, 2 * EXPERTS as u64);
        assert_eq!(check.pad_bytes, ARENA_TAIL_PAD_BYTES as u64);
    }

    #[test]
    fn slowlink_ring_image_verifier_rejects_every_kind_of_corruption() {
        let arena = Arena::new();
        let s = arena.sources(4);
        let experts = [3usize, 4, 9];
        let base = 0x2_0000_0000usize;
        let (image, used, gu, dn) = page_image(&s, &experts, base);
        let ok = |image: &[u8], gu: &[u64], dn: &[u64]| {
            verify_page_image(image, used, base, &experts, &s, gu, dn)
        };
        assert!(ok(&image, &gu, &dn).is_ok());
        // A refilled page: any payload byte differing from the pinned source.
        let at3 = (gu[3] as usize) - base;
        let at9 = (dn[9] as usize) - base;
        for at in [at3, at3 + GU - 1, (dn[3] as usize) - base, at9 + DN - 1] {
            let mut bad = image.clone();
            bad[at] ^= 0x55;
            assert!(ok(&bad, &gu, &dn).is_err(), "payload byte {at}");
        }
        // A nonzero tail pad byte.
        let mut bad = image.clone();
        bad[used] = 1;
        assert!(ok(&bad, &gu, &dn).is_err());
        let mut bad = image.clone();
        bad[used + ARENA_TAIL_PAD_BYTES - 1] = 1;
        assert!(ok(&bad, &gu, &dn).is_err());
        // A short image cannot hide missing bytes.
        assert!(ok(&image[..used], &gu, &dn).is_err());
        // A stale table entry (previous epoch's address), a pointer outside
        // the payload, a missing group entry and a stray non-group entry.
        let mut bad = gu.clone();
        bad[4] += 8;
        assert!(ok(&image, &bad, &dn).is_err());
        let mut bad = dn.clone();
        bad[9] = (base + used) as u64;
        assert!(ok(&image, &gu, &bad).is_err());
        let mut bad = gu.clone();
        bad[3] = 0;
        assert!(ok(&image, &bad, &dn).is_err());
        let mut bad = dn.clone();
        bad[10] = base as u64;
        assert!(ok(&image, &gu, &bad).is_err());
        // Tables of the wrong length.
        assert!(ok(&image, &gu[..EXPERTS - 1], &dn).is_err());
    }

    #[test]
    fn slowlink_ring_image_verifier_distinguishes_same_address_different_epochs() {
        // The same page address holds group A, then group B: B's image verified
        // against A's expectations (a reader that ran after the refill) fails.
        let arena = Arena::new();
        let s = arena.sources(0);
        let base = 0x2_0000_0000usize;
        let a = [0usize, 1, 2, 3];
        let b = [4usize, 5, 6, 7];
        let (img_a, used_a, gu_a, dn_a) = page_image(&s, &a, base);
        let (img_b, used_b, gu_b, dn_b) = page_image(&s, &b, base);
        assert_eq!(used_a, used_b);
        verify_page_image(&img_a, used_a, base, &a, &s, &gu_a, &dn_a).unwrap();
        verify_page_image(&img_b, used_b, base, &b, &s, &gu_b, &dn_b).unwrap();
        assert!(verify_page_image(&img_b, used_a, base, &a, &s, &gu_a, &dn_a).is_err());
        assert!(verify_page_image(&img_a, used_a, base, &a, &s, &gu_b, &dn_b).is_err());
    }

    #[test]
    fn slowlink_ring_selftest_report_counts_repeated_address_epochs() {
        let mut seen: HashMap<usize, Vec<u64>> = HashMap::new();
        for (address, epoch) in [(1usize, 1u64), (2, 2), (1, 3), (1, 5), (2, 6), (3, 7)] {
            seen.entry(address).or_default().push(epoch);
        }
        let report = selftest_epoch_summary(&seen);
        assert_eq!(report.distinct_addresses, 3);
        assert_eq!(report.repeated_address_epochs, 2);
        assert_eq!(
            (report.min_epochs_per_address, report.max_epochs_per_address),
            (1, 3)
        );
        let empty = selftest_epoch_summary(&HashMap::new());
        assert_eq!(empty.distinct_addresses, 0);
        assert_eq!(empty.min_epochs_per_address, 0);
    }

    #[test]
    fn slowlink_ring_split_ranges_translate_per_projection_and_never_merge() {
        let (a_gu, a_dn) = (0x5000_0000usize, 0x6000_0000usize);
        let c_gu = HOST;
        // Numerically adjacent CPU ranges: a merge across them would be tempting.
        let c_dn = HOST + 64 * GU;
        let e = |id: usize| AliasExpert {
            codec: PanelCodec(1),
            gate_up_addr: a_gu + id * GU,
            gate_up_len: GU,
            down_addr: a_dn + id * DN,
            down_len: DN,
        };
        let experts: Vec<Option<AliasExpert>> = (0..8).map(|id| Some(e(id))).collect();
        // SAFETY: never dereferenced.
        let s = unsafe {
            LayerExpertSources::from_alias_split(
                3,
                (c_gu, a_gu, 64 * GU),
                (c_dn, a_dn, 64 * DN),
                &experts,
                Arc::new(0u8),
            )
        }
        .unwrap();
        assert_eq!(s.host_ranges(), &[(c_gu, 64 * GU), (c_dn, 64 * DN)]);
        assert_eq!(s.expert(2).unwrap().gate_up.addr, c_gu + 2 * GU);
        assert_eq!(s.expert(2).unwrap().down.addr, c_dn + 2 * DN);
        let layout = plan_group_layout(&[0, 1, 2, 3], &s).unwrap();
        assert_eq!(layout.runs.len(), 2);
        assert_eq!(layout.runs[0].src, c_gu);
        assert_eq!(layout.runs[0].len, 4 * GU);
        assert_eq!(layout.runs[1].src, c_dn);
        assert_eq!(layout.runs[1].len, 4 * DN);
        assert_eq!(layout.submitted_bytes(), layout.extent_bytes);
        assert_eq!(layout.gate_up[1] - layout.gate_up[0], GU);
        assert_eq!(layout.down[1] - layout.down[0], DN);
        // Adjacent ranges: the last gate/up byte and the first down byte stay
        // two copies even though their CPU addresses touch.
        let touching = vec![Some(ExpertSource {
            codec: PanelCodec(1),
            gate_up: ExpertExtent {
                addr: HOST,
                len: GU,
            },
            down: ExpertExtent {
                addr: HOST + GU,
                len: DN,
            },
        })];
        // SAFETY: never dereferenced.
        let one =
            unsafe { LayerExpertSources::new(0, HOST, GU + DN, touching.clone(), Arc::new(0u8)) }
                .unwrap();
        assert_eq!(plan_group_layout(&[0], &one).unwrap().runs.len(), 1);
        // SAFETY: never dereferenced.
        let split = unsafe {
            LayerExpertSources::new_split(0, (HOST, GU), (HOST + GU, DN), touching, Arc::new(0u8))
        }
        .unwrap();
        assert_eq!(plan_group_layout(&[0], &split).unwrap().runs.len(), 2);
        // A projection naming the other projection's range is refused, as are
        // overlapping ranges.
        let mut bad = e(0);
        bad.down_addr = a_gu;
        // SAFETY: construction must fail.
        assert!(unsafe {
            LayerExpertSources::from_alias_split(
                0,
                (c_gu, a_gu, 64 * GU),
                (c_dn, a_dn, 64 * DN),
                &[Some(bad)],
                Arc::new(0u8),
            )
        }
        .is_err());
        // SAFETY: construction must fail.
        assert!(unsafe {
            LayerExpertSources::new_split(
                0,
                (HOST, 2 * GU),
                (HOST + GU, 2 * DN),
                vec![None],
                Arc::new(0u8),
            )
        }
        .is_err());
        // Direct extents are checked against their own range only.
        let swapped = vec![Some(ExpertSource {
            codec: PanelCodec(1),
            gate_up: ExpertExtent {
                addr: HOST + GU,
                len: DN,
            },
            down: ExpertExtent {
                addr: HOST,
                len: GU,
            },
        })];
        // SAFETY: construction must fail.
        assert!(unsafe {
            LayerExpertSources::new_split(0, (HOST, GU), (HOST + GU, DN), swapped, Arc::new(0u8))
        }
        .is_err());
    }

    #[test]
    fn slowlink_ring_uniform_spec_bounds_every_possible_group_layout() {
        let (gu, dn) = (1_740_800usize, 870_400usize);
        let spec = PanelPoolSpec::uniform(2, 16, 32, 512, gu, dn, 0, 0).unwrap();
        assert_eq!(spec.payload_bytes, 32 * (gu + dn) + UNIFORM_RUN_SLACK);
        // Worst case: every one of the 64 extents is its own run at the worst
        // source alignment; the page still holds it plus the tail pad.
        let experts: Vec<Option<ExpertSource>> = (0..512)
            .map(|id| {
                let base = HOST + id * (gu + dn + 4096) + 255;
                Some(ExpertSource {
                    codec: PanelCodec(1),
                    gate_up: ExpertExtent {
                        addr: base,
                        len: gu,
                    },
                    down: ExpertExtent {
                        addr: base + gu + 2048,
                        len: dn,
                    },
                })
            })
            .collect();
        // SAFETY: never dereferenced.
        let s = unsafe {
            LayerExpertSources::new(
                0,
                HOST,
                512 * (gu + dn + 4096) + 4096,
                experts,
                Arc::new(0u8),
            )
        }
        .unwrap();
        let group: Vec<usize> = (0..32).collect();
        let layout = plan_group_layout(&group, &s).unwrap();
        assert_eq!(layout.runs.len(), 64);
        assert!(layout.used + ARENA_TAIL_PAD_BYTES <= spec.page_bytes().unwrap());
        assert!(PanelPoolSpec::uniform(2, 16, 33, 512, gu, dn, 0, 0).is_err());
        assert!(PanelPoolSpec::uniform(2, 16, 32, 512, usize::MAX, dn, 0, 0).is_err());
        assert!(PanelPoolSpec::uniform(2, 16, 32, 512, 0, dn, 0, 0).is_err());
    }

    #[test]
    fn slowlink_ring_pool_reserve_is_the_charged_mq4_working_set() {
        // MQ4: 1,740,800 + 870,400 bytes per expert, 32-expert pages, two active
        // slots plus a 512-expert lookahead store of 16 pages.
        let group_bytes = 32 * (1_740_800 + 870_400);
        let spec = PanelPoolSpec {
            active_slots: 2,
            lookahead_pages: 16,
            experts_per_slot: 32,
            num_experts: 512,
            payload_bytes: group_bytes,
            timeline_records: 0,
            timeline_layers: 0,
        };
        spec.validate().unwrap();
        let reserve = spec.reserve().unwrap();
        assert_eq!(reserve.pages, 18);
        assert!(reserve.page_bytes >= group_bytes + ARENA_TAIL_PAD_BYTES);
        assert_eq!(reserve.page_bytes % EXTENT_ALIGN, 0);
        assert_eq!(reserve.payload, 18 * reserve.page_bytes);
        assert_eq!(reserve.tables, 18 * 2 * 512 * 8);
        assert_eq!(reserve.staging, reserve.tables);
        assert_eq!(reserve.device_total, reserve.payload + reserve.tables);
        let gib = reserve.payload as f64 / (1u64 << 30) as f64;
        assert!((1.40..1.50).contains(&gib), "payload {gib} GiB");
        // Invalid or overflowing geometry never reaches an allocation.
        for bad in [
            PanelPoolSpec {
                active_slots: 0,
                ..spec
            },
            PanelPoolSpec {
                experts_per_slot: 33,
                ..spec
            },
            PanelPoolSpec {
                num_experts: 0,
                ..spec
            },
            PanelPoolSpec {
                payload_bytes: 0,
                ..spec
            },
            PanelPoolSpec {
                payload_bytes: usize::MAX,
                ..spec
            },
            PanelPoolSpec {
                lookahead_pages: usize::MAX / 2,
                ..spec
            },
        ] {
            assert!(bad.validate().is_err());
        }
    }

    #[test]
    fn slowlink_ring_schedule_sweeps_each_group_once_layer_ahead_in_order() {
        let layers = vec![sources(2), sources(5), sources(6)];
        let experts_per_slot = 4;
        let mut schedule = PanelSchedule::new(layers, experts_per_slot).unwrap();
        let per_layer = EXPERTS / experts_per_slot; // 16 groups
        assert_eq!(schedule.sweep_bytes(), (3 * EXPERTS * (GU + DN)) as u64);
        // Two active + four lookahead pages.
        let mut ring = core(2, 4);
        let issued = schedule.prime(&mut ring, &()).unwrap();
        assert_eq!(
            issued, 6,
            "with no compute layer yet the whole pool prefetches the first host layer"
        );
        assert!(ring
            .backend
            .ops
            .iter()
            .all(|o| !matches!(o, Op::Acquire(_))));
        let mut order = Vec::new();
        let mut prefetched_before_last = false;
        for layer in [2usize, 5, 6] {
            schedule.enter_layer(&mut ring, &(), layer).unwrap();
            for g in 0..per_layer as u32 {
                let t = schedule.require(&mut ring, &(), layer, g).unwrap();
                assert_eq!((t.layer, t.group), (layer, g));
                if layer == 2 && g as usize == per_layer - 1 {
                    prefetched_before_last = schedule.ticket(5, 0).is_some();
                }
                ring.acquire(&(), t).unwrap();
                order.push((layer, g));
                ring.retire(&(), t).unwrap();
                schedule.pump(&mut ring, &()).unwrap();
            }
        }
        assert_eq!(order.len(), 3 * per_layer);
        let stats = ring.stats();
        // Exactly one transfer per group, and every byte submitted once.
        assert_eq!(stats.enqueued, 3 * per_layer as u64);
        assert_eq!(stats.submitted_payload_bytes, schedule.sweep_bytes());
        assert_eq!(stats.discarded, 0);
        // The cross-layer prefetch happened: layer 5's first group was already
        // transferred while layer 2's last group was still to be consumed.
        assert!(
            prefetched_before_last,
            "layer L+1 copies start before layer L finishes"
        );
        assert!(
            schedule.enter_layer(&mut ring, &(), 6).is_err(),
            "no layer after the last"
        );
        assert!(schedule.ticket(5, 3).is_some());
        // Mismatched layer is refused.
        let mut again = PanelSchedule::new(vec![sources(2), sources(5)], 4).unwrap();
        let mut ring2 = core(2, 2);
        again.prime(&mut ring2, &()).unwrap();
        assert!(again.enter_layer(&mut ring2, &(), 5).is_err());
        assert!(PanelSchedule::new(vec![sources(5), sources(2)], 4).is_err());
    }

    #[test]
    fn slowlink_ring_scheduled_next_layer_uses_the_active_slots_the_current_layer_does_not_hold() {
        // 18 pages: layer 1's 16 groups prefetch before any compute, so on entry
        // the two active slots are idle and already carry layer 2's first groups.
        let mut schedule = PanelSchedule::new(vec![sources(1), sources(2)], 4).unwrap();
        let mut ring = core(2, 16);
        assert_eq!(schedule.prime(&mut ring, &()).unwrap(), 16);
        assert!(
            schedule.ticket(2, 0).is_none(),
            "layer 2 is not next before compute starts"
        );
        schedule.enter_layer(&mut ring, &(), 1).unwrap();
        assert_eq!(ring.stats().enqueued, 18);
        assert_eq!(ring.stats().acquired, 0);
        assert!(schedule.ticket(2, 0).is_some() && schedule.ticket(2, 1).is_some());
        assert!(schedule.ticket(2, 2).is_none(), "the page store is full");
        // Layer 1's groups all precede layer 2's: copies are issued in order.
        let ends: Vec<usize> = ring
            .backend
            .ops
            .iter()
            .filter_map(|o| match o {
                Op::End(slot) => Some(*slot),
                _ => None,
            })
            .collect();
        assert_eq!(ends.len(), 18);
        // A pool with no lookahead pages still prefetches through the schedule.
        let mut small = PanelSchedule::new(vec![sources(1), sources(2)], 4).unwrap();
        let mut tiny = core(2, 0);
        assert_eq!(small.prime(&mut tiny, &()).unwrap(), 2);
        // The direct ring API keeps the active slots reserved for the current layer.
        let mut direct = core(2, 3);
        direct.begin_window(&()).unwrap();
        direct.set_layers(None, Some(1)).unwrap();
        let s = sources(1);
        for g in 0..3 {
            assert!(direct.enqueue(&(), &group(&s, g), &s).unwrap().slot >= 2);
        }
        assert_eq!(direct.try_enqueue(&(), &group(&s, 3), &s).unwrap(), None);
    }

    #[test]
    fn slowlink_ring_schedule_pumps_current_layer_before_lookahead() {
        let mut schedule = PanelSchedule::new(vec![sources(1), sources(2)], 4).unwrap();
        let mut ring = core(2, 2);
        schedule.prime(&mut ring, &()).unwrap();
        // Layer 1 groups 0,1 sit in the two lookahead pages.
        schedule.enter_layer(&mut ring, &(), 1).unwrap();
        // Current layer groups took the active slots first; lookahead (layer 2)
        // only got pages that were left.
        assert!(schedule.ticket(1, 2).is_some() && schedule.ticket(1, 3).is_some());
        assert!(schedule.ticket(2, 0).is_none());
        let t = schedule.ticket(1, 0).unwrap();
        ring.acquire(&(), t).unwrap();
        ring.retire(&(), t).unwrap();
        schedule.pump(&mut ring, &()).unwrap();
        // The freed page goes to the current layer's next missing group.
        assert!(schedule.ticket(1, 4).is_some());
        assert!(schedule.ticket(2, 0).is_none());
        assert!(ring.stats().page_full_stalls > 0);
    }

    #[test]
    fn slowlink_ring_copy_starvation_is_counted_from_the_acquire_probe() {
        let mut ring = core(2, 0);
        ring.begin_window(&()).unwrap();
        ring.set_layers(Some(0), None).unwrap();
        let s = sources(0);
        let a = ring.enqueue(&(), &group(&s, 0), &s).unwrap();
        let b = ring.enqueue(&(), &group(&s, 1), &s).unwrap();
        ring.acquire(&(), a).unwrap();
        ring.backend.starved = true;
        ring.acquire(&(), b).unwrap();
        assert_eq!(ring.stats().copy_starved_stalls, 1);
        assert_eq!(ring.stats().acquired, 2);
        assert_eq!(
            ring.stats().acquired_payload_bytes,
            ring.stats().submitted_payload_bytes
        );
    }

    #[test]
    fn slowlink_ring_overlap_summary_counts_cross_layer_intersections() {
        let span = |layer, c0, c1, k: Option<(f64, f64)>| PanelSpanMs {
            layer,
            group: 0,
            bytes: 1,
            copy_start: c0,
            copy_end: c1,
            compute: k,
        };
        let stats = PanelStats {
            page_full_stalls: 2,
            copy_starved_stalls: 3,
            ..PanelStats::default()
        };
        // Layer 1 computes over [10, 30); layer 2's copies run [5, 15) and
        // [20, 40); layer 3's run after everything.
        let spans = vec![
            span(1, 0.0, 5.0, Some((10.0, 20.0))),
            span(2, 5.0, 15.0, Some((40.0, 50.0))),
            span(2, 20.0, 40.0, None),
            span(3, 45.0, 60.0, None),
        ];
        let layers = [LayerSpanMs {
            layer: 1,
            begin: 10.0,
            end: 30.0,
        }];
        let o = summarize_overlap(&spans, &layers, &stats);
        // (L1 compute, L2 copy 5..15): 5 ms; (L1, 20..40): 10 ms.
        assert_eq!(o.cross_layer_intersections, 3);
        assert_eq!(o.cross_layer_overlap_ms, 20.0);
        // Copies sorted by start: 0-5, 5-15, 20-40, 45-60 -> idle 0 + 5 + 5.
        assert_eq!(o.copy_idle_ms, 10.0);
        assert_eq!(o.copy_busy_ms, 5.0 + 10.0 + 20.0 + 15.0);
        assert_eq!(o.compute_busy_ms, 20.0);
        assert_eq!((o.page_full_stalls, o.copy_starved_stalls), (2, 3));
        // Without whole-layer marks the per-group compute spans stand in:
        // layer 1's group computes [10, 20) -> overlaps only copy 5..15.
        let o = summarize_overlap(&spans, &[], &stats);
        assert_eq!(o.cross_layer_intersections, 2);
        assert_eq!(o.cross_layer_overlap_ms, 10.0);
        // No overlap and no data stay zero rather than invented.
        let none = summarize_overlap(&[], &[], &PanelStats::default());
        assert_eq!(none, OverlapSummary::default());
    }

    #[test]
    fn slowlink_ring_timeline_reads_recorded_events_and_reports_overflow() {
        let mut small = spec(1, 1);
        small.timeline_records = 2;
        let mut ring = RingCore::new(small, Mock::new(2)).unwrap();
        ring.begin_window(&()).unwrap();
        ring.set_layers(Some(1), Some(2)).unwrap();
        let (l1, l2) = (sources(1), sources(2));
        ring.mark_layer_begin(&(), 1).unwrap();
        let a = ring.enqueue(&(), &group(&l1, 0), &l1).unwrap();
        let b = ring.enqueue(&(), &group(&l2, 0), &l2).unwrap();
        ring.acquire(&(), a).unwrap();
        ring.retire(&(), a).unwrap();
        ring.mark_layer_end(&(), 1).unwrap();
        ring.set_layers(Some(2), None).unwrap();
        // A third record exceeds the two kept: counted, not silently dropped.
        let c = ring.enqueue(&(), &group(&l2, 1), &l2).unwrap();
        assert_eq!(ring.stats().timeline_overflow, 1);
        ring.acquire(&(), b).unwrap();
        ring.retire(&(), b).unwrap();
        ring.acquire(&(), c).unwrap();
        ring.retire(&(), c).unwrap();
        ring.drain(&()).unwrap();
        let timeline = ring.timeline(&()).unwrap();
        assert_eq!(timeline.spans.len(), 2);
        assert_eq!(timeline.layers.len(), 1);
        assert!(timeline.spans[0].copy_end > timeline.spans[0].copy_start);
        assert!(timeline.spans[0].compute.is_some());
        // Layer 2's copy was enqueued inside layer 1's marked compute interval.
        assert!(timeline.overlap.cross_layer_intersections >= 1);
        assert_eq!(timeline.stats.timeline_overflow, 1);
        // Closing a layer that was never marked is an error (no invented data).
        assert!(ring.mark_layer_end(&(), 9).is_err());
    }

    // ── SP selective schedule on the CPU model of the ring ───────────

    /// `experts` experts back to back, one codec.
    fn sp_sources(layer: usize, experts: usize) -> LayerExpertSources {
        let table = (0..experts)
            .map(|id| {
                let base = HOST + id * (GU + DN);
                Some(ExpertSource {
                    codec: PanelCodec(1),
                    gate_up: ExpertExtent {
                        addr: base,
                        len: GU,
                    },
                    down: ExpertExtent {
                        addr: base + GU,
                        len: DN,
                    },
                })
            })
            .collect();
        // SAFETY: the mock backend never dereferences a source address.
        unsafe { LayerExpertSources::new(layer, HOST, experts * (GU + DN), table, Arc::new(0u8)) }
            .unwrap()
    }

    fn sp_spec(pages: usize, experts: usize, per_slot: usize) -> PanelPoolSpec {
        PanelPoolSpec {
            active_slots: 2,
            lookahead_pages: pages - 2,
            experts_per_slot: per_slot,
            num_experts: experts,
            payload_bytes: per_slot * (GU + DN),
            timeline_records: 64,
            timeline_layers: 8,
        }
    }

    fn sp_ring(pages: usize, experts: usize, per_slot: usize) -> RingCore<Mock> {
        RingCore::new(
            sp_spec(pages, experts, per_slot),
            Mock::sized(pages, experts),
        )
        .unwrap()
    }

    fn ids(group: &[u16]) -> Vec<usize> {
        group.iter().map(|&id| usize::from(id)).collect()
    }

    /// Consume every group of the entered layer, one acquire/retire each.
    fn sp_consume(sp: &mut SpSchedule<'_>, ring: &mut RingCore<Mock>, layer: usize, groups: usize) {
        for group in 0..groups as u32 {
            let ticket = sp.require(ring, &(), layer, group).unwrap();
            ring.acquire(&(), ticket).unwrap();
            ring.retire(&(), ticket).unwrap();
            assert_eq!(sp.pump(ring, &()).unwrap(), 0, "SP never refills a page");
        }
    }

    #[test]
    fn sp_ring_prime_enqueues_nothing_and_declares_no_layer() {
        let sweep = PanelSchedule::new(vec![sources(2), sources(5)], 4).unwrap();
        let mut sp = SpSchedule::new(&sweep);
        let mut ring = core(2, 16);
        assert_eq!(sp.prime(&mut ring, &()).unwrap(), 0);
        assert_eq!(
            ring.backend.ops,
            vec![Op::Base],
            "only the window's base mark"
        );
        assert_eq!(ring.stats().enqueued, 0);
        assert_eq!(ring.stats().submitted_payload_bytes, 0);
        assert_eq!(sp.submitted_bytes(), 0);
        assert_eq!((ring.current, ring.next), (None, None));
        // No layer is current, so the page-order prefetch cannot sneak in.
        let s = sources(2);
        assert!(ring.try_enqueue(&(), &group(&s, 0), &s).is_err());
        assert_eq!(sp.pump(&mut ring, &()).unwrap(), 0);
        assert_eq!(ring.backend.ops, vec![Op::Base]);
        // The sweep schedule on the same sources would have primed 16 groups.
        let mut other = PanelSchedule::new(vec![sources(2), sources(5)], 4).unwrap();
        let mut ring2 = core(2, 16);
        assert_eq!(other.prime(&mut ring2, &()).unwrap(), 16);
    }

    #[test]
    fn sp_ring_enter_layer_enqueues_exactly_the_plan() {
        let sweep = PanelSchedule::new(vec![sources(3), sources(7)], 4).unwrap();
        let mut sp = SpSchedule::new(&sweep);
        let mut ring = core(2, 4);
        sp.prime(&mut ring, &()).unwrap();
        let groups: Vec<Vec<u16>> = vec![vec![1, 5, 9, 13], vec![20, 21, 40, 41], vec![63]];
        sp.plan_layer(3, &groups).unwrap();
        assert_eq!(ring.stats().enqueued, 0, "planning touches no page");
        assert_eq!(ring.backend.ops, vec![Op::Base]);
        assert_eq!(sp.enter_layer(&mut ring, &(), 3).unwrap(), 3);
        assert_eq!(ring.stats().enqueued, 3);
        assert_eq!(ring.stats().page_full_stalls, 0);
        assert_eq!(ring.current, Some(3));
        assert_eq!(sp.planned_groups(3).unwrap().len(), 3);
        for (index, group) in groups.iter().enumerate() {
            let ticket = sp.ticket(3, index as u32).unwrap();
            assert_eq!((ticket.layer, ticket.group), (3, index as u32));
            assert_eq!(sp.require(&mut ring, &(), 3, index as u32).unwrap(), ticket);
            // The page's pointer tables name exactly the planned (sparse) experts.
            let (gate_up, down) = &ring.backend.tables[ticket.slot];
            for table in [gate_up, down] {
                let held: Vec<usize> = table
                    .iter()
                    .enumerate()
                    .filter(|(_, &entry)| entry != 0)
                    .map(|(id, _)| id)
                    .collect();
                assert_eq!(held, ids(group));
            }
        }
        // Nothing beyond the plan exists.
        assert!(sp.ticket(3, 3).is_none());
        assert!(sp.require(&mut ring, &(), 3, 3).is_err());
        assert!(sp.require(&mut ring, &(), 7, 0).is_err());
        sp_consume(&mut sp, &mut ring, 3, 3);
        assert_eq!(ring.stats().enqueued, 3, "retires never refill");
        assert_eq!(ring.stats().discarded, 0);
        // A layer that streams nothing (k = 0) still advances the window.
        sp.plan_layer(7, &[]).unwrap();
        assert_eq!(sp.enter_layer(&mut ring, &(), 7).unwrap(), 0);
        assert_eq!(ring.current, Some(7));
        assert_eq!(ring.stats().enqueued, 3);
    }

    #[test]
    fn sp_ring_submitted_bytes_equal_the_layout_sum() {
        let sweep = PanelSchedule::new(vec![sources(3), sources(7)], 4).unwrap();
        let plans: Vec<(usize, Vec<Vec<u16>>)> = vec![
            (
                3,
                vec![vec![0, 1, 2, 3], vec![10, 12, 14, 16], vec![30, 31]],
            ),
            (7, vec![vec![2, 40, 41, 42], vec![50]]),
        ];
        let mut expected = 0u64;
        let mut experts = 0usize;
        for (layer, groups) in &plans {
            let held = sweep.sources_of(*layer).unwrap();
            for group in groups {
                let layout = plan_group_layout(&ids(group), held).unwrap();
                expected += layout.submitted_bytes() as u64;
                experts += group.len();
            }
        }
        assert_eq!(expected, (experts * (GU + DN)) as u64);
        let mut sp = SpSchedule::new(&sweep);
        let mut ring = core(2, 4);
        sp.prime(&mut ring, &()).unwrap();
        for (layer, groups) in &plans {
            sp.plan_layer(*layer, groups).unwrap();
            assert_eq!(
                sp.enter_layer(&mut ring, &(), *layer).unwrap(),
                groups.len()
            );
            sp_consume(&mut sp, &mut ring, *layer, groups.len());
        }
        let stats = ring.stats();
        assert_eq!(stats.submitted_payload_bytes, expected);
        assert_eq!(sp.submitted_bytes(), expected);
        assert_eq!(stats.acquired_payload_bytes, expected);
        assert_eq!(stats.enqueued, 5);
        assert_eq!((stats.retired, stats.discarded), (5, 0));
        assert!(
            sp.submitted_bytes() < sweep.sweep_bytes(),
            "a selective window submits less than the full sweep"
        );
    }

    #[test]
    fn sp_ring_sixteen_groups_of_thirty_two_enqueue_without_stall() {
        for pages in [16usize, 18] {
            let sweep = PanelSchedule::new(vec![sp_sources(0, 512)], 32).unwrap();
            let mut sp = SpSchedule::new(&sweep);
            let mut ring = sp_ring(pages, 512, 32);
            sp.prime(&mut ring, &()).unwrap();
            let groups: Vec<Vec<u16>> = (0..16)
                .map(|g| (0..32).map(|i| (g * 32 + i) as u16).collect())
                .collect();
            sp.plan_layer(0, &groups).unwrap();
            assert_eq!(sp.enter_layer(&mut ring, &(), 0).unwrap(), 16);
            let stats = ring.stats();
            assert_eq!(stats.enqueued, 16);
            assert_eq!(stats.page_full_stalls, 0, "{pages} pages never stall");
            assert_eq!(stats.submitted_payload_bytes, (512 * (GU + DN)) as u64);
            assert_eq!(ring.free.len(), pages - 16);
            sp_consume(&mut sp, &mut ring, 0, 16);
            assert_eq!(ring.free.len(), pages);
        }
        // Sparse: every other expert, eight groups of 32 ascending experts.
        let sweep = PanelSchedule::new(vec![sp_sources(0, 512)], 32).unwrap();
        let mut sp = SpSchedule::new(&sweep);
        let mut ring = sp_ring(16, 512, 32);
        sp.prime(&mut ring, &()).unwrap();
        let groups: Vec<Vec<u16>> = (0..8)
            .map(|g| (0..32).map(|i| (2 * (g * 32 + i)) as u16).collect())
            .collect();
        sp.plan_layer(0, &groups).unwrap();
        assert_eq!(sp.enter_layer(&mut ring, &(), 0).unwrap(), 8);
        assert_eq!(ring.stats().page_full_stalls, 0);
        assert_eq!(
            ring.stats().submitted_payload_bytes,
            (256 * (GU + DN)) as u64
        );
    }

    #[test]
    fn sp_ring_a_plan_beyond_the_pages_is_refused_before_any_enqueue() {
        let sweep = PanelSchedule::new(vec![sp_sources(0, 512)], 32).unwrap();
        let mut sp = SpSchedule::new(&sweep);
        let mut ring = sp_ring(16, 512, 32);
        sp.prime(&mut ring, &()).unwrap();
        // 17 groups of 30 experts: more groups than the ring's 16 pages.
        let groups: Vec<Vec<u16>> = (0..17)
            .map(|g| (0..30).map(|i| (g * 30 + i) as u16).collect())
            .collect();
        sp.plan_layer(0, &groups).unwrap();
        let error = err_text(sp.enter_layer(&mut ring, &(), 0).unwrap_err());
        assert!(error.contains("17 groups"), "{error}");
        assert_eq!(ring.stats().enqueued, 0);
        assert_eq!(ring.stats().page_full_stalls, 0);
        assert_eq!(ring.backend.ops, vec![Op::Base], "no copy was issued");
        assert!(
            !ring.is_quarantined(),
            "a pre-enqueue refusal is not a failure"
        );
        assert_eq!(ring.current, None);
        assert!(sp.ticket(0, 0).is_none());
        // A new window recovers with a plan that fits.
        sp.prime(&mut ring, &()).unwrap();
        sp.plan_layer(0, &groups[..16]).unwrap();
        assert_eq!(sp.enter_layer(&mut ring, &(), 0).unwrap(), 16);

        // An oversized group for the pool's pages is also refused up front.
        let small = PanelSchedule::new(vec![sources(3)], 4).unwrap();
        let mut sp = SpSchedule::new(&small);
        let mut ring = core(2, 4);
        sp.prime(&mut ring, &()).unwrap();
        sp.plan_layer(3, &[vec![0, 1, 2, 3], vec![4, 5, 6, 7, 8]])
            .unwrap();
        let error = err_text(sp.enter_layer(&mut ring, &(), 3).unwrap_err());
        assert!(error.contains("per page"), "{error}");
        assert_eq!(ring.stats().enqueued, 0);
        assert_eq!(ring.backend.ops, vec![Op::Base]);
        assert!(!ring.is_quarantined());
    }

    #[test]
    fn sp_ring_failure_then_drain_refuses_the_stale_ticket() {
        let sweep = PanelSchedule::new(vec![sources(3), sources(7)], 4).unwrap();
        let mut sp = SpSchedule::new(&sweep);
        let mut ring = core(2, 4);
        sp.prime(&mut ring, &()).unwrap();
        let groups: Vec<Vec<u16>> = vec![vec![0, 1, 2, 3], vec![8, 9, 10, 11], vec![16, 17]];
        sp.plan_layer(3, &groups).unwrap();
        // Begin, Payload, Zero, Tables, End = 5 ops per dense group: fail the
        // second group's copy.
        ring.backend.fail_at = Some(ring.backend.ops.len() + 5);
        assert!(sp.enter_layer(&mut ring, &(), 3).is_err());
        assert!(ring.is_quarantined());
        let first = sp.ticket(3, 0).expect("the group before the failure");
        assert!(sp.ticket(3, 1).is_none());
        assert_eq!(ring.stats().enqueued, 1);
        assert!(err_text(ring.acquire(&(), first).unwrap_err()).contains("quarantined"));
        // Nothing but the drain is accepted while quarantined.
        assert!(sp.prime(&mut ring, &()).is_err());
        ring.drain(&()).unwrap();
        assert!(!ring.is_quarantined());
        assert!(ring.acquire(&(), first).is_err());
        assert!(ring.retire(&(), first).is_err());
        // The next window refills the page under a new epoch: the old ticket
        // stays refused as stale.
        sp.prime(&mut ring, &()).unwrap();
        sp.plan_layer(3, &groups).unwrap();
        assert_eq!(sp.enter_layer(&mut ring, &(), 3).unwrap(), 3);
        let fresh = sp.ticket(3, 0).unwrap();
        assert!(fresh.epoch > first.epoch);
        let error = err_text(ring.acquire(&(), first).unwrap_err());
        assert!(error.contains("stale"), "{error}");
        sp_consume(&mut sp, &mut ring, 3, 3);
        assert_eq!(ring.stats().submitted_payload_bytes, sp.submitted_bytes());
    }

    #[test]
    fn sp_ring_windows_alternate_with_stream_all_windows() {
        let layers = [2usize, 5];
        let mut sweep = PanelSchedule::new(vec![sources(2), sources(5)], 4).unwrap();
        let mut ring = core(2, 16);
        let sp_window =
            |sweep: &PanelSchedule, ring: &mut RingCore<Mock>, plan: &[(usize, Vec<Vec<u16>>)]| {
                let mut sp = SpSchedule::new(sweep);
                assert_eq!(sp.prime(ring, &()).unwrap(), 0);
                let mut groups = 0u64;
                for (layer, planned) in plan {
                    sp.plan_layer(*layer, planned).unwrap();
                    assert_eq!(sp.enter_layer(ring, &(), *layer).unwrap(), planned.len());
                    sp_consume(&mut sp, ring, *layer, planned.len());
                    groups += planned.len() as u64;
                }
                let stats = ring.stats();
                assert_eq!(stats.enqueued, groups, "stats restart with each window");
                assert_eq!(stats.submitted_payload_bytes, sp.submitted_bytes());
                assert_eq!((stats.retired, stats.discarded), (groups, 0));
                sp.submitted_bytes()
            };
        let first = vec![
            (2usize, vec![vec![1u16, 5, 9, 13], vec![20, 21, 40, 41]]),
            (5, vec![vec![63]]),
        ];
        let second = vec![(2usize, vec![vec![0u16, 2, 4, 6]]), (5, vec![])];

        let a = sp_window(&sweep, &mut ring, &first);
        assert_eq!(a, (9 * (GU + DN)) as u64);

        // A stream-all window on the same ring: one full sweep.
        sweep.prime(&mut ring, &()).unwrap();
        for &layer in &layers {
            sweep.enter_layer(&mut ring, &(), layer).unwrap();
            for group in 0..16u32 {
                let ticket = sweep.require(&mut ring, &(), layer, group).unwrap();
                ring.acquire(&(), ticket).unwrap();
                ring.retire(&(), ticket).unwrap();
                sweep.pump(&mut ring, &()).unwrap();
            }
        }
        assert_eq!(ring.stats().submitted_payload_bytes, sweep.sweep_bytes());
        assert_eq!(RingSchedule::submitted_bytes(&sweep), sweep.sweep_bytes());
        assert_eq!(ring.stats().enqueued, 32);

        let b = sp_window(&sweep, &mut ring, &second);
        assert_eq!(b, (4 * (GU + DN)) as u64);
        // And back again: the first plan replays to the same bytes.
        assert_eq!(sp_window(&sweep, &mut ring, &first), a);
    }

    #[test]
    fn sp_ring_plan_and_layer_protocol_refusals_touch_nothing() {
        let mixed = sources_with(3, Arc::new(0u8), |id| if id < 2 { 1 } else { 2 });
        let sweep = PanelSchedule::new(vec![mixed, sources(7)], 4).unwrap();
        let mut sp = SpSchedule::new(&sweep);
        let mut ring = core(2, 4);
        sp.prime(&mut ring, &()).unwrap();
        let too_many: Vec<u16> = (0..33).collect();
        for (layer, bad) in [
            (4usize, vec![vec![0u16]]),               // not a host layer
            (7, vec![vec![5, 3]]),                    // descending inside a group
            (7, vec![vec![1, 2], vec![2, 3]]),        // an expert streamed twice
            (7, vec![vec![4, 5], vec![0, 1]]),        // descending across groups
            (7, vec![vec![]]),                        // empty group
            (7, vec![too_many]),                      // more than a page holds
            (7, vec![vec![0, (EXPERTS + 1) as u16]]), // expert not held
            (3, vec![vec![0, 2]]),                    // two codecs in one group
        ] {
            assert!(sp.plan_layer(layer, &bad).is_err(), "{layer}: {bad:?}");
        }
        assert_eq!(ring.backend.ops, vec![Op::Base]);
        // Refusals leave no plan behind.
        assert!(sp.enter_layer(&mut ring, &(), 3).is_err());
        assert!(sp.planned_groups(3).is_none() && sp.planned_groups(7).is_none());
        // The codec boundary is a legal group boundary.
        sp.plan_layer(3, &[vec![0, 1], vec![2, 3]]).unwrap();
        // One plan at a time; the entered layer is the planned one, once.
        assert!(sp.plan_layer(7, &[vec![1]]).is_err());
        assert!(sp.enter_layer(&mut ring, &(), 7).is_err());
        assert_eq!(sp.enter_layer(&mut ring, &(), 3).unwrap(), 2);
        let live = [sp.ticket(3, 0).unwrap(), sp.ticket(3, 1).unwrap()];
        assert!(sp.enter_layer(&mut ring, &(), 3).is_err());
        assert!(
            sp.plan_layer(3, &[vec![1]]).is_err(),
            "not after the entered layer"
        );
        // The next layer cannot be entered while a page of this one is live;
        // the refusal is not a device failure.
        sp.plan_layer(7, &[vec![1]]).unwrap();
        assert!(sp.enter_layer(&mut ring, &(), 7).is_err());
        assert!(!ring.is_quarantined());
        assert_eq!(ring.current, Some(3));
        for ticket in live {
            ring.acquire(&(), ticket).unwrap();
            ring.retire(&(), ticket).unwrap();
        }
        assert_eq!(sp.enter_layer(&mut ring, &(), 7).unwrap(), 1);
        assert_eq!(ring.current, Some(7));
        // A layer is entered once per window, even after a drain.
        ring.drain(&()).unwrap();
        assert!(sp.enter_layer(&mut ring, &(), 7).is_err());
    }
}
