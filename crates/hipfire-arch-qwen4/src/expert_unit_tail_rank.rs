// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! One EP rank's D2 tail: the device part of [`crate::expert_unit_tail`].
//!
//! Owns, per rank: the tail pool (two blocks laid out like a unit arena), a copy
//! stream with its completion events, the per-window usage log, and the rank's
//! DECODE pointer tables' byte images. Only those tables are ever rewritten: an
//! entry is the unit's canonical pointer (pinned host arena or the static core's
//! VRAM arena, exactly what D1 uploaded) or the address of a completed tail slot
//! holding a byte copy of the same unit. Host-side views and the sealed binding never see a
//! slot. With the prefill flag the chunk (prefill) tables follow the publications of the
//! shared owned units, and the long-prefill unit stage reads the live slots at each staged
//! forward's start ([`UnitTailRank::live_slots`]; `gpu_forward::slab_stage::UnitSlotStage`
//! names them in its stage tables and skips their bytes in its H2D copy). It only ever reads
//! the published map: a slot is overwritten only by a fill issued after a retirement's table
//! upload, which no staged forward straddles.
//!
//! # Window protocol (rank thread; the only cross-rank state is the PEER board)
//!
//! * [`UnitTailRank::note_call`] (the row-gather hook, once per MoE layer
//!   call): a stream-ordered D2D of the call's routed ids into the log.
//! * [`UnitTailRank::step_end`] (after the window's forward and its D2H sync):
//!   one D2H of the log, the window's distinct units against the slots live at
//!   its START, then [`crate::expert_unit_tail::TailPolicy::retire`], the touched
//!   tables' upload on the compute stream, and only then the new fills on the
//!   copy stream (it waits for the upload, so no copy can overwrite a slot a
//!   table still names). Completed fills are published at the next retirement.
//! * With `HIPFIRE_EP_UNIT_TAIL_PEER` (the protocol and its safety argument are the module docs
//!   of [`crate::expert_unit_tail`]) the retirement also posts this rank's publications to the
//!   shared [`PeerBoard`], takes every delta the peers posted so far, applies it to the
//!   full-table image of the peer-owned entries [`peer_shared`] admits, uploads the touched full
//!   tables with its own table uploads (compute stream, before its own fills) and only then
//!   records the ACK the owners release quarantined slots on. Teardown frees the pools with the
//!   mesh, like the static cores peer tables name.
//!
//! Any device error freezes the tail at its last uploaded table and logs once;
//! it never fails a decode (every table state is valid: canonical, or a live
//! slot with complete bytes).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use hip_bridge::{Event, Stream, HIP_EVENT_DISABLE_TIMING};
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::expert_ep_units::EpUnitPlan;
use crate::expert_unit_tail::{
    EpUnitTailConfig, FillOp, PeerBoard, PeerDelta, PeerTake, Retire, TailPolicy, UnitKey,
    WindowUse, QUARANTINE_WINDOWS,
};

/// Ids one window's log holds (`calls x slots`); a window past this stops
/// logging (counted, never fatal: it only thins the next admission).
const LOG_IDS: usize = 1 << 18;
/// Print a counter line every this many retired windows under the log env (fine enough to
/// attribute the cumulative raw counters to one loadgen cell by their `unix_ms` stamps).
const LOG_EVERY: u64 = 16;

/// One owned unit of a managed layer that a tail slot holds at the last retirement: its
/// canonical decode-table entries (the pinned host arena addresses D1 uploaded) and the two slot
/// addresses that the decode table names instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LiveSlot {
    pub expert: usize,
    /// Canonical `[gate_up, down]` device-visible addresses.
    pub canon: [usize; 2],
    /// Slot `[gate_up, down]` addresses (byte copies of the canonical unit).
    pub slot: [usize; 2],
}

/// The units of one layer whose table `image` (u64 LE per expert, as last uploaded) differs from
/// its canonical entries, restricted to the experts `tail_owned` names (the units the tail may
/// hold): exactly the units a slot serves at the last retirement. Ascending by expert.
fn live_slot_units(
    tail_owned: impl Fn(usize) -> bool,
    canon: (&[u64], &[u64]),
    image: (&[u8], &[u8]),
) -> Vec<LiveSlot> {
    let entry = |bytes: &[u8], expert: usize| -> u64 {
        let at = expert * 8;
        u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8-byte table entry"))
    };
    (0..canon.0.len())
        .filter(|&expert| tail_owned(expert))
        .filter_map(|expert| {
            let slot = [entry(image.0, expert), entry(image.1, expert)];
            (slot != [canon.0[expert], canon.1[expert]]).then_some(LiveSlot {
                expert,
                canon: [canon.0[expert] as usize, canon.1[expert] as usize],
                slot: slot.map(|address| address as usize),
            })
        })
        .collect()
}

/// What the tail needs of one decode MoE layer runtime (from the forward).
pub(crate) struct TailLayerInit {
    /// The layer's label (`layer_idx` of its gather calls).
    pub label: usize,
    /// Non-owning aliases of the layer's two decode pointer tables
    /// (`num_experts` u64 each); the forward owns and outlives them.
    pub gate_up_table: GpuTensor,
    pub down_table: GpuTensor,
    /// The entries the tables were uploaded from (canonical pointers).
    pub gate_up_entries: Vec<usize>,
    pub down_entries: Vec<usize>,
    /// The layer's full (chunk / prefill) table, when `HIPFIRE_EP_UNIT_TAIL_PREFILL` is on.
    pub full: Option<TailFullInit>,
}

/// The layer's FULL (chunk / prefill) pointer tables: non-owning aliases and the host entries
/// they were uploaded from. Entries this rank owns that equal the decode table's canonical
/// pointers follow the decode table's slot publications; every other entry is never written.
pub(crate) struct TailFullInit {
    pub gate_up_table: GpuTensor,
    pub down_table: GpuTensor,
    pub gate_up_entries: Vec<usize>,
    pub down_entries: Vec<usize>,
}

/// The rank's mirror of one layer's full tables: byte images as last uploaded and which experts
/// follow the decode table.
struct FullTable {
    table_gate_up: GpuTensor,
    table_down: GpuTensor,
    image_gate_up: Vec<u8>,
    image_down: Vec<u8>,
    /// `shared[expert]`: the full table's canonical entries of `expert` are the decode table's
    /// (so a slot copy of the unit is a valid byte copy for both tables).
    shared: Vec<bool>,
    /// The full-table entries as uploaded at install (the canonical pointers): what a peer-owned
    /// entry returns to when its owner unpublishes the slot, and what [`peer_shared`] compares
    /// the owner's canonical entries with.
    base_gate_up: Vec<u64>,
    base_down: Vec<u64>,
}

/// Which experts of a layer the full table shares with the decode table: the rank's OWNED experts
/// (`e % ranks == rank`) whose full-table entries name exactly the decode table's canonical
/// pointers (same unit bytes, so one slot copy serves both tables). Peer-owned entries name the
/// peer's memory and are never touched; an owned entry that differs (a different residency view)
/// stays canonical in the full table.
fn full_shared(
    rank: usize,
    ranks: usize,
    decode: (&[u64], &[u64]),
    full: (&[u64], &[u64]),
) -> Vec<bool> {
    (0..decode.0.len())
        .map(|expert| {
            expert % ranks == rank
                && decode.0[expert] != 0
                && decode.1[expert] != 0
                && decode.0[expert] == full.0[expert]
                && decode.1[expert] == full.1[expert]
        })
        .collect()
}

/// May `rank`'s full table follow `owner`'s slot for `expert`? Only a PEER-owned entry
/// (`owner != rank`, `expert % ranks == owner`, the static owner of the unit) whose install-time
/// full-table entries `base` are exactly the owner's canonical entries `canon` (same unit bytes,
/// so a copy in the owner's slot is a valid byte copy for this table too), both non-null. Any
/// other entry (an owner's differing residency view, a null pointer, an own entry, which
/// [`full_shared`] covers) is never touched.
fn peer_shared(
    rank: usize,
    ranks: usize,
    owner: usize,
    expert: usize,
    canon: [u64; 2],
    base: [u64; 2],
) -> bool {
    owner != rank
        && owner < ranks
        && expert % ranks == owner
        && canon[0] != 0
        && canon[1] != 0
        && canon == base
}

/// The u64 LE table entry of `expert` in a byte image.
fn image_entry(bytes: &[u8], expert: usize) -> u64 {
    let at = expert * 8;
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8-byte table entry"))
}

/// Apply one peer delta (owner `owner`) to one layer's full-table image: a publication points the
/// entry at the owner's slot, an unpublication returns it to `base` (the canonical entry). Only
/// an entry [`peer_shared`] admits is ever written. `true` when the image changed.
fn apply_peer_delta(
    rank: usize,
    ranks: usize,
    owner: usize,
    base: (&[u64], &[u64]),
    image: (&mut [u8], &mut [u8]),
    delta: &PeerDelta,
) -> bool {
    let expert = delta.unit.1 as usize;
    if expert >= base.0.len()
        || expert >= base.1.len()
        || image.0.len() < (expert + 1) * 8
        || image.1.len() < (expert + 1) * 8
    {
        return false;
    }
    let base_entry = [base.0[expert], base.1[expert]];
    if !peer_shared(rank, ranks, owner, expert, delta.canon, base_entry) {
        return false;
    }
    let want = delta.slot.unwrap_or(base_entry);
    if [image_entry(image.0, expert), image_entry(image.1, expert)] == want {
        return false;
    }
    let at = expert * 8;
    image.0[at..at + 8].copy_from_slice(&want[0].to_le_bytes());
    image.1[at..at + 8].copy_from_slice(&want[1].to_le_bytes());
    true
}

impl FullTable {
    /// [`apply_peer_delta`] on this table's image.
    fn apply_peer(&mut self, rank: usize, ranks: usize, owner: usize, delta: &PeerDelta) -> bool {
        apply_peer_delta(
            rank,
            ranks,
            owner,
            (&self.base_gate_up, &self.base_down),
            (&mut self.image_gate_up, &mut self.image_down),
            delta,
        )
    }
}

/// How one `(layer, expert)` read is served, for this rank.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Not held by this rank (or outside the planned layers).
    Foreign,
    /// Static core / a whole-resident layer: VRAM.
    Core,
    /// A non-core unit of a mixed layer: the tail may hold it.
    Tail,
    /// Host-read unit of a layer the tail does not manage (a wholly host layer).
    HostOnly,
}

struct TailLayer {
    table_gate_up: GpuTensor,
    table_down: GpuTensor,
    canon_gate_up: Vec<u64>,
    canon_down: Vec<u64>,
    /// The byte images of the two tables as last uploaded (u64 LE per expert).
    image_gate_up: Vec<u8>,
    image_down: Vec<u8>,
    /// The layer's full (prefill) tables, mirrored for the experts that share canonical entries.
    full: Option<FullTable>,
}

impl TailLayer {
    /// Point `expert` at `(gate_up, down)` in the decode table and, when its canonical entries
    /// are shared, in the full table.
    fn set(&mut self, expert: usize, gate_up: u64, down: u64) {
        self.image_gate_up[expert * 8..expert * 8 + 8].copy_from_slice(&gate_up.to_le_bytes());
        self.image_down[expert * 8..expert * 8 + 8].copy_from_slice(&down.to_le_bytes());
        if let Some(full) = self.full.as_mut().filter(|full| full.shared[expert]) {
            full.image_gate_up[expert * 8..expert * 8 + 8].copy_from_slice(&gate_up.to_le_bytes());
            full.image_down[expert * 8..expert * 8 + 8].copy_from_slice(&down.to_le_bytes());
        }
    }
}

struct Batch {
    event: Event,
    /// `(slot, generation)` per fill of the batch.
    fills: Vec<(usize, u64)>,
}

struct Inner {
    rank: usize,
    ranks: usize,
    experts: usize,
    trunk_layers: usize,
    /// `layer label -> index into layers` (managed mixed layers only).
    layer_of: Vec<Option<usize>>,
    layers: Vec<TailLayer>,
    /// `[layer * experts + expert]`.
    kinds: Vec<Kind>,
    gate_up_bytes: usize,
    down_bytes: usize,
    pool_gate_up: Option<GpuTensor>,
    pool_down: Option<GpuTensor>,
    copy_stream: Option<Stream>,
    tables_ready: Option<Event>,
    batches: VecDeque<Batch>,
    idle_events: Vec<Event>,
    log: Option<GpuTensor>,
    log_host: Vec<u8>,
    log_used: usize,
    calls: Vec<(u16, u32, u32)>,
    /// Scratch of one retirement: slot reads per `layer * experts + expert` and the indices
    /// touched (so the per-window pass never hashes or clears the whole array).
    reads: Vec<u32>,
    touched: Vec<u32>,
    overflowed: u64,
    policy: TailPolicy,
    /// A device error froze the tail; the message was logged once.
    frozen: Option<String>,
    log_enabled: bool,
    host_only_reads: u64,
    retire_ns: u64,
    log_ns: u64,
    upload_ns: u64,
    fill_ns: u64,
    uploads: u64,
    /// PEER mode: the board shared by every rank of the mesh.
    peer: Option<Arc<PeerBoard>>,
    /// Peer deltas that changed a full-table image / that did not (an unmanaged layer, an entry
    /// [`peer_shared`] does not admit, or an image already in that state).
    peer_applied: u64,
    peer_skipped: u64,
    released: bool,
}

/// The rank's tail (installed in the row-gather hook).
pub(crate) struct UnitTailRank {
    inner: Mutex<Inner>,
}

fn io(error: impl std::fmt::Display) -> String {
    error.to_string()
}

impl UnitTailRank {
    /// Build the rank's tail: the pool, the copy stream and log, seed the pool with
    /// the next-hottest units the static core does not hold and publish them.
    /// `layers` are every decode runtime of the rank's forward; only the plan's
    /// MIXED layers are managed. Fails (never falls back) when the plan has no
    /// ranking, no mixed layer or no unit to hold.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn install(
        gpu: &mut Gpu,
        rank: usize,
        trunk_layers: usize,
        plan: &EpUnitPlan,
        config: EpUnitTailConfig,
        freeze: bool,
        log_enabled: bool,
        layers: Vec<TailLayerInit>,
        peer: Option<Arc<PeerBoard>>,
    ) -> Result<Self, String> {
        use crate::expert_ep_units::EpLayerState;
        let geometry = *plan.geometry();
        let experts = geometry.experts_per_layer;
        let ranks = geometry.ranks;
        let ranking = plan
            .ranking_order()
            .ok_or("the unit plan carries no ranking (a whole-layers plan has no tail)")?;
        if geometry.layers != trunk_layers {
            return Err(format!(
                "the unit plan covers {} layers, the model has {trunk_layers}",
                geometry.layers
            ));
        }
        let (gub, dnb) = (
            geometry.gate_up_bytes as usize,
            geometry.down_bytes as usize,
        );

        // Managed layers: this rank's Mixed layers with a decode runtime.
        let mut layer_of: Vec<Option<usize>> = vec![None; geometry.layers];
        let mut managed: Vec<TailLayer> = Vec::new();
        let (mut full_layers, mut shared_entries) = (0usize, 0usize);
        for init in layers {
            let label = init.label;
            if label >= geometry.layers
                || !matches!(plan.layer_state(rank, label), EpLayerState::Mixed { .. })
            {
                continue;
            }
            if init.gate_up_entries.len() != experts || init.down_entries.len() != experts {
                return Err(format!(
                    "layer {label}: the decode table holds {} / {} entries, the model has {experts} experts",
                    init.gate_up_entries.len(),
                    init.down_entries.len()
                ));
            }
            let canon_gate_up: Vec<u64> = init.gate_up_entries.iter().map(|&p| p as u64).collect();
            let canon_down: Vec<u64> = init.down_entries.iter().map(|&p| p as u64).collect();
            let image = |entries: &[u64]| -> Vec<u8> {
                entries.iter().flat_map(|e| e.to_le_bytes()).collect()
            };
            let (image_gate_up, image_down) = (image(&canon_gate_up), image(&canon_down));
            if init.gate_up_table.buf.size() < experts * 8
                || init.down_table.buf.size() < experts * 8
            {
                return Err(format!(
                    "layer {label}: a decode pointer table is too small"
                ));
            }
            let full = match init.full {
                None => None,
                Some(full) => {
                    if full.gate_up_entries.len() != experts
                        || full.down_entries.len() != experts
                        || full.gate_up_table.buf.size() < experts * 8
                        || full.down_table.buf.size() < experts * 8
                    {
                        return Err(format!(
                            "layer {label}: the full prefill table does not hold {experts} entries"
                        ));
                    }
                    let full_gate_up: Vec<u64> =
                        full.gate_up_entries.iter().map(|&p| p as u64).collect();
                    let full_down: Vec<u64> = full.down_entries.iter().map(|&p| p as u64).collect();
                    let shared = full_shared(
                        rank,
                        ranks,
                        (&canon_gate_up, &canon_down),
                        (&full_gate_up, &full_down),
                    );
                    shared_entries += shared.iter().filter(|&&s| s).count();
                    full_layers += 1;
                    Some(FullTable {
                        table_gate_up: full.gate_up_table,
                        table_down: full.down_table,
                        image_gate_up: image(&full_gate_up),
                        image_down: image(&full_down),
                        shared,
                        base_gate_up: full_gate_up,
                        base_down: full_down,
                    })
                }
            };
            layer_of[label] = Some(managed.len());
            managed.push(TailLayer {
                table_gate_up: init.gate_up_table,
                table_down: init.down_table,
                canon_gate_up,
                canon_down,
                image_gate_up,
                image_down,
                full,
            });
        }
        if managed.is_empty() {
            return Err("the unit plan has no mixed layer on this rank".into());
        }

        // Per-read kinds and the ranking priority of the units the tail may hold.
        let mut kinds = vec![Kind::Foreign; geometry.layers * experts];
        let mut priority: HashMap<UnitKey, u32> = HashMap::new();
        let mut order: Vec<UnitKey> = Vec::new();
        for layer in 0..geometry.layers {
            let managed_layer = layer_of[layer].is_some();
            for expert in (rank..experts).step_by(ranks) {
                kinds[layer * experts + expert] = if plan.is_resident(rank, layer, expert) {
                    Kind::Core
                } else if managed_layer {
                    Kind::Tail
                } else {
                    Kind::HostOnly
                };
            }
        }
        for (pos, &id) in ranking.iter().enumerate() {
            let (layer, expert) = (id as usize / experts, id as usize % experts);
            if layer < geometry.layers && kinds[layer * experts + expert] == Kind::Tail {
                priority.insert((layer as u16, expert as u16), pos as u32);
                order.push((layer as u16, expert as u16));
            }
        }
        if order.is_empty() {
            return Err("no unit of this rank is left for the tail".into());
        }

        let mut policy = TailPolicy::new(config, freeze, priority);
        if peer.is_some() {
            policy.enable_peer_quarantine();
        }
        let seeds = policy.seed(&order);
        let slots = policy.slots();

        // Device state. Any failure frees what was made.
        let mut state = Inner {
            rank,
            ranks,
            experts,
            trunk_layers,
            layer_of,
            layers: managed,
            kinds,
            gate_up_bytes: gub,
            down_bytes: dnb,
            pool_gate_up: None,
            pool_down: None,
            copy_stream: None,
            tables_ready: None,
            batches: VecDeque::new(),
            idle_events: Vec::new(),
            log: None,
            log_host: vec![0u8; LOG_IDS * 4],
            log_used: 0,
            calls: Vec::new(),
            reads: vec![0u32; geometry.layers * experts],
            touched: Vec::new(),
            overflowed: 0,
            policy,
            frozen: None,
            log_enabled,
            host_only_reads: 0,
            retire_ns: 0,
            log_ns: 0,
            upload_ns: 0,
            fill_ns: 0,
            uploads: 0,
            peer,
            peer_applied: 0,
            peer_skipped: 0,
            released: false,
        };
        let built = state.build_device(gpu, slots, &seeds);
        if let Err(error) = built {
            state.release(gpu);
            return Err(format!("tail install on rank {rank}: {error}"));
        }
        if let Some(board) = state.peer.clone() {
            // The warm units are live from the start: post them as retirement 0 (the peers apply
            // them at their first retirement).
            match state.seed_deltas(&seeds) {
                Ok(deltas) => board.post(rank, 0, deltas),
                Err(error) => {
                    state.release(gpu);
                    return Err(format!("tail install on rank {rank}: {error}"));
                }
            }
        }
        eprintln!(
            "qwen4 EP rank {rank}: D2 tail on: {slots} evictable slots ({} MiB pool) over {} mixed layers \
             on static owners (pinned owners are off while the tail is on), \
             {} warm units published, F={} fills/window{}{}",
            ((slots * (gub + dnb)) >> 20),
            state.layers.len(),
            seeds.len(),
            config.fill_per_step,
            if config.boundary_every > 1 {
                format!(", boundary every {} windows", config.boundary_every)
            } else {
                String::new()
            },
            if freeze { ", FROZEN (no admission)" } else { "" }
        );
        if full_layers > 0 {
            eprintln!(
                "qwen4 EP rank {rank}: D2 tail prefill slots on: {full_layers} full (chunk) tables follow the slot \
                 publications of {shared_entries} owned entries (full-table prefill reads a live slot from VRAM, not \
                 the pinned host arena)"
            );
        }
        if state.peer.is_some() {
            eprintln!(
                "qwen4 EP rank {rank}: D2 tail PEER slots on: the full (chunk) tables also follow the slot \
                 publications of the {} peer rank(s) for the peer-owned entries equal to the owner's canonical \
                 ones (a full-table prefill reads a peer's live slot over P2P, not the peer's pinned host arena); \
                 an evicted slot is quarantined until every peer's table dropped it (at most {} slots)",
                ranks - 1,
                QUARANTINE_WINDOWS * config.fill_per_step
            );
        }
        Ok(Self {
            inner: Mutex::new(state),
        })
    }

    /// Row-gather hook: log this call's routed ids (stream-ordered D2D).
    pub(crate) fn note_call(&self, gpu: &mut Gpu, layer_idx: u16, slots: usize, topk: &GpuTensor) {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.note_call(gpu, layer_idx, slots, topk);
    }

    /// End of a decode window (after its forward's D2H sync): retire it.
    pub(crate) fn step_end(&self, gpu: &mut Gpu) {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if inner.frozen.is_some() || inner.released {
            inner.calls.clear();
            inner.log_used = 0;
            return;
        }
        if let Err(error) = inner.step_end(gpu) {
            inner.freeze(&error);
        }
    }

    /// Free the pool, the log, the stream and the events (forward teardown).
    pub(crate) fn release(&self, gpu: &mut Gpu) {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if inner.log_enabled {
            inner.print_line("final");
        }
        inner.release(gpu);
    }
}

impl UnitTailRank {
    /// The units of trunk layer `label` that a tail slot serves as of the LAST RETIREMENT (the
    /// decode table's state: every unit published after its fill completed and not yet evicted),
    /// ascending by expert. Empty for an unmanaged layer or a released tail. The unit stage of a
    /// long prefill forward ([`crate::gpu_forward::slab_stage`]) reads this at the forward's
    /// start; retirements run on the rank thread only after a decode window's forward, never
    /// inside a prefill forward, so the answer holds for the whole forward (the slot lifetime
    /// argument is in `slab_stage`'s `UnitSlotStage` docs).
    pub(crate) fn live_slots(&self, label: usize) -> Vec<LiveSlot> {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.live_slots(label)
    }
}

impl Inner {
    fn live_slots(&self, label: usize) -> Vec<LiveSlot> {
        if self.released {
            return Vec::new();
        }
        let Some(index) = self.layer_of.get(label).copied().flatten() else {
            return Vec::new();
        };
        let layer = &self.layers[index];
        live_slot_units(
            |expert| self.kind(label, expert) == Kind::Tail,
            (&layer.canon_gate_up, &layer.canon_down),
            (&layer.image_gate_up, &layer.image_down),
        )
    }

    fn kind(&self, layer: usize, expert: usize) -> Kind {
        if layer >= self.trunk_layers || expert >= self.experts {
            return Kind::Foreign;
        }
        self.kinds[layer * self.experts + expert]
    }

    fn slot_addresses(&self, slot: usize) -> (u64, u64) {
        let gate_up = self.pool_gate_up.as_ref().expect("pool").buf.as_ptr() as usize
            + slot * self.gate_up_bytes;
        let down =
            self.pool_down.as_ref().expect("pool").buf.as_ptr() as usize + slot * self.down_bytes;
        (gate_up as u64, down as u64)
    }

    fn build_device(
        &mut self,
        gpu: &mut Gpu,
        slots: usize,
        seeds: &[FillOp],
    ) -> Result<(), String> {
        let pad = crate::expert_arena::ARENA_TAIL_PAD_BYTES;
        self.pool_gate_up = Some(
            gpu.zeros(&[slots * self.gate_up_bytes + pad], DType::Raw)
                .map_err(|e| format!("the tail pool's gate/up block ({slots} slots): {e}"))?,
        );
        self.pool_down = Some(
            gpu.zeros(&[slots * self.down_bytes + pad], DType::Raw)
                .map_err(|e| format!("the tail pool's down block ({slots} slots): {e}"))?,
        );
        self.log = Some(
            gpu.zeros(&[LOG_IDS * 4], DType::Raw)
                .map_err(|e| format!("the usage log: {e}"))?,
        );
        // `Gpu::zeros` is a memset on the active stream, and the seed copies below run on the
        // non-blocking copy stream: without this drain a late memset could zero a seeded slot
        // AFTER its copy, and the slot would be published holding zeros (a rank-local wrong
        // read under the full-table prefill route, wrong decode sums on both ranks).
        if let Some(active) = gpu.active_stream.as_ref() {
            gpu.hip.stream_synchronize(active).map_err(io)?;
        } else {
            gpu.hip.device_synchronize().map_err(io)?;
        }
        let stream = gpu.hip.stream_create_non_blocking().map_err(io)?;
        self.copy_stream = Some(stream);
        self.tables_ready = Some(
            gpu.hip
                .event_create_with_flags(HIP_EVENT_DISABLE_TIMING)
                .map_err(io)?,
        );
        // Warm start: copy every seed, wait, then publish them all.
        self.issue_copies(gpu, seeds)?;
        let stream = self.copy_stream.as_ref().expect("stream");
        gpu.hip.stream_synchronize(stream).map_err(io)?;
        let mut touched: HashSet<usize> = HashSet::new();
        for op in seeds {
            let (layer, expert) = (op.unit.0 as usize, op.unit.1 as usize);
            let (gate_up, down) = self.slot_addresses(op.slot);
            let index = self.layer_of[layer].expect("seed layers are managed");
            self.layers[index].set(expert, gate_up, down);
            touched.insert(index);
        }
        // Load time: no kernel reads the tables yet, so a synchronous upload is exact.
        for &index in &touched {
            let layer = &self.layers[index];
            gpu.hip
                .memcpy_htod(
                    &layer
                        .table_gate_up
                        .buf
                        .byte_view(0, layer.image_gate_up.len()),
                    &layer.image_gate_up,
                )
                .map_err(io)?;
            gpu.hip
                .memcpy_htod(
                    &layer.table_down.buf.byte_view(0, layer.image_down.len()),
                    &layer.image_down,
                )
                .map_err(io)?;
            if let Some(full) = layer.full.as_ref() {
                gpu.hip
                    .memcpy_htod(
                        &full
                            .table_gate_up
                            .buf
                            .byte_view(0, full.image_gate_up.len()),
                        &full.image_gate_up,
                    )
                    .map_err(io)?;
                gpu.hip
                    .memcpy_htod(
                        &full.table_down.buf.byte_view(0, full.image_down.len()),
                        &full.image_down,
                    )
                    .map_err(io)?;
            }
        }
        Ok(())
    }

    /// Start the copies of `fills` (canonical host bytes -> pool slot) on the copy
    /// stream, in order.
    fn issue_copies(&self, gpu: &Gpu, fills: &[FillOp]) -> Result<(), String> {
        let stream = self.copy_stream.as_ref().ok_or("no copy stream")?;
        let (pool_gate_up, pool_down) = (
            &self.pool_gate_up.as_ref().ok_or("no pool")?.buf,
            &self.pool_down.as_ref().ok_or("no pool")?.buf,
        );
        for op in fills {
            let (layer, expert) = (op.unit.0 as usize, op.unit.1 as usize);
            let index = self.layer_of[layer].ok_or("a fill names an unmanaged layer")?;
            let canon = &self.layers[index];
            let (src_gate_up, src_down) = (canon.canon_gate_up[expert], canon.canon_down[expert]);
            if src_gate_up == 0 || src_down == 0 {
                return Err(format!("unit ({layer},{expert}) has no canonical source"));
            }
            // SAFETY: a managed unit the static core does not hold lives in its layer's
            // pinned host arena (the canonical entries D1 uploaded name it); that
            // allocation outlives the forward and nothing writes it after load, so the
            // two ranges are valid, immutable host bytes for the copy.
            let (gate_up, down) = unsafe {
                (
                    std::slice::from_raw_parts(
                        src_gate_up as usize as *const u8,
                        self.gate_up_bytes,
                    ),
                    std::slice::from_raw_parts(src_down as usize as *const u8, self.down_bytes),
                )
            };
            gpu.hip
                .memcpy_htod_async(
                    &pool_gate_up.byte_view(op.slot * self.gate_up_bytes, self.gate_up_bytes),
                    gate_up,
                    stream,
                )
                .map_err(io)?;
            gpu.hip
                .memcpy_htod_async(
                    &pool_down.byte_view(op.slot * self.down_bytes, self.down_bytes),
                    down,
                    stream,
                )
                .map_err(io)?;
        }
        Ok(())
    }

    fn note_call(&mut self, gpu: &mut Gpu, layer_idx: u16, slots: usize, topk: &GpuTensor) {
        let layer = layer_idx as usize;
        if self.frozen.is_some()
            || self.released
            || layer >= self.trunk_layers
            || slots == 0
            || topk.buf.size() < slots * 4
        {
            return;
        }
        let Some(log) = self.log.as_ref() else {
            return;
        };
        if self.log_used + slots * 4 > LOG_IDS * 4 {
            self.overflowed += 1;
            return;
        }
        let at = self.log_used;
        let dst = log.buf.byte_view(at, slots * 4);
        let result = match gpu.active_stream.as_ref() {
            Some(stream) => gpu
                .hip
                .memcpy_dtod_async_at(&dst, 0, &topk.buf, 0, slots * 4, stream),
            None => gpu
                .hip
                .memcpy_dtod_async_default_at(&dst, 0, &topk.buf, 0, slots * 4),
        };
        match result {
            Ok(()) => {
                self.calls.push((layer_idx, slots as u32, at as u32));
                self.log_used += slots * 4;
            }
            Err(error) => self.freeze(&format!("usage log copy: {error}")),
        }
    }

    fn step_end(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        if self.calls.is_empty() {
            return Ok(());
        }
        let t0 = Instant::now();
        // The window's forward ended with a D2H; make the stream idle for certain
        // before reading the log and rewriting tables.
        if let Some(stream) = gpu.active_stream.as_ref() {
            gpu.hip.stream_synchronize(stream).map_err(io)?;
        }
        let used = self.log_used;
        {
            let log = self.log.as_ref().ok_or("no log")?;
            gpu.hip
                .memcpy_dtoh(&mut self.log_host[..used], &log.buf.byte_view(0, used))
                .map_err(io)?;
        }
        let calls = std::mem::take(&mut self.calls);
        self.log_used = 0;
        let t_log = Instant::now();

        // The window's units: slot reads per distinct (layer, expert) this rank holds.
        for &(label, slots, at) in &calls {
            let ids = &self.log_host[at as usize..at as usize + slots as usize * 4];
            for raw in ids.chunks_exact(4) {
                let id = i32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]);
                let Ok(expert) = usize::try_from(id) else {
                    continue;
                };
                if self.kind(label as usize, expert) == Kind::Foreign {
                    continue;
                }
                let at = label as usize * self.experts + expert;
                if self.reads[at] == 0 {
                    self.touched.push(at as u32);
                }
                self.reads[at] += 1;
            }
        }
        let mut core_reads = 0u64;
        let mut core_distinct = 0u64;
        let mut host_only = 0u64;
        let mut window: Vec<WindowUse> = Vec::new();
        for &at in &self.touched {
            let at = at as usize;
            let reads = std::mem::take(&mut self.reads[at]);
            let (layer, expert) = (at / self.experts, at % self.experts);
            match self.kinds[at] {
                Kind::Foreign => {}
                Kind::Core => {
                    core_reads += reads as u64;
                    core_distinct += 1;
                }
                Kind::Tail => window.push(WindowUse {
                    unit: (layer as u16, expert as u16),
                    reads,
                }),
                Kind::HostOnly => host_only += reads as u64,
            }
        }
        self.touched.clear();
        self.host_only_reads += host_only;
        window.sort_by_key(|w| w.unit);

        // Completed fills (issued in order on one stream: stop at the first pending).
        let mut completed: Vec<(usize, u64)> = Vec::new();
        while let Some(batch) = self.batches.front() {
            if !gpu.hip.event_query(&batch.event).map_err(io)? {
                break;
            }
            let batch = self.batches.pop_front().expect("front exists");
            completed.extend(batch.fills);
            self.idle_events.push(batch.event);
        }

        let retire = match self.peer.clone() {
            Some(board) => {
                let owner = self.rank;
                self.policy
                    .retire_peer((core_distinct, core_reads), &window, &completed, |m| {
                        board.released_through(owner, m)
                    })
            }
            None => self
                .policy
                .retire((core_distinct, core_reads), &window, &completed),
        };
        // This retirement's index (the retirement counters of all ranks are one clock).
        let j = self.policy.stats().windows;
        let t_retire = Instant::now();

        // Tables first (publish -> slot, evict -> canonical), then the copies.
        let mut touched: HashSet<usize> = HashSet::new();
        for &(unit, slot) in &retire.publish {
            let (gate_up, down) = self.slot_addresses(slot);
            let index = self.layer_of[unit.0 as usize].ok_or("published an unmanaged layer")?;
            self.layers[index].set(unit.1 as usize, gate_up, down);
            touched.insert(index);
        }
        for &unit in &retire.unpublish {
            let index = self.layer_of[unit.0 as usize].ok_or("evicted an unmanaged layer")?;
            let layer = &mut self.layers[index];
            let (gate_up, down) = (
                layer.canon_gate_up[unit.1 as usize],
                layer.canon_down[unit.1 as usize],
            );
            layer.set(unit.1 as usize, gate_up, down);
            touched.insert(index);
        }
        // PEER mode: post this retirement's publications, then take and apply what the peers
        // posted so far to the full tables' peer-owned entries.
        let mut peer_touched: HashSet<usize> = HashSet::new();
        let mut peer_take: Option<PeerTake> = None;
        if let Some(board) = self.peer.clone() {
            board.post(self.rank, j, self.publish_deltas(&retire)?);
            let take = board.take(self.rank);
            if take.newest().is_some_and(|k| k > j) {
                return Err(format!(
                    "peer posts through retirement {:?} reached rank {} at its retirement {j}: \
                     the ranks' windows are out of step",
                    take.newest(),
                    self.rank
                ));
            }
            peer_touched = self.apply_peer(&take);
            peer_take = Some(take);
        }
        let mut order: Vec<usize> = touched.union(&peer_touched).copied().collect();
        order.sort_unstable();
        let compute = gpu.active_stream.as_ref();
        for &index in &order {
            let layer = &self.layers[index];
            if touched.contains(&index) {
                gpu.hip
                    .memcpy_htod_async_on(
                        &layer
                            .table_gate_up
                            .buf
                            .byte_view(0, layer.image_gate_up.len()),
                        &layer.image_gate_up,
                        compute,
                    )
                    .map_err(io)?;
                gpu.hip
                    .memcpy_htod_async_on(
                        &layer.table_down.buf.byte_view(0, layer.image_down.len()),
                        &layer.image_down,
                        compute,
                    )
                    .map_err(io)?;
            }
            if let Some(full) = layer.full.as_ref() {
                gpu.hip
                    .memcpy_htod_async_on(
                        &full
                            .table_gate_up
                            .buf
                            .byte_view(0, full.image_gate_up.len()),
                        &full.image_gate_up,
                        compute,
                    )
                    .map_err(io)?;
                gpu.hip
                    .memcpy_htod_async_on(
                        &full.table_down.buf.byte_view(0, full.image_down.len()),
                        &full.image_down,
                        compute,
                    )
                    .map_err(io)?;
            }
            self.uploads += 1;
        }
        // The ACK: every peer delta this retirement applied is now enqueued on the compute
        // stream ahead of every kernel of the next window. A failed upload returned above and
        // records nothing, so the owners keep the slots this rank may still name quarantined.
        if let (Some(board), Some(take)) = (self.peer.as_ref(), peer_take.as_ref()) {
            board.ack(self.rank, j, take);
        }
        let t_upload = Instant::now();
        if !retire.fills.is_empty() {
            let ready = self.tables_ready.as_ref().ok_or("no tables event")?;
            gpu.hip.event_record(ready, compute).map_err(io)?;
            let stream = self.copy_stream.as_ref().ok_or("no copy stream")?;
            gpu.hip.stream_wait_event(stream, ready).map_err(io)?;
            self.issue_copies(gpu, &retire.fills)?;
            let event = match self.idle_events.pop() {
                Some(event) => event,
                None => gpu
                    .hip
                    .event_create_with_flags(HIP_EVENT_DISABLE_TIMING)
                    .map_err(io)?,
            };
            let stream = self.copy_stream.as_ref().ok_or("no copy stream")?;
            gpu.hip.event_record(&event, Some(stream)).map_err(io)?;
            self.batches.push_back(Batch {
                event,
                fills: retire
                    .fills
                    .iter()
                    .map(|f| (f.slot, f.generation))
                    .collect(),
            });
        }
        let t_end = Instant::now();
        self.log_ns += (t_log - t0).as_nanos() as u64;
        self.retire_ns += (t_retire - t_log).as_nanos() as u64;
        self.upload_ns += (t_upload - t_retire).as_nanos() as u64;
        self.fill_ns += (t_end - t_upload).as_nanos() as u64;
        if self.log_enabled && self.policy.stats().windows % LOG_EVERY == 0 {
            self.print_line("window");
        }
        Ok(())
    }

    /// The deltas of this retirement's publications (a published fill names its slot, an evicted
    /// unit returns to canonical), for the peers.
    fn publish_deltas(&self, retire: &Retire) -> Result<Vec<PeerDelta>, String> {
        let canon = |unit: UnitKey| -> Result<[u64; 2], String> {
            let index = self.layer_of[unit.0 as usize].ok_or("a delta names an unmanaged layer")?;
            let layer = &self.layers[index];
            Ok([
                layer.canon_gate_up[unit.1 as usize],
                layer.canon_down[unit.1 as usize],
            ])
        };
        let mut deltas = Vec::with_capacity(retire.publish.len() + retire.unpublish.len());
        for &(unit, slot) in &retire.publish {
            let (gate_up, down) = self.slot_addresses(slot);
            deltas.push(PeerDelta {
                unit,
                slot: Some([gate_up, down]),
                canon: canon(unit)?,
            });
        }
        for &unit in &retire.unpublish {
            deltas.push(PeerDelta {
                unit,
                slot: None,
                canon: canon(unit)?,
            });
        }
        Ok(deltas)
    }

    /// The deltas that publish the warm start's `seeds`.
    fn seed_deltas(&self, seeds: &[FillOp]) -> Result<Vec<PeerDelta>, String> {
        self.publish_deltas(&Retire {
            publish: seeds.iter().map(|op| (op.unit, op.slot)).collect(),
            ..Retire::default()
        })
    }

    /// Apply the peers' deltas to this rank's full-table images; the layers whose image changed.
    fn apply_peer(&mut self, take: &PeerTake) -> HashSet<usize> {
        let (rank, ranks) = (self.rank, self.ranks);
        let mut touched = HashSet::new();
        for (owner, delta) in &take.deltas {
            let index = self.layer_of.get(delta.unit.0 as usize).copied().flatten();
            let changed = index.and_then(|index| {
                self.layers[index]
                    .full
                    .as_mut()
                    .map(|full| (index, full.apply_peer(rank, ranks, *owner, delta)))
            });
            match changed {
                Some((index, true)) => {
                    touched.insert(index);
                    self.peer_applied += 1;
                }
                _ => self.peer_skipped += 1,
            }
        }
        touched
    }

    fn freeze(&mut self, why: &str) {
        if self.frozen.is_none() {
            eprintln!(
                "qwen4 EP rank {}: D2 tail frozen at its last published table: {why}",
                self.rank
            );
            self.frozen = Some(why.to_string());
            if let Some(board) = self.peer.as_ref() {
                // This rank takes nothing more and never acknowledges again: the owners keep
                // every slot it may still name quarantined.
                board.detach(self.rank);
            }
        }
        self.calls.clear();
        self.log_used = 0;
    }

    fn print_line(&self, tag: &str) {
        let s = self.policy.stats();
        let w = s.windows.max(1) as f64;
        let total = (s.core_reads + s.tail_reads + s.host_reads).max(1) as f64;
        let unit_mb = (self.gate_up_bytes + self.down_bytes) as f64 / 1e6;
        let unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        let peer_note = if self.peer.is_some() {
            format!(
                " quarantined {} (now {}) peer-deltas applied {} skipped {}",
                s.quarantined,
                self.policy
                    .count(crate::expert_unit_tail::SlotPhase::Quarantined),
                self.peer_applied,
                self.peer_skipped
            )
        } else {
            String::new()
        };
        eprintln!(
            "[unit-tail] rank {} {tag} windows {} reads/window {:.1}: core {:.2}% tail {:.2}% host {:.2}% \
             (slot hit {:.2}%, distinct hit {:.2}%) host distinct/window {:.2} ({:.3} MB) \
             fills {} published {} evictions {}{} skipped {} stale {} live {}/{} late-in-flight {} \
             host-only reads {} log overflow {} | us/window: log {:.0} retire {:.0} upload {:.0} fill {:.0} ({} table uploads){} \
             | raw unix_ms={unix_ms} unit_bytes={} core_reads={} tail_reads={} host_reads={} core_distinct={} tail_distinct={} host_distinct={} fills_issued={}",
            self.rank,
            s.windows,
            total / w,
            100.0 * s.core_reads as f64 / total,
            100.0 * s.tail_reads as f64 / total,
            100.0 * s.host_reads as f64 / total,
            100.0 * s.slot_hit(),
            100.0 * s.distinct_hit(),
            s.host_distinct as f64 / w,
            s.host_distinct as f64 / w * unit_mb,
            s.fills_issued,
            s.fills_published,
            s.evictions,
            peer_note,
            s.skipped_full,
            s.stale_completions,
            self.policy.count(crate::expert_unit_tail::SlotPhase::Live),
            self.policy.slots(),
            self.batches.iter().map(|b| b.fills.len()).sum::<usize>(),
            self.host_only_reads,
            self.overflowed,
            self.log_ns as f64 / w / 1e3,
            self.retire_ns as f64 / w / 1e3,
            self.upload_ns as f64 / w / 1e3,
            self.fill_ns as f64 / w / 1e3,
            self.uploads,
            self.frozen
                .as_ref()
                .map_or(String::new(), |why| format!(" FROZEN: {why}")),
            self.gate_up_bytes + self.down_bytes,
            s.core_reads,
            s.tail_reads,
            s.host_reads,
            s.core_distinct,
            s.tail_distinct,
            s.host_distinct,
            s.fills_issued,
        );
    }

    fn release(&mut self, gpu: &mut Gpu) {
        if self.released {
            return;
        }
        self.released = true;
        if let Some(board) = self.peer.as_ref() {
            board.detach(self.rank);
        }
        // Nothing reads a slot or the log after this: drain the copy stream first.
        if let Some(stream) = self.copy_stream.take() {
            let _ = gpu.hip.stream_synchronize(&stream);
            let _ = gpu.hip.stream_destroy(stream);
        }
        for batch in self.batches.drain(..) {
            let _ = gpu.hip.event_destroy(batch.event);
        }
        for event in self.idle_events.drain(..) {
            let _ = gpu.hip.event_destroy(event);
        }
        if let Some(event) = self.tables_ready.take() {
            let _ = gpu.hip.event_destroy(event);
        }
        for tensor in [
            self.pool_gate_up.take(),
            self.pool_down.take(),
            self.log.take(),
        ]
        .into_iter()
        .flatten()
        {
            let _ = gpu.free_tensor(tensor);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_peer_delta, full_shared, image_entry, live_slot_units, peer_shared, LiveSlot,
    };
    use crate::expert_unit_tail::{
        EpUnitTailConfig, PeerBoard, PeerDelta, SlotPhase, TailPolicy, UnitKey, WindowUse,
    };
    use std::collections::HashMap;

    fn image(entries: &[u64]) -> Vec<u8> {
        entries.iter().flat_map(|e| e.to_le_bytes()).collect()
    }

    #[test]
    fn live_slots_are_the_tail_owned_units_whose_image_leaves_canonical() {
        let canon_gu = [100u64, 200, 300, 400];
        let canon_dn = [10u64, 20, 30, 40];
        // Expert 1: published (both entries name a slot). Expert 2: evicted back to canonical.
        // Expert 3: a slot, but not a tail unit (core / foreign): never reported.
        let mut img_gu = canon_gu;
        let mut img_dn = canon_dn;
        img_gu[1] = 7000;
        img_dn[1] = 8000;
        img_gu[3] = 7100;
        img_dn[3] = 8100;
        let live = live_slot_units(
            |expert| expert != 3,
            (&canon_gu, &canon_dn),
            (&image(&img_gu), &image(&img_dn)),
        );
        assert_eq!(
            live,
            [LiveSlot {
                expert: 1,
                canon: [200, 20],
                slot: [7000, 8000],
            }]
        );
    }

    #[test]
    fn no_publication_means_no_live_slot() {
        let canon_gu = [1u64, 2];
        let canon_dn = [3u64, 4];
        let live = live_slot_units(
            |_| true,
            (&canon_gu, &canon_dn),
            (&image(&canon_gu), &image(&canon_dn)),
        );
        assert!(live.is_empty());
    }

    #[test]
    fn full_table_shares_only_owned_experts_with_equal_canonical_entries() {
        // 6 experts, 2 ranks, this rank 1: owned = 1, 3, 5.
        let decode_gu = [10u64, 11, 12, 13, 14, 15];
        let decode_dn = [20u64, 21, 22, 23, 24, 25];
        let mut full_gu = decode_gu;
        let mut full_dn = decode_dn;
        // Expert 3: the full table names a different gate/up pointer; expert 5: a different down.
        full_gu[3] = 99;
        full_dn[5] = 99;
        // Expert 0 (peer-owned) is equal in both tables but must never be shared.
        let shared = full_shared(1, 2, (&decode_gu, &decode_dn), (&full_gu, &full_dn));
        assert_eq!(shared, [false, true, false, false, false, false]);
    }

    #[test]
    fn full_table_never_shares_a_null_entry() {
        let decode_gu = [0u64, 7];
        let decode_dn = [5u64, 0];
        let shared = full_shared(0, 1, (&decode_gu, &decode_dn), (&decode_gu, &decode_dn));
        assert_eq!(shared, [false, false]);
    }

    #[test]
    fn peer_shared_admits_only_a_peer_owned_entry_equal_to_the_owner_canonical() {
        // 2 ranks, this rank 1: rank 0 owns the even experts.
        let canon = [100u64, 200];
        assert!(peer_shared(1, 2, 0, 4, canon, canon));
        // The full entry differs from the owner's canonical entries in either pointer.
        assert!(!peer_shared(1, 2, 0, 4, canon, [100, 201]));
        assert!(!peer_shared(1, 2, 0, 4, canon, [101, 200]));
        // Not a peer: this rank's own entries are `full_shared`'s.
        assert!(!peer_shared(1, 2, 1, 5, canon, canon));
        // The expert is not owned by the delta's owner.
        assert!(!peer_shared(1, 2, 0, 5, canon, canon));
        assert!(!peer_shared(1, 2, 2, 4, canon, canon));
        // A null canonical pointer is never followed.
        assert!(!peer_shared(1, 2, 0, 4, [0, 200], [0, 200]));
        assert!(!peer_shared(1, 2, 0, 4, [100, 0], [100, 0]));
    }

    #[test]
    fn a_peer_delta_never_touches_an_entry_that_differs_from_the_owner_canonical() {
        // 6 experts, 2 ranks, this rank 1: rank 0 owns experts 0, 2 and 4.
        let owner_gu = [10u64, 11, 12, 13, 14, 15];
        let owner_dn = [20u64, 21, 22, 23, 24, 25];
        // This rank's full table (install-time entries): expert 2 names something else than the
        // owner's canonical entries (a different residency view); expert 4 equals them.
        let mut full_gu = owner_gu;
        full_gu[2] = 99;
        let full_dn = owner_dn;
        let mut gu = image(&full_gu);
        let mut dn = image(&full_dn);
        let (start_gu, start_dn) = (gu.clone(), dn.clone());
        let apply = |gu: &mut Vec<u8>, dn: &mut Vec<u8>, owner: usize, delta: PeerDelta| {
            apply_peer_delta(
                1,
                2,
                owner,
                (&full_gu, &full_dn),
                (gu.as_mut_slice(), dn.as_mut_slice()),
                &delta,
            )
        };
        let delta = |expert: u16, slot: Option<[u64; 2]>| PeerDelta {
            unit: (0, expert),
            slot,
            canon: [owner_gu[expert as usize], owner_dn[expert as usize]],
        };
        // The differing entry is never written, by a publication or an unpublication.
        assert!(!apply(&mut gu, &mut dn, 0, delta(2, Some([7000, 8000]))));
        assert!(!apply(&mut gu, &mut dn, 0, delta(2, None)));
        assert_eq!((&gu, &dn), (&start_gu, &start_dn));
        // The equal entry follows its owner: both pointers name the slot, then return to the
        // canonical entries; no neighbour moves.
        assert!(apply(&mut gu, &mut dn, 0, delta(4, Some([7100, 8100]))));
        assert_eq!((image_entry(&gu, 4), image_entry(&dn, 4)), (7100, 8100));
        for expert in [0usize, 1, 2, 3, 5] {
            assert_eq!(image_entry(&gu, expert), image_entry(&start_gu, expert));
            assert_eq!(image_entry(&dn, expert), image_entry(&start_dn, expert));
        }
        assert!(
            !apply(&mut gu, &mut dn, 0, delta(4, Some([7100, 8100]))),
            "idempotent"
        );
        assert!(apply(&mut gu, &mut dn, 0, delta(4, None)));
        assert_eq!((&gu, &dn), (&start_gu, &start_dn));
        // This rank's own experts (1, 3, 5), a delta that claims this rank as its owner, an
        // expert outside the table and a null canonical entry are all ignored.
        assert!(!apply(&mut gu, &mut dn, 0, delta(3, Some([1, 2]))));
        assert!(!apply(&mut gu, &mut dn, 1, delta(3, Some([1, 2]))));
        let beyond = PeerDelta {
            unit: (0, 6),
            slot: Some([1, 2]),
            canon: [16, 26],
        };
        assert!(!apply(&mut gu, &mut dn, 0, beyond));
        let mut null = delta(4, Some([1, 2]));
        null.canon = [0, 0];
        assert!(!apply(&mut gu, &mut dn, 0, null));
        assert_eq!((&gu, &dn), (&start_gu, &start_dn));
    }

    // ---- two-rank lockstep simulation of the board and both policies ----

    const SIM_RANKS: usize = 2;
    const SIM_LAYERS: u16 = 3;
    const SIM_EXPERTS: u16 = 16;

    /// The owner's canonical `[gate_up, down]` entries (its pinned host arena) of a unit.
    fn sim_canon(unit: UnitKey) -> [u64; 2] {
        let id = unit.0 as u64 * SIM_EXPERTS as u64 + unit.1 as u64;
        [
            0x7000_0000_0000 + id * 0x4000,
            0x7000_0000_2000 + id * 0x4000,
        ]
    }

    const SIM_POOL: u64 = 0x1000_0000_0000;
    const SIM_POOL_SPAN: u64 = 0x1_0000_0000;

    /// The addresses of `owner`'s pool slot (VRAM; one slot is 0x4000 bytes apart).
    fn sim_slot_addr(owner: usize, slot: usize) -> [u64; 2] {
        let base = SIM_POOL + owner as u64 * SIM_POOL_SPAN + slot as u64 * 0x4000;
        [base, base + 0x2000]
    }

    fn sim_slot_of(address: u64) -> Option<(usize, usize)> {
        (0..SIM_RANKS).find_map(|owner| {
            let base = SIM_POOL + owner as u64 * SIM_POOL_SPAN;
            (base..base + SIM_POOL_SPAN)
                .contains(&address)
                .then(|| (owner, ((address - base) / 0x4000) as usize))
        })
    }

    /// The install-time full-table entries `reader` holds for a unit: the owner's canonical
    /// entries, except some peer-owned ones that differ (another residency view).
    fn sim_base(reader: usize, unit: UnitKey) -> [u64; 2] {
        let canon = sim_canon(unit);
        if unit.1 as usize % SIM_RANKS != reader && (unit.0 + unit.1) % 5 == 3 {
            [canon[0] + 8, canon[1] + 8]
        } else {
            canon
        }
    }

    fn rand(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    struct SimLayer {
        base_gu: Vec<u64>,
        base_dn: Vec<u64>,
        img_gu: Vec<u8>,
        img_dn: Vec<u8>,
    }

    struct SimRank {
        policy: TailPolicy,
        /// This rank's full-table images, one per layer.
        layers: Vec<SimLayer>,
        /// In-flight fills: `(slot, generation, retirements left)`.
        flight: Vec<(usize, u64, u32)>,
        /// `(owner, k)` -> the retirement at which this rank first applied the owner's post `k`.
        applied_at: HashMap<(usize, u64), u64>,
        covered: Option<u64>,
        /// Slots this rank's policy quarantined -> the retirement they were evicted at.
        quarantined_at: HashMap<usize, u64>,
    }

    #[derive(Default, Debug)]
    struct SimTotals {
        evictions: u64,
        quarantined_refills: u64,
        image_changes: u64,
        published: u64,
    }

    struct Sim {
        ranks: Vec<SimRank>,
        board: PeerBoard,
        /// `pool[owner][slot]`: the unit whose bytes the slot holds, or is being overwritten with.
        pool: Vec<Vec<Option<UnitKey>>>,
        rng: u64,
        quarantine: bool,
        totals: SimTotals,
    }

    impl Sim {
        fn new(t: usize, f: usize, b: u32, quarantine: bool, seed: u64) -> Self {
            let board = PeerBoard::new(SIM_RANKS);
            let (mut ranks, mut pool) = (Vec::new(), Vec::new());
            for r in 0..SIM_RANKS {
                let mut priority = HashMap::new();
                let mut owned: Vec<UnitKey> = Vec::new();
                for layer in 0..SIM_LAYERS {
                    for expert in (r as u16..SIM_EXPERTS).step_by(SIM_RANKS) {
                        priority.insert((layer, expert), layer as u32 * 100 + expert as u32);
                        owned.push((layer, expert));
                    }
                }
                let config = EpUnitTailConfig {
                    tail_units: t,
                    fill_per_step: f,
                    boundary_every: b,
                };
                let mut policy = TailPolicy::new(config, false, priority);
                if quarantine {
                    policy.enable_peer_quarantine();
                }
                let seeds = policy.seed(&owned);
                let mut held = vec![None; t];
                for op in &seeds {
                    held[op.slot] = Some(op.unit);
                }
                // The warm start posts as retirement 0, like `UnitTailRank::install`.
                board.post(
                    r,
                    0,
                    seeds
                        .iter()
                        .map(|op| PeerDelta {
                            unit: op.unit,
                            slot: Some(sim_slot_addr(r, op.slot)),
                            canon: sim_canon(op.unit),
                        })
                        .collect(),
                );
                let layers = (0..SIM_LAYERS)
                    .map(|layer| {
                        let (mut base_gu, mut base_dn) = (Vec::new(), Vec::new());
                        for expert in 0..SIM_EXPERTS {
                            let entry = sim_base(r, (layer, expert));
                            base_gu.push(entry[0]);
                            base_dn.push(entry[1]);
                        }
                        SimLayer {
                            img_gu: image(&base_gu),
                            img_dn: image(&base_dn),
                            base_gu,
                            base_dn,
                        }
                    })
                    .collect();
                ranks.push(SimRank {
                    policy,
                    layers,
                    flight: Vec::new(),
                    applied_at: HashMap::new(),
                    covered: None,
                    quarantined_at: HashMap::new(),
                });
                pool.push(held);
            }
            Self {
                ranks,
                board,
                pool,
                rng: seed | 1,
                quarantine,
                totals: SimTotals::default(),
            }
        }

        /// One rank's retirement of window `w`, in the production order: its policy step, its
        /// post, its fills, then the take / apply / upload / ack of the peers' deltas.
        fn retire(&mut self, r: usize, w: u64) -> Result<(), String> {
            let Sim {
                ranks,
                board,
                pool,
                rng,
                quarantine,
                totals,
            } = self;
            // The window's reads of this rank's own units, skewed to a drifting hot set.
            let owned = SIM_EXPERTS as usize / SIM_RANKS;
            let mut uses: Vec<WindowUse> = Vec::new();
            for _ in 0..8 {
                let layer = (rand(rng) % SIM_LAYERS as u64) as u16;
                let idx = if rand(rng) % 3 != 0 {
                    (rand(rng) as usize % 4 + (w / 40) as usize) % owned
                } else {
                    rand(rng) as usize % owned
                };
                let unit = (layer, (r + idx * SIM_RANKS) as u16);
                let reads = 1 + (rand(rng) % 3) as u32;
                match uses.iter_mut().find(|u| u.unit == unit) {
                    Some(u) => u.reads += reads,
                    None => uses.push(WindowUse { unit, reads }),
                }
            }
            uses.sort_by_key(|u| u.unit);
            let mut completed: Vec<(usize, u64)> = Vec::new();
            ranks[r].flight.retain_mut(|(slot, generation, left)| {
                if *left == 0 {
                    completed.push((*slot, *generation));
                    false
                } else {
                    *left -= 1;
                    true
                }
            });
            if rand(rng) % 17 == 0 {
                completed.push((0, 1 << 40));
            }
            let retire = if *quarantine {
                ranks[r]
                    .policy
                    .retire_peer((0, 0), &uses, &completed, |m| board.released_through(r, m))
            } else {
                ranks[r].policy.retire((0, 0), &uses, &completed)
            };
            let j = ranks[r].policy.stats().windows;
            if j != w {
                return Err(format!("rank {r}: retirement {j} at window {w}"));
            }
            totals.published += retire.publish.len() as u64;
            totals.evictions = ranks.iter().map(|k| k.policy.stats().evictions).sum();
            for slot in 0..ranks[r].policy.slots() {
                if let Some(k) = ranks[r].policy.slot_evicted_at(slot) {
                    ranks[r].quarantined_at.entry(slot).or_insert(k);
                }
            }
            // Post this retirement's publications.
            let mut deltas = Vec::new();
            for &(unit, slot) in &retire.publish {
                deltas.push(PeerDelta {
                    unit,
                    slot: Some(sim_slot_addr(r, slot)),
                    canon: sim_canon(unit),
                });
            }
            for &unit in &retire.unpublish {
                deltas.push(PeerDelta {
                    unit,
                    slot: None,
                    canon: sim_canon(unit),
                });
            }
            board.post(r, j, deltas);
            // The fills this retirement starts: a slot that was quarantined must have been
            // acknowledged by EVERY peer at a retirement strictly before this one.
            for op in &retire.fills {
                if let Some(k) = ranks[r].quarantined_at.remove(&op.slot) {
                    for q in (0..SIM_RANKS).filter(|&q| q != r) {
                        match ranks[q].applied_at.get(&(r, k)) {
                            Some(&applied) if j > applied => {}
                            other => {
                                return Err(format!(
                                    "owner {r}: slot {} evicted at {k} refilled at {j}, peer {q} \
                                     applied the unpublication at {other:?}",
                                    op.slot
                                ))
                            }
                        }
                    }
                    totals.quarantined_refills += 1;
                }
                pool[r][op.slot] = Some(op.unit);
                let latency = (rand(rng) % 4) as u32;
                ranks[r].flight.push((op.slot, op.generation, latency));
            }
            // Take every peer delta posted so far, apply it to the full-table images, "upload",
            // acknowledge.
            let take = board.take(r);
            if let Some(top) = take.newest() {
                if top > j {
                    return Err(format!("rank {r}: peer post {top} ahead of retirement {j}"));
                }
                let owner = (r + 1) % SIM_RANKS;
                let from = ranks[r].covered.map_or(0, |covered| covered + 1);
                for k in from..=top {
                    ranks[r].applied_at.insert((owner, k), j);
                }
                ranks[r].covered = Some(top);
            }
            for (owner, delta) in &take.deltas {
                let layer = &mut ranks[r].layers[delta.unit.0 as usize];
                if apply_peer_delta(
                    r,
                    SIM_RANKS,
                    *owner,
                    (&layer.base_gu, &layer.base_dn),
                    (&mut layer.img_gu, &mut layer.img_dn),
                    delta,
                ) {
                    totals.image_changes += 1;
                }
            }
            board.ack(r, j, &take);
            Ok(())
        }

        /// Every peer-owned full-table entry that left its install-time value names a slot of
        /// its owner that holds exactly that unit's bytes right now (published and not being
        /// overwritten), and an entry that differs from the owner's canonical one never moved.
        fn check(&self) -> Result<(), String> {
            for (q, reader) in self.ranks.iter().enumerate() {
                for (layer, table) in reader.layers.iter().enumerate() {
                    for expert in 0..SIM_EXPERTS as usize {
                        let owner = expert % SIM_RANKS;
                        if owner == q {
                            continue;
                        }
                        let unit = (layer as u16, expert as u16);
                        let entry = [
                            image_entry(&table.img_gu, expert),
                            image_entry(&table.img_dn, expert),
                        ];
                        let base = [table.base_gu[expert], table.base_dn[expert]];
                        if entry == base {
                            continue;
                        }
                        if base != sim_canon(unit) {
                            return Err(format!("rank {q}: unshared entry of {unit:?} moved"));
                        }
                        let Some((slot_owner, slot)) = sim_slot_of(entry[0]) else {
                            return Err(format!("rank {q}: {unit:?} names {entry:x?}"));
                        };
                        if slot_owner != owner || entry != sim_slot_addr(owner, slot) {
                            return Err(format!("rank {q}: {unit:?} names a foreign slot"));
                        }
                        if self.pool[owner][slot] != Some(unit) {
                            return Err(format!(
                                "rank {q} names slot {slot} of owner {owner} for {unit:?}, which \
                                 holds {:?}",
                                self.pool[owner][slot]
                            ));
                        }
                        match self.ranks[owner].policy.slot_phase(slot) {
                            SlotPhase::Live | SlotPhase::Quarantined => {}
                            phase => {
                                return Err(format!(
                                    "rank {q} names slot {slot} of owner {owner} in {phase:?}"
                                ))
                            }
                        }
                    }
                }
            }
            Ok(())
        }
    }

    fn run_sim(
        t: usize,
        f: usize,
        b: u32,
        quarantine: bool,
        seed: u64,
        windows: u64,
    ) -> Result<SimTotals, String> {
        let mut sim = Sim::new(t, f, b, quarantine, seed);
        sim.check()?;
        for w in 1..=windows {
            // Both ranks retire every window; which one goes first is arbitrary.
            let order = if rand(&mut sim.rng) & 1 == 0 {
                [0, 1]
            } else {
                [1, 0]
            };
            for r in order {
                sim.retire(r, w)?;
                sim.check()?;
            }
        }
        Ok(sim.totals)
    }

    /// The lever's safety property over random windows, random copy latency, stale completions
    /// and either rank retiring first: no peer full-table entry ever names a slot that is not
    /// live (or quarantined) with exactly that unit's bytes, in particular no slot is refilled
    /// while a peer image names it, a refill follows every peer's ack by at least one
    /// retirement, and an entry that differs from the owner's canonical one never moves.
    #[test]
    fn two_rank_lockstep_never_refills_a_slot_a_peer_image_names() {
        let mut totals = SimTotals::default();
        for (t, f, b) in [
            (4usize, 2usize, 1u32),
            (6, 3, 1),
            (3, 1, 1),
            (5, 2, 2),
            (10, 4, 1),
            (8, 8, 1),
        ] {
            for seed in [
                0x9e37_79b9_7f4a_7c15u64,
                0x2545_f491_4f6c_dd1d,
                0x1234_5678_9abc_def1,
            ] {
                let run = run_sim(t, f, b, true, seed, 500)
                    .unwrap_or_else(|error| panic!("T={t} F={f} b={b} seed={seed:#x}: {error}"));
                totals.evictions += run.evictions;
                totals.quarantined_refills += run.quarantined_refills;
                totals.image_changes += run.image_changes;
                totals.published += run.published;
            }
        }
        // The runs exercised the protocol: units got published, peers' images followed them,
        // slots were evicted into quarantine and refilled after the acks.
        assert!(
            totals.published > 0 && totals.image_changes > 0,
            "{totals:?}"
        );
        assert!(
            totals.evictions > 0 && totals.quarantined_refills > 0,
            "{totals:?}"
        );
    }

    /// The same runs with the pre-peer eviction (a victim is refilled in the retirement that
    /// evicts it) trip the invariant: the simulation can see the hazard the quarantine closes.
    #[test]
    fn without_quarantine_a_peer_image_names_a_refilled_slot() {
        let broke = (1..=4u64)
            .any(|seed| run_sim(4, 2, 1, false, seed * 0x9e37_79b9_7f4a_7c15, 300).is_err());
        assert!(broke, "the invariant never fired without the quarantine");
    }
}
