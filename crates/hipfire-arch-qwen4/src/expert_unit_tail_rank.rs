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
//! holding a byte copy of the same unit. Host-side views, the sealed binding,
//! the chunk (prefill) tables and the unit stage never see a slot.
//!
//! # Window protocol (rank thread; no cross-rank state)
//!
//! * [`UnitTailRank::note_call`] (the row-gather hook, once per MoE layer
//!   call): a stream-ordered D2D of the call's routed ids into the log.
//! * [`UnitTailRank::step_end`] (after the window's forward and its D2H sync):
//!   one D2H of the log, the window's distinct units against the slots live at
//!   its START, then [`crate::expert_unit_tail::TailPolicy::retire`], the touched
//!   tables' upload on the compute stream, and only then the new fills on the
//!   copy stream (it waits for the upload, so no copy can overwrite a slot a
//!   table still names). Completed fills are published at the next retirement.
//!
//! Any device error freezes the tail at its last uploaded table and logs once;
//! it never fails a decode (every table state is valid: canonical, or a live
//! slot with complete bytes).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;
use std::time::Instant;

use hip_bridge::{Event, Stream, HIP_EVENT_DISABLE_TIMING};
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::expert_ep_units::EpUnitPlan;
use crate::expert_unit_tail::{EpUnitTailConfig, FillOp, TailPolicy, UnitKey, WindowUse};

/// Ids one window's log holds (`calls x slots`); a window past this stops
/// logging (counted, never fatal: it only thins the next admission).
const LOG_IDS: usize = 1 << 18;
/// Print a counter line every this many retired windows under the log env.
const LOG_EVERY: u64 = 64;

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
        let seeds = policy.seed(&order);
        let slots = policy.slots();

        // Device state. Any failure frees what was made.
        let mut state = Inner {
            rank,
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
            released: false,
        };
        let built = state.build_device(gpu, slots, &seeds);
        if let Err(error) = built {
            state.release(gpu);
            return Err(format!("tail install on rank {rank}: {error}"));
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

impl Inner {
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

        let retire = self
            .policy
            .retire((core_distinct, core_reads), &window, &completed);
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
        let mut touched: Vec<usize> = touched.into_iter().collect();
        touched.sort_unstable();
        let compute = gpu.active_stream.as_ref();
        for &index in &touched {
            let layer = &self.layers[index];
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

    fn freeze(&mut self, why: &str) {
        if self.frozen.is_none() {
            eprintln!(
                "qwen4 EP rank {}: D2 tail frozen at its last published table: {why}",
                self.rank
            );
            self.frozen = Some(why.to_string());
        }
        self.calls.clear();
        self.log_used = 0;
    }

    fn print_line(&self, tag: &str) {
        let s = self.policy.stats();
        let w = s.windows.max(1) as f64;
        let total = (s.core_reads + s.tail_reads + s.host_reads).max(1) as f64;
        eprintln!(
            "[unit-tail] rank {} {tag} windows {} reads/window {:.1}: core {:.2}% tail {:.2}% host {:.2}% \
             (slot hit {:.2}%, distinct hit {:.2}%) host distinct/window {:.2} ({:.3} MB) \
             fills {} published {} evictions {} skipped {} stale {} live {}/{} late-in-flight {} \
             host-only reads {} log overflow {} | us/window: log {:.0} retire {:.0} upload {:.0} fill {:.0} ({} table uploads){}",
            self.rank,
            s.windows,
            total / w,
            100.0 * s.core_reads as f64 / total,
            100.0 * s.tail_reads as f64 / total,
            100.0 * s.host_reads as f64 / total,
            100.0 * s.slot_hit(),
            100.0 * s.distinct_hit(),
            s.host_distinct as f64 / w,
            s.host_distinct as f64 / w * 2.6112,
            s.fills_issued,
            s.fills_published,
            s.evictions,
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
        );
    }

    fn release(&mut self, gpu: &mut Gpu) {
        if self.released {
            return;
        }
        self.released = true;
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
    use super::full_shared;

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
}
