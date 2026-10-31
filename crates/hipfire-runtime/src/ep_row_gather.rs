// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next expert-parallel (EP) row all-gather transport.
//!
//! Every rank runs the unchanged single-device program. Routed experts are
//! sharded `e -> rank e % ranks`; the down projection of a decode token (or a
//! few-row verify forward) writes `slots` expanded rows, `rows * k` in slot
//! order (rows of non-owned experts come from a zero dummy expert and are
//! garbage-by-construction). `slots` is passed per call (`1..=` the per-buffer
//! capacity of [`EpRowGatherSpec::slots`]). This transport then fills every
//! non-owned row from its owner before the unchanged canonical combine:
//!
//! 1. [`EpRowGather::publish`] records the rank's producer-complete event for a
//!    buffer index on the stream the producer kernels ran on.
//! 2. [`EpRowGather::gather`] host-waits every peer's event for that buffer
//!    index (device waits require explicit builder opt-in), then launches ONE copy-only kernel that, for
//!    slot `s`, copies row `s` of the owner's peer-mapped buffer into the
//!    rank's own buffer when the owner is not this rank. The owner comes from
//!    an [`EpSlotOwners`]: `ExpertModulo` (`slot_experts[s] % ranks`, kernel
//!    `ep_gather_rows_by_owner`, the static placement) or `Explicit` (an i32
//!    owner per slot written by the same call's owner assignment, kernel
//!    `ep_gather_rows_by_slot_owner`, balanced ownership). Owned rows and
//!    unowned (negative) slots are never written. The copy is uint4
//!    load/store, bit-exact for every pattern (±0, NaN payloads, denormals).
//! 3. [`EpRowGather::exchange`] = publish → host barrier over all ranks → gather,
//!    one host thread per rank.
//!
//! Every rank registers `K` buffers (2..=4, the same `K` on all ranks), selected
//! by a buffer index. Hazard argument: the caller never uses the same buffer
//! index on two consecutive invocations, and all ranks execute the same
//! invocation sequence. Rank `p` overwrites `buffer[b]` only after its gather of
//! the previous invocation, which waited on every peer's producer event of that
//! invocation, which in each peer's stream follows that peer's gather of
//! invocation−2 (the last possible use of `b`) ⇒ no write-after-read. The host
//! barrier guarantees an event is recorded before any peer waits on it, and no
//! rank re-records an event of the same buffer index before every peer has issued
//! its wait. (Two buffers alternating, or a third dedicated one such as an MTP
//! head, both satisfy the rule.)
//!
//! Host-barrier exchanges are eager only: events are created, recorded and
//! waited on the host timeline, which a captured/retained-replay body cannot
//! hold.
//!
//! # Kernel boundaries (#863)
//!
//! With [`EpRowGatherBuilder::kernel_boundaries`] an [`EpRowGather::exchange_layer`]
//! is two kernels on the rank's stream and no host wait, except that a step's
//! first crossing meets the other ranks on the host barrier once, so every
//! rank submits the step together (the device waits are bounded at 500 ms and
//! must not absorb host skew between calls): `ep_boundary_signal_wait`
//! increments the rank's own epoch word with system-scope release and waits,
//! bounded, with system-scope acquire until every rank's word reached that
//! epoch; then the same copy kernel. The epoch is device-derived (every rank
//! runs the same boundary sequence from zeroed words), so the launches are
//! identical every step. The hazard argument above holds with "barrier" read
//! as "boundary": a rank overwrites `buffer[b]` only after its next boundary,
//! which every peer reaches only after its gather of `b`. The host keeps each
//! rank's crossings in a ledger and [`EpRowGather::check_step`] proves the
//! ranks crossed the same boundaries with no buffer rewritten while a peer
//! reads it ([`railgun::mesh`]). A failure poisons every rank's waits
//! ([`EpRowGather::abort`]); a waiter that stops (poison or the 500 ms
//! deadline) leaves its code in the rank's wait block, which the rank's next
//! download reports, failing the step. Epoch words are transport state: they
//! are never part of a model snapshot, and they are freed only by `release`.
//!
//! # Band exchange (output-axis split, #862)
//!
//! [`EpRowGather::exchange_bands`] serves an output-axis split instead: every
//! rank writes its column band of one or more `[rows, m]` outputs into its own
//! buffer (band-compact, `rows * band_bytes` at an offset per output), and the
//! gather copies EVERY rank's band (its own included) into columns
//! `[q * band_bytes, (q + 1) * band_bytes)` of the full output with one pitched
//! copy per rank. Same publish → barrier → wait protocol and the same buffer
//! alternation rule. Each call also carries a geometry tag that must be equal on
//! all ranks, checked after the barrier, so ranks that disagree on the layout
//! fail instead of copying mismatched bands.
//!
//! # Crossing tags and hang diagnostics
//!
//! Every [`EpRowGather::exchange_layer`] crossing carries a [`CrossingTag`]: the
//! call kind ([`GatherKind`]), buffer index, layer, slot count and a per-rank
//! monotonically increasing gather sequence number. The rank stores it in the
//! buffer's tag slot before it publishes and every rank compares all ranks'
//! slots right after the barrier (an atomic store plus `ranks` loads per call,
//! no device work). Ranks that cross the barrier at different gathers (a rank
//! that ran an extra or skipped gather, took another route, used another buffer
//! or disagrees on the layer or row count) therefore fail at once, naming both
//! tags, instead of timing out after 30 s one gather later. A barrier that does
//! time out reports each rank's last entered and last crossed tag and how long
//! ago it last crossed (a rank stalled outside the barrier reads "is between
//! crossings", a rank in another gather reads "is inside a crossing" with its own tag); a wait
//! that passes five seconds prints the same line once while it is still
//! waiting, so a stall that later resolves is still visible.
//!
//! # Lifetime
//!
//! `hip_bridge::Event` has no `Drop`. The producer events are created on each
//! rank's device in [`EpRowGatherBuilder::register`] and are freed by the
//! explicit [`EpRowGather::release`] (one call per rank, on that rank's thread,
//! after every peer is done waiting). Dropping an `EpRowGather` with live events
//! only logs a warning: freeing needs a `HipRuntime` + the right device, which a
//! `Drop` impl does not own.

use hip_bridge::{
    DeviceBuffer, Event, HipError, HipResult, HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED,
    HIP_EVENT_DISABLE_TIMING, HIP_EVENT_RELEASE_TO_SYSTEM,
};
use railgun::mesh;
use rdna_compute::device_wait::DeviceWaitPoison;
use rdna_compute::{Gpu, GpuTensor};
use std::fmt::{self, Write as _};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Default host-barrier timeout.
pub const DEFAULT_BARRIER_TIMEOUT: Duration = Duration::from_secs(30);

/// Bytes of one rank's boundary epoch word allocation (the kernel uses the
/// low u32; ROCm signal memory takes the proven 8-byte allocation).
const EPOCH_BYTES: usize = 8;

/// Largest supported rank count (the kernel takes eight source pointers).
pub const MAX_RANKS: usize = 8;

/// Supported buffers per rank: `MIN_BUFFERS..=MAX_BUFFERS`.
pub const MIN_BUFFERS: usize = 2;
pub const MAX_BUFFERS: usize = 4;

/// Where one gather call takes the owner rank of each slot from. Both are
/// device tensors of `slots` i32 values in slot order, valid on the calling
/// rank's stream before the gather runs.
#[derive(Clone, Copy)]
pub enum EpSlotOwners<'a> {
    /// Static placement: slot `s` is owned by rank `slot_experts[s] % ranks`
    /// (an id outside `[0, n_experts)` owns nothing). The historical kernel
    /// `ep_gather_rows_by_owner`; the second field is `n_experts`.
    ExpertModulo(&'a GpuTensor, usize),
    /// Per-call assigned owners (balanced ownership): slot `s` is owned by rank
    /// `owners[s]` (a negative value owns nothing), written on this rank's
    /// stream by the call's owner assignment. Kernel
    /// `ep_gather_rows_by_slot_owner`.
    Explicit(&'a GpuTensor),
}

/// Shape of one gather: `ranks` participants, a per-buffer capacity of `slots`
/// rows of `row_bytes` each (a call gathers `1..=slots` of them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpRowGatherSpec {
    pub ranks: usize,
    pub slots: usize,
    pub row_bytes: usize,
}

impl EpRowGatherSpec {
    /// Bytes one buffer must hold.
    pub fn buffer_bytes(&self) -> usize {
        self.slots * self.row_bytes
    }

    /// Reject specs the kernel cannot serve. Called at build time, never on the
    /// hot path.
    pub fn validate(&self) -> HipResult<()> {
        if !(2..=MAX_RANKS).contains(&self.ranks) {
            return Err(err(format!(
                "ep_row_gather: ranks must be 2..={MAX_RANKS}, got {}",
                self.ranks
            )));
        }
        if self.slots == 0 {
            return Err(err("ep_row_gather: slots must be >= 1".into()));
        }
        if self.row_bytes == 0 || self.row_bytes % 16 != 0 {
            return Err(err(format!(
                "ep_row_gather: row_bytes must be a non-zero multiple of 16, got {}",
                self.row_bytes
            )));
        }
        let total = self
            .slots
            .checked_mul(self.row_bytes)
            .ok_or_else(|| err("ep_row_gather: slots * row_bytes overflows".into()))?;
        if i32::try_from(self.slots).is_err() || i32::try_from(self.row_bytes / 16).is_err() {
            return Err(err(format!(
                "ep_row_gather: slots/row_bytes exceed i32 kernel arguments ({total} B)"
            )));
        }
        Ok(())
    }

    /// Buffer `index` of `len` bytes must hold every row.
    fn check_buffer_len(&self, index: usize, len: usize) -> HipResult<()> {
        if len < self.buffer_bytes() {
            return Err(err(format!(
                "ep_row_gather: buffer {index} holds {len} B, needs {} B ({} slots x {} B)",
                self.buffer_bytes(),
                self.slots,
                self.row_bytes
            )));
        }
        Ok(())
    }

    /// A call's slot count must be `1..=self.slots` (the per-buffer capacity).
    fn check_call_slots(&self, slots: usize) -> HipResult<()> {
        if slots == 0 || slots > self.slots {
            return Err(err(format!(
                "ep_row_gather: call slots {slots} outside 1..={} (buffer capacity)",
                self.slots
            )));
        }
        Ok(())
    }
}

fn err(msg: String) -> HipError {
    HipError::new(0, &msg)
}

// ───────────────────────────── host barrier ─────────────────────────────

/// Generation-counting spin(+yield) barrier with an abort flag and a timeout.
pub(crate) struct SpinBarrier {
    parties: usize,
    arrived: AtomicUsize,
    generation: AtomicUsize,
    aborted: AtomicBool,
    timeout: Duration,
}

impl SpinBarrier {
    pub(crate) fn new(parties: usize, timeout: Duration) -> Self {
        Self {
            parties,
            arrived: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            aborted: AtomicBool::new(false),
            timeout,
        }
    }

    pub(crate) fn abort(&self) {
        self.aborted.store(true, Ordering::SeqCst);
    }

    pub(crate) fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::SeqCst)
    }

    pub(crate) fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Block until all `parties` arrive. Err when aborted (before or while
    /// waiting) or on timeout; a timeout also aborts the barrier so every other
    /// waiter fails promptly instead of timing out on its own.
    pub(crate) fn wait(&self) -> Result<(), &'static str> {
        self.wait_reporting(self.timeout, &|_| {})
    }

    /// [`Self::wait`] that calls `on_slow(waited)` once when the wait has passed
    /// `slow_after` without the barrier completing (a stalled peer, reported
    /// while it is still stalled). The fast path is unchanged: the clock is only
    /// read after the spin phase.
    pub(crate) fn wait_reporting(
        &self,
        slow_after: Duration,
        on_slow: &dyn Fn(Duration),
    ) -> Result<(), &'static str> {
        if self.is_aborted() {
            return Err("ep_row_gather barrier aborted");
        }
        let generation = self.generation.load(Ordering::Acquire);
        if self.arrived.fetch_add(1, Ordering::AcqRel) + 1 == self.parties {
            self.arrived.store(0, Ordering::Relaxed);
            self.generation.fetch_add(1, Ordering::Release);
            return Ok(());
        }
        let start = Instant::now();
        let mut spins = 0u32;
        let mut reported = false;
        loop {
            if self.generation.load(Ordering::Acquire) != generation {
                return Ok(());
            }
            if self.is_aborted() {
                return Err("ep_row_gather barrier aborted");
            }
            if spins < 2048 {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            let waited = start.elapsed();
            if waited >= self.timeout {
                self.abort();
                return Err("ep_row_gather barrier timed out");
            }
            if !reported && waited >= slow_after {
                reported = true;
                on_slow(waited);
            }
            std::thread::yield_now();
        }
    }
}

// ───────────────────────────── crossing tags ─────────────────────────────

/// A wait that passes this long prints the peers' progress once (capped at half
/// the barrier timeout so short test timeouts report too).
pub(crate) const SLOW_BARRIER_REPORT: Duration = Duration::from_secs(5);

/// One rank's line of a barrier progress report: where it is relative to its
/// last crossing and the tags it last entered and crossed (`0` = none). Shared by
/// the row gather and the tile exchange; `show` renders a transport's tags.
pub(crate) fn describe_rank(
    rank: usize,
    entered: u64,
    crossed: u64,
    crossed_ago_ms: u64,
    doing: &str,
    show: &dyn Fn(u64) -> String,
) -> String {
    let place = if entered == 0 {
        "has entered no crossing"
    } else if entered == crossed {
        "is between crossings"
    } else {
        "is inside a crossing"
    };
    let mut out = format!(
        "rank {rank} {place} (last entered [{}], last crossed [{}]",
        show(entered),
        show(crossed)
    );
    if crossed != 0 {
        let _ = write!(out, " {crossed_ago_ms} ms ago");
    }
    if !doing.is_empty() {
        let _ = write!(out, "; now {doing}");
    }
    out.push(')');
    out
}

/// What a rank's host thread is doing inside a gather (`RankState::phase`): a
/// store per transition, no clock. `PHASE_OUTSIDE` is "not inside a gather".
const PHASE_OUTSIDE: u64 = 0;
const PHASE_PUBLISH: u64 = 1;
const PHASE_BARRIER: u64 = 2;
/// Host-waiting (`event_synchronize`) for a peer's producer event; the peer is
/// in bits 8..16 and the buffer in bits 16..24.
const PHASE_PEER_WAIT: u64 = 3;
const PHASE_COPY: u64 = 4;

fn peer_wait_phase(peer: usize, buffer: usize) -> u64 {
    PHASE_PEER_WAIT | ((peer as u64 & 0xFF) << 8) | ((buffer as u64 & 0xFF) << 16)
}

fn describe_phase(phase: u64) -> String {
    match phase & 0xFF {
        PHASE_PUBLISH => "recording its producer event".to_string(),
        PHASE_BARRIER => "waiting at the barrier".to_string(),
        PHASE_PEER_WAIT => format!(
            "host-waiting for rank {}'s producer event of buffer {} (that rank's stream has \
             not reached the record)",
            (phase >> 8) & 0xFF,
            (phase >> 16) & 0xFF
        ),
        PHASE_COPY => "launching the gather copy".to_string(),
        _ => String::new(),
    }
}

/// The layer of a crossing whose caller did not name one.
pub const NO_LAYER: usize = u16::MAX as usize;

/// The call kind of a [`CrossingTag`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatherKind {
    /// [`EpSlotOwners::ExpertModulo`]: the static `e % ranks` placement.
    ExpertModulo = 1,
    /// [`EpSlotOwners::Explicit`]: per-call assigned (balanced) owners.
    Explicit = 2,
}

impl GatherKind {
    fn of(owners: &EpSlotOwners<'_>) -> Self {
        match owners {
            EpSlotOwners::ExpertModulo(..) => Self::ExpertModulo,
            EpSlotOwners::Explicit(..) => Self::Explicit,
        }
    }
}

/// Identity of one [`EpRowGather::exchange_layer`] crossing, packed into one
/// word so a rank publishes it with a single atomic store: bits 0..4 call kind,
/// 4..8 buffer index, 8..24 layer, 24..40 slots, 40..64 the rank's gather
/// sequence number (modulo 2^24; both ranks wrap together). `0` is "none": no
/// valid crossing has kind 0.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CrossingTag(u64);

impl CrossingTag {
    /// No crossing (never entered).
    pub const NONE: Self = Self(0);

    /// The tag of crossing number `seq` of a rank.
    pub fn new(kind: GatherKind, buffer: usize, layer: usize, slots: usize, seq: u64) -> Self {
        Self(
            (kind as u64)
                | ((buffer as u64 & 0xF) << 4)
                | ((layer.min(NO_LAYER) as u64) << 8)
                | ((slots.min(0xFFFF) as u64) << 24)
                | ((seq & 0xFF_FFFF) << 40),
        )
    }

    pub fn kind(self) -> u8 {
        (self.0 & 0xF) as u8
    }

    pub fn buffer(self) -> usize {
        ((self.0 >> 4) & 0xF) as usize
    }

    pub fn layer(self) -> usize {
        ((self.0 >> 8) & 0xFFFF) as usize
    }

    pub fn slots(self) -> usize {
        ((self.0 >> 24) & 0xFFFF) as usize
    }

    /// The rank's gather sequence number (24 bits).
    pub fn seq(self) -> u64 {
        self.0 >> 40
    }
}

impl fmt::Display for CrossingTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind() {
            0 => return f.write_str("none"),
            1 => "expert_modulo",
            2 => "explicit_owners",
            _ => "unknown_kind",
        };
        write!(f, "{kind} ")?;
        if self.layer() == NO_LAYER {
            f.write_str("layer ?")?;
        } else {
            write!(f, "layer {}", self.layer())?;
        }
        write!(
            f,
            " buffer {} slots {} seq {}",
            self.buffer(),
            self.slots(),
            self.seq()
        )
    }
}

impl fmt::Debug for CrossingTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CrossingTag({self})")
    }
}

// ───────────────────────────── builder ─────────────────────────────

struct RankReg {
    device_id: i32,
    ptrs: Vec<usize>,
    sizes: Vec<usize>,
    events: Vec<Event>,
    boundary: Option<RankBoundary>,
}

/// A rank's kernel-boundary resources (see [`EpRowGatherBuilder::kernel_boundaries`]).
struct RankBoundary {
    /// The rank's monotonic epoch word: signal memory on its device, zero at
    /// registration, incremented only by its own boundary kernels.
    epoch: DeviceBuffer,
    /// Stops this rank's bounded waits when another rank fails.
    poison: DeviceWaitPoison,
}

/// Collects one registration per rank, then builds the shared transport.
pub struct EpRowGatherBuilder {
    spec: EpRowGatherSpec,
    regs: Vec<Option<RankReg>>,
    barrier_timeout: Duration,
    device_waits: bool,
    kernel_boundaries: bool,
}

impl EpRowGatherBuilder {
    /// Opt into stream waits when a launch stream exists. The default is
    /// host waits, independent of streams installed by unrelated features.
    pub fn device_waits(mut self, enabled: bool) -> Self {
        self.device_waits = enabled;
        self
    }

    /// Cross every [`EpRowGather::exchange_layer`] on a device boundary
    /// instead of producer events and the host barrier (#863, see the module
    /// docs): no host wait or host record is left inside the exchange, so the
    /// rank's whole decode step is kernel launches. Must be set before the
    /// first `register`, which then allocates the rank's epoch word and its
    /// wait block and gives the rank its own stream.
    pub fn kernel_boundaries(mut self, enabled: bool) -> Self {
        self.kernel_boundaries = enabled;
        self
    }

    /// Override the host-barrier timeout (default [`DEFAULT_BARRIER_TIMEOUT`]).
    pub fn barrier_timeout(mut self, timeout: Duration) -> Self {
        self.barrier_timeout = timeout;
        self
    }

    /// One call per rank, on any thread, with that rank's `Gpu`. `buffers[b]` is
    /// the rank's expanded-row buffer for buffer index `b` (>= `slots *
    /// row_bytes` bytes; `buffers.len()` = K in 2..=4, identical on every rank).
    /// Creates the rank's producer events (disable-timing, release to system
    /// scope so peer reads observe the producer's writes), one per buffer, on its
    /// device.
    pub fn register(
        &mut self,
        rank: usize,
        gpu: &mut Gpu,
        buffers: &[&GpuTensor],
    ) -> HipResult<()> {
        self.spec.validate()?;
        if rank >= self.spec.ranks {
            return Err(err(format!(
                "ep_row_gather: register rank {rank} out of range (ranks={})",
                self.spec.ranks
            )));
        }
        if self.regs[rank].is_some() {
            return Err(err(format!("ep_row_gather: rank {rank} registered twice")));
        }
        let k = buffers.len();
        if !(MIN_BUFFERS..=MAX_BUFFERS).contains(&k) {
            return Err(err(format!(
                "ep_row_gather: rank {rank} registered {k} buffers, need {MIN_BUFFERS}..={MAX_BUFFERS}"
            )));
        }
        if let Some(expected) = self.regs.iter().flatten().map(|r| r.ptrs.len()).next() {
            if expected != k {
                return Err(err(format!(
                    "ep_row_gather: rank {rank} registered {k} buffers but earlier ranks registered {expected}"
                )));
            }
        }
        let mut ptrs = Vec::with_capacity(k);
        let mut sizes = Vec::with_capacity(k);
        for (index, buf) in buffers.iter().enumerate() {
            self.spec.check_buffer_len(index, buf.buf.size())?;
            let ptr = buf.buf.as_ptr() as usize;
            if ptr == 0 || ptrs.contains(&ptr) {
                return Err(err(
                    "ep_row_gather: buffers must be distinct non-null allocations".into(),
                ));
            }
            ptrs.push(ptr);
            sizes.push(buf.buf.size());
        }
        gpu.bind_thread()?;
        let boundary = if self.kernel_boundaries {
            // A spinning boundary kernel on the legacy null stream would block
            // every other rank's null-stream work on a shared device (logical
            // emulation), so each rank launches on its own stream.
            gpu.ensure_capture_stream()?;
            gpu.prepare_ep_boundary()?;
            let poison = gpu.device_wait_poison()?;
            let epoch = gpu.hip.malloc_signal(EPOCH_BYTES)?;
            if let Err(e) = gpu.hip.memset(&epoch, 0, EPOCH_BYTES) {
                let _ = gpu.hip.free(epoch);
                return Err(e);
            }
            Some(RankBoundary { epoch, poison })
        } else {
            None
        };
        let flags = HIP_EVENT_DISABLE_TIMING | HIP_EVENT_RELEASE_TO_SYSTEM;
        let mut events = Vec::with_capacity(k);
        for _ in 0..k {
            match gpu.hip.event_create_with_flags(flags) {
                Ok(e) => events.push(e),
                Err(e) => {
                    for made in events {
                        let _ = gpu.hip.event_destroy(made);
                    }
                    if let Some(boundary) = boundary {
                        let _ = gpu.hip.free(boundary.epoch);
                    }
                    return Err(e);
                }
            }
        }
        self.regs[rank] = Some(RankReg {
            device_id: gpu.device_id,
            ptrs,
            sizes,
            events,
            boundary,
        });
        Ok(())
    }

    /// Finish: every rank registered, distinct devices.
    pub fn build(self) -> HipResult<Arc<EpRowGather>> {
        self.spec.validate()?;
        let mut ranks = Vec::with_capacity(self.spec.ranks);
        let mut missing = None;
        for (rank, reg) in self.regs.into_iter().enumerate() {
            match reg {
                Some(r) => ranks.push(r),
                None => missing = missing.or(Some(rank)),
            }
        }
        if let Some(rank) = missing {
            return Err(err(format!(
                "ep_row_gather: rank {rank} never registered (builder dropped; registered events leak)"
            )));
        }
        let k = ranks[0].ptrs.len();
        if ranks.iter().any(|r| r.ptrs.len() != k) {
            return Err(err(
                "ep_row_gather: ranks registered different buffer counts".into(),
            ));
        }
        // Logical emulation (`HIPFIRE_EMULATE_GPUS`) deliberately maps several
        // ranks onto one physical device; physical mode needs distinct devices.
        let emulated = crate::config::get().emulate_gpus.is_some();
        for (i, a) in ranks.iter().enumerate() {
            if !emulated && ranks[..i].iter().any(|b| b.device_id == a.device_id) {
                return Err(err(format!(
                    "ep_row_gather: two ranks share device {}",
                    a.device_id
                )));
            }
        }
        let kernel_boundaries = self.kernel_boundaries;
        let states = ranks
            .into_iter()
            .map(|r| RankState {
                device_id: r.device_id,
                tags: r.ptrs.iter().map(|_| AtomicU64::new(0)).collect(),
                ptrs: r.ptrs,
                sizes: r.sizes,
                events: r.events.into_iter().map(|e| Mutex::new(Some(e))).collect(),
                epoch_ptr: r.boundary.as_ref().map_or(0, |b| b.epoch.as_ptr() as usize),
                boundary: Mutex::new(r.boundary),
                ledger: Mutex::new(Vec::new()),
                peers_enabled: AtomicBool::new(false),
                calls: AtomicU64::new(0),
                entered: AtomicU64::new(0),
                crossed: AtomicU64::new(0),
                crossed_at_ms: AtomicU64::new(0),
                phase: AtomicU64::new(PHASE_OUTSIDE),
            })
            .collect();
        Ok(Arc::new(EpRowGather {
            spec: self.spec,
            buffers: k,
            barrier: SpinBarrier::new(self.spec.ranks, self.barrier_timeout),
            ranks: states,
            device_waits: self.device_waits,
            kernel_boundaries,
            started: Instant::now(),
        }))
    }
}

// ───────────────────────────── transport ─────────────────────────────

struct RankState {
    device_id: i32,
    /// Device addresses of this rank's buffers (valid on every peer after
    /// `enable_peers`; ROCm uses one unified address space).
    ptrs: Vec<usize>,
    /// Byte sizes of the same buffers (the band gather's copy-source bounds).
    sizes: Vec<usize>,
    /// Band-exchange geometry tag per buffer index. A rank rewrites the tag of
    /// buffer `b` only at the next invocation on `b`, after the barrier every
    /// peer passes only once it has checked this one.
    tags: Vec<AtomicU64>,
    /// Producer-complete events, one per buffer index. `None` after `release`.
    /// The `Mutex` makes the type `Sync` (`Event` is `Send` only); it is
    /// uncontended by protocol (record and wait of one index never overlap).
    events: Vec<Mutex<Option<Event>>>,
    /// Device address of the rank's epoch word (kernel boundaries; else 0).
    epoch_ptr: usize,
    /// The rank's kernel-boundary resources; `None` without kernel boundaries
    /// and after `release`.
    boundary: Mutex<Option<RankBoundary>>,
    /// The rank's boundary ledger since the last [`EpRowGather::check_step`].
    ledger: Mutex<Vec<mesh::Event>>,
    peers_enabled: AtomicBool,
    /// Crossings this rank has entered (its next gather sequence number).
    calls: AtomicU64,
    /// Tag of the last crossing this rank entered ([`CrossingTag::NONE`] before the first).
    entered: AtomicU64,
    /// Tag of the last crossing this rank completed (barrier passed, tags agreed).
    crossed: AtomicU64,
    /// When `crossed` happened, in milliseconds since the transport was built.
    crossed_at_ms: AtomicU64,
    /// What this rank's host thread is doing inside a gather (`PHASE_*`).
    phase: AtomicU64,
}

/// One output of [`EpRowGather::exchange_bands`]: every rank's band of `rows`
/// rows of `band_bytes` sits band-compact at `offset` in its buffer; the
/// gather writes rank `q`'s band to bytes `[q * band_bytes, (q + 1) *
/// band_bytes)` of each `ranks * band_bytes` row of `dst`.
pub struct EpBand<'a> {
    pub offset: usize,
    pub band_bytes: usize,
    pub rows: usize,
    pub dst: &'a GpuTensor,
}

/// Shared EP row-gather transport (`Arc<EpRowGather>`, one clone per rank thread).
pub struct EpRowGather {
    spec: EpRowGatherSpec,
    buffers: usize,
    ranks: Vec<RankState>,
    barrier: SpinBarrier,
    device_waits: bool,
    kernel_boundaries: bool,
    /// Transport build time: the origin of `RankState::crossed_at_ms`.
    started: Instant,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl EpRowGather {
    pub fn builder(spec: EpRowGatherSpec) -> EpRowGatherBuilder {
        EpRowGatherBuilder {
            spec,
            regs: (0..spec.ranks.min(MAX_RANKS)).map(|_| None).collect(),
            barrier_timeout: DEFAULT_BARRIER_TIMEOUT,
            device_waits: false,
            kernel_boundaries: false,
        }
    }

    /// Whether exchanges cross on device boundaries (see
    /// [`EpRowGatherBuilder::kernel_boundaries`]).
    pub fn kernel_boundaries(&self) -> bool {
        self.kernel_boundaries
    }

    pub fn ranks(&self) -> usize {
        self.spec.ranks
    }

    pub fn spec(&self) -> EpRowGatherSpec {
        self.spec
    }

    /// Buffers registered per rank (K).
    pub fn buffers(&self) -> usize {
        self.buffers
    }

    fn check(&self, gpu: &Gpu, rank: usize, buffer: usize) -> HipResult<()> {
        if rank >= self.spec.ranks || buffer >= self.buffers {
            return Err(err(format!(
                "ep_row_gather: rank {rank} / buffer {buffer} out of range (ranks={}, buffers={})",
                self.spec.ranks, self.buffers
            )));
        }
        if gpu.device_id != self.ranks[rank].device_id {
            return Err(err(format!(
                "ep_row_gather: rank {rank} is device {} but was given a Gpu on device {}",
                self.ranks[rank].device_id, gpu.device_id
            )));
        }
        Ok(())
    }

    /// Enable P2P from this rank's device to every other registered device
    /// (idempotent; call on the rank's thread AFTER every peer-read allocation
    /// exists — ROCm does not map later allocations).
    pub fn enable_peers(&self, rank: usize, gpu: &mut Gpu) -> HipResult<()> {
        self.check(gpu, rank, 0)?;
        gpu.bind_thread()?;
        for (peer, state) in self.ranks.iter().enumerate() {
            // Same physical device (logical emulation): no peer mapping needed.
            if peer == rank || state.device_id == gpu.device_id {
                continue;
            }
            if !gpu.hip.can_access_peer(gpu.device_id, state.device_id)? {
                return Err(err(format!(
                    "ep_row_gather: device {} cannot access peer device {} (rank {rank} -> {peer})",
                    gpu.device_id, state.device_id
                )));
            }
            match gpu.hip.enable_peer_access(state.device_id) {
                Ok(()) => {}
                Err(e) if e.code == HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED => {}
                Err(e) => return Err(e),
            }
        }
        self.ranks[rank]
            .peers_enabled
            .store(true, Ordering::Release);
        Ok(())
    }

    /// Record the rank's producer-complete event for `buffer` on `gpu`'s launch
    /// stream (`gpu.active_stream`, or the null stream when none — it MUST be the
    /// stream the producer kernels used).
    pub fn publish(&self, gpu: &mut Gpu, rank: usize, buffer: usize) -> HipResult<()> {
        self.check(gpu, rank, buffer)?;
        gpu.bind_thread()?;
        let guard = lock(&self.ranks[rank].events[buffer]);
        let event = guard
            .as_ref()
            .ok_or_else(|| err("ep_row_gather: publish after release".into()))?;
        gpu.hip.event_record(event, gpu.active_stream.as_ref())
    }

    /// Wait every peer's `buffer` event, then ONE copy-only kernel: for
    /// slot `s`, `owner` is `slot_owners`' owner of `s` (see [`EpSlotOwners`]);
    /// if `owner != rank` copy row `s` of the owner's `buffer[buffer]` into this
    /// rank's `buffer[buffer]`. Owned slots (and slots with a negative id or
    /// owner) are never written. Bit-exact.
    ///
    /// Host waits are the default even with an active stream. A caller that
    /// opts into `EpRowGatherBuilder::device_waits` uses stream waits when a
    /// stream exists. Events must already be recorded (publish-all first).
    pub fn gather(
        &self,
        gpu: &mut Gpu,
        rank: usize,
        buffer: usize,
        slot_owners: EpSlotOwners<'_>,
        slots: usize,
    ) -> HipResult<()> {
        self.check(gpu, rank, buffer)?;
        self.spec.check_call_slots(slots)?;
        self.wait_peers(gpu, rank, buffer)?;
        self.copy_rows(gpu, rank, buffer, slot_owners, slots)
    }

    /// The gather's copy kernel alone (callers have ordered the peers'
    /// producers before it).
    fn copy_rows(
        &self,
        gpu: &mut Gpu,
        rank: usize,
        buffer: usize,
        slot_owners: EpSlotOwners<'_>,
        slots: usize,
    ) -> HipResult<()> {
        self.ranks[rank].phase.store(PHASE_COPY, Ordering::Relaxed);
        let mut src = [0usize; MAX_RANKS];
        for (peer, state) in self.ranks.iter().enumerate() {
            src[peer] = state.ptrs[buffer];
        }
        let launched = match slot_owners {
            EpSlotOwners::ExpertModulo(slot_experts, n_experts) => gpu.ep_gather_rows_by_owner(
                slot_experts,
                src,
                self.ranks[rank].ptrs[buffer],
                slots,
                self.spec.ranks,
                rank,
                self.spec.row_bytes / 16,
                n_experts,
            ),
            EpSlotOwners::Explicit(owners) => gpu.ep_gather_rows_by_slot_owner(
                owners,
                src,
                self.ranks[rank].ptrs[buffer],
                slots,
                self.spec.ranks,
                rank,
                self.spec.row_bytes / 16,
            ),
        };
        self.ranks[rank]
            .phase
            .store(PHASE_OUTSIDE, Ordering::Relaxed);
        launched
    }

    /// `publish` → host barrier over all ranks (same round) → `gather`. One host
    /// thread per rank. A failure on this rank poisons the barrier so peers
    /// return Err instead of waiting out the timeout. The crossing carries no
    /// layer ([`NO_LAYER`]); production callers use [`Self::exchange_layer`].
    pub fn exchange(
        &self,
        gpu: &mut Gpu,
        rank: usize,
        buffer: usize,
        slot_owners: EpSlotOwners<'_>,
        slots: usize,
    ) -> HipResult<()> {
        self.exchange_layer(gpu, rank, buffer, slot_owners, slots, NO_LAYER)
    }

    /// [`Self::exchange`] for the gather of `layer`: the crossing is tagged
    /// (call kind, buffer, `layer`, `slots`, this rank's gather sequence number)
    /// and the tags of all ranks must be equal after the barrier (see the module
    /// docs); a disagreement aborts the transport and names both tags.
    pub fn exchange_layer(
        &self,
        gpu: &mut Gpu,
        rank: usize,
        buffer: usize,
        slot_owners: EpSlotOwners<'_>,
        slots: usize,
        layer: usize,
    ) -> HipResult<()> {
        if self.kernel_boundaries {
            let result = self.exchange_on_boundary(gpu, rank, buffer, slot_owners, slots, layer);
            if result.is_err() {
                self.abort();
            }
            return result;
        }
        self.spec.check_call_slots(slots)?;
        if let Err(e) = self.check(gpu, rank, buffer) {
            self.abort();
            return Err(e);
        }
        let tag = self.enter(rank, GatherKind::of(&slot_owners), buffer, layer, slots);
        if let Err(e) = self.publish(gpu, rank, buffer) {
            self.abort();
            return Err(e);
        }
        self.cross(rank, tag)?;
        if let Err(e) = self.gather(gpu, rank, buffer, slot_owners, slots) {
            self.abort();
            return Err(e);
        }
        Ok(())
    }

    /// [`Self::exchange_layer`] on a device boundary: one boundary kernel
    /// (publish this rank's epoch, wait every rank's) and the copy kernel, both
    /// on the rank's stream, and the crossing appended to the rank's ledger.
    /// Only a step's first crossing meets the other ranks on the host (paired
    /// submission): the boundary wait is bounded at 500 ms, so ranks must start
    /// submitting a step together rather than with whatever host skew the
    /// previous call left. Later crossings never wait on the host;
    /// [`Self::check_step`] compares the ledgers.
    fn exchange_on_boundary(
        &self,
        gpu: &mut Gpu,
        rank: usize,
        buffer: usize,
        slot_owners: EpSlotOwners<'_>,
        slots: usize,
        layer: usize,
    ) -> HipResult<()> {
        self.spec.check_call_slots(slots)?;
        self.check(gpu, rank, buffer)?;
        if !self.ranks[rank].peers_enabled.load(Ordering::Acquire) {
            return Err(err(format!(
                "ep_row_gather: gather on rank {rank} before enable_peers"
            )));
        }
        if lock(&self.ranks[rank].boundary).is_none() {
            return Err(err("ep_row_gather: boundary after release".into()));
        }
        if lock(&self.ranks[rank].ledger).is_empty() {
            self.barrier.wait().map_err(|message| err(message.into()))?;
        }
        let tag = self.enter(rank, GatherKind::of(&slot_owners), buffer, layer, slots);
        let mut epochs = [0usize; MAX_RANKS];
        for (peer, state) in self.ranks.iter().enumerate() {
            epochs[peer] = state.epoch_ptr;
        }
        gpu.ep_boundary_signal_wait(epochs, self.spec.ranks, rank)?;
        self.copy_rows(gpu, rank, buffer, slot_owners, slots)?;
        lock(&self.ranks[rank].ledger).extend([
            mesh::Event::Write {
                buffer,
                region: mesh::Region::All,
            },
            mesh::Event::Boundary { tag: tag.0 },
            mesh::Event::PeerRead { buffer },
            mesh::Event::Write {
                buffer,
                region: mesh::Region::Unowned,
            },
        ]);
        let state = &self.ranks[rank];
        state
            .crossed_at_ms
            .store(self.started.elapsed().as_millis() as u64, Ordering::Relaxed);
        state.crossed.store(tag.0, Ordering::Release);
        Ok(())
    }

    /// Check and clear every rank's boundary ledger (kernel boundaries; a
    /// no-op otherwise). Call on one thread once every rank's host work for
    /// the step has returned: [`railgun::mesh::check`] must find the same
    /// boundaries in the same order on every rank and no buffer rewritten in
    /// the epoch a peer reads it. A refusal aborts the transport.
    pub fn check_step(&self) -> HipResult<()> {
        if !self.kernel_boundaries {
            return Ok(());
        }
        let ledgers: Vec<Vec<mesh::Event>> = self
            .ranks
            .iter()
            .map(|state| std::mem::take(&mut *lock(&state.ledger)))
            .collect();
        mesh::check(&ledgers).map(drop).map_err(|refusal| {
            self.abort();
            err(format!("ep_row_gather: {refusal}"))
        })
    }

    /// Open this rank's next crossing: take its gather sequence number and
    /// publish the tag (buffer slot for the post-barrier comparison, `entered`
    /// for the progress report). `buffer` is validated by the caller.
    fn enter(
        &self,
        rank: usize,
        kind: GatherKind,
        buffer: usize,
        layer: usize,
        slots: usize,
    ) -> CrossingTag {
        let state = &self.ranks[rank];
        let seq = state.calls.fetch_add(1, Ordering::Relaxed);
        let tag = CrossingTag::new(kind, buffer, layer, slots, seq);
        state.tags[buffer].store(tag.0, Ordering::Release);
        state.entered.store(tag.0, Ordering::Release);
        state.phase.store(PHASE_PUBLISH, Ordering::Relaxed);
        tag
    }

    /// The barrier of crossing `tag`, then the tag comparison. A peer's slot of
    /// this buffer is rewritten only at that peer's next invocation on the same
    /// buffer, which needs the barrier of the invocation in between, which this
    /// rank enters only after this check, so a mismatch is a real divergence.
    fn cross(&self, rank: usize, tag: CrossingTag) -> HipResult<()> {
        self.ranks[rank]
            .phase
            .store(PHASE_BARRIER, Ordering::Relaxed);
        let slow_after = SLOW_BARRIER_REPORT.min(self.barrier.timeout() / 2);
        let waited = self.barrier.wait_reporting(slow_after, &|waited| {
            eprintln!(
                "ep_row_gather: rank {rank} still waiting after {waited:.1?} at [{tag}]; {}",
                self.progress_report()
            );
        });
        if let Err(message) = waited {
            if message.contains("timed out") {
                let text = format!(
                    "{message}: rank {rank} waited {:.1?} at [{tag}]; {}",
                    self.barrier.timeout(),
                    self.progress_report()
                );
                eprintln!("ep_row_gather: {text}");
                return Err(err(text));
            }
            return Err(err(message.into()));
        }
        let buffer = tag.buffer();
        for (peer, state) in self.ranks.iter().enumerate() {
            let theirs = CrossingTag(state.tags[buffer].load(Ordering::Acquire));
            if theirs != tag {
                self.abort();
                let text = format!(
                    "ep_row_gather: ranks crossed the barrier at different gathers: rank {rank} \
                     entered [{tag}] but rank {peer}'s slot for buffer {buffer} holds [{theirs}] \
                     (rank {peer} last entered [{}]; {})",
                    CrossingTag(state.entered.load(Ordering::Acquire)),
                    self.progress_report()
                );
                eprintln!("{text}");
                return Err(err(text));
            }
        }
        let mine = &self.ranks[rank];
        mine.crossed_at_ms
            .store(self.started.elapsed().as_millis() as u64, Ordering::Relaxed);
        mine.crossed.store(tag.0, Ordering::Release);
        Ok(())
    }

    /// Every rank's last entered / last crossed tag and how long ago it last
    /// crossed (one line; a rank not inside a crossing is "between crossings").
    fn progress_report(&self) -> String {
        let now = self.started.elapsed().as_millis() as u64;
        let show = |tag: u64| CrossingTag(tag).to_string();
        self.ranks
            .iter()
            .enumerate()
            .map(|(rank, state)| {
                describe_rank(
                    rank,
                    state.entered.load(Ordering::Acquire),
                    state.crossed.load(Ordering::Acquire),
                    now.saturating_sub(state.crossed_at_ms.load(Ordering::Relaxed)),
                    &describe_phase(state.phase.load(Ordering::Relaxed)),
                    &show,
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// Wait every peer's producer event of `buffer` (host waits unless the
    /// builder opted into device waits).
    fn wait_peers(&self, gpu: &mut Gpu, rank: usize, buffer: usize) -> HipResult<()> {
        gpu.bind_thread()?;
        if !self.ranks[rank].peers_enabled.load(Ordering::Acquire) {
            return Err(err(format!(
                "ep_row_gather: gather on rank {rank} before enable_peers"
            )));
        }
        for (peer, state) in self.ranks.iter().enumerate() {
            if peer == rank {
                continue;
            }
            self.ranks[rank]
                .phase
                .store(peer_wait_phase(peer, buffer), Ordering::Relaxed);
            let guard = lock(&state.events[buffer]);
            let event = guard
                .as_ref()
                .ok_or_else(|| err("ep_row_gather: gather after release".into()))?;
            match (self.device_waits, gpu.active_stream.as_ref()) {
                (true, Some(stream)) => gpu.hip.stream_wait_event(stream, event)?,
                _ => gpu.hip.event_synchronize(event)?,
            }
        }
        Ok(())
    }

    /// Band exchange (see the module docs): `publish` → barrier → geometry
    /// check → one pitched copy per rank and band. `tag` identifies the call's
    /// geometry and must be equal on every rank; a mismatch aborts the
    /// transport. Every band must lie inside the rank's buffer and `dst` must
    /// hold `rows` rows of `ranks * band_bytes`.
    pub fn exchange_bands(
        &self,
        gpu: &mut Gpu,
        rank: usize,
        buffer: usize,
        tag: u64,
        bands: &[EpBand<'_>],
    ) -> HipResult<()> {
        let result = self.check(gpu, rank, buffer).and_then(|()| {
            self.check_bands(bands)?;
            self.ranks[rank].tags[buffer].store(tag, Ordering::Release);
            self.publish(gpu, rank, buffer)
        });
        if let Err(e) = result {
            self.abort();
            return Err(e);
        }
        self.barrier.wait().map_err(|m| err(m.into()))?;
        if let Err(e) = self.gather_bands(gpu, rank, buffer, tag, bands) {
            self.abort();
            return Err(e);
        }
        Ok(())
    }

    fn check_bands(&self, bands: &[EpBand<'_>]) -> HipResult<()> {
        let capacity = self.spec.buffer_bytes();
        for band in bands {
            let end = band
                .rows
                .checked_mul(band.band_bytes)
                .and_then(|bytes| bytes.checked_add(band.offset));
            let dst = band
                .rows
                .checked_mul(band.band_bytes)
                .and_then(|bytes| bytes.checked_mul(self.spec.ranks));
            match (end, dst) {
                (Some(end), Some(dst)) if end <= capacity && dst <= band.dst.buf.size() => {}
                _ => {
                    return Err(err(format!(
                    "ep_row_gather: band of {} rows x {} B at {} exceeds the {capacity} B buffer \
                         or its {} B destination",
                    band.rows,
                    band.band_bytes,
                    band.offset,
                    band.dst.buf.size()
                )))
                }
            }
        }
        Ok(())
    }

    fn gather_bands(
        &self,
        gpu: &mut Gpu,
        rank: usize,
        buffer: usize,
        tag: u64,
        bands: &[EpBand<'_>],
    ) -> HipResult<()> {
        if let Some(peer) = self
            .ranks
            .iter()
            .position(|state| state.tags[buffer].load(Ordering::Acquire) != tag)
        {
            return Err(err(format!(
                "ep_row_gather: band layout of rank {peer} disagrees with rank {rank} (tag {tag:#x})"
            )));
        }
        self.wait_peers(gpu, rank, buffer)?;
        let ranks = self.spec.ranks;
        for band in bands {
            for (owner, state) in self.ranks.iter().enumerate() {
                // SAFETY: non-owning view of the owner's registered buffer, live
                // until that rank's `release`; the copy is bounds-checked against it.
                let src = unsafe {
                    hip_bridge::DeviceBuffer::from_raw(
                        state.ptrs[buffer] as *mut std::ffi::c_void,
                        state.sizes[buffer],
                    )
                };
                gpu.memcpy_dtod_2d_at_ordered_async(
                    &band.dst.buf,
                    owner * band.band_bytes,
                    ranks * band.band_bytes,
                    &src,
                    band.offset,
                    band.band_bytes,
                    band.band_bytes,
                    band.rows,
                )?;
            }
        }
        Ok(())
    }

    /// Poison the barrier: every current/future waiter returns Err (a rank
    /// failed). Barrier waits time out (default 30 s) with Err rather than
    /// hanging. With kernel boundaries it also poisons every live rank's
    /// bounded waits, so a rank spinning on a boundary the failed rank will
    /// never reach stops within its next check instead of its 500 ms deadline,
    /// and that rank's step check reports the poison.
    pub fn abort(&self) {
        self.barrier.abort();
        for state in &self.ranks {
            if let Some(boundary) = lock(&state.boundary).as_ref() {
                // SAFETY: the handle lives in the rank's `boundary`, which
                // `release` empties (under this lock) before the rank's `Gpu`
                // and its wait block are dropped.
                unsafe { boundary.poison.poison(0) };
            }
        }
    }

    /// Whether `abort` was called (or a barrier wait timed out).
    pub fn is_aborted(&self) -> bool {
        self.barrier.is_aborted()
    }

    /// Destroy this rank's producer events and free its boundary epoch word.
    /// Call once per rank on that rank's thread, after the rank's stream and
    /// every peer's stream that waits on these events or reads the epoch word
    /// have drained. Later `publish`/`gather` return Err.
    pub fn release(&self, rank: usize, gpu: &mut Gpu) -> HipResult<()> {
        self.check(gpu, rank, 0)?;
        gpu.bind_thread()?;
        let mut first = Ok(());
        for slot in &self.ranks[rank].events {
            if let Some(event) = lock(slot).take() {
                let r = gpu.hip.event_destroy(event);
                if first.is_ok() {
                    first = r;
                }
            }
        }
        if let Some(boundary) = lock(&self.ranks[rank].boundary).take() {
            let r = gpu.hip.free(boundary.epoch);
            if first.is_ok() {
                first = r;
            }
        }
        first
    }
}

impl Drop for EpRowGather {
    fn drop(&mut self) {
        let live = self
            .ranks
            .iter()
            .filter(|r| r.events.iter().any(|e| lock(e).is_some()))
            .count();
        if live > 0 {
            eprintln!(
                "WARN: EpRowGather dropped with producer events live on {live} rank(s) — \
                 call release(rank, gpu) on each rank first; HIP event handles leaked."
            );
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn spec(ranks: usize, slots: usize, row_bytes: usize) -> EpRowGatherSpec {
        EpRowGatherSpec {
            ranks,
            slots,
            row_bytes,
        }
    }

    #[test]
    fn spec_validation() {
        assert!(spec(2, 10, 10240).validate().is_ok());
        assert!(spec(3, 10, 10240).validate().is_ok());
        assert!(spec(4, 1, 16).validate().is_ok());
        assert!(spec(1, 10, 10240).validate().is_err());
        assert!(spec(0, 10, 10240).validate().is_err());
        assert!(spec(5, 10, 10240).validate().is_ok());
        assert!(spec(6, 10, 10240).validate().is_ok());
        assert!(spec(8, 10, 10240).validate().is_ok());
        assert!(spec(9, 10, 10240).validate().is_err());
        assert!(spec(usize::MAX, 10, 10240).validate().is_err());
        assert!(spec(4, 0, 10240).validate().is_err());
        assert!(spec(4, 10, 0).validate().is_err());
        assert!(spec(4, 10, 10248).validate().is_err());
        assert!(spec(4, 10, 15).validate().is_err());
        assert!(spec(4, usize::MAX, 16).validate().is_err());
        assert!(spec(4, i32::MAX as usize + 1, 16).validate().is_err());
    }

    #[test]
    fn buffer_len_validation() {
        let s = spec(4, 10, 10240);
        assert_eq!(s.buffer_bytes(), 102_400);
        assert!(s.check_buffer_len(0, 102_400).is_ok());
        assert!(s.check_buffer_len(1, 1 << 20).is_ok());
        assert!(s.check_buffer_len(0, 102_399).is_err());
    }

    #[test]
    fn call_slots_within_capacity() {
        let s = spec(4, 80, 10240);
        assert!(s.check_call_slots(1).is_ok());
        assert!(s.check_call_slots(20).is_ok());
        assert!(s.check_call_slots(80).is_ok());
        assert!(s.check_call_slots(0).is_err());
        assert!(s.check_call_slots(81).is_err());
    }

    #[test]
    fn device_waits_require_explicit_opt_in() {
        assert!(!EpRowGather::builder(spec(2, 10, 10240)).device_waits);
        assert!(
            EpRowGather::builder(spec(2, 10, 10240))
                .device_waits(true)
                .device_waits
        );
    }

    #[test]
    fn builder_requires_valid_spec_and_all_ranks() {
        let b = EpRowGather::builder(spec(4, 10, 10248));
        assert!(b.build().is_err());
        let b = EpRowGather::builder(spec(4, 10, 10240));
        assert!(b.build().is_err(), "no rank registered");
    }

    #[test]
    fn send_sync_arc() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EpRowGather>();
        assert_send_sync::<Arc<EpRowGather>>();
    }

    #[test]
    fn barrier_passes_many_generations() {
        let barrier = Arc::new(SpinBarrier::new(4, Duration::from_secs(10)));
        let counter = Arc::new(AtomicUsize::new(0));
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let barrier = barrier.clone();
                let counter = counter.clone();
                std::thread::spawn(move || {
                    for round in 0..500usize {
                        counter.fetch_add(1, Ordering::SeqCst);
                        barrier.wait().expect("barrier");
                        // Everyone of this round has incremented before anyone passes.
                        assert!(counter.load(Ordering::SeqCst) >= (round + 1) * 4);
                        barrier.wait().expect("barrier");
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread");
        }
        assert_eq!(counter.load(Ordering::SeqCst), 2000);
    }

    #[test]
    fn barrier_abort_releases_waiters_and_poisons() {
        let barrier = Arc::new(SpinBarrier::new(3, Duration::from_secs(30)));
        let started = Instant::now();
        let waiters: Vec<_> = (0..2)
            .map(|_| {
                let barrier = barrier.clone();
                std::thread::spawn(move || barrier.wait())
            })
            .collect();
        std::thread::sleep(Duration::from_millis(50));
        barrier.abort();
        for w in waiters {
            assert!(w.join().expect("thread").is_err());
        }
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(barrier.wait().is_err(), "future waiters fail too");
    }

    #[test]
    fn barrier_times_out_with_err_and_aborts() {
        let barrier = SpinBarrier::new(2, Duration::from_millis(100));
        let started = Instant::now();
        let r = barrier.wait();
        assert_eq!(r, Err("ep_row_gather barrier timed out"));
        assert!(started.elapsed() >= Duration::from_millis(100));
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(barrier.is_aborted());
        assert!(barrier.wait().is_err());
    }

    // ── crossing tags: the protocol half of `exchange_layer`, no device ──

    /// A transport with no device state: only the barrier and tag protocol runs.
    pub(crate) fn bare(ranks: usize, timeout: Duration) -> Arc<EpRowGather> {
        let states = (0..ranks)
            .map(|rank| RankState {
                device_id: rank as i32,
                ptrs: Vec::new(),
                sizes: Vec::new(),
                tags: (0..MAX_BUFFERS).map(|_| AtomicU64::new(0)).collect(),
                events: Vec::new(),
                epoch_ptr: 0,
                boundary: Mutex::new(None),
                ledger: Mutex::new(Vec::new()),
                peers_enabled: AtomicBool::new(true),
                calls: AtomicU64::new(0),
                entered: AtomicU64::new(0),
                crossed: AtomicU64::new(0),
                crossed_at_ms: AtomicU64::new(0),
                phase: AtomicU64::new(PHASE_OUTSIDE),
            })
            .collect();
        Arc::new(EpRowGather {
            spec: spec(ranks, 80, 10240),
            buffers: MAX_BUFFERS,
            ranks: states,
            barrier: SpinBarrier::new(ranks, timeout),
            device_waits: false,
            kernel_boundaries: false,
            started: Instant::now(),
        })
    }

    /// One rank's crossing of `exchange_layer` without its device half.
    pub(crate) fn cross_once(
        gather: &EpRowGather,
        rank: usize,
        kind: GatherKind,
        layer: usize,
        slots: usize,
    ) -> HipResult<()> {
        let tag = gather.enter(rank, kind, layer % 2, layer, slots);
        gather.cross(rank, tag)
    }

    /// Publish `rank`'s tag for a crossing without crossing yet (the first half
    /// of [`cross_once`]), so a test can order the ranks' crossings.
    pub(crate) fn enter_tag(
        gather: &EpRowGather,
        rank: usize,
        kind: GatherKind,
        layer: usize,
        slots: usize,
    ) -> CrossingTag {
        gather.enter(rank, kind, layer % 2, layer, slots)
    }

    /// The barrier half of [`cross_once`] for a tag from [`enter_tag`].
    pub(crate) fn cross_tag(gather: &EpRowGather, rank: usize, tag: CrossingTag) -> HipResult<()> {
        gather.cross(rank, tag)
    }

    /// Parties currently waiting on `barrier` (0 once a crossing completes).
    pub(crate) fn arrived(barrier: &SpinBarrier) -> usize {
        barrier.arrived.load(Ordering::Acquire)
    }

    /// Parties currently waiting on the gather's barrier.
    pub(crate) fn gather_arrived(gather: &EpRowGather) -> usize {
        arrived(&gather.barrier)
    }

    #[test]
    fn crossing_tag_packs_and_formats_every_field() {
        let tag = CrossingTag::new(GatherKind::Explicit, 3, 47, 80, 0x12_3456);
        assert_eq!(tag.kind(), 2);
        assert_eq!(tag.buffer(), 3);
        assert_eq!(tag.layer(), 47);
        assert_eq!(tag.slots(), 80);
        assert_eq!(tag.seq(), 0x12_3456);
        assert_eq!(
            tag.to_string(),
            "explicit_owners layer 47 buffer 3 slots 80 seq 1193046"
        );
        // The sequence wraps at 24 bits without disturbing the other fields.
        let wrapped = CrossingTag::new(GatherKind::ExpertModulo, 1, 5, 10, (1 << 24) + 7);
        assert_eq!(wrapped.seq(), 7);
        assert_eq!(
            (wrapped.kind(), wrapped.buffer(), wrapped.layer()),
            (1, 1, 5)
        );
        assert_eq!(CrossingTag::NONE.to_string(), "none");
        assert_ne!(CrossingTag::NONE, tag);
        let unnamed = CrossingTag::new(GatherKind::ExpertModulo, 0, NO_LAYER, 10, 0);
        assert!(unnamed.to_string().contains("layer ?"), "{unnamed}");
    }

    #[test]
    fn lockstep_crossings_agree_and_progress_is_recorded() {
        let gather = bare(2, Duration::from_secs(10));
        let handles: Vec<_> = (0..2)
            .map(|rank| {
                let gather = gather.clone();
                std::thread::spawn(move || {
                    for round in 0..2000usize {
                        // Trunk-like alternation plus a dedicated head buffer.
                        let layer = round % 48;
                        let kind = if layer % 3 == 0 {
                            GatherKind::Explicit
                        } else {
                            GatherKind::ExpertModulo
                        };
                        cross_once(&gather, rank, kind, layer, 1 + round % 80).expect("crossing");
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("thread");
        }
        for state in &gather.ranks {
            assert_eq!(state.calls.load(Ordering::SeqCst), 2000);
            assert_eq!(
                state.entered.load(Ordering::SeqCst),
                state.crossed.load(Ordering::SeqCst)
            );
        }
        assert!(!gather.is_aborted());
    }

    #[test]
    fn a_skipped_gather_fails_at_once_naming_both_tags() {
        let gather = bare(2, Duration::from_secs(30));
        let started = Instant::now();
        let run = |rank: usize, skip: Option<usize>| {
            let gather = gather.clone();
            std::thread::spawn(move || {
                for layer in 0..8usize {
                    if skip == Some(layer) {
                        continue;
                    }
                    cross_once(&gather, rank, GatherKind::ExpertModulo, layer, 50)?;
                }
                Ok::<(), HipError>(())
            })
        };
        // Rank 1 skips layer 3: both ranks cross the barrier of their 4th
        // gather together (seq 3 each) but at layers 3 and 4.
        let (a, b) = (run(0, None), run(1, Some(3)));
        let (a, b) = (a.join().expect("rank 0"), b.join().expect("rank 1"));
        let (a, b) = (a.expect_err("rank 0 fails"), b.expect_err("rank 1 fails"));
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "failed after {:?}, not after the 30 s barrier timeout",
            started.elapsed()
        );
        // The skipped gather desynchronises the ranks by one crossing, so which
        // layers meet at the barrier depends on scheduling. Whichever rank
        // compares first names both ranks' tags and aborts; the other may see
        // that abort before its own comparison, so it fails either way.
        let texts = [a.to_string(), b.to_string()];
        for text in &texts {
            assert!(!text.contains("timed out"), "{text}");
            assert!(
                text.contains("different gathers") || text.contains("aborted"),
                "{text}"
            );
        }
        assert!(
            texts.iter().any(|text| text.contains("different gathers")
                && text.contains("entered [expert_modulo layer")
                && text.contains("slot for buffer")),
            "one rank names both tags: {texts:?}"
        );
        assert!(gather.is_aborted());
    }

    #[test]
    fn a_different_call_kind_or_row_count_is_named() {
        for (kind_b, slots_b, expect) in [
            (GatherKind::Explicit, 50, "explicit_owners"),
            (GatherKind::ExpertModulo, 60, "slots 60"),
        ] {
            let gather = bare(2, Duration::from_secs(30));
            let other = gather.clone();
            let peer = std::thread::spawn(move || cross_once(&other, 1, kind_b, 7, slots_b));
            let mine = cross_once(&gather, 0, GatherKind::ExpertModulo, 7, 50);
            let theirs = peer.join().expect("peer");
            for result in [mine, theirs] {
                let text = result.expect_err("disagreement").to_string();
                assert!(text.contains("different gathers"), "{text}");
                assert!(text.contains(expect) && text.contains("layer 7"), "{text}");
            }
        }
    }

    #[test]
    fn a_peer_stalled_between_gathers_is_reported_at_the_timeout() {
        let gather = bare(2, Duration::from_millis(300));
        // Rank 1 completes three gathers, then goes quiet (a stall outside the
        // gather, e.g. in a host wait); rank 0 enters a fourth.
        let other = gather.clone();
        let peer = std::thread::spawn(move || {
            for layer in 0..3usize {
                cross_once(&other, 1, GatherKind::ExpertModulo, layer, 50).expect("rank 1");
            }
            std::thread::sleep(Duration::from_millis(800));
        });
        for layer in 0..3usize {
            cross_once(&gather, 0, GatherKind::ExpertModulo, layer, 50).expect("rank 0");
        }
        let error = cross_once(&gather, 0, GatherKind::ExpertModulo, 3, 50)
            .expect_err("rank 1 never arrives")
            .to_string();
        peer.join().expect("peer");
        assert!(error.contains("barrier timed out"), "{error}");
        assert!(error.contains("waited"), "{error}");
        assert!(error.contains("at [expert_modulo layer 3"), "{error}");
        assert!(
            error.contains("rank 1 is between crossings")
                && error.contains("last crossed [expert_modulo layer 2"),
            "names the stalled peer's last crossing: {error}"
        );
        assert!(
            error.contains("rank 0 is inside a crossing"),
            "names the waiting rank: {error}"
        );
        assert!(gather.is_aborted());
    }

    #[test]
    fn the_report_names_what_a_stalled_peer_is_host_waiting_for() {
        // A peer blocked in `event_synchronize` of rank 0's event (rank 0's
        // stream stuck behind a device stall) is the signature of a GPU-side
        // hang: its crossing record says "between crossings", its phase says
        // which event it is waiting on.
        let gather = bare(2, Duration::from_millis(200));
        gather.ranks[1]
            .phase
            .store(peer_wait_phase(0, 1), Ordering::Relaxed);
        let report = gather.progress_report();
        assert!(
            report.contains(
                "rank 1 has entered no crossing (last entered [none], last crossed [none]; \
                 now host-waiting for rank 0's producer event of buffer 1"
            ),
            "{report}"
        );
        assert!(
            !report.contains(
                "rank 0 has entered no crossing (last entered [none], last crossed [none]; now"
            ),
            "an idle rank reports no phase: {report}"
        );
        for (phase, text) in [
            (PHASE_PUBLISH, "recording its producer event"),
            (PHASE_BARRIER, "waiting at the barrier"),
            (PHASE_COPY, "launching the gather copy"),
        ] {
            assert_eq!(describe_phase(phase), text);
        }
        assert_eq!(describe_phase(PHASE_OUTSIDE), "");
    }
}
