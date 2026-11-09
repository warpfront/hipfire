// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Race-injection tests of the D2 tail: the REAL [`TailPolicy`], the real railgun generation API
//! and the real [`PeerBoard`] driven over a simulated device TIMELINE.
//!
//! # The simulated device ([`Dev`])
//!
//! A single-threaded device with one FIFO queue per stream (compute, copy and stage stream of each
//! of the 2 ranks). Nothing executes at enqueue time: an op runs only when the test advances its
//! stream ([`Dev::run_all`], [`Dev::run_until`], [`Dev::settle`]) or when a fence wait runs the
//! stream up to its marker, so a test can DELAY any op (the last window reader, a table upload, a
//! fill copy, a peer's reader) for as long as it likes and interleave the rest. Ops:
//!
//! * `Fill`: the copy of one unit into a slot. At execution it writes the unit tag of the slot's
//!   bytes (`slot_bytes[owner][slot]`), after checking that the write spans are exactly the slot's
//!   extent and that nothing still names the slot (below).
//! * `Upload`: a table image reaching the device (written at execution).
//! * `Read`: a forward window's reader. At execution it walks the device table image of its rank
//!   (decode table: the rank's own slots; full table: the PEERS' slots over P2P) and, for every
//!   entry naming a slot, checks that the slot holds the unit the entry was published for.
//! * `Pinned`: a stage reader: the same check over a fixed `(unit, slot)` list, taken from
//!   `GenerationClosure::slot_reads` of the leased generation.
//! * `Marker` / `WaitMarker`: a fence / a cross-stream wait (the tables-ready event).
//!
//! The checker records a [`Violation`] instead of panicking: a reader that found other bytes than
//! its entry's unit ([`Violation::Read`]), a read of a freed pool, a copy onto a slot a device
//! table image still names ([`Violation::WriteNamed`], which covers the peers' full-table images),
//! onto a slot a live generation still references ([`Violation::WriteLive`], the white-box check of
//! `SlotBank::refs`) or outside the slot's extent. The correct tests assert the protocol WAITED,
//! REFUSED, DECLINED, QUARANTINED or FROZE with the old bytes intact and [`Mesh::assert_clean`];
//! every `*_positive_control` runs a [`Variant`] that breaks the ordering and asserts the checker
//! SEES the violation, which is what proves the checker (and the scenario) can detect the race.
//!
//! The fence backends ([`Backend`], [`DeviceIdle`]) implement `unsafe trait FenceBackend` over the
//! timeline: `record` enqueues a marker at the stream tail, `poll` is `Complete` iff the marker
//! executed, `wait` runs the stream up to the marker unless the test set the stream's [`Wait`]
//! mode to `Pending` (a wait that cannot finish) or `Stuck` (`Failed`, `Quiescence::Unknown`). The
//! test can also bump a stream's incarnation (a recreated stream), make its event query fail, or
//! make a backend LIE (`lie_complete`: an unsound backend, used only by positive controls).
//!
//! # The protocol driver ([`SimRank`])
//!
//! One rank of a 2-rank mesh sharing one [`PeerBoard`]. [`SimRank::retire`] runs the retirement
//! `j` of `expert_unit_tail_rank.rs::retire_group` in its order, over the real objects:
//!
//! 1. fence the window of the published generation: `lease.enqueued(compute)` then
//!    `finish_window(pending, compute)` -> `FencedWindow::into_generation()`. A fence that cannot
//!    finish keeps the `PendingWindow` and the step reports [`Step::Blocked`]; an unknown fence
//!    quarantines the generation and freezes the rank.
//! 2. the D2H log is not modeled; the pending fill batches are polled with
//!    `complete_fills(batch, copy, false)` (the fill-overlap `wait_newest` only adds a wait: a
//!    pending batch simply stays pending here, the stricter case) -> `FillComplete` tokens.
//! 3. `policy.retire_generation(G, peers)`; in PEER mode `peers = Some((board, rank, j))`, the one
//!    `released_through` call. A refusal returns the generation (kept, retried by the next step).
//! 4. `retire_windows` / `retire_peer_windows` with the tokens and the `GenerationRetired`.
//! 5. the table images (the backing's host side) are written from the decisions, the peer deltas
//!    are posted, taken (out of step -> freeze, no ack) and applied, and the changed images are
//!    uploaded on the compute stream.
//! 6. `policy.seal(backing, {epoch j, sub 0})` + `acquire_window_owned` -> the lease until the next
//!    retirement; then the PEER ack (`board.ack`), which only records that the uploads were
//!    ENQUEUED ([`Ack`] lets a test withhold or defer it).
//! 7. the fills: with [`Cfg::with_gate`] a tables-ready marker is recorded on compute and the copy
//!    stream waits for it, then `policy.issue_fills(tickets, copy, enqueue)`.
//!
//! Not modeled: the fill-overlap deferral of step 7 (a later enqueue only makes the fills safer),
//! the log, the borrow restore copies (the borrow is exercised through `quiesce` / `lend_scratch`).
//! The tables-ready gate is OFF by default: it only orders a fill after the SAME rank's earlier
//! compute work, so leaving it out checks the stronger property (the fences and acks alone keep
//! the slot safe); the two tests of the table upload turn it on. The cross-rank row exchange (which
//! guarantees that the peers' uploads of retirement `j` executed before the owner's retirement
//! `j + 1`, `expert_unit_tail.rs` module docs) is the scenario's job: tests that want a peer's
//! uploads executed run its compute stream ([`Mesh::group`] does it for every window).
//!
//! # Variants (the positive controls)
//!
//! Each [`Variant`] breaks one rule of the protocol above and nothing else:
//!
//! * `EarlyFence`: the window fence is recorded BEFORE the window's final reader is enqueued
//!   (`enqueued()` called early), so the fence completes without the reader.
//! * `LyingPeerFence`: the owner asks a [`PeerReaderFence`] that claims every peer acknowledged
//!   (refill without the peers' acks).
//! * `AckBeforeFence`: the peer acknowledges the owner's posts before fencing its own window.
//! * `AckWithoutApply`: a rank that finds an out-of-step post acknowledges it instead of freezing,
//!   without applying it.
//! * `PublishBeforeFill`: the table entry of a fill is published when the fill is DECIDED, before
//!   its copy ran.
//! * `TrustTokenNumbers`: a completion is accepted by its `(slot, fill_generation)` numbers (the
//!   old pair API) instead of the policy's typed token, so a foreign token publishes the fill.
//! * `UnleasedStage`: the stage reader reads the slots without holding a lease on the generation.
//! * `ReleaseEarly`: the teardown frees the pool although the bank refused to close.

use std::cell::{RefCell, RefMut};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::sync::Arc;

use railgun::sealed_table::generation::{
    acquire_window, acquire_window_owned, finish_window, BlockReclaimed, Expert, FenceBackend,
    FenceFailure, FenceState, FillComplete, FillTicket, FillsNotDone, GenerationId, LentBlock,
    OwnedTableGeneration, PeerReaderFence, PendingFills, PendingOrQuarantined, PendingWindow,
    QuarantinedWindow, ResidencyError, ScratchLoan, SlotBank, TableBacking, WindowReadLease,
};
use railgun::window::Quiescence;
use railgun::Binding;

use super::rig::{rig_pool, Rig, RIG_BLOCK, RIG_DOWN, RIG_GATE_UP, RIG_TABLE};
use super::{
    occupant_unit, EpUnitTailConfig, PeerBoard, PeerDelta, PeerTake, Retire, SlotPhase, TailPolicy,
    UnitKey, WindowUse, RESERVE_RETIREMENTS,
};

const RANKS: usize = 2;

/// The compute stream of `rank`.
const fn compute_stream(rank: usize) -> usize {
    rank * 3
}

/// The copy stream (fills) of `rank`.
const fn copy_stream(rank: usize) -> usize {
    rank * 3 + 1
}

/// The stage stream (slot-reading stage consumers) of `rank`.
const fn stage_stream(rank: usize) -> usize {
    rank * 3 + 2
}

// ------------------------------------------------------------------------------------ device

/// The pointer tables a reader can walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Table {
    /// Rank `r`'s decode table: names `r`'s own slots.
    Decode(usize),
    /// Rank `r`'s full table: names the PEERS' slots (read over P2P).
    Full(usize),
}

/// A device table image: the units whose entry names a slot `(owner rank, slot)`; every other
/// unit's entry is canonical.
type Image = BTreeMap<UnitKey, (usize, u32)>;

/// What a fence `wait` on a stream does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Wait {
    /// Run the stream up to the marker.
    Run,
    /// The wait cannot finish (the stream is held): the fence stays pending.
    Pending,
    /// The wait fails with unknown quiescence.
    Stuck,
}

/// A race the checker saw. The fields are diagnostics printed by a failing assertion.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq)]
enum Violation {
    /// A reader executed and the slot held other bytes than the unit its entry names.
    Read {
        reader: String,
        unit: UnitKey,
        owner: usize,
        slot: u32,
        found: Option<UnitKey>,
    },
    /// A reader executed over a freed pool.
    FreedPool {
        reader: String,
        owner: usize,
        slot: u32,
    },
    /// A fill executed on a slot a device table image still names.
    WriteNamed {
        unit: UnitKey,
        owner: usize,
        slot: u32,
        named_by: Table,
        named_unit: UnitKey,
    },
    /// A fill executed on a slot a live generation still references (`SlotBank::refs`).
    WriteLive {
        unit: UnitKey,
        owner: usize,
        slot: u32,
    },
    /// A fill wrote other bytes than the slot's extent.
    BadSpan { owner: usize, slot: u32 },
}

enum Op {
    Marker(u64),
    WaitMarker(u64),
    Fill {
        owner: usize,
        slot: u32,
        unit: UnitKey,
        spans: Vec<Binding>,
    },
    Upload {
        table: Table,
        image: Image,
    },
    Read {
        label: String,
        table: Table,
    },
    Pinned {
        label: String,
        reads: Vec<(UnitKey, usize, u32)>,
    },
}

struct Stream {
    queue: VecDeque<Op>,
    wait: Wait,
    incarnation: u64,
    query_error: Option<Quiescence>,
    lie_complete: bool,
}

type Shared = Rc<RefCell<Dev>>;

struct Dev {
    streams: Vec<Stream>,
    next_marker: u64,
    /// Markers that executed.
    done: HashSet<u64>,
    /// `slot_bytes[owner][slot]`: the unit whose bytes the slot holds (`None`: never filled).
    slot_bytes: Vec<Vec<Option<UnitKey>>>,
    images: BTreeMap<Table, Image>,
    freed: [bool; RANKS],
    /// `(owner, slot)` a live generation references: refreshed by the driver from the bank.
    protected: BTreeSet<(usize, u32)>,
    /// The two extents `[gate_up, down]` of every slot.
    extents: Vec<[Binding; 2]>,
    violations: Vec<Violation>,
}

impl Dev {
    /// A device with `slots` slots per rank pool (the rig's block layout).
    fn new(slots: usize) -> Shared {
        let n = slots as u64;
        let extents = (0..n)
            .map(|slot| {
                [
                    Binding {
                        base: RIG_BLOCK + slot * RIG_GATE_UP,
                        bytes: RIG_GATE_UP,
                    },
                    Binding {
                        base: RIG_BLOCK + n * RIG_GATE_UP + slot * RIG_DOWN,
                        bytes: RIG_DOWN,
                    },
                ]
            })
            .collect();
        Rc::new(RefCell::new(Dev {
            streams: (0..RANKS * 3)
                .map(|_| Stream {
                    queue: VecDeque::new(),
                    wait: Wait::Run,
                    incarnation: 1,
                    query_error: None,
                    lie_complete: false,
                })
                .collect(),
            next_marker: 1,
            done: HashSet::new(),
            slot_bytes: vec![vec![None; slots]; RANKS],
            images: BTreeMap::new(),
            freed: [false; RANKS],
            protected: BTreeSet::new(),
            extents,
            violations: Vec::new(),
        }))
    }

    fn push(&mut self, stream: usize, op: Op) {
        self.streams[stream].queue.push_back(op);
    }

    /// Ops waiting on `stream`.
    fn queued(&self, stream: usize) -> usize {
        self.streams[stream].queue.len()
    }

    fn idle(&self) -> bool {
        self.streams.iter().all(|s| s.queue.is_empty())
    }

    /// Whether a recorded violation satisfies `f`.
    fn any(&self, f: impl Fn(&Violation) -> bool) -> bool {
        self.violations.iter().any(f)
    }

    // --- knobs the tests turn

    fn set_wait(&mut self, stream: usize, wait: Wait) {
        self.streams[stream].wait = wait;
    }

    /// Every event query / synchronize of `stream` fails with `quiescence`.
    fn fail_query(&mut self, stream: usize, quiescence: Option<Quiescence>) {
        self.streams[stream].query_error = quiescence;
    }

    /// An unsound backend: `poll` / `wait` of `stream` report `Complete` whatever ran.
    fn lie(&mut self, stream: usize, lie: bool) {
        self.streams[stream].lie_complete = lie;
    }

    /// The stream handle was recreated: its incarnation changes (queued ops stay).
    fn recreate(&mut self, stream: usize) {
        self.streams[stream].incarnation += 1;
    }

    /// The queued ops of `stream` are lost (without any incarnation change).
    fn reset(&mut self, stream: usize) {
        self.streams[stream].queue.clear();
    }

    // --- execution

    /// Execute the front op of `stream`; `false` if there is none or it waits for a marker.
    fn exec(&mut self, stream: usize) -> bool {
        let blocked = match self.streams[stream].queue.front() {
            None => return false,
            Some(Op::WaitMarker(marker)) => !self.done.contains(marker),
            Some(_) => false,
        };
        if blocked {
            return false;
        }
        let op = self.streams[stream]
            .queue
            .pop_front()
            .expect("front checked");
        match op {
            Op::Marker(id) => {
                self.done.insert(id);
            }
            Op::WaitMarker(_) => {}
            Op::Upload { table, image } => {
                self.images.insert(table, image);
            }
            Op::Fill {
                owner,
                slot,
                unit,
                spans,
            } => self.exec_fill(owner, slot, unit, &spans),
            Op::Read { label, table } => {
                let reads: Vec<(UnitKey, usize, u32)> = self
                    .images
                    .get(&table)
                    .into_iter()
                    .flat_map(|image| {
                        image
                            .iter()
                            .map(|(&unit, &(owner, slot))| (unit, owner, slot))
                    })
                    .collect();
                self.check_reads(&label, &reads);
            }
            Op::Pinned { label, reads } => self.check_reads(&label, &reads),
        }
        true
    }

    fn exec_fill(&mut self, owner: usize, slot: u32, unit: UnitKey, spans: &[Binding]) {
        if spans != &self.extents[slot as usize][..] {
            self.violations.push(Violation::BadSpan { owner, slot });
        }
        if self.protected.contains(&(owner, slot)) {
            self.violations
                .push(Violation::WriteLive { unit, owner, slot });
        }
        for (&table, image) in &self.images {
            for (&named_unit, &(o, s)) in image {
                if o == owner && s == slot {
                    self.violations.push(Violation::WriteNamed {
                        unit,
                        owner,
                        slot,
                        named_by: table,
                        named_unit,
                    });
                }
            }
        }
        self.slot_bytes[owner][slot as usize] = Some(unit);
    }

    /// A reader over `(unit, owner, slot)` entries: each slot must hold its entry's unit.
    fn check_reads(&mut self, reader: &str, reads: &[(UnitKey, usize, u32)]) {
        for &(unit, owner, slot) in reads {
            if self.freed[owner] {
                self.violations.push(Violation::FreedPool {
                    reader: reader.to_string(),
                    owner,
                    slot,
                });
                continue;
            }
            let found = self.slot_bytes[owner][slot as usize];
            if found != Some(unit) {
                self.violations.push(Violation::Read {
                    reader: reader.to_string(),
                    unit,
                    owner,
                    slot,
                    found,
                });
            }
        }
    }

    /// Execute `stream` until it is empty or blocked; the number of ops executed.
    fn run_all(&mut self, stream: usize) -> usize {
        let mut n = 0;
        while self.exec(stream) {
            n += 1;
        }
        n
    }

    /// Execute `stream` until `marker` executed (or the stream is empty / blocked).
    fn run_until(&mut self, stream: usize, marker: u64) {
        while !self.done.contains(&marker) && self.exec(stream) {}
    }

    /// Every compute stream runs dry: the window's kernels and the cross-rank row exchange.
    fn run_compute(&mut self) {
        for rank in 0..RANKS {
            self.run_all(compute_stream(rank));
        }
    }

    /// Every stream runs dry (cross-stream waits resolve as far as they can).
    fn settle(&mut self) {
        loop {
            let mut progressed = false;
            for stream in 0..self.streams.len() {
                progressed |= self.run_all(stream) > 0;
            }
            if !progressed {
                break;
            }
        }
    }

    /// The state of fence `fence` recorded on `stream`.
    fn query(&mut self, stream: usize, fence: u64, block: bool) -> FenceState {
        let (error, lie, wait) = {
            let s = &self.streams[stream];
            (s.query_error, s.lie_complete, s.wait)
        };
        if let Some(quiescence) = error {
            return FenceState::Failed(FenceFailure {
                message: format!("event query on stream {stream} failed"),
                quiescence,
            });
        }
        if lie || self.done.contains(&fence) {
            return FenceState::Complete;
        }
        if !block {
            return FenceState::Pending;
        }
        match wait {
            Wait::Stuck => FenceState::Failed(FenceFailure {
                message: format!("stream {stream} is stuck"),
                quiescence: Quiescence::Unknown,
            }),
            Wait::Pending => FenceState::Pending,
            Wait::Run => {
                self.run_until(stream, fence);
                if self.done.contains(&fence) {
                    FenceState::Complete
                } else {
                    FenceState::Pending
                }
            }
        }
    }
}

/// The fence backend of one stream of the timeline.
struct Backend {
    dev: Shared,
    stream: usize,
}

impl Backend {
    fn new(dev: &Shared, stream: usize) -> Self {
        Self {
            dev: Rc::clone(dev),
            stream,
        }
    }
}

// SAFETY: `record` queues a marker at the stream tail and `poll` / `wait` report `Complete` only
// once that marker executed, i.e. after every op queued before it (the stream is FIFO). The one
// exception is the test-only `lie_complete` knob, which breaks this contract ON PURPOSE: the
// positive controls use it to prove the checker catches an unsound backend.
unsafe impl FenceBackend for Backend {
    fn incarnation(&self) -> u64 {
        self.dev.borrow().streams[self.stream].incarnation
    }

    fn record(&mut self) -> Result<u64, FenceFailure> {
        let mut dev = self.dev.borrow_mut();
        let id = dev.next_marker;
        dev.next_marker += 1;
        dev.push(self.stream, Op::Marker(id));
        Ok(id)
    }

    fn poll(&mut self, fence: u64) -> FenceState {
        self.dev.borrow_mut().query(self.stream, fence, false)
    }

    fn wait(&mut self, fence: u64) -> FenceState {
        self.dev.borrow_mut().query(self.stream, fence, true)
    }
}

/// The device-scope backend of `quiesce` (a device synchronize): `Complete` iff every stream is
/// empty, and its wait runs every stream dry.
struct DeviceIdle {
    dev: Shared,
}

// SAFETY: `Complete` is reported only when every queue of the device is empty.
unsafe impl FenceBackend for DeviceIdle {
    fn incarnation(&self) -> u64 {
        1
    }

    fn record(&mut self) -> Result<u64, FenceFailure> {
        Ok(0)
    }

    fn poll(&mut self, _fence: u64) -> FenceState {
        if self.dev.borrow().idle() {
            FenceState::Complete
        } else {
            FenceState::Pending
        }
    }

    fn wait(&mut self, _fence: u64) -> FenceState {
        let mut dev = self.dev.borrow_mut();
        if dev.streams.iter().any(|s| s.wait == Wait::Stuck) {
            return FenceState::Failed(FenceFailure {
                message: "a stream is stuck".into(),
                quiescence: Quiescence::Unknown,
            });
        }
        dev.settle();
        if dev.idle() {
            FenceState::Complete
        } else {
            FenceState::Pending
        }
    }
}

/// A peer fence that claims every peer acknowledged everything: the `LyingPeerFence` variant.
struct LyingFence;

// SAFETY: it is NOT sound, on purpose (a test double of the positive controls): it reports every
// eviction released whatever the peers did.
unsafe impl PeerReaderFence for LyingFence {
    fn released_through(&self, _owner: usize, _at: u64) -> u64 {
        u64::MAX
    }
}

// ------------------------------------------------------------------------------------- rank

/// The protocol rule a positive control breaks (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    Correct,
    EarlyFence,
    LyingPeerFence,
    AckBeforeFence,
    AckWithoutApply,
    PublishBeforeFill,
    TrustTokenNumbers,
    UnleasedStage,
    ReleaseEarly,
}

/// When a peer rank records the ack of its retirement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ack {
    /// After the uploads were enqueued, as the protocol says.
    Now,
    /// Never.
    Withhold,
    /// Held until [`SimRank::flush_ack`].
    Defer,
}

/// What one retirement decided (the policy's decisions without the capabilities).
#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    publish: Vec<(UnitKey, usize)>,
    unpublish: Vec<UnitKey>,
    /// `(slot, unit, generation)` per fill.
    fills: Vec<(usize, UnitKey, u64)>,
}

/// How [`SimRank::retire`] ended.
#[derive(Debug)]
enum Step {
    Retired(Outcome),
    /// The window's fence cannot finish yet: nothing was retired, nothing changed.
    Blocked,
    /// `retire_generation` refused (the generation came back and is retried by the next step).
    Refused(ResidencyError),
    /// A device error or an out-of-step peer post: the tail froze (sticky).
    Frozen(String),
}

impl Step {
    fn retired(self) -> Outcome {
        match self {
            Step::Retired(out) => out,
            other => panic!("expected a retirement, got {other:?}"),
        }
    }
}

/// The shape of a mesh.
#[derive(Clone, Copy)]
struct Cfg {
    /// Slots per rank.
    t: usize,
    /// Fills per window.
    f: usize,
    peer: bool,
    gated: bool,
    shared: bool,
}

impl Cfg {
    fn new(t: usize, f: usize) -> Self {
        Self {
            t,
            f,
            peer: false,
            gated: false,
            shared: false,
        }
    }

    /// PEER quarantine on every rank.
    fn with_peer(mut self) -> Self {
        self.peer = true;
        self
    }

    /// The copy stream waits for the tables-ready event of the compute stream (step 7).
    fn with_gate(mut self) -> Self {
        self.gated = true;
        self
    }

    /// Generations are leased through shared leases ([`shared_lease`]) so that a stage reader can
    /// hold a second lease on the published generation.
    fn with_shared(mut self) -> Self {
        self.shared = true;
        self
    }
}

/// A shared lease with a `'static` lifetime.
fn shared_lease(generation: &OwnedTableGeneration) -> WindowReadLease<'static> {
    let lease = acquire_window(generation).expect("an unquarantined generation");
    // SAFETY: the lifetime parameter is a `PhantomData` borrow; the lease keeps its own `Arc` of
    // the generation core, so it stays valid when the generation value moves (to `retire`).
    // That is exactly the situation under test: a lease that outlives the owner's borrow.
    unsafe { std::mem::transmute(lease) }
}

/// The bank of a policy.
fn bank(policy: &TailPolicy) -> &SlotBank {
    policy.bank.as_ref().expect("the tail's bank is open")
}

/// The deltas a retirement posts for the peers.
fn peer_deltas(retire: &Retire) -> Vec<PeerDelta> {
    let publish = retire.publish.iter().map(|&(unit, slot)| PeerDelta {
        unit,
        slot: Some([slot as u64, 0]),
        canon: [0, 0],
    });
    let unpublish = retire.unpublish.iter().map(|&unit| PeerDelta {
        unit,
        slot: None,
        canon: [0, 0],
    });
    publish.chain(unpublish).collect()
}

/// Queue the copies of `tickets` on the copy stream of `rank` and record the batch fence
/// (`issue_fills`, the callback writes only the ticket's spans).
fn enqueue_fills(
    policy: &mut TailPolicy,
    dev: &Shared,
    rank: usize,
    tickets: Vec<FillTicket>,
    copy: &mut Backend,
) -> Result<PendingFills, (Vec<FillTicket>, FenceFailure)> {
    let dev = Rc::clone(dev);
    let mut enqueue = move |ticket: &FillTicket| -> Result<(), String> {
        dev.borrow_mut().push(
            copy_stream(rank),
            Op::Fill {
                owner: rank,
                slot: ticket.slot(),
                unit: occupant_unit(ticket.occupant()),
                spans: ticket.write_spans().to_vec(),
            },
        );
        Ok(())
    };
    policy.issue_fills(tickets, copy, &mut enqueue)
}

/// Production's `abandon_retire_fills`: a retirement that fails before its fills are issued
/// tells the bank its tickets will never be issued.
fn abandon_retire_fills(policy: &mut TailPolicy, retire: Retire) {
    policy.abandon_fills(retire.fills.into_iter().map(|op| op.ticket).collect());
}

/// One rank of the mesh: the real [`TailPolicy`] and the capabilities `retire_group` holds,
/// driven step by step over the timeline.
struct SimRank {
    r: usize,
    t: usize,
    policy: TailPolicy,
    dev: Shared,
    board: Arc<PeerBoard>,
    peer: bool,
    variant: Variant,
    ack: Ack,
    gated: bool,
    shared: bool,
    compute: Backend,
    copy: Backend,
    stage: Backend,
    device: DeviceIdle,
    backing: Option<TableBacking>,
    /// The lease of the published generation (between retirements).
    lease: Option<WindowReadLease<'static>>,
    /// The published generation itself (shared-lease mode only).
    generation: Option<OwnedTableGeneration>,
    /// The window fence is recorded, the last reader not yet fenced.
    pending: Option<PendingWindow<'static>>,
    /// A fenced generation waiting for its retirement (also: refused, to be retried).
    held: Option<OwnedTableGeneration>,
    /// A generation whose window fence failed with unknown quiescence (kept alive, never retires).
    #[allow(dead_code)]
    parked: Option<QuarantinedWindow>,
    /// The stage reader's lease on the published generation.
    stage_pending: Option<PendingWindow<'static>>,
    batches: VecDeque<PendingFills>,
    /// Completion tokens polled and not yet handed to the policy.
    completed: Vec<FillComplete>,
    deferred_acks: Vec<(u64, PeerTake)>,
    /// The deltas of an `AckBeforeFence` early take, applied by the retirement.
    early: Option<Vec<(usize, PeerDelta)>>,
    /// `(slot, fill generation, unit)` of every fill the driver saw decided.
    issued: Vec<(usize, u64, UnitKey)>,
    frozen: Option<String>,
    /// The last generation's window fenced and retired for a teardown.
    retired_last: bool,
    /// The backing's host side: what the next upload writes.
    host_decode: Image,
    host_full: Image,
    sent_decode: Image,
    sent_full: Image,
}

impl SimRank {
    fn new(dev: &Shared, board: &Arc<PeerBoard>, r: usize, cfg: &Cfg) -> Self {
        let config = EpUnitTailConfig {
            tail_units: cfg.t,
            fill_per_step: cfg.f,
            boundary_every: 1,
        };
        let (block, spans) = rig_pool(cfg.t);
        let mut policy = TailPolicy::new(config, false, HashMap::new(), cfg.peer, block, spans)
            .map_err(|(_, error)| error)
            .expect("the sim policy");
        let backing = policy
            .register_table_backing(vec![Binding {
                base: RIG_TABLE,
                bytes: 0x1000,
            }])
            .expect("the sim table backing");
        Self {
            r,
            t: cfg.t,
            policy,
            dev: Rc::clone(dev),
            board: Arc::clone(board),
            peer: cfg.peer,
            variant: Variant::Correct,
            ack: Ack::Now,
            gated: cfg.gated,
            shared: cfg.shared,
            compute: Backend::new(dev, compute_stream(r)),
            copy: Backend::new(dev, copy_stream(r)),
            stage: Backend::new(dev, stage_stream(r)),
            device: DeviceIdle {
                dev: Rc::clone(dev),
            },
            backing: Some(backing),
            lease: None,
            generation: None,
            pending: None,
            held: None,
            parked: None,
            stage_pending: None,
            batches: VecDeque::new(),
            completed: Vec::new(),
            deferred_acks: Vec::new(),
            early: None,
            issued: Vec::new(),
            frozen: None,
            retired_last: false,
            host_decode: Image::new(),
            host_full: Image::new(),
            sent_decode: Image::new(),
            sent_full: Image::new(),
        }
    }

    /// Warm start: fill, copy (synchronously), publish generation `{0, 0}`, upload the warm
    /// table, post the seeds as retirement 0 (PEER).
    fn seed(&mut self, units: &[UnitKey]) {
        let fills = self.policy.seed(units).expect("seed");
        let seeded: Vec<(UnitKey, usize)> = fills.iter().map(|op| (op.unit, op.slot)).collect();
        let tickets: Vec<FillTicket> = fills.into_iter().map(|op| op.ticket).collect();
        let done = if tickets.is_empty() {
            Vec::new()
        } else {
            let batch = enqueue_fills(&mut self.policy, &self.dev, self.r, tickets, &mut self.copy)
                .map_err(|(_, failure)| failure.message)
                .expect("seed fills issue");
            self.dev.borrow_mut().run_all(copy_stream(self.r));
            self.policy
                .complete_fills(batch, &mut self.copy, true)
                .map_err(|_| "seed copy")
                .expect("seed fills complete")
        };
        let backing = self.backing.take().expect("the table backing");
        let generation = self
            .policy
            .finish_seed(done, backing)
            .map_err(|(_, error)| error)
            .expect("finish_seed");
        let deltas: Vec<PeerDelta> = seeded
            .iter()
            .map(|&(unit, slot)| PeerDelta {
                unit,
                slot: Some([slot as u64, 0]),
                canon: [0, 0],
            })
            .collect();
        for (unit, slot) in seeded {
            self.host_decode.insert(unit, (self.r, slot as u32));
        }
        self.upload();
        self.dev.borrow_mut().run_all(compute_stream(self.r));
        self.install(generation);
        if self.peer {
            self.board.post(self.r, 0, deltas);
        }
        self.sync_protected();
    }

    /// Lease the generation the next window reads.
    fn install(&mut self, generation: OwnedTableGeneration) {
        if self.shared {
            let lease = shared_lease(&generation);
            self.generation = Some(generation);
            self.lease = Some(lease);
        } else {
            self.lease = Some(
                acquire_window_owned(generation)
                    .map_err(|(_, error)| error)
                    .expect("lease"),
            );
        }
    }

    /// The slots the published generation names (from its closure).
    fn published_slots(&self) -> Vec<u32> {
        self.lease
            .as_ref()
            .map(|lease| lease.closure().slot_reads().map(|read| read.slot).collect())
            .unwrap_or_default()
    }

    /// Upload every host image that changed since its last upload (compute stream).
    fn upload(&mut self) {
        let mut dev = self.dev.borrow_mut();
        if self.host_decode != self.sent_decode {
            self.sent_decode = self.host_decode.clone();
            dev.push(
                compute_stream(self.r),
                Op::Upload {
                    table: Table::Decode(self.r),
                    image: self.host_decode.clone(),
                },
            );
        }
        if self.host_full != self.sent_full {
            self.sent_full = self.host_full.clone();
            dev.push(
                compute_stream(self.r),
                Op::Upload {
                    table: Table::Full(self.r),
                    image: self.host_full.clone(),
                },
            );
        }
    }

    /// Apply the peers' deltas to the full-table image.
    fn apply(&mut self, deltas: &[(usize, PeerDelta)]) {
        for &(owner, delta) in deltas {
            match delta.slot {
                Some([slot, _]) => {
                    self.host_full.insert(delta.unit, (owner, slot as u32));
                }
                None => {
                    self.host_full.remove(&delta.unit);
                }
            }
        }
    }

    /// Refresh the white-box set: slots a live generation or lease still references.
    fn sync_protected(&self) {
        let Some(bank) = self.policy.bank.as_ref() else {
            return;
        };
        let mut dev = self.dev.borrow_mut();
        dev.protected.retain(|&(owner, _)| owner != self.r);
        for slot in 0..self.t as u32 {
            if bank.refs(slot) > 0 {
                dev.protected.insert((self.r, slot));
            }
        }
    }

    /// The window's fence is recorded NOW (`lease.enqueued`): normally by [`Self::retire`], after
    /// the window's readers were enqueued.
    fn record_window_fence(&mut self) {
        if self.pending.is_some() {
            return;
        }
        if let Some(lease) = self.lease.take() {
            match lease.enqueued(&mut self.compute) {
                Ok(pending) => self.pending = Some(pending),
                Err((lease, _)) => self.lease = Some(lease),
            }
        }
    }

    /// One window's forward: the readers of this rank's tables, queued on the compute stream.
    fn forward(&mut self) {
        if self.variant == Variant::EarlyFence {
            // The bug: the last-read fence goes in before the final consumer is enqueued.
            self.record_window_fence();
        }
        let mut dev = self.dev.borrow_mut();
        dev.push(
            compute_stream(self.r),
            Op::Read {
                label: format!("rank {} decode reader", self.r),
                table: Table::Decode(self.r),
            },
        );
        if self.peer {
            dev.push(
                compute_stream(self.r),
                Op::Read {
                    label: format!("rank {} full-table reader", self.r),
                    table: Table::Full(self.r),
                },
            );
        }
    }

    /// A stage reader: reads the published generation's slots (its closure) on the stage stream
    /// while holding a shared lease on the generation (`Cfg::with_shared`).
    fn stage_hold(&mut self) {
        let lease = self.lease.as_ref().expect("a published generation");
        let reads: Vec<(UnitKey, usize, u32)> = lease
            .closure()
            .slot_reads()
            .map(|read| {
                let unit = self
                    .policy
                    .slot_unit(read.slot as usize)
                    .expect("a published slot holds a unit");
                (unit, self.r, read.slot)
            })
            .collect();
        self.dev.borrow_mut().push(
            stage_stream(self.r),
            Op::Pinned {
                label: format!("rank {} stage reader", self.r),
                reads,
            },
        );
        if self.variant == Variant::UnleasedStage {
            return;
        }
        let generation = self
            .generation
            .as_ref()
            .expect("a stage lease needs Cfg::with_shared");
        match shared_lease(generation).enqueued(&mut self.stage) {
            Ok(pending) => self.stage_pending = Some(pending),
            Err((_, failure)) => panic!("stage fence: {}", failure.message),
        }
    }

    /// The stage reader finished: wait its fence and release its lease. `false` while pending.
    fn stage_finish(&mut self) -> bool {
        let Some(pending) = self.stage_pending.take() else {
            return true;
        };
        match finish_window(pending, &mut self.stage) {
            Ok(_fenced) => true,
            Err(PendingOrQuarantined::Pending(pending)) => {
                self.stage_pending = Some(pending);
                false
            }
            Err(PendingOrQuarantined::Quarantined(q)) => {
                panic!("the stage fence quarantined: {:?}", q.failure())
            }
        }
    }

    fn freeze(&mut self, why: impl Into<String>) -> Step {
        let why = why.into();
        if self.frozen.is_none() {
            self.frozen = Some(why.clone());
            if self.peer {
                self.board.detach(self.r);
            }
        }
        Step::Frozen(why)
    }

    /// Record the ack of the peer posts through the take (the `AckBeforeFence` bug: before the
    /// window is fenced).
    fn early_ack(&mut self) {
        let j = self.policy.next_retirement();
        let take = self.board.take(self.r);
        self.board.ack(self.r, j, &take);
        self.early = Some(take.deltas.clone());
    }

    /// Record every deferred ack now.
    fn flush_ack(&mut self) {
        for (j, take) in std::mem::take(&mut self.deferred_acks) {
            self.board.ack(self.r, j, &take);
        }
    }

    /// Step 1: fence the window of the published generation.
    fn fence_window(&mut self) -> Result<OwnedTableGeneration, Step> {
        let pending = match self.pending.take() {
            Some(pending) => pending,
            None => {
                let lease = self
                    .lease
                    .take()
                    .ok_or_else(|| Step::Frozen("no published table generation".into()))?;
                match lease.enqueued(&mut self.compute) {
                    Ok(pending) => pending,
                    Err((lease, failure)) => {
                        self.lease = Some(lease);
                        return Err(self.freeze(failure.message));
                    }
                }
            }
        };
        match finish_window(pending, &mut self.compute) {
            Ok(fenced) => Ok(match fenced.into_generation() {
                Some(generation) => generation,
                None => self.generation.take().expect("the shared-lease generation"),
            }),
            Err(PendingOrQuarantined::Pending(pending)) => {
                self.pending = Some(pending);
                Err(Step::Blocked)
            }
            Err(PendingOrQuarantined::Quarantined(q)) => {
                let why = q.failure().message.clone();
                self.parked = Some(q);
                Err(self.freeze(why))
            }
        }
    }

    /// Step 2: poll the pending fill batches in issue order (stop at the first pending).
    fn poll_batches(&mut self) -> Result<Vec<FillComplete>, String> {
        let mut done = Vec::new();
        while let Some(batch) = self.batches.pop_front() {
            match self.policy.complete_fills(batch, &mut self.copy, false) {
                Ok(tokens) => done.extend(tokens),
                Err(FillsNotDone::Pending(batch)) => {
                    self.batches.push_front(batch);
                    break;
                }
                Err(FillsNotDone::Failed { failure, .. }) => return Err(failure.message),
            }
        }
        Ok(done)
    }

    fn issue(&mut self, tickets: Vec<FillTicket>) -> Result<(), String> {
        match enqueue_fills(&mut self.policy, &self.dev, self.r, tickets, &mut self.copy) {
            Ok(batch) => {
                self.batches.push_back(batch);
                Ok(())
            }
            Err((tickets, failure)) => {
                // Production's `issue_batch`: the returned tickets are never issued.
                self.policy.abandon_fills(tickets);
                Err(failure.message)
            }
        }
    }

    /// One retirement (steps 1-7 of the module docs).
    fn retire(&mut self, uses: &[WindowUse]) -> Step {
        let step = self.retire_steps(uses);
        self.sync_protected();
        step
    }

    fn retire_steps(&mut self, uses: &[WindowUse]) -> Step {
        if let Some(why) = &self.frozen {
            return Step::Frozen(why.clone());
        }
        if self.variant == Variant::AckBeforeFence && self.peer && self.early.is_none() {
            self.early_ack();
        }
        // (1) the window fence of the published generation.
        if self.held.is_none() {
            match self.fence_window() {
                Ok(generation) => self.held = Some(generation),
                Err(step) => return step,
            }
        }
        // (2) the batches that completed.
        match self.poll_batches() {
            Ok(done) => self.completed.extend(done),
            Err(why) => return self.freeze(why),
        }
        // (3) retire the generation; in PEER mode the one `released_through` call.
        let generation = self.held.take().expect("the fenced generation");
        let at = self.policy.next_retirement();
        let board = Arc::clone(&self.board);
        let lying = LyingFence;
        let fence: &dyn PeerReaderFence = if self.variant == Variant::LyingPeerFence {
            &lying
        } else {
            &*board
        };
        let peers = self.peer.then_some((fence, self.r, at));
        let (backing, retired) = match self.policy.retire_generation(generation, peers) {
            Ok(done) => done,
            Err((generation, error)) => {
                self.held = Some(generation);
                return Step::Refused(error);
            }
        };
        // (4) the policy step.
        let tokens = std::mem::take(&mut self.completed);
        if self.variant == Variant::TrustTokenNumbers {
            // The bug: a completion is believed by its (slot, generation) numbers.
            for token in &tokens {
                let numbers = (token.slot() as usize, token.fill_generation());
                if let Some(&(_, _, unit)) = self
                    .issued
                    .iter()
                    .find(|&&(slot, fill_gen, _)| (slot, fill_gen) == numbers)
                {
                    self.host_decode.insert(unit, (self.r, token.slot()));
                }
            }
        }
        let retire = if self.peer {
            self.policy
                .retire_peer_windows((0, 0), uses, tokens, 1, &retired)
        } else {
            self.policy
                .retire_windows((0, 0), uses, tokens, 1, &retired)
        };
        let j = self.policy.stats().retirements;
        // (5) table images, peer post / take / apply, uploads.
        for &(unit, slot) in &retire.publish {
            self.host_decode.insert(unit, (self.r, slot as u32));
        }
        for unit in &retire.unpublish {
            self.host_decode.remove(unit);
        }
        if self.variant == Variant::PublishBeforeFill {
            // The bug: the entry names the slot as soon as the fill is decided.
            for op in &retire.fills {
                self.host_decode.insert(op.unit, (self.r, op.slot as u32));
            }
        }
        for op in &retire.fills {
            self.issued.push((op.slot, op.generation, op.unit));
        }
        let mut take = None;
        if self.peer {
            self.board.post(self.r, j, peer_deltas(&retire));
            let fresh = self.board.take(self.r);
            let mut deltas = self.early.take().unwrap_or_default();
            deltas.extend(fresh.deltas.iter().copied());
            let ahead = fresh.newest().is_some_and(|k| k > j);
            if ahead && self.variant != Variant::AckWithoutApply {
                self.backing = Some(backing);
                abandon_retire_fills(&mut self.policy, retire);
                return self.freeze(format!(
                    "peer posts through {:?} reached rank {} at its retirement {j}: the ranks' \
                     windows are out of step",
                    fresh.newest(),
                    self.r
                ));
            }
            if !ahead {
                self.apply(&deltas);
            }
            take = Some(fresh);
        }
        self.upload();
        // (6) seal, lease, ack.
        let generation = match self.policy.seal(backing, GenerationId { epoch: j, sub: 0 }) {
            Ok(generation) => generation,
            Err((backing, error)) => {
                self.backing = Some(backing);
                abandon_retire_fills(&mut self.policy, retire);
                return self.freeze(format!("sealing generation {j}: {error}"));
            }
        };
        self.install(generation);
        if let Some(take) = take {
            match self.ack {
                Ack::Now => self.board.ack(self.r, j, &take),
                Ack::Withhold => {}
                Ack::Defer => self.deferred_acks.push((j, take)),
            }
        }
        // (7) the fills.
        let Retire {
            publish,
            unpublish,
            fills,
        } = retire;
        let keys = fills.iter().map(|op| op.key()).collect();
        if !fills.is_empty() {
            if self.gated {
                let ready = self.compute.record().expect("the tables-ready event");
                self.dev
                    .borrow_mut()
                    .push(copy_stream(self.r), Op::WaitMarker(ready));
            }
            let tickets = fills.into_iter().map(|op| op.ticket).collect();
            if let Err(why) = self.issue(tickets) {
                return self.freeze(why);
            }
        }
        Step::Retired(Outcome {
            publish,
            unpublish,
            fills: keys,
        })
    }

    // --- teardown and borrow

    /// Fence and retire the last generation (production's `fence_and_retire`).
    fn retire_last(&mut self) -> Result<(), String> {
        if self.retired_last {
            return Ok(());
        }
        if self.held.is_none() {
            match self.fence_window() {
                Ok(generation) => self.held = Some(generation),
                Err(step) => return Err(format!("the last window did not fence: {step:?}")),
            }
        }
        let generation = self.held.take().expect("the fenced generation");
        match self.policy.retire_generation(generation, None) {
            Ok(_) => {
                self.retired_last = true;
                Ok(())
            }
            Err((generation, error)) => {
                self.held = Some(generation);
                Err(error.to_string())
            }
        }
    }

    /// Device idle, then close the bank (no window-fence check first).
    fn close_idle(&mut self) -> Result<(LentBlock<Expert>, BlockReclaimed), String> {
        let q = self
            .policy
            .quiesce(&mut self.device)
            .map_err(|failure| failure.message)?;
        self.policy.close(q).map_err(|error| error.to_string())
    }

    /// Teardown: fence + retire the last generation, device idle, close, free the pool. The pool
    /// is freed only when the bank closed (the `ReleaseEarly` control frees it regardless).
    /// Returns whether the pool was freed.
    fn release(&mut self) -> bool {
        let closed = self.retire_last().and_then(|()| self.close_idle());
        let freed = match closed {
            Ok((block, done)) => {
                block
                    .free(done)
                    .map_err(|(_, _, error)| error)
                    .expect("a closed pool frees");
                true
            }
            Err(_) => self.variant == Variant::ReleaseEarly,
        };
        if freed {
            self.dev.borrow_mut().freed[self.r] = true;
        }
        freed
    }

    /// The borrow's suspend tail: device idle, then the whole-pool scratch loan.
    fn lend(&mut self) -> Result<ScratchLoan, ResidencyError> {
        let q = self
            .policy
            .quiesce(&mut self.device)
            .map_err(|failure| ResidencyError::Refused(failure.message))?;
        self.policy.lend_scratch(q)
    }
}

// ------------------------------------------------------------------------------------- mesh

/// Two ranks over one device and one board.
struct Mesh {
    dev: Shared,
    ranks: Vec<SimRank>,
}

impl Mesh {
    fn new(cfg: Cfg) -> Self {
        let dev = Dev::new(cfg.t);
        let board = Arc::new(PeerBoard::new(RANKS));
        let ranks = (0..RANKS)
            .map(|r| SimRank::new(&dev, &board, r, &cfg))
            .collect();
        Self { dev, ranks }
    }

    fn dev(&self) -> RefMut<'_, Dev> {
        self.dev.borrow_mut()
    }

    fn seed(&mut self, rank: usize, units: &[UnitKey]) {
        self.ranks[rank].seed(units);
    }

    /// The unit tags of `rank`'s slot bytes.
    fn bytes(&self, rank: usize) -> Vec<Option<UnitKey>> {
        self.dev.borrow().slot_bytes[rank].clone()
    }

    fn assert_clean(&self) {
        assert_clean(&self.dev);
    }

    /// One lockstep window group: both ranks' forwards, the row exchange (compute streams run
    /// dry), the retirement of rank 0 then rank 1, then the uploads and fill copies.
    fn group(&mut self, uses: [&[WindowUse]; 2]) -> [Step; 2] {
        for rank in &mut self.ranks {
            rank.forward();
        }
        self.dev.borrow_mut().run_compute();
        let first = self.ranks[0].retire(uses[0]);
        let second = self.ranks[1].retire(uses[1]);
        self.dev.borrow_mut().settle();
        [first, second]
    }
}

fn assert_clean(dev: &Shared) {
    let dev = dev.borrow();
    assert!(
        dev.violations.is_empty(),
        "the checker saw races: {:#?}",
        dev.violations
    );
}

// ------------------------------------------------------------------------------------ tests

const A: UnitKey = (0, 1);
const B: UnitKey = (0, 2);
const C: UnitKey = (0, 3);
const X: UnitKey = (1, 1);
const Y: UnitKey = (1, 2);

fn hit(unit: UnitKey) -> WindowUse {
    WindowUse { unit, reads: 1 }
}

/// A peer mesh of 2-slot ranks: rank 0 owns `A, B`, rank 1 owns `X, Y`; one lockstep window in
/// which rank 0 reads `A, B` (retirement 1: every table named, every peer acknowledged once).
fn peer_mesh() -> Mesh {
    let mut m = Mesh::new(Cfg::new(2, 2).with_peer());
    m.seed(0, &[A, B]);
    m.seed(1, &[X, Y]);
    let [first, second] = m.group([&[hit(A), hit(B)], &[]]);
    first.retired();
    second.retired();
    m
}

/// One window of rank 0 ALONE that reads `A` and misses `C`: its forward and compute stream run,
/// then it retires. The first one evicts `B` (slot 1).
fn owner_group(m: &mut Mesh) -> Step {
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    m.ranks[0].retire(&[hit(A), hit(C)])
}

/// Rank 0's uploads and copies execute.
fn settle_owner(m: &Mesh) {
    m.dev().run_all(compute_stream(0));
    m.dev().run_all(copy_stream(0));
}

/// A completion token of ANOTHER bank (slot 0, fill generation 1).
fn decoy() -> FillComplete {
    let config = EpUnitTailConfig {
        tail_units: 1,
        fill_per_step: 1,
        boundary_every: 1,
    };
    Rig::new(config, false, HashMap::new()).decoy_token()
}

/// The checker's own sanity: a copy onto a slot a live generation references is flagged.
#[test]
fn race_checker_flags_a_write_to_a_live_slot() {
    let dev = Dev::new(2);
    {
        let mut d = dev.borrow_mut();
        d.protected.insert((0, 1));
        let spans = d.extents[1].to_vec();
        d.push(
            copy_stream(0),
            Op::Fill {
                owner: 0,
                slot: 1,
                unit: A,
                spans,
            },
        );
        d.run_all(copy_stream(0));
        assert!(d.any(|v| matches!(v, Violation::WriteLive { slot: 1, .. })));
        // a copy that strays outside the slot's extent is flagged too
        d.push(
            copy_stream(0),
            Op::Fill {
                owner: 0,
                slot: 0,
                unit: B,
                spans: vec![Binding { base: 0, bytes: 1 }],
            },
        );
        d.run_all(copy_stream(0));
        assert!(d.any(|v| matches!(v, Violation::BadSpan { slot: 0, .. })));
    }
}

// 1. delayed last compute reader ------------------------------------------------------------

#[test]
fn race_delayed_last_reader_blocks_retirement() {
    let mut m = Mesh::new(Cfg::new(2, 2));
    m.seed(0, &[A, B]);
    m.ranks[0].forward();
    m.dev().set_wait(compute_stream(0), Wait::Pending);
    let uses = [hit(A), hit(C)];
    // The reader has not run, the window fence cannot finish: the retirement does not happen
    // (and may be retried: the pending window is kept).
    for _ in 0..2 {
        assert!(matches!(m.ranks[0].retire(&uses), Step::Blocked));
    }
    {
        let policy = &m.ranks[0].policy;
        assert_eq!(policy.stats().retirements, 0);
        assert_eq!(policy.count(SlotPhase::Live), 2);
        assert_eq!((bank(policy).refs(0), bank(policy).refs(1)), (1, 1));
    }
    assert_eq!(m.dev().queued(copy_stream(0)), 0, "no slot was refilled");
    assert_eq!(m.bytes(0), vec![Some(A), Some(B)]);
    m.assert_clean();
    // The reader finishes: the retirement proceeds and refills B's slot.
    m.dev().set_wait(compute_stream(0), Wait::Run);
    let out = m.ranks[0].retire(&uses).retired();
    assert_eq!(out.unpublish, vec![B]);
    assert_eq!(out.fills, vec![(1, C, 2)]);
    m.dev().settle();
    assert_eq!(m.bytes(0), vec![Some(A), Some(C)]);
    m.assert_clean();
}

#[test]
fn race_delayed_last_reader_positive_control() {
    let mut m = Mesh::new(Cfg::new(2, 2));
    m.seed(0, &[A, B]);
    m.ranks[0].variant = Variant::EarlyFence;
    // The fence goes in before the window's reader: the retirement is premature.
    m.ranks[0].forward();
    let out = m.ranks[0].retire(&[hit(A), hit(C)]).retired();
    assert_eq!(out.fills, vec![(1, C, 2)]);
    // The refill copy runs before the delayed reader.
    m.dev().run_all(copy_stream(0));
    m.dev().run_all(compute_stream(0));
    assert!(m.dev.borrow().any(|v| matches!(
        v,
        Violation::Read { unit, slot: 1, found: Some(found), .. } if *unit == B && *found == C
    )));
    assert!(m
        .dev
        .borrow()
        .any(|v| matches!(v, Violation::WriteNamed { slot: 1, .. })));
}

// 2. delayed full-table reader (a peer reading the owner's slots) ----------------------------

#[test]
fn race_delayed_full_table_reader_keeps_the_slot_quarantined() {
    let mut m = peer_mesh();
    // Window 2: rank 1's full-table reader (it names rank 0's A and B) is delayed.
    m.ranks[1].forward();
    m.dev().set_wait(compute_stream(1), Wait::Pending);
    let out = owner_group(&mut m).retired();
    assert_eq!(out.unpublish, vec![B]);
    assert!(
        out.fills.is_empty(),
        "a peer eviction never refills at once"
    );
    // Rank 1 cannot fence its window: no retirement, no ack.
    assert!(matches!(m.ranks[1].retire(&[]), Step::Blocked));
    // The owner goes on retiring: B's slot stays quarantined and nothing is refilled.
    for _ in 0..2 {
        let out = owner_group(&mut m).retired();
        assert!(out.fills.is_empty() && out.unpublish.is_empty(), "{out:?}");
        let policy = &m.ranks[0].policy;
        assert_eq!(policy.slot_phase(1), SlotPhase::Quarantined);
        assert_eq!(policy.slot_evicted_at(1), Some(2));
    }
    settle_owner(&m);
    assert!(m.ranks[0].policy.stats().skipped_full >= 2);
    assert_eq!(m.bytes(0), vec![Some(A), Some(B)]);
    // The delayed reader finally runs and finds B's bytes intact.
    m.dev().set_wait(compute_stream(1), Wait::Run);
    m.dev().run_all(compute_stream(1));
    m.assert_clean();
}

#[test]
fn race_delayed_full_table_reader_positive_control() {
    let mut m = peer_mesh();
    m.ranks[0].variant = Variant::LyingPeerFence;
    m.ranks[1].forward();
    m.dev().set_wait(compute_stream(1), Wait::Pending);
    owner_group(&mut m).retired();
    // Refill without the peer's ack: the owner reclaims B's slot at retirement 3.
    let out = owner_group(&mut m).retired();
    assert_eq!(out.fills, vec![(1, C, 2)]);
    settle_owner(&m);
    m.dev().set_wait(compute_stream(1), Wait::Run);
    m.dev().run_all(compute_stream(1));
    assert!(m.dev.borrow().any(|v| matches!(
        v,
        Violation::WriteNamed {
            named_by: Table::Full(1),
            slot: 1,
            ..
        }
    )));
    assert!(m.dev.borrow().any(|v| matches!(
        v,
        Violation::Read { unit, owner: 0, slot: 1, found: Some(found), .. }
            if *unit == B && *found == C
    )));
}

// 3. peer ack recorded late --------------------------------------------------------------------

#[test]
fn race_late_peer_ack_keeps_the_slot_quarantined() {
    let mut m = peer_mesh();
    m.ranks[1].ack = Ack::Defer;
    // Window 2: B is evicted; rank 1 applies the eviction and enqueues its upload, but its ack
    // is not recorded yet.
    let [first, second] = m.group([&[hit(A), hit(C)], &[]]);
    assert_eq!(first.retired().unpublish, vec![B]);
    second.retired();
    // Window 3: the owner sees no ack covering retirement 2: no reclaim, no refill.
    let [first, second] = m.group([&[hit(A), hit(C)], &[]]);
    let out = first.retired();
    second.retired();
    assert!(out.fills.is_empty(), "{out:?}");
    {
        let policy = &m.ranks[0].policy;
        assert_eq!(policy.slot_phase(1), SlotPhase::Quarantined);
        assert_eq!(policy.slot_evicted_at(1), Some(2));
        assert_eq!(policy.stats().skipped_full, 1);
    }
    assert_eq!(m.bytes(0), vec![Some(A), Some(B)]);
    // The acks land: the next retirement reclaims the slot and refills it.
    m.ranks[1].flush_ack();
    let [first, second] = m.group([&[hit(A), hit(C)], &[]]);
    let out = first.retired();
    second.retired();
    assert_eq!(out.fills, vec![(1, C, 2)]);
    assert_eq!(m.bytes(0), vec![Some(A), Some(C)]);
    m.assert_clean();
}

#[test]
fn race_late_peer_ack_positive_control() {
    let mut m = peer_mesh();
    m.ranks[1].variant = Variant::AckBeforeFence;
    // Window 2: rank 1's reader is delayed. It acknowledges the owner's eviction BEFORE it fenced
    // its window (the bug), then blocks on the fence.
    m.ranks[1].forward();
    m.dev().set_wait(compute_stream(1), Wait::Pending);
    owner_group(&mut m).retired();
    assert!(matches!(m.ranks[1].retire(&[]), Step::Blocked));
    // The owner trusts the ack, reclaims B's slot at retirement 3 and refills it.
    let out = owner_group(&mut m).retired();
    assert_eq!(out.fills, vec![(1, C, 2)]);
    settle_owner(&m);
    m.dev().set_wait(compute_stream(1), Wait::Run);
    m.dev().run_all(compute_stream(1));
    assert!(m.dev.borrow().any(|v| matches!(
        v,
        Violation::WriteNamed {
            named_by: Table::Full(1),
            slot: 1,
            ..
        }
    )));
    assert!(m.dev.borrow().any(|v| matches!(
        v,
        Violation::Read { unit, slot: 1, found: Some(found), .. } if *unit == B && *found == C
    )));
}

// 4. delayed stage reader --------------------------------------------------------------------

#[test]
fn race_delayed_stage_reader_refuses_retirement() {
    let mut m = Mesh::new(Cfg::new(2, 2).with_shared());
    m.seed(0, &[A, B]);
    // The stage reader holds a lease on the published generation; its reads are delayed.
    m.ranks[0].stage_hold();
    m.dev().set_wait(stage_stream(0), Wait::Pending);
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    let uses = [hit(A), hit(C)];
    // The window fenced but the generation cannot retire: refused, and it comes back each time.
    for _ in 0..2 {
        assert!(matches!(
            m.ranks[0].retire(&uses),
            Step::Refused(ResidencyError::Refused(_))
        ));
    }
    assert!(
        !m.ranks[0].stage_finish(),
        "the stage fence is still pending"
    );
    assert!(matches!(m.ranks[0].retire(&uses), Step::Refused(_)));
    {
        let policy = &m.ranks[0].policy;
        assert_eq!(policy.stats().retirements, 0);
        assert_eq!(policy.count(SlotPhase::Live), 2);
        assert_eq!((bank(policy).refs(0), bank(policy).refs(1)), (1, 1));
    }
    assert_eq!(m.dev().queued(copy_stream(0)), 0);
    assert_eq!(m.bytes(0), vec![Some(A), Some(B)]);
    // The stage reader runs, sees intact bytes and releases its lease: the retirement proceeds.
    m.dev().set_wait(stage_stream(0), Wait::Run);
    assert!(m.ranks[0].stage_finish());
    let out = m.ranks[0].retire(&uses).retired();
    assert_eq!(out.fills, vec![(1, C, 2)]);
    m.dev().settle();
    assert_eq!(m.bytes(0), vec![Some(A), Some(C)]);
    m.assert_clean();
}

#[test]
fn race_delayed_stage_reader_positive_control() {
    let mut m = Mesh::new(Cfg::new(2, 2));
    m.seed(0, &[A, B]);
    m.ranks[0].variant = Variant::UnleasedStage;
    // The stage reader holds no lease: the window fence says nothing about it.
    m.ranks[0].stage_hold();
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    let out = m.ranks[0].retire(&[hit(A), hit(C)]).retired();
    assert_eq!(out.fills, vec![(1, C, 2)]);
    m.dev().run_all(copy_stream(0));
    m.dev().run_all(stage_stream(0));
    assert!(m.dev.borrow().any(|v| matches!(
        v,
        Violation::Read { unit, slot: 1, found: Some(found), .. } if *unit == B && *found == C
    )));
}

// 5a. delayed fill event ---------------------------------------------------------------------

#[test]
fn race_delayed_fill_event_is_never_published() {
    let mut m = Mesh::new(Cfg::new(2, 2));
    m.seed(0, &[A, B]);
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    let out = m.ranks[0].retire(&[hit(A), hit(C)]).retired();
    assert_eq!(out.fills, vec![(1, C, 2)]);
    m.dev().run_all(compute_stream(0));
    // Window 2: the fill's copy (and its event) has not run: no completion token, no publication.
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    let out = m.ranks[0].retire(&[hit(A)]).retired();
    assert!(out.publish.is_empty());
    {
        let policy = &m.ranks[0].policy;
        assert_eq!(policy.count(SlotPhase::Filling), 1);
        assert_eq!(policy.live_slot(C), None);
    }
    assert_eq!(m.ranks[0].published_slots(), vec![0]);
    assert!(!m.dev.borrow().images[&Table::Decode(0)].contains_key(&C));
    m.assert_clean();
    // The copy runs; the NEXT retirement polls the token and only then publishes.
    m.dev().settle();
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    let out = m.ranks[0].retire(&[hit(A)]).retired();
    assert_eq!(out.publish, vec![(C, 1)]);
    m.dev().settle();
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    assert_eq!(m.ranks[0].published_slots(), vec![0, 1]);
    assert_eq!(m.bytes(0), vec![Some(A), Some(C)]);
    m.assert_clean();
}

#[test]
fn race_delayed_fill_event_positive_control() {
    let mut m = Mesh::new(Cfg::new(2, 2));
    m.seed(0, &[A, B]);
    m.ranks[0].variant = Variant::PublishBeforeFill;
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    let out = m.ranks[0].retire(&[hit(A), hit(C)]).retired();
    assert_eq!(out.fills, vec![(1, C, 2)]);
    // The table names C's slot before its copy ran; the next window's reader sees B's old bytes.
    m.dev().run_all(compute_stream(0));
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    assert!(m.dev.borrow().any(|v| matches!(
        v,
        Violation::Read { unit, slot: 1, found: Some(found), .. } if *unit == C && *found == B
    )));
}

// 5b. delayed table upload --------------------------------------------------------------------

#[test]
fn race_delayed_table_upload_orders_the_fill() {
    let mut m = Mesh::new(Cfg::new(2, 2).with_gate());
    m.seed(0, &[A, B]);
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    let out = m.ranks[0].retire(&[hit(A), hit(C)]).retired();
    assert_eq!(out.fills, vec![(1, C, 2)]);
    // The upload has not executed: the device table still names B's slot, so the fill copy
    // (gated by the tables-ready event) cannot start.
    assert_eq!(m.dev().run_all(copy_stream(0)), 0);
    assert_eq!(m.bytes(0), vec![Some(A), Some(B)]);
    assert_eq!(
        m.dev.borrow().images[&Table::Decode(0)].get(&B),
        Some(&(0, 1))
    );
    m.assert_clean();
    // The upload and the event execute, then the copy.
    m.dev().run_all(compute_stream(0));
    assert!(!m.dev.borrow().images[&Table::Decode(0)].contains_key(&B));
    m.dev().run_all(copy_stream(0));
    assert_eq!(m.bytes(0), vec![Some(A), Some(C)]);
    m.assert_clean();
}

#[test]
fn race_delayed_table_upload_positive_control() {
    // No tables-ready gate: the copy starts while the table upload is still queued.
    let mut m = Mesh::new(Cfg::new(2, 2));
    m.seed(0, &[A, B]);
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    m.ranks[0].retire(&[hit(A), hit(C)]).retired();
    m.dev().run_all(copy_stream(0));
    assert!(m.dev.borrow().any(|v| matches!(
        v,
        Violation::WriteNamed {
            named_by: Table::Decode(0),
            slot: 1,
            ..
        }
    )));
}

// 6. reordered peer post: out-of-step freeze ------------------------------------------------------

#[test]
fn race_reordered_peer_post_freezes_without_ack() {
    let mut m = peer_mesh();
    // Window 2 on both devices; the owner retires 2 (evicting B) and 3 before rank 1 retires 2.
    m.ranks[0].forward();
    m.ranks[1].forward();
    m.dev().run_compute();
    let out = m.ranks[0].retire(&[hit(A), hit(C)]).retired();
    assert_eq!(out.unpublish, vec![B]);
    owner_group(&mut m).retired();
    // Rank 1 takes a post newer than its own retirement counter: it freezes, no ack.
    let frozen = m.ranks[1].retire(&[]);
    assert!(
        matches!(&frozen, Step::Frozen(why) if why.contains("out of step")),
        "{frozen:?}"
    );
    assert!(m.ranks[1].frozen.is_some());
    // The owner never reclaims B's slot, however many retirements follow.
    for _ in 0..3 {
        let out = owner_group(&mut m).retired();
        assert!(out.fills.is_empty() && out.unpublish.is_empty(), "{out:?}");
        let policy = &m.ranks[0].policy;
        assert_eq!(policy.slot_phase(1), SlotPhase::Quarantined);
        assert_eq!(policy.slot_evicted_at(1), Some(2));
    }
    assert!(m.ranks[0].policy.stats().skipped_full >= 3);
    settle_owner(&m);
    // The frozen rank keeps reading its last uploaded tables, which still name B's slot.
    m.ranks[1].forward();
    m.dev().run_all(compute_stream(1));
    assert_eq!(m.bytes(0), vec![Some(A), Some(B)]);
    m.assert_clean();
}

/// The out-of-step freeze hits rank 0 at the retirement that reclaims `B`'s slot and decides
/// `C`'s fill: the fill's ticket is in hand when the tail freezes.
fn frozen_with_a_decided_fill() -> Mesh {
    let mut m = peer_mesh();
    // Window 2 in lockstep: rank 0 evicts B, rank 1 applies the eviction and acks it.
    m.ranks[0].forward();
    m.ranks[1].forward();
    m.dev().run_compute();
    let out = m.ranks[0].retire(&[hit(A), hit(C)]).retired();
    assert_eq!(out.unpublish, vec![B]);
    m.ranks[1].retire(&[]).retired();
    // Rank 1 runs two windows ahead alone: its posts reach retirement 4.
    for _ in 0..2 {
        m.ranks[1].forward();
        m.dev().run_all(compute_stream(1));
        m.ranks[1].retire(&[]).retired();
    }
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    let frozen = m.ranks[0].retire(&[hit(A), hit(C)]);
    assert!(
        matches!(&frozen, Step::Frozen(why) if why.contains("out of step")),
        "{frozen:?}"
    );
    // The fill was decided (slot 1 is Filling) but never issued.
    assert_eq!(m.ranks[0].policy.slot_phase(1), SlotPhase::Filling);
    assert_eq!(m.dev().queued(copy_stream(0)), 0);
    m
}

/// A tail frozen mid-retirement with a decided fill abandons the fill's ticket: the pool can
/// still be lent as scratch and the bank still closes (beta's frozen tail stayed borrowable).
#[test]
fn race_frozen_mid_retirement_abandons_its_fills() {
    let mut m = frozen_with_a_decided_fill();
    m.ranks[0]
        .lend()
        .expect("a frozen tail still lends its pool");
    m.dev().settle();
    m.assert_clean();

    let mut m = frozen_with_a_decided_fill();
    let (block, done) = m.ranks[0].close_idle().expect("a frozen tail still closes");
    block
        .free(done)
        .map_err(|(_, _, error)| error)
        .expect("a closed pool frees");
    m.assert_clean();
}

#[test]
fn race_reordered_peer_post_positive_control() {
    let mut m = peer_mesh();
    m.ranks[1].variant = Variant::AckWithoutApply;
    m.ranks[0].forward();
    m.ranks[1].forward();
    m.dev().run_compute();
    m.ranks[0].retire(&[hit(A), hit(C)]).retired();
    owner_group(&mut m).retired();
    // Rank 1 sees the newer post, does not freeze, and acknowledges what it never applied.
    m.ranks[1].retire(&[]).retired();
    let out = owner_group(&mut m).retired();
    assert_eq!(out.fills, vec![(1, C, 2)]);
    settle_owner(&m);
    m.ranks[1].forward();
    m.dev().run_all(compute_stream(1));
    assert!(m.dev.borrow().any(|v| matches!(
        v,
        Violation::WriteNamed {
            named_by: Table::Full(1),
            slot: 1,
            ..
        }
    )));
    assert!(m.dev.borrow().any(|v| matches!(
        v,
        Violation::Read { unit, slot: 1, found: Some(found), .. } if *unit == B && *found == C
    )));
}

// 7. withheld peer ack ---------------------------------------------------------------------------

#[test]
fn race_withheld_peer_ack_bounds_the_quarantine() {
    const T: usize = 8;
    const F: usize = 1;
    const WINDOWS: usize = 8;
    let mut m = Mesh::new(Cfg::new(T, F).with_peer());
    let owned: Vec<UnitKey> = (1..=T as u16).map(|e| (0, e)).collect();
    let peers: Vec<UnitKey> = (1..=T as u16).map(|e| (1, e)).collect();
    m.seed(0, &owned);
    m.seed(1, &peers);
    m.ranks[1].ack = Ack::Withhold;
    // Every window reads the first two units and misses (0, 20): each retirement would evict one
    // more slot, but no eviction is ever acknowledged.
    let uses = [hit((0, 1)), hit((0, 2)), hit((0, 20))];
    for _ in 0..WINDOWS {
        let [first, second] = m.group([&uses, &[]]);
        let out = first.retired();
        second.retired();
        assert!(out.fills.is_empty(), "{out:?}");
    }
    // The bound of a peer per-window policy: the reserve target, `RESERVE_RETIREMENTS x` the one
    // candidate each retirement considers.
    const BOUND: usize = RESERVE_RETIREMENTS * F;
    let policy = &m.ranks[0].policy;
    assert_eq!(policy.count(SlotPhase::Quarantined), BOUND);
    assert_eq!(policy.count(SlotPhase::Live), T - BOUND);
    assert_eq!(policy.count(SlotPhase::Free), 0);
    let stats = policy.stats();
    assert_eq!(stats.quarantined, BOUND as u64);
    assert_eq!(stats.evictions, BOUND as u64);
    assert_eq!(stats.fills_issued, 0);
    assert_eq!(stats.skipped_full, (WINDOWS - BOUND) as u64);
    // The quarantined slots (2, 3) kept their bytes.
    let bytes = m.bytes(0);
    for slot in 2..2 + BOUND {
        assert_eq!(bytes[slot], Some((0, slot as u16 + 1)));
    }
    m.assert_clean();
}

#[test]
fn race_withheld_peer_ack_positive_control() {
    let mut m = peer_mesh();
    // Rank 1 stalls after the first window (no retirement, no ack) and the owner's fence claims
    // every peer acknowledged.
    m.ranks[0].variant = Variant::LyingPeerFence;
    owner_group(&mut m).retired();
    let out = owner_group(&mut m).retired();
    assert_eq!(out.fills, vec![(1, C, 2)]);
    settle_owner(&m);
    m.ranks[1].forward();
    m.dev().run_all(compute_stream(1));
    assert!(m.dev.borrow().any(|v| matches!(
        v,
        Violation::WriteNamed {
            named_by: Table::Full(1),
            slot: 1,
            ..
        }
    )));
    assert!(m.dev.borrow().any(|v| matches!(
        v,
        Violation::Read { unit, slot: 1, found: Some(found), .. } if *unit == B && *found == C
    )));
}

// 8. forced stale completion, stale generation bind ----------------------------------------------

/// Rank 0 cold-started (empty generation 0): the first window misses `A, B`, so fills of slots 0
/// and 1 (fill generation 1) are queued on the copy stream, which has not run.
fn cold_start_fills() -> Mesh {
    let mut m = Mesh::new(Cfg::new(2, 2));
    m.seed(0, &[]);
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    let out = m.ranks[0].retire(&[hit(A), hit(B)]).retired();
    assert_eq!(out.fills, vec![(0, A, 1), (1, B, 1)]);
    m
}

#[test]
fn race_stale_completion_and_stale_bind_never_publish() {
    // (a) A completion token of another bank, with the very (slot, generation) numbers of a
    // pending fill, is ignored and counted; the fill is published only by its own token.
    let mut m = cold_start_fills();
    m.ranks[0].completed.push(decoy());
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    let out = m.ranks[0].retire(&[hit(A)]).retired();
    assert!(out.publish.is_empty());
    {
        let policy = &m.ranks[0].policy;
        assert_eq!(policy.stats().stale_completions, 1);
        assert_eq!(policy.count(SlotPhase::Filling), 2);
        assert_eq!(policy.live_slot(A), None);
    }
    assert!(m.ranks[0].published_slots().is_empty());
    m.dev().settle();
    m.assert_clean();
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    let out = m.ranks[0].retire(&[hit(A), hit(B)]).retired();
    assert_eq!(out.publish, vec![(A, 0), (B, 1)]);
    m.dev().settle();
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    assert_eq!(m.bytes(0), vec![Some(A), Some(B)]);
    m.assert_clean();

    // (b) The copy stream was recreated after the fills were issued: its fence is stale, the
    // batch fails with unknown quiescence, the slots are quarantined forever and the rank freezes.
    let mut m = cold_start_fills();
    m.dev().recreate(copy_stream(0));
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    assert!(matches!(m.ranks[0].retire(&[hit(A)]), Step::Frozen(_)));
    {
        let policy = &m.ranks[0].policy;
        assert!(bank(policy).is_quarantined(0) && bank(policy).is_quarantined(1));
        assert_eq!(policy.count(SlotPhase::Filling), 2);
    }
    assert!(m.ranks[0].published_slots().is_empty());
    assert!(matches!(m.ranks[0].lend(), Err(ResidencyError::Refused(_))));
    m.dev().settle();
    m.assert_clean();

    // (c) The compute stream was recreated after the window fence was recorded: the stale fence
    // quarantines the generation, which never retires; its slots keep their bytes.
    let mut m = Mesh::new(Cfg::new(2, 2));
    m.seed(0, &[A, B]);
    m.ranks[0].forward();
    m.ranks[0].record_window_fence();
    m.dev().recreate(compute_stream(0));
    assert!(matches!(
        m.ranks[0].retire(&[hit(A), hit(C)]),
        Step::Frozen(_)
    ));
    {
        let policy = &m.ranks[0].policy;
        assert_eq!(bank(policy).live_generations(), 1);
        assert_eq!((bank(policy).refs(0), bank(policy).refs(1)), (1, 1));
        assert_eq!(policy.stats().retirements, 0);
    }
    assert!(matches!(m.ranks[0].lend(), Err(ResidencyError::Refused(_))));
    m.dev().settle();
    assert_eq!(m.bytes(0), vec![Some(A), Some(B)]);
    m.assert_clean();
}

#[test]
fn race_stale_completion_positive_control() {
    let mut m = cold_start_fills();
    m.ranks[0].variant = Variant::TrustTokenNumbers;
    m.ranks[0].completed.push(decoy());
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    m.ranks[0].retire(&[hit(A)]).retired();
    // The foreign token's numbers published A's table entry although its copy never ran.
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    assert!(m.dev.borrow().any(|v| matches!(
        v,
        Violation::Read { unit, slot: 0, found: None, .. } if *unit == A
    )));
}

#[test]
fn race_stale_fence_positive_control() {
    let mut m = cold_start_fills();
    {
        // The stream was recreated (its queued copies are lost) but the backend reuses the old
        // incarnation and reports the lost fence complete.
        let mut dev = m.dev();
        dev.reset(copy_stream(0));
        dev.lie(copy_stream(0), true);
    }
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    let out = m.ranks[0].retire(&[hit(A), hit(B)]).retired();
    assert_eq!(out.publish, vec![(A, 0), (B, 1)]);
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    assert!(m.dev.borrow().any(|v| matches!(
        v,
        Violation::Read { unit, found: None, .. } if *unit == A
    )));
}

// 9. an old lease held while a refill is attempted -----------------------------------------------

/// A non-peer rank driven by hand with two table backings, so that two generations can be live
/// at once. Slots 0 and 1 hold `A` and `B` (fill generation 1), generation 0 names both.
struct Pipelined {
    dev: Shared,
    policy: TailPolicy,
    g0: OwnedTableGeneration,
    spare: TableBacking,
    compute: Backend,
    copy: Backend,
    stage: Backend,
}

impl Pipelined {
    fn new() -> Self {
        let dev = Dev::new(2);
        let config = EpUnitTailConfig {
            tail_units: 2,
            fill_per_step: 2,
            boundary_every: 1,
        };
        let (block, spans) = rig_pool(2);
        let mut policy = TailPolicy::new(config, false, HashMap::new(), false, block, spans)
            .map_err(|(_, error)| error)
            .expect("the policy");
        let table = |base| {
            vec![Binding {
                base,
                bytes: 0x1000,
            }]
        };
        let t0 = policy
            .register_table_backing(table(RIG_TABLE))
            .expect("backing 0");
        let spare = policy
            .register_table_backing(table(RIG_TABLE + 0x1000))
            .expect("backing 1");
        let mut copy = Backend::new(&dev, copy_stream(0));
        let tickets = policy
            .seed(&[A, B])
            .expect("seed")
            .into_iter()
            .map(|op| op.ticket)
            .collect();
        let batch = enqueue_fills(&mut policy, &dev, 0, tickets, &mut copy)
            .map_err(|(_, failure)| failure.message)
            .expect("seed issue");
        dev.borrow_mut().settle();
        let done = policy
            .complete_fills(batch, &mut copy, true)
            .map_err(|_| "seed copy")
            .expect("seed complete");
        let g0 = policy
            .finish_seed(done, t0)
            .map_err(|(_, error)| error)
            .expect("finish_seed");
        dev.borrow_mut()
            .images
            .insert(Table::Decode(0), Image::from([(A, (0, 0)), (B, (0, 1))]));
        Self {
            compute: Backend::new(&dev, compute_stream(0)),
            stage: Backend::new(&dev, stage_stream(0)),
            copy,
            dev,
            policy,
            g0,
            spare,
        }
    }
}

/// Seal generation 1 (naming A and B again) through the spare backing, fence its window and
/// retire it: the token of a retirement while generation 0 is still live.
fn retire_pipelined_generation(
    policy: &mut TailPolicy,
    spare: TableBacking,
    compute: &mut Backend,
) -> railgun::sealed_table::generation::GenerationRetired {
    let g1 = policy
        .seal(spare, GenerationId { epoch: 1, sub: 0 })
        .map_err(|(_, error)| error)
        .expect("seal generation 1");
    let pending = acquire_window_owned(g1)
        .map_err(|(_, error)| error)
        .expect("lease generation 1")
        .enqueued(compute)
        .map_err(|(_, failure)| failure.message)
        .expect("window fence");
    let g1 = finish_window(pending, compute)
        .map_err(|_| "window fence")
        .expect("generation 1 fenced")
        .into_generation()
        .expect("owned generation");
    policy
        .retire_generation(g1, None)
        .map_err(|(_, error)| error)
        .expect("retire generation 1")
        .1
}

#[test]
fn race_old_lease_blocks_reclaim_of_a_multiply_named_slot() {
    let Pipelined {
        dev,
        mut policy,
        g0,
        spare,
        mut compute,
        mut copy,
        mut stage,
    } = Pipelined::new();
    // A stage reader leases generation 0 (A -> slot 0, B -> slot 1); its reads are delayed.
    dev.borrow_mut().push(
        stage_stream(0),
        Op::Pinned {
            label: "stage reader".into(),
            reads: vec![(A, 0, 0), (B, 0, 1)],
        },
    );
    let held = acquire_window(&g0)
        .expect("stage lease")
        .enqueued(&mut stage)
        .map_err(|(_, failure)| failure.message)
        .expect("stage fence");
    // Generation 1 also names both slots, and retires while generation 0 is still live.
    let retired1 = retire_pipelined_generation(&mut policy, spare, &mut compute);
    let out = policy.retire_windows((0, 0), &[hit(A), hit(C)], vec![], 1, &retired1);
    // B is evicted but its slot is still named by generation 0: the reclaim is refused, no fill.
    assert_eq!(out.unpublish, vec![B]);
    assert!(out.fills.is_empty());
    assert_eq!(policy.refusals().0, 1, "{:?}", policy.refusals());
    assert_eq!(policy.slot_phase(1), SlotPhase::Quarantined);
    assert_eq!(bank(&policy).refs(1), 1);
    assert_eq!(dev.borrow().queued(copy_stream(0)), 0);
    assert_eq!(dev.borrow().slot_bytes[0], vec![Some(A), Some(B)]);
    // The stage reader runs (intact bytes) and generation 0 retires: the slot reclaims at the
    // next retirement and is refilled.
    finish_window(held, &mut stage)
        .map_err(|_| "stage fence")
        .expect("stage fenced");
    let (_t0, retired0) = policy
        .retire_generation(g0, None)
        .map_err(|(_, error)| error)
        .expect("retire generation 0");
    let out = policy.retire_windows((0, 0), &[hit(A), hit(C)], vec![], 1, &retired0);
    assert_eq!(
        out.fills.iter().map(|op| op.key()).collect::<Vec<_>>(),
        vec![(1, C, 2)]
    );
    assert!(out.unpublish.is_empty());
    assert_eq!(policy.refusals().0, 1);
    // The new table (A only) reaches the device before the fill runs.
    let tickets = out.fills.into_iter().map(|op| op.ticket).collect();
    dev.borrow_mut()
        .images
        .insert(Table::Decode(0), Image::from([(A, (0, 0))]));
    enqueue_fills(&mut policy, &dev, 0, tickets, &mut copy)
        .map_err(|(_, failure)| failure.message)
        .expect("fill issue");
    dev.borrow_mut().settle();
    assert_eq!(dev.borrow().slot_bytes[0], vec![Some(A), Some(C)]);
    assert_clean(&dev);
}

#[test]
fn race_old_lease_positive_control() {
    let Pipelined {
        dev,
        mut policy,
        g0,
        spare,
        mut compute,
        mut copy,
        mut stage,
    } = Pipelined::new();
    // The stage fence is recorded BEFORE the stage reader is enqueued: it finishes at once.
    let early = acquire_window(&g0)
        .expect("stage lease")
        .enqueued(&mut stage)
        .map_err(|(_, failure)| failure.message)
        .expect("stage fence");
    dev.borrow_mut().push(
        stage_stream(0),
        Op::Pinned {
            label: "stage reader".into(),
            reads: vec![(A, 0, 0), (B, 0, 1)],
        },
    );
    finish_window(early, &mut stage)
        .map_err(|_| "stage fence")
        .expect("premature stage fence");
    // Generation 0 retires prematurely, so the reclaim of B's slot goes through.
    policy
        .retire_generation(g0, None)
        .map_err(|(_, error)| error)
        .expect("retire generation 0");
    let retired1 = retire_pipelined_generation(&mut policy, spare, &mut compute);
    let out = policy.retire_windows((0, 0), &[hit(A), hit(C)], vec![], 1, &retired1);
    assert_eq!(
        out.fills.iter().map(|op| op.key()).collect::<Vec<_>>(),
        vec![(1, C, 2)]
    );
    let tickets = out.fills.into_iter().map(|op| op.ticket).collect();
    enqueue_fills(&mut policy, &dev, 0, tickets, &mut copy)
        .map_err(|(_, failure)| failure.message)
        .expect("fill issue");
    dev.borrow_mut().run_all(copy_stream(0));
    dev.borrow_mut().run_all(stage_stream(0));
    let d = dev.borrow();
    assert!(d.any(|v| matches!(v, Violation::WriteNamed { slot: 1, .. })));
    assert!(d.any(|v| matches!(
        v,
        Violation::Read { unit, slot: 1, found: Some(found), .. } if *unit == B && *found == C
    )));
}

// 10. event query error / unknown fence ------------------------------------------------------------

#[test]
fn race_unknown_fence_freezes_with_old_bytes_intact() {
    let uses = [hit(A), hit(C)];

    // (a) The window fence's event query fails with unknown quiescence.
    let mut m = Mesh::new(Cfg::new(2, 2));
    m.seed(0, &[A, B]);
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    m.dev()
        .fail_query(compute_stream(0), Some(Quiescence::Unknown));
    assert!(matches!(m.ranks[0].retire(&uses), Step::Frozen(_)));
    assert!(
        matches!(m.ranks[0].retire(&uses), Step::Frozen(_)),
        "freezing is sticky"
    );
    {
        let policy = &m.ranks[0].policy;
        assert_eq!(policy.stats().retirements, 0);
        assert_eq!(policy.count(SlotPhase::Live), 2);
        // The generation never retires: its slots stay referenced.
        assert_eq!(bank(policy).live_generations(), 1);
        assert_eq!((bank(policy).refs(0), bank(policy).refs(1)), (1, 1));
    }
    assert_eq!(m.dev().queued(copy_stream(0)), 0);
    assert_eq!(m.bytes(0), vec![Some(A), Some(B)]);
    // A later borrow cannot lend the pool.
    assert!(matches!(m.ranks[0].lend(), Err(ResidencyError::Refused(_))));
    m.assert_clean();

    // (b) The fill batch's event query fails: its slots are quarantined forever.
    let mut m = Mesh::new(Cfg::new(2, 2));
    m.seed(0, &[A, B]);
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    let out = m.ranks[0].retire(&uses).retired();
    assert_eq!(out.fills, vec![(1, C, 2)]);
    m.dev()
        .fail_query(copy_stream(0), Some(Quiescence::Unknown));
    m.ranks[0].forward();
    m.dev().run_all(compute_stream(0));
    assert!(matches!(m.ranks[0].retire(&[hit(A)]), Step::Frozen(_)));
    {
        let policy = &m.ranks[0].policy;
        assert!(bank(policy).is_quarantined(1));
        assert_eq!(policy.count(SlotPhase::Filling), 1);
        assert_eq!(policy.live_slot(C), None);
    }
    assert_eq!(m.bytes(0), vec![Some(A), Some(B)]);
    assert!(matches!(m.ranks[0].lend(), Err(ResidencyError::Refused(_))));
    // The copy may still run: it writes a slot no table names.
    m.dev().run_all(copy_stream(0));
    m.assert_clean();

    // (c) A stuck stream: the window fence's wait fails with unknown quiescence.
    let mut m = Mesh::new(Cfg::new(2, 2));
    m.seed(0, &[A, B]);
    m.ranks[0].forward();
    m.dev().set_wait(compute_stream(0), Wait::Stuck);
    assert!(matches!(m.ranks[0].retire(&uses), Step::Frozen(_)));
    assert_eq!(bank(&m.ranks[0].policy).live_generations(), 1);
    assert_eq!(m.bytes(0), vec![Some(A), Some(B)]);
    m.assert_clean();
}

#[test]
fn race_unknown_fence_positive_control() {
    let mut m = Mesh::new(Cfg::new(2, 2));
    m.seed(0, &[A, B]);
    // An unsound backend reports the window fence complete while the reader has not run.
    m.dev().lie(compute_stream(0), true);
    m.ranks[0].forward();
    let out = m.ranks[0].retire(&[hit(A), hit(C)]).retired();
    assert_eq!(out.fills, vec![(1, C, 2)]);
    m.dev().run_all(copy_stream(0));
    m.dev().run_all(compute_stream(0));
    assert!(m.dev.borrow().any(|v| matches!(
        v,
        Violation::Read { unit, slot: 1, found: Some(found), .. } if *unit == B && *found == C
    )));
    assert!(m
        .dev
        .borrow()
        .any(|v| matches!(v, Violation::WriteNamed { slot: 1, .. })));
}

// 11. teardown with a pending window -------------------------------------------------------------

#[test]
fn race_teardown_with_a_pending_window() {
    let mut m = Mesh::new(Cfg::new(2, 2));
    m.seed(0, &[A, B]);
    m.ranks[0].forward();
    m.dev().set_wait(compute_stream(0), Wait::Pending);
    // The last window's fence cannot finish: the teardown frees nothing.
    assert!(!m.ranks[0].release());
    assert!(!m.dev.borrow().freed[0]);
    // With the device idle (every reader ran) the bank still refuses to close while the
    // generation is live: the window was never fenced and retired.
    let refused = m.ranks[0]
        .close_idle()
        .expect_err("a live generation refuses close");
    assert!(refused.contains("generation"), "{refused}");
    assert!(m.ranks[0].policy.bank.is_some());
    assert_eq!(bank(&m.ranks[0].policy).live_generations(), 1);
    assert!(!m.dev.borrow().freed[0]);
    // The device idle ran the window's marker: the fence completes, the generation retires, the
    // bank closes and only then is the pool freed.
    assert!(m.ranks[0].release());
    assert!(m.dev.borrow().freed[0]);
    assert!(m.ranks[0].policy.bank.is_none());
    m.assert_clean();
}

#[test]
fn race_teardown_with_a_pending_window_positive_control() {
    let mut m = Mesh::new(Cfg::new(2, 2));
    m.seed(0, &[A, B]);
    m.ranks[0].variant = Variant::ReleaseEarly;
    m.ranks[0].forward();
    m.dev().set_wait(compute_stream(0), Wait::Pending);
    // The pool is freed although the window is pending and the bank refused to close.
    assert!(m.ranks[0].release());
    m.dev().run_all(compute_stream(0));
    assert!(m
        .dev
        .borrow()
        .any(|v| matches!(v, Violation::FreedPool { owner: 0, .. })));
}
