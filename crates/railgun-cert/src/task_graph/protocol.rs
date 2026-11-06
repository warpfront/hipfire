// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The dispatcher protocol of a persistent pool kernel, proven over its lifted
//! CFG (obligation 4 of [`super`]).
//!
//! The check is a path-partitioned symbolic dataflow. Every 32-bit register
//! (SGPR, and VGPR as its first-active-lane value) holds a small set of
//! symbolic values [`E`]: a dword of the params block the kernarg pointer
//! addresses, the ticket's old value, a field of the claimed task header, the
//! event cell a poll read, a value that crossed the workgroup's LDS broadcast
//! (carrying the facts that held where wave 0 stored it), constants. Facts
//! ([`CL`]..) record what the path has established; states with different
//! facts are never merged, so a property is checked per path class. A branch
//! edge refines the values and facts of its compare (`claim >= n_tasks`,
//! `pred == NONE`, `events[pred] >= expected`, `tmpl == t`, `tmpl == -1`,
//! `wave != 0`). Region bodies are summarised at their entry: the entry state
//! is checked, then every register outside the dispatcher's SGPR range is
//! forgotten and the walk resumes at the region's exit. A dispatcher SGPR the
//! region writes is *clobbered* from there until the dispatcher writes it again:
//! reading it is refused (`task_graph-region-sgpr-park`), except after a restore
//! `v_readlane_b32 sK, vN, lane` from the VGPR lane (constant lane) that held a copy
//! of `sK` (`v_writelane_b32 vN, sK, lane`) on every path into the entry, with `vN`
//! unwritten by the region (every region VGPR def, VOPD halves and wide
//! destinations included); the restore gives `sK` the symbolic value it had at the
//! entry. Parked in a VGPR the region writes, or restored from another lane or
//! another VGPR, is refused.
//!
//! What it proves, on every path, for the six required `TASK_PROTO_*` properties (and,
//! when it holds, the optional seventh):
//!
//! * claim: the only ticket access is one `global_atomic_add_u32` of constant 1
//!   with return, device scope, issued by a single lane of wave 0, on the pointer
//!   loaded from the params block; its result is the claim; the header is read
//!   only from `tasks[claim << k]`.
//! * bounded: a region is entered only on the edge `claim < n_tasks`.
//! * acquire: a region is entered only if `pred == NONE` or an acquire load of
//!   `events[pred]` observed `>= expected` followed by `s_wait_loadcnt 0` and
//!   `global_inv` at device scope; the poll's other exits (the watchdog) leave
//!   `tmpl == -1`, and `tmpl != -1` is required before any region.
//! * release: the one release increment of `events[done]` is reached, after the
//!   body, only with `done != NONE`; a template outside the note's table reaches
//!   it without a body (reported; the host must not submit one).
//! * complete: before it every wave has waited its load and store counters and
//!   passed a workgroup barrier after its body.
//! * once: it executes once per claimed task (not inside an inner loop, by one
//!   lane of wave 0), and every complete trip of the claim loop passes it, skips
//!   it for `done == NONE`, or is a wave that never signals.
//! * region_wg (optional: added to `proved` only when every check below holds): every
//!   region is entered with its `wg_x` register `= header[vx] + step` and its `wg_yz`
//!   register `= header[vy] | header[vz] << 16`, all fields of the claimed task, where
//!   `step` is the counter of a step loop around the region. The induction fact is the
//!   symbolic [`E::Trip`]: the counter is `0` on every entry edge of the loop (with its
//!   bound register holding a header field, optionally `max(.., 1)`) and the loop's
//!   latch increments it by one and takes the back edge only while `step + 1 < bound`
//!   ([`E::TripNext`] refined to [`E::TripCont`] on that edge), so `step` is the trip
//!   number and `step < max(bound, 1)` at every head; a write of the counter anywhere
//!   else loses the fact. Every trip must enter the region (`RAN`) before it takes the
//!   back edge or leaves the loop, and the only way out is the failed bound compare, so
//!   every step of `[0, max(vcount, 1))` runs the region once.
//!   Grid: `gx` is derived from the bytes (the one other header dword the dispatcher keeps
//!   live in a dispatcher SGPR, `s72`; a native region's grid-stride loop reads it, and
//!   its reads of the dispatcher range are checked to be only that register and the
//!   workgroup registers). `gy`/`gz` are the next two header dwords (loaded, distinct from
//!   every consumed field, never read by the dispatcher). A lifted body does not read the
//!   header grid at all: its grid comes from the hidden arguments of its own record, which
//!   the dispatcher never writes, so the relation `header grid == record hidden counts`
//!   is an obligation of whoever builds the arena (the host), stated in the evidence line
//!   and not provable here. `wg_yz` is `vy | vz << 16` without masking: it equals the
//!   hardware `ttmp7` only for `vy, vz < 65536`.
//!
//! Span mode (railgunmega/DYNAMIC.md; a dispatcher that has it proves the two properties below or
//! the pool is refused, because its table accesses would be unattributed): the header's flags word
//! is `0` (a static task, `region_wg_from_header` as before) or `1` (span mode), anything else
//! poisons (the task's template becomes NONE, so no region runs). A span task reads the table at a
//! params pointer (null poisons) and descriptor `list_id` (`list_id >= n_spans` poisons) =
//! `{live_list, count_cell, bitmap, K}`; every one of these is a symbolic value (`E::SpanId`,
//! `E::SpanFld`, `E::SpanCountLe`, `E::SpanIdx`, `E::Entry`) that exists only on the edge of its
//! compare, so the proof is structural:
//!
//! * `span_entry_validated`: the count load is a single-lane plain load of the descriptor's count
//!   cell (ordered after the pred acquire whenever `pred != NONE`: the evidence says so), `count >
//!   K` branches to poison, `k >= count` branches to the NOP template (a constant outside the
//!   note's table: no region, the done release is reached without a body), the entry load's
//!   address is `live_list + 16 * k` with `k < count <= K`, the three entry coordinates are
//!   compared against the header grid (`vx < gx`, `vy < gy`, `vz < gz`, failing edges to poison),
//!   the claim bitmap word `bit >> 5` of `bit = vx + gx * (vy + gy * vz)` is OR-ed with `1 << (bit &
//!   31)` by a returning device-scope atomic and a set bit branches to poison. Each compare edge
//!   sets a fact bit (`SP_ALL`); a region entry on a span-mode path (the facts the slot's store
//!   event, or the single wave's path, carries) needs all of them.
//! * `region_wg_from_live_entry`: on those paths `wg_x = entry.vx + step` with the step counter
//!   bounded by the constant 1 (the dispatcher forces `vcount := 1`) and `wg_yz = entry.vy |
//!   entry.vz << 16`, the entry values carried through the broadcast.
//!
//! Poison is the existing `tmpl == NONE` path; its edges never reach a region for the same reason
//! the watchdog's do (no `tmpl == t` edge is feasible). The accesses through table-loaded pointers
//! are returned as [`SpanSite`]s for the shell attribution (`task_graph`).
//!
//! Multi-wave pools: waves other than 0 run no claim code and take the
//! descriptor from the LDS slot wave 0 filled before a barrier; the check proves
//! the two roles reach the same first barrier, that control is wave-uniform from
//! there on (every conditional branch compares uniform values) except the
//! documented wave-0-only tail, and that no barrier, region or LDS access sits in
//! a divergent segment. The LDS discipline (store, barrier, load, barrier) is
//! checked per wave path.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use peacemaker_ir::cfg::{BlockId, Cond, Terminator};
use peacemaker_ir::effects::Control;
use peacemaker_ir::inst::{Abi, Arch, Inst, Kernel};
use peacemaker_ir::kernarg::KernargFacts;
use peacemaker_ir::passes::cfg::Cfg;
use peacemaker_ir::wait::Counter;
use peacemaker_lift::regions_note::KernelRegions;
use peacemaker_lift::text::print::canonical;
use railgun_corpus::receipt::{
    TaskGridOffsets, TaskProtocolRecord, TaskSpanOffsets, TASK_PROTO_ACQUIRE, TASK_PROTO_BOUNDED,
    TASK_PROTO_CLAIM, TASK_PROTO_COMPLETE, TASK_PROTO_ONCE, TASK_PROTO_REGION_WG,
    TASK_PROTO_REGION_WG_LIVE, TASK_PROTO_RELEASE, TASK_PROTO_SPAN_ENTRY,
};

use super::{refuse, Placed, Refusal, RegionFacts};

/// Facts of a path (the partition key).
const CL: u32 = 1 << 0; // claimed: the ticket fetch-add's result is live
const BD: u32 = 1 << 1; // claim < n_tasks
const HD: u32 = 1 << 2; // header loaded from tasks[claim << k]
const PO: u32 = 1 << 3; // pred == NONE, or pred observed >= expected and acquired
const ACQ: u32 = 1 << 4; // device-scope global_inv after its waits (cleared by an LDS load)
const BODY: u32 = 1 << 5; // a region ran
const BDSYNC: u32 = 1 << 6; // after the body: load/store waits and (multi-wave) a barrier
const SIG: u32 = 1 << 7; // the release increment executed
const DN: u32 = 1 << 8; // done == NONE
const W0: u32 = 1 << 9; // wave 0
const N0: u32 = 1 << 10; // a wave other than 0
const POLLED: u32 = 1 << 11; // events[pred] >= expected observed
const DV: u32 = 1 << 12; // done != NONE
const MERGED: u32 = 1 << 13; // a workgroup barrier executed this trip
const DIV2: u32 = 1 << 14; // in the wave-0-only tail after the release barrier
const STORED: u32 = 1 << 15; // this wave stored the LDS slot, no barrier since
const LOADED: u32 = 1 << 16; // this wave loaded the LDS slot, no barrier since
const RAN: u32 = 1 << 17; // the region of the current step-loop trip was entered
/// Span mode facts (DYNAMIC.md): what the path of a span-mode header has established.
const SP0: u32 = 1 << 18; // header flags == 0 (a static task)
const SP1: u32 = 1 << 19; // header flags == 1 (span mode)
const SPN: u32 = 1 << 20; // the span table pointer is nonzero
const SPX: u32 = 1 << 21; // entry.vx < header gx
const SPY: u32 = 1 << 22; // entry.vy < header gy
const SPZ: u32 = 1 << 23; // entry.vz < header gz
const SPC: u32 = 1 << 24; // the claim bitmap's bit of the entry was clear (returning atomic OR)
const SPK: u32 = 1 << 25; // on the `k >= count` (NOP) edge
const SPP: u32 = 1 << 26; // on a span path: pred != NONE (the NONE edge poisons)
const SPQ: u32 = 1 << 27; // pred == NONE was observed on this path
/// Every fact a span-mode path needs before a region (the live-list index, the count
/// bound and the table index are proved structurally, by the value of the address).
const SP_ALL: u32 = SP1 | SPN | SPX | SPY | SPZ | SPC;

const NONE: u32 = u32::MAX;
const MAX_ALTS: usize = 6;
const MAX_DEPTH: usize = 6;

/// Work bounds of the protocol check (see "Complexity and work bounds" in [`super`]). One
/// exceeded refuses with `task_graph-protocol-budget` and the measured counts; the walk is
/// then incomplete and nothing it saw is admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// State visits of one walk pass, per block of the dispatcher's CFG.
    pub visits_per_block: u32,
    /// Path partitions (`(block, facts)` states) held by one walk pass, per block.
    pub states_per_block: u32,
    /// Walk passes of the LDS-slot fixpoint (the claim slot's contents feed the next pass).
    pub lds_passes: u32,
}

impl Default for Budget {
    fn default() -> Self {
        Budget {
            visits_per_block: 256,
            states_per_block: 64,
            lds_passes: 5,
        }
    }
}

/// A symbolic 32-bit value.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum E {
    Const(u32),
    /// Dword of the kernarg segment pointer received at entry (0 = lo, 1 = hi).
    Karg(u8),
    /// The dword at byte offset `off` of the params block.
    Param(u32),
    /// `v0` at entry: the packed workitem id.
    TidEntry,
    /// The wave's first workitem id: 0 exactly for wave 0.
    WaveId,
    /// Per-lane `mbcnt` and the full workitem id.
    Lane,
    Tid,
    /// The ticket's old value.
    Claim,
    /// Header dword at this byte offset of `tasks[claim << stride]`.
    Hdr(u32),
    Shl(Box<E>, u8),
    /// The event cell read at byte offset `idx` from the events pointer `Param(ptr)`.
    EventAt {
        ptr: u32,
        idx: Box<E>,
    },
    /// Lo/hi dword of `Param(base) + zext(add)`.
    PtrLo {
        base: u32,
        add: Box<E>,
    },
    PtrHi {
        base: u32,
        add: Box<E>,
    },
    /// The dword at byte `off` of the claim slot, as the wave 0 store event that
    /// filled it (`State::choice`) put it there.
    Cell(u32),
    /// An unmodelled value that is the same in every wave.
    Uni,
    /// `a | b` and `a + b` (32-bit, wrapping); the operands are in `Ord` order.
    Or(Box<E>, Box<E>),
    Add(Box<E>, Box<E>),
    /// `max(x, 1)`, unsigned.
    MaxOne(Box<E>),
    /// The trip number (0, 1, ..) of the step loop headed by block `head` at its
    /// head, with `step < max(bound, 1)` established by the loop's own entry and
    /// back-edge conditions; [`E::TripNext`] is that number plus one before the
    /// latch's bound compare, [`E::TripCont`] after the compare's taken edge
    /// (`step + 1 < bound`). Created only at the step-loop edge hook.
    Trip {
        head: usize,
        bound: Box<E>,
    },
    TripNext {
        head: usize,
        bound: Box<E>,
    },
    TripCont {
        head: usize,
        bound: Box<E>,
    },
    /// Span mode: `n_spans`, the u32 at byte 0 of the span table.
    SpanN,
    /// Header dword at this offset (the list id) after `list_id < n_spans` held.
    SpanId(u32),
    /// Dword of span descriptor `list_id`: `off` is its byte offset in the descriptor
    /// (0 live list, 8 count cell, 16 bitmap: pointer pairs, `hi` the upper dword; 24: K).
    SpanFld {
        off: u8,
        hi: bool,
    },
    /// The live count the count load read.
    SpanCount,
    /// The live count after `count <= K` held.
    SpanCountLe,
    /// Header dword at this offset (the span-relative index k) after `k < count <= K` held.
    SpanIdx(u32),
    /// Dword `.0` (0 vx, 1 vy, 2 vz, 3 pad) of the live entry the entry load read.
    Entry(u8),
    /// `a * b` (32-bit, low half) of the claim-bitmap index computation.
    Mul(Box<E>, Box<E>),
    /// A stage of the claim-bitmap index of an entry; `gx`/`gy` are the header offsets of
    /// the grid extents it multiplies by.
    Tile {
        st: TileSt,
        gx: u32,
        gy: u32,
    },
    /// Lane 0's mask of `a >= b` (`gt`: `a > b`), unsigned, of a wave32 vector compare whose
    /// first operand sat in VGPR `reg`; and of `x != 0`.
    VCmp {
        gt: bool,
        a: Box<E>,
        b: Box<E>,
        reg: u16,
    },
    NzMask(Box<E>),
    /// Byte offset `32 + 32 * list_id` of the span descriptor of the list id at header offset
    /// `.0` (after `list_id < n_spans`).
    DescOff(u32),
    /// Lo/hi dword of `live_list + 16 * k` (k at header offset `.k`, after `k < count <= K`).
    LiveAddr {
        k: u32,
        hi: bool,
    },
}

/// A stage of the claim-bitmap index computation `bit = vx + gx * (vy + gy * vz)` of a span
/// entry (`gx`/`gy` the header offsets of the grid extents it multiplies by).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum TileSt {
    /// `vy + gy * vz`
    Yz,
    /// `bit`
    Idx,
    /// `bit >> 5`
    Word,
    /// `(bit >> 5) << 2`
    WordOff,
    /// `bit & 31`
    Low,
    /// `1 << (bit & 31)`
    Mask,
    /// The bitmap word the returning atomic OR read.
    Old,
    /// `old & mask`
    Hit,
}

impl E {
    fn uniform(&self) -> bool {
        match self {
            E::Const(_) | E::Karg(_) | E::Param(_) | E::Cell(_) | E::Uni => true,
            E::Shl(x, _) | E::MaxOne(x) => x.uniform(),
            E::PtrLo { add, .. } | E::PtrHi { add, .. } => add.uniform(),
            E::Or(a, b) | E::Add(a, b) => a.uniform() && b.uniform(),
            E::Trip { bound, .. } | E::TripNext { bound, .. } | E::TripCont { bound, .. } => {
                bound.uniform()
            }
            _ => false,
        }
    }
    fn depth(&self) -> usize {
        match self {
            E::Shl(x, _) | E::MaxOne(x) => 1 + x.depth(),
            E::EventAt { idx, .. } => 1 + idx.depth(),
            E::PtrLo { add, .. } | E::PtrHi { add, .. } => 1 + add.depth(),
            E::Or(a, b) | E::Add(a, b) => 1 + a.depth().max(b.depth()),
            E::Mul(a, b) | E::VCmp { a, b, .. } => 1 + a.depth().max(b.depth()),
            E::NzMask(x) => 1 + x.depth(),
            E::Trip { bound, .. } | E::TripNext { bound, .. } | E::TripCont { bound, .. } => {
                1 + bound.depth()
            }
            _ => 0,
        }
    }
    /// Whether the value is only meaningful for the claim of the current trip of the
    /// claim loop (a header field, the LDS slot, the claim itself, anything derived):
    /// it must not survive into the next trip's code.
    fn per_claim(&self) -> bool {
        match self {
            E::Claim
            | E::Hdr(_)
            | E::Cell(_)
            | E::EventAt { .. }
            | E::Trip { .. }
            | E::TripNext { .. }
            | E::TripCont { .. }
            | E::SpanN
            | E::SpanId(_)
            | E::SpanFld { .. }
            | E::SpanCount
            | E::SpanCountLe
            | E::SpanIdx(_)
            | E::Entry(_)
            | E::Tile { .. } => true,
            E::Shl(x, _) | E::MaxOne(x) | E::NzMask(x) => x.per_claim(),
            E::PtrLo { add, .. } | E::PtrHi { add, .. } => add.per_claim(),
            E::Or(a, b) | E::Add(a, b) | E::Mul(a, b) | E::VCmp { a, b, .. } => {
                a.per_claim() || b.per_claim()
            }
            E::DescOff(_) | E::LiveAddr { .. } => true,
            _ => false,
        }
    }
    fn is_step(&self) -> bool {
        matches!(
            self,
            E::Trip { .. } | E::TripNext { .. } | E::TripCont { .. }
        )
    }
    fn add(a: E, b: E) -> E {
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        E::Add(Box::new(a), Box::new(b))
    }
    fn or(a: E, b: E) -> E {
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        E::Or(Box::new(a), Box::new(b))
    }
}

type Alts = BTreeSet<E>;

fn one(e: E) -> Alts {
    std::iter::once(e).collect()
}

fn only(a: &Alts) -> Option<&E> {
    (a.len() == 1).then(|| a.iter().next().unwrap())
}

fn single(a: &Option<Alts>) -> Option<&E> {
    only(a.as_ref()?)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Exec {
    One,
    Full,
    Top,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rel {
    Eq,
    Lg,
    Ge,
    Lt,
    Gt,
    Le,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Cmp {
    rel: Rel,
    signed: bool,
    a: Option<Alts>,
    b: Option<Alts>,
    a_reg: Option<u16>,
}

impl Cmp {
    fn uniform(&self) -> bool {
        let all = |x: &Option<Alts>| x.as_ref().is_some_and(|s| s.iter().all(E::uniform));
        all(&self.a) && all(&self.b)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct State {
    s: BTreeMap<u16, Alts>,
    v: BTreeMap<u16, Alts>,
    exec: Exec,
    scc: Option<Cmp>,
    /// `base`, and the addends, of the last `s_add_co_u32` on `Param(base)`.
    carry: Option<(u32, Alts)>,
    facts: u32,
    waited: u8,
    tmpl_eq: Option<u32>,
    tmpl_ne: BTreeSet<u32>,
    /// Merge target of the wave-0-only tail.
    div2: Option<usize>,
    /// The store events (indices into the LDS summary) the loaded slot may hold.
    choice: Option<BTreeSet<usize>>,
    /// Parked dispatcher SGPRs: `(vgpr, lane) -> sgpr` while lane `lane` of `vgpr` holds
    /// the current value of that SGPR (`v_writelane_b32 vN, sK, lane` with a constant lane;
    /// killed by any write of either register).
    lanes: BTreeMap<(u16, u8), u16>,
    /// Dispatcher SGPRs a region has overwritten and the dispatcher has not yet restored.
    clob: BTreeMap<u16, Clob>,
}

/// A dispatcher SGPR clobbered by a region: the lanes it was parked in before the entry
/// (empty when the paths disagree) and the symbolic value it held then.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Clob {
    lanes: BTreeSet<(u16, u8)>,
    saved: Option<Alts>,
    /// Template of the region that overwrote it.
    by: u32,
}

/// A constant-lane `v_writelane_b32 vV, sS, L` / `v_readlane_b32 sS, vV, L`.
#[derive(Clone, Copy)]
enum ParkOp {
    Write(u16, u8, u16),
    Read(u16, u16, u8),
}

const WL: u8 = 1;
const WS: u8 = 2;
const WD: u8 = 4;
const WK: u8 = 8;

impl State {
    fn join(&mut self, o: &State) {
        self.s = join_map(&self.s, &o.s);
        self.v = join_map(&self.v, &o.v);
        if self.exec != o.exec {
            self.exec = Exec::Top;
        }
        if self.scc != o.scc {
            self.scc = None;
        }
        if self.carry != o.carry {
            self.carry = None;
        }
        self.waited &= o.waited;
        if self.tmpl_eq != o.tmpl_eq {
            self.tmpl_eq = None;
        }
        self.tmpl_ne = self.tmpl_ne.intersection(&o.tmpl_ne).copied().collect();
        if self.div2 != o.div2 {
            self.div2 = None;
        }
        self.choice = match (&self.choice, &o.choice) {
            (Some(a), Some(b)) => Some(a.union(b).copied().collect()),
            _ => None,
        };
        self.lanes.retain(|k, r| o.lanes.get(k) == Some(r));
        for (r, c) in &o.clob {
            match self.clob.get_mut(r) {
                Some(mine) => {
                    mine.lanes = mine.lanes.intersection(&c.lanes).copied().collect();
                    mine.saved = match (&mine.saved, &c.saved) {
                        (Some(a), Some(b)) => {
                            let u: Alts = a.union(b).cloned().collect();
                            (u.len() <= MAX_ALTS).then_some(u)
                        }
                        _ => None,
                    };
                }
                // clobbered on one path only: not restorable
                None => {
                    self.clob.insert(
                        *r,
                        Clob {
                            lanes: BTreeSet::new(),
                            saved: None,
                            by: c.by,
                        },
                    );
                }
            }
        }
        for (r, c) in &mut self.clob {
            if !o.clob.contains_key(r) {
                c.lanes.clear();
                c.saved = None;
            }
        }
    }
}

fn join_map(a: &BTreeMap<u16, Alts>, b: &BTreeMap<u16, Alts>) -> BTreeMap<u16, Alts> {
    let mut out = BTreeMap::new();
    for (k, x) in a {
        if let Some(y) = b.get(k) {
            let u: Alts = x.union(y).cloned().collect();
            if u.len() <= MAX_ALTS {
                out.insert(*k, u);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// instruction text
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub(super) struct Ins {
    pub(super) mn: String,
    pub(super) ops: Vec<String>,
    pub(super) mods: Vec<String>,
}

impl Ins {
    pub(super) fn parse(text: &str) -> Ins {
        let text = text.split(';').next().unwrap_or("").trim();
        let (mn, rest) = text
            .split_once(char::is_whitespace)
            .map_or((text, ""), |(m, r)| (m, r.trim()));
        let mut ops = Vec::new();
        let mut mods = Vec::new();
        let (mut depth, mut cur) = (0i32, String::new());
        let mut pieces = Vec::new();
        for c in rest.chars() {
            match c {
                '[' | '(' => {
                    depth += 1;
                    cur.push(c)
                }
                ']' | ')' => {
                    depth -= 1;
                    cur.push(c)
                }
                ',' if depth == 0 => pieces.push(std::mem::take(&mut cur)),
                _ => cur.push(c),
            }
        }
        if !cur.trim().is_empty() {
            pieces.push(cur);
        }
        for p in pieces {
            for tok in p.split_whitespace() {
                // `offset:16`, `scope:SCOPE_DEV`, `th:TH_ATOMIC_RETURN` (a register range
                // `s[4:5]` has a colon too, inside brackets)
                if tok.contains(':') && !tok.contains('[') {
                    mods.push(tok.to_owned());
                } else {
                    ops.push(tok.to_owned());
                }
            }
        }
        Ins {
            mn: mn.to_owned(),
            ops,
            mods,
        }
    }
    fn modifier(&self, key: &str) -> Option<&str> {
        self.mods
            .iter()
            .find_map(|m| m.strip_prefix(key)?.strip_prefix(':'))
    }
    pub(super) fn offset(&self) -> i64 {
        self.modifier("offset").and_then(parse_num).unwrap_or(0)
    }
    fn returns(&self) -> bool {
        self.mods.iter().any(|m| m.contains("TH_ATOMIC_RETURN"))
    }
    /// Scope rank: 0 none/CU, 1 SE, 2 device, 3 system.
    fn scope(&self) -> u8 {
        match self.modifier("scope") {
            Some("SCOPE_SYS") => 3,
            Some("SCOPE_DEV") => 2,
            Some("SCOPE_SE") => 1,
            _ => 0,
        }
    }
}

pub(super) fn parse_num(s: &str) -> Option<i64> {
    let s = s.trim();
    let s = s
        .strip_prefix("lit(")
        .and_then(|r| r.strip_suffix(')'))
        .unwrap_or(s);
    let (neg, body) = match s.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, s),
    };
    let v = if let Some(h) = body.strip_prefix("0x") {
        i64::from_str_radix(h, 16).ok()?
    } else {
        body.parse::<i64>().ok()?
    };
    Some(if neg { -v } else { v })
}

/// A register operand: `s12`, `s[4:5]`, `v1`, `v[4:7]`, `exec_lo`, ...
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Rk {
    S,
    V,
    Exec,
    Other,
}

pub(super) fn parse_reg(s: &str) -> Option<(Rk, u16, u16)> {
    let s = s.trim();
    match s {
        "exec_lo" => return Some((Rk::Exec, 0, 1)),
        "exec_hi" => return Some((Rk::Exec, 1, 1)),
        "exec" => return Some((Rk::Exec, 0, 2)),
        "vcc_lo" | "vcc_hi" | "vcc" | "m0" | "null" => return Some((Rk::Other, 0, 1)),
        _ => {}
    }
    if s.starts_with("ttmp") {
        return Some((Rk::Other, 0, 1));
    }
    let (kind, rest) = match s.as_bytes().first()? {
        b's' => (Rk::S, &s[1..]),
        b'v' => (Rk::V, &s[1..]),
        _ => return None,
    };
    if let Some(r) = rest.strip_prefix('[') {
        let r = r.strip_suffix(']')?;
        let (a, b) = r.split_once(':')?;
        let (a, b): (u16, u16) = (a.parse().ok()?, b.parse().ok()?);
        return Some((kind, a, b - a + 1));
    }
    Some((kind, rest.parse().ok()?, 1))
}

/// The store events of one pass, ordered by their facts.
fn events_of(m: &BTreeMap<u32, BTreeMap<u32, Alts>>) -> Vec<Event> {
    m.iter()
        .map(|(facts, cells)| Event {
            facts: *facts,
            cells: cells.clone(),
        })
        .collect()
}

/// How `inst` changes the knowledge of which counters are drained: a zero-wait sets
/// the counter's bit, an issue clears the bit of the counter it counts on.
fn apply_wait(inst: &Inst, mn: &str, waited: u8) -> u8 {
    use peacemaker_ir::effects::MemClass::*;
    if let Some(w) = &inst.mods.wait {
        let mut out = waited;
        for (c, bit) in [
            (Counter::Load, WL),
            (Counter::Store, WS),
            (Counter::Ds, WD),
            (Counter::Km, WK),
        ] {
            if w.per_counter[c as usize] == Some(0) {
                out |= bit;
            }
        }
        return out;
    }
    if mn == "s_sendmsg_rtn_b64" {
        return waited & !WK;
    }
    if mn == "global_inv" {
        // cache maintenance after the waits that complete the observation it acquires
        return waited;
    }
    match inst.effects.mem.map(|m| m.class) {
        None => waited,
        Some(SmemLoad | SmemStore) => waited & !WK,
        Some(DsLoad | DsStore | DsAtomic { .. }) => waited & !WD,
        Some(VmemLoad) => waited & !WL,
        Some(VmemStore) => waited & !WS,
        Some(VmemAtomic { returns }) => waited & !(if returns { WL } else { WS }),
        Some(_) => 0,
    }
}

/// The registers one region writes: the dispatcher-range SGPRs and every VGPR (a wide
/// destination, a VOPD half or a lane write included).
#[derive(Clone, Debug, Default)]
struct RegionDefs {
    s: BTreeSet<u16>,
    v: BTreeSet<u16>,
}

fn region_defs(
    kernel: &Kernel,
    arch: Arch,
    ins: &[Ins],
    p: &Placed,
    lo: u16,
    hi: u16,
) -> RegionDefs {
    use peacemaker_ir::reg::Kind;
    let body = &kernel.body;
    let mut out = RegionDefs::default();
    for pos in p.p_entry..p.p_exit.min(body.layout.len()) {
        let Some(inst) = body.insts.get(body.layout[pos]) else {
            continue;
        };
        // wave-width lane masks: a wave32 compare into `s5` does not write `s6`
        for d in &peacemaker_ir::effects::wave_effects(arch, kernel.wave, inst).defs {
            for r in d.base..d.base + u16::from(d.len) {
                match d.kind {
                    Kind::S if (lo..=hi).contains(&r) => {
                        out.s.insert(r);
                    }
                    Kind::V => {
                        out.v.insert(r);
                    }
                    _ => {}
                }
            }
        }
        // belt and braces from the printed form: the destination of a lane write and of
        // each half of a VOPD pair, whatever the effect table says
        let t = &ins[pos];
        let mut dsts: Vec<&String> = Vec::new();
        if t.mn == "v_writelane_b32" || t.mn.starts_with("v_dual_") {
            dsts.extend(t.ops.first());
            if let Some(i) = t.ops.iter().position(|o| o == "::") {
                dsts.extend(t.ops.get(i + 2));
            }
        }
        for o in dsts {
            if let Some((Rk::V, base, len)) = parse_reg(o) {
                out.v.extend(base..base + len);
            }
        }
    }
    out
}

/// The drained-counter knowledge at a region's exit, starting from every counter
/// drained at its entry (the entry check requires that): a forward must-dataflow
/// over the region's blocks.
fn region_exit_waited(kernel: &Kernel, ins: &[Ins], p: &Placed) -> u8 {
    let body = &kernel.body;
    let nb = body.blocks.len();
    let Some(first) = (0..nb).find(|&b| {
        let r = body.blocks[b].range;
        r.0 <= p.p_entry && p.p_entry < r.1
    }) else {
        return 0;
    };
    let mut at: Vec<Option<u8>> = vec![None; nb];
    at[first] = Some(WL | WS | WD | WK);
    let mut exit: Option<u8> = None;
    let mut work = vec![first];
    let meet = |a: Option<u8>, w: u8| Some(a.map_or(w, |o| o & w));
    while let Some(b) = work.pop() {
        let block = &body.blocks[b];
        let mut w = at[b].expect("queued with a state");
        let (start, end) = (block.range.0.max(p.p_entry), block.range.1.min(p.p_exit));
        for pos in start..end {
            if let Some(inst) = body.insts.get(body.layout[pos]) {
                w = apply_wait(inst, &ins[pos].mn, w);
            }
        }
        if block.range.1 > p.p_exit {
            exit = meet(exit, w);
            continue;
        }
        for s in &block.succs {
            let start = body.blocks[s.0].range.0;
            if start == p.p_exit {
                exit = meet(exit, w);
            } else if start >= p.p_entry && start < p.p_exit {
                let merged = meet(at[s.0], w);
                if at[s.0] != merged {
                    at[s.0] = merged;
                    work.push(s.0);
                }
            }
        }
    }
    exit.unwrap_or(0)
}
// ---------------------------------------------------------------------------
// what the walk derives
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
struct Derived {
    tasks: Option<u32>,
    ticket: Option<u32>,
    events: Option<u32>,
    n_tasks: Option<u32>,
    kernargs: Option<u32>,
    stride_shift: Option<u8>,
    event_shift: Option<u8>,
    template: Option<u32>,
    ka_off: Option<u32>,
    pred: Option<u32>,
    expected: Option<u32>,
    done: Option<u32>,
    slot: Option<u32>,
}
/// One way wave 0 filled the claim slot: the facts of the path that stored it and
/// the symbolic value of each dword it wrote (a `Lane` member marks an unknown one).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Event {
    facts: u32,
    cells: BTreeMap<u32, Alts>,
}

/// A loop of the dispatcher that runs one region once per step: `counter` (an SGPR)
/// is compared `counter < bound` by the latch block, whose taken edge returns to
/// `head`. Only the shape is found here ([`find_step_loops`]); the dataflow proves
/// what the two registers hold.
#[derive(Clone, Debug)]
struct StepLoop {
    head: usize,
    latch: usize,
    counter: u16,
    bound: u16,
    /// The natural loop of the back edge `latch -> head`.
    members: BTreeSet<usize>,
}

/// One region entry's view of the workgroup-from-header facts.
struct WgEntry {
    template: u32,
    /// Dispatcher-range SGPRs holding a plain header dword of the claimed task.
    plain: BTreeMap<u16, u32>,
    /// Dispatcher-range SGPRs a native region reads.
    native_reads: BTreeSet<u16>,
}

/// What `region_wg_from_header` collected on one pass of the walk.
#[derive(Default)]
struct WgRun {
    /// Why the property is not proved (empty: every check held).
    unproven: BTreeSet<String>,
    /// Header offsets named by `wg_x`/`wg_yz`/the bound: `vx`, `vy`, `vz`, `vcount`.
    fields: BTreeMap<&'static str, u32>,
    entries: Vec<WgEntry>,
    /// Templates entered inside each step loop (by head block).
    heads: BTreeMap<usize, BTreeSet<u32>>,
}

struct Walk<'a> {
    kernel: &'a Kernel,
    arch: Arch,
    ins: Vec<Ins>,
    block_of: Vec<usize>,
    notes: &'a KernelRegions,
    placed: &'a [Placed],
    waves: u32,
    head: usize,
    loop_members: BTreeSet<usize>,
    in_region: Vec<bool>,
    entry_of: HashMap<usize, usize>,
    template_ids: BTreeSet<u32>,
    lds_in: Vec<Event>,
    exit_waited: Vec<u8>,
    region_defs: Vec<RegionDefs>,
    steps: BTreeMap<usize, StepLoop>,
}

/// What the walk saw of span mode (DYNAMIC.md).
#[derive(Default)]
struct SpanRun {
    /// Header offsets compared against 0/1 as the flags word, used as the span-relative
    /// index k, used as the list id (each must resolve to exactly one).
    flags: BTreeSet<u32>,
    k: BTreeSet<u32>,
    list: BTreeSet<u32>,
    /// Params offsets of the span table pointer (null-checked).
    spans_arg: BTreeSet<u32>,
    /// Header offsets of the grid extent each entry axis was compared against.
    grid: [BTreeSet<u32>; 3],
    /// `(gx, gy)` header offsets of the bitmap index and of its tested bit.
    tile: BTreeSet<(u32, u32)>,
    /// (the count load requires `SPP | POLLED | ACQ`: refused otherwise)
    /// The done release was reached on the `k >= count` edge without a body.
    nop_release: bool,
    /// Constants written over a register holding header dword `.0`.
    overwrites: BTreeSet<(u32, u32)>,
    /// Span-table accesses: position -> (descriptor field byte offset, A1 class).
    sites: BTreeMap<usize, (u8, &'static str)>,
    /// Positions of the count load, the entry load and the bitmap atomic.
    count_sites: BTreeSet<usize>,
    entry_sites: BTreeSet<usize>,
    claim_sites: BTreeSet<usize>,
    /// Table loads (n_spans and descriptor fields).
    table_sites: BTreeSet<usize>,
    /// Region entries whose path is neither a static (flags == 0) nor a span-mode one.
    unclassified: BTreeSet<u32>,
    /// Span entries checked at region entry, by template.
    live_regions: BTreeSet<u32>,
}

#[derive(Default)]
struct Run {
    refusals: BTreeSet<(usize, &'static str, String)>,
    derived: RefCell<Derived>,
    lds_out: BTreeMap<u32, BTreeMap<u32, Alts>>,
    ticket_sites: BTreeSet<usize>,
    poison_sites: BTreeSet<usize>,
    plain_stores: BTreeSet<(usize, u32)>,
    signal_sites: BTreeSet<usize>,
    header_sites: BTreeSet<usize>,
    reached_entries: BTreeSet<usize>,
    unmatched_signal: bool,
    first_barrier: [Option<usize>; 2],
    backedges: usize,
    /// Header dwords the three b128 loads produced.
    hdr_loaded: BTreeSet<u32>,
    wg: WgRun,
    span: SpanRun,
}

fn set_once<T: PartialEq + Copy + std::fmt::Debug>(
    run: &mut Run,
    pos: usize,
    what: &str,
    slot: fn(&mut Derived) -> &mut Option<T>,
    value: T,
) {
    let mut d = run.derived.borrow_mut();
    let cell = slot(&mut d);
    match cell {
        None => *cell = Some(value),
        Some(old) if *old == value => {}
        Some(old) => {
            let detail = format!("{what} is {old:?} on one path and {value:?} on another");
            drop(d);
            run.refusals
                .insert((pos, "task_graph-protocol-inconsistent", detail));
        }
    }
}

impl Walk<'_> {
    fn pc(&self, pos: usize) -> u32 {
        super::rel_pc(
            self.kernel,
            self.kernel
                .body
                .insts
                .get(self.kernel.body.layout[pos])
                .and_then(|i| i.prov.pc),
        )
        .unwrap_or(0)
    }
    /// The instruction's effects with VALU lane masks at this kernel's wave width.
    fn fx(&self, inst: &Inst) -> peacemaker_ir::effects::Effects {
        peacemaker_ir::effects::wave_effects(self.arch, self.kernel.wave, inst)
    }

    fn inst(&self, pos: usize) -> &Inst {
        self.kernel
            .body
            .insts
            .get(self.kernel.body.layout[pos])
            .expect("laid out")
    }
    fn fail(&self, run: &mut Run, pos: usize, rule: &'static str, detail: impl Into<String>) {
        let detail = format!(
            "pc {:#x} `{}`: {}",
            self.pc(pos),
            self.ins[pos].mn,
            detail.into()
        );
        run.refusals.insert((pos, rule, detail));
    }
    fn multi(&self) -> bool {
        self.waves > 1
    }

    fn value(&self, st: &State, op: &str) -> Option<Alts> {
        if let Some(n) = parse_num(op) {
            return Some(one(E::Const(n as u32)));
        }
        let (k, base, len) = parse_reg(op)?;
        if len != 1 {
            return None;
        }
        match k {
            Rk::S => st.s.get(&base).cloned(),
            Rk::V => st.v.get(&base).cloned(),
            Rk::Other if op.starts_with("ttmp") => Some(one(E::Uni)),
            _ => None,
        }
    }

    fn put(&self, st: &mut State, op: &str, val: Option<Alts>) {
        let Some((k, base, len)) = parse_reg(op) else {
            return;
        };
        let map = match k {
            Rk::S => &mut st.s,
            Rk::V => &mut st.v,
            _ => return,
        };
        for i in 0..len {
            match &val {
                Some(v)
                    if len == 1
                        && v.len() <= MAX_ALTS
                        && v.iter().all(|e| e.depth() < MAX_DEPTH) =>
                {
                    map.insert(base + i, v.clone());
                }
                _ => {
                    map.remove(&(base + i));
                }
            }
        }
    }

    fn put_dword(&self, st: &mut State, kind: Rk, reg: u16, val: Option<Alts>) {
        let map = if kind == Rk::S { &mut st.s } else { &mut st.v };
        match val {
            Some(v) if v.len() <= MAX_ALTS && v.iter().all(|e| e.depth() < MAX_DEPTH) => {
                map.insert(reg, v);
            }
            _ => {
                map.remove(&reg);
            }
        }
    }

    /// A pointer pair `s[b:b+1]` whose dwords are `Param(off)`, `Param(off+4)`.
    fn param_ptr(&self, st: &State, op: &str) -> Option<u32> {
        let (k, base, len) = parse_reg(op)?;
        if k != Rk::S || len != 2 {
            return None;
        }
        match (
            st.s.get(&base)?.iter().next()?,
            st.s.get(&(base + 1))?.iter().next()?,
        ) {
            (E::Param(a), E::Param(b))
                if *b == a + 4 && st.s[&base].len() == 1 && st.s[&(base + 1)].len() == 1 =>
            {
                Some(*a)
            }
            _ => None,
        }
    }

    fn karg_pair(&self, st: &State, op: &str) -> bool {
        let Some((Rk::S, base, 2)) = parse_reg(op) else {
            return false;
        };
        st.s.get(&base) == Some(&one(E::Karg(0))) && st.s.get(&(base + 1)) == Some(&one(E::Karg(1)))
    }

    /// Whether the instruction's explicit scalar/vector inputs are all uniform.
    fn inputs_uniform(&self, st: &State, inst: &Inst) -> bool {
        self.fx(inst).uses.iter().all(|r| {
            let map = match r.kind {
                peacemaker_ir::reg::Kind::S if r.base < 106 => &st.s,
                peacemaker_ir::reg::Kind::V => &st.v,
                peacemaker_ir::reg::Kind::S => return false,
                _ => return true,
            };
            (0..u16::from(r.len)).all(|i| {
                map.get(&(r.base + i))
                    .is_some_and(|a| a.iter().all(E::uniform))
            })
        })
    }

    // ----- the transfer function -----------------------------------------

    fn step(&self, st: &mut State, pos: usize, run: &mut Run) {
        let ins = &self.ins[pos];
        let inst = self.inst(pos);
        let mn = ins.mn.as_str();
        let ops = &ins.ops;
        let mem = inst.effects.mem.is_some() || mn == "s_sendmsg_rtn_b64";
        let mut handled = true;
        // SGPRs parked in VGPR lanes across a region (`v_writelane_b32 vN, sK, lane` before
        // the entry, `v_readlane_b32 sK, vN, lane` after the exit): a lane write records the
        // copy, a read of a clobbered SGPR's own lane restores its symbolic value, and every
        // other write of either register kills the copy.
        for r in &self.fx(inst).uses {
            if r.kind != peacemaker_ir::reg::Kind::S {
                continue;
            }
            for k in r.base..r.base + u16::from(r.len) {
                if let Some(c) = st.clob.get(&k) {
                    self.fail(
                        run,
                        pos,
                        "task_graph-region-sgpr-park",
                        format!("reads s{k}, overwritten by region {} and not restored from its park since", c.by),
                    );
                }
            }
        }
        let park = match (mn, ops.len()) {
            ("v_writelane_b32", 3) => {
                match (parse_reg(&ops[0]), parse_reg(&ops[1]), parse_num(&ops[2])) {
                    (Some((Rk::V, v, 1)), Some((Rk::S, s, 1)), Some(l)) if (0..32).contains(&l) => {
                        Some(ParkOp::Write(v, l as u8, s))
                    }
                    _ => None,
                }
            }
            ("v_readlane_b32", 3) => {
                match (parse_reg(&ops[0]), parse_reg(&ops[1]), parse_num(&ops[2])) {
                    (Some((Rk::S, s, 1)), Some((Rk::V, v, 1)), Some(l)) if (0..32).contains(&l) => {
                        Some(ParkOp::Read(s, v, l as u8))
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        let keeps_lanes = match park {
            Some(ParkOp::Write(..)) => true,
            Some(ParkOp::Read(s, ..)) => st.clob.contains_key(&s),
            None => false,
        };
        if let (true, Some(ParkOp::Read(s, v, l))) = (keeps_lanes, park) {
            // a restore of a clobbered SGPR
            let c = st.clob.remove(&s).expect("clobbered");
            st.lanes.retain(|_, src| *src != s);
            if c.lanes.contains(&(v, l)) {
                match c.saved {
                    Some(a) => {
                        st.s.insert(s, a);
                    }
                    None => {
                        st.s.remove(&s);
                    }
                }
                st.lanes.insert((v, l), s);
            } else {
                st.s.remove(&s);
                let parked = if c.lanes.is_empty() {
                    "no lane on every path".to_owned()
                } else {
                    c.lanes
                        .iter()
                        .map(|(pv, pl)| format!("v{pv} lane {pl}"))
                        .collect::<Vec<_>>()
                        .join(" or ")
                };
                self.fail(
                    run,
                    pos,
                    "task_graph-region-sgpr-park",
                    format!("restores s{s} from v{v} lane {l}, but it was parked in {parked} before region {}", c.by),
                );
            }
        } else if let Some(ParkOp::Write(v, l, s)) = park {
            st.lanes.insert((v, l), s);
            for c in st.clob.values_mut() {
                c.lanes.remove(&(v, l));
            }
            st.v.remove(&v);
        }
        match mn {
            _ if keeps_lanes => {}
            "s_nop" | "s_wait_alu" | "s_delay_alu" | "s_sleep" | "s_clause" | "s_wait_event"
            | "s_waitcnt" => {}
            "s_mov_b32" | "v_mov_b32_e32" | "v_mov_b32" if ops.len() == 2 => {
                if ops[0] == "exec_lo" {
                    st.exec = match parse_num(&ops[1]) {
                        Some(1) => Exec::One,
                        Some(-1) => Exec::Full,
                        _ => Exec::Top,
                    };
                } else {
                    let v = self.value(st, &ops[1]);
                    // a constant written over a header dword (the template VGPR's NOP and
                    // poison values)
                    if let (Some((Rk::V, r, 1)), Some(E::Const(c))) =
                        (parse_reg(&ops[0]), single(&v))
                    {
                        if let Some(E::Hdr(k)) = st.v.get(&r).and_then(only) {
                            run.span.overwrites.insert((*k, *c));
                        }
                    }
                    self.put(st, &ops[0], v);
                }
            }
            "v_readfirstlane_b32" if ops.len() == 2 => {
                let mapped = if st.exec == Exec::Top {
                    None
                } else {
                    self.value(st, &ops[1]).and_then(|a| {
                        a.into_iter()
                            .map(|e| match e {
                                E::TidEntry if st.exec == Exec::Full => Some(E::WaveId),
                                E::TidEntry | E::Lane | E::Tid => None,
                                e => Some(e),
                            })
                            .collect::<Option<Alts>>()
                    })
                };
                self.put(st, &ops[0], mapped);
            }
            "s_lshl_b32" if ops.len() == 3 => {
                let r = match (self.value(st, &ops[1]), parse_num(&ops[2])) {
                    (Some(a), Some(n)) if (0..32).contains(&n) => Some(
                        a.into_iter()
                            .map(|e| match e {
                                E::Const(c) => E::Const(c << n),
                                e => E::Shl(Box::new(e), n as u8),
                            })
                            .collect(),
                    ),
                    _ => None,
                };
                self.put(st, &ops[0], r);
                st.scc = None;
            }
            "v_mbcnt_lo_u32_b32" if ops.len() == 3 && ops[1] == "-1" && ops[2] == "0" => {
                self.put(st, &ops[0], Some(one(E::Lane)));
            }
            "v_add_nc_u32_e32" | "v_add_nc_u32" if ops.len() == 3 => {
                let wave = self.value(st, &ops[1]) == Some(one(E::WaveId));
                let lane = self.value(st, &ops[2]) == Some(one(E::Lane));
                if wave && lane && st.exec == Exec::Full {
                    self.put(st, &ops[0], Some(one(E::Tid)));
                } else if let Some(e) = self.tile_add(st, &ops[1], &ops[2]) {
                    self.put(st, &ops[0], Some(one(e)));
                } else {
                    handled = false;
                }
            }
            "v_mul_lo_u32" | "v_and_b32_e32" | "v_and_b32" | "v_and_b32_e64"
            | "v_lshrrev_b32_e32" | "v_lshrrev_b32" | "v_lshlrev_b32_e32" | "v_lshlrev_b32"
            | "v_lshlrev_b32_e64"
                if ops.len() == 3 && st.exec != Exec::Top =>
            {
                match self.tile_op(st, mn, &ops[1], &ops[2]) {
                    Some(e) => self.put(st, &ops[0], Some(one(e))),
                    None => handled = false,
                }
            }
            m if m.starts_with("v_cmp_") && m.ends_with("_e64") && ops.len() == 3 => {
                // a wave32 compare writes ONE SGPR (the effects table's pair is an
                // over-approximation that would clobber the next register)
                let v = if st.exec == Exec::One {
                    self.mask_op(st, m, &ops[1], &ops[2])
                } else {
                    None
                };
                self.put(st, &ops[0], v.map(one));
            }
            "v_lshl_add_u32" if ops.len() == 4 => {
                // `(list_id << 5) + 32`: the span descriptor offset
                let (a, sh, c) = (
                    self.value(st, &ops[1]),
                    self.value(st, &ops[2]),
                    self.value(st, &ops[3]),
                );
                match (single(&a), single(&sh), single(&c)) {
                    (
                        Some(E::SpanId(l)),
                        Some(E::Const(SPAN_DESC_SHIFT_U32)),
                        Some(E::Const(32)),
                    ) => {
                        self.put(st, &ops[0], Some(one(E::DescOff(*l))));
                    }
                    _ => handled = false,
                }
            }
            "v_mad_co_u64_u32" if ops.len() == 5 => {
                // `live_list + 16 * k` into a VGPR pair
                let k = self.value(st, &ops[2]);
                let sixteen = self.value(st, &ops[3]);
                let base = match parse_reg(&ops[4]) {
                    Some((Rk::V, r, 2)) => (
                        st.v.get(&r).and_then(only).cloned(),
                        st.v.get(&(r + 1)).and_then(only).cloned(),
                    ),
                    _ => (None, None),
                };
                match (parse_reg(&ops[0]), single(&k), single(&sixteen), base) {
                    (
                        Some((Rk::V, d, 2)),
                        Some(E::SpanIdx(k)),
                        Some(E::Const(16)),
                        (
                            Some(E::SpanFld { off: 0, hi: false }),
                            Some(E::SpanFld { off: 0, hi: true }),
                        ),
                    ) => {
                        let k = *k;
                        self.put_dword(st, Rk::V, d, Some(one(E::LiveAddr { k, hi: false })));
                        self.put_dword(st, Rk::V, d + 1, Some(one(E::LiveAddr { k, hi: true })));
                    }
                    (
                        _,
                        _,
                        _,
                        (
                            Some(E::SpanFld { off: 0, hi: false }),
                            Some(E::SpanFld { off: 0, hi: true }),
                        ),
                    ) => {
                        // an address into the live list whose index is not `k < count <= K`
                        self.fail(
                            run,
                            pos,
                            "task_graph-span-entry",
                            "the entry address is not `live_list + 16 * k` with `k < count <= K` established by compares (the count > K and k >= count checks)",
                        );
                        handled = false;
                    }
                    _ => handled = false,
                }
            }
            "s_or_b32" if ops.len() == 3 => {
                let (a, b) = (self.value(st, &ops[1]), self.value(st, &ops[2]));
                match (single(&a), single(&b)) {
                    (Some(x @ E::Param(_)), Some(y @ E::Param(_))) => {
                        let v = E::or(x.clone(), y.clone());
                        self.put(st, &ops[0], Some(one(v)));
                    }
                    _ => handled = false,
                }
                st.scc = None;
            }
            "s_max_u32" if ops.len() == 3 && parse_num(&ops[2]) == Some(1) => {
                let r = self.value(st, &ops[1]).map(|a| {
                    a.into_iter()
                        .map(|e| match e {
                            E::Const(c) => E::Const(c.max(1)),
                            e @ E::MaxOne(_) => e,
                            e => E::MaxOne(Box::new(e)),
                        })
                        .collect::<Alts>()
                });
                self.put(st, &ops[0], r);
                st.scc = None;
            }
            "s_add_co_i32" if ops.len() == 3 => {
                let a = self.value(st, &ops[1]);
                let b = self.value(st, &ops[2]);
                match step_sum(a.as_ref(), b.as_ref()) {
                    Some(v) => self.put(st, &ops[0], Some(v)),
                    None => handled = false,
                }
                st.scc = None;
            }
            "v_lshl_or_b32"
                if ops.len() == 4 && st.exec != Exec::Top && parse_num(&ops[2]) == Some(16) =>
            {
                // the first active lane's `(a << 16) | c`; lane 0 is active under both
                // EXEC values modelled here
                match (self.value(st, &ops[1]), self.value(st, &ops[3])) {
                    (Some(a), Some(c)) if a.len() * c.len() <= MAX_ALTS => {
                        let v: Alts = a
                            .iter()
                            .flat_map(|x| {
                                c.iter()
                                    .map(move |y| E::or(E::Shl(Box::new(x.clone()), 16), y.clone()))
                            })
                            .collect();
                        self.put(st, &ops[0], Some(v));
                    }
                    _ => handled = false,
                }
            }
            "s_load_b32" | "s_load_b64" | "s_load_b128" | "s_load_b256" | "s_load_b512"
                if ops.len() == 3 =>
            {
                let n = match mn {
                    "s_load_b32" => 1,
                    "s_load_b64" => 2,
                    "s_load_b128" => 4,
                    "s_load_b256" => 8,
                    _ => 16,
                };
                let (dk, db, _) = parse_reg(&ops[0]).unwrap_or((Rk::Other, 0, 0));
                if self.karg_pair(st, &ops[1]) && dk == Rk::S {
                    let off = parse_num(&ops[2]).unwrap_or(-1);
                    for i in 0..n {
                        let v = (off >= 0).then(|| one(E::Param(off as u32 + 4 * i as u32)));
                        self.put_dword(st, Rk::S, db + i, v);
                    }
                } else {
                    for i in 0..n {
                        self.put_dword(st, Rk::S, db + i, None);
                    }
                }
            }
            "s_add_co_u32" if ops.len() == 3 => {
                let a = self.value(st, &ops[1]);
                let b = self.value(st, &ops[2]);
                match (single(&a), &b) {
                    (Some(E::Param(base)), Some(add)) => {
                        let base = *base;
                        let lo: Alts = add
                            .iter()
                            .map(|e| E::PtrLo {
                                base,
                                add: Box::new(e.clone()),
                            })
                            .collect();
                        self.put(st, &ops[0], Some(lo));
                        st.carry = Some((base, add.clone()));
                    }
                    _ => {
                        handled = false;
                    }
                }
                st.scc = None;
            }
            "s_add_co_ci_u32" if ops.len() == 3 => {
                let a = self.value(st, &ops[1]);
                let zero = parse_num(&ops[2]) == Some(0);
                match (single(&a), &st.carry, zero) {
                    (Some(E::Param(hi)), Some((base, add)), true) if *hi == base + 4 => {
                        let (base, add) = (*base, add.clone());
                        let v: Alts = add
                            .iter()
                            .map(|e| E::PtrHi {
                                base,
                                add: Box::new(e.clone()),
                            })
                            .collect();
                        self.put(st, &ops[0], Some(v));
                    }
                    _ => handled = false,
                }
                st.scc = None;
                st.carry = None;
            }
            m if m.starts_with("s_cmp_") && ops.len() == 2 => {
                let rest = &m["s_cmp_".len()..];
                let (rel, ty) = rest.split_once('_').unwrap_or(("", ""));
                let rel = match rel {
                    "eq" => Some(Rel::Eq),
                    "lg" => Some(Rel::Lg),
                    "ge" => Some(Rel::Ge),
                    "lt" => Some(Rel::Lt),
                    "gt" => Some(Rel::Gt),
                    "le" => Some(Rel::Le),
                    _ => None,
                };
                st.scc = match (rel, ty) {
                    (Some(rel), "u32" | "i32") => Some(Cmp {
                        rel,
                        signed: ty == "i32",
                        a: self.value(st, &ops[0]),
                        b: self.value(st, &ops[1]),
                        a_reg: parse_reg(&ops[0])
                            .and_then(|(k, b, l)| (k == Rk::S && l == 1).then_some(b)),
                    }),
                    _ => None,
                };
            }
            "s_wait_loadcnt"
            | "s_wait_storecnt"
            | "s_wait_dscnt"
            | "s_wait_kmcnt"
            | "s_wait_loadcnt_dscnt"
            | "s_wait_storecnt_dscnt"
            | "s_wait_samplecnt"
            | "s_wait_bvhcnt"
            | "s_wait_expcnt" => {}
            "global_inv" => {
                if ins.scope() >= 2 && st.waited & WL != 0 {
                    st.facts |= ACQ;
                    if st.facts & POLLED != 0 {
                        st.facts |= PO;
                    }
                }
            }
            "global_atomic_add_u32" => self.atomic(st, pos, run),
            "global_atomic_or_b32" => {
                if self.ins[pos].returns() {
                    self.span_claim(st, pos, run)
                } else {
                    self.poison_or(st, pos, run)
                }
            }
            "global_store_b32" => {
                let ins = &self.ins[pos];
                if ins.ops.len() == 3 {
                    if let Some(ptr) = self.param_ptr(st, &ins.ops[2]) {
                        run.plain_stores.insert((pos, ptr));
                    }
                }
                handled = false;
            }
            "global_load_b128" | "global_load_b32" => self.global_load(st, pos, run),
            "ds_store_b128" | "ds_store_b32" => self.ds_store(st, pos, run),
            "ds_load_b128" | "ds_load_b32" => self.ds_load(st, pos, run),
            "s_barrier_signal" => {}
            "s_barrier_wait" => self.barrier(st, pos, run),
            "s_endpgm" | "s_branch" | "s_cbranch_scc0" | "s_cbranch_scc1" => {}
            _ => handled = false,
        }
        if !handled {
            self.generic(st, inst, mem);
        }
        if !keeps_lanes {
            self.kill_copies(st, inst);
        }
        // the equality knowledge about `tmpl` dies with a write to its register
        let tmpl = self.notes.dispatcher.tmpl;
        if self.fx(inst).defs.iter().any(|d| {
            d.kind == peacemaker_ir::reg::Kind::S
                && d.base <= tmpl
                && d.base + u16::from(d.len) > tmpl
        }) {
            st.tmpl_eq = None;
            st.tmpl_ne.clear();
        }
        if mn != "s_add_co_u32" {
            st.carry = None;
        }
        // waits set bits; each memory issue clears the bit of the counter it counts on
        st.waited = apply_wait(inst, mn, st.waited);
        // single-wave pools have no barrier: the body is complete once drained
        if !self.multi()
            && st.facts & BODY != 0
            && st.facts & BDSYNC == 0
            && st.waited & (WL | WS) == (WL | WS)
        {
            st.facts |= BDSYNC;
        }
    }

    /// An instruction's register writes end the copies they overwrite: a VGPR write kills
    /// the lanes of that register (and the parks the dispatcher could restore from), an SGPR
    /// write kills the copies of that SGPR and its pending restore (the dispatcher has
    /// replaced the value it would have restored).
    fn kill_copies(&self, st: &mut State, inst: &Inst) {
        use peacemaker_ir::reg::Kind;
        for d in &self.fx(inst).defs {
            let (a, b) = (d.base, d.base + u16::from(d.len));
            match d.kind {
                Kind::V => {
                    st.lanes.retain(|(v, _), _| !(a..b).contains(v));
                    for c in st.clob.values_mut() {
                        c.lanes.retain(|(v, _)| !(a..b).contains(v));
                    }
                }
                Kind::S => {
                    st.lanes.retain(|_, s| !(a..b).contains(s));
                    for r in a..b {
                        st.clob.remove(&r);
                    }
                }
                _ => {}
            }
        }
    }

    fn generic(&self, st: &mut State, inst: &Inst, mem: bool) {
        let uniform = !mem
            && inst.effects.implicit.reads == 0
            && self.inputs_uniform(st, inst)
            && !matches!(inst.effects.control, Control::Barrier(_));
        for d in &self.fx(inst).defs {
            match d.kind {
                peacemaker_ir::reg::Kind::S if d.base >= 126 || d.base + u16::from(d.len) > 126 => {
                    st.exec = Exec::Top;
                }
                peacemaker_ir::reg::Kind::S | peacemaker_ir::reg::Kind::V => {
                    let (map, uniform_ok) = if d.kind == peacemaker_ir::reg::Kind::S {
                        (&mut st.s, uniform && d.base < 106)
                    } else {
                        (&mut st.v, uniform)
                    };
                    for i in 0..u16::from(d.len) {
                        if uniform_ok {
                            map.insert(d.base + i, one(E::Uni));
                        } else {
                            map.remove(&(d.base + i));
                        }
                    }
                }
                _ => {}
            }
        }
        if inst.effects.implicit.writes & peacemaker_ir::effects::ImplicitSet::SCC != 0 {
            st.scc = None;
        }
        if inst.effects.implicit.writes & peacemaker_ir::effects::ImplicitSet::EXEC != 0 {
            st.exec = Exec::Top;
        }
        st.carry = None;
    }

    /// `global_atomic_add_u32`: the ticket fetch-add or the release increment.
    fn atomic(&self, st: &mut State, pos: usize, run: &mut Run) {
        let ins = &self.ins[pos];
        let rets = ins.returns();
        let (dst, vaddr, data, saddr) = if rets && ins.ops.len() == 4 {
            (Some(&ins.ops[0]), &ins.ops[1], &ins.ops[2], &ins.ops[3])
        } else if !rets && ins.ops.len() == 3 {
            (None, &ins.ops[0], &ins.ops[1], &ins.ops[2])
        } else {
            self.fail(
                run,
                pos,
                "task_graph-protocol-atomic",
                "unrecognised operand shape",
            );
            return;
        };
        let Some(ptr) = self.param_ptr(st, saddr) else {
            self.fail(
                run,
                pos,
                "task_graph-protocol-atomic",
                "the base is not a pointer loaded from the params block",
            );
            if let Some(d) = dst {
                self.put(st, d, None);
            }
            return;
        };
        if ins.offset() != 0 || ins.scope() < 2 {
            self.fail(
                run,
                pos,
                "task_graph-protocol-atomic",
                "not a device-scope atomic at offset 0",
            );
        }
        if st.exec != Exec::One {
            self.fail(
                run,
                pos,
                "task_graph-protocol-atomic",
                "not issued by a single lane (EXEC is not lane 0 only)",
            );
        }
        if self.value(st, data) != Some(one(E::Const(1))) {
            self.fail(
                run,
                pos,
                "task_graph-protocol-atomic",
                "the increment is not the constant 1",
            );
        }
        let va = self.value(st, vaddr);
        if rets {
            // the ticket fetch-add
            if va != Some(one(E::Const(0))) {
                self.fail(
                    run,
                    pos,
                    "task_graph-claim",
                    "the ticket cell is not at byte 0 of its pointer",
                );
            }
            if self.multi() && st.facts & W0 == 0 {
                self.fail(
                    run,
                    pos,
                    "task_graph-claim",
                    "the claim is not confined to wave 0",
                );
            }
            set_once(run, pos, "ticket pointer", |d| &mut d.ticket, ptr);
            run.ticket_sites.insert(pos);
            if let Some(d) = dst {
                self.put(st, d, Some(one(E::Claim)));
            }
            st.facts |= CL;
        } else {
            // the release increment of events[done]
            run.signal_sites.insert(pos);
            set_once(run, pos, "events pointer", |d| &mut d.events, ptr);
            let done = match single(&va) {
                Some(E::Shl(x, sh)) => {
                    set_once(run, pos, "event stride shift", |d| &mut d.event_shift, *sh);
                    match self.resolve(st, x) {
                        Some((E::Hdr(k), f)) => Some((k, f)),
                        _ => None,
                    }
                }
                _ => None,
            };
            match done {
                Some((k, f)) => {
                    set_once(run, pos, "done event field", |d| &mut d.done, k);
                    if f & (CL | BD | HD | PO) != (CL | BD | HD | PO) {
                        self.fail(
                            run,
                            pos,
                            "task_graph-release",
                            "the done event index is not a field of a claimed, acquired header",
                        );
                    }
                }
                None => self.fail(
                    run,
                    pos,
                    "task_graph-release",
                    "the event index is not `done << k` of a header field",
                ),
            }
            if st.facts & DV == 0 {
                self.fail(
                    run,
                    pos,
                    "task_graph-release",
                    "reachable with done == NONE",
                );
            }
            if self.multi() && (st.facts & W0 == 0 || st.facts & DIV2 == 0) {
                self.fail(
                    run,
                    pos,
                    "task_graph-release",
                    "not confined to wave 0 after the release barrier",
                );
            }
            if st.waited & (WL | WS) != (WL | WS) {
                self.fail(run, pos, "task_graph-release", "the load and store counters are not waited to zero immediately before the release");
            }
            if st.facts & BODY != 0 {
                if st.facts & BDSYNC == 0 {
                    self.fail(run, pos, "task_graph-complete", "the body's memory operations are not waited (and, multi-wave, barriered) before the release");
                }
            } else {
                if st.facts & SPK != 0 {
                    run.span.nop_release = true;
                }
                // a template outside the note's table: released without a body
                let missing: Vec<u32> = self
                    .template_ids
                    .iter()
                    .copied()
                    .filter(|t| !st.tmpl_ne.contains(t))
                    .collect();
                if missing.is_empty() {
                    run.unmatched_signal = true;
                } else {
                    self.fail(
                        run,
                        pos,
                        "task_graph-release",
                        format!("reachable without a body for templates {missing:?}"),
                    );
                }
            }
            st.facts |= SIG;
        }
        st.waited &= !WK;
    }

    fn global_load(&self, st: &mut State, pos: usize, run: &mut Run) {
        let ins = &self.ins[pos];
        if ins.ops.len() != 3 {
            self.fail(
                run,
                pos,
                "task_graph-protocol-load",
                "unrecognised operand shape",
            );
            return;
        }
        let Some((Rk::V, dst, len)) = parse_reg(&ins.ops[0]) else {
            return;
        };
        let clear = |s: &Self, st: &mut State| {
            for i in 0..len {
                s.put_dword(st, Rk::V, dst + i, None);
            }
        };
        if self.span_global(st, pos, run) {
            return;
        }
        let Some(ptr) = self.param_ptr(st, &ins.ops[2]) else {
            clear(self, st);
            return;
        };
        let va = self.value(st, &ins.ops[1]);
        // header: tasks[claim << k], three b128 loads
        if let Some(E::Shl(x, sh)) = single(&va) {
            if **x == E::Claim {
                if ins.mn != "global_load_b128"
                    || st.exec != Exec::One
                    || st.facts & (CL | BD) != (CL | BD)
                {
                    self.fail(
                        run,
                        pos,
                        "task_graph-claim",
                        format!("a header load that is not a single-lane b128 load after claim < n_tasks (EXEC {:?}, path facts {:#x}, s92 {:?}, va {:?})", st.exec, st.facts, st.s.get(&92), va),
                    );
                    clear(self, st);
                    return;
                }
                set_once(run, pos, "tasks pointer", |d| &mut d.tasks, ptr);
                set_once(
                    run,
                    pos,
                    "descriptor stride shift",
                    |d| &mut d.stride_shift,
                    *sh,
                );
                run.header_sites.insert(pos);
                let off = ins.offset();
                for i in 0..len {
                    run.hdr_loaded.insert(off as u32 + 4 * i as u32);
                    self.put_dword(
                        st,
                        Rk::V,
                        dst + i,
                        Some(one(E::Hdr(off as u32 + 4 * i as u32))),
                    );
                }
                st.facts |= HD;
                st.waited &= !WL;
                return;
            }
            // the poll: events[pred << 2]
            if let E::Hdr(p) = &**x {
                if ins.mn == "global_load_b32"
                    && ins.scope() >= 2
                    && st.exec == Exec::One
                    && ins.offset() == 0
                {
                    set_once(run, pos, "events pointer", |d| &mut d.events, ptr);
                    set_once(run, pos, "event stride shift", |d| &mut d.event_shift, *sh);
                    set_once(run, pos, "pred event field", |d| &mut d.pred, *p);
                    self.put_dword(
                        st,
                        Rk::V,
                        dst,
                        Some(one(E::EventAt {
                            ptr,
                            idx: Box::new(E::Shl(Box::new(E::Hdr(*p)), *sh)),
                        })),
                    );
                    st.waited &= !WL;
                    return;
                }
            }
        }
        clear(self, st);
    }

    fn slot_cell(
        &self,
        st: &State,
        pos: usize,
        run: &mut Run,
        ops: &[String],
        width: u32,
    ) -> Option<u32> {
        let base = self.value(st, &ops[0]);
        let Some(E::Const(slot)) = single(&base) else {
            self.fail(
                run,
                pos,
                "task_graph-protocol-lds",
                "LDS address is not a constant slot",
            );
            return None;
        };
        let off = self.ins[pos].offset();
        if off < 0 || off as u32 + width > 64 {
            self.fail(
                run,
                pos,
                "task_graph-protocol-lds",
                "access outside the 64-byte claim slot",
            );
            return None;
        }
        set_once(run, pos, "claim slot", |d| &mut d.slot, *slot);
        Some(*slot + off as u32)
    }

    fn ds_store(&self, st: &mut State, pos: usize, run: &mut Run) {
        let ins = &self.ins[pos];
        let n = if ins.mn == "ds_store_b128" { 4 } else { 1 };
        if ins.ops.len() != 2 {
            self.fail(
                run,
                pos,
                "task_graph-protocol-lds",
                "unrecognised operand shape",
            );
            return;
        }
        let Some(cell) = self.slot_cell(st, pos, run, &ins.ops, 4 * n) else {
            return;
        };
        if st.exec != Exec::One || (self.multi() && st.facts & W0 == 0) {
            self.fail(
                run,
                pos,
                "task_graph-protocol-lds",
                "the slot is stored by more than wave 0 lane 0",
            );
        }
        if st.facts & (LOADED | DIV2) != 0 {
            self.fail(
                run,
                pos,
                "task_graph-protocol-lds",
                "stored before a barrier retired the previous trip's loads",
            );
        }
        let Some((Rk::V, src, _)) = parse_reg(&ins.ops[1]) else {
            return;
        };
        // every store of one path lands in one event, keyed by the path's facts
        let key = st.facts & !STORED;
        for j in 0..n {
            let vals = st.v.get(&(src + j as u16)).cloned();
            let slot = run
                .lds_out
                .entry(key)
                .or_default()
                .entry(cell + 4 * j)
                .or_default();
            match vals {
                Some(a) => slot.extend(a),
                None => {
                    slot.insert(E::Lane);
                }
            }
        }
        st.facts |= STORED;
        st.waited &= !(WD | WK);
    }

    fn ds_load(&self, st: &mut State, pos: usize, run: &mut Run) {
        let ins = &self.ins[pos];
        let n = if ins.mn == "ds_load_b128" { 4 } else { 1 };
        if ins.ops.len() != 2 {
            self.fail(
                run,
                pos,
                "task_graph-protocol-lds",
                "unrecognised operand shape",
            );
            return;
        }
        let addr_ops = [ins.ops[1].clone()];
        let Some(cell) = self.slot_cell(st, pos, run, &addr_ops, 4 * n) else {
            return;
        };
        if st.facts & (STORED | DIV2) != 0 {
            self.fail(
                run,
                pos,
                "task_graph-protocol-lds",
                "loaded before a barrier after this wave's own store",
            );
        }
        let Some((Rk::V, dst, _)) = parse_reg(&ins.ops[0]) else {
            return;
        };
        let events = self.lds_in.len();
        for j in 0..n {
            let v = (events > 0).then(|| one(E::Cell(cell + 4 * j)));
            self.put_dword(st, Rk::V, dst + j as u16, v);
        }
        st.choice = (events > 0).then(|| (0..events).collect());
        st.facts |= LOADED;
        st.facts &= !ACQ;
        st.waited &= !(WD | WK);
    }

    fn barrier(&self, st: &mut State, pos: usize, run: &mut Run) {
        if !self.multi() {
            return;
        }
        if st.facts & DIV2 != 0 {
            self.fail(
                run,
                pos,
                "task_graph-protocol-barrier",
                "a workgroup barrier inside the wave-0-only divergent tail",
            );
        }
        let role = match (st.facts & W0 != 0, st.facts & N0 != 0) {
            (true, false) => 0,
            (false, true) => 1,
            _ => {
                self.fail(
                    run,
                    pos,
                    "task_graph-protocol-barrier",
                    "a barrier on a path whose wave role is not decided",
                );
                return;
            }
        };
        if st.facts & MERGED == 0 {
            match run.first_barrier[role] {
                None => run.first_barrier[role] = Some(pos),
                Some(p) if p == pos => {}
                Some(p) => self.fail(
                    run,
                    pos,
                    "task_graph-protocol-barrier",
                    format!("the wave role's first barrier differs ({p} vs {pos})"),
                ),
            }
        }
        st.facts |= MERGED;
        st.facts &= !(STORED | LOADED);
        if st.facts & BODY != 0 && st.waited & (WL | WS) == (WL | WS) {
            st.facts |= BDSYNC;
        }
    }

    // ----- region entry --------------------------------------------------

    /// The value of `e` in `st`, with the facts it was established under: a slot
    /// cell resolves through the store events the loaded slot may hold (every one
    /// must agree on the value); anything else is established by this path.
    fn resolve(&self, st: &State, e: &E) -> Option<(E, u32)> {
        let E::Cell(off) = e else {
            return Some((e.clone(), st.facts));
        };
        let choice = st.choice.as_ref()?;
        let mut out: Option<(E, u32)> = None;
        for &id in choice {
            let ev = self.lds_in.get(id)?;
            let cell = ev.cells.get(off)?;
            let inner = only(cell)?;
            if *inner == E::Lane || *inner == E::Uni {
                return None;
            }
            out = match out {
                None => Some((inner.clone(), ev.facts)),
                Some((i, g)) if i == *inner => Some((i, g & ev.facts)),
                Some(_) => return None,
            };
        }
        out
    }

    fn witness(&self, st: &State, a: &Alts) -> Option<(E, u32)> {
        let mut out: Option<(E, u32)> = None;
        for e in a {
            let (inner, f) = self.resolve(st, e)?;
            out = match out {
                None => Some((inner, f)),
                Some((i, g)) if i == inner => Some((i, g & f)),
                Some(_) => return None,
            };
        }
        out
    }

    fn enter_region(&self, st: &mut State, pos: usize, placed: &Placed, idx: usize, run: &mut Run) {
        let t = placed.region.template_id;
        let d = &self.notes.dispatcher;
        let need = CL | BD | HD | PO;
        let mut errs: Vec<&str> = Vec::new();
        run.reached_entries.insert(pos);
        // (single-wave: the registers carry the path's own values; multi-wave, the values
        // come from the slot's store events and are checked per event, never by the
        // loading wave's own path facts)
        if !self.multi() && st.facts & SPK != 0 {
            errs.push("entered on the `k >= count` (NOP) edge of a span task");
        }
        if st.tmpl_eq != Some(t) {
            errs.push(
                "not entered on the taken edge of `tmpl == t` (or tmpl is rewritten in between)",
            );
        }
        match st.s.get(&d.tmpl).and_then(|a| self.witness(st, a)) {
            Some((E::Hdr(k), f)) if f & need == need => {
                set_once(run, pos, "template field", |dd| &mut dd.template, k);
            }
            _ => errs.push("the template id is not a header field of a claimed (claim < n_tasks), acquired task"),
        }
        let ka = st.s.get(&d.ka_off).cloned();
        match ka.as_ref().and_then(|a| self.witness(st, a)) {
            Some((E::Hdr(k), f)) if f & need == need => {
                set_once(run, pos, "kernarg offset field", |dd| &mut dd.ka_off, k);
            }
            _ => errs.push("the kernarg offset register is not a header field of the claimed task"),
        }
        let lo = st.s.get(&d.kernargs_pair_low).cloned();
        let hi = st.s.get(&(d.kernargs_pair_low + 1)).cloned();
        match (lo.as_ref().and_then(only), hi.as_ref().and_then(only), &ka) {
            (
                Some(E::PtrLo { base: b1, add: a1 }),
                Some(E::PtrHi { base: b2, add: a2 }),
                Some(ka),
            ) if b1 == b2 && a1 == a2 && ka.contains(&**a1) && ka.len() == 1 => {
                set_once(run, pos, "kernargs pointer", |dd| &mut dd.kernargs, *b1);
            }
            _ => errs.push("s[0:1] is not `kernargs + zext(ka_off)`"),
        }
        // The standalone bodies' certificates (the sealed-table case seeds a
        // nonzero EXEC at kernel entry) hold only if the region is entered
        // with the EXEC the kernel was launched with: every lane of the wave.
        // The dispatcher narrows EXEC to lane 0 for its claim code
        // (`s_mov_b32 exec_lo, 1`) and must have restored it on every path to
        // the entry; an entry with lane 0 only, zero or an unproven EXEC is
        // refused.
        if st.exec != Exec::Full {
            self.fail(
                run,
                pos,
                "task_graph-region-exec",
                format!(
                    "region {t}: EXEC at entry is not the kernel-entry EXEC (all lanes of the wave): the dispatcher's EXEC is lane-0 only, zero or not proven restored on every path to the region"
                ),
            );
        }
        if st.v.get(&0) != Some(&one(E::Tid)) {
            errs.push("v0 is not the workgroup-linear workitem id");
        }
        // Every counter is drained at region entry (PeaceMaker's `drain_all_counters`
        // before each region): the bodies' certificates start from nothing in flight, so
        // an operation of the dispatcher still outstanding is refused whatever the body
        // later waits for.
        if st.waited & (WL | WS | WD | WK) != (WL | WS | WD | WK) {
            self.fail(
                run,
                pos,
                "task_graph-region-drain",
                format!(
                    "region {t}: not every counter is drained at entry (load/store/ds/km drained {:#06b}, path facts {:#x}): a dispatcher memory operation may still be in flight",
                    st.waited, st.facts
                ),
            );
        }
        if st.facts & ACQ == 0 {
            errs.push("no device-scope acquire since the descriptor was read");
        }
        if self.multi() {
            if st.facts & MERGED == 0 {
                errs.push("entered before the broadcast barrier");
            }
            if st.facts & DIV2 != 0 {
                errs.push("entered inside the wave-0-only tail");
            }
        }
        for e in errs {
            self.fail(
                run,
                pos,
                "task_graph-protocol-entry",
                format!("region {t}: {e}"),
            );
        }
        self.check_wg(st, pos, placed, run);
        // The dispatcher's SGPR range survives, except what the region writes: each such
        // SGPR must be parked on every path into the entry in a VGPR lane the region does
        // not write (a copy that survives the region), and stays unusable until the
        // dispatcher restores it from that same lane.
        let defs = &self.region_defs[idx];
        let mut clob = BTreeMap::new();
        for &r in &defs.s {
            let parked: BTreeSet<(u16, u8)> = st
                .lanes
                .iter()
                .filter(|(_, &src)| src == r)
                .map(|(&k, _)| k)
                .collect();
            let usable: BTreeSet<(u16, u8)> = parked
                .iter()
                .copied()
                .filter(|(v, _)| !defs.v.contains(v))
                .collect();
            // Not parked at all: the register is dead for the dispatcher if no path reads
            // it before writing it again (a read of the clobbered register is refused at
            // the read). Parked, but only in VGPRs the region writes: refused here, the
            // dispatcher's restore would read the region's value.
            if usable.is_empty() && !parked.is_empty() {
                self.fail(
                    run,
                    pos,
                    "task_graph-region-sgpr-park",
                    format!(
                        "region {t} writes dispatcher register s{r}, parked in {}, which the region writes",
                        parked
                            .iter()
                            .map(|(v, l)| format!("v{v} lane {l}"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                );
            }
            clob.insert(
                r,
                Clob {
                    lanes: usable,
                    saved: st.s.get(&r).cloned(),
                    by: t,
                },
            );
        }
        let (lo_r, hi_r) = (d.lo, d.hi);
        st.s.retain(|k, _| (lo_r..=hi_r).contains(k) && !defs.s.contains(k));
        st.lanes.retain(|(v, _), src| {
            !defs.v.contains(v) && (lo_r..=hi_r).contains(src) && !defs.s.contains(src)
        });
        for (r, c) in clob {
            st.clob.insert(r, c);
        }
        st.v.clear();
        st.exec = Exec::Top;
        st.scc = None;
        st.carry = None;
        st.waited = self.exit_waited[idx];
        st.facts |= BODY;
        st.facts |= RAN;
        st.facts &= !BDSYNC;
    }

    // ----- edges ----------------------------------------------------------

    /// Refine `st` along the edge where `cmp` is `truth`; false = infeasible.
    fn refine(&self, st: &mut State, cmp: &Cmp, truth: bool, pos: usize, run: &mut Run) -> bool {
        let eq = match cmp.rel {
            Rel::Eq => Some(truth),
            Rel::Lg => Some(!truth),
            _ => None,
        };
        let a = single(&cmp.a).cloned();
        let b = single(&cmp.b).cloned();
        // the step loop's continue edge: `step + 1 < bound` taken
        if !cmp.signed && matches!((cmp.rel, truth), (Rel::Lt, true) | (Rel::Ge, false)) {
            if let (Some(next @ E::TripNext { head, bound }), Some(bd), Some(r)) =
                (&a, &b, cmp.a_reg)
            {
                if **bound == *bd && st.s.get(&r) == Some(&one(next.clone())) {
                    st.s.insert(
                        r,
                        one(E::TripCont {
                            head: *head,
                            bound: bound.clone(),
                        }),
                    );
                }
            }
        }
        if let Some(feasible) = self.refine_span(st, cmp, truth, run) {
            return feasible;
        }
        // wave role
        if a == Some(E::WaveId) && b == Some(E::Const(0)) {
            if let Some(is0) = eq {
                let (want, other) = if is0 { (W0, N0) } else { (N0, W0) };
                if st.facts & other != 0 {
                    return false;
                }
                st.facts |= want;
                if st.facts & MERGED != 0 && is0 {
                    // the second divergence: wave 0 alone does the release tail
                    st.facts |= DIV2;
                }
            }
            return true;
        }
        // claim >= n_tasks
        if cmp.rel == Rel::Ge && !cmp.signed {
            if let (Some(E::Claim), Some(E::Param(n))) = (&a, &b) {
                if !truth && st.facts & CL != 0 {
                    set_once(run, pos, "n_tasks field", |d| &mut d.n_tasks, *n);
                    st.facts |= BD;
                } else if truth {
                    set_once(run, pos, "n_tasks field", |d| &mut d.n_tasks, *n);
                }
                return true;
            }
            // events[pred] >= expected
            if let (Some(E::EventAt { ptr, idx }), Some(E::Hdr(e))) = (&a, &b) {
                if let E::Shl(p, sh) = &**idx {
                    if let E::Hdr(p) = &**p {
                        set_once(run, pos, "events pointer", |d| &mut d.events, *ptr);
                        set_once(run, pos, "pred event field", |d| &mut d.pred, *p);
                        set_once(run, pos, "event stride shift", |d| &mut d.event_shift, *sh);
                        set_once(run, pos, "expected field", |d| &mut d.expected, *e);
                        if truth && st.facts & (CL | BD | HD) == (CL | BD | HD) {
                            st.facts |= POLLED;
                        }
                    }
                }
                return true;
            }
        }
        let (Some(eq), Some(E::Const(c))) = (eq, &b) else {
            return true;
        };
        let c = *c;
        let is_tmpl = cmp.a_reg == Some(self.notes.dispatcher.tmpl);
        // pred / done == NONE (the template register is compared with NONE too, but is
        // neither)
        if c == NONE && !is_tmpl {
            let hdr: Option<(u32, bool)> = match &a {
                Some(E::Hdr(k)) => Some((*k, false)),
                Some(e @ E::Cell(_)) => match self.resolve(st, e) {
                    Some((E::Hdr(k), _)) => Some((k, true)),
                    _ => None,
                },
                _ => None,
            };
            if let Some((k, from_slot)) = hdr {
                // a `Hdr(k) == NONE` compare is the pred's before the acquire, and also after
                // it when `k` is the pred field the poll already named (the span path's
                // `pred == NONE` poison check); any other one is the done event's
                let named_pred = run.derived.borrow().pred == Some(k);
                let is_pred = !from_slot && (st.facts & ACQ == 0 || named_pred);
                if is_pred && !eq && st.facts & SPQ != 0 {
                    // the pred was already found to be NONE on this path: infeasible
                    return false;
                }
                if is_pred && !eq && st.facts & SP1 != 0 {
                    st.facts |= SPP;
                }
                if is_pred {
                    set_once(run, pos, "pred event field", |d| &mut d.pred, k);
                    if eq && st.facts & (CL | BD | HD) == (CL | BD | HD) {
                        st.facts |= PO | SPQ;
                    }
                } else {
                    set_once(run, pos, "done event field", |d| &mut d.done, k);
                    st.facts |= if eq { DN } else { DV };
                }
            }
        }
        // the register's own value: drop the constants and slot events it cannot be
        if let Some(r) = cmp.a_reg {
            let mut dead = false;
            if let Some(alts) = st.s.get_mut(&r) {
                alts.retain(|e| match e {
                    E::Const(x) => (*x == c) == eq,
                    _ => true,
                });
                dead |= alts.is_empty();
            }
            if let (Some(E::Cell(off)), Some(choice)) = (a.clone(), st.choice.as_mut()) {
                choice.retain(|&id| {
                    match self
                        .lds_in
                        .get(id)
                        .and_then(|ev| ev.cells.get(&off))
                        .and_then(only)
                    {
                        Some(E::Const(x)) => (*x == c) == eq,
                        _ => true,
                    }
                });
                dead |= choice.is_empty();
            }
            if dead {
                return false;
            }
            if r == self.notes.dispatcher.tmpl {
                if eq {
                    st.tmpl_eq = Some(c);
                } else {
                    st.tmpl_ne.insert(c);
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// the fixpoint
// ---------------------------------------------------------------------------

struct Fix<'a, 'b> {
    w: &'b Walk<'a>,
    states: BTreeMap<usize, BTreeMap<u32, State>>,
    /// Changed partitions to (re)visit, `(position, facts)`.
    work: Vec<(usize, u32)>,
    run: Run,
    /// `(position, facts)` states held: the path partitions.
    n_states: usize,
    /// `visit` runs so far: one per change of a partition.
    visits: u64,
}

impl Fix<'_, '_> {
    fn propagate(&mut self, pos: usize, st: State) {
        let st_facts = st.facts;
        let slot = self.states.entry(pos).or_default().entry(st_facts);
        use std::collections::btree_map::Entry;
        let changed = match slot {
            Entry::Vacant(v) => {
                v.insert(st);
                self.n_states += 1;
                true
            }
            Entry::Occupied(mut o) => {
                let before = o.get().clone();
                o.get_mut().join(&st);
                *o.get() != before
            }
        };
        // a partition is re-visited only when its own state changed: a visit is a function
        // of `(position, state)` and its effects on the run are idempotent set insertions,
        // so re-running an unchanged partition could only repeat them
        let key = (pos, st_facts);
        if changed && !self.work.contains(&key) {
            self.work.push(key);
        }
    }

    fn edge(&mut self, from_block: usize, to_block: usize, mut st: State) {
        let w = self.w;
        let target = w.kernel.body.blocks[to_block].range.0;
        if to_block == w.head && w.loop_members.contains(&from_block) {
            // a trip of the claim loop ends: every wave and path must have
            // finished its part of the task
            self.run.backedges += 1;
            let last = w.kernel.body.blocks[from_block].range.1 - 1;
            let f = st.facts;
            if f & (SIG | DN | N0) == 0 {
                w.fail(&mut self.run, last, "task_graph-once", "a trip of the claim loop ends without the release increment (or done == NONE, or a non-signalling wave)");
            }
            if f & BODY != 0 && f & BDSYNC == 0 {
                w.fail(
                    &mut self.run,
                    last,
                    "task_graph-complete",
                    "a wave leaves the task without having drained and barriered its body",
                );
            }
            st.facts = if w.multi() { 0 } else { W0 };
            st.scc = None;
            st.carry = None;
            st.tmpl_eq = None;
            st.tmpl_ne.clear();
            st.div2 = None;
            st.choice = None;
            // the next claim's code re-derives every header-derived value
            st.s.retain(|_, a| !a.iter().any(E::per_claim));
            st.v.retain(|_, a| !a.iter().any(E::per_claim));
        }
        if st.div2 == Some(target) {
            st.facts &= !DIV2;
            st.div2 = None;
        }
        if let Some(sl) = w.steps.get(&to_block) {
            w.step_head(&mut st, sl, sl.members.contains(&from_block), &mut self.run);
        }
        for sl in w.steps.values() {
            if sl.members.contains(&from_block) && !sl.members.contains(&to_block) {
                let counter = st.s.get(&sl.counter).and_then(only);
                let failed_compare = matches!(
                    counter,
                    Some(E::TripNext { head, .. }) if *head == sl.head
                );
                if from_block != sl.latch || !failed_compare {
                    self.run.wg.unproven.insert(format!(
                        "the step loop at block {} is left other than by its failed bound compare (edge block {from_block} -> {to_block})",
                        sl.head
                    ));
                } else if st.facts & RAN == 0 {
                    self.run.wg.unproven.insert(format!(
                        "the step loop at block {} is left after a trip that did not enter its region",
                        sl.head
                    ));
                }
            }
        }
        self.propagate(target, st);
    }

    fn visit(&mut self, start: usize, mut st: State) {
        let w = self.w;
        let body = &w.kernel.body;
        let mut pos = start;
        loop {
            if pos >= body.layout.len() {
                return;
            }
            if let Some(&pi) = w.entry_of.get(&pos) {
                let p = &w.placed[pi];
                w.enter_region(&mut st, pos, p, pi, &mut self.run);
                pos = p.p_exit;
                if pos >= body.layout.len() {
                    return;
                }
                continue;
            }
            if w.in_region[pos] {
                // reached the interior without passing an entry
                w.fail(
                    &mut self.run,
                    pos,
                    "task_graph-protocol-entry",
                    "control reaches the middle of a region",
                );
                return;
            }
            w.step(&mut st, pos, &mut self.run);
            let b = w.block_of[pos];
            let block = &body.blocks[b];
            if pos + 1 < block.range.1 {
                pos += 1;
                continue;
            }
            match block.term.clone() {
                Terminator::EndPgm | Terminator::Unreachable => {
                    if w.steps.values().any(|s| s.members.contains(&b)) {
                        self.run
                            .wg
                            .unproven
                            .insert(format!("the wave ends inside a step loop (block {b})"));
                    }
                }
                Terminator::FallThrough => {
                    if let Some(&n) = block.succs.first() {
                        self.edge(b, n.0, st);
                    }
                }
                Terminator::Jump(t) => self.edge(b, t.0, st),
                Terminator::Branch {
                    cond,
                    taken,
                    fallthrough,
                } => {
                    let scc_cond = matches!(cond, Cond::Scc0 | Cond::Scc1);
                    let cmp = if scc_cond { st.scc.clone() } else { None };
                    let role = cmp.as_ref().is_some_and(|c| {
                        single(&c.a) == Some(&E::WaveId) && single(&c.b) == Some(&E::Const(0))
                    });
                    if w.multi() {
                        let uniform = cmp.as_ref().is_some_and(Cmp::uniform);
                        let divergent_ok = st.facts & W0 != 0 && st.facts & MERGED == 0;
                        if !role && !uniform && !divergent_ok {
                            w.fail(&mut self.run, pos, "task_graph-protocol-uniform", "a conditional branch on a value that may differ between waves outside the wave-0 claim segment");
                        }
                    }
                    if !scc_cond && w.multi() {
                        w.fail(
                            &mut self.run,
                            pos,
                            "task_graph-protocol-uniform",
                            "a conditional branch on VCC/EXEC in the dispatcher",
                        );
                    }
                    for (is_taken, target) in [(true, taken), (false, fallthrough)] {
                        let mut s2 = st.clone();
                        let mut feasible = true;
                        if let Some(c) = &cmp {
                            let truth = (cond == Cond::Scc1) == is_taken;
                            feasible = w.refine(&mut s2, c, truth, pos, &mut self.run);
                        }
                        if role
                            && w.multi()
                            && feasible
                            && s2.facts & DIV2 != 0
                            && s2.div2.is_none()
                        {
                            s2.div2 = Some(body.blocks[taken.0].range.0);
                        }
                        if feasible {
                            self.edge(b, target.0, s2);
                        }
                    }
                }
            }
            return;
        }
    }

    /// Run the walk to its fixpoint within `budget`; `Err` names the bound that was
    /// exceeded (with the measured counts) and leaves the walk incomplete, so its result
    /// is never used.
    fn solve(&mut self, init: State, budget: &Budget) -> Result<(), String> {
        let blocks = self.w.kernel.body.blocks.len().max(1) as u64;
        let cap_visits = u64::from(budget.visits_per_block) * blocks;
        let cap_states = (u64::from(budget.states_per_block) * blocks) as usize;
        self.propagate(0, init);
        while let Some((pos, facts)) = self.work.pop() {
            let Some(st) = self.states.get(&pos).and_then(|m| m.get(&facts)).cloned() else {
                continue;
            };
            self.visits += 1;
            if self.visits > cap_visits {
                return Err(format!(
                    "the walk made {} state visits over {blocks} blocks, past its bound of {cap_visits} ({} per block); {} path partitions held",
                    self.visits, budget.visits_per_block, self.n_states
                ));
            }
            if self.n_states > cap_states {
                return Err(format!(
                    "the walk holds {} path partitions ((position, facts) states) over {blocks} blocks, past its bound of {cap_states} ({} per block), after {} visits",
                    self.n_states, budget.states_per_block, self.visits
                ));
            }
            self.visit(pos, st);
        }
        crate::timing::note(|| {
            format!(
                "walk pass: {} visits of {cap_visits}, {} partitions of {cap_states}, {blocks} blocks",
                self.visits, self.n_states
            )
        });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// entry point
// ---------------------------------------------------------------------------

/// What the protocol check hands back.
#[derive(Clone, Debug)]
pub struct ProtocolProof {
    pub record: TaskProtocolRecord,
    /// Kernarg offset of the pointer the regions' record blocks hang off.
    pub kernargs_arg_offset: u32,
    pub lds_slot_base: Option<u32>,
    /// Pointer arguments the dispatcher only stores to (trace, poison).
    pub output_args: BTreeSet<u32>,
    /// `(pc, params offset)` of every plain 32-bit store through a params pointer: the
    /// watchdog's poison store in a dispatcher that has not moved to a set-only atomic
    /// OR with release (the poison contract refuses it).
    pub poison_plain_stores: Vec<(Option<u32>, u32)>,
    pub evidence: Vec<String>,
    /// Span mode, when the dispatcher has it and every check of it is proved.
    pub span: Option<SpanProof>,
}

pub(super) fn check(
    kernel: &Kernel,
    arch: Arch,
    notes: &KernelRegions,
    placed: &[Placed],
    waves: u32,
    shell: &KernargFacts,
    _facts: &[RegionFacts],
    budget: &Budget,
) -> Result<ProtocolProof, Vec<Refusal>> {
    let body = &kernel.body;
    let mut out = Vec::new();
    let Abi::Hsa { descriptor, .. } = &kernel.abi else {
        return Err(vec![refuse("task_graph-protocol-abi", "not an HSA kernel")]);
    };
    if kernel.wave != peacemaker_ir::inst::Wave::Wave32 {
        out.push(refuse(
            "task_graph-protocol-abi",
            "the dispatcher is not wave32",
        ));
    }
    if (descriptor.compute_pgm_rsrc2.0 >> 11) & 3 != 0 {
        out.push(refuse(
            "task_graph-protocol-abi",
            "more than one workitem-id axis is enabled: v0 is not the linear id",
        ));
    }
    // instruction texts
    let mut ins = Vec::with_capacity(body.layout.len());
    for &id in &body.layout {
        let inst = body.insts.get(id).expect("laid out");
        let text = canonical(inst, arch).unwrap_or_default();
        ins.push(Ins::parse(&text));
    }
    let mut block_of = vec![0usize; body.layout.len()];
    for (b, block) in body.blocks.iter().enumerate() {
        block_of[block.range.0..block.range.1].fill(b);
    }
    let mut in_region = vec![false; body.layout.len()];
    let mut entry_of = HashMap::new();
    for (i, p) in placed.iter().enumerate() {
        for pos in p.p_entry..p.p_exit.min(body.layout.len()) {
            in_region[pos] = true;
        }
        entry_of.insert(p.p_entry, i);
    }
    // the claim loop: the outermost loop around a returning global atomic
    let atomic_positions: Vec<usize> = (0..body.layout.len())
        .filter(|&p| !in_region[p] && ins[p].mn.contains("atomic"))
        .collect();
    let cfg = match Cfg::build(body) {
        Ok(c) => c,
        Err(e) => {
            return Err(vec![refuse("task_graph-protocol-cfg", format!("{e:?}"))]);
        }
    };
    // the ticket is the returning `global_atomic_add_u32` (a span claim bitmap's returning
    // OR, later in the object, is not)
    let ticket_candidate = atomic_positions
        .iter()
        .copied()
        .find(|&p| ins[p].returns() && ins[p].mn == "global_atomic_add_u32");
    let Some(cand) = ticket_candidate else {
        return Err(vec![refuse(
            "task_graph-claim",
            "no ticket fetch-add (a returning atomic outside the regions)",
        )]);
    };
    let claim_block = BlockId(block_of[cand]);
    let Some(claim_loop) = cfg
        .loops()
        .iter()
        .filter(|l| l.members.contains(&claim_block))
        .max_by_key(|l| l.members.len())
    else {
        return Err(vec![refuse(
            "task_graph-claim",
            "the ticket fetch-add is not inside a loop: tasks are claimed once",
        )]);
    };
    let head = claim_loop.header.0;
    let loop_members: BTreeSet<usize> = claim_loop.members.iter().map(|b| b.0).collect();
    let steps = find_step_loops(kernel, &cfg, &ins, &in_region, &loop_members, head);
    // static scans of the shell: no flat/scratch, no atomic but the two sites
    for (pos, i) in ins.iter().enumerate() {
        if in_region[pos] {
            continue;
        }
        if i.mn.starts_with("flat_") || i.mn.starts_with("scratch_") || i.mn.starts_with("buffer_")
        {
            out.push(Refusal {
                rule: "task_graph-protocol-memory",
                detail: format!("`{}` in the dispatcher", i.mn),
                pc: super::rel_pc(
                    kernel,
                    body.insts.get(body.layout[pos]).and_then(|x| x.prov.pc),
                ),
            });
        }
    }
    let template_ids: BTreeSet<u32> = notes.regions.iter().map(|r| r.template_id).collect();
    let exit_waited: Vec<u8> = placed
        .iter()
        .map(|p| region_exit_waited(kernel, &ins, p))
        .collect();
    let region_defs: Vec<RegionDefs> = placed
        .iter()
        .map(|p| {
            region_defs(
                kernel,
                arch,
                &ins,
                p,
                notes.dispatcher.lo,
                notes.dispatcher.hi,
            )
        })
        .collect();
    let mut lds_in: Vec<Event> = Vec::new();
    let mut result: Option<Run> = None;
    let mut converged = false;
    let passes = budget.lds_passes.max(1);
    for _ in 0..passes {
        let w = Walk {
            kernel,
            arch,
            ins: ins.clone(),
            block_of: block_of.clone(),
            notes,
            placed,
            waves,
            head,
            loop_members: loop_members.clone(),
            in_region: in_region.clone(),
            entry_of: entry_of.clone(),
            template_ids: template_ids.clone(),
            lds_in: lds_in.clone(),
            exit_waited: exit_waited.clone(),
            region_defs: region_defs.clone(),
            steps: steps.clone(),
        };
        let mut init = State {
            s: BTreeMap::new(),
            v: BTreeMap::new(),
            exec: Exec::Full,
            scc: None,
            carry: None,
            facts: if waves == 1 { W0 } else { 0 },
            waited: WL | WS | WD | WK,
            tmpl_eq: None,
            tmpl_ne: BTreeSet::new(),
            div2: None,
            choice: None,
            lanes: BTreeMap::new(),
            clob: BTreeMap::new(),
        };
        init.s.insert(0, one(E::Karg(0)));
        init.s.insert(1, one(E::Karg(1)));
        init.v.insert(0, one(E::TidEntry));
        let mut fix = Fix {
            w: &w,
            states: BTreeMap::new(),
            work: Vec::new(),
            run: Run::default(),
            n_states: 0,
            visits: 0,
        };
        if let Err(why) =
            crate::timing::timed("fixpoint pass (Walk/Fix)", || fix.solve(init, budget))
        {
            out.push(refuse(
                "task_graph-protocol-budget",
                format!("protocol dataflow: {why}"),
            ));
            return Err(out);
        }
        let run = fix.run;
        let produced = events_of(&run.lds_out);
        let stable = produced == lds_in;
        lds_in = produced;
        result = Some(run);
        if stable {
            converged = true;
            break;
        }
    }
    let run = result.expect("at least one run");
    // the last pass must have been run against the slot contents it itself produced; a
    // walk whose input differs from its output proves nothing
    if !converged {
        out.push(refuse(
            "task_graph-protocol-budget",
            format!(
                "the LDS slot's contents did not reach a fixpoint within {} passes of the walk",
                budget.lds_passes.max(1)
            ),
        ));
    }
    for (pos, rule, detail) in &run.refusals {
        out.push(Refusal {
            rule: *rule,
            detail: detail.clone(),
            pc: super::rel_pc(
                kernel,
                body.insts.get(body.layout[*pos]).and_then(|x| x.prov.pc),
            ),
        });
    }
    // coverage
    for p in placed {
        if !run.reached_entries.contains(&p.p_entry) {
            out.push(refuse(
                "task_graph-protocol-entry",
                format!(
                    "region {} is never entered on any path the dispatcher can take",
                    p.region.template_id
                ),
            ));
        }
    }
    if run.ticket_sites.len() != 1 {
        out.push(refuse(
            "task_graph-claim",
            format!(
                "{} ticket fetch-add sites reached (exactly one is required)",
                run.ticket_sites.len()
            ),
        ));
    }
    if run.signal_sites.len() != 1 {
        out.push(refuse(
            "task_graph-once",
            format!(
                "{} release increment sites reached (exactly one is required)",
                run.signal_sites.len()
            ),
        ));
    }
    let span_present =
        !run.span.flags.is_empty() || !run.span.sites.is_empty() || !run.span.spans_arg.is_empty();
    let want_headers = if span_present { 4 } else { 3 };
    if run.header_sites.len() != want_headers {
        out.push(refuse(
            "task_graph-claim",
            format!(
                "{} header loads reached ({want_headers} b128 loads of the header are required: three for the 48-byte head{})",
                run.header_sites.len(),
                if span_present { ", one for the span words" } else { "" }
            ),
        ));
    }
    for &p in &atomic_positions {
        if !run.ticket_sites.contains(&p)
            && !run.signal_sites.contains(&p)
            && !run.poison_sites.contains(&p)
            && !run.span.claim_sites.contains(&p)
        {
            let swap = ins[p].mn.contains("swap");
            out.push(Refusal {
                rule: if swap {
                    "task_graph-poison-swap"
                } else {
                    "task_graph-protocol-memory"
                },
                detail: if swap {
                    format!("`{}`: a swap/cmpswap on a synchronisation or poison cell; the contract allows only monotonic (set-only OR, ADD, MAX) updates", ins[p].mn)
                } else {
                    format!("an atomic the claim/release protocol does not account for: `{}`", ins[p].mn)
                },
                pc: super::rel_pc(kernel, body.insts.get(body.layout[p]).and_then(|x| x.prov.pc)),
            });
        }
    }
    if run.backedges == 0 {
        out.push(refuse(
            "task_graph-once",
            "the claim loop's back edge is never reached",
        ));
    }
    if waves > 1 && run.first_barrier[0] != run.first_barrier[1] {
        out.push(refuse(
            "task_graph-protocol-barrier",
            format!(
                "wave 0 and the other waves meet at different first barriers ({:?} vs {:?})",
                run.first_barrier[0], run.first_barrier[1]
            ),
        ));
    }
    // the release and the claim run once per trip of the claim loop: neither block
    // lies on a cycle that avoids the loop head (an inner loop)
    for (what, pos) in run
        .signal_sites
        .iter()
        .map(|&p| ("release increment", p))
        .chain(run.ticket_sites.iter().map(|&p| ("ticket fetch-add", p)))
    {
        let b = block_of[pos];
        if !loop_members.contains(&b) || on_inner_cycle(kernel, &loop_members, head, b) {
            out.push(refuse(
                "task_graph-once",
                format!("the {what} is outside the claim loop or on an inner cycle (it could run more than once per claim)"),
            ));
        }
    }
    let d = run.derived.borrow().clone();
    macro_rules! need {
        ($f:ident, $what:expr) => {
            match d.$f {
                Some(v) => v,
                None => {
                    out.push(refuse(
                        "task_graph-protocol-fields",
                        format!("{} was not derived", $what),
                    ));
                    0
                }
            }
        };
    }
    let tasks = need!(tasks, "the tasks pointer");
    let ticket = need!(ticket, "the ticket pointer");
    let events = need!(events, "the events pointer");
    let n_tasks = need!(n_tasks, "the n_tasks field");
    let kernargs = need!(kernargs, "the kernargs pointer");
    let stride_shift = need!(stride_shift, "the descriptor stride");
    let event_shift = need!(event_shift, "the event stride");
    let template_offset = need!(template, "the template field");
    let ka_off_offset = need!(ka_off, "the kernarg offset field");
    let pred = need!(pred, "the pred field");
    let expected = need!(expected, "the expected field");
    let done = need!(done, "the done field");
    if !out.is_empty() {
        return Err(out);
    }
    let roles = [tasks, ticket, events, kernargs];
    for (i, a) in roles.iter().enumerate() {
        if roles[..i].contains(a) {
            out.push(refuse(
                "task_graph-protocol-fields",
                "two protocol roles resolve to the same params pointer",
            ));
        }
    }
    let ptr_arg = |off: u32| {
        shell.args.iter().any(|a| {
            a.offset == off && a.size == 8 && a.role == peacemaker_ir::kernarg::ArgRole::Pointer
        })
    };
    for (name, off) in [
        ("tasks", tasks),
        ("ticket", ticket),
        ("events", events),
        ("kernargs", kernargs),
    ] {
        if !ptr_arg(off) {
            out.push(refuse(
                "task_graph-protocol-fields",
                format!("the {name} role's params dword +{off} is not an 8-byte pointer argument of the shell's A1"),
            ));
        }
    }
    let arg_ok = shell
        .args
        .iter()
        .any(|a| a.offset == n_tasks && a.size >= 4);
    if !arg_ok {
        out.push(refuse(
            "task_graph-protocol-fields",
            format!("n_tasks +{n_tasks} is not a kernel argument"),
        ));
    }
    if !out.is_empty() {
        return Err(out);
    }
    let roles_set: BTreeSet<u32> = roles.into_iter().collect();
    let mut output_args: BTreeSet<u32> = shell
        .args
        .iter()
        .filter(|a| !a.accesses.is_empty() && !roles_set.contains(&a.offset))
        .map(|a| a.offset)
        .collect();
    let wg = wg_verdict(
        &run,
        notes,
        [template_offset, ka_off_offset, pred, expected, done],
    );
    let mut proved = vec![
        TASK_PROTO_ACQUIRE,
        TASK_PROTO_BOUNDED,
        TASK_PROTO_CLAIM,
        TASK_PROTO_COMPLETE,
        TASK_PROTO_ONCE,
        TASK_PROTO_RELEASE,
    ];
    let (grid, wg_evidence) = match wg {
        Ok((g, line)) => {
            proved.push(TASK_PROTO_REGION_WG);
            (Some(g), line)
        }
        Err(why) => (
            None,
            format!(
                "protocol: {TASK_PROTO_REGION_WG} NOT proved: {}",
                why.join("; ")
            ),
        ),
    };
    // span mode: proved or the pool is refused (its table accesses are unattributed
    // otherwise)
    let mut span_proof: Option<SpanProof> = None;
    if span_present {
        if grid.is_none() {
            return Err(vec![refuse(
                "task_graph-span-region",
                format!("span mode needs the header grid fields: {wg_evidence}"),
            )]);
        }
        match span_verdict(
            kernel,
            &run,
            notes,
            grid.as_ref(),
            [template_offset, ka_off_offset, pred, expected, done],
            stride_shift,
        ) {
            Ok(p) => {
                proved.push(TASK_PROTO_SPAN_ENTRY);
                proved.push(TASK_PROTO_REGION_WG_LIVE);
                // the span table is read through a read-only pointer argument of the
                // shell's A1 (n_spans and descriptors), never written
                output_args.remove(&p.offsets.spans_arg_offset);
                if !ptr_arg(p.offsets.spans_arg_offset) {
                    return Err(vec![refuse(
                        "task_graph-span-table",
                        format!(
                            "the span table's params dword +{} is not an 8-byte pointer argument of the shell's A1",
                            p.offsets.spans_arg_offset
                        ),
                    )]);
                }
                span_proof = Some(p);
            }
            Err(rs) => return Err(rs),
        }
    }
    proved.sort_unstable();
    let record = TaskProtocolRecord {
        grid,
        tasks_arg_offset: tasks,
        ticket_arg_offset: ticket,
        events_arg_offset: events,
        n_tasks_arg_offset: n_tasks,
        desc_stride: 1 << stride_shift,
        template_offset,
        ka_off_offset,
        pred_event_offset: pred,
        expected_offset: expected,
        done_event_offset: done,
        none_event: NONE,
        event_stride: 1 << event_shift,
        proved: proved.into_iter().map(str::to_owned).collect(),
        span: span_proof.as_ref().map(|s| s.offsets),
        assumptions: Vec::new(),
    };
    let mut evidence = vec![format!(
        "protocol: claim fetch-add on +{ticket}, header tasks(+{tasks})[claim<<{stride_shift}] fields tmpl@{template_offset} ka_off@{ka_off_offset} pred@{pred} expected@{expected} done@{done}, events(+{events})[i<<{event_shift}], n_tasks@+{n_tasks}, arena kernargs(+{kernargs}); {} wave(s)",
        waves
    )];
    evidence.push(wg_evidence);
    if let Some(s) = &span_proof {
        evidence.extend(s.evidence.iter().cloned());
    }
    if run.unmatched_signal {
        evidence.push(format!(
            "protocol: a task whose template is not one of {:?} is released without running a body (the host must not submit one)",
            template_ids
        ));
    }
    Ok(ProtocolProof {
        record,
        kernargs_arg_offset: kernargs,
        lds_slot_base: d.slot,
        output_args,
        poison_plain_stores: run
            .plain_stores
            .iter()
            .map(|&(pos, ptr)| {
                (
                    super::rel_pc(
                        kernel,
                        body.insts.get(body.layout[pos]).and_then(|x| x.prov.pc),
                    ),
                    ptr,
                )
            })
            .collect(),
        evidence,
        span: span_proof,
    })
}

/// Whether block `b` can reach itself without passing the claim loop's head (it lies
/// on a loop nested inside the claim loop).
fn on_inner_cycle(kernel: &Kernel, members: &BTreeSet<usize>, head: usize, b: usize) -> bool {
    let blocks = &kernel.body.blocks;
    let mut seen = BTreeSet::new();
    let mut work: Vec<usize> = blocks[b].succs.iter().map(|s| s.0).collect();
    while let Some(x) = work.pop() {
        if x == head || !members.contains(&x) || !seen.insert(x) {
            continue;
        }
        if x == b {
            return true;
        }
        work.extend(blocks[x].succs.iter().map(|s| s.0));
    }
    false
}

impl Walk<'_> {
    /// The poison cell's monotonic update: a set-only `global_atomic_or_b32` (no return,
    /// device scope) issued after the load and store counters are waited (a release).
    fn poison_or(&self, st: &mut State, pos: usize, run: &mut Run) {
        let ins = &self.ins[pos];
        if ins.returns() || ins.ops.len() != 3 || self.param_ptr(st, &ins.ops[2]).is_none() {
            self.fail(
                run,
                pos,
                "task_graph-protocol-memory",
                "an OR that is not a no-return atomic on a params pointer",
            );
            return;
        }
        if ins.scope() < 2 || st.waited & (WL | WS) != (WL | WS) {
            self.fail(
                run,
                pos,
                "task_graph-poison-swap",
                "the poison OR is not a device-scope release (counters not waited before it)",
            );
        }
        run.poison_sites.insert(pos);
    }
}

/// `a + b` where one side is a step counter ([`E::Trip`]): the counter plus one (the
/// latch's increment) or the counter added to other values (a region's `vx + step`).
fn step_sum(a: Option<&Alts>, b: Option<&Alts>) -> Option<Alts> {
    let (a, b) = (a?, b?);
    let is_trip = |x: &Alts| matches!(only(x), Some(E::Trip { .. }));
    let (t, other) = if is_trip(b) {
        (b, a)
    } else if is_trip(a) {
        (a, b)
    } else {
        return None;
    };
    let t = only(t)?.clone();
    if let (Some(E::Const(1)), E::Trip { head, bound }) = (only(other), &t) {
        return Some(one(E::TripNext {
            head: *head,
            bound: bound.clone(),
        }));
    }
    if other.iter().any(E::is_step) {
        return None;
    }
    Some(other.iter().map(|e| E::add(e.clone(), t.clone())).collect())
}

/// `header[vx] + step` with `step` the counter of a step loop bounded by
/// `max(header[vcount], 1)`: `(vx, vcount, head block)`.
fn parts_x(e: &E) -> Option<(u32, u32, usize)> {
    let E::Add(a, b) = e else { return None };
    for (h, t) in [(&**a, &**b), (&**b, &**a)] {
        if let (E::Hdr(vx), E::Trip { head, bound }) = (h, t) {
            let raw = match &**bound {
                E::MaxOne(x) => &**x,
                x => x,
            };
            if let E::Hdr(vc) = raw {
                return Some((*vx, *vc, *head));
            }
        }
    }
    None
}

/// `header[vy] | header[vz] << 16`: `(vy, vz)`.
fn parts_yz(e: &E) -> Option<(u32, u32)> {
    let E::Or(a, b) = e else { return None };
    for (s, l) in [(&**a, &**b), (&**b, &**a)] {
        if let (E::Shl(z, 16), E::Hdr(vy)) = (s, l) {
            if let E::Hdr(vz) = &**z {
                return Some((*vy, *vz));
            }
        }
    }
    None
}

/// `entry.vx + step` with `step` the counter of a step loop whose bound is the constant 1
/// (`vcount` forced to 1: `step < 1`, so `step == 0` and the region runs once at `entry.vx`):
/// the head block of the loop.
fn parts_x_live(e: &E) -> Option<usize> {
    let E::Add(a, b) = e else { return None };
    for (h, t) in [(&**a, &**b), (&**b, &**a)] {
        if let (E::Entry(0), E::Trip { head, bound }) = (h, t) {
            let one_step = match &**bound {
                E::Const(1) => true,
                E::MaxOne(x) => **x == E::Const(1),
                _ => false,
            };
            if one_step {
                return Some(*head);
            }
        }
    }
    None
}

/// `entry.vy | entry.vz << 16`.
fn parts_yz_live(e: &E) -> Option<()> {
    let E::Or(a, b) = e else { return None };
    for (s, l) in [(&**a, &**b), (&**b, &**a)] {
        if let (E::Shl(z, 16), E::Entry(1)) = (s, l) {
            if **z == E::Entry(2) {
                return Some(());
            }
        }
    }
    None
}

/// The step loops of the dispatcher: a block outside every region that ends
/// `s_cmp_lt_u32 c, b; s_cbranch_scc1 head` (or `s_cmp_ge_u32 ..; s_cbranch_scc0`),
/// with `head` an inner block that dominates it (a real back edge), not the claim
/// loop's head. A header claimed by two latches is dropped.
fn find_step_loops(
    kernel: &Kernel,
    cfg: &Cfg,
    ins: &[Ins],
    in_region: &[bool],
    claim: &BTreeSet<usize>,
    claim_head: usize,
) -> BTreeMap<usize, StepLoop> {
    let body = &kernel.body;
    let mut found: BTreeMap<usize, StepLoop> = BTreeMap::new();
    let mut dup = BTreeSet::new();
    for &b in claim {
        let block = &body.blocks[b];
        let (start, end) = block.range;
        // (the latch block may start inside the region it follows; its branch does not)
        if start >= end || in_region[end - 1] {
            continue;
        }
        let Terminator::Branch { cond, taken, .. } = &block.term else {
            continue;
        };
        let want = match cond {
            Cond::Scc1 => "s_cmp_lt_u32",
            Cond::Scc0 => "s_cmp_ge_u32",
            _ => continue,
        };
        let mut p = end - 1;
        let cmp = loop {
            if p == start {
                break None;
            }
            p -= 1;
            match ins[p].mn.as_str() {
                "s_wait_alu" | "s_delay_alu" | "s_nop" => continue,
                m if m == want => break Some(p),
                _ => break None,
            }
        };
        let Some(p) = cmp else { continue };
        let reg = |i: usize| match ins[p].ops.get(i).and_then(|o| parse_reg(o)) {
            Some((Rk::S, r, 1)) => Some(r),
            _ => None,
        };
        let (Some(counter), Some(bound)) = (reg(0), reg(1)) else {
            continue;
        };
        let h = taken.0;
        if h == claim_head || !claim.contains(&h) || !cfg.dominates(BlockId(h), BlockId(b)) {
            continue;
        }
        let mut members = BTreeSet::from([h]);
        let mut work = vec![b];
        while let Some(x) = work.pop() {
            if members.insert(x) {
                work.extend(body.blocks[x].preds.iter().map(|q| q.0));
            }
        }
        let sl = StepLoop {
            head: h,
            latch: b,
            counter,
            bound,
            members,
        };
        if found.insert(h, sl).is_some() {
            dup.insert(h);
        }
    }
    for h in dup {
        found.remove(&h);
    }
    found
}

/// Whether `region_wg_from_header` holds after the walk, and the header offsets it
/// rests on: `Ok((grid offsets, evidence))` or the reasons it is not proved.
/// `other` are the offsets of the template, `ka_off`, `pred`, `expected`, `done`
/// fields.
fn wg_verdict(
    run: &Run,
    notes: &KernelRegions,
    other: [u32; 5],
) -> Result<(TaskGridOffsets, String), Vec<String>> {
    let mut why: Vec<String> = run.wg.unproven.iter().cloned().collect();
    let d = &notes.dispatcher;
    let get = |k: &str| run.wg.fields.get(k).copied();
    let (Some(vx), Some(vy), Some(vz), Some(vcount)) =
        (get("vx"), get("vy"), get("vz"), get("vcount"))
    else {
        why.push("the region entries did not yield vx, vy, vz and vcount".into());
        return Err(why);
    };
    let templates: BTreeSet<u32> = run.wg.entries.iter().map(|e| e.template).collect();
    for r in &notes.regions {
        if !templates.contains(&r.template_id) {
            why.push(format!("region {} was not checked", r.template_id));
        }
    }
    for (head, ts) in &run.wg.heads {
        if ts.len() > 1 {
            why.push(format!(
                "the step loop at block {head} runs several regions {ts:?}"
            ));
        }
    }
    let known = [
        other[0], other[1], other[2], other[3], other[4], vx, vy, vz, vcount,
    ];
    for (i, k) in known.iter().enumerate() {
        if known[..i].contains(k) {
            why.push(format!(
                "two header fields resolve to offset {k} (vx@{vx} vy@{vy} vz@{vz} vcount@{vcount})"
            ));
            break;
        }
    }
    // The grid: the one header dword the dispatcher keeps live in a dispatcher
    // SGPR besides the fields above is the virtual grid's x extent (what a native
    // region's grid-stride loop reads); y and z follow it in the header layout and
    // are never read by the dispatcher.
    let mut candidates: BTreeSet<(u16, u32)> = BTreeSet::new();
    for e in &run.wg.entries {
        for (&r, &k) in &e.plain {
            // the span header words (flags, k, list id) the dispatcher keeps live while
            // it validates a span entry are not grid fields
            let span_word = run.span.flags.contains(&k)
                || run.span.k.contains(&k)
                || run.span.list.contains(&k);
            if !known.contains(&k) && !span_word {
                candidates.insert((r, k));
            }
        }
    }
    let grid_reg = match candidates.iter().next() {
        Some(&(r, k)) if candidates.len() == 1 => Some((r, k)),
        _ => {
            why.push(format!(
                "the dispatcher does not keep exactly one other header dword live in its SGPRs ({candidates:?}): the grid x field is not derived"
            ));
            None
        }
    };
    let Some((grid_r, gx)) = grid_reg else {
        return Err(why);
    };
    for e in &run.wg.entries {
        if e.plain.get(&grid_r) != Some(&gx) {
            why.push(format!(
                "region {}: s{grid_r} does not hold header[{gx}]",
                e.template
            ));
        }
        if let Some(x) = e
            .native_reads
            .iter()
            .find(|&&r| r != d.wg_x && r != d.wg_yz && r != grid_r)
        {
            why.push(format!(
                "native region {} reads dispatcher register s{x} beyond the workgroup registers and the grid register s{grid_r}",
                e.template
            ));
        }
    }
    let (gy, gz) = (gx + 4, gx + 8);
    for g in [gx, gy, gz] {
        if known.contains(&g) || !run.hdr_loaded.contains(&g) {
            why.push(format!(
                "grid field offset {g} is not a loaded header dword distinct from the other fields"
            ));
        }
    }
    if !why.is_empty() {
        return Err(why);
    }
    let grid = TaskGridOffsets {
        vx_offset: vx,
        vy_offset: vy,
        vz_offset: vz,
        gx_offset: gx,
        gy_offset: gy,
        gz_offset: gz,
        vcount_offset: vcount,
    };
    let line = format!(
        "protocol: {TASK_PROTO_REGION_WG}: wg_x(s{}) = header[vx@{vx}] + step, wg_yz(s{}) = header[vy@{vy}] | header[vz@{vz}] << 16 (no masking: exact only for vy, vz < 65536), step over [0, max(header[vcount@{vcount}], 1)): counter 0 at loop entry, +1 per trip, back edge while step+1 < bound, every trip enters its region; grid x = header[gx@{gx}] in s{grid_r} (what a native region's stride reads); gy@{gy}/gz@{gz} are the header layout's next two dwords, never read by the dispatcher; lifted bodies take their grid from their record's hidden arguments, which the dispatcher does not write (the host must set them equal to the header grid)",
        d.wg_x, d.wg_yz
    );
    Ok((grid, line))
}

impl Walk<'_> {
    /// The value of `e` with every LDS slot cell replaced by what its store events
    /// agree on, and the facts it was established under (their meet).
    fn canon(&self, st: &State, e: &E) -> Option<(E, u32)> {
        Some(match e {
            E::Cell(_) => self.resolve(st, e)?,
            E::Shl(x, n) => {
                let (c, f) = self.canon(st, x)?;
                (E::Shl(Box::new(c), *n), f)
            }
            E::MaxOne(x) => {
                let (c, f) = self.canon(st, x)?;
                (E::MaxOne(Box::new(c)), f)
            }
            E::Or(a, b) => {
                let ((ca, fa), (cb, fb)) = (self.canon(st, a)?, self.canon(st, b)?);
                (E::or(ca, cb), fa & fb)
            }
            E::Add(a, b) => {
                let ((ca, fa), (cb, fb)) = (self.canon(st, a)?, self.canon(st, b)?);
                (E::add(ca, cb), fa & fb)
            }
            E::Trip { head, bound } => {
                let (c, f) = self.canon(st, bound)?;
                (
                    E::Trip {
                        head: *head,
                        bound: Box::new(c),
                    },
                    f,
                )
            }
            _ => (e.clone(), st.facts),
        })
    }

    /// An edge into the head of a step loop: the induction. From outside the loop the
    /// counter must be exactly 0 (and the bound register a single value): it becomes
    /// [`E::Trip`]. On a back edge it must be [`E::TripCont`] of the same loop (the
    /// increment, compared against the bound on the taken edge), after a trip that
    /// entered its region. Anything else forgets the counter, so no region entry
    /// downstream can use it.
    fn step_head(&self, st: &mut State, sl: &StepLoop, back: bool, run: &mut Run) {
        let cur = st.s.get(&sl.counter).and_then(only).cloned();
        let new = if back {
            if st.facts & RAN == 0 {
                run.wg.unproven.insert(format!(
                    "the step loop at block {} takes its back edge after a trip that did not enter its region",
                    sl.head
                ));
                None
            } else {
                match cur {
                    Some(E::TripCont { head, bound }) if head == sl.head => {
                        Some(E::Trip { head, bound })
                    }
                    _ => None,
                }
            }
        } else {
            let bound = st.s.get(&sl.bound).and_then(only).cloned();
            match (cur, bound) {
                (Some(E::Const(0)), Some(b)) => Some(E::Trip {
                    head: sl.head,
                    bound: Box::new(b),
                }),
                _ => None,
            }
        };
        // a counter that is not the induction value is still one value in every wave
        // when it was uniform (the unmodelled-value rule of `generic`)
        let uniform =
            st.s.get(&sl.counter)
                .is_some_and(|a| a.iter().all(E::uniform))
                .then(|| one(E::Uni));
        self.put_dword(st, Rk::S, sl.counter, new.map(one).or(uniform));
        st.facts &= !RAN;
    }

    fn wg_set(&self, run: &mut Run, t: u32, name: &'static str, v: u32) {
        match run.wg.fields.get(name) {
            None => {
                run.wg.fields.insert(name, v);
            }
            Some(&old) if old == v => {}
            Some(&old) => {
                run.wg.unproven.insert(format!(
                    "region {t}: {name} is header[{v}] here and header[{old}] elsewhere"
                ));
            }
        }
    }

    /// `region_wg_from_header` at one region entry (see the module docs): the
    /// region's `wg_x`/`wg_yz` registers and the header dwords the dispatcher keeps
    /// live. Failures only withhold the property; the pool's other proofs stand.
    fn check_wg(&self, st: &State, pos: usize, placed: &Placed, run: &mut Run) {
        // each store event of the claim slot that may hold the region's descriptor is its own
        // class of path: a static task (flags == 0) or a validated span entry
        match &st.choice {
            Some(c) if !c.is_empty() => {
                for &id in c {
                    let mut one_event = st.clone();
                    one_event.choice = Some(BTreeSet::from([id]));
                    let facts = self.lds_in.get(id).map_or(0, |e| e.facts);
                    self.check_wg_one(&one_event, pos, placed, run, facts);
                }
            }
            _ => self.check_wg_one(st, pos, placed, run, st.facts),
        }
    }

    /// [`Walk::check_wg`] for one path class, `facts` the facts the descriptor was stored
    /// (or held) under: a span entry (`SP1`) must be proved `region_wg_from_live_entry`
    /// (a failure is a refusal: the region would run on unvalidated coordinates), a
    /// static one `region_wg_from_header`.
    fn check_wg_one(&self, st: &State, pos: usize, placed: &Placed, run: &mut Run, facts: u32) {
        let t = placed.region.template_id;
        let d = &self.notes.dispatcher;
        let need = CL | BD | HD | PO;
        let live = facts & SP1 != 0;
        let mut why: Vec<String> = Vec::new();
        let mut found: Vec<(&'static str, u32)> = Vec::new();
        if live {
            run.span.live_regions.insert(t);
            if facts & SP_ALL != SP_ALL {
                why.push(format!(
                    "a span-mode path reaches the region without every check (missing fact bits {:#x} of {:#x}: flags, table pointer, vx/vy/vz < grid, bitmap bit clear)",
                    SP_ALL & !facts,
                    SP_ALL
                ));
            }
        } else if facts & SP0 == 0 {
            run.span.unclassified.insert(t);
        }
        if !placed.region.is_native()
            && (placed.region.wg_x != d.wg_x || placed.region.wg_yz != d.wg_yz)
        {
            why.push(format!(
                "its row reads the workgroup from s{}/s{}, not the dispatcher's s{}/s{}",
                placed.region.wg_x, placed.region.wg_yz, d.wg_x, d.wg_yz
            ));
        }
        let held = |reg: u16| -> Option<E> {
            let (e, f) = self.canon(st, st.s.get(&reg).and_then(only)?)?;
            (f & need == need).then_some(e)
        };
        match held(d.wg_x) {
            Some(e) => match if live {
                parts_x_live(&e).map(|h| (None, h))
            } else {
                parts_x(&e).map(|(vx, vc, h)| (Some((vx, vc)), h))
            } {
                Some((fields, head)) => {
                    if let Some((vx, vc)) = fields {
                        found.push(("vx", vx));
                        found.push(("vcount", vc));
                    }
                    match self.steps.get(&head) {
                        Some(sl) if sl.members.contains(&self.block_of[pos]) => {
                            run.wg.heads.entry(head).or_default().insert(t);
                        }
                        _ => why.push(
                            "its step counter's loop does not contain the region entry".into(),
                        ),
                    }
                }
                None if live => why.push(format!(
                    "wg_x register s{} is not `entry.vx + step` with step < max(1, 1) (vcount forced to 1): {e:?}",
                    d.wg_x
                )),
                None => why.push(format!(
                    "wg_x register s{} is not `header[vx] + step` with step < max(header[vcount], 1): {e:?}",
                    d.wg_x
                )),
            },
            None => why.push(format!(
                "wg_x register s{} holds no value derived from the claimed header",
                d.wg_x
            )),
        }
        match held(d.wg_yz) {
            Some(e) => match if live {
                parts_yz_live(&e).map(|()| None)
            } else {
                parts_yz(&e).map(Some)
            } {
                Some(fields) => {
                    if let Some((vy, vz)) = fields {
                        found.push(("vy", vy));
                        found.push(("vz", vz));
                    }
                }
                None if live => why.push(format!(
                    "wg_yz register s{} is not `entry.vy | entry.vz << 16`: {e:?}",
                    d.wg_yz
                )),
                None => why.push(format!(
                    "wg_yz register s{} is not `header[vy] | header[vz] << 16`: {e:?}",
                    d.wg_yz
                )),
            },
            None => why.push(format!(
                "wg_yz register s{} holds no value derived from the claimed header",
                d.wg_yz
            )),
        }
        let mut plain = BTreeMap::new();
        for r in d.lo..=d.hi {
            if let Some(E::Hdr(k)) = held(r) {
                plain.insert(r, k);
            }
        }
        let mut native_reads = BTreeSet::new();
        if placed.region.is_native() {
            for p in placed.p_entry..placed.p_exit.min(self.kernel.body.layout.len()) {
                for r in &self.fx(self.inst(p)).uses {
                    if r.kind == peacemaker_ir::reg::Kind::S {
                        for i in 0..u16::from(r.len) {
                            let g = r.base + i;
                            if (d.lo..=d.hi).contains(&g) {
                                native_reads.insert(g);
                            }
                        }
                    }
                }
            }
        }
        for (name, v) in found {
            self.wg_set(run, t, name, v);
        }
        if live {
            for w in why {
                self.fail(
                    run,
                    pos,
                    "task_graph-span-region",
                    format!("region {t}: {w}"),
                );
            }
        } else {
            run.wg
                .unproven
                .extend(why.into_iter().map(|w| format!("region {t}: {w}")));
        }
        run.wg.entries.push(WgEntry {
            template: t,
            plain,
            native_reads,
        });
    }
}

// ---------------------------------------------------------------------------
// span mode (railgunmega/DYNAMIC.md)
// ---------------------------------------------------------------------------

/// Byte offset of descriptor 0 of the span table (`n_spans` and 28 reserved bytes first).
const SPAN_DESC_BASE: i64 = 0x20;
/// log2 of a span descriptor's size and of a live entry's size.
const SPAN_DESC_SHIFT: u8 = 5;
const SPAN_DESC_SHIFT_U32: u32 = SPAN_DESC_SHIFT as u32;
const SPAN_ENTRY_SHIFT: u8 = 4;

impl Walk<'_> {
    fn singles(&self, st: &State, a: &str, b: &str) -> Option<(E, E)> {
        let x = self.value(st, a)?;
        let y = self.value(st, b)?;
        Some((only(&x)?.clone(), only(&y)?.clone()))
    }

    /// `bit = vx + gx * (vy + gy * vz)`: the sum that completes one stage.
    fn tile_add(&self, st: &State, a: &str, b: &str) -> Option<E> {
        let (x, y) = self.singles(st, a, b)?;
        for (m, e) in [(&x, &y), (&y, &x)] {
            if let (E::Mul(p, q), E::Entry(f)) = (m, e) {
                match (&**p, &**q, f) {
                    (E::Entry(2), E::Hdr(gy), 1) => {
                        return Some(E::Tile {
                            st: TileSt::Yz,
                            gx: 0,
                            gy: *gy,
                        })
                    }
                    (
                        E::Tile {
                            st: TileSt::Yz, gy, ..
                        },
                        E::Hdr(gx),
                        0,
                    ) => {
                        return Some(E::Tile {
                            st: TileSt::Idx,
                            gx: *gx,
                            gy: *gy,
                        })
                    }
                    _ => {}
                }
            }
        }
        None
    }

    /// The multiplies, shifts and masks of the bitmap index and of the claim test.
    fn tile_op(&self, st: &State, mn: &str, a: &str, b: &str) -> Option<E> {
        let (x, y) = self.singles(st, a, b)?;
        let tile = |t: TileSt, gx: u32, gy: u32| E::Tile { st: t, gx, gy };
        if mn == "v_mul_lo_u32" {
            let (m, h) = match (&x, &y) {
                (E::Hdr(_), o) => (o, &x),
                (o, E::Hdr(_)) => (o, &y),
                _ => return None,
            };
            return match m {
                E::Entry(2) | E::Tile { st: TileSt::Yz, .. } => {
                    Some(E::Mul(Box::new(m.clone()), Box::new(h.clone())))
                }
                _ => None,
            };
        }
        if mn.starts_with("v_and_b32") {
            for (p, q) in [(&x, &y), (&y, &x)] {
                match (p, q) {
                    (
                        E::Const(31),
                        E::Tile {
                            st: TileSt::Idx,
                            gx,
                            gy,
                        },
                    ) => return Some(tile(TileSt::Low, *gx, *gy)),
                    (
                        E::Tile {
                            st: TileSt::Old,
                            gx,
                            gy,
                        },
                        E::Tile {
                            st: TileSt::Mask,
                            gx: hx,
                            gy: hy,
                        },
                    ) if gx == hx && gy == hy => return Some(tile(TileSt::Hit, *gx, *gy)),
                    _ => {}
                }
            }
            return None;
        }
        // `v_lshrrev dst, count, value` / `v_lshlrev dst, count, value`
        if mn.starts_with("v_lshrrev_b32") {
            return match (&x, &y) {
                (
                    E::Const(5),
                    E::Tile {
                        st: TileSt::Idx,
                        gx,
                        gy,
                    },
                ) => Some(tile(TileSt::Word, *gx, *gy)),
                _ => None,
            };
        }
        if mn.starts_with("v_lshlrev_b32") {
            return match (&x, &y) {
                (
                    E::Const(2),
                    E::Tile {
                        st: TileSt::Word,
                        gx,
                        gy,
                    },
                ) => Some(tile(TileSt::WordOff, *gx, *gy)),
                (
                    E::Tile {
                        st: TileSt::Low,
                        gx,
                        gy,
                    },
                    E::Const(1),
                ) => Some(tile(TileSt::Mask, *gx, *gy)),
                _ => None,
            };
        }
        None
    }

    /// Lane 0's `a >= b` / `a > b` (an entry dword against a header dword, the list id against
    /// `n_spans`, k against the bounded count, the count against K) and `0 != x` masks of a
    /// wave32 `v_cmp_*_e64` (whose destination is ONE SGPR).
    fn mask_op(&self, st: &State, mn: &str, a: &str, b: &str) -> Option<E> {
        let (x, y) = self.singles(st, a, b)?;
        let reg = match parse_reg(a) {
            Some((Rk::V, r, 1)) => r,
            _ => u16::MAX,
        };
        let cmp = |gt: bool| {
            Some(E::VCmp {
                gt,
                a: Box::new(x.clone()),
                b: Box::new(y.clone()),
                reg,
            })
        };
        match (mn, &x, &y) {
            ("v_cmp_ge_u32_e64", E::Entry(_), E::Hdr(_))
            | ("v_cmp_ge_u32_e64", E::Hdr(_), E::SpanN)
            | ("v_cmp_ge_u32_e64", E::Hdr(_), E::SpanCountLe) => cmp(false),
            ("v_cmp_gt_u32_e64", E::SpanCount, E::SpanFld { off: 24, hi: false }) => cmp(true),
            (
                "v_cmp_ne_u32_e64",
                E::Const(0),
                h @ E::Tile {
                    st: TileSt::Hit, ..
                },
            )
            | (
                "v_cmp_ne_u32_e64",
                h @ E::Tile {
                    st: TileSt::Hit, ..
                },
                E::Const(0),
            ) => Some(E::NzMask(Box::new(h.clone()))),
            _ => None,
        }
    }

    /// The descriptor field (byte offset 0, 8, 16) whose pointer pair `op` holds.
    fn span_ptr(&self, st: &State, op: &str) -> Option<u8> {
        let (k, base, len) = parse_reg(op)?;
        if k != Rk::S || len != 2 {
            return None;
        }
        match (only(st.s.get(&base)?)?, only(st.s.get(&(base + 1))?)?) {
            (E::SpanFld { off: a, hi: false }, E::SpanFld { off: b, hi: true })
                if a == b && matches!(a, 0 | 8 | 16) =>
            {
                Some(*a)
            }
            _ => None,
        }
    }

    /// A global load in span mode: through the span table pointer (`n_spans`, a descriptor)
    /// or through a pointer a descriptor held (the count cell, the live-list entry
    /// `live_list + 16 * k` with `k < count <= K`). Returns whether the load was one of
    /// these (then fully handled here).
    fn span_global(&self, st: &mut State, pos: usize, run: &mut Run) -> bool {
        let ins = &self.ins[pos];
        let Some((Rk::V, dst, len)) = parse_reg(&ins.ops[0]) else {
            return false;
        };
        let clear = |s: &Self, st: &mut State| {
            for i in 0..len {
                s.put_dword(st, Rk::V, dst + i, None);
            }
        };
        let ctx = st.exec == Exec::One && st.facts & (SP1 | SPN) == (SP1 | SPN) && ins.scope() < 2;
        if ins.ops[2] == "off" {
            // a flat-style load through a VGPR pair a descriptor field or the entry address
            // computation produced
            let Some((Rk::V, base, 2)) = parse_reg(&ins.ops[1]) else {
                return false;
            };
            let (Some(lo), Some(hi)) = (
                st.v.get(&base).and_then(only).cloned(),
                st.v.get(&(base + 1)).and_then(only).cloned(),
            ) else {
                return false;
            };
            match (lo, hi) {
                (E::SpanFld { off: 8, hi: false }, E::SpanFld { off: 8, hi: true }) => {
                    if !(ctx && ins.mn == "global_load_b32" && ins.offset() == 0) {
                        self.fail(
                            run,
                            pos,
                            "task_graph-span-count",
                            "the count load is not a single-lane plain b32 load of the count cell at offset 0 on a flags == 1 path",
                        );
                        clear(self, st);
                        return true;
                    }
                    if st.facts & (ACQ | POLLED | SPP) != (ACQ | POLLED | SPP) {
                        self.fail(
                            run,
                            pos,
                            "task_graph-span-count",
                            "the count load is reachable without the pred acquire (pred != NONE observed >= expected, then the device-scope acquire): a span header with pred == NONE must poison before it",
                        );
                    }
                    run.span.count_sites.insert(pos);
                    run.span.sites.insert(pos, (8, "vmem_load"));
                    self.put_dword(st, Rk::V, dst, Some(one(E::SpanCount)));
                }
                (E::LiveAddr { k, hi: false }, E::LiveAddr { k: k2, hi: true }) if k == k2 => {
                    if !(ctx && ins.mn == "global_load_b128" && ins.offset() == 0) {
                        self.fail(
                            run,
                            pos,
                            "task_graph-span-entry",
                            "the entry load is not a single-lane plain b128 load at offset 0 on a flags == 1 path",
                        );
                        clear(self, st);
                        return true;
                    }
                    run.span.k.insert(k);
                    run.span.entry_sites.insert(pos);
                    run.span.sites.insert(pos, (0, "vmem_load"));
                    for i in 0..len {
                        self.put_dword(st, Rk::V, dst + i, Some(one(E::Entry(i as u8))));
                    }
                }
                _ => return false,
            }
            st.waited &= !WL;
            return true;
        }
        // through the span table pointer: n_spans at +0, descriptor `list_id` at
        // `32 + 32 * list_id`
        let Some(p) = self.param_ptr(st, &ins.ops[2]) else {
            return false;
        };
        if !run.span.spans_arg.contains(&p) {
            return false;
        }
        run.span.table_sites.insert(pos);
        let va = self.value(st, &ins.ops[1]);
        if !ctx {
            self.fail(
                run,
                pos,
                "task_graph-span-table",
                "the span table is read on a path that has not established flags == 1 and a nonzero table pointer (single lane, plain load)",
            );
            clear(self, st);
            return true;
        }
        let off = ins.offset();
        match (single(&va), ins.mn.as_str(), off) {
            (Some(E::Const(0)), "global_load_b32", 0) => {
                self.put_dword(st, Rk::V, dst, Some(one(E::SpanN)));
            }
            (Some(E::DescOff(l)), "global_load_b128", 0 | 16) => {
                run.span.list.insert(*l);
                let f = |off: u8, hi: bool| Some(one(E::SpanFld { off, hi }));
                let words = if off == 0 {
                    [f(0, false), f(0, true), f(8, false), f(8, true)]
                } else {
                    [f(16, false), f(16, true), f(24, false), None]
                };
                for (i, w) in words.into_iter().enumerate() {
                    self.put_dword(st, Rk::V, dst + i as u16, w);
                }
            }
            _ => {
                self.fail(
                    run,
                    pos,
                    "task_graph-span-table",
                    "a load through the span table pointer that is neither `n_spans` (offset 0) nor a descriptor `spans + 32 + 32 * list_id` with `list_id < n_spans` established by a compare",
                );
                clear(self, st);
            }
        }
        st.waited &= !WL;
        true
    }

    /// The returning `global_atomic_or_b32` on the claim bitmap: word `bit >> 5`, mask
    /// `1 << (bit & 31)` of the validated entry's `bit`.
    fn span_claim(&self, st: &mut State, pos: usize, run: &mut Run) {
        let ins = &self.ins[pos];
        let ops = &ins.ops;
        let dst = ops.first().and_then(|o| parse_reg(o));
        let kill = |s: &Self, st: &mut State| {
            if let Some((Rk::V, d, 1)) = dst {
                s.put_dword(st, Rk::V, d, None);
            }
        };
        let field = if ops.len() == 4 {
            self.span_ptr(st, &ops[3])
        } else {
            None
        };
        let Some(16) = field else {
            self.fail(
                run,
                pos,
                "task_graph-protocol-memory",
                "a returning OR that is not on the claim bitmap pointer of a span descriptor",
            );
            kill(self, st);
            return;
        };
        let bounds = SP1 | SPN | SPX | SPY | SPZ;
        let va = self.value(st, &ops[1]);
        let data = self.value(st, &ops[2]);
        let dims = match (single(&va), single(&data)) {
            (
                Some(E::Tile {
                    st: TileSt::WordOff,
                    gx,
                    gy,
                }),
                Some(E::Tile {
                    st: TileSt::Mask,
                    gx: hx,
                    gy: hy,
                }),
            ) if gx == hx && gy == hy => Some((*gx, *gy)),
            _ => None,
        };
        let ok = st.exec == Exec::One
            && ins.scope() >= 2
            && ins.offset() == 0
            && st.facts & bounds == bounds;
        let (true, Some((gx, gy))) = (ok, dims) else {
            self.fail(
                run,
                pos,
                "task_graph-span-claim",
                "the claim is not a single-lane device-scope returning OR of bit `vx + gx * (vy + gy * vz)` of the entry on a path where all three grid compares held",
            );
            kill(self, st);
            return;
        };
        run.span.claim_sites.insert(pos);
        run.span.sites.insert(pos, (16, "atomic_commutative"));
        if let Some((Rk::V, d, 1)) = dst {
            self.put_dword(
                st,
                Rk::V,
                d,
                Some(one(E::Tile {
                    st: TileSt::Old,
                    gx,
                    gy,
                })),
            );
        }
        st.waited &= !WL;
    }

    /// Facts and rewritten values a compare's edge establishes in span mode; `Some(feasible)`
    /// when the compare is a span-mode one.
    fn refine_span(&self, st: &mut State, cmp: &Cmp, truth: bool, run: &mut Run) -> Option<bool> {
        let a = single(&cmp.a).cloned()?;
        let b = single(&cmp.b).cloned()?;
        let eq = match cmp.rel {
            Rel::Eq => Some(truth),
            Rel::Lg => Some(!truth),
            _ => None,
        };
        match (&a, &b) {
            // the flags word: == 0 static, == 1 span mode
            (E::Hdr(k), E::Const(c @ (0 | 1))) if eq.is_some() && !cmp.signed => {
                run.span.flags.insert(*k);
                if eq == Some(true) && st.facts & HD != 0 {
                    st.facts |= if *c == 0 { SP0 } else { SP1 };
                }
                Some(true)
            }
            // the span table pointer is nonzero
            (E::Or(p, q), E::Const(0)) if eq.is_some() => match (&**p, &**q) {
                (E::Param(x), E::Param(y)) if *y == *x + 4 => {
                    run.span.spans_arg.insert(*x);
                    if eq == Some(false) && st.facts & SP1 != 0 {
                        st.facts |= SPN;
                    }
                    Some(true)
                }
                _ => None,
            },
            // a lane-0 vector compare mask against zero: the mask is zero on the edge where the
            // compare is false
            (
                E::VCmp {
                    gt,
                    a: x,
                    b: y,
                    reg,
                },
                E::Const(0),
            ) if eq.is_some() => {
                let held = eq == Some(true);
                let rewrite = |st: &mut State, from: &E, to: E| {
                    if st.v.get(reg) == Some(&one(from.clone())) {
                        st.v.insert(*reg, one(to));
                    }
                };
                match (*gt, &**x, &**y) {
                    // list_id < n_spans
                    (false, E::Hdr(l), E::SpanN) => {
                        if held {
                            rewrite(st, x, E::SpanId(*l));
                        }
                    }
                    // count <= K
                    (true, E::SpanCount, E::SpanFld { off: 24, hi: false }) => {
                        if held {
                            rewrite(st, x, E::SpanCountLe);
                        }
                    }
                    // k < count (the other edge is the NOP)
                    (false, E::Hdr(k), E::SpanCountLe) => {
                        if held {
                            rewrite(st, x, E::SpanIdx(*k));
                        } else {
                            st.facts |= SPK;
                        }
                    }
                    // an entry axis against its header grid extent
                    (false, E::Entry(f @ 0..=2), E::Hdr(g)) => {
                        if held && st.facts & SP1 != 0 {
                            st.facts |= [SPX, SPY, SPZ][*f as usize];
                            run.span.grid[*f as usize].insert(*g);
                        }
                    }
                    _ => return None,
                }
                Some(true)
            }
            // the claim bit was clear
            (E::NzMask(h), E::Const(0)) if eq.is_some() => {
                if let E::Tile {
                    st: TileSt::Hit,
                    gx,
                    gy,
                } = &**h
                {
                    if eq == Some(true) && st.facts & SP1 != 0 {
                        st.facts |= SPC;
                        run.span.tile.insert((*gx, *gy));
                    }
                    Some(true)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// One access through a span-table-loaded pointer: attributed to the `spans` argument's
/// descriptor arena at descriptor byte `field` with the A1 memory class `class`.
#[derive(Clone, Debug)]
pub struct SpanSite {
    pub inst: peacemaker_ir::InstId,
    pub pc: Option<u32>,
    pub field: u32,
    pub class: &'static str,
}

/// What the span-mode check proved (the dispatcher side of DYNAMIC.md).
#[derive(Clone, Debug)]
pub struct SpanProof {
    /// The span ABI the object implements (header words, table layout), as derived.
    pub offsets: TaskSpanOffsets,
    /// The accesses through pointers loaded from the span table (live list, count cell,
    /// claim bitmap): the dispatcher shell's accesses that A1 cannot resolve itself.
    pub sites: Vec<SpanSite>,
    pub evidence: Vec<String>,
}

/// The span-mode properties (`span_entry_validated`, `region_wg_from_live_entry`) after the
/// walk: every span check was reached on its path class (the per-operation checks refuse
/// at their own sites; here the walk-wide consistency), or the refusals.
fn span_verdict(
    kernel: &Kernel,
    run: &Run,
    notes: &KernelRegions,
    grid: Option<&TaskGridOffsets>,
    other: [u32; 5],
    stride_shift: u8,
) -> Result<SpanProof, Vec<Refusal>> {
    let sp = &run.span;
    let mut why: Vec<String> = Vec::new();
    let mut one_of = |name: &str, set: &BTreeSet<u32>| -> Option<u32> {
        if set.len() == 1 {
            set.iter().next().copied()
        } else {
            why.push(format!(
                "{name} resolves to {set:?} (exactly one is required)"
            ));
            None
        }
    };
    let flags = one_of("the span flags header word", &sp.flags);
    let k = one_of("the span-relative index k", &sp.k);
    let list = one_of("the span list id", &sp.list);
    let spans = one_of("the span table pointer", &sp.spans_arg);
    let Some(grid) = grid else {
        why.push(
            "region_wg_from_header is not proved, so the header grid fields are not derived (span mode needs them)"
                .into(),
        );
        return Err(why
            .into_iter()
            .map(|w| refuse("task_graph-span-region", w))
            .collect());
    };
    let gx = [grid.gx_offset, grid.gy_offset, grid.gz_offset];
    for (i, axis) in ["x", "y", "z"].iter().enumerate() {
        if sp.grid[i] != BTreeSet::from([gx[i]]) {
            why.push(format!(
                "entry.v{axis} was compared against header dwords {:?}, not the grid extent g{axis}@{}",
                sp.grid[i], gx[i]
            ));
        }
    }
    if sp.tile != BTreeSet::from([(grid.gx_offset, grid.gy_offset)]) {
        why.push(format!(
            "the claim bitmap index was built over grid extents {:?}, not (gx@{}, gy@{})",
            sp.tile, grid.gx_offset, grid.gy_offset
        ));
    }
    for (name, set) in [
        ("count load", &sp.count_sites),
        ("entry load", &sp.entry_sites),
        ("claim bitmap atomic", &sp.claim_sites),
    ] {
        if set.len() != 1 {
            why.push(format!(
                "{} {name} sites reached (exactly one is required)",
                set.len()
            ));
        }
    }
    if sp.table_sites.len() != 3 {
        why.push(format!(
            "{} span table loads reached (n_spans and the two descriptor loads are required)",
            sp.table_sites.len()
        ));
    }
    if !sp.unclassified.is_empty() {
        why.push(format!(
            "regions {:?} are reachable on a path that is neither a static task (flags == 0) nor a span-mode one",
            sp.unclassified
        ));
    }
    if !sp.nop_release {
        why.push("the `k >= count` edge does not reach the done release without a region".into());
    }
    let template_ids: BTreeSet<u32> = notes.regions.iter().map(|r| r.template_id).collect();
    let template_off = other[0];
    let nops: Vec<u32> = sp
        .overwrites
        .iter()
        .filter(|&&(off, c)| off == template_off && c != NONE)
        .map(|&(_, c)| c)
        .collect();
    let nop = match nops.as_slice() {
        [c] if !template_ids.contains(c) => Some(*c),
        other => {
            why.push(format!(
                "the NOP template (a constant over the template register other than {NONE:#x}) is {other:?}: exactly one outside the note's table {template_ids:?} is required"
            ));
            None
        }
    };
    let desc_stride = 1u32 << stride_shift;
    let fields = [other[0], other[1], other[2], other[3], other[4]];
    let known = [
        fields[0],
        fields[1],
        fields[2],
        fields[3],
        fields[4],
        grid.vx_offset,
        grid.vy_offset,
        grid.vz_offset,
        grid.vcount_offset,
        grid.gx_offset,
        grid.gy_offset,
        grid.gz_offset,
    ];
    let words = [flags, k, list];
    for (i, w) in words.iter().enumerate() {
        if let Some(w) = w {
            if known.contains(w)
                || words[..i].contains(&Some(*w))
                || w % 4 != 0
                || w + 4 > desc_stride
            {
                why.push(format!(
                    "span header word @{w} overlaps another header field or lies outside the {desc_stride}-byte header"
                ));
            }
        }
    }
    if !why.is_empty() {
        return Err(why
            .into_iter()
            .map(|w| refuse("task_graph-span-protocol", w))
            .collect());
    }
    let (flags, k, list, spans, nop) = (
        flags.unwrap(),
        k.unwrap(),
        list.unwrap(),
        spans.unwrap(),
        nop.unwrap(),
    );
    let offsets = TaskSpanOffsets {
        flags_offset: flags,
        k_offset: k,
        list_id_offset: list,
        spans_arg_offset: spans,
        n_spans_offset: 0,
        desc_base: SPAN_DESC_BASE as u32,
        desc_stride: 1 << SPAN_DESC_SHIFT,
        live_list_field: 0,
        count_cell_field: 8,
        bitmap_field: 16,
        cap_field: 24,
        entry_stride: 1 << SPAN_ENTRY_SHIFT,
        nop_template: nop,
    };
    let body = &kernel.body;
    let sites: Vec<SpanSite> = sp
        .sites
        .iter()
        .map(|(&pos, &(field, class))| SpanSite {
            inst: body.layout[pos],
            pc: super::rel_pc(
                kernel,
                body.insts.get(body.layout[pos]).and_then(|x| x.prov.pc),
            ),
            field: u32::from(field),
            class,
        })
        .collect();
    let pcs = |set: &BTreeSet<usize>| -> String {
        set.iter()
            .map(|&p| {
                let pc = super::rel_pc(
                    kernel,
                    body.insts.get(body.layout[p]).and_then(|x| x.prov.pc),
                )
                .unwrap_or(0);
                format!("{pc:#x}")
            })
            .collect::<Vec<_>>()
            .join(",")
    };
    let mut evidence = vec![format!(
        "protocol: {TASK_PROTO_SPAN_ENTRY}: span headers (flags@{flags}: 0 static / 1 span, any other value poisons; k@{k}; list id@{list}) read the table at kernarg +{spans}: null table pointer and list_id >= n_spans poison before any descriptor load; descriptor i at +{}+{}*i = {{live_list@0, count_cell@8, bitmap@16, K@24}}; count load (pc {}) of count_cell; count > K poisons, k >= count turns the task into NOP template {nop:#x} (no region, done released; reaches the release without a body); entry load (pc {}) at live_list + {}*k with k < count <= K; entry.vx/vy/vz < header gx@{}/gy@{}/gz@{} else poison; bitmap word bit = vx + gx*(vy + gy*vz) claimed by a returning device-scope OR (pc {}), a set bit poisons; the validated entry replaces vx/vy/vz and vcount := 1 (poison never reaches a region: its template is NONE)",
        SPAN_DESC_BASE,
        1 << SPAN_DESC_SHIFT,
        pcs(&sp.count_sites),
        pcs(&sp.entry_sites),
        1 << SPAN_ENTRY_SHIFT,
        grid.gx_offset,
        grid.gy_offset,
        grid.gz_offset,
        pcs(&sp.claim_sites),
    )];
    evidence.push(format!(
        "protocol: {TASK_PROTO_REGION_WG_LIVE}: in span mode wg_x = entry.vx + step with step < max(1, 1) (one step, vcount forced to 1; entry values carried through the broadcast) and wg_yz = entry.vy | entry.vz << 16, for template(s) {:?}; static headers keep {TASK_PROTO_REGION_WG}; the claim bitmap index is exact only for gx*gy*gz <= 2^32 (the host sizes the bitmap from the grid)",
        sp.live_regions
    ));
    evidence.push(
        "protocol: a span header with pred == NONE poisons before the table is read; the count load is reached only after pred != NONE was observed >= expected and the device-scope acquire (the producer's release of its entries and count happens-before it)"
            .into(),
    );
    Ok(SpanProof {
        offsets,
        sites,
        evidence,
    })
}
