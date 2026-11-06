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
//! forgotten (the range itself is proven unwritten by the region check) and the
//! walk resumes at the region's exit.
//!
//! What it proves, on every path, for the six `TASK_PROTO_*` properties:
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
    TaskProtocolRecord, TASK_PROTO_ACQUIRE, TASK_PROTO_BOUNDED, TASK_PROTO_CLAIM,
    TASK_PROTO_COMPLETE, TASK_PROTO_ONCE, TASK_PROTO_RELEASE,
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

const NONE: u32 = u32::MAX;
const MAX_ALTS: usize = 6;
const MAX_DEPTH: usize = 6;

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
}

impl E {
    fn uniform(&self) -> bool {
        match self {
            E::Const(_) | E::Karg(_) | E::Param(_) | E::Cell(_) | E::Uni => true,
            E::Shl(x, _) => x.uniform(),
            E::PtrLo { add, .. } | E::PtrHi { add, .. } => add.uniform(),
            _ => false,
        }
    }
    fn depth(&self) -> usize {
        match self {
            E::Shl(x, _) => 1 + x.depth(),
            E::EventAt { idx, .. } => 1 + idx.depth(),
            E::PtrLo { add, .. } | E::PtrHi { add, .. } => 1 + add.depth(),
            _ => 0,
        }
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
struct Ins {
    mn: String,
    ops: Vec<String>,
    mods: Vec<String>,
}

impl Ins {
    fn parse(text: &str) -> Ins {
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
    fn offset(&self) -> i64 {
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

fn parse_num(s: &str) -> Option<i64> {
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
enum Rk {
    S,
    V,
    Exec,
    Other,
}

fn parse_reg(s: &str) -> Option<(Rk, u16, u16)> {
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
    nonzero_waits: Vec<u8>,
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
        inst.effects.uses.iter().all(|r| {
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
        match mn {
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
                } else {
                    handled = false;
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
            "global_atomic_or_b32" => self.poison_or(st, pos, run),
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
        // the equality knowledge about `tmpl` dies with a write to its register
        let tmpl = self.notes.dispatcher.tmpl;
        if inst.effects.defs.iter().any(|d| {
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

    fn generic(&self, st: &mut State, inst: &Inst, mem: bool) {
        let uniform = !mem
            && inst.effects.implicit.reads == 0
            && self.inputs_uniform(st, inst)
            && !matches!(inst.effects.control, Control::Barrier(_));
        for d in &inst.effects.defs {
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
                        "a header load that is not a single-lane b128 load after claim < n_tasks",
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
        if st.exec != Exec::Full {
            errs.push("EXEC is not full");
        }
        if st.v.get(&0) != Some(&one(E::Tid)) {
            errs.push("v0 is not the workgroup-linear workitem id");
        }
        // A counter the dispatcher left undrained (the previous task's release increment
        // is still on STOREcnt at the next trip's first region) is harmless to a body
        // that only waits it to zero; a body that waits a counter to a non-zero count
        // needs it drained.
        let must = self.nonzero_waits[idx];
        if st.waited & must != must {
            errs.push("a memory operation of the dispatcher may still be in flight (the body waits that counter to a non-zero count)");
            let detail = format!(
                "region {t}: counters drained {:#06b} (load/store/ds/km), body needs {:#06b}, path facts {:#x}",
                st.waited, must, st.facts
            );
            self.fail(run, pos, "task_graph-protocol-entry", detail);
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
        // apply the summary: the dispatcher's SGPR range survives, nothing else
        let (lo_r, hi_r) = (d.lo, d.hi);
        st.s.retain(|k, _| (lo_r..=hi_r).contains(k));
        st.v.clear();
        st.exec = Exec::Top;
        st.scc = None;
        st.carry = None;
        st.waited = self.exit_waited[idx];
        st.facts |= BODY;
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
                let is_pred = !from_slot && st.facts & ACQ == 0;
                if is_pred {
                    set_once(run, pos, "pred event field", |d| &mut d.pred, k);
                    if eq && st.facts & (CL | BD | HD) == (CL | BD | HD) {
                        st.facts |= PO;
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
    work: Vec<usize>,
    run: Run,
}

impl Fix<'_, '_> {
    fn propagate(&mut self, pos: usize, st: State) {
        let slot = self.states.entry(pos).or_default().entry(st.facts);
        use std::collections::btree_map::Entry;
        let changed = match slot {
            Entry::Vacant(v) => {
                v.insert(st);
                true
            }
            Entry::Occupied(mut o) => {
                let before = o.get().clone();
                o.get_mut().join(&st);
                *o.get() != before
            }
        };
        if changed && !self.work.contains(&pos) {
            self.work.push(pos);
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
        }
        if st.div2 == Some(target) {
            st.facts &= !DIV2;
            st.div2 = None;
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
                Terminator::EndPgm | Terminator::Unreachable => {}
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

    fn solve(&mut self, init: State) {
        self.propagate(0, init);
        while let Some(pos) = self.work.pop() {
            let states: Vec<State> = self
                .states
                .get(&pos)
                .map(|m| m.values().cloned().collect())
                .unwrap_or_default();
            for st in states {
                self.visit(pos, st);
            }
        }
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
}

pub(super) fn check(
    kernel: &Kernel,
    arch: Arch,
    notes: &KernelRegions,
    placed: &[Placed],
    waves: u32,
    shell: &KernargFacts,
    _facts: &[RegionFacts],
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
    let ticket_candidate = atomic_positions.iter().copied().find(|&p| ins[p].returns());
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
    let nonzero_waits: Vec<u8> = placed
        .iter()
        .map(|p| region_nonzero_waits(kernel, p))
        .collect();
    let mut lds_in: Vec<Event> = Vec::new();
    let mut result: Option<Run> = None;
    for _ in 0..5 {
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
            nonzero_waits: nonzero_waits.clone(),
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
        };
        init.s.insert(0, one(E::Karg(0)));
        init.s.insert(1, one(E::Karg(1)));
        init.v.insert(0, one(E::TidEntry));
        let mut fix = Fix {
            w: &w,
            states: BTreeMap::new(),
            work: Vec::new(),
            run: Run::default(),
        };
        fix.solve(init);
        let run = fix.run;
        let produced = events_of(&run.lds_out);
        let stable = produced == lds_in;
        lds_in = produced;
        result = Some(run);
        if stable {
            break;
        }
    }
    let run = result.expect("at least one run");
    if events_of(&run.lds_out) != lds_in {
        out.push(refuse(
            "task_graph-protocol-lds",
            "the LDS slot's contents did not reach a fixpoint",
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
    if run.header_sites.len() != 3 {
        out.push(refuse(
            "task_graph-claim",
            format!(
                "{} header loads reached (the three b128 loads of the 48-byte head are required)",
                run.header_sites.len()
            ),
        ));
    }
    for &p in &atomic_positions {
        if !run.ticket_sites.contains(&p)
            && !run.signal_sites.contains(&p)
            && !run.poison_sites.contains(&p)
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
    let output_args: BTreeSet<u32> = shell
        .args
        .iter()
        .filter(|a| !a.accesses.is_empty() && !roles_set.contains(&a.offset))
        .map(|a| a.offset)
        .collect();
    let mut proved = vec![
        TASK_PROTO_ACQUIRE,
        TASK_PROTO_BOUNDED,
        TASK_PROTO_CLAIM,
        TASK_PROTO_COMPLETE,
        TASK_PROTO_ONCE,
        TASK_PROTO_RELEASE,
    ];
    proved.sort_unstable();
    let record = TaskProtocolRecord {
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
    };
    let mut evidence = vec![format!(
        "protocol: claim fetch-add on +{ticket}, header tasks(+{tasks})[claim<<{stride_shift}] fields tmpl@{template_offset} ka_off@{ka_off_offset} pred@{pred} expected@{expected} done@{done}, events(+{events})[i<<{event_shift}], n_tasks@+{n_tasks}, arena kernargs(+{kernargs}); {} wave(s)",
        waves
    )];
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

/// The counters a region waits to a non-zero count (bit set): it needs those drained
/// at its entry; a counter it only ever waits to zero tolerates an older operation.
fn region_nonzero_waits(kernel: &Kernel, p: &Placed) -> u8 {
    let body = &kernel.body;
    let mut out = 0u8;
    for pos in p.p_entry..p.p_exit.min(body.layout.len()) {
        let Some(w) = body
            .insts
            .get(body.layout[pos])
            .and_then(|i| i.mods.wait.as_ref())
        else {
            continue;
        };
        for (c, bit) in [
            (Counter::Load, WL),
            (Counter::Store, WS),
            (Counter::Ds, WD),
            (Counter::Km, WK),
        ] {
            if w.per_counter[c as usize].is_some_and(|n| n > 0) {
                out |= bit;
            }
        }
        // any other counter (samples, bvh, exports) is not drained by the dispatcher
        for c in [
            Counter::Sample,
            Counter::Bvh,
            Counter::Exp,
            Counter::Vm,
            Counter::Vs,
            Counter::Lgkm,
        ] {
            if w.per_counter[c as usize].is_some_and(|n| n > 0) {
                out |= WL | WS | WD | WK;
            }
        }
    }
    out
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
