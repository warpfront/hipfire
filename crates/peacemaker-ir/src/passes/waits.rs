//! C5: per-counter weighted wait replay across the CFG.
//!
//! CFG-aware port of `hipfire-isa/src/ledger_replay.rs` onto typed streams,
//! implementing the rule table of core.md §2.6:
//!
//! * Issue: one [`PendingEvent`] per memory instruction. Counter membership
//!   comes from `Effects.mem.counters` (the C1 table column); the per-counter
//!   unit weight comes from the same `isa/gfx12.tbl` row (`Km:1` vs `Km:2`).
//! * Wait `s_wait_X n`: retire the oldest events of counter X until at most
//!   `n` units remain. `Km` and `Store` are out of order and retire only at
//!   `n = 0`; gfx11 `Lgkm` retires partially when all outstanding events
//!   belong to the in-order LDS family, but not when SMEM shares the counter.
//!   `Load` retires partially only within one family (buffer vs global, the
//!   `ledger.rs` rule kept from `ledger_replay.rs:45-61`).
//! * Hazard: touching `defs` of a pending load (RAW/WAW), or redefining
//!   `src_locks` of a pending store (WAR), before retirement is a finding.
//!   Store locks are `Effects.uses` (the `ledger.rs` rule); the WAR def set
//!   mirrors `ledger_replay.rs:131-144` exactly (loads plus `v_*`,
//!   `s_mov*`, `s_add*`, `s_sub*` destinations) so the CFG-aware replay
//!   reproduces the linear replay 1:1 on code without joins.
//! * `s_wait_alu depctr_vm_vsrc(0)` clears VMEM store locks
//!   (`ledger_replay.rs:109-114`); every other `s_wait_alu` is replay-neutral.
//! * `s_endpgm` empties the state; barriers never retire.
//! * Join retains the maximal pending set keyed by issuing `InstId`. For
//!   in-order counters, the minimum younger-unit suffix over reaching paths
//!   determines retirement; mutually exclusive issues never pad a queue.
//!   Out-of-order families still require a zero wait.
//!
//! Depctr bit positions below are pinned by `llvm-mc -mcpu=gfx1201`
//! (single-field probes): `sa_sdst` = bit 0, `va_vcc` = bit 1,
//! `vm_vsrc` = bits [4:2], `va_sdst` = bits [11:9], `va_vdst` = bits [15:12].

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use crate::cfg::{BlockId, Body, InstId};
use crate::effects::{Control, MemClass};
use crate::inst::{Arch, Inst};
use crate::operand::{ImmField, Operand};
use crate::reg::{Kind, RegRef, RegSet};
use crate::state::ObligationKind;
use crate::state::Obligation;
use crate::wait::{Counter, CounterSet, EventId, PendingEvent, WaitFact, WaitState, N};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum WaitError {
    #[error("wait replay needs a gfx11 or gfx12 opcode table (got {0:?})")]
    UnsupportedArch(Arch),
    #[error("layout refers to a tombstoned or missing instruction")]
    DanglingInst { id: InstId },
    #[error("wait replay did not reach a fixpoint within {walks} block walks")]
    NoFixpoint { walks: usize },
}

/// Census of wait instructions by opcode, for T6.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WaitCensus {
    /// Immediates of every `s_wait_kmcnt`, in layout order (KT48: seven 0x0).
    pub kmcnt: Vec<u8>,
    pub loadcnt: usize,
    pub dscnt: usize,
    pub loadcnt_dscnt: usize,
    pub storecnt: usize,
    pub alu: usize,
}

/// Full result of the replay: issued events (with unit weights), per-wait
/// facts, per-site obligations, and the converged state before each inst.
#[derive(Clone, Debug)]
pub struct WaitReplay {
    /// One entry per reachable memory instruction, with stable layout-order IDs.
    pub events: Vec<PendingEvent>,
    pub facts: Vec<WaitFact>,
    pub obligations: Vec<Obligation>,
    /// Converged `WaitState` before each instruction (final pass).
    pub before: HashMap<InstId, WaitState>,
}

fn mem_name(class: MemClass) -> &'static str {
    match class {
        MemClass::VmemLoad => "vmem-load",
        MemClass::VmemStore => "vmem-store",
        MemClass::VmemAtomic { .. } => "vmem-atomic",
        MemClass::DsLoad => "ds-load",
        MemClass::DsStore => "ds-store",
        MemClass::DsAtomic { .. } => "ds-atomic",
        MemClass::SmemLoad => "smem-load",
        MemClass::SmemStore => "smem-store",
        MemClass::Export => "export",
        MemClass::LdsDma => "lds-dma",
        MemClass::FlatLoad => "flat-load",
        MemClass::FlatStore => "flat-store",
        MemClass::FlatAtomic { .. } => "flat-atomic",
    }
}

fn is_load(class: MemClass) -> bool {
    matches!(
        class,
        MemClass::VmemLoad
            | MemClass::DsLoad
            | MemClass::SmemLoad
            | MemClass::FlatLoad
            | MemClass::VmemAtomic { returns: true }
            | MemClass::DsAtomic { returns: true }
            | MemClass::FlatAtomic { returns: true }
    )
}

/// Memory classes the replay tracks. Image/sample rows and flat atomics have
/// no table rows; anything else decodable is replayed.
fn is_tracked(class: MemClass) -> bool {
    matches!(
        class,
        MemClass::VmemLoad
            | MemClass::VmemStore
            | MemClass::VmemAtomic { .. }
            | MemClass::DsLoad
            | MemClass::DsStore
            | MemClass::SmemLoad
            | MemClass::SmemStore
            | MemClass::FlatLoad
            | MemClass::FlatStore
    )
}

/// Load-counter family for the in-order rule: buffer vs everything else,
/// from the encoding form (the `ledger.rs` family rule).
fn family_of(inst: &Inst) -> &'static str {
    match inst.form {
        crate::inst::Form::Vmem(crate::inst::VmemForm::Buffer) => "buffer",
        crate::inst::Form::Vmem(_) => "global",
        crate::inst::Form::Ds => "ds",
        crate::inst::Form::Smem => "smem",
        _ => "scalar",
    }
}

fn opcode_name(inst: &Inst, arch: Arch) -> Option<&'static str> {
    inst.op.name(arch)
}

/// WAR def set mirroring `ledger_replay.rs:131-144`: loads plus the ALU
/// prefixes the text replay credits with destinations.
fn war_defs(inst: &Inst, arch: Arch) -> Vec<RegRef> {
    if inst.effects.mem.is_some_and(|mem| is_load(mem.class)) {
        return inst.effects.defs.iter().copied().collect();
    }
    let name = opcode_name(inst, arch).unwrap_or("");
    if name.starts_with("v_")
        || name.starts_with("s_mov")
        || name.starts_with("s_add")
        || name.starts_with("s_sub")
    {
        inst.effects.defs.iter().copied().collect()
    } else {
        Vec::new()
    }
}

fn dwords(regs: &[RegRef]) -> Vec<(Kind, u16)> {
    let mut out = Vec::new();
    for reg in regs {
        for i in 0..reg.len {
            out.push((reg.kind, reg.base + u16::from(i)));
        }
    }
    out
}

fn regset_dwords(set: &RegSet) -> Vec<(Kind, u16)> {
    dwords(&set.0)
}

fn overlaps(a: &[(Kind, u16)], b: &[(Kind, u16)]) -> bool {
    a.iter().any(|x| b.contains(x))
}

fn explicit_regs(inst: &Inst) -> Vec<RegRef> {
    let mut out: Vec<RegRef> =
        inst.effects.defs.iter().chain(inst.effects.uses.iter()).copied().collect();
    for operand in &inst.operands {
        match operand {
            Operand::Reg(reg) => out.push(*reg),
            Operand::Half(reg, _) => out.push(*reg),
            _ => {}
        }
    }
    out
}

/// Raw SOPP simm16 bits of an `s_wait_alu`, if present.
fn wait_alu_bits(inst: &Inst) -> Option<u16> {
    inst.operands.iter().find_map(|operand| match operand {
        Operand::Imm(ImmField::Sopp(n) | ImmField::Sopk(n)) => Some(*n as u16),
        _ => None,
    })
}

fn is_wait_alu(inst: &Inst, arch: Arch) -> bool {
    opcode_name(inst, arch) == Some("s_wait_alu")
}

fn counter_at(index: usize) -> Counter {
    match index {
        0 => Counter::Load,
        1 => Counter::Store,
        2 => Counter::Ds,
        3 => Counter::Km,
        4 => Counter::Sample,
        5 => Counter::Bvh,
        6 => Counter::Exp,
        7 => Counter::Vm,
        8 => Counter::Vs,
        _ => Counter::Lgkm,
    }
}

/// Unit weight per counter from the table row (`Km:2` → 2 on Km), for the
/// instruction's return form (`@rtn` / `@nortn` rules).
fn unit_weights(inst: &Inst, arch: Arch) -> [u8; N] {
    let mut units = [0u8; N];
    let Some(row) = crate::isa::lookup(arch, inst.op, inst.form) else {
        return units;
    };
    for (counter, weight) in row.counter_rules(crate::isa::atomic_returns(arch, &inst.mods.cpol)) {
        let slot = match counter {
            "Load" => Counter::Load,
            "Store" => Counter::Store,
            "Ds" => Counter::Ds,
            "Km" => Counter::Km,
            "Sample" => Counter::Sample,
            "Bvh" => Counter::Bvh,
            "Exp" => Counter::Exp,
            "Vm" => Counter::Vm,
            "Vs" => Counter::Vs,
            "Lgkm" => Counter::Lgkm,
            _ => continue,
        };
        units[slot as usize] = weight.parse().unwrap_or(1);
    }
    units
}

fn join_states(a: &WaitState, b: &WaitState) -> WaitState {
    let mut merged: HashMap<EventId, PendingEvent> = HashMap::new();
    for event in a.pending.iter().chain(&b.pending) {
        match merged.get_mut(&event.id) {
            None => {
                merged.insert(event.id, event.clone());
            }
            Some(prior) => {
                prior.counters = prior.counters.union(event.counters);
                for i in 0..N {
                    prior.units[i] = prior.units[i].max(event.units[i]);
                    prior.younger[i] = prior.younger[i].min(event.younger[i]);
                }
                prior.satisfied = prior.satisfied.intersection(event.satisfied);
                let mut defs = prior.defs.0.clone();
                for reg in &event.defs.0 {
                    if !defs.contains(reg) {
                        defs.push(*reg);
                    }
                }
                prior.defs = RegSet(defs);
                let mut locks = prior.src_locks.0.clone();
                for reg in &event.src_locks.0 {
                    if !locks.contains(reg) {
                        locks.push(*reg);
                    }
                }
                prior.src_locks = RegSet(locks);
            }
        }
    }
    // Presentation order is deterministic; retirement uses path-local suffix
    // bounds, never this first-issue order. Union size is not a queue length.
    let mut pending: Vec<PendingEvent> = merged.into_values().collect();
    pending.sort_by_key(|event| event.id.0);
    WaitState { pending }
}

struct Recorder {
    facts: Vec<WaitFact>,
    obligations: Vec<Obligation>,
    before: HashMap<InstId, WaitState>,
}

struct EventTable {
    ids: HashMap<InstId, EventId>,
    events: Vec<PendingEvent>,
    next: u64,
}

impl EventTable {
    fn id_of(&mut self, inst: InstId) -> EventId {
        if let Some(id) = self.ids.get(&inst) {
            return *id;
        }
        let id = EventId(self.next);
        self.next += 1;
        self.ids.insert(inst, id);
        id
    }
}

fn retire(
    state: &mut WaitState,
    body: &Body,
    counter: Counter,
    count: u8,
) -> Vec<EventId> {
    if count != 0 {
        if matches!(counter, Counter::Km | Counter::Store | Counter::Vs) {
            return Vec::new();
        }
        // LLVM SIInsertWaitcnts.cpp::counterOutOfOrder: pre-gfx12 LGKM
        // includes LDS and SMEM. SMEM reads can retire out of order; LDS
        // accesses (including atomics) complete in issue order (RDNA3 ISA
        // §9: FLAT LGKM path is in-order with DS). A positive lgkmcnt can
        // therefore retire the oldest LDS only if no SMEM/GDS/message event
        // remains on any reaching path (the joined pending set is maximal).
        if counter == Counter::Lgkm && state.pending.iter().any(|event| {
            event.counters.contains(counter) && !event.satisfied.contains(counter)
                && !matches!(event.class,
                    MemClass::DsLoad | MemClass::DsStore | MemClass::DsAtomic { .. })
        }) {
            return Vec::new();
        }
        if counter == Counter::Load {
            let mut families = HashSet::new();
            for event in state.pending.iter().filter(|e| e.counters.contains(counter)) {
                let inst = body.insts.get(event.inst).expect("replay event names a live inst");
                families.insert(family_of(inst));
            }
            if families.len() > 1 {
                return Vec::new();
            }
        }
    }
    let mut retired = Vec::new();
    for position in (0..state.pending.len()).rev() {
        let event = &state.pending[position];
        if !event.counters.contains(counter) || event.satisfied.contains(counter)
            || (count != 0 && event.younger[counter as usize] < count) {
            continue;
        }
        let event = &mut state.pending[position];
        event.satisfied.insert(counter);
        if event.satisfied == event.counters {
            retired.push(event.id);
            state.pending.remove(position);
        }
    }
    retired
}

fn reg_name((kind, n): (Kind, u16)) -> String {
    let prefix = match kind {
        Kind::V => "v",
        Kind::S => "s",
        Kind::Ttmp => "ttmp",
    };
    format!("{prefix}{n}")
}

/// One obligation per consuming instruction (the linear replay's unit):
/// a store-source WAR wins over a load-destination RAW when both hit.
fn record_finding(
    recorder: &mut Recorder,
    consumer: InstId,
    war_hits: &[PendingEvent],
    raw_hits: &[PendingEvent],
    consumer_touch: &[(Kind, u16)],
    consumer_war: &[(Kind, u16)],
) {
    if !war_hits.is_empty() {
        let first = &war_hits[0];
        let mut blamed: Vec<String> = Vec::new();
        for event in war_hits {
            for dword in regset_dwords(&event.src_locks) {
                if consumer_war.contains(&dword) && !blamed.contains(&reg_name(dword)) {
                    blamed.push(reg_name(dword));
                }
            }
        }
        let rule = match first.class {
            MemClass::DsStore => "wait-war-ds-store",
            MemClass::SmemStore => "wait-war-smem-store",
            _ => "wait-war-vmem-store",
        };
        recorder.obligations.push(Obligation {
            kind: ObligationKind::SrcReadTiming(first.class),
            insts: vec![consumer],
            rule_id: rule.into(),
            text: format!(
                "redefines {} source ({}) before {} retires",
                mem_name(first.class),
                blamed.join(","),
                match first.class {
                    MemClass::DsStore => "DScnt",
                    MemClass::SmemStore => "KMcnt",
                    _ => "STOREcnt",
                }
            ),
        });
        return;
    }
    if !raw_hits.is_empty() {
        let first = &raw_hits[0];
        let mut blamed: Vec<String> = Vec::new();
        for event in raw_hits {
            for dword in regset_dwords(&event.defs) {
                if consumer_touch.contains(&dword) && !blamed.contains(&reg_name(dword)) {
                    blamed.push(reg_name(dword));
                }
            }
        }
        recorder.obligations.push(Obligation {
            kind: ObligationKind::Hazard,
            insts: vec![consumer],
            rule_id: format!("wait-raw-{}", mem_name(first.class)),
            text: format!(
                "touches pending {} destination ({}) before it retires",
                mem_name(first.class),
                blamed.join(",")
            ),
        });
    }
}

fn inst_of(body: &Body, id: InstId) -> Result<&Inst, WaitError> {
    body.insts.get(id).ok_or(WaitError::DanglingInst { id })
}

/// Exits of one block walk. C4 blocks can hold several conditional
/// branches (conditional fall-through is not a leader), so the
/// fall-through exit alone is path-ambiguous: every mid-block conditional
/// branch contributes its own exit state, taken at the branch point.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BlockExits {
    fall: Option<WaitState>,
    branches: Vec<(BlockId, WaitState)>,
}

fn branch_target(inst: &Inst) -> Option<BlockId> {
    inst.operands.iter().find_map(|operand| match operand {
        Operand::Label(id) => Some(*id),
        _ => None,
    })
}

fn push_event(
    arch: Arch,
    state: &mut WaitState,
    events: &mut EventTable,
    id: InstId,
    inst: &Inst,
    mem: &crate::effects::MemEffect,
    record_issued: bool,
) {
    let id_event = events.id_of(id);
    let (defs, locks) = if is_load(mem.class) {
        (RegSet(inst.effects.defs.iter().copied().collect()), RegSet(Vec::new()))
    } else {
        (RegSet(Vec::new()), RegSet(inst.effects.uses.iter().copied().collect()))
    };
    let fresh = PendingEvent {
        id: id_event,
        inst: id,
        counters: mem.counters,
        units: unit_weights(inst, arch),
        younger: [0; N],
        satisfied: CounterSet::default(),
        class: mem.class,
        defs,
        src_locks: locks,
        in_order_type: mem.in_order_type,
    };
    // Reissue belongs at the tail, including across a loop back-edge.
    // A still-pending previous issue is diagnosed above. It still consumes
    // hardware counter units: replacing its static-site record must not
    // subtract those units from other events' proven suffix bounds.
    if let Some(position) = state.pending.iter().position(|e| e.id == id_event) {
        state.pending.remove(position);
    }
    for event in &mut state.pending {
        for i in 0..N {
            let counter = counter_at(i);
            // Out-of-order families only retire at zero. Their suffix is
            // irrelevant; aging it would force useless loop walks to 255.
            if !matches!(counter, Counter::Km | Counter::Store | Counter::Vs)
                && event.counters.contains(counter) {
                event.younger[i] = event.younger[i].saturating_add(fresh.units[i]);
            }
        }
    }
    state.pending.push(fresh.clone());
    if record_issued && events.events.iter().all(|e| e.id != id_event) {
        events.events.push(fresh);
    }
}

/// Walk one block-range from `entry`. With `recorder` set, snapshots the
/// state before each instruction and records facts plus one obligation per
/// consuming instruction; otherwise a dry fixpoint step.
fn walk_block(
    body: &Body,
    arch: Arch,
    range: (usize, usize),
    entry: &WaitState,
    events: &mut EventTable,
    choices: &HashMap<InstId, bool>,
    mut recorder: Option<&mut Recorder>,
) -> Result<BlockExits, WaitError> {
    let mut state = entry.clone();
    let mut branches = Vec::new();
    for index in range.0..range.1 {
        let id = body.layout[index];
        let inst = inst_of(body, id)?;
        if let Some(recorder) = recorder.as_mut() {
            match recorder.before.entry(id) {
                std::collections::hash_map::Entry::Vacant(slot) => { slot.insert(state.clone()); }
                std::collections::hash_map::Entry::Occupied(mut slot) => {
                    let joined = join_states(slot.get(), &state);
                    slot.insert(joined);
                }
            }
        }
        match inst.effects.control {
            Control::EndPgm | Control::Halt | Control::Trap => {
                return Ok(BlockExits { fall: None, branches });
            }
            Control::Wait => {
                if let Some(wait) = &inst.mods.wait {
                    let mut satisfied = HashSet::new();
                    for (counter_index, count) in wait.per_counter.iter().enumerate() {
                        if let Some(n) = count {
                            for retired in retire(&mut state, body, counter_at(counter_index), *n) {
                                satisfied.insert(retired);
                            }
                        }
                    }
                    if let Some(recorder) = recorder.as_mut() {
                        let mut list: Vec<EventId> = satisfied.into_iter().collect();
                        list.sort_by_key(|event| event.0);
                        recorder.facts.push(WaitFact { wait: id, satisfies: list });
                    }
                } else if is_wait_alu(inst, arch)
                    && wait_alu_bits(inst).is_some_and(|bits| bits >> 2 & 7 == 0)
                {
                    for event in state.pending.iter_mut() {
                        if event.counters.contains(Counter::Store) {
                            event.src_locks.0.clear();
                        }
                    }
                }
                continue;
            }
            Control::Jump => {
                if let Some(target) = branch_target(inst) {
                    branches.push((target, state.clone()));
                }
                return Ok(BlockExits { fall: None, branches });
            }
            Control::Branch { .. } => {
                if choices.get(&id) != Some(&false) {
                    if let Some(target) = branch_target(inst) {
                        branches.push((target, state.clone()));
                    }
                }
                if choices.get(&id) == Some(&true) {
                    return Ok(BlockExits { fall: None, branches });
                }
            }
            _ => {}
        }
        if let Some(recorder) = recorder.as_mut() {
            let touch = dwords(&explicit_regs(inst));
            let war = dwords(&war_defs(inst, arch));
            let mut war_hits: Vec<PendingEvent> = Vec::new();
            let mut raw_hits: Vec<PendingEvent> = Vec::new();
            for event in &state.pending {
                if overlaps(&regset_dwords(&event.src_locks), &war) {
                    war_hits.push(event.clone());
                } else if overlaps(&regset_dwords(&event.defs), &touch) {
                    raw_hits.push(event.clone());
                }
            }
            if !war_hits.is_empty() || !raw_hits.is_empty() {
                record_finding(recorder, id, &war_hits, &raw_hits, &touch, &war);
            }
        }
        if let Some(mem) = &inst.effects.mem {
            if is_tracked(mem.class)
                || (matches!(arch, Arch::Gfx1100 | Arch::Gfx1151)
                    && matches!(mem.class, MemClass::DsAtomic { .. }))
            {
                push_event(arch, &mut state, events, id, inst, mem, recorder.is_some());
            }
        }
    }
    Ok(BlockExits { fall: Some(state), branches })
}


/// CFG-aware wait replay with fixpoint over the block graph.
pub fn replay(body: &Body, arch: Arch) -> Result<WaitReplay, WaitError> {
    if !matches!(arch, Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1201) {
        return Err(WaitError::UnsupportedArch(arch));
    }
    for id in &body.layout {
        if body.insts.get(*id).is_none() {
            return Err(WaitError::DanglingInst { id: *id });
        }
    }
    let mut events = EventTable { ids: HashMap::new(), events: Vec::new(), next: 0 };
    for id in &body.layout {
        if body.insts.get(*id).unwrap().effects.mem.is_some() { events.id_of(*id); }
    }
    let mut recorder = Recorder {
        facts: Vec::new(),
        obligations: Vec::new(),
        before: HashMap::new(),
    };
    let ranges: Vec<(BlockId, (usize, usize))> = if body.blocks.is_empty() {
        vec![(BlockId(0), (0, body.layout.len()))]
    } else { body.blocks.iter().map(|block| (block.id, block.range)).collect() };
    // A successful conservative replay already proves every path. Correlating
    // immutable guards is only necessary to discharge remaining hazards.
    let mut partitions = vec![HashMap::new()];
    let mut refined = false;
    loop {
        for choices in &partitions {
            let entries = fixpoint(body, arch, &ranges, choices, &mut events)?;
            for ((_, range), entry) in ranges.iter().zip(&entries) {
                let Some(entry) = entry else { continue };
                walk_block(body, arch, *range, entry, &mut events, choices, Some(&mut recorder))?;
            }
        }
        if refined || recorder.obligations.is_empty() { break; }
        partitions = super::predicates::partitions(body, arch);
        if partitions.len() == 1 && partitions[0].is_empty() { break; }
        recorder.facts.clear();
        recorder.obligations.clear();
        recorder.before.clear();
        refined = true;
    }
    recorder.obligations.sort_unstable_by(|a, b| a.insts.iter().map(|id| id.0).cmp(b.insts.iter().map(|id| id.0))
        .then_with(|| a.rule_id.cmp(&b.rule_id)).then_with(|| a.text.cmp(&b.text)));
    recorder.obligations.dedup();
    let mut facts: Vec<WaitFact> = Vec::new();
    for fact in recorder.facts {
        if let Some(prior) = facts.iter_mut().find(|f| f.wait == fact.wait) {
            for event in fact.satisfies { if !prior.satisfies.contains(&event) { prior.satisfies.push(event); } }
        } else { facts.push(fact); }
    }
    Ok(WaitReplay {
        events: events.events,
        facts,
        obligations: recorder.obligations,
        before: recorder.before,
    })
}

/// Reachable block-entry fixpoint. Transfer captures mid-block branch exits
/// at their instruction positions, including partition-selected edges.
/// Keeping straight-line transfer in one walk avoids retaining and cloning
/// the entire pending state at each instruction during convergence.
fn fixpoint(body: &Body, arch: Arch, ranges: &[(BlockId, (usize, usize))], choices: &HashMap<InstId, bool>, events: &mut EventTable) -> Result<Vec<Option<WaitState>>, WaitError> {
    let count = ranges.len();
    let index: HashMap<BlockId, usize> = ranges.iter().enumerate().map(|(p, (id, _))| (*id, p)).collect();
    let first_at: HashMap<usize, usize> = ranges.iter().enumerate().map(|(p, (_, range))| (range.0, p)).collect();
    let mut incoming: Vec<Vec<(usize, Option<usize>)>> = vec![Vec::new(); count];
    let mut succs = vec![Vec::new(); count];
    for (p, (_, range)) in ranges.iter().enumerate() {
        let shape = walk_block(body, arch, *range, &WaitState::default(), events, choices, None)?;
        if shape.fall.is_some() {
            if let Some(&q) = first_at.get(&range.1).filter(|&&q| q != p) {
                incoming[q].push((p, None));
                succs[p].push(q);
            }
        }
        for (k, (target, _)) in shape.branches.iter().enumerate() {
            if let Some(&q) = index.get(target) {
                incoming[q].push((p, Some(k)));
                if !succs[p].contains(&q) { succs[p].push(q); }
            }
        }
    }
    let mut entries: Vec<Option<WaitState>> = vec![None; count];
    let mut exits: Vec<Option<BlockExits>> = vec![None; count];
    let mut work: std::collections::BTreeSet<usize> = (0..count.min(1)).collect();
    let budget = body.layout.len().saturating_mul(1000).max(1000);
    let mut walks = 0;
    while let Some(p) = work.pop_first() {
        let mut joined = (p == 0).then(WaitState::default);
        for &(q, edge) in &incoming[p] {
            let Some(exit) = &exits[q] else { continue };
            let state = match edge { None => exit.fall.as_ref(), Some(k) => Some(&exit.branches[k].1) };
            if let Some(state) = state {
                joined = Some(match joined { None => state.clone(), Some(prior) => join_states(&prior, state) });
            }
        }
        if joined == entries[p] { continue; }
        walks += 1;
        if walks > budget { return Err(WaitError::NoFixpoint { walks }); }
        let exit = match &joined {
            None => None,
            Some(state) => Some(walk_block(body, arch, ranges[p].1, state, events, choices, None)?),
        };
        entries[p] = joined;
        if exits[p] != exit { exits[p] = exit; work.extend(succs[p].iter().copied()); }
    }
    Ok(entries)
}

/// Count wait instructions by opcode for T6.
pub fn census(body: &Body, arch: Arch) -> Result<WaitCensus, WaitError> {
    if !matches!(arch, Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1201) {
        return Err(WaitError::UnsupportedArch(arch));
    }
    let mut out = WaitCensus::default();
    for id in &body.layout {
        let inst = inst_of(body, *id)?;
        let name = opcode_name(inst, arch).unwrap_or("");
        match name {
            "s_wait_kmcnt" => {
                let count = inst
                    .mods
                    .wait
                    .as_ref()
                    .and_then(|wait| wait.per_counter[Counter::Km as usize])
                    .unwrap_or(0);
                out.kmcnt.push(count);
            }
            "s_wait_loadcnt" => out.loadcnt += 1,
            "s_wait_dscnt" => out.dscnt += 1,
            "s_wait_loadcnt_dscnt" | "s_wait_storecnt_dscnt" => out.loadcnt_dscnt += 1,
            "s_wait_storecnt" => out.storecnt += 1,
            "s_wait_alu" => out.alu += 1,
            _ => {}
        }
    }
    Ok(out)
}
// C5 wait-replay tests: typed re-expressions of the linear
// `ledger_replay.rs:161-230` suite plus the KT48 T6 gate.

#[cfg(test)]
mod c5_tests {
    use smallvec::SmallVec;

    use crate::cfg::Body;
    use crate::effects::MemClass;
    use crate::inst::{Arch, Form, FormFields, Inst, Opcode};
    use crate::operand::{ImmField, Modifiers, Operand, VmemToken};
    use crate::passes::cfg::build_blocks;
    use crate::provenance::Provenance;
    use crate::reg::{Kind, RegRef};
    use crate::state::ObligationKind;
    use crate::wait::Counter;

    use super::{census, replay};

    fn table(name: &str) -> (Opcode, Form) {
        let row = crate::isa::gfx12().iter().find(|row| row.name == name).expect(name);
        (row.op, row.form)
    }

    fn mk(name: &str, operands: Vec<Operand>, mods: Modifiers) -> Inst {
        let (op, form) = table(name);
        Inst::from_parts(
            Arch::Gfx1201,
            op,
            form,
            FormFields::None,
            SmallVec::from_vec(operands),
            mods,
            None,
            Provenance::default(),
        )
        .unwrap_or_else(|error| panic!("{name}: {error:?}"))
    }

    fn mi(name: &str, operands: Vec<Operand>) -> Inst {
        mk(name, operands, Modifiers::default())
    }
    fn v(base: u16, len: u8) -> Operand {
        Operand::Reg(RegRef { kind: Kind::V, base, len })
    }

    fn s(base: u16, len: u8) -> Operand {
        Operand::Reg(RegRef { kind: Kind::S, base, len })
    }

    fn wait_mods(load: Option<u8>, ds: Option<u8>, km: Option<u8>, store: Option<u8>) -> Modifiers {
        let mut wait = crate::wait::WaitImm::default();
        wait.per_counter[Counter::Load as usize] = load;
        wait.per_counter[Counter::Ds as usize] = ds;
        wait.per_counter[Counter::Km as usize] = km;
        wait.per_counter[Counter::Store as usize] = store;
        Modifiers { wait: Some(wait), ..Modifiers::default() }
    }

    fn sopp(n: i16) -> Operand {
        Operand::Imm(ImmField::Sopp(n))
    }

    fn body_of(insts: Vec<Inst>) -> Body {
        let mut body = Body::default();
        for inst in insts {
            let id = body.insts.insert(inst);
            body.layout.push(id);
        }
        body
    }

    fn buffer_load(dst: u16, addr: u16) -> Inst {
        mi("buffer_load_b32",
            vec![v(dst, 1), v(addr, 2), s(4, 4), s(9, 1), Operand::Vmem(VmemToken::Offen)],
        )
    }

    fn vadd(dst: u16, a: u16, b: u16) -> Inst {
        mi("v_add_f32_e32", vec![v(dst, 1), v(a, 1), v(b, 1)])
    }

    fn ok(body: &Body) -> bool {
        replay(body, Arch::Gfx1201).unwrap().obligations.is_empty()
    }

    /// On gfx11 a positive lgkmcnt retires the oldest LDS access when no
    /// SMEM shares LGKM. With mixed outstanding SMEM the same wait cannot
    /// establish which earlier DS load has completed.
    #[test]
    fn gfx11_lgkm_partial_wait_respects_lds_order_and_smem_mixing() {
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            let inst = |name: &str, operands: Vec<Operand>| {
                let row = crate::isa::table(arch).iter().find(|row| row.name == name).expect(name);
                Inst::from_parts(
                    arch, row.op, row.form, FormFields::None,
                    SmallVec::from_vec(operands), Modifiers::default(),
                    None, Provenance::default(),
                ).unwrap_or_else(|error| panic!("{arch:?} {name}: {error:?}"))
            };
            let ds = |dst| inst("ds_load_b32", vec![
                v(dst, 1), v(9, 1), Operand::Imm(ImmField::DsOffset(0)),
            ]);
            let wait = crate::codec::gfx11::decode(arch, &[0xbf89_0432]).unwrap().0;
            assert_eq!(wait.mods.wait.as_ref().unwrap().per_counter[Counter::Lgkm as usize], Some(3));
            let consumer = || inst("v_add_f32_e32", vec![v(20, 1), v(1, 1), v(21, 1)]);
            let ds_only = body_of(vec![ds(1), ds(2), ds(3), ds(4), wait.clone(), consumer()]);
            assert!(replay(&ds_only, arch).unwrap().obligations.is_empty(), "{arch:?}: oldest LDS must retire");
            let ds_later = body_of(vec![ds(1), ds(2), ds(3), ds(4), wait.clone(),
                inst("v_add_f32_e32", vec![v(20, 1), v(2, 1), v(21, 1)])]);
            assert!(replay(&ds_later, arch).unwrap().obligations.iter()
                .any(|obligation| obligation.rule_id == "wait-raw-ds-load"));
            let atomic = inst("ds_min_i32", vec![
                v(30, 1), v(31, 1), Operand::Imm(ImmField::DsOffset(0)),
            ]);
            let atomics_in_order = body_of(vec![atomic, ds(1), ds(2), ds(3), wait.clone(),
                inst("v_mov_b32_e32", vec![v(31, 1), v(21, 1)])]);
            assert!(replay(&atomics_in_order, arch).unwrap().obligations.is_empty(),
                "{arch:?}: oldest LDS atomic has retired before reusing its data source");
            let smem = inst("s_load_b32", vec![
                s(20, 1), s(0, 2), Operand::Imm(ImmField::SmemOffset(0)),
            ]);
            let mixed = body_of(vec![ds(1), smem, ds(2), ds(3), ds(4), wait, consumer()]);
            assert!(replay(&mixed, arch).unwrap().obligations.iter()
                .any(|obligation| obligation.rule_id == "wait-raw-ds-load"),
                "{arch:?}: SMEM makes partial lgkmcnt unable to prove LDS retirement");
        }
    }

    /// gfx11 GLOBAL atomics: the returning (GLC) form's destination is
    /// pending on vmcnt; the non-returning form locks its sources on vscnt,
    /// which `vmcnt(0)` does not retire.
    #[test]
    fn gfx11_global_atomic_counts_on_vmcnt_only_when_returning() {
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            let dec = |words: &[u32]| crate::codec::gfx11::decode(arch, words).unwrap().0;
            let rules = |insts: Vec<Inst>| replay(&body_of(insts), arch).unwrap().obligations
                .into_iter().map(|o| o.rule_id).collect::<Vec<_>>();
            let add_rtn = || dec(&[0xdcd6_4000, 0x0204_020a]); // global_atomic_add_u32 v2, v10, v2, s[4:5] glc
            let swap = || dec(&[0xdcce_0000, 0x0000_0000]); // global_atomic_swap_b32 v0, v0, s[0:1]
            let vmcnt0 = || dec(&[0xbf89_03f7]);
            let vscnt0 = || dec(&[0xbc7c_0000]);
            let read_v2 = || dec(&[0x7e04_0502]); // v_readfirstlane_b32 s2, v2
            let write_v0 = || dec(&[0x7e00_0280]); // v_mov_b32_e32 v0, 0
            assert_eq!(rules(vec![add_rtn(), read_v2()]), ["wait-raw-vmem-atomic"], "{arch:?}");
            assert!(rules(vec![add_rtn(), vmcnt0(), read_v2()]).is_empty(), "{arch:?}");
            assert_eq!(rules(vec![swap(), vmcnt0(), write_v0()]), ["wait-war-vmem-store"], "{arch:?}");
            assert!(rules(vec![swap(), vscnt0(), write_v0()]).is_empty(), "{arch:?}");
        }
    }

    /// Linear `load_wait_uses_only_retired_destinations`, typed.
    #[test]
    fn load_wait_retires_oldest_first() {
        let legal = body_of(vec![
            buffer_load(0, 8),
            buffer_load(1, 8),
            mk("s_wait_loadcnt", vec![sopp(1)], wait_mods(Some(1), None, None, None)),
            vadd(2, 0, 3),
        ]);
        assert!(ok(&legal));
        let tight = body_of(vec![
            buffer_load(0, 8),
            buffer_load(1, 8),
            mk("s_wait_loadcnt", vec![sopp(2)], wait_mods(Some(2), None, None, None)),
            vadd(2, 0, 3),
        ]);
        assert!(!ok(&tight));
    }

    /// Linear `combined_wait_keeps_young_vmem_pending`, typed.
    #[test]
    fn combined_wait_keeps_young_vmem_pending() {
        let stream = body_of(vec![
            buffer_load(0, 8),
            mi("ds_load_b32", vec![v(1, 1), v(9, 1), Operand::Imm(ImmField::DsOffset(0))]),
            mk(
                "s_wait_loadcnt_dscnt",
                vec![sopp(0x100)],
                wait_mods(Some(1), Some(0), None, None),
            ),
            vadd(2, 1, 3),
        ]);
        assert!(ok(&stream));
        let mut late = stream;
        let tail = mi("v_add_f32_e32", vec![v(2, 1), v(0, 1), v(3, 1)]);
        let id = late.insts.insert(tail);
        late.layout.push(id);
        assert!(!ok(&late));
    }

    /// Linear `asymmetric_combined_wait_decodes_load_then_ds`, typed.
    #[test]
    fn asymmetric_combined_wait() {
        let mut insts = Vec::new();
        for i in 0..8 {
            insts.push(buffer_load(i, 20));
        }
        for i in 0..4 {
            insts.push(mi("ds_load_b32",
                vec![v(8 + i, 1), v(21, 1), Operand::Imm(ImmField::DsOffset(0))],
            ));
        }
        insts.push(mk(
            "s_wait_loadcnt_dscnt",
            vec![sopp(0x703)],
            wait_mods(Some(7), Some(3), None, None),
        ));
        insts.push(vadd(22, 0, 8));
        assert!(ok(&body_of(insts.clone())));
        // LOADcnt 3 leaves v0..v4 pending: use of v0 faults.
        let mut weak = insts.clone();
        let at = weak.len() - 2;
        weak[at] = mk(
            "s_wait_loadcnt_dscnt",
            vec![sopp(0x307)],
            wait_mods(Some(3), Some(7), None, None),
        );
        assert!(!ok(&body_of(weak)));
        // DScnt 7 retires nothing: use of v8 faults.
        let mut weak_ds = insts.clone();
        let at = weak_ds.len() - 2;
        weak_ds[at] = mk(
            "s_wait_loadcnt_dscnt",
            vec![sopp(0x707)],
            wait_mods(Some(7), Some(7), None, None),
        );
        assert!(!ok(&body_of(weak_ds)));
    }

    /// Linear `store_lock_requires_storecnt_before_redefinition`, typed.
    #[test]
    fn store_lock_needs_storecnt() {
        let source = body_of(vec![
            mi("buffer_store_b32",
                vec![v(0, 1), v(1, 1), s(4, 4), s(8, 1), Operand::Vmem(VmemToken::Offen)],
            ),
            mi("v_mov_b32_e32", vec![v(0, 1), v(5, 1)]),
        ]);
        assert!(!ok(&source));
        let drained = body_of(vec![
            mi("buffer_store_b32",
                vec![v(0, 1), v(1, 1), s(4, 4), s(8, 1), Operand::Vmem(VmemToken::Offen)],
            ),
            mk("s_wait_storecnt", vec![sopp(0)], wait_mods(None, None, None, Some(0))),
            mi("v_mov_b32_e32", vec![v(0, 1), v(5, 1)]),
        ]);
        assert!(ok(&drained));
        // Reading (not redefining) a store source is fine.
        let read = body_of(vec![
            mi("buffer_store_b32",
                vec![v(0, 1), v(1, 1), s(4, 4), s(8, 1), Operand::Vmem(VmemToken::Offen)],
            ),
            vadd(2, 0, 3),
        ]);
        assert!(ok(&read));
    }

    /// Linear `smem_load_requires_kmcnt_before_scalar_use`, typed.
    #[test]
    fn smem_load_needs_kmcnt() {
        let source = body_of(vec![
            mi("s_load_b32", vec![s(4, 1), s(0, 2), Operand::Imm(ImmField::SmemOffset(0))]),
            mi("s_add_co_u32", vec![s(5, 1), s(4, 1), s(6, 1)]),
        ]);
        assert!(!ok(&source));
        let drained = body_of(vec![
            mi("s_load_b32", vec![s(4, 1), s(0, 2), Operand::Imm(ImmField::SmemOffset(0))]),
            mk("s_wait_kmcnt", vec![sopp(0)], wait_mods(None, None, Some(0), None)),
            mi("s_add_co_u32", vec![s(5, 1), s(4, 1), s(6, 1)]),
        ]);
        assert!(ok(&drained));
    }

    /// Linear `vm_vsrc_releases_store_sources…`, typed (VALU move stands in
    /// for the text test's `v_writelane_b32`, which has no M1 table row).
    #[test]
    fn vm_vsrc_releases_store_sources() {
        let store = || {
            mi("global_store_b128",
                vec![v(9, 2), v(10, 4), s(4, 2), Operand::Vmem(VmemToken::Off)],
            )
        };
        let blocked = body_of(vec![store(), mi("v_mov_b32_e32", vec![v(9, 1), v(5, 1)])]);
        assert!(!ok(&blocked));
        // depctr_vm_vsrc(0) == simm16 0xFF83 (llvm-mc pinned).
        let released = body_of(vec![
            store(),
            mk("s_wait_alu", vec![sopp(0xFF83u16 as i16)], Modifiers::default()),
            mi("v_mov_b32_e32", vec![v(9, 1), v(5, 1)]),
        ]);
        assert!(ok(&released));
        // Any other depctr does not release store sources.
        let other = body_of(vec![
            store(),
            mk("s_wait_alu", vec![sopp(0xFF9Eu16 as i16)], Modifiers::default()),
            mi("v_mov_b32_e32", vec![v(9, 1), v(5, 1)]),
        ]);
        assert!(!ok(&other));
    }

    /// Linear `mixed_vmem_loads_require_zero_wait`, typed.
    #[test]
    fn mixed_vmem_loads_need_zero_wait() {
        let mixed = body_of(vec![
            mi("buffer_load_b32",
                vec![v(0, 2), v(8, 2), s(4, 4), s(9, 1), Operand::Vmem(VmemToken::Offen)],
            ),
            mi("global_load_b64", vec![v(2, 1), v(8, 2), s(4, 2)]),
            mk("s_wait_loadcnt", vec![sopp(1)], wait_mods(Some(1), None, None, None)),
            vadd(3, 0, 4),
        ]);
        assert!(!ok(&mixed));
        let mut zero = body_of(vec![
            mi("buffer_load_b32",
                vec![v(0, 2), v(8, 2), s(4, 4), s(9, 1), Operand::Vmem(VmemToken::Offen)],
            ),
            mi("global_load_b64", vec![v(2, 1), v(8, 2), s(4, 2)]),
            mk("s_wait_loadcnt", vec![sopp(0)], wait_mods(Some(0), None, None, None)),
            vadd(3, 0, 4),
        ]);
        assert!(ok(&zero));
        // Same-family loads still retire partially: oldest of two globals.
        zero.layout.pop();
        let tail = vadd(3, 0, 4);
        let _ = tail;
        let same = body_of(vec![
            mi("global_load_b64", vec![v(0, 1), v(8, 2), s(4, 2)]),
            mi("global_load_b64", vec![v(1, 2), v(9, 2), s(4, 2)]),
            mk("s_wait_loadcnt", vec![sopp(1)], wait_mods(Some(1), None, None, None)),
            vadd(3, 0, 4),
        ]);
        assert!(ok(&same));
    }

    // ---- KT48 T6 gate ----

    fn kt48_body() -> Body {
        const IMAGE: &[u8] =
            include_bytes!("../../../peacemaker-lift/tests/fixtures/kt48/hipcc.co");
        const START: usize = 0x6f00;
        const SIZE: usize = 10_604;
        let words: Vec<u32> = IMAGE[START..START + SIZE]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let mut body = Body::default();
        let mut index = 0;
        while index < words.len() {
            let (inst, count) = crate::codec::gfx12::decode(&words[index..]).expect("KT48 decodes");
            let id = body.insts.insert(inst);
            body.layout.push(id);
            index += count;
        }
        assert_eq!(body.layout.len(), 1696);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).expect("KT48 CFG builds");
        body
    }

    #[test]
    fn kt48_wait_census() {
        let body = kt48_body();
        let census = census(&body, Arch::Gfx1201).unwrap();
        assert_eq!(census.kmcnt.len(), 7, "seven s_wait_kmcnt");
        assert!(census.kmcnt.iter().all(|count| *count == 0), "all kmcnt 0x0");
        assert_eq!(census.loadcnt, 16);
        assert_eq!(census.dscnt, 28);
        assert_eq!(census.loadcnt_dscnt, 3);
        assert_eq!(census.alu, 142);
    }

    /// T6 wait gate, corrected count. The linear text replay reports 122
    /// findings on KT48; the sound CFG-aware replay reports 124, all of the
    /// same single class. Decomposition (verified against
    /// `ledger_replay::replay_hazards` run on the canonical text of the same
    /// 1,696 instructions):
    /// * 120 findings are common to both replays;
    /// * 2 linear findings (the `ds_load_b128` at layout 917 and the use at
    ///   925) come from the linear walk's infeasible layout fall-through
    ///   across the unconditional `s_branch` at 915 (dword-pc target 1434 =
    ///   layout 967; layout 916 is reachable only via a later block, whose
    ///   exit has the DS stores retired). They are linear false positives;
    /// * 4 findings here (956, 957, 973, 974) are feasible first-iteration
    ///   paths the linear walk misses: the unconditional jump at 915 enters
    ///   the loop at 967 without executing the `s_wait_dscnt` waits at
    ///   928-949, so the `ds_store_2addr_b64` sources outstanding since 782
    ///   are still locked when 956/957 redefine v134/v135 and 973/974
    ///   redefine v130-133/v169-172. Same phenomenon as the other 120, same
    ///   disposition (diagnostic until the DS-source pin).
    #[test]
    fn kt48_replay_obligations_are_ds_store_war() {
        let body = kt48_body();
        let replay = replay(&body, Arch::Gfx1201).unwrap();
        assert_eq!(replay.obligations.len(), 124, "120 linear-common + 4 first-iteration sites");
        for obligation in &replay.obligations {
            assert_eq!(obligation.kind, ObligationKind::SrcReadTiming(MemClass::DsStore));
            assert_eq!(obligation.rule_id, "wait-war-ds-store");
            assert_eq!(obligation.insts.len(), 1);
        }
        // Every finding redefines a source of a ds_store_2addr_b64.
        for obligation in &replay.obligations {
            let consumer = obligation.insts[0];
            let inst = body.insts.get(consumer).unwrap();
            let mut touch = Vec::new();
            for reg in inst.effects.defs.iter().chain(inst.effects.uses.iter()) {
                for i in 0..reg.len {
                    touch.push((reg.kind, reg.base + u16::from(i)));
                }
            }
            let blamed = replay.events.iter().any(|event| {
                event.class == MemClass::DsStore
                    && body.insts.get(event.inst).unwrap().op.name(Arch::Gfx1201)
                        == Some("ds_store_2addr_b64")
                    && event.src_locks.0.iter().any(|reg| {
                        (0..reg.len).any(|i| touch.contains(&(reg.kind, reg.base + u16::from(i))))
                    })
            });
            assert!(blamed, "finding at {consumer:?} names no ds_store_2addr_b64 source");
        }
    }
    /// Path-sensitivity pin: layout positions 956/957/973/974 are
    /// obligations (feasible first-iteration paths via the 915 jump that
    /// skip the 928-949 waits), while 917/925 are not (reachable only from
    /// a later block; the 915->916 layout fall-through never executes).
    #[test]
    fn kt48_path_sensitive_sites() {
        let body = kt48_body();
        let replay = replay(&body, Arch::Gfx1201).unwrap();
        let positions: Vec<usize> = replay
            .obligations
            .iter()
            .map(|obligation| {
                body.layout.iter().position(|id| *id == obligation.insts[0]).unwrap()
            })
            .collect();
        for expected in [956, 957, 973, 974] {
            assert!(positions.contains(&expected), "missing feasible-path site {expected}");
        }
        for infeasible in [917, 925] {
            assert!(!positions.contains(&infeasible), "infeasible-path site {infeasible} flagged");
        }
    }

    #[test]
    fn kt48_kmcnt_units_and_retirement() {
        let body = kt48_body();
        let replay = replay(&body, Arch::Gfx1201).unwrap();
        let mut smem: Vec<_> =
            replay.events.iter().filter(|event| event.class == MemClass::SmemLoad).collect();
        assert_eq!(smem.len(), 6, "KT48 has six SMEM loads");
        smem.sort_by_key(|event| body.layout.iter().position(|id| *id == event.inst));
        // Layout order is [2,1,1,1,2,2]; the gate's [1,1,1,2,2,2] is the
        // same multiset grouped by width (3x b32 at 1 unit, 2x b128 + 1x
        // b256 at 2 units per RDNA4 ISA 5.7.1).
        let mut units: Vec<u8> =
            smem.iter().map(|event| event.units[Counter::Km as usize]).collect();
        units.sort();
        assert_eq!(units, vec![1, 1, 1, 2, 2, 2]);
        for event in &smem {
            let name = body.insts.get(event.inst).unwrap().op.name(Arch::Gfx1201).unwrap_or("?");
            let expected = if name.contains("b32") {
                1
            } else {
                assert!(name.contains("b128") || name.contains("b256"), "{name}");
                2
            };
            assert_eq!(event.units[Counter::Km as usize], expected, "{name}");
        }
        // Every SMEM event is retired by an s_wait_kmcnt 0x0.
        for event in &smem {
            let retired = replay.facts.iter().any(|fact| {
                fact.satisfies.contains(&event.id)
                    && body.insts.get(fact.wait).unwrap().op.name(Arch::Gfx1201)
                        == Some("s_wait_kmcnt")
            });
            assert!(retired, "smem event {:?} never retired by kmcnt 0x0", event.inst);
        }
    }

    #[test]
    fn kt48_mutated_kmcnt_produces_obligation() {
        let body = kt48_body();
        let wait_ids: Vec<_> = body
            .layout
            .iter()
            .filter(|id| {
                body.insts.get(**id).unwrap().op.name(Arch::Gfx1201) == Some("s_wait_kmcnt")
            })
            .copied()
            .collect();
        assert_eq!(wait_ids.len(), 7);
        let mut produced = false;
        for wait in wait_ids {
            let mut mutated = body.clone();
            let inst = mutated.insts.get_mut(wait).unwrap();
            inst.mods.wait.as_mut().unwrap().per_counter[Counter::Km as usize] = Some(1);
            let replay = replay(&mutated, Arch::Gfx1201).unwrap();
            if replay.obligations.iter().any(|obligation| {
                matches!(obligation.kind, ObligationKind::Hazard)
                    && obligation.rule_id == "wait-raw-smem-load"
            }) {
                produced = true;
                break;
            }
        }
        assert!(produced, "kmcnt 0x0 -> 0x1 must strand an s_load destination");
    }

    /// A pending load reaches its consumer across any number of blocks. The
    /// entry fixpoint used to re-walk every block per round with a
    /// 1000-round cap, so state advanced one block per round: this chain
    /// was never reached (and a 4485-block kernel took hours).
    #[test]
    fn pending_load_crosses_a_long_block_chain() {
        let mut insts = vec![buffer_load(0, 8)];
        insts.extend((0..1500).map(|_| mi("s_cbranch_scc0", vec![sopp(0)])));
        insts.push(vadd(2, 0, 3));
        insts.push(mi("s_endpgm", vec![]));
        let mut body = body_of(insts);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        assert!(body.blocks.len() > 1500);
        let consumer = body.layout[body.layout.len() - 2];
        let replay = replay(&body, Arch::Gfx1201).unwrap();
        assert!(replay.obligations.iter().any(|o| o.rule_id == "wait-raw-vmem-load" && o.insts == [consumer]),
            "{:?}", replay.obligations);
    }

}
