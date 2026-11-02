// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Wait ledger: which memory operations are still pending, and which
//! `s_waitcnt`-class waits an instruction needs before it may run.
//!
//! RDNA (gfx11 VMcnt/VScnt/LGKMcnt, gfx12 LOADcnt/STOREcnt/DScnt/KMcnt) rules
//! are documented on the items below. The CDNA3 (gfx942, wave64) rules are:
//!
//! * **VM_CNT** counts every vector-memory operation, loads *and* stores
//!   (MUBUF/MTBUF/FLAT format), in one counter: there is no VS_CNT. Reads and
//!   writes return in issue order, including mixing reads and writes
//!   (CDNA3 ISA §4.4 p.19). Every `VmemLoad`/`VmemStore` therefore forms one
//!   in-order stream (`OrderClass::Gfx9Vmem`); a store occupies a slot of it
//!   exactly like a load.
//! * **LGKM_CNT**: DS operations return in issue order (loads and stores
//!   alike); SMEM reads may return out of order, so only a wait of zero is
//!   legitimate while one is pending; operations of different types return
//!   out of order (§4.4 p.19-20). A counter that mixes DS and SMEM
//!   operations therefore always waits for zero.
//! * **EXP_CNT** (GWS/export) is recorded out of order: waits drain to zero.
//! * **Counter capacities** are read from `peacemaker_cdna::Caps`
//!   (`vmcnt_max` 63, `lgkmcnt_max` 15, `expcnt_max` 7 for gfx942). A wait
//!   threshold `T` retires, in issue order, every operation but the youngest
//!   `T`, so an operation with `younger` younger operations of its counter
//!   needs `T <= younger`. When `younger >= max` the required wait is
//!   `max - 1` (vm 62, lgkm 14, exp 6): a wait of exactly `max` asks for
//!   "at most `max` outstanding", which the counter cannot exceed, so it
//!   would be a no-op, whereas `max - 1` retires the operation (it is older
//!   than more than `max - 1` younger operations). This mirrors LLVM's
//!   `SIInsertWaitcnts.cpp` `WaitcntBrackets::determineWait`, which waits for
//!   `getWaitCountMax - 1` on a maxed-out counter instead of overflowing it.
//!   `required`/`retire_loads_of` apply it to in-order counters; below `max`
//!   younger the count is exact. The RDNA rule (exact up to 63 younger, `0`
//!   above) is unchanged.
//! * **Barrier is not a memory wait.** `s_barrier` only forces the waves of a
//!   workgroup to reach it (§4.3 p.19) and the gfx90a+ back-off barrier
//!   (`Caps::back_off_barrier`) does not wait for outstanding memory; the
//!   program must. A barrier that publishes LDS must drain DS stores first
//!   (`pending_stores`, LDS RAW) and a barrier after which another wave may
//!   overwrite an LDS slot must first retire this wave's earlier DS loads of
//!   it (`retire_loads_of`, LDS WAR): LGKM_CNT is decremented for LDS reads
//!   when data has returned to VGPRs and for LDS writes when written to LDS
//!   (§4.4 p.19-20), and nothing in the barrier does that.
//!   [UNVERIFIED: confirm on MI300X: that `s_barrier` retires no LGKM_CNT
//!   operation.]
//! * `wait_instruction` on gfx942 only produces `s_waitcnt` text for `Vm`
//!   (`vmcnt`), `Lgkm` (`lgkmcnt`) and `Exp` (`expcnt`) and rejects every
//!   other counter, and counts above the counter's capacity.
//! * `release_store_sources` is a no-op on gfx942 (no `depctr_vm_vsrc`; a
//!   store's VM_CNT slot is also its completion).

use crate::{
    arch::Arch,
    insn::{Instruction, MemoryClass},
    reg::RegRef,
};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Counter {
    Load,
    Store,
    Ds,
    Km,
    Vm,
    Vs,
    Lgkm,
    Exp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryFamily {
    Buffer,
    Global,
    Ds,
    Smem,
    Export,
}

/// The set of operations one counter retires in issue order. On gfx12,
/// `buffer_load*` and `global_load*` are both LOADcnt's single `VMEM_ACCESS`
/// event (ROCm 10.0 `SIInsertWaitcnts.cpp` `getVmemWaitEventType`,
/// `counterOutOfOrder`), so they share one class. On gfx942 `buffer_*` and
/// `global_*` loads and stores share VM_CNT's single in-order stream
/// (`Gfx9Vmem`, CDNA3 ISA §4.4 p.19). Every other operation is ordered only
/// within its family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderClass {
    Family(MemoryFamily),
    Gfx12VmemRead,
    Gfx9Vmem,
}

#[derive(Clone, Debug)]
pub struct Pending {
    pub id: u64,
    pub counter: Counter,
    pub defs: Vec<RegRef>,
    pub src_locks: Vec<RegRef>,
    pub in_order: bool,
    pub family: MemoryFamily,
    pub order: OrderClass,
    /// Pending on only some of the control paths that joined here.
    pub maybe: bool,
    /// Ids of the same counter slot on other joined paths (`Ledger::join`).
    pub aliases: Vec<u64>,
    /// LDS slots a DS load reads (`Ledger::tag_lds_load`; union over joined
    /// paths). A barrier that retires one of them must first retire the load.
    pub lds_slots: Vec<usize>,
}
impl Pending {
    fn has(&self, id: u64) -> bool {
        self.id == id || self.aliases.contains(&id)
    }
    fn ids(&self) -> impl Iterator<Item = u64> + '_ {
        std::iter::once(self.id).chain(self.aliases.iter().copied())
    }
    /// A load whose counter retires its operations in issue order.
    fn positional(&self) -> bool {
        self.in_order && !self.maybe && self.src_locks.is_empty()
    }
}

#[derive(Clone, Debug, Serialize)]
pub enum Reason {
    Use { reg: String },
    Waw { reg: String },
    War { reg: String },
    Barrier,
    LoopFixpoint,
}

#[derive(Clone, Debug, Serialize)]
pub struct WaitProof {
    pub pc_index: usize,
    pub insn: String,
    pub counter: Counter,
    pub count: u8,
    pub reason: Reason,
}

/// `arch` is the architecture of the last recorded operation (it selects the
/// counter capacities); it is `None` until something was recorded, and a
/// ledger with nothing pending needs none.
#[derive(Clone, Debug, Default)]
pub struct Ledger {
    pending: Vec<Pending>,
    next_id: u64,
    arch: Option<Arch>,
}

/// Largest count an in-order counter of a CDNA architecture can express
/// (`Caps::{vmcnt,lgkmcnt,expcnt}_max`); `None` on RDNA and for counters CDNA
/// does not have.
fn cdna_counter_max(arch: Option<Arch>, counter: Counter) -> Option<usize> {
    let caps = arch?.cdna()?.caps();
    match counter {
        Counter::Vm => Some(usize::from(caps.vmcnt_max)),
        Counter::Lgkm => Some(usize::from(caps.lgkmcnt_max)),
        Counter::Exp => Some(usize::from(caps.expcnt_max)),
        _ => None,
    }
}

impl Ledger {
    pub fn record(&mut self, arch: Arch, insn: &Instruction) {
        let Some(class) = insn.memory else { return };
        self.arch = Some(arch);
        let gfx942 = arch == Arch::Gfx942;
        let mnemonic = insn.mnemonic();
        let (counter, in_order, load) = match (arch.gfx12(), class) {
            (true, MemoryClass::VmemLoad) => (Counter::Load, true, true),
            (true, MemoryClass::VmemStore) => (Counter::Store, true, false),
            (true, MemoryClass::DsLoad) => (Counter::Ds, true, true),
            (true, MemoryClass::DsStore) => (Counter::Ds, true, false),
            (true, MemoryClass::SmemLoad) => (Counter::Km, false, true),
            (false, MemoryClass::VmemLoad) => (Counter::Vm, true, true),
            // gfx942 has no VS_CNT: stores share VM_CNT with loads.
            (false, MemoryClass::VmemStore) => {
                (if gfx942 { Counter::Vm } else { Counter::Vs }, true, false)
            }
            (false, MemoryClass::DsLoad) => (Counter::Lgkm, true, true),
            (false, MemoryClass::DsStore) => (Counter::Lgkm, true, false),
            (false, MemoryClass::SmemLoad) => (Counter::Lgkm, false, true),
            (_, MemoryClass::Export) => (Counter::Exp, false, false),
        };
        let family = match class {
            MemoryClass::VmemLoad | MemoryClass::VmemStore if mnemonic.starts_with("buffer_") => {
                MemoryFamily::Buffer
            }
            MemoryClass::VmemLoad | MemoryClass::VmemStore => MemoryFamily::Global,
            MemoryClass::DsLoad | MemoryClass::DsStore => MemoryFamily::Ds,
            MemoryClass::SmemLoad => MemoryFamily::Smem,
            MemoryClass::Export => MemoryFamily::Export,
        };
        let order = if arch.gfx12()
            && class == MemoryClass::VmemLoad
            && (mnemonic.starts_with("buffer_load") || mnemonic.starts_with("global_load"))
        {
            OrderClass::Gfx12VmemRead
        } else if gfx942 && matches!(class, MemoryClass::VmemLoad | MemoryClass::VmemStore) {
            OrderClass::Gfx9Vmem
        } else {
            OrderClass::Family(family)
        };
        self.push(counter, in_order, load, family, order, insn);
    }

    fn push(
        &mut self,
        counter: Counter,
        in_order: bool,
        load: bool,
        family: MemoryFamily,
        order: OrderClass,
        insn: &Instruction,
    ) {
        self.pending.push(Pending {
            id: self.next_id,
            counter,
            defs: if load { insn.defs.clone() } else { vec![] },
            src_locks: if load { vec![] } else { insn.uses.clone() },
            in_order,
            family,
            order,
            maybe: false,
            aliases: vec![],
            lds_slots: vec![],
        });
        self.next_id += 1;
    }

    /// The count that retires `self.pending[index]`: the younger operations of
    /// its counter when the counter retires in issue order, else zero. On
    /// gfx942 `younger >= max` yields `max - 1` (a wait of `max` would be a
    /// no-op; see the module doc); RDNA waits for zero past 63.
    fn retire_count(&self, index: usize) -> u8 {
        let pending = &self.pending[index];
        let peers = self.pending.iter().filter(|p| p.counter == pending.counter);
        let ordered = pending.in_order
            && peers
                .clone()
                .all(|p| p.in_order && !p.maybe && p.order == pending.order);
        if !ordered {
            return 0;
        }
        let younger = self
            .pending
            .iter()
            .skip(index + 1)
            .filter(|p| p.counter == pending.counter)
            .count();
        match cdna_counter_max(self.arch, pending.counter) {
            Some(max) => younger.min(max.saturating_sub(1)) as u8,
            None if younger <= 63 => younger as u8,
            None => 0,
        }
    }

    pub fn required(&self, insn: &Instruction) -> Vec<(Counter, u8, Reason)> {
        let mut waits: Vec<(Counter, u8, Reason)> = Vec::new();
        for (index, pending) in self.pending.iter().enumerate() {
            let reason = pending
                .defs
                .iter()
                .find_map(|def| {
                    insn.uses
                        .iter()
                        .find(|use_| def.overlaps(**use_))
                        .map(|_| Reason::Use {
                            reg: def.to_string(),
                        })
                        .or_else(|| {
                            insn.defs
                                .iter()
                                .find(|write| def.overlaps(**write))
                                .map(|_| Reason::Waw {
                                    reg: def.to_string(),
                                })
                        })
                })
                .or_else(|| {
                    pending.src_locks.iter().find_map(|lock| {
                        insn.defs
                            .iter()
                            .find(|write| lock.overlaps(**write))
                            .map(|_| Reason::War {
                                reg: lock.to_string(),
                            })
                    })
                });
            let Some(reason) = reason else { continue };
            let count = self.retire_count(index);
            if let Some(wait) = waits
                .iter_mut()
                .find(|(counter, _, _)| *counter == pending.counter)
            {
                if count < wait.1 {
                    *wait = (pending.counter, count, reason)
                }
            } else {
                waits.push((pending.counter, count, reason));
            }
        }
        waits
    }

    /// Mark the most recently recorded operation (a DS load) as reading LDS
    /// slot `slot`.
    pub fn tag_lds_load(&mut self, slot: usize) -> Result<(), String> {
        match self.pending.last_mut() {
            Some(p)
                if p.id + 1 == self.next_id
                    && p.family == MemoryFamily::Ds
                    && p.src_locks.is_empty() =>
            {
                if !p.lds_slots.contains(&slot) {
                    p.lds_slots.push(slot)
                }
                Ok(())
            }
            _ => Err("tag_lds_load: the last recorded operation is not a pending DS load".into()),
        }
    }

    /// The wait that retires every pending DS load reading one of `slots`
    /// (`None` when there is none). A workgroup barrier does not wait for
    /// memory counters: on a WGP the other CU's waves may store to a retired
    /// slot before this wave's earlier load of it has read LDS (WAR), so a
    /// barrier that retires a slot must retire its loads first. On gfx942 the
    /// barrier is likewise not an LGKM_CNT wait (CDNA3 ISA §4.3 p.19,
    /// §4.4 p.19-20; `Caps::back_off_barrier`): the DS load retires when its
    /// data returned to VGPRs, which only this wait guarantees. The count is
    /// `required`'s: the younger operations of the counter when it retires in
    /// issue order (`max - 1` once `max` or more are younger on gfx942), else
    /// zero.
    pub fn retire_loads_of(&self, slots: &[usize]) -> Option<(Counter, u8)> {
        let (index, pending) = self
            .pending
            .iter()
            .enumerate()
            .rev()
            .find(|(_, p)| p.lds_slots.iter().any(|s| slots.contains(s)))?;
        Some((pending.counter, self.retire_count(index)))
    }

    /// An operation retires once `count` younger operations of its counter
    /// are certainly issued after it (without joined `maybe` operations:
    /// all but the youngest `count`).
    pub fn wait(&mut self, counter: Counter, count: u8) {
        let mut younger = 0usize;
        let mut keep = vec![true; self.pending.len()];
        for (index, p) in self.pending.iter().enumerate().rev() {
            if p.counter != counter {
                continue;
            }
            if younger >= usize::from(count) {
                keep[index] = false
            }
            if !p.maybe {
                younger += 1
            }
        }
        let mut keep = keep.into_iter();
        self.pending.retain(|_| keep.next().unwrap_or(true));
    }

    /// Join the ledger of another control path into this point, per
    /// operation id: ids say which operation a `Wave::wait` token awaits,
    /// so every id pending on either path stays pending. Ids stay unique
    /// across paths (`resume`), so an equal id is the same operation.
    ///
    /// A counter whose pending operations are in-order loads of one order
    /// class on both paths, with equal shapes, retires them by position: the
    /// paths' operations are merged slot by slot, each slot certain and
    /// carrying both paths' ids. On any other counter (stores, SMEM and
    /// other out-of-order families, or different shapes) an operation
    /// pending on only one path is `maybe`: waits on its counter drain to
    /// zero (`required`) and a partial wait retires only what is certain
    /// (`wait`), which holds on both paths.
    pub fn join(&mut self, other: &Ledger) {
        fn on(l: &Ledger, c: Counter) -> Vec<&Pending> {
            l.pending.iter().filter(|p| p.counter == c).collect()
        }
        let mut positional = Vec::new();
        for c in self.pending.iter().chain(&other.pending).map(|p| p.counter) {
            if positional.contains(&c) {
                continue;
            }
            let (mine, theirs) = (on(self, c), on(other, c));
            let order = mine.first().map(|p| p.order);
            let slot = |p: &&Pending| p.positional() && Some(p.order) == order;
            if mine.len() == theirs.len()
                && mine.iter().chain(&theirs).all(slot)
                && mine.iter().zip(&theirs).all(|(p, o)| p.defs == o.defs)
            {
                positional.push(c);
            }
        }
        let mut merged: Vec<Pending> = Vec::with_capacity(self.pending.len());
        for c in &positional {
            let theirs = on(other, *c);
            for (p, o) in self.pending.iter().filter(|p| p.counter == *c).zip(theirs) {
                let mut p = p.clone();
                for id in o.ids() {
                    if !p.has(id) {
                        p.aliases.push(id)
                    }
                }
                for s in &o.lds_slots {
                    if !p.lds_slots.contains(s) {
                        p.lds_slots.push(*s)
                    }
                }
                merged.push(p);
            }
        }
        for p in self
            .pending
            .iter()
            .filter(|p| !positional.contains(&p.counter))
        {
            let mut p = p.clone();
            match other.pending.iter().find(|o| o.ids().any(|id| p.has(id))) {
                Some(o) => {
                    p.maybe |= o.maybe;
                    for id in o.ids() {
                        if !p.has(id) {
                            p.aliases.push(id)
                        }
                    }
                    for s in &o.lds_slots {
                        if !p.lds_slots.contains(s) {
                            p.lds_slots.push(*s)
                        }
                    }
                }
                None => p.maybe = true,
            }
            merged.push(p);
        }
        for o in other
            .pending
            .iter()
            .filter(|o| !positional.contains(&o.counter))
        {
            if !self.pending.iter().any(|p| o.ids().any(|id| p.has(id))) {
                merged.push(Pending {
                    maybe: true,
                    ..o.clone()
                });
            }
        }
        merged.sort_by_key(|p| p.id);
        self.pending = merged;
        self.next_id = self.next_id.max(other.next_id);
        self.arch = self.arch.or(other.arch);
    }
    /// Continue at a branch target with the ledger of the branch point.
    /// Ids issued on the path that fell through stay used, so the two
    /// paths' operations keep distinct ids when they join later.
    pub fn resume(&mut self, at: Ledger) {
        let next_id = self.next_id.max(at.next_id);
        let arch = at.arch.or(self.arch);
        *self = at;
        self.next_id = next_id;
        self.arch = arch;
    }

    pub fn drain(&mut self) -> Vec<(Counter, u8, Reason)> {
        let mut waits = Vec::new();
        for pending in &self.pending {
            if !waits
                .iter()
                .any(|(counter, _, _)| *counter == pending.counter)
            {
                waits.push((pending.counter, 0, Reason::Barrier));
            }
        }
        for (counter, _, _) in &waits {
            self.wait(*counter, 0)
        }
        waits
    }

    /// After `s_wait_alu depctr_vm_vsrc(0)` every issued VMEM store has read
    /// its source registers: pending stores lock nothing and define nothing,
    /// so they leave the ledger (completion order is never inferred for them).
    /// gfx942 has no such wait and its stores share VM_CNT with loads, so
    /// there it releases nothing.
    pub fn release_store_sources(&mut self) {
        if self.arch == Some(Arch::Gfx942) {
            return;
        }
        self.pending
            .retain(|p| !matches!(p.counter, Counter::Store | Counter::Vs));
    }

    /// An LDS store not yet retired (DScnt on gfx12, LGKMcnt on gfx11 and
    /// gfx942): a barrier that publishes LDS must drain it first.
    pub fn pending_stores(&self) -> bool {
        self.pending.iter().any(|p| {
            matches!(p.counter, Counter::Ds | Counter::Lgkm)
                && p.family == MemoryFamily::Ds
                && !p.src_locks.is_empty()
        })
    }
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
    /// Id of the most recently recorded operation.
    pub fn last_id(&self) -> Option<u64> {
        self.next_id.checked_sub(1)
    }
    /// Operation `id` (on any joined path) has not been retired by a wait.
    pub fn is_pending(&self, id: u64) -> bool {
        self.pending.iter().any(|p| p.has(id))
    }
    /// Pending operations without their ids: two ledgers with equal shapes
    /// require the same waits for every later instruction.
    pub fn shape(&self) -> Vec<(Counter, Vec<RegRef>, Vec<RegRef>, bool, MemoryFamily, bool)> {
        self.pending
            .iter()
            .map(|p| {
                (
                    p.counter,
                    p.defs.clone(),
                    p.src_locks.clone(),
                    p.in_order,
                    p.family,
                    p.maybe,
                )
            })
            .collect()
    }

    pub fn wait_instruction(
        arch: Arch,
        waits: &[(Counter, u8, Reason)],
    ) -> Result<Vec<(Counter, u8, String, Reason)>, String> {
        let mut out = Vec::new();
        for (counter, count, reason) in waits {
            let text = if let Some(cdna) = arch.cdna() {
                let caps = cdna.caps();
                let (name, max) = match counter {
                    Counter::Vm => ("vmcnt", caps.vmcnt_max),
                    Counter::Lgkm => ("lgkmcnt", caps.lgkmcnt_max),
                    Counter::Exp => ("expcnt", caps.expcnt_max),
                    _ => return Err(format!("{counter:?} is not a {} counter", arch.name())),
                };
                if *count > max {
                    return Err(format!(
                        "{name}({count}) exceeds the {} maximum {max}",
                        arch.name()
                    ));
                }
                format!("s_waitcnt {name}({count})")
            } else {
                match (arch.gfx12(), counter) {
                    (true, Counter::Load) => format!("s_wait_loadcnt {count:#x}"),
                    (true, Counter::Store) => format!("s_wait_storecnt {count:#x}"),
                    (true, Counter::Ds) => format!("s_wait_dscnt {count:#x}"),
                    (true, Counter::Km) => format!("s_wait_kmcnt {count:#x}"),
                    (false, Counter::Vm) => format!("s_waitcnt vmcnt({count})"),
                    (false, Counter::Vs) => format!("s_waitcnt_vscnt null, {count:#x}"),
                    (false, Counter::Lgkm) => format!("s_waitcnt lgkmcnt({count})"),
                    (_, Counter::Exp) => format!("s_waitcnt expcnt({count})"),
                    _ => return Err("counter/architecture mismatch".into()),
                }
            };
            out.push((*counter, *count, text, reason.clone()));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod join_tests {
    use super::*;
    use crate::reg::Kind;

    fn r(kind: Kind, base: u8) -> RegRef {
        RegRef { kind, base, len: 1 }
    }
    /// Two paths issue the same instructions after a fork (distinct ids).
    fn paths(arch: Arch, insns: &[Instruction]) -> (Ledger, Ledger, Vec<u64>, Vec<u64>) {
        let mut a = Ledger::default();
        for i in insns {
            a.record(arch, i)
        }
        let ids_a = (0..insns.len() as u64).collect::<Vec<_>>();
        let mut b = Ledger {
            pending: vec![],
            next_id: a.next_id,
            arch: None,
        };
        for i in insns {
            b.record(arch, i)
        }
        let ids_b = ids_a.iter().map(|i| i + a.next_id).collect();
        (a, b, ids_a, ids_b)
    }

    /// gfx11 LGKMcnt shared by DS and SMEM, and gfx12 KMcnt, retire out of
    /// order: an equal-shape join keeps both paths' operations (every id),
    /// marked maybe, so a use drains the counter. In-order loads merge slot
    /// by slot, each slot carrying both ids, and keep exact counts.
    #[test]
    fn equal_shape_join_merges_by_position_only_for_in_order_loads() {
        let ds = Instruction::new(
            "ds_load_b32 v1, v2",
            vec![r(Kind::V, 1)],
            vec![r(Kind::V, 2)],
        )
        .memory(MemoryClass::DsLoad);
        let smem = Instruction::new("s_load_b32 s4, s[0:1], 0x0", vec![r(Kind::S, 4)], vec![])
            .memory(MemoryClass::SmemLoad);
        let vm = Instruction::new(
            "global_load_b32 v5, v2, s[0:1]",
            vec![r(Kind::V, 5)],
            vec![r(Kind::V, 2)],
        )
        .memory(MemoryClass::VmemLoad);
        let use_v1 = Instruction::new("v_mov_b32 v3, v1", vec![r(Kind::V, 3)], vec![r(Kind::V, 1)]);
        let ds6 = Instruction::new(
            "ds_load_b32 v6, v2",
            vec![r(Kind::V, 6)],
            vec![r(Kind::V, 2)],
        )
        .memory(MemoryClass::DsLoad);
        for (arch, insns, maybe, wait) in [
            (
                Arch::Gfx1100,
                vec![ds.clone(), smem.clone(), ds6.clone()],
                true,
                0,
            ),
            (Arch::Gfx1201, vec![smem.clone(), smem.clone()], true, 0),
            (
                Arch::Gfx1100,
                vec![ds.clone(), vm.clone(), ds6.clone()],
                false,
                1,
            ),
        ] {
            let (mut a, b, ids_a, ids_b) = paths(arch, &insns);
            a.join(&b);
            assert!(
                ids_a.iter().chain(&ids_b).all(|&id| a.is_pending(id)),
                "{arch:?} lost an id"
            );
            assert_eq!(
                a.shape().len(),
                insns.len() * if maybe { 2 } else { 1 },
                "{arch:?}"
            );
            let lgkm = a
                .shape()
                .into_iter()
                .filter(|s| matches!(s.0, Counter::Lgkm | Counter::Km))
                .all(|s| s.5 == maybe);
            assert!(lgkm, "{arch:?}: slots maybe={maybe}");
            if insns[0].mnemonic().starts_with("ds_") {
                assert_eq!(
                    a.required(&use_v1).first().map(|w| w.1),
                    Some(wait),
                    "{arch:?}"
                );
            }
        }
    }

    /// gfx12: a mixed buffer/global window waits for exactly the younger
    /// LOADcnt operations behind the consumed load; gfx11 VMcnt and a flat
    /// load in the window still drain to zero.
    #[test]
    fn gfx12_mixed_buffer_global_window_waits_by_position() {
        let load = |text: &str, d: u8| {
            Instruction::new(text, vec![r(Kind::V, d)], vec![r(Kind::V, 8)])
                .memory(MemoryClass::VmemLoad)
        };
        let use_of = |d: u8| {
            Instruction::new(
                "v_mov_b32 v40, v0",
                vec![r(Kind::V, 40)],
                vec![r(Kind::V, d)],
            )
        };
        let window = [
            load("buffer_load_b32 v0, v8, s[4:7], s9 offen", 0),
            load("global_load_b32 v1, v8, s[4:5]", 1),
            load("buffer_load_b32 v2, v8, s[4:7], s9 offen offset:4", 2),
            load("global_load_b32 v3, v8, s[4:5] offset:4", 3),
        ];
        let mut l = Ledger::default();
        for i in &window {
            l.record(Arch::Gfx1201, i)
        }
        for (d, n) in [(0u8, 3u8), (1, 2), (2, 1), (3, 0)] {
            assert_eq!(
                l.required(&use_of(d)).first().map(|w| (w.0, w.1)),
                Some((Counter::Load, n)),
                "v{d}"
            );
        }
        let mut gfx11 = Ledger::default();
        for i in &window {
            gfx11.record(Arch::Gfx1100, i)
        }
        assert_eq!(gfx11.required(&use_of(0)).first().map(|w| w.1), Some(0));
        let mut flat = Ledger::default();
        for i in [window[0].clone(), load("flat_load_b32 v1, v[8:9]", 1)] {
            flat.record(Arch::Gfx1201, &i)
        }
        assert_eq!(flat.required(&use_of(0)).first().map(|w| w.1), Some(0));
    }
}

#[cfg(test)]
mod gfx942_tests {
    use super::*;
    use crate::reg::Kind;

    fn r(kind: Kind, base: u8) -> RegRef {
        RegRef { kind, base, len: 1 }
    }
    fn v(base: u8) -> RegRef {
        r(Kind::V, base)
    }
    fn load(text: &str, d: u8) -> Instruction {
        Instruction::new(text, vec![v(d)], vec![v(8)]).memory(MemoryClass::VmemLoad)
    }
    fn store(text: &str, data: u8) -> Instruction {
        Instruction::new(text, vec![], vec![v(8), v(data)]).memory(MemoryClass::VmemStore)
    }
    fn ds_load(d: u8) -> Instruction {
        Instruction::new(format!("ds_read_b32 v{d}, v2"), vec![v(d)], vec![v(2)])
            .memory(MemoryClass::DsLoad)
    }
    fn ds_store(data: u8) -> Instruction {
        Instruction::new(
            format!("ds_write_b32 v2, v{data}"),
            vec![],
            vec![v(2), v(data)],
        )
        .memory(MemoryClass::DsStore)
    }
    fn smem(d: u8) -> Instruction {
        Instruction::new(
            format!("s_load_dword s{d}, s[0:1], 0x0"),
            vec![r(Kind::S, d)],
            vec![],
        )
        .memory(MemoryClass::SmemLoad)
    }
    /// Reads `d`.
    fn read(d: u8) -> Instruction {
        Instruction::new("v_mov_b32 v250, v0", vec![v(250)], vec![v(d)])
    }
    /// Overwrites `d` without reading it.
    fn write(d: u8) -> Instruction {
        Instruction::new("v_mov_b32 v0, 0", vec![v(d)], vec![])
    }
    fn first(l: &Ledger, i: &Instruction) -> Option<(Counter, u8)> {
        l.required(i).first().map(|w| (w.0, w.1))
    }
    fn ledger(arch: Arch, insns: &[Instruction]) -> Ledger {
        let mut l = Ledger::default();
        for i in insns {
            l.record(arch, i)
        }
        l
    }

    /// Loads and stores of buffer and global share VM_CNT in issue order
    /// (CDNA3 ISA §4.4 p.19): the wait is the number of younger VM_CNT
    /// operations of either kind, and no VS_CNT exists.
    #[test]
    fn mixed_buffer_global_loads_and_stores_wait_by_position() {
        let window = [
            load("global_load_dword v1, v8, s[4:5]", 1),
            store("global_store_dword v8, v20, s[4:5]", 20),
            load("buffer_load_dword v2, v8, s[4:7], s9 offen", 2),
            store("buffer_store_dword v21, v8, s[4:7], s9 offen", 21),
            load("global_load_dword v3, v8, s[4:5] offset:4", 3),
        ];
        let l = ledger(Arch::Gfx942, &window);
        assert!(
            l.shape().iter().all(|s| s.0 == Counter::Vm && s.3),
            "every operation is an in-order Vm slot"
        );
        for (d, n) in [(1u8, 4u8), (2, 2), (3, 0)] {
            assert_eq!(first(&l, &read(d)), Some((Counter::Vm, n)), "v{d}");
        }
        // Overwriting a store's data register (WAR) waits for the stores' slot.
        for (d, n) in [(20u8, 3u8), (21, 1)] {
            let w = l.required(&write(d));
            assert_eq!((w[0].0, w[0].1), (Counter::Vm, n), "store data v{d}");
            assert!(
                matches!(&w[0].2, Reason::War { .. }),
                "store data v{d}: {:?}",
                w[0].2
            );
        }
        // The same window on gfx1100 puts the stores on VScnt.
        let rdna = ledger(Arch::Gfx1100, &window);
        assert_eq!(
            rdna.shape().iter().filter(|s| s.0 == Counter::Vs).count(),
            2
        );
        // gfx11 keeps buffer and global in separate order classes: drains to zero.
        assert_eq!(first(&rdna, &read(1)), Some((Counter::Vm, 0)));
    }

    /// SMEM returns out of order: only zero is a legitimate wait, for its own
    /// slot and for DS operations sharing LGKM_CNT with it.
    #[test]
    fn smem_waits_for_zero() {
        let l = ledger(Arch::Gfx942, &[smem(4), smem(5), ds_load(1), ds_load(6)]);
        let s4 = Instruction::new(
            "s_mov_b32 s60, s4",
            vec![r(Kind::S, 60)],
            vec![r(Kind::S, 4)],
        );
        assert_eq!(first(&l, &s4), Some((Counter::Lgkm, 0)));
        assert_eq!(first(&l, &read(1)), Some((Counter::Lgkm, 0)));
        assert_eq!(first(&l, &read(6)), Some((Counter::Lgkm, 0)));
        let only = ledger(Arch::Gfx942, &[smem(4), smem(5)]);
        assert_eq!(first(&only, &s4), Some((Counter::Lgkm, 0)));
    }

    /// DS loads and stores return in issue order on LGKM_CNT; a store's data
    /// registers are locked until it retires.
    #[test]
    fn ds_counts_and_pending_stores() {
        let mut l = ledger(
            Arch::Gfx942,
            &[ds_load(1), ds_store(3), ds_load(4), ds_load(5)],
        );
        for (d, n) in [(1u8, 3u8), (4, 1), (5, 0)] {
            assert_eq!(first(&l, &read(d)), Some((Counter::Lgkm, n)), "v{d}");
        }
        assert_eq!(first(&l, &write(3)), Some((Counter::Lgkm, 2)));
        assert!(l.pending_stores());
        l.wait(Counter::Lgkm, 2);
        assert!(
            !l.pending_stores(),
            "the store is older than the two youngest DS loads"
        );
        assert!(!ledger(
            Arch::Gfx942,
            &[ds_load(1), store("global_store_dword v8, v3, s[4:5]", 3)]
        )
        .pending_stores());
    }

    /// At or past the counter capacity (lgkmcnt 15, vmcnt 63; `Caps`) the
    /// count is `max - 1` on gfx942 (a wait of `max` would be a no-op, as in
    /// LLVM `SIInsertWaitcnts` `determineWait`); RDNA keeps its rule (exact
    /// up to 63 younger, zero beyond).
    #[test]
    fn counts_cap_below_cdna_maximum() {
        let ds: Vec<_> = (0..21u8).map(|i| ds_load(10 + i)).collect();
        let l = ledger(Arch::Gfx942, &ds);
        for (d, n) in [(10u8, 14u8), (15, 14), (16, 14), (17, 13), (30, 0)] {
            assert_eq!(first(&l, &read(d)), Some((Counter::Lgkm, n)), "v{d}");
        }
        let rdna = ledger(Arch::Gfx1100, &ds);
        assert_eq!(first(&rdna, &read(10)), Some((Counter::Lgkm, 20)));

        let vm: Vec<_> = (0..70u8)
            .map(|i| load("global_load_dword v100, v8, s[4:5]", 100 + i))
            .collect();
        let l = ledger(Arch::Gfx942, &vm);
        for (d, n) in [(100u8, 62u8), (106, 62), (107, 62), (108, 61), (169, 0)] {
            assert_eq!(first(&l, &read(d)), Some((Counter::Vm, n)), "v{d}");
        }
        let rdna = ledger(Arch::Gfx1100, &vm);
        assert_eq!(first(&rdna, &read(100)), Some((Counter::Vm, 0)));
        assert_eq!(first(&rdna, &read(106)), Some((Counter::Vm, 63)));
        assert_eq!(first(&rdna, &read(107)), Some((Counter::Vm, 62)));
    }

    /// A barrier is not an LGKM wait: the DS loads of a retired slot must be
    /// retired first (WAR), by position, `max - 1` = lgkmcnt 14 at 15 or more
    /// younger, or zero
    /// behind an SMEM operation.
    #[test]
    fn retire_loads_of_orders_ds_loads_before_the_barrier() {
        let ds: Vec<_> = (0..21u8).map(|i| ds_load(10 + i)).collect();
        let tagged = |arch: Arch, insns: &[Instruction]| {
            let mut l = Ledger::default();
            for (i, insn) in insns.iter().enumerate() {
                l.record(arch, insn);
                if i == 0 {
                    l.tag_lds_load(3).unwrap()
                }
            }
            l
        };
        assert_eq!(
            tagged(Arch::Gfx942, &ds[..3]).retire_loads_of(&[3]),
            Some((Counter::Lgkm, 2))
        );
        assert_eq!(
            tagged(Arch::Gfx942, &ds).retire_loads_of(&[3]),
            Some((Counter::Lgkm, 14))
        );
        assert_eq!(
            tagged(Arch::Gfx1100, &ds).retire_loads_of(&[3]),
            Some((Counter::Lgkm, 20))
        );
        assert_eq!(tagged(Arch::Gfx942, &ds[..3]).retire_loads_of(&[4]), None);
        let mut with_smem = vec![ds[0].clone(), smem(4)];
        with_smem.extend_from_slice(&ds[1..3]);
        assert_eq!(
            tagged(Arch::Gfx942, &with_smem).retire_loads_of(&[3]),
            Some((Counter::Lgkm, 0))
        );
        let mut not_ds = ledger(Arch::Gfx942, &[load("global_load_dword v1, v8, s[4:5]", 1)]);
        assert!(not_ds.tag_lds_load(3).is_err());
    }

    /// Two paths issuing the same in-order global/buffer loads merge slot by
    /// slot (exact counts, every id pending); a path with a store (its
    /// source lock is not positional) leaves every operation `maybe`, so the
    /// wait drains to zero.
    #[test]
    fn two_path_join() {
        let loads = [
            load("global_load_dword v1, v8, s[4:5]", 1),
            load("buffer_load_dword v2, v8, s[4:7], s9 offen", 2),
        ];
        let mut a = ledger(Arch::Gfx942, &loads);
        let mut b = Ledger {
            pending: vec![],
            next_id: a.next_id,
            arch: None,
        };
        for i in &loads {
            b.record(Arch::Gfx942, i)
        }
        let ids: Vec<u64> = (0..4).collect();
        a.join(&b);
        assert!(ids.iter().all(|&id| a.is_pending(id)));
        assert_eq!(a.shape().len(), 2);
        assert!(a.shape().iter().all(|s| !s.5));
        assert_eq!(first(&a, &read(1)), Some((Counter::Vm, 1)));

        let mixed = [
            load("global_load_dword v1, v8, s[4:5]", 1),
            store("global_store_dword v8, v20, s[4:5]", 20),
        ];
        let mut a = ledger(Arch::Gfx942, &mixed);
        let mut b = Ledger {
            pending: vec![],
            next_id: a.next_id,
            arch: None,
        };
        for i in &mixed {
            b.record(Arch::Gfx942, i)
        }
        a.join(&b);
        assert_eq!(a.shape().len(), 4);
        assert!(a.shape().iter().all(|s| s.5));
        assert_eq!(first(&a, &read(1)), Some((Counter::Vm, 0)));
    }

    /// Export is EXP_CNT, unordered.
    #[test]
    fn export_waits_on_expcnt_for_zero() {
        let exp = Instruction::new("exp mrt0 v1, v1, v1, v1", vec![], vec![v(1)])
            .memory(MemoryClass::Export);
        let l = ledger(Arch::Gfx942, &[exp.clone(), exp]);
        assert_eq!(first(&l, &write(1)), Some((Counter::Exp, 0)));
    }

    /// gfx942 keeps pending stores (no `depctr_vm_vsrc`); RDNA releases them.
    #[test]
    fn release_store_sources_is_a_gfx942_no_op() {
        let st = store("global_store_dword v8, v20, s[4:5]", 20);
        let mut l = ledger(Arch::Gfx942, &[st.clone()]);
        l.release_store_sources();
        assert!(!l.is_empty());
        let mut rdna = ledger(Arch::Gfx1100, &[st]);
        rdna.release_store_sources();
        assert!(rdna.is_empty());
    }

    /// gfx942 waits are `s_waitcnt` over Vm/Lgkm/Exp only, within the
    /// capacities; every other counter and an unencodable count are errors.
    #[test]
    fn wait_instruction_text_and_errors() {
        let waits = [
            (Counter::Vm, 63, Reason::Barrier),
            (Counter::Lgkm, 15, Reason::Barrier),
            (Counter::Exp, 7, Reason::Barrier),
        ];
        let text: Vec<String> = Ledger::wait_instruction(Arch::Gfx942, &waits)
            .unwrap()
            .into_iter()
            .map(|w| w.2)
            .collect();
        assert_eq!(
            text,
            [
                "s_waitcnt vmcnt(63)",
                "s_waitcnt lgkmcnt(15)",
                "s_waitcnt expcnt(7)"
            ]
        );
        for counter in [
            Counter::Load,
            Counter::Store,
            Counter::Ds,
            Counter::Km,
            Counter::Vs,
        ] {
            assert!(
                Ledger::wait_instruction(Arch::Gfx942, &[(counter, 0, Reason::Barrier)]).is_err(),
                "{counter:?}"
            );
        }
        for (counter, count) in [(Counter::Vm, 64), (Counter::Lgkm, 16), (Counter::Exp, 8)] {
            assert!(
                Ledger::wait_instruction(Arch::Gfx942, &[(counter, count, Reason::Barrier)])
                    .is_err(),
                "{counter:?} {count}"
            );
        }
        // RDNA text is unchanged.
        let rdna = Ledger::wait_instruction(
            Arch::Gfx1100,
            &[
                (Counter::Vs, 3, Reason::Barrier),
                (Counter::Lgkm, 20, Reason::Barrier),
            ],
        )
        .unwrap();
        assert_eq!(
            (rdna[0].2.as_str(), rdna[1].2.as_str()),
            ("s_waitcnt_vscnt null, 0x3", "s_waitcnt lgkmcnt(20)")
        );
        assert!(
            Ledger::wait_instruction(Arch::Gfx1100, &[(Counter::Load, 0, Reason::Barrier)])
                .is_err()
        );
    }
}
