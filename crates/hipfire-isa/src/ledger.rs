use crate::{
    arch::Arch,
    insn::{Instruction, MemoryClass},
    reg::RegRef,
};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Counter { Load, Store, Ds, Km, Vm, Vs, Lgkm, Exp }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryFamily { Buffer, Global, Ds, Smem, Export }

#[derive(Clone, Debug)]
pub struct Pending {
    pub id: u64,
    pub counter: Counter,
    pub defs: Vec<RegRef>,
    pub src_locks: Vec<RegRef>,
    pub in_order: bool,
    pub family: MemoryFamily,
    /// Pending on only some of the control paths that joined here.
    pub maybe: bool,
    /// Ids of the same counter slot on other joined paths (`Ledger::join`).
    pub aliases: Vec<u64>,
}
impl Pending {
    fn has(&self, id: u64) -> bool { self.id == id || self.aliases.contains(&id) }
    fn ids(&self) -> impl Iterator<Item = u64> + '_ { std::iter::once(self.id).chain(self.aliases.iter().copied()) }
    /// A load whose counter retires its operations in issue order.
    fn positional(&self) -> bool { self.in_order && !self.maybe && self.src_locks.is_empty() }
}

#[derive(Clone, Debug, Serialize)]
pub enum Reason { Use { reg: String }, Waw { reg: String }, War { reg: String }, Barrier, LoopFixpoint }

#[derive(Clone, Debug, Serialize)]
pub struct WaitProof { pub pc_index: usize, pub insn: String, pub counter: Counter, pub count: u8, pub reason: Reason }

#[derive(Clone, Debug, Default)]
pub struct Ledger { pending: Vec<Pending>, next_id: u64 }

impl Ledger {
    pub fn record(&mut self, arch: Arch, insn: &Instruction) {
        let Some(class) = insn.memory else { return };
        let (counter, in_order, load) = match (arch.gfx12(), class) {
            (true, MemoryClass::VmemLoad) => (Counter::Load, true, true),
            (true, MemoryClass::VmemStore) => (Counter::Store, true, false),
            (true, MemoryClass::DsLoad) => (Counter::Ds, true, true),
            (true, MemoryClass::DsStore) => (Counter::Ds, true, false),
            (true, MemoryClass::SmemLoad) => (Counter::Km, false, true),
            (false, MemoryClass::VmemLoad) => (Counter::Vm, true, true),
            (false, MemoryClass::VmemStore) => (Counter::Vs, true, false),
            (false, MemoryClass::DsLoad) => (Counter::Lgkm, true, true),
            (false, MemoryClass::DsStore) => (Counter::Lgkm, true, false),
            (false, MemoryClass::SmemLoad) => (Counter::Lgkm, false, true),
            (_, MemoryClass::Export) => (Counter::Exp, false, false),
        };
        let family = match class {
            MemoryClass::VmemLoad | MemoryClass::VmemStore if insn.mnemonic().starts_with("buffer_") => MemoryFamily::Buffer,
            MemoryClass::VmemLoad | MemoryClass::VmemStore => MemoryFamily::Global,
            MemoryClass::DsLoad | MemoryClass::DsStore => MemoryFamily::Ds,
            MemoryClass::SmemLoad => MemoryFamily::Smem,
            MemoryClass::Export => MemoryFamily::Export,
        };
        self.pending.push(Pending {
            id: self.next_id, counter,
            defs: if load { insn.defs.clone() } else { vec![] },
            src_locks: if load { vec![] } else { insn.uses.clone() },
            in_order, family, maybe: false, aliases: vec![],
        });
        self.next_id += 1;
    }

    pub fn required(&self, insn: &Instruction) -> Vec<(Counter, u8, Reason)> {
        let mut waits: Vec<(Counter, u8, Reason)> = Vec::new();
        for (index, pending) in self.pending.iter().enumerate() {
            let reason = pending.defs.iter().find_map(|def| {
                insn.uses.iter().find(|use_| def.overlaps(**use_))
                    .map(|_| Reason::Use { reg: def.to_string() })
                    .or_else(|| insn.defs.iter().find(|write| def.overlaps(**write))
                        .map(|_| Reason::Waw { reg: def.to_string() }))
            }).or_else(|| pending.src_locks.iter().find_map(|lock| {
                insn.defs.iter().find(|write| lock.overlaps(**write))
                    .map(|_| Reason::War { reg: lock.to_string() })
            }));
            let Some(reason) = reason else { continue };
            let peers = self.pending.iter().filter(|p| p.counter == pending.counter);
            let ordered = pending.in_order && peers.clone().all(|p| p.in_order && !p.maybe && p.family == pending.family);
            let younger = self.pending.iter().skip(index + 1).filter(|p| p.counter == pending.counter).count();
            let count = if ordered && younger <= 63 { younger as u8 } else { 0 };
            if let Some(wait) = waits.iter_mut().find(|(counter, _, _)| *counter == pending.counter) {
                if count < wait.1 { *wait = (pending.counter, count, reason) }
            } else {
                waits.push((pending.counter, count, reason));
            }
        }
        waits
    }

    /// An operation retires once `count` younger operations of its counter
    /// are certainly issued after it (without joined `maybe` operations:
    /// all but the youngest `count`).
    pub fn wait(&mut self, counter: Counter, count: u8) {
        let mut younger = 0usize;
        let mut keep = vec![true; self.pending.len()];
        for (index, p) in self.pending.iter().enumerate().rev() {
            if p.counter != counter { continue }
            if younger >= usize::from(count) { keep[index] = false }
            if !p.maybe { younger += 1 }
        }
        let mut keep = keep.into_iter();
        self.pending.retain(|_| keep.next().unwrap_or(true));
    }

    /// Join the ledger of another control path into this point, per
    /// operation id: ids say which operation a `Wave::wait` token awaits,
    /// so every id pending on either path stays pending. Ids stay unique
    /// across paths (`resume`), so an equal id is the same operation.
    ///
    /// A counter whose pending operations are in-order loads of one family
    /// on both paths, with equal shapes, retires them by position: the
    /// paths' operations are merged slot by slot, each slot certain and
    /// carrying both paths' ids. On any other counter (stores, SMEM and
    /// other out-of-order families, or different shapes) an operation
    /// pending on only one path is `maybe`: waits on its counter drain to
    /// zero (`required`) and a partial wait retires only what is certain
    /// (`wait`), which holds on both paths.
    pub fn join(&mut self, other: &Ledger) {
        fn on(l: &Ledger, c: Counter) -> Vec<&Pending> { l.pending.iter().filter(|p| p.counter == c).collect() }
        let mut positional = Vec::new();
        for c in self.pending.iter().chain(&other.pending).map(|p| p.counter) {
            if positional.contains(&c) { continue }
            let (mine, theirs) = (on(self, c), on(other, c));
            let family = mine.first().map(|p| p.family);
            let slot = |p: &&Pending| p.positional() && Some(p.family) == family;
            if mine.len() == theirs.len() && mine.iter().chain(&theirs).all(slot)
                && mine.iter().zip(&theirs).all(|(p, o)| p.defs == o.defs) {
                positional.push(c);
            }
        }
        let mut merged: Vec<Pending> = Vec::with_capacity(self.pending.len());
        for c in &positional {
            let theirs = on(other, *c);
            for (p, o) in self.pending.iter().filter(|p| p.counter == *c).zip(theirs) {
                let mut p = p.clone();
                for id in o.ids() { if !p.has(id) { p.aliases.push(id) } }
                merged.push(p);
            }
        }
        for p in self.pending.iter().filter(|p| !positional.contains(&p.counter)) {
            let mut p = p.clone();
            match other.pending.iter().find(|o| o.ids().any(|id| p.has(id))) {
                Some(o) => {
                    p.maybe |= o.maybe;
                    for id in o.ids() { if !p.has(id) { p.aliases.push(id) } }
                }
                None => p.maybe = true,
            }
            merged.push(p);
        }
        for o in other.pending.iter().filter(|o| !positional.contains(&o.counter)) {
            if !self.pending.iter().any(|p| o.ids().any(|id| p.has(id))) {
                merged.push(Pending { maybe: true, ..o.clone() });
            }
        }
        merged.sort_by_key(|p| p.id);
        self.pending = merged;
        self.next_id = self.next_id.max(other.next_id);
    }
    /// Continue at a branch target with the ledger of the branch point.
    /// Ids issued on the path that fell through stay used, so the two
    /// paths' operations keep distinct ids when they join later.
    pub fn resume(&mut self, at: Ledger) {
        let next_id = self.next_id.max(at.next_id);
        *self = at;
        self.next_id = next_id;
    }

    pub fn drain(&mut self) -> Vec<(Counter, u8, Reason)> {
        let mut waits = Vec::new();
        for pending in &self.pending {
            if !waits.iter().any(|(counter, _, _)| *counter == pending.counter) {
                waits.push((pending.counter, 0, Reason::Barrier));
            }
        }
        for (counter, _, _) in &waits { self.wait(*counter, 0) }
        waits
    }

    /// After `s_wait_alu depctr_vm_vsrc(0)` every issued VMEM store has read
    /// its source registers: pending stores lock nothing and define nothing,
    /// so they leave the ledger (completion order is never inferred for them).
    pub fn release_store_sources(&mut self) {
        self.pending.retain(|p| !matches!(p.counter, Counter::Store | Counter::Vs));
    }

    /// An LDS store not yet retired (DScnt on gfx12, LGKMcnt on gfx11): a
    /// barrier that publishes LDS must drain it first.
    pub fn pending_stores(&self) -> bool {
        self.pending.iter().any(|p| matches!(p.counter, Counter::Ds | Counter::Lgkm)
            && p.family == MemoryFamily::Ds && !p.src_locks.is_empty())
    }
    pub fn is_empty(&self) -> bool { self.pending.is_empty() }
    /// Id of the most recently recorded operation.
    pub fn last_id(&self) -> Option<u64> { self.next_id.checked_sub(1) }
    /// Operation `id` (on any joined path) has not been retired by a wait.
    pub fn is_pending(&self, id: u64) -> bool { self.pending.iter().any(|p| p.has(id)) }
    /// Pending operations without their ids: two ledgers with equal shapes
    /// require the same waits for every later instruction.
    pub fn shape(&self) -> Vec<(Counter, Vec<RegRef>, Vec<RegRef>, bool, MemoryFamily, bool)> {
        self.pending.iter().map(|p| (p.counter, p.defs.clone(), p.src_locks.clone(), p.in_order, p.family, p.maybe)).collect()
    }

    pub fn wait_instruction(arch: Arch, waits: &[(Counter, u8, Reason)]) -> Result<Vec<(Counter, u8, String, Reason)>, String> {
        let mut out = Vec::new();
        for (counter, count, reason) in waits {
            let text = match (arch.gfx12(), counter) {
                (true, Counter::Load) => format!("s_wait_loadcnt {count:#x}"),
                (true, Counter::Store) => format!("s_wait_storecnt {count:#x}"),
                (true, Counter::Ds) => format!("s_wait_dscnt {count:#x}"),
                (true, Counter::Km) => format!("s_wait_kmcnt {count:#x}"),
                (false, Counter::Vm) => format!("s_waitcnt vmcnt({count})"),
                (false, Counter::Vs) => format!("s_waitcnt_vscnt null, {count:#x}"),
                (false, Counter::Lgkm) => format!("s_waitcnt lgkmcnt({count})"),
                (_, Counter::Exp) => format!("s_waitcnt expcnt({count})"),
                _ => return Err("counter/architecture mismatch".into()),
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

    fn r(kind: Kind, base: u8) -> RegRef { RegRef { kind, base, len: 1 } }
    /// Two paths issue the same instructions after a fork (distinct ids).
    fn paths(arch: Arch, insns: &[Instruction]) -> (Ledger, Ledger, Vec<u64>, Vec<u64>) {
        let mut a = Ledger::default();
        for i in insns { a.record(arch, i) }
        let ids_a = (0..insns.len() as u64).collect::<Vec<_>>();
        let mut b = Ledger { pending: vec![], next_id: a.next_id };
        for i in insns { b.record(arch, i) }
        let ids_b = ids_a.iter().map(|i| i + a.next_id).collect();
        (a, b, ids_a, ids_b)
    }

    /// gfx11 LGKMcnt shared by DS and SMEM, and gfx12 KMcnt, retire out of
    /// order: an equal-shape join keeps both paths' operations (every id),
    /// marked maybe, so a use drains the counter. In-order loads merge slot
    /// by slot, each slot carrying both ids, and keep exact counts.
    #[test]
    fn equal_shape_join_merges_by_position_only_for_in_order_loads() {
        let ds = Instruction::new("ds_load_b32 v1, v2", vec![r(Kind::V, 1)], vec![r(Kind::V, 2)]).memory(MemoryClass::DsLoad);
        let smem = Instruction::new("s_load_b32 s4, s[0:1], 0x0", vec![r(Kind::S, 4)], vec![]).memory(MemoryClass::SmemLoad);
        let vm = Instruction::new("global_load_b32 v5, v2, s[0:1]", vec![r(Kind::V, 5)], vec![r(Kind::V, 2)]).memory(MemoryClass::VmemLoad);
        let use_v1 = Instruction::new("v_mov_b32 v3, v1", vec![r(Kind::V, 3)], vec![r(Kind::V, 1)]);
        let ds6 = Instruction::new("ds_load_b32 v6, v2", vec![r(Kind::V, 6)], vec![r(Kind::V, 2)]).memory(MemoryClass::DsLoad);
        for (arch, insns, maybe, wait) in [
            (Arch::Gfx1100, vec![ds.clone(), smem.clone(), ds6.clone()], true, 0),
            (Arch::Gfx1201, vec![smem.clone(), smem.clone()], true, 0),
            (Arch::Gfx1100, vec![ds.clone(), vm.clone(), ds6.clone()], false, 1),
        ] {
            let (mut a, b, ids_a, ids_b) = paths(arch, &insns);
            a.join(&b);
            assert!(ids_a.iter().chain(&ids_b).all(|&id| a.is_pending(id)), "{arch:?} lost an id");
            assert_eq!(a.shape().len(), insns.len() * if maybe { 2 } else { 1 }, "{arch:?}");
            let lgkm = a.shape().into_iter().filter(|s| matches!(s.0, Counter::Lgkm | Counter::Km)).all(|s| s.5 == maybe);
            assert!(lgkm, "{arch:?}: slots maybe={maybe}");
            if insns[0].mnemonic().starts_with("ds_") {
                assert_eq!(a.required(&use_v1).first().map(|w| w.1), Some(wait), "{arch:?}");
            }
        }
    }
}
