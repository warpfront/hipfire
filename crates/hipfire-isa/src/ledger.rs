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
            in_order, family,
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
            let ordered = pending.in_order && peers.clone().all(|p| p.in_order && p.family == pending.family);
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

    pub fn wait(&mut self, counter: Counter, count: u8) {
        let mut remaining = self.pending.iter().filter(|p| p.counter == counter).count();
        self.pending.retain(|p| {
            if p.counter != counter { return true }
            let retire = remaining > usize::from(count);
            remaining -= 1;
            !retire
        });
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

    pub fn pending_stores(&self) -> bool {
        self.pending.iter().any(|p| p.counter == Counter::Ds && !p.src_locks.is_empty())
    }
    pub fn is_empty(&self) -> bool { self.pending.is_empty() }

    pub fn wait_instruction(arch: Arch, waits: &[(Counter, u8, Reason)]) -> Result<Vec<(Counter, u8, String, Reason)>, String> {
        let mut out = Vec::new();
        for (counter, count, reason) in waits {
            let text = match (arch.gfx12(), counter) {
                (true, Counter::Load) => format!("s_wait_loadcnt {count}"),
                (true, Counter::Store) => format!("s_wait_storecnt {count}"),
                (true, Counter::Ds) => format!("s_wait_dscnt {count}"),
                (true, Counter::Km) => format!("s_wait_kmcnt {count}"),
                (false, Counter::Vm) => format!("s_waitcnt vmcnt({count})"),
                (false, Counter::Vs) => format!("s_waitcnt_vscnt null, {count}"),
                (false, Counter::Lgkm) => format!("s_waitcnt lgkmcnt({count})"),
                (_, Counter::Exp) => format!("s_waitcnt expcnt({count})"),
                _ => return Err("counter/architecture mismatch".into()),
            };
            out.push((*counter, *count, text, reason.clone()));
        }
        Ok(out)
    }
}
