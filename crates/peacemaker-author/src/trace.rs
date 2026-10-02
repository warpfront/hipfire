//! A reference backend for tests and documentation: it records instruction
//! text and runs a strict slot machine. Unlike the production builder it
//! never inserts a drain on its own: a barrier with an LDS store pending is
//! an error, so a program that passes here got every drain from the types.
//! On gfx12 targets a barrier is the split `s_barrier_signal` /
//! `s_barrier_wait` pair, as the production builder lowers it.

use crate::backend::{Auth, Backend, EventId, Seal, SlotTransition};
use crate::target::LdsCounter;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Free,
    Publishing,
    Published,
    Reading,
}

/// `Trace`'s state at a branch: LDS slots and pending stores.
pub struct TraceFork(Vec<(String, Slot)>, Vec<u64>);

#[derive(Clone, Debug)]
pub struct Trace {
    arch: &'static str,
    /// Emitted instructions and labels, in order.
    pub text: Vec<String>,
    slots: Vec<(String, Slot)>,
    stores: Vec<u64>,
    next: u64,
    signalled: Option<Vec<SlotTransition>>,
    seal: Seal,
    exits: Vec<String>,
    /// Code falls through to the current point (false after `s_branch`).
    reachable: bool,
}

impl Trace {
    pub fn new(arch: &'static str) -> Self {
        Self {
            arch,
            text: Vec::new(),
            slots: Vec::new(),
            stores: Vec::new(),
            next: 0,
            signalled: None,
            seal: Seal::default(),
            exits: Vec::new(),
            reachable: true,
        }
    }
    /// Instructions whose mnemonic is `m`.
    pub fn count(&self, m: &str) -> usize {
        self.text.iter().filter(|t| t.split_whitespace().next() == Some(m)).count()
    }
    /// A raw ALU instruction. LDS, barriers, branches and the program end
    /// belong to the typed core.
    pub fn raw(&mut self, t: impl Into<String>) -> Result<(), String> {
        let t = t.into();
        let m = t.split_whitespace().next().unwrap_or("");
        if m.starts_with("ds_") || m.starts_with("s_barrier") || m.starts_with("s_branch") || m.starts_with("s_cbranch") || m == "s_endpgm" {
            return Err(format!("{m} goes through the typed core"));
        }
        if self.text.last().is_some_and(|l| l == "s_endpgm") {
            return Err("instruction after s_endpgm".into());
        }
        self.emit(t);
        Ok(())
    }
    /// Every reserved exit was placed (the production builder's `finish`
    /// makes the same check).
    pub fn finish(&self) -> Result<(), String> {
        match self.exits.iter().find(|e| !self.text.iter().any(|t| t.strip_suffix(':') == Some(e.as_str()))) {
            Some(e) => Err(format!("kernel exit {e} is never placed")),
            None => Ok(()),
        }
    }
    fn emit(&mut self, t: impl Into<String>) {
        self.text.push(t.into());
    }
    fn gfx12(&self) -> bool {
        self.arch.starts_with("gfx12")
    }
    fn place(&mut self, name: &str) -> Result<(), String> {
        if self.text.iter().any(|t| t.strip_suffix(':') == Some(name)) {
            return Err(format!("duplicate label {name}"));
        }
        self.emit(format!("{name}:"));
        Ok(())
    }
    fn transitions(&mut self, ts: &[SlotTransition], apply: bool) -> Result<(), String> {
        if !self.stores.is_empty() {
            return Err("barrier with an LDS store pending".into());
        }
        for t in ts {
            let (id, from, to) = match *t {
                SlotTransition::Retire(id) => (id, Slot::Reading, Slot::Free),
                SlotTransition::Ready(id) => (id, Slot::Publishing, Slot::Published),
            };
            let slot = self.slots.get_mut(id).ok_or("unknown LDS slot")?;
            if slot.1 != from {
                return Err(format!("{t:?}: {} is {:?}", slot.0, slot.1));
            }
            if apply {
                slot.1 = to;
            }
        }
        Ok(())
    }
    fn signal(&mut self, transitions: &[SlotTransition]) -> Result<(), String> {
        if self.signalled.is_some() {
            return Err("split barrier already in flight".into());
        }
        self.transitions(transitions, false)?;
        self.signalled = Some(transitions.to_vec());
        self.emit("s_barrier_signal -1");
        Ok(())
    }
    fn arrive(&mut self) -> Result<(), String> {
        let ts = self.signalled.take().ok_or("barrier wait without signal")?;
        self.transitions(&ts, true)?;
        self.emit("s_barrier_wait 0xffff");
        Ok(())
    }
}

impl Backend for Trace {
    type Insn = String;
    fn arch_name(&self) -> &'static str {
        self.arch
    }
    fn seal(&mut self, auth: &Auth) -> Result<(), String> {
        self.seal.seal(auth)
    }
    fn position(&self) -> usize {
        self.text.len()
    }
    fn lds_slot(&mut self, auth: &Auth, name: &str, base: u32, len: u32) -> Result<usize, String> {
        self.seal.check(auth)?;
        let _ = (base, len);
        self.slots.push((name.into(), Slot::Free));
        Ok(self.slots.len() - 1)
    }
    fn lds_relayout(&mut self, auth: &Auth) -> Result<(), String> {
        self.seal.check(auth)?;
        if let Some((name, s)) = self.slots.iter().find(|(_, s)| *s != Slot::Free) {
            return Err(format!("{name} is {s:?} at relayout"));
        }
        Ok(())
    }
    fn ds_store(&mut self, auth: &Auth, slots: &[usize], insn: String) -> Result<EventId, String> {
        self.seal.check(auth)?;
        for &id in slots {
            let slot = self.slots.get_mut(id).ok_or("unknown LDS slot")?;
            if !matches!(slot.1, Slot::Free | Slot::Publishing) {
                return Err(format!("{} cannot be stored while {:?}", slot.0, slot.1));
            }
            slot.1 = Slot::Publishing;
        }
        self.emit(insn);
        self.stores.push(self.next);
        self.next += 1;
        Ok(EventId(self.next - 1))
    }
    fn ds_load(&mut self, auth: &Auth, slots: &[usize], insn: String) -> Result<(), String> {
        self.seal.check(auth)?;
        for &id in slots {
            let slot = self.slots.get_mut(id).ok_or("unknown LDS slot")?;
            if !matches!(slot.1, Slot::Published | Slot::Reading) {
                return Err(format!("{} is not published", slot.0));
            }
            slot.1 = Slot::Reading;
        }
        self.emit(insn);
        Ok(())
    }
    fn lds_store_pending(&self, event: EventId) -> bool {
        self.stores.contains(&event.0)
    }
    fn drain_lds_stores(&mut self, auth: &Auth, counter: LdsCounter) -> Result<(), String> {
        self.seal.check(auth)?;
        self.emit(match counter {
            LdsCounter::Lgkm => "s_waitcnt lgkmcnt(0)",
            LdsCounter::Ds => "s_wait_dscnt 0x0",
        });
        self.stores.clear();
        Ok(())
    }
    fn barrier(&mut self, auth: &Auth, transitions: &[SlotTransition]) -> Result<(), String> {
        self.seal.check(auth)?;
        if self.gfx12() {
            self.signal(transitions)?;
            return self.arrive();
        }
        self.transitions(transitions, true)?;
        self.emit("s_barrier");
        Ok(())
    }
    fn barrier_signal(&mut self, auth: &Auth, transitions: &[SlotTransition]) -> Result<(), String> {
        self.seal.check(auth)?;
        if !self.gfx12() {
            return Err("split barriers require gfx12".into());
        }
        self.signal(transitions)
    }
    fn barrier_wait(&mut self, auth: &Auth) -> Result<(), String> {
        self.seal.check(auth)?;
        self.arrive()
    }
    fn label(&mut self, auth: &Auth, name: &str) -> Result<(), String> {
        self.seal.check(auth)?;
        if self.exits.iter().any(|e| e == name) {
            return Err(format!("{name} is a kernel exit: only Workgroup::end places it"));
        }
        self.place(name)
    }
    fn reserve_exit(&mut self, auth: &Auth, name: &str) -> Result<(), String> {
        self.seal.check(auth)?;
        if self.exits.iter().any(|e| e == name) || self.text.iter().any(|t| t.strip_suffix(':') == Some(name)) {
            return Err(format!("duplicate label {name}"));
        }
        self.exits.push(name.into());
        Ok(())
    }
    fn place_exit(&mut self, auth: &Auth, name: &str) -> Result<(), String> {
        self.seal.check(auth)?;
        if !self.exits.iter().any(|e| e == name) {
            return Err(format!("{name} is not a reserved kernel exit"));
        }
        if self.signalled.is_some() {
            return Err("kernel ends with a split barrier in flight".into());
        }
        self.place(name)
    }
    fn end_program(&mut self, auth: &Auth) -> Result<(), String> {
        self.seal.check(auth)?;
        if !self.exits.iter().any(|e| self.text.iter().any(|t| t.strip_suffix(':') == Some(e.as_str()))) {
            return Err("s_endpgm without a placed kernel exit".into());
        }
        self.emit("s_endpgm");
        self.seal.end(auth)
    }
    fn scalar_compare(&mut self, auth: &Auth, insn: String) -> Result<(), String> {
        self.seal.check(auth)?;
        if !(insn.starts_with("s_cmp") || insn.starts_with("s_bitcmp")) {
            return Err(format!("{insn} does not define SCC"));
        }
        self.emit(insn);
        Ok(())
    }
    fn branch_scc1(&mut self, auth: &Auth, target: &str) -> Result<(), String> {
        self.seal.check(auth)?;
        self.emit(format!("s_cbranch_scc1 {target}"));
        Ok(())
    }
    fn branch_scc0(&mut self, auth: &Auth, target: &str) -> Result<(), String> {
        self.seal.check(auth)?;
        self.emit(format!("s_cbranch_scc0 {target}"));
        Ok(())
    }
    fn branch(&mut self, auth: &Auth, target: &str) -> Result<(), String> {
        self.seal.check(auth)?;
        self.emit(format!("s_branch {target}"));
        self.reachable = false;
        Ok(())
    }
    type Fork = TraceFork;
    fn fork(&self) -> Self::Fork {
        TraceFork(self.slots.clone(), self.stores.clone())
    }
    fn resume(&mut self, auth: &Auth, TraceFork(slots, stores): Self::Fork) -> Result<(), String> {
        self.seal.check(auth)?;
        self.slots = slots;
        self.stores = stores;
        self.reachable = true;
        Ok(())
    }
    fn join(&mut self, auth: &Auth, TraceFork(slots, stores): Self::Fork) -> Result<(), String> {
        self.seal.check(auth)?;
        if !self.reachable {
            self.slots = slots;
            self.stores = stores;
            self.reachable = true;
            return Ok(());
        }
        if slots.len() != self.slots.len() {
            return Err("paths join with different LDS layouts".into());
        }
        for (mine, (name, theirs)) in self.slots.iter_mut().zip(slots) {
            mine.1 = match (mine.1, theirs) {
                (a, b) if a == b => a,
                // Read on one path only: either way no other wave reads it
                // unless a barrier retires this wave's reads.
                (Slot::Published | Slot::Reading, Slot::Published | Slot::Reading) => Slot::Reading,
                // Stored on one path only: the region is `Writing` on both,
                // and the next barrier publishes what any wave stored.
                (Slot::Free | Slot::Publishing, Slot::Free | Slot::Publishing) => Slot::Publishing,
                (a, b) => return Err(format!("paths join with {name} {a:?} on one and {b:?} on the other")),
            };
        }
        // A store pending on either path is pending at the join.
        for s in stores {
            if !self.stores.contains(&s) {
                self.stores.push(s);
            }
        }
        self.stores.sort_unstable();
        Ok(())
    }
    fn lds_peer_stores(&mut self, auth: &Auth, slots: &[usize]) -> Result<(), String> {
        self.seal.check(auth)?;
        for &id in slots {
            let slot = self.slots.get_mut(id).ok_or("unknown LDS slot")?;
            if slot.1 != Slot::Free {
                return Err(format!("{} cannot be stored by other waves while {:?}", slot.0, slot.1));
            }
            slot.1 = Slot::Publishing;
        }
        Ok(())
    }
    fn exec_from_scc(&mut self, auth: &Auth) -> Result<(), String> {
        self.seal.check(auth)?;
        self.emit("s_cselect_b32 exec_lo, -1, 0");
        Ok(())
    }
    fn exec_all(&mut self, auth: &Auth) -> Result<(), String> {
        self.seal.check(auth)?;
        self.emit("s_mov_b32 exec_lo, -1");
        Ok(())
    }
    fn loop_(&mut self, auth: &Auth, head: &str, body: &dyn Fn(&mut Self) -> Result<(), String>) -> Result<(), String> {
        self.seal.check(auth)?;
        let entry = self.stores.len();
        let mut first = self.clone();
        first.place(head)?;
        let start = first.text.len();
        body(&mut first)?;
        if first.stores.len() != entry {
            return Err("loop back edge leaves a different number of LDS stores pending than its entry".into());
        }
        let mut second = first.clone();
        second.text.truncate(start - 1);
        second.place(head)?;
        body(&mut second)?;
        if second.text[start..] != first.text[start..] {
            return Err("loop body did not reach a fixed point".into());
        }
        // The core carries the last run's state (its store events) past the loop.
        *self = second;
        Ok(())
    }
}
