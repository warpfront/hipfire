//! `Builder` as the lowering backend of the typed core
//! (`peacemaker_author::Workgroup<T, Builder>`). Every request maps onto the
//! builder call a hand-written kernel would make, in the same order, so a
//! ported kernel emits the same bytes and the same proof; the builder's
//! ledger, slot machine, hazards and loop fixpoint stay the second evaluator.
//! Each entry point checks the typed core's `Auth` against the session that
//! sealed this builder.
use crate::{Builder, MemoryScope, hazard::{Gfx11Hazards, Gfx12Sgpr}, insn::{Instruction, Sop}, lds::{Lds, SlotState}, ledger::{Counter, Ledger}, reg::RegRef};
use peacemaker_author::{Auth, Backend, EventId, LdsCounter, SlotTransition};

/// Builder state at a branch: the wait ledger, the LDS slot machine and
/// every hazard tracker that decides a later wait or guard. (`s_delay_alu`
/// issue hints are reset by the label each branch target is.)
pub struct BuilderFork { ledger: Ledger, lds: Lds, sgpr: Gfx12Sgpr, trans: Gfx11Hazards, wmma: Vec<RegRef> }

impl Backend for Builder {
    type Insn = Instruction;
    fn arch_name(&self) -> &'static str { self.spec.arch.name() }
    fn seal(&mut self, auth: &Auth) -> Result<(), String> { self.seal.seal(auth) }
    fn position(&self) -> usize { self.program.instructions.len() }
    fn lds_slot(&mut self, auth: &Auth, name: &str, base: u32, len: u32) -> Result<usize, String> { self.seal.check(auth)?; self.lds.add(name, base, len) }
    fn lds_relayout(&mut self, auth: &Auth) -> Result<(), String> { self.seal.check(auth)?; self.lds.relayout() }
    fn ds_store(&mut self, auth: &Auth, slots: &[usize], insn: Instruction) -> Result<EventId, String> {
        self.seal.check(auth)?;
        let (&first, rest) = slots.split_first().ok_or("LDS store without a slot")?;
        let old = self.lds.clone();
        // Every further slot the instruction writes takes its Publishing step first.
        let result = rest.iter().try_for_each(|&slot| self.lds.store(slot)).and_then(|()| self.ds_store_slot(first, insn));
        if result.is_err() { self.lds = old }
        result?;
        self.ledger.last_id().map(EventId).ok_or_else(|| "LDS store left no ledger entry".into())
    }
    fn ds_load(&mut self, auth: &Auth, slots: &[usize], insn: Instruction) -> Result<(), String> {
        self.seal.check(auth)?;
        let (&first, rest) = slots.split_first().ok_or("LDS load without a slot")?;
        let old = self.lds.clone();
        let result = rest.iter().try_for_each(|&slot| self.lds.load(slot)).and_then(|()| self.ds_load_slot(first, insn));
        if result.is_err() { self.lds = old }
        result
    }
    fn lds_store_pending(&self, event: EventId) -> bool { self.ledger.is_pending(event.0) }
    fn drain_lds_stores(&mut self, auth: &Auth, counter: LdsCounter) -> Result<(), String> {
        self.seal.check(auth)?;
        // The drain `barrier` inserted before the typed core owned it.
        self.wait(match counter { LdsCounter::Lgkm => Counter::Lgkm, LdsCounter::Ds => Counter::Ds }, 0)
    }
    fn barrier(&mut self, auth: &Auth, transitions: &[SlotTransition]) -> Result<(), String> { self.seal.check(auth)?; self.full_barrier(transitions, MemoryScope::LdsOnly) }
    fn barrier_signal(&mut self, auth: &Auth, transitions: &[SlotTransition]) -> Result<(), String> { self.seal.check(auth)?; self.signal(transitions) }
    fn barrier_wait(&mut self, auth: &Auth) -> Result<(), String> { self.seal.check(auth)?; self.arrive() }
    fn label(&mut self, auth: &Auth, name: &str) -> Result<(), String> { self.seal.check(auth)?; Builder::label(self, name) }
    fn reserve_exit(&mut self, auth: &Auth, name: &str) -> Result<(), String> {
        self.seal.check(auth)?;
        if self.exits.iter().chain(&self.labels).any(|l| l == name) { return Err(format!("duplicate label {name}")) }
        self.exits.push(name.into());
        Ok(())
    }
    fn place_exit(&mut self, auth: &Auth, name: &str) -> Result<(), String> {
        self.seal.check(auth)?;
        if !self.exits.iter().any(|e| e == name) { return Err(format!("{name} is not a reserved kernel exit")) }
        if self.pending_barrier.is_some() { return Err("kernel ends with a split barrier in flight".into()) }
        self.place_label(name)
    }
    fn end_program(&mut self, auth: &Auth) -> Result<(), String> {
        self.seal.check(auth)?;
        if !self.exits.iter().any(|e| self.labels.contains(e)) { return Err("s_endpgm without a placed kernel exit".into()) }
        self.emit(Sop::End.encode(self.spec.arch)?)?;
        self.seal.end(auth)
    }
    fn scalar_compare(&mut self, auth: &Auth, insn: Instruction) -> Result<(), String> {
        self.seal.check(auth)?;
        let m = insn.mnemonic();
        if !(m.starts_with("s_cmp") || m.starts_with("s_bitcmp")) { return Err(format!("{m} does not define SCC")) }
        self.push(insn)
    }
    fn branch_scc1(&mut self, auth: &Auth, target: &str) -> Result<(), String> { self.seal.check(auth)?; self.emit(Instruction::new(format!("s_cbranch_scc1 {target}"), vec![], vec![])) }
    fn branch_scc0(&mut self, auth: &Auth, target: &str) -> Result<(), String> { self.seal.check(auth)?; self.emit(Instruction::new(format!("s_cbranch_scc0 {target}"), vec![], vec![])) }
    fn branch(&mut self, auth: &Auth, target: &str) -> Result<(), String> {
        self.seal.check(auth)?;
        self.emit(Instruction::new(format!("s_branch {target}"), vec![], vec![]))?;
        self.reachable = false;
        Ok(())
    }
    type Fork = BuilderFork;
    fn fork(&self) -> BuilderFork {
        BuilderFork { ledger: self.ledger.clone(), lds: self.lds.clone(), sgpr: self.hazard.clone(), trans: self.gfx11_hazard.clone(), wmma: self.previous_wmma_dst.clone() }
    }
    fn resume(&mut self, auth: &Auth, at: BuilderFork) -> Result<(), String> {
        self.seal.check(auth)?;
        self.ledger.resume(at.ledger);
        self.lds = at.lds;
        self.hazard = at.sgpr;
        self.gfx11_hazard = at.trans;
        self.previous_wmma_dst = at.wmma;
        self.reachable = true;
        Ok(())
    }
    fn join(&mut self, auth: &Auth, other: BuilderFork) -> Result<(), String> {
        self.seal.check(auth)?;
        if !self.reachable {
            // Nothing falls through: the state is the branch's.
            self.ledger.resume(other.ledger);
            self.lds = other.lds;
            self.hazard = other.sgpr;
            self.gfx11_hazard = other.trans;
            self.previous_wmma_dst = other.wmma;
            self.reachable = true;
            return Ok(());
        }
        self.lds.join(&other.lds)?;
        self.ledger.join(&other.ledger);
        self.join_hazards(&other.sgpr, &other.trans, &other.wmma);
        Ok(())
    }
    fn lds_peer_stores(&mut self, auth: &Auth, slots: &[usize]) -> Result<(), String> {
        self.seal.check(auth)?;
        for &id in slots {
            let slot = self.lds.slots.get_mut(id).ok_or("unknown LDS slot")?;
            if slot.state != SlotState::Free { return Err(format!("{} cannot be stored by other waves while {:?}", slot.name, slot.state)) }
            slot.state = SlotState::Publishing;
        }
        Ok(())
    }
    fn exec_from_scc(&mut self, auth: &Auth) -> Result<(), String> { self.seal.check(auth)?; self.push(Instruction::new("s_cselect_b32 exec_lo, -1, 0", vec![], vec![])) }
    fn exec_all(&mut self, auth: &Auth) -> Result<(), String> { self.seal.check(auth)?; self.push(Instruction::new("s_mov_b32 exec_lo, -1", vec![], vec![])) }
    fn loop_(&mut self, auth: &Auth, head: &str, body: &dyn Fn(&mut Self) -> Result<(), String>) -> Result<(), String> { self.seal.check(auth)?; self.fixpoint_loop(head, body) }
}
