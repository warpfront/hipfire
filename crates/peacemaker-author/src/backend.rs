//! The lowering seam. The typed core decides *what* happens (which LDS
//! slots a store touches, which transitions a barrier carries, when a store
//! is drained); a backend turns each request into instructions and runs the
//! value-level checks behind it (wait ledger, LDS slot machine, hazards,
//! loop fixpoint). `hipfire-isa`'s `Builder` is the production backend;
//! `crate::trace::Trace` is a small reference backend for tests.
//!
//! Every entry point that changes backend state takes an [`Auth`]. Only the
//! typed core mints one, so a kernel that reaches the backend through
//! `Wave::isa` (raw ALU/VMEM/SMEM instructions) cannot call them, and a
//! backend refuses the token of any session but the one that sealed it.

use crate::target::LdsCounter;
use std::sync::atomic::{AtomicU64, Ordering};

/// Build-time identity of one issued memory operation (a backend ledger id).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct EventId(pub u64);

/// One LDS slot transition carried by a barrier (the builder's
/// `lds::Transition`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotTransition {
    /// Every wave finished reading the slot: it may be rewritten.
    Retire(usize),
    /// The slot's drained stores become visible to every wave.
    Ready(usize),
}

/// The typed core's authorization to drive a backend, one session per
/// `Workgroup::new`. It has no public constructor and is neither `Clone`
/// nor `Copy`; backends receive it by reference.
pub struct Auth {
    session: u64,
}
impl Auth {
    pub(crate) fn mint() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self { session: NEXT.fetch_add(1, Ordering::Relaxed) }
    }
    /// The same session for a scope the core re-enters (a loop body).
    pub(crate) fn reenter(&self) -> Self {
        Self { session: self.session }
    }
}

/// A backend's ownership record: the session that sealed it and whether its
/// program has ended. A backend holds one and checks every authorized call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Seal {
    session: Option<u64>,
    ended: bool,
}
impl Seal {
    /// A typed core owns the backend: its untyped LDS, barrier and loop entry
    /// points must refuse.
    pub fn is_sealed(&self) -> bool {
        self.session.is_some()
    }
    /// Hand the backend to `auth`'s session. Refused when another session
    /// already owns it: a `Workgroup` cannot be rebuilt from a wave scope
    /// over the backend its own workgroup already drives.
    pub fn seal(&mut self, auth: &Auth) -> Result<(), String> {
        if self.session.is_some() {
            return Err("backend is already owned by a Workgroup: a second Workgroup over it would escape its scope".into());
        }
        self.session = Some(auth.session);
        Ok(())
    }
    /// `auth` belongs to the owning session and the program has not ended.
    pub fn check(&self, auth: &Auth) -> Result<(), String> {
        if self.session != Some(auth.session) {
            return Err("authorization from a session that does not own this backend".into());
        }
        if self.ended {
            return Err("the kernel already ended: nothing may follow its exit".into());
        }
        Ok(())
    }
    /// The program reached its exit: every later authorized call refuses.
    pub fn end(&mut self, auth: &Auth) -> Result<(), String> {
        self.check(auth)?;
        self.ended = true;
        Ok(())
    }
}

pub trait Backend: Sized {
    /// A raw, register-allocated instruction.
    type Insn;
    /// Architecture name, compared with `Target::NAME`.
    fn arch_name(&self) -> &'static str;
    /// The typed core owns LDS, barriers and loops from now on: the
    /// backend's own untyped entry points for them must refuse, and so must
    /// a second seal (`Seal::seal`).
    fn seal(&mut self, auth: &Auth) -> Result<(), String>;
    /// Program position (instructions emitted so far, waits included).
    fn position(&self) -> usize;

    fn lds_slot(&mut self, auth: &Auth, name: &str, base: u32, len: u32) -> Result<usize, String>;
    /// End the slot layout; every slot must be Free.
    fn lds_relayout(&mut self, auth: &Auth) -> Result<(), String>;
    /// One DS store touching every slot in `slots`; returns its event.
    fn ds_store(&mut self, auth: &Auth, slots: &[usize], insn: Self::Insn) -> Result<EventId, String>;
    /// One DS load reading every slot in `slots`.
    fn ds_load(&mut self, auth: &Auth, slots: &[usize], insn: Self::Insn) -> Result<(), String>;
    /// The LDS store `event` has not been retired by any wait yet.
    fn lds_store_pending(&self, event: EventId) -> bool;
    /// Wait until every issued LDS store has completed (count 0: stores
    /// retire out of order with respect to the same counter's loads).
    fn drain_lds_stores(&mut self, auth: &Auth, counter: LdsCounter) -> Result<(), String>;

    /// Full barrier carrying `transitions`.
    fn barrier(&mut self, auth: &Auth, transitions: &[SlotTransition]) -> Result<(), String>;
    /// gfx12 split barrier: signal.
    fn barrier_signal(&mut self, auth: &Auth, transitions: &[SlotTransition]) -> Result<(), String>;
    /// gfx12 split barrier: wait for the signal in flight.
    fn barrier_wait(&mut self, auth: &Auth) -> Result<(), String>;

    fn label(&mut self, auth: &Auth, name: &str) -> Result<(), String>;
    /// Reserve `name` as a kernel exit: from now on only `place_exit` may
    /// place it, and the backend refuses to finish a program that never
    /// placed it.
    fn reserve_exit(&mut self, auth: &Auth, name: &str) -> Result<(), String>;
    /// Place the reserved exit `name`; only `end_program` and raw
    /// straight-line instructions may follow.
    fn place_exit(&mut self, auth: &Auth, name: &str) -> Result<(), String>;
    /// `s_endpgm` after a placed exit; the program has ended (`Seal::end`).
    fn end_program(&mut self, auth: &Auth) -> Result<(), String>;
    /// Issue a scalar compare (`s_cmp*`, `s_bitcmp*`) that defines SCC.
    fn scalar_compare(&mut self, auth: &Auth, insn: Self::Insn) -> Result<(), String>;
    /// `s_cbranch_scc1 target`.
    fn branch_scc1(&mut self, auth: &Auth, target: &str) -> Result<(), String>;
    /// `s_cbranch_scc0 target`.
    fn branch_scc0(&mut self, auth: &Auth, target: &str) -> Result<(), String>;
    /// `s_branch target`. Nothing reaches the point after it until a
    /// `join` brings in the state of a branch to that point.
    fn branch(&mut self, auth: &Auth, target: &str) -> Result<(), String>;
    /// Backend state at a branch: everything later waits, hazard guards and
    /// slot checks depend on (wait ledger, LDS slots, hazard trackers).
    type Fork;
    fn fork(&self) -> Self::Fork;
    /// Continue at the branch's target with the state of the branch point.
    fn resume(&mut self, auth: &Auth, at: Self::Fork) -> Result<(), String>;
    /// `other` is another path into the current point: make the current
    /// state hold on both, conservatively (every wait or hazard guard either
    /// path needs is emitted), or refuse when the paths disagree on LDS
    /// ownership. When nothing falls through to the current point (it
    /// follows a `branch`), the state is `other`'s.
    fn join(&mut self, auth: &Auth, other: Self::Fork) -> Result<(), String>;
    /// Other waves of the workgroup have drained stores into `slots`, which
    /// the next barrier publishes (the reading side of a wave-role handoff).
    fn lds_peer_stores(&mut self, auth: &Auth, slots: &[usize]) -> Result<(), String>;
    /// EXEC = SCC ? all lanes : none.
    fn exec_from_scc(&mut self, auth: &Auth) -> Result<(), String>;
    /// EXEC = all lanes.
    fn exec_all(&mut self, auth: &Auth) -> Result<(), String>;
    /// Emit `body` once as a loop at `head`; the backend verifies that its
    /// state reaches a fixed point across the back edge. `body` may be run
    /// more than once; the last run's code is the one kept.
    fn loop_(&mut self, auth: &Auth, head: &str, body: &dyn Fn(&mut Self) -> Result<(), String>) -> Result<(), String>;
}
