//! Execution scopes, condition handles and loops (design §4.6-§4.7).
//!
//! - `Workgroup`: workgroup-uniform control. Barriers, LDS layout,
//!   `loop_carried`, `wg_skip_if`, kernel exits and the wave-role `handoff`
//!   live here. It dereferences to `Wave`.
//! - `Wave`: one wave's program. LDS stores/loads, waits, raw ISA, and
//!   wave-uniform scopes (`skip_if`, `exec_if`, `if_else`, `loop_until`)
//!   whose bodies see only a `Wave`, so a barrier under wave- or
//!   lane-dependent control does not compile.
//!
//! A `Workgroup` owns its backend for one session (`backend::Auth`): the
//! backend refuses a second `Workgroup::new`, so a wave scope cannot
//! rebuild a workgroup scope from `Wave::isa`, and raw access through
//! `isa` cannot reach the authorized LDS, barrier, branch and loop entry
//! points.
//!
//! Every place two control paths meet (a skip target, a `Forward` label)
//! joins the backend state of every path into it
//! (`Backend::join`): waits and hazard guards hold on each path, and paths
//! that disagree on LDS ownership are refused. Branches only go forward,
//! except a loop's back edge, so no path re-enters code it has left.
//!
//! Conditions are affine handles: `Uniform<Scc>` (wave-uniform, from any
//! scalar compare) and `WgUniform<Scc>` (workgroup-uniform). A handle is
//! consumed by exactly one branch and refused if SCC was redefined since
//! its compare. In M0 register values are pinned (hand register plans keep
//! byte identity), so the workgroup uniformity of a compare's operands is
//! the caller's claim at `Workgroup::scmp_wg_uniform`, not a derivation.

use crate::backend::{Auth, Backend, SlotTransition};
use crate::lds::{AllFree, LdsRegion, Lowering, Published, Ring, State, StoreTarget, Transitions, Free, Slots, Writing, ready};
use crate::target::{SplitBarrier, Target, WaitModel};
use crate::wait::{Drained, Event, LdsWrite, Pending, Pendings};
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

/// The scalar condition code.
pub enum Scc {}

/// A wave-uniform condition.
#[must_use = "a condition must be consumed by a branch"]
pub struct Uniform<C> {
    at: usize,
    _p: PhantomData<fn() -> C>,
}
/// A workgroup-uniform condition: every wave of the workgroup branches the
/// same way, so the branch may skip or repeat a barrier.
#[must_use = "a condition must be consumed by a branch"]
pub struct WgUniform<C> {
    at: usize,
    _p: PhantomData<fn() -> C>,
}
/// The exit of a `Wave::loop_until` loop: its label and the backend state
/// of every `break_if` that branches to it.
pub struct LoopExit<B: Backend> {
    label: String,
    exits: RefCell<Vec<B::Fork>>,
    /// Breaks are recorded on the loop body's first (kept) emission.
    recording: Cell<bool>,
}
/// A kernel exit: its label followed by `s_endpgm`. `Workgroup::exit_if`
/// and `Workgroup::exit_unless` branch to it; `Workgroup::end` places it,
/// after which nothing more may be emitted, and the backend refuses to
/// finish a kernel that never placed it.
#[must_use = "a kernel exit must be placed with `Workgroup::end`"]
pub struct End {
    label: String,
    /// A branch reaches the exit (its state is not joined at the label).
    branched: Cell<bool>,
}
impl End {
    pub fn label(&self) -> &str {
        &self.label
    }
}
fn fresh<B: Backend>(b: &B, at: usize) -> Result<(), String> {
    if b.position() != at {
        return Err("SCC was redefined between its compare and the branch that consumes it".into());
    }
    Ok(())
}

mod sealed {
    pub trait Sealed {}
    /// Only this crate can respawn carried state.
    pub struct Token;
}
use sealed::Token;

/// State carried around a loop or across a join. The loop body is emitted
/// twice (the backend's fixpoint check), so the core respawns the
/// back-edge state for the second emission; nothing else can duplicate a
/// token or region.
pub trait Carried: sealed::Sealed + Sized {
    #[doc(hidden)]
    fn respawn(&self, token: &Token) -> Self;
}
impl sealed::Sealed for () {}
impl Carried for () {
    fn respawn(&self, _: &Token) -> Self {}
}
impl<R, S: State> sealed::Sealed for LdsRegion<R, S> {}
impl<R, S: State> Carried for LdsRegion<R, S> {
    fn respawn(&self, _: &Token) -> Self {
        LdsRegion::new(self.slots, self.last)
    }
}
impl<R, C: State, N: State> sealed::Sealed for Ring<R, C, N> {}
impl<R, C: State, N: State> Carried for Ring<R, C, N> {
    fn respawn(&self, _: &Token) -> Self {
        self.duplicate()
    }
}
impl<W: WaitModel, E: Event> sealed::Sealed for Pending<W, E> {}
impl<W: WaitModel, E: Event> Carried for Pending<W, E> {
    fn respawn(&self, _: &Token) -> Self {
        Pending::new(self.last)
    }
}
impl<E: Event> sealed::Sealed for Drained<E> {}
impl<E: Event> Carried for Drained<E> {
    fn respawn(&self, _: &Token) -> Self {
        Drained::new(self.last)
    }
}
macro_rules! carried_tuple {
    ($($t:ident),+) => {
        impl<$($t: Carried),+> sealed::Sealed for ($($t,)+) {}
        impl<$($t: Carried),+> Carried for ($($t,)+) {
            #[allow(non_snake_case)]
            fn respawn(&self, token: &Token) -> Self {
                let ($($t,)+) = self;
                ($($t.respawn(token),)+)
            }
        }
    };
}
carried_tuple!(A);
carried_tuple!(A, B);
carried_tuple!(A, B, C);
carried_tuple!(A, B, C, D);
carried_tuple!(A, B, C, D, E);
carried_tuple!(A, B, C, D, E, F);
carried_tuple!(A, B, C, D, E, F, G);
carried_tuple!(A, B, C, D, E, F, G, H);

/// One wave of a kernel on target `T`, lowered by backend `B`.
pub struct Wave<'b, T: Target, B: Backend> {
    b: &'b mut B,
    auth: Auth,
    wave_exit: Cell<bool>,
    _t: PhantomData<T>,
}

/// Workgroup-uniform scope of a kernel; dereferences to its `Wave`.
pub struct Workgroup<'b, T: Target, B: Backend> {
    wave: Wave<'b, T, B>,
}
impl<'b, T: Target, B: Backend> Deref for Workgroup<'b, T, B> {
    type Target = Wave<'b, T, B>;
    fn deref(&self) -> &Self::Target {
        &self.wave
    }
}
impl<T: Target, B: Backend> DerefMut for Workgroup<'_, T, B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.wave
    }
}

/// A split barrier in flight: its regions are unreachable until
/// `Workgroup::wait_arrived` hands them back in their new states.
#[must_use = "a signalled barrier must be waited"]
pub struct Arrived<X> {
    x: X,
}

/// The forward branch targets of one `Wave::forward` block. Each target
/// is placed once, after every branch to it, and the state there is the
/// join of every path that reaches it.
pub struct Forward<B: Backend> {
    targets: Vec<(String, Vec<B::Fork>, bool)>,
}
impl<B: Backend> Forward<B> {
    fn open(&mut self, label: &str) -> Result<usize, String> {
        match self.targets.iter().position(|(l, _, _)| l == label) {
            Some(i) if self.targets[i].2 => Err(format!("branch to {label} after it was placed: branches only go forward")),
            Some(i) => Ok(i),
            None => {
                self.targets.push((label.into(), Vec::new(), false));
                Ok(self.targets.len() - 1)
            }
        }
    }
    /// `s_cbranch_scc1 label`: when `cond` holds, continue at `label`.
    pub fn branch_if<T: Target>(&mut self, w: &mut Wave<'_, T, B>, cond: Uniform<Scc>, label: &str) -> Result<(), String> {
        fresh(w.b, cond.at)?;
        let i = self.open(label)?;
        w.b.branch_scc1(&w.auth, label)?;
        self.targets[i].1.push(w.b.fork());
        Ok(())
    }
    /// `s_cbranch_scc0 label`: when `cond` does not hold, continue at `label`.
    pub fn branch_unless<T: Target>(&mut self, w: &mut Wave<'_, T, B>, cond: Uniform<Scc>, label: &str) -> Result<(), String> {
        fresh(w.b, cond.at)?;
        let i = self.open(label)?;
        w.b.branch_scc0(&w.auth, label)?;
        self.targets[i].1.push(w.b.fork());
        Ok(())
    }
    /// `s_branch label`: nothing falls through past it; the next `place`
    /// continues from the paths branching to its label.
    pub fn goto<T: Target>(&mut self, w: &mut Wave<'_, T, B>, label: &str) -> Result<(), String> {
        let i = self.open(label)?;
        w.b.branch(&w.auth, label)?;
        self.targets[i].1.push(w.b.fork());
        Ok(())
    }
    /// Place `label`: its state joins every branch to it with the code
    /// falling through (if any).
    pub fn place<T: Target>(&mut self, w: &mut Wave<'_, T, B>, label: &str) -> Result<(), String> {
        let (_, forks, placed) = self
            .targets
            .iter_mut()
            .find(|(l, _, _)| l == label)
            .ok_or_else(|| format!("{label} is not the target of a branch in this block"))?;
        if *placed {
            return Err(format!("{label} is already placed"));
        }
        *placed = true;
        for fork in forks.drain(..) {
            w.b.join(&w.auth, fork).map_err(|e| format!("the paths cannot join at {label}: {e}"))?;
        }
        w.b.label(&w.auth, label)
    }
}

impl<'b, T: Target, B: Backend> Wave<'b, T, B> {
    /// The backend for raw instructions (ALU, VMEM, SMEM, crossbar). Its
    /// LDS, barrier, branch and loop entry points need an `Auth` this scope
    /// never hands out, and a `Workgroup` cannot be rebuilt over it.
    pub fn isa(&mut self) -> &mut B {
        self.b
    }
    /// A `Free` region or ring buffer as `Writing` with an empty token, so a
    /// sequence of stores can be folded by passing the pair back.
    pub fn begin_write<X: StoreTarget<T::Waits>>(&mut self, x: X) -> X::Out {
        x.begin()
    }
    /// One LDS store into `x`; returns the writing state and its token.
    pub fn ds_store<X: StoreTarget<T::Waits>>(&mut self, x: X, insn: B::Insn) -> Result<X::Out, String> {
        let e = self.b.ds_store(&self.auth, x.slots().ids(), insn)?;
        Ok(x.stored(e))
    }
    /// One LDS store covering two regions (one instruction, two slots).
    pub fn ds_store2<X: StoreTarget<T::Waits>, Y: StoreTarget<T::Waits>>(
        &mut self,
        x: X,
        y: Y,
        insn: B::Insn,
    ) -> Result<(X::Out, Y::Out), String> {
        let (sx, sy) = (x.slots(), y.slots());
        let mut ids = [0usize; 8];
        let n = sx.ids().len() + sy.ids().len();
        ids[..sx.ids().len()].copy_from_slice(sx.ids());
        ids[sx.ids().len()..n].copy_from_slice(sy.ids());
        let e = self.b.ds_store(&self.auth, &ids[..n], insn)?;
        Ok((x.stored(e), y.stored(e)))
    }
    /// One LDS load from a published region. Its destination registers
    /// are awaited by the backend on first use.
    pub fn ds_load<R>(&mut self, r: &LdsRegion<R, Published>, insn: B::Insn) -> Result<(), String> {
        self.b.ds_load(&self.auth, r.slots.ids(), insn)
    }
    /// One LDS load from a ring's current buffer.
    pub fn ds_load_cur<R, N: State>(&mut self, r: &Ring<R, Published, N>, insn: B::Insn) -> Result<(), String> {
        let slots: Slots = r.cur_slots().ok_or("read of a ring buffer no barrier has published")?;
        self.b.ds_load(&self.auth, slots.ids(), insn)
    }
    /// Wait for a token's operations; emits the counter wait the backend
    /// still needs (none when an earlier wait retired them).
    pub fn wait<E: Event>(&mut self, p: Pending<T::Waits, E>) -> Result<Drained<E>, String> {
        if p.last.is_some_and(|e| self.b.lds_store_pending(e)) {
            self.b.drain_lds_stores(&self.auth, <T::Waits as WaitModel>::LDS)?;
        }
        Ok(Drained::new(p.last))
    }
    /// Wait for several tokens with one counter wait.
    pub fn wait_all<P: Pendings<T::Waits>>(&mut self, p: P) -> Result<P::Drained, String> {
        let mut pending = false;
        let b = &*self.b;
        p.each_event(&mut |e| pending |= b.lds_store_pending(e));
        if pending {
            self.b.drain_lds_stores(&self.auth, <T::Waits as WaitModel>::LDS)?;
        }
        Ok(p.drained())
    }
    /// A scalar compare (`s_cmp*`, `s_bitcmp*`): SCC is wave-uniform.
    pub fn scmp(&mut self, insn: B::Insn) -> Result<Uniform<Scc>, String> {
        self.b.scalar_compare(&self.auth, insn)?;
        Ok(Uniform { at: self.b.position(), _p: PhantomData })
    }
    /// Branch over `body` to `target` when `cond` holds. Both paths reach
    /// `target` with `state`'s type, and the backend joins the branch
    /// point's state into the body's; the body cannot reach a barrier.
    pub fn skip_if<S>(
        &mut self,
        cond: Uniform<Scc>,
        target: &str,
        state: S,
        body: impl FnOnce(&mut Self, S) -> Result<S, String>,
    ) -> Result<S, String> {
        fresh(self.b, cond.at)?;
        self.b.branch_scc1(&self.auth, target)?;
        let skipped = self.b.fork();
        let state = body(self, state)?;
        self.b.join(&self.auth, skipped)?;
        self.b.label(&self.auth, target)?;
        Ok(state)
    }
    /// Skip `body` when `cond` does not hold, joining both paths at `target`.
    pub fn skip_unless<S>(
        &mut self,
        cond: Uniform<Scc>,
        target: &str,
        state: S,
        body: impl FnOnce(&mut Self, S) -> Result<S, String>,
    ) -> Result<S, String> {
        fresh(self.b, cond.at)?;
        self.b.branch_scc0(&self.auth, target)?;
        let skipped = self.b.fork();
        let state = body(self, state)?;
        self.b.join(&self.auth, skipped)?;
        self.b.label(&self.auth, target)?;
        Ok(state)
    }
    /// Leave through the reserved kernel exit unless `cond` holds.
    /// No workgroup barrier may follow a wave-uniform exit.
    pub fn exit_unless(&mut self, cond: Uniform<Scc>, end: &End) -> Result<(), String> {
        fresh(self.b, cond.at)?;
        self.b.branch_scc0(&self.auth, &end.label)?;
        end.branched.set(true);
        self.wave_exit.set(true);
        Ok(())
    }
    /// Run `body` with EXEC = all lanes if `cond`, else none, then restore
    /// EXEC. Every instruction still issues, so state flows straight
    /// through; the body cannot reach a barrier.
    pub fn exec_if<R>(&mut self, cond: Uniform<Scc>, body: impl FnOnce(&mut Self) -> Result<R, String>) -> Result<R, String> {
        fresh(self.b, cond.at)?;
        self.b.exec_from_scc(&self.auth)?;
        let out = body(self)?;
        self.b.exec_all(&self.auth)?;
        Ok(out)
    }
    /// A block of wave-uniform forward branches (`Forward`): the body cannot
    /// reach a barrier, and every target it branches to must be placed
    /// inside it, after the branch.
    pub fn forward<R>(&mut self, body: impl FnOnce(&mut Self, &mut Forward<B>) -> Result<R, String>) -> Result<R, String> {
        let mut f = Forward { targets: Vec::new() };
        let out = body(self, &mut f)?;
        if let Some((label, _, _)) = f.targets.iter().find(|(_, _, placed)| !placed) {
            return Err(format!("forward target {label} is never placed"));
        }
        Ok(out)
    }
    /// A loop at `head` whose back edge is `s_branch head`; it is left only
    /// by `break_if` branches to `exit`, which labels the code after it. The
    /// body is wave scope (no barrier) and carries no typed state; the
    /// backend checks its state reaches a fixed point. The code after the
    /// loop continues from the join of every `break_if`'s state (the back
    /// edge never reaches it), so a wait or LDS transition the body makes
    /// after a break is not assumed at the exit.
    pub fn loop_until(
        &mut self,
        head: &str,
        exit: &str,
        body: impl for<'x> Fn(&mut Wave<'x, T, B>, &LoopExit<B>) -> Result<(), String>,
    ) -> Result<(), String> {
        let x = LoopExit { label: exit.into(), exits: RefCell::new(Vec::new()), recording: Cell::new(true) };
        let auth = &self.auth;
        let emit = |b: &mut B| -> Result<(), String> {
            // Keep exits from the final fixed-point emission, not the
            // first run whose loop-head state has not yet been joined.
            x.exits.borrow_mut().clear();
            x.recording.set(true);
            let mut w = Wave { b, auth: auth.reenter(), wave_exit: Cell::new(self.wave_exit.get()), _t: PhantomData };
            let out = body(&mut w, &x);
            self.wave_exit.set(w.wave_exit.get());
            out?;
            x.recording.set(false);
            w.b.branch(&w.auth, head)
        };
        self.b.loop_(&self.auth, head, &emit)?;
        let mut exits = x.exits.into_inner().into_iter();
        let first = exits.next().ok_or_else(|| format!("loop {head} has no break_if: it never reaches {exit}"))?;
        self.b.resume(&self.auth, first)?;
        for other in exits {
            self.b.join(&self.auth, other)?;
        }
        self.b.label(&self.auth, exit)
    }
    /// Leave the enclosing `loop_until` when `cond` holds; the exit
    /// continues from this point's state (joined with the other breaks').
    pub fn break_if(&mut self, cond: Uniform<Scc>, exit: &LoopExit<B>) -> Result<(), String> {
        fresh(self.b, cond.at)?;
        self.b.branch_scc1(&self.auth, &exit.label)?;
        if exit.recording.get() {
            exit.exits.borrow_mut().push(self.b.fork());
        }
        Ok(())
    }
    /// Two-way branch: when `cond` holds, jump to `else_label` and run
    /// `els` from the state of the branch point; otherwise run `then`,
    /// which ends with `s_branch join`. The caller places `join` right
    /// after (the else arm falls through to it). The backend joins the
    /// arms' states there; neither arm can reach a barrier.
    pub fn if_else(
        &mut self,
        cond: Uniform<Scc>,
        else_label: &str,
        join: &str,
        then: impl FnOnce(&mut Self) -> Result<(), String>,
        els: impl FnOnce(&mut Self) -> Result<(), String>,
    ) -> Result<(), String> {
        fresh(self.b, cond.at)?;
        self.b.branch_scc1(&self.auth, else_label)?;
        let at = self.b.fork();
        then(self)?;
        self.b.branch(&self.auth, join)?;
        let then_end = self.b.fork();
        self.b.resume(&self.auth, at)?;
        self.b.label(&self.auth, else_label)?;
        els(self)?;
        self.b.join(&self.auth, then_end).map_err(|e| format!("the arms cannot join at {join}: {e}"))
    }
    /// A label in straight-line code (a register lifetime boundary).
    pub fn label(&mut self, name: &str) -> Result<(), String> {
        self.b.label(&self.auth, name)
    }
}

impl<'b, T: Target, B: Backend> Workgroup<'b, T, B> {
    fn check_barrier_scope(&self) -> Result<(), String> {
        if self.wave.wave_exit.get() { Err("workgroup barrier after a wave-uniform kernel exit".into()) } else { Ok(()) }
    }
    /// Take ownership of a backend built for `T`. From here on the
    /// backend's untyped LDS, barrier and loop entry points refuse, and so
    /// does a second `Workgroup::new` over it.
    pub fn new(b: &'b mut B) -> Result<Self, String> {
        if b.arch_name() != T::NAME {
            return Err(format!("backend targets {}, kernel is typed for {}", b.arch_name(), T::NAME));
        }
        let auth = Auth::mint();
        b.seal(&auth)?;
        Ok(Self { wave: Wave { b, auth, wave_exit: Cell::new(false), _t: PhantomData } })
    }
    /// Declare one LDS region (one backend slot; ids follow declaration order).
    pub fn lds<R: 'static>(&mut self, name: &str, base: u32, len: u32) -> Result<LdsRegion<R, Free>, String> {
        if base.checked_add(len).is_none_or(|end| end > T::LDS_BYTES) {
            return Err(format!("LDS region {name} [{base}, +{len}) exceeds {} bytes on {}", T::LDS_BYTES, T::NAME));
        }
        Ok(LdsRegion::new(Slots::one(self.wave.b.lds_slot(&self.wave.auth, name, base, len)?), None))
    }
    /// End the slot layout so a later phase can carve its own. Takes every
    /// region (all `Free`); the backend refuses if any slot is still live.
    pub fn relayout<X: AllFree>(&mut self, regions: X) -> Result<(), String> {
        drop(regions);
        self.wave.b.lds_relayout(&self.wave.auth)
    }
    fn lower<X: Transitions>(x: &X) -> Result<Lowering, String> {
        let mut l = Lowering::new();
        x.lower(&mut l)?;
        Ok(l)
    }
    /// One workgroup barrier carrying the transitions `x`.
    pub fn barrier<X: Transitions>(&mut self, x: X) -> Result<X::After, String> {
        self.check_barrier_scope()?;
        let l = Self::lower(&x)?;
        self.wave.b.barrier(&self.wave.auth, l.list())?;
        Ok(x.after())
    }
    /// Split barrier, first half: this wave has arrived; its regions are
    /// held until `wait_arrived`.
    pub fn signal<X: Transitions>(&mut self, x: X) -> Result<Arrived<X>, String>
    where
        T: SplitBarrier,
    {
        self.check_barrier_scope()?;
        let l = Self::lower(&x)?;
        self.wave.b.barrier_signal(&self.wave.auth, l.list())?;
        Ok(Arrived { x })
    }
    /// Split barrier, second half.
    pub fn wait_arrived<X: Transitions>(&mut self, a: Arrived<X>) -> Result<X::After, String>
    where
        T: SplitBarrier,
    {
        self.check_barrier_scope()?;
        self.wave.b.barrier_wait(&self.wave.auth)?;
        Ok(a.x.after())
    }
    /// A scalar compare whose operands the caller asserts are
    /// workgroup-uniform (kernel arguments, workgroup ids, constants and
    /// counters derived only from them).
    pub fn scmp_wg_uniform(&mut self, insn: B::Insn) -> Result<WgUniform<Scc>, String> {
        self.wave.b.scalar_compare(&self.wave.auth, insn)?;
        Ok(WgUniform { at: self.wave.b.position(), _p: PhantomData })
    }
    /// Branch over `body` (which may hold barriers) to `target` when
    /// `cond` holds; both paths reach `target` with `state`'s type, and the
    /// backend joins the branch point's state into the body's.
    pub fn wg_skip_if<S>(
        &mut self,
        cond: WgUniform<Scc>,
        target: &str,
        state: S,
        body: impl FnOnce(&mut Self, S) -> Result<S, String>,
    ) -> Result<S, String> {
        fresh(self.wave.b, cond.at)?;
        self.wave.b.branch_scc1(&self.wave.auth, target)?;
        let skipped = self.wave.b.fork();
        let state = body(self, state)?;
        self.wave.b.join(&self.wave.auth, skipped)?;
        self.wave.b.label(&self.wave.auth, target)?;
        Ok(state)
    }
    /// Reserve the kernel exit `label` (placed later by `end`).
    pub fn exit(&mut self, label: &str) -> Result<End, String> {
        self.wave.b.reserve_exit(&self.wave.auth, label)?;
        Ok(End { label: label.into(), branched: Cell::new(false) })
    }
    /// Leave the kernel when `cond` holds: skipping every later barrier is
    /// legal only for the whole workgroup, and only to the kernel's exit.
    pub fn exit_if(&mut self, cond: WgUniform<Scc>, end: &End) -> Result<(), String> {
        fresh(self.wave.b, cond.at)?;
        self.wave.b.branch_scc1(&self.wave.auth, &end.label)?;
        end.branched.set(true);
        Ok(())
    }
    /// Unless `cond` holds, run `body` and leave the kernel through `end`
    /// (`s_branch`); when it holds, continue at `target` from the branch
    /// point's state (no path through `body` reaches it).
    pub fn exit_unless(
        &mut self,
        cond: WgUniform<Scc>,
        target: &str,
        end: &End,
        body: impl FnOnce(&mut Self) -> Result<(), String>,
    ) -> Result<(), String> {
        fresh(self.wave.b, cond.at)?;
        self.wave.b.branch_scc1(&self.wave.auth, target)?;
        let taken = self.wave.b.fork();
        body(self)?;
        self.wave.b.branch(&self.wave.auth, &end.label)?;
        end.branched.set(true);
        self.wave.b.join(&self.wave.auth, taken)?;
        self.wave.b.label(&self.wave.auth, target)
    }
    /// Place the kernel exit (`end:` then `s_endpgm`). Nothing may follow.
    pub fn end(&mut self, end: End) -> Result<(), String> {
        self.end_with(end, |_| Ok(()))
    }
    /// Place the kernel exit, run `tail` in wave scope, then `s_endpgm`.
    /// A branch to the exit does not carry its state into `tail`, so an
    /// exit something branches to takes no tail.
    pub fn end_with(&mut self, end: End, tail: impl FnOnce(&mut Wave<'b, T, B>) -> Result<(), String>) -> Result<(), String> {
        self.wave.b.place_exit(&self.wave.auth, &end.label)?;
        let at = self.wave.b.position();
        tail(&mut self.wave)?;
        if end.branched.get() && self.wave.b.position() != at {
            return Err(format!("kernel exit {} is a branch target: it takes no tail", end.label));
        }
        self.wave.b.end_program(&self.wave.auth)
    }
    /// A loop at `head`: `body` receives the carried state and returns the
    /// same type plus the workgroup-uniform condition of its back edge
    /// (`s_cbranch_scc1 head`). Pending tokens and region states are part
    /// of `S`, so a store that reaches the loop head undrained, where a
    /// barrier needs `Drained`, is a type error. The backend still checks
    /// the exact wait state reaches a fixed point; the loop's only exit is
    /// the fall-through of its back edge.
    pub fn loop_carried<S: Carried>(
        &mut self,
        head: &str,
        state: S,
        body: impl for<'x> Fn(&mut Workgroup<'x, T, B>, S) -> Result<(S, WgUniform<Scc>), String>,
    ) -> Result<S, String> {
        let input = RefCell::new(Some(state));
        let last: RefCell<Option<S>> = RefCell::new(None);
        let auth = &self.wave.auth;
        let emit = |b: &mut B| -> Result<(), String> {
            let entry = input.borrow_mut().take();
            let s = match entry {
                Some(s) => s,
                None => last.borrow().as_ref().ok_or("loop re-emitted before its first emission finished")?.respawn(&Token),
            };
            let mut wg = Workgroup { wave: Wave { b, auth: auth.reenter(), wave_exit: Cell::new(self.wave.wave_exit.get()), _t: PhantomData } };
            let out = body(&mut wg, s);
            self.wave.wave_exit.set(wg.wave.wave_exit.get());
            let (out, back) = out?;
            fresh(wg.wave.b, back.at)?;
            wg.wave.b.branch_scc1(&wg.wave.auth, head)?;
            // The backend keeps the last run's code: carry its state out.
            *last.borrow_mut() = Some(out);
            Ok(())
        };
        self.wave.b.loop_(&self.wave.auth, head, &emit)?;
        last.into_inner().ok_or_else(|| format!("loop {head} emitted no body"))
    }
    /// Wave-role LDS handoff, the kernel's last phase. Waves where `readers`
    /// holds branch to `reader_label`; the others (writers) run `write`,
    /// publish `region` at one barrier and branch to the kernel exit `end`.
    /// Readers resume from the state of the branch, meet the writers at a
    /// barrier of their own, and run `read` with the region `Published`;
    /// then `end` is placed and the kernel is over. Each wave crosses
    /// exactly one barrier here, the writers leave the kernel, and the
    /// readers' continuation is wave scope ending at the exit, so no later
    /// barrier can wait on the departed writers.
    pub fn handoff<R: 'static, Out>(
        &mut self,
        readers: Uniform<Scc>,
        reader_label: &str,
        end: End,
        region: LdsRegion<R, Free>,
        write: impl FnOnce(&mut Wave<'_, T, B>, LdsRegion<R, Free>) -> Result<(LdsRegion<R, Writing>, Pending<T::Waits, LdsWrite<R>>), String>,
        read: impl FnOnce(&mut Wave<'_, T, B>, LdsRegion<R, Published>, &End) -> Result<Out, String>,
    ) -> Result<Out, String> {
        fresh(self.wave.b, readers.at)?;
        self.wave.b.branch_scc1(&self.wave.auth, reader_label)?;
        let at = self.wave.b.fork();
        let slots = region.slots;
        let (written, pending) = write(&mut self.wave, region)?;
        let drained = self.wave.wait(pending)?;
        let _writers_leave = self.barrier((ready(written, drained),))?;
        self.wave.b.branch(&self.wave.auth, &end.label)?;
        self.wave.b.resume(&self.wave.auth, at)?;
        self.wave.b.label(&self.wave.auth, reader_label)?;
        self.wave.b.lds_peer_stores(&self.wave.auth, slots.ids())?;
        let ts: Vec<SlotTransition> = slots.ids().iter().map(|&id| SlotTransition::Ready(id)).collect();
        self.check_barrier_scope()?;
        self.wave.b.barrier(&self.wave.auth, &ts)?;
        let out = read(&mut self.wave, LdsRegion::new(slots, None), &end)?;
        self.end(end)?;
        Ok(out)
    }
}
