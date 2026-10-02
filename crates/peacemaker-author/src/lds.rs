//! LDS regions with a workgroup-level ownership state (design §4.5).
//!
//! The arena is split into typed regions (`LdsRegion<R, S>`, one backend slot
//! per declared leaf). The state is the builder's slot machine lifted into
//! types:
//! - `Free`: no wave reads it; a wave may start writing.
//! - `Writing`: this wave has stores into it; nobody may read it yet.
//! - `Published`: every wave may read it; nobody may write it.
//!
//! Rules (each is a rustc error when broken):
//! 1. `Writing -> Published` only at a barrier (`ready`), and only with
//!    `Drained` evidence for the region's stores: `s_barrier` does not wait
//!    for memory counters (the Halo VerifyAttn race).
//! 2. `Published -> Free` only at a barrier (`retire`): "every other wave has
//!    finished reading" is a cross-wave fact (the gfx11 FA2 mailbox race).
//! 3. `join` needs every part `Free`; `Workgroup::relayout` consumes `Free`
//!    regions only.
//! 4. A two-buffer `Ring` is typed per role (`cur`, `next`), so rotating at
//!    a barrier keeps loop-carried types equal across the back edge.
//!
//! Exact addresses and bounds stay value-level (backend slot machine,
//! certifier LDS facts).

use crate::backend::{EventId, SlotTransition};
use crate::target::WaitModel;
use crate::wait::{Drained, LdsWrite, Pending};
use std::marker::PhantomData;

mod sealed {
    pub trait State {}
    pub trait Store {}
    pub trait Join {}
    pub trait Transition {}
}
/// No wave reads the region.
pub enum Free {}
/// This wave has (possibly pending) stores into the region.
pub enum Writing {}
/// Readable by every wave; not writable.
pub enum Published {}
impl sealed::State for Free {}
impl sealed::State for Writing {}
impl sealed::State for Published {}
/// A region ownership state.
pub trait State: sealed::State + 'static {}
impl<S: sealed::State + 'static> State for S {}

/// Backend slots of one region (a declared leaf, or joined leaves).
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct Slots {
    ids: [usize; Slots::MAX],
    len: usize,
}
impl Slots {
    const MAX: usize = 4;
    pub(crate) fn one(id: usize) -> Self {
        Self { ids: [id; Self::MAX], len: 1 }
    }
    pub(crate) fn ids(&self) -> &[usize] {
        &self.ids[..self.len]
    }
    fn join(a: Self, b: Self) -> Result<Self, String> {
        if a.len + b.len > Self::MAX {
            return Err(format!("a joined LDS region holds at most {} slots", Self::MAX));
        }
        let mut out = a;
        out.ids[a.len..a.len + b.len].copy_from_slice(b.ids());
        out.len += b.len;
        Ok(out)
    }
    fn split(self, first: usize) -> Result<(Self, Self), String> {
        if first == 0 || first >= self.len {
            return Err(format!("cannot split a {}-slot region after {first} slots", self.len));
        }
        let mut b = self;
        b.ids.copy_within(first..self.len, 0);
        b.len = self.len - first;
        Ok((Self { ids: self.ids, len: first }, b))
    }
}

/// A typed LDS region `R` in state `S`.
pub struct LdsRegion<R, S: State> {
    pub(crate) slots: Slots,
    /// Youngest store into the region (state `Writing`).
    pub(crate) last: Option<EventId>,
    _p: PhantomData<fn() -> (R, S)>,
}
impl<R, S: State> LdsRegion<R, S> {
    pub(crate) fn new(slots: Slots, last: Option<EventId>) -> Self {
        Self { slots, last, _p: PhantomData }
    }
    fn to<T: State>(self) -> LdsRegion<R, T> {
        LdsRegion::new(self.slots, self.last)
    }
}

/// Type-only marker of a ring's tag and role states.
type Roles<R, C, N> = PhantomData<fn() -> (R, C, N)>;

/// A two-buffer LDS ring of regions tagged `R`, typed per role: the
/// current buffer in state `C` and the next one in state `N`. The physical
/// index of each role is a build-time value (`cur_index`), as a hand kernel
/// keeps its buffer offset in a register.
pub struct Ring<R, C: State, N: State> {
    bufs: [Slots; 2],
    cur: usize,
    /// The current buffer holds data published by a barrier (false only
    /// before the first rotation of `into_steady`).
    cur_live: bool,
    /// Youngest store into the next buffer.
    pub(crate) last: Option<EventId>,
    _p: Roles<R, C, N>,
}
impl<R> Ring<R, Free, Free> {
    /// A ring over two equal regions; `first` is written first.
    pub fn new(first: LdsRegion<R, Free>, second: LdsRegion<R, Free>) -> Self {
        Self { bufs: [first.slots, second.slots], cur: 1, cur_live: false, last: None, _p: PhantomData }
    }
}
impl<R, N: State> Ring<R, Free, N> {
    /// Enter steady-state role typing before a loop whose head barrier
    /// rotates: the vacant current buffer is not retired by the first
    /// `rotate`, and a read of it is refused.
    pub fn into_steady(self) -> Ring<R, Published, N> {
        self.to()
    }
}
impl<R, C: State, N: State> Ring<R, C, N> {
    /// Physical index (0 or 1, declaration order) of the current buffer.
    pub fn cur_index(&self) -> usize {
        self.cur
    }
    /// Physical index of the next buffer.
    pub fn next_index(&self) -> usize {
        1 - self.cur
    }
    pub(crate) fn cur_slots(&self) -> Option<Slots> {
        self.cur_live.then_some(self.bufs[self.cur])
    }
    pub(crate) fn next_slots(&self) -> Slots {
        self.bufs[1 - self.cur]
    }
    pub(crate) fn duplicate(&self) -> Self {
        Ring { bufs: self.bufs, cur: self.cur, cur_live: self.cur_live, last: self.last, _p: PhantomData }
    }
    fn to<C2: State, N2: State>(self) -> Ring<R, C2, N2> {
        Ring { bufs: self.bufs, cur: self.cur, cur_live: self.cur_live, last: self.last, _p: PhantomData }
    }
    fn rotated<C2: State, N2: State>(self) -> Ring<R, C2, N2> {
        Ring { bufs: self.bufs, cur: 1 - self.cur, cur_live: true, last: None, _p: PhantomData }
    }
}

/// Something an LDS store may write: a `Free` region (or a ring's `Free`
/// next buffer), or one already `Writing` together with its pending token.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot take an LDS store",
    label = "not writable here",
    note = "a store needs a `Free` region, or a `Writing` region passed together with its `Pending` token; a `Published` region may still be read by other waves until a barrier retires it"
)]
pub trait StoreTarget<W: WaitModel>: sealed::Store {
    /// The writing state with its token: `(region, Pending)`.
    type Out;
    #[doc(hidden)]
    fn slots(&self) -> Slots;
    #[doc(hidden)]
    fn begin(self) -> Self::Out;
    #[doc(hidden)]
    fn stored(self, event: EventId) -> Self::Out;
}
impl<R> sealed::Store for LdsRegion<R, Free> {}
impl<W: WaitModel, R: 'static> StoreTarget<W> for LdsRegion<R, Free> {
    type Out = (LdsRegion<R, Writing>, Pending<W, LdsWrite<R>>);
    fn slots(&self) -> Slots {
        self.slots
    }
    fn begin(self) -> Self::Out {
        (self.to(), Pending::new(None))
    }
    fn stored(self, event: EventId) -> Self::Out {
        (LdsRegion::new(self.slots, Some(event)), Pending::new(Some(event)))
    }
}
impl<W: WaitModel, R> sealed::Store for (LdsRegion<R, Writing>, Pending<W, LdsWrite<R>>) {}
impl<W: WaitModel, R: 'static> StoreTarget<W> for (LdsRegion<R, Writing>, Pending<W, LdsWrite<R>>) {
    type Out = Self;
    fn slots(&self) -> Slots {
        self.0.slots
    }
    fn begin(self) -> Self {
        self
    }
    fn stored(self, event: EventId) -> Self {
        (LdsRegion::new(self.0.slots, Some(event)), Pending::new(Some(event)))
    }
}
impl<R, C: State> sealed::Store for Ring<R, C, Free> {}
impl<W: WaitModel, R: 'static, C: State> StoreTarget<W> for Ring<R, C, Free> {
    type Out = (Ring<R, C, Writing>, Pending<W, LdsWrite<R>>);
    fn slots(&self) -> Slots {
        self.next_slots()
    }
    fn begin(self) -> Self::Out {
        (self.to(), Pending::new(None))
    }
    fn stored(self, event: EventId) -> Self::Out {
        let mut ring: Ring<R, C, Writing> = self.to();
        ring.last = Some(event);
        (ring, Pending::new(Some(event)))
    }
}
impl<W: WaitModel, R, C: State> sealed::Store for (Ring<R, C, Writing>, Pending<W, LdsWrite<R>>) {}
impl<W: WaitModel, R: 'static, C: State> StoreTarget<W> for (Ring<R, C, Writing>, Pending<W, LdsWrite<R>>) {
    type Out = Self;
    fn slots(&self) -> Slots {
        self.0.next_slots()
    }
    fn begin(self) -> Self {
        self
    }
    fn stored(self, event: EventId) -> Self {
        let mut ring = self.0;
        ring.last = Some(event);
        (ring, Pending::new(Some(event)))
    }
}

/// Retire and Ready slot lists of one barrier.
pub struct Lowering {
    items: [SlotTransition; Lowering::MAX],
    retire: usize,
    ready: usize,
}
impl Lowering {
    const MAX: usize = 32;
    pub(crate) fn new() -> Self {
        Self { items: [SlotTransition::Retire(0); Self::MAX], retire: 0, ready: 0 }
    }
    fn push(&mut self, t: SlotTransition) -> Result<(), String> {
        let n = self.retire + self.ready;
        if n == Self::MAX {
            return Err(format!("one barrier carries at most {} slot transitions", Self::MAX));
        }
        // Retires first, then readies, each in transition order: the slot
        // list every hand-written builder barrier passes.
        match t {
            SlotTransition::Retire(_) => {
                self.items.copy_within(self.retire..n, self.retire + 1);
                self.items[self.retire] = t;
                self.retire += 1;
            }
            SlotTransition::Ready(_) => {
                self.items[n] = t;
                self.ready += 1;
            }
        }
        Ok(())
    }
    fn retire(&mut self, s: Slots) -> Result<(), String> {
        s.ids().iter().try_for_each(|&id| self.push(SlotTransition::Retire(id)))
    }
    fn ready(&mut self, s: Slots) -> Result<(), String> {
        s.ids().iter().try_for_each(|&id| self.push(SlotTransition::Ready(id)))
    }
    pub(crate) fn list(&self) -> &[SlotTransition] {
        &self.items[..self.retire + self.ready]
    }
}

fn covers(region_last: Option<EventId>, drained: Option<EventId>) -> Result<(), String> {
    if region_last != drained {
        return Err("drain evidence does not cover the region's youngest store".into());
    }
    Ok(())
}

/// One barrier-carried transition.
pub trait Transition: sealed::Transition {
    type After;
    #[doc(hidden)]
    fn lower(&self, l: &mut Lowering) -> Result<(), String>;
    #[doc(hidden)]
    fn after(self) -> Self::After;
}

/// Publish a drained region: `Writing -> Published`.
pub struct Ready<R> {
    region: LdsRegion<R, Writing>,
    drained: Drained<LdsWrite<R>>,
}
pub fn ready<R: 'static>(region: LdsRegion<R, Writing>, drained: Drained<LdsWrite<R>>) -> Ready<R> {
    Ready { region, drained }
}
impl<R> sealed::Transition for Ready<R> {}
impl<R: 'static> Transition for Ready<R> {
    type After = LdsRegion<R, Published>;
    fn lower(&self, l: &mut Lowering) -> Result<(), String> {
        covers(self.region.last, self.drained.last)?;
        l.ready(self.region.slots)
    }
    fn after(self) -> Self::After {
        self.region.to()
    }
}

/// End every wave's reads of a region: `Published -> Free`.
pub struct Retire<R> {
    region: LdsRegion<R, Published>,
}
pub fn retire<R: 'static>(region: LdsRegion<R, Published>) -> Retire<R> {
    Retire { region }
}
impl<R> sealed::Transition for Retire<R> {}
impl<R: 'static> Transition for Retire<R> {
    type After = LdsRegion<R, Free>;
    fn lower(&self, l: &mut Lowering) -> Result<(), String> {
        l.retire(self.region.slots)
    }
    fn after(self) -> Self::After {
        LdsRegion::new(self.region.slots, None)
    }
}

/// Hand a ring over: retire the current buffer, publish the drained next
/// one, and swap the roles.
pub struct Rotate<R> {
    ring: Ring<R, Published, Writing>,
    drained: Drained<LdsWrite<R>>,
}
pub fn rotate<R: 'static>(ring: Ring<R, Published, Writing>, drained: Drained<LdsWrite<R>>) -> Rotate<R> {
    Rotate { ring, drained }
}
impl<R> sealed::Transition for Rotate<R> {}
impl<R: 'static> Transition for Rotate<R> {
    type After = Ring<R, Published, Free>;
    fn lower(&self, l: &mut Lowering) -> Result<(), String> {
        covers(self.ring.last, self.drained.last)?;
        if let Some(cur) = self.ring.cur_slots() {
            l.retire(cur)?;
        }
        l.ready(self.ring.next_slots())
    }
    fn after(self) -> Self::After {
        self.ring.rotated()
    }
}

/// First publication of a fresh ring: publish the drained next buffer and
/// swap the roles (nothing to retire).
pub struct Prime<R> {
    ring: Ring<R, Free, Writing>,
    drained: Drained<LdsWrite<R>>,
}
pub fn prime<R: 'static>(ring: Ring<R, Free, Writing>, drained: Drained<LdsWrite<R>>) -> Prime<R> {
    Prime { ring, drained }
}
impl<R> sealed::Transition for Prime<R> {}
impl<R: 'static> Transition for Prime<R> {
    type After = Ring<R, Published, Free>;
    fn lower(&self, l: &mut Lowering) -> Result<(), String> {
        covers(self.ring.last, self.drained.last)?;
        l.ready(self.ring.next_slots())
    }
    fn after(self) -> Self::After {
        self.ring.rotated()
    }
}

/// End every wave's reads of a ring's current buffer.
pub struct RetireCur<R, N: State> {
    ring: Ring<R, Published, N>,
}
pub fn retire_cur<R: 'static, N: State>(ring: Ring<R, Published, N>) -> RetireCur<R, N> {
    RetireCur { ring }
}
impl<R, N: State> sealed::Transition for RetireCur<R, N> {}
impl<R: 'static, N: State> Transition for RetireCur<R, N> {
    type After = Ring<R, Free, N>;
    fn lower(&self, l: &mut Lowering) -> Result<(), String> {
        match self.ring.cur_slots() {
            Some(cur) => l.retire(cur),
            None => Ok(()),
        }
    }
    fn after(self) -> Self::After {
        let mut ring: Ring<R, Free, N> = self.ring.to();
        ring.cur_live = false;
        ring
    }
}

/// The transitions one barrier carries: a tuple of `Transition`s.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a set of barrier transitions",
    note = "pass a tuple of `ready`, `retire`, `rotate`, `prime` or `retire_cur` transitions, e.g. `(ready(r, d),)`"
)]
pub trait Transitions {
    type After;
    #[doc(hidden)]
    fn lower(&self, l: &mut Lowering) -> Result<(), String>;
    #[doc(hidden)]
    fn after(self) -> Self::After;
}
macro_rules! transitions_tuple {
    ($($t:ident),+) => {
        impl<$($t: Transition),+> Transitions for ($($t,)+) {
            type After = ($($t::After,)+);
            #[allow(non_snake_case)]
            fn lower(&self, l: &mut Lowering) -> Result<(), String> {
                let ($($t,)+) = self;
                $($t.lower(l)?;)+
                Ok(())
            }
            #[allow(non_snake_case)]
            fn after(self) -> Self::After {
                let ($($t,)+) = self;
                ($($t.after(),)+)
            }
        }
    };
}
transitions_tuple!(A);
transitions_tuple!(A, B);
transitions_tuple!(A, B, C);
transitions_tuple!(A, B, C, D);
transitions_tuple!(A, B, C, D, E);
transitions_tuple!(A, B, C, D, E, F);
transitions_tuple!(A, B, C, D, E, F, G);
transitions_tuple!(A, B, C, D, E, F, G, H);

/// A part of the joined region `J`, in the state `S` every part must share.
#[diagnostic::on_unimplemented(
    message = "the trait bound `{Self}: JoinPart<{J}, {S}>` is not satisfied",
    label = "not a `{S}` part",
    note = "other waves may still be reading it; only a barrier can retire their reads",
    note = "retire it at a barrier you already have, or add one: `wg.barrier((retire(region),))`"
)]
pub trait JoinPart<J, S: State>: sealed::Join {
    #[doc(hidden)]
    fn part_slots(&self) -> Slots;
}
impl<R> sealed::Join for LdsRegion<R, Free> {}
impl<J, R> JoinPart<J, Free> for LdsRegion<R, Free> {
    fn part_slots(&self) -> Slots {
        self.slots
    }
}
/// Re-carve two `Free` regions into one region `J` (a store or load then
/// covers every slot of both).
pub fn join<J: 'static, A: JoinPart<J, Free>, B: JoinPart<J, Free>>(a: A, b: B) -> Result<LdsRegion<J, Free>, String> {
    Ok(LdsRegion::new(Slots::join(a.part_slots(), b.part_slots())?, None))
}
/// Two `Free` regions split from one.
pub type FreeParts<A, B> = (LdsRegion<A, Free>, LdsRegion<B, Free>);
impl<J: 'static> LdsRegion<J, Free> {
    /// Undo a `join`: the first `first` slots become `A`, the rest `B`.
    pub fn split<A: 'static, B: 'static>(self, first: usize) -> Result<FreeParts<A, B>, String> {
        let (a, b) = self.slots.split(first)?;
        Ok((LdsRegion::new(a, None), LdsRegion::new(b, None)))
    }
}

/// Regions that are all `Free` (consumed by `Workgroup::relayout`).
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not entirely `Free`",
    note = "the LDS can be re-carved only after a barrier has retired every reader of every region"
)]
pub trait AllFree {}
impl<R> AllFree for LdsRegion<R, Free> {}
impl<R> AllFree for Ring<R, Free, Free> {}
macro_rules! all_free_tuple {
    ($($t:ident),+) => { impl<$($t: AllFree),+> AllFree for ($($t,)+) {} };
}
all_free_tuple!(A);
all_free_tuple!(A, B);
all_free_tuple!(A, B, C);
all_free_tuple!(A, B, C, D);
all_free_tuple!(A, B, C, D, E);
all_free_tuple!(A, B, C, D, E, F);
all_free_tuple!(A, B, C, D, E, F, G);
all_free_tuple!(A, B, C, D, E, F, G, H);
