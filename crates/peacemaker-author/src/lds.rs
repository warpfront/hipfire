// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

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
//! 4. A `Ring<R, C, N, D>` of `D` buffers (2 to 4, default 2) is typed per
//!    role (`cur`, `next`; the `D - 2` buffers between them are published
//!    "ahead" data, or vacant while the ring fills), so rotating at a
//!    barrier keeps loop-carried types equal across the back edge. Only
//!    `next` takes stores; a rotation retires `cur`, publishes `next` and
//!    moves `cur` to `(cur + 1) % D`. Which buffers hold data, and which is
//!    `cur`, are build-time values: a read of a vacant `cur` is refused.
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

/// The states a ring may be in when it crosses between a typed body and the
/// runtime `Driver` (`runtime::Lend`): `Free` and `Published`. A `Writing`
/// buffer carries a `Pending` token; it crosses only through
/// `Lend::take_writing` / `Lend::give_writing`, which move the token with the
/// ring and leave the driver's record of the buffer's stores alone.
pub trait Lendable: State {
    #[doc(hidden)]
    const PHASE: crate::runtime::Phase;
}
impl Lendable for Free {
    const PHASE: crate::runtime::Phase = crate::runtime::Phase::RtFree;
}
impl Lendable for Published {
    const PHASE: crate::runtime::Phase = crate::runtime::Phase::RtPublished;
}

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
        Self {
            ids: [id; Self::MAX],
            len: 1,
        }
    }
    pub(crate) fn ids(&self) -> &[usize] {
        &self.ids[..self.len]
    }
    /// Slots from build-time ids (the inverse of `ids`).
    pub(crate) fn from_ids(ids: &[usize]) -> Result<Self, String> {
        if ids.is_empty() || ids.len() > Self::MAX {
            return Err(format!(
                "a ring buffer holds 1 to {} slots, not {}",
                Self::MAX,
                ids.len()
            ));
        }
        let mut out = Self {
            ids: [ids[0]; Self::MAX],
            len: ids.len(),
        };
        out.ids[..ids.len()].copy_from_slice(ids);
        Ok(out)
    }
    fn join(a: Self, b: Self) -> Result<Self, String> {
        if a.len + b.len > Self::MAX {
            return Err(format!(
                "a joined LDS region holds at most {} slots",
                Self::MAX
            ));
        }
        let mut out = a;
        out.ids[a.len..a.len + b.len].copy_from_slice(b.ids());
        out.len += b.len;
        Ok(out)
    }
    fn split(self, first: usize) -> Result<(Self, Self), String> {
        if first == 0 || first >= self.len {
            return Err(format!(
                "cannot split a {}-slot region after {first} slots",
                self.len
            ));
        }
        let mut b = self;
        b.ids.copy_within(first..self.len, 0);
        b.len = self.len - first;
        Ok((
            Self {
                ids: self.ids,
                len: first,
            },
            b,
        ))
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
        Self {
            slots,
            last,
            _p: PhantomData,
        }
    }
    fn to<T: State>(self) -> LdsRegion<R, T> {
        LdsRegion::new(self.slots, self.last)
    }
}

/// Type-only marker of a ring's tag and role states.
type Roles<R, C, N> = PhantomData<fn() -> (R, C, N)>;

/// The deepest ring the typed core builds.
pub const MAX_RING_DEPTH: usize = 4;

/// A `D`-buffer LDS ring (`2 <= D <= 4`, default 2) of regions tagged `R`,
/// typed per role: the current buffer in state `C` and the next one in
/// state `N`. Ring order from `cur`: `cur`, `ahead_1 .. ahead_{D-2}`,
/// `next`; the ahead buffers are `Published` (or vacant while the ring
/// fills) and only `next` takes stores. The physical index of each role is
/// a build-time value (`cur_index`), as a hand kernel keeps its buffer
/// offset in a register; `next_index() = (cur + D - 1) % D`, and a rotation
/// moves `cur` to `(cur + 1) % D`.
pub struct Ring<R, C: State, N: State, const D: usize = 2> {
    bufs: [Slots; D],
    cur: usize,
    /// Per buffer: it holds data published by a barrier and not yet retired
    /// (the current buffer is not live before the ring has filled).
    live: [bool; D],
    /// Some barrier has published a buffer of this ring.
    published: bool,
    /// Youngest store into the next buffer.
    pub(crate) last: Option<EventId>,
    _p: Roles<R, C, N>,
}
impl<R> Ring<R, Free, Free> {
    /// A two-buffer ring over two equal regions; `first` is written first.
    pub fn new(first: LdsRegion<R, Free>, second: LdsRegion<R, Free>) -> Self {
        Self::fresh([first.slots, second.slots])
    }
}
impl<R, const D: usize> Ring<R, Free, Free, D> {
    fn fresh(bufs: [Slots; D]) -> Self {
        Self {
            bufs,
            cur: 1 % D,
            live: [false; D],
            published: false,
            last: None,
            _p: PhantomData,
        }
    }
    /// A `D`-buffer ring over `D` equal regions (`2 <= D <= 4`); the first
    /// store goes to `bufs[0]`, so the initial current buffer is `1 % D`.
    pub fn new_n(bufs: [LdsRegion<R, Free>; D]) -> Result<Self, String> {
        if !(2..=MAX_RING_DEPTH).contains(&D) {
            return Err(format!("a ring has 2 to {MAX_RING_DEPTH} buffers, not {D}"));
        }
        Ok(Self::fresh(bufs.map(|b| b.slots)))
    }
    /// Choose which buffer takes the first store: `cur` is set so that
    /// `next_index() == k`. Refused once the ring has published anything.
    pub fn starting_at(mut self, k: usize) -> Result<Self, String> {
        if k >= D {
            return Err(format!("ring start buffer {k} is outside a depth-{D} ring"));
        }
        if self.published || self.live.iter().any(|&l| l) || self.last.is_some() {
            return Err(format!(
                "a depth-{D} ring can only choose its first buffer before anything is published"
            ));
        }
        self.cur = (k + 1) % D;
        Ok(self)
    }
}
impl<R, N: State, const D: usize> Ring<R, Free, N, D> {
    /// Enter steady-state role typing before a loop whose head barrier
    /// rotates: the vacant current buffer is not retired by the first
    /// `rotate`, and a read of it is refused.
    pub fn into_steady(self) -> Ring<R, Published, N, D> {
        self.to()
    }
}
impl<R, C: State, N: State, const D: usize> Ring<R, C, N, D> {
    /// Number of buffers.
    pub fn depth(&self) -> usize {
        D
    }
    /// Physical index (`0..D`, declaration order) of the current buffer.
    pub fn cur_index(&self) -> usize {
        self.cur
    }
    /// Physical index of the next buffer: `(cur + D - 1) % D`.
    pub fn next_index(&self) -> usize {
        (self.cur + D - 1) % D
    }
    pub(crate) fn cur_slots(&self) -> Option<Slots> {
        self.live[self.cur].then_some(self.bufs[self.cur])
    }
    pub(crate) fn next_slots(&self) -> Slots {
        self.bufs[self.next_index()]
    }
    /// A ring's whole build-time state, for a hand-over to the runtime
    /// `Driver` (`runtime::Lend`).
    pub(crate) fn into_parts(self) -> RingParts {
        RingParts {
            bufs: self.bufs.to_vec(),
            cur: self.cur,
            live: self.live.to_vec(),
            published: self.published,
            last: self.last,
        }
    }
    /// The inverse of `into_parts`; the parts' depth must be `D`.
    pub(crate) fn from_parts(p: RingParts) -> Result<Self, String> {
        let depth = p.bufs.len();
        let (Ok(bufs), Ok(live)) = (p.bufs.try_into(), p.live.try_into()) else {
            return Err(format!(
                "ring has {depth} buffers, the typed body takes {D}"
            ));
        };
        Ok(Ring {
            bufs,
            cur: p.cur,
            live,
            published: p.published,
            last: p.last,
            _p: PhantomData,
        })
    }
    pub(crate) fn duplicate(&self) -> Self {
        Ring {
            bufs: self.bufs,
            cur: self.cur,
            live: self.live,
            published: self.published,
            last: self.last,
            _p: PhantomData,
        }
    }
    /// The ring values (`bufs`, `cur`, `live`, `last`) of two paths agree.
    pub(crate) fn agrees(&self, other: &Self) -> bool {
        self.cur == other.cur
            && self.live == other.live
            && self.published == other.published
            && self.last == other.last
            && self
                .bufs
                .iter()
                .zip(&other.bufs)
                .all(|(a, b)| a.ids() == b.ids())
    }
    fn to<C2: State, N2: State>(self) -> Ring<R, C2, N2, D> {
        Ring {
            bufs: self.bufs,
            cur: self.cur,
            live: self.live,
            published: self.published,
            last: self.last,
            _p: PhantomData,
        }
    }
    /// A barrier retired `cur` (if live), published `next`, and moved `cur`
    /// to the buffer after it.
    fn rotated<C2: State, N2: State>(self) -> Ring<R, C2, N2, D> {
        let mut live = self.live;
        live[self.cur] = false;
        live[self.next_index()] = true;
        Ring {
            bufs: self.bufs,
            cur: (self.cur + 1) % D,
            live,
            published: true,
            last: None,
            _p: PhantomData,
        }
    }
}

/// The fields of a `Ring` that cross to and from the runtime `Driver`.
pub(crate) struct RingParts {
    pub(crate) bufs: Vec<Slots>,
    pub(crate) cur: usize,
    pub(crate) live: Vec<bool>,
    pub(crate) published: bool,
    pub(crate) last: Option<EventId>,
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
        (
            LdsRegion::new(self.slots, Some(event)),
            Pending::new(Some(event)),
        )
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
        (
            LdsRegion::new(self.0.slots, Some(event)),
            Pending::new(Some(event)),
        )
    }
}
impl<R, C: State, const D: usize> sealed::Store for Ring<R, C, Free, D> {}
impl<W: WaitModel, R: 'static, C: State, const D: usize> StoreTarget<W> for Ring<R, C, Free, D> {
    type Out = (Ring<R, C, Writing, D>, Pending<W, LdsWrite<R>>);
    fn slots(&self) -> Slots {
        self.next_slots()
    }
    fn begin(self) -> Self::Out {
        (self.to(), Pending::new(None))
    }
    fn stored(self, event: EventId) -> Self::Out {
        let mut ring: Ring<R, C, Writing, D> = self.to();
        ring.last = Some(event);
        (ring, Pending::new(Some(event)))
    }
}
impl<W: WaitModel, R, C: State, const D: usize> sealed::Store
    for (Ring<R, C, Writing, D>, Pending<W, LdsWrite<R>>)
{
}
impl<W: WaitModel, R: 'static, C: State, const D: usize> StoreTarget<W>
    for (Ring<R, C, Writing, D>, Pending<W, LdsWrite<R>>)
{
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
        Self {
            items: [SlotTransition::Retire(0); Self::MAX],
            retire: 0,
            ready: 0,
        }
    }
    fn push(&mut self, t: SlotTransition) -> Result<(), String> {
        let n = self.retire + self.ready;
        if n == Self::MAX {
            return Err(format!(
                "one barrier carries at most {} slot transitions",
                Self::MAX
            ));
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
        s.ids()
            .iter()
            .try_for_each(|&id| self.push(SlotTransition::Retire(id)))
    }
    fn ready(&mut self, s: Slots) -> Result<(), String> {
        s.ids()
            .iter()
            .try_for_each(|&id| self.push(SlotTransition::Ready(id)))
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
pub struct Rotate<R, const D: usize = 2> {
    ring: Ring<R, Published, Writing, D>,
    drained: Drained<LdsWrite<R>>,
}
pub fn rotate<R: 'static, const D: usize>(
    ring: Ring<R, Published, Writing, D>,
    drained: Drained<LdsWrite<R>>,
) -> Rotate<R, D> {
    Rotate { ring, drained }
}
impl<R, const D: usize> sealed::Transition for Rotate<R, D> {}
impl<R: 'static, const D: usize> Transition for Rotate<R, D> {
    type After = Ring<R, Published, Free, D>;
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
pub struct Prime<R, const D: usize = 2> {
    ring: Ring<R, Free, Writing, D>,
    drained: Drained<LdsWrite<R>>,
}
pub fn prime<R: 'static, const D: usize>(
    ring: Ring<R, Free, Writing, D>,
    drained: Drained<LdsWrite<R>>,
) -> Prime<R, D> {
    Prime { ring, drained }
}
impl<R, const D: usize> sealed::Transition for Prime<R, D> {}
impl<R: 'static, const D: usize> Transition for Prime<R, D> {
    type After = Ring<R, Published, Free, D>;
    fn lower(&self, l: &mut Lowering) -> Result<(), String> {
        covers(self.ring.last, self.drained.last)?;
        l.ready(self.ring.next_slots())
    }
    fn after(self) -> Self::After {
        self.ring.rotated()
    }
}

/// End every wave's reads of a ring's current buffer.
pub struct RetireCur<R, N: State, const D: usize = 2> {
    ring: Ring<R, Published, N, D>,
}
pub fn retire_cur<R: 'static, N: State, const D: usize>(
    ring: Ring<R, Published, N, D>,
) -> RetireCur<R, N, D> {
    RetireCur { ring }
}
impl<R, N: State, const D: usize> sealed::Transition for RetireCur<R, N, D> {}
impl<R: 'static, N: State, const D: usize> Transition for RetireCur<R, N, D> {
    type After = Ring<R, Free, N, D>;
    fn lower(&self, l: &mut Lowering) -> Result<(), String> {
        match self.ring.cur_slots() {
            Some(cur) => l.retire(cur),
            None => Ok(()),
        }
    }
    fn after(self) -> Self::After {
        let mut ring: Ring<R, Free, N, D> = self.ring.to();
        ring.live[ring.cur] = false;
        ring
    }
}

/// Move a ring on without a store: retire the current buffer (if live) and
/// make the published `ahead_1` buffer current. Needs a published buffer
/// after `cur` (depth 3 or 4 once the ring has filled); a depth-2 ring never
/// has one, so it is refused at the barrier.
pub struct Advance<R, const D: usize = 2> {
    ring: Ring<R, Published, Free, D>,
}
pub fn advance<R: 'static, const D: usize>(ring: Ring<R, Published, Free, D>) -> Advance<R, D> {
    Advance { ring }
}
impl<R, const D: usize> sealed::Transition for Advance<R, D> {}
impl<R: 'static, const D: usize> Transition for Advance<R, D> {
    type After = Ring<R, Published, Free, D>;
    fn lower(&self, l: &mut Lowering) -> Result<(), String> {
        let ring = &self.ring;
        if !ring.live[(ring.cur + 1) % D] {
            return Err(format!(
                "advance of a depth-{D} ring needs a published buffer after the current one"
            ));
        }
        match ring.cur_slots() {
            Some(cur) => l.retire(cur),
            None => Ok(()),
        }
    }
    fn after(self) -> Self::After {
        let mut ring = self.ring;
        ring.live[ring.cur] = false;
        ring.cur = (ring.cur + 1) % D;
        ring
    }
}

/// The transitions one barrier carries: a tuple of `Transition`s.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a set of barrier transitions",
    note = "pass a tuple of `ready`, `retire`, `rotate`, `prime`, `advance` or `retire_cur` transitions, e.g. `(ready(r, d),)`"
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
/// A barrier that moves no LDS ownership: a pure workgroup rendezvous.
impl Transitions for () {
    type After = ();
    fn lower(&self, _: &mut Lowering) -> Result<(), String> {
        Ok(())
    }
    fn after(self) -> Self::After {}
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
pub fn join<J: 'static, A: JoinPart<J, Free>, B: JoinPart<J, Free>>(
    a: A,
    b: B,
) -> Result<LdsRegion<J, Free>, String> {
    Ok(LdsRegion::new(
        Slots::join(a.part_slots(), b.part_slots())?,
        None,
    ))
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
impl<R, const D: usize> AllFree for Ring<R, Free, Free, D> {}
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
