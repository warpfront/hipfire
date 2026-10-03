//! Runtime-checked driver: the typed core's rules with dynamic region ids
//! (R6: the standalone `.rip` front end elaborates into this, because it
//! cannot spell `LdsRegion<R, Writing>` in source text).
//!
//! `Driver` makes the same `Backend` calls, in the same order, as
//! `Workgroup`/`Wave` do for the same request, so a kernel lowered through
//! it re-emits the bytes of its typed twin. It mints its [`Auth`] privately
//! (one session per `Driver::new`) and seals the backend exactly as
//! `Workgroup::new` does; raw instructions reach the backend only through
//! `isa`/`raw`, which hold no `Auth`.
//!
//! What the types enforce at compile time the driver enforces before it
//! calls the backend, per region/ring id (the backend's wait ledger, slot
//! machine and loop fixpoint stay the second evaluator; nothing of the IR
//! passes is re-implemented here):
//! - `Free -> Writing` by a store (`begin_write`/`ds_store`), `Writing ->
//!   Published` only at a barrier (`Transition::Ready`) and only after a
//!   `wait` covering the region's youngest store (the Halo VerifyAttn
//!   race), `Published -> Free` only at a barrier (`Transition::Retire`,
//!   the gfx11 FA2 mailbox race). A published region cannot be stored into
//!   or joined until it is retired.
//! - A ring is two regions with `cur`/`next` roles; `Rotate`, `Prime` and
//!   `RetireCur` are its barrier transitions.
//! - Workgroup-scope operations (barriers, `lds`, kernel exits,
//!   `loop_carried`, `handoff`) are refused under wave-uniform control
//!   (`skip_if`, `exec_if`, `if_else`, `forward`, `loop_until`) and after a
//!   wave-uniform kernel exit.
//! - A loop body (or conditional body) must leave every region in the
//!   ownership state it found it, including whether the place's token is
//!   still `Pending` or was waited (`Drained`): the typed core states this
//!   by returning the same `S`. The token kind is driver state (set by
//!   `begin_write`/`ds_store`, cleared by `wait`), independent of whether
//!   a store event exists and of raw counter waits made through `isa`.
//!
//! A rejected request is refused before any backend call, leaves the
//! driver unchanged, and emits nothing. An `Err` that comes from the
//! backend (or from a loop/join, which has already emitted) ends the
//! lowering: stop at the first `Err`.
//!
//! A region is Free after `lds`. Phase names of the front end map onto the
//! driver as: acquire = `lds`/`begin_write`, write = `ds_store`, drain =
//! `wait`, publish = `barrier(&[Ready])`, read = `ds_load`/`ds_load_cur`,
//! retire = `barrier(&[Retire])`.

use crate::backend::{Auth, Backend, EventId, SlotTransition};
use crate::target::{SplitBarrier, Target, WaitModel};
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A joined region holds at most this many backend slots (as `LdsRegion`).
const MAX_JOIN_SLOTS: usize = 4;
/// One barrier carries at most this many slot transitions (as `Lowering`).
const MAX_BARRIER_TRANSITIONS: usize = 32;

/// A dynamic LDS region id (from `lds`, `join`, `split`). Valid for the
/// `Driver` that issued it only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RegionId {
    owner: u64,
    idx: usize,
}
/// A dynamic two-buffer ring id (from `ring`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RingId {
    owner: u64,
    idx: usize,
}
/// What a store or wait names: a region, or a ring's `next` buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    Region(RegionId),
    Ring(RingId),
}
impl From<RegionId> for Place {
    fn from(r: RegionId) -> Self {
        Place::Region(r)
    }
}
impl From<RingId> for Place {
    fn from(r: RingId) -> Self {
        Place::Ring(r)
    }
}

/// Workgroup-level ownership of a region (or of one ring buffer).
///
/// The variants carry an `Rt` prefix so rustc diagnostics for the typed
/// core's `Free`/`Writing`/`Published` states stay unqualified; `Debug`
/// prints the unprefixed ownership names that error text and `.rip` programs
/// use.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// No wave reads it; a wave may start writing.
    RtFree,
    /// This wave has stores into it; nobody may read it yet.
    RtWriting,
    /// Readable by every wave; not writable.
    RtPublished,
}
impl std::fmt::Debug for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Phase::RtFree => "Free",
            Phase::RtWriting => "Writing",
            Phase::RtPublished => "Published",
        })
    }
}

/// One barrier-carried transition (the typed core's `ready`, `retire`,
/// `rotate`, `prime`, `retire_cur`). A barrier lowers all retires, then all
/// readies, each in slice order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transition {
    /// `Writing -> Published`; needs a `wait` covering the youngest store.
    Ready(RegionId),
    /// `Published -> Free`.
    Retire(RegionId),
    /// Ring `(cur Published, next Writing)`: retire `cur` (if it holds a
    /// published tile), publish the drained `next`, swap the roles.
    Rotate(RingId),
    /// Ring `(cur Free, next Writing)`: publish the drained `next` and swap
    /// the roles (nothing to retire).
    Prime(RingId),
    /// Ring `cur` `Published -> Free`.
    RetireCur(RingId),
}
impl Transition {
    fn place(self) -> Place {
        match self {
            Transition::Ready(r) | Transition::Retire(r) => Place::Region(r),
            Transition::Rotate(g) | Transition::Prime(g) | Transition::RetireCur(g) => Place::Ring(g),
        }
    }
}

/// A wave-uniform condition (SCC from a scalar compare). Consumed by one
/// branch; refused if SCC was redefined since the compare.
#[derive(Clone, Copy, Debug)]
#[must_use = "a condition must be consumed by a branch"]
pub struct Cond {
    owner: u64,
    at: usize,
}
/// A workgroup-uniform condition: the caller's claim, as
/// `Workgroup::scmp_wg_uniform`. Copyable like `Cond`: the branch that
/// consumes it emits an instruction, so a second use fails the freshness check.
#[derive(Clone, Copy, Debug)]
#[must_use = "a condition must be consumed by a branch"]
pub struct WgCond {
    owner: u64,
    at: usize,
}
/// A kernel exit reserved by `Driver::exit` and placed by `end`/`end_with`.
/// Clones share the "a branch reaches the exit" flag; placing it twice is
/// refused by the backend (the program has ended).
#[derive(Clone, Debug)]
#[must_use = "a kernel exit must be placed with `Driver::end`"]
pub struct Exit {
    owner: u64,
    label: String,
    branched: Rc<Cell<bool>>,
}
impl Exit {
    pub fn label(&self) -> &str {
        &self.label
    }
}
/// A split barrier in flight: its regions are held until `wait_arrived`.
#[derive(Clone, Debug)]
#[must_use = "a signalled barrier must be waited"]
pub struct Arrived {
    serial: u64,
    ts: Vec<Transition>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hold {
    Live,
    /// In a split barrier between `signal` and `wait_arrived`.
    Barrier,
    /// Joined, split, made a ring buffer, or re-laid out.
    Gone,
}

/// A place's store bookkeeping. `pending` mirrors the typed core's token
/// kind: a `Writing` place carries a `Pending` token from `begin_write` or
/// any store until `wait` turns it into `Drained`. It is a property of the
/// token, not of whether a store event exists (a `begin_write` place is
/// `Pending` with no store), and a raw counter wait through `isa` does not
/// change it (only the backend ledger sees that); the backend still checks
/// at joins and loop fixpoints that nothing is really outstanding.
#[derive(Clone, Debug, Default)]
struct Stores {
    /// The youngest store on each live path into this point (one per
    /// incoming path after a join): a `wait` asks the backend about every
    /// one, so a path that did not wait still gets the drain. Cleared by a
    /// wait or a publishing/retiring barrier; a new store collapses it to
    /// itself (on one path a drain that retires the youngest store retires
    /// the older ones too).
    events: Vec<EventId>,
    pending: bool,
}
impl Stores {
    fn undrained(&self) -> bool {
        self.pending
    }
    fn clear(&mut self) {
        *self = Self::default();
    }
    fn store(&mut self, e: EventId) {
        self.events = vec![e];
        self.pending = true;
    }
    /// Another path (of equal shape, so equal `pending`) into this point.
    fn join(&mut self, other: &Stores) {
        for &e in &other.events {
            if !self.events.contains(&e) {
                self.events.push(e);
            }
        }
        self.events.sort_unstable();
    }
}

#[derive(Clone, Debug)]
struct Region {
    name: String,
    slots: Vec<usize>,
    phase: Phase,
    stores: Stores,
    hold: Hold,
}
#[derive(Clone, Debug)]
struct RingBuf {
    name: String,
    bufs: [Vec<usize>; 2],
    cur: usize,
    /// `cur` holds data a barrier published (false only before the first
    /// rotation of a steady ring).
    cur_live: bool,
    cur_phase: Phase,
    next: Phase,
    stores: Stores,
    hold: Hold,
}

/// Everything a fork or a loop back edge must reproduce.
#[derive(Clone, Debug)]
struct St {
    regions: Vec<Region>,
    rings: Vec<RingBuf>,
    /// The split barrier in flight: its signal serial and transitions.
    in_flight: Option<(u64, Vec<Transition>)>,
    /// Code falls through to the current point (false after `s_branch`).
    reachable: bool,
    wave_exit: bool,
}

#[derive(Clone, Copy)]
enum Loc {
    Region(usize),
    Ring(usize),
}

fn desc(phase: Phase, undrained: bool, hold: Hold) -> String {
    match hold {
        Hold::Gone => "consumed".into(),
        Hold::Barrier => "held by a split barrier".into(),
        Hold::Live => format!("{phase:?}{}", if undrained { " with an undrained store" } else { "" }),
    }
}

/// Why two states differ in ownership, or `None` when they agree (event
/// ids and ring physical indices are build-time values, not shape).
fn shape_diff(a: &St, b: &St) -> Option<String> {
    if a.regions.len() != b.regions.len() || a.rings.len() != b.rings.len() {
        return Some("one path declares LDS regions or rings the other lacks".into());
    }
    if a.in_flight.is_some() != b.in_flight.is_some() {
        return Some("a split barrier is in flight on one path only".into());
    }
    for (x, y) in a.regions.iter().zip(&b.regions) {
        let (dx, dy) = (desc(x.phase, x.stores.undrained(), x.hold), desc(y.phase, y.stores.undrained(), y.hold));
        if dx != dy {
            return Some(format!("{} is {dx} on one path and {dy} on the other", x.name));
        }
    }
    for (x, y) in a.rings.iter().zip(&b.rings) {
        let same = x.hold == y.hold
            && x.cur_phase == y.cur_phase
            && x.next == y.next
            && x.stores.undrained() == y.stores.undrained();
        if !same {
            let d = |r: &RingBuf| format!("cur {:?}, next {}", r.cur_phase, desc(r.next, r.stores.undrained(), r.hold));
            return Some(format!("ring {} is [{}] on one path and [{}] on the other", x.name, d(x), d(y)));
        }
    }
    None
}

/// Join `other` into `into` (paths of equal shape meeting at one point).
fn merge(into: &mut St, other: &St) -> Result<(), String> {
    if let Some(d) = shape_diff(into, other) {
        return Err(format!("the paths disagree on LDS ownership: {d}"));
    }
    for (x, y) in into.regions.iter_mut().zip(&other.regions) {
        x.stores.join(&y.stores);
    }
    for (x, y) in into.rings.iter_mut().zip(&other.rings) {
        x.stores.join(&y.stores);
    }
    into.wave_exit |= other.wave_exit;
    into.reachable = true;
    Ok(())
}

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
static NEXT_SIGNAL: AtomicU64 = AtomicU64::new(1);

/// One workgroup's kernel over a backend `B` for target `T`, driven by
/// dynamic region ids. Workgroup scope and wave scope are one value; the
/// scope the driver is in is a runtime property (`skip_if` and friends run
/// their bodies in wave scope).
pub struct Driver<'b, T: Target, B: Backend> {
    b: &'b mut B,
    auth: Auth,
    owner: u64,
    st: St,
    /// Nesting depth of wave-uniform control: workgroup operations refuse
    /// while it is above zero.
    wave_scope: u32,
    _t: PhantomData<T>,
}

/// The break states of a `Driver::loop_until` (label and, per `break_if`,
/// the backend fork and driver state at the branch).
pub struct Breaks<B: Backend> {
    label: String,
    exits: RefCell<Vec<(B::Fork, St)>>,
    recording: Cell<bool>,
}

struct Dest<B: Backend> {
    label: String,
    forks: Vec<(B::Fork, St)>,
    placed: bool,
}
/// The forward branch targets of one `Driver::forward` block. Each target
/// is placed once, after every branch to it; the state there is the join of
/// every path into it.
pub struct Forward<B: Backend> {
    targets: Vec<Dest<B>>,
}
impl<B: Backend> Forward<B> {
    fn open(&mut self, label: &str) -> Result<usize, String> {
        match self.targets.iter().position(|d| d.label == label) {
            Some(i) if self.targets[i].placed => Err(format!("branch to {label} after it was placed: branches only go forward")),
            Some(i) => Ok(i),
            None => {
                self.targets.push(Dest { label: label.into(), forks: Vec::new(), placed: false });
                Ok(self.targets.len() - 1)
            }
        }
    }
    /// `s_cbranch_scc1 label`: when `cond` holds, continue at `label`.
    pub fn branch_if<T: Target>(&mut self, w: &mut Driver<'_, T, B>, cond: Cond, label: &str) -> Result<(), String> {
        w.fresh(cond.owner, cond.at)?;
        let i = self.open(label)?;
        w.b.branch_scc1(&w.auth, label)?;
        self.targets[i].forks.push((w.b.fork(), w.st.clone()));
        Ok(())
    }
    /// `s_cbranch_scc0 label`: when `cond` does not hold, continue at `label`.
    pub fn branch_unless<T: Target>(&mut self, w: &mut Driver<'_, T, B>, cond: Cond, label: &str) -> Result<(), String> {
        w.fresh(cond.owner, cond.at)?;
        let i = self.open(label)?;
        w.b.branch_scc0(&w.auth, label)?;
        self.targets[i].forks.push((w.b.fork(), w.st.clone()));
        Ok(())
    }
    /// `s_branch label`: nothing falls through past it.
    pub fn goto<T: Target>(&mut self, w: &mut Driver<'_, T, B>, label: &str) -> Result<(), String> {
        let i = self.open(label)?;
        w.b.branch(&w.auth, label)?;
        self.targets[i].forks.push((w.b.fork(), w.st.clone()));
        w.st.reachable = false;
        Ok(())
    }
    /// Place `label`: its state joins every branch to it with the code
    /// falling through (if any).
    pub fn place<T: Target>(&mut self, w: &mut Driver<'_, T, B>, label: &str) -> Result<(), String> {
        let d = self.targets.iter_mut().find(|d| d.label == label).ok_or_else(|| format!("{label} is not the target of a branch in this block"))?;
        if d.placed {
            return Err(format!("{label} is already placed"));
        }
        d.placed = true;
        for (fork, st) in d.forks.drain(..) {
            w.b.join(&w.auth, fork).map_err(|e| format!("the paths cannot join at {label}: {e}"))?;
            w.join_state(st).map_err(|e| format!("the paths cannot join at {label}: {e}"))?;
        }
        w.b.label(&w.auth, label)
    }
}

impl<'b, T: Target, B: Backend> Driver<'b, T, B> {
    /// Take ownership of a backend built for `T` (the runtime twin of
    /// `Workgroup::new`): its untyped LDS, barrier and loop entry points
    /// refuse from here on, and so does a second `Driver`/`Workgroup` over it.
    pub fn new(b: &'b mut B) -> Result<Self, String> {
        if b.arch_name() != T::NAME {
            return Err(format!("backend targets {}, kernel is typed for {}", b.arch_name(), T::NAME));
        }
        let auth = Auth::mint();
        b.seal(&auth)?;
        let st = St { regions: Vec::new(), rings: Vec::new(), in_flight: None, reachable: true, wave_exit: false };
        Ok(Self { b, auth, owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed), st, wave_scope: 0, _t: PhantomData })
    }

    // ---- raw access ---------------------------------------------------

    /// The backend for raw instructions (ALU, VMEM, SMEM, crossbar); it
    /// holds no `Auth`, so LDS, barrier, branch and loop entry points stay
    /// out of reach.
    pub fn isa(&mut self) -> &mut B {
        self.b
    }
    /// Emit raw instructions through `f` (the closure form of `isa`).
    pub fn raw<R>(&mut self, f: impl FnOnce(&mut B) -> Result<R, String>) -> Result<R, String> {
        f(self.b)
    }
    /// Program position (instructions emitted so far).
    pub fn position(&self) -> usize {
        self.b.position()
    }

    // ---- ids and introspection ----------------------------------------

    fn region_ix(&self, id: RegionId) -> Result<usize, String> {
        if id.owner != self.owner {
            return Err("region id from another Driver".into());
        }
        let r = self.st.regions.get(id.idx).ok_or("unknown LDS region")?;
        match r.hold {
            Hold::Live => Ok(id.idx),
            Hold::Barrier => Err(format!("{} is held by a split barrier in flight until wait_arrived", r.name)),
            Hold::Gone => Err(format!("{} was consumed (joined, split, made a ring buffer or re-laid out)", r.name)),
        }
    }
    fn ring_ix(&self, id: RingId) -> Result<usize, String> {
        if id.owner != self.owner {
            return Err("ring id from another Driver".into());
        }
        let r = self.st.rings.get(id.idx).ok_or("unknown LDS ring")?;
        match r.hold {
            Hold::Live => Ok(id.idx),
            Hold::Barrier => Err(format!("ring {} is held by a split barrier in flight until wait_arrived", r.name)),
            Hold::Gone => Err(format!("ring {} was re-laid out", r.name)),
        }
    }
    fn loc(&self, p: Place) -> Result<Loc, String> {
        match p {
            Place::Region(r) => self.region_ix(r).map(Loc::Region),
            Place::Ring(g) => self.ring_ix(g).map(Loc::Ring),
        }
    }
    fn name_phase(&self, l: Loc) -> (&str, Phase) {
        match l {
            Loc::Region(i) => (&self.st.regions[i].name, self.st.regions[i].phase),
            Loc::Ring(i) => (&self.st.rings[i].name, self.st.rings[i].next),
        }
    }
    fn stores_mut(&mut self, l: Loc) -> &mut Stores {
        match l {
            Loc::Region(i) => &mut self.st.regions[i].stores,
            Loc::Ring(i) => &mut self.st.rings[i].stores,
        }
    }
    /// Phase of a region.
    pub fn region_phase(&self, r: RegionId) -> Result<Phase, String> {
        Ok(self.st.regions[self.region_ix(r)?].phase)
    }
    /// `(cur, next)` phases of a ring's buffers.
    pub fn ring_phases(&self, g: RingId) -> Result<(Phase, Phase), String> {
        let r = &self.st.rings[self.ring_ix(g)?];
        Ok((r.cur_phase, r.next))
    }
    /// Physical index (0 or 1, declaration order) of the ring's current buffer.
    pub fn cur_index(&self, g: RingId) -> Result<usize, String> {
        Ok(self.st.rings[self.ring_ix(g)?].cur)
    }
    /// Physical index of the ring's next buffer.
    pub fn next_index(&self, g: RingId) -> Result<usize, String> {
        Ok(1 - self.st.rings[self.ring_ix(g)?].cur)
    }

    // ---- scope guards -------------------------------------------------

    fn wg(&self, what: &str) -> Result<(), String> {
        if self.wave_scope > 0 {
            return Err(format!("{what} is workgroup scope: it cannot run under wave- or lane-dependent control"));
        }
        Ok(())
    }
    fn check_barrier_scope(&self) -> Result<(), String> {
        if self.st.wave_exit { Err("workgroup barrier after a wave-uniform kernel exit".into()) } else { Ok(()) }
    }
    fn scoped<R>(&mut self, f: impl FnOnce(&mut Self) -> Result<R, String>) -> Result<R, String> {
        self.wave_scope += 1;
        let out = f(self);
        self.wave_scope -= 1;
        out
    }
    fn fresh(&self, owner: u64, at: usize) -> Result<(), String> {
        if owner != self.owner {
            return Err("condition from another Driver".into());
        }
        if self.b.position() != at {
            return Err("SCC was redefined between its compare and the branch that consumes it".into());
        }
        Ok(())
    }
    fn own_exit(&self, end: &Exit) -> Result<(), String> {
        if end.owner != self.owner {
            return Err(format!("kernel exit {} belongs to another Driver", end.label));
        }
        Ok(())
    }
    fn restore(&mut self, saved: St) {
        let wave_exit = self.st.wave_exit;
        self.st = saved;
        self.st.wave_exit |= wave_exit;
        self.st.reachable = true;
    }
    /// A path with state `other` meets the current point.
    fn join_state(&mut self, other: St) -> Result<(), String> {
        if self.st.reachable {
            merge(&mut self.st, &other)
        } else {
            self.restore(other);
            Ok(())
        }
    }

    // ---- LDS layout ---------------------------------------------------

    /// Declare one LDS region (one backend slot; ids follow declaration order).
    pub fn lds(&mut self, name: &str, base: u32, len: u32) -> Result<RegionId, String> {
        self.wg("lds")?;
        if base.checked_add(len).is_none_or(|end| end > T::LDS_BYTES) {
            return Err(format!("LDS region {name} [{base}, +{len}) exceeds {} bytes on {}", T::LDS_BYTES, T::NAME));
        }
        let slot = self.b.lds_slot(&self.auth, name, base, len)?;
        self.st.regions.push(Region { name: name.into(), slots: vec![slot], phase: Phase::RtFree, stores: Stores::default(), hold: Hold::Live });
        Ok(RegionId { owner: self.owner, idx: self.st.regions.len() - 1 })
    }
    /// End the slot layout so a later phase can carve its own. Every region
    /// and ring must be `Free`; all ids are invalid afterwards.
    pub fn relayout(&mut self) -> Result<(), String> {
        self.wg("relayout")?;
        for r in self.st.regions.iter().filter(|r| r.hold != Hold::Gone) {
            if r.hold != Hold::Live || r.phase != Phase::RtFree {
                return Err(format!("{} is {} at relayout: a barrier must retire every reader first", r.name, desc(r.phase, r.stores.undrained(), r.hold)));
            }
        }
        for g in self.st.rings.iter().filter(|g| g.hold != Hold::Gone) {
            if g.hold != Hold::Live || g.cur_phase != Phase::RtFree || g.next != Phase::RtFree {
                return Err(format!("ring {} is not Free at relayout: a barrier must retire every reader first", g.name));
            }
        }
        self.b.lds_relayout(&self.auth)?;
        self.st.regions.iter_mut().for_each(|r| r.hold = Hold::Gone);
        self.st.rings.iter_mut().for_each(|g| g.hold = Hold::Gone);
        Ok(())
    }
    /// Re-carve `Free` regions into one region (a store or load then covers
    /// every slot of all parts, in order).
    pub fn join(&mut self, parts: &[RegionId]) -> Result<RegionId, String> {
        if parts.len() < 2 {
            return Err("a join needs at least two regions".into());
        }
        let mut slots = Vec::new();
        let mut name = String::new();
        for (n, &p) in parts.iter().enumerate() {
            if parts[..n].contains(&p) {
                return Err("a region cannot be joined with itself".into());
            }
            let r = &self.st.regions[self.region_ix(p)?];
            if r.phase != Phase::RtFree {
                return Err(format!("cannot join {}: it is {:?}; other waves may still be reading it, only a barrier can retire their reads", r.name, r.phase));
            }
            slots.extend_from_slice(&r.slots);
            if n > 0 {
                name.push('+');
            }
            name.push_str(&r.name);
        }
        if slots.len() > MAX_JOIN_SLOTS {
            return Err(format!("a joined LDS region holds at most {MAX_JOIN_SLOTS} slots"));
        }
        for &p in parts {
            self.st.regions[p.idx].hold = Hold::Gone;
        }
        self.st.regions.push(Region { name, slots, phase: Phase::RtFree, stores: Stores::default(), hold: Hold::Live });
        Ok(RegionId { owner: self.owner, idx: self.st.regions.len() - 1 })
    }
    /// Undo a `join`: the first `first` slots become one region, the rest another.
    pub fn split(&mut self, r: RegionId, first: usize) -> Result<(RegionId, RegionId), String> {
        let i = self.region_ix(r)?;
        let g = &self.st.regions[i];
        if g.phase != Phase::RtFree {
            return Err(format!("cannot split {}: it is {:?}", g.name, g.phase));
        }
        if first == 0 || first >= g.slots.len() {
            return Err(format!("cannot split a {}-slot region after {first} slots", g.slots.len()));
        }
        let (a, b) = g.slots.split_at(first);
        let (a, b) = (Region { name: format!("{}.0", g.name), slots: a.to_vec(), phase: Phase::RtFree, stores: Stores::default(), hold: Hold::Live }, Region { name: format!("{}.1", g.name), slots: b.to_vec(), phase: Phase::RtFree, stores: Stores::default(), hold: Hold::Live });
        self.st.regions[i].hold = Hold::Gone;
        self.st.regions.push(a);
        self.st.regions.push(b);
        let n = self.st.regions.len();
        Ok((RegionId { owner: self.owner, idx: n - 2 }, RegionId { owner: self.owner, idx: n - 1 }))
    }
    /// A two-buffer ring over two `Free` regions; `first` is written first.
    pub fn ring(&mut self, name: &str, first: RegionId, second: RegionId) -> Result<RingId, String> {
        if first == second {
            return Err("a ring needs two distinct regions".into());
        }
        let (i, j) = (self.region_ix(first)?, self.region_ix(second)?);
        for k in [i, j] {
            let r = &self.st.regions[k];
            if r.phase != Phase::RtFree {
                return Err(format!("ring {name}: {} is {:?}, a ring is built over Free regions", r.name, r.phase));
            }
        }
        let bufs = [self.st.regions[i].slots.clone(), self.st.regions[j].slots.clone()];
        self.st.regions[i].hold = Hold::Gone;
        self.st.regions[j].hold = Hold::Gone;
        self.st.rings.push(RingBuf { name: name.into(), bufs, cur: 1, cur_live: false, cur_phase: Phase::RtFree, next: Phase::RtFree, stores: Stores::default(), hold: Hold::Live });
        Ok(RingId { owner: self.owner, idx: self.st.rings.len() - 1 })
    }
    /// Enter steady-state role typing before a loop whose head barrier
    /// rotates: the vacant current buffer is not retired by the first
    /// `Rotate`, and a read of it is refused.
    pub fn ring_steady(&mut self, g: RingId) -> Result<(), String> {
        let i = self.ring_ix(g)?;
        let r = &mut self.st.rings[i];
        if r.cur_phase != Phase::RtFree || r.cur_live {
            return Err(format!("ring {}: only a ring whose current buffer is Free and vacant can become steady", r.name));
        }
        r.cur_phase = Phase::RtPublished;
        Ok(())
    }

    // ---- stores, loads, waits -----------------------------------------

    fn storable(&self, l: Loc) -> Result<(), String> {
        match self.name_phase(l) {
            (_, Phase::RtFree | Phase::RtWriting) => Ok(()),
            (name, Phase::RtPublished) => Err(format!("cannot store into {name}: it is Published and other waves may still be reading it until a barrier retires it")),
        }
    }
    fn slots_of(st: &St, l: Loc) -> &[usize] {
        match l {
            Loc::Region(i) => &st.regions[i].slots,
            Loc::Ring(i) => &st.rings[i].bufs[1 - st.rings[i].cur],
        }
    }
    fn note_store(&mut self, l: Loc, e: EventId) {
        match l {
            Loc::Region(i) => self.st.regions[i].phase = Phase::RtWriting,
            Loc::Ring(i) => self.st.rings[i].next = Phase::RtWriting,
        }
        self.stores_mut(l).store(e);
    }
    /// A `Free` region (or ring `next` buffer) as `Writing` with an empty
    /// `Pending` token and no store yet (as `Wave::begin_write`): stores
    /// may follow, possibly under a `skip_if`, and a `wait` is still what
    /// a publishing barrier demands.
    pub fn begin_write(&mut self, to: impl Into<Place>) -> Result<(), String> {
        let l = self.loc(to.into())?;
        match l {
            Loc::Region(i) if self.st.regions[i].phase == Phase::RtFree => {
                self.st.regions[i].phase = Phase::RtWriting;
                self.st.regions[i].stores.pending = true;
            }
            Loc::Ring(i) if self.st.rings[i].next == Phase::RtFree => {
                self.st.rings[i].next = Phase::RtWriting;
                self.st.rings[i].stores.pending = true;
            }
            _ => {
                let (name, phase) = self.name_phase(l);
                return Err(format!("begin_write of {name}: it is {phase:?}, only a Free place can start writing"));
            }
        }
        Ok(())
    }
    /// One LDS store into a `Free` or `Writing` region, or into a ring's
    /// `next` buffer.
    pub fn ds_store(&mut self, to: impl Into<Place>, insn: B::Insn) -> Result<(), String> {
        let l = self.loc(to.into())?;
        self.storable(l)?;
        let e = self.b.ds_store(&self.auth, Self::slots_of(&self.st, l), insn)?;
        self.note_store(l, e);
        Ok(())
    }
    /// One LDS store covering two places (one instruction, both slot lists).
    pub fn ds_store2(&mut self, x: impl Into<Place>, y: impl Into<Place>, insn: B::Insn) -> Result<(), String> {
        let (px, py) = (x.into(), y.into());
        if px == py {
            return Err("a store cannot cover the same place twice".into());
        }
        let (lx, ly) = (self.loc(px)?, self.loc(py)?);
        self.storable(lx)?;
        self.storable(ly)?;
        let (sx, sy) = (Self::slots_of(&self.st, lx), Self::slots_of(&self.st, ly));
        let mut ids = [0usize; 2 * MAX_JOIN_SLOTS];
        let n = sx.len() + sy.len();
        ids[..sx.len()].copy_from_slice(sx);
        ids[sx.len()..n].copy_from_slice(sy);
        let e = self.b.ds_store(&self.auth, &ids[..n], insn)?;
        self.note_store(lx, e);
        self.note_store(ly, e);
        Ok(())
    }
    /// One LDS load from a `Published` region.
    pub fn ds_load(&mut self, r: RegionId, insn: B::Insn) -> Result<(), String> {
        let i = self.region_ix(r)?;
        let g = &self.st.regions[i];
        if g.phase != Phase::RtPublished {
            return Err(format!("{} is not published: it is {:?}", g.name, g.phase));
        }
        self.b.ds_load(&self.auth, &g.slots, insn)
    }
    /// One LDS load from a ring's current buffer.
    pub fn ds_load_cur(&mut self, g: RingId, insn: B::Insn) -> Result<(), String> {
        let i = self.ring_ix(g)?;
        let r = &self.st.rings[i];
        if r.cur_phase != Phase::RtPublished {
            return Err(format!("ring {}: current buffer is {:?}, not published", r.name, r.cur_phase));
        }
        if !r.cur_live {
            return Err("read of a ring buffer no barrier has published".into());
        }
        self.b.ds_load(&self.auth, &r.bufs[r.cur], insn)
    }
    /// Wait for a `Writing` place's stores; emits the counter wait the
    /// backend still needs (none when an earlier wait retired them). This
    /// is the evidence `Ready`/`Rotate`/`Prime` demand.
    pub fn wait(&mut self, p: impl Into<Place>) -> Result<(), String> {
        self.wait_all(&[p.into()])
    }
    /// Wait for several places with one counter wait.
    pub fn wait_all(&mut self, ps: &[Place]) -> Result<(), String> {
        let mut locs = Vec::with_capacity(ps.len());
        for &p in ps {
            let l = self.loc(p)?;
            let (name, phase) = self.name_phase(l);
            if phase != Phase::RtWriting {
                return Err(format!("wait on {name}: no store of this wave is in flight ({phase:?})"));
            }
            locs.push(l);
        }
        let b = &*self.b;
        let st = &self.st;
        let pending = locs.iter().any(|&l| {
            let s = match l {
                Loc::Region(i) => &st.regions[i].stores,
                Loc::Ring(i) => &st.rings[i].stores,
            };
            s.events.iter().any(|&e| b.lds_store_pending(e))
        });
        if pending {
            self.b.drain_lds_stores(&self.auth, <T::Waits as WaitModel>::LDS)?;
        }
        for l in locs {
            self.stores_mut(l).clear();
        }
        Ok(())
    }

    // ---- barriers -----------------------------------------------------

    fn covered(name: &str, s: &Stores) -> Result<(), String> {
        if s.undrained() {
            return Err(format!("barrier would publish {name} with an undrained LDS store: wait for it first (s_barrier does not wait for memory counters)"));
        }
        Ok(())
    }
    /// Validate `ts` against the current states and lower them: all
    /// retires, then all readies, each in slice order.
    fn plan(&self, ts: &[Transition]) -> Result<Vec<SlotTransition>, String> {
        let (mut retire, mut ready) = (Vec::new(), Vec::new());
        for (n, t) in ts.iter().enumerate() {
            if ts[..n].iter().any(|u| u.place() == t.place()) {
                return Err("one barrier carries at most one transition per region or ring".into());
            }
            match *t {
                Transition::Ready(r) => {
                    let g = &self.st.regions[self.region_ix(r)?];
                    if g.phase != Phase::RtWriting {
                        return Err(format!("ready of {}: it is {:?}, only a Writing region can be published", g.name, g.phase));
                    }
                    Self::covered(&g.name, &g.stores)?;
                    ready.extend(g.slots.iter().map(|&id| SlotTransition::Ready(id)));
                }
                Transition::Retire(r) => {
                    let g = &self.st.regions[self.region_ix(r)?];
                    if g.phase != Phase::RtPublished {
                        return Err(format!("retire of {}: it is {:?}, only a Published region can be retired", g.name, g.phase));
                    }
                    retire.extend(g.slots.iter().map(|&id| SlotTransition::Retire(id)));
                }
                Transition::Rotate(g) => {
                    let r = &self.st.rings[self.ring_ix(g)?];
                    if r.cur_phase != Phase::RtPublished || r.next != Phase::RtWriting {
                        return Err(format!("rotate of ring {}: cur is {:?}, next is {:?}; it needs cur Published and next Writing", r.name, r.cur_phase, r.next));
                    }
                    Self::covered(&r.name, &r.stores)?;
                    if r.cur_live {
                        retire.extend(r.bufs[r.cur].iter().map(|&id| SlotTransition::Retire(id)));
                    }
                    ready.extend(r.bufs[1 - r.cur].iter().map(|&id| SlotTransition::Ready(id)));
                }
                Transition::Prime(g) => {
                    let r = &self.st.rings[self.ring_ix(g)?];
                    if r.cur_phase != Phase::RtFree || r.next != Phase::RtWriting {
                        return Err(format!("prime of ring {}: cur is {:?}, next is {:?}; it needs cur Free and next Writing", r.name, r.cur_phase, r.next));
                    }
                    Self::covered(&r.name, &r.stores)?;
                    ready.extend(r.bufs[1 - r.cur].iter().map(|&id| SlotTransition::Ready(id)));
                }
                Transition::RetireCur(g) => {
                    let r = &self.st.rings[self.ring_ix(g)?];
                    if r.cur_phase != Phase::RtPublished {
                        return Err(format!("retire_cur of ring {}: cur is {:?}, only a Published buffer can be retired", r.name, r.cur_phase));
                    }
                    if r.cur_live {
                        retire.extend(r.bufs[r.cur].iter().map(|&id| SlotTransition::Retire(id)));
                    }
                }
            }
        }
        if retire.len() + ready.len() > MAX_BARRIER_TRANSITIONS {
            return Err(format!("one barrier carries at most {MAX_BARRIER_TRANSITIONS} slot transitions"));
        }
        retire.extend(ready);
        Ok(retire)
    }
    fn apply(&mut self, ts: &[Transition]) {
        for t in ts {
            match *t {
                Transition::Ready(r) => {
                    let g = &mut self.st.regions[r.idx];
                    g.phase = Phase::RtPublished;
                    g.stores.clear();
                }
                Transition::Retire(r) => {
                    let g = &mut self.st.regions[r.idx];
                    g.phase = Phase::RtFree;
                    g.stores.clear();
                }
                Transition::Rotate(g) | Transition::Prime(g) => {
                    let r = &mut self.st.rings[g.idx];
                    r.cur = 1 - r.cur;
                    r.cur_live = true;
                    r.cur_phase = Phase::RtPublished;
                    r.next = Phase::RtFree;
                    r.stores.clear();
                }
                Transition::RetireCur(g) => {
                    let r = &mut self.st.rings[g.idx];
                    r.cur_phase = Phase::RtFree;
                    r.cur_live = false;
                }
            }
        }
    }
    fn hold(&mut self, ts: &[Transition], hold: Hold) {
        for t in ts {
            match t.place() {
                Place::Region(r) => self.st.regions[r.idx].hold = hold,
                Place::Ring(g) => self.st.rings[g.idx].hold = hold,
            }
        }
    }
    /// One workgroup barrier carrying `ts`.
    pub fn barrier(&mut self, ts: &[Transition]) -> Result<(), String> {
        self.wg("barrier")?;
        self.check_barrier_scope()?;
        let list = self.plan(ts)?;
        self.b.barrier(&self.auth, &list)?;
        self.apply(ts);
        Ok(())
    }
    /// Split barrier, first half (gfx12): this wave has arrived; the named
    /// regions are held until `wait_arrived`.
    pub fn signal(&mut self, ts: &[Transition]) -> Result<Arrived, String>
    where
        T: SplitBarrier,
    {
        self.wg("signal")?;
        self.check_barrier_scope()?;
        if self.st.in_flight.is_some() {
            return Err("a split barrier is already in flight".into());
        }
        let list = self.plan(ts)?;
        self.b.barrier_signal(&self.auth, &list)?;
        self.hold(ts, Hold::Barrier);
        let serial = NEXT_SIGNAL.fetch_add(1, Ordering::Relaxed);
        self.st.in_flight = Some((serial, ts.to_vec()));
        Ok(Arrived { serial, ts: ts.to_vec() })
    }
    /// Split barrier, second half.
    pub fn wait_arrived(&mut self, a: Arrived) -> Result<(), String>
    where
        T: SplitBarrier,
    {
        self.wg("wait_arrived")?;
        self.check_barrier_scope()?;
        if self.st.in_flight.as_ref().map(|(n, _)| *n) != Some(a.serial) {
            return Err("wait_arrived without the signal that produced it".into());
        }
        self.b.barrier_wait(&self.auth)?;
        self.hold(&a.ts, Hold::Live);
        self.apply(&a.ts);
        self.st.in_flight = None;
        Ok(())
    }

    // ---- conditions, branches, loops ----------------------------------

    /// A scalar compare (`s_cmp*`, `s_bitcmp*`): SCC is wave-uniform.
    pub fn scmp(&mut self, insn: B::Insn) -> Result<Cond, String> {
        self.b.scalar_compare(&self.auth, insn)?;
        Ok(Cond { owner: self.owner, at: self.b.position() })
    }
    /// A scalar compare whose operands the caller asserts are
    /// workgroup-uniform (kernel arguments, workgroup ids, constants and
    /// counters derived only from them).
    pub fn scmp_wg_uniform(&mut self, insn: B::Insn) -> Result<WgCond, String> {
        self.wg("scmp_wg_uniform")?;
        self.b.scalar_compare(&self.auth, insn)?;
        Ok(WgCond { owner: self.owner, at: self.b.position() })
    }
    /// A label in straight-line code (a register lifetime boundary).
    pub fn label(&mut self, name: &str) -> Result<(), String> {
        self.b.label(&self.auth, name)
    }
    fn skip(&mut self, owner: u64, at: usize, target: &str, scc0: bool, wave: bool, body: impl FnOnce(&mut Self) -> Result<(), String>) -> Result<(), String> {
        self.fresh(owner, at)?;
        if scc0 {
            self.b.branch_scc0(&self.auth, target)?;
        } else {
            self.b.branch_scc1(&self.auth, target)?;
        }
        let skipped = self.b.fork();
        let before = self.st.clone();
        if wave { self.scoped(body) } else { body(self) }?;
        if let Some(d) = shape_diff(&before, &self.st) {
            return Err(format!("the skipped path and the body disagree on LDS ownership at {target}: {d}"));
        }
        self.b.join(&self.auth, skipped)?;
        // The skipped path keeps the stores the body's path may have
        // waited on: both paths' evidence survives the join.
        merge(&mut self.st, &before)?;
        self.b.label(&self.auth, target)
    }
    /// Branch over `body` to `target` when `cond` holds. The body runs in
    /// wave scope (no barrier) and must leave LDS ownership as it found it.
    pub fn skip_if(&mut self, cond: Cond, target: &str, body: impl FnOnce(&mut Self) -> Result<(), String>) -> Result<(), String> {
        self.skip(cond.owner, cond.at, target, false, true, body)
    }
    /// Skip `body` when `cond` does not hold.
    pub fn skip_unless(&mut self, cond: Cond, target: &str, body: impl FnOnce(&mut Self) -> Result<(), String>) -> Result<(), String> {
        self.skip(cond.owner, cond.at, target, true, true, body)
    }
    /// `skip_if` over a workgroup-uniform condition: `body` may hold barriers.
    pub fn wg_skip_if(&mut self, cond: WgCond, target: &str, body: impl FnOnce(&mut Self) -> Result<(), String>) -> Result<(), String> {
        self.wg("wg_skip_if")?;
        self.skip(cond.owner, cond.at, target, false, false, body)
    }
    /// Run `body` with EXEC = all lanes if `cond`, else none, then restore
    /// EXEC. The body is wave scope.
    pub fn exec_if<R>(&mut self, cond: Cond, body: impl FnOnce(&mut Self) -> Result<R, String>) -> Result<R, String> {
        self.fresh(cond.owner, cond.at)?;
        self.b.exec_from_scc(&self.auth)?;
        let out = self.scoped(body)?;
        self.b.exec_all(&self.auth)?;
        Ok(out)
    }
    /// A block of wave-uniform forward branches; every target it branches
    /// to must be placed inside it, after the branch. The body is wave scope.
    pub fn forward<R>(&mut self, body: impl FnOnce(&mut Self, &mut Forward<B>) -> Result<R, String>) -> Result<R, String> {
        let mut f = Forward { targets: Vec::new() };
        let out = self.scoped(|d| body(d, &mut f))?;
        if let Some(d) = f.targets.iter().find(|d| !d.placed) {
            return Err(format!("forward target {} is never placed", d.label));
        }
        Ok(out)
    }
    /// Two-way branch: when `cond` holds, jump to `else_label` and run `els`
    /// from the state of the branch point; otherwise run `then`, which ends
    /// with `s_branch join`. The caller places `join` right after. Neither
    /// arm can reach a barrier or change LDS ownership.
    pub fn if_else(
        &mut self,
        cond: Cond,
        else_label: &str,
        join: &str,
        then: impl FnOnce(&mut Self) -> Result<(), String>,
        els: impl FnOnce(&mut Self) -> Result<(), String>,
    ) -> Result<(), String> {
        self.fresh(cond.owner, cond.at)?;
        self.b.branch_scc1(&self.auth, else_label)?;
        let at = self.b.fork();
        let saved = self.st.clone();
        self.scoped(then)?;
        if let Some(d) = shape_diff(&saved, &self.st) {
            return Err(format!("the then arm changes LDS ownership: {d}"));
        }
        self.b.branch(&self.auth, join)?;
        self.st.reachable = false;
        let then_end = (self.b.fork(), self.st.clone());
        self.b.resume(&self.auth, at)?;
        self.restore(saved.clone());
        self.b.label(&self.auth, else_label)?;
        self.scoped(els)?;
        if let Some(d) = shape_diff(&saved, &self.st) {
            return Err(format!("the else arm changes LDS ownership: {d}"));
        }
        self.b.join(&self.auth, then_end.0).map_err(|e| format!("the arms cannot join at {join}: {e}"))?;
        merge(&mut self.st, &then_end.1).map_err(|e| format!("the arms cannot join at {join}: {e}"))
    }
    /// A loop at `head` whose back edge is `s_branch head`, left only by
    /// `break_if` branches to `exit`. The body is wave scope; the backend
    /// checks its state reaches a fixed point and the driver checks LDS
    /// ownership is the same at the back edge as at the entry. The code after
    /// the loop continues from the join of every `break_if`'s state.
    pub fn loop_until(
        &mut self,
        head: &str,
        exit: &str,
        body: impl for<'x> Fn(&mut Driver<'x, T, B>, &Breaks<B>) -> Result<(), String>,
    ) -> Result<(), String> {
        let x = Breaks { label: exit.into(), exits: RefCell::new(Vec::new()), recording: Cell::new(true) };
        let (entry, scope, owner) = (self.st.clone(), self.wave_scope + 1, self.owner);
        let carried = RefCell::new(entry.clone());
        let auth = &self.auth;
        let emit = |b: &mut B| -> Result<(), String> {
            x.exits.borrow_mut().clear();
            x.recording.set(true);
            let mut d = Driver { b, auth: auth.reenter(), owner, st: carried.borrow().clone(), wave_scope: scope, _t: PhantomData };
            body(&mut d, &x)?;
            x.recording.set(false);
            if let Some(diff) = shape_diff(&entry, &d.st) {
                return Err(format!("loop {head}: LDS ownership differs between the entry and the back edge: {diff}"));
            }
            d.b.branch(&d.auth, head)?;
            d.st.reachable = true;
            *carried.borrow_mut() = d.st;
            Ok(())
        };
        self.b.loop_(&self.auth, head, &emit)?;
        let wave_exit = carried.into_inner().wave_exit;
        let mut exits = x.exits.into_inner().into_iter();
        let (first, first_st) = exits.next().ok_or_else(|| format!("loop {head} has no break_if: it never reaches {exit}"))?;
        self.b.resume(&self.auth, first)?;
        self.restore(first_st);
        for (other, st) in exits {
            self.b.join(&self.auth, other)?;
            merge(&mut self.st, &st)?;
        }
        self.st.wave_exit |= wave_exit;
        self.b.label(&self.auth, exit)
    }
    /// Leave the enclosing `loop_until` when `cond` holds.
    pub fn break_if(&mut self, cond: Cond, exit: &Breaks<B>) -> Result<(), String> {
        self.fresh(cond.owner, cond.at)?;
        self.b.branch_scc1(&self.auth, &exit.label)?;
        if exit.recording.get() {
            exit.exits.borrow_mut().push((self.b.fork(), self.st.clone()));
        }
        Ok(())
    }
    /// A workgroup loop at `head`: `body` returns the workgroup-uniform
    /// condition of its back edge (`s_cbranch_scc1 head`). The driver
    /// refuses a body whose back-edge LDS ownership (including whether the
    /// youngest store is drained) differs from the entry's: a store that
    /// reaches the loop head undrained, where a barrier needs it drained,
    /// is the Halo VerifyAttn race. The backend still checks the exact wait
    /// state reaches a fixed point.
    pub fn loop_carried(&mut self, head: &str, body: impl for<'x> Fn(&mut Driver<'x, T, B>) -> Result<WgCond, String>) -> Result<(), String> {
        self.wg("loop_carried")?;
        let (entry, owner) = (self.st.clone(), self.owner);
        let carried = RefCell::new(entry.clone());
        let emitted = Cell::new(false);
        let auth = &self.auth;
        let emit = |b: &mut B| -> Result<(), String> {
            let mut d = Driver { b, auth: auth.reenter(), owner, st: carried.borrow().clone(), wave_scope: 0, _t: PhantomData };
            let back = body(&mut d)?;
            d.fresh(back.owner, back.at)?;
            if let Some(diff) = shape_diff(&entry, &d.st) {
                return Err(format!("loop {head}: LDS ownership differs between the entry and the back edge: {diff}"));
            }
            d.b.branch_scc1(&d.auth, head)?;
            d.st.reachable = true;
            *carried.borrow_mut() = d.st;
            emitted.set(true);
            Ok(())
        };
        self.b.loop_(&self.auth, head, &emit)?;
        if !emitted.get() {
            return Err(format!("loop {head} emitted no body"));
        }
        self.st = carried.into_inner();
        Ok(())
    }

    // ---- kernel exits ---------------------------------------------------

    /// Reserve the kernel exit `label` (placed later by `end`).
    pub fn exit(&mut self, label: &str) -> Result<Exit, String> {
        self.wg("exit")?;
        self.b.reserve_exit(&self.auth, label)?;
        Ok(Exit { owner: self.owner, label: label.into(), branched: Rc::new(Cell::new(false)) })
    }
    /// Leave the kernel when `cond` holds: skipping every later barrier is
    /// legal only for the whole workgroup, and only to the kernel's exit.
    pub fn exit_if(&mut self, cond: WgCond, end: &Exit) -> Result<(), String> {
        self.wg("exit_if")?;
        self.own_exit(end)?;
        self.fresh(cond.owner, cond.at)?;
        self.b.branch_scc1(&self.auth, &end.label)?;
        end.branched.set(true);
        Ok(())
    }
    /// Leave through the reserved kernel exit unless `cond` holds (wave
    /// scope). No workgroup barrier may follow.
    pub fn wave_exit_unless(&mut self, cond: Cond, end: &Exit) -> Result<(), String> {
        self.own_exit(end)?;
        self.fresh(cond.owner, cond.at)?;
        self.b.branch_scc0(&self.auth, &end.label)?;
        end.branched.set(true);
        self.st.wave_exit = true;
        Ok(())
    }
    /// Unless `cond` holds, run `body` and leave the kernel through `end`
    /// (`s_branch`); when it holds, continue at `target` from the branch
    /// point's state.
    pub fn wg_exit_unless(
        &mut self,
        cond: WgCond,
        target: &str,
        end: &Exit,
        body: impl FnOnce(&mut Self) -> Result<(), String>,
    ) -> Result<(), String> {
        self.wg("wg_exit_unless")?;
        self.own_exit(end)?;
        self.fresh(cond.owner, cond.at)?;
        self.b.branch_scc1(&self.auth, target)?;
        let taken = self.b.fork();
        let saved = self.st.clone();
        body(self)?;
        self.b.branch(&self.auth, &end.label)?;
        end.branched.set(true);
        self.b.join(&self.auth, taken)?;
        self.restore(saved);
        self.b.label(&self.auth, target)
    }
    /// Place the kernel exit (`end:` then `s_endpgm`). Nothing may follow.
    pub fn end(&mut self, end: Exit) -> Result<(), String> {
        self.end_with(end, |_| Ok(()))
    }
    /// Place the kernel exit, run `tail` in wave scope, then `s_endpgm`. An
    /// exit something branches to takes no tail.
    pub fn end_with(&mut self, end: Exit, tail: impl FnOnce(&mut Self) -> Result<(), String>) -> Result<(), String> {
        self.wg("end")?;
        self.own_exit(&end)?;
        self.b.place_exit(&self.auth, &end.label)?;
        let at = self.b.position();
        self.scoped(tail)?;
        if end.branched.get() && self.b.position() != at {
            return Err(format!("kernel exit {} is a branch target: it takes no tail", end.label));
        }
        self.b.end_program(&self.auth)
    }
    /// Wave-role LDS handoff, the kernel's last phase (the runtime twin of
    /// `Workgroup::handoff`). Waves where `readers` holds branch to
    /// `reader_label`; the others run `write` (wave scope, leaving `region`
    /// `Writing`), wait, publish it at one barrier and branch to `end`.
    /// Readers resume from the branch point's state, meet the writers at a
    /// barrier of their own, and run `read` with the region `Published`;
    /// then `end` is placed and the kernel is over.
    pub fn handoff<R>(
        &mut self,
        readers: Cond,
        reader_label: &str,
        end: Exit,
        region: RegionId,
        write: impl FnOnce(&mut Self, RegionId) -> Result<(), String>,
        read: impl FnOnce(&mut Self, RegionId, &Exit) -> Result<R, String>,
    ) -> Result<R, String> {
        self.wg("handoff")?;
        self.own_exit(&end)?;
        let i = self.region_ix(region)?;
        if self.st.regions[i].phase != Phase::RtFree {
            return Err(format!("handoff of {}: it is {:?}, a handoff starts from a Free region", self.st.regions[i].name, self.st.regions[i].phase));
        }
        self.fresh(readers.owner, readers.at)?;
        self.b.branch_scc1(&self.auth, reader_label)?;
        let at = self.b.fork();
        let saved = self.st.clone();
        self.scoped(|d| write(d, region))?;
        if self.st.regions[i].phase != Phase::RtWriting {
            return Err(format!("handoff writers left {} {:?}: they must write it", self.st.regions[i].name, self.st.regions[i].phase));
        }
        self.wait(region)?;
        self.barrier(&[Transition::Ready(region)])?;
        self.b.branch(&self.auth, &end.label)?;
        end.branched.set(true);
        self.st.reachable = false;
        self.b.resume(&self.auth, at)?;
        self.restore(saved);
        self.b.label(&self.auth, reader_label)?;
        self.b.lds_peer_stores(&self.auth, &self.st.regions[i].slots)?;
        self.check_barrier_scope()?;
        let ts: Vec<SlotTransition> = self.st.regions[i].slots.iter().map(|&id| SlotTransition::Ready(id)).collect();
        self.b.barrier(&self.auth, &ts)?;
        self.st.regions[i].phase = Phase::RtPublished;
        self.st.regions[i].stores.clear();
        let out = self.scoped(|d| read(d, region, &end))?;
        self.end(end)?;
        Ok(out)
    }
}
