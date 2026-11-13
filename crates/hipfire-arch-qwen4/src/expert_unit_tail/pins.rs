// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dynamic pins of the D2 tail (`HIPFIRE_EP_UNIT_TAIL_PINS`): the pinned owner assignment of
//! D1.9 running TOGETHER with the tail.
//!
//! A pinned unit layer (`ep_unit_layer_balance == BalancedPinned`) runs, per MoE call, the pinned
//! owner assignment on every rank: an expert `e` whose pin byte is a rank is computed by that
//! rank, every other (`0xFF`, FREE) expert is dealt `idx_free % ranks` in ascending id order. The
//! static plane ([`crate::expert_ep_units::ep_unit_pin_plane`]) pins `e` to its holder `e % ranks`
//! iff the holder keeps it in its VRAM core. A tail slot is just as local to its holder as a core
//! unit, so with the tail the plane is DYNAMIC:
//!
//! ```text
//! pin(e) at retirement r  =  holder(e)   if e is core-resident on its holder (static plan)
//!                            holder(e)   if the holder published e in a slot at a retirement
//!                                        k <= r - 2 and has not unpublished it at a k' <= r - 2
//!                            FREE        otherwise
//! ```
//!
//! # What a pin can and cannot break
//!
//! A pin only chooses WHICH rank computes an expert. Every rank's full table names, for every
//! expert, an address holding exactly that unit's bytes (canonical: core VRAM, pinned host arena;
//! or a published tail slot, which the tail only ever fills with a byte copy of the canonical
//! unit). So whatever the plane says, the owner reads the right bytes: **a pin is never a
//! numeric**. What MUST hold is that every rank runs the owner assignment over the SAME plane:
//! the kernel is run redundantly on every rank from identical routed ids, and the call is exact
//! only when the ranks agree on every owner (an expert owned by two ranks is summed twice, one
//! owned by none is dropped). Plane identity is the whole safety obligation; the rest is
//! performance (a pinned unit is read from the holder's VRAM instead of its host arena).
//!
//! # Plane identity: the plane is a function of the retirement index
//!
//! Every rank counts the same retirement clock `j` (1, 2, ...): the retirement cadence
//! ([`super::RetireCadence`]) decides a group end from the routed-id sequence of the windows,
//! which every rank sees identically, and the tail's pin step runs at every group end whether or
//! not the rank's tail is frozen or has anything to retire. The warm start posts as `k = 0`.
//!
//! * POST. At its retirement `k` a rank posts the deltas of its OWN publications
//!   (`(unit, on)`: a fill published at `k` is `on`, a unit evicted at `k` is `off`), ALWAYS: a
//!   retirement with nothing to say posts an EMPTY marker, so a rank's presence through `k` is
//!   checkable. Posts are contiguous per rank (`k` after `k - 1`, the board enforces it) and carry
//!   the digest of the posting rank's planes after its own step `k`.
//! * APPLY. At its retirement `r >= 2` a rank takes from the board the posts of EVERY rank
//!   (itself included) with stamp `<= r - 2` it has not applied yet ([`PIN_LAG`]), orders them by
//!   `(k, owner)` and applies them to its planes, then uploads the changed plane bytes on its
//!   active stream (enqueued before any kernel of the next window, exactly like the table
//!   uploads). Posts newer than `r - 2` are left on the board whatever the thread timing
//!   delivered, so the applied SET depends on `r` alone. A unit has one holder and so one poster:
//!   per unit the deltas are applied in the poster's own order whatever the cross-rank
//!   interleaving, and the plane after `r` is a pure function of the posts `<= r - 2`
//!   ([`PinPlanes::apply`]).
//! * ASSERT. [`PeerBoard::pin_take`] fails ([`PinError::Missing`]) unless every rank posted
//!   through `r - 2`. Every rank also compares the digests of the other ranks' posts with its own
//!   record ([`PinProtocol::retire`]): the byte identity of the planes is checked live at every
//!   retirement, at a lag of [`PIN_LAG`].
//!
//! # Why the posts `<= r - 2` are always there (the visibility argument)
//!
//! Let rank `B` post stamp `k` at the END of its retirement `k` (a mutex-guarded board write on
//! `B`'s host thread), and rank `A` take through `r - 2` at its retirement `r`.
//!
//! 1. Retirement `r` of `A` runs after `A`'s last window of group `r` ended and `A` synchronized
//!    its stream. That window's forward made at least one trunk MoE call (a window without one is
//!    not a window of the cadence), and the call's cross-rank row exchange (or the trunk-shard
//!    fused all-reduce) consumed rows computed by `B`'s kernels of the SAME window.
//! 2. `B`'s kernels of a window of group `r` were enqueued by `B`'s host thread after `B`'s
//!    retirement `r - 1` returned (group `r - 1` ends before group `r`'s windows start, per
//!    rank), hence after `B`'s post `r - 1` (host program order).
//! 3. `A`'s synchronize returned only after those kernels executed (the exchange needed them), so
//!    `B`'s post `r - 1` precedes, in real time, `A`'s retirement `r`. Both threads take the same
//!    board mutex: a critical section that precedes another in real time is ordered before it in
//!    the mutex's total order, so `A`'s `take` sees the post.
//!
//! So at `A`'s retirement `r` every rank has posted through `r - 1`; the protocol asks only for
//! `r - 2`, a whole retirement group of margin on top of the argument. The same chain shows the
//! ranks are never more than one retirement apart: `A` at retirement `r + 1` implies `B`
//! finished `r`.
//!
//! # The failure path
//!
//! The visibility argument cannot fail while the mesh runs in lockstep (every window of every
//! rank has the same MoE calls; the pinned assignment itself needs that). If it nevertheless
//! fails (a rank missing a post, a non-contiguous post, a digest that differs, a plane upload
//! that failed) the rank does NOT guess a plane: it records the fault on the board
//! ([`PeerBoard::pin_set_fault`]) and the pinned owner assignment of EVERY rank refuses to launch
//! from then on (`UnitTailRank::pins_fault`, checked by `assign_owners`), so the decode errors
//! out through the mesh's ordinary error path instead of running a call over planes that may
//! differ. There is no deterministic "freeze the pins on all ranks" alternative: a rank that
//! found a post missing cannot know what the others applied, and freezing a subset of the ranks
//! is exactly the divergence the plane identity forbids. Two cases:
//!
//! * A missing / out-of-order post or a failed plane upload is detected BEFORE any divergent
//!   use: the detecting rank applies nothing, launches no further pinned assignment, and a
//!   healthy rank cannot complete a call whose exchange the faulted rank never joins.
//! * A digest mismatch means the by-construction identity was broken (a bug, or ranks with
//!   different plans) and is detected [`PIN_LAG`] retirements after the first diverged plane:
//!   the check is an assertion of the identity claim, not a mechanism the identity relies on.
//!
//! # A frozen tail
//!
//! A device error freezes a rank's TAIL (it stops retiring: no more fills, no more table
//! uploads), not its pin step. The frozen rank keeps the cadence clock (the cadence is never
//! reset by a freeze under this lever), posts an empty marker at every retirement and keeps
//! applying every rank's deltas: its own last publications stay pinned (its tables keep naming
//! them forever), the other ranks keep evolving theirs, and the planes stay identical. A failed
//! retirement posts an empty marker rather than the deltas it computed, so the pins never claim
//! a publication the tables did not take; an eviction whose table upload completed before the
//! retirement failed keeps its unit pinned, and the holder then reads it canonically from its
//! host arena (a performance loss on a frozen tail, never a numeric).

use std::collections::VecDeque;
use std::ops::DerefMut;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use rdna_compute::moe::QWEN4_EP_OWNER_PIN_FREE;

use super::{PeerBoard, UnitKey};

/// A post stamped `k` is applied by the retirements `r >= k + PIN_LAG` (the visibility
/// argument asks for `r >= k + 1`; the second retirement is margin).
pub const PIN_LAG: u64 = 2;

/// One change of a holder's pins: the unit is published in a slot of its holder (`on`) or went
/// back to canonical (`!on`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PinDelta {
    pub unit: UnitKey,
    pub on: bool,
}

/// One rank's post of retirement `k`: its deltas (possibly none: the presence marker) and the
/// digest of its planes after its own step `k` (`None` for the warm start, `k = 0`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PinPost {
    pub owner: usize,
    pub k: u64,
    pub deltas: Vec<PinDelta>,
    pub digest: Option<u64>,
}

/// Why the pin protocol stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PinError {
    /// A rank has not posted through the retirement the take needs (the ranks' windows are out
    /// of step).
    Missing {
        owner: usize,
        posted: Option<u64>,
        through: u64,
    },
    /// A rank was released: the mesh is going down, nothing more to apply.
    Closed,
    /// Another rank (or this one) recorded a fault.
    Fault(String),
    /// A post or a take out of its contiguous order.
    Order(String),
}

impl std::fmt::Display for PinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PinError::Missing {
                owner,
                posted,
                through,
            } => write!(
                f,
                "rank {owner} has posted its pins through {posted:?}, needed through {through} \
                 (the ranks' windows are out of step)"
            ),
            PinError::Closed => write!(f, "the pin board is closed"),
            PinError::Fault(why) => write!(f, "{why}"),
            PinError::Order(why) => write!(f, "{why}"),
        }
    }
}

#[derive(Debug)]
struct PinState {
    /// Per owner: the newest retirement posted (the warm start is 0).
    posted: Vec<Option<u64>>,
    /// Per owner: the posts (markers included) some reader has not applied yet, ascending `k`.
    posts: Vec<VecDeque<PinPost>>,
    /// Per reader: the newest stamp it took through.
    applied: Vec<Option<u64>>,
    closed: bool,
    fault: Option<String>,
}

/// The pin section of the [`PeerBoard`]: the posts every rank applies at its own pace under the
/// retirement lag (module docs).
#[derive(Debug)]
pub(super) struct PinSection {
    state: Mutex<PinState>,
    /// Set with the first fault (after it is stored in `state`), read lock-free.
    faulted: AtomicBool,
}

impl PinSection {
    pub(super) fn new(ranks: usize) -> Self {
        Self {
            state: Mutex::new(PinState {
                posted: vec![None; ranks],
                posts: (0..ranks).map(|_| VecDeque::new()).collect(),
                applied: vec![None; ranks],
                closed: false,
                fault: None,
            }),
            faulted: AtomicBool::new(false),
        }
    }

    fn lock(&self) -> MutexGuard<'_, PinState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub(super) fn post(
        &self,
        owner: usize,
        k: u64,
        deltas: Vec<PinDelta>,
        digest: Option<u64>,
    ) -> Result<(), PinError> {
        let mut state = self.lock();
        if let Some(why) = state.fault.clone() {
            return Err(PinError::Fault(why));
        }
        if state.closed {
            return Err(PinError::Closed);
        }
        let Some(last) = state.posted.get(owner).copied() else {
            return Err(PinError::Order(format!(
                "rank {owner} is outside the board"
            )));
        };
        let expected = last.map_or(0, |last| last + 1);
        if k != expected {
            return Err(PinError::Order(format!(
                "rank {owner} posted pin retirement {k}, expected {expected}"
            )));
        }
        state.posted[owner] = Some(k);
        state.posts[owner].push_back(PinPost {
            owner,
            k,
            deltas,
            digest,
        });
        Ok(())
    }

    pub(super) fn take(&self, reader: usize, through: u64) -> Result<Vec<PinPost>, PinError> {
        let mut state = self.lock();
        if let Some(why) = state.fault.clone() {
            return Err(PinError::Fault(why));
        }
        if state.closed {
            return Err(PinError::Closed);
        }
        let ranks = state.posted.len();
        if reader >= ranks {
            return Err(PinError::Order(format!(
                "rank {reader} is outside the board"
            )));
        }
        let from = match state.applied[reader] {
            None => 0,
            Some(applied) if through > applied => applied + 1,
            Some(applied) => {
                return Err(PinError::Order(format!(
                    "rank {reader} took pins through {through} after {applied}"
                )))
            }
        };
        for owner in 0..ranks {
            if !state.posted[owner].is_some_and(|posted| posted >= through) {
                return Err(PinError::Missing {
                    owner,
                    posted: state.posted[owner],
                    through,
                });
            }
        }
        let mut out: Vec<PinPost> = Vec::new();
        for owner in 0..ranks {
            out.extend(
                state.posts[owner]
                    .iter()
                    .filter(|post| (from..=through).contains(&post.k))
                    .cloned(),
            );
        }
        // Per unit one owner posts, in ascending `k`: any total order that keeps `k` ascending
        // is the same plane; `(k, owner)` is the one every rank uses.
        out.sort_by_key(|post| (post.k, post.owner));
        state.applied[reader] = Some(through);
        // Drop what every reader took.
        if let Some(low) = state
            .applied
            .iter()
            .copied()
            .collect::<Option<Vec<u64>>>()
            .and_then(|all| all.into_iter().min())
        {
            for queue in &mut state.posts {
                while queue.front().is_some_and(|post| post.k <= low) {
                    queue.pop_front();
                }
            }
        }
        Ok(out)
    }

    pub(super) fn close(&self) {
        self.lock().closed = true;
    }

    pub(super) fn set_fault(&self, why: &str) {
        let mut state = self.lock();
        if state.fault.is_none() {
            state.fault = Some(why.to_string());
        }
        self.faulted.store(true, Ordering::Release);
    }

    /// The first fault, if any. Called by the pinned owner assignment of every MoE call, so the
    /// common (no fault) answer is one atomic load, no lock.
    pub(super) fn fault(&self) -> Option<String> {
        if !self.faulted.load(Ordering::Acquire) {
            return None;
        }
        self.lock().fault.clone()
    }

    pub(super) fn kept(&self, owner: usize) -> usize {
        self.lock().posts[owner].len()
    }
}

/// One layer's plane: the static bytes (the plan's pin plane), the live image and whether the
/// image changed since its last upload.
struct PinPlane<I> {
    label: usize,
    statics: Vec<u8>,
    image: I,
    dirty: bool,
}

/// The rank's pin planes, one per pinned unit layer, ascending by label. The image storage `I`
/// is a byte slice owner (pinned host memory in the rank part, a `Vec<u8>` in the tests).
pub struct PinPlanes<I> {
    ranks: usize,
    planes: Vec<PinPlane<I>>,
    changed: u64,
    skipped: u64,
}

impl<I: DerefMut<Target = [u8]>> PinPlanes<I> {
    /// `planes`: `(label, static plane, image)` per layer; every image starts as its static plane.
    pub fn new(ranks: usize, planes: Vec<(usize, Vec<u8>, I)>) -> Result<Self, String> {
        if ranks == 0 || ranks >= usize::from(QWEN4_EP_OWNER_PIN_FREE) {
            return Err(format!("pin planes for {ranks} ranks"));
        }
        let mut planes: Vec<PinPlane<I>> = planes
            .into_iter()
            .map(|(label, statics, image)| PinPlane {
                label,
                statics,
                image,
                dirty: false,
            })
            .collect();
        planes.sort_by_key(|plane| plane.label);
        for pair in planes.windows(2) {
            if pair[0].label == pair[1].label {
                return Err(format!("layer {} has two pin planes", pair[0].label));
            }
        }
        for plane in &planes {
            if plane.image.len() != plane.statics.len() || plane.image[..] != plane.statics[..] {
                return Err(format!(
                    "layer {}: the pin image does not start as the static plane",
                    plane.label
                ));
            }
            if let Some((expert, &pin)) = plane
                .statics
                .iter()
                .enumerate()
                .find(|&(_, &pin)| pin != QWEN4_EP_OWNER_PIN_FREE && usize::from(pin) >= ranks)
            {
                return Err(format!(
                    "layer {}: expert {expert} is pinned to rank {pin}, the mesh has {ranks}",
                    plane.label
                ));
            }
        }
        Ok(Self {
            ranks,
            planes,
            changed: 0,
            skipped: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.planes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.planes.is_empty()
    }

    /// Label of plane `index` (ascending).
    pub fn label(&self, index: usize) -> usize {
        self.planes[index].label
    }

    /// The bytes of plane `index`.
    pub fn bytes(&self, index: usize) -> &[u8] {
        &self.planes[index].image
    }

    /// The plane of layer `label`.
    pub fn plane(&self, label: usize) -> Option<&[u8]> {
        self.planes
            .binary_search_by_key(&label, |plane| plane.label)
            .ok()
            .map(|at| &self.planes[at].image[..])
    }

    /// The static plane of layer `label`.
    pub fn statics(&self, label: usize) -> Option<&[u8]> {
        self.planes
            .binary_search_by_key(&label, |plane| plane.label)
            .ok()
            .map(|at| &self.planes[at].statics[..])
    }

    /// Apply the posts (already ordered by `(k, owner)`): a delta of a unit its poster holds
    /// sets the pin byte to the holder (`on`) or back to its static byte (`!on`). A delta of a
    /// layer without a plane, an expert outside the plane or a unit the poster does not hold is
    /// counted and ignored (the same on every rank). Returns the bytes that changed.
    pub fn apply(&mut self, posts: &[PinPost]) -> usize {
        let mut changed = 0usize;
        for post in posts {
            for delta in &post.deltas {
                let (layer, expert) = (delta.unit.0 as usize, delta.unit.1 as usize);
                let Ok(at) = self
                    .planes
                    .binary_search_by_key(&layer, |plane| plane.label)
                else {
                    self.skipped += 1;
                    continue;
                };
                let plane = &mut self.planes[at];
                if post.owner >= self.ranks
                    || expert >= plane.statics.len()
                    || expert % self.ranks != post.owner
                {
                    self.skipped += 1;
                    continue;
                }
                let want = if delta.on {
                    post.owner as u8
                } else {
                    plane.statics[expert]
                };
                if plane.image[expert] != want {
                    plane.image[expert] = want;
                    plane.dirty = true;
                    changed += 1;
                }
            }
        }
        self.changed += changed as u64;
        changed
    }

    /// The planes whose image changed since the last call (ascending plane index); clean after.
    pub fn take_dirty(&mut self) -> Vec<usize> {
        self.planes
            .iter_mut()
            .enumerate()
            .filter_map(|(index, plane)| std::mem::take(&mut plane.dirty).then_some(index))
            .collect()
    }

    /// FNV-1a over every plane (label, then bytes), ascending: equal planes, equal digest.
    pub fn digest(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut feed = |byte: u8| {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        };
        for plane in &self.planes {
            for byte in (plane.label as u32).to_le_bytes() {
                feed(byte);
            }
            for &byte in &plane.image[..] {
                feed(byte);
            }
        }
        hash
    }

    /// Pin bytes that currently differ from the static plane (units pinned by a slot).
    pub fn dynamic_pins(&self) -> usize {
        self.planes
            .iter()
            .map(|plane| {
                plane
                    .image
                    .iter()
                    .zip(&plane.statics)
                    .filter(|(live, stat)| live != stat)
                    .count()
            })
            .sum()
    }

    /// `(bytes changed, deltas ignored)` over the planes' life.
    pub fn counts(&self) -> (u64, u64) {
        (self.changed, self.skipped)
    }
}

/// Cumulative counters of one rank's pin protocol.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PinStats {
    /// Retirements stepped (the rank's copy of the mesh's retirement clock).
    pub retirements: u64,
    /// Deltas this rank posted.
    pub posted: u64,
    /// Plane bytes the applied posts changed.
    pub applied: u64,
    /// Digests of other ranks compared with this rank's.
    pub verified: u64,
}

/// One rank's side of the pin protocol (module docs): the clock, the posts, the applied planes.
pub struct PinProtocol<I> {
    rank: usize,
    board: Arc<PeerBoard>,
    planes: PinPlanes<I>,
    clock: u64,
    /// `(k, digest after step k)` of this rank's own posts not yet compared with the others'.
    own: VecDeque<(u64, u64)>,
    stopped: bool,
    fault: Option<String>,
    stats: PinStats,
}

impl<I: DerefMut<Target = [u8]>> PinProtocol<I> {
    pub fn new(rank: usize, board: Arc<PeerBoard>, planes: PinPlanes<I>) -> Self {
        Self {
            rank,
            board,
            planes,
            clock: 0,
            own: VecDeque::new(),
            stopped: false,
            fault: None,
            stats: PinStats::default(),
        }
    }

    pub fn planes(&self) -> &PinPlanes<I> {
        &self.planes
    }

    pub fn planes_mut(&mut self) -> &mut PinPlanes<I> {
        &mut self.planes
    }

    /// The retirement clock: how many retirements this rank stepped.
    pub fn clock(&self) -> u64 {
        self.clock
    }

    pub fn stats(&self) -> PinStats {
        self.stats
    }

    /// The fault recorded by this rank or any other (the pinned assignment must not launch).
    pub fn fault(&self) -> Option<String> {
        self.fault.clone().or_else(|| self.board.pin_fault())
    }

    /// Record `error` as the mesh's pin fault and return its message.
    pub fn fail(&mut self, why: &str) -> String {
        let why = format!("rank {} pins: {why}", self.rank);
        self.board.pin_set_fault(&why);
        self.fault.get_or_insert_with(|| why.clone());
        why
    }

    fn fail_error(&mut self, error: &PinError) -> String {
        self.fail(&error.to_string())
    }

    /// The warm start: post the deltas of the units the tail seeded as live (stamp 0).
    pub fn post_warm(&mut self, deltas: Vec<PinDelta>) -> Result<(), String> {
        if self.clock != 0 {
            return Err(self.fail("the warm start posts before the first retirement"));
        }
        self.stats.posted += deltas.len() as u64;
        match self.board.pin_post(self.rank, 0, deltas, None) {
            Ok(()) | Err(PinError::Closed) => Ok(()),
            Err(error) => Err(self.fail_error(&error)),
        }
    }

    /// One retirement of the mesh's clock, in this order and always (a frozen tail passes no
    /// deltas): advance the clock to `j`, apply every rank's posts with stamp `<= j - 2` not
    /// yet applied (the presence and digest assertions of the module docs), then post `own`
    /// (this retirement's deltas, empty for a marker) with the digest of the planes after the
    /// apply. The caller uploads [`PinPlanes::take_dirty`] on its active stream afterwards.
    /// `Err` is a recorded fault: the planes may not be trusted and the pinned assignment must
    /// not launch.
    pub fn retire(&mut self, own: Vec<PinDelta>) -> Result<(), String> {
        if let Some(why) = self.fault() {
            return Err(why);
        }
        if self.stopped {
            return Ok(());
        }
        self.clock += 1;
        let j = self.clock;
        self.stats.retirements += 1;
        if let Some(through) = j.checked_sub(PIN_LAG) {
            match self.board.pin_take(self.rank, through) {
                Ok(posts) => {
                    self.verify(&posts, through)?;
                    self.stats.applied += self.planes.apply(&posts) as u64;
                }
                Err(PinError::Closed) => {
                    self.stopped = true;
                    return Ok(());
                }
                Err(error) => return Err(self.fail_error(&error)),
            }
        }
        let digest = self.planes.digest();
        self.own.push_back((j, digest));
        self.stats.posted += own.len() as u64;
        match self.board.pin_post(self.rank, j, own, Some(digest)) {
            Ok(()) => Ok(()),
            Err(PinError::Closed) => {
                self.stopped = true;
                Ok(())
            }
            Err(error) => Err(self.fail_error(&error)),
        }
    }

    /// Compare the digests of the other ranks' posts of a take with this rank's own record.
    fn verify(&mut self, posts: &[PinPost], through: u64) -> Result<(), String> {
        for post in posts {
            let (Some(theirs), true) = (post.digest, post.owner != self.rank) else {
                continue;
            };
            let Some(&(_, mine)) = self.own.iter().find(|(k, _)| *k == post.k) else {
                continue;
            };
            self.stats.verified += 1;
            if mine != theirs {
                return Err(self.fail(&format!(
                    "the pin planes differ from rank {}'s after retirement {} \
                     (digest {mine:#018x} vs {theirs:#018x})",
                    post.owner, post.k
                )));
            }
        }
        while self.own.front().is_some_and(|&(k, _)| k <= through) {
            self.own.pop_front();
        }
        Ok(())
    }

    /// This rank is released: the others stop stepping their pins (the mesh is going down).
    pub fn close(&self) {
        self.board.pin_close();
    }
}

impl PeerBoard {
    /// Post rank `owner`'s pin deltas of retirement `k` (module docs of [`pins`](self)): every
    /// retirement posts, an empty one included, `k` contiguous from the warm start's `0`.
    pub fn pin_post(
        &self,
        owner: usize,
        k: u64,
        deltas: Vec<PinDelta>,
        digest: Option<u64>,
    ) -> Result<(), PinError> {
        self.pins.post(owner, k, deltas, digest)
    }

    /// Every rank's posts with stamp in `(last take, through]`, ordered by `(k, owner)`; fails
    /// unless every rank posted through `through`.
    pub fn pin_take(&self, reader: usize, through: u64) -> Result<Vec<PinPost>, PinError> {
        self.pins.take(reader, through)
    }

    /// A rank was released: pin steps stop quietly everywhere.
    pub fn pin_close(&self) {
        self.pins.close();
    }

    /// Record the first pin fault of the mesh.
    pub fn pin_set_fault(&self, why: &str) {
        self.pins.set_fault(why);
    }

    /// The first pin fault of the mesh, if any.
    pub fn pin_fault(&self) -> Option<String> {
        self.pins.fault()
    }

    /// Pin posts of `owner` some rank has not taken yet (diagnostics, tests).
    pub fn pin_kept_posts(&self, owner: usize) -> usize {
        self.pins.kept(owner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_unit_tail::rig::Rig;
    use crate::expert_unit_tail::{EpUnitTailConfig, RetireCadence, UnitKey, WindowUse};
    use std::collections::{HashMap, HashSet};

    const FREE: u8 = QWEN4_EP_OWNER_PIN_FREE;
    const RANKS: usize = 2;
    const LAYERS: u16 = 3;
    const EXPERTS: u16 = 16;

    fn rand(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// The static plane of a test layer: even experts of layer `l` held by rank `e % 2` stay in
    /// the core when `(l * 7 + e) % 4 == 0`.
    fn core_resident(layer: u16, expert: u16) -> bool {
        (layer as usize * 7 + expert as usize) % 4 == 0
    }

    fn static_plane(layer: u16) -> Vec<u8> {
        (0..EXPERTS)
            .map(|expert| {
                if core_resident(layer, expert) {
                    (expert as usize % RANKS) as u8
                } else {
                    FREE
                }
            })
            .collect()
    }

    fn planes() -> PinPlanes<Vec<u8>> {
        PinPlanes::new(
            RANKS,
            (0..LAYERS)
                .map(|layer| (layer as usize, static_plane(layer), static_plane(layer)))
                .collect(),
        )
        .unwrap()
    }

    fn snapshot(planes: &PinPlanes<Vec<u8>>) -> Vec<Vec<u8>> {
        (0..planes.len())
            .map(|i| planes.bytes(i).to_vec())
            .collect()
    }

    // ---- board section ----

    fn delta(layer: u16, expert: u16, on: bool) -> PinDelta {
        PinDelta {
            unit: (layer, expert),
            on,
        }
    }

    #[test]
    fn a_take_needs_every_rank_posted_through_the_stamp_and_returns_k_then_owner_order() {
        let board = PeerBoard::new(2);
        // Nothing posted: even stamp 0 is missing.
        assert_eq!(
            board.pin_take(0, 0),
            Err(PinError::Missing {
                owner: 0,
                posted: None,
                through: 0
            })
        );
        board.pin_post(0, 0, vec![delta(0, 2, true)], None).unwrap();
        // Rank 1 has not posted: the take refuses, and refusing is not consuming.
        assert_eq!(
            board.pin_take(0, 0),
            Err(PinError::Missing {
                owner: 1,
                posted: None,
                through: 0
            })
        );
        board.pin_post(1, 0, vec![delta(0, 3, true)], None).unwrap();
        board
            .pin_post(1, 1, vec![delta(0, 5, true)], Some(7))
            .unwrap();
        board.pin_post(0, 1, vec![], Some(7)).unwrap();
        // Through 0 only: the newer posts stay on the board whatever got posted.
        let take = board.pin_take(0, 0).unwrap();
        assert_eq!(
            take.iter().map(|p| (p.k, p.owner)).collect::<Vec<_>>(),
            [(0, 0), (0, 1)]
        );
        // Rank 1 takes through 1: both stamps, `(k, owner)` order, markers included.
        let take = board.pin_take(1, 1).unwrap();
        assert_eq!(
            take.iter().map(|p| (p.k, p.owner)).collect::<Vec<_>>(),
            [(0, 0), (0, 1), (1, 0), (1, 1)]
        );
        // A take may not go back or repeat.
        assert!(matches!(board.pin_take(1, 1), Err(PinError::Order(_))));
        assert!(matches!(board.pin_take(1, 0), Err(PinError::Order(_))));
        // Rank 0 takes (0, 1]: only stamp 1 is new for it.
        let take = board.pin_take(0, 1).unwrap();
        assert_eq!(
            take.iter().map(|p| (p.k, p.owner)).collect::<Vec<_>>(),
            [(1, 0), (1, 1)]
        );
        // Everything was taken by both readers: nothing is kept.
        assert_eq!(board.pin_kept_posts(0) + board.pin_kept_posts(1), 0);
    }

    #[test]
    fn posts_are_contiguous_per_rank_and_a_fault_or_close_stops_the_board() {
        let board = PeerBoard::new(2);
        assert!(matches!(
            board.pin_post(0, 1, vec![], None),
            Err(PinError::Order(_))
        ));
        board.pin_post(0, 0, vec![], None).unwrap();
        assert!(matches!(
            board.pin_post(0, 0, vec![], None),
            Err(PinError::Order(_))
        ));
        assert!(matches!(
            board.pin_post(0, 2, vec![], None),
            Err(PinError::Order(_))
        ));
        assert!(matches!(
            board.pin_post(5, 0, vec![], None),
            Err(PinError::Order(_))
        ));
        board.pin_post(0, 1, vec![], None).unwrap();
        board.pin_set_fault("first");
        board.pin_set_fault("second");
        assert_eq!(board.pin_fault().as_deref(), Some("first"));
        assert_eq!(
            board.pin_post(0, 2, vec![], None),
            Err(PinError::Fault("first".into()))
        );
        assert_eq!(board.pin_take(0, 0), Err(PinError::Fault("first".into())));
        let board = PeerBoard::new(2);
        board.pin_close();
        assert_eq!(board.pin_post(0, 0, vec![], None), Err(PinError::Closed));
        assert_eq!(board.pin_take(0, 0), Err(PinError::Closed));
    }

    // ---- planes ----

    #[test]
    fn planes_validate_their_static_bytes_and_apply_only_what_the_poster_holds() {
        let mut p = planes();
        let start = snapshot(&p);
        assert_eq!(p.digest(), planes().digest());
        // Rank 0 holds the even experts: a publication pins one, an unpublication frees it.
        let post = |owner: usize, k: u64, deltas: Vec<PinDelta>| PinPost {
            owner,
            k,
            deltas,
            digest: None,
        };
        // Layer 0, expert 2 is not core-resident ((0 * 7 + 2) % 4 != 0).
        assert!(!core_resident(0, 2) && !core_resident(0, 3));
        assert_eq!(p.apply(&[post(0, 0, vec![delta(0, 2, true)])]), 1);
        assert_eq!(p.plane(0).unwrap()[2], 0);
        assert_eq!(p.dynamic_pins(), 1);
        assert_eq!(p.take_dirty(), [0]);
        assert!(p.take_dirty().is_empty());
        // Idempotent.
        assert_eq!(p.apply(&[post(0, 1, vec![delta(0, 2, true)])]), 0);
        assert!(p.take_dirty().is_empty());
        // A rank cannot pin a unit it does not hold (expert 3 is rank 1's), a layer without a
        // plane or an expert outside it: ignored.
        assert_eq!(
            p.apply(&[post(
                0,
                2,
                vec![delta(0, 3, true), delta(9, 2, true), delta(0, 99, true)]
            )]),
            0
        );
        assert_eq!(p.counts().1, 3);
        // Off goes back to the STATIC byte: a core-resident unit stays pinned to its holder.
        assert!(core_resident(0, 0));
        assert_eq!(
            p.apply(&[post(0, 3, vec![delta(0, 0, false), delta(0, 2, false)])]),
            1
        );
        assert_eq!(snapshot(&p), start);
        // The same post seen twice in a different batch split lands on the same plane.
        let mut q = planes();
        q.apply(&[post(
            0,
            0,
            vec![delta(1, 2, true), delta(1, 2, false), delta(1, 4, true)],
        )]);
        let mut r = planes();
        r.apply(&[post(0, 0, vec![delta(1, 2, true)])]);
        r.apply(&[post(0, 1, vec![delta(1, 2, false), delta(1, 4, true)])]);
        assert_eq!(snapshot(&q), snapshot(&r));
        // Bad statics are refused.
        assert!(PinPlanes::new(2, vec![(0usize, vec![0u8, 2, FREE], vec![0u8, 2, FREE])]).is_err());
        assert!(PinPlanes::new(2, vec![(0usize, vec![0u8, 1], vec![0u8, FREE])]).is_err());
        assert!(PinPlanes::new(
            2,
            vec![
                (0usize, vec![0u8], vec![0u8]),
                (0usize, vec![0u8], vec![0u8])
            ]
        )
        .is_err());
    }

    // ---- the two-rank lockstep simulation ----

    /// The plane byte the rule gives unit `(layer, expert)` of `holder` at retirement `j`,
    /// computed from the holder's posted events alone (stamp, unit, on; posting order): core
    /// units are pinned to the holder; any other unit is pinned iff the last event of the unit
    /// with `stamp <= j - PIN_LAG` is a publication.
    fn expected_pin(
        events: &[(u64, UnitKey, bool)],
        layer: u16,
        expert: u16,
        holder: usize,
        j: u64,
    ) -> u8 {
        if core_resident(layer, expert) {
            return holder as u8;
        }
        let last = events
            .iter()
            .filter(|&&(k, unit, _)| unit == (layer, expert) && k + PIN_LAG <= j)
            .last();
        match last {
            Some(&(_, _, true)) => holder as u8,
            _ => FREE,
        }
    }

    /// One rank of the simulation: a tail policy over its own units, its pin protocol, the
    /// fills in flight and the table state its windows see.
    struct SimRank {
        rank: usize,
        policy: Rig,
        proto: PinProtocol<Vec<u8>>,
        flight: Vec<(usize, u64, u32)>,
        /// The units its tables name in a slot right now.
        live: HashSet<UnitKey>,
        /// Every delta it posted: `(stamp, unit, on)`, the warm start as stamp 0.
        events: Vec<(u64, UnitKey, bool)>,
        frozen: bool,
        /// After retirement `j` (index `j - 1`): its planes and its table state.
        planes_after: Vec<Vec<Vec<u8>>>,
        live_after: Vec<HashSet<UnitKey>>,
        done: u64,
    }

    #[derive(Debug, Default)]
    struct SimTotals {
        published: u64,
        evicted: u64,
        /// Retirements at which some plane byte differed from the static plane.
        dynamic_planes: u64,
        /// Pinned units that were no longer live (unpublished within the lag window).
        recently_unpublished: u64,
        verified: u64,
    }

    struct Sim {
        ranks: Vec<SimRank>,
        board: Arc<PeerBoard>,
        rng: u64,
        windows: Vec<u32>,
        /// Retirement at which each rank's tail freezes (`None`: never).
        freeze_at: Vec<Option<u64>>,
        totals: SimTotals,
    }

    impl Sim {
        fn new(
            t: usize,
            f: usize,
            seed: u64,
            windows: Vec<u32>,
            freeze_at: Vec<Option<u64>>,
        ) -> Self {
            let board = Arc::new(PeerBoard::new(RANKS));
            let mut ranks = Vec::new();
            for rank in 0..RANKS {
                let mut priority = HashMap::new();
                let mut owned: Vec<UnitKey> = Vec::new();
                for layer in 0..LAYERS {
                    for expert in (rank as u16..EXPERTS).step_by(RANKS) {
                        if !core_resident(layer, expert) {
                            priority.insert((layer, expert), layer as u32 * 100 + expert as u32);
                            owned.push((layer, expert));
                        }
                    }
                }
                let config = EpUnitTailConfig {
                    tail_units: t,
                    fill_per_step: f,
                    boundary_every: 1,
                };
                let mut policy = Rig::new(config, false, priority);
                let seeds = policy.seed(&owned);
                let mut proto = PinProtocol::new(rank, board.clone(), planes());
                proto
                    .post_warm(
                        seeds
                            .iter()
                            .map(|op| PinDelta {
                                unit: op.1,
                                on: true,
                            })
                            .collect(),
                    )
                    .unwrap();
                ranks.push(SimRank {
                    rank,
                    policy,
                    proto,
                    flight: Vec::new(),
                    live: seeds.iter().map(|op| op.1).collect(),
                    events: seeds.iter().map(|op| (0, op.1, true)).collect(),
                    frozen: false,
                    planes_after: Vec::new(),
                    live_after: Vec::new(),
                    done: 0,
                });
            }
            Self {
                ranks,
                board,
                rng: seed | 1,
                windows,
                freeze_at,
                totals: SimTotals::default(),
            }
        }

        /// Rank `r`'s next retirement `j`, in the production order: the tail's policy step (if
        /// the tail is alive), then the pin step with the deltas of a SUCCESSFUL step (a rank
        /// whose tail froze at `j` posts the empty marker and its tables stay as they were).
        fn retire(&mut self, r: usize) -> Result<(), String> {
            let j = self.ranks[r].done + 1;
            let windows = self.windows[j as usize - 1];
            let Sim {
                ranks,
                rng,
                freeze_at,
                totals,
                ..
            } = self;
            let me = &mut ranks[r];
            if freeze_at[r] == Some(j) {
                me.frozen = true;
            }
            let mut deltas: Vec<PinDelta> = Vec::new();
            if !me.frozen {
                let owned = EXPERTS as usize / RANKS;
                let mut uses: Vec<WindowUse> = Vec::new();
                for _ in 0..10 * windows {
                    let layer = (rand(rng) % LAYERS as u64) as u16;
                    let idx = if rand(rng) % 3 != 0 {
                        (rand(rng) as usize % 4 + (j / 12) as usize) % owned
                    } else {
                        rand(rng) as usize % owned
                    };
                    let unit = (layer, (me.rank + idx * RANKS) as u16);
                    if core_resident(unit.0, unit.1) {
                        continue;
                    }
                    let reads = 1 + (rand(rng) % 3) as u32;
                    match uses.iter_mut().find(|u| u.unit == unit) {
                        Some(u) => u.reads += reads,
                        None => uses.push(WindowUse { unit, reads }),
                    }
                }
                uses.sort_by_key(|u| u.unit);
                let mut completed: Vec<(usize, u64)> = Vec::new();
                me.flight.retain_mut(|(slot, generation, left)| {
                    if *left == 0 {
                        completed.push((*slot, *generation));
                        false
                    } else {
                        *left -= 1;
                        true
                    }
                });
                if rand(rng) % 11 == 0 {
                    completed.push((0, 1 << 40));
                }
                let retire = me.policy.retire_windows((0, 0), &uses, &completed, windows);
                for &(unit, _) in &retire.publish {
                    deltas.push(PinDelta { unit, on: true });
                }
                for &unit in &retire.unpublish {
                    deltas.push(PinDelta { unit, on: false });
                }
                for op in &retire.fills {
                    let latency = (rand(rng) % 4) as u32;
                    me.flight.push((op.0, op.2, latency));
                }
                totals.published += retire.publish.len() as u64;
                totals.evicted += retire.unpublish.len() as u64;
            }
            me.proto.retire(deltas.clone())?;
            for delta in &deltas {
                if delta.on {
                    me.live.insert(delta.unit);
                } else {
                    me.live.remove(&delta.unit);
                }
                me.events.push((j, delta.unit, delta.on));
            }
            me.planes_after.push(snapshot(me.proto.planes()));
            me.live_after.push(me.live.clone());
            me.done = j;
            Ok(())
        }

        /// Once both ranks finished retirement `j`: their planes are byte-identical, equal the
        /// rule's plane computed from the posts alone, and every dynamic pin is a unit the
        /// holder's tables name in a slot, or one it unpublished within the lag.
        fn verify(&mut self, j: u64) -> Result<(), String> {
            let at = j as usize - 1;
            if self.ranks[0].planes_after[at] != self.ranks[1].planes_after[at] {
                return Err(format!("the ranks' pin planes differ after retirement {j}"));
            }
            let mut dynamic = false;
            for holder in 0..RANKS {
                let me = &self.ranks[holder];
                for layer in 0..LAYERS {
                    for expert in (holder as u16..EXPERTS).step_by(RANKS) {
                        let unit = (layer, expert);
                        let pin = me.planes_after[at][layer as usize][expert as usize];
                        let want = expected_pin(&me.events, layer, expert, holder, j);
                        if pin != want {
                            return Err(format!(
                                "retirement {j}: {unit:?} of rank {holder} is pinned {pin:#x}, \
                                 the rule says {want:#x}"
                            ));
                        }
                        if core_resident(layer, expert) {
                            continue;
                        }
                        // The sim's own bookkeeping: `live` is the last posted event's state.
                        let posted_live = me
                            .events
                            .iter()
                            .filter(|&&(k, u, _)| u == unit && k <= j)
                            .last()
                            .is_some_and(|&(_, _, on)| on);
                        let live = me.live_after[at].contains(&unit);
                        if posted_live != live {
                            return Err(format!(
                                "retirement {j}: {unit:?} live {live} posted {posted_live}"
                            ));
                        }
                        if pin != FREE {
                            dynamic = true;
                            if !live {
                                let recent = me
                                    .events
                                    .iter()
                                    .any(|&(k, u, on)| u == unit && !on && k + 1 >= j && k <= j);
                                if !recent {
                                    return Err(format!(
                                        "retirement {j}: {unit:?} of rank {holder} is pinned but \
                                         neither live nor unpublished within the lag"
                                    ));
                                }
                                self.totals.recently_unpublished += 1;
                            }
                        }
                    }
                }
            }
            self.totals.dynamic_planes += u64::from(dynamic);
            self.totals.verified += 1;
            Ok(())
        }
    }

    /// Run `windows.len()` retirements under a random scheduler that honours the lockstep gate
    /// (a rank starts retirement `j` only after the other finished `j - 1`, so the ranks are
    /// never more than one retirement apart; either may go first), verifying every retirement.
    fn run_sim(
        t: usize,
        f: usize,
        seed: u64,
        windows: Vec<u32>,
        freeze_at: Vec<Option<u64>>,
    ) -> Result<SimTotals, String> {
        let n = windows.len() as u64;
        let mut sim = Sim::new(t, f, seed, windows, freeze_at);
        let mut checked = 0;
        while sim.ranks.iter().any(|rank| rank.done < n) {
            let eligible: Vec<usize> = (0..RANKS)
                .filter(|&r| sim.ranks[r].done < n && sim.ranks[1 - r].done >= sim.ranks[r].done)
                .collect();
            let pick = eligible[rand(&mut sim.rng) as usize % eligible.len()];
            sim.retire(pick)?;
            let both = sim.ranks.iter().map(|rank| rank.done).min().unwrap();
            while checked < both {
                checked += 1;
                sim.verify(checked)?;
            }
        }
        assert_eq!(sim.board.pin_fault(), None);
        Ok(sim.totals)
    }

    fn random_groups(seed: u64, retirements: usize) -> Vec<u32> {
        let mut state = seed | 1;
        (0..retirements)
            .map(|_| match rand(&mut state) % 5 {
                0 => 1 + (rand(&mut state) % 8) as u32,
                _ => 1,
            })
            .collect()
    }

    const SEEDS: [u64; 4] = [
        0x9e37_79b9_7f4a_7c15,
        0x2545_f491_4f6c_dd1d,
        0x1234_5678_9abc_def1,
        0xdead_beef_0bad_f00d,
    ];
    const GEOMETRIES: [(usize, usize); 4] = [(4, 2), (6, 3), (3, 1), (10, 4)];

    /// Random routed traffic, random fill latency, stale completions, grouped retirements and
    /// either rank retiring first: both ranks' planes are byte-identical after every
    /// retirement, equal the rule's plane computed from the posts, and a dynamic pin is always
    /// a live published unit or one unpublished less than two retirements ago.
    #[test]
    fn two_rank_lockstep_keeps_the_pin_planes_identical_and_exactly_the_rule() {
        let mut totals = SimTotals::default();
        for (t, f) in GEOMETRIES {
            for seed in SEEDS {
                let run = run_sim(t, f, seed, random_groups(seed, 400), vec![None, None])
                    .unwrap_or_else(|error| panic!("T={t} F={f} seed={seed:#x}: {error}"));
                totals.published += run.published;
                totals.evicted += run.evicted;
                totals.dynamic_planes += run.dynamic_planes;
                totals.recently_unpublished += run.recently_unpublished;
                totals.verified += run.verified;
            }
        }
        // The runs exercised the protocol: units were published and evicted, the planes were
        // dynamic, and pins outlived their units within the lag.
        assert!(totals.published > 0 && totals.evicted > 0, "{totals:?}");
        assert!(totals.dynamic_planes > 0, "{totals:?}");
        assert!(totals.recently_unpublished > 0, "{totals:?}");
    }

    /// One rank's tail freezes mid-run (a failed retirement posts the empty marker, then every
    /// retirement does): the frozen rank keeps its clock and keeps applying every rank's
    /// deltas, the other rank keeps evolving and the planes never diverge. Also with both
    /// ranks freezing, at different retirements.
    #[test]
    fn a_frozen_tail_keeps_counting_retirements_and_applying_pins() {
        let mut dynamic = 0u64;
        for (t, f) in GEOMETRIES {
            for seed in SEEDS {
                for freeze_at in [
                    vec![None, Some(40)],
                    vec![Some(1), None],
                    vec![Some(25), Some(60)],
                    vec![Some(2), Some(3)],
                ] {
                    let retirements = 150;
                    let run = run_sim(
                        t,
                        f,
                        seed,
                        random_groups(seed, retirements),
                        freeze_at.clone(),
                    )
                    .unwrap_or_else(|error| {
                        panic!("T={t} F={f} seed={seed:#x} freeze={freeze_at:?}: {error}")
                    });
                    dynamic += run.dynamic_planes;
                    assert_eq!(run.verified, retirements as u64);
                }
            }
        }
        assert!(dynamic > 0);
    }

    /// A frozen rank's slots stay pinned: after the freeze nothing unpublishes, so the pin of
    /// every unit its tables still name is set at the end of the run, on both ranks, and its
    /// clock kept counting.
    #[test]
    fn a_frozen_rank_keeps_its_last_published_units_pinned_on_every_rank() {
        let windows = vec![1u32; 120];
        let mut sim = Sim::new(6, 3, SEEDS[0], windows, vec![None, Some(30)]);
        let mut checked = 0;
        while sim.ranks.iter().any(|rank| rank.done < 120) {
            for r in [0usize, 1] {
                if sim.ranks[r].done < 120 {
                    sim.retire(r).unwrap();
                }
            }
            let both = sim.ranks.iter().map(|rank| rank.done).min().unwrap();
            while checked < both {
                checked += 1;
                sim.verify(checked).unwrap();
            }
        }
        let frozen = &sim.ranks[1];
        assert!(frozen.frozen && !sim.ranks[0].frozen);
        assert!(!frozen.live.is_empty(), "the frozen tail kept some units");
        for unit in &frozen.live {
            for rank in &sim.ranks {
                let plane = rank.proto.planes().plane(unit.0 as usize).unwrap();
                assert_eq!(
                    plane[unit.1 as usize], 1,
                    "{unit:?} stays pinned to its holder"
                );
            }
        }
        assert_eq!(frozen.proto.clock(), 120);
        // The frozen rank posted nothing after its freeze (posts stop growing).
        assert!(frozen.events.iter().all(|&(k, _, _)| k < 30));
    }

    /// Why the stamp gate exists: a rank that applied every post visible SO FAR (no `r - 2`
    /// gate) would apply a different set depending on which rank retires first, and the planes
    /// would diverge under the very schedules the gated protocol survives (the simulation above).
    #[test]
    fn without_the_stamp_gate_the_planes_diverge_under_the_same_schedules() {
        let mut diverged = false;
        for seed in SEEDS {
            let mut state = seed | 1;
            let mut local = [planes(), planes()];
            let mut posts: Vec<PinPost> = Vec::new();
            let mut seen = [0usize; 2];
            for j in 1..=60u64 {
                let order = if rand(&mut state) & 1 == 0 {
                    [0, 1]
                } else {
                    [1, 0]
                };
                for r in order {
                    let expert = r as u16 + (rand(&mut state) % 8) as u16 * 2;
                    posts.push(PinPost {
                        owner: r,
                        k: j,
                        deltas: vec![delta(1, expert, rand(&mut state) & 1 == 0)],
                        digest: None,
                    });
                    // Apply everything posted so far that this rank has not applied.
                    let fresh: Vec<PinPost> = posts[seen[r]..].to_vec();
                    seen[r] = posts.len();
                    local[r].apply(&fresh);
                }
                diverged |= snapshot(&local[0]) != snapshot(&local[1]);
            }
        }
        assert!(diverged, "the ungated variant never diverged");
    }

    /// A publication stamped `k` pins at retirement `k + 2` on EVERY rank, an unpublication
    /// stamped `k` frees at `k + 2`, never earlier, whichever rank goes first.
    #[test]
    fn a_delta_stamped_k_changes_every_ranks_plane_at_retirement_k_plus_two() {
        let board = Arc::new(PeerBoard::new(2));
        let mut a = PinProtocol::new(0, board.clone(), planes());
        let mut b = PinProtocol::new(1, board.clone(), planes());
        a.post_warm(vec![]).unwrap();
        b.post_warm(vec![]).unwrap();
        // Layer 0, expert 2: held by rank 0, not core-resident.
        assert!(!core_resident(0, 2));
        let script: Vec<Vec<PinDelta>> = (1..=8)
            .map(|j| match j {
                1 => vec![delta(0, 2, true)],
                4 => vec![delta(0, 2, false)],
                _ => vec![],
            })
            .collect();
        let mut pinned = Vec::new();
        for (at, deltas) in script.iter().enumerate() {
            // Alternate who goes first.
            if at % 2 == 0 {
                a.retire(deltas.clone()).unwrap();
                b.retire(vec![]).unwrap();
            } else {
                b.retire(vec![]).unwrap();
                a.retire(deltas.clone()).unwrap();
            }
            let (pa, pb) = (
                a.planes().plane(0).unwrap()[2],
                b.planes().plane(0).unwrap()[2],
            );
            assert_eq!(pa, pb, "retirement {}", at + 1);
            pinned.push(pa == 0);
        }
        // retirement: 1 2 | 3 4 5 | 6 7 8
        assert_eq!(
            pinned,
            [false, false, true, true, true, false, false, false],
            "post stamped 1 applies at 3, post stamped 4 at 6"
        );
        assert_eq!(a.planes().digest(), b.planes().digest());
        assert!(a.stats().verified > 0 && b.stats().verified > 0);
        assert_eq!(a.fault(), None);
    }

    /// A rank that runs ahead of the lockstep gate (here: three retirements while the other has
    /// not retired at all) finds a post missing: a recorded fault, never a guessed plane, and
    /// every rank's pinned assignment is told to refuse from then on.
    #[test]
    fn a_missing_post_is_a_fault_on_every_rank_and_applies_nothing() {
        let board = Arc::new(PeerBoard::new(2));
        let mut a = PinProtocol::new(0, board.clone(), planes());
        let mut b = PinProtocol::new(1, board.clone(), planes());
        a.post_warm(vec![delta(0, 2, true)]).unwrap();
        b.post_warm(vec![]).unwrap();
        a.retire(vec![]).unwrap();
        // Retirement 2 applies stamp 0 only: present on both ranks.
        a.retire(vec![]).unwrap();
        assert_eq!(
            a.planes().plane(0).unwrap()[2],
            0,
            "the warm publication applied"
        );
        // Retirement 3 needs rank 1's stamp 1: rank 1 has not retired.
        let before = snapshot(a.planes());
        let error = a.retire(vec![]).unwrap_err();
        assert!(
            error.contains("rank 1") && error.contains("out of step"),
            "{error}"
        );
        assert_eq!(snapshot(a.planes()), before, "a fault applies nothing");
        // The fault is the mesh's: the OTHER rank refuses too, before touching its planes.
        assert!(board.pin_fault().is_some());
        let before = snapshot(b.planes());
        assert!(b.retire(vec![]).is_err());
        assert!(b.fault().is_some() && a.fault().is_some());
        assert_eq!(snapshot(b.planes()), before);
        assert_eq!(a.retire(vec![]).unwrap_err(), a.fault().unwrap());
    }

    /// Planes that disagree (here: a different static plane on one rank, a plan disagreement)
    /// are caught by the digests two retirements later and become a fault on both ranks.
    #[test]
    fn diverged_planes_are_caught_by_the_digests() {
        let board = Arc::new(PeerBoard::new(2));
        let mut a = PinProtocol::new(0, board.clone(), planes());
        let mut other = static_plane(0);
        other[1] = if other[1] == FREE { 1 } else { FREE };
        let diverged = PinPlanes::new(
            RANKS,
            (0..LAYERS)
                .map(|layer| {
                    let plane = if layer == 0 {
                        other.clone()
                    } else {
                        static_plane(layer)
                    };
                    (layer as usize, plane.clone(), plane)
                })
                .collect(),
        )
        .unwrap();
        let mut b = PinProtocol::new(1, board.clone(), diverged);
        a.post_warm(vec![]).unwrap();
        b.post_warm(vec![]).unwrap();
        let mut caught = None;
        for j in 1..=6 {
            for result in [a.retire(vec![]), b.retire(vec![])] {
                if let Err(error) = result {
                    caught.get_or_insert((j, error));
                }
            }
        }
        let (at, error) = caught.expect("the digest mismatch was never caught");
        assert_eq!(
            at, 3,
            "stamp 1's digest is compared at retirement 3: {error}"
        );
        assert!(error.contains("differ"), "{error}");
        assert!(a.fault().is_some() && b.fault().is_some());
    }

    /// A released rank closes the board: the other ranks' pin steps stop quietly (no fault, no
    /// more applies) because the mesh is going down.
    #[test]
    fn a_released_rank_stops_the_others_quietly() {
        let board = Arc::new(PeerBoard::new(2));
        let mut a = PinProtocol::new(0, board.clone(), planes());
        let mut b = PinProtocol::new(1, board.clone(), planes());
        a.post_warm(vec![]).unwrap();
        b.post_warm(vec![]).unwrap();
        for _ in 0..3 {
            a.retire(vec![]).unwrap();
            b.retire(vec![]).unwrap();
        }
        b.close();
        assert_eq!(a.retire(vec![]), Ok(()));
        assert_eq!(b.retire(vec![]), Ok(()));
        assert_eq!(a.fault(), None);
    }

    /// Real threads racing through the same lockstep gate (a rank starts retirement `j` only
    /// after the other finished `j - 1`) with random jitter: the ranks post and take in
    /// whatever order the scheduler picks and the planes are still identical after every
    /// retirement.
    #[test]
    fn racing_threads_under_the_lockstep_gate_keep_identical_planes() {
        use std::sync::Condvar;
        const N: u64 = 300;
        for seed in SEEDS {
            let board = Arc::new(PeerBoard::new(2));
            let gate = Arc::new((Mutex::new([0u64; 2]), Condvar::new()));
            let mut scripts: Vec<Vec<Vec<PinDelta>>> = Vec::new();
            let mut state = seed | 1;
            for rank in 0..RANKS {
                scripts.push(
                    (0..N)
                        .map(|_| {
                            (0..rand(&mut state) % 4)
                                .map(|_| {
                                    let layer = (rand(&mut state) % LAYERS as u64) as u16;
                                    let expert = rank as u16
                                        + (rand(&mut state) % (EXPERTS as u64 / 2)) as u16 * 2;
                                    PinDelta {
                                        unit: (layer, expert),
                                        on: rand(&mut state) & 1 == 0,
                                    }
                                })
                                .collect()
                        })
                        .collect(),
                );
            }
            let histories: Vec<Vec<Vec<Vec<u8>>>> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..RANKS)
                    .map(|rank| {
                        let (board, gate, script) =
                            (board.clone(), gate.clone(), scripts[rank].clone());
                        scope.spawn(move || {
                            let mut proto = PinProtocol::new(rank, board, planes());
                            proto.post_warm(vec![]).unwrap();
                            let mut jitter = seed ^ ((rank as u64 + 1) * 0x9e37_79b9);
                            let mut history = Vec::new();
                            for j in 1..=N {
                                {
                                    let (lock, cv) = &*gate;
                                    let mut done = lock.lock().unwrap();
                                    while done[1 - rank] < j - 1 {
                                        done = cv.wait(done).unwrap();
                                    }
                                }
                                for _ in 0..rand(&mut jitter) % 40 {
                                    std::thread::yield_now();
                                }
                                proto.retire(script[j as usize - 1].clone()).unwrap();
                                history.push(snapshot(proto.planes()));
                                let (lock, cv) = &*gate;
                                lock.lock().unwrap()[rank] = j;
                                cv.notify_all();
                            }
                            history
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for j in 0..N as usize {
                assert_eq!(
                    histories[0][j],
                    histories[1][j],
                    "seed {seed:#x} retirement {}",
                    j + 1
                );
            }
            assert_eq!(board.pin_fault(), None);
        }
    }

    /// The board does not grow without bound: with both ranks stepping, the posts every rank
    /// took are dropped.
    #[test]
    fn the_board_drops_posts_every_rank_applied() {
        let board = Arc::new(PeerBoard::new(2));
        let mut a = PinProtocol::new(0, board.clone(), planes());
        let mut b = PinProtocol::new(1, board.clone(), planes());
        a.post_warm(vec![]).unwrap();
        b.post_warm(vec![]).unwrap();
        for _ in 0..200 {
            a.retire(vec![delta(0, 2, true)]).unwrap();
            b.retire(vec![]).unwrap();
        }
        assert!(board.pin_kept_posts(0) <= 4 && board.pin_kept_posts(1) <= 4);
    }

    /// Cadence groups of the two ranks agree on the retirement clock the pins use (the clock
    /// is the number of group ends, not the number of policy retirements of either rank).
    #[test]
    fn the_retirement_clock_is_the_cadences_on_every_rank() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut cadences = [
            RetireCadence::new(4096, 1 << 18),
            RetireCadence::new(4096, 1 << 18),
        ];
        let mut clocks = [0u64; 2];
        for _ in 0..2000 {
            let routed = 64 + rand(&mut state) % 2900;
            for (cadence, clock) in cadences.iter_mut().zip(&mut clocks) {
                if cadence.window_end(routed).is_some() {
                    *clock += 1;
                }
            }
            assert_eq!(clocks[0], clocks[1]);
        }
        assert!(clocks[0] > 100);
    }
}
