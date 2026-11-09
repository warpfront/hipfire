// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Test rig: drives a [`TailPolicy`] through the retirement protocol (window fence of the
//! published generation, retirement, policy step, sealing, window lease, fill issue and
//! completion) over an in-order fake device whose fences complete at once.
//!
//! The pre-migration policy tests (and the pin / peer simulations) fed the policy
//! `(slot, generation)` completion pairs and a `released_through` closure. The rig keeps those
//! call shapes: [`Rig::retire`] & co. take the same arguments and return the retirement's
//! decisions, while the rig itself holds every capability the new API needs (the table backing,
//! the generation under its lease, the fill tickets / pending fences) and produces the completion
//! tokens from real `issue_fills` / `complete_fills` calls. A completion pair that names no issued
//! fill (the old "stale completion" tests) is turned into a token of ANOTHER bank, which the
//! policy must ignore and count as stale.

use std::cell::Cell;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use railgun::sealed_table::generation::{
    acquire_window_owned, finish_window, BlockRegistry, Expert, FenceBackend, FenceFailure,
    FenceState, FillComplete, GenerationId, PeerReaderFence, PendingFills, SlotBank, SlotSpan,
    TableBacking, WindowReadLease,
};
use railgun::Binding;

use super::{EpUnitTailConfig, TailPolicy, UnitKey, WindowUse};

/// Bytes of one slot's gate/up extent and of its down extent in the rig's block.
pub(crate) const RIG_GATE_UP: u64 = 0x1000;
pub(crate) const RIG_DOWN: u64 = 0x800;
/// Base address of the rig's pool block.
pub(crate) const RIG_BLOCK: u64 = 0x1000_0000;
/// The pointer-table allocation every rig generation is published through.
pub(crate) const RIG_TABLE: u64 = 0x7000_0000;

/// An in-order fake stream: every fence completes at once. The simplest honest
/// [`FenceBackend`] (the device timeline simulator of the race tests lives in `race`).
pub(crate) struct Instant {
    incarnation: u64,
    next: u64,
}

impl Instant {
    pub(crate) fn new(incarnation: u64) -> Self {
        Self {
            incarnation,
            next: 0,
        }
    }
}

// SAFETY: the fake stream executes everything at enqueue time, so every operation enqueued
// before `record()` returned `f` has finished when any fence is polled or waited.
unsafe impl FenceBackend for Instant {
    fn incarnation(&self) -> u64 {
        self.incarnation
    }
    fn record(&mut self) -> Result<u64, FenceFailure> {
        self.next += 1;
        Ok(self.next)
    }
    fn poll(&mut self, _fence: u64) -> FenceState {
        FenceState::Complete
    }
    fn wait(&mut self, _fence: u64) -> FenceState {
        FenceState::Complete
    }
}

/// The `released_through` closure of the old peer tests as a peer fence: asked exactly once.
struct OnceFence<'a>(Cell<Option<Box<dyn FnOnce(u64) -> u64 + 'a>>>);

// SAFETY: test double. The closure IS the test's statement of what the peers released: the old
// tests passed fixed numbers or the real `PeerBoard::released_through`, and the policy under
// test must decide exactly as it did with those.
unsafe impl PeerReaderFence for OnceFence<'_> {
    fn released_through(&self, _owner: usize, at: u64) -> u64 {
        (self
            .0
            .take()
            .expect("released_through is asked once per retirement"))(at)
    }
}

/// What one retirement decided: [`Retire`](super::Retire) without the capabilities.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RigRetire {
    pub publish: Vec<(UnitKey, usize)>,
    pub unpublish: Vec<UnitKey>,
    /// `(slot, unit, generation)` per fill.
    pub fills: Vec<(usize, UnitKey, u64)>,
}

/// A policy plus the capabilities and fake device that drive it (see the module docs).
pub(crate) struct Rig {
    pub(crate) policy: TailPolicy,
    backing: Option<TableBacking>,
    lease: Option<WindowReadLease<'static>>,
    /// Issued fills not yet completed, one fence each: `(slot, generation)`.
    pending: Vec<((usize, u64), PendingFills)>,
    copy: Instant,
    compute: Instant,
    peer: bool,
    published: bool,
    decoys: u64,
    defer_issue: bool,
    /// Fills decided and not yet issued ([`Self::defer_issue`]).
    unissued: Vec<super::FillOp>,
}

impl Deref for Rig {
    type Target = TailPolicy;
    fn deref(&self) -> &TailPolicy {
        &self.policy
    }
}

impl DerefMut for Rig {
    fn deref_mut(&mut self) -> &mut TailPolicy {
        &mut self.policy
    }
}

/// The pool block and slot extents of a `t`-slot rig bank.
pub(crate) fn rig_pool(
    t: usize,
) -> (
    railgun::sealed_table::generation::LentBlock<Expert>,
    Vec<Vec<SlotSpan>>,
) {
    let block = BlockRegistry::new(0)
        .adopt::<Expert>(RIG_BLOCK, t as u64 * (RIG_GATE_UP + RIG_DOWN), 1)
        .expect("the rig block");
    let spans = (0..t as u64)
        .map(|slot| {
            vec![
                SlotSpan {
                    offset: slot * RIG_GATE_UP,
                    bytes: RIG_GATE_UP,
                },
                SlotSpan {
                    offset: t as u64 * RIG_GATE_UP + slot * RIG_DOWN,
                    bytes: RIG_DOWN,
                },
            ]
        })
        .collect();
    (block, spans)
}

impl Rig {
    /// A policy over `t = config.tail_units` rig slots; `peer` enables PEER quarantine.
    pub(crate) fn with_peer(
        config: EpUnitTailConfig,
        frozen: bool,
        priority: HashMap<UnitKey, u32>,
        peer: bool,
    ) -> Rig {
        let (block, spans) = rig_pool(config.tail_units);
        let mut policy = TailPolicy::new(config, frozen, priority, peer, block, spans)
            .map_err(|(_, error)| error)
            .expect("the rig policy");
        let backing = policy
            .register_table_backing(vec![Binding {
                base: RIG_TABLE,
                bytes: 0x1000,
            }])
            .expect("the rig table backing");
        Rig {
            policy,
            backing: Some(backing),
            lease: None,
            pending: Vec::new(),
            copy: Instant::new(1),
            compute: Instant::new(2),
            peer,
            published: false,
            decoys: 0,
            defer_issue: false,
            unissued: Vec::new(),
        }
    }

    /// A rig without PEER quarantine (`TailPolicy::new` of the old tests).
    pub(crate) fn new(
        config: EpUnitTailConfig,
        frozen: bool,
        priority: HashMap<UnitKey, u32>,
    ) -> Rig {
        Self::with_peer(config, frozen, priority, false)
    }

    /// Warm start: the first `T` of `units` fill slots `0..`, copy, complete and publish as
    /// generation `{0, 0}`. Returns `(slot, unit, generation)` per seed.
    pub(crate) fn seed(&mut self, units: &[UnitKey]) -> Vec<(usize, UnitKey, u64)> {
        assert!(!self.published, "seed before the first retirement");
        let fills = self.policy.seed(units).expect("seed");
        let keys: Vec<_> = fills.iter().map(|op| op.key()).collect();
        let tickets = fills.into_iter().map(|op| op.ticket).collect();
        let pending = self
            .policy
            .issue_fills(tickets, &mut self.copy, &mut |_| Ok(()))
            .map_err(|(_, failure)| failure.message)
            .expect("seed fills issue");
        let done = self
            .policy
            .complete_fills(pending, &mut self.copy, true)
            .map_err(|_| "seed fills complete")
            .expect("seed fills complete");
        let backing = self.backing.take().expect("backing");
        let generation = self
            .policy
            .finish_seed(done, backing)
            .map_err(|(_, error)| error)
            .expect("finish_seed");
        self.lease = Some(
            acquire_window_owned(generation)
                .map_err(|(_, e)| e)
                .expect("lease"),
        );
        self.published = true;
        keys
    }

    /// The first retirement of a rig that was never seeded publishes an empty generation `{0, 0}`.
    fn ensure_published(&mut self) {
        if !self.published {
            self.seed(&[]);
        }
    }

    /// A completion token of ANOTHER bank (a 1-slot decoy): the policy must ignore and count it
    /// stale, whatever slot / generation the old tests named.
    pub(crate) fn decoy_token(&mut self) -> FillComplete {
        self.decoys += 1;
        let block = BlockRegistry::new(9)
            .adopt::<Expert>(0x9000_0000, 64, 1)
            .expect("decoy block");
        let (mut bank, mut cells) = SlotBank::new(
            block,
            vec![vec![SlotSpan {
                offset: 0,
                bytes: 16,
            }]],
            false,
        )
        .map_err(|(_, e)| e)
        .expect("decoy bank");
        let free = match cells.pop().expect("one cell") {
            free => free,
        };
        let (_filling, ticket) = bank
            .start_fill(free, 0)
            .map_err(|(_, e)| e)
            .expect("decoy fill");
        let mut copy = Instant::new(100 + self.decoys);
        let pending = bank
            .issue_fills(vec![ticket], &mut copy, &mut |_| Ok(()))
            .map_err(|(_, f)| f.message)
            .expect("decoy issue");
        bank.complete_fills(pending, &mut copy, true)
            .map_err(|_| "decoy complete")
            .expect("decoy complete")
            .pop()
            .expect("one token")
    }

    /// The completion tokens for the old `(slot, generation)` pairs, in order: an issued fill
    /// that is still pending completes (its fence is polled); anything else is a stale decoy.
    fn tokens(&mut self, completed: &[(usize, u64)]) -> Vec<FillComplete> {
        let mut out = Vec::new();
        for &(slot, generation) in completed {
            match self
                .pending
                .iter()
                .position(|(key, _)| *key == (slot, generation))
            {
                Some(at) => {
                    let (_, batch) = self.pending.remove(at);
                    let mut done = self
                        .policy
                        .complete_fills(batch, &mut self.copy, false)
                        .map_err(|_| "a rig fence completes at once")
                        .expect("complete");
                    out.push(done.pop().expect("one fill"));
                }
                None => out.push(self.decoy_token()),
            }
        }
        out
    }

    /// One retirement of `windows` windows. `peer`: the `released_through(m)` of the old peer
    /// tests (asked once, by the bank's `retire`).
    fn step(
        &mut self,
        core: (u64, u64),
        window: &[WindowUse],
        completed: &[(usize, u64)],
        windows: u32,
        peer: Option<Box<dyn FnOnce(u64) -> u64 + '_>>,
    ) -> RigRetire {
        self.ensure_published();
        // (1) the window fence of the published generation, (2) its retirement.
        let lease = self.lease.take().expect("a published generation");
        let pending = lease
            .enqueued(&mut self.compute)
            .map_err(|(_, f)| f.message)
            .expect("record the last-read fence");
        let fenced = finish_window(pending, &mut self.compute)
            .map_err(|_| "the rig window fences at once")
            .expect("finish_window");
        let generation = fenced.into_generation().expect("owned generation");
        let at = self.policy.next_retirement();
        let once = peer.map(|f| OnceFence(Cell::new(Some(f))));
        let peers = once
            .as_ref()
            .map(|fence| (fence as &dyn PeerReaderFence, 0usize, at));
        let (backing, retired) = self
            .policy
            .retire_generation(generation, peers)
            .map_err(|(_, e)| e)
            .expect("retire_generation");
        let tokens = self.tokens(completed);
        let mut retire = if self.peer {
            self.policy
                .retire_peer_windows(core, window, tokens, windows, &retired)
        } else {
            self.policy
                .retire_windows(core, window, tokens, windows, &retired)
        };
        assert_eq!(self.policy.refusals().0, 0, "{:?}", self.policy.refusals());
        // (6) seal over the written tables, lease; (7) issue the fills (one fence each).
        let j = self.policy.stats().retirements;
        let generation = self
            .policy
            .seal(backing, GenerationId { epoch: j, sub: 0 })
            .map_err(|(_, e)| e)
            .expect("seal");
        self.lease = Some(
            acquire_window_owned(generation)
                .map_err(|(_, e)| e)
                .expect("lease"),
        );
        let fills: Vec<_> = std::mem::take(&mut retire.fills);
        let keys = fills.iter().map(|op| op.key()).collect();
        self.unissued.extend(fills);
        if !self.defer_issue {
            self.issue_unissued();
        }
        RigRetire {
            publish: retire.publish,
            unpublish: retire.unpublish,
            fills: keys,
        }
    }

    /// From now on a retirement's fills wait in the rig ([`Self::issue_unissued`] /
    /// [`Self::abandon_unissued`]) instead of being issued at once: the fill-overlap `deferred`
    /// state of the device part.
    pub(crate) fn defer_issue(&mut self, defer: bool) {
        self.defer_issue = defer;
    }

    /// Issue every waiting fill (one fence each).
    pub(crate) fn issue_unissued(&mut self) {
        for op in std::mem::take(&mut self.unissued) {
            let key = (op.slot, op.generation);
            let batch = self
                .policy
                .issue_fills(vec![op.ticket], &mut self.copy, &mut |_| Ok(()))
                .map_err(|(_, f)| f.message)
                .expect("issue");
            self.pending.push((key, batch));
        }
    }

    /// Drop every waiting fill into the bank's `abandon_fills` (a frozen or released tail).
    pub(crate) fn abandon_unissued(&mut self) {
        let tickets = std::mem::take(&mut self.unissued)
            .into_iter()
            .map(|op| op.ticket)
            .collect();
        self.policy.abandon_fills(tickets);
    }

    /// The old `TailPolicy::retire`.
    pub(crate) fn retire(
        &mut self,
        core: (u64, u64),
        window: &[WindowUse],
        completed: &[(usize, u64)],
    ) -> RigRetire {
        self.retire_windows(core, window, completed, 1)
    }

    /// The old `TailPolicy::retire_windows`.
    pub(crate) fn retire_windows(
        &mut self,
        core: (u64, u64),
        window: &[WindowUse],
        completed: &[(usize, u64)],
        windows: u32,
    ) -> RigRetire {
        self.step(core, window, completed, windows, None)
    }

    /// The old `TailPolicy::retire_peer`.
    pub(crate) fn retire_peer(
        &mut self,
        core: (u64, u64),
        window: &[WindowUse],
        completed: &[(usize, u64)],
        released_through: impl FnOnce(u64) -> u64,
    ) -> RigRetire {
        self.retire_peer_windows(core, window, completed, 1, released_through)
    }

    /// The old `TailPolicy::retire_peer_windows`.
    pub(crate) fn retire_peer_windows(
        &mut self,
        core: (u64, u64),
        window: &[WindowUse],
        completed: &[(usize, u64)],
        windows: u32,
        released_through: impl FnOnce(u64) -> u64,
    ) -> RigRetire {
        self.step(
            core,
            window,
            completed,
            windows,
            Some(Box::new(released_through)),
        )
    }
}
