// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU property/race tests for the generation-owned resource API over a fake
//! device timeline. (The compile-fail examples are doctests on the public
//! items.)

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::Instant;

use super::*;
use crate::window::{AllocationClosure, Quiescence};

// --------------------------------------------------------------- fake device

#[derive(Debug)]
struct DevState {
    incarnation: u64,
    next: u64,
    /// Every fence `<= completed` has completed on the timeline.
    completed: u64,
    fail_at: Option<(u64, Quiescence)>,
    record_fail: Option<FenceFailure>,
    /// `wait` completes everything up to the fence (a real blocking wait);
    /// off: `wait` returns `Pending` (a timed-out wait).
    auto_complete: bool,
    recycled: Vec<u64>,
}

#[derive(Clone)]
struct Dev(Rc<RefCell<DevState>>);

impl Dev {
    fn new() -> Self {
        Dev(Rc::new(RefCell::new(DevState {
            incarnation: 1,
            next: 0,
            completed: 0,
            fail_at: None,
            record_fail: None,
            auto_complete: true,
            recycled: Vec::new(),
        })))
    }
    /// The device finishes everything recorded so far.
    fn complete_all(&self) {
        let mut s = self.0.borrow_mut();
        s.completed = s.next;
    }
    fn last(&self) -> u64 {
        self.0.borrow().next
    }
    fn set_incarnation(&self, i: u64) {
        self.0.borrow_mut().incarnation = i;
    }
    fn fail_fence(&self, id: u64, q: Quiescence) {
        self.0.borrow_mut().fail_at = Some((id, q));
    }
    fn fail_next_fence(&self, q: Quiescence) {
        let id = self.last() + 1;
        self.fail_fence(id, q);
    }
    fn set_auto_complete(&self, v: bool) {
        self.0.borrow_mut().auto_complete = v;
    }
    fn state_of(s: &DevState, f: u64) -> FenceState {
        match s.fail_at {
            Some((id, quiescence)) if id == f => FenceState::Failed(FenceFailure {
                message: format!("injected failure of fence {f}"),
                quiescence,
            }),
            _ if f <= s.completed => FenceState::Complete,
            _ => FenceState::Pending,
        }
    }
}

// SAFETY: the fake timeline reports `Complete` only for fences the test
// declared complete, so it upholds the trait contract by construction.
unsafe impl FenceBackend for Dev {
    fn incarnation(&self) -> u64 {
        self.0.borrow().incarnation
    }
    fn record(&mut self) -> Result<u64, FenceFailure> {
        let mut s = self.0.borrow_mut();
        if let Some(f) = s.record_fail.take() {
            return Err(f);
        }
        s.next += 1;
        Ok(s.next)
    }
    fn poll(&mut self, f: u64) -> FenceState {
        Dev::state_of(&self.0.borrow(), f)
    }
    fn wait(&mut self, f: u64) -> FenceState {
        let mut s = self.0.borrow_mut();
        if s.auto_complete && !matches!(s.fail_at, Some((id, _)) if id == f) {
            s.completed = s.completed.max(f);
        }
        Dev::state_of(&s, f)
    }
    fn recycle(&mut self, f: u64) {
        self.0.borrow_mut().recycled.push(f);
    }
}

/// A fake all-rank acknowledgement board.
struct Board {
    through: Cell<u64>,
    calls: RefCell<Vec<(usize, u64)>>,
}

impl Board {
    fn new(through: u64) -> Self {
        Board {
            through: Cell::new(through),
            calls: RefCell::new(Vec::new()),
        }
    }
}

// SAFETY: the fake returns exactly what the test says the peers released.
unsafe impl PeerReaderFence for Board {
    fn released_through(&self, owner: usize, at: u64) -> u64 {
        self.calls.borrow_mut().push((owner, at));
        self.through.get()
    }
}

// ---------------------------------------------------------------- rig

const BLOCK: u64 = 0x10_0000;
const TABLES: u64 = 0x80_0000;
const SLOT_BYTES: u64 = 256;

fn spans1(n: usize) -> Vec<Vec<SlotSpan>> {
    (0..n)
        .map(|i| {
            vec![SlotSpan {
                offset: i as u64 * SLOT_BYTES,
                bytes: SLOT_BYTES,
            }]
        })
        .collect()
}

struct Rig {
    dev: Dev,
    bank: SlotBank,
    registry: BlockRegistry,
    free: Vec<Option<OwnedSlot<Free>>>,
    staging: Staging,
    backings: Vec<TableBacking>,
    tables_next: u64,
}

impl Rig {
    fn new(n: usize, peer: bool) -> Rig {
        Self::with_block(n, peer, BlockRegistry::new(0), BLOCK)
    }

    fn with_block(n: usize, peer: bool, registry: BlockRegistry, base: u64) -> Rig {
        let block = registry
            .adopt::<Expert>(base, n as u64 * SLOT_BYTES, 1)
            .unwrap();
        let (bank, free) = SlotBank::new(block, spans1(n), peer)
            .map_err(|(_, e)| e)
            .unwrap();
        let staging = bank.staging();
        let mut r = Rig {
            dev: Dev::new(),
            bank,
            registry,
            free: free.into_iter().map(Some).collect(),
            staging,
            backings: Vec::new(),
            tables_next: TABLES + base,
        };
        r.add_backing();
        r
    }

    fn add_backing(&mut self) {
        let b = self
            .bank
            .register_table_backing(vec![Binding {
                base: self.tables_next,
                bytes: 64,
            }])
            .unwrap();
        self.tables_next += 0x1000;
        self.backings.push(b);
    }

    fn take_free(&mut self, i: usize) -> OwnedSlot<Free> {
        self.free[i].take().unwrap()
    }

    /// Free -> Resident through a completed fill.
    fn fill(&mut self, i: usize, occupant: u64) -> OwnedSlot<Resident> {
        let s = self.take_free(i);
        let (s, t) = self
            .bank
            .start_fill(s, occupant)
            .map_err(|(_, e)| e)
            .unwrap();
        let p = self
            .bank
            .issue_fills(vec![t], &mut self.dev, &mut |_| Ok(()))
            .unwrap();
        self.dev.complete_all();
        let mut done = self.bank.complete_fills(p, &mut self.dev, false).unwrap();
        self.bank
            .finish_fill(s, done.pop().unwrap())
            .map_err(|(_, _, e)| e)
            .unwrap()
    }

    fn seal(&mut self, epoch: u64, sub: u32) -> Result<OwnedTableGeneration, ResidencyError> {
        let Some(b) = self.backings.pop() else {
            return Err(ResidencyError::Exhausted("no free table backing".into()));
        };
        self.bank
            .seal(&self.staging, b, GenerationId { epoch, sub })
            .map_err(|(b, e)| {
                self.backings.push(b);
                e
            })
    }

    fn retire(
        &mut self,
        g: OwnedTableGeneration,
    ) -> Result<GenerationRetired, (OwnedTableGeneration, ResidencyError)> {
        let (b, done) = self.bank.retire(g, None)?;
        self.backings.push(b);
        Ok(done)
    }
}

// ------------------------------------------------------------------ blocks

#[test]
fn registry_refuses_null_empty_overflow_and_overlap() {
    let reg = BlockRegistry::new(3);
    assert!(matches!(
        reg.adopt::<Kv>(0, 64, 1),
        Err(ResidencyError::Refused(_))
    ));
    assert!(matches!(
        reg.adopt::<Kv>(0x1000, 0, 1),
        Err(ResidencyError::Refused(_))
    ));
    assert!(matches!(
        reg.adopt::<Kv>(u64::MAX - 10, 64, 1),
        Err(ResidencyError::Refused(_))
    ));
    let a = reg.adopt::<Kv>(0x1000, 0x1000, 1).unwrap();
    assert_eq!(a.rank(), 3);
    assert_eq!(
        a.extent(),
        Binding {
            base: 0x1000,
            bytes: 0x1000
        }
    );
    // Every kind of overlap with the live block.
    for (base, bytes) in [
        (0x1000, 1),
        (0x1fff, 2),
        (0x800, 0x1000),
        (0x800, 0x2000),
        (0x1800, 0x10),
    ] {
        assert!(
            matches!(
                reg.adopt::<Expert>(base, bytes, 2),
                Err(ResidencyError::Overlap(_))
            ),
            "[{base:#x}, +{bytes:#x}) must overlap"
        );
    }
    // Touching ranges do not overlap.
    let lo = reg.adopt::<Expert>(0x800, 0x800, 2).unwrap();
    let hi = reg.adopt::<Expert>(0x2000, 0x10, 2).unwrap();
    assert_ne!(lo.id(), hi.id());
    assert_eq!(reg.live_blocks(), 3);
    // Freed blocks leave the registry; a leaked (dropped) one stays.
    let b = a.free();
    assert_eq!(
        b,
        Binding {
            base: 0x1000,
            bytes: 0x1000
        }
    );
    drop(hi);
    assert_eq!(reg.live_blocks(), 2);
    assert!(matches!(
        reg.adopt::<Kv>(0x2008, 4, 3),
        Err(ResidencyError::Overlap(_))
    ));
    assert!(reg.adopt::<Kv>(0x1000, 0x800, 3).is_ok());
    let _ = lo;
}

#[test]
fn bank_new_refuses_bad_spans_and_returns_the_block() {
    let reg = BlockRegistry::new(0);
    let span = |offset, bytes| SlotSpan { offset, bytes };
    let cases: Vec<(Vec<Vec<SlotSpan>>, bool)> = vec![
        (vec![], false),
        (vec![vec![]], false),
        (vec![vec![span(0, 0)]], false),
        (
            vec![vec![
                span(0, 16),
                span(16, 16),
                span(32, 16),
                span(48, 16),
                span(64, 16),
            ]],
            false,
        ),
        (vec![vec![span(1000, 100)]], false),
        (vec![vec![span(u64::MAX, 2)]], false),
        // overlap inside a slot, across slots, and exact duplicates
        (vec![vec![span(0, 16), span(8, 16)]], true),
        (vec![vec![span(0, 16)], vec![span(15, 16)]], true),
        (vec![vec![span(0, 16)], vec![span(0, 16)]], true),
    ];
    for (slots, overlap) in cases {
        let block = reg.adopt::<Expert>(BLOCK, 1024, 1).unwrap();
        let (block, e) = SlotBank::new(block, slots.clone(), false).unwrap_err();
        assert_eq!(
            matches!(e, ResidencyError::Overlap(_)),
            overlap,
            "{slots:?}: {e}"
        );
        assert!(!overlap || matches!(e, ResidencyError::Overlap(_)));
        block.free_kv_for_test();
    }
    // Four spans of one slot, adjacent, is fine.
    let block = reg.adopt::<Expert>(BLOCK, 1024, 1).unwrap();
    let (bank, free) = SlotBank::new(
        block,
        vec![
            vec![span(0, 16), span(16, 16), span(100, 16), span(200, 16)],
            vec![span(300, 8)],
        ],
        false,
    )
    .map_err(|(_, e)| e)
    .unwrap();
    assert_eq!(bank.slots(), 2);
    assert_eq!(free[0].fill_generation(), 0);
}

impl LentBlock<Expert> {
    /// Test helper: release a block whose bank construction failed.
    fn free_kv_for_test(self) {
        self.registry.deregister(self.base, self.id);
    }
}

#[test]
fn table_backing_must_be_disjoint() {
    let mut r = Rig::new(4, false);
    let t = |base, bytes| Binding { base, bytes };
    // overlaps the block
    assert!(matches!(
        r.bank.register_table_backing(vec![t(BLOCK + 10, 8)]),
        Err(ResidencyError::Overlap(_))
    ));
    // overlaps an earlier backing
    assert!(matches!(
        r.bank
            .register_table_backing(vec![t(TABLES + BLOCK + 8, 8)]),
        Err(ResidencyError::Overlap(_))
    ));
    // overlaps within itself
    assert!(matches!(
        r.bank
            .register_table_backing(vec![t(0x9000_0000, 16), t(0x9000_0008, 16)]),
        Err(ResidencyError::Overlap(_))
    ));
    assert!(matches!(
        r.bank.register_table_backing(vec![]),
        Err(ResidencyError::Refused(_))
    ));
    assert!(matches!(
        r.bank.register_table_backing(vec![t(0, 8)]),
        Err(ResidencyError::Refused(_))
    ));
    // several disjoint tables: fine, and the failed attempts registered nothing
    let b = r
        .bank
        .register_table_backing(vec![t(0x9000_0000, 16), t(0x9000_0010, 16)])
        .unwrap();
    assert_eq!(b.tables().len(), 2);
}

// ------------------------------------------------------------------ slots

#[test]
fn slot_lifecycle_bumps_fill_generation_and_clears_occupant() {
    let mut r = Rig::new(2, false);
    let s0 = r.fill(0, 42);
    assert_eq!((s0.slot(), s0.fill_generation(), s0.occupant()), (0, 1, 42));
    assert_eq!(r.bank.occupied(), vec![(0, 42, 1)]);
    r.bank.publish(&mut r.staging, &s0).unwrap();
    r.bank.publish(&mut r.staging, &s0).unwrap(); // idempotent
    assert_eq!(r.staging.len(), 1);
    assert!(r.staging.contains(0) && !r.staging.contains(1));
    let e0 = r
        .bank
        .unpublish(&mut r.staging, s0)
        .map_err(|(_, e)| e)
        .unwrap();
    assert_eq!(r.staging.len(), 0);
    assert_eq!(r.bank.excluded_from_epoch(0), Some(0), "no seal yet");
    // No generation names it: reclaim needs a token from this bank.
    let g = r.seal(0, 0).unwrap();
    let done = r.retire(g).unwrap();
    let f0 = r.bank.reclaim(e0, &done).map_err(|(_, e)| e).unwrap();
    assert_eq!((f0.fill_generation(), f0.occupant()), (1, 0));
    assert!(r.bank.occupied().is_empty());
    let (s0, t) = r.bank.start_fill(f0, 43).map_err(|(_, e)| e).unwrap();
    assert_eq!(
        (
            s0.fill_generation(),
            t.fill_generation(),
            t.occupant(),
            t.slot()
        ),
        (2, 2, 43, 0)
    );
    assert_eq!(
        t.write_spans(),
        &[Binding {
            base: BLOCK,
            bytes: SLOT_BYTES
        }]
    );
}

#[test]
fn foreign_slots_and_tokens_are_stale() {
    let mut a = Rig::new(2, false);
    let mut b = Rig::with_block(2, false, BlockRegistry::new(1), BLOCK);
    let fb = b.take_free(0);
    let (fb, e) = a.bank.start_fill(fb, 1).unwrap_err();
    assert!(matches!(e, ResidencyError::Stale(_)), "{e}");
    // A completion of bank B (slot 0 gen 1) does not finish bank A's filling slot 0 gen 1.
    let sa = a.take_free(0);
    let (sa, ta) = a.bank.start_fill(sa, 1).map_err(|(_, e)| e).unwrap();
    let pa = a
        .bank
        .issue_fills(vec![ta], &mut a.dev, &mut |_| Ok(()))
        .unwrap();
    a.dev.complete_all();
    let mut da = a.bank.complete_fills(pa, &mut a.dev, false).unwrap();
    let (sb, tb) = b.bank.start_fill(fb, 1).map_err(|(_, e)| e).unwrap();
    let pb = b
        .bank
        .issue_fills(vec![tb], &mut b.dev, &mut |_| Ok(()))
        .unwrap();
    b.dev.complete_all();
    let mut db = b.bank.complete_fills(pb, &mut b.dev, false).unwrap();
    let (sa, db0, e) = a.bank.finish_fill(sa, db.pop().unwrap()).unwrap_err();
    assert!(matches!(e, ResidencyError::Stale(_)), "{e}");
    // The refusal returned both; the right pairs still work.
    let _ = b.bank.finish_fill(sb, db0).map_err(|(_, _, e)| e).unwrap();
    let ra = a
        .bank
        .finish_fill(sa, da.pop().unwrap())
        .map_err(|(_, _, e)| e)
        .unwrap();
    // A slot of bank A is not accepted by bank B either.
    let (ra, e) = b.bank.unpublish(&mut b.staging, ra).unwrap_err();
    assert!(matches!(e, ResidencyError::Stale(_)), "{e}");
    // Staging of another bank is refused.
    let e = a.bank.publish(&mut b.staging, &ra).unwrap_err();
    assert!(matches!(e, ResidencyError::Refused(_)), "{e}");
    a.bank.publish(&mut a.staging, &ra).unwrap();
}

#[test]
fn stale_fill_completion_for_another_slot_or_generation_is_refused() {
    let mut r = Rig::new(2, false);
    let (s0, t0) = {
        let f = r.take_free(0);
        r.bank.start_fill(f, 1).map_err(|(_, e)| e).unwrap()
    };
    let (s1, t1) = {
        let f = r.take_free(1);
        r.bank.start_fill(f, 2).map_err(|(_, e)| e).unwrap()
    };
    let p = r
        .bank
        .issue_fills(vec![t0, t1], &mut r.dev, &mut |_| Ok(()))
        .unwrap();
    assert_eq!(p.fills(), &[(0, 1), (1, 1)]);
    r.dev.complete_all();
    let mut done = r.bank.complete_fills(p, &mut r.dev, false).unwrap();
    let c1 = done.pop().unwrap();
    let c0 = done.pop().unwrap();
    // slot 1's completion cannot finish slot 0
    let (s0, c1, e) = r.bank.finish_fill(s0, c1).unwrap_err();
    assert!(matches!(e, ResidencyError::Stale(_)));
    // a forged generation mismatch (same slot, wrong fill_generation)
    let mut forged = FillComplete {
        bank: c0.bank,
        slot: 0,
        fill_generation: 9,
    };
    let (s0, back, e) = r
        .bank
        .finish_fill(s0, std::mem::replace(&mut forged, c0))
        .unwrap_err();
    assert!(matches!(e, ResidencyError::Stale(_)));
    drop(back);
    let _ = r
        .bank
        .finish_fill(s0, forged)
        .map_err(|(_, _, e)| e)
        .unwrap();
    let _ = r.bank.finish_fill(s1, c1).map_err(|(_, _, e)| e).unwrap();
}

#[test]
fn issue_fills_failures_quarantine_only_when_unproven() {
    // enqueue error after earlier copies: Unknown, whole batch quarantined, lend refused
    let mut r = Rig::new(3, false);
    let mk = |r: &mut Rig, i, occ| {
        let f = r.take_free(i);
        r.bank.start_fill(f, occ).map_err(|(_, e)| e).unwrap()
    };
    let (_s0, t0) = mk(&mut r, 0, 1);
    let (_s1, t1) = mk(&mut r, 1, 2);
    let mut seen = Vec::new();
    let f = r
        .bank
        .issue_fills(vec![t0, t1], &mut r.dev, &mut |t| {
            seen.push(t.slot());
            if t.slot() == 1 {
                Err("hipMemcpy failed".into())
            } else {
                Ok(())
            }
        })
        .unwrap_err()
        .1;
    assert_eq!(seen, vec![0, 1]);
    assert_eq!(f.quiescence, Quiescence::Unknown);
    assert!(r.bank.is_quarantined(0) && r.bank.is_quarantined(1) && !r.bank.is_quarantined(2));
    // record failure: Proven leaves the slots usable-in-principle, Unknown quarantines
    let (_s2, t2) = mk(&mut r, 2, 3);
    r.dev.0.borrow_mut().record_fail = Some(FenceFailure {
        message: "no event".into(),
        quiescence: Quiescence::Proven,
    });
    let (_t2, f) = r
        .bank
        .issue_fills(vec![t2], &mut r.dev, &mut |_| Ok(()))
        .unwrap_err();
    assert_eq!(f.quiescence, Quiescence::Proven);
    assert!(!r.bank.is_quarantined(2));
    // a quarantined bank can neither be lent nor closed
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    assert!(matches!(
        r.bank.lend_scratch(q),
        Err(ResidencyError::Quarantined(_))
    ));
    // lent bank / foreign ticket: refused before anything is enqueued
    let mut other = Rig::with_block(2, false, BlockRegistry::new(9), BLOCK);
    let (_so, to) = mk(&mut other, 0, 1);
    let mut called = false;
    let (_to, f) = r
        .bank
        .issue_fills(vec![to], &mut r.dev, &mut |_| {
            called = true;
            Ok(())
        })
        .unwrap_err();
    assert_eq!(f.quiescence, Quiescence::Proven);
    assert!(!called);
}

#[test]
fn complete_fills_pending_failed_and_wrong_incarnation() {
    let mut r = Rig::new(4, false);
    let batch = |r: &mut Rig, i: usize| {
        let f = r.take_free(i);
        let (s, t) = r.bank.start_fill(f, i as u64).map_err(|(_, e)| e).unwrap();
        let p = r
            .bank
            .issue_fills(vec![t], &mut r.dev, &mut |_| Ok(()))
            .unwrap();
        (s, p)
    };
    // delayed completion: poll says Pending and hands the batch back
    let (s0, p0) = batch(&mut r, 0);
    let p0 = match r.bank.complete_fills(p0, &mut r.dev, false) {
        Err(FillsNotDone::Pending(p)) => p,
        other => panic!("{other:?}"),
    };
    // a blocking wait that times out is Pending too
    r.dev.set_auto_complete(false);
    let p0 = match r.bank.complete_fills(p0, &mut r.dev, true) {
        Err(FillsNotDone::Pending(p)) => p,
        other => panic!("{other:?}"),
    };
    r.dev.complete_all();
    let fence_id = r.dev.last();
    let mut done = r.bank.complete_fills(p0, &mut r.dev, false).unwrap();
    assert_eq!(r.dev.0.borrow().recycled, vec![fence_id]);
    let _ = r
        .bank
        .finish_fill(s0, done.pop().unwrap())
        .map_err(|(_, _, e)| e)
        .unwrap();

    // Proven failure: reported, nothing quarantined
    let (_s1, p1) = batch(&mut r, 1);
    r.dev.fail_fence(r.dev.last(), Quiescence::Proven);
    match r.bank.complete_fills(p1, &mut r.dev, true) {
        Err(FillsNotDone::Failed { fills, failure }) => {
            assert_eq!(fills, vec![(1, 1)]);
            assert_eq!(failure.quiescence, Quiescence::Proven);
        }
        other => panic!("{other:?}"),
    }
    assert!(!r.bank.is_quarantined(1));

    // Unknown failure: the slots are quarantined forever
    let (s2, p2) = batch(&mut r, 2);
    r.dev.fail_fence(r.dev.last(), Quiescence::Unknown);
    match r.bank.complete_fills(p2, &mut r.dev, false) {
        Err(FillsNotDone::Failed { failure, .. }) => {
            assert_eq!(failure.quiescence, Quiescence::Unknown)
        }
        other => panic!("{other:?}"),
    }
    assert!(r.bank.is_quarantined(2));
    // its (forged-here) completion cannot make it resident
    let forged = FillComplete {
        bank: r.bank.id(),
        slot: 2,
        fill_generation: 1,
    };
    let (_s2, _c, e) = r.bank.finish_fill(s2, forged).unwrap_err();
    assert!(matches!(e, ResidencyError::Quarantined(_)), "{e}");

    // Wrong incarnation: the fence belongs to a recreated stream
    let (_s3, p3) = batch(&mut r, 3);
    r.dev.complete_all();
    r.dev.set_incarnation(2);
    match r.bank.complete_fills(p3, &mut r.dev, false) {
        Err(FillsNotDone::Failed { failure, .. }) => {
            assert_eq!(failure.quiescence, Quiescence::Unknown);
            assert!(
                failure.message.contains("incarnation"),
                "{}",
                failure.message
            );
        }
        other => panic!("{other:?}"),
    }
    assert!(r.bank.is_quarantined(3));
}

#[test]
fn incarnation_change_while_recording_is_unknown() {
    struct Flaky(Dev, bool);
    // SAFETY: delegates to the fake; the incarnation flips mid-record on purpose.
    unsafe impl FenceBackend for Flaky {
        fn incarnation(&self) -> u64 {
            let i = self.0.incarnation();
            if self.1 {
                i + 100
            } else {
                i
            }
        }
        fn record(&mut self) -> Result<u64, FenceFailure> {
            let id = self.0.record()?;
            self.1 = true;
            Ok(id)
        }
        fn poll(&mut self, f: u64) -> FenceState {
            self.0.poll(f)
        }
        fn wait(&mut self, f: u64) -> FenceState {
            self.0.wait(f)
        }
    }
    let mut r = Rig::new(1, false);
    let f = r.take_free(0);
    let (_s, t) = r.bank.start_fill(f, 1).map_err(|(_, e)| e).unwrap();
    let mut flaky = Flaky(r.dev.clone(), false);
    let (_t, e) = r
        .bank
        .issue_fills(vec![t], &mut flaky, &mut |_| Ok(()))
        .unwrap_err();
    assert_eq!(e.quiescence, Quiescence::Unknown);
    assert!(r.bank.is_quarantined(0));
}

// ------------------------------------------------------------ generations

#[test]
fn seal_requires_resident_members_and_increasing_ids() {
    let mut r = Rig::new(3, false);
    let s0 = r.fill(0, 10);
    let s1 = r.fill(1, 11);
    r.bank.publish(&mut r.staging, &s0).unwrap();
    r.bank.publish(&mut r.staging, &s1).unwrap();
    let g0 = r.seal(5, 2).unwrap();
    assert_eq!(
        (
            g0.id(),
            g0.len(),
            g0.contains(0),
            g0.contains(1),
            g0.contains(2)
        ),
        (GenerationId { epoch: 5, sub: 2 }, 2, true, true, false)
    );
    assert_eq!(r.bank.refs(0), 1);
    // non-increasing ids (equal, lower epoch, lower sub) are refused and the backing comes back
    r.add_backing();
    for (e, s) in [(5, 2), (5, 1), (4, 9), (0, 0)] {
        let before = r.backings.len();
        let e = r.seal(e, s).unwrap_err();
        assert!(matches!(e, ResidencyError::Refused(_)), "{e}");
        assert_eq!(r.backings.len(), before);
    }
    assert_eq!(r.bank.refs(0), 1, "a refused seal changes no reference");
    let g1 = r.seal(5, 3).unwrap();
    assert_eq!(r.bank.refs(0), 2);
    assert_eq!(r.bank.live_generations(), 2);
    // a member that left Resident (evicting) makes the staging refuse atomically
    let e1 = r
        .bank
        .unpublish(&mut r.staging, s1)
        .map_err(|(_, e)| e)
        .unwrap();
    let mut stale = r.bank.staging();
    stale.bits[0] = 0b10; // names slot 1, which is Evicting
    stale.len = 1;
    r.add_backing();
    let b = r.backings.pop().unwrap();
    let (b, e) = r
        .bank
        .seal(&stale, b, GenerationId { epoch: 9, sub: 0 })
        .unwrap_err();
    assert!(matches!(e, ResidencyError::Refused(_)), "{e}");
    assert_eq!(r.bank.refs(1), 2);
    r.backings.push(b);
    // Seal through a foreign backing is refused.
    let mut other = Rig::new(1, false);
    let foreign = other.backings.pop().unwrap();
    let (_foreign, e) = r
        .bank
        .seal(&r.staging, foreign, GenerationId { epoch: 10, sub: 0 })
        .unwrap_err();
    assert!(matches!(e, ResidencyError::Refused(_)), "{e}");
    let _ = (g0, g1, e1);
}

#[test]
fn closure_lists_tables_and_members_and_declares_into_allocation_closure() {
    let mut r = Rig::new(130, false);
    for i in [0usize, 63, 64, 129] {
        let s = r.fill(i, 100 + i as u64);
        r.bank.publish(&mut r.staging, &s).unwrap();
        std::mem::forget(s);
    }
    let g = r.seal(0, 0).unwrap();
    let c = g.closure();
    assert_eq!(c.generation(), g.id());
    assert_eq!(
        c.table_reads(),
        &[Binding {
            base: TABLES + BLOCK,
            bytes: 64
        }]
    );
    let reads: Vec<_> = c.slot_reads().collect();
    assert_eq!(
        reads.iter().map(|s| s.slot).collect::<Vec<_>>(),
        vec![0, 63, 64, 129]
    );
    for s in &reads {
        assert_eq!(s.fill_generation, 1);
        assert_eq!(
            s.spans,
            &[Binding {
                base: BLOCK + u64::from(s.slot) * SLOT_BYTES,
                bytes: SLOT_BYTES
            }]
        );
        assert_eq!(s.bank, g.bank());
    }
    // a write conflicts iff it overlaps a table or a MEMBER slot
    let w = |base, bytes| c.writes_conflict(Binding { base, bytes });
    assert!(w(BLOCK, 1));
    assert!(w(BLOCK + 63 * SLOT_BYTES + SLOT_BYTES - 1, 8));
    assert!(!w(BLOCK + SLOT_BYTES, SLOT_BYTES), "slot 1 is not a member");
    assert!(w(TABLES + BLOCK + 63, 1));
    assert!(!w(TABLES + BLOCK + 64, 100));
    assert!(!w(BLOCK, 0));
    // declare_into maps every item; an unmapped item is refused and `c` is untouched
    let mut ac = AllocationClosure::default();
    let mut asked = Vec::new();
    c.declare_into(
        |i| {
            asked.push(i);
            Some(match i {
                SlotOrTable::Table(t) => ResourceId(t as u32),
                SlotOrTable::Slot(s) => ResourceId(1000 + s),
            })
        },
        &mut ac,
    )
    .unwrap();
    assert_eq!(
        asked,
        vec![
            SlotOrTable::Table(0),
            SlotOrTable::Slot(0),
            SlotOrTable::Slot(63),
            SlotOrTable::Slot(64),
            SlotOrTable::Slot(129)
        ]
    );
    assert_eq!(ac.reads.len(), 5);
    assert!(ac.writes.is_empty());
    let mut ac2 = AllocationClosure::default();
    let e = c
        .declare_into(
            |i| (i != SlotOrTable::Slot(64)).then_some(ResourceId(1)),
            &mut ac2,
        )
        .unwrap_err();
    assert!(matches!(e, ResidencyError::Refused(_)));
    assert!(ac2.reads.is_empty());
}

// --------------------------------------------------- lease / fence protocol

fn one_generation(r: &mut Rig, epoch: u64) -> OwnedTableGeneration {
    let s = r.fill(0, 7);
    r.bank.publish(&mut r.staging, &s).unwrap();
    std::mem::forget(s);
    r.seal(epoch, 0).unwrap()
}

#[test]
fn delayed_last_reader_fence_keeps_the_generation_unretirable() {
    let mut r = Rig::new(2, false);
    let g = one_generation(&mut r, 0);
    let lease = acquire_window(&g).unwrap();
    assert_eq!(lease.generation(), g.id());
    assert_eq!(lease.closure().slot_reads().count(), 1);
    let pending = lease.enqueued(&mut r.dev).map_err(|(_, f)| f).unwrap();
    assert_eq!(pending.generation(), g.id());
    // The reader is still running: poll hands the window back; the borrow
    // (see the `retire` compile_fail doctest) keeps `g` in place meanwhile.
    let pending = match poll_window(pending, &mut r.dev) {
        Err(PendingOrQuarantined::Pending(p)) => p,
        other => panic!("{other:?}"),
    };
    r.dev.set_auto_complete(false);
    let pending = match finish_window(pending, &mut r.dev) {
        Err(PendingOrQuarantined::Pending(p)) => p,
        other => panic!("{other:?}"),
    };
    assert!(!g.is_quarantined());
    r.dev.complete_all();
    let fenced = poll_window(pending, &mut r.dev).unwrap();
    assert_eq!(fenced.generation(), g.id());
    assert!(
        fenced.into_generation().is_none(),
        "the borrowed form never owned it"
    );
    let done = r.retire(g).unwrap();
    assert_eq!(done.peer_through(), None);
    assert_eq!(r.bank.refs(0), 0);
}

#[test]
fn a_leaked_lease_blocks_retire_at_runtime() {
    let mut r = Rig::new(2, false);
    let g = one_generation(&mut r, 0);
    let lease = acquire_window(&g).unwrap();
    std::mem::forget(lease); // the borrow ends, the Arc count does not
    let (g, e) = r.retire(g).unwrap_err();
    assert!(matches!(e, ResidencyError::Refused(_)), "{e}");
    assert_eq!(r.bank.refs(0), 1);
    assert!(!g.is_quarantined());
}

#[test]
fn owned_lease_round_trips_the_generation_through_fenced() {
    let mut r = Rig::new(2, false);
    let g = one_generation(&mut r, 0);
    let id = g.id();
    let lease = acquire_window_owned(g).map_err(|(_, e)| e).unwrap();
    let pending = lease.enqueued(&mut r.dev).map_err(|(_, f)| f).unwrap();
    let fenced = finish_window(pending, &mut r.dev).unwrap();
    let g = fenced
        .into_generation()
        .expect("owned form returns the generation");
    assert_eq!(g.id(), id);
    r.retire(g).unwrap();
}

#[test]
fn unknown_fence_quarantines_the_generation_forever() {
    let mut r = Rig::new(2, false);
    let g = one_generation(&mut r, 0);
    let lease = acquire_window(&g).unwrap();
    r.dev.fail_next_fence(Quiescence::Unknown);
    let pending = lease.enqueued(&mut r.dev).map_err(|(_, f)| f).unwrap();
    let q = match finish_window(pending, &mut r.dev) {
        Err(PendingOrQuarantined::Quarantined(q)) => q,
        other => panic!("{other:?}"),
    };
    assert_eq!(q.failure().quiescence, Quiescence::Unknown);
    assert_eq!(q.generation(), g.id());
    assert!(g.is_quarantined());
    drop(q);
    // never retirable, never leasable again, so its slot never reclaims
    let (g, e) = r.retire(g).unwrap_err();
    assert!(matches!(e, ResidencyError::Quarantined(_)), "{e}");
    assert!(matches!(
        acquire_window(&g),
        Err(ResidencyError::Quarantined(_))
    ));
    let g_id = g.id();
    let (g, e) = acquire_window_owned(g).unwrap_err();
    assert!(matches!(e, ResidencyError::Quarantined(_)));
    assert_eq!(g.id(), g_id);
    assert_eq!(r.bank.refs(0), 1);
    // evict the member and retire a LATER generation of the same bank: its token
    // does not reclaim a slot the quarantined generation still names
    let s0 = r
        .bank
        .unpublish(&mut r.staging, forged_resident(&r.bank, 0, 1, 7))
        .map_err(|(_, e)| e)
        .unwrap();
    r.add_backing();
    let g1 = r.seal(1, 0).unwrap();
    assert_eq!(g1.len(), 0);
    let done = r.retire(g1).unwrap();
    let (s0, e) = r.bank.reclaim(s0, &done).unwrap_err();
    assert!(matches!(e, ResidencyError::Refused(_)), "{e}");
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    assert!(matches!(
        r.bank.lend_scratch(q),
        Err(ResidencyError::Refused(_))
    ));
    drop(s0);
}

/// Re-materialise the typed handle of a slot the test `forget`-ed.
fn forged_resident(
    bank: &SlotBank,
    slot: u32,
    fill_generation: u64,
    occupant: u64,
) -> OwnedSlot<Resident> {
    OwnedSlot {
        bank: bank.id(),
        slot,
        fill_generation,
        occupant,
        _state: PhantomData,
    }
}

#[test]
fn proven_failure_fences_the_window() {
    let mut r = Rig::new(2, false);
    let g = one_generation(&mut r, 0);
    let lease = acquire_window(&g).unwrap();
    r.dev.fail_next_fence(Quiescence::Proven);
    let pending = lease.enqueued(&mut r.dev).map_err(|(_, f)| f).unwrap();
    let fenced = finish_window(pending, &mut r.dev).unwrap();
    assert_eq!(fenced.generation(), g.id());
    assert!(!g.is_quarantined());
    r.retire(g).unwrap();
}

#[test]
fn record_failure_returns_the_lease_for_a_retry() {
    let mut r = Rig::new(2, false);
    let g = one_generation(&mut r, 0);
    let lease = acquire_window(&g).unwrap();
    r.dev.0.borrow_mut().record_fail = Some(FenceFailure {
        message: "event create".into(),
        quiescence: Quiescence::Unknown,
    });
    let (lease, f) = lease.enqueued(&mut r.dev).unwrap_err();
    assert_eq!(f.quiescence, Quiescence::Unknown);
    assert!(!g.is_quarantined(), "returned, not dropped");
    let pending = lease.enqueued(&mut r.dev).map_err(|(_, f)| f).unwrap();
    finish_window(pending, &mut r.dev).unwrap();
    r.retire(g).unwrap();
}

#[test]
fn wrong_incarnation_window_fence_quarantines() {
    let mut r = Rig::new(2, false);
    let g = one_generation(&mut r, 0);
    let lease = acquire_window(&g).unwrap();
    let pending = lease.enqueued(&mut r.dev).map_err(|(_, f)| f).unwrap();
    r.dev.complete_all();
    r.dev.set_incarnation(7); // the stream was recreated
    let q = match poll_window(pending, &mut r.dev) {
        Err(PendingOrQuarantined::Quarantined(q)) => q,
        other => panic!("{other:?}"),
    };
    assert_eq!(q.failure().quiescence, Quiescence::Unknown);
    assert!(q.failure().message.contains("incarnation"));
    assert!(g.is_quarantined());
    assert!(matches!(
        r.retire(g),
        Err((_, ResidencyError::Quarantined(_)))
    ));
}

#[test]
fn dropped_unfinished_lease_or_pending_window_quarantines() {
    let mut r = Rig::new(2, false);
    let g = one_generation(&mut r, 0);
    drop(acquire_window(&g).unwrap());
    assert!(g.is_quarantined());
    assert!(matches!(
        r.retire(g),
        Err((_, ResidencyError::Quarantined(_)))
    ));

    let mut r = Rig::new(2, false);
    let g = one_generation(&mut r, 0);
    let pending = acquire_window(&g)
        .unwrap()
        .enqueued(&mut r.dev)
        .map_err(|(_, f)| f)
        .unwrap();
    drop(pending);
    assert!(g.is_quarantined());
    assert!(matches!(
        r.retire(g),
        Err((_, ResidencyError::Quarantined(_)))
    ));

    // an owned lease dropped unfinished takes its generation with it (leaked refs)
    let mut r = Rig::new(2, false);
    let g = one_generation(&mut r, 0);
    drop(acquire_window_owned(g).map_err(|(_, e)| e).unwrap());
    assert_eq!(r.bank.refs(0), 1);
    assert_eq!(r.bank.live_generations(), 1);
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    assert!(matches!(
        r.bank.lend_scratch(q),
        Err(ResidencyError::Refused(_))
    ));
}

#[test]
fn a_slot_named_by_two_generations_reclaims_only_after_both_retire() {
    let mut r = Rig::new(3, false);
    r.add_backing();
    let s0 = r.fill(0, 10);
    let s1 = r.fill(1, 11);
    r.bank.publish(&mut r.staging, &s0).unwrap();
    r.bank.publish(&mut r.staging, &s1).unwrap();
    let g0 = r.seal(0, 0).unwrap();
    let g1 = r.seal(1, 0).unwrap();
    assert_eq!(r.bank.refs(0), 2);
    let e0 = r
        .bank
        .unpublish(&mut r.staging, s0)
        .map_err(|(_, e)| e)
        .unwrap();
    assert_eq!(r.bank.excluded_from_epoch(0), Some(2));
    let d0 = r.retire(g0).unwrap();
    assert_eq!(r.bank.refs(0), 1);
    let (e0, e) = r.bank.reclaim(e0, &d0).unwrap_err();
    assert!(matches!(e, ResidencyError::Refused(_)), "{e}");
    // start_fill of a slot still named is also refused (tamper: Free with refs)
    let d1 = r.retire(g1).unwrap();
    assert_eq!(r.bank.refs(0), 0);
    let f0 = r.bank.reclaim(e0, &d1).map_err(|(_, e)| e).unwrap();
    // A Free slot a live generation names can never be refilled.
    let mut f0 = f0;
    r.bank.recs[0].refs = 1;
    let (back, e) = r.bank.start_fill(f0, 1).unwrap_err();
    assert!(matches!(e, ResidencyError::Refused(_)), "{e}");
    f0 = back;
    r.bank.recs[0].refs = 0;
    r.bank.start_fill(f0, 1).map_err(|(_, e)| e).unwrap();
    let _ = s1;
}

#[test]
fn peer_bank_reclaim_waits_for_peer_release() {
    let mut r = Rig::new(3, true);
    r.add_backing();
    let s0 = r.fill(0, 10);
    let s1 = r.fill(1, 11);
    r.bank.publish(&mut r.staging, &s0).unwrap();
    r.bank.publish(&mut r.staging, &s1).unwrap();
    // retirement 0 seals G0 = {0,1}; retirement 1 evicts slot 0, seals G1 = {1}
    let g0 = r.seal(0, 0).unwrap();
    let e0 = r
        .bank
        .unpublish(&mut r.staging, s0)
        .map_err(|(_, e)| e)
        .unwrap();
    assert_eq!(r.bank.excluded_from_epoch(0), Some(1));
    let g1 = r.seal(1, 0).unwrap();

    // retire G0 with peers that only released through epoch 0
    let board = Board::new(0);
    let (b, d0) = r
        .bank
        .retire(g0, Some((&board, 2, 1)))
        .map_err(|(_, e)| e)
        .unwrap();
    r.backings.push(b);
    assert_eq!(
        *board.calls.borrow(),
        vec![(2, 1)],
        "released_through is called exactly once"
    );
    assert_eq!(d0.peer_through(), Some(0));
    let (e0, e) = r.bank.reclaim(e0, &d0).unwrap_err();
    assert!(
        matches!(e, ResidencyError::Refused(_)),
        "peer release 0 < excluded 1: {e}"
    );
    // a token without peer information never reclaims a peer bank's slot
    let d1 = {
        let (b, d) = r.bank.retire(g1, None).map_err(|(_, e)| e).unwrap();
        r.backings.push(b);
        d
    };
    assert_eq!(d1.peer_through(), None);
    let (e0, e) = r.bank.reclaim(e0, &d1).unwrap_err();
    assert!(matches!(e, ResidencyError::Refused(_)), "{e}");
    // a later retirement whose peers released through 1 covers it
    let g2 = {
        let st = &r.staging;
        let b = r.backings.pop().unwrap();
        r.bank
            .seal(st, b, GenerationId { epoch: 2, sub: 0 })
            .map_err(|(_, e)| e)
            .unwrap()
    };
    let board = Board::new(1);
    let (b, d2) = r
        .bank
        .retire(g2, Some((&board, 2, 2)))
        .map_err(|(_, e)| e)
        .unwrap();
    r.backings.push(b);
    assert_eq!(*board.calls.borrow(), vec![(2, 2)]);
    assert_eq!(d2.peer_through(), Some(1));
    let f0 = r.bank.reclaim(e0, &d2).map_err(|(_, e)| e).unwrap();
    assert_eq!(f0.slot(), 0);
}

#[test]
fn retire_refuses_other_banks_generations_without_calling_the_board() {
    let mut a = Rig::new(2, true);
    let mut b = Rig::with_block(2, true, BlockRegistry::new(1), BLOCK);
    let gb = one_generation(&mut b, 0);
    let board = Board::new(5);
    let (_gb, e) = a.bank.retire(gb, Some((&board, 0, 0))).unwrap_err();
    assert!(matches!(e, ResidencyError::Stale(_)), "{e}");
    assert!(board.calls.borrow().is_empty());
    // a refused retire (outstanding lease) does not consult the board either
    let ga = one_generation(&mut a, 0);
    let lease = acquire_window(&ga).unwrap();
    std::mem::forget(lease);
    let (_ga, e) = a.bank.retire(ga, Some((&board, 0, 0))).unwrap_err();
    assert!(matches!(e, ResidencyError::Refused(_)));
    assert!(board.calls.borrow().is_empty());
}

#[test]
fn generation_and_fill_counters_fail_closed_on_wrap() {
    let mut r = Rig::new(2, false);
    // fill_generation at the top: start_fill refuses and returns the slot
    r.bank.recs[0].fill_generation = u64::MAX;
    let mut f0 = r.take_free(0);
    f0.fill_generation = u64::MAX;
    let (f0, e) = r.bank.start_fill(f0, 1).unwrap_err();
    assert!(matches!(e, ResidencyError::Exhausted(_)), "{e}");
    assert_eq!(f0.fill_generation(), u64::MAX);
    assert_eq!(r.bank.phase(0), Some(SlotPhaseView::Free));
    // epoch u64::MAX would make `excluded_from = epoch + 1` wrap
    let s1 = r.fill(1, 5);
    r.bank.publish(&mut r.staging, &s1).unwrap();
    let e = r.seal(u64::MAX, 0).unwrap_err();
    assert!(matches!(e, ResidencyError::Exhausted(_)), "{e}");
    // a member whose reference count would wrap
    r.bank.recs[1].refs = u32::MAX;
    let e = r.seal(1, 0).unwrap_err();
    assert!(matches!(e, ResidencyError::Exhausted(_)), "{e}");
    r.bank.recs[1].refs = 0;
    let g = r.seal(u64::MAX - 1, u32::MAX).unwrap();
    // nothing sorts above the last id any more
    r.add_backing();
    for (e, s) in [(u64::MAX - 1, u32::MAX), (u64::MAX - 1, 0)] {
        assert!(matches!(r.seal(e, s), Err(ResidencyError::Refused(_))));
    }
    assert!(matches!(
        r.seal(u64::MAX, 0),
        Err(ResidencyError::Exhausted(_))
    ));
    let _ = (g, f0);
}

// ----------------------------------------------------- scratch loan / close

#[test]
fn scratch_loan_restore_and_end() {
    let mut r = Rig::new(5, false);
    let s0 = r.fill(0, 10);
    let s1 = r.fill(1, 11);
    // slot 2: filling with the copy issued but not yet completed by anyone
    let f2 = r.take_free(2);
    let (s2, t2) = r.bank.start_fill(f2, 12).map_err(|(_, e)| e).unwrap();
    let p2 = r
        .bank
        .issue_fills(vec![t2], &mut r.dev, &mut |_| Ok(()))
        .unwrap();
    // slot 1 evicting
    let e1 = r
        .bank
        .unpublish(&mut r.staging, s1)
        .map_err(|(_, e)| e)
        .unwrap();
    assert_eq!(r.bank.occupied(), vec![(0, 10, 1), (1, 11, 1), (2, 12, 1)]);
    let f3 = r.take_free(3);

    let q = r.bank.quiesce(&mut r.dev).unwrap();
    let loan = r.bank.lend_scratch(q).unwrap();
    assert!(r.bank.is_lent());
    assert_eq!(loan.span(), r.bank.extent());
    assert_eq!(
        loan.span(),
        Binding {
            base: BLOCK,
            bytes: 5 * SLOT_BYTES
        }
    );

    // everything that could touch the block is refused while lent
    let (f3, e) = r.bank.start_fill(f3, 1).unwrap_err();
    assert!(matches!(e, ResidencyError::Refused(_)), "{e}");
    assert!(matches!(
        r.bank.publish(&mut r.staging, &s0),
        Err(ResidencyError::Refused(_))
    ));
    assert!(matches!(r.seal(0, 0), Err(ResidencyError::Refused(_))));
    let done = {
        // a retirement token of this bank (from a scratch generation of another rig)
        let mut other = Rig::with_block(1, false, BlockRegistry::new(8), BLOCK);
        let g = one_generation(&mut other, 0);
        other.retire(g).unwrap()
    };
    let (e1, e) = r.bank.reclaim(e1, &done).unwrap_err();
    assert!(
        matches!(e, ResidencyError::Stale(_) | ResidencyError::Refused(_)),
        "{e}"
    );
    let q2 = {
        let q2 = r.bank.quiesce(&mut r.dev).unwrap();
        // lending twice is refused
        q2
    };
    assert!(matches!(
        r.bank.lend_scratch(q2),
        Err(ResidencyError::Refused(_))
    ));

    // restore: ascending, one callback per occupied slot, one fence
    let mut seen = Vec::new();
    let before = r.dev.last();
    let pr = r
        .bank
        .issue_restore(&loan, &mut r.dev, &mut |slot, occ, spans| {
            seen.push((slot, occ, spans.to_vec()));
            Ok(())
        })
        .unwrap();
    assert_eq!(r.dev.last(), before + 1);
    assert_eq!(
        seen.iter().map(|(s, o, _)| (*s, *o)).collect::<Vec<_>>(),
        vec![(0, 10), (1, 11), (2, 12)]
    );
    assert_eq!(
        seen[2].2,
        vec![Binding {
            base: BLOCK + 2 * SLOT_BYTES,
            bytes: SLOT_BYTES
        }]
    );

    // the restore fence is pending: end_scratch returns everything and stays lent
    r.dev.set_auto_complete(false);
    let (loan, pr, e) = r.bank.end_scratch(loan, pr, &mut r.dev).unwrap_err();
    assert!(matches!(e, ResidencyError::Refused(_)), "{e}");
    assert!(r.bank.is_lent());
    // a superseded restore is refused
    let pr2 = r
        .bank
        .issue_restore(&loan, &mut r.dev, &mut |_, _, _| Ok(()))
        .unwrap();
    let (loan, pr, e) = r.bank.end_scratch(loan, pr, &mut r.dev).unwrap_err();
    assert!(matches!(e, ResidencyError::Stale(_)), "{e}");
    drop(pr);
    r.dev.complete_all();
    r.bank
        .end_scratch(loan, pr2, &mut r.dev)
        .map_err(|(_, _, e)| e)
        .unwrap();
    assert!(!r.bank.is_lent());

    // normal operation resumes; the issued fill still completes
    let (_s3, _t3) = r.bank.start_fill(f3, 1).map_err(|(_, e)| e).unwrap();
    let mut done = r.bank.complete_fills(p2, &mut r.dev, false).unwrap();
    let _ = r
        .bank
        .finish_fill(s2, done.pop().unwrap())
        .map_err(|(_, _, e)| e)
        .unwrap();
    let _ = (s0, e1);
}

#[test]
fn lend_scratch_is_refused_with_a_live_generation_a_stale_quiesce_or_unissued_ticket() {
    let mut r = Rig::new(4, false);
    let g = one_generation(&mut r, 0);
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    let e = r.bank.lend_scratch(q).unwrap_err();
    assert!(
        matches!(e, ResidencyError::Refused(_)),
        "live generation: {e}"
    );
    r.retire(g).unwrap();

    // fills issued after the quiesce make it stale
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    let f1 = r.take_free(1);
    let (_s1, t1) = r.bank.start_fill(f1, 3).map_err(|(_, e)| e).unwrap();
    // an unissued ticket refuses lending even with a current quiesce
    let q_now = r.bank.quiesce(&mut r.dev).unwrap();
    let e = r.bank.lend_scratch(q_now).unwrap_err();
    assert!(
        matches!(e, ResidencyError::Refused(_)),
        "unissued ticket: {e}"
    );
    let _p = r
        .bank
        .issue_fills(vec![t1], &mut r.dev, &mut |_| Ok(()))
        .unwrap();
    let e = r.bank.lend_scratch(q).unwrap_err();
    assert!(matches!(e, ResidencyError::Stale(_)), "stale quiesce: {e}");
    // a quiesce of another bank is stale too
    let mut other = Rig::with_block(1, false, BlockRegistry::new(2), BLOCK);
    let qo = other.bank.quiesce(&mut other.dev).unwrap();
    assert!(matches!(
        r.bank.lend_scratch(qo),
        Err(ResidencyError::Stale(_))
    ));
    // a current quiesce (fills complete) lends
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    assert!(r.bank.lend_scratch(q).is_ok());
}

#[test]
fn quiesce_failures_are_failures() {
    let mut r = Rig::new(1, false);
    r.dev.set_auto_complete(false);
    assert_eq!(
        r.bank.quiesce(&mut r.dev).unwrap_err().quiescence,
        Quiescence::Unknown
    );
    r.dev.set_auto_complete(true);
    r.dev.fail_next_fence(Quiescence::Proven);
    assert_eq!(
        r.bank.quiesce(&mut r.dev).unwrap_err().quiescence,
        Quiescence::Proven
    );
}

#[test]
fn abandoned_deferred_fills_unblock_lend_and_close_and_are_never_restored() {
    let mut r = Rig::new(3, false);
    let _s0 = r.fill(0, 10);
    let f1 = r.take_free(1);
    let (s1, t1) = r.bank.start_fill(f1, 11).map_err(|(_, e)| e).unwrap();
    let f2 = r.take_free(2);
    let (s2, t2) = r.bank.start_fill(f2, 12).map_err(|(_, e)| e).unwrap();
    // two outstanding tickets: lending is refused
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    assert!(matches!(
        r.bank.lend_scratch(q),
        Err(ResidencyError::Refused(_))
    ));
    // abandon only slot 1's: slot 2 still blocks
    r.bank.abandon_fills(vec![t1]);
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    assert!(matches!(
        r.bank.lend_scratch(q),
        Err(ResidencyError::Refused(_))
    ));
    r.bank.abandon_fills(vec![t2]);
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    let loan = r.bank.lend_scratch(q).unwrap();
    // abandoned slots stay Filling, are not occupied, and are not restored
    assert_eq!(r.bank.phase(1), Some(SlotPhaseView::Filling));
    assert_eq!(r.bank.occupied(), vec![(0, 10, 1)]);
    let mut seen = Vec::new();
    let pr = r
        .bank
        .issue_restore(&loan, &mut r.dev, &mut |slot, _, _| {
            seen.push(slot);
            Ok(())
        })
        .unwrap();
    assert_eq!(seen, vec![0]);
    r.bank
        .end_scratch(loan, pr, &mut r.dev)
        .map_err(|(_, _, e)| e)
        .unwrap();
    // close accepts the abandoned Filling cells
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    let cells = vec![
        SlotCell::Resident(forged_resident(&r.bank, 0, 1, 10)),
        SlotCell::Filling(s1),
        SlotCell::Filling(s2),
    ];
    let (block, tok) = r.bank.close(cells, q).map_err(|(_, _, e)| e).unwrap();
    block.free(tok).map_err(|(_, _, e)| e).unwrap();

    // an outstanding (never abandoned) ticket still refuses close
    let mut r = Rig::new(1, false);
    let f = r.take_free(0);
    let (s, _t) = r.bank.start_fill(f, 1).map_err(|(_, e)| e).unwrap();
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    let (_bank, _cells, e) = r.bank.close(vec![SlotCell::Filling(s)], q).unwrap_err();
    assert!(matches!(e, ResidencyError::Refused(_)), "{e}");
}

#[test]
fn restore_failure_with_unknown_quiescence_quarantines_the_occupied_slots() {
    let mut r = Rig::new(2, false);
    let _s0 = r.fill(0, 10);
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    let loan = r.bank.lend_scratch(q).unwrap();
    r.dev.fail_next_fence(Quiescence::Unknown);
    let pr = r
        .bank
        .issue_restore(&loan, &mut r.dev, &mut |_, _, _| Ok(()))
        .unwrap();
    let (loan, pr, e) = r.bank.end_scratch(loan, pr, &mut r.dev).unwrap_err();
    assert!(matches!(e, ResidencyError::Quarantined(_)), "{e}");
    assert!(r.bank.is_quarantined(0) && !r.bank.is_quarantined(1));
    assert!(r.bank.is_lent(), "the pool stays lent");
    // wrong incarnation is refused without destroying anything: re-issue and finish
    r.dev.0.borrow_mut().fail_at = None;
    let pr2 = r
        .bank
        .issue_restore(&loan, &mut r.dev, &mut |_, _, _| Ok(()))
        .unwrap();
    r.dev.set_incarnation(3);
    let (loan, pr2, e) = r.bank.end_scratch(loan, pr2, &mut r.dev).unwrap_err();
    assert!(matches!(e, ResidencyError::Stale(_)), "{e}");
    drop((pr, pr2, loan));
}

#[test]
fn restore_enqueue_error_is_unknown_and_a_proven_failure_can_be_reissued() {
    let mut r = Rig::new(2, false);
    let _s0 = r.fill(0, 10);
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    let loan = r.bank.lend_scratch(q).unwrap();
    let f = r
        .bank
        .issue_restore(&loan, &mut r.dev, &mut |_, _, _| Err("copy failed".into()))
        .unwrap_err();
    assert_eq!(f.quiescence, Quiescence::Unknown);
    r.dev.fail_next_fence(Quiescence::Proven);
    let pr = r
        .bank
        .issue_restore(&loan, &mut r.dev, &mut |_, _, _| Ok(()))
        .unwrap();
    let (loan, pr, e) = r.bank.end_scratch(loan, pr, &mut r.dev).unwrap_err();
    assert!(matches!(e, ResidencyError::Refused(_)), "{e}");
    assert!(!r.bank.is_quarantined(0));
    drop(pr);
    let pr = r
        .bank
        .issue_restore(&loan, &mut r.dev, &mut |_, _, _| Ok(()))
        .unwrap();
    r.bank
        .end_scratch(loan, pr, &mut r.dev)
        .map_err(|(_, _, e)| e)
        .unwrap();
}

fn cells_of(r: &mut Rig) -> Vec<SlotCell> {
    r.free
        .iter_mut()
        .map(|s| SlotCell::from(s.take().unwrap()))
        .collect()
}

#[test]
fn close_returns_the_block_and_return_to_kv_checks_the_token() {
    let reg = BlockRegistry::new(0);
    let mut a = Rig::with_block(2, false, reg.clone(), BLOCK);
    let mut b = Rig::with_block(2, false, reg.clone(), BLOCK + 0x10_000);
    let (ca, cb) = (cells_of(&mut a), cells_of(&mut b));
    let (qa, qb) = (
        a.bank.quiesce(&mut a.dev).unwrap(),
        b.bank.quiesce(&mut b.dev).unwrap(),
    );
    let (block_a, tok_a) = a.bank.close(ca, qa).map_err(|(_, _, e)| e).unwrap();
    let (block_b, tok_b) = b.bank.close(cb, qb).map_err(|(_, _, e)| e).unwrap();
    assert_eq!(
        (tok_a.block_id(), tok_b.block_id()),
        (block_a.id(), block_b.id())
    );
    // swapped tokens are refused and returned
    let (block_a, tok_b, e) = return_to_kv(block_a, tok_b).unwrap_err();
    assert!(matches!(e, ResidencyError::Stale(_)), "{e}");
    let (block_b, tok_a, e) = block_b.free(tok_a).unwrap_err();
    assert!(matches!(e, ResidencyError::Stale(_)), "{e}");
    let mut kv_a = return_to_kv(block_a, tok_a).map_err(|(_, _, e)| e).unwrap();
    assert_eq!(
        kv_a.kv_binding(),
        Binding {
            base: BLOCK,
            bytes: 2 * SLOT_BYTES
        }
    );
    // lent again = a new loan: a token minted for the previous loan is stale
    let relent = kv_a.lend_to_experts();
    let old_loan = BlockReclaimed {
        block: relent.id(),
        loan: 0,
    };
    let (relent, old_loan, e) = return_to_kv(relent, old_loan).unwrap_err();
    assert!(matches!(e, ResidencyError::Stale(_)), "{e}");
    drop(old_loan);
    // a new bank over the relent block closes into a token for the NEW loan
    let (mut bank2, free2) = SlotBank::new(relent, spans1(2), false)
        .map_err(|(_, e)| e)
        .unwrap();
    let mut dev = Dev::new();
    let q2 = bank2.quiesce(&mut dev).unwrap();
    let cells2: Vec<SlotCell> = free2.into_iter().map(SlotCell::from).collect();
    let (block2, tok2) = bank2.close(cells2, q2).map_err(|(_, _, e)| e).unwrap();
    assert_eq!(
        block2.free(tok2).map_err(|(_, _, e)| e).unwrap().base,
        BLOCK
    );
    assert_eq!(
        block_b.free(tok_b).map_err(|(_, _, e)| e).unwrap().base,
        BLOCK + 0x10_000
    );
    assert_eq!(reg.live_blocks(), 0);
}

fn forged_free(bank: &SlotBank, slot: u32, fill_generation: u64) -> OwnedSlot<Free> {
    OwnedSlot {
        bank: bank.id(),
        slot,
        fill_generation,
        occupant: 0,
        _state: PhantomData,
    }
}

#[test]
fn close_refuses_incomplete_teardown_and_gives_everything_back() {
    let mut r = Rig::new(3, false);
    let s0 = r.fill(0, 10);
    let (f1, f2) = (r.take_free(1), r.take_free(2));
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    // a missing cell
    let (bank, mut cells, e) = r
        .bank
        .close(vec![SlotCell::from(s0), SlotCell::from(f1)], q)
        .unwrap_err();
    assert!(matches!(e, ResidencyError::Refused(_)), "{e}");
    assert_eq!(cells.len(), 2);
    assert_eq!(cells[0].phase(), SlotPhaseView::Resident);
    r.bank = bank;
    // the same slot handed back twice
    cells.push(SlotCell::Free(forged_free(&r.bank, 1, 0)));
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    let (bank, mut cells, e) = r.bank.close(cells, q).unwrap_err();
    assert!(matches!(e, ResidencyError::Refused(_)), "{e}");
    r.bank = bank;
    // a handle that does not match the bank's record
    cells.pop();
    cells.push(SlotCell::Free(forged_free(&r.bank, 2, 5)));
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    let (bank, mut cells, e) = r.bank.close(cells, q).unwrap_err();
    assert!(matches!(e, ResidencyError::Stale(_)), "{e}");
    r.bank = bank;
    cells.pop();
    cells.push(SlotCell::Free(f2));
    // a quiescence older than the bank's last issue
    let q_old = r.bank.quiesce(&mut r.dev).unwrap();
    r.bank.issue_seq += 1;
    let (bank, cells, e) = r.bank.close(cells, q_old).unwrap_err();
    assert!(matches!(e, ResidencyError::Stale(_)), "{e}");
    r.bank = bank;
    r.bank.issue_seq -= 1;
    // all in order: closes, and the block frees against its own token
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    let (block, tok) = r.bank.close(cells, q).map_err(|(_, _, e)| e).unwrap();
    block.free(tok).map_err(|(_, _, e)| e).unwrap();
    assert_eq!(r.registry.live_blocks(), 0);

    // a live generation refuses the close and the scratch loan
    let mut r2 = Rig::new(2, false);
    let g = one_generation(&mut r2, 0);
    let c2 = vec![
        SlotCell::Resident(forged_resident(&r2.bank, 0, 1, 7)),
        SlotCell::Free(r2.take_free(1)),
    ];
    let q2 = r2.bank.quiesce(&mut r2.dev).unwrap();
    let (bank2, _c2, e) = r2.bank.close(c2, q2).unwrap_err();
    assert!(matches!(e, ResidencyError::Refused(_)), "{e}");
    r2.bank = bank2;
    let q2 = r2.bank.quiesce(&mut r2.dev).unwrap();
    assert!(matches!(
        r2.bank.lend_scratch(q2),
        Err(ResidencyError::Refused(_))
    ));
    drop(g);
}

#[test]
fn close_refuses_a_quarantined_slot() {
    let mut r = Rig::new(1, false);
    let f = r.take_free(0);
    let (s, t) = r.bank.start_fill(f, 1).map_err(|(_, e)| e).unwrap();
    let _ = r
        .bank
        .issue_fills(vec![t], &mut r.dev, &mut |_| Err("boom".into()))
        .unwrap_err();
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    let (_bank, _cells, e) = r.bank.close(vec![SlotCell::from(s)], q).unwrap_err();
    assert!(matches!(e, ResidencyError::Quarantined(_)), "{e}");
}

// ------------------------------------------------------- model-based trace

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[test]
fn random_traces_match_the_reference_count_model() {
    const N: usize = 8;
    for seed in 1..=40u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut r = Rig::new(N, false);
        for _ in 0..5 {
            r.add_backing();
        }
        let mut cells: Vec<Option<SlotCell>> = r
            .free
            .iter_mut()
            .map(|s| s.take().map(SlotCell::from))
            .collect();
        let mut gens: Vec<(OwnedTableGeneration, [bool; N])> = Vec::new();
        let mut epoch = 0u64;
        let mut last_fill = [0u64; N];
        for _step in 0..400 {
            let i = rng.below(N);
            match rng.below(6) {
                0 => match cells[i].take() {
                    Some(SlotCell::Free(f)) => {
                        let occ = rng.next() % 1000;
                        match r.bank.start_fill(f, occ) {
                            Ok((s, t)) => {
                                assert!(t.fill_generation() > last_fill[i]);
                                last_fill[i] = t.fill_generation();
                                let p = r
                                    .bank
                                    .issue_fills(vec![t], &mut r.dev, &mut |_| Ok(()))
                                    .unwrap();
                                r.dev.complete_all();
                                let mut d = r.bank.complete_fills(p, &mut r.dev, false).unwrap();
                                let s = r
                                    .bank
                                    .finish_fill(s, d.pop().unwrap())
                                    .map_err(|(_, _, e)| e)
                                    .unwrap();
                                cells[i] = Some(SlotCell::Resident(s));
                            }
                            Err((f, e)) => panic!("a Free slot must start filling: {e}; {f:?}"),
                        }
                    }
                    other => cells[i] = other,
                },
                1 => {
                    if let Some(SlotCell::Resident(s)) = &cells[i] {
                        r.bank.publish(&mut r.staging, s).unwrap();
                    }
                }
                2 => match cells[i].take() {
                    Some(SlotCell::Resident(s)) => {
                        let e = r
                            .bank
                            .unpublish(&mut r.staging, s)
                            .map_err(|(_, e)| e)
                            .unwrap();
                        cells[i] = Some(SlotCell::Evicting(e));
                    }
                    other => cells[i] = other,
                },
                3 => {
                    if !r.backings.is_empty() {
                        let mut m = [false; N];
                        for (s, bit) in m.iter_mut().enumerate() {
                            *bit = r.staging.contains(s as u32);
                        }
                        let g = r.seal(epoch, 0).unwrap();
                        epoch += 1;
                        gens.push((g, m));
                    }
                }
                4 => {
                    if !gens.is_empty() {
                        let k = rng.below(gens.len());
                        let (g, _) = gens.swap_remove(k);
                        r.retire(g).map_err(|(_, e)| e).unwrap();
                    }
                }
                _ => {
                    match cells[i].take() {
                        Some(SlotCell::Evicting(e)) => {
                            // A retirement token of this bank is the reclaim permit; mint one by
                            // sealing and retiring a throw-away generation (it never names an
                            // evicting slot, so it adds no reference to slot `i`).
                            let named = gens.iter().any(|(_, m)| m[i]);
                            match r.seal(epoch, 0) {
                                Ok(g) => {
                                    epoch += 1;
                                    let done = r.retire(g).map_err(|(_, e)| e).unwrap();
                                    match r.bank.reclaim(e, &done) {
                                        Ok(f) => {
                                            assert!(!named, "reclaimed slot {i} still named by a live generation");
                                            cells[i] = Some(SlotCell::Free(f));
                                        }
                                        Err((e, err)) => {
                                            assert!(named, "refused although no generation names slot {i}: {err}");
                                            cells[i] = Some(SlotCell::Evicting(e));
                                        }
                                    }
                                }
                                // no free table backing right now
                                Err(_) => cells[i] = Some(SlotCell::Evicting(e)),
                            }
                        }
                        other => cells[i] = other,
                    }
                }
            }
            for s in 0..N {
                let want = gens.iter().filter(|(_, m)| m[s]).count() as u32;
                // the throw-away generation of branch 5 never names an evicting slot, but may name resident ones
                assert_eq!(r.bank.refs(s as u32), want, "seed {seed} slot {s}");
            }
            assert_eq!(r.bank.live_generations(), gens.len() as u64);
        }
    }
}

// ------------------------------------------------------------- publication cost

fn seal_retire_1536(rounds: usize) -> (f64, f64) {
    const N: usize = 1536;
    let mut r = Rig::new(N, false);
    // fill all slots in ONE batch (setup is not what is measured)
    let mut tickets = Vec::new();
    let mut fills = Vec::new();
    for i in 0..N {
        let f = r.take_free(i);
        let (s, t) = r.bank.start_fill(f, i as u64).map_err(|(_, e)| e).unwrap();
        fills.push(s);
        tickets.push(t);
    }
    let p = r
        .bank
        .issue_fills(tickets, &mut r.dev, &mut |_| Ok(()))
        .unwrap();
    r.dev.complete_all();
    let done = r.bank.complete_fills(p, &mut r.dev, false).unwrap();
    for (s, d) in fills.into_iter().zip(done) {
        let s = r.bank.finish_fill(s, d).map_err(|(_, _, e)| e).unwrap();
        r.bank.publish(&mut r.staging, &s).unwrap();
        std::mem::forget(s);
    }
    assert_eq!(r.staging.len(), N as u32);
    let mut seal_ns = 0u128;
    let mut retire_ns = 0u128;
    let mut backing = r.backings.pop().unwrap();
    for epoch in 0..rounds as u64 {
        let t0 = Instant::now();
        let g = r
            .bank
            .seal(&r.staging, backing, GenerationId { epoch, sub: 0 })
            .map_err(|(_, e)| e)
            .unwrap();
        let t1 = Instant::now();
        let (b, _done) = r.bank.retire(g, None).map_err(|(_, e)| e).unwrap();
        let t2 = Instant::now();
        backing = b;
        seal_ns += (t1 - t0).as_nanos();
        retire_ns += (t2 - t1).as_nanos();
    }
    assert_eq!(r.bank.refs(0), 0);
    (
        seal_ns as f64 / rounds as f64 / 1e3,
        retire_ns as f64 / rounds as f64 / 1e3,
    )
}

#[test]
fn seal_and_retire_of_a_full_1536_slot_bank_is_correct() {
    let (seal_us, retire_us) = seal_retire_1536(3);
    assert!(seal_us.is_finite() && retire_us.is_finite());
}

/// `cargo test -p railgun --release --lib seal_retire_1536_bench -- --ignored --nocapture`
#[test]
#[ignore = "timing: run with --release -- --ignored --nocapture"]
fn seal_retire_1536_bench() {
    let (_, _) = seal_retire_1536(200); // warm
    let rounds = 5000;
    let (seal_us, retire_us) = seal_retire_1536(rounds);
    println!(
        "seal+retire of a 1,536-member bank: seal {seal_us:.2} us + retire {retire_us:.2} us = {:.2} us per publication ({rounds} rounds)",
        seal_us + retire_us
    );
    assert!(
        seal_us + retire_us < 50.0,
        "publication CPU cost {:.2} us >= 50 us",
        seal_us + retire_us
    );
}

/// A refused `issue_fills` (one stale ticket in the batch) hands every ticket
/// back: nothing is stranded as an unissued `Filling`, so after the caller
/// abandons the returned tickets the bank can still be lent and closed.
#[test]
fn issue_fills_refusal_returns_the_tickets() {
    let mut r = Rig::new(2, false);
    let f0 = r.take_free(0);
    let f1 = r.take_free(1);
    let (s0, t0) = r.bank.start_fill(f0, 1).map_err(|(_, e)| e).unwrap();
    let (s1, t1) = r.bank.start_fill(f1, 2).map_err(|(_, e)| e).unwrap();
    // a foreign ticket poisons the batch: refused before any enqueue
    let mut other = Rig::with_block(1, false, BlockRegistry::new(9), BLOCK);
    let fo = other.take_free(0);
    let (_so, to) = other.bank.start_fill(fo, 9).map_err(|(_, e)| e).unwrap();
    let mut called = false;
    let (back, f) = r
        .bank
        .issue_fills(vec![t0, t1, to], &mut r.dev, &mut |_| {
            called = true;
            Ok(())
        })
        .unwrap_err();
    assert_eq!(f.quiescence, Quiescence::Proven);
    assert!(!called);
    assert_eq!(
        back.iter().map(|t| t.slot()).collect::<Vec<_>>(),
        vec![0, 1, 0]
    );
    // unissued tickets still outstanding: lend is refused
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    assert!(matches!(
        r.bank.lend_scratch(q),
        Err(ResidencyError::Refused(_))
    ));
    // the returned tickets are live capabilities: abandoning them unblocks lend and close
    r.bank.abandon_fills(back);
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    let loan = r.bank.lend_scratch(q).unwrap();
    let pr = r
        .bank
        .issue_restore(&loan, &mut r.dev, &mut |_, _, _| Ok(()))
        .unwrap();
    r.bank
        .end_scratch(loan, pr, &mut r.dev)
        .map_err(|(_, _, e)| e)
        .unwrap();
    let q = r.bank.quiesce(&mut r.dev).unwrap();
    let cells = vec![SlotCell::Filling(s0), SlotCell::Filling(s1)];
    let (block, tok) = r.bank.close(cells, q).map_err(|(_, _, e)| e).unwrap();
    block.free(tok).map_err(|(_, _, e)| e).unwrap();
}
