// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Bounded waiting room for serve admission (spec §5.3 S3).
//!
//! Pure host policy — no GPU code, no dependency on `hipfire-arch-qwen35`.
//! This module replaces the parent's **R-A4** admit-or-reject behavior for
//! the new robust multi-slot mode: when a [`SubmitRequest`](crate::serve::SubmitRequest)
//! cannot currently be admitted because every resident session is generating
//! and no slot/credits are free, it is enqueued here instead of rejected,
//! up to finite count and byte limits.
//!
//! # Engine wiring (NOT this slice)
//!
//! The serve engine will:
//!
//! 1. Call [`WaitQueue::try_enqueue`] when an admit would currently `Reject`
//!    because every resident session is generating
//!    (`serve_engine.rs` ~R-A4). A [`WaitError`] is surfaced to the caller as
//!    a typed overload error (HTTP 429 for bounded queue rejection).
//! 2. Call [`WaitQueue::pop_ready`] when a slot frees, **before** cold-admitting
//!    a newly arrived `SubmitRequest`, so queued work keeps its admission age
//!    (spec §5.3 S3: "Preserve admission age across chunking and scheduling
//!    retries").
//! 3. Call [`WaitQueue::expire`] each scheduler tick to drop waiters whose
//!    queue deadline has passed and surface a timeout error to their clients.
//!
//! That wiring lives in `hipfire-arch-qwen35::serve_engine` and is out of
//! scope for this module.
//!
//! # Fairness model
//!
//! Fairness among waiters is **FIFO** by enqueue order. [`FairQueue`](crate::serve_fairness::FairQueue)
//! in `serve_fairness.rs` handles in-engine rotation *after* a request is
//! admitted; this module is only the waiting room for work that could not get
//! a slot/credits yet. The spec's bounded-backfill rule (stop admitting
//! younger conflicting requests once the oldest is individually feasible but
//! credit-starved) is enforced by the engine consulting [`queued_count`] and
//! [`pop_ready`], not by this module.
//!
//! # Zero config
//!
//! `max_count == 0` and `max_bytes == 0` are rejected at construction. The
//! new multi-slot mode must not reinterpret zero as uncapped (spec §5.3:
//! "`serve.max_queue=0` currently means uncapped; the new robust multi-slot
//! mode must reject that combination"). `timeout_ticks == 0` is permitted and
//! means a waiter expires the same tick it is enqueued.

use std::collections::VecDeque;
use std::fmt;

// =========================================================================
// WaitError
// =========================================================================

/// Error raised by [`WaitQueue`] enqueue.
///
/// The engine maps `QueueFull` and `QueueBytes` to a bounded-queue overload
/// error (HTTP 429) and `Duplicate` to a client-side correlation fault.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WaitError {
    /// The queue is at its `max_count` limit.
    QueueFull {
        max_count: usize,
        queued_count: usize,
    },
    /// Enqueueing `bytes` would exceed the queue's `max_bytes` limit.
    QueueBytes {
        max_bytes: u64,
        queued_bytes: u64,
        bytes: u64,
    },
    /// A waiter with the given id is already queued.
    Duplicate(u64),
}

impl fmt::Display for WaitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueueFull {
                max_count,
                queued_count,
            } => write!(
                f,
                "wait queue full: {queued_count}/{max_count} waiters already queued"
            ),
            Self::QueueBytes {
                max_bytes,
                queued_bytes,
                bytes,
            } => write!(
                f,
                "wait queue byte cap exceeded: {bytes} bytes requested, \
                 {queued_bytes}/{max_bytes} already queued"
            ),
            Self::Duplicate(id) => write!(f, "waiter {id} already queued"),
        }
    }
}

impl std::error::Error for WaitError {}

// =========================================================================
// Waiter
// =========================================================================

/// One request waiting for admission.
///
/// `enqueue_tick` is the scheduler tick at which the waiter entered the
/// queue; it is the request's admission age for the spec's oldest-first rule.
/// The deadline is `enqueue_tick + timeout_ticks` (saturating).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Waiter {
    /// Correlation id (matches the engine's `(id, attempt_id)` key).
    pub id: u64,
    /// Canonical pending-input bytes charged to this waiter (spec §5.3).
    pub bytes: u64,
    /// Scheduler tick at enqueue; preserved across pop so the engine can
    /// stamp the admitted request's `admission_tick`.
    pub enqueue_tick: u64,
}

impl Waiter {
    /// Tick at which this waiter expires: `enqueue_tick + timeout_ticks`,
    /// saturating at `u64::MAX`.
    pub fn deadline(&self, timeout_ticks: u64) -> u64 {
        self.enqueue_tick.saturating_add(timeout_ticks)
    }

    /// True once `now_tick` has reached the waiter's deadline.
    fn is_expired(&self, now_tick: u64, timeout_ticks: u64) -> bool {
        now_tick >= self.deadline(timeout_ticks)
    }
}

// =========================================================================
// WaitQueue
// =========================================================================

/// Bounded FIFO waiting room for serve admission (spec §5.3 S3).
///
/// Pure host policy: no GPU code, no arch dependency. The engine enqueues
/// requests that cannot be admitted yet and pops them FIFO when a slot frees,
/// before cold-admitting a new arrival. Count and byte caps are enforced at
/// enqueue; a per-waiter tick deadline drives timeout expiry.
#[derive(Debug)]
pub struct WaitQueue {
    max_count: usize,
    max_bytes: u64,
    timeout_ticks: u64,
    waiters: VecDeque<Waiter>,
    /// Running sum of `Waiter::bytes` currently in `waiters`.
    queued_bytes: u64,
}

impl WaitQueue {
    /// Construct a bounded wait queue.
    ///
    /// `max_count` and `max_bytes` MUST be nonzero — the new multi-slot mode
    /// rejects the zero/uncapped combination rather than reinterpreting it
    /// (spec §5.3). `timeout_ticks == 0` is allowed and means a waiter expires
    /// the same tick it is enqueued.
    pub fn new(max_count: usize, max_bytes: u64, timeout_ticks: u64) -> Result<Self, WaitError> {
        if max_count == 0 {
            return Err(WaitError::QueueFull {
                max_count: 0,
                queued_count: 0,
            });
        }
        if max_bytes == 0 {
            return Err(WaitError::QueueBytes {
                max_bytes: 0,
                queued_bytes: 0,
                bytes: 0,
            });
        }
        Ok(Self {
            max_count,
            max_bytes,
            timeout_ticks,
            waiters: VecDeque::new(),
            queued_bytes: 0,
        })
    }

    /// Enqueue a waiter if count and byte caps allow.
    ///
    /// `bytes` is the canonical pending-input bytes for this request (spec
    /// §5.3). `now_tick` stamps the waiter's admission age. Returns
    /// [`WaitError::Duplicate`] if `id` is already queued,
    /// [`WaitError::QueueFull`] at the count cap, or [`WaitError::QueueBytes`]
    /// if `bytes` would exceed the byte cap. On error the queue is unchanged.
    pub fn try_enqueue(&mut self, id: u64, bytes: u64, now_tick: u64) -> Result<(), WaitError> {
        if self.waiters.iter().any(|w| w.id == id) {
            return Err(WaitError::Duplicate(id));
        }
        if self.waiters.len() >= self.max_count {
            return Err(WaitError::QueueFull {
                max_count: self.max_count,
                queued_count: self.waiters.len(),
            });
        }
        let new_bytes = self
            .queued_bytes
            .checked_add(bytes)
            .ok_or(WaitError::QueueBytes {
                max_bytes: self.max_bytes,
                queued_bytes: self.queued_bytes,
                bytes,
            })?;
        if new_bytes > self.max_bytes {
            return Err(WaitError::QueueBytes {
                max_bytes: self.max_bytes,
                queued_bytes: self.queued_bytes,
                bytes,
            });
        }
        self.waiters.push_back(Waiter {
            id,
            bytes,
            enqueue_tick: now_tick,
        });
        self.queued_bytes = new_bytes;
        Ok(())
    }

    /// Pop the oldest non-expired waiter, FIFO.
    ///
    /// Expired waiters at the head are skipped (and dropped from the queue,
    /// their bytes released) until a non-expired one is found or the queue is
    /// drained. The engine should call [`expire`](Self::expire) each tick to
    /// surface timeouts to clients; `pop_ready` silently reclaims any expired
    /// head it encounters so a ready waiter is never blocked behind a dead
    /// one.
    pub fn pop_ready(&mut self, now_tick: u64) -> Option<Waiter> {
        while let Some(front) = self.waiters.front() {
            if front.is_expired(now_tick, self.timeout_ticks) {
                let expired = self.waiters.pop_front().expect("front checked nonempty");
                self.queued_bytes -= expired.bytes;
                continue;
            }
            let ready = self.waiters.pop_front().expect("front checked nonempty");
            self.queued_bytes -= ready.bytes;
            return Some(ready);
        }
        None
    }

    /// Remove and return all waiters whose deadline has passed at `now_tick`.
    ///
    /// Returned in FIFO order. The engine surfaces a timeout error to each
    /// expired waiter's client. Idempotent: returns empty when nothing is
    /// expired.
    pub fn expire(&mut self, now_tick: u64) -> Vec<Waiter> {
        let mut expired = Vec::new();
        while let Some(front) = self.waiters.front() {
            if front.is_expired(now_tick, self.timeout_ticks) {
                let w = self.waiters.pop_front().expect("front checked nonempty");
                self.queued_bytes -= w.bytes;
                expired.push(w);
            } else {
                break;
            }
        }
        expired
    }

    /// Cancel a queued waiter by id. Idempotent: returns `true` if a waiter
    /// was removed, `false` if no waiter with `id` was queued.
    pub fn remove(&mut self, id: u64) -> bool {
        if let Some(pos) = self.waiters.iter().position(|w| w.id == id) {
            let removed = self.waiters.remove(pos).expect("position checked");
            self.queued_bytes -= removed.bytes;
            true
        } else {
            false
        }
    }

    /// Number of waiters currently queued.
    pub fn queued_count(&self) -> usize {
        self.waiters.len()
    }

    /// Total canonical bytes currently queued.
    pub fn queued_bytes(&self) -> u64 {
        self.queued_bytes
    }

    /// Configured count cap.
    pub fn max_count(&self) -> usize {
        self.max_count
    }

    /// Configured byte cap.
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Configured per-waiter timeout in scheduler ticks.
    pub fn timeout_ticks(&self) -> u64 {
        self.timeout_ticks
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn queue(max_count: usize, max_bytes: u64, timeout_ticks: u64) -> WaitQueue {
        WaitQueue::new(max_count, max_bytes, timeout_ticks).expect("valid config")
    }

    /// Enqueue then pop preserves FIFO order across multiple waiters.
    #[test]
    fn enqueue_pop_fifo() {
        let mut q = queue(8, 1 << 20, 100);
        q.try_enqueue(1, 10, 0).unwrap();
        q.try_enqueue(2, 20, 1).unwrap();
        q.try_enqueue(3, 30, 2).unwrap();

        assert_eq!(q.queued_count(), 3);
        assert_eq!(q.queued_bytes(), 60);

        // FIFO: pop returns oldest non-expired first.
        assert_eq!(q.pop_ready(5).map(|w| w.id), Some(1));
        assert_eq!(q.pop_ready(5).map(|w| w.id), Some(2));
        assert_eq!(q.pop_ready(5).map(|w| w.id), Some(3));
        assert!(q.pop_ready(5).is_none());
        assert_eq!(q.queued_count(), 0);
        assert_eq!(q.queued_bytes(), 0);
    }

    /// Enqueue preserves the admission-age tick on the popped waiter.
    #[test]
    fn pop_ready_carries_enqueue_tick() {
        let mut q = queue(4, 1 << 20, 100);
        q.try_enqueue(7, 5, 42).unwrap();
        let w = q.pop_ready(50).expect("ready");
        assert_eq!(w.id, 7);
        assert_eq!(w.bytes, 5);
        assert_eq!(w.enqueue_tick, 42);
    }

    /// Byte cap rejects an enqueue that would exceed it; queue unchanged.
    #[test]
    fn byte_cap_rejects() {
        let mut q = queue(8, 100, 100);
        q.try_enqueue(1, 60, 0).unwrap();
        // 60 + 50 = 110 > 100 -> reject.
        let err = q.try_enqueue(2, 50, 0).unwrap_err();
        assert_eq!(
            err,
            WaitError::QueueBytes {
                max_bytes: 100,
                queued_bytes: 60,
                bytes: 50,
            }
        );
        // Queue unchanged.
        assert_eq!(q.queued_count(), 1);
        assert_eq!(q.queued_bytes(), 60);
        // An enqueue that exactly fits the cap is accepted.
        q.try_enqueue(3, 40, 0).unwrap();
        assert_eq!(q.queued_bytes(), 100);
    }

    /// Count cap rejects once `max_count` waiters are queued; queue unchanged.
    #[test]
    fn count_cap_rejects() {
        let mut q = queue(2, 1 << 20, 100);
        q.try_enqueue(1, 1, 0).unwrap();
        q.try_enqueue(2, 1, 0).unwrap();
        let err = q.try_enqueue(3, 1, 0).unwrap_err();
        assert_eq!(
            err,
            WaitError::QueueFull {
                max_count: 2,
                queued_count: 2,
            }
        );
        assert_eq!(q.queued_count(), 2);
    }

    /// A waiter whose deadline has passed is returned by `expire`, not by
    /// `pop_ready`. Expired waiters are dropped from the queue and their
    /// bytes released.
    #[test]
    fn timeout_expire_drops_expired() {
        let mut q = queue(8, 1 << 20, 10);
        q.try_enqueue(1, 5, 0).unwrap(); // deadline 10
        q.try_enqueue(2, 7, 2).unwrap(); // deadline 12
        q.try_enqueue(3, 9, 4).unwrap(); // deadline 14

        // At tick 11: waiter 1 is expired (deadline 10), 2 and 3 are not.
        let expired = q.expire(11);
        assert_eq!(expired.iter().map(|w| w.id).collect::<Vec<_>>(), vec![1]);
        assert_eq!(q.queued_count(), 2);
        assert_eq!(q.queued_bytes(), 16);

        // pop_ready at tick 11 returns the next non-expired (waiter 2).
        assert_eq!(q.pop_ready(11).map(|w| w.id), Some(2));
        assert_eq!(q.queued_bytes(), 9);

        // At tick 14 waiter 3 is expired.
        let expired = q.expire(14);
        assert_eq!(expired.iter().map(|w| w.id).collect::<Vec<_>>(), vec![3]);
        assert_eq!(q.queued_count(), 0);
        assert_eq!(q.queued_bytes(), 0);
    }

    /// `pop_ready` skips an expired head so a ready waiter behind it is not
    /// blocked; the expired head's bytes are released.
    #[test]
    fn pop_ready_skips_expired_head() {
        let mut q = queue(8, 1 << 20, 5);
        q.try_enqueue(1, 10, 0).unwrap(); // deadline 5
        q.try_enqueue(2, 20, 0).unwrap(); // deadline 5, still queued behind 1
                                          // At tick 6 both are expired; pop_ready returns None and clears them.
        assert!(q.pop_ready(6).is_none());
        assert_eq!(q.queued_count(), 0);
        assert_eq!(q.queued_bytes(), 0);
    }

    /// `pop_ready` skips an expired head and returns the live waiter behind.
    #[test]
    fn pop_ready_returns_live_behind_expired() {
        let mut q = queue(8, 1 << 20, 10);
        q.try_enqueue(1, 10, 0).unwrap(); // deadline 10
        q.try_enqueue(2, 20, 5).unwrap(); // deadline 15
                                          // At tick 11: waiter 1 expired, waiter 2 still live.
        assert_eq!(q.pop_ready(11).map(|w| w.id), Some(2));
        assert_eq!(q.queued_count(), 0);
        assert_eq!(q.queued_bytes(), 0);
    }

    /// Duplicate id is rejected; the original waiter is preserved.
    #[test]
    fn duplicate_rejected() {
        let mut q = queue(8, 1 << 20, 100);
        q.try_enqueue(1, 10, 0).unwrap();
        let err = q.try_enqueue(1, 20, 1).unwrap_err();
        assert_eq!(err, WaitError::Duplicate(1));
        assert_eq!(q.queued_count(), 1);
        // The original waiter (10 bytes) is intact.
        let w = q.pop_ready(1).expect("ready");
        assert_eq!(w.bytes, 10);
    }

    /// `remove` cancels a queued waiter and releases its bytes; idempotent.
    #[test]
    fn remove_cancels_idempotent() {
        let mut q = queue(8, 1 << 20, 100);
        q.try_enqueue(1, 10, 0).unwrap();
        q.try_enqueue(2, 20, 0).unwrap();
        assert_eq!(q.queued_bytes(), 30);

        assert!(q.remove(1));
        assert_eq!(q.queued_count(), 1);
        assert_eq!(q.queued_bytes(), 20);

        // Removing the same id again is a no-op.
        assert!(!q.remove(1));
        assert_eq!(q.queued_count(), 1);

        // Removing an unknown id is a no-op.
        assert!(!q.remove(99));

        // Waiter 2 is still queued and pops FIFO.
        assert_eq!(q.pop_ready(1).map(|w| w.id), Some(2));
    }

    /// `remove` of a middle waiter preserves FIFO order of the remainder.
    #[test]
    fn remove_middle_preserves_fifo() {
        let mut q = queue(8, 1 << 20, 100);
        q.try_enqueue(1, 1, 0).unwrap();
        q.try_enqueue(2, 1, 0).unwrap();
        q.try_enqueue(3, 1, 0).unwrap();
        assert!(q.remove(2));
        assert_eq!(q.pop_ready(1).map(|w| w.id), Some(1));
        assert_eq!(q.pop_ready(1).map(|w| w.id), Some(3));
        assert!(q.pop_ready(1).is_none());
    }

    /// Zero `max_count` is rejected at construction (no uncapped reinterp).
    #[test]
    fn zero_max_count_rejected() {
        let err = WaitQueue::new(0, 1 << 20, 100).unwrap_err();
        assert_eq!(
            err,
            WaitError::QueueFull {
                max_count: 0,
                queued_count: 0,
            }
        );
    }

    /// Zero `max_bytes` is rejected at construction (no uncapped reinterp).
    #[test]
    fn zero_max_bytes_rejected() {
        let err = WaitQueue::new(8, 0, 100).unwrap_err();
        assert_eq!(
            err,
            WaitError::QueueBytes {
                max_bytes: 0,
                queued_bytes: 0,
                bytes: 0,
            }
        );
    }

    /// `timeout_ticks == 0` is permitted: a waiter expires the same tick.
    #[test]
    fn zero_timeout_expires_same_tick() {
        let mut q = queue(8, 1 << 20, 0);
        q.try_enqueue(1, 10, 5).unwrap();
        // At the enqueue tick the deadline (5) has been reached -> expired.
        let expired = q.expire(5);
        assert_eq!(expired.iter().map(|w| w.id).collect::<Vec<_>>(), vec![1]);
        assert_eq!(q.queued_count(), 0);
        // Before the enqueue tick (impossible in practice, but checks the
        // boundary) the waiter is still live.
        q.try_enqueue(2, 10, 5).unwrap();
        assert!(q.expire(4).is_empty());
        assert_eq!(q.queued_count(), 1);
    }

    /// Byte overflow on enqueue is reported as a byte-cap error, not panic.
    #[test]
    fn byte_overflow_is_byte_error() {
        let mut q = queue(8, u64::MAX, 100);
        q.try_enqueue(1, u64::MAX - 1, 0).unwrap();
        // Adding 3 would overflow u64.
        let err = q.try_enqueue(2, 3, 0).unwrap_err();
        assert!(matches!(err, WaitError::QueueBytes { .. }));
        assert_eq!(q.queued_count(), 1);
    }

    /// Accessors report configured caps and live counters.
    #[test]
    fn accessors() {
        let mut q = queue(4, 1000, 50);
        assert_eq!(q.max_count(), 4);
        assert_eq!(q.max_bytes(), 1000);
        assert_eq!(q.timeout_ticks(), 50);
        assert_eq!(q.queued_count(), 0);
        assert_eq!(q.queued_bytes(), 0);
        q.try_enqueue(1, 100, 0).unwrap();
        assert_eq!(q.queued_count(), 1);
        assert_eq!(q.queued_bytes(), 100);
    }

    /// `expire` is idempotent when nothing is expired.
    #[test]
    fn expire_idempotent_empty() {
        let mut q = queue(8, 1 << 20, 100);
        q.try_enqueue(1, 10, 0).unwrap();
        assert!(q.expire(5).is_empty());
        assert_eq!(q.queued_count(), 1);
        assert!(q.expire(5).is_empty());
    }
}
