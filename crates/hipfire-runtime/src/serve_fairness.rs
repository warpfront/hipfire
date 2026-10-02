// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Host-side fairness policy for concurrent serve (spec §5.3 S3).
//!
//! Pure policy type — no GPU code, no dependency on `hipfire-arch-qwen35`.
//! The scheduler / serve-engine calls [`FairQueue::select`] each step to
//! obtain a row-budget-aware grant set, replacing the forward panic-assert
//! as the trunk-row throttle (spec §5.2 S2 / §5.3 S3).
//!
//! # Fairness model
//!
//! Each admitted request is placed in one of two **bands**: *aged* (skipped
//! for a complete admission round) or *normal*. Within a band, requests are
//! ordered by oldest `admission_tick` first, then by lowest
//! `uncached_prefill_tokens` (best cache locality) as a bounded tie-break
//! (spec §5.3 S3.5).
//!
//! [`select`](FairQueue::select) allocates rows in four phases:
//!
//! 1. **Decode** — one row per decoding request, up to `max_decode_lanes`.
//!    A request in MTP verify mode does **not** also receive a decode row
//!    (spec §5.3 S3).
//! 2. **Prefill** — a nonzero `prefill_quantum` via persistent cursor
//!    rotation so every runnable prefill is served within *N* ticks (spec
//!    §5.3 S3.2).
//! 3. **Verify** — MTP verify rows from the remaining quota.
//! 4. **Forced** — forced-token rows from the remaining quota.
//!
//! If the oldest request is individually feasible but could not acquire full
//! credits, [`Selection::starved_oldest`] is set so the caller can stop
//! admitting younger conflicting requests (bounded backfill, spec §5.3 S3.4).

use std::cmp::min;
use std::cmp::Ordering;
use std::fmt;

// =========================================================================
// Error
// =========================================================================

/// Error raised by [`FairQueue`] construction or mutation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FairnessError {
    /// `max_batch_tokens < decode_lanes + prefill_min_tokens` — the budget
    /// cannot accommodate even one decode lane plus the minimum prefill
    /// quantum, so the policy would deadlock.
    InvalidConfig {
        max_batch_tokens: u64,
        decode_lanes: u64,
        prefill_min_tokens: u64,
    },
    /// `prefill_min_tokens == 0` — a zero minimum quantum admits zero-row
    /// prefill grants and disables the sliver guard, livelocking the queue.
    /// The minimum must be at least 1.
    ZeroMinTokens,
    /// A request with the given id is already admitted.
    DuplicateId(u64),
    /// No request with the given id is currently admitted.
    NotFound(u64),
    /// A checked-arithmetic row sum overflowed `u64`.
    RowOverflow,
}

impl fmt::Display for FairnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig {
                max_batch_tokens,
                decode_lanes,
                prefill_min_tokens,
            } => write!(
                f,
                "fairness config invalid: max_batch_tokens ({max_batch_tokens}) \
                 < decode_lanes ({decode_lanes}) + prefill_min_tokens ({prefill_min_tokens})"
            ),
            Self::ZeroMinTokens => write!(f, "prefill_min_tokens must be at least 1"),
            Self::DuplicateId(id) => write!(f, "request {id} already admitted"),
            Self::NotFound(id) => write!(f, "request {id} not found"),
            Self::RowOverflow => write!(f, "fairness row count overflow"),
        }
    }
}

impl std::error::Error for FairnessError {}

// =========================================================================
// Grant
// =========================================================================

/// Per-request row grant for one step.
///
/// Mirrors the row categories of
/// [`StepReservation`](crate::serve_contract::StepReservation) but as
/// per-request grants with row counts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Grant {
    /// One ordinary decode row.
    Decode { id: u64 },
    /// `rows` prefill rows for this request this step.
    Prefill { id: u64, rows: u64 },
    /// MTP verify rows (`k + 1`).
    Verify { id: u64, rows: u64 },
    /// Forced-token (jump-forward) rows.
    Forced { id: u64, rows: u64 },
}

impl Grant {
    /// Row count consumed by this grant.
    pub fn rows(&self) -> u64 {
        match self {
            Self::Decode { .. } => 1,
            Self::Prefill { rows, .. } | Self::Verify { rows, .. } | Self::Forced { rows, .. } => {
                *rows
            }
        }
    }

    /// Request id this grant is for.
    pub fn id(&self) -> u64 {
        match self {
            Self::Decode { id }
            | Self::Prefill { id, .. }
            | Self::Verify { id, .. }
            | Self::Forced { id, .. } => *id,
        }
    }
}

// =========================================================================
// Selection
// =========================================================================

/// Result of [`FairQueue::select`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Selection {
    /// Grants issued this step, in allocation order (decode, prefill, verify,
    /// forced).
    pub grants: Vec<Grant>,
    /// The oldest request is individually feasible (fits `max_batch_tokens`
    /// alone) but could not acquire full credits this step. The caller should
    /// stop admitting younger conflicting requests (bounded backfill, spec
    /// §5.3 S3.4).
    pub starved_oldest: bool,
    /// The id of the fairness-best request that received NOTHING this step
    /// (the request `starved_oldest` describes). The consumer's backfill
    /// mask must target THIS request — masking to "the oldest active
    /// session" instead starves the wrong request whenever the starved one
    /// is not the absolute oldest in flight, and freezes every healthy
    /// slot to zero rows in the meantime.
    pub starved_id: Option<u64>,
}

// =========================================================================
// FairRequest
// =========================================================================

/// Per-request record tracked by [`FairQueue`] (spec §5.3 S3.1).
#[derive(Clone, Debug)]
pub struct FairRequest {
    /// Caller-assigned request id.
    pub id: u64,
    /// Fairness domain (e.g. session id).
    pub domain_id: String,
    /// Tick at which this request was admitted. Preserved across prefill
    /// chunks — never reset (spec §5.3 S3.1).
    pub admission_tick: u64,
    /// Uncached prefill tokens remaining. Used as the cache-locality tie-break
    /// (lower = more cached = preferred) within the same fairness band (spec
    /// §5.3 S3.5). Driven to zero as prefill completes.
    pub uncached_prefill_tokens: u64,
    /// Set by [`FairQueue::skip_round_complete`] when this request was skipped
    /// for one complete admission round. Aged requests are selected before
    /// cache-locality tie-breaks (spec §5.3 S3.3).
    pub aged: bool,
    /// Last tick on which this request received any grant (`None` = never
    /// served).
    pub last_served_tick: Option<u64>,
    /// Whether this request wants an ordinary decode row this step.
    pub wants_decode: bool,
    /// MTP verify rows needed this step (0 = none).
    pub verify_rows: u64,
    /// Forced-token rows needed this step (0 = none).
    pub forced_rows: u64,
}

impl FairRequest {
    /// Total rows this request needs this step, using checked arithmetic.
    /// A verify request does not also count a decode row (spec §5.3 S3).
    fn needed_rows(&self) -> Option<u64> {
        let decode = if self.wants_decode && self.verify_rows == 0 {
            1u64
        } else {
            0u64
        };
        let a = decode.checked_add(self.uncached_prefill_tokens)?;
        let b = a.checked_add(self.verify_rows)?;
        b.checked_add(self.forced_rows)
    }
}

// =========================================================================
// FairQueue
// =========================================================================

/// Host-side fairness queue — pure policy, no GPU code (spec §5.3 S3).
///
/// The caller (scheduler / serve-engine) admits requests, updates their
/// per-step needs, and calls [`select`](Self::select) each step to obtain a
/// row-budget-aware grant set. The global trunk-row budget
/// (`max_batch_tokens`) is enforced proactively here so the forward
/// panic-assert in the engine is no longer the throttle.
#[derive(Debug)]
pub struct FairQueue {
    max_batch_tokens: u64,
    decode_lanes: u64,
    prefill_min_tokens: u64,
    requests: Vec<FairRequest>,
    /// Persistent cursor for round-robin prefill rotation (spec §5.3 S3.2).
    prefill_cursor: usize,
    /// Monotonic tick, advanced once per [`select`](Self::select) call.
    tick: u64,
    /// Tick at the last [`skip_round_complete`](Self::skip_round_complete).
    last_round_tick: u64,
}

impl FairQueue {
    /// Construct a fairness queue.
    ///
    /// Returns [`FairnessError::InvalidConfig`] if
    /// `max_batch_tokens < decode_lanes + prefill_min_tokens` — the budget
    /// cannot accommodate even one decode lane plus the minimum prefill
    /// quantum, so the policy would deadlock.
    pub fn new(
        max_batch_tokens: u64,
        decode_lanes: u64,
        prefill_min_tokens: u64,
    ) -> Result<Self, FairnessError> {
        if prefill_min_tokens == 0 {
            return Err(FairnessError::ZeroMinTokens);
        }
        let min_budget = decode_lanes
            .checked_add(prefill_min_tokens)
            .ok_or(FairnessError::RowOverflow)?;
        if max_batch_tokens < min_budget {
            return Err(FairnessError::InvalidConfig {
                max_batch_tokens,
                decode_lanes,
                prefill_min_tokens,
            });
        }
        Ok(Self {
            max_batch_tokens,
            decode_lanes,
            prefill_min_tokens,
            requests: Vec::new(),
            prefill_cursor: 0,
            tick: 0,
            last_round_tick: 0,
        })
    }

    /// Admit a new request.
    ///
    /// `uncached_prefill_tokens` is the initial cache-locality weight (lower =
    /// more cached). The request starts with `wants_decode = false`,
    /// `verify_rows = 0`, `forced_rows = 0` — the caller updates these via
    /// [`set_needs`](Self::set_needs) before each [`select`](Self::select).
    pub fn admit(
        &mut self,
        id: u64,
        domain_id: impl Into<String>,
        uncached_prefill_tokens: u64,
    ) -> Result<(), FairnessError> {
        if self.requests.iter().any(|r| r.id == id) {
            return Err(FairnessError::DuplicateId(id));
        }
        self.requests.push(FairRequest {
            id,
            domain_id: domain_id.into(),
            admission_tick: self.tick,
            uncached_prefill_tokens,
            aged: false,
            last_served_tick: None,
            wants_decode: false,
            verify_rows: 0,
            forced_rows: 0,
        });
        Ok(())
    }

    /// Update a request's per-step needs. Called before [`select`](Self::select).
    pub fn set_needs(
        &mut self,
        id: u64,
        wants_decode: bool,
        verify_rows: u64,
        forced_rows: u64,
    ) -> Result<(), FairnessError> {
        let req = self
            .requests
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or(FairnessError::NotFound(id))?;
        req.wants_decode = wants_decode;
        req.verify_rows = verify_rows;
        req.forced_rows = forced_rows;
        Ok(())
    }

    /// Update a request's remaining uncached prefill tokens (cache-locality
    /// weight). Called as prefill progresses.
    pub fn set_uncached_prefill(&mut self, id: u64, tokens: u64) -> Result<(), FairnessError> {
        let req = self
            .requests
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or(FairnessError::NotFound(id))?;
        req.uncached_prefill_tokens = tokens;
        Ok(())
    }

    /// Remove a completed request.
    pub fn remove(&mut self, id: u64) -> Result<(), FairnessError> {
        let len_before = self.requests.len();
        self.requests.retain(|r| r.id != id);
        if self.requests.len() == len_before {
            return Err(FairnessError::NotFound(id));
        }
        if !self.requests.is_empty() {
            self.prefill_cursor %= self.requests.len();
        } else {
            self.prefill_cursor = 0;
        }
        Ok(())
    }

    /// Current monotonic tick.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Number of admitted requests.
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Read-only access to a request by id.
    pub fn get(&self, id: u64) -> Option<&FairRequest> {
        self.requests.iter().find(|r| r.id == id)
    }

    /// Mark requests that were not served since the last round as aged, and
    /// clear the flag from requests that WERE served (spec §5.3 S3.3).
    /// Age is a property of "skipped for one complete round", not a permanent
    /// promotion: a request returns to the normal band once serviced and can
    /// be re-aged by a later skipped round. The caller invokes this after one
    /// complete admission round.
    pub fn skip_round_complete(&mut self) {
        for req in &mut self.requests {
            let not_served_since_round = match req.last_served_tick {
                None => true,
                Some(t) => t < self.last_round_tick,
            };
            let admitted_at_or_before_round = req.admission_tick <= self.last_round_tick;
            req.aged = not_served_since_round && admitted_at_or_before_round;
        }
        self.last_round_tick = self.tick;
    }

    // ---------------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------------

    /// Fairness comparator: aged first, then oldest `admission_tick`, then
    /// lowest `uncached_prefill_tokens` (best cache locality). Used only to
    /// compare two requests during bounded O(n) scans — the entire queue is
    /// never sorted (spec §5.3 S3.5).
    fn fairness_cmp(a: &FairRequest, b: &FairRequest) -> Ordering {
        // Aged band first.
        match (a.aged, b.aged) {
            (true, false) => return Ordering::Less,
            (false, true) => return Ordering::Greater,
            _ => {}
        }
        // Oldest admission_tick first.
        match a.admission_tick.cmp(&b.admission_tick) {
            Ordering::Equal => {}
            ord => return ord,
        }
        // Cache locality: lower uncached = more cached = preferred.
        a.uncached_prefill_tokens.cmp(&b.uncached_prefill_tokens)
    }

    /// Find the index of the best request matching `predicate` by fairness
    /// order. O(n) bounded scan.
    fn best_index<F: Fn(&FairRequest) -> bool>(&self, predicate: F) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (i, r) in self.requests.iter().enumerate() {
            if !predicate(r) {
                continue;
            }
            match best {
                None => best = Some(i),
                Some(bi) => {
                    if Self::fairness_cmp(r, &self.requests[bi]) == Ordering::Less {
                        best = Some(i);
                    }
                }
            }
        }
        best
    }

    // ---------------------------------------------------------------------
    // Select
    // ---------------------------------------------------------------------

    /// Select grants for this step (spec §5.3 S3).
    ///
    /// Enforces:
    /// - Decode lanes get base 1-row allocation first (up to
    ///   `max_decode_lanes`), but a Verify request does NOT also get a Decode
    ///   row.
    /// - If prefill is waiting, allocate a nonzero `prefill_quantum` via
    ///   persistent cursor rotation.
    /// - Remaining rows go to more prefill (cursor rotation), then MTP verify
    ///   and forced rows from the remaining quota.
    /// - Total rows ≤ `remaining_rows`.
    /// - [`Selection::starved_oldest`] is set when the oldest request is
    ///   individually feasible but could not acquire full credits (bounded
    ///   backfill, spec §5.3 S3.4).
    pub fn select(
        &mut self,
        max_decode_lanes: u64,
        prefill_quantum: u64,
        remaining_rows: u64,
    ) -> Selection {
        if self.requests.is_empty() {
            self.tick += 1;
            return Selection::default();
        }

        // Snapshot each request's needed_rows before mutation (for
        // starved_oldest check). Uses checked arithmetic (spec §5.1).
        let needs_snapshot: Vec<Option<u64>> =
            self.requests.iter().map(|r| r.needed_rows()).collect();

        let mut grants: Vec<Grant> = Vec::new();
        let mut used: u64 = 0;
        let mut served_ids: Vec<u64> = Vec::new();

        // --- Phase 1: Decode — base 1-row allocation ---
        // A request gets a decode row only if it wants decode AND is not
        // verifying (verify-not-plus-decode, spec §5.3 S3).
        let mut decode_count: u64 = 0;
        while decode_count < max_decode_lanes {
            let avail = match remaining_rows.checked_sub(used) {
                Some(a) if a > 0 => a,
                _ => break,
            };
            let best = self.best_index(|r| {
                r.wants_decode && r.verify_rows == 0 && !served_ids.contains(&r.id)
            });
            match best {
                Some(idx) => {
                    grants.push(Grant::Decode {
                        id: self.requests[idx].id,
                    });
                    used += 1; // safe: avail > 0
                    served_ids.push(self.requests[idx].id);
                    self.requests[idx].last_served_tick = Some(self.tick);
                    decode_count += 1;
                }
                None => break,
            }
        }

        // --- Phase 2: Prefill — cursor-bounded scan with fairness tie-break ---
        // Rotate through prefilling requests. Each gets up to prefill_quantum
        // rows (bounded by remaining budget and uncached tokens). A request
        // is only started if at least prefill_min_tokens rows are available.
        //
        // The persistent cursor provides fair rotation (each runnable prefill
        // served within N ticks, spec §5.3 S3.2). Fairness order (aged, oldest,
        // cache locality) is the primary sort; cursor proximity is the
        // tie-break so equal-fairness requests rotate. This is a bounded O(n)
        // scan per pick, not a full-queue sort (spec §5.3 S3.5).
        let n = self.requests.len();
        if prefill_quantum > 0 {
            let mut picked = 0;
            loop {
                if picked >= n {
                    break; // every prefilling request had a chance
                }
                let avail = match remaining_rows.checked_sub(used) {
                    Some(a) if a > 0 => a,
                    _ => break,
                };
                // Scan from cursor, find best prefilling candidate by
                // fairness order. Cursor proximity is the implicit tie-break
                // (earlier in scan = closer to cursor = preferred on ties).
                let start = self.prefill_cursor % n;
                let mut best: Option<usize> = None;
                for i in 0..n {
                    let idx = (start + i) % n;
                    let r = &self.requests[idx];
                    if r.uncached_prefill_tokens == 0 || served_ids.contains(&r.id) {
                        continue;
                    }
                    if avail < self.prefill_min_tokens && r.uncached_prefill_tokens > avail {
                        continue;
                    }
                    match best {
                        None => best = Some(idx),
                        Some(bi) => {
                            if Self::fairness_cmp(r, &self.requests[bi]) == Ordering::Less {
                                best = Some(idx);
                            }
                            // Equal fairness: keep earlier (closer to cursor).
                        }
                    }
                }
                match best {
                    Some(idx) => {
                        let uncached = self.requests[idx].uncached_prefill_tokens;
                        let rows = min(min(prefill_quantum, uncached), avail);
                        if rows < self.prefill_min_tokens && rows < uncached {
                            // Below the minimum quantum AND this grant would
                            // not finish the request's prompt: skip so the
                            // budget isn't frittered into slivers. A grant
                            // that COMPLETES the prompt (rows == uncached)
                            // is always allowed — refusing it would livelock
                            // any request whose final chunk is smaller than
                            // the minimum quantum (spec §5.3: prefill must
                            // remain runnable in mixed mode).
                            self.prefill_cursor = (idx + 1) % n;
                            picked += 1;
                            continue;
                        }
                        grants.push(Grant::Prefill {
                            id: self.requests[idx].id,
                            rows,
                        });
                        used = match used.checked_add(rows) {
                            Some(v) => v,
                            None => break,
                        };
                        served_ids.push(self.requests[idx].id);
                        self.requests[idx].uncached_prefill_tokens =
                            match uncached.checked_sub(rows) {
                                Some(v) => v,
                                None => break,
                            };
                        self.requests[idx].last_served_tick = Some(self.tick);
                        self.prefill_cursor = (idx + 1) % n;
                        picked += 1;
                    }
                    None => break, // no more prefilling requests
                }
            }
        }

        // --- Phase 3: Verify — from remaining quota ---
        // Collect verify candidates and iterate in fairness order. Bounded:
        // at most n_slots entries, not a full-queue sort.
        let mut verify_candidates: Vec<usize> = self
            .requests
            .iter()
            .enumerate()
            .filter(|(_, r)| r.verify_rows > 0 && !served_ids.contains(&r.id))
            .map(|(i, _)| i)
            .collect();
        verify_candidates
            .sort_by(|&a, &b| Self::fairness_cmp(&self.requests[a], &self.requests[b]));
        for &idx in &verify_candidates {
            let rows = self.requests[idx].verify_rows;
            let avail = match remaining_rows.checked_sub(used) {
                Some(a) => a,
                None => break,
            };
            if rows <= avail {
                grants.push(Grant::Verify {
                    id: self.requests[idx].id,
                    rows,
                });
                used = match used.checked_add(rows) {
                    Some(v) => v,
                    None => break,
                };
                served_ids.push(self.requests[idx].id);
                self.requests[idx].last_served_tick = Some(self.tick);
            }
            // If it doesn't fit, skip — younger/smaller ones might.
        }

        // --- Phase 4: Forced — from remaining quota ---
        let mut forced_candidates: Vec<usize> = self
            .requests
            .iter()
            .enumerate()
            .filter(|(_, r)| r.forced_rows > 0 && !served_ids.contains(&r.id))
            .map(|(i, _)| i)
            .collect();
        forced_candidates
            .sort_by(|&a, &b| Self::fairness_cmp(&self.requests[a], &self.requests[b]));
        for &idx in &forced_candidates {
            let rows = self.requests[idx].forced_rows;
            let avail = match remaining_rows.checked_sub(used) {
                Some(a) => a,
                None => break,
            };
            if rows <= avail {
                grants.push(Grant::Forced {
                    id: self.requests[idx].id,
                    rows,
                });
                used = match used.checked_add(rows) {
                    Some(v) => v,
                    None => break,
                };
                served_ids.push(self.requests[idx].id);
                self.requests[idx].last_served_tick = Some(self.tick);
            }
        }

        // --- Bounded backfill: starved_oldest (spec §5.3 S3.4) ---
        // Find the oldest request (by fairness order) that has unserved needs
        // and is individually feasible (needed_rows <= max_batch_tokens).
        //
        // "Unserved" means got NOTHING this step. A request that received a
        // grant is being PROGRESSIVELY served — a multi-tick prefill always
        // has remaining tokens after any finite grant, so counting remaining
        // tokens alone as unserviced latches starved_oldest for the whole
        // prefill and (via the engine's backfill mask) starves every other
        // slot to zero rows until the prefill completes — the exact
        // serialization this policy was meant to prevent.
        let mut starved_oldest = false;
        let mut best_starved: Option<usize> = None;
        for (i, r) in self.requests.iter().enumerate() {
            let prefill_remaining = r.uncached_prefill_tokens;
            let got_prefill = grants
                .iter()
                .any(|g| matches!(g, Grant::Prefill { id, .. } if *id == r.id));
            let got_decode = grants
                .iter()
                .any(|g| matches!(g, Grant::Decode { id } if *id == r.id));
            let got_verify = grants
                .iter()
                .any(|g| matches!(g, Grant::Verify { id, .. } if *id == r.id));
            let got_forced = grants
                .iter()
                .any(|g| matches!(g, Grant::Forced { id, .. } if *id == r.id));
            let unserved = (prefill_remaining > 0 && !got_prefill)
                || (r.wants_decode && r.verify_rows == 0 && !got_decode)
                || (r.verify_rows > 0 && !got_verify)
                || (r.forced_rows > 0 && !got_forced);
            if !unserved {
                continue;
            }
            let needed = needs_snapshot[i].unwrap_or(u64::MAX);
            if needed > self.max_batch_tokens {
                continue; // not individually feasible
            }
            match best_starved {
                None => best_starved = Some(i),
                Some(bi) => {
                    if Self::fairness_cmp(r, &self.requests[bi]) == Ordering::Less {
                        best_starved = Some(i);
                    }
                }
            }
        }
        starved_oldest = best_starved.is_some();
        let starved_id = best_starved.map(|i| self.requests[i].id);

        self.tick += 1;
        Selection {
            grants,
            starved_oldest,
            starved_id,
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn queue(max_batch_tokens: u64, decode_lanes: u64, prefill_min: u64) -> FairQueue {
        FairQueue::new(max_batch_tokens, decode_lanes, prefill_min).expect("valid config")
    }

    /// Regression: a decode lane and a multi-tick prefill running together
    /// must NOT latch starved_oldest (the prefill received a grant and is
    /// being progressively served). The old "remaining tokens > 0 = unserved"
    /// test latched starvation for the whole prefill and the engine's
    /// backfill mask starved the prefilling slot to zero rows until the
    /// decode finished — serializing two concurrent requests.
    #[test]
    fn partial_prefill_grant_does_not_latch_starved_oldest() {
        let mut q = queue(8192, 2, 1);
        // A: decoding (wants one decode row), admitted first.
        q.admit(0, "default", 0).unwrap();
        q.set_needs(0, true, 0, 0).unwrap();
        // B: 4320-token prefill, admitted one tick later.
        q.admit(1, "default", 4320).unwrap();
        q.set_needs(1, false, 0, 0).unwrap();

        let sel = q.select(2, 1, 8188);
        assert!(sel
            .grants
            .iter()
            .any(|g| matches!(g, Grant::Decode { id: 0 })));
        assert!(sel
            .grants
            .iter()
            .any(|g| matches!(g, Grant::Prefill { id: 1, .. })));
        assert!(
            !sel.starved_oldest,
            "both requests received service — starvation must not latch"
        );

        // The legitimate case still fires: the oldest request got NOTHING
        // this step while a younger one was served.
        let mut q2 = queue(8192, 2, 1);
        q2.admit(0, "default", 4320).unwrap(); // oldest: prefill
        q2.set_needs(0, false, 0, 0).unwrap();
        q2.admit(1, "default", 0).unwrap(); // younger: decode
        q2.set_needs(1, true, 0, 0).unwrap();
        // Exhaust the budget with the younger decode lanes so the oldest
        // prefill cannot receive even its minimum quantum.
        let sel2 = q2.select(2, 1, 1);
        assert!(
            sel2.starved_oldest,
            "oldest feasible request with zero service must set starved_oldest"
        );
    }

    /// Test 1: aged request beats younger cache hit.
    ///
    /// Two prefilling requests with the same admission_tick: B has better
    /// cache locality (lower uncached). Without aging, B wins the prefill
    /// quantum. After A is skipped for a round and marked aged, A wins
    /// instead (aged band > normal band, spec §5.3 S3.3).
    #[test]
    fn aged_request_beats_younger_cache_hit() {
        let mut q = queue(4096, 0, 1);
        q.admit(1, "s1", 100).unwrap(); // A: bad cache (high uncached)
        q.admit(2, "s2", 50).unwrap(); // B: better cache (lower uncached)
                                       // Both prefilling, no decode.
        q.set_needs(1, false, 0, 0).unwrap();
        q.set_needs(2, false, 0, 0).unwrap();

        // First select: B wins (same admission_tick, lower uncached = better
        // cache locality). B gets 10 prefill rows, uncached drops to 40.
        let sel = q.select(0, 10, 10);
        assert_eq!(sel.grants.len(), 1);
        assert!(matches!(sel.grants[0], Grant::Prefill { id: 2, rows: 10 }));
        // A is individually feasible (100 <= 4096) but was starved this tick.
        assert!(sel.starved_oldest);

        // A was not served → mark aged.
        q.skip_round_complete();
        assert!(q.get(1).unwrap().aged, "A should be aged");
        assert!(!q.get(2).unwrap().aged, "B should not be aged");

        // Second select: A is aged, B is not → A wins despite worse cache.
        let sel = q.select(0, 10, 10);
        assert_eq!(sel.grants.len(), 1);
        assert!(
            matches!(sel.grants[0], Grant::Prefill { id: 1, rows: 10 }),
            "aged A should beat non-aged B, got {:?}",
            sel.grants[0]
        );
    }

    /// Test 2: prefill cursor rotation.
    ///
    /// With N runnable prefills and one quantum per step, each is served
    /// within N ticks (spec §5.3 S3.2).
    #[test]
    fn prefill_cursor_rotation() {
        let mut q = queue(4096, 0, 1);
        // Three prefilling requests, each with 100 uncached tokens.
        q.admit(10, "d0", 100).unwrap();
        q.admit(11, "d1", 100).unwrap();
        q.admit(12, "d2", 100).unwrap();
        // No decode needed; prefill only.
        // prefill_quantum = 4096 (enough for one full request), but we want
        // to verify rotation: with quantum=1, each gets 1 row per step.
        // Actually, with quantum=1 and 3 requests, in 3 ticks each gets served.
        let quantum = 1u64;
        let mut served_ids = Vec::new();
        for _ in 0..3 {
            let sel = q.select(0, quantum, 4096);
            for g in &sel.grants {
                if let Grant::Prefill { id, .. } = g {
                    served_ids.push(*id);
                }
            }
        }
        // Each request should have been served at least once in 3 ticks.
        assert!(served_ids.contains(&10), "id 10 not served in 3 ticks");
        assert!(served_ids.contains(&11), "id 11 not served in 3 ticks");
        assert!(served_ids.contains(&12), "id 12 not served in 3 ticks");
    }

    /// Test 3: verify-not-plus-decode.
    ///
    /// A request with verify_rows > 0 does NOT also receive a Decode grant
    /// (spec §5.3 S3).
    #[test]
    fn verify_not_plus_decode() {
        let mut q = queue(4096, 4, 1);
        q.admit(1, "s1", 0).unwrap();
        // Request wants decode AND has verify rows.
        q.set_needs(1, true, 3, 0).unwrap();

        let sel = q.select(4, 0, 4096);
        // Should get Verify, not Decode.
        assert_eq!(sel.grants.len(), 1);
        assert!(
            matches!(sel.grants[0], Grant::Verify { id: 1, rows: 3 }),
            "expected Verify grant, got {:?}",
            sel.grants[0]
        );
        // No Decode grant for this request.
        assert!(!sel.grants.iter().any(|g| matches!(g, Grant::Decode { .. })));
    }

    /// Test 4: oldest-blocked stops younger (starved_oldest).
    ///
    /// The oldest request needs prefill but the budget is consumed by decode
    /// lanes. It is individually feasible (fits max_batch_tokens alone) but
    /// cannot get its quantum this tick → starved_oldest is set (spec §5.3
    /// S3.4).
    #[test]
    fn oldest_blocked_stops_younger() {
        // Small budget: 10 rows max. decode_lanes=4, prefill_min=1.
        // 4 decode rows + 1 prefill min = 5 <= 10, valid config.
        let mut q = queue(10, 4, 1);

        // Oldest request: wants prefill (50 tokens), no decode.
        q.admit(1, "s1", 50).unwrap();
        // Younger requests: want decode.
        q.admit(2, "s2", 0).unwrap();
        q.admit(3, "s3", 0).unwrap();
        q.admit(4, "s4", 0).unwrap();
        q.admit(5, "s5", 0).unwrap();

        // Oldest needs prefill only.
        q.set_needs(1, false, 0, 0).unwrap();
        // Younger all want decode.
        q.set_needs(2, true, 0, 0).unwrap();
        q.set_needs(3, true, 0, 0).unwrap();
        q.set_needs(4, true, 0, 0).unwrap();
        q.set_needs(5, true, 0, 0).unwrap();

        // remaining_rows = 10. 4 decode rows used, 6 left for prefill.
        // Oldest needs 50 prefill but only 6 available → gets 6 rows.
        // But wait, prefill_quantum might limit it. Let's use quantum=10.
        let sel = q.select(4, 10, 10);
        // 4 decode grants + 1 prefill grant (6 rows).
        assert_eq!(sel.grants.len(), 5);
        // The oldest request got prefill.
        assert!(sel
            .grants
            .iter()
            .any(|g| matches!(g, Grant::Prefill { id: 1, rows: 6 })));

        // Now test actual starvation: remaining_rows too small for prefill_min.
        // Reset: re-admit with fresh queue.
        let mut q = queue(10, 4, 5);
        q.admit(1, "s1", 50).unwrap();
        q.admit(2, "s2", 0).unwrap();
        q.admit(3, "s3", 0).unwrap();
        q.admit(4, "s4", 0).unwrap();
        q.admit(5, "s5", 0).unwrap();
        q.set_needs(1, false, 0, 0).unwrap();
        q.set_needs(2, true, 0, 0).unwrap();
        q.set_needs(3, true, 0, 0).unwrap();
        q.set_needs(4, true, 0, 0).unwrap();
        q.set_needs(5, true, 0, 0).unwrap();

        // 4 decode rows used, 6 left. prefill_min=5, so 6 >= 5 → prefill gets 6.
        // To starve: use remaining_rows=8. 4 decode + 4 left. prefill_min=5 > 4.
        // But wait, decode_lanes=4 and prefill_min=5: 4+5=9 <= 10, valid.
        let sel = q.select(4, 10, 8);
        // 4 decode grants. 4 rows left, prefill_min=5 → no prefill.
        assert!(sel.grants.iter().all(|g| matches!(g, Grant::Decode { .. })));
        // Oldest request (id=1) has 50 uncached, needed_rows=50 <= 10? No!
        // 50 > 10, so it's NOT individually feasible. starved_oldest should be false.
        // Let me fix: make the oldest request's needs fit in max_batch_tokens.
        assert!(
            !sel.starved_oldest,
            "oldest not individually feasible, should not starve"
        );

        // Now make oldest individually feasible: uncached=5 (fits in 10).
        let mut q = queue(10, 4, 5);
        q.admit(1, "s1", 5).unwrap();
        q.admit(2, "s2", 0).unwrap();
        q.admit(3, "s3", 0).unwrap();
        q.admit(4, "s4", 0).unwrap();
        q.admit(5, "s5", 0).unwrap();
        q.set_needs(1, false, 0, 0).unwrap();
        q.set_needs(2, true, 0, 0).unwrap();
        q.set_needs(3, true, 0, 0).unwrap();
        q.set_needs(4, true, 0, 0).unwrap();
        q.set_needs(5, true, 0, 0).unwrap();

        // remaining_rows=8: 4 decode, 4 left. prefill_min=5 > 4 → no prefill.
        // Oldest needs 5 prefill, 5 <= 10 (max_batch_tokens) → feasible.
        // But couldn't get it → starved_oldest = true.
        let sel = q.select(4, 10, 8);
        assert!(sel.grants.iter().all(|g| matches!(g, Grant::Decode { .. })));
        assert!(
            sel.starved_oldest,
            "oldest is individually feasible but starved, starved_oldest should be true"
        );
    }

    /// Test 5: empty queue.
    #[test]
    fn empty_queue() {
        let mut q = queue(4096, 4, 1);
        let sel = q.select(4, 64, 4096);
        assert!(sel.grants.is_empty());
        assert!(!sel.starved_oldest);
        // Tick still advances.
        assert_eq!(q.tick(), 1);
    }

    /// Test 6: overflow-safe row sums.
    ///
    /// Checked arithmetic throughout — no panics on overflow (spec §5.1).
    #[test]
    fn overflow_safe_row_sums() {
        // Constructor: decode_lanes + prefill_min_tokens must not overflow.
        let q = FairQueue::new(u64::MAX, u64::MAX, 1);
        assert!(q.is_err(), "overflow in constructor should error");

        // Valid queue with u64::MAX uncached tokens.
        let mut q = queue(4096, 1, 1);
        q.admit(1, "s1", u64::MAX).unwrap();
        q.set_needs(1, true, u64::MAX, u64::MAX).unwrap();
        // needed_rows: 0 (decode suppressed by verify) + u64::MAX + u64::MAX + u64::MAX
        // → overflow → None. needed_rows() returns None.
        assert_eq!(q.get(1).unwrap().needed_rows(), None);

        // select should not panic; starved_oldest uses u64::MAX for overflowed needs.
        let sel = q.select(1, 64, 4096);
        // Verify rows = u64::MAX > 4096, won't fit. No grants.
        // starved_oldest: needed = None → u64::MAX > max_batch_tokens → not feasible → false.
        assert!(!sel.starved_oldest);
    }

    /// Constructor rejects invalid config.
    #[test]
    fn constructor_rejects_invalid_config() {
        let err = FairQueue::new(5, 4, 2).unwrap_err();
        assert!(matches!(err, FairnessError::InvalidConfig { .. }));
    }

    #[test]
    fn constructor_rejects_zero_prefill_minimum() {
        assert!(matches!(
            FairQueue::new(4096, 4, 0),
            Err(FairnessError::ZeroMinTokens)
        ));
    }

    /// Duplicate admit is rejected.
    #[test]
    fn duplicate_admit_rejected() {
        let mut q = queue(4096, 4, 1);
        q.admit(1, "s1", 100).unwrap();
        assert!(matches!(
            q.admit(1, "s1", 50),
            Err(FairnessError::DuplicateId(1))
        ));
    }

    /// set_needs on unknown id is rejected.
    #[test]
    fn set_needs_unknown_id() {
        let mut q = queue(4096, 4, 1);
        assert!(matches!(
            q.set_needs(99, true, 0, 0),
            Err(FairnessError::NotFound(99))
        ));
    }

    /// remove works and clamps cursor.
    #[test]
    fn remove_request() {
        let mut q = queue(4096, 4, 1);
        q.admit(1, "s1", 100).unwrap();
        q.admit(2, "s2", 100).unwrap();
        q.remove(1).unwrap();
        assert_eq!(q.len(), 1);
        assert!(q.get(1).is_none());
        assert!(q.get(2).is_some());
        // Remove unknown.
        assert!(matches!(q.remove(99), Err(FairnessError::NotFound(99))));
    }

    /// Prefill quantum is respected — no more than quantum rows per request.
    #[test]
    fn prefill_quantum_respected() {
        let mut q = queue(4096, 0, 1);
        q.admit(1, "s1", 1000).unwrap();
        let sel = q.select(0, 64, 4096);
        assert_eq!(sel.grants.len(), 1);
        assert!(matches!(sel.grants[0], Grant::Prefill { id: 1, rows: 64 }));
        // uncached reduced by 64.
        assert_eq!(q.get(1).unwrap().uncached_prefill_tokens, 936);
    }

    /// Multiple decode requests all get served when budget allows.
    #[test]
    fn multiple_decode_grants() {
        let mut q = queue(4096, 4, 1);
        q.admit(1, "s1", 0).unwrap();
        q.admit(2, "s2", 0).unwrap();
        q.admit(3, "s3", 0).unwrap();
        q.set_needs(1, true, 0, 0).unwrap();
        q.set_needs(2, true, 0, 0).unwrap();
        q.set_needs(3, true, 0, 0).unwrap();
        let sel = q.select(4, 0, 4096);
        assert_eq!(sel.grants.len(), 3);
        assert!(sel.grants.iter().all(|g| matches!(g, Grant::Decode { .. })));
    }

    /// Forced rows are allocated from remaining quota.
    #[test]
    fn forced_rows_allocated() {
        let mut q = queue(4096, 4, 1);
        q.admit(1, "s1", 0).unwrap();
        q.set_needs(1, false, 0, 5).unwrap();
        let sel = q.select(0, 0, 4096);
        assert_eq!(sel.grants.len(), 1);
        assert!(matches!(sel.grants[0], Grant::Forced { id: 1, rows: 5 }));
    }

    /// Decode + prefill + verify + forced in one step.
    #[test]
    fn mixed_grants_in_one_step() {
        let mut q = queue(4096, 4, 1);
        q.admit(1, "s1", 0).unwrap(); // decode
        q.admit(2, "s2", 100).unwrap(); // prefill
        q.admit(3, "s3", 0).unwrap(); // verify
        q.set_needs(1, true, 0, 0).unwrap();
        q.set_needs(2, false, 0, 0).unwrap();
        q.set_needs(3, false, 2, 0).unwrap();
        let sel = q.select(4, 64, 4096);
        let has_decode = sel
            .grants
            .iter()
            .any(|g| matches!(g, Grant::Decode { id: 1 }));
        let has_prefill = sel
            .grants
            .iter()
            .any(|g| matches!(g, Grant::Prefill { id: 2, .. }));
        let has_verify = sel
            .grants
            .iter()
            .any(|g| matches!(g, Grant::Verify { id: 3, .. }));
        assert!(has_decode && has_prefill && has_verify);
    }

    /// Total grant rows never exceed remaining_rows.
    #[test]
    fn total_rows_within_budget() {
        let mut q = queue(4096, 4, 1);
        q.admit(1, "s1", 0).unwrap();
        q.admit(2, "s2", 500).unwrap();
        q.admit(3, "s3", 0).unwrap();
        q.set_needs(1, true, 0, 0).unwrap();
        q.set_needs(2, false, 0, 0).unwrap();
        q.set_needs(3, false, 0, 3).unwrap();
        let remaining = 20u64;
        let sel = q.select(4, 64, remaining);
        let total: u64 = sel.grants.iter().map(|g| g.rows()).sum();
        assert!(
            total <= remaining,
            "total {total} exceeds remaining {remaining}"
        );
    }

    /// Prefill tail smaller than the minimum quantum must still be granted
    /// (it completes the prompt). Regression: refusing grants below
    /// prefill_min_tokens livelocked any request whose final chunk was
    /// smaller than the quantum — the slot never finished prefill.
    #[test]
    fn prefill_tail_below_min_quantum_completes() {
        let mut q = queue(4096, 0, 512);
        q.admit(1, "s1", 3).unwrap(); // only 3 uncached tokens left
        q.set_needs(1, false, 0, 0).unwrap();

        // quantum = prefill_min = 512; avail is plentiful.
        let sel = q.select(0, 512, 4096);
        assert_eq!(sel.grants.len(), 1, "final sliver must be granted");
        assert!(
            matches!(sel.grants[0], Grant::Prefill { id: 1, rows: 3 }),
            "expected the 3-row completing grant, got {:?}",
            sel.grants[0]
        );
        assert_eq!(q.get(1).unwrap().uncached_prefill_tokens, 0);

        // The same rule under a tight budget: avail=2 < uncached=3 < min=512.
        // The grant doesn't complete the prompt and is below the quantum, so
        // it may be skipped (sliver control still applies).
        let mut q = queue(4096, 0, 512);
        q.admit(1, "s1", 3).unwrap();
        q.set_needs(1, false, 0, 0).unwrap();
        let sel = q.select(0, 512, 2);
        assert!(
            sel.grants.is_empty(),
            "2-row non-completing sliver stays refused"
        );
    }

    #[test]
    fn final_sub_minimum_prefill_fits_tight_remaining_budget() {
        let mut q = queue(4096, 0, 512);
        q.admit(1, "s1", 3).unwrap();
        let sel = q.select(0, 512, 3);
        assert_eq!(
            sel.grants,
            vec![Grant::Prefill { id: 1, rows: 3 }],
            "a final prompt tail must progress even when the caller reserved the rest for verify rows"
        );
    }

    /// Age is recomputed per round: a served request leaves the aged band
    /// and can be re-aged by a later skipped round (spec §5.3 S3.3).
    #[test]
    fn aged_band_is_recomputed_not_permanent() {
        let mut q = queue(4096, 0, 1);
        q.admit(1, "s1", 100).unwrap();
        q.admit(2, "s2", 100).unwrap();
        q.set_needs(1, false, 0, 0).unwrap();
        q.set_needs(2, false, 0, 0).unwrap();

        // Round 1: id 1 wins (same tick, same locality — stable order).
        let sel = q.select(0, 10, 10);
        assert!(matches!(sel.grants[0], Grant::Prefill { id: 1, .. }));
        q.skip_round_complete();
        assert!(q.get(2).unwrap().aged, "skipped request ages");

        // Round 2: aged id 2 wins.
        let sel = q.select(0, 10, 10);
        assert!(matches!(sel.grants[0], Grant::Prefill { id: 2, .. }));
        q.skip_round_complete();
        assert!(
            !q.get(2).unwrap().aged,
            "served request must leave the aged band"
        );
        assert!(q.get(1).unwrap().aged, "now-starved request ages in turn");
    }
}
