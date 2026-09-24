// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// SessionTable — which session holds which slot.
//
// Two properties prevent whole classes of bug:
//
// 1. Admission runs BEFORE the slot is taken, so a rejected request leaves
//    the pool untouched. Otherwise a rejected client silently consumes
//    capacity.
// 2. Session ids are never reused: a monotonically increasing counter, not
//    a free-list index. A stale id from a closed session resolves to
//    `None`, never silently addressing whoever now holds that slot.

use std::collections::HashMap;

use crate::admission::{AdmissionController, AdmitError};
use crate::prefix::{plan_turn, TurnPlan};
use rdna_compute::slot_pool::{SlotId, SlotPool};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

/// Where a session's GPU state currently lives.
///
/// A property of the SESSION, not the slot: a restored session may land in a
/// different slot than it left, so nothing may assume slot affinity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    /// Holds a slot; its KV and recurrent state are on the GPU.
    Resident,
    /// Holds no slot, but a snapshot exists and can be restored.
    Swapped,
    /// Holds no slot and no snapshot. Its tokens are still authoritative, so
    /// the next turn re-prefills from them. Every swap failure lands here.
    Cold,
}

pub struct Session {
    /// `None` whenever the session is not `Resident`.
    pub slot: Option<SlotId>,
    pub granted_ctx: usize,
    pub tokens: Vec<u32>,
    pub next_pos: usize,
    pub residency: Residency,
    /// Hashes of this conversation's USER turns, in order.
    ///
    /// Identity is the user turns alone, deliberately. The assistant side is
    /// whatever *we* generated, and the client's echo of it may differ —
    /// reasoning split into a separate channel, whitespace, or an edited
    /// message. Matching on the user turns and then replaying our own tokens
    /// keeps the prompt aligned with the KV we actually hold.
    pub convo: Vec<u64>,
    /// Monotonic stamp for LRU. Bumped by `touch`.
    pub last_used: u64,
    /// Whether this session currently holds a VRAM admission. Admission is
    /// charged for GPU-resident state only: a Swapped or Cold session holds no
    /// KV on the device, so leaving it charged would let idle conversations
    /// exhaust the budget while slots sit free. Private so the charge can only
    /// be taken and returned through the methods below, exactly once each.
    admitted: bool,
}

#[derive(Default)]
pub struct SessionTable {
    sessions: HashMap<u64, Session>,
    next_id: u64,
    clock: u64,
}

impl SessionTable {
    /// Admit a session and assign it a slot.
    ///
    /// Admission runs FIRST: only a successful `adm.admit` takes a slot from
    /// `pool`, so a rejected request leaves the pool untouched. If admission
    /// succeeds but the pool has no free slot, the admitted budget is handed
    /// straight back so the two stay consistent.
    pub fn open(
        &mut self,
        pool: &mut SlotPool,
        adm: &mut AdmissionController,
        requested_ctx: usize,
    ) -> Result<SessionId, AdmitError> {
        let granted_ctx = adm.admit(requested_ctx)?;
        let slot = match pool.acquire() {
            Some(slot) => slot,
            None => {
                adm.release(granted_ctx);
                return Err(AdmitError::PoolFull);
            }
        };
        // Monotonically increasing, never reused: a stale id from a closed
        // session must resolve to `None`, never silently address whoever now
        // holds that slot.
        let id = self.next_id;
        self.next_id += 1;
        self.sessions.insert(
            id,
            Session {
                slot: Some(slot),
                granted_ctx,
                tokens: Vec::new(),
                next_pos: 0,
                residency: Residency::Resident,
                convo: Vec::new(),
                last_used: {
                    self.clock += 1;
                    self.clock
                },
                admitted: true,
            },
        );
        Ok(SessionId(id))
    }

    /// Close a session, returning its slot and budget together. The budget is
    /// returned only if the session still holds it; a Swapped or Cold session
    /// already gave it back when it left the GPU.
    pub fn close(&mut self, pool: &mut SlotPool, adm: &mut AdmissionController, id: SessionId) {
        if let Some(session) = self.sessions.remove(&id.0) {
            if let Some(slot) = session.slot {
                pool.release(slot);
            }
            if session.admitted {
                adm.release(session.granted_ctx);
            }
        }
    }

    /// Re-take the VRAM admission of a session that left the GPU, before its
    /// state is restored into a slot. No-op for a session that already holds
    /// it. On error the session stays unadmitted; the caller should mark it
    /// Cold.
    pub fn readmit(
        &mut self,
        adm: &mut AdmissionController,
        id: SessionId,
    ) -> Result<(), AdmitError> {
        let Some(s) = self.sessions.get_mut(&id.0) else {
            return Ok(());
        };
        if !s.admitted {
            adm.admit(s.granted_ctx)?;
            s.admitted = true;
        }
        Ok(())
    }

    /// Whether the session currently holds a VRAM admission.
    pub fn is_admitted(&self, id: SessionId) -> bool {
        self.sessions.get(&id.0).is_some_and(|s| s.admitted)
    }

    pub fn get(&self, id: SessionId) -> Option<&Session> {
        self.sessions.get(&id.0)
    }

    pub fn get_mut(&mut self, id: SessionId) -> Option<&mut Session> {
        self.sessions.get_mut(&id.0)
    }

    /// Start a turn: decide how much of `prompt` this session's slot already
    /// holds, rewind to that point, and report what remains to prefill.
    ///
    /// The rewind moves no data. Lowering `seq_len` is enough because KV past
    /// that point is overwritten by the next prefill, and `positions[]` -- not
    /// `seq_len` -- is what bounds attention per row.
    ///
    /// The caller must then prefill `prompt[plan.reused..]` and append those
    /// tokens to `session.tokens`.
    ///
    /// Only ever *lowers* `seq_len` and truncates tokens, so the worst
    /// outcome of a wrong answer here is a full re-prefill, never stale KV
    /// being read as valid.
    pub fn begin_turn(
        &mut self,
        pool: &mut SlotPool,
        id: SessionId,
        prompt: &[u32],
    ) -> Result<TurnPlan, String> {
        let session = self
            .sessions
            .get_mut(&id.0)
            .ok_or_else(|| format!("begin_turn: unknown session {}", id.0))?;
        let slot = session
            .slot
            .ok_or_else(|| format!("begin_turn: session {} holds no slot", id.0))?;
        let held = pool.descriptors()[slot.0].seq_len as usize;
        let plan = plan_turn(&session.tokens, held, prompt);
        pool.set_seq_len(slot, plan.reused)
            .map_err(|e| format!("begin_turn: {e}"))?;
        session.tokens.truncate(plan.reused);
        session.next_pos = plan.reused;
        Ok(plan)
    }

    /// Find a resident, non-busy session whose conversation `convo` is one user
    /// turn shorter than `want` and a prefix of it — i.e. `want` continues it.
    ///
    /// Exactly one turn shorter, not merely a prefix: a session two turns
    /// behind is missing an assistant reply that never entered its KV, so
    /// appending the newest user turn to it would skip a turn.
    /// Returns None when more than one non-busy matching candidate exists —
    /// an LRU pick among duplicates would append to the wrong session.
    pub fn find_continuation(&self, want: &[u64], busy: &[SessionId]) -> Option<SessionId> {
        if want.len() < 2 {
            return None;
        }
        let expect = &want[..want.len() - 1];
        let mut candidates: Vec<(u64, u64)> = Vec::new();
        for (id, s) in self.sessions.iter() {
            if s.residency == Residency::Cold {
                continue;
            }
            if s.convo.as_slice() != expect {
                continue;
            }
            if s.tokens.is_empty() {
                continue;
            }
            if busy.iter().any(|b| b.0 == *id) {
                continue;
            }
            candidates.push((*id, s.last_used));
        }
        if candidates.len() != 1 {
            return None;
        }
        Some(SessionId(candidates[0].0))
    }

    /// Confirm that `id` is the session a tool-result iteration names, and that
    /// it may be extended in place.
    ///
    /// A tool-result turn adds no user turn, so `want` matches the session's
    /// conversation exactly — which is precisely why it must NOT be searched
    /// for: every duplicate of the same conversation matches equally well, and
    /// an LRU pick among them appends `<tool_response>` blocks to whichever
    /// session was touched last, not to the one that emitted the calls being
    /// answered. The caller identifies the session from the tool-call ids it
    /// handed the client; this only re-checks the session is still that
    /// conversation, so a stale or evicted id falls back to a cold prefill.
    pub fn confirm_reentry(
        &self,
        id: SessionId,
        want: &[u64],
        busy: &[SessionId],
    ) -> Option<SessionId> {
        if want.is_empty() || busy.iter().any(|b| b.0 == id.0) {
            return None;
        }
        let s = self.sessions.get(&id.0)?;
        (s.residency != Residency::Cold && s.convo.as_slice() == want && !s.tokens.is_empty())
            .then_some(id)
    }

    /// Mark a session as most-recently used, for LRU.
    pub fn touch(&mut self, id: SessionId) {
        self.clock += 1;
        let now = self.clock;
        if let Some(s) = self.sessions.get_mut(&id.0) {
            s.last_used = now;
        }
    }

    /// The least-recently-used session that currently holds a slot and is not
    /// in `busy`.
    ///
    /// Sessions mid-generation are never candidates: preempting one would
    /// strand a half-finished response. When every resident session is busy
    /// this returns `None`, and the caller must queue or reject rather than
    /// thrash.
    pub fn lru_idle_victim(&self, busy: &[SessionId]) -> Option<SessionId> {
        self.sessions
            .iter()
            .filter(|(id, s)| s.slot.is_some() && !busy.iter().any(|b| b.0 == **id))
            .min_by_key(|(_, s)| s.last_used)
            .map(|(id, _)| SessionId(*id))
    }

    /// Give up the slot and its VRAM admission, keeping the session
    /// restorable from its snapshot.
    pub fn mark_swapped(
        &mut self,
        pool: &mut SlotPool,
        adm: &mut AdmissionController,
        id: SessionId,
    ) {
        if let Some(s) = self.sessions.get_mut(&id.0) {
            if let Some(slot) = s.slot.take() {
                pool.release(slot);
            }
            if s.admitted {
                adm.release(s.granted_ctx);
                s.admitted = false;
            }
            s.residency = Residency::Swapped;
        }
    }

    /// Give up the slot with no usable snapshot. The tokens survive, so the
    /// next turn re-prefills — slow, never wrong. Every swap failure ends here.
    /// The VRAM admission is returned too; a re-prefill opens a fresh session.
    pub fn mark_cold(&mut self, pool: &mut SlotPool, adm: &mut AdmissionController, id: SessionId) {
        if let Some(s) = self.sessions.get_mut(&id.0) {
            if let Some(slot) = s.slot.take() {
                pool.release(slot);
            }
            if s.admitted {
                adm.release(s.granted_ctx);
                s.admitted = false;
            }
            s.residency = Residency::Cold;
            s.next_pos = 0;
        }
    }

    /// Re-attach a session to a slot after its state has been restored.
    /// Call [`Self::readmit`] first; this does not touch admission.
    pub fn mark_resident(&mut self, id: SessionId, slot: SlotId, seq_len: usize) {
        self.clock += 1;
        let now = self.clock;
        if let Some(s) = self.sessions.get_mut(&id.0) {
            s.slot = Some(slot);
            s.residency = Residency::Resident;
            s.next_pos = seq_len;
            s.last_used = now;
        }
    }

    /// Iterate sessions as `(id, session)`. For diagnostics.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &Session)> {
        self.sessions.iter().map(|(id, s)| (*id, s))
    }

    pub fn resident(&self) -> usize {
        self.sessions.values().filter(|s| s.slot.is_some()).count()
    }

    pub fn active(&self) -> usize {
        self.sessions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::{AdmissionController, ModelFootprint};
    use rdna_compute::slot_pool::SlotPool;

    const GIB: u64 = 1024 * 1024 * 1024;
    const PPB: usize = 1088;

    fn rig(n_slots: usize) -> (SlotPool, AdmissionController, SessionTable) {
        let pool = SlotPool::new(n_slots, 4096, PPB).unwrap();
        let adm = AdmissionController::new(
            ModelFootprint {
                weights_bytes: GIB,
                kv_bytes_per_token: 1024,
            },
            32 * GIB,
        );
        (pool, adm, SessionTable::default())
    }

    #[test]
    fn open_assigns_a_slot_and_close_returns_it() {
        let (mut pool, mut adm, mut t) = rig(1);
        let id = t.open(&mut pool, &mut adm, 1024).unwrap();
        assert_eq!(t.active(), 1);
        // The single slot is taken.
        assert!(t.open(&mut pool, &mut adm, 1024).is_err());
        t.close(&mut pool, &mut adm, id);
        assert_eq!(t.active(), 0);
        // And is reusable.
        t.open(&mut pool, &mut adm, 1024)
            .expect("slot must be reusable");
    }

    #[test]
    fn a_rejected_admission_does_not_consume_a_slot() {
        let (mut pool, mut adm, mut t) = rig(2);
        // Far beyond the budget.
        assert!(t.open(&mut pool, &mut adm, 100_000_000).is_err());
        assert_eq!(
            t.active(),
            0,
            "a rejected session must leave the pool untouched"
        );
        t.open(&mut pool, &mut adm, 1024)
            .expect("pool must still have both slots");
    }

    #[test]
    fn sessions_keep_independent_token_history() {
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        let b = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.get_mut(a).unwrap().tokens.extend_from_slice(&[1, 2, 3]);
        t.get_mut(b).unwrap().tokens.push(9);
        assert_eq!(t.get(a).unwrap().tokens, vec![1, 2, 3]);
        assert_eq!(t.get(b).unwrap().tokens, vec![9]);
    }

    #[test]
    fn closing_frees_budget_for_a_later_session() {
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        let before = adm.used_bytes();
        t.close(&mut pool, &mut adm, a);
        assert!(adm.used_bytes() < before, "close must return budget");
    }

    #[test]
    fn begin_turn_rewinds_the_slot_and_reports_the_suffix() {
        let (mut pool, mut adm, mut t) = rig(1);
        let id = t.open(&mut pool, &mut adm, 1024).unwrap();
        // Turn 1: four tokens are prefilled and recorded.
        {
            let s = t.get_mut(id).unwrap();
            s.tokens.extend_from_slice(&[1, 2, 3, 4]);
            s.next_pos = 4;
        }
        pool.set_seq_len(t.get(id).unwrap().slot.unwrap(), 4)
            .unwrap();

        // Turn 2 continues the same conversation.
        let plan = t.begin_turn(&mut pool, id, &[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(plan.reused, 4);
        assert_eq!(plan.to_prefill, 2);
        let s = t.get(id).unwrap();
        assert_eq!(s.next_pos, 4, "next_pos must resume at the reuse point");
        assert_eq!(
            s.tokens,
            vec![1, 2, 3, 4],
            "tokens truncated to the reuse point"
        );
        assert_eq!(pool.descriptors()[s.slot.unwrap().0].seq_len, 4);
    }

    #[test]
    fn begin_turn_on_divergence_rewinds_to_the_common_prefix() {
        let (mut pool, mut adm, mut t) = rig(1);
        let id = t.open(&mut pool, &mut adm, 1024).unwrap();
        {
            let s = t.get_mut(id).unwrap();
            s.tokens.extend_from_slice(&[1, 2, 3, 4]);
            s.next_pos = 4;
        }
        pool.set_seq_len(t.get(id).unwrap().slot.unwrap(), 4)
            .unwrap();

        let plan = t.begin_turn(&mut pool, id, &[1, 2, 7, 8]).unwrap();
        assert_eq!(plan.reused, 2);
        assert_eq!(plan.to_prefill, 2);
        let s = t.get(id).unwrap();
        assert_eq!(s.tokens, vec![1, 2], "diverged tokens are dropped");
        assert_eq!(s.next_pos, 2);
        assert_eq!(
            pool.descriptors()[s.slot.unwrap().0].seq_len,
            2,
            "the slot must forget the diverged KV"
        );
    }

    #[test]
    fn begin_turn_on_an_unknown_session_is_an_error_not_a_panic() {
        let (mut pool, mut adm, mut t) = rig(1);
        let id = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.close(&mut pool, &mut adm, id);
        assert!(t.begin_turn(&mut pool, id, &[1, 2]).is_err());
    }

    #[test]
    fn find_continuation_matches_the_previous_turn() {
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        {
            let s = t.get_mut(a).unwrap();
            s.convo = vec![11];
            s.tokens = vec![1, 2, 3];
        }
        assert_eq!(t.find_continuation(&[11, 22], &[]), Some(a));
    }

    #[test]
    fn find_continuation_matches_a_swapped_session() {
        // The whole point of swap: its snapshot holds the KV this turn wants.
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        {
            let s = t.get_mut(a).unwrap();
            s.convo = vec![11];
            s.tokens = vec![1, 2, 3];
        }
        t.mark_swapped(&mut pool, &mut adm, a);
        assert_eq!(t.find_continuation(&[11, 22], &[]), Some(a));
    }

    #[test]
    fn find_continuation_refuses_a_cold_session() {
        // Cold means no snapshot exists, so there is nothing to restore.
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        {
            let s = t.get_mut(a).unwrap();
            s.convo = vec![11];
            s.tokens = vec![1, 2, 3];
        }
        t.mark_cold(&mut pool, &mut adm, a);
        assert_eq!(t.find_continuation(&[11, 22], &[]), None);
    }

    #[test]
    fn find_continuation_refuses_a_different_conversation() {
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        {
            let s = t.get_mut(a).unwrap();
            s.convo = vec![99];
            s.tokens = vec![1];
        }
        assert_eq!(t.find_continuation(&[11, 22], &[]), None);
    }

    #[test]
    fn find_continuation_refuses_a_session_two_turns_behind() {
        // Its KV never saw turn 2's exchange, so appending turn 3 would skip one.
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        {
            let s = t.get_mut(a).unwrap();
            s.convo = vec![11];
            s.tokens = vec![1];
        }
        assert_eq!(t.find_continuation(&[11, 22, 33], &[]), None);
    }

    #[test]
    fn find_continuation_needs_a_first_turn_to_continue() {
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.get_mut(a).unwrap().convo = vec![11];
        assert_eq!(
            t.find_continuation(&[11], &[]),
            None,
            "turn 1 is not a continuation"
        );
    }

    #[test]
    fn confirm_reentry_accepts_the_named_session_holding_that_conversation() {
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        {
            let s = t.get_mut(a).unwrap();
            s.convo = vec![11, 22];
            s.tokens = vec![1, 2, 3];
        }
        assert_eq!(t.confirm_reentry(a, &[11, 22], &[]), Some(a));
    }

    #[test]
    fn confirm_reentry_never_falls_through_to_a_duplicate_conversation() {
        // Two sessions, same user turns, different generated tool calls. The
        // caller names the one that emitted the calls being answered; the other
        // must not stand in for it however recently it was used.
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        let b = t.open(&mut pool, &mut adm, 1024).unwrap();
        for id in [a, b] {
            let s = t.get_mut(id).unwrap();
            s.convo = vec![11];
            s.tokens = vec![1, 2, 3];
        }
        t.touch(b);
        assert_eq!(t.confirm_reentry(a, &[11], &[]), Some(a));
        t.close(&mut pool, &mut adm, a);
        assert_eq!(
            t.confirm_reentry(a, &[11], &[]),
            None,
            "a closed session must not resolve to its surviving twin"
        );
    }

    #[test]
    fn confirm_reentry_refuses_a_session_whose_conversation_moved_on() {
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        {
            let s = t.get_mut(a).unwrap();
            s.convo = vec![11, 22];
            s.tokens = vec![1, 2, 3];
        }
        assert_eq!(t.confirm_reentry(a, &[11], &[]), None);
    }

    #[test]
    fn confirm_reentry_refuses_cold_busy_and_empty_sessions() {
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        {
            let s = t.get_mut(a).unwrap();
            s.convo = vec![11];
            s.tokens = vec![1];
        }
        assert_eq!(t.confirm_reentry(a, &[11], &[a]), None, "busy");
        t.get_mut(a).unwrap().tokens.clear();
        assert_eq!(t.confirm_reentry(a, &[11], &[]), None, "nothing in its KV");
        t.get_mut(a).unwrap().tokens = vec![1];
        t.mark_cold(&mut pool, &mut adm, a);
        assert_eq!(t.confirm_reentry(a, &[11], &[]), None, "cold");
    }

    #[test]
    fn lru_victim_is_the_least_recently_used_idle_session() {
        let (mut pool, mut adm, mut t) = rig(3);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        let b = t.open(&mut pool, &mut adm, 1024).unwrap();
        let c = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.touch(a);
        t.touch(c);
        t.touch(b); // a is now oldest
        assert_eq!(t.lru_idle_victim(&[]), Some(a));
    }

    #[test]
    fn a_busy_session_is_never_evicted() {
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        let b = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.touch(a);
        t.touch(b); // a is oldest, but busy
        assert_eq!(t.lru_idle_victim(&[a]), Some(b));
        assert_eq!(
            t.lru_idle_victim(&[a, b]),
            None,
            "all resident sessions busy => no victim, caller must queue"
        );
    }

    #[test]
    fn a_swapped_session_holds_no_slot_and_frees_it() {
        let (mut pool, mut adm, mut t) = rig(1);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.mark_swapped(&mut pool, &mut adm, a);
        assert!(t.get(a).unwrap().slot.is_none());
        assert_eq!(t.get(a).unwrap().residency, Residency::Swapped);
        assert_eq!(t.resident(), 0);
        t.open(&mut pool, &mut adm, 1024)
            .expect("the freed slot must be reusable");
    }

    #[test]
    fn a_cold_session_keeps_its_tokens() {
        let (mut pool, mut adm, mut t) = rig(1);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.get_mut(a).unwrap().tokens.extend_from_slice(&[1, 2, 3]);
        t.mark_cold(&mut pool, &mut adm, a);
        assert_eq!(t.get(a).unwrap().residency, Residency::Cold);
        assert_eq!(
            t.get(a).unwrap().tokens,
            vec![1, 2, 3],
            "tokens are authoritative and must survive going cold"
        );
        assert_eq!(t.get(a).unwrap().next_pos, 0, "cold means recompute from 0");
    }

    #[test]
    fn a_restored_session_may_land_in_a_different_slot() {
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        let first = t.get(a).unwrap().slot.unwrap();
        t.mark_swapped(&mut pool, &mut adm, a);
        // The pool hands back whichever slot is free, which is usually the one
        // just released. What matters is that mark_resident accepts ANY slot --
        // residency is a property of the session, not the slot -- so restore
        // into a deliberately different one.
        let other = SlotId(if first.0 == 0 { 1 } else { 0 });
        t.mark_resident(a, other, 7);
        assert_eq!(t.get(a).unwrap().slot, Some(other));
        assert_eq!(t.get(a).unwrap().next_pos, 7);
        assert_eq!(t.get(a).unwrap().residency, Residency::Resident);
    }

    #[test]
    fn begin_turn_on_a_swapped_session_is_an_error() {
        let (mut pool, mut adm, mut t) = rig(1);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.mark_swapped(&mut pool, &mut adm, a);
        assert!(
            t.begin_turn(&mut pool, a, &[1, 2]).is_err(),
            "a session with no slot cannot begin a turn"
        );
    }

    /// Budget that fits the weights plus exactly `n` sessions of `ctx`.
    fn rig_budget(
        n_slots: usize,
        ctx: usize,
        n: u64,
    ) -> (SlotPool, AdmissionController, SessionTable) {
        let pool = SlotPool::new(n_slots, 4096, PPB).unwrap();
        let adm = AdmissionController::new(
            ModelFootprint {
                weights_bytes: GIB,
                kv_bytes_per_token: 1024,
            },
            GIB + n * ctx as u64 * 1024 + 1,
        );
        (pool, adm, SessionTable::default())
    }

    #[test]
    fn eviction_returns_admission_so_new_conversations_keep_opening() {
        // 2 slots, budget for 2 sessions. Every new conversation evicts the
        // LRU idle one, the way serve_engine's admit path does. Before
        // admission was residency-scoped, swapped sessions stayed charged and
        // the third conversation was refused with WouldExceedBudget while a
        // slot sat free.
        let (mut pool, mut adm, mut t) = rig_budget(2, 1024, 2);
        for i in 0..10 {
            let id = match t.open(&mut pool, &mut adm, 1024) {
                Ok(id) => id,
                Err(_) => {
                    let v = t.lru_idle_victim(&[]).expect("idle victim");
                    t.mark_swapped(&mut pool, &mut adm, v);
                    t.open(&mut pool, &mut adm, 1024)
                        .unwrap_or_else(|e| panic!("conversation {i} refused: {e}"))
                }
            };
            t.touch(id);
        }
        assert_eq!(t.resident(), 2);
        assert_eq!(adm.used_bytes(), GIB + 2 * 1024 * 1024);
    }

    #[test]
    fn closing_a_swapped_session_does_not_release_twice() {
        // release() matches by granted ctx, so a second release of a session
        // that already gave its budget back would silently drop ANOTHER
        // resident session's charge and let admission over-commit VRAM.
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        let _b = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.mark_swapped(&mut pool, &mut adm, a);
        let _c = t.open(&mut pool, &mut adm, 1024).unwrap();
        let charged = adm.used_bytes();
        assert_eq!(charged, GIB + 2 * 1024 * 1024);
        t.close(&mut pool, &mut adm, a);
        assert_eq!(adm.used_bytes(), charged, "b and c are still resident");
    }

    #[test]
    fn a_cold_session_returns_its_admission() {
        let (mut pool, mut adm, mut t) = rig(1);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.mark_cold(&mut pool, &mut adm, a);
        assert!(!t.is_admitted(a));
        assert_eq!(adm.used_bytes(), 0);
        t.close(&mut pool, &mut adm, a);
        assert_eq!(adm.used_bytes(), 0);
    }

    #[test]
    fn readmit_charges_a_restored_session_once() {
        let (mut pool, mut adm, mut t) = rig(1);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.mark_swapped(&mut pool, &mut adm, a);
        assert_eq!(adm.used_bytes(), 0);
        t.readmit(&mut adm, a).unwrap();
        t.readmit(&mut adm, a).unwrap(); // idempotent
        assert_eq!(adm.used_bytes(), GIB + 1024 * 1024);
        let slot = pool.acquire().unwrap();
        t.mark_resident(a, slot, 0);
        t.close(&mut pool, &mut adm, a);
        assert_eq!(adm.used_bytes(), 0);
    }

    #[test]
    fn readmit_refused_by_budget_leaves_the_session_unadmitted() {
        let (mut pool, mut adm, mut t) = rig_budget(2, 1024, 1);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.mark_swapped(&mut pool, &mut adm, a);
        let _b = t.open(&mut pool, &mut adm, 1024).unwrap();
        assert!(t.readmit(&mut adm, a).is_err());
        assert!(!t.is_admitted(a));
        t.close(&mut pool, &mut adm, a);
        assert_eq!(adm.used_bytes(), GIB + 1024 * 1024, "b keeps its charge");
    }

    #[test]
    fn a_closed_session_id_is_not_reusable_by_accident() {
        let (mut pool, mut adm, mut t) = rig(1);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.close(&mut pool, &mut adm, a);
        assert!(t.get(a).is_none(), "a closed session must not resolve");
    }
}
