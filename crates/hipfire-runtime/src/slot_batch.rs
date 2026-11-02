// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// SlotBatch — one forward step's ragged work across N slots.
//
// A step mixes freely: one slot verifying 8 draft tokens, another
// chunk-prefilling 256, others decoding 1 each. That raggedness is what SP1's
// kernels were built for.

use crate::scheduler::{
    is_runnable_decode, is_runnable_prefill, vl_waiting, PendingWork, Scheduler, SpecKind,
};
use rdna_compute::slot_pool::SlotId;

#[derive(Debug, Clone, Default)]
pub struct SlotBatch {
    /// Per-slot token counts for this step. 0 means the slot is idle.
    pub m_per_slot: Vec<usize>,
    /// Flat token ids, packed across slots in slot order.
    pub tokens: Vec<u32>,
    /// Per-row ABSOLUTE position within that row's own slot.
    ///
    /// Authoritative for the causal bound — never `desc.seq_len`. The two
    /// differ whenever a slot has more than one query row, and conflating them
    /// caused SP1's only Critical defect.
    pub positions: Vec<i32>,
    /// Slot index for each flat row.
    pub row_slot: Vec<i32>,
    /// Per-row absolute M-RoPE phases `[t, h, w]`, aligned with the flat rows.
    ///
    /// Empty = the step runs 1D RoPE off `positions`. When non-empty it MUST
    /// have one entry per row and the forward dispatches the batched M-RoPE
    /// kernel instead. Rows of plain text slots carry `[p, p, p]` (p = the
    /// row's position), which the M-RoPE kernel reduces to bit-identical
    /// angles — so a mixed text+VL step can run the one kernel for everyone.
    /// KV addressing and causal bounds still read `positions`; only the RoPE
    /// phase differs.
    pub pos3: Vec<[i32; 3]>,
    /// Per-row index into that row's slot's external-embedding matrix (the
    /// vision tower's output for the request), or -1 to use the token
    /// embedding table. Empty = no external rows this step. Only VL prefill
    /// rows carry a non-negative index, and they must appear in prompt order
    /// so indices line up with the request's visual token stream.
    pub ext_emb: Vec<i32>,
}

impl SlotBatch {
    /// Build a step from `(slot, tokens, start_pos)` triples. Slots with no
    /// tokens contribute no rows.
    pub fn build(per_slot: &[(SlotId, &[u32], usize)]) -> Self {
        let mut b = SlotBatch::default();
        for (slot, toks, start_pos) in per_slot {
            b.m_per_slot.push(toks.len());
            for (i, t) in toks.iter().enumerate() {
                b.tokens.push(*t);
                b.positions.push((start_pos + i) as i32);
                b.row_slot.push(slot.0 as i32);
            }
        }
        b
    }

    pub fn total_rows(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

// ---- §4.2 frozen cross-request step contract ----------------------------
//
// Family-neutral, opaque plan types shared by the runtime planner, the arch
// VMM executor, and the engine/daemon driver. The engine keeps its own
// `AttemptKey`/`LaneTicket` registry and maps each `RequestEpoch` to it; this
// crate never imports engine or generation types.

/// Opaque request identity for one admitted owner. `request_tag` maps to the
/// engine's AttemptKey/admission/LaneTicket registry; `owner_generation`
/// bumps on every reuse of the same request state so a stale commit can
/// never release or advance a newer owner's buffers. Generation 0 is never
/// admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestEpoch {
    pub request_tag: u64,
    pub owner_generation: u64,
}

impl RequestEpoch {
    /// Generation 0 marks an empty/unadmitted slot.
    pub const fn is_admitted(&self) -> bool {
        self.owner_generation != 0
    }
}

/// Contiguous flat-row range `[begin, begin+len)` within one step's `SlotBatch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowRange {
    pub begin: usize,
    pub len: usize,
}

impl RowRange {
    pub const fn end(&self) -> usize {
        self.begin + self.len
    }
}

/// What a request's rows do this step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestStepKind {
    /// One ordinary autoregressive decode row (the pending seed).
    Ar,
    /// Speculative verify: `[seed, draft_1..draft_len]`, `draft_len + 1` rows.
    /// Only the seed is host-known at plan time: rows past it hold host-only
    /// seed placeholders that must never be executed or captured. The
    /// executor replaces every Verify range with its epoch-tagged
    /// `[seed, candidates…]` (exactly `draft_len + 1`) before provision and
    /// fails closed if that draft is absent or stale.
    Verify { draft_len: usize },
    /// Prompt chunk prefill.
    Prefill,
    /// Forced/head-fill rows the executor must run (no sampling).
    Forced,
}

/// One admitted request's rows in a step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestRows {
    pub epoch: RequestEpoch,
    pub rows: RowRange,
    pub kind: RequestStepKind,
}

/// One planned step. `requests` is in stable slot order and lists only
/// admitted owners; each `RowRange` indexes `batch`'s flat rows, and
/// `batch.positions` stays the per-request absolute causal authority.
#[derive(Debug, Clone, Default)]
pub struct BatchStepPlan {
    pub batch: SlotBatch,
    pub requests: Vec<RequestRows>,
    pub decode_rows: usize,
    pub prefill_rows: usize,
    pub verify_rows: usize,
    pub forced_rows: usize,
}

impl BatchStepPlan {
    pub fn total_rows(&self) -> usize {
        self.batch.total_rows()
    }
}

/// Executor result of one forward: identifies the completed step and the
/// packed per-row target picks, indexed by flat row. Device hidden/tapes
/// stay executor-owned until the matching commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepOutput {
    pub step_id: u64,
    pub target_picks: Vec<u32>,
}

/// Per-request committed result of one step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestAdvance {
    pub epoch: RequestEpoch,
    /// Newly committed token ids (accepted drafts + bonus, or the AR pick).
    pub committed_ids: Vec<u32>,
    /// Absolute position after this commit (next row's position).
    pub committed_position: usize,
    pub accepted_drafts: usize,
    pub verified_rows: usize,
    /// Terminal reason when the request finished inside this step.
    pub finish: Option<String>,
}

/// Step planner over the existing [`Scheduler`] row allocation (decode
/// first, rotating prefill quantum, per-request chunk maximum). Unlike
/// `Scheduler::next_batch_eligible` it never drains `PendingWork`:
/// `plan_step` is pure apart from holding the provisional rotation cursor,
/// and [`BatchPlanner::publish`] applies a plan only after its step commits.
pub struct BatchPlanner {
    pub scheduler: Scheduler,
    /// Verify draft length for a decoding `SpecKind::Mtp` request (K).
    pub mtp_draft_len: usize,
    /// Verify draft length for a decoding `SpecKind::Dflash` request
    /// (block − 1).
    pub dflash_draft_len: usize,
    /// Per-`work`-index forced prefill chunk length (empty = scheduler
    /// allocation). `Some(n)`: that request's prefill takes exactly `n` rows
    /// or none this step (an executor whose arithmetic depends on chunk
    /// boundaries, e.g. the exact VMM route, sets this every step).
    pub forced_prefill_len: Vec<Option<usize>>,
    pending_cursor: Option<usize>,
}

impl BatchPlanner {
    pub fn new(scheduler: Scheduler, mtp_draft_len: usize, dflash_draft_len: usize) -> Self {
        Self {
            scheduler,
            mtp_draft_len,
            dflash_draft_len,
            forced_prefill_len: Vec::new(),
            pending_cursor: None,
        }
    }

    fn draft_len(&self, spec: SpecKind) -> usize {
        match spec {
            SpecKind::None => 0,
            SpecKind::Mtp => self.mtp_draft_len,
            SpecKind::Dflash => self.dflash_draft_len,
        }
    }

    /// Plan one step. `epochs`/`eligible` are aligned with `work`. Verify
    /// lanes (decoding spec requests) are admitted first in slot order, but
    /// never into the rows needed for the AR decode lanes plus one positive
    /// prefill quantum, so verify work cannot starve prefill. Remaining
    /// budget goes to the `Scheduler` allocation. No `PendingWork` mutation.
    pub fn plan_step(
        &mut self,
        work: &[PendingWork],
        epochs: &[RequestEpoch],
        eligible: &[bool],
        row_budget: usize,
        prefill_min_tokens: usize,
    ) -> Result<BatchStepPlan, String> {
        let n = work.len();
        if epochs.len() != n || eligible.len() != n {
            return Err(format!(
                "plan_step: work={n} epochs={} eligible={} must align",
                epochs.len(),
                eligible.len()
            ));
        }
        let vl_seq = self.scheduler.vl_sequential;
        let admitted: Vec<bool> = (0..n)
            .map(|i| eligible[i] && epochs[i].is_admitted())
            .collect();
        for i in 0..n {
            if admitted[i] && work[i].vl_prefill.is_some() && vl_seq {
                return Err(format!(
                    "plan_step: slot {} is owned by the sequential VL path",
                    work[i].slot.0
                ));
            }
        }

        // Reserve AR decode rows and one prefill quantum ahead of verify.
        let ar_lanes = (0..n)
            .filter(|&i| admitted[i] && is_runnable_decode(&work[i], vl_seq))
            .count();
        let prefill_reserve = (0..n)
            .filter(|&i| admitted[i] && is_runnable_prefill(&work[i], vl_seq))
            .map(|i| {
                prefill_min_tokens
                    .max(1)
                    .min(work[i].remaining_prompt.len())
            })
            .max()
            .unwrap_or(0);
        let verify_cap = row_budget.saturating_sub(ar_lanes + prefill_reserve);

        let mut verify = vec![0usize; n];
        let mut verify_used = 0usize;
        for i in 0..n {
            let w = &work[i];
            if !admitted[i] || !(w.spec.active() && w.decoding) || vl_waiting(w) {
                continue;
            }
            if w.remaining_prompt.len() != 1 {
                return Err(format!(
                    "plan_step: verify slot {} must carry exactly one pending seed, has {}",
                    w.slot.0,
                    w.remaining_prompt.len()
                ));
            }
            let rows = self.draft_len(w.spec) + 1;
            if verify_used + rows > verify_cap {
                continue;
            }
            verify[i] = rows;
            verify_used += rows;
        }

        let mut alloc = self.scheduler.allocate_rows(
            work,
            row_budget - verify_used,
            prefill_min_tokens,
            &admitted,
        );
        if !self.forced_prefill_len.is_empty() {
            // Re-allocate prefill rows: decode rows stay, each forced
            // request gets exactly its length if it fits, rotating from the
            // cursor so two prefills alternate.
            let mut avail = row_budget - verify_used;
            for i in 0..n {
                if alloc.alloc[i] > 0 && work[i].decoding {
                    avail -= alloc.alloc[i];
                } else {
                    alloc.alloc[i] = 0;
                }
            }
            let pf: Vec<usize> = (0..n)
                .filter(|&i| admitted[i] && is_runnable_prefill(&work[i], vl_seq))
                .collect();
            let start = if pf.is_empty() {
                0
            } else {
                self.scheduler.prefill_cursor % pf.len()
            };
            for k in 0..pf.len() {
                let i = pf[(start + k) % pf.len()];
                let want = match self.forced_prefill_len.get(i).copied().flatten() {
                    Some(w) => w.min(work[i].remaining_prompt.len()),
                    None => continue,
                };
                if want > 0 && want <= avail {
                    alloc.alloc[i] = want;
                    avail -= want;
                }
            }
            alloc.next_prefill_cursor =
                (!pf.is_empty()).then(|| (self.scheduler.prefill_cursor + 1) % pf.len());
        }
        let any_pos3 = self.scheduler.any_pos3(work);
        let mut plan = BatchStepPlan::default();
        plan.batch.m_per_slot = vec![0; n];
        for i in 0..n {
            let w = &work[i];
            let (take, kind) = if verify[i] > 0 {
                (
                    verify[i],
                    RequestStepKind::Verify {
                        draft_len: verify[i] - 1,
                    },
                )
            } else if alloc.alloc[i] > 0 {
                let k = if w.decoding {
                    RequestStepKind::Ar
                } else {
                    RequestStepKind::Prefill
                };
                (alloc.alloc[i], k)
            } else {
                continue;
            };
            let begin = plan.batch.tokens.len();
            let mut visual_idx = w.vl_prefill.as_ref().map_or(0, |vl| vl.visual_idx);
            for j in 0..take {
                // Verify draft rows past the seed carry the seed as a host
                // placeholder; the executor owns the real draft ids.
                let t = w.remaining_prompt[j.min(w.remaining_prompt.len() - 1)];
                let pos = w.next_pos + j;
                plan.batch.tokens.push(t);
                plan.batch.positions.push(pos as i32);
                plan.batch.row_slot.push(w.slot.0 as i32);
                if !any_pos3 {
                    continue;
                }
                match w.vl_prefill.as_ref() {
                    Some(vl) => {
                        plan.batch.pos3.push(vl.pos3(pos));
                        if kind == RequestStepKind::Prefill
                            && t == vl.image_pad_id
                            && visual_idx < vl.n_visual_tokens
                        {
                            plan.batch.ext_emb.push(visual_idx as i32);
                            visual_idx += 1;
                        } else {
                            plan.batch.ext_emb.push(-1);
                        }
                    }
                    None => {
                        plan.batch.pos3.push([pos as i32 + w.pos3_delta; 3]);
                        plan.batch.ext_emb.push(-1);
                    }
                }
            }
            plan.batch.m_per_slot[i] = take;
            match kind {
                RequestStepKind::Ar => plan.decode_rows += take,
                RequestStepKind::Prefill => plan.prefill_rows += take,
                RequestStepKind::Verify { .. } => plan.verify_rows += take,
                RequestStepKind::Forced => plan.forced_rows += take,
            }
            plan.requests.push(RequestRows {
                epoch: epochs[i],
                rows: RowRange { begin, len: take },
                kind,
            });
        }
        self.pending_cursor = alloc.next_prefill_cursor;
        Ok(plan)
    }

    /// Publish a committed step: drain consumed prefill tokens and advance
    /// positions, apply each request's committed advance (the next pending
    /// seed is the last committed id), and publish the provisional rotation
    /// cursor. Call only after `commit_step` succeeded; on a failed step call
    /// [`BatchPlanner::discard`] instead and `work` stays untouched.
    /// `work_epochs` must be the aligned epochs passed to `plan_step`.
    pub fn publish(
        &mut self,
        work: &mut [PendingWork],
        work_epochs: &[RequestEpoch],
        plan: &BatchStepPlan,
        advances: &[RequestAdvance],
    ) -> Result<(), String> {
        if work_epochs.len() != work.len() {
            return Err("publish: epochs must align with work".into());
        }
        // Validate the whole commit before mutating anything: every planned
        // owner must still be admitted at the same generation, prefill takes
        // must fit, and every non-prefill request needs exactly its own
        // advance. Advances for unplanned epochs are rejected.
        for rr in &plan.requests {
            if !rr.epoch.is_admitted() {
                return Err(format!("publish: unadmitted epoch {:?}", rr.epoch));
            }
            let i = work_epochs
                .iter()
                .position(|x| *x == rr.epoch)
                .ok_or_else(|| format!("publish: stale or unknown epoch {:?}", rr.epoch))?;
            let n_adv = advances.iter().filter(|a| a.epoch == rr.epoch).count();
            match rr.kind {
                RequestStepKind::Prefill => {
                    // A prompt-completing chunk carries exactly one advance
                    // (the first sampled token); earlier chunks carry none.
                    let left = work[i].remaining_prompt.len();
                    let completes = left == rr.rows.len;
                    if left < rr.rows.len || n_adv > usize::from(completes) {
                        return Err(format!(
                            "publish: invalid prefill commit for {:?}",
                            rr.epoch
                        ));
                    }
                    if let Some(a) = advances.iter().find(|a| a.epoch == rr.epoch) {
                        if a.committed_position != work[i].next_pos + rr.rows.len {
                            return Err(format!(
                                "publish: prefill advance position {} != {}",
                                a.committed_position,
                                work[i].next_pos + rr.rows.len
                            ));
                        }
                    }
                }
                _ => {
                    if n_adv != 1 {
                        return Err(format!("publish: {n_adv} advances for {:?}", rr.epoch));
                    }
                }
            }
        }
        if let Some(a) = advances
            .iter()
            .find(|a| !plan.requests.iter().any(|r| r.epoch == a.epoch))
        {
            return Err(format!(
                "publish: advance for unplanned epoch {:?}",
                a.epoch
            ));
        }
        for rr in &plan.requests {
            let i = work_epochs
                .iter()
                .position(|x| *x == rr.epoch)
                .ok_or_else(|| format!("publish: stale or unknown epoch {:?}", rr.epoch))?;
            let w = &mut work[i];
            match rr.kind {
                RequestStepKind::Prefill => {
                    if w.remaining_prompt.len() < rr.rows.len {
                        return Err(format!("publish: slot {} prompt underflow", w.slot.0));
                    }
                    if let Some(vl) = w.vl_prefill.as_mut() {
                        let pads = w.remaining_prompt[..rr.rows.len]
                            .iter()
                            .filter(|&&t| t == vl.image_pad_id)
                            .count();
                        vl.visual_idx = (vl.visual_idx + pads).min(vl.n_visual_tokens);
                    }
                    w.remaining_prompt.drain(..rr.rows.len);
                    w.next_pos += rr.rows.len;
                    if let Some(adv) = advances.iter().find(|a| a.epoch == rr.epoch) {
                        w.decoding = true;
                        if adv.finish.is_none() {
                            if let Some(&seed) = adv.committed_ids.last() {
                                w.remaining_prompt.push(seed);
                            }
                        }
                    }
                }
                RequestStepKind::Ar | RequestStepKind::Verify { .. } | RequestStepKind::Forced => {
                    let adv = advances
                        .iter()
                        .find(|a| a.epoch == rr.epoch)
                        .ok_or_else(|| format!("publish: no advance for {:?}", rr.epoch))?;
                    w.next_pos = adv.committed_position;
                    w.remaining_prompt.clear();
                    if adv.finish.is_none() {
                        if let Some(&seed) = adv.committed_ids.last() {
                            w.remaining_prompt.push(seed);
                        }
                    }
                }
            }
        }
        if let Some(c) = self.pending_cursor.take() {
            self.scheduler.prefill_cursor = c;
        }
        Ok(())
    }

    /// Drop a failed/aborted step's provisional cursor; `work` is untouched.
    pub fn discard(&mut self) {
        self.pending_cursor = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdna_compute::slot_pool::SlotId;

    #[test]
    fn packs_tokens_in_slot_order() {
        let b = SlotBatch::build(&[
            (SlotId(0), &[10u32, 11][..], 100),
            (SlotId(1), &[20u32][..], 5),
        ]);
        assert_eq!(b.tokens, vec![10, 11, 20]);
        assert_eq!(b.m_per_slot, vec![2, 1]);
        assert_eq!(b.total_rows(), 3);
    }

    #[test]
    fn positions_advance_within_a_slot_from_its_own_start() {
        // Slot 0 verifying 3 tokens at start_pos 100 occupies 100,101,102.
        // Slot 1 decoding 1 token at start_pos 5 occupies 5. They are
        // independent -- positions are per-slot absolute, not batch-global.
        let b = SlotBatch::build(&[
            (SlotId(0), &[1u32, 2, 3][..], 100),
            (SlotId(1), &[9u32][..], 5),
        ]);
        assert_eq!(b.positions, vec![100, 101, 102, 5]);
    }

    #[test]
    fn row_slot_maps_every_flat_row_to_its_slot() {
        let b = SlotBatch::build(&[
            (SlotId(0), &[1u32, 2, 3][..], 0),
            (SlotId(2), &[7u32][..], 0),
        ]);
        assert_eq!(b.row_slot, vec![0, 0, 0, 2]);
    }

    #[test]
    fn idle_slots_contribute_no_rows() {
        let b = SlotBatch::build(&[(SlotId(0), &[][..], 0), (SlotId(1), &[5u32][..], 42)]);
        assert_eq!(b.m_per_slot, vec![0, 1]);
        assert_eq!(b.tokens, vec![5]);
        assert_eq!(b.positions, vec![42]);
        assert_eq!(b.row_slot, vec![1]);
    }

    #[test]
    fn an_all_idle_batch_is_empty() {
        let b = SlotBatch::build(&[(SlotId(0), &[][..], 0)]);
        assert!(b.is_empty());
        assert_eq!(b.total_rows(), 0);
    }

    #[test]
    fn mixed_prefill_and_decode_is_the_shape_this_exists_for() {
        // slot 0 verifies 8 draft tokens, slot 1 chunk-prefills 256,
        // slots 2-3 decode 1 each.
        let p0: Vec<u32> = (0..8).collect();
        let p1: Vec<u32> = (0..256).collect();
        let b = SlotBatch::build(&[
            (SlotId(0), &p0[..], 1000),
            (SlotId(1), &p1[..], 0),
            (SlotId(2), &[1u32][..], 50),
            (SlotId(3), &[2u32][..], 77),
        ]);
        assert_eq!(b.total_rows(), 266);
        assert_eq!(b.row_slot.iter().filter(|&&s| s == 1).count(), 256);
        assert_eq!(b.positions[b.positions.len() - 1], 77);
    }

    fn pw(
        slot: usize,
        prompt: Vec<u32>,
        next_pos: usize,
        decoding: bool,
        spec: SpecKind,
    ) -> PendingWork {
        PendingWork {
            slot: SlotId(slot),
            remaining_prompt: prompt,
            next_pos,
            decoding,
            vl_prefill: None,
            spec,
            spec_cycles: 0,
            spec_committed: 0,
            spec_retire_fails: 0,
            pos3_delta: 0,
        }
    }

    fn ep(tag: u64) -> RequestEpoch {
        RequestEpoch {
            request_tag: tag,
            owner_generation: 1,
        }
    }

    fn planner() -> BatchPlanner {
        BatchPlanner::new(
            Scheduler {
                chunk_size: 64,
                vl_sequential: false,
                prefill_cursor: 0,
            },
            3,
            15,
        )
    }

    #[test]
    fn plan_step_mixes_verify_ar_prefill_without_draining() {
        let work = vec![
            pw(0, vec![7], 100, true, SpecKind::Mtp),
            pw(1, vec![9], 50, true, SpecKind::None),
            pw(2, (0..40).collect(), 0, false, SpecKind::None),
        ];
        let epochs = [ep(1), ep(2), ep(3)];
        let mut p = planner();
        let plan = p.plan_step(&work, &epochs, &[true; 3], 128, 8).unwrap();
        assert_eq!(plan.verify_rows, 4);
        assert_eq!(plan.decode_rows, 1);
        assert_eq!(plan.prefill_rows, 40);
        assert_eq!(
            plan.requests[0].kind,
            RequestStepKind::Verify { draft_len: 3 }
        );
        assert_eq!(plan.requests[0].rows, RowRange { begin: 0, len: 4 });
        assert_eq!(&plan.batch.positions[..4], &[100, 101, 102, 103]);
        assert_eq!(plan.requests[2].rows, RowRange { begin: 5, len: 40 });
        // plan_step is side-effect free on work.
        assert_eq!(work[2].remaining_prompt.len(), 40);
        assert_eq!(p.scheduler.prefill_cursor, 0);
    }

    #[test]
    fn verify_never_consumes_the_prefill_quantum() {
        let work = vec![
            pw(0, vec![7], 10, true, SpecKind::Dflash),
            pw(1, (0..40).collect(), 0, false, SpecKind::None),
        ];
        let mut p = planner();
        // 16 verify rows + 8 quantum > 20 budget: verify lane waits.
        let plan = p
            .plan_step(&work, &[ep(1), ep(2)], &[true; 2], 20, 8)
            .unwrap();
        assert_eq!(plan.verify_rows, 0);
        assert_eq!(plan.prefill_rows, 20);
    }

    #[test]
    fn unadmitted_epochs_contribute_no_rows() {
        let work = vec![pw(0, vec![1], 0, true, SpecKind::None)];
        let mut p = planner();
        let e = RequestEpoch {
            request_tag: 1,
            owner_generation: 0,
        };
        let plan = p.plan_step(&work, &[e], &[true], 8, 1).unwrap();
        assert!(plan.requests.is_empty());
    }

    #[test]
    fn publish_applies_commit_and_rejects_stale_epoch() {
        let mut work = vec![
            pw(0, vec![9], 50, true, SpecKind::None),
            pw(1, (0..10).collect(), 0, false, SpecKind::None),
        ];
        let epochs = [ep(1), ep(2)];
        let mut p = planner();
        let plan = p.plan_step(&work, &epochs, &[true; 2], 64, 1).unwrap();
        let adv = RequestAdvance {
            epoch: ep(1),
            committed_ids: vec![42],
            committed_position: 51,
            accepted_drafts: 0,
            verified_rows: 1,
            finish: None,
        };
        let stale = [
            RequestEpoch {
                request_tag: 1,
                owner_generation: 2,
            },
            ep(2),
        ];
        assert!(p.publish(&mut work, &stale, &plan, &[adv.clone()]).is_err());
        p.publish(&mut work, &epochs, &plan, &[adv]).unwrap();
        assert_eq!(work[0].remaining_prompt, vec![42]);
        assert_eq!(work[0].next_pos, 51);
        assert!(work[1].remaining_prompt.is_empty());
        assert_eq!(work[1].next_pos, 10);
    }

    #[test]
    fn completing_prefill_publishes_first_token_and_turns_decoding() {
        let mut work = vec![pw(0, (0..10).collect(), 0, false, SpecKind::None)];
        let epochs = [ep(1)];
        let mut p = planner();
        let plan = p.plan_step(&work, &epochs, &[true], 64, 1).unwrap();
        let mut adv = RequestAdvance {
            epoch: ep(1),
            committed_ids: vec![77],
            committed_position: 9,
            accepted_drafts: 0,
            verified_rows: 0,
            finish: None,
        };
        assert!(p
            .publish(&mut work, &epochs, &plan, &[adv.clone()])
            .is_err());
        assert_eq!(work[0].remaining_prompt.len(), 10);
        adv.committed_position = 10;
        p.publish(&mut work, &epochs, &plan, &[adv]).unwrap();
        assert!(work[0].decoding);
        assert_eq!(work[0].remaining_prompt, vec![77]);
    }
}
