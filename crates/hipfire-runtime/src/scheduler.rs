// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Scheduler — decides what goes into each step's SlotBatch.
//
// Deliberately minimal: round-robin, chunked prefill mixed with decode, no
// preemption or priorities. Batching wins ~1.36x at 8 slots and recovers
// under a tenth of the available bandwidth headroom, so contorting the
// scheduler to maximise batch width is not where the remaining performance
// is. Correctness and not blocking other slots matter more.
//
// The one property that matters: a long prompt is chunked to `chunk_size`
// tokens per step so it cannot block other slots (in particular decoding
// slots) for its whole duration. See `prefill_and_decode_mix_in_one_batch`.

use crate::slot_batch::SlotBatch;
use rdna_compute::page_pool::PAGE_TOKENS;
use rdna_compute::slot_pool::SlotId;

/// Trim a prompt-completing prefill round to end at the last page boundary
/// (spec §4.5): a resumable recurrent-state checkpoint can only be captured
/// when the round ends exactly at a published page boundary, so a round that
/// would finish a prompt mid-page is trimmed to the boundary and the sub-page
/// tail runs as its own final round. Every prompt longer than one page then
/// produces a resumable checkpoint at `floor(len / PAGE_TOKENS) * PAGE_TOKENS`.
///
/// No effect on page-aligned prompts, sub-page prompts, or non-completing
/// takes; an operator quantum above one page (`prefill_min_tokens`) wins —
/// no trim — because alignment must not silently undercut the configured
/// service floor.
fn page_align_take(
    take: usize,
    take_start_pos: usize,
    remaining_incl_take: usize,
    prefill_min: usize,
) -> usize {
    if take >= remaining_incl_take && remaining_incl_take > 0 && prefill_min <= PAGE_TOKENS {
        let end = take_start_pos + remaining_incl_take;
        if end % PAGE_TOKENS != 0 && end > PAGE_TOKENS {
            let boundary = (end / PAGE_TOKENS) * PAGE_TOKENS;
            if boundary > take_start_pos {
                return boundary - take_start_pos;
            }
        }
    }
    take
}

pub struct Scheduler {
    pub chunk_size: usize,
    /// Serve VL slots on the legacy sequential per-token M-RoPE path instead
    /// of the batched paged-capable one. Opt-in via HIPFIRE_VL_SEQUENTIAL=1;
    /// the batched path is the default (and the only one compatible with a
    /// paged pool).
    pub vl_sequential: bool,
    /// Rotation cursor for the prefill quantum (spec §5.2 S2). Each step the
    /// first prefilling slot in rotation order gets the minimum prefill
    /// quantum before extra budget is distributed round-robin; advancing this
    /// cursor across ticks hands the quantum to a different prefilling slot
    /// so two long prefills interleave instead of one blocking the other.
    pub prefill_cursor: usize,
}

/// Which speculative mechanism owns a slot's decode loop. `None` = plain AR.
/// MTP and DFlash2 are mutually exclusive per load (the engine loads at most
/// one drafter), so a slot never carries both.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SpecKind {
    #[default]
    None,
    /// In-trunk MTP head (K+1 verify rows per cycle).
    Mtp,
    /// DFlash2 candidate-selector draft (B verify rows per cycle).
    Dflash,
}

impl SpecKind {
    /// True while this slot's decode is owned by a spec draft/verify cycle.
    pub fn active(self) -> bool {
        !matches!(self, SpecKind::None)
    }
}

/// One slot's outstanding work as seen by the scheduler.
pub struct PendingWork {
    pub slot: SlotId,
    /// Prompt tokens not yet fed through prefill. Drained front-to-back as
    /// chunks are taken.
    pub remaining_prompt: Vec<u32>,
    /// Next absolute position this slot will occupy.
    pub next_pos: usize,
    /// Once prefill is complete, the slot decodes one token per step.
    pub decoding: bool,
    /// VL state. When present, this slot is served by the sequential
    /// M-RoPE path (vision_forward + per-token prefill/decode) and MUST
    /// be skipped by the batched 1D-RoPE scheduler.
    pub vl_prefill: Option<VlPrefill>,

    /// Speculative-decode ownership. `Mtp`/`Dflash` while prefilling mean the
    /// slot's prompt chunks still flow through the batched scheduler (the
    /// speculator's private state is filled post-forward); once `decoding`,
    /// the slot is owned by the spec draft/verify cycle and MUST be skipped
    /// by the batched scheduler — its verify rows are injected separately.
    pub spec: SpecKind,
    /// Rolling adaptive-retire window: decode cycles spent under speculation
    /// and tokens committed across them. When the window's mean advance stays
    /// below `MTP_RETIRE_MIN_ADVANCE`, the engine retires the slot to plain
    /// AR decode (a spec cycle costs ~2x an AR step, so a drafter that cannot
    /// beat that loses wall-clock — the genre-conditional trap).
    pub spec_cycles: usize,
    pub spec_committed: usize,
    /// Consecutive retire-windows whose mean advance fell below the line.
    pub spec_retire_fails: usize,
    /// M-RoPE phase offset for a TEXT continuation whose conversation
    /// carries image-turn KV (the session's `rope_delta`; 0 = pure text).
    ///
    /// KV rows are addressed by token index, but the image turn's stored
    /// keys carry compressed-grid phases: every row this slot prefills or
    /// decodes must take its phase from `pos + rope_delta`, not from the
    /// raw token index, or cross-turn attention re-rotates by the whole
    /// delta. Rows still emit `ext_emb = -1` — this is a phase fix, not an
    /// embedding splice.
    pub pos3_delta: i32,
}

/// Visual + M-RoPE data for one slot's VL request.
///
/// On the default batched path the scheduler splices vision embeddings (via
/// per-row `SlotBatch::ext_emb` indices) and M-RoPE phases (via
/// `SlotBatch::pos3`) into the chunked batch like any other slot. The
/// sequential fallback (`vl_sequential`) instead owns the slot for the whole
/// request and prefills/decodes per token.
pub struct VlPrefill {
    /// Vision-tower patches; consumed once by `vision_forward`.
    pub patches: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
    /// Post-merge visual token count.
    pub n_visual_tokens: usize,
    /// Filled by `vision_forward` (`n_visual_tokens * dim` floats).
    pub embeddings: Vec<f32>,
    /// Model hidden dim (embedding stride).
    pub dim: usize,
    /// Index into `embeddings` for the next visual token to splice.
    pub visual_idx: usize,
    /// Prompt token id that carries one visual embedding per occurrence.
    pub image_pad_id: u32,
    /// 3D M-RoPE positions for every prompt token, offset by base.
    pub mrope_positions: Vec<[i32; 3]>,
    /// rope_delta for decode positions past the prompt.
    pub rope_delta: i32,
    /// Base sequence position for this request's prefill.
    pub base: usize,
}

impl VlPrefill {
    /// Absolute rope phases for position `pos`: this request's prompt table
    /// while in range, `pos + rope_delta` beyond it. Same rule as
    /// `MropeCtx::pos3` on the sequential path.
    pub fn pos3(&self, pos: usize) -> [i32; 3] {
        match pos
            .checked_sub(self.base)
            .and_then(|i| self.mrope_positions.get(i))
        {
            Some(p) => *p,
            None => [pos as i32 + self.rope_delta; 3],
        }
    }
}

impl Scheduler {
    /// Build the next step's `SlotBatch` under a global trunk-row budget
    /// (spec §5.2 S2). `remaining_rows` is the row budget for this step's
    /// prefill+decode work — the caller subtracts MTP verify rows (injected
    /// later) from `max_batch_tokens` and passes the rest here, so the
    /// returned batch's `prefill_rows + decode_rows` never exceeds it and the
    /// full step (prefill + decode + verify) never exceeds `max_batch_tokens`.
    ///
    /// Allocation order (spec §5.2 S2): every runnable decode slot gets 1 row
    /// first; if that would exceed the budget, fewer decode lanes are admitted.
    /// Then, if any prefill is runnable and space remains, at least
    /// `prefill_min_tokens` rows go to one prefilling slot — the slot chosen
    /// by the persistent `prefill_cursor` so the quantum rotates across ticks.
    /// Unused budget adds extra prefill rows by rotation, still capping each
    /// slot at `chunk_size` (a per-request MAXIMUM, not the total budget).
    /// Advances `next_pos` (and drains `remaining_prompt`) for whatever it
    /// takes.
    pub fn next_batch(
        &mut self,
        work: &mut [PendingWork],
        remaining_rows: usize,
        prefill_min_tokens: usize,
    ) -> SlotBatch {
        // No FairQueue in scope (unit tests, single-slot carriers): every
        // slot is eligible. The serve engine calls `next_batch_eligible`
        // directly with the FairQueue's grant mask (spec §5.3 S3).
        let all = vec![true; work.len()];
        self.next_batch_eligible(work, remaining_rows, prefill_min_tokens, &all)
    }

    /// Same as [`next_batch`](Self::next_batch) but only slots `s` with
    /// `eligible[s] == true` may contribute rows. An un-granted slot
    /// contributes 0 rows and its `remaining_prompt` is left untouched (it
    /// is NOT drained) so no tokens are lost. The serve engine uses this to
    /// enforce FairQueue grants (spec §5.3 S3): the FairQueue decides WHO is
    /// eligible this tick; this scheduler still decides HOW MANY rows each
    /// eligible slot gets under `remaining_rows`.
    pub fn next_batch_eligible(
        &mut self,
        work: &mut [PendingWork],
        remaining_rows: usize,
        prefill_min_tokens: usize,
        eligible: &[bool],
    ) -> SlotBatch {
        let n = work.len();
        let plan = self.allocate_rows(work, remaining_rows, prefill_min_tokens, eligible);
        if let Some(c) = plan.next_prefill_cursor {
            self.prefill_cursor = c;
        }
        let alloc = plan.alloc;
        // A step carrying VL rows runs the batched M-RoPE kernel for EVERY
        // row (text rows take [p, p, p], bit-identical to 1D RoPE), so the
        // per-row pos3/ext side arrays exist whenever some slot is VL — or
        // carries a continuation rope delta, whose rows need the shifted
        // phases for the same kernel.
        let any_pos3 = self.any_pos3(work);
        let mut b = SlotBatch::default();
        b.m_per_slot = vec![0; n];

        // ---- Build the batch in slot order from the allocation ----
        // Flat arrays are packed in slot order (the forward reads them via
        // `m_per_slot` offsets), so the row-building pass runs in slot order
        // even though the allocation was phased decode-then-prefill.
        for i in 0..n {
            let take = alloc[i];
            if take == 0 {
                continue;
            }
            let w = &mut work[i];
            let toks: Vec<u32> = w.remaining_prompt.drain(..take).collect();
            let start_pos = w.next_pos;
            b.m_per_slot[i] = toks.len();
            for (j, t) in toks.iter().enumerate() {
                let pos = start_pos + j;
                b.tokens.push(*t);
                b.positions.push(pos as i32);
                b.row_slot.push(w.slot.0 as i32);
                if !any_pos3 {
                    continue;
                }
                match w.vl_prefill.as_mut() {
                    Some(vl) => {
                        b.pos3.push(vl.pos3(pos));
                        // Image pads splice visual embeddings in prompt
                        // order; everything else uses the token table.
                        if *t == vl.image_pad_id && vl.visual_idx < vl.n_visual_tokens {
                            b.ext_emb.push(vl.visual_idx as i32);
                            vl.visual_idx += 1;
                        } else {
                            b.ext_emb.push(-1);
                        }
                    }
                    None => {
                        // Text row: [p, p, p] is bit-identical to 1D RoPE;
                        // a continuation of an image conversation shifts the
                        // phase by its session's rope delta instead.
                        b.pos3.push([pos as i32 + w.pos3_delta; 3]);
                        b.ext_emb.push(-1);
                    }
                }
            }
            w.next_pos += toks.len();
        }
        b
    }

    /// True when a step over `work` must carry per-row `pos3`/`ext_emb`.
    pub fn any_pos3(&self, work: &[PendingWork]) -> bool {
        work.iter()
            .any(|w| (w.vl_prefill.is_some() && !self.vl_sequential) || w.pos3_delta != 0)
    }

    /// The row-allocation half of [`next_batch_eligible`](Self::next_batch_eligible)
    /// with no side effects: reads `work` and the rotation cursor, returns
    /// per-slot row counts and the cursor value the caller publishes only
    /// once the step built from this allocation has committed.
    pub fn allocate_rows(
        &self,
        work: &[PendingWork],
        remaining_rows: usize,
        prefill_min_tokens: usize,
        eligible: &[bool],
    ) -> RowAllocation {
        let n = work.len();
        // ---- Phase 1: decode (1 row per runnable decode slot, slot order) ----
        // Admit decode lanes in slot order until the budget is exhausted; a
        // step that cannot afford every decode lane serves a prefix of them
        // rather than overflowing (spec §5.2 S2). A slot not in the FairQueue
        // grant mask (`eligible[s] == false`) is skipped entirely.
        let mut alloc = vec![0usize; n];
        let mut next_prefill_cursor = None;
        let mut used = 0usize;
        for i in 0..n {
            if !is_runnable_decode(&work[i], self.vl_sequential)
                || !eligible.get(i).copied().unwrap_or(false)
            {
                continue;
            }
            if used + 1 > remaining_rows {
                break;
            }
            alloc[i] = 1;
            used += 1;
        }

        // ---- Phase 2: prefill (round-robin from the rotation cursor) ----
        // The first prefilling slot in rotation order receives at least
        // `prefill_min_tokens` rows (when its prompt and the remaining budget
        // allow); leftover budget is distributed by rotation, capping each
        // slot at `chunk_size`. The cursor advances each tick so a different
        // prefilling slot gets the quantum next step.
        let mut avail = remaining_rows.saturating_sub(used);
        if avail > 0 {
            let prefill_slots: Vec<usize> = (0..n)
                .filter(|&i| {
                    is_runnable_prefill(&work[i], self.vl_sequential)
                        && eligible.get(i).copied().unwrap_or(false)
                })
                .collect();
            if !prefill_slots.is_empty() {
                let n_pr = prefill_slots.len();
                let start = self.prefill_cursor % n_pr;
                // Per-slot round quota: chunk_size capped to the prompt,
                // quantum-floored, then page-aligned when the take would
                // complete a mid-page prompt (spec §4.5, see
                // `page_align_take`). Both the rotated head allocation and
                // the leftover distribution draw from this quota, so a
                // trimmed head is not refilled within the same round.
                let quota_of = |w: &PendingWork| -> usize {
                    let left = w.remaining_prompt.len();
                    if left == 0 {
                        return 0;
                    }
                    page_align_take(
                        self.chunk_size.min(left).max(prefill_min_tokens.min(left)),
                        w.next_pos,
                        left,
                        prefill_min_tokens,
                    )
                };
                let head = prefill_slots[start];
                let head_remaining = work[head].remaining_prompt.len();
                let head_take = quota_of(&work[head]).min(avail);
                if head_take > 0 && head_remaining > 0 {
                    alloc[head] += head_take;
                    avail -= head_take;
                    used += head_take;
                }
                // Distribute any leftover budget by rotation, capping each
                // slot at `chunk_size` total.
                let mut keep_going = true;
                while avail > 0 && keep_going {
                    keep_going = false;
                    for k in 0..n_pr {
                        if avail == 0 {
                            break;
                        }
                        let i = prefill_slots[(start + k) % n_pr];
                        let already = alloc[i];
                        let prompt_left = work[i].remaining_prompt.len().saturating_sub(already);
                        let cap = quota_of(&work[i]).saturating_sub(already);
                        let take = cap.min(prompt_left).min(avail);
                        if take == 0 {
                            continue;
                        }
                        alloc[i] += take;
                        avail -= take;
                        used += take;
                        keep_going = true;
                    }
                }
                next_prefill_cursor = Some((self.prefill_cursor + 1) % n_pr);
            }
        }
        RowAllocation {
            alloc,
            next_prefill_cursor,
        }
    }
}

/// Side-effect-free result of [`Scheduler::allocate_rows`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowAllocation {
    /// Rows granted per `work` index (decode 1, prefill chunk, else 0).
    pub alloc: Vec<usize>,
    /// Rotation cursor to publish after the step commits; `None` = unchanged.
    pub next_prefill_cursor: Option<usize>,
}

// ---- Runnable predicates (spec §5.2 S2) ----
// A slot is skipped entirely when a sequential VL path owns it or it is a
// decoding MTP slot whose verify rows the caller injects separately. A
// batched VL slot still waiting on its vision-tower embeddings contributes no
// rows (image pads would embed as the raw pad token). Decode slots carry one
// seed token in `remaining_prompt`; prefill slots carry the un-prefilled
// prompt suffix.
/// Sequential VL or a decoding spec slot: owned elsewhere, never batched here.
pub fn skip_entirely(w: &PendingWork, vl_sequential: bool) -> bool {
    (w.vl_prefill.is_some() && vl_sequential) || (w.spec.active() && w.decoding)
}

/// Batched VL slot whose vision-tower embeddings have not landed yet.
pub fn vl_waiting(w: &PendingWork) -> bool {
    w.vl_prefill
        .as_ref()
        .is_some_and(|vl| vl.embeddings.is_empty() && vl.n_visual_tokens > 0)
}

/// A decoding slot that contributes one ordinary decode row this step.
pub fn is_runnable_decode(w: &PendingWork, vl_sequential: bool) -> bool {
    if skip_entirely(w, vl_sequential) || vl_waiting(w) {
        return false;
    }
    w.decoding && !w.remaining_prompt.is_empty()
}

/// A prefilling slot with prompt tokens left to feed.
pub fn is_runnable_prefill(w: &PendingWork, vl_sequential: bool) -> bool {
    if skip_entirely(w, vl_sequential) || vl_waiting(w) {
        return false;
    }
    !w.decoding && !w.remaining_prompt.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdna_compute::slot_pool::SlotId;

    fn prompt(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    #[test]
    fn a_long_prompt_is_chunked_not_run_whole() {
        let mut s = Scheduler {
            chunk_size: 256,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: prompt(1000),
            next_pos: 0,
            decoding: false,
            vl_prefill: None,
            spec: SpecKind::None,
            spec_cycles: 0,
            spec_committed: 0,
            spec_retire_fails: 0,
            pos3_delta: 0,
        }];
        let b = s.next_batch(&mut work, 4096, 1);
        assert_eq!(
            b.total_rows(),
            256,
            "must take one chunk, not the whole prompt"
        );
        assert_eq!(work[0].remaining_prompt.len(), 744);
        assert_eq!(work[0].next_pos, 256);
    }

    #[test]
    fn prefill_and_decode_mix_in_one_batch() {
        let mut s = Scheduler {
            chunk_size: 256,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![
            PendingWork {
                slot: SlotId(0),
                remaining_prompt: prompt(300),
                next_pos: 0,
                decoding: false,
                vl_prefill: None,
                spec: SpecKind::None,
                spec_cycles: 0,
                spec_committed: 0,
                spec_retire_fails: 0,
                pos3_delta: 0,
            },
            PendingWork {
                slot: SlotId(1),
                remaining_prompt: vec![42],
                next_pos: 10,
                decoding: true,
                vl_prefill: None,
                spec: SpecKind::None,
                spec_cycles: 0,
                spec_committed: 0,
                spec_retire_fails: 0,
                pos3_delta: 0,
            },
        ];
        let b = s.next_batch(&mut work, 4096, 1);
        assert_eq!(
            b.m_per_slot,
            vec![256, 1],
            "a prefilling slot must not block a decoding one"
        );
    }

    #[test]
    fn a_prompt_shorter_than_a_chunk_completes_in_one_batch() {
        let mut s = Scheduler {
            chunk_size: 256,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: prompt(10),
            next_pos: 0,
            decoding: false,
            vl_prefill: None,
            spec: SpecKind::None,
            spec_cycles: 0,
            spec_committed: 0,
            spec_retire_fails: 0,
            pos3_delta: 0,
        }];
        let b = s.next_batch(&mut work, 4096, 1);
        assert_eq!(b.total_rows(), 10);
        assert!(work[0].remaining_prompt.is_empty());
    }

    #[test]
    fn an_idle_slot_contributes_nothing() {
        let mut s = Scheduler {
            chunk_size: 256,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: vec![],
            next_pos: 0,
            decoding: false,
            vl_prefill: None,
            spec: SpecKind::None,
            spec_cycles: 0,
            spec_committed: 0,
            spec_retire_fails: 0,
            pos3_delta: 0,
        }];
        let b = s.next_batch(&mut work, 4096, 1);
        assert!(b.is_empty());
    }

    #[test]
    fn a_mid_page_prompt_splits_at_the_last_page_boundary() {
        // Spec §4.5: a checkpoint is only capturable when a prefill round
        // ends exactly at a page boundary. A single-chunk 283-token prompt
        // (chunk_size above the prompt length, the serve default) must be
        // trimmed to 256 rows so the round ends at the boundary; the 27-row
        // tail completes in the next round.
        let mut s = Scheduler {
            chunk_size: 1024,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: prompt(283),
            next_pos: 0,
            decoding: false,
            vl_prefill: None,
            spec: SpecKind::None,
            spec_cycles: 0,
            spec_committed: 0,
            spec_retire_fails: 0,
            pos3_delta: 0,
        }];
        let b = s.next_batch(&mut work, 4096, 1);
        assert_eq!(
            b.total_rows(),
            256,
            "completing round must end at the page boundary"
        );
        assert_eq!(work[0].next_pos, 256);
        let b2 = s.next_batch(&mut work, 4096, 1);
        assert_eq!(b2.total_rows(), 27, "the sub-page tail completes prefill");
        assert!(work[0].remaining_prompt.is_empty());
        assert_eq!(work[0].next_pos, 283);
    }

    #[test]
    fn a_page_aligned_prompt_prefills_in_one_round() {
        // 256 is exactly one page multiple: no trim, no extra round.
        let mut s = Scheduler {
            chunk_size: 1024,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: prompt(256),
            next_pos: 0,
            decoding: false,
            vl_prefill: None,
            spec: SpecKind::None,
            spec_cycles: 0,
            spec_committed: 0,
            spec_retire_fails: 0,
            pos3_delta: 0,
        }];
        let b = s.next_batch(&mut work, 4096, 1);
        assert_eq!(b.total_rows(), 256);
        assert!(work[0].remaining_prompt.is_empty());
    }

    #[test]
    fn quantum_above_a_page_wins_over_alignment() {
        // An operator prefill_min_tokens above one page opts out of the
        // alignment trim — the configured service floor must not be silently
        // undercut.
        let mut s = Scheduler {
            chunk_size: 1024,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: prompt(283),
            next_pos: 0,
            decoding: false,
            vl_prefill: None,
            spec: SpecKind::None,
            spec_cycles: 0,
            spec_committed: 0,
            spec_retire_fails: 0,
            pos3_delta: 0,
        }];
        let b = s.next_batch(&mut work, 4096, 200);
        assert_eq!(
            b.total_rows(),
            283,
            "quantum above a page: no alignment trim"
        );
        assert!(work[0].remaining_prompt.is_empty());
    }

    fn vl_state(n_visual: usize, n_prompt: usize) -> VlPrefill {
        VlPrefill {
            patches: vec![],
            grid_h: 1,
            grid_w: 1,
            n_visual_tokens: n_visual,
            embeddings: vec![0.0; n_visual * 4],
            dim: 4,
            visual_idx: 0,
            image_pad_id: 42,
            // Phases for the prompt: text rows [p,p,p]; an "image" row at
            // position 4 stretched on h/w (values arbitrary but distinct
            // from the 1D phase, to prove the table wins).
            mrope_positions: (0..n_prompt)
                .map(|p| {
                    if p == 4 {
                        [4, 9, 9]
                    } else {
                        [p as i32, p as i32, p as i32]
                    }
                })
                .collect(),
            rope_delta: 7,
            base: 0,
        }
    }

    #[test]
    fn vl_slots_waiting_on_vision_forward_are_skipped() {
        // Embeddings not yet produced: the slot must not enter the batch
        // (image pads would embed garbage), but the decoding slot next to it
        // is unaffected.
        let mut s = Scheduler {
            chunk_size: 256,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![
            PendingWork {
                slot: SlotId(0),
                remaining_prompt: prompt(10),
                next_pos: 0,
                decoding: false,
                vl_prefill: Some(VlPrefill {
                    embeddings: vec![],
                    ..vl_state(1, 10)
                }),
                spec: SpecKind::None,
                spec_cycles: 0,
                spec_committed: 0,
                spec_retire_fails: 0,
                pos3_delta: 0,
            },
            PendingWork {
                slot: SlotId(1),
                remaining_prompt: vec![7],
                next_pos: 3,
                decoding: true,
                vl_prefill: None,
                spec: SpecKind::None,
                spec_cycles: 0,
                spec_committed: 0,
                spec_retire_fails: 0,
                pos3_delta: 0,
            },
        ];
        let b = s.next_batch(&mut work, 4096, 1);
        assert_eq!(b.m_per_slot, vec![0, 1], "unprepared VL slots must wait");
        assert_eq!(work[0].remaining_prompt.len(), 10);
        assert_eq!(work[0].next_pos, 0);
        assert!(work[1].remaining_prompt.is_empty());
    }

    #[test]
    fn batched_vl_rows_carry_mrope_phases() {
        let mut s = Scheduler {
            chunk_size: 256,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        // Prompt where position 4 is an image pad.
        let mut toks = prompt(10);
        toks[4] = 42;
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: toks,
            next_pos: 0,
            decoding: false,
            vl_prefill: Some(vl_state(1, 10)),
            spec: SpecKind::None,
            spec_cycles: 0,
            spec_committed: 0,
            spec_retire_fails: 0,
            pos3_delta: 0,
        }];
        let b = s.next_batch(&mut work, 4096, 1);
        assert_eq!(b.total_rows(), 10);
        assert_eq!(b.pos3.len(), 10, "one phase triplet per row");
        assert_eq!(b.pos3[4], [4, 9, 9], "the image row reads the table");
        assert_eq!(b.pos3[0], [0, 0, 0], "text rows stay at [p, p, p]");
        assert_eq!(b.ext_emb, vec![-1, -1, -1, -1, 0, -1, -1, -1, -1, -1]);
        assert_eq!(work[0].vl_prefill.as_ref().unwrap().visual_idx, 1);
    }

    #[test]
    fn vl_decode_rows_use_pos_plus_rope_delta() {
        let mut s = Scheduler {
            chunk_size: 4,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: vec![9],
            next_pos: 12,
            decoding: true,
            vl_prefill: Some(vl_state(2, 4)),
            spec: SpecKind::None,
            spec_cycles: 0,
            spec_committed: 0,
            spec_retire_fails: 0,
            pos3_delta: 0,
        }];
        let b = s.next_batch(&mut work, 4096, 1);
        assert_eq!(b.total_rows(), 1);
        // Position 12 is past the 4-row table: [p + rope_delta; 3].
        assert_eq!(b.pos3[0], [12 + 7, 12 + 7, 12 + 7]);
        assert_eq!(b.ext_emb[0], -1, "decode rows never splice embeddings");
    }

    #[test]
    fn sequential_vl_slots_still_bypass_the_batch() {
        let mut s = Scheduler {
            chunk_size: 256,
            vl_sequential: true,
            prefill_cursor: 0,
        };
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: prompt(10),
            next_pos: 0,
            decoding: false,
            vl_prefill: Some(vl_state(1, 10)),
            spec: SpecKind::None,
            spec_cycles: 0,
            spec_committed: 0,
            spec_retire_fails: 0,
            pos3_delta: 0,
        }];
        let b = s.next_batch(&mut work, 4096, 1);
        assert!(b.is_empty(), "sequential mode owns the slot");
        assert!(b.pos3.is_empty(), "no VL rows implies no side arrays");
        assert_eq!(work[0].remaining_prompt.len(), 10);
    }

    #[test]
    fn mtp_prefill_slots_batch_but_decoding_mtp_slots_are_skipped() {
        let mut s = Scheduler {
            chunk_size: 256,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![
            PendingWork {
                // MTP slot still prefilling: its prompt chunk flows through
                // the batched path (the head-KV fill runs post-forward).
                slot: SlotId(0),
                remaining_prompt: prompt(10),
                next_pos: 0,
                decoding: false,
                vl_prefill: None,
                spec: SpecKind::Mtp,
                spec_cycles: 0,
                spec_committed: 0,
                spec_retire_fails: 0,
                pos3_delta: 0,
            },
            PendingWork {
                // MTP slot decoding: owned by the draft/verify cycle; its
                // verify rows are injected by the engine, not the scheduler.
                slot: SlotId(1),
                remaining_prompt: vec![7],
                next_pos: 3,
                decoding: true,
                vl_prefill: None,
                spec: SpecKind::Mtp,
                spec_cycles: 0,
                spec_committed: 0,
                spec_retire_fails: 0,
                pos3_delta: 0,
            },
        ];
        let b = s.next_batch(&mut work, 4096, 1);
        assert_eq!(
            b.m_per_slot,
            vec![10, 0],
            "prefilling MTP slots batch; decoding MTP slots are injected separately"
        );
        assert!(work[0].remaining_prompt.is_empty());
        assert_eq!(work[0].next_pos, 10);
        assert_eq!(work[1].remaining_prompt, vec![7], "seed stays put");
        assert_eq!(work[1].next_pos, 3);
    }

    #[test]
    fn continuation_rope_delta_shifts_phases_without_splicing_embeddings() {
        // A text follow-up to an image turn: no vl_prefill, but the session's
        // rope delta must phase-shift every row while ext_emb stays -1 (no
        // vision matrix exists any more). A pure-text neighbour keeps
        // [p, p, p].
        let mut s = Scheduler {
            chunk_size: 256,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![
            PendingWork {
                slot: SlotId(0),
                remaining_prompt: prompt(3),
                next_pos: 300,
                decoding: false,
                vl_prefill: None,
                spec: SpecKind::None,
                spec_cycles: 0,
                spec_committed: 0,
                spec_retire_fails: 0,
                pos3_delta: -240,
            },
            PendingWork {
                slot: SlotId(1),
                remaining_prompt: vec![9],
                next_pos: 5,
                decoding: true,
                vl_prefill: None,
                spec: SpecKind::None,
                spec_cycles: 0,
                spec_committed: 0,
                spec_retire_fails: 0,
                pos3_delta: 0,
            },
        ];
        let b = s.next_batch(&mut work, 4096, 1);
        assert_eq!(b.m_per_slot, vec![3, 1]);
        assert_eq!(b.pos3.len(), 4, "delta rows put the step on M-RoPE too");
        assert_eq!(b.pos3[0], [300 - 240; 3], "phase = pos + session delta");
        assert_eq!(b.pos3[2], [302 - 240; 3]);
        assert_eq!(b.pos3[3], [5, 5, 5], "pure-text rows stay at [p, p, p]");
        assert!(b.ext_emb.iter().all(|&e| e == -1), "never splice here");
    }

    #[test]
    fn plain_text_step_still_skips_the_pos3_arrays() {
        let mut s = Scheduler {
            chunk_size: 256,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: prompt(4),
            next_pos: 0,
            decoding: false,
            vl_prefill: None,
            spec: SpecKind::Mtp,
            spec_cycles: 0,
            spec_committed: 0,
            spec_retire_fails: 0,
            pos3_delta: 0,
        }];
        let b = s.next_batch(&mut work, 4096, 1);
        assert_eq!(b.total_rows(), 4);
        assert!(b.pos3.is_empty(), "no VL rows implies no side arrays");
        assert!(b.ext_emb.is_empty());
    }

    fn work_prefill(slot: usize, prompt_len: usize, next_pos: usize) -> PendingWork {
        PendingWork {
            slot: SlotId(slot),
            remaining_prompt: prompt(prompt_len),
            next_pos,
            decoding: false,
            vl_prefill: None,
            spec: SpecKind::None,
            spec_cycles: 0,
            spec_committed: 0,
            spec_retire_fails: 0,
            pos3_delta: 0,
        }
    }

    fn work_decode(slot: usize, seed: u32, next_pos: usize) -> PendingWork {
        PendingWork {
            slot: SlotId(slot),
            remaining_prompt: vec![seed],
            next_pos,
            decoding: true,
            vl_prefill: None,
            spec: SpecKind::None,
            spec_cycles: 0,
            spec_committed: 0,
            spec_retire_fails: 0,
            pos3_delta: 0,
        }
    }

    #[test]
    fn mixed_decode_and_prefill_never_exceeds_max_batch_tokens() {
        // 4 decode slots + 1 prefill slot (300 tokens, chunk 256). Budget 6.
        // Decode gets 4 rows (one each), prefill gets 2 (remaining budget).
        let mut s = Scheduler {
            chunk_size: 256,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![
            work_decode(0, 1, 10),
            work_decode(1, 2, 20),
            work_decode(2, 3, 30),
            work_decode(3, 4, 40),
            work_prefill(4, 300, 0),
        ];
        let b = s.next_batch(&mut work, 6, 1);
        assert!(
            b.total_rows() <= 6,
            "batch must not exceed budget: got {}",
            b.total_rows()
        );
        assert_eq!(
            b.m_per_slot[0..4],
            vec![1, 1, 1, 1],
            "all 4 decode slots served"
        );
        assert_eq!(b.m_per_slot[4], 2, "prefill gets the remaining 2 rows");
        assert_eq!(work[4].remaining_prompt.len(), 298);
        assert_eq!(work[4].next_pos, 2);
    }

    #[test]
    fn one_huge_prefill_chunk_is_truncated_to_remaining_budget() {
        // A single prefill slot with a huge prompt. Budget 10, chunk 256.
        // The slot gets only 10 rows (the budget), not the whole chunk.
        let mut s = Scheduler {
            chunk_size: 256,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![work_prefill(0, 1000, 0)];
        let b = s.next_batch(&mut work, 10, 1);
        assert_eq!(b.total_rows(), 10, "prefill truncated to remaining budget");
        assert_eq!(work[0].remaining_prompt.len(), 990);
        assert_eq!(work[0].next_pos, 10);
    }

    #[test]
    fn rotation_serves_two_prefills_across_ticks() {
        // Two prefill slots, budget 4, chunk 256, prefill_min 1.
        // Tick 1: cursor 0 → slot 0 gets the quantum (at least 1 row), then
        // leftover budget (3) is distributed round-robin: slot 0 gets more
        // up to chunk, then slot 1. With chunk=256 and budget=4, slot 0
        // gets 4 rows total (quantum 1 + 3 extra), slot 1 gets 0.
        // Tick 2: cursor advances to 1 → slot 1 gets the quantum first.
        let mut s = Scheduler {
            chunk_size: 256,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![work_prefill(0, 100, 0), work_prefill(1, 100, 0)];
        // Tick 1: cursor=0, slot 0 is the head.
        let b1 = s.next_batch(&mut work, 4, 1);
        assert_eq!(b1.total_rows(), 4);
        assert!(b1.m_per_slot[0] > 0, "slot 0 gets rows on tick 1");
        // Tick 2: cursor=1, slot 1 is the head.
        let b2 = s.next_batch(&mut work, 4, 1);
        assert_eq!(b2.total_rows(), 4);
        assert!(
            b2.m_per_slot[1] > 0,
            "slot 1 gets rows on tick 2 (rotation): {:?}",
            b2.m_per_slot
        );
        // Both slots made progress across the two ticks.
        assert!(work[0].next_pos > 0 && work[1].next_pos > 0);
    }

    #[test]
    fn zero_remaining_budget_returns_empty_batch() {
        // Budget 0: no rows can be allocated, even with runnable work.
        let mut s = Scheduler {
            chunk_size: 256,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![work_decode(0, 1, 10), work_prefill(1, 300, 0)];
        let b = s.next_batch(&mut work, 0, 1);
        assert!(b.is_empty(), "zero budget → empty batch");
        assert_eq!(work[0].remaining_prompt, vec![1], "decode seed untouched");
        assert_eq!(
            work[1].remaining_prompt.len(),
            300,
            "prefill prompt untouched"
        );
    }

    #[test]
    fn decode_lanes_admitted_only_up_to_budget() {
        // 5 decode slots, budget 3: only 3 decode lanes get a row.
        let mut s = Scheduler {
            chunk_size: 256,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![
            work_decode(0, 1, 10),
            work_decode(1, 2, 20),
            work_decode(2, 3, 30),
            work_decode(3, 4, 40),
            work_decode(4, 5, 50),
        ];
        let b = s.next_batch(&mut work, 3, 1);
        assert_eq!(b.total_rows(), 3);
        assert_eq!(b.m_per_slot, vec![1, 1, 1, 0, 0]);
        // The unserved decode slots keep their seed.
        assert_eq!(work[3].remaining_prompt, vec![4]);
        assert_eq!(work[4].remaining_prompt, vec![5]);
    }

    #[test]
    fn prefill_min_tokens_guaranteed_when_space_remains() {
        // 1 decode (1 row) + 1 prefill (100 tokens). Budget 5, prefill_min 3.
        // Decode gets 1, prefill gets at least 3 (the min quantum).
        let mut s = Scheduler {
            chunk_size: 256,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![work_decode(0, 1, 10), work_prefill(1, 100, 0)];
        let b = s.next_batch(&mut work, 5, 3);
        assert_eq!(b.m_per_slot[0], 1, "decode gets 1 row");
        assert!(
            b.m_per_slot[1] >= 3,
            "prefill gets at least prefill_min_tokens: got {}",
            b.m_per_slot[1]
        );
        assert!(b.total_rows() <= 5);
    }

    /// An ineligible slot contributes 0 rows and its `remaining_prompt` is
    /// NOT drained — no tokens are lost. This is the contract the serve
    /// engine relies on when it passes the FairQueue grant mask: an
    /// un-granted slot must contribute 0 rows (spec §5.3 S3).
    #[test]
    fn ineligible_slot_contributes_zero_rows_and_keeps_its_prompt() {
        let mut s = Scheduler {
            chunk_size: 256,
            vl_sequential: false,
            prefill_cursor: 0,
        };
        let mut work = vec![
            PendingWork {
                slot: SlotId(0),
                remaining_prompt: prompt(100),
                next_pos: 0,
                decoding: false,
                vl_prefill: None,
                spec: SpecKind::None,
                spec_cycles: 0,
                spec_committed: 0,
                spec_retire_fails: 0,
                pos3_delta: 0,
            },
            PendingWork {
                slot: SlotId(1),
                remaining_prompt: prompt(100),
                next_pos: 0,
                decoding: false,
                vl_prefill: None,
                spec: SpecKind::None,
                spec_cycles: 0,
                spec_committed: 0,
                spec_retire_fails: 0,
                pos3_delta: 0,
            },
        ];
        // Only slot 0 is eligible (FairQueue granted it); slot 1 is not.
        let eligible = [true, false];
        let b = s.next_batch_eligible(&mut work, 4096, 1, &eligible);
        assert_eq!(
            b.m_per_slot,
            vec![100, 0],
            "ineligible slot contributes 0 rows"
        );
        // Slot 1's prompt is untouched (not drained).
        assert_eq!(
            work[1].remaining_prompt.len(),
            100,
            "ineligible slot keeps its prompt"
        );
        assert_eq!(work[1].next_pos, 0);
        // Slot 0's prompt was consumed.
        assert!(work[0].remaining_prompt.is_empty());
    }
}
