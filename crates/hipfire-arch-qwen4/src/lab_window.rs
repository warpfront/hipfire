// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Read-only lab observation of the production sampled native-MTP accepting
//! window (`feature = "lab"`; the module does not exist in a non-lab build).
//!
//! Three taps feed one [`LabWindow`] per `mtp_step`:
//!
//! - **window tap** (`Qwen4MtpDrafter::lab_observe_windows`): the proposals
//!   (every draft id, rejected ones included, with the proposal row `q` it
//!   was drawn from), the transformed verify target rows `p` the acceptance
//!   actually read, each accept/reject verdict with the stream uniforms it
//!   consumed, the terminal (residual or bonus) draw, and the keep count.
//!   The host route and the device-accept route
//!   (`HIPFIRE_QWEN4_MTP_DEVICE_ACCEPT=1`) produce the same records; the
//!   device route reads its resident target rows back **only while armed**.
//! - **state tap** (`Qwen4State::lab_observe_state`): GDN residual reset
//!   events (`Reset` / `Restore` / `PartialRollback`) and capture-ring
//!   events (verify write half, rollback re-run half, restore of the live
//!   slot).
//! - **forward-logits tap** (`Qwen4Bundle::lab_observe`): per speculative
//!   forward, the count of non-finite logits of every output row, and the
//!   rows themselves when `keep_logits`.
//!
//! Every tap is pure read: it never touches the request RNG, the acceptance
//! arithmetic, routing or any launch, and an unarmed tap does no work (each
//! hook is one `Option`/`bool` test; nothing is read back or allocated).
//! Armed device readbacks synchronize; that is why they are lab-only.

use hipfire_runtime::spec::MtpWindow;
use hipfire_runtime::spec_sampling::{DraftVerdict, SparseDist, SpecRng};

/// A truncated distribution as `SparseDist::entries()`: `(token, probability)`
/// in descending probability order.
pub type LabDist = Vec<(u32, f32)>;

/// Draws a window is allowed to consume from the request stream when its
/// offsets are measured; far above `2 * max_k + 2`.
const MAX_WINDOW_DRAWS: usize = 256;

/// Which verify route the window ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabVerifyRoute {
    /// Target-only window (AR calibration / retired AR token): one row, a
    /// bonus-style draw, no proposals.
    TargetOnly,
    /// One `k + 1`-row verify forward and one ordered acceptance.
    Batched,
    /// One-row target forward per draft step, verdict per step.
    Interleaved,
}

/// One uniform of the request-seeded stream: `index` counts the draws of the
/// window from its start (`LabWindow::uniforms[index] == u`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LabDraw {
    pub index: usize,
    pub u: f32,
}

/// The verdict of one proposal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LabVerdict {
    /// Accepted with the uniform `draw` (`u * q < p`).
    Accept { draw: LabDraw },
    /// Rejected with `draw`; `residual` is the replacement drawn from
    /// `(p - q)+` with `residual_draw`.
    Reject {
        draw: LabDraw,
        residual_draw: LabDraw,
        residual: u32,
    },
}

/// One proposal of the window.
#[derive(Clone, Debug, PartialEq)]
pub struct LabProposal {
    /// The draft token id (rejected proposals included).
    pub token: u32,
    /// The proposal row `q`: the draft head's distribution the token was
    /// drawn from, or the point mass on `token` for an n-gram takeover
    /// candidate (`point_mass`). Empty for the naive verifier.
    pub q: LabDist,
    pub point_mass: bool,
    /// The stream uniform that drew `token` from `q`; `None` for a point
    /// mass or the naive verifier.
    pub draw: Option<LabDraw>,
    /// `None`: never verified (a batched window's proposals after its first
    /// rejection, every proposal of the naive verifier).
    pub verdict: Option<LabVerdict>,
    /// `p(token)` of the target row that judged it (`None`: unverified).
    pub p_token: Option<f32>,
    /// `q(token)`.
    pub q_token: f32,
}

/// One verify target row `p` the acceptance read.
#[derive(Clone, Debug, PartialEq)]
pub struct LabTargetRow {
    /// Window verify-row index (0 = the seed's row).
    pub row: usize,
    /// After penalties, temperature, top_k, top_p and min_p: the entries the
    /// acceptance used, identical on the host and device routes.
    pub p: LabDist,
    /// Read back from the device accept's resident rows.
    pub from_device: bool,
}

/// The terminal draw of an all-accepted window (the bonus) or of a
/// target-only window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LabBonus {
    pub row: usize,
    pub draw: LabDraw,
    pub token: u32,
}

/// What a GDN residual transition did (see `GdnResidualTransition`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabResidualKind {
    Reset,
    Restore,
    PartialRollback,
}

/// One submitted full-owner residual zero sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LabResidualEvent {
    pub kind: LabResidualKind,
    /// `Qwen4State::gdn_residual_resets()` after this event.
    pub total_after: u64,
}

/// A capture-ring event (the GDN recurrent-state ring has two halves of
/// `ring_rows` slots; `half` is `base / ring_rows`, 0 or 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabRingEvent {
    /// An armed `rows`-row verify wrote its states into the free half
    /// starting at slot `base` and made slot `live_after` live.
    Verify {
        rows: usize,
        ring_rows: usize,
        live_before: usize,
        base: usize,
        half: u8,
        live_after: usize,
    },
    /// A partial rollback to `keep` of `rows` rows re-ran the kept rows into
    /// the free half (relative to the snapshot's live slot `live_before`).
    Rollback {
        rows: usize,
        keep: usize,
        ring_rows: usize,
        live_before: usize,
        base: usize,
        half: u8,
        live_after: usize,
    },
    /// A snapshot restore put slot `live` back as the live slot.
    Restore { live: usize },
    /// The ring was (re)allocated; slot 0 is live.
    Realloc { ring_rows: usize },
}

/// State-tap events drained together.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LabStateEvents {
    pub residual: Vec<LabResidualEvent>,
    pub ring: Vec<LabRingEvent>,
}

/// The state tap held by `Qwen4State`.
#[derive(Debug, Default)]
pub struct LabStateTap {
    armed: bool,
    events: LabStateEvents,
}

impl LabStateTap {
    pub(crate) fn set_armed(&mut self, armed: bool) {
        self.armed = armed;
        if !armed {
            self.events = LabStateEvents::default();
        }
    }

    pub(crate) fn armed(&self) -> bool {
        self.armed
    }

    pub(crate) fn take(&mut self) -> LabStateEvents {
        std::mem::take(&mut self.events)
    }

    pub(crate) fn residual(&mut self, kind: LabResidualKind, total_after: u64) {
        if self.armed {
            self.events
                .residual
                .push(LabResidualEvent { kind, total_after });
        }
    }

    pub(crate) fn ring(&mut self, event: LabRingEvent) {
        if self.armed {
            self.events.ring.push(event);
        }
    }
}

/// One speculative forward's logits as the verify read them.
#[derive(Clone, Debug, PartialEq)]
pub struct LabForwardLogits {
    /// The forward's input tokens.
    pub tokens: Vec<u32>,
    /// A prompt prefill forward (not a verify/capture/replay row block).
    pub prefill: bool,
    /// Output rows of the forward (`logits.len() == rows * vocab` when kept).
    pub rows: usize,
    pub vocab: usize,
    /// Non-finite (NaN or infinite) f32 values per output row. Read right
    /// after the forward, BEFORE the GPU penalty prepass and every host
    /// policy stage: raw model logits (pre-penalty, pre-temperature).
    pub nonfinite: Vec<u64>,
    /// The raw rows (same point as `nonfinite`), kept only when armed with
    /// `keep_logits`. Penalties are applied later, in place on the device
    /// (`SampledVerify::penalize_rows`) or on a host copy; the history is not
    /// recorded but is `suffix_W(prompt ‖ emitted ‖ proposals[..row].token)`.
    pub logits: Option<Vec<f32>>,
}

/// The forward-logits tap held by `Qwen4Bundle`.
#[derive(Debug, Default)]
pub struct LabLogitsTap {
    armed: bool,
    keep_logits: bool,
    forwards: Vec<LabForwardLogits>,
}

impl LabLogitsTap {
    pub(crate) fn set_armed(&mut self, armed: bool, keep_logits: bool) {
        self.armed = armed;
        self.keep_logits = armed && keep_logits;
        if !armed {
            self.forwards.clear();
        }
    }

    pub(crate) fn armed(&self) -> bool {
        self.armed
    }

    pub(crate) fn take(&mut self) -> Vec<LabForwardLogits> {
        std::mem::take(&mut self.forwards)
    }

    /// Record one forward whose `logits` (`rows * vocab` words) were just
    /// downloaded. Only called while armed.
    pub(crate) fn record(
        &mut self,
        tokens: &[u32],
        prefill: bool,
        rows: usize,
        vocab: usize,
        logits: &[f32],
    ) {
        let nonfinite = logits
            .chunks(vocab.max(1))
            .map(|row| row.iter().filter(|value| !value.is_finite()).count() as u64)
            .collect();
        self.forwards.push(LabForwardLogits {
            tokens: tokens.to_vec(),
            prefill,
            rows,
            vocab,
            nonfinite,
            logits: self.keep_logits.then(|| logits.to_vec()),
        });
    }
}

/// How the drafter's window tap is armed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LabWindowArm {
    pub armed: bool,
    /// Also keep the verify forwards' logit rows (`LabForwardLogits::logits`).
    pub keep_logits: bool,
}

/// Everything observed in one `mtp_step` window.
#[derive(Clone, Debug)]
pub struct LabWindow {
    pub position: usize,
    pub seed_token: u32,
    /// `Debug` of the planned `WindowRoute`.
    pub plan_route: String,
    pub plan_k: usize,
    pub ngram_hit: bool,
    /// The request verifies sampled (`false`: greedy window; only the state
    /// and forward-logits observations are populated).
    pub sampled: bool,
    /// Naive verifier (`HIPFIRE_MTP_SAMPLED_MODE=naive`): per-row AR-sampler
    /// draws in `naive_draws`, no `q`/`p` rows.
    pub naive: bool,
    pub verify_route: LabVerifyRoute,
    /// The window took a device row build or device accept.
    pub device_route: bool,
    /// The proposals were point masses (n-gram takeover candidates).
    pub point_mass: bool,
    /// Every proposal drawn, in draw order: the first `drafts_generated` were
    /// verified (up to the first rejection); any beyond were drawn and pruned
    /// before the verify.
    pub proposals: Vec<LabProposal>,
    pub rows: Vec<LabTargetRow>,
    pub bonus: Option<LabBonus>,
    /// Naive verifier: `(verify row, drawn token)` in draw order.
    pub naive_draws: Vec<(usize, u32)>,
    /// Every uniform the window consumed from the request stream, in order
    /// (draft draws, verdicts, residual and bonus draws).
    pub uniforms: Vec<f32>,
    /// The two result words of every device accept in the window
    /// (`[accepted, terminal]`, `terminal == u32::MAX` for a stopping accept).
    pub device_words: Vec<[u32; 2]>,
    pub committed: Vec<u32>,
    pub accepted: usize,
    pub drafts_generated: usize,
    /// Verify rows the window ran (seed row included).
    pub verify_rows: usize,
    /// Verify rows kept, seed row included: `1` after a rejection at proposal
    /// 0 (keep0), `verify_rows` after a full accept, else partial.
    pub keep: usize,
    /// The window's `Err`, when it failed.
    pub error: Option<String>,
    pub gdn_residual_resets_before: u64,
    pub gdn_residual_resets_after: u64,
    /// State events since the previous window's end (restores, resets).
    pub pre_window_state: LabStateEvents,
    /// Speculative forwards (a prefill, forced advances) observed since the
    /// previous window's end.
    pub pre_window_forwards: Vec<LabForwardLogits>,
    /// State events inside the window.
    pub state: LabStateEvents,
    /// The capture-ring half the window's armed verify wrote (`None` when it
    /// ran no armed multi-row verify).
    pub ring_half: Option<u8>,
    /// The window's speculative forwards' logits observations.
    pub forwards: Vec<LabForwardLogits>,
    /// The stream position the window started at, for offset measurement.
    rng_start: Option<SpecRng>,
}

/// Whether two streams are at the same position (`SpecRng` hides its word).
fn rng_eq(a: &SpecRng, b: &SpecRng) -> bool {
    format!("{a:?}") == format!("{b:?}")
}

/// The next `n` uniforms of `at`, without advancing it.
pub(crate) fn peek_uniforms(at: &SpecRng, n: usize) -> Vec<f32> {
    let mut clone = *at;
    (0..n).map(|_| clone.next_f32()).collect()
}

impl LabWindow {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin(
        position: usize,
        seed_token: u32,
        plan_route: String,
        plan_k: usize,
        ngram_hit: bool,
        verify_route: LabVerifyRoute,
        sampled: bool,
        naive: bool,
        rng: Option<SpecRng>,
        resets_before: u64,
        pre_window_state: LabStateEvents,
        pre_window_forwards: Vec<LabForwardLogits>,
    ) -> Self {
        Self {
            position,
            seed_token,
            plan_route,
            plan_k,
            ngram_hit,
            sampled,
            naive,
            verify_route,
            device_route: false,
            point_mass: false,
            proposals: Vec::new(),
            rows: Vec::new(),
            bonus: None,
            naive_draws: Vec::new(),
            uniforms: Vec::new(),
            device_words: Vec::new(),
            committed: Vec::new(),
            accepted: 0,
            drafts_generated: 0,
            verify_rows: 0,
            keep: 0,
            error: None,
            gdn_residual_resets_before: resets_before,
            gdn_residual_resets_after: resets_before,
            pre_window_state,
            pre_window_forwards,
            state: LabStateEvents::default(),
            ring_half: None,
            forwards: Vec::new(),
            rng_start: rng,
        }
    }

    /// Draws consumed between the window start and `at` (`None`: not within
    /// [`MAX_WINDOW_DRAWS`], or the window has no stream).
    pub(crate) fn offset_of(&self, at: &SpecRng) -> Option<usize> {
        let mut clone = self.rng_start?;
        (0..=MAX_WINDOW_DRAWS).find(|_| {
            let hit = rng_eq(&clone, at);
            if !hit {
                clone.next_u64();
            }
            hit
        })
    }

    fn draw_at(&self, base: Option<usize>, tape: &[f32], offset: usize) -> LabDraw {
        LabDraw {
            index: base.map_or(usize::MAX, |base| base + offset),
            u: tape.get(offset).copied().unwrap_or(f32::NAN),
        }
    }

    /// `token` was just drawn from `q` with the uniform `before.next_f32()`.
    pub(crate) fn note_draft(&mut self, token: u32, q: &[(u32, f32)], before: &SpecRng) {
        let base = self.offset_of(before);
        let tape = peek_uniforms(before, 1);
        let draw = self.draw_at(base, &tape, 0);
        let q_token = q
            .iter()
            .find(|&&(t, _)| t == token)
            .map_or(0.0, |&(_, p)| p);
        self.proposals.push(LabProposal {
            token,
            q: q.to_vec(),
            point_mass: false,
            draw: Some(draw),
            verdict: None,
            p_token: None,
            q_token,
        });
    }

    /// Target row `row` (window verify-row index) as the acceptance read it.
    pub(crate) fn note_row(&mut self, row: usize, p: &[(u32, f32)], from_device: bool) {
        self.rows.push(LabTargetRow {
            row,
            p: p.to_vec(),
            from_device,
        });
    }

    fn row_prob(&self, row: usize, token: u32) -> Option<f32> {
        self.rows.iter().find(|r| r.row == row).map(|r| {
            r.p.iter()
                .find(|&&(t, _)| t == token)
                .map_or(0.0, |&(_, p)| p)
        })
    }

    /// A one-draft verdict of an interleaved window against the row just
    /// noted: `before` is the stream before the verdict's draws. `q` is the
    /// proposal row (the head's or the point mass); a proposal noted by
    /// `note_draft` is completed, any other (a takeover candidate) is added.
    pub(crate) fn note_step(
        &mut self,
        draft: u32,
        q: &[(u32, f32)],
        point_mass: bool,
        verdict: DraftVerdict,
        before: &SpecRng,
    ) {
        self.verify_route = LabVerifyRoute::Interleaved;
        self.point_mass |= point_mass;
        let base = self.offset_of(before);
        let tape = peek_uniforms(before, 2);
        let row = self.rows.len().saturating_sub(1);
        let p_token = self.row_prob(row, draft);
        let verdict = match verdict {
            DraftVerdict::Accept => LabVerdict::Accept {
                draw: self.draw_at(base, &tape, 0),
            },
            DraftVerdict::Reject(residual) => LabVerdict::Reject {
                draw: self.draw_at(base, &tape, 0),
                residual_draw: self.draw_at(base, &tape, 1),
                residual,
            },
        };
        let q_token = q
            .iter()
            .find(|&&(t, _)| t == draft)
            .map_or(0.0, |&(_, p)| p);
        match self.proposals.last_mut() {
            Some(last) if !point_mass && last.token == draft && last.verdict.is_none() => {
                last.verdict = Some(verdict);
                last.p_token = p_token;
            }
            _ => self.proposals.push(LabProposal {
                token: draft,
                q: q.to_vec(),
                point_mass,
                draw: None,
                verdict: Some(verdict),
                p_token,
                q_token,
            }),
        }
    }

    /// The bonus (or target-only) draw of row `row`: `before.next_f32()`.
    pub(crate) fn note_bonus(&mut self, token: u32, before: &SpecRng) {
        let base = self.offset_of(before);
        let tape = peek_uniforms(before, 1);
        let row = self.rows.len().saturating_sub(1);
        self.bonus = Some(LabBonus {
            row,
            draw: self.draw_at(base, &tape, 0),
            token,
        });
    }

    /// A naive AR-sampler draw of verify row `row`.
    pub(crate) fn note_naive_draw(&mut self, row: usize, token: u32) {
        self.naive_draws.push((row, token));
    }

    /// A whole batched acceptance over `drafts`: `qs` are the proposal rows,
    /// `committed` / `accepted` / `hit_eos` the acceptance result and `before`
    /// the stream before the accept (the drafts were already drawn). The
    /// target rows must already be noted. `words` are the device accept's
    /// result words, `None` on the host route.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn note_batched(
        &mut self,
        drafts: &[u32],
        qs: &[SparseDist],
        point_mass: bool,
        committed: &[u32],
        accepted: usize,
        hit_eos: bool,
        before: &SpecRng,
        words: Option<[u32; 2]>,
    ) {
        let k = drafts.len();
        self.verify_route = LabVerifyRoute::Batched;
        self.point_mass |= point_mass;
        self.verify_rows = k + 1;
        let accepted_eos = hit_eos && committed.len() == accepted;
        self.keep = accepted - usize::from(accepted_eos) + 1;
        if let Some(words) = words {
            self.device_words.push(words);
        }
        let base = self.offset_of(before);
        let tape = peek_uniforms(before, k + 1);
        // Point-mass takeover candidates were never drawn: add them; drafted
        // proposals were noted by `note_draft` as they were drawn (a drawn
        // proposal the window pruned stays past `k`, never verified).
        if self.proposals.len() < k {
            self.proposals.clear();
            for (&token, q) in drafts.iter().zip(qs) {
                self.proposals.push(LabProposal {
                    token,
                    q: q.entries().to_vec(),
                    point_mass,
                    draw: None,
                    verdict: None,
                    p_token: None,
                    q_token: q.prob(token),
                });
            }
        }
        let terminal = committed.len() == accepted + 1;
        for j in 0..k {
            let verdict = if j < accepted {
                Some(LabVerdict::Accept {
                    draw: self.draw_at(base, &tape, j),
                })
            } else if j == accepted && terminal {
                Some(LabVerdict::Reject {
                    draw: self.draw_at(base, &tape, j),
                    residual_draw: self.draw_at(base, &tape, j + 1),
                    residual: committed[accepted],
                })
            } else {
                None
            };
            let p_token = verdict.and_then(|_| self.row_prob(j, drafts[j]));
            let proposal = &mut self.proposals[j];
            proposal.verdict = verdict;
            proposal.p_token = p_token;
        }
        if accepted == k && terminal {
            self.bonus = Some(LabBonus {
                row: k,
                draw: self.draw_at(base, &tape, k),
                token: committed[k],
            });
        }
    }

    /// The naive batched verdict: drafts are argmax proposals (no `q`), each
    /// row's draw is in `naive_draws`.
    pub(crate) fn note_naive_batched(
        &mut self,
        drafts: &[u32],
        committed: &[u32],
        accepted: usize,
        hit_eos: bool,
    ) {
        let k = drafts.len();
        self.verify_route = LabVerifyRoute::Batched;
        self.point_mass = true;
        self.verify_rows = k + 1;
        let accepted_eos = hit_eos && committed.len() == accepted;
        self.keep = accepted - usize::from(accepted_eos) + 1;
        self.proposals = drafts
            .iter()
            .map(|&token| LabProposal {
                token,
                q: Vec::new(),
                point_mass: true,
                draw: None,
                verdict: None,
                p_token: None,
                q_token: 0.0,
            })
            .collect();
    }

    /// Close the window: the result, the stream's consumption and the state
    /// and forward observations.
    pub(crate) fn finish(
        &mut self,
        result: &Result<MtpWindow, String>,
        rng_end: Option<&SpecRng>,
        resets_after: u64,
        state: LabStateEvents,
        forwards: Vec<LabForwardLogits>,
    ) {
        match result {
            Ok(window) => {
                self.committed = window.committed.clone();
                self.accepted = window.accepted;
                self.drafts_generated = window.drafts_generated;
            }
            Err(error) => self.error = Some(error.clone()),
        }
        if self.verify_route != LabVerifyRoute::Batched {
            self.verify_rows = self.committed.len();
            self.keep = self.committed.len();
        }
        if let Some(end) = rng_end {
            if let Some(consumed) = self.offset_of(end) {
                self.uniforms = peek_uniforms(
                    &self.rng_start.expect("a measured offset has a start"),
                    consumed,
                );
            }
        }
        self.gdn_residual_resets_after = resets_after;
        self.ring_half = state.ring.iter().find_map(|event| match event {
            LabRingEvent::Verify { half, .. } => Some(*half),
            _ => None,
        });
        self.state = state;
        self.forwards = forwards;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(entries: &[(u32, f32)]) -> SparseDist {
        let mut dist = SparseDist::default();
        match entries {
            [(token, _)] => dist.set_point_mass(*token),
            _ => panic!("tests only build point masses"),
        }
        dist
    }

    fn started(rng: SpecRng) -> LabWindow {
        LabWindow::begin(
            7,
            11,
            "Native".to_string(),
            3,
            false,
            LabVerifyRoute::Batched,
            true,
            false,
            Some(rng),
            0,
            LabStateEvents::default(),
            Vec::new(),
        )
    }

    #[test]
    fn offsets_and_peeks_follow_the_stream_without_advancing_it() {
        let start = SpecRng::new(5);
        let window = started(start);
        let mut at = start;
        for expect in 0..6 {
            assert_eq!(window.offset_of(&at), Some(expect));
            at.next_f32();
        }
        let before = at;
        let tape = peek_uniforms(&at, 3);
        assert_eq!(
            tape,
            (0..3).map(|_| at.next_f32()).collect::<Vec<_>>(),
            "a peek reads the draws the stream then takes"
        );
        assert!(rng_eq(&before, &before) && !rng_eq(&before, &at));
    }

    #[test]
    fn batched_reject_records_verdicts_residual_keep_and_no_bonus() {
        let start = SpecRng::new(9);
        let mut window = started(start);
        let drafts = [4u32, 5, 6];
        let qs: Vec<SparseDist> = drafts.iter().map(|&t| dist(&[(t, 1.0)])).collect();
        window.note_row(0, &[(4, 0.9), (1, 0.1)], false);
        window.note_row(1, &[(5, 0.2), (8, 0.8)], false);
        // Accepted 1 of 3, rejected proposal 1 with residual token 8.
        window.note_batched(&drafts, &qs, true, &[4, 8], 1, false, &start, None);
        let tape = peek_uniforms(&start, 4);
        assert_eq!(window.keep, 2, "seed row plus one kept draft");
        assert_eq!(window.verify_rows, 4);
        assert_eq!(window.bonus, None);
        assert_eq!(
            window.proposals[0].verdict,
            Some(LabVerdict::Accept {
                draw: LabDraw {
                    index: 0,
                    u: tape[0]
                }
            })
        );
        assert_eq!(
            window.proposals[1].verdict,
            Some(LabVerdict::Reject {
                draw: LabDraw {
                    index: 1,
                    u: tape[1]
                },
                residual_draw: LabDraw {
                    index: 2,
                    u: tape[2]
                },
                residual: 8,
            })
        );
        assert_eq!(window.proposals[1].p_token, Some(0.2));
        assert_eq!(
            window.proposals[2].verdict, None,
            "past the first rejection"
        );
        assert!(window.point_mass);
    }

    #[test]
    fn batched_full_accept_records_the_bonus_row_and_eos_stop_has_no_terminal() {
        let start = SpecRng::new(3);
        let mut window = started(start);
        let drafts = [4u32, 5];
        let qs: Vec<SparseDist> = drafts.iter().map(|&t| dist(&[(t, 1.0)])).collect();
        window.note_batched(&drafts, &qs, true, &[4, 5, 9], 2, false, &start, None);
        let tape = peek_uniforms(&start, 3);
        assert_eq!(window.keep, 3);
        assert_eq!(
            window.bonus,
            Some(LabBonus {
                row: 2,
                draw: LabDraw {
                    index: 2,
                    u: tape[2]
                },
                token: 9
            })
        );
        // An accepted EOS draft stops the window with no terminal; the verify
        // keeps one row fewer (`target_commit_accept_len`).
        let mut eos = started(start);
        eos.note_batched(
            &drafts,
            &qs,
            true,
            &[4, 5],
            2,
            true,
            &start,
            Some([2, u32::MAX]),
        );
        assert_eq!(eos.keep, 2);
        assert_eq!(eos.bonus, None);
        assert_eq!(eos.device_words, vec![[2, u32::MAX]]);
    }

    #[test]
    fn finish_measures_the_consumed_uniforms_and_the_ring_half() {
        let start = SpecRng::new(1);
        let mut window = started(start);
        let mut end = start;
        for _ in 0..4 {
            end.next_f32();
        }
        let result: Result<MtpWindow, String> = Ok(MtpWindow {
            committed: vec![4, 5],
            accepted: 1,
            drafts_generated: 3,
        });
        let mut state = LabStateEvents::default();
        state.ring.push(LabRingEvent::Verify {
            rows: 4,
            ring_rows: 4,
            live_before: 0,
            base: 4,
            half: 1,
            live_after: 7,
        });
        window.finish(&result, Some(&end), 2, state, Vec::new());
        assert_eq!(window.uniforms, peek_uniforms(&start, 4));
        assert_eq!(window.ring_half, Some(1));
        assert_eq!(window.gdn_residual_resets_after, 2);
        assert_eq!(window.committed, vec![4, 5]);
    }

    #[test]
    fn unarmed_taps_record_nothing() {
        let mut state = LabStateTap::default();
        state.residual(LabResidualKind::Reset, 1);
        state.ring(LabRingEvent::Restore { live: 0 });
        assert!(!state.armed());
        assert_eq!(state.take(), LabStateEvents::default());
        let mut logits = LabLogitsTap::default();
        assert!(!logits.armed());
        assert!(logits.take().is_empty());
        logits.set_armed(true, false);
        logits.record(&[1], false, 1, 3, &[0.0, f32::NAN, f32::INFINITY]);
        let taken = logits.take();
        assert_eq!(taken[0].nonfinite, vec![2]);
        assert_eq!(taken[0].logits, None, "rows are kept only on request");
        logits.set_armed(false, true);
        assert!(logits.take().is_empty());
    }
}
