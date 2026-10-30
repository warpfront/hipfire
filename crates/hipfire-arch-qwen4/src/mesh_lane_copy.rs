// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Root-side copy lane and head-lag bookkeeping of the multicard native-MTP
//! lanes (`release-0.4.3/mgmtp/DESIGN.md` §5.2, `AGENTS.md:940-1179`).
//!
//! Host only. [`crate::mesh_lane_protocol::MeshRoot`] owns one [`CopyLane`]
//! per armed request: the request-local [`MtpCopyContext`] seeded from the
//! FULL prompt at admission, the retained output stream (the admission seed
//! first, as the singleton emitter holds it), and the request's own copy
//! shadow acceptance. The take-length chooser, the shadow observation and
//! the lag bookkeeping are the singleton drafter's own
//! ([`crate::mtp_spec::copy_take_decision`],
//! [`crate::mtp_spec::copy_shadow_counts`], [`crate::mtp_spec::HeadLag`]),
//! priced by the lane's own floor ([`crate::mtp_lanes::LaneSpecPolicy`]).
//!
//! - **Epoch scope.** A context is built (or recycled and reseeded; its
//!   tables are never trusted across requests, see
//!   [`MtpCopyContext`]) at admission and returned to the pool at release:
//!   nothing crosses epochs.
//! - **Retained output only.** History grows only at publish by the window's
//!   committed tokens ([`CopyLane::append`]); candidates and rejected tails
//!   never enter it, and the tables are never cleared on low acceptance.
//! - **Root only.** Followers never see the history, the shadow table or the
//!   chooser; they replay the plan's candidates and the decision's head
//!   schedule.
//! - **Staging.** [`CopyLane::offer`] reads (and only touches scratch of)
//!   the committed context; everything the window changes — the shadow
//!   table, the `active` flag, the lag book — is a clone staged in the step
//!   and applied at publish.

use crate::lane::LabRoutes;
use crate::mesh_lane_protocol::{HeadAppend, HeadEnd, HeadState};
use crate::mtp_lanes::LaneSpecPolicy;
use crate::mtp_spec::{
    copy_shadow_counts, copy_take_decision, CopyTakeInput, HeadLag, NgramDecision, NGRAM_DEPTHS,
};
use hipfire_runtime::ngram_mod::{MtpCopyContext, NgramModConfig};
use hipfire_runtime::spec_economics::{observe_agreement_counts, AGREEMENT_PRIOR, DEFAULT_DECAY};

/// Allocation pool of the copy contexts (one per armed request). A recycled
/// context is reseeded by `begin_request`, which drops its history; its
/// tables are reused uncleared exactly as the singleton reuses them across
/// requests (a stale slot can never propose).
pub(crate) struct CopyPool {
    config: NgramModConfig,
    free: Vec<MtpCopyContext>,
}

impl CopyPool {
    /// Validates `config` by building the first context.
    pub(crate) fn new(config: NgramModConfig) -> Result<Self, String> {
        let first = MtpCopyContext::new(config).map_err(str::to_string)?;
        Ok(Self {
            config,
            free: vec![first],
        })
    }

    /// A fresh request-local copy lane: history = the FULL `prompt` plus the
    /// admission `seed` (the first emitted token), indexed.
    pub(crate) fn checkout(&mut self, prompt: &[u32], seed: u32) -> Result<CopyLane, String> {
        let mut ctx = match self.free.pop() {
            Some(ctx) => ctx,
            None => MtpCopyContext::new(self.config).map_err(str::to_string)?,
        };
        ctx.begin_request(prompt);
        let stream = vec![seed];
        ctx.sync_emitted(&stream);
        Ok(CopyLane {
            ctx,
            stream,
            shadow: CopyShadow::new(copy_depths(&self.config)),
        })
    }

    /// Return a released/cancelled request's context (history dropped).
    pub(crate) fn recycle(&mut self, lane: CopyLane) {
        let mut ctx = lane.ctx;
        ctx.reset_request();
        self.free.push(ctx);
    }
}

/// Depth entries of a request's copy acceptance table.
fn copy_depths(config: &NgramModConfig) -> usize {
    config.n_max.min(NGRAM_DEPTHS).max(1)
}

/// The mutable per-request copy state a window changes: staged in the step,
/// published with the rest of the root state.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CopyShadow {
    /// The copy source may still take windows. Cleared, for the rest of the
    /// request, when the floor reports takeovers costing no less per token
    /// than AR ([`LaneSpecPolicy::observe_external_copy`]).
    pub(crate) active: bool,
    /// Decayed (accepted, compared) counts of the source per candidate
    /// depth: its own acceptance, apart from the native agreement.
    pub(crate) agreement: Vec<(f32, f32)>,
    /// The last shadow observation accepted its first candidate: the source
    /// is still on the text being emitted.
    pub(crate) last_first_accepted: bool,
}

impl CopyShadow {
    fn new(depths: usize) -> Self {
        Self {
            active: true,
            agreement: vec![AGREEMENT_PRIOR; depths],
            last_first_accepted: false,
        }
    }

    /// Compare the window's whole pool `proposal` with the tokens `truth`
    /// the window committed, whichever source ran it (censored at the
    /// committed length).
    pub(crate) fn observe(&mut self, proposal: &[u32], truth: &[u32]) {
        if truth.is_empty() {
            return;
        }
        let (matched, horizon) = copy_shadow_counts(proposal, truth);
        observe_agreement_counts(&mut self.agreement, matched, horizon, DEFAULT_DECAY);
        self.last_first_accepted = matched >= 1;
    }
}

/// One window's pool proposal and the root's decision on it, staged in the
/// step (never part of the lane until the window publishes its shadow).
#[derive(Clone, Debug)]
pub(crate) struct CopyProposal {
    /// The whole pool proposal (up to `n_max`), kept for the shadow
    /// observation.
    pub(crate) proposal: Vec<u32>,
    /// The chooser's decision (`take == 0`: declined).
    pub(crate) decision: NgramDecision,
    /// The candidates the window verifies (the take, cut after the first
    /// EOS); empty when declined.
    pub(crate) drafts: Vec<u32>,
}

/// A request's copy lane.
pub(crate) struct CopyLane {
    ctx: MtpCopyContext,
    /// `[admission seed] ++ retained output`: the emitter's stream, whose
    /// last token is the pending seed.
    stream: Vec<u32>,
    /// The committed shadow state.
    pub(crate) shadow: CopyShadow,
}

impl CopyLane {
    /// `prompt ++ [seed] ++ retained output`.
    pub(crate) fn history(&self) -> &[u32] {
        self.ctx.history()
    }

    /// Append a published window's committed tokens to the retained output
    /// and index them.
    pub(crate) fn append(&mut self, emitted: &[u32]) {
        self.stream.extend_from_slice(emitted);
        self.ctx.sync_emitted(&self.stream);
    }

    /// Ask the pool for a proposal over the authoritative history (the
    /// pending seed is its last token) and run the take-length chooser on
    /// `shadow`, the request's (possibly staged) state. `take_cap` is the
    /// admitted copy depth cap, `max_drafts` the window's draft room (remaining
    /// tokens, context, row-exact width). `None` when the source may not
    /// take the window (inactive, the floor's calibration pending, no
    /// proposal); the chooser may still decline (`drafts` empty), leaving the
    /// proposal for the shadow observation as the singleton does.
    pub(crate) fn offer(
        &mut self,
        shadow: &CopyShadow,
        policy: &LaneSpecPolicy,
        lab: LabRoutes,
        take_cap: usize,
        max_drafts: usize,
        eos: Option<u32>,
    ) -> Option<CopyProposal> {
        if !shadow.active || policy.copy_blocked(lab) {
            return None;
        }
        let config = *self.ctx.config();
        // The extra length past the take is only observed, never taken.
        let proposal = self.ctx.propose(&self.stream, config.n_max)?.to_vec();
        let cap = take_cap.min(max_drafts).min(shadow.agreement.len());
        let decision = copy_take_decision(
            &CopyTakeInput {
                available: proposal.len(),
                cap,
                n_min: config.n_min,
                agreement: &shadow.agreement,
                retired: policy.retired(),
                last_first_accepted: shadow.last_first_accepted,
            },
            || policy.native_rate(max_drafts),
            |d| policy.takeover_cost(d),
        );
        let mut drafts = Vec::new();
        if decision.take > 0 {
            let mut take = decision.take;
            if let Some(eos) = eos {
                if let Some(i) = proposal[..take].iter().position(|&t| t == eos) {
                    take = i + 1;
                }
            }
            drafts.extend_from_slice(&proposal[..take]);
        }
        Some(CopyProposal {
            proposal,
            decision,
            drafts,
        })
    }
}

/// What one decided window does to a lane's head, root-side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HeadOutcome {
    /// Where the head ends (the decision's `head_end`).
    pub(crate) end: HeadEnd,
    /// Teacher-forced rows the head owners append in `prepare_commit`.
    pub(crate) append: HeadAppend,
    /// The lane's head state after publish.
    pub(crate) head: HeadState,
    /// The lane's owed-row record after publish.
    pub(crate) lag: HeadLag,
}

/// One window's head-side facts.
pub(crate) struct HeadWindow<'a> {
    /// The head's state at the window's start: after the plan's catch-up,
    /// when it scheduled one (an `Aligned` head with nothing owed).
    pub(crate) head: HeadState,
    /// The owed rows at the window's start (empty unless `head` is `Lag`).
    pub(crate) lag: &'a HeadLag,
    pub(crate) position: usize,
    pub(crate) seed: u32,
    pub(crate) consumed: usize,
    /// The window runs the head's own rows (native, copy takeover, ArSynced
    /// or forced): its consumed rows are teacher-forced and the head ends
    /// aligned. `false`: a target-only retired AR token that owes its head
    /// row.
    pub(crate) live: bool,
    /// Fixed capacity of the owed-row record (`head_lag_rows`, at least 1).
    pub(crate) capacity: usize,
}

/// The singleton's `HeadLag` transitions as root bookkeeping
/// ([`HeadLag::push`], flush at capacity, flush before any takeover or
/// forced realignment): which rows the head owners append, where the head
/// ends, and the record that follows.
///
/// A retired lane with no copy future never reaches here (its plan head is
/// `Free`: no lag hidden copies).
pub(crate) fn head_outcome(w: &HeadWindow<'_>) -> HeadOutcome {
    let end_row = w.position + w.consumed;
    if w.head == HeadState::Free {
        return HeadOutcome {
            end: HeadEnd::Free,
            append: HeadAppend::default(),
            head: HeadState::Free,
            lag: HeadLag::default(),
        };
    }
    let owed_before = if matches!(w.head, HeadState::Lag { .. }) {
        w.lag.tokens.len()
    } else {
        0
    };
    if w.live {
        // Flush before the takeover / forced realignment / native window
        // (the owed rows first), then the window's retained rows.
        return HeadOutcome {
            end: HeadEnd::Aligned { end: end_row },
            append: HeadAppend {
                owed: owed_before,
                retained: w.consumed,
            },
            head: HeadState::Aligned,
            lag: HeadLag::default(),
        };
    }
    // A retired AR token: the target forwards alone and the head owes its
    // row (token `p` with the target hidden of `p - 1`).
    debug_assert_eq!(w.consumed, 1, "an AR window consumes one row");
    let mut book = if owed_before > 0 {
        w.lag.clone()
    } else {
        HeadLag::default()
    };
    book.push(w.position, w.seed);
    if book.tokens.len() >= w.capacity.max(1) {
        // A full record catches the head up at once.
        HeadOutcome {
            end: HeadEnd::Aligned { end: end_row },
            append: HeadAppend {
                owed: owed_before,
                retained: w.consumed,
            },
            head: HeadState::Aligned,
            lag: HeadLag::default(),
        }
    } else {
        HeadOutcome {
            end: HeadEnd::Lag {
                from: book.start,
                to: end_row,
            },
            append: HeadAppend::default(),
            head: HeadState::Lag { from: book.start },
            lag: book,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(ratio: f32, weight: f32) -> Vec<(f32, f32)> {
        vec![(ratio * weight, weight); 7]
    }

    fn input<'a>(
        agreement: &'a [(f32, f32)],
        retired: bool,
        last_first_accepted: bool,
    ) -> CopyTakeInput<'a> {
        CopyTakeInput {
            available: 7,
            cap: 7,
            n_min: 2,
            agreement,
            retired,
            last_first_accepted,
        }
    }

    #[test]
    fn mesh_lane_protocol_copy_chooser_is_the_singletons_and_retired_lanes_compare_to_ar() {
        let cost = |d: usize| crate::mtp_spec::MTP_WINDOW_COST[d];
        let t = table(0.99, 20.0);
        // A retired request's alternative is one AR row (1.0), and it only
        // takes while the last shadow observation accepted its first
        // candidate.
        let cold = copy_take_decision(&input(&t, true, false), || 99.0, cost);
        assert_eq!(cold.take, 0, "retired and cold: declined");
        let warm = copy_take_decision(&input(&t, true, true), || 99.0, cost);
        assert!(warm.take >= 2, "{warm:?}");
        assert_eq!(warm.native_rate, 1.0, "AR is the comparator, not native");
        // A live request compares with the native window.
        let live = copy_take_decision(&input(&t, false, true), || 99.0, cost);
        assert_eq!(live.take, 0, "native at 99 tokens per unit wins");
        assert_eq!(live.native_rate, 99.0);
        // Established: acceptance, evidence, first-candidate and take length.
        assert!(
            !warm.confirm,
            "an established source omits the confirmation"
        );
        let fresh = table(0.8, 2.0);
        let d = copy_take_decision(&input(&fresh, false, false), || 1.0, cost);
        assert!(d.confirm, "an unestablished source must confirm: {d:?}");
        let d = copy_take_decision(&input(&t, false, false), || 1.0, cost);
        assert!(d.confirm, "never established without a first acceptance");
    }

    #[test]
    fn mesh_lane_protocol_head_outcome_transitions_mirror_the_singletons_head_lag() {
        let empty = HeadLag::default();
        let w = |head, lag: &HeadLag, live, capacity| {
            HeadWindow {
                head,
                lag,
                position: 40,
                seed: 9,
                consumed: if live { 3 } else { 1 },
                live,
                capacity,
            }
            .clone_outcome()
        };
        // Head-free: nothing, ever.
        let o = w(HeadState::Free, &empty, true, 4);
        assert_eq!((o.end, o.head), (HeadEnd::Free, HeadState::Free));
        // A live window on an aligned head: the retained rows only.
        let o = w(HeadState::Aligned, &empty, true, 4);
        assert_eq!(
            o.append,
            HeadAppend {
                owed: 0,
                retained: 3
            }
        );
        assert_eq!(o.end, HeadEnd::Aligned { end: 43 });
        // A retired AR token owes its row; the head is not touched.
        let o = w(HeadState::Aligned, &empty, false, 4);
        assert_eq!(o.append, HeadAppend::default());
        assert_eq!(o.end, HeadEnd::Lag { from: 40, to: 41 });
        assert_eq!((o.lag.start, o.lag.tokens.clone()), (40, vec![9]));
        // It keeps growing, contiguous from the head position...
        let lag = o.lag.clone();
        let mut next = HeadWindow {
            head: o.head,
            lag: &lag,
            position: 41,
            seed: 10,
            consumed: 1,
            live: false,
            capacity: 4,
        };
        let o2 = head_outcome(&next);
        assert_eq!(o2.end, HeadEnd::Lag { from: 40, to: 42 });
        assert_eq!(o2.lag.tokens, vec![9, 10]);
        // ... a takeover flushes the owed rows first ...
        next.live = true;
        next.consumed = 4;
        let o3 = head_outcome(&next);
        assert_eq!(
            o3.append,
            HeadAppend {
                owed: 1,
                retained: 4
            }
        );
        assert_eq!((o3.head, o3.lag.tokens.len()), (HeadState::Aligned, 0));
        // ... and a full record flushes at capacity (one batched fill).
        let o4 = head_outcome(
            &HeadWindow {
                capacity: 2,
                ..next
            }
            .as_ar(),
        );
        assert_eq!(
            o4.append,
            HeadAppend {
                owed: 1,
                retained: 1
            }
        );
        assert_eq!(o4.end, HeadEnd::Aligned { end: 42 });
    }

    impl<'a> HeadWindow<'a> {
        fn clone_outcome(self) -> HeadOutcome {
            head_outcome(&self)
        }

        fn as_ar(mut self) -> Self {
            self.live = false;
            self.consumed = 1;
            self
        }
    }
}
