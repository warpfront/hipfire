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

    pub(crate) fn config(&self) -> &NgramModConfig {
        &self.config
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
