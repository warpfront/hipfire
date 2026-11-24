// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Speculative decode on the engine loop (G12 P3), over the shared
//! [`Speculator`] / [`SpecTarget`] seam.
//!
//! A speculative request's [`RequestRun`](super::RequestRun) owns its
//! [`Drafting`]: the drafter itself (moved out of the load for the request's
//! life) and the window state of the `generate_spec` loop. It names
//! [`Work::Verify`](super::Work::Verify) (one window, or the drafter's AR step
//! once retired) and [`Work::Advance`](super::Work::Advance) (forced template
//! tokens, the terminal flush); [`DraftForwards`] runs them on a single-card
//! target. Window semantics follow the legacy loop: a window starts at the
//! target's `position` on the last committed token (its seed, not yet run),
//! commits `SpecStep::emit`, and its last token is the next seed. A window the
//! emitter stops or forces strictly inside is trimmed back to its kept prefix
//! ([`Trim`]): repaired in place, else reset (a terminal) or re-prefilled (a
//! continuing request).

use super::Forwards;
use crate::qwen::{
    spec_forced_advance, spec_prefill, spec_repair_terminal_prefix, spec_reset_for_realign,
};
use hipfire_engine::terminal::check_abort;
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::session_cache::SessionRoute;
use hipfire_runtime::spec::{
    FloorControl, MtpRequestStats, PrefillOutcome, SpecAdvance, SpecRequestConfig, SpecStep,
    SpecTarget, Speculator,
};
use hipfire_runtime::spec_retire::{BreakEven, SpecRetire};
use std::ops::Range;
use std::time::Instant;

/// A request's drafter and its window state.
pub struct Drafting {
    pub(super) drafter: Box<dyn Speculator>,
    /// Committed tokens the target has not run yet, oldest first; the last is
    /// the next window's seed.
    pub(super) owed: Vec<u32>,
    /// Decode tokens committed so far: the drafter's repeat / n-gram context.
    pub(super) emitted: Vec<u32>,
    pub(super) retire: SpecRetire,
    pub(super) tally: Tally,
    /// An unrepaired terminal window reset the live state: nothing to flush,
    /// nothing to keep live.
    pub(super) invalidated: bool,
}

impl Drafting {
    /// `retire`: the spec → AR count line (`None` never retires).
    pub fn new(drafter: Box<dyn Speculator>, retire: Option<BreakEven>) -> Self {
        Self {
            drafter,
            owed: Vec::new(),
            emitted: Vec::new(),
            retire: SpecRetire::new(retire),
            tally: Tally::default(),
            invalidated: false,
        }
    }
}

/// A speculative request's windows, for its `done`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    /// Speculative windows (AR steps excluded).
    pub windows: usize,
    /// Drafts accepted over every window.
    pub accepted: usize,
    /// The drafter's AR steps after retirement.
    pub ar_steps: usize,
    /// Position of the retirement to AR.
    pub retired_at: Option<usize>,
}

/// What [`Forwards::trim`] left of a window committed past its kept prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trim {
    /// Target and drafter hold the pre-window state plus the kept prefix.
    Repaired,
    /// A terminal the drafter could not repair: the live state was reset.
    Reset,
    /// A continuing request re-prefilled its committed history.
    Realigned,
}

/// A speculative route's request: the drafter's configuration and the
/// route's `done` fields.
#[derive(Clone)]
pub struct SpecRequest {
    pub config: SpecRequestConfig,
    /// Fold the route's speculative fields into `done`.
    pub stamp: fn(&mut serde_json::Value, &SpecDone),
}

/// A finished speculative request, for its route's `done` stamp.
pub struct SpecDone {
    pub drafter: &'static str,
    pub tally: Tally,
    /// The drafter's proposal width.
    pub block: usize,
    pub stats: MtpRequestStats,
}

impl SpecDone {
    pub fn of(d: &Drafting) -> Self {
        Self {
            drafter: d.drafter.name(),
            tally: Tally {
                retired_at: d.retire.retired_at(),
                ..d.tally
            },
            block: d.drafter.block_size(),
            stats: d.drafter.request_stats(),
        }
    }
}

/// A speculative request's start on the resident target, before anything
/// runs: the reused prompt prefix and the drafter's session route.
pub struct Plan {
    pub reused: usize,
    pub route: Option<SessionRoute>,
    retire: Option<BreakEven>,
}

impl Plan {
    /// The drafter takes the request's configuration; the plan is the
    /// deepest session point it resumes (a target without a speculative
    /// session plan resumes the drafter's `Mtp` snapshots) and the spec → AR
    /// count line (greedy, or an AR step that samples, on a drafter with an
    /// AR step; a producer-owned floor decides inside its own step). `Err`:
    /// the engine cannot serve this drafter (nothing changed).
    pub fn new(
        model: &dyn ArchModel,
        drafter: &mut dyn Speculator,
        prompt: &[u32],
        config: &SpecRequestConfig,
        temperature: f32,
    ) -> Result<Self, String> {
        let producer_floor = match drafter.floor_control() {
            FloorControl::Driver(_) => {
                return Err("a driver-owned spec floor keeps the legacy loop".into())
            }
            FloorControl::Producer => true,
            FloorControl::Unsupported => false,
        };
        let route = drafter.session_route();
        let reused = if model.spec_session_reuse() {
            crate::qwen::plan_spec_session(model, &*drafter, prompt).map_or(0, |(_, start)| start)
        } else {
            model.session_plan(prompt, route.unwrap_or(SessionRoute::Mtp))
        };
        drafter.configure_request(*config);
        let retire = (!producer_floor
            && (temperature <= 1e-6 || drafter.supports_sampled_ar_step())
            && drafter.supports_ar_step())
        .then(|| {
            crate::qwen::spec_retire_break_even(
                hipfire_config::developer_var("HIPFIRE_SPEC_RETIRE")
                    .ok()
                    .as_deref(),
                drafter.retire_break_even(),
            )
        })
        .flatten();
        Ok(Self {
            reused,
            route,
            retire,
        })
    }

    /// The request's [`Drafting`] over the drafter it now owns.
    pub fn drafting(&self, drafter: Box<dyn Speculator>) -> Drafting {
        Drafting::new(drafter, self.retire)
    }
}

/// A client abort of a speculative request past its prompt: a target that
/// keeps sessions keeps the `fed` rows of `history` it ran live (target and
/// drafter sit at that row between forwards), as the legacy cancel does, so
/// the next turn continues it. `false`: nothing kept, the caller rolls back.
pub fn keep_fed(
    state: &mut dyn ArchModel,
    gpus: &mut hipfire_runtime::multi_gpu::Gpus,
    history: &[u32],
    fed: usize,
    prompt_len: usize,
) -> bool {
    if !state.spec_session_reuse() || fed < prompt_len || fed > history.len() {
        return false;
    }
    if !crate::common::fail_closed_device_sync(&mut gpus.devices[0]).rolled_back {
        return false;
    }
    state.session_commit_live_mesh(gpus, &history[..fed]);
    true
}

/// The engine's speculative forwards on a single-card target: the legacy
/// loop's prefill, window, repair, realign and forced-advance calls.
pub struct DraftForwards<'a> {
    pub gpu: &'a mut rdna_compute::Gpu,
    pub target: &'a mut dyn SpecTarget,
    pub id: &'a str,
    pub prompt: &'a [u32],
    /// The session route the drafter resumes on (it begins its own on `Mtp`;
    /// the target begins an `Ar` one).
    pub route: Option<SessionRoute>,
    /// Decode tokens the request may emit (maps the target's context ahead).
    pub budget: usize,
    pub temperature: f32,
    /// When the prompt finished (the prefill/decode split).
    pub prompt_done: &'a mut Option<Instant>,
    /// A batch request's attempt, whose scope the prompt's progress reports
    /// to; `None`: the daemon's request scope.
    pub attempt: Option<u64>,
}

impl Forwards for DraftForwards<'_> {
    fn prefill(&mut self, _rows: Range<usize>) -> Result<(usize, Option<Vec<f32>>), String> {
        Err("a speculative prompt runs through its drafter".into())
    }

    fn decode(&mut self, _token: Option<u32>, _top1: bool) -> Result<(u32, super::Next), String> {
        Err("a speculative request decodes in windows".into())
    }

    fn draft_prefill(
        &mut self,
        d: &mut dyn Speculator,
        start: usize,
    ) -> Result<Option<u32>, String> {
        let id = self.id;
        let _progress = self
            .attempt
            .map(|attempt| hipfire_runtime::prefill_progress::scope(id, Some(attempt)));
        let prompt = self.prompt;
        // The request writes the prompt, its budget and one window past the
        // last seed (one-shot advisory for the session begin inside prefill).
        self.target.session_request_write_limit(
            prompt
                .len()
                .saturating_add(self.budget)
                .saturating_add(d.block_size()),
        );
        // Two-stage prefill: the first token returns before the drafter's
        // last head fill ([`Forwards::finish_prefill`]).
        d.defer_prefill_tail(true);
        let outcome = spec_prefill(
            None,
            self.gpu,
            d,
            self.target,
            self.route,
            prompt,
            &prompt[start..],
            start,
            start > 0,
            None,
            &|| check_abort(id),
        );
        d.defer_prefill_tail(false);
        match outcome? {
            PrefillOutcome::Ready { first_token } => {
                self.target.map_ahead_for_decode(
                    self.gpu,
                    prompt.len(),
                    self.budget,
                    d.block_size(),
                );
                *self.prompt_done = Some(Instant::now());
                Ok(Some(first_token))
            }
            PrefillOutcome::Aborted => Ok(None),
        }
    }

    fn finish_prefill(&mut self, d: &mut dyn Speculator) -> Result<(), String> {
        d.finish_prefill(self.gpu, self.target)
    }

    fn verify(
        &mut self,
        d: &mut dyn Speculator,
        position: usize,
        seed: u32,
        emitted: &[u32],
        max_emit: usize,
        ar: bool,
    ) -> Result<SpecStep, String> {
        if ar {
            return d
                .ar_step(self.gpu, self.target, position, seed, emitted)
                .map(|token| SpecStep::new([token], token, 0, 0));
        }
        d.step(
            self.gpu,
            self.target,
            position,
            seed,
            emitted,
            None,
            self.temperature,
            max_emit,
        )
    }

    fn advance(
        &mut self,
        d: &mut dyn Speculator,
        tokens: &[u32],
        position: usize,
    ) -> Result<(), String> {
        let id = self.id;
        let abort = || check_abort(id);
        if spec_forced_advance(None, self.gpu, d, self.target, tokens, position, &abort)? {
            return Ok(());
        }
        // An abort here ends the request at the next poll; its rollback
        // resets the partly advanced state.
        match self
            .target
            .spec_advance(self.gpu, tokens, position, false, &abort, None)?
        {
            SpecAdvance::Ready { .. } | SpecAdvance::Aborted => Ok(()),
        }
    }

    fn trim(
        &mut self,
        d: &mut dyn Speculator,
        start: usize,
        seed: u32,
        kept: &[u32],
        ar: bool,
        replay: Option<&[u32]>,
    ) -> Result<Trim, String> {
        // An AR step retains no pre-window state to restore.
        if !ar && spec_repair_terminal_prefix(None, self.gpu, d, self.target, start, seed, kept)? {
            return Ok(Trim::Repaired);
        }
        let Some(tail) = replay else {
            if let Some(error) = spec_reset_for_realign(None, self.gpu, d, self.target, true) {
                return Err(format!("terminal prefix reset failed: {error}"));
            }
            return Ok(Trim::Reset);
        };
        if let Some(error) = spec_reset_for_realign(None, self.gpu, d, self.target, false) {
            return Err(format!("prefix realign reset failed: {error}"));
        }
        let mut history = Vec::with_capacity(self.prompt.len() + tail.len());
        history.extend_from_slice(self.prompt);
        history.extend_from_slice(tail);
        let id = self.id;
        // The prefill's pick is not history; an abort ends the request at the
        // next poll.
        spec_prefill(
            None,
            self.gpu,
            d,
            self.target,
            self.route,
            &history,
            &history,
            0,
            false,
            None,
            &|| check_abort(id),
        )
        .map_err(|error| format!("spec_prefix_realign: {error}"))?;
        Ok(Trim::Realigned)
    }
}

/// The drafter's name stands for it in a `Debug` print.
impl std::fmt::Debug for Drafting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Drafting")
            .field("drafter", &self.drafter.name())
            .field("owed", &self.owed)
            .field("tally", &self.tally)
            .finish()
    }
}

#[cfg(test)]
pub(crate) use test_drafter::NoDrafter;

#[cfg(test)]
mod test_drafter {
    use super::*;
    use rdna_compute::Gpu;

    /// A drafter the fake forwards of the engine's tests never call: they
    /// model target and drafter themselves.
    pub(crate) struct NoDrafter;

    impl Speculator for NoDrafter {
        fn prefill(
            &mut self,
            _: &mut Gpu,
            _: &mut dyn SpecTarget,
            _: &[u32],
            _: &[u32],
            _: usize,
            _: bool,
            _: Option<usize>,
            _: &dyn Fn() -> bool,
        ) -> Result<PrefillOutcome, String> {
            unreachable!()
        }
        fn step(
            &mut self,
            _: &mut Gpu,
            _: &mut dyn SpecTarget,
            _: usize,
            _: u32,
            _: &[u32],
            _: Option<&mut dyn hipfire_runtime::spec::SpecGrammar>,
            _: f32,
            _: usize,
        ) -> Result<SpecStep, String> {
            unreachable!()
        }
        fn reset(&mut self, _: &mut Gpu) -> Result<(), String> {
            unreachable!()
        }
        fn block_size(&self) -> usize {
            4
        }
        fn ctx_capacity(&self) -> usize {
            1 << 20
        }
        fn free(self: Box<Self>, _: &mut Gpu) {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{drive, Emit, Exit, Finish, RequestRun, Verdict};
    use hipfire_runtime::sampler::SamplerConfig;

    const VOCAB: u64 = 50;
    const EOS: u32 = 49;
    const CLOSE: [u32; 2] = [47, 48];

    /// The target's greedy pick after `state`: a hash of everything it ran.
    fn truth(state: &[u32]) -> u32 {
        let h = state.iter().fold(17u64, |h, &t| {
            h.wrapping_mul(31).wrapping_add(u64::from(t) + 1)
        });
        // Never the EOS or a close token by accident.
        (h % (VOCAB - 3)) as u32
    }

    /// Target and drafter: `ran` is every token the target ran. A window
    /// commits `width` greedy tokens (all accepted but the bonus).
    struct Model {
        prompt: Vec<u32>,
        ran: Vec<u32>,
        width: usize,
        /// A terminal the drafter cannot repair resets; a continuing one
        /// re-prefills.
        repairs: bool,
        trims: Vec<(usize, Vec<u32>, Option<Vec<u32>>)>,
        /// Where each drafter prompt fill started.
        prefills: Vec<usize>,
    }

    impl Forwards for Model {
        fn prefill(&mut self, _: Range<usize>) -> Result<(usize, Option<Vec<f32>>), String> {
            unreachable!()
        }
        fn decode(
            &mut self,
            _: Option<u32>,
            _: bool,
        ) -> Result<(u32, crate::engine::Next), String> {
            unreachable!()
        }
        fn draft_prefill(
            &mut self,
            _: &mut dyn Speculator,
            start: usize,
        ) -> Result<Option<u32>, String> {
            self.prefills.push(start);
            self.ran = self.prompt.clone();
            Ok(Some(truth(&self.ran)))
        }
        fn finish_prefill(&mut self, _: &mut dyn Speculator) -> Result<(), String> {
            Ok(())
        }
        fn verify(
            &mut self,
            _: &mut dyn Speculator,
            position: usize,
            seed: u32,
            _: &[u32],
            max_emit: usize,
            _: bool,
        ) -> Result<SpecStep, String> {
            assert_eq!(
                self.ran.len(),
                position,
                "a window starts where the target is"
            );
            self.ran.push(seed);
            let n = self.width.min(max_emit);
            let mut emit = Vec::new();
            for i in 0..n {
                let t = truth(&self.ran);
                emit.push(t);
                if i + 1 < n {
                    self.ran.push(t);
                }
            }
            let last = *emit.last().unwrap();
            Ok(SpecStep::new(emit, last, n - 1, n - 1))
        }
        fn advance(
            &mut self,
            _: &mut dyn Speculator,
            tokens: &[u32],
            position: usize,
        ) -> Result<(), String> {
            assert_eq!(self.ran.len(), position);
            self.ran.extend_from_slice(tokens);
            Ok(())
        }
        fn trim(
            &mut self,
            _: &mut dyn Speculator,
            start: usize,
            seed: u32,
            kept: &[u32],
            _: bool,
            replay: Option<&[u32]>,
        ) -> Result<Trim, String> {
            self.trims
                .push((start, kept.to_vec(), replay.map(<[u32]>::to_vec)));
            if self.repairs {
                self.ran.truncate(start);
                self.ran.push(seed);
                self.ran.extend_from_slice(&kept[..kept.len() - 1]);
                return Ok(Trim::Repaired);
            }
            Ok(match replay {
                None => {
                    self.ran.clear();
                    Trim::Reset
                }
                Some(tail) => {
                    self.ran = [&self.prompt[..], tail].concat();
                    Trim::Realigned
                }
            })
        }
    }

    /// The wire: `eos_at`-th emission is the EOS id; the `close_at`-th forces
    /// the two close tokens.
    #[derive(Default)]
    struct Wire {
        history: Vec<u32>,
        emitted: Vec<u32>,
        eos_at: Option<usize>,
        close_at: Option<usize>,
        cap: Option<usize>,
    }

    impl Emit for Wire {
        fn commit_prompt(&mut self) {}
        fn emit(&mut self, token: u32) -> Result<(Verdict, Vec<u32>), String> {
            self.emitted.push(token);
            self.history.push(token);
            let n = self.emitted.len();
            if self.eos_at == Some(n) {
                // The fake model never picks the EOS id: substitute it.
                *self.emitted.last_mut().unwrap() = EOS;
                return Ok((Verdict::Eos, Vec::new()));
            }
            let close = if self.close_at == Some(n) {
                CLOSE.to_vec()
            } else {
                Vec::new()
            };
            Ok((Verdict::Continue, close))
        }
        fn history(&self) -> &[u32] {
            &self.history
        }
        fn aborted(&self) -> bool {
            false
        }
        fn window_cap(&self) -> Option<usize> {
            self.cap
        }
    }

    fn greedy() -> SamplerConfig {
        SamplerConfig {
            temperature: 0.0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            repeat_window: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            blocked_tokens: Vec::new(),
            top_k: None,
            min_p: None,
        }
    }

    fn model(width: usize, repairs: bool) -> Model {
        Model {
            prompt: vec![1, 2, 3, 4, 5],
            ran: Vec::new(),
            width,
            repairs,
            trims: Vec::new(),
            prefills: Vec::new(),
        }
    }

    fn run(m: &mut Model, wire: &mut Wire, budget: usize) -> (Exit, RequestRun) {
        run_from(m, wire, budget, 0)
    }

    fn run_from(
        m: &mut Model,
        wire: &mut Wire,
        budget: usize,
        reused: usize,
    ) -> (Exit, RequestRun) {
        let mut run = RequestRun::new(m.prompt.len(), reused, budget, greedy(), 1)
            .with_drafting(Drafting::new(Box::new(NoDrafter), None));
        let exit = drive(&mut run, m, wire);
        (exit, run)
    }

    /// The tokens greedy AR emits: each the target's pick after the prompt
    /// and every earlier emission (forced tokens included).
    fn ar_reference(prompt: &[u32], emitted: &[u32]) -> Vec<u32> {
        let mut state = prompt.to_vec();
        let mut out = Vec::new();
        for &t in emitted {
            if CLOSE.contains(&t) || t == EOS {
                out.push(t);
            } else {
                out.push(truth(&state));
            }
            state.push(t);
        }
        out
    }

    #[test]
    fn a_whole_prompt_reuse_still_runs_the_drafter_prefill_for_the_first_token() {
        let mut m = model(4, true);
        let mut wire = Wire::default();
        let (exit, _) = run_from(&mut m, &mut wire, 6, 5);
        assert_eq!(exit, Exit::Finished(Finish::Length));
        assert_eq!(m.prefills, [5]);
        assert_eq!(wire.emitted, ar_reference(&m.prompt, &wire.emitted));
    }

    #[test]
    fn full_windows_emit_the_ar_stream_and_flush_the_last_seed() {
        let mut m = model(4, true);
        let mut wire = Wire::default();
        let (exit, run) = run(&mut m, &mut wire, 10);
        assert_eq!(exit, Exit::Finished(Finish::Length));
        assert_eq!(wire.emitted, ar_reference(&m.prompt, &wire.emitted));
        assert_eq!(wire.emitted.len(), 10);
        // first token, then windows of 4, 4, 1 (the budget caps the last).
        let d = run.drafting().unwrap();
        assert_eq!((d.tally.windows, d.tally.accepted), (3, 3 + 3));
        assert!(m.trims.is_empty());
        assert_eq!(
            m.ran,
            [&m.prompt[..], &wire.emitted[..]].concat(),
            "the target ends on the history"
        );
    }

    #[test]
    fn a_stop_inside_a_window_is_repaired_and_the_stop_flushed() {
        let mut m = model(4, true);
        let mut wire = Wire {
            eos_at: Some(3),
            ..Wire::default()
        };
        let (exit, run) = run(&mut m, &mut wire, 10);
        assert_eq!(exit, Exit::Finished(Finish::Stop));
        assert_eq!(wire.emitted.len(), 3);
        assert_eq!(run.generated(), 2, "EOS is not counted");
        assert_eq!(m.trims.len(), 1);
        assert_eq!(m.trims[0].2, None, "a terminal trims without a replay");
        // The EOS committed like any token (the wire's history holds the id
        // the model ran).
        assert_eq!(m.ran, [&m.prompt[..], &wire.history[..]].concat());
        assert!(!run.drafting().unwrap().invalidated);
    }

    #[test]
    fn an_unrepairable_terminal_resets_and_flushes_nothing() {
        let mut m = model(4, false);
        let mut wire = Wire {
            eos_at: Some(3),
            ..Wire::default()
        };
        let (exit, run) = run(&mut m, &mut wire, 10);
        assert_eq!(exit, Exit::Finished(Finish::Stop));
        assert!(m.ran.is_empty(), "no flush onto the reset state");
        assert!(run.drafting().unwrap().invalidated);
    }

    #[test]
    fn a_close_at_a_window_end_runs_ahead_of_the_next_window() {
        let mut m = model(4, true);
        // Emission 5 ends the second window (1 + 4).
        let mut wire = Wire {
            close_at: Some(5),
            ..Wire::default()
        };
        let (exit, _) = run(&mut m, &mut wire, 12);
        assert_eq!(exit, Exit::Finished(Finish::Length));
        assert_eq!(&wire.emitted[5..7], &CLOSE);
        assert_eq!(wire.emitted, ar_reference(&m.prompt, &wire.emitted));
        assert!(m.trims.is_empty());
        assert_eq!(m.ran, [&m.prompt[..], &wire.emitted[..]].concat());
    }

    #[test]
    fn a_close_inside_a_window_realigns_when_unrepairable() {
        let mut m = model(4, false);
        let mut wire = Wire {
            close_at: Some(3),
            ..Wire::default()
        };
        let (exit, _) = run(&mut m, &mut wire, 12);
        assert_eq!(exit, Exit::Finished(Finish::Length));
        assert_eq!(&wire.emitted[3..5], &CLOSE);
        assert_eq!(wire.emitted, ar_reference(&m.prompt, &wire.emitted));
        // The realign replays every committed token but the seed.
        assert_eq!(m.trims[0].2.as_deref(), Some(&wire.emitted[..2]));
        assert_eq!(m.ran, [&m.prompt[..], &wire.emitted[..]].concat());
    }

    #[test]
    fn the_window_cap_keeps_a_close_on_a_window_end() {
        let mut m = model(4, false);
        let mut wire = Wire {
            close_at: Some(3),
            cap: Some(2),
            ..Wire::default()
        };
        let (exit, _) = run(&mut m, &mut wire, 12);
        assert_eq!(exit, Exit::Finished(Finish::Length));
        assert!(m.trims.is_empty(), "windows of 2 end exactly on the close");
        assert_eq!(wire.emitted, ar_reference(&m.prompt, &wire.emitted));
    }
}
