// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! G12 engine loop (`generation.engine_loop = engine`), model- and
//! hardware-agnostic.
//!
//! [`RequestRun`] is one request's state machine. It never touches a GPU:
//! [`RequestRun::next_work`] names the next forward and the `apply_*` calls
//! fold each forward's result back in (sample with the request's own RNG,
//! count, stop, finish). [`step`] runs one forward of it over two seams:
//! [`Forwards`] (the model on its mesh, [`mesh::MeshForwards`] over any
//! `MeshFamily`) and [`Emit`] (the wire protocol, [`chatml::ChatMlStream`]).
//! [`drive`] steps one request to its end; [`run::run`] owns its admission,
//! the budget, metrics, the terminal and the one attested rollback for every
//! error and abort exit. [`sched::Scheduler`] steps N requests at once
//! (mixed prompt and decode epochs); [`serve::drive`] serves it to the
//! daemon's continuous batch. A speculative request owns its drafter
//! ([`spec::Drafting`]) and decodes in windows ([`Work::Verify`]).
//!
//! Intended differences from the legacy loops: the RNG is per request (not
//! the process-global sampler state), an EOS token is not counted in
//! `tokens`, and a stop on the last budget token finishes `stop`.

pub mod chatml;
pub mod gemma;
pub mod mesh;
pub mod run;
pub mod sched;
pub mod serve;
pub mod spec;

use hipfire_runtime::sampler::{apply_logit_policy_cpu, SamplerConfig};
use hipfire_runtime::spec::{SpecStep, Speculator};
use std::collections::VecDeque;
use std::ops::Range;

/// Whether `route` runs on the engine loop in this process: the flag is
/// `engine` and the route is on the allowlist.
pub fn runs_on_engine(route: crate::ar::GenerationRoute) -> bool {
    hipfire_runtime::config::get().engine_loop && ENGINE_ROUTES.contains(&route)
}

const ENGINE_ROUTES: &[crate::ar::GenerationRoute] = &[
    crate::ar::GenerationRoute::Qwen4Ar,
    crate::ar::GenerationRoute::Qwen4Spec,
    crate::ar::GenerationRoute::Gemma4Ar,
];

/// The budget a request may decode, or the `context_length` refusal, decided
/// before `gen_start`: `prompt + max_tokens + 1` trailer row must fit `max_seq`.
pub fn admit_budget(
    prompt_len: usize,
    max_tokens: usize,
    max_seq: usize,
    fit: impl Fn(usize, usize, usize) -> usize,
) -> Result<usize, String> {
    if max_tokens == 0 {
        return Err("max_tokens must be > 0".into());
    }
    let budget = fit(max_tokens, prompt_len + 1, max_seq);
    let required = prompt_len
        .checked_add(budget)
        .and_then(|n| n.checked_add(1));
    // A fit that leaves no decode row is a refusal too.
    if budget == 0 || required.is_none_or(|n| n > max_seq) {
        return Err(format!(
            "request exceeds context window: prompt={prompt_len} + max_tokens={max_tokens} + trailer=1 > max_seq={max_seq}"
        ));
    }
    Ok(budget)
}

/// The next forward of a request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Work {
    /// Prompt rows still to prefill; the family runs one tile from the start.
    Prefill {
        rows: Range<usize>,
    },
    /// Run `token` (KV write), then it is emitted. `None`: the device argmax
    /// of the last row, known on the host once the forward ran.
    Decode {
        token: Option<u32>,
    },
    /// One speculative window from the target's position on `seed`, the
    /// last committed token (not yet run), committing at most `max_emit`
    /// tokens; with `ar` the drafter's one-token AR step (retired).
    Verify {
        seed: u32,
        max_emit: usize,
        ar: bool,
    },
    /// Run committed `tokens` through target and drafter: forced template
    /// tokens ahead of the seed, or the terminal flush.
    Advance {
        tokens: Vec<u32>,
    },
    Done(Finish),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    Stop,
    Length,
}

impl Finish {
    /// The one finish rule: a natural stop (EOS, stop sequence, filter, think
    /// budget) is `stop` even on the last budget token; only an exhausted
    /// budget without one is `length`.
    pub fn of(natural_stop: bool) -> Self {
        if natural_stop {
            Self::Stop
        } else {
            Self::Length
        }
    }

    pub fn reason(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

/// What a decode forward left for choosing the next token.
pub enum Next {
    /// Device argmax of the row, read back (plain greedy).
    Top1(u32),
    /// The row, for the host sampler.
    Logits(Vec<f32>),
    /// The row stays on the device; the next decode runs its argmax.
    Device,
}

/// How the driver classified the token it just emitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Continue,
    /// EOS id or terminator: ends the turn, not counted in `tokens`.
    Eos,
    /// Stop sequence, filter stop or a spent think budget: counted.
    Stop,
}

pub struct RequestRun {
    prompt_len: usize,
    position: usize,
    budget: usize,
    generated: usize,
    /// The next decode's input: `Some(Some(t))` a host token, `Some(None)`
    /// the device argmax of the last row.
    pending: Option<Option<u32>>,
    /// Template tokens forced ahead of sampling (think-budget close).
    forced: VecDeque<u32>,
    finish: Option<Finish>,
    sampler: SamplerConfig,
    rng: u32,
    /// Prompt rows resident when this prefill (re)started: progress counts
    /// from here.
    prompt_base: usize,
    /// A contained prompt fault already reran the prompt (once at most).
    reran: bool,
    /// A speculative request's drafter and windows.
    draft: Option<spec::Drafting>,
}

impl RequestRun {
    /// `reused` prompt rows are already resident; `budget` is from
    /// [`admit_budget`]; `seed` seeds this request's sampler RNG.
    pub fn new(
        prompt_len: usize,
        reused: usize,
        budget: usize,
        sampler: SamplerConfig,
        seed: u32,
    ) -> Self {
        Self {
            prompt_len,
            position: reused,
            budget,
            generated: 0,
            pending: None,
            forced: VecDeque::new(),
            finish: None,
            sampler,
            // xorshift32 has a fixed point at 0.
            rng: seed.max(1),
            prompt_base: reused,
            reran: false,
            draft: None,
        }
    }

    /// A speculative request: it decodes in windows of `drafting`'s drafter,
    /// which it owns until [`Self::into_drafting`].
    pub fn with_drafting(mut self, drafting: spec::Drafting) -> Self {
        self.draft = Some(drafting);
        self
    }

    pub fn drafting(&self) -> Option<&spec::Drafting> {
        self.draft.as_ref()
    }

    pub fn into_drafting(self) -> Option<spec::Drafting> {
        self.draft
    }

    pub fn next_work(&self) -> Work {
        if let Some(d) = &self.draft {
            return match self.finish {
                Some(finish) if d.owed.is_empty() => Work::Done(finish),
                Some(_) => Work::Advance {
                    tokens: d.owed.clone(),
                },
                // Nothing is committed before the drafter's prompt fill, which
                // runs even when the session restores the whole prompt (it
                // then only picks the first token).
                None if d.owed.is_empty() => Work::Prefill {
                    rows: self.position..self.prompt_len,
                },
                None if d.owed.len() > 1 => Work::Advance {
                    tokens: d.owed[..d.owed.len() - 1].to_vec(),
                },
                None => Work::Verify {
                    seed: d.owed[0],
                    max_emit: self.remaining(),
                    ar: d.retire.is_retired(),
                },
            };
        }
        if let Some(finish) = self.finish {
            Work::Done(finish)
        } else if self.position < self.prompt_len {
            Work::Prefill {
                rows: self.position..self.prompt_len,
            }
        } else {
            Work::Decode {
                token: self.pending.expect("a decoding request has its next token"),
            }
        }
    }

    /// Whether the next token comes from the device argmax: plain greedy, no
    /// penalty or blocked token, and no forced token queued.
    pub fn wants_top1(&self) -> bool {
        self.forced.is_empty() && self.greedy()
    }

    /// Whether every token this request picks is the row's argmax (forced
    /// template tokens aside): no temperature, penalty or blocked token.
    pub fn greedy(&self) -> bool {
        self.sampler.temperature <= 0.0
            && self.sampler.blocked_tokens.is_empty()
            && hipfire_runtime::sampler::PenaltyTable::flags_for(&self.sampler) == 0
    }

    /// Decode rows this request may still run.
    pub fn remaining(&self) -> usize {
        self.budget - self.generated
    }

    /// A prefill tile ended at `end`; the final one hands its logits row,
    /// which picks the first token against `history`.
    pub fn apply_prefill(
        &mut self,
        end: usize,
        logits: Option<Vec<f32>>,
        history: &[u32],
    ) -> Result<(), String> {
        if end <= self.position || end > self.prompt_len {
            return Err(format!(
                "prefill tile ended at {end}, outside {}..={}",
                self.position + 1,
                self.prompt_len
            ));
        }
        self.position = end;
        match (end == self.prompt_len, logits) {
            (true, Some(mut row)) => {
                self.pending = Some(Some(self.sample(&mut row, history)));
                Ok(())
            }
            (false, None) => Ok(()),
            _ => Err("only the final prefill tile carries logits".into()),
        }
    }

    /// The decode of the pending token finished and the driver emitted it as
    /// `verdict`. `close` queues template tokens (think-budget close) that run
    /// before anything is sampled again.
    pub fn apply_decode(&mut self, next: Next, verdict: Verdict, close: &[u32], history: &[u32]) {
        self.pending = None;
        if verdict != Verdict::Eos {
            self.generated += 1;
        }
        if verdict != Verdict::Continue {
            self.finish = Some(Finish::of(true));
            return;
        }
        if self.generated >= self.budget {
            self.finish = Some(Finish::of(false));
            return;
        }
        self.forced.extend(close);
        self.pending = Some(match (self.forced.pop_front(), next) {
            (Some(token), _) => Some(token),
            (None, Next::Top1(token)) => Some(token),
            (None, Next::Logits(mut row)) => Some(self.sample(&mut row, history)),
            (None, Next::Device) => None,
        });
    }

    /// Completion tokens counted so far (EOS excluded).
    pub fn generated(&self) -> usize {
        self.generated
    }

    /// The prompt reruns from `start` (a contained prompt fault): once, and
    /// only while still prefilling.
    pub fn restart_prefill(&mut self, start: usize) -> Result<(), String> {
        if self.reran || self.pending.is_some() || start >= self.prompt_len {
            return Err(format!(
                "prompt rerun from {start} after a rerun, after prefill finished or past the prompt"
            ));
        }
        self.position = start;
        self.prompt_base = start;
        self.reran = true;
        Ok(())
    }

    /// A speculative request committed and emitted `token` as `verdict`;
    /// `close` are template tokens forced next. It is owed to the target
    /// until a forward runs it. Whether the request continues without a
    /// forced token.
    fn spec_commit(&mut self, token: u32, verdict: Verdict, close: &[u32]) -> bool {
        let d = self.draft.as_mut().expect("a speculative request");
        d.owed.push(token);
        d.emitted.push(token);
        if verdict != Verdict::Eos {
            self.generated += 1;
        }
        if verdict != Verdict::Continue {
            self.finish = Some(Finish::of(true));
            return false;
        }
        if self.generated >= self.budget {
            self.finish = Some(Finish::of(false));
            return false;
        }
        self.forced.extend(close);
        close.is_empty()
    }

    /// A speculative request at a window boundary leaves its drafting for a
    /// lane that runs its windows: its next row is its seed (already
    /// emitted). `None`: not at a window boundary, nothing changed.
    fn take_window_drafting(&mut self) -> Option<spec::Drafting> {
        let d = self.draft.as_ref()?;
        if self.finish.is_some() || d.owed.len() != 1 || !self.forced.is_empty() {
            return None;
        }
        let d = self.draft.take()?;
        self.pending = Some(Some(d.owed[0]));
        Some(d)
    }

    /// A lane window committed `token` and the driver emitted it as
    /// `verdict`; while the request continues it is the next row's seed.
    /// Whether it continues.
    fn apply_window_token(&mut self, token: u32, verdict: Verdict) -> bool {
        self.pending = None;
        if verdict != Verdict::Eos {
            self.generated += 1;
        }
        if verdict != Verdict::Continue {
            self.finish = Some(Finish::of(true));
            return false;
        }
        if self.generated >= self.budget {
            self.finish = Some(Finish::of(false));
            return false;
        }
        self.pending = Some(Some(token));
        true
    }

    /// The window from `start` committed `step.emit`, of which the request
    /// kept the first `kept`; `trim` is what became of the rest.
    fn apply_window(
        &mut self,
        start: usize,
        step: &SpecStep,
        kept: usize,
        ar: bool,
        trim: Option<spec::Trim>,
    ) {
        let prompt_len = self.prompt_len;
        let d = self.draft.as_mut().expect("a speculative request");
        match trim {
            // The window ran its seed and every kept token but the last.
            None | Some(spec::Trim::Repaired) => {
                self.position = start + kept;
                d.owed.drain(..kept);
            }
            Some(spec::Trim::Reset) => {
                self.position = 0;
                d.owed.clear();
                d.invalidated = true;
            }
            Some(spec::Trim::Realigned) => {
                self.position = prompt_len + d.emitted.len() - 1;
                d.owed.drain(..d.owed.len() - 1);
            }
        }
        if ar {
            d.tally.ar_steps += 1;
            return;
        }
        d.tally.windows += 1;
        d.tally.accepted += step.accepted;
        // A window verifies its seed row plus the drafts it proposed.
        if d.retire
            .observe(step.emit.len(), step.proposed + 1, self.position)
        {
            eprintln!(
                "[spec-retire] drafter={} retired to AR at position {} (generated {}, {} windows)",
                d.drafter.name(),
                self.position,
                self.generated,
                d.tally.windows,
            );
        }
    }

    fn sample(&mut self, row: &mut [f32], history: &[u32]) -> u32 {
        apply_logit_policy_cpu(row, history, &self.sampler);
        hipfire_runtime::llama::sample_top_k_p_with_state(
            row,
            self.sampler.temperature,
            self.sampler.top_p,
            self.sampler.top_k,
            self.sampler.min_p,
            &mut self.rng,
        )
    }
}

/// One `DoneMetrics` for every engine route: wall-clock phases and counts.
pub struct DoneMetrics {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub generated: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
}

impl DoneMetrics {
    /// The done envelope (`loop: engine`) with the route's finish reason.
    pub fn done_value(&self, id: &str, finish_reason: &str) -> serde_json::Value {
        let rate = |n: usize, ms: f64| {
            if ms > 0.0 {
                n as f64 / (ms / 1000.0)
            } else {
                0.0
            }
        };
        let computed = self.prompt_tokens - self.cached_tokens;
        let mut done = crate::ar::qwen_ar_done_value(
            id,
            finish_reason,
            self.generated,
            rate(self.generated, self.total_ms),
            computed,
            self.prefill_ms,
            rate(computed, self.prefill_ms),
            rate(self.generated, self.decode_ms),
            self.decode_ms,
            self.prefill_ms,
            self.cached_tokens,
        );
        done["loop"] = "engine".into();
        done
    }
}

/// A model's forwards for [`drive`]: one prompt epoch at a time, one decode
/// epoch per token. Model- and hardware-agnostic (the mesh implementation is
/// [`mesh::MeshForwards`]); none rolls back on its own.
pub trait Forwards {
    /// Run the next prompt epoch from `rows.start`; returns its end and, on
    /// the final one, the logits row.
    fn prefill(&mut self, rows: Range<usize>) -> Result<(usize, Option<Vec<f32>>), String>;
    /// Run `token` (`None`: the device argmax of the last row) and return
    /// the token it ran; `top1` asks for the next token's argmax instead of
    /// the row ([`Next::Top1`] or, kept on the device, [`Next::Device`]).
    fn decode(&mut self, token: Option<u32>, top1: bool) -> Result<(u32, Next), String>;
    /// After a failed prompt epoch: `Some(start)` when the model contained
    /// the fault and the prompt reruns from `start` (already begun there).
    fn prefill_rerun(&mut self) -> Result<Option<usize>, String> {
        Ok(None)
    }
    /// Speculative: the prompt from `start` through the request's drafter
    /// (which owns its chunking); its first pick, `None` when aborted.
    fn draft_prefill(
        &mut self,
        _d: &mut dyn Speculator,
        _start: usize,
    ) -> Result<Option<u32>, String> {
        Err(NO_DRAFTER.into())
    }
    /// Speculative: the drafter's prompt fill it left owed behind the first
    /// token, after that token went out.
    fn finish_prefill(&mut self, _d: &mut dyn Speculator) -> Result<(), String> {
        Err(NO_DRAFTER.into())
    }
    /// Speculative: one window ([`Work::Verify`]) from `position`; `emitted`
    /// is the request's decode history.
    fn verify(
        &mut self,
        _d: &mut dyn Speculator,
        _position: usize,
        _seed: u32,
        _emitted: &[u32],
        _max_emit: usize,
        _ar: bool,
    ) -> Result<SpecStep, String> {
        Err(NO_DRAFTER.into())
    }
    /// Speculative: run committed `tokens` from `position` ([`Work::Advance`]).
    fn advance(
        &mut self,
        _d: &mut dyn Speculator,
        _tokens: &[u32],
        _position: usize,
    ) -> Result<(), String> {
        Err(NO_DRAFTER.into())
    }
    /// Speculative: the window from `start` on `seed` committed past `kept`:
    /// repair it, else reset the live state (a terminal, `replay` `None`) or
    /// re-prefill the prompt plus `replay` (the decode tokens the target
    /// must hold).
    fn trim(
        &mut self,
        _d: &mut dyn Speculator,
        _start: usize,
        _seed: u32,
        _kept: &[u32],
        _ar: bool,
        _replay: Option<&[u32]>,
    ) -> Result<spec::Trim, String> {
        Err(NO_DRAFTER.into())
    }
}

const NO_DRAFTER: &str = "these forwards run no drafter";

/// A request's wire side for [`drive`]: commits and classifies tokens.
pub trait Emit {
    /// The whole prompt is resident: it joins the committed history.
    fn commit_prompt(&mut self);
    /// Commit and emit `token` (it has run); returns how it classified and
    /// any template tokens to force next.
    fn emit(&mut self, token: u32) -> Result<(Verdict, Vec<u32>), String>;
    /// Committed conversation the host sampler penalizes over.
    fn history(&self) -> &[u32];
    fn aborted(&self) -> bool;
    /// The most tokens a speculative window may commit so that a forced
    /// intervention (think-budget close) can only fall on its last one.
    fn window_cap(&self) -> Option<usize> {
        None
    }
}

/// How [`drive`] left a request: finished (still uncommitted), or an exit
/// the caller must roll back.
#[derive(Debug, PartialEq, Eq)]
pub enum Exit {
    Finished(Finish),
    Aborted,
    Failed(String),
}

/// The batch-of-one loop: [`step`] until done, polling abort before every
/// forward (each prompt epoch included).
pub fn drive(
    run: &mut RequestRun,
    forwards: &mut impl Forwards,
    emit: &mut (impl Emit + ?Sized),
) -> Exit {
    loop {
        if let Work::Done(finish) = run.next_work() {
            return Exit::Finished(finish);
        }
        if emit.aborted() {
            return Exit::Aborted;
        }
        if let Err(error) = step(run, &mut Both(forwards, emit)) {
            return Exit::Failed(error);
        }
    }
}

/// One forward of a request and its apply: the next prompt epoch (reporting
/// `prefill_progress`; a contained fault reruns the prompt once from the
/// model's re-planned start), one decode, whose token is emitted, or one
/// speculative window or advance ([`spec_step`]). `Err`: the request failed.
pub fn step(run: &mut RequestRun, io: &mut (impl Forwards + Emit + ?Sized)) -> Result<(), String> {
    if run.draft.is_some() {
        return spec_step(run, io);
    }
    match run.next_work() {
        Work::Prefill { rows } => {
            let _phase = hipfire_runtime::prefill_progress::prompt_phase();
            match io.prefill(rows) {
                Ok((end, logits)) => {
                    if logits.is_some() {
                        io.commit_prompt();
                    }
                    run.apply_prefill(end, logits, io.history())?;
                    hipfire_runtime::prefill_progress::report(
                        end - run.prompt_base,
                        run.prompt_len - run.prompt_base,
                    );
                    Ok(())
                }
                Err(error) if !run.reran => match io.prefill_rerun() {
                    Ok(Some(start)) => {
                        eprintln!("[engine] prompt reruns from {start}: {error}");
                        run.restart_prefill(start)
                    }
                    Ok(None) => Err(error),
                    Err(recover) => Err(format!("{error}; rerun failed: {recover}")),
                },
                Err(error) => Err(error),
            }
        }
        Work::Decode { token } => {
            let (ran, next) = io.decode(token, run.wants_top1())?;
            let (verdict, close) = io.emit(ran)?;
            run.apply_decode(next, verdict, &close, io.history());
            Ok(())
        }
        Work::Verify { .. } | Work::Advance { .. } => Err("an AR request has no drafter".into()),
        Work::Done(_) => Ok(()),
    }
}

/// [`step`] of a speculative request. The prompt runs through the drafter,
/// whose first pick goes out at once (before the fill it left owed). A
/// window's tokens are emitted in order until one stops or forces template
/// tokens; a window committed past that point is trimmed back
/// ([`Forwards::trim`]). Forced tokens are emitted when forced and run ahead
/// of the next window; the terminal flush runs the tokens still owed.
fn spec_step(run: &mut RequestRun, io: &mut (impl Forwards + Emit + ?Sized)) -> Result<(), String> {
    match run.next_work() {
        Work::Prefill { rows } => {
            let first = {
                let _phase = hipfire_runtime::prefill_progress::prompt_phase();
                let d = run.draft.as_mut().expect("a speculative request");
                io.draft_prefill(&mut *d.drafter, rows.start)?
            };
            // Aborted: the next poll ends the request.
            let Some(first) = first else {
                return Ok(());
            };
            io.commit_prompt();
            run.position = run.prompt_len;
            let (verdict, close) = io.emit(first)?;
            run.spec_commit(first, verdict, &close);
            let d = run.draft.as_mut().expect("a speculative request");
            io.finish_prefill(&mut *d.drafter)?;
            spec_emit_forced(run, io)
        }
        Work::Verify { seed, max_emit, ar } => {
            let max_emit = io.window_cap().map_or(max_emit, |cap| max_emit.min(cap));
            let start = run.position;
            let d = run.draft.as_mut().expect("a speculative request");
            let window = io.verify(&mut *d.drafter, start, seed, &d.emitted, max_emit, ar)?;
            if window.emit.is_empty() || window.emit.len() > max_emit {
                return Err(format!(
                    "a window of at most {max_emit} committed {} tokens",
                    window.emit.len()
                ));
            }
            let mut kept = 0;
            for &token in &window.emit {
                let (verdict, close) = io.emit(token)?;
                kept += 1;
                if !run.spec_commit(token, verdict, &close) {
                    break;
                }
            }
            let trim = if kept < window.emit.len() {
                let finishing = run.finish.is_some();
                let d = run.draft.as_mut().expect("a speculative request");
                // A continuing request re-prefills every committed token
                // but the next seed.
                let replay = (!finishing).then(|| &d.emitted[..d.emitted.len() - 1]);
                Some(io.trim(
                    &mut *d.drafter,
                    start,
                    seed,
                    &window.emit[..kept],
                    ar,
                    replay,
                )?)
            } else {
                None
            };
            run.apply_window(start, &window, kept, ar, trim);
            spec_emit_forced(run, io)
        }
        Work::Advance { tokens } => {
            let d = run.draft.as_mut().expect("a speculative request");
            io.advance(&mut *d.drafter, &tokens, run.position)?;
            run.position += tokens.len();
            d.owed.drain(..tokens.len());
            Ok(())
        }
        Work::Decode { .. } => Err("a speculative request decodes in windows".into()),
        Work::Done(_) => Ok(()),
    }
}

/// Emit the template tokens the last emission forced (they run ahead of the
/// next window), until the request finishes.
fn spec_emit_forced(run: &mut RequestRun, io: &mut (impl Emit + ?Sized)) -> Result<(), String> {
    while run.finish.is_none() {
        let Some(token) = run.forced.pop_front() else {
            break;
        };
        let (verdict, close) = io.emit(token)?;
        run.spec_commit(token, verdict, &close);
    }
    Ok(())
}

/// A request's [`Forwards`] and [`Emit`] as one [`step`] seam.
struct Both<'a, F, E: ?Sized>(&'a mut F, &'a mut E);

impl<F: Forwards, E: ?Sized> Forwards for Both<'_, F, E> {
    fn prefill(&mut self, rows: Range<usize>) -> Result<(usize, Option<Vec<f32>>), String> {
        self.0.prefill(rows)
    }

    fn decode(&mut self, token: Option<u32>, top1: bool) -> Result<(u32, Next), String> {
        self.0.decode(token, top1)
    }

    fn prefill_rerun(&mut self) -> Result<Option<usize>, String> {
        self.0.prefill_rerun()
    }

    fn draft_prefill(
        &mut self,
        d: &mut dyn Speculator,
        start: usize,
    ) -> Result<Option<u32>, String> {
        self.0.draft_prefill(d, start)
    }

    fn finish_prefill(&mut self, d: &mut dyn Speculator) -> Result<(), String> {
        self.0.finish_prefill(d)
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
        self.0.verify(d, position, seed, emitted, max_emit, ar)
    }

    fn advance(
        &mut self,
        d: &mut dyn Speculator,
        tokens: &[u32],
        position: usize,
    ) -> Result<(), String> {
        self.0.advance(d, tokens, position)
    }

    fn trim(
        &mut self,
        d: &mut dyn Speculator,
        start: usize,
        seed: u32,
        kept: &[u32],
        ar: bool,
        replay: Option<&[u32]>,
    ) -> Result<spec::Trim, String> {
        self.0.trim(d, start, seed, kept, ar, replay)
    }
}

impl<F, E: Emit + ?Sized> Emit for Both<'_, F, E> {
    fn commit_prompt(&mut self) {
        self.1.commit_prompt();
    }

    fn emit(&mut self, token: u32) -> Result<(Verdict, Vec<u32>), String> {
        self.1.emit(token)
    }

    fn history(&self) -> &[u32] {
        self.1.history()
    }

    fn aborted(&self) -> bool {
        self.1.aborted()
    }

    fn window_cap(&self) -> Option<usize> {
        self.1.window_cap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn row(winner: usize) -> Vec<f32> {
        let mut row = vec![0.0; 8];
        row[winner] = 1.0;
        row
    }

    /// Prefill `len` rows in tiles of `tile`, final row picks `first`.
    fn prefilled(
        len: usize,
        reused: usize,
        tile: usize,
        budget: usize,
        first: usize,
    ) -> RequestRun {
        let mut run = RequestRun::new(len, reused, budget, greedy(), 7);
        while let Work::Prefill { rows } = run.next_work() {
            let end = (rows.start + tile).min(rows.end);
            let logits = (end == len).then(|| row(first));
            run.apply_prefill(end, logits, &[]).unwrap();
        }
        run
    }

    #[test]
    fn prefill_tiles_from_the_reused_prefix_then_decodes_the_first_pick() {
        let mut run = RequestRun::new(10, 4, 3, greedy(), 7);
        assert_eq!(run.next_work(), Work::Prefill { rows: 4..10 });
        run.apply_prefill(8, None, &[]).unwrap();
        assert_eq!(run.next_work(), Work::Prefill { rows: 8..10 });
        run.apply_prefill(10, Some(row(5)), &[]).unwrap();
        assert_eq!(run.next_work(), Work::Decode { token: Some(5) });
    }

    #[test]
    fn prefill_tiles_that_lose_or_misplace_logits_are_refused() {
        let mut run = RequestRun::new(4, 0, 3, greedy(), 7);
        assert!(
            run.apply_prefill(4, None, &[]).is_err(),
            "final tile without logits"
        );
        assert!(
            run.apply_prefill(2, Some(row(1)), &[]).is_err(),
            "logits on a middle tile"
        );
        assert!(run.apply_prefill(0, None, &[]).is_err(), "empty tile");
        assert!(
            run.apply_prefill(5, Some(row(1)), &[]).is_err(),
            "past the prompt"
        );
    }

    #[test]
    fn budget_exhaustion_finishes_length_after_exactly_budget_tokens() {
        let mut run = prefilled(3, 0, 8, 2, 1);
        run.apply_decode(Next::Top1(2), Verdict::Continue, &[], &[]);
        assert_eq!(run.next_work(), Work::Decode { token: Some(2) });
        run.apply_decode(Next::Top1(3), Verdict::Continue, &[], &[]);
        assert_eq!(run.next_work(), Work::Done(Finish::Length));
        assert_eq!(run.generated(), 2);
    }

    #[test]
    fn a_stop_on_the_last_budget_token_finishes_stop() {
        let mut run = prefilled(3, 0, 8, 1, 1);
        run.apply_decode(Next::Top1(2), Verdict::Stop, &[], &[]);
        assert_eq!(run.next_work(), Work::Done(Finish::Stop));
        assert_eq!(run.generated(), 1);
    }

    #[test]
    fn eos_finishes_stop_and_is_not_counted() {
        let mut run = prefilled(3, 0, 8, 4, 1);
        run.apply_decode(Next::Top1(2), Verdict::Continue, &[], &[]);
        run.apply_decode(Next::Top1(0), Verdict::Eos, &[], &[]);
        assert_eq!(run.next_work(), Work::Done(Finish::Stop));
        assert_eq!(run.generated(), 1);
    }

    #[test]
    fn forced_close_tokens_run_before_sampling_resumes() {
        let mut run = prefilled(3, 0, 8, 10, 1);
        assert!(run.wants_top1());
        run.apply_decode(Next::Top1(4), Verdict::Continue, &[6, 7], &[]);
        assert!(!run.wants_top1(), "a queued forced token needs no argmax");
        assert_eq!(run.next_work(), Work::Decode { token: Some(6) });
        run.apply_decode(Next::Top1(4), Verdict::Continue, &[], &[]);
        assert_eq!(run.next_work(), Work::Decode { token: Some(7) });
        run.apply_decode(Next::Top1(4), Verdict::Continue, &[], &[]);
        assert_eq!(run.next_work(), Work::Decode { token: Some(4) });
    }

    #[test]
    fn a_device_argmax_feeds_the_next_decode_unless_a_forced_token_waits() {
        let mut run = prefilled(3, 0, 8, 10, 1);
        run.apply_decode(Next::Device, Verdict::Continue, &[], &[]);
        assert_eq!(run.next_work(), Work::Decode { token: None });
        run.apply_decode(Next::Device, Verdict::Continue, &[9], &[]);
        assert_eq!(
            run.next_work(),
            Work::Decode { token: Some(9) },
            "a forced close token replaces the device argmax"
        );
    }

    #[test]
    fn sampled_requests_reproduce_per_seed_and_never_read_the_global_rng() {
        let sampled = SamplerConfig {
            temperature: 1.0,
            ..greedy()
        };
        let draw = |seed| {
            let mut run = RequestRun::new(1, 0, 64, sampled.clone(), seed);
            let flat = vec![0.0f32; 8];
            run.apply_prefill(1, Some(flat.clone()), &[]).unwrap();
            let mut ids = Vec::new();
            for _ in 0..32 {
                let Work::Decode { token: Some(token) } = run.next_work() else {
                    unreachable!()
                };
                ids.push(token);
                run.apply_decode(Next::Logits(flat.clone()), Verdict::Continue, &[], &[]);
            }
            ids
        };
        let a = draw(11);
        hipfire_runtime::llama::reset_cpu_sampler_rng(99);
        assert_eq!(a, draw(11), "same seed, same stream, global state ignored");
        assert_ne!(a, draw(12));
        assert!(
            a.iter().any(|&t| t != a[0]),
            "a flat row is actually sampled"
        );
    }

    #[test]
    fn host_sampling_applies_penalties_over_the_history() {
        let penalized = SamplerConfig {
            repeat_penalty: 100.0,
            repeat_window: 8,
            ..greedy()
        };
        let mut run = RequestRun::new(1, 0, 4, penalized, 1);
        assert!(!run.wants_top1());
        let mut first = row(3);
        first[2] = 0.9;
        run.apply_prefill(1, Some(first), &[3]).unwrap();
        assert_eq!(run.next_work(), Work::Decode { token: Some(2) });
    }

    #[test]
    fn budget_admission_clamps_and_refuses_before_start() {
        let no_fit = |max: usize, _: usize, _: usize| max;
        assert_eq!(admit_budget(10, 5, 16, no_fit), Ok(5));
        assert!(admit_budget(10, 6, 16, no_fit).is_err(), "10 + 6 + 1 > 16");
        assert!(admit_budget(10, 0, 64, no_fit).is_err());
        let fit = |max: usize, used: usize, cap: usize| max.min(cap - used);
        assert_eq!(admit_budget(10, 100, 16, fit), Ok(5));
        assert!(
            admit_budget(15, 100, 16, fit).is_err(),
            "a fit that leaves no decode row is refused"
        );
    }

    use std::cell::Cell;
    use std::rc::Rc;

    /// Forwards that follow a script: the decode of `t` leaves `t + 1` as the
    /// device argmax. A shared counter places an abort or failure at an exact
    /// forward; `rerun_from` contains every prompt failure at that start.
    struct FakeForwards {
        len: usize,
        tile: usize,
        position: usize,
        count: Rc<Cell<usize>>,
        fail_at: Vec<usize>,
        rerun_from: Option<usize>,
        /// The last token a decode ran (the device argmax source).
        last: u32,
    }

    struct FakeEmit {
        count: Rc<Cell<usize>>,
        abort_at: Option<usize>,
        eos: u32,
        emitted: Vec<u32>,
        history: Vec<u32>,
        prompt_commits: usize,
    }

    fn fakes(len: usize, tile: usize) -> (FakeForwards, FakeEmit) {
        let count = Rc::new(Cell::new(0));
        (
            FakeForwards {
                len,
                tile,
                position: 0,
                count: count.clone(),
                fail_at: Vec::new(),
                rerun_from: None,
                last: 0,
            },
            FakeEmit {
                count,
                abort_at: None,
                eos: 0,
                emitted: Vec::new(),
                history: Vec::new(),
                prompt_commits: 0,
            },
        )
    }

    impl FakeForwards {
        fn forward(&mut self) -> Result<(), String> {
            let n = self.count.get() + 1;
            self.count.set(n);
            if self.fail_at.contains(&n) {
                return Err(format!("forward {n} failed"));
            }
            Ok(())
        }
    }

    impl Forwards for FakeForwards {
        fn prefill(&mut self, rows: Range<usize>) -> Result<(usize, Option<Vec<f32>>), String> {
            assert_eq!(rows.start, self.position);
            self.forward()?;
            self.position = (rows.start + self.tile).min(rows.end);
            Ok((self.position, (self.position == self.len).then(|| row(1))))
        }

        /// Greedy on the device: the decode of `t` leaves `t + 1` as the next
        /// row's argmax, which the following `None` decode runs.
        fn decode(&mut self, token: Option<u32>, top1: bool) -> Result<(u32, Next), String> {
            assert!(top1);
            self.forward()?;
            let ran = token.unwrap_or(self.last + 1);
            self.last = ran;
            Ok((ran, Next::Device))
        }

        fn prefill_rerun(&mut self) -> Result<Option<usize>, String> {
            if let Some(start) = self.rerun_from {
                self.position = start;
            }
            Ok(self.rerun_from)
        }
    }

    impl Emit for FakeEmit {
        fn commit_prompt(&mut self) {
            self.prompt_commits += 1;
        }

        fn emit(&mut self, token: u32) -> Result<(Verdict, Vec<u32>), String> {
            self.emitted.push(token);
            self.history.push(token);
            let verdict = if token == self.eos {
                Verdict::Eos
            } else {
                Verdict::Continue
            };
            Ok((verdict, Vec::new()))
        }

        fn history(&self) -> &[u32] {
            &self.history
        }

        fn aborted(&self) -> bool {
            self.abort_at.is_some_and(|at| self.count.get() >= at)
        }
    }

    #[test]
    fn drive_runs_every_tile_then_decodes_to_the_budget() {
        let (mut fwd, mut emit) = fakes(10, 4);
        let mut run = RequestRun::new(10, 0, 3, greedy(), 1);
        assert_eq!(
            drive(&mut run, &mut fwd, &mut emit),
            Exit::Finished(Finish::Length)
        );
        assert_eq!(fwd.count.get(), 3 + 3, "three tiles, three decodes");
        assert_eq!(emit.emitted, [1, 2, 3]);
        assert_eq!(emit.prompt_commits, 1, "the prompt joins the history once");
    }

    #[test]
    fn drive_stops_on_eos_without_counting_it() {
        let (mut fwd, mut emit) = fakes(2, 8);
        emit.eos = 3;
        let mut run = RequestRun::new(2, 0, 10, greedy(), 1);
        assert_eq!(
            drive(&mut run, &mut fwd, &mut emit),
            Exit::Finished(Finish::Stop)
        );
        assert_eq!(emit.emitted, [1, 2, 3]);
        assert_eq!(run.generated(), 2);
    }

    #[test]
    fn drive_polls_abort_between_prefill_tiles() {
        let (mut fwd, mut emit) = fakes(10, 4);
        emit.abort_at = Some(1);
        let mut run = RequestRun::new(10, 0, 3, greedy(), 1);
        assert_eq!(drive(&mut run, &mut fwd, &mut emit), Exit::Aborted);
        assert_eq!(fwd.count.get(), 1, "no tile runs after the abort");
        assert!(emit.emitted.is_empty());
    }

    #[test]
    fn drive_polls_abort_between_decode_steps() {
        let (mut fwd, mut emit) = fakes(2, 8);
        emit.abort_at = Some(3);
        let mut run = RequestRun::new(2, 0, 10, greedy(), 1);
        assert_eq!(drive(&mut run, &mut fwd, &mut emit), Exit::Aborted);
        assert_eq!(emit.emitted, [1, 2]);
    }

    #[test]
    fn drive_turns_an_uncontained_failure_into_a_failed_exit_at_once() {
        for fail_at in [2, 4] {
            let (mut fwd, mut emit) = fakes(8, 4);
            fwd.fail_at = vec![fail_at];
            let mut run = RequestRun::new(8, 0, 10, greedy(), 1);
            assert_eq!(
                drive(&mut run, &mut fwd, &mut emit),
                Exit::Failed(format!("forward {fail_at} failed"))
            );
            assert_eq!(fwd.count.get(), fail_at, "nothing runs after the failure");
        }
    }

    #[test]
    fn a_contained_prompt_fault_reruns_the_prompt_once_from_the_replanned_start() {
        let (mut fwd, mut emit) = fakes(12, 4);
        fwd.fail_at = vec![2];
        fwd.rerun_from = Some(0);
        let mut run = RequestRun::new(12, 0, 2, greedy(), 1);
        assert_eq!(
            drive(&mut run, &mut fwd, &mut emit),
            Exit::Finished(Finish::Length)
        );
        // tile 1, failed tile 2, then three tiles from 0, then two decodes.
        assert_eq!(fwd.count.get(), 2 + 3 + 2);
        assert_eq!(emit.emitted, [1, 2]);

        let (mut fwd, mut emit) = fakes(12, 4);
        fwd.fail_at = vec![2, 4];
        fwd.rerun_from = Some(0);
        let mut run = RequestRun::new(12, 0, 2, greedy(), 1);
        assert_eq!(
            drive(&mut run, &mut fwd, &mut emit),
            Exit::Failed("forward 4 failed".into()),
            "a second fault is not rerun"
        );
    }

    #[test]
    fn a_prompt_rerun_after_prefill_is_refused() {
        let mut run = RequestRun::new(4, 0, 2, greedy(), 1);
        run.apply_prefill(4, Some(row(1)), &[]).unwrap();
        assert!(run.restart_prefill(0).is_err());
    }
}
