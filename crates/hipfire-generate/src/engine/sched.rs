// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The engine loop's N-request scheduler (G12, the #868 batch-of-N step).
//!
//! Every live request is a [`RequestRun`]; [`Scheduler::tick`] is one step of
//! all of them, and the tick boundary is the yield point between epochs: the
//! caller drains its inbox, admits arrivals and settles terminals there.
//!
//! - **Resident.** One request at a time owns the model's resident state: it
//!   prefills there at the singleton's prompt cadence (hybrid state is
//!   chunking-sensitive, so prompts never run as lane tiles), and a request
//!   that is alone also decodes there exactly as the batch of one does
//!   (device-token greedy decode, nothing drained between tokens). Requests
//!   wait for it in arrival order.
//! - **Lanes.** A greedy request that decodes while others are live moves into
//!   a lane of its own ([`Batch::park`]); every tick runs one decode epoch
//!   over the decoding lanes, at most `lane_rows` of them, round-robin. A lane
//!   row is byte-identical to the singleton's decode row, so a request's
//!   tokens do not depend on how many others ran beside it.
//! - **One tick** = resident epochs, then one lane epoch: a prompt and other
//!   requests' decode rows advance in the same step. The resident state runs
//!   one epoch per tick (a prompt chunk, a decode, or the move into a lane);
//!   once its request moved into a lane the next queued request takes it in
//!   the same tick, so a burst of short prompts all prefill before the lane
//!   epoch that runs their first tokens; requests that arrive while the
//!   resident state is free join the burst ([`Scheduler::takes_arrivals`]).
//!   The requests that moved this tick stream their first lane token
//!   together right before that epoch, not after it ran them; one that this
//!   token ends never runs it. A prompt epoch that takes the tick past its
//!   row budget ([`Batch::row_budget`]) streams the tokens held so far
//!   first: a held token never waits for more prompt rows than one tick
//!   runs.
//! - A request that samples (temperature, penalties) parks like a greedy one:
//!   its lane row comes back as logits and it draws with its own RNG, so its
//!   tokens are the singleton's for the same seed.
//! - An abort or failure ends only its own request; [`Ended`] names where its
//!   state was, so the caller releases exactly that.
//! - A speculative request ([`super::spec`]) runs one window ([`Work::Verify`])
//!   per tick on the resident state, beside the lane epoch of the AR
//!   requests. Where the lanes run windows ([`Batch::lane_takes`]) it moves
//!   into a lane at a window boundary like a decoding AR request, its
//!   drafter's state with it, and the drafter goes back to the load for the
//!   next one; every tick then runs one window epoch over the window lanes.
//!   The tick's first prompt streams its first token at once; one that
//!   moves into its lane right after a later prompt of the tick holds its
//!   first token ([`Emit::holds_first`]) and streams it with the tick's
//!   burst, as an AR request streams its first lane token (so a burst's
//!   first tokens do not leave one prompt apart, the serial prefill's shape).
//!   "First" counts prompt rows: a prompt the session restored whole runs
//!   0 rows, so the request after it is still the tick's first prompt.
//!   Otherwise it stays resident until it finishes and the next queued
//!   request takes the resident state after it. [`Ended`] hands a resident
//!   one's drafter back.

use super::spec::{Drafting, Trim};
use super::{step, Emit, Exit, Forwards, Next, RequestRun, Verdict, Work};
use hipfire_runtime::spec::{SpecStep, Speculator};
use std::collections::VecDeque;
use std::ops::Range;

/// Where a request's state lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    /// Waiting for the resident state; nothing allocated.
    Queued,
    Resident,
    Lane,
}

/// The model and wire side of the scheduled requests, addressed by the
/// caller's request index.
pub trait Batch {
    /// Requests the model's lanes hold at once (0: no lanes).
    fn lane_capacity(&mut self) -> usize;
    /// Lanes one lane epoch runs at most.
    fn lane_rows(&mut self) -> usize;
    /// Rows one tick runs at most (`serve.max_batch_tokens`): first tokens
    /// held for a burst never wait for prompt rows past it.
    fn row_budget(&mut self) -> usize;
    /// Begin `idx` on the resident state (session restore or reset).
    fn begin(&mut self, idx: usize) -> Result<RequestRun, String>;
    /// [`Forwards::prefill`] of the resident `idx`.
    fn prefill(
        &mut self,
        idx: usize,
        rows: Range<usize>,
    ) -> Result<(usize, Option<Vec<f32>>), String>;
    /// [`Forwards::prefill_rerun`] of the resident `idx`.
    fn prefill_rerun(&mut self, idx: usize) -> Result<Option<usize>, String>;
    /// [`Forwards::decode`] of the resident `idx`; `keep` lets a greedy
    /// row's argmax stay on the device ([`Next::Device`]).
    fn decode(
        &mut self,
        idx: usize,
        token: Option<u32>,
        top1: bool,
        keep: bool,
    ) -> Result<(u32, Next), String>;
    /// Move the resident `idx` into a lane; its next row runs `seed`.
    fn park(&mut self, idx: usize, seed: u32, remaining: usize) -> Result<(), String>;
    /// Whether `idx` may decode in a lane: an AR request in AR lanes, a
    /// speculative one in lanes that run windows.
    fn lane_takes(&mut self, idx: usize) -> bool;
    /// Move the resident speculative `idx`, at a window boundary, into a
    /// lane that runs its windows ([`RequestRun::take_window_drafting`]); on
    /// `Err` `run` is unchanged.
    fn park_window(&mut self, idx: usize, run: &mut RequestRun) -> Result<(), String>;
    /// One lane epoch: each `(idx, token, logits)` runs `token`; each row's
    /// argmax ([`Next::Top1`]) or, with `logits`, the row ([`Next::Logits`]).
    fn step_lanes(&mut self, rows: &[(usize, u32, bool)]) -> Result<Vec<Next>, String>;
    /// One window epoch: each `(idx, drafts)` lane verifies at most `drafts`
    /// drafts on its seed; returns each one's committed tokens.
    fn step_windows(&mut self, rows: &[(usize, usize)]) -> Result<Vec<Vec<u32>>, String>;
    /// [`Emit::commit_prompt`] of `idx`.
    fn commit_prompt(&mut self, idx: usize);
    /// [`Emit::emit`] of `idx`.
    fn emit(&mut self, idx: usize, token: u32) -> Result<(Verdict, Vec<u32>), String>;
    /// [`Emit::history`] of `idx`.
    fn history(&self, idx: usize) -> &[u32];
    /// [`Emit::aborted`] of `idx`.
    fn aborted(&self, idx: usize) -> bool;
    /// [`Emit::window_cap`] of `idx`.
    fn window_cap(&self, idx: usize) -> Option<usize>;
    /// The speculative forwards of the resident `idx`.
    type Draft<'a>: Forwards
    where
        Self: 'a;
    fn draft(&mut self, idx: usize) -> Result<Self::Draft<'_>, String>;
}

/// A request that left the scheduler: how, and where its state was.
#[derive(Debug)]
pub struct Ended {
    pub idx: usize,
    pub place: Place,
    pub exit: Exit,
    /// Completion tokens it counted.
    pub generated: usize,
    /// A speculative request's drafter and windows.
    pub drafting: Option<Drafting>,
    /// Rows its model state ran (prompt included).
    pub fed: usize,
}

struct Req {
    place: Place,
    /// `None` while queued.
    run: Option<RequestRun>,
    /// Moved into its lane this tick; its next row's token is not streamed
    /// yet ([`Scheduler::stream_fresh`]).
    fresh: bool,
    /// The next lane row's token was already streamed, with this
    /// classification: the lane epoch only runs it.
    ahead: Option<(Verdict, Vec<u32>)>,
    /// Its lane runs speculative windows.
    window: bool,
}

#[derive(Default)]
pub struct Scheduler {
    reqs: Vec<Option<Req>>,
    /// Admitted requests waiting for the resident state, oldest first.
    queue: VecDeque<usize>,
    resident: Option<usize>,
    /// A request that ended on the resident state still holds it until the
    /// caller settled it ([`Self::release_resident`]): a finished one until
    /// its terminal commits, an aborted or failed one until its rollback ran
    /// (and a speculative one's drafter is back), so the next request never
    /// begins on a state that rollback then resets.
    held: bool,
    /// Round-robin start of the next lane epoch.
    cursor: usize,
    /// Prompt rows this tick ran so far ([`Batch::row_budget`]).
    prompt_rows: usize,
}

impl Scheduler {
    /// Admit request `idx`: it waits for the resident state.
    pub fn admit(&mut self, idx: usize) {
        if self.reqs.len() <= idx {
            self.reqs.resize_with(idx + 1, || None);
        }
        self.reqs[idx] = Some(Req {
            place: Place::Queued,
            run: None,
            fresh: false,
            ahead: None,
            window: false,
        });
        self.queue.push_back(idx);
    }

    /// The resident request that ended was settled: the resident state takes
    /// the next request.
    pub fn release_resident(&mut self) {
        self.held = false;
    }

    /// The drafter a live speculative request holds (route decisions read
    /// the load's drafter while it is out).
    pub fn drafter(&self) -> Option<&dyn Speculator> {
        self.reqs
            .iter()
            .flatten()
            .find_map(|req| req.run.as_ref()?.drafting())
            .map(|d| &*d.drafter)
    }

    /// Every live speculative request's drafting, taken out of its run (a
    /// driver that stops serving hands the drafter back to its load).
    pub fn take_drafting(&mut self) -> Vec<Drafting> {
        self.reqs
            .iter_mut()
            .flatten()
            .filter_map(|req| req.run.take().and_then(RequestRun::into_drafting))
            .collect()
    }

    /// Whether a tick would run anything.
    pub fn busy(&self) -> bool {
        self.reqs
            .iter()
            .flatten()
            .any(|req| req.place != Place::Queued)
            || !self.queue.is_empty()
    }

    fn lanes(&self) -> impl Iterator<Item = usize> + '_ {
        self.reqs
            .iter()
            .enumerate()
            .filter(|(_, req)| req.as_ref().is_some_and(|req| req.place == Place::Lane))
            .map(|(idx, _)| idx)
    }

    fn run_mut(&mut self, idx: usize) -> &mut RequestRun {
        self.reqs[idx]
            .as_mut()
            .and_then(|req| req.run.as_mut())
            .expect("a scheduled request past the queue has its run")
    }

    fn end(&mut self, idx: usize, exit: Exit) -> Ended {
        let req = self.reqs[idx].take().expect("ending a live request");
        if self.resident == Some(idx) {
            self.resident = None;
            self.held = true;
        }
        self.queue.retain(|&queued| queued != idx);
        // A token streamed ahead of its lane row counts too.
        let generated =
            req.run.as_ref().map_or(0, RequestRun::generated) + usize::from(req.ahead.is_some());
        Ended {
            idx,
            place: req.place,
            exit,
            generated,
            fed: req.run.as_ref().map_or(0, |run| run.position),
            drafting: req.run.and_then(RequestRun::into_drafting),
        }
    }

    /// One step of every live request: [`Self::prompts`], then
    /// [`Self::lanes_step`]. Returns the requests that ended.
    pub fn tick(&mut self, b: &mut impl Batch) -> Vec<Ended> {
        let mut ended = self.prompts(b);
        ended.extend(self.lanes_step(b));
        ended
    }

    /// The resident epochs of a tick. A request that left the resident state
    /// for a lane hands it to the next in line within the tick: queued
    /// prompts that each finish in one epoch all reach this tick's lane
    /// epoch (their first tokens share it), a longer prompt takes one epoch
    /// per tick. Aborts are polled again before every epoch inside the tick,
    /// as `drive` does. While [`Self::takes_arrivals`], requests admitted
    /// after it returned run their prompts in this tick on a further call.
    pub fn prompts(&mut self, b: &mut impl Batch) -> Vec<Ended> {
        let mut ended = Vec::new();
        let live: Vec<usize> = (0..self.reqs.len())
            .filter(|&idx| self.reqs[idx].is_some())
            .collect();
        for idx in live {
            if b.aborted(idx) {
                ended.push(self.end(idx, Exit::Aborted));
            }
        }
        while self.start(b, &mut ended) && self.resident_epoch(b, &mut ended) {}
        ended
    }

    /// Whether a request admitted now still begins in this tick: the
    /// resident state is free and nothing holds it.
    pub fn takes_arrivals(&self) -> bool {
        self.resident.is_none() && !self.held
    }

    /// The lane epoch that ends a tick.
    pub fn lanes_step(&mut self, b: &mut impl Batch) -> Vec<Ended> {
        let mut ended = Vec::new();
        self.lane_epoch(b, &mut ended);
        self.prompt_rows = 0;
        ended
    }

    /// Whether a request holds the resident state now: the one already on
    /// it, or the queue front if it can later leave it for a lane (or,
    /// without lanes, if it is alone).
    fn start(&mut self, b: &mut impl Batch, ended: &mut Vec<Ended>) -> bool {
        if self.resident.is_some() {
            return true;
        }
        if self.held {
            return false;
        }
        let idx = loop {
            let Some(&idx) = self.queue.front() else {
                return false;
            };
            if !b.aborted(idx) {
                break idx;
            }
            ended.push(self.end(idx, Exit::Aborted));
        };
        let lanes = self.lanes().count();
        let capacity = b.lane_capacity();
        if lanes > 0 && lanes >= capacity {
            return false;
        }
        self.queue.pop_front();
        self.resident = Some(idx);
        let req = self.reqs[idx].as_mut().expect("a queued request is live");
        req.place = Place::Resident;
        match b.begin(idx) {
            Ok(run) => {
                req.run = Some(run);
                true
            }
            Err(error) => {
                ended.push(self.end(idx, Exit::Failed(error)));
                false
            }
        }
    }

    /// The resident request's next epoch; a request that decodes while
    /// others are live moves into a lane (after one drained decode when its
    /// next token is still on the device; a speculative one at its next
    /// window boundary). Whether the resident state is free for the next
    /// request (it moved or was aborted).
    fn resident_epoch(&mut self, b: &mut impl Batch, ended: &mut Vec<Ended>) -> bool {
        let Some(idx) = self.resident else {
            return false;
        };
        if b.aborted(idx) {
            ended.push(self.end(idx, Exit::Aborted));
            return true;
        }
        let prompt = match self.run_mut(idx).next_work() {
            Work::Prefill { rows } => Some(rows.len()),
            _ => None,
        };
        // The tick's first prompt streams its first token at once; later
        // ones hold theirs for the burst ([`Emit::holds_first`]).
        let later = self.prompt_rows > 0;
        if let Some(rows) = prompt {
            // First tokens held for this tick's burst go out before a prompt
            // epoch that takes the tick past its row budget.
            self.prompt_rows += rows;
            if self.prompt_rows > b.row_budget() {
                self.stream_fresh(b, ended);
            }
        }
        let lanes = self.lanes().count();
        let others = lanes > 0 || !self.queue.is_empty();
        let parks = others && lanes < b.lane_capacity() && b.lane_takes(idx);
        // Where a request moves: its seed is host-known (a speculative
        // request's already emitted).
        let boundary = |work: &Work| {
            matches!(
                work,
                Work::Decode { token: Some(_) } | Work::Verify { ar: false, .. }
            )
        };
        let run = self.run_mut(idx);
        // A parking request at its boundary moves without an epoch; every
        // other resident request runs one.
        if !(parks && boundary(&run.next_work())) {
            let mut one = One {
                b,
                idx,
                keep: !parks,
                hold: parks && later,
            };
            if let Err(error) = step(run, &mut one) {
                ended.push(self.end(idx, Exit::Failed(error)));
                return false;
            }
        }
        let run = self.run_mut(idx);
        let work = run.next_work();
        if let Work::Done(finish) = work {
            ended.push(self.end(idx, Exit::Finished(finish)));
            return false;
        }
        if !(parks && boundary(&work)) {
            return false;
        }
        let window = matches!(work, Work::Verify { .. });
        let held = run.generated() == 0;
        let parked = match work {
            Work::Decode {
                token: Some(seed), ..
            } => {
                let remaining = run.remaining();
                b.park(idx, seed, remaining)
            }
            _ => b.park_window(idx, run),
        };
        if let Err(error) = parked {
            ended.push(self.end(idx, Exit::Failed(error)));
            return false;
        }
        self.resident = None;
        let req = self.reqs[idx].as_mut().expect("parked request is live");
        req.place = Place::Lane;
        // A window lane's seed was streamed on the resident state, unless
        // its prompt held it (nothing emitted yet).
        req.fresh = !window || held;
        req.window = window;
        true
    }

    /// Stream the first lane token of every request that moved into its lane
    /// this tick, all at once before the lane epoch runs them (a burst's
    /// first tokens go out together, one lane step earlier than run-then-
    /// emit). A token that ends its request never runs.
    fn stream_fresh(&mut self, b: &mut impl Batch, ended: &mut Vec<Ended>) {
        let fresh: Vec<usize> = self
            .lanes()
            .filter(|&idx| self.reqs[idx].as_ref().is_some_and(|req| req.fresh))
            .collect();
        for idx in fresh {
            if b.aborted(idx) {
                ended.push(self.end(idx, Exit::Aborted));
                continue;
            }
            let req = self.reqs[idx].as_mut().expect("a lane request is live");
            req.fresh = false;
            let window = req.window;
            let run = req.run.as_mut().expect("a lane request has its run");
            let Work::Decode { token: Some(seed) } = run.next_work() else {
                continue;
            };
            if window {
                self.window_token(b, idx, seed, ended);
                continue;
            }
            let remaining = run.remaining();
            match b.emit(idx, seed) {
                Ok((verdict, close)) if verdict == Verdict::Continue && remaining > 1 => {
                    self.reqs[idx]
                        .as_mut()
                        .expect("a lane request is live")
                        .ahead = Some((verdict, close));
                }
                Ok((verdict, close)) => {
                    let run = self.run_mut(idx);
                    // The pick is never read: the request finishes on this token.
                    run.apply_decode(Next::Top1(seed), verdict, &close, b.history(idx));
                    if let Work::Done(finish) = run.next_work() {
                        ended.push(self.end(idx, Exit::Finished(finish)));
                    }
                }
                Err(error) => ended.push(self.end(idx, Exit::Failed(error))),
            }
        }
    }

    /// One decode epoch over at most `lane_rows` decoding lanes, round-robin
    /// (window lanes run theirs as one window epoch).
    fn lane_epoch(&mut self, b: &mut impl Batch, ended: &mut Vec<Ended>) {
        self.stream_fresh(b, ended);
        let lanes: Vec<usize> = self.lanes().collect();
        if lanes.is_empty() {
            return;
        }
        let take = lanes.len().min(b.lane_rows().max(1));
        let mut rows = Vec::with_capacity(take);
        let mut windows = Vec::new();
        for j in 0..take {
            let idx = lanes[(self.cursor + j) % lanes.len()];
            if b.aborted(idx) {
                ended.push(self.end(idx, Exit::Aborted));
                continue;
            }
            let window = self.reqs[idx].as_ref().is_some_and(|req| req.window);
            let run = self.run_mut(idx);
            match run.next_work() {
                // A window commits at most its drafts plus one.
                Work::Decode { token: Some(_) } if window => {
                    windows.push((idx, run.remaining() - 1));
                }
                Work::Decode { token: Some(token) } => rows.push((idx, token, !run.greedy())),
                work => {
                    let error = format!("a lane request has no host token to run: {work:?}");
                    ended.push(self.end(idx, Exit::Failed(error)));
                }
            }
        }
        self.cursor = (self.cursor + take) % lanes.len();
        if !windows.is_empty() {
            self.window_epoch(b, &windows, ended);
        }
        if rows.is_empty() {
            return;
        }
        let picks = match b.step_lanes(&rows) {
            Ok(picks) if picks.len() == rows.len() => picks,
            failed => {
                let error = failed.err().unwrap_or_else(|| {
                    format!("a lane epoch of {} rows returned another count", rows.len())
                });
                for &(idx, ..) in &rows {
                    ended.push(self.end(idx, Exit::Failed(error.clone())));
                }
                return;
            }
        };
        for (&(idx, token, _), pick) in rows.iter().zip(picks) {
            let ahead = self.reqs[idx].as_mut().and_then(|req| req.ahead.take());
            let emitted = ahead.map_or_else(|| b.emit(idx, token), Ok);
            let run = self.run_mut(idx);
            match emitted {
                Ok((verdict, close)) => run.apply_decode(pick, verdict, &close, b.history(idx)),
                Err(error) => {
                    ended.push(self.end(idx, Exit::Failed(error)));
                    continue;
                }
            }
            if let Work::Done(finish) = run.next_work() {
                ended.push(self.end(idx, Exit::Finished(finish)));
            }
        }
    }

    /// One window epoch: every `(idx, drafts)` window lane commits a window,
    /// whose tokens are emitted in order until one ends the request (the
    /// rest is dropped with the lane).
    fn window_epoch(
        &mut self,
        b: &mut impl Batch,
        rows: &[(usize, usize)],
        ended: &mut Vec<Ended>,
    ) {
        let windows = match b.step_windows(rows) {
            Ok(windows) if windows.len() == rows.len() => windows,
            failed => {
                let error = failed.err().unwrap_or_else(|| {
                    format!(
                        "a window epoch of {} lanes returned another count",
                        rows.len()
                    )
                });
                for &(idx, _) in rows {
                    ended.push(self.end(idx, Exit::Failed(error.clone())));
                }
                return;
            }
        };
        for (&(idx, _), committed) in rows.iter().zip(windows) {
            for token in committed {
                if !self.window_token(b, idx, token, ended) {
                    break;
                }
            }
        }
    }

    /// Emit a window lane's committed `token`; a request it ends or fails
    /// leaves. Whether the request continues.
    fn window_token(
        &mut self,
        b: &mut impl Batch,
        idx: usize,
        token: u32,
        ended: &mut Vec<Ended>,
    ) -> bool {
        let continues = match b.emit(idx, token) {
            // Template tokens are never forced on a window lane
            // ([`Batch::lane_takes`]).
            Ok((_, close)) if !close.is_empty() => {
                Err("a window lane was forced template tokens".to_string())
            }
            Ok((verdict, _)) => Ok(self.run_mut(idx).apply_window_token(token, verdict)),
            Err(error) => Err(error),
        };
        match continues {
            Ok(true) => true,
            Ok(false) => {
                if let Work::Done(finish) = self.run_mut(idx).next_work() {
                    ended.push(self.end(idx, Exit::Finished(finish)));
                }
                false
            }
            Err(error) => {
                ended.push(self.end(idx, Exit::Failed(error)));
                false
            }
        }
    }
}

/// The resident request as one request's [`Forwards`] and [`Emit`], so its
/// epochs run the batch of one's [`step`].
struct One<'a, B> {
    b: &'a mut B,
    idx: usize,
    keep: bool,
    /// A speculative prompt that moves into its lane after the tick's first
    /// prompt: its first token streams with the tick's burst.
    hold: bool,
}

impl<B: Batch> Forwards for One<'_, B> {
    fn prefill(&mut self, rows: Range<usize>) -> Result<(usize, Option<Vec<f32>>), String> {
        self.b.prefill(self.idx, rows)
    }

    fn decode(&mut self, token: Option<u32>, top1: bool) -> Result<(u32, Next), String> {
        self.b.decode(self.idx, token, top1, self.keep)
    }

    fn prefill_rerun(&mut self) -> Result<Option<usize>, String> {
        self.b.prefill_rerun(self.idx)
    }

    fn draft_prefill(
        &mut self,
        d: &mut dyn Speculator,
        start: usize,
    ) -> Result<Option<u32>, String> {
        self.b.draft(self.idx)?.draft_prefill(d, start)
    }

    fn finish_prefill(&mut self, d: &mut dyn Speculator) -> Result<(), String> {
        self.b.draft(self.idx)?.finish_prefill(d)
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
        self.b
            .draft(self.idx)?
            .verify(d, position, seed, emitted, max_emit, ar)
    }

    fn advance(
        &mut self,
        d: &mut dyn Speculator,
        tokens: &[u32],
        position: usize,
    ) -> Result<bool, String> {
        self.b.draft(self.idx)?.advance(d, tokens, position)
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
        self.b
            .draft(self.idx)?
            .trim(d, start, seed, kept, ar, replay)
    }
}

impl<B: Batch> Emit for One<'_, B> {
    fn commit_prompt(&mut self) {
        self.b.commit_prompt(self.idx);
    }

    fn emit(&mut self, token: u32) -> Result<(Verdict, Vec<u32>), String> {
        self.b.emit(self.idx, token)
    }

    fn history(&self) -> &[u32] {
        self.b.history(self.idx)
    }

    fn aborted(&self) -> bool {
        self.b.aborted(self.idx)
    }

    fn window_cap(&self) -> Option<usize> {
        self.b.window_cap(self.idx)
    }

    fn holds_first(&self) -> bool {
        self.hold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::sampler::SamplerConfig;
    use std::collections::HashMap;

    const VOCAB: usize = 40;
    const CHUNK: usize = 3;

    /// The fake model: a row's argmax is a hash of the state's whole history,
    /// so a token depends only on its own request's tokens.
    fn pick(state: &[u32]) -> u32 {
        let h = state.iter().fold(17u64, |h, &t| {
            h.wrapping_mul(31).wrapping_add(u64::from(t) + 1)
        });
        (h % VOCAB as u64) as u32
    }

    #[derive(Debug, Clone, PartialEq)]
    enum Call {
        Prefill(usize),
        Decode { idx: usize, keep: bool },
        Park(usize),
        Lanes(Vec<usize>),
        Emit(usize),
        Window(usize),
        Windows(Vec<usize>),
    }

    #[derive(Default)]
    struct Fake {
        prompts: HashMap<usize, Vec<u32>>,
        budget: usize,
        eos: u32,
        capacity: usize,
        rows: usize,
        /// [`Batch::row_budget`].
        row_budget: usize,
        /// Each request's model state: the tokens it ran.
        state: HashMap<usize, Vec<u32>>,
        /// The device argmax a keep-mode decode left.
        device: HashMap<usize, u32>,
        history: HashMap<usize, Vec<u32>>,
        emitted: HashMap<usize, Vec<u32>>,
        aborted: HashMap<usize, bool>,
        in_lane: HashMap<usize, bool>,
        /// Requests that sample (temperature 0.8) instead of argmax.
        sampled: Vec<usize>,
        /// `(a, b)`: the client aborts `b` while `a` prefills (mid-tick).
        abort_during_prefill: Option<(usize, usize)>,
        /// Requests that decode in speculative windows of 3 greedy tokens.
        spec: Vec<usize>,
        /// The lanes run speculative windows (only speculative requests park).
        windows: bool,
        /// A window lane's pending seed.
        seeds: HashMap<usize, u32>,
        /// Requests whose every emit fails.
        emit_fails: Vec<usize>,
        calls: Vec<Call>,
    }

    /// A speculative request's forwards on the fake model.
    struct FakeDraft<'a> {
        idx: usize,
        prompt: &'a [u32],
        state: &'a mut Vec<u32>,
        calls: &'a mut Vec<Call>,
    }

    impl Forwards for FakeDraft<'_> {
        fn prefill(&mut self, _: Range<usize>) -> Result<(usize, Option<Vec<f32>>), String> {
            unreachable!()
        }
        fn decode(&mut self, _: Option<u32>, _: bool) -> Result<(u32, Next), String> {
            unreachable!()
        }
        fn draft_prefill(
            &mut self,
            _: &mut dyn Speculator,
            _: usize,
        ) -> Result<Option<u32>, String> {
            self.calls.push(Call::Prefill(self.idx));
            *self.state = self.prompt.to_vec();
            Ok(Some(pick(self.state)))
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
            self.calls.push(Call::Window(self.idx));
            assert_eq!(self.state.len(), position);
            self.state.push(seed);
            let n = max_emit.min(3);
            let mut emit = Vec::new();
            for i in 0..n {
                emit.push(pick(self.state));
                if i + 1 < n {
                    self.state.push(*emit.last().unwrap());
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
        ) -> Result<bool, String> {
            assert_eq!(self.state.len(), position);
            self.state.extend_from_slice(tokens);
            Ok(true)
        }
        fn trim(
            &mut self,
            _: &mut dyn Speculator,
            start: usize,
            seed: u32,
            kept: &[u32],
            _: bool,
            _: Option<&[u32]>,
        ) -> Result<Trim, String> {
            self.state.truncate(start);
            self.state.push(seed);
            self.state.extend_from_slice(&kept[..kept.len() - 1]);
            Ok(Trim::Repaired)
        }
    }

    /// A logits row over the state's history for a sampled request: several
    /// near values, so the draw depends on the request's own RNG.
    fn row_of(state: &[u32]) -> Vec<f32> {
        let h = u64::from(pick(state)) + 1;
        (0..VOCAB as u64)
            .map(|i| ((h * (i + 3)) % 11) as f32 / 4.0)
            .collect()
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

    fn one_hot(token: u32) -> Vec<f32> {
        let mut row = vec![0.0; VOCAB];
        row[token as usize] = 1.0;
        row
    }

    impl Batch for Fake {
        fn lane_capacity(&mut self) -> usize {
            self.capacity
        }

        fn lane_rows(&mut self) -> usize {
            self.rows
        }

        fn row_budget(&mut self) -> usize {
            self.row_budget
        }

        fn begin(&mut self, idx: usize) -> Result<RequestRun, String> {
            self.state.insert(idx, Vec::new());
            let len = self.prompts[&idx].len();
            let sampler = if self.sampled.contains(&idx) {
                SamplerConfig {
                    temperature: 0.8,
                    ..greedy()
                }
            } else {
                greedy()
            };
            let run = RequestRun::new(len, 0, self.budget, sampler, 7 + idx as u32);
            Ok(if self.spec.contains(&idx) {
                run.with_drafting(Drafting::new(Box::new(super::super::spec::NoDrafter), None))
            } else {
                run
            })
        }

        fn prefill(
            &mut self,
            idx: usize,
            rows: Range<usize>,
        ) -> Result<(usize, Option<Vec<f32>>), String> {
            assert!(
                !self.in_lane.contains_key(&idx),
                "prompts run on the resident state"
            );
            self.calls.push(Call::Prefill(idx));
            if let Some((_, other)) = self.abort_during_prefill.filter(|&(a, _)| a == idx) {
                self.aborted.insert(other, true);
            }
            let end = (rows.start + CHUNK).min(rows.end);
            let state = self.state.get_mut(&idx).unwrap();
            state.extend_from_slice(&self.prompts[&idx][rows.start..end]);
            let row = if self.sampled.contains(&idx) {
                row_of(state)
            } else {
                one_hot(pick(state))
            };
            Ok((end, (end == rows.end).then_some(row)))
        }

        fn prefill_rerun(&mut self, _idx: usize) -> Result<Option<usize>, String> {
            Ok(None)
        }

        fn decode(
            &mut self,
            idx: usize,
            token: Option<u32>,
            top1: bool,
            keep: bool,
        ) -> Result<(u32, Next), String> {
            assert_eq!(top1, !self.sampled.contains(&idx));
            self.calls.push(Call::Decode { idx, keep });
            let ran = match token {
                Some(token) => token,
                None => self.device.remove(&idx).expect("a device token was left"),
            };
            let state = self.state.get_mut(&idx).unwrap();
            state.push(ran);
            if !top1 {
                return Ok((ran, Next::Logits(row_of(state))));
            }
            let next = pick(state);
            if keep {
                self.device.insert(idx, next);
                Ok((ran, Next::Device))
            } else {
                Ok((ran, Next::Top1(next)))
            }
        }

        fn park(&mut self, idx: usize, _seed: u32, _remaining: usize) -> Result<(), String> {
            assert!(
                !self.device.contains_key(&idx),
                "parks with its token on the host"
            );
            self.calls.push(Call::Park(idx));
            self.in_lane.insert(idx, true);
            Ok(())
        }

        fn lane_takes(&mut self, idx: usize) -> bool {
            self.spec.contains(&idx) == self.windows
        }

        fn park_window(&mut self, idx: usize, run: &mut RequestRun) -> Result<(), String> {
            self.calls.push(Call::Park(idx));
            let d = run.drafting().ok_or("a window lane needs a drafter")?;
            // The lane's committed tokens end in its seed, streamed or held.
            let streamed = self.emitted.get(&idx).map_or(&[][..], Vec::as_slice);
            assert_eq!(d.emitted.last(), d.owed.last());
            assert!(d.emitted.starts_with(streamed) && d.emitted.len() <= streamed.len() + 1);
            run.take_window_drafting()
                .ok_or("not at a window boundary")?;
            let Work::Decode { token: Some(seed) } = run.next_work() else {
                return Err("a window lane runs its seed next".into());
            };
            self.seeds.insert(idx, seed);
            self.in_lane.insert(idx, true);
            Ok(())
        }

        fn step_lanes(&mut self, rows: &[(usize, u32, bool)]) -> Result<Vec<Next>, String> {
            self.calls
                .push(Call::Lanes(rows.iter().map(|r| r.0).collect()));
            Ok(rows
                .iter()
                .map(|&(idx, token, logits)| {
                    assert!(self.in_lane.contains_key(&idx));
                    assert_eq!(logits, self.sampled.contains(&idx));
                    let state = self.state.get_mut(&idx).unwrap();
                    state.push(token);
                    if logits {
                        Next::Logits(row_of(state))
                    } else {
                        Next::Top1(pick(state))
                    }
                })
                .collect())
        }

        /// The fake's resident window ([`FakeDraft::verify`]) in a lane.
        fn step_windows(&mut self, rows: &[(usize, usize)]) -> Result<Vec<Vec<u32>>, String> {
            self.calls
                .push(Call::Windows(rows.iter().map(|r| r.0).collect()));
            Ok(rows
                .iter()
                .map(|&(idx, drafts)| {
                    assert!(self.in_lane.contains_key(&idx));
                    let state = self.state.get_mut(&idx).unwrap();
                    state.push(self.seeds[&idx]);
                    let n = (drafts + 1).min(3);
                    let mut committed = Vec::new();
                    for i in 0..n {
                        committed.push(pick(state));
                        if i + 1 < n {
                            state.push(*committed.last().unwrap());
                        }
                    }
                    self.seeds.insert(idx, *committed.last().unwrap());
                    committed
                })
                .collect())
        }

        fn commit_prompt(&mut self, idx: usize) {
            self.history.insert(idx, self.prompts[&idx].clone());
        }

        fn emit(&mut self, idx: usize, token: u32) -> Result<(Verdict, Vec<u32>), String> {
            self.calls.push(Call::Emit(idx));
            if self.emit_fails.contains(&idx) {
                return Err(format!("emit of {idx} failed"));
            }
            self.emitted.entry(idx).or_default().push(token);
            self.history.get_mut(&idx).unwrap().push(token);
            let verdict = if token == self.eos {
                Verdict::Eos
            } else {
                Verdict::Continue
            };
            Ok((verdict, Vec::new()))
        }

        fn history(&self, idx: usize) -> &[u32] {
            self.history.get(&idx).map_or(&[], Vec::as_slice)
        }

        fn aborted(&self, idx: usize) -> bool {
            self.aborted.get(&idx).copied().unwrap_or(false)
        }

        fn window_cap(&self, _: usize) -> Option<usize> {
            None
        }

        type Draft<'a> = FakeDraft<'a>;

        fn draft(&mut self, idx: usize) -> Result<FakeDraft<'_>, String> {
            // The client aborts the other request while this prompt runs.
            if let Some((_, other)) = self.abort_during_prefill.filter(|&(a, _)| a == idx) {
                self.aborted.insert(other, true);
            }
            Ok(FakeDraft {
                idx,
                prompt: &self.prompts[&idx],
                state: self.state.get_mut(&idx).unwrap(),
                calls: &mut self.calls,
            })
        }
    }

    fn fake(prompts: &[Vec<u32>], capacity: usize, rows: usize) -> Fake {
        Fake {
            prompts: prompts.iter().cloned().enumerate().collect(),
            budget: 12,
            eos: 0,
            capacity,
            rows,
            row_budget: 100,
            // Request 1 samples in every scenario.
            sampled: vec![1],
            ..Fake::default()
        }
    }

    fn prompts() -> Vec<Vec<u32>> {
        vec![
            vec![5, 6, 7, 8, 9, 10, 11],
            vec![3, 1, 4, 1, 5],
            vec![2, 7, 1, 8, 2, 8, 1, 8, 2, 8],
            vec![9],
        ]
    }

    /// Tick until every request ended; the ended list in order.
    fn run_all(s: &mut Scheduler, b: &mut Fake) -> Vec<Ended> {
        let mut ended = Vec::new();
        for _ in 0..1000 {
            if !s.busy() {
                return ended;
            }
            for e in s.tick(b) {
                if e.place == Place::Resident {
                    s.release_resident();
                }
                ended.push(e);
            }
        }
        panic!("the scheduler did not drain");
    }

    /// Each request's tokens served alone.
    fn alone() -> Vec<Vec<u32>> {
        let prompts = prompts();
        (0..prompts.len())
            .map(|idx| {
                let mut b = fake(&prompts, 4, 4);
                let mut s = Scheduler::default();
                s.admit(idx);
                run_all(&mut s, &mut b);
                assert!(
                    !b.calls
                        .iter()
                        .any(|c| matches!(c, Call::Park(_) | Call::Lanes(_))),
                    "a lone request never leaves the resident state"
                );
                b.emitted.remove(&idx).unwrap()
            })
            .collect()
    }

    #[test]
    fn tokens_do_not_depend_on_concurrency_or_the_row_budget() {
        let reference = alone();
        for (capacity, rows) in [(4, 4), (4, 1), (2, 2), (1, 1)] {
            let prompts = prompts();
            let mut b = fake(&prompts, capacity, rows);
            let mut s = Scheduler::default();
            for idx in 0..prompts.len() {
                s.admit(idx);
            }
            let ended = run_all(&mut s, &mut b);
            assert_eq!(ended.len(), prompts.len());
            assert!(ended.iter().all(|e| matches!(e.exit, Exit::Finished(_))));
            for (idx, tokens) in reference.iter().enumerate() {
                assert_eq!(
                    &b.emitted[&idx], tokens,
                    "request {idx} at capacity {capacity}, rows {rows}"
                );
            }
            if capacity > 1 {
                assert!(
                    b.calls.contains(&Call::Park(1)),
                    "the sampled request decodes in a lane too"
                );
            }
            if capacity > 1 && rows > 1 {
                assert!(
                    b.calls
                        .iter()
                        .any(|c| matches!(c, Call::Lanes(rows) if rows.len() > 1)),
                    "lanes decode together"
                );
            }
        }
    }

    #[test]
    fn a_speculative_request_verifies_on_the_resident_state_beside_the_lanes() {
        let reference = alone();
        let prompts = prompts();
        let mut b = fake(&prompts, 4, 4);
        b.spec = vec![2];
        let mut s = Scheduler::default();
        for idx in 0..prompts.len() {
            s.admit(idx);
        }
        let ended = run_all(&mut s, &mut b);
        for (idx, tokens) in reference.iter().enumerate() {
            assert_eq!(&b.emitted[&idx], tokens, "request {idx}");
        }
        assert!(
            !b.calls.contains(&Call::Park(2)),
            "a speculative request keeps the resident state"
        );
        // A window and a lane epoch share a tick.
        let windows: Vec<usize> = (0..b.calls.len())
            .filter(|&i| b.calls[i] == Call::Window(2))
            .collect();
        assert!(windows.len() > 1);
        assert!(
            b.calls[windows[0]..windows[1]]
                .iter()
                .any(|c| matches!(c, Call::Lanes(_))),
            "AR lanes decode between windows"
        );
        let spec = ended.iter().find(|e| e.idx == 2).unwrap();
        assert!(
            spec.drafting.is_some(),
            "the drafter comes back with its request"
        );
        assert!(ended
            .iter()
            .filter(|e| e.idx != 2)
            .all(|e| e.drafting.is_none()));
    }

    #[test]
    fn speculative_requests_run_their_windows_together_in_window_lanes() {
        let reference = alone();
        for (capacity, rows) in [(4, 4), (4, 2), (2, 2), (1, 1)] {
            let prompts = prompts();
            let mut b = fake(&prompts, capacity, rows);
            b.spec = vec![0, 2, 3];
            b.windows = true;
            let mut s = Scheduler::default();
            for idx in 0..prompts.len() {
                s.admit(idx);
            }
            let ended = run_all(&mut s, &mut b);
            assert!(ended.iter().all(|e| matches!(e.exit, Exit::Finished(_))));
            for (idx, tokens) in reference.iter().enumerate() {
                assert_eq!(
                    &b.emitted[&idx], tokens,
                    "request {idx} at capacity {capacity}, rows {rows}"
                );
            }
            assert!(
                !b.calls.contains(&Call::Park(1)),
                "an AR request does not park in window lanes"
            );
            if capacity > 1 && rows > 1 {
                assert!(
                    b.calls
                        .iter()
                        .any(|c| matches!(c, Call::Windows(lanes) if lanes.len() > 1)),
                    "window lanes verify together"
                );
            }
            // A parked request's drafter went back to the load at its park.
            for e in &ended {
                assert_eq!(
                    e.drafting.is_some(),
                    e.place == Place::Resident && e.idx != 1
                );
            }
        }
    }

    #[test]
    fn a_prompt_epoch_and_other_requests_decode_rows_share_a_tick() {
        let prompts = prompts();
        let mut b = fake(&prompts, 4, 4);
        b.eos = u32::MAX;
        let mut s = Scheduler::default();
        s.admit(0);
        // Alone: prompt epochs, then device-token decode on the resident state.
        for _ in 0..4 {
            s.tick(&mut b);
        }
        let last_forward = b.calls.iter().rev().find(|c| !matches!(c, Call::Emit(_)));
        assert_eq!(last_forward, Some(&Call::Decode { idx: 0, keep: true }));
        s.admit(2);
        b.calls.clear();
        s.tick(&mut b);
        // A peer queued: one drained decode brings the token to the host, the
        // request moves into a lane, the peer's first prompt epoch takes the
        // resident state, then the lane's token streams and its row runs,
        // all in this tick.
        assert_eq!(
            b.calls,
            vec![
                Call::Decode {
                    idx: 0,
                    keep: false
                },
                Call::Emit(0),
                Call::Park(0),
                Call::Prefill(2),
                Call::Emit(0),
                Call::Lanes(vec![0])
            ]
        );
        b.calls.clear();
        s.tick(&mut b);
        assert_eq!(
            b.calls,
            vec![Call::Prefill(2), Call::Lanes(vec![0]), Call::Emit(0)]
        );
    }

    #[test]
    fn a_burst_of_short_prompts_streams_first_tokens_before_its_lane_epoch() {
        let prompts = vec![vec![1, 2], vec![3, 4, 5], vec![6], vec![7, 8, 9, 10]];
        let mut b = fake(&prompts, 4, 4);
        b.eos = u32::MAX;
        let mut s = Scheduler::default();
        for idx in 0..4 {
            s.admit(idx);
        }
        s.tick(&mut b);
        // Prompts of one epoch each: each prefills and moves into a lane;
        // then every first token streams together and one lane epoch runs
        // them. The fourth prompt takes two epochs.
        assert_eq!(
            b.calls,
            vec![
                Call::Prefill(0),
                Call::Park(0),
                Call::Prefill(1),
                Call::Park(1),
                Call::Prefill(2),
                Call::Park(2),
                Call::Prefill(3),
                Call::Emit(0),
                Call::Emit(1),
                Call::Emit(2),
                Call::Lanes(vec![0, 1, 2])
            ]
        );
    }

    #[test]
    fn a_burst_of_speculative_prompts_streams_first_tokens_before_its_window_epoch() {
        let prompts = vec![vec![1, 2], vec![3, 4, 5], vec![6]];
        let mut b = fake(&prompts, 4, 4);
        b.eos = u32::MAX;
        b.sampled = Vec::new();
        b.spec = vec![0, 1, 2];
        b.windows = true;
        let mut s = Scheduler::default();
        for idx in 0..3 {
            s.admit(idx);
        }
        s.tick(&mut b);
        // The tick's first prompt streams its first token at once; the
        // later ones move into their window lanes holding theirs, which
        // stream together right before one window epoch runs them all.
        assert_eq!(
            b.calls,
            vec![
                Call::Prefill(0),
                Call::Emit(0),
                Call::Park(0),
                Call::Prefill(1),
                Call::Park(1),
                Call::Prefill(2),
                Call::Park(2),
                Call::Emit(1),
                Call::Emit(2),
                Call::Windows(vec![0, 1, 2]),
                Call::Emit(0),
                Call::Emit(0),
                Call::Emit(0),
                Call::Emit(1),
                Call::Emit(1),
                Call::Emit(1),
                Call::Emit(2),
                Call::Emit(2),
                Call::Emit(2),
            ]
        );
    }

    #[test]
    fn held_first_tokens_stream_before_a_prompt_past_the_row_budget() {
        let prompts = vec![vec![1, 2], vec![3; 150], vec![6]];
        let mut b = fake(&prompts, 4, 4);
        b.eos = u32::MAX;
        b.sampled = Vec::new();
        b.spec = vec![0, 1, 2];
        b.windows = true;
        let mut s = Scheduler::default();
        for idx in 0..3 {
            s.admit(idx);
        }
        s.tick(&mut b);
        // The long prompt takes the tick past its 100-row budget: the first
        // token it held goes out before the next prompt.
        assert_eq!(
            b.calls[..10],
            [
                Call::Prefill(0),
                Call::Emit(0),
                Call::Park(0),
                Call::Prefill(1),
                Call::Park(1),
                Call::Emit(1),
                Call::Prefill(2),
                Call::Park(2),
                Call::Emit(2),
                Call::Windows(vec![0, 1, 2]),
            ]
        );
    }

    #[test]
    fn the_first_prompt_of_every_tick_streams_its_first_token_at_once() {
        let prompts = vec![vec![1, 2], vec![3, 4, 5], vec![6, 7], vec![8, 9]];
        let mut b = fake(&prompts, 4, 4);
        b.eos = u32::MAX;
        b.sampled = Vec::new();
        b.spec = vec![0, 1, 2, 3];
        b.windows = true;
        let mut s = Scheduler::default();
        s.admit(0);
        s.admit(1);
        s.tick(&mut b);
        b.calls.clear();
        // The next tick counts its prompt rows from zero again.
        s.admit(2);
        s.admit(3);
        s.tick(&mut b);
        assert_eq!(
            b.calls[..7],
            [
                Call::Prefill(2),
                Call::Emit(2),
                Call::Park(2),
                Call::Prefill(3),
                Call::Park(3),
                Call::Emit(3),
                Call::Windows(vec![0, 1, 2, 3]),
            ]
        );
    }

    #[test]
    fn requests_admitted_while_the_resident_state_is_free_join_the_ticks_burst() {
        let prompts = vec![vec![1, 2], vec![3, 4, 5], vec![6], vec![7, 8]];
        let mut b = fake(&prompts, 4, 4);
        b.eos = u32::MAX;
        b.sampled = Vec::new();
        b.spec = vec![0, 1, 2, 3];
        b.windows = true;
        let mut s = Scheduler::default();
        s.admit(0);
        s.admit(1);
        s.prompts(&mut b);
        // Both moved into their lanes: the resident state is free, so the
        // next arrivals prefill in this tick and stream with them.
        assert!(s.takes_arrivals());
        s.admit(2);
        s.admit(3);
        s.prompts(&mut b);
        s.lanes_step(&mut b);
        assert_eq!(
            b.calls[..13],
            [
                Call::Prefill(0),
                Call::Emit(0),
                Call::Park(0),
                Call::Prefill(1),
                Call::Park(1),
                Call::Prefill(2),
                Call::Park(2),
                Call::Prefill(3),
                Call::Park(3),
                Call::Emit(1),
                Call::Emit(2),
                Call::Emit(3),
                Call::Windows(vec![0, 1, 2, 3]),
            ]
        );
        // A lone request keeps the resident state: an arrival waits for the
        // next tick.
        let mut b = fake(&prompts, 4, 4);
        b.eos = u32::MAX;
        let mut s = Scheduler::default();
        s.admit(0);
        s.prompts(&mut b);
        assert!(!s.takes_arrivals());
    }

    #[test]
    fn a_held_first_token_that_ends_its_request_never_runs_in_its_window_lane() {
        let prompts = vec![vec![9], vec![1, 2], vec![3, 4, 5], vec![6], vec![7, 8]];
        let setup = |b: &mut Fake| {
            b.eos = pick(&[1, 2]);
            b.sampled = Vec::new();
            b.spec = vec![0, 1, 2, 3, 4];
            b.windows = true;
            b.capacity = 5;
        };
        let mut b = fake(&prompts, 4, 4);
        setup(&mut b);
        // Behind the tick's first prompt: 1's first token is EOS, 2 is
        // aborted after its park, 3's emit fails.
        b.abort_during_prefill = Some((3, 2));
        b.emit_fails = vec![3];
        let mut s = Scheduler::default();
        for idx in 0..5 {
            s.admit(idx);
        }
        let ended = run_all(&mut s, &mut b);
        let ended_of = |idx| ended.iter().find(|e| e.idx == idx).unwrap();
        assert_eq!(ended_of(1).exit, Exit::Finished(super::super::Finish::Stop));
        assert_eq!(ended_of(2).exit, Exit::Aborted);
        assert!(matches!(ended_of(3).exit, Exit::Failed(_)));
        assert!(matches!(ended_of(4).exit, Exit::Finished(_)));
        assert!((1..5)
            .all(|idx| ended_of(idx).place == Place::Lane && ended_of(idx).drafting.is_none()));
        assert_eq!(b.emitted[&1], vec![b.eos]);
        assert!(!b.emitted.contains_key(&2) && !b.emitted.contains_key(&3));
        assert!(b.calls.iter().all(
            |c| !matches!(c, Call::Windows(lanes) if lanes.iter().any(|l| (1..4).contains(l)))
        ));
        // The survivor's tokens are its own alone.
        let mut alone = fake(&prompts, 4, 4);
        setup(&mut alone);
        let mut s = Scheduler::default();
        s.admit(4);
        run_all(&mut s, &mut alone);
        assert_eq!(b.emitted[&4], alone.emitted[&4]);
    }

    #[test]
    fn a_first_token_that_ends_the_request_never_runs_in_its_lane() {
        let prompts = vec![vec![1, 2], vec![3, 4, 5, 6]];
        let mut b = fake(&prompts, 4, 4);
        b.eos = pick(&prompts[0]);
        let mut s = Scheduler::default();
        s.admit(0);
        s.admit(1);
        let ended = s.tick(&mut b);
        assert_eq!(ended.len(), 1);
        assert_eq!(ended[0].idx, 0);
        assert_eq!(ended[0].place, Place::Lane);
        assert_eq!(ended[0].exit, Exit::Finished(super::super::Finish::Stop));
        assert_eq!(b.emitted[&0], vec![b.eos]);
        assert!(
            !b.calls
                .iter()
                .any(|c| matches!(c, Call::Lanes(rows) if rows.contains(&0))),
            "{:?}",
            b.calls
        );
    }

    #[test]
    fn lanes_take_turns_under_a_row_budget() {
        let prompts = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let mut b = fake(&prompts, 3, 2);
        b.budget = 30;
        let mut s = Scheduler::default();
        for idx in 0..3 {
            s.admit(idx);
        }
        for _ in 0..12 {
            s.tick(&mut b);
        }
        let steps: Vec<Vec<usize>> = b
            .calls
            .iter()
            .filter_map(|c| match c {
                Call::Lanes(rows) => Some(rows.clone()),
                _ => None,
            })
            .collect();
        let full: Vec<&Vec<usize>> = steps.iter().filter(|rows| rows.len() == 2).collect();
        assert!(full.len() >= 3, "{steps:?}");
        // Over three full steps of two rows, every lane ran exactly twice.
        let mut runs = [0; 3];
        for rows in &full[full.len() - 3..] {
            for &idx in rows.iter() {
                runs[idx] += 1;
            }
        }
        assert_eq!(runs, [2, 2, 2], "{steps:?}");
    }

    #[test]
    fn an_abort_ends_only_its_request() {
        let reference = alone();
        let prompts = prompts();
        let mut b = fake(&prompts, 4, 4);
        let mut s = Scheduler::default();
        for idx in 0..prompts.len() {
            s.admit(idx);
        }
        let mut ended = Vec::new();
        // Request 1 decodes in a lane; request 3 is still queued.
        while !b.in_lane.contains_key(&1) {
            ended.extend(s.tick(&mut b));
        }
        b.aborted.insert(1, true);
        b.aborted.insert(3, true);
        ended.extend(s.tick(&mut b));
        b.aborted.clear();
        ended.extend(run_all(&mut s, &mut b));
        let exit = |idx| {
            ended
                .iter()
                .find(|e| e.idx == idx)
                .map(|e| (&e.exit, e.place))
        };
        assert_eq!(exit(1), Some((&Exit::Aborted, Place::Lane)));
        assert_eq!(exit(3), Some((&Exit::Aborted, Place::Queued)));
        for idx in [0, 2] {
            assert!(matches!(exit(idx), Some((Exit::Finished(_), _))));
            assert_eq!(b.emitted[&idx], reference[idx], "request {idx}");
        }
        assert!(!b.emitted.contains_key(&3));
    }

    #[test]
    fn the_next_request_waits_for_an_aborted_resident_requests_rollback() {
        let prompts = prompts();
        let mut b = fake(&prompts, 4, 4);
        // Request 2 speculates: it keeps the resident state while it decodes.
        b.spec = vec![2];
        let mut s = Scheduler::default();
        s.admit(2);
        s.admit(3);
        assert!(s.drafter().is_none(), "no request runs yet");
        while !b.calls.contains(&Call::Window(2)) {
            assert!(s.tick(&mut b).is_empty());
        }
        assert!(s.drafter().is_some(), "the live request holds the drafter");
        b.aborted.insert(2, true);
        b.calls.clear();
        let ended = s.tick(&mut b);
        assert_eq!(ended.len(), 1);
        assert_eq!(
            (&ended[0].exit, ended[0].place),
            (&Exit::Aborted, Place::Resident)
        );
        assert!(ended[0].drafting.is_some());
        assert!(s.drafter().is_none(), "the drafter left with the request");
        assert!(
            !b.calls.contains(&Call::Prefill(3)),
            "nothing begins on the resident state before its rollback"
        );
        assert!(s.tick(&mut b).is_empty());
        assert!(!b.calls.contains(&Call::Prefill(3)));
        s.release_resident();
        s.tick(&mut b);
        assert!(b.calls.contains(&Call::Prefill(3)));
    }

    #[test]
    fn an_abort_latched_mid_tick_stops_the_request_before_its_next_epoch() {
        let prompts = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let mut b = fake(&prompts, 4, 4);
        b.eos = u32::MAX;
        // 1 is aborted while 0 prefills, before it took the resident state.
        b.abort_during_prefill = Some((0, 1));
        let mut s = Scheduler::default();
        for idx in 0..3 {
            s.admit(idx);
        }
        let ended = s.tick(&mut b);
        assert!(!b.calls.contains(&Call::Prefill(1)), "{:?}", b.calls);
        assert!(!b.emitted.contains_key(&1));
        let one = ended
            .iter()
            .find(|e| e.idx == 1)
            .expect("1 ended this tick");
        assert_eq!((&one.exit, one.place), (&Exit::Aborted, Place::Queued));
        // The resident state went on to the next request in the same tick.
        assert!(b.calls.contains(&Call::Prefill(2)), "{:?}", b.calls);
    }

    #[test]
    fn a_request_parked_this_tick_and_aborted_later_in_it_streams_nothing() {
        let prompts = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let mut b = fake(&prompts, 4, 4);
        b.eos = u32::MAX;
        // 0 parks after its prompt; it is aborted while 2 prefills, before
        // the tick's lane epoch would stream its first token.
        b.abort_during_prefill = Some((2, 0));
        let mut s = Scheduler::default();
        for idx in 0..3 {
            s.admit(idx);
        }
        let ended = s.tick(&mut b);
        assert!(b.calls.contains(&Call::Park(0)), "{:?}", b.calls);
        assert!(!b.emitted.contains_key(&0), "{:?}", b.emitted);
        let zero = ended
            .iter()
            .find(|e| e.idx == 0)
            .expect("0 ended this tick");
        assert_eq!((&zero.exit, zero.place), (&Exit::Aborted, Place::Lane));
        assert!(b.emitted.contains_key(&1) && b.emitted.contains_key(&2));
    }

    #[test]
    fn a_full_lane_set_holds_the_queue() {
        let prompts = prompts();
        let mut b = fake(&prompts, 1, 1);
        b.eos = u32::MAX;
        let mut s = Scheduler::default();
        s.admit(0);
        s.admit(1);
        while !b.in_lane.contains_key(&0) {
            s.tick(&mut b);
        }
        let before = b.calls.len();
        s.tick(&mut b);
        assert!(
            !b.calls[before..].contains(&Call::Prefill(1)),
            "request 1 waits for the lane request 0 holds"
        );
    }
}
