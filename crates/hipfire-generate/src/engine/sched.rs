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
//!   epoch that runs their first tokens. The requests that moved this tick
//!   stream their first lane token together right before that epoch, not
//!   after it ran them; one that this token ends never runs it.
//! - A request that samples (temperature, penalties) parks like a greedy one:
//!   its lane row comes back as logits and it draws with its own RNG, so its
//!   tokens are the singleton's for the same seed.
//! - An abort or failure ends only its own request; [`Ended`] names where its
//!   state was, so the caller releases exactly that.

use super::{step, Emit, Exit, Forwards, Next, RequestRun, Verdict, Work};
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
    /// One lane epoch: each `(idx, token, logits)` runs `token`; each row's
    /// argmax ([`Next::Top1`]) or, with `logits`, the row ([`Next::Logits`]).
    fn step_lanes(&mut self, rows: &[(usize, u32, bool)]) -> Result<Vec<Next>, String>;
    /// [`Emit::commit_prompt`] of `idx`.
    fn commit_prompt(&mut self, idx: usize);
    /// [`Emit::emit`] of `idx`.
    fn emit(&mut self, idx: usize, token: u32) -> Result<(Verdict, Vec<u32>), String>;
    /// [`Emit::history`] of `idx`.
    fn history(&self, idx: usize) -> &[u32];
    /// [`Emit::aborted`] of `idx`.
    fn aborted(&self, idx: usize) -> bool;
}

/// A request that left the scheduler: how, and where its state was.
#[derive(Debug)]
pub struct Ended {
    pub idx: usize,
    pub place: Place,
    pub exit: Exit,
    /// Completion tokens it counted.
    pub generated: usize,
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
}

#[derive(Default)]
pub struct Scheduler {
    reqs: Vec<Option<Req>>,
    /// Admitted requests waiting for the resident state, oldest first.
    queue: VecDeque<usize>,
    resident: Option<usize>,
    /// A finished request still holds the resident state (its terminal is
    /// not settled): [`Self::release_resident`].
    held: bool,
    /// Round-robin start of the next lane epoch.
    cursor: usize,
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
        });
        self.queue.push_back(idx);
    }

    /// The finished resident request's terminal settled: the resident state
    /// takes the next request.
    pub fn release_resident(&mut self) {
        self.held = false;
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
            self.held = matches!(exit, Exit::Finished(_));
        }
        self.queue.retain(|&queued| queued != idx);
        Ended {
            idx,
            place: req.place,
            exit,
            // A token streamed ahead of its lane row counts too.
            generated: req.run.as_ref().map_or(0, RequestRun::generated)
                + usize::from(req.ahead.is_some()),
        }
    }

    /// One step of every live request; returns the requests that ended.
    pub fn tick(&mut self, b: &mut impl Batch) -> Vec<Ended> {
        let mut ended = Vec::new();
        let live: Vec<usize> = (0..self.reqs.len())
            .filter(|&idx| self.reqs[idx].is_some())
            .collect();
        for idx in live {
            if b.aborted(idx) {
                ended.push(self.end(idx, Exit::Aborted));
            }
        }
        // A request that left the resident state for a lane hands it to the
        // next in line within the tick: queued prompts that each finish in
        // one epoch all reach this tick's lane epoch (their first tokens
        // share it), a longer prompt takes one epoch per tick. Aborts are
        // polled again before every epoch inside the tick, as `drive` does.
        while self.start(b, &mut ended) && self.resident_epoch(b, &mut ended) {}
        self.lane_epoch(b, &mut ended);
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
    /// next token is still on the device). Whether the resident state is
    /// free for the next request (it moved or was aborted).
    fn resident_epoch(&mut self, b: &mut impl Batch, ended: &mut Vec<Ended>) -> bool {
        let Some(idx) = self.resident else {
            return false;
        };
        if b.aborted(idx) {
            ended.push(self.end(idx, Exit::Aborted));
            return true;
        }
        let lanes = self.lanes().count();
        let others = lanes > 0 || !self.queue.is_empty();
        let parks = others && lanes < b.lane_capacity();
        let run = self.run_mut(idx);
        // A parking request whose next token is host-known moves without an
        // epoch; every other resident request runs one.
        if !(parks && matches!(run.next_work(), Work::Decode { token: Some(_) })) {
            let mut one = One {
                b,
                idx,
                keep: !parks,
            };
            if let Err(error) = step(run, &mut one) {
                ended.push(self.end(idx, Exit::Failed(error)));
                return false;
            }
        }
        let run = self.run_mut(idx);
        let seed = match run.next_work() {
            Work::Done(finish) => {
                ended.push(self.end(idx, Exit::Finished(finish)));
                return false;
            }
            Work::Decode { token: Some(seed) } if parks => seed,
            _ => return false,
        };
        let remaining = run.remaining();
        if let Err(error) = b.park(idx, seed, remaining) {
            ended.push(self.end(idx, Exit::Failed(error)));
            return false;
        }
        self.resident = None;
        let req = self.reqs[idx].as_mut().expect("parked request is live");
        req.place = Place::Lane;
        req.fresh = true;
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
            let run = req.run.as_mut().expect("a lane request has its run");
            let Work::Decode { token: Some(seed) } = run.next_work() else {
                continue;
            };
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

    /// One decode epoch over at most `lane_rows` decoding lanes, round-robin.
    fn lane_epoch(&mut self, b: &mut impl Batch, ended: &mut Vec<Ended>) {
        self.stream_fresh(b, ended);
        let lanes: Vec<usize> = self.lanes().collect();
        if lanes.is_empty() {
            return;
        }
        let take = lanes.len().min(b.lane_rows().max(1));
        let mut rows = Vec::with_capacity(take);
        for j in 0..take {
            let idx = lanes[(self.cursor + j) % lanes.len()];
            let run = self.run_mut(idx);
            match run.next_work() {
                Work::Decode { token: Some(token) } => rows.push((idx, token, !run.greedy())),
                work => {
                    let error = format!("a lane request has no host token to run: {work:?}");
                    ended.push(self.end(idx, Exit::Failed(error)));
                }
            }
        }
        self.cursor = (self.cursor + take) % lanes.len();
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
}

/// The resident request as one request's [`Forwards`] and [`Emit`], so its
/// epochs run the batch of one's [`step`].
struct One<'a, B> {
    b: &'a mut B,
    idx: usize,
    keep: bool,
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
    }

    #[derive(Default)]
    struct Fake {
        prompts: HashMap<usize, Vec<u32>>,
        budget: usize,
        eos: u32,
        capacity: usize,
        rows: usize,
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
        calls: Vec<Call>,
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
            Ok(RequestRun::new(
                len,
                0,
                self.budget,
                sampler,
                7 + idx as u32,
            ))
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

        fn commit_prompt(&mut self, idx: usize) {
            self.history.insert(idx, self.prompts[&idx].clone());
        }

        fn emit(&mut self, idx: usize, token: u32) -> Result<(Verdict, Vec<u32>), String> {
            self.calls.push(Call::Emit(idx));
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
    }

    fn fake(prompts: &[Vec<u32>], capacity: usize, rows: usize) -> Fake {
        Fake {
            prompts: prompts.iter().cloned().enumerate().collect(),
            budget: 12,
            eos: 0,
            capacity,
            rows,
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
                if e.place == Place::Resident && matches!(e.exit, Exit::Finished(_)) {
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
    fn an_abort_latched_mid_tick_stops_the_request_before_its_next_epoch() {
        let prompts = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let mut b = fake(&prompts, 4, 4);
        b.eos = u32::MAX;
        // 1 is aborted while 0 prefills; 2 is aborted while 1 would prefill.
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
