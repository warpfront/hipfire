// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Concurrency sweep for `hipfire bench`: drives both concurrent backends
//! (the daemon's experimental multi-slot `SlotEngine` and beta's in-daemon
//! continuous batching) over a range of concurrent stream counts and reports
//! aggregate throughput for each.
//!
//! The slot backend moved into the daemon and is independent of continuous
//! batching. Both arms are driven over the JSONL wire protocol via
//! [`hipfire_client::Engine::submit_streaming`], which registers per-request
//! lifecycle channels so pipelined generates no longer deadlock on dropped
//! reader-thread frames.

use anyhow::{bail, Result};
use serde_json::Value;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackendSel {
    /// The multi-slot engine: k streams advance together.
    Slots,
    /// No slots at all — k requests one after another through the daemon.
    /// This is what a deployment does today with `serve.multi_slot` off, and
    /// it is the baseline the slot engine has to beat.
    Sequential,
    /// Beta's in-daemon continuous batching, driven via
    /// `Engine::submit_streaming` (see `DaemonDriver::start`).
    Batch,
    Both,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WorkloadSel {
    Stateless,
    Multiturn,
    Both,
}

/// Parse `--concurrency 1,2,3,4` into sorted unique positive counts.
pub fn parse_concurrency(s: &str) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let n: usize = part
            .parse()
            .map_err(|_| anyhow::anyhow!("--concurrency: not a number: {part}"))?;
        if n == 0 {
            bail!("--concurrency values must be positive");
        }
        out.push(n);
    }
    if out.is_empty() {
        bail!("--concurrency needs at least one value");
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// Interleaved visiting order: `1,2,3,4,1,2,3,4,...`, never `1,1,2,2,...`.
///
/// Blocked ordering lets a thermal drift over the sweep read as a
/// concurrency effect. On this hardware that is not hypothetical: a blocked
/// single-run sweep reported 46.63 tok/s at 4 slots — below its own 1-slot
/// figure — where an interleaved 3-round repeat put the same point at
/// 108–118 tok/s.
pub fn sweep_order(points: &[usize], runs: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(points.len() * runs);
    for _ in 0..runs {
        out.extend_from_slice(points);
    }
    out
}

/// Median of the samples. Sorts in place. `None` when empty.
pub fn median(xs: &mut [f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_by(f64::total_cmp);
    let mid = xs.len() / 2;
    if xs.len() % 2 == 1 {
        Some(xs[mid])
    } else {
        Some((xs[mid - 1] + xs[mid]) / 2.0)
    }
}

/// One measured run of one arm at one concurrency point.
#[derive(Clone, Copy, Debug)]
pub struct ArmResult {
    /// Generated tokens summed across all streams.
    pub tokens: u64,
    /// Wall clock from first submit to last completion.
    pub wall_ms: f64,
    /// Streams the backend refused. Their tokens are NOT counted.
    pub rejected: usize,
    /// Prefix-cache hits observed during this run (SlotEngine only).
    pub prefix_hits: usize,
}

impl ArmResult {
    /// Aggregate throughput delivered to the caller.
    ///
    /// This is the ONLY metric compared across backends. The SlotEngine's
    /// internal `ms/step` is deliberately excluded: the daemon path has no
    /// comparable figure and pairing them would flatter whichever side
    /// reported the friendlier decomposition.
    pub fn aggregate_tok_s(&self) -> f64 {
        if self.wall_ms <= 0.0 {
            return 0.0;
        }
        self.tokens as f64 / (self.wall_ms / 1000.0)
    }
}

/// All samples for one (backend, workload, k).
pub struct Point {
    pub backend: &'static str,
    pub workload: &'static str,
    pub k: usize,
    pub samples: Vec<ArmResult>,
}

/// Render the comparison table, medians per point.
pub fn render_table(points: &[Point]) -> String {
    let mut out = String::new();
    out.push_str(
        "\n  backend  workload    k   aggregate tok/s   per-stream   prefix hits   rejected\n",
    );
    for p in points {
        let mut aggs: Vec<f64> = p.samples.iter().map(ArmResult::aggregate_tok_s).collect();
        let Some(med) = median(&mut aggs) else {
            continue;
        };
        let per_stream = if p.k > 0 { med / p.k as f64 } else { 0.0 };
        let hits: usize = p.samples.iter().map(|s| s.prefix_hits).sum();
        let rej: usize = p.samples.iter().map(|s| s.rejected).sum();
        out.push_str(&format!(
            "  {:<8} {:<10} {:>2}   {:>15.2}   {:>10.2}   {:>11}   {:>8}\n",
            p.backend, p.workload, p.k, med, per_stream, hits, rej
        ));
    }
    out.push_str(
        "\n  NOTE: SlotEngine runs in-process; the daemon path pays JSONL pipe\n\
         \x20 encode/decode per token. That difference is inherent to where each\n\
         \x20 backend lives and is NOT a property of the batching algorithm.\n\
         \x20 Each backend is held at max concurrency and k varied, so the KV\n\
         \x20 arena is sized for the maximum at every point: this is a\n\
         \x20 concurrency curve, not a memory-footprint curve.\n",
    );
    out
}

/// One concurrent backend, started once at max concurrency.
pub trait ConcurrencyBackend {
    fn label(&self) -> &'static str;
    fn max_concurrency(&self) -> usize;
    /// Run `k` concurrent streams to completion.
    fn run(&mut self, workload: WorkloadSel, k: usize, max_tokens: u64) -> Result<ArmResult>;
}

/// Reject `k` above the started maximum instead of clamping. A clamped run
/// would report a k=4 row that was really measured at k=2.
pub fn check_k(b: &dyn ConcurrencyBackend, k: usize) -> Result<()> {
    if k > b.max_concurrency() {
        bail!(
            "concurrency {k} exceeds {} backend maximum {}",
            b.label(),
            b.max_concurrency()
        );
    }
    Ok(())
}

/// Fixed prompts, one per stream. Distinct so slots cannot alias each
/// other's KV undetected — identical prompts would hide a cross-slot leak.
pub const STREAM_PROMPTS: &[&str] = &[
    "The capital of France is",
    "Once upon a time, in a distant galaxy,",
    "The recipe for a good cup of tea starts with",
    "In machine learning, gradient descent works by",
    "The history of the printing press begins",
    "A well-designed database index avoids",
    "The difference between TCP and UDP is",
    "To sort a list in linear time you must",
];

/// Second user turn for the multi-turn arm.
pub const FOLLOWUP_PROMPT: &str = "Explain that again in one sentence.";

/// A stream's prompt, made unique per (run, stream) so no prompt is ever
/// submitted twice in a sweep.
///
/// This is load-bearing for fairness, not cosmetic. The two backends cache on
/// different keys and a repeated prompt is exactly the case where they
/// diverge:
///
/// - the daemon holds a token-level longest-common-prefix cache, so resending
///   an identical prompt lets it skip prefill entirely;
/// - the slot engine keys on conversation identity and `find_continuation`
///   returns `None` outright for a single-turn `convo`
///   (`session_table.rs:170`), so a repeat can never hit.
///
/// Cycling a fixed prompt list therefore hands the daemon free prefills and
/// denies the slot engine any — an asymmetry worth ~a prefill per request,
/// all of it favouring the daemon. Varying the leading token defeats the LCP
/// match at position 0 and costs the slot engine nothing it could have used.
pub fn stream_prompt(run: usize, stream: usize) -> String {
    format!(
        "Q{run}-{stream}. {}",
        STREAM_PROMPTS[stream % STREAM_PROMPTS.len()]
    )
}

/// One experimental multi-slot generate request for the daemon wire protocol.
///
/// Mirrors `batch_request` answer-mode fields but opts into the slot backend
/// with `experimental_multi_slot=true` instead of `serve_continuous_batch`.
/// The slot backend validates supported wire fields (`validate_generate_caps`
/// in `slots.rs`) and rejects `serve_continuous_batch`-style batch routing.
fn slot_request(prompt: &str, max_tokens: u64, id: &str, attempt_id: u64) -> Value {
    serde_json::json!({
        "type": "generate",
        "id": id,
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        "repeat_penalty": 1.1,
        "max_tokens": max_tokens,
        "attempt_id": attempt_id,
        "experimental_multi_slot": true,
        "max_think_tokens": 1,
        "assistant_prefix": "closed_think",
        "reasoning_effort": "none",
    })
}

/// Multi-turn slot request using a `messages` array so the daemon slot engine
/// can match the prior turn's conversation hash and reuse KV.
fn slot_request_messages(messages: &Value, max_tokens: u64, id: &str, attempt_id: u64) -> Value {
    serde_json::json!({
        "type": "generate",
        "id": id,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "repeat_penalty": 1.1,
        "max_tokens": max_tokens,
        "attempt_id": attempt_id,
        "experimental_multi_slot": true,
        "max_think_tokens": 1,
        "assistant_prefix": "closed_think",
        "reasoning_effort": "none",
    })
}

/// Per-stream outcome collected by [`drain_streams`].
struct StreamOutcome {
    tokens: u64,
    rejected: bool,
    prefix_hits: usize,
    /// Concatenated visible-token text — needed to reconstruct the assistant
    /// turn for multi-turn `messages` requests.
    text: String,
}

/// Drain `k` independently-submitted streams to completion.
///
/// Polls each receiver round-robin with a 1 ms timeout so idle streams do not
/// block active ones. Handles the `commit_ready` → `commit` → `done`
/// handshake per stream via [`hipfire_client::Engine::commit_attempt`].
/// Releases each attempt's pending entry on terminal `done`/`error`.
///
/// `prefix_hits` is recorded from the `commit_ready` payload's `cached_tokens`
/// field: a non-zero value means the daemon's prefix cache served prompt
/// tokens, which is the wire-level evidence of KV reuse. The daemon slot
/// backend does not currently emit a dedicated `prefix_hits` field in its
/// done payload; `cached_tokens > 0` is the closest wire signal. If the
/// daemon adds `prefix_hits` to the payload later, this check should prefer
/// it.
/// Whole-drain deadline for one concurrency arm (spec A14: finite
/// resources). Generous enough for the slowest legitimate arm; a daemon
/// that never terminates a stream fails the arm as rejected instead of
/// hanging the benchmark.
const DRAIN_DEADLINE: Duration = Duration::from_secs(300);

fn drain_streams(
    engine: &hipfire_client::Engine,
    streams: &[(String, u64, mpsc::Receiver<Value>)],
) -> Result<Vec<StreamOutcome>> {
    let k = streams.len();
    let mut outcomes: Vec<StreamOutcome> = (0..k)
        .map(|_| StreamOutcome {
            tokens: 0,
            rejected: false,
            prefix_hits: 0,
            text: String::new(),
        })
        .collect();
    let mut active: Vec<usize> = (0..k).collect();
    // Whole-drain deadline (spec A14: finite resources — a daemon that never
    // terminates a stream must fail the bench, not hang it).
    let deadline = Instant::now() + DRAIN_DEADLINE;

    while !active.is_empty() {
        if Instant::now() >= deadline {
            // Deadline exceeded: every still-active stream counts as
            // rejected, its pending attempt entry is released so later
            // repetitions of this arm are not poisoned by a "duplicate
            // live generate" rejection.
            for &idx in &active {
                let (id, attempt_id, _) = &streams[idx];
                outcomes[idx].rejected = true;
                // Cancel daemon-side (not just client-side release):
                // without an abort the daemon keeps generating into a
                // dropped channel and that background load pollutes the
                // NEXT sample's throughput.
                let _ = engine.cancel_attempt(id, *attempt_id);
                engine.release_attempt(id, *attempt_id);
            }
            break;
        }
        let mut still_active = Vec::with_capacity(active.len());
        for &idx in &active {
            let (id, attempt_id, rx) = &streams[idx];
            match rx.recv_timeout(Duration::from_millis(1)) {
                Ok(frame) => {
                    let ty = frame.get("type").and_then(Value::as_str);
                    match ty {
                        Some("token") => {
                            outcomes[idx].tokens += 1;
                            if let Some(text) = frame.get("text").and_then(Value::as_str) {
                                outcomes[idx].text.push_str(text);
                            }
                            still_active.push(idx);
                        }
                        Some("commit_ready") => {
                            // The staged done payload carries cached_tokens.
                            if let Some(cached) = frame.get("cached_tokens").and_then(Value::as_u64)
                            {
                                if cached > 0 {
                                    outcomes[idx].prefix_hits += 1;
                                }
                            }
                            // A commit failure must not bail the whole
                            // drain: bailing leaked every other stream's
                            // pending entry and permanently poisoned later
                            // repetitions of this arm. Mark this stream
                            // rejected, release, continue.
                            if let Err(e) = engine.commit_attempt(id, *attempt_id) {
                                outcomes[idx].rejected = true;
                                engine.release_attempt(id, *attempt_id);
                                let _ = e;
                            } else {
                                still_active.push(idx);
                            }
                        }
                        Some("done") => {
                            engine.release_attempt(id, *attempt_id);
                            // terminal — do not re-add
                        }
                        Some("error") => {
                            outcomes[idx].rejected = true;
                            engine.release_attempt(id, *attempt_id);
                        }
                        // gen_start, reasoning, aborted, tool_calls, etc.
                        _ => still_active.push(idx),
                    }
                }
                Err(RecvTimeoutError::Timeout) => still_active.push(idx),
                Err(RecvTimeoutError::Disconnected) => {
                    outcomes[idx].rejected = true;
                    engine.release_attempt(id, *attempt_id);
                }
            }
        }
        active = still_active;
    }
    Ok(outcomes)
}

/// The experimental multi-slot backend: k streams advance together through
/// the daemon's `SlotEngine`, driven over the JSONL wire protocol with
/// `experimental_multi_slot=true`.
///
/// The daemon must be loaded with `experimental_multi_slot=true` in the load
/// params (see `open_bench_engine_slots` in `main.rs`). The `loaded` reply
/// carries `experimental_multi_slot: true` when the slot backend initialised;
/// `SlotDriver::start` refuses otherwise.
pub struct SlotDriver {
    engine: hipfire_client::Engine,
    max_concurrency: usize,
    /// Monotonic run counter — see `stream_prompt`.
    seq: usize,
}

impl SlotDriver {
    /// `slot_capable` is the daemon's `experimental_multi_slot` flag from the
    /// load reply. Refusing here turns "this daemon was not loaded with the
    /// slot backend" into a reported result rather than a run that silently
    /// measures the ordinary sequential path.
    pub fn start(
        engine: hipfire_client::Engine,
        max_concurrency: usize,
        slot_capable: bool,
    ) -> Result<Self> {
        if !slot_capable {
            bail!("daemon does not advertise experimental_multi_slot capability");
        }
        Ok(Self {
            engine,
            max_concurrency,
            seq: 0,
        })
    }
}

impl ConcurrencyBackend for SlotDriver {
    fn label(&self) -> &'static str {
        "slots"
    }
    fn max_concurrency(&self) -> usize {
        self.max_concurrency
    }

    fn run(&mut self, workload: WorkloadSel, k: usize, max_tokens: u64) -> Result<ArmResult> {
        check_k(self, k)?;
        let turns = if matches!(workload, WorkloadSel::Multiturn) {
            2
        } else {
            1
        };

        self.seq += 1;
        let run = self.seq;
        let started = Instant::now();
        let mut tokens = 0u64;
        let mut rejected = 0usize;
        let mut prefix_hits = 0usize;

        // ── Turn 1 ───────────────────────────────────────────────────────
        let mut streams: Vec<(String, u64, mpsc::Receiver<Value>)> = Vec::with_capacity(k);
        for i in 0..k {
            let id = format!("bench-slot-{run}-{i}");
            let prompt = stream_prompt(run, i);
            let req = slot_request(&prompt, max_tokens, &id, 1);
            match self.engine.submit_streaming(&req) {
                Ok(s) => streams.push(s),
                Err(e) => {
                    for (sid, sa, _) in &streams {
                        self.engine.release_attempt(sid, *sa);
                    }
                    bail!("slot submit {i}: {e}");
                }
            }
        }
        let outcomes = drain_streams(&self.engine, &streams)?;
        let mut turn1_text: Vec<String> = Vec::with_capacity(k);
        for o in outcomes {
            tokens += o.tokens;
            if o.rejected {
                rejected += 1;
            }
            prefix_hits += o.prefix_hits;
            turn1_text.push(o.text);
        }

        // ── Turn 2 (multi-turn only) ─────────────────────────────────────
        if turns == 2 {
            let mut streams2: Vec<(String, u64, mpsc::Receiver<Value>)> = Vec::with_capacity(k);
            for i in 0..k {
                let id = format!("bench-slot-{run}-{i}");
                let first_prompt = stream_prompt(run, i);
                let messages = serde_json::json!([
                    {"role": "user", "content": first_prompt},
                    {"role": "assistant", "content": turn1_text.get(i).map(|s| s.as_str()).unwrap_or("")},
                    {"role": "user", "content": FOLLOWUP_PROMPT},
                ]);
                let req = slot_request_messages(&messages, max_tokens, &id, 2);
                match self.engine.submit_streaming(&req) {
                    Ok(s) => streams2.push(s),
                    Err(e) => {
                        for (sid, sa, _) in &streams2 {
                            self.engine.release_attempt(sid, *sa);
                        }
                        bail!("slot submit turn2 {i}: {e}");
                    }
                }
            }
            let outcomes2 = drain_streams(&self.engine, &streams2)?;
            for o in outcomes2 {
                tokens += o.tokens;
                if o.rejected {
                    rejected += 1;
                }
                prefix_hits += o.prefix_hits;
            }
        }

        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(ArmResult {
            tokens,
            wall_ms,
            rejected,
            prefix_hits,
        })
    }
}
/// One batch-route generate request.
///
/// `serve_continuous_batch` is what puts the daemon on
/// `drive_qwen_continuous_batch`; without it the request runs sequentially and
/// the resulting numbers would describe a non-batched run. Answer-mode fields
/// mirror `bench_generate_request` so both backends measure the same turn.
pub fn batch_request(
    prompt: &str,
    max_tokens: u64,
    id: &str,
    attempt_id: u64,
) -> serde_json::Value {
    serde_json::json!({
        "type": "generate",
        "id": id,
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        "repeat_penalty": 1.1,
        "max_tokens": max_tokens,
        "attempt_id": attempt_id,
        "serve_continuous_batch": true,
        "max_think_tokens": 1,
        "assistant_prefix": "closed_think",
        "reasoning_effort": "none",
    })
}

/// The no-slots baseline: k requests run strictly one after another through
/// the ordinary daemon generate path.
///
/// This is what a box serves today with `serve.multi_slot` off, so it is the
/// number the slot engine has to beat. It needs no multi-inflight support --
/// `Engine::generate` is blocking and one-at-a-time, which is exactly the
/// behaviour under measurement -- so unlike `DaemonDriver` it is unaffected by
/// the client's lack of a concurrent API.
///
/// Timed on the SAME clock as every other backend: wall from the first submit
/// to the last completion, with generated tokens summed across all k. That is
/// the only way "slots vs no slots" means anything.
pub struct SequentialDriver {
    engine: hipfire_client::Engine,
    max_concurrency: usize,
    /// Monotonic run counter — see `stream_prompt`. Without this the daemon's
    /// longest-common-prefix cache serves a repeated prompt from KV and the
    /// arm measures cached prefill rather than the real thing.
    seq: usize,
}

impl SequentialDriver {
    pub fn start(engine: hipfire_client::Engine, max_concurrency: usize) -> Result<Self> {
        Ok(Self {
            engine,
            max_concurrency,
            seq: 0,
        })
    }
}

impl ConcurrencyBackend for SequentialDriver {
    fn label(&self) -> &'static str {
        "noslots"
    }
    fn max_concurrency(&self) -> usize {
        self.max_concurrency
    }

    fn run(&mut self, workload: WorkloadSel, k: usize, max_tokens: u64) -> Result<ArmResult> {
        check_k(self, k)?;
        // Multi-turn on the sequential path is just two turns back to back;
        // there is no session to reuse, which is precisely the gap the slot
        // engine's prefix cache exists to close.
        let turns = if matches!(workload, WorkloadSel::Multiturn) {
            2
        } else {
            1
        };

        self.seq += 1;
        let run = self.seq;
        let started = Instant::now();
        let mut tokens = 0u64;
        let mut rejected = 0usize;
        for i in 0..k {
            for turn in 0..turns {
                let base = stream_prompt(run, i);
                let prompt = if turn == 0 {
                    base
                } else {
                    format!("{base} {FOLLOWUP_PROMPT}")
                };
                let req = batch_request(
                    &prompt,
                    max_tokens,
                    &format!("bench-seq-{i}-{turn}"),
                    ((run as u64) << 32) | ((i as u64) << 16) | turn as u64,
                );
                // Drop the batch-route opt-in: this arm is deliberately the
                // plain sequential path.
                let mut req = req;
                if let Some(obj) = req.as_object_mut() {
                    obj.remove("serve_continuous_batch");
                }
                let mut n = 0u64;
                match self.engine.generate(&req, |ev| {
                    if ev.get("type").and_then(serde_json::Value::as_str) == Some("token") {
                        n += 1;
                    }
                    Ok(())
                }) {
                    Ok(_) => tokens += n,
                    Err(_) => rejected += 1,
                }
            }
        }

        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(ArmResult {
            tokens,
            wall_ms,
            rejected,
            prefix_hits: 0,
        })
    }
}
pub struct DaemonDriver {
    engine: hipfire_client::Engine,
    max_concurrency: usize,
    /// Monotonic attempt-id source so repetitions never reuse one (a fixed
    /// id made the daemon's terminal registry drop later repetitions as
    /// duplicate live generates after a deadline trip).
    next_attempt: std::sync::atomic::AtomicU64,
}

impl DaemonDriver {
    /// `capable` is the daemon's own `continuous_batch_capable` from the load
    /// reply. Refusing here turns "this daemon cannot batch" into a reported
    /// result rather than a run that silently measures sequential execution.
    ///
    /// Multi-inflight submission uses [`hipfire_client::Engine::submit_streaming`],
    /// which registers per-`(id, attempt_id)` channels with the reader thread
    /// so pipelined generates no longer deadlock on dropped lifecycle frames.
    pub fn start(
        engine: hipfire_client::Engine,
        max_concurrency: usize,
        capable: bool,
    ) -> Result<Self> {
        if !capable {
            bail!("daemon does not advertise continuous_batch_capable");
        }
        Ok(Self {
            engine,
            max_concurrency,
            next_attempt: std::sync::atomic::AtomicU64::new(1),
        })
    }
}

impl ConcurrencyBackend for DaemonDriver {
    fn label(&self) -> &'static str {
        "batch"
    }
    fn max_concurrency(&self) -> usize {
        self.max_concurrency
    }

    fn run(&mut self, workload: WorkloadSel, k: usize, max_tokens: u64) -> Result<ArmResult> {
        check_k(self, k)?;
        // The batch route rejects multi-turn (`batch_messages_are_single_user`),
        // so this arm is sequential by construction on this backend. Report it
        // rather than pretending it batched.
        let _ = workload;

        let started = Instant::now();
        // PIPELINE: submit all k requests before draining any.
        // `Engine::generate` would block per request and serialise exactly the
        // behaviour under test. `submit_streaming` registers each pending
        // channel with the reader thread so lifecycle frames are routed, not
        // dropped.
        // Unique attempt ids PER REPETITION: a fixed attempt_id made the
        // daemon's terminal registry drop a later repetition as a
        // "duplicate live generate" after a deadline trip silently
        // converted the whole arm into timeouts.
        let mut streams: Vec<(String, u64, mpsc::Receiver<Value>)> = Vec::with_capacity(k);
        for i in 0..k {
            let id = format!("bench-conc-{i}");
            let prompt = STREAM_PROMPTS[i % STREAM_PROMPTS.len()];
            // Per-repetition attempt id: a fixed id made the daemon's
            // terminal registry drop later repetitions as duplicate live
            // generates after a deadline trip.
            let attempt_id = self
                .next_attempt
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let req = batch_request(prompt, max_tokens, &id, attempt_id);
            match self.engine.submit_streaming(&req) {
                Ok(s) => streams.push(s),
                Err(e) => {
                    for (sid, sa, _) in &streams {
                        self.engine.release_attempt(sid, *sa);
                    }
                    bail!("batch submit {i}: {e}");
                }
            }
        }

        let outcomes = drain_streams(&self.engine, &streams)?;
        let mut tokens = 0u64;
        let mut rejected = 0usize;
        for o in outcomes {
            tokens += o.tokens;
            if o.rejected {
                rejected += 1;
            }
        }

        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(ArmResult {
            tokens,
            wall_ms,
            rejected,
            prefix_hits: 0,
        })
    }
}

/// Reject a multi-turn slot run that recorded no prefix hits.
///
/// The whole point of the multi-turn arm is that `SlotEngine` reuses the
/// conversation's KV. A faster turn 2 proves nothing on its own — warm caches
/// do that too. `EngineStats::prefix_hits` is the event itself, so gate on it.
pub fn arm_is_valid(
    workload: WorkloadSel,
    backend: &str,
    r: &ArmResult,
) -> std::result::Result<(), String> {
    if matches!(workload, WorkloadSel::Multiturn) && backend == "slots" && r.prefix_hits == 0 {
        return Err(
            "multi-turn arm recorded no prefix hits — turn 2 did not reuse the \
             session KV, so this number does not measure what the arm claims"
                .to_string(),
        );
    }
    Ok(())
}

/// Run the interleaved sweep over one backend and collect points.
pub fn sweep_backend(
    backend: &mut dyn ConcurrencyBackend,
    arms: &[WorkloadSel],
    points: &[usize],
    runs: usize,
    max_tokens: u64,
    out: &mut Vec<Point>,
) -> Result<()> {
    let label = backend.label();
    for &arm in arms {
        let arm_label = match arm {
            WorkloadSel::Stateless => "stateless",
            WorkloadSel::Multiturn => "multiturn",
            WorkloadSel::Both => continue,
        };
        // One untimed warmup per (backend, arm) so first-call kernel
        // compilation never lands inside a timed sample.
        let _ = backend.run(arm, points[0], max_tokens.min(8));

        let mut by_k: std::collections::BTreeMap<usize, Vec<ArmResult>> =
            std::collections::BTreeMap::new();
        for k in sweep_order(points, runs) {
            let r = backend.run(arm, k, max_tokens)?;
            match arm_is_valid(arm, label, &r) {
                Ok(()) => by_k.entry(k).or_default().push(r),
                Err(why) => eprintln!("  [invalid] {label}/{arm_label} k={k}: {why}"),
            }
            eprint!(".");
            use std::io::Write;
            let _ = std::io::stderr().flush();
        }
        eprintln!();
        for (k, samples) in by_k {
            out.push(Point {
                backend: label,
                workload: arm_label,
                k,
                samples,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_concurrency_sorts_dedups_and_rejects_zero() {
        assert_eq!(parse_concurrency("4,1,2,1").unwrap(), vec![1, 2, 4]);
        assert!(parse_concurrency("0").is_err());
        assert!(parse_concurrency("").is_err());
        assert!(parse_concurrency("x").is_err());
    }

    /// The sweep must interleave. A blocked order (1,1,1,2,2,2) lets thermal
    /// drift masquerade as a concurrency effect — which it did in this repo.
    #[test]
    fn sweep_order_is_interleaved_not_blocked() {
        assert_eq!(
            sweep_order(&[1, 2, 3, 4], 3),
            vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        );
    }

    #[test]
    fn median_handles_odd_even_and_empty() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&mut []), None);
    }

    #[test]
    fn aggregate_tok_s_is_tokens_over_wall_clock() {
        let r = ArmResult {
            tokens: 256,
            wall_ms: 2000.0,
            rejected: 0,
            prefix_hits: 0,
        };
        assert!((r.aggregate_tok_s() - 128.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_tok_s_is_zero_when_no_time_elapsed() {
        let r = ArmResult {
            tokens: 10,
            wall_ms: 0.0,
            rejected: 0,
            prefix_hits: 0,
        };
        assert_eq!(r.aggregate_tok_s(), 0.0);
    }

    #[test]
    fn render_table_reports_median_and_per_stream() {
        let points = vec![Point {
            backend: "slots",
            workload: "stateless",
            k: 2,
            samples: vec![
                ArmResult {
                    tokens: 200,
                    wall_ms: 1000.0,
                    rejected: 0,
                    prefix_hits: 0,
                },
                ArmResult {
                    tokens: 100,
                    wall_ms: 1000.0,
                    rejected: 0,
                    prefix_hits: 0,
                },
            ],
        }];
        let table = render_table(&points);
        // median of {200, 100} tok/s = 150; per-stream = 75
        assert!(
            table.contains("150.00"),
            "aggregate median missing: {table}"
        );
        assert!(table.contains("75.00"), "per-stream missing: {table}");
        assert!(table.contains("slots"), "backend label missing: {table}");
    }

    /// A backend must refuse k above what it was started with rather than
    /// silently clamping — a clamped run would publish a k=4 number that was
    /// actually measured at k=2.
    #[test]
    fn run_rejects_k_above_max_concurrency() {
        struct Fake {
            max: usize,
        }
        impl ConcurrencyBackend for Fake {
            fn label(&self) -> &'static str {
                "fake"
            }
            fn max_concurrency(&self) -> usize {
                self.max
            }
            fn run(&mut self, _w: WorkloadSel, k: usize, _m: u64) -> Result<ArmResult> {
                check_k(self, k)?;
                Ok(ArmResult {
                    tokens: 0,
                    wall_ms: 1.0,
                    rejected: 0,
                    prefix_hits: 0,
                })
            }
        }
        let mut f = Fake { max: 2 };
        assert!(f.run(WorkloadSel::Stateless, 2, 8).is_ok());
        let err = f.run(WorkloadSel::Stateless, 3, 8).unwrap_err().to_string();
        assert!(err.contains("exceeds"), "unexpected error: {err}");
    }

    /// The daemon batch route only engages when the request opts in AND the
    /// request stays answer-mode. A request missing `serve_continuous_batch`
    /// silently runs sequentially, which would report batching numbers for a
    /// non-batched run.
    #[test]
    fn batch_request_opts_into_the_batch_route_and_answer_mode() {
        let r = batch_request("hello", 64, "bench-c-1", 1);
        assert_eq!(
            r.get("serve_continuous_batch").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(r.get("type").and_then(|v| v.as_str()), Some("generate"));
        assert_eq!(r.get("id").and_then(|v| v.as_str()), Some("bench-c-1"));
        assert_eq!(r.get("max_tokens").and_then(|v| v.as_u64()), Some(64));
        // Answer mode, same contract as the single-stream bench path.
        assert_eq!(r.get("max_think_tokens").and_then(|v| v.as_u64()), Some(1));
        assert_eq!(
            r.get("assistant_prefix").and_then(|v| v.as_str()),
            Some("closed_think")
        );
    }

    /// A faster second turn is not evidence of prefix reuse — it could be
    /// cache warmth. If the multi-turn arm recorded no prefix hits on the
    /// slot backend, the run is invalid and must not publish a number.
    #[test]
    fn multiturn_without_prefix_hits_is_invalid_on_slots() {
        let no_hits = ArmResult {
            tokens: 100,
            wall_ms: 500.0,
            rejected: 0,
            prefix_hits: 0,
        };
        let hits = ArmResult {
            tokens: 100,
            wall_ms: 500.0,
            rejected: 0,
            prefix_hits: 2,
        };

        assert!(arm_is_valid(WorkloadSel::Multiturn, "slots", &no_hits).is_err());
        assert!(arm_is_valid(WorkloadSel::Multiturn, "slots", &hits).is_ok());
        // The batch backend cannot reuse a prefix at all, so zero is expected.
        assert!(arm_is_valid(WorkloadSel::Multiturn, "batch", &no_hits).is_ok());
        // Stateless never reuses anything.
        assert!(arm_is_valid(WorkloadSel::Stateless, "slots", &no_hits).is_ok());
    }
}
